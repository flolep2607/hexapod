//! Headless driver for the hexapod core: train a policy, benchmark the
//! simulator, or check how a learned policy transfers to courses it never saw.
//!
//! ```text
//! hexapod train  [--course mixed] [--iters 200] [--seed 1] [--preset tripod]
//! hexapod oneleg [--moves 6] [--leg L1] [--seed 1]
//! hexapod watch  [--course flat] [--seconds 8] [--speed 1.5]
//! hexapod bench  [--course mixed]
//! hexapod sweep  [--iters 150]
//! hexapod speed  [--iters 200]      commanded vs achieved speed
//! hexapod jump   [--iters 200]      parkour: distance, waypoints, jumps
//! hexapod servo                     the same gait on every servo
//! ```
//!
//! `oneleg` is the empty-field drill: five legs hold their world plants
//! (friction only, nothing welded to the floor) while one foot lifts
//! and plants a random reachable spot in its workspace.
//!
//! `watch` runs the Rapier plant and prints pose, 3-axis velocity, heading,
//! stance-foot slip and range/bearing to the next waypoint — numbers you can
//! read when a canvas walk is too small or too icy to judge by eye.
//!
//! `--course` takes any of `flat steps rubble gaps mixed ramps slalom slick
//! gauntlet jump`, matched case-insensitively against the names the simulator
//! defines; `hexapod courses` prints them as JSON for the web build.
//!
//! `--leg-mass G` overrides the swinging mass of one leg in grams; `0` makes
//! the legs weightless, which is what the simulator assumed before it had a
//! leg-inertia model.
//!
//! `--legs N` sets the frame: any even count from 4 to 10. Four legs start on
//! the crawl rather than the alternating gait, because a trot stands on two
//! diagonal feet and this simulator judges stability statically.
//!
//! `--servo NAME` picks the actuator the simulator drives its joints with, and
//! `--mass` / `--scale` the machine it is driving. They change what the
//! optimiser converges to, not just what the bill of materials says.

use std::time::Instant;

use hexapod_core::ars::ArsConfig;
use hexapod_core::hardware::{shortlist, Build, Provenance, TorqueMeter, PRICES_CHECKED, SERVOS};
use hexapod_core::sim::Sim;
use hexapod_core::power::{parts_of, solve, Kind, Sizing, TorqueTrace};
use hexapod_core::policy::Preset;
use hexapod_core::sim::{
    evaluate, rollout, Cmd, CRUISE_MAX, CRUISE_MIN, DT, JUMP_CRUISE_MAX, JUMP_CRUISE_MIN,
    JUMP_EVAL_SPEEDS,
};
use hexapod_core::oneleg::{OneLegDrill, Phase};
use hexapod_core::walker::WalkSample;
use hexapod_core::{Course, Frame, Physics, Policy, Terrain, Trainer};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("train");

    let course = parse_course(flag(&args, "--course").unwrap_or_else(|| "mixed".into()));
    let iters: usize = flag(&args, "--iters")
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let seed: u64 = flag(&args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let frame = Frame::new(
        flag(&args, "--legs")
            .and_then(|v| v.parse().ok())
            .unwrap_or(6),
    );
    let preset = match flag(&args, "--preset") {
        Some(v) => match v.as_str() {
            "ripple" => Preset::Ripple,
            "wave" => Preset::Wave,
            "tripod" | "alternate" | "trot" => Preset::Tripod,
            other => {
                eprintln!("unknown preset {other:?}; try tripod, ripple or wave");
                std::process::exit(2)
            }
        },
        // Six legs and up start on the alternating gait; four have to crawl,
        // because a trot is not statically stable and this simulator judges
        // stability statically.
        None => Preset::default_for(frame),
    };

    let build = build_from(&args);
    let servo = flag(&args, "--servo").and_then(|name| {
        SERVOS
            .iter()
            .find(|s| s.part.eq_ignore_ascii_case(&name))
            .or_else(|| {
                eprintln!("unknown servo {name:?}; known: {}", servo_names());
                std::process::exit(2)
            })
    });
    let mut phys = build.physics(servo);
    // Per-leg swinging mass, grams. `0` makes the legs weightless, which is
    // what the simulator assumed before there was a leg-inertia model at all.
    if let Some(g) = flag(&args, "--leg-mass").and_then(|v| v.parse::<f64>().ok()) {
        phys.leg = if g <= 0.0 {
            hexapod_core::LegMass::WEIGHTLESS
        } else {
            let total = g / 1000.0;
            hexapod_core::LegMass {
                femur_kg: total * 0.556,
                tibia_kg: total * 0.444,
            }
        };
    }

    let mut cfg = ArsConfig::default();
    if let Some(v) = flag(&args, "--dirs").and_then(|v| v.parse().ok()) {
        cfg.n_dirs = v;
    }
    if let Some(v) = flag(&args, "--top").and_then(|v| v.parse().ok()) {
        cfg.n_top = v;
    }
    if let Some(v) = flag(&args, "--alpha").and_then(|v| v.parse().ok()) {
        cfg.alpha = v;
    }
    if let Some(v) = flag(&args, "--sigma").and_then(|v| v.parse().ok()) {
        cfg.sigma = v;
    }
    if let Some(v) = flag(&args, "--horizon").and_then(|v| v.parse().ok()) {
        cfg.horizon = v;
    }

    match cmd {
        "bench" => bench(frame, course, seed, phys),
        "bom" => bom(frame, course, seed, iters, cfg, phys, build),
        "sweep" => sweep(frame, iters, cfg, phys, seed),
        "speed" => speed(frame, course, seed, iters, cfg, phys),
        "jump" => jump(frame, seed, iters, cfg, phys),
        "servo" => servo_shootout(frame, course, seed, iters, cfg, build),
        "servos" => servos_json(),
        "parts" => parts_json(),
        "courses" => courses_json(),
        "system" => system(frame, course, seed, iters, cfg, phys, &args),
        "oneleg" | "reach" => oneleg(frame, seed, phys, &args),
        "scene" | "scenes" => scenes(frame, phys, &args),
        "watch" => {
            let course = if flag(&args, "--course").is_some() {
                course
            } else {
                Course::Flat
            };
            watch(frame, course, seed, preset, phys, &args)
        }
        _ => train(frame, course, seed, iters, preset, cfg, phys),
    }
}

fn servo_names() -> String {
    SERVOS
        .iter()
        .map(|s| s.part)
        .collect::<Vec<_>>()
        .join(", ")
}

fn build_from(args: &[String]) -> Build {
    let mut build = Build::default();
    if let Some(v) = flag(args, "--mass").and_then(|v| v.parse().ok()) {
        build.mass_kg = v;
    }
    if let Some(v) = flag(args, "--scale").and_then(|v| v.parse().ok()) {
        build.scale = v;
    }
    build
}

fn flag(args: &[String], name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == name)?;
    args.get(i + 1).cloned()
}

fn parse_course(s: String) -> Course {
    hexapod_core::terrain::COURSES
        .iter()
        .copied()
        .find(|c| c.name().eq_ignore_ascii_case(&s))
        .unwrap_or(Course::Mixed)
}

fn train(
    frame: Frame,
    course: Course,
    seed: u64,
    iters: usize,
    preset: Preset,
    cfg: ArsConfig,
    phys: Physics,
) {
    let terrain = Terrain::new(course, seed);
    let mut t = Trainer::new(Policy::seeded(preset, frame), cfg, phys, seed ^ 0xA5A5);

    println!(
        "machine: {} on {} legs | {:.2} kg, scale {:.3}, servo {:.1} kg-cm at {:.0} rpm no-load, mu {:.2}",
        frame.label(),
        frame.legs(),
        phys.mass_kg,
        phys.scale,
        phys.actuator.stall_nm * hexapod_core::hardware::NM_TO_KGCM,
        phys.actuator.omega_max * 60.0 / std::f64::consts::TAU,
        phys.mu
    );
    println!(
        "legs: {:.0} g of swinging mass each, {:.0} g in total ({:.0}% of the machine)",
        phys.leg.total() * 1000.0,
        phys.swing_mass(frame) * 1000.0,
        phys.swing_mass(frame) / phys.mass_kg * 100.0
    );
    if course.is_jump() {
        println!(
            "commanded speeds sampled from {JUMP_CRUISE_MIN:.1} to {JUMP_CRUISE_MAX:.1} m/s; \
             scored on the mean over 4.0 / 5.0 / 5.5 — a running jump, not a walk"
        );
    } else {
        println!(
            "commanded speeds sampled from {CRUISE_MIN:.1} to {CRUISE_MAX:.1} m/s; \
             scored on the mean over 2.0 / 4.0 / 5.5"
        );
    }
    println!(
        "course {} seed {}  |  ARS dirs={} top={} alpha={} sigma={} horizon={}s",
        course.name(),
        seed,
        cfg.n_dirs,
        cfg.n_top,
        cfg.alpha,
        cfg.sigma,
        cfg.horizon
    );
    println!(
        "obstacles: {}  |  route: {} waypoints, corridor +-{:.1} m between two walls",
        terrain.obstacles.len(),
        terrain.waypoints.len(),
        terrain.wall_x()
    );

    let start = Instant::now();
    t.record_baseline(&terrain);
    println!(
        "\nbaseline  reward {:8.2}   distance {:6.2} m   {}",
        t.baseline_reward,
        t.baseline_distance,
        if t.last_eval.fell { "FELL" } else { "survived" }
    );
    println!(
        "\n{:>5}  {:>9}  {:>9}  {:>8}  {:>8}  {:>7}  {:>6}  {:>7}",
        "iter", "reward", "best", "v err", "slip m", "stub", "CoT", "fell"
    );

    for i in 1..=iters {
        t.iterate(&terrain);
        if i % (iters / 20).max(1) == 0 || i == iters {
            let e = t.last_eval;
            println!(
                "{i:>5}  {:>9.2}  {:>9.2}  {:>8.2}  {:>8.2}  {:>7.2}  {:>6.2}  {:>7}",
                e.reward,
                t.best_reward,
                e.speed_error,
                e.slip,
                e.stub_total,
                e.cot,
                if e.fell { "yes" } else { "" }
            );
        }
    }

    let secs = start.elapsed().as_secs_f64();
    let best = t.best_policy();
    let g = best.gait();

    println!("\n--- learned gait ---");
    println!("cycle time    {:6.3} s", g.cycle);
    println!("stride        {:6.3} m", g.stride);
    println!("step height   {:6.3} m", g.step_h);
    println!("body height   {:6.3} m", g.body_h);
    println!("stance width  {:6.3} m", g.stance_w);
    println!("duty          {:6.3}", g.duty);
    print!("phase offsets ");
    for (i, o) in g.offsets.iter().enumerate() {
        print!("{}={:.3} ", frame.name(i), o);
    }
    println!("\nfeedback norm {:6.3}", best.feedback_norm());

    let gain = if t.baseline_reward.abs() > 1e-6 {
        (t.best_reward - t.baseline_reward) / t.baseline_reward.abs() * 100.0
    } else {
        0.0
    };
    println!("\n--- result ---");
    println!(
        "reward   {:8.2}  ->  {:8.2}   ({gain:+.0}%)",
        t.baseline_reward, t.best_reward
    );
    println!(
        "distance {:8.2}  ->  {:8.2} m",
        t.baseline_distance, t.best_distance
    );
    println!(
        "{} rollouts in {:.1}s  ({:.0} rollouts/s)",
        t.rollouts,
        secs,
        t.rollouts as f64 / secs
    );
    println!("\n{}", sparkline(&t.curve));

    println!("\n--- speed tracking ---");
    speed_table(&terrain, &Policy::seeded(preset, frame), &best, &phys, cfg.horizon);
}

/// Commanded speed against achieved speed, for the baseline and the learned
/// policy. This is the whole point of the reward: the hand-tuned gait has one
/// speed it can walk at, and the learned one has a range.
fn speed_table(
    terrain: &Terrain,
    base: &Policy,
    learned: &Policy,
    phys: &Physics,
    horizon: f64,
) {
    println!(
        "{:>10} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8} {:>7}",
        "commanded", "base m/s", "base err", "got m/s", "err", "cycle", "stride", "duty"
    );
    let mut sum_b = 0.0;
    let mut sum_l = 0.0;
    let speeds = [2.0, 2.75, 3.5, 4.25, 5.0, 5.75];
    for &v in speeds.iter() {
        let a = rollout(terrain, base, phys, horizon, Cmd::at(v), None);
        let b = rollout(terrain, learned, phys, horizon, Cmd::at(v), None);
        let av = a.distance / (a.steps as f64 * DT);
        let bv = b.distance / (b.steps as f64 * DT);
        sum_b += a.speed_error;
        sum_l += b.speed_error;
        println!(
            "{v:>9.2}  {av:>9.2} {:>9.2} {bv:>9.2} {:>9.2} {:>8.3} {:>8.3} {:>7.3}",
            a.speed_error, b.speed_error, b.mean_cycle, b.mean_stride, b.mean_duty
        );
    }
    let n = speeds.len() as f64;
    println!("{:>10} {:>19.2} {:>19.2}", "mean err", sum_b / n, sum_l / n);
    println!(
        "\ncycle, stride and duty are the learned policy's *online* values,\n\
         averaged over the rollout. A speed-conditioned policy moves them."
    );
}

fn bench(frame: Frame, course: Course, seed: u64, phys: Physics) {
    let terrain = Terrain::new(course, seed);
    let p = Policy::seeded(Preset::default_for(frame), frame);

    let n = 400;
    let start = Instant::now();
    let mut steps = 0usize;
    for _ in 0..n {
        steps += rollout(&terrain, &p, &phys, 8.0, Cmd::at(4.0), None).steps;
    }
    let secs = start.elapsed().as_secs_f64();

    println!("course {}", course.name());
    println!("{n} rollouts, {steps} steps in {secs:.2}s");
    println!("{:.0} steps/s", steps as f64 / secs);
    println!("{:.2} us/step", secs / steps as f64 * 1e6);
    println!(
        "{:.1} simulated seconds per wall second",
        steps as f64 * DT / secs
    );
}

/// Rapier plant, printed as numbers: pose, 3-axis velocity, heading, slip,
/// range and bearing to the next waypoint.
fn watch(
    frame: Frame,
    course: Course,
    seed: u64,
    _preset: Preset,
    phys: Physics,
    args: &[String],
) {
    let seconds: f64 = flag(args, "--seconds")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8.0);
    let every: f64 = f64::max(
        flag(args, "--every")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.10),
        DT,
    );
    let speed: f64 = flag(args, "--speed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.5);
    let nav = args.iter().any(|a| a == "--nav");

    let terrain = Terrain::new(course, seed);
    let mut drill = OneLegDrill::spawn_on(frame, &phys, &terrain, seed, true);
    let cmd = Cmd {
        fwd: 1.0,
        turn: 0.0,
        cruise: speed,
        nav,
    };
    drill.set_cmd(cmd);

    println!(
        "# hexapod watch  {} on {}  course={} seed={}  cmd={:.2} m/s  nav={}  crawl",
        frame.label(),
        frame.legs(),
        course.name(),
        seed,
        speed,
        if nav { "on" } else { "off" },
    );
    println!(
        "# t      x      y      z     yaw   pit   rol     vx     vy     vz   |v|  along   slip  \
         dψ   wp_d   brg  wp  reach stance"
    );

    let ticks = (seconds / DT).round() as usize;
    let emit_every = (every / DT).round().max(1.0) as usize;
    let mut samples: Vec<WalkSample> = Vec::new();
    let wall = Instant::now();

    for k in 0..=ticks {
        if k > 0 {
            drill.set_cmd(cmd);
            drill.step(DT);
        }
        let fallen = drill.sample().fallen;
        if k % emit_every == 0 || k == ticks || fallen {
            let s = watch_sample(&drill, terrain.waypoints.len());
            print_watch_row(&s);
            samples.push(s);
            if fallen {
                println!("# FALLEN at t={:.2}s", s.t);
                break;
            }
        }
    }

    let elapsed = wall.elapsed().as_secs_f64();
    print_watch_summary(&samples, speed, elapsed);
}

fn watch_sample(drill: &OneLegDrill, wp_n: usize) -> WalkSample {
    let s = drill.sample();
    let n = drill.frame.legs();
    let mut stance = [false; hexapod_core::MAX_LEGS];
    for i in 0..n {
        stance[i] = !(i == s.moving && s.phase.swinging());
    }
    let (hs, hc) = s.yaw.sin_cos();
    let along = s.vel[0] * (-hs) + s.vel[2] * hc;
    let speed = (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt();
    WalkSample {
        t: s.t,
        pos: s.pos,
        yaw: s.yaw,
        pitch: s.pitch,
        roll: s.roll,
        vel: s.vel,
        speed,
        along,
        slip: s.slip,
        yaw_rate: 0.0,
        heading_deg: s.yaw.to_degrees(),
        wp: 0,
        wp_n,
        wp_dist: 0.0,
        bearing: 0.0,
        bearing_deg: 0.0,
        reached: 0,
        cmd_speed: 0.0,
        n_legs: n,
        stance,
        fallen: s.fallen,
    }
}

fn print_watch_row(s: &WalkSample) {
    println!(
        "{:5.2} {:+6.3} {:+6.3} {:+6.3} {:+6.1} {:+5.1} {:+5.1} {:+6.3} {:+6.3} {:+6.3} \
         {:5.2} {:+6.3} {:5.3} {:+5.2} {:6.2} {:+5.1} {:3} {:5} {}",
        s.t,
        s.pos[0],
        s.pos[1],
        s.pos[2],
        s.heading_deg,
        s.pitch.to_degrees(),
        s.roll.to_degrees(),
        s.vel[0],
        s.vel[1],
        s.vel[2],
        s.speed,
        s.along,
        s.slip,
        s.yaw_rate,
        s.wp_dist,
        s.bearing_deg,
        s.wp + 1,
        s.reached,
        s.stance_bits()
    );
}

fn print_watch_summary(samples: &[WalkSample], cmd: f64, wall_s: f64) {
    let Some(first) = samples.first() else {
        return;
    };
    let last = *samples.last().unwrap();
    let n = samples.len().max(1) as f64;
    let mean = |f: fn(&WalkSample) -> f64| samples.iter().map(f).sum::<f64>() / n;
    let peak_along = samples
        .iter()
        .map(|s| s.along)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = samples.iter().map(|s| s.pos[1]).fold(f64::INFINITY, f64::min);
    let alongs: Vec<f32> = samples.iter().map(|s| s.along as f32).collect();

    println!();
    println!("--- {:.2} s ({:.2} s wall) ---", last.t, wall_s);
    println!(
        "pose         x={:+.3}  y={:.3}  z={:+.3}   heading {:+.1}°  pitch {:+.1}°  roll {:+.1}°",
        last.pos[0],
        last.pos[1],
        last.pos[2],
        last.heading_deg,
        last.pitch.to_degrees(),
        last.roll.to_degrees()
    );
    println!(
        "progress     Δz={:+.3} m  Δx={:+.3} m  heading drift {:+.1}°",
        last.pos[2] - first.pos[2],
        last.pos[0] - first.pos[0],
        last.heading_deg - first.heading_deg
    );
    println!(
        "velocity     mean |v|={:.3} m/s  mean along-heading={:+.3}  peak along={:+.3}  cmd={:.2}",
        mean(|s| s.speed),
        mean(|s| s.along),
        peak_along,
        cmd
    );
    println!(
        "axis         mean vx={:+.3}  vy={:+.3}  vz={:+.3}   end vx={:+.3} vy={:+.3} vz={:+.3}",
        mean(|s| s.vel[0]),
        mean(|s| s.vel[1]),
        mean(|s| s.vel[2]),
        last.vel[0],
        last.vel[1],
        last.vel[2]
    );
    println!(
        "slip         mean {:.3} m/s  end {:.3} m/s   (stance-foot rubber vs floor)",
        mean(|s| s.slip),
        last.slip
    );
    println!(
        "waypoint     {}/{}  dist={:.2} m  bearing {:+.1}°  reached {}",
        last.wp + 1,
        last.wp_n.max(1),
        last.wp_dist,
        last.bearing_deg,
        last.reached
    );
    println!(
        "height       min {:.3}  end {:.3}{}",
        min_y,
        last.pos[1],
        if last.fallen { "  FALLEN" } else { "" }
    );
    println!("along spark  {}", sparkline(&alongs));
}

/// Empty field: five legs hold their standing setpoints, one foot relocates
/// inside its reachable workspace. Nothing is welded to the floor.
fn oneleg(frame: Frame, seed: u64, phys: Physics, args: &[String]) {
    let moves: usize = flag(args, "--moves")
        .and_then(|v| v.parse().ok())
        .unwrap_or(frame.legs());
    let every: f64 = f64::max(
        flag(args, "--every")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.10),
        DT,
    );
    let mut drill = OneLegDrill::spawn(frame, &phys, seed);
    if let Some(name) = flag(args, "--leg") {
        let idx = (0..frame.legs()).find(|&i| {
            frame.name(i).eq_ignore_ascii_case(&name) || name == i.to_string()
        });
        match idx {
            Some(i) => drill.pin_leg(i),
            None => {
                eprintln!(
                    "unknown leg {name:?}; try {}",
                    (0..frame.legs())
                        .map(|i| frame.name(i))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                std::process::exit(2);
            }
        }
    }

    println!(
        "# hexapod oneleg  {} on {}  seed={}  moves={}  empty field, friction only",
        frame.label(),
        frame.legs(),
        seed,
        moves
    );
    println!(
        "# stance legs hold their world plants; the free foot lifts and \
         plants a random reachable spot in its workspace"
    );
    println!(
        "# t     phase  mv leg    cmdx   cmdy   cmdz    actx   acty   actz  \
         err  Δxz  drift travel   slip     vx     vy     vz  y    yaw"
    );

    let emit_every = (every / DT).round().max(1.0) as usize;
    let mut k = 0usize;
    let mut last_move = 0usize;
    let mut last_leg = 0usize;
    let mut last_phase = Phase::Settle;
    let wall = Instant::now();
    let mut summaries: Vec<(usize, &'static str, f64, f64, f64, f64)> = Vec::new();
    // per-move peaks: travel, stance_drift, chassis_xz, end reach_err
    let mut peak_travel = 0.0;
    let mut peak_drift = 0.0;
    let mut peak_xz = 0.0;

    loop {
        if k > 0 {
            drill.step(DT);
        }
        let s = drill.sample();
        if s.move_i != last_move {
            summaries.push((
                last_move,
                frame.name(last_leg),
                peak_travel,
                peak_drift,
                peak_xz,
                s.reach_err,
            ));
            peak_travel = 0.0;
            peak_drift = 0.0;
            peak_xz = 0.0;
            last_move = s.move_i;
        }
        last_leg = s.moving;
        peak_travel = peak_travel.max(s.moving_travel);
        peak_drift = peak_drift.max(s.stance_drift);
        peak_xz = peak_xz.max(s.chassis_xz);

        if k % emit_every == 0 || s.phase != last_phase || s.fallen {
            println!(
                "{:5.2} {:>6} {:3} {:<3} {:+6.3} {:+6.3} {:+6.3}  {:+6.3} {:+6.3} {:+6.3} \
                 {:5.3} {:5.3} {:5.3} {:6.3} {:6.3} {:+6.3} {:+6.3} {:+6.3} {:5.3} {:+5.1}{}",
                s.t,
                s.phase.name(),
                s.move_i,
                frame.name(s.moving),
                s.cmd_body[0],
                s.cmd_body[1],
                s.cmd_body[2],
                s.foot_body[0],
                s.foot_body[1],
                s.foot_body[2],
                s.reach_err,
                s.chassis_xz,
                s.stance_drift,
                s.moving_travel,
                s.slip,
                s.vel[0],
                s.vel[1],
                s.vel[2],
                s.pos[1],
                s.yaw.to_degrees(),
                if s.fallen { " FALLEN" } else { "" }
            );
        }
        last_phase = s.phase;
        if s.fallen {
            println!("# FALLEN at t={:.2}s", s.t);
            break;
        }
        if s.move_i >= moves && s.phase == Phase::Lift && s.phase_u < 0.05 && k > 10 {
            break;
        }
        k += 1;
        if k > (120.0 / DT) as usize {
            break;
        }
    }

    println!();
    println!(
        "--- {} moves in {:.2} s ({:.2} s wall) ---",
        summaries.len().max(last_move),
        drill.t,
        wall.elapsed().as_secs_f64()
    );
    println!(
        "{:<6} {:<4} {:>8} {:>10} {:>10}",
        "move", "leg", "travel", "stanceΔ", "chassisΔ"
    );
    for (i, name, travel, drift, xz, _) in &summaries {
        println!(
            "{:<6} {:<4} {:8.3} {:10.3} {:10.3}",
            i, name, travel, drift, xz
        );
    }
    let s = drill.sample();
    println!(
        "end  y={:.3}  yaw={:+.1}°  pitch={:+.1}°  |v|={:.3}  fallen={}",
        s.pos[1],
        s.yaw.to_degrees(),
        s.pitch.to_degrees(),
        (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt(),
        s.fallen
    );
}

/// Train on MIXED, then check the policy on courses it never trained on.
fn sweep(frame: Frame, iters: usize, cfg: ArsConfig, phys: Physics, seed: u64) {
    let train_terrain = Terrain::new(Course::Mixed, seed);
    let mut t = Trainer::new(Policy::seeded(Preset::default_for(frame), frame), cfg, phys, seed ^ 0xA5A5);
    t.record_baseline(&train_terrain);
    for _ in 0..iters {
        t.iterate(&train_terrain);
    }

    let base = Policy::seeded(Preset::default_for(frame), frame);
    let best = t.best_policy();

    println!("trained {iters} iterations on MIXED seed {seed}\n");
    println!(
        "{:<10} {:>10} {:>10} {:>11} {:>11} {:>9} {:>7} {:>9}",
        "course", "base rwd", "learn rwd", "base dist", "learn dist", "waypoints", "fell", "verdict"
    );

    let mut plan: Vec<(Course, u64)> = hexapod_core::terrain::COURSES
        .iter()
        .copied()
        .enumerate()
        .map(|(i, c)| (c, seed + 100 * i as u64))
        .collect();
    // MIXED is what it trained on, so run it on the training seed as well.
    plan.insert(4, (Course::Mixed, seed));
    for (course, cseed) in plan {
        let terr = Terrain::new(course, cseed);
        let a = evaluate(&terr, &base, &phys, cfg.horizon);
        let b = evaluate(&terr, &best, &phys, cfg.horizon);
        println!(
            "{:<10} {:>10.2} {:>10.2} {:>10.2}m {:>10.2}m {:>4} ->{:>4} {:>7} {:>9}",
            format!("{}{}", course.name(), if cseed == seed { "" } else { "*" }),
            a.reward,
            b.reward,
            a.distance,
            b.distance,
            a.reached,
            b.reached,
            match (a.fell, b.fell) {
                (true, true) => "both",
                (true, false) => "base",
                (false, true) => "learned",
                _ => "",
            },
            if b.reward > a.reward { "better" } else { "worse" }
        );
    }
    println!("\n* = course seed the policy never trained on");
    println!(
        "\"fell\" marks which side went over in at least one of the three\n\
         evaluation speeds. Reward is not distance: a policy can score better\n\
         by taking fewer penalties over less ground."
    );
}

/// Size the servos for a build, comparing the hand-tuned gait against a
/// trained one on the same machine.
fn bom(frame: Frame, course: Course, seed: u64, iters: usize, cfg: ArsConfig, phys: Physics, build: Build) {
    let terrain = Terrain::new(course, seed);
    let base = Policy::seeded(Preset::default_for(frame), frame);

    let mut t = Trainer::new(Policy::seeded(Preset::default_for(frame), frame), cfg, phys, seed ^ 0xA5A5);
    t.record_baseline(&terrain);
    for _ in 0..iters {
        t.iterate(&terrain);
    }
    let learned = t.best_policy();

    let measure = |p: &Policy| {
        let g = p.gait();
        let mut s = Sim::default();
        s.reset(&terrain, &g, &phys);
        let mut m = TorqueMeter::default();
        for _ in 0..(cfg.horizon / hexapod_core::DT) as usize {
            s.step(&terrain, p, &g, hexapod_core::DT, Cmd::at(4.0));
            m.observe(&s, &build);
            if s.fallen {
                break;
            }
        }
        m
    };

    let links = build.link_mm();
    println!(
        "build: {} ({} legs, {} joints), {:.1} kg, scale {:.3}",
        frame.label(),
        frame.legs(),
        frame.legs() * 3,
        build.mass_kg,
        build.scale
    );
    println!(
        "links: coxa {:.0} mm, femur {:.0} mm, tibia {:.0} mm",
        links[0], links[1], links[2]
    );
    println!(
        "allowances: dynamic x{:.2}, traction {:.0}%, safety x{:.2}\n",
        build.dynamic_factor,
        build.traction_ratio * 100.0,
        build.safety
    );

    println!(
        "{:<12} {:>9} {:>9} {:>9} {:>12} {:>12}",
        "gait", "coxa", "femur", "tibia", "peak foot N", "required"
    );
    let mut required = 0.0f64;
    for (name, p) in [("hand-tuned", &base), ("learned", &learned)] {
        let m = measure(p);
        let k = m.peak_kgcm();
        let req = m.required_kgcm(&build);
        if name == "learned" {
            required = req;
        }
        println!(
            "{name:<12} {:>8.2} {:>8.2} {:>8.2} {:>11.1} {:>10.1} kg-cm",
            k[0], k[1], k[2], m.peak_foot_load, req
        );
    }

    let gb = base.gait();
    let gl = learned.gait();
    println!(
        "\nstance width {:.2} -> {:.2} sim units ({:.0} -> {:.0} mm)",
        gb.stance_w,
        gl.stance_w,
        gb.stance_w * build.scale * 1000.0,
        gl.stance_w * build.scale * 1000.0
    );

    let joints = frame.legs() * 3;
    println!("\n--- servos clearing {required:.1} kg-cm, {joints} per robot ---");
    println!(
        "{:<11} {:<10} {:>7} {:>6} {:>8} {:>16} {:>14}",
        "part", "maker", "kg-cm", "head", "bus", "full set (USD)", "set mass"
    );
    for s in shortlist(required) {
        let (lo, hi) = s.build_cost(joints);
        println!(
            "{:<11} {:<10} {:>7.1} {:>5.2}x {:>8} {:>7.0} - {:<6.0} {:>11.2} kg",
            s.part,
            s.maker,
            s.stall_kgcm,
            s.headroom(required),
            s.bus.name(),
            lo,
            hi,
            s.set_mass_kg(joints)
        );
    }

    println!("\n--- excluded (insufficient torque) ---");
    for s in SERVOS.iter().filter(|s| !s.meets(required)) {
        println!(
            "{:<11} {:>7.1} kg-cm  ({:.2}x required)",
            s.part,
            s.stall_kgcm,
            s.headroom(required)
        );
    }

    println!("\nprices: vendor-checked {PRICES_CHECKED}; marketplace bands indicative.");
    for s in SERVOS.iter().filter(|s| s.provenance == Provenance::Vendor) {
        println!(
            "  {:<10} ${:>6.2} at {:<14} {}",
            s.part,
            s.vendor_usd.unwrap_or(0.0),
            s.vendor_name,
            s.source
        );
    }
}

/// Train once, then report commanded speed against achieved speed.
///
/// This is the answer to the question the old reward could not ask. The
/// earlier version rewarded raw distance, so the optimiser pinned cycle time
/// and stride to their bounds and ran flat out; the version after that hard-
/// coded a single 4 m/s cruise, which merely moved the specialisation. Here
/// the command is an input, sampled per rollout and fed to the policy, so a
/// gait that only works at one speed cannot score.
fn speed(frame: Frame, course: Course, seed: u64, iters: usize, cfg: ArsConfig, phys: Physics) {
    let terrain = Terrain::new(course, seed);
    let base = Policy::seeded(Preset::default_for(frame), frame);
    let mut t = Trainer::new(Policy::seeded(Preset::default_for(frame), frame), cfg, phys, seed ^ 0xA5A5);
    t.record_baseline(&terrain);
    for _ in 0..iters {
        t.iterate(&terrain);
    }
    let learned = t.best_policy();

    println!(
        "{} seed {}, {iters} iterations, commanded speeds sampled from {CRUISE_MIN:.1} to {CRUISE_MAX:.1} m/s\n",
        course.name(),
        seed
    );
    speed_table(&terrain, &base, &learned, &phys, cfg.horizon);

    let gb = base.gait();
    let gl = learned.gait();
    println!(
        "\nnominal gait speed (stride / duty / cycle): {:.2} -> {:.2} m/s",
        gb.nominal_speed(),
        gl.nominal_speed()
    );
    println!(
        "the learned policy also modulates cycle and stride online, which is\n\
         what lets one gait cover the range instead of one speed."
    );
}

/// Train on JUMP, then print distance, waypoints and jumps. The analogue of
/// `hexapod speed`, for a course you cannot walk.
fn jump(frame: Frame, seed: u64, iters: usize, cfg: ArsConfig, phys: Physics) {
    let terrain = Terrain::new(Course::Jump, seed);
    let base = Policy::seeded(Preset::default_for(frame), frame);
    let mut t = Trainer::new(
        Policy::seeded(Preset::default_for(frame), frame),
        cfg,
        phys,
        seed ^ 0xA5A5,
    );
    t.record_baseline(&terrain);
    for _ in 0..iters {
        t.iterate(&terrain);
    }
    let learned = t.best_policy();

    println!(
        "JUMP seed {}, {iters} iterations, commanded speeds sampled from {JUMP_CRUISE_MIN:.1} to {JUMP_CRUISE_MAX:.1} m/s\n",
        seed
    );
    println!(
        "{:>10} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7}",
        "commanded", "base m", "base wp", "got m", "got wp", "jumps", "g", "broke"
    );
    for &v in JUMP_EVAL_SPEEDS.iter() {
        let a = rollout(&terrain, &base, &phys, cfg.horizon, Cmd::at(v), None);
        let b = rollout(&terrain, &learned, &phys, cfg.horizon, Cmd::at(v), None);
        println!(
            "{v:>9.2}  {:>9.2} {:>9} {:>9.2} {:>9} {:>7} {:>7.1} {:>7}",
            a.distance,
            a.reached,
            b.distance,
            b.reached,
            b.jumps,
            b.impact_g,
            if b.broken { "BROKE" } else { "ok" }
        );
    }
    println!(
        "\nbaseline reward {:7.2}  learned {:7.2}  ({:+.0}%)",
        t.baseline_reward,
        t.best_reward,
        100.0 * (t.best_reward - t.baseline_reward) / t.baseline_reward.abs().max(1e-6)
    );
    println!(
        "baseline distance {:6.2} m  learned {:6.2} m",
        t.baseline_distance, t.best_distance
    );
}

/// The same course and the same learner, once per servo.
///
/// The servo is not a post-hoc sizing decision any more: its torque-speed line
/// drives the joints, so it changes what the optimiser converges to.
fn servo_shootout(frame: Frame, course: Course, seed: u64, iters: usize, cfg: ArsConfig, build: Build) {
    let terrain = Terrain::new(course, seed);
    println!(
        "{} seed {}, {iters} iterations per servo, {:.2} kg at scale {:.3}\n",
        course.name(),
        seed,
        build.mass_kg,
        build.scale
    );
    println!(
        "{:<11} {:>7} {:>8} {:>9} {:>9} {:>8} {:>7} {:>7} {:>6}",
        "servo", "kg-cm", "rpm", "base rwd", "best rwd", "peak/stall", "cycle", "duty", "CoT"
    );

    for servo in SERVOS.iter() {
        let phys = build.physics(Some(servo));
        let mut t = Trainer::new(Policy::seeded(Preset::default_for(frame), frame), cfg, phys, seed ^ 0xA5A5);
        t.record_baseline(&terrain);
        for _ in 0..iters {
            t.iterate(&terrain);
        }
        let best = t.best_policy();
        let g = best.gait();
        let e = evaluate(&terrain, &best, &phys, cfg.horizon);
        println!(
            "{:<11} {:>7.1} {:>8.0} {:>9.2} {:>9.2} {:>9.2}x {:>7.3} {:>7.3} {:>6.2}",
            servo.part,
            servo.stall_kgcm,
            phys.actuator.omega_max * 60.0 / std::f64::consts::TAU,
            t.baseline_reward,
            t.best_reward,
            e.peak_servo_load,
            g.cycle,
            g.duty,
            e.cot
        );
    }
    println!(
        "\npeak/stall above 1.00 means the servo was driven past its rating and\n\
         the leg gave way under load."
    );
}

/// Emit the servo catalogue as JSON. `build.sh` inlines this into the web
/// bundle so the browser and the simulator share one source of truth.
fn servos_json() {
    println!("{{");
    println!("  \"checked\": \"{}\",", PRICES_CHECKED);
    println!("  \"servos\": [");
    for (i, s) in SERVOS.iter().enumerate() {
        let comma = if i + 1 == SERVOS.len() { "" } else { "," };
        println!(
            "    {{\"part\":{:?},\"maker\":{:?},\"stall\":{},\"volts\":{},\"mass\":{},\
             \"speed\":{},\"bus\":{:?},\"metal\":{},\"feedback\":{},\"low\":{},\"high\":{},\
             \"vendor\":{},\"vendorName\":{:?},\"source\":{:?},\"checked\":{},\"note\":{:?}}}{}",
            s.part,
            s.maker,
            s.stall_kgcm,
            s.at_volts,
            s.mass_g,
            s.speed_s60,
            s.bus.name(),
            s.metal_gear,
            s.feedback,
            s.market_usd.0,
            s.market_usd.1,
            match s.vendor_usd {
                Some(v) => format!("{v}"),
                None => "null".into(),
            },
            s.vendor_name,
            s.source,
            s.provenance == Provenance::Vendor,
            s.note,
            comma
        );
    }
    println!("  ]");
    println!("}}");
}

/// Whole-machine sizing: current, battery, regulator, controller, sensors.
fn system(
    frame: Frame,
    course: Course,
    seed: u64,
    iters: usize,
    cfg: ArsConfig,
    phys: Physics,
    args: &[String],
) {
    let scale = phys.scale;
    let mut sizing = Sizing::default();
    if let Some(v) = flag(args, "--chassis").and_then(|v| v.parse().ok()) {
        sizing.chassis_kg = v;
    }
    if let Some(v) = flag(args, "--runtime").and_then(|v| v.parse().ok()) {
        sizing.runtime_min = v;
    }

    let terrain = Terrain::new(course, seed);
    let mut policy = Policy::seeded(Preset::default_for(frame), frame);
    let mut label = "hand-tuned";
    if iters > 0 {
        let mut t = Trainer::new(Policy::seeded(Preset::default_for(frame), frame), cfg, phys, seed ^ 0xA5A5);
        t.record_baseline(&terrain);
        for _ in 0..iters {
            t.iterate(&terrain);
        }
        policy = t.best_policy();
        label = "learned";
    }

    let trace = TorqueTrace::record(&terrain, &policy, &phys, 8.0);
    println!(
        "gait: {label} on {} | chassis {:.2} kg | femur {:.0} mm | target runtime {:.0} min\n",
        course.name(),
        sizing.chassis_kg,
        0.8 * scale * 1000.0,
        sizing.runtime_min
    );

    println!(
        "{:<10} {:>7} {:>8} {:>8} {:>7} {:>7} {:>8} {:>8}",
        "servo", "all-up", "needs", "battery", "mean", "peak", "runtime", "total"
    );
    println!(
        "{:<10} {:>7} {:>8} {:>8} {:>7} {:>7} {:>8} {:>8}",
        "", "kg", "kg-cm", "kg", "A", "A", "min", "USD"
    );

    let mut best: Option<(f64, &'static str)> = None;
    for servo in hexapod_core::SERVOS {
        let s = solve(&trace, servo, &sizing);
        let verdict = if !s.converged {
            s.failure
        } else if !s.servo_ok {
            "under-torqued"
        } else {
            ""
        };
        println!(
            "{:<10} {:>7.2} {:>8.1} {:>8.2} {:>7.2} {:>7.1} {:>8.0} {:>8.0}  {}",
            servo.part,
            s.all_up_kg,
            s.required_kgcm,
            s.battery_kg,
            s.mean_amps,
            s.peak_amps,
            s.runtime_min,
            s.cost_usd,
            verdict
        );
        if s.converged && s.servo_ok && best.is_none_or(|b| s.cost_usd < b.0) {
            best = Some((s.cost_usd, servo.part));
        }
    }

    let Some((_, pick)) = best else {
        println!("\nno servo in the catalogue can build this machine.");
        return;
    };
    let servo = hexapod_core::SERVOS.iter().find(|s| s.part == pick).unwrap();
    let s = solve(&trace, servo, &sizing);

    println!("\n=== cheapest viable build: {} ===\n", servo.part);
    println!("converged in {} iterations of the mass/current loop", s.iterations);
    println!("  chassis      {:>6.2} kg", s.chassis_kg);
    println!("  18x servo    {:>6.2} kg", s.servo_kg);
    println!("  battery      {:>6.2} kg", s.battery_kg);
    println!("  electronics  {:>6.2} kg", s.electronics_kg);
    println!("  ------------------------");
    println!("  all-up       {:>6.2} kg", s.all_up_kg);
    println!();
    println!(
        "  peak joint torque {:.1} kg-cm vs {:.1} kg-cm stall ({:.2}x)",
        s.peak_torque_kgcm,
        servo.stall_kgcm,
        servo.stall_kgcm / s.peak_torque_kgcm
    );
    println!(
        "  mean draw {:.2} A ({:.0} W), peak {:.1} A, {:.0} min endurance",
        s.mean_amps, s.mean_watts, s.peak_amps, s.runtime_min
    );

    println!("\n--- parts ---");
    let line = |kind: &str, name: &str, qty: usize, unit: f64, note: &str| {
        println!(
            "  {:<13} {:<30} x{:<3} ${:>7.2}   {}",
            kind,
            name,
            qty,
            unit * qty as f64,
            note
        );
    };
    line(
        "SERVO",
        servo.part,
        frame.legs() * 3,
        servo.unit_low(),
        servo.bus.name(),
    );
    if let Some(b) = s.battery {
        line("BATTERY", b.name, 1, b.unit_price(), b.note);
    }
    match s.regulator {
        Some(r) => line("REGULATOR", r.name, 1, r.unit_price(), r.note),
        None => println!(
            "  {:<13} {:<30} pack voltage matches the servo bus",
            "REGULATOR", "none"
        ),
    }
    if let Some(d) = s.driver {
        line("DRIVER", d.name, s.driver_count, d.unit_price(), d.note);
    }
    if let Some(c) = s.compute {
        line("COMPUTE", c.name, 1, c.unit_price(), c.note);
    }
    let n = s.sensing;
    if let Some(r) = parts_of(Kind::Ranger).next() {
        line("RANGEFINDER", r.name, n.rangers, r.unit_price(), "one per leg");
    }
    if let Some(m) = parts_of(Kind::Support).next() {
        line("SUPPORT", m.name, 1, m.unit_price(), m.note);
    }
    if let Some(i) = parts_of(Kind::Imu).next() {
        line("IMU", i.name, 1, i.unit_price(), "pitch and roll");
    }
    println!("  {:<13} {:<30}     ${:>7.2}", "", "TOTAL", s.cost_usd);

    println!("\n--- what the policy demands of the sensors ---");
    println!("  the observation vector is not free: each input is a measurement.");
    println!(
        "  6x terrain lookahead  ->  {:.2} m range, >= {:.0} Hz, {:.0} mm resolution",
        n.lookahead_m, n.min_rate_hz, n.resolution_mm
    );
    println!("  pitch, roll           ->  IMU at the control rate");
    println!(
        "  stability margin      ->  which feet are loaded: {}",
        if n.contact_from_bus {
            "free, the servo bus reports load"
        } else {
            "needs 6 contact switches or FSRs"
        }
    );
}

/// The course list, so the dashboard's buttons come from the same enum the
/// simulator switches on rather than a hand-kept copy of it.
fn courses_json() {
    let names: Vec<String> = hexapod_core::terrain::COURSES
        .iter()
        .map(|c| format!("\"{}\"", c.name()))
        .collect();
    println!("{{\"courses\":[{}]}}", names.join(","));
}

/// Emit the non-servo component catalogue as JSON for the web bundle.
fn parts_json() {
    use hexapod_core::power::PARTS;
    println!("{{ \"parts\": [");
    for (i, p) in PARTS.iter().enumerate() {
        let comma = if i + 1 == PARTS.len() { "" } else { "," };
        println!(
            "    {{\"name\":{:?},\"maker\":{:?},\"kind\":{:?},\"mass\":{},\"low\":{},\"high\":{},\
             \"vendor\":{},\"vendorName\":{:?},\"source\":{:?},\"note\":{:?},\
             \"accuracy\":{},\"volts\":{},\"capacity\":{},\"rating\":{},\"unit\":{}}}{}",
            p.name,
            p.maker,
            p.kind.name(),
            p.mass_g,
            p.market_usd.0,
            p.market_usd.1,
            match p.vendor_usd {
                Some(v) => format!("{v}"),
                None => "null".into(),
            },
            p.vendor_name,
            p.source,
            p.note,
            p.accuracy_mm,
            p.volts,
            p.capacity,
            p.rating,
            p.unit_price(),
            comma
        );
    }
    println!("  ]");
    println!("}}");
}

fn sparkline(v: &[f32]) -> String {
    if v.is_empty() {
        return String::new();
    }
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let lo = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let span = (hi - lo).max(1e-6);
    let step = (v.len() / 100).max(1);
    let mut s = String::new();
    for chunk in v.chunks(step) {
        let m = chunk.iter().sum::<f32>() / chunk.len() as f32;
        let i = (((m - lo) / span) * 7.0).round().clamp(0.0, 7.0) as usize;
        s.push(BARS[i]);
    }
    format!("{s}   [{lo:.1} .. {hi:.1}]")
}

// ------------------------------------------------------------------- scenes

/// One headless scenario with pass/fail numbers, so a locomotion change can be
/// judged from a terminal instead of a canvas. Every scene prints the same
/// columns; `--json` emits them for scripting.
///
///   hexapod scene            run them all
///   hexapod scene walk       run one
///   hexapod scene --json     machine readable
struct Report {
    name: &'static str,
    /// Worst joint tracking error over the run, radians. The god motor's own
    /// score: if this is not near zero the joints are not following the gait
    /// and every other number is measuring the wrong machine.
    track: f64,
    /// Lowest the chassis got, sim units. Ride height is 0.88.
    min_y: f64,
    /// Ground covered in the plane, sim units.
    travel: f64,
    /// Metres of stance-foot sliding summed over every planted foot.
    slip: f64,
    /// Distance to the waypoint at the end, or -1 when the scene has none.
    to_goal: f64,
    fell: bool,
    /// Simulated seconds per wall second.
    rt: f64,
}

impl Report {
    fn row(&self) -> String {
        format!(
            "{:<12} track {:6.3}  min_y {:6.3}  travel {:7.3}  slip {:8.2}  goal {:7.3}  {:>5}  {:6.1}x",
            self.name,
            self.track,
            self.min_y,
            self.travel,
            self.slip,
            self.to_goal,
            if self.fell { "FELL" } else { "ok" },
            self.rt
        )
    }
    fn json(&self) -> String {
        format!(
            "{{\"name\":\"{}\",\"track\":{:.4},\"min_y\":{:.4},\"travel\":{:.4},\"slip\":{:.3},\"to_goal\":{:.4},\"fell\":{},\"rt\":{:.2}}}",
            self.name, self.track, self.min_y, self.travel, self.slip, self.to_goal, self.fell, self.rt
        )
    }
}

fn scenes(frame: Frame, mut phys: Physics, args: &[String]) {
    if let Some(v) = flag(args, "--stiff").and_then(|v| v.parse().ok()) {
        phys.motor_stiff = v;
    }
    if let Some(v) = flag(args, "--damp").and_then(|v| v.parse().ok()) {
        phys.motor_damp = v;
    }
    if let Some(v) = flag(args, "--substeps").and_then(|v| v.parse().ok()) {
        phys.substeps = v;
    }
    if let Some(v) = flag(args, "--solver").and_then(|v| v.parse().ok()) {
        phys.solver_iters = v;
    }
    if let Some(v) = flag(args, "--maxf").and_then(|v| v.parse().ok()) {
        phys.motor_max = v;
    }
    // args[0] is the subcommand. A scene name is the next bare word that is
    // not the value of a flag.
    let mut want = String::new();
    let mut k = 1;
    while k < args.len() {
        if args[k].starts_with("--") {
            k += 2;
            continue;
        }
        want = args[k].clone();
        break;
    }
    let json = args.iter().any(|a| a == "--json");
    let secs: f64 = flag(args, "--seconds")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20.0);
    let mut rows = Vec::new();
    // The tripod path is a different controller from the crawl and breaks in
    // different ways, so it gets its own scenes rather than sharing a knob.
    for (name, course) in [("tripod", Course::Flat), ("tripod-rubble", Course::Rubble)] {
        if !want.is_empty() && want != name {
            continue;
        }
        rows.push(run_tripod(name, frame, phys, course, secs));
    }
    let all: [(&'static str, Course, f64, f64); 6] = [
        ("stand", Course::Flat, 0.0, 0.0),
        ("walk", Course::Flat, 1.0, 0.0),
        ("turn", Course::Flat, 0.0, 1.0),
        ("rubble", Course::Rubble, 1.0, 0.0),
        ("steps", Course::Steps, 1.0, 0.0),
        ("slalom", Course::Slalom, 1.0, 0.0),
    ];
    for (name, course, fwd, turn) in all {
        if !want.is_empty() && want != name {
            continue;
        }
        rows.push(run_scene(name, frame, phys, course, fwd, turn, secs));
    }
    if json {
        println!(
            "[{}]",
            rows.iter().map(|r| r.json()).collect::<Vec<_>>().join(",")
        );
    } else {
        println!("scene        joint-tracking  ride    ground     foot-slip   waypoint  state   speed");
        for r in &rows {
            println!("{}", r.row());
        }
    }
}

fn run_scene(
    name: &'static str,
    frame: Frame,
    phys: Physics,
    course: Course,
    fwd: f64,
    turn: f64,
    secs: f64,
) -> Report {
    use hexapod_core::sim::DT;
    let terrain = Terrain::new(course, 1);
    let mut drill = OneLegDrill::spawn_on(frame, &phys, &terrain, 1, true);
    drill.set_cmd(Cmd {
        fwd,
        turn,
        cruise: 0.35,
        nav: false,
    });
    let n = (secs / DT) as usize;
    let mut min_y = f64::INFINITY;
    let mut track: f64 = 0.0;
    let mut slip = 0.0f64;
    let mut fell = false;
    let mut prev: Option<[[f64; 3]; hexapod_core::robot::MAX_LEGS]> = None;
    let t0 = Instant::now();
    for _ in 0..n {
        drill.step(DT);
        let s = drill.sample();
        min_y = min_y.min(s.pos[1]);
        fell |= s.fallen;
        // Commanded joints versus where the joints actually are.
        let cmd = drill.cmd_q();
        for i in 0..frame.legs() {
            let at = drill.plant.leg_q(i);
            for j in 0..3 {
                track = track.max((at[j] - cmd[i][j]).abs());
            }
        }
        let mut feet = [[0.0; 3]; hexapod_core::robot::MAX_LEGS];
        for i in 0..frame.legs() {
            feet[i] = drill.plant.leg_joints_world(i)[3];
        }
        if let Some(p) = prev {
            for i in 0..frame.legs() {
                if i == s.moving && s.phase.swinging() {
                    continue;
                }
                let d = feet[i][0] - p[i][0];
                let e = feet[i][2] - p[i][2];
                slip += (d * d + e * e).sqrt();
            }
        }
        prev = Some(feet);
    }
    let rt = (n as f64 * DT) / t0.elapsed().as_secs_f64();
    let s = drill.sample();
    // Range to the waypoint the machine is currently aiming at.
    let w = terrain.waypoint(0);
    let (dx, dz) = (w[0] - s.pos[0], w[1] - s.pos[2]);
    let to_goal = (dx * dx + dz * dz).sqrt();
    Report {
        name,
        track,
        min_y,
        travel: s.chassis_xz,
        slip,
        to_goal,
        fell,
        rt,
    }
}

fn run_tripod(
    name: &'static str,
    frame: Frame,
    phys: Physics,
    course: Course,
    secs: f64,
) -> Report {
    use hexapod_core::sim::DT;
    let (mut walker, terrain, policy, gait) =
        hexapod_core::walker::open_loop_walk(frame, course, 1, phys);
    let cmd = Cmd {
        fwd: 1.0,
        turn: 0.0,
        cruise: 1.5,
        nav: false,
    };
    let n = (secs / DT) as usize;
    let mut min_y = f64::INFINITY;
    let mut fell = false;
    let t0 = Instant::now();
    for _ in 0..n {
        walker.step(&terrain, &policy, &gait, DT, cmd);
        let s = walker.sample();
        min_y = min_y.min(s.pos[1]);
        fell |= s.fallen;
    }
    let rt = (n as f64 * DT) / t0.elapsed().as_secs_f64();
    let s = walker.sample();
    let w = terrain.waypoint(0);
    let (dx, dz) = (w[0] - s.pos[0], w[1] - s.pos[2]);
    Report {
        name,
        track: 0.0,
        min_y,
        travel: (s.pos[0] * s.pos[0] + s.pos[2] * s.pos[2]).sqrt(),
        slip: 0.0,
        to_goal: (dx * dx + dz * dz).sqrt(),
        fell,
        rt,
    }
}
