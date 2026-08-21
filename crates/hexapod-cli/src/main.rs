//! Headless driver for the hexapod core: train a policy, benchmark the
//! simulator, or check how a learned policy transfers to courses it never saw.
//!
//! ```text
//! hexapod train  [--course mixed] [--iters 200] [--seed 1] [--preset tripod]
//! hexapod joint-train [--backend nexus-gpu] [--batch-envs 128] [--iters 400]
//! hexapod joint-eval --policy policy.txt [--backend nexus-gpu] [--stage mixed]
//! hexapod bench  [--course mixed]
//! hexapod sweep  [--iters 150]
//! hexapod train-all [--iters 200] [--train-seeds 1] [--eval-seeds 2]
//! hexapod eval-all --policy policy.txt [--eval-seeds 2]
//! hexapod speed  [--iters 200]      commanded vs achieved speed
//! hexapod jump   [--iters 200]      parkour: distance, waypoints, jumps
//! hexapod servo                     the same gait on every servo
//! ```
//!
//! `joint-train` is the joint-level trainer: the policy commands all eighteen
//! motors directly against batched Nexus/Rapier worlds, with no gait and no IK,
//! working up a curriculum from standing to parkour. It writes a `hexapod-joint-v1`
//! checkpoint, which is a different format from the gait policies above
//! because it drives a different thing. Native only; `--backend rapier` keeps a
//! portable CPU reference path.
//!
//! `--course` takes any of `flat steps rubble gaps mixed ramps slalom slick
//! gauntlet jump`, matched case-insensitively against the names the simulator
//! defines; `hexapod courses` prints them as JSON for the web build.
//!
//! `--leg-mass G` overrides the swinging mass of one leg in grams; `0` makes
//! the legs weightless, which is what the simulator assumed before it had a
//! leg-inertia model.
//!
//! `--legs N` sets the frame: any even count from 4 to 10.
//!
//! `--servo NAME` picks the actuator the simulator drives its joints with, and
//! `--mass` / `--scale` the machine it is driving. They change what the
//! optimiser converges to, not just what the bill of materials says.

use std::time::Instant;

use hexapod_core::ars::ArsConfig;
use hexapod_core::hardware::{Build, PRICES_CHECKED, Provenance, SERVOS, TorqueMeter, shortlist};
use hexapod_core::policy::Preset;
use hexapod_core::power::{Kind, Sizing, TorqueTrace, parts_of, solve};
use hexapod_core::sim::Sim;
use hexapod_core::sim::{
    CRUISE_MAX, CRUISE_MIN, Cmd, DT, JUMP_CRUISE_MAX, JUMP_CRUISE_MIN, JUMP_EVAL_SPEEDS, evaluate,
    rollout,
};
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
    if let Some(v) = flag(&args, "--workers").and_then(|v| v.parse().ok()) {
        cfg.workers = v;
    }
    if let Some(v) = flag(&args, "--batch").and_then(|v| v.parse().ok()) {
        cfg.scenarios_per_direction = v;
    }

    match cmd {
        "bench" => bench(frame, course, seed, phys),
        "bom" => bom(frame, course, seed, iters, cfg, phys, build),
        "sweep" => sweep(frame, iters, cfg, phys, seed),
        "joint-train" => joint_train(frame, phys, &args),
        "joint-eval" => joint_eval(course, phys, &args),
        "train-all" | "all-terrain" => all_terrain(frame, seed, iters, cfg, phys, &args),
        "eval-all" => eval_all(seed, cfg, phys, &args),
        "speed" => speed(frame, course, seed, iters, cfg, phys),
        "jump" => jump(frame, seed, iters, cfg, phys),
        "servo" => servo_shootout(
            frame,
            course,
            seed,
            iters,
            cfg,
            build,
            flag(&args, "--seeds")
                .and_then(|v| v.parse().ok())
                .unwrap_or(3usize)
                .max(1),
        ),
        "servos" => servos_json(),
        "parts" => parts_json(),
        "courses" => courses_json(),
        "system" => system(frame, course, seed, iters, cfg, phys, &args),
        _ => train(frame, course, seed, iters, preset, cfg, phys),
    }
}

fn servo_names() -> String {
    SERVOS.iter().map(|s| s.part).collect::<Vec<_>>().join(", ")
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
        let (lo, hi) = hexapod_core::sim::cruise_band(course);
        let e = hexapod_core::sim::eval_speeds(course);
        println!(
            "commanded speeds sampled from {lo:.1} to {hi:.1} m/s; \
             scored on the mean over {:.1} / {:.1} / {:.1} — a running jump, not a walk",
            e[0], e[1], e[2]
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
    speed_table(
        &terrain,
        &Policy::seeded(preset, frame),
        &best,
        &phys,
        cfg.horizon,
    );
}

/// Commanded speed against achieved speed, for the baseline and the learned
/// policy. This is the whole point of the reward: the hand-tuned gait has one
/// speed it can walk at, and the learned one has a range.
fn speed_table(terrain: &Terrain, base: &Policy, learned: &Policy, phys: &Physics, horizon: f64) {
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
fn sweep(frame: Frame, iters: usize, cfg: ArsConfig, phys: Physics, seed: u64) {
    let train_terrain = Terrain::new(Course::Mixed, seed);
    let mut t = Trainer::new(
        Policy::seeded(Preset::default_for(frame), frame),
        cfg,
        phys,
        seed ^ 0xA5A5,
    );
    t.record_baseline(&train_terrain);
    for _ in 0..iters {
        t.iterate(&train_terrain);
    }

    let base = Policy::seeded(Preset::default_for(frame), frame);
    let best = t.best_policy();

    println!("trained {iters} iterations on MIXED seed {seed}\n");
    println!(
        "{:<10} {:>10} {:>10} {:>11} {:>11} {:>9} {:>7} {:>9}",
        "course",
        "base rwd",
        "learn rwd",
        "base dist",
        "learn dist",
        "waypoints",
        "fell",
        "verdict"
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
            if b.reward > a.reward {
                "better"
            } else {
                "worse"
            }
        );
    }
    println!("\n* = course seed the policy never trained on");
    println!(
        "\"fell\" marks which side went over in at least one of the three\n\
         evaluation speeds. Reward is not distance: a policy can score better\n\
         by taking fewer penalties over less ground."
    );
}

/// Train one policy across every terrain family, then evaluate on held-out
/// seeds. Long episodes are intentional: the finish is at 64 m, so an
/// eight-second gait-tuning horizon cannot possibly supervise completion.
fn all_terrain(
    frame: Frame,
    seed: u64,
    iters: usize,
    mut cfg: ArsConfig,
    phys: Physics,
    args: &[String],
) {
    if flag(args, "--horizon").is_none() {
        cfg.horizon = 45.0;
    }
    if flag(args, "--batch").is_none() {
        cfg.scenarios_per_direction = 3;
    }
    let train_seeds = flag(args, "--train-seeds")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let eval_seeds = flag(args, "--eval-seeds")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2)
        .max(1);
    let unique_suite = terrain_suite(seed, train_seeds);
    let hard_repeat = flag(args, "--hard-repeat")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3);
    if cfg.n_dirs < hexapod_core::terrain::COURSES.len() {
        eprintln!(
            "note: {} directions cover the {n} terrain families over multiple iterations; --dirs {n} or more covers every family per update",
            cfg.n_dirs,
            n = hexapod_core::terrain::COURSES.len()
        );
    }

    let starting = if let Some(path) = flag(args, "--resume") {
        let policy = load_policy(&path).unwrap_or_else(|error| {
            eprintln!("could not load policy {path:?}: {error}");
            std::process::exit(2);
        });
        println!("resuming checkpoint: {path}");
        policy
    } else {
        Policy::seeded(Preset::default_for(frame), frame)
    };
    let frame = starting.frame;
    let baseline = Policy::seeded(Preset::default_for(frame), frame);
    let mut train_suite =
        difficulty_weighted_suite(&unique_suite, &starting, &phys, cfg.horizon, hard_repeat);
    if let Some(name) = flag(args, "--focus") {
        let Some(course) = hexapod_core::terrain::COURSES
            .iter()
            .copied()
            .find(|course| course.name().eq_ignore_ascii_case(&name))
        else {
            eprintln!("unknown focus course {name:?}");
            std::process::exit(2);
        };
        let repeats = flag(args, "--focus-repeat")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(20);
        let focused: Vec<Terrain> = unique_suite
            .iter()
            .filter(|terrain| terrain.course == course)
            .cloned()
            .collect();
        for _ in 0..repeats {
            train_suite.extend(focused.iter().cloned());
        }
        println!("focus: {} (+{repeats} copies per seed)", course.name());
    }
    let mut trainer = Trainer::new(starting, cfg, phys, seed ^ 0xA5A5);
    println!(
        "all-terrain curriculum: {} weighted scenarios from {} unique ({} courses x {train_seeds} seeds), {}s horizon",
        train_suite.len(),
        unique_suite.len(),
        hexapod_core::terrain::COURSES.len(),
        cfg.horizon
    );
    println!(
        "ARS dirs={} top={} batch={} alpha={} sigma={} | final waypoint is terminal",
        cfg.n_dirs, cfg.n_top, cfg.scenarios_per_direction, cfg.alpha, cfg.sigma
    );

    let wall = Instant::now();
    trainer.record_suite_baseline(&unique_suite);
    println!(
        "start     reward {:8.2}  route {:6.1}%  completion {:6.1}%",
        trainer.baseline_reward,
        trainer.last_eval.waypoint_fraction * 100.0,
        trainer.baseline_completion_rate * 100.0
    );
    println!(
        "\n{:>5} {:>9} {:>9} {:>8} {:>9} {:>9} {:>8}",
        "iter", "reward", "best", "route", "complete", "best cmp", "roll/s"
    );
    for i in 1..=iters {
        trainer.iterate_suite_with_eval(&train_suite, &unique_suite);
        if i % (iters / 20).max(1) == 0 || i == iters {
            let elapsed = wall.elapsed().as_secs_f64().max(1e-9);
            println!(
                "{i:>5} {:>9.2} {:>9.2} {:>7.1}% {:>8.1}% {:>8.1}% {:>8.1}",
                trainer.last_eval.reward,
                trainer.best_reward,
                trainer.last_eval.waypoint_fraction * 100.0,
                trainer.last_eval.completion_rate * 100.0,
                trainer.best_completion_rate * 100.0,
                trainer.rollouts as f64 / elapsed
            );
        }
    }

    let learned = trainer.best_policy();
    let heldout_seed = seed.wrapping_add(1_000_000);
    let (mean_completion, worst_completion) = print_heldout_matrix(
        &baseline,
        &learned,
        &phys,
        cfg.horizon,
        heldout_seed,
        eval_seeds,
    );
    println!(
        "\nheld-out completion: mean {:.1}%, worst course {:.1}% | {} exploratory rollouts in {:.1}s",
        mean_completion * 100.0,
        worst_completion * 100.0,
        trainer.rollouts,
        wall.elapsed().as_secs_f64()
    );
    println!(
        "best policy: feedback norm {:.3}, route {:.1}%, completion {:.1}%",
        learned.feedback_norm(),
        trainer.best_waypoint_fraction * 100.0,
        trainer.best_completion_rate * 100.0
    );
    if let Some(path) = flag(args, "--output") {
        if let Err(error) = save_policy(&path, &learned) {
            eprintln!("could not write policy {path:?}: {error}");
            std::process::exit(1);
        }
        println!("checkpoint: {path}");
    }
}

fn print_heldout_matrix(
    baseline: &Policy,
    learned: &Policy,
    phys: &Physics,
    horizon: f64,
    seed: u64,
    eval_seeds: usize,
) -> (f64, f64) {
    println!(
        "\nheld-out evaluation: {eval_seeds} unseen seed(s) per course, 3 speed commands each"
    );
    println!(
        "{:<10} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "course", "base cmp", "learn cmp", "base route", "learn route", "finish s"
    );
    let mut worst_completion = 1.0f64;
    let mut mean_completion = 0.0;
    for (course_i, course) in hexapod_core::terrain::COURSES.iter().copied().enumerate() {
        let mut base_completion = 0.0;
        let mut learned_completion = 0.0;
        let mut base_route = 0.0;
        let mut learned_route = 0.0;
        let mut finish_time = 0.0;
        for seed_i in 0..eval_seeds {
            let scenario_seed = seed
                .wrapping_add((course_i as u64).wrapping_mul(10_000))
                .wrapping_add(seed_i as u64);
            let terrain = Terrain::new(course, scenario_seed);
            let a = evaluate(&terrain, baseline, phys, horizon);
            let b = evaluate(&terrain, learned, phys, horizon);
            base_completion += a.completion_rate;
            learned_completion += b.completion_rate;
            base_route += a.waypoint_fraction;
            learned_route += b.waypoint_fraction;
            finish_time += b.finish_time;
        }
        let n = eval_seeds as f64;
        base_completion /= n;
        learned_completion /= n;
        base_route /= n;
        learned_route /= n;
        finish_time /= n;
        worst_completion = worst_completion.min(learned_completion);
        mean_completion += learned_completion / hexapod_core::terrain::COURSES.len() as f64;
        println!(
            "{:<10} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}% {:>9.2}",
            course.name(),
            base_completion * 100.0,
            learned_completion * 100.0,
            base_route * 100.0,
            learned_route * 100.0,
            finish_time
        );
    }
    (mean_completion, worst_completion)
}

fn terrain_suite(seed: u64, seeds_per_course: usize) -> Vec<Terrain> {
    hexapod_core::terrain::COURSES
        .iter()
        .copied()
        .enumerate()
        .flat_map(|(course_i, course)| {
            (0..seeds_per_course).map(move |seed_i| {
                let scenario_seed = seed
                    .wrapping_add((course_i as u64).wrapping_mul(10_000))
                    .wrapping_add(seed_i as u64);
                Terrain::new(course, scenario_seed)
            })
        })
        .collect()
}

/// Repeat scenarios in proportion to the seeded policy's failure rate. Every
/// terrain remains present once; `hard_repeat` only spends more of the finite
/// rollout budget where the current controller cannot yet finish.
fn difficulty_weighted_suite(
    terrains: &[Terrain],
    baseline: &Policy,
    phys: &Physics,
    horizon: f64,
    hard_repeat: usize,
) -> Vec<Terrain> {
    let copies = terrains
        .iter()
        .map(|terrain| {
            let completion = evaluate(terrain, baseline, phys, horizon).completion_rate;
            1 + ((1.0 - completion) * hard_repeat as f64).ceil() as usize
        })
        .collect::<Vec<_>>();
    let mut weighted = Vec::new();
    for round in 0..copies.iter().copied().max().unwrap_or(0) {
        for (terrain, &n) in terrains.iter().zip(&copies) {
            if round < n {
                weighted.push(terrain.clone());
            }
        }
    }
    weighted
}

fn save_policy(path: &str, policy: &Policy) -> std::io::Result<()> {
    std::fs::write(path, hexapod_core::checkpoint::to_text(policy))
}

fn load_policy(path: &str) -> Result<Policy, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    hexapod_core::checkpoint::from_text(&text)
}

fn eval_all(seed: u64, mut cfg: ArsConfig, phys: Physics, args: &[String]) {
    if flag(args, "--horizon").is_none() {
        cfg.horizon = 45.0;
    }
    let eval_seeds = flag(args, "--eval-seeds")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2)
        .max(1);
    let Some(path) = flag(args, "--policy") else {
        eprintln!("eval-all requires --policy PATH");
        std::process::exit(2);
    };
    let learned = load_policy(&path).unwrap_or_else(|error| {
        eprintln!("could not load policy {path:?}: {error}");
        std::process::exit(2);
    });
    let baseline = Policy::seeded(Preset::default_for(learned.frame), learned.frame);
    println!(
        "policy {path} | {} legs | horizon {}s | seed {seed}",
        learned.frame.legs(),
        cfg.horizon
    );
    if let Some(name) = flag(args, "--course") {
        let Some(course) = hexapod_core::terrain::COURSES
            .iter()
            .copied()
            .find(|course| course.name().eq_ignore_ascii_case(&name))
        else {
            eprintln!("unknown course {name:?}");
            std::process::exit(2);
        };
        print_course_detail(
            course,
            &baseline,
            &learned,
            &phys,
            cfg.horizon,
            seed,
            eval_seeds,
        );
        return;
    }
    let (mean, worst) =
        print_heldout_matrix(&baseline, &learned, &phys, cfg.horizon, seed, eval_seeds);
    println!(
        "\ncompletion: mean {:.1}%, worst course {:.1}%",
        mean * 100.0,
        worst * 100.0
    );
}

fn print_course_detail(
    course: Course,
    baseline: &Policy,
    learned: &Policy,
    phys: &Physics,
    horizon: f64,
    seed: u64,
    eval_seeds: usize,
) {
    let speeds: &[f64] = hexapod_core::sim::eval_speeds(course);
    println!(
        "\n{:<7} {:>5} {:>5} {:>9} {:>8} {:>10} {:>8} {:>7} {:>7} {:>9}",
        "policy", "seed", "m/s", "reward", "z m", "waypoints", "time s", "x m", "jumps", "state"
    );
    for seed_i in 0..eval_seeds {
        let scenario_seed = seed.wrapping_add(seed_i as u64);
        let terrain = Terrain::new(course, scenario_seed);
        for (label, policy) in [("base", baseline), ("learn", learned)] {
            for &speed in speeds {
                let result = rollout(&terrain, policy, phys, horizon, Cmd::at(speed), None);
                let state = if result.completed {
                    "FINISH"
                } else if result.broken {
                    "BROKE"
                } else if result.fell {
                    "FELL"
                } else if result.finished {
                    "SKIPPED"
                } else {
                    "TIMEOUT"
                };
                println!(
                    "{label:<7} {scenario_seed:>5} {speed:>5.1} {:>9.2} {:>8.2} {:>4}/{:<5} {:>8.2} {:>7.2} {:>7} {:>9}",
                    result.reward,
                    result.distance,
                    result.reached,
                    terrain.waypoints.len(),
                    result.elapsed,
                    result.end_x,
                    result.jumps,
                    state
                );
            }
        }
    }
}

/// Size the servos for a build, comparing the hand-tuned gait against a
/// trained one on the same machine.
fn bom(
    frame: Frame,
    course: Course,
    seed: u64,
    iters: usize,
    cfg: ArsConfig,
    phys: Physics,
    build: Build,
) {
    let terrain = Terrain::new(course, seed);
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
/// Train the same course once per servo, over several seeds.
///
/// Several, not one, because one is not a measurement. On a single seed the
/// spread between servos here is about nine points of reward and the spread
/// between seeds for the *same* servo is twenty-five, so a one-seed table
/// invites a conclusion its own noise cannot support -- it once read as the
/// strongest servo in the catalogue being the worst to walk on. The columns
/// that survive averaging are the ones worth reading: peak load against
/// rating, and cost of transport.
fn servo_shootout(
    frame: Frame,
    course: Course,
    seed: u64,
    iters: usize,
    cfg: ArsConfig,
    build: Build,
    seeds: usize,
) {
    println!(
        "{} seeds {}-{}, {iters} iterations per servo, {:.2} kg at scale {:.3}\n",
        course.name(),
        seed,
        seed + seeds as u64 - 1,
        build.mass_kg,
        build.scale
    );
    println!(
        "{:<11} {:>7} {:>6} {:>9} {:>9} {:>7} {:>11} {:>7} {:>6}",
        "servo", "kg-cm", "rpm", "base rwd", "best rwd", "range", "peak/stall", "worst", "CoT"
    );

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let range = |v: &[f64]| {
        v.iter().copied().fold(f64::MIN, f64::max) - v.iter().copied().fold(f64::MAX, f64::min)
    };

    for servo in SERVOS.iter() {
        let phys = build.physics(Some(servo));
        let (mut base, mut best, mut load, mut cot) = (vec![], vec![], vec![], vec![]);
        for k in 0..seeds as u64 {
            let terrain = Terrain::new(course, seed + k);
            let mut t = Trainer::new(
                Policy::seeded(Preset::default_for(frame), frame),
                cfg,
                phys,
                (seed + k) ^ 0xA5A5,
            );
            t.record_baseline(&terrain);
            for _ in 0..iters {
                t.iterate(&terrain);
            }
            let e = evaluate(&terrain, &t.best_policy(), &phys, cfg.horizon);
            base.push(t.baseline_reward);
            best.push(t.best_reward);
            load.push(e.peak_servo_load);
            cot.push(e.cot);
        }
        println!(
            "{:<11} {:>7.1} {:>6.0} {:>9.2} {:>9.2} {:>6.1} {:>10.2}x {:>6.2}x {:>6.2}",
            servo.part,
            servo.stall_kgcm,
            phys.actuator.omega_max * 60.0 / std::f64::consts::TAU,
            mean(&base),
            mean(&best),
            range(&best),
            mean(&load),
            load.iter().copied().fold(f64::MIN, f64::max),
            mean(&cot)
        );
    }
    println!(
        "\npeak/stall above 1.00 means the servo was driven past its rating and the\n\
         leg gave way under load. `worst` is the highest of the {seeds} seeds, which is\n\
         the number that sizes a build; `range` is the spread of `best rwd` across\n\
         them, and any reward gap smaller than it is not a result."
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
    let servo = hexapod_core::SERVOS
        .iter()
        .find(|s| s.part == pick)
        .unwrap();
    let s = solve(&trace, servo, &sizing);

    println!("\n=== cheapest viable build: {} ===\n", servo.part);
    println!(
        "converged in {} iterations of the mass/current loop",
        s.iterations
    );
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
        line(
            "RANGEFINDER",
            r.name,
            n.rangers,
            r.unit_price(),
            "one per leg",
        );
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

// ------------------------------------------------------------- joint-train

/// Train a joint-level policy through the curriculum and write a checkpoint.
///
///   hexapod joint-train [--iters 400] [--dirs 16] [--top 6] [--alpha 0.002]
///                       [--sigma 0.02] [--scenarios 2] [--seed 1]
///                       [--backend nexus-gpu] [--batch-envs 128]
///                       [--resume policy.txt] [--stage mixed]
///                       [--out checkpoints/joint-v1.txt]
fn joint_train(frame: Frame, mut phys: Physics, args: &[String]) {
    use hexapod_core::joint_rl::{
        JointBackend, JointCfg, JointPolicy, Stage, from_text, to_text, train_curriculum_from,
    };

    motor_flags(&mut phys, args);
    let num = |k: &str, d: f64| -> f64 { flag(args, k).and_then(|v| v.parse().ok()).unwrap_or(d) };
    let iters = num("--iters", 400.0) as usize;
    let seed = num("--seed", 1.0) as u64;
    let out = flag(args, "--out").unwrap_or_else(|| "checkpoints/joint-v1.txt".into());
    // Defaults come from the core, not from here: two places to change a step
    // size is one place to forget, and the last time these drifted apart a
    // tuning run silently used the value it was supposed to be replacing.
    let d = JointCfg::default();
    let backend = flag(args, "--backend")
        .map(|value| parse_joint_backend(&value))
        .unwrap_or(JointBackend::Rapier);
    let cfg = JointCfg {
        dirs: num("--dirs", d.dirs as f64) as usize,
        top: num("--top", d.top as f64) as usize,
        alpha: num("--alpha", d.alpha),
        sigma: num("--sigma", d.sigma),
        scenarios: num("--scenarios", d.scenarios as f64) as usize,
        workers: num("--workers", d.workers as f64) as usize,
        backend,
        batch_envs: num("--batch-envs", d.batch_envs as f64) as usize,
        device: num("--device", d.device as f64) as usize,
    };

    println!(
        "# hexapod joint-train  {} legs · {} motors · {} weights",
        frame.legs(),
        hexapod_core::joint_rl::n_act(frame),
        hexapod_core::joint_rl::n_theta(frame),
    );
    println!(
        "# {} dirs x {} scenarios x 2 sides, alpha {:.3}, sigma {:.3}, budget {iters}",
        cfg.dirs, cfg.scenarios, cfg.alpha, cfg.sigma
    );
    println!(
        "# backend {} · batch {} · device {}",
        cfg.backend.name(),
        cfg.batch_envs.max(1),
        cfg.device,
    );
    println!(
        "# curriculum: {}",
        hexapod_core::joint_rl::STAGES
            .iter()
            .map(|s| s.name())
            .collect::<Vec<_>>()
            .join(" -> ")
    );
    println!();
    println!("  iter  stage       score  target   dist    secs  feet   air  ");

    let t0 = Instant::now();
    let mut last_stage: Option<Stage> = None;
    let resume = flag(args, "--resume");
    let start_stage = flag(args, "--stage")
        .map(|s| parse_joint_stage(&s))
        .unwrap_or(if resume.is_some() {
            Stage::Mixed
        } else {
            Stage::Stand
        });
    let initial = match resume.as_deref() {
        Some(path) => {
            let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
                eprintln!("could not read joint checkpoint {path}: {e}");
                std::process::exit(1)
            });
            let policy = from_text(&text).unwrap_or_else(|e| {
                eprintln!("could not load joint checkpoint {path}: {e}");
                std::process::exit(2)
            });
            if policy.frame != frame {
                eprintln!(
                    "checkpoint has {} legs but --legs selected {}",
                    policy.frame.legs(),
                    frame.legs()
                );
                std::process::exit(2)
            }
            println!("# resuming {path} from {}", start_stage.name());
            policy
        }
        None => JointPolicy::seeded(frame, seed),
    };
    let policy = train_curriculum_from(initial, &phys, &cfg, iters, seed, start_stage, |p, _| {
        // One line per stage check. A promotion is the interesting event, so
        // it is marked rather than left to be inferred from the score.
        let mark = if p.promoted { "  <- cleared" } else { "" };
        let fresh = last_stage != Some(p.stage);
        if fresh {
            println!("  ---- {} ----", p.stage.name());
        }
        last_stage = Some(p.stage);
        println!(
            "  {:>4}  {:<10} {:>6.3}  {:>6.3} {:>6.2} {:>7.2} {:>5.2} {:>5.2}{}  [{:.0}s]",
            p.iter,
            p.stage.name(),
            p.score,
            p.stage.promote_at(),
            p.eval.distance,
            p.eval.secs,
            p.eval.support,
            p.eval.air,
            mark,
            t0.elapsed().as_secs_f64(),
        );
    });

    let text = to_text(&policy);
    match std::fs::write(&out, &text) {
        Ok(()) => println!("\nwrote {out}  ({} bytes)", text.len()),
        Err(e) => {
            eprintln!("\ncould not write {out}: {e}");
            std::process::exit(1);
        }
    }
    let metadata = format!(
        concat!(
            "hexapod-joint-run-v1\n",
            "hexapod_version={}\n",
            "nexus3d_version=0.5.0\n",
            "backend={}\n",
            "device={}\n",
            "batch_envs={}\n",
            "legs={}\n",
            "seed={}\n",
            "start_stage={}\n",
            "budget={}\n",
            "dirs={}\n",
            "top={}\n",
            "scenarios={}\n",
            "alpha={:.17e}\n",
            "sigma={:.17e}\n",
            "motor_stiff={:.17e}\n",
            "motor_damp={:.17e}\n",
            "motor_max={:.17e}\n",
            "substeps={}\n",
            "solver_iters={}\n",
        ),
        env!("CARGO_PKG_VERSION"),
        cfg.backend.name(),
        cfg.device,
        cfg.batch_envs.max(1),
        frame.legs(),
        seed,
        start_stage.name(),
        iters,
        cfg.dirs,
        cfg.top,
        cfg.scenarios,
        cfg.alpha,
        cfg.sigma,
        phys.motor_stiff,
        phys.motor_damp,
        phys.motor_max,
        phys.substeps,
        phys.solver_iters,
    );
    let metadata_path = format!("{out}.meta");
    if let Err(error) = std::fs::write(&metadata_path, metadata) {
        eprintln!("could not write run metadata {metadata_path}: {error}");
        std::process::exit(1);
    }
    println!("wrote {metadata_path}");
}

fn parse_joint_stage(value: &str) -> hexapod_core::joint_rl::Stage {
    use hexapod_core::joint_rl::Stage;
    match value.to_ascii_lowercase().replace('_', "-").as_str() {
        "stand" => Stage::Stand,
        "walk" | "walk-flat" => Stage::WalkFlat,
        "run" | "run-flat" => Stage::RunFlat,
        "rough" => Stage::Rough,
        "gaps" => Stage::Gaps,
        "jump" => Stage::Jump,
        "mixed" | "all" => Stage::Mixed,
        other => {
            eprintln!(
                "unknown joint curriculum stage {other:?}; try stand, walk-flat, run-flat, rough, gaps, jump, or mixed"
            );
            std::process::exit(2)
        }
    }
}

fn parse_joint_backend(value: &str) -> hexapod_core::joint_rl::JointBackend {
    use hexapod_core::joint_rl::JointBackend;
    match value.to_ascii_lowercase().replace('_', "-").as_str() {
        "nexus" | "nexus-gpu" | "nexus-webgpu" => JointBackend::NexusGpu,
        "rapier" | "cpu" => JointBackend::Rapier,
        other => {
            eprintln!("unknown joint backend {other:?}; try nexus-gpu or rapier");
            std::process::exit(2)
        }
    }
}

/// Evaluate a saved motor-level policy with the same rollout and route
/// contract used during training. With no `--course`, the selected stage's
/// whole course set is reported; `mixed` therefore means all fifteen.
fn joint_eval(course: Course, mut phys: Physics, args: &[String]) {
    use hexapod_core::joint_rl::{
        JointBackend, JointCfg, Stage, evaluate_on_courses_backend, from_text,
    };

    motor_flags(&mut phys, args);
    let Some(path) = flag(args, "--policy") else {
        eprintln!("joint-eval requires --policy PATH");
        std::process::exit(2)
    };
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("could not read joint checkpoint {path}: {e}");
        std::process::exit(1)
    });
    let policy = from_text(&text).unwrap_or_else(|e| {
        eprintln!("could not load joint checkpoint {path}: {e}");
        std::process::exit(2)
    });
    let stage = flag(args, "--stage")
        .map(|s| parse_joint_stage(&s))
        .unwrap_or(Stage::Mixed);
    let count = flag(args, "--eval-seeds")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1);
    let base = flag(args, "--seed")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(100_001);
    let seeds: Vec<u64> = (0..count).map(|i| base + i as u64 * 101).collect();
    let selected: Vec<Course> = if args.iter().any(|a| a == "--course") {
        vec![course]
    } else {
        stage.courses().to_vec()
    };
    let defaults = JointCfg::default();
    let cfg = JointCfg {
        backend: flag(args, "--backend")
            .map(|value| parse_joint_backend(&value))
            .unwrap_or(JointBackend::Rapier),
        batch_envs: flag(args, "--batch-envs")
            .and_then(|value| value.parse().ok())
            .unwrap_or(defaults.batch_envs),
        device: flag(args, "--device")
            .and_then(|value| value.parse().ok())
            .unwrap_or(defaults.device),
        ..defaults
    };

    println!(
        "# joint-eval {path} · {} legs · {} · {} · {} held-out seed(s)",
        policy.frame.legs(),
        stage.name(),
        cfg.backend.name(),
        seeds.len()
    );
    println!("course        score    dist   wp %  finish %  time   feet  fell");
    let results: Vec<_> = selected
        .iter()
        .map(|&c| {
            let result = evaluate_on_courses_backend(&policy, &phys, stage, &[c], &seeds, &cfg)
                .unwrap_or_else(|error| {
                    eprintln!("joint evaluation failed on {}: {error}", c.name());
                    std::process::exit(1)
                });
            (c, result)
        })
        .collect();
    for (c, r) in &results {
        println!(
            "{:<11} {:>7.3} {:>7.2} {:>6.1} {:>9.1} {:>6.2} {:>6.2} {:>5}",
            c.name(),
            r.score,
            r.distance,
            100.0 * r.waypoint_fraction,
            100.0 * r.completion_rate,
            r.finish_time,
            r.support,
            if r.fell { "yes" } else { "no" },
        );
    }
    let n = results.len().max(1) as f64;
    let mean = |f: fn(&hexapod_core::joint_rl::JointRollout) -> f64| {
        results.iter().map(|(_, r)| f(r)).sum::<f64>() / n
    };
    println!(
        "mean        {:>7.3} {:>7.2} {:>6.1} {:>9.1} {:>6.2} {:>6.2} {:>5}",
        mean(|r| r.score),
        mean(|r| r.distance),
        100.0 * mean(|r| r.waypoint_fraction),
        100.0 * mean(|r| r.completion_rate),
        mean(|r| r.finish_time),
        mean(|r| r.support),
        if results.iter().any(|(_, r)| r.fell) {
            "yes"
        } else {
            "no"
        },
    );
}

fn motor_flags(phys: &mut Physics, args: &[String]) {
    if let Some(v) = flag(args, "--stiff").and_then(|v| v.parse().ok()) {
        phys.motor_stiff = v;
    }
    if let Some(v) = flag(args, "--damp").and_then(|v| v.parse().ok()) {
        phys.motor_damp = v;
    }
    if let Some(v) = flag(args, "--maxf").and_then(|v| v.parse().ok()) {
        phys.motor_max = v;
    }
    if let Some(v) = flag(args, "--substeps").and_then(|v| v.parse().ok()) {
        phys.substeps = v;
    }
    if let Some(v) = flag(args, "--solver").and_then(|v| v.parse().ok()) {
        phys.solver_iters = v;
    }
    if let Some(v) = flag(args, "--pgs").and_then(|v| v.parse().ok()) {
        phys.pgs_iters = v;
    }
    if let Some(v) = flag(args, "--footmu").and_then(|v| v.parse().ok()) {
        phys.foot_mu = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_suite_contains_every_course_and_seed() {
        let suite = terrain_suite(11, 2);
        assert_eq!(suite.len(), hexapod_core::terrain::COURSES.len() * 2);
        for course in hexapod_core::terrain::COURSES {
            assert_eq!(suite.iter().filter(|t| t.course == course).count(), 2);
        }
    }

    #[test]
    fn difficulty_weighting_keeps_easy_scenarios_and_repeats_failures() {
        let suite = terrain_suite(3, 1);
        let frame = Frame::default();
        let baseline = Policy::seeded(Preset::default_for(frame), frame);
        let uniform = difficulty_weighted_suite(&suite, &baseline, &Physics::default(), 0.0, 0);
        let weighted = difficulty_weighted_suite(&suite, &baseline, &Physics::default(), 0.0, 3);
        assert_eq!(uniform.len(), suite.len());
        assert_eq!(weighted.len(), suite.len() * 4);
        for course in hexapod_core::terrain::COURSES {
            assert!(weighted.iter().any(|t| t.course == course));
        }
    }

    #[test]
    fn policy_checkpoint_round_trips_exactly() {
        let frame = Frame::default();
        let mut policy = Policy::seeded(Preset::default_for(frame), frame);
        policy.theta[7] = 0.123456789;
        policy.norm.n = 42.0;
        policy.norm.mean[3] = -0.75;
        policy.norm.m2[5] = 9.25;
        let path = std::env::temp_dir().join(format!(
            "hexapod-policy-{}-{}.txt",
            std::process::id(),
            0x5eed_u64
        ));
        save_policy(path.to_str().unwrap(), &policy).unwrap();
        let loaded = load_policy(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.frame, policy.frame);
        assert_eq!(loaded.theta, policy.theta);
        assert_eq!(loaded.base_offsets, policy.base_offsets);
        assert_eq!(loaded.feedback, policy.feedback);
        assert_eq!(loaded.norm.n, policy.norm.n);
        assert_eq!(loaded.norm.mean, policy.norm.mean);
        assert_eq!(loaded.norm.m2, policy.norm.m2);
        assert!(loaded.norm.frozen);
    }
}
