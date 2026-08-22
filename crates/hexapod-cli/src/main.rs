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

use hexapod_core::hardware::{Build, PRICES_CHECKED, Provenance, SERVOS, TorqueMeter, shortlist};
use hexapod_core::power::{Kind, Sizing, TorqueTrace, parts_of, solve};
use hexapod_core::robot::{MAX_LEGS, Stance};
use hexapod_core::{Course, Frame, Physics, Terrain};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");

    let course = parse_course(flag(&args, "--course").unwrap_or_else(|| "mixed".into()));
    let frame = Frame::new(
        flag(&args, "--legs")
            .and_then(|v| v.parse().ok())
            .unwrap_or(6),
    );
    let build = build_from(&args);
    let mut phys = build.physics(flag(&args, "--servo").and_then(|name| {
        SERVOS
            .iter()
            .find(|s| s.part.eq_ignore_ascii_case(&name))
            .or_else(|| {
                eprintln!(
                    "unknown servo {name:?}; known: {}",
                    SERVOS.iter().map(|s| s.part).collect::<Vec<_>>().join(", ")
                );
                std::process::exit(2)
            })
    }));
    motor_flags(&mut phys, &args);

    match cmd {
        "joint-train" => joint_train(frame, phys, &args),
        "joint-eval" => joint_eval(course, phys, &args),
        "bom" => bom(frame, course, phys, build, &args),
        "system" => system(frame, course, phys, &args),
        "servos" => servos_json(),
        "parts" => parts_json(),
        "courses" => courses_json(),
        other => {
            if other != "help" && other != "--help" {
                eprintln!("unknown command {other:?}\n");
            }
            eprintln!(
                "hexapod <command>\n\
                 \n\
                 \x20 joint-train   train the motor-level policy\n\
                 \x20 joint-eval    evaluate a joint checkpoint\n\
                 \x20 bom           joint torque against the servo catalogue\n\
                 \x20 system        whole-machine sizing, mass/current fixed point\n\
                 \x20 servos        servo catalogue as JSON\n\
                 \x20 parts         parts catalogue as JSON\n\
                 \x20 courses       course catalogue as JSON\n"
            );
            std::process::exit(2)
        }
    }
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

/// Emit the servo catalogue as JSON. `build.sh` inlines this into the web
/// bundle so the browser and the simulator share one source of truth.
/// Torque over a joint-level rollout, folded into both sizing views.
///
/// The hand-written gaits used to supply the trajectory here. With them gone
/// the trajectory comes from the articulated plant itself: a checkpoint if one
/// is given, otherwise the standing pose held, which is the load case a servo
/// has to carry regardless of how it walks.
///
/// Foot load is shared equally between the feet that are down. The plant
/// reports contact, not normal force, so this is an assumption — and the honest
/// direction of it, since a real machine loads its feet unevenly and the worst
/// foot carries more than its share. `Build::dynamic_factor` is the allowance.
fn joint_torque_trace(
    frame: Frame,
    phys: &Physics,
    terrain: &Terrain,
    build: &Build,
    actor: Option<&str>,
    secs: f64,
) -> (TorqueMeter, TorqueTrace) {
    use hexapod_core::joint_rl::{n_act, JointEnv, Stage};
    let mut env = JointEnv::new(frame, phys, terrain.clone(), Stage::WalkFlat);
    let policy = actor.map(|path| {
        let text = std::fs::read_to_string(path).unwrap_or_else(|error| {
            eprintln!("could not read {path}: {error}");
            std::process::exit(1)
        });
        hexapod_core::joint_rl::from_text(&text).unwrap_or_else(|error| {
            eprintln!("{path} is not a joint policy: {error}");
            std::process::exit(1)
        })
    });

    let joints = frame.legs() * 3;
    let mut meter = TorqueMeter::default();
    let mut trace = TorqueTrace {
        per_kg: Vec::new(),
        ticks: 0,
        joints,
        frame,
        stance: Stance::default(),
        scale: phys.scale,
    };

    let probe = Build {
        mass_kg: 1.0,
        ..*build
    };
    let ticks = (secs / hexapod_core::DT) as usize;
    let zero = vec![0.0; n_act(frame)];
    for _ in 0..ticks {
        let mut action = zero.clone();
        if let Some(p) = policy.as_ref() {
            let observation = env.state().to_vec();
            p.act(&observation, &mut action);
        }
        if env.step(&action).is_err() {
            env.reset();
            continue;
        }
        let q = env.joint_angles();
        let down = env.foot_contacts();
        let n_down = down.iter().take(frame.legs()).filter(|c| **c).count();
        let mut share = [0.0; MAX_LEGS];
        if n_down > 0 {
            for leg in 0..frame.legs() {
                if down[leg] {
                    share[leg] = 1.0 / n_down as f64;
                }
            }
        }
        meter.observe(frame, &q, &down, &share, build);

        // The same pose priced at one kilogram, which is what the sizing loop
        // rescales for every candidate mass.
        let mut per_kg = TorqueMeter::default();
        per_kg.observe(frame, &q, &down, &share, &probe);
        let peak = per_kg.peak;
        let mut row = vec![0.0; joints];
        for leg in 0..frame.legs() {
            for c in 0..3 {
                row[leg * 3 + c] = peak[c];
            }
        }
        trace.observe(&row, joints);
    }
    (meter, trace)
}

fn bom(frame: Frame, course: Course, phys: Physics, build: Build, args: &[String]) {
    let terrain = Terrain::new(course, 1);
    let actor = flag(args, "--policy");
    if actor.is_none() {
        println!(
            "note: no --policy given, so this is the standing load only. A walking\n\
             \x20     machine asks for several times more; pass a joint checkpoint to\n\
             \x20     size against one.\n"
        );
    }
    let (meter, _) = joint_torque_trace(frame, &phys, &terrain, &build, actor.as_deref(), 8.0);
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
        "source", "coxa", "femur", "tibia", "peak foot N", "required"
    );
    let mut required = 0.0f64;
    {
        let name = if actor.is_some() { "learned" } else { "standing" };
        let m = &meter;
        let k = m.peak_kgcm();
        let req = m.required_kgcm(&build);
        required = req;
        println!(
            "{name:<12} {:>8.2} {:>8.2} {:>8.2} {:>11.1} {:>10.1} kg-cm",
            k[0], k[1], k[2], m.peak_foot_load, req
        );
    }


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

fn system(frame: Frame, course: Course, phys: Physics, args: &[String]) {
    let scale = phys.scale;
    let mut sizing = Sizing::default();
    if let Some(v) = flag(args, "--chassis").and_then(|v| v.parse().ok()) {
        sizing.chassis_kg = v;
    }
    if let Some(v) = flag(args, "--runtime").and_then(|v| v.parse().ok()) {
        sizing.runtime_min = v;
    }

    let terrain = Terrain::new(course, 1);
    let actor = flag(args, "--policy");
    let label = if actor.is_some() { "learned" } else { "standing" };
    if actor.is_none() {
        println!(
            "note: no --policy given, so this sizes the standing load only.\n"
        );
    }
    let (_, trace) =
        joint_torque_trace(frame, &phys, &terrain, &Build::default(), actor.as_deref(), 8.0);
    println!(
        "torque: {label} on {} | chassis {:.2} kg | femur {:.0} mm | target runtime {:.0} min\n",
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
