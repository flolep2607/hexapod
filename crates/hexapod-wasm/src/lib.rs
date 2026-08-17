//! C-ABI bridge from the hexapod core to the browser dashboard.
//!
//! No `wasm-bindgen`: the module exports plain `extern "C"` functions and a
//! pointer to a flat `f32` buffer that JavaScript reads through a typed array
//! view over the wasm linear memory. That keeps the toolchain to a plain
//! `cargo build --target wasm32-unknown-unknown` and the artefact under 50 kB.
//!
//! All state is a single process-global. wasm32 here is single-threaded and
//! JavaScript calls in one at a time, so there is no aliasing to worry about.

pub mod layout;

use core::ptr::addr_of_mut;

use hexapod_core::ars::{ArsConfig, Trainer};
use hexapod_core::hardware::{Build, TorqueMeter};
use hexapod_core::policy::{n_theta, Gait, Policy, Preset, GAIT_BOUNDS};
use hexapod_core::math::{squash, unsquash};
use hexapod_core::sim::{Cmd, Sim, CRUISE_DEFAULT, CRUISE_MAX, CRUISE_MIN, JUMP_CRUISE_DEFAULT, JUMP_CRUISE_MAX, JUMP_CRUISE_MIN};
use hexapod_core::dynamics::Physics;
use hexapod_core::hardware::{Servo, NM_TO_KGCM, SERVOS};
use hexapod_core::terrain::{Course, Terrain, Z_MAX};
use hexapod_core::{Frame, MAX_LEGS, MIN_LEGS};

use layout::*;

const MODE_BASELINE: u32 = 0;
const MODE_LEARNED: u32 = 1;

struct App {
    frame: Frame,
    terrain: Terrain,
    course: Course,
    course_seed: u64,
    preset: Preset,

    /// Hand-tuned policy the sliders edit. Feedback layer stays at zero, so
    /// this is pure open-loop walking.
    baseline: Policy,
    trainer: Trainer,
    /// Cached copy of the trainer's best, so stepping does not clone per frame.
    learned: Policy,
    trained: bool,

    live: Sim,
    live_gait: Gait,
    mode: u32,
    since_fall: f64,

    build: Build,
    /// Index into `SERVOS` of the servo driving the joints, or `None` for the
    /// generic 20 kg-cm default.
    servo: Option<usize>,
    phys: Physics,
    /// Commanded cruise speed, m/s.
    cruise: f64,
    /// Let the policy steer itself along the course's route.
    nav: bool,
    meter: TorqueMeter,
    sizing: Sizing,

    telemetry: Vec<f32>,
    course_buf: Vec<f32>,
    route_buf: Vec<f32>,
    torque_buf: Vec<f32>,
    system_buf: Vec<f32>,
}

static mut APP: Option<App> = None;

#[inline]
#[allow(static_mut_refs)]
fn app() -> &'static mut App {
    unsafe { (*addr_of_mut!(APP)).as_mut().expect("hx_init not called") }
}

fn make(seed: u64) -> App {
    let course = Course::Mixed;
    let frame = Frame::new(6);
    let preset = Preset::default_for(frame);
    let terrain = Terrain::new(course, seed);
    let baseline = Policy::seeded(preset, frame);
    let build = Build::default();
    let phys = build.physics(None);
    let trainer = Trainer::new(
        Policy::seeded(preset, frame),
        ArsConfig {
            horizon: 12.0,
            ..ArsConfig::default()
        },
        phys,
        seed ^ 0xA5A5,
    );
    let learned = baseline.clone();
    let live_gait = baseline.gait();

    let mut a = App {
        frame,
        terrain,
        course,
        course_seed: seed,
        preset,
        baseline,
        trainer,
        learned,
        trained: false,
        live: Sim::default(),
        live_gait,
        mode: MODE_BASELINE,
        since_fall: 0.0,
        build,
        servo: None,
        phys,
        cruise: CRUISE_DEFAULT,
        nav: true,
        meter: TorqueMeter::default(),
        sizing: Sizing::default(),
        telemetry: vec![0.0; T_LEN],
        course_buf: Vec::new(),
        route_buf: Vec::new(),
        torque_buf: vec![0.0; 8],
        system_buf: vec![0.0; S_LEN],
    };
    a.course_buf = a.terrain.export();
    a.route_buf = a.terrain.export_route();
    a.live.reset(&a.terrain, &a.live_gait, &a.phys);
    a
}

// ---------------------------------------------------------------- lifecycle

#[no_mangle]
pub extern "C" fn hx_init(seed: u32) {
    unsafe { APP = Some(make(seed as u64)) };
    app().reset_live();
}

#[no_mangle]
pub extern "C" fn hx_set_course(kind: u32, seed: u32) {
    let a = app();
    a.course = Course::from_u32(kind);
    a.course_seed = seed as u64;
    a.terrain = Terrain::new(a.course, a.course_seed);
    a.course_buf = a.terrain.export();
    a.route_buf = a.terrain.export_route();
    a.cruise = if a.course.is_jump() {
        JUMP_CRUISE_DEFAULT
    } else {
        CRUISE_DEFAULT
    };
    a.reset_training();
    a.reset_live();
}

/// Set the number of legs. Even counts only, `MIN_LEGS..=MAX_LEGS`; anything
/// else is clamped. Returns the count actually adopted, and 0 if nothing
/// changed.
///
/// A different leg count is a different machine with a different policy shape,
/// so everything learned is discarded.
#[no_mangle]
pub extern "C" fn hx_set_legs(legs: u32) -> u32 {
    let a = app();
    let next = Frame::new(legs as usize);
    if next == a.frame {
        return 0;
    }
    a.frame = next;
    // A frame that cannot hold itself up on an alternating gait has to start
    // on the crawl instead, or it falls over before the user sees anything.
    if !next.alternating_is_stable() && a.preset == Preset::Tripod {
        a.preset = Preset::default_for(next);
    }
    a.baseline = Policy::seeded(a.preset, a.frame);
    a.reset_training();
    a.reset_live();
    a.meter = TorqueMeter::default();
    next.legs() as u32
}

/// The gait preset in force. Changing the leg count can change it, because a
/// frame that cannot stand on an alternating gait is moved to the crawl.
#[no_mangle]
pub extern "C" fn hx_preset() -> u32 {
    app().preset as u32
}

#[no_mangle]
pub extern "C" fn hx_legs() -> u32 {
    app().frame.legs() as u32
}

#[no_mangle]
pub extern "C" fn hx_legs_min() -> u32 {
    MIN_LEGS as u32
}

#[no_mangle]
pub extern "C" fn hx_legs_max() -> u32 {
    MAX_LEGS as u32
}

/// 1 when an alternating half-set gait keeps this frame statically stable.
#[no_mangle]
pub extern "C" fn hx_alternating_is_stable() -> u32 {
    u32::from(app().frame.alternating_is_stable())
}

/// Number of parameters the policy carries on the current frame, and the shape
/// of the feedback matrix that makes up most of them.
#[no_mangle]
pub extern "C" fn hx_theta_len() -> u32 {
    n_theta(app().frame) as u32
}

#[no_mangle]
pub extern "C" fn hx_n_obs() -> u32 {
    hexapod_core::policy::n_obs(app().frame) as u32
}

#[no_mangle]
pub extern "C" fn hx_n_act() -> u32 {
    hexapod_core::policy::n_act(app().frame) as u32
}

/// Phase offset of `leg` under `preset`, on the current frame.
///
/// The dashboard classifies a *measured* footfall pattern by matching it
/// against these, so the named patterns it compares against are the ones the
/// simulator actually defines rather than a second copy of the tables.
#[no_mangle]
pub extern "C" fn hx_preset_offset(preset: u32, leg: u32) -> f64 {
    let a = app();
    Preset::from_u32(preset).offsets(a.frame)[(leg as usize).min(MAX_LEGS - 1)]
}

#[no_mangle]
pub extern "C" fn hx_preset_count() -> u32 {
    3
}

#[no_mangle]
pub extern "C" fn hx_set_preset(p: u32) {
    let a = app();
    a.preset = Preset::from_u32(p);
    a.baseline = Policy::seeded(a.preset, a.frame);
    a.reset_training();
    a.reset_live();
}

#[no_mangle]
pub extern "C" fn hx_set_mode(mode: u32) {
    let a = app();
    if a.mode != mode {
        a.mode = mode;
        a.reset_live();
    }
}

#[no_mangle]
pub extern "C" fn hx_reset_live() {
    app().reset_live();
}

// ------------------------------------------------------------- manual gait

/// Set gait scalar `idx` (0..6) on the hand-tuned policy.
#[no_mangle]
pub extern "C" fn hx_set_param(idx: u32, value: f64) {
    let a = app();
    let i = idx as usize;
    if i >= GAIT_BOUNDS.len() {
        return;
    }
    a.baseline.theta[i] = unsquash(value, GAIT_BOUNDS[i].0, GAIT_BOUNDS[i].1);
    if a.mode == MODE_BASELINE {
        a.live_gait = a.baseline.gait();
    }
}

#[no_mangle]
pub extern "C" fn hx_get_param(idx: u32) -> f64 {
    let a = app();
    let i = idx as usize;
    if i >= GAIT_BOUNDS.len() {
        return 0.0;
    }
    squash(a.baseline.theta[i], GAIT_BOUNDS[i].0, GAIT_BOUNDS[i].1)
}

// ---------------------------------------------------------------- training

#[no_mangle]
pub extern "C" fn hx_set_train_cfg(dirs: u32, top: u32, alpha: f64, sigma: f64, horizon: f64) {
    let a = app();
    a.trainer.cfg.n_dirs = (dirs as usize).clamp(2, 32);
    a.trainer.cfg.n_top = (top as usize).clamp(1, a.trainer.cfg.n_dirs);
    a.trainer.cfg.alpha = alpha;
    a.trainer.cfg.sigma = sigma;
    a.trainer.cfg.horizon = horizon.clamp(2.0, 30.0);
}

#[no_mangle]
pub extern "C" fn hx_reset_training() {
    app().reset_training();
}

/// Run `iters` ARS iterations. Returns the best reward so far.
#[no_mangle]
pub extern "C" fn hx_train(iters: u32) -> f64 {
    let a = app();
    if a.trainer.curve.is_empty() {
        a.trainer.record_baseline(&a.terrain);
    }
    for _ in 0..iters {
        a.trainer.iterate(&a.terrain);
    }
    a.trained = true;
    a.learned = a.trainer.best_policy();
    a.trainer.best_reward
}

/// Number of ARS iterations completed.
#[no_mangle]
pub extern "C" fn hx_iterations() -> u32 {
    app().trainer.iter as u32
}

// ------------------------------------------------------------------- build

#[no_mangle]
pub extern "C" fn hx_set_build(scale: f64, mass_kg: f64, safety: f64) {
    let a = app();
    let changed = (a.build.scale - scale).abs() > 1e-12
        || (a.build.mass_kg - mass_kg).abs() > 1e-12;
    a.build.scale = scale.clamp(0.02, 1.0);
    a.build.mass_kg = mass_kg.clamp(0.05, 200.0);
    a.build.safety = safety.clamp(1.0, 4.0);
    a.meter = TorqueMeter::default();
    // Mass and scale set the torque the joints see, so they are simulator
    // inputs, not just sizing inputs.
    if changed {
        a.refresh_physics();
    }
}

/// Choose the servo whose torque-speed line drives the joints. `0xFFFF_FFFF`
/// restores the generic default. Returns 1 if the machine changed.
#[no_mangle]
pub extern "C" fn hx_set_servo(index: u32) -> u32 {
    let a = app();
    let next = if (index as usize) < SERVOS.len() {
        Some(index as usize)
    } else {
        None
    };
    if a.servo == next {
        return 0;
    }
    a.servo = next;
    a.refresh_physics();
    1
}

/// Commanded cruise speed, clamped to the range training samples from on
/// the current course.
#[no_mangle]
pub extern "C" fn hx_set_cruise(v: f64) {
    let a = app();
    a.cruise = if a.course.is_jump() {
        v.clamp(JUMP_CRUISE_MIN, JUMP_CRUISE_MAX)
    } else {
        v.clamp(CRUISE_MIN, CRUISE_MAX)
    };
}

#[no_mangle]
pub extern "C" fn hx_cruise_lo() -> f64 {
    if app().course.is_jump() {
        JUMP_CRUISE_MIN
    } else {
        CRUISE_MIN
    }
}

#[no_mangle]
pub extern "C" fn hx_cruise_hi() -> f64 {
    if app().course.is_jump() {
        JUMP_CRUISE_MAX
    } else {
        CRUISE_MAX
    }
}

/// Measure peak joint torques over one full gait cycle of the active policy.
/// Writes `[coxa, femur, tibia, required, peak_foot_N, cycle_s, stance_mm, set_mass_kg]`
/// in kg-cm where applicable.
#[no_mangle]
pub extern "C" fn hx_measure_torque() -> *const f32 {
    let a = app();
    let policy: &Policy = if a.mode == MODE_LEARNED && a.trained {
        &a.learned
    } else {
        &a.baseline
    };
    let gait = if a.mode == MODE_LEARNED && a.trained {
        a.learned.gait()
    } else {
        a.baseline.gait()
    };

    let mut s = Sim::default();
    s.reset(&a.terrain, &gait, &a.phys);
    let mut m = TorqueMeter::default();
    let steps = (8.0 / hexapod_core::DT) as usize;
    for _ in 0..steps {
        s.step(&a.terrain, policy, &gait, hexapod_core::DT, Cmd::at(a.cruise));
        m.observe(&s, &a.build);
        if s.fallen {
            break;
        }
    }

    let k = m.peak_kgcm();
    a.torque_buf[0] = k[0] as f32;
    a.torque_buf[1] = k[1] as f32;
    a.torque_buf[2] = k[2] as f32;
    a.torque_buf[3] = m.required_kgcm(&a.build) as f32;
    a.torque_buf[4] = m.peak_foot_load as f32;
    a.torque_buf[5] = gait.cycle as f32;
    a.torque_buf[6] = (gait.stance_w * a.build.scale * 1000.0) as f32;
    a.torque_buf[7] = (gait.body_h * a.build.scale * 1000.0) as f32;
    a.meter = m;
    a.torque_buf.as_ptr()
}

// -------------------------------------------------------------------- step

#[no_mangle]
pub extern "C" fn hx_step(dt: f64, fwd: f64, turn: f64) {
    let a = app();
    a.step(dt, fwd, turn);
    a.publish();
}

// ------------------------------------------------------------ data access

#[no_mangle]
pub extern "C" fn hx_telemetry_ptr() -> *const f32 {
    app().telemetry.as_ptr()
}

#[no_mangle]
pub extern "C" fn hx_telemetry_len() -> u32 {
    T_LEN as u32
}

#[no_mangle]
pub extern "C" fn hx_course_ptr() -> *const f32 {
    app().course_buf.as_ptr()
}

/// Number of obstacles; each is 5 floats `[x0, x1, z0, z1, top]`.
#[no_mangle]
pub extern "C" fn hx_course_len() -> u32 {
    (app().course_buf.len() / 5) as u32
}

#[no_mangle]
pub extern "C" fn hx_route_ptr() -> *const f32 {
    app().route_buf.as_ptr()
}

/// Number of waypoints; each is two floats `[x, z]`.
#[no_mangle]
pub extern "C" fn hx_route_len() -> u32 {
    (app().route_buf.len() / 2) as u32
}

/// Turn the route-following autopilot on or off. With it off the machine only
/// turns when it is told to, which is what the arrow keys do anyway.
#[no_mangle]
pub extern "C" fn hx_set_nav(on: u32) {
    app().nav = on != 0;
}

#[no_mangle]
pub extern "C" fn hx_course_count() -> u32 {
    hexapod_core::terrain::COURSES.len() as u32
}

#[no_mangle]
pub extern "C" fn hx_curve_ptr() -> *const f32 {
    app().trainer.curve.as_ptr()
}

#[no_mangle]
pub extern "C" fn hx_curve_len() -> u32 {
    app().trainer.curve.len() as u32
}

#[no_mangle]
pub extern "C" fn hx_dist_curve_ptr() -> *const f32 {
    app().trainer.dist_curve.as_ptr()
}

impl App {
    fn active_gait(&self) -> Gait {
        if self.mode == MODE_LEARNED && self.trained {
            self.learned.gait()
        } else {
            self.baseline.gait()
        }
    }

    fn reset_live(&mut self) {
        self.live_gait = self.active_gait();
        self.live.reset(&self.terrain, &self.live_gait, &self.phys);
        self.since_fall = 0.0;
    }

    fn servo(&self) -> Option<&'static Servo> {
        self.servo.map(|i| &SERVOS[i])
    }

    /// Rebuild the physics from the current build and servo. Anything already
    /// learned was learned for a different machine, so it is discarded.
    fn refresh_physics(&mut self) {
        self.phys = self.build.physics(self.servo());
        self.reset_training();
        self.reset_live();
    }

    fn reset_training(&mut self) {
        self.trainer = Trainer::new(
            Policy::seeded(self.preset, self.frame),
            self.trainer.cfg,
            self.phys,
            self.course_seed ^ 0xA5A5,
        );
        self.trained = false;
        self.learned = self.baseline.clone();
    }

    fn step(&mut self, dt: f64, fwd: f64, turn: f64) {
        // Auto-recover so the viewport never gets stuck on a fallen robot.
        if self.live.fallen || self.live.broken || self.live.pos[2] > Z_MAX - 8.0 {
            self.since_fall += dt;
            if self.since_fall > 1.2 {
                self.reset_live();
            }
            return;
        }

        // Steering goes back to the policy the moment nobody is asking for a
        // turn, so the route is followed by default and overridden on demand.
        let turn = turn.clamp(-1.0, 1.0);
        let cmd = Cmd {
            fwd: fwd.clamp(-1.0, 1.0),
            turn,
            cruise: self.cruise,
            nav: self.nav && turn.abs() < 0.02,
        };
        let policy: &Policy = if self.mode == MODE_LEARNED && self.trained {
            &self.learned
        } else {
            &self.baseline
        };
        // Substep so a long frame cannot destabilise the integrator.
        let n = ((dt / hexapod_core::DT).round() as usize).clamp(1, 6);
        let h = dt / n as f64;
        for _ in 0..n {
            self.live.step(&self.terrain, policy, &self.live_gait, h, cmd);
        }
    }

    fn publish(&mut self) {
        let t = &mut self.telemetry;
        let s = &self.live;

        for i in 0..3 {
            t[T_POS + i] = s.pos[i] as f32;
        }
        t[T_YAW] = s.yaw as f32;
        t[T_PITCH] = s.pitch as f32;
        t[T_ROLL] = s.roll as f32;

        let n = self.frame.legs();
        for leg in 0..n {
            for p in 0..4 {
                for c in 0..3 {
                    t[T_JOINTS + leg * 12 + p * 3 + c] = s.joints[leg][p][c] as f32;
                }
            }
            t[T_STANCE + leg] = if s.feet[leg].stance { 1.0 } else { 0.0 };
            t[T_LOAD + leg] = s.feet[leg].load as f32;
            t[T_LEGPHASE + leg] = s.feet[leg].leg_phase as f32;
            t[T_STEPH + leg] = s.feet[leg].step_h as f32;
            t[T_LEG_LOAD + leg] = s.feet[leg].load_frac as f32;
            for c in 0..3 {
                t[T_TD + leg * 3 + c] = s.feet[leg].td[c] as f32;
                t[T_Q + leg * 3 + c] = s.q[leg][c] as f32;
                t[T_QCMD + leg * 3 + c] = s.q_cmd[leg][c] as f32;
            }
        }

        for i in 0..MAX_LEGS {
            t[T_HULL + i * 2] = s.hull[i][0] as f32;
            t[T_HULL + i * 2 + 1] = s.hull[i][1] as f32;
        }
        t[T_HULL_N] = s.hull_n as f32;
        t[T_LEGS] = n as f32;
        t[T_BODY_R] = self.frame.body_r() as f32;

        t[T_PHASE] = s.phase as f32;
        t[T_MARGIN] = s.margin as f32;
        t[T_DIST] = s.dist as f32;
        t[T_SPEED] = s.speed as f32;
        t[T_POWER] = s.power as f32;
        t[T_TIME] = s.t as f32;
        t[T_FALLEN] = if s.fallen { 1.0 } else { 0.0 };
        t[T_BLOCKED] = if s.blocked { 1.0 } else { 0.0 };
        t[T_ADVANCE] = s.advance_frac as f32;
        t[T_COM] = s.com_drift[0] as f32;
        t[T_COM + 1] = s.com_drift[1] as f32;
        t[T_STUB] = s.stub_total as f32;
        t[T_COLLISIONS] = s.collisions as f32;
        for i in 0..3 {
            t[T_PLANE + i] = s.plane[i] as f32;
        }

        let g = self.live_gait;
        t[T_GAIT] = g.cycle as f32;
        t[T_GAIT + 1] = g.stride as f32;
        t[T_GAIT + 2] = g.step_h as f32;
        t[T_GAIT + 3] = g.body_h as f32;
        t[T_GAIT + 4] = g.stance_w as f32;
        t[T_GAIT + 5] = g.duty as f32;
        for i in 0..n {
            t[T_OFFSETS + i] = g.offsets[i] as f32;
        }

        let tr = &self.trainer;
        t[T_ITER] = tr.iter as f32;
        t[T_ROLLOUTS] = tr.rollouts as f32;
        t[T_BEST_R] = if tr.best_reward.is_finite() {
            tr.best_reward as f32
        } else {
            0.0
        };
        t[T_BASE_R] = tr.baseline_reward as f32;
        t[T_BEST_D] = tr.best_distance as f32;
        t[T_BASE_D] = tr.baseline_distance as f32;
        t[T_EVAL_R] = tr.last_eval.reward as f32;
        t[T_EVAL_D] = tr.last_eval.distance as f32;
        t[T_EVAL_FELL] = if tr.last_eval.fell { 1.0 } else { 0.0 };
        t[T_FEEDBACK] = if self.trained {
            self.learned.feedback_norm() as f32
        } else {
            0.0
        };
        t[T_CURVE_N] = tr.curve.len() as f32;
        t[T_MODE] = self.mode as f32;
        t[T_TRAINED] = if self.trained { 1.0 } else { 0.0 };
        t[T_EVAL_VERR] = tr.last_eval.speed_error as f32;
        t[T_EVAL_COT] = tr.last_eval.cot as f32;

        t[T_CMD_SPEED] = self.cruise as f32;
        t[T_VEL] = s.vel[0] as f32;
        t[T_VEL + 1] = s.vel[1] as f32;
        t[T_ACCEL] = s.accel[0] as f32;
        t[T_ACCEL + 1] = s.accel[1] as f32;
        t[T_SLIP] = s.slip_total as f32;
        t[T_SLIP_RATE] = s.slip as f32;
        t[T_TRACTION] = s.traction as f32;
        t[T_SERVO_LOAD] = s.servo_load as f32;
        t[T_SERVO_LAG] = s.servo_lag as f32;
        t[T_DROOP] = s.droop as f32;
        t[T_TORQUE_PEAK] = s.torque_peak as f32;
        t[T_WORK] = s.work as f32;
        t[T_COT] = s.cost_of_transport() as f32;
        t[T_CYCLE_NOW] = s.cycle_now as f32;
        t[T_STRIDE_NOW] = s.stride_now as f32;
        t[T_DUTY_NOW] = s.duty_now as f32;
        t[T_MU] = self.phys.mu as f32;
        t[T_STALL] = (self.phys.actuator.stall_nm * NM_TO_KGCM) as f32;
        t[T_NOLOAD_RPM] =
            (self.phys.actuator.omega_max * 60.0 / core::f64::consts::TAU) as f32;
        t[T_LEG_TORQUE] = s.leg_torque as f32;
        for c in 0..3 {
            t[T_LEG_REACT + c] = s.leg_react[c] as f32;
        }
        t[T_LEG_KG] = self.phys.leg.total() as f32;

        let w = self.terrain.waypoint(s.wp);
        t[T_WP] = w[0] as f32;
        t[T_WP + 1] = w[1] as f32;
        t[T_WP_I] = s.wp as f32;
        t[T_WP_N] = self.terrain.waypoints.len() as f32;
        t[T_WP_DIST] = s.wp_dist as f32;
        t[T_BEARING] = s.bearing as f32;
        t[T_REACHED] = s.reached as f32;
        t[T_STEER] = s.steer as f32;
        t[T_NAV] = f32::from(u8::from(self.nav));
        t[T_WALL_X] = self.terrain.wall_x() as f32;
        let scan = hexapod_core::policy::obs_scan(self.frame);
        for i in 0..hexapod_core::policy::N_SCAN {
            t[T_SCAN + i] = s.obs[scan + i] as f32;
        }

        t[T_VY] = s.vy as f32;
        t[T_AIRBORNE] = if s.airborne { 1.0 } else { 0.0 };
        t[T_APEX] = s.apex as f32;
        t[T_HOP_APEX] = s.hop_apex as f32;
        t[T_BROKEN] = if s.broken { 1.0 } else { 0.0 };
        t[T_IMPACT] = s.impact_g as f32;
        t[T_JUMPS] = s.jumps as f32;
        t[T_TASK] = if s.jump_clock > 0.0 { 1.0 } else { 0.0 };
        t[T_CLEARANCE] = s.clearance as f32;
    }
}

// --------------------------------------------------- parameter introspection

/// Lower bound of gait scalar `idx`, so the UI never has to duplicate the
/// ranges declared in the core.
#[no_mangle]
pub extern "C" fn hx_param_lo(idx: u32) -> f64 {
    GAIT_BOUNDS.get(idx as usize).map(|b| b.0).unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn hx_param_hi(idx: u32) -> f64 {
    GAIT_BOUNDS.get(idx as usize).map(|b| b.1).unwrap_or(1.0)
}

/// Gait value `idx` for a given policy: `0..6` are the scalars in
/// `GAIT_BOUNDS` order, `6..12` are the per-leg phase offsets.
/// `mode` 0 selects the hand-tuned policy, 1 the learned one.
#[no_mangle]
pub extern "C" fn hx_gait(mode: u32, idx: u32) -> f64 {
    let a = app();
    let g = if mode == MODE_LEARNED && a.trained {
        a.learned.gait()
    } else {
        a.baseline.gait()
    };
    match idx {
        0 => g.cycle,
        1 => g.stride,
        2 => g.step_h,
        3 => g.body_h,
        4 => g.stance_w,
        5 => g.duty,
        6..=11 => g.offsets[(idx - 6) as usize],
        _ => 0.0,
    }
}

/// Terrain height at a point, for drawing the elevation profile.
#[no_mangle]
pub extern "C" fn hx_height(x: f64, z: f64) -> f64 {
    app().terrain.height(x, z)
}

// --------------------------------------------------------- system sizing

use hexapod_core::power::{parts_of, solve, Kind, Part, Sizing, TorqueTrace, PARTS};

/// Index of a part in the catalogue.
///
/// Matched by name, not by pointer: `PARTS` is a `const`, so each use site may
/// get its own promoted copy of the array and `ptr::eq` compares references
/// into two different allocations. Names are asserted unique in the core.
fn part_index(p: Option<&'static Part>) -> f32 {
    match p {
        Some(p) => PARTS
            .iter()
            .position(|q| q.name == p.name)
            .map(|i| i as f32)
            .unwrap_or(-1.0),
        None => -1.0,
    }
}

#[no_mangle]
pub extern "C" fn hx_set_sizing(chassis_kg: f64, runtime_min: f64, safety: f64) {
    let a = app();
    a.sizing.chassis_kg = chassis_kg.clamp(0.05, 40.0);
    a.sizing.runtime_min = runtime_min.clamp(1.0, 600.0);
    a.sizing.safety = safety.clamp(1.0, 4.0);
}

/// Size a whole machine around servo `servo_idx` and the active gait.
/// Writes the `S_*` layout and returns a pointer to it.
#[no_mangle]
pub extern "C" fn hx_solve_system(servo_idx: u32) -> *const f32 {
    let a = app();
    let servo = match hexapod_core::SERVOS.get(servo_idx as usize) {
        Some(s) => s,
        None => return a.system_buf.as_ptr(),
    };

    // The trace only depends on the gait and the scale, so cache it.
    let policy: Policy = if a.mode == MODE_LEARNED && a.trained {
        a.learned.clone()
    } else {
        a.baseline.clone()
    };
    let trace = TorqueTrace::record(&a.terrain, &policy, &a.phys, 8.0);
    let s = solve(&trace, servo, &a.sizing);

    let b = &mut a.system_buf;
    b[S_CONVERGED] = s.converged as u32 as f32;
    b[S_ALLUP] = s.all_up_kg as f32;
    b[S_SERVO_KG] = s.servo_kg as f32;
    b[S_BATT_KG] = s.battery_kg as f32;
    b[S_CHASSIS_KG] = s.chassis_kg as f32;
    b[S_ELEC_KG] = s.electronics_kg as f32;
    b[S_PEAK_TORQUE] = s.peak_torque_kgcm as f32;
    b[S_REQ_TORQUE] = s.required_kgcm as f32;
    b[S_SERVO_OK] = s.servo_ok as u32 as f32;
    b[S_MEAN_A] = s.mean_amps as f32;
    b[S_PEAK_A] = s.peak_amps as f32;
    b[S_MEAN_SERVO_A] = s.mean_servo_amps as f32;
    b[S_PEAK_SERVO_A] = s.peak_servo_amps as f32;
    b[S_WATTS] = s.mean_watts as f32;
    b[S_REQ_WH] = s.required_wh as f32;
    b[S_RUNTIME] = s.runtime_min as f32;
    b[S_COST] = s.cost_usd as f32;
    b[S_COST_SERVOS] = s.cost_servos as f32;
    b[S_BATTERY_I] = part_index(s.battery);
    b[S_REG_I] = part_index(s.regulator);
    b[S_DRIVER_I] = part_index(s.driver);
    b[S_DRIVER_N] = s.driver_count as f32;
    b[S_COMPUTE_I] = part_index(s.compute);
    b[S_RANGER_I] = part_index(parts_of(Kind::Ranger).next());
    b[S_SUPPORT_I] = part_index(parts_of(Kind::Support).next());
    b[S_IMU_I] = part_index(parts_of(Kind::Imu).next());
    b[S_LOOKAHEAD] = s.sensing.lookahead_m as f32;
    b[S_RATE_HZ] = s.sensing.min_rate_hz as f32;
    b[S_RES_MM] = s.sensing.resolution_mm as f32;
    b[S_RANGERS] = s.sensing.rangers as f32;
    b[S_JOINTS] = trace.joints as f32;
    b[S_CONTACT_BUS] = s.sensing.contact_from_bus as u32 as f32;
    b[S_ITERATIONS] = s.iterations as f32;
    b[S_FAILURE] = if s.converged && s.servo_ok {
        0.0
    } else if !s.converged && s.failure.starts_with("diverged") {
        1.0
    } else if !s.converged {
        2.0
    } else {
        3.0
    };
    b.as_ptr()
}

#[no_mangle]
pub extern "C" fn hx_servo_count() -> u32 {
    hexapod_core::SERVOS.len() as u32
}
