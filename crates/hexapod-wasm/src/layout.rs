//! Offsets into the shared `f32` telemetry buffer.
//!
//! `build.sh` parses this file to generate `web/layout.gen.js`, so the browser
//! and the simulator can never disagree about the layout. Only lines of the
//! form `pub const T_NAME: usize = N;` are extracted — keep it that way.
//!
//! Per-leg blocks are sized for `MAX_LEGS` (10) whatever the machine actually
//! has; `T_LEGS` says how many of the slots are live.

pub const T_POS: usize = 0; // 3: body x, y, z
pub const T_YAW: usize = 3;
pub const T_PITCH: usize = 4;
pub const T_ROLL: usize = 5;

pub const T_JOINTS: usize = 8; // 120: 10 legs x (hip, knee, ankle, foot) x xyz
pub const T_STANCE: usize = 128; // 10: 1 while the foot carries load
pub const T_LOAD: usize = 138; // 10: share of body weight
pub const T_LEGPHASE: usize = 148; // 10
pub const T_STEPH: usize = 158; // 10: commanded step height per leg
pub const T_TD: usize = 168; // 30: predicted touchdown xyz per leg
pub const T_Q: usize = 198; // 30: joint angles, 3 per leg
pub const T_QCMD: usize = 228; // 30: commanded joint angles, 3 per leg
pub const T_LEG_LOAD: usize = 258; // 10: worst joint demand per leg / stall
pub const T_HULL: usize = 268; // 20: support polygon, up to 10 xz pairs
pub const T_HULL_N: usize = 288;

/// How many legs this machine has. Everything above is sized for ten.
pub const T_LEGS: usize = 6;
/// Chassis circumradius, which grows with the leg count.
pub const T_BODY_R: usize = 7;

pub const T_PHASE: usize = 289;
pub const T_MARGIN: usize = 290;
pub const T_DIST: usize = 291;
pub const T_SPEED: usize = 292;
pub const T_POWER: usize = 293;
pub const T_TIME: usize = 294;
pub const T_FALLEN: usize = 295;
pub const T_BLOCKED: usize = 296;
pub const T_ADVANCE: usize = 297;
pub const T_COM: usize = 298; // 2: centre-of-mass drift, x and z (stability)
pub const T_STUB: usize = 300;
pub const T_COLLISIONS: usize = 301;
pub const T_PLANE: usize = 302; // 3: support plane a, b, c

pub const T_GAIT: usize = 305; // 6: cycle, stride, step height, body height, stance width, duty
pub const T_OFFSETS: usize = 311; // 10: per-leg phase offsets

pub const T_ITER: usize = 321;
pub const T_ROLLOUTS: usize = 322;
pub const T_BEST_R: usize = 323;
pub const T_BASE_R: usize = 324;
pub const T_BEST_D: usize = 325;
pub const T_BASE_D: usize = 326;
pub const T_EVAL_R: usize = 327;
pub const T_EVAL_D: usize = 328;
pub const T_FEEDBACK: usize = 329;
pub const T_CURVE_N: usize = 330;
pub const T_MODE: usize = 331; // 0 = hand-tuned baseline, 1 = learned policy
pub const T_TRAINED: usize = 332; // 1 once at least one iteration has run
pub const T_EVAL_FELL: usize = 333;
pub const T_EVAL_VERR: usize = 334;
pub const T_EVAL_COT: usize = 335;

// --- dynamics: contact, actuators, momentum and leg mass -----------------

pub const T_CMD_SPEED: usize = 336;
pub const T_VEL: usize = 337; // 2: world x, z
pub const T_ACCEL: usize = 339; // 2: world x, z
pub const T_SLIP: usize = 341; // metres skidded, cumulative
pub const T_SLIP_RATE: usize = 342; // metres this tick
pub const T_TRACTION: usize = 343; // demanded traction / available traction
pub const T_SERVO_LOAD: usize = 344; // worst joint demand / stall torque
pub const T_SERVO_LAG: usize = 345; // rms joint tracking error, rad
pub const T_DROOP: usize = 346; // chassis sag from servo tracking, metres
pub const T_TORQUE_PEAK: usize = 347; // worst joint torque, N-m
pub const T_WORK: usize = 348; // joules
pub const T_COT: usize = 349; // cost of transport
pub const T_CYCLE_NOW: usize = 350; // live cycle time after modulation
pub const T_STRIDE_NOW: usize = 351; // live stride after modulation
pub const T_DUTY_NOW: usize = 352; // live duty factor after modulation
pub const T_MU: usize = 353; // friction coefficient in force
pub const T_STALL: usize = 354; // servo stall torque, kg-cm
pub const T_NOLOAD_RPM: usize = 355;
pub const T_LEG_TORQUE: usize = 356; // worst torque spent on the leg's own mass
pub const T_LEG_REACT: usize = 357; // 3: swing reaction into the chassis, N
pub const T_LEG_KG: usize = 360; // swinging mass of one leg, kg

// --- navigation: the route, the walls, and what the machine can see ------

pub const T_WP: usize = 361; // 2: xz of the waypoint being chased
pub const T_WP_I: usize = 363; // index of it along the route
pub const T_WP_N: usize = 364; // how many waypoints the course has
pub const T_WP_DIST: usize = 365; // metres to it
pub const T_BEARING: usize = 366; // radians off the nose, positive to the left
pub const T_REACHED: usize = 367; // waypoints actually arrived at
pub const T_STEER: usize = 368; // yaw command in force, -1 to 1
pub const T_NAV: usize = 369; // 1 while the policy is steering itself
pub const T_WALL_X: usize = 370; // where the two invisible walls are
pub const T_SCAN: usize = 371; // 6: forward terrain scan, near/far x left/mid/right

pub const T_VY: usize = 377; // vertical velocity, m/s
pub const T_AIRBORNE: usize = 378;
pub const T_APEX: usize = 379; // best extra clearance this episode
pub const T_HOP_APEX: usize = 380; // current/last hop
pub const T_BROKEN: usize = 381;
pub const T_IMPACT: usize = 382; // peak landing demand, g
pub const T_JUMPS: usize = 383;
pub const T_TASK: usize = 384; // 1 while a hop is in progress
pub const T_CLEARANCE: usize = 385;
/// 1 while the live view is the Rapier articulated plant.
pub const T_PLANT: usize = 386;
pub const T_N_HINGES: usize = 387;
/// World xyz of the mass-weighted centre of mass of the drawn robot.
pub const T_COM3: usize = 388; // 3

/// Always 0. The drill that owned this slot is gone; the offset stays so the
/// page's telemetry layout does not have to be renumbered.
pub const T_ONELEG: usize = 391;
/// Index of the free leg.
pub const T_MOVE_LEG: usize = 392;
/// 0 settle, 1 lift, 2 shift, 3 place, 4 pause.
pub const T_MOVE_PHASE: usize = 393;
/// How many plants the free leg has completed.
pub const T_MOVE_I: usize = 394;
/// Furthest a stance foot has slid from its plant at the start of this move.
pub const T_STANCE_DRIFT: usize = 395;
/// Chassis travel in xz since the start of this move.
pub const T_CHASSIS_XZ: usize = 396;
/// Moving-foot clearance above the floor, metres (centre minus sole radius).
pub const T_FOOT_CLEAR: usize = 397;
/// World xyz of each foot at the start of the current move. 10 legs × 3.
pub const T_ORIGIN: usize = 398;
/// World xyz the free foot is aiming at.
pub const T_DEST: usize = 428; // 3

pub const T_LEN: usize = 432;

// --- system-sizing result buffer, written by hx_solve_system --------------

pub const S_CONVERGED: usize = 0;
pub const S_ALLUP: usize = 1;
pub const S_SERVO_KG: usize = 2;
pub const S_BATT_KG: usize = 3;
pub const S_CHASSIS_KG: usize = 4;
pub const S_ELEC_KG: usize = 5;
pub const S_PEAK_TORQUE: usize = 6;
pub const S_REQ_TORQUE: usize = 7;
pub const S_SERVO_OK: usize = 8;
pub const S_MEAN_A: usize = 9;
pub const S_PEAK_A: usize = 10;
pub const S_MEAN_SERVO_A: usize = 11;
pub const S_PEAK_SERVO_A: usize = 12;
pub const S_WATTS: usize = 13;
pub const S_REQ_WH: usize = 14;
pub const S_RUNTIME: usize = 15;
pub const S_COST: usize = 16;
pub const S_COST_SERVOS: usize = 17;
pub const S_BATTERY_I: usize = 18;
pub const S_REG_I: usize = 19;
pub const S_DRIVER_I: usize = 20;
pub const S_DRIVER_N: usize = 21;
pub const S_COMPUTE_I: usize = 22;
pub const S_RANGER_I: usize = 23;
pub const S_SUPPORT_I: usize = 24;
pub const S_IMU_I: usize = 25;
pub const S_LOOKAHEAD: usize = 26;
pub const S_RATE_HZ: usize = 27;
pub const S_RES_MM: usize = 28;
pub const S_RANGERS: usize = 29;
pub const S_CONTACT_BUS: usize = 30;
pub const S_ITERATIONS: usize = 31;
/// 0 none, 1 diverged, 2 no pack fits, 3 under-torqued.
pub const S_FAILURE: usize = 32;
/// Servos on this machine: three per leg.
pub const S_JOINTS: usize = 33;
pub const S_LEN: usize = 36;
