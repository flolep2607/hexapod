//! The simulator.
//!
//! A *centroidal rigid-body* model: the chassis carries linear and angular
//! momentum, the feet transmit bounded Coulomb friction, and the joints are
//! driven by a servo with a real torque-speed line. The three things that used
//! to be free are not any more.
//!
//! **Contact.** A stance foot can transmit at most `mu * N` horizontally. Past
//! that it skids, the body keeps the momentum it already had, and the contact
//! point moves in the world — corrupting the support polygon and the plane fit
//! for every tick that follows. Ground grip varies: loose rubble is worth about
//! two thirds of firm ground.
//!
//! **Actuators.** Joints are integrated toward the inverse-kinematics solution
//! at a rate that falls linearly to zero at stall, and past stall they give way
//! and the leg folds. The torque and speed numbers come from the servo chosen
//! in the hardware catalogue, so the gait you train is a gait for that servo.
//!
//! **Momentum.** The body cannot change speed faster than traction allows, it
//! coasts when a foot slips, and its own acceleration throws the centre of mass
//! around — braking hard pitches the mass forward over the toes, which is what
//! really tips a legged robot that stops in a hurry.
//!
//! **Leg mass.** A swinging leg holds itself up against gravity and has to be
//! accelerated and stopped, and its joints pay for both in torque. The reaction
//! goes into the chassis and eats the same friction budget everything else
//! does. Swing used to be free.
//!
//! The frame is parametric in leg count: four legs to ten, in pairs. Nothing
//! in here knows it is a hexapod.
//!
//! What is still modelled kinematically: links are rigid, contact is resolved
//! once per tick rather than by an impulse solver, and the leg-inertia terms
//! are diagonal — no off-diagonal mass coupling, no Coriolis.

use crate::dynamics::{
    collapse_direction, joint_torques, leg_com, leg_torques, LegState, Physics, G,
};
use crate::math::{ang_diff, clamp, frac, hypot2, polygon_margin, rot_y, V3};
use crate::policy::{
    act_body_dh, act_cycle, act_duty, act_pitch, act_steer, act_stride, n_obs, obs_bearing,
    obs_cmd_speed, obs_corridor, obs_range, obs_scan, Gait, Policy, MAX_ACT, MAX_OBS,
    N_FIXED_OBS, N_SCAN,
};
use crate::robot::{
    clamp_joints, fk_body, fk_world, solve_ik, to_body, Frame, BODY_H, FEMUR, MAX_LEGS,
    TIBIA,
};
use crate::terrain::{Terrain, WAYPOINT_R};

/// Simulation tick. 100 Hz keeps swing arcs smooth without making rollouts
/// expensive, and matches the control rate of a real servo bus.
pub const DT: f64 = 1.0 / 100.0;

/// Clearance the foot keeps above terrain before it counts as catching.
const FOOT_CLEAR: f64 = 0.03;
/// How strongly the chassis conforms to the slope under it.
const COMPLIANCE: f64 = 0.55;
/// Peak yaw rate at full turn command, rad/s.
pub const TURN_RATE: f64 = 1.1;

/// Speeds the robot can be commanded to hold, m/s. Training samples uniformly
/// from this range, so there is no single speed to specialise on.
pub const CRUISE_MIN: f64 = 1.5;
pub const CRUISE_MAX: f64 = 6.0;
/// What the dashboard's speed dial starts at.
pub const CRUISE_DEFAULT: f64 = 4.0;
/// Speeds the evaluation rollout averages over. Slow, middling and fast: a
/// policy has to be able to do all three to score.
pub const EVAL_SPEEDS: [f64; 3] = [2.0, 4.0, 5.5];

/// Body-velocity controller bandwidth, 1/s. The gait asks for a speed; this is
/// how hard it asks. Anything traction cannot deliver becomes slip.
const ACCEL_GAIN: f64 = 4.0;
/// Yaw-rate controller bandwidth, 1/s.
const YAW_GAIN: f64 = 6.0;
/// Vertical leg stiffness and damping, 1/s^2 and 1/s.
const KP_Y: f64 = 900.0;
const KD_Y: f64 = 58.0;
/// Attitude stiffness and damping.
const KP_A: f64 = 260.0;
const KD_A: f64 = 26.0;
/// Ceiling on a leg's centre-of-mass acceleration, m/s^2. Roughly six g,
/// which is past anything a hobby servo can do to a leg and well short of the
/// numbers a discrete contact event produces.
const A_LEG_MAX: f64 = 60.0;
/// Smoothing constant for the swing reaction, seconds.
const REACT_TAU: f64 = 0.03;
/// Highest a foot can be planted above the support plane. A leg has finite
/// reach, so anything taller than this is not a step — it is a wall, and the
/// foot has to go somewhere else.
const MAX_FOOTHOLD: f64 = 0.62;
/// Radii searched, in order, when the intended touchdown is unreachable.
const FOOTHOLD_R: [f64; 2] = [0.35, 0.70];
/// Forward terrain scan: ranges ahead of the body, and bearings across it.
const SCAN_AHEAD: [f64; 2] = [1.4, 3.0];
const SCAN_SIDE: [f64; 3] = [-1.1, 0.0, 1.1];
/// How much of the modulation range one unit of policy action is worth.
const CYCLE_TRIM: f64 = 0.45;
const STRIDE_TRIM: f64 = 0.45;
const DUTY_TRIM: f64 = 0.35;

// --- reward ---------------------------------------------------------------
//
// Everything below is a rate, multiplied by `dt`. There is deliberately no
// distance term: the task is to *hold a commanded speed*, and distance is what
// happens when you succeed.

const W_TRACK: f64 = 12.0;
const W_LAT: f64 = 1.5;
const W_YAW: f64 = 0.6;
const W_MARGIN: f64 = 6.0;
/// Per radian of bearing error to the next waypoint. Small: the machine is
/// asked to *get there*, not to point at it every instant, and a detour round
/// something in the way has to stay cheaper than walking into it.
const W_HEADING: f64 = 2.0;
/// Per metre closed on the current waypoint — and never more than the metres
/// the *commanded* speed would have covered anyway. That cap is what keeps it
/// from being the distance term this reward was written to avoid: past the
/// command there is nothing more to earn, so all it can pay for is pointing
/// the right way.
const W_PROGRESS: f64 = 1.2;
/// Each waypoint reached. Small on purpose — arrivals come at whatever rate
/// the route is spaced at, so a large one would be mileage again.
const W_WAYPOINT: f64 = 5.0;
/// Per joule of mechanical work, so this is a real energy price.
const W_WORK: f64 = 0.020;
/// Per metre of foot skid.
const W_SLIP: f64 = 9.0;
/// Per metre a swinging foot was pushed up by terrain it clipped.
const W_STUB: f64 = 2.2;
const W_COLL: f64 = 15.0;
/// Per second spent with a joint asking for more than its servo has.
const W_OVERLOAD: f64 = 3.0;
const ALIVE: f64 = 0.4;
/// A fall has to cost real money. The forfeited remainder of the rollout is
/// not enough on its own — go over in the last second and you have already
/// banked most of the run — so going over is priced at roughly a third of a
/// clean rollout, which is about what it costs on a real machine.
const FALL_PENALTY: f64 = 40.0;

/// Width of the speed-tracking Gaussian: a fixed floor plus a share of the
/// command, so a 2 m/s request and a 6 m/s request are graded comparably
/// rather than the fast one being forgiven for the same absolute error.
///
/// It has to be tight. Make it generous and holding the command becomes free,
/// the tracking term turns into a constant, and the optimiser goes back to
/// spending all its effort somewhere else.
#[inline]
fn track_width(target: f64) -> f64 {
    0.18 + 0.12 * target.abs()
}

#[derive(Clone, Copy, Debug)]
pub struct Cmd {
    /// Forward throttle in [-1, 1].
    pub fwd: f64,
    /// Yaw command in [-1, 1]. Ignored while `nav` is set.
    pub turn: f64,
    /// Cruise speed at full throttle, m/s.
    pub cruise: f64,
    /// Let the policy steer itself along the course's route. Training always
    /// does; the dashboard hands control back the moment someone touches the
    /// turn keys.
    pub nav: bool,
}

impl Default for Cmd {
    fn default() -> Self {
        Cmd {
            fwd: 0.0,
            turn: 0.0,
            cruise: CRUISE_DEFAULT,
            nav: true,
        }
    }
}

impl Cmd {
    pub fn forward() -> Cmd {
        Cmd {
            fwd: 1.0,
            ..Cmd::default()
        }
    }

    /// Full throttle at a specific commanded speed.
    pub fn at(speed: f64) -> Cmd {
        Cmd {
            fwd: 1.0,
            cruise: speed,
            ..Cmd::default()
        }
    }

    /// Commanded ground speed, m/s, signed.
    #[inline]
    pub fn speed(&self) -> f64 {
        self.cruise * self.fwd
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Foot {
    pub world: V3,
    /// Ground contact point. Fixed while the foot grips; it moves when the
    /// foot skids.
    pub plant: V3,
    pub lift_from: V3,
    /// Predicted or actual touchdown point.
    pub td: V3,
    pub stance: bool,
    pub leg_phase: f64,
    pub load: f64,
    /// How far this foot was pushed up by terrain this step.
    pub stub: f64,
    /// Commanded step height after the policy's per-leg adjustment.
    pub step_h: f64,
    /// Friction multiplier of the surface under the contact point, sampled at
    /// touchdown.
    pub grip: f64,
    /// Ground gradient at the contact point, `(dy/dx, dy/dz)`.
    pub slope: [f64; 2],
    /// Worst fraction of stall torque any joint in this leg is asking for.
    pub load_frac: f64,
}

#[derive(Clone)]
pub struct Sim {
    pub frame: Frame,
    pub phys: Physics,

    pub t: f64,
    pub phase: f64,
    pub pos: V3,
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,

    /// Horizontal velocity in world space, m/s.
    pub vel: [f64; 2],
    /// Vertical velocity, m/s.
    pub vy: f64,
    /// Horizontal acceleration actually achieved this tick, m/s^2.
    pub accel: [f64; 2],
    pub yaw_rate: f64,
    pub pitch_rate: f64,
    pub roll_rate: f64,

    pub feet: [Foot; MAX_LEGS],
    /// Actual joint angles, as the servos have managed to reach them.
    pub q: [[f64; 3]; MAX_LEGS],
    /// What the inverse kinematics asked for.
    pub q_cmd: [[f64; 3]; MAX_LEGS],
    pub joints: [[V3; 4]; MAX_LEGS],

    /// Least-squares plane through the planted feet: `y = a*x + b*z + c`.
    pub plane: [f64; 3],
    pub hull: [[f64; 2]; MAX_LEGS],
    pub hull_n: usize,
    pub margin: f64,

    /// Centre-of-mass excursion away from the geometric centre.
    pub com_drift: [f64; 2],
    com_vel: [f64; 2],
    unstable_for: f64,
    /// Load-weighted stance tracking error carried over from the last tick.
    /// Only the *change* in it moves the chassis: a steady tracking error is a
    /// steady offset, not a continuous drag.
    prev_track_err: [f64; 3],
    /// Last tick's leg centres of mass and joint rates, for the finite
    /// differences the leg-inertia model runs on.
    legs: LegState,

    pub fallen: bool,
    pub blocked: bool,
    pub advance_frac: f64,

    /// Index of the waypoint being chased, and how many have been reached.
    pub wp: usize,
    pub reached: usize,
    /// Range and signed bearing to that waypoint: bearing is positive when it
    /// is to the left of where the machine is pointing.
    pub wp_dist: f64,
    pub bearing: f64,
    /// Yaw command in force, whoever produced it.
    pub steer: f64,

    pub start_z: f64,
    pub dist: f64,
    pub speed: f64,
    /// Mechanical work done at the joints, joules.
    pub work: f64,
    /// Mechanical power at the joints, watts, lightly smoothed.
    pub power: f64,
    /// Total distance the feet have skidded, metres.
    pub slip_total: f64,
    /// Skid this tick, metres.
    pub slip: f64,
    /// Traction demanded divided by traction available. Below 1 the feet are
    /// gripping with room to spare; at 1 they are on the point of letting go;
    /// above 1 they are skidding.
    pub traction: f64,
    /// Worst joint demand this tick, as a fraction of stall torque.
    pub servo_load: f64,
    /// RMS joint tracking error, radians.
    pub servo_lag: f64,
    /// How far the chassis is sitting below where the legs were told to hold
    /// it, metres. Servo sag, in other words.
    pub droop: f64,
    /// Worst joint torque this tick, newton-metres.
    pub torque_peak: f64,
    /// Worst torque a joint spent on the leg's own weight and inertia rather
    /// than on the load underfoot, newton-metres.
    pub leg_torque: f64,
    /// Force the swinging legs are pushing back into the chassis, newtons.
    pub leg_react: V3,
    /// Live cycle time, stride and duty factor after the policy's modulation.
    pub cycle_now: f64,
    pub stride_now: f64,
    pub duty_now: f64,
    pub cmd_speed: f64,

    pub stub_total: f64,
    pub collisions: f64,

    pub obs: [f64; MAX_OBS],
    pub act: [f64; MAX_ACT],
}

impl Default for Sim {
    fn default() -> Self {
        Sim {
            frame: Frame::default(),
            phys: Physics::default(),
            t: 0.0,
            phase: 0.0,
            pos: [0.0, 0.9, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            vel: [0.0; 2],
            vy: 0.0,
            accel: [0.0; 2],
            yaw_rate: 0.0,
            pitch_rate: 0.0,
            roll_rate: 0.0,
            feet: [Foot::default(); MAX_LEGS],
            q: [[0.0; 3]; MAX_LEGS],
            q_cmd: [[0.0; 3]; MAX_LEGS],
            joints: [[[0.0; 3]; 4]; MAX_LEGS],
            plane: [0.0, 0.0, 0.0],
            hull: [[0.0; 2]; MAX_LEGS],
            hull_n: 0,
            margin: 0.0,
            com_drift: [0.0; 2],
            com_vel: [0.0; 2],
            unstable_for: 0.0,
            prev_track_err: [0.0; 3],
            legs: LegState::default(),
            fallen: false,
            blocked: false,
            advance_frac: 1.0,
            wp: 0,
            reached: 0,
            wp_dist: 0.0,
            bearing: 0.0,
            steer: 0.0,
            start_z: 0.0,
            dist: 0.0,
            speed: 0.0,
            work: 0.0,
            power: 0.0,
            slip_total: 0.0,
            slip: 0.0,
            traction: 1.0,
            servo_load: 0.0,
            servo_lag: 0.0,
            droop: 0.0,
            torque_peak: 0.0,
            leg_torque: 0.0,
            leg_react: [0.0; 3],
            cycle_now: 0.5,
            stride_now: 1.0,
            duty_now: 0.5,
            cmd_speed: 0.0,
            stub_total: 0.0,
            collisions: 0.0,
            obs: [0.0; MAX_OBS],
            act: [0.0; MAX_ACT],
        }
    }
}

impl Sim {
    pub fn reset(&mut self, terrain: &Terrain, gait: &Gait, phys: &Physics) {
        let start = [0.0, 0.0, 0.0];
        let ground = terrain.height(start[0], start[2]);
        let n = gait.frame.legs();

        *self = Sim {
            frame: gait.frame,
            phys: *phys,
            pos: [start[0], ground + gait.body_h, start[2]],
            start_z: start[2],
            plane: [0.0, 0.0, ground],
            cycle_now: gait.cycle,
            stride_now: gait.stride,
            duty_now: gait.duty,
            ..Sim::default()
        };

        for i in 0..n {
            let lp = frac(gait.offsets[i]);
            let d = self.frame.dir(i);
            let out = gait.stance_w * 0.5 + gait.trim(i);
            let wx = self.pos[0] + d[0] * out;
            let wz = self.pos[2] + d[2] * out;
            let wy = terrain.height(wx, wz);
            let p = [wx, wy, wz];
            let (gx, gz) = terrain.slope(wx, wz);
            self.feet[i] = Foot {
                world: p,
                plant: p,
                lift_from: p,
                td: p,
                stance: lp < gait.duty,
                leg_phase: lp,
                load: if lp < gait.duty { 1.0 / 3.0 } else { 0.0 },
                stub: 0.0,
                step_h: gait.step_h,
                grip: terrain.grip(wx, wz),
                slope: [gx, gz],
                load_frac: 0.0,
            };
        }
        // Start the joints exactly where the kinematics wants them, so the
        // servo model has nothing to catch up on at t = 0. The leg-inertia
        // finite differences stay unprimed for one tick so the first step is
        // not billed for an acceleration out of nowhere.
        for i in 0..n {
            let tb = to_body(self.feet[i].world, self.pos, self.yaw, self.pitch, self.roll);
            self.q[i] = solve_ik(self.frame, i, tb).q;
            self.q_cmd[i] = self.q[i];
            self.joints[i] = fk_world(self.frame, i, self.q[i], self.pos, self.yaw, self.pitch, self.roll);
        }
        self.update_support();
        self.update_route(terrain);
    }

    /// One simulation tick. Returns the reward earned during it.
    pub fn step(
        &mut self,
        terrain: &Terrain,
        policy: &Policy,
        gait: &Gait,
        dt: f64,
        cmd: Cmd,
    ) -> f64 {
        if self.fallen {
            return 0.0;
        }
        let n = self.frame.legs();

        self.build_obs(terrain, gait, cmd);
        let mut act = [0.0; MAX_ACT];
        policy.act(&self.obs, &mut act);
        self.act = act;

        // --- the policy modulates frequency and step length -----------------
        //
        // These two are how any legged controller changes speed, and giving
        // them to the policy is what makes a commanded speed trackable at all.
        let cycle = clamp(
            gait.cycle * (1.0 + CYCLE_TRIM * act[act_cycle(self.frame)]),
            crate::policy::GAIT_BOUNDS[0].0,
            crate::policy::GAIT_BOUNDS[0].1,
        );
        let stride = clamp(
            gait.stride * (1.0 + STRIDE_TRIM * act[act_stride(self.frame)]),
            crate::policy::GAIT_BOUNDS[1].0,
            crate::policy::GAIT_BOUNDS[1].1,
        );
        // Duty factor is the third lever, and the one with a right answer:
        // animals raise it as they slow down and drop it as they speed up,
        // because at speed a high duty needs a stride the leg does not have.
        let duty = clamp(
            gait.duty * (1.0 + DUTY_TRIM * act[act_duty(self.frame)]),
            crate::policy::GAIT_BOUNDS[5].0,
            crate::policy::GAIT_BOUNDS[5].1,
        );
        self.cycle_now = cycle;
        self.stride_now = stride;
        self.duty_now = duty;
        self.cmd_speed = cmd.speed();

        // Who is steering. Under `nav` the policy is, which is the only way a
        // route means anything; otherwise the turn command comes from outside.
        self.steer = if cmd.nav {
            clamp(act[act_steer(self.frame)], -1.0, 1.0)
        } else {
            cmd.turn
        };
        let w_cmd = TURN_RATE * self.steer;
        let body_dh = 0.20 * act[act_body_dh(self.frame)];

        // --- gait clock and stance/swing transitions ------------------------
        self.phase = frac(self.phase + dt / cycle);
        for i in 0..n {
            let lp = frac(self.phase + gait.offsets[i]);
            let was = self.feet[i].stance;
            let now = lp < duty;
            self.feet[i].leg_phase = lp;

            if was && !now {
                self.feet[i].lift_from = self.feet[i].world;
            } else if !was && now {
                let td = self.feet[i].td;
                let p = [td[0], terrain.height(td[0], td[2]), td[2]];
                let (gx, gz) = terrain.slope(p[0], p[2]);
                self.feet[i].plant = p;
                self.feet[i].world = p;
                self.feet[i].grip = terrain.grip(p[0], p[2]);
                self.feet[i].slope = [gx, gz];
            }
            self.feet[i].stance = now;
        }

        self.fit_plane();

        // --- what the ground can give ---------------------------------------
        //
        // Available tangential acceleration is `mu * N / m`, summed over the
        // feet that are down, with each foot's grip and the cosine loss from
        // the slope it is standing on.
        //
        // Load share here is the geometric one, `1/n`, not the filtered value
        // used for rendering and torque: the filter lags a tripod swap by a
        // few ticks, and reading traction off it makes the whole machine
        // briefly frictionless every time the feet change over.
        let n_stance = self.feet.iter().filter(|f| f.stance).count();
        let mut mu_n = 0.0;
        if n_stance > 0 {
            let share = 1.0 / n_stance as f64;
            for f in self.feet.iter().filter(|f| f.stance) {
                let g2 = f.slope[0] * f.slope[0] + f.slope[1] * f.slope[1];
                mu_n += self.phys.mu * f.grip * share / (1.0 + g2).sqrt();
            }
        }
        // A leg being flung forward pushes the chassis back, and the normal
        // load under the feet rises and falls with the vertical part of it.
        // This is last tick's value: the reaction cannot be known until the
        // joints have been integrated, which happens further down.
        let m_all = self.phys.mass_kg.max(1e-3);
        let normal_scale = clamp(1.0 + self.leg_react[1] / (m_all * G), 0.2, 1.8);
        let a_avail = mu_n * G * normal_scale;
        let a_react = [self.leg_react[0] / m_all, self.leg_react[2] / m_all];

        // Downslope pull along the support plane.
        let (gx, gz) = (self.plane[0], self.plane[1]);
        let gs = G / (1.0 + gx * gx + gz * gz);
        let a_grav = [-gx * gs, -gz * gs];

        // --- how far the legs can let the body travel ------------------------
        let step_world = [self.vel[0] * dt, 0.0, self.vel[1] * dt];
        let mut k = 1.0;
        if self.strain_at(1.0, step_world, self.yaw_rate * dt, gait, body_dh) > 1e-4 {
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            for _ in 0..7 {
                let mid = 0.5 * (lo + hi);
                if self.strain_at(mid, step_world, self.yaw_rate * dt, gait, body_dh) > 1e-4 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            k = lo;
        }
        self.advance_frac = k;

        // --- translation ------------------------------------------------------
        let fwd = rot_y([0.0, 0.0, 1.0], self.yaw);
        let v_cmd = cmd.speed();
        let mut a_leg = if k < 1.0 {
            // A leg at the end of its envelope is a strut: it asks for whatever
            // deceleration would stop the body this tick, and friction decides
            // how much of that it gets.
            [
                (self.vel[0] * k - self.vel[0]) / dt - a_grav[0],
                (self.vel[1] * k - self.vel[1]) / dt - a_grav[1],
            ]
        } else {
            [
                (fwd[0] * v_cmd - self.vel[0]) * ACCEL_GAIN - a_grav[0],
                (fwd[2] * v_cmd - self.vel[1]) * ACCEL_GAIN - a_grav[1],
            ]
        };
        // Holding station against the swing reaction is work the feet have to
        // do on top of whatever the body is being asked for.
        a_leg[0] -= a_react[0];
        a_leg[1] -= a_react[1];

        let demand = hypot2(a_leg[0], a_leg[1]);
        self.traction = if a_avail > 1e-9 {
            demand / a_avail
        } else if demand > 1e-9 {
            f64::INFINITY
        } else {
            0.0
        };
        self.slip = 0.0;
        if demand > a_avail {
            let s = a_avail / demand;
            // Whatever friction could not deliver, the feet slid instead.
            let skid = (demand - a_avail) * dt * dt;
            let dir = [a_leg[0] / demand, a_leg[1] / demand];
            a_leg = [a_leg[0] * s, a_leg[1] * s];
            self.slip = skid;
            self.slip_total += skid;
            // The foot pushing forward skids backward, and vice versa.
            for f in self.feet.iter_mut() {
                if f.stance {
                    f.plant[0] -= dir[0] * skid;
                    f.plant[2] -= dir[1] * skid;
                }
            }
        }

        self.accel = [a_leg[0] + a_grav[0], a_leg[1] + a_grav[1]];
        if n_stance == 0 {
            // Nothing down: no traction at all, only gravity.
            self.accel = [0.0, 0.0];
        }
        self.vel[0] += self.accel[0] * dt;
        self.vel[1] += self.accel[1] * dt;

        // --- what is in the way -------------------------------------------------
        //
        // The corridor is fenced, and a slalom wall is taller than a leg can
        // reach, so both are places the chassis simply cannot be. Hitting one
        // takes the velocity component into it and keeps the component along
        // it: a machine that arrives at a wall square stops, and one that
        // arrives at an angle slides along and gets round. That difference is
        // the whole reason for having a steering action.
        let r_body = self.frame.body_r() * 0.9;
        let clear = gait.body_h + body_dh - BODY_H * 0.5;
        let (px, pz) = (self.pos[0], self.pos[2]);
        let (dx, dz) = (self.vel[0] * dt, self.vel[1] * dt);
        let collision = if self.chassis_fits(terrain, px + dx, pz + dz, r_body, clear) {
            self.pos[0] = px + dx;
            self.pos[2] = pz + dz;
            0.0
        } else if self.chassis_fits(terrain, px + dx, pz, r_body, clear) {
            self.pos[0] = px + dx;
            self.vel[1] = 0.0;
            1.0
        } else if self.chassis_fits(terrain, px, pz + dz, r_body, clear) {
            self.pos[2] = pz + dz;
            self.vel[0] = 0.0;
            1.0
        } else {
            self.vel = [0.0, 0.0];
            1.0
        };
        self.blocked = collision > 0.0;
        self.collisions += collision;

        // --- yaw ---------------------------------------------------------------
        // Turning is the same traction budget acting on a lever arm of roughly
        // half the stance width.
        let arm = (gait.stance_w * 0.5).max(0.2);
        let alpha_max = a_avail / arm;
        let alpha = clamp((w_cmd - self.yaw_rate) * YAW_GAIN, -alpha_max, alpha_max);
        self.yaw_rate += alpha * dt;
        self.yaw += self.yaw_rate * dt;

        // --- vertical ----------------------------------------------------------
        //
        // The legs cannot hold the chassis higher than they can actually reach,
        // so the servo sag measured last tick comes off the target. Push it in
        // as a position correction instead and the spring simply undoes it,
        // and an undersized servo shows no sag at all.
        let support_y =
            self.plane_y(self.pos[0], self.pos[2]) + gait.body_h + body_dh - self.droop;
        // Ride over what is underneath — but only as high as the legs reach.
        // Without the cap, a chassis that gets a wall under its footprint is
        // lifted onto it, the feet leave the ground, and the whole machine
        // falls off the other side.
        let clear_y = (terrain.height_disc(self.pos[0], self.pos[2], self.frame.body_r() * 0.9)
            + BODY_H * 0.6)
            .min(support_y + MAX_FOOTHOLD);
        let y_target = support_y.max(clear_y);
        let ay = if n_stance > 0 {
            KP_Y * (y_target - self.pos[1]) - KD_Y * self.vy
        } else {
            -G
        };
        self.vy += ay * dt;
        self.pos[1] += self.vy * dt;

        // --- attitude ----------------------------------------------------------
        let target_pitch = (-self.plane[1]).atan() * COMPLIANCE + 0.18 * act[act_pitch(self.frame)];
        let target_roll = self.plane[0].atan() * COMPLIANCE;
        let ap = KP_A * (target_pitch - self.pitch) - KD_A * self.pitch_rate;
        let ar = KP_A * (target_roll - self.roll) - KD_A * self.roll_rate;
        self.pitch_rate += ap * dt;
        self.roll_rate += ar * dt;
        self.pitch += self.pitch_rate * dt;
        self.roll += self.roll_rate * dt;

        // --- where the feet are told to be --------------------------------------
        let t_swing = (1.0 - duty) * cycle;
        let mut want: [V3; MAX_LEGS] = [[0.0; 3]; MAX_LEGS];
        let mut step_stub = 0.0;
        for i in 0..n {
            let lp = self.feet[i].leg_phase;
            let long_off = 0.30 * act[n + i];
            let sh = clamp(gait.step_h * (1.0 + 0.9 * act[i]), 0.05, 0.90);
            self.feet[i].step_h = sh;

            if self.feet[i].stance {
                want[i] = self.feet[i].plant;
                let t_remain = (duty - lp) * cycle + t_swing;
                self.feet[i].td =
                    self.predict_td(terrain, gait, i, t_remain, stride, cycle, duty, long_off);
                continue;
            }

            let u = clamp((lp - duty) / (1.0 - duty), 0.0, 1.0);
            let t_remain = (1.0 - u) * t_swing;
            let td =
                self.predict_td(terrain, gait, i, t_remain, stride, cycle, duty, long_off);
            self.feet[i].td = td;

            let from = self.feet[i].lift_from;
            let x = from[0] + (td[0] - from[0]) * u;
            let z = from[2] + (td[2] - from[2]) * u;
            let arc =
                from[1] + (td[1] - from[1]) * u + sh * (core::f64::consts::PI * u).sin();

            // A swinging foot rides over what it passes — but only as far as
            // the leg reaches. Past that it is not clearing an obstacle, it is
            // jammed against a wall, and the step is charged for it.
            let ground = (terrain.height(x, z) + FOOT_CLEAR)
                .min(self.plane_y(x, z) + MAX_FOOTHOLD);
            let (y, stub) = if arc < ground {
                (ground, ground - arc)
            } else {
                (arc, 0.0)
            };
            self.feet[i].stub = stub;
            step_stub += stub;
            want[i] = [x, y, z];
        }

        // --- the servos ----------------------------------------------------------
        let weight = self.phys.mass_kg * G;
        // Traction ratio is measured now, not assumed: it is the horizontal
        // acceleration the legs are actually producing.
        let traction_ratio = hypot2(a_leg[0], a_leg[1]) / G;
        let mut work_step = 0.0;
        let mut lag2 = 0.0;
        let mut worst_load = 0.0f64;
        let mut worst_tau = 0.0f64;
        let mut overload = 0.0;
        self.leg_torque = 0.0;

        // Reaction from the legs' own mass, summed as it is computed.
        let mut react = [0.0f64; 3];

        for i in 0..n {
            let tb = to_body(want[i], self.pos, self.yaw, self.pitch, self.roll);
            self.q_cmd[i] = solve_ik(self.frame, i, tb).q;

            let jb = fk_body(self.frame, i, self.q[i]);
            let f_v = weight * self.feet[i].load * self.phys.dynamic;
            let f_h = f_v * traction_ratio;
            let mut tau = joint_torques(&jb, f_v, f_h, self.phys.scale);

            // What the leg costs to carry and to swing. Joint accelerations
            // come from differencing the rates the servos actually achieved
            // last tick, so an actuator that could not keep up is not billed
            // for an acceleration it never produced.
            let mut ddq = [0.0f64; 3];
            if self.legs.primed {
                for j in 0..3 {
                    ddq[j] = -self.legs.dq[i][j] / dt;
                }
            }
            let own = leg_torques(&jb, ddq, &self.phys.leg, self.phys.scale);
            for j in 0..3 {
                tau[j] += own[j];
            }
            self.leg_torque = self.leg_torque.max(own[1].max(own[2]));

            // Chassis reaction: the leg's centre of mass is accelerating, and
            // something has to push it. Finite-differenced in the body frame,
            // which is where the leg actually moves.
            let com = leg_com(&jb, &self.phys.leg, self.phys.scale);
            if self.legs.primed {
                for c in 0..3 {
                    let v = (com[c] - self.legs.com[i][c]) / dt;
                    // Contact is resolved discretely, so a foot that lands on
                    // an obstacle can move a centimetre in one tick and the
                    // second difference reads an acceleration no servo could
                    // produce. Cap it at what the joints can actually deliver.
                    let a = clamp((v - self.legs.com_vel[i][c]) / dt, -A_LEG_MAX, A_LEG_MAX);
                    react[c] -= self.phys.leg.total() * a;
                    self.legs.com_vel[i][c] = v;
                }
            }
            self.legs.com[i] = com;

            let collapse = collapse_direction(self.q[i], FEMUR, TIBIA);

            let mut leg_load = 0.0f64;
            for j in 0..3 {
                let load = self.phys.actuator.load(tau[j]);
                leg_load = leg_load.max(load);
                worst_load = worst_load.max(load);
                worst_tau = worst_tau.max(tau[j].abs());

                let omega = self.phys.actuator.rate_limit(tau[j]);
                let e = self.q_cmd[i][j] - self.q[i][j];
                let mut rate = clamp(self.phys.actuator.kp * e, -omega, omega);

                // Past stall the joint gives way in the direction the load is
                // pushing it, which for the two pitch joints is leg collapse.
                if j > 0 {
                    let bd = self.phys.actuator.backdrive(tau[j]);
                    if bd > 0.0 {
                        overload += dt;
                        rate += if collapse[j - 1] >= 0.0 { bd } else { -bd };
                    }
                }

                let dq = rate * dt;
                self.q[i][j] += dq;
                work_step += tau[j].abs() * dq.abs();
                lag2 += e * e;
                ddq[j] += rate / dt; // completes (rate - prev_rate) / dt
                self.legs.dq[i][j] = rate;
            }
            clamp_joints(&mut self.q[i]);
            self.feet[i].load_frac = leg_load;
        }
        self.feet
            .iter_mut()
            .for_each(|f| f.load_frac = f.load_frac.min(9.0));
        self.legs.primed = true;
        // The reaction is in the body frame; the horizontal part has to be
        // pushed through the feet like anything else, so it goes on the
        // traction bill next tick.
        let rw = crate::math::body_to_world(react, self.yaw, self.pitch, self.roll);
        // Smoothed: the chassis has mass of its own and does not respond to a
        // single tick of impulse, and the tick-to-tick noise in a second
        // difference is not a real force.
        let a = 1.0 - (-dt / REACT_TAU).exp();
        for c in 0..3 {
            self.leg_react[c] += (rw[c] - self.leg_react[c]) * a;
        }
        self.servo_load = worst_load;
        self.torque_peak = worst_tau;
        self.servo_lag = (lag2 / (n * 3) as f64).sqrt();
        self.work += work_step;
        self.power += (work_step / dt - self.power) * 0.05;

        for i in 0..n {
            self.joints[i] = fk_world(self.frame, i, self.q[i], self.pos, self.yaw, self.pitch, self.roll);
        }

        // --- the chassis rides on whatever the legs actually managed -------------
        //
        // If the servos are sagging, the feet are higher relative to the body
        // than commanded, which physically means the body is lower. Take the
        // load-weighted mean error over the legs carrying weight and move the
        // chassis by it.
        let mut err = [0.0; 3];
        let mut wsum = 0.0;
        for i in 0..n {
            if !self.feet[i].stance {
                continue;
            }
            let wgt = self.feet[i].load.max(1e-3);
            for c in 0..3 {
                err[c] += (self.joints[i][3][c] - self.feet[i].plant[c]) * wgt;
            }
            wsum += wgt;
        }
        if wsum > 0.0 {
            for c in 0..3 {
                err[c] /= wsum;
            }
        }
        // A joint that is a fixed distance behind its command holds the body a
        // fixed distance from where the kinematics wanted it — it does not push
        // it further away every tick. So the chassis moves by the *change* in
        // the tracking error, which integrates to exactly the current offset.
        let mut shift = [
            err[0] - self.prev_track_err[0],
            0.0,
            err[2] - self.prev_track_err[2],
        ];
        self.prev_track_err = err;
        // Obstruction is a hard constraint, so it has to survive this
        // correction too. A few millimetres a tick is exactly how something
        // seeps through a barrier that is only checked during motion, and a
        // chassis that ends up inside a wall gets lifted onto it by the
        // clearance term below and then falls off.
        if !self.chassis_fits(
            terrain,
            self.pos[0] - shift[0],
            self.pos[2] - shift[2],
            r_body,
            clear,
        ) {
            shift[0] = 0.0;
            shift[2] = 0.0;
        }
        self.pos[0] -= shift[0];
        self.pos[2] -= shift[2];
        for i in 0..n {
            for k4 in 0..4 {
                for c in 0..3 {
                    self.joints[i][k4][c] -= shift[c];
                }
            }
        }
        // Vertically the error is a standing offset, not a per-tick nudge: it
        // feeds the height target above, and it is bounded because the joints
        // run out of travel.
        self.droop = clamp(err[1], -0.5, 0.5);

        // Ground contact is a hard constraint: a foot the servos put below the
        // terrain is stopped by the terrain, and the joints take the difference.
        for i in 0..n {
            let foot = self.joints[i][3];
            let floor = (terrain.height(foot[0], foot[2])
                + if self.feet[i].stance { 0.0 } else { FOOT_CLEAR })
                .min(self.plane_y(foot[0], foot[2]) + MAX_FOOTHOLD);
            if foot[1] < floor - 1e-9 {
                let pen = floor - foot[1];
                if !self.feet[i].stance {
                    self.feet[i].stub += pen;
                    step_stub += pen;
                }
                let tb = to_body(
                    [foot[0], floor, foot[2]],
                    self.pos,
                    self.yaw,
                    self.pitch,
                    self.roll,
                );
                self.q[i] = solve_ik(self.frame, i, tb).q;
                self.joints[i] =
                    fk_world(self.frame, i, self.q[i], self.pos, self.yaw, self.pitch, self.roll);
            }
            self.feet[i].world = self.joints[i][3];
        }
        self.stub_total += step_stub;

        // --- support and stability ------------------------------------------------
        self.update_support();

        // The centre of mass moves relative to the feet under two influences:
        // the downslope pull, and the pseudo-force from the body's own
        // acceleration. Braking throws it forward; that is the momentum term.
        let a_com = [a_grav[0] - self.accel[0], a_grav[1] - self.accel[1]];
        self.com_vel[0] = (self.com_vel[0] + a_com[0] * dt) * 0.88;
        self.com_vel[1] = (self.com_vel[1] + a_com[1] * dt) * 0.88;
        self.com_drift[0] += self.com_vel[0] * dt;
        self.com_drift[1] += self.com_vel[1] * dt;
        if self.margin > 0.05 {
            self.com_drift[0] *= 0.982;
            self.com_drift[1] *= 0.982;
        }

        if self.margin < 0.0 {
            self.unstable_for += dt;
        } else {
            self.unstable_for = (self.unstable_for - dt * 2.0).max(0.0);
        }

        let drift_mag = hypot2(self.com_drift[0], self.com_drift[1]);
        if self.unstable_for > 0.28
            || drift_mag > 0.95
            || self.pitch.abs() > 0.75
            || self.roll.abs() > 0.75
            || self.pos[1] < self.plane_y(self.pos[0], self.pos[2]) - 0.15
        {
            self.fallen = true;
        }

        // --- bookkeeping -------------------------------------------------------
        self.t += dt;
        self.dist = self.pos[2] - self.start_z;
        let ground_speed = hypot2(self.vel[0], self.vel[1]);
        self.speed += (ground_speed - self.speed) * 0.08;

        // How much ground was closed on the waypoint being chased, measured
        // against the same waypoint at both ends so advancing to the next one
        // does not read as a sudden loss.
        let was = self.wp_dist;
        let w = terrain.waypoint(self.wp);
        let closed = was - hypot2(w[0] - self.pos[0], w[1] - self.pos[2]);
        let progress = closed.min(cmd.speed().abs() * dt);
        let before = self.reached;
        self.update_route(terrain);
        let arrivals = (self.reached - before) as f64;

        // --- reward -------------------------------------------------------------
        let target = cmd.speed();
        let along = fwd[0] * self.vel[0] + fwd[2] * self.vel[1];
        let lateral = -fwd[2] * self.vel[0] + fwd[0] * self.vel[1];
        let err_v = (along - target) / track_width(target);
        let track = (-err_v * err_v).exp();

        let mut r = dt
            * (W_TRACK * track - W_LAT * lateral.abs()
                - W_YAW * (self.yaw_rate - w_cmd).abs()
                - W_HEADING * self.bearing.abs()
                + W_MARGIN * clamp(self.margin, 0.0, 0.25)
                - W_COLL * collision
                + ALIVE)
            + W_PROGRESS * progress
            + W_WAYPOINT * arrivals
            - W_WORK * work_step
            - W_SLIP * self.slip
            - W_STUB * step_stub
            - W_OVERLOAD * overload;

        if self.fallen {
            r -= FALL_PENALTY;
        }
        r
    }

    /// Mechanical cost of transport: joules per newton of weight per metre.
    /// Dimensionless, and the standard way to compare legged machines.
    pub fn cost_of_transport(&self) -> f64 {
        let d = self.dist.abs();
        if d < 0.5 {
            return 0.0;
        }
        self.work / (self.phys.mass_kg * G * d)
    }

    // ---------------------------------------------------------------- helpers

    /// Where leg `i` will touch down, `t_remain` seconds from now.
    ///
    /// The foot is placed half a stride ahead of its neutral position, and the
    /// body is projected forward at the speed it is *actually* travelling. If
    /// the stride and the speed disagree the leg spends its stance stroke
    /// off-centre in its workspace, and runs out of reach at one end — which is
    /// the physical penalty for a gait whose parameters do not match its
    /// commanded speed.
    fn predict_td(
        &self,
        terrain: &Terrain,
        gait: &Gait,
        leg: usize,
        t_remain: f64,
        stride: f64,
        cycle: f64,
        duty: f64,
        long_off: f64,
    ) -> V3 {
        let d = self.frame.dir(leg);
        let out = gait.stance_w * 0.5 + gait.trim(leg);
        let neutral = [d[0] * out, -gait.body_h, d[2] * out];

        let w = self.yaw_rate;
        let stroke_t = duty * cycle;
        let ft = [
            neutral[0] + w * neutral[2] * stroke_t * 0.5,
            neutral[1],
            neutral[2] + stride * 0.5 + long_off,
        ];

        let fwd = rot_y([0.0, 0.0, 1.0], self.yaw);
        let v_now = fwd[0] * self.vel[0] + fwd[2] * self.vel[1];
        let yaw_td = self.yaw + w * t_remain;
        let mid = rot_y([0.0, 0.0, 1.0], self.yaw + w * t_remain * 0.5);
        let bx = self.pos[0] + mid[0] * v_now * t_remain;
        let bz = self.pos[2] + mid[2] * v_now * t_remain;

        let r = rot_y(ft, yaw_td);
        let x = bx + r[0];
        let z = bz + r[2];
        self.foothold(terrain, x, z)
    }

    /// The place a foot aimed at `(x, z)` can actually be put.
    ///
    /// Normally that is `(x, z)` itself. Where the course has something too
    /// tall to stand on — a slalom wall is nearly twice the length of a leg —
    /// the step is deflected to the nearest reachable ground, which is what
    /// stops the kinematics from being asked to plant a foot two metres up and
    /// then tipping the whole support plane over when it half succeeds. When
    /// there is nowhere within reach the step lands short and the leg pays the
    /// usual price for stubbing.
    fn foothold(&self, terrain: &Terrain, x: f64, z: f64) -> V3 {
        let y = terrain.height(x, z);
        if y - self.plane_y(x, z) <= MAX_FOOTHOLD {
            return [x, y, z];
        }
        // Search outward, starting in the direction of the body — the leg is
        // more likely to reach a foothold on the near side of the wall.
        let a0 = (self.pos[0] - x).atan2(self.pos[2] - z);
        for r in FOOTHOLD_R {
            for k in 0..8 {
                // 0, +1, -1, +2, ... in eighths of a turn, so the closest
                // bearings to the body are tried first.
                let step = ((k + 1) / 2) as f64 * if k % 2 == 0 { 1.0 } else { -1.0 };
                let a = a0 + step * core::f64::consts::TAU / 8.0;
                let (sx, sz) = (x + r * a.sin(), z + r * a.cos());
                let h = terrain.height(sx, sz);
                if h - self.plane_y(sx, sz) <= MAX_FOOTHOLD {
                    return [sx, h, sz];
                }
            }
        }
        [x, self.plane_y(x, z) + MAX_FOOTHOLD, z]
    }

    /// Worst leg-envelope violation if the chassis advanced by `k * delta`.
    fn strain_at(&self, k: f64, delta: V3, dyaw: f64, gait: &Gait, body_dh: f64) -> f64 {
        let x = self.pos[0] + delta[0] * k;
        let z = self.pos[2] + delta[2] * k;
        let y = self.plane_y(x, z) + gait.body_h + body_dh;
        let yaw = self.yaw + dyaw * k;
        let pos = [x, y, z];

        let mut worst = 0.0f64;
        for i in 0..self.frame.legs() {
            if !self.feet[i].stance {
                continue;
            }
            let target = to_body(self.feet[i].plant, pos, yaw, self.pitch, self.roll);
            let s = solve_ik(self.frame, i, target).strain;
            if s > worst {
                worst = s;
            }
        }
        worst
    }

    /// Least-squares plane through the currently planted feet.
    fn fit_plane(&mut self) {
        let mut n = 0.0;
        let (mut sx, mut sz, mut sy) = (0.0, 0.0, 0.0);
        let (mut sxx, mut szz, mut sxz) = (0.0, 0.0, 0.0);
        let (mut sxy, mut szy) = (0.0, 0.0);
        for f in self.feet.iter() {
            if !f.stance {
                continue;
            }
            let (x, y, z) = (f.plant[0], f.plant[1], f.plant[2]);
            n += 1.0;
            sx += x;
            sz += z;
            sy += y;
            sxx += x * x;
            szz += z * z;
            sxz += x * z;
            sxy += x * y;
            szy += z * y;
        }
        if n < 3.0 {
            return; // keep the previous plane
        }

        // Normal equations with a small ridge term for collinear supports.
        let r = 1e-6;
        let m = [
            [sxx + r, sxz, sx],
            [sxz, szz + r, sz],
            [sx, sz, n + r],
        ];
        let b = [sxy, szy, sy];

        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if det.abs() < 1e-12 {
            return;
        }

        let solve = |col: usize| {
            let mut c = m;
            for row in 0..3 {
                c[row][col] = b[row];
            }
            (c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
                - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
                + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]))
                / det
        };
        self.plane = [solve(0), solve(1), solve(2)];
    }

    #[inline]
    pub fn plane_y(&self, x: f64, z: f64) -> f64 {
        self.plane[0] * x + self.plane[1] * z + self.plane[2]
    }

    /// Is there room for the chassis here? `clear` is the underside's height
    /// above the support plane.
    #[inline]
    fn chassis_fits(&self, terrain: &Terrain, x: f64, z: f64, r: f64, clear: f64) -> bool {
        !terrain.obstructed(x, z, r, self.plane_y(x, z) + clear)
    }

    /// Advance along the route and re-measure where the next waypoint is.
    ///
    /// A waypoint counts as reached inside its radius, and also once the
    /// machine is well past it: arriving beside one and having to turn back
    /// for it is not a task anybody wants a walking robot to learn.
    fn update_route(&mut self, terrain: &Terrain) {
        let last = terrain.waypoints.len().saturating_sub(1);
        loop {
            let w = terrain.waypoint(self.wp);
            let (dx, dz) = (w[0] - self.pos[0], w[1] - self.pos[2]);
            let d = hypot2(dx, dz);
            if self.wp < last && d < WAYPOINT_R {
                self.wp += 1;
                self.reached += 1;
                continue;
            }
            // Left behind. Move on to the next one, but do not pretend it was
            // reached: walking past a waypoint at four metres' range is exactly
            // the behaviour the route exists to discourage.
            if self.wp < last && w[1] < self.pos[2] - 1.5 {
                self.wp += 1;
                continue;
            }
            self.wp_dist = d;
            // Bearing in the body frame, so it is zero exactly when the
            // waypoint is straight ahead however the machine is oriented.
            // Positive means it is off to the left.
            let b = crate::math::inv_rot_y([dx, 0.0, dz], self.yaw);
            self.bearing = ang_diff(b[0].atan2(b[2]));
            return;
        }
    }

    fn update_support(&mut self) {
        let mut pts: [[f64; 2]; MAX_LEGS] = [[0.0; 2]; MAX_LEGS];
        let mut n = 0;
        for f in self.feet.iter() {
            if f.stance {
                pts[n] = [f.plant[0], f.plant[2]];
                n += 1;
            }
        }
        self.hull_n = crate::math::convex_hull_xz(&pts[..n], &mut self.hull);

        let com = [
            self.pos[0] + self.com_drift[0],
            self.pos[2] + self.com_drift[1],
        ];
        self.margin = polygon_margin(&self.hull[..self.hull_n], com);

        let share = if n > 0 { 1.0 / n as f64 } else { 0.0 };
        for f in self.feet.iter_mut() {
            let target = if f.stance { share } else { 0.0 };
            f.load += (target - f.load) * 0.25;
        }
    }

    fn build_obs(&mut self, terrain: &Terrain, gait: &Gait, cmd: Cmd) {
        let n = self.frame.legs();
        let plane_here = self.plane_y(self.pos[0], self.pos[2]);
        let duty = self.duty_now;
        let t_swing = (1.0 - duty) * self.cycle_now;

        self.obs[0] = (self.pos[1] - (plane_here + gait.body_h)) / 0.25;
        self.obs[1] = self.pitch;
        self.obs[2] = self.roll;
        self.obs[3] = self.margin;
        self.obs[4] = (core::f64::consts::TAU * self.phase).sin();
        self.obs[5] = (core::f64::consts::TAU * self.phase).cos();

        let fwd = rot_y([0.0, 0.0, 1.0], self.yaw);
        let along = fwd[0] * self.vel[0] + fwd[2] * self.vel[1];
        let target = cmd.speed();
        self.obs[6] = (along - target) / track_width(target);

        for i in 0..n {
            let lp = self.feet[i].leg_phase;
            let t_remain = if lp < duty {
                (duty - lp) * self.cycle_now + t_swing
            } else {
                let u = (lp - duty) / (1.0 - duty);
                (1.0 - u) * t_swing
            };
            let p = self.predict_td(
                terrain,
                gait,
                i,
                t_remain,
                self.stride_now,
                self.cycle_now,
                duty,
                0.0,
            );
            self.obs[N_FIXED_OBS + i] = p[1] - self.plane_y(p[0], p[2]);
        }

        // The commanded speed itself, so one policy can serve every command
        // instead of memorising one.
        self.obs[obs_cmd_speed(self.frame)] = target / CRUISE_MAX;

        // --- navigation --------------------------------------------------------
        //
        // Where the machine is being asked to go, how far away it is, and how
        // much room there is on either side. Without these the steering action
        // has nothing to act on and settles at whatever constant scores best.
        self.obs[obs_bearing(self.frame)] = self.bearing / core::f64::consts::FRAC_PI_2;
        self.obs[obs_range(self.frame)] = self.wp_dist.min(16.0) / 16.0;
        self.obs[obs_corridor(self.frame)] = self.pos[0] / terrain.wall_x();

        // A forward scan, in the body frame, of how much higher the ground
        // ahead is than the ground underfoot. Two ranges by three bearings —
        // enough to tell a step up from a wall, and which side of the wall the
        // gap is on. The per-leg lookaheads above only ever see the next
        // footfall, which is far too late to steer on.
        let base = obs_scan(self.frame);
        for (i, ahead) in SCAN_AHEAD.iter().enumerate() {
            for (j, side) in SCAN_SIDE.iter().enumerate() {
                let p = rot_y([*side, 0.0, *ahead], self.yaw);
                let (px, pz) = (self.pos[0] + p[0], self.pos[2] + p[2]);
                let rise = terrain.probe(px, pz) - self.plane_y(px, pz);
                self.obs[base + i * SCAN_SIDE.len() + j] = clamp(rise, -1.5, 2.0);
            }
        }
        debug_assert_eq!(SCAN_AHEAD.len() * SCAN_SIDE.len(), N_SCAN);
    }
}

/// Result of a single rollout.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rollout {
    pub reward: f64,
    pub distance: f64,
    pub steps: usize,
    pub fell: bool,
    pub stub_total: f64,
    /// Mechanical work at the joints, joules.
    pub work: f64,
    /// Total foot skid, metres.
    pub slip: f64,
    /// Dimensionless mechanical cost of transport.
    pub cot: f64,
    /// Mean absolute speed error against the command, m/s.
    pub speed_error: f64,
    /// Worst joint demand seen, as a fraction of servo stall torque.
    pub peak_servo_load: f64,
    /// Waypoints reached, and ticks spent against a wall or an obstacle.
    pub reached: usize,
    pub collisions: f64,
    /// Mean cycle time, stride and duty the policy actually ran, after its
    /// online modulation. These are what a speed-conditioned policy varies.
    pub mean_cycle: f64,
    pub mean_stride: f64,
    pub mean_duty: f64,
}

/// Run a policy on a course for `secs` seconds of simulated time.
///
/// `norm_sink`, when supplied, accumulates observation statistics — this is
/// the state-normalisation half of ARS V2.
pub fn rollout(
    terrain: &Terrain,
    policy: &Policy,
    phys: &Physics,
    secs: f64,
    cmd: Cmd,
    norm_sink: Option<&mut crate::policy::Normalizer>,
) -> Rollout {
    let gait = policy.gait();
    let no = n_obs(gait.frame);
    let mut sim = Sim::default();
    sim.reset(terrain, &gait, phys);

    let n = (secs / DT) as usize;
    let mut total = 0.0;
    let mut sink = norm_sink;

    let mut steps = 0;
    let mut err_sum = 0.0;
    let mut peak_load = 0.0f64;
    let (mut c_sum, mut s_sum, mut d_sum) = (0.0, 0.0, 0.0);
    for _ in 0..n {
        total += sim.step(terrain, policy, &gait, DT, cmd);
        if let Some(s) = sink.as_deref_mut() {
            s.observe(&sim.obs, no);
        }
        let fwd = rot_y([0.0, 0.0, 1.0], sim.yaw);
        let along = fwd[0] * sim.vel[0] + fwd[2] * sim.vel[1];
        err_sum += (along - cmd.speed()).abs();
        peak_load = peak_load.max(sim.servo_load);
        c_sum += sim.cycle_now;
        s_sum += sim.stride_now;
        d_sum += sim.duty_now;
        steps += 1;
        if sim.fallen {
            break;
        }
    }

    Rollout {
        reward: total,
        distance: sim.dist,
        steps,
        fell: sim.fallen,
        stub_total: sim.stub_total,
        work: sim.work,
        slip: sim.slip_total,
        cot: sim.cost_of_transport(),
        speed_error: if steps > 0 {
            err_sum / steps as f64
        } else {
            0.0
        },
        peak_servo_load: peak_load,
        reached: sim.reached,
        collisions: sim.collisions,
        mean_cycle: mean(c_sum, steps),
        mean_stride: mean(s_sum, steps),
        mean_duty: mean(d_sum, steps),
    }
}

#[inline]
fn mean(sum: f64, n: usize) -> f64 {
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// Score a policy the way training does: the same course at several commanded
/// speeds, averaged. A policy that can only do one speed cannot win here.
pub fn evaluate(
    terrain: &Terrain,
    policy: &Policy,
    phys: &Physics,
    secs: f64,
) -> Rollout {
    let mut acc = Rollout::default();
    let n = EVAL_SPEEDS.len() as f64;
    for &s in EVAL_SPEEDS.iter() {
        let r = rollout(terrain, policy, phys, secs, Cmd::at(s), None);
        acc.reward += r.reward / n;
        acc.distance += r.distance / n;
        acc.steps += r.steps;
        acc.fell |= r.fell;
        acc.stub_total += r.stub_total / n;
        acc.work += r.work / n;
        acc.slip += r.slip / n;
        acc.cot += r.cot / n;
        acc.speed_error += r.speed_error / n;
        acc.peak_servo_load = acc.peak_servo_load.max(r.peak_servo_load);
        acc.reached += r.reached;
        acc.collisions += r.collisions / n;
        acc.mean_cycle += r.mean_cycle / n;
        acc.mean_stride += r.mean_stride / n;
        acc.mean_duty += r.mean_duty / n;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::Actuator;
    use crate::policy::{n_gait, n_theta, Preset};
    use crate::terrain::Course;

    fn baseline() -> Policy {
        Policy::seeded(Preset::Tripod, Frame::default())
    }

    fn frames() -> Vec<Frame> {
        (crate::robot::MIN_LEGS..=MAX_LEGS)
            .step_by(2)
            .map(Frame::new)
            .collect()
    }

    /// The baseline gait's own self-consistent speed, which is what it can
    /// hold open-loop before the learner gives it anything else.
    fn native_cmd() -> Cmd {
        Cmd::at(baseline().gait().nominal_speed())
    }

    #[test]
    fn baseline_walks_forward_on_flat_ground_without_falling() {
        let t = Terrain::new(Course::Flat, 1);
        let r = rollout(
            &t,
            &baseline(),
            &Physics::default(),
            8.0,
            native_cmd(),
            None,
        );
        assert!(!r.fell, "baseline fell on flat ground");
        assert!(r.distance > 8.0, "only travelled {:.2} m", r.distance);
        assert!(r.reward > 0.0, "reward {:.2}", r.reward);
    }

    #[test]
    fn the_body_holds_the_speed_it_is_commanded() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        for &s in &[2.5, 4.0, 5.0] {
            let r = rollout(&t, &p, &Physics::default(), 6.0, Cmd::at(s), None);
            let avg = r.distance / (r.steps as f64 * DT);
            assert!(
                (avg - s).abs() / s < 0.20,
                "commanded {s:.1}, averaged {avg:.2}"
            );
        }
    }

    #[test]
    fn speed_comes_from_the_command_not_from_the_stride() {
        // Two gaits with very different nominal speeds, both asked for 3 m/s,
        // both have to deliver 3 m/s. This is the property that made pinning
        // stride and cycle to their bounds pointless.
        let t = Terrain::new(Course::Flat, 1);
        let mut slow = baseline();
        let mut fast = baseline();
        slow.theta[1] = crate::math::unsquash(0.55, 0.30, 1.45);
        fast.theta[1] = crate::math::unsquash(1.40, 0.30, 1.45);
        assert!(fast.gait().nominal_speed() > slow.gait().nominal_speed() * 2.0);

        for p in [&slow, &fast] {
            let r = rollout(&t, p, &Physics::default(), 5.0, Cmd::at(3.0), None);
            let avg = r.distance / (r.steps as f64 * DT);
            assert!((avg - 3.0).abs() < 0.6, "averaged {avg:.2} for 3.0 m/s");
        }
    }

    #[test]
    fn standing_still_stays_put_and_upright() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let g = p.gait();
        let phys = Physics::default();
        let mut s = Sim::default();
        s.reset(&t, &g, &phys);
        for _ in 0..600 {
            s.step(&t, &p, &g, DT, Cmd::default());
        }
        assert!(!s.fallen);
        assert!(s.pos[2].abs() < 0.15, "drifted to z={}", s.pos[2]);
        assert!(s.margin > 0.0, "unstable while standing: {}", s.margin);
    }

    #[test]
    fn feet_never_sink_below_the_terrain() {
        let t = Terrain::new(Course::Rubble, 4);
        let p = baseline();
        let g = p.gait();
        let phys = Physics::default();
        let mut s = Sim::default();
        s.reset(&t, &g, &phys);
        for _ in 0..800 {
            s.step(&t, &p, &g, DT, native_cmd());
            if s.fallen {
                break;
            }
            for f in s.feet.iter() {
                let ground = t.height(f.world[0], f.world[2]);
                assert!(
                    f.world[1] >= ground - 1e-6,
                    "foot at {:.3} under ground {:.3}",
                    f.world[1],
                    ground
                );
            }
        }
    }

    #[test]
    fn stance_feet_grip_when_there_is_traction_to_spare() {
        // Flat ground, ideal actuators, absurd friction: no contact point
        // should move at all.
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let g = p.gait();
        let mut s = Sim::default();
        s.reset(&t, &g, &Physics::ideal());
        let n = s.frame.legs();
        let mut planted = [[0.0f64; 3]; MAX_LEGS];
        let mut was = [false; MAX_LEGS];
        for _ in 0..500 {
            s.step(&t, &p, &g, DT, native_cmd());
            for i in 0..n {
                if s.feet[i].stance {
                    if was[i] {
                        let d = hypot2(
                            s.feet[i].plant[0] - planted[i][0],
                            s.feet[i].plant[2] - planted[i][2],
                        );
                        assert!(d < 1e-12, "leg {i} slid {d}");
                    }
                    planted[i] = s.feet[i].plant;
                }
                was[i] = s.feet[i].stance;
            }
        }
        assert_eq!(s.slip_total, 0.0);
    }

    #[test]
    fn feet_skid_when_the_ground_cannot_supply_the_force() {
        // Same course, same gait, same command — only mu changes.
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let grippy = Physics {
            mu: 1.2,
            ..Physics::default()
        };
        let icy = Physics {
            mu: 0.08,
            ..Physics::default()
        };
        let a = rollout(&t, &p, &grippy, 6.0, Cmd::at(5.0), None);
        let b = rollout(&t, &p, &icy, 6.0, Cmd::at(5.0), None);
        assert!(
            b.slip > a.slip * 5.0,
            "ice should skid far more: {:.3} vs {:.3}",
            b.slip,
            a.slip
        );
        assert!(
            b.distance < a.distance,
            "ice should get less far: {:.2} vs {:.2}",
            b.distance,
            a.distance
        );
    }

    #[test]
    fn traction_utilisation_reads_as_a_fraction_of_the_budget() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let g = p.gait();
        let mut s = Sim::default();
        s.reset(&t, &g, &Physics::default());
        let mut launch = 0.0f64;
        let mut cruise = 0.0f64;
        for i in 0..400 {
            s.step(&t, &p, &g, DT, Cmd::at(3.0));
            assert!(s.traction >= 0.0 && s.traction.is_finite());
            if i < 100 {
                launch = launch.max(s.traction);
            } else {
                cruise = cruise.max(s.traction);
            }
        }
        // Getting to 3 m/s from a standstill asks for more than 0.85 g, so the
        // feet scrabble briefly — that is the correct answer, not a bug.
        assert!(launch > 1.0, "launch was free: {launch:.2}");
        // Once up to speed, holding it spends only part of what is on offer.
        assert!(cruise < 1.0, "flat cruise saturated traction at {cruise:.2}");

        // On ice the same walk asks for more than there is.
        let icy = Physics {
            mu: 0.05,
            ..Physics::default()
        };
        let mut s = Sim::default();
        s.reset(&t, &g, &icy);
        let mut over = false;
        for _ in 0..400 {
            s.step(&t, &p, &g, DT, Cmd::at(3.0));
            over |= s.traction > 1.0;
        }
        assert!(over, "ice never saturated traction");
    }

    #[test]
    fn loose_rubble_grips_less_than_firm_ground() {
        let t = Terrain::new(Course::Rubble, 4);
        assert!(t.grip(0.0, -3.0) > t.grip(
            (t.obstacles[0].x0 + t.obstacles[0].x1) * 0.5,
            (t.obstacles[0].z0 + t.obstacles[0].z1) * 0.5
        ));
    }

    #[test]
    fn a_weaker_servo_walks_worse_than_a_stronger_one() {
        let t = Terrain::new(Course::Mixed, 6);
        let p = baseline();
        let strong = Physics {
            actuator: Actuator::from_rating(0.14, 25.0),
            ..Physics::default()
        };
        let weak = Physics {
            actuator: Actuator::from_rating(0.30, 6.0),
            ..Physics::default()
        };
        let a = rollout(&t, &p, &strong, 8.0, Cmd::at(4.0), None);
        let b = rollout(&t, &p, &weak, 8.0, Cmd::at(4.0), None);
        assert!(
            b.reward < a.reward,
            "the weak servo should score worse: {:.2} vs {:.2}",
            b.reward,
            a.reward
        );
        assert!(
            b.peak_servo_load > 1.0,
            "the weak servo should be driven past stall, got {:.2}",
            b.peak_servo_load
        );
    }

    #[test]
    fn an_overloaded_servo_lets_the_chassis_sag() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let g = p.gait();
        let weak = Physics {
            actuator: Actuator::from_rating(0.30, 4.0),
            ..Physics::default()
        };
        let mut s = Sim::default();
        s.reset(&t, &g, &weak);
        let stand = g.body_h;
        let mut lowest = f64::INFINITY;
        for _ in 0..300 {
            s.step(&t, &p, &g, DT, Cmd::at(3.0));
            lowest = lowest.min(s.pos[1] - s.plane_y(s.pos[0], s.pos[2]));
        }
        assert!(
            lowest < stand - 0.02,
            "chassis never sagged: {lowest:.3} vs {stand:.3}"
        );
    }

    #[test]
    fn momentum_survives_a_command_to_stop() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let g = p.gait();
        let phys = Physics::default();
        let mut s = Sim::default();
        s.reset(&t, &g, &phys);
        for _ in 0..400 {
            s.step(&t, &p, &g, DT, Cmd::at(5.0));
        }
        let running = hypot2(s.vel[0], s.vel[1]);
        assert!(running > 3.0, "never got going: {running:.2}");
        // One tick of "stop" cannot remove 5 m/s.
        s.step(&t, &p, &g, DT, Cmd::default());
        let after = hypot2(s.vel[0], s.vel[1]);
        assert!(
            after > running - 0.25,
            "lost {:.2} m/s in one tick",
            running - after
        );
        // But a second of it can.
        for _ in 0..100 {
            s.step(&t, &p, &g, DT, Cmd::default());
        }
        assert!(hypot2(s.vel[0], s.vel[1]) < 1.0);
    }

    #[test]
    fn work_and_cost_of_transport_are_physical() {
        let t = Terrain::new(Course::Flat, 1);
        let r = rollout(
            &t,
            &baseline(),
            &Physics::default(),
            8.0,
            Cmd::at(4.0),
            None,
        );
        assert!(r.work > 0.0 && r.work.is_finite());
        // Legged machines land somewhere around 0.1 to 10; anything outside
        // that means the units are wrong.
        assert!(
            r.cot > 0.01 && r.cot < 40.0,
            "cost of transport {:.3} is not plausible",
            r.cot
        );
    }

    #[test]
    fn a_heavier_robot_does_more_work_for_the_same_walk() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let light = Physics {
            mass_kg: 1.0,
            ..Physics::default()
        };
        let heavy = Physics {
            mass_kg: 3.0,
            ..Physics::default()
        };
        let a = rollout(&t, &p, &light, 6.0, Cmd::at(4.0), None);
        let b = rollout(&t, &p, &heavy, 6.0, Cmd::at(4.0), None);
        // Not proportional: swinging the legs costs the same either way, and
        // only the part of the work that goes through the feet scales with the
        // payload the robot is carrying.
        assert!(b.work > a.work * 1.15, "{:.1} vs {:.1}", b.work, a.work);
    }

    #[test]
    fn rough_ground_is_harder_than_flat_ground() {
        let p = baseline();
        let phys = Physics::default();
        let flat = rollout(
            &Terrain::new(Course::Flat, 1),
            &p,
            &phys,
            10.0,
            Cmd::at(4.0),
            None,
        );
        let rough = rollout(
            &Terrain::new(Course::Mixed, 7),
            &p,
            &phys,
            10.0,
            Cmd::at(4.0),
            None,
        );
        assert!(
            rough.reward < flat.reward,
            "rough {:.2} should score below flat {:.2}",
            rough.reward,
            flat.reward
        );
    }

    #[test]
    fn simulation_is_deterministic() {
        let t = Terrain::new(Course::Mixed, 11);
        let p = baseline();
        let phys = Physics::default();
        let a = rollout(&t, &p, &phys, 6.0, Cmd::at(4.0), None);
        let b = rollout(&t, &p, &phys, 6.0, Cmd::at(4.0), None);
        assert_eq!(a.steps, b.steps);
        assert_eq!(a.reward.to_bits(), b.reward.to_bits());
    }

    #[test]
    fn no_nans_under_extreme_parameters() {
        let t = Terrain::new(Course::Mixed, 2);
        let phys = Physics::default();
        let mut r = crate::math::Rng::new(99);
        for _ in 0..40 {
            let mut p = baseline();
            for v in p.theta.iter_mut().take(n_theta(Frame::default())) {
                *v = r.normal() * 6.0;
            }
            let out = rollout(&t, &p, &phys, 3.0, Cmd::at(4.0), None);
            assert!(out.reward.is_finite(), "reward went non-finite");
            assert!(out.distance.is_finite());
            assert!(out.work.is_finite() && out.cot.is_finite());
        }
    }

    #[test]
    fn turn_command_changes_heading() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let g = p.gait();
        let phys = Physics::default();
        let mut s = Sim::default();
        s.reset(&t, &g, &phys);
        for _ in 0..300 {
            s.step(
                &t,
                &p,
                &g,
                DT,
                Cmd {
                    fwd: 0.6,
                    turn: 1.0,
                    nav: false,
                    ..Cmd::default()
                },
            );
        }
        assert!(s.yaw.abs() > 0.5, "yaw only reached {}", s.yaw);
    }

    #[test]
    fn the_policy_can_modulate_cycle_stride_and_duty_from_the_command() {
        // Wire the commanded-speed observation straight to the duty action and
        // check the gait clock actually moves. Without this plumbing a policy
        // has one gait for every speed.
        let t = Terrain::new(Course::Flat, 1);
        let phys = Physics::default();
        let f = Frame::default();
        let no = n_obs(f);
        let cmd_obs = obs_cmd_speed(f);
        let duty_row = n_gait(f) + act_duty(f) * no + cmd_obs;

        let mut p = baseline();
        p.theta[duty_row] = 4.0;
        let slow = rollout(&t, &p, &phys, 4.0, Cmd::at(2.0), None);
        let fast = rollout(&t, &p, &phys, 4.0, Cmd::at(5.5), None);
        assert!(
            fast.mean_duty > slow.mean_duty + 0.02,
            "duty did not follow the command: {:.3} vs {:.3}",
            slow.mean_duty,
            fast.mean_duty
        );

        let mut q = baseline();
        q.theta[n_gait(f) + act_cycle(f) * no + cmd_obs] = 4.0;
        let a = rollout(&t, &q, &phys, 4.0, Cmd::at(2.0), None);
        let b = rollout(&t, &q, &phys, 4.0, Cmd::at(5.5), None);
        assert!(b.mean_cycle > a.mean_cycle + 0.01, "cycle time is not wired");
    }

    #[test]
    fn the_baseline_gait_is_untouched_by_the_modulation_actions() {
        // Iteration 0 has a zero feedback block, so the live gait must be
        // exactly the hand-tuned one. That is what makes the comparison in the
        // dashboard honest.
        let t = Terrain::new(Course::Flat, 1);
        let g = baseline().gait();
        let r = rollout(&t, &baseline(), &Physics::default(), 4.0, Cmd::at(3.0), None);
        assert!((r.mean_cycle - g.cycle).abs() < 1e-12);
        assert!((r.mean_stride - g.stride).abs() < 1e-12);
        assert!((r.mean_duty - g.duty).abs() < 1e-12);
    }

    #[test]
    fn a_swinging_leg_costs_torque_even_with_nothing_underfoot() {
        // Before leg mass existed a leg in the air carried no load at all, so
        // it moved at the servo's no-load speed however fast it was asked to.
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let g = p.gait();
        let heavy = Physics::default();
        let weightless = Physics {
            leg: crate::dynamics::LegMass::WEIGHTLESS,
            ..Physics::default()
        };

        let mut peak = [0.0f64; 2];
        for (k, phys) in [weightless, heavy].iter().enumerate() {
            let mut s = Sim::default();
            s.reset(&t, &g, phys);
            for _ in 0..400 {
                s.step(&t, &p, &g, DT, Cmd::at(4.0));
                peak[k] = peak[k].max(s.leg_torque);
            }
        }
        assert_eq!(peak[0], 0.0, "weightless legs should cost nothing");
        assert!(peak[1] > 0.0, "leg mass costs no torque: {}", peak[1]);
    }

    #[test]
    fn heavier_legs_are_harder_to_swing_and_cost_more_work() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let light = Physics {
            leg: crate::dynamics::LegMass::from_servo(0.020),
            ..Physics::default()
        };
        let heavy = Physics {
            leg: crate::dynamics::LegMass::from_servo(0.120),
            ..Physics::default()
        };
        let a = rollout(&t, &p, &light, 6.0, Cmd::at(4.0), None);
        let b = rollout(&t, &p, &heavy, 6.0, Cmd::at(4.0), None);
        assert!(
            b.work > a.work,
            "heavy legs did no more work: {:.1} vs {:.1}",
            b.work,
            a.work
        );
        assert!(
            b.peak_servo_load > a.peak_servo_load,
            "heavy legs did not load the servos harder: {:.2} vs {:.2}",
            b.peak_servo_load,
            a.peak_servo_load
        );
    }

    #[test]
    fn swinging_legs_push_back_on_the_chassis() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let g = p.gait();
        let mut s = Sim::default();
        s.reset(&t, &g, &Physics::default());
        let mut peak = 0.0f64;
        for _ in 0..400 {
            s.step(&t, &p, &g, DT, Cmd::at(5.0));
            peak = peak.max(hypot2(s.leg_react[0], s.leg_react[2]));
            assert!(s.leg_react.iter().all(|v| v.is_finite()));
        }
        // Real but not dominant: a fraction of the robot's own weight.
        let weight = Physics::default().mass_kg * G;
        assert!(peak > 0.05, "no reaction at all: {peak:.3} N");
        assert!(
            peak < weight,
            "reaction {peak:.2} N exceeds the {weight:.2} N robot"
        );
    }

    #[test]
    fn every_leg_count_walks() {
        // The whole point of the frame being a value: four legs to ten, same
        // simulator, same policy code, nobody falls over standing still.
        let t = Terrain::new(Course::Flat, 1);
        let phys = Physics::default();
        for f in frames() {
            let p = Policy::seeded(Preset::default_for(f), f);
            let r = rollout(&t, &p, &phys, 8.0, Cmd::at(3.0), None);
            assert!(!r.fell, "{} legs fell on flat ground", f.legs());
            let avg = r.distance / (r.steps as f64 * DT);
            assert!(
                (avg - 3.0).abs() < 0.6,
                "{} legs averaged {avg:.2} m/s for a 3.0 command",
                f.legs()
            );
            assert!(r.work.is_finite() && r.cot.is_finite());
        }
    }

    #[test]
    fn a_trotting_quadruped_is_not_statically_stable() {
        // Two diagonal feet are a line, not a polygon, so the margin goes
        // negative and the robot goes over. Real quadrupeds trot because a
        // trot is dynamically stable — this simulator judges stability
        // statically, and says so rather than quietly getting it wrong.
        let t = Terrain::new(Course::Flat, 1);
        let f = Frame::new(4);
        let phys = Physics::default();
        let trot = rollout(
            &t,
            &Policy::seeded(Preset::Tripod, f),
            &phys,
            8.0,
            Cmd::at(3.0),
            None,
        );
        let crawl = rollout(
            &t,
            &Policy::seeded(Preset::Wave, f),
            &phys,
            8.0,
            Cmd::at(3.0),
            None,
        );
        assert!(trot.fell, "a two-foot support polygon held the robot up");
        assert!(!crawl.fell, "the crawl should keep three feet down");
        assert!(crawl.reward > trot.reward);
    }

    #[test]
    fn more_legs_hold_a_bigger_support_polygon() {
        let t = Terrain::new(Course::Flat, 1);
        let phys = Physics::default();
        let margin = |legs: usize| {
            let f = Frame::new(legs);
            let p = Policy::seeded(Preset::Wave, f);
            let g = p.gait();
            let mut s = Sim::default();
            s.reset(&t, &g, &phys);
            let mut worst = f64::INFINITY;
            for _ in 0..500 {
                s.step(&t, &p, &g, DT, Cmd::at(2.5));
                worst = worst.min(s.margin);
            }
            worst
        };
        let (four, ten) = (margin(4), margin(10));
        assert!(
            ten > four,
            "ten legs were less stable than four: {ten:.3} vs {four:.3}"
        );
    }

    /// A hand-wired steering policy: bearing to the next waypoint straight
    /// into the steer action, with the sign the geometry actually calls for.
    /// This is the plumbing test — that the machine *can* be steered by what it
    /// observes. Whether the learner finds the same wiring is a separate
    /// question, and one the trainer answers.
    fn autopilot(gain: f64) -> Policy {
        let f = Frame::default();
        let mut p = baseline();
        p.theta[n_gait(f) + act_steer(f) * n_obs(f) + obs_bearing(f)] = gain;
        p
    }

    #[test]
    fn the_corridor_is_fenced_on_both_sides() {
        // Drive hard into one wall, then the other. The machine may lean on
        // them; it may not walk through them.
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let g = p.gait();
        let limit = t.wall_x() - Frame::default().body_r() * 0.9;
        for turn in [1.0, -1.0] {
            let mut s = Sim::default();
            s.reset(&t, &g, &Physics::default());
            for _ in 0..1500 {
                s.step(
                    &t,
                    &p,
                    &g,
                    DT,
                    Cmd {
                        fwd: 1.0,
                        turn,
                        nav: false,
                        ..Cmd::default()
                    },
                );
                assert!(
                    s.pos[0].abs() <= limit + 1e-6,
                    "escaped the corridor at x={:.3}, limit {limit:.3}",
                    s.pos[0]
                );
                if s.fallen {
                    break;
                }
            }
        }
    }

    #[test]
    fn waypoints_are_reached_in_order_and_the_route_never_runs_out() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let g = p.gait();
        let mut s = Sim::default();
        s.reset(&t, &g, &Physics::default());
        let mut seen = s.wp;
        for _ in 0..2500 {
            s.step(&t, &p, &g, DT, Cmd::at(4.0));
            assert!(s.wp == seen || s.wp == seen + 1, "route jumped {seen} -> {}", s.wp);
            assert!(s.wp < t.waypoints.len());
            seen = s.wp;
            if s.fallen {
                break;
            }
        }
        assert_eq!(s.reached, s.wp, "reached count and index disagree");
        assert!(s.reached >= 2, "only reached {} waypoints", s.reached);
    }

    #[test]
    fn the_bearing_is_zero_when_the_waypoint_is_dead_ahead() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let g = p.gait();
        let mut s = Sim::default();
        s.reset(&t, &g, &Physics::default());
        // The route on a flat course runs straight down the centreline, and
        // the machine spawns on it pointing along it.
        assert!(s.bearing.abs() < 1e-9, "bearing {} at spawn", s.bearing);

        // Turn ninety degrees and the waypoint is off to one side by as much.
        s.yaw = core::f64::consts::FRAC_PI_2;
        s.update_route(&t);
        assert!(
            (s.bearing.abs() - core::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "bearing {} after a quarter turn",
            s.bearing
        );
    }

    #[test]
    fn steering_gets_a_machine_through_a_slalom_that_walking_straight_does_not() {
        // The point of the whole exercise: a course where the way forward is
        // not forward. The straight walker parks against the first wall; the
        // one that steers toward its waypoints goes round.
        let t = Terrain::new(Course::Slalom, 3);
        let phys = Physics::default();
        let straight = rollout(&t, &baseline(), &phys, 22.0, Cmd::at(3.0), None);

        let mut best = straight;
        for gain in [-2.0, -3.5, -5.0] {
            let r = rollout(&t, &autopilot(gain), &phys, 22.0, Cmd::at(3.0), None);
            if r.reached > best.reached {
                best = r;
            }
        }
        assert!(
            best.reached > straight.reached,
            "steering reached {} waypoints, straight ahead reached {}",
            best.reached,
            straight.reached
        );
        assert!(
            best.distance > straight.distance + 3.0,
            "steering got {:.1} m, straight ahead got {:.1} m",
            best.distance,
            straight.distance
        );
    }

    #[test]
    fn a_foot_is_never_planted_on_top_of_a_wall() {
        // Slalom walls are nearly twice the length of a leg. Deflecting the
        // step is what keeps the support plane from being fitted through a
        // foot two metres in the air.
        let t = Terrain::new(Course::Slalom, 3);
        let p = autopilot(-3.5);
        let g = p.gait();
        let mut s = Sim::default();
        s.reset(&t, &g, &Physics::default());
        for _ in 0..2200 {
            s.step(&t, &p, &g, DT, Cmd::at(3.0));
            if s.fallen {
                break;
            }
            for i in 0..s.frame.legs() {
                let f = s.feet[i].world;
                assert!(
                    f[1] - s.plane_y(f[0], f[2]) < 1.0,
                    "leg {i} standing {:.2} m above the support plane",
                    f[1] - s.plane_y(f[0], f[2])
                );
            }
        }
    }

    #[test]
    fn the_forward_scan_sees_a_wall_before_the_feet_do() {
        // The per-leg lookaheads only ever see the next footfall. Steering
        // needs to know sooner than that.
        let t = Terrain::new(Course::Slalom, 3);
        let p = baseline();
        let g = p.gait();
        let f = Frame::default();
        let mut s = Sim::default();
        s.reset(&t, &g, &Physics::default());

        let wall = t.obstacles.iter().find(|o| o.top > 1.0).unwrap();
        // Park the machine three metres short of the middle of that segment,
        // facing it, and read the scan.
        s.pos[0] = (wall.x0 + wall.x1) * 0.5;
        s.pos[2] = wall.z0 - 2.6;
        s.build_obs(&t, &g, Cmd::at(3.0));
        let scan = &s.obs[obs_scan(f)..obs_scan(f) + N_SCAN];
        assert!(
            scan.iter().any(|v| *v > 0.9),
            "nothing tall in the scan in front of a 1.8 m wall: {scan:?}"
        );

        // And out in the open it sees nothing.
        s.pos = [0.0, s.pos[1], -2.0];
        s.build_obs(&t, &g, Cmd::at(3.0));
        let clear = &s.obs[obs_scan(f)..obs_scan(f) + N_SCAN];
        assert!(clear.iter().all(|v| v.abs() < 0.2), "phantom obstacle: {clear:?}");
    }

    #[test]
    fn the_walls_are_visible_to_the_scan_even_though_nothing_draws_them() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let g = p.gait();
        let f = Frame::default();
        let mut s = Sim::default();
        s.reset(&t, &g, &Physics::default());
        s.pos[0] = t.wall_x() - 1.0;
        s.yaw = -core::f64::consts::FRAC_PI_2; // nose toward the wall
        s.build_obs(&t, &g, Cmd::at(3.0));
        let scan = &s.obs[obs_scan(f)..obs_scan(f) + N_SCAN];
        assert!(
            scan.iter().any(|v| *v > 0.9),
            "the fence is invisible to the sensor too: {scan:?}"
        );
        assert!((s.obs[obs_corridor(f)] - (1.0 - 1.0 / t.wall_x())).abs() < 1e-9);
    }

    #[test]
    fn evaluation_averages_over_several_commanded_speeds() {
        let t = Terrain::new(Course::Flat, 1);
        let p = baseline();
        let phys = Physics::default();
        let e = evaluate(&t, &p, &phys, 5.0);
        let mut manual = 0.0;
        for &s in EVAL_SPEEDS.iter() {
            manual += rollout(&t, &p, &phys, 5.0, Cmd::at(s), None).reward;
        }
        assert!((e.reward - manual / EVAL_SPEEDS.len() as f64).abs() < 1e-9);
    }
}
