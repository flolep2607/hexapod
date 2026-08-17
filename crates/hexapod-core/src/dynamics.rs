//! Contact, actuator and body dynamics.
//!
//! Everything in this file exists to answer one question the earlier version of
//! the simulator could not: *what stops the robot from doing what the gait asks
//! for?* There are three answers, and they are the three things a kinematic
//! walker gets for free and a real one does not.
//!
//! **Friction.** A foot can only push as hard as `mu * N`. Beyond that it
//! skids, the body does not get the acceleration the gait asked for, and the
//! contact point moves — which corrupts the support polygon and the plane fit
//! for every subsequent tick.
//!
//! **Actuators.** A servo has a torque-speed line: it can move fast or push
//! hard, not both. Joints are integrated toward their commanded angle at a rate
//! that falls linearly to zero at stall, and past stall they *back-drive* — the
//! leg folds under the load and the chassis sags. The numbers come from the
//! servo selected in the hardware catalogue, so the machine you are costing is
//! the machine you are simulating.
//!
//! **Momentum.** The chassis carries linear and angular velocity. It cannot
//! change speed faster than traction allows, it keeps going when a foot slips,
//! and accelerating throws its centre of mass backwards — which is what
//! actually tips a legged robot when it stops in a hurry.
//!
//! **Leg mass.** The femur and tibia assemblies have weight, and it is not
//! small — on a hobby hexapod the legs are most of the robot. A swinging leg
//! has to hold itself up against gravity and be accelerated and stopped, both
//! of which the joint pays for in torque, and the reaction goes into the
//! chassis. Before this the swing phase was free: a leg in the air carried no
//! load, so it moved at the servo's no-load speed no matter how fast it was
//! asked to.
//!
//! What is still missing, stated plainly: links are rigid, contact is resolved
//! once per tick rather than by an impulse solver, and the leg-inertia terms
//! are the diagonal ones — each joint sees the mass distal to it, without the
//! off-diagonal coupling or the Coriolis terms of a full mass matrix. This is
//! a *centroidal* rigid-body model with Coulomb contact, actuator limits and
//! lumped leg inertia, not a general articulated multibody engine.

use crate::math::{clamp, V3};
use crate::robot::{Frame, MAX_LEGS};

pub const G: f64 = 9.81;

/// Newton-metres to kilogram-centimetres, the unit servos are sold in.
pub const NM_TO_KGCM: f64 = 10.1972;

/// Position-loop gain of a hobby servo, 1/s. Hobby servos are proportional
/// controllers with a deadband; this is the proportional part.
const SERVO_KP: f64 = 22.0;

/// How fast an overloaded joint gives way, as a fraction of its no-load speed
/// per unit of overload. Worm and high-ratio gearboxes back-drive slowly.
const BACKDRIVE: f64 = 0.35;

/// One servo, reduced to the two numbers that change how the robot walks.
#[derive(Clone, Copy, Debug)]
pub struct Actuator {
    /// No-load angular rate, rad/s.
    pub omega_max: f64,
    /// Stall torque, newton-metres, at the rated voltage.
    pub stall_nm: f64,
    /// Proportional position-loop gain, 1/s.
    pub kp: f64,
}

impl Default for Actuator {
    /// A generic 20 kg-cm metal-gear digital servo: 0.16 s/60 deg, 1.96 N-m.
    fn default() -> Self {
        Actuator::from_rating(0.16, 20.0)
    }
}

impl Actuator {
    /// Build from the two figures every servo is sold on: seconds per 60
    /// degrees at no load, and stall torque in kg-cm.
    pub fn from_rating(s60: f64, stall_kgcm: f64) -> Actuator {
        let s60 = s60.max(1e-3);
        Actuator {
            omega_max: (core::f64::consts::PI / 3.0) / s60,
            stall_nm: (stall_kgcm / NM_TO_KGCM).max(1e-6),
            kp: SERVO_KP,
        }
    }

    /// An idealised joint: infinitely strong, and deadbeat at the simulator's
    /// 100 Hz control rate — `kp * dt == 1`, so it lands exactly on its
    /// command each tick instead of ringing past it. Used to isolate contact
    /// effects from actuator effects in tests.
    pub fn ideal() -> Actuator {
        Actuator {
            omega_max: 1.0e6,
            stall_nm: 1.0e6,
            kp: 100.0,
        }
    }

    /// Fraction of stall torque this joint is being asked for.
    #[inline]
    pub fn load(&self, tau_nm: f64) -> f64 {
        tau_nm.abs() / self.stall_nm
    }

    /// Available angular rate while holding `tau_nm`. A brushed DC motor's
    /// torque-speed characteristic is a straight line from no-load speed at
    /// zero torque to zero speed at stall.
    #[inline]
    pub fn rate_limit(&self, tau_nm: f64) -> f64 {
        self.omega_max * (1.0 - clamp(self.load(tau_nm), 0.0, 1.0))
    }

    /// Rate, rad/s, at which a joint asked for more than it has gives way.
    /// Zero until the demand passes stall.
    #[inline]
    pub fn backdrive(&self, tau_nm: f64) -> f64 {
        let over = self.load(tau_nm) - 1.0;
        if over <= 0.0 {
            0.0
        } else {
            self.omega_max * BACKDRIVE * clamp(over, 0.0, 1.5)
        }
    }
}

/// The physical machine the simulated gait is running on.
///
/// `scale` and `mass_kg` are the same numbers the hardware tab uses, so the
/// torque a joint sees in the simulator is the torque the catalogue sizes a
/// servo against. There is one build, not two.
#[derive(Clone, Copy, Debug)]
pub struct Physics {
    /// All-up mass of the real build, kilograms.
    pub mass_kg: f64,
    /// Simulator units to metres.
    pub scale: f64,
    /// Coulomb friction coefficient of a foot on clean, firm ground.
    pub mu: f64,
    /// Mass of the part of one leg that actually swings.
    pub leg: LegMass,
    /// Allowance for landing impact and imperfect load sharing between the
    /// feet that are down. Mirrors `Build::dynamic_factor` so the torque the
    /// simulator drives a joint with is the torque the catalogue sizes it on.
    pub dynamic: f64,
    pub actuator: Actuator,
}

impl Default for Physics {
    fn default() -> Self {
        Physics {
            mass_kg: 2.0,
            scale: 0.10,
            // Rubber foot on dry board. Loose ground scales this down.
            mu: 0.85,
            leg: LegMass::from_servo(0.060),
            dynamic: 1.5,
            actuator: Actuator::default(),
        }
    }
}

impl Physics {
    /// Perfect traction, perfect actuators. Every remaining effect in the
    /// simulator is then purely kinematic, which is what makes it a useful
    /// control in tests.
    pub fn ideal() -> Physics {
        Physics {
            mu: 40.0,
            actuator: Actuator::ideal(),
            leg: LegMass::WEIGHTLESS,
            ..Physics::default()
        }
    }

    /// Mass of all the swinging legs together, kilograms.
    #[inline]
    pub fn swing_mass(&self, frame: Frame) -> f64 {
        frame.legs() as f64 * self.leg.total()
    }
}

/// The swinging half of a leg, split by link.
///
/// Only two of the three servos move with the leg: the coxa servo is bolted to
/// the chassis and turns about a vertical axis through its own mount, so it
/// neither swings nor lifts.
#[derive(Clone, Copy, Debug)]
pub struct LegMass {
    /// Femur assembly: knee servo, link, hardware.
    pub femur_kg: f64,
    /// Tibia assembly: ankle servo, link, foot.
    pub tibia_kg: f64,
}

impl LegMass {
    /// Massless legs, for isolating everything else in tests.
    pub const WEIGHTLESS: LegMass = LegMass {
        femur_kg: 0.0,
        tibia_kg: 0.0,
    };

    /// Estimated from one servo's mass. Structure, horns, screws and the foot
    /// add roughly half a servo again to the femur and a fifth to the tibia.
    pub fn from_servo(servo_kg: f64) -> LegMass {
        LegMass {
            femur_kg: servo_kg * 1.5,
            tibia_kg: servo_kg * 1.2,
        }
    }

    #[inline]
    pub fn total(&self) -> f64 {
        self.femur_kg + self.tibia_kg
    }
}

/// Where the swinging mass of one leg sits, in the body frame, in metres.
///
/// Links are treated as uniform rods, so each centre of mass is the midpoint
/// of its link.
#[inline]
pub fn leg_com(joints: &[V3; 4], mass: &LegMass, scale: f64) -> V3 {
    let mid = |a: V3, b: V3| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
    let f = mid(joints[1], joints[2]);
    let t = mid(joints[2], joints[3]);
    let m = mass.total();
    if m <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let w = |a: f64, b: f64| (a * mass.femur_kg + b * mass.tibia_kg) / m * scale;
    [w(f[0], t[0]), w(f[1], t[1]), w(f[2], t[2])]
}

/// Mass-weighted centre of mass of the chassis plus every swinging link.
///
/// `pos` and `joints` must share a frame — world in the dashboard, body in
/// the torque path. Coxa mass stays in the chassis: that servo does not swing.
pub fn robot_com(pos: V3, joints: &[[V3; 4]], phys: &Physics) -> V3 {
    let swing = phys.leg.total() * joints.len() as f64;
    let chassis = (phys.mass_kg - swing).max(0.15);
    let mut acc = [pos[0] * chassis, pos[1] * chassis, pos[2] * chassis];
    let mut m = chassis;
    let lm = phys.leg.total();
    if lm > 0.0 {
        for j in joints {
            let c = leg_com(j, &phys.leg, 1.0);
            acc[0] += c[0] * lm;
            acc[1] += c[1] * lm;
            acc[2] += c[2] * lm;
            m += lm;
        }
    }
    [acc[0] / m, acc[1] / m, acc[2] / m]
}

/// Torque each joint pays for carrying and swinging the leg itself, N-m.
///
/// Two terms, both of which used to be zero:
///
/// *Gravity.* A leg in the air still weighs something, and the joint above it
/// holds it up. The moment is the link's weight times the horizontal distance
/// from the joint to the link's centre of mass — the same Jacobian-transpose
/// argument used for the foot load, applied to the leg's own mass.
///
/// *Inertia.* Accelerating and stopping the leg costs `I * alpha`, where `I`
/// is the second moment of the mass distal to the joint about that joint's
/// axis. This is the diagonal of the mass matrix; the off-diagonal coupling
/// and the Coriolis terms are not modelled.
///
/// Both are returned as magnitudes, because the caller adds them to a torque
/// demand rather than integrating a signed equation of motion.
pub fn leg_torques(
    joints: &[V3; 4],
    ddq: [f64; 3],
    mass: &LegMass,
    scale: f64,
) -> [f64; 3] {
    if mass.total() <= 0.0 {
        return [0.0; 3];
    }
    let mid = |a: V3, b: V3| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
    let com_f = mid(joints[1], joints[2]);
    let com_t = mid(joints[2], joints[3]);

    // Horizontal lever arm, metres — what a vertical force acts through.
    let rx = |a: V3, b: V3| {
        let dx = b[0] - a[0];
        let dz = b[2] - a[2];
        (dx * dx + dz * dz).sqrt() * scale
    };
    // Full distance, metres — what a rotation acts through.
    let r3 = |a: V3, b: V3| {
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() * scale
    };

    let (mf, mt) = (mass.femur_kg, mass.tibia_kg);
    let (hip, knee, ankle) = (joints[0], joints[1], joints[2]);

    // Gravity. The coxa turns about a vertical axis, so leg weight makes no
    // moment on it.
    let g_knee = mf * G * rx(knee, com_f) + mt * G * rx(knee, com_t);
    let g_ankle = mt * G * rx(ankle, com_t);

    // Second moments about each joint axis. The coxa's axis is vertical, so
    // only the horizontal offset counts; the two pitch joints swing the leg in
    // its own plane, where the full distance does.
    let i_coxa = mf * rx(hip, com_f).powi(2) + mt * rx(hip, com_t).powi(2);
    let i_knee = mf * r3(knee, com_f).powi(2) + mt * r3(knee, com_t).powi(2);
    let i_ankle = mt * r3(ankle, com_t).powi(2);

    [
        i_coxa * ddq[0].abs(),
        g_knee + i_knee * ddq[1].abs(),
        g_ankle + i_ankle * ddq[2].abs(),
    ]
}

/// Scratch space for the finite differences the leg-inertia model needs.
#[derive(Clone)]
pub struct LegState {
    pub com: [V3; MAX_LEGS],
    pub com_vel: [V3; MAX_LEGS],
    pub dq: [[f64; 3]; MAX_LEGS],
    pub primed: bool,
}

impl Default for LegState {
    fn default() -> Self {
        LegState {
            com: [[0.0; 3]; MAX_LEGS],
            com_vel: [[0.0; 3]; MAX_LEGS],
            dq: [[0.0; 3]; MAX_LEGS],
            primed: false,
        }
    }
}

/// Static joint torques for one leg, newton-metres.
///
/// `joints` is the leg's forward kinematics in the body frame, in simulator
/// units; `scale` converts the lever arms to metres. `f_v` is the vertical
/// load the foot carries and `f_h` the horizontal traction it is transmitting,
/// both newtons.
///
/// For a vertical force the moment about a horizontal joint axis is the force
/// times the *horizontal* distance from the joint to the foot — the Jacobian
/// transpose specialised to a vertical load. The coxa turns about a vertical
/// axis, so a vertical load produces no moment on it and traction sizes it
/// instead.
#[inline]
pub fn joint_torques(joints: &[V3; 4], f_v: f64, f_h: f64, scale: f64) -> [f64; 3] {
    let foot = joints[3];
    let r = |a: V3| {
        let dx = foot[0] - a[0];
        let dz = foot[2] - a[2];
        (dx * dx + dz * dz).sqrt() * scale
    };
    [f_h * r(joints[0]), f_v * r(joints[1]), f_v * r(joints[2])]
}

/// How much the foot y-coordinate rises, in the body frame, per radian of
/// femur and tibia rotation. A loaded joint that gives way moves along the
/// positive direction of these — the leg folds and the chassis drops.
#[inline]
pub fn collapse_direction(q: [f64; 3], femur: f64, tibia: f64) -> [f64; 2] {
    let a23 = (q[1] + q[2]).cos();
    [femur * q[1].cos() + tibia * a23, tibia * a23]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::{fk_body, fk_world, Frame, FEMUR, TIBIA};

    #[test]
    fn the_ideal_joint_is_deadbeat_at_the_control_rate() {
        // kp * dt must be 1: any more and the joint overshoots its command
        // every tick and the "ideal" control case rings instead of tracking.
        let a = Actuator::ideal();
        assert!((a.kp * crate::sim::DT - 1.0).abs() < 1e-12);
    }

    #[test]
    fn torque_speed_line_runs_from_no_load_to_stall() {
        let a = Actuator::from_rating(0.16, 20.0);
        assert!((a.rate_limit(0.0) - a.omega_max).abs() < 1e-12);
        assert!(a.rate_limit(a.stall_nm) < 1e-12);
        let half = a.rate_limit(a.stall_nm * 0.5);
        assert!((half - a.omega_max * 0.5).abs() < 1e-9, "{half}");
    }

    #[test]
    fn a_joint_only_gives_way_past_stall() {
        let a = Actuator::from_rating(0.16, 20.0);
        assert_eq!(a.backdrive(a.stall_nm * 0.99), 0.0);
        assert!(a.backdrive(a.stall_nm * 1.5) > 0.0);
        // And it gives way faster the harder it is overloaded.
        assert!(a.backdrive(a.stall_nm * 2.0) > a.backdrive(a.stall_nm * 1.2));
    }

    #[test]
    fn rating_conversion_matches_the_catalogue_units() {
        // 0.16 s/60 deg is one sixth of a turn in 0.16 s.
        let a = Actuator::from_rating(0.16, 10.1972);
        assert!((a.stall_nm - 1.0).abs() < 1e-9, "{}", a.stall_nm);
        let rpm = a.omega_max * 60.0 / core::f64::consts::TAU;
        assert!((rpm - 62.5).abs() < 0.1, "{rpm} rpm");
    }

    #[test]
    fn torque_grows_with_lever_arm_and_scale() {
        let near = fk_body(Frame::default(), 3, [0.0, -0.5, -1.2]);
        let far = fk_body(Frame::default(), 3, [0.0, -0.1, -0.6]);
        let a = joint_torques(&near, 10.0, 3.0, 0.1);
        let b = joint_torques(&far, 10.0, 3.0, 0.1);
        assert!(b[1] > a[1], "further out must cost more: {a:?} vs {b:?}");
        let scaled = joint_torques(&far, 10.0, 3.0, 0.2);
        assert!((scaled[1] - b[1] * 2.0).abs() < 1e-9);
    }

    #[test]
    fn a_leg_in_the_air_still_has_to_be_held_up() {
        let f = Frame::default();
        let j = fk_body(f, 3, [0.0, 0.08, -1.33]);
        let m = LegMass::from_servo(0.060);
        let still = leg_torques(&j, [0.0; 3], &m, 0.10);
        // Gravity alone loads the two pitch joints and not the coxa, whose
        // axis is vertical.
        assert_eq!(still[0], 0.0);
        assert!(still[1] > 0.0 && still[2] > 0.0, "{still:?}");
        // The knee carries the tibia as well as the femur, so it pays more.
        assert!(still[1] > still[2]);
    }

    #[test]
    fn swinging_a_leg_costs_more_than_holding_it() {
        let f = Frame::default();
        let j = fk_body(f, 3, [0.0, 0.08, -1.33]);
        let m = LegMass::from_servo(0.060);
        let still = leg_torques(&j, [0.0; 3], &m, 0.10);
        let moving = leg_torques(&j, [12.0, 40.0, 40.0], &m, 0.10);
        for k in 0..3 {
            assert!(moving[k] >= still[k], "joint {k} got cheaper when moving");
        }
        assert!(moving[0] > 0.0, "the coxa should feel angular acceleration");
        // Inertia scales with the square of the link length.
        let big = leg_torques(&j, [12.0, 40.0, 40.0], &m, 0.20);
        let inert_small = moving[2] - still[2];
        let inert_big = big[2] - still[2] * 2.0;
        assert!(
            (inert_big / inert_small - 4.0).abs() < 1e-9,
            "inertia did not scale as length squared: {inert_big} vs {inert_small}"
        );
    }

    #[test]
    fn weightless_legs_cost_nothing_at_all() {
        let j = fk_body(Frame::default(), 3, [0.0, 0.08, -1.33]);
        let t = leg_torques(&j, [50.0; 3], &LegMass::WEIGHTLESS, 0.10);
        assert_eq!(t, [0.0; 3]);
        assert_eq!(leg_com(&j, &LegMass::WEIGHTLESS, 0.10), [0.0; 3]);
    }

    #[test]
    fn the_leg_centre_of_mass_sits_between_the_two_links() {
        let f = Frame::default();
        let j = fk_body(f, 3, [0.0, 0.08, -1.33]);
        let com = leg_com(&j, &LegMass::from_servo(0.060), 1.0);
        // Below the hip and outboard of it, but not past the foot.
        assert!(com[1] < j[1][1] && com[1] > j[3][1], "{com:?}");
        let out = |v: V3| (v[0] * v[0] + v[2] * v[2]).sqrt();
        assert!(out(com) > out(j[1]) && out(com) < out(j[3]), "{com:?}");
    }

    #[test]
    fn weightless_legs_put_the_centre_of_mass_on_the_chassis() {
        let phys = Physics {
            leg: LegMass::WEIGHTLESS,
            ..Physics::default()
        };
        let pos = [1.0, 2.0, 3.0];
        assert_eq!(robot_com(pos, &[], &phys), pos);
    }

    #[test]
    fn hanging_legs_pull_the_centre_of_mass_below_the_chassis() {
        let f = Frame::default();
        let phys = Physics::default();
        let pos = [0.0, 1.0, 0.0];
        let q = [0.0, 0.08, -1.33];
        let joints: Vec<_> = (0..f.legs())
            .map(|i| fk_world(f, i, q, pos, 0.0, 0.0, 0.0))
            .collect();
        let com = robot_com(pos, &joints, &phys);
        assert!(com[1] < pos[1] - 0.05, "{com:?}");
        assert!(com[1] > 0.0, "{com:?}");
        assert!(com[0].abs() < 0.25 && com[2].abs() < 0.25, "{com:?}");
    }

    #[test]
    fn collapsing_a_loaded_leg_raises_the_foot_toward_the_body() {
        let q = [0.0, -0.35, -1.1];
        let d = collapse_direction(q, FEMUR, TIBIA);
        let eps = 1e-6;
        let y0 = fk_body(Frame::default(), 3, q)[3][1];
        let y2 = fk_body(Frame::default(), 3, [q[0], q[1] + eps, q[2]])[3][1];
        let y3 = fk_body(Frame::default(), 3, [q[0], q[1], q[2] + eps])[3][1];
        assert!((((y2 - y0) / eps) - d[0]).abs() < 1e-4);
        assert!((((y3 - y0) / eps) - d[1]).abs() < 1e-4);
        assert!(d[0] > 0.0 && d[1] > 0.0);
    }
}
