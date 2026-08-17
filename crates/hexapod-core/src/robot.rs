//! Robot geometry and the analytic 3-DOF leg solver.
//!
//! The frame is parametric in leg count. Legs come in left/right pairs spread
//! front to back, indexed `[L1, R1, L2, R2, ...]` — pair `i` occupies indices
//! `2i` (left) and `2i+1` (right) — matching the row order of the gait diagram
//! in the dashboard. A hexapod is three pairs; the same code runs four, eight
//! or ten legs.

use crate::math::{body_to_world, clamp, hypot2, rot_y, world_to_body, V3};

/// Most legs the fixed-size arrays through the simulator will hold. Raising it
/// costs telemetry width and nothing else.
pub const MAX_LEGS: usize = 10;
pub const MIN_LEGS: usize = 4;

/// Chassis thickness. Independent of how many legs hang off it.
pub const BODY_H: f64 = 0.30;

pub const COXA: f64 = 0.30;
pub const FEMUR: f64 = 0.80;
pub const TIBIA: f64 = 1.00;

/// Planar reach limits of the femur/tibia pair, with a small margin so the
/// solver never sits exactly on a singularity.
pub const REACH_MAX: f64 = FEMUR + TIBIA - 0.03;
pub const REACH_MIN: f64 = 0.24;

/// Half-angle the outermost pair is swept to, radians. The front pair sits at
/// `+SPREAD` off the lateral axis and the rear pair at `-SPREAD`, with the
/// rest spaced evenly between. At three pairs this reproduces the original
/// hand-built hexapod exactly: 50, 0, -50 degrees.
const SPREAD: f64 = 50.0 * DEG;

/// Mechanical travel of each joint, radians: coxa, femur, tibia.
///
/// A servo horn stops somewhere, and a knee does not hyperextend. Without
/// these a joint that has been overloaded past stall keeps giving way for as
/// long as the load is there, and the chassis sinks forever instead of
/// arriving on the ground.
/// Collider radius of a leg link, simulator units. The plant builds capsules
/// this thick and the trajectory generator keeps them out of blocks.
pub const LINK_R: f64 = 0.05;

/// Rubber foot, simulator units. Centered on the kinematic foot so a tilted
/// tibia still meets the plane; the gait aims the kinematic point this far
/// above the ground so the ball kisses rather than tunnels.
pub const FOOT_R: f64 = 0.05;

pub const Q_LIMIT: [(f64, f64); 3] = [
    (-100.0 * DEG, 100.0 * DEG),
    (-110.0 * DEG, 110.0 * DEG),
    (-165.0 * DEG, -2.0 * DEG),
];

/// Clamp joint angles to the mechanical travel above.
#[inline]
pub fn clamp_joints(q: &mut [f64; 3]) {
    for (v, (lo, hi)) in q.iter_mut().zip(Q_LIMIT.iter()) {
        *v = clamp(*v, *lo, *hi);
    }
}

const DEG: f64 = core::f64::consts::PI / 180.0;

const NAMES: [&str; MAX_LEGS] = ["L1", "R1", "L2", "R2", "L3", "R3", "L4", "R4", "L5", "R5"];

/// How many legs the machine has, and everything that follows from it.
///
/// This is a value, not a constant, because the leg count is a thing you can
/// change in the dashboard. Every geometric quantity that used to be a `const`
/// hangs off it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frame {
    legs: usize,
}

impl Default for Frame {
    fn default() -> Self {
        Frame::new(6)
    }
}

impl Frame {
    /// Clamps to an even count within `MIN_LEGS..=MAX_LEGS`. Odd counts are
    /// rounded down: the whole frame is built out of left/right pairs, and a
    /// machine with an odd leg is a different machine.
    pub fn new(legs: usize) -> Frame {
        let legs = (legs & !1).clamp(MIN_LEGS, MAX_LEGS);
        Frame { legs }
    }

    #[inline]
    pub fn legs(&self) -> usize {
        self.legs
    }

    #[inline]
    pub fn pairs(&self) -> usize {
        self.legs / 2
    }

    /// Pair index and side (`false` left, `true` right) of a leg.
    #[inline]
    pub fn split(&self, leg: usize) -> (usize, bool) {
        (leg / 2, leg % 2 == 1)
    }

    /// Circumradius of the chassis. It has to grow with the leg count or the
    /// hips end up on top of each other.
    #[inline]
    pub fn body_r(&self) -> f64 {
        0.65 + 0.10 * self.pairs() as f64
    }

    /// Radius at which the hips are mounted.
    #[inline]
    pub fn hip_r(&self) -> f64 {
        self.body_r() - 0.15
    }

    /// Outward yaw of a leg, radians. `dir = [cos a, 0, sin a]`, `+Z` forward.
    pub fn yaw(&self, leg: usize) -> f64 {
        let (pair, right) = self.split(leg);
        let p = self.pairs();
        let a = if p < 2 {
            0.0
        } else {
            SPREAD * (1.0 - 2.0 * pair as f64 / (p - 1) as f64)
        };
        if right {
            a
        } else {
            core::f64::consts::PI - a
        }
    }

    /// Outward unit vector of a leg in the body frame.
    #[inline]
    pub fn dir(&self, leg: usize) -> V3 {
        let a = self.yaw(leg);
        [a.cos(), 0.0, a.sin()]
    }

    /// Hip mount point of a leg in the body frame.
    #[inline]
    pub fn hip(&self, leg: usize) -> V3 {
        let d = self.dir(leg);
        let r = self.hip_r();
        [d[0] * r, 0.0, d[2] * r]
    }

    pub fn name(&self, leg: usize) -> &'static str {
        NAMES[leg.min(MAX_LEGS - 1)]
    }

    /// Name of the gait an alternating preset produces on this frame. Six legs
    /// alternating is the tripod; four legs is the trot.
    pub fn alternating_name(&self) -> &'static str {
        match self.legs {
            4 => "TROT",
            6 => "TRIPOD",
            8 => "TETRAPOD",
            _ => "ALTERNATE",
        }
    }

    /// Whether an alternating half-set gait leaves the machine statically
    /// stable. Four legs alternating is a trot, which stands on two diagonal
    /// feet — a line, not a polygon. Real quadrupeds trot perfectly well
    /// because a trot is *dynamically* stable, and this simulator judges
    /// stability by where the centre of mass sits relative to the support
    /// polygon. So on four legs the alternating preset is offered and it falls
    /// over, which is the honest answer for the model being run.
    #[inline]
    pub fn alternating_is_stable(&self) -> bool {
        self.legs >= 6
    }

    /// The preset that actually works on this frame. Six legs and up get the
    /// alternating gait; four legs have to crawl.
    #[inline]
    pub fn default_preset_index(&self) -> u32 {
        if self.alternating_is_stable() {
            0
        } else {
            2
        }
    }

    /// What the machine is called.
    pub fn label(&self) -> &'static str {
        match self.legs {
            4 => "QUADRUPED",
            6 => "HEXAPOD",
            8 => "OCTOPOD",
            10 => "DECAPOD",
            _ => "WALKER",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Solution {
    /// Coxa yaw, femur pitch, tibia pitch.
    pub q: [f64; 3],
    /// How far past the reachable envelope the target was, in metres.
    /// Zero when the target was reachable.
    pub strain: f64,
}

/// Analytic inverse kinematics for one leg.
///
/// `target` is in the body frame. Returns joint angles for the knee-up
/// configuration, plus the amount by which the request exceeded the leg's
/// envelope (the target is clamped to the envelope in that case).
pub fn solve_ik(frame: Frame, leg: usize, target: V3) -> Solution {
    // Into the leg frame: translate to the hip, then undo the mount yaw so the
    // leg points along local +X.
    let h = frame.hip(leg);
    let rel = [target[0] - h[0], target[1] - h[1], target[2] - h[2]];
    let t = rot_y(rel, -frame.yaw(leg));

    let q1 = t[2].atan2(t[0]);

    let planar = hypot2(t[0], t[2]) - COXA;
    let dy = t[1];
    let d_raw = hypot2(planar, dy);

    let d = clamp(d_raw, REACH_MIN, REACH_MAX);
    let strain = if d_raw > REACH_MAX {
        d_raw - REACH_MAX
    } else if d_raw < REACH_MIN {
        REACH_MIN - d_raw
    } else {
        0.0
    };

    // Law of cosines. q3 is negative: the knee folds downward.
    let c3 = clamp(
        (d * d - FEMUR * FEMUR - TIBIA * TIBIA) / (2.0 * FEMUR * TIBIA),
        -1.0,
        1.0,
    );
    let q3 = -c3.acos();

    let c2 = clamp(
        (d * d + FEMUR * FEMUR - TIBIA * TIBIA) / (2.0 * FEMUR * d),
        -1.0,
        1.0,
    );
    let q2 = dy.atan2(planar) + c2.acos();

    Solution {
        q: [q1, q2, q3],
        strain,
    }
}

/// Forward kinematics: hip, knee, ankle and foot of leg `i` in the body frame.
pub fn fk_body(frame: Frame, leg: usize, q: [f64; 3]) -> [V3; 4] {
    let h = frame.hip(leg);
    let a = frame.yaw(leg) + q[0];
    let (sa, ca) = a.sin_cos();

    let knee = [h[0] + ca * COXA, h[1], h[2] + sa * COXA];

    let (s2, c2) = q[1].sin_cos();
    let ankle = [
        knee[0] + ca * FEMUR * c2,
        knee[1] + FEMUR * s2,
        knee[2] + sa * FEMUR * c2,
    ];

    let a23 = q[1] + q[2];
    let (s23, c23) = a23.sin_cos();
    let foot = [
        ankle[0] + ca * TIBIA * c23,
        ankle[1] + TIBIA * s23,
        ankle[2] + sa * TIBIA * c23,
    ];

    [h, knee, ankle, foot]
}

/// Forward kinematics in world space, given the body pose.
pub fn fk_world(
    frame: Frame,
    leg: usize,
    q: [f64; 3],
    pos: V3,
    yaw: f64,
    pitch: f64,
    roll: f64,
) -> [V3; 4] {
    let local = fk_body(frame, leg, q);
    let mut out = [[0.0; 3]; 4];
    for k in 0..4 {
        let w = body_to_world(local[k], yaw, pitch, roll);
        out[k] = [pos[0] + w[0], pos[1] + w[1], pos[2] + w[2]];
    }
    out
}

/// World point into the body frame.
#[inline]
pub fn to_body(p: V3, pos: V3, yaw: f64, pitch: f64, roll: f64) -> V3 {
    world_to_body([p[0] - pos[0], p[1] - pos[1], p[2] - pos[2]], yaw, pitch, roll)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(a: V3, b: V3) -> f64 {
        let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    /// Every even leg count the frame accepts.
    fn frames() -> Vec<Frame> {
        (MIN_LEGS..=MAX_LEGS).step_by(2).map(Frame::new).collect()
    }

    #[test]
    fn the_six_legged_frame_matches_the_original_hand_built_geometry() {
        // The hexapod numbers this project started from, now emitted by the
        // general formula rather than a table. If this drifts, every result
        // recorded in the README stops being comparable.
        let f = Frame::new(6);
        let want = [130.0, 50.0, 180.0, 0.0, 230.0, -50.0];
        for (leg, deg) in want.iter().enumerate() {
            assert!(
                (f.yaw(leg) - deg * DEG).abs() < 1e-12,
                "leg {leg}: {} deg, wanted {deg}",
                f.yaw(leg) / DEG
            );
        }
        assert!((f.body_r() - 0.95).abs() < 1e-12);
        assert!((f.hip_r() - 0.80).abs() < 1e-12);
        assert_eq!(f.label(), "HEXAPOD");
        assert_eq!(f.alternating_name(), "TRIPOD");
    }

    #[test]
    fn leg_counts_are_even_and_bounded() {
        assert_eq!(Frame::new(7).legs(), 6);
        assert_eq!(Frame::new(0).legs(), MIN_LEGS);
        assert_eq!(Frame::new(999).legs(), MAX_LEGS);
        assert_eq!(Frame::new(8).pairs(), 4);
    }

    #[test]
    fn legs_are_mirrored_pairs_spread_front_to_back() {
        for f in frames() {
            let mut prev = f64::INFINITY;
            for pair in 0..f.pairs() {
                let (l, r) = (2 * pair, 2 * pair + 1);
                let (dl, dr) = (f.dir(l), f.dir(r));
                // Mirror across the centre line: same Z, opposite X.
                assert!((dl[2] - dr[2]).abs() < 1e-12, "pair {pair} not mirrored");
                assert!((dl[0] + dr[0]).abs() < 1e-12, "pair {pair} not mirrored");
                // Left is to the left, right is to the right.
                assert!(dl[0] < 0.0 && dr[0] > 0.0, "pair {pair} sides swapped");
                // Pairs march monotonically from front to back.
                assert!(dr[2] < prev, "pair {pair} is not behind the one before");
                prev = dr[2];
            }
        }
    }

    #[test]
    fn hips_never_collide_however_many_legs() {
        for f in frames() {
            for a in 0..f.legs() {
                for b in (a + 1)..f.legs() {
                    let d = dist(f.hip(a), f.hip(b));
                    assert!(
                        d > COXA,
                        "{} legs: hips {a} and {b} are {d:.3} apart",
                        f.legs()
                    );
                }
            }
        }
    }

    #[test]
    fn ik_then_fk_returns_the_requested_foot() {
        // Sweep reachable targets around each leg's neutral stance, on every
        // frame — the solver must not know how many legs there are.
        for f in frames() {
            for leg in 0..f.legs() {
                let d = f.dir(leg);
                for &out in &[1.15, 1.42, 1.65] {
                    for &down in &[0.55, 0.88, 1.15] {
                        for &fwd in &[-0.35, 0.0, 0.35] {
                            let target = [
                                f.hip(leg)[0] + d[0] * (out - f.hip_r()) + d[2] * fwd,
                                -down,
                                f.hip(leg)[2] + d[2] * (out - f.hip_r()) - d[0] * fwd,
                            ];
                            let s = solve_ik(f, leg, target);
                            assert_eq!(
                                s.strain,
                                0.0,
                                "{} legs, leg {leg}, target {target:?} unreachable",
                                f.legs()
                            );
                            let got = fk_body(f, leg, s.q)[3];
                            assert!(
                                dist(got, target) < 1e-9,
                                "leg {leg}: wanted {target:?}, got {got:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn unreachable_targets_report_strain_and_clamp() {
        let f = Frame::default();
        let d = f.dir(0);
        let far = [d[0] * 6.0, -0.5, d[2] * 6.0];
        let s = solve_ik(f, 0, far);
        assert!(s.strain > 1.0, "strain={}", s.strain);
        // The returned pose is still valid, just short of the request.
        let foot = fk_body(f, 0, s.q)[3];
        assert!(foot[0].is_finite() && foot[1].is_finite());
    }

    #[test]
    fn knee_stays_above_the_foot() {
        let f = Frame::default();
        let s = solve_ik(f, 3, [f.dir(3)[0] * 1.42, -0.88, f.dir(3)[2] * 1.42]);
        let j = fk_body(f, 3, s.q);
        assert!(j[2][1] > j[3][1], "ankle should sit above the foot");
    }

    #[test]
    fn the_neutral_stance_sits_inside_the_joint_travel() {
        // If the limits excluded the working pose the robot could not stand,
        // and that has to hold for every frame, not just the hexapod.
        for f in frames() {
            for leg in 0..f.legs() {
                let d = f.dir(leg);
                let s = solve_ik(f, leg, [d[0] * 1.42, -0.88, d[2] * 1.42]);
                let mut q = s.q;
                clamp_joints(&mut q);
                assert_eq!(
                    q,
                    s.q,
                    "{} legs, leg {leg}: neutral pose {:?} hits a stop",
                    f.legs(),
                    s.q
                );
            }
        }
    }

    #[test]
    fn a_collapsing_leg_runs_out_of_travel() {
        let mut q = [0.0, 9.0, 9.0];
        clamp_joints(&mut q);
        assert_eq!(q[1], Q_LIMIT[1].1);
        assert_eq!(q[2], Q_LIMIT[2].1);
        // And the knee cannot hyperextend past straight.
        assert!(Q_LIMIT[2].1 < 0.0);
    }

    #[test]
    fn world_round_trip() {
        let f = Frame::default();
        let pos = [1.0, 0.9, 4.0];
        let (yaw, pitch, roll) = (0.4, 0.1, -0.05);
        let s = solve_ik(f, 2, [f.dir(2)[0] * 1.4, -0.88, f.dir(2)[2] * 1.4]);
        let w = fk_world(f, 2, s.q, pos, yaw, pitch, roll)[3];
        let b = to_body(w, pos, yaw, pitch, roll);
        let expect = fk_body(f, 2, s.q)[3];
        assert!(dist(b, expect) < 1e-9);
    }
}
