//! The thing being optimised: a gait parameterisation plus a linear feedback
//! policy, packed into one flat parameter vector `theta`.
//!
//! Layout of `theta`, for a machine with `n` legs:
//! ```text
//!   [0 ..  5]        gait scalars, squashed into physical ranges
//!   [6 .. 6+n)       per-leg phase offset deltas, relative to the seeded gait
//!   [6+n, 7+n]       lateral stance trim, front pair / rear pair
//!   [8+n ..  ]       the n_act x n_obs feedback matrix, row-major
//! ```
//! Starting the feedback block at zero makes the initial policy exactly the
//! hand-tuned gait, so iteration 0 of training *is* the baseline. Yaw toward
//! the next waypoint lives in the plant, not in this matrix: a zero steer row
//! still weaves a slalom.
//!
//! The gait scalars are the *nominal* gait. Three of the actions — cycle time,
//! stride and duty factor — scale them online every tick. Those three are how
//! any legged controller changes speed, and without them a gait has exactly
//! one speed it walks well at, so a commanded speed is not something a policy
//! can be asked to hold. A seventh action, jump, is a takeoff trigger: above
//! a threshold it crouches, pushes and lifts every foot at once, which is the
//! only way this machine leaves the ground. The walking gait never does that,
//! because its duty factor cannot, and the seeded feedback is zero, so a
//! walking rollout of iteration 0 is bit-identical to what it was.
//!
//! Every dimension here is a function of the leg count, so the vectors are
//! sized at `MAX_*` and the live length is carried by the [`Frame`].

use crate::math::{clamp, frac, squash, unsquash};
use crate::math::V3;
use crate::robot::{fk_world, solve_ik, Frame, MAX_LEGS};
use crate::terrain::Terrain;

/// Observations that do not depend on how many legs there are: body height
/// error, pitch, roll, stability margin, gait phase as sin/cos, and speed
/// error against the command.
pub const N_FIXED_OBS: usize = 7;
/// Probes in the forward terrain scan: two ranges by three bearings. This is
/// the only thing the policy knows about ground it has not reached yet, and
/// without it steering round an obstacle is guesswork.
pub const N_SCAN: usize = 6;
/// Observations that follow the per-leg block: commanded speed, bearing and
/// range to the next waypoint, where the machine is between the two walls, the
/// scan, vertical velocity, and whether the machine is airborne.
pub const N_TAIL_OBS: usize = 6 + N_SCAN;
/// Actions that do not depend on leg count: body height trim, pitch trim, the
/// three gait modulations, jump, and steering.
pub const N_FIXED_ACT: usize = 7;
/// Gait entries that do not depend on leg count: six scalars and two trims.
pub const N_FIXED_GAIT: usize = 8;

/// Ceilings for the fixed-size arrays. One terrain lookahead per leg plus the
/// tail block; two actions per leg plus the fixed block.
pub const MAX_OBS: usize = N_FIXED_OBS + MAX_LEGS + N_TAIL_OBS;
pub const MAX_ACT: usize = N_FIXED_ACT + 2 * MAX_LEGS;

/// Observation count for a frame: the fixed block, one terrain lookahead under
/// each leg's predicted touchdown, and the navigation tail.
#[inline]
pub fn n_obs(frame: Frame) -> usize {
    N_FIXED_OBS + frame.legs() + N_TAIL_OBS
}

/// Action count: per-leg step height and touchdown offset, plus the fixed
/// block.
#[inline]
pub fn n_act(frame: Frame) -> usize {
    2 * frame.legs() + N_FIXED_ACT
}

#[inline]
pub fn n_gait(frame: Frame) -> usize {
    N_FIXED_GAIT + frame.legs()
}

#[inline]
pub fn n_theta(frame: Frame) -> usize {
    n_gait(frame) + n_act(frame) * n_obs(frame)
}

/// Indices of the tail observations, after the per-leg lookaheads.
#[inline]
pub fn obs_cmd_speed(frame: Frame) -> usize {
    N_FIXED_OBS + frame.legs()
}
/// Signed heading error to the next waypoint, in half-turns.
#[inline]
pub fn obs_bearing(frame: Frame) -> usize {
    N_FIXED_OBS + frame.legs() + 1
}
/// Range to the next waypoint.
#[inline]
pub fn obs_range(frame: Frame) -> usize {
    N_FIXED_OBS + frame.legs() + 2
}
/// Where the machine sits between the two walls, in [-1, 1].
#[inline]
pub fn obs_corridor(frame: Frame) -> usize {
    N_FIXED_OBS + frame.legs() + 3
}
/// First of the `N_SCAN` forward terrain probes.
#[inline]
pub fn obs_scan(frame: Frame) -> usize {
    N_FIXED_OBS + frame.legs() + 4
}
/// Vertical velocity, normalised by a few metres per second.
#[inline]
pub fn obs_vy(frame: Frame) -> usize {
    obs_scan(frame) + N_SCAN
}
/// 1 while no foot is in contact, 0 otherwise.
#[inline]
pub fn obs_airborne(frame: Frame) -> usize {
    obs_vy(frame) + 1
}

/// Indices of the fixed actions, after the two per-leg blocks.
#[inline]
pub fn act_body_dh(frame: Frame) -> usize {
    2 * frame.legs()
}
#[inline]
pub fn act_pitch(frame: Frame) -> usize {
    2 * frame.legs() + 1
}
#[inline]
pub fn act_cycle(frame: Frame) -> usize {
    2 * frame.legs() + 2
}
#[inline]
pub fn act_stride(frame: Frame) -> usize {
    2 * frame.legs() + 3
}
#[inline]
pub fn act_duty(frame: Frame) -> usize {
    2 * frame.legs() + 4
}
/// Takeoff trigger. Above a threshold the simulator runs a hop: crouch, push,
/// lift every foot at once, which is the only way the machine leaves the
/// ground. The walking gait never does that, because its duty factor cannot,
/// and the seeded weight is zero, so a walking rollout does not jump.
#[inline]
pub fn act_jump(frame: Frame) -> usize {
    2 * frame.legs() + 5
}
/// Yaw command, in units of [`TURN_RATE`]. This is the machine steering
/// itself; without it a waypoint is something to be told about and not
/// something that can be reached.
///
/// [`TURN_RATE`]: crate::sim::TURN_RATE
#[inline]
pub fn act_steer(frame: Frame) -> usize {
    2 * frame.legs() + 6
}

/// `(lo, hi)` for each squashed gait scalar.
pub const GAIT_BOUNDS: [(f64, f64); 6] = [
    (0.32, 1.10), // cycle time, s
    (0.30, 1.45), // stride, m
    (0.08, 0.75), // step height, m
    (0.55, 1.25), // body height, m
    (1.70, 3.30), // stance width, m
    (0.45, 0.92), // duty factor
];

pub const GAIT_LABELS: [&str; 6] = [
    "CYCLE TIME",
    "STRIDE LENGTH",
    "STEP HEIGHT",
    "BODY HEIGHT",
    "STANCE WIDTH",
    "DUTY FACTOR",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// Alternating half-sets. On six legs this is the tripod, on four the
    /// trot, on eight a pair of tetrapods.
    Tripod = 0,
    Ripple = 1,
    Wave = 2,
}

impl Preset {
    pub fn from_u32(v: u32) -> Preset {
        match v {
            1 => Preset::Ripple,
            2 => Preset::Wave,
            _ => Preset::Tripod,
        }
    }

    /// The preset a frame should start on: the alternating gait where it is
    /// statically stable, the crawl where it is not.
    pub fn default_for(frame: Frame) -> Preset {
        Preset::from_u32(frame.default_preset_index())
    }

    pub fn name(self, frame: Frame) -> &'static str {
        match self {
            Preset::Tripod => frame.alternating_name(),
            Preset::Ripple => "RIPPLE",
            Preset::Wave => "WAVE",
        }
    }

    /// Phase offsets in `[L1, R1, L2, R2, ...]` order.
    ///
    /// All three are closed forms in the pair index `i` and the side `s`, and
    /// all three reproduce the original hand-built hexapod exactly at three
    /// pairs — which is worth checking, and is.
    pub fn offsets(self, frame: Frame) -> [f64; MAX_LEGS] {
        let p = frame.pairs() as f64;
        let n = frame.legs() as f64;
        // Both sequenced gaits put the two sides half a cycle apart. Ripple
        // steps by pair, so on an even pair count half a cycle lands the right
        // side exactly on the left side's phases; nudging it by half a leg-step
        // spaces all `n` legs evenly instead. Wave steps by leg, where half a
        // cycle is already exactly `p` leg-steps, so it never needs the nudge —
        // and neither does an odd pair count, which is why the three-pair
        // hexapod comes out identical to the original hand-written table.
        let ripple_side = if frame.pairs().is_multiple_of(2) {
            0.5 + 0.5 / p
        } else {
            0.5
        };
        let mut out = [0.0; MAX_LEGS];
        for leg in 0..frame.legs() {
            let (pair, right) = frame.split(leg);
            let i = pair as f64;
            let s = f64::from(u8::from(right));
            out[leg] = match self {
                // Neighbouring pairs in opposite half-sets, sides likewise, so
                // the legs that are down always form a spread-out set.
                Preset::Tripod => 0.5 * ((pair + usize::from(right)) % 2) as f64,
                // A metachronal wave running back to front, the two sides half
                // a cycle apart.
                Preset::Ripple => frac(-i / p + s * ripple_side),
                // One leg at a time: the same wave, but spaced by leg rather
                // than by pair, so no two ever lift together.
                Preset::Wave => frac(i / n + s * 0.5),
            };
        }
        out
    }

    /// Fraction of the cycle a leg spends on the ground, for this pattern.
    /// Alternating halves need half; the two wave patterns keep all but two,
    /// or all but one, legs down at a time.
    pub fn duty(self, frame: Frame) -> f64 {
        let n = frame.legs() as f64;
        match self {
            Preset::Tripod => 0.50,
            Preset::Ripple => 1.0 - 2.0 / n,
            Preset::Wave => 1.0 - 1.0 / n,
        }
    }
}

/// Decoded, physically meaningful gait parameters.
#[derive(Clone, Copy, Debug)]
pub struct Gait {
    pub frame: Frame,
    pub cycle: f64,
    pub stride: f64,
    pub step_h: f64,
    pub body_h: f64,
    pub stance_w: f64,
    pub duty: f64,
    pub offsets: [f64; MAX_LEGS],
    pub trim_front: f64,
    pub trim_rear: f64,
}

impl Gait {
    /// Body speed that keeps the feet from scrubbing: the chassis must travel
    /// exactly one stride while a leg is on the ground.
    #[inline]
    pub fn nominal_speed(&self) -> f64 {
        self.stride / (self.duty * self.cycle)
    }

    /// Lateral stance trim for a leg. The outermost pairs are trimmed, the
    /// ones in between are not.
    #[inline]
    pub fn trim(&self, leg: usize) -> f64 {
        let (pair, _) = self.frame.split(leg);
        if pair == 0 {
            self.trim_front
        } else if pair + 1 == self.frame.pairs() {
            self.trim_rear
        } else {
            0.0
        }
    }
}

/// Welford running mean/variance over observations — the state normaliser
/// from ARS V2. Without it the terrain-lookahead inputs, which are an order of
/// magnitude smaller than the phase inputs, get effectively ignored.
#[derive(Clone)]
pub struct Normalizer {
    pub n: f64,
    pub mean: [f64; MAX_OBS],
    pub m2: [f64; MAX_OBS],
    pub frozen: bool,
}

impl Default for Normalizer {
    fn default() -> Self {
        Normalizer {
            n: 0.0,
            mean: [0.0; MAX_OBS],
            m2: [1.0; MAX_OBS],
            frozen: false,
        }
    }
}

impl Normalizer {
    pub fn observe(&mut self, obs: &[f64; MAX_OBS], n_obs: usize) {
        if self.frozen {
            return;
        }
        self.n += 1.0;
        for i in 0..n_obs {
            let d = obs[i] - self.mean[i];
            self.mean[i] += d / self.n;
            self.m2[i] += d * (obs[i] - self.mean[i]);
        }
    }

    #[inline]
    pub fn std(&self, i: usize) -> f64 {
        if self.n < 2.0 {
            return 1.0;
        }
        (self.m2[i] / self.n).sqrt().max(1e-3)
    }

    #[inline]
    pub fn apply(&self, obs: &[f64; MAX_OBS], out: &mut [f64; MAX_OBS], n_obs: usize) {
        if self.n < 2.0 {
            *out = *obs;
            return;
        }
        for i in 0..n_obs {
            out[i] = clamp((obs[i] - self.mean[i]) / self.std(i), -8.0, 8.0);
        }
    }
}

#[derive(Clone)]
pub struct Policy {
    pub frame: Frame,
    pub theta: Vec<f64>,
    pub base_offsets: [f64; MAX_LEGS],
    pub norm: Normalizer,
    /// Scales the feedback layer. Zero reduces the policy to its open-loop gait.
    pub feedback: f64,
}

impl Policy {
    /// Seed from a preset gait and the hand-tuned parameters shown in the UI.
    pub fn seeded(preset: Preset, frame: Frame) -> Policy {
        let base = [0.47, 1.08, 0.46, 0.88, 2.84, preset.duty(frame)];
        let mut theta = vec![0.0; n_theta(frame)];
        for i in 0..6 {
            theta[i] = unsquash(base[i], GAIT_BOUNDS[i].0, GAIT_BOUNDS[i].1);
        }
        Policy {
            frame,
            theta,
            base_offsets: preset.offsets(frame),
            norm: Normalizer::default(),
            feedback: 1.0,
        }
    }

    #[inline]
    pub fn n_obs(&self) -> usize {
        n_obs(self.frame)
    }

    #[inline]
    pub fn n_act(&self) -> usize {
        n_act(self.frame)
    }

    pub fn gait(&self) -> Gait {
        let t = &self.theta;
        let n = self.frame.legs();
        let mut offsets = [0.0; MAX_LEGS];
        for i in 0..n {
            // +-0.5 of a cycle around the seeded offset.
            offsets[i] = frac(self.base_offsets[i] + 0.5 * t[6 + i].tanh());
        }
        Gait {
            frame: self.frame,
            cycle: squash(t[0], GAIT_BOUNDS[0].0, GAIT_BOUNDS[0].1),
            stride: squash(t[1], GAIT_BOUNDS[1].0, GAIT_BOUNDS[1].1),
            step_h: squash(t[2], GAIT_BOUNDS[2].0, GAIT_BOUNDS[2].1),
            body_h: squash(t[3], GAIT_BOUNDS[3].0, GAIT_BOUNDS[3].1),
            stance_w: squash(t[4], GAIT_BOUNDS[4].0, GAIT_BOUNDS[4].1),
            duty: squash(t[5], GAIT_BOUNDS[5].0, GAIT_BOUNDS[5].1),
            offsets,
            trim_front: 0.25 * t[6 + n].tanh(),
            trim_rear: 0.25 * t[7 + n].tanh(),
        }
    }

    /// `a = tanh(W * normalise(obs))`, so every action is bounded in (-1, 1).
    pub fn act(&self, obs: &[f64; MAX_OBS], out: &mut [f64; MAX_ACT]) {
        let (no, na) = (self.n_obs(), self.n_act());
        let mut s = [0.0f64; MAX_OBS];
        self.norm.apply(obs, &mut s, no);
        let base = n_gait(self.frame);
        for a in 0..na {
            let row = base + a * no;
            let mut acc = 0.0;
            for o in 0..no {
                acc += self.theta[row + o] * s[o];
            }
            out[a] = (acc * self.feedback).tanh();
        }
    }

    /// L2 norm of the feedback block — how far the learner has moved away
    /// from pure open-loop walking.
    pub fn feedback_norm(&self) -> f64 {
        self.theta[n_gait(self.frame)..]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames() -> Vec<Frame> {
        (crate::robot::MIN_LEGS..=MAX_LEGS)
            .step_by(2)
            .map(Frame::new)
            .collect()
    }

    #[test]
    fn seeded_policy_reproduces_the_hand_tuned_gait() {
        let f = Frame::new(6);
        let p = Policy::seeded(Preset::Tripod, f);
        let g = p.gait();
        assert!((g.cycle - 0.47).abs() < 1e-6);
        assert!((g.stride - 1.08).abs() < 1e-6);
        assert!((g.step_h - 0.46).abs() < 1e-6);
        assert!((g.body_h - 0.88).abs() < 1e-6);
        assert!((g.stance_w - 2.84).abs() < 1e-6);
        assert!((g.duty - 0.50).abs() < 1e-6);
        assert_eq!(g.offsets, Preset::Tripod.offsets(f));
    }

    #[test]
    fn the_six_legged_presets_match_the_original_tables() {
        // The hand-built hexapod offsets, now produced by closed forms that
        // also work for four, eight and ten legs.
        const S: f64 = 1.0 / 6.0;
        let f = Frame::new(6);
        let want: [(Preset, [f64; 6], f64); 3] = [
            (Preset::Tripod, [0.0, 0.5, 0.5, 0.0, 0.0, 0.5], 0.5),
            (
                Preset::Ripple,
                [0.0, 3.0 * S, 4.0 * S, S, 2.0 * S, 5.0 * S],
                2.0 / 3.0,
            ),
            (
                Preset::Wave,
                [0.0, 3.0 * S, S, 4.0 * S, 2.0 * S, 5.0 * S],
                5.0 / 6.0,
            ),
        ];
        for (preset, offsets, duty) in want {
            let got = preset.offsets(f);
            for (leg, o) in offsets.iter().enumerate() {
                assert!(
                    (got[leg] - o).abs() < 1e-12,
                    "{preset:?} leg {leg}: {} != {o}",
                    got[leg]
                );
            }
            assert!((preset.duty(f) - duty).abs() < 1e-12, "{preset:?} duty");
        }
    }

    #[test]
    fn zero_feedback_block_gives_zero_action() {
        for f in frames() {
            let p = Policy::seeded(Preset::Tripod, f);
            let obs = [0.3; MAX_OBS];
            let mut act = [9.0; MAX_ACT];
            p.act(&obs, &mut act);
            assert!(act[..p.n_act()].iter().all(|a| *a == 0.0));
        }
    }

    #[test]
    fn a_quadruped_starts_on_the_crawl_because_a_trot_is_not_static() {
        let f = Frame::new(4);
        assert!(!f.alternating_is_stable());
        assert_eq!(Preset::default_for(f), Preset::Wave);
        assert_eq!(Preset::default_for(Frame::new(6)), Preset::Tripod);
    }

    #[test]
    fn every_leg_gets_a_distinct_phase_in_the_sequenced_gaits() {
        for f in frames() {
            for preset in [Preset::Ripple, Preset::Wave] {
                let o = preset.offsets(f);
                for a in 0..f.legs() {
                    for b in (a + 1)..f.legs() {
                        assert!(
                            (o[a] - o[b]).abs() > 1e-9,
                            "{} legs, {preset:?}: legs {a} and {b} share a phase",
                            f.legs()
                        );
                    }
                }
                assert!(o[..f.legs()].iter().all(|v| (0.0..1.0).contains(v)));
            }
        }
    }

    #[test]
    fn alternating_splits_the_legs_into_two_equal_groups() {
        for f in frames() {
            let o = Preset::Tripod.offsets(f);
            let down = o[..f.legs()].iter().filter(|v| **v == 0.0).count();
            assert_eq!(
                down * 2,
                f.legs(),
                "{} legs: {down} in the first group",
                f.legs()
            );
        }
        // And on a hexapod that group is a tripod: one side's front and rear,
        // the other side's middle.
        assert_eq!(Frame::new(6).alternating_name(), "TRIPOD");
        assert_eq!(Frame::new(4).alternating_name(), "TROT");
    }

    #[test]
    fn sequenced_gaits_keep_more_legs_down_the_more_there_are() {
        // A ten-legged wave has nine feet on the ground; a four-legged one has
        // three. Duty has to follow the leg count or the machine falls over.
        let mut prev = 0.0;
        for f in frames() {
            let d = Preset::Wave.duty(f);
            assert!(d > prev, "{} legs: duty {d} did not rise", f.legs());
            assert!(d > Preset::Ripple.duty(f));
            prev = d;
        }
    }

    #[test]
    fn dimensions_follow_the_leg_count() {
        for f in frames() {
            let n = f.legs();
            assert_eq!(n_obs(f), N_FIXED_OBS + n + N_TAIL_OBS);
            assert_eq!(n_act(f), 2 * n + N_FIXED_ACT);
            assert!(n_obs(f) <= MAX_OBS && n_act(f) <= MAX_ACT);
            assert_eq!(n_theta(f), n_gait(f) + n_act(f) * n_obs(f));
            assert_eq!(Policy::seeded(Preset::Tripod, f).theta.len(), n_theta(f));
            // The action indices must not collide with the per-leg block.
            assert_eq!(act_body_dh(f), 2 * n);
            assert_eq!(act_steer(f), n_act(f) - 1);
            // Nor may the tail observations, and the scan plus the jump
            // sensors must fit inside it.
            assert_eq!(obs_cmd_speed(f), N_FIXED_OBS + n);
            assert_eq!(obs_airborne(f) + 1, n_obs(f));
            assert_eq!(act_jump(f) + 1, act_steer(f));
        }
        // The hexapod, with the navigation tail, the jump sensors, and the
        // jump action sitting in front of steering.
        assert_eq!(n_obs(Frame::new(6)), 25);
        assert_eq!(n_act(Frame::new(6)), 19);
        assert_eq!(n_theta(Frame::new(6)), 489);
    }

    #[test]
    fn gait_stays_inside_bounds_for_extreme_theta() {
        for f in frames() {
            let mut p = Policy::seeded(Preset::Tripod, f);
            for (i, v) in p.theta.iter_mut().enumerate() {
                *v = if i % 2 == 0 { 40.0 } else { -40.0 };
            }
            let g = p.gait();
            let vals = [g.cycle, g.stride, g.step_h, g.body_h, g.stance_w, g.duty];
            for (i, v) in vals.iter().enumerate() {
                assert!(
                    *v >= GAIT_BOUNDS[i].0 - 1e-9 && *v <= GAIT_BOUNDS[i].1 + 1e-9,
                    "{} out of bounds: {v}",
                    GAIT_LABELS[i]
                );
            }
            assert!(g.offsets[..f.legs()].iter().all(|o| (0.0..1.0).contains(o)));
        }
    }

    #[test]
    fn only_the_outermost_pairs_are_trimmed() {
        let f = Frame::new(10);
        let mut p = Policy::seeded(Preset::Tripod, f);
        p.theta[6 + f.legs()] = 4.0;
        p.theta[7 + f.legs()] = -4.0;
        let g = p.gait();
        assert!(g.trim(0) > 0.1 && g.trim(1) > 0.1, "front pair untrimmed");
        assert!(g.trim(8) < -0.1 && g.trim(9) < -0.1, "rear pair untrimmed");
        for leg in 2..8 {
            assert_eq!(g.trim(leg), 0.0, "leg {leg} should not be trimmed");
        }
    }

    #[test]
    fn normaliser_whitens() {
        let f = Frame::default();
        let no = n_obs(f);
        let mut n = Normalizer::default();
        let mut r = crate::math::Rng::new(3);
        for _ in 0..5000 {
            let mut o = [0.0; MAX_OBS];
            for (i, slot) in o.iter_mut().enumerate().take(no) {
                *slot = 5.0 + i as f64 + 2.0 * r.normal();
            }
            n.observe(&o, no);
        }
        let mut probe = [0.0; MAX_OBS];
        for (i, slot) in probe.iter_mut().enumerate().take(no) {
            *slot = 5.0 + i as f64;
        }
        let mut out = [0.0; MAX_OBS];
        n.apply(&probe, &mut out, no);
        assert!(out[..no].iter().all(|v| v.abs() < 0.25), "{out:?}");
    }

    /// Converting a world ground point through a sagged `pos[1]` used to fold
    /// every leg: the same foothold is closer in the body frame, IK shortens,
    /// the chassis sits down. Stance has to keep asking for ride height.
    #[test]
    fn a_sagged_chassis_still_commands_ride_height() {
        use crate::terrain::{Course, Terrain};
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let terrain = Terrain::new(Course::Flat, 1);
        let pos = [0.0, 0.40, 0.0];
        let mut shortest = 0.0f64;
        for phase in [0.0, 0.1, 0.2, 0.3] {
            for leg in 0..frame.legs() {
                let t = foot_on_terrain(
                    frame, &gait, leg, phase, gait.stride, gait.duty, gait.cycle, gait.body_h,
                    gait.step_h, 0.0, &terrain, pos, 0.0, 0.0, 0.0,
                );
                shortest = shortest.min(t[1]);
                let lp = crate::math::frac(phase + gait.offsets[leg]);
                if lp < gait.duty {
                    assert!(
                        t[1] < -gait.body_h + 0.08,
                        "stance leg {leg} folded under a sagged body: y={}",
                        t[1]
                    );
                }
            }
        }
        assert!(
            shortest < -0.80,
            "no stance target asked for ride height, shortest={shortest}"
        );
    }

    #[test]
    fn a_swing_into_a_wall_retracts_instead_of_scraping_the_face() {
        use crate::terrain::{Course, Obstacle, Terrain};
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let mut terrain = Terrain::new(Course::Flat, 1);
        let wall = Obstacle {
            x0: 0.85,
            x1: 2.40,
            z0: -0.40,
            z1: 2.20,
            top: 1.80,
            grip: 1.0,
        };
        terrain.push(wall.x0, wall.x1, wall.z0, wall.z1, wall.top, wall.grip);
        terrain.rebuild_buckets();
        assert!(
            terrain.blocked_column(1.20, 0.50, 0.0, 0.62),
            "test wall never made it into the height field (h={})",
            terrain.height(1.20, 0.50)
        );
        let pos = [0.0, gait.body_h, 0.0];
        let mut inside = 0usize;
        let mut retracted = 0usize;
        for k in 0..40 {
            let phase = k as f64 / 40.0;
            for leg in 0..frame.legs() {
                let t = foot_on_terrain(
                    frame, &gait, leg, phase, gait.stride, gait.duty, gait.cycle, gait.body_h,
                    gait.step_h, 0.0, &terrain, pos, 0.0, 0.0, 0.0,
                );
                let w = crate::math::body_to_world(t, 0.0, 0.0, 0.0);
                let p = [pos[0] + w[0], pos[1] + w[1], pos[2] + w[2]];
                let in_xz = p[0] > wall.x0 + 0.04
                    && p[0] < wall.x1 - 0.04
                    && p[2] > wall.z0 + 0.04
                    && p[2] < wall.z1 - 0.04;
                if in_xz && p[1] < wall.top - 0.05 && p[1] > 0.05 {
                    inside += 1;
                }
                // Right-side legs only: left feet sit at x<0 and would pass
                // `p[0] < wall.x0` without ever having seen the wall.
                if leg % 2 == 1 && p[0] < wall.x0 - 0.01 {
                    retracted += 1;
                }
            }
        }
        assert_eq!(inside, 0, "swing still aims into the wall {inside} times");
        assert!(retracted > 0, "never produced a target on the hip side of the wall");
    }
}

/// Body-frame foot target for an open-loop gait. Stance sweeps along the body
/// +Z axis, swing returns with a sine lift. This is the whole walking
/// programme: IK turns it into 18 joint angles, Rapier does the rest.
///
/// `turn` is the commanded yaw rate in rad/s. Over one stance the body turns
/// through `turn * duty * cycle`, so the planted foot has to travel along an arc
/// about the body's yaw axis rather than straight aft — without it the machine
/// has no way to steer at all and wanders off on whatever heading the contacts
/// hand it.
#[allow(clippy::too_many_arguments)]
pub fn foot_in_body(
    frame: Frame,
    gait: &Gait,
    leg: usize,
    phase: f64,
    stride: f64,
    duty: f64,
    cycle: f64,
    body_h: f64,
    step_h: f64,
    turn: f64,
) -> V3 {
    use crate::math::{frac, rot_y};
    let d = frame.dir(leg);
    let out = gait.stance_w * 0.5 + gait.trim(leg);
    let neutral = [d[0] * out, -body_h + crate::robot::FOOT_R - 0.03, d[2] * out];
    let lp = frac(phase + gait.offsets[leg]);
    // `s` runs -0.5 at touchdown to +0.5 at lift-off. A planted foot travels
    // *forward* in the body frame. That is the opposite of the usual "stance
    // sweeps aft" cartoon, and it is what this plant actually pushes against:
    // commanding the foot aft made the chassis walk backward the moment the
    // feet gripped. The swing mirrors it so the stroke stays continuous.
    let (s, lift) = if lp < duty {
        (lp / duty.max(1e-6) - 0.5, 0.0)
    } else {
        let u = (lp - duty) / (1.0 - duty).max(1e-6);
        (
            0.5 - u,
            step_h * (core::f64::consts::PI * u).sin(),
        )
    };
    let arc = rot_y(neutral, turn * duty * cycle * s);
    [arc[0], arc[1] + lift, arc[2] + stride * s]
}

/// The same target, but told what is under the foot.
///
/// The longitudinal sweep is left alone — that aft stance stroke is what pushes
/// the machine along, and anchoring the foot to a world point instead (which is
/// what the centroidal planner does, since there the body advances by fiat)
/// leaves the real plant with nothing to push against. What the terrain gets to
/// change is where the foot ends up:
///
/// * the foot lands on whatever it is over — the top of a block counts as ground
///   as long as it is inside the leg's reach, which is how the machine steps up
///   onto a kerb instead of kicking it;
/// * a swinging foot rides `FOOT_CLEAR` above whatever it passes over, so it
///   crosses an obstacle rather than dragging through it;
/// * a swing that would go through something too tall to stand on is pulled in
///   toward the hip, not glued to the wall face — `push_xz` on a swing is how
///   a tibia spends the whole step scraping a slalom wall;
/// * and the pose the whole leg would have to take is checked, not just the
///   foot: a target that puts the femur or the tibia through a block is pulled
///   in toward the hip until the links are clear, because a foot standing in
///   free air is no use if the knee is inside the crate.
///
/// Stance height is a body-frame command (`-body_h` plus how much higher this
/// foothold is than the ground under the chassis). Converting a world ground
/// point through the *measured* body height folds the legs the moment the deck
/// dips: the same world foot is closer in the body frame, IK shortens, and the
/// chassis sits down. The open-loop standing pose does not do that, which is
/// why a held pose stayed up while a walk collapsed.
///
/// `pos`/`yaw` are the body's *measured* pose, so the leg reacts to the rock
/// that is actually in front of it.
#[allow(clippy::too_many_arguments)]
pub fn foot_on_terrain(
    frame: Frame,
    gait: &Gait,
    leg: usize,
    phase: f64,
    stride: f64,
    duty: f64,
    cycle: f64,
    body_h: f64,
    step_h: f64,
    turn: f64,
    terrain: &Terrain,
    pos: V3,
    yaw: f64,
    _pitch: f64,
    _roll: f64,
) -> V3 {
    use crate::math::{body_to_world, frac, world_to_body};
    use crate::robot::{FOOT_R, LINK_R};
    use crate::sim::{FOOT_CLEAR, MAX_FOOTHOLD};

    // The stroke is laid out in a level frame at yaw: where the foot should go
    // does not depend on how the deck happens to be tipped this instant.
    let base = foot_in_body(
        frame, gait, leg, phase, stride, duty, cycle, body_h, step_h, turn,
    );
    let w = body_to_world(base, yaw, 0.0, 0.0);
    let (mut x, mut z) = (pos[0] + w[0], pos[2] + w[2]);

    // Ground under the chassis, not `pos[1] - body_h`: that second form is the
    // sagged deck talking, and we refuse to listen to it for the ride height.
    let plane = terrain.height(pos[0], pos[2]);
    let lp = frac(phase + gait.offsets[leg]);
    let stance = lp < duty;

    if stance {
        let pushed = terrain.push_xz(x, z, plane, MAX_FOOTHOLD);
        x = pushed.0;
        z = pushed.1;
    } else if swing_blocked(terrain, x, z, plane, MAX_FOOTHOLD, LINK_R) {
        // Pull the swing in toward the hip instead of sliding along a wall.
        let hip = frame.hip(leg);
        let hw = body_to_world(hip, yaw, 0.0, 0.0);
        let (hx, hz) = (pos[0] + hw[0], pos[2] + hw[2]);
        let (x0, z0) = (x, z);
        let mut lo = 0.0;
        let mut hi = 1.0;
        for _ in 0..12 {
            let m = 0.5 * (lo + hi);
            let mx = x0 + (hx - x0) * m;
            let mz = z0 + (hz - z0) * m;
            if swing_blocked(terrain, mx, mz, plane, MAX_FOOTHOLD, LINK_R) {
                lo = m;
            } else {
                hi = m;
            }
        }
        let t = (hi + 0.12).min(1.0);
        x = x0 + (hx - x0) * t;
        z = z0 + (hz - z0) * t;
    }

    let ground = terrain.height(x, z).min(plane + MAX_FOOTHOLD);
    let dy = ground - plane;

    // Keep the gait's body-frame stroke (including its ride height). Only the
    // horizontal point may have been pushed or retracted; `dy` is how much
    // higher this foothold is than the ground under the chassis. Using the
    // measured `pos[1]` for Y is what folded the legs when the deck dipped.
    let horiz = world_to_body([x - pos[0], 0.0, z - pos[2]], yaw, 0.0, 0.0);
    let mut target = [horiz[0], base[1] + dy, horiz[2]];
    if !stance {
        let u = (lp - duty) / (1.0 - duty).max(1e-6);
        let lift = FOOT_CLEAR + step_h * (core::f64::consts::PI * u).sin();
        let y_world = (ground + FOOT_R + lift) - pos[1];
        target[1] = target[1].max(y_world);
    }

    clear_links(frame, leg, target, terrain, pos, yaw)
}

pub(crate) fn swing_blocked(terrain: &Terrain, x: f64, z: f64, floor: f64, max_step: f64, r: f64) -> bool {
    use crate::terrain::CORRIDOR_HALF;
    if x.abs() > CORRIDOR_HALF - r {
        return true;
    }
    terrain.blocked_column(x, z, floor, max_step)
        || terrain.blocked_column(x + r, z, floor, max_step)
        || terrain.blocked_column(x - r, z, floor, max_step)
        || terrain.blocked_column(x, z + r, floor, max_step)
        || terrain.blocked_column(x, z - r, floor, max_step)
}

/// Pull a foot target in toward the hip until no link chords a wall.
///
/// Twelve tries at 20% a step: enough to walk a leg out of a crate it was about
/// to reach through, and it gives up rather than folding the leg under the body
/// when there is no clear pose at all — a stubbed step is better than a
/// collapsed stance.
pub(crate) fn clear_links(
    frame: Frame,
    leg: usize,
    target: V3,
    terrain: &Terrain,
    pos: V3,
    yaw: f64,
) -> V3 {
    let hip = frame.hip(leg);
    let mut t = target;
    for _ in 0..12 {
        if !leg_hits_block(frame, leg, t, terrain, pos, yaw) {
            return t;
        }
        for c in 0..3 {
            t[c] += (hip[c] - t[c]) * 0.20;
        }
    }
    t
}

/// True when any of this leg's links would pass through a block.
///
/// `Terrain::segment_hits_solid` already does the swept test, insets by the same
/// wall padding the rest of the course logic uses, and is what the terrain tests
/// pin down — so the links are handed to it rather than sampled here.
fn leg_hits_block(
    frame: Frame,
    leg: usize,
    target: V3,
    terrain: &Terrain,
    pos: V3,
    yaw: f64,
) -> bool {
    use crate::robot::LINK_R;
    use crate::sim::MAX_FOOTHOLD;
    let q = solve_ik(frame, leg, target).q;
    let j = fk_world(frame, leg, q, pos, yaw, 0.0, 0.0);
    (0..3).any(|k| {
        let (a, b) = (j[k], j[k + 1]);
        if terrain.segment_hits_solid(a, b) {
            return true;
        }
        // Capsules are LINK_R thick. A zero-radius chord can miss a wall the
        // physical tibia still kicks, so offset the chord sideways for the
        // unclimbable test only — thickening against a kerb would retract a
        // legitimate step onto it.
        let dx = b[0] - a[0];
        let dz = b[2] - a[2];
        let len = crate::math::hypot2(dx, dz).max(1e-6);
        let nx = -dz / len * LINK_R;
        let nz = dx / len * LINK_R;
        let shift = |p: V3, s: f64| [p[0] + nx * s, p[1], p[2] + nz * s];
        terrain.segment_hits_wall(shift(a, 1.0), shift(b, 1.0), 0.0, MAX_FOOTHOLD)
            || terrain.segment_hits_wall(shift(a, -1.0), shift(b, -1.0), 0.0, MAX_FOOTHOLD)
    })
}

/// Shortest cycle time this stroke can be run at without asking a joint to turn
/// faster than the servo's no-load speed.
///
/// Real hardware does not get to ignore this: past the no-load speed a servo has
/// no torque left to give, so the joint stops following and the leg waves about a
/// few degrees from where it was told to be. Commanding a gait the actuators
/// cannot execute does not make the machine faster, it makes it imprecise.
///
/// The stroke's shape is fixed in phase, so joint travel per unit phase does not
/// depend on the cycle: sample it once and the shortest feasible cycle is that
/// travel divided by the speed limit. One leg is enough — the others run the same
/// stroke on a phase offset.
#[allow(clippy::too_many_arguments)]
pub fn feasible_cycle(
    frame: Frame,
    gait: &Gait,
    stride: f64,
    duty: f64,
    cycle: f64,
    body_h: f64,
    step_h: f64,
    turn: f64,
    omega_max: f64,
) -> f64 {
    if omega_max <= 1e-6 {
        return cycle;
    }
    const N: usize = 16;
    let mut prev = [0.0f64; 3];
    let mut peak = 0.0f64;
    for k in 0..=N {
        let phase = frac(k as f64 / N as f64 - gait.offsets[0]);
        let foot = foot_in_body(frame, gait, 0, phase, stride, duty, cycle, body_h, step_h, turn);
        let q = solve_ik(frame, 0, foot).q;
        if k > 0 {
            for j in 0..3 {
                // Radians per unit phase, i.e. per cycle.
                peak = peak.max((q[j] - prev[j]).abs() * N as f64);
            }
        }
        prev = q;
    }
    peak / omega_max
}
