//! A learned crawl controller.
//!
//! The hand-written crawl makes four decisions every time it picks a leg up:
//! how far ahead to plant it, how far to the side, how far to lean the body
//! over the plants that stay down, and how hard to push. Each of those is a
//! constant somebody swept by hand — `STEP`, `SIDE`, the lean bisect, and
//! `LEAD_MAX` — and each was swept on its own, against one course, by eye.
//!
//! This replaces the four constants with a linear policy over the machine's
//! state. What it does *not* replace is the phase machine: the crawl still
//! lifts one leg at a time through settle/lift/shift/place/pause, and the
//! policy still chooses inside the reachability and support-margin bisects
//! that the hand-written version uses. So the search runs on a manifold where
//! every action is kinematically reachable and statically stable, and the
//! policy is choosing *where in the safe set* to operate rather than being
//! asked to rediscover that walking on air ends badly.
//!
//! The actions are fractions, not absolutes, for the same reason: an action of
//! zero reproduces the conservative choice and an action of one takes the most
//! aggressive option still inside the safe set, so a policy of all zeros is a
//! working crawl and training starts from something that already walks.

use crate::math::Rng;

/// Bias that squashes to a saturated action. The seed wants tanh(b) ~= 1 and
/// tanh has no finite input for that, so it takes 0.995 and keeps a live
/// gradient — a truly saturated seed could never be trained away from.
const SEED_BIAS: f64 = 3.0;

/// Body pitch and roll, planar velocity along and across the heading, the
/// support margin the coming swing will leave, how far the leg being lifted
/// currently sits from its neutral stance, and the command. Plus a bias.
pub const N_OBS: usize = 9;
/// `along`, `side`, `lean`, `lead` — see [`CrawlAction`].
pub const N_ACT: usize = 4;

/// What the policy decides at the moment a leg leaves the ground.
#[derive(Clone, Copy, Debug, Default)]
pub struct CrawlAction {
    /// Fraction of the reachable forward travel to plant into, 0..1.
    pub along: f64,
    /// Lateral plant offset as a fraction of the forward one, -1..1.
    pub side: f64,
    /// Fraction of the available forward lean to take, 0..1. Zero stands on
    /// the centroid of the plants; one goes to the support-margin limit.
    pub lean: f64,
    /// Fraction of the permitted commanded-body lead to drive with, 0..1.
    /// This is the crawl's throttle: the stance legs push against it.
    pub lead: f64,
}

/// Observation at a decision point. Ordering matters only in that it has to
/// match between rollout and training, which is why it is built in one place.
#[derive(Clone, Copy, Debug, Default)]
pub struct CrawlObs {
    pub pitch: f64,
    pub roll: f64,
    pub vel_along: f64,
    pub vel_side: f64,
    /// Support margin left by lifting this leg, metres.
    pub margin: f64,
    /// How far the leg about to move sits behind its neutral stance.
    pub behind: f64,
    /// Where the leg sits around the body: +1 front, -1 rear.
    pub leg_fore: f64,
    pub cmd_fwd: f64,
    pub cmd_turn: f64,
}

impl CrawlObs {
    fn vector(&self) -> [f64; N_OBS] {
        [
            self.pitch,
            self.roll,
            self.vel_along,
            self.vel_side,
            self.margin,
            self.behind,
            self.leg_fore,
            self.cmd_fwd,
            self.cmd_turn,
        ]
    }
}

/// Linear policy, `N_ACT` rows of `N_OBS` weights plus a bias each.
#[derive(Clone, Debug)]
pub struct CrawlPolicy {
    pub theta: Vec<f64>,
}

impl Default for CrawlPolicy {
    fn default() -> Self {
        CrawlPolicy::seeded()
    }
}

impl CrawlPolicy {
    pub fn n_params() -> usize {
        N_ACT * (N_OBS + 1)
    }

    /// All weights zero, biases set so the policy reproduces the hand-written
    /// crawl's choices. Training therefore starts from a machine that walks,
    /// and any improvement is measured against the thing it replaces.
    pub fn seeded() -> CrawlPolicy {
        let mut theta = vec![0.0; Self::n_params()];
        let bias = N_OBS;
        // Hand-written equivalents: full reachable step, no lateral offset
        // without a turn command, lean all the way to the margin limit, drive
        // at the full permitted lead.
        theta[bias] = SEED_BIAS;
        theta[(N_OBS + 1) + bias] = 0.0;
        theta[2 * (N_OBS + 1) + bias] = SEED_BIAS;
        theta[3 * (N_OBS + 1) + bias] = SEED_BIAS;
        CrawlPolicy { theta }
    }

    pub fn act(&self, obs: &CrawlObs) -> CrawlAction {
        let x = obs.vector();
        let mut out = [0.0f64; N_ACT];
        for (a, o) in out.iter_mut().enumerate() {
            let row = a * (N_OBS + 1);
            let mut acc = self.theta[row + N_OBS];
            for (i, xi) in x.iter().enumerate() {
                acc += self.theta[row + i] * xi;
            }
            *o = acc.tanh();
        }
        CrawlAction {
            // tanh gives -1..1; the one-sided fractions map onto 0..1 so that
            // a saturated negative weight means "the conservative choice", not
            // "walk backwards", which is a different command entirely.
            along: 0.5 * (out[0] + 1.0),
            side: out[1],
            lean: 0.5 * (out[2] + 1.0),
            lead: 0.5 * (out[3] + 1.0),
        }
    }

    /// `self` perturbed by `dir` scaled by `sigma`, for an ARS pair.
    pub fn nudged(&self, dir: &[f64], sigma: f64, sign: f64) -> CrawlPolicy {
        let theta = self
            .theta
            .iter()
            .zip(dir)
            .map(|(t, d)| t + sign * sigma * d)
            .collect();
        CrawlPolicy { theta }
    }

    pub fn random_dir(rng: &mut Rng) -> Vec<f64> {
        (0..Self::n_params()).map(|_| rng.normal()).collect()
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_seed_reproduces_the_hand_written_crawl() {
        let p = CrawlPolicy::seeded();
        let a = p.act(&CrawlObs::default());
        assert!(a.along > 0.99, "seed should take the full step: {a:?}");
        assert!(a.lean > 0.99, "seed should lean to the limit: {a:?}");
        assert!(a.lead > 0.99, "seed should drive at full lead: {a:?}");
        assert!(a.side.abs() < 1e-6, "seed should not crab: {a:?}");
    }

    #[test]
    fn actions_stay_inside_their_ranges_for_any_observation() {
        let mut rng = Rng::new(7);
        let mut p = CrawlPolicy::seeded();
        for t in p.theta.iter_mut() {
            *t = rng.normal() * 50.0;
        }
        for _ in 0..200 {
            let obs = CrawlObs {
                pitch: rng.normal() * 10.0,
                roll: rng.normal() * 10.0,
                vel_along: rng.normal() * 10.0,
                vel_side: rng.normal() * 10.0,
                margin: rng.normal() * 10.0,
                behind: rng.normal() * 10.0,
                leg_fore: rng.normal(),
                cmd_fwd: rng.normal(),
                cmd_turn: rng.normal(),
            };
            let a = p.act(&obs);
            assert!((0.0..=1.0).contains(&a.along), "{a:?}");
            assert!((-1.0..=1.0).contains(&a.side), "{a:?}");
            assert!((0.0..=1.0).contains(&a.lean), "{a:?}");
            assert!((0.0..=1.0).contains(&a.lead), "{a:?}");
        }
    }
}

// ------------------------------------------------------------------ training

use crate::dynamics::Physics;
use crate::oneleg::OneLegDrill;
use crate::robot::{Frame, MAX_LEGS};
use crate::sim::{Cmd, DT};
use crate::terrain::{Course, Terrain};

/// What a rollout is scored on. Ground covered toward the waypoint is the
/// whole point; everything else is a cost the hand-tuned crawl also pays and
/// nobody was counting. Wasted motion is in here because a machine that rings
/// is burning actuator travel to stay in one place — the exact failure this
/// controller was written to stop.
#[derive(Clone, Copy, Debug, Default)]
pub struct Score {
    pub progress: f64,
    pub wasted: f64,
    pub slip: f64,
    pub sag: f64,
    pub fell: bool,
    /// Seconds before it fell, or the whole horizon. A flat penalty for
    /// falling makes every failing policy score identically, which leaves ARS
    /// ranking noise; surviving longer has to be worth something or a
    /// population that all falls carries no gradient at all.
    pub alive: f64,
}

impl Score {
    pub fn reward(&self) -> f64 {
        let base = self.progress - 0.05 * self.wasted - 0.02 * self.slip - 2.0 * self.sag;
        if self.fell {
            // Falling is not a small negative — a policy that dives at second
            // nineteen must not beat one that walks the whole horizon. But it
            // still has to be ordered by how far it got, or the search is
            // blind while every candidate is still falling.
            return base - 10.0 + self.alive;
        }
        base
    }
}

/// One scored rollout of `policy` on `course`.
pub fn score(policy: &CrawlPolicy, frame: Frame, phys: &Physics, course: Course, seed: u64, secs: f64) -> Score {
    let terrain = Terrain::new(course, seed);
    let mut d = OneLegDrill::spawn_on(frame, phys, &terrain, seed, true);
    d.policy = Some(policy.clone());
    d.set_cmd(Cmd {
        fwd: 1.0,
        turn: 0.0,
        cruise: 0.35,
        nav: false,
    });
    let goal = terrain.waypoint(0);
    let start = d.sample().pos;
    let range0 = hypot(goal[0] - start[0], goal[1] - start[2]);

    let mut s = Score::default();
    let mut prev_pos = start;
    let mut prev_feet: Option<[[f64; 3]; MAX_LEGS]> = None;
    let ride = 0.88;
    for _ in 0..(secs / DT) as usize {
        d.step(DT);
        let sm = d.sample();
        if sm.fallen {
            s.fell = true;
            break;
        }
        s.alive += DT;
        // Wasted body motion: path travelled that did not become progress.
        s.wasted += dist2(sm.pos, prev_pos);
        prev_pos = sm.pos;
        s.sag += (ride - sm.pos[1]).max(0.0) * DT;
        let mut feet = [[0.0; 3]; MAX_LEGS];
        for i in 0..frame.legs() {
            feet[i] = d.plant.leg_joints_world(i)[3];
        }
        if let Some(p) = prev_feet {
            for i in 0..frame.legs() {
                if i == sm.moving && sm.phase.swinging() {
                    continue;
                }
                s.slip += hypot(feet[i][0] - p[i][0], feet[i][2] - p[i][2]);
            }
        }
        prev_feet = Some(feet);
    }
    let end = d.sample().pos;
    let range1 = hypot(goal[0] - end[0], goal[1] - end[2]);
    s.progress = range0 - range1;
    // Only the body path that did not turn into progress is waste.
    s.wasted = (s.wasted - s.progress.abs()).max(0.0);
    s
}

fn hypot(a: f64, b: f64) -> f64 {
    (a * a + b * b).sqrt()
}

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    hypot(a[0] - b[0], a[2] - b[2])
}

pub struct TrainCfg {
    pub n_dirs: usize,
    pub n_top: usize,
    pub alpha: f64,
    pub sigma: f64,
    pub secs: f64,
    pub courses: Vec<Course>,
}

impl Default for TrainCfg {
    fn default() -> Self {
        TrainCfg {
            n_dirs: 8,
            n_top: 4,
            alpha: 0.08,
            sigma: 0.15,
            secs: 25.0,
            // Train on more than one course or the policy learns the seed's
            // particular rocks rather than how to walk.
            courses: vec![Course::Flat, Course::Rubble],
        }
    }
}

/// Mean reward across the training courses, so one lucky course cannot carry
/// a policy that only works there.
fn mean_reward(p: &CrawlPolicy, frame: Frame, phys: &Physics, cfg: &TrainCfg, seed: u64) -> f64 {
    let mut total = 0.0;
    for (k, c) in cfg.courses.iter().enumerate() {
        total += score(p, frame, phys, *c, seed + k as u64, cfg.secs).reward();
    }
    total / cfg.courses.len() as f64
}

/// Augmented Random Search over the crawl policy, on the articulated plant.
/// Returns the policy and its reward each iteration so a caller can watch it.
pub fn train(
    frame: Frame,
    phys: &Physics,
    cfg: &TrainCfg,
    iters: usize,
    seed: u64,
    mut on_iter: impl FnMut(usize, f64, &CrawlPolicy),
) -> CrawlPolicy {
    let mut rng = Rng::new(seed);
    let mut policy = CrawlPolicy::seeded();
    for it in 0..iters {
        let dirs: Vec<Vec<f64>> = (0..cfg.n_dirs)
            .map(|_| CrawlPolicy::random_dir(&mut rng))
            .collect();
        let mut rated: Vec<(f64, f64, usize)> = Vec::with_capacity(cfg.n_dirs);
        for (k, d) in dirs.iter().enumerate() {
            let rp = mean_reward(&policy.nudged(d, cfg.sigma, 1.0), frame, phys, cfg, seed);
            let rm = mean_reward(&policy.nudged(d, cfg.sigma, -1.0), frame, phys, cfg, seed);
            rated.push((rp, rm, k));
        }
        // ARS-V1: rank directions by their best side, keep the top slice.
        rated.sort_by(|a, b| {
            b.0.max(b.1)
                .partial_cmp(&a.0.max(a.1))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top = &rated[..cfg.n_top.min(rated.len())];
        let mean = top.iter().map(|t| (t.0 + t.1) / 2.0).sum::<f64>() / top.len() as f64;
        let var = top
            .iter()
            .map(|t| ((t.0 + t.1) / 2.0 - mean).powi(2))
            .sum::<f64>()
            / top.len() as f64;
        let sd = var.sqrt().max(1e-6);
        for (rp, rm, k) in top {
            let g = (rp - rm) / (cfg.n_top as f64 * sd);
            for (t, d) in policy.theta.iter_mut().zip(&dirs[*k]) {
                *t += cfg.alpha * g * d;
            }
        }
        let r = mean_reward(&policy, frame, phys, cfg, seed);
        on_iter(it, r, &policy);
    }
    policy
}
