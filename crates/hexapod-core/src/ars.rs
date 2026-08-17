//! Augmented Random Search (Mania, Guy & Recht, 2018) — the V2-t variant:
//! finite-difference gradient estimates along random directions, scaled by the
//! spread of the returns, using only the best-performing directions, over a
//! linear policy with normalised observations.
//!
//! It is a derivative-free method, so the simulator never has to be
//! differentiable, and it parallelises trivially across rollouts.

use crate::dynamics::Physics;
use crate::math::Rng;
use crate::policy::{n_theta, Normalizer, Policy};
use crate::sim::{
    evaluate, rollout, Cmd, Rollout, CRUISE_MAX, CRUISE_MIN, JUMP_CRUISE_MAX, JUMP_CRUISE_MIN,
};
use crate::terrain::Terrain;

#[derive(Clone, Copy, Debug)]
pub struct ArsConfig {
    /// Random directions sampled per iteration. Each costs two rollouts.
    pub n_dirs: usize,
    /// How many of the best directions contribute to the update.
    pub n_top: usize,
    /// Step size.
    pub alpha: f64,
    /// Exploration noise.
    pub sigma: f64,
    /// Rollout horizon, seconds of simulated time.
    pub horizon: f64,
}

impl Default for ArsConfig {
    fn default() -> Self {
        ArsConfig {
            // Sixteen directions, not eight. ARS estimates its update from the
            // spread of returns across sampled directions, so the population
            // it needs grows with the parameter count — and the parameter
            // count grew by seventy per cent when the policy gained a forward
            // scan and something to steer toward. Eight directions still
            // converge, they just converge to a worse policy: the walk-to-run
            // duty trend flattens out and speed tracking lands behind the
            // hand-tuned baseline. More iterations do not fix it; more
            // directions do.
            n_dirs: 16,
            n_top: 6,
            alpha: 0.025,
            sigma: 0.04,
            horizon: 8.0,
        }
    }
}

pub struct Trainer {
    pub policy: Policy,
    pub cfg: ArsConfig,
    /// The machine being trained for. Change the servo and the optimum moves.
    pub phys: Physics,
    pub rng: Rng,
    pub iter: usize,
    pub rollouts: usize,
    /// Evaluation reward after each iteration.
    pub curve: Vec<f32>,
    /// Distance covered by the evaluation rollout, per iteration.
    pub dist_curve: Vec<f32>,
    pub best_reward: f64,
    pub best_distance: f64,
    pub best_theta: Vec<f64>,
    /// The normaliser is part of the policy, not a training detail: replaying
    /// `best_theta` against a later normaliser does not reproduce the run.
    pub best_norm: Normalizer,
    pub baseline_reward: f64,
    pub baseline_distance: f64,
    pub last_eval: Rollout,

    deltas: Vec<f64>,
    scratch: Policy,
}

impl Trainer {
    pub fn new(policy: Policy, cfg: ArsConfig, phys: Physics, seed: u64) -> Trainer {
        let policy_frame = policy.frame;
        let best_theta = policy.theta.clone();
        let scratch = policy.clone();
        Trainer {
            policy,
            cfg,
            phys,
            rng: Rng::new(seed),
            iter: 0,
            rollouts: 0,
            curve: Vec::new(),
            dist_curve: Vec::new(),
            best_reward: f64::NEG_INFINITY,
            best_distance: 0.0,
            best_theta,
            best_norm: Normalizer::default(),
            baseline_reward: 0.0,
            baseline_distance: 0.0,
            last_eval: Rollout::default(),
            deltas: vec![0.0; cfg.n_dirs * n_theta(policy_frame)],
            scratch,
        }
    }

    /// Score the current parameters over the full spread of commanded speeds,
    /// with the normaliser frozen.
    ///
    /// Averaging over several speeds is what keeps the score honest. A policy
    /// that pins its parameters to whatever runs fastest scores badly here,
    /// because it cannot also be asked to walk at 2 m/s.
    pub fn evaluate(&mut self, terrain: &Terrain) -> Rollout {
        let mut p = self.policy.clone();
        p.norm.frozen = true;
        let r = evaluate(terrain, &p, &self.phys, self.cfg.horizon);
        self.last_eval = r;
        r
    }

    pub fn record_baseline(&mut self, terrain: &Terrain) {
        let r = self.evaluate(terrain);
        self.baseline_reward = r.reward;
        self.baseline_distance = r.distance;
        self.best_reward = r.reward;
        self.best_distance = self.baseline_distance;
        self.best_theta.copy_from_slice(&self.policy.theta);
        self.best_norm = self.policy.norm.clone();
        self.curve.push(r.reward as f32);
        self.dist_curve.push(r.distance as f32);
    }

    /// One ARS iteration: `2 * n_dirs` exploratory rollouts plus one
    /// evaluation rollout.
    pub fn iterate(&mut self, terrain: &Terrain) -> f64 {
        let n = n_theta(self.policy.frame);
        let cfg = self.cfg;

        for k in 0..cfg.n_dirs {
            let base = k * n;
            for j in 0..n {
                self.deltas[base + j] = self.rng.normal();
            }
        }

        // Observation statistics are pooled across every rollout in the
        // iteration, then folded back into the policy afterwards.
        let mut norm_accum: Normalizer = self.policy.norm.clone();
        norm_accum.frozen = false;

        let mut rewards = vec![(0.0f64, 0.0f64); cfg.n_dirs];

        for k in 0..cfg.n_dirs {
            let base = k * n;

            // One command per direction, shared by both sides of the finite
            // difference — otherwise the difference measures the command draw
            // rather than the perturbation. JUMP samples a faster band: the
            // trenches are a running jump, not a walk.
            let cmd = if terrain.course.is_jump() {
                Cmd::at(JUMP_CRUISE_MIN + self.rng.unit() * (JUMP_CRUISE_MAX - JUMP_CRUISE_MIN))
            } else {
                Cmd::at(CRUISE_MIN + self.rng.unit() * (CRUISE_MAX - CRUISE_MIN))
            };

            for sign in [1.0f64, -1.0f64] {
                for j in 0..n {
                    self.scratch.theta[j] =
                        self.policy.theta[j] + sign * cfg.sigma * self.deltas[base + j];
                }
                self.scratch.norm = self.policy.norm.clone();
                self.scratch.frame = self.policy.frame;
                self.scratch.base_offsets = self.policy.base_offsets;
                self.scratch.feedback = self.policy.feedback;

                let r = rollout(
                    terrain,
                    &self.scratch,
                    &self.phys,
                    cfg.horizon,
                    cmd,
                    Some(&mut norm_accum),
                );
                self.rollouts += 1;
                if sign > 0.0 {
                    rewards[k].0 = r.reward;
                } else {
                    rewards[k].1 = r.reward;
                }
            }
        }

        // Rank directions by their best side, keep the top b.
        let mut order: Vec<usize> = (0..cfg.n_dirs).collect();
        order.sort_by(|&a, &b| {
            let ka = rewards[a].0.max(rewards[a].1);
            let kb = rewards[b].0.max(rewards[b].1);
            kb.partial_cmp(&ka).unwrap_or(core::cmp::Ordering::Equal)
        });
        let b = cfg.n_top.min(cfg.n_dirs);
        let top = &order[..b];

        // Scale the step by the spread of the returns that produced it. This
        // is what makes ARS insensitive to reward magnitude.
        let mut mean = 0.0;
        for &k in top {
            mean += rewards[k].0 + rewards[k].1;
        }
        mean /= (2 * b) as f64;
        let mut var = 0.0;
        for &k in top {
            var += (rewards[k].0 - mean).powi(2) + (rewards[k].1 - mean).powi(2);
        }
        let sigma_r = (var / (2 * b) as f64).sqrt();

        if sigma_r > 1e-8 {
            let scale = cfg.alpha / (b as f64 * sigma_r);
            for &k in top {
                let diff = rewards[k].0 - rewards[k].1;
                if diff == 0.0 {
                    continue;
                }
                let base = k * n;
                for j in 0..n {
                    self.policy.theta[j] += scale * diff * self.deltas[base + j];
                }
            }
        }

        self.policy.norm = norm_accum;
        self.iter += 1;

        let ev = self.evaluate(terrain);
        self.curve.push(ev.reward as f32);
        self.dist_curve.push(ev.distance as f32);
        if ev.reward > self.best_reward {
            self.best_reward = ev.reward;
            self.best_distance = ev.distance;
            self.best_theta.copy_from_slice(&self.policy.theta);
            self.best_norm = self.policy.norm.clone();
        }
        ev.reward
    }

    /// The best parameters found so far, ready to run.
    pub fn best_policy(&self) -> Policy {
        let mut p = self.policy.clone();
        p.theta.copy_from_slice(&self.best_theta);
        p.norm = self.best_norm.clone();
        p.norm.frozen = true;
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{n_gait, Preset};
    use crate::terrain::Course;

    fn trainer(tseed: u64) -> Trainer {
        Trainer::new(
            Policy::seeded(Preset::Tripod, crate::robot::Frame::default()),
            ArsConfig::default(),
            Physics::default(),
            tseed,
        )
    }

    #[test]
    fn training_improves_on_the_baseline() {
        let terrain = Terrain::new(Course::Mixed, 5);
        let mut t = trainer(0xC0FFEE);
        t.record_baseline(&terrain);
        for _ in 0..60 {
            t.iterate(&terrain);
        }
        assert!(
            t.best_reward > t.baseline_reward,
            "no improvement: baseline {:.2}, best {:.2}",
            t.baseline_reward,
            t.best_reward
        );
    }

    #[test]
    fn best_theta_actually_reproduces_best_reward() {
        let terrain = Terrain::new(Course::Rubble, 3);
        let mut t = trainer(7);
        t.record_baseline(&terrain);
        for _ in 0..12 {
            t.iterate(&terrain);
        }
        let replay = evaluate(&terrain, &t.best_policy(), &t.phys, t.cfg.horizon);
        assert!(
            (replay.reward - t.best_reward).abs() < 1e-9,
            "replay {:.4} != recorded {:.4}",
            replay.reward,
            t.best_reward
        );
    }

    #[test]
    fn training_is_reproducible_from_its_seed() {
        let terrain = Terrain::new(Course::Steps, 2);
        let run = || {
            let mut t = trainer(99);
            t.record_baseline(&terrain);
            for _ in 0..8 {
                t.iterate(&terrain);
            }
            t.curve.clone()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn the_learner_moves_both_the_gait_and_the_feedback_layer() {
        let terrain = Terrain::new(Course::Mixed, 1);
        let mut t = trainer(21);
        let before = t.policy.theta.clone();
        t.record_baseline(&terrain);
        for _ in 0..20 {
            t.iterate(&terrain);
        }
        let n = n_gait(t.policy.frame);
        let gait_moved = (0..n).any(|i| (t.policy.theta[i] - before[i]).abs() > 1e-6);
        assert!(gait_moved, "gait parameters never moved");
        assert!(
            t.policy.feedback_norm() > 1e-6,
            "feedback layer never left zero"
        );
    }

    #[test]
    fn exploration_samples_the_whole_commanded_speed_range() {
        // Both sides of a direction must share a command, and the commands
        // across an iteration must span the range rather than sit on one
        // speed — that is what stops the optimiser specialising.
        let mut r = Rng::new(4);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for _ in 0..200 {
            let v = CRUISE_MIN + r.unit() * (CRUISE_MAX - CRUISE_MIN);
            assert!((CRUISE_MIN..=CRUISE_MAX).contains(&v));
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(hi - lo > (CRUISE_MAX - CRUISE_MIN) * 0.8);
    }

    #[test]
    fn jump_training_improves_on_falling_in_the_trench() {
        let terrain = Terrain::new(Course::Jump, 1);
        let mut t = trainer(0xBEEF);
        t.cfg.horizon = 6.0;
        t.record_baseline(&terrain);
        for _ in 0..40 {
            t.iterate(&terrain);
        }
        assert!(
            t.best_reward > t.baseline_reward
                || t.best_distance > t.baseline_distance + 0.5,
            "no improvement: baseline reward {:.2} dist {:.2}, best reward {:.2} dist {:.2}",
            t.baseline_reward,
            t.baseline_distance,
            t.best_reward,
            t.best_distance
        );
    }
}
