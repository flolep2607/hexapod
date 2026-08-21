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
use crate::sim::{cruise_band, evaluate, rollout, Cmd, Rollout};
use crate::terrain::Terrain;

/// Suite training is a navigation task first and a gait-quality task second.
/// These scales make reaching one more ordered waypoint dominate plausible
/// reward noise, and completing the route dominate any partial trajectory.
const ROUTE_OBJECTIVE_SCALE: f64 = 5_000.0;
const COMPLETION_OBJECTIVE_SCALE: f64 = 10_000.0;

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
    /// Native rollout worker count. Zero uses available parallelism; WASM is
    /// always sequential because browser threads require a separate runtime.
    pub workers: usize,
    /// Terrain/seed scenarios averaged for each perturbation sign. A suite
    /// uses mini-batches to keep direction ranking from comparing an easy
    /// course's absolute return with a hard course's; single-course training
    /// needs only one.
    pub scenarios_per_direction: usize,
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
            workers: 0,
            scenarios_per_direction: 1,
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
    pub best_completion_rate: f64,
    pub best_waypoint_fraction: f64,
    pub best_theta: Vec<f64>,
    /// The normaliser is part of the policy, not a training detail: replaying
    /// `best_theta` against a later normaliser does not reproduce the run.
    pub best_norm: Normalizer,
    pub baseline_reward: f64,
    pub baseline_distance: f64,
    pub baseline_completion_rate: f64,
    pub last_eval: Rollout,

    deltas: Vec<f64>,
}

impl Trainer {
    pub fn new(policy: Policy, cfg: ArsConfig, phys: Physics, seed: u64) -> Trainer {
        let policy_frame = policy.frame;
        let best_theta = policy.theta.clone();
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
            best_completion_rate: 0.0,
            best_waypoint_fraction: 0.0,
            best_theta,
            best_norm: Normalizer::default(),
            baseline_reward: 0.0,
            baseline_distance: 0.0,
            baseline_completion_rate: 0.0,
            last_eval: Rollout::default(),
            deltas: vec![0.0; cfg.n_dirs * n_theta(policy_frame)],
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
        self.record_baseline_result(r);
    }

    /// Record a baseline against a balanced terrain suite. The best policy is
    /// subsequently selected against this same aggregate task, not whichever
    /// individual terrain happened to be sampled last.
    pub fn record_suite_baseline(&mut self, terrains: &[Terrain]) {
        let r = self.evaluate_suite(terrains);
        self.record_baseline_result(r);
    }

    fn record_baseline_result(&mut self, r: Rollout) {
        self.baseline_reward = r.reward;
        self.baseline_distance = r.distance;
        self.baseline_completion_rate = r.completion_rate;
        self.best_reward = r.reward;
        self.best_distance = self.baseline_distance;
        self.best_completion_rate = r.completion_rate;
        self.best_waypoint_fraction = r.waypoint_fraction;
        self.best_theta.copy_from_slice(&self.policy.theta);
        self.best_norm = self.policy.norm.clone();
        self.curve.push(r.reward as f32);
        self.dist_curve.push(r.distance as f32);
    }

    /// Evaluate equally across all supplied terrain/seed scenarios and the
    /// normal per-course speed schedule.
    pub fn evaluate_suite(&mut self, terrains: &[Terrain]) -> Rollout {
        assert!(!terrains.is_empty(), "training suite must contain terrain");
        let mut p = self.policy.clone();
        p.norm.frozen = true;
        let mut acc = Rollout::default();
        acc.finished = true;
        acc.completed = true;
        let n = terrains.len() as f64;
        for terrain in terrains {
            let r = evaluate(terrain, &p, &self.phys, self.cfg.horizon);
            merge_rollout(&mut acc, r, 1.0 / n);
        }
        self.last_eval = acc;
        acc
    }

    /// One ARS iteration: `2 * n_dirs` exploratory rollouts plus one
    /// evaluation rollout.
    pub fn iterate(&mut self, terrain: &Terrain) -> f64 {
        self.iterate_suite(core::slice::from_ref(terrain))
    }

    /// One stochastic ARS iteration over a balanced terrain suite.
    ///
    /// Direction `k` is assigned a deterministic rotating mini-batch. Both
    /// perturbation signs see exactly the same terrains and speeds, so their
    /// difference remains a valid finite difference. When the iteration's
    /// direction-batches contain at least as many slots as the suite, every
    /// scenario contributes to every update; smaller searches cover the whole
    /// suite over successive iterations.
    pub fn iterate_suite(&mut self, terrains: &[Terrain]) -> f64 {
        self.iterate_suite_with_eval(terrains, terrains)
    }

    /// Train from a weighted curriculum but select checkpoints on a separate,
    /// balanced validation suite. Duplicating a hard scenario should change
    /// the gradient budget, not make a specialist checkpoint look globally
    /// better merely because that course was counted many more times.
    pub fn iterate_suite_with_eval(
        &mut self,
        terrains: &[Terrain],
        eval_terrains: &[Terrain],
    ) -> f64 {
        assert!(!terrains.is_empty(), "training suite must contain terrain");
        assert!(
            !eval_terrains.is_empty(),
            "evaluation suite must contain terrain"
        );
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

        // One command per direction, shared by both sides of the finite
        // difference — otherwise the difference measures the command draw
        // rather than the perturbation. JUMP samples a faster band: the
        // trenches are a running jump, not a walk.
        let batch = cfg.scenarios_per_direction.max(1);
        let assignments: Vec<Vec<(usize, Cmd)>> = (0..cfg.n_dirs)
            .map(|k| {
                (0..batch)
                    .map(|sample| {
                        let slot = k * batch + sample;
                        let terrain_i = suite_terrain_index(
                            self.iter,
                            slot,
                            cfg.n_dirs * batch,
                            terrains.len(),
                        );
                        let terrain = &terrains[terrain_i];
                        let (lo, hi) = cruise_band(terrain.course);
                        let cmd = Cmd::at(lo + self.rng.unit() * (hi - lo));
                        (terrain_i, cmd)
                    })
                    .collect()
            })
            .collect();

        #[cfg(not(target_family = "wasm"))]
        let mut results = {
            let workers = if cfg.workers == 0 {
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1)
            } else {
                cfg.workers
            }
            .max(1)
            .min(cfg.n_dirs.max(1));
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(workers);
                for worker in 0..workers {
                    let assignments = &assignments;
                    let deltas = &self.deltas;
                    let policy = &self.policy;
                    let phys = &self.phys;
                    handles.push(scope.spawn(move || {
                        let mut out = Vec::new();
                        for k in (worker..cfg.n_dirs).step_by(workers) {
                            let base = k * n;
                            out.push(run_direction(
                                k,
                                policy,
                                &deltas[base..base + n],
                                terrains,
                                &assignments[k],
                                phys,
                                cfg,
                            ));
                        }
                        out
                    }));
                }
                handles
                    .into_iter()
                    .flat_map(|h| h.join().expect("ARS rollout worker panicked"))
                    .collect::<Vec<_>>()
            })
        };

        #[cfg(target_family = "wasm")]
        let mut results = assignments
            .iter()
            .enumerate()
            .map(|(k, scenarios)| {
                let base = k * n;
                run_direction(
                    k,
                    &self.policy,
                    &self.deltas[base..base + n],
                    terrains,
                    scenarios,
                    &self.phys,
                    cfg,
                )
            })
            .collect::<Vec<_>>();

        // Worker completion order must not affect the policy. Merge both
        // returns and observation statistics in direction/sign order.
        results.sort_by_key(|r| r.k);
        let mut rewards = vec![(0.0f64, 0.0f64); cfg.n_dirs];
        for r in results {
            rewards[r.k] = (r.plus_reward, r.minus_reward);
            norm_accum.merge(&r.plus_norm, self.policy.n_obs());
            norm_accum.merge(&r.minus_norm, self.policy.n_obs());
        }
        self.rollouts += 2 * cfg.n_dirs * batch;

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

        let ev = if eval_terrains.len() == 1 {
            self.evaluate(&eval_terrains[0])
        } else {
            self.evaluate_suite(eval_terrains)
        };
        self.curve.push(ev.reward as f32);
        self.dist_curve.push(ev.distance as f32);
        let better = if eval_terrains.len() == 1 {
            ev.reward > self.best_reward
        } else {
            training_objective(&ev)
                > COMPLETION_OBJECTIVE_SCALE * self.best_completion_rate
                    + ROUTE_OBJECTIVE_SCALE * self.best_waypoint_fraction
                    + self.best_reward
        };
        if better {
            self.best_reward = ev.reward;
            self.best_distance = ev.distance;
            self.best_completion_rate = ev.completion_rate;
            self.best_waypoint_fraction = ev.waypoint_fraction;
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

struct DirectionResult {
    k: usize,
    plus_reward: f64,
    minus_reward: f64,
    plus_norm: Normalizer,
    minus_norm: Normalizer,
}

fn run_direction(
    k: usize,
    policy: &Policy,
    delta: &[f64],
    terrains: &[Terrain],
    scenarios: &[(usize, Cmd)],
    phys: &Physics,
    cfg: ArsConfig,
) -> DirectionResult {
    let completion_objective = terrains.len() > 1;
    let run = |sign: f64| {
        let mut candidate = policy.clone();
        for ((theta, base), d) in candidate.theta.iter_mut().zip(&policy.theta).zip(delta) {
            *theta = *base + sign * cfg.sigma * d;
        }
        let mut norm = Normalizer::empty();
        let reward = scenarios
            .iter()
            .map(|&(terrain_i, cmd)| {
                let result = rollout(
                    &terrains[terrain_i],
                    &candidate,
                    phys,
                    cfg.horizon,
                    cmd,
                    Some(&mut norm),
                );
                if completion_objective {
                    training_objective(&result)
                } else {
                    result.reward
                }
            })
            .sum::<f64>()
            / scenarios.len() as f64;
        (reward, norm)
    };
    let (plus_reward, plus_norm) = run(1.0);
    let (minus_reward, minus_norm) = run(-1.0);
    DirectionResult {
        k,
        plus_reward,
        minus_reward,
        plus_norm,
        minus_norm,
    }
}

#[inline]
fn training_objective(result: &Rollout) -> f64 {
    COMPLETION_OBJECTIVE_SCALE * result.completion_rate
        + ROUTE_OBJECTIVE_SCALE * result.waypoint_fraction
        + result.reward
}

#[inline]
fn suite_terrain_index(iter: usize, direction: usize, n_dirs: usize, n_terrains: usize) -> usize {
    let mut stride = n_terrains / 2 + 1;
    while gcd(stride, n_terrains) != 1 {
        stride += 1;
    }
    ((iter * n_dirs + direction) * stride) % n_terrains
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Add a rollout summary into an aggregate. Counts which are meaningful as
/// totals (`steps`, `reached`, jumps) remain totals; physical and task metrics
/// are weighted means. Completion booleans mean every episode succeeded.
fn merge_rollout(acc: &mut Rollout, r: Rollout, weight: f64) {
    acc.reward += r.reward * weight;
    acc.distance += r.distance * weight;
    acc.elapsed += r.elapsed * weight;
    acc.end_x += r.end_x * weight;
    acc.steps += r.steps;
    acc.fell |= r.fell;
    acc.stub_total += r.stub_total * weight;
    acc.work += r.work * weight;
    acc.slip += r.slip * weight;
    acc.cot += r.cot * weight;
    acc.speed_error += r.speed_error * weight;
    acc.peak_servo_load = acc.peak_servo_load.max(r.peak_servo_load);
    acc.reached += r.reached;
    acc.finished &= r.finished;
    acc.completed &= r.completed;
    acc.waypoint_fraction += r.waypoint_fraction * weight;
    acc.completion_rate += r.completion_rate * weight;
    acc.finish_time += r.finish_time * weight;
    acc.collisions += r.collisions * weight;
    acc.mean_cycle += r.mean_cycle * weight;
    acc.mean_stride += r.mean_stride * weight;
    acc.mean_duty += r.mean_duty * weight;
    acc.apex = acc.apex.max(r.apex);
    acc.jumps += r.jumps;
    acc.broken |= r.broken;
    acc.impact_g = acc.impact_g.max(r.impact_g);
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
        // 120, not 60: on the STS3250's torque-speed line the hand-tuned tripod
        // already scores 85.99 here, and random search needs about a hundred
        // iterations to beat a baseline that good. Measured on this seed it
        // reaches 86.53 by 100, 88.21 by 120 and 90.22 by 200.
        for _ in 0..120 {
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
    fn mixed_training_beats_the_seed_with_the_dashboard_config() {
        // The in-page trainer uses MIXED seed 1, horizon 12, seed ^ 0xA5A5.
        // A hop trigger that fires on ARS noise pins best_theta at the seed
        // and the dashboard reports +0% forever.
        let terrain = Terrain::new(Course::Mixed, 1);
        let mut t = Trainer::new(
            Policy::seeded(Preset::Tripod, crate::robot::Frame::default()),
            ArsConfig {
                horizon: 12.0,
                ..ArsConfig::default()
            },
            Physics::default(),
            1u64 ^ 0xA5A5,
        );
        t.record_baseline(&terrain);
        for _ in 0..40 {
            t.iterate(&terrain);
        }
        assert!(
            t.best_reward > t.baseline_reward,
            "dashboard-config MIXED never beat the seed: {:.2} -> {:.2}",
            t.baseline_reward,
            t.best_reward
        );
        assert!(
            t.best_policy().feedback_norm() > 1e-6,
            "best policy still open-loop"
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
        // across an iteration must span the course's band rather than sit on
        // one speed — that is what stops the optimiser specialising.
        for course in [Course::Mixed, Course::Jump, Course::Chasm] {
            let (min, max) = cruise_band(course);
            let mut r = Rng::new(4);
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for _ in 0..200 {
                let v = min + r.unit() * (max - min);
                assert!((min..=max).contains(&v), "{course:?} sampled {v}");
                lo = lo.min(v);
                hi = hi.max(v);
            }
            assert!(hi - lo > (max - min) * 0.8, "{course:?} barely explored");
        }
        // And the bands are ordered by what the course physically demands: a
        // stepable course can be walked, a JUMP trench needs a run, and a
        // CHASM trench needs a faster one.
        assert!(cruise_band(Course::Jump).0 > cruise_band(Course::Mixed).0);
        assert!(cruise_band(Course::Chasm).0 > cruise_band(Course::Jump).0);
    }

    #[test]
    fn suite_schedule_is_balanced_and_deterministic() {
        let schedule = || {
            (0..5)
                .flat_map(|it| (0..4).map(move |k| suite_terrain_index(it, k, 4, 10)))
                .collect::<Vec<_>>()
        };
        assert_eq!(schedule(), schedule());
        let visits = schedule();
        for terrain in 0..10 {
            assert_eq!(visits.iter().filter(|&&v| v == terrain).count(), 2);
        }
    }

    #[test]
    fn suite_objective_prefers_route_progress_over_parking() {
        let parked = Rollout {
            reward: -118.0,
            ..Rollout::default()
        };
        let attempted = Rollout {
            reward: -300.0,
            waypoint_fraction: 1.0 / 15.0,
            ..Rollout::default()
        };
        let completed = Rollout {
            reward: -500.0,
            waypoint_fraction: 1.0,
            completion_rate: 1.0,
            ..Rollout::default()
        };
        assert!(training_objective(&attempted) > training_objective(&parked));
        assert!(training_objective(&completed) > training_objective(&attempted));
    }

    #[test]
    fn suite_training_and_evaluation_are_reproducible() {
        let terrains = [
            Terrain::new(Course::Flat, 1),
            Terrain::new(Course::Rubble, 2),
            Terrain::new(Course::Slalom, 3),
        ];
        let run = || {
            let mut t = trainer(123);
            t.cfg.n_dirs = 3;
            t.cfg.n_top = 2;
            t.cfg.horizon = 0.2;
            t.deltas.resize(t.cfg.n_dirs * n_theta(t.policy.frame), 0.0);
            t.record_suite_baseline(&terrains);
            for _ in 0..2 {
                t.iterate_suite(&terrains);
            }
            (
                t.curve.clone(),
                t.last_eval.completion_rate,
                t.last_eval.waypoint_fraction,
            )
        };
        assert_eq!(run(), run());
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn parallel_and_sequential_rollouts_produce_the_same_update() {
        let terrains = [
            Terrain::new(Course::Flat, 1),
            Terrain::new(Course::Rubble, 2),
            Terrain::new(Course::Slalom, 3),
            Terrain::new(Course::Jump, 4),
        ];
        let run = |workers| {
            let mut t = trainer(456);
            t.cfg.n_dirs = 4;
            t.cfg.n_top = 2;
            t.cfg.horizon = 0.3;
            t.cfg.workers = workers;
            t.deltas.resize(t.cfg.n_dirs * n_theta(t.policy.frame), 0.0);
            t.record_suite_baseline(&terrains);
            t.iterate_suite(&terrains);
            (t.policy.theta, t.policy.norm.mean, t.policy.norm.m2)
        };
        assert_eq!(run(1), run(4));
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
