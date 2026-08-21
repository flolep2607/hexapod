use candle_core::Device;
use hexapod_core::Physics;
use hexapod_core::joint_rl::{
    ACT_RANGE, JointEnv, JointReplay, JointRollout, ObsNorm, Stage, n_act, n_obs,
};
use hexapod_core::math::Rng;
use hexapod_core::robot::Frame;
use hexapod_core::terrain::{Course, Terrain};
use hexapod_sac::{SacAgent, SacConfig, UpdateStats};
use rayon::prelude::*;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

type AppResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Debug)]
struct TrainConfig {
    stage: Stage,
    steps: usize,
    start_speed: f64,
    speed_ramp_steps: usize,
    environments: usize,
    replay_capacity: usize,
    warmup_steps: usize,
    warmup_action_std: f64,
    warmup_hold_fraction: f64,
    batch_size: usize,
    updates_per_step: f64,
    reduced: bool,
    curriculum: bool,
    policy_warmup_updates: usize,
    eval_interval: usize,
    eval_episodes: usize,
    seed: u64,
    hidden: usize,
    actor_lr: f64,
    reward_scale: f64,
    initial_alpha: f64,
    target_entropy_per_action: f64,
    action_prior_cost: f64,
    device: String,
    out: PathBuf,
    init: Option<PathBuf>,
    eval: Option<PathBuf>,
}

impl TrainConfig {
    fn from_args() -> AppResult<Self> {
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        if args.iter().any(|arg| arg == "-h" || arg == "--help") {
            print_help();
            std::process::exit(0);
        }
        let stage = parse_stage(value(&args, "--stage").as_deref().unwrap_or("walk-flat"))?;
        Ok(Self {
            stage,
            steps: parse(&args, "--steps", 500_000)?,
            start_speed: parse(&args, "--start-speed", stage.speed())?,
            speed_ramp_steps: parse(&args, "--speed-ramp-steps", 0)?,
            environments: parse(&args, "--envs", 16)?,
            replay_capacity: parse(&args, "--replay", 1_000_000)?,
            warmup_steps: parse(&args, "--warmup", 10_000)?,
            warmup_action_std: parse(&args, "--warmup-action-std", 0.05)?,
            warmup_hold_fraction: parse(&args, "--warmup-hold-fraction", 0.50)?,
            batch_size: parse(&args, "--batch", 256)?,
            updates_per_step: parse(&args, "--utd", 1.0)?,
            reduced: args.iter().any(|a| a == "--reduced"),
            curriculum: args.iter().any(|a| a == "--curriculum"),
            policy_warmup_updates: parse(&args, "--policy-warmup-updates", 1_000)?,
            eval_interval: parse(&args, "--eval-interval", 10_000)?,
            eval_episodes: parse(&args, "--eval-episodes", 8)?,
            seed: parse(&args, "--seed", 1)?,
            hidden: parse(&args, "--hidden", 256)?,
            actor_lr: parse(&args, "--actor-lr", 3.0e-5)?,
            reward_scale: parse(&args, "--reward-scale", 5.0)?,
            initial_alpha: parse(&args, "--initial-alpha", 0.001)?,
            target_entropy_per_action: parse(&args, "--target-entropy-per-action", -2.5)?,
            action_prior_cost: parse(&args, "--action-prior-cost", 1.0)?,
            device: value(&args, "--device").unwrap_or_else(|| "cpu".into()),
            out: PathBuf::from(
                value(&args, "--out")
                    .unwrap_or_else(|| "checkpoints/joint-sac-walk-v1.safetensors".into()),
            ),
            init: value(&args, "--init").map(PathBuf::from),
            eval: value(&args, "--eval").map(PathBuf::from),
        })
    }

    fn validate(&self) -> AppResult<()> {
        if self.steps == 0
            || self.environments == 0
            || self.replay_capacity == 0
            || self.batch_size == 0
            || self.eval_interval == 0
            || self.eval_episodes == 0
            || self.hidden == 0
        {
            return Err("counts must all be non-zero".into());
        }
        if self.replay_capacity < self.batch_size {
            return Err("--replay must be at least --batch".into());
        }
        if !self.start_speed.is_finite()
            || self.start_speed < 0.0
            || self.start_speed > self.stage.speed()
        {
            return Err(format!(
                "--start-speed must be between zero and the {} target {:.2}",
                self.stage.name(),
                self.stage.speed()
            )
            .into());
        }
        if !self.updates_per_step.is_finite() || self.updates_per_step < 0.0 {
            return Err("--utd must be finite and non-negative".into());
        }
        if !self.warmup_action_std.is_finite() || self.warmup_action_std <= 0.0 {
            return Err("--warmup-action-std must be finite and positive".into());
        }
        if !self.warmup_hold_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.warmup_hold_fraction)
        {
            return Err("--warmup-hold-fraction must be between zero and one".into());
        }
        if !self.actor_lr.is_finite() || self.actor_lr <= 0.0 {
            return Err("--actor-lr must be finite and positive".into());
        }
        if !self.reward_scale.is_finite() || self.reward_scale <= 0.0 {
            return Err("--reward-scale must be finite and positive".into());
        }
        if !self.initial_alpha.is_finite() || self.initial_alpha <= 0.0 {
            return Err("--initial-alpha must be finite and positive".into());
        }
        if !self.target_entropy_per_action.is_finite() {
            return Err("--target-entropy-per-action must be finite".into());
        }
        if !self.action_prior_cost.is_finite() || self.action_prior_cost < 0.0 {
            return Err("--action-prior-cost must be finite and non-negative".into());
        }
        Ok(())
    }
}

fn main() -> AppResult<()> {
    let config = TrainConfig::from_args()?;
    config.validate()?;
    let device = select_device(&config.device)?;
    if let Some(policy) = config.eval.clone() {
        evaluate_checkpoint(&config, &device, &policy)
    } else {
        train(config, device)
    }
}

fn train(config: TrainConfig, device: Device) -> AppResult<()> {
    let frame = Frame::new(6);
    let observations = n_obs(frame);
    let actions = n_act(frame);
    let physics = if config.reduced {
        Physics::reduced()
    } else {
        Physics::default()
    };
    let stage = config.stage;
    let ladder_ceiling = ceiling(stage);
    // Every environment starts on the easiest rung. Promotion takes one
    // episode, so the fleet spreads itself within seconds; seeding it spread
    // out instead just fills replay with a policy falling over on terrain it
    // cannot yet walk on.
    let reviewers = (config.environments as f64 * REVIEW_SHARE).round() as usize;
    let mut rungs = config.curriculum.then(|| {
        (0..config.environments)
            .map(|index| Rung {
                level: 0,
                course: LADDER[0].courses()[0],
                review: index < reviewers,
            })
            .collect::<Vec<_>>()
    });
    let mut episode_seed = 0u64;
    let mut bar = FinishBar::default();
    let mut reach = ReachBar::default();
    let mut horizons = vec![stage.horizon(); config.environments];
    let mut environments = (0..config.environments)
        .map(|index| {
            JointEnv::new(
                frame,
                &physics,
                Terrain::new(Course::Flat, config.seed + index as u64),
                stage,
            )
        })
        .collect::<Vec<_>>();
    let mut states = environments
        .iter()
        .map(|environment| environment.state().to_vec())
        .collect::<Vec<_>>();
    let mut normalizer = if let Some(path) = &config.init {
        let (hidden, normalizer) = load_checkpoint_state(path, observations, actions)?;
        if hidden != config.hidden {
            return Err(format!(
                "initial checkpoint uses hidden width {hidden}, but --hidden is {}",
                config.hidden
            )
            .into());
        }
        normalizer
    } else {
        ObsNorm::new(observations)
    };
    let mut replay = JointReplay::new(config.replay_capacity, observations, actions)?;
    let mut rng = Rng::new(config.seed ^ 0x51AC_2026);
    let sac_config = SacConfig {
        hidden: config.hidden,
        actor_lr: config.actor_lr,
        reward_scale: config.reward_scale,
        initial_alpha: config.initial_alpha,
        target_entropy_per_action: config.target_entropy_per_action,
        action_prior_cost: config.action_prior_cost,
        ..SacConfig::default()
    };
    let mut agent = SacAgent::new(observations, actions, &device, sac_config, config.seed)?;
    if let Some(path) = &config.init {
        agent.load_actor_prefix(path)?;
        agent.freeze_actor_prior()?;
    }
    let mut evaluation_agent =
        SacAgent::new(observations, actions, &Device::Cpu, sac_config, config.seed)?;

    for (environment, horizon) in environments.iter_mut().zip(&horizons) {
        environment.set_horizon(*horizon);
    }

    let started = Instant::now();
    let mut transitions = 0usize;
    let mut next_evaluation = config.eval_interval;
    let mut update_budget = 0.0f64;
    let mut updates = 0usize;
    let mut last_stats: Option<UpdateStats> = None;
    let mut best_score = f64::NEG_INFINITY;

    println!(
        "# native SAC · {} · speed {:.2}->{:.2} over {} steps · {} envs · replay {} · batch {} · UTD {:.2} · train {:?} · eval Cpu",
        stage.name(),
        config.start_speed,
        stage.speed(),
        config.speed_ramp_steps,
        config.environments,
        config.replay_capacity,
        config.batch_size,
        config.updates_per_step,
        device
    );
    println!(
        " steps   replay updates  cmd≤  score   dist  feet    alpha entropy   q      losses (critic/actor)  wall"
    );

    if config.init.is_some() {
        evaluation_agent.copy_actor_from(&agent)?;
        let evaluation = evaluate(
            &evaluation_agent,
            &normalizer,
            &physics,
            frame,
            stage,
            observations,
            config.eval_episodes,
            config.seed + 1_000_001,
        )?;
        best_score = evaluation.score;
        save_checkpoint(&agent, &normalizer, &config, &evaluation)?;
        println!(
            "{transitions:>7} {replay_len:>8} {updates:>7} {command:>5.2} {score:>6.3} {distance:>6.2} {support:>5.2} {alpha:>8.6} {entropy:>7.3} {q:>6.2}  {critic:>8.4}/{actor:>8.4}  {wall:>5.0}s",
            transitions = 0,
            replay_len = 0,
            updates = 0,
            command = stage.speed(),
            score = evaluation.score,
            distance = evaluation.distance,
            support = evaluation.support,
            alpha = agent.alpha()?,
            entropy = 0.0,
            q = 0.0,
            critic = 0.0,
            actor = 0.0,
            wall = started.elapsed().as_secs_f64(),
        );
    }

    while transitions < config.steps {
        let command_ceiling = curriculum_speed(
            stage,
            config.start_speed,
            config.speed_ramp_steps,
            transitions,
        );
        let environment_count = environments.len();
        for (index, (environment, state)) in environments.iter_mut().zip(&mut states).enumerate() {
            let training_command = match rungs.as_ref() {
                // Each rung carries its own command, including the run-up the
                // parkour courses need; the stratified ramp is a flat-ground
                // tool and does not belong on a trench.
                Some(rungs) => rungs[index].command(),
                None => stratified_speed(
                    config.start_speed,
                    command_ceiling,
                    index,
                    environment_count,
                ),
            };
            let observation = environment.set_command(training_command)?;
            state.copy_from_slice(observation);
        }
        for state in &states {
            normalizer.observe(state);
        }
        let take = (config.steps - transitions).min(environments.len());
        let unit_actions = if transitions < config.warmup_steps && config.init.is_none() {
            let hold = (take as f64 * config.warmup_hold_fraction).round() as usize;
            (0..take)
                .flat_map(|environment| {
                    (0..actions)
                        .map(|_| {
                            if environment < hold {
                                0.0
                            } else {
                                (rng.normal() * config.warmup_action_std).clamp(-1.0, 1.0) as f32
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        } else {
            let flat = states[..take]
                .iter()
                .flat_map(|state| state.iter().copied())
                .collect::<Vec<_>>();
            agent.action(&flat, &normalizer, true, &mut rng)?
        };
        let physical_actions = unit_actions
            .chunks_exact(actions)
            .map(|action| {
                action
                    .iter()
                    .map(|value| *value as f64 * ACT_RANGE)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let results = environments[..take]
            .par_iter_mut()
            .zip(&physical_actions)
            .map(|(environment, action)| environment.step(action))
            .collect::<Vec<_>>();

        for index in 0..take {
            let step = results[index]
                .as_ref()
                .map_err(|error| format!("environment {index}: {error}"))?;
            replay.push(
                &states[index],
                &unit_actions[index * actions..(index + 1) * actions]
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>(),
                step.learning_reward,
                &step.observation,
                step.terminated,
                step.truncated,
            )?;
            states[index] = if step.terminated || step.truncated {
                let summary = environments[index].summary();
                if summary.completed {
                    bar.observe(summary.secs);
                }
                reach.observe(summary.distance);
                let floor = rungs
                    .as_ref()
                    .map_or(stage.horizon(), |rungs| rungs[index].stage().horizon());
                horizons[index] = grow_horizon(horizons[index], floor, &summary, reach.value());
                if let Some(rungs) = rungs.as_mut() {
                    let progress = rungs[index].progress(&summary);
                    rungs[index] = advance(rungs[index], ladder_ceiling, progress, &mut rng);
                    episode_seed += 1;
                    environments[index] = rung_env(
                        rungs[index],
                        frame,
                        &physics,
                        config.seed ^ (0x9E37_79B9 * episode_seed),
                    );
                }
                environments[index].set_horizon(horizons[index]);
                environments[index].set_finish_bar(bar.value());
                environments[index].reset().to_vec()
            } else {
                step.observation.clone()
            };
        }
        transitions += take;

        if transitions >= config.warmup_steps && replay.len() >= config.batch_size {
            update_budget += config.updates_per_step * take as f64;
            while update_budget >= 1.0 {
                let batch = replay.sample(config.batch_size, &mut rng)?;
                last_stats = Some(agent.update(
                    &batch,
                    &normalizer,
                    &mut rng,
                    updates >= config.policy_warmup_updates,
                )?);
                updates += 1;
                update_budget -= 1.0;
            }
        }

        if transitions >= next_evaluation || transitions == config.steps {
            evaluation_agent.copy_actor_from(&agent)?;
            let evaluation = evaluate(
                &evaluation_agent,
                &normalizer,
                &physics,
                frame,
                stage,
                observations,
                config.eval_episodes,
                config.seed + 1_000_001,
            )?;
            let stats = last_stats.unwrap_or(UpdateStats {
                critic_loss: 0.0,
                actor_loss: 0.0,
                alpha_loss: 0.0,
                alpha: agent.alpha()?,
                mean_q: 0.0,
                mean_entropy_per_action: 0.0,
            });
            println!(
                "{transitions:>7} {replay_len:>8} {updates:>7} {command:>5.2} {score:>6.3} {distance:>6.2} {support:>5.2} {alpha:>8.6} {entropy:>7.3} {q:>6.2}  {critic:>8.4}/{actor:>8.4}  {wall:>5.0}s",
                replay_len = replay.len(),
                command = command_ceiling,
                score = evaluation.score,
                distance = evaluation.distance,
                support = evaluation.support,
                alpha = stats.alpha,
                entropy = stats.mean_entropy_per_action,
                q = stats.mean_q,
                critic = stats.critic_loss,
                actor = stats.actor_loss,
                wall = started.elapsed().as_secs_f64(),
            );
            if let Some(rungs) = rungs.as_ref() {
                let mut histogram = [0usize; LADDER.len()];
                for rung in rungs.iter() {
                    histogram[rung.level] += 1;
                }
                let climbers = rungs.iter().filter(|r| !r.review).collect::<Vec<_>>();
                let mean = if climbers.is_empty() {
                    0.0
                } else {
                    climbers.iter().map(|r| r.level as f64).sum::<f64>() / climbers.len() as f64
                };
                println!(
                    "#   bar {} · reach {:.2} m · horizon {:.1}-{:.1} s",
                    if bar.value() > 0.0 {
                        format!("{:.2} s", bar.value())
                    } else {
                        "none yet".into()
                    },
                    reach.value(),
                    horizons.iter().copied().fold(f64::INFINITY, f64::min),
                    horizons.iter().copied().fold(0.0, f64::max),
                );
                println!(
                    "#   rungs {} · climbing {:.2} · review {} · ceiling {}",
                    histogram
                        .iter()
                        .take(ladder_ceiling + 1)
                        .enumerate()
                        .map(|(level, count)| format!("{}={count}", LADDER[level].name()))
                        .collect::<Vec<_>>()
                        .join(" "),
                    mean,
                    rungs.len() - climbers.len(),
                    LADDER[ladder_ceiling].name(),
                );
            }
            save_actor_to(
                &snapshot_path(&config.out),
                &agent,
                &normalizer,
                &config,
                &evaluation,
            )?;
            if evaluation.score > best_score {
                best_score = evaluation.score;
                save_checkpoint(&agent, &normalizer, &config, &evaluation)?;
            }
            while next_evaluation <= transitions {
                next_evaluation += config.eval_interval;
            }
        }
    }
    Ok(())
}

fn evaluate(
    agent: &SacAgent,
    normalizer: &ObsNorm,
    physics: &Physics,
    frame: Frame,
    stage: Stage,
    actor_observations: usize,
    episodes: usize,
    seed: u64,
) -> AppResult<JointRollout> {
    let mut total = JointRollout::default();
    let mut rng = Rng::new(seed ^ 0xE7A1);
    // Round-robin the stage's courses rather than pinning flat ground, so a
    // terrain stage is scored on what it trains on. `JointEnv` takes its
    // command from `speed_for(course)`, which is what gives the parkour pair
    // its run-up. Pick `--eval-episodes` as a multiple of the course count if
    // you want the mean to weight them evenly.
    let courses = stage.courses();
    for episode in 0..episodes {
        let mut environment = JointEnv::new(
            frame,
            physics,
            Terrain::new(courses[episode % courses.len()], seed + episode as u64),
            stage,
        );
        while !environment.is_done() {
            let state = environment.state();
            if actor_observations > state.len() {
                return Err(format!(
                    "actor expects {actor_observations} observations, environment provides {}",
                    state.len()
                )
                .into());
            }
            let unit = agent.action(&state[..actor_observations], normalizer, false, &mut rng)?;
            let action = unit
                .iter()
                .map(|value| *value as f64 * ACT_RANGE)
                .collect::<Vec<_>>();
            environment.step(&action)?;
        }
        let rollout = environment.summary();
        total.score += rollout.score;
        total.distance += rollout.distance;
        total.secs += rollout.secs;
        total.support += rollout.support;
        total.air += rollout.air;
        total.reached += rollout.reached;
        total.waypoint_fraction += rollout.waypoint_fraction;
        total.completion_rate += rollout.completion_rate;
        total.finish_time += rollout.finish_time;
        total.fell |= rollout.fell;
    }
    let count = episodes as f64;
    total.score /= count;
    total.distance /= count;
    total.secs /= count;
    total.support /= count;
    total.air /= count;
    total.waypoint_fraction /= count;
    total.completion_rate /= count;
    total.finish_time /= count;
    Ok(total)
}

fn evaluate_checkpoint(config: &TrainConfig, device: &Device, path: &Path) -> AppResult<()> {
    let frame = Frame::new(6);
    let observations = n_obs(frame);
    let actions = n_act(frame);
    let metadata = std::fs::read_to_string(meta_path(path))?;
    let stage = parse_stage(metadata_required(&metadata, "stage")?)?;
    // Which plant this actor was trained against. Absent in checkpoints
    // written before the reduced-coordinate path existed, and those are all
    // impulse — so absent means false, not unknown.
    let physics = if metadata_field(&metadata, "reduced") == Some("true") {
        Physics::reduced()
    } else {
        Physics::default()
    };
    let (hidden, normalizer) = load_checkpoint_state(path, observations, actions)?;
    let actor_observations = normalizer.mean.len();

    let mut agent = SacAgent::new(
        actor_observations,
        actions,
        device,
        SacConfig {
            hidden,
            ..SacConfig::default()
        },
        config.seed,
    )?;
    agent.load_actor_prefix(path)?;
    let rollout = evaluate(
        &agent,
        &normalizer,
        &physics,
        frame,
        stage,
        actor_observations,
        config.eval_episodes,
        config.seed + 1_000_001,
    )?;
    println!(
        "# SAC checkpoint {} · {} · {} held-out episode(s) · {:?}",
        path.display(),
        stage.name(),
        config.eval_episodes,
        device
    );
    println!("score   dist   wp %  finish %  time  feet  fell");
    println!(
        "{:.3}  {:>5.2}  {:>5.1}  {:>8.1}  {:>4.2}  {:>4.2}  {}",
        rollout.score,
        rollout.distance,
        100.0 * rollout.waypoint_fraction,
        100.0 * rollout.completion_rate,
        rollout.finish_time,
        rollout.support,
        if rollout.fell { "yes" } else { "no" },
    );
    Ok(())
}

fn save_checkpoint(
    agent: &SacAgent,
    normalizer: &ObsNorm,
    config: &TrainConfig,
    evaluation: &JointRollout,
) -> AppResult<()> {
    save_actor_to(&config.out, agent, normalizer, config, evaluation)
}

/// The most recent actor, written at every evaluation whatever it scored.
///
/// `save_checkpoint` promotes on improvement, which is the right rule for
/// selection and the wrong one for a run left alone overnight: a long plateau
/// followed by a crash loses every transition since the last gain. This is the
/// crash-recovery copy, never the promoted one — resume from it with `--init`,
/// but judge the run by `--out`.
fn snapshot_path(out: &Path) -> PathBuf {
    let mut value = out.as_os_str().to_owned();
    value.push(".latest");
    PathBuf::from(value)
}

fn save_actor_to(
    path: &Path,
    agent: &SacAgent,
    normalizer: &ObsNorm,
    config: &TrainConfig,
    evaluation: &JointRollout,
) -> AppResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    agent.save_actor(path)?;
    let metadata = format!(
        "format=hexapod-sac-actor-v1\nstage={}\nscore={:.17}\ndistance={:.17}\nseed={}\ninit={}\nstart_speed={:.17}\nspeed_ramp_steps={}\nobservations={}\nactions={}\nhidden={}\nactor_lr={:.17}\nreward_scale={:.17}\ninitial_alpha={:.17}\ntarget_entropy_per_action={:.17}\naction_prior_cost={:.17}\nwarmup_action_std={:.17}\nwarmup_hold_fraction={:.17}\npolicy_warmup_updates={}\nreduced={}\ncurriculum={}\nnorm_n={:.17}\nnorm_mean={}\nnorm_m2={}\n",
        config.stage.name(),
        evaluation.score,
        evaluation.distance,
        config.seed,
        config
            .init
            .as_deref()
            .map_or_else(|| "none".into(), |path| path.display().to_string()),
        config.start_speed,
        config.speed_ramp_steps,
        normalizer.mean.len(),
        n_act(Frame::new(6)),
        config.hidden,
        config.actor_lr,
        config.reward_scale,
        config.initial_alpha,
        config.target_entropy_per_action,
        config.action_prior_cost,
        config.warmup_action_std,
        config.warmup_hold_fraction,
        config.policy_warmup_updates,
        config.reduced,
        config.curriculum,
        normalizer.n,
        join_f64(&normalizer.mean),
        join_f64(&normalizer.m2),
    );
    std::fs::write(meta_path(path), metadata)?;
    Ok(())
}

fn meta_path(path: &Path) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(".meta");
    PathBuf::from(value)
}

fn load_checkpoint_state(
    path: &Path,
    observations: usize,
    actions: usize,
) -> AppResult<(usize, ObsNorm)> {
    let metadata = std::fs::read_to_string(meta_path(path))?;
    if metadata_field(&metadata, "format") != Some("hexapod-sac-actor-v1") {
        return Err("checkpoint metadata is not hexapod-sac-actor-v1".into());
    }
    let stored_observations = metadata_required(&metadata, "observations")?.parse::<usize>()?;
    let stored_actions = metadata_required(&metadata, "actions")?.parse::<usize>()?;
    let legacy_observations = observations.saturating_sub(actions);
    if (stored_observations != observations && stored_observations != legacy_observations)
        || stored_actions != actions
    {
        return Err(format!(
            "checkpoint dimensions {stored_observations}x{stored_actions}, expected {observations}x{actions} (or legacy {legacy_observations}x{actions})"
        )
        .into());
    }
    let hidden = metadata_required(&metadata, "hidden")?.parse::<usize>()?;
    let mut normalizer = ObsNorm::new(stored_observations);
    normalizer.n = metadata_required(&metadata, "norm_n")?.parse()?;
    normalizer.mean = parse_f64_list(metadata_required(&metadata, "norm_mean")?)?;
    normalizer.m2 = parse_f64_list(metadata_required(&metadata, "norm_m2")?)?;
    if normalizer.mean.len() != stored_observations || normalizer.m2.len() != stored_observations {
        return Err("checkpoint normalizer width does not match its observation width".into());
    }
    if stored_observations == legacy_observations {
        normalizer.mean.resize(observations, 0.0);
        // A unit scale is a conservative default for observations that the
        // migrated actor initially ignores through zero-padded input weights.
        normalizer.m2.resize(observations, normalizer.n.max(1.0));
    }
    normalizer.frozen = true;
    Ok((hidden, normalizer))
}

fn join_f64(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.17}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn metadata_field<'a>(metadata: &'a str, key: &str) -> Option<&'a str> {
    metadata.lines().find_map(|line| {
        let (candidate, value) = line.split_once('=')?;
        (candidate == key).then_some(value)
    })
}

fn metadata_required<'a>(metadata: &'a str, key: &str) -> AppResult<&'a str> {
    metadata_field(metadata, key)
        .ok_or_else(|| format!("checkpoint metadata is missing {key}").into())
}

fn parse_f64_list(value: &str) -> AppResult<Vec<f64>> {
    value
        .split(',')
        .map(|item| item.parse::<f64>().map_err(Into::into))
        .collect()
}

fn select_device(name: &str) -> AppResult<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" | "cuda:0" => Ok(Device::new_cuda(0)?),
        #[cfg(feature = "cuda")]
        value if value.starts_with("cuda:") => {
            let ordinal = value[5..].parse::<usize>()?;
            Ok(Device::new_cuda(ordinal)?)
        }
        #[cfg(not(feature = "cuda"))]
        value if value.starts_with("cuda") => {
            Err("CUDA support is not compiled; rebuild hexapod-sac with --features cuda".into())
        }
        _ => Err(format!("unknown device {name:?}; use cpu or cuda[:N]").into()),
    }
}

fn value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|argument| argument == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn parse<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> AppResult<T>
where
    T::Err: Error + 'static,
{
    match value(args, flag) {
        Some(value) => Ok(value.parse()?),
        None => Ok(default),
    }
}

fn parse_stage(value: &str) -> AppResult<Stage> {
    match value.to_ascii_lowercase().replace('_', "-").as_str() {
        "walk-flat" | "walk" => Ok(Stage::WalkFlat),
        "run-flat" | "run" => Ok(Stage::RunFlat),
        "rough" => Ok(Stage::Rough),
        "gaps" => Ok(Stage::Gaps),
        "jump" => Ok(Stage::Jump),
        "mixed" => Ok(Stage::Mixed),
        _ => Err(format!(
            "unsupported SAC stage {value:?}; use walk-flat, run-flat, rough, gaps, jump or mixed"
        )
        .into()),
    }
}

/// Difficulty ladder for the per-environment curriculum, in the order the
/// stages already define. Nothing here is a new judgement about what is hard:
/// each rung is one existing stage, with its own courses, command and horizon.
/// `MIXED` is a rung and not just a ceiling: the four stages below it name six
/// courses between them, and `MIXED` is the only stage that samples all
/// fifteen. Without it here, SLALOM, SLICK, RAMPS, GAUNTLET, BEAM, PILLARS,
/// WASHBOARD and GLACIER would never be trained at any difficulty.
const LADDER: [Stage; 5] = [
    Stage::WalkFlat,
    Stage::Rough,
    Stage::Gaps,
    Stage::Jump,
    Stage::Mixed,
];

/// Share of environments that ignore promotion and resample the whole ladder
/// every episode.
///
/// Promotion moves an environment up exactly one rung, so while the fleet is
/// climbing, the rungs it has left empty out and nothing in the replay window
/// remembers them. The buffer is no defence: 2M transitions at ~1,900 steps/s
/// is about eighteen minutes of experience, so flat-ground transitions are gone
/// long before a terrain stage is finished with. These environments are the
/// standing review set that keeps every cleared rung in the data distribution.
const REVIEW_SHARE: f64 = 0.25;

/// Episodes in the moving average behind the finish bar.
const FINISH_WINDOW: f64 = 200.0;

/// How much an episode moves its environment's horizon, up or down.
const HORIZON_STEP: f64 = 1.15;

/// Longest an episode may grow, as a multiple of its stage's own horizon.
///
/// This was 6.0, and 6.0 was too much. Episodes reached 42 s against a stage
/// horizon of 8, so at a fixed transition budget the fleet saw roughly a
/// quarter as many episodes, resets and course seeds — and the score stopped
/// improving. Long episodes are how route-following gets learned and short ones
/// are where sample diversity comes from; 2.0 is a compromise that cannot run
/// away from the diversity.
const HORIZON_MAX: f64 = 2.0;

/// Episodes in the moving average behind the reach bar.
const REACH_WINDOW: f64 = 200.0;

/// The finish time worth half the terminal bonus: the best the policy has
/// managed, never the merely recent.
///
/// A bar that tracks a moving average alone can be ridden — get slower, let it
/// follow you down, recover, and collect the improvement again. Taking the
/// better of the recent average and the best ever makes it monotone, so a level
/// already reached pays exactly what it paid before and there is nothing to
/// farm. Lower is better here, so the ratchet is a minimum.
#[derive(Clone, Copy, Default)]
struct FinishBar {
    best: f64,
    recent: f64,
}

impl FinishBar {
    /// Zero until an episode has actually arrived. No terminal bonus is paid
    /// before that: there is nothing to be half as good as yet.
    fn value(self) -> f64 {
        self.best
    }

    fn observe(&mut self, secs: f64) {
        if secs <= 0.0 {
            return;
        }
        self.recent = if self.recent <= 0.0 {
            secs
        } else {
            self.recent + (secs - self.recent) / FINISH_WINDOW
        };
        self.best = if self.best <= 0.0 {
            self.recent
        } else {
            self.best.min(self.recent)
        };
    }
}

/// Highest rung `stage` allows.
fn ceiling(stage: Stage) -> usize {
    match stage {
        Stage::Stand | Stage::WalkFlat | Stage::RunFlat => 0,
        Stage::Rough => 1,
        Stage::Gaps => 2,
        Stage::Jump => 3,
        Stage::Mixed => 4,
    }
}

/// Where one environment currently sits.
#[derive(Clone, Copy)]
struct Rung {
    level: usize,
    course: Course,
    /// Never promoted or demoted; resamples the whole ladder instead.
    review: bool,
}

impl Rung {
    fn stage(self) -> Stage {
        LADDER[self.level]
    }

    fn command(self) -> f64 {
        self.stage().speed_for(self.course)
    }

    fn sample(level: usize, review: bool, rng: &mut Rng) -> Rung {
        let courses = LADDER[level].courses();
        let course = courses[rng.next_u64() as usize % courses.len()];
        Rung { level, course, review }
    }

    /// How much of this rung the episode got through, on a scale where 1.0 is
    /// a pass.
    ///
    /// Neither available measure works alone. Ground covered against
    /// `command x horizon` is the right test on the four-second flat stage and
    /// unreachable on `MIXED`, whose 2.5 m/s over 30 s asks for 75 m of a course
    /// about forty long. Route fraction is the right test on `MIXED` and close
    /// to meaningless on flat, where four seconds of walking clears one of
    /// eight waypoints however well it went. Whichever says the episode went
    /// well is the one that did the measuring.
    fn progress(self, rollout: &JointRollout) -> f64 {
        let reach = self.command() * self.stage().horizon();
        let by_ground = if reach > 0.0 { rollout.distance / reach } else { 0.0 };
        by_ground.max(rollout.waypoint_fraction)
    }
}

/// Promote an environment that covered its ground, demote one that did not.
///
/// This is the `legged_gym` rule rather than a global stage gate: levels are
/// per robot, so every rung below the ceiling stays populated for as long as
/// some environment keeps failing out of it, and easy terrain never leaves the
/// replay. A global ladder promotes the whole run at once and then has nothing
/// left that can still walk on flat ground.
///
/// At the ceiling a passing episode resamples anywhere at or below it, which is
/// what stops the top rung from crowding out everything it was built on.
fn advance(rung: Rung, ceiling: usize, progress: f64, rng: &mut Rng) -> Rung {
    if rung.review {
        return Rung::sample(rng.next_u64() as usize % (ceiling + 1), true, rng);
    }
    if progress >= 0.8 {
        if rung.level >= ceiling {
            return Rung::sample(rng.next_u64() as usize % (ceiling + 1), false, rng);
        }
        return Rung::sample(rung.level + 1, false, rng);
    }
    if progress < 0.5 && rung.level > 0 {
        return Rung::sample(rung.level - 1, false, rng);
    }
    Rung::sample(rung.level, false, rng)
}

/// Monotone best of a moving average of ground covered. Same ratchet as
/// [`FinishBar`], the other way up because more distance is better.
#[derive(Clone, Copy, Default)]
struct ReachBar {
    best: f64,
    recent: f64,
}

impl ReachBar {
    fn value(self) -> f64 {
        self.best
    }

    fn observe(&mut self, metres: f64) {
        if !metres.is_finite() || metres <= 0.0 {
            return;
        }
        self.recent = if self.recent <= 0.0 {
            metres
        } else {
            self.recent + (metres - self.recent) / REACH_WINDOW
        };
        self.best = self.best.max(self.recent);
    }
}

/// How long this environment's next episode should be.
///
/// The clock is extended when the clock is what stopped it. Falling is not a
/// shortage of time and neither is standing still, so both give some back —
/// short episodes are cheap, and a policy that is not moving learns faster from
/// many resets than from long ones. Arriving extends nothing: an episode that
/// completed had time to spare.
///
/// Both earlier conditions were the wrong shape. Requiring
/// `waypoint_fraction >= 0.5` — half the whole route — could never be earned:
/// the machine walked 4.1 m of a 40 m course, so every environment stayed
/// pinned at its stage horizon and the finish was unreachable by construction.
/// Replacing it with "still moving forward when the clock expired" is true of
/// *any* walking policy, so the horizon grew unconditionally to the cap and
/// stayed there, and the score stopped improving. Neither one measured
/// competence.
///
/// `reach` does: it is the monotone best of a moving average of ground covered,
/// so only an episode that beat what the policy has been managing earns more
/// time. As the policy improves the bar rises with it, which makes the growth
/// self-limiting rather than a one-way trip to the ceiling.
fn grow_horizon(current: f64, floor: f64, summary: &JointRollout, reach: f64) -> f64 {
    let factor = if summary.completed {
        1.0
    } else if summary.fell || summary.distance <= 0.0 {
        1.0 / HORIZON_STEP
    } else if summary.distance >= reach {
        HORIZON_STEP
    } else {
        1.0
    };
    (current * factor).clamp(floor, floor * HORIZON_MAX)
}

fn rung_env(rung: Rung, frame: Frame, physics: &Physics, seed: u64) -> JointEnv {
    JointEnv::new(
        frame,
        physics,
        Terrain::new(rung.course, seed),
        rung.stage(),
    )
}

fn curriculum_speed(stage: Stage, start: f64, ramp_steps: usize, transitions: usize) -> f64 {
    if ramp_steps == 0 {
        return stage.speed();
    }
    let progress = (transitions as f64 / ramp_steps as f64).clamp(0.0, 1.0);
    start + progress * (stage.speed() - start)
}

fn stratified_speed(start: f64, ceiling: f64, index: usize, count: usize) -> f64 {
    if count <= 1 {
        return ceiling;
    }
    let fraction = index.min(count - 1) as f64 / (count - 1) as f64;
    start + fraction * (ceiling - start)
}

fn print_help() {
    println!(
        "hexapod-sac — native off-policy motor learner\n\n\
         --stage NAME         walk-flat or run-flat (default walk-flat)\n\
         --steps N            collected transitions (default 500000)\n\
         --start-speed X      initial command for a speed curriculum\n\
         --speed-ramp-steps N transitions used to reach the stage speed\n\
         --envs N             parallel reusable Rapier worlds (default 16)\n\
         --replay N           replay capacity (default 1000000)\n\
         --warmup N           random transitions before gradients (default 10000)\n\
         --warmup-action-std X Gaussian warm-up scale in unit actions (default 0.05)\n\
         --warmup-hold-fraction X exact standing share of warm-up envs (default 0.50)\n\
         --batch N            replay minibatch (default 256)\n\
         --utd X              gradient updates per transition (default 1.0)\n\
         --reduced            reduced-coordinate plant: same machine, ~1.6x faster\n\
         --curriculum         per-environment difficulty levels up to --stage\n\
         --policy-warmup-updates N critic-only updates before actor training (default 1000)\n\
         --eval-interval N    held-out evaluation cadence (default 10000)\n\
         --eval-episodes N    held-out episodes (default 8)\n\
         --hidden N           units in each actor/critic layer (default 256)\n\
         --actor-lr X         actor learning rate (default 3e-5)\n\
         --reward-scale X     Bellman reward scale (default 5)\n\
         --initial-alpha X    initial entropy coefficient (default 0.001)\n\
         --target-entropy-per-action X entropy target per joint (default -2.5)\n\
         --action-prior-cost X normalized quadratic actor prior (default 1)\n\
         --device cpu|cuda:N  tensor device (CUDA requires --features cuda)\n\
         --seed N             deterministic run seed\n\
         --out PATH           best actor safetensors checkpoint\n\
         --init PATH          fine-tune an actor with its frozen normalizer\n\
         --eval PATH          evaluate a saved actor with its frozen normalizer"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sac_stage_names_cover_the_whole_ladder() {
        assert_eq!(parse_stage("walk-flat").expect("walk"), Stage::WalkFlat);
        assert_eq!(parse_stage("RUN_FLAT").expect("run"), Stage::RunFlat);
        // Terrain used to be refused here because only the ARS trainer could
        // reach it. `--stage` now names the curriculum's ceiling instead.
        assert_eq!(parse_stage("rough").expect("rough"), Stage::Rough);
        assert_eq!(parse_stage("mixed").expect("mixed"), Stage::Mixed);
        assert!(parse_stage("parkour").is_err());
    }

    /// The promotion rule is the only place difficulty moves, so it is the one
    /// thing here that has to be right: a rung that never demotes fills replay
    /// with a policy falling into trenches, and one that never promotes is a
    /// flat-ground trainer with extra steps.
    #[test]
    fn rungs_promote_on_progress_and_demote_without_it() {
        let mut rng = Rng::new(7);
        let top = ceiling(Stage::Mixed);
        assert_eq!(top, LADDER.len() - 1);
        assert_eq!(ceiling(Stage::WalkFlat), 0);
        assert_eq!(ceiling(Stage::Rough), 1);

        let start = Rung { level: 1, course: Stage::Rough.courses()[0], review: false };

        // Covered its ground: up one, and onto that stage's own courses.
        let up = advance(start, top, 0.9, &mut rng);
        assert_eq!(up.level, 2);
        assert!(Stage::Gaps.courses().contains(&up.course));

        // Barely moved: down one. In between: held, so a rung it is only just
        // surviving is not thrown away.
        assert_eq!(advance(start, top, 0.1, &mut rng).level, 0);
        assert_eq!(advance(start, top, 0.65, &mut rng).level, 1);

        // The bottom rung is the floor, however badly it goes.
        let floor = Rung { level: 0, course: Course::Flat, review: false };
        assert_eq!(advance(floor, top, 0.0, &mut rng).level, 0);

        // A ceiling is a ceiling, at the top and lower down: `--stage rough`
        // must not wander onto trenches.
        let peak = Rung { level: top, course: Stage::Mixed.courses()[0], review: false };
        let capped = Rung { level: 1, course: Stage::Rough.courses()[0], review: false };
        for _ in 0..256 {
            assert!(advance(peak, top, 1.0, &mut rng).level <= top);
            assert!(advance(capped, 1, 1.0, &mut rng).level <= 1);
        }
    }

    /// Both progress measures are needed, and each is wrong outside its own
    /// regime — this pins which one takes over where.
    #[test]
    fn progress_uses_ground_on_flat_and_the_route_on_mixed() {
        let flat = Rung { level: 0, course: Course::Flat, review: false };
        // Four seconds of walking clears one of eight waypoints, so the route
        // fraction is tiny while the episode was in fact a pass.
        let walked = JointRollout { distance: 3.3, waypoint_fraction: 0.125, ..Default::default() };
        assert!(flat.progress(&walked) >= 0.8, "3.3 m against a 3.2 m reach is a pass");

        let mixed = Rung { level: ceiling(Stage::Mixed), course: Course::Slalom, review: false };
        // 2.5 m/s over 30 s asks 75 m of a course about forty long, so ground
        // covered can never pass; finishing the route has to.
        let finished = JointRollout { distance: 38.0, waypoint_fraction: 1.0, ..Default::default() };
        assert!(mixed.progress(&finished) >= 0.8, "a completed route is a pass");
        let stalled = JointRollout { distance: 2.0, waypoint_fraction: 0.1, ..Default::default() };
        assert!(mixed.progress(&stalled) < 0.5, "going nowhere is not");
    }

    /// The review set is what stops a cleared rung from leaving the data. The
    /// replay window is about eighteen minutes of experience, so nothing else
    /// remembers flat ground once the fleet has climbed off it.
    #[test]
    fn the_review_set_keeps_every_cleared_rung_in_the_mix() {
        let mut rng = Rng::new(11);
        let top = ceiling(Stage::Mixed);
        let mut seen = [false; LADDER.len()];
        let mut rung = Rung { level: top, course: Stage::Mixed.courses()[0], review: true };
        for _ in 0..2000 {
            // A reviewer passing at the ceiling must still come back down.
            rung = advance(rung, top, 1.0, &mut rng);
            assert!(rung.review, "a reviewer must stay a reviewer");
            assert!(rung.level <= top);
            seen[rung.level] = true;
        }
        assert!(seen.iter().all(|hit| *hit), "review never revisited some rung: {seen:?}");
    }

    /// Time is extended only when the machine just did better than it has been
    /// doing. Both earlier conditions were the wrong shape: half-the-route
    /// could never be earned, and "still moving forward" is true of any walking
    /// policy, so the horizon went straight to the cap and the score stalled.
    #[test]
    fn only_beating_its_own_reach_buys_more_time() {
        let floor = 8.0;
        let walked = |metres: f64| JointRollout { distance: metres, ..Default::default() };
        let reach = 5.0;

        // Beat the standing best: it can use more time.
        assert!(grow_horizon(floor, floor, &walked(6.0), reach) > floor);
        // Moving, but no better than usual: hold. This is the case that used to
        // grow unconditionally and run the horizon to its ceiling.
        assert_eq!(grow_horizon(floor, floor, &walked(4.0), reach), floor);
        // Went nowhere, or backwards, or fell: give it back, short resets being
        // cheaper for a policy that is not moving. Never below the stage's own.
        assert!(grow_horizon(floor * 1.5, floor, &walked(0.0), reach) < floor * 1.5);
        assert!(grow_horizon(floor * 1.5, floor, &walked(-1.0), reach) < floor * 1.5);
        let fell = JointRollout { distance: 9.0, fell: true, ..Default::default() };
        assert!(grow_horizon(floor * 1.5, floor, &fell, reach) < floor * 1.5);
        assert_eq!(grow_horizon(floor, floor, &fell, reach), floor);
        // Arrived: it had time to spare.
        let done = JointRollout { distance: 40.0, completed: true, ..Default::default() };
        assert_eq!(grow_horizon(floor * 1.5, floor, &done, reach), floor * 1.5);

        // Bounded, and the cap is close enough that episode length cannot run
        // away from the reset diversity.
        let mut horizon = floor;
        for _ in 0..500 {
            horizon = grow_horizon(horizon, floor, &walked(99.0), reach);
        }
        assert_eq!(horizon, floor * HORIZON_MAX);
        assert!(HORIZON_MAX <= 2.0, "a big cap starved the fleet of resets");
    }

    /// The reach bar is the same ratchet as the finish bar, the other way up:
    /// it must not slip when the policy has a bad patch, or a bad patch would
    /// buy the horizon growth that a good one earned.
    #[test]
    fn the_reach_bar_never_slips() {
        let mut reach = ReachBar::default();
        assert_eq!(reach.value(), 0.0);
        for _ in 0..2000 {
            reach.observe(4.0);
        }
        let four = reach.value();
        assert!(four > 0.0 && four <= 4.0);
        for _ in 0..20_000 {
            reach.observe(0.5);
            assert!(reach.value() >= four - 1e-12, "reach slipped to {}", reach.value());
        }
    }

    /// The whole reason the bar is a ratchet. A bar that followed a moving
    /// average could be ridden: get slower, let it drift down with you,
    /// recover, and be paid for the same improvement twice. Taking the best
    /// ever means a level already reached pays what it paid before.
    #[test]
    fn the_finish_bar_never_slips() {
        let mut bar = FinishBar::default();
        assert_eq!(bar.value(), 0.0, "no bar before anything has arrived");

        // Improving pulls it down.
        for _ in 0..2000 {
            bar.observe(10.0);
        }
        let after_ten = bar.value();
        assert!(after_ten > 0.0 && after_ten < 10.5);
        for _ in 0..2000 {
            bar.observe(7.0);
        }
        let after_seven = bar.value();
        assert!(after_seven < after_ten, "a faster run must move the bar in");

        // Getting slower again does not give it back, however long it goes on.
        for _ in 0..20_000 {
            bar.observe(12.0);
            assert!(
                bar.value() <= after_seven + 1e-12,
                "bar slipped to {} from {after_seven}",
                bar.value()
            );
        }
    }

    /// Every course has to be reachable by some rung. The ladder first shipped
    /// topping out at JUMP, which silently meant SLALOM, SLICK, RAMPS,
    /// GAUNTLET, BEAM, PILLARS, WASHBOARD and GLACIER were never trained at
    /// any difficulty — nine of fifteen courses, and nothing failed.
    #[test]
    fn the_ladder_reaches_every_course() {
        let reachable: Vec<Course> = LADDER
            .iter()
            .flat_map(|stage| stage.courses().iter().copied())
            .collect();
        for course in hexapod_core::terrain::COURSES {
            assert!(
                reachable.contains(&course),
                "no rung of the ladder ever trains {course:?}"
            );
        }
    }

}
