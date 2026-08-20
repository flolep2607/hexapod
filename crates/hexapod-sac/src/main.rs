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
    let physics = Physics::default();
    let stage = config.stage;
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
            let training_command = stratified_speed(
                config.start_speed,
                command_ceiling,
                index,
                environment_count,
            );
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
    for episode in 0..episodes {
        let mut environment = JointEnv::new(
            frame,
            physics,
            Terrain::new(Course::Flat, seed + episode as u64),
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
        &Physics::default(),
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
    if let Some(parent) = config.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    agent.save_actor(&config.out)?;
    let metadata = format!(
        "format=hexapod-sac-actor-v1\nstage={}\nscore={:.17}\ndistance={:.17}\nseed={}\ninit={}\nstart_speed={:.17}\nspeed_ramp_steps={}\nobservations={}\nactions={}\nhidden={}\nactor_lr={:.17}\nreward_scale={:.17}\ninitial_alpha={:.17}\ntarget_entropy_per_action={:.17}\naction_prior_cost={:.17}\nwarmup_action_std={:.17}\nwarmup_hold_fraction={:.17}\npolicy_warmup_updates={}\nnorm_n={:.17}\nnorm_mean={}\nnorm_m2={}\n",
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
        normalizer.n,
        join_f64(&normalizer.mean),
        join_f64(&normalizer.m2),
    );
    std::fs::write(meta_path(&config.out), metadata)?;
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
        _ => Err(format!("unsupported SAC stage {value:?}; use walk-flat or run-flat").into()),
    }
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
    fn sac_stage_names_are_explicit_and_reject_harder_terrain() {
        assert_eq!(parse_stage("walk-flat").expect("walk"), Stage::WalkFlat);
        assert_eq!(parse_stage("RUN_FLAT").expect("run"), Stage::RunFlat);
        assert!(parse_stage("mixed").is_err());
    }

    #[test]
    fn speed_curriculum_reaches_and_holds_the_stage_target() {
        assert_eq!(curriculum_speed(Stage::RunFlat, 0.8, 0, 0), 2.0);
        assert_eq!(curriculum_speed(Stage::RunFlat, 0.8, 100, 0), 0.8);
        assert!((curriculum_speed(Stage::RunFlat, 0.8, 100, 50) - 1.4).abs() < 1e-12);
        assert_eq!(curriculum_speed(Stage::RunFlat, 0.8, 100, 100), 2.0);
        assert_eq!(curriculum_speed(Stage::RunFlat, 0.8, 100, 200), 2.0);
        assert_eq!(stratified_speed(0.8, 2.0, 0, 16), 0.8);
        assert_eq!(stratified_speed(0.8, 2.0, 15, 16), 2.0);
        assert_eq!(stratified_speed(0.8, 2.0, 0, 1), 2.0);
    }
}
