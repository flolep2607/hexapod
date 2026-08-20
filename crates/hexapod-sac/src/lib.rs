//! Soft Actor-Critic for the motor-level hexapod environment.
//!
//! This crate deliberately owns the tensor backend while `hexapod-core` owns
//! environment semantics and replay. The split keeps browser builds small and
//! lets the learner use CUDA without making GPU libraries simulator
//! dependencies.

use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, linear};
use hexapod_core::joint_rl::{JointReplayBatch, ObsNorm};
use hexapod_core::math::Rng;

const LOG_STD_MIN: f64 = -5.0;
const LOG_STD_MAX: f64 = 2.0;
const LOG_2PI: f64 = 1.837_877_066_409_345_3;

#[derive(Clone, Debug)]
struct Mlp {
    l1: Linear,
    l2: Linear,
    out: Linear,
}

impl Mlp {
    fn new(input: usize, hidden: usize, output: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            l1: linear(input, hidden, vb.pp("l1"))?,
            l2: linear(hidden, hidden, vb.pp("l2"))?,
            out: linear(hidden, output, vb.pp("out"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.l1.forward(input)?.relu()?;
        let hidden = self.l2.forward(&hidden)?.relu()?;
        self.out.forward(&hidden)
    }
}

#[derive(Clone, Debug)]
struct Actor {
    net: Mlp,
    actions: usize,
}

impl Actor {
    fn new(observations: usize, actions: usize, hidden: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            net: Mlp::new(observations, hidden, actions * 2, vb)?,
            actions,
        })
    }

    fn distribution(&self, observations: &Tensor) -> Result<(Tensor, Tensor)> {
        let output = self.net.forward(observations)?;
        let mean = output.narrow(1, 0, self.actions)?;
        let log_std = output
            .narrow(1, self.actions, self.actions)?
            .clamp(LOG_STD_MIN, LOG_STD_MAX)?;
        Ok((mean, log_std))
    }

    fn deterministic(&self, observations: &Tensor) -> Result<Tensor> {
        self.distribution(observations)?.0.tanh()
    }

    fn sample(&self, observations: &Tensor, epsilon: &Tensor) -> Result<(Tensor, Tensor)> {
        let (mean, log_std) = self.distribution(observations)?;
        let std = log_std.exp()?;
        let pre_tanh = mean.add(&std.mul(epsilon)?)?;
        let action = pre_tanh.tanh()?;

        // Reparameterised diagonal Gaussian followed by tanh. Epsilon is made
        // by the project's seeded host RNG so CPU and CUDA collect the same
        // stochastic policy trajectory for a given seed.
        let gaussian = epsilon
            .sqr()?
            .add(&log_std.affine(2.0, LOG_2PI)?)?
            .affine(-0.5, 0.0)?;
        let correction = action
            .sqr()?
            .affine(-1.0, 1.0)?
            .affine(1.0, 1.0e-6)?
            .log()?;
        let log_probability = gaussian.sub(&correction)?.sum(1)?;
        Ok((action, log_probability))
    }
}

#[derive(Clone, Debug)]
struct TwinCritic {
    q1: Mlp,
    q2: Mlp,
}

impl TwinCritic {
    fn new(observations: usize, actions: usize, hidden: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let input = observations + actions;
        Ok(Self {
            q1: Mlp::new(input, hidden, 1, vb.pp("q1"))?,
            q2: Mlp::new(input, hidden, 1, vb.pp("q2"))?,
        })
    }

    fn forward(&self, observations: &Tensor, actions: &Tensor) -> Result<(Tensor, Tensor)> {
        let input = Tensor::cat(&[observations, actions], 1)?;
        Ok((
            self.q1.forward(&input)?.squeeze(1)?,
            self.q2.forward(&input)?.squeeze(1)?,
        ))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SacConfig {
    pub hidden: usize,
    pub gamma: f64,
    pub tau: f64,
    pub actor_lr: f64,
    pub critic_lr: f64,
    pub alpha_lr: f64,
    /// Episode scores are intentionally O(1), making individual transition
    /// deltas small. This scale keeps Q targets around unity.
    pub reward_scale: f64,
    pub initial_alpha: f64,
}

impl Default for SacConfig {
    fn default() -> Self {
        Self {
            hidden: 256,
            gamma: 0.99,
            tau: 0.005,
            actor_lr: 3.0e-4,
            critic_lr: 3.0e-4,
            alpha_lr: 3.0e-4,
            reward_scale: 25.0,
            initial_alpha: 0.1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UpdateStats {
    pub critic_loss: f32,
    pub actor_loss: f32,
    pub alpha_loss: f32,
    pub alpha: f32,
    pub mean_q: f32,
}

pub struct SacAgent {
    observations: usize,
    actions: usize,
    device: Device,
    config: SacConfig,
    actor: Actor,
    critic: TwinCritic,
    target: TwinCritic,
    actor_vars: VarMap,
    critic_vars: VarMap,
    target_vars: VarMap,
    log_alpha: Var,
    actor_optim: AdamW,
    critic_optim: AdamW,
    alpha_optim: AdamW,
}

impl SacAgent {
    pub fn new(
        observations: usize,
        actions: usize,
        device: &Device,
        config: SacConfig,
        seed: u64,
    ) -> Result<Self> {
        let actor_vars = VarMap::new();
        let actor = Actor::new(
            observations,
            actions,
            config.hidden,
            VarBuilder::from_varmap(&actor_vars, DType::F32, device),
        )?;
        let critic_vars = VarMap::new();
        let critic = TwinCritic::new(
            observations,
            actions,
            config.hidden,
            VarBuilder::from_varmap(&critic_vars, DType::F32, device),
        )?;
        let target_vars = VarMap::new();
        let target = TwinCritic::new(
            observations,
            actions,
            config.hidden,
            VarBuilder::from_varmap(&target_vars, DType::F32, device),
        )?;

        deterministic_init(&actor_vars, seed, device)?;
        initialize_actor_output(&actor_vars, actions, device)?;
        deterministic_init(&critic_vars, seed ^ 0xA5A5_A5A5_A5A5_A5A5, device)?;
        copy_parameters(&critic_vars, &target_vars, 1.0)?;

        let log_alpha = Var::from_tensor(&Tensor::new(config.initial_alpha.ln() as f32, device)?)?;
        let actor_optim = AdamW::new(actor_vars.all_vars(), adam_parameters(config.actor_lr))?;
        let critic_optim = AdamW::new(critic_vars.all_vars(), adam_parameters(config.critic_lr))?;
        let alpha_optim = AdamW::new(vec![log_alpha.clone()], adam_parameters(config.alpha_lr))?;

        Ok(Self {
            observations,
            actions,
            device: device.clone(),
            config,
            actor,
            critic,
            target,
            actor_vars,
            critic_vars,
            target_vars,
            log_alpha,
            actor_optim,
            critic_optim,
            alpha_optim,
        })
    }

    pub fn action(
        &self,
        raw_observations: &[f64],
        normalizer: &ObsNorm,
        stochastic: bool,
        rng: &mut Rng,
    ) -> Result<Vec<f32>> {
        if !raw_observations.len().is_multiple_of(self.observations) {
            candle_core::bail!(
                "observation payload {} is not divisible by width {}",
                raw_observations.len(),
                self.observations
            )
        }
        let batch = raw_observations.len() / self.observations;
        let normalized = normalize_f64(raw_observations, self.observations, normalizer);
        let observations = Tensor::from_vec(normalized, (batch, self.observations), &self.device)?;
        let actions = if stochastic {
            let epsilon = epsilon_tensor(batch, self.actions, rng, &self.device)?;
            self.actor.sample(&observations, &epsilon)?.0
        } else {
            self.actor.deterministic(&observations)?
        };
        actions.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()
    }

    pub fn alpha(&self) -> Result<f32> {
        self.log_alpha.as_tensor().exp()?.to_vec0()
    }

    pub fn update(
        &mut self,
        batch: &JointReplayBatch,
        normalizer: &ObsNorm,
        rng: &mut Rng,
    ) -> Result<UpdateStats> {
        let batch_size = batch.rewards.len();
        if batch_size == 0
            || batch.observation_width != self.observations
            || batch.action_width != self.actions
        {
            candle_core::bail!("SAC batch dimensions do not match the agent")
        }
        let observations = Tensor::from_vec(
            normalize_f32(&batch.observations, self.observations, normalizer),
            (batch_size, self.observations),
            &self.device,
        )?;
        let next_observations = Tensor::from_vec(
            normalize_f32(&batch.next_observations, self.observations, normalizer),
            (batch_size, self.observations),
            &self.device,
        )?;
        let actions = Tensor::from_vec(
            batch.actions.clone(),
            (batch_size, self.actions),
            &self.device,
        )?;
        let rewards = Tensor::from_vec(batch.rewards.clone(), batch_size, &self.device)?;
        let not_terminal = Tensor::from_vec(
            batch
                .terminated
                .iter()
                .map(|terminated| if *terminated { 0.0f32 } else { 1.0 })
                .collect::<Vec<_>>(),
            batch_size,
            &self.device,
        )?;

        let alpha = self.log_alpha.as_tensor().exp()?;
        let next_epsilon = epsilon_tensor(batch_size, self.actions, rng, &self.device)?;
        let (next_actions, next_log_probability) =
            self.actor.sample(&next_observations, &next_epsilon)?;
        let (target_q1, target_q2) = self
            .target
            .forward(&next_observations, &next_actions.detach())?;
        let target_q = target_q1
            .minimum(&target_q2)?
            .sub(&next_log_probability.broadcast_mul(&alpha.detach())?)?;
        let target = rewards
            .affine(self.config.reward_scale, 0.0)?
            .add(
                &not_terminal
                    .mul(&target_q)?
                    .affine(self.config.gamma, 0.0)?,
            )?
            .detach();

        let (q1, q2) = self.critic.forward(&observations, &actions)?;
        let critic_loss = q1
            .sub(&target)?
            .sqr()?
            .mean_all()?
            .add(&q2.sub(&target)?.sqr()?.mean_all()?)?;
        let critic_loss_value = critic_loss.to_vec0::<f32>()?;
        self.critic_optim.backward_step(&critic_loss)?;

        let actor_epsilon = epsilon_tensor(batch_size, self.actions, rng, &self.device)?;
        let (policy_actions, log_probability) = self.actor.sample(&observations, &actor_epsilon)?;
        let (policy_q1, policy_q2) = self.critic.forward(&observations, &policy_actions)?;
        let policy_q = policy_q1.minimum(&policy_q2)?;
        let actor_loss = log_probability
            .broadcast_mul(&alpha.detach())?
            .sub(&policy_q)?
            .mean_all()?;
        let actor_loss_value = actor_loss.to_vec0::<f32>()?;
        self.actor_optim.backward_step(&actor_loss)?;

        let entropy_error = log_probability
            .detach()
            .affine(1.0, -(self.actions as f64))?;
        let alpha_loss = self
            .log_alpha
            .as_tensor()
            .broadcast_mul(&entropy_error)?
            .affine(-1.0, 0.0)?
            .mean_all()?;
        let alpha_loss_value = alpha_loss.to_vec0::<f32>()?;
        self.alpha_optim.backward_step(&alpha_loss)?;

        copy_parameters(&self.critic_vars, &self.target_vars, self.config.tau)?;
        let mean_q = policy_q.mean_all()?.to_vec0::<f32>()?;
        Ok(UpdateStats {
            critic_loss: critic_loss_value,
            actor_loss: actor_loss_value,
            alpha_loss: alpha_loss_value,
            alpha: self.alpha()?,
            mean_q,
        })
    }

    pub fn save_actor<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        self.actor_vars.save(path)
    }

    pub fn load_actor<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        self.actor_vars.load(path)
    }
}

fn adam_parameters(lr: f64) -> ParamsAdamW {
    ParamsAdamW {
        lr,
        weight_decay: 0.0,
        ..ParamsAdamW::default()
    }
}

fn epsilon_tensor(batch: usize, actions: usize, rng: &mut Rng, device: &Device) -> Result<Tensor> {
    let epsilon = (0..batch * actions)
        .map(|_| rng.normal() as f32)
        .collect::<Vec<_>>();
    Tensor::from_vec(epsilon, (batch, actions), device)
}

fn normalize_f64(values: &[f64], width: usize, norm: &ObsNorm) -> Vec<f32> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| normalized(*value, index % width, norm) as f32)
        .collect()
}

fn normalize_f32(values: &[f32], width: usize, norm: &ObsNorm) -> Vec<f32> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| normalized(*value as f64, index % width, norm) as f32)
        .collect()
}

fn normalized(value: f64, index: usize, norm: &ObsNorm) -> f64 {
    if norm.n < 2.0 || index >= norm.mean.len() {
        return value;
    }
    let deviation = (norm.m2[index] / norm.n).sqrt().max(1.0e-3);
    ((value - norm.mean[index]) / deviation).clamp(-8.0, 8.0)
}

fn deterministic_init(vars: &VarMap, seed: u64, device: &Device) -> Result<()> {
    let mut rng = Rng::new(seed);
    let data = vars.data().lock().expect("variable map poisoned");
    let mut variables = data
        .iter()
        .map(|(name, variable)| (name.clone(), variable.clone()))
        .collect::<Vec<_>>();
    drop(data);
    variables.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, variable) in variables {
        let shape = variable.shape().clone();
        let values = if shape.rank() == 1 {
            vec![0.0f32; shape.elem_count()]
        } else {
            let fan_in = shape.dims().last().copied().unwrap_or(1).max(1);
            let scale = (2.0 / fan_in as f64).sqrt();
            (0..shape.elem_count())
                .map(|_| (rng.normal() * scale) as f32)
                .collect()
        };
        variable.set(&Tensor::from_vec(values, shape, device)?)?;
        debug_assert!(!name.is_empty());
    }
    Ok(())
}

fn initialize_actor_output(vars: &VarMap, actions: usize, device: &Device) -> Result<()> {
    let data = vars.data().lock().expect("actor variable map poisoned");
    let weight = data
        .get("out.weight")
        .ok_or_else(|| candle_core::Error::Msg("actor is missing out.weight".into()))?;
    let bias = data
        .get("out.bias")
        .ok_or_else(|| candle_core::Error::Msg("actor is missing out.bias".into()))?;
    weight.set(&Tensor::zeros(weight.shape(), DType::F32, device)?)?;
    let mut values = vec![0.0f32; actions * 2];
    values[actions..].fill(-1.5);
    bias.set(&Tensor::from_vec(values, actions * 2, device)?)?;
    Ok(())
}

fn copy_parameters(source: &VarMap, target: &VarMap, tau: f64) -> Result<()> {
    let source = source
        .data()
        .lock()
        .expect("source variable map poisoned")
        .iter()
        .map(|(name, variable)| (name.clone(), variable.as_tensor().detach()))
        .collect::<Vec<_>>();
    let target = target.data().lock().expect("target variable map poisoned");
    for (name, source) in source {
        let Some(variable) = target.get(&name) else {
            candle_core::bail!("target network is missing parameter {name}")
        };
        let next = if tau >= 1.0 {
            source
        } else {
            variable
                .as_tensor()
                .affine(1.0 - tau, 0.0)?
                .add(&source.affine(tau, 0.0)?)?
        };
        variable.set(&next)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexapod_core::joint_rl::JointReplay;

    #[test]
    fn same_seed_initializes_the_same_actor_exactly() {
        let norm = ObsNorm::new(3);
        let mut rng_a = Rng::new(1);
        let mut rng_b = Rng::new(1);
        let a = SacAgent::new(3, 2, &Device::Cpu, SacConfig::default(), 44).expect("agent A");
        let b = SacAgent::new(3, 2, &Device::Cpu, SacConfig::default(), 44).expect("agent B");
        let action_a = a
            .action(&[0.2, -0.7, 1.1], &norm, false, &mut rng_a)
            .expect("action A");
        let action_b = b
            .action(&[0.2, -0.7, 1.1], &norm, false, &mut rng_b)
            .expect("action B");
        assert_eq!(action_a, action_b);
    }

    #[test]
    fn actor_starts_at_the_exact_standing_action() {
        let norm = ObsNorm::new(3);
        let mut rng = Rng::new(2);
        let agent = SacAgent::new(3, 2, &Device::Cpu, SacConfig::default(), 45).expect("agent");
        let action = agent
            .action(&[4.0, -2.0, 0.7], &norm, false, &mut rng)
            .expect("action");
        assert_eq!(action, vec![0.0; 2]);
    }

    #[test]
    fn actor_checkpoint_round_trips_exactly() {
        let norm = ObsNorm::new(3);
        let mut rng = Rng::new(5);
        let source = SacAgent::new(3, 2, &Device::Cpu, SacConfig::default(), 8).expect("source");
        let expected = source
            .action(&[0.1, 0.2, -0.3], &norm, false, &mut rng)
            .expect("source action");
        let path = std::env::temp_dir().join(format!(
            "hexapod-sac-actor-roundtrip-{}.safetensors",
            std::process::id()
        ));
        source.save_actor(&path).expect("save actor");
        let mut restored =
            SacAgent::new(3, 2, &Device::Cpu, SacConfig::default(), 99).expect("restored");
        restored.load_actor(&path).expect("load actor");
        let actual = restored
            .action(&[0.1, 0.2, -0.3], &norm, false, &mut rng)
            .expect("restored action");
        std::fs::remove_file(path).expect("remove temporary checkpoint");
        assert_eq!(actual, expected);
    }

    #[test]
    fn sac_update_is_finite_and_changes_the_policy_on_a_toy_continuous_task() {
        let observations = 3;
        let actions = 2;
        let mut agent = SacAgent::new(
            observations,
            actions,
            &Device::Cpu,
            SacConfig {
                hidden: 32,
                reward_scale: 1.0,
                ..SacConfig::default()
            },
            7,
        )
        .expect("agent");
        let norm = ObsNorm::new(observations);
        let mut rng = Rng::new(11);
        let before = agent
            .action(&[1.0, -0.5, 0.25], &norm, false, &mut rng)
            .expect("action before");
        let mut replay = JointReplay::new(512, observations, actions).expect("replay");
        for i in 0..512 {
            let state = [1.0, -0.5, 0.25];
            let a0 = -1.0 + 2.0 * (i % 32) as f64 / 31.0;
            let a1 = -1.0 + 2.0 * (i / 32) as f64 / 15.0;
            let reward = -((a0 - 0.6).powi(2) + (a1 + 0.35).powi(2));
            replay
                .push(&state, &[a0, a1], reward, &state, true, false)
                .expect("transition");
        }

        let mut last = None;
        for _ in 0..80 {
            let batch = replay.sample(128, &mut rng).expect("batch");
            let stats = agent.update(&batch, &norm, &mut rng).expect("update");
            assert!(stats.critic_loss.is_finite());
            assert!(stats.actor_loss.is_finite());
            assert!(stats.alpha.is_finite() && stats.alpha > 0.0);
            last = Some(stats);
        }
        let after = agent
            .action(&[1.0, -0.5, 0.25], &norm, false, &mut rng)
            .expect("action after");
        let delta = before
            .iter()
            .zip(&after)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        assert!(delta > 1.0e-3, "optimizer did not move actor: {delta}");
        let target = [0.6f32, -0.35];
        let before_error = before
            .iter()
            .zip(target)
            .map(|(action, target)| (action - target).powi(2))
            .sum::<f32>();
        let after_error = after
            .iter()
            .zip(target)
            .map(|(action, target)| (action - target).powi(2))
            .sum::<f32>();
        assert!(
            after_error < before_error,
            "toy policy did not move toward optimum: {before:?} ({before_error}) -> {after:?} ({after_error})"
        );
        assert!(last.expect("update stats").critic_loss < 20.0);
    }
}
