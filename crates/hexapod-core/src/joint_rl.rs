//! Joint-level locomotion policy: the network drives all eighteen motors.
//!
//! Everything else in this crate that walks does it by handing a *gait* some
//! parameters — cycle time, stride, duty — and letting hand-written inverse
//! kinematics turn a phase into foot targets. The policy there chooses how to
//! modulate a walk somebody already wrote.
//!
//! This module removes that scaffolding. The policy's output *is* the joint
//! command: three angles per leg, every control tick, straight into the
//! articulated plant's motors. There is no gait clock driving the legs, no IK
//! solving for a foothold, and no scripted trajectory anywhere — the machine
//! moves because Rapier's contact solver says the feet push the ground.
//!
//! What the policy is given is its own state and where it is trying to get to:
//! joint angles and rates, body attitude and velocity, which feet are down,
//! and the range and bearing to the next waypoint. Plus a free-running clock,
//! which is an *input* and not a trajectory: nothing tells the policy what to
//! do at a given phase, it only gets to know that time is periodic. Without it
//! a feedback-only controller has to build its own oscillator out of contact
//! transitions before it can take a second step, and that is a much harder
//! search than it needs to be.
//!
//! Actions are offsets from the standing pose rather than absolute angles, so
//! the all-zero policy stands still instead of collapsing. That matters for
//! [`Stage::Stand`]: iteration zero is already on its feet, and training makes
//! it *stay* there rather than discovering the floor from scratch.

use crate::dynamics::Physics;
use crate::math::{clamp, hypot2, inv_rot_y, Rng};
use crate::plant::ArticulatedPlant;
use crate::policy::{Policy, Preset};
use crate::robot::{Frame, MAX_LEGS, Q_LIMIT};
use crate::sim::DT;
use crate::terrain::{Course, Terrain};

/// Widest joint travel, radians, the policy may ask for in one direction away
/// from the standing pose. The mechanical limits still clamp on top of this;
/// this is about keeping the *search* in a sane region early on, when a policy
/// that swings a hip through its whole range just falls over and learns
/// nothing from having done so.
pub const ACT_RANGE: f64 = 0.40;

/// Fastest the *commanded* joint target may move, rad/s. This is not a limit
/// on the motors — they stay as strong as they were — it limits how fast the
/// policy is allowed to change its mind.
///
/// Without it the policy can step a hip a radian between one tick and the next,
/// and a motor strong enough to follow that on a 2 kg machine with 2 m legs
/// throws the whole robot. Measured, that is exactly what random search found
/// first: chassis 1.5-2.5 m in the air, 0.3 of six feet down on average, and
/// walking scored 0.454 at the seed falling to 0.017 in twenty-four iterations.
/// A real controller slews its setpoint; so does this one.
pub const MAX_JOINT_RATE: f64 = 6.0;

/// Weight on how much the command moved tick to tick. Joint-level policies
/// need this or they buzz: a controller that shakes at the control rate can
/// look like it is tracking a speed while being useless on a real machine.
pub const SMOOTH_COST: f64 = 0.08;

/// Physics ticks per policy decision. The plant still runs at its own rate;
/// the policy is asked once every this many ticks and its command is held in
/// between.
///
/// Two things come from this. The credit for an outcome is spread over half as
/// many decisions, which is the part that helps the search. And a held command
/// cannot chatter at the control rate, which is what a real controller running
/// at 50 Hz on 100 Hz physics also gets for free.
pub const DECIMATION: usize = 2;

/// Hidden units in the policy network. Kept small on purpose: ARS estimates
/// its gradient from a handful of random directions, and the variance of that
/// estimate grows with the number of parameters — 48 units meant 3666 weights
/// probed by 16 directions, which is not an estimate so much as a guess.
pub const N_HIDDEN: usize = 24;

/// Return spread below which an iteration is treated as carrying no gradient.
/// Scores are per-second and bounded by roughly 1.0, so this is a fraction of
/// a percent of the range.
const SPREAD_FLOOR: f64 = 1.0e-4;

/// Observation width for a six-legged machine. Recomputed per frame by
/// [`n_obs`]; this is the constant the buffers are sized against.
pub const MAX_JOINT_OBS: usize = 3 * MAX_LEGS  // joint angles
    + 3 * MAX_LEGS                            // joint rates
    + MAX_LEGS                                // foot contacts
    + 3                                       // body velocity, body frame
    + 3                                       // body angular velocity
    + 2                                       // pitch, roll
    + 1                                       // ride height above support
    + 2                                       // range, bearing to waypoint
    + 1                                       // commanded speed
    + 2 * MAX_LEGS                            // per-leg clock sin, cos
    + 1; // bias

pub const MAX_JOINT_ACT: usize = 3 * MAX_LEGS;

/// Observation width for `frame`.
pub fn n_obs(frame: Frame) -> usize {
    let n = frame.legs();
    // 3n angles + 3n rates + n contacts + 2n phase, then the body block.
    9 * n + 3 + 3 + 2 + 1 + 2 + 1 + 1
}

/// Action width for `frame`.
pub fn n_act(frame: Frame) -> usize {
    3 * frame.legs()
}

/// Parameter count of the network for `frame`.
pub fn n_theta(frame: Frame) -> usize {
    let (i, h, o) = (n_obs(frame), N_HIDDEN, n_act(frame));
    i * h + h + h * o + o
}

/// One tanh hidden layer. ARS never differentiates the policy, so the only
/// thing a network costs over a linear map is the forward pass — and the
/// nonlinearity is what lets one controller behave differently with a foot
/// down than with it in the air, which a linear map cannot express.
#[derive(Clone, Debug)]
pub struct JointPolicy {
    pub frame: Frame,
    pub theta: Vec<f64>,
    /// Running observation statistics. ARS-V2 normalises inputs, and with
    /// joint angles in radians next to velocities in m/s the scales differ by
    /// enough that without it the first layer is effectively ignoring half its
    /// inputs.
    pub norm: ObsNorm,
}

impl JointPolicy {
    /// Zero output weights, small random input weights. The zero last layer is
    /// deliberate: the policy starts by emitting exactly the standing pose, so
    /// the first rollout of [`Stage::Stand`] scores whatever standing scores
    /// and every improvement from there is measured against a machine that was
    /// at least upright.
    pub fn seeded(frame: Frame, seed: u64) -> Self {
        let (i, h) = (n_obs(frame), N_HIDDEN);
        let mut rng = Rng::new(seed);
        let mut theta = vec![0.0; n_theta(frame)];
        let scale = (1.0 / i as f64).sqrt();
        for w in theta[..i * h].iter_mut() {
            *w = rng.normal() * scale;
        }
        // hidden bias, output weights and output bias all stay zero
        JointPolicy {
            frame,
            theta,
            norm: ObsNorm::new(n_obs(frame)),
        }
    }

    /// Joint offsets, in radians, for one observation.
    pub fn act(&self, obs: &[f64], out: &mut [f64]) {
        let (i, h, o) = (n_obs(self.frame), N_HIDDEN, n_act(self.frame));
        let (w1, rest) = self.theta.split_at(i * h);
        let (b1, rest) = rest.split_at(h);
        let (w2, b2) = rest.split_at(h * o);

        let mut hid = [0.0f64; N_HIDDEN];
        for (j, hv) in hid.iter_mut().enumerate().take(h) {
            let row = &w1[j * i..j * i + i];
            let mut acc = b1[j];
            for (w, x) in row.iter().zip(obs.iter().take(i)) {
                acc += w * x;
            }
            *hv = acc.tanh();
        }
        for k in 0..o {
            let row = &w2[k * h..k * h + h];
            let mut acc = b2[k];
            for (w, x) in row.iter().zip(hid.iter().take(h)) {
                acc += w * x;
            }
            out[k] = acc.tanh() * ACT_RANGE;
        }
    }
}

/// Welford mean/variance over observations, frozen once training stops so a
/// replay reproduces the run it was trained in.
#[derive(Clone, Debug)]
pub struct ObsNorm {
    pub n: f64,
    pub mean: Vec<f64>,
    pub m2: Vec<f64>,
    pub frozen: bool,
}

impl ObsNorm {
    pub fn new(width: usize) -> Self {
        ObsNorm {
            n: 0.0,
            mean: vec![0.0; width],
            m2: vec![1.0; width],
            frozen: false,
        }
    }

    pub fn observe(&mut self, obs: &[f64]) {
        if self.frozen {
            return;
        }
        self.n += 1.0;
        for (i, x) in obs.iter().enumerate().take(self.mean.len()) {
            let d = x - self.mean[i];
            self.mean[i] += d / self.n;
            self.m2[i] += d * (x - self.mean[i]);
        }
    }

    pub fn apply(&self, obs: &mut [f64]) {
        if self.n < 2.0 {
            return;
        }
        for (i, x) in obs.iter_mut().enumerate().take(self.mean.len()) {
            let sd = (self.m2[i] / self.n).sqrt().max(1e-3);
            *x = clamp((*x - self.mean[i]) / sd, -8.0, 8.0);
        }
    }
}

/// Curriculum stages, in the order they are trained.
///
/// Each one adds a single new difficulty rather than a new course, so a
/// failure says which capability is missing. Promotion is on score, not on a
/// fixed iteration count: a stage that has not been solved is not one to build
/// on, and spending the whole budget on the first hard stage is the correct
/// outcome — it is information, where silently moving on is not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Hold a standing pose on flat ground, no command. Learns to carry its
    /// own weight before anything else.
    Stand,
    /// Track a slow forward speed on flat ground.
    WalkFlat,
    /// The same command, faster.
    RunFlat,
    /// Broken ground: steps and rubble, still no gaps.
    Rough,
    /// Gaps a stride wide, which can be stepped rather than jumped.
    Gaps,
    /// Parkour: trenches wider than a stride, so they have to be cleared.
    Jump,
    /// Everything, sampled.
    Mixed,
}

pub const STAGES: [Stage; 7] = [
    Stage::Stand,
    Stage::WalkFlat,
    Stage::RunFlat,
    Stage::Rough,
    Stage::Gaps,
    Stage::Jump,
    Stage::Mixed,
];

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Stage::Stand => "STAND",
            Stage::WalkFlat => "WALK-FLAT",
            Stage::RunFlat => "RUN-FLAT",
            Stage::Rough => "ROUGH",
            Stage::Gaps => "GAPS",
            Stage::Jump => "JUMP",
            Stage::Mixed => "MIXED",
        }
    }

    /// Courses this stage samples from.
    pub fn courses(self) -> &'static [Course] {
        match self {
            Stage::Stand | Stage::WalkFlat | Stage::RunFlat => &[Course::Flat],
            Stage::Rough => &[Course::Steps, Course::Rubble],
            Stage::Gaps => &[Course::Gaps],
            Stage::Jump => &[Course::Jump, Course::Chasm],
            Stage::Mixed => &[
                Course::Flat,
                Course::Steps,
                Course::Rubble,
                Course::Gaps,
                Course::Mixed,
                Course::Jump,
            ],
        }
    }

    /// Commanded forward speed, m/s. Standing is a zero command, and the
    /// parkour stages need a run-up: a trench wider than a stride is not
    /// crossed at a walk however good the controller is.
    pub fn speed(self) -> f64 {
        match self {
            Stage::Stand => 0.0,
            Stage::WalkFlat => 0.8,
            Stage::RunFlat => 2.0,
            Stage::Rough => 1.5,
            Stage::Gaps => 2.0,
            Stage::Jump => 4.0,
            Stage::Mixed => 2.5,
        }
    }

    /// Rollout length, seconds.
    pub fn horizon(self) -> f64 {
        match self {
            Stage::Stand => 2.0,
            Stage::WalkFlat | Stage::RunFlat => 4.0,
            _ => 8.0,
        }
    }

    /// Mean score over the stage's courses at which the next stage opens.
    /// These are per-second rates, so they do not move when the horizon does.
    pub fn promote_at(self) -> f64 {
        match self {
            // Standing still scores ~0.16 under the gated reward, so these
            // are thresholds a policy has to actually move to reach.
            Stage::Stand => 0.80,
            Stage::WalkFlat => 0.45,
            Stage::RunFlat => 0.40,
            Stage::Rough => 0.32,
            Stage::Gaps => 0.28,
            Stage::Jump => 0.22,
            Stage::Mixed => f64::INFINITY,
        }
    }
}

/// What one rollout produced.
#[derive(Clone, Copy, Debug, Default)]
pub struct JointRollout {
    /// Reward per simulated second, so horizons stay comparable.
    pub score: f64,
    /// Ground covered along the course, metres.
    pub distance: f64,
    /// Simulated seconds survived.
    pub secs: f64,
    pub fell: bool,
    /// Mean number of feet on the ground.
    pub support: f64,
    /// Peak height the chassis reached above its standing height.
    pub air: f64,
}

/// The standing joint pose: what an all-zero action commands.
pub fn stand_pose(frame: Frame, phys: &Physics, terrain: &Terrain) -> [[f64; 3]; MAX_LEGS] {
    // The plant already computes this when it spawns a machine standing, so
    // taking it from there keeps the two definitions from drifting apart.
    let gait = Policy::seeded(Preset::default_for(frame), frame).gait();
    let plant = ArticulatedPlant::standing(frame, &gait, phys, terrain);
    plant.leg_q_all()
}

/// Run one episode and score it.
pub fn rollout(
    policy: &JointPolicy,
    phys: &Physics,
    terrain: &Terrain,
    stage: Stage,
    mut norm_sink: Option<&mut ObsNorm>,
) -> JointRollout {
    let frame = policy.frame;
    let n = frame.legs();
    let no = n_obs(frame);
    let na = n_act(frame);

    let gait = Policy::seeded(Preset::default_for(frame), frame).gait();
    // Phase reference only — the same split the hand-written tripod uses for
    // *which* legs swing together, with nothing about what they should do.
    let phase_off = gait.offsets;
    let mut plant = ArticulatedPlant::standing(frame, &gait, phys, terrain);
    let neutral = plant.leg_q_all();
    let substeps = plant.substeps.max(1);
    // One step so the broad-phase BVH exists before anything queries it.
    plant.step(DT);

    let (start, _, _, _) = plant.chassis_pose();
    let stand_y = start[1];
    let cmd = stage.speed();

    let mut obs = vec![0.0; no];
    let mut act = vec![0.0; na];
    let mut last_q = plant.leg_q_all();
    // The command is stateful: it slews from wherever it is, starting from
    // standing, so the first tick cannot snap the legs anywhere.
    let mut q_cmd = neutral;
    let mut total = 0.0;
    let mut support_sum = 0.0;
    let mut air = 0.0f64;
    let mut steps = 0usize;
    let mut fell = false;
    let mut clock = 0.0f64;
    // Contact averaged over a window, not read off one tick. See `reward`.
    let mut duty = frame.legs() as f64;
    let ticks = (stage.horizon() / DT) as usize;

    for tick in 0..ticks {
        let (pos, yaw, pitch, roll) = plant.chassis_pose();
        let vel = plant.chassis_vel();
        let avel = plant.chassis_angvel();
        let q = plant.leg_q_all();
        let contact = plant.foot_contacts();

        // A tilt past this is not recoverable by a position-controlled leg and
        // every later tick is scored on a machine that is already lost.
        if pitch.abs() > 1.0 || roll.abs() > 1.0 || plant.chassis_dead(vel) {
            fell = true;
            break;
        }

        let support = plant.support_under(pos[0], pos[2], pos[1] + 4.0).unwrap_or(0.0);
        let ride = pos[1] - support;
        let body_v = inv_rot_y([vel[0], vel[1], vel[2]], yaw);
        let (range, bearing) = waypoint(terrain, pos, yaw);

        let mut w = 0usize;
        for i in 0..n {
            for c in 0..3 {
                obs[w] = q[i][c] - neutral[i][c];
                w += 1;
            }
        }
        for i in 0..n {
            for c in 0..3 {
                obs[w] = (q[i][c] - last_q[i][c]) / DT * 0.05;
                w += 1;
            }
        }
        for i in 0..n {
            obs[w] = if contact[i] { 1.0 } else { 0.0 };
            w += 1;
        }
        obs[w] = body_v[0];
        obs[w + 1] = body_v[1];
        obs[w + 2] = body_v[2];
        obs[w + 3] = avel[0];
        obs[w + 4] = avel[1];
        obs[w + 5] = avel[2];
        obs[w + 6] = pitch;
        obs[w + 7] = roll;
        obs[w + 8] = ride - stand_y;
        obs[w + 9] = range;
        obs[w + 10] = bearing;
        obs[w + 11] = cmd;
        for i in 0..n {
            let ph = (clock + phase_off[i]) * std::f64::consts::TAU;
            obs[w + 12 + i * 2] = ph.sin();
            obs[w + 13 + i * 2] = ph.cos();
        }
        obs[w + 12 + n * 2] = 1.0;

        if tick % DECIMATION == 0 {
            if let Some(sink) = norm_sink.as_deref_mut() {
                sink.observe(&obs);
            }
            policy.norm.apply(&mut obs);
            policy.act(&obs, &mut act);
        }

        let mut jerk = 0.0;
        for i in 0..n {
            for c in 0..3 {
                let (lo, hi) = Q_LIMIT[c];
                let want = clamp(neutral[i][c] + act[i * 3 + c], lo, hi);
                let slew = MAX_JOINT_RATE * DT;
                let moved = clamp(want - q_cmd[i][c], -slew, slew);
                jerk += moved.abs();
                q_cmd[i][c] += moved;
            }
        }

        last_q = q;
        plant.drive(&q_cmd, phys, DT);
        for _ in 0..substeps {
            plant.step(DT / substeps as f64);
        }

        let down = contact.iter().take(n).filter(|c| **c).count();
        // ~0.3 s time constant: long enough to span the flight phase of a
        // dynamic gait, short enough that sustained flight still reads as
        // sustained flight.
        duty += (down as f64 - duty) * (DT / 0.30);
        support_sum += down as f64;
        air = air.max(pos[1] - stand_y);
        let tick = reward(stage, cmd, &body_v, pitch, roll, ride, stand_y, down, duty, n, bearing);
        // Normalised by the most the command could have moved this tick, so
        // the cost does not change meaning when the slew limit or leg count
        // does.
        let churn = jerk / (MAX_JOINT_RATE * DT * 3.0 * n as f64);
        total += (tick - SMOOTH_COST * churn).max(0.0);
        steps += 1;
        clock += DT / 0.5;

        if terrain.waypoints.is_empty() && pos[2] > crate::terrain::Z_MAX - 2.0 {
            break;
        }
    }

    let (end, _, _, _) = plant.chassis_pose();
    let secs = steps as f64 * DT;
    // A fall is scored on the time it survived, not averaged over it: falling
    // at one second and standing for three must not come out the same.
    let denom = stage.horizon().max(1e-6);
    JointRollout {
        score: total * DT / denom,
        distance: end[2] - start[2],
        secs,
        fell,
        support: if steps == 0 {
            0.0
        } else {
            support_sum / steps as f64
        },
        air,
    }
}

/// Range and bearing to the next waypoint ahead, in the body frame.
fn waypoint(terrain: &Terrain, pos: [f64; 3], yaw: f64) -> (f64, f64) {
    let next = terrain
        .waypoints
        .iter()
        .find(|w| w[1] > pos[2] + 0.2)
        .copied();
    match next {
        Some(w) => {
            let d = [w[0] - pos[0], 0.0, w[1] - pos[2]];
            let b = inv_rot_y(d, yaw);
            (hypot2(d[0], d[2]).min(30.0) * 0.1, b[0].atan2(b[2].max(1e-6)))
        }
        // No route: straight down the corridor is the goal.
        None => {
            let b = inv_rot_y([-pos[0], 0.0, 4.0], yaw);
            (0.4, b[0].atan2(b[2].max(1e-6)))
        }
    }
}

/// Per-tick reward. Bounded above by roughly 1.0 so a stage's score is
/// readable as "what fraction of the best possible did it get".
#[allow(clippy::too_many_arguments)]
fn reward(
    stage: Stage,
    cmd: f64,
    body_v: &[f64; 3],
    pitch: f64,
    roll: f64,
    ride: f64,
    stand_y: f64,
    down: usize,
    // Feet on the ground, averaged over a ~0.3 s window.
    duty: f64,
    legs: usize,
    bearing: f64,
) -> f64 {
    // Upright and at ride height. This is the whole objective for `Stand` and
    // a precondition everywhere else: a controller that scores well on speed
    // while dragging its belly is not one to build the next stage on.
    let level = (-(pitch * pitch + roll * roll) / 0.08).exp();
    let height = (-((ride - stand_y) / 0.18).powi(2)).exp();

    if stage == Stage::Stand {
        let still = (-(body_v[0] * body_v[0] + body_v[2] * body_v[2]) / 0.05).exp();
        let feet = down as f64 / legs as f64;
        return 0.40 * level + 0.30 * height + 0.20 * still + 0.10 * feet;
    }

    // Sustained flight earns nothing — these motors can throw a 2 kg machine
    // on 2 m legs clean into the air, and unchecked that is the first thing
    // random search finds. But the test is the *windowed* contact, not this
    // tick's. Judging it per tick outlaws the flight phase every dynamic gait
    // has, and it did: a policy that had learned to cover 3.5 m in 4 s — 0.87
    // m/s against a 0.8 m/s command — scored 0.07, because its brief contacts
    // read as "airborne" on most individual ticks.
    if duty < 0.15 {
        return 0.0;
    }

    // Speed along the heading. Deliberately *not* the gait-level trainer's
    // Gaussian: that is a scoring function, and this has to be a guide.
    //
    // A Gaussian centred on the command is flat where training starts. At rest
    // against a 0.8 m/s command it reads exp(-(0.8/0.37)^2) = 0.009, and its
    // slope there is nearly zero, so nothing pulls the machine into moving at
    // all — measured, walking sat at 0.20 for iterations while lifting legs
    // and covering no ground. Rising linearly to the command gives a constant
    // gradient from a standstill; above the command it falls off, so this is
    // still speed *tracking* and not a prize for going as fast as possible.
    let along = body_v[2];
    let width = 0.25 + 0.15 * cmd.abs();
    let track = if along <= 0.0 {
        0.0
    } else if along < cmd {
        along / cmd.max(1e-6)
    } else {
        (-((along - cmd) / width).powi(2)).exp()
    };
    let aim = (-(bearing * bearing) / 0.5).exp();

    // Feet have to leave the ground, but not all of them and not five of six.
    // A flat "some foot is up" bonus paid the same for a tripod as for hopping
    // on one leg, and hopping is easier to find — mean support fell to 1.2 of
    // six. Peaking at half the legs is what a tripod looks like, and it is the
    // only support pattern that both moves and stays statically safe.
    let half = legs as f64 * 0.5;
    let gait = (-((duty - half) / 1.5).powi(2)).exp();

    // Posture is a *gate*, not a term to be paid for.
    //
    // Additively, standing perfectly still collected 0.20 for being level plus
    // 0.15 for ride height plus 0.10 for facing the right way — 0.454 of a
    // possible 1.0 for doing nothing. Worse, the first move any policy makes
    // costs some of that attitude reward before it earns any tracking reward,
    // so the search had to cross a valley to start walking and mostly did not.
    //
    // As a multiplier, holding attitude is worth nothing on its own and losing
    // it forfeits everything. Standing still now scores about 0.16, walking
    // well still approaches 1.0, and the path between them runs downhill.
    //
    // Support is part of the gate too, not a bonus. As a 0.10 term the machine
    // kept trading it away: mean support fell to 1.3 feet of six, because
    // hopping on one leg still collected most of the speed reward. A hexapod
    // on fewer than two feet is not walking, whatever its velocity says, so
    // below that the tick is worth almost nothing.
    let enough_feet = ((duty - 0.35) / 1.4).clamp(0.0, 1.0);
    let posture = level * height * enough_feet;
    (0.75 * track + 0.15 * aim + 0.10 * gait) * posture
}

/// ARS configuration for the joint-level trainer.
#[derive(Clone, Copy, Debug)]
pub struct JointCfg {
    pub dirs: usize,
    pub top: usize,
    pub alpha: f64,
    pub sigma: f64,
    /// Courses sampled per perturbation, averaged. More than one keeps a
    /// direction from winning on a lucky seed.
    pub scenarios: usize,
    pub workers: usize,
}

impl Default for JointCfg {
    fn default() -> Self {
        JointCfg {
            dirs: 16,
            top: 6,
            // Small. The network's input weights are scaled 1/sqrt(inputs),
            // about 0.14, and ARS divides its step by the spread of the
            // returns — which is small precisely when the top directions
            // agree. At alpha 0.02 that combination moved weights by ~30% of
            // their own scale per iteration and walking diverged from 0.454 to
            // 0.033 in twelve.
            // Swept, 10 iterations each on WALK-FLAT. 0.0005 and 0.001 climb
            // steadily to ~0.16 with all six feet planted and no ground
            // covered — refining the stand rather than leaving it. 0.002
            // reaches 0.208. 0.005 escapes the standing basin by iteration 4
            // (support 6.00 -> 2.87) and reaches 0.342 while actually
            // travelling; 0.010 jumps out immediately and then comes apart.
            alpha: 0.005,
            // Measured, not guessed. Exploration noise has to be big enough to
            // produce visible leg motion and small enough that the machine
            // does not simply fall over — past 0.05 the mean feet-on-ground
            // drops from 0.76 to 0.36, every perturbation scores about zero,
            // and the spread of returns that ARS steers by collapses from
            // 0.113 to 0.024. Past 0.30 the output tanh saturates at ~23
            // degrees of joint travel and more noise buys nothing at all.
            sigma: 0.05,
            scenarios: 2,
            workers: 0,
        }
    }
}

/// Mean score of `policy` on `stage`, over a fixed set of seeds.
pub fn evaluate(policy: &JointPolicy, phys: &Physics, stage: Stage, seeds: &[u64]) -> JointRollout {
    let courses = stage.courses();
    let mut acc = JointRollout::default();
    let mut n = 0.0;
    for &seed in seeds {
        for &course in courses {
            let terrain = Terrain::new(course, seed);
            let r = rollout(policy, phys, &terrain, stage, None);
            acc.score += r.score;
            acc.distance += r.distance;
            acc.secs += r.secs;
            acc.support += r.support;
            acc.air = acc.air.max(r.air);
            acc.fell |= r.fell;
            n += 1.0;
        }
    }
    if n > 0.0 {
        acc.score /= n;
        acc.distance /= n;
        acc.secs /= n;
        acc.support /= n;
    }
    acc
}

/// One ARS iteration on `stage`. Returns the mean score of the perturbed
/// rollouts, which is what a training curve should plot: the evaluation score
/// is measured separately and on fixed seeds.
pub fn iterate(
    policy: &mut JointPolicy,
    phys: &Physics,
    stage: Stage,
    cfg: &JointCfg,
    rng: &mut Rng,
    iter: usize,
) -> f64 {
    let n = n_theta(policy.frame);
    let dirs = cfg.dirs.max(1);
    let mut deltas = vec![0.0; dirs * n];
    for d in deltas.iter_mut() {
        *d = rng.normal();
    }

    // The terrain a direction runs on is shared by both signs of the finite
    // difference. Give the two sides different ground and the difference
    // measures the course draw, not the perturbation.
    let courses = stage.courses();
    let batch = cfg.scenarios.max(1);
    let plan: Vec<Vec<Terrain>> = (0..dirs)
        .map(|k| {
            (0..batch)
                .map(|b| {
                    let slot = iter * dirs * batch + k * batch + b;
                    let course = courses[slot % courses.len()];
                    // Rotate seeds so a stage does not overfit one layout.
                    Terrain::new(course, 1 + (slot / courses.len()) as u64 % 16)
                })
                .collect()
        })
        .collect();

    let workers = if cfg.workers == 0 {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    } else {
        cfg.workers
    }
    .max(1)
    .min(dirs);

    // (direction, plus score, minus score, pooled observation stats)
    let mut results: Vec<(usize, f64, f64, ObsNorm)> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let deltas = &deltas;
            let plan = &plan;
            let base_policy = &*policy;
            handles.push(scope.spawn(move || {
                let mut out = Vec::new();
                for k in (worker..dirs).step_by(workers) {
                    let mut norm = ObsNorm::new(n_obs(base_policy.frame));
                    let mut side = |sign: f64| {
                        let mut p = base_policy.clone();
                        for (j, w) in p.theta.iter_mut().enumerate() {
                            *w += sign * cfg.sigma * deltas[k * n + j];
                        }
                        let mut acc = 0.0;
                        for terrain in &plan[k] {
                            acc += rollout(&p, phys, terrain, stage, Some(&mut norm)).score;
                        }
                        acc / plan[k].len() as f64
                    };
                    let plus = side(1.0);
                    let minus = side(-1.0);
                    out.push((k, plus, minus, norm));
                }
                out
            }));
        }
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("joint-RL rollout worker panicked"))
            .collect()
    });

    // Fold the pooled observation statistics back in before the weight step,
    // so the next iteration normalises with what this one actually saw. Skipped
    // once the scaling is frozen — see `train_curriculum`, which fixes it up
    // front precisely so the gradient and the evaluation agree.
    for (_, _, _, norm) in results.iter().filter(|_| !policy.norm.frozen) {
        if norm.n < 1.0 {
            continue;
        }
        let total = policy.norm.n + norm.n;
        for i in 0..policy.norm.mean.len() {
            let d = norm.mean[i] - policy.norm.mean[i];
            policy.norm.mean[i] += d * norm.n / total;
            policy.norm.m2[i] += norm.m2[i] + d * d * policy.norm.n * norm.n / total;
        }
        policy.norm.n = total;
    }

    // ARS-V1t: rank directions by their better side, keep the top slice, and
    // scale the step by the spread of the returns that went into it.
    results.sort_by(|a, b| b.1.max(b.2).total_cmp(&a.1.max(a.2)));
    let top = cfg.top.clamp(1, results.len());
    let used = &results[..top];
    let mean = used.iter().flat_map(|r| [r.1, r.2]).sum::<f64>() / (2 * top) as f64;
    let var = used
        .iter()
        .flat_map(|r| [r.1, r.2])
        .map(|s| (s - mean) * (s - mean))
        .sum::<f64>()
        / (2 * top) as f64;
    let sd = var.sqrt();

    // ARS scales its step by the spread of the returns, which blows up when
    // there is no spread. A saturated stage is exactly that case: every
    // direction scores the same, `plus - minus` is small but `sd` is smaller,
    // and the ratio takes a huge step for no reason — measured, this turned a
    // policy that stood perfectly (1.000) into one that fell (0.236) in six
    // iterations. No spread means no information, so take no step.
    if sd > SPREAD_FLOOR {
        for (k, plus, minus, _) in used {
            let step = cfg.alpha / (top as f64 * sd) * (plus - minus);
            for j in 0..n {
                policy.theta[j] += step * deltas[k * n + j];
            }
        }
    }

    results.iter().flat_map(|r| [r.1, r.2]).sum::<f64>() / (2 * results.len()) as f64
}

/// Where a curriculum run has got to.
#[derive(Clone, Debug)]
pub struct Progress {
    pub stage: Stage,
    pub iter: usize,
    /// Evaluation score of the current policy on the current stage.
    pub score: f64,
    pub eval: JointRollout,
    /// True on the iteration a stage was cleared.
    pub promoted: bool,
}

/// Train through the curriculum, stopping when the budget runs out or every
/// stage is cleared. `on_iter` is called after each iteration so a CLI can
/// print a curve without this module knowing about stdout.
///
/// Each stage is *evaluated before it is trained*. That is not a
/// micro-optimisation: with motors this strong the standing stage is already
/// saturated at the seed, and a saturated objective is a maximum that ARS can
/// only wander away from. Training it for even a few iterations measurably
/// wrecks a working policy — 1.000 down to 0.236 over six — because the step
/// is scaled by the spread of returns, and when the only available spread is
/// "some perturbations fall over", every direction points downhill. Checking
/// first means a stage that is already solved costs nothing and damages
/// nothing.
pub fn train_curriculum(
    frame: Frame,
    phys: &Physics,
    cfg: &JointCfg,
    budget: usize,
    seed: u64,
    mut on_iter: impl FnMut(&Progress, &JointPolicy),
) -> JointPolicy {
    let mut policy = JointPolicy::seeded(frame, seed);
    let mut rng = Rng::new(seed ^ 0x9e37_79b9_7f4a_7c15);
    let eval_seeds = [101u64, 202, 303];
    let mut iter = 0usize;

    // Fix the observation scaling up front, then leave it alone.
    //
    // Updating it every iteration makes the policy a moving target: the
    // rollouts that produce the gradient run under one normaliser and the
    // evaluation that judges the result runs under the next, so the two
    // measure different controllers. It is worse on the first iteration, where
    // `apply` is a no-op until there are two samples and normalisation
    // switches on abruptly afterwards. Symptom: the evaluation score walked
    // *downhill* below the untrained policy while training thought it was
    // improving. A fixed scaling is slightly stale by the end of a stage and
    // consistent throughout, which is the better trade.
    {
        let mut warm = ObsNorm::new(n_obs(frame));
        let probe = STAGES.get(1).copied().unwrap_or(Stage::WalkFlat);
        for &course in probe.courses() {
            for seed in [1u64, 2, 3] {
                let terrain = Terrain::new(course, seed);
                let mut p = policy.clone();
                // Perturbed, so the statistics cover moving as well as
                // standing — a normaliser fitted only to a motionless machine
                // gives every joint-rate input a near-zero variance and then
                // amplifies its noise.
                for w in p.theta.iter_mut() {
                    *w += 0.05 * rng.normal();
                }
                rollout(&p, phys, &terrain, probe, Some(&mut warm));
            }
        }
        warm.frozen = true;
        policy.norm = warm;
    }

    for &stage in STAGES.iter() {
        loop {
            let eval = evaluate(&policy, phys, stage, &eval_seeds);
            let promoted = eval.score >= stage.promote_at();
            on_iter(
                &Progress {
                    stage,
                    iter,
                    score: eval.score,
                    eval,
                    promoted,
                },
                &policy,
            );
            if promoted || iter >= budget {
                break;
            }
            iterate(&mut policy, phys, stage, cfg, &mut rng, iter);
            iter += 1;
        }
        if iter >= budget {
            break;
        }
    }

    policy.norm.frozen = true;
    policy
}


/// Text format for a joint-level checkpoint.
///
/// Deliberately not the `hexapod-policy-v1` format: that one describes a gait
/// controller — feedback matrix, per-leg phase offsets — and this policy has
/// none of those. Sharing the magic string would let the page load one where
/// the other is meant and silently drive the wrong thing.
pub const JOINT_MAGIC: &str = "hexapod-joint-v1";

pub fn to_text(policy: &JointPolicy) -> String {
    let values = |xs: &[f64]| {
        xs.iter()
            .map(|v| format!("{v:.17e}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!(
        "{JOINT_MAGIC}\nframe={}\nhidden={}\nact_range={:.17e}\nnorm_n={:.17e}\nnorm_mean={}\nnorm_m2={}\ntheta={}\n",
        policy.frame.legs(),
        N_HIDDEN,
        ACT_RANGE,
        policy.norm.n,
        values(&policy.norm.mean),
        values(&policy.norm.m2),
        values(&policy.theta),
    )
}

pub fn from_text(text: &str) -> Result<JointPolicy, String> {
    let mut lines = text.lines();
    let magic = lines.next().unwrap_or("").trim();
    if magic != JOINT_MAGIC {
        return Err(format!(
            "not a joint-level checkpoint: expected {JOINT_MAGIC}, found {magic:?}"
        ));
    }
    let mut fields: Vec<(String, String)> = Vec::new();
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            return Err(format!("malformed line: {line:?}"));
        };
        fields.push((k.trim().to_string(), v.trim().to_string()));
    }
    let get = |key: &str| -> Result<String, String> {
        fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
            .ok_or_else(|| format!("checkpoint is missing {key}"))
    };
    let nums = |v: &str| -> Result<Vec<f64>, String> {
        v.split_whitespace()
            .map(|t| t.parse::<f64>().map_err(|e| format!("bad number {t:?}: {e}")))
            .collect()
    };

    let legs: usize = get("frame")?
        .parse()
        .map_err(|e| format!("bad frame: {e}"))?;
    if !(crate::robot::MIN_LEGS..=MAX_LEGS).contains(&legs) {
        return Err(format!("checkpoint is for {legs} legs"));
    }
    let frame = Frame::new(legs);
    let hidden: usize = get("hidden")?
        .parse()
        .map_err(|e| format!("bad hidden: {e}"))?;
    if hidden != N_HIDDEN {
        return Err(format!(
            "checkpoint has {hidden} hidden units, this build has {N_HIDDEN}"
        ));
    }
    let theta = nums(&get("theta")?)?;
    if theta.len() != n_theta(frame) {
        return Err(format!(
            "checkpoint has {} weights, a {legs}-leg policy needs {}",
            theta.len(),
            n_theta(frame)
        ));
    }
    let mean = nums(&get("norm_mean")?)?;
    let m2 = nums(&get("norm_m2")?)?;
    if mean.len() != n_obs(frame) || m2.len() != n_obs(frame) {
        return Err(format!(
            "checkpoint normaliser is {} wide, this policy observes {}",
            mean.len(),
            n_obs(frame)
        ));
    }
    Ok(JointPolicy {
        frame,
        theta,
        norm: ObsNorm {
            n: get("norm_n")?.parse().map_err(|e| format!("bad norm_n: {e}"))?,
            mean,
            m2,
            frozen: true,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An all-zero action has to command exactly the pose the plant spawned
    /// standing in. If it does not, stage one starts from a machine that is
    /// already falling and nothing after it means very much.
    #[test]
    fn a_zero_action_commands_the_standing_pose() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let neutral = stand_pose(frame, &phys, &terrain);

        let policy = JointPolicy::seeded(frame, 1);
        let mut act = vec![0.0; n_act(frame)];
        let obs = vec![0.0; n_obs(frame)];
        policy.act(&obs, &mut act);
        for a in act.iter() {
            assert!(a.abs() < 1e-12, "seeded policy already moves a joint: {a}");
        }
        for i in 0..frame.legs() {
            for c in 0..3 {
                let (lo, hi) = Q_LIMIT[c];
                assert!(
                    neutral[i][c] >= lo - 1e-9 && neutral[i][c] <= hi + 1e-9,
                    "standing pose is outside the joint travel: leg {i} joint {c}"
                );
            }
        }
    }

    /// The seeded policy holds the machine up on flat ground. This is the
    /// baseline every later stage builds on, and it is also the check that the
    /// plant's motors can carry the chassis at all.
    #[test]
    fn the_seeded_policy_stands_without_falling() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let policy = JointPolicy::seeded(frame, 1);
        let r = rollout(&policy, &phys, &terrain, Stage::Stand, None);
        assert!(!r.fell, "the machine fell over standing still");
        assert!(
            r.secs > Stage::Stand.horizon() - 0.05,
            "only survived {:.2}s of {:.2}",
            r.secs,
            Stage::Stand.horizon()
        );
        assert!(
            r.support > 5.0,
            "feet left the ground while standing: mean support {:.2}",
            r.support
        );
        assert!(r.score > 0.4, "standing scored {:.3}", r.score);
    }

    /// The observation vector has to be exactly filled — a width mismatch
    /// silently shifts every input the network reads.
    #[test]
    fn the_observation_vector_is_the_width_it_claims() {
        for legs in [4usize, 6, 10] {
            let frame = Frame::new(legs);
            let expect = 9 * legs + 12 + 1;
            assert_eq!(n_obs(frame), expect, "{legs} legs");
            assert_eq!(n_act(frame), 3 * legs);
            assert!(n_obs(frame) <= MAX_JOINT_OBS);
            assert!(n_act(frame) <= MAX_JOINT_ACT);
        }
    }

    /// With motors this strong, standing is free: the seeded policy already
    /// saturates the stage's reward. So the curriculum has to promote past it
    /// without training on it — a saturated objective is a maximum, and ARS
    /// stepping away from one took a perfect 1.000 down to 0.236 in six
    /// iterations when the stage was trained before being checked.
    #[test]
    fn a_solved_stage_is_promoted_without_being_trained_on() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let cfg = JointCfg {
            dirs: 4,
            top: 2,
            scenarios: 1,
            ..JointCfg::default()
        };
        let seeded = JointPolicy::seeded(frame, 7);
        let before = evaluate(&seeded, &phys, Stage::Stand, &[101]).score;
        assert!(
            before >= Stage::Stand.promote_at(),
            "standing is not already solved: {before:.3}"
        );

        // One iteration of budget: enough to promote through STAND, not enough
        // to get anywhere on the stage after it. Record the weights at every
        // stage check, so we can see whether passing through STAND cost any.
        let mut seen: Vec<(Stage, bool, Vec<f64>)> = Vec::new();
        train_curriculum(frame, &phys, &cfg, 1, 7, |p, pol| {
            seen.push((p.stage, p.promoted, pol.theta.clone()));
        });

        let stand: Vec<_> = seen.iter().filter(|(s, _, _)| *s == Stage::Stand).collect();
        assert_eq!(
            stand.len(),
            1,
            "STAND was visited {} times; a solved stage should be checked once",
            stand.len()
        );
        assert!(
            stand[0].1,
            "STAND was not promoted despite scoring {before:.3}"
        );
        assert!(
            seen.iter().any(|(s, _, _)| *s == Stage::WalkFlat),
            "the curriculum never reached the stage after standing"
        );
        // Untouched: not one weight was spent on a stage that was already won.
        // (Later stages do move them, and should — a policy learning to walk
        // is *supposed* to stop scoring well at holding perfectly still.)
        assert_eq!(
            stand[0].2, seeded.theta,
            "passing through a solved stage still changed the policy"
        );
    }

    /// The stage after standing is the first one that has to be learned, so it
    /// is the one where a spread actually exists and the step should move.
    #[test]
    fn walking_has_a_gradient_to_follow() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let cfg = JointCfg {
            dirs: 8,
            top: 3,
            scenarios: 1,
            ..JointCfg::default()
        };
        let mut policy = JointPolicy::seeded(frame, 3);
        let mut rng = Rng::new(5);
        let before = policy.theta.clone();
        iterate(&mut policy, &phys, Stage::WalkFlat, &cfg, &mut rng, 0);
        let moved = before
            .iter()
            .zip(policy.theta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            moved > 1e-9,
            "a stage with a real objective took no step at all"
        );
    }

    /// A checkpoint that does not reproduce its evaluation is not a
    /// checkpoint, so the round trip has to be exact rather than close.
    #[test]
    fn a_joint_checkpoint_round_trips_exactly() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let mut policy = JointPolicy::seeded(frame, 4);
        let mut rng = Rng::new(9);
        let cfg = JointCfg {
            dirs: 4,
            top: 2,
            scenarios: 1,
            ..JointCfg::default()
        };
        iterate(&mut policy, &phys, Stage::WalkFlat, &cfg, &mut rng, 0);

        let text = to_text(&policy);
        let back = from_text(&text).expect("round trip");
        assert_eq!(back.frame.legs(), frame.legs());
        assert_eq!(back.theta, policy.theta);
        assert_eq!(back.norm.mean, policy.norm.mean);
        assert_eq!(back.norm.m2, policy.norm.m2);

        // And it drives identically.
        let terrain = Terrain::new(Course::Flat, 2);
        let a = rollout(&policy, &phys, &terrain, Stage::WalkFlat, None);
        let b = rollout(&back, &phys, &terrain, Stage::WalkFlat, None);
        assert_eq!(a.score.to_bits(), b.score.to_bits(), "replay diverged");
    }

    /// A gait-level checkpoint must not load as a joint-level one. They drive
    /// different things and the failure would be silent.
    #[test]
    fn a_gait_checkpoint_is_refused() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::default_for(frame), frame);
        let text = crate::checkpoint::to_text(&gait);
        let err = from_text(&text).expect_err("a gait policy loaded as a joint policy");
        assert!(err.contains("joint-level"), "unhelpful error: {err}");
    }

    /// Where the wall-clock actually goes. Not an assertion — a measurement,
    /// printed so a tuning decision is made on numbers.
    #[test]
    fn bench_where_the_time_goes() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let gait = Policy::seeded(Preset::default_for(frame), frame).gait();

        // Cost of building one Rapier world.
        let t0 = std::time::Instant::now();
        let reps = 40;
        for _ in 0..reps {
            let p = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
            std::hint::black_box(p.substeps);
        }
        let setup = t0.elapsed().as_secs_f64() / reps as f64;

        // Cost of one full rollout.
        let policy = JointPolicy::seeded(frame, 1);
        let t1 = std::time::Instant::now();
        let reps = 8;
        for _ in 0..reps {
            std::hint::black_box(rollout(&policy, &phys, &terrain, Stage::WalkFlat, None));
        }
        let roll = t1.elapsed().as_secs_f64() / reps as f64;

        let horizon = Stage::WalkFlat.horizon();
        println!("  plant setup      {:>8.1} ms", setup * 1e3);
        println!("  rollout ({horizon}s sim) {:>8.1} ms  -> {:.1}x realtime", roll * 1e3, horizon / roll);
        println!("  setup share      {:>8.1} %", 100.0 * setup / roll);

        // One iteration's worth, serial, for comparison against the observed
        // wall time of the real (threaded) loop.
        let cfg = JointCfg { dirs: 16, top: 5, scenarios: 1, ..JointCfg::default() };
        let serial = roll * (cfg.dirs * 2 * cfg.scenarios) as f64;
        println!("  16 dirs serial   {:>8.2} s", serial);
        println!("  cores            {:>8?}", std::thread::available_parallelism());

        let mut p2 = JointPolicy::seeded(frame, 1);
        let mut rng = Rng::new(1);
        let t2 = std::time::Instant::now();
        iterate(&mut p2, &phys, Stage::WalkFlat, &cfg, &mut rng, 0);
        println!("  iterate() actual {:>8.2} s", t2.elapsed().as_secs_f64());

        let t3 = std::time::Instant::now();
        evaluate(&p2, &phys, Stage::WalkFlat, &[101, 202, 303]);
        println!("  evaluate() 3 sds {:>8.2} s", t3.elapsed().as_secs_f64());
    }

    /// How big an action does a sigma-sized weight perturbation actually
    /// produce? With the output layer seeded to zero the answer can be "a
    /// degree and a half", which is not exploration — it is a policy that
    /// cannot discover locomotion because it never tries any.
    #[test]
    fn bench_exploration_scale() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let base = JointPolicy::seeded(frame, 1);
        let no = n_obs(frame);
        let na = n_act(frame);

        println!("  sigma  max|act| deg   score spread   dist spread   feet");
        for sigma in [0.02f64, 0.05, 0.15, 0.30, 0.60] {
            let mut rng = Rng::new(4);
            let mut peak = 0.0f64;
            let mut scores: Vec<f64> = Vec::new();
            let mut dists: Vec<f64> = Vec::new();
            let mut feet = 0.0;
            for _ in 0..6 {
                let mut p = base.clone();
                for w in p.theta.iter_mut() {
                    *w += sigma * rng.normal();
                }
                let mut obs = vec![0.0; no];
                for (i, o) in obs.iter_mut().enumerate() {
                    *o = if i >= 6 * frame.legs() { 1.0 } else { 0.0 };
                }
                let mut act = vec![0.0; na];
                p.act(&obs, &mut act);
                peak = peak.max(act.iter().fold(0.0f64, |m, a| m.max(a.abs())));

                let r = rollout(&p, &phys, &terrain, Stage::WalkFlat, None);
                scores.push(r.score);
                dists.push(r.distance);
                feet += r.support;
            }
            let sp = scores.iter().cloned().fold(f64::MIN, f64::max)
                - scores.iter().cloned().fold(f64::MAX, f64::min);
            let dp = dists.iter().cloned().fold(f64::MIN, f64::max)
                - dists.iter().cloned().fold(f64::MAX, f64::min);
            println!(
                "  {sigma:>5.2}  {:>10.1}     {:>9.4}     {:>9.2}   {:>5.2}",
                peak.to_degrees(),
                sp,
                dp,
                feet / 6.0
            );
        }
    }
}
