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
//! the range and bearing to the next waypoint, whether this task requires a
//! jump, the next trench's near/far lip distances, and forward terrain heights.
//! Plus a free-running clock, which is an *input* and not a trajectory: nothing
//! tells the policy what to do at a given phase, it only gets to know that time
//! is periodic. Without it a feedback-only controller has to build its own
//! oscillator out of contact transitions before it can take a second step,
//! and that is a much harder search than it needs to be.
//!
//! Actions are offsets from the standing pose rather than absolute angles, so
//! the all-zero policy stands still instead of collapsing. That matters for
//! [`Stage::Stand`]: iteration zero is already on its feet, and training makes
//! it *stay* there rather than discovering the floor from scratch.

use crate::dynamics::Physics;
use crate::math::{Rng, clamp, hypot2, inv_rot_y, rot_y};
#[cfg(feature = "nexus-gpu")]
use crate::nexus_plant::NexusPlantBatch;
use crate::plant::ArticulatedPlant;
use crate::policy::{Policy, Preset};
use crate::robot::{Frame, MAX_LEGS, Q_LIMIT};
use crate::sim::DT;
use crate::terrain::{COURSES, Course, Terrain, WAYPOINT_R};

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

/// The unit the speed term is measured in, simulator units.
///
/// This is a unit conversion and nothing else. The term is *linear* in speed,
/// so summing it over an episode gives ground covered divided by
/// `REFERENCE_SPEED` — the constant scales the number and cannot express a
/// preference about how the ground was covered.
///
/// There used to be a commanded speed per stage (0.8 walking, 2.0 running, 4.0
/// for the parkour run-up) that the reward tracked, with a penalty above it.
/// That answers the wrong question. The task is to reach a point in space as
/// soon as possible through whatever is in the way, so nothing should prefer a
/// slower machine and no stage should carry a speed it is supposed to sit on.
///
/// It was briefly a saturating curve, which was worse than a target. Any
/// function of speed that is *concave* prefers a steady pace to a varying one
/// covering the same ground — Jensen's inequality — and the measured size of
/// that was 24.5%: crossing two ticks at 1.0 paid 1.245 where covering the same
/// distance as 2.0 then a standstill paid 1.000. It paid the machine not to
/// brake. Sometimes slowing into a trench lip is how you clear it and keep
/// going, exactly as slowing into a corner is how you leave it faster, and a
/// concave per-tick reward cannot represent that.
///
/// Linear is the only shape with no opinion: the sum depends on the distance
/// and not on the profile, so where the speed went is left to the policy and
/// the whole preference for arriving sooner lives in the terminal bonus, which
/// is where it can see the trade.
///
/// In physical units this is 0.2 m/s — simulator speeds are ten times physical,
/// the legs being 1.8 units of a 0.18 m reach.
pub const REFERENCE_SPEED: f64 = 2.0;

/// Weight on the terminal reward for reaching the target.
///
/// The per-tick shaping sums to 1.0 over a full episode at perfect quality, so
/// reaching the target *cost* the policy return before this existed: finishing
/// at half the horizon terminates the episode, which cuts the accumulation to
/// 0.5, and `terminated` cuts the bootstrap too, so 0.5 was the whole return
/// against 1.0 for never arriving. Dawdling beat arriving. Everything about
/// the route lived in `episode_score`, which drives checkpoint selection and
/// curriculum promotion and never reaches the gradient.
///
/// The dense term is now paid per step rather than per horizon-fraction, so
/// `gamma` is what bounds the return an early finish gives up: at most
/// `per_step / (1 - gamma)`, about 30 for a good gait, no matter how long the
/// clock runs. At 60.0 the bonus at the bar covers that in full and beating
/// the bar pays double, so arriving is never worse than dawdling and the
/// bound holds without the constant referring to the horizon.
pub const FINISH_BONUS: f64 = 60.0;

/// Spread of the terminal reward around the bar, as a fraction of it.
///
/// Arriving 30% under the bar scores about 0.73 of the bonus and 30% over
/// about 0.27. Being relative is what keeps this scale-free: there is no time
/// or speed constant, because the bar is measured rather than chosen.
pub const FINISH_SPREAD: f64 = 0.30;

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
/// estimate grows with the number of parameters. With the terrain scan a
/// 48-unit shape would carry 4866 weights probed by only 16 directions, which
/// is not an estimate so much as a guess.
pub const N_HIDDEN: usize = 24;

/// Forward height samples supplied to the policy. Five ranges across three
/// lanes are enough to distinguish a step, a narrow gap and a long trench
/// before a foot reaches it. They are observations only: no gait or jump is
/// scheduled from them.
pub const N_TERRAIN_SCAN: usize = 15;

const SCAN_RANGES: [f64; 5] = [0.5, 1.0, 1.5, 2.2, 3.2];
const SCAN_LANES: [f64; 3] = [-1.0, 0.0, 1.0];

/// Return spread below which an iteration is treated as carrying no gradient.
/// Per-tick shaping is bounded by roughly 1.0 and the route terms are bounded,
/// so this remains a fraction of a percent of the useful return range.
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
    + 1                                       // course requires jumping
    + 2                                       // next trench near/far lips
    + N_TERRAIN_SCAN                          // forward terrain heights
    + 2 * MAX_LEGS                            // per-leg clock sin, cos
    + 1                                       // bias
    + 3 * MAX_LEGS; // executed joint setpoints

pub const MAX_JOINT_ACT: usize = 3 * MAX_LEGS;

/// Observation width for `frame`.
pub fn n_obs(frame: Frame) -> usize {
    let n = frame.legs();
    // 3n angles + 3n rates + n contacts + 2n phase + 3n setpoints, then the
    // body block. Setpoints are appended so legacy observations stay a prefix.
    12 * n + 3 + 3 + 2 + 1 + 2 + 1 + 1 + 2 + N_TERRAIN_SCAN + 1
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

    fn merge(&mut self, other: &ObsNorm) {
        if self.frozen || other.n < 1.0 {
            return;
        }
        if self.mean.len() != other.mean.len() {
            return;
        }
        let total = self.n + other.n;
        for i in 0..self.mean.len() {
            let delta = other.mean[i] - self.mean[i];
            self.mean[i] += delta * other.n / total;
            self.m2[i] += other.m2[i] + delta * delta * self.n * other.n / total;
        }
        self.n = total;
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
            Stage::Jump => &[Course::Hurdles, Course::Jump, Course::Chasm],
            Stage::Mixed => &COURSES,
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

    /// Command for one course. The final stage samples every course, but the
    /// parkour pair still needs its run-up rather than an impossible 2.5 m/s
    /// walking command.
    pub fn speed_for(self, course: Course) -> f64 {
        if self == Stage::Mixed && course.is_jump() {
            Stage::Jump.speed()
        } else {
            self.speed()
        }
    }

    /// Rollout length, seconds.
    pub fn horizon(self) -> f64 {
        match self {
            Stage::Stand => 2.0,
            Stage::WalkFlat | Stage::RunFlat => 4.0,
            Stage::Mixed => 30.0,
            _ => 8.0,
        }
    }

    /// How much of the shaping pays for a tripod-shaped support pattern.
    ///
    /// On flat ground this is load-bearing: without it the search finds
    /// hopping, which collects most of the speed reward on one leg, and mean
    /// support fell to 1.2 of six. On terrain it is the opposite. It is a
    /// hand-specified gait, and a hand-specified gait constrains exploration
    /// exactly where the machine needs to improvise — the published ablation
    /// on this morphology (Li et al. 2024, 18-DoF hexapod) is that a
    /// prescribed tripod contact reward *hindered* obstacle learning and a
    /// style prior learned from flat ground did not.
    ///
    /// Measured here on the first run that reached the flat ceiling: support
    /// climbed 3.34 -> 3.86 -> 4.01 feet while the score fell 0.867 -> 0.812,
    /// so the term demonstrably steers the gait rather than merely floor it.
    /// It is kept where it was earned and dropped where it is not.
    pub fn gait_weight(self) -> f64 {
        match self {
            Stage::Stand | Stage::WalkFlat | Stage::RunFlat => 0.10,
            _ => 0.0,
        }
    }

    /// Mean score over the stage's courses at which the next stage opens. The
    /// shaping component is normalized by horizon; ordered route terms have
    /// the same fixed scale on every stage.
    pub fn promote_at(self) -> f64 {
        match self {
            // Standing still has some per-tick shaping but the episode-level
            // progress gate makes its locomotion score zero. These thresholds
            // therefore require actual movement.
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
    /// Episode fitness: per-second shaping plus ordered-route and finish terms.
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
    /// Ordered route state. `finished` means the terminal waypoint was
    /// entered; `completed` additionally requires every earlier waypoint.
    pub reached: usize,
    pub finished: bool,
    pub completed: bool,
    pub waypoint_fraction: f64,
    pub completion_rate: f64,
    /// Time to the finish, with a failure charged the full stage horizon.
    pub finish_time: f64,
}

/// The standing joint pose: what an all-zero action commands.
pub fn stand_pose(frame: Frame, phys: &Physics, terrain: &Terrain) -> [[f64; 3]; MAX_LEGS] {
    // The plant already computes this when it spawns a machine standing, so
    // taking it from there keeps the two definitions from drifting apart.
    let gait = Policy::seeded(Preset::default_for(frame), frame).gait();
    let plant = ArticulatedPlant::standing(frame, &gait, phys, terrain);
    plant.leg_q_all()
}

#[derive(Clone, Copy, Debug, Default)]
struct RouteState {
    wp: usize,
    reached: usize,
    finished: bool,
    range: f64,
    bearing: f64,
}

impl RouteState {
    /// Advance an ordered route without crediting waypoints merely passed at
    /// a distance. This mirrors the centroidal simulator's finish semantics,
    /// so the two trainers agree on what solving a course means.
    fn update(&mut self, terrain: &Terrain, pos: [f64; 3], yaw: f64) {
        if self.finished {
            return;
        }
        if terrain.waypoints.is_empty() {
            self.finished = true;
            self.range = 0.0;
            self.bearing = 0.0;
            return;
        }
        let last = terrain.waypoints.len() - 1;
        loop {
            let w = terrain.waypoint(self.wp);
            let (dx, dz) = (w[0] - pos[0], w[1] - pos[2]);
            let range = hypot2(dx, dz);
            if range < WAYPOINT_R {
                self.reached += 1;
                if self.wp == last {
                    self.finished = true;
                    self.range = range;
                    self.bearing = 0.0;
                    return;
                }
                self.wp += 1;
                continue;
            }
            if self.wp < last && w[1] < pos[2] - 1.5 {
                self.wp += 1;
                continue;
            }
            self.range = range;
            let body = inv_rot_y([dx, 0.0, dz], yaw);
            self.bearing = body[0].atan2(body[2]);
            return;
        }
    }
}

fn terrain_scan(terrain: &Terrain, pos: [f64; 3], yaw: f64, floor: f64, out: &mut [f64]) {
    debug_assert!(out.len() >= N_TERRAIN_SCAN);
    let mut k = 0;
    for distance in SCAN_RANGES {
        for lateral in SCAN_LANES {
            let d = rot_y([lateral, 0.0, distance], yaw);
            // Relative height makes the same step look the same after the
            // machine has climbed onto a platform. The clamp is the sensor's
            // finite useful range and keeps a deep trench from dominating the
            // network merely because it has a larger number.
            out[k] = clamp(
                terrain.probe(pos[0] + d[0], pos[2] + d[2]) - floor,
                -2.0,
                2.0,
            );
            k += 1;
        }
    }
}

#[inline]
fn jump_required(course: Course) -> f64 {
    f64::from(u8::from(course.is_jump()))
}

/// Signed metres from the chassis to the next parkour trench's near and far
/// lips. A continuous range lets a policy time crouch and lift-off from speed;
/// fixed height probes alone only say that some sample happens to be void.
fn jump_lip_distances(terrain: &Terrain, z: f64) -> [f64; 2] {
    if !terrain.course.is_jump() {
        return [0.0, 0.0];
    }
    terrain
        .obstacles
        .iter()
        .filter(|obstacle| obstacle.top < -0.1 && obstacle.z1 >= z - 0.2)
        .min_by(|a, b| a.z0.total_cmp(&b.z0))
        .map(|pit| [pit.z0 - z, pit.z1 - z])
        // A positive sentinel is distinct from being on a lip. The normalizer
        // handles scale; clamping keeps the last clear straight finite.
        .unwrap_or([8.0, 8.0])
        .map(|distance| clamp(distance, -2.0, 8.0))
}

/// Warmed initial state for one Rapier course. Cloning this is a cheap reset:
/// all rigid bodies, contacts, joints and broad-phase state return to the same
/// deterministic tick without rebuilding the authored robot and terrain.
#[derive(Clone)]
struct RapierEnv {
    plant: ArticulatedPlant,
    neutral: [[f64; 3]; MAX_LEGS],
    start: [f64; 3],
    stand_y: f64,
}

impl RapierEnv {
    fn new(frame: Frame, phys: &Physics, terrain: &Terrain) -> Self {
        let gait = Policy::seeded(Preset::default_for(frame), frame).gait();
        let mut plant = ArticulatedPlant::standing(frame, &gait, phys, terrain);
        let neutral = plant.leg_q_all();
        // Build the broad-phase once. Every reset cloned from this point has
        // the same initialized contact state as the scalar reference rollout.
        plant.step(DT);
        let (start, _, _, _) = plant.chassis_pose();
        Self {
            plant,
            neutral,
            start,
            stand_y: start[1],
        }
    }
}

#[derive(Clone, Copy)]
struct TickState {
    pos: [f64; 3],
    yaw: f64,
    pitch: f64,
    roll: f64,
    vel: [f64; 3],
    avel: [f64; 3],
    body_v: [f64; 3],
    q: [[f64; 3]; MAX_LEGS],
    contact: [bool; MAX_LEGS],
    support: f64,
    ride: f64,
}

/// Result of one motor-policy decision.
///
/// A decision advances [`DECIMATION`] physics ticks unless the episode ends
/// first. `reward` is the change in the episode score, so summing transition
/// rewards reproduces [`JointRollout::score`] exactly. `learning_reward` is the
/// local control signal intended for a value learner; strict episode scoring
/// remains authoritative for evaluation and curriculum promotion.
#[derive(Clone, Debug)]
pub struct JointStep {
    pub observation: Vec<f64>,
    pub reward: f64,
    /// Dense control reward without history-dependent episode gates. It can be
    /// negative when the local support estimate falls below a usable gait.
    pub learning_reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}

/// Contiguous replay sample, stored row-major as `batch × width`.
#[derive(Clone, Debug)]
pub struct JointReplayBatch {
    pub observations: Vec<f32>,
    pub actions: Vec<f32>,
    pub rewards: Vec<f32>,
    pub next_observations: Vec<f32>,
    /// True MDP termination: falls and completed routes. A value target must
    /// not bootstrap through this transition.
    pub terminated: Vec<bool>,
    /// Time or authored-world limit. Value targets may bootstrap through this
    /// transition even though collection resets the environment.
    pub truncated: Vec<bool>,
    pub observation_width: usize,
    pub action_width: usize,
}

/// Deterministic, gradually allocated circular replay buffer.
///
/// Storage is structure-of-arrays and contiguous so a sampled batch can move
/// directly to a tensor backend. Capacity is not allocated up front: choosing
/// a million-transition replay therefore does not reserve hundreds of
/// megabytes before the warm-up collector has produced any experience.
#[derive(Clone, Debug)]
pub struct JointReplay {
    capacity: usize,
    observation_width: usize,
    action_width: usize,
    len: usize,
    write_head: usize,
    observations: Vec<f32>,
    actions: Vec<f32>,
    rewards: Vec<f32>,
    next_observations: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
}

impl JointReplay {
    pub fn new(
        capacity: usize,
        observation_width: usize,
        action_width: usize,
    ) -> Result<Self, String> {
        if capacity == 0 || observation_width == 0 || action_width == 0 {
            return Err("replay capacity and widths must all be non-zero".into());
        }
        Ok(Self {
            capacity,
            observation_width,
            action_width,
            len: 0,
            write_head: 0,
            observations: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            next_observations: Vec::new(),
            terminated: Vec::new(),
            truncated: Vec::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes currently occupied by transition payloads (not allocator slack).
    pub fn payload_bytes(&self) -> usize {
        let floats = self.observations.len()
            + self.actions.len()
            + self.rewards.len()
            + self.next_observations.len();
        floats * size_of::<f32>()
            + (self.terminated.len() + self.truncated.len()) * size_of::<bool>()
    }

    pub fn push(
        &mut self,
        observation: &[f64],
        action: &[f64],
        reward: f64,
        next_observation: &[f64],
        terminated: bool,
        truncated: bool,
    ) -> Result<(), String> {
        if observation.len() != self.observation_width
            || next_observation.len() != self.observation_width
            || action.len() != self.action_width
        {
            return Err(format!(
                "replay transition widths state={}/{} action={}/{}, expected state={} action={}",
                observation.len(),
                next_observation.len(),
                action.len(),
                self.action_width,
                self.observation_width,
                self.action_width,
            ));
        }
        if !reward.is_finite()
            || observation.iter().any(|value| !value.is_finite())
            || next_observation.iter().any(|value| !value.is_finite())
            || action.iter().any(|value| !value.is_finite())
        {
            return Err("replay transition contains NaN or infinity".into());
        }

        let slot = self.write_head;
        if self.len < self.capacity {
            self.observations
                .extend(observation.iter().map(|value| *value as f32));
            self.actions
                .extend(action.iter().map(|value| *value as f32));
            self.rewards.push(reward as f32);
            self.next_observations
                .extend(next_observation.iter().map(|value| *value as f32));
            self.terminated.push(terminated);
            self.truncated.push(truncated);
            self.len += 1;
        } else {
            let obs = slot * self.observation_width;
            let act = slot * self.action_width;
            for (target, value) in self.observations[obs..obs + self.observation_width]
                .iter_mut()
                .zip(observation)
            {
                *target = *value as f32;
            }
            for (target, value) in self.next_observations[obs..obs + self.observation_width]
                .iter_mut()
                .zip(next_observation)
            {
                *target = *value as f32;
            }
            for (target, value) in self.actions[act..act + self.action_width]
                .iter_mut()
                .zip(action)
            {
                *target = *value as f32;
            }
            self.rewards[slot] = reward as f32;
            self.terminated[slot] = terminated;
            self.truncated[slot] = truncated;
        }
        self.write_head = (self.write_head + 1) % self.capacity;
        Ok(())
    }

    /// Sample with replacement using the project's seeded, platform-stable
    /// generator. Sampling is reproducible across worker counts and machines.
    pub fn sample(&self, batch_size: usize, rng: &mut Rng) -> Result<JointReplayBatch, String> {
        if batch_size == 0 {
            return Err("replay batch size must be non-zero".into());
        }
        if self.is_empty() {
            return Err("cannot sample an empty replay buffer".into());
        }
        let mut batch = JointReplayBatch {
            observations: Vec::with_capacity(batch_size * self.observation_width),
            actions: Vec::with_capacity(batch_size * self.action_width),
            rewards: Vec::with_capacity(batch_size),
            next_observations: Vec::with_capacity(batch_size * self.observation_width),
            terminated: Vec::with_capacity(batch_size),
            truncated: Vec::with_capacity(batch_size),
            observation_width: self.observation_width,
            action_width: self.action_width,
        };
        for _ in 0..batch_size {
            let slot = (rng.next_u64() % self.len as u64) as usize;
            let obs = slot * self.observation_width;
            let act = slot * self.action_width;
            batch
                .observations
                .extend_from_slice(&self.observations[obs..obs + self.observation_width]);
            batch
                .actions
                .extend_from_slice(&self.actions[act..act + self.action_width]);
            batch.rewards.push(self.rewards[slot]);
            batch
                .next_observations
                .extend_from_slice(&self.next_observations[obs..obs + self.observation_width]);
            batch.terminated.push(self.terminated[slot]);
            batch.truncated.push(self.truncated[slot]);
        }
        Ok(batch)
    }
}

/// Reusable step-wise Rapier environment for replay-based RL.
///
/// Observations returned here are raw. A learner should update its observation
/// statistics only from collected training transitions and apply the frozen
/// transform for evaluation. [`rollout`] does exactly that with [`ObsNorm`].
/// Actions are joint offsets in radians and are clamped to the same safe range
/// as the joint policy.
pub struct JointEnv {
    frame: Frame,
    phys: Physics,
    terrain: Terrain,
    stage: Stage,
    initial: RapierEnv,
    plant: ArticulatedPlant,
    neutral: [[f64; 3]; MAX_LEGS],
    start: [f64; 3],
    stand_y: f64,
    phase_off: [f64; MAX_LEGS],
    substeps: usize,
    cmd: f64,
    last_q: [[f64; 3]; MAX_LEGS],
    q_cmd: [[f64; 3]; MAX_LEGS],
    total: f64,
    support_sum: f64,
    air: f64,
    steps: usize,
    fell: bool,
    clock: f64,
    route: RouteState,
    finish_time: f64,
    /// Episode length in seconds. Starts at the stage's own horizon and is the
    /// trainer's to raise once the policy can use the time.
    horizon: f64,
    /// Finish time that scores half the terminal bonus. Zero until the trainer
    /// has seen an episode reach the target, and no bonus is paid before then:
    /// there is nothing to be half as good as yet.
    finish_bar: f64,
    duty: f64,
    max_ticks: usize,
    boundary: bool,
    observation: Vec<f64>,
}

impl JointEnv {
    pub fn new(frame: Frame, phys: &Physics, terrain: Terrain, stage: Stage) -> Self {
        let initial = RapierEnv::new(frame, phys, &terrain);
        Self::from_initial(frame, *phys, terrain, stage, initial)
    }

    fn from_initial(
        frame: Frame,
        phys: Physics,
        terrain: Terrain,
        stage: Stage,
        initial: RapierEnv,
    ) -> Self {
        let gait = Policy::seeded(Preset::default_for(frame), frame).gait();
        let plant = initial.plant.clone();
        let neutral = initial.neutral;
        let start = initial.start;
        let stand_y = initial.stand_y;
        let mut env = Self {
            frame,
            phys,
            cmd: stage.speed_for(terrain.course),
            max_ticks: (stage.horizon() / DT) as usize,
            horizon: stage.horizon(),
            finish_bar: 0.0,
            terrain,
            stage,
            initial,
            substeps: plant.substeps.max(1),
            last_q: plant.leg_q_all(),
            plant,
            neutral,
            start,
            stand_y,
            phase_off: gait.offsets,
            q_cmd: neutral,
            total: 0.0,
            support_sum: 0.0,
            air: 0.0,
            steps: 0,
            fell: false,
            clock: 0.0,
            route: RouteState::default(),
            finish_time: stage.horizon(),
            duty: frame.legs() as f64,
            boundary: false,
            observation: vec![0.0; n_obs(frame)],
        };
        env.refresh_observation();
        env
    }

    /// Restore the exact warmed initial physics state.
    pub fn reset(&mut self) -> &[f64] {
        self.plant = self.initial.plant.clone();
        self.neutral = self.initial.neutral;
        self.start = self.initial.start;
        self.stand_y = self.initial.stand_y;
        self.substeps = self.plant.substeps.max(1);
        self.last_q = self.plant.leg_q_all();
        self.q_cmd = self.neutral;
        self.total = 0.0;
        self.support_sum = 0.0;
        self.air = 0.0;
        self.steps = 0;
        self.fell = false;
        self.clock = 0.0;
        self.route = RouteState::default();
        self.finish_time = self.horizon;
        self.duty = self.frame.legs() as f64;
        self.boundary = false;
        self.refresh_observation();
        &self.observation
    }

    pub fn state(&self) -> &[f64] {
        &self.observation
    }

    /// Change the commanded forward speed without resetting physical state.
    /// This is an exogenous observation update used by speed curricula.
    pub fn set_command(&mut self, speed: f64) -> Result<&[f64], String> {
        if !speed.is_finite() || speed < 0.0 {
            return Err("joint command speed must be finite and non-negative".into());
        }
        self.cmd = speed;
        let tick = self.sample();
        self.fill_observation(&tick);
        Ok(&self.observation)
    }

    /// Episode length in seconds, and the tick budget that follows from it.
    ///
    /// Call between episodes. A short horizon early is cheap — many resets,
    /// fast turnover on the basic gait — and useless later, when the finish is
    /// out of reach however well the machine moves. Extending it is only worth
    /// anything once the time is being used, which is the trainer's call.
    pub fn set_horizon(&mut self, secs: f64) {
        self.horizon = secs.max(DT);
        self.max_ticks = (self.horizon / DT).max(1.0) as usize;
        self.finish_time = self.horizon;
    }

    pub fn horizon(&self) -> f64 {
        self.horizon
    }

    /// The finish time worth half the terminal bonus. See [`FINISH_SPREAD`].
    pub fn set_finish_bar(&mut self, secs: f64) {
        self.finish_bar = secs.max(0.0);
    }

    pub fn is_done(&self) -> bool {
        self.terminated() || self.truncated()
    }

    fn terminated(&self) -> bool {
        self.fell || self.route.finished
    }

    fn truncated(&self) -> bool {
        !self.terminated() && (self.steps >= self.max_ticks || self.boundary)
    }

    fn sample(&self) -> TickState {
        let (pos, yaw, pitch, roll) = self.plant.chassis_pose();
        let vel = self.plant.chassis_vel();
        let avel = self.plant.chassis_angvel();
        let q = self.plant.leg_q_all();
        let contact = self.plant.foot_contacts();
        let support = self
            .plant
            .support_under(pos[0], pos[2], pos[1] + 4.0)
            .unwrap_or(0.0);
        TickState {
            pos,
            yaw,
            pitch,
            roll,
            vel,
            avel,
            body_v: inv_rot_y([vel[0], vel[1], vel[2]], yaw),
            q,
            contact,
            support,
            ride: pos[1] - support,
        }
    }

    fn update_terminal_state(&mut self, tick: &TickState) {
        if tick.pitch.abs() > 1.0 || tick.roll.abs() > 1.0 || self.plant.chassis_dead(tick.vel) {
            self.fell = true;
            return;
        }
        let was_finished = self.route.finished;
        self.route.update(&self.terrain, tick.pos, tick.yaw);
        if !was_finished && self.route.finished {
            self.finish_time = self.steps as f64 * DT;
        }
        if self.terrain.waypoints.is_empty() && tick.pos[2] > crate::terrain::Z_MAX - 2.0 {
            self.boundary = true;
        }
    }

    fn fill_observation(&mut self, tick: &TickState) {
        let n = self.frame.legs();
        let mut w = 0usize;
        for i in 0..n {
            for c in 0..3 {
                self.observation[w] = tick.q[i][c] - self.neutral[i][c];
                w += 1;
            }
        }
        for i in 0..n {
            for c in 0..3 {
                self.observation[w] = (tick.q[i][c] - self.last_q[i][c]) / DT * 0.05;
                w += 1;
            }
        }
        for i in 0..n {
            self.observation[w] = if tick.contact[i] { 1.0 } else { 0.0 };
            w += 1;
        }
        self.observation[w] = tick.body_v[0];
        self.observation[w + 1] = tick.body_v[1];
        self.observation[w + 2] = tick.body_v[2];
        self.observation[w + 3] = tick.avel[0];
        self.observation[w + 4] = tick.avel[1];
        self.observation[w + 5] = tick.avel[2];
        self.observation[w + 6] = tick.pitch;
        self.observation[w + 7] = tick.roll;
        self.observation[w + 8] = tick.ride - self.stand_y;
        self.observation[w + 9] = self.route.range;
        self.observation[w + 10] = self.route.bearing;
        self.observation[w + 11] = self.cmd;
        self.observation[w + 12] = jump_required(self.terrain.course);
        let lips = jump_lip_distances(&self.terrain, tick.pos[2]);
        self.observation[w + 13] = lips[0];
        self.observation[w + 14] = lips[1];
        let scan = w + 15;
        terrain_scan(
            &self.terrain,
            tick.pos,
            tick.yaw,
            tick.support,
            &mut self.observation[scan..scan + N_TERRAIN_SCAN],
        );
        let phase = scan + N_TERRAIN_SCAN;
        for i in 0..n {
            let ph = (self.clock + self.phase_off[i]) * std::f64::consts::TAU;
            self.observation[phase + i * 2] = ph.sin();
            self.observation[phase + i * 2 + 1] = ph.cos();
        }
        self.observation[phase + n * 2] = 1.0;
        let setpoints = phase + n * 2 + 1;
        for i in 0..n {
            for c in 0..3 {
                self.observation[setpoints + i * 3 + c] = clamp(
                    (self.q_cmd[i][c] - self.neutral[i][c]) / ACT_RANGE,
                    -1.0,
                    1.0,
                );
            }
        }
        debug_assert_eq!(setpoints + n * 3, self.observation.len());
    }

    fn refresh_observation(&mut self) -> TickState {
        let tick = self.sample();
        self.update_terminal_state(&tick);
        self.fill_observation(&tick);
        tick
    }

    /// Advance one policy decision using joint offsets in radians.
    pub fn step(&mut self, action: &[f64]) -> Result<JointStep, String> {
        let expected = n_act(self.frame);
        if action.len() != expected {
            return Err(format!(
                "joint action width {}, expected {expected}",
                action.len()
            ));
        }
        if self.is_done() {
            return Err("cannot step a finished joint episode; reset it first".into());
        }

        let score_before = self.summary().score;
        let n = self.frame.legs();
        let mut learning_reward = 0.0;
        for _ in 0..DECIMATION {
            let was_finished = self.route.finished;
            let tick = self.refresh_observation();
            // Reaching the target, paid once, on the tick it happens. Sooner is
            // worth more, and it is paid here rather than after the `is_done`
            // break because arriving is exactly what ends the episode.
            if !was_finished && self.route.finished {
                learning_reward += FINISH_BONUS * self.promptness(self.finish_time);
            }
            if self.is_done() {
                break;
            }

            let mut jerk = 0.0;
            for i in 0..n {
                for c in 0..3 {
                    let (lo, hi) = Q_LIMIT[c];
                    let offset = clamp(action[i * 3 + c], -ACT_RANGE, ACT_RANGE);
                    let want = clamp(self.neutral[i][c] + offset, lo, hi);
                    let slew = MAX_JOINT_RATE * DT;
                    let moved = clamp(want - self.q_cmd[i][c], -slew, slew);
                    jerk += moved.abs();
                    self.q_cmd[i][c] += moved;
                }
            }

            self.last_q = tick.q;
            self.plant.drive(&self.q_cmd, &self.phys, DT);
            for _ in 0..self.substeps {
                self.plant.step(DT / self.substeps as f64);
            }

            let down = tick
                .contact
                .iter()
                .take(n)
                .filter(|contact| **contact)
                .count();
            self.duty += (down as f64 - self.duty) * (DT / 0.30);
            self.support_sum += down as f64;
            self.air = self.air.max(tick.pos[1] - self.stand_y);
            let shaped = reward(
                self.stage,
                &tick.body_v,
                tick.pitch,
                tick.roll,
                tick.ride,
                self.stand_y,
                down,
                self.duty,
                n,
                self.route.bearing,
            );
            let churn = jerk / (MAX_JOINT_RATE * DT * 3.0 * n as f64);
            let scored_tick = (shaped - SMOOTH_COST * churn).max(0.0);
            self.total += scored_tick;
            // Per-step and O(1) on purpose. Dividing this by the horizon (as
            // `base_score` still does, because a *score* must compare across
            // horizons) made each transition worth ~4e-4, so Q converged to
            // 0.18 and the actor's prior term was the same size as the whole
            // value range. `gamma` bounds the return at reward/(1-gamma)
            // regardless of episode length, which is what keeps FINISH_BONUS
            // comparable without referring to the clock.
            learning_reward += scored_tick;
            self.steps += 1;
            self.clock += DT / 0.5;
            if self.steps >= self.max_ticks {
                break;
            }
        }
        self.refresh_observation();
        let score_after = self.summary().score;
        Ok(JointStep {
            observation: self.observation.clone(),
            reward: score_after - score_before,
            learning_reward,
            terminated: self.terminated(),
            truncated: self.truncated(),
        })
    }

    /// Fraction of the terminal bonus an arrival at `secs` earns: 0.5 at the
    /// bar, rising toward 1.0 for a faster one and falling toward 0 for a
    /// slower. Half before any bar exists, so the arrival that sets the bar is
    /// paid the average rate rather than nothing.
    fn promptness(&self, secs: f64) -> f64 {
        // Half -- "average" -- until a bar exists. Returning zero meant the
        // first episode ever to arrive was paid nothing for arriving while
        // still forfeiting the rest of its clock, and that episode is the one
        // that sets the bar.
        if self.finish_bar <= 0.0 {
            return 0.5;
        }
        let z = (self.finish_bar - secs) / (FINISH_SPREAD * self.finish_bar);
        1.0 / (1.0 + (-z).exp())
    }

    /// Episode metrics at the current state.
    pub fn summary(&self) -> JointRollout {
        let (end, _, _, _) = self.plant.chassis_pose();
        let secs = self.steps as f64 * DT;
        let route_len = self.terrain.waypoints.len();
        let waypoint_fraction = if route_len == 0 {
            1.0
        } else {
            self.route.reached as f64 / route_len as f64
        };
        let completed = self.route.finished && self.route.reached == route_len;
        let base_score = self.total * DT / self.horizon.max(1e-6);
        let mean_support = if self.steps == 0 {
            // The environment is constructed from a warmed standing plant, so
            // its pre-step support is every foot, not none. Reading it as zero
            // opened a fresh episode on a fully closed support gate.
            self.frame.legs() as f64
        } else {
            self.support_sum / self.steps as f64
        };
        let score = episode_score(
            self.stage,
            base_score,
            end[2] - self.start[2],
            mean_support,
            self.frame.legs(),
            waypoint_fraction,
            completed,
            secs,
        );
        JointRollout {
            score,
            distance: end[2] - self.start[2],
            secs,
            fell: self.fell,
            support: mean_support,
            air: self.air,
            reached: self.route.reached,
            finished: self.route.finished,
            completed,
            waypoint_fraction,
            completion_rate: f64::from(u8::from(completed)),
            finish_time: self.finish_time,
        }
    }
}

/// Run one episode and score it.
pub fn rollout(
    policy: &JointPolicy,
    phys: &Physics,
    terrain: &Terrain,
    stage: Stage,
    norm_sink: Option<&mut ObsNorm>,
) -> JointRollout {
    let env = RapierEnv::new(policy.frame, phys, terrain);
    rollout_in_env(policy, phys, terrain, stage, norm_sink, env)
}

fn rollout_in_env(
    policy: &JointPolicy,
    phys: &Physics,
    terrain: &Terrain,
    stage: Stage,
    mut norm_sink: Option<&mut ObsNorm>,
    env: RapierEnv,
) -> JointRollout {
    let mut joint_env = JointEnv::from_initial(policy.frame, *phys, terrain.clone(), stage, env);
    let mut act = vec![0.0; n_act(policy.frame)];
    while !joint_env.is_done() {
        let mut obs = joint_env.state().to_vec();
        if let Some(sink) = norm_sink.as_deref_mut() {
            sink.observe(&obs);
        }
        policy.norm.apply(&mut obs);
        policy.act(&obs, &mut act);
        joint_env
            .step(&act)
            .expect("policy action always has the environment width");
    }
    joint_env.summary()
}

#[cfg(feature = "nexus-gpu")]
#[derive(Clone, Debug)]
struct NexusBatchRollout {
    rollout: JointRollout,
    norm: ObsNorm,
}

#[cfg(feature = "nexus-gpu")]
struct NexusEpisode {
    neutral: [[f64; 3]; MAX_LEGS],
    obs: Vec<f64>,
    act: Vec<f64>,
    last_q: [[f64; 3]; MAX_LEGS],
    q_cmd: [[f64; 3]; MAX_LEGS],
    norm: ObsNorm,
    start: [f64; 3],
    stand_y: f64,
    cmd: f64,
    total: f64,
    support_sum: f64,
    air: f64,
    steps: usize,
    fell: bool,
    clock: f64,
    route: RouteState,
    finish_time: f64,
    duty: f64,
    active: bool,
}

/// Lockstep Nexus rollout. Policy inference and scoring remain identical to
/// [`rollout`]; only the articulated physics step is batched on the GPU.
#[cfg(feature = "nexus-gpu")]
fn rollout_nexus_batch(
    policies: &[JointPolicy],
    phys: &Physics,
    terrains: &[Terrain],
    stage: Stage,
    collect_norm: bool,
    device: usize,
) -> Result<Vec<NexusBatchRollout>, String> {
    if policies.len() != terrains.len() || policies.is_empty() {
        return Err(
            "Nexus rollout policies and terrains must have the same non-zero length".into(),
        );
    }
    let frame = policies[0].frame;
    if policies.iter().any(|p| p.frame != frame) {
        return Err("one Nexus batch cannot mix robot frames".into());
    }
    let n = frame.legs();
    let no = n_obs(frame);
    let na = n_act(frame);
    let gait = Policy::seeded(Preset::default_for(frame), frame).gait();
    let phase_off = gait.offsets;
    let mut plant = NexusPlantBatch::new(frame, &gait, phys, terrains, device)?;
    debug_assert_eq!(plant.len(), policies.len());

    // Match the scalar plant's broad-phase warm-up step.
    plant.step()?;
    let initial = plant.snapshots()?;
    let mut episodes = Vec::with_capacity(policies.len());
    for (env, (snapshot, terrain)) in initial.iter().zip(terrains).enumerate() {
        let neutral = plant.neutral(env);
        let mut route = RouteState::default();
        route.update(terrain, snapshot.pos, snapshot.yaw);
        episodes.push(NexusEpisode {
            neutral,
            obs: vec![0.0; no],
            act: vec![0.0; na],
            last_q: snapshot.q,
            q_cmd: neutral,
            norm: ObsNorm::new(no),
            start: snapshot.pos,
            stand_y: snapshot.pos[1],
            cmd: stage.speed_for(terrain.course),
            total: 0.0,
            support_sum: 0.0,
            air: 0.0,
            steps: 0,
            fell: false,
            clock: 0.0,
            route,
            finish_time: stage.horizon(),
            duty: n as f64,
            active: true,
        });
    }

    let ticks = (stage.horizon() / DT) as usize;
    let mut snapshots = initial;
    for tick in 0..ticks {
        let mut commands = episodes.iter().map(|e| e.q_cmd).collect::<Vec<_>>();
        let mut any_active = false;
        for env in 0..episodes.len() {
            let ep = &mut episodes[env];
            if !ep.active {
                continue;
            }
            any_active = true;
            let state = snapshots[env];
            let terrain = &terrains[env];
            if state.pitch.abs() > 1.0 || state.roll.abs() > 1.0 || state.chassis_contact {
                ep.fell = true;
                ep.active = false;
                continue;
            }

            let support = terrain.probe(state.pos[0], state.pos[2]);
            let ride = state.pos[1] - support;
            let body_v = inv_rot_y(state.vel, state.yaw);
            let was_finished = ep.route.finished;
            ep.route.update(terrain, state.pos, state.yaw);
            if !was_finished && ep.route.finished {
                ep.finish_time = tick as f64 * DT;
            }
            if ep.route.finished {
                ep.active = false;
                continue;
            }
            let (range, bearing) = (ep.route.range, ep.route.bearing);

            let mut w = 0usize;
            for i in 0..n {
                for c in 0..3 {
                    ep.obs[w] = state.q[i][c] - ep.neutral[i][c];
                    w += 1;
                }
            }
            for i in 0..n {
                for c in 0..3 {
                    ep.obs[w] = (state.q[i][c] - ep.last_q[i][c]) / DT * 0.05;
                    w += 1;
                }
            }
            for i in 0..n {
                ep.obs[w] = f64::from(u8::from(state.contacts[i]));
                w += 1;
            }
            ep.obs[w] = body_v[0];
            ep.obs[w + 1] = body_v[1];
            ep.obs[w + 2] = body_v[2];
            ep.obs[w + 3] = state.angvel[0];
            ep.obs[w + 4] = state.angvel[1];
            ep.obs[w + 5] = state.angvel[2];
            ep.obs[w + 6] = state.pitch;
            ep.obs[w + 7] = state.roll;
            ep.obs[w + 8] = ride - ep.stand_y;
            ep.obs[w + 9] = range;
            ep.obs[w + 10] = bearing;
            ep.obs[w + 11] = ep.cmd;
            ep.obs[w + 12] = jump_required(terrain.course);
            let lips = jump_lip_distances(terrain, state.pos[2]);
            ep.obs[w + 13] = lips[0];
            ep.obs[w + 14] = lips[1];
            let scan = w + 15;
            terrain_scan(
                terrain,
                state.pos,
                state.yaw,
                support,
                &mut ep.obs[scan..scan + N_TERRAIN_SCAN],
            );
            let phase = scan + N_TERRAIN_SCAN;
            for i in 0..n {
                let ph = (ep.clock + phase_off[i]) * std::f64::consts::TAU;
                ep.obs[phase + i * 2] = ph.sin();
                ep.obs[phase + i * 2 + 1] = ph.cos();
            }
            ep.obs[phase + n * 2] = 1.0;

            if tick % DECIMATION == 0 {
                if collect_norm {
                    ep.norm.observe(&ep.obs);
                }
                policies[env].norm.apply(&mut ep.obs);
                policies[env].act(&ep.obs, &mut ep.act);
            }

            let mut jerk = 0.0;
            for i in 0..n {
                for c in 0..3 {
                    let (lo, hi) = Q_LIMIT[c];
                    let want = clamp(ep.neutral[i][c] + ep.act[i * 3 + c], lo, hi);
                    let slew = MAX_JOINT_RATE * DT;
                    let moved = clamp(want - ep.q_cmd[i][c], -slew, slew);
                    jerk += moved.abs();
                    ep.q_cmd[i][c] += moved;
                }
            }
            commands[env] = ep.q_cmd;
            ep.last_q = state.q;

            let down = state.contacts.iter().take(n).filter(|c| **c).count();
            ep.duty += (down as f64 - ep.duty) * (DT / 0.30);
            ep.support_sum += down as f64;
            ep.air = ep.air.max(state.pos[1] - ep.stand_y);
            let shaped = reward(
                stage,
                &body_v,
                state.pitch,
                state.roll,
                ride,
                ep.stand_y,
                down,
                ep.duty,
                n,
                bearing,
            );
            let churn = jerk / (MAX_JOINT_RATE * DT * 3.0 * n as f64);
            ep.total += (shaped - SMOOTH_COST * churn).max(0.0);
            ep.steps += 1;
            ep.clock += DT / 0.5;
            if terrain.waypoints.is_empty() && state.pos[2] > crate::terrain::Z_MAX - 2.0 {
                ep.active = false;
            }
        }
        if !any_active {
            break;
        }
        plant.drive(&commands, phys)?;
        plant.step()?;
        snapshots = plant.snapshots()?;
    }

    let final_state = snapshots;
    let mut out = Vec::with_capacity(episodes.len());
    for (env, mut ep) in episodes.into_iter().enumerate() {
        let end = final_state[env];
        let was_finished = ep.route.finished;
        ep.route.update(&terrains[env], end.pos, end.yaw);
        if !was_finished && ep.route.finished {
            ep.finish_time = ep.steps as f64 * DT;
        }
        let secs = ep.steps as f64 * DT;
        let route_len = terrains[env].waypoints.len();
        let waypoint_fraction = if route_len == 0 {
            1.0
        } else {
            ep.route.reached as f64 / route_len as f64
        };
        let completed = ep.route.finished && ep.route.reached == route_len;
        let mean_support = if ep.steps == 0 {
            0.0
        } else {
            ep.support_sum / ep.steps as f64
        };
        let distance = end.pos[2] - ep.start[2];
        let base_score = ep.total * DT / stage.horizon().max(1e-6);
        let score = episode_score(
            stage,
            base_score,
            distance,
            mean_support,
            n,
            waypoint_fraction,
            completed,
            secs,
        );
        out.push(NexusBatchRollout {
            rollout: JointRollout {
                score,
                distance,
                secs,
                fell: ep.fell,
                support: mean_support,
                air: ep.air,
                reached: ep.route.reached,
                finished: ep.route.finished,
                completed,
                waypoint_fraction,
                completion_rate: f64::from(u8::from(completed)),
                finish_time: ep.finish_time,
            },
            norm: ep.norm,
        });
    }
    Ok(out)
}

fn episode_score(
    stage: Stage,
    base_score: f64,
    distance: f64,
    support: f64,
    legs: usize,
    waypoint_fraction: f64,
    completed: bool,
    // Simulated seconds the episode lasted. An episode terminates on route
    // finish, so for a completed run this *is* the time to the target.
    secs: f64,
) -> f64 {
    if stage == Stage::Stand {
        return base_score;
    }
    // Tick reward alone can still be gamed by oscillating forward during
    // rewarded instants and drifting farther backward between them. Keep it
    // only in proportion to net progress along the course. A quarter of
    // commanded progress opens the full shaping reward; no or negative
    // progress earns none.
    let expected = (REFERENCE_SPEED * stage.horizon()).max(1e-6);
    let progress_gate = clamp(distance / (0.25 * expected), 0.0, 1.0);
    // A one-foot hopper can have forward instants and respectable net travel,
    // but it is not a usable hexapod gait. Penalise the whole episode by mean
    // contact, reaching full credit at roughly two feet on a hexapod. Squaring
    // keeps a brief flight phase affordable and makes sustained hopping steep.
    let support_gate = support_gate(support, legs);
    // Reaching the target was worth a flat 1.0 however long it took, so a
    // controller that strolled to the finish scored exactly like one that ran.
    // Time to the target is the actual objective, so pay for it: arriving
    // immediately is worth 2.0, arriving as the horizon expires is worth the
    // old 1.0, and nothing below that changes.
    let promptness = if completed {
        1.0 + (1.0 - secs / stage.horizon().max(1e-6)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    // Support is *only* ever a multiplier, never a subtraction. `reward` already
    // gates every tick by `enough_feet`, so subtracting a deficit on top of
    // that double-counted it -- and it was the one term that could drive a
    // score, or a transition reward, below zero. A negative per-step reward
    // makes falling over an escape from further penalty, because `terminated`
    // cuts the bootstrap: the first run with a gradient strong enough to
    // optimise anything went to 0.60 feet of six and scored -0.229.
    base_score * progress_gate * support_gate + 0.35 * waypoint_fraction + promptness
}

fn support_gate(support: f64, legs: usize) -> f64 {
    let target = (0.35 * legs as f64).max(1.0);
    clamp(support / target, 0.0, 1.0).powi(2)
}

/// Per-tick reward. Moving toward the target at [`REFERENCE_SPEED`] is 1.0;
/// faster is proportionally more, with no ceiling, because a ceiling is a pace
/// to settle at.
#[allow(clippy::too_many_arguments)]
fn reward(
    stage: Stage,
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

    // Speed toward the ordered waypoint. Using speed along the body's heading
    // lets a policy turn around and improve its score while its world progress
    // becomes negative. Projecting velocity onto the target direction makes
    // that exploit score zero and still gives a smooth guide while turning.
    // Deliberately *not* the gait-level trainer's Gaussian: that is a scoring
    // function, and this has to be a guide.
    //
    // A Gaussian centred on the command is flat where training starts. At rest
    // against a 0.8 m/s command it reads exp(-(0.8/0.37)^2) = 0.009, and its
    // slope there is nearly zero, so nothing pulls the machine into moving at
    // all — measured, walking sat at 0.20 for iterations while lifting legs
    // and covering no ground. Rising linearly to the command gives a constant
    // gradient from a standstill; above the command it falls off, so this is
    // still speed *tracking* and not a prize for going as fast as possible.
    let along = body_v[0] * bearing.sin() + body_v[2] * bearing.cos();
    // Linear in speed toward the target, so the sum over an episode is the
    // ground covered and nothing about the profile. See [`REFERENCE_SPEED`]:
    // both a saturating curve and a tracked command paid the machine to hold
    // one pace, and holding one pace is not how a trench gets crossed.
    //
    // Unbounded above on purpose. The machine's top speed is set by its servos,
    // the posture gate pays nothing while airborne, and there is no fastest
    // speed worth naming — that is the point.
    let track = (along / REFERENCE_SPEED).max(0.0);
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
    // it forfeits everything. Standing still now has about 0.16 of per-tick
    // shaping (and zero episode fitness after the progress gate), walking well
    // still approaches 1.0, and the path between them runs downhill.
    //
    // Support is part of the gate too, not a bonus. As a 0.10 term the machine
    // kept trading it away: mean support fell to 1.3 feet of six, because
    // hopping on one leg still collected most of the speed reward. A hexapod
    // on fewer than two feet is not walking, whatever its velocity says, so
    // below that the tick is worth almost nothing.
    let enough_feet = ((duty - 0.35) / 1.4).clamp(0.0, 1.0);
    let posture = level * height * enough_feet;
    // Whatever the tripod term does not take goes to speed tracking, so the
    // weights still sum to one and a terrain stage is not quietly scored out
    // of a lower maximum than a flat one.
    let gait_w = stage.gait_weight();
    ((0.85 - gait_w) * track + 0.15 * aim + gait_w * gait) * posture
}

/// ARS configuration for the joint-level trainer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum JointBackend {
    /// Reusable paired Rapier worlds on CPU. This is the reference path.
    #[default]
    Rapier,
    /// Independent Rapier-compatible worlds batched by Nexus on a native GPU.
    NexusGpu,
}

impl JointBackend {
    pub const fn name(self) -> &'static str {
        match self {
            JointBackend::Rapier => "rapier",
            JointBackend::NexusGpu => "nexus-gpu",
        }
    }
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
    pub backend: JointBackend,
    /// Maximum environments kept in flight at once. On Rapier this caps CPU
    /// rollout workers; on Nexus it is the number finalized in one GPU batch.
    pub batch_envs: usize,
    /// Native GPU adapter index. Nexus WebGPU currently supports adapter zero.
    pub device: usize,
}

impl Default for JointCfg {
    fn default() -> Self {
        JointCfg {
            dirs: 16,
            top: 6,
            // Small. ARS divides its step by return spread, which is smallest
            // precisely when the top directions agree. With the elitist
            // validation centre, 0.005 made ten consecutive rejected steps;
            // 0.002 improved the same checkpoint 0.221 -> 0.252 -> 0.281 in
            // five updates while raising mean support to 1.81 feet.
            alpha: 0.002,
            // Exploration noise has to produce visible leg motion without
            // immediately throwing away support. On the support-gated reward,
            // 0.02 reached 0.93 m with 1.76 mean feet down in seven updates;
            // 0.05 reached a similar distance with only 1.11 feet and rescored
            // at less than a third as much. Larger perturbations mostly throw
            // the plant and collapse the informative spread.
            sigma: 0.02,
            scenarios: 2,
            workers: 0,
            backend: JointBackend::Rapier,
            batch_envs: 128,
            device: 0,
        }
    }
}

/// Mean score of `policy` on `stage`, over a fixed set of seeds.
pub fn evaluate(policy: &JointPolicy, phys: &Physics, stage: Stage, seeds: &[u64]) -> JointRollout {
    evaluate_on_courses(policy, phys, stage, stage.courses(), seeds)
}

/// Mean result over an explicit course slice. This is also the primitive the
/// CLI uses for per-course held-out reporting; training and evaluation then
/// cannot quietly disagree about rollout semantics.
pub fn evaluate_on_courses(
    policy: &JointPolicy,
    phys: &Physics,
    stage: Stage,
    courses: &[Course],
    seeds: &[u64],
) -> JointRollout {
    let mut acc = JointRollout {
        finished: true,
        completed: true,
        reached: usize::MAX,
        ..JointRollout::default()
    };
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
            acc.reached = acc.reached.min(r.reached);
            acc.finished &= r.finished;
            acc.completed &= r.completed;
            acc.waypoint_fraction += r.waypoint_fraction;
            acc.completion_rate += r.completion_rate;
            acc.finish_time += r.finish_time;
            n += 1.0;
        }
    }
    if n > 0.0 {
        acc.score /= n;
        acc.distance /= n;
        acc.secs /= n;
        acc.support /= n;
        acc.waypoint_fraction /= n;
        acc.completion_rate /= n;
        acc.finish_time /= n;
    } else {
        acc.reached = 0;
        acc.finished = false;
        acc.completed = false;
    }
    acc
}

/// Evaluate with the rollout backend selected by `cfg`.
///
/// The CPU function above remains the stable reference primitive. This
/// fallible entry point is used by training and the CLI so GPU initialization
/// or simulation failures are reported instead of silently switching physics.
pub fn evaluate_on_courses_backend(
    policy: &JointPolicy,
    phys: &Physics,
    stage: Stage,
    courses: &[Course],
    seeds: &[u64],
    cfg: &JointCfg,
) -> Result<JointRollout, String> {
    if cfg.backend == JointBackend::Rapier {
        let mut terrains = Vec::with_capacity(courses.len() * seeds.len());
        for &seed in seeds {
            for &course in courses {
                terrains.push(Terrain::new(course, seed));
            }
        }
        return Ok(mean_rollouts(&parallel_rapier_rollouts(
            policy, phys, stage, &terrains, cfg,
        )));
    }

    #[cfg(not(feature = "nexus-gpu"))]
    {
        let _ = (policy, phys, stage, courses, seeds);
        return Err(
            "the Nexus backend requires building hexapod-core with feature `nexus-gpu`".into(),
        );
    }

    #[cfg(feature = "nexus-gpu")]
    {
        let mut terrains = Vec::with_capacity(courses.len() * seeds.len());
        for &seed in seeds {
            for &course in courses {
                terrains.push(Terrain::new(course, seed));
            }
        }
        let batch_envs = cfg.batch_envs.max(1);
        let mut rollouts = Vec::with_capacity(terrains.len());
        for chunk in terrains.chunks(batch_envs) {
            let policies = vec![policy.clone(); chunk.len()];
            rollouts.extend(
                rollout_nexus_batch(&policies, phys, chunk, stage, false, cfg.device)?
                    .into_iter()
                    .map(|r| r.rollout),
            );
        }
        Ok(mean_rollouts(&rollouts))
    }
}

fn parallel_rapier_rollouts(
    policy: &JointPolicy,
    phys: &Physics,
    stage: Stage,
    terrains: &[Terrain],
    cfg: &JointCfg,
) -> Vec<JointRollout> {
    if terrains.is_empty() {
        return Vec::new();
    }
    let requested_workers = if cfg.workers == 0 {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    } else {
        cfg.workers
    };
    let workers = requested_workers
        .max(1)
        .min(cfg.batch_envs.max(1))
        .min(terrains.len());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut indexed = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let next = &next;
            handles.push(scope.spawn(move || {
                let mut out = Vec::new();
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(terrain) = terrains.get(index) else {
                        break;
                    };
                    out.push((index, rollout(policy, phys, terrain, stage, None)));
                }
                out
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("joint-RL evaluation worker panicked"))
            .collect::<Vec<_>>()
    });
    indexed.sort_by_key(|(index, _)| *index);
    indexed.into_iter().map(|(_, rollout)| rollout).collect()
}

#[derive(Debug)]
struct RapierRunResult {
    job: usize,
    side: usize,
    score: f64,
    norm: ObsNorm,
}

/// Evaluate paired ARS perturbations on reusable, warmed Rapier snapshots.
///
/// A direction/scenario pair shares exactly the same authored world and
/// initial contact state between its positive and negative side. Each sign is
/// scheduled independently, exposing `2 * directions * scenarios` work units
/// to the CPU and balancing early falls against longer surviving episodes.
fn rapier_ars_results(
    base_policy: &JointPolicy,
    phys: &Physics,
    stage: Stage,
    cfg: &JointCfg,
    deltas: &[f64],
    plan: &[Vec<Terrain>],
    n_theta: usize,
) -> Vec<(usize, f64, f64, ObsNorm)> {
    let dirs = plan.len();
    if dirs == 0 {
        return Vec::new();
    }

    let mut plus_policies = Vec::with_capacity(dirs);
    let mut minus_policies = Vec::with_capacity(dirs);
    for direction in 0..dirs {
        let mut plus = base_policy.clone();
        let mut minus = base_policy.clone();
        for j in 0..n_theta {
            let shift = cfg.sigma * deltas[direction * n_theta + j];
            plus.theta[j] += shift;
            minus.theta[j] -= shift;
        }
        plus_policies.push(plus);
        minus_policies.push(minus);
    }

    let jobs = plan.iter().map(Vec::len).sum::<usize>();
    if jobs == 0 {
        return (0..dirs)
            .map(|direction| (direction, 0.0, 0.0, ObsNorm::new(n_obs(base_policy.frame))))
            .collect();
    }
    let mut metadata = Vec::with_capacity(jobs);
    for (direction, scenarios) in plan.iter().enumerate() {
        for scenario in 0..scenarios.len() {
            metadata.push((direction, scenario));
        }
    }

    // Author each course once, then clone that warmed snapshot independently
    // for both perturbation signs. Keeping signs as separate scheduler jobs is
    // important: direction-level scheduling caps concurrency at the direction
    // count and makes one long-lived sign hold up its whole paired job.
    let templates = metadata
        .iter()
        .map(|&(direction, scenario)| {
            RapierEnv::new(base_policy.frame, phys, &plan[direction][scenario])
        })
        .collect::<Vec<_>>();
    let runs = jobs * 2;
    let requested_workers = if cfg.workers == 0 {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    } else {
        cfg.workers
    };
    let workers = requested_workers
        .max(1)
        .min(cfg.batch_envs.max(1))
        .min(runs);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut pairs = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            handles.push(scope.spawn(|| {
                let mut out = Vec::new();
                loop {
                    let run = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if run >= runs {
                        break;
                    }
                    let side = run / jobs;
                    let job = run % jobs;
                    let (direction, scenario) = metadata[job];
                    let terrain = &plan[direction][scenario];
                    let mut norm = ObsNorm::new(n_obs(base_policy.frame));
                    let collect_norm = !base_policy.norm.frozen;
                    let selected = if side == 0 {
                        &plus_policies[direction]
                    } else {
                        &minus_policies[direction]
                    };
                    let score = rollout_in_env(
                        selected,
                        phys,
                        terrain,
                        stage,
                        collect_norm.then_some(&mut norm),
                        templates[job].clone(),
                    )
                    .score;
                    out.push(RapierRunResult {
                        job,
                        side,
                        score,
                        norm,
                    });
                }
                out
            }));
        }
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("joint-RL Rapier worker panicked"))
            .collect::<Vec<_>>()
    });
    pairs.sort_by_key(|result| (result.side, result.job));

    let mut sums = vec![[0.0; 2]; dirs];
    let mut counts = vec![0usize; dirs];
    let mut norms = (0..dirs)
        .map(|_| ObsNorm::new(n_obs(base_policy.frame)))
        .collect::<Vec<_>>();
    for result in &pairs {
        let direction = metadata[result.job].0;
        sums[direction][result.side] += result.score;
        if result.side == 0 {
            counts[direction] += 1;
        }
        // Sorting by sign then scenario matches the scalar implementation's
        // observation order: all positive scenarios, then all negative ones.
        norms[direction].merge(&result.norm);
    }

    (0..dirs)
        .map(|direction| {
            let count = counts[direction].max(1) as f64;
            (
                direction,
                sums[direction][0] / count,
                sums[direction][1] / count,
                norms[direction].clone(),
            )
        })
        .collect()
}

fn mean_rollouts(rollouts: &[JointRollout]) -> JointRollout {
    if rollouts.is_empty() {
        return JointRollout::default();
    }
    let mut acc = JointRollout {
        finished: true,
        completed: true,
        reached: usize::MAX,
        ..JointRollout::default()
    };
    for r in rollouts {
        acc.score += r.score;
        acc.distance += r.distance;
        acc.secs += r.secs;
        acc.support += r.support;
        acc.air = acc.air.max(r.air);
        acc.fell |= r.fell;
        acc.reached = acc.reached.min(r.reached);
        acc.finished &= r.finished;
        acc.completed &= r.completed;
        acc.waypoint_fraction += r.waypoint_fraction;
        acc.completion_rate += r.completion_rate;
        acc.finish_time += r.finish_time;
    }
    let n = rollouts.len() as f64;
    acc.score /= n;
    acc.distance /= n;
    acc.secs /= n;
    acc.support /= n;
    acc.waypoint_fraction /= n;
    acc.completion_rate /= n;
    acc.finish_time /= n;
    acc
}

#[cfg(feature = "nexus-gpu")]
fn nexus_ars_results(
    base_policy: &JointPolicy,
    phys: &Physics,
    stage: Stage,
    cfg: &JointCfg,
    deltas: &[f64],
    plan: &[Vec<Terrain>],
    n_theta: usize,
) -> Result<Vec<(usize, f64, f64, ObsNorm)>, String> {
    let dirs = plan.len();
    let jobs = dirs * 2 * plan.first().map_or(0, Vec::len);
    let mut policies = Vec::with_capacity(jobs);
    let mut terrains = Vec::with_capacity(jobs);
    let mut metadata = Vec::with_capacity(jobs);
    for (direction, scenarios) in plan.iter().enumerate() {
        for (side, sign) in [(0usize, 1.0), (1usize, -1.0)] {
            let mut perturbed = base_policy.clone();
            for (j, weight) in perturbed.theta.iter_mut().enumerate() {
                *weight += sign * cfg.sigma * deltas[direction * n_theta + j];
            }
            for terrain in scenarios {
                policies.push(perturbed.clone());
                terrains.push(terrain.clone());
                metadata.push((direction, side));
            }
        }
    }

    let mut sums = vec![[0.0; 2]; dirs];
    let mut counts = vec![[0usize; 2]; dirs];
    let mut norms = (0..dirs)
        .map(|_| ObsNorm::new(n_obs(base_policy.frame)))
        .collect::<Vec<_>>();
    let batch_envs = cfg.batch_envs.max(1);
    for start in (0..policies.len()).step_by(batch_envs) {
        let end = (start + batch_envs).min(policies.len());
        let batch = rollout_nexus_batch(
            &policies[start..end],
            phys,
            &terrains[start..end],
            stage,
            !base_policy.norm.frozen,
            cfg.device,
        )?;
        for (offset, result) in batch.into_iter().enumerate() {
            let (direction, side) = metadata[start + offset];
            sums[direction][side] += result.rollout.score;
            counts[direction][side] += 1;
            norms[direction].merge(&result.norm);
        }
    }

    Ok((0..dirs)
        .map(|direction| {
            let plus = sums[direction][0] / counts[direction][0].max(1) as f64;
            let minus = sums[direction][1] / counts[direction][1].max(1) as f64;
            (direction, plus, minus, norms[direction].clone())
        })
        .collect())
}

#[cfg(not(feature = "nexus-gpu"))]
fn nexus_ars_results(
    _base_policy: &JointPolicy,
    _phys: &Physics,
    _stage: Stage,
    _cfg: &JointCfg,
    _deltas: &[f64],
    _plan: &[Vec<Terrain>],
    _n_theta: usize,
) -> Result<Vec<(usize, f64, f64, ObsNorm)>, String> {
    Err("the Nexus backend requires building hexapod-core with feature `nexus-gpu`".into())
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

    // (direction, plus score, minus score, pooled observation stats)
    let mut results: Vec<(usize, f64, f64, ObsNorm)> = match cfg.backend {
        JointBackend::Rapier => rapier_ars_results(policy, phys, stage, cfg, &deltas, &plan, n),
        JointBackend::NexusGpu => nexus_ars_results(policy, phys, stage, cfg, &deltas, &plan, n)
            .unwrap_or_else(|e| panic!("Nexus ARS rollout failed: {e}")),
    };

    // Fold the pooled observation statistics back in before the weight step,
    // so the next iteration normalises with what this one actually saw. Skipped
    // once the scaling is frozen — see `train_curriculum`, which fixes it up
    // front precisely so the gradient and the evaluation agree.
    if !policy.norm.frozen {
        for (_, _, _, norm) in &results {
            policy.norm.merge(norm);
        }
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

fn warm_normalizer(
    policy: &JointPolicy,
    phys: &Physics,
    cfg: &JointCfg,
    rng: &mut Rng,
) -> Result<ObsNorm, String> {
    // Include the sensor extremes as well as moving body state. Fitting on
    // flat alone gives every height probe near-zero variance, so the first
    // real step or trench saturates the normalised input and is barely
    // distinguishable from every other obstacle.
    let probes = [
        (Stage::WalkFlat, Course::Flat),
        (Stage::Rough, Course::Steps),
        (Stage::Rough, Course::Rubble),
        (Stage::Gaps, Course::Gaps),
        (Stage::Jump, Course::Jump),
        (Stage::Rough, Course::Slalom),
        (Stage::Rough, Course::Slick),
    ];
    let mut jobs = Vec::with_capacity(probes.len());
    for (stage, course) in probes {
        let mut perturbed = policy.clone();
        // Perturbed, so the statistics cover moving as well as standing — a
        // normalizer fitted only to a motionless machine gives every
        // joint-rate input near-zero variance and amplifies its noise.
        for weight in &mut perturbed.theta {
            *weight += 0.05 * rng.normal();
        }
        jobs.push((stage, Terrain::new(course, 1), perturbed));
    }

    let mut warm = ObsNorm::new(n_obs(policy.frame));
    if cfg.backend == JointBackend::Rapier {
        for (stage, terrain, perturbed) in jobs {
            rollout(&perturbed, phys, &terrain, stage, Some(&mut warm));
        }
        return Ok(warm);
    }

    #[cfg(not(feature = "nexus-gpu"))]
    {
        let _ = jobs;
        Err("the Nexus backend requires building hexapod-core with feature `nexus-gpu`".into())
    }

    #[cfg(feature = "nexus-gpu")]
    {
        // A Nexus simulation has one horizon/reward stage per batch. Group the
        // warm-up courses by stage while preserving their deterministic order.
        for &stage in &[Stage::WalkFlat, Stage::Rough, Stage::Gaps, Stage::Jump] {
            let selected = jobs
                .iter()
                .filter(|(job_stage, _, _)| *job_stage == stage)
                .collect::<Vec<_>>();
            if selected.is_empty() {
                continue;
            }
            let policies = selected
                .iter()
                .map(|(_, _, policy)| policy.clone())
                .collect::<Vec<_>>();
            let terrains = selected
                .iter()
                .map(|(_, terrain, _)| terrain.clone())
                .collect::<Vec<_>>();
            for result in rollout_nexus_batch(&policies, phys, &terrains, stage, true, cfg.device)?
            {
                warm.merge(&result.norm);
            }
        }
        Ok(warm)
    }
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
    on_iter: impl FnMut(&Progress, &JointPolicy),
) -> JointPolicy {
    train_curriculum_from(
        JointPolicy::seeded(frame, seed),
        phys,
        cfg,
        budget,
        seed,
        Stage::Stand,
        on_iter,
    )
}

/// Continue a checkpoint from a chosen curriculum stage. Stage is explicit
/// because the checkpoint contains controller state, not training history;
/// guessing from its current scores can send a working locomotion policy back
/// through STAND and optimize its movement away.
pub fn train_curriculum_from(
    mut policy: JointPolicy,
    phys: &Physics,
    cfg: &JointCfg,
    budget: usize,
    seed: u64,
    start_stage: Stage,
    mut on_iter: impl FnMut(&Progress, &JointPolicy),
) -> JointPolicy {
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
    if policy.norm.n < 2.0 {
        let mut warm = warm_normalizer(&policy, phys, cfg, &mut rng)
            .unwrap_or_else(|e| panic!("could not fit joint observation normalizer: {e}"));
        warm.frozen = true;
        policy.norm = warm;
    }

    let first = STAGES.iter().position(|s| *s == start_stage).unwrap_or(0);
    for &stage in STAGES[first..].iter() {
        let mut stage_best = policy.clone();
        let mut stage_best_score = f64::NEG_INFINITY;
        loop {
            let eval = evaluate_on_courses_backend(
                &policy,
                phys,
                stage,
                stage.courses(),
                &eval_seeds,
                cfg,
            )
            .unwrap_or_else(|e| panic!("joint evaluation failed: {e}"));
            let improved = eval.score > stage_best_score;
            if improved {
                stage_best_score = eval.score;
                stage_best = policy.clone();
            }
            let promoted = stage_best_score >= stage.promote_at();
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
                // ARS is not monotonic. A late noisy update must not be the
                // checkpoint merely because it happened last; carry the best
                // validation policy from this stage into the next one (or out
                // of the run when the budget ended).
                policy = stage_best;
                break;
            }
            if !improved {
                // The validation already paid to tell us this trial was
                // harmful. Perturb the best known centre again instead of
                // compounding a noisy downhill step for dozens of updates.
                policy = stage_best.clone();
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
            .map(|t| {
                t.parse::<f64>()
                    .map_err(|e| format!("bad number {t:?}: {e}"))
            })
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
    let norm_n = get("norm_n")?
        .parse::<f64>()
        .map_err(|e| format!("bad norm_n: {e}"))?;
    let mut theta = nums(&get("theta")?)?;
    let mut mean = nums(&get("norm_mean")?)?;
    let mut m2 = nums(&get("norm_m2")?)?;

    // Preserve checkpoints as observations evolve by adding initially unused
    // input columns. Resumed training can learn their weights without changing
    // the policy's pre-migration output.
    let current_obs = n_obs(frame);
    let old_obs = mean.len();
    let added = current_obs.saturating_sub(old_obs);
    let old_theta = old_obs * N_HIDDEN + N_HIDDEN + N_HIDDEN * n_act(frame) + n_act(frame);
    if added == n_act(frame)
        && theta.len() == old_theta
        && mean.len() == old_obs
        && m2.len() == old_obs
    {
        // Executed setpoints were appended after the legacy bias input, so the
        // complete old observation remains an exact prefix.
        let rest = theta.split_off(old_obs * N_HIDDEN);
        let mut migrated = Vec::with_capacity(n_theta(frame));
        for row in theta.chunks_exact(old_obs) {
            migrated.extend_from_slice(row);
            migrated.extend(std::iter::repeat_n(0.0, added));
        }
        migrated.extend(rest);
        theta = migrated;
        mean.extend(std::iter::repeat_n(0.0, added));
        m2.extend(std::iter::repeat_n(norm_n.max(1.0), added));
    } else if (1..=3).contains(&added)
        && theta.len() == old_theta
        && mean.len() == old_obs
        && m2.len() == old_obs
    {
        let task_index = 7 * frame.legs() + 12;
        let rest = theta.split_off(old_obs * N_HIDDEN);
        let mut migrated = Vec::with_capacity(n_theta(frame));
        for row in theta.chunks_exact(old_obs) {
            migrated.extend_from_slice(&row[..task_index]);
            migrated.extend(std::iter::repeat_n(0.0, added));
            migrated.extend_from_slice(&row[task_index..]);
        }
        migrated.extend(rest);
        theta = migrated;
        for offset in 0..added {
            mean.insert(task_index + offset, 0.0);
            // Unit variance avoids saturating new features under a frozen
            // legacy normalizer. Inserted network weights remain zero until
            // retraining.
            m2.insert(task_index + offset, norm_n.max(1.0));
        }
    }
    if theta.len() != n_theta(frame) {
        return Err(format!(
            "checkpoint has {} weights, a {legs}-leg policy needs {}",
            theta.len(),
            n_theta(frame)
        ));
    }
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
            n: norm_n,
            mean,
            m2,
            frozen: true,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_direction_scheduled_rapier_results(
        base_policy: &JointPolicy,
        phys: &Physics,
        stage: Stage,
        cfg: &JointCfg,
        deltas: &[f64],
        plan: &[Vec<Terrain>],
        width: usize,
    ) -> Vec<(usize, f64, f64)> {
        let dirs = plan.len();
        let workers = cfg.workers.max(1).min(dirs);
        let mut results = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for worker in 0..workers {
                handles.push(scope.spawn(move || {
                    let mut out = Vec::new();
                    for direction in (worker..dirs).step_by(workers) {
                        let side = |sign: f64| {
                            let mut policy = base_policy.clone();
                            for (j, weight) in policy.theta.iter_mut().enumerate() {
                                *weight += sign * cfg.sigma * deltas[direction * width + j];
                            }
                            plan[direction]
                                .iter()
                                .map(|terrain| rollout(&policy, phys, terrain, stage, None).score)
                                .sum::<f64>()
                                / plan[direction].len() as f64
                        };
                        out.push((direction, side(1.0), side(-1.0)));
                    }
                    out
                }));
            }
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("legacy benchmark worker panicked"))
                .collect::<Vec<_>>()
        });
        results.sort_by_key(|result| result.0);
        results
    }

    #[cfg(feature = "nexus-gpu")]
    #[test]
    #[ignore = "requires a native GPU adapter"]
    fn nexus_batch_runs_joint_policy_rollouts() {
        let frame = Frame::new(6);
        let policy = JointPolicy::seeded(frame, 1);
        let policies = vec![policy; 2];
        let terrains = vec![
            Terrain::new(Course::Flat, 11),
            Terrain::new(Course::Flat, 12),
        ];
        let results = rollout_nexus_batch(
            &policies,
            &Physics::default(),
            &terrains,
            Stage::Stand,
            true,
            0,
        )
        .expect("Nexus joint rollout");
        assert_eq!(results.len(), 2);
        for result in results {
            assert!(result.rollout.score.is_finite());
            assert!(result.rollout.secs > 0.0);
            assert!(result.norm.n > 0.0);
        }
    }

    #[cfg(feature = "nexus-gpu")]
    #[test]
    #[ignore = "Nexus 0.5 multibody motors do not yet match the Rapier standing reference"]
    fn nexus_seeded_stand_matches_rapier_reference() {
        let frame = Frame::new(6);
        let policy = JointPolicy::seeded(frame, 1);
        let terrain = Terrain::new(Course::Flat, 11);
        let phys = Physics::default();
        let rapier = rollout(&policy, &phys, &terrain, Stage::Stand, None);
        let nexus = rollout_nexus_batch(
            std::slice::from_ref(&policy),
            &phys,
            std::slice::from_ref(&terrain),
            Stage::Stand,
            false,
            0,
        )
        .expect("Nexus standing rollout")[0]
            .rollout;
        assert!(
            (nexus.score - rapier.score).abs() < 0.10 && nexus.fell == rapier.fell,
            "standing parity failed: Rapier score={:.3} fell={}, Nexus score={:.3} fell={}",
            rapier.score,
            rapier.fell,
            nexus.score,
            nexus.fell,
        );
    }

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

    #[test]
    fn observation_exposes_executed_setpoints_and_reset_clears_them() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut env = JointEnv::new(frame, &phys, terrain, Stage::WalkFlat);
        let setpoint_start = n_obs(frame) - n_act(frame);
        assert!(
            env.state()[setpoint_start..]
                .iter()
                .all(|value| *value == 0.0)
        );

        let step = env
            .step(&vec![ACT_RANGE; n_act(frame)])
            .expect("valid joint action");
        let setpoints = &step.observation[setpoint_start..];
        assert!(setpoints.iter().any(|value| value.abs() > 0.01));
        assert!(setpoints.iter().all(|value| value.abs() <= 1.0));

        let reset = env.reset();
        assert!(reset[setpoint_start..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn speed_curriculum_updates_the_command_observation_without_stepping() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut env = JointEnv::new(frame, &phys, terrain, Stage::RunFlat);
        let command_index = 7 * frame.legs() + 11;
        let before = env.summary();
        assert_eq!(env.state()[command_index], Stage::RunFlat.speed());

        let state = env.set_command(0.8).expect("valid curriculum speed");
        assert_eq!(state[command_index], 0.8);
        let after = env.summary();
        assert_eq!(after.secs.to_bits(), before.secs.to_bits());
        assert_eq!(after.distance.to_bits(), before.distance.to_bits());
        assert!(env.set_command(f64::NAN).is_err());
        assert!(env.set_command(-0.1).is_err());
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
            let expect = 12 * legs + 16 + N_TERRAIN_SCAN;
            assert_eq!(n_obs(frame), expect, "{legs} legs");
            assert_eq!(n_act(frame), 3 * legs);
            assert!(n_obs(frame) <= MAX_JOINT_OBS);
            assert!(n_act(frame) <= MAX_JOINT_ACT);
        }
        assert_eq!(Stage::Mixed.courses(), &COURSES);
        assert_eq!(Stage::Mixed.speed_for(Course::Jump), Stage::Jump.speed());
        assert_eq!(Stage::Mixed.speed_for(Course::Flat), Stage::Mixed.speed());
    }

    #[test]
    fn jump_required_signal_separates_parkour_from_stepable_gaps() {
        assert_eq!(jump_required(Course::Gaps), 0.0);
        assert_eq!(jump_required(Course::Flat), 0.0);
        assert_eq!(jump_required(Course::Jump), 1.0);
        assert_eq!(jump_required(Course::Chasm), 1.0);
    }

    #[test]
    fn jump_lip_ranges_supply_takeoff_timing_and_advance_to_a_fresh_trench() {
        let terrain = Terrain::new(Course::Jump, 9);
        let mut pits = terrain
            .obstacles
            .iter()
            .filter(|obstacle| obstacle.top < -0.1)
            .collect::<Vec<_>>();
        pits.sort_by(|a, b| a.z0.total_cmp(&b.z0));
        let first = pits.first().expect("JUMP course has no trench");
        let before = jump_lip_distances(&terrain, first.z0 - 1.0);
        assert!((before[0] - 1.0).abs() < 1e-9);
        assert!(before[1] > before[0]);
        let over = jump_lip_distances(&terrain, 0.5 * (first.z0 + first.z1));
        assert!(
            over[0] < 0.0 && over[1] > 0.0,
            "over-trench ranges {over:?}"
        );
        if let Some(second) = pits.get(1) {
            let fresh = jump_lip_distances(&terrain, first.z1 + 0.3);
            assert!((fresh[0] - (second.z0 - first.z1 - 0.3)).abs() < 1e-9);
            assert!(
                fresh[0] > 0.0,
                "next trench was not freshly armed: {fresh:?}"
            );
        }
        assert_eq!(
            jump_lip_distances(&Terrain::new(Course::Gaps, 9), 0.0),
            [0.0, 0.0]
        );
    }

    #[test]
    fn cloned_rapier_environment_is_an_exact_reset() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 17);
        let policy = JointPolicy::seeded(frame, 5);
        let env = RapierEnv::new(frame, &phys, &terrain);
        let a = rollout_in_env(&policy, &phys, &terrain, Stage::Stand, None, env.clone());
        let b = rollout_in_env(&policy, &phys, &terrain, Stage::Stand, None, env);
        assert_eq!(a.score.to_bits(), b.score.to_bits());
        assert_eq!(a.distance.to_bits(), b.distance.to_bits());
        assert_eq!(a.secs.to_bits(), b.secs.to_bits());
        assert_eq!(a.support.to_bits(), b.support.to_bits());
        assert_eq!(a.air.to_bits(), b.air.to_bits());
        assert_eq!(a.fell, b.fell);
        assert_eq!(a.reached, b.reached);
        assert_eq!(a.finished, b.finished);
        assert_eq!(a.completed, b.completed);
    }

    #[test]
    fn stepwise_rewards_reproduce_the_evaluated_episode_and_reset_exactly() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 17);
        let policy = JointPolicy::seeded(frame, 5);
        let expected = rollout(&policy, &phys, &terrain, Stage::Stand, None);
        let mut env = JointEnv::new(frame, &phys, terrain, Stage::Stand);
        let initial = env.state().to_vec();
        let mut action = vec![0.0; n_act(frame)];
        let mut reward_sum = 0.0;
        let mut learning_reward_sum = 0.0;
        let mut final_step = None;

        while !env.is_done() {
            let mut observation = env.state().to_vec();
            policy.norm.apply(&mut observation);
            policy.act(&observation, &mut action);
            let step = env.step(&action).expect("valid joint action");
            assert_eq!(step.observation.len(), n_obs(frame));
            reward_sum += step.reward;
            assert!(step.learning_reward >= 0.0);
            learning_reward_sum += step.learning_reward;
            final_step = Some(step);
        }

        let actual = env.summary();
        let final_step = final_step.expect("episode produced no transition");
        assert!(final_step.truncated);
        assert!(!final_step.terminated);
        assert!((reward_sum - actual.score).abs() < 1e-12);
        // The dense learning reward is deliberately no longer the score: it is
        // paid per step so Q lands in a range the critic can resolve, while
        // `score` stays horizon-normalized so checkpoints compare across
        // horizons. Proportionality is the invariant that survives.
        let ticks = actual.secs / DT;
        assert!(
            (learning_reward_sum * DT / env.horizon() - actual.score).abs() < 1e-9,
            "learning reward {learning_reward_sum} over {ticks} ticks did not \
             telescope to score {}",
            actual.score
        );
        assert_eq!(actual.score.to_bits(), expected.score.to_bits());
        assert_eq!(actual.distance.to_bits(), expected.distance.to_bits());
        assert_eq!(actual.secs.to_bits(), expected.secs.to_bits());
        assert_eq!(actual.support.to_bits(), expected.support.to_bits());
        assert_eq!(actual.air.to_bits(), expected.air.to_bits());

        let reset = env.reset();
        assert_eq!(reset.len(), initial.len());
        for (a, b) in reset.iter().zip(initial) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn value_learning_gets_a_dense_signal_before_net_progress_is_earned() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 17);
        let mut env = JointEnv::new(frame, &phys, terrain, Stage::WalkFlat);
        let action = vec![0.0; n_act(frame)];
        let mut exact_reward = 0.0;
        let mut learning_reward = 0.0;

        while !env.is_done() {
            let step = env.step(&action).expect("valid standing action");
            exact_reward += step.reward;
            learning_reward += step.learning_reward;
        }

        let rollout = env.summary();
        assert!((exact_reward - rollout.score).abs() < 1e-12);
        assert!(
            learning_reward > exact_reward + 0.05,
            "dense reward {learning_reward:.3} did not guide beyond exact score {exact_reward:.3}"
        );
        assert!(
            rollout.distance < 0.25 * Stage::WalkFlat.speed() * Stage::WalkFlat.horizon(),
            "standing unexpectedly earned the full net-progress gate"
        );
    }

    /// Support is a multiplier everywhere and a subtraction nowhere, so no
    /// amount of speed buys back a projectile trajectory and no episode can
    /// earn a negative reward for trying one.
    #[test]
    fn support_gates_a_projectile_without_ever_paying_a_negative_reward() {
        assert_eq!(support_gate(3.0, 6), 1.0);
        assert!(support_gate(2.0, 6) > 0.8);
        assert!(support_gate(0.5, 6) < 0.1);

        // Unbounded speed cannot outrun the gate: a hexapod on half a foot
        // going ten times as fast still scores below a tripod at walking pace.
        let tick = |speed: f64, duty: f64| {
            reward(
                Stage::Rough, &[0.0, 0.0, speed], 0.0, 0.0, 0.0, 0.0,
                duty.round() as usize, duty, 6, 0.0,
            )
        };
        let projectile = tick(20.0, 0.5);
        let tripod = tick(2.0, 3.0);
        assert!(
            projectile < tripod,
            "a projectile at 10x the speed scored {projectile} against {tripod}"
        );

        // And nothing anywhere is negative, which is what keeps falling over
        // from being an escape: `terminated` cuts the bootstrap, so a negative
        // per-step reward makes ending the episode the profitable move.
        for duty in [0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 6.0] {
            for speed in [-5.0, 0.0, 1.0, 20.0] {
                assert!(tick(speed, duty) >= 0.0, "reward({speed}, {duty}) went negative");
            }
        }
    }

    #[test]
    fn stepwise_environment_rejects_malformed_actions_and_steps_after_done() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let mut env = JointEnv::new(frame, &phys, Terrain::new(Course::Flat, 3), Stage::Stand);
        let err = env.step(&[0.0]).expect_err("short action was accepted");
        assert!(err.contains("width"), "unhelpful action error: {err}");

        let action = vec![0.0; n_act(frame)];
        while !env.is_done() {
            env.step(&action).expect("valid action");
        }
        let err = env
            .step(&action)
            .expect_err("finished episode accepted another step");
        assert!(err.contains("reset"), "unhelpful terminal error: {err}");
    }

    #[test]
    fn replay_allocates_gradually_overwrites_oldest_and_samples_deterministically() {
        let mut replay = JointReplay::new(3, 2, 1).expect("replay");
        assert_eq!(replay.payload_bytes(), 0);
        for i in 1..=3 {
            replay
                .push(
                    &[i as f64, i as f64 + 0.5],
                    &[i as f64 * 0.1],
                    i as f64,
                    &[i as f64 + 1.0, i as f64 + 1.5],
                    i == 2,
                    i == 3,
                )
                .expect("transition");
        }
        assert_eq!(replay.len(), 3);
        assert_eq!(replay.payload_bytes(), 3 * (6 * size_of::<f32>() + 2));

        replay
            .push(&[4.0, 4.5], &[0.4], 4.0, &[5.0, 5.5], false, false)
            .expect("overwrite");
        assert_eq!(replay.rewards, vec![4.0, 2.0, 3.0]);
        assert_eq!(replay.terminated, vec![false, true, false]);
        assert_eq!(replay.truncated, vec![false, false, true]);

        let mut a_rng = Rng::new(91);
        let mut b_rng = Rng::new(91);
        let a = replay.sample(16, &mut a_rng).expect("sample A");
        let b = replay.sample(16, &mut b_rng).expect("sample B");
        assert_eq!(a.observations, b.observations);
        assert_eq!(a.actions, b.actions);
        assert_eq!(a.rewards, b.rewards);
        assert_eq!(a.next_observations, b.next_observations);
        assert_eq!(a.terminated, b.terminated);
        assert_eq!(a.truncated, b.truncated);
    }

    #[test]
    fn replay_refuses_bad_shapes_non_finite_values_and_empty_samples() {
        assert!(JointReplay::new(0, 2, 1).is_err());
        let mut replay = JointReplay::new(4, 2, 1).expect("replay");
        assert!(replay.sample(1, &mut Rng::new(1)).is_err());
        assert!(
            replay
                .push(&[0.0], &[0.0], 0.0, &[0.0, 0.0], false, false)
                .is_err()
        );
        assert!(
            replay
                .push(&[0.0, f64::NAN], &[0.0], 0.0, &[0.0, 0.0], false, false)
                .is_err()
        );
    }

    #[test]
    fn parallel_rapier_evaluation_matches_scalar_order_exactly() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let policy = JointPolicy::seeded(frame, 8);
        let courses = [Course::Flat, Course::Gaps];
        let seeds = [3, 4];
        let scalar = evaluate_on_courses(&policy, &phys, Stage::Stand, &courses, &seeds);
        let cfg = JointCfg {
            workers: 4,
            batch_envs: 4,
            ..JointCfg::default()
        };
        let parallel =
            evaluate_on_courses_backend(&policy, &phys, Stage::Stand, &courses, &seeds, &cfg)
                .expect("parallel Rapier evaluation");
        assert_eq!(scalar.score.to_bits(), parallel.score.to_bits());
        assert_eq!(scalar.distance.to_bits(), parallel.distance.to_bits());
        assert_eq!(scalar.secs.to_bits(), parallel.secs.to_bits());
        assert_eq!(scalar.support.to_bits(), parallel.support.to_bits());
        assert_eq!(scalar.air.to_bits(), parallel.air.to_bits());
        assert_eq!(scalar.fell, parallel.fell);
        assert_eq!(scalar.reached, parallel.reached);
        assert_eq!(scalar.finished, parallel.finished);
        assert_eq!(scalar.completed, parallel.completed);
    }

    #[test]
    fn route_state_only_completes_after_ordered_waypoints() {
        let terrain = Terrain::new(Course::Flat, 1);
        let mut facing_away = RouteState::default();
        facing_away.update(&terrain, [0.0, 1.0, 0.0], std::f64::consts::PI);
        assert!(
            facing_away.bearing.abs() > 3.0,
            "a waypoint behind the body looked ahead: {:.3} rad",
            facing_away.bearing
        );

        let mut route = RouteState::default();
        for &w in &terrain.waypoints {
            route.update(&terrain, [w[0], 1.0, w[1]], 0.0);
        }
        assert!(route.finished);
        assert_eq!(route.reached, terrain.waypoints.len());

        let first = terrain.waypoints[0];
        let mut missed = RouteState::default();
        missed.update(
            &terrain,
            [first[0] + WAYPOINT_R + 1.0, 1.0, first[1] + 2.0],
            0.0,
        );
        assert_eq!(missed.reached, 0, "passing beside a waypoint earned credit");
        assert_eq!(missed.wp, 1, "a missed waypoint should not trap the route");
    }

    #[test]
    fn forward_scan_sees_a_trench_before_contact() {
        let terrain = Terrain::new(Course::Gaps, 3);
        let pit = terrain
            .obstacles
            .iter()
            .find(|o| o.top < 0.0)
            .expect("GAPS course has no trench");
        let pos = [0.0, 1.0, pit.z0 - 1.0];
        let mut scan = [0.0; N_TERRAIN_SCAN];
        terrain_scan(&terrain, pos, 0.0, 0.0, &mut scan);
        assert!(
            scan.iter().any(|h| *h < -0.1),
            "the trench was invisible in the forward scan: {scan:?}"
        );
    }

    #[test]
    fn speed_away_from_the_waypoint_cannot_improve_tracking_reward() {
        let body_v = [0.0, 0.0, 0.8];
        let toward = reward(
            Stage::WalkFlat,
            &body_v,
            0.0,
            0.0,
            1.0,
            1.0,
            3,
            3.0,
            6,
            0.0,
        );
        let away = reward(
            Stage::WalkFlat,
            &body_v,
            0.0,
            0.0,
            1.0,
            1.0,
            3,
            3.0,
            6,
            std::f64::consts::PI,
        );
        // A ratio rather than an absolute gap: the speed term is linear now, so
        // an absolute margin only says how far 0.8 is from REFERENCE_SPEED.
        assert!(
            toward > away * 4.0,
            "toward {toward:.3}, away {away:.3}"
        );
        assert_eq!(
            episode_score(Stage::WalkFlat, 0.9, -0.5, 3.0, 6, 0.0, false, Stage::WalkFlat.horizon()),
            0.0,
            "an episode that moved backward kept shaping reward"
        );
        let stable = episode_score(Stage::WalkFlat, 0.9, 1.0, 2.1, 6, 0.0, false, Stage::WalkFlat.horizon());
        let hopping = episode_score(Stage::WalkFlat, 0.9, 1.0, 0.7, 6, 0.0, false, Stage::WalkFlat.horizon());
        let parked = episode_score(Stage::WalkFlat, 0.9, 0.0, 6.0, 6, 0.0, false, Stage::WalkFlat.horizon());
        assert_eq!(parked, 0.0, "safe standing should be a neutral fallback");
        // Hopping used to be required to score *below* standing, which took a
        // subtracted support deficit -- the one term that could make a reward
        // negative, and so the one that made falling over an escape from
        // accruing more of it. What actually has to hold is that hopping is
        // dominated, not that it is punished: `enough_feet` already caps it at
        // a quarter of a tripod's per-tick reward, and here at a ninth of a
        // tripod's episode score. Standing < hopping < walking is monotone in
        // usefulness and leaves no valley between standing and a gait, which
        // is the trap the additive posture terms fell into.
        assert!(
            parked <= hopping && hopping < stable,
            "ordering broke: parked {parked:.3}, hopping {hopping:.3}, stable {stable:.3}"
        );
        assert!(
            stable > hopping * 5.0,
            "sustained hopping was not dominated: stable {stable:.3}, hopping {hopping:.3}"
        );
    }

    /// With motors this strong, standing is free: the seeded policy already
    /// saturates the stage's reward. So the curriculum has to promote past it
    /// without training on it — a saturated objective is a maximum, and ARS
    /// stepping away from one took a perfect 1.000 down to 0.236 in six
    /// iterations when the stage was trained before being checked.
    /// Sometimes slowing down is how you go faster overall — braking into a
    /// trench lip to clear it, the way braking into a corner is how you leave
    /// it quicker. A per-tick reward that is *concave* in speed cannot
    /// represent that: by Jensen's inequality it strictly prefers one steady
    /// pace to any varying one covering the same ground. The saturating curve
    /// this replaced paid 1.245 for two ticks at 1.0 against 1.000 for the same
    /// distance taken as 2.0 and a standstill — a 24.5% tax on braking.
    ///
    /// Linear is the only shape with no such opinion.
    #[test]
    fn where_the_speed_went_does_not_change_the_reward() {
        let at = |speed: f64| {
            reward(
                Stage::WalkFlat,
                &[0.0, 0.0, speed],
                0.0,
                0.0,
                1.0,
                1.0,
                3,
                3.0,
                6,
                0.0,
            )
        };
        // Equal ground, three different profiles.
        let steady = at(1.0) + at(1.0) + at(1.0) + at(1.0);
        let burst = at(4.0) + at(0.0) + at(0.0) + at(0.0);
        let ramped = at(0.5) + at(1.5) + at(1.5) + at(0.5);
        assert!(
            (steady - burst).abs() < 1e-9,
            "a burst scored {burst} against {steady} for the same distance"
        );
        assert!(
            (steady - ramped).abs() < 1e-9,
            "a ramp scored {ramped} against {steady} for the same distance"
        );

        // Which leaves covering more ground in the same time as the only way to
        // score more. Note the comparison is at equal tick counts: `aim` and
        // `gait` are paid per tick, so covering the same ground in fewer ticks
        // banks less of them — which is exactly the hole [`FINISH_BONUS`]
        // fills, since arriving is what ends an episode early.
        let faster = at(2.0) + at(2.0) + at(2.0) + at(2.0);
        assert!(faster > steady, "{faster} should beat {steady} over four ticks");
    }

    /// Reaching the target used to cost the policy return. The per-tick shaping
    /// sums to 1.0 over a full episode, and arriving *terminates* the episode,
    /// so finishing halfway through the horizon banked 0.5 against 1.0 for
    /// never arriving — and `terminated` cuts the bootstrap, so that was the
    /// whole return. Dawdling beat arriving, and every route term lived in
    /// `episode_score`, which the gradient never sees.
    #[test]
    fn arriving_beats_running_out_the_clock() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        // Course and stage are irrelevant here; only the arithmetic is.
        let mut env = JointEnv::new(frame, &phys, Terrain::new(Course::Flat, 3), Stage::Gaps);
        env.set_horizon(8.0);

        // The dense term is paid per step, so what an early finish gives up is
        // bounded by the discounted sum, not by the horizon: at most
        // per_step/(1-gamma). A good gait banks about 0.15 a tick and
        // DECIMATION ticks a step, against the 0.99 the trainer discounts at.
        let dawdled_forever = 0.15 * DECIMATION as f64 / (1.0 - 0.99);

        // Before a bar exists arriving pays the average rate. Zero here meant
        // the episode that sets the bar was never paid for setting it.
        assert_eq!(env.promptness(4.0), 0.5);

        env.set_finish_bar(8.0);
        // Arriving exactly at the bar covers that bound in full; beating it
        // pays up to double.
        let at_bar = FINISH_BONUS * env.promptness(8.0);
        assert!(
            at_bar >= dawdled_forever,
            "bonus at the bar {at_bar} must cover the {dawdled_forever} an \
             early finish gives up"
        );

        // Sooner is strictly better, all the way down.
        let mut previous = f64::NEG_INFINITY;
        for tenths in (1..=160).rev() {
            let here = env.promptness(tenths as f64 * 0.1);
            assert!(here > previous, "arriving sooner scored less: {here} <= {previous}");
            previous = here;
        }
        assert!(env.promptness(1.0) > 0.9, "well under the bar should be near full");
        assert!(env.promptness(16.0) < 0.1, "well over the bar should be near nothing");
    }

    /// The task is to reach a point in space as soon as possible, so nothing in
    /// the objective may prefer a slower machine. The old curve did: it fell
    /// off above the commanded speed, and a run that saturated it spent 13M
    /// transitions trading stride for planted feet because going faster was
    /// worth nothing.
    #[test]
    fn going_faster_toward_the_target_is_never_worth_less() {
        // Level, at ride height, three feet down, aimed straight at the
        // waypoint: everything except speed held at its best.
        let at = |speed: f64| {
            reward(
                Stage::WalkFlat,
                &[0.0, 0.0, speed],
                0.0,
                0.0,
                1.0,
                1.0,
                3,
                3.0,
                6,
                0.0,
            )
        };

        let reference = REFERENCE_SPEED;
        // The one scale the reward knows, and hitting it is exactly 1.0 so a
        // stage score still reads as a fraction. No stage carries a speed of
        // its own any more.
        assert!(
            (at(reference) - 1.0).abs() < 1e-9,
            "reference speed should score 1.0, got {}",
            at(reference)
        );

        // Strictly increasing all the way out, including well past the
        // reference where the old curve was falling.
        // Standing still keeps only the speed-independent floor: aimed the
        // right way in a tripod pose. The episode-level progress gate is what
        // makes that worth nothing over a whole rollout.
        let mut previous = at(0.0);
        assert!(previous > 0.0 && previous < 0.3, "standing floor was {previous}");
        assert_eq!(at(-5.0), previous, "backwards is worth no more than still");
        for step in 1..=40 {
            let here = at(reference * step as f64 * 0.25);
            assert!(
                here > previous,
                "reward fell going from {previous} to {here} at {}x reference",
                step as f64 * 0.25
            );
            previous = here;
        }
        // Unbounded on purpose: no speed is the fastest worth having.
        assert!(at(4.0 * reference) > at(2.0 * reference) * 1.5);

    }

    /// Reaching the target was a flat bonus however long it took, so a
    /// controller that strolled to the finish scored exactly like one that ran.
    #[test]
    fn reaching_the_target_sooner_scores_higher() {
        let horizon = Stage::Gaps.horizon();
        let score = |secs: f64, completed: bool| {
            episode_score(Stage::Gaps, 0.5, 10.0, 3.0, 6, 1.0, completed, secs)
        };
        let prompt = score(0.1 * horizon, true);
        let slow = score(horizon, true);
        let unfinished = score(horizon, false);
        assert!(
            prompt > slow,
            "arriving early ({prompt}) must beat arriving late ({slow})"
        );
        assert!(slow > unfinished, "arriving at all must beat not arriving");
        // The late arrival is worth what a finish was worth before, so the
        // change only ever adds.
        assert!((slow - unfinished - 1.0).abs() < 1e-9);
    }

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
        let mut seeded = JointPolicy::seeded(frame, 7);
        // This test is about promotion, not fitting observation statistics.
        // A frozen identity-like normaliser keeps a full multi-terrain Rapier
        // warmup out of the ordinary unit suite.
        seeded.norm.n = 2.0;
        seeded.norm.frozen = true;
        let before = evaluate(&seeded, &phys, Stage::Stand, &[101]).score;
        assert!(
            before >= Stage::Stand.promote_at(),
            "standing is not already solved: {before:.3}"
        );

        // One iteration of budget: enough to promote through STAND, not enough
        // to get anywhere on the stage after it. Record the weights at every
        // stage check, so we can see whether passing through STAND cost any.
        let mut seen: Vec<(Stage, bool, Vec<f64>)> = Vec::new();
        train_curriculum_from(seeded.clone(), &phys, &cfg, 1, 7, Stage::Stand, |p, pol| {
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

    #[test]
    fn rapier_ars_is_deterministic_across_worker_counts() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let mut serial = JointPolicy::seeded(frame, 23);
        let mut parallel = serial.clone();
        serial.norm.n = 2.0;
        serial.norm.frozen = true;
        parallel.norm = serial.norm.clone();
        let mut serial_rng = Rng::new(29);
        let mut parallel_rng = Rng::new(29);
        let serial_cfg = JointCfg {
            dirs: 2,
            top: 1,
            scenarios: 2,
            workers: 1,
            batch_envs: 1,
            ..JointCfg::default()
        };
        let parallel_cfg = JointCfg {
            workers: 4,
            batch_envs: 4,
            ..serial_cfg
        };
        let serial_score = iterate(
            &mut serial,
            &phys,
            Stage::WalkFlat,
            &serial_cfg,
            &mut serial_rng,
            0,
        );
        let parallel_score = iterate(
            &mut parallel,
            &phys,
            Stage::WalkFlat,
            &parallel_cfg,
            &mut parallel_rng,
            0,
        );
        assert_eq!(serial_score.to_bits(), parallel_score.to_bits());
        assert_eq!(serial.theta, parallel.theta);
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

    #[test]
    fn checkpoint_without_course_aware_jump_features_is_migrated_losslessly() {
        let frame = Frame::new(6);
        let mut expected = JointPolicy::seeded(frame, 12);
        expected.norm.n = 25.0;
        let task_index = 7 * frame.legs() + 12;
        for row in expected.theta[..n_obs(frame) * N_HIDDEN].chunks_exact_mut(n_obs(frame)) {
            row[task_index..task_index + 3].fill(0.0);
        }

        let mut legacy = expected.clone();
        let rest = legacy.theta.split_off(n_obs(frame) * N_HIDDEN);
        let mut old_theta = Vec::with_capacity(legacy.theta.len() - N_HIDDEN + rest.len());
        for row in legacy.theta.chunks_exact(n_obs(frame)) {
            old_theta.extend_from_slice(&row[..task_index]);
            old_theta.extend_from_slice(&row[task_index + 3..]);
        }
        old_theta.extend(rest);
        legacy.theta = old_theta;
        legacy.norm.mean.drain(task_index..task_index + 3);
        legacy.norm.m2.drain(task_index..task_index + 3);

        let migrated = from_text(&to_text(&legacy)).expect("migrate task-bit checkpoint");
        assert_eq!(migrated.theta, expected.theta);
        assert_eq!(migrated.norm.mean.len(), n_obs(frame));
        assert_eq!(&migrated.norm.mean[task_index..task_index + 3], &[0.0; 3]);
        assert_eq!(
            &migrated.norm.m2[task_index..task_index + 3],
            &[legacy.norm.n; 3]
        );
    }

    #[test]
    fn checkpoint_without_executed_setpoints_is_migrated_losslessly() {
        let frame = Frame::new(6);
        let mut expected = JointPolicy::seeded(frame, 13);
        expected.norm.n = 25.0;
        let old_obs = n_obs(frame) - n_act(frame);
        for row in expected.theta[..n_obs(frame) * N_HIDDEN].chunks_exact_mut(n_obs(frame)) {
            row[old_obs..].fill(0.0);
        }

        let mut legacy = expected.clone();
        let rest = legacy.theta.split_off(n_obs(frame) * N_HIDDEN);
        let mut old_theta = Vec::with_capacity(old_obs * N_HIDDEN + rest.len());
        for row in legacy.theta.chunks_exact(n_obs(frame)) {
            old_theta.extend_from_slice(&row[..old_obs]);
        }
        old_theta.extend(rest);
        legacy.theta = old_theta;
        legacy.norm.mean.truncate(old_obs);
        legacy.norm.m2.truncate(old_obs);

        let migrated = from_text(&to_text(&legacy)).expect("migrate setpoint observations");
        assert_eq!(migrated.theta, expected.theta);
        assert_eq!(migrated.norm.mean.len(), n_obs(frame));
        assert_eq!(&migrated.norm.mean[old_obs..], &vec![0.0; n_act(frame)]);
        assert_eq!(
            &migrated.norm.m2[old_obs..],
            &vec![legacy.norm.n; n_act(frame)]
        );
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
    #[ignore = "manual performance measurement; not a correctness test"]
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
        println!(
            "  rollout ({horizon}s sim) {:>8.1} ms  -> {:.1}x realtime",
            roll * 1e3,
            horizon / roll
        );
        println!("  setup share      {:>8.1} %", 100.0 * setup / roll);

        // One iteration's worth, serial, for comparison against the observed
        // wall time of the real (threaded) loop.
        let cfg = JointCfg {
            dirs: 16,
            top: 5,
            scenarios: 1,
            ..JointCfg::default()
        };
        let serial = roll * (cfg.dirs * 2 * cfg.scenarios) as f64;
        println!("  16 dirs serial   {:>8.2} s", serial);
        println!(
            "  cores            {:>8?}",
            std::thread::available_parallelism()
        );

        let mut p2 = JointPolicy::seeded(frame, 1);
        let mut rng = Rng::new(1);
        let t2 = std::time::Instant::now();
        iterate(&mut p2, &phys, Stage::WalkFlat, &cfg, &mut rng, 0);
        println!("  iterate() actual {:>8.2} s", t2.elapsed().as_secs_f64());

        let t3 = std::time::Instant::now();
        evaluate(&p2, &phys, Stage::WalkFlat, &[101, 202, 303]);
        println!("  evaluate() 3 sds {:>8.2} s", t3.elapsed().as_secs_f64());
    }

    #[test]
    #[ignore = "manual release-mode throughput comparison"]
    fn bench_reusable_parallel_rapier_batch() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let mut policy = JointPolicy::seeded(frame, 41);
        policy.norm.n = 2.0;
        policy.norm.frozen = true;
        let cfg = JointCfg {
            dirs: 16,
            top: 3,
            scenarios: 2,
            workers: 12,
            batch_envs: 16,
            ..JointCfg::default()
        };
        let width = n_theta(frame);
        let mut rng = Rng::new(43);
        let deltas = (0..cfg.dirs * width)
            .map(|_| rng.normal())
            .collect::<Vec<_>>();
        let plan = (0..cfg.dirs)
            .map(|direction| {
                (0..cfg.scenarios)
                    .map(|scenario| {
                        Terrain::new(
                            Course::Flat,
                            1 + (direction * cfg.scenarios + scenario) as u64,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let t0 = std::time::Instant::now();
        let legacy = legacy_direction_scheduled_rapier_results(
            &policy,
            &phys,
            Stage::Stand,
            &cfg,
            &deltas,
            &plan,
            width,
        );
        let legacy_secs = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let batched = rapier_ars_results(&policy, &phys, Stage::Stand, &cfg, &deltas, &plan, width);
        let batched_secs = t1.elapsed().as_secs_f64();
        for (old, new) in legacy.iter().zip(&batched) {
            assert_eq!(old.0, new.0);
            assert_eq!(old.1.to_bits(), new.1.to_bits());
            assert_eq!(old.2.to_bits(), new.2.to_bits());
        }
        println!("legacy direction scheduler  {legacy_secs:.3} s");
        println!("reusable scenario batch     {batched_secs:.3} s");
        println!(
            "speedup                     {:.2}x",
            legacy_secs / batched_secs
        );
    }

    /// How big an action does a sigma-sized weight perturbation actually
    /// produce? With the output layer seeded to zero the answer can be "a
    /// degree and a half", which is not exploration — it is a policy that
    /// cannot discover locomotion because it never tries any.
    #[test]
    #[ignore = "manual exploration sweep; not a correctness test"]
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

