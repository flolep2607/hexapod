//! Empty-field one-leg drill: friction holds the plants, nothing is welded.
//!
//! Stance motors hold their settled angles and yield under load. They are not
//! walked. Only the free foot is driven, along one eased world chord.

use crate::dynamics::Physics;
use crate::math::{body_to_world, convex_hull_xz, hypot2, lerp, polygon_margin, Rng, V3, MAX_HULL};
use crate::plant::ArticulatedPlant;
use crate::policy::Gait;
use crate::robot::{
    clamp_joints, fk_body, solve_ik, to_body, Frame, FOOT_R, MAX_LEGS, REACH_MAX, REACH_MIN,
};
use crate::crawl_rl::{CrawlAction, CrawlObs, CrawlPolicy};
use crate::sim::{Cmd, DT};
use crate::terrain::{Course, Terrain};

const SETTLE: f64 = 0.70;
const LIFT_T: f64 = 0.70;
const SHIFT_T: f64 = 1.00;
const PLACE_T: f64 = 0.70;
const PAUSE: f64 = 0.55;
const SWING_T: f64 = LIFT_T + SHIFT_T + PLACE_T;
/// High enough that a watching eye can tell the sole left the floor.
const LIFT_H: f64 = 0.22;
/// Ride height and radial plant, independent of any walk gait.
const RIDE: f64 = 0.88;
/// Crawl: how far ahead of its neutral stance the free foot will try to plant,
/// metres. A ceiling, not a setpoint — the leg's workspace usually binds first.
const STEP: f64 = 0.50;
/// Lateral share of a STEP given to a full turn command.
const SIDE: f64 = 0.2;
/// Support margin the commanded COM leans up to, metres. The lean stops here,
/// so this is the whole stability budget for a swing: raise it for a machine
/// that wobbles, lower it to walk closer to the edge.
const MARGIN: f64 = 0.12;
/// Below this margin a leg is not liftable at all and the crawl tries the next.
const VETO: f64 = 0.02;
/// How far the commanded body may lead the real chassis. This is the crawl's
/// drive — the legs push against it — so raising it walks harder. It is also
/// the only strain the machine is allowed to store, so it is what a stopped
/// machine has to shed before it stands still.
const LEAD_MAX: f64 = 0.06;
/// How fast the commanded chassis rides to its goal, m/s. This is the crawl's
/// throttle: the legs push against the lead the command holds over the real
/// chassis, so raising it walks faster and skates the plants harder.
const SHIFT_V: f64 = 0.10;
/// Crawl: commanded yaw change after each plant at full turn, radians.
const YAW_STEP: f64 = 0.10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Settle,
    Lift,
    Shift,
    Place,
    Pause,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Settle => "settle",
            Phase::Lift => "lift",
            Phase::Shift => "shift",
            Phase::Place => "place",
            Phase::Pause => "pause",
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Phase::Settle => 0,
            Phase::Lift => 1,
            Phase::Shift => 2,
            Phase::Place => 3,
            Phase::Pause => 4,
        }
    }

    /// The free foot is off the floor, or coming down onto a new plant.
    pub fn swinging(self) -> bool {
        matches!(self, Phase::Lift | Phase::Shift | Phase::Place)
    }
}

/// Stand, then repeatedly relocate one foot inside its workspace.
pub struct OneLegDrill {
    pub frame: Frame,
    pub stand: Gait,
    pub phys: Physics,
    pub plant: ArticulatedPlant,
    /// Last planted joints. Stance motors hold these and yield; they are not walked.
    pub q_hold: [[f64; 3]; MAX_LEGS],
    /// Last planted body-frame foot, used to sample the next reachable plant.
    pub hold_body: [V3; MAX_LEGS],
    /// World plants. Grey X on the canvas; stance motors are not IK'd here.
    pub origin_world: [V3; MAX_LEGS],
    pub origin_pos: V3,
    pub moving: usize,
    /// World lift-off of the free foot, frozen at [`Self::begin_move`].
    pub from: V3,
    /// World landing, frozen at [`Self::begin_move`]. The canvas mark is this.
    pub dest: V3,
    /// Body-frame sample that produced `dest`, so the next plant can avoid it.
    pub dest_body: V3,
    /// Lagged yaw-only chassis the IK uses. Pitch/roll stay out: feeding them
    /// back into the swing retargets the free foot and tips the machine.
    pub ik_pos: V3,
    /// Where `ik_pos` is riding to. The crawl aims this once per plant; the
    /// chassis travels to it at [`SHIFT_V`] over the whole swing.
    pub ik_goal: [f64; 2],
    pub ik_yaw: f64,
    pub phase: Phase,
    pub phase_t: f64,
    pub move_i: usize,
    pub t: f64,
    rng: Rng,
    n: usize,
    fixed: Option<usize>,
    terrain: Terrain,
    /// Walk: dest follows `cmd`, chassis shifts after each plant. Drill: random relocate.
    pub crawl: bool,
    /// Learned crawl controller. `None` runs the hand-written constants, which
    /// is what [`CrawlPolicy::seeded`] reproduces, so the two are comparable.
    pub policy: Option<CrawlPolicy>,
    /// Decided once per plant and held for that move.
    act: CrawlAction,
    cmd: Cmd,
}

#[derive(Clone, Copy, Debug)]
pub struct OneLegSample {
    pub t: f64,
    pub move_i: usize,
    pub moving: usize,
    pub phase: Phase,
    pub phase_u: f64,
    pub pos: V3,
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
    pub vel: V3,
    pub cmd_body: V3,
    pub cmd_world: V3,
    pub foot_body: V3,
    pub foot_world: V3,
    pub dest_body: V3,
    pub dest_world: V3,
    pub reach_err: f64,
    pub chassis_xz: f64,
    pub stance_drift: f64,
    pub moving_travel: f64,
    pub slip: f64,
    /// Moving foot centre height minus the sole radius. Negative means the
    /// ball is in the floor; near zero is a scrape; a real lift is > 0.15 m.
    pub foot_clear: f64,
    pub fallen: bool,
}

impl OneLegDrill {
    /// Empty-field drill: random relocates, chassis stays.
    pub fn spawn(frame: Frame, phys: &Physics, seed: u64) -> Self {
        Self::spawn_on(frame, phys, &Terrain::new(Course::Flat, seed), seed, false)
    }

    /// Same plant on `terrain`. `crawl` walks by planting along the command.
    pub fn spawn_on(
        frame: Frame,
        phys: &Physics,
        terrain: &Terrain,
        seed: u64,
        crawl: bool,
    ) -> Self {
        let stand = stand_pose(frame);
        let mut plant = ArticulatedPlant::standing(frame, &stand, phys, terrain);
        let n = frame.legs();
        let mut q_hold = [[0.0; 3]; MAX_LEGS];
        let mut hold_body = [[0.0; 3]; MAX_LEGS];
        for i in 0..n {
            hold_body[i] = standing_foot(frame, &stand, i);
            q_hold[i] = solve_ik(frame, i, hold_body[i]).q;
            clamp_joints(&mut q_hold[i]);
        }
        for _ in 0..((SETTLE / DT) as usize) {
            plant.drive(&q_hold, phys, DT);
            substep(&mut plant, DT);
        }
        let (origin_pos, origin_yaw, _, _) = plant.chassis_pose();
        let mut origin_world = [[0.0; 3]; MAX_LEGS];
        for i in 0..n {
            origin_world[i] = plant.leg_joints_world(i)[3];
            q_hold[i] = plant.leg_q(i);
            hold_body[i] = to_body(origin_world[i], origin_pos, origin_yaw, 0.0, 0.0);
        }
        plant.lock(&[true; MAX_LEGS], phys);
        OneLegDrill {
            frame,
            stand,
            phys: *phys,
            plant,
            q_hold,
            hold_body,
            origin_world,
            origin_pos,
            moving: 0,
            from: origin_world[0],
            dest: origin_world[0],
            dest_body: hold_body[0],
            ik_pos: origin_pos,
            ik_goal: [origin_pos[0], origin_pos[2]],
            ik_yaw: origin_yaw,
            phase: Phase::Settle,
            phase_t: 0.0,
            move_i: 0,
            t: 0.0,
            rng: Rng::new(seed ^ 0xA11E_65),
            n,
            fixed: None,
            terrain: terrain.clone(),
            crawl,
            policy: None,
            act: CrawlAction {
                along: 1.0,
                side: 0.0,
                lean: 1.0,
                lead: 1.0,
            },
            cmd: Cmd {
                nav: false,
                ..Cmd::default()
            },
        }
    }

    /// Keep relocating the same leg instead of cycling around the frame.
    pub fn pin_leg(&mut self, leg: usize) {
        self.crawl = false;
        self.fixed = Some(leg % self.n);
        self.moving = leg % self.n;
        self.from = self.origin_world[self.moving];
        self.dest = self.origin_world[self.moving];
        self.dest_body = self.hold_body[self.moving];
    }

    pub fn set_cmd(&mut self, cmd: Cmd) {
        self.cmd = cmd;
    }

    /// Joint setpoints this tick. Stance (and a planted free foot) keep the
    /// settled joints; only a swing is IK'd onto the world chord.
    pub fn cmd_q(&self) -> [[f64; 3]; MAX_LEGS] {
        let mut q = self.q_hold;
        if self.phase.swinging() {
            q[self.moving] = solve_ik(self.frame, self.moving, self.body_of(self.cmd_world())).q;
            clamp_joints(&mut q[self.moving]);
        }
        q
    }

    /// Skip the standing pause and start a lift on the pinned leg immediately.
    pub fn start_lifting(&mut self) {
        self.begin_move();
        self.phase = Phase::Lift;
        self.phase_t = 0.0;
    }

    pub fn step(&mut self, dt: f64) {
        if self.crawl {
            self.ride_body(dt);
        }
        let q = self.cmd_q();
        self.plant.drive(&q, &self.phys, dt);
        substep(&mut self.plant, dt);

        self.t += dt;
        self.phase_t += dt;
        if self.phase_t >= self.phase_dur() {
            self.advance_phase();
        }
    }

    pub fn sample(&self) -> OneLegSample {
        let (pos, yaw, pitch, roll) = self.plant.chassis_pose();
        let vel = self.plant.chassis_vel();
        let u = (self.phase_t / self.phase_dur()).clamp(0.0, 1.0);
        let cmd_world = self.cmd_world();
        let cmd_body = self.body_of(cmd_world);
        let foot_world = self.plant.leg_joints_world(self.moving)[3];
        let foot_body = to_body(foot_world, pos, yaw, pitch, roll);
        let reach_err = dist(foot_world, cmd_world);
        let mut stance_drift = 0.0f64;
        for i in 0..self.n {
            if i == self.moving {
                continue;
            }
            stance_drift = stance_drift.max(dist(
                self.plant.leg_joints_world(i)[3],
                self.origin_world[i],
            ));
        }
        OneLegSample {
            t: self.t,
            move_i: self.move_i,
            moving: self.moving,
            phase: self.phase,
            phase_u: u,
            pos,
            yaw,
            pitch,
            roll,
            vel,
            cmd_body,
            cmd_world,
            foot_body,
            foot_world,
            dest_body: self.dest_body,
            dest_world: self.dest,
            reach_err,
            chassis_xz: hypot2(pos[0] - self.origin_pos[0], pos[2] - self.origin_pos[2]),
            stance_drift,
            moving_travel: dist(foot_world, self.origin_world[self.moving]),
            slip: self.plant.foot_slip(),
            foot_clear: foot_world[1] - FOOT_R,
            fallen: self.plant.chassis_y() < 0.45 || self.plant.pitch_abs() > 0.80,
        }
    }

    fn phase_dur(&self) -> f64 {
        match self.phase {
            Phase::Settle => SETTLE,
            Phase::Lift => LIFT_T,
            Phase::Shift => SHIFT_T,
            Phase::Place => PLACE_T,
            Phase::Pause => PAUSE,
        }
    }

    fn cmd_world(&self) -> V3 {
        match self.phase {
            Phase::Settle => self.from,
            Phase::Pause => self.dest,
            Phase::Lift | Phase::Shift | Phase::Place => self.swing_world(smooth(self.swing_clock())),
        }
    }

    fn swing_clock(&self) -> f64 {
        let t = match self.phase {
            Phase::Lift => self.phase_t,
            Phase::Shift => LIFT_T + self.phase_t,
            Phase::Place => LIFT_T + SHIFT_T + self.phase_t,
            _ => 0.0,
        };
        (t / SWING_T).clamp(0.0, 1.0)
    }

    fn swing_world(&self, s: f64) -> V3 {
        let a = LIFT_T / SWING_T;
        let b = (LIFT_T + SHIFT_T) / SWING_T;
        let apex_y = self.from[1].max(self.dest[1]) + LIFT_H;
        if s <= a {
            let u = (s / a).clamp(0.0, 1.0);
            [self.from[0], lerp(self.from[1], apex_y, u), self.from[2]]
        } else if s <= b {
            let u = ((s - a) / (b - a)).clamp(0.0, 1.0);
            [
                lerp(self.from[0], self.dest[0], u),
                apex_y,
                lerp(self.from[2], self.dest[2], u),
            ]
        } else {
            let u = ((s - b) / (1.0 - b).max(1e-9)).clamp(0.0, 1.0);
            [self.dest[0], lerp(apex_y, self.dest[1], u), self.dest[2]]
        }
    }

    fn body_of(&self, world: V3) -> V3 {
        to_body(world, self.ik_pos, self.ik_yaw, 0.0, 0.0)
    }

    fn advance_phase(&mut self) {
        self.phase_t = 0.0;
        self.phase = match self.phase {
            Phase::Settle => {
                if self.crawl && !self.want_move() {
                    Phase::Pause
                } else if self.crawl {
                    self.begin_move();
                    Phase::Pause
                } else {
                    self.begin_move();
                    Phase::Lift
                }
            }
            Phase::Lift => Phase::Shift,
            Phase::Shift => Phase::Place,
            Phase::Place => {
                let foot = self.plant.leg_joints_world(self.moving)[3];
                self.origin_world[self.moving] = foot;
                self.hold_body[self.moving] = self.body_of(foot);
                self.q_hold[self.moving] = self.plant.leg_q(self.moving);
                clamp_joints(&mut self.q_hold[self.moving]);
                self.plant.lock(&[true; MAX_LEGS], &self.phys);
                if self.crawl {
                    self.ik_yaw += YAW_STEP * self.cmd.turn.clamp(-1.0, 1.0);
                    self.move_i += 1;
                    self.begin_move();
                }
                Phase::Pause
            }
            Phase::Pause => {
                if self.crawl && !self.want_move() {
                    Phase::Pause
                } else if self.crawl {
                    Phase::Lift
                } else {
                    self.move_i += 1;
                    self.begin_move();
                    Phase::Lift
                }
            }
        };
    }

    fn begin_move(&mut self) {
        if self.crawl {
            self.begin_crawl_move();
            return;
        }
        self.moving = self.fixed.unwrap_or(self.move_i % self.n);
        self.from = self.plant.leg_joints_world(self.moving)[3];
        self.origin_world[self.moving] = self.from;
        let (pos, yaw, _, _) = self.plant.chassis_pose();
        self.ik_pos[0] = pos[0];
        self.ik_pos[2] = pos[2];
        self.ik_goal = [pos[0], pos[2]];
        self.ik_yaw = yaw;
        let avoid = self.hold_body[self.moving];
        for _ in 0..80 {
            let cand = sample_plant(self.frame, &self.stand, self.moving, &mut self.rng, avoid);
            let dest = self.world_plant(cand);
            if reachable(self.frame, self.moving, self.body_of(dest))
                && hypot2(dest[0] - self.from[0], dest[2] - self.from[2]) >= 0.28
            {
                self.dest_body = cand;
                self.dest = dest;
                return;
            }
        }
        self.dest_body = standing_foot(self.frame, &self.stand, self.moving);
        self.dest = self.world_plant(self.dest_body);
    }

    fn want_move(&self) -> bool {
        self.cmd.fwd.abs() > 0.08 || self.cmd.turn.abs() > 0.08
    }

    fn begin_crawl_move(&mut self) {
        for _ in 0..self.n {
            self.moving = self.move_i % self.n;
            self.from = self.plant.leg_joints_world(self.moving)[3];
            self.origin_world[self.moving] = self.from;
            self.act = self.decide();
            if self.shift_to_plants(self.moving) >= VETO {
                break;
            }
            self.move_i += 1;
        }
        self.pick_crawl_dest();
    }

    /// One policy query per plant, on the state at the moment the leg lifts.
    /// Held for the whole move so the swing is not chasing a target that moves
    /// under it — the same reason the hand-written version freezes `dest`.
    fn decide(&mut self) -> CrawlAction {
        let Some(p) = self.policy.as_ref() else {
            return CrawlAction {
                along: 1.0,
                side: 0.0,
                lean: 1.0,
                lead: 1.0,
            };
        };
        let (pos, yaw, pitch, roll) = self.plant.chassis_pose();
        let v = self.plant.chassis_vel();
        let (s, c) = yaw.sin_cos();
        let (fx, fz) = (-s, c);
        let neutral = standing_foot(self.frame, &self.stand, self.moving);
        let foot_b = to_body(self.from, pos, yaw, 0.0, 0.0);
        let d = self.frame.dir(self.moving);
        let obs = CrawlObs {
            pitch,
            roll,
            vel_along: v[0] * fx + v[2] * fz,
            vel_side: v[0] * fz - v[2] * fx,
            margin: self.remaining_margin(self.moving, [pos[0], pos[2]]),
            behind: neutral[2] - foot_b[2],
            leg_fore: d[2],
            cmd_fwd: self.cmd.fwd,
            cmd_turn: self.cmd.turn,
        };
        p.act(&obs)
    }

    /// Can the chassis stand here through the coming swing? Two conditions, and
    /// both bind: the COM has to sit MARGIN inside the feet that stay down, and
    /// every one of those feet has to remain inside its leg's workspace. Lean
    /// on stability alone and the body walks out past its own rear legs, which
    /// then stop holding their plants and are dragged along the floor.
    fn body_holds(&self, skip: usize, at: [f64; 2]) -> bool {
        if self.remaining_margin(skip, at) < MARGIN {
            return false;
        }
        let pos = [at[0], self.ik_pos[1], at[1]];
        (0..self.n).all(|i| {
            i == skip
                || reachable(
                    self.frame,
                    i,
                    to_body(self.origin_world[i], pos, self.ik_yaw, 0.0, 0.0),
                )
        })
    }

    fn remaining_margin(&self, skip: usize, com: [f64; 2]) -> f64 {
        let mut pts = [[0.0; 2]; MAX_LEGS];
        let mut n = 0;
        for i in 0..self.n {
            if i == skip {
                continue;
            }
            let p = self.plant.leg_joints_world(i)[3];
            pts[n] = [p[0], p[2]];
            n += 1;
        }
        let mut hull = [[0.0; 2]; MAX_HULL];
        let h = convex_hull_xz(&pts[..n], &mut hull);
        polygon_margin(&hull[..h], com)
    }

    /// Commanded chassis for the coming swing, and the support margin it
    /// bought. A walking animal does not balance on the centre of its support
    /// polygon; it leans onto the edge it is about to walk over, and that lean
    /// is what lets the next foot reach. So: laterally on the plants, and as
    /// far along the heading as the feet that stay down still allow. Standing
    /// on the centroid instead wastes the whole front half of the polygon, and
    /// caps the stride at what a foot can reach from a centred hip.
    fn shift_to_plants(&mut self, skip: usize) -> f64 {
        let (s, c) = self.ik_yaw.sin_cos();
        let fx = -s;
        let fz = c;
        let mut cx = 0.0;
        let mut cz = 0.0;
        let mut n = 0.0;
        for i in 0..self.n {
            let p = self.plant.leg_joints_world(i)[3];
            self.origin_world[i] = p;
            cx += p[0];
            cz += p[2];
            n += 1.0;
        }
        if n < 1.0 {
            return 0.0;
        }
        cx /= n;
        cz /= n;

        // Furthest forward lean that still keeps MARGIN inside the plants that
        // stay down. The centroid end is the one known to be safe, so a bisect
        // that never finds room simply leaves the body centred.
        let mut lo = 0.0;
        let mut hi = self.stand.stance_w;
        for _ in 0..12 {
            let m = 0.5 * (lo + hi);
            if self.body_holds(skip, [cx + m * fx, cz + m * fz]) {
                lo = m;
            } else {
                hi = m;
            }
        }
        let lo = lo * self.act.lean.clamp(0.0, 1.0);
        self.ik_goal = [cx + lo * fx, cz + lo * fz];
        self.remaining_margin(skip, self.ik_goal)
    }

    /// Carry the commanded chassis toward its goal and re-solve the plants
    /// under it. Stance feet keep their world positions; only the joints that
    /// hold them there change, so the machine walks its body over its feet.
    fn ride_body(&mut self, dt: f64) {
        let dx = self.ik_goal[0] - self.ik_pos[0];
        let dz = self.ik_goal[1] - self.ik_pos[2];
        let d = hypot2(dx, dz);
        if d < 1e-9 {
            return;
        }
        let k = (SHIFT_V * dt / d).min(1.0);
        self.ik_pos[0] += dx * k;
        self.ik_pos[2] += dz * k;

        // The lead of the commanded body over the real chassis is what the
        // stance legs push against — it is the drive, so it cannot be zero.
        // Left unbounded it is also a stored position error: the chassis never
        // catches up, so when the command stops the legs keep shoving a body
        // that has nowhere to go, and it rings. Cap it and the strain drains
        // away with the command instead of outliving it.
        let (cp, _, _, _) = self.plant.chassis_pose();
        let (lx, lz) = (self.ik_pos[0] - cp[0], self.ik_pos[2] - cp[2]);
        let lead = hypot2(lx, lz);
        let cap = LEAD_MAX * self.act.lead.clamp(0.0, 1.0).max(0.05);
        if lead > cap {
            self.ik_pos[0] = cp[0] + lx * cap / lead;
            self.ik_pos[2] = cp[2] + lz * cap / lead;
        }
        for i in 0..self.n {
            if i == self.moving && self.phase.swinging() {
                continue;
            }
            let body = self.body_of(self.origin_world[i]);
            if !reachable(self.frame, i, body) {
                continue;
            }
            self.hold_body[i] = body;
            self.q_hold[i] = solve_ik(self.frame, i, body).q;
            clamp_joints(&mut self.q_hold[i]);
        }
    }

    /// Furthest plant along the command this leg can actually reach, up to
    /// STEP. Halving a nominal offset until it fits lands back near the neutral
    /// stance nearly every time, which caps the stride at whatever the nominal
    /// happened to be; the leg's own workspace is the honest limit, so search
    /// for it. `u` runs from the neutral stance out to the full step.
    fn pick_crawl_dest(&mut self) {
        let along = STEP * self.cmd.fwd.clamp(-1.0, 1.0);
        let side = STEP * (SIDE * self.cmd.turn.clamp(-1.0, 1.0) + self.act.side);
        let neutral = standing_foot(self.frame, &self.stand, self.moving);
        let at = |u: f64| {
            let b = [neutral[0] + side * u, neutral[1], neutral[2] + along * u];
            self.world_plant(b)
        };
        let mut lo = 0.0;
        let mut hi = 1.0;
        for _ in 0..10 {
            let m = 0.5 * (lo + hi);
            if reachable(self.frame, self.moving, self.body_of(at(m))) {
                lo = m;
            } else {
                hi = m;
            }
        }
        let dest = at(lo * self.act.along.clamp(0.0, 1.0));
        self.dest = dest;
        self.dest_body = self.body_of(dest);
    }

    fn world_plant(&self, body: V3) -> V3 {
        let w = body_to_world(body, self.ik_yaw, 0.0, 0.0);
        let x = self.ik_pos[0] + w[0];
        let z = self.ik_pos[2] + w[2];
        [x, self.terrain.height(x, z) + FOOT_R, z]
    }
}

fn stand_pose(frame: Frame) -> Gait {
    Gait {
        frame,
        cycle: 1.0,
        stride: 0.0,
        step_h: 0.0,
        body_h: RIDE,
        stance_w: 2.0 * (frame.hip_r() + 0.40),
        duty: 1.0,
        offsets: [0.0; MAX_LEGS],
        trim_front: 0.0,
        trim_rear: 0.0,
    }
}

/// Standing foot in the body frame: under the hip, at ride height, half stance width out.
pub fn standing_foot(frame: Frame, gait: &Gait, leg: usize) -> V3 {
    let d = frame.dir(leg);
    let out = gait.stance_w * 0.5 + gait.trim(leg);
    [d[0] * out, -gait.body_h + FOOT_R, d[2] * out]
}

/// A random reachable plant on the floor, inside this leg's workspace.
pub fn sample_plant(frame: Frame, gait: &Gait, leg: usize, rng: &mut Rng, avoid: V3) -> V3 {
    let hip = frame.hip(leg);
    let y = -gait.body_h + FOOT_R;
    let yaw0 = frame.yaw(leg);
    let r_lo = REACH_MIN + 0.20;
    let r_hi = (REACH_MAX - 0.12).min(gait.stance_w * 0.55 + 0.35);
    for _ in 0..80 {
        let a = yaw0 + rng.range(-1.05, 1.05);
        let r = rng.range(r_lo, r_hi);
        let t = [hip[0] + r * a.cos(), y, hip[2] + r * a.sin()];
        if !reachable(frame, leg, t) {
            continue;
        }
        if hypot2(t[0], t[2]) < frame.body_r() + 0.10 {
            continue;
        }
        if hypot2(t[0] - avoid[0], t[2] - avoid[2]) < 0.28 {
            continue;
        }
        return t;
    }
    let d = frame.dir(leg);
    let fallback = [
        hip[0] + d[0] * 0.55,
        y,
        hip[2] + d[2] * 0.55,
    ];
    if reachable(frame, leg, fallback) {
        fallback
    } else {
        standing_foot(frame, gait, leg)
    }
}

fn reachable(frame: Frame, leg: usize, target: V3) -> bool {
    let sol = solve_ik(frame, leg, target);
    if sol.strain > 1e-4 {
        return false;
    }
    let foot = fk_body(frame, leg, sol.q)[3];
    dist(foot, target) < 0.03
}

fn smooth(u: f64) -> f64 {
    0.5 - 0.5 * (core::f64::consts::PI * u.clamp(0.0, 1.0)).cos()
}

fn dist(a: V3, b: V3) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn substep(plant: &mut ArticulatedPlant, dt: f64) {
    // Contact at the control rate skates the plants, so the tick is split.
    // How far is [`Physics::substeps`]; the scene runner sweeps it.
    let n = plant.substeps;
    let h = dt / n as f64;
    for _ in 0..n {
        plant.step(h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::DT;

    #[test]
    fn sampled_plants_are_inside_the_workspace() {
        let frame = Frame::new(6);
        let gait = stand_pose(frame);
        let mut rng = Rng::new(3);
        for leg in 0..6 {
            let avoid = standing_foot(frame, &gait, leg);
            for _ in 0..20 {
                let t = sample_plant(frame, &gait, leg, &mut rng, avoid);
                assert!(
                    reachable(frame, leg, t),
                    "leg {leg} sampled unreachable {:?}",
                    t
                );
                assert!(
                    (t[1] + gait.body_h - FOOT_R).abs() < 1e-9,
                    "plant not on the floor: y={}",
                    t[1]
                );
            }
        }
    }

    #[test]
    fn five_legs_hold_by_friction_while_one_relocates() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let mut drill = OneLegDrill::spawn(frame, &phys, 1);
        drill.pin_leg(0);
        let ticks = (6.0 / DT) as usize;
        let mut max_stance = 0.0f64;
        let mut max_chassis = 0.0f64;
        let mut max_travel = 0.0f64;
        let mut max_clear = 0.0f64;
        let mut min_y = f64::INFINITY;
        let mut dest0 = None;
        let mut dest_wander = 0.0f64;
        let mut cmd_reversals = 0usize;
        let mut prev_cmd: Option<V3> = None;
        let mut prev_to_dest: Option<f64> = None;
        let mut land_err = 0.0f64;
        let mut stance_path = 0.0f64;
        let mut prev_stance: Option<[V3; MAX_LEGS]> = None;
        let mut max_overshoot = 0.0f64;
        for _ in 0..ticks {
            drill.step(DT);
            let s = drill.sample();
            assert_eq!(s.moving, 0, "drill left L1");
            min_y = min_y.min(s.pos[1]);
            max_clear = max_clear.max(s.foot_clear);
            if s.phase != Phase::Settle {
                max_stance = max_stance.max(s.stance_drift);
                max_chassis = max_chassis.max(s.chassis_xz);
                max_travel = max_travel.max(s.moving_travel);
            }
            if s.phase == Phase::Pause || s.phase == Phase::Place && s.phase_u > 0.85 {
                land_err = land_err.max(hypot2(
                    s.foot_world[0] - s.dest_world[0],
                    s.foot_world[2] - s.dest_world[2],
                ));
            }
            if s.phase == Phase::Place || s.phase == Phase::Pause {
                let from_dest = hypot2(
                    drill.from[0] - s.dest_world[0],
                    drill.from[2] - s.dest_world[2],
                );
                let from_foot = hypot2(
                    drill.from[0] - s.foot_world[0],
                    drill.from[2] - s.foot_world[2],
                );
                if from_foot > from_dest + 0.04 {
                    max_overshoot = max_overshoot.max(hypot2(
                        s.foot_world[0] - s.dest_world[0],
                        s.foot_world[2] - s.dest_world[2],
                    ));
                }
            }
            let mut feet = [[0.0; 3]; MAX_LEGS];
            for i in 0..6 {
                feet[i] = drill.plant.leg_joints_world(i)[3];
            }
            if let Some(prev) = prev_stance {
                for i in 1..6 {
                    stance_path += dist(feet[i], prev[i]);
                }
            }
            prev_stance = Some(feet);
            if s.phase.swinging() {
                let dest = dest0.get_or_insert(s.dest_world);
                dest_wander = dest_wander.max(dist(s.dest_world, *dest));
                let to_dest = hypot2(
                    s.cmd_world[0] - s.dest_world[0],
                    s.cmd_world[2] - s.dest_world[2],
                );
                if let (Some(prev), Some(d0)) = (prev_cmd, prev_to_dest) {
                    let step = dist(s.cmd_world, prev);
                    if s.phase == Phase::Shift && to_dest > d0 + 0.02 && step > 0.004 {
                        cmd_reversals += 1;
                    }
                }
                prev_cmd = Some(s.cmd_world);
                prev_to_dest = Some(to_dest);
            } else {
                dest0 = None;
                prev_cmd = None;
                prev_to_dest = None;
            }
            assert!(!s.fallen, "fell at t={:.2} y={:.3} pitch={:.3} drift={:.3} land={:.3}", s.t, s.pos[1], s.pitch, s.stance_drift, hypot2(s.foot_world[0]-s.dest_world[0], s.foot_world[2]-s.dest_world[2]));
        }
        eprintln!(
            "oneleg: min_y={min_y:.3} travel={max_travel:.3} stance_drift={max_stance:.3} \
             chassis_xz={max_chassis:.3} max_clear={max_clear:.3} dest_wander={dest_wander:.4} \
             reversals={cmd_reversals} land_err={land_err:.3} overshoot={max_overshoot:.3} \
             stance_path={stance_path:.3}"
        );
        assert!(
            min_y > 0.55,
            "sat down: min_y={min_y:.3}"
        );
        assert!(
            max_clear > 0.12,
            "moving foot never left the floor: clearance={max_clear:.3}"
        );
        assert!(
            max_travel > 0.12,
            "moving foot barely left its plant: travel={max_travel:.3}"
        );
        assert!(
            max_stance < 0.16,
            "stance feet wandered: stance_drift={max_stance:.3} travel={max_travel:.3}"
        );
        assert!(
            stance_path < 4.0,
            "stance feet trembled: path={stance_path:.3} m over 6 s"
        );
        assert!(
            land_err < 0.10,
            "moving foot missed the landing mark: {land_err:.3} m"
        );
        assert!(
            max_overshoot < 0.12,
            "moving foot ran past the mark: {max_overshoot:.3} m"
        );
        assert!(
            max_chassis < 0.70,
            "chassis walked away: Δxz={max_chassis:.3} travel={max_travel:.3} stance={max_stance:.3}"
        );
        assert!(
            dest_wander < 0.01,
            "landing mark crawled during the swing: {dest_wander:.4} m"
        );
        assert!(
            cmd_reversals == 0,
            "swing command reversed {cmd_reversals} times"
        );
    }

    #[test]
    fn the_landing_is_a_world_point_not_a_body_frame_ghost() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let mut drill = OneLegDrill::spawn(frame, &phys, 1);
        drill.pin_leg(0);
        drill.start_lifting();
        let dest0 = drill.dest;
        let mut pitch_seen = 0.0f64;
        let ticks = ((LIFT_T + SHIFT_T + PLACE_T) / DT) as usize;
        for _ in 0..ticks {
            drill.step(DT);
            let s = drill.sample();
            pitch_seen = pitch_seen.max(s.pitch.abs());
            assert!(
                dist(s.dest_world, dest0) < 1e-9,
                "dest moved from {:?} to {:?} at t={:.3} pitch={:.3}",
                dest0,
                s.dest_world,
                s.t,
                s.pitch
            );
            if s.phase.swinging() {
                assert_eq!(s.moving, 0);
            }
        }
        assert!(
            drill.move_i >= 1 || drill.phase == Phase::Pause,
            "never finished a plant: phase={:?} move={}",
            drill.phase,
            drill.move_i
        );
        let _ = pitch_seen;
    }

    #[test]
    fn settled_feet_do_not_buzz_in_place() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let mut drill = OneLegDrill::spawn(frame, &phys, 1);
        let mut path = 0.0f64;
        let mut prev: Option<[V3; MAX_LEGS]> = None;
        let ticks = (1.0 / DT) as usize;
        for _ in 0..ticks {
            drill.step(DT);
            let mut feet = [[0.0; 3]; MAX_LEGS];
            for i in 0..6 {
                feet[i] = drill.plant.leg_joints_world(i)[3];
            }
            if let Some(p) = prev {
                for i in 0..6 {
                    path += dist(feet[i], p[i]);
                }
            }
            prev = Some(feet);
            assert!(!drill.sample().fallen, "sat down while standing still");
        }
        eprintln!("sit-still path={path:.4} (6 legs, 1 s)");
        assert!(
            path < 0.70,
            "standing feet buzzed: path={path:.3} m over 1 s"
        );
    }

    #[test]
    fn crawl_advances_with_one_foot_in_the_air() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut drill = OneLegDrill::spawn_on(frame, &phys, &terrain, 1, true);
        drill.set_cmd(Cmd::forward());
        // One plant every ~3 s, six to a cycle: a window shorter than a couple
        // of cycles measures the settle transient, not the walking.
        let ticks = (60.0 / DT) as usize;
        let mut min_y = f64::INFINITY;
        let mut max_swing = 0u32;
        let mut stance_path = 0.0f64;
        let mut prev: Option<[V3; MAX_LEGS]> = None;
        for _ in 0..ticks {
            drill.step(DT);
            let s = drill.sample();
            min_y = min_y.min(s.pos[1]);
            assert!(!s.fallen, "fell at t={:.2} y={:.3} pitch={:.3}", s.t, s.pos[1], s.pitch);
            let mut swing = 0u32;
            for i in 0..6 {
                if i == s.moving && s.phase.swinging() {
                    swing += 1;
                }
            }
            max_swing = max_swing.max(swing);
            let mut feet = [[0.0; 3]; MAX_LEGS];
            for i in 0..6 {
                feet[i] = drill.plant.leg_joints_world(i)[3];
            }
            if let Some(p) = prev {
                for i in 0..6 {
                    if i == s.moving && s.phase.swinging() {
                        continue;
                    }
                    stance_path += dist(feet[i], p[i]);
                }
            }
            prev = Some(feet);
        }
        let s = drill.sample();
        eprintln!(
            "crawl: min_y={min_y:.3} chassis_xz={:.3} max_swing={max_swing} stance_path={stance_path:.3} moves={}",
            s.chassis_xz, drill.move_i
        );
        assert!(min_y > 0.55, "sat down: min_y={min_y:.3}");
        assert!(max_swing <= 1, "crawled with {max_swing} feet in the air");
        assert!(
            s.chassis_xz > 3.0,
            "crawl did not walk: Δxz={:.3}",
            s.chassis_xz
        );
        assert!(
            stance_path < 9.0,
            "stance feet skated: path={stance_path:.3} m over 60 s"
        );
    }

    #[test]
    fn crawl_keeps_the_body_behind_the_front_plants() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut drill = OneLegDrill::spawn_on(frame, &phys, &terrain, 1, true);
        drill.set_cmd(Cmd::forward());
        let ticks = (12.0 / DT) as usize;
        let mut min_y = f64::INFINITY;
        let mut max_overhang = f64::NEG_INFINITY;
        let mut seq = Vec::new();
        let mut prev_phase = Phase::Settle;
        for _ in 0..ticks {
            drill.step(DT);
            let s = drill.sample();
            min_y = min_y.min(s.pos[1]);
            assert!(!s.fallen, "fell at t={:.2} y={:.3} pitch={:.3} moves={}", s.t, s.pos[1], s.pitch, drill.move_i);
            if s.phase == Phase::Lift && prev_phase != Phase::Lift {
                seq.push(s.moving);
            }
            if s.phase.swinging() {
                assert!(
                    drill.remaining_margin(s.moving, [drill.ik_pos[0], drill.ik_pos[2]]) >= 0.0,
                    "lifted L{} with COM outside remaining plants t={:.2}",
                    s.moving + 1,
                    s.t
                );
            }
            prev_phase = s.phase;
            let (sn, cs) = s.yaw.sin_cos();
            let fx = -sn;
            let fz = cs;
            let body = s.pos[0] * fx + s.pos[2] * fz;
            let mut front = f64::NEG_INFINITY;
            for i in 0..6 {
                let f = drill.plant.leg_joints_world(i)[3];
                front = front.max(f[0] * fx + f[2] * fz);
            }
            max_overhang = max_overhang.max(body - front);
        }
        eprintln!(
            "crawl overhang: max(body-front)={max_overhang:.3} min_y={min_y:.3} moves={} seq={seq:?}",
            drill.move_i
        );
        assert!(min_y > 0.55, "sat down: min_y={min_y:.3}");
        assert!(
            max_overhang < 0.08,
            "chassis walked past the front plants: overhang={max_overhang:.3}"
        );
        assert!(seq.len() >= 3, "too few steps: {seq:?}");
        for (k, &leg) in seq.iter().enumerate() {
            assert_eq!(leg, k % 6, "crawl skipped legs: {seq:?}");
        }
    }

    #[test]
    fn crawl_recenters_onto_the_plants_that_will_remain() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut drill = OneLegDrill::spawn_on(frame, &phys, &terrain, 1, true);
        drill.ik_pos[0] = drill.origin_world[0][0];
        drill.ik_pos[2] = drill.origin_world[0][2];
        assert!(
            drill.remaining_margin(0, [drill.ik_pos[0], drill.ik_pos[2]]) < 0.0,
            "COM on a plant should be outside the other five"
        );
        drill.shift_to_plants(0);
        // The body rides at SHIFT_V now, so give it time to cover a full
        // stance radius rather than the old snap-per-call.
        for _ in 0..(30.0 / DT) as usize {
            drill.ride_body(DT);
        }
        assert!(
            drill.remaining_margin(0, [drill.ik_pos[0], drill.ik_pos[2]]) > 0.04,
            "remaining-plant centroid must sit inside the pentagon"
        );
    }

    #[test]
    fn crawl_holds_still_when_the_command_is_stop() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut drill = OneLegDrill::spawn_on(frame, &phys, &terrain, 1, true);
        let mut path = 0.0f64;
        let mut prev: Option<[V3; MAX_LEGS]> = None;
        let ticks = (1.0 / DT) as usize;
        for _ in 0..ticks {
            drill.step(DT);
            let mut feet = [[0.0; 3]; MAX_LEGS];
            for i in 0..6 {
                feet[i] = drill.plant.leg_joints_world(i)[3];
            }
            if let Some(p) = prev {
                for i in 0..6 {
                    path += dist(feet[i], p[i]);
                }
            }
            prev = Some(feet);
            assert!(!drill.sample().fallen, "sat down while standing still");
            assert!(
                !drill.sample().phase.swinging(),
                "stop command still swung {:?}",
                drill.phase
            );
        }
        eprintln!("crawl sit-still path={path:.4} (6 legs, 1 s)");
        assert!(
            path < 0.70,
            "standing feet buzzed: path={path:.3} m over 1 s"
        );
    }
}
