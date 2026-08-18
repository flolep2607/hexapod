//! Empty-field one-leg drill: friction holds the plants, nothing is welded.
//!
//! The swing is planned in the *world*. At lift-off the start and the landing
//! are snapshotted as world points and the IK origin is frozen with them, so
//! the red mark does not crawl when the chassis rocks and the free foot is
//! not retargeted by pitch or roll. Stance legs keep the standing joint
//! setpoints they were spawned with — only one command changes.

use crate::dynamics::Physics;
use crate::math::{body_to_world, hypot2, lerp, Rng, V3};
use crate::plant::ArticulatedPlant;
use crate::policy::{Gait, Policy, Preset};
use crate::robot::{
    clamp_joints, fk_body, solve_ik, to_body, Frame, FOOT_R, MAX_LEGS, REACH_MAX, REACH_MIN,
};
use crate::terrain::{Course, Terrain};

const SETTLE: f64 = 0.70;
const LIFT_T: f64 = 0.55;
const SHIFT_T: f64 = 0.80;
const PLACE_T: f64 = 0.55;
const PAUSE: f64 = 0.55;
/// High enough that a watching eye can tell the sole left the floor. 18 cm
/// looked like a scrape; 40 cm is a deliberate pick-up.
const LIFT_H: f64 = 0.40;

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
    pub gait: Gait,
    pub phys: Physics,
    pub plant: ArticulatedPlant,
    /// Standing joint setpoints. Stance legs are driven to these every tick
    /// so only the free leg's command changes.
    pub q_hold: [[f64; 3]; MAX_LEGS],
    /// Last planted body-frame foot, used to sample the next reachable plant.
    pub hold_body: [V3; MAX_LEGS],
    /// World plants captured at the start of the current move. Stance IK holds
    /// these; they do not move while the free foot is in the air.
    pub origin_world: [V3; MAX_LEGS],
    pub origin_pos: V3,
    pub moving: usize,
    /// World lift-off of the free foot, frozen at [`Self::begin_move`].
    pub from: V3,
    /// World landing, frozen at [`Self::begin_move`]. The canvas mark is this.
    pub dest: V3,
    /// Body-frame sample that produced `dest`, so the next plant can avoid it.
    pub dest_body: V3,
    /// Yaw-only chassis the IK uses. Height is the ride height at lift-off;
    /// xz/yaw lag the plant so solver jitter does not retarget the joints.
    pub ik_pos: V3,
    pub ik_yaw: f64,
    pub phase: Phase,
    pub phase_t: f64,
    pub move_i: usize,
    pub t: f64,
    rng: Rng,
    n: usize,
    fixed: Option<usize>,
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
    pub fn spawn(frame: Frame, phys: &Physics, seed: u64) -> Self {
        let gait = Policy::seeded(Preset::default_for(frame), frame).gait();
        let terrain = Terrain::new(Course::Flat, seed);
        let plant = ArticulatedPlant::standing(frame, &gait, phys, &terrain);
        let n = frame.legs();
        let mut q_hold = [[0.0; 3]; MAX_LEGS];
        let mut hold_body = [[0.0; 3]; MAX_LEGS];
        for i in 0..n {
            hold_body[i] = standing_foot(frame, &gait, i);
            q_hold[i] = solve_ik(frame, i, hold_body[i]).q;
            clamp_joints(&mut q_hold[i]);
        }
        let mut origin_world = [[0.0; 3]; MAX_LEGS];
        for i in 0..n {
            origin_world[i] = plant.leg_joints_world(i)[3];
        }
        let (origin_pos, origin_yaw, _, _) = plant.chassis_pose();
        OneLegDrill {
            frame,
            gait,
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
            ik_yaw: origin_yaw,
            phase: Phase::Settle,
            phase_t: 0.0,
            move_i: 0,
            t: 0.0,
            rng: Rng::new(seed ^ 0xA11E_65),
            n,
            fixed: None,
        }
    }

    /// Keep relocating the same leg instead of cycling around the frame.
    pub fn pin_leg(&mut self, leg: usize) {
        self.fixed = Some(leg % self.n);
        self.moving = leg % self.n;
        self.from = self.origin_world[self.moving];
        self.dest = self.origin_world[self.moving];
        self.dest_body = self.hold_body[self.moving];
    }

    /// Joint setpoints this tick. Stance legs keep the standing pose they
    /// were spawned with; only the free foot follows the world-space chord.
    pub fn cmd_q(&self) -> [[f64; 3]; MAX_LEGS] {
        let u = (self.phase_t / self.phase_dur()).clamp(0.0, 1.0);
        let mut q = self.q_hold;
        q[self.moving] = solve_ik(self.frame, self.moving, self.body_of(self.cmd_world(smooth(u)))).q;
        clamp_joints(&mut q[self.moving]);
        q
    }

    /// Skip the standing pause and start a lift on the pinned leg immediately.
    pub fn start_lifting(&mut self) {
        self.begin_move();
        self.phase = Phase::Lift;
        self.phase_t = 0.0;
    }

    pub fn step(&mut self, dt: f64) {
        if self.phase == Phase::Settle && self.phase_t == 0.0 && self.move_i == 0 {
            self.capture_origin();
        }

        let q = self.cmd_q();
        self.plant.drive(&q, &self.phys, dt);
        self.plant.step(dt);

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
        let cmd_world = self.cmd_world(smooth(u));
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

    fn cmd_world(&self, u: f64) -> V3 {
        match self.phase {
            Phase::Settle => self.from,
            Phase::Pause => self.dest,
            Phase::Lift => [self.from[0], self.from[1] + LIFT_H * u, self.from[2]],
            Phase::Shift => [
                lerp(self.from[0], self.dest[0], u),
                self.from[1].max(self.dest[1]) + LIFT_H,
                lerp(self.from[2], self.dest[2], u),
            ],
            Phase::Place => [
                self.dest[0],
                lerp(self.from[1].max(self.dest[1]) + LIFT_H, self.dest[1], u),
                self.dest[2],
            ],
        }
    }

    fn body_of(&self, world: V3) -> V3 {
        to_body(world, self.ik_pos, self.ik_yaw, 0.0, 0.0)
    }

    fn advance_phase(&mut self) {
        self.phase_t = 0.0;
        self.phase = match self.phase {
            Phase::Settle => {
                self.begin_move();
                Phase::Lift
            }
            Phase::Lift => Phase::Shift,
            Phase::Shift => Phase::Place,
            Phase::Place => Phase::Pause,
            Phase::Pause => {
                let planted = self.body_of(self.dest);
                self.q_hold[self.moving] = solve_ik(self.frame, self.moving, planted).q;
                clamp_joints(&mut self.q_hold[self.moving]);
                self.hold_body[self.moving] = planted;
                self.origin_world[self.moving] = self.dest;
                self.move_i += 1;
                self.begin_move();
                Phase::Lift
            }
        };
    }

    fn begin_move(&mut self) {
        self.moving = self.fixed.unwrap_or(self.move_i % self.n);
        self.dest_body = sample_plant(
            self.frame,
            &self.gait,
            self.moving,
            &mut self.rng,
            self.hold_body[self.moving],
        );
        self.capture_origin();
        self.from = self.origin_world[self.moving];
        let w = body_to_world(self.dest_body, self.ik_yaw, 0.0, 0.0);
        self.dest = [
            self.ik_pos[0] + w[0],
            FOOT_R,
            self.ik_pos[2] + w[2],
        ];
    }

    fn capture_origin(&mut self) {
        let (pos, yaw, _, _) = self.plant.chassis_pose();
        self.origin_pos = pos;
        self.ik_pos = pos;
        self.ik_yaw = yaw;
        for i in 0..self.n {
            self.origin_world[i] = self.plant.leg_joints_world(i)[3];
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::DT;

    #[test]
    fn sampled_plants_are_inside_the_workspace() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
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
        let q_stance0 = drill.q_hold[1];
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
                let q1 = drill.cmd_q()[1];
                for j in 0..3 {
                    assert!(
                        (q1[j] - q_stance0[j]).abs() < 1e-12,
                        "stance leg 1 commanded while L1 swung: {:?} vs {:?}",
                        q1,
                        q_stance0
                    );
                }
            } else {
                dest0 = None;
                prev_cmd = None;
                prev_to_dest = None;
            }
            assert!(!s.fallen, "fell at t={:.2} y={:.3}", s.t, s.pos[1]);
        }
        eprintln!(
            "oneleg: min_y={min_y:.3} travel={max_travel:.3} stance_drift={max_stance:.3} \
             chassis_xz={max_chassis:.3} max_clear={max_clear:.3} dest_wander={dest_wander:.4} \
             reversals={cmd_reversals}"
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
            max_stance < max_travel,
            "stance feet slid as far as the swing: stance_drift={max_stance:.3} travel={max_travel:.3}"
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
}
