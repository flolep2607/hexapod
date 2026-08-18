//! Rapier walking loop shared by the dashboard and `hexapod watch`.
//!
//! Stance feet are locked to the world point they touched. The centroidal gait
//! sweeps a foot a whole stride per stance window, which is correct for a body
//! that already moves at `stride/(duty*cycle)`. The articulated plant starts
//! at rest, so that same sweep drags the rubber through the floor at a couple
//! of metres per second and the machine skates. Holding the plant, swinging
//! the next foot a short step behind the hip, and damping yaw on the stance
//! tripod is how this position-controlled walk actually moves: the body walks
//! over the locked feet instead of the feet skating under the body.

use crate::dynamics::Physics;
use crate::math::{body_to_world, hypot2, V3};
use crate::plant::ArticulatedPlant;
use crate::policy::{
    clear_links, feasible_cycle, swing_blocked, Gait, Policy, Preset,
};
use crate::robot::{solve_ik, to_body, Frame, FOOT_R, LINK_R, MAX_LEGS};
use crate::sim::{Cmd, Sim, FOOT_CLEAR, MAX_FOOTHOLD};
use crate::terrain::Terrain;

/// How far from the hip, as a fraction of stride, a swinging foot plants.
/// Negative: behind the hip. Planting *ahead* of a stationary body leaves
/// the COM behind the new tripod and the machine sits back; planting behind
/// lets the COM walk over the lock.
const PLANT_ALONG: f64 = -0.12;
/// Persistent body-frame offset on every stance foot, metres. Zero until the
/// locked plants themselves produce a net force; a few centimetres of "lean"
/// here is a yaw moment on a tripod.
const STANCE_LEAN: f64 = 0.0;
/// Metres of left/right stance split per (rad/s) of yaw-rate error. The
/// tripod plants two feet on one side, so without this the chassis pirouettes
/// the first time a swing reaction appears.
const YAW_GAIN: f64 = 0.16;

/// World plants captured at touchdown, one per leg.
#[derive(Clone, Debug)]
pub struct StanceLocks {
    lock: [Option<V3>; MAX_LEGS],
    last: [bool; MAX_LEGS],
}

impl Default for StanceLocks {
    fn default() -> Self {
        StanceLocks {
            lock: [None; MAX_LEGS],
            last: [false; MAX_LEGS],
        }
    }
}

impl StanceLocks {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Capture any stance foot that just landed, using the Rapier foot pose.
    pub fn capture(&mut self, sim: &Sim, plant: &ArticulatedPlant) {
        let n = sim.frame.legs();
        for i in 0..n {
            let now = sim.feet[i].stance;
            if now && !self.last[i] {
                self.lock[i] = Some(plant.leg_joints_world(i)[3]);
            } else if !now {
                self.lock[i] = None;
            }
            self.last[i] = now;
        }
    }
}

/// One Rapier robot, the gait clock that drives it, and the world locks.
pub struct ArticulatedWalker {
    pub sim: Sim,
    pub plant: ArticulatedPlant,
    pub locks: StanceLocks,
    pub frame: Frame,
    pub phys: Physics,
}

impl ArticulatedWalker {
    pub fn spawn(frame: Frame, gait: &Gait, phys: &Physics, terrain: &Terrain) -> Self {
        let mut sim = Sim::default();
        sim.reset(terrain, gait, phys);
        let plant = ArticulatedPlant::standing(frame, gait, phys, terrain);
        let (p, yaw, pitch, roll) = plant.chassis_pose();
        sim.observe_pose(p, yaw, pitch, roll, plant.chassis_vel());
        for i in 0..frame.legs() {
            sim.feet[i].world = plant.leg_joints_world(i)[3];
        }
        let mut locks = StanceLocks::new();
        locks.capture(&sim, &plant);
        ArticulatedWalker {
            sim,
            plant,
            locks,
            frame,
            phys: *phys,
        }
    }

    /// Seeded open-loop gait with the cycle floor the live plant actually uses.
    pub fn seeded(frame: Frame, preset: Preset, phys: &Physics, terrain: &Terrain) -> Self {
        let mut gait = Policy::seeded(preset, frame).gait();
        prepare_live_gait(frame, &mut gait, phys);
        Self::spawn(frame, &gait, phys, terrain)
    }

    pub fn step(
        &mut self,
        terrain: &Terrain,
        policy: &Policy,
        gait: &Gait,
        dt: f64,
        cmd: Cmd,
    ) {
        drive_articulated(
            &mut self.sim,
            &mut self.plant,
            &mut self.locks,
            terrain,
            policy,
            gait,
            &self.phys,
            dt,
            cmd,
        );
    }

    pub fn sample(&self) -> WalkSample {
        WalkSample::from_state(&self.sim, &self.plant)
    }
}

/// Cap step height and lengthen the clock so the motors can track the stroke.
pub fn prepare_live_gait(frame: Frame, gait: &mut Gait, phys: &Physics) {
    gait.step_h = gait.step_h.min(0.32 * gait.body_h);
    gait.cycle = gait.cycle.max(feasible_cycle(
        frame,
        gait,
        gait.stride,
        gait.duty,
        gait.cycle,
        gait.body_h,
        gait.step_h,
        0.0,
        phys.actuator.omega_max,
    ));
}

/// One control tick of the articulated plant: gait clock, world-locked
/// stance, swing toward a foothold ahead of the hip, motors, Rapier step.
pub fn drive_articulated(
    sim: &mut Sim,
    plant: &mut ArticulatedPlant,
    locks: &mut StanceLocks,
    terrain: &Terrain,
    policy: &Policy,
    gait: &Gait,
    phys: &Physics,
    dt: f64,
    cmd: Cmd,
) {
    sim.tick_gait(terrain, policy, gait, dt, cmd);

    let (bp, byaw, bpitch, broll) = plant.chassis_pose();
    sim.observe_pose(bp, byaw, bpitch, broll, plant.chassis_vel());
    locks.capture(sim, plant);

    let n = sim.frame.legs();
    let body_h = gait.body_h + 0.20 * sim.act[crate::policy::act_body_dh(sim.frame)];
    let v_cmd = cmd.speed();
    let lean = if v_cmd.abs() < 0.05 {
        0.0
    } else {
        STANCE_LEAN * v_cmd.signum()
    };
    let plane = terrain.height(bp[0], bp[2]);
    let turn = 1.5 * crate::math::ang_diff(sim.yaw - byaw);
    let yaw_w = plant.chassis_angvel()[1];
    // Damp measured spin only. Feeding `sim.yaw_rate` in as a desired rate
    // made route pursuit add a tripod split with the wrong sign and the
    // machine walked off-axis chasing its own yaw.
    let w_err = -yaw_w;

    let mut q_cmd = sim.q_cmd;
    for i in 0..n {
        let target = if let Some(lock) = locks.lock[i] {
            let horiz = crate::math::world_to_body(
                [lock[0] - bp[0], 0.0, lock[2] - bp[2]],
                byaw,
                0.0,
                0.0,
            );
            let ground = terrain.height(lock[0], lock[2]);
            let (_, right) = sim.frame.split(i);
            let side = if right { 1.0 } else { -1.0 };
            [
                horiz[0],
                -body_h + FOOT_R - 0.03 + (ground - plane),
                horiz[2] + lean + YAW_GAIN * w_err * side,
            ]
        } else {
            swing_ahead(
                sim.frame,
                gait,
                i,
                sim.phase,
                sim.stride_now,
                sim.duty_now,
                body_h,
                sim.feet[i].step_h,
                turn,
                terrain,
                bp,
                byaw,
                bpitch,
                broll,
                sim.feet[i].lift_from,
            )
        };
        q_cmd[i] = solve_ik(sim.frame, i, target).q;
        sim.q_cmd[i] = q_cmd[i];
        let w = body_to_world(target, byaw, bpitch, broll);
        sim.feet[i].td = [bp[0] + w[0], bp[1] + w[1], bp[2] + w[2]];
    }

    plant.drive(&q_cmd, phys, dt);
    let pre = plant.chassis_vel();
    plant.step(dt);
    let (p, yaw, pitch, roll) = plant.chassis_pose();
    let v = plant.chassis_vel();
    sim.observe_pose(p, yaw, pitch, roll, v);
    for i in 0..n {
        sim.feet[i].world = plant.leg_joints_world(i)[3];
    }
    let slip_v = plant.foot_slip();
    sim.slip = slip_v * dt;
    sim.slip_total += sim.slip;
    sim.traction = 0.0;
    if plant.chassis_dead(pre) {
        sim.fallen = true;
    }
}

fn swing_ahead(
    frame: Frame,
    gait: &Gait,
    leg: usize,
    phase: f64,
    stride: f64,
    duty: f64,
    body_h: f64,
    step_h: f64,
    turn: f64,
    terrain: &Terrain,
    pos: V3,
    yaw: f64,
    pitch: f64,
    roll: f64,
    lift_from: V3,
) -> V3 {
    use crate::math::frac;
    let lp = frac(phase + gait.offsets[leg]);
    let u = ((lp - duty) / (1.0 - duty).max(1e-6)).clamp(0.0, 1.0);

    let d = frame.dir(leg);
    let out = gait.stance_w * 0.5 + gait.trim(leg);
    let dest_b = [
        d[0] * out,
        -body_h + FOOT_R - 0.03,
        d[2] * out + PLANT_ALONG * stride,
    ];
    let dest_b = crate::math::rot_y(dest_b, turn * duty * gait.cycle * PLANT_ALONG.abs());
    let w = body_to_world(dest_b, yaw, 0.0, 0.0);
    let (mut x, mut z) = (pos[0] + w[0], pos[2] + w[2]);
    let plane = terrain.height(pos[0], pos[2]);

    if swing_blocked(terrain, x, z, plane, MAX_FOOTHOLD, LINK_R) {
        let hip = frame.hip(leg);
        let hw = body_to_world(hip, yaw, 0.0, 0.0);
        let (hx, hz) = (pos[0] + hw[0], pos[2] + hw[2]);
        let (x0, z0) = (x, z);
        let mut lo = 0.0;
        let mut hi = 1.0;
        for _ in 0..12 {
            let m = 0.5 * (lo + hi);
            let mx = x0 + (hx - x0) * m;
            let mz = z0 + (hz - z0) * m;
            if swing_blocked(terrain, mx, mz, plane, MAX_FOOTHOLD, LINK_R) {
                lo = m;
            } else {
                hi = m;
            }
        }
        let t = (hi + 0.12).min(1.0);
        x = x0 + (hx - x0) * t;
        z = z0 + (hz - z0) * t;
    } else {
        let pushed = terrain.push_xz(x, z, plane, MAX_FOOTHOLD);
        x = pushed.0;
        z = pushed.1;
    }

    let from = if hypot2(lift_from[0] - pos[0], lift_from[2] - pos[2]) > 3.0 {
        // Uninitialised lift-from (still at spawn, metres away): drop a chord
        // from the hip instead of teleporting the swing across the course.
        let hip = frame.hip(leg);
        let hw = body_to_world(hip, yaw, 0.0, 0.0);
        [
            pos[0] + hw[0],
            terrain.height(pos[0] + hw[0], pos[2] + hw[2]) + FOOT_R,
            pos[2] + hw[2],
        ]
    } else {
        lift_from
    };
    x = from[0] + (x - from[0]) * u;
    z = from[2] + (z - from[2]) * u;
    let ground = terrain.height(x, z).min(plane + MAX_FOOTHOLD);
    let lift = FOOT_CLEAR + step_h * (core::f64::consts::PI * u).sin();
    let y_world = ground + FOOT_R + lift;
    let mut target = to_body([x, y_world, z], pos, yaw, pitch, roll);
    let ride = -body_h + FOOT_R - 0.03 + (ground - plane);
    target[1] = target[1].max(ride);
    clear_links(frame, leg, target, terrain, pos, yaw)
}

/// Snapshot the CLI and tests print: pose, 3-axis velocity, heading, waypoint.
#[derive(Clone, Copy, Debug)]
pub struct WalkSample {
    pub t: f64,
    pub pos: V3,
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
    pub vel: V3,
    pub speed: f64,
    pub along: f64,
    pub slip: f64,
    pub yaw_rate: f64,
    pub heading_deg: f64,
    pub wp: usize,
    pub wp_n: usize,
    pub wp_dist: f64,
    pub bearing: f64,
    pub bearing_deg: f64,
    pub reached: usize,
    pub cmd_speed: f64,
    pub n_legs: usize,
    pub stance: [bool; MAX_LEGS],
    pub fallen: bool,
}

impl WalkSample {
    pub fn from_state(sim: &Sim, plant: &ArticulatedPlant) -> Self {
        let (pos, yaw, pitch, roll) = plant.chassis_pose();
        let vel = plant.chassis_vel();
        let (hs, hc) = yaw.sin_cos();
        let hx = -hs;
        let hz = hc;
        let along = vel[0] * hx + vel[2] * hz;
        let speed = hypot2(vel[0], vel[2]);
        let mut stance = [false; MAX_LEGS];
        let n = sim.frame.legs();
        for i in 0..n {
            stance[i] = sim.feet[i].stance;
        }
        WalkSample {
            t: sim.t,
            pos,
            yaw,
            pitch,
            roll,
            vel,
            speed,
            along,
            slip: plant.foot_slip(),
            yaw_rate: plant.chassis_angvel()[1],
            heading_deg: yaw.to_degrees(),
            wp: sim.wp,
            wp_n: 0,
            wp_dist: sim.wp_dist,
            bearing: sim.bearing,
            bearing_deg: sim.bearing.to_degrees(),
            reached: sim.reached,
            cmd_speed: sim.cmd_speed,
            n_legs: n,
            stance,
            fallen: sim.fallen,
        }
    }

    pub fn with_wp_n(mut self, n: usize) -> Self {
        self.wp_n = n;
        self
    }

    pub fn stance_bits(&self) -> String {
        (0..self.n_legs)
            .map(|i| if self.stance[i] { '1' } else { '0' })
            .collect()
    }
}

/// Seeded tripod on a course, for tests and the CLI.
pub fn open_loop_walk(
    frame: Frame,
    course: crate::terrain::Course,
    seed: u64,
    phys: Physics,
) -> (ArticulatedWalker, Terrain, Policy, Gait) {
    let terrain = Terrain::new(course, seed);
    let policy = Policy::seeded(Preset::default_for(frame), frame);
    let mut gait = policy.gait();
    prepare_live_gait(frame, &mut gait, &phys);
    let walker = ArticulatedWalker::spawn(frame, &gait, &phys, &terrain);
    (walker, terrain, policy, gait)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::foot_on_terrain;
    use crate::sim::DT;
    use crate::terrain::Course;

    #[test]
    fn a_world_locked_tripod_walks_instead_of_skating() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let (mut walker, terrain, policy, gait) =
            open_loop_walk(frame, Course::Flat, 1, phys);
        let cmd = Cmd {
            fwd: 1.0,
            turn: 0.0,
            cruise: 1.5,
            nav: false,
        };
        let ticks = (8.0 / DT) as usize;
        let mut slip_acc = 0.0;
        let mut slip_n = 0usize;
        let mut min_y = f64::INFINITY;
        for k in 0..ticks {
            walker.step(&terrain, &policy, &gait, DT, cmd);
            let s = walker.sample();
            if k > 80 {
                min_y = min_y.min(s.pos[1]);
                slip_acc += s.slip;
                slip_n += 1;
            }
        }
        let s = walker.sample();
        let mean_slip = slip_acc / slip_n.max(1) as f64;
        assert!(
            !s.fallen,
            "fell: y={:.3} pitch={:.3}",
            s.pos[1], s.pitch
        );
        assert!(
            min_y > 0.55,
            "sat down while walking: min_y={min_y:.3} end_y={:.3}",
            s.pos[1]
        );
        assert!(
            s.pos[2] > 1.20,
            "did not walk forward: z={:.3} x={:.3} yaw={:.3} along={:.3} slip={mean_slip:.3}",
            s.pos[2],
            s.pos[0],
            s.yaw,
            s.along
        );
        assert!(
            s.pos[0].abs() < 1.20,
            "walked off-axis: x={:.3} yaw={:.3}",
            s.pos[0],
            s.yaw
        );
        assert!(
            s.yaw.abs() < 0.45,
            "spun: yaw={:.3} x={:.3} z={:.3}",
            s.yaw,
            s.pos[0],
            s.pos[2]
        );
        // Tripod swaps spike slip for a few ticks; a skate sits at ~2 m/s
        // for the whole window.
        assert!(
            mean_slip < 1.25,
            "stance feet still skating: mean slip {mean_slip:.3} m/s (z={:.3} along={:.3})",
            s.pos[2],
            s.along
        );
    }

    #[test]
    fn foot_on_terrain_is_not_used_for_stance_locks() {
        // The walker must not command the kinematic stride sweep on a planted
        // foot: that is the ice path. A lock at touchdown is a world point.
        let frame = Frame::new(6);
        let phys = Physics::default();
        let (mut walker, terrain, policy, gait) =
            open_loop_walk(frame, Course::Flat, 1, phys);
        walker.step(&terrain, &policy, &gait, DT, Cmd::at(1.0));
        let (bp, byaw, bpitch, broll) = walker.plant.chassis_pose();
        let mut locked = 0usize;
        for i in 0..6 {
            if !walker.sim.feet[i].stance {
                continue;
            }
            let Some(lock) = walker.locks.lock[i] else {
                continue;
            };
            locked += 1;
            let swept = foot_on_terrain(
                frame,
                &gait,
                i,
                walker.sim.phase,
                walker.sim.stride_now,
                walker.sim.duty_now,
                walker.sim.cycle_now,
                gait.body_h,
                walker.sim.feet[i].step_h,
                0.0,
                &terrain,
                bp,
                byaw,
                bpitch,
                broll,
            );
            let held = to_body(lock, bp, byaw, bpitch, broll);
            let d = hypot2(swept[0] - held[0], swept[2] - held[2]);
            assert!(
                d > 0.02,
                "stance lock matched the kinematic sweep on leg {i} (d={d:.4}); \
                 the plant would skate again"
            );
        }
        assert!(locked >= 3, "tripod should plant three feet, locked={locked}");
    }
}
