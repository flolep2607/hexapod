//! Articulated hexapod plant: Rapier rigid bodies, 18 revolute joints, ground
//! friction.
//!
//! Gait, analytic IK and the servo torque-speed line stay in this crate. Rapier
//! is the engine those three drive: a chassis, three links per leg, motors on
//! every hinge, and a height-field for the course. The first milestone is a
//! hexapod that stands on a plane and then walks a tripod gait — that already
//! uses the engine; it does not require writing one.

use rapier3d::prelude::*;

use crate::dynamics::Physics;
use crate::math::V3;
use crate::policy::Gait;
use crate::robot::{
    clamp_joints, fk_world, solve_ik, Frame, BODY_H, COXA, FEMUR, MAX_LEGS, Q_LIMIT, TIBIA,
};
use crate::terrain::{Terrain, CORRIDOR_HALF, Z_MAX, Z_MIN};

const STIFF: f32 = 120.0;
const DAMP: f32 = 16.0;
const FOOT_R: f32 = 0.05;
const LINK_R: f32 = 0.05;

#[inline]
fn v(p: V3, s: f32) -> Vector {
    Vector::new(p[0] as f32 * s, p[1] as f32 * s, p[2] as f32 * s)
}

#[inline]
fn wrap(a: f32) -> f32 {
    let pi = std::f32::consts::PI;
    (a + pi).rem_euclid(2.0 * pi) - pi
}

fn hinge(
    parent: &Pose,
    child: &Pose,
    world_anchor: Vector,
    world_axis: Vector,
    limits: [f32; 2],
    stall: f32,
) -> RevoluteJoint {
    let axis = {
        let n = world_axis.length();
        if n < 1e-6 {
            Vector::Y
        } else {
            world_axis / n
        }
    };
    // Joint local X is the hinge. Building both frames from the same world
    // pose makes the spawn configuration the rest pose, so the solver is not
    // asked to unwind a 90° error on the first tick.
    let joint_rot = Rotation::from_rotation_arc(Vector::X, axis);
    let world = Pose::from_parts(world_anchor, joint_rot);
    let mut joint = RevoluteJointBuilder::new(Vector::X)
        .contacts_enabled(false)
        .limits(limits)
        .motor_model(MotorModel::ForceBased)
        .motor_position(0.0, STIFF, DAMP)
        .motor_max_force(stall)
        .build();
    joint.data.set_local_frame1(parent.inverse() * world);
    joint.data.set_local_frame2(child.inverse() * world);
    joint
}

#[inline]
fn dir_from_to(a: Vector, b: Vector) -> Vector {
    let d = b - a;
    let n = d.length();
    if n < 1e-6 {
        Vector::X
    } else {
        d / n
    }
}

/// One Rapier hinge, remembered so we can retarget it every tick.
#[derive(Clone, Copy)]
struct Hinge {
    joint: ImpulseJointHandle,
    parent: RigidBodyHandle,
    child: RigidBodyHandle,
    q0: f32,
}

struct LegBodies {
    _coxa: RigidBodyHandle,
    _femur: RigidBodyHandle,
    _tibia: RigidBodyHandle,
    _foot: ColliderHandle,
    hinges: [Hinge; 3],
}

/// Rapier world for one robot on one course.
pub struct ArticulatedPlant {
    scale: f32,
    n: usize,
    chassis: RigidBodyHandle,
    legs: [Option<LegBodies>; MAX_LEGS],
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    ccd: CCDSolver,
    integration: IntegrationParameters,
    gravity: Vector,
}

impl ArticulatedPlant {
    /// Spawn a standing robot on `terrain`, joints at the gait's neutral IK.
    pub fn standing(frame: Frame, gait: &Gait, phys: &Physics, terrain: &Terrain) -> Self {
        // Rapier is tuned for metre-scale objects. The gait already lives in
        // simulator metres (~2 m hexapod); we run the plant in that space so
        // contacts are not 1.5 cm spheres. Servo stall is a real N·m for the
        // 28 cm machine, so it is scaled by 1/scale to keep tau/(mgL).
        let s = 1.0f32;
        let stall_sim = phys.actuator.stall_nm as f32 / phys.scale as f32;
        let n = frame.legs();
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut impulse_joints = ImpulseJointSet::new();
        let multibody_joints = MultibodyJointSet::new();

        let ground = terrain.height(0.0, 0.0);
        let pos = [0.0, ground + gait.body_h + 0.22, 0.0];
        let mut q0 = [[0.0f64; 3]; MAX_LEGS];
        for i in 0..n {
            let d = frame.dir(i);
            let out = gait.stance_w * 0.5 + gait.trim(i);
            let target = [d[0] * out, -gait.body_h, d[2] * out];
            let mut q = solve_ik(frame, i, target).q;
            clamp_joints(&mut q);
            q0[i] = q;
        }

        let groups_ground = InteractionGroups::new(
            Group::GROUP_1,
            Group::GROUP_2 | Group::GROUP_3,
            InteractionTestMode::And,
        );
        let groups_chassis =
            InteractionGroups::new(Group::GROUP_2, Group::GROUP_1, InteractionTestMode::And);
        let groups_foot =
            InteractionGroups::new(Group::GROUP_3, Group::GROUP_1, InteractionTestMode::And);
        let groups_link = InteractionGroups::none();

        // --- course -----------------------------------------------------------
        let x0 = -CORRIDOR_HALF as f32 * s;
        let x1 = CORRIDOR_HALF as f32 * s;
        let z0 = Z_MIN as f32 * s;
        let z1 = Z_MAX as f32 * s;
        // One cuboid for the walkable plane. Obstacle blocks sit on top of it.
        // A heightfield plus this floor double-hit the feet and launched the
        // chassis; pits are still the centroidal learner's problem.
        let floor_h = 0.40f32;
        let floor = bodies.insert(RigidBodyBuilder::fixed().translation(Vector::new(
            0.0,
            ground as f32 - floor_h,
            0.5 * (z0 + z1),
        )));
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(x1 - x0 + 4.0, floor_h, 0.5 * (z1 - z0) + 4.0)
                .friction(phys.mu as f32)
                .restitution(0.0)
                .collision_groups(groups_ground),
            floor,
            &mut bodies,
        );
        for ob in &terrain.obstacles {
            if ob.top <= 0.04 {
                continue;
            }
            let hx = 0.5 * (ob.x1 - ob.x0) as f32 * s;
            let hz = 0.5 * (ob.z1 - ob.z0) as f32 * s;
            let hy = 0.5 * ob.top as f32 * s;
            if hx < 0.02 || hz < 0.02 || hy < 0.02 {
                continue;
            }
            let block = bodies.insert(RigidBodyBuilder::fixed().translation(Vector::new(
                0.5 * (ob.x0 + ob.x1) as f32 * s,
                ground as f32 + hy,
                0.5 * (ob.z0 + ob.z1) as f32 * s,
            )));
            colliders.insert_with_parent(
                ColliderBuilder::cuboid(hx, hy, hz)
                    .friction((phys.mu * ob.grip) as f32)
                    .restitution(0.0)
                    .collision_groups(groups_ground),
                block,
                &mut bodies,
            );
        }

        let wall_h = 4.0 * s;
        let wall_t = 0.15 * s;
        for &side in &[-1.0f32, 1.0] {
            let wb = bodies.insert(RigidBodyBuilder::fixed().translation(Vector::new(
                side * (CORRIDOR_HALF as f32 * s + wall_t),
                wall_h,
                0.5 * (z0 + z1),
            )));
            colliders.insert_with_parent(
                ColliderBuilder::cuboid(wall_t, wall_h, 0.5 * (z1 - z0))
                    .friction(phys.mu as f32)
                    .collision_groups(groups_ground),
                wb,
                &mut bodies,
            );
        }

        // --- chassis ----------------------------------------------------------
        let swing = phys.swing_mass(frame) as f32;
        let chassis_kg = (phys.mass_kg as f32 - swing).max(0.15);
        let body_r = frame.body_r() as f32 * s;
        let body_h = BODY_H as f32 * s;
        let chassis = bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(v(pos, s))
                .additional_mass(chassis_kg)
                .can_sleep(false)
                .linear_damping(0.15)
                .angular_damping(0.40),
        );
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(body_r * 0.72, body_h * 0.45, body_r * 0.72)
                .mass(chassis_kg)
                .friction(0.3)
                .collision_groups(groups_chassis),
            chassis,
            &mut bodies,
        );

        let femur_kg = (phys.leg.femur_kg as f32).max(0.008);
        let tibia_kg = (phys.leg.tibia_kg as f32).max(0.008);
        let coxa_kg = 0.012f32;

        let mut legs: [Option<LegBodies>; MAX_LEGS] = std::array::from_fn(|_| None);
        let stall = stall_sim;

        for i in 0..n {
            let jw = fk_world(frame, i, q0[i], pos, 0.0, 0.0, 0.0);
            let hip = v(jw[0], s);
            let knee = v(jw[1], s);
            let ankle = v(jw[2], s);
            let foot = v(jw[3], s);

            let coxa_mid = 0.5 * (hip + knee);
            let femur_mid = 0.5 * (knee + ankle);
            let tibia_mid = 0.5 * (ankle + foot);

            let coxa_dir = dir_from_to(hip, knee);
            let femur_dir = dir_from_to(knee, ankle);
            let tibia_dir = dir_from_to(ankle, foot);
            let pitch_axis = {
                let a = Vector::Y.cross(coxa_dir);
                let n = a.length();
                if n < 1e-5 {
                    Vector::Z
                } else {
                    a / n
                }
            };

            let coxa_rot = Rotation::from_rotation_arc(Vector::Y, coxa_dir);
            let femur_rot = Rotation::from_rotation_arc(Vector::Y, femur_dir);
            let tibia_rot = Rotation::from_rotation_arc(Vector::Y, tibia_dir);

            let coxa = bodies.insert(
                RigidBodyBuilder::dynamic()
                    .translation(coxa_mid)
                    .rotation(coxa_rot.to_scaled_axis())
                    .additional_mass(coxa_kg)
                    .can_sleep(false)
                    .linear_damping(0.15)
                    .angular_damping(0.40),
            );
            let femur = bodies.insert(
                RigidBodyBuilder::dynamic()
                    .translation(femur_mid)
                    .rotation(femur_rot.to_scaled_axis())
                    .additional_mass(femur_kg)
                    .can_sleep(false)
                    .linear_damping(0.15)
                    .angular_damping(0.40),
            );
            let tibia = bodies.insert(
                RigidBodyBuilder::dynamic()
                    .translation(tibia_mid)
                    .rotation(tibia_rot.to_scaled_axis())
                    .additional_mass(tibia_kg)
                    .can_sleep(false)
                    .linear_damping(0.15)
                    .angular_damping(0.40),
            );

            let coxa_len = (COXA as f32 * s * 0.5).max(0.004);
            let femur_len = (FEMUR as f32 * s * 0.5).max(0.004);
            let tibia_len = (TIBIA as f32 * s * 0.5).max(0.004);
            let thick = LINK_R * s;

            colliders.insert_with_parent(
                ColliderBuilder::capsule_y(coxa_len, thick)
                    .mass(coxa_kg)
                    .friction(0.2)
                    .collision_groups(groups_link),
                coxa,
                &mut bodies,
            );
            colliders.insert_with_parent(
                ColliderBuilder::capsule_y(femur_len, thick)
                    .mass(femur_kg)
                    .friction(0.2)
                    .collision_groups(groups_link),
                femur,
                &mut bodies,
            );
            colliders.insert_with_parent(
                ColliderBuilder::capsule_y(tibia_len, thick)
                    .mass(tibia_kg)
                    .friction(0.2)
                    .collision_groups(groups_link),
                tibia,
                &mut bodies,
            );
            let foot_col = colliders.insert_with_parent(
                ColliderBuilder::ball(FOOT_R * s)
                    .mass(tibia_kg * 0.15)
                    .friction(phys.mu as f32)
                    .restitution(0.0)
                    .collision_groups(groups_foot)
                    .translation(Vector::new(0.0, tibia_len + FOOT_R * s * 0.35, 0.0)),
                tibia,
                &mut bodies,
            );

            let chassis_pose = *bodies[chassis].position();
            let coxa_pose = *bodies[coxa].position();
            let femur_pose = *bodies[femur].position();
            let tibia_pose = *bodies[tibia].position();

            let coxa_joint = hinge(
                &chassis_pose,
                &coxa_pose,
                hip,
                Vector::Y,
                [Q_LIMIT[0].0 as f32, Q_LIMIT[0].1 as f32],
                stall,
            );
            let femur_joint = hinge(
                &coxa_pose,
                &femur_pose,
                knee,
                pitch_axis,
                [Q_LIMIT[1].0 as f32, Q_LIMIT[1].1 as f32],
                stall,
            );
            let tibia_joint = hinge(
                &femur_pose,
                &tibia_pose,
                ankle,
                pitch_axis,
                [Q_LIMIT[2].0 as f32, Q_LIMIT[2].1 as f32],
                stall,
            );

            let h_coxa = impulse_joints.insert(chassis, coxa, coxa_joint, true);
            let h_femur = impulse_joints.insert(coxa, femur, femur_joint, true);
            let h_tibia = impulse_joints.insert(femur, tibia, tibia_joint, true);

            legs[i] = Some(LegBodies {
                _coxa: coxa,
                _femur: femur,
                _tibia: tibia,
                _foot: foot_col,
                hinges: [
                    Hinge {
                        joint: h_coxa,
                        parent: chassis,
                        child: coxa,
                        q0: q0[i][0] as f32,
                    },
                    Hinge {
                        joint: h_femur,
                        parent: coxa,
                        child: femur,
                        q0: q0[i][1] as f32,
                    },
                    Hinge {
                        joint: h_tibia,
                        parent: femur,
                        child: tibia,
                        q0: q0[i][2] as f32,
                    },
                ],
            });
        }

        let integration = IntegrationParameters {
            dt: crate::sim::DT as f32,
            num_solver_iterations: 12,
            num_internal_pgs_iterations: 2,
            length_unit: 1.0,
            ..Default::default()
        };

        ArticulatedPlant {
            scale: s,
            n,
            chassis,
            legs,
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            ccd: CCDSolver::new(),
            integration,
            gravity: Vector::new(0.0, -9.81, 0.0),
        }
    }

    /// Drive every hinge toward `q_cmd` with the servo's stall torque as the cap.
    pub fn drive(&mut self, q_cmd: &[[f64; 3]; MAX_LEGS], phys: &Physics) {
        let stall = phys.actuator.stall_nm as f32 / phys.scale as f32;
        let omega_max = phys.actuator.omega_max as f32;
        for i in 0..self.n {
            let Some(leg) = self.legs[i].as_ref() else {
                continue;
            };
            for j in 0..3 {
                let h = leg.hinges[j];
                let target = wrap(q_cmd[i][j] as f32 - h.q0);
                let Some(joint) = self.impulse_joints.get_mut(h.joint, true) else {
                    continue;
                };
                let Some(rev) = joint.data.as_revolute_mut() else {
                    continue;
                };
                let parent = &self.bodies[h.parent];
                let child = &self.bodies[h.child];
                let w = child.angvel() - parent.angvel();
                let axis = {
                    let p = *parent.rotation() * Vector::Y;
                    // Approximate: use relative spin about the hinge.
                    p
                };
                let omega = w.dot(axis);
                let avail = stall * (1.0 - (omega.abs() / omega_max).clamp(0.0, 1.0));
                rev.set_motor_position(target, STIFF, DAMP);
                rev.set_motor_max_force(avail.max(0.02 * stall));
            }
        }
    }

    pub fn step(&mut self, dt: f64) {
        self.integration.dt = dt as f32;
        self.pipeline.step(
            self.gravity,
            &self.integration,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd,
            &(),
            &(),
        );
    }

    fn joint_angle(&self, h: Hinge) -> f32 {
        let Some(joint) = self.impulse_joints.get(h.joint) else {
            return h.q0;
        };
        let Some(rev) = joint.data.as_revolute() else {
            return h.q0;
        };
        let a = rev.angle(
            self.bodies[h.parent].rotation(),
            self.bodies[h.child].rotation(),
        );
        h.q0 + a
    }

    /// Rapier hinge angles for one leg, in the same coxa/femur/tibia order as
    /// the centroidal plant. The dashboard overlays these on the gait-clock
    /// body pose; it must not write them back into `Sim` or a missed motor
    /// track (high speed, a quadruped tip) rewinds the integrator.
    pub fn leg_q(&self, i: usize) -> [f64; 3] {
        let Some(leg) = self.legs.get(i).and_then(|l| l.as_ref()) else {
            return [0.0; 3];
        };
        [
            self.joint_angle(leg.hinges[0]) as f64,
            self.joint_angle(leg.hinges[1]) as f64,
            self.joint_angle(leg.hinges[2]) as f64,
        ]
    }

    /// Chassis height in simulator units.
    pub fn chassis_y(&self) -> f64 {
        self.bodies[self.chassis].translation().y as f64 / self.scale as f64
    }

    /// Horizontal progress along +Z, simulator units.
    pub fn chassis_z(&self) -> f64 {
        self.bodies[self.chassis].translation().z as f64 / self.scale as f64
    }

    pub fn pitch_abs(&self) -> f64 {
        let fwd = *self.bodies[self.chassis].rotation() * Vector::Z;
        (fwd.y as f64).abs()
    }
}

/// Body-frame foot target for an open-loop gait. Stance sweeps aft, swing
/// returns with a sine lift. This is the whole walking programme: IK turns it
/// into 18 joint angles, Rapier does the rest.
pub fn foot_in_body(
    frame: Frame,
    gait: &Gait,
    leg: usize,
    phase: f64,
    stride: f64,
    duty: f64,
    body_h: f64,
    step_h: f64,
) -> V3 {
    use crate::math::frac;
    let d = frame.dir(leg);
    let out = gait.stance_w * 0.5 + gait.trim(leg);
    let lp = frac(phase + gait.offsets[leg]);
    let (long, y) = if lp < duty {
        let u = lp / duty.max(1e-6);
        (stride * (u - 0.5), -body_h)
    } else {
        let u = (lp - duty) / (1.0 - duty).max(1e-6);
        (
            stride * (0.5 - u),
            -body_h + step_h * (core::f64::consts::PI * u).sin(),
        )
    };
    [d[0] * out, y, d[2] * out + long]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Policy, Preset};
    use crate::robot::Frame;
    use crate::terrain::{Course, Terrain};

    fn hold(plant: &mut ArticulatedPlant, q: &[[f64; 3]; MAX_LEGS], phys: &Physics, secs: f64) {
        let n = (secs / crate::sim::DT).round() as usize;
        for _ in 0..n {
            plant.drive(q, phys);
            plant.step(crate::sim::DT);
        }
    }

    #[test]
    fn a_hexapod_stands_on_a_plane() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);

        let mut q = [[0.0; 3]; MAX_LEGS];
        for i in 0..6 {
            let d = frame.dir(i);
            let out = gait.stance_w * 0.5;
            q[i] = solve_ik(frame, i, [d[0] * out, -gait.body_h, d[2] * out]).q;
        }
        let y0 = plant.chassis_y();
        hold(&mut plant, &q, &phys, 1.2);
        let y1 = plant.chassis_y();
        assert!(
            plant.pitch_abs() < 0.35,
            "tipped over: pitch {} y0={y0:.3} y1={y1:.3}",
            plant.pitch_abs()
        );
        assert!(
            (0.50..1.40).contains(&y1),
            "chassis not standing: y0={y0:.3} y1={y1:.3}"
        );
    }

    #[test]
    fn a_tripod_gait_walks_forward() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);

        let z0 = plant.chassis_z();
        let mut phase = 0.0;
        let ticks = (4.0 / crate::sim::DT) as usize;
        for _ in 0..ticks {
            phase = crate::math::frac(phase + crate::sim::DT / gait.cycle);
            let mut q = [[0.0; 3]; MAX_LEGS];
            for i in 0..6 {
                let target = foot_in_body(
                    frame, &gait, i, phase, gait.stride, gait.duty, gait.body_h, gait.step_h,
                );
                q[i] = solve_ik(frame, i, target).q;
            }
            plant.drive(&q, &phys);
            plant.step(crate::sim::DT);
        }
        let dz = plant.chassis_z() - z0;
        assert!(
            dz > 0.8,
            "tripod did not walk: Δz={dz:.3} (want > 0.8 sim-m in 4 s)"
        );
        assert!(plant.pitch_abs() < 0.55, "fell while walking");
    }

    #[test]
    fn hexapod_floor_catches_a_chassis() {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut impulse_joints = ImpulseJointSet::new();
        let mut multibody_joints = MultibodyJointSet::new();
        let mut pipeline = PhysicsPipeline::new();
        let mut islands = IslandManager::new();
        let mut broad = BroadPhaseBvh::new();
        let mut narrow = NarrowPhase::new();
        let mut ccd = CCDSolver::new();
        let integration = IntegrationParameters { dt: 0.01, ..Default::default() };
        let floor_h = 0.40f32;
        let floor = bodies.insert(RigidBodyBuilder::fixed().translation(Vector::new(0.0, -floor_h, 29.0)));
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(14.0, floor_h, 40.0).friction(0.85),
            floor,
            &mut bodies,
        );
        let ch = bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 1.1, 0.0))
                .additional_mass(1.0)
                .can_sleep(false),
        );
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(0.7, 0.13, 0.7).mass(1.0).friction(0.3),
            ch,
            &mut bodies,
        );
        for _ in 0..200 {
            pipeline.step(
                Vector::new(0.0, -9.81, 0.0),
                &integration,
                &mut islands,
                &mut broad,
                &mut narrow,
                &mut bodies,
                &mut colliders,
                &mut impulse_joints,
                &mut multibody_joints,
                &mut ccd,
                &(),
                &(),
            );
        }
        let y = bodies[ch].translation().y;
        assert!(y > 0.05 && y < 0.6, "chassis-only y={y}");
    }

    #[test]
    fn a_box_lands_on_the_floor() {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut impulse_joints = ImpulseJointSet::new();
        let mut multibody_joints = MultibodyJointSet::new();
        let mut pipeline = PhysicsPipeline::new();
        let mut islands = IslandManager::new();
        let mut broad = BroadPhaseBvh::new();
        let mut narrow = NarrowPhase::new();
        let mut ccd = CCDSolver::new();
        let integration = IntegrationParameters { dt: 0.01, ..Default::default() };
        let floor = bodies.insert(RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5, 0.0)));
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(20.0, 0.5, 20.0).friction(0.8),
            floor,
            &mut bodies,
        );
        let ball = bodies.insert(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 2.0, 0.0)));
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(0.2, 0.2, 0.2).mass(2.0),
            ball,
            &mut bodies,
        );
        for _ in 0..200 {
            pipeline.step(
                Vector::new(0.0, -9.81, 0.0),
                &integration,
                &mut islands,
                &mut broad,
                &mut narrow,
                &mut bodies,
                &mut colliders,
                &mut impulse_joints,
                &mut multibody_joints,
                &mut ccd,
                &(),
                &(),
            );
        }
        let y = bodies[ball].translation().y;
        assert!(y > 0.05 && y < 0.8, "box did not land, y={y}");
    }

    #[test]
    fn eighteen_revolute_hinges() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
        let mut hinges = 0usize;
        for i in 0..6 {
            hinges += plant.legs[i].as_ref().unwrap().hinges.len();
        }
        assert_eq!(hinges, 18);
    }
}
