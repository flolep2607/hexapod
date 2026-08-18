//! Articulated hexapod plant: Rapier rigid bodies, 18 revolute joints, ground
//! friction.
//!
//! Gait, analytic IK and the servo torque-speed line stay in this crate. Rapier
//! is the engine those three drive: a chassis, three links per leg, motors on
//! every hinge, and a height-field for the course. Links collide with terrain
//! and the chassis; adjacent parent/child contacts stay off. The canvas reads
//! Rapier body transforms and must not write them into `Sim`.

use rapier3d::prelude::*;

use crate::dynamics::Physics;
use crate::math::V3;
use crate::policy::Gait;
use crate::robot::{
    clamp_joints, fk_world, solve_ik, Frame, BODY_H, COXA, FEMUR, FOOT_R as FOOT_R_M, LINK_R,
    MAX_LEGS, Q_LIMIT, TIBIA,
};
use crate::terrain::{Terrain, CORRIDOR_HALF, Z_MAX, Z_MIN};

/// Motor gains, as multiples of the joint's stall torque: an error of a sixth
/// of a radian asks for stall. Tied to stall so they follow the plant's length
/// scale instead of having to be retuned alongside it.
const STIFF_PER_STALL: f32 = 6.1;
const DAMP_PER_STALL: f32 = 0.82;
const FOOT_R: f32 = FOOT_R_M as f32;
/// Collider `user_data`: the walkable plane. Any chassis contact is fatal.
const HIT_FLOOR: u128 = 1;
/// Collider `user_data`: a block or corridor wall. Fatal only on a hard hit.
const HIT_SOLID: u128 = 2;
/// Closing speed, m/s, at which a chassis-vs-solid contact kills the machine.
const IMPACT_KILL: f32 = 1.2;

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
    // Negated on purpose. `crate::math::rot_y` turns +x toward +z, which is a
    // left-handed rotation about Y; Rapier is right-handed. Left as it comes,
    // every hinge turns the mirror image of the analytic joint angle — and since
    // the angle is read back through the same mirror it still reads correct,
    // so nothing catches it except comparing the real link poses against
    // `fk_world`, which is what `rapier_kinematics_match_the_analytic_model`
    // does. The spawn pose is angle-independent and so looked fine, while every
    // step, foothold and obstacle dodge was computed for a robot mirrored
    // fore-and-aft from the one on screen.
    let axis = {
        let n = world_axis.length();
        if n < 1e-6 {
            -Vector::Y
        } else {
            -world_axis / n
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
        .motor_position(0.0, STIFF_PER_STALL * stall, DAMP_PER_STALL * stall)
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
    /// The servo's own setpoint, relative to `q0`. It slews toward the command
    /// at the servo's no-load speed and is deliberately independent of where
    /// the joint actually is: a setpoint clamped against the measured angle can
    /// never build up an error, so the motor stays weak and the leg mushes.
    set: f32,
}

#[cfg_attr(not(test), allow(dead_code))]
struct LegBodies {
    _coxa: RigidBodyHandle,
    _femur: RigidBodyHandle,
    tibia: RigidBodyHandle,
    foot: ColliderHandle,
    hinges: [Hinge; 3],
}

/// Rapier world for one robot on one course.
pub struct ArticulatedPlant {
    scale: f32,
    n: usize,
    chassis: RigidBodyHandle,
    chassis_col: ColliderHandle,
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
        // contacts are not 1.5 cm spheres.
        //
        // Servo stall is a real N·m for the 28 cm machine, so it is scaled by
        // 1/scale to keep tau/(mgL).
        let s = 1.0f32;
        let stall_sim = phys.actuator.stall_nm as f32 / phys.scale as f32;
        let n = frame.legs();
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut impulse_joints = ImpulseJointSet::new();
        let multibody_joints = MultibodyJointSet::new();

        let ground = terrain.height(0.0, 0.0);
        let pos = [0.0, ground + gait.body_h, 0.0];
        let mut q0 = [[0.0f64; 3]; MAX_LEGS];
        for i in 0..n {
            let d = frame.dir(i);
            let out = gait.stance_w * 0.5 + gait.trim(i);
            let target = [d[0] * out, -gait.body_h + crate::robot::FOOT_R, d[2] * out];
            let mut q = solve_ik(frame, i, target).q;
            clamp_joints(&mut q);
            q0[i] = q;
        }

        // 1 = terrain, 2 = chassis, 3 = feet, 4 = links. Adjacent parent/child
        // still skip via contacts_enabled(false) on the hinge. Floor hits
        // chassis and feet only — tibia capsules on the plane skate. Solids
        // (crates, walls) still hit links so a tibia cannot occupy a block.
        let groups_floor = InteractionGroups::new(
            Group::GROUP_1,
            Group::GROUP_2 | Group::GROUP_3,
            InteractionTestMode::And,
        );
        let groups_solid = InteractionGroups::new(
            Group::GROUP_1,
            Group::GROUP_2 | Group::GROUP_3 | Group::GROUP_4,
            InteractionTestMode::And,
        );
        let groups_chassis = InteractionGroups::new(
            Group::GROUP_2,
            Group::GROUP_1 | Group::GROUP_4,
            InteractionTestMode::And,
        );
        let groups_foot = InteractionGroups::new(
            Group::GROUP_3,
            Group::GROUP_1 | Group::GROUP_3 | Group::GROUP_4,
            InteractionTestMode::And,
        );
        let groups_link = InteractionGroups::new(
            Group::GROUP_4,
            Group::GROUP_1 | Group::GROUP_2 | Group::GROUP_3 | Group::GROUP_4,
            InteractionTestMode::And,
        );

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
                .friction_combine_rule(CoefficientCombineRule::Max)
                .restitution(0.0)
                .collision_groups(groups_floor)
                .user_data(HIT_FLOOR),
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
                    .collision_groups(groups_solid)
                    .user_data(HIT_SOLID),
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
                    .collision_groups(groups_solid)
                    .user_data(HIT_SOLID),
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
                .can_sleep(false)
                .ccd_enabled(true)
                .linear_damping(0.05)
                .angular_damping(0.32),
        );
        // A cylinder matches the visual disc. The old cube's corners stuck out
        // past the hips and scraped every wall the body walked past.
        let chassis_col = colliders.insert_with_parent(
            ColliderBuilder::cylinder(body_h * 0.40, body_r * 0.70)
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
                    .can_sleep(false)
                    .linear_damping(0.08)
                    .angular_damping(0.20),
            );
            let femur = bodies.insert(
                RigidBodyBuilder::dynamic()
                    .translation(femur_mid)
                    .rotation(femur_rot.to_scaled_axis())
                    .can_sleep(false)
                    .linear_damping(0.08)
                    .angular_damping(0.20),
            );
            let tibia = bodies.insert(
                RigidBodyBuilder::dynamic()
                    .translation(tibia_mid)
                    .rotation(tibia_rot.to_scaled_axis())
                    .can_sleep(false)
                    .ccd_enabled(true)
                    .linear_damping(0.08)
                    .angular_damping(0.20),
            );

            // Kinematic half-length. Rapier's capsule_y half-height is the
            // cylinder, plus hemispheres of `thick`, so feeding L/2 made every
            // link stick past its hinges and kick walls the planner had cleared.
            let coxa_half = (COXA as f32 * s * 0.5).max(0.004);
            let femur_half = (FEMUR as f32 * s * 0.5).max(0.004);
            let tibia_half = (TIBIA as f32 * s * 0.5).max(0.004);
            let thick = LINK_R as f32 * s;
            let cap = |half: f32| (half - thick).max(0.008);

            colliders.insert_with_parent(
                ColliderBuilder::capsule_y(cap(coxa_half), thick)
                    .mass(coxa_kg)
                    .friction(0.2)
                    .collision_groups(groups_link),
                coxa,
                &mut bodies,
            );
            colliders.insert_with_parent(
                ColliderBuilder::capsule_y(cap(femur_half), thick)
                    .mass(femur_kg)
                    .friction(0.2)
                    .collision_groups(groups_link),
                femur,
                &mut bodies,
            );
            colliders.insert_with_parent(
                ColliderBuilder::capsule_y(cap(tibia_half), thick)
                    .mass(tibia_kg)
                    .friction(0.2)
                    .collision_groups(groups_link),
                tibia,
                &mut bodies,
            );
            let foot_col = colliders.insert_with_parent(
                ColliderBuilder::ball(FOOT_R * s)
                    .mass(tibia_kg * 0.15)
                    .friction((phys.mu as f32).max(1.15))
                    .friction_combine_rule(CoefficientCombineRule::Max)
                    .restitution(0.0)
                    .collision_groups(groups_foot)
                    // Centered on the kinematic foot so a tilted tibia still
                    // puts rubber on the plane. Insetting along the tibia left
                    // the contact patch in the air and the machine skated.
                    .translation(Vector::new(0.0, tibia_half, 0.0)),
                tibia,
                &mut bodies,
            );

            let chassis_pose = *bodies[chassis].position();
            let coxa_pose = *bodies[coxa].position();
            let femur_pose = *bodies[femur].position();
            let tibia_pose = *bodies[tibia].position();

            // A hinge's zero is its spawn pose, so `Q_LIMIT` — which is in
            // absolute joint angles — has to be shifted by q0. Feeding it raw
            // put the band a whole radian off on the pitch joints and pinned
            // the tibia against a bound it should never have reached, which is
            // what made every leg twitch instead of step.
            let lim = |k: usize| {
                [
                    (Q_LIMIT[k].0 - q0[i][k]) as f32,
                    (Q_LIMIT[k].1 - q0[i][k]) as f32,
                ]
            };
            let coxa_joint = hinge(&chassis_pose, &coxa_pose, hip, Vector::Y, lim(0), stall);
            let femur_joint = hinge(&coxa_pose, &femur_pose, knee, pitch_axis, lim(1), stall);
            let tibia_joint = hinge(&femur_pose, &tibia_pose, ankle, pitch_axis, lim(2), stall);

            let h_coxa = impulse_joints.insert(chassis, coxa, coxa_joint, true);
            let h_femur = impulse_joints.insert(coxa, femur, femur_joint, true);
            let h_tibia = impulse_joints.insert(femur, tibia, tibia_joint, true);

            legs[i] = Some(LegBodies {
                _coxa: coxa,
                _femur: femur,
                tibia,
                foot: foot_col,
                hinges: [
                    Hinge {
                        joint: h_coxa,
                        parent: chassis,
                        child: coxa,
                        q0: q0[i][0] as f32,
                        set: 0.0,
                    },
                    Hinge {
                        joint: h_femur,
                        parent: coxa,
                        child: femur,
                        q0: q0[i][1] as f32,
                        set: 0.0,
                    },
                    Hinge {
                        joint: h_tibia,
                        parent: femur,
                        child: tibia,
                        q0: q0[i][2] as f32,
                        set: 0.0,
                    },
                ],
            });
        }

        let integration = IntegrationParameters {
            dt: crate::sim::DT as f32,
            num_solver_iterations: 16,
            num_internal_pgs_iterations: 4,
            length_unit: 1.0,
            ..Default::default()
        };

        ArticulatedPlant {
            scale: s,
            n,
            chassis,
            chassis_col,
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
    ///
    /// `dt` is the step the caller is about to take: the target is rate-limited
    /// to the servo's no-load speed over it. Without that the motor is asked to
    /// cross a whole swing in one tick, the derate below sees a joint moving
    /// past `omega_max`, and the force cap collapses to its floor exactly when
    /// the leg needs to move — the whole machine ends up twitching in place
    /// instead of walking. Rate-limiting first is what a servo actually does.
    pub fn drive(&mut self, q_cmd: &[[f64; 3]; MAX_LEGS], phys: &Physics, dt: f64) {
        let stall = phys.actuator.stall_nm as f32 / phys.scale as f32;
        let omega_max = phys.actuator.omega_max as f32;
        let max_step = omega_max * dt as f32;
        for i in 0..self.n {
            for j in 0..3 {
                let Some(leg) = self.legs[i].as_ref() else {
                    continue;
                };
                let h = leg.hinges[j];
                let at = self.joint_angle(h) - h.q0;
                let want = wrap(q_cmd[i][j] as f32 - h.q0);
                let target = h.set + wrap(want - h.set).clamp(-max_step, max_step);
                if let Some(leg) = self.legs[i].as_mut() {
                    leg.hinges[j].set = target;
                }
                let Some(joint) = self.impulse_joints.get_mut(h.joint, true) else {
                    continue;
                };
                let Some(rev) = joint.data.as_revolute_mut() else {
                    continue;
                };
                // The hinge is joint-local X, so the world axis is the parent's
                // rotation applied to frame 1's X — not the parent's Y, which
                // only matched the coxa and read chassis roll as femur speed,
                // collapsing the torque cap to the 2% floor on every pitch
                // joint.
                let hinge_axis = rev.data.local_axis1();
                let parent = &self.bodies[h.parent];
                let child = &self.bodies[h.child];
                let w = child.angvel() - parent.angvel();
                let axis = *parent.rotation() * hinge_axis;
                let omega = w.dot(axis);
                // A motor loses torque only in the direction it is already
                // spinning; braking or holding against the spin has its full
                // stall available. Derating both ways left the joint with no
                // torque to stop an overshoot, so the legs flailed past target
                // and the body sagged.
                let along = (target - at) * omega > 0.0;
                let avail = if along {
                    stall * (1.0 - (omega.abs() / omega_max).clamp(0.0, 1.0))
                } else {
                    stall
                };
                rev.set_motor_position(target, STIFF_PER_STALL * stall, DAMP_PER_STALL * stall);
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
    /// the centroidal plant. Telemetry may read these; `Sim` must not.
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

    fn hinge_world(&self, h: Hinge) -> Vector {
        let Some(joint) = self.impulse_joints.get(h.joint) else {
            return self.bodies[h.parent].translation();
        };
        let parent = &self.bodies[h.parent];
        parent.translation() + *parent.rotation() * joint.data.local_anchor1()
    }

    fn to_sim(&self, p: Vector) -> V3 {
        let s = self.scale as f64;
        [p.x as f64 / s, p.y as f64 / s, p.z as f64 / s]
    }

    /// Hip, knee, ankle, foot in simulator world units, from Rapier bodies.
    ///
    /// The foot is the kinematic distal end of the tibia, not the contact
    /// sphere's centre — that ball sits `FOOT_R` inward so it rests on the
    /// plane, and drawing it as the foot shortened every tibia by a centimetre.
    pub fn leg_joints_world(&self, i: usize) -> [V3; 4] {
        let Some(leg) = self.legs.get(i).and_then(|l| l.as_ref()) else {
            return [[0.0; 3]; 4];
        };
        let tibia = &self.bodies[leg.tibia];
        let half = (TIBIA as f32 * self.scale * 0.5).max(0.004);
        let foot = tibia.translation() + *tibia.rotation() * Vector::new(0.0, half, 0.0);
        [
            self.to_sim(self.hinge_world(leg.hinges[0])),
            self.to_sim(self.hinge_world(leg.hinges[1])),
            self.to_sim(self.hinge_world(leg.hinges[2])),
            self.to_sim(foot),
        ]
    }

    /// Chassis translation and yaw/pitch/roll matching [`crate::math::body_to_world`].
    /// Canvas-only: do not write this into `Sim`.
    pub fn chassis_pose(&self) -> (V3, f64, f64, f64) {
        let body = &self.bodies[self.chassis];
        let pos = self.to_sim(body.translation());
        let fwd = *body.rotation() * Vector::Z;
        let yaw = (-fwd.x).atan2(fwd.z) as f64;
        let pitch = -fwd.y.clamp(-1.0, 1.0).asin() as f64;
        let right = *body.rotation() * Vector::X;
        let (s, c) = yaw.sin_cos();
        let rx = right.x as f64 * c + right.z as f64 * s;
        let ry = right.y as f64;
        let roll = ry.atan2(rx);
        (pos, yaw, pitch, roll)
    }

    /// Chassis linear velocity, simulator units per second.
    pub fn chassis_vel(&self) -> V3 {
        self.to_sim(self.bodies[self.chassis].linvel())
    }

    /// Chassis height in simulator units.
    pub fn chassis_y(&self) -> f64 {
        self.bodies[self.chassis].translation().y as f64 / self.scale as f64
    }

    /// Horizontal progress along +Z, simulator units.
    pub fn chassis_z(&self) -> f64 {
        self.bodies[self.chassis].translation().z as f64 / self.scale as f64
    }

    /// Chassis angular velocity, rad/s.
    pub fn chassis_angvel(&self) -> V3 {
        let w = self.bodies[self.chassis].angvel();
        [w.x as f64, w.y as f64, w.z as f64]
    }

    /// Mean horizontal speed of feet that are touching the walkable plane, m/s,
    /// taken at the rubber rather than the tibia COM.
    pub fn foot_slip(&self) -> f64 {
        let mut slip = 0.0f64;
        let mut n = 0usize;
        for i in 0..self.n {
            let Some(leg) = self.legs[i].as_ref() else {
                continue;
            };
            let touching = self.narrow_phase.contact_pairs_with(leg.foot).any(|pair| {
                if !pair.has_any_active_contact() {
                    return false;
                }
                let other = if pair.collider1 == leg.foot {
                    pair.collider2
                } else {
                    pair.collider1
                };
                self.colliders[other].user_data == HIT_FLOOR
            });
            if !touching {
                continue;
            }
            let tibia = &self.bodies[leg.tibia];
            let half = (TIBIA as f32 * self.scale * 0.5).max(0.004);
            let offset = *tibia.rotation() * Vector::new(0.0, half, 0.0);
            let v = tibia.linvel() + tibia.angvel().cross(offset);
            let s = self.scale as f64;
            slip += ((v.x * v.x + v.z * v.z).sqrt() as f64) / s;
            n += 1;
        }
        if n == 0 {
            0.0
        } else {
            slip / n as f64
        }
    }

    pub fn pitch_abs(&self) -> f64 {
        let fwd = *self.bodies[self.chassis].rotation() * Vector::Z;
        (fwd.y as f64).abs()
    }

    /// Belly on the floor, or a chassis hit on a block/wall faster than
    /// [`IMPACT_KILL`]. `pre_vel` is the chassis velocity *before* the step
    /// that produced the contacts — after the solver the normal component is
    /// already gone.
    pub fn chassis_dead(&self, pre_vel: V3) -> bool {
        let vel = Vector::new(pre_vel[0] as f32, pre_vel[1] as f32, pre_vel[2] as f32);
        let col = self.chassis_col;
        for pair in self.narrow_phase.contact_pairs_with(col) {
            if !pair.has_any_active_contact() {
                continue;
            }
            let other = if pair.collider1 == col {
                pair.collider2
            } else {
                pair.collider1
            };
            let kind = self.colliders[other].user_data;
            if kind == HIT_FLOOR {
                return true;
            }
            if kind != HIT_SOLID {
                continue;
            }
            for m in &pair.manifolds {
                let n = m.data.normal;
                let closing = if pair.collider1 == col {
                    vel.dot(n)
                } else {
                    -vel.dot(n)
                };
                if closing > IMPACT_KILL {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{foot_in_body, foot_on_terrain, Policy, Preset};
    use crate::robot::Frame;
    use crate::terrain::{Course, Obstacle, Terrain};

    fn phys_omega() -> f64 {
        Physics::default().actuator.omega_max
    }

    fn hold(plant: &mut ArticulatedPlant, q: &[[f64; 3]; MAX_LEGS], phys: &Physics, secs: f64) {
        let n = (secs / crate::sim::DT).round() as usize;
        for _ in 0..n {
            plant.drive(q, phys, crate::sim::DT);
            plant.step(crate::sim::DT);
        }
    }

    fn standing_q(frame: Frame, gait: &Gait) -> [[f64; 3]; MAX_LEGS] {
        let mut q = [[0.0; 3]; MAX_LEGS];
        for i in 0..frame.legs() {
            let d = frame.dir(i);
            let out = gait.stance_w * 0.5;
            q[i] = solve_ik(frame, i, [d[0] * out, -gait.body_h + crate::robot::FOOT_R, d[2] * out]).q;
        }
        q
    }

    fn in_core(p: V3, ob: &Obstacle, margin: f64) -> bool {
        p[0] > ob.x0 + margin
            && p[0] < ob.x1 - margin
            && p[2] > ob.z0 + margin
            && p[2] < ob.z1 - margin
            && p[1] > margin
            && p[1] < ob.top - margin
    }

    #[test]
    fn a_hexapod_stands_on_a_plane() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);

        let q = standing_q(frame, &gait);
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
        let phys = Physics::default();
        let (mut walker, terrain, policy, gait) =
            crate::walker::open_loop_walk(frame, Course::Flat, 1, phys);
        let cmd = crate::sim::Cmd {
            fwd: 1.0,
            turn: 0.0,
            cruise: 1.5,
            nav: false,
        };
        let ticks = (8.0 / crate::sim::DT) as usize;
        for _ in 0..ticks {
            walker.step(&terrain, &policy, &gait, crate::sim::DT, cmd);
        }
        let s = walker.sample();
        assert!(!s.fallen, "fell at y={:.3}", s.pos[1]);
        assert!(
            s.pos[2] > 1.2,
            "tripod did not walk forward: z={:.3} x={:.3} yaw={:.3} (cycle={:.3} stride={:.3})",
            s.pos[2],
            s.pos[0],
            s.yaw,
            gait.cycle,
            gait.stride
        );
        assert!(s.yaw.abs() < 0.45, "spun while walking: yaw={:.3}", s.yaw);
        assert!(s.pos[1] > 0.55, "sat down: y={:.3}", s.pos[1]);
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

    /// Every leg has to actually sweep when the gait leaves the servo room to
    /// track it. Absolute joint limits on a hinge whose zero is the spawn pose
    /// pinned the tibia, and a setpoint clamped against the measured angle left
    /// the motor too weak to move: both showed up here as legs twitching over a
    /// few degrees instead of stepping.
    #[test]
    fn legs_sweep_when_the_servo_can_keep_up() {
        let frame = Frame::new(6);
        let mut gait = Policy::seeded(Preset::Tripod, frame).gait();
        gait.cycle *= 2.0; // the default clock outruns the default servo
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);

        let mut lo = [[9.0f64; 3]; MAX_LEGS];
        let mut hi = [[-9.0f64; 3]; MAX_LEGS];
        let mut phase = 0.0;
        for k in 0..800 {
            let mut q = [[0.0f64; 3]; MAX_LEGS];
            for i in 0..frame.legs() {
                let foot = foot_in_body(
                    frame, &gait, i, phase, gait.stride, gait.duty, gait.cycle, gait.body_h,
                    gait.step_h, 0.0,
                );
                q[i] = solve_ik(frame, i, foot).q;
            }
            plant.drive(&q, &phys, crate::sim::DT);
            plant.step(crate::sim::DT);
            phase = crate::math::frac(phase + crate::sim::DT / gait.cycle);
            if k > 200 {
                for i in 0..frame.legs() {
                    let m = plant.leg_q(i);
                    for j in 0..3 {
                        lo[i][j] = lo[i][j].min(m[j]);
                        hi[i][j] = hi[i][j].max(m[j]);
                    }
                }
            }
        }
        for i in 0..frame.legs() {
            let sweep = [
                hi[i][0] - lo[i][0],
                hi[i][1] - lo[i][1],
                hi[i][2] - lo[i][2],
            ];
            assert!(sweep[0] > 0.5, "leg {i} coxa barely moved: {sweep:?}");
            assert!(sweep[1] > 0.3, "leg {i} femur barely moved: {sweep:?}");
            assert!(sweep[2] > 0.3, "leg {i} tibia barely moved: {sweep:?}");
        }
    }

    /// Faced with a kerb it could step onto, the planner has to put the foot on
    /// top of it — never inside it.
    ///
    /// Kinematic on purpose: how far the machine gets to walk before it meets the
    /// kerb is a traction question, and a target aimed into the face of a block
    /// is wrong whether or not the robot ever arrives. The open-loop stroke is
    /// checked alongside to show what the terrain term is actually buying.
    #[test]
    fn a_foot_is_placed_on_a_kerb_not_into_it() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let mut terrain = Terrain::new(Course::Flat, 1);
        let kerb = Obstacle {
            x0: -3.0,
            x1: 3.0,
            z0: 1.6,
            z1: 4.6,
            top: 0.35,
            grip: 1.0,
        };
        terrain.push(kerb.x0, kerb.x1, kerb.z0, kerb.z1, kerb.top, kerb.grip);
        terrain.rebuild_buckets();
        // Standing just short of the kerb, so the front legs' stroke crosses it.
        let pos = [0.0, terrain.height(0.0, 0.6) + gait.body_h, 0.6];

        let inside = |p: V3| {
            p[0] > kerb.x0
                && p[0] < kerb.x1
                && p[2] > kerb.z0
                && p[2] < kerb.z1
                && p[1] < kerb.top - 0.02
        };
        let world = |t: V3| {
            let w = crate::math::body_to_world(t, 0.0, 0.0, 0.0);
            [pos[0] + w[0], pos[1] + w[1], pos[2] + w[2]]
        };

        let (mut aware_in, mut blind_in, mut aware_over) = (0usize, 0usize, 0usize);
        for k in 0..240 {
            let phase = k as f64 / 240.0;
            for leg in 0..frame.legs() {
                let aware = world(foot_on_terrain(
                    frame, &gait, leg, phase, gait.stride, gait.duty, gait.cycle, gait.body_h,
                    gait.step_h, 0.0, &terrain, pos, 0.0, 0.0, 0.0,
                ));
                let blind = world(foot_in_body(
                    frame, &gait, leg, phase, gait.stride, gait.duty, gait.cycle, gait.body_h,
                    gait.step_h, 0.0,
                ));
                if inside(aware) {
                    aware_in += 1;
                    if aware_in < 4 {
                        println!("offender leg{leg} phase {phase:.3} target {aware:.3?}");
                    }
                }
                if inside(blind) {
                    blind_in += 1;
                }
                if aware[0] > kerb.x0
                    && aware[0] < kerb.x1
                    && aware[2] > kerb.z0
                    && aware[2] < kerb.z1
                {
                    aware_over += 1;
                }
            }
        }
        assert!(
            aware_over > 0,
            "no target ever reached over the kerb, so nothing was tested"
        );
        assert_eq!(
            aware_in, 0,
            "terrain-aware target still aims inside the kerb {aware_in} times \
             ({aware_over} targets were over it)"
        );
        assert!(
            blind_in > 20,
            "the open-loop stroke was expected to aim into the kerb; got {blind_in}"
        );
    }

    /// Rapier has to agree with `fk_world` about where a leg is once it has
    /// moved, not just at spawn.
    ///
    /// This is the check that was missing. `crate::math::rot_y` is left-handed
    /// about Y and Rapier is right-handed, so the hinges ran mirrored — and
    /// because the angle was read back through the same mirror, the joint
    /// numbers looked perfect while the links were somewhere else entirely.
    /// Comparing angles cannot catch it; comparing link positions can.
    #[test]
    fn rapier_kinematics_match_the_analytic_model() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let q0 = standing_q(frame, &gait);

        for j in 0..3 {
            let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
            let mut cmd = q0;
            for i in 0..frame.legs() {
                cmd[i][j] += 0.30;
            }
            hold(&mut plant, &cmd, &phys, 4.0);

            let (p, yaw, pitch, roll) = plant.chassis_pose();
            for leg in 0..frame.legs() {
                // FK of the angles Rapier reports must land on the links Rapier
                // actually has. Slack covers servo droop under load and the
                // foot sphere, not a mirrored axis — that shows up as decimetres.
                let fk = fk_world(frame, leg, plant.leg_q(leg), p, yaw, pitch, roll);
                let real = plant.leg_joints_world(leg);
                for (k, name) in [(1, "knee"), (2, "ankle"), (3, "foot")] {
                    let d = ((real[k][0] - fk[k][0]).powi(2)
                        + (real[k][1] - fk[k][1]).powi(2)
                        + (real[k][2] - fk[k][2]).powi(2))
                    .sqrt();
                    assert!(
                        d < 0.10,
                        "joint {j} moved: leg {leg} {name} is {d:.3} from where fk_world puts it"
                    );
                }
            }
        }
    }

    #[test]
    fn chassis_and_feet_use_ccd() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
        assert!(plant.bodies[plant.chassis].is_ccd_enabled());
        for i in 0..6 {
            let tibia = plant.legs[i].as_ref().unwrap().tibia;
            assert!(plant.bodies[tibia].is_ccd_enabled(), "leg {i} tibia");
        }
    }

    #[test]
    fn a_tibia_does_not_occupy_a_block() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let mut terrain = Terrain::new(Course::Flat, 1);
        // Right-side stance: a prism the standing tibia would thread if links
        // still belonged to no collision group.
        let wall = Obstacle {
            x0: 0.55,
            x1: 2.20,
            z0: -0.55,
            z1: 0.55,
            top: 0.70,
            grip: 1.0,
        };
        terrain.push(wall.x0, wall.x1, wall.z0, wall.z1, wall.top, wall.grip);
        terrain.rebuild_buckets();
        let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
        let q = standing_q(frame, &gait);
        hold(&mut plant, &q, &phys, 0.8);

        let mut inside = Vec::new();
        for i in 0..6 {
            let j = plant.leg_joints_world(i);
            for (name, p) in [("knee", j[1]), ("ankle", j[2]), ("foot", j[3])] {
                if in_core(p, &wall, 0.12) {
                    inside.push(format!("L{i} {name} {:?}", p));
                }
            }
        }
        assert!(
            inside.is_empty(),
            "link still occupies the block: {}",
            inside.join("; ")
        );
        assert!(
            plant.chassis_y().is_finite() && plant.pitch_abs() < 1.2,
            "plant exploded: y={} pitch={}",
            plant.chassis_y(),
            plant.pitch_abs()
        );
    }

    #[test]
    fn a_four_leg_trot_keeps_stepping() {
        let frame = Frame::new(4);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
        let mut phase = 0.0;
        let ticks = (2.0 / crate::sim::DT) as usize;
        for _ in 0..ticks {
            phase = crate::math::frac(phase + crate::sim::DT / gait.cycle);
            let mut q = [[0.0; 3]; MAX_LEGS];
            for i in 0..4 {
                let target = foot_in_body(
                    frame, &gait, i, phase, gait.stride, gait.duty, gait.cycle, gait.body_h,
                    gait.step_h, 0.0,
                );
                q[i] = solve_ik(frame, i, target).q;
            }
            plant.drive(&q, &phys, crate::sim::DT);
            plant.step(crate::sim::DT);
        }
        assert!(
            plant.chassis_y().is_finite() && plant.pitch_abs().is_finite(),
            "plant reset or exploded: y={} pitch={}",
            plant.chassis_y(),
            plant.pitch_abs()
        );
    }

    #[test]
    fn canvas_pose_comes_from_rapier_bodies() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
        let (pos, yaw, pitch, roll) = plant.chassis_pose();
        assert!((pos[1] - plant.chassis_y()).abs() < 1e-5);
        assert!(yaw.abs() < 0.2 && pitch.abs() < 0.4 && roll.abs() < 0.4);
        let foot = plant.leg_joints_world(0)[3];
        assert!(foot[1] < pos[1], "foot should be below chassis {foot:?} {pos:?}");
    }

    #[test]
    fn a_standing_chassis_is_not_dead() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
        let q = standing_q(frame, &gait);
        hold(&mut plant, &q, &phys, 0.4);
        assert!(
            !plant.chassis_dead(plant.chassis_vel()),
            "standing belly should clear the floor"
        );
    }

    #[test]
    fn chassis_on_the_floor_is_dead() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
        plant.bodies[plant.chassis].set_translation(Vector::new(0.0, 0.10, 0.0), true);
        plant.bodies[plant.chassis].set_linvel(Vector::new(0.0, -3.0, 0.0), true);
        let pre = plant.chassis_vel();
        plant.step(crate::sim::DT);
        assert!(plant.chassis_dead(pre), "belly on the floor should kill");
    }

    fn shove_chassis(plant: &mut ArticulatedPlant, z: f32, vz: f32) {
        let t = plant.bodies[plant.chassis].translation();
        plant.bodies[plant.chassis].set_translation(Vector::new(t.x, t.y, z), true);
        plant.bodies[plant.chassis].set_linvel(Vector::new(0.0, 0.0, vz), true);
    }

    fn wall_course() -> Terrain {
        let mut terrain = Terrain::new(Course::Flat, 1);
        terrain.push(-2.0, 2.0, 1.15, 2.40, 1.80, 1.0);
        terrain.rebuild_buckets();
        terrain
    }

    #[test]
    fn a_fast_chassis_hit_on_a_wall_is_dead() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = wall_course();
        let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
        // Overlap the near face and keep walking-speed into it. Joints would
        // otherwise pin the chassis at spawn and the wall would never be hit.
        shove_chassis(&mut plant, 0.55, 4.0);
        let mut dead = false;
        for _ in 0..12 {
            let pre = plant.chassis_vel();
            plant.step(crate::sim::DT);
            if plant.chassis_dead(pre) {
                dead = true;
                break;
            }
        }
        assert!(dead, "walking-speed chassis hit on a wall should kill");
    }

    fn hinge_len(plant: &ArticulatedPlant, i: usize) -> [f64; 3] {
        let j = plant.leg_joints_world(i);
        let d = |a: V3, b: V3| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        [d(j[0], j[1]), d(j[1], j[2]), d(j[2], j[3])]
    }

    fn drive_terrain(
        plant: &mut ArticulatedPlant,
        frame: Frame,
        gait: &Gait,
        phys: &Physics,
        terrain: &Terrain,
        phase: f64,
    ) {
        let (bp, byaw, bpitch, broll) = plant.chassis_pose();
        let mut q = [[0.0; 3]; MAX_LEGS];
        for i in 0..frame.legs() {
            let target = foot_on_terrain(
                frame, gait, i, phase, gait.stride, gait.duty, gait.cycle, gait.body_h,
                gait.step_h, 0.0, terrain, bp, byaw, bpitch, broll,
            );
            q[i] = solve_ik(frame, i, target).q;
        }
        plant.drive(&q, phys, crate::sim::DT);
        plant.step(crate::sim::DT);
    }

    fn walk_cycle(frame: Frame, gait: &Gait) -> f64 {
        gait.cycle.max(crate::policy::feasible_cycle(
            frame,
            gait,
            gait.stride,
            gait.duty,
            gait.cycle,
            gait.body_h,
            gait.step_h,
            0.0,
            phys_omega(),
        ))
    }

    /// Impulse joints can stretch. The canvas draws these hinges as the legs,
    /// so a centimetre of drift is a wrong-looking tibia and a sagging deck.
    #[test]
    fn hinges_keep_their_kinematic_length() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = Terrain::new(Course::Flat, 1);
        let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
        let q = standing_q(frame, &gait);
        hold(&mut plant, &q, &phys, 5.0);
        let mut worst = 0.0f64;
        for i in 0..6 {
            let l = hinge_len(&plant, i);
            worst = worst
                .max((l[0] - COXA).abs())
                .max((l[1] - FEMUR).abs())
                .max((l[2] - TIBIA).abs());
        }
        assert!(
            worst < 0.04,
            "hinges stretched/shortened by {worst:.3} (coxa/femur/tibia should stay \
             {COXA}/{FEMUR}/{TIBIA}); lens0={:?}",
            hinge_len(&plant, 0)
        );
        assert!(
            plant.chassis_y() > 0.70,
            "standing chassis sat down: y={}",
            plant.chassis_y()
        );
    }

    /// The live dashboard world-locks stance and keeps ride height as a
    /// body-frame command. Converting world ground through a dipped `pos[1]`
    /// used to fold the legs on the first step.
    #[test]
    fn terrain_aware_tripod_does_not_sit_down() {
        let frame = Frame::new(6);
        let phys = Physics::default();
        let (mut walker, terrain, policy, gait) =
            crate::walker::open_loop_walk(frame, Course::Flat, 1, phys);
        let cmd = crate::sim::Cmd {
            fwd: 1.0,
            turn: 0.0,
            cruise: 1.5,
            nav: false,
        };
        let mut min_y = f64::INFINITY;
        let ticks = (6.0 / crate::sim::DT) as usize;
        for k in 0..ticks {
            walker.step(&terrain, &policy, &gait, crate::sim::DT, cmd);
            if k > 50 {
                min_y = min_y.min(walker.plant.chassis_y());
            }
        }
        let s = walker.sample();
        assert!(
            min_y > 0.55,
            "terrain-aware walk sat down: min_y={min_y:.3} end_y={:.3} z={:.3}",
            s.pos[1],
            s.pos[2]
        );
        assert!(s.pos[2] > 0.70, "did not walk: z={:.3}", s.pos[2]);
        assert!(s.pitch.abs() < 0.55, "fell while walking");
        assert!(!s.fallen);
    }

    /// A wall beside the machine, not in front of it. Swing used to be projected
    /// onto the face (`push_xz`) so the tibia spent the whole step kicking it.
    #[test]
    fn walking_beside_a_wall_does_not_put_links_inside_it() {
        let frame = Frame::new(6);
        let mut gait = Policy::seeded(Preset::Tripod, frame).gait();
        gait.cycle = walk_cycle(frame, &gait);
        let phys = Physics::default();
        let mut terrain = Terrain::new(Course::Flat, 1);
        let wall = Obstacle {
            x0: 1.75,
            x1: 3.40,
            z0: 0.80,
            z1: 5.50,
            top: 1.80,
            grip: 1.0,
        };
        terrain.push(wall.x0, wall.x1, wall.z0, wall.z1, wall.top, wall.grip);
        terrain.rebuild_buckets();
        let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
        let mut phase = 0.0;
        let mut inside = 0usize;
        let ticks = (4.0 / crate::sim::DT) as usize;
        for _ in 0..ticks {
            drive_terrain(&mut plant, frame, &gait, &phys, &terrain, phase);
            phase = crate::math::frac(phase + crate::sim::DT / gait.cycle);
            for i in 0..6 {
                let j = plant.leg_joints_world(i);
                for p in j {
                    if in_core(p, &wall, 0.10) {
                        inside += 1;
                    }
                }
            }
        }
        assert_eq!(
            inside, 0,
            "links occupied the side wall {inside} times; y={:.3} z={:.3}",
            plant.chassis_y(),
            plant.chassis_z()
        );
        assert!(
            plant.chassis_y() > 0.50,
            "sat down beside the wall: y={}",
            plant.chassis_y()
        );
    }

    #[test]
    fn a_slow_chassis_touch_on_a_wall_is_not_dead() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::Tripod, frame).gait();
        let phys = Physics::default();
        let terrain = wall_course();
        let mut plant = ArticulatedPlant::standing(frame, &gait, &phys, &terrain);
        shove_chassis(&mut plant, 0.55, 0.25);
        let mut touched = false;
        for _ in 0..12 {
            let pre = plant.chassis_vel();
            plant.step(crate::sim::DT);
            let hit = plant.narrow_phase.contact_pairs_with(plant.chassis_col).any(|p| {
                p.has_any_active_contact()
                    && [p.collider1, p.collider2].iter().any(|c| {
                        *c != plant.chassis_col && plant.colliders[*c].user_data == HIT_SOLID
                    })
            });
            if hit {
                touched = true;
                assert!(
                    !plant.chassis_dead(pre),
                    "a slow scrape should not kill, closing {:?}",
                    pre
                );
                break;
            }
        }
        assert!(touched, "never reached the wall");
    }
}
