//! Nexus 0.5 native-GPU batch backend for joint-level reinforcement learning.
//!
//! Nexus owns one independent physics environment per rollout. Policy and
//! reward code stay on the host for now, but state is read once per tensor for
//! the whole batch; there is no per-environment GPU synchronization.

use futures::executor::block_on;
use khal::backend::{Backend, GpuBackend, WebGpu};
use nexus3d::pipeline::NexusPipeline;
use nexus3d::rapier::dynamics::JointAxis;
use nexus3d::rbd::dynamics::RbdSimParams;
use nexus3d::rbd::queries::GpuIndexedContact;
use nexus3d::state::{NexusCapacities, NexusState};
use std::sync::{Mutex, OnceLock};

use crate::dynamics::Physics;
use crate::math::V3;
use crate::plant::ArticulatedPlant;
use crate::policy::Gait;
use crate::robot::{Frame, MAX_LEGS};
use crate::sim::DT;
use crate::terrain::Terrain;

/// Host snapshot with exactly the state consumed by `joint_rl`.
#[derive(Clone, Copy, Debug)]
pub struct NexusSnapshot {
    pub q: [[f64; 3]; MAX_LEGS],
    pub pos: V3,
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
    pub vel: V3,
    pub angvel: V3,
    pub contacts: [bool; MAX_LEGS],
    pub chassis_contact: bool,
}

impl Default for NexusSnapshot {
    fn default() -> Self {
        Self {
            q: [[0.0; 3]; MAX_LEGS],
            pos: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            vel: [0.0; 3],
            angvel: [0.0; 3],
            contacts: [false; MAX_LEGS],
            chassis_contact: false,
        }
    }
}

#[derive(Clone)]
struct EnvMap {
    n: usize,
    chassis_body: u32,
    chassis_collider: u32,
    feet: [u32; MAX_LEGS],
    motor_links: [[u32; 3]; MAX_LEGS],
    joint_parents: [[u32; 3]; MAX_LEGS],
    joint_children: [[u32; 3]; MAX_LEGS],
    joint_frame1: [[[f32; 4]; 3]; MAX_LEGS],
    joint_frame2: [[[f32; 4]; 3]; MAX_LEGS],
    neutral: [[f64; 3]; MAX_LEGS],
    previous_pos: V3,
    previous_rot: nexus3d::rbd::glamx::Quat,
}

/// A fixed-size set of independent articulated worlds resident on one GPU.
/// Create a new batch for each ARS chunk; construction is intentionally
/// separate from stepping so shader compilation/finalization can be timed.
pub struct NexusPlantBatch {
    backend: GpuBackend,
    state: NexusState,
    envs: Vec<EnvMap>,
}

static NEXUS_BACKEND: OnceLock<Result<GpuBackend, String>> = OnceLock::new();
static NEXUS_PIPELINE: OnceLock<Mutex<NexusPipeline>> = OnceLock::new();

fn shared_backend(device: usize) -> Result<GpuBackend, String> {
    if device != 0 {
        return Err("Nexus WebGPU currently selects the system adapter; --device must be 0".into());
    }
    NEXUS_BACKEND
        .get_or_init(|| {
            let gpu = block_on(WebGpu::default())
                .map_err(|e| format!("could not initialize native WebGPU: {e}"))?;
            Ok(GpuBackend::WebGpu(gpu))
        })
        .clone()
}

impl NexusPlantBatch {
    pub fn new(
        frame: Frame,
        gait: &Gait,
        phys: &Physics,
        terrains: &[Terrain],
        device: usize,
    ) -> Result<Self, String> {
        if terrains.is_empty() {
            return Err("a Nexus batch needs at least one terrain".into());
        }
        let backend = shared_backend(device)?;
        let mut state = NexusState::new(
            NexusCapacities::default()
                .rbd_batches(terrains.len() as u32)
                .rbd_collisions(512),
        );
        let substeps = phys.substeps.max(1);
        state.rbd_steps_per_frame = substeps as u32;

        let mut authored = Vec::with_capacity(terrains.len());
        for (env, terrain) in terrains.iter().enumerate() {
            if env > 0 {
                let added = state.add_environment();
                debug_assert_eq!(added, env);
            }
            let scene =
                ArticulatedPlant::standing(frame, gait, phys, terrain).into_nexus_scene()?;
            let mut params = RbdSimParams::tgs_soft();
            params.dt = DT as f32 / substeps as f32;
            params.num_solver_iterations = phys.solver_iters as u32;
            state.set_rbd_sim_params(env, params);
            let metadata = (
                scene.n,
                scene.chassis,
                scene.chassis_collider_slot,
                scene.foot_collider_slots,
                scene.joint_link_slots,
                scene.joint_parents,
                scene.joint_children,
                scene.joint_frame1,
                scene.joint_frame2,
                scene.neutral,
            );
            *state.rbd_world_mut(env) = scene.world;
            authored.push(metadata);
        }

        block_on(state.finalize(&backend))
            .map_err(|e| format!("could not finalize Nexus batch: {e}"))?;
        let stride = state
            .rbd
            .as_ref()
            .ok_or_else(|| "Nexus finalized without an RBD state".to_string())?
            .num_colliders_per_batch();

        let mut envs = Vec::with_capacity(authored.len());
        for (env, scene) in authored.into_iter().enumerate() {
            let (
                n,
                chassis,
                chassis_collider_slot,
                foot_slots,
                motor_links,
                joint_parent_handles,
                joint_child_handles,
                joint_frame1,
                joint_frame2,
                neutral,
            ) = scene;
            let body_id = |handle: nexus3d::rapier::prelude::RigidBodyHandle| {
                state.rbd2gpu[env]
                    .get(handle.0)
                    .map(|r| r.gpu_id)
                    .ok_or_else(|| {
                        format!("Nexus did not map body {handle:?} in environment {env}")
                    })
            };
            let chassis_body = body_id(chassis)?;
            let mut joint_parents = [[u32::MAX; 3]; MAX_LEGS];
            let mut joint_children = [[u32::MAX; 3]; MAX_LEGS];
            for i in 0..n {
                for j in 0..3 {
                    joint_parents[i][j] = body_id(joint_parent_handles[i][j])?;
                    joint_children[i][j] = body_id(joint_child_handles[i][j])?;
                }
            }
            let pose = state
                .rbd_world(env)
                .bodies
                .get(chassis)
                .ok_or_else(|| format!("Nexus lost chassis in environment {env}"))?
                .position();
            let p = pose.translation;
            let rot = pose.rotation;
            let mut feet = [u32::MAX; MAX_LEGS];
            for i in 0..n {
                feet[i] = env as u32 * stride + foot_slots[i];
            }
            envs.push(EnvMap {
                n,
                chassis_body,
                chassis_collider: env as u32 * stride + chassis_collider_slot,
                feet,
                motor_links,
                joint_parents,
                joint_children,
                joint_frame1,
                joint_frame2,
                neutral,
                previous_pos: [p.x as f64, p.y as f64, p.z as f64],
                previous_rot: rot,
            });
        }

        Ok(Self {
            backend,
            state,
            envs,
        })
    }

    pub fn len(&self) -> usize {
        self.envs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    pub fn neutral(&self, env: usize) -> [[f64; 3]; MAX_LEGS] {
        self.envs[env].neutral
    }

    /// Retarget every environment without rebuilding GPU state or dropping the
    /// motor's accumulated warm-start impulse.
    pub fn drive(
        &mut self,
        commands: &[[[f64; 3]; MAX_LEGS]],
        phys: &Physics,
    ) -> Result<(), String> {
        if commands.len() != self.envs.len() {
            return Err(format!(
                "Nexus received {} commands for {} environments",
                commands.len(),
                self.envs.len()
            ));
        }
        let rbd = self
            .state
            .rbd
            .as_mut()
            .ok_or_else(|| "Nexus RBD state is unavailable".to_string())?;
        let multibodies = rbd.multibodies_mut();
        let axis = JointAxis::AngX as usize;
        for (batch, (map, q_cmd)) in self.envs.iter().zip(commands).enumerate() {
            let mut updates = Vec::with_capacity(map.n * 3);
            for i in 0..map.n {
                for j in 0..3 {
                    let link = map.motor_links[i][j];
                    let mut motor = multibodies
                        .motor(batch as u32, link, axis)
                        .ok_or_else(|| format!("Nexus lost motor {i}:{j} in batch {batch}"))?;
                    motor.target_pos = (q_cmd[i][j] - map.neutral[i][j]) as f32;
                    motor.stiffness = phys.motor_stiff as f32;
                    motor.damping = phys.motor_damp as f32;
                    motor.max_force = phys.motor_max as f32;
                    updates.push((link, axis, motor));
                }
            }
            multibodies
                .set_motors(&self.backend, batch as u32, &updates)
                .map_err(|e| format!("could not upload Nexus motors: {e}"))?;
        }
        Ok(())
    }

    pub fn step(&mut self) -> Result<(), String> {
        let mut pipeline = NEXUS_PIPELINE
            .get_or_init(|| Mutex::new(NexusPipeline::default()))
            .lock()
            .map_err(|_| "shared Nexus pipeline lock was poisoned".to_string())?;
        block_on(pipeline.simulate(&self.backend, &mut self.state, None))
            .map_err(|e| format!("Nexus simulation failed: {e}"))
    }

    /// One bulk read per live tensor. The returned ordering is always the same
    /// as the input terrain ordering, independent of GPU scheduling.
    pub fn snapshots(&mut self) -> Result<Vec<NexusSnapshot>, String> {
        let rbd = self
            .state
            .rbd
            .as_ref()
            .ok_or_else(|| "Nexus RBD state is unavailable".to_string())?;
        let poses: Vec<nexus3d::rbd::math::Pose> =
            block_on(self.backend.slow_read_vec(rbd.body_poses().buffer()))
                .map_err(|e| format!("could not read Nexus body poses: {e}"))?;
        let contact_lens: Vec<u32> =
            block_on(self.backend.slow_read_vec(rbd.contacts_len().buffer()))
                .map_err(|e| format!("could not read Nexus contact counts: {e}"))?;
        let contacts: Vec<GpuIndexedContact> =
            block_on(self.backend.slow_read_vec(rbd.contacts().buffer()))
                .map_err(|e| format!("could not read Nexus contacts: {e}"))?;

        let contact_stride = contacts.len() / self.envs.len().max(1);
        let mut out = Vec::with_capacity(self.envs.len());
        for (batch, map) in self.envs.iter_mut().enumerate() {
            let pose = poses
                .get(map.chassis_body as usize)
                .ok_or_else(|| format!("Nexus chassis pose is missing for batch {batch}"))?;
            let p = pose.translation;
            let pos = [p.x as f64, p.y as f64, p.z as f64];
            let rot = pose.rotation;
            let fwd = rot * nexus3d::rbd::glamx::Vec3::Z;
            let yaw = (-fwd.x).atan2(fwd.z) as f64;
            let pitch = -fwd.y.clamp(-1.0, 1.0).asin() as f64;
            let right = rot * nexus3d::rbd::glamx::Vec3::X;
            let (sy, cy) = yaw.sin_cos();
            let roll = (right.y as f64).atan2(right.x as f64 * cy + right.z as f64 * sy);
            let vel = [
                (pos[0] - map.previous_pos[0]) / DT,
                (pos[1] - map.previous_pos[1]) / DT,
                (pos[2] - map.previous_pos[2]) / DT,
            ];
            let delta = rot * map.previous_rot.conjugate();
            let axis = delta.to_scaled_axis() / DT as f32;
            let angvel = [axis.x as f64, axis.y as f64, axis.z as f64];
            map.previous_pos = pos;
            map.previous_rot = rot;

            let mut q = map.neutral;
            for i in 0..map.n {
                for j in 0..3 {
                    let parent = poses
                        .get(map.joint_parents[i][j] as usize)
                        .ok_or_else(|| format!("Nexus joint parent {i}:{j} is missing"))?
                        .rotation;
                    let child = poses
                        .get(map.joint_children[i][j] as usize)
                        .ok_or_else(|| format!("Nexus joint child {i}:{j} is missing"))?
                        .rotation;
                    let a = map.joint_frame1[i][j];
                    let b = map.joint_frame2[i][j];
                    let joint1 =
                        parent * nexus3d::rbd::glamx::Quat::from_xyzw(a[0], a[1], a[2], a[3]);
                    let joint2 =
                        child * nexus3d::rbd::glamx::Quat::from_xyzw(b[0], b[1], b[2], b[3]);
                    let error = joint1.conjugate() * joint2;
                    let signed = if joint1.dot(joint2) < 0.0 {
                        -2.0 * error.x.clamp(-1.0, 1.0).asin()
                    } else {
                        2.0 * error.x.clamp(-1.0, 1.0).asin()
                    };
                    q[i][j] += signed as f64;
                }
            }

            let mut foot_contact = [false; MAX_LEGS];
            let mut chassis_contact = false;
            let count = contact_lens.get(batch).copied().unwrap_or(0) as usize;
            let start = batch * contact_stride;
            for manifold in contacts.iter().skip(start).take(count.min(contact_stride)) {
                if manifold.contact.len == 0 {
                    continue;
                }
                let a = manifold.colliders.x;
                let b = manifold.colliders.y;
                chassis_contact |= a == map.chassis_collider || b == map.chassis_collider;
                for (i, foot) in map.feet.iter().take(map.n).enumerate() {
                    foot_contact[i] |= a == *foot || b == *foot;
                }
            }

            out.push(NexusSnapshot {
                q,
                pos,
                yaw,
                pitch,
                roll,
                vel,
                angvel,
                contacts: foot_contact,
                chassis_contact,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Policy, Preset};
    use crate::terrain::Course;

    #[test]
    #[ignore = "requires a native GPU adapter"]
    fn two_environments_step_and_report_finite_state() {
        let frame = Frame::new(6);
        let gait = Policy::seeded(Preset::default_for(frame), frame).gait();
        let phys = Physics::default();
        let terrains = vec![Terrain::new(Course::Flat, 7), Terrain::new(Course::Flat, 7)];
        let mut batch = NexusPlantBatch::new(frame, &gait, &phys, &terrains, 0)
            .expect("initialize Nexus batch");
        let mut commands = (0..batch.len())
            .map(|env| batch.neutral(env))
            .collect::<Vec<_>>();
        commands[0][0][0] += 0.10;
        for _ in 0..5 {
            batch.drive(&commands, &phys).expect("upload motors");
            batch.step().expect("step Nexus");
        }
        let states = batch.snapshots().expect("read Nexus state");
        assert_eq!(states.len(), 2);
        for state in &states {
            assert!(state.pos.into_iter().all(f64::is_finite));
            assert!(state.q.into_iter().flatten().all(f64::is_finite));
            assert!((0.2..2.0).contains(&state.pos[1]));
        }
        assert!(
            (states[0].q[0][0] - states[1].q[0][0]).abs() > 1.0e-5,
            "distinct motor targets did not produce distinct joint state"
        );
    }
}
