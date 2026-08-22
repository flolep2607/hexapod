//! Legged locomotion on an articulated plant: one chassis, three revolute
//! joints per leg, Rapier's contact solver, a servo torque-speed line and leg
//! inertia. The frame is parametric in leg count — four legs to ten.
//!
//! There are no hand-written gaits. The only controller is the motor-level
//! policy in [`joint_rl`], which commands joint offsets from the standing pose
//! at 50 Hz and is trained by SAC in the `hexapod-sac` crate; analytic IK
//! survives only to place the feet for that standing pose. [`hardware`] and
//! [`power`] size real servos against torque measured from that policy.
//!
//! ```no_run
//! # #[cfg(feature = "rapier")] {
//! use hexapod_core::joint_rl::{JointEnv, Stage};
//! use hexapod_core::{Course, Frame, Physics, Terrain};
//! let terrain = Terrain::new(Course::Mixed, 7);
//! let mut env = JointEnv::new(Frame::new(6), &Physics::default(), terrain, Stage::WalkFlat);
//! let action = vec![0.0; hexapod_core::joint_rl::n_act(Frame::new(6))];
//! env.step(&action).expect("a standing action is always valid");
//! println!("{:.3}", env.summary().score);
//! # }
//! ```

pub mod aba;
pub mod dynamics;
pub mod hardware;
pub mod math;
pub mod power;
pub mod robot;
pub mod terrain;
#[cfg(feature = "rapier")]
pub mod plant;
// The joint-level policy drives the articulated plant's motors directly.
#[cfg(feature = "rapier")]
pub mod joint_rl;
#[cfg(feature = "nexus-gpu")]
pub mod nexus_plant;

pub use dynamics::{Actuator, LegMass, Physics, DT};
pub use hardware::{shortlist, Build, Servo, TorqueMeter, SERVOS};
pub use power::{solve, Kind, Part, Sizing, Solution, TorqueTrace, PARTS};
pub use robot::{Frame, MAX_LEGS, MIN_LEGS};
pub use terrain::{Course, Terrain};
