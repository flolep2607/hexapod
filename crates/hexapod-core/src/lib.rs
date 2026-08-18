//! Legged locomotion: a centroidal rigid-body simulator with Coulomb contact,
//! servo dynamics and leg inertia, plus a policy-search trainer and a hardware
//! sizer. The frame is parametric in leg count — four legs to ten.
//!
//! With the `rapier` feature the live dashboard also runs an articulated plant:
//! one chassis, three revolute joints per leg, and Rapier's contact solver.
//! Gait, analytic IK and the servo torque cap are still this crate; Rapier is
//! the engine they drive. ARS keeps the centroidal step so training stays cheap
//! enough to run in the page.
//!
//! ```no_run
//! use hexapod_core::*;
//! let terrain = Terrain::new(Course::Mixed, 7);
//! let frame = Frame::new(6);
//! let phys = Physics::default();
//! let policy = Policy::seeded(Preset::default_for(frame), frame);
//! let mut trainer = Trainer::new(policy, ArsConfig::default(), phys, 1);
//! trainer.record_baseline(&terrain);
//! for _ in 0..200 {
//!     trainer.iterate(&terrain);
//! }
//! println!("{:.1} -> {:.1}", trainer.baseline_reward, trainer.best_reward);
//! ```

pub mod ars;
pub mod dynamics;
pub mod hardware;
pub mod math;
pub mod policy;
pub mod power;
pub mod robot;
pub mod sim;
pub mod terrain;
#[cfg(feature = "rapier")]
pub mod plant;
#[cfg(feature = "rapier")]
pub mod walker;
#[cfg(feature = "rapier")]
pub mod oneleg;

pub use ars::{ArsConfig, Trainer};
pub use dynamics::{Actuator, LegMass, Physics};
pub use hardware::{shortlist, Build, Servo, TorqueMeter, SERVOS};
pub use power::{solve, Kind, Part, Sizing, Solution, TorqueTrace, PARTS};
pub use policy::{n_act, n_obs, n_theta, Gait, Policy, Preset};
pub use robot::{Frame, MAX_LEGS, MIN_LEGS};
pub use sim::{
    evaluate, rollout, Cmd, Rollout, Sim, CRUISE_MAX, CRUISE_MIN, DT, JUMP_CRUISE_DEFAULT,
    JUMP_CRUISE_MAX, JUMP_CRUISE_MIN,
};
pub use terrain::{Course, Terrain};
#[cfg(feature = "rapier")]
pub use walker::{ArticulatedWalker, WalkSample};
#[cfg(feature = "rapier")]
pub use oneleg::{OneLegDrill, OneLegSample};
