//! Articulated-body dynamics for this machine and no other.
//!
//! Reduced coordinates: eighteen joint angles and a floating base, not nineteen
//! bodies and ninety constraint rows. A hinge is a coordinate here, so it cannot
//! be violated — the property the impulse plant spends eight solver iterations
//! and four PGS passes approximating to `|cos| = 0.9913`.
//!
//! # What this is not
//!
//! There is no integrator, no contact, no friction and no collision. This file
//! answers one question — given positions, velocities and torques, what are the
//! accelerations — and it is checked against conservation laws rather than
//! against the Rapier plant. That is deliberate: a correct reduced-coordinate
//! solution *should not* reproduce a maximal-coordinate trajectory, so a
//! mismatch with Rapier would carry no information. The tests at the bottom
//! need no reference implementation.
//!
//! # Convention
//!
//! Featherstone's spatial algebra, body-fixed frames, angular part first:
//! motion is `(w, v)` and force is `(n, f)`. A transform is carried as a
//! rotation and a translation rather than a materialised `6x6`, and the spatial
//! inertias are explicit `6x6` matrices. That is slower than the unrolled scalar
//! form a GPU port wants and much easier to read, which is the right trade for
//! the slice whose job is to be provably right.
//!
//! Units are the plant's: lengths in simulator units, masses in kilograms,
//! gravity 9.81. The identities below hold in any consistent set, but keeping
//! the plant's means the inertias can be lifted across unchanged.

use crate::dynamics::{Physics, G};
use crate::math::V3;
use crate::robot::{Frame, COXA, FEMUR, LINK_R, MAX_LEGS, TIBIA};

/// Joints on the largest frame: three per leg.
pub const MAX_JOINTS: usize = 3 * MAX_LEGS;

/// Spatial vector. Angular first: `(w, v)` for motion, `(n, f)` for force.
pub type Spatial = [f64; 6];

/// Spatial `6x6`: inertia, or a materialised transform.
pub type Spatial6 = [[f64; 6]; 6];

/// A `3x3`, for rotations and rotational inertias.
pub type M3 = [[f64; 3]; 3];

// ---------------------------------------------------------------- 3x3 helpers

fn skew(v: V3) -> M3 {
    [
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ]
}

fn m3_mul(a: &M3, b: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn m3_mul_v(a: &M3, v: V3) -> V3 {
    [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]
}

fn m3_scale(a: &M3, s: f64) -> M3 {
    let mut out = *a;
    for row in out.iter_mut() {
        for value in row.iter_mut() {
            *value *= s;
        }
    }
    out
}

fn m3_add(a: &M3, b: &M3) -> M3 {
    let mut out = *a;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] += b[i][j];
        }
    }
    out
}

/// Rotation about the y axis, which is the coxa's axis and the leg mounting.
fn rot_y(angle: f64) -> M3 {
    let (s, c) = angle.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// Rotation about the z axis, which is the femur's and the tibia's.
fn rot_z(angle: f64) -> M3 {
    let (s, c) = angle.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

// ------------------------------------------------------------ spatial algebra

/// One frame relative to another: rotation parent-to-child, and the child's
/// origin expressed in the parent.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub rot: M3,
    pub translation: V3,
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0, 0.0, 0.0],
    };

    /// Compose: `self` is parent-to-mid, `other` is mid-to-child.
    fn then(&self, other: &Transform) -> Transform {
        // A point `p` in the parent goes to `R_o (R_s (p - t_s) - t_o)`, so the
        // composed translation is expressed back in the parent's frame.
        let inv_rot_s = transpose(&self.rot);
        Transform {
            rot: m3_mul(&other.rot, &self.rot),
            translation: add3(self.translation, m3_mul_v(&inv_rot_s, other.translation)),
        }
    }

    /// Move a motion vector from the parent frame into this frame.
    fn motion(&self, m: &Spatial) -> Spatial {
        let w = [m[0], m[1], m[2]];
        let v = [m[3], m[4], m[5]];
        // v_child = R (v_parent - t x w_parent)
        let shifted = sub3(v, cross(self.translation, w));
        let wc = m3_mul_v(&self.rot, w);
        let vc = m3_mul_v(&self.rot, shifted);
        [wc[0], wc[1], wc[2], vc[0], vc[1], vc[2]]
    }

    /// Move a force vector from this frame back into the parent frame.
    ///
    /// The dual of [`Transform::motion`], which is what makes the inbound pass
    /// the transpose of the outbound one.
    fn force_to_parent(&self, f: &Spatial) -> Spatial {
        let rt = transpose(&self.rot);
        let n = m3_mul_v(&rt, [f[0], f[1], f[2]]);
        let force = m3_mul_v(&rt, [f[3], f[4], f[5]]);
        let moment = add3(n, cross(self.translation, force));
        [
            moment[0], moment[1], moment[2], force[0], force[1], force[2],
        ]
    }

    /// Move a spatial inertia from this frame back into the parent frame.
    fn inertia_to_parent(&self, i: &Spatial6) -> Spatial6 {
        // `X* I X`, built column by column so the transpose is never written
        // out: push each parent basis motion through, apply I, pull back.
        let mut out = [[0.0; 6]; 6];
        for col in 0..6 {
            let mut basis = [0.0; 6];
            basis[col] = 1.0;
            let child = self.motion(&basis);
            let force = mat_vec(i, &child);
            let parent = self.force_to_parent(&force);
            for row in 0..6 {
                out[row][col] = parent[row];
            }
        }
        out
    }
}

fn transpose(a: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[j][i];
        }
    }
    out
}

fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn mat_vec(m: &Spatial6, v: &Spatial) -> Spatial {
    let mut out = [0.0; 6];
    for i in 0..6 {
        out[i] = (0..6).map(|j| m[i][j] * v[j]).sum();
    }
    out
}

fn dot6(a: &Spatial, b: &Spatial) -> f64 {
    (0..6).map(|i| a[i] * b[i]).sum()
}

/// Spatial cross product for motion vectors: `v x m`.
fn cross_motion(v: &Spatial, m: &Spatial) -> Spatial {
    let (wv, lv) = ([v[0], v[1], v[2]], [v[3], v[4], v[5]]);
    let (wm, lm) = ([m[0], m[1], m[2]], [m[3], m[4], m[5]]);
    let w = cross(wv, wm);
    let l = add3(cross(wv, lm), cross(lv, wm));
    [w[0], w[1], w[2], l[0], l[1], l[2]]
}

/// Spatial cross product for force vectors: `v x* f`.
fn cross_force(v: &Spatial, f: &Spatial) -> Spatial {
    let (wv, lv) = ([v[0], v[1], v[2]], [v[3], v[4], v[5]]);
    let (nf, ff) = ([f[0], f[1], f[2]], [f[3], f[4], f[5]]);
    let n = add3(cross(wv, nf), cross(lv, ff));
    let force = cross(wv, ff);
    [n[0], n[1], n[2], force[0], force[1], force[2]]
}

/// Spatial inertia of a body: mass, centre of mass in the body frame, and the
/// rotational inertia about that centre.
fn spatial_inertia(mass: f64, com: V3, about_com: &M3) -> Spatial6 {
    let cx = skew(com);
    // `Ibar = Ic + m cx cx^T`, and `cx^T = -cx`.
    let cxcxt = m3_scale(&m3_mul(&cx, &transpose(&cx)), mass);
    let ibar = m3_add(about_com, &cxcxt);
    let mcx = m3_scale(&cx, mass);
    let mut out = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = ibar[i][j];
            out[i][j + 3] = mcx[i][j];
            out[i + 3][j] = -mcx[i][j];
        }
        out[i + 3][i + 3] = mass;
    }
    out
}

/// Gauss-Jordan with partial pivoting. Six by six, once per robot per step.
fn invert6(m: &Spatial6) -> Option<Spatial6> {
    let mut a = *m;
    let mut inv = [[0.0; 6]; 6];
    for i in 0..6 {
        inv[i][i] = 1.0;
    }
    for col in 0..6 {
        let mut pivot = col;
        for row in (col + 1)..6 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1.0e-12 {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let d = a[col][col];
        for k in 0..6 {
            a[col][k] /= d;
            inv[col][k] /= d;
        }
        for row in 0..6 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..6 {
                a[row][k] -= factor * a[col][k];
                inv[row][k] -= factor * inv[col][k];
            }
        }
    }
    Some(inv)
}

// -------------------------------------------------------------------- the tree

/// Which link a joint drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Link {
    Coxa,
    Femur,
    Tibia,
}

/// The machine, as the dynamics needs it.
///
/// Built once from the frame and the mass model, then held. Body 0 is the
/// chassis; joint `j` drives body `j + 1`, and `parent[j]` is the body it hangs
/// from — `0` for a coxa, the link above otherwise. Depth is three and the six
/// legs never touch each other, so the only place the branches meet is body 0,
/// which is exactly the `6x6` the inbound pass assembles.
#[derive(Clone, Debug)]
pub struct Model {
    pub frame: Frame,
    pub joints: usize,
    /// Body each joint's parent is, indexed by joint.
    pub parent: [usize; MAX_JOINTS],
    /// Joint axis in the driven body's own frame. Revolute, so the motion
    /// subspace is one column and this is its angular part.
    pub axis: [V3; MAX_JOINTS],
    /// Fixed part of the parent-to-child transform, before the joint angle.
    pub tree: [Transform; MAX_JOINTS],
    /// Spatial inertia of each body, its own frame. Index 0 is the chassis.
    pub inertia: [Spatial6; MAX_JOINTS + 1],
    /// Gravity as a linear acceleration, world frame.
    pub gravity: V3,
}

impl Model {
    /// Build from the frame and the physics the plant is running.
    ///
    /// Every number is lifted from the existing model rather than re-derived:
    /// link lengths from `robot`, link masses from [`Physics::leg`], and the
    /// chassis mass is what is left of `mass_kg` once the legs are taken out —
    /// the same split [`crate::dynamics::robot_com`] makes.
    ///
    /// The coxa link is massless, which is that model's own assumption: its
    /// servo is bolted to the chassis and does not swing, so its mass is already
    /// inside the chassis figure. A zero-inertia link is fine here because it
    /// has children — the articulated inertia at its joint comes from below —
    /// but it is a modelling weakness rather than a physical fact, and the
    /// `articulated_inertia_stays_positive` test is what would catch it turning
    /// into a singularity.
    pub fn new(frame: Frame, phys: &Physics) -> Model {
        let legs = frame.legs();
        let femur_kg = phys.leg.femur_kg;
        let tibia_kg = phys.leg.tibia_kg;
        let chassis_kg = (phys.mass_kg - legs as f64 * (femur_kg + tibia_kg)).max(0.15);

        let mut model = Model {
            frame,
            joints: 3 * legs,
            parent: [0; MAX_JOINTS],
            axis: [[0.0; 3]; MAX_JOINTS],
            tree: [Transform::IDENTITY; MAX_JOINTS],
            inertia: [[[0.0; 6]; 6]; MAX_JOINTS + 1],
            gravity: [0.0, -G, 0.0],
        };

        // Chassis: a solid cylinder about the vertical, centred on the origin.
        let r = frame.body_r();
        let h = crate::robot::BODY_H;
        let flat = chassis_kg * (3.0 * r * r + h * h) / 12.0;
        let upright = 0.5 * chassis_kg * r * r;
        model.inertia[0] = spatial_inertia(
            chassis_kg,
            [0.0, 0.0, 0.0],
            &[[flat, 0.0, 0.0], [0.0, upright, 0.0], [0.0, 0.0, flat]],
        );

        for leg in 0..legs {
            let (coxa, femur, tibia) = (3 * leg, 3 * leg + 1, 3 * leg + 2);

            // Coxa turns about the vertical at the hip, with the leg's own yaw
            // folded into the fixed part so the child frame has x pointing out.
            model.parent[coxa] = 0;
            model.axis[coxa] = [0.0, 1.0, 0.0];
            model.tree[coxa] = Transform {
                rot: rot_y(-frame.yaw(leg)),
                translation: frame.hip(leg),
            };
            model.inertia[coxa + 1] = spatial_inertia(0.0, [COXA * 0.5, 0.0, 0.0], &[[0.0; 3]; 3]);

            // Femur pitches about z, a rod along the child's x.
            model.parent[femur] = coxa + 1;
            model.axis[femur] = [0.0, 0.0, 1.0];
            model.tree[femur] = Transform {
                rot: Transform::IDENTITY.rot,
                translation: [COXA, 0.0, 0.0],
            };
            model.inertia[femur + 1] = rod(femur_kg, FEMUR);

            // Tibia, the same joint one link out.
            model.parent[tibia] = femur + 1;
            model.axis[tibia] = [0.0, 0.0, 1.0];
            model.tree[tibia] = Transform {
                rot: Transform::IDENTITY.rot,
                translation: [FEMUR, 0.0, 0.0],
            };
            model.inertia[tibia + 1] = rod(tibia_kg, TIBIA);
        }
        model
    }

    /// Transform from a joint's parent into the driven body, at angle `q`.
    fn joint_transform(&self, joint: usize, q: f64) -> Transform {
        let axis = self.axis[joint];
        let rot = if axis[1] != 0.0 { rot_y(q) } else { rot_z(q) };
        self.tree[joint].then(&Transform {
            rot,
            translation: [0.0, 0.0, 0.0],
        })
    }

    /// Motion subspace of a revolute joint: angular part only.
    fn subspace(&self, joint: usize) -> Spatial {
        let a = self.axis[joint];
        [a[0], a[1], a[2], 0.0, 0.0, 0.0]
    }

    /// Gravitational wrench on a body, in that body's frame.
    ///
    /// `I * [0; g]` is exactly the weight acting at the centre of mass: the
    /// linear part comes out `m g` and the angular part `c x m g`.
    fn weight(&self, body: usize, rot_world_to_body: &M3) -> Spatial {
        let g = m3_mul_v(rot_world_to_body, self.gravity);
        mat_vec(&self.inertia[body], &[0.0, 0.0, 0.0, g[0], g[1], g[2]])
    }
}

/// A uniform capsule along the child frame's x axis, hinged at the origin.
fn rod(mass: f64, length: f64) -> Spatial6 {
    let along = 0.5 * mass * LINK_R * LINK_R;
    let across = mass * (3.0 * LINK_R * LINK_R + length * length) / 12.0;
    spatial_inertia(
        mass,
        [length * 0.5, 0.0, 0.0],
        &[[along, 0.0, 0.0], [0.0, across, 0.0], [0.0, 0.0, across]],
    )
}

/// Where the machine is and how it is moving, in reduced coordinates.
///
/// The base is carried as its spatial velocity in its own frame and its
/// orientation relative to the world, which is all the dynamics needs. Position
/// is the integrator's problem, and there is no integrator here yet.
#[derive(Clone, Copy, Debug)]
pub struct State {
    /// Joint angles.
    pub q: [f64; MAX_JOINTS],
    /// Joint rates.
    pub qd: [f64; MAX_JOINTS],
    /// Base spatial velocity, base frame, angular first.
    pub base_v: Spatial,
    /// World-to-base rotation, for gravity.
    pub base_rot: M3,
}

impl Default for State {
    fn default() -> Self {
        State {
            q: [0.0; MAX_JOINTS],
            qd: [0.0; MAX_JOINTS],
            base_v: [0.0; 6],
            base_rot: Transform::IDENTITY.rot,
        }
    }
}

/// Wrenches applied from outside, each in its own body's frame. Index 0 is the
/// base. Contact will land here; nothing does yet.
#[derive(Clone, Copy, Debug)]
pub struct External {
    pub wrench: [Spatial; MAX_JOINTS + 1],
}

impl Default for External {
    fn default() -> Self {
        External {
            wrench: [[0.0; 6]; MAX_JOINTS + 1],
        }
    }
}

/// Per-body kinematics, shared by both algorithms.
struct Kinematics {
    /// Parent-to-body transform including the joint angle, indexed by joint.
    x: [Transform; MAX_JOINTS],
    /// Body-frame spatial velocity, indexed by body.
    v: [Spatial; MAX_JOINTS + 1],
    /// World-to-body rotation, indexed by body.
    rot: [M3; MAX_JOINTS + 1],
    /// Velocity-product acceleration, indexed by joint.
    c: [Spatial; MAX_JOINTS],
}

fn kinematics(model: &Model, state: &State) -> Kinematics {
    let mut k = Kinematics {
        x: [Transform::IDENTITY; MAX_JOINTS],
        v: [[0.0; 6]; MAX_JOINTS + 1],
        rot: [Transform::IDENTITY.rot; MAX_JOINTS + 1],
        c: [[0.0; 6]; MAX_JOINTS],
    };
    k.v[0] = state.base_v;
    k.rot[0] = state.base_rot;
    for joint in 0..model.joints {
        let body = joint + 1;
        let parent = model.parent[joint];
        let x = model.joint_transform(joint, state.q[joint]);
        let s = model.subspace(joint);
        let mut v = x.motion(&k.v[parent]);
        for i in 0..6 {
            v[i] += s[i] * state.qd[joint];
        }
        let mut vj = [0.0; 6];
        for i in 0..6 {
            vj[i] = s[i] * state.qd[joint];
        }
        k.c[joint] = cross_motion(&v, &vj);
        k.x[joint] = x;
        k.v[body] = v;
        k.rot[body] = m3_mul(&x.rot, &k.rot[parent]);
    }
    k
}

/// What the joints must be driven with to produce a given acceleration.
///
/// Recursive Newton-Euler. `base_a` is the base's spatial acceleration in its
/// own frame; the returned wrench is what the world must apply to the base to
/// make that acceleration happen, which is zero for a machine in free flight.
pub fn rnea(
    model: &Model,
    state: &State,
    qdd: &[f64; MAX_JOINTS],
    base_a: &Spatial,
    external: &External,
) -> ([f64; MAX_JOINTS], Spatial) {
    let k = kinematics(model, state);
    let mut a = [[0.0; 6]; MAX_JOINTS + 1];
    let mut f = [[0.0; 6]; MAX_JOINTS + 1];
    a[0] = *base_a;

    // Outward: accelerations, then the wrench each body needs on its own.
    for body in 0..=model.joints {
        if body > 0 {
            let joint = body - 1;
            let s = model.subspace(joint);
            let mut acc = k.x[joint].motion(&a[model.parent[joint]]);
            for i in 0..6 {
                acc[i] += k.c[joint][i] + s[i] * qdd[joint];
            }
            a[body] = acc;
        }
        let inertial = mat_vec(&model.inertia[body], &a[body]);
        let momentum = mat_vec(&model.inertia[body], &k.v[body]);
        let gyroscopic = cross_force(&k.v[body], &momentum);
        let weight = model.weight(body, &k.rot[body]);
        for i in 0..6 {
            f[body][i] = inertial[i] + gyroscopic[i] - weight[i] - external.wrench[body][i];
        }
    }

    // Inward: a joint carries everything distal to it.
    let mut tau = [0.0; MAX_JOINTS];
    for joint in (0..model.joints).rev() {
        let body = joint + 1;
        tau[joint] = dot6(&model.subspace(joint), &f[body]);
        let carried = k.x[joint].force_to_parent(&f[body]);
        let parent = model.parent[joint];
        for i in 0..6 {
            f[parent][i] += carried[i];
        }
    }
    (tau, f[0])
}

/// What the machine does when the joints are driven that way.
///
/// The articulated-body algorithm. Three passes: outward for velocities,
/// inward accumulating an articulated inertia and bias wrench per body, then
/// outward again for accelerations. The base's `6x6` is assembled by the six
/// legs and solved once — the one point in the robot where the branches meet,
/// and the one place a wide kernel has to narrow.
pub fn aba(
    model: &Model,
    state: &State,
    tau: &[f64; MAX_JOINTS],
    external: &External,
) -> (Spatial, [f64; MAX_JOINTS]) {
    let k = kinematics(model, state);
    let mut ia = model.inertia;
    let mut pa = [[0.0; 6]; MAX_JOINTS + 1];
    for body in 0..=model.joints {
        let momentum = mat_vec(&model.inertia[body], &k.v[body]);
        let gyroscopic = cross_force(&k.v[body], &momentum);
        let weight = model.weight(body, &k.rot[body]);
        for i in 0..6 {
            pa[body][i] = gyroscopic[i] - weight[i] - external.wrench[body][i];
        }
    }

    // Inward.
    let mut u = [[0.0; 6]; MAX_JOINTS];
    let mut d = [0.0; MAX_JOINTS];
    let mut w = [0.0; MAX_JOINTS];
    for joint in (0..model.joints).rev() {
        let body = joint + 1;
        let s = model.subspace(joint);
        u[joint] = mat_vec(&ia[body], &s);
        d[joint] = dot6(&s, &u[joint]);
        debug_assert!(
            d[joint] > 1.0e-12,
            "joint {joint} has no articulated inertia about its own axis; a \
             massless link with no massive children is unsolvable"
        );
        w[joint] = tau[joint] - dot6(&s, &pa[body]);

        let mut ia_child = ia[body];
        for r in 0..6 {
            for c in 0..6 {
                ia_child[r][c] -= u[joint][r] * u[joint][c] / d[joint];
            }
        }
        let mut pa_child = pa[body];
        let bias = mat_vec(&ia_child, &k.c[joint]);
        for i in 0..6 {
            pa_child[i] += bias[i] + u[joint][i] * w[joint] / d[joint];
        }

        let parent = model.parent[joint];
        let lifted = k.x[joint].inertia_to_parent(&ia_child);
        let carried = k.x[joint].force_to_parent(&pa_child);
        for r in 0..6 {
            for c in 0..6 {
                ia[parent][r][c] += lifted[r][c];
            }
            pa[parent][r] += carried[r];
        }
    }

    // The base: free-floating, so whatever wrench the world applies is already
    // in `pa[0]` and the acceleration follows from one inversion.
    let base_a = match invert6(&ia[0]) {
        Some(inv) => {
            let mut negated = pa[0];
            for value in negated.iter_mut() {
                *value = -*value;
            }
            mat_vec(&inv, &negated)
        }
        None => [0.0; 6],
    };

    // Outward.
    let mut a = [[0.0; 6]; MAX_JOINTS + 1];
    a[0] = base_a;
    let mut qdd = [0.0; MAX_JOINTS];
    for joint in 0..model.joints {
        let body = joint + 1;
        let s = model.subspace(joint);
        let mut carried = k.x[joint].motion(&a[model.parent[joint]]);
        for i in 0..6 {
            carried[i] += k.c[joint][i];
        }
        qdd[joint] = (w[joint] - dot6(&u[joint], &carried)) / d[joint];
        let mut acc = carried;
        for i in 0..6 {
            acc[i] += s[i] * qdd[joint];
        }
        a[body] = acc;
    }
    (base_a, qdd)
}

/// Kinetic energy, for the conservation checks.
pub fn kinetic_energy(model: &Model, state: &State) -> f64 {
    let k = kinematics(model, state);
    (0..=model.joints)
        .map(|body| 0.5 * dot6(&k.v[body], &mat_vec(&model.inertia[body], &k.v[body])))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rng;

    fn model() -> Model {
        Model::new(Frame::new(6), &Physics::default())
    }

    /// A pose and motion that exercises every term: off-axis, spinning, and
    /// nothing symmetric enough to hide a sign error.
    fn awkward(seed: u64) -> (Model, State, [f64; MAX_JOINTS]) {
        let m = model();
        let mut rng = Rng::new(seed);
        let mut state = State::default();
        let mut qdd = [0.0; MAX_JOINTS];
        for joint in 0..m.joints {
            state.q[joint] = rng.normal() * 0.4;
            state.qd[joint] = rng.normal() * 0.8;
            qdd[joint] = rng.normal() * 1.5;
        }
        state.base_v = [
            rng.normal() * 0.3,
            rng.normal() * 0.3,
            rng.normal() * 0.3,
            rng.normal() * 0.5,
            rng.normal() * 0.5,
            rng.normal() * 0.5,
        ];
        // A base orientation that is not the identity, so gravity has to be
        // rotated properly rather than accidentally.
        state.base_rot = m3_mul(&rot_z(0.21), &rot_y(-0.37));
        (m, state, qdd)
    }

    /// The one test that needs no reference implementation: inverse dynamics
    /// and forward dynamics have to invert each other. Anything that is wrong
    /// in only one of them shows up here, and a sign convention that is wrong
    /// in both still fails the conservation tests below.
    #[test]
    fn forward_and_inverse_dynamics_invert_each_other() {
        for seed in [1u64, 7, 99] {
            let (m, state, qdd) = awkward(seed);
            let mut external = External::default();
            let mut rng = Rng::new(seed ^ 0xBEEF);
            for body in 0..=m.joints {
                for i in 0..6 {
                    external.wrench[body][i] = rng.normal() * 0.2;
                }
            }
            let base_a = [
                rng.normal() * 0.4,
                rng.normal() * 0.4,
                rng.normal() * 0.4,
                rng.normal() * 0.6,
                rng.normal() * 0.6,
                rng.normal() * 0.6,
            ];

            // What holds that acceleration up, including the wrench the world
            // has to put on the base.
            let (tau, base_wrench) = rnea(&m, &state, &qdd, &base_a, &external);

            // Hand the same torques back, with that base wrench as an external
            // load, and the accelerations have to come out again.
            let mut with_base = external;
            for i in 0..6 {
                with_base.wrench[0][i] += base_wrench[i];
            }
            let (got_base, got_qdd) = aba(&m, &state, &tau, &with_base);

            for i in 0..6 {
                assert!(
                    (got_base[i] - base_a[i]).abs() < 1.0e-9,
                    "seed {seed}: base acceleration {i} came back {} against {}",
                    got_base[i],
                    base_a[i]
                );
            }
            for joint in 0..m.joints {
                assert!(
                    (got_qdd[joint] - qdd[joint]).abs() < 1.0e-9,
                    "seed {seed}: joint {joint} came back {} against {}",
                    got_qdd[joint],
                    qdd[joint]
                );
            }
        }
    }

    /// Power in equals the rate the kinetic energy grows. Exact to roundoff,
    /// no integration involved, and it fails for a sign convention that is
    /// consistently wrong — which is the failure invertibility cannot see.
    #[test]
    fn power_in_matches_the_rate_energy_grows() {
        for seed in [3u64, 41] {
            let (m, state, _) = awkward(seed);
            let mut rng = Rng::new(seed ^ 0x5151);
            let mut tau = [0.0; MAX_JOINTS];
            for joint in 0..m.joints {
                tau[joint] = rng.normal() * 2.0;
            }
            let external = External::default();
            let (base_a, qdd) = aba(&m, &state, &tau, &external);

            // d/dt of the kinetic energy, from the accelerations.
            let k = kinematics(&m, &state);
            let mut a = [[0.0; 6]; MAX_JOINTS + 1];
            a[0] = base_a;
            let mut rate = 0.0;
            for body in 0..=m.joints {
                if body > 0 {
                    let joint = body - 1;
                    let s = m.subspace(joint);
                    let mut acc = k.x[joint].motion(&a[m.parent[joint]]);
                    for i in 0..6 {
                        acc[i] += k.c[joint][i] + s[i] * qdd[joint];
                    }
                    a[body] = acc;
                }
                let momentum = mat_vec(&m.inertia[body], &k.v[body]);
                let net = {
                    let inertial = mat_vec(&m.inertia[body], &a[body]);
                    let gyroscopic = cross_force(&k.v[body], &momentum);
                    let mut out = [0.0; 6];
                    for i in 0..6 {
                        out[i] = inertial[i] + gyroscopic[i];
                    }
                    out
                };
                rate += dot6(&k.v[body], &net);
            }

            // Power the actuators and gravity put in.
            let mut power: f64 = (0..m.joints).map(|j| tau[j] * state.qd[j]).sum();
            for body in 0..=m.joints {
                power += dot6(&k.v[body], &m.weight(body, &k.rot[body]));
            }

            assert!(
                (rate - power).abs() < 1.0e-9 * power.abs().max(1.0),
                "seed {seed}: energy grew at {rate} while {power} went in"
            );
        }
    }

    /// Free fall costs nothing. Every body accelerating together at `g` needs
    /// no joint torque and no wrench from the world — the check that gravity is
    /// a field and not eighteen fudge terms.
    #[test]
    fn free_fall_needs_no_torque_and_no_wrench() {
        let m = model();
        let mut state = State::default();
        let mut rng = Rng::new(0xFA11);
        for joint in 0..m.joints {
            state.q[joint] = rng.normal() * 0.5;
        }
        state.base_rot = m3_mul(&rot_z(-0.4), &rot_y(0.8));

        // Still, so the whole machine is a rigid body, falling.
        let g_base = m3_mul_v(&state.base_rot, m.gravity);
        let base_a = [0.0, 0.0, 0.0, g_base[0], g_base[1], g_base[2]];
        let (tau, wrench) = rnea(
            &m,
            &state,
            &[0.0; MAX_JOINTS],
            &base_a,
            &External::default(),
        );

        for joint in 0..m.joints {
            assert!(
                tau[joint].abs() < 1.0e-9,
                "joint {joint} needed {} N-m to fall",
                tau[joint]
            );
        }
        for i in 0..6 {
            assert!(
                wrench[i].abs() < 1.0e-9,
                "falling needed a wrench from the world: component {i} is {}",
                wrench[i]
            );
        }
    }

    /// And the same thing from the other side: released from rest with no
    /// torques, the base accelerates at exactly `g` and no joint moves.
    #[test]
    fn released_from_rest_the_whole_machine_falls_together() {
        let m = model();
        let mut state = State::default();
        state.q[1] = 0.3;
        state.q[2] = -0.6;
        state.base_rot = rot_y(0.5);
        let (base_a, qdd) = aba(&m, &state, &[0.0; MAX_JOINTS], &External::default());

        let expected = m3_mul_v(&state.base_rot, m.gravity);
        for i in 0..3 {
            assert!(base_a[i].abs() < 1.0e-9, "it started spinning: {base_a:?}");
            assert!(
                (base_a[i + 3] - expected[i]).abs() < 1.0e-9,
                "fell at {:?} instead of {expected:?}",
                &base_a[3..]
            );
        }
        for joint in 0..m.joints {
            assert!(
                qdd[joint].abs() < 1.0e-9,
                "joint {joint} moved while falling: {}",
                qdd[joint]
            );
        }
    }

    /// The articulated inertia about every joint axis stays positive. This is
    /// what a massless coxa link costs: it is only solvable because it has
    /// massive children, and this is the assertion that notices if that stops
    /// being true.
    #[test]
    fn articulated_inertia_stays_positive() {
        let (m, state, _) = awkward(11);
        // `aba` debug-asserts it internally; run it in a configuration where
        // the legs are folded as far as the travel allows, which is where the
        // projection is smallest.
        let mut folded = state;
        for leg in 0..m.frame.legs() {
            folded.q[3 * leg] = 0.0;
            folded.q[3 * leg + 1] = 1.9;
            folded.q[3 * leg + 2] = -2.4;
        }
        let (_, qdd) = aba(&m, &folded, &[0.0; MAX_JOINTS], &External::default());
        for joint in 0..m.joints {
            assert!(qdd[joint].is_finite(), "joint {joint} produced {}", qdd[joint]);
        }
    }
}
