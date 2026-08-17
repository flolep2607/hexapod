//! Small vector helpers, a deterministic PRNG, and 2-D convex-hull utilities.
//!
//! Everything is `f64` and dependency-free so the same code runs natively and
//! on `wasm32-unknown-unknown`.

pub type V3 = [f64; 3];

#[inline]
pub fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[inline]
pub fn lerp3(a: V3, b: V3, t: f64) -> V3 {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

#[inline]
pub fn hypot2(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

/// Rotate `v` about the world Y axis. Maps `+X` toward `+Z` for positive `a`.
#[inline]
pub fn rot_y(v: V3, a: f64) -> V3 {
    let (s, c) = a.sin_cos();
    [v[0] * c - v[2] * s, v[1], v[0] * s + v[2] * c]
}

/// Inverse of [`rot_y`].
#[inline]
pub fn inv_rot_y(v: V3, a: f64) -> V3 {
    rot_y(v, -a)
}

/// Rotate about X (pitch), then Z (roll), then Y (yaw) — the body's frame.
#[inline]
pub fn body_to_world(v: V3, yaw: f64, pitch: f64, roll: f64) -> V3 {
    let (sp, cp) = pitch.sin_cos();
    let p = [v[0], v[1] * cp - v[2] * sp, v[1] * sp + v[2] * cp];
    let (sr, cr) = roll.sin_cos();
    let r = [p[0] * cr - p[1] * sr, p[0] * sr + p[1] * cr, p[2]];
    rot_y(r, yaw)
}

/// Inverse of [`body_to_world`].
#[inline]
pub fn world_to_body(v: V3, yaw: f64, pitch: f64, roll: f64) -> V3 {
    let y = rot_y(v, -yaw);
    let (sr, cr) = (-roll).sin_cos();
    let r = [y[0] * cr - y[1] * sr, y[0] * sr + y[1] * cr, y[2]];
    let (sp, cp) = (-pitch).sin_cos();
    [r[0], r[1] * cp - r[2] * sp, r[1] * sp + r[2] * cp]
}

/// Wrap into `[0, 1)`.
#[inline]
pub fn frac(x: f64) -> f64 {
    let f = x - x.floor();
    if f >= 1.0 {
        0.0
    } else {
        f
    }
}

/// Shortest signed angular difference, in `(-PI, PI]`.
#[inline]
pub fn ang_diff(a: f64) -> f64 {
    let mut d = a % (2.0 * core::f64::consts::PI);
    if d > core::f64::consts::PI {
        d -= 2.0 * core::f64::consts::PI;
    }
    if d < -core::f64::consts::PI {
        d += 2.0 * core::f64::consts::PI;
    }
    d
}

/// Map an unbounded real into `(lo, hi)` so the optimiser can search freely
/// while the simulator only ever sees physically sane parameters.
#[inline]
pub fn squash(raw: f64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * 0.5 * (raw.tanh() + 1.0)
}

/// Inverse of [`squash`], used to seed the optimiser from a hand-tuned gait.
#[inline]
pub fn unsquash(v: f64, lo: f64, hi: f64) -> f64 {
    let t = clamp((v - lo) / (hi - lo), 1e-6, 1.0 - 1e-6);
    (2.0 * t - 1.0).atanh()
}

/// xoshiro-style deterministic PRNG. Reproducible across platforms, which
/// matters because a training run has to be replayable from its seed.
#[derive(Clone)]
pub struct Rng {
    s: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            s: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Uniform in `[lo, hi)`.
    #[inline]
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Standard normal via Box-Muller.
    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}

/// Convex hull of up to 8 points on the XZ plane, counter-clockwise when
/// viewed from above. Monotone chain; returns the number of hull points
/// written into `out`.
/// Most points a support polygon can have: one per foot.
pub const MAX_HULL: usize = crate::robot::MAX_LEGS;

pub fn convex_hull_xz(pts: &[[f64; 2]], out: &mut [[f64; 2]; MAX_HULL]) -> usize {
    let n = pts.len();
    if n == 0 {
        return 0;
    }
    if n <= 2 {
        for (i, p) in pts.iter().enumerate() {
            out[i] = *p;
        }
        return n;
    }

    let mut idx: [usize; MAX_HULL] = [0; MAX_HULL];
    for (i, slot) in idx.iter_mut().enumerate().take(n) {
        *slot = i;
    }
    // Insertion sort by (x, z) — n is at most one point per leg.
    for i in 1..n {
        let mut j = i;
        while j > 0 {
            let a = pts[idx[j - 1]];
            let b = pts[idx[j]];
            if a[0] > b[0] || (a[0] == b[0] && a[1] > b[1]) {
                idx.swap(j - 1, j);
                j -= 1;
            } else {
                break;
            }
        }
    }

    #[inline]
    fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    }

    let mut hull: [[f64; 2]; 2 * MAX_HULL] = [[0.0; 2]; 2 * MAX_HULL];
    let mut k = 0usize;

    for &i in idx.iter().take(n) {
        let p = pts[i];
        while k >= 2 && cross(hull[k - 2], hull[k - 1], p) <= 0.0 {
            k -= 1;
        }
        hull[k] = p;
        k += 1;
    }
    let lower = k + 1;
    for &i in idx.iter().take(n - 1).rev() {
        let p = pts[i];
        while k >= lower && cross(hull[k - 2], hull[k - 1], p) <= 0.0 {
            k -= 1;
        }
        hull[k] = p;
        k += 1;
    }

    let m = (k - 1).min(MAX_HULL);
    out[..m].copy_from_slice(&hull[..m]);
    m
}

/// Signed distance from `p` to a CCW convex polygon: positive inside,
/// negative outside. This is the static stability margin of the robot.
pub fn polygon_margin(hull: &[[f64; 2]], p: [f64; 2]) -> f64 {
    let n = hull.len();
    if n == 0 {
        return -1.0;
    }
    if n < 3 {
        // Degenerate support (0-2 feet down): treat as unstable.
        return -0.5;
    }
    let mut best = f64::INFINITY;
    for i in 0..n {
        let a = hull[i];
        let b = hull[(i + 1) % n];
        let ex = b[0] - a[0];
        let ez = b[1] - a[1];
        let len = hypot2(ex, ez);
        if len < 1e-9 {
            continue;
        }
        // Positive when p is to the left of a->b, i.e. inside a CCW polygon.
        let d = (ex * (p[1] - a[1]) - ez * (p[0] - a[0])) / len;
        if d < best {
            best = d;
        }
    }
    if best.is_finite() {
        best
    } else {
        -0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squash_roundtrip() {
        for v in [0.1, 0.5, 0.9, 1.5] {
            let r = unsquash(v, 0.0, 2.0);
            assert!((squash(r, 0.0, 2.0) - v).abs() < 1e-9, "v={v}");
        }
    }

    #[test]
    fn body_frame_roundtrip() {
        let v = [0.4, -0.9, 1.3];
        let w = body_to_world(v, 0.7, -0.2, 0.15);
        let b = world_to_body(w, 0.7, -0.2, 0.15);
        for i in 0..3 {
            assert!((b[i] - v[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn hull_of_square_is_ccw_and_contains_centre() {
        let pts = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]];
        let mut out = [[0.0; 2]; MAX_HULL];
        let n = convex_hull_xz(&pts, &mut out);
        assert_eq!(n, 4);
        assert!((polygon_margin(&out[..n], [0.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!(polygon_margin(&out[..n], [1.5, 0.0]) < 0.0);
    }

    #[test]
    fn hull_drops_interior_points() {
        let pts = [
            [2.0, 0.0],
            [0.0, 2.0],
            [-2.0, 0.0],
            [0.0, -2.0],
            [0.1, 0.1],
            [-0.2, 0.0],
        ];
        let mut out = [[0.0; 2]; MAX_HULL];
        let n = convex_hull_xz(&pts, &mut out);
        assert_eq!(n, 4, "interior points must not appear on the hull");
    }

    #[test]
    fn rng_is_deterministic_and_normal_is_centred() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        assert_eq!(a.next_u64(), b.next_u64());

        let mut r = Rng::new(1234);
        let n = 20000;
        let mut sum = 0.0;
        let mut sq = 0.0;
        for _ in 0..n {
            let v = r.normal();
            sum += v;
            sq += v * v;
        }
        let mean = sum / n as f64;
        let var = sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean={mean}");
        assert!((var - 1.0).abs() < 0.1, "var={var}");
    }
}
