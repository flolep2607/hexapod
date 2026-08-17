//! Obstacle courses, represented as an axis-aligned height field.
//!
//! The field is a list of rectangular prisms (positive `top` = block,
//! negative `top` = pit) bucketed along Z so that a height query touches only
//! a handful of candidates. Height lookups sit in the innermost loop of every
//! training rollout, so this is the one place worth keeping tight.
//!
//! A course is more than its obstacles. It is fenced by two invisible walls at
//! the corridor edges — there is nothing outside them and a policy that wanders
//! off is not solving anything — and it carries a **route**: a list of
//! waypoints the machine is asked to reach in order. On the open courses the
//! route runs straight down the middle and asking for it changes nothing. On
//! the ones with something in the way it threads the gaps, which is the whole
//! point of having it.

use crate::math::{clamp, Rng, V3};

/// Half-width of the walkable corridor, in metres. Also where the two
/// invisible walls are: the robot cannot cross them, and neither can a rock.
pub const CORRIDOR_HALF: f64 = 5.0;
/// Course start, behind the robot's spawn.
pub const Z_MIN: f64 = -6.0;
/// Course end. Rollouts are scored on progress toward this.
pub const Z_MAX: f64 = 64.0;

/// Spacing of the waypoints on a course with nothing to steer around.
const ROUTE_STEP: f64 = 8.0;
/// How close counts as reached, in metres.
pub const WAYPOINT_R: f64 = 1.6;

const BUCKET: f64 = 2.0;
const N_BUCKETS: usize = ((Z_MAX - Z_MIN) / BUCKET) as usize + 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Course {
    Flat = 0,
    Steps = 1,
    Rubble = 2,
    Gaps = 3,
    Mixed = 4,
    Ramps = 5,
    Slalom = 6,
    Slick = 7,
    Gauntlet = 8,
    Jump = 9,
}

/// Every course, in the order the dashboard lists them.
pub const COURSES: [Course; 10] = [
    Course::Flat,
    Course::Steps,
    Course::Rubble,
    Course::Gaps,
    Course::Mixed,
    Course::Ramps,
    Course::Slalom,
    Course::Slick,
    Course::Gauntlet,
    Course::Jump,
];

impl Course {
    pub fn from_u32(v: u32) -> Course {
        COURSES.get(v as usize).copied().unwrap_or(Course::Flat)
    }

    pub fn name(self) -> &'static str {
        match self {
            Course::Flat => "FLAT",
            Course::Steps => "STEPS",
            Course::Rubble => "RUBBLE",
            Course::Gaps => "GAPS",
            Course::Mixed => "MIXED",
            Course::Ramps => "RAMPS",
            Course::Slalom => "SLALOM",
            Course::Slick => "SLICK",
            Course::Gauntlet => "GAUNTLET",
            Course::Jump => "JUMP",
        }
    }

    /// JUMP is a parkour walking course: trenches wider than a stride, and
    /// platforms you can only reach by jumping the gap in front of them. The
    /// reward is still speed tracking; the jump action is how you stay on it.
    #[inline]
    pub fn is_jump(self) -> bool {
        matches!(self, Course::Jump)
    }
}

/// Friction multiplier of the base ground. Everything else is measured
/// against it.
pub const GRIP_GROUND: f64 = 1.0;
/// Loose debris shifts underfoot; a foot on rubble grips noticeably less.
pub const GRIP_RUBBLE: f64 = 0.62;
/// A step tread is firm but its edge is not, and feet land near edges.
pub const GRIP_STEP: f64 = 0.88;
/// Down in a trench, on whatever collected there.
pub const GRIP_PIT: f64 = 0.75;
/// A sheet of ice. Thin enough to be no obstacle at all, and the reason the
/// traction meter exists.
pub const GRIP_ICE: f64 = 0.22;
/// How thick that sheet is. It has to have *some* height or the height field,
/// which resolves grip through whichever surface supplies the height, would
/// never see it.
const ICE_THICK: f64 = 0.01;
/// Height of a slalom wall. Well above anything a leg can reach, so it is not
/// an obstacle to climb — it is somewhere the machine cannot go.
pub const WALL_TOP: f64 = 1.8;
/// Keep query points this far outside a solid so sitting on a face does not
/// count as being inside it.
const WALL_PAD: f64 = 0.02;

#[derive(Clone, Copy, Debug)]
pub struct Obstacle {
    pub x0: f64,
    pub x1: f64,
    pub z0: f64,
    pub z1: f64,
    /// Height above ground for a block, or negative depth for a pit.
    pub top: f64,
    /// Friction multiplier of this surface, relative to the base ground.
    pub grip: f64,
}

impl Obstacle {
    #[inline]
    fn contains(&self, x: f64, z: f64) -> bool {
        x >= self.x0 && x <= self.x1 && z >= self.z0 && z <= self.z1
    }

    /// Closest point of the rectangle to `(x, z)` is within `r`. This is the
    /// exact disc test; sampling the circumference misses the cardinal
    /// extrema and any wall that sits between those six points.
    #[inline]
    fn intersects_disc(&self, x: f64, z: f64, r: f64) -> bool {
        let qx = clamp(x, self.x0, self.x1);
        let qz = clamp(z, self.z0, self.z1);
        let dx = x - qx;
        let dz = z - qz;
        dx * dx + dz * dz <= r * r
    }
}

/// Slab test: does the segment `a→b` overlap the AABB? `t` in `[0, 1]`.
fn segment_hits_aabb(a: V3, b: V3, x0: f64, x1: f64, y0: f64, y1: f64, z0: f64, z1: f64) -> bool {
    let p = a;
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let lo = [x0, y0, z0];
    let hi = [x1, y1, z1];
    let mut tmin = 0.0f64;
    let mut tmax = 1.0f64;
    for i in 0..3 {
        if d[i].abs() < 1e-14 {
            if p[i] < lo[i] || p[i] > hi[i] {
                return false;
            }
            continue;
        }
        let mut t1 = (lo[i] - p[i]) / d[i];
        let mut t2 = (hi[i] - p[i]) / d[i];
        if t1 > t2 {
            core::mem::swap(&mut t1, &mut t2);
        }
        tmin = tmin.max(t1);
        tmax = tmax.min(t2);
        if tmin > tmax {
            return false;
        }
    }
    true
}

#[derive(Clone)]
pub struct Terrain {
    pub course: Course,
    pub seed: u64,
    pub obstacles: Vec<Obstacle>,
    /// The route, in order. Always ends at the far end of the corridor.
    pub waypoints: Vec<[f64; 2]>,
    buckets: Vec<Vec<u16>>,
}

impl Terrain {
    pub fn new(course: Course, seed: u64) -> Terrain {
        let mut t = Terrain {
            course,
            seed,
            obstacles: Vec::new(),
            waypoints: Vec::new(),
            buckets: vec![Vec::new(); N_BUCKETS],
        };
        t.generate();
        t.finish_route();
        t.rebuild_buckets();
        if t.course.is_jump() {
            t.snap_waypoints_to_ground();
        }
        t
    }

    /// Where the two invisible walls are, as a distance from the centreline.
    #[inline]
    pub fn wall_x(&self) -> f64 {
        CORRIDOR_HALF
    }

    /// The waypoint at `i`, saturating at the last one so a machine that has
    /// run the whole course still has something to aim at.
    #[inline]
    pub fn waypoint(&self, i: usize) -> [f64; 2] {
        let n = self.waypoints.len();
        if n == 0 {
            return [0.0, Z_MAX];
        }
        self.waypoints[i.min(n - 1)]
    }

    /// Would a chassis of radius `r`, its underside at `under`, be inside
    /// something here? The corridor walls answer this the same way a rock does,
    /// which is the point: there is one obstruction test, not two.
    pub fn obstructed(&self, x: f64, z: f64, r: f64, under: f64) -> bool {
        if x.abs() > CORRIDOR_HALF - r {
            return true;
        }
        self.height_disc(x, z, r) > under + 0.02
    }

    #[inline]
    fn bucket_of(z: f64) -> usize {
        let i = ((z - Z_MIN) / BUCKET).floor();
        if i < 0.0 {
            0
        } else if i as usize >= N_BUCKETS {
            N_BUCKETS - 1
        } else {
            i as usize
        }
    }

    fn rebuild_buckets(&mut self) {
        for b in self.buckets.iter_mut() {
            b.clear();
        }
        for (i, ob) in self.obstacles.iter().enumerate() {
            let b0 = Self::bucket_of(ob.z0);
            let b1 = Self::bucket_of(ob.z1);
            for b in b0..=b1 {
                self.buckets[b].push(i as u16);
            }
        }
    }

    /// Terrain height at a point. Blocks win over pits where they overlap.
    #[inline]
    pub fn height(&self, x: f64, z: f64) -> f64 {
        if !(Z_MIN..=Z_MAX).contains(&z) || x.abs() > CORRIDOR_HALF {
            return 0.0;
        }
        let mut top = 0.0f64;
        let mut pit = 0.0f64;
        for &i in &self.buckets[Self::bucket_of(z)] {
            let ob = &self.obstacles[i as usize];
            if ob.contains(x, z) {
                if ob.top >= 0.0 {
                    if ob.top > top {
                        top = ob.top;
                    }
                } else if ob.top < pit {
                    pit = ob.top;
                }
            }
        }
        if top > 0.0 {
            top
        } else {
            pit
        }
    }

    /// Friction multiplier of the surface a foot standing here would be on.
    ///
    /// Whichever obstacle supplies the height supplies the grip, so a foot on
    /// top of a rubble block gets the rubble's number and a foot beside it gets
    /// the ground's.
    pub fn grip(&self, x: f64, z: f64) -> f64 {
        if !(Z_MIN..=Z_MAX).contains(&z) || x.abs() > CORRIDOR_HALF {
            return GRIP_GROUND;
        }
        let mut top = 0.0f64;
        let mut pit = 0.0f64;
        let mut grip = GRIP_GROUND;
        let mut pit_grip = GRIP_GROUND;
        for &i in &self.buckets[Self::bucket_of(z)] {
            let ob = &self.obstacles[i as usize];
            if ob.contains(x, z) {
                if ob.top >= 0.0 {
                    if ob.top > top {
                        top = ob.top;
                        grip = ob.grip;
                    }
                } else if ob.top < pit {
                    pit = ob.top;
                    pit_grip = ob.grip;
                }
            }
        }
        if top > 0.0 {
            grip
        } else if pit < 0.0 {
            pit_grip
        } else {
            GRIP_GROUND
        }
    }

    /// Height a forward-looking sensor would report. The same as [`height`]
    /// inside the corridor, and the fence outside it: the walls are invisible,
    /// not undetectable, and a policy that cannot see them cannot avoid them.
    ///
    /// [`height`]: Terrain::height
    #[inline]
    pub fn probe(&self, x: f64, z: f64) -> f64 {
        if x.abs() >= CORRIDOR_HALF {
            WALL_TOP
        } else {
            self.height(x, z)
        }
    }

    /// Surface gradient `(dy/dx, dy/dz)` around a point, by central difference.
    /// Used for the downslope pull and for the cosine loss on normal load.
    pub fn slope(&self, x: f64, z: f64) -> (f64, f64) {
        const H: f64 = 0.12;
        let gx = (self.height(x + H, z) - self.height(x - H, z)) / (2.0 * H);
        let gz = (self.height(x, z + H) - self.height(x, z - H)) / (2.0 * H);
        (gx, gz)
    }

    /// Highest point under a disc — used for body-clearance checks.
    ///
    /// This is an exact disc-versus-AABB test, not a handful of sample
    /// points. Six samples on the circumference sit at `0.866 r` along the
    /// cardinals, so a wall the chassis has already overlapped by a tenth of
    /// a metre used to be invisible, and a wall in the next Z-bucket was not
    /// even a candidate.
    pub fn height_disc(&self, x: f64, z: f64, r: f64) -> f64 {
        let r = r.max(0.0);
        let mut h = 0.0f64;
        let b0 = Self::bucket_of(z - r);
        let b1 = Self::bucket_of(z + r);
        for b in b0..=b1 {
            for &i in &self.buckets[b] {
                let ob = &self.obstacles[i as usize];
                if ob.top <= h {
                    continue;
                }
                if ob.intersects_disc(x, z, r) {
                    h = ob.top;
                }
            }
        }
        if h > 0.0 {
            h
        } else {
            self.height(x, z)
        }
    }

    /// Is this column something a foot cannot stand on? Relative to `floor`
    /// so a staircase the machine is already climbing does not turn into a
    /// wall, and a slalom wall the machine is standing in front of does.
    #[inline]
    pub fn blocked_column(&self, x: f64, z: f64, floor: f64, max_step: f64) -> bool {
        if x.abs() >= CORRIDOR_HALF {
            return true;
        }
        self.height(x, z) - floor > max_step
    }

    /// Nearest point that is not inside an unclimbable column. A step aimed
    /// at the middle of a slalom wall used to be left there, at a capped
    /// height, which is how a foot (and then the whole support polygon) ended
    /// up inside the block.
    pub fn push_xz(&self, mut x: f64, mut z: f64, floor: f64, max_step: f64) -> (f64, f64) {
        let limit = CORRIDOR_HALF - WALL_PAD;
        x = clamp(x, -limit, limit);
        for _ in 0..4 {
            if !self.blocked_column(x, z, floor, max_step) {
                return (x, z);
            }
            let mut best: Option<Obstacle> = None;
            for &i in &self.buckets[Self::bucket_of(z)] {
                let ob = self.obstacles[i as usize];
                if ob.contains(x, z) && ob.top - floor > max_step
                    && best.map(|b| ob.top > b.top).unwrap_or(true) {
                        best = Some(ob);
                    }
            }
            let Some(ob) = best else {
                return (x, z);
            };
            let left = x - ob.x0;
            let right = ob.x1 - x;
            let back = z - ob.z0;
            let fwd = ob.z1 - z;
            if left <= right && left <= back && left <= fwd {
                x = ob.x0 - WALL_PAD;
            } else if right <= back && right <= fwd {
                x = ob.x1 + WALL_PAD;
            } else if back <= fwd {
                z = ob.z0 - WALL_PAD;
            } else {
                z = ob.z1 + WALL_PAD;
            }
            x = clamp(x, -limit, limit);
        }
        (x, z)
    }

    /// A 3-D point inside an unclimbable prism, or past the corridor fence.
    /// Sitting on a face (within `WALL_PAD`) is contact, not penetration.
    pub fn solid_at(&self, p: V3, floor: f64, max_step: f64) -> bool {
        if p[1] <= 0.0 {
            return false;
        }
        if p[0].abs() > CORRIDOR_HALF + WALL_PAD && p[1] < WALL_TOP {
            return true;
        }
        if !self.blocked_column(p[0], p[2], floor, max_step) {
            return false;
        }
        if p[1] >= self.height(p[0], p[2]) - WALL_PAD {
            return false;
        }
        for (dx, dz) in [
            (WALL_PAD, 0.0),
            (-WALL_PAD, 0.0),
            (0.0, WALL_PAD),
            (0.0, -WALL_PAD),
        ] {
            if !self.blocked_column(p[0] + dx, p[2] + dz, floor, max_step) {
                return false;
            }
        }
        true
    }

    /// Would this segment enter an unclimbable solid? Climbable terrain stays
    /// a height field — only the foot catches on it — because treating a
    /// staircase as a volume would stop the machine walking up it.
    pub fn segment_hits_wall(&self, a: V3, b: V3, floor: f64, max_step: f64) -> bool {
        if self.solid_at(a, floor, max_step) || self.solid_at(b, floor, max_step) {
            return true;
        }
        let b0 = Self::bucket_of(a[2].min(b[2]));
        let b1 = Self::bucket_of(a[2].max(b[2]));
        for bucket in b0..=b1 {
            for &i in &self.buckets[bucket] {
                let ob = &self.obstacles[i as usize];
                if ob.top <= 0.0 || ob.top - floor <= max_step {
                    continue;
                }
                let x0 = ob.x0 + WALL_PAD;
                let x1 = ob.x1 - WALL_PAD;
                let z0 = ob.z0 + WALL_PAD;
                let z1 = ob.z1 - WALL_PAD;
                if x1 <= x0 || z1 <= z0 {
                    continue;
                }
                if segment_hits_aabb(a, b, x0, x1, 0.0, ob.top, z0, z1) {
                    return true;
                }
            }
        }
        false
    }

    /// Would a link pass through the interior of any positive-height terrain
    /// prism? Unlike [`segment_hits_wall`], this includes climbable blocks.
    /// A foot may rest on a top face, but a femur or tibia may not tunnel
    /// through the volume merely because the block is short enough to climb.
    pub fn segment_hits_solid(&self, a: V3, b: V3) -> bool {
        for p in [a, b] {
            if p[0].abs() > CORRIDOR_HALF + WALL_PAD && p[1] > WALL_PAD && p[1] < WALL_TOP - WALL_PAD {
                return true;
            }
        }
        let b0 = Self::bucket_of(a[2].min(b[2]));
        let b1 = Self::bucket_of(a[2].max(b[2]));
        for bucket in b0..=b1 {
            for &i in &self.buckets[bucket] {
                let ob = &self.obstacles[i as usize];
                if ob.top <= WALL_PAD * 2.0 {
                    continue;
                }
                let x0 = ob.x0 + WALL_PAD;
                let x1 = ob.x1 - WALL_PAD;
                let z0 = ob.z0 + WALL_PAD;
                let z1 = ob.z1 - WALL_PAD;
                let y0 = WALL_PAD;
                let y1 = ob.top - WALL_PAD;
                if x1 <= x0 || y1 <= y0 || z1 <= z0 {
                    continue;
                }
                if segment_hits_aabb(a, b, x0, x1, y0, y1, z0, z1) {
                    return true;
                }
            }
        }
        false
    }

    fn push(&mut self, x0: f64, x1: f64, z0: f64, z1: f64, top: f64, grip: f64) {
        self.obstacles.push(Obstacle {
            x0,
            x1,
            z0,
            z1,
            top,
            grip,
        });
    }

    fn generate(&mut self) {
        let mut r = Rng::new(self.seed ^ ((self.course as u64) << 32));
        match self.course {
            Course::Flat => {}
            Course::Steps => self.gen_steps(&mut r, 6.0, Z_MAX - 4.0),
            Course::Rubble => self.gen_rubble(&mut r, 5.0, Z_MAX - 4.0, 1.0),
            Course::Gaps => self.gen_gaps(&mut r, 6.0, Z_MAX - 4.0),
            Course::Mixed => {
                self.gen_rubble(&mut r, 5.0, 20.0, 0.75);
                self.gen_steps(&mut r, 22.0, 38.0);
                self.gen_gaps(&mut r, 40.0, 52.0);
                self.gen_rubble(&mut r, 53.0, Z_MAX - 4.0, 1.1);
            }
            Course::Ramps => self.gen_ramps(&mut r, 6.0, Z_MAX - 4.0),
            Course::Slalom => self.gen_slalom(&mut r, 8.0, Z_MAX - 6.0),
            Course::Slick => self.gen_slick(&mut r, 5.0, Z_MAX - 4.0),
            Course::Gauntlet => {
                self.gen_rubble(&mut r, 5.0, 14.0, 0.9);
                self.gen_ramps(&mut r, 15.0, 27.0);
                self.gen_slalom(&mut r, 28.0, 42.0);
                self.gen_gaps(&mut r, 43.0, 50.0);
                self.gen_slick(&mut r, 51.0, Z_MAX - 4.0);
            }
            Course::Jump => self.gen_parkour(&mut r),
        }
    }

    /// Parkour: trenches too wide to step (max stride 1.45 m) and platforms
    /// you can only reach by jumping the gap in front of them. Adjacent
    /// platforms of this height would be a step; the gap is what makes them
    /// a jump.
    fn gen_parkour(&mut self, r: &mut Rng) {
        let mut z = 3.2 + r.range(0.0, 0.35);
        let mut n = 0usize;
        while z < Z_MAX - 8.0 {
            let gap = r.range(1.55, 1.95);
            let z1 = z + gap;
            self.push(-CORRIDOR_HALF, CORRIDOR_HALF, z, z1, -0.90, GRIP_PIT);
            if n % 2 == 1 {
                let h = r.range(0.16, 0.30);
                let len = r.range(2.4, 3.8);
                self.push(
                    -CORRIDOR_HALF,
                    CORRIDOR_HALF,
                    z1,
                    z1 + len,
                    h,
                    GRIP_STEP,
                );
                self.waypoints.push([0.0, z1 + (len * 0.45).min(1.6)]);
                z = z1 + len + r.range(2.0, 3.2);
            } else {
                self.waypoints.push([0.0, z1 + 1.4]);
                z = z1 + r.range(2.6, 4.2);
            }
            n += 1;
        }
    }

    /// Centreline stations that landed in a trench are walked forward onto
    /// solid ground. JUMP places its own landings; this is the backstop so a
    /// finish_route station cannot ask the machine to stand in a pit.
    fn snap_waypoints_to_ground(&mut self) {
        let n = self.waypoints.len();
        for i in 0..n {
            let (x, mut z) = (self.waypoints[i][0], self.waypoints[i][1]);
            if self.height(x, z) >= -0.12 {
                continue;
            }
            while z < Z_MAX - 2.0 && self.height(x, z) < -0.12 {
                z += 0.20;
            }
            self.waypoints[i][1] = z;
        }
    }

    /// Ascending / descending staircases spanning the corridor.
    fn gen_steps(&mut self, r: &mut Rng, z_from: f64, z_to: f64) {
        let mut z = z_from;
        while z < z_to - 4.0 {
            let n_up = 3 + (r.unit() * 3.0) as usize;
            let rise = r.range(0.16, 0.34);
            let tread = r.range(0.45, 0.80);
            // Up.
            for k in 0..n_up {
                let h = rise * (k + 1) as f64;
                self.push(-CORRIDOR_HALF, CORRIDOR_HALF, z, z + tread, h, GRIP_STEP);
                z += tread;
            }
            // Plateau.
            let plateau = r.range(1.0, 2.4);
            let top_h = rise * n_up as f64;
            self.push(
                -CORRIDOR_HALF,
                CORRIDOR_HALF,
                z,
                z + plateau,
                top_h,
                GRIP_STEP,
            );
            z += plateau;
            // Down.
            for k in (0..n_up).rev() {
                let h = rise * k as f64;
                if h > 0.0 {
                    self.push(-CORRIDOR_HALF, CORRIDOR_HALF, z, z + tread, h, GRIP_STEP);
                }
                z += tread;
            }
            z += r.range(1.5, 3.5);
        }
    }

    /// Scattered debris. `density` scales the count.
    fn gen_rubble(&mut self, r: &mut Rng, z_from: f64, z_to: f64, density: f64) {
        let span = z_to - z_from;
        let n = (span * 2.8 * density) as usize;
        for _ in 0..n {
            let cx = r.range(-CORRIDOR_HALF + 0.2, CORRIDOR_HALF - 0.2);
            let cz = r.range(z_from, z_to);
            let sx = r.range(0.30, 0.95);
            let sz = r.range(0.30, 0.95);
            let h = r.range(0.10, 0.58);
            self.push(
                cx - sx * 0.5,
                cx + sx * 0.5,
                cz - sz * 0.5,
                cz + sz * 0.5,
                h,
                GRIP_RUBBLE,
            );
        }
    }

    /// Trenches across the corridor.
    fn gen_gaps(&mut self, r: &mut Rng, z_from: f64, z_to: f64) {
        let mut z = z_from;
        while z < z_to {
            let w = r.range(0.45, 1.05);
            self.push(-CORRIDOR_HALF, CORRIDOR_HALF, z, z + w, -0.90, GRIP_PIT);
            z += w + r.range(2.5, 5.0);
        }
    }

    /// Long grades, half of them banked across the corridor.
    ///
    /// A staircase is a sequence of shocks; a ramp is a sustained tilt, and the
    /// two are not the same problem. A banked one is worse still: the support
    /// plane rolls, the downhill legs carry more, and the machine slides
    /// sideways unless it does something about it.
    fn gen_ramps(&mut self, r: &mut Rng, z_from: f64, z_to: f64) {
        const SLAB: f64 = 0.30;
        const STRIPS: usize = 6;
        let mut z = z_from;
        while z < z_to - 6.0 {
            let run = r.range(3.5, 6.5);
            let rise = r.range(0.55, 1.30);
            // Cross-fall from one edge of the corridor to the other, or none.
            let bank = if r.unit() < 0.45 {
                r.range(0.30, 0.75) * if r.unit() < 0.5 { 1.0 } else { -1.0 }
            } else {
                0.0
            };
            let flat = r.range(1.2, 2.6);
            let total = run * 2.0 + flat;
            let mut d = 0.0;
            while d < total {
                // Up the first run, level across the crown, down the second.
                let h = if d < run {
                    rise * d / run
                } else if d < run + flat {
                    rise
                } else {
                    rise * (1.0 - (d - run - flat) / run)
                };
                // A level slab is one prism; a banked one is the same slab cut
                // into strips that step down across the corridor. Height
                // lookups are the innermost loop of every rollout, so an
                // unbanked ramp is not paid for six times over.
                let n = if bank == 0.0 { 1 } else { STRIPS };
                let w = 2.0 * CORRIDOR_HALF / n as f64;
                for s in 0..n {
                    let x0 = -CORRIDOR_HALF + w * s as f64;
                    // Strip centre in [-1, 1] across the corridor.
                    let u = (x0 + w * 0.5) / CORRIDOR_HALF;
                    let top = (h + bank * u).max(0.02);
                    self.push(x0, x0 + w, z + d, z + d + SLAB, top, GRIP_STEP);
                }
                d += SLAB;
            }
            z += total + r.range(2.0, 4.0);
        }
    }

    /// Staggered walls with a gate in each, and the route threaded through
    /// them. Nothing here can be climbed or stepped over — the only way past a
    /// wall is round it, which is why the machine needs somewhere to steer to.
    fn gen_slalom(&mut self, r: &mut Rng, z_from: f64, z_to: f64) {
        // Wide enough for the machine and its legs to fit through — a gate a
        // walker cannot physically pass is not an obstacle course, it is a
        // dead end.
        const GATE_HALF: f64 = 1.75;
        const THICK: f64 = 0.7;
        let mut z = z_from;
        // Alternate which side the gap is on, so the route actually weaves
        // instead of happening to line up.
        let mut side = if r.unit() < 0.5 { 1.0 } else { -1.0 };
        while z < z_to {
            // Bounded so both wall segments survive: an opening against the
            // corridor edge is a corner to cut, not a gate to aim at.
            let reach = CORRIDOR_HALF - GATE_HALF - 0.75;
            let gate = side * r.range(reach * 0.70, reach);
            if gate - GATE_HALF > -CORRIDOR_HALF {
                self.push(
                    -CORRIDOR_HALF,
                    gate - GATE_HALF,
                    z,
                    z + THICK,
                    WALL_TOP,
                    GRIP_STEP,
                );
            }
            if gate + GATE_HALF < CORRIDOR_HALF {
                self.push(
                    gate + GATE_HALF,
                    CORRIDOR_HALF,
                    z,
                    z + THICK,
                    WALL_TOP,
                    GRIP_STEP,
                );
            }
            // Approach and exit, so the machine lines up before the gap rather
            // than arriving at it sideways.
            self.waypoints.push([gate, z - 2.2]);
            self.waypoints.push([gate, z + THICK + 1.4]);
            side = -side;
            z += THICK + r.range(5.0, 7.5);
        }
    }

    /// Sheets of ice, one centimetre thick and worth about a fifth of the grip
    /// of the ground around them.
    fn gen_slick(&mut self, r: &mut Rng, z_from: f64, z_to: f64) {
        let span = z_to - z_from;
        let n = (span * 0.42) as usize;
        for _ in 0..n {
            let cx = r.range(-CORRIDOR_HALF + 1.0, CORRIDOR_HALF - 1.0);
            let cz = r.range(z_from, z_to);
            let sx = r.range(1.6, 4.2);
            let sz = r.range(1.6, 4.2);
            self.push(
                (cx - sx * 0.5).max(-CORRIDOR_HALF),
                (cx + sx * 0.5).min(CORRIDOR_HALF),
                cz - sz * 0.5,
                cz + sz * 0.5,
                ICE_THICK,
                GRIP_ICE,
            );
        }
        // A few low humps, so it is not only a friction test.
        self.gen_rubble(r, z_from, z_to, 0.25);
    }

    /// Fill in whatever the course did not route for itself: waypoints down
    /// the centreline, in order, ending past the far end of the obstacles.
    fn finish_route(&mut self) {
        self.waypoints.sort_by(|a, b| a[1].total_cmp(&b[1]));
        let mut out: Vec<[f64; 2]> = Vec::new();
        let mut z = ROUTE_STEP;
        let mut placed = self.waypoints.iter().copied().peekable();
        while z < Z_MAX - 2.0 {
            // Anything the generator placed before this station comes first,
            // and suppresses the station if it is close enough to serve.
            let mut covered = false;
            while placed.peek().is_some_and(|w| w[1] <= z) {
                let w = placed.next().unwrap();
                out.push(w);
                covered = true;
            }
            if !covered {
                out.push([0.0, z]);
            }
            z += ROUTE_STEP;
        }
        out.extend(placed.filter(|w| w[1] < Z_MAX - 2.0));
        out.push([0.0, Z_MAX - 2.0]);
        self.waypoints = out;
    }

    /// Flat `[x0, x1, z0, z1, top]` rows for the renderer.
    pub fn export(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.obstacles.len() * 5);
        for ob in &self.obstacles {
            out.push(ob.x0 as f32);
            out.push(ob.x1 as f32);
            out.push(ob.z0 as f32);
            out.push(ob.z1 as f32);
            out.push(ob.top as f32);
        }
        out
    }

    /// Flat `[x, z]` rows for the renderer.
    pub fn export_route(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.waypoints.len() * 2);
        for w in &self.waypoints {
            out.push(w[0] as f32);
            out.push(w[1] as f32);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_course_is_flat() {
        let t = Terrain::new(Course::Flat, 1);
        assert_eq!(t.height(0.0, 10.0), 0.0);
        assert!(t.obstacles.is_empty());
    }

    #[test]
    fn generation_is_deterministic() {
        let a = Terrain::new(Course::Mixed, 42);
        let b = Terrain::new(Course::Mixed, 42);
        assert_eq!(a.obstacles.len(), b.obstacles.len());
        for (x, y) in a.obstacles.iter().zip(b.obstacles.iter()) {
            assert_eq!(x.z0, y.z0);
            assert_eq!(x.top, y.top);
        }
        let c = Terrain::new(Course::Mixed, 43);
        assert!(a.obstacles.len() != c.obstacles.len() || a.obstacles[0].z0 != c.obstacles[0].z0);
    }

    #[test]
    fn bucket_lookup_matches_brute_force() {
        let t = Terrain::new(Course::Mixed, 9);
        let mut r = Rng::new(5);
        for _ in 0..4000 {
            let x = r.range(-CORRIDOR_HALF, CORRIDOR_HALF);
            let z = r.range(Z_MIN, Z_MAX);
            let fast = t.height(x, z);

            let mut top = 0.0f64;
            let mut pit = 0.0f64;
            for ob in &t.obstacles {
                if ob.contains(x, z) {
                    if ob.top >= 0.0 {
                        top = top.max(ob.top);
                    } else {
                        pit = pit.min(ob.top);
                    }
                }
            }
            let brute = if top > 0.0 { top } else { pit };
            assert!((fast - brute).abs() < 1e-12, "at ({x},{z})");
        }
    }

    #[test]
    fn courses_actually_contain_obstacles() {
        for c in COURSES.iter().copied().filter(|c| *c != Course::Flat) {
            let t = Terrain::new(c, 3);
            assert!(!t.obstacles.is_empty(), "{c:?} generated nothing");
        }
        assert!(Terrain::new(Course::Gaps, 3)
            .obstacles
            .iter()
            .any(|o| o.top < 0.0));
    }

    #[test]
    fn grip_follows_whichever_surface_supplies_the_height() {
        let t = Terrain::new(Course::Rubble, 7);
        let ob = t.obstacles[0];
        let (cx, cz) = ((ob.x0 + ob.x1) * 0.5, (ob.z0 + ob.z1) * 0.5);
        assert_eq!(t.height(cx, cz), ob.top);
        assert_eq!(t.grip(cx, cz), GRIP_RUBBLE);
        // Off the course entirely, and behind the first obstacle.
        assert_eq!(t.grip(0.0, Z_MIN - 1.0), GRIP_GROUND);
        assert_eq!(t.grip(0.0, 0.0), GRIP_GROUND);
        const { assert!(GRIP_RUBBLE < GRIP_GROUND) };
    }

    #[test]
    fn gaps_are_slippery_underfoot_and_still_pits() {
        let t = Terrain::new(Course::Gaps, 3);
        let ob = t.obstacles.iter().find(|o| o.top < 0.0).unwrap();
        let cz = (ob.z0 + ob.z1) * 0.5;
        assert!(t.height(0.0, cz) < 0.0);
        assert_eq!(t.grip(0.0, cz), GRIP_PIT);
    }

    #[test]
    fn slope_matches_the_height_field_it_is_measured_from() {
        let t = Terrain::new(Course::Steps, 2);
        // Flat ground has no gradient anywhere the corridor is empty.
        assert_eq!(t.slope(0.0, 0.0), (0.0, 0.0));
        // Step risers do, and the gradient agrees with the heights either
        // side of the edge it is measured across.
        const H: f64 = 0.12;
        let edge = t
            .obstacles
            .iter()
            .map(|o| o.z0)
            .find(|&z| (t.height(0.0, z + H) - t.height(0.0, z - H)).abs() > 0.05)
            .expect("a staircase with no risers");
        let (_, gz) = t.slope(0.0, edge);
        let expect = (t.height(0.0, edge + H) - t.height(0.0, edge - H)) / (2.0 * H);
        assert!((gz - expect).abs() < 1e-12);
        assert!(gz.abs() > 0.2, "riser gradient too shallow: {gz}");
    }

    #[test]
    fn outside_corridor_is_ground() {
        let t = Terrain::new(Course::Rubble, 2);
        assert_eq!(t.height(CORRIDOR_HALF + 1.0, 20.0), 0.0);
        assert_eq!(t.height(0.0, Z_MAX + 5.0), 0.0);
    }

    #[test]
    fn every_course_carries_a_route_that_runs_the_length_of_it() {
        for c in COURSES {
            let t = Terrain::new(c, 5);
            let w = &t.waypoints;
            assert!(w.len() >= 6, "{c:?}: only {} waypoints", w.len());
            // In order, inside the walls, and finishing at the far end.
            for pair in w.windows(2) {
                assert!(
                    pair[1][1] > pair[0][1],
                    "{c:?}: waypoints out of order at z={}",
                    pair[0][1]
                );
            }
            assert!(w.iter().all(|p| p[0].abs() < CORRIDOR_HALF - 1.0));
            assert!(w[0][1] > 0.0, "{c:?}: first waypoint is behind the spawn");
            assert!(w.last().unwrap()[1] > Z_MAX - 4.0);
        }
    }

    #[test]
    fn the_slalom_route_goes_round_the_walls_and_not_through_them() {
        let t = Terrain::new(Course::Slalom, 11);
        // Every wall is unclimbable, and the gap in it is where the route goes.
        let walls: Vec<_> = t.obstacles.iter().filter(|o| o.top > 1.0).collect();
        assert!(walls.len() >= 8, "only {} wall segments", walls.len());
        for w in &walls {
            let cz = (w.z0 + w.z1) * 0.5;
            assert!(t.height((w.x0 + w.x1) * 0.5, cz) > 1.0);
        }
        // Every waypoint has somewhere to stand.
        for p in &t.waypoints {
            assert!(
                t.height(p[0], p[1]) < 1.0,
                "waypoint ({:.2}, {:.2}) is inside a wall",
                p[0],
                p[1]
            );
        }
        // And the route is not a straight line down the middle.
        let sway = t
            .waypoints
            .iter()
            .map(|p| p[0].abs())
            .fold(0.0f64, f64::max);
        assert!(sway > 1.0, "route never leaves the centreline: {sway:.2}");
    }

    #[test]
    fn the_corridor_walls_obstruct_and_so_does_a_slalom_wall() {
        let t = Terrain::new(Course::Slalom, 4);
        let (r, under) = (0.9, 0.7);
        assert!(!t.obstructed(0.0, 2.0, r, under), "open ground is blocked");
        assert!(
            t.obstructed(CORRIDOR_HALF - 0.5, 2.0, r, under),
            "walked through the wall"
        );
        assert!(t.obstructed(-CORRIDOR_HALF - 3.0, 2.0, r, under));
        let w = t.obstacles.iter().find(|o| o.top > 1.0).unwrap();
        assert!(t.obstructed((w.x0 + w.x1) * 0.5, (w.z0 + w.z1) * 0.5, r, under));
    }

    #[test]
    fn ice_is_slippery_without_being_an_obstacle() {
        let t = Terrain::new(Course::Slick, 6);
        let ice = t
            .obstacles
            .iter()
            .find(|o| o.grip == GRIP_ICE)
            .expect("no ice on the slick course");
        let (cx, cz) = ((ice.x0 + ice.x1) * 0.5, (ice.z0 + ice.z1) * 0.5);
        assert_eq!(t.grip(cx, cz), GRIP_ICE);
        assert!(t.height(cx, cz) < 0.02, "ice is not a step");
        const { assert!(GRIP_ICE < GRIP_RUBBLE) };
    }

    #[test]
    fn a_ramp_is_a_grade_and_not_a_staircase() {
        let t = Terrain::new(Course::Ramps, 8);
        // Somewhere on the course the ground rises steadily over several
        // metres, which is what makes it a ramp and not a step.
        let climb = (0..600)
            .map(|i| {
                let z = 6.0 + i as f64 * 0.1;
                t.height(0.0, z)
            })
            .fold(0.0f64, f64::max);
        assert!(
            climb > 0.5,
            "highest point on the ramps is only {climb:.2} m"
        );
        // And at least one section is banked: the two edges are at different
        // heights at the same z.
        let banked = (0..600).any(|i| {
            let z = 6.0 + i as f64 * 0.1;
            (t.height(-4.0, z) - t.height(4.0, z)).abs() > 0.25
        });
        assert!(banked, "no cross-slope anywhere on the ramps");
    }

    #[test]
    fn jump_trenches_are_wider_than_a_stride() {
        let t = Terrain::new(Course::Jump, 1);
        let pits: Vec<_> = t.obstacles.iter().filter(|o| o.top < 0.0).collect();
        assert!(!pits.is_empty(), "JUMP has no trenches");
        for p in &pits {
            let w = p.z1 - p.z0;
            assert!(
                w > 1.45,
                "trench at z={:.2} is {:.2} m — still steppable",
                p.z0,
                w
            );
        }
        assert!(
            t.obstacles.iter().any(|o| o.top > 0.10 && o.top < 0.40),
            "JUMP has no platforms to land on"
        );
        for w in &t.waypoints {
            assert!(
                t.height(w[0], w[1]) > -0.12,
                "waypoint ({:.2}, {:.2}) is in a pit",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn a_disc_that_barely_overlaps_a_wall_is_obstructed() {
        // Six samples on the circumference sit at 0.866 r along Z, so a 5 cm
        // overlap used to be invisible. The chassis then walked into the block.
        let t = Terrain::new(Course::Slalom, 3);
        let wall = t.obstacles.iter().find(|o| o.top > 1.0).unwrap();
        let r = 0.90;
        let x = (wall.x0 + wall.x1) * 0.5;
        let z = wall.z0 - r + 0.05;
        assert!(
            t.height(x, z) < 0.05,
            "centre should still be in front of the wall"
        );
        assert!(
            t.height_disc(x, z, r) > 1.0,
            "disc overlaps a 1.8 m wall but height_disc missed it: {}",
            t.height_disc(x, z, r)
        );
        assert!(t.obstructed(x, z, r, 0.7));
    }

    #[test]
    fn push_xz_takes_a_point_out_of_a_wall_instead_of_leaving_it_inside() {
        let t = Terrain::new(Course::Slalom, 3);
        let wall = t.obstacles.iter().find(|o| o.top > 1.0).unwrap();
        let (cx, cz) = ((wall.x0 + wall.x1) * 0.5, (wall.z0 + wall.z1) * 0.5);
        assert!(t.height(cx, cz) > 1.0);
        let (x, z) = t.push_xz(cx, cz, 0.0, 0.62);
        assert!(
            t.height(x, z) < 0.62,
            "pushed to ({x:.3}, {z:.3}) which is still {:.2} m high",
            t.height(x, z)
        );
        // Nearest face of a 0.7 m wall is the front or back, not metres away.
        assert!((z - cz).abs() < 0.5);
    }

    #[test]
    fn a_leg_segment_through_a_wall_is_a_hit_and_one_in_front_is_not() {
        let t = Terrain::new(Course::Slalom, 3);
        let wall = t.obstacles.iter().find(|o| o.top > 1.0).unwrap();
        let cx = (wall.x0 + wall.x1) * 0.5;
        let through = t.segment_hits_wall(
            [cx, 0.4, wall.z0 - 0.4],
            [cx, 0.4, wall.z1 + 0.4],
            0.0,
            0.62,
        );
        assert!(through, "a tibia going through a slalom wall was missed");
        let clear = t.segment_hits_wall(
            [cx, 0.4, wall.z0 - 1.2],
            [cx, 0.4, wall.z0 - 0.4],
            0.0,
            0.62,
        );
        assert!(!clear, "open ground in front of the wall is blocked");
    }

    #[test]
    fn a_climbable_block_is_still_solid_to_leg_links() {
        let mut t = Terrain::new(Course::Flat, 1);
        t.push(-0.5, 0.5, 1.0, 2.0, 0.30, GRIP_STEP);
        t.rebuild_buckets();

        assert!(
            t.segment_hits_solid([-0.8, 0.15, 1.5], [0.8, 0.15, 1.5]),
            "a femur tunnels through a climbable step"
        );
        assert!(
            !t.segment_hits_solid([-0.8, 0.32, 1.5], [0.8, 0.32, 1.5]),
            "a link above the top face is reported inside"
        );
        assert!(
            !t.segment_hits_solid([0.0, 0.30, 0.8], [0.0, 0.30, 1.5]),
            "contact with the top face is penetration"
        );
    }
}
