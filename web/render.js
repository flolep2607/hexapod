/* Software 3-D renderer for the hexapod stage.
 *
 * Flat-shaded convex quads, back-face culled, near-plane clipped and drawn
 * back-to-front. That is all this scene needs, and it gives exactly the
 * crisp technical look the panel is going for — no WebGL, no dependencies.
 *
 * Shadows are the same faces projected onto the local surface along the light
 * and drawn as low-alpha quads in the depth pass so they sit on blocks.
 */

const COL = {
  shell: [0.58, 0.61, 0.62],
  shellBright: [0.82, 0.84, 0.84],
  shellDark: [0.075, 0.08, 0.085],
  carbon: [0.018, 0.021, 0.025],
  carbonEdge: [0.14, 0.15, 0.16],
  metal: [0.42, 0.44, 0.45],
  metalLight: [0.76, 0.77, 0.76],
  rubber: [0.018, 0.02, 0.023],
  cable: [0.012, 0.014, 0.017],
  pcb: [0.035, 0.20, 0.48],
  pcbLight: [0.12, 0.42, 0.68],
  wireRed: [0.72, 0.12, 0.055],
  wireGold: [0.92, 0.48, 0.08],
  wireBrown: [0.24, 0.055, 0.025],
  led: [0.16, 0.82, 0.28],
  accent: [0.898, 0.224, 0.114],
  accentDark: [0.55, 0.105, 0.055],
  block: [0.88, 0.88, 0.865],
  blockTop: [0.95, 0.95, 0.94],
  pit: [0.62, 0.62, 0.61],
  pitFloor: [0.42, 0.42, 0.42],
  wall: [0.78, 0.48, 0.38],
  wallTop: [0.90, 0.64, 0.52],
  step: [0.70, 0.74, 0.80],
  stepTop: [0.84, 0.87, 0.91],
  ramp: [0.62, 0.74, 0.64],
  rampTop: [0.76, 0.86, 0.78],
  rubble: [0.84, 0.76, 0.58],
  rubbleTop: [0.93, 0.87, 0.70],
  ice: [0.70, 0.84, 0.88],
  iceTop: [0.86, 0.93, 0.95],
  hit: [0.90, 0.22, 0.11],
  hitTop: [0.96, 0.48, 0.34],
};

const LIGHT = norm([-0.35, -1.0, -0.28]);
const NEAR = 0.35;
const VERTEX_FLOATS = 7;

/** Reusable, geometrically growing vertex storage. The stage emits many small
 * faces, so retaining this memory avoids rebuilding large JS arrays and then
 * copying them into fresh Float32Arrays every frame. */
class VertexStream {
  constructor(capacity = 16384) {
    this.data = new Float32Array(capacity);
    this.length = 0;
  }

  reset() {
    this.length = 0;
  }

  _reserve(extra) {
    const needed = this.length + extra;
    if (needed <= this.data.length) return;
    let capacity = this.data.length;
    while (capacity < needed) capacity *= 2;
    const grown = new Float32Array(capacity);
    grown.set(this.data);
    this.data = grown;
  }

  vertex(x, y, z, r, g, b, a) {
    this._reserve(VERTEX_FLOATS);
    const d = this.data;
    let i = this.length;
    d[i++] = x;
    d[i++] = y;
    d[i++] = z;
    d[i++] = r;
    d[i++] = g;
    d[i++] = b;
    d[i++] = a;
    this.length = i;
  }

  view() {
    return this.data.subarray(0, this.length);
  }
}

function norm(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

class Stage {
  constructor(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.az = -0.62;
    this.el = 0.38;
    this.dist = 8.0;
    this.target = [0, 0.7, 0];
    /* Wall-clock of the last draw, so the camera's follow is a time constant
     * rather than a per-frame fraction: a policy at 6 m/s outruns a fixed
     * fraction whenever the frame rate dips. */
    this.lastFollow = 0;
    this.faces = [];
    this.obstacles = new Float32Array(0);
    this.route = new Float32Array(0);
    this._terrainCache = [];
    this._hitOb = new Uint8Array(0);
    this._hitLegs = new Uint8Array(0);
    this._hitResult = { hitOb: this._hitOb, hitLegs: this._hitLegs, chassis: false };
    this._triangles = new VertexStream(131072);
    this._shadows = new VertexStream(8192);
    this._edges = new VertexStream(131072);
    this.dpr = 1;
    this.showSupport = true;
    this.showTargets = true;
    this.showRoute = true;
    this.view = "orbit";
    this.orbit = { az: -0.62, el: 0.38, dist: 8.0 };
    this.onView = null;
    this.contact = { blocked: false, chassis: false, legs: 0, obstacles: 0 };
    this._resizeDirty = true;
    this._cssWidth = 0;
    this._cssHeight = 0;
    if (typeof ResizeObserver !== "undefined") {
      this._resizeObserver = new ResizeObserver((entries) => {
        const rect = entries[entries.length - 1].contentRect;
        this._cssWidth = rect.width;
        this._cssHeight = rect.height;
        this._resizeDirty = true;
      });
      this._resizeObserver.observe(canvas);
    }
    this._bindInput();
  }

  _bindInput() {
    let dragging = false;
    let px = 0;
    let py = 0;
    const cv = this.cv;

    cv.addEventListener("pointerdown", (e) => {
      dragging = true;
      px = e.clientX;
      py = e.clientY;
      cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener("pointerup", (e) => {
      dragging = false;
      cv.releasePointerCapture(e.pointerId);
    });
    cv.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      if (this.view !== "orbit") this.setView("orbit");
      this.az -= (e.clientX - px) * 0.008;
      this.el = Math.max(-0.15, Math.min(1.35, this.el + (e.clientY - py) * 0.006));
      px = e.clientX;
      py = e.clientY;
    });
    cv.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.dist = Math.max(3.5, Math.min(34, this.dist * (1 + e.deltaY * 0.0012)));
      },
      { passive: false }
    );
  }

  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (!this._resizeDirty && dpr === this.dpr) return;
    if (!this._cssWidth || !this._cssHeight) {
      const r = this.cv.getBoundingClientRect();
      this._cssWidth = r.width;
      this._cssHeight = r.height;
    }
    const w = Math.max(1, Math.round(this._cssWidth * dpr));
    const h = Math.max(1, Math.round(this._cssHeight * dpr));
    if (this.cv.width !== w || this.cv.height !== h) {
      this.cv.width = w;
      this.cv.height = h;
    }
    this.dpr = dpr;
    this.w = w;
    this.h = h;
    this._resizeDirty = false;
  }

  setCourse(buf, route) {
    this.obstacles = buf;
    this.route = route || new Float32Array(0);
    this._cacheTerrain();
  }

  /** Obstacles do not move between course changes. Build their tiled faces
   * once instead of recreating every vertex and face on every rendered frame.
   * A second face list preserves the collision highlight without mutation. */
  _cacheTerrain() {
    const savedFaces = this.faces;
    const ob = this.obstacles;
    this._terrainCache = [];
    for (let i = 0; i + 4 < ob.length; i += 5) {
      const x0 = ob[i];
      const x1 = ob[i + 1];
      const z0 = ob[i + 2];
      const z1 = ob[i + 3];
      const top = ob[i + 4];
      const kind = this._kind(x0, x1, z0, z1, top);
      const build = (hit) => {
        this.faces = [];
        const [side, cap] = this._palette(kind, hit);
        if (top >= 0) this._box(x0, x1, 0, top, z0, z1, side, cap, false);
        else this._box(x0, x1, top, 0, z0, z1, COL.pit, COL.pitFloor, true);
        return this.faces;
      };
      const normal = build(false);
      const hit = kind === "pit" || kind === "ice" ? normal : build(true);
      this._terrainCache.push({ z0, z1, normal, hit });
    }
    this.faces = savedFaces;
  }

  /** `orbit` is free; `top` and `side` lock the camera until drag or a button. */
  setView(mode) {
    if (mode !== "orbit" && mode !== "top" && mode !== "side") return;
    const from = this.view;
    if (from === "orbit" && mode !== "orbit") {
      this.orbit = { az: this.az, el: this.el, dist: this.dist };
    }
    this.view = mode;
    if (mode === "orbit") {
      if (from !== "orbit") {
        this.az = this.orbit.az;
        this.el = this.orbit.el;
        this.dist = this.orbit.dist;
      }
    } else if (mode === "top") {
      this.dist = Math.max(this.dist, 16);
    } else {
      this.dist = Math.max(this.dist, 13);
    }
    if (this.onView) this.onView(mode);
  }

  /* ---------------------------------------------------------- camera */

  _camera() {
    if (this.view === "top") {
      this.el = 1.52;
      this.az = 0;
    } else if (this.view === "side") {
      this.el = 0.06;
      this.az = Math.PI / 2;
    }
    const ce = Math.cos(this.el);
    const dir = [ce * Math.sin(this.az), Math.sin(this.el), ce * Math.cos(this.az)];
    this.eye = [
      this.target[0] + dir[0] * this.dist,
      this.target[1] + dir[1] * this.dist,
      this.target[2] + dir[2] * this.dist,
    ];
    const fwd = norm(sub(this.target, this.eye));
    const right = norm(cross(fwd, [0, 1, 0]));
    const up = cross(right, fwd);
    this.basis = { fwd, right, up };
    this.focal = (0.5 * this.h) / Math.tan(0.5 * 0.86);
    this.cx = this.w * 0.5;
    this.cy = this.h * 0.5;
  }

  /** World point to view space: x right, y up, z forward depth. */
  _view(p) {
    const d = sub(p, this.eye);
    const b = this.basis;
    return [dot(d, b.right), dot(d, b.up), dot(d, b.fwd)];
  }

  _project(v) {
    const s = this.focal / v[2];
    return [this.cx + v[0] * s, this.cy - v[1] * s];
  }

  /* ----------------------------------------------------------- geometry */

  _quad(a, b, c, d, color) {
    this.faces.push({ p: [a, b, c, d], c: color });
  }

  _tri(a, b, c, color) {
    this.faces.push({ p: [a, b, c], c: color });
  }

  /** Faceted cylinder between two points. Six sides are enough to read as a
   * machined tube while keeping this dependency-free renderer inexpensive. */
  _segment(a, b, r, color, sides = 6) {
    const ax = sub(b, a);
    const len = Math.hypot(ax[0], ax[1], ax[2]);
    if (len < 1e-6) return;
    const axis = [ax[0] / len, ax[1] / len, ax[2] / len];
    let u = cross(axis, [0, 1, 0]);
    if (Math.hypot(u[0], u[1], u[2]) < 1e-4) u = cross(axis, [1, 0, 0]);
    u = norm(u);
    const v = cross(axis, u);

    const c = [];
    for (let i = 0; i < sides; i++) {
      const a0 = (i / sides) * Math.PI * 2 + Math.PI / sides;
      const su = Math.cos(a0);
      const sv = Math.sin(a0);
      c.push([
        [a[0] + (u[0] * su + v[0] * sv) * r, a[1] + (u[1] * su + v[1] * sv) * r, a[2] + (u[2] * su + v[2] * sv) * r],
        [b[0] + (u[0] * su + v[0] * sv) * r, b[1] + (u[1] * su + v[1] * sv) * r, b[2] + (u[2] * su + v[2] * sv) * r],
      ]);
    }
    for (let i = 0; i < sides; i++) {
      const j = (i + 1) % sides;
      this._quad(c[i][0], c[i][1], c[j][1], c[j][0], color);
    }
    // Caps keep the limbs from looking hollow at grazing angles.
    this.faces.push({ p: c.map((x) => x[0]), c: color });
    this.faces.push({ p: c.map((x) => x[1]).reverse(), c: color });
  }

  /** Low-poly ellipsoid for bearings, servo bosses, feet and optical parts. */
  _ellipsoid(center, radius, color, slices = 8, stacks = 4) {
    const rx = Array.isArray(radius) ? radius[0] : radius;
    const ry = Array.isArray(radius) ? radius[1] : radius;
    const rz = Array.isArray(radius) ? radius[2] : radius;
    const rings = [];
    for (let j = 0; j <= stacks; j++) {
      const lat = -Math.PI / 2 + (j / stacks) * Math.PI;
      const ring = [];
      for (let i = 0; i < slices; i++) {
        const lon = (i / slices) * Math.PI * 2;
        ring.push([
          center[0] + Math.cos(lat) * Math.cos(lon) * rx,
          center[1] + Math.sin(lat) * ry,
          center[2] + Math.cos(lat) * Math.sin(lon) * rz,
        ]);
      }
      rings.push(ring);
    }
    for (let j = 0; j < stacks; j++) {
      for (let i = 0; i < slices; i++) {
        const k = (i + 1) % slices;
        this._quad(rings[j][i], rings[j + 1][i], rings[j + 1][k], rings[j][k], color);
      }
    }
  }

  /** Oriented box expressed in chassis-local coordinates. */
  _localBox(xf, center, half, side, top = side) {
    const [cx, cy, cz] = center;
    const [hx, hy, hz] = half;
    const p = (x, y, z) => xf([cx + x * hx, cy + y * hy, cz + z * hz]);
    const nnn = p(-1, -1, -1);
    const pnn = p(1, -1, -1);
    const ppn = p(1, 1, -1);
    const npn = p(-1, 1, -1);
    const nnp = p(-1, -1, 1);
    const pnp = p(1, -1, 1);
    const ppp = p(1, 1, 1);
    const npp = p(-1, 1, 1);
    this._quad(nnn, pnn, ppn, npn, side);
    this._quad(pnp, nnp, npp, ppp, side);
    this._quad(nnp, nnn, npn, npp, side);
    this._quad(pnn, pnp, ppp, ppn, side);
    this._quad(npn, ppn, ppp, npp, top);
    this._quad(nnp, pnp, pnn, nnn, side);
  }

  /** Box in an arbitrary orthonormal basis. Used for rectangular servo cases,
   * where the output shaft must visibly define exactly one hinge axis. */
  _basisBox(center, axes, half, side, top = side) {
    const point = (sx, sy, sz) => [
      center[0] + axes[0][0] * half[0] * sx + axes[1][0] * half[1] * sy + axes[2][0] * half[2] * sz,
      center[1] + axes[0][1] * half[0] * sx + axes[1][1] * half[1] * sy + axes[2][1] * half[2] * sz,
      center[2] + axes[0][2] * half[0] * sx + axes[1][2] * half[1] * sy + axes[2][2] * half[2] * sz,
    ];
    const nnn = point(-1, -1, -1);
    const pnn = point(1, -1, -1);
    const ppn = point(1, 1, -1);
    const npn = point(-1, 1, -1);
    const nnp = point(-1, -1, 1);
    const pnp = point(1, -1, 1);
    const ppp = point(1, 1, 1);
    const npp = point(-1, 1, 1);
    this._quad(nnn, pnn, ppn, npn, side);
    this._quad(pnp, nnp, npp, ppp, side);
    this._quad(nnp, nnn, npn, npp, side);
    this._quad(pnn, pnp, ppp, ppn, side);
    this._quad(npn, ppn, ppp, npp, top);
    this._quad(nnp, pnp, pnn, nnn, side);
  }

  /** One revolute actuator: a bolted rectangular motor, continuous output
   * shaft and horn. There are deliberately no spherical joint shapes here. */
  _servo(p, axisHint, mountHint, r, hit) {
    const axis = norm(axisHint);
    const along = dot(mountHint, axis);
    let mount = norm([
      mountHint[0] - axis[0] * along,
      mountHint[1] - axis[1] * along,
      mountHint[2] - axis[2] * along,
    ]);
    if (Math.hypot(mount[0], mount[1], mount[2]) < 1e-4) {
      mount = norm(cross(axis, [0, 1, 0]));
    }
    const upright = norm(cross(mount, axis));
    const casing = hit ? COL.hit : COL.shellDark;
    this._basisBox(
      p,
      [axis, upright, mount],
      [r * 0.62, r * 0.82, r * 0.72],
      casing,
      hit ? COL.hitTop : COL.shell
    );

    // Thin mounting ears at both ends are the characteristic silhouette of
    // the off-the-shelf digital servos used by the reference machine.
    for (const d of [-r * 0.92, r * 0.92]) {
      const tab = [p[0] + upright[0] * d, p[1] + upright[1] * d, p[2] + upright[2] * d];
      this._basisBox(tab, [axis, upright, mount], [r * 0.82, r * 0.13, r * 0.30], casing, hit ? COL.hitTop : COL.shell);
    }

    const at = (d) => [p[0] + axis[0] * d, p[1] + axis[1] * d, p[2] + axis[2] * d];
    const face = (a, b, c) => [
      p[0] + axis[0] * a + upright[0] * b + mount[0] * c,
      p[1] + axis[1] * a + upright[1] * b + mount[1] * c,
      p[2] + axis[2] * a + upright[2] * b + mount[2] * c,
    ];
    // End caps and four visible fasteners give every joint a readable
    // mechanical assembly instead of a generic grey block.
    this._ellipsoid(at(-r * 0.90), r * 0.24, COL.metalLight, 8, 3);
    this._ellipsoid(at(r * 0.90), r * 0.24, COL.metalLight, 8, 3);
    for (const a of [-0.34, 0.34]) {
      for (const b of [-0.42, 0.42]) {
        const screw = face(a * r, b * r, r * 0.74);
        this._ellipsoid(screw, r * 0.075, hit ? COL.hitTop : COL.metalLight, 7, 3);
      }
    }
    this._segment(at(-r * 0.90), at(r * 0.90), r * 0.17, COL.metalLight, 8);
    this._segment(at(r * 0.58), at(r * 0.72), r * 0.48, hit ? COL.hitTop : COL.shell, 10);
  }

  /** Walls plus one horizontal face: the lid of a block, the floor of a pit.
   * Large faces are split so back-to-front sorting does not let a femur that
   * is *in front* of the near edge of a six-metre wall paint on top of it. */
  _box(x0, x1, y0, y1, z0, z1, side, cap, pit) {
    const P = (x, y, z) => [x, y, z];
    const capY = pit ? y0 : y1;
    this._faceGrid(P(x0, capY, z0), P(x1, capY, z0), P(x1, capY, z1), P(x0, capY, z1), cap);
    this._faceGrid(P(x0, y0, z0), P(x0, y1, z0), P(x0, y1, z1), P(x0, y0, z1), side);
    this._faceGrid(P(x1, y0, z1), P(x1, y1, z1), P(x1, y1, z0), P(x1, y0, z0), side);
    this._faceGrid(P(x1, y0, z0), P(x1, y1, z0), P(x0, y1, z0), P(x0, y0, z0), side);
    this._faceGrid(P(x0, y0, z1), P(x0, y1, z1), P(x1, y1, z1), P(x1, y0, z1), side);
  }

  /** Split an axis-aligned quad into tiles so painter's algorithm can order
   * them against nearby legs. */
  _faceGrid(a, b, _c, d, color) {
    const TILE = 0.7;
    const w = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const h = Math.hypot(d[0] - a[0], d[1] - a[1], d[2] - a[2]);
    const nw = Math.max(1, Math.ceil(w / TILE));
    const nh = Math.max(1, Math.ceil(h / TILE));
    if (nw === 1 && nh === 1) {
      this._quad(a, b, _c, d, color);
      return;
    }
    const pt = (u, v) => [
      a[0] + (b[0] - a[0]) * u + (d[0] - a[0]) * v,
      a[1] + (b[1] - a[1]) * u + (d[1] - a[1]) * v,
      a[2] + (b[2] - a[2]) * u + (d[2] - a[2]) * v,
    ];
    for (let i = 0; i < nw; i++) {
      for (let j = 0; j < nh; j++) {
        const u0 = i / nw;
        const u1 = (i + 1) / nw;
        const v0 = j / nh;
        const v1 = (j + 1) / nh;
        this._quad(pt(u0, v0), pt(u1, v0), pt(u1, v1), pt(u0, v1), color);
      }
    }
  }

  /** Open aluminium chassis inspired by hobby and research hexapods: two side
   * rails, crossmembers, visible control electronics and a serviceable wiring
   * harness. Nothing is hidden under a cosmetic shell. */
  _chassis(pos, yaw, pitch, roll, R, hit) {
    const H = 0.13;
    const sy = Math.sin(yaw);
    const cyw = Math.cos(yaw);
    const sp = Math.sin(pitch);
    const cp = Math.cos(pitch);
    const sr = Math.sin(roll);
    const cr = Math.cos(roll);

    const xf = (v) => {
      let y = v[1] * cp - v[2] * sp;
      let z = v[1] * sp + v[2] * cp;
      let x = v[0] * cr - y * sr;
      y = v[0] * sr + y * cr;
      return [pos[0] + x * cyw - z * sy, pos[1] + y, pos[2] + x * sy + z * cyw];
    };

    const frame = hit ? COL.hit : COL.shell;
    const frameTop = hit ? COL.hitTop : COL.shellBright;

    // Four bolted rails leave the underside and most of the electronics open.
    for (const sx of [-R * 0.55, R * 0.55]) {
      this._localBox(xf, [sx, 0, 0], [R * 0.085, H * 0.50, R * 0.68], frame, frameTop);
    }
    for (const sz of [-R * 0.59, R * 0.59]) {
      this._localBox(xf, [0, 0, sz], [R * 0.55, H * 0.50, R * 0.085], frame, frameTop);
    }
    this._segment(xf([-R * 0.52, 0, -R * 0.55]), xf([R * 0.52, 0, R * 0.55]), 0.024, hit ? COL.hitTop : COL.carbonEdge, 6);
    this._segment(xf([R * 0.52, 0, -R * 0.55]), xf([-R * 0.52, 0, R * 0.55]), 0.024, hit ? COL.hitTop : COL.carbonEdge, 6);

    // Main controller PCB, daughterboard, pin headers and exposed components.
    this._localBox(xf, [0, H * 0.78, R * 0.03], [R * 0.37, 0.026, R * 0.40], hit ? COL.hit : COL.pcb, hit ? COL.hitTop : COL.pcbLight);
    this._localBox(xf, [-R * 0.14, H * 1.12, R * 0.04], [R * 0.12, 0.035, R * 0.15], frame, frameTop);
    this._localBox(xf, [R * 0.17, H * 1.08, R * 0.13], [R * 0.105, 0.030, R * 0.10], COL.carbon, COL.carbonEdge);
    this._localBox(xf, [R * 0.19, H * 1.08, -R * 0.20], [R * 0.12, 0.032, R * 0.08], COL.carbon, COL.carbonEdge);

    // A row of pale servo headers makes the board read as functional at a
    // glance, even from the default orbit camera.
    for (let i = -3; i <= 3; i++) {
      this._localBox(
        xf,
        [i * R * 0.075, H * 1.11, R * 0.31],
        [R * 0.022, 0.030, R * 0.035],
        COL.metal,
        COL.metalLight
      );
    }

    // Small heat sink and power-distribution hardware.
    this._localBox(xf, [R * 0.25, H * 1.13, -R * 0.06], [R * 0.10, 0.035, R * 0.12], COL.carbon, COL.carbonEdge);
    for (let i = -2; i <= 2; i++) {
      this._localBox(xf, [R * 0.25 + i * R * 0.035, H * 1.42, -R * 0.06], [R * 0.010, 0.040, R * 0.10], COL.carbon, COL.shell);
    }
    for (const x of [-R * 0.29, -R * 0.23]) {
      this._ellipsoid(xf([x, H * 1.19, -R * 0.25]), 0.020, hit ? COL.hitTop : COL.led, 8, 3);
    }

    // Stainless standoffs and deck bolts remain visible around the perimeter.
    for (const [x, z] of [
      [-0.55, -0.59], [0.55, -0.59], [0.55, 0.59], [-0.55, 0.59],
      [-0.55, 0], [0.55, 0],
    ]) {
      const base = xf([x * R, H * 0.45, z * R]);
      const cap = xf([x * R, H * 0.92, z * R]);
      this._segment(base, cap, 0.022, COL.metalLight, 8);
    }

    // Routed signal and power harnesses are intentionally visible. Their
    // slight doglegs keep the three conductors individually legible.
    const wire = (points, color, radius = 0.009) => {
      for (let i = 0; i + 1 < points.length; i++) this._segment(xf(points[i]), xf(points[i + 1]), radius, color, 5);
    };
    for (const side of [-1, 1]) {
      wire([[0, H * 1.32, R * 0.25], [side * R * 0.18, H * 1.55, R * 0.18], [side * R * 0.51, H * 1.13, R * 0.39]], COL.wireGold);
      wire([[0, H * 1.25, R * 0.22], [side * R * 0.20, H * 1.45, R * 0.13], [side * R * 0.51, H * 1.05, R * 0.35]], COL.wireRed);
      wire([[0, H * 1.18, R * 0.19], [side * R * 0.22, H * 1.34, R * 0.08], [side * R * 0.51, H * 0.98, R * 0.31]], COL.wireBrown);
    }
  }

  _legs(J, stance, legs, hitLegs, movingLeg) {
    const at = (a, b, t) => [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
    const linkFrame = (a, b, height, spread, color, hit) => {
      const p0 = at(a, b, 0.09);
      const p1 = at(a, b, 0.91);
      const axis = norm(sub(p1, p0));
      let side = cross(axis, [0, 1, 0]);
      if (Math.hypot(side[0], side[1], side[2]) < 1e-4) side = [1, 0, 0];
      side = norm(side);
      let up = norm(cross(side, axis));
      if (up[1] < 0) up = [-up[0], -up[1], -up[2]];
      const length = Math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
      const mid = at(p0, p1, 0.5);
      const plateColor = hit ? COL.hit : color;
      const topColor = hit ? COL.hitTop : COL.shellBright;

      // A pair of thin perforated side plates carries each link, with steel
      // spacers tying the assembly together at both ends.
      for (const sign of [-1, 1]) {
        const center = [
          mid[0] + side[0] * spread * sign,
          mid[1] + side[1] * spread * sign,
          mid[2] + side[2] * spread * sign,
        ];
        this._basisBox(center, [side, up, axis], [0.017, height, length * 0.50], plateColor, topColor);
        for (const t of [0.26, 0.50, 0.74]) {
          const q0 = at(p0, p1, t);
          const normal = [side[0] * sign, side[1] * sign, side[2] * sign];
          const q = [
            q0[0] + normal[0] * (spread + 0.020),
            q0[1] + normal[1] * (spread + 0.020),
            q0[2] + normal[2] * (spread + 0.020),
          ];
          this._segment(
            [q[0] - normal[0] * 0.005, q[1] - normal[1] * 0.005, q[2] - normal[2] * 0.005],
            [q[0] + normal[0] * 0.006, q[1] + normal[1] * 0.006, q[2] + normal[2] * 0.006],
            height * 0.27,
            hit ? COL.hitTop : COL.cable,
            8
          );
        }
      }
      for (const p of [p0, p1]) {
        this._segment(
          [p[0] - side[0] * (spread + 0.035), p[1] - side[1] * (spread + 0.035), p[2] - side[2] * (spread + 0.035)],
          [p[0] + side[0] * (spread + 0.035), p[1] + side[1] * (spread + 0.035), p[2] + side[2] * (spread + 0.035)],
          0.018,
          COL.metalLight,
          8
        );
      }
      return side;
    };
    const wireBundle = (a, b, side) => {
      for (const [offset, color] of [[-0.014, COL.wireBrown], [0, COL.wireRed], [0.014, COL.wireGold]]) {
        const lift = (p, y, s) => [p[0] + side[0] * s, p[1] + y, p[2] + side[2] * s];
        const p0 = lift(at(a, b, 0.03), 0.075, offset);
        const pm = lift(at(a, b, 0.48), 0.105, offset);
        const p1 = lift(at(a, b, 0.96), 0.070, offset);
        this._segment(p0, pm, 0.007, color, 5);
        this._segment(pm, p1, 0.007, color, 5);
      }
    };

    for (let leg = 0; leg < legs; leg++) {
      const o = leg * 12;
      const hip = [J[o], J[o + 1], J[o + 2]];
      const knee = [J[o + 3], J[o + 4], J[o + 5]];
      const ankle = [J[o + 6], J[o + 7], J[o + 8]];
      const foot = [J[o + 9], J[o + 10], J[o + 11]];
      const moving = movingLeg === leg;
      const shell = hitLegs && hitLegs[leg] ? COL.hit : moving ? COL.accent : COL.shell;

      const coxa = sub(knee, hip);
      const femur = sub(ankle, knee);
      const tibia = sub(foot, ankle);
      let pitchAxis = cross(coxa, femur);
      if (Math.hypot(pitchAxis[0], pitchAxis[1], pitchAxis[2]) < 1e-4) {
        pitchAxis = cross(coxa, [0, 1, 0]);
      }

      // The coxa rotates about one vertical shaft. Femur and tibia use one
      // transverse pitch axis each; their parallel shafts make the 3-DOF
      // serial chain explicit without implying ball-and-socket joints.
      const hit = hitLegs && hitLegs[leg];
      const outward = norm(coxa);
      const shoulder = [hip[0] - outward[0] * 0.20, hip[1] - outward[1] * 0.20, hip[2] - outward[2] * 0.20];
      linkFrame(shoulder, hip, 0.060, 0.052, moving ? COL.accentDark : COL.shell, hit);
      const side0 = linkFrame(hip, knee, 0.065, 0.052, shell, hit);
      const side1 = linkFrame(knee, ankle, 0.058, 0.047, shell, hit);
      const side2 = linkFrame(ankle, foot, 0.038, 0.032, moving ? COL.accentDark : COL.shell, hit);

      // The servo boxes sit between each pair of plates, just as they do on
      // the physical reference build.
      this._servo(hip, [0, 1, 0], coxa, 0.13, hit);
      this._servo(knee, pitchAxis, femur, 0.12, hit);
      this._servo(ankle, pitchAxis, tibia, 0.088, hit);

      wireBundle(hip, knee, side0);
      wireBundle(knee, ankle, side1);
      wireBundle(ankle, foot, side2);

      // Open rectangular foot cage with triangular braces and four small
      // rubber contact pads, rather than a moulded sci-fi shoe.
      let toe = norm([tibia[0], 0, tibia[2]]);
      if (Math.hypot(toe[0], toe[2]) < 1e-4) toe = [0, 0, 1];
      const across = norm(cross([0, 1, 0], toe));
      const corner = (s, f, y = 0.020) => [
        foot[0] + across[0] * s * 0.09 + toe[0] * f * 0.072,
        foot[1] + y,
        foot[2] + across[2] * s * 0.09 + toe[2] * f * 0.072,
      ];
      const fl = corner(-1, 1);
      const fr = corner(1, 1);
      const rl = corner(-1, -1);
      const rr = corner(1, -1);
      const footColor = hit ? COL.hit : moving ? COL.accentDark : COL.shell;
      for (const [a, b] of [[fl, fr], [fr, rr], [rr, rl], [rl, fl]]) this._segment(a, b, 0.014, footColor, 6);
      const topL = corner(-0.56, -0.08, 0.10);
      const topR = corner(0.56, -0.08, 0.10);
      for (const [a, b] of [[fl, topL], [rl, topL], [fr, topR], [rr, topR], [topL, topR]]) this._segment(a, b, 0.012, footColor, 6);
      for (const p of [fl, fr, rl, rr]) this._ellipsoid([p[0], p[1] - 0.010, p[2]], [0.026, 0.014, 0.026], COL.rubber, 8, 2);

      if (stance[leg] > 0.5) this._ellipsoid([foot[0], foot[1] + 0.010, foot[2]], [0.068, 0.008, 0.052], COL.accentDark, 8, 2);
      if (moving) this._ellipsoid([foot[0], foot[1] + 0.045, foot[2]], [0.065, 0.065, 0.065], COL.accent, 10, 3);
    }
  }

  _kind(x0, x1, z0, z1, top) {
    if (top < 0) return "pit";
    if (top <= 0.02) return "ice";
    const spanX = x1 - x0;
    const spanZ = z1 - z0;
    // Ramp slabs are 30 cm deep. Classify them before the wall-height cut so a
    // banked crown taller than a metre stays a ramp, not a slalom wall.
    if (spanZ < 0.36 && spanX > 1.2) return "ramp";
    if (top > 1.55) return "wall";
    if (spanX > 8) return "step";
    return "rubble";
  }

  _palette(kind, hit) {
    if (hit && kind !== "pit" && kind !== "ice") return [COL.hit, COL.hitTop];
    switch (kind) {
      case "wall":
        return [COL.wall, COL.wallTop];
      case "step":
        return [COL.step, COL.stepTop];
      case "ramp":
        return [COL.ramp, COL.rampTop];
      case "rubble":
        return [COL.rubble, COL.rubbleTop];
      case "ice":
        return [COL.ice, COL.iceTop];
      default:
        return [COL.pit, COL.pitFloor];
    }
  }

  _segmentHitsBox(a, b, x0, x1, y0, y1, z0, z1) {
    const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const lo = [x0, y0, z0];
    const hi = [x1, y1, z1];
    let t0 = 0;
    let t1 = 1;
    for (let axis = 0; axis < 3; axis++) {
      if (Math.abs(d[axis]) < 1e-10) {
        if (a[axis] < lo[axis] || a[axis] > hi[axis]) return false;
        continue;
      }
      let q0 = (lo[axis] - a[axis]) / d[axis];
      let q1 = (hi[axis] - a[axis]) / d[axis];
      if (q0 > q1) [q0, q1] = [q1, q0];
      t0 = Math.max(t0, q0);
      t1 = Math.min(t1, q1);
      if (t0 > t1) return false;
    }
    return true;
  }

  /** Which obstacles the chassis disc or any part of a leg intersects. */
  _hits(pos, bodyR, joints, legs) {
    const ob = this.obstacles;
    const n = Math.floor(ob.length / 5);
    if (this._hitOb.length !== n) this._hitOb = new Uint8Array(n);
    else this._hitOb.fill(0);
    if (this._hitLegs.length !== legs) this._hitLegs = new Uint8Array(legs);
    else this._hitLegs.fill(0);
    const hitOb = this._hitOb;
    const hitLegs = this._hitLegs;
    const r = bodyR || 0.95;
    let chassis = false;
    for (let i = 0; i < n; i++) {
      const x0 = ob[i * 5];
      const x1 = ob[i * 5 + 1];
      const z0 = ob[i * 5 + 2];
      const z1 = ob[i * 5 + 3];
      const top = ob[i * 5 + 4];
      if (top <= 0.02) continue;
      const qx = Math.max(x0, Math.min(x1, pos[0]));
      const qz = Math.max(z0, Math.min(z1, pos[2]));
      const dx = pos[0] - qx;
      const dz = pos[2] - qz;
      if (dx * dx + dz * dz <= r * r && pos[1] - 0.18 < top) {
        hitOb[i] = 1;
        chassis = true;
      }
      for (let leg = 0; leg < legs; leg++) {
        const o = leg * 12;
        for (let k = 0; k < 3; k++) {
          const a = [joints[o + k * 3], joints[o + k * 3 + 1], joints[o + k * 3 + 2]];
          const b = [joints[o + (k + 1) * 3], joints[o + (k + 1) * 3 + 1], joints[o + (k + 1) * 3 + 2]];
          if (this._segmentHitsBox(a, b, x0 + 0.02, x1 - 0.02, 0.02, top - 0.02, z0 + 0.02, z1 - 0.02)) {
            hitOb[i] = 1;
            hitLegs[leg] = 1;
            break;
          }
        }
      }
    }
    this._hitResult.hitOb = hitOb;
    this._hitResult.hitLegs = hitLegs;
    this._hitResult.chassis = chassis;
    return this._hitResult;
  }

  _terrain(zc, hitOb) {
    for (let i = 0; i < this._terrainCache.length; i++) {
      const entry = this._terrainCache[i];
      if (entry.z1 < zc - 14 || entry.z0 > zc + 30) continue;
      const faces = hitOb && hitOb[i] ? entry.hit : entry.normal;
      for (let j = 0; j < faces.length; j++) this.faces.push(faces[j]);
    }
  }

  /* ------------------------------------------------------------- drawing */

  _clipNear(poly) {
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      const ain = a[2] > NEAR;
      const bin = b[2] > NEAR;
      if (ain) out.push(a);
      if (ain !== bin) {
        const t = (NEAR - a[2]) / (b[2] - a[2]);
        out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, NEAR]);
      }
    }
    return out;
  }

  _line(a, b, style, width, dash) {
    const va = this._view(a);
    const vb = this._view(b);
    if (va[2] <= NEAR && vb[2] <= NEAR) return;
    let pa = va;
    let pb = vb;
    if (va[2] <= NEAR || vb[2] <= NEAR) {
      const t = (NEAR - va[2]) / (vb[2] - va[2]);
      const mid = [va[0] + (vb[0] - va[0]) * t, va[1] + (vb[1] - va[1]) * t, NEAR];
      if (va[2] <= NEAR) pa = mid;
      else pb = mid;
    }
    const A = this._project(pa);
    const B = this._project(pb);
    const c = this.ctx;
    c.save();
    c.strokeStyle = style;
    c.lineWidth = (width || 1) * this.dpr;
    if (dash) c.setLineDash(dash.map((d) => d * this.dpr));
    c.beginPath();
    c.moveTo(A[0], A[1]);
    c.lineTo(B[0], B[1]);
    c.stroke();
    c.restore();
  }

  _ringXZ(cx, cz, y, r, style, width, dash) {
    this._ringAround([cx, y, cz], [1, 0, 0], [0, 0, 1], r, style, width, dash);
  }

  _ringAround(c, u, v, r, style, width, dash) {
    const ctx = this.ctx;
    const pts = [];
    for (let i = 0; i <= 20; i++) {
      const a = (i / 20) * Math.PI * 2;
      const ca = Math.cos(a) * r;
      const sa = Math.sin(a) * r;
      const q = this._view([
        c[0] + u[0] * ca + v[0] * sa,
        c[1] + u[1] * ca + v[1] * sa,
        c[2] + u[2] * ca + v[2] * sa,
      ]);
      if (q[2] <= NEAR) return;
      pts.push(this._project(q));
    }
    ctx.save();
    ctx.strokeStyle = style;
    ctx.lineWidth = (width || 1) * this.dpr;
    if (dash) ctx.setLineDash(dash.map((d) => d * this.dpr));
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (const p of pts.slice(1)) ctx.lineTo(p[0], p[1]);
    ctx.stroke();
    ctx.restore();
  }

  _grid(zc) {
    const style = "rgba(20,20,22,0.07)";
    const x0 = -9;
    const x1 = 9;
    const za = Math.floor(zc - 12);
    const zb = Math.ceil(zc + 26);
    for (let x = Math.ceil(x0); x <= x1; x += 1) {
      this._line([x, 0, za], [x, 0, zb], x % 5 === 0 ? "rgba(20,20,22,0.13)" : style, 1);
    }
    for (let z = za; z <= zb; z += 1) {
      this._line([x0, 0, z], [x1, 0, z], z % 5 === 0 ? "rgba(20,20,22,0.13)" : style, 1);
    }
  }

  /* The two walls are invisible, which is the point: nothing is drawn out
   * there. What is drawn is the line on the ground where they stand, and — as
   * the chassis comes within a metre — a few faint uprights, so a machine
   * pinned against one is not a mystery. */
  _walls(px, zc, wallX) {
    const za = Math.floor(zc - 12);
    const zb = Math.ceil(zc + 26);
    for (const x of [-wallX, wallX]) {
      // Only the wall you are actually close to lights up.
      const near = Math.max(0, 1 - Math.abs(px - x) / 1.6);
      this._line(
        [x, 0.01, za],
        [x, 0.01, zb],
        `rgba(229,57,29,${(0.18 + 0.5 * near).toFixed(3)})`,
        1.2,
        [7, 5]
      );
      if (near <= 0.02) continue;
      for (let z = za + (((za % 2) + 2) % 2); z <= zb; z += 2) {
        this._line([x, 0, z], [x, 1.1, z], `rgba(229,57,29,${(0.28 * near).toFixed(3)})`, 1, [3, 6]);
      }
    }
  }

  /** The route: where the machine is being asked to go, and in what order. */
  _routeMarks(zc, wpIndex) {
    const r = this.route;
    for (let i = 0; i * 2 + 1 < r.length; i++) {
      const x = r[i * 2];
      const z = r[i * 2 + 1];
      if (z < zc - 8 || z > zc + 30) continue;
      const done = i < wpIndex;
      const live = i === wpIndex;
      const col = done ? "rgba(20,20,22,0.14)" : live ? "#e5391d" : "rgba(20,20,22,0.30)";
      this._ringXZ(x, z, 0.03, live ? 1.6 : 1.0, col, live ? 1.8 : 1.1, live ? null : [4, 5]);
      if (live) {
        // A post, so the target is findable when it is off past a wall.
        this._line([x, 0, z], [x, 1.6, z], "rgba(229,57,29,0.55)", 1.4);
      }
      // The leg of the route into this waypoint.
      if (i > 0) {
        const pz = r[i * 2 - 1];
        if (pz > zc - 12) {
          this._line(
            [r[i * 2 - 2], 0.03, pz],
            [x, 0.03, z],
            done ? "rgba(20,20,22,0.10)" : "rgba(20,20,22,0.22)",
            1,
            [3, 5]
          );
        }
      }
    }
  }

  _shade(face) {
    const p = face.p;
    let n = norm(cross(sub(p[1], p[0]), sub(p[2], p[0])));
    // Orient toward the camera so a face is lit the same whichever way it
    // happens to be wound.
    if (dot(n, sub(p[0], this.eye)) > 0) n = [-n[0], -n[1], -n[2]];
    const lam = Math.max(0, -dot(n, LIGHT));
    const i = 0.56 + 0.44 * lam;
    const c = face.c;
    const r = Math.round(Math.min(255, c[0] * i * 255));
    const g = Math.round(Math.min(255, c[1] * i * 255));
    const b = Math.round(Math.min(255, c[2] * i * 255));
    return (r << 16) | (g << 8) | b;
  }

  _geometryContext() {
    if (this.geo) return this.geo;
    const canvas = document.createElement("canvas");
    canvas.className = "geometry-layer";
    this.cv.insertAdjacentElement("afterend", canvas);
    const gl = canvas.getContext("webgl", {
      alpha: true,
      antialias: true,
      depth: true,
      premultipliedAlpha: true,
    });
    if (!gl) return null;

    const compile = (kind, source) => {
      const shader = gl.createShader(kind);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error("geometry shader:", gl.getShaderInfoLog(shader));
        return null;
      }
      return shader;
    };
    const vert = compile(
      gl.VERTEX_SHADER,
      "attribute vec3 p; attribute vec4 c; varying vec4 color; void main(){gl_Position=vec4(p,1.0);color=c;}"
    );
    const frag = compile(
      gl.FRAGMENT_SHADER,
      "precision mediump float; varying vec4 color; void main(){gl_FragColor=color;}"
    );
    if (!vert || !frag) return null;
    const program = gl.createProgram();
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("geometry program:", gl.getProgramInfoLog(program));
      return null;
    }
    this.geo = {
      canvas,
      gl,
      program,
      p: gl.getAttribLocation(program, "p"),
      c: gl.getAttribLocation(program, "c"),
      vb: gl.createBuffer(),
      vertexCapacity: 0,
    };
    return this.geo;
  }

  /** Draw solid geometry through the GPU depth buffer. Average-depth painter
   * sorting was visibly wrong wherever a long leg crossed a wall or detailed
   * parts intersected. A real per-pixel depth test makes visibility exact and
   * retains antialiasing without the cost of a JavaScript raster loop. */
  _paintFaces() {
    const ctx = this.ctx;
    const items = [];
    const shadows = [];

    for (const f of this.faces) {
      const vs = f.p.map((p) => this._view(p));
      let visible = false;
      for (const v of vs) {
        if (v[2] > NEAR) visible = true;
      }
      if (!visible) continue;
      const poly = this._clipNear(vs);
      if (poly.length < 3) continue;
      if (f.a) shadows.push({ poly, color: 0x141418, a: f.a });
      else items.push({ poly, color: this._shade(f) });
    }

    const geo = this._geometryContext();
    if (!geo) {
      // Old browsers without WebGL retain a conservative triangle painter.
      const tris = [];
      const push = (it, a) => {
        for (let i = 1; i + 1 < it.poly.length; i++) {
          const poly = [it.poly[0], it.poly[i], it.poly[i + 1]];
          tris.push({ poly, color: it.color, a, z: (poly[0][2] + poly[1][2] + poly[2][2]) / 3 });
        }
      };
      for (const it of items) push(it, 1);
      for (const it of shadows) push(it, it.a);
      tris.sort((a, b) => b.z - a.z);
      for (const it of tris) {
        const pts = it.poly.map((v) => this._project(v));
        const r = (it.color >> 16) & 255;
        const g = (it.color >> 8) & 255;
        const b = it.color & 255;
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        ctx.lineTo(pts[1][0], pts[1][1]);
        ctx.lineTo(pts[2][0], pts[2][1]);
        ctx.closePath();
        ctx.fillStyle =
          it.a < 1
            ? `rgba(${r},${g},${b},${it.a})`
            : `rgb(${r},${g},${b})`;
        ctx.fill();
      }
      return;
    }

    const triangles = this._triangles;
    const shadowsBuf = this._shadows;
    const edges = this._edges;
    triangles.reset();
    shadowsBuf.reset();
    edges.reset();
    const vertex = (v, out, rgb, alpha = 1, edge = false) => {
      const scale = this.focal / v[2];
      const px = this.cx + v[0] * scale;
      const py = this.cy - v[1] * scale;
      out.vertex(
        (px / this.w) * 2 - 1,
        1 - (py / this.h) * 2,
        1 - (2 * NEAR) / v[2] - (edge ? 0.00035 : 0),
        (((rgb >> 16) & 255) / 255) * alpha,
        (((rgb >> 8) & 255) / 255) * alpha,
        ((rgb & 255) / 255) * alpha,
        alpha
      );
    };
    const fan = (it, out, alpha) => {
      for (let i = 1; i + 1 < it.poly.length; i++) {
        vertex(it.poly[0], out, it.color, alpha);
        vertex(it.poly[i], out, it.color, alpha);
        vertex(it.poly[i + 1], out, it.color, alpha);
      }
    };
    for (const it of items) {
      fan(it, triangles, 1);
      for (let i = 0; i < it.poly.length; i++) {
        const j = (i + 1) % it.poly.length;
        vertex(it.poly[i], edges, 0x18181a, 0.34, true);
        vertex(it.poly[j], edges, 0x18181a, 0.34, true);
      }
    }
    for (const it of shadows) fan(it, shadowsBuf, it.a);

    const { canvas, gl, program } = geo;
    if (canvas.width !== this.w || canvas.height !== this.h) {
      canvas.width = this.w;
      canvas.height = this.h;
    }
    gl.viewport(0, 0, this.w, this.h);
    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(program);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.disable(gl.BLEND);

    const triangleCount = triangles.length / VERTEX_FLOATS;
    const shadowStart = triangleCount;
    const shadowCount = shadowsBuf.length / VERTEX_FLOATS;
    const edgeStart = shadowStart + shadowCount;
    const edgeCount = edges.length / VERTEX_FLOATS;
    const totalFloats = triangles.length + shadowsBuf.length + edges.length;
    gl.bindBuffer(gl.ARRAY_BUFFER, geo.vb);
    if (totalFloats > geo.vertexCapacity) {
      geo.vertexCapacity = Math.max(totalFloats, geo.vertexCapacity ? geo.vertexCapacity * 2 : 16384);
      gl.bufferData(gl.ARRAY_BUFFER, geo.vertexCapacity * Float32Array.BYTES_PER_ELEMENT, gl.DYNAMIC_DRAW);
    }
    let floatOffset = 0;
    if (triangles.length) {
      gl.bufferSubData(gl.ARRAY_BUFFER, floatOffset * Float32Array.BYTES_PER_ELEMENT, triangles.view());
      floatOffset += triangles.length;
    }
    if (shadowsBuf.length) {
      gl.bufferSubData(gl.ARRAY_BUFFER, floatOffset * Float32Array.BYTES_PER_ELEMENT, shadowsBuf.view());
      floatOffset += shadowsBuf.length;
    }
    if (edges.length) gl.bufferSubData(gl.ARRAY_BUFFER, floatOffset * Float32Array.BYTES_PER_ELEMENT, edges.view());
    const stride = VERTEX_FLOATS * Float32Array.BYTES_PER_ELEMENT;
    gl.enableVertexAttribArray(geo.p);
    gl.vertexAttribPointer(geo.p, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(geo.c);
    gl.vertexAttribPointer(geo.c, 4, gl.FLOAT, false, stride, 3 * Float32Array.BYTES_PER_ELEMENT);
    gl.depthMask(true);
    gl.drawArrays(gl.TRIANGLES, 0, triangleCount);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    if (shadowCount) gl.drawArrays(gl.TRIANGLES, shadowStart, shadowCount);
    if (edgeCount) gl.drawArrays(gl.LINES, edgeStart, edgeCount);
    gl.depthMask(true);
  }

  /** Light-ray hit on the highest obstacle top at that xz, else the floor. */
  _shadowP(p) {
    const ly = LIGHT[1];
    let tHit = -p[1] / ly;
    let yHit = 0;
    const ob = this.obstacles;
    for (let i = 0; i + 4 < ob.length; i += 5) {
      const top = ob[i + 4];
      if (top <= 0) continue;
      const t = (top - p[1]) / ly;
      if (t <= 0 || t >= tHit) continue;
      const x = p[0] + LIGHT[0] * t;
      const z = p[2] + LIGHT[2] * t;
      if (x >= ob[i] && x <= ob[i + 1] && z >= ob[i + 2] && z <= ob[i + 3]) {
        tHit = t;
        yHit = top;
      }
    }
    return [p[0] + LIGHT[0] * tHit, yHit + 0.004, p[2] + LIGHT[2] * tHit];
  }

  /** A few soft contact pools read much better than projecting every tiny
   * robot polygon, which made overlapping shadows turn into a noisy stain. */
  _paintShadows(pos, yaw, bodyR, joints, legs) {
    const ellipse = (center, rx, rz, alpha, y) => {
      const poly = [];
      const cs = Math.cos(yaw);
      const sn = Math.sin(yaw);
      for (let i = 0; i < 16; i++) {
        const a = (i / 16) * Math.PI * 2;
        const x = Math.cos(a) * rx;
        const z = Math.sin(a) * rz;
        poly.push(this._shadowP([center[0] + x * cs - z * sn, y, center[2] + x * sn + z * cs]));
      }
      this.faces.push({ p: poly, a: alpha });
    };

    // Broad, low-opacity chassis shadow, layered to give it a soft edge.
    ellipse(pos, bodyR * 1.30, bodyR * 1.45, 0.018, pos[1]);
    ellipse(pos, bodyR * 1.03, bodyR * 1.17, 0.026, pos[1]);

    // Small contact pools retain the legged silhouette without darkening the
    // entire area beneath a swinging leg.
    for (let leg = 0; leg < legs; leg++) {
      const o = leg * 12 + 9;
      const foot = [joints[o], joints[o + 1], joints[o + 2]];
      const contact = Math.max(0.15, 1 - Math.min(1, Math.max(0, foot[1]) * 2.5));
      ellipse(foot, 0.19, 0.16, 0.018 * contact, Math.max(foot[1], 0.08));
      ellipse(foot, 0.12, 0.10, 0.026 * contact, Math.max(foot[1], 0.08));
    }
  }

  /* ---------------------------------------------------------------- frame */

  draw(t, L) {
    this.resize();
    const ctx = this.ctx;
    const pos = [t[L.T_POS], t[L.T_POS + 1], t[L.T_POS + 2]];

    // Follow the robot without snapping — except when it is not where it was
    // at all. A respawn at the start line, or a course change, teleports the
    // body tens of metres, and easing over that leaves the machine off screen
    // for as long as it takes to catch up.
    const now = performance.now();
    const dt = this.lastFollow ? Math.min(0.5, (now - this.lastFollow) / 1000) : 1 / 60;
    this.lastFollow = now;
    const jumped = Math.hypot(pos[0] - this.target[0], pos[2] - this.target[2]) > 4;
    const k = jumped ? 1 : 1 - Math.exp(-dt / 0.12);
    this.target[0] += (pos[0] - this.target[0]) * k;
    this.target[1] += (pos[1] * 0.75 - this.target[1]) * k;
    this.target[2] += (pos[2] - this.target[2]) * k;
    this._camera();

    ctx.fillStyle = "#fbfbfa";
    ctx.fillRect(0, 0, this.w, this.h);

    this._grid(pos[2]);
    this._walls(pos[0], pos[2], t[L.T_WALL_X] || 5);
    if (this.showRoute) this._routeMarks(pos[2], Math.round(t[L.T_WP_I]));

    this.faces.length = 0;
    const legs = Math.round(t[L.T_LEGS]) || 6;
    const bodyR = t[L.T_BODY_R] || 0.95;
    const joints = t.subarray(L.T_JOINTS, L.T_JOINTS + legs * 12);
    const hits = this._hits(pos, bodyR, joints, legs);
    const blocked = t[L.T_BLOCKED] > 0.5;
    let nLegsHit = 0;
    for (let i = 0; i < hits.hitLegs.length; i++) if (hits.hitLegs[i]) nLegsHit++;
    let nOb = 0;
    for (let i = 0; i < hits.hitOb.length; i++) if (hits.hitOb[i]) nOb++;
    this.contact.blocked = blocked;
    this.contact.chassis = hits.chassis;
    this.contact.legs = nLegsHit;
    this.contact.obstacles = nOb;
    this._terrain(pos[2], hits.hitOb);
    this._chassis(pos, t[L.T_YAW], t[L.T_PITCH], t[L.T_ROLL], bodyR, blocked || hits.chassis);
    const swinging = L.T_MOVE_PHASE != null && t[L.T_MOVE_PHASE] >= 1 && t[L.T_MOVE_PHASE] <= 3;
    const movingLeg = swinging ? Math.round(t[L.T_MOVE_LEG]) : -1;
    this._legs(joints, t.subarray(L.T_STANCE, L.T_STANCE + legs), legs, hits.hitLegs, movingLeg);

    const bad = t[L.T_MARGIN] < 0.05;
    const com =
      L.T_COM3 != null
        ? [t[L.T_COM3], t[L.T_COM3 + 1], t[L.T_COM3 + 2]]
        : [pos[0] + t[L.T_COM], pos[1], pos[2] + t[L.T_COM + 1]];
    if (this.showSupport) {
      this._ellipsoid(com, 0.09, bad ? COL.hit : COL.accent, 10, 4);
    }

    if (this.view !== "top") this._paintShadows(pos, t[L.T_YAW], bodyR, joints, legs);
    this._paintFaces();

    // Support polygon on the ground, mass-weighted CoM in 3-D with its plumb.
    if (this.showSupport) {
      const n = Math.round(t[L.T_HULL_N]);
      if (n >= 3) {
        for (let i = 0; i < n; i++) {
          const a = [t[L.T_HULL + i * 2], 0.02, t[L.T_HULL + i * 2 + 1]];
          const j = (i + 1) % n;
          const b = [t[L.T_HULL + j * 2], 0.02, t[L.T_HULL + j * 2 + 1]];
          this._line(a, b, "rgba(229,57,29,0.5)", 1.2, [6, 4]);
        }
      }
      const [cx, cy, cz] = com;
      const gy = 0.02;
      const col = bad ? "#e5391d" : "rgba(20,20,22,0.6)";
      const s = 0.18;
      this._line([cx - s, cy, cz], [cx + s, cy, cz], col, 1.6);
      this._line([cx, cy - s, cz], [cx, cy + s, cz], col, 1.6);
      this._line([cx, cy, cz - s], [cx, cy, cz + s], col, 1.6);
      this._ringAround(com, [1, 0, 0], [0, 1, 0], 0.13, col, 1.2);
      this._ringAround(com, [1, 0, 0], [0, 0, 1], 0.13, col, 1.2);
      this._ringAround(com, [0, 1, 0], [0, 0, 1], 0.13, col, 1.2);
      this._line([cx, cy, cz], [cx, gy, cz], "rgba(20,20,22,0.22)", 1, [3, 4]);
      this._line([cx - 0.22, gy, cz], [cx + 0.22, gy, cz], col, 1.5);
      this._line([cx, gy, cz - 0.22], [cx, gy, cz + 0.22], col, 1.5);
      this._ringXZ(cx, cz, gy, 0.13, col, 1.2);
    }

    // Foot contacts and where each swinging leg intends to land.
    for (let leg = 0; leg < legs; leg++) {
      const o = L.T_JOINTS + leg * 12 + 9;
      const fx = t[o];
      const fz = t[o + 2];
      const down = t[L.T_STANCE + leg] > 0.5;
      if (down) {
        const load = t[L.T_LOAD + leg];
        this._ringXZ(fx, fz, t[o + 1] + 0.01, 0.14 + load * 0.16, "#e5391d", 1.6);
      }
      if (this.showTargets && !down) {
        const td = L.T_TD + leg * 3;
        this._ringXZ(t[td], t[td + 2], t[td + 1] + 0.01, 0.11, "rgba(20,20,22,0.32)", 1, [3, 3]);
        this._line(
          [fx, t[o + 1], fz],
          [t[td], t[td + 1], t[td + 2]],
          "rgba(20,20,22,0.16)",
          1,
          [2, 4]
        );
      }
      if (hits.hitLegs[leg]) {
        this._ringXZ(fx, fz, t[o + 1] + 0.04, 0.22, "#e5391d", 2.2);
      }
    }

    if (L.T_ORIGIN != null && t[L.T_PLANT] > 0.5) {
      const moving = Math.round(t[L.T_MOVE_LEG]);
      for (let leg = 0; leg < legs; leg++) {
        const ox = t[L.T_ORIGIN + leg * 3];
        const oy = t[L.T_ORIGIN + leg * 3 + 1] + 0.02;
        const oz = t[L.T_ORIGIN + leg * 3 + 2];
        const col = leg === moving ? "rgba(229,57,29,0.55)" : "rgba(20,20,22,0.55)";
        this._line([ox - 0.14, oy, oz - 0.14], [ox + 0.14, oy, oz + 0.14], col, 2);
        this._line([ox - 0.14, oy, oz + 0.14], [ox + 0.14, oy, oz - 0.14], col, 2);
        const fo = L.T_JOINTS + leg * 12 + 9;
        const drift = Math.hypot(t[fo] - ox, t[fo + 2] - oz);
        if (drift > 0.04) {
          this._line([ox, oy, oz], [t[fo], t[fo + 1], t[fo + 2]], "rgba(229,57,29,0.45)", 1.2, [4, 3]);
        }
      }
      const dx = t[L.T_DEST];
      const dy = t[L.T_DEST + 1];
      const dz = t[L.T_DEST + 2];
      this._ringXZ(dx, dz, dy + 0.02, 0.16, "#e5391d", 1.8, [4, 3]);
      const fo = L.T_JOINTS + moving * 12 + 9;
      this._line([t[fo], t[fo + 1], t[fo + 2]], [dx, dy, dz], "rgba(229,57,29,0.35)", 1.2, [3, 4]);
      this._line([t[fo], t[fo + 1], t[fo + 2]], [t[fo], 0.02, t[fo + 2]], "rgba(229,57,29,0.7)", 1.8);
    }

    // Ground footprint of every obstacle the chassis or a link is inside.
    const ob = this.obstacles;
    for (let i = 0; i < hits.hitOb.length; i++) {
      if (!hits.hitOb[i]) continue;
      const x0 = ob[i * 5];
      const x1 = ob[i * 5 + 1];
      const z0 = ob[i * 5 + 2];
      const z1 = ob[i * 5 + 3];
      const y = Math.max(0.04, ob[i * 5 + 4] + 0.02);
      this._line([x0, y, z0], [x1, y, z0], "#e5391d", 2);
      this._line([x1, y, z0], [x1, y, z1], "#e5391d", 2);
      this._line([x1, y, z1], [x0, y, z1], "#e5391d", 2);
      this._line([x0, y, z1], [x0, y, z0], "#e5391d", 2);
    }
  }
}

window.Stage = Stage;
