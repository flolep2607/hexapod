/* Software 3-D renderer for the hexapod stage.
 *
 * Flat-shaded convex quads, back-face culled, near-plane clipped and drawn
 * back-to-front. That is all this scene needs, and it gives exactly the
 * crisp technical look the panel is going for — no WebGL, no dependencies.
 *
 * Shadows are the same faces projected onto y=0 along the light direction and
 * filled at low alpha; overlapping limbs accumulate into a soft blob for free.
 */

const COL = {
  shell: [0.925, 0.925, 0.915],
  shellDark: [0.84, 0.84, 0.83],
  accent: [0.898, 0.224, 0.114],
  joint: [0.32, 0.32, 0.33],
  block: [0.88, 0.88, 0.865],
  blockTop: [0.95, 0.95, 0.94],
  pit: [0.62, 0.62, 0.61],
  pitFloor: [0.42, 0.42, 0.42],
};

const LIGHT = norm([-0.35, -1.0, -0.28]);
const NEAR = 0.35;

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
    this.faces = [];
    this.obstacles = new Float32Array(0);
    this.route = new Float32Array(0);
    this.dpr = 1;
    this.showSupport = true;
    this.showTargets = true;
    this.showRoute = true;
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
    const r = this.cv.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(r.width * dpr));
    const h = Math.max(1, Math.round(r.height * dpr));
    if (this.cv.width !== w || this.cv.height !== h) {
      this.cv.width = w;
      this.cv.height = h;
    }
    this.dpr = dpr;
    this.w = w;
    this.h = h;
  }

  setCourse(buf, route) {
    this.obstacles = buf;
    this.route = route || new Float32Array(0);
  }

  /* ---------------------------------------------------------- camera */

  _camera() {
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

  /** Box between two points, with a square cross-section of half-width r. */
  _segment(a, b, r, color) {
    const ax = sub(b, a);
    const len = Math.hypot(ax[0], ax[1], ax[2]);
    if (len < 1e-6) return;
    const axis = [ax[0] / len, ax[1] / len, ax[2] / len];
    let u = cross(axis, [0, 1, 0]);
    if (Math.hypot(u[0], u[1], u[2]) < 1e-4) u = cross(axis, [1, 0, 0]);
    u = norm(u);
    const v = cross(axis, u);

    const c = [];
    for (const [su, sv] of [
      [1, 1],
      [1, -1],
      [-1, -1],
      [-1, 1],
    ]) {
      c.push([
        [a[0] + (u[0] * su + v[0] * sv) * r, a[1] + (u[1] * su + v[1] * sv) * r, a[2] + (u[2] * su + v[2] * sv) * r],
        [b[0] + (u[0] * su + v[0] * sv) * r, b[1] + (u[1] * su + v[1] * sv) * r, b[2] + (u[2] * su + v[2] * sv) * r],
      ]);
    }
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      this._quad(c[i][0], c[i][1], c[j][1], c[j][0], color);
    }
    // Caps keep the limbs from looking hollow at grazing angles.
    this._quad(c[0][0], c[1][0], c[2][0], c[3][0], color);
    this._quad(c[3][1], c[2][1], c[1][1], c[0][1], color);
  }

  /** Fraction [t0,t1] along a segment, drawn thicker — the accent collars. */
  _collar(a, b, t0, t1, r, color) {
    const p = (t) => [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
    this._segment(p(t0), p(t1), r, color);
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

  /** Chassis: an octagonal prism with a bevelled deck. */
  _chassis(pos, yaw, pitch, roll, R) {
    const H = 0.30;
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

    const ring = (r, y) => {
      const out = [];
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * Math.PI * 2 + Math.PI / 8;
        out.push(xf([Math.cos(a) * r, y, Math.sin(a) * r]));
      }
      return out;
    };

    const bot = ring(R * 0.9, -H * 0.5);
    const mid = ring(R, H * 0.12);
    const top = ring(R * 0.78, H * 0.5);

    for (let i = 0; i < 8; i++) {
      const j = (i + 1) % 8;
      this._quad(bot[i], mid[i], mid[j], bot[j], COL.shellDark);
      this._quad(mid[i], top[i], top[j], mid[j], COL.shell);
    }
    this.faces.push({ p: top, c: COL.shell });
    this.faces.push({ p: bot.slice().reverse(), c: COL.shellDark });
  }

  _legs(J, stance, legs) {
    for (let leg = 0; leg < legs; leg++) {
      const o = leg * 12;
      const hip = [J[o], J[o + 1], J[o + 2]];
      const knee = [J[o + 3], J[o + 4], J[o + 5]];
      const ankle = [J[o + 6], J[o + 7], J[o + 8]];
      const foot = [J[o + 9], J[o + 10], J[o + 11]];

      this._segment(hip, knee, 0.075, COL.shell);
      this._collar(hip, knee, 0.0, 0.34, 0.098, COL.accent);

      this._segment(knee, ankle, 0.062, COL.shell);
      this._collar(knee, ankle, 0.0, 0.17, 0.088, COL.accent);
      this._collar(knee, ankle, 0.8, 0.94, 0.072, COL.accent);

      this._segment(ankle, foot, 0.046, COL.shell);
      this._collar(ankle, foot, 0.0, 0.14, 0.07, COL.accent);
      this._collar(ankle, foot, 0.84, 1.0, 0.052, COL.accent);

      if (stance[leg] > 0.5) {
        this._collar(ankle, foot, 0.95, 1.04, 0.068, COL.accent);
      }
    }
  }

  _terrain(zc) {
    const ob = this.obstacles;
    for (let i = 0; i + 4 < ob.length; i += 5) {
      const z0 = ob[i + 2];
      const z1 = ob[i + 3];
      if (z1 < zc - 14 || z0 > zc + 30) continue;
      const top = ob[i + 4];
      if (top >= 0) {
        this._box(ob[i], ob[i + 1], 0, top, z0, z1, COL.block, COL.blockTop, false);
      } else {
        this._box(ob[i], ob[i + 1], top, 0, z0, z1, COL.pit, COL.pitFloor, true);
      }
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
    const c = this.ctx;
    const pts = [];
    for (let i = 0; i <= 20; i++) {
      const a = (i / 20) * Math.PI * 2;
      const v = this._view([cx + Math.cos(a) * r, y, cz + Math.sin(a) * r]);
      if (v[2] <= NEAR) return;
      pts.push(this._project(v));
    }
    c.save();
    c.strokeStyle = style;
    c.lineWidth = (width || 1) * this.dpr;
    if (dash) c.setLineDash(dash.map((d) => d * this.dpr));
    c.beginPath();
    c.moveTo(pts[0][0], pts[0][1]);
    for (const p of pts.slice(1)) c.lineTo(p[0], p[1]);
    c.stroke();
    c.restore();
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
    return `rgb(${Math.round(Math.min(255, c[0] * i * 255))},${Math.round(
      Math.min(255, c[1] * i * 255)
    )},${Math.round(Math.min(255, c[2] * i * 255))})`;
  }

  _paintFaces() {
    const ctx = this.ctx;
    const items = [];

    for (const f of this.faces) {
      const vs = f.p.map((p) => this._view(p));
      let depth = 0;
      let visible = false;
      for (const v of vs) {
        depth += v[2];
        if (v[2] > NEAR) visible = true;
      }
      if (!visible) continue;
      items.push({ f, vs, d: depth / vs.length });
    }
    items.sort((a, b) => b.d - a.d);

    for (const it of items) {
      const poly = this._clipNear(it.vs);
      if (poly.length < 3) continue;
      const pts = poly.map((v) => this._project(v));

      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath();
      ctx.fillStyle = this._shade(it.f);
      ctx.fill();
      ctx.lineWidth = 0.7 * this.dpr;
      ctx.strokeStyle = "rgba(24,24,26,0.30)";
      ctx.stroke();
    }
  }

  _paintShadows(from) {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = "rgba(20,20,24,0.055)";
    for (let i = from; i < this.faces.length; i++) {
      const f = this.faces[i];
      const flat = f.p.map((p) => {
        const t = p[1] / -LIGHT[1];
        return [p[0] + LIGHT[0] * t, 0.004, p[2] + LIGHT[2] * t];
      });
      const vs = flat.map((p) => this._view(p));
      const poly = this._clipNear(vs);
      if (poly.length < 3) continue;
      const pts = poly.map((v) => this._project(v));
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let k = 1; k < pts.length; k++) ctx.lineTo(pts[k][0], pts[k][1]);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }

  /* ---------------------------------------------------------------- frame */

  draw(t, L) {
    this.resize();
    const ctx = this.ctx;
    const pos = [t[L.T_POS], t[L.T_POS + 1], t[L.T_POS + 2]];

    // Follow the robot without snapping.
    const k = 0.09;
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
    this._terrain(pos[2]);
    const robotFrom = this.faces.length;
    // Leg count and chassis size are simulator state, not constants: the
    // frame is parametric from four legs to ten.
    const legs = Math.round(t[L.T_LEGS]) || 6;
    this._chassis(pos, t[L.T_YAW], t[L.T_PITCH], t[L.T_ROLL], t[L.T_BODY_R] || 0.95);
    this._legs(
      t.subarray(L.T_JOINTS, L.T_JOINTS + legs * 12),
      t.subarray(L.T_STANCE, L.T_STANCE + legs),
      legs
    );

    this._paintShadows(robotFrom);
    this._paintFaces();

    // Support polygon and centre of mass.
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
      const cx = pos[0] + t[L.T_COM];
      const cz = pos[2] + t[L.T_COM + 1];
      const bad = t[L.T_MARGIN] < 0.05;
      const col = bad ? "#e5391d" : "rgba(20,20,22,0.6)";
      this._line([cx - 0.22, 0.02, cz], [cx + 0.22, 0.02, cz], col, 1.5);
      this._line([cx, 0.02, cz - 0.22], [cx, 0.02, cz + 0.22], col, 1.5);
      this._ringXZ(cx, cz, 0.02, 0.13, col, 1.2);
      this._line([pos[0], pos[1], pos[2]], [cx, 0.02, cz], "rgba(20,20,22,0.18)", 1, [3, 4]);
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
    }
  }
}

window.Stage = Stage;
