/* Hexapod Gait Lab — browser side.
 *
 * Everything numeric happens in the wasm module; this file reads a flat f32
 * telemetry buffer out of wasm linear memory, draws it, and sends commands
 * back. Buffer offsets come from HX_LAYOUT, generated at build time from the
 * Rust source, so the two sides cannot drift apart.
 */

const L = window.HX_LAYOUT;
const CATALOGUE = window.HX_SERVOS;
const PARTS = window.HX_PARTS;

const PARAMS = [
  { label: "Cycle time", unit: "s", dp: 3 },
  { label: "Stride length", unit: "m", dp: 2 },
  { label: "Step height", unit: "m", dp: 2 },
  { label: "Body height", unit: "m", dp: 2 },
  { label: "Stance width", unit: "m", dp: 2 },
  { label: "Duty factor", unit: "", dp: 3 },
];
const ALL_LEGS = ["L1", "R1", "L2", "R2", "L3", "R3", "L4", "R4", "L5", "R5"];
/// Names of the legs this machine actually has.
const legNames = () => ALL_LEGS.slice(0, state.legs);
/* The course list comes from the Rust enum the simulator switches on, so the
 * buttons cannot drift out of step with it. The prose is keyed by name. */
const COURSES = (window.HX_COURSES || ["FLAT"]).map(
  (c) => c[0] + c.slice(1).toLowerCase()
);
const PRESETS = ["Tripod", "Ripple", "Wave"];

const COURSE_NOTES = {
  FLAT: "Level ground. The reference case — if a policy cannot beat the baseline here, it has learned nothing.",
  STEPS:
    "Staircases spanning the corridor, 16–34 cm per riser. Rewards lifting the feet and riding the body up the slope.",
  RUBBLE:
    "Scattered debris 10–58 cm tall. The lookahead inputs matter most here: each leg has to clear whatever is under its own landing spot.",
  GAPS: "Trenches 45–105 cm wide and 90 cm deep. A foot that lands in one usually ends the run, so foot placement dominates.",
  MIXED:
    "Rubble, then stairs, then trenches, then rubble again. The default training course.",
  RAMPS:
    "Grades up to 1.3 m over three to six metres, and about half of them banked across the corridor. A staircase is a sequence of shocks; a ramp is a sustained tilt, and a banked one slides you sideways the whole way up.",
    SLALOM:
        "Walls with a 3.5 m gate in each, staggered left and right. Nothing here can be climbed or jumped — a wall is 1.8 m, as tall as a fully stretched leg, and the gait cannot take off. The only way past is round it: Follow route yaws toward each gate.",
  SLICK:
    "Sheets of ice a centimetre thick and worth about a fifth of the grip of the ground around them, with a few low humps for company. Watch the traction meter, not the terrain.",
  GAUNTLET:
    "Rubble, ramps, a slalom, trenches, then ice. Everything the generator can build, in one run.",
  JUMP:
    "Parkour. Trenches wider than a stride, and platforms you can only reach by jumping the gap in front of them. The command is still a speed: run, jump, land without stripping the servos.",
};
const courseName = (i) => (COURSES[i] || "Flat").toUpperCase();
const isJump = () => courseName(state.courseKind) === "JUMP";

const COL = {
  ink: "#141416",
  accent: "#e5391d",
  dim: "#8a8a82",
  rule: "#d8d8d2",
  soft: "#f7f7f5",
  graphite: "#55554f",
};

let api = null;
let stage = null;
let course = new Float32Array(0);

const state = {
  paused: false,
  timeScale: 1,
  training: false,
  mode: 0,
  courseKind: 4,
  legs: 6,
  seed: 1,
  preset: 0,
  cmd: { fwd: 1, turn: 0 },
  keys: new Set(),
  stepUs: 0,
  iterMs: 0,
  build: { mass: 2.0, femurMm: 80, safety: 1.35 },
  cruise: 4.0,
  onelegLeg: 0,
  /* Index into the servo catalogue whose torque-speed line drives the joints,
   * or -1 for the generic 20 kg-cm default. */
  servo: -1,
  torque: null,
  sizing: { chassis: 0.45, runtime: 20 },
  pick: 4,
};

const $ = (id) => document.getElementById(id);
/* Phases live on a circle, so their arithmetic does too. */
const frac = (x) => x - Math.floor(x);
const circDist = (a, b) => {
  const d = Math.abs(frac(a) - frac(b));
  return Math.min(d, 1 - d);
};
const circMean = (xs) => {
  let sx = 0;
  let sy = 0;
  for (const x of xs) {
    sx += Math.cos(2 * Math.PI * x);
    sy += Math.sin(2 * Math.PI * x);
  }
  return frac(Math.atan2(sy, sx) / (2 * Math.PI));
};
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "—");
const signed = (v, d = 2) => {
  const s = fmt(v, d);
  return v >= 0 ? `+${s}` : s;
};

/* ------------------------------------------------------------------ wasm */

function decodeBase64(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Fresh view — wasm memory can be reallocated by any call that grows it. */
function telemetry() {
  return new Float32Array(api.memory.buffer, api.hx_telemetry_ptr(), api.hx_telemetry_len());
}

function readCourse() {
  const n = api.hx_course_len();
  course = new Float32Array(api.memory.buffer, api.hx_course_ptr(), n * 5).slice();
  const m = api.hx_route_len();
  stage.setCourse(course, new Float32Array(api.memory.buffer, api.hx_route_ptr(), m * 2).slice());
}

function curve() {
  const n = api.hx_curve_len();
  if (!n) return new Float32Array(0);
  return new Float32Array(api.memory.buffer, api.hx_curve_ptr(), n);
}

/* ------------------------------------------------------------------- log */

const logLines = [];
function log(msg) {
  const t = telemetry();
  const stamp = `[${fmt(t[L.T_TIME], 1).padStart(6, "0")}]`;
  logLines.push({ stamp, msg });
  if (logLines.length > 400) logLines.shift();
  for (const el of [$("logMini"), $("logFull")]) {
    if (!el) continue;
    const keep = el === $("logMini") ? 40 : 400;
    el.innerHTML = logLines
      .slice(-keep)
      .map((l) => `<div><span class="t">${l.stamp}</span> <span class="m">${l.msg}</span></div>`)
      .join("");
    el.scrollTop = el.scrollHeight;
  }
}

/* ---------------------------------------------------------------- charts */

/* A canvas is a replaced element: its width/height attributes are also its
 * intrinsic size, so writing the backing store feeds back into layout. Pin the
 * CSS height once — from the authored `height` attribute, which is what every
 * canvas here is laid out at — and derive *both* backing dimensions from the
 * CSS box after that. Sizing the width from the box and the height from the
 * attribute is what made the dials, which CSS shrinks to 84 px, come out with
 * an 84x168 buffer and draw every circle as an ellipse. */
function setCanvasHeight(cv, px) {
  cv._cssH = px;
  cv.style.height = `${px}px`;
}

function fitCanvas(cv) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (cv._cssH === undefined) setCanvasHeight(cv, Number(cv.getAttribute("height")) || cv.height);
  const r = cv.getBoundingClientRect();
  const w = Math.max(1, Math.round((r.width || cv._cssH) * dpr));
  const h = Math.max(1, Math.round(cv._cssH * dpr));
  if (cv.width !== w) cv.width = w;
  if (cv.height !== h) cv.height = h;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h, dpr };
}

/** Rolling multi-series line chart with an optional filled first series. */
class Trace {
  constructor(cv, series, cap = 260) {
    this.cv = cv;
    this.series = series;
    this.cap = cap;
    this.data = series.map(() => []);
  }
  push(vals) {
    for (let i = 0; i < vals.length; i++) {
      const d = this.data[i];
      d.push(vals[i]);
      if (d.length > this.cap) d.shift();
    }
  }
  draw() {
    const { ctx, w, h, dpr } = fitCanvas(this.cv);
    let lo = Infinity;
    let hi = -Infinity;
    for (const d of this.data) {
      for (const v of d) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!Number.isFinite(lo)) return;
    const pad = (hi - lo) * 0.15 + 1e-3;
    lo -= pad;
    hi += pad;
    const X = (i, n) => (i / Math.max(1, n - 1)) * w;
    const Y = (v) => h - ((v - lo) / (hi - lo)) * h;

    ctx.strokeStyle = "rgba(20,20,22,0.07)";
    ctx.lineWidth = dpr;
    for (let g = 1; g < 4; g++) {
      ctx.beginPath();
      ctx.moveTo(0, (h * g) / 4);
      ctx.lineTo(w, (h * g) / 4);
      ctx.stroke();
    }

    this.series.forEach((s, si) => {
      const d = this.data[si];
      if (d.length < 2) return;
      ctx.beginPath();
      d.forEach((v, i) => (i ? ctx.lineTo(X(i, d.length), Y(v)) : ctx.moveTo(X(i, d.length), Y(v))));
      if (s.fill) {
        ctx.lineTo(X(d.length - 1, d.length), h);
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.fillStyle = s.fill;
        ctx.fill();
        ctx.beginPath();
        d.forEach((v, i) => (i ? ctx.lineTo(X(i, d.length), Y(v)) : ctx.moveTo(X(i, d.length), Y(v))));
      }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = (s.width || 1.4) * dpr;
      ctx.stroke();
    });

    ctx.fillStyle = COL.dim;
    ctx.font = `${9 * dpr}px ui-monospace, monospace`;
    ctx.fillText(hi.toFixed(2), 4 * dpr, 11 * dpr);
    ctx.fillText(lo.toFixed(2), 4 * dpr, h - 4 * dpr);
  }
}

function drawDial(cv, value, max, label, warn, unit, mark) {
  const { ctx, w, h, dpr } = fitCanvas(cv);
  const r = Math.min(w, h) * 0.36;
  const cx = w / 2;
  const cy = h / 2;
  const swept = Math.max(0, Math.min(1, value / max));
  // The gauge sweeps 270°, from south-west round to south-east.
  const A0 = Math.PI * 0.75;
  const at = (f) => -A0 + f * Math.PI * 1.5;
  ctx.lineWidth = 4 * dpr;
  ctx.strokeStyle = COL.rule;
  ctx.beginPath();
  ctx.arc(cx, cy, r, -A0, A0);
  ctx.stroke();
  ctx.strokeStyle = warn ? COL.accent : COL.ink;
  ctx.beginPath();
  ctx.arc(cx, cy, r, -A0, at(swept));
  ctx.stroke();
  // Where the reading stops being fine, so the sweep means something.
  if (mark !== undefined) {
    const a = at(Math.max(0, Math.min(1, mark)));
    ctx.strokeStyle = COL.dim;
    ctx.lineWidth = 1.4 * dpr;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * (r - 6 * dpr), cy + Math.sin(a) * (r - 6 * dpr));
    ctx.lineTo(cx + Math.cos(a) * (r + 6 * dpr), cy + Math.sin(a) * (r + 6 * dpr));
    ctx.stroke();
  }
  ctx.fillStyle = COL.ink;
  ctx.textAlign = "center";
  ctx.font = `700 ${21 * dpr}px ui-monospace, monospace`;
  ctx.fillText(label, cx, cy + 4 * dpr);
  if (unit) {
    ctx.fillStyle = COL.dim;
    ctx.font = `${9 * dpr}px ui-monospace, monospace`;
    ctx.fillText(unit, cx, cy + 17 * dpr);
  }
  ctx.textAlign = "left";
}

/* --------------------------------------------------- measured gait pattern
 *
 * The old panel drew the gait's *schedule*: phase offsets out of the parameter
 * vector, laid out as bars. It was a picture of the table, and it was right by
 * construction — it said TRIPOD whether or not the machine was walking one.
 *
 * This one records what the feet actually did. Every frame it takes the stance
 * flag and the load share of each leg straight out of the telemetry buffer and
 * appends them to a ring; the panel is that recording, drawn as a footfall
 * raster with time running left to right. Everything under it — cycle time,
 * duty per leg, how many legs are ever in the air together, which named
 * pattern that adds up to — is measured from the same recording, so when a
 * learned policy invents a coordination nobody has a name for, the panel says
 * so instead of repeating the label it started from.
 */
const GAIT_WINDOW = 4.0; // seconds kept
const GAIT_CAP = 900; // samples

class Footfalls {
  constructor() {
    this.t = [];
    this.stance = [];
    this.load = [];
    this.phase = [];
    this.legs = 0;
  }
  reset(legs) {
    this.legs = legs;
    this.t.length = 0;
    this.stance.length = 0;
    this.load.length = 0;
    this.phase.length = 0;
  }
  push(t, tel, L, legs) {
    if (legs !== this.legs) this.reset(legs);
    // The clock runs backwards on a reset, and a fallen run repeats its last
    // instant; neither belongs in the record.
    const last = this.t[this.t.length - 1];
    if (last !== undefined && t <= last) this.reset(legs);
    const s = new Uint8Array(legs);
    const q = new Float32Array(legs);
    for (let i = 0; i < legs; i++) {
      s[i] = tel[L.T_STANCE + i] > 0.5 ? 1 : 0;
      q[i] = tel[L.T_LOAD + i];
    }
    this.t.push(t);
    this.stance.push(s);
    this.load.push(q);
    this.phase.push(tel[L.T_PHASE]);
    while (this.t.length > GAIT_CAP || (this.t.length > 2 && t - this.t[0] > GAIT_WINDOW)) {
      this.t.shift();
      this.stance.shift();
      this.load.shift();
      this.phase.shift();
    }
  }

  /** Duty factor of each leg over the window. */
  duty() {
    const n = this.legs;
    const out = new Array(n).fill(0);
    if (!this.t.length) return out;
    for (const s of this.stance) for (let i = 0; i < n; i++) out[i] += s[i];
    return out.map((v) => v / this.stance.length);
  }

  /** Times at which each leg planted, from the rising edges of its record. */
  onsets(leg) {
    const out = [];
    for (let k = 1; k < this.t.length; k++) {
      if (!this.stance[k - 1][leg] && this.stance[k][leg]) out.push(this.t[k]);
    }
    return out;
  }

  /** Median interval between one leg's footfalls: the cycle it actually ran. */
  cycle() {
    const gaps = [];
    for (let i = 0; i < this.legs; i++) {
      const o = this.onsets(i);
      for (let k = 1; k < o.length; k++) gaps.push(o[k] - o[k - 1]);
    }
    if (!gaps.length) return 0;
    gaps.sort((a, b) => a - b);
    return gaps[gaps.length >> 1];
  }

  /** Most legs ever off the ground at once — what separates the patterns. */
  peakSwing() {
    let peak = 0;
    for (const s of this.stance) {
      let up = 0;
      for (let i = 0; i < this.legs; i++) up += 1 - s[i];
      if (up > peak) peak = up;
    }
    return peak;
  }

  /** Phase offset of each leg, recovered from the gait clock at its footfalls.
   *
   * A leg plants when its own phase wraps, which happens at global phase
   * `frac(-offset)` — so reading the clock at each rising edge inverts back to
   * the offset the leg is really running, whatever the parameter vector says
   * it should be. */
  offsets() {
    const out = new Array(this.legs).fill(null);
    for (let i = 0; i < this.legs; i++) {
      const seen = [];
      for (let k = 1; k < this.t.length; k++) {
        if (!this.stance[k - 1][i] && this.stance[k][i]) seen.push(frac(-this.phase[k]));
      }
      if (seen.length) out[i] = circMean(seen);
    }
    return out;
  }

  /** The named pattern this footfall record adds up to, if any.
   *
   * Matched against the presets the simulator defines — asked for over the
   * bridge rather than kept as a second copy here — so a coordination the
   * learner invented reads as its own thing instead of as the label it was
   * seeded from. */
  classify() {
    const n = this.legs;
    if (this.t.length < 40 || !n) return "—";
    if (this.peakSwing() === 0) return "STANDING";
    const measured = this.offsets();
    if (measured.some((v) => v === null)) return "—";

    let best = null;
    for (let p = 0; p < api.hx_preset_count(); p++) {
      let sq = 0;
      for (let i = 0; i < n; i++) {
        const d = circDist(measured[i], api.hx_preset_offset(p, i));
        sq += d * d;
      }
      const rms = Math.sqrt(sq / n);
      if (!best || rms < best.rms) best = { p, rms };
    }
    return best.rms < 0.05 ? presetName(best.p).toUpperCase() : "IRREGULAR";
  }
}

const falls = new Footfalls();

function drawGaitDiagram(t) {
  const cv = $("cGait");
  const legs = Math.round(t[L.T_LEGS]) || 6;
  const rowH = 17;
  const want = legs * rowH + 22;
  if (cv._cssH !== want) setCanvasHeight(cv, want);

  const { ctx, w, h, dpr } = fitCanvas(cv);
  const labelW = 22 * dpr;
  // A gutter on the right for each leg's measured duty factor.
  const gutter = 26 * dpr;
  const trackW = w - labelW - gutter;
  const names = legNames();
  const rows = Math.min(legs, names.length);
  const rh = (h - 16 * dpr) / rows;
  ctx.font = `${8.5 * dpr}px ui-monospace, monospace`;

  const rec = falls;
  const t0 = rec.t.length ? rec.t[0] : 0;
  const t1 = rec.t.length ? rec.t[rec.t.length - 1] : 1;
  const span = Math.max(1e-3, t1 - t0);
  const X = (tt) => labelW + ((tt - t0) / span) * trackW;

  // One second of grid, so the raster reads as a time axis and not a pattern.
  ctx.strokeStyle = "rgba(20,20,22,0.10)";
  ctx.lineWidth = dpr;
  for (let s = Math.ceil(t0); s < t1; s += 1) {
    ctx.beginPath();
    ctx.moveTo(X(s), 0);
    ctx.lineTo(X(s), h - 14 * dpr);
    ctx.stroke();
  }

  const duty = rec.duty();
  for (let i = 0; i < rows; i++) {
    const y = i * rh + rh * 0.16;
    const bh = rh * 0.62;
    ctx.fillStyle = COL.dim;
    ctx.fillText(names[i], 0, y + bh * 0.85);
    ctx.fillStyle = COL.soft;
    ctx.fillRect(labelW, y, trackW, bh);

    // Contiguous stance runs, shaded by how much weight the foot carried.
    let k = 0;
    while (k < rec.t.length) {
      if (!rec.stance[k][i]) {
        k++;
        continue;
      }
      const from = k;
      let load = 0;
      while (k < rec.t.length && rec.stance[k][i]) {
        load += rec.load[k][i];
        k++;
      }
      const share = (load / (k - from)) * legs; // 1.0 = an even share
      const a = 0.30 + Math.min(0.70, share * 0.42);
      ctx.fillStyle = `rgba(20,20,22,${a.toFixed(3)})`;
      const x0 = X(rec.t[from]);
      const x1 = X(rec.t[Math.min(k, rec.t.length - 1)]);
      ctx.fillRect(x0, y, Math.max(dpr, x1 - x0), bh);
    }

    ctx.fillStyle = COL.dim;
    ctx.textAlign = "right";
    ctx.fillText(duty[i].toFixed(2), w - 1 * dpr, y + bh * 0.85);
    ctx.textAlign = "left";
  }

  // Now, at the right-hand edge.
  ctx.strokeStyle = COL.accent;
  ctx.lineWidth = 1.4 * dpr;
  ctx.beginPath();
  ctx.moveTo(labelW + trackW, 0);
  ctx.lineTo(labelW + trackW, h - 14 * dpr);
  ctx.stroke();

  const cycle = rec.cycle();
  const mean = duty.length ? duty.reduce((a, b) => a + b, 0) / duty.length : 0;
  ctx.fillStyle = COL.dim;
  ctx.fillText(
    `${rec.classify()} · cycle ${cycle ? cycle.toFixed(2) + " s" : "—"} · duty ${mean.toFixed(
      2
    )} · ${rec.peakSwing()} of ${legs} in the air`,
    0,
    h - 3 * dpr
  );
}

function drawCurve() {
  const cv = $("cCurve");
  const { ctx, w, h, dpr } = fitCanvas(cv);
  const c = curve();
  if (c.length < 1) {
    ctx.fillStyle = COL.dim;
    ctx.font = `${11 * dpr}px ui-monospace, monospace`;
    ctx.fillText("No iterations yet — press TRAIN.", 12 * dpr, h / 2);
    return;
  }
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of c) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const pad = (hi - lo) * 0.12 + 1;
  lo -= pad;
  hi += pad;
  const X = (i) => (i / Math.max(1, c.length - 1)) * w;
  const Y = (v) => h - ((v - lo) / (hi - lo)) * h;

  ctx.strokeStyle = "rgba(20,20,22,0.07)";
  ctx.lineWidth = dpr;
  for (let g = 1; g < 5; g++) {
    ctx.beginPath();
    ctx.moveTo(0, (h * g) / 5);
    ctx.lineTo(w, (h * g) / 5);
    ctx.stroke();
  }

  // Baseline reference: iteration 0 is the hand-tuned gait by construction.
  const base = c[0];
  ctx.strokeStyle = COL.graphite;
  ctx.setLineDash([6 * dpr, 4 * dpr]);
  ctx.lineWidth = 1.2 * dpr;
  ctx.beginPath();
  ctx.moveTo(0, Y(base));
  ctx.lineTo(w, Y(base));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "rgba(20,20,22,0.45)";
  ctx.lineWidth = 1.1 * dpr;
  ctx.beginPath();
  for (let i = 0; i < c.length; i++) i ? ctx.lineTo(X(i), Y(c[i])) : ctx.moveTo(X(i), Y(c[i]));
  ctx.stroke();

  let best = -Infinity;
  ctx.strokeStyle = COL.accent;
  ctx.lineWidth = 2 * dpr;
  ctx.beginPath();
  for (let i = 0; i < c.length; i++) {
    best = Math.max(best, c[i]);
    i ? ctx.lineTo(X(i), Y(best)) : ctx.moveTo(X(i), Y(best));
  }
  ctx.stroke();

  ctx.fillStyle = COL.accent;
  ctx.beginPath();
  ctx.arc(X(c.length - 1), Y(best), 3 * dpr, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = COL.dim;
  ctx.font = `${9 * dpr}px ui-monospace, monospace`;
  ctx.fillText(`best ${best.toFixed(1)}`, 5 * dpr, 12 * dpr);
  ctx.fillText(`hand-tuned ${base.toFixed(1)}`, 5 * dpr, Y(base) - 5 * dpr);
  ctx.fillText(`${c.length - 1} iterations`, 5 * dpr, h - 5 * dpr);
}

function drawProfile() {
  const cv = $("cProfile");
  const { ctx, w, h, dpr } = fitCanvas(cv);
  const z0 = -4;
  const z1 = 62;
  const n = 460;
  const hs = [];
  let lo = 0;
  let hi = 0.3;
  for (let i = 0; i < n; i++) {
    const z = z0 + ((z1 - z0) * i) / (n - 1);
    const v = api.hx_height(0, z);
    hs.push(v);
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const pad = (hi - lo) * 0.2 + 0.05;
  const Y = (v) => h - ((v - lo + pad) / (hi - lo + pad * 2)) * h;

  ctx.strokeStyle = "rgba(20,20,22,0.10)";
  ctx.lineWidth = dpr;
  ctx.beginPath();
  ctx.moveTo(0, Y(0));
  ctx.lineTo(w, Y(0));
  ctx.stroke();

  ctx.beginPath();
  hs.forEach((v, i) => {
    const x = (i / (n - 1)) * w;
    i ? ctx.lineTo(x, Y(v)) : ctx.moveTo(x, Y(v));
  });
  ctx.lineTo(w, Y(lo - pad));
  ctx.lineTo(0, Y(lo - pad));
  ctx.closePath();
  ctx.fillStyle = "rgba(20,20,22,0.07)";
  ctx.fill();

  ctx.beginPath();
  hs.forEach((v, i) => {
    const x = (i / (n - 1)) * w;
    i ? ctx.lineTo(x, Y(v)) : ctx.moveTo(x, Y(v));
  });
  ctx.strokeStyle = COL.ink;
  ctx.lineWidth = 1.3 * dpr;
  ctx.stroke();

  // Where the robot is right now.
  const t = telemetry();
  const rx = ((t[L.T_POS + 2] - z0) / (z1 - z0)) * w;
  ctx.strokeStyle = COL.accent;
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath();
  ctx.moveTo(rx, 0);
  ctx.lineTo(rx, h);
  ctx.stroke();

  ctx.fillStyle = COL.dim;
  ctx.font = `${9 * dpr}px ui-monospace, monospace`;
  ctx.fillText(`${hi.toFixed(2)} m`, 4 * dpr, 11 * dpr);
  ctx.fillText(`${lo.toFixed(2)} m`, 4 * dpr, h - 4 * dpr);
}

/* ------------------------------------------------------------------- UI */

/* The alternating preset is named for what it produces on this frame: a
 * tripod on six legs, a trot on four. */
/* Simulator units to metres, the same number the hardware tab is built on. */
function simScale() {
  return state.build.femurMm / 1000 / 0.8;
}

function presetName(i) {
  if (i !== 0) return PRESETS[i];
  return { 4: "Trot", 6: "Tripod", 8: "Tetrapod" }[state.legs] || "Alternate";
}

/* The first preset is named for what it produces on this frame, and is
 * flagged when it cannot hold the machine up statically. */
function refreshPresetButtons() {
  const stable = api.hx_alternating_is_stable() === 1;
  const names = [presetName(0), "Ripple", "Wave"];
  $("presetBtns").innerHTML = names
    .map(
      (p, i) =>
        `<button class="btn" data-preset="${i}" data-on="${i === state.preset}"${
          i === 0 && !stable ? ' title="Not statically stable on this frame"' : ""
        }>${p}${i === 0 && !stable ? " \u26a0" : ""}</button>`
    )
    .join("");
  $("presetNote").textContent = stable
    ? ""
    : "the alternating gait stands on two feet here — it falls over";
}

/* Anything with one row per leg has to be rebuilt when the count changes. */
function buildLegUI() {
  $("loadBars").innerHTML = legNames()
    .map(
      (n, i) =>
        `<div class="bar"><span>${n}</span><div class="track"><div class="fill" id="load${i}"></div></div></div>`
    )
    .join("");
  const ol = $("onelegBtns");
  if (ol) {
    ol.innerHTML = legNames()
      .map(
        (n, i) =>
          `<button class="btn" data-oneleg="${i}" data-on="${i === state.onelegLeg}">${n}</button>`
      )
      .join("");
  }
}

function buildStaticUI() {
  buildLegUI();

  $("sliders").innerHTML = PARAMS.map(
    (p, i) => `<label class="slider">
      <span class="hd"><span class="k">${p.label}</span><b id="pv${i}">—</b></span>
      <input type="range" id="pr${i}" min="0" max="1000" step="1" value="500">
    </label>`
  ).join("");

  PARAMS.forEach((p, i) => {
    const lo = api.hx_param_lo(i);
    const hi = api.hx_param_hi(i);
    const el = $(`pr${i}`);
    el.value = Math.round(((api.hx_get_param(i) - lo) / (hi - lo)) * 1000);
    el.addEventListener("input", () => {
      const v = lo + ((hi - lo) * el.value) / 1000;
      api.hx_set_param(i, v);
      $(`pv${i}`).textContent = `${v.toFixed(p.dp)}${p.unit ? " " + p.unit : ""}`;
      if (state.mode === 0) api.hx_reset_live();
    });
  });

  refreshPresetButtons();
  $("courseBtns").innerHTML = COURSES.map(
    (c, i) => `<button class="btn" data-course="${i}" data-on="${i === state.courseKind}">${c}</button>`
  ).join("");

  const rl = $("rLegs");
  rl.min = api.hx_legs_min();
  rl.max = api.hx_legs_max();
  rl.value = state.legs;
  $("vLegs").textContent = `${state.legs} legs`;
  $("legsNote").textContent = `${api.hx_legs_min()}–${api.hx_legs_max()}, in pairs`;

  const lo = api.hx_cruise_lo();
  const hi = api.hx_cruise_hi();
  const r = $("rCruise");
  r.min = lo;
  r.max = hi;
  r.step = isJump() ? 0.01 : 0.1;
  r.value = state.cruise;
  syncCruiseLabels();

  $("selServo").innerHTML =
    `<option value="-1">Generic 20 kg·cm digital</option>` +
    CATALOGUE.servos
      .map(
        (v, i) =>
          `<option value="${i}">${v.part} — ${v.stall} kg·cm, ${v.speed}s/60°</option>`
      )
      .join("");
  $("selServo").value = String(state.servo);
}

/* A different machine cannot use a policy learned for the old one, so
 * training is dropped and the display goes back to the hand-tuned gait. */
function afterMachineChange() {
  state.training = false;
  $("btnTrain").dataset.on = "false";
  $("btnTrain").textContent = "Train";
  setMode(0);
  updateTrainingPanel();
  updateHardware();
  updateSystem();
}

/* One line describing the machine the simulator is currently driving. Read
 * from the catalogue rather than telemetry so it is right immediately, before
 * the next frame publishes. */
function describeMachine() {
  const v = state.servo >= 0 ? CATALOGUE.servos[state.servo] : null;
  const stall = v ? v.stall : 20;
  const s60 = v ? v.speed : 0.16;
  const rpm = 60 / (6 * s60);
  const label =
    { 4: "QUADRUPED", 6: "HEXAPOD", 8: "OCTOPOD", 10: "DECAPOD" }[state.legs] || "WALKER";
  $("machineNote").textContent = `${state.build.mass.toFixed(1)} kg · ${stall.toFixed(
    1
  )} kg·cm · ${rpm.toFixed(0)} rpm`;
  $("hModel").textContent = `${label} · ${state.legs * 3} DOF`;
  $("hudDof").textContent = String(state.legs * 3);
  $("hudLegs").textContent = String(state.legs);
  // The policy's shape is a function of the frame, so read it rather than
  // writing the hexapod's numbers into the copy.
  $("trTheta").textContent = `${api.hx_theta_len()} parameters`;
  $("trShape").textContent = `${api.hx_n_act()}×${api.hx_n_obs()}`;
}

function syncSliders() {
  PARAMS.forEach((p, i) => {
    const lo = api.hx_param_lo(i);
    const hi = api.hx_param_hi(i);
    const v = state.mode === 1 ? api.hx_gait(1, i) : api.hx_get_param(i);
    $(`pv${i}`).textContent = `${v.toFixed(p.dp)}${p.unit ? " " + p.unit : ""}`;
    const el = $(`pr${i}`);
    el.value = Math.round(((v - lo) / (hi - lo)) * 1000);
    el.disabled = state.mode !== 0;
  });
  $("paramLock").textContent =
    state.mode === 1 ? "set by policy" : state.mode === 2 ? "one-leg drill" : "";
}

function setMode(mode) {
  if (mode === 2 && state.courseKind !== 0) {
    state.courseKind = 0;
    api.hx_set_course(0, state.seed);
    readCourse();
    document
      .querySelectorAll("[data-course]")
      .forEach((b) => (b.dataset.on = String(+b.dataset.course === 0)));
    $("hCourse").textContent = "FLAT";
    $("trCourse").textContent = "FLAT";
    $("tSummary").textContent =
      `FLAT · seed ${state.seed} · ${api.hx_course_len()} obstacles · ${api.hx_route_len()} waypoints`;
    $("tNote").textContent = COURSE_NOTES.FLAT || "";
  }
  state.mode = mode;
  api.hx_set_mode(mode);
  $("btnBase").dataset.on = mode === 0;
  $("btnLearn").dataset.on = mode === 1;
  const btnOl = $("btnOneleg");
  const btnWalk = $("btnWalk");
  if (btnOl) btnOl.dataset.on = String(mode === 2);
  if (btnWalk) btnWalk.dataset.on = String(mode !== 2);
  $("hPolicy").textContent = mode === 2 ? "ONE LEG" : mode === 1 ? "LEARNED" : "HAND-TUNED";
  syncSliders();
  refreshGaitTable();
  updateHardware();
  if (mode === 2) {
    api.hx_set_oneleg_leg(state.onelegLeg);
    state.cmd = { fwd: 0, turn: 0 };
    document.querySelectorAll("[data-cmd]").forEach((b) => (b.dataset.on = "false"));
    const stop = document.querySelector('[data-cmd="stop"]');
    if (stop) stop.dataset.on = "true";
    if (stage && stage.setView) {
      stage.setView("orbit");
      stage.az = -1.05;
      stage.el = 0.34;
      stage.dist = 5.2;
    }
    log(`drill.oneleg("${legNames()[state.onelegLeg] || "L1"}")`);
  } else {
    if (state.cmd.fwd === 0 && state.cmd.turn === 0) {
      state.cmd = { fwd: 1, turn: 0 };
      document.querySelectorAll("[data-cmd]").forEach((b) => (b.dataset.on = "false"));
      const fwd = document.querySelector('[data-cmd="fwd"]');
      if (fwd) fwd.dataset.on = "true";
    }
    log(mode === 1 ? "policy.use(\"learned\")" : "policy.use(\"hand-tuned\")");
  }
}

function setTab(name) {
  document.querySelectorAll(".tab").forEach((b) => {
    b.setAttribute("aria-selected", String(b.dataset.tab === name));
  });
  document.querySelectorAll(".pane").forEach((p) => {
    p.dataset.active = String(p.dataset.pane === name);
  });
  if (name === "terrain") drawProfile();
  if (name === "training") drawCurve();
  if (name === "hardware") updateHardware();
  if (name === "system") updateSystem();
}

function refreshGaitTable() {
  const trained = telemetry()[L.T_TRAINED] > 0.5;
  const rows = PARAMS.map((p, i) => {
    const a = api.hx_gait(0, i);
    const b = trained ? api.hx_gait(1, i) : NaN;
    const d = trained ? b - a : NaN;
    const arrow = !Number.isFinite(d) ? "" : d > 0.001 ? "▲" : d < -0.001 ? "▼" : "·";
    return `<tr><td>${p.label}</td><td>${a.toFixed(p.dp)}</td><td>${
      trained ? b.toFixed(p.dp) : "—"
    }</td><td>${trained ? `${arrow} ${Math.abs(d).toFixed(p.dp)}` : "—"}</td></tr>`;
  });
  rows.push(
    `<tr><td>Phase offsets</td><td>${legNames()
      .map((_, i) => api.hx_gait(0, 6 + i).toFixed(2))
      .join(" ")}</td><td colspan="2">${
      trained
        ? legNames()
            .map((_, i) => api.hx_gait(1, 6 + i).toFixed(2))
            .join(" ")
        : "—"
    }</td></tr>`
  );
  // Cycle time, stride and duty above are the *nominal* values. The policy
  // scales all three every tick, so the live numbers are in the stage HUD.
  const t = telemetry();
  rows.push(
    `<tr><td>Running now</td><td colspan="3">cycle ${fmt(t[L.T_CYCLE_NOW], 3)} s · stride ${fmt(
      t[L.T_STRIDE_NOW],
      2
    )} m · duty ${fmt(t[L.T_DUTY_NOW], 3)} — modulated online at ${fmt(
      t[L.T_CMD_SPEED],
      1
    )} m/s</td></tr>`
  );
  $("tblGait").querySelector("tbody").innerHTML = rows.join("");
}

/* --------------------------------------------------------------- hardware */

function updateHardware() {
  if (!api) return;
  const scale = state.build.femurMm / 1000 / 0.8; // femur is 0.80 sim units
  api.hx_set_build(scale, state.build.mass, state.build.safety);
  const ptr = api.hx_measure_torque();
  const q = new Float32Array(api.memory.buffer, ptr, 8);
  state.torque = Array.from(q);

  $("tqCoxa").textContent = fmt(q[0], 2);
  $("tqFemur").textContent = fmt(q[1], 2);
  $("tqTibia").textContent = fmt(q[2], 2);
  $("tqReq").textContent = fmt(q[3], 1);
  $("hwDims").textContent =
    `coxa ${(scale * 300).toFixed(0)} mm · femur ${(scale * 800).toFixed(0)} mm · ` +
    `tibia ${(scale * 1000).toFixed(0)} mm · stance ${q[6].toFixed(0)} mm · ` +
    `ride height ${q[7].toFixed(0)} mm`;
  $("tqNote").textContent =
    `Peak foot load ${q[4].toFixed(1)} N. Sized from the ${
      state.mode === 1 ? "learned" : "hand-tuned"
    } gait with a ${state.build.safety.toFixed(2)}× factor over a 1.5× dynamic allowance. ` +
    `Stall ratings are a ceiling, not a duty point — continuous torque is far lower.`;

  const req = q[3];
  const rows = CATALOGUE.servos
    .slice()
    .sort((a, b) => a.low * 18 - b.low * 18)
    .map((s) => {
      const pass = s.stall >= req;
      const head = req > 0 ? s.stall / req : 0;
      const vend = s.vendor == null ? "—" : `$${(s.vendor * 18).toFixed(0)}`;
      return `<tr data-pass="${pass}">
        <td><a href="${s.source}" target="_blank" rel="noopener">${s.part}</a> <span class="k">${s.maker}</span></td>
        <td>${s.stall.toFixed(1)} @ ${s.volts}V</td>
        <td>${head.toFixed(2)}×</td>
        <td>${((s.mass * 18) / 1000).toFixed(2)} kg</td>
        <td>${s.bus}${s.feedback ? " ⟳" : ""}</td>
        <td>$${(s.low * 18).toFixed(0)} – $${(s.high * 18).toFixed(0)}</td>
        <td>${vend}${s.checked ? "" : ' <span class="k">est</span>'}</td>
        <td><span class="tag ${pass ? "pass" : "fail"}">${pass ? "OK" : "LOW"}</span></td>
      </tr>`;
    });
  $("tblServo").querySelector("tbody").innerHTML = rows.join("");

  const checked = CATALOGUE.servos.filter((s) => s.checked);
  $("priceNote").innerHTML =
    `Distributor prices read from vendor pages on ${CATALOGUE.checked}: ` +
    checked.map((s) => `${s.part} $${s.vendor.toFixed(2)} at ${s.vendorName}`).join(", ") +
    `. Marketplace bands are indicative street prices for AliExpress-grade listings and are not quotes — ` +
    `the same part routinely varies 3–5× between a marketplace seller and a western distributor, and clone ` +
    `quality varies with it. Verify before ordering. Prices exclude brackets, horns, wiring, a controller ` +
    `and a battery, which together usually match the servo bill.`;
}

/* ---------------------------------------------------------------- system */

const FAILURE = [
  "",
  "no solution — the robot cannot carry the battery this endurance needs",
  "no stock pack has both the energy and the peak-current rating",
  "under-torqued at the converged mass",
];

/** Solve the whole machine around servo `i`. Returns the S_* buffer, copied. */
function solveSystem(i) {
  const ptr = api.hx_solve_system(i);
  return new Float32Array(api.memory.buffer, ptr, L.S_LEN).slice();
}

function part(i) {
  return i >= 0 && i < PARTS.length ? PARTS[i] : null;
}

function updateSystem() {
  if (!api) return;
  api.hx_set_sizing(state.sizing.chassis, state.sizing.runtime, state.build.safety);

  // One full solve per servo — the whole point of the tab is the comparison.
  const rows = CATALOGUE.servos.map((sv, i) => ({ sv, i, s: solveSystem(i) }));

  let cheapest = -1;
  let best = Infinity;
  for (const r of rows) {
    const ok = r.s[L.S_CONVERGED] > 0.5 && r.s[L.S_SERVO_OK] > 0.5;
    if (ok && r.s[L.S_COST] < best) {
      best = r.s[L.S_COST];
      cheapest = r.i;
    }
  }
  if (cheapest >= 0 && !state.pickLocked) state.pick = cheapest;

  $("tblSystem").querySelector("tbody").innerHTML = rows
    .map(({ sv, i, s }) => {
      const ok = s[L.S_CONVERGED] > 0.5 && s[L.S_SERVO_OK] > 0.5;
      const fail = FAILURE[Math.round(s[L.S_FAILURE])];
      const batt = part(Math.round(s[L.S_BATTERY_I]));
      return `<tr data-pass="${ok}" data-servo="${i}" style="cursor:pointer"${
        i === state.pick ? ' data-sel="1"' : ""
      }>
        <td>${sv.part}${i === state.pick ? ' <span class="k">◀ built</span>' : ""}</td>
        <td>${s[L.S_ALLUP].toFixed(2)} kg</td>
        <td>${s[L.S_REQ_TORQUE].toFixed(1)}</td>
        <td>${sv.stall.toFixed(1)}</td>
        <td>${batt ? batt.name.split(" ").slice(0, 2).join(" ") : "—"}</td>
        <td>${s[L.S_MEAN_A].toFixed(2)}</td>
        <td>${s[L.S_PEAK_A].toFixed(1)}</td>
        <td>${ok ? s[L.S_RUNTIME].toFixed(0) + " min" : "—"}</td>
        <td>$${s[L.S_COST].toFixed(0)}</td>
        <td><span class="tag ${ok ? "pass" : "fail"}">${ok ? "OK" : "NO"}</span>${
        ok ? "" : ` <span class="k">${fail}</span>`
      }</td>
      </tr>`;
    })
    .join("");

  const s = solveSystem(state.pick);
  const sv = CATALOGUE.servos[state.pick];
  const ok = s[L.S_CONVERGED] > 0.5 && s[L.S_SERVO_OK] > 0.5;

  $("sysStatus").textContent = ok
    ? `${sv.part} · $${s[L.S_COST].toFixed(0)} · ${s[L.S_ALLUP].toFixed(2)} kg`
    : FAILURE[Math.round(s[L.S_FAILURE])];
  $("sysIters").textContent = `${Math.round(s[L.S_ITERATIONS])} iterations to converge`;
  $("sysAllUp").textContent = `${s[L.S_ALLUP].toFixed(2)} kg all-up`;
  $("sysMeanA").textContent = s[L.S_MEAN_A].toFixed(2);
  $("sysPeakA").textContent = s[L.S_PEAK_A].toFixed(1);
  $("sysWatts").textContent = s[L.S_WATTS].toFixed(0);
  $("sysRuntime").textContent = ok ? s[L.S_RUNTIME].toFixed(0) : "—";
  $("sysPick").textContent = `${sv.part} · ${sv.bus}`;

  const total = s[L.S_ALLUP] || 1;
  const budget = [
    ["Servos ×18", s[L.S_SERVO_KG]],
    ["Chassis", s[L.S_CHASSIS_KG]],
    ["Battery", s[L.S_BATT_KG]],
    ["Electronics", s[L.S_ELEC_KG]],
  ];
  $("massBars").innerHTML = budget
    .map(
      ([n, v]) => `<div class="bar" style="grid-template-columns:88px 1fr 62px">
        <span>${n}</span>
        <div class="track"><div class="fill" style="width:${(v / total) * 100}%"></div></div>
        <span style="text-align:right">${v.toFixed(2)} kg</span></div>`
    )
    .join("");

  // Parts list for the chosen build.
  const rangers = Math.round(s[L.S_RANGERS]);
  const list = [
    ["Servo", { name: sv.part, maker: sv.maker, mass: sv.mass, unit: sv.low, note: sv.bus + (sv.feedback ? ", reports load" : ", no feedback") }, 18],
    ["Battery", part(Math.round(s[L.S_BATTERY_I])), 1],
    ["Regulator", part(Math.round(s[L.S_REG_I])), 1],
    [sv.bus === "PWM" ? "Servo driver" : "Bus adapter", part(Math.round(s[L.S_DRIVER_I])), Math.round(s[L.S_DRIVER_N])],
    ["Compute", part(Math.round(s[L.S_COMPUTE_I])), 1],
    ["Rangefinder", part(Math.round(s[L.S_RANGER_I])), rangers],
    ["I²C mux", part(Math.round(s[L.S_SUPPORT_I])), 1],
    ["IMU", part(Math.round(s[L.S_IMU_I])), 1],
  ];

  let mass = 0;
  const body = list
    .map(([role, p, qty]) => {
      if (!p) {
        return `<tr><td>${role}</td><td>none</td><td>—</td><td>—</td><td>—</td>
          <td class="k">pack voltage already matches the servo bus</td></tr>`;
      }
      mass += (p.mass * qty) / 1000;
      const href = p.source ? `<a href="${p.source}" target="_blank" rel="noopener">${p.name}</a>` : p.name;
      return `<tr><td>${role}</td><td>${href} <span class="k">${p.maker || ""}</span></td>
        <td>×${qty}</td><td>${((p.mass * qty) / 1000).toFixed(2)} kg</td>
        <td>$${(p.unit * qty).toFixed(2)}</td><td class="why">${p.note || ""}</td></tr>`;
    })
    .join("");
  $("tblParts").querySelector("tbody").innerHTML =
    body +
    `<tr><td><b>Total</b></td><td></td><td></td><td><b>${mass.toFixed(2)} kg</b></td>
      <td><b>$${s[L.S_COST].toFixed(2)}</b></td>
      <td class="why">excludes horns, fasteners, charger and printed parts</td></tr>`;

  // Sensor requirements versus what the catalogue part actually delivers.
  const rp = part(Math.round(s[L.S_RANGER_I]));
  const imu = part(Math.round(s[L.S_IMU_I]));
  const need = {
    range: s[L.S_LOOKAHEAD],
    rate: s[L.S_RATE_HZ],
    res: s[L.S_RES_MM],
  };
  const verdict = (pass) =>
    `<span class="tag ${pass ? "pass" : "fail"}">${pass ? "OK" : "SHORT"}</span>`;
  const contactBus = s[L.S_CONTACT_BUS] > 0.5;

  $("tblSense").querySelector("tbody").innerHTML = [
    [
      `Terrain height ×${rangers}`,
      `${need.range.toFixed(2)} m reach`,
      rp ? rp.name : "—",
      rp ? `${(rp.capacity / 1000).toFixed(1)} m` : "—",
      verdict(rp && rp.capacity / 1000 >= need.range),
    ],
    [
      "…sampled during swing",
      `≥ ${need.rate.toFixed(0)} Hz`,
      rp ? rp.name : "—",
      rp ? `${rp.rating.toFixed(0)} Hz` : "—",
      verdict(rp && rp.rating >= need.rate),
    ],
    [
      "…resolving obstacles",
      `${need.res.toFixed(1)} mm`,
      rp ? rp.name : "—",
      rp ? `±${rp.accuracy.toFixed(0)} mm` : "—",
      verdict(rp && rp.accuracy > 0 && rp.accuracy <= need.res),
    ],
    [
      "Body pitch, roll",
      "control rate",
      imu ? imu.name : "—",
      imu ? `${imu.rating.toFixed(0)} Hz fused` : "—",
      verdict(true),
    ],
    [
      "Stability margin",
      "which feet are loaded",
      contactBus ? "servo bus telemetry" : "6× contact switch or FSR",
      contactBus ? "free with this servo" : "not in this build",
      verdict(contactBus),
    ],
  ]
    .map(
      (r) =>
        `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4]}</td></tr>`
    )
    .join("");

  const resShort = rp && !(rp.accuracy > 0 && rp.accuracy <= need.res);
  $("senseNote").innerHTML =
    (resShort
      ? `<b>The resolution line does not pass.</b> The gait wants ${need.res.toFixed(
          1
        )} mm of height discrimination and a ${rp.name} is about ±${rp.accuracy.toFixed(
          0
        )} mm — worse in sunlight and against dark or angled surfaces. Expect the policy to be noisier on
        the real robot than in simulation: either scale the machine up so the terrain features are larger
        relative to sensor noise, or filter the lookahead over several samples and accept the lag. `
      : "") +
    (contactBus
      ? "Serial-bus servos report load, so the stability-margin input costs nothing extra — a real reason to pay more per servo."
      : "PWM servos report nothing back, so foot contact needs its own sensors; the parts list does not yet include them.");
}

/* ------------------------------------------------------------------ loop */

const traceTorque = new Trace($("cTorque"), [
  { color: COL.ink },
  { color: COL.accent, fill: "rgba(229,57,29,0.10)" },
]);
const traceFoot = new Trace($("cFoot"), [
  { color: COL.ink, fill: "rgba(20,20,22,0.07)" },
]);

let lastFrame = performance.now();
let slowAcc = 0;

function frame(now) {
  const dt = Math.min(0.05, (now - lastFrame) / 1000) * state.timeScale;
  lastFrame = now;

  readKeys();

  if (!state.paused) {
    const t0 = performance.now();
    api.hx_step(dt, state.cmd.fwd, state.cmd.turn);
    state.stepUs += ((performance.now() - t0) * 1000 - state.stepUs) * 0.1;
  }

  if (state.training) {
    const t0 = performance.now();
    api.hx_train(1);
    state.iterMs = performance.now() - t0;
    drawCurve();
    updateTrainingPanel();
    const trained = telemetry();
    if (trained[L.T_TRAINED] > 0.5 && state.mode !== 1) setMode(1);
  }

  const t = telemetry();
  stage.draw(t, L);
  if (!state.paused) {
    falls.push(t[L.T_TIME], t, L, Math.round(t[L.T_LEGS]) || 6);
  }
  drawGaitDiagram(t);

  slowAcc += dt;
  if (slowAcc > 0.06) {
    slowAcc = 0;
    if (!state.paused) {
      traceTorque.push([t[L.T_Q + 1], t[L.T_Q + 2]]);
      traceFoot.push([t[L.T_JOINTS + 10]]);
    }
    traceTorque.draw();
    traceFoot.draw();
    updateReadouts(t);
  }

  requestAnimationFrame(frame);
}

function updateReadouts(t) {
  const secs = t[L.T_TIME];
  $("hClock").textContent =
    `${String(Math.floor(secs / 60)).padStart(2, "0")}:${fmt(secs % 60, 1).padStart(4, "0")}` +
    (state.timeScale === 1 ? "" : ` · ${state.timeScale}×`);
  $("hCourse").textContent = courseName(state.courseKind);
  const hinges = Math.round(t[L.T_N_HINGES] || state.legs * 3);
  $("hSolver").textContent =
    t[L.T_PLANT] > 0.5 ? `RAPIER ${hinges}-REV` : "CENTROIDAL IK";

  const fallen = t[L.T_FALLEN] > 0.5;
  const broken = t[L.T_BROKEN] > 0.5;
  const airborne = t[L.T_AIRBORNE] > 0.5;
  const blocked = t[L.T_BLOCKED] > 0.5;
  const moving = Math.abs(state.cmd.fwd) > 0.01 || Math.abs(state.cmd.turn) > 0.01;
  $("hState").textContent = broken
    ? "BROKEN"
    : fallen
    ? "DEAD"
    : airborne
    ? t[L.T_TASK] > 0.5
      ? "JUMPING"
      : "AIRBORNE"
    : blocked
    ? "BLOCKED"
    : state.paused
    ? "HELD"
    : moving
    ? state.cmd.turn !== 0
      ? "TURNING"
      : "WALKING"
    : "STANDING";
  const oneleg = L.T_ONELEG != null && t[L.T_ONELEG] > 0.5;
  const hudWalk = $("hudWalk");
  const hudDrill = $("hudDrill");
  if (hudWalk) hudWalk.hidden = oneleg;
  if (hudDrill) hudDrill.hidden = !oneleg;
  if (oneleg) {
    const phases = ["SETTLE", "LIFT", "SHIFT", "PLACE", "PAUSE"];
    const movingLeg = Math.round(t[L.T_MOVE_LEG]);
    const phaseName = phases[Math.round(t[L.T_MOVE_PHASE])] || "ONE-LEG";
    $("hudOlLeg").textContent = legNames()[movingLeg] || String(movingLeg);
    $("hudOlPhase").textContent = phaseName;
    $("hudOlMove").textContent = String(Math.round(t[L.T_MOVE_I]));
    $("hudOlClear").textContent = fmt(t[L.T_FOOT_CLEAR], 2);
    $("hudOlDrift").textContent = fmt(t[L.T_STANCE_DRIFT], 2);
    $("hudOlXz").textContent = fmt(t[L.T_CHASSIS_XZ], 2);
    $("hState").textContent = phaseName;
    $("hudGait").textContent = "ONE LEG";
    const callout = $("drillCallout");
    const calloutLine = $("drillCalloutLine");
    if (callout) callout.hidden = false;
    if (calloutLine) {
      calloutLine.textContent = `${legNames()[movingLeg] || "L1"} · ${phaseName} · the red leg is the only one commanded`;
    }
  } else {
    const callout = $("drillCallout");
    if (callout) callout.hidden = true;
  }
  $("banner").textContent = broken ? "BROKEN" : "DEAD";
  $("banner").dataset.on = String(fallen || broken);

  $("hudGait").textContent =
    state.mode === 2 ? "ONE LEG" : state.mode === 1 ? "LEARNED" : presetName(state.preset).toUpperCase();
  $("hudDuty").textContent = fmt(t[L.T_DUTY_NOW], 2);
  $("hudY").textContent = (t[L.T_POS + 1] >= 0 ? "+" : "") + fmt(t[L.T_POS + 1], 2);
  $("hudPhase").textContent = fmt(t[L.T_PHASE], 2);
  $("hudMargin").textContent = fmt(t[L.T_MARGIN], 2);
  $("hudV").textContent = fmt(Math.hypot(t[L.T_VEL], t[L.T_VEL + 1]), 2);
  $("hudVc").textContent = fmt(t[L.T_CMD_SPEED], 2);
  $("hudVx").textContent = signed(t[L.T_VEL], 2);
  $("hudVy").textContent = signed(t[L.T_VY], 2);
  $("hudVz").textContent = signed(t[L.T_VEL + 1], 2);
  $("hudSlip").textContent = fmt(t[L.T_SLIP_RATE] > 0 ? t[L.T_SLIP_RATE] / (1 / 100) : 0, 2);
  const head = (t[L.T_YAW] * 180) / Math.PI;
  $("hudHead").textContent = `${head >= 0 ? "+" : ""}${head.toFixed(0)}°`;
  $("hudW").textContent = fmt(t[L.T_STEER] * 1.1, 2);
  // Where it is being asked to go, and whether it is choosing that itself.
  const nav = t[L.T_NAV] > 0.5 && Math.abs(state.cmd.turn) < 0.02;
  $("hudNav").textContent = nav ? "AUTO" : "MANUAL";
  const hit = stage.contact;
  const colliding = hit && (hit.blocked || hit.chassis || hit.legs > 0);
  const hudHit = $("hudHit");
  if (colliding) {
    const bits = [];
    if (hit.blocked || hit.chassis) bits.push("CHASSIS");
    if (hit.legs) bits.push(`${hit.legs} LEG${hit.legs === 1 ? "" : "S"}`);
    hudHit.textContent = `HIT · ${bits.join(" · ")}`;
  } else {
    hudHit.textContent = "";
  }
  hudHit.classList.toggle("hot", !!colliding);
  $("hudWp").textContent = `${Math.round(t[L.T_WP_I]) + 1}/${Math.round(t[L.T_WP_N])}`;
  $("hudWpD").textContent = fmt(t[L.T_WP_DIST], 1);
  const brg = (t[L.T_BEARING] * 180) / Math.PI;
  $("hudBrg").textContent = `${brg >= 0 ? "+" : ""}${brg.toFixed(0)}°`;
  $("pReached").textContent = Math.round(t[L.T_REACHED]);
  $("navNote").textContent = `${Math.round(t[L.T_WP_N])} waypoints · ±${fmt(t[L.T_WALL_X], 1)} m`;
  // How close the nearer of the two walls is to the chassis rim.
  const room = t[L.T_WALL_X] - Math.abs(t[L.T_POS]) - t[L.T_BODY_R];
  $("pWall").textContent = fmt(Math.max(0, room), 2);
  $("pWall").classList.toggle("hot", room < 0.15);
  $("hudSag").textContent = fmt(t[L.T_DROOP] * simScale() * 1000, 1);
  $("hudCycle").textContent = fmt(t[L.T_CYCLE_NOW], 3);
  $("hudStride").textContent = fmt(t[L.T_STRIDE_NOW], 2);

  const nLegs = state.legs;
  for (let i = 0; i < nLegs; i++) {
    const el = $(`load${i}`);
    if (!el) continue;
    el.style.width = `${Math.min(100, t[L.T_LOAD + i] * nLegs * 32)}%`;
    // "Carrying more than its share" is relative to how many legs there are.
    el.classList.toggle("warn", t[L.T_LOAD + i] > 2 / nLegs);
  }

  drawDial($("dSolver"), state.stepUs, 900, `${Math.round(state.stepUs)}`, state.stepUs > 700, "\u00B5s", 700 / 900);
  const margin = t[L.T_MARGIN];
  drawDial($("dStab"), Math.max(0, margin), 0.7, margin.toFixed(2), margin < 0.08, "m", 0.08 / 0.7);

  const target = t[L.T_CMD_SPEED] || 4;
  const v = Math.hypot(t[L.T_VEL], t[L.T_VEL + 1]);
  $("mSpeedLabel").textContent = isJump()
    ? `Speed / commanded · ${Math.round(t[L.T_JUMPS] || 0)} jumps`
    : "Speed / commanded";
  $("mSpeed").textContent = `${fmt(v, 2)} / ${fmt(target, 1)} m/s`;
  $("fSpeed").style.width = `${Math.min(100, (v / target) * 100)}%`;

  // How much of the friction budget the gait is spending. 100% means the feet
  // are on the point of letting go; above it they are skidding.
  const used = Math.min(t[L.T_TRACTION], 9.99);
  $("mTrac").textContent = `${Math.round(used * 100)}%`;
  $("fTrac").style.width = `${Math.min(100, used * 100)}%`;
  $("mTrac").classList.toggle("hot", used > 1);

  const load = t[L.T_SERVO_LOAD];
  $("mServo").textContent = `${Math.round(Math.min(999, load * 100))}%`;
  $("fServo").style.width = `${Math.min(100, load * 100)}%`;
  $("mServo").classList.toggle("hot", load > 1);

  $("mStub").textContent = fmt(t[L.T_STUB], 2);
  $("fStub").style.width = `${Math.min(100, t[L.T_STUB] * 8)}%`;

  $("pSlip").textContent = fmt(t[L.T_SLIP], 2);
  $("pDroop").textContent = fmt(t[L.T_DROOP] * simScale() * 1000, 1);
  $("pLag").textContent = fmt((t[L.T_SERVO_LAG] * 180) / Math.PI, 1);
  const cot = t[L.T_COT];
  $("pCot").textContent = cot > 0 ? fmt(cot, 2) : "—";
  // What the leg costs to carry and swing, separate from the load underfoot.
  $("pLegTq").textContent = fmt(t[L.T_LEG_TORQUE] * 10.1972, 2);
  $("pReact").textContent = fmt(
    Math.hypot(t[L.T_LEG_REACT], t[L.T_LEG_REACT + 2]),
    2
  );
}

function updateTrainingPanel() {
  const t = telemetry();
  const base = t[L.T_BASE_R];
  const best = t[L.T_BEST_R];
  $("sBase").textContent = fmt(base, 1);
  $("sBest").textContent = fmt(best, 1);
  $("sIter").textContent = Math.round(t[L.T_ITER]);
  $("sIterMs").textContent = `${state.iterMs.toFixed(0)} ms each`;
  $("sRoll").textContent = Math.round(t[L.T_ROLLOUTS]);
  $("sFeed").textContent = fmt(t[L.T_FEEDBACK], 2);
  const gain = base !== 0 ? ((best - base) / Math.abs(base)) * 100 : 0;
  $("sGain").textContent = t[L.T_ITER] > 0 ? `${gain >= 0 ? "+" : ""}${gain.toFixed(0)}%` : "—";
  $("trStatus").textContent = state.training
    ? `training · ${state.iterMs.toFixed(0)} ms/iteration`
    : t[L.T_ITER] > 0
    ? "stopped"
    : "idle";
  const trained = t[L.T_TRAINED] > 0.5;
  $("btnLearn").disabled = !trained;
  $("policyNote").textContent = trained
    ? "Switch between the hand-tuned gait and the learned policy on the same course."
    : "Train a policy to enable the comparison.";
  if (!trained && state.mode === 1) setMode(0);
  refreshGaitTable();
}

/* ------------------------------------------------------------------ input */

function readKeys() {
  const k = state.keys;
  if (k.size === 0) return;
  let f = 0;
  let tn = 0;
  if (k.has("w")) f = 1;
  if (k.has("s")) f = -1;
  if (k.has("q")) tn = 1;
  if (k.has("e")) tn = -1;
  if (k.has("w") || k.has("s") || k.has("q") || k.has("e")) {
    state.cmd.fwd = f;
    state.cmd.turn = tn;
  }
}

function setRate(x) {
  state.timeScale = Math.min(4, Math.max(0.25, Math.round(x * 4) / 4));
  $("rRate").value = state.timeScale;
  $("vRate").textContent = `${state.timeScale}×`;
}

function wire() {
  document.querySelectorAll(".tab").forEach((b) =>
    b.addEventListener("click", () => setTab(b.dataset.tab))
  );

  $("btnPause").addEventListener("click", () => {
    state.paused = !state.paused;
    $("btnPause").dataset.on = String(state.paused);
    $("btnPause").textContent = state.paused ? "Resume" : "Pause";
    log(state.paused ? "sim.pause()" : "sim.resume()");
  });
  $("rRate").addEventListener("input", () => setRate(+$("rRate").value));
  $("btnReset").addEventListener("click", () => {
    api.hx_reset_live();
    log("sim.reset()");
  });

  $("btnBase").addEventListener("click", () => setMode(0));
  $("btnLearn").addEventListener("click", () => setMode(1));
  $("btnOneleg").addEventListener("click", () => setMode(2));
  $("btnWalk").addEventListener("click", () => setMode(0));

  $("btnTrain").addEventListener("click", () => {
    state.training = !state.training;
    $("btnTrain").dataset.on = String(state.training);
    $("btnTrain").textContent = state.training ? "Stop" : "Train";
    log(state.training ? "ars.start()" : "ars.stop()");
    updateTrainingPanel();
  });
  $("btnResetTrain").addEventListener("click", () => {
    state.training = false;
    $("btnTrain").dataset.on = "false";
    $("btnTrain").textContent = "Train";
    api.hx_reset_training();
    setMode(0);
    drawCurve();
    updateTrainingPanel();
    log("ars.reset()");
  });

  const cfg = () => {
    const dirs = +$("rDirs").value;
    const top = Math.min(+$("rTop").value, dirs);
    $("rTop").max = dirs;
    api.hx_set_train_cfg(dirs, top, +$("rAlpha").value, +$("rSigma").value, +$("rHorizon").value);
    $("vDirs").textContent = dirs;
    $("vTop").textContent = top;
    $("vAlpha").textContent = (+$("rAlpha").value).toFixed(3);
    $("vSigma").textContent = (+$("rSigma").value).toFixed(3);
    $("vHorizon").textContent = `${(+$("rHorizon").value).toFixed(1)} s`;
  };
  ["rDirs", "rTop", "rAlpha", "rSigma", "rHorizon"].forEach((id) =>
    $(id).addEventListener("input", cfg)
  );
  cfg();

  document.addEventListener("click", (e) => {
    const ol = e.target.closest("[data-oneleg]");
    if (ol) {
      state.onelegLeg = +ol.dataset.oneleg;
      document
        .querySelectorAll("[data-oneleg]")
        .forEach((b) => (b.dataset.on = String(+b.dataset.oneleg === state.onelegLeg)));
      api.hx_set_oneleg_leg(state.onelegLeg);
      if (state.mode !== 2) setMode(2);
      else log(`drill.leg("${legNames()[state.onelegLeg]}")`);
    }
    const pb = e.target.closest("[data-preset]");
    if (pb) {
      state.preset = +pb.dataset.preset;
      api.hx_set_preset(state.preset);
      document
        .querySelectorAll("[data-preset]")
        .forEach((b) => (b.dataset.on = String(+b.dataset.preset === state.preset)));
      setMode(0);
      syncSliders();
      drawCurve();
      updateTrainingPanel();
      log(`gait.set("${PRESETS[state.preset].toLowerCase()}")`);
    }
    const cb = e.target.closest("[data-course]");
    if (cb) {
      state.courseKind = +cb.dataset.course;
      applyCourse();
    }
    const cmd = e.target.closest("[data-cmd]");
    if (cmd) {
      const c = cmd.dataset.cmd;
      const map = {
        fwd: { fwd: 1, turn: 0 },
        back: { fwd: -1, turn: 0 },
        stop: { fwd: 0, turn: 0 },
        left: { fwd: 0.35, turn: 1 },
        right: { fwd: 0.35, turn: -1 },
        turnL: { fwd: 0, turn: 1 },
        turnR: { fwd: 0, turn: -1 },
      };
      state.cmd = { ...map[c] };
      document
        .querySelectorAll("[data-cmd]")
        .forEach((b) => (b.dataset.on = String(b.dataset.cmd === c)));
      log(`cmd.move("${c}")`);
    }
  });

  $("btnNav").addEventListener("click", () => {
    const on = $("btnNav").dataset.on !== "true";
    $("btnNav").dataset.on = String(on);
    api.hx_set_nav(on ? 1 : 0);
    log(`nav.follow(${on})`);
  });
  $("btnRoute").addEventListener("click", () => {
    stage.showRoute = !stage.showRoute;
    $("btnRoute").dataset.on = String(stage.showRoute);
  });

  const syncCam = (mode) => {
    $("hudCam").textContent = mode.toUpperCase();
    $("btnCamOrbit").dataset.on = String(mode === "orbit");
    $("btnCamTop").dataset.on = String(mode === "top");
    $("btnCamSide").dataset.on = String(mode === "side");
  };
  stage.onView = syncCam;
  const setCam = (mode) => {
    stage.setView(mode);
    log(`cam.set("${mode}")`);
  };
  $("btnCamOrbit").addEventListener("click", () => setCam("orbit"));
  $("btnCamTop").addEventListener("click", () => setCam("top"));
  $("btnCamSide").addEventListener("click", () => setCam("side"));

  $("rSeed").addEventListener("input", () => {
    state.seed = +$("rSeed").value;
    $("vSeed").textContent = state.seed;
    applyCourse();
  });
  $("btnReroll").addEventListener("click", () => {
    state.seed = (state.seed % 60) + 1;
    $("rSeed").value = state.seed;
    $("vSeed").textContent = state.seed;
    applyCourse();
  });

  $("rLegs").addEventListener("input", () => {
    const want = +$("rLegs").value;
    const got = api.hx_set_legs(want);
    if (!got) return;
    state.legs = got;
    afterMachineChange();
    // The frame decides which preset can hold it up.
    state.preset = api.hx_preset();
    buildLegUI();
    refreshPresetButtons();
    describeMachine();
    $("vLegs").textContent = `${got} legs`;
    log(`frame.legs(${got})`);
  });

  $("rCruise").addEventListener("input", () => {
    state.cruise = +$("rCruise").value;
    api.hx_set_cruise(state.cruise);
    syncCruiseLabels();
  });

  $("selServo").addEventListener("change", () => {
    const i = +$("selServo").value;
    state.servo = i;
    // Changing the machine invalidates anything learned for the old one.
    if (api.hx_set_servo(i < 0 ? 0xffffffff : i)) afterMachineChange();
    describeMachine();
    log(`sim.servo("${i < 0 ? "generic 20 kg-cm" : CATALOGUE.servos[i].part}")`);
  });

  const hw = () => {
    state.build.mass = +$("rMass").value;
    state.build.femurMm = +$("rScale").value;
    state.build.safety = +$("rSafety").value;
    $("vMass").textContent = `${state.build.mass.toFixed(1)} kg`;
    $("vScale").textContent = `${state.build.femurMm} mm`;
    $("vSafety").textContent = `${state.build.safety.toFixed(2)}×`;
    updateHardware();
    describeMachine();
  };
  ["rMass", "rScale", "rSafety"].forEach((id) => $(id).addEventListener("input", hw));

  const sz = () => {
    state.sizing.chassis = +$("rChassis").value;
    state.sizing.runtime = +$("rRuntime").value;
    $("vChassis").textContent = `${state.sizing.chassis.toFixed(2)} kg`;
    $("vRuntime").textContent = `${state.sizing.runtime} min`;
    updateSystem();
  };
  ["rChassis", "rRuntime"].forEach((id) => $(id).addEventListener("input", sz));

  $("tblSystem").addEventListener("click", (e) => {
    const row = e.target.closest("[data-servo]");
    if (!row) return;
    state.pick = +row.dataset.servo;
    state.pickLocked = true;
    updateSystem();
    log(`build.servo("${CATALOGUE.servos[state.pick].part}")`);
  });

  addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if ("wsqe".includes(k) && k.length === 1) {
      state.keys.add(k);
      e.preventDefault();
    }
    if (k === "x") state.cmd = { fwd: 0, turn: 0 };
    if (k === " ") {
      $("btnPause").click();
      e.preventDefault();
    }
    if (k === "[" || k === "]") {
      setRate(state.timeScale * (k === "]" ? 2 : 0.5));
      e.preventDefault();
    }
    const typing =
      e.target &&
      (e.target.tagName === "SELECT" ||
        (e.target.tagName === "INPUT" && e.target.type !== "range"));
    if (!typing && (k === "1" || k === "2" || k === "3")) {
      setCam(k === "1" ? "orbit" : k === "2" ? "top" : "side");
      e.preventDefault();
    }
  });
  addEventListener("keyup", (e) => state.keys.delete(e.key.toLowerCase()));
  addEventListener("resize", () => {
    drawCurve();
    if ($("cProfile").getBoundingClientRect().width) drawProfile();
  });
}

function applyCourse() {
  api.hx_set_course(state.courseKind, state.seed);
  readCourse();
  document
    .querySelectorAll("[data-course]")
    .forEach((b) => (b.dataset.on = String(+b.dataset.course === state.courseKind)));
  state.training = false;
  $("btnTrain").dataset.on = "false";
  $("btnTrain").textContent = "Train";
  setMode(0);
  const name = courseName(state.courseKind);
  $("trCourse").textContent = name;
  $("tSummary").textContent =
    `${name} · seed ${state.seed} · ${api.hx_course_len()} obstacles · ${api.hx_route_len()} waypoints`;
  $("tNote").textContent = COURSE_NOTES[name] || "";
  // The command dial is a speed. JUMP samples a faster band because the
  // trenches are a running jump.
  const lo = api.hx_cruise_lo();
  const hi = api.hx_cruise_hi();
  const r = $("rCruise");
  r.min = lo;
  r.max = hi;
  r.step = 0.1;
  state.cruise = isJump() ? 4.5 : 4.0;
  r.value = state.cruise;
  api.hx_set_cruise(state.cruise);
  syncCruiseLabels();
  drawProfile();
  drawCurve();
  updateTrainingPanel();
  updateHardware();
  log(`course.set("${name.toLowerCase()}", seed=${state.seed})`);
}

function syncCruiseLabels() {
  const lo = api.hx_cruise_lo();
  const hi = api.hx_cruise_hi();
  $("cruiseTitle").textContent = "Commanded speed";
  $("cruiseHold").textContent = "Hold this cruise";
  $("vCruise").textContent = `${state.cruise.toFixed(1)} m/s`;
  $("cruiseNote").textContent = `trained over ${lo.toFixed(1)}–${hi.toFixed(1)} m/s`;
  $("cruiseHelp").textContent = isJump()
    ? "The reward is still speed tracking. The trenches are wider than a stride, so the only way to hold the command is to jump them — and land without stripping the servos. The seed jumps when it sees a pit; Train is how it gets further."
    : "The reward is speed tracking, not distance, and the command is an input to the policy. Move it and watch the learned gait change its cycle time, stride and duty factor to keep up — the hand-tuned one cannot, because it has no feedback layer.";
}

/* ------------------------------------------------------------------ boot */

async function boot() {
  const { instance } = await WebAssembly.instantiate(decodeBase64(window.HX_WASM_B64), {});
  api = instance.exports;
  api.hx_init(1);

  stage = new window.Stage($("view"));
  buildStaticUI();
  wire();
  readCourse();
  syncSliders();

  $("tSummary").textContent =
    `MIXED · seed 1 · ${api.hx_course_len()} obstacles · ${api.hx_route_len()} waypoints`;
  $("tNote").textContent = COURSE_NOTES.MIXED;
  refreshGaitTable();
  updateHardware();
  updateSystem();
  updateTrainingPanel();
  describeMachine();
  setMode(2);

  /* Hooks for the end-to-end test, which drives the real page rather than
   * reimplementing any of it. Nothing else uses them. */
  window.__hxFalls = falls;
  window.__hxOneleg = () => {
    const t = telemetry();
    const legs = Math.round(t[L.T_LEGS]) || 6;
    const stance = [];
    for (let i = 0; i < legs; i++) stance.push(t[L.T_STANCE + i] > 0.5 ? 1 : 0);
    const phases = ["settle", "lift", "shift", "place", "pause"];
    return {
      on: L.T_ONELEG != null && t[L.T_ONELEG] > 0.5,
      moving: Math.round(t[L.T_MOVE_LEG] || 0),
      phase: Math.round(t[L.T_MOVE_PHASE] || 0),
      phaseName: phases[Math.round(t[L.T_MOVE_PHASE] || 0)] || "—",
      clear: t[L.T_FOOT_CLEAR] || 0,
      drift: t[L.T_STANCE_DRIFT] || 0,
      chassis: t[L.T_CHASSIS_XZ] || 0,
      stance,
      policy: $("hPolicy").textContent,
      state: $("hState").textContent,
      course: $("hCourse").textContent,
    };
  };
  // Advance the actual WASM plant deterministically without waiting for wall
  // time. The smoke suite still verifies the animation loop separately; long
  // physics assertions use this to avoid sleeping through simulated seconds.
  window.__hxStepSamples = (count) => {
    const n = Math.min(1200, Math.max(0, Math.floor(count)));
    let t = telemetry();
    let hull = { n: 0, span: Infinity };
    let speed = { closest: 0, error: Infinity };
    for (let i = 0; i < n; i++) {
      api.hx_step(1 / 60, state.cmd.fwd, state.cmd.turn);
      t = telemetry();
      const last = falls.t[falls.t.length - 1];
      // A fallen run repeats its final instant until recovery. The real-time
      // loop gets one duplicate per frame; a tight test loop can get hundreds.
      if (last === undefined || t[L.T_TIME] !== last) {
        falls.push(t[L.T_TIME], t, L, Math.round(t[L.T_LEGS]) || 6);
      }
      const hullN = Math.round(t[L.T_HULL_N]);
      let hullSpan = 0;
      for (let point = 0; point < hullN; point++) {
        hullSpan = Math.max(
          hullSpan,
          Math.hypot(
            t[L.T_HULL + point * 2] - t[L.T_POS],
            t[L.T_HULL + point * 2 + 1] - t[L.T_POS + 2]
          )
        );
      }
      if (hullN >= 3 && hullSpan < hull.span) hull = { n: hullN, span: hullSpan };
      const measuredSpeed = Math.hypot(t[L.T_VEL], t[L.T_VEL + 1]);
      const speedError = Math.abs(measuredSpeed - state.cruise);
      if (speedError < speed.error) speed = { closest: measuredSpeed, error: speedError };
    }
    updateReadouts(t);
    drawGaitDiagram(t);
    return {
      samples: falls.t.length,
      time: t[L.T_TIME],
      reached: t[L.T_REACHED],
      cycle: falls.cycle(),
      kind: falls.classify(),
      hull,
      speed,
    };
  };
  window.__hxRoute = () => api.hx_route_len();
  window.__hxDuty = () => {
    const t = telemetry();
    return { duty: t[L.T_DUTY_NOW], cycle: t[L.T_CYCLE_NOW] };
  };
  // Furthest support-polygon vertex from the drawn chassis: hull is Rapier
  // feet when the plant is live, so this is a same-frame span check.
  window.__hxHullSpan = () => {
    const t = telemetry();
    const n = Math.round(t[L.T_HULL_N]);
    let m = 0;
    for (let i = 0; i < n; i++) {
      m = Math.max(
        m,
        Math.hypot(t[L.T_HULL + i * 2] - t[L.T_POS], t[L.T_HULL + i * 2 + 1] - t[L.T_POS + 2])
      );
    }
    return { n, span: m };
  };
  window.__hxSway = () => {
    const n = api.hx_route_len();
    const r = new Float32Array(api.memory.buffer, api.hx_route_ptr(), n * 2);
    let m = 0;
    for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(r[i * 2]));
    return m;
  };
  window.__ready = true;

  log(`boot: wasm ready, ${state.legs * 3} DOF, analytic IK`);
  log('cmd.move("f") — WASD / QE / X / SPACE / 1–3');
  requestAnimationFrame(frame);
}

boot().catch((e) => {
  document.body.innerHTML = `<pre style="padding:24px;color:#c0341c">Failed to start: ${e}</pre>`;
});
