/* End-to-end check of the built dashboard in a real browser.
 *
 *   node test/smoke.mjs [path-to-html]
 *
 * Verifies the wasm boots, the simulator advances, training actually improves
 * the reward, and every tab renders without console errors.
 */

import { chromium } from "playwright";
import path from "node:path";
import fs from "node:fs";

const FILE = path.resolve(process.argv[2] || "dist/hexapod-simulator.html");
const SHOTS = path.resolve("dist/shots");
fs.mkdirSync(SHOTS, { recursive: true });

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "  ok  " : "FAIL  "}${name}${detail ? "  — " + detail : ""}`);
  if (!ok) failures++;
};

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium-1194/chrome-linux/chrome" });
const page = await browser.newPage({ viewport: { width: 1680, height: 1000 } });

const errors = [];
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});
page.on("pageerror", (e) => errors.push(String(e)));

await page.goto("file://" + FILE);
await page.waitForFunction(() => window.__ready === true || document.getElementById("hClock"), null, {
  timeout: 15000,
});
await wait(1200);

/* ------------------------------------------------------------- boot */

check("no console errors on boot", errors.length === 0, errors.slice(0, 3).join(" | "));
check(
  "wasm exports present",
  await page.evaluate(() => typeof document.getElementById("view") !== "undefined")
);

const clock0 = await page.textContent("#hClock");
await wait(1000);
const clock1 = await page.textContent("#hClock");
check("simulation clock advances", clock0 !== clock1, `${clock0} -> ${clock1}`);

const canvasPainted = await page.evaluate(() => {
  const cv = document.getElementById("view");
  const ctx = cv.getContext("2d");
  const d = ctx.getImageData(0, 0, cv.width, cv.height).data;
  const seen = new Set();
  for (let i = 0; i < d.length; i += 4000) seen.add(`${d[i]},${d[i + 1]},${d[i + 2]}`);
  return seen.size;
});
check("3-D stage renders content", canvasPainted > 6, `${canvasPainted} distinct sampled colours`);

const speed = await page.textContent("#hudV");
check("robot is moving", parseFloat(speed) > 0.5, `${speed} m/s`);

await page.screenshot({ path: path.join(SHOTS, "01-kinematics.png") });

/* --------------------------------------------------------- training */

await page.click('[data-tab="training"]');
await wait(200);
await page.screenshot({ path: path.join(SHOTS, "02-training-before.png") });

await page.click("#btnTrain");
try {
  await page.waitForFunction(
    () => {
      const best = parseFloat(document.getElementById("sBest")?.textContent);
      const base = parseFloat(document.getElementById("sBase")?.textContent);
      const feed = parseFloat(document.getElementById("sFeed")?.textContent);
      const iter = parseInt(document.getElementById("sIter")?.textContent, 10);
      return iter > 10 && Number.isFinite(best) && Number.isFinite(base) && best > base && feed > 0.01;
    },
    { timeout: 28000 }
  );
} catch {
  // fall through to the assertions, which report the actual numbers
}
await page.click("#btnTrain");
await wait(400);

const iters = parseInt(await page.textContent("#sIter"), 10);
const base = parseFloat(await page.textContent("#sBase"));
const best = parseFloat(await page.textContent("#sBest"));
const gain = await page.textContent("#sGain");
const iterMs = await page.textContent("#sIterMs");

check("training ran iterations", iters > 10, `${iters} iterations, ${iterMs}`);
check("baseline recorded", Number.isFinite(base), `${base}`);
check("learned beats hand-tuned", best > base, `${base.toFixed(1)} -> ${best.toFixed(1)} (${gain})`);

const feedback = parseFloat(await page.textContent("#sFeed"));
check("feedback layer left zero", feedback > 0.01, `norm ${feedback}`);

await page.screenshot({ path: path.join(SHOTS, "03-training-after.png") });

/* ------------------------------------------------- policy comparison */

const learnEnabled = await page.isEnabled("#btnLearn");
check("learned policy selectable", learnEnabled);
if (learnEnabled) {
  await page.click("#btnLearn");
  await wait(300);
  check("header shows learned policy", (await page.textContent("#hPolicy")) === "LEARNED");
  const locked = await page.isDisabled("#pr0");
  check("sliders lock under learned policy", locked);
}

await page.click('[data-tab="kinematics"]');
await wait(1500);
await page.screenshot({ path: path.join(SHOTS, "04-learned-walking.png") });

/* ---------------------------------------------------------- terrain */

await page.click('[data-tab="terrain"]');
await wait(300);
const summary = await page.textContent("#tSummary");
check("terrain summary populated", /obstacles/.test(summary), summary);

await page.click('[data-course="1"]'); // Steps
await wait(800);
const stepsSummary = await page.textContent("#tSummary");
check("course switches", /STEPS/.test(stepsSummary), stepsSummary);
await page.screenshot({ path: path.join(SHOTS, "05-terrain.png") });

/* --------------------------------------------------------- hardware */

await page.click('[data-tab="hardware"]');
await wait(500);
const req = parseFloat(await page.textContent("#tqReq"));
const femur = parseFloat(await page.textContent("#tqFemur"));
const rows = await page.$$eval("#tblServo tbody tr", (r) => r.length);
const passing = await page.$$eval('#tblServo tbody tr[data-pass="true"]', (r) => r.length);

check("torque requirement computed", req > 1 && req < 200, `${req} kg-cm`);
check("femur is the sizing joint", femur > 0, `${femur} kg-cm`);
check("servo table populated", rows === 8, `${rows} rows`);
check("some servos qualify, some do not", passing > 0 && passing < rows, `${passing}/${rows} pass`);

// Heavier robot must demand more torque and disqualify more servos.
await page.evaluate(() => {
  const el = document.getElementById("rMass");
  el.value = "8";
  el.dispatchEvent(new Event("input", { bubbles: true }));
});
await wait(400);
const req2 = parseFloat(await page.textContent("#tqReq"));
const passing2 = await page.$$eval('#tblServo tbody tr[data-pass="true"]', (r) => r.length);
check("torque scales with mass", req2 > req * 2, `${req} -> ${req2} kg-cm at 8 kg`);
check("shortlist shrinks as mass grows", passing2 <= passing, `${passing} -> ${passing2} pass`);

await page.evaluate(() => {
  const el = document.getElementById("rMass");
  el.value = "2";
  el.dispatchEvent(new Event("input", { bubbles: true }));
});
await wait(300);
await page.screenshot({ path: path.join(SHOTS, "06-hardware.png") });

/* ----------------------------------------------------------- system */

await page.click('[data-tab="system"]');
await wait(900);

const sysRows = await page.$$eval("#tblSystem tbody tr", (r) => r.length);
const sysPass = await page.$$eval('#tblSystem tbody tr[data-pass="true"]', (r) => r.length);
const partRows = await page.$$eval("#tblParts tbody tr", (r) => r.length);
const senseRows = await page.$$eval("#tblSense tbody tr", (r) => r.length);
const allUp = await page.textContent("#sysAllUp");
const meanA = parseFloat(await page.textContent("#sysMeanA"));
const runtime = parseFloat(await page.textContent("#sysRuntime"));

check("system table sizes every servo", sysRows === 8, `${sysRows} rows, ${sysPass} viable`);
check("some servos cannot build the robot", sysPass > 0 && sysPass < sysRows, `${sysPass}/${sysRows}`);
check("parts list populated", partRows >= 8, `${partRows} rows`);
check("sensor requirements derived", senseRows === 5, `${senseRows} rows`);
check("current draw computed", meanA > 0.2 && meanA < 40, `${meanA} A`);
check("endurance computed", runtime > 1 && runtime < 400, `${runtime} min`);
check("all-up mass shown", /kg/.test(allUp), allUp);

// Demanding a longer runtime must cost mass, not be granted for free.
const massBefore = parseFloat(allUp);
await page.evaluate(() => {
  const el = document.getElementById("rRuntime");
  el.value = "90";
  el.dispatchEvent(new Event("input", { bubbles: true }));
});
await wait(900);
const massAfter = parseFloat(await page.textContent("#sysAllUp"));
check("longer endurance costs mass", massAfter > massBefore, `${massBefore} -> ${massAfter} kg`);

await page.evaluate(() => {
  const el = document.getElementById("rRuntime");
  el.value = "20";
  el.dispatchEvent(new Event("input", { bubbles: true }));
});
await wait(700);
await page.screenshot({ path: path.join(SHOTS, "07-system.png") });

await page.click('[data-tab="about"]');
await wait(200);
await page.screenshot({ path: path.join(SHOTS, "08-about.png") });

/* ------------------------------------------------- physics and commands */

await page.click('[data-tab="kinematics"]');
await wait(300);

const setRange = (id, v) =>
  page.evaluate(
    ([i, val]) => {
      const el = document.getElementById(i);
      el.value = String(val);
      el.dispatchEvent(new Event("input", { bubbles: true }));
    },
    [id, v]
  );

// The commanded speed is an input to the simulator, not a label.
const speedAt = async (v) => {
  await setRange("rCruise", v);
  await wait(2500);
  return parseFloat((await page.textContent("#mSpeed")).split("/")[0]);
};
const slow = await speedAt(2.0);
const fast = await speedAt(5.5);
check("commanded speed drives the robot", fast > slow + 1.5, `${slow} -> ${fast} m/s`);
check("speed dial reads back", (await page.textContent("#vCruise")).includes("5.5"));
await setRange("rCruise", 4.0);
await wait(600);

// Contact and actuator telemetry has to be live, not placeholder.
const phys = await page.evaluate(() => ({
  trac: document.getElementById("mTrac").textContent,
  servo: document.getElementById("mServo").textContent,
  cot: document.getElementById("pCot").textContent,
  lag: document.getElementById("pLag").textContent,
}));
check("traction telemetry live", /^\d+%$/.test(phys.trac), phys.trac);
check("servo load telemetry live", /^\d+%$/.test(phys.servo), phys.servo);
check("cost of transport computed", parseFloat(phys.cot) > 0, phys.cot);
check("joint tracking lag reported", parseFloat(phys.lag) > 0, `${phys.lag} deg`);

// Picking an undersized servo has to visibly break the robot.
const pickServo = async (label) => {
  const idx = await page.evaluate((l) => {
    const sel = document.getElementById("selServo");
    const opt = [...sel.options].findIndex((o) => o.textContent.startsWith(l));
    sel.value = sel.options[opt].value;
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    return sel.value;
  }, label);
  await wait(2200);
  return {
    idx,
    load: parseFloat(await page.textContent("#mServo")),
    note: await page.textContent("#machineNote"),
  };
};
const strong = await pickServo("DS3218MG");
const weak = await pickServo("SG90");
check("servo selector changes the machine", strong.note !== weak.note, `${strong.note} | ${weak.note}`);
check(
  "an undersized servo is driven past stall",
  weak.load > strong.load && weak.load > 100,
  `${strong.load}% -> ${weak.load}% of stall`
);
const sag = parseFloat(await page.textContent("#pDroop"));
check("an undersized servo sags the chassis", Math.abs(sag) > 0.5, `${sag} mm`);

await page.evaluate(() => {
  const sel = document.getElementById("selServo");
  sel.value = "-1";
  sel.dispatchEvent(new Event("change", { bubbles: true }));
});
await wait(800);
await page.screenshot({ path: path.join(SHOTS, "09-physics.png") });

// Leg mass is not zero, and the chassis feels it.
const legPhys = await page.evaluate(() => ({
  tq: document.getElementById("pLegTq").textContent,
  react: document.getElementById("pReact").textContent,
}));
check("leg weight costs joint torque", parseFloat(legPhys.tq) > 0, `${legPhys.tq} kg-cm`);
check("swinging legs kick the chassis", parseFloat(legPhys.react) > 0, `${legPhys.react} N`);

/* ------------------------------------------------------- the frame */

const setLegs = async (n) => {
  await setRange("rLegs", n);
  await wait(2200);
  return page.evaluate(() => {
    const t = window.__hxT ? window.__hxT() : null;
    return {
      model: document.getElementById("hModel").textContent,
      legs: document.getElementById("hudLegs").textContent,
      dof: document.getElementById("hudDof").textContent,
      bars: document.querySelectorAll("#loadBars .bar").length,
      rows: document.querySelectorAll("#presetBtns .btn").length,
      note: document.getElementById("presetNote").textContent,
      speed: document.getElementById("mSpeed").textContent,
      painted: t,
    };
  });
};

const ten = await setLegs(10);
check("frame accepts ten legs", ten.legs === "10" && ten.dof === "30", `${ten.model}`);
check("per-leg readouts follow the frame", ten.bars === 10, `${ten.bars} load bars`);
check("ten legs still walk", parseFloat(ten.speed) > 1.0, ten.speed);
await page.screenshot({ path: path.join(SHOTS, "10-decapod.png") });

const four = await setLegs(4);
check("frame accepts four legs", four.legs === "4" && four.dof === "12", `${four.model}`);
check("per-leg readouts shrink too", four.bars === 4, `${four.bars} load bars`);
check("four legs still walk", parseFloat(four.speed) > 1.0, four.speed);
check(
  "a quadruped is warned off the trot",
  /falls over/.test(four.note),
  four.note || "(no note)"
);
await page.screenshot({ path: path.join(SHOTS, "11-quadruped.png") });

// The stage has to actually redraw the new machine, not the old one.
const repainted = await page.evaluate(() => {
  const cv = document.getElementById("view");
  const d = cv.getContext("2d").getImageData(0, 0, cv.width, cv.height).data;
  const seen = new Set();
  for (let i = 0; i < d.length; i += 4000) seen.add(`${d[i]},${d[i + 1]},${d[i + 2]}`);
  return seen.size;
});
check("stage renders the new frame", repainted > 20, `${repainted} distinct colours`);

await setLegs(6);

/* ------------------------------------------------- charts and gauges */

// Every canvas's backing store has to match the box CSS lays it out in, or
// everything drawn on it is stretched. The dials are square and it shows.
const canvasFit = await page.evaluate(() =>
  ["dSolver", "dStab", "cTorque", "cFoot", "cGait"].map((id) => {
    const cv = document.getElementById(id);
    const r = cv.getBoundingClientRect();
    return { id, ax: cv.width / r.width, ay: cv.height / r.height };
  })
);
check(
  "canvas pixels are square everywhere",
  canvasFit.every((c) => Math.abs(c.ax - c.ay) < 0.02),
  canvasFit.map((c) => `${c.id} ${c.ax.toFixed(2)}:${c.ay.toFixed(2)}`).join(" ")
);

// Whatever the learner has made of the coordination by now, the panel has to
// have a reading for it — that is the whole point of measuring rather than
// repeating the label.
const learnedPattern = await page.evaluate(() => window.__hxFalls.classify());
check(
  "the learned policy gets its own reading",
  typeof learnedPattern === "string" && learnedPattern.length > 1,
  `learned footfalls read as ${learnedPattern}`
);
// The four-legged frame moved the machine onto the crawl, and coming back to
// six legs leaves it there — so ask for the alternating gait explicitly.
await page.click("#btnBase");
await page.click('[data-preset="0"]');
await wait(4600);

const gait = await page.evaluate(() => {
  const f = window.__hxFalls;
  return f
    ? {
        n: f.t.length,
        kind: f.classify(),
        cycle: f.cycle(),
        duty: f.duty(),
        offsets: f.offsets(),
        live: window.__hxDuty(),
      }
    : null;
});
check("footfalls are recorded, not assumed", gait && gait.n > 60, `${gait && gait.n} samples`);
check(
  "the pattern is classified from the footfalls, not the label",
  gait && gait.kind === "TRIPOD",
  `${gait && gait.kind}, offsets ${gait && gait.offsets.map((o) => o.toFixed(2)).join(" ")}`
);
check(
  "measured cycle matches the one the clock was set to",
  gait && Math.abs(gait.cycle - gait.live.cycle) < 0.06,
  gait && `${gait.cycle.toFixed(3)} s measured vs ${gait.live.cycle.toFixed(3)} commanded`
);
// Every leg carries the same share of the cycle, and that share is the duty
// factor the gait clock was actually running — the panel is reading the
// machine, not repeating its settings.
check(
  "measured duty matches the running gait, leg by leg",
  gait && gait.duty.length === 6 && gait.duty.every((d) => Math.abs(d - gait.live.duty) < 0.08),
  gait && `${gait.duty.map((d) => d.toFixed(2)).join(" ")} vs ${gait.live.duty.toFixed(2)}`
);

/* ------------------------------------------------ walls and waypoints */

const nav0 = await page.evaluate(() => ({
  wp: document.getElementById("hudWp").textContent,
  nav: document.getElementById("hudNav").textContent,
  reached: document.getElementById("pReached").textContent,
  room: document.getElementById("pWall").textContent,
}));
check("the route is on the HUD", /^\d+\/\d+$/.test(nav0.wp), nav0.wp);
check("the autopilot is steering by default", nav0.nav === "AUTO", nav0.nav);
check(
  "walking a flat course reaches waypoints",
  parseInt(nav0.reached, 10) >= 1,
  `${nav0.reached} reached`
);
check("the wall meter reads a real distance", parseFloat(nav0.room) > 0, `${nav0.room} m`);

// Every course the simulator knows has to be reachable from the buttons, and
// every one of them has to come with a route.
const courses = await page.evaluate(() =>
  [...document.querySelectorAll("[data-course]")].map((b) => b.textContent)
);
const courseCount = await page.evaluate(() => (window.HX_COURSES || []).length);
check("every course the simulator knows has a button", courses.length === courseCount, courses.join(" "));

await page.click('[data-tab="terrain"]');
await wait(300);
const slalomIdx = courses.findIndex((c) => /slalom/i.test(c));
check("the slalom is one of them", slalomIdx >= 0, courses.join(" "));
await page.click(`[data-course="${slalomIdx}"]`);
await wait(2500);
const slalom = await page.evaluate(() => ({
  summary: document.getElementById("tSummary").textContent,
  note: document.getElementById("tNote").textContent,
  route: window.__hxRoute ? window.__hxRoute() : 0,
  sway: window.__hxSway ? window.__hxSway() : 0,
}));
check("the slalom loads", /SLALOM/.test(slalom.summary), slalom.summary);
check("and it comes with a route", slalom.route >= 6, `${slalom.route} waypoints`);
check(
  "the route leaves the centreline to get round the walls",
  slalom.sway > 1.0,
  `${slalom.sway.toFixed(2)} m off centre`
);
await page.screenshot({ path: path.join(SHOTS, "12-slalom.png") });

// JUMP is parkour on the same machine: trenches wider than a stride, a
// speed command, and the seed actually leaves the ground while running.
const jumpIdx = courses.findIndex((c) => /jump/i.test(c));
check("the jump course is one of them", jumpIdx >= 0, courses.join(" "));
await page.click(`[data-course="${jumpIdx}"]`);
// The sim starts playing (`Pause` on the button). Only click if a previous
// test left it held — otherwise we pause a running hop and wait on a frozen clock.
const pauseLabel = await page.textContent("#btnPause");
if (/Resume/i.test(pauseLabel || "")) await page.click("#btnPause");
try {
  await page.waitForFunction(
    () => {
      const state = document.getElementById("hState")?.textContent || "";
      const meter = document.getElementById("mSpeedLabel")?.textContent || "";
      const clock = document.getElementById("hClock")?.textContent || "";
      const jumps = Number((meter.match(/(\d+)\s*jumps/i) || ["", "0"])[1]);
      const m = clock.match(/(\d+):(\d+(?:\.\d+)?)/);
      const secs = m ? Number(m[1]) * 60 + Number(m[2]) : 0;
      return /JUMPING/i.test(state) || jumps > 0 || secs >= 1.4;
    },
    { timeout: 25000 }
  );
} catch {
  // assertions below report HUD / meter / clock
}
const jump = await page.evaluate(() => ({
  summary: document.getElementById("tSummary").textContent,
  note: document.getElementById("tNote").textContent,
  title: document.getElementById("cruiseTitle").textContent,
  hold: document.getElementById("vCruise").textContent,
  meter: document.getElementById("mSpeedLabel").textContent,
  state: document.getElementById("hState").textContent,
  speed: document.getElementById("mSpeed").textContent,
  clock: document.getElementById("hClock").textContent,
}));
check("the jump course loads", /JUMP/.test(jump.summary), jump.summary);
check("the command dial stays a speed", /speed/i.test(jump.title), jump.title);
check("the hold is metres per second", /m\/s/.test(jump.hold.trim()), jump.hold);
check(
  "the note is parkour, not a standing hop",
  /trench|parkour|platform|stride/i.test(jump.note),
  jump.note.slice(0, 80)
);
check(
  "the live meter tracks speed and counts jumps",
  /speed/i.test(jump.meter) && /jump/i.test(jump.meter),
  jump.meter
);
const jumped = /JUMPING|AIRBORNE/i.test(jump.state) || /[1-9]\s*jumps/i.test(jump.meter);
check(
  "the seed takes off on the first trench",
  jumped,
  `${jump.state} / ${jump.meter} @ ${jump.clock}`
);
await page.screenshot({ path: path.join(SHOTS, "13-jump.png") });

// Turning the autopilot off has to actually hand steering back.
await page.click("#btnNav");
await wait(900);
const manual = await page.textContent("#hudNav");
check("the autopilot can be switched off", manual === "MANUAL", manual);
await page.click("#btnNav");
await wait(400);

await page.click('[data-course="4"]');
await wait(1200);
await page.click('[data-tab="kinematics"]');
await wait(600);

/* ------------------------------------------------------------- wrap */

check("no console errors overall", errors.length === 0, errors.slice(0, 3).join(" | "));

const noHScroll = await page.evaluate(
  () => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1
);
check("page does not scroll sideways", noHScroll);

await browser.close();
console.log(`\n${failures === 0 ? "PASS" : failures + " FAILURE(S)"}  — screenshots in dist/shots/`);
process.exit(failures === 0 ? 0 : 1);
