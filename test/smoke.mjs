/* Parallel end-to-end check of the built dashboard in real browser pages.
 *
 *   node test/smoke.mjs [path-to-html]
 *   SMOKE_SCREENSHOTS=1 node test/smoke.mjs  # also write visual snapshots
 *
 * Independent flows get independent pages and run concurrently. Assertions
 * that truly share simulator state remain ordered inside one scenario.
 */

import { chromium } from "playwright";
import path from "node:path";
import fs from "node:fs";

const FILE = path.resolve(process.argv[2] || "dist/hexapod-simulator.html");
const SHOTS = path.resolve("dist/shots");
const TAKE_SCREENSHOTS = process.env.SMOKE_SCREENSHOTS === "1";
if (TAKE_SCREENSHOTS) fs.mkdirSync(SHOTS, { recursive: true });
const builtHtml = fs.readFileSync(FILE, "utf8");
const HAS_DEV_FILE_RELOADER =
  builtHtml.includes("r.headers.get('last-modified')") && builtHtml.includes("cache:'no-store'");

const ciChromium = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const executablePath =
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || (fs.existsSync(ciChromium) ? ciChromium : undefined);
const browser = await chromium.launch(executablePath ? { executablePath } : {});

async function openHarness(name) {
  const context = await browser.newContext({ viewport: { width: 1680, height: 1000 } });
  const page = await context.newPage();
  // A --fast artifact reloads itself when dist changes. Smoke pages must stay
  // pinned to the artifact they opened even if a dev watcher rebuilds it.
  await page.addInitScript(() => {
    window.setInterval = () => 0;
  });
  const checks = [];
  const errors = [];
  const hasDevFileReloader = HAS_DEV_FILE_RELOADER;

  page.on("console", (message) => {
    if (message.type() !== "error") return;
    const text = message.text();
    const expectedDevReloadError =
      hasDevFileReloader &&
      ((text.includes("Access to fetch at 'file://") && text.includes("blocked by CORS policy")) ||
        text === "Failed to load resource: net::ERR_FAILED");
    if (!expectedDevReloadError) errors.push(text);
  });
  page.on("pageerror", (error) => errors.push(String(error)));

  const check = (label, ok, detail = "") => checks.push({ label, ok: Boolean(ok), detail });
  const waitFor = async (fn, arg = null, timeout = 5000) => {
    try {
      return await page.waitForFunction(fn, arg, { timeout });
    } catch {
      return null;
    }
  };
  const setRange = (id, value) =>
    page.evaluate(
      ([rangeId, rangeValue]) => {
        const el = document.getElementById(rangeId);
        el.value = String(rangeValue);
        el.dispatchEvent(new Event("input", { bubbles: true }));
      },
      [id, value]
    );
  const screenshot = (file) =>
    TAKE_SCREENSHOTS ? page.screenshot({ path: path.join(SHOTS, file) }) : Promise.resolve();
  const stepSamples = (count) => page.evaluate((n) => window.__hxStepSamples(n), count);
  const nextFrame = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(resolve)));

  await page.goto("file://" + FILE);
  await page.waitForFunction(() => window.__ready === true, null, { timeout: 15000 });

  return {
    name,
    page,
    context,
    checks,
    errors,
    check,
    waitFor,
    setRange,
    screenshot,
    stepSamples,
    nextFrame,
  };
}

async function bootAndStatic(h) {
  const { page, check, waitFor, setRange, screenshot, errors } = h;
  await waitFor(() => /ONE/.test(document.getElementById("hPolicy")?.textContent || ""));
  check("no console errors on boot", errors.length === 0, errors.slice(0, 3).join(" | "));
  check(
    "wasm exports present",
    await page.evaluate(() => typeof document.getElementById("view") !== "undefined")
  );

  const clock0 = await page.textContent("#hClock");
  await waitFor((before) => document.getElementById("hClock")?.textContent !== before, clock0, 2000);
  const clock1 = await page.textContent("#hClock");
  check("simulation clock advances", clock0 !== clock1, `${clock0} -> ${clock1}`);

  const canvasPainted = await page.evaluate(() => {
    const cv = document.getElementById("view");
    const data = cv.getContext("2d").getImageData(0, 0, cv.width, cv.height).data;
    const seen = new Set();
    for (let i = 0; i < data.length; i += 4000) {
      seen.add(`${data[i]},${data[i + 1]},${data[i + 2]}`);
    }
    return seen.size;
  });
  check("3-D stage renders content", canvasPainted > 6, `${canvasPainted} distinct sampled colours`);
  const drill = await page.evaluate(() => window.__hxOneleg());
  check(
    "the page boots on the one-leg drill",
    drill.on && /ONE/.test(drill.policy),
    JSON.stringify({ on: drill.on, policy: drill.policy, phase: drill.phaseName })
  );
  check(
    "the one-leg callout is on the stage",
    await page.$eval("#drillCallout", (el) => !el.hidden)
  );

  await page.click("#btnCamTop");
  check("top camera toggle", (await page.textContent("#hudCam")).includes("TOP"));
  check("top camera button latches", (await page.getAttribute("#btnCamTop", "data-on")) === "true");
  await page.keyboard.press("3");
  check("side camera key", (await page.textContent("#hudCam")).includes("SIDE"));
  await page.click("#btnCamOrbit");
  check("orbit camera restore", (await page.textContent("#hudCam")).includes("ORBIT"));
  await screenshot("01-kinematics.png");
  await page.click("#btnPause");

  const characterSet = await page.evaluate(() => document.characterSet);
  check("the page is decoded as UTF-8", characterSet === "UTF-8", characterSet);

  await page.click('[data-tab="terrain"]');
  const summary = await page.textContent("#tSummary");
  check("terrain summary populated", /obstacles/.test(summary), summary);
  await page.click('[data-course="1"]');
  const stepsSummary = await page.textContent("#tSummary");
  check("course switches", /STEPS/.test(stepsSummary), stepsSummary);
  await screenshot("05-terrain.png");

  check(
    "servo and battery pages are removed",
    (await page.$('[data-tab="hardware"]')) === null &&
      (await page.$('[data-tab="system"]')) === null &&
      (await page.$("#selServo")) === null
  );
  await page.click('[data-tab="about"]');
  await screenshot("06-about.png");

  const canvasFit = await page.evaluate(() =>
    ["dSolver", "dStab", "cTorque", "cFoot"].map((id) => {
      const cv = document.getElementById(id);
      const rect = cv.getBoundingClientRect();
      return { id, ax: cv.width / rect.width, ay: cv.height / rect.height };
    })
  );
  check(
    "canvas pixels are square everywhere",
    canvasFit.every((canvas) => Math.abs(canvas.ax - canvas.ay) < 0.02),
    canvasFit.map((canvas) => `${canvas.id} ${canvas.ax.toFixed(2)}:${canvas.ay.toFixed(2)}`).join(" ")
  );
  const noHScroll = await page.evaluate(
    () => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1
  );
  check("page does not scroll sideways", noHScroll);
}

const clickWalk = async (page) => {
  if ((await page.getAttribute("#btnWalk", "data-on")) !== "true") {
    await page.click("#btnWalk");
  }
};

async function speedAndPhysics(h) {
  const { page, check, setRange, stepSamples } = h;
  await clickWalk(page);
  await page.click("#btnPause");
  const speedAt = async (target) => {
    await setRange("rCruise", target);
    const rollout = await stepSamples(150);
    return rollout.speed.closest;
  };
  const slow = await speedAt(2.0);
  const fast = await speedAt(5.5);
  check(
    "commanded speed drives the robot",
    fast > slow + 1.5,
    `${slow.toFixed(2)} -> ${fast.toFixed(2)} m/s`
  );
  check("speed dial reads back", (await page.textContent("#vCruise")).includes("5.5"));
  await setRange("rCruise", 4.0);
  await setRange("rRate", 2);
  check("sim speed dial reads back", (await page.textContent("#vRate")) === "2×");
  await setRange("rRate", 1);
  let stepped = await stepSamples(36);

  let hull = stepped.hull;
  for (let attempt = 0; attempt < 10 && !(hull.n >= 3 && hull.span < 1.5); attempt++) {
    stepped = await stepSamples(18);
    if (stepped.hull.n >= 3 && stepped.hull.span < hull.span) hull = stepped.hull;
  }
  const physics = await page.evaluate(() => ({
    traction: document.getElementById("mTrac").textContent,
    servo: document.getElementById("mServo").textContent,
    cot: document.getElementById("pCot").textContent,
    lag: document.getElementById("pLag").textContent,
  }));
  check("traction telemetry live", /^\d+%$/.test(physics.traction), physics.traction);
  check("servo load telemetry live", /^\d+%$/.test(physics.servo), physics.servo);
  check("cost of transport computed", parseFloat(physics.cot) > 0, physics.cot);
  check("joint tracking lag reported", parseFloat(physics.lag) > 0, `${physics.lag} deg`);
  check(
    "support polygon tracks the drawn body",
    hull.n >= 3 && hull.span < 1.5,
    `${hull.n} pts, ${hull.span.toFixed(2)} m`
  );
}

async function frameState(h, legs) {
  const { page, setRange, stepSamples, nextFrame } = h;
  await clickWalk(page);
  if (/Pause/i.test((await page.textContent("#btnPause")) || "")) await page.click("#btnPause");
  await setRange("rRate", 1);
  await setRange("rLegs", legs);
  await stepSamples(132);
  let speed = parseFloat(await page.textContent("#mSpeed"));
  for (let attempt = 0; attempt < 100 && !(speed > 1); attempt++) {
    await stepSamples(6);
    speed = parseFloat(await page.textContent("#mSpeed"));
  }
  await nextFrame();
  return page.evaluate(() => ({
      model: document.getElementById("hModel").textContent,
      legs: document.getElementById("hudLegs").textContent,
      dof: document.getElementById("hudDof").textContent,
      bars: document.querySelectorAll("#loadBars .bar").length,
      note: document.getElementById("presetNote").textContent,
      speed: document.getElementById("mSpeed").textContent,
    }));
}

async function decapod(h) {
  const { page, check, screenshot } = h;
  const state = await frameState(h, 10);
  check("frame accepts ten legs", state.legs === "10" && state.dof === "30", state.model);
  check("per-leg readouts follow the frame", state.bars === 10, `${state.bars} load bars`);
  check("ten legs still walk", parseFloat(state.speed) > 1.0, state.speed);
  const painted = await page.evaluate(() => {
    const cv = document.getElementById("view");
    const data = cv.getContext("2d").getImageData(0, 0, cv.width, cv.height).data;
    const seen = new Set();
    for (let i = 0; i < data.length; i += 4000) {
      seen.add(`${data[i]},${data[i + 1]},${data[i + 2]}`);
    }
    return seen.size;
  });
  check("stage renders the new frame", painted > 20, `${painted} distinct colours`);
  await screenshot("10-decapod.png");
}

async function quadruped(h) {
  const { check, screenshot } = h;
  const state = await frameState(h, 4);
  check("frame accepts four legs", state.legs === "4" && state.dof === "12", state.model);
  check("per-leg readouts shrink too", state.bars === 4, `${state.bars} load bars`);
  check("four legs still walk", parseFloat(state.speed) > 1.0, state.speed);
  check("a quadruped is warned off the trot", /falls over/.test(state.note), state.note || "(no note)");
  await screenshot("11-quadruped.png");
}

async function gaitAndNavigation(h) {
  const { page, check, waitFor, stepSamples } = h;
  await page.click("#btnBase");
  await page.click('[data-preset="0"]');
  await waitFor(
    () =>
      window.__hxFalls.t.length > 40 &&
      window.__hxFalls.cycle() > 0 &&
      window.__hxFalls.classify() === "TRIPOD",
    null,
    8000
  );
  const gait = await page.evaluate(() => {
    const falls = window.__hxFalls;
    const duty = new Array(falls.legs).fill(0);
    const elapsed = falls.t[falls.t.length - 1] - falls.t[0];
    for (let sample = 1; sample < falls.t.length; sample++) {
      const dt = falls.t[sample] - falls.t[sample - 1];
      for (let leg = 0; leg < falls.legs; leg++) duty[leg] += falls.stance[sample - 1][leg] * dt;
    }
    return {
      n: falls.t.length,
      kind: falls.classify(),
      cycle: falls.cycle(),
      duty: duty.map((value) => value / elapsed),
      offsets: falls.offsets(),
      live: window.__hxDuty(),
    };
  });
  check("footfalls are recorded, not assumed", gait.n > 40, `${gait.n} samples`);
  check(
    "the pattern is classified from the footfalls, not the label",
    gait.kind === "TRIPOD",
    `${gait.kind}, offsets ${gait.offsets.map((value) => (value === null ? "—" : value.toFixed(2))).join(" ")}`
  );
  check(
    "measured cycle matches the one the clock was set to",
    Math.abs(gait.cycle - gait.live.cycle) < 0.06,
    `${gait.cycle.toFixed(3)} s measured vs ${gait.live.cycle.toFixed(3)} commanded`
  );
  check(
    "measured duty matches the running gait, leg by leg",
    gait.duty.length === 6 && gait.duty.every((value) => Math.abs(value - gait.live.duty) < 0.08),
    `${gait.duty.map((value) => value.toFixed(2)).join(" ")} vs ${gait.live.duty.toFixed(2)}`
  );

  await page.click("#btnPause");
  let stepped = await stepSamples(180);
  for (let attempt = 0; attempt < 5 && stepped.reached < 1; attempt++) stepped = await stepSamples(180);
  const nav = await page.evaluate(() => ({
    waypoint: document.getElementById("hudWp").textContent,
    mode: document.getElementById("hudNav").textContent,
    reached: document.getElementById("pReached").textContent,
    room: document.getElementById("pWall").textContent,
  }));
  check("the route is on the HUD", /^\d+\/\d+$/.test(nav.waypoint), nav.waypoint);
  check("the autopilot is steering by default", nav.mode === "AUTO", nav.mode);
  check("walking a flat course reaches waypoints", parseInt(nav.reached, 10) >= 1, `${nav.reached} reached`);
  check("the wall meter reads a real distance", parseFloat(nav.room) > 0, `${nav.room} m`);
}

async function courses(h) {
  const { page, check, waitFor, setRange, screenshot, stepSamples, nextFrame } = h;
  if (/Pause/i.test((await page.textContent("#btnPause")) || "")) await page.click("#btnPause");
  await setRange("rRate", 2);
  const names = await page.evaluate(() =>
    [...document.querySelectorAll("[data-course]")].map((button) => button.textContent)
  );
  const courseCount = await page.evaluate(() => (window.HX_COURSES || []).length);
  check("every course the simulator knows has a button", names.length === courseCount, names.join(" "));

  await page.click('[data-tab="terrain"]');
  const slalomIndex = names.findIndex((name) => /slalom/i.test(name));
  check("the slalom is one of them", slalomIndex >= 0, names.join(" "));
  await page.click(`[data-course="${slalomIndex}"]`);
  const slalom = await page.evaluate(() => ({
    summary: document.getElementById("tSummary").textContent,
    route: window.__hxRoute ? window.__hxRoute() : 0,
    sway: window.__hxSway ? window.__hxSway() : 0,
  }));
  check("the slalom loads", /SLALOM/.test(slalom.summary), slalom.summary);
  check("and it comes with a route", slalom.route >= 6, `${slalom.route} waypoints`);
  check("the route leaves the centreline to get round the walls", slalom.sway > 1.0, `${slalom.sway.toFixed(2)} m off centre`);
  await screenshot("12-slalom.png");

  await page.click('[data-tab="kinematics"]');
  await page.click("#btnCamTop");
  check("top camera on the slalom", (await page.textContent("#hudCam")).includes("TOP"));
  await screenshot("13-slalom-top.png");
  await page.click("#btnCamSide");
  check("side camera on the slalom", (await page.textContent("#hudCam")).includes("SIDE"));
  await screenshot("14-slalom-side.png");
  await page.click("#btnCamOrbit");

  const jumpIndex = names.findIndex((name) => /jump/i.test(name));
  check("the jump course is one of them", jumpIndex >= 0, names.join(" "));
  await page.click('[data-tab="terrain"]');
  await page.click(`[data-course="${jumpIndex}"]`);
  await stepSamples(60);
  await nextFrame();
  const jump = await page.evaluate(() => ({
    summary: document.getElementById("tSummary").textContent,
    note: document.getElementById("tNote").textContent,
    title: document.getElementById("cruiseTitle").textContent,
    hold: document.getElementById("vCruise").textContent,
    meter: document.getElementById("mSpeedLabel").textContent,
    state: document.getElementById("hState").textContent,
    clock: document.getElementById("hClock").textContent,
  }));
  check("the jump course loads", /JUMP/.test(jump.summary), jump.summary);
  check("the command dial stays a speed", /speed/i.test(jump.title), jump.title);
  check("the hold is metres per second", /m\/s/.test(jump.hold.trim()), jump.hold);
  check("the note is parkour, not a standing hop", /trench|parkour|platform|stride/i.test(jump.note), jump.note.slice(0, 80));
  check("the live meter tracks speed and counts jumps", /speed/i.test(jump.meter) && /jump/i.test(jump.meter), jump.meter);
  const jumped = /JUMPING|AIRBORNE/i.test(jump.state) || /[1-9]\s*jumps/i.test(jump.meter);
  check("the seed takes off on the first trench", jumped, `${jump.state} / ${jump.meter} @ ${jump.clock}`);
  await screenshot("15-jump.png");

  await page.click("#btnNav");
  await stepSamples(1);
  await waitFor(() => document.getElementById("hudNav")?.textContent === "MANUAL", null, 1000);
  const manual = await page.textContent("#hudNav");
  check("the autopilot can be switched off", manual === "MANUAL", manual);
}

async function onelegDrill(h) {
  const { page, check, stepSamples, screenshot } = h;
  await page.click("#btnPause");
  check("one-leg button is in the page", await page.$("#btnOneleg") !== null);
  await page.click("#btnOneleg");
  const header = await page.textContent("#hPolicy");
  check("policy chip says ONE LEG", /ONE/.test(header), header);
  check("one-leg button latches", (await page.getAttribute("#btnOneleg", "data-on")) === "true");
  check("walk button unlatches", (await page.getAttribute("#btnWalk", "data-on")) !== "true");

  let maxClear = 0;
  let sawOneSwing = false;
  let extraSwing = false;
  let destWander = 0;
  let dest0 = null;
  let on = false;
  for (let i = 0; i < 10; i++) {
    await stepSamples(18);
    const s = await page.evaluate(() => window.__hxOneleg());
    on = s.on;
    maxClear = Math.max(maxClear, s.clear);
    const swinging = s.stance.filter((v) => !v).length;
    if (s.phase >= 1 && s.phase <= 3) {
      if (swinging === 1 && s.moving === 0) sawOneSwing = true;
      if (swinging > 1) extraSwing = true;
      if (dest0) {
        destWander = Math.max(
          destWander,
          Math.hypot(s.dest[0] - dest0[0], s.dest[1] - dest0[1], s.dest[2] - dest0[2])
        );
      } else {
        dest0 = s.dest;
      }
    } else {
      dest0 = null;
    }
  }
  check("telemetry says the drill is on", on);
  check("only L1 drops its stance flag", sawOneSwing && !extraSwing, `clear ${maxClear.toFixed(3)}`);
  check("the free foot leaves the floor", maxClear > 0.10, `${maxClear.toFixed(3)} m clearance`);
  check("the landing mark stays put during the swing", destWander < 0.02, `${destWander.toFixed(4)} m`);
  const hud = await page.evaluate(() => ({
    gait: document.getElementById("hudGait").textContent,
    drill: document.getElementById("hudDrill").hidden,
    walk: document.getElementById("hudWalk").hidden,
    course: document.getElementById("hCourse").textContent,
    cam: document.getElementById("hudCam").textContent,
  }));
  check("HUD switches to the drill readout", hud.walk === true && hud.drill === false, JSON.stringify(hud));
  check("the empty field is FLAT", /FLAT/.test(hud.course), hud.course);
  check(
    "the one-leg callout is on the stage",
    await page.$eval("#drillCallout", (el) => !el.hidden)
  );
  await screenshot("16-oneleg.png");

  await page.click("#btnWalk");
  check("walk restores the gait", (await page.getAttribute("#btnWalk", "data-on")) === "true");
  check("one-leg turns off", (await page.getAttribute("#btnOneleg", "data-on")) !== "true");
  check(
    "inactive one-leg button is not the orange primary action",
    !(await page.$eval("#btnOneleg", (button) => button.classList.contains("primary")))
  );
}

const scenarios = [
  ["dashboard", bootAndStatic],
  ["courses", courses],
  ["physics", async (harness) => {
    await speedAndPhysics(harness);
  }],
  ["frames", async (harness) => {
    await decapod(harness);
    await quadruped(harness);
  }],
  ["gait + navigation", gaitAndNavigation],
  ["one leg", onelegDrill],
];

const started = performance.now();
const SCENARIO_TIMEOUT_MS = 20000;
const withScenarioTimeout = async (promise, name) => {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(
          () => reject(new Error(`${name} exceeded ${SCENARIO_TIMEOUT_MS / 1000}s`)),
          SCENARIO_TIMEOUT_MS
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
};
const results = await Promise.all(
  scenarios.map(async ([name, scenario]) => {
    const scenarioStarted = performance.now();
    let harness;
    try {
      harness = await openHarness(name);
      await withScenarioTimeout(scenario(harness), name);
    } catch (error) {
      if (!harness) {
        return {
          name,
          checks: [{ label: "scenario completed", ok: false, detail: String(error) }],
          errors: [],
          elapsed: performance.now() - scenarioStarted,
        };
      }
      harness.check("scenario completed", false, String(error));
    } finally {
      if (harness) await harness.context.close();
    }
    return {
      name,
      checks: harness.checks,
      errors: harness.errors,
      elapsed: performance.now() - scenarioStarted,
    };
  })
);

await browser.close();

let failures = 0;
for (const result of results) {
  console.log(`\n${result.name}  — ${(result.elapsed / 1000).toFixed(2)}s`);
  for (const item of result.checks) {
    console.log(`${item.ok ? "  ok  " : "FAIL  "}${item.label}${item.detail ? "  — " + item.detail : ""}`);
    if (!item.ok) failures++;
  }
}

const errors = results.flatMap((result) => result.errors.map((error) => `${result.name}: ${error}`));
const clean = errors.length === 0;
console.log(`${clean ? "  ok  " : "FAIL  "}no console errors overall${clean ? "" : "  — " + errors.slice(0, 3).join(" | ")}`);
if (!clean) failures++;

const elapsed = (performance.now() - started) / 1000;
console.log(
  `\n${failures === 0 ? "PASS" : failures + " FAILURE(S)"}  — ${scenarios.length} parallel scenarios in ${elapsed.toFixed(2)}s` +
    (TAKE_SCREENSHOTS ? " · screenshots in dist/shots/" : "")
);
process.exit(failures === 0 ? 0 : 1);
