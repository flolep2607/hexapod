/* Live physics worker.
 *
 * The dashboard keeps a second WASM instance on the UI thread for synchronous
 * configuration, training reports and catalogue queries. This worker mirrors
 * every mutation that can affect the live plant and owns the real-time step
 * loop. Telemetry is copied back at display rate; the buffer is small enough
 * that transfer latency is lost in the noise, while Rapier no longer competes
 * with geometry construction on the UI thread.
 */

let api = null;
let telemetryLen = 0;
let timeIndex = 0;
let control = { paused: false, rate: 1, fwd: 1, turn: 0 };
let debt = 0;
let dropped = 0;
let lastPump = 0;
let lastPublish = 0;
let lastTelemetryWall = 0;
let lastTelemetryTime = 0;
let actualRate = 0;
let stepUs = 0;
let timer = 0;
let synced = false;
let scheduled = false;
let burstStreak = 0;
const pumpChannel = new MessageChannel();

const DT = 1 / 100;
const STEP_CHUNK = 2 * DT;
const BURST_MS = 8;
const MAX_DEBT = 0.5;
const PUBLISH_MS = 1000 / 30;

function telemetry() {
  return new Float32Array(api.memory.buffer, api.hx_telemetry_ptr(), telemetryLen);
}

function publish(now) {
  api.hx_publish();
  const snapshot = telemetry().slice();
  const simTime = snapshot[timeIndex];
  if (lastTelemetryWall > 0 && simTime >= lastTelemetryTime) {
    const wall = (now - lastTelemetryWall) / 1000;
    if (wall > 0) {
      const sample = (simTime - lastTelemetryTime) / wall;
      actualRate += (sample - actualRate) * 0.2;
    }
  } else {
    actualRate = 0;
  }
  lastTelemetryWall = now;
  lastTelemetryTime = simTime;
  lastPublish = now;
  postMessage(
    { type: "telemetry", data: snapshot.buffer, actualRate, stepUs, dropped },
    [snapshot.buffer]
  );
}

function schedule(delay) {
  if (scheduled) return;
  scheduled = true;
  if (delay <= 0) pumpChannel.port2.postMessage(0);
  else {
    clearTimeout(timer);
    timer = setTimeout(() => {
      scheduled = false;
      pump();
    }, delay);
  }
}

pumpChannel.port1.onmessage = () => {
  scheduled = false;
  pump();
};

function pump() {
  const now = performance.now();
  if (!lastPump) lastPump = now;
  const wall = Math.min(0.05, Math.max(0, (now - lastPump) / 1000));
  lastPump = now;

  if (!control.paused) {
    debt += wall * control.rate;
    if (debt > MAX_DEBT) {
      dropped += debt - MAX_DEBT;
      debt = MAX_DEBT;
    }

    const started = performance.now();
    let ticks = 0;
    while (debt >= DT - 1e-9 && performance.now() - started < BURST_MS) {
      const slice = Math.min(debt, STEP_CHUNK);
      api.hx_step_quiet(slice, control.fwd, control.turn);
      debt -= slice;
      ticks += Math.round(slice / DT);
    }
    if (ticks) {
      const sampleUs = ((performance.now() - started) * 1000) / ticks;
      stepUs += (sampleUs - stepUs) * 0.1;
    }
  } else {
    debt = 0;
  }

  const after = performance.now();
  if (synced && after - lastPublish >= PUBLISH_MS) publish(after);
  if (debt >= DT) {
    // MessageChannel avoids the browser's nested-timer clamp, but an endless
    // chain can starve control messages at 10x. Periodically return through
    // the timer queue so Pause and mode/config changes are observed promptly.
    burstStreak += 1;
    schedule(burstStreak >= 12 ? 1 : 0);
    if (burstStreak >= 12) burstStreak = 0;
  } else {
    burstStreak = 0;
    schedule(2);
  }
}

onmessage = async (event) => {
  const msg = event.data;
  if (msg.type === "init") {
    try {
      const { instance } = await WebAssembly.instantiate(msg.wasm, {});
      api = instance.exports;
      telemetryLen = msg.telemetryLen;
      timeIndex = msg.timeIndex;
      api.hx_init(msg.seed);
      lastPump = performance.now();
      lastPublish = lastPump;
      postMessage({ type: "ready" });
    } catch (error) {
      postMessage({ type: "error", error: String(error) });
    }
    return;
  }
  if (!api) return;
  if (msg.type === "sync") {
    synced = true;
    lastPump = performance.now();
    lastPublish = lastPump - PUBLISH_MS;
    schedule(0);
    return;
  }
  if (msg.type === "call") {
    const fn = api[msg.name];
    if (typeof fn === "function") fn(...msg.args);
    return;
  }
  if (msg.type === "control") {
    control = msg.control;
    if (control.paused) debt = 0;
  }
};
