// Synthetic tests for the snap gesture + click detectors in content/inject.js.
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const { SnapDetector, ClickDetector, handFeatures } = require('../content/inject.js');

// ── tiny kinematic hand model (metres). Palm in the x/y plane, fingers along +y, curl towards -z.
const deg = (d) => (d * Math.PI) / 180;
const FINGERS = {
  index: { mcp: [0.026, 0.086], len: [0.040, 0.025, 0.020] },
  middle: { mcp: [0.004, 0.090], len: [0.045, 0.028, 0.022] },
  ring: { mcp: [-0.016, 0.085], len: [0.042, 0.026, 0.021] },
  pinky: { mcp: [-0.034, 0.074], len: [0.032, 0.020, 0.018] },
};
function finger(f, a1, a2, a3) {
  const pts = [[f.mcp[0], f.mcp[1], 0]];
  let ang = 0, p = pts[0];
  [a1, a2, a3].forEach((a, i) => {
    ang += deg(a);
    p = [p[0], p[1] + Math.cos(ang) * f.len[i], p[2] - Math.sin(ang) * f.len[i]];
    pts.push(p);
  });
  return pts;
}
const lerp3 = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
function thumbTo(tip) {
  const cmc = [0.022, 0.018, -0.005];
  const mcp = lerp3(cmc, tip, 0.42), ip = lerp3(cmc, tip, 0.72);
  mcp[0] += 0.012; ip[0] += 0.008; // slight outward bow
  return [cmc, mcp, ip, tip];
}
function hand(pose, noise = 0.0025, rnd = Math.random) {
  const w = [[0, 0, 0]];
  const th = thumbTo(pose.thumbTip(pose));
  w.push(...th);
  w.push(...finger(FINGERS.index, ...pose.index));
  w.push(...finger(FINGERS.middle, ...pose.middle));
  w.push(...finger(FINGERS.ring, ...pose.ring));
  w.push(...finger(FINGERS.pinky, ...pose.pinky));
  const out = new Float32Array(63);
  w.forEach((p, i) => { for (let k = 0; k < 3; k++) out[i * 3 + k] = p[k] + (rnd() - 0.5) * 2 * noise; });
  return out;
}
const tipOf = (f, a) => finger(FINGERS[f], ...a)[3];
const CURLED = [75, 100, 60];
const POSES = {
  open: { index: [5, 5, 5], middle: [5, 5, 5], ring: [5, 5, 5], pinky: [5, 5, 5], thumbTip: () => [0.075, 0.07, -0.01] },
  set: { index: [25, 30, 15], middle: [38, 42, 20], ring: CURLED, pinky: CURLED, thumbTip: (p) => { const t = tipOf('middle', p.middle); return [t[0] + 0.004, t[1] - 0.006, t[2] - 0.010]; } },
  post: { index: [20, 25, 10], middle: [80, 105, 65], ring: CURLED, pinky: CURLED, thumbTip: () => [0.045, 0.105, -0.035] },
  fist: { index: CURLED, middle: CURLED, ring: CURLED, pinky: CURLED, thumbTip: () => [0.01, 0.06, -0.06] },
  typing: { index: [30, 35, 15], middle: [30, 35, 15], ring: [30, 35, 15], pinky: [30, 35, 15], thumbTip: () => [0.05, 0.06, -0.03] },
  pinchIdx: { index: [35, 40, 20], middle: [20, 20, 10], ring: [20, 20, 10], pinky: [20, 20, 10], thumbTip: (p) => { const t = tipOf('index', p.index); return [t[0] + 0.004, t[1] - 0.004, t[2] - 0.008]; } },
};
function mix(a, b, t) {
  const r = {};
  for (const k of ['index', 'middle', 'ring', 'pinky']) r[k] = a[k].map((v, i) => v + (b[k][i] - v) * t);
  const ta = a.thumbTip(a), tb = b.thumbTip(b);
  r.thumbTip = () => lerp3(ta, tb, t);
  return r;
}

// sequence = list of [poseOrMixer, durationMs]; returns detector events
function run(seq, sensitivity, opts = {}) {
  const d = new SnapDetector();
  d.setSensitivity(sensitivity);
  const fps = opts.fps || 30, dt = 1000 / fps;
  let t = 1000;
  const events = [];
  let seed = opts.seed || 1;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (const [p, dur] of seq) {
    const n = Math.max(1, Math.round(dur / dt));
    for (let i = 0; i < n; i++) {
      const pose = typeof p === 'function' ? p(i / Math.max(1, n - 1)) : p;
      const hands = pose ? [{ label: 'Right', world: hand(pose, opts.noise ?? 0.0025, rnd) }] : [];
      events.push(...d.update(t, hands));
      t += dt;
    }
  }
  return events;
}

let failures = 0;
const check = (name, ok, extra = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${extra}`); if (!ok) failures++; };

console.log('features open', handFeatures(hand(POSES.open, 0)));
console.log('features set ', handFeatures(hand(POSES.set, 0)));
console.log('features post', handFeatures(hand(POSES.post, 0)));

for (const s of [1, 5, 10]) {
  for (let seed = 1; seed <= 5; seed++) {
    const o = { seed };
    // real snap: open → set (hold) → snap (one frame) → post
    let ev = run([[POSES.open, 400], [(t) => mix(POSES.open, POSES.set, t), 300], [POSES.set, 250], [POSES.post, 300], [POSES.open, 300]], s, o);
    check(`snap s=${s} seed=${seed}`, ev.length === 1, ev.map((e) => e.score.toFixed(2)).join(','));
    // snap with one blurred / missing frame in between
    ev = run([[POSES.set, 250], [null, 33], [POSES.post, 300]], s, o);
    check(`snap+dropout s=${s} seed=${seed}`, ev.length === 1);
    // slow fist from the set pose (600 ms) — not a snap
    ev = run([[POSES.set, 300], [(t) => mix(POSES.set, POSES.fist, t), 600], [POSES.fist, 300]], s, o);
    check(`slow fist s=${s} seed=${seed}`, ev.length === 0);
    // set pose then opening the hand
    ev = run([[POSES.set, 300], [(t) => mix(POSES.set, POSES.open, t), 150], [POSES.open, 300]], s, o);
    check(`open from set s=${s} seed=${seed}`, ev.length === 0);
    // typing jiggle
    ev = run([[(t) => mix(POSES.typing, POSES.open, 0.5 + 0.5 * Math.sin(t * 40)), 3000]], s, { ...o, noise: 0.004 });
    check(`typing s=${s} seed=${seed}`, ev.length === 0);
    // index pinch + release fast
    ev = run([[POSES.pinchIdx, 300], [POSES.open, 300], [POSES.pinchIdx, 300], [POSES.fist, 300]], s, o);
    check(`index pinch s=${s} seed=${seed}`, ev.length === 0);
    // waving
    ev = run([[(t) => mix(POSES.open, POSES.typing, 0.5 + 0.5 * Math.sin(t * 25)), 2000]], s, o);
    check(`wave s=${s} seed=${seed}`, ev.length === 0);
  }
  // 15 fps camera
  const ev = run([[POSES.set, 300], [POSES.post, 300]], s, { fps: 15 });
  check(`snap 15fps s=${s}`, ev.length === 1);
}

// ── click detector: synthetic snap click vs speech-like noise
function audio(sr, secs, fn) { const a = new Float32Array(sr * secs); for (let i = 0; i < a.length; i++) a[i] = fn(i / sr, i); return a; }
let s = 7;
const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5);
for (const sens of [1, 5, 10]) {
  const sr = 48000;
  const sig = audio(sr, 3, (t) => {
    let v = rnd() * 0.004; // room noise
    if (t > 1.0 && t < 1.02) v += rnd() * 0.9 * Math.exp(-(t - 1.0) * 250); // snap click at 1.0 s
    return v;
  });
  const cd = new ClickDetector(); cd.setSensitivity(sens);
  const got = [];
  cd.onClick = (t) => got.push(t);
  for (let i = 0; i < sig.length; i += 480) cd.process(sig.subarray(i, i + 480), sr, ((i + 480) / sr) * 1000);
  check(`click s=${sens}`, got.length === 1 && Math.abs(got[0] - 1000) < 15, JSON.stringify(got.map((x) => Math.round(x))));
  // speech-like: sustained band-limited bursts with slow envelope
  const speech = audio(sr, 3, (t) => rnd() * 0.004 + Math.sin(2 * Math.PI * 180 * t) * 0.2 * Math.max(0, Math.sin(2 * Math.PI * 3 * t)) + rnd() * 0.15 * Math.max(0, Math.sin(2 * Math.PI * 3 * t)) ** 2);
  const cd2 = new ClickDetector(); cd2.setSensitivity(sens);
  const got2 = [];
  cd2.onClick = (t) => got2.push(t);
  for (let i = 0; i < speech.length; i += 480) cd2.process(speech.subarray(i, i + 480), sr, ((i + 480) / sr) * 1000);
  check(`speech no click s=${sens}`, got2.length === 0, `(${got2.length})`);
}

console.log(failures ? `${failures} FAILED` : 'all passed');
process.exit(failures ? 1 : 0);
