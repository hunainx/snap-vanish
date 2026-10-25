// Snap Vanish — on-device ML engine.
// Runs inside an extension-origin iframe so the page's CSP never touches the WASM/ML code.
// Talks to the page-side pipeline over a private MessageChannel. Nothing here touches the network:
// every model and runtime file is loaded from the extension package itself.

import { HandLandmarker, ImageSegmenter } from '../vendor/mediapipe/vision_bundle.mjs';

const base = new URL('../', import.meta.url).href;
const fileset = {
  wasmLoaderPath: base + 'vendor/mediapipe/wasm/vision_wasm_internal.js',
  wasmBinaryPath: base + 'vendor/mediapipe/wasm/vision_wasm_internal.wasm',
};

let port = null;
let hands = null;
let segmenter = null;
let ready = false;
let initError = null;
let lastTs = 0;
let inpaintWorker = null;
const inpaintJobs = new Map();

async function createTasks(delegate) {
  const h = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: base + 'models/hand_landmarker.task', delegate },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  const s = await ImageSegmenter.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: base + 'models/selfie_segmenter_landscape.tflite', delegate },
    runningMode: 'VIDEO',
    outputConfidenceMasks: true,
    outputCategoryMask: false,
  });
  return { h, s };
}

// Pick GPU or CPU. GPU is normally much faster, but on machines with software-only WebGL it can be
// 100× slower, so the engine times the first frames and switches if the other delegate wins.
const perf = { delegate: 'GPU', n: 0, ema: 0, results: {}, switching: false };
const forced = new URLSearchParams(location.search).get('delegate');

async function useDelegate(delegate) {
  const t = await createTasks(delegate);
  if (hands) hands.close();
  if (segmenter) segmenter.close();
  hands = t.h; segmenter = t.s;
  perf.delegate = delegate; perf.n = 0; perf.ema = 0;
}

async function init() {
  try {
    const first = forced === 'cpu' ? 'CPU' : 'GPU';
    try { await useDelegate(first); }
    catch (e) { console.warn('[Snap Vanish] GPU delegate unavailable, using CPU', e); await useDelegate('CPU'); }
    if (forced) perf.results = { GPU: 1, CPU: 1 };
    ready = true;
    post({ type: 'ready', delegate: perf.delegate });
  } catch (e) {
    initError = String(e && e.message || e);
    post({ type: 'error', error: initError });
  }
}

function notePerf(ms) {
  perf.n++;
  perf.ema = perf.n === 1 ? ms : perf.ema * 0.8 + ms * 0.2;
  if (forced) return;
  const early = perf.n >= 3 && perf.ema > 400 && !perf.early;
  if (perf.switching || (perf.n !== 15 && !early)) return;
  if (early) perf.early = true;
  perf.results[perf.delegate] = perf.ema;
  const other = perf.delegate === 'GPU' ? 'CPU' : 'GPU';
  let target = null;
  if (!(other in perf.results) && perf.ema > 45) target = other;                       // too slow: try the other one
  else if (other in perf.results && perf.results[other] < perf.ema * 0.85) target = other; // the other was faster
  if (!target) return;
  perf.switching = true;
  ready = false;
  useDelegate(target)
    .catch(() => { perf.results[target] = Infinity; return useDelegate(perf.delegate === target ? other : perf.delegate); })
    .finally(() => { perf.switching = false; ready = true; post({ type: 'ready', delegate: perf.delegate }); });
}

function post(msg, transfer) {
  if (port) port.postMessage(msg, transfer || []);
}

function nextTs() {
  const t = Math.max(lastTs + 1, Math.round(performance.now()));
  lastTs = t;
  return t;
}

function runFrame(msg) {
  const { id, bitmap, wantHands, wantSeg } = msg;
  const out = { type: 'result', id, hands: null, mask: null, mw: 0, mh: 0 };
  const transfer = [];
  if (!ready) {
    bitmap.close();
    out.notReady = true;
    post(out);
    return;
  }
  const t0 = performance.now();
  const ts = nextTs();
  try {
    if (wantHands) {
      const r = hands.detectForVideo(bitmap, ts);
      out.hands = [];
      const n = r.landmarks ? r.landmarks.length : 0;
      for (let i = 0; i < n; i++) {
        const lm = r.landmarks[i];
        const wl = r.worldLandmarks && r.worldLandmarks[i];
        const img = new Float32Array(63);
        const world = new Float32Array(63);
        for (let j = 0; j < 21; j++) {
          img[j * 3] = lm[j].x; img[j * 3 + 1] = lm[j].y; img[j * 3 + 2] = lm[j].z;
          if (wl) { world[j * 3] = wl[j].x; world[j * 3 + 1] = wl[j].y; world[j * 3 + 2] = wl[j].z; }
        }
        const hd = r.handedness && r.handedness[i] && r.handedness[i][0];
        out.hands.push({ img, world, label: hd ? hd.categoryName : 'Unknown', score: hd ? hd.score : 0 });
      }
    }
    if (wantSeg) {
      const r = segmenter.segmentForVideo(bitmap, ts);
      const masks = r.confidenceMasks || [];
      const m = masks.length > 1 ? masks[masks.length - 1] : masks[0];
      if (m) {
        const f = m.getAsFloat32Array();
        const { u8, w, h } = downsample(f, m.width, m.height, 256);
        out.mask = u8; out.mw = w; out.mh = h;
        transfer.push(u8.buffer);
      }
      if (r.close) r.close(); else masks.forEach((x) => x.close());
    }
  } catch (e) {
    out.error = String(e && e.message || e);
  }
  bitmap.close();
  out.ms = performance.now() - t0;
  out.delegate = perf.delegate;
  notePerf(out.ms);
  post(out, transfer);
}

// Box-filter a float confidence mask down to `tw` pixels wide (the segmenter's native resolution)
// so the page side has less to move and process.
function downsample(f, w, h, tw) {
  if (w <= tw) {
    const u8 = new Uint8Array(f.length);
    for (let i = 0; i < f.length; i++) u8[i] = f[i] * 255 + 0.5;
    return { u8, w, h };
  }
  const th = Math.max(1, Math.round(h * tw / w));
  const u8 = new Uint8Array(tw * th);
  const sx = w / tw, sy = h / th;
  for (let y = 0; y < th; y++) {
    const y0 = Math.floor(y * sy), y1 = Math.max(y0 + 1, Math.floor((y + 1) * sy));
    for (let x = 0; x < tw; x++) {
      const x0 = Math.floor(x * sx), x1 = Math.max(x0 + 1, Math.floor((x + 1) * sx));
      let s = 0;
      for (let yy = y0; yy < y1; yy++) { const row = yy * w; for (let xx = x0; xx < x1; xx++) s += f[row + xx]; }
      u8[y * tw + x] = (s / ((y1 - y0) * (x1 - x0))) * 255 + 0.5;
    }
  }
  return { u8, w: tw, h: th };
}

function getInpaintWorker() {
  if (inpaintWorker) return inpaintWorker;
  inpaintWorker = new Worker(new URL('inpaint-worker.js', import.meta.url), { type: 'module' });
  inpaintWorker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'loaded') { post({ type: 'inpaint-status', status: 'ready', ep: m.ep }); return; }
    if (m.type === 'load-error') { post({ type: 'inpaint-status', status: 'error', error: m.error }); return; }
    const job = inpaintJobs.get(m.id);
    inpaintJobs.delete(m.id);
    if (!job) return;
    const tr = m.rgba ? [m.rgba.buffer] : [];
    if (m.shadow) tr.push(m.shadow.buffer);
    post({ type: 'inpaint-result', id: m.id, rgba: m.rgba, shadow: m.shadow, shadowInfo: m.shadowInfo, w: m.w, h: m.h, ms: m.ms, ep: m.ep, error: m.error }, tr);
  };
  inpaintWorker.onerror = (e) => post({ type: 'inpaint-status', status: 'error', error: e.message || 'worker failed' });
  inpaintWorker.postMessage({ type: 'load' });
  return inpaintWorker;
}

function onMessage(e) {
  const msg = e.data;
  if (!msg) return;
  if (msg.type === 'frame') runFrame(msg);
  else if (msg.type === 'inpaint') {
    inpaintJobs.set(msg.id, true);
    getInpaintWorker().postMessage(msg, msg.band ? [msg.rgba.buffer, msg.band.buffer] : [msg.rgba.buffer]);
  } else if (msg.type === 'warm-inpaint') {
    getInpaintWorker();
  } else if (msg.type === 'ping') {
    post({ type: ready ? 'ready' : (initError ? 'error' : 'loading'), error: initError });
  }
}

window.addEventListener('message', (e) => {
  if (e.data && e.data.type === 'snapvanish-connect' && e.ports && e.ports[0]) {
    if (port) port.close();
    port = e.ports[0];
    port.onmessage = onMessage;
    port.start();
    if (ready) post({ type: 'ready' });
    else if (initError) post({ type: 'error', error: initError });
  }
});

init();
