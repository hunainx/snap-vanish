// Snap Vanish — background inpainting worker (MI-GAN, MIT licence, via ONNX Runtime Web).
// Fills the parts of the room the camera has never seen (the area behind you) with generated texture.
// Runs entirely on this computer: WebGPU when available, otherwise WebAssembly on the CPU.

import * as ort from '../vendor/ort/ort.webgpu.min.mjs';

ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.logLevel = 'error';

let session = null;
let ep = null;
let loading = null;
let modelBuf = null;

async function makeSession(provider) {
  if (!modelBuf) {
    const res = await fetch(new URL('../models/migan_pipeline_v2.onnx', import.meta.url));
    modelBuf = new Uint8Array(await res.arrayBuffer());
  }
  return ort.InferenceSession.create(modelBuf, { executionProviders: [provider], graphOptimizationLevel: 'all' });
}

async function load() {
  if (loading) return loading;
  loading = (async () => {
    if (self.navigator && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) { session = await makeSession('webgpu'); ep = 'webgpu'; }
      } catch (e) { session = null; }
    }
    if (!session) { session = await makeSession('wasm'); ep = 'wasm'; }
    return session;
  })();
  try {
    await loading;
    self.postMessage({ type: 'loaded', ep });
  } catch (e) {
    loading = null;
    self.postMessage({ type: 'load-error', error: String(e && e.message || e) });
    throw e;
  }
  return loading;
}

async function run(w, h, image, mask) {
  const feeds = {
    image: new ort.Tensor('uint8', image, [1, 3, h, w]),
    mask: new ort.Tensor('uint8', mask, [1, 1, h, w]),
  };
  const out = await session.run(feeds);
  return out.result || out[Object.keys(out)[0]];
}

// Binary erosion (separable min filter) of a 0/1 mask.
function erode(src, w, h, r) {
  const tmp = new Uint8Array(w * h), dst = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let v = 1;
      for (let i = Math.max(0, x - r), e = Math.min(w - 1, x + r); i <= e; i++) if (!src[row + i]) { v = 0; break; }
      tmp[row + x] = v;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let v = 1;
      for (let i = Math.max(0, y - r), e = Math.min(h - 1, y + r); i <= e; i++) if (!tmp[i * w + x]) { v = 0; break; }
      dst[y * w + x] = v;
    }
  }
  return dst;
}

// Shadow finder: inside a ring around the person, a known plate pixel counts as shadow when it is
// clearly darker than what the surrounding room predicts there (push-pull of the room with the ring
// left out). Those pixels are handed to the inpainter too, and replaced while you're vanished.
function findShadow(rgba, known, band, bw, bh, w, h) {
  const n = w * h;
  const inBand = new Uint8Array(n);
  for (let y = 0; y < h; y++) {
    const by = Math.min(bh - 1, Math.floor(y * bh / h));
    for (let x = 0; x < w; x++) {
      const bx = Math.min(bw - 1, Math.floor(x * bw / w));
      inBand[y * w + x] = band[by * bw + bx] > 127 ? 1 : 0;
    }
  }
  const img = new Uint8Array(3 * n), msk = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (known[i] && !inBand[i]) {
      msk[i] = 255; img[i] = rgba[i * 4]; img[n + i] = rgba[i * 4 + 1]; img[2 * n + i] = rgba[i * 4 + 2];
    }
  }
  pushPull(img, msk, w, h);
  const lum = (r, g, b) => 0.299 * r + 0.587 * g + 0.114 * b;
  let raw = new Uint8Array(n), count = 0;
  for (let i = 0; i < n; i++) {
    if (!inBand[i] || !known[i]) continue;
    const L = lum(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]);
    const E = lum(img[i], img[n + i], img[2 * n + i]);
    // darker by 22%+ and by at least 16 levels, with the same colour balance (a shadow, not a dark object)
    if (L < E * 0.78 && E - L > 16) {
      const s1 = rgba[i * 4] + rgba[i * 4 + 1] + rgba[i * 4 + 2] + 1, s2 = img[i] + img[n + i] + img[2 * n + i] + 1;
      const dr = rgba[i * 4] / s1 - img[i] / s2, dg = rgba[i * 4 + 1] / s1 - img[n + i] / s2;
      if (Math.abs(dr) < 0.04 && Math.abs(dg) < 0.04) { raw[i] = 1; count++; }
    }
  }
  // too little = nothing there; too much = probably dark furniture, not a shadow: leave the room alone
  if (count < n * 0.002 || count > n * 0.08) return null;
  // grow slightly so the soft edge of the shadow goes too
  const grown = new Uint8Array(n);
  const r = 3;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!raw[y * w + x]) continue;
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < w && yy < h && inBand[yy * w + xx]) grown[yy * w + xx] = 255;
    }
  }
  return grown;
}

// Push-pull diffusion: fills holes with smooth colour bled in from the surroundings. MI-GAN gives far
// better results when the hole starts out with plausible colour instead of black.
function pushPull(image, mask, w, h) {
  const levels = [];
  let cw = w, ch = h;
  let col = new Float32Array(w * h * 3);
  let wt = new Float32Array(w * h);
  const n = w * h;
  for (let i = 0; i < n; i++) {
    if (mask[i]) { wt[i] = 1; col[i * 3] = image[i]; col[i * 3 + 1] = image[n + i]; col[i * 3 + 2] = image[2 * n + i]; }
  }
  levels.push({ col, wt, w: cw, h: ch });
  while (cw > 1 && ch > 1) {
    const nw = cw >> 1, nh = ch >> 1;
    const ncol = new Float32Array(nw * nh * 3), nwt = new Float32Array(nw * nh);
    for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
      const o = y * nw + x;
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const s = (y * 2 + dy) * cw + (x * 2 + dx);
        nwt[o] += wt[s];
        ncol[o * 3] += col[s * 3]; ncol[o * 3 + 1] += col[s * 3 + 1]; ncol[o * 3 + 2] += col[s * 3 + 2];
      }
    }
    col = ncol; wt = nwt; cw = nw; ch = nh;
    levels.push({ col, wt, w: cw, h: ch });
  }
  // pull: from coarsest to finest, fill uncovered pixels with the upsampled coarser estimate
  let top = levels[levels.length - 1];
  let est = new Float32Array(top.w * top.h * 3);
  for (let i = 0; i < top.w * top.h; i++) {
    const k = top.wt[i] > 1e-6 ? 1 / top.wt[i] : 0;
    est[i * 3] = top.col[i * 3] * k; est[i * 3 + 1] = top.col[i * 3 + 1] * k; est[i * 3 + 2] = top.col[i * 3 + 2] * k;
  }
  for (let l = levels.length - 2; l >= 0; l--) {
    const L = levels[l], P = levels[l + 1];
    const out = new Float32Array(L.w * L.h * 3);
    for (let y = 0; y < L.h; y++) for (let x = 0; x < L.w; x++) {
      const i = y * L.w + x;
      // bilinear sample of the coarser estimate
      const fx = Math.min(P.w - 1, Math.max(0, (x + 0.5) / 2 - 0.5)), fy = Math.min(P.h - 1, Math.max(0, (y + 0.5) / 2 - 0.5));
      const x0 = Math.floor(fx), y0 = Math.floor(fy), x1 = Math.min(P.w - 1, x0 + 1), y1 = Math.min(P.h - 1, y0 + 1);
      const ax = fx - x0, ay = fy - y0;
      const a = Math.min(1, L.wt[i]);
      const k = L.wt[i] > 1e-6 ? 1 / L.wt[i] : 0;
      for (let c = 0; c < 3; c++) {
        const up = (est[(y0 * P.w + x0) * 3 + c] * (1 - ax) + est[(y0 * P.w + x1) * 3 + c] * ax) * (1 - ay)
          + (est[(y1 * P.w + x0) * 3 + c] * (1 - ax) + est[(y1 * P.w + x1) * 3 + c] * ax) * ay;
        out[i * 3 + c] = L.col[i * 3 + c] * k * a + up * (1 - a);
      }
    }
    est = out;
  }
  for (let i = 0; i < n; i++) {
    if (!mask[i]) { image[i] = est[i * 3]; image[n + i] = est[i * 3 + 1]; image[2 * n + i] = est[i * 3 + 2]; }
  }
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'load') { load().catch(() => {}); return; }
  if (m.type !== 'inpaint') return;
  const { id, rgba, w, h, band, bw, bh } = m;
  const t0 = performance.now();
  try {
    await load();
    const n = w * h;
    const image = new Uint8Array(3 * n);
    const mask = new Uint8Array(n);
    let holes = 0;
    // Treat a thin fringe around every hole as unknown too: edges of the learned plate are the least
    // trustworthy pixels (soft hair, motion blur), and the model blends better from clean context.
    const known0 = new Uint8Array(n);
    for (let i = 0; i < n; i++) known0[i] = rgba[i * 4 + 3] > 127 ? 1 : 0;
    const knownE = erode(known0, w, h, Math.max(3, Math.round(w / 110)));
    const shadow = band ? findShadow(rgba, knownE, band, bw, bh, w, h) : null;
    if (shadow) for (let i = 0; i < n; i++) if (shadow[i]) knownE[i] = 0;
    for (let i = 0; i < n; i++) {
      const known = knownE[i] === 1;
      mask[i] = known ? 255 : 0;
      if (known) {
        image[i] = rgba[i * 4];
        image[n + i] = rgba[i * 4 + 1];
        image[2 * n + i] = rgba[i * 4 + 2];
      } else holes++;
    }
    let result;
    if (holes === 0) {
      result = rgba;
      for (let i = 0; i < n; i++) result[i * 4 + 3] = 0;
    } else {
      pushPull(image, mask, w, h);
      let t;
      try {
        t = await run(w, h, image, mask);
      } catch (err) {
        if (ep !== 'wasm') {
          // Some GPUs can't run every op — fall back to the CPU once and stay there.
          session = await makeSession('wasm'); ep = 'wasm';
          t = await run(w, h, image, mask);
        } else throw err;
      }
      const d = t.data;
      result = new Uint8Array(4 * n);
      for (let i = 0; i < n; i++) {
        result[i * 4] = d[i];
        result[i * 4 + 1] = d[n + i];
        result[i * 4 + 2] = d[2 * n + i];
        result[i * 4 + 3] = mask[i] ? 0 : 255; // 255 = generated by the model
      }
    }
    const msgOut = { id, rgba: result, w, h, ms: performance.now() - t0, ep };
    const tr = [result.buffer];
    if (shadow) { msgOut.shadow = shadow; tr.push(shadow.buffer); }
    msgOut.shadowInfo = band ? (shadow ? 'found' : 'none') : 'off';
    self.postMessage(msgOut, tr);
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message || err), ms: performance.now() - t0, ep });
  }
};
