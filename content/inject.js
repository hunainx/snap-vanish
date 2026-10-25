// Snap Vanish — page-side pipeline (runs in the page's MAIN world at document_start).
//
// What it does:
//   • Wraps navigator.mediaDevices.getUserMedia. When the meeting asks for your camera, it gets a
//     processed copy of the camera instead (same device, same resolution).
//   • Every frame is passed straight through until you snap. Meanwhile it quietly learns what the
//     room behind you looks like (a "clean plate"), and an on-device inpainting model fills in the
//     parts the camera has never seen.
//   • A real finger snap (hand-tracking gesture, confirmed by the click sound) turns you into dust
//     and mutes your mic. Snap again and you reassemble and the mic comes back.
//
// Privacy: frames never leave this computer. The ML runs in an extension-origin iframe on this
// machine, models are bundled with the extension, and nothing here makes a network request.

(() => {
  'use strict';

  // ───────────────────────────── shared helpers ─────────────────────────────
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const lerp = (a, b, t) => a + (b - a) * t;
  const DEFAULTS = {
    enabled: true,
    sensitivity: 5,       // 1 (strict) … 10 (easy)
    duration: 3,          // seconds
    direction: 'right',   // right | left | up | scatter
    requireSound: true,   // confirm the gesture with the snap's click sound
    syncMuteButton: true, // also press the meeting app's own mute button
    clearShadow: false,   // also remove your shadow from the wall (experimental)
  };

  // ───────────────────────────── snap gesture detector ─────────────────────────────
  // Works on MediaPipe *world* landmarks (metres, hand-centred), so it doesn't care how far the hand
  // is from the camera. A snap is: thumb tip pressed against the middle fingertip (the "set"), then,
  // within a fraction of a second, the middle finger slams down towards the palm while the thumb flies
  // apart from it. Waving, typing, pointing, pinching and slowly making a fist don't match that.
  const dist3 = (w, a, b) => {
    const dx = w[a * 3] - w[b * 3], dy = w[a * 3 + 1] - w[b * 3 + 1], dz = w[a * 3 + 2] - w[b * 3 + 2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };

  function handFeatures(world) {
    const palm = Math.max(dist3(world, 0, 9), 1e-4);         // wrist → middle knuckle
    return {
      palm,
      dTM: dist3(world, 4, 12) / palm,                         // thumb tip ↔ middle tip
      cMid: dist3(world, 12, 0) / palm,                        // middle tip ↔ wrist (curl)
      cRing: dist3(world, 16, 0) / palm,                       // ring tip ↔ wrist
      dTI: dist3(world, 4, 8) / palm,                          // thumb tip ↔ index tip
    };
  }

  class SnapDetector {
    constructor() {
      this.hands = new Map();
      this.setSensitivity(5);
      this.last = null;
    }
    setSensitivity(s) {
      const k = (clamp(s, 1, 10) - 1) / 9; // 0 = strict, 1 = easy
      this.k = k;
      this.armThr = lerp(0.30, 0.46, k);     // set pose: thumb–middle distance
      this.minArmFrames = k > 0.6 ? 1 : 2;
      this.releaseThr = lerp(0.58, 0.45, k);  // after: thumb–middle separation
      this.curlDrop = lerp(0.34, 0.16, k);   // middle finger must drop this much towards the wrist
      this.curlMax = lerp(1.12, 1.3, k);     // …and end up this close to the palm
      this.maxGapMs = lerp(170, 300, k);     // set → release must be this quick
      this.scoreThr = lerp(0.6, 0.38, k);
    }
    // hands: [{label, world: Float32Array(63)}], t: ms
    update(t, hands) {
      const events = [];
      const seen = new Set();
      for (const h of hands || []) {
        let key = h.label || 'Unknown';
        if (seen.has(key)) key += '#2';
        seen.add(key);
        const f = handFeatures(h.world);
        const ev = this.updateFeatures(key, t, f);
        if (ev) events.push(ev);
      }
      // forget hands that left the frame for too long
      for (const [key, st] of this.hands) {
        if (!seen.has(key) && st.lastArmT && t - st.lastArmT > this.maxGapMs * 1.7) this.hands.delete(key);
      }
      return events;
    }
    updateFeatures(key, t, f) {
      let st = this.hands.get(key);
      if (!st) { st = { armFrames: 0, lastArmT: 0, armD: 0, armC: 0, prevT: t }; this.hands.set(key, st); }
      const armedPose = f.dTM < this.armThr && f.cMid > 1.05 && f.cMid < 2.0;
      let stage = 'idle';
      let ev = null;
      if (armedPose) {
        // Keep the tightest, freshest "set" values.
        const gap = t - st.lastArmT;
        if (st.armFrames === 0 || gap > this.maxGapMs) st.armFrames = 0;
        st.armFrames++;
        st.lastArmT = t;
        st.armD = f.dTM;
        st.armC = f.cMid;
        stage = st.armFrames >= this.minArmFrames ? 'set' : 'setting';
      } else if (st.armFrames >= this.minArmFrames) {
        const gap = t - st.lastArmT;
        const missed = (t - st.prevT) > 80; // hand dropped out for a frame or two (motion blur)
        const maxGap = missed ? this.maxGapMs * 1.6 : this.maxGapMs;
        if (gap <= maxGap) {
          const sep = f.dTM;
          const drop = st.armC - f.cMid;
          const ok = sep > this.releaseThr && drop > this.curlDrop && f.cMid < this.curlMax;
          if (ok) {
            const s1 = clamp((sep - st.armD) / 0.8, 0, 1);
            const s2 = clamp(drop / 0.7, 0, 1);
            const s3 = clamp(1 - gap / (this.maxGapMs * 1.6), 0, 1);
            const bonus = f.cRing < 1.25 ? 0.08 : 0;
            const score = clamp(0.38 * s1 + 0.38 * s2 + 0.24 * s3 + bonus, 0, 1);
            stage = score >= this.scoreThr ? 'snap' : 'weak';
            if (score >= this.scoreThr) ev = { t, tArm: st.lastArmT, score, key };
            st.armFrames = 0;
          } else {
            stage = 'released?';
            // A slow open/close is not a snap: once the hand clearly moved on, disarm.
            if (f.dTM > this.armThr * 1.6 && f.cMid > st.armC - 0.05) st.armFrames = 0;
          }
        } else {
          st.armFrames = 0;
        }
      } else {
        st.armFrames = 0;
      }
      st.prevT = t;
      this.last = { key, t, dTM: f.dTM, cMid: f.cMid, stage, armThr: this.armThr, releaseThr: this.releaseThr };
      return ev;
    }
  }

  // ───────────────────────────── snap sound (click) detector ─────────────────────────────
  // A finger snap is a very short, sharp, broadband click: energy above ~1.5 kHz jumps 10–20 dB over
  // the room's noise floor in a few milliseconds and dies away within ~50 ms. Speech and music don't
  // look like that. This is only ever used to *confirm* a hand-tracking snap.
  class ClickDetector {
    constructor() {
      this.sr = 48000;
      this.setSensitivity(5);
      this.reset();
    }
    reset() {
      this.hpY = 0; this.hpX = 0;
      this.acc = 0; this.peak = 0; this.n = 0;
      this.floor = 1e-6;
      this.hist = [];
      this.cand = null;
      this.clicks = [];
      this.level = 0;
      this.available = false;
    }
    setSensitivity(s) {
      const k = (clamp(s, 1, 10) - 1) / 9;
      this.K = lerp(40, 9, k);          // onset must be this many × the noise floor (≈16 dB → 9.5 dB)
      this.minPeak = lerp(0.05, 0.012, k);
    }
    setRate(sr) {
      if (sr === this.sr && this.alpha) return;
      this.sr = sr;
      const rc = 1 / (2 * Math.PI * 1500);
      const dt = 1 / sr;
      this.alpha = rc / (rc + dt);
      this.hop = Math.max(32, Math.round(sr * 0.004)); // 4 ms blocks
    }
    // samples: Float32Array (mono), tEnd: performance.now() at the last sample
    process(samples, sr, tEnd) {
      this.setRate(sr);
      const a = this.alpha, hop = this.hop;
      const tStart = tEnd - (samples.length / sr) * 1000;
      for (let i = 0; i < samples.length; i++) {
        const x = samples[i];
        const y = a * (this.hpY + x - this.hpX);
        this.hpX = x; this.hpY = y;
        this.acc += y * y;
        const ay = y < 0 ? -y : y;
        if (ay > this.peak) this.peak = ay;
        if (++this.n >= hop) {
          this.block(this.acc / hop, this.peak, tStart + (i / sr) * 1000);
          this.acc = 0; this.peak = 0; this.n = 0;
        }
      }
    }
    block(e, peak, t) {
      this.level = this.level * 0.8 + Math.sqrt(e) * 0.2;
      const h = this.hist;
      let prev = 0;
      const nPrev = Math.min(5, h.length);
      for (let i = h.length - nPrev; i < h.length; i++) prev += h[i];
      prev = nPrev ? prev / nPrev : e;
      if (this.cand) {
        const c = this.cand;
        c.blocks++;
        if (e > c.maxE) c.maxE = e;
        if (e < c.maxE * 0.18) c.decayed = true;
        if (c.blocks >= 12) { // ~48 ms after onset
          if (c.decayed) this.emit(c.t);
          this.cand = null;
        }
      } else if (e > this.floor * this.K && peak > this.minPeak && e > prev * 6) {
        this.cand = { t, maxE: e, blocks: 0, decayed: false };
      }
      if (!this.cand) {
        this.floor = e < this.floor ? this.floor * 0.8 + e * 0.2 : this.floor * 0.997 + e * 0.003;
        if (this.floor < 1e-9) this.floor = 1e-9;
      }
      h.push(e);
      if (h.length > 16) h.shift();
    }
    emit(t) {
      this.clicks.push(t);
      if (this.clicks.length > 20) this.clicks.shift();
      if (this.onClick) this.onClick(t);
    }
    hasClickIn(a, b) {
      for (let i = this.clicks.length - 1; i >= 0; i--) {
        const t = this.clicks[i];
        if (t >= a && t <= b) return true;
        if (t < a) break;
      }
      return false;
    }
  }

  // Node test hook — no browser APIs below this line run under Node.
  if (typeof window === 'undefined' || typeof navigator === 'undefined' || !navigator.mediaDevices) {
    if (typeof module !== 'undefined') module.exports = { SnapDetector, ClickDetector, handFeatures, DEFAULTS };
    return;
  }

  const FLAG = Symbol.for('snapvanish.installed');
  if (window[FLAG]) return;
  Object.defineProperty(window, FLAG, { value: true });

  const TAG = '[Snap Vanish]';
  const TO_MAIN = 'snapvanish:to-main';
  const TO_ISO = 'snapvanish:to-iso';
  const now = () => performance.now();
  const warned = new Set();
  const warnOnce = (k, ...a) => { if (!warned.has(k)) { warned.add(k); console.warn(TAG, ...a); } };

  // ───────────────────────────── bridge to the extension ─────────────────────────────
  let settings = { ...DEFAULTS };
  let base = null;
  let debugMode = false;
  let engineQuery = '';
  let resolveSettings;
  const settingsReady = new Promise((r) => (resolveSettings = r));

  function toIso(msg) {
    document.dispatchEvent(new CustomEvent(TO_ISO, { detail: JSON.stringify(msg) }));
  }

  function applySettings(s) {
    settings = { ...DEFAULTS, ...settings, ...s };
    settings.sensitivity = clamp(+settings.sensitivity || 5, 1, 10);
    settings.duration = clamp(+settings.duration || 3, 1, 10);
    core.detector.setSensitivity(settings.sensitivity);
    core.click.setSensitivity(settings.sensitivity);
    if (!settings.enabled && core.state !== 'visible') core.forceVisible();
  }

  document.addEventListener(TO_MAIN, (e) => {
    let msg;
    try { msg = JSON.parse(e.detail); } catch (_) { return; }
    if (!msg || !msg.type) return;
    switch (msg.type) {
      case 'config':
        base = msg.base;
        debugMode = !!msg.debug;
        if (debugMode && typeof msg.engineQuery === 'string') engineQuery = msg.engineQuery;
        applySettings(msg.settings || {});
        resolveSettings();
        break;
      case 'settings':
        applySettings(msg.settings || {});
        break;
      case 'snap':
        core.toggle('manual');
        break;
      case 'reset-room':
        core.resetRoom();
        break;
      case 'get-status':
        toIso({ type: 'status', reqId: msg.reqId, status: core.status() });
        break;
      case 'debug-inpaint':
        if (debugMode && core.lastInpaintDbg) {
          const d = core.lastInpaintDbg;
          const b64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
          toIso({ type: 'debug-inpaint', w: d.w, h: d.h, input: b64(d.input), output: b64(d.output) });
        }
        break;
      case 'debug-dump':
        if (debugMode && core.mask) {
          const m = core.mask;
          toIso({ type: 'debug-dump', w: m.w, h: m.h, mask: Array.from(m.u8), age: now() - core.lastMask, mlSent: core.mlSent.size });
        }
        break;
    }
  });

  // ───────────────────────────── ML engine client ─────────────────────────────
  class Engine {
    constructor(baseUrl) {
      this.url = baseUrl + 'engine/engine.html' + (engineQuery || '');
      this.origin = new URL(baseUrl).origin;
      this.status = 'loading';
      this.error = null;
      this.inflight = false;
      this.inflightSince = 0;
      this.id = 0;
      this.inpaintCb = new Map();
      this.inpaintStatus = 'idle';
      this.inpaintEp = null;
      this.mount();
    }
    mount() {
      const f = document.createElement('iframe');
      f.src = this.url;
      f.setAttribute('aria-hidden', 'true');
      f.tabIndex = -1;
      f.allow = '';
      f.style.cssText = 'position:fixed!important;left:-4px!important;top:-4px!important;width:2px!important;height:2px!important;opacity:0!important;pointer-events:none!important;border:0!important;z-index:-1!important;';
      f.addEventListener('load', () => this.connect());
      this.frame = f;
      (document.documentElement || document).appendChild(f);
      this.watch = setInterval(() => {
        if (!this.frame.isConnected) {
          this.status = 'loading';
          (document.documentElement || document).appendChild(this.frame);
        }
      }, 2000);
    }
    connect() {
      if (this.port) this.port.close();
      const ch = new MessageChannel();
      this.port = ch.port1;
      this.port.onmessage = (e) => this.onMessage(e.data);
      this.port.start();
      this.inflight = false;
      this.frame.contentWindow.postMessage({ type: 'snapvanish-connect' }, this.origin, [ch.port2]);
    }
    onMessage(m) {
      switch (m.type) {
        case 'ready': this.status = 'ready'; this.error = null; break;
        case 'error': this.status = 'error'; this.error = m.error; console.warn(TAG, 'engine:', m.error); break;
        case 'loading': this.status = 'loading'; break;
        case 'result':
          this.inflight = false;
          if (m.notReady) return;
          if (m.error) warnOnce('ml', 'ML frame error', m.error);
          core.onResult(m);
          break;
        case 'inpaint-status':
          if (m.status === 'error') { this.inpaintStatus = 'error'; console.warn(TAG, 'inpaint:', m.error); }
          else if (this.inpaintStatus !== 'working') this.inpaintStatus = 'ready';
          if (m.ep) this.inpaintEp = m.ep;
          break;
        case 'inpaint-result': {
          const cb = this.inpaintCb.get(m.id);
          this.inpaintCb.delete(m.id);
          this.inpaintStatus = m.error ? 'error' : 'ready';
          if (m.ep) this.inpaintEp = m.ep;
          if (cb) cb(m);
          break;
        }
      }
    }
    sendFrame(bitmap, wantHands, wantSeg, t) {
      if (!this.port) { bitmap.close(); this.inflight = false; return; }
      const id = ++this.id;
      this.port.postMessage({ type: 'frame', id, bitmap, wantHands, wantSeg, t }, [bitmap]);
      core.mlSent.set(id, { t, wantSeg });
    }
    inpaint(rgba, w, h, band, cb) {
      if (!this.port) return false;
      const id = ++this.id;
      this.inpaintCb.set(id, cb);
      this.inpaintStatus = 'working';
      const msg = { type: 'inpaint', id, rgba, w, h };
      const transfer = [rgba.buffer];
      if (band) { msg.band = band.data; msg.bw = band.w; msg.bh = band.h; transfer.push(band.data.buffer); }
      this.port.postMessage(msg, transfer);
      return true;
    }
    warmInpaint() { if (this.port) this.port.postMessage({ type: 'warm-inpaint' }); }
  }

  // ───────────────────────────── WebGL renderer ─────────────────────────────
  const GLSL_COMMON = `
    uniform vec4 uBox;      // person bounding box in uv (minx, miny, maxx, maxy)
    uniform vec2 uDir;      // wind direction (image space, y down)
    uniform float uMode;    // 0 = directional, 1 = scatter
    uniform float uAspect;  // width / height
    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
    float vnoise(vec2 p) {
      vec2 i = floor(p), f = fract(p);
      vec2 u = f * f * (3.0 - 2.0 * f);
      return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
    }
    float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p *= 2.03; a *= 0.5; } return s; }
    // 0 = turns to dust first, 1 = last
    float delayAt(vec2 uv) {
      vec2 c = (uBox.xy + uBox.zw) * 0.5;
      vec2 ext = max(uBox.zw - uBox.xy, vec2(0.05)) * vec2(uAspect, 1.0);
      vec2 p = (uv - c) * vec2(uAspect, 1.0);
      float base;
      if (uMode > 0.5) {
        base = 1.0 - clamp(length(p / (0.5 * ext)), 0.0, 1.0);
      } else {
        float e = abs(uDir.x) * ext.x + abs(uDir.y) * ext.y;
        base = 0.5 - dot(p, uDir) / max(e, 1e-3);
      }
      float n = fbm(uv * vec2(uAspect, 1.0) * 7.0);
      return clamp(base * 0.62 + (n - 0.45) * 0.75 + 0.2, 0.0, 1.0);
    }
  `;

  const VS_QUAD = `#version 300 es
    in vec2 aPos;
    uniform float uFlipY;
    out vec2 vUv;
    void main() {
      vec2 uv = aPos * 0.5 + 0.5;
      vUv = vec2(uv.x, uFlipY > 0.5 ? 1.0 - uv.y : uv.y);
      gl_Position = vec4(aPos, 0.0, 1.0);
    }`;

  // Clean-plate learning. A pixel only joins the plate once it has been confidently "background" AND
  // its colour has held steady for several updates — that keeps moving hair/edges/shadows of you
  // from being baked into the empty room. Two outputs: the plate itself and a per-pixel candidate.
  const FS_PLATE = `#version 300 es
    precision highp float;
    in vec2 vUv;
    layout(location = 0) out vec4 oPlate;
    layout(location = 1) out vec4 oCand;
    layout(location = 2) out vec4 oE;
    uniform sampler2D uPrev, uCand, uPrevE, uLive, uMaskP;
    uniform float uRate, uReset, uE;
    void main() {
      vec4 prev = uReset > 0.5 ? vec4(0.0) : texture(uPrev, vUv);
      vec4 cand = uReset > 0.5 ? vec4(0.0) : texture(uCand, vUv);
      vec4 pe = uReset > 0.5 ? vec4(0.5, 0.0, 0.0, 1.0) : texture(uPrevE, vUv);
      vec3 live = texture(uLive, vUv).rgb;
      float m = texture(uMaskP, vUv).r;
      bool bg = m < 0.06;
      float count = cand.a * 10.0;
      if (bg) {
        if (distance(live, cand.rgb) < 0.07) { count = min(count + 1.0, 10.0); cand.rgb = mix(cand.rgb, live, 0.35); }
        else { cand.rgb = live; count = 0.0; }
      } else {
        count = 0.0;
      }
      bool stable = bg && count >= 3.0;
      vec4 plate = prev;
      if (stable) plate = vec4(prev.a > 0.5 ? mix(prev.rgb, cand.rgb, uRate) : cand.rgb, 1.0);
      oPlate = plate;
      oCand = vec4(cand.rgb, count / 10.0);
      oE = stable ? vec4(uE * 0.5, 0.0, 0.0, 1.0) : pe;
    }`;

  const FS_PREMUL = `#version 300 es
    precision highp float;
    in vec2 vUv; out vec4 o;
    uniform sampler2D uPlate, uPlateE;
    uniform float uENow;
    void main() {
      vec4 c = texture(uPlate, vUv); float a = step(0.5, c.a);
      vec3 rgb = clamp(c.rgb * (uENow / max(texture(uPlateE, vUv).r * 2.0, 0.2)), 0.0, 1.0); // all in today's light
      o = vec4(rgb * a, a);
    }`;

  // Lighting match: per-cell averages of the live frame over pixels that are visible room (not you).
  // Their median change against a baseline is the room's exposure index E. Every plate pixel stores
  // the E it was last seen at, and the AI fill stores the E of the moment it was made, so anything
  // drawn from the past is scaled by E_now / E_then and follows today's lighting.
  const FS_STATS = `#version 300 es
    precision highp float;
    in vec2 vUv;
    layout(location = 0) out vec4 oLive;
    layout(location = 1) out vec4 oPlate;
    uniform sampler2D uLive, uPlate, uMaskP;
    uniform vec2 uCell;
    void main() {
      vec3 sl = vec3(0.0), sp = vec3(0.0); float sw = 0.0;
      for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) {
        vec2 uv = vUv + (vec2(float(i), float(j)) - 1.5) / 4.0 * uCell;
        vec4 p = texture(uPlate, uv);
        float w = (1.0 - smoothstep(0.03, 0.15, texture(uMaskP, uv).r)) * step(0.5, p.a);
        sl += texture(uLive, uv).rgb * w; sp += p.rgb * w; sw += w;
      }
      oLive = vec4(sl / 16.0, sw / 16.0);
      oPlate = vec4(sp / 16.0, sw / 16.0);
    }`;

  const FS_COMPOSITE = `#version 300 es
    precision highp float;
    in vec2 vUv; out vec4 o;
    uniform sampler2D uLive, uMask, uMaskD, uPlate, uPlateE, uFill, uPP, uShadow;
    uniform float uFillMix, uBgMix, uFront, uEdge, uShadowOn, uENow, uFillE;
    ${GLSL_COMMON}
    vec3 pushPull(vec2 uv) {
      for (int l = 1; l <= 10; l++) {
        vec4 c = textureLod(uPP, uv, float(l));
        if (c.a > 0.004) return c.rgb / c.a;
      }
      return vec3(0.45);
    }
    vec3 room(vec2 uv) {
      vec4 p = texture(uPlate, uv);
      float k = smoothstep(0.35, 0.9, p.a);
      float pe = max(texture(uPlateE, uv).r * 2.0, 0.2);
      vec3 seen = clamp(p.rgb * (uENow / pe), 0.0, 1.0); // the room as last seen, in today's light
      vec3 base = k > 0.999 ? seen : mix(pushPull(uv), seen, k); // (push-pull is already in today's light)
      if (uFillMix < 0.001) return base;
      vec4 f = texture(uFill, uv);                       // alpha = 1 where the AI generated the pixel
      vec3 lit = clamp(f.rgb * (uENow / max(uFillE, 0.2)), 0.0, 1.0);
      return mix(base, lit, uFillMix * smoothstep(0.2, 0.8, f.a));
    }
    vec3 roomLit(vec2 uv) { return room(uv); }
    void main() {
      vec3 live = texture(uLive, vUv).rgb;
      float m = smoothstep(0.3, 0.75, texture(uMask, vUv).r);
      float md = smoothstep(0.04, 0.35, texture(uMaskD, vUv).r);
      md = max(md, smoothstep(0.3, 0.7, texture(uShadow, vUv).r) * uShadowOn * uFillMix);
      vec3 empty = mix(live, roomLit(vUv), md * uBgMix);
      float d = delayAt(vUv);
      float remain = smoothstep(uFront, uFront + uEdge, d);
      float band = remain * (1.0 - smoothstep(uFront + uEdge, uFront + uEdge * 3.5, d));
      float rim = remain * (1.0 - smoothstep(uFront + uEdge, uFront + uEdge * 1.6, d));
      vec3 ash = vec3(0.27, 0.24, 0.22);
      vec3 body = mix(live, live * 0.5 + ash * 0.25, band * 0.85);
      body += vec3(0.42, 0.15, 0.05) * rim * 0.3;   // a faint smouldering edge where you crumble
      o = vec4(mix(empty, clamp(body, 0.0, 1.0), m * remain), 1.0);
    }`;

  const VS_DUST = `#version 300 es
    precision highp float;
    uniform sampler2D uSnap, uSnapMask;
    uniform ivec2 uGrid;
    uniform vec2 uRes;
    uniform float uTau, uSpread, uEdge, uPt;
    ${GLSL_COMMON}
    out vec4 vCol;
    out vec2 vDir;
    out float vStretch;
    flat out float vShape;
    vec2 drift(vec2 uv, vec2 dir, float a, float r1, float r2, float r3, float r4, float diag) {
      float L = diag * mix(0.3, 0.9, r4);
      float e = a * a * (0.55 + 0.45 * a);
      vec2 perp = vec2(-dir.y, dir.x);
      vec2 off = dir * L * e;
      off += perp * sin(a * 5.0 + r1 * 6.2831) * diag * 0.03 * a * (0.5 + r2);
      off += vec2(sin(uv.y * 23.0 + a * 6.0 + r3 * 5.0), cos(uv.x * 19.0 + a * 5.0 + r1 * 4.0)) * diag * 0.018 * a;
      off.y -= diag * 0.09 * a * a * mix(0.3, 1.0, r2);
      return off;
    }
    void main() {
      int gw = uGrid.x;
      vec2 g = vec2(float(gl_VertexID % gw), float(gl_VertexID / gw));
      float r1 = hash(g), r2 = hash(g + 17.31), r3 = hash(g + 41.77), r4 = hash(g + 3.13);
      vec2 uv = (g + vec2(r1, r2)) / vec2(uGrid);
      float m = texture(uSnapMask, uv).r;
      float d = delayAt(uv);
      float ts = uSpread * d / (1.0 + uEdge);
      float life = mix(0.27, 0.41, r3);
      float a = (uTau - ts) / life;
      if (m < 0.5 || a <= 0.0 || a >= 1.0) { gl_Position = vec4(3.0, 3.0, 3.0, 1.0); gl_PointSize = 0.0; return; }
      vec3 c = texture(uSnap, uv).rgb;
      vec2 p = uv * uRes;
      vec2 dir = uDir;
      if (uMode > 0.5) {
        vec2 cc = (uBox.xy + uBox.zw) * 0.5 * uRes;
        dir = normalize(p - cc + vec2(0.01, -0.01));
        dir = normalize(dir + vec2(-dir.y, dir.x) * (r4 - 0.5) * 1.2);
      }
      float diag = length(uRes);
      vec2 off = drift(uv, dir, a, r1, r2, r3, r4, diag);
      vec2 step2 = drift(uv, dir, min(a + 0.02, 1.0), r1, r2, r3, r4, diag) - off;
      p += off;
      gl_Position = vec4(p.x / uRes.x * 2.0 - 1.0, 1.0 - p.y / uRes.y * 2.0, 0.0, 1.0);
      // flake size: mostly fine grains, some flakes, a few clumps that break up quickly
      float big = r4 > 0.985 ? 3.2 : (r4 > 0.93 ? 2.0 : 1.0);
      float shrink = r4 > 0.985 ? 0.8 : 0.5;
      float size = uPt * mix(0.8, 1.7, r2) * big * (1.0 - shrink * a);
      // motion streak: stretch the flake along its direction of travel
      float sl = length(step2);
      float stretch = clamp(1.0 + sl / max(size * 0.9, 1.0), 1.0, 3.5);
      vDir = sl > 0.001 ? step2 / sl : vec2(1.0, 0.0);
      vStretch = stretch;
      gl_PointSize = size * stretch;
      vec3 ash = vec3(0.31, 0.28, 0.25) * mix(0.65, 1.3, r1);
      vec3 col = mix(c * 0.92, ash, smoothstep(0.03, 0.42, a));
      col = mix(col, vec3(0.5, 0.2, 0.08), (1.0 - smoothstep(0.0, 0.1, a)) * 0.22);
      float alpha = (1.0 - smoothstep(0.5, 1.0, a)) * mix(0.7, 1.0, r3) * (0.82 + 0.18 * sin(a * 31.0 + r1 * 6.2831));
      vCol = vec4(col, alpha);
      vShape = r2;
    }`;

  const FS_DUST = `#version 300 es
    precision highp float;
    in vec4 vCol; in vec2 vDir; in float vStretch; flat in float vShape; out vec4 o;
    void main() {
      vec2 q = gl_PointCoord - 0.5;
      vec2 k = vec2(dot(q, vDir), dot(q, vec2(-vDir.y, vDir.x)) * vStretch);
      float r = vShape > 0.5 ? max(abs(k.x), abs(k.y)) : length(k);
      float al = vCol.a * (1.0 - smoothstep(0.28, 0.5, r));
      if (al < 0.01) discard;
      o = vec4(vCol.rgb * al, al);
    }`;

  class Renderer {
    constructor() {
      this.canvas = new OffscreenCanvas(16, 16);
      const gl = this.canvas.getContext('webgl2', {
        alpha: false, antialias: false, depth: false, stencil: false,
        premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance',
      });
      if (!gl) throw new Error('WebGL2 unavailable');
      this.gl = gl;
      this.canvas.addEventListener('webglcontextlost', (e) => { e.preventDefault(); this.lost = true; });
      this.progPlate = this.program(VS_QUAD, FS_PLATE);
      this.progPremul = this.program(VS_QUAD, FS_PREMUL);
      this.progComp = this.program(VS_QUAD, FS_COMPOSITE);
      this.progDust = this.program(VS_DUST, FS_DUST);
      this.progStats = this.program(VS_QUAD, FS_STATS);
      this.quad = gl.createVertexArray();
      gl.bindVertexArray(this.quad);
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      for (const p of [this.progPlate, this.progPremul, this.progComp, this.progStats]) {
        const loc = gl.getAttribLocation(p.prog, 'aPos');
        if (loc >= 0) { gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0); }
      }
      this.empty = gl.createVertexArray();
      gl.bindVertexArray(null);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      this.texLive = this.tex();
      this.texSnap = this.tex();
      this.texMask = this.tex(); this.texMaskD = this.tex(); this.texMaskP = this.tex(); this.texSnapMask = this.tex();
      this.texFill = this.tex();
      this.texShadow = this.tex();
      this.resetExposure();
      this.uploadR8(this.texShadow, new Uint8Array(1), 1, 1);
      this.W = 0; this.H = 0; this.PW = 0; this.PH = 0;
      this.maskVer = -1;
      this.fillReady = false;
      this.fillMix = 0;
      this.plateFrames = 0;
      this.plateDirty = false;
      this.box = [0.3, 0.2, 0.7, 1.0];
      for (const t of [this.texMask, this.texMaskD, this.texMaskP, this.texSnapMask]) this.uploadR8(t, new Uint8Array(1), 1, 1);
    }
    tex() {
      const gl = this.gl, t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
      return t;
    }
    program(vs, fs) {
      const gl = this.gl;
      const sh = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s));
        return s;
      };
      const prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
      gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
      const locs = {};
      const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
      for (let i = 0; i < n; i++) {
        const info = gl.getActiveUniform(prog, i);
        locs[info.name] = gl.getUniformLocation(prog, info.name);
      }
      return { prog, locs };
    }
    use(p, uniforms, textures) {
      const gl = this.gl;
      gl.useProgram(p.prog);
      let unit = 0;
      for (const [name, t] of Object.entries(textures || {})) {
        if (!(name in p.locs)) continue;
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.uniform1i(p.locs[name], unit++);
      }
      for (const [name, v] of Object.entries(uniforms || {})) {
        const loc = p.locs[name];
        if (!loc) continue;
        if (typeof v === 'number') gl.uniform1f(loc, v);
        else if (v.int) (v.length === 2 ? gl.uniform2i(loc, v[0], v[1]) : gl.uniform1i(loc, v[0]));
        else if (v.length === 2) gl.uniform2f(loc, v[0], v[1]);
        else if (v.length === 4) gl.uniform4f(loc, v[0], v[1], v[2], v[3]);
      }
    }
    uploadR8(t, data, w, h) {
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, data);
    }
    rgbaTarget(w, h, mip, withCand) {
      const gl = this.gl;
      const mk = () => {
        const t = this.tex();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        return t;
      };
      const t = mk();
      if (mip) {
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.generateMipmap(gl.TEXTURE_2D);
      }
      const c = withCand ? mk() : null;
      if (c) { gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); }
      const e = withCand === 'exposure' ? mk() : null;
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      if (c) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, c, 0);
      if (e) gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, e, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { t, c, e, fb };
    }
    ensureSize(W, H) {
      if (W === this.W && H === this.H) return;
      const gl = this.gl;
      const aspectChanged = !this.W || Math.abs(W / H - this.W / this.H) > 0.01;
      this.W = W; this.H = H;
      this.canvas.width = W; this.canvas.height = H;
      this.gw = Math.min(640, W);
      this.gh = Math.max(1, Math.round(this.gw * H / W));
      if (aspectChanged) {
        for (const x of [this.plateA, this.plateB, this.pp]) if (x) { gl.deleteTexture(x.t); if (x.c) gl.deleteTexture(x.c); if (x.e) gl.deleteTexture(x.e); gl.deleteFramebuffer(x.fb); }
        this.PW = 640;
        this.PH = Math.max(2, Math.round(640 * H / W / 2) * 2);
        this.plateA = this.rgbaTarget(this.PW, this.PH, false, 'exposure');
        this.plateB = this.rgbaTarget(this.PW, this.PH, false, 'exposure');
        this.pp = this.rgbaTarget(this.PW, this.PH, true);
        if (this.stats) { gl.deleteTexture(this.stats.t); gl.deleteTexture(this.stats.c); gl.deleteFramebuffer(this.stats.fb); }
        this.SW = 48;
        this.SH = Math.max(2, Math.round(48 * H / W));
        this.stats = this.rgbaTarget(this.SW, this.SH, false, true);
        this.resetExposure();
        this.uploadR8(this.texShadow, new Uint8Array(1), 1, 1);
        this.plateFrames = 0;
        this.fillReady = false; this.fillMix = 0;
        this.resetNext = true;
      }
    }
    uploadFrame(frame) {
      const gl = this.gl;
      this.ensureSize(frame.displayWidth, frame.displayHeight);
      gl.bindTexture(gl.TEXTURE_2D, this.texLive);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frame);
    }
    uploadMasks(mask) {
      if (!mask || mask.ver === this.maskVer) return;
      this.maskVer = mask.ver;
      this.uploadR8(this.texMask, mask.u8, mask.w, mask.h);
      this.uploadR8(this.texMaskD, mask.dil, mask.w, mask.h);
      this.uploadR8(this.texMaskP, mask.dilP, mask.w, mask.h);
    }
    snapshot(frame, mask) {
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.texSnap);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frame);
      if (mask) {
        this.uploadR8(this.texSnapMask, mask.u8, mask.w, mask.h);
        // bounding box of the person
        let x0 = 1, y0 = 1, x1 = 0, y1 = 0, any = false;
        const { u8, w, h } = mask;
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
          if (u8[y * w + x] > 127) {
            any = true;
            const u = x / w, v = y / h;
            if (u < x0) x0 = u; if (u > x1) x1 = u; if (v < y0) y0 = v; if (v > y1) y1 = v;
          }
        }
        this.box = any ? [x0, y0, x1 + 1 / w, y1 + 1 / h] : [0.3, 0.2, 0.7, 1.0];
      } else {
        this.uploadR8(this.texSnapMask, new Uint8Array(1), 1, 1);
      }
    }
    updatePlate(rate) {
      const gl = this.gl;
      gl.disable(gl.BLEND);
      gl.bindVertexArray(this.quad);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.plateB.fb);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
      gl.viewport(0, 0, this.PW, this.PH);
      this.use(this.progPlate, { uFlipY: 0, uRate: rate, uReset: this.resetNext ? 1 : 0, uE: this.E },
        { uPrev: this.plateA.t, uCand: this.plateA.c, uPrevE: this.plateA.e, uLive: this.texLive, uMaskP: this.texMaskP });
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      this.resetNext = false;
      [this.plateA, this.plateB] = [this.plateB, this.plateA];
      // premultiplied + mipmapped copy for the quick hole fill
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.pp.fb);
      this.use(this.progPremul, { uFlipY: 0, uENow: this.E }, { uPlate: this.plateA.t, uPlateE: this.plateA.e });
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindTexture(gl.TEXTURE_2D, this.pp.t);
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.plateFrames++;
      this.plateDirty = true;
    }
    resetExposure() {
      this.E = 1; this.base = null; this.fillE = 1; this.pendingFillE = null; this.ecells = 0;
    }
    // Low-res averages of the live frame over visible room (and of the plate, unused here).
    readStats() {
      if (!this.stats || !this.plateFrames) return null;
      const gl = this.gl, sw = this.SW, sh = this.SH, n = sw * sh;
      gl.disable(gl.BLEND);
      gl.bindVertexArray(this.quad);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.stats.fb);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
      gl.viewport(0, 0, sw, sh);
      this.use(this.progStats, { uFlipY: 0, uCell: [1 / sw, 1 / sh] }, { uLive: this.texLive, uPlate: this.plateA.t, uMaskP: this.texMaskP });
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      const live = new Uint8Array(n * 4);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(0, 0, sw, sh, gl.RGBA, gl.UNSIGNED_BYTE, live);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return live;
    }
    // The plate is going to the AI now: its fill will be in this exposure.
    captureRef() { this.pendingFillE = this.E; }
    // Room exposure index: median change of the visible room against the first good look at it.
    updateExposure() {
      const live = this.readStats();
      if (!live) return;
      const n = this.SW * this.SH;
      const lum = (i) => (0.299 * live[i * 4] + 0.587 * live[i * 4 + 1] + 0.114 * live[i * 4 + 2]) / Math.max(live[i * 4 + 3], 1);
      if (!this.base) {
        let good = 0;
        for (let i = 0; i < n; i++) if (live[i * 4 + 3] > 77) good++;
        if (good < n * 0.2) return;
        this.base = new Float32Array(n);
        for (let i = 0; i < n; i++) this.base[i] = live[i * 4 + 3] > 77 ? lum(i) : -1;
        this.E = 1;
        return;
      }
      const r = [];
      for (let i = 0; i < n; i++) {
        if (live[i * 4 + 3] > 77 && this.base[i] > 0.02) r.push(lum(i) / this.base[i]);
        else if (live[i * 4 + 3] > 77 && this.base[i] < 0) this.base[i] = lum(i) / this.E; // newly seen cell
      }
      this.ecells = r.length;
      if (r.length < 12) return;
      r.sort((a, b) => a - b);
      const med = clamp(r[r.length >> 1], 0.4, 2.5);
      this.E = this.E * 0.6 + med * 0.4;
    }
    resetPlate() {
      this.resetExposure();
      this.uploadR8(this.texShadow, new Uint8Array(1), 1, 1);
      this.resetNext = true;
      this.plateFrames = 0;
      this.fillReady = false;
      this.fillMix = 0;
    }
    readPlate() {
      const gl = this.gl;
      const buf = new Uint8Array(this.PW * this.PH * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.plateA.fb);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(0, 0, this.PW, this.PH, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      let known = 0;
      for (let i = 3; i < buf.length; i += 4) if (buf[i] > 127) known++;
      this.plateDirty = false;
      return { rgba: buf, w: this.PW, h: this.PH, coverage: known / (this.PW * this.PH) };
    }
    setFill(rgba, w, h, shadow, instant) {
      if (w !== this.PW || h !== this.PH) return false;
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.texFill);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
      if (shadow && shadow.length === w * h) this.uploadR8(this.texShadow, shadow, w, h);
      else this.uploadR8(this.texShadow, new Uint8Array(1), 1, 1);
      this.fillReady = true;
      if (instant) this.fillMix = 1; // nobody is watching the fill yet: no need to fade it in
      this.fillE = this.pendingFillE != null ? this.pendingFillE : this.E; // the light the fill was made in
      this.pendingFillE = null;
      return true;
    }
    render(p, timestamp) {
      const gl = this.gl;
      if (this.fillReady) this.fillMix = Math.min(1, this.fillMix + 0.06); else this.fillMix = 0;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.W, this.H);
      gl.disable(gl.BLEND);
      gl.bindVertexArray(this.quad);
      const common = { uBox: this.box, uDir: p.dir, uMode: p.mode, uAspect: this.W / this.H, uEdge: p.edge };
      this.use(this.progComp, {
        ...common, uFlipY: 1, uFillMix: this.fillMix, uBgMix: p.bgMix, uFront: p.bodyFront, uShadowOn: p.shadow ? 1 : 0,
        uENow: this.E, uFillE: this.fillE,
      }, {
        uLive: this.texLive, uMask: this.texMask, uMaskD: this.texMaskD, uShadow: this.texShadow,
        uPlateE: this.plateA ? this.plateA.e : this.texShadow,
        uPlate: this.plateA ? this.plateA.t : this.texFill, uFill: this.texFill, uPP: this.pp ? this.pp.t : this.texFill,
      });
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      if (p.dust) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.bindVertexArray(this.empty);
        this.use(this.progDust, {
          ...common, uGrid: Object.assign([this.gw, this.gh], { int: true }), uRes: [this.W, this.H],
          uTau: p.tau, uSpread: p.spread, uPt: (this.W / this.gw) * 1.35,
        }, { uSnap: this.texSnap, uSnapMask: this.texSnapMask });
        gl.drawArrays(gl.POINTS, 0, this.gw * this.gh);
        gl.disable(gl.BLEND);
      }
      gl.bindVertexArray(null);
      return new VideoFrame(this.canvas, { timestamp, alpha: 'discard' });
    }
  }

  // ───────────────────────────── microphone control ─────────────────────────────
  function findMicButtons() {
    const out = [];
    const els = document.querySelectorAll('button,[role="button"],[role="switch"]');
    for (const el of els) {
      const label = (el.getAttribute('aria-label') || el.getAttribute('data-tooltip') || el.getAttribute('title') || '').trim();
      if (!label || label.length > 90) continue;
      if (/camera|video|everyone|all participants|speaker|request|ask to|noise|setting|option|more|share/i.test(label)) continue;
      if (!/mic|microphone|mute/i.test(label)) continue;
      let state = null;
      const dm = el.getAttribute('data-is-muted');
      if (dm === 'true') state = 'muted';
      else if (dm === 'false') state = 'unmuted';
      if (!state) {
        if (/\bunmute\b|turn on (the |your )?mic|mic(rophone)? (is )?off/i.test(label)) state = 'muted';
        else if (/\bmute\b|turn off (the |your )?mic/i.test(label)) state = 'unmuted';
      }
      if (!state) continue;
      const r = el.getBoundingClientRect();
      if (r.width < 4 || r.height < 4) continue;
      out.push({ el, state, label });
    }
    return out;
  }

  function appMicState() {
    const b = findMicButtons();
    return b.length ? b[0].state : null;
  }

  function pressAppMic(wantMuted) {
    const b = findMicButtons().find((x) => x.state === (wantMuted ? 'unmuted' : 'muted'));
    if (!b) return false;
    try { b.el.click(); return true; } catch (_) { return false; }
  }

  class AudioController {
    constructor() {
      this.tracks = new Set();
      this.weMuted = new Set();
      this.clickedApp = false;
      this.analyser = null;
    }
    register(track, isCopy) {
      if (this.tracks.has(track)) return;
      this.tracks.add(track);
      const drop = () => {
        this.tracks.delete(track);
        if (this.analyser && this.analyser.src === track) {
          if (core.state === 'visible') this.stopAnalyser();
          else this.analyser.orphan = true; // keep listening for the "come back" snap
        }
      };
      track.addEventListener('ended', drop);
      const origStop = track.stop.bind(track);
      const origClone = track.clone.bind(track);
      Object.defineProperty(track, 'stop', { value: () => { origStop(); drop(); }, configurable: true, writable: true });
      // Copies of the mic track must mute with it.
      Object.defineProperty(track, 'clone', { value: () => { const c = origClone(); this.register(c, true); return c; }, configurable: true, writable: true });
      if (core.state !== 'visible' && track.enabled) { track.enabled = false; this.weMuted.add(track); }
      if (!isCopy) this.startAnalyser(track);
    }
    startAnalyser(track) {
      this.stopAnalyser();
      let clone;
      try { clone = MediaStreamTrack.prototype.clone.call(track); clone.enabled = true; } catch (_) { return; }
      const a = { src: track, clone, alive: true, orphan: false };
      this.analyser = a;
      const det = core.click;
      if (typeof MediaStreamTrackProcessor === 'function') {
        const reader = new MediaStreamTrackProcessor({ track: clone }).readable.getReader();
        a.reader = reader;
        let buf = new Float32Array(0);
        (async () => {
          try {
            for (;;) {
              const { value, done } = await reader.read();
              if (done || !a.alive) { if (value) value.close(); break; }
              try {
                const n = value.numberOfFrames;
                if (buf.length < n) buf = new Float32Array(n);
                value.copyTo(buf.subarray(0, n), { planeIndex: 0, format: 'f32-planar' });
                det.available = true;
                det.process(buf.subarray(0, n), value.sampleRate, now());
              } catch (e) { warnOnce('audio', 'audio frame error', e); }
              value.close();
            }
          } catch (_) { /* track ended */ }
        })();
      } else {
        // Fallback: Web Audio. Needs the page to have had a click/keypress before it can start.
        try {
          const ctx = new AudioContext();
          const src = ctx.createMediaStreamSource(new MediaStream([clone]));
          const sp = ctx.createScriptProcessor(1024, 1, 1);
          sp.onaudioprocess = (e) => { det.available = ctx.state === 'running'; det.process(e.inputBuffer.getChannelData(0), ctx.sampleRate, now()); };
          src.connect(sp); sp.connect(ctx.destination);
          const resume = () => ctx.resume().catch(() => {});
          resume();
          addEventListener('pointerdown', resume, true);
          addEventListener('keydown', resume, true);
          a.ctx = ctx;
        } catch (e) { warnOnce('audio2', 'audio analysis unavailable', e); }
      }
    }
    stopAnalyser() {
      const a = this.analyser;
      if (!a) return;
      a.alive = false;
      try { a.reader && a.reader.cancel(); } catch (_) {}
      try { a.ctx && a.ctx.close(); } catch (_) {}
      try { a.clone.stop(); } catch (_) {}
      this.analyser = null;
      core.click.available = false;
    }
    mute() {
      for (const t of this.tracks) {
        if (t.readyState === 'live' && t.enabled) { t.enabled = false; this.weMuted.add(t); }
      }
      this.clickedApp = settings.syncMuteButton ? pressAppMic(true) : false;
    }
    unmute() {
      const clicked = this.clickedApp;
      this.clickedApp = false;
      if (clicked) pressAppMic(false);
      const restore = () => {
        const st = clicked ? appMicState() : null;
        if (st === 'muted') { this.weMuted.clear(); return; } // the app says muted — don't fight it
        for (const t of this.weMuted) if (t.readyState === 'live') t.enabled = true;
        this.weMuted.clear();
      };
      if (clicked) setTimeout(restore, 250); else restore();
    }
    settle() {
      if (this.analyser && this.analyser.orphan) this.stopAnalyser();
    }
  }

  // ───────────────────────────── the effect core ─────────────────────────────
  const DIRS = { right: [1, 0], left: [-1, 0], up: [0, -1], scatter: [1, 0] };
  const SPREAD = 0.58;
  const EDGE = 0.06;

  const core = {
    state: 'visible',     // visible | vanishing | vanished | reappearing | settling
    tau: 0,
    animStart: 0,
    animDur: 0,
    settleStart: 0,
    cooldownUntil: 0,
    pendingSnapshot: false,
    pipelines: new Set(),
    primary: null,
    engine: null,
    renderer: null,
    rendererFailed: false,
    detector: new SnapDetector(),
    click: new ClickDetector(),
    audio: null,
    mask: null,
    mlSent: new Map(),
    lastSend: 0,
    lastSeg: 0,
    lastPlate: 0,
    lastMask: 0,
    mlTimes: [],
    pending: null,
    lastInpaint: -1e9,
    inpaintBusy: false,
    coverage: 0,
    lastGesture: null,
    lastDebug: 0,

    ensureEngine() {
      if (!this.engine && base) this.engine = new Engine(base);
      return this.engine;
    },
    getRenderer() {
      if (this.renderer && !this.renderer.lost) return this.renderer;
      if (this.rendererFailed) return null;
      try { this.renderer = new Renderer(); } catch (e) { this.rendererFailed = true; console.warn(TAG, e); return null; }
      return this.renderer;
    },
    addPipeline(p) { this.pipelines.add(p); this.primary = p; this.emitState(); },
    removePipeline(p) {
      this.pipelines.delete(p);
      if (this.primary === p) this.primary = [...this.pipelines].pop() || null;
      if (!this.pipelines.size && this.state !== 'visible') this.forceVisible();
      this.emitState();
    },

    toggle(source) {
      const t = now();
      if (!settings.enabled) return;
      if (!this.pipelines.size && source !== 'manual') return;
      if (source !== 'manual' && t < this.cooldownUntil) return;
      if (this.state === 'visible') this.startVanish();
      else if (this.state === 'vanished') this.startReappear();
      this.pending = null;
    },
    startVanish() {
      this.state = 'vanishing';
      this.animStart = now();
      this.animDur = settings.duration * 1000;
      this.tau = 0;
      this.pendingSnapshot = true;
      this.audio.mute();
      this.requestInpaint(true);
      this.emitState();
    },
    startReappear() {
      this.state = 'reappearing';
      this.animStart = now();
      this.animDur = settings.duration * 1000 * 0.85;
      this.tau = 1;
      this.pendingSnapshot = true;
      this.emitState();
    },
    forceVisible() {
      const was = this.state;
      this.state = 'visible';
      this.tau = 0;
      if (was !== 'visible') { this.audio.unmute(); this.audio.settle(); }
      this.emitState();
    },
    tick(t) {
      if (this.state === 'vanishing') {
        this.tau = clamp((t - this.animStart) / this.animDur, 0, 1);
        if (this.tau >= 1) { this.state = 'vanished'; this.cooldownUntil = t + 700; this.emitState(); }
      } else if (this.state === 'reappearing') {
        this.tau = 1 - clamp((t - this.animStart) / this.animDur, 0, 1);
        if (this.tau <= 0) {
          this.state = 'settling'; this.settleStart = t; this.cooldownUntil = t + 900;
          this.audio.unmute();
          this.emitState();
        }
      } else if (this.state === 'settling') {
        if (t - this.settleStart > 350) { this.state = 'visible'; this.audio.settle(); this.emitState(); }
      }
      if (this.pending && t > this.pending.until) {
        this.debugNote('Saw the snap gesture but heard no click');
        this.pending = null;
      }
    },

    // ── per-frame work, called for every camera frame ──
    processFrame(frame, pipe) {
      const t = now();
      if (!settings.enabled) {
        if (this.state !== 'visible') this.forceVisible();
        return frame;
      }
      this.ensureEngine();
      const primary = pipe === this.primary;
      if (primary) this.maybeSendML(frame, t);
      this.tick(t);
      const rendering = this.state !== 'visible';
      const maskFresh = this.mask && t - this.lastMask < 400;
      const wantPlate = primary && maskFresh && t - this.lastPlate > (rendering ? 120 : 220);
      if (!rendering && !wantPlate && !this.pendingSnapshot) {
        if (primary) this.maybeInpaint(t);
        return frame;
      }
      const r = this.getRenderer();
      if (!r) return frame;
      r.uploadFrame(frame);
      r.uploadMasks(this.mask);
      if (wantPlate) { r.updateExposure(); r.updatePlate(rendering ? 0.12 : 0.22); this.lastPlate = t; }
      if (this.pendingSnapshot) {
        r.snapshot(frame, this.mask);
        this.pendingSnapshot = false;
        this.animStart = t;
      }
      if (primary) this.maybeInpaint(t);
      if (!rendering) return frame;
      const dir = DIRS[settings.direction] || DIRS.right;
      let bgMix = 1;
      if (this.state === 'vanishing') bgMix = clamp(this.tau / 0.08, 0, 1);
      else if (this.state === 'settling') bgMix = 1 - clamp((t - this.settleStart) / 350, 0, 1);
      const tau = this.state === 'settling' ? 0 : this.tau;
      // Dissolve front in delay units. A pixel crumbles when the front passes it; its dust particle is
      // born at the same moment (see VS_DUST). Coming back, the body re-forms a little ahead of the
      // incoming dust so the grains land on you instead of leaving gaps.
      const front = (tau / SPREAD) * (1 + EDGE) - EDGE * 0.5;
      const bodyFront = this.state === 'reappearing' || this.state === 'settling' ? front - EDGE * 1.5 : front;
      const out = r.render({
        dir, mode: settings.direction === 'scatter' ? 1 : 0, edge: EDGE, spread: SPREAD,
        tau, bodyFront, bgMix, shadow: !!settings.clearShadow,
        dust: this.state === 'vanishing' || this.state === 'reappearing',
      }, frame.timestamp);
      frame.close();
      return out;
    },

    maybeSendML(frame, t) {
      const eng = this.engine;
      if (!eng || eng.status !== 'ready') return;
      if (eng.inflight && t - eng.inflightSince < 1500) return;
      if (t - this.lastSend < 30) return;
      const wantSeg = this.state !== 'visible' || t - this.lastSeg > 110;
      eng.inflight = true; eng.inflightSince = t;
      this.lastSend = t;
      if (wantSeg) this.lastSeg = t;
      const w = 640;
      const h = Math.max(2, Math.round(w * frame.displayHeight / frame.displayWidth / 2) * 2);
      let f2;
      try { f2 = frame.clone(); } catch (_) { eng.inflight = false; return; }
      createImageBitmap(f2, { resizeWidth: w, resizeHeight: h, resizeQuality: 'medium' })
        .then((bmp) => { f2.close(); eng.sendFrame(bmp, true, wantSeg, t); })
        .catch((e) => { f2.close(); eng.inflight = false; warnOnce('bmp', e); });
    },

    onResult(m) {
      const sent = this.mlSent.get(m.id);
      this.mlSent.delete(m.id);
      if (this.mlSent.size > 50) this.mlSent.clear();
      const t = sent ? sent.t : now();
      this.mlTimes.push(now());
      if (this.mlTimes.length > 30) this.mlTimes.shift();
      if (m.ms) this.mlMs = this.mlMs ? this.mlMs * 0.8 + m.ms * 0.2 : m.ms;
      if (m.mask) this.updateMask(m.mask, m.mw, m.mh);
      if (m.hands) {
        const events = this.detector.update(t, m.hands);
        for (const ev of events) this.onGesture(ev);
      }
      this.maybeDebug();
    },

    updateMask(u8in, w, h) {
      let m = this.mask;
      const n = w * h;
      if (!m || m.w !== w || m.h !== h) {
        m = this.mask = { w, h, soft: new Float32Array(n), u8: new Uint8Array(n), dil: new Uint8Array(n), dilP: new Uint8Array(n), nd: new Uint8Array(n), ndP: new Uint8Array(n), tmp: new Uint8Array(n), ver: 0 };
        for (let i = 0; i < n; i++) m.soft[i] = u8in[i] / 255;
      }
      const s = m.soft, u8 = m.u8;
      for (let i = 0; i < n; i++) {
        s[i] = s[i] * 0.35 + (u8in[i] / 255) * 0.65;
        u8[i] = s[i] * 255 + 0.5;
      }
      // Dilate, then hold the previous masks for a moment (decaying max) so a quick move — or the ML
      // falling a frame or two behind — never lets a sliver of you show through while you're "gone".
      dilate(u8, m.nd, m.tmp, w, h, Math.max(2, Math.round(w / 85)));
      dilate(u8, m.ndP, m.tmp, w, h, Math.max(4, Math.round(w / 36)));
      const hold = this.state === 'visible' ? 0.7 : 0.86;
      for (let i = 0; i < n; i++) {
        const a = m.dil[i] * hold, b = m.dilP[i] * hold;
        m.dil[i] = m.nd[i] > a ? m.nd[i] : a;
        m.dilP[i] = m.ndP[i] > b ? m.ndP[i] : b;
      }
      m.ver++;
      this.lastMask = now();
    },

    onGesture(ev) {
      const t = now();
      this.lastGesture = { score: ev.score, at: t };
      if (t < this.cooldownUntil || (this.state !== 'visible' && this.state !== 'vanished')) return;
      const audioOk = settings.requireSound && this.click.available;
      if (audioOk) {
        if (this.click.hasClickIn(ev.tArm - 150, ev.t + 120)) { this.debugNote('Snap!'); this.toggle('gesture'); }
        else this.pending = { ev, until: t + 380 };
      } else if (ev.score >= this.detector.scoreThr + 0.12) {
        this.debugNote('Snap! (gesture only)');
        this.toggle('gesture');
      } else this.debugNote('Gesture too weak without the click sound');
    },

    onClick(tc) {
      const p = this.pending;
      if (p && tc >= p.ev.tArm - 150 && tc <= p.ev.t + 420) {
        this.pending = null;
        this.debugNote('Snap!');
        this.toggle('gesture');
      }
    },

    // ── background learning + AI fill ──
    maybeInpaint(t) {
      const r = this.renderer, eng = this.engine;
      if (!r || !eng || eng.status !== 'ready' || this.inpaintBusy || !r.plateFrames) return;
      const urgent = this.inpaintUrgent;
      if (!urgent) {
        if (this.state !== 'visible') return;
        if (!r.plateDirty || r.plateFrames < 6) return;
        if (t - this.lastInpaint < (r.fillReady ? 9000 : 2500)) return;
      }
      this.inpaintUrgent = false;
      r.captureRef();
      const plate = r.readPlate();
      this.coverage = plate.coverage;
      if (plate.coverage <= 0.02) return;
      this.inpaintBusy = true;
      this.lastInpaint = t;
      const inputCopy = debugMode ? plate.rgba.slice() : null;
      const band = settings.clearShadow ? this.shadowBand() : null;
      const ok = eng.inpaint(plate.rgba, plate.w, plate.h, band, (res) => {
        this.inpaintBusy = false;
        this.shadowInfo = res.shadowInfo;
        if (debugMode && res.rgba) this.lastInpaintDbg = { w: res.w, h: res.h, input: inputCopy, output: res.rgba.slice() };
        if (res.error) { warnOnce('inp', 'inpainting failed', res.error); return; }
        if (this.renderer) this.renderer.setFill(res.rgba, res.w, res.h, res.shadow, this.state === 'visible');
        if (this.inpaintUrgent) this.lastInpaint = -1e9;
      });
      if (!ok) this.inpaintBusy = false;
    },
    // A ring around you where your shadow is likely to fall (outside the person mask, inside a wide margin).
    shadowBand() {
      const m = this.mask;
      if (!m) return null;
      const n = m.w * m.h, big = new Uint8Array(n), tmp = new Uint8Array(n), band = new Uint8Array(n);
      dilate(m.u8, big, tmp, m.w, m.h, Math.max(8, Math.round(m.w / 9)));
      for (let i = 0; i < n; i++) band[i] = big[i] > 64 && m.dil[i] < 64 ? 255 : 0;
      return { data: band, w: m.w, h: m.h };
    },
    requestInpaint(urgent) {
      if (urgent) {
        const r = this.renderer;
        if (r && r.fillReady && now() - this.lastInpaint < 15000) return;
        this.inpaintUrgent = true;
      }
    },
    resetRoom() {
      if (this.renderer) this.renderer.resetPlate();
      this.coverage = 0;
      this.lastInpaint = -1e9;
    },

    // ── reporting ──
    status() {
      const eng = this.engine;
      const mt = this.mlTimes;
      const fps = mt.length > 2 ? (mt.length - 1) / ((mt[mt.length - 1] - mt[0]) / 1000) : 0;
      return {
        enabled: settings.enabled,
        camera: this.pipelines.size > 0,
        state: this.state,
        engine: eng ? eng.status : 'idle',
        engineError: eng && eng.error,
        roomLearned: Math.round(this.coverage * 100),
        aiFill: this.renderer && this.renderer.fillReady ? 'ready' : (eng ? eng.inpaintStatus : 'idle'),
        shadow: this.shadowInfo || 'off',
        light: this.renderer ? `E=${this.renderer.E.toFixed(3)} fillE=${(this.renderer.fillE || 1).toFixed(3)} n=${this.renderer.ecells}` : 'none',
        roomReady: !!(this.renderer && this.renderer.fillReady && this.coverage >= 0.8),
        roomAlmost: this.coverage >= 0.5,
        inpaintEp: eng && eng.inpaintEp,
        sound: this.click.available,
        mlFps: Math.round(fps),
        mlMs: Math.round(this.mlMs || 0),
        host: location.hostname,
      };
    },
    emitState() { toIso({ type: 'state', state: this.state, camera: this.pipelines.size > 0 }); },
    debugNote(text) { if (debugMode) toIso({ type: 'debug-note', text, t: Date.now() }); },
    maybeDebug() {
      if (!debugMode) return;
      const t = now();
      if (t - this.lastDebug < 90) return;
      this.lastDebug = t;
      toIso({ type: 'debug', hand: this.detector.last && t - this.detector.last.t < 400 ? this.detector.last : null, level: this.click.level, sound: this.click.available, status: this.status() });
    },
  };
  core.audio = new AudioController();
  core.click.onClick = (t) => { core.onClick(t); if (debugMode) toIso({ type: 'debug-click', t: Date.now() }); };

  // separable max filter (dilation) on a w×h byte image
  function dilate(src, dst, tmp, w, h, r) {
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        let m = 0;
        const a = x - r < 0 ? 0 : x - r, b = x + r >= w ? w - 1 : x + r;
        for (let i = a; i <= b; i++) { const v = src[row + i]; if (v > m) m = v; }
        tmp[row + x] = m;
      }
    }
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        let m = 0;
        const a = y - r < 0 ? 0 : y - r, b = y + r >= h ? h - 1 : y + r;
        for (let i = a; i <= b; i++) { const v = tmp[i * w + x]; if (v > m) m = v; }
        dst[y * w + x] = m;
      }
    }
  }

  // ───────────────────────────── camera track wrapping ─────────────────────────────
  function proxyTrack(out, raw, pipe) {
    const def = (k, d) => { try { Object.defineProperty(out, k, { configurable: true, ...d }); } catch (_) {} };
    def('label', { get: () => raw.label });
    def('getSettings', { value: () => ({ ...raw.getSettings() }) });
    def('getCapabilities', { value: () => (raw.getCapabilities ? raw.getCapabilities() : {}) });
    def('getConstraints', { value: () => raw.getConstraints() });
    def('applyConstraints', { value: (c) => raw.applyConstraints(c) });
    const origStop = out.stop.bind(out);
    const origClone = out.clone.bind(out);
    let stopped = false;
    def('stop', { value: () => { if (stopped) return; stopped = true; origStop(); pipe.release(); } });
    def('clone', { value: () => { const c = origClone(); pipe.refs++; proxyTrack(c, raw, pipe); return c; } });
  }

  class Pipeline {
    constructor(raw) {
      this.raw = raw;
      this.refs = 1;
      this.alive = true;
      this.proc = new MediaStreamTrackProcessor({ track: raw, maxBufferSize: 2 });
      this.gen = new MediaStreamTrackGenerator({ kind: 'video' });
      this.reader = this.proc.readable.getReader();
      this.writer = this.gen.writable.getWriter();
      this.out = this.gen;
      proxyTrack(this.gen, raw, this);
      raw.addEventListener('ended', () => {
        this.destroy();
        try { this.gen.dispatchEvent(new Event('ended')); } catch (_) {}
      });
      core.addPipeline(this);
      this.run();
    }
    async run() {
      try {
        for (;;) {
          const { value, done } = await this.reader.read();
          if (done) break;
          if (!this.alive) { value.close(); break; }
          let out = value;
          try { out = core.processFrame(value, this); } catch (e) { warnOnce('pf', 'frame processing error', e); out = value; }
          try { await this.writer.write(out); } catch (e) { try { out.close(); } catch (_) {} break; }
        }
      } catch (_) { /* stream ended */ }
      this.destroy();
    }
    release() { if (--this.refs <= 0) this.destroy(); }
    destroy() {
      if (!this.alive) return;
      this.alive = false;
      try { this.reader.cancel(); } catch (_) {}
      try { this.writer.close(); } catch (_) {}
      try { this.raw.stop(); } catch (_) {}
      core.removePipeline(this);
    }
  }

  // Fallback for browsers without insertable streams: <video> → VideoFrame → canvas.captureStream().
  class CanvasPipeline {
    constructor(raw) {
      this.raw = raw;
      this.refs = 1;
      this.alive = true;
      const s = raw.getSettings();
      this.video = document.createElement('video');
      this.video.muted = true; this.video.playsInline = true;
      this.video.srcObject = new MediaStream([raw]);
      this.video.play().catch(() => {});
      this.canvas = document.createElement('canvas');
      this.canvas.width = s.width || 1280; this.canvas.height = s.height || 720;
      this.ctx = this.canvas.getContext('2d');
      this.out = this.canvas.captureStream(s.frameRate || 30).getVideoTracks()[0];
      proxyTrack(this.out, raw, this);
      core.addPipeline(this);
      this.loop();
    }
    loop() {
      if (!this.alive) return;
      const v = this.video;
      if (v.readyState >= 2 && v.videoWidth) {
        try {
          const f = new VideoFrame(v, { timestamp: Math.round(performance.now() * 1000) });
          if (this.canvas.width !== f.displayWidth) { this.canvas.width = f.displayWidth; this.canvas.height = f.displayHeight; }
          const o = core.processFrame(f, this);
          this.ctx.drawImage(o, 0, 0);
          o.close();
        } catch (e) { warnOnce('cp', e); }
      }
      setTimeout(() => this.loop(), 33);
    }
    release() { if (--this.refs <= 0) this.destroy(); }
    destroy() {
      if (!this.alive) return;
      this.alive = false;
      try { this.raw.stop(); } catch (_) {}
      core.removePipeline(this);
    }
  }

  const hasBreakout = typeof window.MediaStreamTrackProcessor === 'function' && typeof window.MediaStreamTrackGenerator === 'function';

  async function wrapStream(stream) {
    await Promise.race([settingsReady, new Promise((r) => setTimeout(r, 1500))]);
    if (!settings.enabled || !base) return stream;
    const vids = stream.getVideoTracks();
    const auds = stream.getAudioTracks();
    for (const t of auds) core.audio.register(t);
    if (!vids.length) return stream;
    core.ensureEngine();
    if (core.engine) setTimeout(() => core.engine.warmInpaint(), 3000);
    const tracks = [];
    for (const t of vids) {
      let p = null;
      try { p = hasBreakout ? new Pipeline(t) : new CanvasPipeline(t); } catch (e) { console.warn(TAG, 'could not wrap camera', e); }
      tracks.push(p ? p.out : t);
    }
    return new MediaStream([...tracks, ...auds]);
  }

  const proto = MediaDevices.prototype;
  const origGUM = proto.getUserMedia;
  if (typeof origGUM === 'function') {
    const patched = {
      getUserMedia(constraints) {
        return origGUM.call(this, constraints).then((stream) =>
          wrapStream(stream).catch((e) => { console.warn(TAG, e); return stream; }));
      },
    }.getUserMedia;
    Object.defineProperty(proto, 'getUserMedia', { value: patched, configurable: true, writable: true, enumerable: true });
  }

  toIso({ type: 'hello' });
})();
