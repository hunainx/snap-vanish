// Snap Vanish — bridge between the extension (popup, shortcut, settings) and the page-side pipeline.
// Runs in the content-script ISOLATED world; talks to content/inject.js with DOM CustomEvents.
(() => {
  'use strict';
  const TO_MAIN = 'snapvanish:to-main';
  const TO_ISO = 'snapvanish:to-iso';
  const DEFAULTS = { enabled: true, sensitivity: 5, duration: 3, direction: 'right', requireSound: true, syncMuteButton: true, clearShadow: false };

  let settings = null;
  let camera = false;
  let state = 'visible';
  let reqId = 0;
  const waiters = new Map();
  const base = chrome.runtime.getURL('');

  const send = (msg) => document.dispatchEvent(new CustomEvent(TO_MAIN, { detail: JSON.stringify(msg) }));
  const sendConfig = () => { if (settings) send({ type: 'config', base, settings }); };
  const tell = (msg) => { try { chrome.runtime.sendMessage(msg).catch(() => {}); } catch (_) { /* extension reloaded */ } };

  chrome.storage.sync.get(DEFAULTS, (s) => { settings = { ...DEFAULTS, ...s }; sendConfig(); });
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'sync' || !settings) return;
    for (const k of Object.keys(changes)) settings[k] = changes[k].newValue;
    send({ type: 'settings', settings });
  });

  document.addEventListener(TO_ISO, (e) => {
    let msg;
    try { msg = JSON.parse(e.detail); } catch (_) { return; }
    if (!msg) return;
    if (msg.type === 'hello') sendConfig();
    else if (msg.type === 'status') {
      const w = waiters.get(msg.reqId);
      if (w) { waiters.delete(msg.reqId); w(msg.status); }
    } else if (msg.type === 'state') {
      const changed = msg.camera !== camera || msg.state !== state;
      camera = msg.camera; state = msg.state;
      if (changed) tell({ type: 'state', state, camera });
    }
  });

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg || !msg.type) return false;
    if (msg.type === 'snap') { if (camera) send({ type: 'snap' }); return false; }
    if (msg.type === 'reset-room') { if (camera) send({ type: 'reset-room' }); return false; }
    if (msg.type === 'get-status') {
      if (camera) {
        const id = ++reqId;
        waiters.set(id, (status) => sendResponse({ camera: true, ...status }));
        send({ type: 'get-status', reqId: id });
        setTimeout(() => { if (waiters.has(id)) { waiters.delete(id); sendResponse({ camera: true, state, timeout: true }); } }, 900);
        return true;
      }
      if (window === window.top) {
        // Let a frame that actually has the camera answer first.
        setTimeout(() => sendResponse({ camera: false, supported: true, host: location.hostname }), 300);
        return true;
      }
    }
    return false;
  });
})();
