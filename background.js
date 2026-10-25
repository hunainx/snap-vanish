// Snap Vanish — service worker: default settings, keyboard shortcut, toolbar badge.
const DEFAULTS = { enabled: true, sensitivity: 5, duration: 3, direction: 'right', requireSound: true, syncMuteButton: true, clearShadow: false };

chrome.runtime.onInstalled.addListener(async () => {
  const cur = await chrome.storage.sync.get(null);
  const missing = {};
  for (const [k, v] of Object.entries(DEFAULTS)) if (!(k in cur)) missing[k] = v;
  if (Object.keys(missing).length) await chrome.storage.sync.set(missing);
});

// Which tabs currently have the camera running through Snap Vanish.
async function cameraTabs() {
  const { cams = {} } = await chrome.storage.session.get('cams');
  return cams;
}
async function setCamera(tabId, info) {
  const cams = await cameraTabs();
  if (info && info.camera) cams[tabId] = { state: info.state, at: Date.now() };
  else delete cams[tabId];
  await chrome.storage.session.set({ cams });
}

// The tab to act on: the active tab if it has the camera, otherwise the most recent tab that does.
async function targetTab() {
  const [active] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  const cams = await cameraTabs();
  if (active && cams[active.id]) return { id: active.id, other: false };
  const ids = Object.entries(cams).sort((a, b) => b[1].at - a[1].at).map(([id]) => +id);
  for (const id of ids) {
    try { await chrome.tabs.get(id); return { id, other: true }; } catch (_) { await setCamera(id, null); }
  }
  return active ? { id: active.id, other: false, noCamera: true } : null;
}

chrome.commands.onCommand.addListener(async (cmd) => {
  if (cmd !== 'snap-toggle') return;
  const t = await targetTab();
  if (t) chrome.tabs.sendMessage(t.id, { type: 'snap' }).catch(() => {});
});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === 'state' && sender.tab) {
    const tabId = sender.tab.id;
    setCamera(tabId, msg);
    const gone = msg.state === 'vanishing' || msg.state === 'vanished';
    chrome.action.setBadgeBackgroundColor({ tabId, color: '#3B2F55' }).catch(() => {});
    chrome.action.setBadgeText({ tabId, text: gone ? 'GONE' : (msg.camera ? 'ON' : '') }).catch(() => {});
    return false;
  }
  if (msg && msg.type === 'target-tab') {
    targetTab().then(sendResponse);
    return true;
  }
  return false;
});

chrome.tabs.onRemoved.addListener((tabId) => { setCamera(tabId, null); });
chrome.tabs.onUpdated.addListener((tabId, info) => { if (info.status === 'loading' && info.url) setCamera(tabId, null); });
