const DEFAULTS = { enabled: true, sensitivity: 5, duration: 3, direction: 'right', requireSound: true, syncMuteButton: true, clearShadow: false };
const $ = (id) => document.getElementById(id);
let settings = { ...DEFAULTS };
let target = null;
let lastStatus = null;

function save(patch) {
  Object.assign(settings, patch);
  chrome.storage.sync.set(patch);
  render();
}

function render() {
  const sw = $('enabled');
  sw.setAttribute('aria-checked', String(!!settings.enabled));
  document.body.classList.toggle('off', !settings.enabled);
  $('sensitivity').value = settings.sensitivity;
  $('sensOut').textContent = settings.sensitivity;
  $('duration').value = settings.duration;
  $('durOut').textContent = Number(settings.duration).toFixed(1) + ' s';
  $('requireSound').checked = !!settings.requireSound;
  $('syncMuteButton').checked = !!settings.syncMuteButton;
  $('clearShadow').checked = !!settings.clearShadow;
  for (const r of document.querySelectorAll('input[name="dir"]')) r.checked = r.value === settings.direction;
}

function setStatus(kind, headline, detail) {
  const box = document.querySelector('.status');
  box.className = 'status ' + (kind || '');
  $('headline').textContent = headline;
  $('detail').textContent = detail || '';
}

function renderStatus(s) {
  lastStatus = s;
  const snap = $('snap');
  snap.textContent = s && s.state === 'vanished' ? 'Bring me back' : 'Snap now';
  const animating = s && (s.state === 'vanishing' || s.state === 'reappearing' || s.state === 'settling');
  snap.disabled = !settings.enabled || !s || !s.camera || animating;
  $('relearn').disabled = !s || !s.camera;
  const where = target && target.other ? ` (meeting in another tab)` : '';

  if (!settings.enabled) return setStatus('', 'Switched off', 'Turn it on before you join. Already in a call? Turn your camera off and on once after switching on.');
  if (!s) return setStatus('wait', 'Open your meeting', 'Works with Google Meet, Microsoft Teams and Zoom in the browser.');
  if (!s.camera) return setStatus('wait', 'Waiting for your camera', 'Turn your camera on in the meeting. If it is already on, turn it off and on once.');
  if (s.engine === 'error') return setStatus('wait', "The on-device models didn't load", s.engineError || 'Reload the meeting tab and try again.');
  if (s.engine !== 'ready') return setStatus('wait', 'Getting ready', 'Loading the on-device models' + where);
  switch (s.state) {
    case 'vanishing': return setStatus('gone', 'Turning to dust…', 'Your mic is muted.');
    case 'vanished': return setStatus('gone', 'Gone. Your mic is muted.', 'Snap again to come back' + where + '.');
    case 'reappearing':
    case 'settling': return setStatus('gone', 'Coming back…', 'Your mic unmutes as you reappear.');
  }
  if (!s.roomReady && s.roomAlmost) return setStatus('wait', 'Almost ready' + where, `${s.roomLearned}% learned. Lean left, then right, to show what's behind you. Best at 80%.`);
  if (!s.roomReady) return setStatus('wait', 'Learning your room' + where, `${s.roomLearned}% learned. Sit still a moment, then lean left and right once.`);
  const fill = s.aiFill === 'ready' ? 'AI fill ready' : s.aiFill === 'working' ? 'AI fill working' : s.aiFill === 'error' ? 'AI fill unavailable, using a simpler fill' : 'AI fill warming up';
  const sound = s.sound ? '' : ' Mic not available, so the gesture alone is used.';
  setStatus('ready', 'Ready. Snap to vanish' + where + '.', `Room learned ${s.roomLearned}%. ${fill}.${sound}`);
}

function ask(msg) {
  return new Promise((resolve) => {
    if (!target) return resolve(null);
    chrome.tabs.sendMessage(target.id, msg, (r) => { void chrome.runtime.lastError; resolve(r || null); });
  });
}

async function poll() {
  target = await chrome.runtime.sendMessage({ type: 'target-tab' }).catch(() => null);
  const s = await ask({ type: 'get-status' });
  renderStatus(s && s.camera !== undefined ? s : null);
}

async function showKeys() {
  const cmds = await chrome.commands.getAll();
  const c = cmds.find((x) => x.name === 'snap-toggle');
  const el = $('keys');
  el.textContent = '';
  if (!c || !c.shortcut) { el.textContent = 'not set'; return; }
  c.shortcut.split('+').forEach((k, i) => {
    if (i) el.append('+');
    const kbd = document.createElement('kbd');
    kbd.textContent = k;
    el.append(kbd);
  });
}

chrome.storage.sync.get(DEFAULTS, (s) => { settings = { ...DEFAULTS, ...s }; render(); poll(); });

$('enabled').addEventListener('click', () => save({ enabled: !settings.enabled }));
let tSens, tDur;
$('sensitivity').addEventListener('input', (e) => { $('sensOut').textContent = e.target.value; clearTimeout(tSens); tSens = setTimeout(() => save({ sensitivity: +e.target.value }), 150); });
$('duration').addEventListener('input', (e) => { $('durOut').textContent = Number(e.target.value).toFixed(1) + ' s'; clearTimeout(tDur); tDur = setTimeout(() => save({ duration: +e.target.value }), 150); });
$('requireSound').addEventListener('change', (e) => save({ requireSound: e.target.checked }));
$('syncMuteButton').addEventListener('change', (e) => save({ syncMuteButton: e.target.checked }));
$('clearShadow').addEventListener('change', (e) => save({ clearShadow: e.target.checked }));
for (const r of document.querySelectorAll('input[name="dir"]')) r.addEventListener('change', () => save({ direction: r.value }));

$('snap').addEventListener('click', async () => {
  $('snap').disabled = true;
  await ask({ type: 'snap' });
  setTimeout(poll, 150);
});
$('relearn').addEventListener('click', () => { ask({ type: 'reset-room' }); setStatus('ready', 'Learning the room again', 'Sit still for a few seconds, then lean left and right once.'); });
$('practice').addEventListener('click', () => chrome.tabs.create({ url: chrome.runtime.getURL('try/try.html') }));
$('changeKeys').addEventListener('click', () => chrome.tabs.create({ url: 'chrome://extensions/shortcuts' }));

showKeys();
setInterval(poll, 800);
