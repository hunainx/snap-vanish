// Practice page: acts as the bridge for content/inject.js (loaded directly on this extension page).
(() => {
  const TO_MAIN = 'snapvanish:to-main';
  const TO_ISO = 'snapvanish:to-iso';
  const DEFAULTS = { enabled: true, sensitivity: 5, duration: 3, direction: 'right', requireSound: true, syncMuteButton: true, clearShadow: false };
  const $ = (id) => document.getElementById(id);
  const send = (msg) => document.dispatchEvent(new CustomEvent(TO_MAIN, { detail: JSON.stringify(msg) }));

  let settings = { ...DEFAULTS };
  let stream = null;
  let state = 'visible';
  let reqId = 0;
  const sendConfig = () => send({ type: 'config', base: chrome.runtime.getURL(''), settings, debug: true, engineQuery: location.search });

  chrome.storage.sync.get(DEFAULTS, (s) => { settings = { ...DEFAULTS, ...s }; sendConfig(); });
  chrome.storage.onChanged.addListener((ch, area) => {
    if (area !== 'sync') return;
    for (const k of Object.keys(ch)) settings[k] = ch[k].newValue;
    send({ type: 'settings', settings });
  });

  const log = (text) => {
    const li = document.createElement('li');
    li.textContent = text;
    $('log').prepend(li);
    while ($('log').children.length > 5) $('log').lastChild.remove();
  };
  const flash = (el, cls, ms) => { el.classList.add(cls); clearTimeout(el._t); el._t = setTimeout(() => el.classList.remove(cls), ms); };

  const STATE_TEXT = { vanishing: 'Turning to dust…', vanished: 'Gone. Mic muted.', reappearing: 'Coming back…' };

  document.addEventListener(TO_ISO, (e) => {
    const msg = JSON.parse(e.detail);
    if (msg.type === 'hello') sendConfig();
    else if (msg.type === 'state') {
      state = msg.state;
      const b = $('stateBadge');
      b.hidden = !STATE_TEXT[state];
      b.textContent = STATE_TEXT[state] || '';
      $('snap').textContent = state === 'vanished' ? 'Bring me back' : 'Snap now';
    } else if (msg.type === 'debug') {
      const h = msg.hand;
      $('dSet').classList.toggle('on', !!h && (h.stage === 'set' || h.stage === 'setting'));
      if (h && (h.stage === 'snap' || h.stage === 'weak')) flash($('dSnap'), h.stage === 'snap' ? 'on' : 'wait', 900);
      $('level').style.width = Math.min(100, Math.sqrt(msg.level || 0) * 260) + '%';
      $('clickHint').textContent = msg.sound ? 'The sharp sound of the snap' : 'Mic not available, using the gesture alone';
      renderStatus(msg.status);
    } else if (msg.type === 'debug-click') {
      flash($('dClick'), 'on', 700);
    } else if (msg.type === 'debug-note') {
      log(msg.text);
    } else if (msg.type === 'status') {
      renderStatus(msg.status);
    }
  });

  function renderStatus(s) {
    if (!s) return;
    const parts = [];
    if (!s.enabled) parts.push('Snap Vanish is switched off in the toolbar popup');
    else if (s.engine === 'loading') parts.push('Loading the on-device models');
    else if (s.engine === 'error') parts.push('Models failed to load: ' + (s.engineError || 'unknown error'));
    else {
      parts.push(s.roomReady ? 'Ready to snap' : s.roomAlmost ? 'Almost ready: lean left, then right, so it can see behind you' : 'Learning your room: sit still, then lean left and right once');
      parts.push(`Room learned: ${s.roomLearned}%`);
      parts.push(s.aiFill === 'ready' ? 'AI fill ready' : s.aiFill === 'working' ? 'AI fill working' : s.aiFill === 'error' ? 'AI fill unavailable, using a simpler fill' : 'AI fill warming up');
    }
    $('status').textContent = parts.join('. ');
  }

  $('start').addEventListener('click', async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: true });
      $('out').srcObject = new MediaStream(stream.getVideoTracks());
      $('empty').hidden = true;
      $('snap').disabled = false;
      $('reset').disabled = false;
      log('Camera on. Sit still for a moment so the room can be learned.');
    } catch (e) {
      log('Camera blocked: ' + e.message);
    }
  });
  $('snap').addEventListener('click', () => send({ type: 'snap' }));
  $('reset').addEventListener('click', () => { send({ type: 'reset-room' }); log('Learning the room again.'); });

  setInterval(() => send({ type: 'get-status', reqId: ++reqId }), 1000);
})();
