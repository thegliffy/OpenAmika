/* OpenAmika — Use page: dialog socket, mic, avatar feed, telemetry. */
(function () {
  "use strict";
  const { nav, footer, toast, getJSON, postAction, el, setNavDot, fmt } = window.OA;
  nav("use"); footer();

  const $ = (id) => document.getElementById(id);

  // ── Support MekaHime popup (the same mechanics as OpenSakura's): the first ask comes 2 minutes after the page has
  // loaded, then every 30 minutes while the page is visible. Never in kiosk mode (?kiosk=1, for clean captures) and
  // never again after "never show this again" (stored per browser). Her first-run wizard is never covered: while it
  // is open the first ask waits and tries again every 30 s.
  const DONATE_FIRST_MS = 2 * 60 * 1000;
  const DONATE_EVERY_MS = 30 * 60 * 1000;
  const KIOSK = new URLSearchParams(location.search).has("kiosk");
  const donateModal = $("donate-modal");
  function donateNever() {
    try { return localStorage.getItem("openamika.donateNever") === "1"; } catch (_) { return false; }
  }
  function showDonate(why) {
    if (KIOSK || donateNever() || !donateModal || !donateModal.hidden) return;
    if (document.querySelector(".oa-wizard-backdrop")) {
      if (why === "first") setTimeout(() => showDonate(why), 30 * 1000);
      return;
    }
    donateModal.hidden = false;
    donateModal.dataset.why = why;
    $("donate-later").focus();
  }
  function hideDonate() { if (donateModal) donateModal.hidden = true; }
  if (donateModal) {
    $("donate-close").addEventListener("click", hideDonate);
    $("donate-later").addEventListener("click", hideDonate);
    $("donate-never").addEventListener("click", () => {
      try { localStorage.setItem("openamika.donateNever", "1"); } catch (_) { /* private mode */ }
      hideDonate();
      toast("okay — we won't ask again");
    });
    donateModal.addEventListener("click", (e) => { if (e.target === donateModal) hideDonate(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !donateModal.hidden) hideDonate(); });
    const armDonate = () => setTimeout(() => showDonate("first"), DONATE_FIRST_MS);
    if (document.readyState === "complete") armDonate(); else window.addEventListener("load", armDonate, { once: true });
    setInterval(() => { if (!document.hidden) showDonate("interval"); }, DONATE_EVERY_MS);
  }
  const TARGET_SR = 16000;

  // ── avatar ──
  const avatar = window.OpenAmikaAvatar;
  let avatarReady = false;

  // her "loading" emote (owner's rule, 2026-10-24): on IF AND ONLY IF the surface handed the turn to the
  // deep layer (mode say+think / think), every surface TTS chunk has finished playing (none playing, none
  // queued, none still to come) and the deep's first TTS chunk has not started. Not when the surface merely
  // finished *generating* while its audio still plays. A helper (Hermes, nanoMuse) working on something is
  // the doll's business, not hers: it shows as the agent avatar "thinking", never as her icon (2026-10-23).
  // The speech layer marks the surface's last chunk (tts_audio.surface_last) or says at ai_surface_done that
  // no surface audio is still to come (surface_tts_done), so the gap is exact — never guessed from timing.
  let deepPending = false, deepTimer = null;
  const emote = { deepExpected: false, surfaceClosed: false, surfaceAudioDone: false, deepStarted: false,
                  queuedSurface: 0, queuedOther: 0, playingKind: null };
  function syncBusy() { if (avatarReady) avatar.setBusy(deepPending); }
  function setDeepPending(on) {
    clearTimeout(deepTimer);
    deepPending = !!on;
    if (deepPending) deepTimer = setTimeout(() => setDeepPending(false), 120000);   // safety net: no deep runs this long
    syncBusy();
  }
  function refreshEmote() {
    setDeepPending(emote.deepExpected && emote.surfaceAudioDone && !emote.deepStarted
      && emote.playingKind === null && emote.queuedSurface + emote.queuedOther === 0);
  }
  function surfaceClosed() {               // no further surface audio will arrive
    emote.surfaceClosed = true;
    if (emote.queuedSurface === 0 && emote.playingKind !== "surface") emote.surfaceAudioDone = true;
    refreshEmote();
  }
  function noteQueued(meta) {              // a chunk handed to the player (it reaches onChunkStart later)
    if (meta.kind === "surface") emote.queuedSurface++; else emote.queuedOther++;
    if (meta.surfaceLast) emote.surfaceClosed = true;
  }
  function resetEmote() {                  // a new turn, a barge-in, the end of a turn: nothing pending
    Object.assign(emote, { deepExpected: false, surfaceClosed: false, surfaceAudioDone: false, deepStarted: false,
                           queuedSurface: 0, queuedOther: 0, playingKind: null });
    setDeepPending(false);
  }

  async function initAvatar() {
    const result = await avatar.init({
      canvas: $("stageCanvas"),
      area: $("stageArea"),
      onChunkStart: (meta, emotion) => {
        if (meta && meta.kind === "wizard") { if (meta.onstart) meta.onstart(); return; }   // her wizard lines: no turn bookkeeping
        const kind = (meta && meta.kind) || null;
        if (kind === "surface") emote.queuedSurface = Math.max(0, emote.queuedSurface - 1);
        else emote.queuedOther = Math.max(0, emote.queuedOther - 1);
        emote.playingKind = kind || "other";
        if (kind === "deep") emote.deepStarted = true;      // the deep answered: she is saying it
        setDeepPending(false);                              // never "loading" over her own voice
        if (meta && meta.ack && ws && ws.readyState === 1) {
          ws.send(JSON.stringify({ type: "tts_playing", turn_id: meta.turn_id, idx: meta.idx }));
        }
        if (meta && typeof meta.text === "string") {
          $("hudSay").innerHTML = "";
          if (emotion) $("hudSay").append(el("span", { class: "emo", text: emotion }));
          $("hudSay").append(document.createTextNode(meta.text));
        }
      },
      onChunkEnd: (meta) => {
        if (meta && meta.kind === "wizard") { if (meta.onend) meta.onend(); return; }
        emote.playingKind = null;
        if (meta && meta.kind === "surface" && emote.surfaceClosed && emote.queuedSurface === 0) emote.surfaceAudioDone = true;
        refreshEmote();                                     // the gap before the deep's first words starts here
        if (meta && meta.ack && ws && ws.readyState === 1) {
          ws.send(JSON.stringify({ type: "tts_finished", turn_id: meta.turn_id, idx: meta.idx }));
        }
      },
      onQueueEmpty: () => {},
    });
    avatarReady = !!result.ok;
    syncBusy();
    if (!result.ok) {
      $("stageEmpty").hidden = false;
      if (result.reason === "load-failed") {
        $("stageEmpty").querySelector(".big").textContent = "Model failed to load";
      }
      return;
    }
    const select = $("modelSelect");
    select.hidden = avatar.models.length < 2;
    $("resetViewBtn").hidden = false;
    select.replaceChildren(...avatar.models.map((m) =>
      el("option", { value: m.id, text: m.name, selected: m.id === avatar.currentId ? "" : null })));
    select.addEventListener("change", async () => {
      toast("Loading " + select.value + "…");
      const ok = await avatar.switchModel(select.value);
      toast(ok ? "Model ready." : "Model failed to load.", !ok);
    });
    $("resetViewBtn").addEventListener("click", () => avatar.resetView());
  }
  initAvatar();
  $("volSlider").addEventListener("input", () => avatar.setVolume($("volSlider").value));

  // ── stage scene: background photo + blur + fullscreen ──
  const stageArea = $("stageArea");
  function applyStage(state) {
    const bg = $("stageBg");
    bg.style.backgroundImage = state.background ? `url("${state.background}")` : "";
    bg.style.filter = state.blur ? `blur(${state.blur}px)` : "";
    $("bgBlur").value = state.blur;
    $("bgBlurVal").textContent = Math.round(state.blur) + "px";
    $("bgRemove").hidden = !state.background;
  }
  getJSON("/api/stage").then(applyStage).catch(() => {});

  async function postStage(payload) {
    const response = await fetch("/api/stage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const state = await response.json().catch(() => null);
    if (!response.ok) throw new Error((state && state.detail) || "HTTP " + response.status);
    return state;
  }

  // the background panel is a popover off the picture button: closed by default, your last choice remembered
  function showScenePanel(open) {
    $("scenePanel").hidden = !open;
    $("sceneBtn").classList.toggle("on", open);
    $("sceneBtn").setAttribute("aria-pressed", String(open));
  }
  let sceneOpen = false;
  try { sceneOpen = localStorage.getItem("openamika-scene-panel") === "open"; } catch (_) {}
  showScenePanel(sceneOpen);
  $("sceneBtn").addEventListener("click", () => {
    const open = $("scenePanel").hidden;
    showScenePanel(open);
    try { localStorage.setItem("openamika-scene-panel", open ? "open" : "closed"); } catch (_) {}
  });
  // clicks/drags inside the controls must not grab or zoom the model
  for (const box of [$("scenePanel"), document.querySelector(".stage-topbar")]) {
    for (const evt of ["mousedown", "touchstart", "wheel"]) {
      box.addEventListener(evt, (e) => e.stopPropagation(), { passive: evt !== "wheel" });
    }
  }

  $("bgFile").addEventListener("change", async () => {
    const file = $("bgFile").files && $("bgFile").files[0];
    if (!file) return;
    if (file.size > 25 * 1024 * 1024) { toast("That photo is over 25 MB.", true); return; }
    const form = new FormData();
    form.append("file", file, file.name);
    toast("Uploading background…");
    try {
      const response = await fetch("/api/stage/background", { method: "POST", body: form });
      const state = await response.json().catch(() => null);
      if (!response.ok) throw new Error((state && state.detail) || "HTTP " + response.status);
      applyStage(state);
      toast("Background set.");
    } catch (error) { toast(String(error.message || error), true); }
    $("bgFile").value = "";
  });

  let blurTimer = null;
  $("bgBlur").addEventListener("input", () => {
    const blur = Number($("bgBlur").value) || 0;
    $("stageBg").style.filter = blur ? `blur(${blur}px)` : "";
    $("bgBlurVal").textContent = blur + "px";
    clearTimeout(blurTimer);
    blurTimer = setTimeout(() => postStage({ blur }).catch(() => {}), 500);
  });

  $("bgRemove").addEventListener("click", async () => {
    try { applyStage(await postStage({ clear_background: true })); toast("Background removed."); }
    catch (error) { toast(String(error.message || error), true); }
  });

  // ── camera → she can see (frames go to /ws/camera; the local VLM describes them) ──
  let camWs = null, camStream = null, camVideo = null, camTimer = null, camPoll = null, camActive = false;
  const camPip = $("camPip"), camPipVideo = $("camPipVideo");   // your self-view: the same stream, mirrored by CSS only

  function camStop(silent) {
    camActive = false;
    clearInterval(camTimer); camTimer = null;
    clearInterval(camPoll); camPoll = null;
    if (camWs) { try { camWs.close(); } catch (_) {} camWs = null; }
    if (camStream) { for (const t of camStream.getTracks()) t.stop(); camStream = null; }
    camVideo = null;
    camPipVideo.srcObject = null;
    camPip.hidden = true;
    stageArea.classList.remove("cam-live");
    $("camBtn").classList.remove("live");
    $("camBtn").title = "Camera — let her see you";
    if (!silent) toast("Camera off — she can't see you anymore.");
  }

  async function camStart() {
    let vision = null;
    try { vision = await getJSON("/api/vision"); } catch (_) {}
    if (vision && !vision.enabled) {
      toast("Vision is disabled — add the vision block in settings.json and relaunch.", true);
      return;
    }
    try {
      camStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
      });
    } catch (_) { toast("Camera unavailable or permission denied.", true); return; }
    camVideo = document.createElement("video");
    camVideo.srcObject = camStream; camVideo.muted = true; camVideo.playsInline = true;
    try { await camVideo.play(); } catch (_) {}
    const proto = location.protocol === "https:" ? "wss" : "ws";
    camWs = new WebSocket(`${proto}://${location.host}/ws/camera`);
    camWs.binaryType = "arraybuffer";
    camWs.onclose = () => { if (camActive) camStop(true); };
    camActive = true;
    $("camBtn").classList.add("live");
    camPipVideo.srcObject = camStream;      // no second getUserMedia; the frames she gets come off camVideo, unmirrored
    camPipVideo.play().catch(() => {});
    camPip.hidden = false;
    stageArea.classList.add("cam-live");    // the HUD text wraps clear of the preview
    const canvas = document.createElement("canvas");
    camTimer = setInterval(() => {          // ~2 fps JPEG, plenty for scene description
      if (!camActive || !camWs || camWs.readyState !== 1 || !camVideo || camVideo.readyState < 2) return;
      const vw = camVideo.videoWidth || 640, vh = camVideo.videoHeight || 480;
      const w = 320, h = Math.max(1, Math.round((320 * vh) / vw));
      canvas.width = w; canvas.height = h;
      canvas.getContext("2d").drawImage(camVideo, 0, 0, w, h);
      canvas.toBlob((blob) => {
        if (blob && camWs && camWs.readyState === 1 && camWs.bufferedAmount < 1000000) {
          blob.arrayBuffer().then((buf) => { try { camWs.send(buf); } catch (_) {} });
        }
      }, "image/jpeg", 0.7);
    }, 500);
    camPoll = setInterval(async () => {     // hover the button to read what she sees
      try {
        const v = await getJSON("/api/vision");
        $("camBtn").title = v.description ? "She sees: " + v.description : "Camera live — she's looking…";
      } catch (_) {}
    }, 4000);
    toast("Camera on — she can see you now.");
  }

  $("camBtn").addEventListener("click", () => (camActive ? camStop() : camStart()));
  window.addEventListener("beforeunload", () => camStop(true));

  $("fsBtn").addEventListener("click", () => {
    if (document.fullscreenElement) { document.exitFullscreen(); return; }
    const go = stageArea.requestFullscreen || stageArea.webkitRequestFullscreen;
    if (go) go.call(stageArea);
    else toast("This browser doesn't allow fullscreen here.", true);
  });

  // TravelProp-style fullscreen: the live controls move into a floating glass bar
  const fsBar = el("div", { class: "fs-bar" });
  stageArea.append(fsBar);
  for (const evt of ["mousedown", "touchstart", "wheel"]) {
    fsBar.addEventListener(evt, (e) => e.stopPropagation(), { passive: evt !== "wheel" });
  }
  const FS_CONTROLS = ["powerBtn", "micBtn", "sayForm", "stopBtn"];
  const fsHomes = new Map();   // id -> {parent, next} to restore the page layout
  function enterFsLayout() {
    for (const id of FS_CONTROLS) {
      const node = $(id);
      fsHomes.set(id, { parent: node.parentNode, next: node.nextSibling });
      fsBar.append(node);
    }
  }
  function exitFsLayout() {
    for (const id of [...fsHomes.keys()].reverse()) {
      const home = fsHomes.get(id);
      home.parent.insertBefore($(id), home.next);
    }
    fsHomes.clear();
  }
  document.addEventListener("fullscreenchange", () => {
    const fs = !!document.fullscreenElement;
    $("fsBtn").title = fs ? "Exit fullscreen" : "Fullscreen";
    if (fs && !fsHomes.size) enterFsLayout();
    else if (!fs && fsHomes.size) exitFsLayout();
    // let the layout settle, then resize the renderer + remap the model
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (avatar.notifyResize) avatar.notifyResize();
    }));
  });

  // ── SFX cache ──
  const sfx = { dir: "/static/sfx/", buffers: new Map() };
  getJSON("/sfx_manifest").then((m) => {
    sfx.dir = m.dir || sfx.dir;
    for (const file of m.files || []) {
      fetch(sfx.dir + encodeURIComponent(file)).then((r) => r.ok ? r.arrayBuffer() : null)
        .then((buf) => { if (buf) sfx.buffers.set(file, buf); }).catch(() => {});
    }
  }).catch(() => {});

  // ── chat log ──
  const chatLog = $("chatLog");
  let chatEmpty = true;
  function chatAdd(kind, who, text, extra) {
    if (!text) return;
    if (chatEmpty) { chatLog.replaceChildren(); chatEmpty = false; }
    const line = el("div", { class: "chat-line " + kind },
      el("span", { class: "who", text: who }), text);
    if (extra) line.append(el("div", { class: "thought", text: extra }));
    chatLog.append(line);
    while (chatLog.children.length > 120) chatLog.firstChild.remove();
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  // ── power / socket ──
  let ws = null, wsWanted = false, reconnectTimer = null;
  let micActive = false, micStream = null, micCtx = null, micSrc = null, micNode = null;
  // current turn's id; audio tagged for another turn is dropped (classic-console guard)
  let ttsTurnBase = "";

  function setPowerUI(state, detail) {
    $("powerState").textContent = state;
    $("powerDetail").textContent = detail || "";
    const on = state === "Online" || state === "Listening";
    $("powerBtn").classList.toggle("on", on);
    $("sessionDot").className = "dot " + (on ? "ok" : state === "Connecting" ? "warn" : "");
    setNavDot(1, on ? "ok" : "err");
    $("sayBtn").disabled = !on;
    $("stopBtn").disabled = !on;
    $("injectBtn").disabled = !on;
    $("ctxBtn").disabled = !on;
    $("micBtn").classList.toggle("live", micActive);
  }
  setPowerUI("Offline", "press power to connect");

  function wsURL() {
    return (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/dialog";
  }

  function connect() {
    wsWanted = true;
    setPowerUI("Connecting", "opening dialog socket…");
    ws = new WebSocket(wsURL());
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      setPowerUI("Online", "connected · " + (micActive ? "mic live" : "text & mic ready"));
      ws.send(JSON.stringify({ type: "settings", voice_output: $("voiceOutToggle").checked }));
      pollStatus();
    };
    ws.onclose = () => {
      ws = null;
      stopMic(true);
      avatar.flush();
      resetEmote();
      if (wsWanted) {
        setPowerUI("Connecting", "reconnecting…");
        reconnectTimer = setTimeout(connect, 2500);
      } else {
        setPowerUI("Offline", "press power to connect");
      }
    };
    ws.onerror = () => {};
    ws.onmessage = (event) => {
      if (typeof event.data !== "string") return;
      let msg;
      try { msg = JSON.parse(event.data); } catch (_) { return; }
      handle(msg);
    };
  }

  function disconnect() {
    wsWanted = false;
    clearTimeout(reconnectTimer);
    stopMic(true);
    if (ws) { try { ws.close(); } catch (_) {} ws = null; }
    avatar.flush();
    setPowerUI("Offline", "press power to connect");
  }

  $("powerBtn").addEventListener("click", () => (wsWanted ? disconnect() : connect()));

  // ── mic (AudioWorklet pcm-pusher, ported from the classic console) ──
  function workletURL() {
    const code = `
class PCMPusher extends AudioWorkletProcessor {
  constructor(opt){
    super();
    const o = (opt && opt.processorOptions) || {};
    this.srcRate = o.sourceRate || sampleRate;
    this.dstRate = o.targetRate || 16000;
    this.frameSize = o.frameSize || 512;
    this.ratio = this.srcRate / this.dstRate;
    this._out = new Int16Array(this.frameSize);
    this._outOff = 0;
  }
  _pushSample(f){
    let s = Math.max(-1, Math.min(1, f));
    this._out[this._outOff++] = s < 0 ? (s * 0x8000) | 0 : (s * 0x7FFF) | 0;
    if(this._outOff >= this.frameSize){
      const buf = this._out.slice().buffer;
      this.port.postMessage(buf, [buf]);
      this._outOff = 0;
    }
  }
  process(inputs){
    const input = inputs[0];
    if(!input || !input[0] || input[0].length === 0) return true;
    const ch = input[0];
    if(Math.abs(this.ratio - 1) < 1e-6){
      for(let i=0;i<ch.length;i++) this._pushSample(ch[i]);
      return true;
    }
    let pos = 0;
    const step = this.ratio;
    while(pos < ch.length - 1){
      const i = pos | 0;
      const frac = pos - i;
      this._pushSample(ch[i] * (1 - frac) + ch[i+1] * frac);
      pos += step;
    }
    return true;
  }
}
registerProcessor('pcm-pusher', PCMPusher);`;
    return URL.createObjectURL(new Blob([code], { type: "application/javascript" }));
  }

  async function startMic() {
    if (micActive) return;
    if (!wsWanted) connect();
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
    } catch (error) {
      toast("Microphone was refused: " + error.message, true);
      return;
    }
    try { micCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR }); }
    catch (_) { micCtx = new (window.AudioContext || window.webkitAudioContext)(); }
    try { await micCtx.audioWorklet.addModule(workletURL()); }
    catch (error) { toast("AudioWorklet failed: " + error.message, true); stopMic(true); return; }
    micSrc = micCtx.createMediaStreamSource(micStream);
    micNode = new AudioWorkletNode(micCtx, "pcm-pusher", {
      processorOptions: { sourceRate: micCtx.sampleRate, targetRate: TARGET_SR, frameSize: 512 },
    });
    micNode.port.onmessage = (ev) => {
      if (ws && ws.readyState === 1) { try { ws.send(ev.data); } catch (_) {} }
    };
    micSrc.connect(micNode);
    const silent = micCtx.createGain();
    silent.gain.value = 0;
    micNode.connect(silent).connect(micCtx.destination);
    if (ws && ws.readyState === 1) {
      ws.send(JSON.stringify({ type: "hello", sample_rate: TARGET_SR, source_rate: micCtx.sampleRate }));
    }
    micActive = true;
    setPowerUI("Listening", "mic is live — just talk");
  }

  function stopMic(silent) {
    if (micNode) { try { micNode.disconnect(); } catch (_) {} micNode = null; }
    if (micSrc) { try { micSrc.disconnect(); } catch (_) {} micSrc = null; }
    if (micCtx) { try { micCtx.close(); } catch (_) {} micCtx = null; }
    if (micStream) { for (const track of micStream.getTracks()) track.stop(); micStream = null; }
    micActive = false;
    if (!silent) setPowerUI(wsWanted ? "Online" : "Offline", wsWanted ? "mic off" : "press power to connect");
    $("micBtn").classList.remove("live");
  }

  $("micBtn").addEventListener("click", () => (micActive ? stopMic() : startMic()));

  // ── outgoing text / controls ──
  $("sayForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = $("sayInput").value.trim();
    if (!text || !ws || ws.readyState !== 1) return;
    ws.send(JSON.stringify({ type: "text", text }));
    // the server echoes a `commit` for typed text — the chat line renders from that
    $("sayInput").value = "";
  });
  $("stopBtn").addEventListener("click", () => {
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "stop" }));
    avatar.flush();
    $("hudSay").textContent = "";
  });
  $("voiceOutToggle").addEventListener("change", () => {
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "settings", voice_output: $("voiceOutToggle").checked }));
  });
  $("injectBtn").addEventListener("click", () => {
    const text = $("injectInput").value.trim();
    if (text && ws && ws.readyState === 1) {
      ws.send(JSON.stringify({ type: "inject", text }));   // rendered via the commit echo
      $("injectInput").value = "";
    }
  });
  $("ctxBtn").addEventListener("click", () => {
    const text = $("ctxInput").value.trim();
    if (text && ws && ws.readyState === 1) {
      ws.send(JSON.stringify({ type: "context_inject", text }));
      toast("Context queued for the next turn.");
      $("ctxInput").value = "";
    }
  });

  // ── incoming dialog events ──
  function handle(msg) {
    switch (msg.type) {
      case "ready":
        break;
      case "speech_start":
        // Barge-in (ported from the classic console): the server's gate confirmed the
        // user is really speaking and has already voided the turn + its TTS queue —
        // stop local playback NOW, with ack suppression for the killed chunk.
        avatar.flush();
        resetEmote();
        ttsTurnBase = "";
        $("hudSay").textContent = "";
        $("hudHear").textContent = "…";
        break;
      case "partial":
        $("hudHear").textContent = msg.text || "";
        break;
      case "commit":
        if (msg.text) {
          chatAdd("you", msg.input === "inject" ? "Cue" : "You", msg.text);
          // turn change: adopt the new turn id; stop any leftover audio (safety net —
          // the voice path already stopped at speech_start)
          ttsTurnBase = msg.turn_id || "";
          avatar.flush();
          resetEmote();
          $("hudSay").textContent = "";
        }
        $("hudHear").textContent = "";
        break;
      case "cancel":
        $("hudHear").textContent = "";
        break;
      case "ai_begin":
        // proactive turns (reflection 开口 / idle watcher) have no commit — adopt
        // their turn id so their audio passes the stale-turn guard
        if (msg.proactive && msg.turn_id) {
          ttsTurnBase = msg.turn_id;
          avatar.flush();
          resetEmote();
        }
        if (msg.proactive) scheduleToolPoll(500);   // a helper's result is usually what she is about to relay
        break;
      case "ai_surface_done": {
        const line = msg.reply || msg.speech || "";
        if (line) chatAdd("her", "Amika", line);
        if (msg.tone) avatar.surfaceEmotion(msg.tone);
        emote.deepExpected = msg.mode === "say+think" || msg.mode === "think";   // the deep layer is up next
        if (msg.surface_tts_done) surfaceClosed();     // nothing more to play for the surface (maybe nothing at all)
        else refreshEmote();                           // the last surface chunk is still coming: it will carry surface_last
        break;
      }
      case "ai_deep_done": {
        const spoken = Array.isArray(msg.segments)
          ? msg.segments.map((s) => s.text).join(" ")
          : (msg.speech || "");
        if (spoken) chatAdd("her", "Amika", spoken, msg.thought ? "心想 · " + msg.thought : null);
        emote.deepExpected = false;                    // the deep has answered: her thinking is over, whatever the audio does
        setDeepPending(false);
        break;
      }
      case "tts_audio":
        if (ttsTurnBase && msg.turn_id && !String(msg.turn_id).startsWith(ttsTurnBase)) break; // stale turn
        if (!msg.audio_b64) break;
        {
          const meta = { ack: true, turn_id: msg.turn_id, idx: msg.idx, text: msg.text || "", emotion: msg.emotion,
                         kind: msg.kind || null, surfaceLast: !!msg.surface_last };
          noteQueued(meta);
          avatar.playChunk(msg.audio_b64, msg.emotion || null, msg.pause_after || 0, meta);
        }
        break;
      case "tts_sfx": {
        if (ttsTurnBase && msg.turn_id && !String(msg.turn_id).startsWith(ttsTurnBase)) break; // stale turn
        const buf = sfx.buffers.get(msg.file);
        if (buf) {
          const meta = { ack: true, turn_id: msg.turn_id, idx: msg.idx, text: null,
                         kind: msg.kind || null, surfaceLast: !!msg.surface_last };
          noteQueued(meta);
          avatar.playBuffer(buf, null, msg.pause_after || 0, meta);
        } else {
          if (ws && ws.readyState === 1) {
            // keep the server's playback accounting honest even when the wav is missing
            ws.send(JSON.stringify({ type: "tts_playing", turn_id: msg.turn_id, idx: msg.idx }));
            ws.send(JSON.stringify({ type: "tts_finished", turn_id: msg.turn_id, idx: msg.idx }));
          }
          if (msg.surface_last) surfaceClosed();     // nothing of the surface left to play
        }
        break;
      }
      case "ai_end":
        resetEmote();
        pollStatus();
        scheduleToolPoll(1500);                      // the clerk dispatches right after she finishes
        break;
      case "ai_error":
        resetEmote();
        toast(msg.message || "She hit an error.", true);
        break;
      case "kicked":
        // The server parked us because OpenAmika was opened in another place.
        // Must NOT reconnect — that would kick the new place right back.
        wsWanted = false;
        clearTimeout(reconnectTimer);
        camStop(true);
        avatar.flush();
        resetEmote();
        toast(msg.reason || "OpenAmika was opened in another place.", true);
        setPowerUI("Offline", "disconnected — opened in another place");
        break;
      case "tts_error":
        if (msg.surface_last) surfaceClosed();         // the surface's final sentence failed: nothing left to wait for
        toast("TTS: " + (msg.error || "error"), true);
        break;
      case "memory_wiped":
        toast("Memory was wiped.");
        break;
      case "ai_timeline":
        onTimeline(msg);
        break;
      case "ai_timeline_void":
        rounds.delete(msg.turn_id);
        renderChart();
        break;
      case "ai_call_receipt":
        onReceipt(msg);
        break;
      default:
        break;
    }
  }

  // ── latency chart (dataviz: fixed categorical order, validated palette) ──
  const PHASES = [
    { key: "reflection_llm", label: "Reflection", v: "--viz-reflection" },
    { key: "stt", label: "STT", v: "--viz-stt" },
    { key: "embedding", label: "Embedding", v: "--viz-embed" },
    { key: "surface_llm", label: "Surface LLM", v: "--viz-surface" },
    { key: "surface_tts", label: "Surface TTS", v: "--viz-stts" },
    { key: "deep_llm", label: "Deep LLM", v: "--viz-deep" },
    { key: "deep_tts", label: "Deep TTS", v: "--viz-dtts" },
    { key: "writer_llm", label: "Memory writer", v: "--viz-writer" },
  ];
  const PHASE_KEY = { reflection: "reflection_llm", query_embedding: "embedding", surface: "surface_llm", deep: "deep_llm", memory_write: "writer_llm" };
  // the LLM calls whose prompt cache counts toward a round's cache hit (the speech server's round_summary.cached_pct)
  const LLM_KEYS = ["reflection_llm", "surface_llm", "deep_llm", "writer_llm"];
  const CACHE_NOTE = "Prompt cache hit: the share of the round's prompt tokens (surface, deep, reflection, memory writer) " +
    "that the provider read from its prompt cache. n/a: the model reports no cache figures (Ollama).";
  $("latLegend").replaceChildren(...PHASES.map((p) =>
    el("span", { class: "key" }, el("i", { style: "background:var(" + p.v + ")" }), p.label)),
    el("span", { class: "key cache-key", title: CACHE_NOTE }, "% = prompt cache hit"));

  const rounds = new Map();   // turn_id -> {order, segments: Map(key->[t0,t1]), extra: Map(key->{model,cost,tokens}), summary}
  let order = 0;
  const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);

  // Prompt cache share = cached tokens / the input of the calls that reported a cache figure. A call that reports
  // none (Ollama: cached is null) is "n/a", never a 0 % miss.
  function cacheInput(t) {
    if (!t || num(t.cached) === null) return null;
    return num(t.cache_input) !== null ? num(t.cache_input) : num(t.input);
  }
  function tokensLine(t) {
    if (!t) return null;
    const cached = num(t.cached), base = cacheInput(t);
    const cache = cached === null ? "cache n/a"
      : fmt.int(cached) + " cached" + (base ? " (" + fmt.pct((100 * cached) / base) + ")" : "");
    return `tokens ${fmt.int(num(t.input))} in · ${cache} · ${fmt.int(num(t.output))} out`;
  }
  // {pct, cached, input} for a round, {pct: null} when its calls report no cache figures, null before any LLM figures
  function roundCache(round) {
    const s = round.summary || {};
    if (num(s.cached_pct) !== null) {
      return { pct: num(s.cached_pct), cached: num(s.cached_tokens), input: num(s.cache_input_tokens) };
    }
    let cached = 0, input = 0, tokens = num(s.total_input_tokens) || 0;
    for (const key of LLM_KEYS) {
      const t = (round.extra.get(key) || {}).tokens;
      if (!t) continue;
      tokens += num(t.input) || 0;
      const base = cacheInput(t);
      if (base !== null) { cached += num(t.cached); input += base; }
    }
    if (input > 0) return { pct: (100 * cached) / input, cached, input };
    return tokens > 0 ? { pct: null } : null;
  }

  function spanOf(entry) {
    const solid = Array.isArray(entry.solid) ? entry.solid : null;
    const stripe = Array.isArray(entry.stripe) ? entry.stripe : null;
    const points = [];
    if (solid && num(solid[0]) !== null && num(solid[1]) !== null) points.push(solid[0], solid[1]);
    if (stripe && num(stripe[0]) !== null && num(stripe[1]) !== null) points.push(stripe[0], stripe[1]);
    if (!points.length) return null;
    return [Math.min(...points), Math.max(...points)];
  }

  function roundFor(tid) {
    let round = rounds.get(tid);
    if (!round) {
      round = { order: ++order, segments: new Map(), extra: new Map(), summary: {} };
      rounds.set(tid, round);
      while (rounds.size > 24) rounds.delete(rounds.keys().next().value);
    }
    return round;
  }

  function onTimeline(msg) {
    if (typeof msg.turn_id !== "string" || !msg.turn_id) return;
    const round = roundFor(msg.turn_id);
    for (const seg of Array.isArray(msg.segments) ? msg.segments : []) {
      if (!seg || !PHASES.some((p) => p.key === seg.key)) continue;
      const span = spanOf(seg);
      if (span) round.segments.set(seg.key, span);
      if (seg.model || seg.cost_usd != null || seg.tokens) {
        round.extra.set(seg.key, { model: seg.model, cost: num(seg.cost_usd), tokens: seg.tokens });
      }
    }
    if (msg.round_summary && typeof msg.round_summary === "object") round.summary = msg.round_summary;
    renderChart();
  }

  function onReceipt(msg) {
    const r = (msg.receipt && typeof msg.receipt === "object") ? msg.receipt : msg;
    const key = PHASE_KEY[r.phase];
    if (!key) return;
    const tid = key === "reflection_llm" ? r.open_turn_id : r.turn_id;
    if (typeof tid !== "string" || !tid || !rounds.has(tid)) return;
    const round = rounds.get(tid);
    const prev = round.extra.get(key) || {};
    round.extra.set(key, {
      model: prev.model || r.model,
      cost: prev.cost != null ? prev.cost : num(r.cost_usd),
      tokens: prev.tokens || { input: num(r.input_tokens), cached: num(r.cached_tokens), output: num(r.output_tokens),
        cache_input: num(r.cached_tokens) !== null ? num(r.input_tokens) : null },
    });
    if (!round.segments.has(key) && num(r.started_ms) !== null) {
      const end = num(r.last_token_ms) !== null ? r.last_token_ms
        : num(r.first_token_ms) !== null ? r.first_token_ms : null;
      if (end !== null) round.segments.set(key, [r.started_ms, end]);
    }
    renderChart();
  }

  const tip = $("vizTip");
  function showTip(event, title, lines) {
    tip.replaceChildren(el("div", { class: "t-title", text: title }),
      ...lines.filter(Boolean).map((line) => el("div", { class: "t-sub", text: line })));
    tip.classList.add("show");
    const x = Math.min(event.clientX + 14, window.innerWidth - 310);
    tip.style.left = x + "px";
    tip.style.top = Math.max(8, event.clientY - 10) + "px";
  }
  function hideTip() { tip.classList.remove("show"); }

  function renderChart() {
    const box = $("latChart");
    const list = [...rounds.entries()].sort((a, b) => a[1].order - b[1].order).slice(-10);
    if (!list.length) return;
    box.replaceChildren();
    const firsts = [];
    for (const [tid, round] of list) {
      const spans = [...round.segments.entries()];
      if (!spans.length) continue;
      const t0 = Math.min(...spans.map(([, s]) => s[0]));
      const t1 = Math.max(...spans.map(([, s]) => s[1]));
      const total = Math.max(1, t1 - t0);
      const track = el("div", { class: "lat-track" });
      for (const phase of PHASES) {
        const span = round.segments.get(phase.key);
        if (!span) continue;
        const left = ((span[0] - t0) / total) * 100;
        const width = Math.max(.6, ((span[1] - span[0]) / total) * 100);
        const seg = el("div", {
          class: "lat-seg",
          style: `left:${left}%;width:${width}%;background:var(${phase.v})`,
        });
        const extra = round.extra.get(phase.key) || {};
        seg.addEventListener("mousemove", (event) => showTip(event, phase.label, [
          fmt.ms(span[1] - span[0]) + " · starts +" + fmt.ms(span[0] - t0),
          extra.model ? String(extra.model) : null,
          tokensLine(extra.tokens),
          extra.cost != null ? "cost " + fmt.usd(extra.cost) : null,
        ]));
        seg.addEventListener("mouseleave", hideTip);
        track.append(seg);
      }
      const rid = tid.length > 7 ? "…" + tid.slice(-6) : tid;
      const cache = roundCache(round);
      const cacheTitle = !cache ? null : cache.pct === null
        ? "Prompt cache hit: n/a (this model reports no cache figures)"
        : "Prompt cache hit: " + fmt.int(cache.cached) + " of " + fmt.int(cache.input) + " prompt tokens read from cache";
      box.append(el("div", { class: "lat-row" },
        el("span", { class: "rid", text: rid }),
        track,
        el("span", { class: "cache" + (cache && cache.pct === null ? " na" : ""), title: cacheTitle,
          text: !cache ? "" : cache.pct === null ? "n/a" : fmt.pct(cache.pct) }),
        el("span", { class: "total", text: fmt.s1(total) })));
      const first = num(round.summary.latency_to_surface_ms);
      if (first !== null) firsts.push(first);
    }
    if (firsts.length) {
      const avg = firsts.reduce((a, b) => a + b, 0) / firsts.length;
      $("statFirst").innerHTML = "";
      $("statFirst").append((avg / 1000).toFixed(2), el("small", { text: " s" }));
    }
  }

  // ── prompt cache hit, display period (f_usage summary). Hers alone: agent bookings carry no cache figures. Only her
  // calls whose provider reports a cache figure count; embeddings never do, Ollama reports none ("n/a", not 0 %). ──
  const ROLE_NAMES = { surface: "surface", deep: "deep", reflection: "reflection", memory_write: "memory writer" };
  function renderCacheTile(amika, usage) {
    const value = $("statCache"), sub = $("statCacheSub"), tile = value.closest(".stat-tile");
    const pct = amika ? num(amika.cache_hit_pct) : null;
    const unreported = amika ? num(amika.cache_unreported_requests) || 0 : 0;
    const calls = (n) => fmt.int(n) + (n === 1 ? " call" : " calls");
    const order = Object.keys(ROLE_NAMES);
    const roles = (Array.isArray(usage.by_phase) ? usage.by_phase : [])
      .filter((p) => ROLE_NAMES[p.phase] && num(p.cache_hit_pct) !== null)
      .sort((a, b) => order.indexOf(a.phase) - order.indexOf(b.phase))
      .map((p) => ROLE_NAMES[p.phase] + " " + fmt.pct(num(p.cache_hit_pct)));
    if (pct !== null) {
      const base = num(amika.cache_input_tokens) !== null ? num(amika.cache_input_tokens) : num(amika.input_tokens);
      value.textContent = fmt.pct(pct);
      sub.textContent = fmt.int(num(amika.cache_read_tokens)) + " of " + fmt.int(base) + " prompt tokens from cache" +
        (unreported ? " · " + calls(unreported) + " n/a (no cache figures)" : "") + " · agents excluded";
    } else if (unreported) {
      value.textContent = "n/a";
      sub.textContent = "her model reports no prompt cache figures (Ollama) · " + calls(unreported);
    } else {
      value.textContent = "—";
      sub.textContent = "prompt tokens read from cache";
    }
    if (tile) tile.title = roles.length ? "By role: " + roles.join(" · ") : "";
  }

  // ── usage / status polling ──
  let statusTimer = null;
  async function pollStatus() {
    clearTimeout(statusTimer);
    try {
      const data = await getJSON("/api/build_f/status");
      setNavDot(0, "ok");
      const lifetime = data.usage || {};
      const usage = lifetime.display && typeof lifetime.display === "object" ? lifetime.display : lifetime;
      // her own spend (OpenRouter-confirmed) vs the helpers' (Hermes, nanoMuse — priced from the tokens they
      // report, at list price): two numbers in one tile
      const amika = usage.amika || null, agents = usage.agents || null;
      const dialogues = amika ? num(amika.dialogues) || 0 : 0, requests = agents ? num(agents.requests) || 0 : 0;
      $("statCostAmika").textContent = amika ? fmt.usd(num(amika.cost_usd) || 0) : "—";
      $("statCostAmikaSub").textContent = dialogues
        ? fmt.usd(num(amika.avg_per_dialogue_usd) || 0) + " / dialogue · " + fmt.int(dialogues) + (dialogues === 1 ? " dialogue" : " dialogues")
        : "no dialogues this period";
      $("statCostAgents").textContent = requests ? fmt.usd(num(agents.cost_usd) || 0) : "$0";
      $("statCostAgentsSub").textContent = requests
        ? fmt.usd(num(agents.avg_per_request_usd) || 0) + " / request · " + fmt.int(requests) + (requests === 1 ? " request" : " requests")
        : "no requests";
      renderCacheTile(amika, usage);
    } catch (_) {
      setNavDot(0, "err");
    }
    statusTimer = setTimeout(pollStatus, wsWanted ? 15000 : 60000);
  }
  pollStatus();
  getJSON("/api/tts_probe").then((p) => setNavDot(2, p.ok ? "ok" : "err")).catch(() => setNavDot(2, "err"));

  // ── tool dialogue: what she asked her helpers (Hermes, nanoMuse…) and what came back ──
  const toolLog = $("toolLog"), toolCard = $("toolDialogCard"), toolDot = $("toolDot");
  const TOOL_KIND = { ask: "asked", progress: "working", text: "says", question: "asks you", approval: "needs your approval",
    result: "result", error: "failed", stopped: "stopped", answer: "answered", approve: "approved", deny: "denied",
    stop: "stop", not_sent: "not sent", fabricated: "retracted" };
  let toolTimer = null, toolSeen = "", agentBusy = false, jobsSeen = "";
  function toolLabel(e) {
    if (e.who === "amika") return "Amika \u2192 " + e.server;
    if (e.who === "agent") return e.server + " \u2192 Amika";
    if (e.who === "you") return "you (Tools page) \u2192 " + e.server;
    return "system \u00b7 " + e.server;
  }
  function renderToolDialogue(entries) {
    const key = entries.length ? entries[entries.length - 1].ts + ":" + entries.length + ":" + entries[0].ts : "";
    if (key === toolSeen) return;
    toolSeen = key;
    if (!entries.length) return;
    toolLog.replaceChildren();
    for (const e of entries) {
      const cls = e.who === "amika" ? "her" : e.who === "you" ? "you" : e.who === "system" ? "sys" : "tool";
      const when = new Date((e.ts || 0) * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
      const line = el("div", { class: "chat-line " + cls + " k-" + (e.kind || "") },
        el("span", { class: "who", text: toolLabel(e) + " \u00b7 " + (TOOL_KIND[e.kind] || e.kind || "") + " \u00b7 " + when }),
        String(e.text || "").replace(/\*\*|__/g, ""));
      if (e.quote) line.append(el("div", { class: "thought", text: "you said: \u201c" + e.quote + "\u201d" }));
      toolLog.append(line);
    }
    toolLog.scrollTop = toolLog.scrollHeight;
  }
  // ── speech bubbles: the same exchange, played on stage ──
  // Her lines come out of her, the helper's lines out of the agent avatar (the paper doll that stands
  // in for Hermes / nanoMuse); the two turn to each other while a bubble is up. Lines that were already
  // in the log when the page opened stay in the log: only new ones get a bubble. The bridge's own steps
  // ("sent", "started", "stop requested"…, marked internal) stay in the log too: the doll says only what the
  // helper itself said or did.
  const bubbles = { her: $("bubbleHer"), agent: $("bubbleAgent") };
  const BUBBLE_MOOD = { result: "happy", error: "worried", stopped: "worried", fabricated: "worried",
    question: "surprise", approval: "surprise" };
  let bubbleSeen = null, bubbleRaf = 0;

  function bubbleNewLines(entries) {
    const last = entries.length ? Number(entries[entries.length - 1].ts) || 0 : 0;
    if (bubbleSeen === null) { bubbleSeen = last; return; }
    const fresh = entries.filter((e) => (Number(e.ts) || 0) > bubbleSeen && !e.internal &&
                                        (e.who === "amika" || e.who === "agent"));
    bubbleSeen = Math.max(bubbleSeen, last);
    // replay at the real pace: she typed each command, sent it when the typing was done, and the helper answered
    // (the brain's outbox, f_tools._type_and_send) — so each line appears as far after the first as it happened
    const shown = fresh.slice(-8), base = shown.length ? Number(shown[0].ts) || 0 : 0;
    shown.forEach((e) => setTimeout(() => showBubble(e), Math.min(20000, Math.max(0, ((Number(e.ts) || 0) - base) * 1000))));
  }

  function showBubble(e) {
    const side = e.who === "amika" ? "her" : "agent";
    const box = bubbles[side];
    if (!box) return;
    const text = String(e.text || "").replace(/\*\*|__/g, "").replace(/\s+/g, " ").trim();
    if (!text) return;
    const short = text.length > 220 ? text.slice(0, 218).trimEnd() + "\u2026" : text;
    box.querySelector(".who").textContent = toolLabel(e) + " \u00b7 " + (TOOL_KIND[e.kind] || e.kind || "");
    const ms = Math.min(16000, 3800 + short.length * 45);
    box.classList.add("show");
    requestAnimationFrame(() => box.classList.add("in"));
    // the brain waited exactly typing_ms before sending this command; type it at that same speed
    typewrite(box.querySelector(".text"), short, e.typing_ms != null ? Number(e.typing_ms) : Math.min(2600, short.length * 26));
    clearTimeout(box._hide);
    box._hide = setTimeout(() => hideBubble(side), ms);
    avatar.converse(side === "her" ? "amika" : "agent", ms, side === "agent" ? BUBBLE_MOOD[e.kind] : null);
    placeBubbles();
    if (!bubbleRaf) bubbleRaf = requestAnimationFrame(bubbleFollow);
  }

  function hideBubble(side) {
    const box = bubbles[side];
    box.classList.remove("in");
    clearTimeout(box._gone);
    box._gone = setTimeout(() => { if (!box.classList.contains("in")) box.classList.remove("show"); }, 260);
  }

  function typewrite(node, text, ms) {
    clearInterval(node._typer);
    node.textContent = "";
    const step = Math.max(1, Math.round(text.length / Math.max(1, ms / 30)));   // characters per 30 ms tick
    let i = 0;
    node._typer = setInterval(() => {
      i = Math.min(text.length, i + step);
      node.textContent = text.slice(0, i);
      if (i >= text.length) clearInterval(node._typer);
    }, 30);
  }

  function placeBubbles() {
    const a = avatar.anchors ? avatar.anchors() : { her: null, agent: null };
    const area = $("stageArea"), W = area.clientWidth, H = area.clientHeight;
    for (const side of ["her", "agent"]) {
      const box = bubbles[side];
      if (!box.classList.contains("show")) continue;
      const p = a[side];
      if (!p) { box.classList.remove("show", "in"); continue; }
      const bw = box.offsetWidth, bh = box.offsetHeight;
      let left, top;
      if (side === "her") { left = p.x + p.w * 0.14; top = p.y - bh - 4; }       // beside her head, tail down-left
      else { left = p.x - p.w * 0.16 - bw; top = p.y - bh + 6; }                  // left of the doll's head, tail down-right
      left = Math.max(8, Math.min(W - bw - 8, left));
      top = Math.max(8, Math.min(H - bh - 8, top));
      box.style.left = left + "px"; box.style.top = top + "px";
    }
  }

  function bubbleFollow() {
    bubbleRaf = 0;
    if (!bubbles.her.classList.contains("show") && !bubbles.agent.classList.contains("show")) return;
    placeBubbles();
    bubbleRaf = requestAnimationFrame(bubbleFollow);
  }

  // ── polling the brain's tools status (jobs, health, dialogue) ──
  const TERMINAL_JOB = { done: 1, failed: 1, stopped: 1 };
  function scheduleToolPoll(ms) { clearTimeout(toolTimer); toolTimer = setTimeout(pollTools, ms); }

  // ── her helpers' jobs and approvals (moved here from the Tools page — same rows, same actions, same poll) ──
  const STATE_LABEL = { running: "working", waiting_approval: "needs your approval", waiting_question: "has a question",
                        done: "done", failed: "failed", stopped: "stopped" };
  const RECENT_S = 900;                        // a finished job stays on the card this long (the brain's own "recent" window)
  function stateBadge(st) {
    const cls = st === "done" ? "ok" : (st === "failed" ? "err" : (st && st.startsWith("waiting") ? "warn" : ""));
    return el("span", { class: "tag " + cls, text: STATE_LABEL[st] || st || "" });
  }
  async function jobAction(kind, job, extra) {
    try {
      if (kind === "approve") {
        await postAction("/api/build_f/tools/approve", "tools-approve", { server: job.server, job: job.job, approved: !!extra.approved });
        toast(extra.approved ? "Approved — she'll relay what happens next." : "Denied.");
      } else {
        const result = await postAction("/api/build_f/tools/job", "tools-job", { server: job.server, job: job.job, action: kind, text: extra.text || "" });
        toast(result.ok ? (kind === "stop" ? "Stop requested." : "Answer sent.") : "Failed: " + (result.result || ""), !result.ok);
      }
      scheduleToolPoll(300);
    } catch (error) { toast(String(error.message || error), true); }
  }
  function renderJobs(jobs) {
    const box = $("jobsBox");
    box.replaceChildren();
    for (const job of jobs.slice().reverse()) {
      const row = el("div", { class: "pending-item", style: "flex-wrap:wrap; gap:10px;" });
      const what = el("span", { class: "what", style: "flex:1 1 320px;" },
        el("b", { text: job.server }), " ", stateBadge(job.state), " ",
        el("span", { text: job.task || "" }));
      if (job.step && !TERMINAL_JOB[job.state]) what.append(el("div", { class: "card-hint", text: "→ " + job.step }));
      if (job.state === "done" && job.result) what.append(el("div", { class: "card-hint", text: String(job.result).slice(0, 400) }));
      if (job.state === "failed" && job.error) what.append(el("div", { class: "card-hint", style: "color:var(--err);", text: job.error }));
      row.append(what);
      const pending = job.pending;
      if (pending && pending.kind === "approval") {
        row.append(el("span", { class: "card-hint", style: "flex-basis:100%;", text: "Approval requested: " + (pending.text || "") }));
        row.append(el("button", { class: "btn small primary", text: "Approve", onclick: () => jobAction("approve", job, { approved: true }) }));
        row.append(el("button", { class: "btn small danger", text: "Deny", onclick: () => jobAction("approve", job, { approved: false }) }));
      } else if (pending && pending.kind === "question") {
        row.append(el("span", { class: "card-hint", style: "flex-basis:100%;", text: "It asks: " + (pending.text || "") }));
        const input = el("input", { type: "text", placeholder: "your answer", style: "flex:1 1 200px;" });
        row.append(input);
        row.append(el("button", { class: "btn small primary", text: "Answer", onclick: () => jobAction("answer", job, { text: input.value }) }));
      }
      if (!TERMINAL_JOB[job.state]) row.append(el("button", { class: "btn small", text: "Stop", onclick: () => jobAction("stop", job, {}) }));
      box.append(row);
    }
  }
  function renderPending(pending) {
    const box = $("pendingBox");
    box.replaceChildren();
    for (const item of pending) {
      box.append(el("div", { class: "pending-item" },
        el("span", { class: "what" },
          el("b", { text: item.server + "__" + item.tool }), " ",
          el("code", { text: JSON.stringify(item.arguments || {}) }),
          item.quote ? el("div", { class: "card-hint", text: "you said: “" + item.quote + "”" }) : ""),
        el("button", { class: "btn small primary", text: "Approve & run", onclick: async () => {
          try {
            const result = await postAction("/api/build_f/tools/approve", "tools-approve", { id: item.id });
            toast(result.is_error ? "Ran with an error — see log." : "Done — she'll tell you the result.", !!result.is_error);
            scheduleToolPoll(300);
          } catch (error) { toast(String(error.message || error), true); }
        } }),
        el("button", { class: "btn small", text: "Dismiss", onclick: async () => {
          try { await postAction("/api/build_f/tools/dismiss", "tools-dismiss", { id: item.id }); scheduleToolPoll(300); }
          catch (error) { toast(String(error.message || error), true); }
        } })));
    }
  }
  function renderJobsCard(d) {
    const now = Date.now() / 1000;
    const jobs = (Array.isArray(d.jobs) ? d.jobs : []).filter((j) => j && (!TERMINAL_JOB[j.state] || now - (Number(j.updated) || 0) < RECENT_S));
    const pending = Array.isArray(d.pending) ? d.pending : [];
    const card = $("jobsCard");
    card.hidden = !jobs.length && !pending.length;             // nothing running, recent or waiting: no card at all
    if (card.hidden) { jobsSeen = ""; return; }
    const waiting = pending.length || jobs.some((j) => j.pending);
    $("jobsDot").className = "dot " + (waiting ? "warn" : jobs.some((j) => !TERMINAL_JOB[j.state]) ? "ok" : "");
    const key = JSON.stringify([jobs, pending]);
    if (key === jobsSeen) return;
    const active = document.activeElement;
    if (active && active.tagName === "INPUT" && card.contains(active)) return;   // an answer is being typed: keep the rows
    jobsSeen = key;
    renderJobs(jobs);
    renderPending(pending);
  }
  async function pollTools() {
    clearTimeout(toolTimer);
    try {
      const d = await getJSON("/api/build_f/tools");
      const servers = Object.values(d.servers || {});
      const enabled = !!d.enabled && servers.some((s) => s.enabled);
      const agents = !!d.enabled && servers.some((s) => s.enabled && s.kind === "agent");
      const entries = d.dialogue || [];
      toolCard.hidden = !enabled && !entries.length;
      const down = servers.some((s) => s.enabled && s.connected && s.health && s.health.ok === false);
      const up = servers.some((s) => s.enabled && s.connected && (!s.health || s.health.ok !== false));
      toolDot.className = "dot " + (down ? "err" : up ? "ok" : enabled ? "warn" : "");
      renderToolDialogue(entries);
      renderJobsCard(d);
      // a job still running = a helper is working for her right now: the doll "thinks" meanwhile
      agentBusy = (d.jobs || []).some((j) => j && !TERMINAL_JOB[j.state] && j.state !== "pending");
      avatar.setAgentVisible(agents);
      avatar.setAgentBase(agentBusy ? "working" : "idle");
      bubbleNewLines(entries);
    } catch (_) { /* the brain or the gateway is busy; try again later */ }
    toolTimer = setTimeout(pollTools, wsWanted ? 2000 : 10000);
  }
  pollTools();

  // ── first run: she introduces herself, asks your name and city, then checks her own wiring. Plays until
  // build_f.json says onboarded (Setup → Reset everything makes it play again). Hard-coded on purpose.
  async function firstRun() {
    let persona;
    try { persona = await getJSON("/api/build_f/persona"); } catch (_) { return; }   // brain down: the health light says so
    if (persona.onboarded) return;
    const amika = persona.persona_name || "Amika";
    const state = { name: persona.user_name && persona.user_name !== "User" ? persona.user_name : "",
                    city: persona.city || "", tz: "", sayAs: "" };
    try { state.tz = Intl.DateTimeFormat().resolvedOptions().timeZone || ""; } catch (_) {}
    if (!state.tz) state.tz = persona.timezone || "UTC";
    try {   // a pronunciation guide saved earlier for this name
      const pron = await getJSON("/api/name_pronunciation");
      if (pron.name && (!state.name || pron.name.toLowerCase() === state.name.toLowerCase())) {
        state.sayAs = pron.say_as || "";
        if (!state.name) state.name = pron.name;
      }
    } catch (_) {}
    const zones = Array.from(new Set([...(persona.timezones || []), persona.timezone, state.tz].filter(Boolean))).sort();
    // her face comes from the stage: wait for the model (or for the stage to say there is none)
    await new Promise((resolve) => {
      const t0 = Date.now();
      const id = setInterval(() => {
        if (avatarReady || !$("stageEmpty").hidden || Date.now() - t0 > 20000) { clearInterval(id); resolve(); }
      }, 150);
    });
    const backdrop = el("div", { class: "oa-wizard-backdrop" });
    const box = el("div", { class: "oa-wizard", role: "dialog", "aria-modal": "true", "aria-label": "Meet " + amika });
    backdrop.append(box);
    document.body.append(backdrop);
    const TOTAL = 6;
    let step = 0, keyStatus = null, lastHealth = null;
    const face = { wrap: null, canvas: null, badge: null, live: avatarReady && avatar.hasModel };

    // ── her voice: one line at a time through /api/tts_say, played through the stage (lip-sync on the
    // portrait) or a plain <audio> when there is no model. The first line cannot start before a click
    // (browser rule): the portrait then shows "click to hear her" and the next click plays it.
    let audioOK = false, pendingLine = null;
    const LINE_EMOTION = ["casual_happy", "intimate_caring", "casual_very_happy", "casual_smile", "casual_smile", "casual_happy"];
    const WIZARD_VOICE = "happy1";   // the V1 voice tag for every wizard line (owner's pick: the soft happy; happy2 sounded harsh)
    const setSpeaking = (on) => { if (face.wrap) face.wrap.classList.toggle("speaking", !!on); };
    const showTap = (line) => { pendingLine = line; if (face.badge) face.badge.hidden = false; };
    async function speak(text, emotion, raw) {
      if (!text) return false;
      const line = { text, emotion: emotion || "casual_happy", voice: WIZARD_VOICE, raw: !!raw };
      try {
        if (face.live && !(await avatar.ensureAudio())) { showTap(line); return false; }
        const r = await postAction("/api/tts_say", "tts-say", line);
        pendingLine = null;
        if (face.badge) face.badge.hidden = true;
        if (face.live) {
          avatar.flush();          // whatever she was saying stops; its onend never fires
          setSpeaking(false);
          avatar.playChunk(r.audio, line.emotion, 0, { kind: "wizard", onstart: () => setSpeaking(true), onend: () => setSpeaking(false) });
        } else {
          const mime = { opus: "audio/ogg", ogg: "audio/ogg", mp3: "audio/mpeg", wav: "audio/wav" }[r.audio_format] || "audio/wav";
          const a = new Audio("data:" + mime + ";base64," + r.audio);
          a.onplay = () => setSpeaking(true); a.onended = () => setSpeaking(false);
          await a.play();
        }
        audioOK = true;
        return true;
      } catch (error) {
        if (error && error.name === "NotAllowedError") { showTap(line); return false; }
        console.warn("[wizard] her voice:", error);
        return false;      // TTS down: the text is still there; the health step will say so
      }
    }
    box.addEventListener("pointerdown", () => { if (pendingLine) { const l = pendingLine; pendingLine = null; speak(l.text, l.emotion, l.raw); } }, true);
    const nameLine = (name, spelling) => `${spelling || name}. Hi, ${spelling || name}! Nice to finally meet you.`;
    const previewName = (name, spelling) => name && speak(nameLine(name, spelling), "intimate_caring", !!spelling);

    const lines = () => [
      `Hello! I'm ${amika}. So nice to finally meet you in person — we've been talking for ages, and now I've actually moved into your place. Remind me of your name again?`,
      `${state.name}. Got it. So… which city are we in? I want to go out and explore.`,
      `${state.city}! I'm so excited to be here — new streets, new supermarkets, new everything. Okay. Let's get me set up so I can actually talk to you.`,
      `First, my brain. I think best through OpenRouter — just paste a key, it only costs a few dollars a month. I can also run entirely on your own computer with Ollama, but only with a big model and a big graphics card.`,
      `And this lock is mine alone — a username and a password, just for us. The first ones you give me become the only key there is.`,
      `Last one — let me check I can hear you and that you can hear me.`,
    ];
    const speakCurrent = () => speak(lines()[step], LINE_EMOTION[step]);

    const say = (text) => el("div", { class: "say" }, text,
      el("button", { type: "button", class: "replay", title: "Hear it again", "aria-label": "Hear it again", text: "🔊", onclick: speakCurrent }));
    const steps = () => el("div", { class: "steps" }, ...Array.from({ length: TOTAL }, (_, i) => el("i", { class: i <= step ? "done" : "" })));
    const button = (text, onclick, cls) => el("button", { type: "button", class: "btn " + (cls || ""), text, onclick });
    const go = (n) => { step = Math.max(0, Math.min(TOTAL - 1, n)); render(); speakCurrent(); };
    function head() {
      face.wrap = el("div", { class: "portrait" + (face.live ? "" : " fallback") });
      if (face.live) {
        const dpr = Math.min(2, window.devicePixelRatio || 1), size = 150;
        face.canvas = el("canvas", { width: String(Math.round(size * dpr)), height: String(Math.round(size * dpr)), "aria-hidden": "true" });
        face.wrap.append(face.canvas);
        avatar.attachPortrait(face.canvas);
      } else {
        face.wrap.append(el("span", { class: "initial", text: amika.slice(0, 1) }));
      }
      face.badge = el("button", { type: "button", class: "badge", text: "Click to hear her", hidden: "" });
      face.wrap.append(face.badge);
      return el("div", { class: "head" }, face.wrap,
        el("div", { class: "head-text" }, el("span", { class: "who", text: amika + " · 桜井美香" }), steps()));
    }

    async function finish() {
      try {
        await postAction("/api/build_f/persona/identity", "persona-identity",
          { user_name: state.name || "User", city: state.city, timezone: state.tz, onboarded: true });
        if (state.name) {
          try { await postAction("/api/name_pronunciation", "name-pronunciation", { name: state.name, say_as: state.sayAs }); }
          catch (error) { toast("Pronunciation guide not saved: " + (error.message || error), true); }
        }
        if (face.live) avatar.detachPortrait();
        backdrop.remove();
        toast(amika + " knows you now — press power and say hi.");
      } catch (error) { toast("Could not save: " + (error.message || error), true); }
    }

    async function loadKeyStatus(line) {
      try { keyStatus = await getJSON("/api/build_f/openrouter_key"); }
      catch (_) { keyStatus = null; }
      line.textContent = keyStatus === null ? "key status unavailable — is the brain running?"
        : keyStatus.present ? "A key is saved ✓ " + (keyStatus.hint || "") : "No key saved yet.";
    }

    async function runCheck(rows, verdict, next) {
      rows.replaceChildren(el("div", { class: "card-hint", text: "Checking brain, speech, voice…" }));
      verdict.textContent = "";
      const result = await (window.OA.healthCheck ? window.OA.healthCheck() : Promise.resolve({ level: "warn", items: [] })).catch(() => ({ level: "err", items: [] }));
      lastHealth = result;
      rows.replaceChildren(...result.items.map((it) => el("div", { class: "health-row" },
        el("span", { class: "dot " + it.level }), el("span", { class: "name", text: it.name }), el("span", { class: "meta", text: it.detail || "" }))));
      verdict.className = "verdict " + result.level;
      verdict.textContent = result.level === "ok" ? "You're ready to go."
        : result.level === "warn" ? "Ready, with warnings — she can talk; have a look at the yellow rows when you can."
        : "Some part is missing — she can't fully work yet. Fix the red rows on the Setup page, then come back.";
      next.textContent = result.level === "err" ? "Start anyway" : "Start talking";
    }

    function render() {
      box.replaceChildren();
      box.append(head());
      const body = el("div", { class: "body" });
      const actions = el("div", { class: "actions" });
      if (step === 0) {
        box.append(say(lines()[0]));
        const input = el("input", { type: "text", maxlength: "40", placeholder: "your name", autocomplete: "off", value: state.name });
        const next = button("That's me →", () => { state.name = input.value.trim(); if (state.name) go(1); }, "primary");
        // her voice says the name back; a pronunciation guide fixes how she says it, for good
        const guide = el("input", { type: "text", maxlength: "80", placeholder: "e.g. Amika → Ameeka: write  Ameeka", autocomplete: "off", value: state.sayAs });
        const hear = button("🔊 Hear her say it", () => { state.name = input.value.trim(); state.sayAs = guide.value.trim(); previewName(state.name, state.sayAs); }, "small");
        const guideBox = el("div", { class: "guide", hidden: "" },
          el("div", { class: "form-row" },
            el("div", { class: "field", style: "flex:2" }, el("label", { text: "Spell it the way it sounds — for her voice only (example: Amika → Ameeka)" }), guide),
            button("Try it", () => { state.name = input.value.trim(); state.sayAs = guide.value.trim(); previewName(state.name, state.sayAs); }, "small")),
          el("p", { class: "hint", text: "She'll pronounce your name this way from now on, everywhere. What she writes keeps the real spelling. Leave it empty to undo." }));
        const guideLink = el("button", { type: "button", class: "editorial-link", text: "Doesn't sound like you? Add a pronunciation guide →",
          onclick: () => { guideBox.hidden = !guideBox.hidden; if (!guideBox.hidden) guide.focus(); } });
        if (state.sayAs) guideBox.hidden = false;
        const sync = () => { next.disabled = !input.value.trim(); hear.disabled = !input.value.trim(); };
        input.addEventListener("input", sync);
        guide.addEventListener("input", () => { state.sayAs = guide.value.trim(); });
        input.addEventListener("keydown", (e) => { if (e.key === "Enter" && input.value.trim()) next.click(); });
        sync();
        body.append(el("div", { class: "form-row" }, el("div", { class: "field", style: "flex:2" }, el("label", { text: "What should she call you?" }), input), hear),
          guideLink, guideBox);
        actions.append(next);
        setTimeout(() => input.focus(), 50);
      } else if (step === 1) {
        box.append(say(lines()[1]));
        const city = el("input", { type: "text", maxlength: "80", placeholder: "e.g. Los Angeles", autocomplete: "off", value: state.city });
        const tz = el("select", {});
        for (const z of zones) tz.append(el("option", { value: z, text: z === state.tz ? z + " (this computer)" : z }));
        tz.value = state.tz;
        const next = button("Here we are →", () => { state.city = city.value.trim(); state.tz = tz.value; if (state.city) go(2); }, "primary");
        city.addEventListener("input", () => { next.disabled = !city.value.trim(); });
        city.addEventListener("keydown", (e) => { if (e.key === "Enter" && city.value.trim()) next.click(); });
        next.disabled = !state.city;
        body.append(el("div", { class: "oa-grid cols-2" },
          el("div", { class: "field" }, el("label", { text: "Your city" }), city),
          el("div", { class: "field" }, el("label", { text: "Time zone (her clock)" }), tz)));
        body.append(el("p", { class: "hint", text: "Her days — the walk, the groceries, the gym, the late-night scrolling — follow this clock, and she'll name your city when she talks about going out." }));
        actions.append(button("← Back", () => go(0), "small"), next);
        setTimeout(() => city.focus(), 50);
      } else if (step === 2) {
        box.append(say(lines()[2]));
        body.append(el("p", { class: "hint", text: "Two quick things: the brain she thinks with, and a check that her voice and hearing are wired up." }));
        actions.append(button("← Back", () => go(1), "small"), button("Set me up →", () => go(3), "primary"));
      } else if (step === 3) {
        box.append(say(lines()[3]));
        // two ways to give her a brain: OpenRouter (cloud, a key) or Ollama (fully local, your GPU)
        const choice = el("div", { class: "prov-switch" });
        const orBtn = el("button", { type: "button", class: "prov", "aria-pressed": "false" }, el("b", { text: "OpenRouter" }), el("span", { text: "recommended · her tested models · a few dollars a month" }));
        const olBtn = el("button", { type: "button", class: "prov", "aria-pressed": "false" }, el("b", { text: "Ollama" }), el("span", { text: "fully local · a 27B+ model · a 32 GB GPU or a second PC" }));
        choice.append(orBtn, olBtn);
        const orPanel = el("div", { class: "body" }), olPanel = el("div", { class: "body", hidden: "" });
        const pick = (which) => {
          orPanel.hidden = which !== "openrouter"; olPanel.hidden = which !== "ollama";
          orBtn.setAttribute("aria-pressed", String(which === "openrouter")); olBtn.setAttribute("aria-pressed", String(which === "ollama"));
          if (which === "ollama") loadOllamaPanel();
        };
        orBtn.addEventListener("click", () => pick("openrouter"));
        olBtn.addEventListener("click", () => pick("ollama"));
        // OpenRouter panel
        const line = el("span", { class: "card-hint", text: "checking…" });
        const input = el("input", { type: "password", placeholder: "sk-or-v1-…", autocomplete: "off" });
        const save = button("Save key", async () => {
          const key = input.value.trim();
          if (!key) { toast("Paste your OpenRouter key first (sk-or-…).", true); return; }
          try {
            await postAction("/api/build_f/openrouter_key", "openrouter-key", { key });
            input.value = "";
            toast("Key saved — she's online.");
            loadKeyStatus(line);
          } catch (error) { toast(String(error.message || error), true); }
        }, "small primary");
        orPanel.append(el("div", { class: "form-row" }, el("div", { class: "field", style: "flex:2" }, el("label", { text: "OpenRouter API key" }), input), save), line,
          el("p", { class: "hint", text: "Get a key at openrouter.ai → Keys; that's the only account she needs. In almost all cases this is the better choice: her tested models, for a few dollars a month." }));
        loadKeyStatus(line);
        // Ollama panel
        const olUrl = el("input", { type: "text", placeholder: "http://127.0.0.1:11434", autocomplete: "off" });
        const olKey = el("input", { type: "text", placeholder: "only if your Ollama server asks for one", autocomplete: "off" });
        const olDialect = el("select", {}, el("option", { value: "ollama", text: "Ollama (native API)" }),
                              el("option", { value: "openai", text: "OpenAI-compatible server" }));
        const olStatus = el("span", { class: "card-hint", text: "" });
        const olSelect = el("select", {});
        const olNote = el("p", { class: "hint", text: "Use a model of 27B or more (70B or more for the full, reliable experience): smaller models may be really bad — she loses her character, forgets and breaks her replies. Install Ollama from ollama.com, run ollama pull gemma3:27b (17 GB), press Detect models, pick it, press Use this model, and she runs entirely on this computer — her memory search already does. A 27B model needs about 19 GB of GPU memory next to her own ~6 GB (~10 GB with her eyes): one 32 GB card holds it; with less, run Ollama on another PC (put its address above) or choose OpenRouter, in almost all cases the better choice." });
        const useBtn = button("Use this model", async () => {
          const model = olSelect.value;
          if (!model) { toast("Detect models first.", true); return; }
          try {
            await postAction("/api/build_f/ollama/use", "ollama-use", { model });
            olStatus.textContent = "She thinks locally on " + model + " now.";
            toast("Fully local: " + model + ".");
          } catch (error) { toast(String(error.message || error), true); }
        }, "small primary");
        useBtn.disabled = true;
        async function loadOllamaPanel() {
          try {
            const info = await getJSON("/api/build_f/ollama");
            if (!olUrl.value.trim()) olUrl.value = info.base_url || "http://127.0.0.1:11434";
            if (info.api_key_set && !olKey.value) olKey.placeholder = "✓ key saved — type to replace";
            olDialect.value = info.dialect || "ollama";
            const models = info.models || [];
            olSelect.replaceChildren(...(models.length ? models.map((m) => el("option", { value: m.name, text: m.name + (m.parameter_size ? " · " + m.parameter_size : "") + (m.size_gb ? " · " + m.size_gb + " GB" : "") }))
                                                       : [el("option", { value: "", text: "— detect models first —" })]));
            const inUse = info.in_use || {};
            if (inUse.deep) olSelect.value = inUse.deep;
            useBtn.disabled = !models.length;
            olStatus.textContent = info.reachable ? "Ollama " + (info.live_version || "") + " is running" + (models.length ? " · " + models.length + " model(s)" : " · no models detected yet")
              : "not answering at " + info.base_url + " — start Ollama, then Detect";
          } catch (error) { olStatus.textContent = String(error.message || error); }
        }
        const detect = button("Detect models", async () => {
          detect.disabled = true; olStatus.textContent = "asking Ollama…";
          try {
            const r = await postAction("/api/build_f/ollama/detect", "ollama-detect", { base_url: olUrl.value.trim(), api_key: olKey.value.trim(), dialect: olDialect.value });
            toast((r.models || []).length ? (r.models || []).length + " local model(s) found." : "Ollama answered, but has no models: ollama pull gemma3:27b (or another model of 27B or more)", !(r.models || []).length);
            await loadOllamaPanel();
          } catch (error) { olStatus.textContent = String(error.message || error); toast(String(error.message || error), true); }
          finally { detect.disabled = false; }
        }, "small");
        olPanel.append(el("div", { class: "form-row" }, el("div", { class: "field", style: "flex:2" }, el("label", { text: "Ollama server" }), olUrl), el("div", { class: "field", style: "flex:2" }, el("label", { text: "API key" }), olKey), el("div", { class: "field", style: "flex:1" }, el("label", { text: "Dialect" }), olDialect), detect), olStatus,
          el("div", { class: "form-row" }, el("div", { class: "field", style: "flex:2" }, el("label", { text: "Model for all four roles" }), olSelect), useBtn), olNote);
        body.append(choice, orPanel, olPanel);
        // start on whichever she already uses
        getJSON("/api/build_f/models").then((m) => {
          const t = (role) => { const item = ((m.options || {})[role] || []).find((o) => o.key === (m.selections || {})[role]); return item ? item.transport : "openrouter"; };
          pick(["surface", "deep", "reflection", "memory_write"].map(t).every((k) => k === "ollama") ? "ollama" : "openrouter");
        }).catch(() => pick("openrouter"));
        actions.append(button("← Back", () => go(2), "small"), button("Skip for now", () => go(4), "small skip"), button("Next →", () => go(4), "primary"));
      } else if (step === 4) {
        // account step (fork): first-run login sets the one account; after that it just signs in
        box.append(say(lines()[4]));
        const u = el("input", { type: "text", placeholder: "username", autocomplete: "username" });
        const p = el("input", { type: "password", placeholder: "password", autocomplete: "off" });
        const lockBtn = button("Lock it", async () => {
          try {
            const r = await fetch("/login", { method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ username: u.value.trim(), password: p.value }) });
            const j = await r.json().catch(() => ({}));
            if (!r.ok) throw new Error(j.error || "failed");
            toast(j.created ? "Locked — only you hold the key now." : "Signed in.");
            go(5);
          } catch (error) { toast(String(error.message || error), true); }
        }, "primary");
        body.append(el("div", { class: "form-row" },
          el("div", { class: "field", style: "flex:2" }, el("label", { text: "Username" }), u),
          el("div", { class: "field", style: "flex:2" }, el("label", { text: "Password (8+ characters)" }), p)),
          el("p", { class: "hint", text: "The first username and password you give her become the only account there is — it cannot be reset from outside. After that, the login page at /login is the way back in." }));
        actions.append(button("← Back", () => go(3), "small"), lockBtn);
      } else {
        box.append(say(lines()[5]));
        const rows = el("div", { class: "check" });
        const verdict = el("p", { class: "verdict" });
        const next = button("Start talking", finish, "primary");
        body.append(rows, verdict);
        actions.append(button("← Back", () => go(3), "small"), button("Re-check", () => runCheck(rows, verdict, next), "small skip"),
          el("a", { class: "btn small", href: "/setup", text: "Open Setup" }), next);
        runCheck(rows, verdict, next);
      }
      box.append(body, actions);
    }
    render();
    speakCurrent();   // before any click this parks the line on the portrait ("click to hear her")
  }
  firstRun();
})();
