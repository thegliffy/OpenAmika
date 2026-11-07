/* OpenAmika — Setup page: health, models, voice adapter, avatar, personality. */
(function () {
  "use strict";
  const { nav, footer, toast, getJSON, postAction, el, setNavDot } = window.OA;
  nav("setup"); footer();
  const $ = (id) => document.getElementById(id);

  // ───────────────────── health ─────────────────────
  async function health() {
    const rows = $("healthRows");
    rows.replaceChildren();
    const add = (state, name, meta) => rows.append(
      el("div", { class: "health-row" },
        el("span", { class: "dot " + state }),
        el("span", { class: "name", text: name }),
        el("span", { class: "meta", text: meta })));
    // brain (via speech proxy)
    try {
      const s = await getJSON("/api/build_f/status");
      const checks = s.template_check || {};
      const problems = Object.values(checks).reduce((a, list) => a + (Array.isArray(list) ? list.length : 0), 0);
      add(problems ? "warn" : "ok", "Brain", `${s.app || "?"} · templates ${problems ? problems + " warning(s)" : "OK"}`);
      const memory = s.memory || {};
      add(memory.degraded ? "warn" : "ok", "Memory",
        `${memory.events != null ? memory.events + " conversations" : "—"} · ${memory.facts != null ? memory.facts + " facts" : "—"}${memory.pending ? " · " + memory.pending + " pending" : ""}`);
      setNavDot(0, "ok");
    } catch (error) { add("err", "Brain", String(error.message || error)); setNavDot(0, "err"); }
    // speech
    try {
      const h = await getJSON("/health");
      add("ok", "Speech / STT", `${h.device || "?"} · cuda ${h.cuda_available ? "✓ " + (h.gpu_name || "") : "✗ (cpu)"}`);
      setNavDot(1, "ok");
    } catch (error) { add("err", "Speech / STT", String(error.message || error)); setNavDot(1, "err"); }
    // tts
    try {
      const p = await getJSON("/api/tts_probe");
      add(p.ok ? "ok" : "err", "TTS", `${p.url} · ${p.character}${p.ok ? "" : " · " + (p.error || "unreachable")}`);
      $("ttsUrlShow").textContent = p.url;
      $("ttsCharShow").textContent = p.character;
      $("ttsDot").className = "dot " + (p.ok ? "ok" : "err");
      $("ttsDotNote").textContent = p.ok ? "reachable" : (p.error || "unreachable — install her voice (envs\\brain\\Scripts\\python.exe scripts\\setup_tts.py), then start OpenAmika");
      setNavDot(2, p.ok ? "ok" : "err");
    } catch (_) { add("err", "TTS", "probe failed"); setNavDot(2, "err"); }
    // avatar + tools
    try {
      const a = await getJSON("/api/avatar/models");
      const mine = a.models.filter((m) => m.role !== "agent");
      add(mine.length ? "ok" : "err", "Live2D", a.models.length ? a.models.map((m) => m.name + (m.role === "agent" ? " (agent)" : "")).join(", ")
        + (mine.length ? "" : " — none for her: run envs\\brain\\Scripts\\python.exe scripts\\setup_tts.py") : "no model in avatar/models/ — run envs\\brain\\Scripts\\python.exe scripts\\setup_tts.py (her voice and Live2D model)");
    } catch (_) { add("err", "Live2D", "registry failed"); }
    try {
      const t = await getJSON("/api/build_f/tools");
      const names = Object.keys(t.servers || {});
      add(t.enabled ? "ok" : "warn", "Tools / MCP", t.enabled ? `enabled · ${names.length} server(s)` : "disabled (Tools page)");
    } catch (_) { add("warn", "Tools / MCP", "not available"); }
    // the launcher's pre-flight (logs/preflight.json): the lights the console showed at start — red never
    // blocked the launch, so anything not green is repeated here where it is easy to see
    try {
      const p = await getJSON("/api/preflight");
      const s = p.summary || {};
      if (p.at) {
        const level = s.err ? "err" : s.warn ? "warn" : "ok";
        add(level, "Startup checks", `${s.ok || 0} ok · ${s.warn || 0} warning(s) · ${s.err || 0} problem(s) · launcher at ${String(p.at).replace("T", " ")}`);
        for (const c of (p.checks || [])) {
          if (c.level !== "ok") add(c.level === "err" ? "err" : "warn", "↳ " + c.name, c.detail);
        }
      }
    } catch (_) {}
  }
  $("healthRefresh").addEventListener("click", health);
  health();

  // ───────────────────── vision ─────────────────────
  async function loadVision() {
    const rows = $("visionRows");
    const add = (state, name, meta) =>
      el("div", { class: "health-row" },
        el("span", { class: "dot " + state }),
        el("span", { class: "name", text: name }),
        el("span", { class: "meta", text: meta }));
    try {
      const v = await getJSON("/api/vision");
      if (!v.enabled) {
        rows.replaceChildren(add("warn", "Vision", "disabled — no vision block in settings.json (relaunch after adding one)"));
        return;
      }
      const s = v.service;
      const fresh = [];
      fresh.push(s
        ? add(s.model_loaded ? "ok" : "warn", "VLM service",
              `${s.model || "?"} · ${s.model_loaded ? "model loaded" : "loading…"} · feed ${s.feed_connected ? "connected" : "waiting"} · ${s.frames_seen || 0} frames`)
        : add("err", "VLM service", `unreachable at ${v.url} — check logs/vision.log`));
      fresh.push(add(v.camera_clients ? "ok" : "warn", "Camera feed",
        v.camera_clients ? `${v.camera_clients} camera client(s) streaming` : "no camera — press the camera button on the Use page"));
      fresh.push(add(v.description ? "ok" : "warn", "She sees",
        v.description || (v.last_description ? `(stale) ${v.last_description}` : "nothing described yet")));
      rows.replaceChildren(...fresh);
    } catch (error) {
      rows.replaceChildren(add("err", "Vision", String(error.message || error)));
    }
  }
  loadVision();
  setInterval(() => { if (!document.hidden) loadVision(); }, 3000);

  $("meterReset").addEventListener("click", async () => {
    try {
      await postAction("/api/build_f/meter/reset", "reset-meter", { action: "reset_display" });
      toast("Cost meter reset — a fresh display period starts now.");
      health();
    } catch (error) { toast(String(error.message || error), true); }
  });

  // ── danger zone (bottom of the page): wipe her memory / reset everything ──
  async function dangerAction(route, action, word, inputId, buttonId, done) {
    const input = $(inputId), button = $(buttonId), note = $("dangerNote");
    if (input.value.trim() !== word) { toast("Type " + word + " to confirm.", true); input.focus(); return; }
    button.disabled = true;
    note.textContent = "Working… (she has to be idle: the stage is disconnected while this runs)";
    try {
      const response = await fetch(route, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Build-F-Action": action },
        body: JSON.stringify({ confirmation: word }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = payload.detail || payload.error || ("HTTP " + response.status);
        throw new Error(response.status === 409 ? detail + " Power her off on the Use page, wait a few seconds, and try again." : detail);
      }
      input.value = "";
      note.textContent = done(payload);
      toast(note.textContent);
      health();
      loadPersona();
      loadBuilder();
    } catch (error) {
      note.textContent = String(error.message || error);
      toast(String(error.message || error), true);
    } finally { button.disabled = false; }
  }
  $("wipeBtn").addEventListener("click", () => dangerAction("/api/build_f/wipe", "wipe-memory", "WIPE", "wipeConfirm", "wipeBtn",
    (p) => "Memory wiped. Backup " + (p.backup_id || "kept") + " under memory_f/backups/."));
  $("resetBtn").addEventListener("click", () => dangerAction("/api/build_f/reset", "reset-all", "RESET", "resetConfirm", "resetBtn",
    (p) => "Reset done: memory backup " + (p.backup_id || "kept") + " · restored " + ((p.restored || []).join(", ") || "nothing") +
           " · names and city cleared. Open the Use page — she will introduce herself again."));

  // ───────────────────── models ─────────────────────
  let modelData = null;
  const ROLE_LABELS = { surface: "Surface — instant reply", deep: "Deep — her real voice", reflection: "Reflection — background coach", memory_write: "Memory writer" };

  async function loadModels() {
    try {
      modelData = await getJSON("/api/build_f/models");
    } catch (error) {
      $("modelsNote").textContent = "Model choices unavailable: " + (error.message || error);
      return;
    }
    // options[role] is a LIST of {key, name, description, disabled, disabled_reason, …}
    const ROLES = ["surface", "deep", "reflection", "memory_write"];
    const optionList = (role) => Array.isArray(modelData.options && modelData.options[role]) ? modelData.options[role] : [];
    const selections = modelData.selections || {};

    // always-visible summary of what she's running right now
    const summary = $("modelSummary");
    summary.replaceChildren(...ROLES.map((role) => {
      const sel = selections[role];
      const item = optionList(role).find((o) => o.key === sel);
      return el("div", { class: "row" },
        el("span", { class: "role", text: (ROLE_LABELS[role] || role).split(" — ")[0] }),
        el("span", { class: "mdl", text: item ? item.name : (sel || "not selected") }));
    }));
    // where she thinks: the provider switch reflects the live selection (mixed = both panels apply)
    const kinds = new Set(ROLES.map((role) => { const item = optionList(role).find((o) => o.key === selections[role]); return item ? item.transport : "openrouter"; }));
    const onOllama = kinds.has("ollama"), onCloud = kinds.has("openrouter") || kinds.has("alibaba") || kinds.has("native");
    $("provOpenrouter").classList.toggle("active", onCloud);
    $("provOllama").classList.toggle("active", onOllama);
    $("provOpenrouter").querySelector("span").textContent = onCloud && !onOllama ? "in use · recommended · her tested models" : "recommended · her tested models · a few dollars a month";
    $("provOllama").querySelector("span").textContent = onOllama && !onCloud ? "in use · fully local · a 27B+ model" : "fully local · a 27B+ model · a 32 GB GPU or a second PC";
    $("orBackNote").hidden = !onOllama;
    if (!provChosen) showProvider(onOllama && !onCloud ? "ollama" : "openrouter");

    const grid = $("roleGrid");
    grid.replaceChildren();
    for (const role of ROLES) {
      const options = optionList(role);
      const select = el("select", { id: "role_" + role });
      for (const item of options) {
        const option = el("option", { value: item.key, text: (item.name || item.key) + (item.disabled ? " (disabled)" : "") });
        if (item.disabled) option.disabled = true;
        select.append(option);
      }
      if (selections[role]) select.value = selections[role];
      const description = el("div", { class: "card-hint", style: "min-height:30px" });
      const refreshDesc = () => {
        const item = options.find((o) => o.key === select.value);
        description.textContent = (item && item.description) || "";
      };
      select.addEventListener("change", refreshDesc);
      refreshDesc();
      grid.append(el("div", { class: "field" }, el("label", { text: ROLE_LABELS[role] || role }), select, description));
    }
    loadOllama();
    const presets = $("presetSelect");
    presets.replaceChildren(...(Array.isArray(modelData.presets) ? modelData.presets : []).map((p) => {
      const name = (p && p.name) || String(p);
      const option = el("option", { value: name, text: name, title: (p && p.description) || "" });
      if (p && p.available === false) option.disabled = true;
      return option;
    }));
  }
  loadModels();

  // ── provider switch: OpenRouter (cloud) / Ollama (fully local) ──
  let provChosen = false;
  function showProvider(which) {
    $("orPanel").hidden = which !== "openrouter";
    $("ollamaPanel").hidden = which !== "ollama";
    $("provOpenrouter").setAttribute("aria-pressed", String(which === "openrouter"));
    $("provOllama").setAttribute("aria-pressed", String(which === "ollama"));
  }
  $("provOpenrouter").addEventListener("click", () => { provChosen = true; showProvider("openrouter"); });
  $("provOllama").addEventListener("click", () => { provChosen = true; showProvider("ollama"); loadOllama(); });
  $("orBack").addEventListener("click", async () => {
    try {
      const r = await postAction("/api/build_f/preset", "apply-preset", { provider: "openrouter" });
      toast("Back on OpenRouter: " + (r.applied_preset || "preset applied") + ".");
      loadModels();
    } catch (error) { toast(String(error.message || error), true); }
  });

  let ollamaInfo = null;
  function olModelLabel(m) {
    const bits = [m.parameter_size, m.quantization, m.size_gb ? m.size_gb + " GB" : ""].filter(Boolean);
    const caps = [m.thinking ? "thinking" : "", m.vision ? "vision" : ""].filter(Boolean);
    return m.name + (bits.length ? " · " + bits.join(" ") : "") + (caps.length ? " · " + caps.join(", ") : "");
  }
  async function loadOllama() {
    try {
      ollamaInfo = await getJSON("/api/build_f/ollama");
    } catch (error) {
      $("olStatus").textContent = "status unavailable: " + (error.message || error);
      return;
    }
    if (!$("olUrl").value.trim()) $("olUrl").value = ollamaInfo.base_url || "http://127.0.0.1:11434";
    if (ollamaInfo.api_key_set && !$("olKey").value) $("olKey").placeholder = "✓ key saved — type to replace";
    $("olDialect").value = ollamaInfo.dialect || "ollama";
    const n = (ollamaInfo.models || []).length;
    $("olStatus").textContent = ollamaInfo.reachable
      ? "Ollama " + (ollamaInfo.live_version || "") + " is running · " + (n ? n + " model(s) detected" : "no models detected yet — press Detect")
      : "not answering at " + ollamaInfo.base_url + " — start Ollama, then Detect" + (n ? " (" + n + " model(s) remembered)" : "");
    const select = $("olModel");
    const inUse = ollamaInfo.in_use || {};
    const current = inUse.deep || inUse.surface || "";
    select.replaceChildren(...(n ? (ollamaInfo.models || []).map((m) => el("option", { value: m.name, text: olModelLabel(m) }))
                                 : [el("option", { value: "", text: "— detect models first —" })]));
    if (current) select.value = current;
    $("olUse").disabled = !n;
    const roles = Object.keys(inUse);
    $("olNote").textContent = roles.length === 4 ? "All four roles run on " + (new Set(Object.values(inUse)).size === 1 ? current : "local models") + "."
      : roles.length ? "Local for: " + roles.join(", ") + " (the rest on the cloud)." : "";
  }
  $("olDetect").addEventListener("click", async () => {
    $("olDetect").disabled = true;
    $("olStatus").textContent = "asking Ollama…";
    try {
      const r = await postAction("/api/build_f/ollama/detect", "ollama-detect", { base_url: $("olUrl").value.trim(), api_key: $("olKey").value.trim(), dialect: $("olDialect").value });
      toast((r.models || []).length ? (r.models || []).length + " local model(s) detected — pick one and press Use this model." : "Ollama answered, but it has no models yet: ollama pull gemma3:27b (or another model of 27B or more)", !(r.models || []).length);
      await loadOllama();
      loadModels();   // the role grid now offers the local models too
    } catch (error) { $("olStatus").textContent = String(error.message || error); toast(String(error.message || error), true); }
    finally { $("olDetect").disabled = false; }
  });
  $("olUse").addEventListener("click", async () => {
    const model = $("olModel").value;
    if (!model) { toast("Detect models first.", true); return; }
    try {
      const r = await postAction("/api/build_f/ollama/use", "ollama-use", { model });
      toast("She now thinks locally on " + model + " — fully local." + (r.transport === "ollama" ? "" : " (" + r.transport + ")"));
      loadModels(); loadOllama();
    } catch (error) { toast(String(error.message || error), true); }
  });

  // ── OpenRouter API key ──
  async function loadKey() {
    try {
      const k = await getJSON("/api/build_f/openrouter_key");
      $("orKeyStatus").textContent = k.present ? "Saved ✓ " + (k.hint || "") : "No key saved yet — paste one to bring her online.";
    } catch (_) { $("orKeyStatus").textContent = "key status unavailable (is the brain running?)"; }
  }
  loadKey();
  $("orKeySave").addEventListener("click", async () => {
    const key = $("orKeyInput").value.trim();
    if (!key) { toast("Paste your OpenRouter key first (sk-or-…).", true); return; }
    try {
      const result = await postAction("/api/build_f/openrouter_key", "openrouter-key", { key });
      $("orKeyInput").value = "";
      toast(result.note || "Key saved.");
      loadKey();
    } catch (error) { toast(String(error.message || error), true); }
  });

  $("modelsSave").addEventListener("click", async () => {
    const selections = {};
    for (const role of ["surface", "deep", "reflection", "memory_write"]) {
      const select = $("role_" + role);
      if (select) selections[role] = select.value;
    }
    try {
      await postAction("/api/build_f/models", "select-models", { selections });
      toast("Model choices saved.");
      loadModels();
    } catch (error) { toast(String(error.message || error), true); }
  });
  $("presetApply").addEventListener("click", async () => {
    try {
      await postAction("/api/build_f/preset", "apply-preset", { preset: $("presetSelect").value });
      toast("Preset applied.");
      loadModels();
    } catch (error) { toast(String(error.message || error), true); }
  });

  // ───────────────────── voice adapter ─────────────────────
  let vmap = null;
  const FAMILY = (id) => id.replace(/\d+$/, "");

  async function loadVoiceMap() {
    vmap = await getJSON("/api/voice_map").catch(() => null);
    if (!vmap) { $("vmapGrid").replaceChildren(el("div", { class: "card-hint", text: "Voice map unavailable." })); return; }
    $("vmapEnabled").checked = !!vmap.enabled;
    $("vmapVoice").value = vmap.voice || "";
    $("vmapVoices").replaceChildren(...(vmap.v1_voices || []).map((v) => el("option", { value: v })));
    $("ttsCharShow").textContent = vmap.native_character || $("ttsCharShow").textContent;
    // group old ids by family for optgroups
    const families = new Map();
    for (const id of vmap.v1_ids || []) {
      const fam = FAMILY(id);
      if (!families.has(fam)) families.set(fam, []);
      families.get(fam).push(id);
    }
    const grid = $("vmapGrid");
    grid.replaceChildren();
    for (const [register, emotions] of Object.entries(vmap.v2_grouped || {})) {
      const cells = el("div", { class: "vmap-cells" });
      for (const emotion of emotions) {
        const key = register + "_" + emotion;
        const select = el("select", { "data-key": key });
        for (const [fam, ids] of families) {
          const group = el("optgroup", { label: fam });
          for (const id of ids) group.append(el("option", { value: id, text: id }));
          select.append(group);
        }
        select.value = (vmap.map && vmap.map[key]) || (vmap.defaults && vmap.defaults[key]) || "neutral1";
        const cell = el("div", { class: "vmap-cell" }, el("label", { text: key }), select);
        const markChanged = () => cell.classList.toggle("changed", select.value !== (vmap.defaults ? vmap.defaults[key] : select.value));
        select.addEventListener("change", markChanged);
        markChanged();
        cells.append(cell);
      }
      grid.append(el("div", { class: "vmap-register" }, el("h4", { text: register }), cells));
    }
  }
  loadVoiceMap();

  async function saveVoiceMap() {
    const map = {};
    document.querySelectorAll("#vmapGrid select[data-key]").forEach((select) => {
      map[select.dataset.key] = select.value;
    });
    try {
      const response = await fetch("/api/voice_map", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: $("vmapEnabled").checked, voice: $("vmapVoice").value.trim(), map }),
      });
      if (!response.ok) throw new Error("save failed (" + response.status + ")");
      toast($("vmapEnabled").checked
        ? "Voice adapter ON — speaking through " + ($("vmapVoice").value.trim() || "default") + "."
        : "Voice adapter OFF — raw 69-key tags pass through (needs a V2-native voice).");
      loadVoiceMap();
    } catch (error) { toast(String(error.message || error), true); }
  }
  $("vmapSave").addEventListener("click", saveVoiceMap);
  $("vmapReset").addEventListener("click", async () => {
    await fetch("/api/voice_map/reset", { method: "POST" }).catch(() => {});
    toast("Voice map reset to the shipped default (adapter ON, MekaHime_Amika, built-in map).");
    loadVoiceMap();
  });

  // ───────────────────── avatar ─────────────────────
  let avatars = [];
  async function loadAvatars() {
    const payload = await getJSON("/api/avatar/models").catch(() => ({ models: [] }));
    avatars = payload.models || [];
    $("avSimpleList").textContent = avatars.length
      ? "Installed models: " + avatars.map((m) => m.name).join(" · ")
      : "No Live2D model installed yet: envs\\brain\\Scripts\\python.exe scripts\\setup_tts.py installs hers together with her voice. Any other Cubism model works too: put its folder in avatar/models/ and reload this page.";
    const select = $("avModelSelect");
    select.replaceChildren(...(avatars.length
      ? avatars.map((m) => el("option", { value: m.id, text: m.name }))
      : [el("option", { value: "", text: "— no models found —" })]));
    if (avatars.length) loadAvatarConfig(avatars[0].id);
    else { $("avBanks").value = ""; $("avNote").textContent = "Add a model folder, then Rescan."; }
  }
  async function loadAvatarConfig(id) {
    const model = avatars.find((m) => m.id === id);
    if (!model) return;
    const doc = await getJSON(model.banks_url).catch(() => ({ config: {}, banks: {} }));
    $("avDisplayName").value = (doc.config && doc.config.display_name) || model.name || "";
    $("avMouthOpen").value = (doc.config && doc.config.mouth_open_param) || "";
    $("avMouthForm").value = (doc.config && doc.config.mouth_form_param) || "";
    $("avScale").value = (doc.config && doc.config.scale) || "";
    // show only non-empty bank entries to keep the editor readable
    const banks = {};
    for (const [key, entry] of Object.entries(doc.banks || {})) {
      if (entry && ((entry.emotes && entry.emotes.length) || (entry.movements && entry.movements.length))) banks[key] = entry;
    }
    $("avBanks").value = JSON.stringify({ banks }, null, 2);
    $("avNote").textContent = "Empty keys fall back to the motion-naming convention.";
  }
  $("avModelSelect").addEventListener("change", () => loadAvatarConfig($("avModelSelect").value));
  $("avReload").addEventListener("click", loadAvatars);
  loadAvatars();

  $("avSave").addEventListener("click", async () => {
    const id = $("avModelSelect").value;
    if (!id) { toast("No model selected.", true); return; }
    let parsed;
    try {
      parsed = $("avBanks").value.trim() ? JSON.parse($("avBanks").value) : { banks: {} };
    } catch (error) { toast("Banks JSON is invalid: " + error.message, true); return; }
    const config = {};
    if ($("avDisplayName").value.trim()) config.display_name = $("avDisplayName").value.trim();
    if ($("avMouthOpen").value.trim()) config.mouth_open_param = $("avMouthOpen").value.trim();
    if ($("avMouthForm").value.trim()) config.mouth_form_param = $("avMouthForm").value.trim();
    if ($("avScale").value) config.scale = Number($("avScale").value);
    try {
      const response = await fetch("/api/avatar/models/" + encodeURIComponent(id) + "/banks", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config, banks: parsed.banks || parsed }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "save failed");
      toast("Avatar config saved. The Use page picks it up on next load.");
    } catch (error) { toast(String(error.message || error), true); }
  });

  // ───────────────────── persona builder (fill-in-the-blank) ─────────────────────
  const STORY_LABELS = {
    "WHO YOU ARE": ["Who she is — where she comes from", "出身"],
    "YOUR FAMILY": ["Her family", "家庭"],
    "YOUR SCHOOL LIFE": ["Her school & daily life", "日常"],
    "YOUR DAYS HERE": ["Her days here — the new life", "日常"],
    "YOUR PATH": ["Her path so far", "经历"],
    "YOUR ROLE": ["Her role", "身份"],
    "YOUR PRIVATE SELF": ["Her private self — what nobody else sees", "私下"],
    "现在": ["Her life right now — and you in it", "现在"],
    "PERSONALITY": ["Personality traits (one per line, keep the dash)", "性格"],
  };
  const DAY_ORDER = ["清晨", "上午", "中午", "下午", "傍晚", "晚上", "深夜", "凌晨"];
  const DAY_EN = { "清晨": "early morning", "上午": "morning", "中午": "noon", "下午": "afternoon",
                   "傍晚": "evening", "晚上": "night", "深夜": "late night", "凌晨": "pre-dawn" };
  const ABOUT_FIELDS = [
    ["recent_mood", "Her mood these days", "心境"],
    ["current_action", "What's happening whenever you two talk", "此刻"],
    ["relationship", "Your relationship — her honest inner view", "关系"],
    ["surface_context", "Your relationship — the one-breath version (written as 你…)", "速记"],
    ["communication_style", "Language & call habits", "说话"],
  ];
  let pbDirty = { story: {}, days: {}, about: {} };

  function pbField(group, key, labelText, jp, text, rows) {
    const area = el("textarea", { rows: String(rows || 4), spellcheck: "false" });
    area.value = text || "";
    const wrap = el("div", { class: "pb-field" },
      el("label", {}, labelText, el("i", { text: jp || "" })), area);
    area.addEventListener("input", () => {
      pbDirty[group][key] = area.value;
      wrap.classList.add("dirty");
    });
    return wrap;
  }

  // one time slot of her day: five possibilities behind [1]…[5]; the brain lives one of them each day
  const SLOT_COUNT = 5;
  function pbSlot(key, period, options) {
    const list = Array.from({ length: SLOT_COUNT }, (_, i) => (typeof options[i] === "string" ? options[i] : ""));
    const saved = list.slice();
    let current = 0;
    const area = el("textarea", { rows: "3", spellcheck: "false" });
    const tabs = el("div", { class: "pb-slot-tabs", role: "tablist" });
    const buttons = list.map((_, i) => el("button", { type: "button", role: "tab", text: String(i + 1), title: "Possibility " + (i + 1) }));
    const wrap = el("div", { class: "pb-slot" },
      el("div", { class: "pb-slot-head" },
        el("label", {}, `What's her ${DAY_EN[period]} like? `, el("i", { text: period })), tabs),
      area,
      el("div", { class: "pb-slot-note", text: "Five possibilities — each day she lives one of them (20% each). Leave a tab empty to use fewer." }));
    const paint = () => {
      buttons.forEach((b, j) => {
        b.classList.toggle("active", j === current);
        b.classList.toggle("dirty", list[j] !== saved[j]);
        b.title = "Possibility " + (j + 1) + (list[j].trim() ? "" : " (empty)");
      });
    };
    const show = (i) => { current = i; area.value = list[i]; paint(); };
    buttons.forEach((b, i) => { b.addEventListener("click", () => show(i)); tabs.append(b); });
    area.addEventListener("input", () => {
      list[current] = area.value;
      pbDirty.days[key] = list.map((s) => s.trim()).filter(Boolean);
      wrap.classList.add("dirty");
      paint();
    });
    show(0);
    return wrap;
  }

  function pbGroup(title, open, children) {
    const details = el("details", { class: "oa-details" }, el("summary", { text: title }));
    if (open) details.setAttribute("open", "");
    const inner = el("div", { class: "inner" });
    for (const child of children) inner.append(child);
    details.append(inner);
    return details;
  }

  async function loadBuilder() {
    const box = $("personaBuilder");
    let data;
    try { data = await getJSON("/api/build_f/persona/simple"); }
    catch (error) {
      box.replaceChildren(el("p", { class: "card-hint", text: "Persona builder unavailable: " + (error.message || error) }));
      return;
    }
    pbDirty = { story: {}, days: {}, about: {} };
    const story = (data.story || []).map((s) => {
      const [label, jp] = STORY_LABELS[s.key] || [s.key, ""];
      return pbField("story", s.key, label, jp, s.text, Math.min(10, Math.max(3, Math.ceil((s.text || "").length / 90))));
    });
    const dayFields = (prefix, title) => {
      const fields = [];
      for (const period of DAY_ORDER) {
        const key = prefix + "/" + period;
        if (data.days && key in data.days) {
          const value = data.days[key];
          fields.push(Array.isArray(value)
            ? pbSlot(key, period, value)
            : pbField("days", key, `What's her ${DAY_EN[period]} like? `, period, value, 3));
        }
      }
      return el("div", { class: "pb-days" }, ...fields);
    };
    const about = ABOUT_FIELDS
      .filter(([key]) => data.about && data.about[key] !== undefined && data.about[key] !== "")
      .map(([key, label, jp]) => pbField("about", key, label, jp, data.about[key], 4));
    box.replaceChildren(
      pbGroup("Her story · 她是谁", true, story.length ? story : [el("p", { class: "card-hint", text: "No story sections found." })]),
      pbGroup("Her days — weekdays · 平日", false, [dayFields("weekday", "weekday")]),
      pbGroup("Her days — weekends · 周末", false, [dayFields("weekend", "weekend")]),
      pbGroup("You two · 你们之间", false, about),
      el("p", { class: "card-hint", style: "margin-top:8px", text:
        "{user_name} = your name wherever she'd say it · {city} = your city · {weekday} fills in automatically · she speaks whatever language you use with her." }),
    );
  }
  loadBuilder();

  $("pbSave").addEventListener("click", async () => {
    const payload = {};
    for (const group of ["story", "days", "about"]) {
      if (Object.keys(pbDirty[group]).length) payload[group] = pbDirty[group];
    }
    if (!Object.keys(payload).length) { toast("Nothing changed yet — edit a blank first."); return; }
    try {
      const result = await postAction("/api/build_f/persona/simple", "persona-simple", payload);
      const problems = result.problems || [];
      $("pbNote").textContent = problems.length
        ? "Saved with " + problems.length + " format warning(s): " + problems.slice(0, 2).join(" · ")
        : "Saved & live (" + (result.touched || []).join(", ") + ").";
      toast(problems.length ? "Saved — check the format warnings." : "Saved — she's already living it.", problems.length > 0);
      loadBuilder();
      loadPersona();
    } catch (error) { toast(String(error.message || error), true); }
  });

  // ───────────────────── personality (raw files · professional) ─────────────────────
  let persona = null;
  const edited = {};   // target -> edited content

  function showPersonaFile() {
    if (!persona) return;
    const target = $("pTarget").value;
    $("pEditor").value = edited[target] !== undefined ? edited[target] : (persona.files[target] || "");
    const checkMap = { deep: "例子.txt", surface: "表层.txt", thinking: "思考.txt" };
    const tag = $("pCheckTag");
    if (target === "scenes") { tag.hidden = true; return; }
    const problems = (persona.template_check && (persona.template_check[checkMap[target]] || persona.template_check[target])) || [];
    tag.hidden = false;
    tag.textContent = problems.length ? problems.length + " template warning(s)" : "template check OK";
    tag.style.color = problems.length ? "var(--warn)" : "";
    tag.title = problems.join("\n");
  }
  function fillTimezones(list, current) {
    // the brain's common zones + whatever is configured + this browser's own zone, so "mine" is one click
    let mine = "";
    try { mine = Intl.DateTimeFormat().resolvedOptions().timeZone || ""; } catch (_) { mine = ""; }
    const zones = Array.from(new Set([...(list || []), current, mine].filter(Boolean))).sort();
    const select = $("pTimezone");
    select.replaceChildren(...zones.map((z) => el("option", { value: z, text: z === mine ? z + " (this computer)" : z })));
    if (current) select.value = current;
  }
  function showClock(clock) {
    $("pClock").textContent = clock ? "Her clock now: " + clock.local + " " + (clock.abbrev || "") + " · " + clock.timezone : "";
  }
  async function loadPersona() {
    try {
      persona = await getJSON("/api/build_f/persona");
      $("pName").value = persona.persona_name || "";
      $("pUser").value = persona.user_name || "";
      $("pCity").value = persona.city || "";
      try {   // the pronunciation guide (config/word_replace.json, role user_name) when it is for this name
        const pron = await getJSON("/api/name_pronunciation");
        $("pSayAs").value = pron.name && pron.name.toLowerCase() === (persona.user_name || "").toLowerCase() ? (pron.say_as || "") : "";
      } catch (_) {}
      fillTimezones(persona.timezones, persona.timezone || "");
      showClock(persona.clock);
      showPersonaFile();
    } catch (error) {
      $("pNote").textContent = "Personality data unavailable: " + (error.message || error);
    }
  }
  $("pTarget").addEventListener("change", showPersonaFile);
  $("pEditor").addEventListener("input", () => { edited[$("pTarget").value] = $("pEditor").value; });
  $("pRevert").addEventListener("click", () => {
    delete edited[$("pTarget").value];
    showPersonaFile();
    toast("Edits discarded.");
  });
  $("pIdentitySave").addEventListener("click", async () => {
    try {
      const result = await postAction("/api/build_f/persona/identity", "persona-identity", {
        persona_name: $("pName").value.trim() || "Amika", user_name: $("pUser").value.trim() || "User",
        timezone: $("pTimezone").value, city: $("pCity").value.trim(),
      });
      $("pIdentityNote").textContent = result.note || "";
      showClock(result.clock);
      const name = $("pUser").value.trim();
      if (name) {
        try { await postAction("/api/name_pronunciation", "name-pronunciation", { name, say_as: $("pSayAs").value.trim() }); }
        catch (error) { toast("Pronunciation guide not saved: " + (error.message || error), true); }
      }
      toast("Saved — she calls you " + (result.user_name || "that") + (result.city ? " and lives in " + result.city : "") + ".");
    } catch (error) { toast(String(error.message || error), true); }
  });
  // her voice says your name (with the guide typed above, as-is): /api/tts_say → one <audio>
  let hearAudio = null;
  $("pHear").addEventListener("click", async () => {
    const name = $("pUser").value.trim(), sayAs = $("pSayAs").value.trim();
    if (!name) { toast("Type the name she should call you first.", true); return; }
    const spoken = sayAs || name;
    $("pHear").disabled = true;
    try {
      const r = await postAction("/api/tts_say", "tts-say", { text: `${spoken}. Hi, ${spoken}! Nice to meet you.`, emotion: "intimate_caring", voice: "happy1", raw: !!sayAs });
      if (hearAudio) { try { hearAudio.pause(); } catch (_) {} }
      const mime = { opus: "audio/ogg", ogg: "audio/ogg", mp3: "audio/mpeg", wav: "audio/wav" }[r.audio_format] || "audio/wav";
      hearAudio = new Audio("data:" + mime + ";base64," + r.audio);
      await hearAudio.play();
    } catch (error) { toast("Her voice is unavailable: " + (error.message || error), true); }
    finally { $("pHear").disabled = false; }
  });

  $("pSave").addEventListener("click", async () => {
    const target = $("pTarget").value;
    try {
      const result = await postAction("/api/build_f/persona/save", "persona-save", {
        target, content: $("pEditor").value,
      });
      delete edited[target];
      const problems = result.problems || [];
      $("pNote").textContent = problems.length
        ? "Saved with " + problems.length + " template warning(s): " + problems.slice(0, 3).join(" · ")
        : (result.note || "Saved and hot-reloaded.");
      toast(problems.length ? "Saved — check the template warnings." : "Saved & live.", problems.length > 0);
      loadPersona();
    } catch (error) { toast(String(error.message || error), true); }
  });
  loadPersona();

  // ───────────────────── text her (Discord / Telegram bots) ─────────────────────
  const MSG = { discord: "Discord", telegram: "Telegram" };
  function msgRows(status) {
    const rows = $("msgRows");
    rows.replaceChildren();
    const add = (state, name, meta) => rows.append(
      el("div", { class: "health-row" },
        el("span", { class: "dot " + state }),
        el("span", { class: "name", text: name }),
        el("span", { class: "meta", text: meta })));
    add(status.brain ? "ok" : "err", "Brain link", status.brain ? "linked — her text turns run on the same brain as the stage" : "not linked to the brain");
    for (const p of Object.keys(MSG)) {
      const cfg = status[p] || {}, live = (status.live || {})[p] || {};
      if (!cfg.token_present) { add("", MSG[p], p === "discord" && status.discord_library === false ? "discord.py is not installed in the speech python" : "no token saved"); continue; }
      if (!cfg.enabled) { add("", MSG[p], "token saved · switched off"); continue; }
      if (live.error) { add("err", MSG[p], live.error); continue; }
      if (!live.connected) { add("warn", MSG[p], "connecting…"); continue; }
      const who = (cfg.allowed_users || []).length;
      add(who ? "ok" : "warn", MSG[p], "online as " + (live.name || "bot") +
        (who ? " · " + who + " allowed user" + (who > 1 ? "s" : "") : " · nobody is allowed yet — add your user id"));
    }
    if (status.last_chat && status.last_chat.label) add("ok", "Her last chat", status.last_chat.label + " — where she goes when she speaks first");
  }
  function msgFill(status) {
    for (const p of Object.keys(MSG)) {
      const cfg = status[p] || {}, cap = p[0].toUpperCase() + p.slice(1);
      $("msg" + cap + "Hint").textContent = cfg.token_present ? "(saved " + (cfg.token_hint || "") + ")" : "";
      $("msg" + cap + "Users").value = (cfg.allowed_users || []).join(", ");
      $("msg" + cap + "On").checked = !!cfg.enabled;
      $("msg" + cap + "Clear").hidden = !cfg.token_present;
    }
  }
  async function loadMessengers(fill) {
    try {
      const status = await getJSON("/api/build_f/messengers");
      msgRows(status);
      if (fill) msgFill(status);
    } catch (error) {
      $("msgRows").replaceChildren(el("div", { class: "card-hint", text: "Messengers service unavailable: " + (error.message || error) }));
    }
  }
  async function msgSave(body, done) {
    try {
      const result = await postAction("/api/build_f/messengers", "messengers", body);
      $("msgDiscordToken").value = ""; $("msgTelegramToken").value = "";
      msgRows(result); msgFill(result);
      const bad = Object.keys(MSG).filter((p) => (result.live || {})[p] && result.live[p].error);
      $("msgNote").textContent = bad.length ? bad.map((p) => MSG[p] + ": " + result.live[p].error).join(" · ") : (done || "");
      toast(bad.length ? "Saved — but a bot could not connect, see the row above." : (done || "Saved."), bad.length > 0);
    } catch (error) { toast(String(error.message || error), true); }
  }
  $("msgSave").addEventListener("click", () => msgSave({
    discord: { token: $("msgDiscordToken").value.trim(), allowed_users: $("msgDiscordUsers").value, enabled: $("msgDiscordOn").checked },
    telegram: { token: $("msgTelegramToken").value.trim(), allowed_users: $("msgTelegramUsers").value, enabled: $("msgTelegramOn").checked },
  }, "Saved — bots are (re)connecting."));
  $("msgDiscordClear").addEventListener("click", () => msgSave({ discord: { clear_token: true } }, "Discord token forgotten."));
  $("msgTelegramClear").addEventListener("click", () => msgSave({ telegram: { clear_token: true } }, "Telegram token forgotten."));
  loadMessengers(true);
  setInterval(() => loadMessengers(false), 10000);
})();
