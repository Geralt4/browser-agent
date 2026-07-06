// Side panel logic for Browser Agent.
// - Tabs: Chat + Settings
// - Settings: API key via native host (OS keychain) — plaintext fallback removed
// - Chat: task submission + SSE streaming + gate modal
// - Vision nudge: shown when the selected model is not in the vision models list

const API_BASE = "http://127.0.0.1:8000";
const KEYRING_SERVICE = "browser-agent";

// State
let currentTaskId = null;
let eventSource = null;
let currentGateId = null;
let savedConfig = null; // last-loaded config from storage
let taskRunning = false; // re-entrancy guard for sendTask

// ---------------- DOM helpers ----------------
const $ = (id) => document.getElementById(id);
function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ---------------- tabs ----------------
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    document.querySelectorAll(".tab").forEach((b) => {
      b.classList.toggle("active", b === btn);
      b.setAttribute("aria-selected", b === btn ? "true" : "false");
    });
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.toggle("active", p.id === `tab-${target}`));
  });
});

// Escape key dismisses the gate modal (denies the action).
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && $("gate-overlay").classList.contains("open")) {
    resolveGate(false);
  }
});

// ---------------- keychain bridge (API-based, primary) ----------------
// Primary path: proxy keychain operations through the local API server.
// This works on Brave (which blocks native messaging for unpacked
// extensions even when allowed_origins matches and the
// NativeMessagingAllowlist policy is set) and on Chrome alike. The
// native host is retained as a fallback for environments where the
// API server isn't reachable.

let keychainAvailable = null; // null = unchecked, true/false after ping

async function keychainCall(cmd, payload) {
  const r = await fetch(`${API_BASE}/api/keychain/${cmd}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error((data && data.error) || `HTTP ${r.status}`);
  }
  return await r.json();
}

async function checkKeychain() {
  try {
    const resp = await keychainCall("ping");
    keychainAvailable = resp && resp.ok === true;
  } catch {
    keychainAvailable = false;
  }
  return keychainAvailable;
}

// ---------------- storage ----------------
// Chrome storage callbacks should check chrome.runtime.lastError; otherwise
// a quota-exceeded or corrupted-store failure is silently swallowed.
function _checkStorageError() {
  if (chrome.runtime.lastError) {
    throw new Error(chrome.runtime.lastError.message);
  }
}

async function loadSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(
      ["provider", "baseUrl", "model", "visionMode", "visionModels", "authToken"],
      (sync) => {
        try { _checkStorageError(); } catch (e) { resolve({ provider: "openai", baseUrl: "", model: "", visionMode: "vision", visionModels: "", apiKey: "", usingLocalFallback: false, authToken: "" }); return; }
        // apiKey is no longer stored in chrome.storage.local (the plaintext
        // fallback was removed). It's loaded from the OS keychain in init()
        // via loadApiKey(). usingLocalFallback is always false now.
        resolve({
          provider: sync.provider || "openai",
          baseUrl: sync.baseUrl || "",
          model: sync.model || "",
          visionMode: sync.visionMode || "vision",
          visionModels: sync.visionModels || "",
          apiKey: "",
          usingLocalFallback: false,
          authToken: sync.authToken || "",
        });
      }
    );
  });
}

async function saveSettings(s) {
  // `apiKey` and `usingLocalFallback` in chrome.storage.local are owned by
  // storeApiKey (which writes to the OS keychain via the native host, or
  // falls back to local storage). Writing them here would clobber the key
  // storeApiKey just wrote — see Phase 1.5 fix for issue #7.
  await new Promise((resolve, reject) =>
    chrome.storage.sync.set(
      {
        provider: s.provider,
        baseUrl: s.baseUrl,
        model: s.model,
        visionMode: s.visionMode,
        visionModels: s.visionModels,
        authToken: s.authToken || "",
      },
      () => {
        try { _checkStorageError(); resolve(); } catch (e) { reject(e); }
      }
    )
  );
}

async function storeApiKey(key) {
  if (!key) {
    if (keychainAvailable) {
      try { await keychainCall("delete", { service: KEYRING_SERVICE, key: "llm_api_key" }); }
      catch {}
    }
    await new Promise((r) => chrome.storage.local.remove("apiKey", () => { try { _checkStorageError(); } catch {} r(); }));
    return;
  }
  if (keychainAvailable) {
    try {
      await keychainCall("set", { service: KEYRING_SERVICE, key: "llm_api_key", value: key });
      await new Promise((r) => chrome.storage.local.remove(["apiKey", "usingLocalFallback"], () => { try { _checkStorageError(); } catch {} r(); }));
      return;
    } catch {
      // fall through to error
    }
  }
  // No keychain bridge available — refuse to store the key unencrypted.
  // chrome.storage.local is plaintext on disk and readable by any process
  // with filesystem access. Show a hard error so the user knows the key
  // was NOT saved, rather than silently storing it insecurely.
  throw new Error(
    "Cannot save API key: no keychain bridge available. " +
    "Start the browser-agent server, or install the native messaging host " +
    "for OS keychain storage. The key was NOT saved to chrome.storage.local " +
    "(refused: plaintext storage is disabled)."
  );
}

async function loadApiKey() {
  if (keychainAvailable) {
    try {
      const resp = await keychainCall("get", { service: KEYRING_SERVICE, key: "llm_api_key" });
      if (resp && resp.ok && resp.value) return { key: resp.value, fromLocal: false };
    } catch {}
  }
  // No keychain bridge available — do NOT fall back to chrome.storage.local.
  // Reading plaintext storage would return a key that was stored insecurely
  // (by the old fallback path, now removed). Return empty so the user is
  // prompted to re-enter the key into the secure keychain bridge.
  return { key: "", fromLocal: false };
}

// ---------------- settings form ----------------
function fillSettings(s) {
  $("provider").value = s.provider;
  $("base-url").value = s.baseUrl;
  // The model <select> starts disabled with only a placeholder option, so
  // setting .value on it would silently no-op. Inject the saved model as a
  // real option instead, so it's visible (and re-selectable) on reload.
  if (s.model) {
    const sel = $("model");
    sel.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = s.model;
    opt.textContent = s.model;
    opt.selected = true;
    sel.appendChild(opt);
    sel.disabled = false;
  }
  $("vision-mode").value = s.visionMode;
  $("vision-models").value = s.visionModels;
  $("api-key").value = s.apiKey || ""; // pre-fill if stored locally
  $("auth-token").value = s.authToken || "";
  updateVisionNudge();
}

function readSettings() {
  return {
    provider: $("provider").value,
    baseUrl: $("base-url").value.trim(),
    model: $("model").value.trim(),
    visionMode: $("vision-mode").value,
    visionModels: $("vision-models").value.trim(),
    apiKey: $("api-key").value.trim(),
    authToken: $("auth-token").value.trim(),
  };
}

function updateVisionNudge() {
  const model = $("model").value.trim().toLowerCase();
  const visionModels = $("vision-models").value.toLowerCase()
    .split(",").map((s) => s.trim()).filter(Boolean);
  const nudge = $("vision-nudge");
  if (model && visionModels.length > 0 && !visionModels.includes(model)) {
    nudge.classList.remove("hidden");
  } else {
    nudge.classList.add("hidden");
  }
}

$("model").addEventListener("change", updateVisionNudge);
$("vision-models").addEventListener("input", updateVisionNudge);

$("toggle-key").addEventListener("click", () => {
  const inp = $("api-key");
  if (inp.type === "password") { inp.type = "text"; $("toggle-key").textContent = "Hide"; }
  else                          { inp.type = "password"; $("toggle-key").textContent = "Show"; }
});

$("fetch-models").addEventListener("click", async () => {
  const s = readSettings();
  if (!s.baseUrl) { setStatus("models-status", "Set API Base URL first.", "error"); return; }
  if (!s.apiKey)  { setStatus("models-status", "Set API Key first.", "error"); return; }
  setStatus("models-status", "Fetching…");
  $("fetch-models").disabled = true;
  try {
    const params = new URLSearchParams({ base_url: s.baseUrl });
    const r = await fetch(`${API_BASE}/api/models?${params}`, {
      headers: { "X-API-Key": s.apiKey },
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    populateModels(data.models || [], s.model);
    setStatus("models-status", `Loaded ${(data.models || []).length} models.`, "ok");
  } catch (e) {
    setStatus("models-status", `Failed: ${e.message}`, "error");
  } finally {
    $("fetch-models").disabled = false;
  }
});

function populateModels(models, selected) {
  const sel = $("model");
  sel.innerHTML = "";
  if (!models.length) {
    sel.innerHTML = '<option value="">— no models found —</option>';
    sel.disabled = true;
    return;
  }
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m; opt.textContent = m;
    if (m === selected) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.disabled = false;
  updateVisionNudge();
}

$("settings-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const s = readSettings();
  const status = $("save-status");
  status.textContent = "Saving…";
  status.className = "save-status";
  try {
    await storeApiKey(s.apiKey);
    const persist = { ...s, apiKey: "" };
    await saveSettings(persist);
    savedConfig = { ...persist, apiKey: s.apiKey };
    status.textContent = "Saved.";
    status.className = "save-status ok";
  } catch (err) {
    status.textContent = `Save failed: ${err.message}`;
    status.className = "save-status error";
  }
  setTimeout(() => { status.textContent = ""; status.className = "save-status"; }, 2500);
});

function setStatus(id, text, cls) {
  const el = $(id);
  el.textContent = text;
  el.className = "hint " + (cls || "");
}

// ---------------- chat tab ----------------
function appendMsg(cls, html) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.innerHTML = html;
  $("stream").appendChild(div);
  $("stream").scrollTop = $("stream").scrollHeight;
  return div;
}

$("send-btn").addEventListener("click", sendTask);
$("task-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendTask(); }
});

async function sendTask() {
  if (taskRunning) return; // guard against re-entrant calls
  const task = $("task-input").value.trim();
  if (!task) return;
  if (!savedConfig || !savedConfig.model) {
    appendMsg("msg--error", "Pick a model in Settings first.");
    document.querySelector('.tab[data-tab="settings"]').click();
    return;
  }
  if (!savedConfig.apiKey) {
    appendMsg("msg--error", "Set an API key in Settings first.");
    document.querySelector('.tab[data-tab="settings"]').click();
    return;
  }

  taskRunning = true;
  $("task-input").disabled = true;
  $("send-btn").disabled = true;
  $("stream").innerHTML = "";
  appendMsg("msg--system", 'Submitting task… <span class="spinner"></span>');
  if (eventSource) {
    // Null out handlers before close so stale onerror/onmessage callbacks
    // can't fire asynchronously and corrupt the new EventSource (they read
    // the module-level `eventSource` variable, not the specific instance).
    eventSource.onerror = null;
    eventSource.onmessage = null;
    eventSource.close();
  }

  try {
    const r = await fetch(`${API_BASE}/api/task`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": savedConfig.apiKey,
        ...(savedConfig.authToken ? { "X-Auth-Token": savedConfig.authToken } : {}),
      },
      body: JSON.stringify({
        task,
        provider: savedConfig.provider,
        base_url: savedConfig.baseUrl || null,
        model: savedConfig.model,
        vision_mode: savedConfig.visionMode,
        vision_models: savedConfig.visionModels || null,
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    currentTaskId = data.task_id;
    openStream();
  } catch (e) {
    appendMsg("msg--error", `Failed: ${esc(e.message)}`);
    resetChatUI();
  }
}

function openStream() {
  if (!currentTaskId) return;
  eventSource = new EventSource(`${API_BASE}/api/task/${currentTaskId}/stream`);
  eventSource.onmessage = (e) => {
    // A real event arrived — if we'd shown a "reconnecting" banner, the
    // stream is back. Clear the flag and let the event render normally.
    if (eventSource && eventSource._reconnectingShown) {
      eventSource._reconnectingShown = false;
    }
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    handleStreamEvent(msg);
  };
  eventSource.onerror = () => {
    // The server closes the stream only after sending done/error, and the
    // done/error handlers call resetChatUI() which closes the EventSource —
    // so by the time onerror fires here, the stream is truly gone (or the
    // browser is mid-reconnect). Distinguish the two via readyState:
    //   CONNECTING (0) = EventSource is auto-retrying; don't kill the view.
    //   CLOSED      (2) = the connection is gone; show "Connection lost"
    //                    and reset (no-op if resetChatUI already ran).
    if (!eventSource) return; // already cleaned up by done/error handler
    if (eventSource.readyState === EventSource.CONNECTING) {
      if (!eventSource._reconnectingShown) {
        eventSource._reconnectingShown = true;
        appendMsg("msg--system", "Reconnecting…");
      }
      return;
    }
    appendMsg("msg--error", "Connection lost");
    resetChatUI();
  };
}

function handleStreamEvent(msg) {
  if (msg.type === "start") {
    $("stream").innerHTML = "";
    appendMsg("msg--system", esc(msg.message));
  } else if (msg.type === "system") {
    appendMsg("msg--nudge", esc(msg.message));
  } else if (msg.type === "step") {
    let html = '<div class="step-header"><span class="step-badge">Step ' + esc(msg.step_n) + '</span>';
    if (msg.next_subgoal) html += '<span class="step-label">Next: ' + esc(msg.next_subgoal) + '</span>';
    html += '</div>';
    if (msg.assessment) html += '<div class="step-field"><strong>Assessment:</strong> ' + esc(msg.assessment) + '</div>';
    if (msg.memory)     html += '<div class="step-field"><strong>Memory:</strong> ' + esc(msg.memory) + '</div>';
    if (msg.action)     html += '<div class="step-field"><strong>Action:</strong> <code>' + esc(msg.action) + '</code></div>';
    appendMsg("msg--step", html);
  } else if (msg.type === "gate") {
    currentGateId = msg.gate_id;
    let details = '<dt>Action</dt><dd>' + esc(msg.name) + '</dd>';
    if (msg.summary) details += '<dt>Details</dt><dd>' + esc(msg.summary) + '</dd>';
    for (const k in msg.params) {
      if (k !== "index" && k !== "new_tab") details += '<dt>' + esc(k) + '</dt><dd>' + esc(String(msg.params[k])) + '</dd>';
    }
    $("gate-detail").innerHTML = details;
    $("gate-overlay").classList.add("open");
  } else if (msg.type === "done") {
    appendMsg("msg--done", "<strong>Done:</strong> " + esc(msg.result));
    resetChatUI();
  } else if (msg.type === "error") {
    appendMsg("msg--error", "Error: " + esc(msg.message));
    resetChatUI();
  }
}

function resetChatUI() {
  taskRunning = false;
  $("task-input").disabled = false;
  $("send-btn").disabled = false;
  $("task-input").focus();
  if (eventSource) { eventSource.close(); eventSource = null; }
  currentTaskId = null;
}

$("gate-approve").addEventListener("click", () => resolveGate(true));
$("gate-deny").addEventListener("click", () => resolveGate(false));
async function resolveGate(approved) {
  if (!currentGateId) return;
  const approveBtn = $("gate-approve");
  const denyBtn = $("gate-deny");
  approveBtn.disabled = true;
  denyBtn.disabled = true;
  try {
    const r = await fetch(`${API_BASE}/api/gate/${approved ? "approve" : "deny"}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gate_id: currentGateId }),
    });
    if (r.status === 404) {
      // Gate already expired server-side (300s timeout, see safety/gate.py).
      // The gate is gone either way, so close the overlay but tell the user
      // why nothing happened.
      appendMsg("msg--error", "This gate has expired — the task may have timed out.");
    }
    $("gate-overlay").classList.remove("open");
    currentGateId = null;
  } catch (e) {
    // Network error: keep the overlay open so the user can retry.
    appendMsg("msg--error", `Could not reach server: ${esc(e.message)}`);
  } finally {
    approveBtn.disabled = false;
    denyBtn.disabled = false;
  }
}

// ---------------- init ----------------
async function init() {
  // The API-based keychain bridge is the sole keychain path. It works on
  // Brave and Chrome alike — no native messaging required.
  await checkKeychain();
  savedConfig = await loadSettings();

  // If the saved config has no model, try to pre-fill the API key from the
  // keychain bridge (the user may have entered it before but it's only in
  // the OS keychain, not in chrome.storage.local).
  if (!savedConfig.apiKey) {
    const k = await loadApiKey();
    if (k.key) {
      savedConfig.apiKey = k.key;
      $("api-key").value = k.key;
    }
  }

  fillSettings(savedConfig);

  if (!keychainAvailable) {
    $("storage-warning").classList.remove("hidden");
    $("api-key-hint").textContent =
      "No keychain bridge available. Start the browser-agent server. API keys cannot be saved without OS keychain storage.";
  } else {
    $("api-key-hint").textContent =
      "Stored in your OS keychain via the local API server.";
  }

  // Probe backend reachability
  try {
    const r = await fetch(`${API_BASE}/api/config`, { method: "GET" });
    if (!r.ok) throw new Error();
  } catch {
    $("connection-warning").classList.remove("hidden");
  }
}

// init() is async; fire it from an IIFE so the top-level await happens
// without blocking the script's synchronous startup. Errors are non-fatal
// — the side panel still loads; sendTask guards on savedConfig being null.
(async () => {
  try { await init(); } catch { /* init failure is non-fatal */ }
})();
