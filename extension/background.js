// Background service worker for Browser Agent.
// Owns the connection to the native messaging host (OS keychain bridge) and
// opens the side panel when the user clicks the extension icon.

const NATIVE_HOST_NAME = "com.browseragent.native_host";
const KEYRING_SERVICE = "browser-agent";

// Open the side panel when the user clicks the extension action.
chrome.action.onClicked.addListener(async (tab) => {
  try { await chrome.sidePanel.open({ tabId: tab.id }); } catch { /* tab closed */ }
});

// Configure the side panel to open on action click.
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

// Monotonic counter for tagging outgoing native-host calls. Each call gets a
// unique id; the host echoes it back in its response, and we use it to match
// responses to the promise that owns them on the shared port.
let nextNativeId = 1;

// Create a fresh native port for each call. We previously cached the port
// to avoid the cost of reconnecting, but chrome.runtime.connectNative is
// near-instant (it reuses the underlying host process) and a cached port
// introduced TOCTOU races and leaked onDisconnect listeners across calls.
function connectNative() {
  return chrome.runtime.connectNative(NATIVE_HOST_NAME);
}

function sendToNative(message, timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    let port;
    try {
      port = connectNative();
    } catch (err) {
      reject(new Error(`native host unavailable: ${err.message}`));
      return;
    }
    const id = nextNativeId++;
    // `settled` prevents double-resolve/reject when the timeout, response,
    // and disconnect handlers race against each other.
    let settled = false;
    const cleanup = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { port.disconnect(); } catch { /* already disconnected */ }
    };
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      port.onMessage.removeListener(handler);
      port.onDisconnect.removeListener(disconnectHandler);
      reject(new Error("native host timed out"));
    }, timeoutMs);
    const handler = (response) => {
      // Ignore messages that belong to other outstanding calls on this port.
      if (!response || response._id !== id) return;
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      port.onMessage.removeListener(handler);
      port.onDisconnect.removeListener(disconnectHandler);
      resolve(response);
    };
    const disconnectHandler = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      port.onMessage.removeListener(handler);
      try { port.disconnect(); } catch { /* already disconnected */ }
      reject(new Error("native host disconnected"));
    };
    port.onMessage.addListener(handler);
    port.onDisconnect.addListener(disconnectHandler);
    try {
      port.postMessage({ ...message, _id: id });
    } catch (err) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      port.onMessage.removeListener(handler);
      port.onDisconnect.removeListener(disconnectHandler);
      reject(new Error(`native host postMessage failed: ${err.message}`));
    }
  });
}

// Generic message router for the side panel.
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.kind === "native") {
    sendToNative(msg.payload)
      .then((response) => sendResponse({ ok: true, response }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true; // keep the message channel open for async response
  }
  return false;
});
