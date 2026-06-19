// Background service worker for Browser Agent.
// Owns the connection to the native messaging host (OS keychain bridge) and
// opens the side panel when the user clicks the extension icon.

const NATIVE_HOST_NAME = "com.browseragent.native_host";
const KEYRING_SERVICE = "browser-agent";

// Open the side panel when the user clicks the extension action.
chrome.action.onClicked.addListener(async (tab) => {
  await chrome.sidePanel.open({ tabId: tab.id });
});

// Configure the side panel to open on action click.
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

// In-memory cache of the native host connection (sockets are expensive to
// create and are short-lived on the OS side; we keep one warm).
let nativePort = null;

function connectNative() {
  if (nativePort) return nativePort;
  nativePort = chrome.runtime.connectNative(NATIVE_HOST_NAME);
  nativePort.onDisconnect.addListener(() => {
    nativePort = null;
  });
  return nativePort;
}

function sendToNative(message) {
  return new Promise((resolve, reject) => {
    let port;
    try {
      port = connectNative();
    } catch (err) {
      reject(new Error(`native host unavailable: ${err.message}`));
      return;
    }
    const handler = (response) => {
      port.onMessage.removeListener(handler);
      port.onDisconnect.removeListener(disconnectHandler);
      resolve(response);
    };
    const disconnectHandler = () => {
      port.onMessage.removeListener(handler);
      nativePort = null;
      reject(new Error("native host disconnected"));
    };
    port.onMessage.addListener(handler);
    port.onDisconnect.addListener(disconnectHandler);
    port.postMessage(message);
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
