// Background service worker for Browser Agent.
// Opens the side panel when the user clicks the extension icon.
// Also connects to the native messaging host on startup to trigger
// the auto-launch of the local API server (if it isn't already running).

// Open the side panel when the user clicks the extension action.
chrome.action.onClicked.addListener(async (tab) => {
  try { await chrome.sidePanel.open({ tabId: tab.id }); } catch { /* tab closed */ }
});

// Configure the side panel to open on action click.
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

// Connect to the native host on startup. This triggers Chrome to
// launch native_host.py, which checks if the API server is running
// on 127.0.0.1:8000 and spawns it if not. The connection is
// immediately closed — we don't need to send or receive messages;
// the side panel communicates with the API server directly via HTTP.
let nativePort = null;
try {
  nativePort = chrome.runtime.connectNative("com.browseragent.native_host");
  nativePort.disconnect();
} catch {
  // Native host not installed — the side panel will show a helpful
  // error message when it can't reach the API server.
}
