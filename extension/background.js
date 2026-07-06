// Background service worker for Browser Agent.
// Opens the side panel when the user clicks the extension icon.

// Open the side panel when the user clicks the extension action.
chrome.action.onClicked.addListener(async (tab) => {
  try { await chrome.sidePanel.open({ tabId: tab.id }); } catch { /* tab closed */ }
});

// Configure the side panel to open on action click.
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
