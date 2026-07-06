# Privacy Policy for Browser Agent

**Last updated: July 6, 2026**

## Overview

Browser Agent is a browser automation tool that runs entirely on your machine. This privacy policy explains what data the extension and its companion server handle, where it's stored, and who has access to it.

## Data We Collect

**We collect nothing.** Browser Agent does not:

- Send any data to us, the developers
- Use analytics, telemetry, or crash reporting
- Track your browsing activity
- Include any third-party SDKs or services
- Communicate with any remote server other than the LLM provider you configure

## What Data the Extension Handles

The Browser Agent Chrome extension handles the following data, all of which stays on your device:

### API Key
- **What**: The API key for your LLM provider (OpenAI, Moonshot, or any OpenAI-compatible endpoint).
- **Where it's stored**: Your operating system's encrypted keychain (macOS Keychain, Windows Credential Manager, or Linux Secret Service). The extension communicates with a local API server running on your machine (`127.0.0.1:8000`) which writes the key to the OS keychain. The key is never stored in browser local storage, never written to disk in plaintext, and never transmitted off your machine except to the LLM provider you configure.
- **How it's used**: The key is sent to your configured LLM provider's API endpoint to authenticate requests for browser automation tasks. It is sent directly from your machine to the provider — it never passes through any intermediary server.

### Settings
- **What**: Your provider selection, model name, base URL, vision preferences, and an optional server auth token.
- **Where it's stored**: Chrome's `storage.sync` (synced across your Chrome instances if you have Chrome Sync enabled). These are non-sensitive configuration values.
- **How it's used**: To configure the browser automation agent with your preferred LLM provider and model.

### Task Descriptions
- **What**: The natural-language tasks you type into the extension (e.g., "Go to example.com and return the H1 heading").
- **Where it's stored**: Transmitted to the local server (`127.0.0.1:8000`) and held in memory only for the duration of the task. Not persisted to disk.
- **How it's used**: Sent to your LLM provider as part of the browser automation prompt. The LLM provider's own privacy policy applies to data you send them.

### Browser Automation Data
- **What**: The extension's companion server runs a headless browser (via Playwright) that navigates to websites, reads page content, and performs actions on your behalf. Page content (DOM) is sent to your LLM provider as context for the automation agent.
- **Where it's stored**: In memory only, for the duration of the task. Not persisted.
- **How it's used**: To complete the automation task you requested.

## Data Sharing

Browser Agent shares data with exactly two destinations, both under your control:

1. **Your LLM provider** — The API key, task description, and page content are sent to the provider you configure (e.g., OpenAI, Moonshot). That provider's privacy policy governs how they handle this data.

2. **Your local server** — All data flows through the companion server running on `127.0.0.1:8000` on your machine. This server is part of Browser Agent and is bound to localhost only — it cannot be reached from other machines on your network.

No data is shared with the Browser Agent developers or any other third party.

## Data Retention

- **API key**: Stored in your OS keychain until you delete it (via the extension's Settings tab or your OS keychain manager).
- **Settings**: Stored in Chrome Sync until you clear them.
- **Task data**: Held in memory only during task execution. Discarded when the task completes or the server stops.

## Your Rights

Since all data stays on your machine, you have full control:

- **Delete your API key** at any time by clearing the API Key field in Settings and clicking Save, or by opening your OS keychain manager and removing the "browser-agent" entry.
- **Clear settings** by resetting the extension's stored data in Chrome's extension settings.
- **Stop the server** to prevent any further data processing.

## Security

- API keys are stored in your operating system's encrypted credential store, not in plaintext browser storage.
- The companion server binds to `127.0.0.1` (localhost) only — it is not accessible from your local network or the internet.
- The server supports an optional authentication token (`BROWSER_AGENT_API_TOKEN`) to prevent unauthorized local processes from submitting tasks.
- Page content from websites is sanitized to remove hidden elements and instruction-injection patterns before being sent to the LLM.

## Changes to This Policy

We may update this privacy policy from time to time. Changes will be posted in the project repository and reflected in the "Last updated" date.

## Contact

This project is open source. For questions about this privacy policy or the extension's data handling, open an issue on the GitHub repository: https://github.com/Geralt4/browser-agent
