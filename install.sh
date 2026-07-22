#!/usr/bin/env bash
# Install Browser Agent: project deps, background server, native host, extension setup.
#
# Usage:
#   ./install.sh                    # Step-by-step (prompts for extension ID)
#   ./install.sh <extension-id>     # Non-interactive
#
# What it does:
#   1. Installs Python dependencies (uv sync)
#   2. Installs Playwright browser binaries
#   3. Registers macOS LaunchAgent (or Linux systemd user unit) so the
#      API server auto-starts on login
#   4. Starts the API server immediately
#   5. Registers the native messaging host for OS keychain access
#   6. Registers the Brave policy plist (macOS + BRAVE_INSTALL_POLICY=1)
#   7. Prints instructions to load the unpacked extension in Chrome/Brave

set -euo pipefail

# ── Repo root ──────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTENSION_DIR="${REPO_DIR}/extension"
NATIVE_HOST_DIR="${EXTENSION_DIR}/native_host"

# ── Colors ─────────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}==>${NC} ${BOLD}$*${NC}"; }
warn()  { echo -e "${YELLOW}==>${NC} ${BOLD}$*${NC}"; }
err()   { echo -e "${RED}==>${NC} ${BOLD}$*${NC}" >&2; }

# ── Pre-flight checks ──────────────────────────────────────────────────

if ! command -v uv &>/dev/null; then
  err "uv is not installed. Install it first:"
  err "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if ! command -v node &>/dev/null; then
  warn "node is not in PATH — Playwright browser install may fail."
  warn "Install Node.js from https://nodejs.org/"
fi

# ── 1. Install project deps ────────────────────────────────────────────

info "Installing Python dependencies …"
cd "${REPO_DIR}"
uv sync

# ── 2. Install Playwright browsers ────────────────────────────────────

info "Installing Playwright browsers …"
uv run playwright install --with-deps chromium 2>/dev/null || true

# ── 3. Register background service ─────────────────────────────────────

PLIST_LABEL="com.browseragent.server"
PLIST_DEST=""
SERVICE_STARTED=false

case "$(uname -s)" in
  Darwin)
    PLIST_DEST="${HOME}/Library/LaunchAgents/${PLIST_LABEL}.plist"
    info "Installing LaunchAgent at ${PLIST_DEST} …"

    mkdir -p "${HOME}/Library/LaunchAgents"
    cat > "${PLIST_DEST}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${PLIST_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${REPO_DIR}/.venv/bin/uv</string>
    <string>run</string>
    <string>browser-agent-ui</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${REPO_DIR}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${HOME}/Library/Logs/browser-agent-ui.log</string>
  <key>StandardErrorPath</key>
  <string>${HOME}/Library/Logs/browser-agent-ui.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>${REPO_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict>
</plist>
PLIST

    # Unload any previous version, then load the new one.
    launchctl bootout "gui/$(id -u)/${PLIST_LABEL}" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "${PLIST_DEST}" 2>/dev/null || true
    SERVICE_STARTED=true
    info "LaunchAgent registered. The server will auto-start on login."
    ;;

  Linux)
    UNIT_DEST="${HOME}/.config/systemd/user/${PLIST_LABEL}.service"
    info "Installing systemd user unit at ${UNIT_DEST} …"

    mkdir -p "${HOME}/.config/systemd/user"
    cat > "${UNIT_DEST}" <<UNIT
[Unit]
Description=Browser Agent API Server
After=network.target

[Service]
ExecStart=${REPO_DIR}/.venv/bin/uv run browser-agent-ui
WorkingDirectory=${REPO_DIR}
Restart=on-failure
RestartSec=3
Environment=PATH=${REPO_DIR}/.venv/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
UNIT

    systemctl --user daemon-reload 2>/dev/null || true
    systemctl --user enable "${PLIST_LABEL}.service" 2>/dev/null || true
    systemctl --user start "${PLIST_LABEL}.service" 2>/dev/null || true
    SERVICE_STARTED=true
    info "Systemd unit registered. The server will auto-start on login."
    ;;

  *)
    warn "Unsupported platform: $(uname -s). Skipping background service."
    ;;
esac

# ── 4. Start the server immediately ────────────────────────────────────

if [ "${SERVICE_STARTED}" = false ]; then
  info "Starting API server in background …"
  cd "${REPO_DIR}"
  nohup uv run browser-agent-ui > /tmp/browser-agent-ui.log 2>&1 &
  disown
  # Give it a moment to start.
  sleep 2
  info "Server started."
else
  info "Background service started by launchctl/systemd."
fi

# ── 5. Register native messaging host ──────────────────────────────────

cd "${REPO_DIR}"

EXTENSION_ID="${1:-}"
if [ -z "${EXTENSION_ID}" ]; then
  warn "No extension ID supplied."
  info "After loading the extension in chrome://extensions, re-run:"
  info "  ${0} <extension-id>"
  info "to pin the native host to your extension."
  # Run without an ID so the host is still registered (no allowed_origins).
  bash "${NATIVE_HOST_DIR}/install.sh" "" 2>/dev/null || true
else
  bash "${NATIVE_HOST_DIR}/install.sh" "${EXTENSION_ID}"
fi

# ── 6. (macOS) Brave policy ──────────────────────────────────────────

if [[ "$(uname -s)" == "Darwin" && -n "${EXTENSION_ID}" && "${BRAVE_INSTALL_POLICY:-0}" == "1" ]]; then
  info "Writing Brave policy plist …"
  BRAVE_INSTALL_POLICY=1 bash "${NATIVE_HOST_DIR}/install.sh" "${EXTENSION_ID}" 2>/dev/null || true
fi

# ── 7. Instructions ───────────────────────────────────────────────────

echo ""
echo -e "${GREEN}${BOLD}✓ Browser Agent installed.${NC}"
echo ""
echo "  ${BOLD}Load the extension:${NC}"
echo "    1. Open Chrome/Brave and go to chrome://extensions"
echo "    2. Enable 'Developer mode' (top-right toggle)"
echo "    3. Click 'Load unpacked' and select:"
echo "       ${EXTENSION_DIR}"
echo "    4. Copy the 32-character extension ID from the tile"
echo "    5. Re-run:  ${0} <extension-id>"
echo ""
echo "  ${BOLD}Start a task:${NC}"
echo "    Click the extension icon → side panel opens."
echo "    Set your API key and model in the Settings tab."
echo "    Type a task in the Chat tab (e.g. 'go to example.com')."
echo ""
echo "  ${BOLD}Server status:${NC}"
echo "    The API server runs on http://127.0.0.1:8000"
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "    Manage the service: launchctl kickstart gui/$(id -u)/${PLIST_LABEL}"
  echo "    Logs: ~/Library/Logs/browser-agent-ui.log"
elif [[ "$(uname -s)" == "Linux" ]]; then
  echo "    Manage the service: systemctl --user restart ${PLIST_LABEL}"
fi
echo ""
