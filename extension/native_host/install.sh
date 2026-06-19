#!/usr/bin/env bash
# Registers the browser-agent native messaging host with Chrome so that the
# extension can use the OS keychain for API key storage.
#
# After running this:
#   1. Load the extension in chrome://extensions (Developer mode > Load unpacked)
#   2. Copy the extension's ID
#   3. Re-run this script with the ID:  ./install.sh <extension-id>
#
# Idempotent — safe to re-run.

set -euo pipefail

HOST_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST_PY="${HOST_DIR}/native_host.py"
TEMPLATE="${HOST_DIR}/com.browseragent.json.template"
FINAL="${HOST_DIR}/com.browseragent.json"

if [[ ! -f "${HOST_PY}" ]]; then
  echo "ERROR: native_host.py not found at ${HOST_PY}" >&2
  exit 1
fi

EXTENSION_ID="${1:-}"

# Pick a location Chrome looks for native messaging hosts. Each platform has
# its own well-known directory.
TARGET_DIR=""
case "$(uname -s)" in
  Darwin)
    TARGET_DIR="${HOME}/Library/Application Support/Google/Chrome/NativeMessagingHosts"
    ;;
  Linux)
    TARGET_DIR="${HOME}/.config/google-chrome/NativeMessagingHosts"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "ERROR: Windows is not supported by this script yet. Register the host via the Windows Registry manually." >&2
    echo "See: https://developer.chrome.com/docs/apps/nativeMessaging/#native-messaging-host-location" >&2
    exit 1
    ;;
  *)
    echo "ERROR: unrecognized platform: $(uname -s)" >&2
    exit 1
    ;;
esac

# Render the manifest with absolute paths. If an extension ID was supplied
# we also lock the manifest to that ID (allowed_origins must list the full
# chrome-extension://ID/ origin).
python3 - <<PY
import json, sys, pathlib
template = pathlib.Path("${TEMPLATE}").read_text()
template = template.replace("REPLACE_WITH_ABSOLUTE_PATH_TO_NATIVE_HOST", "${HOST_DIR}")
allowed = ["chrome-extension://${EXTENSION_ID}/"] if "${EXTENSION_ID}" else []
data = json.loads(template)
data["allowed_origins"] = allowed
pathlib.Path("${FINAL}").write_text(json.dumps(data, indent=2) + "\n")
PY

mkdir -p "${TARGET_DIR}"
cp "${FINAL}" "${TARGET_DIR}/com.browseragent.json"

echo ""
echo "Native host registered at: ${TARGET_DIR}/com.browseragent.json"
if [[ -z "${EXTENSION_ID}" ]]; then
  echo ""
  echo "Next steps:"
  echo "  1. Open chrome://extensions in Chrome"
  echo "  2. Enable 'Developer mode' (top right)"
  echo "  3. Click 'Load unpacked' and select the parent extension/ directory"
  echo "  4. Copy the extension's ID"
  echo "  5. Re-run: $0 <extension-id>"
  echo ""
  echo "(Re-running with the ID pins the native host to your extension and "
  echo "prevents other extensions from invoking it.)"
else
  echo "Pinned to extension ID: ${EXTENSION_ID}"
fi
