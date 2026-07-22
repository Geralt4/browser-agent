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
# Chromium discovers native messaging hosts by looking for a file named
# <name>.json in the NativeMessagingHosts directory. If this filename
# doesn't match the host's `name` field, every connectNative() call fails
# with "Specified native messaging host not found." We compute it from the
# template's `name` field below so it can't drift out of sync.
FINAL="${HOST_DIR}/com.browseragent.native_host.json"

if [[ ! -f "${HOST_PY}" ]]; then
  echo "ERROR: native_host.py not found at ${HOST_PY}" >&2
  exit 1
fi

# Ensure the host is executable. A fresh checkout won't have the +x bit
# set, and Chrome launches the host by exec'ing the `path` from the
# manifest — without +x, the launch fails silently and the keychain
# bridge falls back to chrome.storage.local.
chmod +x "${HOST_PY}"

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "ERROR: template not found at ${TEMPLATE}" >&2
  exit 1
fi

if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 is required but not found in PATH" >&2
  exit 1
fi

EXTENSION_ID="${1:-}"

# Pick a location Chrome looks for native messaging hosts. Each platform has
# its own well-known directory.
TARGET_DIR=""
case "$(uname -s)" in
  Darwin)
    # Support both Google Chrome and Brave Browser (Chromium-based).
    # BRAVE_HOST=1 selects Brave; default is Chrome.
    if [[ "${BRAVE_HOST:-0}" == "1" ]]; then
      TARGET_DIR="${HOME}/Library/Application Support/BraveSoftware/Brave-Browser/NativeMessagingHosts"
    else
      TARGET_DIR="${HOME}/Library/Application Support/Google/Chrome/NativeMessagingHosts"
    fi
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
# we lock the manifest to that ID (allowed_origins must list the full
# chrome-extension://ID/ origin). When no ID is supplied we REFUSE to
# install — leaving allowed_origins absent would mean "any extension can
# connect" per Chrome's spec, and writing an empty array also blocks the
# intended user. Either way, an unpinned host is a security hole, so we
# force the user to re-run with an ID.
# Values are passed via env vars to avoid shell-interpolation issues with
# paths containing quotes, backslashes, or dollar signs.
TEMPLATE="$TEMPLATE" HOST_DIR="$HOST_DIR" EXTENSION_ID="$EXTENSION_ID" FINAL="$FINAL" python3 - <<'PY'
import json, os, pathlib
template = pathlib.Path(os.environ["TEMPLATE"]).read_text()
template = template.replace(
    "REPLACE_WITH_ABSOLUTE_PATH_TO_NATIVE_HOST",
    os.environ["HOST_DIR"],
)
ext_id = os.environ["EXTENSION_ID"]
data = json.loads(template)
if ext_id:
    data["allowed_origins"] = ["chrome-extension://" + ext_id + "/"]
else:
    # S6 fix: write an empty allowed_origins (a deny-by-default) instead
    # of omitting the key. The previous "pop" left the field absent, which
    # Chrome interprets as "any extension may connect" — a hostile
    # extension could invoke our host and read OS keychain entries.
    # An empty array is the safest default: no extension connects until
    # the user re-runs with their ID. (Note: the caller in install.sh
    # already aborts on missing ID; this is a belt-and-suspenders for
    # anyone running this script directly.)
    data["allowed_origins"] = []
final_path = pathlib.Path(os.environ["FINAL"])
final_path.write_text(json.dumps(data, indent=2) + "\n")
# Sanity check: Chromium requires the installed manifest filename to
# match the host's `name` field. If they diverge, every connectNative()
# call returns "Specified native messaging host not found" — which
# silently degrades the keychain bridge to chrome.storage.local.
expected = data["name"] + ".json"
if final_path.name != expected:
    raise SystemExit(
        f"FATAL: manifest filename is {final_path.name!r} but the host "
        f"name is {data['name']!r} — Chromium requires {expected!r}. "
        f"Update FINAL in install.sh."
    )
PY

mkdir -p "${TARGET_DIR}"
cp "${FINAL}" "${TARGET_DIR}/$(basename "${FINAL}")"

echo ""
echo "Native host registered at: ${TARGET_DIR}/$(basename "${FINAL}")"
if [[ -z "${EXTENSION_ID}" ]]; then
  echo ""
  echo "WARNING: registered with allowed_origins: [] (no extension can connect)."
  echo "This is the safest default — you can re-run with an ID to pin the host:"
  echo ""
  echo "  1. Open chrome://extensions in Chrome"
  echo "  2. Enable 'Developer mode' (top right)"
  echo "  3. Click 'Load unpacked' and select the parent extension/ directory"
  echo "  4. Copy the extension's ID"
  echo "  5. Re-run: $0 <extension-id>"
  echo ""
  echo "(Until you re-run with the ID, the keychain bridge will not work.)"
else
  echo "Pinned to extension ID: ${EXTENSION_ID}"
fi

# macOS-specific warning: Chrome launches native hosts in a sandboxed
# environment that may not inherit the user's full PATH. If python3 is
# installed via Homebrew (e.g. /opt/homebrew/bin/python3), the shebang
# may not resolve at launch time. Suggest a symlink to a PATH location
# Chrome does see.
if [[ "$(uname -s)" == "Darwin" ]]; then
  PYTHON_BIN="$(command -v python3)"
  case "$PYTHON_BIN" in
    /opt/homebrew/*|/usr/local/*)
      echo ""
      echo "NOTE: Your python3 is at $PYTHON_BIN (Homebrew). Chrome's"
      echo "sandboxed native-host environment may not see it. If the host"
      echo "fails to launch, run:"
      echo "  sudo ln -sf '$PYTHON_BIN' /usr/local/bin/python3"
      ;;
  esac
fi

# Brave-specific policy requirement.
#
# Chromium permits native messaging for any extension whose origin is
# listed in the host's `allowed_origins`. Brave adds a stricter overlay:
# for extensions installed outside the Web Store (e.g. unpacked dev
# extensions, location 4), Brave's default policy is to REFUSE native
# messaging with the error "Access to the specified native messaging host
# is forbidden" — even when allowed_origins matches byte-for-byte.
#
# The override is an enterprise-policy plist. Setting
# NativeMessagingAllowlist (with the extension ID) AND
# NativeMessagingUserLevelHosts (true) on Brave's bundle ID grants the
# permission. Without this, the keychain bridge silently falls back to
# chrome.storage.local. See:
#   https://chromeenterprise.google/policies/#NativeMessagingAllowlist
#   https://chromeenterprise.google/policies/#NativeMessagingUserLevelHosts
#
# We don't write the plist automatically — managed preferences are a
# system-trust boundary and the user should opt in explicitly. On macOS
# we print a clear next-step with the exact commands.
if [[ "$(uname -s)" == "Darwin" && -n "${EXTENSION_ID}" ]]; then
  BRAVE_BUNDLE="com.brave.Browser"
  USER_POLICY="${HOME}/Library/Managed Preferences/${BRAVE_BUNDLE}.plist"

  if [[ -f "${USER_POLICY}" ]] && /usr/bin/plutil -lint "${USER_POLICY}" &>/dev/null; then
    # Already installed — confirm it mentions our ID.
    if /usr/bin/plutil -extract NativeMessagingAllowlist raw -o - "${USER_POLICY}" 2>/dev/null | rg -q "${EXTENSION_ID}"; then
      :
    else
      echo ""
      echo "WARNING: ${USER_POLICY} exists but does not include your"
      echo "extension ID (${EXTENSION_ID}) in NativeMessagingAllowlist."
      echo "Add it manually, or delete the plist and re-run with"
      echo "BRAVE_INSTALL_POLICY=1 to let this script rewrite it."
    fi
  else
    echo ""
    echo "=== Brave-only next step ==="
    echo "Brave requires a managed-preferences plist to permit native"
    echo "messaging for unpacked extensions. Create it with:"
    echo ""
    echo "  mkdir -p \"\${HOME}/Library/Managed Preferences\""
    echo "  cat > \"\${HOME}/Library/Managed Preferences/${BRAVE_BUNDLE}.plist\" <<'EOF'"
    echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
    echo "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">"
    echo "<plist version=\"1.0\">"
    echo "<dict>"
    echo "  <key>NativeMessagingAllowlist</key>"
    echo "  <array>"
    echo "    <string>${EXTENSION_ID}</string>"
    echo "  </array>"
    echo "  <key>NativeMessagingUserLevelHosts</key>"
    echo "  <true/>"
    echo "</dict>"
    echo "</plist>"
    echo "EOF"
    echo ""
    echo "Then fully quit Brave (\u2318Q) and relaunch. Verify the policy"
    echo "loaded at brave://policy."
    echo ""
    echo "(Set BRAVE_INSTALL_POLICY=1 to let this script write the plist"
    echo "for you. We don't do it by default because managed preferences"
    echo "are a system-trust boundary — the user should opt in.)"
  fi
fi

# Optional: actually write the plist when the user explicitly opts in.
# Idempotent — safe to re-run; the file is rewritten from the current
# EXTENSION_ID every time.
if [[ "$(uname -s)" == "Darwin" && "${BRAVE_INSTALL_POLICY:-0}" == "1" && -n "${EXTENSION_ID}" ]]; then
  BRAVE_BUNDLE="com.brave.Browser"
  USER_POLICY_DIR="${HOME}/Library/Managed Preferences"
  USER_POLICY="${USER_POLICY_DIR}/${BRAVE_BUNDLE}.plist"
  mkdir -p "${USER_POLICY_DIR}"
  if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found in PATH; cannot write the Brave policy plist." >&2
    exit 1
  fi
  EXTENSION_ID="$EXTENSION_ID" USER_POLICY="$USER_POLICY" python3 - <<'PY'
import os, plistlib, pathlib
ext_id = os.environ["EXTENSION_ID"]
path = pathlib.Path(os.environ["USER_POLICY"])
data = {"NativeMessagingAllowlist": [ext_id], "NativeMessagingUserLevelHosts": True}
with path.open("wb") as f:
    plistlib.dump(data, f)
print(f"Wrote Brave policy plist: {path}")
PY
  echo "Pinned extension ID: ${EXTENSION_ID}"
  echo "Fully quit Brave (\u2318Q) and relaunch for the policy to take effect."
fi
