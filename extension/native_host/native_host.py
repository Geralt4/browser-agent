#!/usr/bin/env python3
"""Chrome native messaging host for browser-agent.

This is a tiny stdio proxy between the Chrome extension and the OS keychain.
The extension sends JSON commands over stdin (Chrome's native messaging
protocol = 4-byte little-endian length prefix + UTF-8 JSON), and this script
executes the command against the keyring library (macOS Keychain / Windows
Credential Manager / Linux Secret Service) and writes the response to stdout.

Additionally, on the first ``ping`` from the extension, the host checks
whether the local API server (``browser-agent-ui``) is running on
127.0.0.1:8000. If not, it spawns it as a background subprocess so the
extension's HTTP API calls work without manual server startup. This makes
"Load unpacked" in Chrome work with zero terminal commands.

Commands:
    {"cmd": "ping"}
    {"cmd": "set_key",    "service": str, "key": str, "value": str}
    {"cmd": "get_key",    "service": str, "key": str}
    {"cmd": "delete_key", "service": str, "key": str}
    {"cmd": "list_keys",  "service": str}

Responses:
    {"ok": true,  ...}    on success
    {"ok": false, "error": str}  on failure

The host is registered with Chrome via install.sh. Chrome's manifest schema
is documented here:
https://developer.chrome.com/docs/apps/nativeMessaging/#native-messaging-host
"""
from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Chrome's native messaging limit. A hostile extension could otherwise send
# a length prefix of 0xFFFFFFFF and cause a 4 GB allocation attempt.
MAX_MESSAGE_SIZE = 1024 * 1024

# Security: validate the `service` and `key` params against a tight
# allowlist so a compromised extension can't write arbitrary entries
# into the OS keychain. Mirrors the server-side validation in server.py.
_ALLOWED_SERVICES = frozenset({"browser-agent", "browser-agent-test"})
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

API_SERVER_URL = "http://127.0.0.1:8000"
# Repo root = 3 levels up from extension/native_host/ -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_STARTED = False


def _ensure_server() -> None:
    """Check if the API server is running; spawn it if not.

    Called once on the first ``ping`` from the extension. After the
    initial spawn, ``SERVER_STARTED`` is set to ``True`` to avoid
    re-checking on every ping.
    """
    global SERVER_STARTED
    if SERVER_STARTED:
        return

    # Check whether 127.0.0.1:8000 already responds.
    req = urllib.request.Request(f"{API_SERVER_URL}/api/keychain/ping", method="POST")
    try:
        urllib.request.urlopen(req, timeout=2.0)
        SERVER_STARTED = True
        return
    except Exception:
        pass

    # Server is not running — spawn it as a background subprocess.
    try:
        subprocess.Popen(
            ["uv", "run", "browser-agent-ui"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        # ``uv`` not in PATH — try ``sys.executable -m`` as a fallback.
        try:
            subprocess.Popen(
                [sys.executable, "-m", "browser_agent.ui.server"],
                cwd=str(REPO_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            # Both attempts failed; the user will need to start the server
            # manually. The side panel's error message tells them how.
            pass
    SERVER_STARTED = True


def _validate_keychain_params(service: str, key: str) -> str | None:
    """Return an error string if params are invalid, None if OK."""
    if service not in _ALLOWED_SERVICES:
        return f"unknown service: {service!r}"
    if not _KEY_PATTERN.match(key or ""):
        return "invalid key (must be 1-128 chars, [A-Za-z0-9_-])"
    return None


def main() -> int:
    try:
        import keyring
    except ImportError:
        send({"ok": False, "error": "keyring library not installed", "_id": None})
        return 1

    try:
        while True:
            # Read the 4-byte length prefix.
            header = sys.stdin.buffer.read(4)
            if not header or len(header) < 4:
                return 0
            (length,) = struct.unpack("<I", header)
            # Cap at Chrome's native messaging limit (1 MB) to prevent
            # 4 GB allocation DoS via a hostile length prefix. Drain the
            # oversized payload to keep the stream aligned.
            if length > MAX_MESSAGE_SIZE:
                send({"ok": False, "error": f"message exceeds {MAX_MESSAGE_SIZE} byte limit", "_id": None})
                sys.stdin.buffer.read(min(length, MAX_MESSAGE_SIZE))
                continue
            if length == 0:
                raw = b""
            else:
                raw = sys.stdin.buffer.read(length)
                # A short read means EOF mid-message; the stream is now
                # desynchronized, so we exit cleanly rather than crash on
                # the next length parse.
                if len(raw) < length:
                    return 0
            try:
                msg = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                send({"ok": False, "error": f"invalid JSON: {exc}", "_id": None})
                continue

            if not isinstance(msg, dict):
                # Valid JSON, but not a dict — msg.get() would raise and
                # crash the whole host. Return an error and keep the loop
                # alive so a single bad message can't kill the process.
                send({"ok": False, "error": "expected a JSON object", "_id": None})
                continue

            # Initialize _id before any msg.get() so the outer crash handler
            # always has a defined value.
            _id = None
            cmd = msg.get("cmd")
            # Echo the request's correlation id so the extension can match
            # responses to outstanding calls on a shared port. None when
            # the caller didn't supply one (or the request was unparseable).
            _id = msg.get("_id")
            try:
                if cmd == "ping":
                    _ensure_server()
                    send({"ok": True, "pong": True, "_id": _id})
                elif cmd == "set_key":
                    service = msg.get("service", "")
                    key = msg.get("key", "")
                    err = _validate_keychain_params(service, key)
                    if err:
                        send({"ok": False, "error": err, "_id": _id})
                        continue
                    keyring.set_password(service, key, msg["value"])
                    send({"ok": True, "_id": _id})
                elif cmd == "get_key":
                    service = msg.get("service", "")
                    key = msg.get("key", "")
                    err = _validate_keychain_params(service, key)
                    if err:
                        send({"ok": False, "error": err, "_id": _id})
                        continue
                    value = keyring.get_password(service, key)
                    send({"ok": True, "value": value, "_id": _id})
                elif cmd == "delete_key":
                    service = msg.get("service", "")
                    key = msg.get("key", "")
                    err = _validate_keychain_params(service, key)
                    if err:
                        send({"ok": False, "error": err, "_id": _id})
                        continue
                    try:
                        keyring.delete_password(service, key)
                    except keyring.errors.PasswordDeleteError:
                        pass
                    send({"ok": True, "_id": _id})
                elif cmd == "list_keys":
                    send({"ok": True, "hint": "list_keys not supported by keyring API", "_id": _id})
                else:
                    send({"ok": False, "error": f"unknown cmd: {cmd!r}", "_id": _id})
            except Exception as exc:  # noqa: BLE001
                send({"ok": False, "error": f"{type(exc).__name__}: {exc}", "_id": _id})

    except BrokenPipeError:
        return 0
    except Exception as exc:  # noqa: BLE001
        send({"ok": False, "error": f"host crashed: {exc}", "_id": _id})
        return 1


def send(message: dict) -> None:
    """Write one Chrome native messaging message to stdout.

    Writes a 4-byte little-endian length prefix + UTF-8 JSON body, then
    flushes so Chrome receives it immediately.
    """
    body = json.dumps(message).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(body)))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    sys.exit(main())
