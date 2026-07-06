#!/Users/giannismanologlou/Documents/Browser-agent/.venv/bin/python3
"""Chrome native messaging host for browser-agent.

This is a tiny stdio proxy between the Chrome extension and the OS keychain.
The extension sends JSON commands over stdin (Chrome's native messaging
protocol = 4-byte little-endian length prefix + UTF-8 JSON), and this script
executes the command against the keyring library (macOS Keychain / Windows
Credential Manager / Linux Secret Service) and writes the response to stdout.

Commands:
    {"cmd": "set_key",    "service": str, "key": str, "value": str}
    {"cmd": "get_key",    "service": str, "key": str}
    {"cmd": "delete_key", "service": str, "key": str}
    {"cmd": "list_keys",  "service": str}
    {"cmd": "ping"}

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
import sys

# Chrome's native messaging limit. A hostile extension could otherwise send
# a length prefix of 0xFFFFFFFF and cause a 4 GB allocation attempt.
MAX_MESSAGE_SIZE = 1024 * 1024

# Security: validate the `service` and `key` params against a tight
# allowlist so a compromised extension can't write arbitrary entries
# into the OS keychain. Mirrors the server-side validation in server.py.
_ALLOWED_SERVICES = frozenset({"browser-agent", "browser-agent-test"})
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


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
