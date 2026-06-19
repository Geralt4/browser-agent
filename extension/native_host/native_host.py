#!/usr/bin/env python3
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
import struct
import sys


def main() -> int:
    try:
        import keyring
    except ImportError:
        send({"ok": False, "error": "keyring library not installed"})
        return 1

    try:
        while True:
            raw = read_message()
            if raw is None:
                return 0
            try:
                msg = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                send({"ok": False, "error": f"invalid JSON: {exc}"})
                continue

            cmd = msg.get("cmd")
            try:
                if cmd == "ping":
                    send({"ok": True, "pong": True})
                elif cmd == "set_key":
                    keyring.set_password(
                        msg["service"], msg["key"], msg["value"]
                    )
                    send({"ok": True})
                elif cmd == "get_key":
                    value = keyring.get_password(msg["service"], msg["key"])
                    send({"ok": True, "value": value})
                elif cmd == "delete_key":
                    try:
                        keyring.delete_password(msg["service"], msg["key"])
                    except keyring.errors.PasswordDeleteError:
                        pass
                    send({"ok": True})
                elif cmd == "list_keys":
                    send({"ok": True, "hint": "list_keys not supported by keyring API"})
                else:
                    send({"ok": False, "error": f"unknown cmd: {cmd!r}"})
            except Exception as exc:  # noqa: BLE001
                send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    except BrokenPipeError:
        return 0
    except Exception as exc:  # noqa: BLE001
        send({"ok": False, "error": f"host crashed: {exc}"})
        return 1


def read_message() -> bytes | None:
    """Read one Chrome native messaging message from stdin.

    Returns None on EOF. Format: 4-byte little-endian length prefix + UTF-8 body.
    """
    header = sys.stdin.buffer.read(4)
    if not header or len(header) < 4:
        return None
    (length,) = struct.unpack("<I", header)
    if length == 0:
        return b""
    return sys.stdin.buffer.read(length)


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
