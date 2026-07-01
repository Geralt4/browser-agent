"""Smoke tests for the native messaging host (keyring bridge).

We spawn the host as a subprocess and feed it messages using the
Chrome native messaging protocol (4-byte LE length + UTF-8 JSON).
This exercises the real wire format and the real keyring library.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

HOST = Path(__file__).resolve().parents[1] / "extension" / "native_host" / "native_host.py"


def _encode(msg: dict) -> bytes:
    body = json.dumps(msg).encode("utf-8")
    return struct.pack("<I", len(body)) + body


def _decode(stream) -> dict:
    header = stream.read(4)
    assert len(header) == 4, f"unexpected EOF reading header: got {len(header)} bytes"
    (length,) = struct.unpack("<I", header)
    body = stream.read(length)
    assert len(body) == length, f"short read: expected {length}, got {len(body)}"
    return json.loads(body.decode("utf-8"))


def _run_host(messages: list[dict], timeout: float = 15.0) -> list[dict]:
    """Run the host as a subprocess, feed it messages, return responses."""
    proc = subprocess.Popen(
        [sys.executable, str(HOST)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    responses = []
    try:
        for msg in messages:
            proc.stdin.write(_encode(msg))
            proc.stdin.flush()
            responses.append(_decode(proc.stdout))
        # Close stdin so the host exits cleanly
        proc.stdin.close()
        proc.wait(timeout=timeout)
    except Exception:
        proc.kill()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        pytest.fail(f"host subprocess failed: {stderr}")
    return responses


def _make_unique_key(prefix: str) -> str:
    """Use a unique key per test to avoid clobbering real keychain entries."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class TestNativeHost:
    def test_ping(self):
        r = _run_host([{"cmd": "ping"}])
        assert r[0] == {"ok": True, "pong": True, "_id": None}

    def test_set_then_get_then_delete(self):
        service = "browser-agent-test"
        key = _make_unique_key("api")
        value = "sk-test-" + uuid.uuid4().hex

        r = _run_host([
            {"cmd": "set_key", "service": service, "key": key, "value": value},
            {"cmd": "get_key", "service": service, "key": key},
            {"cmd": "delete_key", "service": service, "key": key},
            {"cmd": "get_key", "service": service, "key": key},
        ])
        assert r[0] == {"ok": True, "_id": None}
        assert r[1] == {"ok": True, "value": value, "_id": None}
        assert r[2] == {"ok": True, "_id": None}
        # After delete, value is None
        assert r[3] == {"ok": True, "value": None, "_id": None}

    def test_unknown_cmd(self):
        r = _run_host([{"cmd": "wat"}])
        assert r[0]["ok"] is False
        assert "unknown cmd" in r[0]["error"]
        assert r[0]["_id"] is None

    def test_echoes_request_id(self):
        r = _run_host([{"cmd": "ping", "_id": "abc-123"}])
        assert r[0] == {"ok": True, "pong": True, "_id": "abc-123"}

    def test_get_missing_key_returns_null(self):
        service = "browser-agent-test"
        key = _make_unique_key("nonexistent")
        r = _run_host([{"cmd": "get_key", "service": service, "key": key}])
        assert r[0] == {"ok": True, "value": None, "_id": None}

    def test_delete_missing_key_is_idempotent(self):
        service = "browser-agent-test"
        key = _make_unique_key("ghost")
        r = _run_host([{"cmd": "delete_key", "service": service, "key": key}])
        assert r[0] == {"ok": True, "_id": None}

    def test_invalid_json(self):
        proc = subprocess.Popen(
            [sys.executable, str(HOST)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            proc.stdin.write(_encode({"not_a_real_message": True}))
            proc.stdin.flush()
            r = _decode(proc.stdout)
            proc.stdin.close()
            proc.wait(timeout=10)
            # The host will try to read msg.get("cmd") and get None, then return unknown cmd
            assert r["ok"] is False
            assert r["_id"] is None
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_non_dict_json_returns_error_and_continues(self):
        """A non-dict JSON value (e.g. a list) must not crash the host.
        The host should return an error and stay alive for the next message."""
        proc = subprocess.Popen(
            [sys.executable, str(HOST)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            # Send a JSON array (valid JSON, not a dict)
            proc.stdin.write(_encode([1, 2, 3]))  # type: ignore[arg-type]
            proc.stdin.flush()
            r = _decode(proc.stdout)
            assert r["ok"] is False
            assert "expected a JSON object" in r["error"]
            assert r["_id"] is None

            # Host must still be alive — send a ping
            proc.stdin.write(_encode({"cmd": "ping"}))
            proc.stdin.flush()
            r2 = _decode(proc.stdout)
            assert r2 == {"ok": True, "pong": True, "_id": None}

            proc.stdin.close()
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_string_json_returns_error_and_continues(self):
        """A bare JSON string must not crash the host either."""
        proc = subprocess.Popen(
            [sys.executable, str(HOST)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            proc.stdin.write(_encode("hello"))  # type: ignore[arg-type]
            proc.stdin.flush()
            r = _decode(proc.stdout)
            assert r["ok"] is False
            assert "expected a JSON object" in r["error"]
            proc.stdin.close()
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
