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

    def test_rejects_unknown_service(self):
        """A compromised extension must not be able to write to arbitrary
        keychain services — only browser-agent / browser-agent-test are allowed."""
        key = _make_unique_key("evil")
        r = _run_host([
            {"cmd": "set_key", "service": "evil-corp", "key": key, "value": "stolen"},
        ])
        assert r[0]["ok"] is False
        assert "unknown service" in r[0]["error"]

    def test_rejects_invalid_key_chars(self):
        """Keys with spaces or special chars must be rejected."""
        r = _run_host([
            {"cmd": "get_key", "service": "browser-agent", "key": "has spaces!"},
        ])
        assert r[0]["ok"] is False
        assert "invalid key" in r[0]["error"]

    def test_rejects_empty_key(self):
        r = _run_host([
            {"cmd": "get_key", "service": "browser-agent", "key": ""},
        ])
        assert r[0]["ok"] is False
        assert "invalid key" in r[0]["error"]

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


NATIVE_HOST_DIR = HOST.parent
TEMPLATE = NATIVE_HOST_DIR / "com.browseragent.json.template"
INSTALL_SH = NATIVE_HOST_DIR / "install.sh"


def _read_template_name() -> str:
    """Pull the `name` field out of the template without running the script."""
    data = json.loads(TEMPLATE.read_text())
    return data["name"]


class TestInstallManifest:
    """Regression tests for install.sh's manifest filename.

    Chromium discovers native messaging hosts by looking for a file named
    <name>.json in the NativeMessagingHosts directory. If the filename
    doesn't match the host's `name` field, every connectNative() call fails
    with "Specified native messaging host not found" and the keychain
    bridge silently degrades to chrome.storage.local.
    """

    def test_generated_manifest_filename_matches_name_field(self):
        """The manifest file the script produces must be named <name>.json.

        install.sh writes the rendered manifest to FINAL=.../<name>.json.
        If anyone changes the script's `name` field but forgets to update
        FINAL (or vice versa), Chromium silently fails to discover the
        host. This guard runs install.sh in an isolated temp dir and
        inspects what it produced.
        """
        import shutil
        import tempfile

        if not INSTALL_SH.exists():
            pytest.skip(f"install.sh not present at {INSTALL_SH}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Mirror the directory layout the script expects: native_host.py,
            # the template, and a writable output dir. We don't need a
            # working ${HOME}/Library/... path because we never run the
            # platform-specific TARGET_DIR branch (Linux is fine here).
            staging = tmp_path / "staging"
            staging.mkdir()
            shutil.copy(HOST, staging / "native_host.py")
            shutil.copy(TEMPLATE, staging / "com.browseragent.json.template")
            script = staging / "install.sh"
            shutil.copy(INSTALL_SH, script)
            script.chmod(0o755)

            # We can't easily redirect the macOS / Linux TARGET_DIRs without
            # editing the script, so we only verify the on-disk FINAL the
            # script wrote. That file is what install.sh *also* copies into
            # the per-platform directory.
            result = subprocess.run(
                [str(script), "test-extension-id-12345678901234567890"],
                capture_output=True, text=True, timeout=30,
            )
            # The script's Darwin/Linux branches both `cp "${FINAL}"
            # "${TARGET_DIR}/..."` and will succeed on macOS / Linux
            # test hosts. On macOS it copies into the user's Brave/Chrome
            # NativeMessagingHosts dir — harmless, idempotent. We don't
            # assert the cp succeeded; we only inspect the FINAL file the
            # script always writes regardless of platform.
            assert result.returncode == 0, (
                f"install.sh failed: stdout={result.stdout!r} "
                f"stderr={result.stderr!r}"
            )

            final = staging / "com.browseragent.native_host.json"
            assert final.exists(), (
                f"install.sh did not produce {final.name!r}; "
                f"stdout={result.stdout!r} stderr={result.stderr!r}"
            )

            data = json.loads(final.read_text())
            assert data["name"] + ".json" == final.name, (
                f"manifest filename {final.name!r} does not match the "
                f"host's `name` field {data['name']!r}; Chromium will fail "
                f"to discover this host with 'Specified native messaging "
                f"host not found'."
            )
            # And the pinned origin must be present and well-formed.
            assert data["allowed_origins"] == [
                "chrome-extension://test-extension-id-12345678901234567890/"
            ]

    def test_install_sh_final_uses_host_name_filename(self):
        """Belt-and-braces: assert the script's own FINAL variable name
        ends in <name>.json, so an edit that renames the host without
        updating FINAL fails this test."""
        if not INSTALL_SH.exists():
            pytest.skip(f"install.sh not present at {INSTALL_SH}")

        source = INSTALL_SH.read_text()
        name = _read_template_name()
        # FINAL is set on a single line; the basename must be <name>.json.
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("FINAL="):
                final_path = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                basename = final_path.rsplit("/", 1)[-1]
                assert basename == name + ".json", (
                    f"FINAL basename is {basename!r} but host name is "
                    f"{name!r}; install.sh would produce a manifest "
                    f"Chromium cannot discover."
                )
                return
        pytest.fail("could not find FINAL= assignment in install.sh")


class TestBraveNativeMessagingPolicy:
    """Brave requires an explicit enterprise policy to permit unpacked dev
    extensions to use native messaging. Without it, connectNative() returns
    'Access to the specified native messaging host is forbidden' even when
    the host's `allowed_origins` matches the extension ID byte-for-byte.

    The fix is a managed-preferences plist at:
        ~/Library/Managed Preferences/com.brave.Browser.plist
    setting NativeMessagingAllowlist (with the extension ID) and
    NativeMessagingUserLevelHosts (true).

    This test asserts the plist is present, valid, and well-formed. It
    only runs on macOS and only when the plist exists — so a developer
    without Brave (or running on CI) is not broken.
    """

    PLIST_PATH = Path.home() / "Library/Managed Preferences" / "com.brave.Browser.plist"

    def _require_plist(self):
        if sys.platform != "darwin":
            pytest.skip("Brave managed-preferences plist is macOS-specific")
        if not self.PLIST_PATH.exists():
            pytest.skip(
                f"Brave policy plist not present at {self.PLIST_PATH}. "
                f"If you're live-testing the extension on Brave, see the "
                f"extension/native_host/ directory for the policy install "
                f"instructions."
            )

    def test_plist_is_valid(self):
        self._require_plist()
        # plistlib raises on malformed plist; we want any parse error to
        # fail loudly so a corrupt managed-preferences file is caught at
        # test time rather than at browser restart.
        import plistlib
        data = plistlib.loads(self.PLIST_PATH.read_bytes())
        assert isinstance(data, dict)

    def test_plist_has_native_messaging_allowlist(self):
        self._require_plist()
        import plistlib
        data = plistlib.loads(self.PLIST_PATH.read_bytes())
        allowlist = data.get("NativeMessagingAllowlist")
        assert isinstance(allowlist, list), (
            f"NativeMessagingAllowlist must be a list, got {type(allowlist).__name__}. "
            f"Without an allowlist, Brave forbids native messaging for unpacked "
            f"extensions and connectNative() fails with 'forbidden'."
        )
        assert all(isinstance(item, str) and len(item) == 32 for item in allowlist), (
            f"Each allowlist entry must be a 32-char extension ID; got {allowlist!r}"
        )
        assert len(allowlist) > 0, "NativeMessagingAllowlist is empty"

    def test_plist_allows_user_level_hosts(self):
        self._require_plist()
        import plistlib
        data = plistlib.loads(self.PLIST_PATH.read_bytes())
        assert data.get("NativeMessagingUserLevelHosts") is True, (
            "NativeMessagingUserLevelHosts must be true. install.sh installs the "
            "host at ~/Library/.../NativeMessagingHosts/ (the user level); if this "
            "policy is false or missing, Brave will refuse to load user-level hosts."
        )

    def test_install_sh_documented_requirement(self):
        """Belt-and-braces: the install script must mention the Brave policy
        requirement so future installers don't silently fall back to the
        'forbidden' failure mode."""
        if not INSTALL_SH.exists():
            pytest.skip(f"install.sh not present at {INSTALL_SH}")
        source = INSTALL_SH.read_text()
        assert "NativeMessagingAllowlist" in source or "Brave" in source and "policy" in source.lower(), (
            "install.sh must document the Brave NativeMessagingAllowlist policy "
            "requirement. Without it, users installing on Brave will see "
            "'Access to the specified native messaging host is forbidden' and "
            "the keychain bridge will silently fall back to chrome.storage.local."
        )
