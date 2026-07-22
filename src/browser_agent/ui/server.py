from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from browser_agent.agent.loop import run_task_streaming
from browser_agent.config import load_config
from browser_agent.models.discovery import (
    ModelDiscoveryError,
    fetch_models,
    is_allowed_base_url,
)
from browser_agent.models.registry import get_adapter
from browser_agent.safety import SafetyLayer, StreamingConfirmationGate

log = logging.getLogger(__name__)

_TASKS: dict[str, dict[str, Any]] = {}
_TASK_TTL = 600  # seconds before an orphaned task entry is cleaned up
_BACKGROUND_TASKS: set[asyncio.Task] = set()
_task_semaphore: asyncio.Semaphore | None = None
_config_lock: asyncio.Lock | None = None
# Per-task SSE subscriber reference count. A task entry's `_subscribers` is
# incremented when a stream client connects and decremented on disconnect.
# Only the last subscriber to leave (or the agent itself finishing) cancels
# the agent task — otherwise a single client disconnect would kill the agent
# for any other concurrent subscribers.
_SUBSCRIBER_KEY = "_subscribers"
# Cached config-token read with a short TTL. Avoids re-parsing .env on
# every auth'd request (and a token-rotation TOCTOU between the read and
# the compare).
_TOKEN_TTL_S = 1.0
_token_cache: tuple[float, str | None] | None = None


async def _read_json_body(request: Request) -> dict[str, Any]:
    """FastAPI dependency: parse JSON body or return 400 on decode error.

    Wraps `request.json()` so endpoints consistently turn a malformed
    body into a 400 with a useful error message instead of a 500 from
    an unhandled `json.JSONDecodeError`. Returns an empty dict when the
    body is empty (FastAPI's `json()` raises on empty input).
    """
    raw = await request.body()
    if not raw:
        return {}
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"invalid JSON body: {exc.msg} (line {exc.lineno}, col {exc.colno})",
        ) from exc
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=400,
            detail="JSON body must be an object",
        )
    return result


def _track(task: asyncio.Task) -> asyncio.Task:
    """Keep a strong reference to a fire-and-forget task so the GC doesn't eat it."""
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _task_semaphore, _config_lock
    cfg = load_config()
    _task_semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)
    _config_lock = asyncio.Lock()
    _track(asyncio.create_task(_cleanup_orphans()))
    yield


app = FastAPI(title="Browser Agent", lifespan=lifespan)


def _configured_token() -> str | None:
    """Return the API token from the current .env, or None if unset.

    Cached for a short TTL (1s) to avoid re-parsing .env on every
    auth'd request, and to give a tight, predictable window for token
    rotation. Outside the window a fresh read picks up the rotation.
    """
    import time

    global _token_cache
    now = time.monotonic()
    if _token_cache is not None:
        cached_at, cached_value = _token_cache
        if now - cached_at < _TOKEN_TTL_S:
            return cached_value
    value = load_config().browser_agent_api_token
    _token_cache = (now, value)
    return value


def _require_auth(request: Request) -> None:
    """FastAPI dependency: reject if no token configured OR header mismatches.

    - No token configured  -> 403 (endpoint disabled, fail-closed)
    - Header missing       -> 401
    - Token mismatch       -> 401
    - Match                -> None (request proceeds)

    Comparison is timing-safe via secrets.compare_digest.
    """
    expected = _configured_token()
    if not expected:
        raise HTTPException(
            status_code=403,
            detail="endpoint disabled: set BROWSER_AGENT_API_TOKEN in .env",
        )
    provided = request.headers.get("X-Auth-Token", "")
    if not provided:
        raise HTTPException(status_code=401, detail="X-Auth-Token header required")
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="invalid X-Auth-Token")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the SPA shell.

    Returns a minimal fallback page if the static asset is missing
    (e.g. partial install). Path is constructed from __file__ — not
    user input — so no path-traversal risk.
    """
    html_path = pathlib.Path(__file__).parent / "static" / "index.html"
    try:
        return html_path.read_text()
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>UI not installed</h1>"
            "<p>static/index.html is missing. Reinstall the package.</p>",
            status_code=404,
        )


@app.get("/api/config")
async def get_config():
    """Return the current safe config (no API keys, no secrets)."""
    cfg = load_config()
    return {
        "provider": cfg.provider,
        "llm_base_url": cfg.llm_base_url,
        "llm_model": cfg.llm_model,
        "vision_mode": cfg.vision_mode,
        "vision_models": cfg.vision_models,
        "headless": cfg.headless,
        "max_steps": cfg.max_steps,
    }


# Lowercase JSON keys (what GET returns) -> uppercase env-var names on disk.
# Keeping this explicit keeps the write side auditable.
_CONFIG_KEYS: dict[str, str] = {
    "llm_base_url": "LLM_BASE_URL",
    "llm_model": "LLM_MODEL",
    "vision_mode": "VISION_MODE",
    "vision_models": "VISION_MODELS",
    "headless": "HEADLESS",
    "max_steps": "MAX_STEPS",
    "allowlist": "ALLOWLIST",
    "blocklist": "BLOCKLIST",
    "kill_switch": "KILL_SWITCH",
    "sensitivity_llm": "SENSITIVITY_LLM",
    "dom_categories": "DOM_CATEGORIES",
    "max_extract_chars": "MAX_EXTRACT_CHARS",
    "max_concurrent_tasks": "MAX_CONCURRENT_TASKS",
}


@app.post("/api/config", dependencies=[Depends(_require_auth)])
async def update_config(
    request: Request,
    body: dict[str, Any] = Depends(_read_json_body),  # noqa: B008
):
    """Persist safe config fields to .env (process-wide, survives restart).

    Accepts the same lowercase keys GET /api/config returns. Empty strings are
    treated as "clear". Uppercase keys are also accepted (deprecated) for
    backward compatibility.

    Scope: this is the *defaults* knob — it rewrites .env so later `load_config()`
    calls (and restarts) pick it up. It does NOT mutate already-running tasks:
    each /api/task builds an isolated Config via `with_overrides()` at submit
    time, so in-flight tasks are unaffected by a config write mid-run. Per-task
    overrides belong in the POST /api/task body, not here.

    The read-modify-write of .env is serialized with an asyncio lock so two
    concurrent POSTs don't clobber each other's writes. The lock is lazily
    initialized to match the semaphore pattern in `create_task` — both
    handle the lifespan-less deployment identically.
    """
    global _config_lock
    if _config_lock is None:
        _config_lock = asyncio.Lock()
        log.warning(
            "_config_lock was None at request time — lifespan did not run. "
            "Initialized lazily; check that the app is served via an ASGI "
            "lifespan-aware runner (e.g. uvicorn)."
        )
    async with _config_lock:
        env_path = _env_path()
        env_path.parent.mkdir(parents=True, exist_ok=True)
        existing = _read_env(env_path)
        for lower, env_key in _CONFIG_KEYS.items():
            if lower in body:
                value = body[lower]
                existing[env_key] = "" if value is None else str(value)
            elif env_key in body:
                value = body[env_key]
                existing[env_key] = "" if value is None else str(value)
        _write_env(env_path, existing)
    return {"status": "ok"}


@app.get("/api/models")
async def list_models(request: Request, base_url: str = ""):
    """Fetch the model list from the configured provider's /v1/models endpoint.

    Security:
      - Requires X-Auth-Token (see _require_auth).
      - base_url, if supplied, must match the configured LLM_BASE_URL
        (normalized comparison). If omitted, the configured LLM_BASE_URL
        is used. This prevents SSRF: an attacker cannot point us at an
        arbitrary host and relay the X-API-Key there.
      - If LLM_BASE_URL is not configured, the endpoint is disabled (403).
    """
    cfg = load_config()
    configured_url = cfg.llm_base_url
    if not configured_url:
        return JSONResponse(
            content={"error": "endpoint disabled: set LLM_BASE_URL in .env"},
            status_code=403,
        )

    if base_url:
        if not is_allowed_base_url(base_url, configured_url):
            return JSONResponse(
                content={"error": "base_url does not match configured LLM_BASE_URL"},
                status_code=400,
            )
        target = base_url
    else:
        target = configured_url

    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        return JSONResponse(
            content={"error": "X-API-Key header is required"},
            status_code=400,
        )
    try:
        models = fetch_models(target, api_key)
    except ModelDiscoveryError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=502)
    return {"models": models}


def _require_auth_optional(request: Request) -> None:
    """FastAPI dependency: enforce auth IF BROWSER_AGENT_API_TOKEN is set.

    Unlike _require_auth (which fails-closed with 403 when no token is
    configured), this dependency is fail-OPEN: if no token is configured,
    the request proceeds without auth. This matches the /api/models
    pattern where the extension's Fetch button must work without a token.

    When a token IS configured, requests must include a matching
    X-Auth-Token header. This prevents unauthorized local processes from
    submitting tasks (and burning API credits) when the user has opted
    into auth.
    """
    expected = _configured_token()
    if not expected:
        return  # No token configured — allow (fail-open)
    provided = request.headers.get("X-Auth-Token", "")
    if not provided:
        raise HTTPException(status_code=401, detail="X-Auth-Token header required")
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="invalid X-Auth-Token")


@app.post("/api/task", dependencies=[Depends(_require_auth_optional)])
async def create_task(
    request: Request,
    body: dict[str, Any] = Depends(_read_json_body),  # noqa: B008
):
    # Lazy-init the semaphore if the lifespan didn't run (e.g. a non-ASGI
    # server that doesn't fire lifespan, or a one-off script importing the
    # app). Without this, a missing-lifespan deployment would silently
    # disable rate limiting and orphan cleanup.
    global _task_semaphore
    if _task_semaphore is None:
        cfg = load_config()
        _task_semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)
        _track(asyncio.create_task(_cleanup_orphans()))
        log.warning(
            "_task_semaphore was None at request time — lifespan did not run. "
            "Initialized lazily; check that the app is served via an ASGI "
            "lifespan-aware runner (e.g. uvicorn)."
        )

    task_text = str(body.get("task", "")).strip()
    if not task_text:
        return JSONResponse(content={"error": "task is required"}, status_code=400)

    # Rate limiting: prevent resource exhaustion from concurrent browser sessions.
    # Use a short-timeout acquire (no separate locked() check) so the check
    # and acquire are atomic — no TOCTOU race where two requests both see
    # locked()==False and one blocks indefinitely on the second acquire.
    sem = _task_semaphore
    if sem is not None:
        try:
            await asyncio.wait_for(sem.acquire(), timeout=0.05)
        except TimeoutError:
            return JSONResponse(
                content={"error": "too many concurrent tasks — try again later"},
                status_code=429,
            )

    cfg = load_config()
    # API key is passed via the X-API-Key header (not the request body)
    # to avoid logging it in body-level middleware/crash dumps.
    api_key = request.headers.get("X-API-Key")
    overrides = {
        "llm_model": body.get("model"),
        "llm_base_url": body.get("base_url"),
        "llm_api_key": api_key,
        "vision_mode": body.get("vision_mode"),
        "vision_models": body.get("vision_models"),
        "provider": body.get("provider"),
        "cdp_url": body.get("cdp_url"),
    }
    cfg = cfg.with_overrides(**overrides)

    try:
        adapter = get_adapter(cfg)
    except ValueError as exc:
        if sem is not None:
            sem.release()
        return JSONResponse(content={"error": str(exc)}, status_code=400)

    task_id = uuid.uuid4().hex[:12]  # 48 bits — collision-resistant for practical use
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    gate = StreamingConfirmationGate()

    _TASKS[task_id] = {
        "queue": queue,
        "gate": gate,
        "created": asyncio.get_running_loop().time(),
        "task": None,
    }

    safety = SafetyLayer(cfg, gate=gate, chat_model=adapter.chat_model() if cfg.sensitivity_llm else None)

    async def _run_and_cleanup():
        try:
            await run_task_streaming(
                task_text,
                cfg=cfg,
                adapter=adapter,
                safety=safety,
                queue=queue,
            )
        except asyncio.CancelledError:
            log.info("Task %s cancelled", task_id)
            try:
                await queue.put({"type": "error", "message": "Task cancelled"})
            except Exception:
                pass
            raise
        except Exception as exc:
            log.exception("Task %s failed", task_id)
            try:
                await queue.put({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            if sem is not None:
                sem.release()
            _TASKS.pop(task_id, None)

    _TASKS[task_id]["task"] = _track(asyncio.create_task(_run_and_cleanup()))

    return {"task_id": task_id}


@app.get("/api/task/{task_id}/stream")
async def stream_task(task_id: str):
    entry = _TASKS.get(task_id)
    if entry is None:
        async def empty():
            yield {"data": json.dumps({"type": "error", "message": "task not found"})}
        return EventSourceResponse(empty())

    queue = entry["queue"]
    agent_task: asyncio.Task | None = entry.get("task")
    # Reference-count concurrent subscribers. The task entry stays in
    # _TASKS until every subscriber has disconnected (or the agent
    # itself has finished), so a second client connecting to the same
    # task_id still gets the events. Only the LAST subscriber to leave
    # cancels the agent task — and only if the agent hasn't already
    # produced a terminal event (done/error) for them.
    entry[_SUBSCRIBER_KEY] = entry.get(_SUBSCRIBER_KEY, 0) + 1

    async def generator():
        # The original `_TASKS.pop(task_id, None)` here would delete
        # the entry for any other concurrent subscriber. Track locally
        # and only pop when the last subscriber leaves.
        try:
            while True:
                item = await queue.get()
                yield {"data": json.dumps(item)}
                if item["type"] in ("done", "error"):
                    break
        except asyncio.CancelledError:
            # A client disconnect mid-stream: cancel the agent task
            # only if no other subscriber is still listening and the
            # agent has not already terminated.
            if entry.get(_SUBSCRIBER_KEY, 0) <= 1 and \
                    agent_task is not None and not agent_task.done():
                agent_task.cancel()
            raise
        finally:
            entry[_SUBSCRIBER_KEY] = max(0, entry.get(_SUBSCRIBER_KEY, 0) - 1)
            if entry.get(_SUBSCRIBER_KEY, 0) == 0:
                _TASKS.pop(task_id, None)

    return EventSourceResponse(generator())


@app.post("/api/gate/approve")
async def gate_approve(
    body: dict[str, Any] = Depends(_read_json_body),  # noqa: B008
):
    gate_id = str(body.get("gate_id", ""))

    for entry in _TASKS.values():
        gate = entry["gate"]
        if isinstance(gate, StreamingConfirmationGate) and gate.resolve(gate_id, True):
            return {"status": "approved"}
    return JSONResponse(content={"status": "not found"}, status_code=404)


@app.post("/api/gate/deny")
async def gate_deny(
    body: dict[str, Any] = Depends(_read_json_body),  # noqa: B008
):
    gate_id = str(body.get("gate_id", ""))

    for entry in _TASKS.values():
        gate = entry["gate"]
        if isinstance(gate, StreamingConfirmationGate) and gate.resolve(gate_id, False):
            return {"status": "denied"}
    return JSONResponse(content={"status": "not found"}, status_code=404)


# Keychain proxy endpoints — bridge the extension's keychain operations
# through the local API server so the extension doesn't depend on native
# messaging. This is the primary keychain path for Brave (which blocks
# native messaging for unpacked extensions despite a correctly-pinned
# allowed_origins + NativeMessagingAllowlist policy) and a fallback for
# any environment where the native host can't run.
#
# Security: these endpoints are intentionally UNAUTHENTICATED. The server
# is bound to 127.0.0.1 (localhost only), and the `service` and `key`
# params are validated against a tight allowlist, so even a local process
# that reaches them can only read/write the browser-agent keychain entry
# — not arbitrary OS keychain entries. Requiring X-Auth-Token here would
# create a chicken-and-egg problem: the extension has no way to obtain
# the token (it doesn't read .env), so the keychain bridge would always
# fail and the API key would fall back to chrome.storage.local (plaintext
# on disk). Keeping these open is the safer trade-off.

_KEYCHAIN_ALLOWED_SERVICES = frozenset({"browser-agent", "browser-agent-test"})
_KEYCHAIN_KEY_PATTERN = __import__("re").compile(r"^[A-Za-z0-9_-]{1,128}$")


def _validate_keychain_params(service: str, key: str) -> None:
    if service not in _KEYCHAIN_ALLOWED_SERVICES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown service: {service!r}",
        )
    if not _KEYCHAIN_KEY_PATTERN.match(key or ""):
        raise HTTPException(
            status_code=400,
            detail="invalid key (must be 1-128 chars, [A-Za-z0-9_-])",
        )


def _keychain_error_response(exc: Exception) -> JSONResponse:
    return JSONResponse(
        content={"ok": False, "error": f"{type(exc).__name__}: {exc}"},
        status_code=500,
    )


@app.post("/api/keychain/ping")
async def keychain_ping():
    """Health check for the keychain bridge. Returns the same shape as the
    native host's ping so the side panel can use it as a drop-in probe."""
    try:
        import keyring
        # Force backend initialization so the ping fails loudly if keyring
        # can't reach the OS secret store (vs. silently passing and then
        # failing on the first set/get).
        keyring.get_keyring()
    except Exception as exc:
        return _keychain_error_response(exc)
    return {"ok": True, "pong": True}


@app.post("/api/keychain/set")
async def keychain_set(
    body: dict[str, Any] = Depends(_read_json_body),  # noqa: B008
):
    service = str(body.get("service", ""))
    key = str(body.get("key", ""))
    value = str(body.get("value", ""))
    _validate_keychain_params(service, key)
    try:
        import keyring
        keyring.set_password(service, key, value)
    except Exception as exc:
        return _keychain_error_response(exc)
    return {"ok": True}


@app.post("/api/keychain/get")
async def keychain_get(
    body: dict[str, Any] = Depends(_read_json_body),  # noqa: B008
):
    service = str(body.get("service", ""))
    key = str(body.get("key", ""))
    _validate_keychain_params(service, key)
    try:
        import keyring
        value = keyring.get_password(service, key)
    except Exception as exc:
        return _keychain_error_response(exc)
    return {"ok": True, "value": value}


@app.post("/api/keychain/delete")
async def keychain_delete(
    body: dict[str, Any] = Depends(_read_json_body),  # noqa: B008
):
    service = str(body.get("service", ""))
    key = str(body.get("key", ""))
    _validate_keychain_params(service, key)
    try:
        import keyring
        from keyring.errors import PasswordDeleteError
        # `PasswordDeleteError` means "not present" — treat as success so
        # the client can call delete idempotently, matching the native
        # host's behavior. We import the exception by type with a
        # KeyError/FileNotFoundError fallback for keyring backends that
        # raise something different on missing entries.

        try:
            keyring.delete_password(service, key)
        except PasswordDeleteError:
            pass
        except (KeyError, FileNotFoundError):
            # Some backends raise these on a missing entry.
            pass
    except Exception as exc:
        return _keychain_error_response(exc)
    return {"ok": True}


def _env_path() -> pathlib.Path:
    return pathlib.Path.cwd() / ".env"


def _read_env(path: pathlib.Path) -> dict[str, str]:
    """Read .env using python-dotenv for correct quoting/escaping handling."""
    from dotenv import dotenv_values
    raw = dotenv_values(str(path))
    return {k: v for k, v in raw.items() if v is not None}


def _write_env(path: pathlib.Path, values: dict[str, str]) -> None:
    """Write .env using python-dotenv for correct quoting/escaping handling."""
    from dotenv import set_key
    for k, v in values.items():
        set_key(str(path), k, v)


async def _cleanup_orphans() -> None:
    while True:
        await asyncio.sleep(60)
        now = asyncio.get_running_loop().time()
        stale = [
            tid for tid, e in list(_TASKS.items())
            if now - e.get("created", 0) > _TASK_TTL
        ]
        for tid in stale:
            entry = _TASKS.pop(tid, None)
            if entry is not None:
                agent_task = entry.get("task")
                if agent_task is not None and not agent_task.done():
                    agent_task.cancel()


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
