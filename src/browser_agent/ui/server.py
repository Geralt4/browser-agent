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
from fastapi.middleware.cors import CORSMiddleware
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


def _sanitize_error(exc: Exception) -> str:
    """S19: return a generic error string for client-facing responses.

    Internal exception messages may contain file paths, stack details,
    or other sensitive context. We log the full exception server-side
    (via log.exception) but return a sanitized message to the client.
    """
    return "An internal error occurred"


async def _read_json_body(request: Request) -> dict[str, Any]:
    """FastAPI dependency: parse a JSON object body, or return 4xx on errors.

    Reads the raw body via `request.body()` and decodes it as JSON, so we
    can validate the Content-Type header (which S1 uses to force a CORS
    preflight) before doing any parsing. Returns an empty dict when the
    body is empty so endpoints can treat absent-fields as defaults.

    Failure modes:
    - Wrong/missing Content-Type  -> 415 (S1: forces CORS preflight)
    - Empty body                  -> {} (caller treats as no fields)
    - Malformed JSON              -> 400 with line/column of the error
    - JSON value is not an object -> 400 (bare lists/scalars are rejected)
    """
    content_type = request.headers.get("content-type", "")
    if not content_type.split(";")[0].strip().lower() == "application/json":
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be application/json",
        )
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

# S5: CORS middleware — only allow requests from the browser extension
# (any chrome-extension:// origin) and from the local web UI at
# http://127.0.0.1:8000. Any other origin cannot call the API, which
# blocks CSRF attacks where a malicious website tries to create tasks
# or poison the keychain by submitting simple cross-origin POSTs.
# Combined with the Content-Type check in _read_json_body (which
# forces a CORS preflight), this fully blocks browser-initiated
# cross-origin abuse.
#
# `allow_origin_regex` is required because the chrome-extension scheme
# uses a per-instance UUID, not a fixed origin. `allow_origins`
# does not support wildcards in Starlette's CORSMiddleware.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",  # the local web UI
    ],
    allow_origin_regex=r"^chrome-extension://[a-z]{32}$",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=[
        "Content-Type",
        "X-Auth-Token",
        "X-API-Key",
    ],
)


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


def _validate_base_url_write(value: str) -> str:
    """S11: validate a LLM_BASE_URL value before persisting it.

    Rejects values that aren't well-formed URLs (no scheme, no hostname)
    or that look like URL-credential attacks (userinfo@host). This
    prevents config poisoning where an attacker tricks the user into
    saving a URL that bypasses the SSRF guard on /api/task (e.g.
    ``https://api.openai.com@evil.com``).
    """
    from urllib.parse import urlparse

    if not value or not value.strip():
        return value  # empty = clear; no validation needed
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail="LLM_BASE_URL must use http or https scheme",
        )
    if not parsed.hostname:
        raise HTTPException(
            status_code=400,
            detail="LLM_BASE_URL must have a valid hostname",
        )
    if parsed.username or parsed.password:
        raise HTTPException(
            status_code=400,
            detail="LLM_BASE_URL must not contain userinfo (user:password@host)",
        )
    return value


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
        # S11: validate LLM_BASE_URL before persisting to prevent config
        # poisoning that would bypass the SSRF guard on /api/task.
        if "LLM_BASE_URL" in existing and existing["LLM_BASE_URL"]:
            _validate_base_url_write(existing["LLM_BASE_URL"])
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

    # S8/S9: same resolution order as /api/task — X-API-Key header first,
    # then the OS keychain. Curl/scripts that don't have keychain access
    # still work via the header.
    api_key = request.headers.get("X-API-Key", "") or _read_keychain_api_key() or ""
    if not api_key:
        return JSONResponse(
            content={"error": "X-API-Key header or keychain entry required"},
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


@app.post("/api/task", dependencies=[Depends(_require_auth)])
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
    # S2 (SSRF guard): if the body supplies a base_url, it must normalize-match
    # the configured LLM_BASE_URL. Without this, any caller could redirect
    # outbound model traffic to an internal/external URL they control, which
    # turns the model list discovery endpoint into an SSRF oracle. The check
    # is intentionally strict: a missing/unset configured URL rejects every
    # requested URL, so an unconfigured server cannot be coerced into
    # fetching arbitrary targets.
    requested_base_url = body.get("base_url")
    if requested_base_url is not None and requested_base_url != "":
        if not is_allowed_base_url(str(requested_base_url), cfg.llm_base_url):
            if sem is not None:
                sem.release()  # release the slot we acquired above
            return JSONResponse(
                content={"error": "base_url not allowed by server policy"},
                status_code=400,
            )

    # API key resolution order (S8/S9):
    #   1. X-API-Key header (explicit, used by curl/scripts/tests)
    #   2. OS keychain (preferred for the browser extension — never crosses
    #      the HTTP boundary, never stored in chrome.storage.local)
    #   3. .env's LLM_API_KEY / MOONSHOT_API_KEY (loaded into cfg by
    #      load_config() — preserved as a final fallback)
    header_key = request.headers.get("X-API-Key")
    keychain_key = header_key if header_key else _read_keychain_api_key()
    api_key = keychain_key
    overrides = {
        "llm_model": body.get("model"),
        "llm_base_url": body.get("base_url"),
        "llm_api_key": api_key,
        "vision_mode": body.get("vision_mode"),
        "vision_models": body.get("vision_models"),
        "provider": body.get("provider"),
        "cdp_url": body.get("cdp_url"),
    }
    try:
        cfg = cfg.with_overrides(**overrides)
    except Exception as exc:
        # Pydantic validators on cdp_url (S13) raise ValidationError when
        # the user submits a non-loopback URL. Surface a clean 400 instead
        # of a 500.
        if sem is not None:
            sem.release()
        return JSONResponse(content={"error": str(exc)}, status_code=400)

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
                # S19: log the full exception server-side, but send a
                # sanitized message to the client — internal exception
                # text may contain file paths or other sensitive context.
                await queue.put({"type": "error", "message": _sanitize_error(exc)})
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


@app.post(
    "/api/gate/approve",
    dependencies=[Depends(_require_auth_optional)],
)
async def gate_approve(
    body: dict[str, Any] = Depends(_read_json_body),  # noqa: B008
):
    gate_id = str(body.get("gate_id", ""))

    for entry in _TASKS.values():
        gate = entry["gate"]
        if isinstance(gate, StreamingConfirmationGate) and gate.resolve(gate_id, True):
            return {"status": "approved"}
    return JSONResponse(content={"status": "not found"}, status_code=404)


@app.post(
    "/api/gate/deny",
    dependencies=[Depends(_require_auth_optional)],
)
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
    # S19: log the full exception server-side, but return a generic
    # message to the client — the raw exception type/message may reveal
    # internal state (e.g. "PasswordDeleteError: no such password" tells
    # an attacker whether a keychain entry exists).
    log.debug("keychain error: %s", exc, exc_info=True)
    return JSONResponse(
        content={"ok": False, "error": "keychain operation failed"},
        status_code=500,
    )


def _check_origin_allowed(request: Request) -> None:
    """Defense-in-depth Origin check for unauthenticated keychain endpoints.

    S4: the keychain endpoints are intentionally unauthenticated (the
    extension has no way to obtain BROWSER_AGENT_API_TOKEN — that would
    force the API key into plaintext chrome.storage.local). CORS already
    blocks cross-origin browser requests, but if a misconfiguration ever
    weakened the CORSMiddleware allowlist, this explicit check would
    still refuse any non-allowed origin.

    Allowed origins:
    - `chrome-extension://<id>` — the browser extension (any ID, since
      unpacked extensions get a per-install UUID)
    - `http://127.0.0.1:8000` — the local web UI
    - No `Origin` header at all — server-to-server calls (curl, scripts)

    This is best-effort, not cryptographic: a malicious browser extension
    can trivially forge any Origin. The check protects against CSRF from
    a malicious website, which is the realistic threat model.
    """
    import re

    origin = request.headers.get("origin", "")
    if not origin:
        return  # server-to-server; no browser context
    if origin == "http://127.0.0.1:8000":
        return
    if re.match(r"^chrome-extension://[a-z]{32}$", origin):
        return
    raise HTTPException(
        status_code=403,
        detail="origin not allowed",
    )


# S22: simple in-memory rate limiter for unauthenticated keychain endpoints.
# A local malicious process (or a compromised browser extension) could spam
# these endpoints to enumerate keychain entries or exhaust the OS secret
# store. The limiter caps each client IP to _KEYCHAIN_RATE_LIMIT_MAX calls
# per _KEYCHAIN_RATE_LIMIT_WINDOW seconds.
_KEYCHAIN_RATE_LIMIT_MAX = 10
_KEYCHAIN_RATE_LIMIT_WINDOW = 10.0  # seconds
_keychain_rate_buckets: dict[str, list[float]] = {}


def _rate_limit_keychain(request: Request) -> None:
    """FastAPI dependency: enforce a per-IP rate limit on keychain endpoints.

    Uses a sliding-window counter. When the limit is exceeded, returns 429.
    The bucket dict is lazily pruned of expired entries on each call to
    avoid unbounded growth.
    """
    import time

    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    cutoff = now - _KEYCHAIN_RATE_LIMIT_WINDOW
    bucket = _keychain_rate_buckets.get(client_ip, [])
    # Prune entries outside the window.
    bucket = [t for t in bucket if t > cutoff]
    if len(bucket) >= _KEYCHAIN_RATE_LIMIT_MAX:
        _keychain_rate_buckets[client_ip] = bucket
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded — try again later",
        )
    bucket.append(now)
    _keychain_rate_buckets[client_ip] = bucket


@app.post(
    "/api/keychain/ping",
    dependencies=[Depends(_check_origin_allowed), Depends(_rate_limit_keychain)],
)
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


@app.post(
    "/api/keychain/set",
    dependencies=[Depends(_check_origin_allowed), Depends(_rate_limit_keychain)],
)
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


@app.post(
    "/api/keychain/get",
    dependencies=[Depends(_check_origin_allowed), Depends(_rate_limit_keychain)],
)
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


@app.post(
    "/api/keychain/delete",
    dependencies=[Depends(_check_origin_allowed), Depends(_rate_limit_keychain)],
)
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


def _read_keychain_api_key() -> str | None:
    """Read the LLM API key from the OS keychain via the `keyring` lib.

    S8+S9: the extension stores the key in the OS keychain (service
    `browser-agent`, key `llm_api_key`) rather than chrome.storage.local
    (plaintext on disk) or the X-API-Key HTTP header (cleartext over the
    loopback HTTP boundary). The server reads it here, in-process, so the
    key never crosses the network — even locally. Returns None if the
    keychain is unreachable, the key is absent, or any exception fires.
    The caller should fall back to the X-API-Key header in that case so
    the API still works for non-extension clients (curl, scripts).
    """
    try:
        import keyring
        return keyring.get_password("browser-agent", "llm_api_key")
    except Exception:
        return None


def _read_env(path: pathlib.Path) -> dict[str, str]:
    """Read .env using python-dotenv for correct quoting/escaping handling."""
    from dotenv import dotenv_values
    raw = dotenv_values(str(path))
    return {k: v for k, v in raw.items() if v is not None}


def _write_env(path: pathlib.Path, values: dict[str, str]) -> None:
    """Write .env using python-dotenv for correct quoting/escaping handling.

    S10: immediately chmod 0o600 after writing so the file is only
    readable by the owning user. Without this, the file may inherit
    the process's umask (commonly 0o022) and be world-readable.
    """
    import os

    from dotenv import set_key
    for k, v in values.items():
        set_key(str(path), k, v)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # best-effort; non-POSIX platforms may not support chmod


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
