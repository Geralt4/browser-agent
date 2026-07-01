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


def _track(task: asyncio.Task) -> asyncio.Task:
    """Keep a strong reference to a fire-and-forget task so the GC doesn't eat it."""
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _task_semaphore
    cfg = load_config()
    _task_semaphore = asyncio.Semaphore(cfg.max_concurrent_tasks)
    _track(asyncio.create_task(_cleanup_orphans()))
    yield


app = FastAPI(title="Browser Agent", lifespan=lifespan)


def _configured_token() -> str | None:
    """Return the API token from the current .env, or None if unset.

    Read per-request (not cached at startup) so token rotation via .env edit
    takes effect without a restart, and so tests can monkeypatch the env.
    """
    return load_config().browser_agent_api_token


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
    html = (pathlib.Path(__file__).parent / "static" / "index.html").read_text()
    return html


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
async def update_config(request: Request):
    """Persist safe config fields to .env (process-wide, survives restart).

    Accepts the same lowercase keys GET /api/config returns. Empty strings are
    treated as "clear". Uppercase keys are also accepted (deprecated) for
    backward compatibility.

    Scope: this is the *defaults* knob — it rewrites .env so later `load_config()`
    calls (and restarts) pick it up. It does NOT mutate already-running tasks:
    each /api/task builds an isolated Config via `with_overrides()` at submit
    time, so in-flight tasks are unaffected by a config write mid-run. Per-task
    overrides belong in the POST /api/task body, not here.
    """
    body = await request.json()
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


@app.post("/api/task")
async def create_task(request: Request):
    body = await request.json()
    task_text = body.get("task", "").strip()
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
    overrides = {
        "llm_model": body.get("model"),
        "llm_base_url": body.get("base_url"),
        "llm_api_key": body.get("api_key"),
        "vision_mode": body.get("vision_mode"),
        "vision_models": body.get("vision_models"),
        "provider": body.get("provider"),
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

    async def generator():
        try:
            while True:
                item = await queue.get()
                yield {"data": json.dumps(item)}
                if item["type"] in ("done", "error"):
                    break
        except asyncio.CancelledError:
            if agent_task is not None and not agent_task.done():
                agent_task.cancel()
        finally:
            _TASKS.pop(task_id, None)

    return EventSourceResponse(generator())


@app.post("/api/gate/approve")
async def gate_approve(request: Request):
    body = await request.json()
    gate_id = body.get("gate_id", "")

    for entry in _TASKS.values():
        gate = entry["gate"]
        if isinstance(gate, StreamingConfirmationGate) and gate.resolve(gate_id, True):
            return {"status": "approved"}
    return JSONResponse(content={"status": "not found"}, status_code=404)


@app.post("/api/gate/deny")
async def gate_deny(request: Request):
    body = await request.json()
    gate_id = body.get("gate_id", "")

    for entry in _TASKS.values():
        gate = entry["gate"]
        if isinstance(gate, StreamingConfirmationGate) and gate.resolve(gate_id, False):
            return {"status": "denied"}
    return JSONResponse(content={"status": "not found"}, status_code=404)


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
