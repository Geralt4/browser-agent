from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from browser_agent.agent.loop import run_task_streaming
from browser_agent.config import load_config
from browser_agent.models.discovery import ModelDiscoveryError, fetch_models
from browser_agent.models.registry import get_adapter
from browser_agent.safety import SafetyLayer, StreamingConfirmationGate

log = logging.getLogger(__name__)

_TASKS: dict[str, dict[str, Any]] = {}
_TASK_TTL = 600  # seconds before an orphaned task entry is cleaned up
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _track(task: asyncio.Task) -> asyncio.Task:
    """Keep a strong reference to a fire-and-forget task so the GC doesn't eat it."""
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


@asynccontextmanager
async def lifespan(app: FastAPI):
    _track(asyncio.create_task(_cleanup_orphans()))
    yield


app = FastAPI(title="Browser Agent", lifespan=lifespan)


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


@app.post("/api/config")
async def update_config(request: Request):
    """Update the process-wide .env-loaded config.

    Persists to .env so subsequent restarts pick it up. Accepts the same safe
    fields as GET /api/config. Empty strings are treated as "clear".
    """
    body = await request.json()
    env_path = _env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_env(env_path)
    for key in ("LLM_BASE_URL", "LLM_MODEL", "VISION_MODE", "VISION_MODELS"):
        if key in body:
            value = body[key]
            existing[key] = "" if value is None else str(value)
    _write_env(env_path, existing)
    return {"status": "ok"}


@app.get("/api/models")
async def list_models(request: Request, base_url: str = ""):
    """Fetch the model list from the provider's /v1/models endpoint.

    The API key is read from the X-API-Key header to avoid leaking it into
    server access logs, browser history, and proxy logs.
    """
    api_key = request.headers.get("X-API-Key", "")
    if not base_url or not api_key:
        return JSONResponse(
            content={"error": "base_url query param and X-API-Key header are required"},
            status_code=400,
        )
    try:
        models = fetch_models(base_url, api_key)
    except ModelDiscoveryError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=502)
    return {"models": models}


@app.post("/api/task")
async def create_task(request: Request):
    body = await request.json()
    task_text = body.get("task", "").strip()
    if not task_text:
        return JSONResponse(content={"error": "task is required"}, status_code=400)

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
        return JSONResponse(content={"error": str(exc)}, status_code=400)

    task_id = uuid.uuid4().hex[:8]
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
        except Exception as exc:
            log.exception("Task %s failed", task_id)
            try:
                await queue.put({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
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
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in pathlib.Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _write_env(path: pathlib.Path, values: dict[str, str]) -> None:
    lines = []
    seen: set[str] = set()
    if path.exists():
        for line in pathlib.Path(path).read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                lines.append(line)
                continue
            k = s.split("=", 1)[0].strip()
            if k in values:
                lines.append(f"{k}={values[k]}")
                seen.add(k)
            else:
                lines.append(line)
    for k, v in values.items():
        if k not in seen:
            lines.append(f"{k}={v}")
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


async def _cleanup_orphans() -> None:
    while True:
        await asyncio.sleep(60)
        now = asyncio.get_running_loop().time()
        stale = [tid for tid, e in _TASKS.items() if now - e.get("created", 0) > _TASK_TTL]
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
