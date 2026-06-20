"""Benchmark runner — executes tasks.jsonl against the agent and records results.

Usage:
    python -m benchmarks.run              # auto (per-task heuristic, default)
    python -m benchmarks.run --vision     # always use vision (A/B test)
    python -m benchmarks.run --dom        # always DOM-only
    python -m benchmarks.run --ids nav-01,form-02  # specific tasks
    python -m benchmarks.run --category navigation  # filter by category

Output: benchmarks/results/<timestamp>_<mode>.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from browser_agent.agent.loop import run_task
from browser_agent.config import Config, load_config
from browser_agent.models.registry import get_adapter
from browser_agent.safety import SafetyLayer
from browser_agent.safety.gate import ConfirmationGate
from browser_agent.safety.types import SafetyDecision

BENCH_DIR = Path(__file__).parent
TASKS_FILE = BENCH_DIR / "tasks.jsonl"
RESULTS_DIR = BENCH_DIR / "results"


def load_tasks() -> list[dict]:
    tasks = []
    with open(TASKS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def filter_tasks(tasks: list[dict], ids: list[str] | None, category: str | None) -> list[dict]:
    if ids:
        id_set = set(ids)
        return [t for t in tasks if t["id"] in id_set]
    if category:
        return [t for t in tasks if t["category"] == category]
    return tasks


async def run_one(task: dict, cfg: Config, gate: ConfirmationGate) -> dict[str, object]:
    adapter = get_adapter(cfg)
    safety = SafetyLayer(cfg, gate=gate)

    started = time.monotonic()
    try:
        history = await run_task(task["task"], cfg=cfg, adapter=adapter, safety=safety)
        result = history.final_result() if hasattr(history, "final_result") else str(history)
        elapsed = time.monotonic() - started
        steps = len(history.history) if hasattr(history, "history") else -1
        return {
            "id": task["id"],
            "category": task["category"],
            "status": "ok",
            "result": str(result),
            "steps": steps,
            "elapsed_s": round(elapsed, 2),
            "error": "",
        }
    except Exception as exc:
        elapsed = time.monotonic() - started
        return {
            "id": task["id"],
            "category": task["category"],
            "status": "error",
            "result": "",
            "steps": 0,
            "elapsed_s": round(elapsed, 2),
            "error": str(exc)[:200],
        }


async def run_all(tasks: list[dict], cfg: Config, gate: ConfirmationGate, concurrency: int = 1) -> list[dict]:
    results: list[dict] = []
    sem = asyncio.Semaphore(concurrency)

    async def bounded(task: dict) -> None:
        async with sem:
            results.append(await run_one(task, cfg, gate))

    await asyncio.gather(*(bounded(t) for t in tasks))
    results.sort(key=lambda r: str(r["id"]))
    return results


def write_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "status", "result", "steps", "elapsed_s", "error"])
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run browser-agent benchmark")
    parser.add_argument("--vision", action="store_true", help="Always use vision (vision_mode=vision)")
    parser.add_argument("--dom", action="store_true", help="Always DOM-only (vision_mode=dom)")
    parser.add_argument("--ids", type=str, help="Comma-separated task IDs to run")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel tasks (default 1)")
    parser.add_argument(
        "--gate",
        choices=["auto-deny", "auto-approve"],
        default="auto-deny",
        help="Confirmation gate strategy (default: auto-deny)",
    )
    args = parser.parse_args()

    _HAS_KEY = bool(os.getenv("MOONSHOT_API_KEY") or os.getenv("LLM_API_KEY"))
    if not _HAS_KEY:
        print("ERROR: MOONSHOT_API_KEY or LLM_API_KEY must be set in .env")
        sys.exit(1)

    tasks = load_tasks()
    ids = [i.strip() for i in args.ids.split(",")] if args.ids else None
    tasks = filter_tasks(tasks, ids, args.category)

    cfg = load_config()
    if args.vision:
        cfg.vision_mode = "vision"
    elif args.dom:
        cfg.vision_mode = "dom"

    async def _auto_deny(action):
        return SafetyDecision(allow=False, reason="auto-denied (benchmark)")

    async def _auto_approve(action):
        return SafetyDecision(allow=True, reason="auto-approved (benchmark)")

    gate = ConfirmationGate(
        confirm=_auto_deny if args.gate == "auto-deny" else _auto_approve
    )

    mode = cfg.vision_mode
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{ts}_{mode}.csv"

    print(f"Running {len(tasks)} tasks (mode={mode}, gate={args.gate}, concurrency={args.concurrency})")
    results = asyncio.run(run_all(tasks, cfg, gate, concurrency=args.concurrency))

    write_csv(results, out)

    ok = sum(1 for r in results if r["status"] == "ok")
    err = sum(1 for r in results if r["status"] == "error")
    total_s = sum(float(r["elapsed_s"]) for r in results)
    avg_s = total_s / len(results) if results else 0

    print(f"\nDone.  {ok} ok  {err} errors  {total_s:.1f}s total  {avg_s:.1f}s avg")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
