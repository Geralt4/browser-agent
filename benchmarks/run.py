"""Benchmark runner — executes tasks.jsonl against the agent and records results.

Usage:
    python -m benchmarks.run                     # auto (per-task heuristic, default)
    python -m benchmarks.run --vision            # always use vision (A/B test)
    python -m benchmarks.run --dom               # always DOM-only
    python -m benchmarks.run --mode dom_only     # equivalent to --dom
    python -m benchmarks.run --mode vision_enabled # equivalent to --vision
    python -m benchmarks.run --ids nav-01,form-02  # specific tasks
    python -m benchmarks.run --category navigation  # filter by category
    python -m benchmarks.run --repeats 3         # run each task 3x to tame LLM variance

Output:
    benchmarks/results/<timestamp>_<mode>.csv  (row-by-row, resume-friendly)
    benchmarks/results/<timestamp>_<mode>.json (final aggregate with summary)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from browser_agent.agent.loop import run_task_with_model
from browser_agent.config import Config, load_config
from browser_agent.models.registry import get_adapter
from browser_agent.safety import SafetyLayer
from browser_agent.safety.gate import ConfirmationGate
from browser_agent.safety.types import SafetyDecision

BENCH_DIR = Path(__file__).parent
TASKS_FILE = BENCH_DIR / "tasks.jsonl"
RESULTS_DIR = BENCH_DIR / "results"

DEFAULT_TIMEOUT_S = 300
DEFAULT_SNAPSHOT_CHARS = 500

CSV_FIELDS = [
    "id",
    "category",
    "status",
    "passed",
    "expect",
    "prompt",
    "result",
    "steps",
    "elapsed_s",
    "tokens_in",
    "tokens_out",
    "final_url",
    "dom_snapshot_excerpt",
    "error",
]


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


def _check_expect(result: str, expect: str | None) -> bool:
    """Case-insensitive word-boundary match of `expect` against the agent result.

    Returns True iff `expect` is non-empty and appears as a whole word/phrase
    (case-insensitively) in `result`. Uses lookbehind/lookahead word-boundary
    pairs to avoid substring false-positives like "Success" matching
    "unsuccessful". A None/empty `expect` is treated as "no ground truth
    provided" and returns False.

    Uses `(?<![A-Za-z0-9_])...(?![A-Za-z0-9_])` rather than `\b...\b` so
    non-word-char-bounded expect values like "(Domain)", "$10.00", "C++"
    still match. Previously `\b` silently failed because there was no word
    char on the `expect` side of the boundary for `\b` to anchor to.
    """
    if not expect:
        return False
    pattern = (
        r"(?<![A-Za-z0-9_])" + re.escape(expect) + r"(?![A-Za-z0-9_])"
    )
    return bool(re.search(pattern, result or "", re.IGNORECASE))


# CSV columns used when --repeats > 1. Mirrors CSV_FIELDS but with repeat-aware
# columns in place of single-run fields. Per-run detail lives only in the JSON
# under tasks[].per_run.
REPEAT_CSV_FIELDS = [
    "id",
    "category",
    "runs",
    "pass_count",
    "pass_rate",
    "mean_latency_s",
    "std_latency_s",
    "mean_tokens_in",
    "mean_tokens_out",
    "mean_steps",
    "timeouts",
    "errors",
    "expect",
    "prompt",
    "best_result",
    "final_url",
    "dom_snapshot_excerpt",
]


def _stddev(xs: list[float]) -> float:
    """Sample standard deviation. Returns 0.0 for n<2 (no variance defined)."""
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return round(math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)), 2)


def _avg(xs: list[float]) -> float:
    """Mean of a list, rounded to 1 dp. Returns 0.0 on empty."""
    return round(sum(xs) / len(xs), 1) if xs else 0.0


def _aggregate(task: dict, per_run: list[dict]) -> dict:
    """Collapse N per-run rows into one summary row.

    `per_run` is the list of N `run_one` outputs for this task. The aggregated
    row has a different schema (see REPEAT_CSV_FIELDS) and a `per_run` field
    holding the originals for auditing. The first passing result is kept in
    `best_result`; if nothing passed, the last result is kept so the row still
    shows what the agent produced.
    """
    n = len(per_run)
    passes = sum(1 for r in per_run if r.get("passed") == "yes")
    timeouts = sum(1 for r in per_run if r.get("status") == "timeout")
    errors = sum(1 for r in per_run if r.get("status") == "error")
    elapsed = [float(r["elapsed_s"]) for r in per_run]
    steps = [int(r.get("steps") or 0) for r in per_run if r.get("status") == "ok"]
    tokens_in = [int(r.get("tokens_in") or 0) for r in per_run]
    tokens_out = [int(r.get("tokens_out") or 0) for r in per_run]

    def _mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    best = next((r for r in per_run if r.get("passed") == "yes"), per_run[-1] if per_run else {})

    return {
        "id": task["id"],
        "category": task["category"],
        "runs": n,
        "pass_count": passes,
        "pass_rate": round(passes / n * 100, 1) if n else 0.0,
        "mean_latency_s": _mean(elapsed),
        "std_latency_s": _stddev(elapsed),
        "mean_tokens_in": _mean([float(t) for t in tokens_in]),
        "mean_tokens_out": _mean([float(t) for t in tokens_out]),
        "mean_steps": _mean([float(s) for s in steps]),
        "timeouts": timeouts,
        "errors": errors,
        "expect": task.get("expect") or "",
        "prompt": task.get("task", ""),
        "best_result": str(best.get("result", "")),
        "final_url": best.get("final_url", "") or "",
        "dom_snapshot_excerpt": best.get("dom_snapshot_excerpt", "") or "",
        "per_run": per_run,
    }


async def _make_step_state_holder(snapshot_chars: int) -> tuple[dict, object]:
    """Build a `(holder, callback)` pair for capturing live DOM/URL per step.

    The benchmark passes the callback to `run_task_with_model(..., on_step_state=...)`.
    browser-use invokes it on every step with the live `browser_state_summary`,
    so the DOM and URL are captured *while the browser session is alive* —
    by the time `agent.run()` returns, the session is torn down.

    `holder` is a dict that the callback writes into:
      - "dom": str (truncated to snapshot_chars)
      - "url": str (last observed URL)
    """
    holder: dict[str, str] = {"dom": "", "url": ""}

    def _callback(browser_state_summary: object, model_output: object, n_steps: int) -> None:
        try:
            url = getattr(browser_state_summary, "url", None)
            if url:
                holder["url"] = str(url)
            dom_state = getattr(browser_state_summary, "dom_state", None)
            if dom_state is not None:
                text = dom_state.llm_representation()
                if text:
                    holder["dom"] = text[:snapshot_chars]
        except Exception:
            # Defensive: a single bad step should not poison the whole run.
            pass

    return holder, _callback


async def run_one(
    task: dict,
    cfg: Config,
    gate: ConfirmationGate,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    snapshot_chars: int = DEFAULT_SNAPSHOT_CHARS,
) -> dict[str, object]:
    adapter = get_adapter(cfg)
    safety = SafetyLayer(cfg, gate=gate)
    wrapped_llm, totals = adapter.with_token_counter()
    state_holder, state_callback = await _make_step_state_holder(snapshot_chars)

    started = time.monotonic()
    try:
        history = await asyncio.wait_for(
            run_task_with_model(
                task["task"],
                cfg=cfg,
                llm=wrapped_llm,
                supports_vision=adapter.supports_vision,
                safety=safety,
                category=task.get("category"),
                on_step_state=state_callback,
            ),
            timeout=timeout_s,
        )
        elapsed = time.monotonic() - started
        result = history.final_result() if hasattr(history, "final_result") else str(history)
        steps = len(history.history) if hasattr(history, "history") else -1
        expect = task.get("expect")
        passed = _check_expect(str(result), expect)
        return {
            "id": task["id"],
            "category": task["category"],
            "status": "ok",
            "passed": "yes" if passed else "no",
            "expect": expect or "",
            "prompt": task.get("task", ""),
            "result": str(result),
            "steps": steps,
            "elapsed_s": round(elapsed, 2),
            "tokens_in": totals.input,
            "tokens_out": totals.output,
            "final_url": state_holder["url"],
            "dom_snapshot_excerpt": state_holder["dom"],
            "error": "",
        }
    except TimeoutError:
        elapsed = time.monotonic() - started
        return {
            "id": task["id"],
            "category": task["category"],
            "status": "timeout",
            "passed": "no",
            "expect": task.get("expect") or "",
            "prompt": task.get("task", ""),
            "result": "",
            "steps": 0,
            "elapsed_s": round(elapsed, 2),
            "tokens_in": totals.input,
            "tokens_out": totals.output,
            "final_url": "",
            "dom_snapshot_excerpt": "",
            "error": f"timed out after {timeout_s:.0f}s",
        }
    except Exception as exc:
        elapsed = time.monotonic() - started
        return {
            "id": task["id"],
            "category": task["category"],
            "status": "error",
            "passed": "no",
            "expect": task.get("expect") or "",
            "prompt": task.get("task", ""),
            "result": "",
            "steps": 0,
            "elapsed_s": round(elapsed, 2),
            "tokens_in": totals.input,
            "tokens_out": totals.output,
            "final_url": "",
            "dom_snapshot_excerpt": "",
            "error": str(exc)[:200],
        }


async def run_one_repeated(
    task: dict,
    cfg: Config,
    gate: ConfirmationGate,
    *,
    repeats: int = 1,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    snapshot_chars: int = DEFAULT_SNAPSHOT_CHARS,
) -> dict[str, object]:
    """Run `run_one` N times and return an aggregated row.

    When `repeats == 1` returns the single row directly (no `per_run` wrapper)
    so the schema is identical to the non-repeated case. Otherwise the schema
    switches to the aggregated form (see `_aggregate`).
    """
    if repeats <= 1:
        return await run_one(
            task,
            cfg,
            gate,
            timeout_s=timeout_s,
            snapshot_chars=snapshot_chars,
        )
    per_run: list[dict] = []
    for _ in range(repeats):
        per_run.append(
            await run_one(
                task,
                cfg,
                gate,
                timeout_s=timeout_s,
                snapshot_chars=snapshot_chars,
            )
        )
    return _aggregate(task, per_run)


async def run_all(
    tasks: list[dict],
    cfg: Config,
    gate: ConfirmationGate,
    *,
    concurrency: int = 1,
    csv_path: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    snapshot_chars: int = DEFAULT_SNAPSHOT_CHARS,
    repeats: int = 1,
) -> list[dict]:
    """Run tasks and stream each result to `csv_path` as it completes.

    With concurrency=1 (default), results arrive in task order. With higher
    concurrency they arrive in completion order; the returned list is sorted
    by id so callers see a stable order regardless.

    When `repeats > 1` each task is run N times and the row is the aggregated
    form; the CSV switches to REPEAT_CSV_FIELDS, per-run detail lives only in
    the JSON's tasks[].per_run.
    """
    results: list[dict] = []
    sem = asyncio.Semaphore(concurrency)

    # Open the CSV once and flush each row so partial progress is visible
    # even if the run is killed or crashes mid-way. The header is written
    # immediately so `head`/`wc -l` work before any task finishes.
    csv_file = None
    writer = None
    csv_fields = REPEAT_CSV_FIELDS if repeats > 1 else CSV_FIELDS
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        csv_file.flush()

    async def bounded(task: dict) -> None:
        async with sem:
            row = await run_one_repeated(
                task,
                cfg,
                gate,
                repeats=repeats,
                timeout_s=timeout_s,
                snapshot_chars=snapshot_chars,
            )
            results.append(row)
            if writer is not None and csv_file is not None:
                writer.writerow(row)
                csv_file.flush()

    try:
        await asyncio.gather(*(bounded(t) for t in tasks))
    finally:
        if csv_file is not None:
            csv_file.close()

    results.sort(key=lambda r: str(r["id"]))
    return results


def _resolve_mode(args: argparse.Namespace, cfg: Config) -> str:
    """Return the effective vision mode string for naming/JSON metadata.

    Priority: --mode > --vision / --dom > cfg.vision_mode. The values
    dom_only / vision_enabled are aliases for dom / vision.
    """
    if args.mode == "dom_only":
        return "dom"
    if args.mode == "vision_enabled":
        return "vision"
    if args.mode in ("dom", "vision", "auto", "category"):
        return args.mode
    if args.vision:
        return "vision"
    if args.dom:
        return "dom"
    return cfg.vision_mode


def _apply_mode(cfg: Config, mode: str) -> Config:
    if mode in ("dom", "vision", "auto", "category"):
        return cfg.with_overrides(vision_mode=mode)
    return cfg


def _summarize(results: list[dict], meta: dict) -> dict:
    """Build the summary block written to benchmark_results.json.

    Handles both single-run rows (status/passed/elapsed_s) and aggregated
    repeat rows (runs/pass_count/mean_latency_s). The `pass_rate` is a
    weighted mean across tasks: for single-run rows, sum(passed)/sum(total);
    for repeat rows, sum(pass_count)/sum(runs). For repeat rows, latency
    and token fields are means so the per-task aggregate is itself an
    average — we re-average those to get a global mean.
    """
    total = len(results)
    is_aggregated = bool(results) and "runs" in results[0] and results[0]["runs"] > 1

    if is_aggregated:
        total_runs = sum(int(r.get("runs") or 0) for r in results)
        total_pass = sum(int(r.get("pass_count") or 0) for r in results)
        total_timeouts = sum(int(r.get("timeouts") or 0) for r in results)
        total_errors = sum(int(r.get("errors") or 0) for r in results)
        pass_rate = round(total_pass / total_runs * 100, 1) if total_runs else 0.0
        ok = total_runs - total_timeouts - total_errors
        elapsed = [float(r.get("mean_latency_s") or 0) for r in results]
        steps = [float(r.get("mean_steps") or 0) for r in results if r.get("mean_steps")]
        tokens_in = [float(r.get("mean_tokens_in") or 0) for r in results]
        tokens_out = [float(r.get("mean_tokens_out") or 0) for r in results]
        std_latency = [float(r.get("std_latency_s") or 0) for r in results]
    else:
        passed = sum(1 for r in results if r.get("passed") == "yes")
        ok = sum(1 for r in results if r.get("status") == "ok")
        total_timeouts = sum(1 for r in results if r.get("status") == "timeout")
        total_errors = sum(1 for r in results if r.get("status") == "error")
        total_pass = passed
        total_runs = total
        pass_rate = round(passed / total * 100, 1) if total else 0.0
        steps = [int(r["steps"]) for r in results if r.get("status") == "ok" and r.get("steps") not in (None, "")]
        elapsed = [float(r["elapsed_s"]) for r in results]
        tokens_in = [int(r.get("tokens_in") or 0) for r in results]
        tokens_out = [int(r.get("tokens_out") or 0) for r in results]
        std_latency = []

    has_expect = sum(1 for r in results if r.get("expect"))

    def _avg(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 1) if xs else 0.0

    out = {
        **meta,
        "total": total,
        "ok": ok,
        "timeouts": total_timeouts,
        "errors": total_errors,
        "passed": total_pass,
        "has_expect": has_expect,
        "pass_rate": pass_rate,
        "ok_rate": round(ok / total_runs * 100, 1) if total_runs else 0.0,
        "avg_steps": _avg([float(s) for s in steps]),
        "avg_latency_s": _avg(elapsed),
        "total_latency_s": round(sum(elapsed), 1),
        "avg_tokens_in": _avg(tokens_in),
        "avg_tokens_out": _avg(tokens_out),
        "total_tokens_in": int(sum(tokens_in)),
        "total_tokens_out": int(sum(tokens_out)),
    }
    if is_aggregated:
        out["repeats"] = meta.get("repeats", results[0].get("runs", 1) if results else 1)
        out["mean_std_latency_s"] = _avg(std_latency)
    return out


def write_json(results: list[dict], path: Path, meta: dict) -> None:
    """Write the per-task array + summary to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "summary": _summarize(results, meta), "tasks": results}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run browser-agent benchmark")
    parser.add_argument(
        "--mode",
        choices=["dom_only", "vision_enabled", "dom", "vision", "auto", "category"],
        default=None,
        help=(
            "Routing mode for the run. Aliases: dom_only≡dom, "
            "vision_enabled≡vision. `category` uses data-driven routing from "
            "Config.dom_categories (set DOM_CATEGORIES in .env). "
            "If omitted, falls back to --dom/--vision."
        ),
    )
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
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Per-task timeout in seconds (default {DEFAULT_TIMEOUT_S:.0f})",
    )
    parser.add_argument(
        "--snapshot-chars",
        type=int,
        default=DEFAULT_SNAPSHOT_CHARS,
        help=f"Max chars of final DOM snapshot to record (default {DEFAULT_SNAPSHOT_CHARS})",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path for the benchmark_results.json file (default: alongside CSV)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Run each task N times and aggregate (default 1 = single-run). "
            "Use ≥3 to dampen LLM sampling variance for A/B comparisons."
        ),
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
    mode = _resolve_mode(args, cfg)
    cfg = _apply_mode(cfg, mode)

    async def _auto_deny(action):
        return SafetyDecision(allow=False, reason="auto-denied (benchmark)")

    async def _auto_approve(action):
        return SafetyDecision(allow=True, reason="auto-approved (benchmark)")

    gate = ConfirmationGate(
        confirm=_auto_deny if args.gate == "auto-deny" else _auto_approve
    )

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = f"{mode}_x{args.repeats}" if args.repeats > 1 else mode
    out_csv = RESULTS_DIR / f"{ts}_{suffix}.csv"
    out_json = (
        Path(args.output_json)
        if args.output_json
        else RESULTS_DIR / f"{ts}_{suffix}.json"
    )

    print(
        f"Running {len(tasks)} tasks (mode={mode}, gate={args.gate}, "
        f"concurrency={args.concurrency}, timeout={args.timeout:.0f}s, "
        f"repeats={args.repeats})"
    )
    print(f"Results (incremental): {out_csv}")
    print(f"Results (aggregate JSON): {out_json}")
    results = asyncio.run(
        run_all(
            tasks,
            cfg,
            gate,
            concurrency=args.concurrency,
            csv_path=out_csv,
            timeout_s=args.timeout,
            snapshot_chars=args.snapshot_chars,
            repeats=args.repeats,
        )
    )

    # When repeats > 1, each `results[i]` is an aggregated row with a different
    # schema (pass_count/runs instead of passed/status). The post-run summary
    # below branches on `args.repeats` and reads only the columns that exist
    # in each mode. Anything we'd want to print for both modes (total time,
    # total tokens) is computed once above.
    total_s = sum(
        float(r.get("mean_latency_s") or r.get("elapsed_s") or 0) for r in results
    )
    avg_s = total_s / len(results) if results else 0
    if args.repeats > 1:
        total_in = sum(int(r.get("mean_tokens_in") or 0) for r in results)
        total_out = sum(int(r.get("mean_tokens_out") or 0) for r in results)
    else:
        total_in = sum(int(r.get("tokens_in") or 0) for r in results)
        total_out = sum(int(r.get("tokens_out") or 0) for r in results)

    meta = {
        "mode": mode,
        "timestamp": ts,
        "model": cfg.llm_model or "",
        "provider": cfg.provider,
        "concurrency": args.concurrency,
        "timeout_s": args.timeout,
        "gate": args.gate,
        "repeats": args.repeats,
        "total_elapsed_s": round(total_s, 1),
    }
    write_json(results, out_json, meta)

    if args.repeats > 1:
        # Aggregate-mode summary: pass_rate is a weighted mean across runs.
        total_runs = sum(int(r.get("runs") or 0) for r in results)
        total_pass = sum(int(r.get("pass_count") or 0) for r in results)
        agg_timeouts = sum(int(r.get("timeouts") or 0) for r in results)
        agg_errors = sum(int(r.get("errors") or 0) for r in results)
        agg_run_ok = total_runs - agg_timeouts - agg_errors
        agg_pass_rate = (total_pass / total_runs * 100) if total_runs else 0.0
        std_mean = _avg([float(r.get("std_latency_s") or 0) for r in results])
        print(
            f"\nDone.  {agg_run_ok}/{total_runs} run-ok  {agg_timeouts} timeouts  "
            f"{agg_errors} errors  {total_s:.1f}s total  {avg_s:.1f}s avg per task"
        )
        print(
            f"Tokens: {total_in:,} in / {total_out:,} out "
            f"(avg {total_in // max(total_runs, 1):,} in, "
            f"{total_out // max(total_runs, 1):,} out per run)"
        )
        print(
            f"Pass rate (mean over {args.repeats} repeats): "
            f"{total_pass}/{total_runs} = {agg_pass_rate:.1f}%  "
            f"mean within-task std: {std_mean:.1f}s"
        )
    else:
        ok = sum(1 for r in results if r["status"] == "ok")
        err = sum(1 for r in results if r["status"] == "error")
        timeouts = sum(1 for r in results if r["status"] == "timeout")
        passed = sum(1 for r in results if r["passed"] == "yes")
        print(
            f"\nDone.  {ok} ok  {timeouts} timeouts  {err} errors  "
            f"{total_s:.1f}s total  {avg_s:.1f}s avg"
        )
        print(
            f"Tokens: {total_in:,} in / {total_out:,} out "
            f"(avg {total_in // max(len(results), 1):,} in, "
            f"{total_out // max(len(results), 1):,} out per task)"
        )
        print(
            f"Passed expect-check: {passed}/{len(results)} "
            f"({passed / len(results) * 100:.1f}%)"
        )
        no_expect = sum(1 for r in results if not r.get("expect"))
        if no_expect:
            print(f"  ({no_expect} tasks had no `expect` field — counted as not passed)")
    print(f"CSV:   {out_csv}")
    print(f"JSON:  {out_json}")


if __name__ == "__main__":
    main()
