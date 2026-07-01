"""Benchmark analysis — compare runs and produce a report.

Usage:
    python -m benchmarks.analyze results/20250101_120000_dom.csv
    python -m benchmarks.analyze results/20250101_120000_dom.csv results/20250101_130000_vision.csv
    python -m benchmarks.analyze --by-category results/*_x3.csv

Handles both single-run CSVs (one row per task, `passed` column is "yes"/"no")
and aggregated repeat CSVs (one row per task, `pass_count` and `runs` columns).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _is_aggregated(rows: list[dict]) -> bool:
    """Detect whether the CSV is from a --repeats N run (aggregated form)."""
    return bool(rows) and "pass_count" in rows[0] and "runs" in rows[0]


def _pass_rate_for_row(r: dict) -> float:
    """Per-row pass rate: 0.0, 1.0, or 0-100 from pass_count/runs."""
    if "pass_count" in r and "runs" in r and int(r["runs"] or 0) > 0:
        return int(r["pass_count"]) / int(r["runs"])
    return 1.0 if r.get("passed") == "yes" else 0.0


def stats(rows: list[dict]) -> dict:
    total = len(rows)
    aggregated = _is_aggregated(rows)

    # Weighted pass rate: sum(pass_count)/sum(runs) for repeats, or
    # count(passed=="yes")/total for single-run. Using a weighted mean
    # here instead of mean(per_row) keeps tasks with more runs from
    # dominating the aggregate.
    if aggregated:
        total_runs = sum(int(r.get("runs") or 0) for r in rows)
        total_pass = sum(int(r.get("pass_count") or 0) for r in rows)
        total_timeouts = sum(int(r.get("timeouts") or 0) for r in rows)
        total_errors = sum(int(r.get("errors") or 0) for r in rows)
        passed = total_pass
        ok = total_runs - total_timeouts - total_errors
        steps = [float(r.get("mean_steps") or 0) for r in rows if r.get("mean_steps") not in (None, "")]
        elapsed = [float(r["mean_latency_s"]) for r in rows if r.get("mean_latency_s") not in (None, "")]
        tokens_in = [float(r.get("mean_tokens_in") or 0) for r in rows]
        tokens_out = [float(r.get("mean_tokens_out") or 0) for r in rows]
        std_lat = [float(r.get("std_latency_s") or 0) for r in rows]
    else:
        passed = sum(1 for r in rows if r.get("passed") == "yes")
        ok = sum(1 for r in rows if r.get("status") == "ok")
        total_timeouts = sum(1 for r in rows if r.get("status") == "timeout")
        total_errors = sum(1 for r in rows if r.get("status") == "error")
        total_runs = total
        steps_raw = [int(r["steps"]) for r in rows if r.get("status") == "ok" and r["steps"]]
        steps = [float(s) for s in steps_raw]
        elapsed = [float(r["elapsed_s"]) for r in rows]
        tokens_in = [int(r.get("tokens_in") or 0) for r in rows]
        tokens_out = [int(r.get("tokens_out") or 0) for r in rows]
        std_lat = []

    has_expect = sum(1 for r in rows if r.get("expect"))

    def _avg(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 1) if xs else 0.0

    pass_rate_pct = (passed / total_runs * 100) if total_runs else 0.0
    pass_rate_field = round(pass_rate_pct, 1)

    return {
        "total": total,
        "ok": ok,
        "errors": total_errors,
        "timeouts": total_timeouts,
        "success_rate": round(ok / total_runs * 100, 1) if total_runs else 0,
        "passed": passed,
        "total_runs": total_runs,
        "pass_rate": pass_rate_field,
        "has_expect": has_expect,
        "aggregated": aggregated,
        "steps": {
            "avg": _avg(steps),
            "min": min(steps) if steps else 0,
            "max": max(steps) if steps else 0,
        },
        "latency": {
            "total": round(sum(elapsed), 1),
            "avg": _avg(elapsed),
            "min": min(elapsed) if elapsed else 0,
            "max": max(elapsed) if elapsed else 0,
            "std": round(math.sqrt(sum((x - (sum(elapsed) / len(elapsed))) ** 2 for x in elapsed) / (len(elapsed) - 1)), 2) if len(elapsed) > 1 else 0,
        },
        "tokens": {
            "in_total": sum(tokens_in),
            "out_total": sum(tokens_out),
            "in_avg": _avg([float(t) for t in tokens_in]),
            "out_avg": _avg([float(t) for t in tokens_out]),
        },
        "within_task_latency_std": _avg(std_lat),
    }


def by_category(rows: list[dict]) -> dict[str, dict]:
    cats: dict[str, list[dict]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    return {c: stats(rs) for c, rs in cats.items()}


def print_report(name: str, s: dict) -> None:
    print(f"\n── {name} ──")
    print(f"  Tasks:       {s['total']}")
    if s["aggregated"]:
        print(f"  Runs:        {s['total_runs']}  (aggregated across {s['total_runs'] // s['total'] if s['total'] else 0} repeats per task)")
    print(f"  Ok (ran):    {s['ok']} ({s['success_rate']}%)")
    print(f"  Passed:      {s['passed']}/{s['total_runs']} ({s['pass_rate']}%)  ← expect-check")
    print(f"  Errors:      {s['errors']}    Timeouts: {s['timeouts']}")
    print(
        f"  Steps:       avg {s['steps']['avg']}  "
        f"(min {s['steps']['min']}, max {s['steps']['max']})"
    )
    print(
        f"  Latency:     avg {s['latency']['avg']}s  "
        f"(min {s['latency']['min']}s, max {s['latency']['max']}s, "
        f"total {s['latency']['total']}s, std {s['latency']['std']}s)"
    )
    if s["aggregated"] and s["within_task_latency_std"]:
        print(f"  Within-task std:  {s['within_task_latency_std']}s")
    print(
        f"  Tokens:      {s['tokens']['in_total']:,} in / {s['tokens']['out_total']:,} out "
        f"(avg {s['tokens']['in_avg']:,} in / {s['tokens']['out_avg']:,} out per task)"
    )


def print_comparison(a_name: str, a: dict, b_name: str, b: dict) -> None:
    print(f"\n── A/B Comparison: {a_name} vs {b_name} ──")
    print(f"  {'Metric':<22} {a_name:<14} {b_name:<14} {'Delta':<12}")
    print(f"  {'─'*22} {'─'*14} {'─'*14} {'─'*12}")

    def row(label: str, va: object, vb: object, fmt: str = "s") -> None:
        if fmt == ".1f":
            sa, sb = f"{float(va):.1f}", f"{float(vb):.1f}"
            delta = float(vb) - float(va)
            sd = f"{delta:+.1f}"
        elif fmt == ".1f%%":
            sa, sb = f"{float(va):.1f}%", f"{float(vb):.1f}%"
            delta = float(vb) - float(va)
            sd = f"{delta:+.1f}pp"
        else:
            sa, sb = str(va), str(vb)
            delta = int(vb) - int(va) if isinstance(va, (int, float)) else 0
            sd = f"{delta:+d}"
        print(f"  {label:<22} {sa:<14} {sb:<14} {sd:<12}")

    row("Pass rate (expect)", a["pass_rate"], b["pass_rate"], ".1f%%")
    row("Success rate (ok)", a["success_rate"], b["success_rate"], ".1f%%")
    row("Avg steps", a["steps"]["avg"], b["steps"]["avg"], ".1f")
    row("Avg latency (s)", a["latency"]["avg"], b["latency"]["avg"], ".1f")
    row("Total time (s)", a["latency"]["total"], b["latency"]["total"], ".1f")
    row("Errors", a["errors"], b["errors"], "d")
    row("Timeouts", a["timeouts"], b["timeouts"], "d")


def per_category_delta(
    a_rows: list[dict], b_rows: list[dict], *, n_bootstrap: int = 1000
) -> list[dict]:
    """Compute per-category pass-rate delta with bootstrap CIs.

    Each returned dict has: category, runs_a, pass_a, rate_a, runs_b, pass_b,
    rate_b, delta_pp, ci_low_pp, ci_high_pp, suggestion. The CIs are on the
    delta (b - a) at 95%. `suggestion` is one of "vision", "dom", "tie",
    "noise" — a routing recommendation when the CI excludes zero:
      - vision: delta CI is strictly positive (vision wins)
      - dom:    delta CI is strictly negative (DOM wins)
      - tie:    CI includes zero but the point estimate is non-trivial
      - noise:  small sample + CI includes zero (no signal)
    """
    import random
    from collections import defaultdict

    def per_task_pass(rows: list[dict]) -> dict[str, list[int]]:
        out: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            n = int(r.get("runs") or 1)
            p = int(r.get("pass_count") or (1 if r.get("passed") == "yes" else 0))
            out[r["category"]].extend([1] * p + [0] * (n - p))
        return out

    a = per_task_pass(a_rows)
    b = per_task_pass(b_rows)
    cats = sorted(set(a) | set(b))
    out = []
    rng = random.Random(0xC0FFEE)
    for cat in cats:
        a_passes = a.get(cat, [])
        b_passes = b.get(cat, [])
        n_a, n_b = len(a_passes), len(b_passes)
        rate_a = (sum(a_passes) / n_a * 100) if n_a else 0.0
        rate_b = (sum(b_passes) / n_b * 100) if n_b else 0.0
        delta = rate_b - rate_a

        # Bootstrap CI on the delta: resample both per-task lists with
        # replacement, recompute the delta each time. The 2.5/97.5
        # percentiles of the resampled deltas form the 95% CI.
        if n_a >= 2 and n_b >= 2:
            deltas: list[float] = []
            for _ in range(n_bootstrap):
                rs_a = [a_passes[rng.randrange(n_a)] for _ in range(n_a)]
                rs_b = [b_passes[rng.randrange(n_b)] for _ in range(n_b)]
                deltas.append(sum(rs_b) / n_b * 100 - sum(rs_a) / n_a * 100)
            deltas.sort()
            ci_low = round(deltas[int(0.025 * n_bootstrap)], 1)
            ci_high = round(deltas[int(0.975 * n_bootstrap) - 1], 1)
        else:
            ci_low = ci_high = 0.0

        if n_a + n_b < 6 or (ci_low <= 0 <= ci_high):
            suggestion = "noise" if n_a + n_b < 6 else "tie"
        elif ci_low > 0:
            suggestion = "vision"
        elif ci_high < 0:
            suggestion = "dom"
        else:
            suggestion = "tie"

        out.append(
            {
                "category": cat,
                "runs_a": n_a,
                "pass_a": sum(a_passes),
                "rate_a": round(rate_a, 1),
                "runs_b": n_b,
                "pass_b": sum(b_passes),
                "rate_b": round(rate_b, 1),
                "delta_pp": round(delta, 1),
                "ci_low_pp": ci_low,
                "ci_high_pp": ci_high,
                "suggestion": suggestion,
            }
        )
    return out


def print_per_category_delta(
    a_name: str, b_name: str, rows: list[dict]
) -> None:
    print(f"\n── Per-category delta: {a_name} → {b_name} ──")
    print(
        f"  {'category':<14} {a_name:>10} {b_name:>10} "
        f"{'Δ pp':>8} {'95% CI':>16}  {'suggestion':>10}"
    )
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*8} {'─'*16}  {'─'*10}")
    for r in rows:
        ci = f"[{r['ci_low_pp']:+5.1f}, {r['ci_high_pp']:+5.1f}]"
        print(
            f"  {r['category']:<14} {r['rate_a']:>9.1f}% {r['rate_b']:>9.1f}% "
            f"{r['delta_pp']:>+7.1f} {ci:>16}  {r['suggestion']:>10}"
        )

    routing = [r for r in rows if r["suggestion"] in ("vision", "dom")]
    if routing:
        dom_routed = [r["category"] for r in routing if r["suggestion"] == "dom"]
        vis_routed = [r["category"] for r in routing if r["suggestion"] == "vision"]
        print()
        print("  Suggested routing rule:")
        if vis_routed:
            print(f"    vision (CI excludes 0, vision better): {', '.join(vis_routed)}")
        if dom_routed:
            print(f"    dom    (CI excludes 0, DOM better):    {', '.join(dom_routed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("files", nargs="+", help="One or two CSV result files")
    parser.add_argument("--by-category", action="store_true", help="Break down by category")
    parser.add_argument(
        "--per-category-delta",
        action="store_true",
        help=(
            "Compute per-category pass-rate delta with 95% bootstrap CI. "
            "Use with two repeat CSVs (e.g. *_x3.csv from --repeats runs) "
            "to drive the per-category routing rule."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples for the CI (default 1000)",
    )
    args = parser.parse_args()

    paths = [Path(f) for f in args.files]
    for p in paths:
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    datasets = [(p.stem, load_csv(p)) for p in paths]

    for name, rows in datasets:
        print_report(name, stats(rows))
        if args.by_category:
            for cat, s in by_category(rows).items():
                print(
                    f"    [{cat}] {s['ok']}/{s['total_runs']} ok, "
                    f"{s['passed']}/{s['total_runs']} passed "
                    f"({s['pass_rate']}%)  avg {s['latency']['avg']}s"
                )

    if len(datasets) == 2:
        (a_name, a_rows), (b_name, b_rows) = datasets
        print_comparison(a_name, stats(a_rows), b_name, stats(b_rows))
        if args.per_category_delta:
            delta_rows = per_category_delta(
                a_rows, b_rows, n_bootstrap=args.bootstrap_samples
            )
            print_per_category_delta(a_name, b_name, delta_rows)


if __name__ == "__main__":
    main()
