"""Benchmark analysis — compare runs and produce a report.

Usage:
    python -m benchmarks.analyze results/20250101_120000_dom.csv
    python -m benchmarks.analyze results/20250101_120000_dom.csv results/20250101_130000_vision.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def stats(rows: list[dict]) -> dict:
    ok = sum(1 for r in rows if r["status"] == "ok")
    err = sum(1 for r in rows if r["status"] == "error")
    total = len(rows)
    steps = [int(r["steps"]) for r in rows if r["status"] == "ok" and r["steps"]]
    elapsed = [float(r["elapsed_s"]) for r in rows]

    return {
        "total": total,
        "ok": ok,
        "errors": err,
        "success_rate": round(ok / total * 100, 1) if total else 0,
        "avg_steps": round(sum(steps) / len(steps), 1) if steps else 0,
        "min_steps": min(steps) if steps else 0,
        "max_steps": max(steps) if steps else 0,
        "total_s": round(sum(elapsed), 1),
        "avg_s": round(sum(elapsed) / len(elapsed), 1) if elapsed else 0,
        "min_s": round(min(elapsed), 1) if elapsed else 0,
        "max_s": round(max(elapsed), 1) if elapsed else 0,
    }


def by_category(rows: list[dict]) -> dict[str, dict]:
    cats: dict[str, list[dict]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    return {c: stats(rs) for c, rs in cats.items()}


def print_report(name: str, s: dict) -> None:
    print(f"\n── {name} ──")
    print(f"  Tasks:       {s['total']}")
    print(f"  Success:     {s['ok']} ({s['success_rate']}%)")
    print(f"  Errors:      {s['errors']}")
    print(f"  Steps:       avg {s['avg_steps']}  (min {s['min_steps']}, max {s['max_steps']})")
    print(f"  Latency:     avg {s['avg_s']}s  (min {s['min_s']}s, max {s['max_s']}s, total {s['total_s']}s)")


def print_comparison(a_name: str, a: dict, b_name: str, b: dict) -> None:
    print(f"\n── A/B Comparison: {a_name} vs {b_name} ──")
    print(f"  {'Metric':<20} {a_name:<12} {b_name:<12} {'Delta':<12}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12}")

    def row(label: str, va: object, vb: object, fmt: str = "s", delta_fmt: str = "s") -> None:
        if fmt == ".1f":
            sa, sb = f"{float(va):.1f}", f"{float(vb):.1f}"
            delta = float(vb) - float(va)
            sd = f"{delta:+.1f}" if delta_fmt == ".1f" else f"{delta:+d}"
        elif fmt == ".1f%%":
            sa, sb = f"{float(va):.1f}%", f"{float(vb):.1f}%"
            delta = float(vb) - float(va)
            sd = f"{delta:+.1f}pp"
        else:
            sa, sb = str(va), str(vb)
            delta = int(vb) - int(va) if isinstance(va, (int, float)) else 0
            sd = f"{delta:+d}"
        print(f"  {label:<20} {sa:<12} {sb:<12} {sd:<12}")

    row("Success rate", a["success_rate"], b["success_rate"], ".1f%%")
    row("Avg steps", a["avg_steps"], b["avg_steps"], ".1f")
    row("Avg latency (s)", a["avg_s"], b["avg_s"], ".1f")
    row("Total time (s)", a["total_s"], b["total_s"], ".1f")
    row("Errors", a["errors"], b["errors"], "d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("files", nargs="+", help="One or two CSV result files")
    parser.add_argument("--by-category", action="store_true", help="Break down by category")
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
                print(f"    [{cat}] {s['ok']}/{s['total']} ({s['success_rate']}%)  avg {s['avg_s']}s")

    if len(datasets) == 2:
        (a_name, a_rows), (b_name, b_rows) = datasets
        print_comparison(a_name, stats(a_rows), b_name, stats(b_rows))


if __name__ == "__main__":
    main()
