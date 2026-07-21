"""Tests for the benchmark harness: token counter, timeout, JSON writer, mode aliasing.

Hermetic — no live LLM calls. We stub the chat model and the adapter.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from browser_use.llm.base import ChatInvokeCompletion
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeUsage

from benchmarks import run as bench
from browser_agent.agent.loop import _VisionOnlyAdapter
from browser_agent.config import Config
from browser_agent.models.base import ModelAdapter
from browser_agent.models.token_counter import TokenCountingChatModel, TokenTotals
from browser_agent.safety.gate import ConfirmationGate
from browser_agent.safety.types import SafetyDecision

# ── Token counter ────────────────────────────────────────────────────────


class _StubChat:
    """Minimal chat-model stub that returns a fixed completion + usage."""

    def __init__(self, usage: ChatInvokeUsage | None) -> None:
        self._usage = usage
        self.calls = 0

    @property
    def provider(self) -> str:
        return "stub"

    @property
    def name(self) -> str:
        return "stub"

    @property
    def model_name(self) -> str:
        return "stub-model"

    @property
    def model(self) -> str:
        return "stub-model"

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[Any]:
        self.calls += 1
        return ChatInvokeCompletion(
            completion="ok",
            usage=self._usage,
        )


def _make_usage(in_t: int, out_t: int) -> ChatInvokeUsage:
    return ChatInvokeUsage(
        prompt_tokens=in_t,
        prompt_cached_tokens=None,
        prompt_cache_creation_tokens=None,
        prompt_image_tokens=None,
        completion_tokens=out_t,
        total_tokens=in_t + out_t,
    )


def test_token_counter_extracts_openai_usage():
    async def _run():
        inner = _StubChat(_make_usage(100, 25))
        wrapper = TokenCountingChatModel(inner)
        completion = await wrapper.ainvoke([])
        assert completion.completion == "ok"
        assert wrapper.totals.input == 100
        assert wrapper.totals.output == 25
        assert wrapper.totals.calls == 1

    asyncio.run(_run())


def test_token_counter_accumulates_across_calls():
    async def _run():
        inner = _StubChat(_make_usage(50, 10))
        wrapper = TokenCountingChatModel(inner)
        for _ in range(3):
            await wrapper.ainvoke([])
        assert wrapper.totals.input == 150
        assert wrapper.totals.output == 30
        assert wrapper.totals.calls == 3
        assert inner.calls == 3

    asyncio.run(_run())


def test_token_counter_handles_missing_usage():
    async def _run():
        inner = _StubChat(usage=None)
        wrapper = TokenCountingChatModel(inner)
        await wrapper.ainvoke([])
        assert wrapper.totals.input == 0
        assert wrapper.totals.output == 0
        assert wrapper.totals.calls == 1  # call still counted

    asyncio.run(_run())


def test_token_counter_passes_through_model_name():
    inner = _StubChat(_make_usage(1, 1))
    wrapper = TokenCountingChatModel(inner)
    assert wrapper.model_name == "stub-model"
    assert wrapper.provider == "stub"


# ── ModelAdapter.with_token_counter default ───────────────────────────────


class _StubAdapter(ModelAdapter):
    """Adapter that exposes a stub chat model through the default helper."""

    name = "stub"
    supports_vision = False

    def __init__(self, usage: ChatInvokeUsage | None) -> None:
        self._chat = _StubChat(usage)

    def chat_model(self) -> _StubChat:
        return self._chat


def test_adapter_with_token_counter_wraps_default_chat_model():
    async def _run():
        adapter = _StubAdapter(_make_usage(7, 3))
        wrapped, totals = adapter.with_token_counter()
        assert isinstance(wrapped, TokenCountingChatModel)
        assert isinstance(totals, TokenTotals)
        await wrapped.ainvoke([])
        assert totals.input == 7
        assert totals.output == 3

    asyncio.run(_run())


# ── run_one timeout ───────────────────────────────────────────────────────


def test_run_one_returns_timeout_row():
    """A hanging chat model should produce status='timeout', not crash."""

    # Monkey-patch run_task_with_model inside the benchmark module so the
    # test runs in <1s regardless of the configured timeout.
    from benchmarks import run as bench_mod

    async def _fake_run_task_with_model(task_text, **kwargs: Any) -> object:
        await asyncio.sleep(60)
        return _FakeHistory(steps=0, final="")

    original = bench_mod.run_task_with_model
    bench_mod.run_task_with_model = _fake_run_task_with_model  # type: ignore[assignment]
    try:
        cfg = Config(llm_api_key="sk-test", llm_model="m", vision_mode="dom")

        async def _dummy_confirm(action):
            return SafetyDecision(allow=True, reason="ok")

        gate = ConfirmationGate(confirm=_dummy_confirm)
        task = {
            "id": "hang-01",
            "category": "test",
            "task": "this will hang",
            "expect": "",
        }

        async def _drive():
            return await bench.run_one(task, cfg, gate, timeout_s=0.05, snapshot_chars=10)

        row = asyncio.run(_drive())
        assert row["status"] == "timeout"
        assert row["passed"] == "no"
        assert "timed out" in row["error"]
        assert row["steps"] == 0
    finally:
        bench_mod.run_task_with_model = original  # type: ignore[assignment]


class _FakeHistory:
    def __init__(self, steps: int, final: str) -> None:
        self.history = [object()] * steps
        self._final = final

    def final_result(self) -> str:
        return self._final


# ── write_json schema ─────────────────────────────────────────────────────


def test_write_json_emits_meta_summary_and_tasks():
    results = [
        {
            "id": "a",
            "category": "nav",
            "status": "ok",
            "passed": "yes",
            "expect": "x",
            "prompt": "go to a",
            "result": "x",
            "steps": 2,
            "elapsed_s": 1.5,
            "tokens_in": 100,
            "tokens_out": 20,
            "final_url": "https://a",
            "dom_snapshot_excerpt": "<h1>x</h1>",
            "error": "",
        },
        {
            "id": "b",
            "category": "nav",
            "status": "timeout",
            "passed": "no",
            "expect": "y",
            "prompt": "go to b",
            "result": "",
            "steps": 0,
            "elapsed_s": 30.0,
            "tokens_in": 50,
            "tokens_out": 0,
            "final_url": "",
            "dom_snapshot_excerpt": "",
            "error": "timed out after 30s",
        },
    ]
    meta = {
        "mode": "dom",
        "timestamp": "20260101_000000",
        "model": "test-model",
        "provider": "stub",
        "concurrency": 1,
        "timeout_s": 30,
        "gate": "auto-deny",
        "total_elapsed_s": 31.5,
    }
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "results.json"
        bench.write_json(results, out, meta)
        data = json.loads(out.read_text())

    assert data["meta"] == meta
    assert data["summary"]["total"] == 2
    assert data["summary"]["ok"] == 1
    assert data["summary"]["timeouts"] == 1
    assert data["summary"]["errors"] == 0
    assert data["summary"]["passed"] == 1
    assert data["summary"]["pass_rate"] == 50.0
    assert data["summary"]["avg_latency_s"] == 15.8  # (1.5 + 30.0) / 2
    assert data["summary"]["total_tokens_in"] == 150
    assert data["summary"]["total_tokens_out"] == 20
    assert len(data["tasks"]) == 2


def test_write_json_handles_empty_results():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "empty.json"
        bench.write_json([], out, {"mode": "dom"})
        data = json.loads(out.read_text())
    assert data["summary"]["total"] == 0
    assert data["summary"]["pass_rate"] == 0.0
    assert data["tasks"] == []


# ── Mode resolution / aliases ─────────────────────────────────────────────


def test_resolve_mode_dom_only_aliases_to_dom():
    args = argparse.Namespace(mode="dom_only", dom=False, vision=False)
    cfg = Config()
    assert bench._resolve_mode(args, cfg) == "dom"


def test_resolve_mode_vision_enabled_aliases_to_vision():
    args = argparse.Namespace(mode="vision_enabled", dom=False, vision=False)
    cfg = Config()
    assert bench._resolve_mode(args, cfg) == "vision"


def test_resolve_mode_falls_back_to_dom_flag():
    args = argparse.Namespace(mode=None, dom=True, vision=False)
    cfg = Config(vision_mode="auto")
    assert bench._resolve_mode(args, cfg) == "dom"


def test_resolve_mode_falls_back_to_vision_flag():
    args = argparse.Namespace(mode=None, dom=False, vision=True)
    cfg = Config(vision_mode="auto")
    assert bench._resolve_mode(args, cfg) == "vision"


def test_resolve_mode_defaults_to_cfg_vision_mode():
    args = argparse.Namespace(mode=None, dom=False, vision=False)
    cfg = Config(vision_mode="auto")
    assert bench._resolve_mode(args, cfg) == "auto"


def test_apply_mode_overrides_vision_mode():
    cfg = Config(vision_mode="auto")
    out = bench._apply_mode(cfg, "dom")
    assert out.vision_mode == "dom"
    # with_overrides must return a new copy; the original is untouched.
    assert cfg.vision_mode == "auto"


# ── Vision shim ───────────────────────────────────────────────────────────


def test_vision_only_adapter_reports_supports_vision():
    shim = _VisionOnlyAdapter(supports_vision=True)
    assert shim.supports_vision is True
    assert shim.name == "_benchmark_shim"


# ── Repeat aggregation ────────────────────────────────────────────────────


def _per_run_row(*, passed: str, elapsed_s: float, status: str = "ok",
                 tokens_in: int = 0, tokens_out: int = 0, steps: int = 0,
                 result: str = "") -> dict:
    return {
        "id": "x",
        "category": "test",
        "status": status,
        "passed": passed,
        "expect": "EX",
        "prompt": "p",
        "result": result,
        "steps": steps,
        "elapsed_s": elapsed_s,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "final_url": "https://x",
        "dom_snapshot_excerpt": "<h1>x</h1>",
        "error": "",
    }


def test_aggregate_aggregates_pass_count_and_rate():
    task = {"id": "x", "category": "test", "task": "p", "expect": "EX"}
    per_run = [
        _per_run_row(passed="yes", elapsed_s=10.0),
        _per_run_row(passed="yes", elapsed_s=20.0),
        _per_run_row(passed="no", elapsed_s=30.0),
        _per_run_row(passed="no", elapsed_s=40.0),
    ]
    agg = bench._aggregate(task, per_run)
    assert agg["runs"] == 4
    assert agg["pass_count"] == 2
    assert agg["pass_rate"] == 50.0
    assert agg["mean_latency_s"] == 25.0
    assert agg["std_latency_s"] > 0  # non-zero for n>=2
    assert agg["timeouts"] == 0
    assert agg["errors"] == 0
    assert len(agg["per_run"]) == 4


def test_aggregate_picks_first_passing_result_as_best():
    task = {"id": "x", "category": "test", "task": "p", "expect": "EX"}
    per_run = [
        _per_run_row(passed="no", elapsed_s=10.0, result="wrong"),
        _per_run_row(passed="yes", elapsed_s=20.0, result="CORRECT"),
        _per_run_row(passed="yes", elapsed_s=30.0, result="also-correct"),
    ]
    agg = bench._aggregate(task, per_run)
    assert agg["best_result"] == "CORRECT"
    assert agg["pass_count"] == 2


def test_aggregate_falls_back_to_last_result_when_none_passed():
    task = {"id": "x", "category": "test", "task": "p", "expect": "EX"}
    per_run = [
        _per_run_row(passed="no", elapsed_s=10.0, result="wrong1"),
        _per_run_row(passed="no", elapsed_s=20.0, result="wrong2"),
    ]
    agg = bench._aggregate(task, per_run)
    assert agg["best_result"] == "wrong2"
    assert agg["pass_count"] == 0
    assert agg["pass_rate"] == 0.0


def test_aggregate_counts_timeouts_and_errors():
    task = {"id": "x", "category": "test", "task": "p", "expect": "EX"}
    per_run = [
        _per_run_row(passed="no", elapsed_s=10.0, status="ok"),
        _per_run_row(passed="no", elapsed_s=300.0, status="timeout"),
        _per_run_row(passed="no", elapsed_s=0.0, status="error"),
    ]
    agg = bench._aggregate(task, per_run)
    assert agg["timeouts"] == 1
    assert agg["errors"] == 1


def test_run_one_repeated_with_repeats_1_returns_run_one_shape():
    """repeats=1 must keep the legacy schema (no runs/per_run fields)."""

    from benchmarks import run as bench_mod

    async def _fake_run_one(task, cfg, gate, **kwargs):
        return _per_run_row(passed="yes", elapsed_s=12.0)

    original = bench_mod.run_one
    bench_mod.run_one = _fake_run_one  # type: ignore[assignment]
    try:
        cfg = Config(llm_api_key="sk", llm_model="m", vision_mode="dom")

        async def _dummy_confirm(action):
            return SafetyDecision(allow=True, reason="ok")

        gate = ConfirmationGate(confirm=_dummy_confirm)
        task = {"id": "x", "category": "test", "task": "p", "expect": "EX"}

        async def _drive():
            return await bench.run_one_repeated(task, cfg, gate, repeats=1)

        row = asyncio.run(_drive())
        # Legacy schema: no runs/per_run aggregation
        assert "runs" not in row
        assert "per_run" not in row
        assert row["status"] == "ok"
    finally:
        bench_mod.run_one = original  # type: ignore[assignment]


def test_run_one_repeated_with_repeats_3_aggregates():
    """repeats=3 must call run_one 3x and return aggregated form."""

    from benchmarks import run as bench_mod

    call_count = {"n": 0}

    async def _fake_run_one(task, cfg, gate, **kwargs):
        call_count["n"] += 1
        # alternate pass/fail/pass to make the aggregate non-trivial
        passed = "yes" if call_count["n"] in (1, 3) else "no"
        return _per_run_row(passed=passed, elapsed_s=float(call_count["n"]) * 10)

    original = bench_mod.run_one
    bench_mod.run_one = _fake_run_one  # type: ignore[assignment]
    try:
        cfg = Config(llm_api_key="sk", llm_model="m", vision_mode="dom")

        async def _dummy_confirm(action):
            return SafetyDecision(allow=True, reason="ok")

        gate = ConfirmationGate(confirm=_dummy_confirm)
        task = {"id": "x", "category": "test", "task": "p", "expect": "EX"}

        async def _drive():
            return await bench.run_one_repeated(task, cfg, gate, repeats=3)

        row = asyncio.run(_drive())
        assert call_count["n"] == 3
        assert row["runs"] == 3
        assert row["pass_count"] == 2
        assert row["pass_rate"] == round(2 / 3 * 100, 1)
        assert row["mean_latency_s"] == 20.0  # (10+20+30)/3
        assert len(row["per_run"]) == 3
    finally:
        bench_mod.run_one = original  # type: ignore[assignment]


# ── Summarize / analyze schema detection ───────────────────────────────────


def test_summarize_aggregated_uses_weighted_pass_rate():
    """When the rows have runs/pass_count, _summarize must compute the
    weighted pass rate (sum(pass_count) / sum(runs)), not a mean of
    per-task rates — otherwise tasks with more runs dominate."""
    rows = [
        {  # task A: 4 runs, 4 passes
            "id": "a", "category": "n", "runs": 4, "pass_count": 4,
            "pass_rate": 100.0, "mean_latency_s": 10.0, "std_latency_s": 0.0,
            "mean_tokens_in": 100, "mean_tokens_out": 10, "mean_steps": 2.0,
            "timeouts": 0, "errors": 0, "expect": "x", "best_result": "x",
            "final_url": "", "dom_snapshot_excerpt": "",
        },
        {  # task B: 4 runs, 0 passes
            "id": "b", "category": "n", "runs": 4, "pass_count": 0,
            "pass_rate": 0.0, "mean_latency_s": 20.0, "std_latency_s": 0.0,
            "mean_tokens_in": 200, "mean_tokens_out": 20, "mean_steps": 3.0,
            "timeouts": 0, "errors": 0, "expect": "y", "best_result": "",
            "final_url": "", "dom_snapshot_excerpt": "",
        },
    ]
    meta = {"mode": "dom", "repeats": 4}
    s = bench._summarize(rows, meta)
    # Weighted: 4/8 = 50%. Per-row mean would also be 50% here so we
    # also assert the totals are right.
    assert s["pass_rate"] == 50.0
    assert s["total"] == 2  # 2 distinct tasks
    assert s["passed"] == 4  # cumulative pass count across all runs
    assert s["repeats"] == 4
    assert s["timeouts"] == 0
    assert s["errors"] == 0


def test_summarize_legacy_single_run_shape_still_works():
    rows = [
        {
            "id": "a", "category": "n", "status": "ok", "passed": "yes",
            "expect": "x", "elapsed_s": 10.0, "tokens_in": 100,
            "tokens_out": 10, "steps": 2,
        },
        {
            "id": "b", "category": "n", "status": "ok", "passed": "no",
            "expect": "y", "elapsed_s": 20.0, "tokens_in": 200,
            "tokens_out": 20, "steps": 3,
        },
    ]
    meta = {"mode": "dom", "repeats": 1}
    s = bench._summarize(rows, meta)
    assert s["pass_rate"] == 50.0
    assert s["total"] == 2
    assert s["passed"] == 1
    assert s["ok"] == 2
    assert "repeats" not in s or s.get("repeats") == 1


# ── analyze.py schema detection ──────────────────────────────────────────


def test_analyze_stats_handles_legacy_single_run_csv(tmp_path):
    from benchmarks.analyze import stats

    csv_path = tmp_path / "legacy.csv"
    csv_path.write_text(
        "id,category,status,passed,expect,prompt,result,steps,elapsed_s,tokens_in,tokens_out,final_url,dom_snapshot_excerpt,error\n"
        "a,n,ok,yes,x,p,r,2,10,100,10,,,\n"
        "b,n,ok,no,y,p,r,3,20,200,20,,,\n"
    )
    import csv as _csv
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    s = stats(rows)
    assert s["aggregated"] is False
    assert s["total"] == 2
    assert s["pass_rate"] == 50.0
    assert s["passed"] == 1


def test_analyze_stats_handles_aggregated_repeat_csv(tmp_path):
    from benchmarks.analyze import stats

    csv_path = tmp_path / "agg.csv"
    csv_path.write_text(
        "id,category,runs,pass_count,pass_rate,mean_latency_s,std_latency_s,mean_tokens_in,mean_tokens_out,mean_steps,timeouts,errors,expect,prompt,best_result,final_url,dom_snapshot_excerpt\n"
        "a,n,3,2,66.7,10,1,100,10,2,0,0,x,p,r,,,\n"
        "b,n,3,1,33.3,20,2,200,20,3,0,0,y,p,,,,\n"
    )
    import csv as _csv
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    s = stats(rows)
    assert s["aggregated"] is True
    assert s["total"] == 2
    assert s["total_runs"] == 6
    assert s["passed"] == 3
    assert s["pass_rate"] == 50.0  # weighted: 3/6
    assert s["timeouts"] == 0


# ── DOM snapshot via step callback ─────────────────────────────────────────


class _FakeDomState:
    def __init__(self, text: str) -> None:
        self._text = text

    def llm_representation(self) -> str:
        return self._text


class _FakeBrowserStateSummary:
    def __init__(self, url: str, dom_text: str) -> None:
        self.url = url
        self.dom_state = _FakeDomState(dom_text)


def test_step_state_callback_captures_dom_and_url():
    holder, callback = asyncio.run(bench._make_step_state_holder(snapshot_chars=200))
    # Simulate two steps: navigate, then read more.
    callback(_FakeBrowserStateSummary("https://a.example/", "<h1>page A</h1>"), None, 1)
    callback(_FakeBrowserStateSummary("https://b.example/", "<h1>page B</h1>"), None, 2)
    # Latest URL wins; latest DOM wins (truncated to 200).
    assert holder["url"] == "https://b.example/"
    assert holder["dom"] == "<h1>page B</h1>"


def test_step_state_callback_truncates_dom_to_snapshot_chars():
    long_dom = "x" * 1000
    holder, callback = asyncio.run(bench._make_step_state_holder(snapshot_chars=50))
    callback(_FakeBrowserStateSummary("https://a/", long_dom), None, 1)
    assert len(holder["dom"]) == 50


def test_step_state_callback_handles_missing_attributes():
    """A bad step (None fields, missing dom_state) must not crash the run."""

    class _WeirdSummary:
        url = None
        # No `dom_state` attribute at all.

    holder, callback = asyncio.run(bench._make_step_state_holder(snapshot_chars=200))
    # No exception expected even with a degenerate summary.
    callback(_WeirdSummary(), None, 1)
    assert holder["url"] == ""
    assert holder["dom"] == ""


def test_step_state_callback_handles_dom_text_exception():
    """If `llm_representation` itself raises, swallow it (defensive)."""

    class _BoomDom:
        def llm_representation(self) -> str:
            raise RuntimeError("boom")

    class _Summary:
        url = "https://x/"
        dom_state = _BoomDom()

    holder, callback = asyncio.run(bench._make_step_state_holder(snapshot_chars=200))
    callback(_Summary(), None, 1)
    assert holder["url"] == "https://x/"
    assert holder["dom"] == ""  # exception swallowed


def test_run_one_populates_dom_excerpt_via_callback():
    """End-to-end: stub run_task_with_model to call the callback, verify the
    resulting row has a non-empty dom_snapshot_excerpt and final_url."""

    from benchmarks import run as bench_mod

    async def _fake_run_task_with_model(task_text, **kwargs):
        on_step_state = kwargs.get("on_step_state")
        if on_step_state is not None:
            on_step_state(
                _FakeBrowserStateSummary(
                    "https://captured.example/",
                    "<h1>Hello</h1><p>captured during agent run</p>",
                ),
                None,
                1,
            )
            on_step_state(
                _FakeBrowserStateSummary(
                    "https://captured2.example/",
                    "<h1>Final</h1>",
                ),
                None,
                2,
            )
        return _FakeHistory(steps=2, final="Final result")

    original = bench_mod.run_task_with_model
    bench_mod.run_task_with_model = _fake_run_task_with_model  # type: ignore[assignment]
    try:
        cfg = Config(llm_api_key="sk", llm_model="m", vision_mode="dom")

        async def _dummy_confirm(action):
            return SafetyDecision(allow=True, reason="ok")

        gate = ConfirmationGate(confirm=_dummy_confirm)
        task = {
            "id": "dom-01",
            "category": "test",
            "task": "go somewhere",
            "expect": "Final result",
        }

        async def _drive():
            return await bench.run_one(task, cfg, gate, snapshot_chars=500)

        row = asyncio.run(_drive())
        assert row["status"] == "ok"
        assert row["passed"] == "yes"
        # Last step's URL and DOM should be in the row.
        assert row["final_url"] == "https://captured2.example/"
        assert row["dom_snapshot_excerpt"] == "<h1>Final</h1>"
    finally:
        bench_mod.run_task_with_model = original  # type: ignore[assignment]


def test_run_one_timeout_row_has_empty_dom_excerpt():
    """A timeout means no steps ran, so dom_snapshot_excerpt stays empty."""

    from benchmarks import run as bench_mod

    async def _fake_run_task_with_model(task_text, **kwargs):
        await asyncio.sleep(60)
        return _FakeHistory(steps=0, final="")

    original = bench_mod.run_task_with_model
    bench_mod.run_task_with_model = _fake_run_task_with_model  # type: ignore[assignment]
    try:
        cfg = Config(llm_api_key="sk", llm_model="m", vision_mode="dom")

        async def _dummy_confirm(action):
            return SafetyDecision(allow=True, reason="ok")

        gate = ConfirmationGate(confirm=_dummy_confirm)
        task = {
            "id": "hang-02",
            "category": "test",
            "task": "this will hang",
            "expect": "",
        }

        async def _drive():
            return await bench.run_one(task, cfg, gate, timeout_s=0.05)

        row = asyncio.run(_drive())
        assert row["status"] == "timeout"
        assert row["dom_snapshot_excerpt"] == ""
        assert row["final_url"] == ""
    finally:
        bench_mod.run_task_with_model = original  # type: ignore[assignment]


# ── Per-category delta with bootstrap CI ───────────────────────────────────


def test_per_category_delta_flags_strong_signal_as_vision():
    """Vision significantly better than DOM in a category → suggestion='vision'."""
    from benchmarks.analyze import per_category_delta

    # 5 tasks in 'a', each with 10 runs. Vision passes 9/10, DOM passes 1/10.
    a_rows = [{"category": "a", "runs": 10, "pass_count": 1} for _ in range(5)]
    b_rows = [{"category": "a", "runs": 10, "pass_count": 9} for _ in range(5)]
    out = per_category_delta(a_rows, b_rows, n_bootstrap=500)
    assert len(out) == 1
    row = out[0]
    assert row["category"] == "a"
    assert row["rate_a"] == 10.0
    assert row["rate_b"] == 90.0
    assert row["delta_pp"] == 80.0
    assert row["suggestion"] == "vision"
    assert row["ci_low_pp"] > 0  # CI excludes zero


def test_per_category_delta_flags_strong_dom_signal():
    from benchmarks.analyze import per_category_delta

    a_rows = [{"category": "b", "runs": 10, "pass_count": 9} for _ in range(5)]
    b_rows = [{"category": "b", "runs": 10, "pass_count": 1} for _ in range(5)]
    out = per_category_delta(a_rows, b_rows, n_bootstrap=500)
    row = out[0]
    assert row["delta_pp"] == -80.0
    assert row["suggestion"] == "dom"
    assert row["ci_high_pp"] < 0


def test_per_category_delta_flags_overlapping_ci_as_tie():
    from benchmarks.analyze import per_category_delta

    a_rows = [{"category": "c", "runs": 4, "pass_count": 2} for _ in range(4)]
    b_rows = [{"category": "c", "runs": 4, "pass_count": 3} for _ in range(4)]
    out = per_category_delta(a_rows, b_rows, n_bootstrap=200)
    row = out[0]
    assert row["ci_low_pp"] <= 0 <= row["ci_high_pp"]
    assert row["suggestion"] in ("tie", "vision")  # bootstrap can flip


def test_per_category_delta_tiny_sample_marked_as_noise():
    from benchmarks.analyze import per_category_delta

    a_rows = [{"category": "d", "runs": 1, "pass_count": 1}]
    b_rows = [{"category": "d", "runs": 1, "pass_count": 0}]
    out = per_category_delta(a_rows, b_rows, n_bootstrap=50)
    row = out[0]
    assert row["suggestion"] == "noise"  # n_a + n_b = 2 < 6


def test_per_category_delta_handles_empty_categories():
    from benchmarks.analyze import per_category_delta

    a_rows = [{"category": "x", "runs": 5, "pass_count": 3}]
    b_rows: list[dict] = []  # category only in a
    out = per_category_delta(a_rows, b_rows, n_bootstrap=50)
    row = out[0]
    assert row["runs_b"] == 0
    assert row["rate_b"] == 0.0
    # With only one side, the CI is degenerate (no bootstrap possible).
    # We accept any of {tie, dom} here — the key is no crash.
    assert row["suggestion"] in ("tie", "dom", "noise")
