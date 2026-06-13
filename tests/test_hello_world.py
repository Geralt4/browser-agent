import asyncio
import os

import pytest

from browser_agent.agent.loop import run_task
from browser_agent.config import load_config
from browser_agent.models.registry import get_adapter
from browser_agent.safety import SafetyLayer

_HAS_KEY = bool(os.getenv("MOONSHOT_API_KEY") or os.getenv("LLM_API_KEY"))


@pytest.mark.skipif(not _HAS_KEY, reason="no model API key configured (Phase 1 e2e)")
def test_hello_world_returns_h1():
    cfg = load_config()
    adapter = get_adapter(cfg)
    safety = SafetyLayer(cfg)
    history = asyncio.run(
        run_task(
            "Go to https://example.com and return the page's H1 heading text.",
            cfg=cfg,
            adapter=adapter,
            safety=safety,
        )
    )
    result = history.final_result()
    assert result is not None
    assert "example domain" in str(result).lower()
