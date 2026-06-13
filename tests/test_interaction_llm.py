import asyncio
import os

import pytest

from browser_agent.agent.loop import run_task
from browser_agent.config import load_config
from browser_agent.models.registry import get_adapter
from browser_agent.safety import SafetyLayer

_HAS_KEY = bool(os.getenv("MOONSHOT_API_KEY") or os.getenv("LLM_API_KEY"))


@pytest.mark.skipif(not _HAS_KEY, reason="no model API key configured (Phase 1 e2e)")
def test_agent_types_and_clicks(fixture_url):
    """LLM-driven counterpart: the model must use the gated type + click tools."""
    url = fixture_url("search.html")
    cfg = load_config()
    adapter = get_adapter(cfg)
    safety = SafetyLayer(cfg)
    task = (
        f"Go to {url}. Type the word 'hello' into the search text box, then click "
        f"the Search button. Then report the text that appears on the page."
    )
    history = asyncio.run(run_task(task, cfg=cfg, adapter=adapter, safety=safety))
    result = history.final_result()
    assert result is not None
    assert "blocked by safety layer" not in str(result).lower()
