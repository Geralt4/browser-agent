from __future__ import annotations

import asyncio
import sys

from browser_agent.agent.loop import run_task
from browser_agent.config import load_config
from browser_agent.models.registry import get_adapter
from browser_agent.safety import SafetyLayer


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print('usage: browser-agent "<natural-language task>"')
        raise SystemExit(2)

    task = " ".join(args)
    cfg = load_config()
    adapter = get_adapter(cfg)
    safety = SafetyLayer(cfg)

    print(f"[browser-agent] provider={cfg.provider} model={adapter.name} task={task!r}")
    history = asyncio.run(run_task(task, cfg=cfg, adapter=adapter, safety=safety))

    result = history.final_result() if hasattr(history, "final_result") else history
    print("\n=== result ===")
    print(result)
