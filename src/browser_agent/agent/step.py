from __future__ import annotations

from pydantic import BaseModel


class AgentStep(BaseModel):
    """One step of the agent's structured per-step output.

    CLAUDE.md:54-55 requires the agent to surface, per step:
      - assessment: did the last action work?
      - memory: progress notes
      - next_subgoal: what to do next
      - action: the action taken in this step

    Centralizing the shape here means the loop's on_step callback, the UI's
    SSE handler, and the extension's renderer all read from one schema. The
    browser-use AgentState attribute names are mapped to these fields in
    exactly one place (agent.loop._extract_step) so a browser-use rename
    only has to be fixed once.
    """

    step_n: int
    assessment: str = ""
    memory: str = ""
    next_subgoal: str = ""
    action: str = ""
