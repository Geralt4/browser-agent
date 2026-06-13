from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PendingAction(BaseModel):
    """An action the agent wants to take, presented to the safety layer."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.name}({args})"


class SafetyDecision(BaseModel):
    """The safety layer's verdict on a PendingAction."""

    allow: bool
    reason: str = ""
