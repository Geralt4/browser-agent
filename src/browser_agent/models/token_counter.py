from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion


@dataclass
class TokenTotals:
    """Cumulative token usage across one agent run.

    `input` is the sum of `prompt_tokens` reported by the wrapped chat model.
    `output` is the sum of `completion_tokens`. `calls` is the number of
    `ainvoke` round-trips observed (useful for averaging per-step cost).
    """

    input: int = 0
    output: int = 0
    calls: int = 0


class TokenCountingChatModel(BaseChatModel):
    """Proxy around a BaseChatModel that records per-call token usage.

    `browser-use`'s `AgentHistory` does not surface token counts, so the
    benchmark harness wraps the adapter's chat model with this proxy before
    handing it to `Agent(llm=...)`. Each `ainvoke` increments `totals.input`
    by `usage.prompt_tokens` and `totals.output` by `usage.completion_tokens`
    when the provider reports usage; providers that omit it contribute 0
    (so totals may under-report rather than over-report).

    Only attribute and method access is forwarded; `model_name` / `provider`
    are passed through so logs and browser-use telemetry stay accurate.
    """

    def __init__(self, inner: BaseChatModel) -> None:
        self._inner = inner
        self.totals = TokenTotals()
        self._model_name = getattr(inner, "model_name", None) or getattr(inner, "model", "unknown")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return getattr(self._inner, "provider", "unknown")

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", self._model_name)

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[Any]:
        result = await self._inner.ainvoke(messages, output_format, **kwargs)
        self.totals.calls += 1
        usage = getattr(result, "usage", None)
        if usage is not None:
            in_tokens = getattr(usage, "prompt_tokens", None)
            out_tokens = getattr(usage, "completion_tokens", None)
            if isinstance(in_tokens, int):
                self.totals.input += in_tokens
            if isinstance(out_tokens, int):
                self.totals.output += out_tokens
        return result
