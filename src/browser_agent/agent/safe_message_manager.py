from __future__ import annotations

from typing import Literal

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.views import (
    ActionResult,
    AgentOutput,
    AgentStepInfo,
)
from browser_use.browser.views import BrowserStateSummary
from browser_use.llm.messages import ContentPartTextParam

from browser_agent.safety.injection import sanitize


class InjectionSafeMessageManager(MessageManager):
    """MessageManager that sanitizes DOM content before the LLM sees it.

    Wraps the standard browser-use MessageManager, applying the injection
    filter to the DOM representation embedded in each state message.
    """

    def create_state_messages(
        self,
        browser_state_summary: BrowserStateSummary,
        model_output: AgentOutput | None = None,
        result: list[ActionResult] | None = None,
        step_info: AgentStepInfo | None = None,
        use_vision: bool | Literal["auto"] = True,
        page_filtered_actions: str | None = None,
        sensitive_data=None,
        available_file_paths: list[str] | None = None,
        unavailable_skills_info: str | None = None,
        plan_description: str | None = None,
        skip_state_update: bool = False,
        **kwargs,
    ) -> None:
        super().create_state_messages(
            browser_state_summary=browser_state_summary,
            model_output=model_output,
            result=result,
            step_info=step_info,
            use_vision=use_vision,
            page_filtered_actions=page_filtered_actions,
            sensitive_data=sensitive_data,
            available_file_paths=available_file_paths,
            unavailable_skills_info=unavailable_skills_info,
            plan_description=plan_description,
            skip_state_update=skip_state_update,
            **kwargs,
        )

        state_msg = self.state.history.state_message
        if state_msg is None:
            return

        content = state_msg.content
        if isinstance(content, str):
            state_msg.content = sanitize(content)
        elif isinstance(content, list):
            for i, part in enumerate(content):
                if isinstance(part, ContentPartTextParam):
                    content[i] = ContentPartTextParam(text=sanitize(part.text))

        self.last_state_message_text = state_msg.text
