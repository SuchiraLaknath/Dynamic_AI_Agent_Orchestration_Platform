"""Test doubles: a scripted chat model and a recording event bus.

Lets the whole graph run with no API key and no network, so tests assert on
orchestration behaviour -- which agents were selected, what order they ran in,
what the trace contained -- rather than on model output.

These are test-only. Nothing in app/ imports this module.
"""

from typing import Any, Callable, Iterator

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

from app.events import EventName


class StubChatModel(BaseChatModel):
    """A chat model whose replies are decided by a function of the messages.

    `responder` receives the message list and returns an AIMessage, so a test
    can branch on the system prompt to give the planner, a worker and the
    synthesizer different scripted answers from one object.

    `structured_responder` supplies the planner's structured output directly,
    bypassing tool-call parsing that a real model would do.
    """

    responder: Callable[[list[BaseMessage]], AIMessage]
    structured_responder: Callable[[list[BaseMessage]], Any] | None = None
    model: str = "stub-model"
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "stub-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responder(messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StubChatModel":
        """Accept the binding and ignore it; the responder decides what is called."""
        return self

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable:
        """Return the scripted plan in the shape `include_raw=True` produces."""

        def respond(messages: list[BaseMessage]) -> Any:
            if self.structured_responder is None:
                raise AssertionError("StubChatModel was asked for structured output with no script")
            self.calls.append(list(messages))
            parsed = self.structured_responder(messages)
            raw = AIMessage(
                content="",
                usage_metadata={"input_tokens": 1200, "output_tokens": 150, "total_tokens": 1350},
            )
            if isinstance(parsed, Exception):
                return {"raw": raw, "parsed": None, "parsing_error": parsed}
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(respond)


def scripted_sequence(messages: list[AIMessage]) -> Callable[[list[BaseMessage]], AIMessage]:
    """A responder that returns queued messages in order, then repeats the last."""
    remaining: Iterator[AIMessage] = iter(messages)
    last: dict[str, AIMessage] = {}

    def respond(_: list[BaseMessage]) -> AIMessage:
        try:
            last["value"] = next(remaining)
        except StopIteration:
            pass
        return last["value"]

    return respond


def usage(input_tokens: int = 800, output_tokens: int = 120) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class RecordingBus:
    """Captures published events instead of fanning them out to SSE clients."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    async def emit(self, run_id: str, name: EventName, data: dict[str, Any]) -> None:
        self.events.append((run_id, str(name), data))

    def names(self) -> list[str]:
        return [name for _, name, _ in self.events]

    def of(self, name: str) -> list[dict[str, Any]]:
        return [data for _, event_name, data in self.events if event_name == name]
