"""Builds a runnable worker from an AgentSpec and the tools it selected.

This is where "agents are data" becomes real: `build_worker` closes over a spec
and a tool list and returns a coroutine. There is no agent class and no
subclassing -- adding a specialist is a record in agents.yaml, and this function
does not change.

The tool loop here is hand-written rather than delegated to a prebuilt ReAct
agent, for three reasons: `max_iterations` is enforced per spec, every tool call
emits `tool.called`/`tool.result` for the trace, and token usage is attributed
per call. A prebuilt agent would hide all three inside a subgraph.

It does not choose which agent runs (that is the planner) or open MCP sessions
(that is app.mcp.manager).
"""

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.models import AgentSpec
from app.costs import Usage, price_response, sum_usage
from app.mcp.manager import McpToolCallError, call_tool_with_policy
from app.mcp.registry import DiscoveredTool

# Emits one trace event. Injected so the factory stays independent of the bus.
EventEmitter = Callable[[str, dict[str, Any]], Awaitable[None]]


class AgentExecutionError(RuntimeError):
    """Raised when a worker cannot produce an answer for its task."""


@dataclass(frozen=True)
class WorkerResult:
    """What one agent produced for one task. A record, not an agent."""

    output: str
    usage: Usage
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def build_worker(
    spec: AgentSpec,
    tools: list[DiscoveredTool],
    model: BaseChatModel,
    emit: EventEmitter,
) -> Callable[[str, str], Awaitable[WorkerResult]]:
    """Return a coroutine that runs `spec` against one objective.

    `model` is passed in rather than constructed here so tests can substitute a
    stub chat model without patching module internals.
    """
    tools_by_bare_name = {tool.tool.name: tool for tool in tools}
    bound_model = model.bind_tools([tool.tool for tool in tools]) if tools else model

    async def run_task(task_id: str, objective: str) -> WorkerResult:
        messages: list[BaseMessage] = [
            SystemMessage(content=spec.system_prompt),
            HumanMessage(content=objective),
        ]
        usages: list[Usage] = []
        recorded_calls: list[dict[str, Any]] = []

        for iteration in range(spec.max_iterations):
            response = await bound_model.ainvoke(messages)
            usages.append(_usage_of(spec.model, response))
            messages.append(response)

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                text = _text_of(response)
                if not text:
                    raise AgentExecutionError(
                        f"Agent {spec.id!r} returned neither tool calls nor text on task {task_id!r}"
                    )
                return WorkerResult(text, sum_usage(usages), recorded_calls)

            for call in tool_calls:
                messages.append(
                    await _invoke_tool(spec, task_id, call, tools_by_bare_name, emit, recorded_calls)
                )

        # Out of iterations with the model still calling tools. Returning the
        # last text would be a guess, so this is an explicit failure.
        raise AgentExecutionError(
            f"Agent {spec.id!r} hit its {spec.max_iterations}-iteration limit on task "
            f"{task_id!r} without producing an answer. Raise max_iterations in agents.yaml "
            "or narrow the objective."
        )

    return run_task


async def _invoke_tool(
    spec: AgentSpec,
    task_id: str,
    call: dict[str, Any],
    tools_by_bare_name: dict[str, DiscoveredTool],
    emit: EventEmitter,
    recorded_calls: list[dict[str, Any]],
) -> ToolMessage:
    """Run one tool call and return the message that reports it to the model.

    A tool failure is returned to the model as an error ToolMessage rather than
    raised: the agent can often recover by calling something else, and killing
    the run on one bad call throws away the work already done.
    """
    requested = call.get("name", "")
    discovered = tools_by_bare_name.get(requested)
    namespaced = discovered.name if discovered else requested
    arguments = call.get("args", {}) or {}

    await emit("tool.called", {
        "task_id": task_id, "agent_id": spec.id, "tool": namespaced, "arguments": arguments,
    })

    if discovered is None:
        # Only reachable if the model invents a name; it was bound to this list.
        content = (
            f"Tool {requested!r} is not available to you. Available: "
            f"{', '.join(sorted(tools_by_bare_name)) or 'none'}"
        )
        await emit("tool.result", {
            "task_id": task_id, "agent_id": spec.id, "tool": namespaced,
            "ok": False, "preview": content,
        })
        return ToolMessage(content=content, tool_call_id=call.get("id", ""), status="error")

    try:
        result = await call_tool_with_policy(discovered.tool, arguments)
        content, ok = _stringify(result), True
    except McpToolCallError as error:
        content, ok = f"Tool call failed: {error}", False

    recorded_calls.append({"tool": namespaced, "arguments": arguments, "ok": ok})
    await emit("tool.result", {
        "task_id": task_id, "agent_id": spec.id, "tool": namespaced,
        "ok": ok, "preview": content[:600],
    })
    return ToolMessage(
        content=content, tool_call_id=call.get("id", ""), status="success" if ok else "error"
    )


def _usage_of(model: str, response: BaseMessage) -> Usage:
    return price_response(model, getattr(response, "usage_metadata", None))


def _text_of(message: BaseMessage) -> str:
    """Flatten a response's content to text, tolerating content-block lists."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part).strip()


def _stringify(result: Any) -> str:
    """Render an MCP tool result as text for the model."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        texts = [
            block.get("text", "") if isinstance(block, dict) else str(block) for block in result
        ]
        return "\n".join(text for text in texts if text)
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


__all__ = ["build_worker", "AgentExecutionError", "WorkerResult", "EventEmitter"]
