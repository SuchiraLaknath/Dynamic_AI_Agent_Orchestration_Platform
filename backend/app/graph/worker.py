"""The worker node: runs exactly one task with exactly one agent.

Resolves the task's agent_id to a spec, binds only the tools that spec's
selectors match, builds a runnable with the factory, and runs it. If the spec
requires approval, it interrupts the graph first and waits for a human decision.

It does not decide which agent runs, or in what order -- it is handed a task.
"""

from typing import Any, Awaitable, Callable

from langgraph.types import interrupt

from app.agents.factory import AgentExecutionError, build_worker
from app.agents.registry import AgentRegistry
from app.chat_models import build_chat_model
from app.events import EventName
from app.mcp.registry import ToolRegistry

EventEmitter = Callable[[str, str, dict[str, Any]], Awaitable[None]]
ChatModelBuilder = Callable[[str], Any]


def build_worker_node(
    agents: AgentRegistry,
    tools: ToolRegistry,
    emit: EventEmitter,
    chat_model_builder: ChatModelBuilder = build_chat_model,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Return the LangGraph node that executes one dispatched task.

    Dependencies are closed over rather than read from globals so the graph can
    be built against a stub registry and a stub model in tests.
    """

    async def run_worker(payload: dict[str, Any]) -> dict[str, Any]:
        run_id = payload["run_id"]
        task_id = payload["task_id"]
        spec = agents.get(payload["agent_id"])
        objective = _objective_with_context(payload)

        if spec.requires_approval:
            decision = interrupt({
                "task_id": task_id,
                "agent_id": spec.id,
                "objective": payload["objective"],
                "reason": f"Agent {spec.id!r} is configured as requiring approval.",
                "tools": spec.tool_selectors,
            })
            if not _is_approved(decision):
                return _declined_result(task_id, spec.id, decision)

        await emit(run_id, EventName.TASK_STARTED, {
            "task_id": task_id,
            "agent_id": spec.id,
            "objective": payload["objective"],
            "model": spec.model,
            "tools": [tool.name for tool in tools.select(spec.tool_selectors)],
        })

        async def emit_tool_event(name: str, data: dict[str, Any]) -> None:
            await emit(run_id, EventName(name), data)

        runnable = build_worker(
            spec, tools.select(spec.tool_selectors), chat_model_builder(spec.model), emit_tool_event
        )

        try:
            result = await runnable(task_id, objective)
        except AgentExecutionError as error:
            # Recorded as a failed task, not raised: sibling tasks have already
            # done real work, and the synthesizer can report a partial answer.
            return {
                "task_outputs": {
                    task_id: {
                        "task_id": task_id, "agent_id": spec.id,
                        "status": "failed", "output": str(error),
                    }
                }
            }

        return {
            "task_outputs": {
                task_id: {
                    "task_id": task_id,
                    "agent_id": spec.id,
                    "status": "completed",
                    "output": result.output,
                    "tool_calls": result.tool_calls,
                }
            },
            "usage_records": [{"task_id": task_id, "agent_id": spec.id, **result.usage.model_dump()}],
        }

    return run_worker


def _objective_with_context(payload: dict[str, Any]) -> str:
    """The objective, plus the output of every task this one depends on.

    This is how data moves along a DAG edge: a dependency's answer is quoted
    into the dependent's prompt, rather than both agents re-fetching it.
    """
    objective = payload["objective"]
    upstream = payload.get("dependency_outputs") or {}
    if not upstream:
        return f"Overall goal: {payload.get('goal', '')}\n\nYour task: {objective}"

    context = "\n\n".join(
        f"--- Output of upstream task {task_id!r} (agent {result.get('agent_id')}) ---\n"
        f"{result.get('output', '')}"
        for task_id, result in upstream.items()
    )
    return (
        f"Overall goal: {payload.get('goal', '')}\n\n"
        f"Your task: {objective}\n\n"
        f"You have been given the output of the tasks yours depends on. Build on it; "
        f"do not re-derive it.\n\n{context}"
    )


def _is_approved(decision: Any) -> bool:
    """Read a human's resume value. Anything that is not an explicit yes is a no."""
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, dict):
        return bool(decision.get("approved"))
    return False


def _declined_result(task_id: str, agent_id: str, decision: Any) -> dict[str, Any]:
    reason = decision.get("reason", "") if isinstance(decision, dict) else ""
    return {
        "task_outputs": {
            task_id: {
                "task_id": task_id,
                "agent_id": agent_id,
                "status": "declined",
                "output": (
                    f"A human declined this task, so agent {agent_id!r} did not run and "
                    f"nothing was written.{f' Reason given: {reason}' if reason else ''}"
                ),
            }
        }
    }
