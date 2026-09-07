"""The worker node: creates one agent for one task, then runs it.

Takes the agent the planner synthesized, realizes it against its capability
envelope -- which is where a request for tools it may not have is refused --
binds exactly the tools it was granted, and runs it. If the envelope requires
approval, it interrupts the graph first and waits for a human decision.

It does not design agents or decide their order; it is handed a task.
"""

from typing import Any, Awaitable, Callable

from langgraph.types import interrupt

from app.agents.composer import EnvelopeViolationError, realize_agent
from app.agents.factory import AgentExecutionError, build_worker
from app.agents.models import SynthesizedAgent
from app.agents.registry import CapabilityRegistry
from app.chat_models import build_chat_model
from app.events import EventName
from app.mcp.registry import ToolRegistry

EventEmitter = Callable[[str, str, dict[str, Any]], Awaitable[None]]
ChatModelBuilder = Callable[[str], Any]


def build_worker_node(
    capabilities: CapabilityRegistry,
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
        capability = capabilities.get(payload["capability_id"])
        objective = _objective_with_context(payload)

        try:
            spec = realize_agent(
                capability, SynthesizedAgent.model_validate(payload["agent"]), tools
            )
        except (EnvelopeViolationError, ValueError) as error:
            # The planner already dry-ran this check, so reaching here means the
            # plan was tampered with after planning. Refuse rather than degrade.
            return _failed_result(task_id, payload.get("agent", {}).get("name", "unknown"),
                                  f"Agent could not be created: {error}")

        if spec.requires_approval:
            decision = interrupt({
                "task_id": task_id,
                "agent_id": spec.id,
                "capability_id": capability.id,
                "objective": payload["objective"],
                "reason": (
                    f"Agent {spec.id!r} runs under capability {capability.id!r}, which "
                    "requires human approval."
                ),
                "tools": [tool.name for tool in tools.select(spec.tool_selectors)],
                "system_prompt": spec.system_prompt,
            })
            if not _is_approved(decision):
                return _declined_result(task_id, spec.id, decision)

        await emit(run_id, EventName.TASK_STARTED, {
            "task_id": task_id,
            "agent_id": spec.id,
            "capability_id": capability.id,
            "objective": payload["objective"],
            "model": spec.model,
            "tools": [tool.name for tool in tools.select(spec.tool_selectors)],
            # The generated agent is published so a run stays auditable: the
            # prompt is not in any config file, so the trace is the only record.
            "role": spec.role,
            "system_prompt": spec.system_prompt,
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
            return _failed_result(task_id, spec.id, str(error), capability.id)

        return {
            "task_outputs": {
                task_id: {
                    "task_id": task_id,
                    "agent_id": spec.id,
                    "capability_id": capability.id,
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


def _failed_result(
    task_id: str, agent_id: str, message: str, capability_id: str = ""
) -> dict[str, Any]:
    """Record a task as failed without killing the run."""
    return {
        "task_outputs": {
            task_id: {
                "task_id": task_id,
                "agent_id": agent_id,
                "capability_id": capability_id,
                "status": "failed",
                "output": message,
            }
        }
    }


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
