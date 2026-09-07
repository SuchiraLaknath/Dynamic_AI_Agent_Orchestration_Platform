"""Turns a natural-language goal into a validated plan of purpose-built agents.

Retrieves a menu -- the capability envelopes best matching the goal, the tools
most relevant to it, and summaries of similar past runs -- then makes one
structured-output LLM call that *designs an agent for each task*: its name, its
role, the system prompt it will run under, and the tools it needs.

The planner chooses capability ids from the menu; it writes everything else. An
unknown capability id, or an agent reaching outside its envelope, is retried
once with the error fed back, then fails the run.

It does not execute anything, and it does not decide whether an agent is
allowed to exist -- that is app.agents.composer.
"""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.composer import EnvelopeViolationError, realize_agent
from app.agents.models import CapabilitySpec
from app.agents.registry import CapabilityRegistry
from app.costs import Usage, price_response
from app.graph.state import PlanValidationError, TaskPlan, validate_plan
from app.mcp.registry import DiscoveredTool, ToolRegistry

PLANNER_SYSTEM_PROMPT = """\
You are the planner for a multi-agent orchestration platform. For a user's goal \
you design the specialist agents that will handle it, and decide what order they \
run in. The agents do not exist yet -- you are creating them.

For each task you produce two things: the CAPABILITY it runs under, chosen from \
the menu, and the AGENT you are creating inside that capability, which you write \
yourself.

Rules you cannot break:
1. `capability_id` must be copied exactly from the AVAILABLE CAPABILITIES list. \
Never invent one. It sets what the agent is permitted to touch.
2. `agent.tool_selectors` must be a subset of that capability's allowed tools. \
Asking for anything else is rejected. Request only the tools the task truly needs \
-- fewer is better. A capability with no tools means the agent must reason purely \
over upstream task output.
3. `agent.system_prompt` is the agent's instructions, written by you for this \
task. State how to work and what to produce: the method to follow, the arithmetic \
to show, what not to guess at. Write it in the second person, addressed to the \
agent. Do not restate the objective, and do not repeat the capability's own \
policy -- that is appended automatically.
4. `agent.name` is snake_case and describes what this agent does here, e.g. \
'sprint_history_reader'. `agent.role` is one sentence on its purpose.
5. Produce the SMALLEST plan that fully answers the goal. One capable agent is a \
better plan than four. Do not add a task because a capability is available.
6. Set depends_on only for a real data dependency -- when a task cannot start \
until it has another task's output. Tasks that can run at the same time must have \
no dependency between them, so they run in parallel.
7. Keep the DAG shallow. Prefer two or three layers at most.
8. Each objective says what that agent must produce and which upstream task's \
output to build on. Do not restate the whole goal.

If no capability on the menu can do part of the goal, leave that part out rather \
than stretching a capability to cover it."""


class PlannerError(RuntimeError):
    """Raised when the planner cannot produce a valid plan."""


def render_capability_menu(specs: list[CapabilitySpec], tools: ToolRegistry) -> str:
    """The capability half of the menu.

    Allowed selectors are shown resolved to concrete tool names, because that is
    what the planner's `tool_selectors` are checked against -- showing it the
    globs instead invites a subset it cannot actually have.
    """
    lines = []
    for spec in specs:
        permitted = [tool.name for tool in tools.select(spec.allowed_tool_selectors)]
        lines.append(
            f"- capability_id: {spec.id!r}\n"
            f"  can do: {' '.join(spec.description.split())}\n"
            f"  tools an agent here may request: {', '.join(permitted) or 'none (reasons over prior output)'}\n"
            f"  models allowed: {', '.join(spec.allowed_models)}\n"
            f"  max iterations: {spec.max_iterations_limit}"
            + ("\n  NOTE: a human must approve any agent created here." if spec.requires_approval else "")
        )
    return "\n".join(lines)


def render_tool_menu(tools: list[DiscoveredTool]) -> str:
    """The tool half. Informs objectives; agents bind their own tools by selector."""
    if not tools:
        return "(no MCP tools were discovered)"
    return "\n".join(
        f"- {tool.name}: {' '.join(tool.description.split())[:180]}" for tool in tools
    )


def render_memory(similar_runs: list[dict[str, Any]]) -> str:
    """Prior runs on similar goals, so the planner can reuse a shape that worked."""
    if not similar_runs:
        return "(no similar past runs)"
    return "\n".join(
        f"- goal: {run['goal']!r}\n  capabilities used: {', '.join(run.get('agent_ids', []))}"
        f"\n  outcome: {run.get('summary', '')[:240]}"
        for run in similar_runs
    )


def build_planning_prompt(
    goal: str,
    specs: list[CapabilitySpec],
    capability_tools: ToolRegistry,
    tools: list[DiscoveredTool],
    similar_runs: list[dict[str, Any]],
) -> str:
    return (
        f"USER GOAL:\n{goal}\n\n"
        f"AVAILABLE CAPABILITIES (copy capability_id exactly from these):\n"
        f"{render_capability_menu(specs, capability_tools)}\n\n"
        f"WHAT THE MCP TOOLS DO:\n{render_tool_menu(tools)}\n\n"
        f"SIMILAR PAST RUNS:\n{render_memory(similar_runs)}\n\n"
        "Design the agents and produce the task plan."
    )


def check_envelopes(plan: TaskPlan, capabilities: CapabilityRegistry, tools: ToolRegistry) -> None:
    """Dry-run every synthesized agent through its envelope, discarding the result.

    Done at planning time rather than at execution time so a violation is
    something the planner can be told about and retry, instead of a task dying
    halfway through a run that has already spent money.
    """
    for task in plan.tasks:
        realize_agent(capabilities.get(task.capability_id), task.agent, tools)


async def plan_goal(
    goal: str,
    capabilities: CapabilityRegistry,
    tools: ToolRegistry,
    similar_runs: list[dict[str, Any]],
    model: BaseChatModel,
    capability_top_k: int,
    tool_top_k: int,
) -> tuple[TaskPlan, list[CapabilitySpec], Usage]:
    """Retrieve a menu, design agents against it, validate, retry once, then give up.

    The retry feeds the validation error back rather than resampling blind: the
    failure is almost always one bad id or one tool too many, and naming it is
    what fixes it.
    """
    menu = capabilities.search_by_capability(goal, capability_top_k)
    relevant_tools = tools.search(goal, tool_top_k)
    prompt = build_planning_prompt(goal, menu, tools, relevant_tools, similar_runs)

    structured = model.with_structured_output(TaskPlan, include_raw=True)
    messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    usage = Usage()
    last_error: Exception | None = None

    for attempt in (1, 2):
        response = await structured.ainvoke(messages)
        raw, parsed = response.get("raw"), response.get("parsed")
        usage = usage + price_response(getattr(model, "model", ""), getattr(raw, "usage_metadata", None))

        if parsed is None:
            last_error = PlannerError(
                f"Planner returned no parsable plan: {response.get('parsing_error')}"
            )
        else:
            try:
                validated = validate_plan(parsed, capabilities.capability_ids)
                check_envelopes(validated, capabilities, tools)
                return validated, menu, usage
            except (PlanValidationError, EnvelopeViolationError) as error:
                last_error = error

        if attempt == 1:
            messages.append(HumanMessage(content=(
                f"That plan was rejected: {last_error}\n\n"
                "Produce a corrected plan. Every capability_id must be copied exactly from "
                "the AVAILABLE CAPABILITIES list above, and every agent's tool_selectors "
                "must be a subset of the tools listed for its capability."
            )))

    raise PlannerError(f"Planner failed after 2 attempts. Last error: {last_error}") from last_error


def summarize_plan(plan: TaskPlan) -> list[dict[str, Any]]:
    """The plan as JSON-safe dicts, for graph state and the run.started event."""
    return [json.loads(task.model_dump_json()) for task in plan.tasks]
