"""Turns a natural-language goal into a validated task DAG.

Retrieves a menu -- the agents whose roles best match the goal, the tools most
relevant to it, and summaries of similar past runs -- then makes one
structured-output LLM call that must choose from that menu. The result is
validated against the live registry; an invented agent id is retried once with
the error fed back, then fails the run.

It does not execute anything. It only decides what should run and in what order.
"""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.models import AgentSpec
from app.agents.registry import AgentRegistry
from app.costs import Usage, price_response
from app.graph.state import PlanValidationError, TaskPlan, validate_plan
from app.mcp.registry import DiscoveredTool, ToolRegistry

PLANNER_SYSTEM_PROMPT = """\
You are the planner for a multi-agent orchestration platform. You decide which \
specialist agents handle a user's goal, and in what order.

Rules you cannot break:
1. Use ONLY agent ids from the AVAILABLE AGENTS list. Never invent an id, never \
guess at one, never use an agent that is not listed.
2. Produce the SMALLEST plan that fully answers the goal. One capable agent is a \
better plan than four. Do not add an agent because it is available.
3. Set depends_on only for a real data dependency -- when a task cannot start \
until it has another task's output. Tasks that can run at the same time must have \
no dependency between them, so they run in parallel.
4. Keep the DAG shallow. Prefer two or three layers at most.
5. Each objective is an instruction to one agent: say what it must produce, and \
name which upstream task's output it should build on. Do not restate the whole goal.

You are choosing from a menu. If nothing on the menu can do part of the goal, \
leave that part out rather than inventing an agent for it."""


class PlannerError(RuntimeError):
    """Raised when the planner cannot produce a valid plan."""


def render_agent_menu(specs: list[AgentSpec]) -> str:
    """The agent half of the menu. Ids are quoted because they are what gets copied."""
    return "\n".join(
        f"- id: {spec.id!r}\n"
        f"  can do: {' '.join(spec.role.split())}\n"
        f"  tools it holds: {', '.join(spec.tool_selectors) or 'none (reasons over prior output)'}"
        for spec in specs
    )


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
        f"- goal: {run['goal']!r}\n  agents used: {', '.join(run.get('agent_ids', []))}"
        f"\n  outcome: {run.get('summary', '')[:240]}"
        for run in similar_runs
    )


def build_planning_prompt(
    goal: str,
    specs: list[AgentSpec],
    tools: list[DiscoveredTool],
    similar_runs: list[dict[str, Any]],
) -> str:
    return (
        f"USER GOAL:\n{goal}\n\n"
        f"AVAILABLE AGENTS (choose only from these ids):\n{render_agent_menu(specs)}\n\n"
        f"MCP TOOLS AVAILABLE TO THOSE AGENTS:\n{render_tool_menu(tools)}\n\n"
        f"SIMILAR PAST RUNS:\n{render_memory(similar_runs)}\n\n"
        "Produce the task plan."
    )


async def plan_goal(
    goal: str,
    agents: AgentRegistry,
    tools: ToolRegistry,
    similar_runs: list[dict[str, Any]],
    model: BaseChatModel,
    agent_top_k: int,
    tool_top_k: int,
) -> tuple[TaskPlan, list[AgentSpec], Usage]:
    """Retrieve a menu, plan against it, validate, retry once, then give up.

    The retry feeds the validation error back rather than resampling blind: the
    failure is almost always one bad id, and naming it is what fixes it.
    """
    menu = agents.search_by_role(goal, agent_top_k)
    relevant_tools = tools.search(goal, tool_top_k)
    prompt = build_planning_prompt(goal, menu, relevant_tools, similar_runs)

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
                return validate_plan(parsed, agents.agent_ids), menu, usage
            except PlanValidationError as error:
                last_error = error

        if attempt == 1:
            messages.append(HumanMessage(content=(
                f"That plan was rejected: {last_error}\n\n"
                "Produce a corrected plan. Every agent_id must be copied exactly from the "
                "AVAILABLE AGENTS list above."
            )))

    raise PlannerError(f"Planner failed after 2 attempts. Last error: {last_error}") from last_error


def summarize_plan(plan: TaskPlan) -> list[dict[str, Any]]:
    """The plan as JSON-safe dicts, for graph state and the run.started event."""
    return [json.loads(task.model_dump_json()) for task in plan.tasks]
