"""Assembles and compiles the LangGraph.

Wires four nodes -- planner, dispatcher, worker, synthesizer -- into the shape
that gives this platform its behaviour: plan once, then repeatedly dispatch
every task whose dependencies are met, in parallel, until none are left, then
synthesize.

The dispatcher is what makes the DAG dynamic. The graph's topology is fixed and
tiny; the *plan* decides how many workers run and in what order, at runtime,
through `Send`.

It does not own the database or the event bus; those are passed in.
"""

from typing import Any, Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents.registry import CapabilityRegistry
from app.chat_models import build_chat_model
from app.events import EventName
from app.graph.planner import plan_goal, summarize_plan
from app.graph.state import OrchestrationState, ready_tasks
from app.graph.synthesizer import build_synthesizer_node
from app.graph.worker import build_worker_node
from app.mcp.registry import ToolRegistry
from app.settings import Settings

EventEmitter = Callable[[str, str, dict[str, Any]], Awaitable[None]]
MemoryLookup = Callable[[str], Awaitable[list[dict[str, Any]]]]
PlanRecorder = Callable[[str, list[dict[str, Any]]], Awaitable[None]]


def build_planner_node(
    capabilities: CapabilityRegistry,
    tools: ToolRegistry,
    settings: Settings,
    emit: EventEmitter,
    find_similar_runs: MemoryLookup,
    record_plan: PlanRecorder,
    chat_model_builder: Callable[[str], Any],
) -> Callable[[OrchestrationState], Awaitable[dict[str, Any]]]:
    """Return the node that turns the goal into a validated plan."""

    async def run_planner(state: OrchestrationState) -> dict[str, Any]:
        goal, run_id = state["goal"], state["run_id"]
        similar_runs = await find_similar_runs(goal)

        plan, menu, usage = await plan_goal(
            goal=goal,
            capabilities=capabilities,
            tools=tools,
            similar_runs=similar_runs,
            model=chat_model_builder(settings.planner_model),
            capability_top_k=settings.planner_agent_top_k,
            tool_top_k=settings.planner_tool_top_k,
        )
        tasks = summarize_plan(plan)
        await record_plan(run_id, tasks)

        await emit(run_id, EventName.RUN_STARTED, {
            "goal": goal,
            "plan": tasks,
            "reasoning": plan.reasoning,
            "considered_agents": [spec.id for spec in menu],
            "recalled_runs": [run["run_id"] for run in similar_runs],
        })
        return {
            "plan": tasks,
            "plan_reasoning": plan.reasoning,
            "similar_runs": similar_runs,
            "usage_records": [{"task_id": "planner", "agent_id": "planner", **usage.model_dump()}],
        }

    return run_planner


def route_ready_tasks(state: OrchestrationState) -> list[Send] | str:
    """Dispatch every currently-runnable task at once, or move on to synthesis.

    This function is the scheduler. Returning a list of `Send` fans out one
    worker per ready task in a single superstep; returning "synthesizer" ends
    the loop. Because it is re-evaluated after every superstep, a plan of any
    shape executes with maximum available parallelism and no fixed topology.
    """
    plan = state.get("plan", [])
    completed = state.get("task_outputs", {})
    pending = ready_tasks(plan, completed)

    if not pending:
        return "synthesizer"

    return [
        Send("worker", {
            "run_id": state["run_id"],
            "goal": state["goal"],
            "task_id": task["task_id"],
            "capability_id": task["capability_id"],
            "agent": task["agent"],
            "objective": task["objective"],
            "dependency_outputs": {
                dependency: completed[dependency]
                for dependency in task.get("depends_on", [])
                if dependency in completed
            },
        })
        for task in pending
    ]


async def _pass_through(state: OrchestrationState) -> dict[str, Any]:
    """The dispatcher node itself does nothing; its conditional edges do the work."""
    return {}


def build_orchestration_graph(
    capabilities: CapabilityRegistry,
    tools: ToolRegistry,
    settings: Settings,
    emit: EventEmitter,
    find_similar_runs: MemoryLookup,
    record_plan: PlanRecorder,
    checkpointer: Any = None,
    chat_model_builder: Callable[[str], Any] = build_chat_model,
) -> Any:
    """Compile the graph. `checkpointer` is required for approval interrupts."""
    builder = StateGraph(OrchestrationState)

    builder.add_node(
        "planner",
        build_planner_node(
            capabilities, tools, settings, emit, find_similar_runs, record_plan, chat_model_builder
        ),
    )
    builder.add_node("dispatcher", _pass_through)
    builder.add_node("worker", build_worker_node(capabilities, tools, emit, chat_model_builder))
    builder.add_node(
        "synthesizer", build_synthesizer_node(chat_model_builder, settings.synthesizer_model)
    )

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "dispatcher")
    builder.add_conditional_edges("dispatcher", route_ready_tasks, ["worker", "synthesizer"])
    # Workers return to the dispatcher, which re-evaluates what is now runnable.
    builder.add_edge("worker", "dispatcher")
    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=checkpointer)
