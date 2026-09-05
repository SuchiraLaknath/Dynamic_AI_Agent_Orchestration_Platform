"""A whole run, end to end: real mock MCP server, stubbed LLM, real graph.

Asserts on orchestration rather than prose -- which agents ran, in what order,
which tools each one reached, what the trace contained, and that the approval
gate actually stops a write. The model is the only stub.
"""

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph.build import build_orchestration_graph
from app.graph.state import TaskPlan
from tests.stubs import RecordingBus, StubChatModel, usage

GOAL = "Predict the velocity for our next sprint based on previous Jira sprints."

FORECAST_PLAN = TaskPlan.model_validate({
    "reasoning": "Fetch history, then forecast and capacity in parallel, then review risk.",
    "tasks": [
        {"task_id": "fetch", "agent_id": "jira_analyst",
         "objective": "Retrieve completed points for recent sprints.", "depends_on": []},
        {"task_id": "forecast", "agent_id": "velocity_forecaster",
         "objective": "Rolling average of completed points.", "depends_on": ["fetch"]},
        {"task_id": "capacity", "agent_id": "capacity_planner",
         "objective": "Adjust for next sprint availability.", "depends_on": ["fetch"]},
        {"task_id": "risk", "agent_id": "risk_reviewer",
         "objective": "Review the commitment for risk.", "depends_on": ["forecast", "capacity"]},
    ],
})

PUBLISH_PLAN = TaskPlan.model_validate({
    "reasoning": "Fetch, then publish.",
    "tasks": [
        {"task_id": "fetch", "agent_id": "jira_analyst",
         "objective": "Retrieve recent sprints.", "depends_on": []},
        {"task_id": "publish", "agent_id": "sprint_plan_publisher",
         "objective": "Publish 37 points to SPR-39.", "depends_on": ["fetch"]},
    ],
})


def responder(messages):
    """Route by system prompt: one stub serves planner, every worker, and synthesizer."""
    system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
    already_used_a_tool = any(isinstance(m, ToolMessage) for m in messages)

    if "final answer for a user" in system:
        return AIMessage(content="**Forecast: 37 points for SPR-39.**", usage_metadata=usage())
    if "Summarize, in at most two sentences" in system:
        return AIMessage(content="Forecast next-sprint velocity at ~37 points.",
                         usage_metadata=usage())

    if not already_used_a_tool:
        if "sprint history" in system:
            return AIMessage(content="", usage_metadata=usage(), tool_calls=[
                {"name": "get_sprints", "args": {"limit": 6}, "id": "c1"}])
        if "team availability" in system:
            return AIMessage(content="", usage_metadata=usage(), tool_calls=[
                {"name": "get_team_capacity", "args": {}, "id": "c2"}])
        if "publish an agreed sprint commitment" in system:
            return AIMessage(content="", usage_metadata=usage(), tool_calls=[
                {"name": "publish_sprint_plan",
                 "args": {"sprint_id": "SPR-39", "committed_points": 37, "summary": "avg"},
                 "id": "c3"}])

    return AIMessage(content="Answered from the data I was given.", usage_metadata=usage())


async def no_memory(_goal: str) -> list:
    return []


async def ignore_plan(_run_id: str, _plan: list) -> None:
    return None


def compile_graph(agents, tools, settings, bus, plan: TaskPlan):
    return build_orchestration_graph(
        agents=agents,
        tools=tools,
        settings=settings,
        emit=bus.emit,
        find_similar_runs=no_memory,
        record_plan=ignore_plan,
        checkpointer=InMemorySaver(),
        chat_model_builder=lambda model_id: StubChatModel(
            responder=responder, structured_responder=lambda m: plan
        ),
    )


@pytest.fixture
async def forecast_run(agents, tools, settings):
    bus = RecordingBus()
    graph = compile_graph(agents, tools, settings, bus, FORECAST_PLAN)
    state = await graph.ainvoke(
        {"run_id": "run_test", "goal": GOAL, "task_outputs": {}, "usage_records": []},
        config={"configurable": {"thread_id": "run_test"}, "recursion_limit": 50},
    )
    return state, bus


async def test_the_trace_has_the_expected_event_sequence(forecast_run):
    _, bus = forecast_run
    names = bus.names()

    assert names[0] == "run.started"
    assert names.count("task.started") == 4, "one per planned task"
    assert set(names) <= {"run.started", "task.started", "tool.called", "tool.result"}, (
        "workers emit only the documented event vocabulary; the terminal event is "
        "published by the orchestrator, which is not exercised here"
    )
    # Every tool.called is answered by a tool.result, in order.
    tool_events = [name for name in names if name.startswith("tool.")]
    assert tool_events == ["tool.called", "tool.result"] * (len(tool_events) // 2)


async def test_run_started_reports_the_plan_and_the_agents_considered(forecast_run):
    _, bus = forecast_run
    started = bus.of("run.started")[0]

    assert started["goal"] == GOAL
    assert [t["task_id"] for t in started["plan"]] == ["fetch", "forecast", "capacity", "risk"]
    assert started["reasoning"]
    assert set(started["considered_agents"]) <= {
        "jira_analyst", "velocity_forecaster", "capacity_planner",
        "risk_reviewer", "sprint_plan_publisher",
    }


async def test_every_planned_agent_ran_and_completed(forecast_run):
    state, _ = forecast_run
    outputs = state["task_outputs"]

    assert set(outputs) == {"fetch", "forecast", "capacity", "risk"}
    assert all(result["status"] == "completed" for result in outputs.values())
    assert outputs["fetch"]["agent_id"] == "jira_analyst"
    assert outputs["risk"]["agent_id"] == "risk_reviewer"


async def test_dependencies_ran_before_their_dependents(forecast_run):
    _, bus = forecast_run
    order = [data["task_id"] for data in bus.of("task.started")]

    assert order.index("fetch") < order.index("forecast")
    assert order.index("fetch") < order.index("capacity")
    assert order.index("forecast") < order.index("risk")
    assert order.index("capacity") < order.index("risk")


async def test_independent_tasks_are_dispatched_in_the_same_layer(forecast_run):
    """forecast and capacity both depend only on fetch, so nothing runs between them."""
    _, bus = forecast_run
    order = [data["task_id"] for data in bus.of("task.started")]

    assert abs(order.index("forecast") - order.index("capacity")) == 1


async def test_each_agent_reached_only_its_own_tools(forecast_run):
    _, bus = forecast_run
    called = {(data["agent_id"], data["tool"]) for data in bus.of("tool.called")}

    assert ("jira_analyst", "jira.get_sprints") in called
    assert ("capacity_planner", "jira.get_team_capacity") in called
    # risk_reviewer has no selectors, and nobody touched the write tool.
    assert not [pair for pair in called if pair[0] == "risk_reviewer"]
    assert not [pair for pair in called if pair[1] == "jira.publish_sprint_plan"]


async def test_real_fixture_data_came_back_through_mcp(forecast_run):
    _, bus = forecast_run
    previews = " ".join(data["preview"] for data in bus.of("tool.result"))

    assert "SPR-" in previews, "sprint ids came from the mock server's fixture"
    assert all(data["ok"] for data in bus.of("tool.result"))


async def test_upstream_output_is_handed_to_the_dependent_task(agents, tools, settings):
    """A dependency's answer must reach its dependent's prompt, not be re-fetched."""
    seen: list[str] = []
    bus = RecordingBus()

    def capturing_responder(messages):
        seen.append(" ".join(str(m.content) for m in messages))
        return responder(messages)

    graph = build_orchestration_graph(
        agents=agents, tools=tools, settings=settings, emit=bus.emit,
        find_similar_runs=no_memory, record_plan=ignore_plan, checkpointer=InMemorySaver(),
        chat_model_builder=lambda model_id: StubChatModel(
            responder=capturing_responder, structured_responder=lambda m: FORECAST_PLAN
        ),
    )
    await graph.ainvoke(
        {"run_id": "run_ctx", "goal": GOAL, "task_outputs": {}, "usage_records": []},
        config={"configurable": {"thread_id": "run_ctx"}, "recursion_limit": 50},
    )

    assert any("Output of upstream task 'fetch'" in prompt for prompt in seen)


async def test_the_run_produces_a_synthesized_answer_and_a_cost(forecast_run):
    state, _ = forecast_run

    assert state["final_answer"].startswith("**Forecast:")
    records = state["usage_records"]
    assert {r["task_id"] for r in records} >= {"planner", "fetch", "synthesizer"}
    assert sum(r["input_tokens"] for r in records) > 0
    assert any("summary" in r for r in records), "a memory summary is produced for recall"


# --- the human approval gate -------------------------------------------------

async def test_a_write_agent_pauses_the_run_before_running(agents, tools, settings):
    bus = RecordingBus()
    graph = compile_graph(agents, tools, settings, bus, PUBLISH_PLAN)
    config = {"configurable": {"thread_id": "run_hold"}, "recursion_limit": 50}

    state = await graph.ainvoke(
        {"run_id": "run_hold", "goal": "Publish the plan.", "task_outputs": {},
         "usage_records": []},
        config=config,
    )

    assert "__interrupt__" in state, "the graph paused instead of writing"
    pending = state["__interrupt__"][0].value
    assert pending["agent_id"] == "sprint_plan_publisher"
    assert pending["tools"] == ["jira.publish_sprint_plan"]
    assert "publish" not in state.get("task_outputs", {})
    assert not [d for d in bus.of("tool.called") if d["tool"] == "jira.publish_sprint_plan"]


async def test_declining_means_the_write_never_happens(agents, tools, settings):
    bus = RecordingBus()
    graph = compile_graph(agents, tools, settings, bus, PUBLISH_PLAN)
    config = {"configurable": {"thread_id": "run_decline"}, "recursion_limit": 50}
    await graph.ainvoke(
        {"run_id": "run_decline", "goal": "Publish the plan.", "task_outputs": {},
         "usage_records": []},
        config=config,
    )

    state = await graph.ainvoke(
        Command(resume={"approved": False, "reason": "not ready"}), config=config
    )

    assert state["task_outputs"]["publish"]["status"] == "declined"
    assert "not ready" in state["task_outputs"]["publish"]["output"]
    assert not [d for d in bus.of("tool.called") if d["tool"] == "jira.publish_sprint_plan"]
    assert state["final_answer"], "the run still finishes and reports what was skipped"


async def test_approving_lets_the_write_through(agents, tools, settings):
    bus = RecordingBus()
    graph = compile_graph(agents, tools, settings, bus, PUBLISH_PLAN)
    config = {"configurable": {"thread_id": "run_approve"}, "recursion_limit": 50}
    await graph.ainvoke(
        {"run_id": "run_approve", "goal": "Publish the plan.", "task_outputs": {},
         "usage_records": []},
        config=config,
    )

    state = await graph.ainvoke(Command(resume={"approved": True}), config=config)

    assert state["task_outputs"]["publish"]["status"] == "completed"
    assert [d["tool"] for d in bus.of("tool.called")].count("jira.publish_sprint_plan") == 1
