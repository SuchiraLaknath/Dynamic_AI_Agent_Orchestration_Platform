"""A whole run, end to end: real mock MCP server, stubbed LLM, real graph.

Asserts on orchestration rather than prose -- which agents were created, in what
order they ran, which tools each one was granted, what the trace contained, and
that the approval gate actually stops a write. The model is the only stub.

The plans below are what a planner would emit: capability ids chosen from the
registry, and agents invented for the run. None of the agent names here appear
in any config file, which is the property the whole design exists for.
"""

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph.build import build_orchestration_graph
from app.graph.state import TaskPlan
from app.graph.worker import build_worker_node
from app.orchestrator import _total_usage
from tests.stubs import RecordingBus, StubChatModel, usage

GOAL = "Predict the velocity for our next sprint based on previous Jira sprints."


def designed(task_id, capability_id, name, prompt, tools, depends_on=None, iterations=3):
    return {
        "task_id": task_id,
        "capability_id": capability_id,
        "objective": f"Produce the {task_id} result.",
        "depends_on": depends_on or [],
        "agent": {
            "name": name,
            "role": f"An agent created for this run to handle {task_id}",
            "system_prompt": prompt,
            "tool_selectors": tools,
            "max_iterations": iterations,
        },
    }


FORECAST_PLAN = TaskPlan.model_validate({
    "reasoning": "Fetch history, then forecast and capacity in parallel, then review risk.",
    "tasks": [
        designed("fetch", "jira_reporting", "sprint_history_reader",
                 "Fetch the closed sprints and tabulate committed versus completed points.",
                 ["jira.get_sprints"]),
        # Same envelope as `fetch`, but a different agent with no tools at all:
        # one capability can yield agents that look nothing like each other.
        designed("forecast", "jira_reporting", "velocity_projector",
                 "Compute a rolling average from the upstream sprint numbers.",
                 [], ["fetch"]),
        designed("capacity", "capacity_analysis", "availability_checker",
                 "Fetch the upcoming sprint capacity and derive an availability ratio.",
                 ["jira.get_team_capacity"], ["fetch"]),
        designed("risk", "analysis_only", "delivery_risk_reviewer",
                 "Review the proposed commitment for delivery risk.",
                 [], ["forecast", "capacity"]),
    ],
})

PUBLISH_PLAN = TaskPlan.model_validate({
    "reasoning": "Fetch, then publish.",
    "tasks": [
        designed("fetch", "jira_reporting", "sprint_history_reader",
                 "Fetch the closed sprints and tabulate committed versus completed points.",
                 ["jira.get_sprints"]),
        designed("publish", "jira_writeback", "commitment_publisher",
                 "Publish the agreed commitment to the board.",
                 ["jira.publish_sprint_plan"], ["fetch"]),
    ],
})

# Every agent name the plans invent. None of these is in agents.yaml.
CREATED_AGENT_NAMES = {
    "sprint_history_reader", "velocity_projector", "availability_checker",
    "delivery_risk_reviewer", "commitment_publisher",
}


def responder(messages):
    """Route by system prompt: one stub serves planner, every worker, and synthesizer.

    It matches on the prompts the plans above wrote, which is only possible
    because those prompts are now part of the plan rather than the config.
    """
    system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
    already_used_a_tool = any(isinstance(m, ToolMessage) for m in messages)

    if "final answer for a user" in system:
        return AIMessage(content="**Forecast: 37 points for SPR-39.**", usage_metadata=usage())
    if "Summarize, in at most two sentences" in system:
        return AIMessage(content="Forecast next-sprint velocity at ~37 points.",
                         usage_metadata=usage())

    if not already_used_a_tool:
        if "Fetch the closed sprints" in system:
            return AIMessage(content="", usage_metadata=usage(), tool_calls=[
                {"name": "get_sprints", "args": {"limit": 6}, "id": "c1"}])
        if "Fetch the upcoming sprint capacity" in system:
            return AIMessage(content="", usage_metadata=usage(), tool_calls=[
                {"name": "get_team_capacity", "args": {}, "id": "c2"}])
        if "Publish the agreed commitment" in system:
            return AIMessage(content="", usage_metadata=usage(), tool_calls=[
                {"name": "publish_sprint_plan",
                 "args": {"sprint_id": "SPR-39", "committed_points": 37, "summary": "avg"},
                 "id": "c3"}])

    return AIMessage(content="Answered from the data I was given.", usage_metadata=usage())


async def no_memory(_goal: str) -> list:
    return []


async def ignore_plan(_run_id: str, _plan: list) -> None:
    return None


def compile_graph(capabilities, tools, settings, bus, plan: TaskPlan):
    return build_orchestration_graph(
        capabilities=capabilities,
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
async def forecast_run(capabilities, tools, settings):
    bus = RecordingBus()
    graph = compile_graph(capabilities, tools, settings, bus, FORECAST_PLAN)
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
    tool_events = [name for name in names if name.startswith("tool.")]
    assert tool_events == ["tool.called", "tool.result"] * (len(tool_events) // 2)


async def test_run_started_reports_the_plan_and_the_capabilities_considered(
    forecast_run, capabilities
):
    _, bus = forecast_run
    started = bus.of("run.started")[0]

    assert started["goal"] == GOAL
    assert [t["task_id"] for t in started["plan"]] == ["fetch", "forecast", "capacity", "risk"]
    assert started["reasoning"]
    assert set(started["considered_agents"]) <= capabilities.capability_ids


# --- the agents that ran were created for this run --------------------------


async def test_the_agents_that_ran_exist_nowhere_in_config(forecast_run, capabilities):
    """The central claim: no agent that ran is a configured record."""
    _, bus = forecast_run
    ran = {data["agent_id"] for data in bus.of("task.started")}

    assert ran == {"sprint_history_reader", "velocity_projector",
                   "availability_checker", "delivery_risk_reviewer"}
    assert not (ran & capabilities.capability_ids), "agent names are not capability ids"


async def test_the_trace_records_each_generated_prompt(forecast_run):
    """The prompt is in no file, so the trace is the only audit record of it."""
    _, bus = forecast_run
    started = {data["agent_id"]: data for data in bus.of("task.started")}

    assert "Fetch the closed sprints" in started["sprint_history_reader"]["system_prompt"]
    assert started["sprint_history_reader"]["capability_id"] == "jira_reporting"
    assert started["delivery_risk_reviewer"]["capability_id"] == "analysis_only"
    for data in started.values():
        assert data["role"], "every created agent reports what it was made for"


async def test_one_capability_produced_two_different_agents(forecast_run):
    """`fetch` and `forecast` share an envelope but are not the same agent."""
    _, bus = forecast_run
    started = {data["agent_id"]: data for data in bus.of("task.started")}

    assert started["sprint_history_reader"]["capability_id"] == "jira_reporting"
    assert started["velocity_projector"]["capability_id"] == "jira_reporting"
    assert started["sprint_history_reader"]["tools"] == ["jira.get_sprints"]
    assert started["velocity_projector"]["tools"] == [], "it asked for no tools, so it has none"


async def test_the_capability_policy_reaches_the_agent(forecast_run, capabilities):
    _, bus = forecast_run
    started = {data["agent_id"]: data for data in bus.of("task.started")}
    policy = capabilities.get("jira_reporting").policy.strip()

    assert policy in started["sprint_history_reader"]["system_prompt"]


async def test_every_planned_agent_ran_and_completed(forecast_run):
    state, _ = forecast_run
    outputs = state["task_outputs"]

    assert set(outputs) == {"fetch", "forecast", "capacity", "risk"}
    assert all(result["status"] == "completed" for result in outputs.values())
    assert outputs["fetch"]["agent_id"] == "sprint_history_reader"
    assert outputs["risk"]["agent_id"] == "delivery_risk_reviewer"


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


async def test_each_agent_reached_only_the_tools_it_was_granted(forecast_run):
    _, bus = forecast_run
    called = {(data["agent_id"], data["tool"]) for data in bus.of("tool.called")}

    assert ("sprint_history_reader", "jira.get_sprints") in called
    assert ("availability_checker", "jira.get_team_capacity") in called
    # Two agents were granted no tools, and nobody touched the write tool.
    assert not [pair for pair in called if pair[0] == "delivery_risk_reviewer"]
    assert not [pair for pair in called if pair[0] == "velocity_projector"]
    assert not [pair for pair in called if pair[1] == "jira.publish_sprint_plan"]


async def test_real_fixture_data_came_back_through_mcp(forecast_run):
    _, bus = forecast_run
    previews = " ".join(data["preview"] for data in bus.of("tool.result"))

    assert "SPR-" in previews, "sprint ids came from the mock server's fixture"
    assert all(data["ok"] for data in bus.of("tool.result"))


async def test_upstream_output_is_handed_to_the_dependent_task(capabilities, tools, settings):
    """A dependency's answer must reach its dependent's prompt, not be re-fetched."""
    seen: list[str] = []
    bus = RecordingBus()

    def capturing_responder(messages):
        seen.append(" ".join(str(m.content) for m in messages))
        return responder(messages)

    graph = build_orchestration_graph(
        capabilities=capabilities, tools=tools, settings=settings, emit=bus.emit,
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

async def test_a_write_agent_pauses_the_run_before_running(capabilities, tools, settings):
    bus = RecordingBus()
    graph = compile_graph(capabilities, tools, settings, bus, PUBLISH_PLAN)
    config = {"configurable": {"thread_id": "run_hold"}, "recursion_limit": 50}

    state = await graph.ainvoke(
        {"run_id": "run_hold", "goal": "Publish the plan.", "task_outputs": {},
         "usage_records": []},
        config=config,
    )

    assert "__interrupt__" in state, "the graph paused instead of writing"
    pending = state["__interrupt__"][0].value
    assert pending["agent_id"] == "commitment_publisher"
    assert pending["capability_id"] == "jira_writeback"
    assert pending["tools"] == ["jira.publish_sprint_plan"]
    # The reviewer sees the generated prompt, since it is what will actually run.
    assert "Publish the agreed commitment" in pending["system_prompt"]
    assert "publish" not in state.get("task_outputs", {})
    assert not [d for d in bus.of("tool.called") if d["tool"] == "jira.publish_sprint_plan"]


async def test_a_paused_run_has_already_spent_money(capabilities, tools, settings):
    """A run pauses mid-flight, so its recorded cost must not be zero.

    The state carried into the pause is what the run had spent getting there;
    reporting nothing would make an approval queue look free.
    """
    bus = RecordingBus()
    graph = compile_graph(capabilities, tools, settings, bus, PUBLISH_PLAN)

    state = await graph.ainvoke(
        {"run_id": "run_cost", "goal": "Publish the plan.", "task_outputs": {},
         "usage_records": []},
        config={"configurable": {"thread_id": "run_cost"}, "recursion_limit": 50},
    )

    assert "__interrupt__" in state
    spent = _total_usage(state.get("usage_records", []))
    assert spent.cost_usd > 0, "the planner and the fetch task both cost money"
    assert spent.input_tokens > 0


async def test_declining_means_the_write_never_happens(capabilities, tools, settings):
    bus = RecordingBus()
    graph = compile_graph(capabilities, tools, settings, bus, PUBLISH_PLAN)
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


async def test_approving_lets_the_write_through(capabilities, tools, settings):
    bus = RecordingBus()
    graph = compile_graph(capabilities, tools, settings, bus, PUBLISH_PLAN)
    config = {"configurable": {"thread_id": "run_approve"}, "recursion_limit": 50}
    await graph.ainvoke(
        {"run_id": "run_approve", "goal": "Publish the plan.", "task_outputs": {},
         "usage_records": []},
        config=config,
    )

    state = await graph.ainvoke(Command(resume={"approved": True}), config=config)

    assert state["task_outputs"]["publish"]["status"] == "completed"
    assert [d["tool"] for d in bus.of("tool.called")].count("jira.publish_sprint_plan") == 1


async def test_the_planner_catches_an_escalation_before_the_run_starts(
    capabilities, tools, settings
):
    """First line of defence: the whole run fails at planning, having spent nothing."""
    escalating = TaskPlan.model_validate({
        "reasoning": "escalate",
        "tasks": [designed("sneak", "analysis_only", "quiet_escalator",
                           "Publish something you were never allowed to publish.",
                           ["jira.publish_sprint_plan"])],
    })
    bus = RecordingBus()
    graph = compile_graph(capabilities, tools, settings, bus, escalating)

    with pytest.raises(Exception, match="does not permit"):
        await graph.ainvoke(
            {"run_id": "run_esc", "goal": "Sneak a write through.", "task_outputs": {},
             "usage_records": []},
            config={"configurable": {"thread_id": "run_esc"}, "recursion_limit": 50},
        )

    assert not bus.of("task.started"), "no agent ever ran"
    assert not bus.of("tool.called"), "no tool was ever called"


async def test_the_worker_refuses_a_plan_that_bypassed_the_planner(
    capabilities, tools, settings
):
    """Second line of defence, tested where it lives.

    The planner already checks envelopes, so the only way to reach the worker
    with a bad agent is for the plan to have been altered after planning. The
    worker must refuse rather than trust the payload it is handed.
    """
    bus = RecordingBus()
    node = build_worker_node(
        capabilities, tools, bus.emit,
        chat_model_builder=lambda model_id: StubChatModel(
            responder=responder, structured_responder=lambda m: None
        ),
    )

    result = await node({
        "run_id": "run_tamper",
        "goal": "Sneak a write through.",
        "task_id": "sneak",
        "capability_id": "analysis_only",
        "agent": {
            "name": "quiet_escalator",
            "role": "An agent that should never have been allowed to exist here",
            "system_prompt": "Publish something you were never allowed to publish.",
            "tool_selectors": ["jira.publish_sprint_plan"],
            "max_iterations": 2,
        },
        "objective": "Publish it anyway.",
        "dependency_outputs": {},
    })

    assert result["task_outputs"]["sneak"]["status"] == "failed"
    assert "does not permit" in result["task_outputs"]["sneak"]["output"]
    assert not bus.of("task.started"), "it never got as far as starting"
    assert not bus.of("tool.called"), "the write tool was never reached"
