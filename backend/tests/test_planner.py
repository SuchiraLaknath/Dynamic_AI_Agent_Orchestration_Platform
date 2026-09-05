"""The planner contract: choose from the menu, or fail.

The rule this file exists to protect is that the planner never invents an agent
id. Everything downstream assumes a plan only names agents that exist.
"""

import pytest

from app.graph.planner import PlannerError, build_planning_prompt, plan_goal, render_agent_menu
from app.graph.state import PlanValidationError, TaskPlan, validate_plan
from tests.stubs import StubChatModel

GOAL = "Predict the velocity for our next sprint based on previous Jira sprints."


def plan(*tasks: dict) -> TaskPlan:
    return TaskPlan.model_validate({"reasoning": "because", "tasks": list(tasks)})


def task(task_id: str, agent_id: str, depends_on: list[str] | None = None) -> dict:
    return {
        "task_id": task_id,
        "agent_id": agent_id,
        "objective": f"do {task_id}",
        "depends_on": depends_on or [],
    }


async def run_planner(agents, tools, settings, responder, **overrides):
    model = StubChatModel(responder=lambda m: None, structured_responder=responder)
    return await plan_goal(
        goal=overrides.get("goal", GOAL),
        agents=agents,
        tools=tools,
        similar_runs=overrides.get("similar_runs", []),
        model=model,
        agent_top_k=settings.planner_agent_top_k,
        tool_top_k=settings.planner_tool_top_k,
    )


async def test_a_valid_plan_is_returned_with_the_menu_it_chose_from(agents, tools, settings):
    expected = plan(task("t1", "jira_analyst"), task("t2", "velocity_forecaster", ["t1"]))

    result, menu, usage = await run_planner(agents, tools, settings, lambda m: expected)

    assert [t.agent_id for t in result.tasks] == ["jira_analyst", "velocity_forecaster"]
    assert {spec.id for spec in menu} <= agents.agent_ids, "menu comes from the registry"
    assert usage.calls == 1


async def test_an_invented_agent_id_is_rejected_then_retried_once(agents, tools, settings):
    """The retry feeds the error back; a corrected plan on attempt two succeeds."""
    attempts: list = []
    good = plan(task("t1", "jira_analyst"))

    def responder(messages):
        attempts.append(messages)
        if len(attempts) == 1:
            return plan(task("t1", "jira_wizard_9000"))
        return good

    result, _, usage = await run_planner(agents, tools, settings, responder)

    assert result.tasks[0].agent_id == "jira_analyst"
    assert len(attempts) == 2, "exactly one retry"
    assert usage.calls == 2
    # The correction has to name the bad id, or the model cannot fix it.
    correction = str(attempts[1][-1].content)
    assert "jira_wizard_9000" in correction
    assert "jira_analyst" in correction


async def test_an_invented_agent_id_twice_fails_the_run(agents, tools, settings):
    with pytest.raises(PlannerError, match="jira_wizard_9000"):
        await run_planner(agents, tools, settings, lambda m: plan(task("t1", "jira_wizard_9000")))


async def test_unparsable_output_fails_after_the_retry(agents, tools, settings):
    with pytest.raises(PlannerError):
        await run_planner(agents, tools, settings, lambda m: ValueError("not a plan"))


async def test_the_prompt_offers_only_registered_agent_ids(agents, tools, settings):
    menu = agents.search_by_role(GOAL, settings.planner_agent_top_k)
    prompt = build_planning_prompt(GOAL, menu, tools.search(GOAL, 12), [])

    assert GOAL in prompt
    for spec in menu:
        assert repr(spec.id) in prompt, "ids are quoted so the model copies them exactly"
    assert "jira.get_sprints" in prompt, "discovered tool names reach the planner"


async def test_recalled_runs_are_offered_to_the_planner(agents, tools, settings):
    memory = [{"run_id": "run_old", "goal": "Forecast sprint 38", "agent_ids": ["jira_analyst"],
               "summary": "Used a 6-sprint rolling average."}]
    captured: list = []

    def responder(messages):
        captured.append(str(messages[-1].content))
        return plan(task("t1", "jira_analyst"))

    await run_planner(agents, tools, settings, responder, similar_runs=memory)

    assert "rolling average" in captured[0], "semantic memory reaches the planner prompt"


def test_the_agent_menu_shows_capability_and_tools(agents):
    rendered = render_agent_menu(agents.all_specs())

    assert "'jira_analyst'" in rendered
    assert "jira.get_sprints" in rendered
    assert "none (reasons over prior output)" in rendered, "a tool-less agent says so"


# --- plan validation, the guard the planner depends on -----------------------

def test_a_plan_naming_an_unknown_agent_is_invalid():
    with pytest.raises(PlanValidationError, match="unknown agent id"):
        validate_plan(plan(task("t1", "ghost")), {"jira_analyst"})


def test_a_plan_with_a_dangling_dependency_is_invalid():
    with pytest.raises(PlanValidationError, match="not in the plan"):
        validate_plan(plan(task("t1", "jira_analyst", ["missing"])), {"jira_analyst"})


def test_a_cyclic_plan_is_invalid():
    cyclic = plan(task("a", "jira_analyst", ["b"]), task("b", "jira_analyst", ["a"]))
    with pytest.raises(PlanValidationError, match="cycle"):
        validate_plan(cyclic, {"jira_analyst"})


def test_a_self_dependent_task_is_invalid():
    with pytest.raises(PlanValidationError, match="depends on itself"):
        validate_plan(plan(task("a", "jira_analyst", ["a"])), {"jira_analyst"})


def test_duplicate_task_ids_are_invalid():
    with pytest.raises(PlanValidationError, match="unique"):
        validate_plan(plan(task("a", "jira_analyst"), task("a", "jira_analyst")), {"jira_analyst"})


def test_a_shallow_parallel_dag_is_valid():
    valid = plan(
        task("fetch", "jira_analyst"),
        task("forecast", "velocity_forecaster", ["fetch"]),
        task("capacity", "capacity_planner", ["fetch"]),
        task("risk", "risk_reviewer", ["forecast", "capacity"]),
    )
    assert validate_plan(valid, {
        "jira_analyst", "velocity_forecaster", "capacity_planner", "risk_reviewer"
    }) is valid
