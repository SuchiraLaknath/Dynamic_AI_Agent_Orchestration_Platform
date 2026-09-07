"""The planner contract: design agents inside the menu's envelopes, or fail.

Two rules this file protects. The planner never invents a capability id, and an
agent it designs never reaches outside that capability's envelope. Everything
downstream assumes both, and both are checked at planning time so the failure is
retryable rather than fatal mid-run.
"""

import pytest

from app.graph.planner import (
    PlannerError,
    build_planning_prompt,
    plan_goal,
    render_capability_menu,
)
from app.graph.state import PlanValidationError, TaskPlan, validate_plan
from tests.stubs import StubChatModel

GOAL = "Predict the velocity for our next sprint based on previous Jira sprints."


def plan(*tasks: dict) -> TaskPlan:
    return TaskPlan.model_validate({"reasoning": "because", "tasks": list(tasks)})


def task(
    task_id: str,
    capability_id: str,
    depends_on: list[str] | None = None,
    *,
    name: str | None = None,
    tool_selectors: list[str] | None = None,
    **agent_overrides,
) -> dict:
    """A task whose agent is designed at 'runtime', as the planner would emit it."""
    return {
        "task_id": task_id,
        "capability_id": capability_id,
        "objective": f"do {task_id}",
        "depends_on": depends_on or [],
        "agent": {
            "name": name or f"{task_id}_agent",
            "role": f"An agent created to do {task_id} for this particular run",
            "system_prompt": f"Do {task_id}. Show your working.",
            "tool_selectors": tool_selectors if tool_selectors is not None else [],
            "max_iterations": 3,
            **agent_overrides,
        },
    }


async def run_planner(capabilities, tools, settings, responder, **overrides):
    model = StubChatModel(responder=lambda m: None, structured_responder=responder)
    return await plan_goal(
        goal=overrides.get("goal", GOAL),
        capabilities=capabilities,
        tools=tools,
        similar_runs=overrides.get("similar_runs", []),
        model=model,
        capability_top_k=settings.planner_agent_top_k,
        tool_top_k=settings.planner_tool_top_k,
    )


async def test_a_valid_plan_is_returned_with_the_menu_it_chose_from(
    capabilities, tools, settings
):
    expected = plan(
        task("t1", "jira_reporting", tool_selectors=["jira.get_sprints"]),
        task("t2", "analysis_only", ["t1"]),
    )

    result, menu, usage = await run_planner(capabilities, tools, settings, lambda m: expected)

    assert [t.capability_id for t in result.tasks] == ["jira_reporting", "analysis_only"]
    assert {spec.id for spec in menu} <= capabilities.capability_ids
    assert usage.calls == 1


async def test_the_planner_creates_agents_that_are_not_in_any_config(
    capabilities, tools, settings
):
    """The point of the design: these agents exist only for this run."""
    designed = plan(
        task(
            "t1",
            "jira_reporting",
            name="sprint_history_reader",
            tool_selectors=["jira.get_sprints"],
            system_prompt="Pull six sprints and tabulate committed versus completed.",
        )
    )

    result, _, _ = await run_planner(capabilities, tools, settings, lambda m: designed)

    agent = result.tasks[0].agent
    assert agent.name == "sprint_history_reader"
    assert agent.name not in capabilities.capability_ids
    assert "tabulate committed versus completed" in agent.system_prompt


# --- the two things the planner must never get away with --------------------


async def test_an_invented_capability_id_is_rejected_then_retried_once(
    capabilities, tools, settings
):
    """The retry feeds the error back; a corrected plan on attempt two succeeds."""
    attempts: list = []
    good = plan(task("t1", "jira_reporting", tool_selectors=["jira.get_sprints"]))

    def responder(messages):
        attempts.append(messages)
        if len(attempts) == 1:
            return plan(task("t1", "jira_wizard_9000"))
        return good

    result, _, usage = await run_planner(capabilities, tools, settings, responder)

    assert result.tasks[0].capability_id == "jira_reporting"
    assert len(attempts) == 2, "exactly one retry"
    assert usage.calls == 2
    correction = str(attempts[1][-1].content)
    assert "jira_wizard_9000" in correction, "the correction must name the bad id"


async def test_an_agent_escaping_its_envelope_is_rejected_at_planning_time(
    capabilities, tools, settings
):
    """A read-only capability cannot be handed the write tool, even by a valid plan.

    Caught here rather than at execution, so it costs a retry instead of a
    half-finished run.
    """
    attempts: list = []

    def responder(messages):
        attempts.append(messages)
        if len(attempts) == 1:
            return plan(
                task("t1", "jira_reporting", tool_selectors=["jira.publish_sprint_plan"])
            )
        return plan(task("t1", "jira_reporting", tool_selectors=["jira.get_sprints"]))

    result, _, _ = await run_planner(capabilities, tools, settings, responder)

    assert len(attempts) == 2, "the escalation forced a retry"
    assert result.tasks[0].agent.tool_selectors == ["jira.get_sprints"]
    correction = str(attempts[1][-1].content)
    assert "jira.publish_sprint_plan" in correction


async def test_a_persistent_escalation_fails_the_run(capabilities, tools, settings):
    with pytest.raises(PlannerError, match="publish_sprint_plan"):
        await run_planner(
            capabilities,
            tools,
            settings,
            lambda m: plan(
                task("t1", "analysis_only", tool_selectors=["jira.publish_sprint_plan"])
            ),
        )


async def test_an_invented_capability_id_twice_fails_the_run(capabilities, tools, settings):
    with pytest.raises(PlannerError, match="jira_wizard_9000"):
        await run_planner(
            capabilities, tools, settings, lambda m: plan(task("t1", "jira_wizard_9000"))
        )


async def test_unparsable_output_fails_after_the_retry(capabilities, tools, settings):
    with pytest.raises(PlannerError):
        await run_planner(capabilities, tools, settings, lambda m: ValueError("not a plan"))


# --- the menu ---------------------------------------------------------------


async def test_the_prompt_offers_only_registered_capability_ids(capabilities, tools, settings):
    menu = capabilities.search_by_capability(GOAL, settings.planner_agent_top_k)
    prompt = build_planning_prompt(GOAL, menu, tools, tools.search(GOAL, 12), [])

    assert GOAL in prompt
    for spec in menu:
        assert repr(spec.id) in prompt, "ids are quoted so the model copies them exactly"
    assert "jira.get_sprints" in prompt, "discovered tool names reach the planner"


async def test_the_menu_shows_resolved_tools_not_globs(capabilities, tools):
    """The planner is checked against resolved names, so it must be shown those."""
    rendered = render_capability_menu(capabilities.all_specs(), tools)

    assert "'jira_reporting'" in rendered
    assert "jira.get_sprints" in rendered
    assert "none (reasons over prior output)" in rendered, "a tool-less envelope says so"
    assert "a human must approve" in rendered, "the approval envelope is flagged"


async def test_recalled_runs_are_offered_to_the_planner(capabilities, tools, settings):
    memory = [{"run_id": "run_old", "goal": "Forecast sprint 38",
               "agent_ids": ["jira_reporting"], "summary": "Used a 6-sprint rolling average."}]
    captured: list = []

    def responder(messages):
        captured.append(str(messages[-1].content))
        return plan(task("t1", "jira_reporting", tool_selectors=["jira.get_sprints"]))

    await run_planner(capabilities, tools, settings, responder, similar_runs=memory)

    assert "rolling average" in captured[0], "semantic memory reaches the planner prompt"


# --- plan validation, the structural guard ----------------------------------

def test_a_plan_naming_an_unknown_capability_is_invalid():
    with pytest.raises(PlanValidationError, match="unknown capability id"):
        validate_plan(plan(task("t1", "ghost")), {"jira_reporting"})


def test_a_plan_with_a_dangling_dependency_is_invalid():
    with pytest.raises(PlanValidationError, match="not in the plan"):
        validate_plan(plan(task("t1", "jira_reporting", ["missing"])), {"jira_reporting"})


def test_a_cyclic_plan_is_invalid():
    cyclic = plan(task("a", "jira_reporting", ["b"]), task("b", "jira_reporting", ["a"]))
    with pytest.raises(PlanValidationError, match="cycle"):
        validate_plan(cyclic, {"jira_reporting"})


def test_a_self_dependent_task_is_invalid():
    with pytest.raises(PlanValidationError, match="depends on itself"):
        validate_plan(plan(task("a", "jira_reporting", ["a"])), {"jira_reporting"})


def test_duplicate_task_ids_are_invalid():
    doubled = plan(task("a", "jira_reporting"), task("a", "jira_reporting"))
    with pytest.raises(PlanValidationError, match="unique"):
        validate_plan(doubled, {"jira_reporting"})


def test_a_shallow_parallel_dag_is_valid():
    valid = plan(
        task("fetch", "jira_reporting"),
        task("forecast", "jira_reporting", ["fetch"]),
        task("capacity", "capacity_analysis", ["fetch"]),
        task("risk", "analysis_only", ["forecast", "capacity"]),
    )
    assert validate_plan(valid, {"jira_reporting", "capacity_analysis", "analysis_only"}) is valid
