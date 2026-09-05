"""The shared graph state, the plan contract, and the layering rule.

Defines what flows between planner, workers and synthesizer, and the reducers
that make concurrent worker writes safe. `TaskPlan` is the planner's structured
output contract -- the shape the LLM is constrained to and the shape the
registry validates against.

It holds no orchestration logic beyond `ready_tasks`, which is the single
definition of "what can run next".
"""

import re
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field, field_validator

TASK_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class PlanValidationError(ValueError):
    """Raised when a plan is structurally impossible: unknown ids, cycles, dangling deps."""


class TaskSpec(BaseModel):
    """One unit of work: run this agent against this objective."""

    task_id: str = Field(description="Short unique id for this task within the plan.")
    agent_id: str = Field(description="Must be an agent id from the provided menu.")
    objective: str = Field(
        min_length=1, description="What this agent must produce, in one or two sentences."
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="task_ids whose output this task needs. Empty means it can start immediately.",
    )

    @field_validator("task_id")
    @classmethod
    def _check_task_id(cls, value: str) -> str:
        if not TASK_ID_PATTERN.match(value):
            raise ValueError(f"task_id {value!r} must be 1-64 chars of [A-Za-z0-9_-]")
        return value


class TaskPlan(BaseModel):
    """The planner's structured output: a shallow DAG of tasks."""

    reasoning: str = Field(
        default="", description="One or two sentences on why these agents, for the trace."
    )
    tasks: list[TaskSpec] = Field(min_length=1, description="The tasks to run.")


def _merge_outputs(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Merge concurrent worker results. Task ids are unique, so keys never collide."""
    return {**(left or {}), **(right or {})}


def _append_records(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Concatenate usage records from concurrent workers."""
    return [*(left or []), *(right or [])]


class OrchestrationState(TypedDict, total=False):
    """State shared by every node. Concurrent writers only touch reduced keys."""

    run_id: str
    goal: str
    plan: list[dict[str, Any]]
    plan_reasoning: str
    similar_runs: list[dict[str, Any]]

    # Workers fan out with Send and write concurrently, so these need reducers.
    task_outputs: Annotated[dict[str, dict[str, Any]], _merge_outputs]
    usage_records: Annotated[list[dict[str, Any]], _append_records]

    final_answer: str
    failure: str



def validate_plan(plan: TaskPlan, known_agent_ids: set[str]) -> TaskPlan:
    """Reject a plan the graph could not execute, naming exactly what is wrong.

    This is the guard behind "the planner chooses from a retrieved menu": an
    invented agent id fails here, loudly, rather than becoming a mystery at
    execution time.
    """
    task_ids = [task.task_id for task in plan.tasks]
    duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
    if duplicates:
        raise PlanValidationError(f"Plan reuses task_id(s) {duplicates}; task ids must be unique.")

    invented = sorted({t.agent_id for t in plan.tasks if t.agent_id not in known_agent_ids})
    if invented:
        raise PlanValidationError(
            f"Plan names unknown agent id(s) {invented}. Choose only from: "
            f"{', '.join(sorted(known_agent_ids))}"
        )

    known_task_ids = set(task_ids)
    for task in plan.tasks:
        dangling = sorted(set(task.depends_on) - known_task_ids)
        if dangling:
            raise PlanValidationError(
                f"Task {task.task_id!r} depends on task(s) {dangling} that are not in the plan."
            )
        if task.task_id in task.depends_on:
            raise PlanValidationError(f"Task {task.task_id!r} depends on itself.")

    _reject_cycles(plan)
    return plan


def _reject_cycles(plan: TaskPlan) -> None:
    """Fail if the dependency graph cannot be topologically ordered."""
    pending = {task.task_id: set(task.depends_on) for task in plan.tasks}
    settled: set[str] = set()

    while pending:
        ready = [task_id for task_id, deps in pending.items() if deps <= settled]
        if not ready:
            raise PlanValidationError(
                f"Plan has a dependency cycle among tasks {sorted(pending)}."
            )
        settled.update(ready)
        for task_id in ready:
            del pending[task_id]


def ready_tasks(
    plan: list[dict[str, Any]], completed: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """The tasks whose dependencies are all satisfied and which have not run.

    The single definition of "what can run next". Everything it returns is
    independent, so the graph dispatches the whole list in parallel.
    """
    done = set(completed or {})
    return [
        task
        for task in plan
        if task["task_id"] not in done and set(task.get("depends_on", [])) <= done
    ]
