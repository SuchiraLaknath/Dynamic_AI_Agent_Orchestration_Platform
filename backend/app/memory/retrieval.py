"""Reads and writes for the episodic and semantic memory tiers.

Every database access the platform makes outside the LangGraph checkpointer
goes through these functions: recording a run and its trace as it happens, and
finding similar past runs by goal embedding to inject into a later plan.

It contains no orchestration logic and never decides what to do with what it
retrieves -- that is the planner's job.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.costs import Usage
from app.embeddings import embed_text
from app.memory.models import Run, Step

SessionFactory = async_sessionmaker[AsyncSession]


class RunNotFoundError(KeyError):
    """Raised when a run id is not in the database."""


async def start_run(sessions: SessionFactory, run_id: str, goal: str) -> None:
    """Insert the run row. Its goal is embedded now so it is searchable later."""
    async with sessions() as session, session.begin():
        session.add(Run(id=run_id, goal=goal, status="running", goal_embedding=embed_text(goal)))


async def record_step(
    sessions: SessionFactory, run_id: str, sequence: int, name: str, data: dict[str, Any]
) -> None:
    """Append one trace event durably, mirroring what went out over SSE."""
    async with sessions() as session, session.begin():
        session.add(Step(run_id=run_id, sequence=sequence, name=name, data=data))


async def record_plan(
    sessions: SessionFactory, run_id: str, plan: list[dict[str, Any]]
) -> None:
    """Store the plan and the agents it selected, once the planner has committed."""
    async with sessions() as session, session.begin():
        run = await session.get(Run, run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        run.plan = plan
        run.agent_ids = sorted({task["agent_id"] for task in plan})


async def finish_run(
    sessions: SessionFactory,
    run_id: str,
    status: str,
    usage: Usage,
    final_answer: str | None = None,
    failure: str | None = None,
    summary: str | None = None,
) -> None:
    """Close out a run with its outcome and what it cost."""
    async with sessions() as session, session.begin():
        run = await session.get(Run, run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        run.status = status
        run.final_answer = final_answer
        run.failure = failure
        run.summary = summary
        run.input_tokens = usage.input_tokens
        run.output_tokens = usage.output_tokens
        run.cost_usd = usage.cost_usd
        run.completed_at = datetime.now(timezone.utc)


async def get_run(sessions: SessionFactory, run_id: str) -> dict[str, Any]:
    """The full record of one run, including its durable trace."""
    async with sessions() as session:
        run = await session.get(Run, run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        steps = (
            await session.scalars(
                select(Step).where(Step.run_id == run_id).order_by(Step.sequence)
            )
        ).all()
        return {
            "run_id": run.id,
            "goal": run.goal,
            "status": run.status,
            "plan": run.plan or [],
            "agent_ids": run.agent_ids or [],
            "final_answer": run.final_answer,
            "failure": run.failure,
            "usage": {
                "input_tokens": run.input_tokens,
                "output_tokens": run.output_tokens,
                "cost_usd": run.cost_usd,
            },
            "created_at": run.created_at,
            "completed_at": run.completed_at,
            "trace": [
                {
                    "sequence": step.sequence,
                    "name": step.name,
                    "data": step.data,
                    "created_at": step.created_at,
                }
                for step in steps
            ],
        }


async def find_similar_runs(
    sessions: SessionFactory, goal: str, limit: int
) -> list[dict[str, Any]]:
    """The most similar previously-completed runs, by goal embedding.

    Ordered by pgvector's cosine distance operator. Only completed runs are
    returned -- an in-flight or failed run has nothing useful to teach the
    planner. A database that is not reachable yields no memory rather than
    failing the run: memory is an optimization, not a dependency.
    """
    vector = embed_text(goal)
    try:
        async with sessions() as session:
            statement = (
                select(Run)
                .where(Run.status == "completed", Run.goal_embedding.is_not(None))
                .order_by(Run.goal_embedding.cosine_distance(vector))
                .limit(limit)
            )
            runs = (await session.scalars(statement)).all()
            return [run.as_summary() for run in runs]
    except Exception as error:  # noqa: BLE001 - degraded, and said so
        print(f"[memory] similar-run lookup unavailable, planning without it: {error}")
        return []
