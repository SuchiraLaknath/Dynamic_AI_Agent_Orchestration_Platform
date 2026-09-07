"""Drives one compiled graph for one run: events out, state persisted.

Bridges three things that otherwise know nothing about each other -- the
LangGraph execution, the SSE event bus, and the Postgres run history. Starting a
run schedules it as a background task so POST /runs can return a run_id
immediately and the client can attach to the stream.

Approval interrupts are detected here rather than inside the worker node,
because a node that emitted its own `approval.required` would emit it a second
time when the graph replays that node on resume.

It does not decide anything about the plan or the agents.
"""

import asyncio
import uuid
from typing import Any, Awaitable, Callable

from langgraph.types import Command

from app.costs import Usage, sum_usage
from app.events import EventBus, EventName
from app.memory import retrieval
from app.memory.retrieval import SessionFactory


EventEmitter = Callable[[str, EventName, dict[str, Any]], Awaitable[None]]


def build_event_recorder(bus: EventBus, sessions: SessionFactory) -> EventEmitter:
    """Return the emitter every graph node publishes through.

    Publishes to subscribers and durably records the same event -- both, always:
    the SSE stream is the live view and `steps` is the replay, and a trace that
    exists in one but not the other is worse than neither.

    A free function rather than an Orchestrator method so the graph, which needs
    an emitter, can be built before the orchestrator that runs it.
    """

    async def emit(run_id: str, name: EventName, data: dict[str, Any]) -> None:
        event = await bus.publish(run_id, name, data)
        await retrieval.record_step(sessions, run_id, event.sequence, str(name), data)

    return emit


class Orchestrator:
    """Owns run execution for the process. One instance, held on app.state."""

    def __init__(
        self, graph: Any, bus: EventBus, sessions: SessionFactory, emit: EventEmitter
    ) -> None:
        self._graph = graph
        self._bus = bus
        self._sessions = sessions
        self._emit = emit
        self._tasks: set[asyncio.Task[None]] = set()

    async def start_run(self, goal: str) -> str:
        """Create the run, schedule it, and return its id immediately."""
        run_id = f"run_{uuid.uuid4().hex[:16]}"
        await retrieval.start_run(self._sessions, run_id, goal)
        self._schedule(self._execute(run_id, {
            "run_id": run_id, "goal": goal, "task_outputs": {}, "usage_records": []
        }))
        return run_id

    async def resume_run(self, run_id: str, approved: bool, reason: str = "") -> None:
        """Resume a run paused at an approval interrupt."""
        self._schedule(
            self._execute(run_id, Command(resume={"approved": approved, "reason": reason}))
        )

    def _schedule(self, coroutine: Any) -> None:
        """Run in the background, keeping a reference so it is not garbage collected."""
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _execute(self, run_id: str, graph_input: Any) -> None:
        """Execute or resume the graph, then publish the terminal event."""
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 50}
        try:
            state = await self._graph.ainvoke(graph_input, config=config)
        except Exception as error:  # noqa: BLE001 - surfaced to the client as run.failed
            await self._fail_run(run_id, f"{type(error).__name__}: {error}")
            return

        interrupts = state.get("__interrupt__") if isinstance(state, dict) else None
        if interrupts:
            await self._pause_for_approval(run_id, state, interrupts)
            return

        await self._complete_run(run_id, state)

    async def _pause_for_approval(
        self, run_id: str, state: dict[str, Any], interrupts: Any
    ) -> None:
        """Publish what needs approving and leave the run paused.

        The graph's own checkpoint is the pause -- nothing is held in memory
        here, so the process can restart and the run is still resumable.

        The spend so far is recorded rather than left at zero: by the time a run
        pauses it has usually already run several agents, and a run waiting on a
        human should not report that it cost nothing.
        """
        payload = getattr(interrupts[0], "value", {}) or {}
        await self._emit(run_id, EventName.APPROVAL_REQUIRED, dict(payload))
        await retrieval.pause_run(
            self._sessions, run_id, _total_usage(state.get("usage_records", []))
        )

    async def _complete_run(self, run_id: str, state: dict[str, Any]) -> None:
        usage = _total_usage(state.get("usage_records", []))
        answer = state.get("final_answer", "")
        summary = next(
            (record["summary"] for record in state.get("usage_records", []) if "summary" in record),
            answer[:400],
        )
        await retrieval.finish_run(
            self._sessions, run_id, status="completed", usage=usage,
            final_answer=answer, summary=summary,
        )
        await self._emit(run_id, EventName.RUN_COMPLETED, {
            "final_answer": answer,
            "task_outputs": state.get("task_outputs", {}),
            "usage": usage.model_dump(),
        })

    async def _fail_run(self, run_id: str, message: str) -> None:
        # No usage is passed: the graph raised rather than returned, so its
        # accumulated records are unreachable here. Leaving the recorded totals
        # alone keeps whatever a pause already wrote instead of zeroing it.
        await retrieval.finish_run(
            self._sessions, run_id, status="failed", failure=message
        )
        await self._emit(run_id, EventName.RUN_FAILED, {"error": message})


def _total_usage(records: list[dict[str, Any]]) -> Usage:
    """Sum the per-call usage records the nodes accumulated in state."""
    return sum_usage([
        Usage(**{key: value for key, value in record.items() if key in Usage.model_fields})
        for record in records
    ])
