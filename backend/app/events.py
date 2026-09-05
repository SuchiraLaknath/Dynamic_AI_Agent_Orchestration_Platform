"""In-process event bus: one publisher, many SSE subscribers, per run.

Fans run events out to every open SSE connection for that run, and keeps a
replay buffer so a client that connects after POST /runs still receives the
events it missed. Event names are a fixed vocabulary -- see EventName -- because
the frontend switches on them.

It is deliberately in-process. A single API process needs no broker, and adding
one would put a box on the architecture diagram that nothing justifies. The
cost is that this does not survive a restart or fan out across replicas; the
durable copy of a trace is the `steps` table, and the SSE stream is the live
view of it.
"""

import asyncio
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field


class EventName(StrEnum):
    """The complete wire vocabulary. The frontend switches on these values."""

    RUN_STARTED = "run.started"
    TASK_STARTED = "task.started"
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    APPROVAL_REQUIRED = "approval.required"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"


TERMINAL_EVENTS = {EventName.RUN_COMPLETED, EventName.RUN_FAILED}


class RunEvent(BaseModel):
    """One entry in a run's trace, as it goes over the wire and into `steps`."""

    run_id: str
    sequence: int
    name: EventName
    created_at: datetime
    data: dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """Per-run fan-out with replay. One instance per process, held on app.state."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[RunEvent | None]]] = {}
        self._history: dict[str, list[RunEvent]] = {}
        self._finished: set[str] = set()

    async def publish(self, run_id: str, name: EventName, data: dict[str, Any]) -> RunEvent:
        """Record an event and hand it to every current subscriber."""
        history = self._history.setdefault(run_id, [])
        event = RunEvent(
            run_id=run_id,
            sequence=len(history),
            name=name,
            created_at=datetime.now(timezone.utc),
            data=data,
        )
        history.append(event)

        for queue in self._subscribers.get(run_id, []):
            await queue.put(event)

        if name in TERMINAL_EVENTS:
            self._finished.add(run_id)
            for queue in self._subscribers.get(run_id, []):
                await queue.put(None)
        return event

    async def subscribe(self, run_id: str) -> AsyncIterator[RunEvent]:
        """Yield a run's events: everything already published, then live ones.

        Replaying history first is what makes the POST-then-connect sequence
        safe -- without it a fast run would finish before the browser opened the
        stream, and the client would hang on a run that was already over.
        """
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        replayed = list(self._history.get(run_id, []))
        self._subscribers.setdefault(run_id, []).append(queue)

        try:
            for event in replayed:
                yield event

            if run_id in self._finished and replayed:
                return

            while True:
                event = await queue.get()
                if event is None:
                    return
                # Live events that arrived during replay would otherwise be sent
                # twice; the sequence number is what makes that detectable.
                if event.sequence >= len(replayed):
                    yield event
        finally:
            subscribers = self._subscribers.get(run_id, [])
            if queue in subscribers:
                subscribers.remove(queue)
            if not subscribers:
                self._subscribers.pop(run_id, None)

    def history(self, run_id: str) -> list[RunEvent]:
        """Everything published for a run so far, oldest first."""
        return list(self._history.get(run_id, []))

    def forget(self, run_id: str) -> None:
        """Drop a finished run's buffer. The durable trace lives in Postgres."""
        self._history.pop(run_id, None)
        self._finished.discard(run_id)
