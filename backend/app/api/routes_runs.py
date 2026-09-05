"""HTTP routes for runs: submit, read, stream, approve.

Thin. Every route validates its input, calls the orchestrator or the memory
layer, and returns. Orchestration logic lives in app.orchestrator; this module
exists to turn it into HTTP.
"""

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import ApprovalRequest, CreateRunRequest, CreateRunResponse, RunResponse
from app.memory import retrieval
from app.memory.retrieval import RunNotFoundError

router = APIRouter(tags=["runs"])

# Statuses a run can be resumed from. Anything else is already over.
RESUMABLE_STATUSES = {"awaiting_approval"}


@router.post("/runs", response_model=CreateRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_run(payload: CreateRunRequest, request: Request) -> CreateRunResponse:
    """Submit a goal. Returns immediately; watch /runs/{id}/events for progress."""
    run_id = await request.app.state.orchestrator.start_run(payload.goal)
    return CreateRunResponse(run_id=run_id)


@router.get("/runs/{run_id}", response_model=RunResponse)
async def read_run(run_id: str, request: Request) -> RunResponse:
    """The final state of a run, its plan, its cost and its durable trace."""
    try:
        record = await retrieval.get_run(request.app.state.sessions, run_id)
    except RunNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No run with id {run_id!r}") from None
    return RunResponse.model_validate(record)


@router.get("/runs/{run_id}/events")
async def stream_run_events(run_id: str, request: Request) -> EventSourceResponse:
    """Server-sent events for one run's trace.

    The stream replays everything already published before going live, so a
    client that connects after POST /runs never misses the beginning -- and a
    run that finished before the browser attached still renders in full.
    """
    bus = request.app.state.bus

    async def event_source() -> AsyncIterator[dict[str, Any]]:
        async for event in bus.subscribe(run_id):
            if await request.is_disconnected():
                return
            yield {
                "event": str(event.name),
                "id": str(event.sequence),
                "data": json.dumps({
                    "run_id": event.run_id,
                    "sequence": event.sequence,
                    "name": str(event.name),
                    "created_at": event.created_at.isoformat(),
                    "data": event.data,
                }),
            }

    return EventSourceResponse(event_source())


@router.post("/runs/{run_id}/approve", response_model=CreateRunResponse)
async def approve_run(
    run_id: str, payload: ApprovalRequest, request: Request
) -> CreateRunResponse:
    """Resume a run paused at an approval interrupt, approving or declining it."""
    try:
        record = await retrieval.get_run(request.app.state.sessions, run_id)
    except RunNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No run with id {run_id!r}") from None

    if record["status"] not in RESUMABLE_STATUSES:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Run {run_id!r} is {record['status']!r}, not awaiting approval.",
        )

    await request.app.state.orchestrator.resume_run(run_id, payload.approved, payload.reason)
    return CreateRunResponse(run_id=run_id, status="running")
