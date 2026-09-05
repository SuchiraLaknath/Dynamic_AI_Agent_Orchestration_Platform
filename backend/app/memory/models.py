"""SQLAlchemy tables for episodic and semantic memory.

`runs` is one row per submitted goal -- what was asked, what was planned, what
came back, and what it cost. `steps` is the durable copy of the event trace,
which is what makes a finished run replayable after the in-process bus has
forgotten it.

The working-memory tier is not here: that is LangGraph's Postgres checkpointer,
which owns its own tables in the same database.
"""

from datetime import datetime, timezone
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIMENSIONS = 384


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    """One submitted goal and everything the platform did about it."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    goal: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="running", index=True)

    plan: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    agent_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure: Mapped[str | None] = mapped_column(Text, nullable=True)

    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # Semantic tier: the goal's embedding, searched to find similar past runs.
    goal_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    steps: Mapped[list["Step"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="Step.sequence"
    )

    def as_summary(self) -> dict[str, Any]:
        """The compact view injected into a later planner prompt."""
        return {
            "run_id": self.id,
            "goal": self.goal,
            "agent_ids": self.agent_ids or [],
            "summary": self.summary or "",
        }


class Step(Base):
    """One trace event, durably. Mirrors what went out over SSE."""

    __tablename__ = "steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(48))
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped[Run] = relationship(back_populates="steps")
