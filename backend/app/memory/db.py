"""Database engine, session factory, and first-run schema creation.

Owns the single async engine for the process and the one place the pgvector
extension and tables get created. Also converts the SQLAlchemy URL into the
plain DSN that LangGraph's Postgres checkpointer needs, so both tiers are
configured from one environment variable.

Schema is created with `create_all` rather than a migration tool: this is a
single-service project with no deployed history to migrate from. A real
deployment would put Alembic here.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.memory.models import Base


class DatabaseUnavailableError(RuntimeError):
    """Raised when the database cannot be reached or prepared at startup."""


def build_engine(database_url: str) -> AsyncEngine:
    """Create the process-wide async engine."""
    return create_async_engine(database_url, pool_pre_ping=True, future=True)


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def to_checkpointer_dsn(database_url: str) -> str:
    """Strip SQLAlchemy's driver suffix for LangGraph's checkpointer.

    SQLAlchemy addresses a driver (`postgresql+psycopg://`); psycopg, which the
    checkpointer uses directly, wants a plain libpq DSN.
    """
    return database_url.replace("postgresql+psycopg://", "postgresql://").replace(
        "postgresql+asyncpg://", "postgresql://"
    )


async def create_schema(engine: AsyncEngine) -> None:
    """Enable pgvector and create the episodic and semantic tables.

    Idempotent, so it is safe on every boot.
    """
    try:
        async with engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(Base.metadata.create_all)
    except Exception as error:  # noqa: BLE001 - re-raised as a named startup failure
        raise DatabaseUnavailableError(
            f"Could not prepare the database. Is Postgres up with the pgvector extension "
            f"available? Underlying error: {error}"
        ) from error
