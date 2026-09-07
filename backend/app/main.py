"""FastAPI application: startup wiring and route mounting.

Owns the process lifecycle. On boot it loads the two YAML registries, opens a
session to every configured MCP server and discovers their tools, prepares the
database, and compiles the graph -- then hands those objects to the routers
through `app.state`. On shutdown it closes the MCP sessions and the checkpointer
connection pool.

It contains no orchestration logic. Everything here is wiring.
"""

from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agents.registry import load_capability_registry
from app.api import routes_registry, routes_runs
from app.events import EventBus
from app.graph.build import build_orchestration_graph
from app.mcp.manager import McpManager, load_server_connections
from app.mcp.registry import ToolRegistry
from app.memory import retrieval
from app.memory.db import build_engine, build_session_factory, create_schema, to_checkpointer_dsn
from app.orchestrator import Orchestrator, build_event_recorder
from app.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build every long-lived object once, in dependency order."""
    settings = get_settings()
    shutdown = AsyncExitStack()

    # Registries first: a malformed agents.yaml should stop the process here,
    # with a message naming the file, rather than at the first request.
    capabilities = load_capability_registry(settings.agents_config_path)

    mcp_manager = McpManager(load_server_connections(settings.mcp_config_path))
    shutdown.push_async_callback(mcp_manager.aclose)
    tools = ToolRegistry(await mcp_manager.discover_tools())

    engine = build_engine(settings.database_url)
    shutdown.push_async_callback(engine.dispose)
    await create_schema(engine)
    sessions = build_session_factory(engine)

    checkpointer = await shutdown.enter_async_context(
        AsyncPostgresSaver.from_conn_string(to_checkpointer_dsn(settings.database_url))
    )
    await checkpointer.setup()

    bus = EventBus()

    async def find_similar_runs(goal: str) -> list[dict[str, Any]]:
        return await retrieval.find_similar_runs(sessions, goal, settings.planner_memory_top_k)

    async def record_plan(run_id: str, plan: list[dict[str, Any]]) -> None:
        await retrieval.record_plan(sessions, run_id, plan)

    emit = build_event_recorder(bus, sessions)
    graph = build_orchestration_graph(
        capabilities=capabilities,
        tools=tools,
        settings=settings,
        emit=emit,
        find_similar_runs=find_similar_runs,
        record_plan=record_plan,
        checkpointer=checkpointer,
    )
    orchestrator = Orchestrator(graph=graph, bus=bus, sessions=sessions, emit=emit)

    app.state.settings = settings
    app.state.capabilities = capabilities
    app.state.tools = tools
    app.state.bus = bus
    app.state.sessions = sessions
    app.state.orchestrator = orchestrator

    print(
        f"[startup] {len(capabilities)} capabilities, {len(tools)} tools from "
        f"{len(mcp_manager.server_names)} MCP server(s): {', '.join(sorted(tools.tool_names))}"
    )
    try:
        yield
    finally:
        await shutdown.aclose()


app = FastAPI(
    title="Dynamic AI Agent Orchestration Platform",
    version="0.1.0",
    description=(
        "Submit a natural-language goal. The planner designs specialist agents for it at "
        "runtime -- writing each one's prompt and choosing its tools -- inside the "
        "permission envelopes declared in YAML, runs them as a DAG, and returns a "
        "synthesized answer with a full execution trace."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_runs.router)
app.include_router(routes_registry.router)


@app.get("/health", tags=["meta"])
async def read_health() -> dict[str, str]:
    """Liveness probe, used by the compose healthcheck."""
    return {"status": "ok"}
