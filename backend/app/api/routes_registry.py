"""Read-only routes exposing what the platform loaded at startup.

`GET /agents` and `GET /tools` let the UI show the registry the planner is
choosing from, which is the evidence that routing is configuration-driven
rather than hardcoded. Both read process state; neither can change anything.
"""

from fastapi import APIRouter, Request

from app.api.schemas import AgentResponse, ToolResponse

router = APIRouter(tags=["registry"])


@router.get("/agents", response_model=list[AgentResponse])
async def list_agents(request: Request) -> list[AgentResponse]:
    """The loaded capability registry, with each envelope's ceiling resolved to real tools.

    These are the envelopes agents are created inside, not agents themselves --
    the agents that actually ran are in each run's trace, because they are
    written at runtime and exist nowhere on disk.
    """
    capabilities = request.app.state.capabilities
    tools = request.app.state.tools
    return [
        AgentResponse(
            id=spec.id,
            role=" ".join(spec.description.split()),
            model=spec.default_model,
            tool_selectors=spec.allowed_tool_selectors,
            bound_tools=[tool.name for tool in tools.select(spec.allowed_tool_selectors)],
            allowed_models=spec.allowed_models,
            max_iterations=spec.max_iterations_limit,
            requires_approval=spec.requires_approval,
        )
        for spec in capabilities.all_specs()
    ]


@router.get("/tools", response_model=list[ToolResponse])
async def list_tools(request: Request) -> list[ToolResponse]:
    """Every tool discovered from the configured MCP servers at startup."""
    return [
        ToolResponse(**tool.describe()) for tool in request.app.state.tools.all_tools()
    ]
