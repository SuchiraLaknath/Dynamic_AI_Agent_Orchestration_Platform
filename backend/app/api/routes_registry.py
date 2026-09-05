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
    """The loaded agent registry, with each agent's selectors resolved to real tools."""
    agents = request.app.state.agents
    tools = request.app.state.tools
    return [
        AgentResponse(
            id=spec.id,
            role=" ".join(spec.role.split()),
            model=spec.model,
            tool_selectors=spec.tool_selectors,
            bound_tools=[tool.name for tool in tools.select(spec.tool_selectors)],
            max_iterations=spec.max_iterations,
            requires_approval=spec.requires_approval,
        )
        for spec in agents.all_specs()
    ]


@router.get("/tools", response_model=list[ToolResponse])
async def list_tools(request: Request) -> list[ToolResponse]:
    """Every tool discovered from the configured MCP servers at startup."""
    return [
        ToolResponse(**tool.describe()) for tool in request.app.state.tools.all_tools()
    ]
