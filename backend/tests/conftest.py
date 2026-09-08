"""Shared fixtures.

The MCP fixture starts the real mock Jira server over stdio rather than faking
it, so tool discovery, schema parsing and tool invocation are all exercised for
real. The LLM is the only thing stubbed.
"""

from pathlib import Path
from typing import AsyncIterator

import pytest

from app import embeddings
from app.agents.registry import CapabilityRegistry, load_capability_registry
from app.mcp.manager import McpManager, load_server_connections
from app.mcp.registry import ToolRegistry
from app.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    """Keep the suite hermetic: no test embeds over the network unless it asks.

    Both registry fixtures embed every description when they are constructed, so
    with a key configured the whole suite would otherwise make live API calls on
    every run -- slow, costly, and dependent on the network. Tests that mean to
    exercise the hosted path opt back in explicitly.
    """
    monkeypatch.setattr(embeddings, "uses_hosted_embeddings", lambda: False)
    embeddings._cache.clear()
    yield
    embeddings._cache.clear()


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def capabilities(settings: Settings) -> CapabilityRegistry:
    """The project's real agents.yaml, so tests fail if it drifts out of shape."""
    return load_capability_registry(settings.agents_config_path)


@pytest.fixture
async def tools(settings: Settings) -> AsyncIterator[ToolRegistry]:
    """Tools discovered from the real mock MCP server."""
    manager = McpManager(load_server_connections(settings.mcp_config_path))
    try:
        yield ToolRegistry(await manager.discover_tools())
    finally:
        await manager.aclose()


@pytest.fixture
def write_agents_yaml(tmp_path: Path):
    """Write an agents.yaml into a temp dir and return its path."""

    def write(content: str) -> Path:
        path = tmp_path / "agents.yaml"
        path.write_text(content)
        return path

    return write
