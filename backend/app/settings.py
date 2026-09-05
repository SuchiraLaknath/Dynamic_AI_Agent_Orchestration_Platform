"""Process configuration, read once from the environment.

Loads every tunable the backend has into a single frozen object. It does not
read the agent or MCP registries -- those are YAML files owned by
`app.agents.registry` and `app.mcp.manager`; this module only says where they
live.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Every environment variable the backend reads. See `.env.example`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""

    # The planner and synthesizer are platform-level roles, so their models are
    # process configuration. Worker models are per-agent and live in agents.yaml.
    planner_model: str = "claude-opus-5"
    synthesizer_model: str = "claude-opus-5"

    database_url: str = "postgresql+psycopg://orchestrator:orchestrator@localhost:5433/orchestrator"

    agents_config_path: Path = BACKEND_ROOT / "config" / "agents.yaml"
    mcp_config_path: Path = BACKEND_ROOT / "config" / "mcp_servers.yaml"

    # Applied per MCP tool call by app.mcp.manager.
    mcp_tool_timeout_seconds: float = 30.0
    mcp_tool_max_attempts: int = 3

    # How many agents and tools the planner sees. Deliberately small: the
    # planner picks from a menu, it does not read the whole registry.
    planner_agent_top_k: int = 5
    planner_tool_top_k: int = 12
    planner_memory_top_k: int = 3

    embedding_dimensions: int = 384

    cors_allow_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached so the whole process shares one Settings instance."""
    return Settings()
