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

    # Anchored to the repository root rather than left relative: the documented
    # way to run locally is `cd backend && uvicorn app.main:app`, and a relative
    # path resolves against that working directory, so the root .env was never
    # read. Under compose the values arrive as real environment variables, which
    # take precedence over this file either way.
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    anthropic_api_key: str = ""

    # Embeddings only, for planner retrieval and semantic memory. Anthropic has
    # no embeddings API, so this is a second provider. Leaving it blank falls
    # back to the offline lexical embedder, which is what the tests run on.
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"

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

    # text-embedding-3-small is natively 1536, but the v3 models accept a
    # `dimensions` parameter. 384 keeps stored vectors the same width as the
    # existing Vector(384) column, so the model swap needed no migration.
    embedding_dimensions: int = 384

    cors_allow_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached so the whole process shares one Settings instance."""
    return Settings()
