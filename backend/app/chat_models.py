"""Constructs chat model clients from a model id.

One function, so the planner, workers and synthesizer all build their clients
the same way and a test can substitute a stub at a single seam. Model ids come
from configuration -- agents.yaml for workers, environment for the planner and
synthesizer -- never from a literal in orchestration code.

It does not choose which model to use.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from app.settings import get_settings


class MissingApiKeyError(RuntimeError):
    """Raised when a run needs the LLM but no API key is configured."""


def build_chat_model(model_id: str, max_tokens: int = 8000) -> BaseChatModel:
    """Return a chat client for `model_id`.

    Fails here, with an actionable message, rather than letting an empty key
    surface as a 401 from inside a worker three nodes deep.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise MissingApiKeyError(
            "ANTHROPIC_API_KEY is not set, so no LLM call can be made. Set it in .env "
            "(see .env.example). Tests stub the model and do not need it."
        )
    return ChatAnthropic(
        model=model_id,
        api_key=settings.anthropic_api_key,
        max_tokens=max_tokens,
        timeout=120,
        stop=None,
    )
