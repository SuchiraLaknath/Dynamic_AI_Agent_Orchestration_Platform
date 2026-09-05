"""Token accounting and USD pricing for LLM calls.

Turns the `usage_metadata` LangChain attaches to a response into a token and
cost record, so a run can report what it spent. Prices are per million tokens,
listed rather than fetched: a wrong number here is a reporting bug, not an
outage, and a network call on a hot path to price a request is not worth it.

It does not meter, budget, or cap spend. It reports after the fact.
"""

from pydantic import BaseModel, Field

# USD per 1M tokens, (input, output). Anthropic first-party API rates.
MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Used when a model id is not in the table, so an unpriced model shows up as an
# obvious zero rather than crashing a run that otherwise succeeded.
UNKNOWN_MODEL_PRICE = (0.0, 0.0)


class Usage(BaseModel):
    """Tokens and dollars for one LLM call, or a sum over several."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    unpriced_models: list[str] = Field(default_factory=list)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 6),
            calls=self.calls + other.calls,
            unpriced_models=sorted(set(self.unpriced_models) | set(other.unpriced_models)),
        )


def price_call(model: str, input_tokens: int, output_tokens: int) -> Usage:
    """Cost one LLM call. An unknown model is recorded, not guessed at."""
    input_rate, output_rate = MODEL_PRICES_PER_MTOK.get(model, UNKNOWN_MODEL_PRICE)
    cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(cost, 6),
        calls=1,
        unpriced_models=[] if model in MODEL_PRICES_PER_MTOK else [model],
    )


def price_response(model: str, usage_metadata: dict | None) -> Usage:
    """Cost one LangChain response from its `usage_metadata`.

    A response without usage metadata still counts as a call, so the call count
    and the token count can disagree visibly instead of the call vanishing.
    """
    if not usage_metadata:
        return Usage(calls=1)
    return price_call(
        model,
        int(usage_metadata.get("input_tokens", 0)),
        int(usage_metadata.get("output_tokens", 0)),
    )


def sum_usage(usages: list[Usage]) -> Usage:
    """Total a list of per-call usages into one figure."""
    total = Usage()
    for usage in usages:
        total = total + usage
    return total
