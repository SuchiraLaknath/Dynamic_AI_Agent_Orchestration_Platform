"""The embedder has two backends, and both have to behave.

The hosted model is what makes retrieval match meaning, but the offline fallback
is what lets this suite run at all: the registry fixtures embed every capability
and tool description at construction, so without a working fallback every test
run would make live API calls.

The hosted path is exercised by one test that skips when no key is configured.
"""

import pytest

from app import embeddings
from app.embeddings import (
    cosine_similarity,
    embed_text,
    embed_text_lexically,
    embed_texts,
)
from app.settings import get_settings

# Captured before conftest's autouse fixture forces every test offline, so the
# hosted tests below can opt back in and the skip condition can see the truth.
_REAL_USES_HOSTED = embeddings.uses_hosted_embeddings
requires_openai_key = pytest.mark.skipif(
    not _REAL_USES_HOSTED(), reason="no OPENAI_API_KEY configured"
)


@pytest.fixture
def hosted(monkeypatch):
    """Undo the suite-wide offline default for one test."""
    monkeypatch.setattr(embeddings, "uses_hosted_embeddings", _REAL_USES_HOSTED)
    embeddings._cache.clear()


# --- the offline fallback ---------------------------------------------------


def test_the_fallback_returns_a_unit_vector_of_the_configured_width():
    vector = embed_text_lexically("predict sprint velocity from past sprints")
    dimensions = get_settings().embedding_dimensions

    assert len(vector) == dimensions, "stored vectors must match the Vector(384) column"
    assert abs(sum(value * value for value in vector) ** 0.5 - 1.0) < 1e-9


def test_the_fallback_gives_no_signal_rather_than_dividing_by_zero():
    """Empty text has no features, so it must yield zeros, not an error."""
    vector = embed_text_lexically("   ...   ")

    assert set(vector) == {0.0}
    assert cosine_similarity(vector, embed_text_lexically("sprint velocity")) == 0.0


def test_the_fallback_is_deterministic_across_processes():
    """Vectors written to Postgres in one process are compared in the next one.

    This is why the fallback hashes with blake2b rather than the builtin `hash`,
    whose string seed is randomized per process.
    """
    assert embed_text_lexically("sprint velocity") == embed_text_lexically("sprint velocity")


# --- shared behaviour, whichever backend is active --------------------------


def test_an_unrelated_goal_scores_below_a_relevant_one():
    goal = embed_text("predict our next sprint velocity from past Jira sprints")
    relevant = embed_text("summarizes historical Jira sprint records and story points")
    unrelated = embed_text("converts supplier invoices between foreign currencies")

    assert cosine_similarity(goal, relevant) > cosine_similarity(goal, unrelated)


def test_identical_text_is_only_embedded_once(monkeypatch):
    """A single run embeds its goal four times; the cache must collapse those.

    Without it each of those is a separate network round trip once a hosted
    model is configured.
    """
    calls: list[list[str]] = []
    real = embeddings.embed_text_lexically

    def counting(text: str) -> list[float]:
        calls.append([text])
        return real(text)

    monkeypatch.setattr(embeddings, "embed_text_lexically", counting)
    monkeypatch.setattr(embeddings, "uses_hosted_embeddings", lambda: False)

    for _ in range(4):
        embed_text("predict our next sprint velocity")

    assert len(calls) == 1, "the same text must not be embedded twice"


def test_a_batch_embeds_each_distinct_text_once_and_keeps_order():
    texts = ["sprint velocity", "team capacity", "sprint velocity"]

    vectors = embed_texts(texts)

    assert len(vectors) == 3
    assert vectors[0] == vectors[2], "a repeated text returns the same vector"
    assert vectors[0] != vectors[1]


def test_comparing_different_widths_is_an_explicit_error():
    with pytest.raises(ValueError, match="length"):
        cosine_similarity([0.0] * 4, [0.0] * 8)


# --- the hosted path --------------------------------------------------------


@requires_openai_key
def test_hosted_embeddings_are_unit_length_at_the_configured_width(hosted):
    """Truncating via `dimensions` breaks OpenAI's normalization; we restore it.

    `cosine_similarity` is a bare dot product, so a non-unit vector would rank
    by magnitude as well as direction.
    """
    vector = embed_text("predict our next sprint velocity from past Jira sprints")

    assert len(vector) == get_settings().embedding_dimensions
    assert abs(sum(value * value for value in vector) ** 0.5 - 1.0) < 1e-6


@requires_openai_key
def test_hosted_embeddings_match_meaning_where_the_fallback_cannot(hosted):
    """The reason for the swap: no shared words, but the same intent.

    "throughput" and "iteration" appear nowhere in the velocity text.
    """
    query = "Forecast our throughput for the coming iteration."
    velocity = "Predict the velocity for our next sprint based on previous Jira sprints."
    unrelated = "Draft release notes from the issues completed in SPR-38."

    asked = embed_text(query)
    assert cosine_similarity(asked, embed_text(velocity)) > cosine_similarity(
        asked, embed_text(unrelated)
    )
