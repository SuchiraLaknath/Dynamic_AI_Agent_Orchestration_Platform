"""Text to vector, for planner retrieval and semantic memory.

Uses OpenAI's `text-embedding-3-small` when `OPENAI_API_KEY` is configured, and
falls back to a local lexical vectorizer when it is not.

Two things are worth knowing about that choice.

**Why a hosted model.** The fallback below is a signed hashing vectorizer, which
compares shared vocabulary rather than meaning. Measured against these
capability descriptions it collides 23-29% of its features into 384 buckets and
puts the wrong capability first on four goals out of five, and no amount of
extra dimensions fixes it. A real embedding makes cosine similarity measure what
it is supposed to measure: "how quickly is the team shipping work" lands near
"velocity" despite sharing no words with it. Anthropic has no embeddings API, so
this is necessarily a second provider.

**Why 384 dimensions.** The model is natively 1536, but the v3 models accept a
`dimensions` parameter. Asking for 384 keeps the stored vectors the same width
as the existing `Vector(384)` column, so swapping the model needed no schema
change and no migration.

The fallback is not decoration: the test suite constructs both registries, which
embed at startup, so without it every test run would make live API calls. It
keeps tests and offline development working, at reduced retrieval quality.

This module does not train, batch across processes, or persist its cache.
"""

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache

from app.settings import get_settings

# ---------------------------------------------------------------------------
# Lexical fallback: a signed hashing vectorizer, used when no API key is set.
# ---------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Character n-grams let "sprints" match "sprint" and "forecasting" match
# "forecast" without a stemmer.
_CHAR_NGRAM_SIZE = 4
_WORD_WEIGHT = 2.0
_CHAR_WEIGHT = 1.0


def _feature_slot(feature: str, dimensions: int) -> tuple[int, float]:
    """Map a feature to a bucket and a sign.

    Hashed with blake2b rather than the builtin `hash`, whose string seed is
    randomized per process -- vectors written to Postgres in one process have to
    match vectors computed in the next one.
    """
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=9).digest()
    index = int.from_bytes(digest[:8], "big") % dimensions
    sign = 1.0 if digest[8] & 1 else -1.0
    return index, sign


def _features(text: str) -> list[tuple[str, float]]:
    words = _TOKEN_PATTERN.findall(text.lower())
    features: list[tuple[str, float]] = [(f"w:{word}", _WORD_WEIGHT) for word in words]

    joined = " ".join(words)
    for start in range(max(0, len(joined) - _CHAR_NGRAM_SIZE + 1)):
        features.append((f"c:{joined[start : start + _CHAR_NGRAM_SIZE]}", _CHAR_WEIGHT))
    return features


def embed_text_lexically(text: str) -> list[float]:
    """Vector for one text, computed locally with no network call.

    Empty or punctuation-only text yields a zero vector, whose cosine similarity
    against anything is 0 -- callers get "no signal", not a division by zero.
    """
    dimensions = get_settings().embedding_dimensions
    vector = [0.0] * dimensions

    counts: Counter[str] = Counter()
    weights: dict[str, float] = {}
    for feature, weight in _features(text):
        counts[feature] += 1
        weights[feature] = weight

    # Sublinear term frequency. Without it a long description that repeats a
    # common n-gram outranks a short one that is actually about the goal.
    for feature, count in counts.items():
        index, sign = _feature_slot(feature, dimensions)
        vector[index] += sign * weights[feature] * (1.0 + math.log(count))

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


# ---------------------------------------------------------------------------
# Hosted embeddings
# ---------------------------------------------------------------------------


class EmbeddingError(RuntimeError):
    """Raised when the embedding provider is configured but cannot be reached."""


@lru_cache(maxsize=1)
def _client():
    """Build the OpenAI client once, or return None when no key is configured."""
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key)


def uses_hosted_embeddings() -> bool:
    """True when a real embedding model is in use rather than the fallback."""
    return _client() is not None


def _embed_remotely(texts: list[str]) -> list[list[float]]:
    """Embed a batch in one API call.

    Batched because the registries embed every capability and tool description at
    startup; one request for all of them beats one request each.
    """
    settings = get_settings()
    try:
        response = _client().embeddings.create(
            model=settings.embedding_model,
            input=texts,
            dimensions=settings.embedding_dimensions,
        )
    except Exception as error:  # noqa: BLE001 - re-raised as a named error
        raise EmbeddingError(
            f"Could not embed with {settings.embedding_model!r}: {error}. Unset "
            "OPENAI_API_KEY to fall back to the offline lexical embedder."
        ) from error
    # The API preserves input order, but sorting by index makes that explicit
    # rather than assumed.
    ordered = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
    # Re-normalize. OpenAI normalizes the full-width vector, but asking for
    # fewer `dimensions` truncates it, and a truncated unit vector is no longer
    # unit length. `cosine_similarity` below is a bare dot product, so without
    # this it would silently rank by vector magnitude as well as direction.
    return [_normalize(vector) for vector in ordered]


def _normalize(vector: list[float]) -> list[float]:
    """Scale a vector to unit length, leaving an all-zero vector alone."""
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

# Cache keyed on text. A single run embeds its goal four times -- once to store
# it, once to find similar runs, once to rank capabilities and once to rank
# tools -- and without this each of those is a separate network round trip.
_cache: dict[str, list[float]] = {}


def embed_text(text: str) -> list[float]:
    """Return a vector for one piece of text, from cache when already seen."""
    if text in _cache:
        return _cache[text]
    vector = embed_texts([text])[0]
    return vector


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed several texts, making at most one API call for the uncached ones."""
    missing = [text for text in dict.fromkeys(texts) if text not in _cache]
    if missing:
        if uses_hosted_embeddings():
            computed = _embed_remotely(missing)
        else:
            computed = [embed_text_lexically(text) for text in missing]
        _cache.update(zip(missing, computed))
    return [_cache[text] for text in texts]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Similarity of two vectors from `embed_text`, which are already unit-length."""
    if len(left) != len(right):
        raise ValueError(f"Cannot compare vectors of length {len(left)} and {len(right)}")
    return sum(a * b for a, b in zip(left, right))
