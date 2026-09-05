"""Text to vector, for planner retrieval and semantic memory.

Implements one deterministic, dependency-free embedding: a signed hashing
vectorizer over word and character n-grams. It is a lexical embedder, not a
neural one -- it captures shared vocabulary and morphology, not paraphrase.

The reason it is good enough here is scale: the planner retrieves from a
registry of a handful of agents and tools, where top-k is close to the whole
registry anyway, so retrieval is about ranking rather than recall. `embed_text`
is the single seam to swap for a hosted embedding model (Voyage, OpenAI) when
the registry grows past the point where lexical ranking holds up.

It does not train, cache across processes, or learn IDF weights.
"""

import hashlib
import math
import re
from collections import Counter

from app.settings import get_settings

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


def embed_text(text: str) -> list[float]:
    """Return a unit-length vector for one piece of text.

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

    # Sublinear term frequency. Without it a long role description that repeats
    # a common n-gram outranks a short one that is actually about the goal.
    for feature, count in counts.items():
        index, sign = _feature_slot(feature, dimensions)
        vector[index] += sign * weights[feature] * (1.0 + math.log(count))

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch. Present so the hosted-model swap has a batching seam."""
    return [embed_text(text) for text in texts]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Dot product of two vectors from `embed_text`, which are already unit-length."""
    if len(left) != len(right):
        raise ValueError(f"Cannot compare vectors of length {len(left)} and {len(right)}")
    return sum(a * b for a, b in zip(left, right))
