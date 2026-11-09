"""Small text helpers shared by indexing and answer composition."""

from __future__ import annotations

import hashlib
import re

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "what",
        "when",
        "with",
    }
)


def stable_id(*parts: str) -> str:
    """Return a short deterministic identifier for a document or chunk."""

    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def tokenize(text: str, *, remove_stopwords: bool = False) -> list[str]:
    tokens = _WORD_RE.findall(text.lower())
    if remove_stopwords:
        return [token for token in tokens if token not in STOPWORDS]
    return tokens


def content_tokens(text: str) -> set[str]:
    """Return lightly stemmed content tokens for transparent lexical scoring.

    This is used by the local reranker and extractive answer composer. BM25
    itself continues to use exact tokens from :func:`tokenize`.
    """

    return {_stem(token) for token in tokenize(text, remove_stopwords=True)}


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    return [sentence.strip() for sentence in _SENTENCE_RE.split(compact) if sentence.strip()]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
