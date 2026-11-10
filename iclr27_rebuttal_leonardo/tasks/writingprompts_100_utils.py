"""Lightweight automatic diagnostics for long-form WritingPrompts outputs."""

from __future__ import annotations

import re


_WORD_RE = re.compile(r"\b[\w'’-]+\b", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _ngram_distinctness(tokens: list[str], order: int) -> float:
    if len(tokens) < order:
        return 1.0 if tokens else 0.0
    ngrams = [tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)]
    return len(set(ngrams)) / len(ngrams)


def _sentence_repetition(text: str) -> float:
    sentences = [
        re.sub(r"\s+", " ", sentence.strip().lower())
        for sentence in _SENTENCE_RE.split(text)
        if sentence.strip()
    ]
    if not sentences:
        return 0.0
    return 1.0 - len(set(sentences)) / len(sentences)


def process_results(doc, results):
    del doc
    prediction = results[0].strip()
    tokens = [token.lower() for token in _WORD_RE.findall(prediction)]
    distinct_4 = _ngram_distinctness(tokens, 4)
    distinct_8 = _ngram_distinctness(tokens, 8)
    return {
        "word_count": float(len(tokens)),
        "distinct_4": distinct_4,
        "distinct_8": distinct_8,
        "repetition_4": 1.0 - distinct_4,
        "repetition_8": 1.0 - distinct_8,
        "sentence_repetition": _sentence_repetition(prediction),
    }
