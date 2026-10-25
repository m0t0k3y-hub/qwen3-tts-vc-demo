"""Text helpers for long-text generation."""

from __future__ import annotations

import re

# Split after sentence-ending punctuation (a '.' only when followed by whitespace, so "1.5" stays whole) and at
# line breaks.
_SENTENCE_END = re.compile(r"(?<=[。！？!?])|(?<=\.)(?=\s)|\n")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def batch_lines(text: str) -> list[str]:
    """One text per non-empty line (F-05)."""
    return [line.strip() for line in text.splitlines() if line.strip()]
