"""Transcript text formatting helpers."""

from __future__ import annotations

import re


def format_text(text: str) -> str:
    """Normalize whitespace, punctuation spacing, and sentence capitalization."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"(?<=[.!?])\s*(\w)", lambda m: m.group(1).upper(), text)
    text = text[0].upper() + text[1:]
    text = re.sub(r"([.,;:!?])([^\s])", r"\1 \2", text)
    return text
