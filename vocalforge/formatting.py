"""Transcript text formatting helpers."""

from __future__ import annotations

import re

# Arabic / Persian / Urdu blocks (plus presentation forms).
_ARABIC_SCRIPT_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

# Clean Helvetica-like UI sans available to this Tk build (chat-style readability).
_UI_FONT_PREFERRED = (
    "texgyreheros",
    "latin modern sans",
    "texgyreadventor",
    "DejaVu Sans",
    "Liberation Sans",
    "Arial",
)

_ARABIC_FONT_PREFERRED = (
    "clearlyu arabic",
    "ClearlyU Arabic",
    "Noto Naskh Arabic",
    "Noto Sans Arabic",
    "DejaVu Sans",
    "FreeSerif",
)


def format_text(text: str) -> str:
    """Normalize whitespace, punctuation spacing, and sentence capitalization."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    # Latin-oriented cleanup; leave Arabic-script transcripts mostly as-is.
    if contains_arabic_script(text):
        return text
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"(?<=[.!?])\s*(\w)", lambda m: m.group(1).upper(), text)
    text = text[0].upper() + text[1:]
    text = re.sub(r"([.,;:!?])([^\s])", r"\1 \2", text)
    return text


def contains_arabic_script(text: str) -> bool:
    return bool(_ARABIC_SCRIPT_RE.search(text))


def prepare_ui_text(text: str) -> str:
    """Shape Arabic-script text for LTR Tk widgets (display only).

    Clipboard and saved transcripts should keep the original logical Unicode.
    """
    if not text or not contains_arabic_script(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        return text
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:  # noqa: BLE001
        return text


def _pick_font_family(root, preferred: tuple[str, ...], fallback: str = "Arial") -> str:
    try:
        families = {str(name).lower(): str(name) for name in root.tk.call("font", "families")}
    except Exception:  # noqa: BLE001
        return fallback
    for name in preferred:
        found = families.get(name.lower())
        if found:
            return found
    return fallback


def resolve_ui_font_family(root) -> str:
    """Readable sans-serif for labels/buttons (closest available to a chat UI font)."""
    return _pick_font_family(root, _UI_FONT_PREFERRED)


def resolve_transcript_font_family(root, text: str = "") -> str:
    """Font for the transcript pane; Arabic-capable when needed."""
    if text and contains_arabic_script(text):
        return _pick_font_family(root, _ARABIC_FONT_PREFERRED, fallback=resolve_ui_font_family(root))
    return resolve_ui_font_family(root)
