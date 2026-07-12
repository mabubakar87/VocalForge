"""Clipboard copy and optional simulated paste."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClipboardSettings:
    auto_paste: bool = False


def copy_text(text: str) -> None:
    """Copy text to the system clipboard."""
    import pyperclip

    pyperclip.copy(text)
    logger.info("Text copied to clipboard.")


def paste_hotkey() -> None:
    """Simulate Ctrl+V. Prefer leaving this disabled by default."""
    import pyautogui

    pyautogui.hotkey("ctrl", "v")


def deliver_text(text: str, settings: ClipboardSettings | None = None) -> None:
    """Copy text and optionally paste. Clipboard failures are logged, not raised."""
    settings = settings or ClipboardSettings()
    try:
        copy_text(text)
        if settings.auto_paste:
            paste_hotkey()
    except Exception as exc:  # noqa: BLE001
        logger.error("Error delivering text to clipboard: %s", exc)
