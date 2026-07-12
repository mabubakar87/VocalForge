"""Upload / media format helpers (decoded by Faster-Whisper / PyAV)."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_UPLOAD_EXTENSIONS = frozenset(
    {
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg",
        ".opus",
        ".webm",
        ".mp4",
        ".mkv",
        ".aac",
        ".wma",
    }
)

UPLOAD_FILEDIALOG_TYPES = [
    (
        "Audio / video",
        " ".join(f"*{ext}" for ext in sorted(SUPPORTED_UPLOAD_EXTENSIONS)),
    ),
    ("WAV", "*.wav"),
    ("MP3", "*.mp3"),
    ("All files", "*.*"),
]


def is_supported_upload(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_UPLOAD_EXTENSIONS
