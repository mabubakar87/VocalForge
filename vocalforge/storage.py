"""Application path and file-store helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    """Writable locations used by VocalForge."""

    root: Path
    models: Path
    audio: Path
    transcripts: Path
    log_file: Path

    def ensure(self) -> "AppPaths":
        """Create required directories if they do not exist."""
        for path in (self.models, self.audio, self.transcripts):
            path.mkdir(parents=True, exist_ok=True)
        return self


def default_paths(root: Path | None = None) -> AppPaths:
    """Build paths relative to the application package root (repo root)."""
    if root is None:
        root = Path(__file__).resolve().parent.parent
    root = Path(root)
    return AppPaths(
        root=root,
        models=root / "Models",
        audio=root / "audio_files",
        transcripts=root / "transcripts",
        log_file=root / "vocalforge.log",
    ).ensure()


def recorded_audio_path(paths: AppPaths) -> Path:
    return paths.audio / "recorded_audio.wav"


def next_transcript_path(paths: AppPaths, when: datetime | None = None) -> Path:
    """Return a transcript path. If the second collides, add a numeric suffix."""
    when = when or datetime.now()
    stamp = when.strftime("%Y%m%d_%H%M%S")
    candidate = paths.transcripts / f"transcript_{stamp}.txt"
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = paths.transcripts / f"transcript_{stamp}_{index}.txt"
        if not candidate.exists():
            return candidate
        index += 1


def save_transcript(paths: AppPaths, text: str, when: datetime | None = None) -> Path:
    """Write transcript text and return the created path."""
    path = next_transcript_path(paths, when=when)
    path.write_text(text, encoding="utf-8")
    return path
