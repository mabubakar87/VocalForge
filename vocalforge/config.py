"""Versioned application configuration."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


@dataclass
class AppConfig:
    schema_version: int = SCHEMA_VERSION
    selected_model: str = ""
    active_profile: str = ""
    preferred_device: str = ""  # "", "cpu", or "cuda"
    language: str | None = None  # Whisper language code; None = auto-detect
    task: str = "transcribe"  # always "transcribe" (legacy "translate" is migrated away)
    auto_paste: bool = False
    input_device: str | None = None


def default_config() -> AppConfig:
    return AppConfig()


def load_config(path: Path) -> AppConfig:
    """Load config from JSON. Invalid/missing files yield safe defaults."""
    if not path.exists():
        return default_config()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config root must be an object")
        known = {f.name for f in fields(AppConfig)}
        filtered = {k: v for k, v in raw.items() if k in known}
        cfg = AppConfig(**filtered)
        if cfg.schema_version != SCHEMA_VERSION:
            logger.warning(
                "Config schema_version=%s; upgrading compatible fields to %s.",
                cfg.schema_version,
                SCHEMA_VERSION,
            )
            cfg.schema_version = SCHEMA_VERSION
        return cfg
    except Exception as exc:  # noqa: BLE001
        logger.warning("Invalid config at %s (%s); using defaults.", path, exc)
        return default_config()


def save_config(path: Path, config: AppConfig) -> None:
    """Atomically write configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)
