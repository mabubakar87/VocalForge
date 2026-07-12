"""Optional advanced-pipeline capability registry (Phase 4).

Probes are import-only. They never download models or load heavy weights.
Base transcription must work when every extra reports not_installed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class ExtraStatus(str, Enum):
    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"  # importable; not yet gated/wired as READY
    READY = "ready"  # reserved for post-approval integration


@dataclass(frozen=True)
class ExtraCapability:
    id: str
    label: str
    summary: str
    candidate_engines: tuple[str, ...]
    import_names: tuple[str, ...]
    proposal_doc: str

    def probe(self) -> ExtraStatus:
        """Return status for Setup. Enhancement is READY when the CLI path works."""
        if self.id == "enhancement":
            try:
                from vocalforge.enhancement import is_enhancement_available

                if is_enhancement_available():
                    return ExtraStatus.READY
            except Exception:  # noqa: BLE001
                pass
            return ExtraStatus.NOT_INSTALLED
        for name in self.import_names:
            try:
                importlib.import_module(name)
            except Exception:  # noqa: BLE001 — missing optional deps are expected
                continue
            return ExtraStatus.INSTALLED
        return ExtraStatus.NOT_INSTALLED


EXTRA_ORDER: tuple[str, ...] = (
    "alignment",
    "diarization",
    "enhancement",
    "separation",
)

EXTRAS: dict[str, ExtraCapability] = {
    "alignment": ExtraCapability(
        id="alignment",
        label="Forced word alignment",
        summary="Finer word-level timing than Faster-Whisper timestamps alone.",
        candidate_engines=("WhisperX", "Wav2Vec2"),
        import_names=("whisperx",),
        proposal_doc="docs/proposals/alignment.md",
    ),
    "diarization": ExtraCapability(
        id="diarization",
        label="Speaker diarization",
        summary="Label who spoke when. May require model license acceptance.",
        candidate_engines=("PyAnnote", "WhisperX"),
        import_names=("pyannote.audio", "whisperx"),
        proposal_doc="docs/proposals/diarization.md",
    ),
    "enhancement": ExtraCapability(
        id="enhancement",
        label="Speech enhancement",
        summary="Optional denoising before transcription (DeepFilterNet CLI + soxr).",
        candidate_engines=("DeepFilterNet",),
        # Prefer Rust CLI probe via vocalforge.enhancement; import names are fallback.
        import_names=("df", "deepfilternet"),
        proposal_doc="docs/proposals/enhancement.md",
    ),
    "separation": ExtraCapability(
        id="separation",
        label="Source separation",
        summary="Isolate vocals from music before transcription.",
        candidate_engines=("Demucs",),
        import_names=("demucs",),
        proposal_doc="docs/proposals/separation.md",
    ),
}


def list_extras() -> Sequence[ExtraCapability]:
    return tuple(EXTRAS[extra_id] for extra_id in EXTRA_ORDER)


def status_label(status: ExtraStatus) -> str:
    return {
        ExtraStatus.NOT_INSTALLED: "Not installed",
        ExtraStatus.INSTALLED: "Installed (not enabled)",
        ExtraStatus.READY: "Ready",
    }[status]


def probe_all() -> list[tuple[ExtraCapability, ExtraStatus]]:
    return [(extra, extra.probe()) for extra in list_extras()]
