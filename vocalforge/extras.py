"""Optional advanced-pipeline capability registry (Phase 4).

Probes are lightweight. They never download models or load heavy weights.
Base transcription must work when every extra reports not_installed.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class ExtraStatus(str, Enum):
    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"  # importable; not yet gated/wired as READY
    READY = "ready"  # usable (deps + token/cache as required)


@dataclass(frozen=True)
class ExtraCapability:
    id: str
    label: str
    summary: str
    candidate_engines: tuple[str, ...]
    import_names: tuple[str, ...]
    proposal_doc: str

    def probe(self, *, hf_token: str | None = None) -> ExtraStatus:
        """Return Setup status. Optional hf_token comes from config / env."""
        if self.id == "enhancement":
            try:
                from vocalforge.enhancement import is_enhancement_available

                if is_enhancement_available():
                    return ExtraStatus.READY
            except Exception:  # noqa: BLE001
                pass
            return ExtraStatus.NOT_INSTALLED

        if self.id == "diarization":
            try:
                from vocalforge.diarization import (
                    is_diarization_ready,
                    is_pyannote_importable,
                )

                if not is_pyannote_importable():
                    return ExtraStatus.NOT_INSTALLED
                if is_diarization_ready(hf_token):
                    return ExtraStatus.READY
                # Library present but offline weights not vendored yet.
                return ExtraStatus.NOT_INSTALLED
            except Exception:  # noqa: BLE001
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
        summary=(
            "Label who spoke when (offline pyannote community-1 under Models/). "
            "Vendor once: scripts/vendor_diarization_models.py."
        ),
        candidate_engines=("PyAnnote community-1",),
        import_names=("pyannote.audio",),
        proposal_doc="docs/proposals/diarization.md",
    ),
    "enhancement": ExtraCapability(
        id="enhancement",
        label="Speech enhancement",
        summary="Optional denoising before transcription (DeepFilterNet CLI + soxr).",
        candidate_engines=("DeepFilterNet",),
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


def probe_all(hf_token: str | None = None) -> list[tuple[ExtraCapability, ExtraStatus]]:
    token = hf_token
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return [(extra, extra.probe(hf_token=token)) for extra in list_extras()]
