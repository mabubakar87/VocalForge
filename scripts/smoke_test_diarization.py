#!/usr/bin/env python3
"""Smoke test for P4-020 diarization + speaker merge.

Usage:
  export HF_TOKEN=hf_xxx
  PYTHONPATH=. python scripts/smoke_test_diarization.py
  PYTHONPATH=. python scripts/smoke_test_diarization.py /absolute/or/relative/clip.wav

Pass a real multi-speaker WAV path. Without an argument, uses the short
enhancement fixture (single-speaker; still validates the pipeline loads).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vocalforge.config import load_config
from vocalforge.diarization import (
    DiarizationError,
    default_local_pipeline_dir,
    diarize_wav,
    is_diarization_ready,
    local_pipeline_ready,
    resolve_hf_token,
)
from vocalforge.merge import assign_speakers, format_speaker_transcript

DEFAULT_AUDIO = ROOT / "tests" / "fixtures" / "enhancement" / "noisy_3s_16k.wav"


def _load_token() -> str | None:
    env = resolve_hf_token(None)
    if env:
        return env
    cfg = load_config(ROOT / "config.json")
    return resolve_hf_token(cfg.hf_token)


def _dummy_whisper_segments() -> list[dict]:
    return [
        {"start": 0.0, "end": 4.0, "text": "Hello, thanks for joining."},
        {"start": 4.0, "end": 9.0, "text": "Happy to be here."},
        {"start": 9.0, "end": 12.0, "text": "Shall we begin?"},
    ]


def main() -> int:
    token = _load_token()
    explicit = len(sys.argv) > 1
    audio = Path(sys.argv[1]).expanduser() if explicit else DEFAULT_AUDIO

    print("Diarization ready:", is_diarization_ready(token, models_root=ROOT / "Models"))
    print("Local pipeline:", default_local_pipeline_dir(ROOT / "Models"))
    print("Vendored:", local_pipeline_ready(default_local_pipeline_dir(ROOT / "Models")))
    print("Audio:", audio.resolve() if audio.exists() else audio)

    if not local_pipeline_ready(default_local_pipeline_dir(ROOT / "Models")):
        print(
            "ERROR: Offline weights missing. Vendor once:\n"
            "  PYTHONPATH=. python scripts/vendor_diarization_models.py"
        )
        return 2

    if not audio.is_file():
        print(
            "ERROR: WAV not found.",
            f"You passed {audio!s} — use a real file path, e.g.:",
            f"  PYTHONPATH=. python scripts/smoke_test_diarization.py {DEFAULT_AUDIO}",
            sep="\n",
        )
        return 2

    try:
        turns = diarize_wav(
            audio,
            hf_token=token,
            device="cpu",
            models_root=ROOT / "Models",
        )
    except DiarizationError as exc:
        print("Diarization failed (baseline ASR would continue):", exc)
        return 1

    print("Turns:")
    print(json.dumps(turns, indent=2))
    # Scale dummy text windows to the clip when turns exist.
    labeled = assign_speakers(_dummy_whisper_segments(), turns)
    print("--- transcript (dummy Whisper text + real speaker turns) ---")
    print(format_speaker_transcript(labeled))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
