# Phase 4 — Optional Advanced Pipelines

Branches: `phase-4-advanced-pipelines` (foundation + enhancement), `phase-4-diarization` (P4-020 + translate profiles / Setup UX).

### Shipped

- Extras registry + Setup **Optional Extras** (4 equal columns) + scrollable Setup
- **Enhancement:** DeepFilterNet CLI + soxr; home **Enhance**
- **Diarization:** offline `pyannote/speaker-diarization-community-1` under `Models/diarization/`; vendor script; home **Diarize**; GPU when CUDA torch is available (Whisper unloaded briefly)
- **Translate profiles:** Multilingual Small (`small`) and Medium (`medium`) with `supports_translate`; home **Translate** (disabled on turbo / English-only)
- Profiles ordered by size; Setup Model row shows `name (compute_type)`

### Diarization install

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install 'pyannote.audio>=3.1.0,<5'
export HF_TOKEN=hf_...   # first vendor only, if needed
PYTHONPATH=. python scripts/vendor_diarization_models.py
PYTHONPATH=. python scripts/smoke_test_diarization.py
```

### Next

1. Alignment / separation proposals (P4-010 / P4-040)
2. Diarization hardware measurements on long multi-speaker audio

| ID | Capability | Status |
|----|------------|--------|
| P4-020 | Speaker diarization | Shipped (Ready when vendored) |
| P4-030 | Speech enhancement | Shipped (Ready) |
| P4-010 | Forced alignment | Proposal stub |
| P4-040 | Source separation | Proposal stub |

See `docs/proposals/diarization.md`, `docs/proposals/enhancement.md`, root `README.md`.
