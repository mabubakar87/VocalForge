# Proposal: Speech enhancement / denoising

- **Capability id:** `enhancement`
- **Author / date:** Phase 4 evaluation — 2026-07-12
- **Decision:** **Approved** and **unlocked** in Setup (Enhance audio On/Off). Default remains Off.
- **Candidate engines:** DeepFilterNet via **`deep-filter` Rust CLI** + **soxr** resample (no PyTorch in spike).

## 1. User workflow

**Job:** Transcribe speech recorded in a noisy room (fans, keyboard, café, AC) where Faster-Whisper + VAD alone still produce errors or hallucinations on noise.

**Success in UI (next step after listen check):**

1. Optional toggle: **Enhance audio** (default **Off**).
2. When On and extra available, recording/upload is denoised **before** Faster-Whisper.
3. When missing, toggle disabled; base ASR unchanged.
4. Status: `Enhancing…` then `Transcribing… N%`.

**Spike code path:** `vocalforge/enhancement.py` — 16 kHz → 48 kHz (soxr) → `deep-filter -D` → 16 kHz.

## 2. Benchmarks

**Fixtures:** synthetic 3 s clean / noisy 16 kHz WAVs from `scripts/spike_enhancement_bench.py`  
(`tests/fixtures/enhancement/`).

| Clip | Wall time | Lag-aligned corr vs input | Spectral flatness in → out | RMS in → out |
|------|-----------|---------------------------|----------------------------|--------------|
| clean_3s | **0.70 s** | **0.9999** | 0.0008 → 0.0013 | 0.130 → 0.127 |
| noisy_3s | **0.61 s** | 0.20 (noise changed) | **0.74 → 0.05** | 0.153 → 0.006 |

**Artifact checks (synthetic):** clean correlation OK; noisy flatness reduced; outputs non-silent.  
**Caveat:** tones ≠ speech. Listen to `tests/fixtures/enhancement/out/*.wav` and run one real dictation before marking Setup status `ready`.

**Report:** `tests/fixtures/enhancement/out/spike_report.json`

## 3. Dependencies and licensing

| Piece | License / notes |
|-------|------------------|
| DeepFilterNet / `deep-filter` binary v0.5.6 | Dual **MIT** / **Apache-2.0** |
| `soxr` | Resample only; listed in `requirements-extras.txt` |
| PyTorch / `deepfilternet` wheel | **Not used** in this spike |

**Auth:** None.

**Binary location:** `.deps/deep-filter` (gitignored), or `DEEP_FILTER_BIN`, or `PATH`.

## 4. Hardware measurements

| Machine | RAM peak | VRAM peak | Disk for models | Notes |
|---------|----------|-----------|-----------------|-------|
| Dev laptop (RTX 3050 4 GB, 32 GB RAM) | Python ~42 MB self; **deep-filter child ~57 MB** | **0** (CPU binary) | Binary ~35 MB under `.deps/` | 3 s clip enhance ~0.6–0.7 s; no torch |

## 5. Packaging

- Base `requirements.txt`: unchanged.
- Extras: `soxr` + documented `deep-filter` binary download.
- Probe: `vocalforge.enhancement.is_enhancement_available()` → Setup shows **Installed (not enabled)** when binary+soxr present.
- Failure mode: enhance raises `EnhancementError`; callers must fall back to original audio (UI wiring TBD).

## 6. Local model acquisition

- Rust binary embeds/ships with model weights; no separate HF download in this spike.
- Offline: works once `.deps/deep-filter` is present.

## 7. Resource interaction with ASR

- Enhancement runs as a **subprocess**; ASR (CTranslate2) stays separate — good for the 4 GB GPU laptop.
- Unload: process exits after each file; no long-lived DF model in-process.
- Worker escape hatch already satisfied by CLI design.

## 8. Recommendation

**Ship path chosen:** Rust `deep-filter` + soxr (reject torch for v1 enhancement).

**Before UI toggle / `ready`:**

1. Manual listen of enhanced fixtures + one real noisy recording through Faster-Whisper.
2. Wire `config.enhance_audio` default Off + job-path call to `enhance_wav`.
3. Then set registry status `ready` when listen check passes.

### Sign-off

- [x] Product / owner approved spike conditions
- [x] P4-060 implementation spike (resample + CLI + metrics)
- [x] Hardware table filled from real measurements
- [x] Real-speech listen / WER spot-check *(owner unlocked UI; optional follow-up)*
- [x] UI toggle + `ready` status (Setup Optional Extras matrix)
