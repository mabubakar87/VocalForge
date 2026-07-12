# Proposal: Speaker diarization

- **Capability id:** `diarization`
- **Author / date:** Phase 4 evaluation — 2026-07-12
- **Decision:** **Approved** (P4-020 implementation on `phase-4-diarization`)
- **Candidate engines:** **PyAnnote community-1** (`pyannote/speaker-diarization-community-1`) primary; WhisperX deferred

## 1. User workflow

**Job:** Transcribe interviews, meetings, and multi-person recordings so the transcript shows **who spoke when**, not only the words.

**Success in UI:**

1. Optional home toggle: **Diarize** (default **Off**), only enabled when the extra is Ready.
2. When On: after optional Enhance, run diarization (CUDA if available, else CPU), then Faster-Whisper ASR, then merge speaker labels onto timed segments.
3. Transcript display (and saved `.txt`) uses lines like:
   ```
   [SPEAKER_00 0:00–0:04] Hello, thanks for joining.
   [SPEAKER_01 0:04–0:09] Happy to be here.
   ```
4. Setup shows Ready when `pyannote.audio` imports **and** weights are vendored under `Models/diarization/`.
5. Missing extra, missing vendor, or diarization failure → unlabeled ASR continues (no crash).

## 2. Implementation (P4-020)

| Piece | Location |
|-------|----------|
| Pipeline wrapper | `vocalforge/diarization.py` — offline local `config.yaml` load |
| Vendor (one-time) | `scripts/vendor_diarization_models.py` → `Models/diarization/pyannote_community_1/` |
| Merge | `vocalforge/merge.py` — `assign_speakers()` / word-level group |
| Job path | `vocalforge/jobs.py` — enhance → diarize (try/except) → ASR → merge |
| Config | `config.hf_token` (vendor download only), `config.diarize_speakers` |
| Extras probe | Ready iff pyannote importable **and** local weights vendored |
| Smoke | `scripts/smoke_test_diarization.py` |

**Pipeline id:** `pyannote/speaker-diarization-community-1` (vendored offline)  
**VRAM rule:** prefer CUDA when `torch.cuda.is_available()`; unload Whisper briefly on small GPUs, then ASR (CTranslate2).  
**Offline rule:** runtime sets `HF_HUB_OFFLINE=1` only while loading the local pipeline; no Hub HEAD checks per job.

## 3. Dependencies and licensing

| Piece | License / notes |
|-------|------------------|
| `pyannote.audio` ≥ 3.1 | Open-source package (`requirements-extras.txt` Diarization Extra) |
| `pyannote/speaker-diarization-community-1` | Gated on Hugging Face — accept terms + read token |
| PyTorch | Install per machine (CPU wheel recommended for 4 GB VRAM hosts); **not** in base `requirements.txt` |

**Auth:**

1. Hugging Face account + accept gated model conditions (one-time vendor only).
2. Run `PYTHONPATH=. python scripts/vendor_diarization_models.py` (uses `HF_TOKEN` / Setup token if hub cache empty).
3. Runtime never needs the network for diarization after weights are under `Models/`.

## 4. Hardware notes

| Machine | Notes |
|---------|-------|
| Dev laptop (RTX 3050 4 GB) | Prefer CPU diarize so Whisper keeps CUDA; serialize load/unload |
| CPU-only | Supported; slower RTF |

Fill RAM/VRAM peaks after a real two-speaker smoke run.

## 5. Packaging

- Base: unchanged.
- Extras: see `# Diarization Extra` in `requirements-extras.txt`.
- Probe: import + token/cache → Ready; otherwise Not installed.

## 6. Recommendation / sign-off

**Approved** for optional extra (default Off). Baseline single-speaker ASR must keep working when diarization is Off, missing, or fails.

### Sign-off

- [x] Bounded P4-020 implementation (auth + diarize + merge + UI toggle)
- [ ] Hardware table filled from real measurements
- [x] Decision updated to **Approved**
