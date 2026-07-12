# Proposal: Speaker diarization

- **Capability id:** `diarization`
- **Author / date:** Phase 4 evaluation — 2026-07-12
- **Decision:** Recommended **Approve with conditions** (awaiting explicit sign-off before implementation spike)
- **Candidate engines:** **PyAnnote** (`pyannote.audio` + gated HF pipelines) primary; WhisperX diarization as optional wrapper (still PyAnnote under the hood)

## 1. User workflow

**Job:** Transcribe interviews, meetings, and multi-person recordings so the transcript shows **who spoke when**, not only the words.

**Success in UI:**

1. Optional home toggle: **Diarize speakers** (default **Off**), only enabled when the extra is Ready.
2. When On: after optional Enhance, run diarization, then Faster-Whisper ASR, then merge speaker labels onto timed segments.
3. Transcript display (and saved `.txt`) uses lines like:
   ```
   [SPEAKER_00 0:00–0:04] Hello, thanks for joining.
   [SPEAKER_01 0:04–0:09] Happy to be here.
   ```
4. Setup shows status: Not installed / Needs HF token / Ready (cached) / Error.
5. Missing extra or missing token → toggle Off; base single-speaker ASR unchanged.

**Why this track next:** Highest workflow impact after enhancement; forces the HF auth/compliance design early; fits the pipeline  
`Enhance → Diarize → ASR` and defines speaker×time data shapes before alignment work.

## 2. Benchmarks

**Method (spike, before READY):**

| Fixture | Purpose |
|---------|---------|
| Single-speaker clean dictation | Expect one dominant speaker; no crash |
| Two-speaker interview (or synthetic A/B turns) | Expect ≥2 speaker labels; turns roughly match turns |
| Overlapping speech (if available) | Document failure modes (pyannote is imperfect on overlap) |
| Same clip with Enhance On vs Off | Confirm enhance→diarize chain does not break |

**Metrics:**

- Wall time: diarize-only and enhance+diarize+transcribe
- Peak RSS / VRAM with ASR unloaded vs co-resident
- Speaker count vs ground truth (manual for spike)
- Qualitative: label stability (SPEAKER_00 vs flip-flop)

**Baseline:** Current path = optional Enhance + Faster-Whisper (no speakers).

## 3. Dependencies and licensing

| Piece | License / notes |
|-------|------------------|
| `pyannote.audio` ≥ 3.1 | Open-source package |
| `pyannote/speaker-diarization-3.1` | Model card **MIT**, but **gated** on Hugging Face (must accept user conditions / share contact info) |
| `pyannote/segmentation-3.0` | Also **gated** — must accept separately |
| PyTorch | Required (same packaging concern as rejected for enhancement v1 — keep out of **base** `requirements.txt`) |
| WhisperX | Optional; diarization still uses PyAnnote + HF token. Prefer **not** pulling WhisperX for v1 diarization to avoid duplicating ASR |

**Auth (non-negotiable for first download):**

1. User creates a Hugging Face account.
2. Accept conditions on **both** gated repos (diarization + segmentation).
3. Create a read token with access to public gated repos.
4. Supply token once via Setup (stored as `hf_token` in local `config.json` — gitignored — or `HF_TOKEN` env). Never ship a token in the repo.

**Offline after first success:** Models live in HF/pyannote cache; subsequent runs can use `HF_HUB_OFFLINE=1`. First-run **requires** network + accepted gates.  
**Do not** claim a universal no-login install.

**Alternative community pipeline:** `pyannote/speaker-diarization-community-1` (CC-BY-4.0) — still gated for download; evaluate in spike if 3.1 UX is too painful, but keep one primary pin.

## 4. Hardware measurements

| Machine | RAM peak | VRAM peak | Disk for models | Notes |
|---------|----------|-----------|-----------------|-------|
| Dev laptop (RTX 3050 4 GB, 32 GB RAM) | *TBD at spike* | *TBD — prefer diarize then ASR, not both on GPU* | PyAnnote pipeline + embeddings often **hundreds of MB–~1+ GB** | 4 GB VRAM + `large-v3-turbo` is contested; **serialize** load/unload |
| CPU-only | *TBD* | 0 | Same | Must work, expect slower RTF |

## 5. Packaging

- Base: unchanged (`requirements.txt` — Faster-Whisper / CTranslate2 only).
- Extras subgroup in `requirements-extras.txt` (commented until Approved):
  - `torch` / `torchaudio` (CPU default; CUDA optional doc)
  - `pyannote.audio>=3.1,<4`
- Probe: import `pyannote.audio` **and** valid token / cached pipeline → Ready; import without token → Needs setup.
- Failure modes: clear Setup errors for “gates not accepted”, “invalid token”, “offline without cache”.

**Engine choice for VocalForge:** **Standalone PyAnnote** after our existing Faster-Whisper path — not a full WhisperX swap. WhisperX remains a future option if we later want bundled align+diarize.

## 6. Local model acquisition

- First run: `Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=...)` downloads into HF cache.
- Disk-space check before download (Phase 2 pattern).
- Document exact cache dirs after spike (`~/.cache/huggingface`, `~/.cache/torch/pyannote`, etc.).
- Optional advanced: vendor local `config.yaml` + weights for air-gapped (document only; not required for v1).

## 7. Resource interaction with ASR

**Recommended intermediate shapes** (design now, implement in spike):

```text
SpeakerTurn { start: float, end: float, speaker: str }
TranscriptSegment { start, end, text, speaker?: str }
```

**Pipeline order:**

1. Optional `enhance_wav` (existing)
2. Optional `diarize_wav` → `list[SpeakerTurn]`
3. Faster-Whisper `transcribe` with segment timestamps (enable timestamps when diarize is On)
4. `assign_speakers(segments, turns)` by time overlap → labeled transcript

**Resource rules:**

- Do **not** keep PyAnnote + large Whisper on 4 GB VRAM together by default.
- Prefer: diarize (GPU or CPU) → release pipeline → ASR (existing CTranslate2 device).
- Aligns with plan §7 serialize load/unload; worker process if in-process RSS fights.

## 8. Recommendation

**Approve with conditions:**

1. Ship diarization as an **optional extra** (default Off); base install untouched.
2. **Auth UX first** in the spike: Setup fields for HF token + checklist link to accept both gated models; status machine: Not installed → Needs HF setup → Ready.
3. Use **PyAnnote standalone** + merge-onto-Faster-Whisper segments; defer WhisperX as the primary engine.
4. Measure on the 3050 laptop; force serialized GPU use if needed.
5. Mark Setup `ready` only after: token+cache works offline once, two-speaker fixture labeled sanely, and enhance→diarize→ASR chain works.

**Reject / defer if:** token/gates UX is unacceptable for the product promise, or 4 GB hardware cannot run diarize with acceptable latency even on CPU.

---

### Sign-off

- [ ] Product / owner approves conditions above
- [ ] Bounded P4-020 implementation spike opened (auth + diarize + merge; no full UI unlock until metrics)
- [ ] Hardware table filled from real measurements
- [ ] Decision updated to **Approved** or **Rejected**
