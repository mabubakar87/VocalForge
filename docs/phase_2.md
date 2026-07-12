# Phase 2 — Hardware-Aware Faster-Whisper Profiles

Branch: `phase-2-profiles` (from `phase-1-stabilize`).

Phase 2 keeps **Faster-Whisper only**. Profiles select model, device, compute
type, and (for High Accuracy) language. No WhisperX, PyAnnote, Demucs, or
DeepFilterNet.

## Summary

VocalForge no longer uses a free-form model dropdown. Users pick a hardware
profile from **Setup / Profiles**, optionally choose CPU vs GPU, and (on High
Accuracy) select language. The main window shows the active session: profile,
model, device, and language.

## Profiles

| ID | Label | Model | Notes |
|----|-------|-------|-------|
| `lightweight` | Lightweight | `distil-small.en` | English-only, CPU-oriented (`int8`) |
| `balanced` | Balanced | `distil-medium.en` | English-only; CUDA when usable |
| `high_accuracy` | High Accuracy | `large-v3-turbo` | Multilingual; language selectable |

Recommendation (deterministic):

- No usable CUDA → Lightweight
- CUDA OK, VRAM &lt; 6 GB or unknown → Balanced
- CUDA OK, VRAM ≥ 6 GB → High Accuracy recommended

Local cache detection checks Hugging Face snapshot folders under `Models/`
(including aliases for `large-v3-turbo` redirects such as `mobiuslabsgmbh`).

## Configuration (`config.json`, schema v2)

Persisted fields used by Phase 2:

- `active_profile`
- `selected_model`
- `preferred_device` (`cpu` / `cuda`)
- `language` (Whisper code or `null` for auto-detect)
- `task` (`transcribe`; legacy `translate` maps to English)

## UI

### Main window

- Brand + **Active Session** cards: Profile, Model, Transcription via, Language
- Setup / Profiles button
- Anti-aliased record button (Pillow)
- Transcript pane with Arabic/Urdu display shaping for Tk
- Modern dark theme (`vocalforge/ui/theme.py`)

### Setup / Profiles

- System Specification cards (OS, CUDA, device, VRAM, disk, recommended profile)
- CPU / GPU segmented control (reloads active profile when device changes)
- Profile comparison table (Description, Model, Device, Compute, Language, Status, Action)
- High Accuracy language dropdown (Auto-detect, English, Urdu, Japanese, …)
- Changing language applies immediately and refreshes the home session card;
  activates High Accuracy if it was not already active
- Setup window opens with a quick “Loading…” shell, then builds content;
  later opens reuse the same window to avoid Tk widget-creation delay

## Runtime behavior

- Profile activation blocked while loading / recording / transcribing
- Disk-space check before download & activate
- CUDA→CPU fallback on load errors remains from Phase 1
- Transcription passes `language` / `task` into Faster-Whisper

## Dependencies added

- `arabic-reshaper`, `python-bidi` — RTL/Urdu display in the transcript pane
- `Pillow` — smooth record-button images

## Tests

Unit coverage includes profile recommendation, local availability, language
mode mapping, config round-trip (`preferred_device`), and transcription kwargs.

## Remaining acceptance (manual)

Tracked as partial in the task breakdown:

- **P2-007 / P2-008**: offline transcription with a cached model; clear error
  when a model is missing locally; manual CUDA profile switch checklist

## Key files

- `vocalforge/profiles.py` — profiles, recommendation, cache readiness, language helpers
- `vocalforge/ui/__init__.py` — main window + Setup dashboard
- `vocalforge/ui/theme.py` — theme, dropdown, segmented control, record images
- `vocalforge/config.py` — schema fields
- `vocalforge/transcription.py` — language/task on `transcribe()`
- `tests/test_profiles.py` — profile unit tests
