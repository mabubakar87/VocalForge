# Manual smoke test — VocalForge Phase 1 baseline

Use this checklist after installing dependencies (`docs/runtime_requirements.md`).

## Machine notes (fill in when running)

| Item | Value |
|------|--------|
| OS | |
| Python | |
| Launch command | `./run.sh` or `python main.py` |
| Device selected at startup | CPU / CUDA |
| Model used | `distil-small.en` recommended |

## Checklist

1. [ ] Application launches; window title is **VocalForge**.
2. [ ] Status shows no model ready until a model is selected.
3. [ ] Record / Upload controls are disabled until a model loads.
4. [ ] Select `distil-small.en` from the dropdown.
5. [ ] Status shows download/load progress, then success (includes device).
6. [ ] Click the record button; status shows **Recording...**; button turns red.
7. [ ] Speak briefly, click again; status shows processing then **Transcription Completed**.
8. [ ] Transcript appears in the text area.
9. [ ] A new file appears under `transcripts/`.
10. [ ] Audio is written under `audio_files/recorded_audio.wav`.
11. [ ] Clipboard contains the transcript (auto-paste is **off** by default).
12. [ ] Upload `tests/fixtures/silence_1s.wav` or a short spoken WAV; transcription finishes without crash.
13. [ ] Close the window while idle; process exits cleanly.
14. [ ] Optional: close while recording or loading; process exits without hang.

## Known limitations (current baseline)

- Global `Ctrl+Q` hotkey requires elevated privileges on many Linux systems;
  use the on-screen button instead.
- First model download needs network access; later runs can use the local
  `Models/` cache.
- CUDA requires matching runtime libraries; otherwise the app falls back to CPU.
- `run.sh` may set local PortAudio / CUDA library paths for development only.

## Fixture

- `tests/fixtures/silence_1s.wav` — 1 second near-silence, 16 kHz mono PCM.
- Expected transcript for silence: empty or near-empty text (model-dependent).
- For spoken integration tests, record a short local sample or reuse a private
  WAV under `audio_files/` (not committed).

## Last run log

| Date | Result | Notes |
|------|--------|-------|
| 2026-07-12 | Partial (pre-refactor) | App recorded and transcribed on CUDA after PortAudio + cuBLAS path fixes; hotkey failed without root; Tk thread-safety and cuBLAS issues were fixed before Phase 1 modularization. |
