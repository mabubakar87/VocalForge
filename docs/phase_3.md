# Phase 3 — Core Transcription Experience

Branch: `phase-3-transcription-ux`.

Builds on Phase 2 profiles. Focus: VAD, decode options, mic selection, formats,
progress/cancel, transcript history — plus home-screen language control for
High Accuracy.

### Done

- **P3-001** Home language dropdown for High Accuracy
- **P3-002** VAD filter toggle (default on, Faster-Whisper Silero)
- **P3-003** Beam size (`1` / `5` / `10`, default `5`) and word timestamps
- **P3-004** Microphone dropdown on the home screen (`config.input_device`)
- **P3-005** Upload formats beyond WAV (PyAV via Faster-Whisper)
- **P3-006** Transcription progress percent + Cancel (recording or transcribe)
- **P3-007** Transcript history dropdown, Copy, and Save As

### Defaults (persisted in `config.json`)

| Setting | Default |
|---------|---------|
| `vad_filter` | `true` |
| `beam_size` | `5` |
| `word_timestamps` | `false` |
| `input_device` | `null` (system default) |
| `language` / `task` | auto / `transcribe` (High Accuracy) |

See `plan/01-Upgrade/task_breakdown_phase_3.md` for the checklist.
