# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

VocalForge is a desktop speech-to-text app: Tkinter UI + Faster-Whisper (CTranslate2) ASR, running locally (CPU or CUDA). Optional Phase 4 extras add DeepFilterNet audio enhancement and offline PyAnnote speaker diarization without touching the base install.

## Before starting a new task: use the swarm (but ask first)

The multi-agent swarm (`docs/agent_workflow.md`, invoked via the `/swarm-feature` skill) is the default way to build **any new task** in this repo — a feature, a change, a fix, anything that involves writing code. It plans the work into `plans/<NN-name>/`, then runs it in ~10% batches (assign → isolated-worktree execution → adversarial review → integration), pausing for a human Continue/Pivot/Rollback decision between batches.

**Always ask before initiating a swarm.** When a request looks like a task, propose running it through the swarm and get the user's go-ahead before invoking `/swarm-feature` — don't spin one up unprompted. The swarm launches many agents and creates branches/worktrees, so kicking one off is the user's call, not an inferred default.

These never need the swarm — just do them directly, no need to ask:

- Answering questions
- Reading or explaining code
- Trivial one-line edits

## Commands

```bash
# Install (Python 3.10+, verified on 3.11)
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Run
python main.py            # any platform
./run.sh                  # Linux dev convenience: sets LD_LIBRARY_PATH for local PortAudio/CUDA libs

# Tests
pip install -r requirements-dev.txt
pytest                                   # default: excludes gpu, audio_device, network, slow markers
pytest -m integration                    # include integration-marked tests
pytest -m "gpu"                          # run only GPU tests (needs real CUDA + CTranslate2)
pytest tests/test_jobs.py::test_name     # single test
```

Test markers (`pytest.ini`): `integration`, `gpu`, `audio_device`, `network`, `slow`. Anything hardware- or network-dependent must be marked so the default `pytest` run stays hermetic. `tests/manual_smoke_test.md` is a manual (non-automated) checklist for full end-to-end verification (record → transcribe → clipboard → history) that can't be scripted.

Hardware-adjacent modules test via dependency injection rather than mocking the underlying library: `capabilities.py`'s `evaluate_capabilities(run=..., find_library=..., cuda_device_count=...)`, `profiles.py`'s `probe_vram_gb(run=...)`, and `audio.py`'s `list_input_devices()` all accept an injectable callable so tests can simulate hardware states without real GPUs/mics. Follow this pattern for new hardware probes instead of monkeypatching imports.

Optional extras (not part of default install; see `requirements-extras.txt` and `docs/proposals/`):
```bash
# Enhancement: Rust deep-filter binary at .deps/deep-filter, + soxr
# Diarization:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install 'pyannote.audio>=3.1.0,<5'
PYTHONPATH=. python scripts/vendor_diarization_models.py   # one-time vendor into Models/diarization/
PYTHONPATH=. python scripts/smoke_test_diarization.py path/to.wav

# Enhancement bench (measures wall time, peak RSS, signal-correlation vs. clean fixture):
PYTHONPATH=. python scripts/spike_enhancement_bench.py
```

## Architecture

### Threading model — the load-bearing pattern

The UI is single-threaded Tkinter; everything hardware/IO-bound (model load, recording, transcription, enhancement, diarization) runs on background daemon threads owned by `JobController` (`vocalforge/jobs.py`). Communication back to the UI thread is one-directional and queue-based:

1. `JobController` never touches Tk widgets. It emits `JobEvent` objects via the `emit` callback passed into its constructor.
2. `vocalforge/ui/__init__.py` wires `emit` to push events onto a `queue.Queue` (`self.ui_queue`), and polls it on the Tk main loop via `root.after(50, self._poll_ui_queue)`.
3. All UI mutation happens only inside that poll handler, on the main thread.

When adding a new background operation, follow this shape: do the work on a `threading.Thread(daemon=True)`, emit `JobEvent`s for state/progress/results, and never call Tk methods from the worker thread.

### App state machine

`vocalforge/state.py` defines `AppState` (STARTING, NO_MODEL, LOADING_MODEL, READY, RECORDING, TRANSCRIBING, ERROR, SHUTTING_DOWN) with an explicit `ALLOWED_TRANSITIONS` table. `JobController` gates every operation (`can_record`, `can_upload`, `can_change_profile`, `can_cancel`) through this machine before starting work, and routes back to READY/ERROR/NO_MODEL when a job finishes, fails, or is cancelled. Invalid transitions are rejected (logged), not raised, except via `StateMachine.transition` (used rarely; `try_transition` is the normal path). Any new job type should add its own guard here rather than checking ad hoc flags.

Each job also tracks a `job_id` (`uuid4().hex`) and compares against `self._active_job_id`; late callbacks/results from a superseded or cancelled job are dropped via `_is_current(job_id)` checks. Cancellation is cooperative via `threading.Event` (`_stop_recording`, `_cancel_transcription`) — long-running loops (recording wait loop, per-segment transcription loop) poll these events.

### Hardware capability detection → profiles → model load

- `vocalforge/capabilities.py` (`evaluate_capabilities`) probes `nvidia-smi`, CTranslate2's reported CUDA device count, and whether the actual CUDA runtime libs (`cublas`, `cudart`) are loadable/discoverable (via `ctypes.util.find_library`, `LD_LIBRARY_PATH`/`PATH`/`CUDA_PATH` directory scans, and a direct `ctypes.CDLL` load attempt). CUDA is only selected when **all three** checks pass; otherwise it falls back to CPU with a logged `fallback_reason`. This is deliberately conservative — a visible GPU without loadable runtime libs must not be selected. All probes are injectable (`run`, `find_library`, `cuda_device_count` params) specifically so tests can simulate pass/fail combinations without real hardware (see `tests/test_capabilities.py`).
- `vocalforge/profiles.py` defines `PROFILES` (Lightweight/Balanced/Multilingual Small/Multilingual Medium/High Accuracy — see README table), each a `ProfileSpec` with model name, preferred device, per-device compute type, disk size, translate support, and known HF cache directory name(s). Setup screen logic (local availability, VRAM/disk checks, translate eligibility) lives here.
- `vocalforge/transcription.py` (`TranscriptionService`) owns the actual `faster_whisper.WhisperModel` instance. `load_model_with_cpu_fallback` retries once on CPU if the CUDA load fails with a CUDA/cublas/cudnn-related error string — this is the runtime safety net beneath the static capability probe. `release()` drops the model (used to free VRAM before diarization runs on GPU).

### Optional extras are probed, never assumed

`vocalforge/extras.py` is a lightweight registry (`EXTRAS`, ids: alignment, diarization, enhancement, separation) that reports `ExtraStatus` (`not_installed` / `installed` / `ready`) via cheap `probe()` calls — import checks only, never model downloads or heavy loads. `enhancement`/`diarization` delegate to `is_enhancement_available()` / `is_diarization_ready()` in their respective modules. The Setup UI (`vocalforge/ui/__init__.py`) renders this registry as an "Optional Extras" panel. New optional pipelines (alignment, separation are stubs) must go through `docs/proposals/_TEMPLATE.md` → explicit approval → wiring through `extras.py`, per `docs/proposals/README.md`. Base transcription must keep working when every extra reports `not_installed`.

`JobController._transcribe_path` is where enhance/diarize are actually invoked in the transcription pipeline, in order: (1) optional DeepFilterNet enhancement (WAV-only, silently skipped for non-WAV uploads or when unavailable), (2) optional PyAnnote diarization (WAV-only; temporarily calls `transcription.release()` to free VRAM if diarizing on CUDA, then reloads the Whisper model afterward), (3) Whisper transcription itself, forcing `word_timestamps=True` mid-flight if diarization is on (needed so `vocalforge/merge.py` can split one Whisper segment across multiple speaker turns by word-level timing overlap).

### Audio paths

- Recording: `vocalforge/audio.py` (`AudioRecorder`, sounddevice-based) → saved WAV at `recorded_audio_path()`.
- Upload: `vocalforge/media.py` validates supported formats (WAV/MP3/M4A/FLAC/OGG/...) before handing off to the same `_transcribe_path` pipeline.
- Enhancement/diarization only run on `.wav` inputs — non-WAV uploads skip both with a logged warning, even if the toggle is on.

### Config and storage

- `vocalforge/config.py`: versioned `AppConfig` dataclass (`SCHEMA_VERSION`), JSON-backed at `config.json` (gitignored, user-local; may hold `hf_token`). Missing/invalid config yields safe defaults rather than erroring.
- `vocalforge/storage.py`: all writable app paths (`Models/`, `audio_files/`, `transcripts/`, `vocalforge.log`) resolve relative to repo root via `default_paths()`. Transcripts are timestamped files with numeric-suffix collision handling; `list_transcripts` powers the history dropdown.

### Text formatting: display vs. stored/copied text differ for RTL scripts

`vocalforge/formatting.py` has two distinct paths that must not be conflated: `format_text()` normalizes whitespace/punctuation/capitalization for Latin-script transcripts (Arabic-script text is left mostly as-is), while `prepare_ui_text()` reshapes Arabic-script text (via `arabic_reshaper` + `python-bidi`) **only for display** in the LTR Tk text widget. Clipboard delivery (`clipboard.py`) and saved transcripts (`storage.py`) must always use the original logical-order text, never the bidi-reshaped display copy — reshaping is presentation-only and would corrupt copy/paste or saved files if persisted.

### CUDA/PortAudio platform quirks

`run.sh` is a **development convenience only** for one Linux laptop (sets `LD_LIBRARY_PATH` to a local extracted PortAudio + optional Ollama-bundled CUDA libs) — it is not the distribution strategy; don't assume its paths generalize. On Windows, `run.bat` just calls `python main.py`. See `docs/runtime_requirements.md` for the CUDA fallback contract and per-OS setup, and the recent commit fixing CUDA runtime library detection across Windows/Linux in `capabilities.py`.
