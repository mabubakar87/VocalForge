# Clean-install verification (Phase 1)

Perform these steps on a machine/environment that does **not** depend on this
repository's `.deps/` directory or Ollama CUDA paths unless you are validating
the optional development launcher.

## Procedure

1. Create the supported Python virtual environment (3.10+).
2. Install runtime dependencies: `pip install -r requirements.txt`.
3. Install system PortAudio and Tkinter as documented in
   `docs/runtime_requirements.md`.
4. Run default tests: `pip install -r requirements-dev.txt && pytest`.
5. Launch in CPU mode (`python main.py`) or after unsetting CUDA-related
   library paths.
6. Select/download `distil-small.en`.
7. Upload `tests/fixtures/silence_1s.wav` and confirm the UI completes without
   crashing.
8. Quit the app. Disable networking (or unplug) and relaunch.
9. With the model already cached under `Models/`, load it again and transcribe
   the fixture offline.

## Pass criteria

- Default `pytest` is green without network/GPU/microphone.
- App launches and reaches a usable state.
- Cached-model offline transcription works.
- Missing CUDA libraries produce CPU fallback rather than a crash.

## Notes

- `./run.sh` may still be used on the original development laptop; it is not
  required for a clean install.
- Record spoken samples separately if you need non-empty transcripts.
