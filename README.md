# VocalForge

Desktop speech-to-text app built with **Tkinter** and **Faster-Whisper**. Record from the mic or upload audio, transcribe locally, and copy formatted text to the clipboard. Optional Phase 4 extras add enhancement and speaker diarization without changing the base install.

Launch on this Linux laptop with `./run.sh` (PortAudio + CUDA library paths). Elsewhere: `python main.py` inside the venv.

## Features

- **Hardware profiles** (Setup): Lightweight → High Accuracy, sized smallest to largest
- **Multilingual Small / Medium**: full Whisper **speech→English translate** (`task=translate`)
- **High Accuracy (`large-v3-turbo`)**: fast multilingual ASR — **not** trained for translate
- **Home session controls**: Processing (CPU/GPU), Language, VAD, beam, timestamps, Enhance, Diarize, Translate
- **Optional extras**: DeepFilterNet enhance; offline PyAnnote speaker diarization
- Multi-format upload, cancel, history, clipboard delivery

## Profiles (Setup)

| Profile | Model | Approx. size | Translate → English |
|---------|--------|--------------|---------------------|
| Lightweight | `distil-small.en` | ~160 MB | No (English-only) |
| Multilingual Small | `small` | ~480 MB | **Yes** (recommended for translate) |
| Balanced | `distil-medium.en` | ~800 MB | No (English-only) |
| Multilingual Medium | `medium` | ~1.5 GB | **Yes** |
| High Accuracy | `large-v3-turbo` | ~1.6 GB | **No** (turbo) |

Models live under `Models/` (gitignored). First download needs network.

## Requirements

- Python **3.10+** (verified on 3.11)
- System: Tkinter, PortAudio — see `docs/runtime_requirements.md`
- Base Python packages: `requirements.txt`
- Optional extras: `requirements-extras.txt` (do **not** mix into base until you need them)

## Installation

```bash
git clone https://github.com/mabubakar87/VocalForge.git
cd VocalForge
python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
./run.sh                   # or: python main.py
```

Dev/tests: `pip install -r requirements-dev.txt && pytest`

### Optional: enhancement (DeepFilterNet)

Place the Rust `deep-filter` binary at `.deps/deep-filter` and install `soxr` from `requirements-extras.txt`. Toggle **Enhance** on the home screen.

### Optional: speaker diarization (PyAnnote)

```bash
# Prefer CUDA torch for long files (CPU torch cannot use the GPU for pyannote)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install 'pyannote.audio>=3.1.0,<5'
# Accept gated model terms on Hugging Face once, then vendor offline weights:
export HF_TOKEN=hf_...   # if cache empty
PYTHONPATH=. python scripts/vendor_diarization_models.py
```

Runtime loads `Models/diarization/pyannote_community_1/` with `HF_HUB_OFFLINE=1` (no Hub checks per job). Toggle **Diarize** when Setup shows Ready.

Smoke: `PYTHONPATH=. python scripts/smoke_test_diarization.py path/to.wav`

### Translate to English

1. Activate **Multilingual Small** or **Multilingual Medium** in Setup  
2. Set **Language** (source) e.g. Chinese or Auto-detect  
3. Turn **Translate** On  

Do not expect translate on `large-v3-turbo`.

## Usage

1. Open **Setup**, activate a profile (download if needed).
2. On home: set Processing / Language / VAD / Decode options as needed.
3. Record (circle button) or **Upload**; optional Enhance / Diarize / Translate.
4. Transcript appears in the text area and is copied to the clipboard.
5. Use history to reload recent transcripts.

Global hotkey (`Ctrl+Q`) may require root on Linux; use the record button if unavailable.

## Documentation

| Doc | Topic |
|-----|--------|
| `docs/phase_2.md` | Hardware profiles & Setup |
| `docs/phase_3.md` | Transcription UX |
| `docs/phase_4.md` | Optional extras (enhance, diarize) |
| `docs/proposals/` | Extra evaluation proposals |
| `docs/runtime_requirements.md` | PortAudio / CUDA |
| `docs/clean_install.md` | Clean install notes |

## Logging

Application log: `vocalforge.log` in the project root (see app startup).

## Notes

- First Whisper / diarization vendor download needs the network; later runs can be offline for those assets.
- Uploads support common formats when the media helpers allow them; recordings are WAV.
- `config.json` is local and gitignored (may hold `hf_token` for one-time diarization vendor).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Faster-Whisper, CTranslate2, Tkinter, sounddevice, DeepFilterNet, pyannote.audio.
