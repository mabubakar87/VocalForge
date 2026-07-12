# VocalForge native runtime requirements

Supported Python: **3.10+** (verified on 3.11).

## CPU-only (all platforms)

1. Install system PortAudio so `sounddevice` can open the microphone.
2. Create a virtual environment and install Python dependencies.
3. Launch with `python main.py`.

### Linux (Debian/Ubuntu)

```bash
sudo apt-get install -y portaudio19-dev python3-tk
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python main.py
```

### Windows

- Install a supported Python build that includes Tkinter.
- Install PortAudio via the sounddevice wheels when available, or install a
  PortAudio DLL on `PATH`.
- Then:

```bat
python -m venv venv
venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
python main.py
```

### macOS

```bash
brew install portaudio
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python main.py
```

## CUDA (optional)

CUDA is used only when all of the following succeed:

1. An NVIDIA driver is available (`nvidia-smi`).
2. CTranslate2 reports at least one CUDA device.
3. Required CUDA runtime libraries are loadable (for example `libcublas` and
   `libcudart` matching the CTranslate2 build).
4. Model initialization succeeds on CUDA.

If any step fails, VocalForge falls back to CPU and shows a warning.

Do **not** rely on machine-specific paths such as
`/usr/local/lib/ollama/cuda_v12` for a supported installation. Install the
CUDA runtime libraries required by your CTranslate2 wheel, or run in CPU mode.

`run.sh` is a **development convenience** for this repository’s Linux laptop
setup (local PortAudio extract + optional Ollama CUDA libs). It is not the
distribution strategy.

## Development launcher

```bash
./run.sh
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Default tests exclude GPU, microphone, network, and slow markers.
