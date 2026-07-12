#!/usr/bin/env bash
# Launch VocalForge with PortAudio + CUDA runtime libs on LD_LIBRARY_PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PORTAUDIO_LIB="$ROOT/.deps/extracted/usr/lib/x86_64-linux-gnu"
# Ollama ships CUDA 12 libs; use them when a full CUDA toolkit isn't installed.
OLLAMA_CUDA="/usr/local/lib/ollama/cuda_v12"
EXTRA_LIBS="$PORTAUDIO_LIB"
if [[ -d "$OLLAMA_CUDA" ]]; then
  EXTRA_LIBS="$OLLAMA_CUDA:$EXTRA_LIBS"
fi
export LD_LIBRARY_PATH="$EXTRA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# shellcheck source=/dev/null
source "$ROOT/venv/bin/activate"
exec python "$ROOT/main.py" "$@"
