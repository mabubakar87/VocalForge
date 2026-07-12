"""Optional GPU integration tests. Run with: pytest -m gpu"""

from __future__ import annotations

from pathlib import Path

import pytest

from vocalforge.capabilities import evaluate_capabilities
from vocalforge.storage import default_paths
from vocalforge.transcription import TranscriptionService

FIXTURE = Path(__file__).parent / "fixtures" / "silence_1s.wav"


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.slow
def test_gpu_transcribe_silence_fixture():
    caps = evaluate_capabilities()
    if caps.selected_device != "cuda":
        pytest.skip(caps.fallback_reason or "CUDA not available")
    paths = default_paths()
    service = TranscriptionService(paths.models, device="cuda")
    # Prefer a tiny already-cached model name if present; otherwise skip.
    model = "distil-small.en"
    try:
        device = service.load_model_with_cpu_fallback(model)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"model unavailable: {exc}")
    assert device == "cuda"
    result = service.transcribe(FIXTURE)
    assert isinstance(result.formatted_text, str)
