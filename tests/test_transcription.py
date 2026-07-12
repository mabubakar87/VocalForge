from pathlib import Path

from vocalforge.transcription import TranscriptionService


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    def transcribe(self, _path: str):
        return [FakeSegment(" hello "), FakeSegment("world.")], None


def test_load_preserves_previous_model_on_failure(tmp_path: Path):
    calls = {"n": 0}

    def factory(name, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeModel()
        raise RuntimeError("cublas missing")

    service = TranscriptionService(tmp_path, device="cuda", model_factory=factory)
    service.load_model("first")
    assert service.is_ready
    try:
        service.load_model("second")
        assert False, "expected failure"
    except RuntimeError:
        pass
    assert service.model_name == "first"
    result = service.transcribe(tmp_path / "x.wav")
    assert "Hello" in result.formatted_text or result.formatted_text.startswith("Hello")


def test_cpu_fallback_on_cuda_error(tmp_path: Path):
    devices = []

    def factory(name, **kwargs):
        devices.append(kwargs["device"])
        if kwargs["device"] == "cuda":
            raise RuntimeError("Library libcublas.so.12 is not found")
        return FakeModel()

    service = TranscriptionService(tmp_path, device="cuda", model_factory=factory)
    used = service.load_model_with_cpu_fallback("distil-small.en")
    assert used == "cpu"
    assert devices == ["cuda", "cpu"]
