from pathlib import Path
import threading

from vocalforge.transcription import TranscriptionCancelled, TranscriptionService, format_timestamp


class FakeSegment:
    def __init__(self, text: str, start: float = 0.0, end: float = 1.0) -> None:
        self.text = text
        self.start = start
        self.end = end


class FakeInfo:
    def __init__(self, duration: float = 2.0) -> None:
        self.duration = duration


class FakeModel:
    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def transcribe(self, _path: str, **kwargs):
        self.last_kwargs = kwargs
        return [FakeSegment(" hello "), FakeSegment("world.")], FakeInfo(2.0)


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


def test_transcribe_passes_language_and_task(tmp_path: Path):
    model = FakeModel()

    def factory(name, **kwargs):
        return model

    service = TranscriptionService(tmp_path, device="cpu", model_factory=factory)
    service.load_model("large-v3-turbo")
    service.set_language_settings("en", "transcribe")
    service.transcribe(tmp_path / "x.wav")
    assert model.last_kwargs["task"] == "transcribe"
    assert model.last_kwargs["language"] == "en"

    service.set_language_settings("ur", "transcribe")
    service.transcribe(tmp_path / "x.wav")
    assert model.last_kwargs["task"] == "transcribe"
    assert model.last_kwargs["language"] == "ur"


def test_transcribe_passes_vad_filter(tmp_path: Path):
    model = FakeModel()

    def factory(name, **kwargs):
        return model

    service = TranscriptionService(tmp_path, device="cpu", model_factory=factory)
    service.load_model("distil-small.en")
    service.set_vad_filter(True)
    service.transcribe(tmp_path / "x.wav")
    assert model.last_kwargs["vad_filter"] is True
    assert model.last_kwargs["vad_parameters"]["min_silence_duration_ms"] == 500

    service.set_vad_filter(False)
    service.transcribe(tmp_path / "x.wav")
    assert model.last_kwargs["vad_filter"] is False
    assert "vad_parameters" not in model.last_kwargs


def test_transcribe_passes_beam_and_timestamps(tmp_path: Path):
    model = FakeModel()

    def factory(name, **kwargs):
        return model

    service = TranscriptionService(tmp_path, device="cpu", model_factory=factory)
    service.load_model("distil-small.en")
    service.set_decode_options(beam_size=10, word_timestamps=True)
    result = service.transcribe(tmp_path / "x.wav")
    assert model.last_kwargs["beam_size"] == 10
    assert model.last_kwargs["word_timestamps"] is True
    assert "[" in result.formatted_text
    assert "–" in result.formatted_text


def test_transcribe_cancel_and_progress(tmp_path: Path):
    class SlowModel:
        def transcribe(self, _path: str, **kwargs):
            def gen():
                yield FakeSegment("one", 0.0, 1.0)
                yield FakeSegment("two", 1.0, 2.0)

            return gen(), FakeInfo(2.0)

    service = TranscriptionService(tmp_path, device="cpu", model_factory=lambda *a, **k: SlowModel())
    service.load_model("distil-small.en")
    cancel = threading.Event()
    progress: list[float] = []

    def on_progress(value: float) -> None:
        progress.append(value)
        if value >= 0.5:
            cancel.set()

    try:
        service.transcribe(
            tmp_path / "x.wav",
            cancel_event=cancel,
            progress_callback=on_progress,
        )
        assert False, "expected TranscriptionCancelled"
    except TranscriptionCancelled:
        pass
    assert progress
    assert format_timestamp(65) == "1:05"
