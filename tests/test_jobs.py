import time
from pathlib import Path

from vocalforge.clipboard import ClipboardSettings
from vocalforge.jobs import EventType, JobController, JobEvent
from vocalforge.state import AppState
from vocalforge.storage import default_paths
from vocalforge.transcription import TranscriptionService


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    def transcribe(self, _path: str, **kwargs):
        return [FakeSegment("test complete")], None


def _wait_for(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_model_load_job(tmp_path: Path):
    events: list[JobEvent] = []

    def factory(name, **kwargs):
        return FakeModel()

    paths = default_paths(tmp_path)
    service = TranscriptionService(paths.models, device="cpu", model_factory=factory)
    controller = JobController(
        paths,
        service,
        emit=events.append,
        clipboard_settings=ClipboardSettings(auto_paste=False),
    )
    controller.bootstrap()
    assert controller.load_model("distil-small.en")
    assert _wait_for(lambda: controller.state.state is AppState.READY)
    assert any(e.type is EventType.MODEL_LOADED for e in events)


def test_rejects_record_without_model(tmp_path: Path):
    paths = default_paths(tmp_path)
    service = TranscriptionService(paths.models, device="cpu", model_factory=lambda *a, **k: FakeModel())
    controller = JobController(paths, service, emit=lambda e: None)
    controller.bootstrap()
    assert controller.start_recording() is False


def test_rejects_unsupported_upload(tmp_path: Path):
    events: list[JobEvent] = []

    def factory(name, **kwargs):
        return FakeModel()

    paths = default_paths(tmp_path)
    service = TranscriptionService(paths.models, device="cpu", model_factory=factory)
    controller = JobController(paths, service, emit=events.append)
    controller.bootstrap()
    assert controller.load_model("distil-small.en")
    assert _wait_for(lambda: controller.state.state is AppState.READY)
    bad = tmp_path / "notes.txt"
    bad.write_text("nope", encoding="utf-8")
    assert controller.transcribe_upload(bad) is False
    assert any(e.type is EventType.JOB_FAILED for e in events)


def test_cancel_during_transcription(tmp_path: Path):
    events: list[JobEvent] = []

    class BlockingModel:
        def transcribe(self, _path: str, **kwargs):
            import time

            def gen():
                yield type("Seg", (), {"text": "a", "start": 0.0, "end": 0.5})()
                time.sleep(0.35)
                yield type("Seg", (), {"text": "b", "start": 0.5, "end": 1.0})()

            return gen(), type("Info", (), {"duration": 1.0})()

    paths = default_paths(tmp_path)
    service = TranscriptionService(
        paths.models, device="cpu", model_factory=lambda *a, **k: BlockingModel()
    )
    controller = JobController(paths, service, emit=events.append)
    controller.bootstrap()
    assert controller.load_model("distil-small.en")
    assert _wait_for(lambda: controller.state.state is AppState.READY)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    assert controller.transcribe_upload(audio)
    assert _wait_for(lambda: any(e.type is EventType.PROGRESS for e in events))
    assert controller.cancel()
    assert _wait_for(lambda: controller.state.state is AppState.READY)
    assert any(e.type is EventType.TRANSCRIPTION_CANCELLED for e in events)
    assert service.is_ready
