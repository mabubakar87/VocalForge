"""Job controller coordinating model load, recording, and transcription."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from vocalforge.audio import AudioRecorder, save_wav
from vocalforge.clipboard import ClipboardSettings, deliver_text
from vocalforge.state import AppState, StateMachine
from vocalforge.storage import AppPaths, recorded_audio_path, save_transcript
from vocalforge.transcription import TranscriptionService

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    STATE_CHANGED = "STATE_CHANGED"
    MODEL_LOADED = "MODEL_LOADED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    RECORDING_SAVED = "RECORDING_SAVED"
    TRANSCRIPTION_COMPLETED = "TRANSCRIPTION_COMPLETED"
    JOB_FAILED = "JOB_FAILED"
    SHUTDOWN_COMPLETED = "SHUTDOWN_COMPLETED"
    STATUS = "STATUS"


@dataclass(frozen=True)
class JobEvent:
    type: EventType
    job_id: str | None = None
    payload: dict[str, Any] | None = None


class JobController:
    def __init__(
        self,
        paths: AppPaths,
        transcription: TranscriptionService,
        emit: Callable[[JobEvent], None],
        clipboard_settings: ClipboardSettings | None = None,
        recorder: AudioRecorder | None = None,
    ) -> None:
        self.paths = paths
        self.transcription = transcription
        self.emit = emit
        self.clipboard_settings = clipboard_settings or ClipboardSettings(auto_paste=False)
        self.recorder = recorder or AudioRecorder()
        self.state = StateMachine(AppState.STARTING)
        self._lock = threading.Lock()
        self._active_job_id: Optional[str] = None
        self._stop_recording = threading.Event()

    def _set_state(self, target: AppState, job_id: str | None = None) -> bool:
        if not self.state.try_transition(target):
            return False
        self.emit(JobEvent(EventType.STATE_CHANGED, job_id=job_id, payload={"state": target.value}))
        return True

    def bootstrap(self) -> None:
        self._set_state(AppState.NO_MODEL)

    def _begin_job(self, allowed_from_ready: bool = True) -> str | None:
        with self._lock:
            if self.state.is_busy() and self.state.state is not AppState.READY:
                return None
            job_id = uuid.uuid4().hex
            self._active_job_id = job_id
            return job_id

    def _is_current(self, job_id: str) -> bool:
        return self._active_job_id == job_id

    def can_change_profile(self) -> bool:
        return self.state.state in {AppState.NO_MODEL, AppState.READY, AppState.ERROR, AppState.STARTING}

    def activate_profile(self, profile_id: str, device: str, compute_type: str, model_name: str) -> bool:
        """Apply profile runtime settings and load the profile model."""
        if not self.can_change_profile():
            self.emit(
                JobEvent(
                    EventType.JOB_FAILED,
                    payload={"message": "Cannot change profile while a job is active."},
                )
            )
            return False
        self.transcription.device = device
        self.transcription.compute_type = compute_type
        return self.load_model(model_name)

    def load_model(self, model_name: str) -> bool:
        if not model_name:
            self.emit(
                JobEvent(
                    EventType.JOB_FAILED,
                    payload={"message": "No model selected. Choose a model from the dropdown."},
                )
            )
            return False

        if self.state.state is AppState.STARTING:
            self._set_state(AppState.NO_MODEL)
        if self.state.state is AppState.ERROR:
            self._set_state(AppState.READY if self.transcription.is_ready else AppState.NO_MODEL)
        if self.state.state not in {AppState.NO_MODEL, AppState.READY}:
            return False
        if not self._set_state(AppState.LOADING_MODEL):
            return False

        job_id = uuid.uuid4().hex
        self._active_job_id = job_id
        self.emit(JobEvent(EventType.STATUS, job_id=job_id, payload={"text": f"Downloading {model_name}..."}))

        def worker() -> None:
            try:
                device = self.transcription.load_model_with_cpu_fallback(model_name)
                if not self._is_current(job_id):
                    return
                self._set_state(AppState.READY, job_id=job_id)
                self.emit(
                    JobEvent(
                        EventType.MODEL_LOADED,
                        job_id=job_id,
                        payload={"model": model_name, "device": device},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Model load failed")
                if not self._is_current(job_id):
                    return
                if self.transcription.is_ready:
                    self._set_state(AppState.READY, job_id=job_id)
                else:
                    self._set_state(AppState.NO_MODEL, job_id=job_id)
                self.emit(
                    JobEvent(
                        EventType.MODEL_LOAD_FAILED,
                        job_id=job_id,
                        payload={"message": str(exc)},
                    )
                )

        threading.Thread(target=worker, daemon=True).start()
        return True

    def start_recording(self) -> bool:
        if not self.transcription.is_ready or not self.state.can_record():
            return False
        if not self._set_state(AppState.RECORDING):
            return False
        job_id = uuid.uuid4().hex
        self._active_job_id = job_id
        self._stop_recording.clear()
        self.emit(JobEvent(EventType.STATUS, job_id=job_id, payload={"text": "Recording...", "fg": "red"}))

        def worker() -> None:
            try:
                self.recorder.start()
                while not self._stop_recording.is_set():
                    self._stop_recording.wait(0.05)
                audio = self.recorder.stop()
                if not self._is_current(job_id):
                    return
                path = recorded_audio_path(self.paths)
                save_wav(path, audio)
                self.emit(
                    JobEvent(
                        EventType.RECORDING_SAVED,
                        job_id=job_id,
                        payload={"path": str(path)},
                    )
                )
                self._transcribe_path(path, job_id=job_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Recording failed")
                if self._is_current(job_id):
                    self._set_state(AppState.ERROR, job_id=job_id)
                    self.emit(
                        JobEvent(
                            EventType.JOB_FAILED,
                            job_id=job_id,
                            payload={"message": str(exc)},
                        )
                    )
                    if self.transcription.is_ready:
                        self._set_state(AppState.READY, job_id=job_id)

        threading.Thread(target=worker, daemon=True).start()
        return True

    def stop_recording(self) -> None:
        if self.state.state is AppState.RECORDING:
            self.emit(JobEvent(EventType.STATUS, payload={"text": "Processing..."}))
            self._stop_recording.set()

    def toggle_recording(self) -> None:
        if self.state.state is AppState.RECORDING:
            self.stop_recording()
        else:
            self.start_recording()

    def transcribe_upload(self, file_path: str | Path) -> bool:
        if not self.transcription.is_ready or not self.state.can_upload():
            return False
        if not self._set_state(AppState.TRANSCRIBING):
            return False
        job_id = uuid.uuid4().hex
        self._active_job_id = job_id
        self.emit(JobEvent(EventType.STATUS, job_id=job_id, payload={"text": "Transcribing..."}))

        def worker() -> None:
            self._transcribe_path(Path(file_path), job_id=job_id)

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _transcribe_path(self, file_path: Path, job_id: str) -> None:
        try:
            if self.state.state is AppState.RECORDING:
                self._set_state(AppState.TRANSCRIBING, job_id=job_id)
            result = self.transcription.transcribe(file_path)
            if not self._is_current(job_id):
                return
            transcript_path = save_transcript(self.paths, result.formatted_text)
            deliver_text(result.formatted_text, self.clipboard_settings)
            self._set_state(AppState.READY, job_id=job_id)
            self.emit(
                JobEvent(
                    EventType.TRANSCRIPTION_COMPLETED,
                    job_id=job_id,
                    payload={
                        "text": result.formatted_text,
                        "transcript_path": str(transcript_path),
                        "device": result.device,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Transcription failed")
            if not self._is_current(job_id):
                return
            self._set_state(AppState.ERROR, job_id=job_id)
            self.emit(
                JobEvent(
                    EventType.JOB_FAILED,
                    job_id=job_id,
                    payload={"message": str(exc)},
                )
            )
            if self.transcription.is_ready:
                self._set_state(AppState.READY, job_id=job_id)

    def shutdown(self) -> None:
        self._set_state(AppState.SHUTTING_DOWN)
        self._stop_recording.set()
        try:
            self.recorder.close()
        except Exception as exc:  # noqa: BLE001
            logger.error("Recorder cleanup failed: %s", exc)
        self.transcription.release()
        self.emit(JobEvent(EventType.SHUTDOWN_COMPLETED))
