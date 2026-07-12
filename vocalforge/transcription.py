"""Faster-Whisper model loading and transcription."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from vocalforge.formatting import format_text

logger = logging.getLogger(__name__)

DEFAULT_BEAM_SIZE = 5
BEAM_SIZE_CHOICES = (1, 5, 10)


class TranscriptionCancelled(Exception):
    """Raised when a transcription job is cancelled mid-stream."""


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


@dataclass
class TranscriptionResult:
    text: str
    formatted_text: str
    device: str
    model_name: str


class TranscriptionService:
    """Owns the active Whisper model and transcription calls."""

    def __init__(
        self,
        download_root: Path,
        device: str = "cpu",
        compute_type: str | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.download_root = Path(download_root)
        self.device = device
        self.compute_type = compute_type or ("int8" if device == "cpu" else "default")
        self.language: str | None = None
        self.task: str = "transcribe"
        self.vad_filter: bool = True
        self.beam_size: int = DEFAULT_BEAM_SIZE
        self.word_timestamps: bool = False
        self._model_factory = model_factory
        self._model: Any = None
        self._model_name: Optional[str] = None

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _create_model(self, model_name: str, device: str, compute_type: str) -> Any:
        if self._model_factory is not None:
            return self._model_factory(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=str(self.download_root),
            )
        from faster_whisper import WhisperModel

        kwargs: dict[str, Any] = {
            "device": device,
            "download_root": str(self.download_root),
        }
        if compute_type and compute_type != "default":
            kwargs["compute_type"] = compute_type
        return WhisperModel(model_name, **kwargs)

    def load_model(self, model_name: str, device: str | None = None, compute_type: str | None = None) -> str:
        """Load a model. On failure, keep the previous model if one exists."""
        target_device = device or self.device
        target_compute = compute_type or self.compute_type
        previous_model = self._model
        previous_name = self._model_name
        try:
            new_model = self._create_model(model_name, target_device, target_compute)
        except Exception:
            # Preserve previous model on failure.
            self._model = previous_model
            self._model_name = previous_name
            raise

        self._model = new_model
        self._model_name = model_name
        self.device = target_device
        self.compute_type = target_compute
        logger.info("Model '%s' loaded on %s.", model_name, target_device)
        return target_device

    def load_model_with_cpu_fallback(self, model_name: str) -> str:
        """Try configured device, then fall back to CPU once on CUDA runtime errors."""
        try:
            return self.load_model(model_name)
        except Exception as exc:
            message = str(exc).lower()
            cuda_related = any(
                token in message
                for token in ("cublas", "cuda", "cudnn", "libcudart", "nvrtc")
            )
            if self.device != "cpu" and cuda_related:
                logger.warning("CUDA model load failed (%s); retrying on CPU.", exc)
                return self.load_model(model_name, device="cpu", compute_type="int8")
            raise

    def set_language_settings(self, language: str | None, task: str = "transcribe") -> None:
        self.language = language
        self.task = task if task in {"transcribe", "translate"} else "transcribe"

    def set_vad_filter(self, enabled: bool) -> None:
        self.vad_filter = bool(enabled)

    def set_decode_options(self, *, beam_size: int | None = None, word_timestamps: bool | None = None) -> None:
        if beam_size is not None:
            self.beam_size = max(1, int(beam_size))
        if word_timestamps is not None:
            self.word_timestamps = bool(word_timestamps)

    def transcribe(
        self,
        file_path: str | Path,
        *,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        if self._model is None or self._model_name is None:
            raise RuntimeError("No model is loaded.")
        kwargs: dict[str, Any] = {
            "task": self.task,
            "vad_filter": self.vad_filter,
            "beam_size": self.beam_size,
            "word_timestamps": self.word_timestamps,
        }
        if self.language:
            kwargs["language"] = self.language
        if self.vad_filter:
            # Slightly less conservative than library default (2000ms) for short dictation.
            kwargs["vad_parameters"] = {"min_silence_duration_ms": 500}
        logger.info(
            "Transcribing %s (vad_filter=%s, language=%s, beam_size=%s, word_timestamps=%s).",
            file_path,
            self.vad_filter,
            self.language or "auto",
            self.beam_size,
            self.word_timestamps,
        )
        segments, info = self._model.transcribe(str(file_path), **kwargs)
        duration = float(getattr(info, "duration", 0.0) or 0.0) if info is not None else 0.0
        pieces: list[str] = []
        for segment in segments:
            if cancel_event is not None and cancel_event.is_set():
                raise TranscriptionCancelled()
            raw = (getattr(segment, "text", "") or "").strip()
            if self.word_timestamps:
                start = float(getattr(segment, "start", 0.0) or 0.0)
                end = float(getattr(segment, "end", start) or start)
                if raw:
                    pieces.append(f"[{format_timestamp(start)}–{format_timestamp(end)}] {raw}")
            elif raw:
                pieces.append(raw)
            if progress_callback is not None and duration > 0:
                end = float(getattr(segment, "end", 0.0) or 0.0)
                progress_callback(min(1.0, max(0.0, end / duration)))

        if progress_callback is not None:
            progress_callback(1.0)

        text = "\n".join(pieces) if self.word_timestamps else " ".join(pieces)
        formatted = format_text(text) if not self.word_timestamps else text
        return TranscriptionResult(
            text=text,
            formatted_text=formatted,
            device=self.device,
            model_name=self._model_name,
        )

    def release(self) -> None:
        self._model = None
        self._model_name = None
