"""Faster-Whisper model loading and transcription."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from vocalforge.formatting import format_text

logger = logging.getLogger(__name__)


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

    def transcribe(self, file_path: str | Path) -> TranscriptionResult:
        if self._model is None or self._model_name is None:
            raise RuntimeError("No model is loaded.")
        segments, _info = self._model.transcribe(str(file_path))
        text = " ".join(segment.text for segment in segments)
        formatted = format_text(text)
        return TranscriptionResult(
            text=text,
            formatted_text=formatted,
            device=self.device,
            model_name=self._model_name,
        )

    def release(self) -> None:
        self._model = None
        self._model_name = None
