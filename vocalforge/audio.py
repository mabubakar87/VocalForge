"""Microphone recording and WAV serialization."""

from __future__ import annotations

import array
import logging
import queue
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_INPUT_LABEL = "System default"


@dataclass(frozen=True)
class InputDevice:
    """Selectable microphone / input device."""

    name: str
    index: int | None  # None means PortAudio/sounddevice default
    max_input_channels: int = 1

    @property
    def label(self) -> str:
        if self.index is None:
            return DEFAULT_INPUT_LABEL
        return self.name


def save_wav(filename: str | Path, data: array.array, samplerate: int = 16000) -> Path:
    """Write 16-bit mono PCM samples to a WAV file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(data.tobytes())
    logger.info("Audio saved to %s.", path)
    return path


def _import_sounddevice():
    """Import sounddevice lazily so unit tests can load this module without PortAudio."""
    try:
        import sounddevice as sd
    except OSError as exc:
        raise OSError(
            "PortAudio library not found. Install PortAudio (see docs/runtime_requirements.md) "
            "or launch with ./run.sh on this development machine."
        ) from exc
    return sd


def list_input_devices(
    query_devices: Callable[[], Any] | None = None,
) -> list[InputDevice]:
    """Return input-capable devices, always starting with System default."""
    devices = [InputDevice(name=DEFAULT_INPUT_LABEL, index=None, max_input_channels=1)]
    try:
        if query_devices is not None:
            raw = query_devices()
        else:
            sd = _import_sounddevice()
            raw = sd.query_devices()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not enumerate audio input devices: %s", exc)
        return devices

    seen_names: set[str] = set()
    for index, info in enumerate(raw):
        try:
            channels = int(info.get("max_input_channels", 0) or 0)
            name = str(info.get("name", f"Device {index}")).strip() or f"Device {index}"
        except Exception:  # noqa: BLE001
            continue
        if channels <= 0:
            continue
        # Keep names unique in the UI; later duplicates get an index suffix.
        label = name
        if label in seen_names:
            label = f"{name} ({index})"
        seen_names.add(label)
        devices.append(InputDevice(name=label, index=index, max_input_channels=channels))
    return devices


def resolve_input_device_index(
    saved_name: str | None,
    devices: list[InputDevice] | None = None,
) -> int | None:
    """Map a saved device label to a sounddevice device index (None = default)."""
    if not saved_name or saved_name == DEFAULT_INPUT_LABEL:
        return None
    catalog = devices if devices is not None else list_input_devices()
    for device in catalog:
        if device.name == saved_name and device.index is not None:
            return device.index
    logger.warning("Saved input device %r is unavailable; using system default.", saved_name)
    return None


class AudioRecorder:
    """Capture 16 kHz mono int16 audio until stopped."""

    def __init__(
        self,
        samplerate: int = 16000,
        channels: int = 1,
        device: int | None = None,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self._queue: queue.Queue = queue.Queue()
        self._recording = False
        self._stream: Optional[Any] = None
        self._lock = threading.Lock()

    def set_device(self, device: int | None) -> None:
        """Select PortAudio device index; None uses the system default."""
        with self._lock:
            if self._recording:
                raise RuntimeError("Cannot change input device while recording.")
            self.device = device

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.warning("Audio input status: %s", status)
        if self._recording:
            self._queue.put(indata.copy())

    def start(self) -> None:
        sd = _import_sounddevice()
        with self._lock:
            if self._recording:
                return
            self._clear_queue()
            self._recording = True
            kwargs: dict[str, Any] = {
                "samplerate": self.samplerate,
                "channels": self.channels,
                "dtype": "int16",
                "callback": self._callback,
            }
            if self.device is not None:
                kwargs["device"] = self.device
            self._stream = sd.InputStream(**kwargs)
            self._stream.start()
            logger.info(
                "Recording started on %s.",
                "system default" if self.device is None else f"device index {self.device}",
            )

    def stop(self) -> array.array:
        """Stop capture and return collected samples."""
        with self._lock:
            self._recording = False
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001
                logger.error("Error closing audio stream: %s", exc)

        audio_data = array.array("h")
        while True:
            try:
                chunk = self._queue.get_nowait()
            except queue.Empty:
                break
            audio_data.extend(chunk.flatten())
        return audio_data

    def close(self) -> None:
        """Idempotent cleanup."""
        if self._recording or self._stream is not None:
            self.stop()

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


def record_until_flag(stop_flag: threading.Event, samplerate: int = 16000) -> array.array:
    """Record until ``stop_flag`` is set. Compatibility helper for existing callers."""
    recorder = AudioRecorder(samplerate=samplerate)
    recorder.start()
    try:
        while not stop_flag.is_set():
            stop_flag.wait(0.05)
        return recorder.stop()
    finally:
        recorder.close()
