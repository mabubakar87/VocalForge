"""Microphone recording and WAV serialization."""

from __future__ import annotations

import array
import logging
import queue
import threading
import wave
from pathlib import Path
from typing import Optional

import sounddevice as sd

logger = logging.getLogger(__name__)


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


class AudioRecorder:
    """Capture 16 kHz mono int16 audio until stopped."""

    def __init__(self, samplerate: int = 16000, channels: int = 1) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self._queue: queue.Queue = queue.Queue()
        self._recording = False
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            logger.warning("Audio input status: %s", status)
        if self._recording:
            self._queue.put(indata.copy())

    def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._clear_queue()
            self._recording = True
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype="int16",
                callback=self._callback,
            )
            self._stream.start()

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
