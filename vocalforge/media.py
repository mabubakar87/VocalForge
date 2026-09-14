"""Upload / media format helpers (decoded by Faster-Whisper / PyAV)."""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Target rate for converted uploads (matches ASR / DeepFilterNet input).
ASR_SAMPLE_RATE = 16_000

SUPPORTED_UPLOAD_EXTENSIONS = frozenset(
    {
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg",
        ".opus",
        ".webm",
        ".mp4",
        ".mkv",
        ".aac",
        ".wma",
    }
)

UPLOAD_FILEDIALOG_TYPES = [
    (
        "Audio / video",
        " ".join(f"*{ext}" for ext in sorted(SUPPORTED_UPLOAD_EXTENSIONS)),
    ),
    ("WAV", "*.wav"),
    ("MP3", "*.mp3"),
    ("All files", "*.*"),
]


class MediaError(RuntimeError):
    """Media decode / conversion failed."""


def is_supported_upload(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_UPLOAD_EXTENSIONS


def is_wav_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".wav"


def converted_wav_path(output_dir: str | Path, source: str | Path) -> Path:
    """Destination for a non-WAV upload converted to mono PCM WAV."""
    source = Path(source)
    return Path(output_dir) / f"{source.stem}_converted.wav"


def write_wav_mono_pcm16(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write float32 mono samples in [-1, 1] as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return path


def decode_to_wav(
    source: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = ASR_SAMPLE_RATE,
) -> Path:
    """Decode any PyAV-readable audio/video file to mono 16-bit PCM WAV.

    Uses the first audio stream. Raises MediaError on failure.
    """
    source = Path(source)
    output_path = Path(output_path)
    if not source.is_file():
        raise MediaError(f"Media file not found: {source}")

    try:
        import av
    except ImportError as exc:
        raise MediaError(
            "PyAV (av) is required to decode uploads. It is normally installed with faster-whisper."
        ) from exc

    try:
        container = av.open(str(source))
    except av.AVError as exc:
        raise MediaError(f"Could not open media file: {source}") from exc

    try:
        if not container.streams.audio:
            raise MediaError(f"No audio stream found in: {source.name}")

        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )
        chunks: list[np.ndarray] = []

        def _consume(frame: av.AudioFrame | None) -> None:
            for out_frame in resampler.resample(frame):
                # plane 0 is interleaved mono s16
                arr = out_frame.to_ndarray()
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                chunks.append(arr.astype(np.int16, copy=False))

        for frame in container.decode(audio=0):
            _consume(frame)
        _consume(None)  # flush resampler
    except MediaError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"Failed to decode audio from {source.name}: {exc}") from exc
    finally:
        container.close()

    if not chunks:
        raise MediaError(f"Decoded empty audio from: {source.name}")

    pcm = np.concatenate(chunks)
    samples = pcm.astype(np.float32) / 32768.0
    write_wav_mono_pcm16(output_path, samples, sample_rate)
    logger.info(
        "Converted %s → %s (%d Hz mono PCM, %.2fs).",
        source.name,
        output_path.name,
        sample_rate,
        len(samples) / float(sample_rate),
    )
    return output_path


def ensure_wav(
    source: str | Path,
    output_dir: str | Path,
    *,
    sample_rate: int = ASR_SAMPLE_RATE,
) -> Path:
    """Return a WAV path for *source*, converting via PyAV when needed.

    Existing ``.wav`` files are returned as-is (resolved). Other supported
    formats are decoded to ``{stem}_converted.wav`` under *output_dir*.
    """
    source = Path(source)
    if is_wav_path(source):
        return source.resolve()
    dest = converted_wav_path(output_dir, source)
    return decode_to_wav(source, dest, sample_rate=sample_rate)
