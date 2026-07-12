"""Optional speech enhancement via DeepFilterNet (Rust CLI) + soxr resample.

Pipeline: 16 kHz mono → 48 kHz → deep-filter → 16 kHz mono.
Base ASR does not import this module unless enhancement is enabled.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

ASR_RATE = 16_000
DF_RATE = 48_000


class EnhancementError(RuntimeError):
    """Enhancement failed or is unavailable."""


@dataclass(frozen=True)
class EnhancementResult:
    output_path: Path
    duration_s: float
    backend: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_deep_filter_binary(
    explicit: str | Path | None = None,
    *,
    which: Callable[[str], str | None] | None = None,
) -> Path | None:
    """Locate the deep-filter CLI (PATH, DEEP_FILTER_BIN, or .deps/deep-filter)."""
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_file() and os.access(path, os.X_OK) else None

    env = os.environ.get("DEEP_FILTER_BIN")
    if env:
        path = Path(env)
        if path.is_file() and os.access(path, os.X_OK):
            return path

    finder = which or shutil.which
    on_path = finder("deep-filter")
    if on_path:
        return Path(on_path)

    bundled = _repo_root() / ".deps" / "deep-filter"
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return bundled
    return None


def is_enhancement_available(binary: str | Path | None = None) -> bool:
    try:
        import soxr  # noqa: F401
    except ImportError:
        return False
    return resolve_deep_filter_binary(binary) is not None


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Return float32 mono samples in [-1, 1] and sample rate."""
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise EnhancementError(f"Only 16-bit PCM WAV supported (got sampwidth={width}).")
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def write_wav_mono(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


def resample_audio(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples.astype(np.float32, copy=False)
    try:
        import soxr
    except ImportError as exc:
        raise EnhancementError(
            "soxr is required for enhancement resampling. "
            "Install the enhancement extras (see requirements-extras.txt)."
        ) from exc
    out = soxr.resample(samples.astype(np.float32), src_rate, dst_rate)
    return np.asarray(out, dtype=np.float32)


def _run_deep_filter(binary: Path, input_48k: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    # -D compensates STFT/model lookahead delay so enhanced audio stays aligned.
    cmd = [str(binary), "-D", "-o", str(output_dir), str(input_48k)]
    logger.info("Running deep-filter: %s", " ".join(cmd))
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise EnhancementError("deep-filter timed out.") from exc
    except OSError as exc:
        raise EnhancementError(f"Failed to launch deep-filter: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise EnhancementError(
            f"deep-filter exited {completed.returncode}"
            + (f": {detail}" if detail else "")
        )

    # CLI writes <stem> or <stem>_DeepFilterNet*.wav into output_dir.
    candidates = sorted(output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise EnhancementError(f"deep-filter produced no WAV in {output_dir}")
    # Prefer a file that is not the input copy name collision.
    for candidate in candidates:
        if candidate.name != input_48k.name or candidate.resolve() != input_48k.resolve():
            return candidate
    return candidates[0]


def enhance_wav(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    binary: str | Path | None = None,
    work_dir: str | Path | None = None,
) -> EnhancementResult:
    """Enhance a WAV file and return a 16 kHz mono result path."""
    import time

    started = time.perf_counter()
    src = Path(input_path)
    if not src.is_file():
        raise EnhancementError(f"Input audio not found: {src}")

    df_bin = resolve_deep_filter_binary(binary)
    if df_bin is None:
        raise EnhancementError(
            "deep-filter binary not found. Place it at .deps/deep-filter, "
            "add it to PATH, or set DEEP_FILTER_BIN."
        )

    samples, rate = read_wav_mono(src)
    if samples.size == 0:
        raise EnhancementError("Input audio is empty.")

    samples_48 = resample_audio(samples, rate, DF_RATE)

    out_path = Path(output_path) if output_path else src.with_name(f"{src.stem}_enhanced.wav")
    cleanup_root: Path | None = None
    try:
        if work_dir is not None:
            root = Path(work_dir)
            root.mkdir(parents=True, exist_ok=True)
        else:
            cleanup_root = Path(tempfile.mkdtemp(prefix="vf-enhance-"))
            root = cleanup_root

        input_48 = root / "input_48k.wav"
        write_wav_mono(input_48, samples_48, DF_RATE)
        df_out_dir = root / "df_out"
        enhanced_48_path = _run_deep_filter(df_bin, input_48, df_out_dir)
        enhanced_48, enhanced_rate = read_wav_mono(enhanced_48_path)
        if enhanced_rate != DF_RATE:
            enhanced_48 = resample_audio(enhanced_48, enhanced_rate, DF_RATE)
        enhanced_16 = resample_audio(enhanced_48, DF_RATE, ASR_RATE)
        write_wav_mono(out_path, enhanced_16, ASR_RATE)
    finally:
        if cleanup_root is not None:
            shutil.rmtree(cleanup_root, ignore_errors=True)

    duration = time.perf_counter() - started
    logger.info("Enhanced %s → %s in %.2fs via deep-filter", src, out_path, duration)
    return EnhancementResult(output_path=out_path, duration_s=duration, backend="deep-filter")


def samples_from_int16(pcm: array, sample_rate: int = ASR_RATE) -> tuple[np.ndarray, int]:
    data = np.frombuffer(pcm.tobytes(), dtype=np.int16).astype(np.float32) / 32768.0
    return data, sample_rate
