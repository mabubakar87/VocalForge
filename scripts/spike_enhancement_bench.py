#!/usr/bin/env python3
"""P4-060 enhancement spike: fixtures, timing, peak RSS, artifact checks."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np

from vocalforge.enhancement import (
    ASR_RATE,
    enhance_wav,
    is_enhancement_available,
    read_wav_mono,
    write_wav_mono,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "enhancement"
OUT = ROOT / "tests" / "fixtures" / "enhancement" / "out"


def _peak_rss_mb() -> float:
    # Linux: ru_maxrss is kilobytes.
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0


def _peak_self_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def make_fixtures() -> dict[str, Path]:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    duration_s = 3.0
    n = int(ASR_RATE * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float32)

    # Synthetic "speech-like" harmonics (not real speech, but stable for smoke).
    clean = (
        0.22 * np.sin(2 * np.pi * 180.0 * t)
        + 0.12 * np.sin(2 * np.pi * 360.0 * t)
        + 0.06 * np.sin(2 * np.pi * 540.0 * t)
    ).astype(np.float32)
    # Amplitude envelope to mimic syllables.
    env = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t))
    clean = (clean * env).astype(np.float32)

    noise = (0.08 * rng.standard_normal(n)).astype(np.float32)
    noisy = np.clip(clean + noise, -1.0, 1.0).astype(np.float32)

    clean_path = FIXTURES / "clean_3s_16k.wav"
    noisy_path = FIXTURES / "noisy_3s_16k.wav"
    write_wav_mono(clean_path, clean, ASR_RATE)
    write_wav_mono(noisy_path, noisy, ASR_RATE)
    return {"clean": clean_path, "noisy": noisy_path, "clean_ref": clean}


def spectral_flatness(samples: np.ndarray) -> float:
    """Rough artifact proxy: very high flatness ≈ noise-like / destroyed speech."""
    window = np.hanning(min(len(samples), 2048))
    chunk = samples[: len(window)] * window
    mag = np.abs(np.fft.rfft(chunk)) + 1e-12
    geo = float(np.exp(np.mean(np.log(mag))))
    arith = float(np.mean(mag))
    return geo / arith


def best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 4000) -> tuple[float, int]:
    """Max Pearson correlation allowing a small delay (STFT lookahead)."""
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    best = -1.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1, 16):
        if lag < 0:
            x, y = a[-lag:], b[: n + lag]
        elif lag > 0:
            x, y = a[: n - lag], b[lag:]
        else:
            x, y = a, b
        if len(x) < 256:
            continue
        if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if corr > best:
            best = corr
            best_lag = lag
    return best, best_lag


def run() -> dict:
    if not is_enhancement_available():
        raise SystemExit(
            "Enhancement unavailable. Need soxr + .deps/deep-filter (or DEEP_FILTER_BIN)."
        )

    fixtures = make_fixtures()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"fixtures": {}, "peaks_mb": {}}

    for name in ("clean", "noisy"):
        src = fixtures[name]
        dest = OUT / f"{name}_enhanced_16k.wav"
        rss_before = _peak_self_mb()
        t0 = time.perf_counter()
        result = enhance_wav(src, dest)
        wall = time.perf_counter() - t0
        out, rate = read_wav_mono(result.output_path)
        src_audio, _ = read_wav_mono(src)
        corr, lag = best_lag_corr(src_audio, out)
        report["fixtures"][name] = {
            "input": str(src),
            "output": str(result.output_path),
            "sample_rate": rate,
            "wall_s": round(wall, 3),
            "enhance_reported_s": round(result.duration_s, 3),
            "corr_vs_input_lag_aligned": round(corr, 4),
            "best_lag_samples": lag,
            "spectral_flatness_in": round(spectral_flatness(src_audio), 4),
            "spectral_flatness_out": round(spectral_flatness(out), 4),
            "peak_abs_out": round(float(np.max(np.abs(out))), 4),
            "rms_in": round(float(np.sqrt(np.mean(src_audio**2))), 4),
            "rms_out": round(float(np.sqrt(np.mean(out**2))), 4),
        }
        report["peaks_mb"][name] = {
            "self_rss_mb_after": round(_peak_self_mb(), 1),
            "children_rss_mb": round(_peak_rss_mb(), 1),
            "self_rss_mb_before": round(rss_before, 1),
        }

    clean_corr = report["fixtures"]["clean"]["corr_vs_input_lag_aligned"]
    noisy_flat_in = report["fixtures"]["noisy"]["spectral_flatness_in"]
    noisy_flat_out = report["fixtures"]["noisy"]["spectral_flatness_out"]
    report["artifact_checks"] = {
        "clean_corr_vs_input_lag_aligned": clean_corr,
        "clean_corr_ok": clean_corr > 0.7,
        "noisy_flatness_reduced": noisy_flat_out < noisy_flat_in,
        "outputs_non_silent": all(
            report["fixtures"][n]["rms_out"] > 1e-4 for n in ("clean", "noisy")
        ),
        "notes": (
            "Synthetic harmonic tones are not speech; DeepFilterNet is speech-trained. "
            "Use lag-aligned corr + flatness as smoke guards; listen to out/*.wav and "
            "run a real speech clip before READY."
        ),
    }

    report_path = OUT / "spike_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")
    return report


if __name__ == "__main__":
    run()
