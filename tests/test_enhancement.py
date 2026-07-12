"""Unit tests for enhancement resampling + binary resolution (no deep-filter required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vocalforge.enhancement import (
    ASR_RATE,
    DF_RATE,
    EnhancementError,
    is_enhancement_available,
    read_wav_mono,
    resample_audio,
    resolve_deep_filter_binary,
    write_wav_mono,
)


def test_resample_roundtrip_energy():
    t = np.linspace(0, 1.0, ASR_RATE, endpoint=False, dtype=np.float32)
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    up = resample_audio(tone, ASR_RATE, DF_RATE)
    assert up.shape[0] == DF_RATE
    down = resample_audio(up, DF_RATE, ASR_RATE)
    assert down.shape[0] == ASR_RATE
    # Correlation should stay high for a pure tone.
    corr = float(np.corrcoef(tone, down)[0, 1])
    assert corr > 0.98


def test_wav_roundtrip(tmp_path: Path):
    t = np.linspace(0, 0.25, ASR_RATE // 4, endpoint=False, dtype=np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    write_wav_mono(path, tone, ASR_RATE)
    loaded, rate = read_wav_mono(path)
    assert rate == ASR_RATE
    assert loaded.shape == tone.shape
    assert float(np.max(np.abs(loaded - tone))) < 2 / 32768


def test_resolve_binary_explicit(tmp_path: Path):
    fake = tmp_path / "deep-filter"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    assert resolve_deep_filter_binary(fake) == fake
    missing = tmp_path / "missing"
    assert resolve_deep_filter_binary(missing) is None


def test_enhance_requires_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from vocalforge import enhancement as mod

    monkeypatch.setattr(mod, "resolve_deep_filter_binary", lambda *_a, **_k: None)
    tone = np.zeros(ASR_RATE, dtype=np.float32)
    path = tmp_path / "x.wav"
    write_wav_mono(path, tone, ASR_RATE)
    with pytest.raises(EnhancementError, match="deep-filter"):
        mod.enhance_wav(path, tmp_path / "out.wav")


@pytest.mark.skipif(not is_enhancement_available(), reason="deep-filter + soxr not available")
def test_enhance_wav_smoke(tmp_path: Path):
    from vocalforge.enhancement import enhance_wav

    t = np.linspace(0, 1.0, ASR_RATE, endpoint=False, dtype=np.float32)
    speechish = (0.15 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    noise = (0.05 * np.random.default_rng(0).standard_normal(ASR_RATE)).astype(np.float32)
    path = tmp_path / "noisy.wav"
    write_wav_mono(path, speechish + noise, ASR_RATE)
    result = enhance_wav(path, tmp_path / "enhanced.wav")
    assert result.output_path.exists()
    out, rate = read_wav_mono(result.output_path)
    assert rate == ASR_RATE
    assert out.size > 0
    assert result.backend == "deep-filter"
