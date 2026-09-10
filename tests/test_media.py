from pathlib import Path

import numpy as np
import pytest

from vocalforge.media import (
    ASR_SAMPLE_RATE,
    MediaError,
    converted_wav_path,
    decode_to_wav,
    ensure_wav,
    is_supported_upload,
    is_wav_path,
    write_wav_mono_pcm16,
)


def test_supported_upload_extensions():
    assert is_supported_upload("a.wav")
    assert is_supported_upload("a.MP3")
    assert is_supported_upload(Path("clip.m4a"))
    assert is_supported_upload("talk.flac")
    assert is_supported_upload("clip.mp4")
    assert not is_supported_upload("notes.txt")
    assert not is_supported_upload("image.png")


def test_is_wav_path():
    assert is_wav_path("a.wav")
    assert is_wav_path("A.WAV")
    assert not is_wav_path("a.mp4")


def _write_tone_wav(path: Path, *, seconds: float = 0.25, rate: int = 16_000) -> Path:
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    samples = 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    return write_wav_mono_pcm16(path, samples, rate)


def test_ensure_wav_passthrough(tmp_path: Path):
    wav = _write_tone_wav(tmp_path / "clip.wav")
    assert ensure_wav(wav, tmp_path) == wav.resolve()


def test_decode_to_wav_from_wav_via_pyav(tmp_path: Path):
    src = _write_tone_wav(tmp_path / "src.wav", seconds=0.5, rate=22_050)
    out = tmp_path / "out.wav"
    result = decode_to_wav(src, out, sample_rate=ASR_SAMPLE_RATE)
    assert result == out
    assert out.is_file()
    import wave

    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == ASR_SAMPLE_RATE
        assert wf.getnframes() > 0


def test_ensure_wav_converts_non_wav(tmp_path: Path):
    """PyAV can re-open WAV as a container; rename to .mp3-like stem via decode path."""
    # Use a real non-wav container if we can mux; otherwise decode_to_wav on WAV
    # covered above. Here ensure_wav with a .flac produced by PyAV if available.
    av = pytest.importorskip("av")
    src_wav = _write_tone_wav(tmp_path / "tone.wav", seconds=0.3)
    flac_path = tmp_path / "tone.flac"
    try:
        inp = av.open(str(src_wav))
        out = av.open(str(flac_path), mode="w", format="flac")
        in_stream = inp.streams.audio[0]
        out_stream = out.add_stream("flac", rate=in_stream.rate)
        for frame in inp.decode(audio=0):
            frame.pts = None
            for packet in out_stream.encode(frame):
                out.mux(packet)
        for packet in out_stream.encode(None):
            out.mux(packet)
        out.close()
        inp.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not mux FLAC for test: {exc}")

    assert flac_path.is_file()
    converted = ensure_wav(flac_path, tmp_path)
    assert converted == converted_wav_path(tmp_path, flac_path)
    assert converted.is_file()
    assert converted.suffix.lower() == ".wav"


def test_decode_missing_file(tmp_path: Path):
    with pytest.raises(MediaError, match="not found"):
        decode_to_wav(tmp_path / "missing.mp4", tmp_path / "out.wav")
