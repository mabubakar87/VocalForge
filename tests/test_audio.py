import array
from pathlib import Path

from vocalforge.audio import save_wav


def test_save_wav_writes_header_and_samples(tmp_path: Path):
    path = tmp_path / "sample.wav"
    data = array.array("h", [0, 1000, -1000, 0])
    saved = save_wav(path, data, samplerate=16000)
    assert saved.exists()
    assert saved.stat().st_size > 44
