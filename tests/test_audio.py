import array
from pathlib import Path

from vocalforge.audio import (
    DEFAULT_INPUT_LABEL,
    AudioRecorder,
    list_input_devices,
    resolve_input_device_index,
    save_wav,
)


def test_save_wav_writes_header_and_samples(tmp_path: Path):
    path = tmp_path / "sample.wav"
    data = array.array("h", [0, 1000, -1000, 0])
    saved = save_wav(path, data, samplerate=16000)
    assert saved.exists()
    assert saved.stat().st_size > 44


def test_list_input_devices_includes_default_and_inputs_only():
    raw = [
        {"name": "Speakers", "max_input_channels": 0},
        {"name": "USB Mic", "max_input_channels": 1},
        {"name": "Headset", "max_input_channels": 2},
    ]
    devices = list_input_devices(query_devices=lambda: raw)
    assert devices[0].name == DEFAULT_INPUT_LABEL
    assert devices[0].index is None
    assert [d.name for d in devices[1:]] == ["USB Mic", "Headset"]
    assert devices[1].index == 1
    assert devices[2].index == 2


def test_resolve_input_device_index_matches_saved_name():
    raw = [
        {"name": "Speakers", "max_input_channels": 0},
        {"name": "USB Mic", "max_input_channels": 1},
    ]
    devices = list_input_devices(query_devices=lambda: raw)
    assert resolve_input_device_index(None, devices) is None
    assert resolve_input_device_index(DEFAULT_INPUT_LABEL, devices) is None
    assert resolve_input_device_index("USB Mic", devices) == 1
    assert resolve_input_device_index("Missing Mic", devices) is None


def test_recorder_set_device_updates_selection():
    recorder = AudioRecorder(device=None)
    recorder.set_device(2)
    assert recorder.device == 2
    recorder.set_device(None)
    assert recorder.device is None
