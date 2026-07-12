from pathlib import Path

from vocalforge.config import AppConfig, load_config, save_config


def test_missing_config_returns_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.json")
    assert cfg.auto_paste is False
    assert cfg.selected_model == ""


def test_round_trip(tmp_path: Path):
    path = tmp_path / "config.json"
    save_config(
        path,
        AppConfig(
            selected_model="distil-small.en",
            active_profile="lightweight",
            preferred_device="cpu",
            vad_filter=False,
            beam_size=10,
            word_timestamps=True,
            enhance_audio=True,
            auto_paste=False,
        ),
    )
    loaded = load_config(path)
    assert loaded.selected_model == "distil-small.en"
    assert loaded.active_profile == "lightweight"
    assert loaded.preferred_device == "cpu"
    assert loaded.vad_filter is False
    assert loaded.beam_size == 10
    assert loaded.word_timestamps is True
    assert loaded.enhance_audio is True
    assert loaded.auto_paste is False
    assert loaded.schema_version == 2


def test_old_config_upgrades(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text('{"schema_version": 1, "selected_model": "distil-small.en"}', encoding="utf-8")
    loaded = load_config(path)
    assert loaded.selected_model == "distil-small.en"
    assert loaded.active_profile == ""
    assert loaded.schema_version == 2


def test_invalid_json_uses_defaults(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text("{not-json", encoding="utf-8")
    cfg = load_config(path)
    assert isinstance(cfg, AppConfig)
    assert cfg.selected_model == ""


def test_save_config_leaves_no_tmp_files(tmp_path: Path):
    path = tmp_path / "config.json"
    save_config(path, AppConfig(selected_model="distil-small.en"))
    leftovers = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob(".config-*.tmp"))
    assert leftovers == []
    assert path.exists()
