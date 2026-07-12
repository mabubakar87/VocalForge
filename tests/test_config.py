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
            auto_paste=False,
        ),
    )
    loaded = load_config(path)
    assert loaded.selected_model == "distil-small.en"
    assert loaded.active_profile == "lightweight"
    assert loaded.preferred_device == "cpu"
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
