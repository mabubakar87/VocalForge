from pathlib import Path

from vocalforge.config import AppConfig, load_config, save_config


def test_missing_config_returns_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.json")
    assert cfg.auto_paste is False
    assert cfg.selected_model == ""


def test_round_trip(tmp_path: Path):
    path = tmp_path / "config.json"
    save_config(path, AppConfig(selected_model="distil-small.en", auto_paste=False))
    loaded = load_config(path)
    assert loaded.selected_model == "distil-small.en"
    assert loaded.auto_paste is False


def test_invalid_json_uses_defaults(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text("{not-json", encoding="utf-8")
    cfg = load_config(path)
    assert isinstance(cfg, AppConfig)
    assert cfg.selected_model == ""
