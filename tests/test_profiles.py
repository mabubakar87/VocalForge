from pathlib import Path

from vocalforge.capabilities import CapabilityReport
from vocalforge.profiles import (
    PROFILE_ORDER,
    PROFILES,
    has_enough_disk_for_profile,
    is_model_available_locally,
    language_mode_from_config,
    language_settings_for_mode,
    recommend_profile,
    resolve_language_settings,
    resolve_runtime,
)


def _caps(device: str = "cpu", **kwargs) -> CapabilityReport:
    base = dict(
        os_name="Linux",
        architecture="x86_64",
        ram_gb=16.0,
        selected_device=device,
        nvidia_smi_available=device == "cuda",
        ctranslate2_cuda_device_count=1 if device == "cuda" else 0,
        cuda_runtime_libs_ok=device == "cuda",
    )
    base.update(kwargs)
    return CapabilityReport(**base)


def test_recommend_cpu_is_lightweight():
    rec = recommend_profile(_caps("cpu"), vram_gb=None)
    assert rec.profile_id == "lightweight"


def test_recommend_cuda_unknown_vram_is_balanced():
    rec = recommend_profile(_caps("cuda"), vram_gb=None)
    assert rec.profile_id == "balanced"


def test_recommend_cuda_high_vram_is_high_accuracy():
    rec = recommend_profile(_caps("cuda"), vram_gb=8.0)
    assert rec.profile_id == "high_accuracy"


def test_recommend_cuda_low_vram_is_balanced():
    rec = recommend_profile(_caps("cuda"), vram_gb=4.0)
    assert rec.profile_id == "balanced"


def test_resolve_runtime_cpu_profile():
    runtime = resolve_runtime(PROFILES["balanced"], _caps("cpu"))
    assert runtime.device == "cpu"
    assert runtime.compute_type == "int8"


def test_resolve_runtime_honors_preferred_cpu_on_cuda_host():
    runtime = resolve_runtime(
        PROFILES["balanced"], _caps("cuda"), preferred_device="cpu"
    )
    assert runtime.device == "cpu"


def test_resolve_runtime_cuda_when_usable():
    runtime = resolve_runtime(
        PROFILES["lightweight"], _caps("cuda"), preferred_device="cuda"
    )
    assert runtime.device == "cuda"
    assert runtime.compute_type == PROFILES["lightweight"].compute_type_cuda


def test_resolve_runtime_falls_back_when_cuda_not_usable():
    runtime = resolve_runtime(
        PROFILES["balanced"], _caps("cpu"), preferred_device="cuda"
    )
    assert runtime.device == "cpu"


def test_model_missing_locally(tmp_path: Path):
    assert is_model_available_locally(tmp_path, PROFILES["lightweight"]) is False


def test_model_available_with_snapshot(tmp_path: Path):
    profile = PROFILES["lightweight"]
    snap = tmp_path / profile.cache_dirname / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"x")
    (snap / "config.json").write_text("{}", encoding="utf-8")
    assert is_model_available_locally(tmp_path, profile) is True


def test_high_accuracy_detects_mobiuslabs_cache(tmp_path: Path):
    profile = PROFILES["high_accuracy"]
    snap = tmp_path / profile.cache_dirname / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"x")
    (snap / "config.json").write_text("{}", encoding="utf-8")
    assert is_model_available_locally(tmp_path, profile) is True


def test_disk_check(tmp_path: Path):
    assert has_enough_disk_for_profile(tmp_path, PROFILES["lightweight"]) is True


def test_english_only_profiles_lock_language():
    language, task = resolve_language_settings(
        PROFILES["balanced"], language="ur", task="translate"
    )
    assert language == "en"
    assert task == "transcribe"


def test_multilingual_small_supports_translate():
    language, task = resolve_language_settings(
        PROFILES["multilingual_small"], language="zh", task="translate"
    )
    assert language == "zh"
    assert task == "translate"
    assert PROFILES["multilingual_small"].model == "small"
    assert PROFILES["multilingual_small"].supports_translate is True
    assert PROFILE_ORDER == (
        "lightweight",
        "multilingual_small",
        "balanced",
        "multilingual_medium",
        "high_accuracy",
    )


def test_multilingual_medium_supports_translate():
    language, task = resolve_language_settings(
        PROFILES["multilingual_medium"], language="zh", task="translate"
    )
    assert language == "zh"
    assert task == "translate"
    assert PROFILES["multilingual_medium"].model == "medium"
    assert PROFILES["multilingual_medium"].supports_translate is True


def test_high_accuracy_language_modes():
    assert language_settings_for_mode("en") == ("en", "transcribe")
    assert language_settings_for_mode("ur") == ("ur", "transcribe")
    # Translate task must not remap the language dropdown selection.
    assert language_mode_from_config(None, "translate") == "auto"
    assert language_mode_from_config("ur", "translate") == "ur"
    assert language_mode_from_config("ur", "transcribe") == "ur"
    language, task = resolve_language_settings(
        PROFILES["high_accuracy"], language="ur", task="transcribe"
    )
    assert language == "ur"
    assert task == "transcribe"
    language, task = resolve_language_settings(
        PROFILES["high_accuracy"], language="zh", task="translate"
    )
    assert language == "zh"
    # turbo does not support translate — forced back to transcribe
    assert task == "transcribe"
    language, task = resolve_language_settings(
        PROFILES["high_accuracy"], language=None, task="translate"
    )
    assert language is None
    assert task == "transcribe"
