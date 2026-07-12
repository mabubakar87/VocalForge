"""Faster-Whisper execution profiles and local model readiness."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from vocalforge.capabilities import CapabilityReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfileSpec:
    id: str
    label: str
    model: str
    preferred_device: str
    compute_type_cuda: str
    compute_type_cpu: str
    approx_size_bytes: int
    size_label: str
    cache_dirname: str
    description: str
    english_only: bool = False
    # Hugging Face may redirect model ids; keep known local cache folder aliases.
    cache_dirnames: tuple[str, ...] = ()


PROFILES: dict[str, ProfileSpec] = {
    "lightweight": ProfileSpec(
        id="lightweight",
        label="Lightweight",
        model="distil-small.en",
        preferred_device="cpu",
        compute_type_cuda="int8",
        compute_type_cpu="int8",
        approx_size_bytes=160 * 1024 * 1024,
        size_label="~160 MB",
        cache_dirname="models--Systran--faster-distil-whisper-small.en",
        description="Fast CPU transcription with a small English model.",
        english_only=True,
    ),
    "balanced": ProfileSpec(
        id="balanced",
        label="Balanced",
        model="distil-medium.en",
        preferred_device="cuda",
        compute_type_cuda="int8_float16",
        compute_type_cpu="int8",
        approx_size_bytes=800 * 1024 * 1024,
        size_label="~800 MB",
        cache_dirname="models--Systran--faster-distil-whisper-medium.en",
        description="Good accuracy on CUDA when available; CPU fallback supported.",
        english_only=True,
    ),
    "high_accuracy": ProfileSpec(
        id="high_accuracy",
        label="High Accuracy",
        model="large-v3-turbo",
        preferred_device="cuda",
        compute_type_cuda="float16",
        compute_type_cpu="int8",
        approx_size_bytes=1600 * 1024 * 1024,
        size_label="~1.6 GB",
        # faster-whisper resolves large-v3-turbo to mobiuslabsgmbh (HF may redirect further).
        cache_dirname="models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
        description="Highest quality Faster-Whisper model in this release.",
        english_only=False,
        cache_dirnames=(
            "models--Systran--faster-whisper-large-v3-turbo",
            "models--dropbox-dash--faster-whisper-large-v3-turbo",
        ),
    ),
}

PROFILE_ORDER = ("lightweight", "balanced", "high_accuracy")

# High Accuracy language modes: (id, label, whisper_language|None, task)
LANGUAGE_OPTIONS: tuple[tuple[str, str, str | None, str], ...] = (
    ("auto", "Auto-detect", None, "transcribe"),
    ("en", "English", "en", "transcribe"),
    ("ur", "Urdu", "ur", "transcribe"),
    ("hi", "Hindi", "hi", "transcribe"),
    ("ar", "Arabic", "ar", "transcribe"),
    ("fr", "French", "fr", "transcribe"),
    ("de", "German", "de", "transcribe"),
    ("es", "Spanish", "es", "transcribe"),
    ("zh", "Chinese", "zh", "transcribe"),
    ("ja", "Japanese", "ja", "transcribe"),
)

LANGUAGE_OPTION_BY_ID = {item[0]: item for item in LANGUAGE_OPTIONS}


@dataclass(frozen=True)
class ProfileRecommendation:
    profile_id: str
    reason: str


@dataclass(frozen=True)
class ProfileRuntime:
    profile: ProfileSpec
    device: str
    compute_type: str


def probe_vram_gb(run: Callable[..., Any] = subprocess.run) -> float | None:
    """Best-effort VRAM total in GiB via nvidia-smi (no PyTorch)."""
    try:
        completed = run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        first = completed.stdout.strip().splitlines()[0].strip()
        # Values are MiB from nvidia-smi.
        return round(float(first) / 1024.0, 1)
    except Exception as exc:  # noqa: BLE001
        logger.info("VRAM probe failed: %s", exc)
        return None


def recommend_profile(
    capabilities: CapabilityReport,
    vram_gb: float | None = None,
) -> ProfileRecommendation:
    if capabilities.selected_device != "cuda":
        return ProfileRecommendation(
            "lightweight",
            "CUDA is unavailable or incomplete; Lightweight (CPU) is recommended.",
        )
    if vram_gb is None:
        return ProfileRecommendation(
            "balanced",
            "CUDA is available but VRAM could not be measured; Balanced is recommended.",
        )
    if vram_gb >= 6:
        return ProfileRecommendation(
            "high_accuracy",
            f"CUDA with ~{vram_gb} GB VRAM detected; High Accuracy is recommended.",
        )
    return ProfileRecommendation(
        "balanced",
        f"CUDA with ~{vram_gb} GB VRAM detected; Balanced is recommended.",
    )


def resolve_runtime(
    profile: ProfileSpec,
    capabilities: CapabilityReport,
    preferred_device: str | None = None,
) -> ProfileRuntime:
    """Choose device/compute for a profile.

    ``preferred_device`` is a user override (``cpu`` / ``cuda``). CUDA is used
    only when the override asks for it and CUDA is actually usable.
    """
    cuda_ok = (
        capabilities.nvidia_smi_available
        and capabilities.ctranslate2_cuda_device_count > 0
        and capabilities.cuda_runtime_libs_ok
    )
    requested = (preferred_device or profile.preferred_device or "cpu").lower()
    if requested == "cuda" and cuda_ok:
        return ProfileRuntime(profile, "cuda", profile.compute_type_cuda)
    return ProfileRuntime(profile, "cpu", profile.compute_type_cpu)


def cuda_is_usable(capabilities: CapabilityReport) -> bool:
    return (
        capabilities.nvidia_smi_available
        and capabilities.ctranslate2_cuda_device_count > 0
        and capabilities.cuda_runtime_libs_ok
    )


def is_model_available_locally(models_dir: Path, profile: ProfileSpec) -> bool:
    """Return True when a usable Faster-Whisper cache snapshot exists."""
    candidates = (profile.cache_dirname, *profile.cache_dirnames)
    for dirname in candidates:
        cache_root = Path(models_dir) / dirname
        if not cache_root.is_dir():
            continue
        snapshots = cache_root / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir():
                continue
            if (snapshot / "model.bin").exists() and (snapshot / "config.json").exists():
                return True
    return False


def free_disk_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def has_enough_disk_for_profile(path: Path, profile: ProfileSpec, margin: float = 1.2) -> bool:
    needed = int(profile.approx_size_bytes * margin)
    return free_disk_bytes(path) >= needed


def profile_display_models() -> list[str]:
    """Dropdown-compatible labels derived from profiles plus empty selection."""
    labels = [""]
    for profile_id in PROFILE_ORDER:
        profile = PROFILES[profile_id]
        labels.append(f"{profile.model} ({profile.size_label})")
    return labels


def profile_for_model(model_name: str) -> ProfileSpec | None:
    for profile in PROFILES.values():
        if profile.model == model_name:
            return profile
    return None


def parse_model_name(display: str) -> str:
    return display.split(" (")[0].strip() if display else ""


def language_mode_from_config(language: str | None, task: str) -> str:
    """Map saved language/task settings to a LANGUAGE_OPTIONS id."""
    # Obsolete "translate" task maps to English (works better on large-v3-turbo).
    if (task or "").lower() == "translate":
        return "en"
    if language:
        code = language.lower()
        if code in LANGUAGE_OPTION_BY_ID:
            return code
    return "auto"


def language_settings_for_mode(mode_id: str) -> tuple[str | None, str]:
    """Return (whisper_language, task) for a High Accuracy language mode id."""
    option = LANGUAGE_OPTION_BY_ID.get(mode_id) or LANGUAGE_OPTION_BY_ID["auto"]
    return option[2], option[3]


def resolve_language_settings(
    profile: ProfileSpec,
    *,
    language: str | None,
    task: str,
) -> tuple[str | None, str]:
    """Effective language/task for a profile (English-only models are locked)."""
    if profile.english_only:
        return "en", "transcribe"
    # Migrate obsolete translate task to forced English transcription.
    if (task or "").lower() == "translate":
        return "en", "transcribe"
    return language, "transcribe"


def language_option_label(mode_id: str) -> str:
    option = LANGUAGE_OPTION_BY_ID.get(mode_id)
    return option[1] if option else "Auto-detect"
