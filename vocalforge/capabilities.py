"""Hardware and runtime capability detection."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

REQUIRED_CUDA_LIBS = ("cublas", "cudart")


@dataclass(frozen=True)
class CapabilityReport:
    os_name: str
    architecture: str
    ram_gb: float | None
    nvidia_smi_available: bool
    ctranslate2_cuda_device_count: int
    cuda_runtime_libs_ok: bool
    selected_device: str
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probe_nvidia_smi(run: Callable[..., Any] = subprocess.run) -> bool:
    try:
        run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        logger.info("nvidia-smi unavailable: %s", exc)
        return False


def _probe_ctranslate2_cuda_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception as exc:  # noqa: BLE001
        logger.info("CTranslate2 CUDA probe failed: %s", exc)
        return 0


def _library_candidates(name: str) -> list[str]:
    return [
        f"lib{name}.so",
        f"lib{name}.so.12",
        f"lib{name}.so.11",
        f"{name}.so.12",
        name,
    ]


def _paths_from_ld_library_path() -> list[Path]:
    raw = os.environ.get("LD_LIBRARY_PATH", "")
    return [Path(part) for part in raw.split(":") if part]


def _probe_cuda_runtime_libs(
    find_library: Callable[[str], str | None] = ctypes.util.find_library,
) -> bool:
    """Return True when required CUDA libs are discoverable or loadable."""
    search_dirs = _paths_from_ld_library_path()
    for name in REQUIRED_CUDA_LIBS:
        found = False
        for candidate in (name, f"{name}.so.12", f"lib{name}.so.12"):
            if find_library(candidate):
                found = True
                break
        if not found:
            for directory in search_dirs:
                for filename in _library_candidates(name):
                    path = directory / filename
                    if path.exists():
                        found = True
                        break
                if found:
                    break
        if not found:
            for filename in _library_candidates(name):
                try:
                    ctypes.CDLL(filename)
                    found = True
                    break
                except OSError:
                    continue
        if not found:
            logger.info("CUDA library not found: %s", name)
            return False
    return True


def _probe_ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:  # noqa: BLE001
        return None


def evaluate_capabilities(
    *,
    run: Callable[..., Any] = subprocess.run,
    find_library: Callable[[str], str | None] = ctypes.util.find_library,
    cuda_device_count: Callable[[], int] | None = None,
    prefer_cuda: bool = True,
) -> CapabilityReport:
    """Return a fresh capability report. Prefer CUDA only when fully verified."""
    nvidia_ok = _probe_nvidia_smi(run=run)
    count_fn = cuda_device_count or _probe_ctranslate2_cuda_count
    device_count = count_fn()
    libs_ok = _probe_cuda_runtime_libs(find_library=find_library)

    selected = "cpu"
    reason: str | None = None
    if prefer_cuda and nvidia_ok and device_count > 0 and libs_ok:
        selected = "cuda"
    elif prefer_cuda:
        missing = []
        if not nvidia_ok:
            missing.append("nvidia-smi unavailable")
        if device_count <= 0:
            missing.append("CTranslate2 reports no CUDA devices")
        if not libs_ok:
            missing.append("required CUDA runtime libraries not loadable")
        reason = "; ".join(missing) if missing else "CUDA not usable"
        logger.warning("Falling back to CPU: %s", reason)

    return CapabilityReport(
        os_name=platform.system(),
        architecture=platform.machine(),
        ram_gb=_probe_ram_gb(),
        nvidia_smi_available=nvidia_ok,
        ctranslate2_cuda_device_count=device_count,
        cuda_runtime_libs_ok=libs_ok,
        selected_device=selected,
        fallback_reason=reason,
    )
