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


def _find_library_query_names(name: str) -> list[str]:
    """Names to pass to ctypes.util.find_library (Linux .so and Windows DLL stems)."""
    return [
        name,
        f"lib{name}",
        f"{name}.so.12",
        f"lib{name}.so.12",
        f"{name}64_13",
        f"{name}64_12",
        f"{name}64_11",
    ]


def _library_candidates(name: str) -> list[str]:
    """On-disk filenames for a CUDA lib stem on Linux and Windows."""
    return [
        f"lib{name}.so",
        f"lib{name}.so.13",
        f"lib{name}.so.12",
        f"lib{name}.so.11",
        f"{name}.so.12",
        name,
        f"{name}64_13.dll",
        f"{name}64_12.dll",
        f"{name}64_11.dll",
        f"{name}.dll",
    ]


def _cuda_library_search_dirs() -> list[Path]:
    """Directories that may contain CUDA runtime libraries on Linux or Windows."""
    dirs: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            dirs.append(path)

    ld_sep = os.pathsep if os.name == "nt" else ":"
    for part in os.environ.get("LD_LIBRARY_PATH", "").split(ld_sep):
        if part:
            add(Path(part))

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        root = Path(cuda_path)
        for sub in ("bin", "lib64", "lib/x64", "lib"):
            add(root / sub)

    for part in os.environ.get("PATH", "").split(os.pathsep):
        if part:
            add(Path(part))

    return dirs


def _probe_cuda_runtime_libs(
    find_library: Callable[[str], str | None] = ctypes.util.find_library,
) -> bool:
    """Return True when required CUDA libs are discoverable or loadable."""
    search_dirs = _cuda_library_search_dirs()
    for name in REQUIRED_CUDA_LIBS:
        found = False
        for candidate in _find_library_query_names(name):
            if find_library(candidate):
                found = True
                break
        if not found:
            for directory in search_dirs:
                for filename in _library_candidates(name):
                    path = directory / filename
                    if path.is_file():
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
