"""Optional speaker diarization via pyannote community pipeline.

Loads the pipeline from a **vendored local** tree under Models/diarization/
(offline — no Hugging Face HEAD checks). Failures raise DiarizationError;
callers must catch and continue with unlabeled ASR.

Audio is preloaded as a waveform dict so pyannote does not need torchcodec.
"""

from __future__ import annotations

import gc
import logging
import os
import warnings
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

COMMUNITY_PIPELINE_ID = "pyannote/speaker-diarization-community-1"
LOCAL_PIPELINE_REL = Path("diarization") / "pyannote_community_1"

# Files that must exist for offline load (community-1 layout).
_REQUIRED_RELATIVE = (
    Path("config.yaml"),
    Path("segmentation") / "pytorch_model.bin",
    Path("embedding") / "pytorch_model.bin",
    Path("plda") / "plda.npz",
    Path("plda") / "xvec_transform.npz",
)


class DiarizationError(RuntimeError):
    """Diarization unavailable or failed."""


def resolve_hf_token(explicit: str | None = None) -> str | None:
    """Return a Hugging Face token from argument or environment (never log the value)."""
    for candidate in (
        explicit,
        os.environ.get("HF_TOKEN"),
        os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return None


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_local_pipeline_dir(models_root: Path | None = None) -> Path:
    """Return Models/diarization/pyannote_community_1 (or VOCALFORGE_DIARIZATION_MODEL_DIR)."""
    override = os.environ.get("VOCALFORGE_DIARIZATION_MODEL_DIR")
    if override and override.strip():
        return Path(override).expanduser().resolve()
    root = Path(models_root) if models_root is not None else project_root() / "Models"
    return (root / LOCAL_PIPELINE_REL).resolve()


def local_pipeline_config(model_dir: Path | None = None) -> Path:
    return (model_dir or default_local_pipeline_dir()) / "config.yaml"


def local_pipeline_ready(model_dir: Path | None = None) -> bool:
    """True when the vendored offline tree has config + weight files."""
    base = model_dir or default_local_pipeline_dir()
    return all((base / rel).is_file() for rel in _REQUIRED_RELATIVE)


def missing_local_assets(model_dir: Path | None = None) -> list[str]:
    base = model_dir or default_local_pipeline_dir()
    return [str(rel) for rel in _REQUIRED_RELATIVE if not (base / rel).is_file()]


def is_pyannote_importable() -> bool:
    try:
        importlib_import_pyannote()
    except Exception:  # noqa: BLE001
        return False
    return True


def importlib_import_pyannote() -> Any:
    # torchcodec often fails on CPU-only torch (missing libnvrtc). We preload WAVs
    # ourselves, so the import-time UserWarning is noise — especially at app startup.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"pyannote\.audio\.core\.io",
        )
        from pyannote.audio import Pipeline  # noqa: F401

    return Pipeline


def diarization_cache_likely_present() -> bool:
    """HF hub cache still counts as a source for the one-time vendor script."""
    home = Path.home()
    candidates = [
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--pyannote--speaker-diarization-community-1",
        home / ".cache" / "torch" / "pyannote",
        home / ".cache" / "pyannote",
    ]
    for path in candidates:
        if path.is_dir() and any(path.rglob("*")):
            return True
    return False


def is_diarization_ready(hf_token: str | None = None, *, models_root: Path | None = None) -> bool:
    """True when pyannote imports and the offline vendored pipeline is present."""
    del hf_token  # Token is only needed for the one-time vendor download.
    if not is_pyannote_importable():
        return False
    return local_pipeline_ready(default_local_pipeline_dir(models_root))


@contextmanager
def _hf_offline_mode() -> Iterator[None]:
    """Force Hugging Face libraries to skip network for this load only."""
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_wav_as_pyannote_dict(path: Path) -> dict[str, Any]:
    """Load PCM WAV with stdlib + torch — avoids pyannote's broken torchcodec path.

    Returns {"waveform": Tensor[channel, time], "sample_rate": int}.
    """
    try:
        import torch
    except ImportError as exc:
        raise DiarizationError("torch is required for diarization.") from exc

    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except wave.Error as exc:
        raise DiarizationError(f"Could not read WAV file: {path}") from exc

    if width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise DiarizationError(f"Unsupported WAV sample width: {width}")

    if channels > 1:
        samples = samples.reshape(-1, channels).T  # (channel, time)
    else:
        samples = samples.reshape(1, -1)

    waveform = torch.from_numpy(np.ascontiguousarray(samples))
    return {"waveform": waveform, "sample_rate": int(rate)}


def _load_pipeline(device: str, *, model_dir: Path | None = None) -> Any:
    """Instantiate from local config.yaml only (strict offline)."""
    Pipeline = importlib_import_pyannote()
    base = model_dir or default_local_pipeline_dir()
    config_path = base / "config.yaml"
    if not local_pipeline_ready(base):
        missing = ", ".join(missing_local_assets(base)) or "(unknown)"
        raise DiarizationError(
            f"Offline diarization weights missing under {base}. "
            f"Missing: {missing}. "
            "Run: PYTHONPATH=. python scripts/vendor_diarization_models.py"
        )

    try:
        with _hf_offline_mode():
            # Path to local config.yaml — $model/* resolves next to this file.
            pipeline = Pipeline.from_pretrained(str(config_path))
    except Exception as exc:  # noqa: BLE001
        raise DiarizationError(
            f"Failed to load local pyannote pipeline from {config_path}. "
            "Re-run scripts/vendor_diarization_models.py."
        ) from exc

    logger.info("Loaded diarization pipeline from local path %s (offline).", config_path)

    try:
        import torch

        target = torch.device(device if device in {"cpu", "cuda"} else "cpu")
        if target.type == "cuda" and not torch.cuda.is_available():
            target = torch.device("cpu")
        if hasattr(pipeline, "to"):
            pipeline.to(target)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not move diarization pipeline to %s (%s); using default.", device, exc)
    return pipeline


def _annotation_to_turns(annotation: Any) -> list[dict[str, float | str]]:
    """Normalize pyannote Annotation / DiarizeOutput into [{start, end, speaker}]."""
    if annotation is None:
        return []
    nested = getattr(annotation, "speaker_diarization", None)
    if nested is not None:
        annotation = nested
    if not hasattr(annotation, "itertracks"):
        raise DiarizationError(f"Unexpected diarization output type: {type(annotation)!r}")
    turns: list[dict[str, float | str]] = []
    for turn, _track, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    turns.sort(key=lambda item: float(item["start"]))
    return turns


def _release_pipeline(pipeline: Any) -> None:
    try:
        del pipeline
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def resolve_diarization_device(preferred: str = "auto") -> str:
    """Pick cuda when PyTorch CUDA is available; otherwise cpu.

    Note: Faster-Whisper can use CUDA via CTranslate2 even when the installed
    torch wheel is CPU-only — diarization needs a CUDA torch build to use GPU.
    """
    choice = (preferred or "auto").strip().lower()
    if choice == "cpu":
        return "cpu"
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_ok = False
    if choice == "cuda":
        if cuda_ok:
            return "cuda"
        logger.warning(
            "Diarization requested CUDA but this torch build has no GPU "
            "(install a CUDA wheel from pytorch.org). Falling back to CPU."
        )
        return "cpu"
    # auto
    if cuda_ok:
        return "cuda"
    logger.info(
        "Diarization using CPU (torch has no CUDA). "
        "Install CUDA torch to accelerate long files — see requirements-extras.txt."
    )
    return "cpu"


def diarize_wav(
    file_path: str | Path,
    hf_token: str | None = None,
    *,
    device: str = "auto",
    model_dir: Path | None = None,
    models_root: Path | None = None,
) -> list[dict[str, float | str]]:
    """Run community diarization and return [{start, end, speaker}, ...].

    device: ``auto`` (CUDA if torch CUDA is available, else CPU), ``cuda``, or ``cpu``.
    Always releases the pipeline before returning. Uses vendored local weights only.
    """
    del hf_token  # Kept for call-site compatibility; runtime load is offline-only.
    path = Path(file_path)
    if not path.is_file():
        raise DiarizationError(f"Audio file not found: {path}")

    if not is_pyannote_importable():
        raise DiarizationError(
            "pyannote.audio is not installed. See requirements-extras.txt (Diarization Extra)."
        )

    resolved_dir = model_dir
    if resolved_dir is None and models_root is not None:
        resolved_dir = default_local_pipeline_dir(models_root)
    if resolved_dir is None:
        resolved_dir = default_local_pipeline_dir()

    resolved_device = resolve_diarization_device(device)
    pipeline = None
    try:
        audio = _load_wav_as_pyannote_dict(path)
        pipeline = _load_pipeline(device=resolved_device, model_dir=resolved_dir)
        logger.info("Running diarization on %s (device=%s).", path, resolved_device)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\..*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\..*")
            annotation = pipeline(audio)
        return _annotation_to_turns(annotation)
    except DiarizationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DiarizationError(f"Diarization failed: {exc}") from exc
    finally:
        if pipeline is not None:
            _release_pipeline(pipeline)
