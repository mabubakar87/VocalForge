#!/usr/bin/env python3
"""One-time vendor of pyannote community-1 weights into Models/ (offline runtime).

Copies from the Hugging Face hub cache when present, otherwise downloads once
with HF_TOKEN, then writes:

  Models/diarization/pyannote_community_1/
    config.yaml
    segmentation/pytorch_model.bin
    embedding/pytorch_model.bin
    plda/plda.npz
    plda/xvec_transform.npz

After this succeeds, VocalForge loads diarization with HF_HUB_OFFLINE=1 and
never HEAD-checks huggingface.co at runtime.

Usage:
  export HF_TOKEN=hf_xxx   # only needed if hub cache is empty
  PYTHONPATH=. python scripts/vendor_diarization_models.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vocalforge.config import load_config
from vocalforge.diarization import (
    COMMUNITY_PIPELINE_ID,
    default_local_pipeline_dir,
    local_pipeline_ready,
    missing_local_assets,
    resolve_hf_token,
)

HUB_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--pyannote--speaker-diarization-community-1"
)


def _find_hub_snapshot() -> Path | None:
    snaps = HUB_CACHE / "snapshots"
    if not snaps.is_dir():
        return None
    candidates = sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in candidates:
        if (snap / "config.yaml").is_file():
            return snap
    return None


def _copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    # Copy real files (resolve hub snapshot symlinks into blobs).
    shutil.copytree(src, dest, symlinks=False, dirs_exist_ok=False)


def _download_via_hub(token: str | None, dest: Path) -> None:
    """Download community-1 into HF cache, then copy into dest."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {COMMUNITY_PIPELINE_ID} (one-time)…")
    kwargs: dict = {"repo_id": COMMUNITY_PIPELINE_ID, "repo_type": "model"}
    if token:
        kwargs["token"] = token
    cached = Path(snapshot_download(**kwargs))
    print(f"Hub cache snapshot: {cached}")
    _copy_tree(cached, dest)


def main() -> int:
    models_root = ROOT / "Models"
    dest = default_local_pipeline_dir(models_root)
    token = resolve_hf_token(None)
    if not token:
        cfg = load_config(ROOT / "config.json")
        token = resolve_hf_token(cfg.hf_token)

    print("Target:", dest)

    if local_pipeline_ready(dest):
        print("Already vendored and complete. Offline diarization is ready.")
        return 0

    snap = _find_hub_snapshot()
    try:
        if snap is not None:
            print(f"Copying from local HF cache snapshot:\n  {snap}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            _copy_tree(snap, dest)
        else:
            if not token:
                print(
                    "ERROR: No HF cache and no HF_TOKEN / config.hf_token.\n"
                    "Accept gated model terms, set a read token, then re-run."
                )
                return 1
            dest.parent.mkdir(parents=True, exist_ok=True)
            _download_via_hub(token, dest)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: vendor failed: {exc}")
        return 1

    if not local_pipeline_ready(dest):
        print("ERROR: vendor incomplete. Missing:", ", ".join(missing_local_assets(dest)))
        return 1

    # community-1 config already uses $model/... relative paths — leave as-is.
    print("Vendored OK. Runtime will load:")
    print(f"  {dest / 'config.yaml'}")
    print("No Hugging Face network calls are needed for diarization after this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
