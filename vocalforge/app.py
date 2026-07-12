"""Application entry helpers."""

from __future__ import annotations

import logging
import threading
import tkinter as tk

from vocalforge.capabilities import evaluate_capabilities
from vocalforge.clipboard import ClipboardSettings
from vocalforge.config import load_config, save_config
from vocalforge.jobs import JobController
from vocalforge.profiles import probe_vram_gb
from vocalforge.storage import default_paths
from vocalforge.transcription import TranscriptionService
from vocalforge.ui import create_app


def configure_logging(log_file) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )


def try_start_hotkey(controller: JobController) -> None:
    """Best-effort global hotkey. Never fatal; Linux often requires elevated privileges."""

    def listener() -> None:
        try:
            import keyboard

            keyboard.add_hotkey("ctrl+q", controller.toggle_recording)
            keyboard.wait()
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Global hotkey unavailable (use the record button instead): %s",
                exc,
            )

    threading.Thread(target=listener, daemon=True).start()


def run() -> None:
    paths = default_paths()
    configure_logging(paths.log_file)
    logging.info("Application started.")

    capabilities = evaluate_capabilities()
    vram_gb = probe_vram_gb()
    logging.info(
        "Using device: %s%s",
        capabilities.selected_device,
        f" ({capabilities.fallback_reason})" if capabilities.fallback_reason else "",
    )
    if vram_gb is not None:
        logging.info("Detected VRAM: %.1f GB", vram_gb)

    config_path = paths.root / "config.json"
    config = load_config(config_path)
    save_config(config_path, config)

    transcription = TranscriptionService(
        download_root=paths.models,
        device=capabilities.selected_device,
        compute_type="int8" if capabilities.selected_device == "cpu" else "default",
    )
    controller = JobController(
        paths=paths,
        transcription=transcription,
        emit=lambda _event: None,
        clipboard_settings=ClipboardSettings(auto_paste=config.auto_paste),
    )
    try:
        controller.set_input_device(config.input_device)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not apply saved input device (%s); using default.", exc)
    transcription.set_vad_filter(config.vad_filter)
    transcription.set_decode_options(
        beam_size=config.beam_size,
        word_timestamps=config.word_timestamps,
    )
    controller.set_enhance_audio(config.enhance_audio)

    root = tk.Tk()
    create_app(
        root,
        controller,
        config,
        config_path,
        capabilities=capabilities,
        paths=paths,
        vram_gb=vram_gb,
    )
    try_start_hotkey(controller)
    root.mainloop()
