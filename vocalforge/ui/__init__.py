"""Tkinter main window for VocalForge."""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

from vocalforge.clipboard import ClipboardSettings
from vocalforge.config import AppConfig, load_config, save_config
from vocalforge.jobs import EventType, JobController, JobEvent
from vocalforge.state import AppState

logger = logging.getLogger(__name__)


class MainWindow:
    def __init__(
        self,
        root: tk.Tk,
        controller: JobController,
        config: AppConfig,
        config_path,
        models: list[str],
    ) -> None:
        self.root = root
        self.controller = controller
        self.config = config
        self.config_path = config_path
        self.models = models
        self.ui_queue: queue.Queue[JobEvent] = queue.Queue()
        self._poll_after_id: Optional[str] = None
        self._closing = False

        self.root.title("VocalForge")
        self.root.geometry("500x500")
        self.root.configure(bg="#2E2E2E")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_widgets()
        self._apply_control_state(AppState.STARTING)
        self.controller.emit = self.enqueue_event
        self.controller.bootstrap()
        self._poll_ui_queue()

        if self.config.selected_model:
            display = self._display_for_model(self.config.selected_model)
            if display:
                self.model_combobox.set(display)
                self.controller.load_model(self.config.selected_model)

    def enqueue_event(self, event: JobEvent) -> None:
        self.ui_queue.put(event)

    def _display_for_model(self, model_name: str) -> str:
        for item in self.models:
            if item.startswith(model_name):
                return item
        return model_name

    def _build_widgets(self) -> None:
        frame = tk.Frame(self.root, bg="#2E2E2E")
        frame.pack(pady=10)

        tk.Label(frame, text="Select Model:", fg="white", bg="#2E2E2E", font=("Arial", 10)).pack(pady=5)
        self.model_combobox = ttk.Combobox(frame, values=self.models, state="readonly")
        self.model_combobox.set("")
        self.model_combobox.pack(pady=5)
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_select)

        self.status_label = tk.Label(
            frame, text="Start Recording", fg="white", bg="#2E2E2E", font=("Arial", 12)
        )
        self.status_label.pack(pady=5)

        self.canvas = tk.Canvas(frame, width=80, height=80, bg="#2E2E2E", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.record_button = self.canvas.create_oval(10, 10, 70, 70, fill="#4CAF50", outline="")
        self.canvas.create_oval(25, 25, 55, 55, fill="white", outline="")
        self.canvas.bind("<Button-1>", lambda _event: self.on_record_clicked())

        style = ttk.Style()
        style.configure(
            "Rounded.TButton",
            borderwidth=0,
            relief="flat",
            background="#555",
            foreground="black",
            font=("Arial", 10),
            padding=10,
        )
        style.configure(
            "Rounded.TFrame",
            background="#1E1E1E",
            borderwidth=0,
            relief="flat",
        )

        self.upload_button = ttk.Button(
            self.root, text="Upload Audio File", command=self.on_upload, style="Rounded.TButton"
        )
        self.upload_button.pack(pady=10)

        text_frame = ttk.Frame(self.root, style="Rounded.TFrame")
        text_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        self.text_output = scrolledtext.ScrolledText(
            text_frame,
            height=5,
            width=50,
            wrap=tk.WORD,
            bg="#1E1E1E",
            fg="white",
            font=("Arial", 12),
            bd=0,
            highlightthickness=0,
        )
        self.text_output.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    def on_model_select(self, _event=None) -> None:
        selected = self.model_combobox.get()
        if not selected:
            return
        model_name = selected.split(" (")[0]
        self.config.selected_model = model_name
        save_config(self.config_path, self.config)
        self.controller.load_model(model_name)

    def on_record_clicked(self) -> None:
        self.controller.toggle_recording()

    def on_upload(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            self.controller.transcribe_upload(file_path)

    def _apply_control_state(self, state: AppState) -> None:
        ready = state is AppState.READY
        recording = state is AppState.RECORDING
        can_select = state in {AppState.NO_MODEL, AppState.READY, AppState.ERROR}
        self.model_combobox.configure(state="readonly" if can_select else "disabled")
        self.upload_button.configure(state=("normal" if ready else "disabled"))
        if recording:
            self.canvas.itemconfig(self.record_button, fill="#F44336")
        else:
            self.canvas.itemconfig(self.record_button, fill="#4CAF50")
        # Recording is only clickable when ready or already recording (to stop).
        if ready or recording:
            self.canvas.bind("<Button-1>", lambda _event: self.on_record_clicked())
        else:
            self.canvas.bind("<Button-1>", lambda _event: None)

    def _set_status(self, text: str, fg: str = "white") -> None:
        self.status_label.config(text=text, fg=fg)

    def _handle_event(self, event: JobEvent) -> None:
        payload = event.payload or {}
        if event.type is EventType.STATE_CHANGED:
            state = AppState(payload["state"])
            self._apply_control_state(state)
        elif event.type is EventType.STATUS:
            self._set_status(payload.get("text", ""), payload.get("fg", "white"))
        elif event.type is EventType.MODEL_LOADED:
            model = payload.get("model", "")
            device = payload.get("device", "")
            self._set_status(f"Model '{model}' loaded on {device}.")
        elif event.type is EventType.MODEL_LOAD_FAILED:
            self._set_status("Error loading model", fg="red")
            messagebox.showerror("Model Error", payload.get("message", "Unknown error"))
        elif event.type is EventType.TRANSCRIPTION_COMPLETED:
            text = payload.get("text", "")
            self.text_output.delete(1.0, tk.END)
            self.text_output.insert(tk.END, text)
            self._set_status("Transcription Completed")
        elif event.type is EventType.JOB_FAILED:
            self._set_status("Error", fg="red")
            message = payload.get("message")
            if message:
                messagebox.showerror("Error", message)
        elif event.type is EventType.SHUTDOWN_COMPLETED:
            pass

    def _poll_ui_queue(self) -> None:
        if self._closing:
            return
        try:
            while True:
                event = self.ui_queue.get_nowait()
                try:
                    self._handle_event(event)
                except Exception as exc:  # noqa: BLE001
                    logger.error("UI callback failed: %s", exc)
        except queue.Empty:
            pass
        self._poll_after_id = self.root.after(50, self._poll_ui_queue)

    def on_close(self) -> None:
        self._closing = True
        if self._poll_after_id is not None:
            try:
                self.root.after_cancel(self._poll_after_id)
            except Exception:  # noqa: BLE001
                pass
        self.controller.shutdown()
        self.root.destroy()


def create_app(
    root: tk.Tk,
    controller: JobController,
    config: AppConfig,
    config_path,
    models: list[str],
) -> MainWindow:
    # Keep clipboard settings in sync with config.
    controller.clipboard_settings = ClipboardSettings(auto_paste=config.auto_paste)
    return MainWindow(root, controller, config, config_path, models)
