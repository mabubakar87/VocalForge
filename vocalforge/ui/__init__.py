"""Tkinter main window and setup dashboard for VocalForge."""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from typing import Optional

from vocalforge.capabilities import CapabilityReport
from vocalforge.clipboard import ClipboardSettings
from vocalforge.config import AppConfig, save_config
from vocalforge.formatting import prepare_ui_text, resolve_transcript_font_family, resolve_ui_font_family
from vocalforge.jobs import EventType, JobController, JobEvent
from vocalforge.ui.theme import ModernDropdown, SegmentedControl, Theme, hoverable_button, make_record_button_images
from vocalforge.profiles import (
    LANGUAGE_OPTIONS,
    PROFILE_ORDER,
    PROFILES,
    cuda_is_usable,
    free_disk_bytes,
    has_enough_disk_for_profile,
    is_model_available_locally,
    language_mode_from_config,
    language_option_label,
    language_settings_for_mode,
    profile_for_model,
    probe_vram_gb,
    recommend_profile,
    resolve_language_settings,
    resolve_runtime,
)
from vocalforge.state import AppState
from vocalforge.storage import AppPaths

logger = logging.getLogger(__name__)


class SetupDashboard(tk.Toplevel):
    """Hardware-aware profile setup window."""

    def __init__(
        self,
        master: tk.Tk,
        *,
        capabilities: CapabilityReport,
        paths: AppPaths,
        config: AppConfig,
        config_path,
        controller: JobController,
        on_activated,
        on_settings_changed=None,
        vram_gb: float | None,
    ) -> None:
        super().__init__(master)
        self.title("VocalForge Setup")
        self.configure(bg=Theme.bg)
        self._ui_font = resolve_ui_font_family(self)
        self.geometry("920x860")
        self.minsize(820, 800)
        self.capabilities = capabilities
        self.paths = paths
        self.config = config
        self.config_path = config_path
        self.controller = controller
        self.on_activated = on_activated
        self.on_settings_changed = on_settings_changed or (lambda: None)
        self.vram_gb = vram_gb
        self.recommendation = recommend_profile(capabilities, vram_gb)
        self._device_var = tk.StringVar(value=self._default_device_choice())
        self._language_mode = tk.StringVar(
            value=language_mode_from_config(self.config.language, self.config.task)
        )
        self._language_labels = {label: mode_id for mode_id, label, _lang, _task in LANGUAGE_OPTIONS}
        self._language_label_by_id = {mode_id: label for mode_id, label, _lang, _task in LANGUAGE_OPTIONS}
        self._table_host: tk.Frame | None = None
        self._body: tk.Frame | None = None
        self._built = False
        self.protocol("WM_DELETE_WINDOW", self._close)
        # Show the window immediately; this Tk build is slow to create many widgets (~2s).
        self._loading = tk.Label(
            self,
            text="Loading setup…",
            bg=Theme.bg,
            fg=Theme.muted,
            font=(self._ui_font, 12),
        )
        self._loading.pack(expand=True, pady=48)
        self.update_idletasks()
        self.after_idle(self._deferred_build)

    def _deferred_build(self) -> None:
        if not self.winfo_exists():
            return
        if self._loading is not None:
            self._loading.destroy()
            self._loading = None
        self._build()
        self._built = True

    def present(self) -> None:
        """Show an existing Setup window quickly (reuse instead of recreating)."""
        self._device_var.set(self._default_device_choice())
        self._language_mode.set(
            language_mode_from_config(self.config.language, self.config.task)
        )
        self.deiconify()
        self.lift()
        self.focus_force()
        if self._built:
            self.after_idle(self._refresh_profiles_table)

    def _refresh_profiles_table(self) -> None:
        if self._table_host is None or not self.winfo_exists():
            return
        for child in self._table_host.winfo_children():
            child.destroy()
        self._build_profiles_table(self._table_host)

    def _default_device_choice(self) -> str:
        if self.controller.transcription.is_ready:
            active = self.controller.transcription.device
            if active in {"cpu", "cuda"}:
                if active == "cuda" and not cuda_is_usable(self.capabilities):
                    return "cpu"
                return active
        preferred = (self.config.preferred_device or "").lower()
        if preferred in {"cpu", "cuda"}:
            if preferred == "cuda" and not cuda_is_usable(self.capabilities):
                return "cpu"
            return preferred
        selected = self.capabilities.selected_device
        return selected if selected in {"cpu", "cuda"} else "cpu"

    def _preferred_device(self) -> str:
        return self._device_var.get()

    def _build(self) -> None:
        if self._body is not None:
            self._body.destroy()
        self._body = tk.Frame(self, bg=Theme.bg)
        self._body.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        self._build_system_spec(self._body)
        self._build_processing_preference(self._body)
        self._table_host = tk.Frame(self._body, bg=Theme.bg)
        self._table_host.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self._build_profiles_table(self._table_host)
        hoverable_button(
            self._body,
            text="Close",
            command=self._close,
            font_family=self._ui_font,
            bg=Theme.idle_btn,
            fg=Theme.text,
            hover_bg=Theme.idle_btn_hover,
            padx=18,
            pady=8,
        ).pack(pady=(14, 0))

    def _close(self) -> None:
        self.on_settings_changed()
        self.withdraw()

    def _build_system_spec(self, parent: tk.Frame) -> None:
        section = tk.Frame(parent, bg=Theme.surface, padx=16, pady=14, highlightbackground=Theme.border, highlightthickness=1)
        section.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            section,
            text="System Specification",
            fg=Theme.text,
            bg=Theme.surface,
            font=(self._ui_font, 13, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        free_gb = free_disk_bytes(self.paths.root) / (1024**3)
        cuda_status = "Available" if cuda_is_usable(self.capabilities) else "Unavailable"
        if self.capabilities.fallback_reason and not cuda_is_usable(self.capabilities):
            cuda_status = f"Unavailable ({self.capabilities.fallback_reason})"

        specs = [
            ("OS", f"{self.capabilities.os_name} ({self.capabilities.architecture})"),
            ("CUDA", cuda_status),
            ("Detected device", self.capabilities.selected_device.upper()),
            ("VRAM", f"{self.vram_gb} GB" if self.vram_gb is not None else "Unknown"),
            ("Free disk", f"{free_gb:.1f} GB"),
            ("Recommended profile", PROFILES[self.recommendation.profile_id].label),
        ]
        grid = tk.Frame(section, bg=Theme.surface)
        grid.pack(fill=tk.X)
        for column in range(3):
            grid.grid_columnconfigure(column, weight=1, uniform="specs")
        for index, (label, value) in enumerate(specs):
            row, column = divmod(index, 3)
            cell = tk.Frame(grid, bg=Theme.surface_2, padx=12, pady=10)
            cell.grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
            tk.Label(cell, text=label.upper(), fg=Theme.muted, bg=Theme.surface_2, font=(self._ui_font, 8), anchor="w").pack(
                fill=tk.X
            )
            tk.Label(
                cell,
                text=value,
                fg=Theme.text,
                bg=Theme.surface_2,
                font=(self._ui_font, 10, "bold"),
                anchor="w",
                wraplength=240,
                justify="left",
            ).pack(fill=tk.X, pady=(4, 0))
        tk.Label(
            section,
            text=self.recommendation.reason,
            fg=Theme.muted,
            bg=Theme.surface,
            font=(self._ui_font, 9),
            wraplength=860,
            justify="left",
            anchor="w",
        ).pack(fill=tk.X, pady=(10, 0))

    def _build_processing_preference(self, parent: tk.Frame) -> None:
        section = tk.Frame(parent, bg=Theme.surface, padx=16, pady=14, highlightbackground=Theme.border, highlightthickness=1)
        section.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            section,
            text="Processing Device",
            fg=Theme.text,
            bg=Theme.surface,
            font=(self._ui_font, 13, "bold"),
        ).pack(anchor="w", pady=(0, 6))
        tk.Label(
            section,
            text="Choose whether transcription should run on CPU or GPU. Changing this reloads the active profile on the selected device.",
            fg=Theme.muted,
            bg=Theme.surface,
            font=(self._ui_font, 9),
            wraplength=860,
            justify="left",
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 10))
        disabled = set() if cuda_is_usable(self.capabilities) else {"cuda"}
        SegmentedControl(
            section,
            options=(("cpu", "CPU"), ("cuda", "GPU (CUDA)")),
            variable=self._device_var,
            command=self._on_device_preference_changed,
            font_family=self._ui_font,
            disabled_values=disabled,
        ).pack(anchor="w")
        if not cuda_is_usable(self.capabilities):
            tk.Label(
                section,
                text="GPU is disabled because CUDA is not usable on this machine.",
                fg=Theme.warning,
                bg=Theme.surface,
                font=(self._ui_font, 9),
                anchor="w",
            ).pack(fill=tk.X, pady=(10, 0))

    def _on_device_preference_changed(self) -> None:
        choice = self._preferred_device()
        if choice == "cuda" and not cuda_is_usable(self.capabilities):
            self._device_var.set("cpu")
            choice = "cpu"
            messagebox.showwarning("GPU Unavailable", "CUDA is not usable on this machine. Staying on CPU.")
        self.config.preferred_device = choice
        save_config(self.config_path, self.config)
        if self._table_host is not None:
            for child in self._table_host.winfo_children():
                child.destroy()
            self._build_profiles_table(self._table_host)
        self._apply_device_to_active_profile(choice)

    def _apply_device_to_active_profile(self, preferred: str) -> None:
        """Reload the active profile when the processing device changes."""
        profile_id = self.config.active_profile
        if profile_id not in PROFILES:
            return
        profile = PROFILES[profile_id]
        runtime = resolve_runtime(profile, self.capabilities, preferred_device=preferred)
        current = (
            self.controller.transcription.device
            if self.controller.transcription.is_ready
            else None
        )
        if current == runtime.device:
            return
        if not self.controller.can_change_profile():
            messagebox.showwarning(
                "Busy",
                "Device preference saved. It will apply the next time you activate a profile.",
            )
            return
        ok = self.controller.activate_profile(
            profile.id,
            runtime.device,
            runtime.compute_type,
            profile.model,
        )
        if ok:
            self.on_activated(profile)

    def _build_profiles_table(self, parent: tk.Frame) -> None:
        table = tk.Frame(parent, bg=Theme.bg)
        table.pack(fill=tk.BOTH, expand=True)
        table.grid_columnconfigure(0, weight=0, minsize=110)
        for column in range(1, 4):
            table.grid_columnconfigure(column, weight=1, uniform="profiles")
        self._table_cell(table, 0, 0, "", bold=True, bg=Theme.bg)
        for column, profile_id in enumerate(PROFILE_ORDER, start=1):
            profile = PROFILES[profile_id]
            title = profile.label
            badges = []
            if profile_id == self.recommendation.profile_id:
                badges.append("Recommended")
            if self.config.active_profile == profile_id:
                badges.append("Active")
            if badges:
                title = f"{title}\n[{' · '.join(badges)}]"
            self._table_cell(table, 0, column, title, bold=True, center=True, bg=Theme.surface)
        row_specs = [
            ("Description", "description"),
            ("Model", "model"),
            ("Device", "device"),
            ("Compute", "compute"),
            ("Language", "language"),
            ("Status", "status"),
        ]
        for row_index, (label, key) in enumerate(row_specs, start=1):
            bg = Theme.surface_2 if row_index % 2 else Theme.surface
            self._table_cell(table, row_index, 0, label, bold=True, bg=bg, anchor="w")
            for column, profile_id in enumerate(PROFILE_ORDER, start=1):
                if key == "language":
                    self._add_language_cell(table, row_index, column, profile_id, bg=bg)
                else:
                    self._table_cell(
                        table, row_index, column, self._profile_field(profile_id, key), bg=bg, center=True
                    )
        action_row = len(row_specs) + 1
        self._table_cell(table, action_row, 0, "Action", bold=True, bg=Theme.surface, anchor="w")
        for column, profile_id in enumerate(PROFILE_ORDER, start=1):
            action = tk.Frame(table, bg=Theme.surface, padx=8, pady=10)
            action.grid(row=action_row, column=column, sticky="nsew", padx=4, pady=4)
            self._add_action_button(action, profile_id)

    def _profile_field(self, profile_id: str, key: str) -> str:
        profile = PROFILES[profile_id]
        runtime = resolve_runtime(profile, self.capabilities, preferred_device=self._preferred_device())
        local = is_model_available_locally(self.paths.models, profile)
        if key == "description":
            return profile.description
        if key == "model":
            return profile.model
        if key == "device":
            return runtime.device
        if key == "compute":
            return runtime.compute_type
        if key == "status":
            return "Ready locally" if local else f"Download needed ({profile.size_label})"
        if key == "language":
            if profile.english_only:
                return "English only"
            return language_option_label(self._language_mode.get())
        return ""

    def _add_language_cell(
        self,
        parent: tk.Frame,
        row: int,
        column: int,
        profile_id: str,
        *,
        bg: str,
    ) -> None:
        profile = PROFILES[profile_id]
        cell = tk.Frame(parent, bg=bg, padx=8, pady=8)
        cell.grid(row=row, column=column, sticky="nsew", padx=4, pady=2)
        if profile.english_only:
            tk.Label(
                cell,
                text="English only",
                fg=Theme.muted,
                bg=bg,
                font=(self._ui_font, 10),
                justify="center",
                anchor="center",
            ).pack(fill=tk.BOTH, expand=True)
            return
        labels = [label for _mode_id, label, _lang, _task in LANGUAGE_OPTIONS]
        current_id = self._language_mode.get()
        ModernDropdown(
            cell,
            options=labels,
            selected=self._language_label_by_id.get(current_id, "Auto-detect"),
            on_select=self._on_language_selected,
            font_family=self._ui_font,
            width=16,
        ).pack(fill=tk.X, expand=True)

    def _on_language_selected(self, label: str) -> None:
        mode_id = self._language_labels.get(label, "auto")
        self._language_mode.set(mode_id)
        language, task = language_settings_for_mode(mode_id)
        self.config.language = language
        self.config.task = task
        save_config(self.config_path, self.config)

        # Language belongs to High Accuracy — activate that profile if needed.
        if self.config.active_profile != "high_accuracy":
            self._activate("high_accuracy")
            return

        self.controller.transcription.set_language_settings(language, task)
        self.on_settings_changed()

    def _apply_language_to_transcription(self) -> None:
        profile_id = self.config.active_profile
        if profile_id not in PROFILES:
            return
        profile = PROFILES[profile_id]
        language, task = resolve_language_settings(
            profile,
            language=self.config.language,
            task=self.config.task,
        )
        self.controller.transcription.set_language_settings(language, task)

    def _table_cell(
        self,
        parent: tk.Frame,
        row: int,
        column: int,
        text: str,
        *,
        bold: bool = False,
        center: bool = False,
        bg: str = Theme.surface,
        anchor: str = "center",
    ) -> None:
        cell = tk.Frame(parent, bg=bg, padx=8, pady=8)
        cell.grid(row=row, column=column, sticky="nsew", padx=4, pady=2)
        tk.Label(
            cell,
            text=text,
            fg=Theme.text if bold else Theme.muted,
            bg=bg,
            font=(self._ui_font, 10, "bold" if bold else "normal"),
            justify="center" if center else "left",
            anchor=anchor if not center else "center",
            wraplength=220 if column else 100,
        ).pack(fill=tk.BOTH, expand=True)

    def _add_action_button(self, parent: tk.Frame, profile_id: str) -> None:
        profile = PROFILES[profile_id]
        local = is_model_available_locally(self.paths.models, profile)
        active = self.config.active_profile == profile_id
        btn_text = "Activated" if active else ("Activate" if local else "Download & Activate")
        if active:
            btn = hoverable_button(
                parent,
                text=btn_text,
                command=lambda: None,
                font_family=self._ui_font,
                bg=Theme.success,
                fg=Theme.success_text,
                hover_bg=Theme.success,
                disabledforeground=Theme.success_text,
            )
            btn.configure(state="disabled")
        else:
            btn = hoverable_button(
                parent,
                text=btn_text,
                command=lambda p=profile_id: self._activate(p),
                font_family=self._ui_font,
                bg=Theme.idle_btn,
                fg=Theme.text,
                hover_bg=Theme.idle_btn_hover,
            )
            if not self.controller.can_change_profile():
                btn.configure(state="disabled")
        btn.pack(fill=tk.X)

    def _activate(self, profile_id: str) -> None:
        if not self.controller.can_change_profile():
            messagebox.showwarning("Busy", "Cannot change profile while a job is active.")
            return
        profile = PROFILES[profile_id]
        if not is_model_available_locally(self.paths.models, profile):
            if not has_enough_disk_for_profile(self.paths.root, profile):
                messagebox.showerror(
                    "Insufficient Disk Space",
                    f"Not enough free disk space to download {profile.label} ({profile.size_label}).",
                )
                return
        preferred = self._preferred_device()
        runtime = resolve_runtime(profile, self.capabilities, preferred_device=preferred)
        language, task = resolve_language_settings(
            profile,
            language=self.config.language,
            task=self.config.task,
        )
        self.config.active_profile = profile.id
        self.config.selected_model = profile.model
        self.config.preferred_device = preferred
        if not profile.english_only:
            mode_id = self._language_mode.get()
            language, task = language_settings_for_mode(mode_id)
            self.config.language = language
            self.config.task = task
        save_config(self.config_path, self.config)
        self.controller.transcription.set_language_settings(language, task)
        ok = self.controller.activate_profile(
            profile.id,
            runtime.device,
            runtime.compute_type,
            profile.model,
        )
        if ok:
            self.on_activated(profile)
            self.on_settings_changed()
            self.withdraw()


class MainWindow:
    def __init__(
        self,
        root: tk.Tk,
        controller: JobController,
        config: AppConfig,
        config_path,
        capabilities: CapabilityReport,
        paths: AppPaths,
        vram_gb: float | None = None,
    ) -> None:
        self.root = root
        self.controller = controller
        self.config = config
        self.config_path = config_path
        self.capabilities = capabilities
        self.paths = paths
        self.vram_gb = vram_gb if vram_gb is not None else probe_vram_gb()
        self.ui_queue: queue.Queue[JobEvent] = queue.Queue()
        self._poll_after_id: Optional[str] = None
        self._closing = False
        self._current_state = AppState.STARTING

        self.root.title("VocalForge")
        self.root.geometry("560x660")
        self.root.configure(bg=Theme.bg)
        self._ui_font = resolve_ui_font_family(self.root)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._setup_win: SetupDashboard | None = None
        self._build_widgets()
        self._apply_control_state(AppState.STARTING)
        self.controller.emit = self.enqueue_event
        self.controller.bootstrap()
        self._poll_ui_queue()
        self._restore_or_prompt_profile()

    def enqueue_event(self, event: JobEvent) -> None:
        self.ui_queue.put(event)

    def _resolve_startup_profile_id(self) -> str | None:
        if self.config.active_profile in PROFILES:
            return self.config.active_profile
        if self.config.selected_model:
            profile = profile_for_model(self.config.selected_model)
            if profile is not None:
                return profile.id
        return None

    def _restore_or_prompt_profile(self) -> None:
        profile_id = self._resolve_startup_profile_id()
        if profile_id is not None:
            self._activate_saved_profile(profile_id)
            return
        self._set_status("Choose a profile in Setup / Profiles")
        self.root.after(200, self.open_setup)

    def _build_widgets(self) -> None:
        frame = tk.Frame(self.root, bg=Theme.bg)
        frame.pack(pady=14, padx=18, fill=tk.X)

        brand = tk.Label(
            frame,
            text="VocalForge",
            fg=Theme.accent,
            bg=Theme.bg,
            font=(self._ui_font, 18, "bold"),
            anchor="w",
        )
        brand.pack(fill=tk.X, pady=(0, 10))

        info = tk.Frame(
            frame,
            bg=Theme.surface,
            padx=14,
            pady=12,
            highlightbackground=Theme.border,
            highlightthickness=1,
        )
        info.pack(fill=tk.X, pady=(0, 12))
        tk.Label(
            info,
            text="ACTIVE SESSION",
            fg=Theme.muted,
            bg=Theme.surface,
            font=(self._ui_font, 8, "bold"),
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 8))

        rows = tk.Frame(info, bg=Theme.surface)
        rows.pack(fill=tk.X)
        for column in range(2):
            rows.grid_columnconfigure(column, weight=1, uniform="session")

        self.profile_value = self._session_cell(rows, 0, 0, "Profile", "None")
        self.model_value = self._session_cell(rows, 0, 1, "Model", "Not selected")
        self.device_value = self._session_cell(rows, 1, 0, "Transcription via", "—")
        self.language_value = self._session_cell(rows, 1, 1, "Language", "—")
        self._refresh_profile_label()

        self.setup_button = hoverable_button(
            frame,
            text="Setup / Profiles",
            command=self.open_setup,
            font_family=self._ui_font,
            bg=Theme.surface_2,
            fg=Theme.text,
            hover_bg=Theme.surface_3,
            padx=16,
            pady=9,
        )
        self.setup_button.pack(pady=(0, 10))

        self.status_label = tk.Label(
            frame,
            text="Start Recording",
            fg=Theme.text,
            bg=Theme.bg,
            font=(self._ui_font, 12),
        )
        self.status_label.pack(pady=4)

        self.canvas = tk.Canvas(frame, width=88, height=88, bg=Theme.bg, highlightthickness=0, cursor="hand2")
        self.canvas.pack(pady=12)
        self._record_idle_img, self._record_active_img = make_record_button_images(self.root, size=88)
        self.record_button = self.canvas.create_image(44, 44, image=self._record_idle_img)
        self.canvas.bind("<Button-1>", lambda _event: self.on_record_clicked())

        self.upload_button = hoverable_button(
            self.root,
            text="Upload Audio File",
            command=self.on_upload,
            font_family=self._ui_font,
            bg=Theme.idle_btn,
            fg=Theme.text,
            hover_bg=Theme.idle_btn_hover,
            padx=16,
            pady=9,
        )
        self.upload_button.pack(pady=8)

        text_frame = tk.Frame(
            self.root,
            bg=Theme.surface,
            highlightbackground=Theme.border,
            highlightthickness=1,
        )
        text_frame.pack(pady=10, padx=18, fill=tk.BOTH, expand=True)
        tk.Label(
            text_frame,
            text="TRANSCRIPT",
            fg=Theme.muted,
            bg=Theme.surface,
            font=(self._ui_font, 8, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(10, 0))
        transcript_font = (resolve_transcript_font_family(self.root), 13)
        self._last_transcript = ""
        self.text_output = scrolledtext.ScrolledText(
            text_frame,
            height=5,
            width=50,
            wrap=tk.WORD,
            bg=Theme.surface,
            fg=Theme.text,
            insertbackground=Theme.accent,
            font=transcript_font,
            bd=0,
            highlightthickness=0,
            padx=8,
            pady=8,
        )
        self.text_output.pack(pady=(4, 10), padx=10, fill=tk.BOTH, expand=True)
        self.text_output.bind("<<Copy>>", self._copy_logical_transcript)
        self.text_output.bind("<Control-c>", self._copy_logical_transcript)
        self.text_output.bind("<Control-C>", self._copy_logical_transcript)

    def _session_cell(
        self, parent: tk.Frame, row: int, column: int, label: str, value: str
    ) -> tk.Label:
        cell = tk.Frame(parent, bg=Theme.surface_2, padx=10, pady=8)
        cell.grid(row=row, column=column, sticky="nsew", padx=3, pady=3)
        tk.Label(cell, text=label.upper(), fg=Theme.muted, bg=Theme.surface_2, font=(self._ui_font, 8), anchor="w").pack(
            fill=tk.X
        )
        value_label = tk.Label(
            cell,
            text=value,
            fg=Theme.text,
            bg=Theme.surface_2,
            font=(self._ui_font, 10, "bold"),
            anchor="w",
            wraplength=220,
            justify="left",
        )
        value_label.pack(fill=tk.X, pady=(4, 0))
        return value_label

    def _language_display_label(self) -> str:
        profile_id = self.config.active_profile
        if profile_id not in PROFILES:
            return "—"
        profile = PROFILES[profile_id]
        if profile.english_only:
            return "English only"
        return language_option_label(
            language_mode_from_config(self.config.language, self.config.task)
        )

    def _set_transcript_display(self, text: str) -> None:
        """Show transcript with Arabic shaping for Tk; keep logical text for copy."""
        self._last_transcript = text
        self.text_output.configure(font=(resolve_transcript_font_family(self.root, text), 13))
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, prepare_ui_text(text))

    def _copy_logical_transcript(self, _event=None):
        if self._last_transcript:
            self.root.clipboard_clear()
            self.root.clipboard_append(self._last_transcript)
        return "break"

    def _device_display_label(self, device: str | None) -> str:
        if not device:
            return "—"
        normalized = device.lower()
        if normalized == "cuda":
            return "GPU (CUDA)"
        if normalized == "cpu":
            return "CPU"
        return device.upper()

    def _refresh_profile_label(self) -> None:
        profile_id = self.config.active_profile
        if profile_id in PROFILES:
            profile = PROFILES[profile_id]
            self.profile_value.config(text=profile.label)
            self.model_value.config(text=f"{profile.model}\n({profile.size_label})")
        else:
            self.profile_value.config(text="None")
            self.model_value.config(text="Not selected")

        device = None
        if self.controller.transcription.is_ready:
            device = self.controller.transcription.device
        elif self.config.preferred_device:
            device = self.config.preferred_device
        self.device_value.config(text=self._device_display_label(device))
        self.language_value.config(text=self._language_display_label())

    def open_setup(self) -> None:
        if self._setup_win is not None:
            try:
                if self._setup_win.winfo_exists():
                    self._setup_win.present()
                    return
            except tk.TclError:
                self._setup_win = None
        self._setup_win = SetupDashboard(
            self.root,
            capabilities=self.capabilities,
            paths=self.paths,
            config=self.config,
            config_path=self.config_path,
            controller=self.controller,
            on_activated=self._on_profile_activated,
            on_settings_changed=self._refresh_profile_label,
            vram_gb=self.vram_gb,
        )

    def _on_profile_activated(self, profile) -> None:
        self._refresh_profile_label()

    def _activate_saved_profile(self, profile_id: str) -> None:
        profile = PROFILES[profile_id]
        preferred = self.config.preferred_device or self.capabilities.selected_device
        runtime = resolve_runtime(profile, self.capabilities, preferred_device=preferred)
        language, task = resolve_language_settings(
            profile,
            language=self.config.language,
            task=self.config.task,
        )
        self.config.active_profile = profile.id
        self.config.selected_model = profile.model
        self.config.preferred_device = runtime.device
        save_config(self.config_path, self.config)
        self.controller.transcription.set_language_settings(language, task)
        self._refresh_profile_label()
        self.controller.activate_profile(profile.id, runtime.device, runtime.compute_type, profile.model)

    def on_record_clicked(self) -> None:
        self.controller.toggle_recording()

    def on_upload(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            self.controller.transcribe_upload(file_path)

    def _apply_control_state(self, state: AppState) -> None:
        self._current_state = state
        ready = state is AppState.READY
        recording = state is AppState.RECORDING
        can_select = state in {AppState.NO_MODEL, AppState.READY, AppState.ERROR}
        self.setup_button.configure(state=("normal" if can_select else "disabled"))
        self.upload_button.configure(state=("normal" if ready else "disabled"))
        if recording:
            self.canvas.itemconfig(self.record_button, image=self._record_active_img)
        else:
            self.canvas.itemconfig(self.record_button, image=self._record_idle_img)
        if ready or recording:
            self.canvas.bind("<Button-1>", lambda _event: self.on_record_clicked())
        else:
            self.canvas.bind("<Button-1>", lambda _event: None)

    def _set_status(self, text: str, fg: str | None = None) -> None:
        self.status_label.config(text=text, fg=fg or Theme.text)

    def _handle_event(self, event: JobEvent) -> None:
        payload = event.payload or {}
        if event.type is EventType.STATE_CHANGED:
            state = AppState(payload["state"])
            self._apply_control_state(state)
        elif event.type is EventType.STATUS:
            self._set_status(payload.get("text", ""), payload.get("fg") or Theme.text)
        elif event.type is EventType.MODEL_LOADED:
            device = payload.get("device", "")
            self._refresh_profile_label()
            self._set_status(f"Ready — transcribing via {self._device_display_label(device)}.")
        elif event.type is EventType.MODEL_LOAD_FAILED:
            self._set_status("Error loading model", fg=Theme.danger)
            messagebox.showerror("Model Error", payload.get("message", "Unknown error"))
        elif event.type is EventType.TRANSCRIPTION_COMPLETED:
            text = payload.get("text", "")
            self._set_transcript_display(text)
            self._set_status("Transcription Completed")
        elif event.type is EventType.JOB_FAILED:
            self._set_status("Error", fg=Theme.danger)
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
        if self._setup_win is not None:
            try:
                if self._setup_win.winfo_exists():
                    self._setup_win.destroy()
            except tk.TclError:
                pass
            self._setup_win = None
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
    capabilities: CapabilityReport,
    paths: AppPaths,
    vram_gb: float | None = None,
) -> MainWindow:
    controller.clipboard_settings = ClipboardSettings(auto_paste=config.auto_paste)
    return MainWindow(
        root,
        controller,
        config,
        config_path,
        capabilities=capabilities,
        paths=paths,
        vram_gb=vram_gb,
    )
