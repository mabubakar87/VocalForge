"""Tkinter main window and setup dashboard for VocalForge."""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext
from typing import Optional

from vocalforge.audio import DEFAULT_INPUT_LABEL, list_input_devices
from vocalforge.capabilities import CapabilityReport
from vocalforge.clipboard import ClipboardSettings
from vocalforge.config import AppConfig, save_config
from vocalforge.extras import ExtraStatus, probe_all, status_label
from vocalforge.formatting import prepare_ui_text, resolve_transcript_font_family, resolve_ui_font_family
from vocalforge.jobs import EventType, JobController, JobEvent
from vocalforge.media import UPLOAD_FILEDIALOG_TYPES, is_supported_upload
from vocalforge.ui.theme import (
    InfoTip,
    ModernDropdown,
    SegmentedControl,
    Theme,
    hoverable_button,
    make_record_button_images,
)
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
from vocalforge.storage import AppPaths, list_transcripts, read_transcript, transcript_label
from vocalforge.transcription import BEAM_SIZE_CHOICES, DEFAULT_BEAM_SIZE

logger = logging.getLogger(__name__)

SESSION_TOOLTIPS = {
    "device": (
        "Run transcription on CPU or GPU (CUDA). Changing this reloads the active "
        "profile on the selected device. GPU is disabled when CUDA is not usable."
    ),
    "language": (
        "Language spoken in the audio (source). Auto-detect lets the model guess; "
        "pick a language for better accuracy when you know it. "
        "Does not change output language — use Translate for Chinese/etc. → English. "
        "Available on Multilingual Small / Medium and High Accuracy profiles."
    ),
    "translate": (
        "Whisper speech→English translation. Requires Multilingual Small or "
        "Medium (faster-whisper-small / medium). large-v3-turbo cannot translate."
    ),
    "microphone": (
        "Input device used for recording. System default follows your OS setting; "
        "choose a specific mic if you have more than one."
    ),
    "vad": (
        "Voice Activity Detection strips long silence before transcription. "
        "On reduces hallucinations on quiet tails; Off sends the full recording "
        "when soft speech is being clipped."
    ),
    "beam": (
        "How thoroughly the decoder searches for the best wording. "
        "1 is fastest, 5 is the balanced default, 10 is slower and often "
        "slightly more accurate on difficult audio."
    ),
    "timestamps": (
        "When On, each transcript segment is prefixed with [start–end] timing. "
        "Useful for reviewing longer files; slightly more work for the model."
    ),
    "enhance": (
        "When On, WAV recordings/uploads are denoised with DeepFilterNet before "
        "transcription. Default Off. Requires .deps/deep-filter and soxr."
    ),
    "diarize": (
        "When On, speakers are labeled with pyannote before transcription. "
        "Uses GPU when CUDA torch is available (Whisper is unloaded briefly "
        "to free VRAM). Needs two different voices for SPEAKER_00 vs SPEAKER_01. "
        "Vendor once: python scripts/vendor_diarization_models.py"
    ),
    "history": (
        "Reload a previously saved transcript from this session’s transcripts folder."
    ),
}


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
        self.geometry("920x900")
        self.minsize(820, 600)
        self.capabilities = capabilities
        self.paths = paths
        self.config = config
        self.config_path = config_path
        self.controller = controller
        self.on_activated = on_activated
        self.on_settings_changed = on_settings_changed or (lambda: None)
        self.vram_gb = vram_gb
        self.recommendation = recommend_profile(capabilities, vram_gb)
        self._language_mode = tk.StringVar(
            value=language_mode_from_config(self.config.language, self.config.task)
        )
        self._language_labels = {label: mode_id for mode_id, label, _lang, _task in LANGUAGE_OPTIONS}
        self._language_label_by_id = {mode_id: label for mode_id, label, _lang, _task in LANGUAGE_OPTIONS}
        self._table_host: tk.Frame | None = None
        self._body: tk.Frame | None = None
        self._scroll_shell: tk.Frame | None = None
        self._scroll_canvas: tk.Canvas | None = None
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
        self._bind_setup_mousewheel()
        if self._scroll_canvas is not None:
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

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
        """Prefer the home-screen device setting stored in config."""
        return self._default_device_choice()

    def _build(self) -> None:
        if self._scroll_shell is not None:
            self._scroll_shell.destroy()
            self._scroll_shell = None
            self._scroll_canvas = None
            self._body = None

        self._scroll_shell = tk.Frame(self, bg=Theme.bg)
        self._scroll_shell.pack(fill=tk.BOTH, expand=True)

        self._scroll_canvas = tk.Canvas(
            self._scroll_shell,
            bg=Theme.bg,
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = tk.Scrollbar(
            self._scroll_shell,
            orient=tk.VERTICAL,
            command=self._scroll_canvas.yview,
        )
        self._scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._body = tk.Frame(self._scroll_canvas, bg=Theme.bg)
        self._scroll_window = self._scroll_canvas.create_window(
            (0, 0), window=self._body, anchor="nw"
        )

        def _on_body_configure(_event=None) -> None:
            if self._scroll_canvas is None:
                return
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

        def _on_canvas_configure(event) -> None:
            if self._scroll_canvas is None:
                return
            self._scroll_canvas.itemconfigure(self._scroll_window, width=event.width)

        self._body.bind("<Configure>", _on_body_configure)
        self._scroll_canvas.bind("<Configure>", _on_canvas_configure)

        content = tk.Frame(self._body, bg=Theme.bg)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
        self._build_system_spec(content)
        self._table_host = tk.Frame(content, bg=Theme.bg)
        self._table_host.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self._build_profiles_table(self._table_host)
        self._build_optional_extras(content)
        hoverable_button(
            content,
            text="Close",
            command=self._close,
            font_family=self._ui_font,
            bg=Theme.idle_btn,
            fg=Theme.text,
            hover_bg=Theme.idle_btn_hover,
            padx=18,
            pady=8,
        ).pack(pady=(14, 0))
        self._bind_setup_mousewheel()
        self.after_idle(_on_body_configure)

    def _on_setup_mousewheel(self, event) -> str | None:
        if self._scroll_canvas is None or not self.winfo_exists():
            return None
        if getattr(event, "num", None) == 4:
            self._scroll_canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            self._scroll_canvas.yview_scroll(3, "units")
        else:
            delta = int(getattr(event, "delta", 0) or 0)
            if delta == 0:
                return None
            # Windows: multiples of 120; macOS: smaller values.
            steps = -1 * int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
            self._scroll_canvas.yview_scroll(steps, "units")
        return "break"

    def _bind_setup_mousewheel(self) -> None:
        """Bind wheel events on Setup widgets (not bind_all — avoids stealing main UI)."""
        root = self._scroll_shell
        if root is None:
            return

        def _walk(widget: tk.Misc) -> None:
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(seq, self._on_setup_mousewheel, add="+")
            for child in widget.winfo_children():
                _walk(child)

        _walk(root)

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
        for column in range(6):
            grid.grid_columnconfigure(column, weight=1, uniform="specs")
        for index, (label, value) in enumerate(specs):
            cell = tk.Frame(grid, bg=Theme.surface_2, padx=8, pady=8)
            cell.grid(row=0, column=index, sticky="nsew", padx=3, pady=2)
            tk.Label(cell, text=label.upper(), fg=Theme.muted, bg=Theme.surface_2, font=(self._ui_font, 8), anchor="w").pack(
                fill=tk.X
            )
            tk.Label(
                cell,
                text=value,
                fg=Theme.text,
                bg=Theme.surface_2,
                font=(self._ui_font, 9, "bold"),
                anchor="w",
                wraplength=130,
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

    def _build_optional_extras(self, parent: tk.Frame) -> None:
        section = tk.Frame(
            parent,
            bg=Theme.surface,
            padx=16,
            pady=14,
            highlightbackground=Theme.border,
            highlightthickness=1,
        )
        section.pack(fill=tk.X, pady=(12, 0))
        header = tk.Frame(section, bg=Theme.surface)
        header.pack(fill=tk.X, pady=(0, 6))
        tk.Label(
            header,
            text="Optional Extras (Phase 4)",
            fg=Theme.text,
            bg=Theme.surface,
            font=(self._ui_font, 13, "bold"),
        ).pack(side=tk.LEFT)
        InfoTip(
            header,
            text=(
                "Advanced pipelines stay out of the base install until a proposal "
                "is approved. Status is an import check only — nothing is downloaded here."
            ),
            font_family=self._ui_font,
            bg=Theme.surface,
        ).pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(
            section,
            text=(
                "Speech enhancement unlocks with deep-filter + soxr. "
                "Speaker diarization unlocks after vendoring offline weights "
                "(scripts/vendor_diarization_models.py → Models/diarization/). "
                "HF token is only needed for that one-time download."
            ),
            fg=Theme.muted,
            bg=Theme.surface,
            font=(self._ui_font, 9),
            wraplength=860,
            justify="left",
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 10))

        token_row = tk.Frame(section, bg=Theme.surface)
        token_row.pack(fill=tk.X, pady=(0, 10))
        tk.Label(
            token_row,
            text="Hugging Face token",
            fg=Theme.text,
            bg=Theme.surface,
            font=(self._ui_font, 9, "bold"),
            anchor="w",
        ).pack(side=tk.LEFT)
        self._hf_token_var = tk.StringVar(value=self.config.hf_token or "")
        token_entry = tk.Entry(
            token_row,
            textvariable=self._hf_token_var,
            show="•",
            width=36,
            bg=Theme.surface_2,
            fg=Theme.text,
            insertbackground=Theme.text,
            relief=tk.FLAT,
            font=(self._ui_font, 9),
        )
        token_entry.pack(side=tk.LEFT, padx=(10, 8))
        tk.Button(
            token_row,
            text="Save token",
            command=self._save_hf_token,
            bg=Theme.surface_2,
            fg=Theme.text,
            relief=tk.FLAT,
            font=(self._ui_font, 9),
            padx=10,
            pady=4,
            cursor="hand2",
        ).pack(side=tk.LEFT)
        tk.Label(
            token_row,
            text="Or set HF_TOKEN in the environment",
            fg=Theme.muted,
            bg=Theme.surface,
            font=(self._ui_font, 8),
        ).pack(side=tk.LEFT, padx=(8, 0))

        grid = tk.Frame(section, bg=Theme.surface)
        grid.pack(fill=tk.X)
        extras = list(probe_all(hf_token=self.config.hf_token))
        columns = max(1, len(extras))
        for column in range(columns):
            grid.grid_columnconfigure(column, weight=1, uniform="extras", minsize=140)

        for index, (extra, status) in enumerate(extras):
            cell = tk.Frame(grid, bg=Theme.surface_2, padx=10, pady=10)
            cell.grid(row=0, column=index, sticky="nsew", padx=4, pady=4)
            title = tk.Frame(cell, bg=Theme.surface_2)
            title.pack(fill=tk.X)
            tk.Label(
                title,
                text=extra.label,
                fg=Theme.text,
                bg=Theme.surface_2,
                font=(self._ui_font, 10, "bold"),
                anchor="w",
                wraplength=120,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            status_fg = Theme.muted if status is ExtraStatus.NOT_INSTALLED else Theme.warning
            if status is ExtraStatus.READY:
                status_fg = Theme.success
            tk.Label(
                title,
                text=status_label(status),
                fg=status_fg,
                bg=Theme.surface_2,
                font=(self._ui_font, 8, "bold"),
                anchor="e",
            ).pack(side=tk.RIGHT)
            summary = tk.Label(
                cell,
                text=extra.summary,
                fg=Theme.muted,
                bg=Theme.surface_2,
                font=(self._ui_font, 9),
                wraplength=140,
                justify="left",
                anchor="w",
            )
            summary.pack(fill=tk.X, pady=(6, 0))
            engines = ", ".join(extra.candidate_engines)
            meta = tk.Label(
                cell,
                text=f"Candidates: {engines}  ·  {extra.proposal_doc}",
                fg=Theme.muted,
                bg=Theme.surface_2,
                font=(self._ui_font, 8),
                wraplength=140,
                justify="left",
                anchor="w",
            )
            meta.pack(fill=tk.X, pady=(4, 0))
            if extra.id == "enhancement" and status is ExtraStatus.READY:
                tk.Label(
                    cell,
                    text="Control: home → Enhance",
                    fg=Theme.muted,
                    bg=Theme.surface_2,
                    font=(self._ui_font, 8),
                    anchor="w",
                ).pack(fill=tk.X, pady=(6, 0))
            if extra.id == "diarization" and status is ExtraStatus.READY:
                tk.Label(
                    cell,
                    text="Control: home → Diarize",
                    fg=Theme.muted,
                    bg=Theme.surface_2,
                    font=(self._ui_font, 8),
                    anchor="w",
                ).pack(fill=tk.X, pady=(6, 0))

            def _sync_wrap(event, labels=(summary, meta), title_lbl=title.winfo_children()[0]) -> None:
                width = max(60, int(event.width) - 16)
                for lbl in labels:
                    if int(lbl.cget("wraplength") or 0) != width:
                        lbl.configure(wraplength=width)
                if int(title_lbl.cget("wraplength") or 0) != width:
                    title_lbl.configure(wraplength=width)

            cell.bind("<Configure>", _sync_wrap)

    def _save_hf_token(self) -> None:
        raw = (self._hf_token_var.get() or "").strip()
        self.config.hf_token = raw or None
        save_config(self.config_path, self.config)
        self.controller.set_hf_token(self.config.hf_token)
        self.on_settings_changed()
        messagebox.showinfo(
            "Token saved",
            "Hugging Face token stored in local config.json (gitignored).\n"
            "Re-open Setup to refresh diarization status, or restart the app.",
        )

    def _build_profiles_table(self, parent: tk.Frame) -> None:
        table = tk.Frame(parent, bg=Theme.bg)
        table.pack(fill=tk.BOTH, expand=True)
        profile_count = len(PROFILE_ORDER)
        # Row labels stay narrow; every profile column shares leftover width equally.
        table.grid_columnconfigure(0, weight=0, minsize=110)
        for column in range(1, profile_count + 1):
            table.grid_columnconfigure(column, weight=1, uniform="profiles", minsize=120)
        self._profiles_table = table
        self._table_cell(table, 0, 0, "", bold=True, bg=Theme.bg)
        for column, profile_id in enumerate(PROFILE_ORDER, start=1):
            profile = PROFILES[profile_id]
            size = profile.size_label.lstrip("~").strip()
            title = f"{profile.label} ({size})"
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
        # Setup comparison uses each profile's designed preferred device, not the
        # home Processing override (which would make every column look like int8 on CPU).
        runtime = resolve_runtime(
            profile, self.capabilities, preferred_device=profile.preferred_device
        )
        local = is_model_available_locally(self.paths.models, profile)
        if key == "description":
            return profile.description
        if key == "model":
            return f"{profile.model} ({runtime.compute_type})"
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
        language, _default_task = language_settings_for_mode(mode_id)
        # Preserve Translate-to-English if already enabled.
        task = "translate" if self.config.task == "translate" else "transcribe"
        self.config.language = language
        self.config.task = task
        save_config(self.config_path, self.config)

        # Language/translate need a multilingual profile.
        active = PROFILES.get(self.config.active_profile)
        if active is None or active.english_only:
            self._activate("multilingual_small")
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
        label = tk.Label(
            cell,
            text=text,
            fg=Theme.text if bold else Theme.muted,
            bg=bg,
            font=(self._ui_font, 10, "bold" if bold else "normal"),
            justify="center" if center else "left",
            anchor=anchor if not center else "center",
            wraplength=80,
        )
        label.pack(fill=tk.BOTH, expand=True)

        def _sync_wrap(event, lbl=label) -> None:
            # Keep text wrapping matched to the equal-width profile column.
            width = max(48, int(event.width) - 8)
            if int(lbl.cget("wraplength") or 0) != width:
                lbl.configure(wraplength=width)

        cell.bind("<Configure>", _sync_wrap)

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
        btn.pack(fill=tk.X, expand=True)

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
            language, _default_task = language_settings_for_mode(mode_id)
            task = "translate" if self.config.task == "translate" else "transcribe"
            language, task = resolve_language_settings(
                profile, language=language, task=task
            )
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
        self.root.geometry("760x780")
        self.root.minsize(700, 680)
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

        header = tk.Frame(frame, bg=Theme.bg)
        header.pack(fill=tk.X, pady=(0, 10))
        brand = tk.Label(
            header,
            text="VocalForge",
            fg=Theme.accent,
            bg=Theme.bg,
            font=(self._ui_font, 18, "bold"),
            anchor="w",
        )
        brand.pack(side=tk.LEFT)
        self.setup_button = hoverable_button(
            header,
            text="Setup / Profiles",
            command=self.open_setup,
            font_family=self._ui_font,
            bg=Theme.surface_2,
            fg=Theme.text,
            hover_bg=Theme.surface_3,
            padx=16,
            pady=9,
        )
        self.setup_button.pack(side=tk.RIGHT)

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

        row1 = tk.Frame(info, bg=Theme.surface)
        row1.pack(fill=tk.X)
        for column in range(5):
            row1.grid_columnconfigure(column, weight=1, uniform="session1")

        self.profile_value = self._session_cell(row1, 0, 0, "Profile", "None")
        self.model_value = self._session_cell(row1, 0, 1, "Model", "Not selected")
        self._device_body = self._session_body(
            row1, 0, 2, "Processing", tooltip=SESSION_TOOLTIPS["device"]
        )
        preferred = (self.config.preferred_device or self.capabilities.selected_device or "cpu").lower()
        if preferred == "cuda" and not cuda_is_usable(self.capabilities):
            preferred = "cpu"
        if preferred not in {"cpu", "cuda"}:
            preferred = "cpu"
        self.config.preferred_device = preferred
        self._device_var = tk.StringVar(value=preferred)
        disabled = set() if cuda_is_usable(self.capabilities) else {"cuda"}
        SegmentedControl(
            self._device_body,
            options=(("cpu", "CPU"), ("cuda", "GPU")),
            variable=self._device_var,
            command=self._on_device_preference_changed,
            font_family=self._ui_font,
            disabled_values=disabled,
        ).pack(anchor="w")
        self._language_body = self._session_body(
            row1, 0, 3, "Language", tooltip=SESSION_TOOLTIPS["language"]
        )
        self.language_value: tk.Label | None = None
        self._language_dropdown: ModernDropdown | None = None
        self._language_labels = {label: mode_id for mode_id, label, _lang, _task in LANGUAGE_OPTIONS}

        self._mic_body = self._session_body(
            row1, 0, 4, "Microphone", tooltip=SESSION_TOOLTIPS["microphone"]
        )
        self._mic_dropdown: ModernDropdown | None = None

        row2 = tk.Frame(info, bg=Theme.surface)
        row2.pack(fill=tk.X, pady=(2, 0))
        for column in range(5):
            row2.grid_columnconfigure(column, weight=1, uniform="session2")

        self._vad_body = self._session_body(row2, 0, 0, "VAD", tooltip=SESSION_TOOLTIPS["vad"])
        self._vad_var = tk.StringVar(value="on" if self.config.vad_filter else "off")
        SegmentedControl(
            self._vad_body,
            options=(("on", "On"), ("off", "Off")),
            variable=self._vad_var,
            command=self._on_vad_changed,
            font_family=self._ui_font,
        ).pack(anchor="w")

        self._beam_body = self._session_body(
            row2, 0, 1, "Beam size", tooltip=SESSION_TOOLTIPS["beam"]
        )
        self._beam_dropdown: ModernDropdown | None = None
        beam = self.config.beam_size if self.config.beam_size in BEAM_SIZE_CHOICES else DEFAULT_BEAM_SIZE
        self.config.beam_size = beam
        self._beam_dropdown = ModernDropdown(
            self._beam_body,
            options=[str(size) for size in BEAM_SIZE_CHOICES],
            selected=str(beam),
            on_select=self._on_beam_selected,
            font_family=self._ui_font,
            width=6,
        )
        self._beam_dropdown.pack(anchor="w")

        self._timestamps_body = self._session_body(
            row2, 0, 2, "Timestamps", tooltip=SESSION_TOOLTIPS["timestamps"]
        )
        self._timestamps_var = tk.StringVar(value="on" if self.config.word_timestamps else "off")
        SegmentedControl(
            self._timestamps_body,
            options=(("on", "On"), ("off", "Off")),
            variable=self._timestamps_var,
            command=self._on_timestamps_changed,
            font_family=self._ui_font,
        ).pack(anchor="w")

        self._enhance_body = self._session_body(
            row2, 0, 3, "Enhance", tooltip=SESSION_TOOLTIPS["enhance"]
        )
        from vocalforge.enhancement import is_enhancement_available

        enhance_ready = is_enhancement_available()
        if self.config.enhance_audio and not enhance_ready:
            self.config.enhance_audio = False
            save_config(self.config_path, self.config)
            self.controller.set_enhance_audio(False)
        self._enhance_var = tk.StringVar(value="on" if self.config.enhance_audio else "off")
        if enhance_ready:
            SegmentedControl(
                self._enhance_body,
                options=(("on", "On"), ("off", "Off")),
                variable=self._enhance_var,
                command=self._on_enhance_audio_changed,
                font_family=self._ui_font,
            ).pack(anchor="w")
        else:
            tk.Label(
                self._enhance_body,
                text="Not installed",
                fg=Theme.muted,
                bg=Theme.surface_2,
                font=(self._ui_font, 9),
                anchor="w",
            ).pack(fill=tk.X)

        self._diarize_body = self._session_body(
            row2, 0, 4, "Diarize", tooltip=SESSION_TOOLTIPS["diarize"]
        )
        from vocalforge.diarization import is_diarization_ready

        diarize_ready = is_diarization_ready(
            self.config.hf_token, models_root=self.paths.models
        )
        if self.config.diarize_speakers and not diarize_ready:
            self.config.diarize_speakers = False
            save_config(self.config_path, self.config)
            self.controller.set_diarize_speakers(False)
        self._diarize_var = tk.StringVar(value="on" if self.config.diarize_speakers else "off")
        if diarize_ready:
            SegmentedControl(
                self._diarize_body,
                options=(("on", "On"), ("off", "Off")),
                variable=self._diarize_var,
                command=self._on_diarize_speakers_changed,
                font_family=self._ui_font,
            ).pack(anchor="w")
        else:
            tk.Label(
                self._diarize_body,
                text="Not installed",
                fg=Theme.muted,
                bg=Theme.surface_2,
                font=(self._ui_font, 9),
                anchor="w",
            ).pack(fill=tk.X)

        row3 = tk.Frame(info, bg=Theme.surface)
        row3.pack(fill=tk.X, pady=(2, 0))
        for column in range(5):
            row3.grid_columnconfigure(column, weight=1, uniform="session3")

        self._translate_body = self._session_body(
            row3, 0, 0, "Translate", tooltip=SESSION_TOOLTIPS["translate"]
        )
        self._translate_var = tk.StringVar(
            value="on" if self.config.task == "translate" else "off"
        )
        self._refresh_translate_control()

        self._refresh_profile_label()
        self._refresh_mic_control()

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

        actions = tk.Frame(self.root, bg=Theme.bg)
        actions.pack(pady=8)
        self.upload_button = hoverable_button(
            actions,
            text="Upload Audio File",
            command=self.on_upload,
            font_family=self._ui_font,
            bg=Theme.idle_btn,
            fg=Theme.text,
            hover_bg=Theme.idle_btn_hover,
            padx=16,
            pady=9,
        )
        self.upload_button.pack(side=tk.LEFT, padx=(0, 8))

        self.cancel_button = hoverable_button(
            actions,
            text="Cancel",
            command=self.on_cancel,
            font_family=self._ui_font,
            bg=Theme.surface_2,
            fg=Theme.text,
            hover_bg=Theme.surface_3,
            padx=16,
            pady=9,
        )
        self.cancel_button.pack(side=tk.LEFT)
        self.cancel_button.configure(state="disabled")

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

        history_row = tk.Frame(text_frame, bg=Theme.surface)
        history_row.pack(fill=tk.X, padx=12, pady=(6, 0))
        history_label_row = tk.Frame(history_row, bg=Theme.surface)
        history_label_row.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(
            history_label_row,
            text="History",
            fg=Theme.muted,
            bg=Theme.surface,
            font=(self._ui_font, 8),
        ).pack(side=tk.LEFT)
        InfoTip(
            history_label_row,
            text=SESSION_TOOLTIPS["history"],
            font_family=self._ui_font,
            bg=Theme.surface,
        ).pack(side=tk.LEFT, padx=(4, 0))
        self._history_body = tk.Frame(history_row, bg=Theme.surface)
        self._history_body.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._history_dropdown: ModernDropdown | None = None
        self._history_paths: dict[str, Path] = {}

        actions = tk.Frame(text_frame, bg=Theme.surface)
        actions.pack(fill=tk.X, padx=12, pady=(6, 0))
        self.copy_button = hoverable_button(
            actions,
            text="Copy",
            command=self.on_copy_transcript,
            font_family=self._ui_font,
            bg=Theme.surface_2,
            fg=Theme.text,
            hover_bg=Theme.surface_3,
            padx=12,
            pady=6,
        )
        self.copy_button.pack(side=tk.LEFT, padx=(0, 6))
        self.save_button = hoverable_button(
            actions,
            text="Save As…",
            command=self.on_save_transcript,
            font_family=self._ui_font,
            bg=Theme.surface_2,
            fg=Theme.text,
            hover_bg=Theme.surface_3,
            padx=12,
            pady=6,
        )
        self.save_button.pack(side=tk.LEFT)

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
        self._refresh_history_control()

    def _session_cell(
        self, parent: tk.Frame, row: int, column: int, label: str, value: str
    ) -> tk.Label:
        body = self._session_body(parent, row, column, label)
        value_label = tk.Label(
            body,
            text=value,
            fg=Theme.text,
            bg=Theme.surface_2,
            font=(self._ui_font, 10, "bold"),
            anchor="w",
            wraplength=150,
            justify="left",
        )
        value_label.pack(fill=tk.X, pady=(4, 0))
        return value_label

    def _session_body(
        self,
        parent: tk.Frame,
        row: int,
        column: int,
        label: str,
        tooltip: str | None = None,
    ) -> tk.Frame:
        cell = tk.Frame(parent, bg=Theme.surface_2, padx=10, pady=8)
        cell.grid(row=row, column=column, sticky="nsew", padx=3, pady=3)
        header = tk.Frame(cell, bg=Theme.surface_2)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text=label.upper(),
            fg=Theme.muted,
            bg=Theme.surface_2,
            font=(self._ui_font, 8),
            anchor="w",
        ).pack(side=tk.LEFT)
        if tooltip:
            InfoTip(
                header,
                text=tooltip,
                font_family=self._ui_font,
                bg=Theme.surface_2,
            ).pack(side=tk.LEFT, padx=(4, 0))
        body = tk.Frame(cell, bg=Theme.surface_2)
        body.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        return body

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

    def _refresh_language_control(self) -> None:
        for child in self._language_body.winfo_children():
            child.destroy()
        self.language_value = None
        self._language_dropdown = None

        profile_id = self.config.active_profile
        profile = PROFILES.get(profile_id)
        if profile is not None and not profile.english_only:
            labels = [label for _mode_id, label, _lang, _task in LANGUAGE_OPTIONS]
            current = language_option_label(
                language_mode_from_config(self.config.language, self.config.task)
            )
            self._language_dropdown = ModernDropdown(
                self._language_body,
                options=labels,
                selected=current,
                on_select=self._on_home_language_selected,
                font_family=self._ui_font,
                width=10,
            )
            self._language_dropdown.pack(fill=tk.X)
            return

        self.language_value = tk.Label(
            self._language_body,
            text=self._language_display_label(),
            fg=Theme.text,
            bg=Theme.surface_2,
            font=(self._ui_font, 10, "bold"),
            anchor="w",
            wraplength=150,
            justify="left",
        )
        self.language_value.pack(fill=tk.X)

    def _on_home_language_selected(self, label: str) -> None:
        mode_id = self._language_labels.get(label, "auto")
        language, _default_task = language_settings_for_mode(mode_id)
        task = "translate" if self.config.task == "translate" else "transcribe"
        profile = PROFILES.get(self.config.active_profile)
        if profile is not None and profile.english_only:
            task = "transcribe"
        language, task = resolve_language_settings(
            profile or PROFILES["high_accuracy"],
            language=language,
            task=task,
        )
        self.config.language = language
        self.config.task = task
        save_config(self.config_path, self.config)
        self.controller.transcription.set_language_settings(language, task)
        if hasattr(self, "_translate_var"):
            self._translate_var.set("on" if task == "translate" else "off")
        if self._setup_win is not None:
            try:
                if self._setup_win.winfo_exists():
                    self._setup_win._language_mode.set(mode_id)
            except tk.TclError:
                pass
        self._set_status(f"Language set to {label}.")

    def _refresh_translate_control(self) -> None:
        body = getattr(self, "_translate_body", None)
        if body is None:
            return
        for child in body.winfo_children():
            child.destroy()
        profile = PROFILES.get(self.config.active_profile)
        can_translate = profile is not None and profile.supports_translate
        if self.config.task == "translate" and not can_translate:
            self.config.task = "transcribe"
            save_config(self.config_path, self.config)
            lang = self.config.language if profile and not profile.english_only else "en"
            self.controller.transcription.set_language_settings(lang, "transcribe")
        self._translate_var = tk.StringVar(
            value="on" if self.config.task == "translate" else "off"
        )
        if can_translate:
            SegmentedControl(
                body,
                options=(("on", "On"), ("off", "Off")),
                variable=self._translate_var,
                command=self._on_translate_changed,
                font_family=self._ui_font,
            ).pack(anchor="w")
        else:
            reason = "Use Multilingual Small"
            if profile is not None and profile.english_only:
                reason = "English-only model"
            elif profile is not None and profile.id == "high_accuracy":
                reason = "Not on turbo"
            tk.Label(
                body,
                text=reason,
                fg=Theme.muted,
                bg=Theme.surface_2,
                font=(self._ui_font, 9),
                anchor="w",
            ).pack(fill=tk.X)

    def _on_translate_changed(self) -> None:
        enabled = self._translate_var.get() == "on"
        profile = PROFILES.get(self.config.active_profile)
        if enabled and (profile is None or not profile.supports_translate):
            self._translate_var.set("off")
            messagebox.showwarning(
                "Translate Unavailable",
                "This profile’s model cannot translate to English.\n\n"
                "Activate Multilingual Small or Medium (faster-whisper-small / "
                "medium) for speech→English translation. High Accuracy "
                "(large-v3-turbo) does not support translate.",
            )
            return
        self.config.task = "translate" if enabled else "transcribe"
        language, task = resolve_language_settings(
            profile or PROFILES["high_accuracy"],
            language=self.config.language,
            task=self.config.task,
        )
        self.config.language = language
        self.config.task = task
        save_config(self.config_path, self.config)
        self.controller.transcription.set_language_settings(language, task)
        self._set_status(
            "Translate to English enabled." if task == "translate" else "Transcription mode (no translate)."
        )

    def _refresh_mic_control(self) -> None:
        for child in self._mic_body.winfo_children():
            child.destroy()
        self._mic_dropdown = None
        try:
            devices = list_input_devices()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Microphone list unavailable: %s", exc)
            devices = []
        labels = [device.label for device in devices] or [DEFAULT_INPUT_LABEL]
        saved = self.config.input_device or DEFAULT_INPUT_LABEL
        selected = saved if saved in labels else DEFAULT_INPUT_LABEL
        self._mic_dropdown = ModernDropdown(
            self._mic_body,
            options=labels,
            selected=selected,
            on_select=self._on_mic_selected,
            font_family=self._ui_font,
            width=12,
        )
        self._mic_dropdown.pack(fill=tk.X)

    def _on_mic_selected(self, label: str) -> None:
        if self._current_state is AppState.RECORDING:
            self._set_status("Cannot change microphone while recording.", fg=Theme.warning)
            self._refresh_mic_control()
            return
        saved = None if label == DEFAULT_INPUT_LABEL else label
        try:
            self.controller.set_input_device(saved)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Microphone", str(exc))
            self._refresh_mic_control()
            return
        self.config.input_device = saved
        save_config(self.config_path, self.config)
        self._set_status(f"Microphone: {label}")

    def _on_vad_changed(self) -> None:
        enabled = self._vad_var.get() == "on"
        self.config.vad_filter = enabled
        save_config(self.config_path, self.config)
        self.controller.transcription.set_vad_filter(enabled)
        self._set_status(f"Voice activity filter {'enabled' if enabled else 'disabled'}.")

    def _on_enhance_audio_changed(self) -> None:
        enabled = self._enhance_var.get() == "on"
        from vocalforge.enhancement import is_enhancement_available

        if enabled and not is_enhancement_available():
            self._enhance_var.set("off")
            messagebox.showwarning(
                "Enhancement Unavailable",
                "deep-filter and soxr are required. See requirements-extras.txt.",
            )
            return
        self.config.enhance_audio = enabled
        save_config(self.config_path, self.config)
        self.controller.set_enhance_audio(enabled)
        self._set_status(f"Audio enhancement {'enabled' if enabled else 'disabled'}.")

    def _on_diarize_speakers_changed(self) -> None:
        enabled = self._diarize_var.get() == "on"
        from vocalforge.diarization import is_diarization_ready

        if enabled and not is_diarization_ready(
            self.config.hf_token, models_root=self.paths.models
        ):
            self._diarize_var.set("off")
            messagebox.showwarning(
                "Diarization Unavailable",
                "Offline diarization weights are missing.\n"
                "Run: PYTHONPATH=. python scripts/vendor_diarization_models.py",
            )
            return
        self.config.diarize_speakers = enabled
        save_config(self.config_path, self.config)
        self.controller.set_diarize_speakers(enabled)
        self._set_status(f"Speaker diarization {'enabled' if enabled else 'disabled'}.")

    def _on_device_preference_changed(self) -> None:
        choice = self._device_var.get()
        if choice == "cuda" and not cuda_is_usable(self.capabilities):
            self._device_var.set("cpu")
            choice = "cpu"
            messagebox.showwarning("GPU Unavailable", "CUDA is not usable on this machine. Staying on CPU.")
        self.config.preferred_device = choice
        save_config(self.config_path, self.config)
        self._apply_device_to_active_profile(choice)

    def _apply_device_to_active_profile(self, preferred: str) -> None:
        """Reload the active profile when the processing device changes."""
        profile_id = self.config.active_profile
        if profile_id not in PROFILES:
            self._set_status(f"Device preference saved: {preferred.upper()}.")
            return
        profile = PROFILES[profile_id]
        runtime = resolve_runtime(profile, self.capabilities, preferred_device=preferred)
        current = (
            self.controller.transcription.device
            if self.controller.transcription.is_ready
            else None
        )
        if current == runtime.device:
            self._set_status(f"Already using {self._device_display_label(runtime.device)}.")
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
            self._set_status(f"Switching to {self._device_display_label(runtime.device)}…")

    def _on_beam_selected(self, label: str) -> None:
        try:
            beam = int(label)
        except ValueError:
            beam = DEFAULT_BEAM_SIZE
        self.config.beam_size = beam
        save_config(self.config_path, self.config)
        self.controller.transcription.set_decode_options(beam_size=beam)
        self._set_status(f"Beam size set to {beam}.")

    def _on_timestamps_changed(self) -> None:
        enabled = self._timestamps_var.get() == "on"
        self.config.word_timestamps = enabled
        save_config(self.config_path, self.config)
        self.controller.transcription.set_decode_options(word_timestamps=enabled)
        self._set_status(f"Word timestamps {'enabled' if enabled else 'disabled'}.")

    def _refresh_history_control(self) -> None:
        for child in self._history_body.winfo_children():
            child.destroy()
        self._history_dropdown = None
        self._history_paths = {}
        paths = list_transcripts(self.paths)
        if not paths:
            tk.Label(
                self._history_body,
                text="No saved transcripts yet",
                fg=Theme.muted,
                bg=Theme.surface,
                font=(self._ui_font, 9),
                anchor="w",
            ).pack(fill=tk.X)
            return
        labels: list[str] = []
        for path in paths:
            label = transcript_label(path)
            # Disambiguate collisions in the dropdown.
            if label in self._history_paths:
                label = f"{label} ({path.name})"
            self._history_paths[label] = path
            labels.append(label)
        self._history_dropdown = ModernDropdown(
            self._history_body,
            options=labels,
            selected=labels[0],
            on_select=self._on_history_selected,
            font_family=self._ui_font,
            width=22,
        )
        self._history_dropdown.pack(fill=tk.X)

    def _on_history_selected(self, label: str) -> None:
        path = self._history_paths.get(label)
        if path is None or not path.exists():
            self._set_status("Transcript file missing.", fg=Theme.warning)
            self._refresh_history_control()
            return
        try:
            text = read_transcript(path)
        except OSError as exc:
            messagebox.showerror("History", str(exc))
            return
        self._set_transcript_display(text)
        self._set_status(f"Loaded {path.name}")

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

        preferred = (self.config.preferred_device or "cpu").lower()
        if preferred == "cuda" and not cuda_is_usable(self.capabilities):
            preferred = "cpu"
        if preferred in {"cpu", "cuda"} and self._device_var.get() != preferred:
            self._device_var.set(preferred)
        self._refresh_language_control()
        self._refresh_translate_control()

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

    def on_cancel(self) -> None:
        if self.controller.cancel():
            self._set_status("Cancelling…")

    def on_upload(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=UPLOAD_FILEDIALOG_TYPES)
        if not file_path:
            return
        if not is_supported_upload(file_path):
            messagebox.showerror(
                "Upload",
                "Unsupported audio format. Try WAV, MP3, M4A, FLAC, OGG, or WebM.",
            )
            return
        self.controller.transcribe_upload(file_path)

    def on_copy_transcript(self) -> None:
        if not self._last_transcript:
            self._set_status("Nothing to copy.", fg=Theme.warning)
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self._last_transcript)
        self._set_status("Copied to clipboard.")

    def on_save_transcript(self) -> None:
        if not self._last_transcript:
            self._set_status("Nothing to save.", fg=Theme.warning)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
            initialfile="transcript.txt",
        )
        if not path:
            return
        try:
            Path(path).write_text(self._last_transcript, encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save", str(exc))
            return
        self._set_status(f"Saved {Path(path).name}")
        self._refresh_history_control()

    def _apply_control_state(self, state: AppState) -> None:
        self._current_state = state
        ready = state is AppState.READY
        recording = state is AppState.RECORDING
        transcribing = state is AppState.TRANSCRIBING
        can_select = state in {AppState.NO_MODEL, AppState.READY, AppState.ERROR}
        self.setup_button.configure(state=("normal" if can_select else "disabled"))
        self.upload_button.configure(state=("normal" if ready else "disabled"))
        self.cancel_button.configure(
            state=("normal" if recording or transcribing else "disabled")
        )
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
            self._refresh_history_control()
            self._set_status("Transcription Completed")
        elif event.type is EventType.TRANSCRIPTION_CANCELLED:
            self._set_status(payload.get("message", "Cancelled."), fg=Theme.warning)
        elif event.type is EventType.PROGRESS:
            percent = payload.get("percent")
            if percent is not None:
                self._set_status(f"Transcribing… {percent}%")
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
