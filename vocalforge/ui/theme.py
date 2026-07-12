"""Modern dark UI theme helpers for VocalForge (Tkinter)."""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Sequence


class Theme:
    """Charcoal / teal palette — neo-grotesque UI chrome for Tk."""

    bg = "#12141A"
    surface = "#1A1D26"
    surface_2 = "#242833"
    surface_3 = "#2C3140"
    border = "#3A4052"
    text = "#F2F3F5"
    muted = "#9AA3B2"
    accent = "#2DD4BF"
    accent_hover = "#5EEAD4"
    accent_text = "#042F2E"
    danger = "#F07178"
    warning = "#E6C07B"
    success = "#6BCB8B"
    success_text = "#052E16"
    idle_btn = "#3A4052"
    idle_btn_hover = "#4A5166"


def font_tuple(family: str, size: int, bold: bool = False) -> tuple:
    return (family, size, "bold") if bold else (family, size)


def hoverable_button(
    parent: tk.Misc,
    *,
    text: str,
    command: Callable[[], None],
    font_family: str,
    bg: str,
    fg: str,
    hover_bg: str | None = None,
    active_fg: str | None = None,
    padx: int = 14,
    pady: int = 8,
    disabledforeground: str | None = None,
) -> tk.Button:
    hover = hover_bg or bg
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        font=font_tuple(font_family, 10),
        bg=bg,
        fg=fg,
        activebackground=hover,
        activeforeground=active_fg or fg,
        disabledforeground=disabledforeground or Theme.muted,
        relief="flat",
        bd=0,
        padx=padx,
        pady=pady,
        cursor="hand2",
        highlightthickness=0,
    )

    def on_enter(_event=None) -> None:
        if str(btn.cget("state")) != "disabled":
            btn.configure(bg=hover)

    def on_leave(_event=None) -> None:
        if str(btn.cget("state")) != "disabled":
            btn.configure(bg=bg)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn


class Tooltip:
    """Hover tooltip shown near a widget."""

    def __init__(
        self,
        widget: tk.Misc,
        text: str,
        *,
        font_family: str,
        delay_ms: int = 350,
        wraplength: int = 260,
    ) -> None:
        self._widget = widget
        self._text = text
        self._font_family = font_family
        self._delay_ms = delay_ms
        self._wraplength = wraplength
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self._widget.after(self._delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self._widget.after_cancel(self._after_id)
            except Exception:  # noqa: BLE001
                pass
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._tip is not None or not self._text:
            return
        try:
            if not self._widget.winfo_exists():
                return
        except tk.TclError:
            return
        tip = tk.Toplevel(self._widget)
        tip.wm_overrideredirect(True)
        tip.configure(bg=Theme.border)
        try:
            tip.attributes("-topmost", True)
        except tk.TclError:
            pass
        frame = tk.Frame(tip, bg=Theme.surface_3, padx=1, pady=1)
        frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            frame,
            text=self._text,
            justify=tk.LEFT,
            bg=Theme.surface_3,
            fg=Theme.text,
            font=font_tuple(self._font_family, 9),
            wraplength=self._wraplength,
            padx=10,
            pady=8,
        ).pack()
        tip.update_idletasks()
        x = self._widget.winfo_rootx() + 12
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 6
        tip.geometry(f"+{x}+{y}")
        self._tip = tip

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._tip = None


class InfoTip(tk.Label):
    """Compact (i) control that shows a tooltip on hover."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        text: str,
        font_family: str,
        bg: str = Theme.surface_2,
    ) -> None:
        super().__init__(
            parent,
            text="(i)",
            fg=Theme.accent,
            bg=bg,
            font=font_tuple(font_family, 8, bold=True),
            cursor="question_arrow",
            padx=2,
        )
        Tooltip(self, text, font_family=font_family)


class ModernDropdown(tk.Frame):
    """Flat trigger + popup list — replaces ttk.Combobox."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        options: Sequence[str],
        selected: str,
        on_select: Callable[[str], None],
        font_family: str,
        width: int = 18,
    ) -> None:
        super().__init__(parent, bg=Theme.surface_2, highlightbackground=Theme.border, highlightthickness=1)
        # Do not name this `_options` — Tkinter.Widget uses that method name.
        self._choices = list(options)
        self._on_select = on_select
        self._font_family = font_family
        self._popup: tk.Toplevel | None = None
        self._selected = selected if selected in self._choices else (self._choices[0] if self._choices else "")

        inner = tk.Frame(self, bg=Theme.surface_2)
        inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self._label = tk.Label(
            inner,
            text=self._selected,
            bg=Theme.surface_2,
            fg=Theme.text,
            font=font_tuple(font_family, 10),
            anchor="w",
            padx=10,
            pady=7,
            width=width,
        )
        self._label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chevron = tk.Label(
            inner,
            text="▾",
            bg=Theme.surface_2,
            fg=Theme.muted,
            font=font_tuple(font_family, 10),
            padx=8,
            pady=7,
        )
        chevron.pack(side=tk.RIGHT)
        for widget in (self, inner, self._label, chevron):
            widget.bind("<Button-1>", self._toggle)
            widget.configure(cursor="hand2")

    @property
    def selected(self) -> str:
        return self._selected

    def set_selected(self, value: str) -> None:
        if value in self._choices:
            self._selected = value
            self._label.configure(text=value)

    def _toggle(self, _event=None) -> None:
        if self._popup is not None and self._popup.winfo_exists():
            self._close_popup()
            return
        self._open_popup()

    def _open_popup(self) -> None:
        self.update_idletasks()
        popup = tk.Toplevel(self)
        popup.withdraw()
        popup.overrideredirect(True)
        popup.configure(bg=Theme.border)
        popup.attributes("-topmost", True)
        shell = tk.Frame(popup, bg=Theme.surface, highlightthickness=0)
        shell.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        for option in self._choices:
            row = tk.Frame(shell, bg=Theme.surface)
            row.pack(fill=tk.X)
            is_selected = option == self._selected
            label = tk.Label(
                row,
                text=option,
                bg=Theme.surface_3 if is_selected else Theme.surface,
                fg=Theme.accent if is_selected else Theme.text,
                font=font_tuple(self._font_family, 10, bold=is_selected),
                anchor="w",
                padx=12,
                pady=8,
            )
            label.pack(fill=tk.X)

            def on_enter(_e=None, lbl=label, selected=is_selected) -> None:
                if not selected:
                    lbl.configure(bg=Theme.surface_2)

            def on_leave(_e=None, lbl=label, selected=is_selected) -> None:
                if not selected:
                    lbl.configure(bg=Theme.surface)

            def on_click(_e=None, value=option) -> None:
                self.set_selected(value)
                self._close_popup()
                self._on_select(value)

            for widget in (row, label):
                widget.bind("<Enter>", on_enter)
                widget.bind("<Leave>", on_leave)
                widget.bind("<Button-1>", on_click)
                widget.configure(cursor="hand2")

        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        width = max(self.winfo_width(), 160)
        popup.geometry(f"{width}x{min(320, 36 * len(self._choices) + 4)}+{x}+{y}")
        popup.deiconify()
        popup.focus_force()
        popup.bind("<FocusOut>", lambda _e: self.after(120, self._close_if_unfocused))
        popup.bind("<Escape>", lambda _e: self._close_popup())
        self._popup = popup

    def _close_if_unfocused(self) -> None:
        if self._popup is None or not self._popup.winfo_exists():
            return
        try:
            focused = self._popup.focus_get()
        except Exception:  # noqa: BLE001
            focused = None
        if focused is None:
            self._close_popup()

    def _close_popup(self) -> None:
        if self._popup is not None:
            try:
                self._popup.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._popup = None


class SegmentedControl(tk.Frame):
    """Modern CPU / GPU style toggle."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        options: Sequence[tuple[str, str]],
        variable: tk.StringVar,
        command: Callable[[], None],
        font_family: str,
        disabled_values: set[str] | None = None,
    ) -> None:
        super().__init__(parent, bg=Theme.surface_2, highlightbackground=Theme.border, highlightthickness=1)
        self._variable = variable
        self._command = command
        self._font_family = font_family
        self._disabled = disabled_values or set()
        self._buttons: dict[str, tk.Label] = {}
        inner = tk.Frame(self, bg=Theme.surface_2)
        inner.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        for value, label in options:
            chip = tk.Label(
                inner,
                text=label,
                font=font_tuple(font_family, 10, bold=True),
                padx=16,
                pady=8,
                cursor="hand2",
            )
            chip.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1, pady=1)
            chip.bind("<Button-1>", lambda _e, v=value: self._select(v))
            self._buttons[value] = chip
        self._variable.trace_add("write", lambda *_args: self._refresh())
        self._refresh()

    def _select(self, value: str) -> None:
        if value in self._disabled:
            return
        if self._variable.get() == value:
            return
        self._variable.set(value)
        self._command()

    def _refresh(self) -> None:
        current = self._variable.get()
        for value, chip in self._buttons.items():
            disabled = value in self._disabled
            selected = value == current and not disabled
            if disabled:
                chip.configure(bg=Theme.surface_2, fg=Theme.muted, cursor="arrow")
            elif selected:
                chip.configure(bg=Theme.accent, fg=Theme.accent_text, cursor="hand2")
            else:
                chip.configure(bg=Theme.surface, fg=Theme.text, cursor="hand2")


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    color = value.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def make_record_button_images(master: tk.Misc, size: int = 88):
    """Anti-aliased idle/recording button images for the main window."""
    from PIL import Image, ImageDraw, ImageTk

    def render(ring_color: str):
        scale = 4
        canvas = size * scale
        img = Image.new("RGB", (canvas, canvas), _hex_to_rgb(Theme.bg))
        draw = ImageDraw.Draw(img)
        halo = 6 * scale
        draw.ellipse((halo, halo, canvas - halo, canvas - halo), fill=_hex_to_rgb(Theme.surface_2))
        ring = 18 * scale
        draw.ellipse((ring, ring, canvas - ring, canvas - ring), fill=_hex_to_rgb(ring_color))
        hole = 32 * scale
        draw.ellipse((hole, hole, canvas - hole, canvas - hole), fill=_hex_to_rgb(Theme.bg))
        return img.resize((size, size), Image.Resampling.LANCZOS)

    idle = ImageTk.PhotoImage(render(Theme.success), master=master)
    recording = ImageTk.PhotoImage(render(Theme.danger), master=master)
    return idle, recording
