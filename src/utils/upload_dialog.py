"""
utils/upload_dialog.py

Small pop-up for configuring filebin.net upload and optional shutdown.
All tk variables are owned by BenchmarkGUI and passed in — so closing
the dialog does NOT lose the settings.
"""

import webbrowser
import tkinter as tk
from tkinter import ttk


class UploadDialog(tk.Toplevel):
    """
    Settings window for remote upload.
    Receives already-created tk variables from BenchmarkGUI so that
    closing the window never resets the configuration.
    """

    def __init__(
        self,
        parent: tk.Tk,
        bin_name_var: tk.StringVar,
        upload_var: tk.BooleanVar,
        shutdown_var: tk.BooleanVar,
        shutdown_delay_var: tk.IntVar,
    ):
        super().__init__(parent)
        self.title("Remote Upload Settings")
        self.resizable(False, False)

        # Variables are owned by BenchmarkGUI
        self._bin_name_var       = bin_name_var
        self._upload_var         = upload_var
        self._shutdown_var       = shutdown_var
        self._shutdown_delay_var = shutdown_delay_var

        self._build()
        self._center(parent)
        # Do NOT grab_set — let user interact with main window freely

    # -----------------------------------------------------------------------

    def _build(self) -> None:
        # ---- Bin name ----
        f = ttk.LabelFrame(self, text="Shared Bin (filebin.net)", padding=8)
        f.pack(fill=tk.X, padx=12, pady=(12, 4))

        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Bin name:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._bin_name_var, width=28).pack(side=tk.LEFT, padx=(6, 0))

        row2 = ttk.Frame(f)
        row2.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(row2, text="URL:", foreground="gray").pack(side=tk.LEFT)
        self._url_label = ttk.Label(
            row2,
            text=self._url(),
            foreground="blue",
            cursor="hand2",
            font=("Arial", 9, "underline"),
        )
        self._url_label.pack(side=tk.LEFT, padx=(4, 0))
        self._url_label.bind("<Button-1>", lambda _: webbrowser.open(self._url()))
        self._bin_name_var.trace_add("write", lambda *_: self._url_label.config(text=self._url()))

        ttk.Label(
            f,
            text="Use the same bin name on every PC — all results land in one place.",
            foreground="gray", font=("Arial", 8),
        ).pack(anchor=tk.W, pady=(4, 0))

        # ---- Upload toggle ----
        f2 = ttk.LabelFrame(self, text="Upload", padding=8)
        f2.pack(fill=tk.X, padx=12, pady=4)

        ttk.Checkbutton(
            f2,
            text="Upload JSON to filebin.net after benchmark completes",
            variable=self._upload_var,
        ).pack(anchor=tk.W)

        # ---- Shutdown toggle ----
        f3 = ttk.LabelFrame(self, text="Shutdown", padding=8)
        f3.pack(fill=tk.X, padx=12, pady=4)

        row3 = ttk.Frame(f3)
        row3.pack(fill=tk.X)
        ttk.Checkbutton(
            row3,
            text="Shut down PC after successful upload",
            variable=self._shutdown_var,
            command=self._on_shutdown_toggle,
        ).pack(side=tk.LEFT)

        ttk.Label(row3, text="  Delay (s):").pack(side=tk.LEFT)
        self._delay_spin = ttk.Spinbox(
            row3,
            from_=10, to=300,
            textvariable=self._shutdown_delay_var,
            width=5,
            state="normal" if self._shutdown_var.get() else tk.DISABLED,
        )
        self._delay_spin.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(
            f3,
            text="Cancel shutdown anytime:  shutdown /a  (CMD)",
            foreground="gray", font=("Arial", 8),
        ).pack(anchor=tk.W, pady=(4, 0))

        ttk.Label(
            f3,
            text="WARNING: Shutdown only triggers after a successful upload.",
            foreground="orange", font=("Arial", 8),
        ).pack(anchor=tk.W)

        # ---- Buttons ----
        btn_row = ttk.Frame(self)
        btn_row.pack(pady=(8, 12))
        ttk.Button(btn_row, text="Close", command=self.destroy).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            btn_row, text="Open bin in browser",
            command=lambda: webbrowser.open(self._url()),
        ).pack(side=tk.LEFT, padx=6)

    # -----------------------------------------------------------------------

    def _url(self) -> str:
        name = self._bin_name_var.get().strip()
        return f"https://filebin.net/{name}" if name else "https://filebin.net/"

    def _on_shutdown_toggle(self) -> None:
        if self._shutdown_var.get():
            self._delay_spin.config(state="normal")
            self._upload_var.set(True)
        else:
            self._delay_spin.config(state=tk.DISABLED)

    def _center(self, parent: tk.Tk) -> None:
        self.update_idletasks()
        pw = parent.winfo_x() + parent.winfo_width()  // 2
        ph = parent.winfo_y() + parent.winfo_height() // 2
        self.geometry(f"+{pw - self.winfo_width() // 2}+{ph - self.winfo_height() // 2}")