"""
Remote upload and shutdown settings dialog

Small pop-up window for configuring filebin.net upload and optional PC
shutdown after the benchmark completes. All tk variables are owned by
BenchmarkGUI and passed in, closing the dialog does NOT reset the settings.
"""

import tkinter as tk
import webbrowser
from tkinter import ttk


class UploadDialog(tk.Toplevel):
    """
    Settings window for remote upload and optional post-benchmark shutdown.

    Receives already-created tk variables from BenchmarkGUI so that closing
    the window never resets the configuration.  The dialog does not use
    grab_set() so the user can still interact with the main window while it
    is open.

    Attributes:
        _bin_name_var: Shared StringVar for the filebin.net bin name.
        _upload_var: Shared BooleanVar: True when upload is enabled.
        _shutdown_var: Shared BooleanVar: True when shutdown after upload is enabled.
        _shutdown_delay_var: Shared IntVar: seconds before shutdown executes.
    """

    def __init__(
        self,
        parent: tk.Tk,
        bin_name_var:       tk.StringVar,
        upload_var:         tk.BooleanVar,
        shutdown_var:       tk.BooleanVar,
        shutdown_delay_var: tk.IntVar,
    ):
        """Initialise and display the dialog.

        Args:
            parent: The parent Tk window.
            bin_name_var: Shared bin name variable (owned by BenchmarkGUI).
            upload_var: Shared upload-enabled variable.
            shutdown_var: Shared shutdown-after-upload variable.
            shutdown_delay_var: Shared shutdown delay variable (seconds).
        """
        super().__init__(parent)
        self.title("Remote Upload Settings")
        self.resizable(False, False)

        self._bin_name_var       = bin_name_var
        self._upload_var         = upload_var
        self._shutdown_var       = shutdown_var
        self._shutdown_delay_var = shutdown_delay_var

        self._build()
        self._center(parent)

    # UI construction
    def _build(self) -> None:
        """Construct all dialog panels."""
        self._build_bin_panel()
        self._build_upload_panel()
        self._build_shutdown_panel()
        self._build_buttons()

    def _build_bin_panel(self) -> None:
        """Build the bin-name entry and live URL preview."""
        f = ttk.LabelFrame(self, text="Shared Bin (filebin.net)", padding=8)
        f.pack(fill=tk.X, padx=12, pady=(12, 4))

        # Bin name entry row.
        row = ttk.Frame(f)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Bin name:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self._bin_name_var, width=28).pack(side=tk.LEFT, padx=(6, 0))

        # Live URL label that updates as the bin name is typed.
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

        # Trace the bin name variable so the URL label updates on every keystroke.
        self._bin_name_var.trace_add("write", lambda *_: self._url_label.config(text=self._url()))

        ttk.Label(
            f,
            text="Use the same bin name on every PC: all results land in one place.",
            foreground="gray", font=("Arial", 8),
        ).pack(anchor=tk.W, pady=(4, 0))

    def _build_upload_panel(self) -> None:
        """Build the upload enable/disable toggle."""
        f2 = ttk.LabelFrame(self, text="Upload", padding=8)
        f2.pack(fill=tk.X, padx=12, pady=4)

        ttk.Checkbutton(
            f2,
            text="Upload JSON to filebin.net after benchmark completes",
            variable=self._upload_var,
        ).pack(anchor=tk.W)

    def _build_shutdown_panel(self) -> None:
        """Build the shutdown toggle and delay spinbox."""
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
            # Start disabled; _on_shutdown_toggle() enables it when the checkbox is ticked.
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

    def _build_buttons(self) -> None:
        """Build the Close and Open-in-browser buttons."""
        btn_row = ttk.Frame(self)
        btn_row.pack(pady=(8, 12))
        ttk.Button(btn_row, text="Close", command=self.destroy).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            btn_row,
            text="Open bin in browser",
            command=lambda: webbrowser.open(self._url()),
        ).pack(side=tk.LEFT, padx=6)

    # Helpers
    def _url(self) -> str:
        """
        Return the full filebin.net URL for the current bin name.

        Returns:
            URL string, or the filebin.net root when the bin name is empty.
        """
        name = self._bin_name_var.get().strip()
        return f"https://filebin.net/{name}" if name else "https://filebin.net/"

    def _on_shutdown_toggle(self) -> None:
        """
        Enable the delay spinbox when shutdown is checked, disable when unchecked.

        Also auto-enables upload when shutdown is activated, because shutdown
        without upload would be an accidental data loss scenario.
        """
        if self._shutdown_var.get():
            self._delay_spin.config(state="normal")
            self._upload_var.set(True)   # shutdown requires a successful upload
        else:
            self._delay_spin.config(state=tk.DISABLED)

    def _center(self, parent: tk.Tk) -> None:
        """
        Centre this dialog over the parent window.

        Args:
            parent: The parent Tk window to centre relative to.
        """
        self.update_idletasks()
        pw = parent.winfo_x() + parent.winfo_width()  // 2
        ph = parent.winfo_y() + parent.winfo_height() // 2
        self.geometry(f"+{pw - self.winfo_width() // 2}+{ph - self.winfo_height() // 2}")