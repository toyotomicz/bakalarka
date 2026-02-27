"""
Reusable GUI widget components for the benchmark application.
utils/gui_widgets.py

Widgets:
  ImageSelectionWidget      : Listbox + buttons for adding / removing source images.
  CompressorSelectionWidget : Grid of checkboxes for toggling compressors.
  LevelSelectionWidget      : Compression-level toggles + verification / strip options.
  VerificationResultsWidget : Read-only table showing lossless-verification outcomes.
"""

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Dict, List, Optional


class ImageSelectionWidget:
    """
    Compound widget for building the list of images to benchmark.

    Public buttons (add_images_btn, add_folder_btn, clear_btn) are created but
    left without commands so the parent window can bind its own handlers.
    """

    def __init__(self, parent: tk.Widget, on_update: Optional[Callable] = None):
        self.parent    = parent
        self.on_update = on_update
        self.selected_images: List = []

        self.frame = ttk.LabelFrame(parent, text="Images", padding="10")
        self._build_widgets()

    def _build_widgets(self) -> None:
        # ---- Button row ----
        btn_row = ttk.Frame(self.frame)
        btn_row.pack(fill=tk.X, pady=(0, 5))

        self.add_images_btn = ttk.Button(btn_row, text="Add Images...")
        self.add_images_btn.pack(side=tk.LEFT, padx=5)

        self.add_folder_btn = ttk.Button(btn_row, text="Add Folder...")
        self.add_folder_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(btn_row, text="Clear All")
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # ---- Image listbox ----
        self.listbox = tk.Listbox(self.frame, height=6, cursor="hand2")
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<Double-Button-1>", self._on_double_click)

        # ---- Count label ----
        self.count_label = ttk.Label(self.frame, text="0 images selected")
        self.count_label.pack(anchor=tk.W, pady=(5, 0))

    # -- Public API --

    def add_image(self, path) -> None:
        """Add path to the selection list (duplicates are silently ignored)."""
        if path not in self.selected_images:
            self.selected_images.append(path)
            self.listbox.insert(tk.END, path.name)
            self._update_count()

    def clear_images(self) -> None:
        """Remove all images from the selection."""
        self.selected_images.clear()
        self.listbox.delete(0, tk.END)
        self._update_count()

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    # -- Private helpers --

    def _update_count(self) -> None:
        n = len(self.selected_images)
        self.count_label.config(
            text=f"{n} image{'s' if n != 1 else ''} selected"
        )
        if self.on_update:
            self.on_update()

    def _on_double_click(self, event) -> None:
        """Open a preview window for the double-clicked image."""
        selection = self.listbox.curselection()
        if selection:
            self._show_preview(self.selected_images[selection[0]])

    def _show_preview(self, image_path) -> None:
        """Display a resized preview of image_path in a Toplevel window."""
        try:
            from PIL import Image, ImageTk

            preview = tk.Toplevel(self.parent)
            preview.title(f"Preview: {image_path.name}")
            # Fixed, sensible window size (original had typo: height=6000)
            preview.geometry("800x620")

            img = Image.open(image_path)
            original_size = img.size

            # Scale down to fit inside the preview area.
            img.thumbnail((750, 550), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            label = ttk.Label(preview, image=photo)
            label.image = photo   # Prevent garbage collection of the PhotoImage.
            label.pack(padx=10, pady=10)

            ttk.Label(
                preview,
                text=f"Dimensions: {original_size[0]} × {original_size[1]}",
                font=("Arial", 9),
            ).pack(pady=5)

            ttk.Button(preview, text="Close", command=preview.destroy).pack(pady=5)

        except Exception as exc:
            messagebox.showerror("Preview Error", f"Cannot preview image:\n{exc}")


class CompressorSelectionWidget:
    """
    Grid of checkboxes — one per registered compressor.
    All compressors are selected by default.
    """

    def __init__(self, parent: tk.Widget, compressors: List[str]):
        self.parent     = parent
        self.compressors = compressors
        self.compressor_vars: Dict[str, tk.BooleanVar] = {}

        self.frame = ttk.LabelFrame(parent, text="Compressors", padding="10")
        self._build_widgets()

    def _build_widgets(self) -> None:
        if not self.compressors:
            ttk.Label(
                self.frame,
                text="No compressors found. Check the plugins folder.",
                foreground="red",
            ).pack()
            return

        # Lay out checkboxes in a balanced grid (3 columns).
        cols      = 3
        num_items = len(self.compressors)
        rows      = (num_items + cols - 1) // cols

        for idx, name in enumerate(sorted(self.compressors)):
            row = idx % rows
            col = idx // rows

            var = tk.BooleanVar(value=True)
            self.compressor_vars[name] = var
            ttk.Checkbutton(self.frame, text=name, variable=var).grid(
                row=row, column=col, sticky=tk.W, padx=5, pady=2
            )

        # Select / Deselect All buttons below the grid.
        btn_row = ttk.Frame(self.frame)
        btn_row.grid(row=rows + 1, column=0, columnspan=cols, pady=(10, 0))

        ttk.Button(btn_row, text="Select All",   command=self.select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Deselect All", command=self.deselect_all).pack(side=tk.LEFT, padx=5)

    # -- Public API --

    def get_selected(self) -> List[str]:
        """Return the names of all currently checked compressors."""
        return [name for name, var in self.compressor_vars.items() if var.get()]

    def select_all(self) -> None:
        for var in self.compressor_vars.values():
            var.set(True)

    def deselect_all(self) -> None:
        for var in self.compressor_vars.values():
            var.set(False)

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)


class LevelSelectionWidget:
    """
    Row of compression-level checkboxes plus verification / metadata-strip toggles.

    levels should be a list of (display_name, CompressionLevel) tuples, e.g.:
        [("FASTEST", CompressionLevel.FASTEST), ...]
    """

    def __init__(self, parent: tk.Widget, levels: List[tuple]):
        self.parent = parent
        self.levels = levels
        self.level_vars: Dict = {}

        self.frame = ttk.LabelFrame(parent, text="Compression Levels", padding="10")
        self._build_widgets()

    def _build_widgets(self) -> None:
        # One checkbox per level in a single row.
        for idx, (name, level) in enumerate(self.levels):
            var = tk.BooleanVar(value=False)
            self.level_vars[level] = var
            ttk.Checkbutton(self.frame, text=name, variable=var).grid(
                row=0, column=idx, padx=10
            )

        # Additional options below the level row.
        options = ttk.Frame(self.frame)
        options.grid(row=1, column=0, columnspan=len(self.levels), pady=(10, 0))

        self.verify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options,
            text="Verify lossless compression (pixel-by-pixel comparison)",
            variable=self.verify_var,
        ).pack(anchor=tk.W)

        self.strip_metadata_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options,
            text="Strip metadata (EXIF, XMP, etc.) before compression",
            variable=self.strip_metadata_var,
        ).pack(anchor=tk.W)

    # -- Public API --

    def get_selected(self) -> List:
        """Return the CompressionLevel values for all checked boxes."""
        return [level for level, var in self.level_vars.items() if var.get()]

    def is_verification_enabled(self) -> bool:
        return self.verify_var.get()

    def is_strip_metadata_enabled(self) -> bool:
        return self.strip_metadata_var.get()

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)


class VerificationResultsWidget:
    """
    Modal Toplevel window that shows lossless verification results in a table.

    verification_results is a dict keyed by (image_name, compressor_name)
    with VerificationResult values.
    """

    def __init__(self, parent: tk.Widget, verification_results: Dict):
        self.window               = tk.Toplevel(parent)
        self.window.title("Verification Results Summary")
        self.window.geometry("1000x600")
        self.verification_results = verification_results
        self._build_widgets()

    def _build_widgets(self) -> None:
        # ---- Results table ----
        tree_frame = ttk.Frame(self.window, padding="10")
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = (
            "Image", "Compressor", "Result",
            "Max Difference", "Different Pixels", "Accuracy %",
        )
        column_widths = {
            "Image":            200,
            "Compressor":       150,
            "Result":           100,
            "Max Difference":   120,
            "Different Pixels": 150,
            "Accuracy %":       120,
        }

        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths.get(col, 100))

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Colour coding: green for lossless, red for lossy.
        self.tree.tag_configure("lossless", foreground="dark green")
        self.tree.tag_configure("lossy",    foreground="dark red")

        self._populate_table()
        self._build_summary()

        ttk.Button(self.window, text="Close", command=self.window.destroy).pack(pady=10)

    def _populate_table(self) -> None:
        """Insert one row per verification result."""
        for (image_name, comp_name), v in self.verification_results.items():
            tag = "lossless" if v.is_lossless else "lossy"
            self.tree.insert(
                "", tk.END,
                values=(
                    image_name,
                    comp_name,
                    "LOSSLESS" if v.is_lossless else "LOSSY",
                    f"{v.max_difference:.2f}",
                    f"{v.different_pixels:,}",
                    f"{v.accuracy_percent:.4f}",
                ),
                tags=(tag,),
            )

    def _build_summary(self) -> None:
        """Add a one-line summary bar below the table."""
        if not self.verification_results:
            return

        total    = len(self.verification_results)
        lossless = sum(1 for v in self.verification_results.values() if v.is_lossless)
        lossy    = total - lossless

        summary = (
            f"Total Tests: {total}  |  "
            f"Lossless: {lossless} ({100 * lossless / total:.1f}%)  |  "
            f"Lossy: {lossy} ({100 * lossy / total:.1f}%)"
        )
        ttk.Label(self.window, text=summary, font=("Arial", 10, "bold")).pack(pady=5)