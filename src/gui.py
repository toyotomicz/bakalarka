"""
Image Compression Benchmark — Main GUI Window
gui.py

Builds the Tkinter interface, wires all widgets to BenchmarkRunner,
and handles the benchmark lifecycle (start / stop / results / visualisation).
"""

import os
import threading
import sys
from pathlib import Path
from typing import Dict, List, Optional
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import psutil

from main import CompressorFactory, CompressionLevel, PluginLoader
from benchmark_shared import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkSummarizer,
    ImageFinder,
)
from utils.verification import VerificationResult
from utils.gui_widgets import (
    CompressorSelectionWidget,
    ImageSelectionWidget,
    LevelSelectionWidget,
    VerificationResultsWidget,
)
from utils.benchmark_visualization import (
    BenchmarkData,
    BenchmarkDataLoader,
    open_visualization_window,
)


class BenchmarkGUI:
    """Main application window for the image compression benchmark."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Compression Benchmark")
        self.root.geometry("1400x900")

        # Project directory layout
        self.project_root = Path(__file__).resolve().parent.parent
        self.src_dir      = self.project_root / "src"
        self.plugins_dir  = self.src_dir      / "compressors"
        self.dataset_dir  = self.project_root / "image_datasets"
        self.output_dir   = self.project_root / "benchmark_results"
        self.libs_dir     = self.project_root / "libs"

        # Load all *_compressor.py plugins from the compressors directory.
        PluginLoader.load_plugins_from_directory(self.plugins_dir)

        # Build a display-name → factory-key mapping so the GUI shows the
        # human-readable name from the compressor's @property name instead of
        # the internal registration key (e.g. "CharLS-JPEGLS" instead of "charls").
        self.compressor_mapping: Dict[str, str] = {}
        for key in CompressorFactory.list_available():
            try:
                display_name = CompressorFactory.create(key).name
                self.compressor_mapping[display_name] = key
            except Exception as exc:
                # Broken plugin: fall back to the raw key so it still appears in the list.
                print(f"Warning: could not read name for compressor '{key}': {exc}")
                self.compressor_mapping[key] = key

        self.available_display_names: List[str] = list(self.compressor_mapping.keys())

        # Runtime state
        self.running:        bool                    = False
        self.runner:         Optional[BenchmarkRunner] = None
        self.last_json_path: Optional[Path]          = None

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Create and lay out all top-level UI sections."""
        self._build_header()

        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left column — all controls (~60 % width)
        left = ttk.Frame(main_container)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 2))

        # Right column — console output (~40 % width, fixed at 500 px)
        right = ttk.Frame(main_container, width=500)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 5))
        right.pack_propagate(False)

        # --- Left column contents ---
        self.image_widget = ImageSelectionWidget(left)
        self.image_widget.add_images_btn.config(command=self.add_images)
        self.image_widget.add_folder_btn.config(command=self.add_folder)
        self.image_widget.clear_btn.config(command=self.image_widget.clear_images)
        self.image_widget.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

        self.compressor_widget = CompressorSelectionWidget(left, self.available_display_names)
        self.compressor_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        levels = [
            ("FASTEST",  CompressionLevel.FASTEST),
            ("BALANCED", CompressionLevel.BALANCED),
            ("BEST",     CompressionLevel.BEST),
        ]
        self.level_widget = LevelSelectionWidget(left, levels)
        self.level_widget.level_vars[CompressionLevel.BALANCED].set(True)
        self.level_widget.pack(fill=tk.X, padx=5, pady=5)

        self._build_iteration_settings(left)
        self._build_advanced_settings(left)
        self._build_controls(left)

        # --- Right column contents ---
        self._build_log(right)
        self._log_welcome()

    def _build_header(self) -> None:
        """Application title and subtitle bar."""
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill=tk.X)

        ttk.Label(
            frame,
            text="Image Compression Benchmark",
            font=("Arial", 16, "bold"),
        ).pack()

        ttk.Label(
            frame,
            text="Lossless Compression Analysis and Verification",
            font=("Arial", 10),
        ).pack()

    def _build_iteration_settings(self, parent: ttk.Frame) -> None:
        """Spinboxes for test / warm-up iterations and the auto-visualize toggle."""
        frame = ttk.LabelFrame(parent, text="Benchmark Settings", padding="10")
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Test iterations
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Test Iterations (for averaging):").pack(side=tk.LEFT, padx=5)
        self.iterations_var = tk.IntVar(value=3)
        ttk.Spinbox(row, from_=1, to=10, textvariable=self.iterations_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="(Higher = more accurate, but slower)", foreground="gray").pack(side=tk.LEFT, padx=5)

        # Warm-up iterations
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Warmup Iterations (smoke test):").pack(side=tk.LEFT, padx=5)
        self.warmup_var = tk.IntVar(value=1)
        ttk.Spinbox(row, from_=0, to=5, textvariable=self.warmup_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="(Warms up caches before measuring)", foreground="gray").pack(side=tk.LEFT, padx=5)

        # Auto-open visualisation after benchmark completes
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        self.auto_visualize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row,
            text="Auto-open Visualization after completion",
            variable=self.auto_visualize_var,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(
            row,
            text="(Automatically show charts when benchmark finishes)",
            foreground="gray",
        ).pack(side=tk.LEFT, padx=5)

    def _build_advanced_settings(self, parent: ttk.Frame) -> None:
        """
        Checkboxes / spinboxes for resource monitoring, process isolation,
        and CPU affinity.

        CPU affinity is now fully delegated to BenchmarkConfig + BenchmarkRunner
        (via ProcessIsolator in utils/cpu_affinity.py).  The GUI only collects the
        desired core index and passes it through BenchmarkConfig.cpu_affinity_core.
        """
        frame = ttk.LabelFrame(parent, text="Advanced Settings", padding="10")
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Resource monitoring
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        self.monitor_resources_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            row,
            text="Monitor System Resources (CPU, RAM, I/O)",
            variable=self.monitor_resources_var,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="- Tracks resource usage during compression", foreground="gray").pack(side=tk.LEFT, padx=5)

        # Process isolation (high priority)
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        self.isolate_process_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row,
            text="Process Isolation (High Priority)",
            variable=self.isolate_process_var,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(
            row,
            text="- Sets high process priority for more accurate measurements",
            foreground="gray",
        ).pack(side=tk.LEFT, padx=5)

        # Warning label for process isolation
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(
            row,
            text="Process Isolation may require administrator privileges and affect system responsiveness",
            foreground="orange",
            font=("Arial", 8),
        ).pack(side=tk.LEFT, padx=20)

        # ---- CPU Affinity ------------------------------------------------
        # The checkbox enables pinning; the spinbox selects which core to use.
        # Core 0 is the default but the user can pick any available logical core.
        # Validation against the actual core count happens in BenchmarkRunner.run()
        # so the GUI never raises an error dialog for an invalid core index.
        # Note: core 0 is often busiest (system IRQs on Windows) — the label
        # nudges users toward higher-numbered cores for cleaner results.
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=(6, 2))

        self.cpu_affinity_var = tk.BooleanVar(value=False)
        affinity_check = ttk.Checkbutton(
            row,
            text="Limit to single CPU core",
            variable=self.cpu_affinity_var,
            command=self._on_affinity_toggle,
        )
        affinity_check.pack(side=tk.LEFT, padx=5)

        ttk.Label(row, text="Core:").pack(side=tk.LEFT, padx=(10, 2))

        max_core = max(0, (psutil.cpu_count(logical=True) or 1) - 1)
        self.cpu_core_var = tk.IntVar(value=min(1, max_core))   # default = core 1
        self.cpu_core_spinbox = ttk.Spinbox(
            row,
            from_=0,
            to=max_core,
            textvariable=self.cpu_core_var,
            width=5,
            state=tk.DISABLED,   # enabled only when checkbox is ticked
        )
        self.cpu_core_spinbox.pack(side=tk.LEFT, padx=2)

        ttk.Label(
            row,
            text=f"(0–{max_core} available; avoid core 0 — high IRQ load on Windows)",
            foreground="gray",
        ).pack(side=tk.LEFT, padx=5)

    def _on_affinity_toggle(self) -> None:
        """Enable / disable the core-selection spinbox when the affinity checkbox changes."""
        if self.cpu_affinity_var.get():
            self.cpu_core_spinbox.config(state="normal")
        else:
            self.cpu_core_spinbox.config(state=tk.DISABLED)

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Run / Stop / Visualization / Verify / Results folder buttons + progress bar."""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.X)

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(0, 5))

        self.run_button = ttk.Button(btn_row, text="Run Benchmark", command=self.run_benchmark)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            btn_row, text="Stop", command=self.stop_benchmark, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_row, text="Visualization",  command=self.open_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Verify Results", command=self.show_verification_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Results Folder", command=self.open_results_folder).pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=5)

    def _build_log(self, parent: ttk.Frame) -> None:
        """Scrolled text widget used as the console output pane."""
        frame = ttk.LabelFrame(parent, text="Console Output", padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#f5f5f5",
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def log(self, message: str) -> None:
        """Append a line to the console output and scroll to the bottom."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _log_welcome(self) -> None:
        self.log("Image Compression Benchmark Tool")
        self.log(f"Available compressors: {', '.join(self.available_display_names)}")
        self.log("Select images and compressors to begin.")
        self.log("")

    # -----------------------------------------------------------------------
    # Image selection
    # -----------------------------------------------------------------------

    def add_images(self) -> None:
        """Open a file dialog to add individual image files."""
        files = filedialog.askopenfilenames(
            title="Select Images",
            initialdir=self.project_root / "image_datasets",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All files", "*.*"),
            ],
        )
        for f in files:
            self.image_widget.add_image(Path(f))

    def add_folder(self) -> None:
        """Open a folder dialog and add all images found in the selected directory."""
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            for img in ImageFinder.find_images(Path(folder), recursive=False):
                self.image_widget.add_image(img)

    # -----------------------------------------------------------------------
    # Results folder
    # -----------------------------------------------------------------------

    def open_results_folder(self) -> None:
        """Open the benchmark results directory in Windows Explorer."""
        self.output_dir.mkdir(exist_ok=True)
        os.startfile(self.output_dir)

    # -----------------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------------

    def show_verification_results(self) -> None:
        """Open the verification results window, or warn if no data is available."""
        if not self.runner or not self.runner.verification_results:
            messagebox.showinfo(
                "No Results",
                "No verification results available. Run a benchmark first.",
            )
            return
        VerificationResultsWidget(self.root, self.runner.verification_results)

    # -----------------------------------------------------------------------
    # CPU affinity helpers
    # -----------------------------------------------------------------------

    def _get_cpu_affinity_core(self) -> Optional[int]:
        """
        Return the selected core index when the affinity checkbox is ticked,
        otherwise return None.

        The returned value is stored directly in BenchmarkConfig.cpu_affinity_core.
        All validation and actual affinity-setting is handled by BenchmarkRunner
        via ProcessIsolator — no psutil calls happen here.
        """
        if not self.cpu_affinity_var.get():
            return None
        try:
            return int(self.cpu_core_var.get())
        except (tk.TclError, ValueError):
            return None

    # -----------------------------------------------------------------------
    # Benchmark lifecycle
    # -----------------------------------------------------------------------

    def run_benchmark(self) -> None:
        """Validate inputs, build BenchmarkConfig, and start the worker thread."""
        if not self.image_widget.selected_images:
            messagebox.showwarning("No Images", "Please select at least one image.")
            return

        # Resolve display names back to factory keys before building the config.
        selected_display   = self.compressor_widget.get_selected()
        if not selected_display:
            messagebox.showwarning("No Compressors", "Please select at least one compressor.")
            return

        selected_keys = [self.compressor_mapping[name] for name in selected_display]

        selected_levels = self.level_widget.get_selected()
        if not selected_levels:
            messagebox.showwarning("No Levels", "Please select at least one compression level.")
            return

        verify_enabled    = self.level_widget.is_verification_enabled()
        strip_metadata    = self.level_widget.is_strip_metadata_enabled()
        monitor_resources = self.monitor_resources_var.get()
        isolate_process   = self.isolate_process_var.get()
        num_iterations    = self.iterations_var.get()
        warmup_iterations = self.warmup_var.get()
        cpu_affinity_core = self._get_cpu_affinity_core()

        # Warn the user before raising process priority.
        if isolate_process:
            confirmed = messagebox.askyesno(
                "Process Isolation Warning",
                "Process isolation will set this application to high priority.\n\n"
                "This may:\n"
                "  Require administrator privileges\n"
                "  Make your system less responsive during the benchmark\n"
                "  Provide more accurate measurements\n\n"
                "Continue with process isolation?",
                icon="warning",
            )
            if not confirmed:
                self.isolate_process_var.set(False)
                isolate_process = False

        config = BenchmarkConfig(
            dataset_dir        = self.dataset_dir,
            output_dir         = self.output_dir,
            libs_dir           = self.libs_dir,
            compressor_names   = selected_keys,
            image_paths        = list(self.image_widget.selected_images),
            compression_levels = selected_levels,
            verify_lossless    = verify_enabled,
            strip_metadata     = strip_metadata,
            num_iterations     = num_iterations,
            warmup_iterations  = warmup_iterations,
            monitor_resources  = monitor_resources,
            isolate_process    = isolate_process,
            cpu_affinity_core  = cpu_affinity_core,   # NEW — passed to BenchmarkRunner
        )

        # Transition UI to "running" state.
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start()
        self.log_text.delete(1.0, tk.END)

        self.runner = BenchmarkRunner(config)

        threading.Thread(
            target=self._benchmark_thread,
            args=(config,),
            daemon=True,
        ).start()

    def _benchmark_thread(self, config: BenchmarkConfig) -> None:
        """
        Worker thread: run the benchmark.

        CPU affinity and process isolation are now handled entirely inside
        BenchmarkRunner.run() via ProcessIsolator (utils/cpu_affinity.py).  This thread no longer
        calls _apply_cpu_affinity / _restore_cpu_affinity directly.
        """
        try:
            results, verification_results = self.runner.run(
                progress_callback=lambda msg: self.root.after(0, self.log, msg)
            )

            if results and self.running:
                self.root.after(0, self._save_and_summarize, results, verification_results, config)
            elif not self.running:
                self.root.after(0, self.log, "\nBenchmark stopped by user.")
            else:
                self.root.after(0, self.log, "\nNo results generated.")

        except Exception as exc:
            import traceback
            self.root.after(0, self.log, f"\nError: {exc}")
            self.root.after(0, self.log, traceback.format_exc())

        finally:
            self.root.after(0, self._on_benchmark_finished)

    def _save_and_summarize(
        self,
        results,
        verification_results,
        config: BenchmarkConfig,
    ) -> None:
        """Save the JSON report and print summaries to the console."""
        self.log("\n" + "=" * 70)
        self.log("Saving results...")

        json_dir  = self.output_dir / "json_reports"
        json_path = BenchmarkSummarizer.export_results_json(
            results, verification_results, json_dir, config
        )
        self.last_json_path = json_path

        self.log(f"Results saved to: {json_path.name}")
        self.log(f"Full path: {json_path}")

        BenchmarkSummarizer.print_compression_summary(results, self.log)
        BenchmarkSummarizer.print_verification_summary(verification_results, self.log)

        if config.monitor_resources:
            BenchmarkSummarizer.print_scenario_analysis(results, self.log)

        self.log("\nBenchmark completed successfully!")

        # Optionally open the visualisation window automatically.
        if self.auto_visualize_var.get():
            self.log("\nOpening visualization window...")
            try:
                data = BenchmarkDataLoader.load_from_file(json_path)
                if data:
                    self.root.after(500, lambda: self._open_visualization_with_data(data, auto_show=True))
            except Exception as exc:
                self.log(f"Could not auto-open visualization: {exc}")

    def _on_benchmark_finished(self) -> None:
        """Reset the UI to its idle state after a benchmark run."""
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()

    def stop_benchmark(self) -> None:
        """Request early termination of the running benchmark."""
        if self.runner:
            self.runner.stop()
        self.running = False
        self.log("\nStopping benchmark...")

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------

    def open_visualization(self) -> None:
        """Open a file dialog to select a JSON report for visualisation."""
        json_dir = self.output_dir / "json_reports"
        json_dir.mkdir(parents=True, exist_ok=True)

        filename = filedialog.askopenfilename(
            title="Select Benchmark JSON",
            initialdir=str(json_dir),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if filename:
            data = BenchmarkDataLoader.load_from_file(Path(filename))
            if data:
                self._open_visualization_with_data(data, auto_show=False)
                self.log(f"Opened visualization for: {Path(filename).name}")
            else:
                messagebox.showerror("Error", "Failed to load JSON file.")

    def _open_visualization_with_data(self, data: BenchmarkData, auto_show: bool = False) -> None:
        """Open the visualisation window, optionally auto-showing the first chart."""
        viz_window = open_visualization_window(self.root, data)
        if auto_show:
            self.root.after(200, viz_window.show_compression_ratio)


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """Launch the GUI application."""
    root = tk.Tk()

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass  # Fall back to the default theme if 'clam' is unavailable.

    BenchmarkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()