"""Image Compression Benchmark — Main GUI Window.

Builds and manages the primary Tkinter application window.  All benchmark
execution is delegated to BenchmarkRunner (via a background daemon thread),
and all result formatting / export is delegated to BenchmarkSummarizer.
"""

import os
import platform
import subprocess
import threading
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
from utils.cpu_affinity import IsolationConfig
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
from utils.upload_dialog import UploadDialog
from utils.upload_and_shutdown import upload_and_maybe_shutdown


class BenchmarkGUI:
    """Main application window for the image compression benchmark.

    Responsibilities:
      - Build and lay out all Tkinter widgets.
      - Collect configuration from the UI and create a BenchmarkConfig.
      - Spawn a background thread that runs BenchmarkRunner.run().
      - Marshal progress messages back to the UI thread via root.after().
      - Trigger result export, visualisation, and optional upload on completion.

    Attributes:
        root: The top-level Tk window.
        project_root: Absolute path to the repository root (parent of src/).
        running: True while a benchmark thread is active.
        runner: The active BenchmarkRunner instance, or None.
        last_json_path: Path to the most recently written JSON result file.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Compression Benchmark")
        self.root.geometry("1400x900")

        # Resolve standard project directories relative to this source file.
        self.project_root = Path(__file__).resolve().parent.parent
        self.src_dir      = self.project_root / "src"
        self.plugins_dir  = self.src_dir      / "compressors"
        self.dataset_dir  = self.project_root / "image_datasets"
        self.output_dir   = self.project_root / "benchmark_results"
        self.libs_dir     = self.project_root / "libs"

        # Load all *_compressor.py plugins so they register themselves with the factory.
        PluginLoader.load_plugins_from_directory(self.plugins_dir)

        # Build a display_name -> factory_key mapping for the compressor selector widget.
        # The human-readable name (e.g. 'CharLS-JPEGLS') is used in the UI;
        # the factory key (e.g. 'charls') is passed to BenchmarkConfig.
        self.compressor_mapping: Dict[str, str] = {}
        for key in CompressorFactory.list_available():
            try:
                display_name = CompressorFactory.create(key).name
                self.compressor_mapping[display_name] = key
            except Exception as exc:
                print(f"Warning: could not read name for compressor '{key}': {exc}")
                self.compressor_mapping[key] = key  # fall back to the raw key

        self.available_display_names: List[str] = list(self.compressor_mapping.keys())

        self.running:        bool                      = False
        self.runner:         Optional[BenchmarkRunner] = None
        self.last_json_path: Optional[Path]            = None

        # Upload / shutdown settings — owned here so they survive dialog close/reopen.
        self.bin_name_var       = tk.StringVar(value="benchmark-results-2026")
        self.upload_var         = tk.BooleanVar(value=False)
        self.shutdown_var       = tk.BooleanVar(value=False)
        self.shutdown_delay_var = tk.IntVar(value=60)

        # Lazily created in open_upload_dialog(); lifted if already open.
        self._upload_dialog: Optional[UploadDialog] = None

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the full window layout."""
        self._build_header()

        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left = ttk.Frame(main_container)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 2))

        # Right panel has a fixed width so the log area does not collapse.
        right = ttk.Frame(main_container, width=500)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 5))
        right.pack_propagate(False)

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
        self.level_widget.level_vars[CompressionLevel.BALANCED].set(True)  # sensible default
        self.level_widget.pack(fill=tk.X, padx=5, pady=5)

        self._build_iteration_settings(left)
        self._build_advanced_settings(left)
        self._build_controls(left)

        self._build_log(right)
        self._log_welcome()

    def _build_header(self) -> None:
        """Build the title / subtitle banner at the top of the window."""
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill=tk.X)
        ttk.Label(frame, text="Image Compression Benchmark", font=("Arial", 16, "bold")).pack()
        ttk.Label(frame, text="Lossless Compression Analysis and Verification", font=("Arial", 10)).pack()

    def _build_iteration_settings(self, parent: ttk.Frame) -> None:
        """Build the Benchmark Settings panel (run modes, iteration counts).

        Args:
            parent: The parent frame into which this panel is packed.
        """
        frame = ttk.LabelFrame(parent, text="Benchmark Settings", padding="10")
        frame.pack(fill=tk.X, padx=5, pady=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=(2, 2))
        self.precision_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row,
            text="Precision Run  (2 warm-ups, 12 measurements, drop 2 slowest -> avg 10)",
            variable=self.precision_mode_var,
            command=self._on_precision_toggle,
        ).pack(side=tk.LEFT, padx=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=(2, 6))
        self.standard_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row,
            text="Standard Run  (1 warm-up, 6 measurements, drop 1 slowest -> avg 5)",
            variable=self.standard_mode_var,
            command=self._on_standard_toggle,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 6))

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Test Iterations (for averaging):").pack(side=tk.LEFT, padx=5)
        self.iterations_var = tk.IntVar(value=3)
        self.iterations_spinbox = ttk.Spinbox(row, from_=1, to=50, textvariable=self.iterations_var, width=10)
        self.iterations_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="(Higher = more accurate, but slower)", foreground="gray").pack(side=tk.LEFT, padx=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Warmup Iterations (smoke test):").pack(side=tk.LEFT, padx=5)
        self.warmup_var = tk.IntVar(value=1)
        self.warmup_spinbox = ttk.Spinbox(row, from_=0, to=10, textvariable=self.warmup_var, width=10)
        self.warmup_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="(Warms up caches before measuring)", foreground="gray").pack(side=tk.LEFT, padx=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        self.auto_visualize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Auto-open Visualization after completion", variable=self.auto_visualize_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="(Automatically show charts when benchmark finishes)", foreground="gray").pack(side=tk.LEFT, padx=5)

    def _on_precision_toggle(self) -> None:
        """Handle the Precision Run checkbox.

        Disables manual spinboxes and locks in preset values
        (12 iterations, 2 warm-ups).  Deactivates Standard Run if active.
        """
        if self.precision_mode_var.get():
            self.standard_mode_var.set(False)
            self.iterations_spinbox.config(state=tk.DISABLED)
            self.warmup_spinbox.config(state=tk.DISABLED)
            self.iterations_var.set(12)
            self.warmup_var.set(2)
        else:
            self.iterations_spinbox.config(state="normal")
            self.warmup_spinbox.config(state="normal")

    def _on_standard_toggle(self) -> None:
        """Handle the Standard Run checkbox.

        Disables manual spinboxes and locks in preset values
        (6 iterations, 1 warm-up).  Deactivates Precision Run if active.
        """
        if self.standard_mode_var.get():
            self.precision_mode_var.set(False)
            self.iterations_spinbox.config(state=tk.DISABLED)
            self.warmup_spinbox.config(state=tk.DISABLED)
            self.iterations_var.set(6)
            self.warmup_var.set(1)
        else:
            self.iterations_spinbox.config(state="normal")
            self.warmup_spinbox.config(state="normal")

    def _build_advanced_settings(self, parent: ttk.Frame) -> None:
        """Build the Advanced Settings panel (resource monitoring, isolation, affinity).

        Args:
            parent: The parent frame into which this panel is packed.
        """
        frame = ttk.LabelFrame(parent, text="Advanced Settings", padding="10")
        frame.pack(fill=tk.X, padx=5, pady=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        self.monitor_resources_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Monitor System Resources (CPU, RAM, I/O)", variable=self.monitor_resources_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="- Tracks resource usage during compression", foreground="gray").pack(side=tk.LEFT, padx=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        self.isolate_process_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Process Isolation (High Priority)", variable=self.isolate_process_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="- Sets high process priority for more accurate measurements", foreground="gray").pack(side=tk.LEFT, padx=5)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(
            row,
            text="Process Isolation may require administrator privileges and affect system responsiveness",
            foreground="orange", font=("Arial", 8),
        ).pack(side=tk.LEFT, padx=20)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=(6, 2))
        self.cpu_affinity_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Limit to single CPU core", variable=self.cpu_affinity_var, command=self._on_affinity_toggle).pack(side=tk.LEFT, padx=5)
        ttk.Label(row, text="Core:").pack(side=tk.LEFT, padx=(10, 2))
        max_core = max(0, (psutil.cpu_count(logical=True) or 1) - 1)
        self.cpu_core_var = tk.IntVar(value=min(1, max_core))
        self.cpu_core_spinbox = ttk.Spinbox(row, from_=0, to=max_core, textvariable=self.cpu_core_var, width=5, state=tk.DISABLED)
        self.cpu_core_spinbox.pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text=f"(0–{max_core} available; avoid core 0 — high IRQ load on Windows)", foreground="gray").pack(side=tk.LEFT, padx=5)

    def _on_affinity_toggle(self) -> None:
        """Enable or disable the CPU core spinbox based on the affinity checkbox."""
        self.cpu_core_spinbox.config(state="normal" if self.cpu_affinity_var.get() else tk.DISABLED)

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Build the action button row and the indeterminate progress bar.

        Args:
            parent: The parent frame into which this panel is packed.
        """
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.X)

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(0, 5))

        self.run_button = ttk.Button(btn_row, text="Run Benchmark", command=self.run_benchmark)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(btn_row, text="Stop", command=self.stop_benchmark, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_row, text="Visualization",  command=self.open_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Verify Results", command=self.show_verification_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Results Folder", command=self.open_results_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Remote Upload…", command=self.open_upload_dialog).pack(side=tk.LEFT, padx=5)

        # Indeterminate progress bar shown while the background thread is running.
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=5)

    def _build_log(self, parent: ttk.Frame) -> None:
        """Build the scrollable console output area.

        Args:
            parent: The parent frame into which this panel is packed.
        """
        frame = ttk.LabelFrame(parent, text="Console Output", padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 9), bg="#f5f5f5")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def log(self, message: str) -> None:
        """Append a line to the console output widget.

        Safe to call from the main thread only.  Background threads should
        marshal via root.after(0, self.log, msg).

        Args:
            message: Text to append (a newline is added automatically).
        """
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _log_welcome(self) -> None:
        """Print the startup banner to the console output area."""
        self.log("Image Compression Benchmark Tool")
        self.log(f"Available compressors: {', '.join(self.available_display_names)}")
        self.log("Select images and compressors to begin.")
        self.log("")

    # -----------------------------------------------------------------------
    # Image selection
    # -----------------------------------------------------------------------

    def add_images(self) -> None:
        """Open a file dialog and add selected images to the image list widget."""
        files = filedialog.askopenfilenames(
            title="Select Images",
            initialdir=self.project_root / "image_datasets",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"), ("All files", "*.*")],
        )
        for f in files:
            self.image_widget.add_image(Path(f))

    def add_folder(self) -> None:
        """Open a directory dialog and add all images from the selected folder."""
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            # Non-recursive: add only the top-level images in the chosen folder,
            # not subdirectories, to keep the selection predictable.
            for img in ImageFinder.find_images(Path(folder), recursive=False):
                self.image_widget.add_image(img)

    # -----------------------------------------------------------------------
    # Upload dialog
    # -----------------------------------------------------------------------

    def open_upload_dialog(self) -> None:
        """Open (or focus) the Remote Upload settings dialog."""
        if self._upload_dialog is None or not self._upload_dialog.winfo_exists():
            self._upload_dialog = UploadDialog(
                self.root,
                bin_name_var       = self.bin_name_var,
                upload_var         = self.upload_var,
                shutdown_var       = self.shutdown_var,
                shutdown_delay_var = self.shutdown_delay_var,
            )
        else:
            # Dialog already open — bring it to the front.
            self._upload_dialog.lift()
            self._upload_dialog.focus_set()

    # -----------------------------------------------------------------------
    # Results folder / verification
    # -----------------------------------------------------------------------

    def open_results_folder(self) -> None:
        """Open the benchmark results folder in the OS file manager.

        Uses platform-appropriate commands:
          - Windows  : os.startfile()
          - macOS    : open
          - Linux    : xdg-open
        """
        self.output_dir.mkdir(exist_ok=True)
        system = platform.system()
        try:
            if system == "Windows":
                os.startfile(self.output_dir)  # type: ignore[attr-defined]
            elif system == "Darwin":
                subprocess.Popen(["open", str(self.output_dir)])
            else:
                # Linux and other POSIX systems.
                subprocess.Popen(["xdg-open", str(self.output_dir)])
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open results folder:\n{exc}")

    def show_verification_results(self) -> None:
        """Open the verification results popup, if a run has been completed."""
        if not self.runner or not self.runner.verification_results:
            messagebox.showinfo("No Results", "No verification results available. Run a benchmark first.")
            return
        VerificationResultsWidget(self.root, self.runner.verification_results)

    # -----------------------------------------------------------------------
    # CPU affinity
    # -----------------------------------------------------------------------

    def _get_cpu_affinity_core(self) -> Optional[int]:
        """Read the selected CPU affinity core from the UI.

        Returns:
            The core index as an int, or None when affinity is disabled or the
            spinbox contains an invalid value.
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
        """Dispatch to _start_benchmark with the correct preset or custom settings."""
        if self.precision_mode_var.get():
            self._start_benchmark(warmup=2, iterations=12, trim_top_n=2)
        elif self.standard_mode_var.get():
            self._start_benchmark(warmup=1, iterations=6, trim_top_n=1)
        else:
            # Custom mode: read values from the spinboxes.
            self._start_benchmark(warmup=None, iterations=None, trim_top_n=0)

    def _start_benchmark(self, warmup, iterations, trim_top_n) -> None:
        """Validate the UI state, build a BenchmarkConfig, and start the run thread.

        Args:
            warmup: Warm-up iteration count override, or None to use the spinbox.
            iterations: Measurement iteration count override, or None to use the spinbox.
            trim_top_n: Number of slowest runs to drop before averaging.
        """
        if not self.image_widget.selected_images:
            messagebox.showwarning("No Images", "Please select at least one image.")
            return

        selected_display = self.compressor_widget.get_selected()
        if not selected_display:
            messagebox.showwarning("No Compressors", "Please select at least one compressor.")
            return

        selected_keys = [self.compressor_mapping[name] for name in selected_display]

        selected_levels = self.level_widget.get_selected()
        if not selected_levels:
            messagebox.showwarning("No Levels", "Please select at least one compression level.")
            return

        if self.upload_var.get() and not self.bin_name_var.get().strip():
            messagebox.showwarning("No Bin Name", "Open Remote Upload and enter a bin name first.")
            return

        # Warn explicitly before a destructive system action.
        if self.shutdown_var.get():
            confirmed = messagebox.askyesno(
                "Shutdown Warning",
                f"After the benchmark, results will be uploaded to:\n"
                f"  https://filebin.net/{self.bin_name_var.get().strip()}\n\n"
                f"Then this PC will shut down in {self.shutdown_delay_var.get()} seconds.\n\n"
                "Continue?",
                icon="warning",
            )
            if not confirmed:
                return

        verify_enabled    = self.level_widget.is_verification_enabled()
        strip_metadata    = self.level_widget.is_strip_metadata_enabled()
        monitor_resources = self.monitor_resources_var.get()
        num_iterations    = iterations if iterations is not None else self.iterations_var.get()
        warmup_iterations = warmup    if warmup     is not None else self.warmup_var.get()
        high_priority     = self.isolate_process_var.get()
        cpu_affinity_core = self._get_cpu_affinity_core()

        # Confirm privilege escalation explicitly before attempting it.
        if high_priority:
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
                high_priority = False

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
            trim_top_n         = trim_top_n,
            monitor_resources  = monitor_resources,
            isolation          = IsolationConfig(
                high_priority = high_priority,
                cpu_core      = cpu_affinity_core,
            ),
        )

        # Transition UI to "running" state before spawning the thread.
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start()
        self.log_text.delete(1.0, tk.END)

        self.runner = BenchmarkRunner(config)
        threading.Thread(target=self._benchmark_thread, args=(config,), daemon=True).start()

    def _benchmark_thread(self, config: BenchmarkConfig) -> None:
        """Background thread body: runs the benchmark and schedules UI callbacks.

        All Tkinter calls are marshalled to the main thread via root.after(0, …)
        because Tkinter is not thread-safe.

        Note:
            The ``elif not self.running`` branch below is technically unreachable
            here: self.running is set to False on the main thread (inside
            _on_benchmark_finished, which is queued via root.after), so at the
            point this check runs self.running is always still True.  The logic
            is preserved for clarity and as a guard against future refactoring.

        Args:
            config: The BenchmarkConfig for this run (passed through for the
                _save_and_summarize callback).
        """
        try:
            results, verification_results = self.runner.run(
                progress_callback=lambda msg: self.root.after(0, self.log, msg)
            )
            if results and self.running:
                self.root.after(0, self._save_and_summarize, results, verification_results, config)
            elif not self.running:
                # Reached only if self.running is somehow already False here —
                # see docstring note above.
                self.root.after(0, self.log, "\nBenchmark stopped by user.")
            else:
                self.root.after(0, self.log, "\nNo results generated.")
        except Exception as exc:
            import traceback
            self.root.after(0, self.log, f"\nError: {exc}")
            self.root.after(0, self.log, traceback.format_exc())
        finally:
            self.root.after(0, self._on_benchmark_finished)

    def _save_and_summarize(self, results, verification_results, config: BenchmarkConfig) -> None:
        """Export results to JSON, print summaries, and trigger optional post-run actions.

        Called on the main thread via root.after() after the benchmark thread
        completes successfully.

        Args:
            results: List of BenchmarkResult objects from the run.
            verification_results: Dict of VerificationResult objects.
            config: The BenchmarkConfig used for this run.
        """
        self.log("\n" + "=" * 70)
        self.log("Saving results...")

        json_dir   = self.output_dir / "json_reports"
        json_paths = BenchmarkSummarizer.export_results_json(results, verification_results, json_dir, config)

        for p in json_paths:
            self.log(f"Results saved to: {p.name}")
            self.log(f"Full path: {p}")
        self.last_json_path = json_paths[-1] if json_paths else None

        BenchmarkSummarizer.print_compression_summary(results, self.log)
        BenchmarkSummarizer.print_verification_summary(verification_results, self.log)

        if config.monitor_resources:
            BenchmarkSummarizer.print_scenario_analysis(results, self.log)

        self.log("\nBenchmark completed successfully!")

        # Upload all level JSON files if the user opted in.
        if self.upload_var.get() and json_paths:
            bin_name = self.bin_name_var.get().strip()
            upload_and_maybe_shutdown(
                json_paths     = json_paths,
                bin_name       = bin_name,
                log            = self.log,
                shutdown_after = self.shutdown_var.get(),
                shutdown_delay = self.shutdown_delay_var.get(),
            )

        # Auto-open the visualisation window with the last written JSON file.
        if self.auto_visualize_var.get():
            self.log("\nOpening visualization window...")
            try:
                data = BenchmarkDataLoader.load_from_file(json_paths[-1]) if json_paths else None
                if data:
                    # Delay slightly so the UI has time to finish rendering first.
                    self.root.after(500, lambda: self._open_visualization_with_data(data, auto_show=True))
            except Exception as exc:
                self.log(f"Could not auto-open visualization: {exc}")

    def _on_benchmark_finished(self) -> None:
        """Reset the UI back to the idle state after the benchmark thread exits."""
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()

    def stop_benchmark(self) -> None:
        """Request a graceful stop of the currently running benchmark."""
        if self.runner:
            self.runner.stop()
        self.running = False
        self.log("\nStopping benchmark...")

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------

    def open_visualization(self) -> None:
        """Prompt the user to select a JSON file and open the visualisation window."""
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
        """Open the visualisation window for the given benchmark data.

        Args:
            data: Pre-loaded BenchmarkData object.
            auto_show: When True, automatically navigate to the compression
                ratio chart after a short delay (used by the auto-open path).
        """
        viz_window = open_visualization_window(self.root, data)
        if auto_show:
            self.root.after(200, viz_window.show_compression_ratio)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Create the root Tk window, apply a theme, and start the event loop."""
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass  # clam not available on all platforms — fall back to the default theme
    BenchmarkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()