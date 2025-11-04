"""
Image Compression Benchmark GUI
Main application window and orchestration
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from pathlib import Path
import threading
import sys
from typing import List

from main import (
    CompressorFactory,
    PluginLoader,
    CompressionLevel
)
from benchmark_shared import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkSummarizer,
    ImageFinder
)
from utils.verification import VerificationResult
from utils.gui_widgets import (
    ImageSelectionWidget,
    CompressorSelectionWidget,
    LevelSelectionWidget,
    VerificationResultsWidget
)


class BenchmarkGUI:
    """Main GUI application for image compression benchmark"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compression Benchmark")
        self.root.geometry("1000x900")
        
        # Initialize paths
        self.project_root = Path(__file__).parent
        self.plugins_dir = self.project_root / "compressors"
        self.dataset_dir = self.project_root / "image_datasets"
        self.output_dir = self.project_root / "benchmark_results"
        self.libs_dir = self.project_root / "libs"
        
        # Load plugins
        PluginLoader.load_plugins_from_directory(self.plugins_dir)
        self.available_compressors = CompressorFactory.list_available()
        
        # State
        self.running = False
        self.runner = None
        
        # Create UI
        self.create_widgets()
    
    def create_widgets(self):
        """Create all GUI components"""
        # Header
        self.create_header()
        
        # Image selection
        self.image_widget = ImageSelectionWidget(self.root)
        self.image_widget.add_images_btn.config(command=self.add_images)
        self.image_widget.add_folder_btn.config(command=self.add_folder)
        self.image_widget.clear_btn.config(command=self.image_widget.clear_images)
        self.image_widget.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        
        # Compressor selection
        self.compressor_widget = CompressorSelectionWidget(
            self.root,
            self.available_compressors
        )
        self.compressor_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Level selection
        levels = [
            ("FASTEST", CompressionLevel.FASTEST),
            ("FAST", CompressionLevel.FAST),
            ("BALANCED", CompressionLevel.BALANCED),
            ("GOOD", CompressionLevel.GOOD),
            ("BEST", CompressionLevel.BEST)
        ]
        self.level_widget = LevelSelectionWidget(self.root, levels)
        # Set BALANCED as default
        self.level_widget.level_vars[CompressionLevel.BALANCED].set(True)
        self.level_widget.pack(fill=tk.X, padx=10, pady=5)
        
        # Iteration settings
        self.create_iteration_settings()
        
        # Control buttons
        self.create_controls()
        
        # Log output
        self.create_log()
        
        # Initial message
        self.log_initial_message()
    
    def create_header(self):
        """Create header section"""
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(
            header_frame,
            text="Image Compression Benchmark",
            font=("Arial", 16, "bold")
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            header_frame,
            text="Lossless Compression Analysis and Verification",
            font=("Arial", 10)
        )
        subtitle_label.pack()
    
    def create_iteration_settings(self):
        """Create iteration configuration section"""
        iter_frame = ttk.LabelFrame(self.root, text="Benchmark Settings", padding="10")
        iter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Iterations
        iter_row = ttk.Frame(iter_frame)
        iter_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(iter_row, text="Test Iterations (for averaging):").pack(side=tk.LEFT, padx=5)
        
        self.iterations_var = tk.IntVar(value=3)
        iterations_spin = ttk.Spinbox(
            iter_row,
            from_=1,
            to=10,
            textvariable=self.iterations_var,
            width=10
        )
        iterations_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            iter_row,
            text="(Higher = more accurate, but slower)",
            foreground="gray"
        ).pack(side=tk.LEFT, padx=5)
        
        # Warmup
        warmup_row = ttk.Frame(iter_frame)
        warmup_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(warmup_row, text="Warmup Iterations (smoke test):").pack(side=tk.LEFT, padx=5)
        
        self.warmup_var = tk.IntVar(value=1)
        warmup_spin = ttk.Spinbox(
            warmup_row,
            from_=0,
            to=5,
            textvariable=self.warmup_var,
            width=10
        )
        warmup_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            warmup_row,
            text="(Warms up caches before measuring)",
            foreground="gray"
        ).pack(side=tk.LEFT, padx=5)
    
    def create_controls(self):
        """Create control button section"""
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        self.run_button = ttk.Button(
            control_frame,
            text="Run Benchmark",
            command=self.run_benchmark
        )
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            control_frame,
            text="Stop",
            command=self.stop_benchmark,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="View Verification Results",
            command=self.show_verification_results
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="Open Results Folder",
            command=self.open_results_folder
        ).pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
    
    def create_log(self):
        """Create log output section"""
        log_frame = ttk.LabelFrame(self.root, text="Output", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=12,
            wrap=tk.WORD,
            font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def log(self, message: str):
        """Add message to log output"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def log_initial_message(self):
        """Display initial welcome message"""
        self.log("Image Compression Benchmark Tool")
        self.log(f"Available compressors: {', '.join(self.available_compressors)}")
        self.log("Select images and compressors to begin.")
        self.log("")
    
    def add_images(self):
        """Add individual image files"""
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        for file in files:
            self.image_widget.add_image(Path(file))
    
    def add_folder(self):
        """Add all images from a folder"""
        folder = filedialog.askdirectory(title="Select Folder")
        
        if folder:
            folder_path = Path(folder)
            images = ImageFinder.find_images(folder_path, recursive=False)
            
            for img in images:
                self.image_widget.add_image(img)
    
    def open_results_folder(self):
        """Open results folder in file explorer"""
        import os
        import platform
        
        results_path = self.output_dir
        results_path.mkdir(exist_ok=True)
        
        system = platform.system()
        if system == "Windows":
            os.startfile(results_path)
        elif system == "Darwin":
            os.system(f'open "{results_path}"')
        else:
            os.system(f'xdg-open "{results_path}"')
    
    def show_verification_results(self):
        """Show verification results window"""
        if not self.runner or not self.runner.verification_results:
            messagebox.showinfo(
                "No Results",
                "No verification results available. Run a benchmark first."
            )
            return
        
        VerificationResultsWidget(self.root, self.runner.verification_results)
    
    def run_benchmark(self):
        """Start benchmark execution"""
        # Validate inputs
        if not self.image_widget.selected_images:
            messagebox.showwarning("No Images", "Please select at least one image.")
            return
        
        selected_compressors = self.compressor_widget.get_selected()
        if not selected_compressors:
            messagebox.showwarning("No Compressors", "Please select at least one compressor.")
            return
        
        selected_levels = self.level_widget.get_selected()
        if not selected_levels:
            messagebox.showwarning("No Levels", "Please select at least one compression level.")
            return
        
        verify_enabled = self.level_widget.is_verification_enabled()
        num_iterations = self.iterations_var.get()
        warmup_iterations = self.warmup_var.get()
        
        # Create configuration
        config = BenchmarkConfig(
            dataset_dir=self.dataset_dir,
            output_dir=self.output_dir,
            libs_dir=self.libs_dir,
            compressor_names=selected_compressors,
            image_paths=list(self.image_widget.selected_images),
            compression_levels=selected_levels,
            verify_lossless=verify_enabled,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Update UI state
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start()
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Create runner
        self.runner = BenchmarkRunner(config)
        
        # Run in background thread
        thread = threading.Thread(
            target=self._run_benchmark_thread,
            daemon=True
        )
        thread.start()
    
    def _run_benchmark_thread(self):
        """Execute benchmark in background thread"""
        try:
            # Run benchmark with log callback
            results, verification_results = self.runner.run(
                progress_callback=lambda msg: self.root.after(0, self.log, msg)
            )
            
            # Save and summarize results
            if results and self.running:
                self.root.after(0, self._save_and_summarize, results, verification_results)
            elif not self.running:
                self.root.after(0, self.log, "\nBenchmark stopped by user.")
            else:
                self.root.after(0, self.log, "\nNo results generated.")
                
        except Exception as e:
            self.root.after(0, self.log, f"\nError: {str(e)}")
            import traceback
            self.root.after(0, self.log, traceback.format_exc())
        
        finally:
            self.root.after(0, self._benchmark_finished)
    
    def _save_and_summarize(self, results, verification_results):
        """Save results and display summary"""
        # Save results to JSON
        from main import BenchmarkOrchestrator
        
        self.log("\n" + "="*70)
        self.log("Saving results...")
        
        orchestrator = BenchmarkOrchestrator(
            self.dataset_dir,
            self.output_dir,
            self.libs_dir
        )
        orchestrator.results = results
        orchestrator.export_results(self.output_dir / "results.json")
        
        # Print summaries
        BenchmarkSummarizer.print_compression_summary(results, self.log)
        BenchmarkSummarizer.print_verification_summary(verification_results, self.log)
        
        self.log("\n✅ Benchmark completed successfully!")
    
    def _benchmark_finished(self):
        """Reset UI state after benchmark completion"""
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()
    
    def stop_benchmark(self):
        """Stop running benchmark"""
        if self.runner:
            self.runner.stop()
        self.running = False
        self.log("\nStopping benchmark...")


def main():
    """Launch GUI application"""
    root = tk.Tk()
    
    # Configure style
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    
    # Create application
    app = BenchmarkGUI(root)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()