"""
Image Compression Benchmark GUI
Main application window and orchestration
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from pathlib import Path
import threading
import sys
from typing import List, Dict

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

from utils.benchmark_visualization import (
    open_visualization_window,
    BenchmarkDataLoader,
    BenchmarkData
)


class BenchmarkGUI:
    """Main GUI application for image compression benchmark"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compression Benchmark")
        self.root.geometry("1400x900")
        
        # Initialize paths
        self.project_root = Path(__file__).parent
        self.plugins_dir = self.project_root / "compressors"
        self.dataset_dir = self.project_root / "image_datasets"
        self.output_dir = self.project_root / "benchmark_results"
        self.libs_dir = self.project_root / "libs"
        
        # Load plugins
        PluginLoader.load_plugins_from_directory(self.plugins_dir)
        
        # --- ZMĚNA: Načtení hezkých jmen pro GUI ---
        # Získáme interní klíče (např. 'charls', 'lzw')
        internal_keys = CompressorFactory.list_available()
        
        # Vytvoříme mapu: "Hezké jméno z property" -> "interní klíč"
        # Příklad: { "CharLS-JPEGLS": "charls", "Standard PNG": "png" }
        self.compressor_mapping: Dict[str, str] = {}
        
        for key in internal_keys:
            try:
                # Dočasně vytvoříme instanci kompresoru, abychom přečetli jeho property 'name'
                # Předpokládáme, že create() lze zavolat bez argumentů nebo má defaulty
                temp_instance = CompressorFactory.create(key)
                display_name = temp_instance.name  # Tady se volá vaše @property def name
                self.compressor_mapping[display_name] = key
            except Exception as e:
                print(f"Warning: Could not load name for compressor '{key}': {e}")
                # Fallback: pokud selže instanciace, použijeme interní klíč
                self.compressor_mapping[key] = key

        # Seznam jmen, která chceme zobrazit v GUI (klíče naší mapy)
        self.available_display_names = list(self.compressor_mapping.keys())
        # -------------------------------------------
        
        # State
        self.running = False
        self.runner = None
        self.last_json_path = None # Path to last results JSON
        
        # Create UI
        self.create_widgets()
    
    def create_widgets(self):
        """Create all GUI components"""
        # Header
        self.create_header()
        
        # Main container - two columns
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left column - controls (60% width)
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 2))
        
        # Right column - log output (40% width)
        right_frame = ttk.Frame(main_container, width=500)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 5))
        right_frame.pack_propagate(False)  # Maintain width
        
        # === LEFT COLUMN CONTENTS ===
        
        # Image selection
        self.image_widget = ImageSelectionWidget(left_frame)
        self.image_widget.add_images_btn.config(command=self.add_images)
        self.image_widget.add_folder_btn.config(command=self.add_folder)
        self.image_widget.clear_btn.config(command=self.image_widget.clear_images)
        self.image_widget.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # Compressor selection
        # --- ZMĚNA: Předáváme seznam hezkých jmen (display names) ---
        self.compressor_widget = CompressorSelectionWidget(
            left_frame,
            self.available_display_names
        )
        self.compressor_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Level selection
        levels = [
            ("FASTEST", CompressionLevel.FASTEST),
            ("BALANCED", CompressionLevel.BALANCED),
            ("BEST", CompressionLevel.BEST)
        ]
        self.level_widget = LevelSelectionWidget(left_frame, levels)
        # Set BALANCED as default
        self.level_widget.level_vars[CompressionLevel.BALANCED].set(True)
        self.level_widget.pack(fill=tk.X, padx=5, pady=5)
        
        # Iteration settings
        self.create_iteration_settings(left_frame)
        
        # Advanced settings (NEW)
        self.create_advanced_settings(left_frame)
        
        # Control buttons
        self.create_controls(left_frame)
        
        # === RIGHT COLUMN CONTENTS ===
        
        # Log output
        self.create_log(right_frame)
        
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
    
    def create_iteration_settings(self, parent):
        """Create iteration configuration section"""
        iter_frame = ttk.LabelFrame(parent, text="Benchmark Settings", padding="10")
        iter_frame.pack(fill=tk.X, padx=5, pady=5)
        
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
        
        # AUTO-VISUALIZATION
        viz_row = ttk.Frame(iter_frame)
        viz_row.pack(fill=tk.X, pady=2)
        
        self.auto_visualize_var = tk.BooleanVar(value=True)  # Defaultně zapnuto
        ttk.Checkbutton(
            viz_row,
            text="Auto-open Visualization after completion",
            variable=self.auto_visualize_var
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            viz_row,
            text="(Automatically show charts when benchmark finishes)",
            foreground="gray"
        ).pack(side=tk.LEFT, padx=5)
    
    def create_advanced_settings(self, parent):
        """Create advanced settings section"""
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Settings", padding="10")
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Resource monitoring
        monitor_row = ttk.Frame(advanced_frame)
        monitor_row.pack(fill=tk.X, pady=2)
        
        self.monitor_resources_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            monitor_row,
            text="Monitor System Resources (CPU, RAM, I/O)",
            variable=self.monitor_resources_var
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            monitor_row,
            text="- Tracks resource usage during compression",
            foreground="gray"
        ).pack(side=tk.LEFT, padx=5)
        
        # Process isolation
        isolate_row = ttk.Frame(advanced_frame)
        isolate_row.pack(fill=tk.X, pady=2)
        
        self.isolate_process_var = tk.BooleanVar(value=False)
        isolate_cb = ttk.Checkbutton(
            isolate_row,
            text="Process Isolation (High Priority)",
            variable=self.isolate_process_var
        )
        isolate_cb.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            isolate_row,
            text="- Sets high process priority for more accurate measurements",
            foreground="gray"
        ).pack(side=tk.LEFT, padx=5)
        
        # Warning for isolation
        warning_row = ttk.Frame(advanced_frame)
        warning_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(
            warning_row,
            text="⚠️ Process Isolation may require administrator privileges and affect system responsiveness",
            foreground="orange",
            font=("Arial", 8)
        ).pack(side=tk.LEFT, padx=20)
    
    def create_controls(self, parent):
        """Create control button section"""
        control_frame = ttk.Frame(parent, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Top row - action buttons
        button_row = ttk.Frame(control_frame)
        button_row.pack(fill=tk.X, pady=(0, 5))
        
        self.run_button = ttk.Button(
            button_row,
            text="▶ Run Benchmark",
            command=self.run_benchmark
        )
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            button_row,
            text="⏹ Stop",
            command=self.stop_benchmark,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_row,
            text="📊 Visualization",
            command=self.open_visualization
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_row,
            text="✓ Verify Results",
            command=self.show_verification_results
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_row,
            text="📁 Results Folder",
            command=self.open_results_folder
        ).pack(side=tk.LEFT, padx=5)
        
        # Bottom row - progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=5)
    
    def create_log(self, parent):
        """Create log output section"""
        log_frame = ttk.LabelFrame(parent, text="Console Output", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#f5f5f5"
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
        self.log(f"Available compressors: {', '.join(self.available_display_names)}")
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
        elif system == "Darwin":  # macOS
            os.system(f'open "{results_path}"')
        else:  # Linux and others
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
        
        # Získání vybraných jmen z GUI (např. ["CharLS-JPEGLS"])
        selected_display_names = self.compressor_widget.get_selected()
        
        if not selected_display_names:
            messagebox.showwarning("No Compressors", "Please select at least one compressor.")
            return
            
        # --- ZMĚNA: Překlad hezkých jmen zpět na interní klíče pro konfiguraci ---
        # Mapujeme ["CharLS-JPEGLS"] -> ["charls"]
        selected_compressors_keys = [
            self.compressor_mapping[name] for name in selected_display_names
        ]
        
        selected_levels = self.level_widget.get_selected()
        if not selected_levels:
            messagebox.showwarning("No Levels", "Please select at least one compression level.")
            return
        
        verify_enabled = self.level_widget.is_verification_enabled()
        num_iterations = self.iterations_var.get()
        warmup_iterations = self.warmup_var.get()
        strip_metadata = self.level_widget.is_strip_metadata_enabled()
        monitor_resources = self.monitor_resources_var.get()
        isolate_process = self.isolate_process_var.get()
        
        # Warning for process isolation
        if isolate_process:
            result = messagebox.askyesno(
                "Process Isolation Warning",
                "Process isolation will set this application to high priority.\n\n"
                "This may:\n"
                "• Require administrator/root privileges\n"
                "• Make your system less responsive during benchmark\n"
                "• Provide more accurate measurements\n\n"
                "Continue with process isolation?",
                icon='warning'
            )
            if not result:
                self.isolate_process_var.set(False)
                isolate_process = False
        
        # Create configuration (používáme interní klíče)
        config = BenchmarkConfig(
            dataset_dir=self.dataset_dir,
            output_dir=self.output_dir,
            libs_dir=self.libs_dir,
            compressor_names=selected_compressors_keys, # <-- ZDE POUŽIJEME KLÍČE
            image_paths=list(self.image_widget.selected_images),
            compression_levels=selected_levels,
            verify_lossless=verify_enabled,
            strip_metadata=strip_metadata,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
            monitor_resources=monitor_resources,
            isolate_process=isolate_process
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
            args=(config,),
            daemon=True
        )
        thread.start()
    
    def _run_benchmark_thread(self, config: BenchmarkConfig):
        """Execute benchmark in background thread"""
        try:
            # Run benchmark with log callback
            results, verification_results = self.runner.run(
                progress_callback=lambda msg: self.root.after(0, self.log, msg)
            )
            
            # Save and summarize results
            if results and self.running:
                self.root.after(0, self._save_and_summarize, results, verification_results, config)
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
    
    def _save_and_summarize(self, results, verification_results, config):
        """Save results and display summary"""
        self.log("\n" + "="*70)
        self.log("Saving results...")
        
        # Save to JSON with unique filename in json_reports subdirectory
        json_reports_dir = self.output_dir / "json_reports"
        json_path = BenchmarkSummarizer.export_results_json(
            results,
            verification_results,
            json_reports_dir,
            config
        )
        
        # Store path for visualization
        self.last_json_path = json_path
        
        self.log(f"💾 Results saved to: {json_path.name}")
        self.log(f"   Full path: {json_path}")
        
        # Print summaries
        BenchmarkSummarizer.print_compression_summary(results, self.log)
        BenchmarkSummarizer.print_verification_summary(verification_results, self.log)
        
        # Print scenario analysis if resource monitoring was enabled
        if config.monitor_resources:
            BenchmarkSummarizer.print_scenario_analysis(results, self.log)
        
        self.log("\n✅ Benchmark completed successfully!")
        
        # AUTO-OPEN VISUALIZATION
        if self.auto_visualize_var.get():
            self.log("\n📊 Opening visualization window with auto-generated charts...")
            try:
                data = BenchmarkDataLoader.load_from_file(json_path)
                if data:
                    self.root.after(500, lambda: self.open_visualization_with_data(data, auto_show=True))
            except Exception as e:
                self.log(f"⚠️  Could not auto-open visualization: {e}")
    
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
        
    def open_visualization(self):
        """Open visualization window manually with file dialog"""
        json_reports_dir = self.output_dir / "json_reports"
        json_reports_dir.mkdir(exist_ok=True, parents=True)
        
        initial_dir = str(json_reports_dir) if json_reports_dir.exists() else str(self.output_dir)
        
        filename = filedialog.askopenfilename(
            title="Select Benchmark JSON",
            initialdir=initial_dir,
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            data = BenchmarkDataLoader.load_from_file(Path(filename))
            if data:
                self.open_visualization_with_data(data, auto_show=False)
                self.log(f"📊 Opened visualization for: {Path(filename).name}")
            else:
                messagebox.showerror("Error", "Failed to load JSON file")

    def open_visualization_with_data(self, data: BenchmarkData, auto_show: bool = False):
        """
        Open visualization window with data and optionally auto-show charts
        """
        viz_window = open_visualization_window(self.root, data)
        
        if auto_show:
            self.root.after(200, viz_window.show_compression_ratio)


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