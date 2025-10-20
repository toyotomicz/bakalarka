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
    BenchmarkOrchestrator,
    CompressionLevel
)
from utils.verification import ImageVerifier, VerificationResult
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
        self.root.geometry("1000x800")
        
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
        self.verification_results: Dict[tuple, VerificationResult] = {}
        
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
            patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]
            
            for pattern in patterns:
                for file in folder_path.glob(pattern):
                    self.image_widget.add_image(file)
    
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
        if not self.verification_results:
            messagebox.showinfo(
                "No Results",
                "No verification results available. Run a benchmark first."
            )
            return
        
        VerificationResultsWidget(self.root, self.verification_results)
    
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
        
        # Clear previous results
        self.verification_results = {}
        
        # Update UI state
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start()
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        self.log("=" * 70)
        self.log("Starting Benchmark")
        self.log("=" * 70)
        self.log(f"Images: {len(self.image_widget.selected_images)}")
        self.log(f"Compressors: {', '.join(selected_compressors)}")
        self.log(f"Levels: {', '.join(l.name for l in selected_levels)}")
        
        verify_enabled = self.level_widget.is_verification_enabled()
        self.log(f"Verification: {'Enabled' if verify_enabled else 'Disabled'}")
        self.log("")
        
        # Run in background thread
        thread = threading.Thread(
            target=self._run_benchmark_thread,
            args=(selected_compressors, selected_levels, verify_enabled),
            daemon=True
        )
        thread.start()
    
    def _run_benchmark_thread(
        self,
        compressors: List[str],
        levels: List[CompressionLevel],
        verify: bool
    ):
        """Execute benchmark in background thread"""
        try:
            orchestrator = BenchmarkOrchestrator(
                dataset_dir=self.dataset_dir,
                output_dir=self.output_dir,
                libs_dir=self.libs_dir
            )
            
            for comp_name in compressors:
                if not self.running:
                    break
                
                self.log(f"\n{'='*70}")
                self.log(f"Testing: {comp_name}")
                self.log(f"{'='*70}")
                
                try:
                    lib_path = orchestrator._find_lib_for_compressor(comp_name)
                    compressor = CompressorFactory.create(comp_name, lib_path)
                    
                    for level in levels:
                        if not self.running:
                            break
                        
                        self.log(f"\n  Compression Level: {level.name}")
                        self.log(f"  {'-'*66}")
                        
                        for img_path in self.image_widget.selected_images:
                            if not self.running:
                                break
                            
                            # Run compression
                            result = orchestrator._benchmark_single(
                                compressor, img_path, level
                            )
                            orchestrator.results.append(result)
                            
                            m = result.metrics
                            if m.success:
                                self.log(f"    {img_path.name}")
                                self.log(f"       Size: {m.original_size:,} B -> {m.compressed_size:,} B")
                                self.log(f"       Savings: {m.space_saving_percent:.1f}% | Ratio: {m.compression_ratio:.2f}x")
                                self.log(f"       Compression: {m.compression_time:.3f}s ({m.compression_speed_mbps:.1f} MB/s)")
                                self.log(f"       Decompression: {m.decompression_time:.3f}s ({m.decompression_speed_mbps:.1f} MB/s)")
                                
                                # Verify lossless if enabled
                                if verify:
                                    format_dir = self.output_dir / compressor.name
                                    compressed_path = format_dir / f"{img_path.stem}{compressor.extension}"
                                    
                                    if compressed_path.exists():
                                        verification = ImageVerifier.verify_lossless(
                                            img_path, compressed_path
                                        )
                                        
                                        key = (img_path.name, comp_name)
                                        self.verification_results[key] = verification
                                        
                                        if verification.is_lossless:
                                            self.log(f"       Verification: LOSSLESS (100.0000% accurate)")
                                        else:
                                            self.log(f"       Verification: LOSSY")
                                            self.log(f"          Max difference: {verification.max_difference:.2f}")
                                            self.log(f"          Different pixels: {verification.different_pixels:,} / {verification.total_pixels:,}")
                                            self.log(f"          Accuracy: {verification.accuracy_percent:.4f}%")
                            else:
                                self.log(f"    {img_path.name}: FAILED - {m.error_message}")
                        
                except Exception as e:
                    self.log(f"  Error: {str(e)}")
            
            # Save and summarize results
            if orchestrator.results and self.running:
                self.save_and_summarize(orchestrator, verify)
            elif not self.running:
                self.log("\nBenchmark stopped by user.")
            else:
                self.log("\nNo results generated.")
                
        except Exception as e:
            self.log(f"\nError: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        
        finally:
            self.root.after(0, self._benchmark_finished)
    
    def save_and_summarize(self, orchestrator, verify: bool):
        """Save results and display summary"""
        self.log("\n" + "="*70)
        self.log("Saving results...")
        orchestrator.export_results(self.output_dir / "results.json")
        
        # Compression summary
        self.log("\n" + "="*70)
        self.log("COMPRESSION SUMMARY")
        self.log("="*70)
        
        by_format = {}
        for result in orchestrator.results:
            if result.metrics.success:
                if result.format_name not in by_format:
                    by_format[result.format_name] = []
                by_format[result.format_name].append(result.metrics)
        
        for format_name, metrics_list in sorted(by_format.items()):
            avg_ratio = sum(m.compression_ratio for m in metrics_list) / len(metrics_list)
            avg_savings = sum(m.space_saving_percent for m in metrics_list) / len(metrics_list)
            avg_comp_time = sum(m.compression_time for m in metrics_list) / len(metrics_list)
            avg_decomp_time = sum(m.decompression_time for m in metrics_list) / len(metrics_list)
            
            self.log(f"\n{format_name}")
            self.log(f"   Compression Ratio: {avg_ratio:.2f}x")
            self.log(f"   Space Savings: {avg_savings:.1f}%")
            self.log(f"   Avg Compression Time: {avg_comp_time:.3f}s")
            self.log(f"   Avg Decompression Time: {avg_decomp_time:.3f}s")
        
        # Verification summary
        if verify and self.verification_results:
            self.log("\n" + "="*70)
            self.log("VERIFICATION SUMMARY")
            self.log("="*70)
            
            lossless_count = sum(
                1 for v in self.verification_results.values()
                if v.is_lossless
            )
            total_count = len(self.verification_results)
            lossy_count = total_count - lossless_count
            
            self.log(f"Total Tests: {total_count}")
            self.log(f"Truly Lossless: {lossless_count} ({100*lossless_count/total_count:.1f}%)")
            self.log(f"Lossy: {lossy_count} ({100*lossy_count/total_count:.1f}%)")
            
            if lossy_count > 0:
                self.log("\nLossy compressions detected:")
                for key, verification in self.verification_results.items():
                    if not verification.is_lossless:
                        img_name, comp_name = key
                        self.log(f"   {img_name} with {comp_name}")
                        self.log(f"      Max diff: {verification.max_difference:.2f}, "
                               f"Different pixels: {verification.different_pixels:,}")
        
        self.log("\nBenchmark completed successfully.")
    
    def _benchmark_finished(self):
        """Reset UI state after benchmark completion"""
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()
    
    def stop_benchmark(self):
        """Stop running benchmark"""
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