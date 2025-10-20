"""
Simple GUI for Image Compression Benchmark
gui.py

Basic Tkinter interface for running compression benchmarks.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from pathlib import Path
import threading
import sys
from typing import List

# Import benchmark components
from main import (
    CompressorFactory, 
    PluginLoader, 
    BenchmarkOrchestrator,
    CompressionLevel,
    BenchmarkResult
)


class BenchmarkGUI:
    """Simple GUI for running compression benchmarks"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compression Benchmark")
        self.root.geometry("900x700")
        
        # Initialize benchmark components
        self.project_root = Path(__file__).parent
        self.plugins_dir = self.project_root / "compressors"
        self.dataset_dir = self.project_root / "image_datasets"
        self.output_dir = self.project_root / "benchmark_results"
        self.libs_dir = self.project_root / "libs"
        
        # Load plugins
        PluginLoader.load_plugins_from_directory(self.plugins_dir)
        self.available_compressors = CompressorFactory.list_available()
        
        # Variables
        self.selected_images = []
        self.running = False
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # ===== HEADER =====
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(
            header_frame, 
            text="Image Compression Benchmark", 
            font=("Arial", 16, "bold")
        )
        title_label.pack()
        
        # ===== IMAGE SELECTION =====
        image_frame = ttk.LabelFrame(self.root, text="Images", padding="10")
        image_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        
        # Buttons for image selection
        btn_frame = ttk.Frame(image_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(
            btn_frame, 
            text="Add Images...",
            command=self.add_images
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame, 
            text="Add Folder...",
            command=self.add_folder
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame, 
            text="Clear All",
            command=self.clear_images
        ).pack(side=tk.LEFT, padx=5)
        
        # Image list
        self.image_listbox = tk.Listbox(image_frame, height=6)
        self.image_listbox.pack(fill=tk.BOTH, expand=True)
        
        self.image_count_label = ttk.Label(image_frame, text="0 images selected")
        self.image_count_label.pack(anchor=tk.W, pady=(5, 0))
        
        # ===== COMPRESSOR SELECTION =====
        comp_frame = ttk.LabelFrame(self.root, text="Compressors", padding="10")
        comp_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create checkboxes for each compressor
        self.compressor_vars = {}
        
        if not self.available_compressors:
            ttk.Label(
                comp_frame, 
                text="⚠️ No compressors found! Check plugins folder.",
                foreground="red"
            ).pack()
        else:
            # Split into columns
            cols = 3
            rows = (len(self.available_compressors) + cols - 1) // cols
            
            for idx, comp_name in enumerate(sorted(self.available_compressors)):
                row = idx % rows
                col = idx // rows
                
                var = tk.BooleanVar(value=True)
                self.compressor_vars[comp_name] = var
                
                cb = ttk.Checkbutton(
                    comp_frame,
                    text=comp_name,
                    variable=var
                )
                cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
        
        # Select/Deselect all buttons
        select_frame = ttk.Frame(comp_frame)
        select_frame.grid(row=rows+1, column=0, columnspan=cols, pady=(10, 0))
        
        ttk.Button(
            select_frame,
            text="Select All",
            command=self.select_all_compressors
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            select_frame,
            text="Deselect All",
            command=self.deselect_all_compressors
        ).pack(side=tk.LEFT, padx=5)
        
        # ===== COMPRESSION LEVEL =====
        level_frame = ttk.LabelFrame(self.root, text="⚙️ Compression Levels", padding="10")
        level_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.level_vars = {}
        levels = [
            ("FASTEST", CompressionLevel.FASTEST),
            ("FAST", CompressionLevel.FAST),
            ("BALANCED", CompressionLevel.BALANCED),
            ("GOOD", CompressionLevel.GOOD),
            ("BEST", CompressionLevel.BEST)
        ]
        
        for idx, (name, level) in enumerate(levels):
            var = tk.BooleanVar(value=(level == CompressionLevel.BALANCED))
            self.level_vars[level] = var
            
            ttk.Checkbutton(
                level_frame,
                text=name,
                variable=var
            ).grid(row=0, column=idx, padx=10)
        
        # ===== CONTROL BUTTONS =====
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        self.run_button = ttk.Button(
            control_frame,
            text="▶️ Run Benchmark",
            command=self.run_benchmark,
            style="Accent.TButton"
        )
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            control_frame,
            text="⏹️ Stop",
            command=self.stop_benchmark,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="📁 Open Results Folder",
            command=self.open_results
        ).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            control_frame,
            mode='indeterminate'
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # ===== LOG OUTPUT =====
        log_frame = ttk.LabelFrame(self.root, text="📊 Output", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            wrap=tk.WORD,
            font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial log message
        self.log("Welcome to Image Compression Benchmark!")
        self.log(f"Available compressors: {', '.join(self.available_compressors)}")
        self.log("Add images and select compressors to begin.\n")
    
    def log(self, message):
        """Add message to log output"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def add_images(self):
        """Add individual images"""
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        for file in files:
            path = Path(file)
            if path not in self.selected_images:
                self.selected_images.append(path)
                self.image_listbox.insert(tk.END, path.name)
        
        self.update_image_count()
    
    def add_folder(self):
        """Add all images from a folder"""
        folder = filedialog.askdirectory(title="Select Folder")
        
        if folder:
            folder_path = Path(folder)
            patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]
            
            for pattern in patterns:
                for file in folder_path.glob(pattern):
                    if file not in self.selected_images:
                        self.selected_images.append(file)
                        self.image_listbox.insert(tk.END, file.name)
            
            self.update_image_count()
    
    def clear_images(self):
        """Clear all selected images"""
        self.selected_images = []
        self.image_listbox.delete(0, tk.END)
        self.update_image_count()
    
    def update_image_count(self):
        """Update image count label"""
        count = len(self.selected_images)
        self.image_count_label.config(text=f"{count} image{'s' if count != 1 else ''} selected")
    
    def select_all_compressors(self):
        """Select all compressors"""
        for var in self.compressor_vars.values():
            var.set(True)
    
    def deselect_all_compressors(self):
        """Deselect all compressors"""
        for var in self.compressor_vars.values():
            var.set(False)
    
    def open_results(self):
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
        else:  # Linux
            os.system(f'xdg-open "{results_path}"')
    
    def run_benchmark(self):
        """Run benchmark in separate thread"""
        # Validate inputs
        if not self.selected_images:
            messagebox.showwarning("No Images", "Please select at least one image.")
            return
        
        selected_compressors = [
            name for name, var in self.compressor_vars.items() if var.get()
        ]
        
        if not selected_compressors:
            messagebox.showwarning("No Compressors", "Please select at least one compressor.")
            return
        
        selected_levels = [
            level for level, var in self.level_vars.items() if var.get()
        ]
        
        if not selected_levels:
            messagebox.showwarning("No Levels", "Please select at least one compression level.")
            return
        
        # Disable controls
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start()
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        self.log("=" * 70)
        self.log("Starting Benchmark...")
        self.log("=" * 70)
        self.log(f"Images: {len(self.selected_images)}")
        self.log(f"Compressors: {', '.join(selected_compressors)}")
        self.log(f"Levels: {', '.join(l.name for l in selected_levels)}")
        self.log("")
        
        # Run in thread
        thread = threading.Thread(
            target=self._run_benchmark_thread,
            args=(selected_compressors, selected_levels),
            daemon=True
        )
        thread.start()
    
    def _run_benchmark_thread(self, compressors: List[str], levels: List[CompressionLevel]):
        """Run benchmark in background thread"""
        try:
            # Create orchestrator
            orchestrator = BenchmarkOrchestrator(
                dataset_dir=self.dataset_dir,
                output_dir=self.output_dir,
                libs_dir=self.libs_dir
            )
            
            # Temporarily redirect orchestrator output to our log
            for comp_name in compressors:
                if not self.running:
                    break
                
                self.log(f"\n{'='*70}")
                self.log(f"🔧 Testing: {comp_name}")
                self.log(f"{'='*70}")
                
                try:
                    lib_path = orchestrator._find_lib_for_compressor(comp_name)
                    compressor = CompressorFactory.create(comp_name, lib_path)
                    
                    for level in levels:
                        if not self.running:
                            break
                        
                        self.log(f"\n  📊 Level: {level.name}")
                        self.log(f"  {'-'*66}")
                        
                        for img_path in self.selected_images:
                            if not self.running:
                                break
                            
                            result = orchestrator._benchmark_single(
                                compressor, img_path, level
                            )
                            orchestrator.results.append(result)
                            
                            # Log result
                            m = result.metrics
                            if m.success:
                                self.log(f"    ✅ {img_path.name}")
                                self.log(f"       Size: {m.original_size:,} B → {m.compressed_size:,} B")
                                self.log(f"       Savings: {m.space_saving_percent:.1f}% | Ratio: {m.compression_ratio:.2f}x")
                                self.log(f"       Compression: {m.compression_time:.3f}s ({m.compression_speed_mbps:.1f} MB/s)")
                                self.log(f"       Decompression: {m.decompression_time:.3f}s ({m.decompression_speed_mbps:.1f} MB/s)")
                            else:
                                self.log(f"    ❌ {img_path.name}: {m.error_message}")
                        
                except Exception as e:
                    self.log(f"  ❌ Error: {str(e)}")
            
            # Export results
            if orchestrator.results and self.running:
                self.log("\n" + "="*70)
                self.log("💾 Saving results...")
                orchestrator.export_results(self.output_dir / "results.json")
                
                # Print summary
                self.log("\n" + "="*70)
                self.log("📊 SUMMARY")
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
                    
                    self.log(f"\n🔧 {format_name}")
                    self.log(f"   Compression Ratio: {avg_ratio:.2f}x")
                    self.log(f"   Space Savings: {avg_savings:.1f}%")
                    self.log(f"   Avg Compression Time: {avg_comp_time:.3f}s")
                
                self.log("\n✅ Benchmark completed!")
            elif not self.running:
                self.log("\n⚠️ Benchmark stopped by user.")
            else:
                self.log("\n⚠️ No results generated.")
                
        except Exception as e:
            self.log(f"\n❌ Error: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        
        finally:
            # Re-enable controls
            self.root.after(0, self._benchmark_finished)
    
    def _benchmark_finished(self):
        """Called when benchmark finishes"""
        self.running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()
    
    def stop_benchmark(self):
        """Stop running benchmark"""
        self.running = False
        self.log("\n⏹️ Stopping benchmark...")


def main():
    """Launch GUI application"""
    root = tk.Tk()
    
    # Set theme
    style = ttk.Style()
    try:
        style.theme_use('xpnative')  # Modern theme
    except:
        pass
    
    app = BenchmarkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()