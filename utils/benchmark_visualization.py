"""
Benchmark Visualization Module
Creates scientific-quality charts and graphs from benchmark results

Supports export to PDF, SVG, and PNG formats.
Uses matplotlib with seaborn styling for publication-ready figures.
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np


# Configure matplotlib and seaborn for scientific plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


@dataclass
class BenchmarkData:
    """Parsed benchmark data structure"""
    config: Dict
    summary: Dict
    results: List[Dict]
    verification: List[Dict]
    scenarios: Dict
    metadata: Dict


class BenchmarkDataLoader:
    """Loads and parses benchmark JSON files"""
    
    @staticmethod
    def load_from_file(json_path: Path) -> Optional[BenchmarkData]:
        """
        Load benchmark data from JSON file
        
        Args:
            json_path: Path to benchmark JSON file
            
        Returns:
            BenchmarkData object or None if loading fails
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return BenchmarkData(
                config=data.get('benchmark_config', {}),
                summary=data.get('summary', {}),
                results=data.get('results', []),
                verification=data.get('verification', []),
                scenarios=data.get('scenarios', {}),
                metadata=data.get('benchmark_info', {})
            )
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return None


class ChartGenerator:
    """Generates various chart types from benchmark data"""
    
    @staticmethod
    def create_compression_ratio_comparison(data: BenchmarkData) -> Figure:
        """
        Bar chart comparing compression ratios across formats
        
        Args:
            data: Benchmark data
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by format
        formats = {}
        for result in data.results:
            if result['compression']['success']:
                fmt = result['format']
                if fmt not in formats:
                    formats[fmt] = []
                formats[fmt].append(result['compression']['compression_ratio'])
        
        # Calculate averages
        format_names = list(formats.keys())
        avg_ratios = [np.mean(formats[fmt]) for fmt in format_names]
        std_ratios = [np.std(formats[fmt]) for fmt in format_names]
        
        # Create bar chart
        x_pos = np.arange(len(format_names))
        bars = ax.bar(x_pos, avg_ratios, yerr=std_ratios, 
                      capsize=5, alpha=0.8, edgecolor='black')
        
        # Color bars based on performance
        colors = ['green' if r > 1.0 else 'orange' if r > 0.95 else 'red' 
                  for r in avg_ratios]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for i, (bar, ratio) in enumerate(zip(bars, avg_ratios)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.3f}x',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Compression Format', fontweight='bold')
        ax.set_ylabel('Compression Ratio (higher is better)', fontweight='bold')
        ax.set_title('Compression Ratio Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(format_names, rotation=45, ha='right')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No compression')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_speed_comparison(data: BenchmarkData) -> Figure:
        """
        Grouped bar chart comparing compression and decompression speeds
        
        Args:
            data: Benchmark data
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by format
        formats = {}
        for result in data.results:
            if result['compression']['success']:
                fmt = result['format']
                if fmt not in formats:
                    formats[fmt] = {'comp': [], 'decomp': []}
                formats[fmt]['comp'].append(result['compression']['compression_speed_mbps'])
                formats[fmt]['decomp'].append(result['compression']['decompression_speed_mbps'])
        
        # Calculate averages
        format_names = list(formats.keys())
        comp_speeds = [np.mean(formats[fmt]['comp']) for fmt in format_names]
        decomp_speeds = [np.mean(formats[fmt]['decomp']) for fmt in format_names]
        
        # Create grouped bar chart
        x_pos = np.arange(len(format_names))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, comp_speeds, width, 
                       label='Compression', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x_pos + width/2, decomp_speeds, width,
                       label='Decompression', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Compression Format', fontweight='bold')
        ax.set_ylabel('Speed (MB/s)', fontweight='bold')
        ax.set_title('Compression vs Decompression Speed', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(format_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_resource_usage_chart(data: BenchmarkData) -> Figure:
        """
        Multi-panel chart showing CPU, RAM, and I/O usage
        
        Args:
            data: Benchmark data
            
        Returns:
            Matplotlib Figure object with 3 subplots
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        
        # Collect data by format
        formats = {}
        for result in data.results:
            if result['compression']['success'] and 'system_metrics' in result:
                fmt = result['format']
                if fmt not in formats:
                    formats[fmt] = {'cpu': [], 'ram': [], 'io': []}
                
                sm = result['system_metrics']
                formats[fmt]['cpu'].append(sm['cpu']['avg_process_percent'])
                formats[fmt]['ram'].append(sm['memory']['peak_mb'])
                formats[fmt]['io'].append(sm['io']['total_mb'])
        
        if not formats:
            # No system metrics available
            ax1.text(0.5, 0.5, 'No system metrics available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'Enable resource monitoring', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax3.text(0.5, 0.5, 'in benchmark settings', 
                    ha='center', va='center', transform=ax3.transAxes)
            return fig
        
        format_names = list(formats.keys())
        x_pos = np.arange(len(format_names))
        
        # CPU Usage
        cpu_avg = [np.mean(formats[fmt]['cpu']) for fmt in format_names]
        bars1 = ax1.bar(x_pos, cpu_avg, alpha=0.8, edgecolor='black', color='skyblue')
        ax1.set_xlabel('Format', fontweight='bold')
        ax1.set_ylabel('CPU Usage (%)', fontweight='bold')
        ax1.set_title('Average CPU Usage', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(format_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # RAM Usage
        ram_avg = [np.mean(formats[fmt]['ram']) for fmt in format_names]
        bars2 = ax2.bar(x_pos, ram_avg, alpha=0.8, edgecolor='black', color='lightcoral')
        ax2.set_xlabel('Format', fontweight='bold')
        ax2.set_ylabel('Peak RAM (MB)', fontweight='bold')
        ax2.set_title('Peak RAM Usage', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(format_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # I/O Usage
        io_avg = [np.mean(formats[fmt]['io']) for fmt in format_names]
        bars3 = ax3.bar(x_pos, io_avg, alpha=0.8, edgecolor='black', color='lightgreen')
        ax3.set_xlabel('Format', fontweight='bold')
        ax3.set_ylabel('Total I/O (MB)', fontweight='bold')
        ax3.set_title('Total I/O Operations', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(format_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_scatter_ratio_vs_speed(data: BenchmarkData) -> Figure:
        """
        Scatter plot showing compression ratio vs speed tradeoff
        
        Args:
            data: Benchmark data
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Collect data by format
        formats = {}
        for result in data.results:
            if result['compression']['success']:
                fmt = result['format']
                if fmt not in formats:
                    formats[fmt] = {'ratio': [], 'speed': []}
                formats[fmt]['ratio'].append(result['compression']['compression_ratio'])
                formats[fmt]['speed'].append(result['compression']['compression_speed_mbps'])
        
        # Plot each format
        colors = plt.cm.tab10(np.linspace(0, 1, len(formats)))
        
        for (fmt, values), color in zip(formats.items(), colors):
            ratios = values['ratio']
            speeds = values['speed']
            
            # Calculate average point
            avg_ratio = np.mean(ratios)
            avg_speed = np.mean(speeds)
            
            # Plot individual points
            ax.scatter(speeds, ratios, alpha=0.4, s=50, color=color)
            
            # Plot average point
            ax.scatter(avg_speed, avg_ratio, s=200, marker='*', 
                      edgecolor='black', linewidth=2, color=color, 
                      label=fmt, zorder=10)
        
        ax.set_xlabel('Compression Speed (MB/s)', fontweight='bold')
        ax.set_ylabel('Compression Ratio (higher is better)', fontweight='bold')
        ax.set_title('Compression Ratio vs Speed Tradeoff', fontsize=14, fontweight='bold')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No compression')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add quadrant labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(xlim[1]*0.95, ylim[1]*0.95, 'Ideal\n(Fast & High Ratio)', 
               ha='right', va='top', fontsize=10, alpha=0.5, style='italic')
        ax.text(xlim[0]*1.05, ylim[0]*1.05, 'Poor\n(Slow & Low Ratio)', 
               ha='left', va='bottom', fontsize=10, alpha=0.5, style='italic')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_verification_summary(data: BenchmarkData) -> Figure:
        """
        Pie chart showing lossless verification results
        
        Args:
            data: Benchmark data
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if not data.verification:
            ax.text(0.5, 0.5, 'No verification data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        lossless = sum(1 for v in data.verification if v['is_lossless'])
        lossy = len(data.verification) - lossless
        
        if lossy == 0:
            # All lossless - show special chart
            colors = ['#2ecc71']
            labels = ['100% Lossless']
            sizes = [100]
            explode = [0.1]
        else:
            colors = ['#2ecc71', '#e74c3c']
            labels = [f'Lossless ({lossless})', f'Lossy ({lossy})']
            sizes = [lossless, lossy]
            explode = [0.1, 0]
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels,
                                           colors=colors, autopct='%1.1f%%',
                                           shadow=True, startangle=90)
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        
        ax.set_title('Lossless Verification Results', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_detailed_performance_heatmap(data: BenchmarkData) -> Figure:
        """
        Heatmap showing performance across images and formats
        
        Args:
            data: Benchmark data
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Build matrix: formats x images
        formats = []
        images = []
        
        # Collect unique formats and images
        for result in data.results:
            if result['compression']['success']:
                fmt = result['format']
                img = result['image']
                if fmt not in formats:
                    formats.append(fmt)
                if img not in images:
                    images.append(img)
        
        # Create matrix
        matrix = np.zeros((len(formats), len(images)))
        
        for result in data.results:
            if result['compression']['success']:
                fmt_idx = formats.index(result['format'])
                img_idx = images.index(result['image'])
                matrix[fmt_idx, img_idx] = result['compression']['compression_ratio']
        
        # Create heatmap
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.8, vmax=1.2)
        
        # Set ticks
        ax.set_xticks(np.arange(len(images)))
        ax.set_yticks(np.arange(len(formats)))
        ax.set_xticklabels(images, rotation=45, ha='right')
        ax.set_yticklabels(formats)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Compression Ratio', fontweight='bold')
        
        # Add values in cells
        for i in range(len(formats)):
            for j in range(len(images)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", 
                              fontsize=8, fontweight='bold')
        
        ax.set_title('Compression Ratio Heatmap (Format vs Image)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Images', fontweight='bold')
        ax.set_ylabel('Formats', fontweight='bold')
        
        plt.tight_layout()
        return fig


class VisualizationExporter:
    """Handles exporting figures to various formats"""
    
    @staticmethod
    def export_to_pdf(figures: List[Tuple[str, Figure]], output_path: Path):
        """
        Export multiple figures to a single PDF file
        
        Args:
            figures: List of (title, figure) tuples
            output_path: Output PDF file path
        """
        with PdfPages(output_path) as pdf:
            for title, fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Benchmark Visualization Report'
            d['Author'] = 'Image Compression Benchmark Tool'
            d['Subject'] = 'Performance Analysis'
            d['Keywords'] = 'Compression, Benchmark, Performance'
            d['CreationDate'] = datetime.now()
    
    @staticmethod
    def export_to_png(figure: Figure, output_path: Path, dpi: int = 300):
        """
        Export single figure to PNG
        
        Args:
            figure: Matplotlib Figure object
            output_path: Output PNG file path
            dpi: Resolution in dots per inch
        """
        figure.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                      facecolor='white', edgecolor='none')
        plt.close(figure)
    
    @staticmethod
    def export_to_svg(figure: Figure, output_path: Path):
        """
        Export single figure to SVG (vector format)
        
        Args:
            figure: Matplotlib Figure object
            output_path: Output SVG file path
        """
        figure.savefig(output_path, format='svg', bbox_inches='tight')
        plt.close(figure)


class VisualizationWindow:
    """GUI window for visualization and export"""
    
    def __init__(self, parent, data: Optional[BenchmarkData] = None):
        """
        Initialize visualization window
        
        Args:
            parent: Parent Tkinter window
            data: Optional benchmark data to display
        """
        self.window = tk.Toplevel(parent)
        self.window.title("Benchmark Visualization")
        self.window.geometry("1200x800")
        
        self.data = data
        self.current_figure = None
        
        self.create_widgets()
        
        if data:
            self.load_data(data)
    
    def create_widgets(self):
        """Create GUI components"""
        # Top frame - controls
        control_frame = ttk.Frame(self.window, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Load JSON button
        ttk.Button(
            control_frame,
            text="📂 Load Benchmark JSON",
            command=self.load_json_file
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Export buttons
        ttk.Label(control_frame, text="Export:").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="PDF (All Charts)",
            command=self.export_all_to_pdf
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            control_frame,
            text="PNG (Current)",
            command=self.export_current_to_png
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            control_frame,
            text="SVG (Current)",
            command=self.export_current_to_svg
        ).pack(side=tk.LEFT, padx=2)
        
        # Middle frame - chart selection
        select_frame = ttk.LabelFrame(self.window, text="Select Chart Type", padding="10")
        select_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.chart_buttons = []
        chart_types = [
            ("📊 Compression Ratio Comparison", self.show_compression_ratio),
            ("⚡ Speed Comparison", self.show_speed_comparison),
            ("💻 Resource Usage (CPU/RAM/IO)", self.show_resource_usage),
            ("📈 Ratio vs Speed Tradeoff", self.show_scatter_plot),
            ("✓ Verification Summary", self.show_verification),
            ("🔥 Performance Heatmap", self.show_heatmap)
        ]
        
        for text, command in chart_types:
            btn = ttk.Button(select_frame, text=text, command=command, width=30)
            btn.pack(side=tk.LEFT, padx=5)
            self.chart_buttons.append(btn)
        
        # Bottom frame - canvas for matplotlib
        self.canvas_frame = ttk.Frame(self.window)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initial message
        self.show_initial_message()
    
    def show_initial_message(self):
        """Display initial instruction message"""
        label = ttk.Label(
            self.canvas_frame,
            text="📊 Load a benchmark JSON file or select a chart type to begin",
            font=("Arial", 14),
            justify=tk.CENTER
        )
        label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def load_json_file(self):
        """Open file dialog to load JSON"""
        filename = filedialog.askopenfilename(
            title="Select Benchmark JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            data = BenchmarkDataLoader.load_from_file(Path(filename))
            if data:
                self.load_data(data)
                messagebox.showinfo("Success", f"Loaded: {Path(filename).name}")
            else:
                messagebox.showerror("Error", "Failed to load JSON file")
    
    def load_data(self, data: BenchmarkData):
        """Load benchmark data and enable buttons"""
        self.data = data
        for btn in self.chart_buttons:
            btn.config(state=tk.NORMAL)
    
    def clear_canvas(self):
        """Clear current matplotlib canvas"""
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
    
    def display_figure(self, fig: Figure):
        """Display matplotlib figure in canvas"""
        self.clear_canvas()
        self.current_figure = fig
        
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_compression_ratio(self):
        """Show compression ratio chart"""
        if not self.data:
            return
        fig = ChartGenerator.create_compression_ratio_comparison(self.data)
        self.display_figure(fig)
    
    def show_speed_comparison(self):
        """Show speed comparison chart"""
        if not self.data:
            return
        fig = ChartGenerator.create_speed_comparison(self.data)
        self.display_figure(fig)
    
    def show_resource_usage(self):
        """Show resource usage chart"""
        if not self.data:
            return
        fig = ChartGenerator.create_resource_usage_chart(self.data)
        self.display_figure(fig)
    
    def show_scatter_plot(self):
        """Show scatter plot"""
        if not self.data:
            return
        fig = ChartGenerator.create_scatter_ratio_vs_speed(self.data)
        self.display_figure(fig)
    
    def show_verification(self):
        """Show verification summary"""
        if not self.data:
            return
        fig = ChartGenerator.create_verification_summary(self.data)
        self.display_figure(fig)
    
    def show_heatmap(self):
        """Show performance heatmap"""
        if not self.data:
            return
        fig = ChartGenerator.create_detailed_performance_heatmap(self.data)
        self.display_figure(fig)
    
    def export_current_to_png(self):
        """Export current chart to PNG"""
        if not self.current_figure:
            messagebox.showwarning("No Chart", "Please display a chart first")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save as PNG",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")]
        )
        
        if filename:
            VisualizationExporter.export_to_png(self.current_figure, Path(filename))
            messagebox.showinfo("Success", f"Exported to: {Path(filename).name}")
    
    def export_current_to_svg(self):
        """Export current chart to SVG"""
        if not self.current_figure:
            messagebox.showwarning("No Chart", "Please display a chart first")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save as SVG",
            defaultextension=".svg",
            filetypes=[("SVG files", "*.svg")]
        )
        
        if filename:
            VisualizationExporter.export_to_svg(self.current_figure, Path(filename))
            messagebox.showinfo("Success", f"Exported to: {Path(filename).name}")
    
    def export_all_to_pdf(self):
        """Export all charts to single PDF"""
        if not self.data:
            messagebox.showwarning("No Data", "Please load benchmark data first")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save as PDF",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if filename:
            # Generate all figures
            figures = [
                ("Compression Ratio", ChartGenerator.create_compression_ratio_comparison(self.data)),
                ("Speed Comparison", ChartGenerator.create_speed_comparison(self.data)),
                ("Resource Usage", ChartGenerator.create_resource_usage_chart(self.data)),
                ("Ratio vs Speed", ChartGenerator.create_scatter_ratio_vs_speed(self.data)),
                ("Verification", ChartGenerator.create_verification_summary(self.data)),
                ("Performance Heatmap", ChartGenerator.create_detailed_performance_heatmap(self.data))
            ]
            
            VisualizationExporter.export_to_pdf(figures, Path(filename))
            messagebox.showinfo("Success", f"Exported {len(figures)} charts to PDF")


def open_visualization_window(parent, data: Optional[BenchmarkData] = None):
    """
    Open visualization window
    
    Args:
        parent: Parent Tkinter window
        data: Optional benchmark data
    """
    VisualizationWindow(parent, data)


# Standalone usage
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    app = VisualizationWindow(root)
    root.mainloop()