"""
Benchmark Visualization Module
utils/benchmark_visualization.py

Generates scientific-quality charts from benchmark JSON results and provides
a Tkinter window for interactive exploration and export.

Supported export formats: PDF (all charts), PNG, SVG, CSV.
Chart rendering uses matplotlib with seaborn styling.
"""

import csv
import json
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Global matplotlib / seaborn style
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams.update({
    "figure.figsize":  (10, 6),
    "font.size":       10,
    "axes.labelsize":  11,
    "axes.titlesize":  12,
    "legend.fontsize":  9,
})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkData:
    """Parsed content of a benchmark JSON report."""
    config:   Dict
    summary:  Dict
    results:  List[Dict]
    scenarios: Dict
    metadata: Dict


class BenchmarkDataLoader:
    """Loads and parses benchmark JSON report files."""

    @staticmethod
    def load_from_file(json_path: Path) -> Optional[BenchmarkData]:
        """
        Read a benchmark JSON file and return a BenchmarkData instance.
        Returns None on any parse or I/O error.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            return BenchmarkData(
                config    = data.get("benchmark_config", {}),
                summary   = data.get("summary",          {}),
                results   = data.get("results",          []),
                scenarios  = data.get("scenarios",        {}),
                metadata  = data.get("benchmark_info",   {}),
            )
        except Exception as exc:
            print(f"Error loading benchmark JSON: {exc}")
            return None


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

class ChartGenerator:
    """Static factory methods that produce matplotlib Figure objects."""

    @staticmethod
    def create_compression_ratio_comparison(data: BenchmarkData) -> Figure:
        """Bar chart: average compression ratio per format with std-dev error bars."""
        fig, ax = plt.subplots(figsize=(12, 6))

        formats: Dict[str, List[float]] = {}
        for r in data.results:
            if r["compression"]["success"]:
                formats.setdefault(r["format"], []).append(
                    r["compression"]["compression_ratio"]
                )

        names      = list(formats)
        avg_ratios = [np.mean(formats[f])  for f in names]
        std_ratios = [np.std(formats[f])   for f in names]

        x    = np.arange(len(names))
        bars = ax.bar(x, avg_ratios, yerr=std_ratios, capsize=5, alpha=0.8, edgecolor="black")

        # Colour: green if compressed, orange if marginal, red if expanded.
        for bar, ratio in zip(bars, avg_ratios):
            bar.set_color("green" if ratio > 1.0 else "orange" if ratio > 0.95 else "red")

        for bar, ratio in zip(bars, avg_ratios):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{ratio:.3f}x",
                ha="center", va="bottom", fontweight="bold",
            )

        ax.set_xlabel("Compression Format", fontweight="bold")
        ax.set_ylabel("Compression Ratio (higher is better)", fontweight="bold")
        ax.set_title("Compression Ratio Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No compression")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def create_speed_comparison(data: BenchmarkData) -> Figure:
        """Grouped bar chart: compression vs decompression speed (MB/s) per format."""
        fig, ax = plt.subplots(figsize=(12, 6))

        formats: Dict[str, Dict[str, List[float]]] = {}
        for r in data.results:
            if r["compression"]["success"]:
                fmt = r["format"]
                d   = formats.setdefault(fmt, {"comp": [], "decomp": []})
                d["comp"].append(r["compression"]["compression_speed_mbps"])
                d["decomp"].append(r["compression"]["decompression_speed_mbps"])

        names        = list(formats)
        comp_speeds  = [np.mean(formats[f]["comp"])   for f in names]
        decomp_speeds = [np.mean(formats[f]["decomp"]) for f in names]

        x     = np.arange(len(names))
        width = 0.35

        ax.bar(x - width / 2, comp_speeds,   width, label="Compression",   alpha=0.8, edgecolor="black")
        ax.bar(x + width / 2, decomp_speeds, width, label="Decompression",  alpha=0.8, edgecolor="black")

        ax.set_xlabel("Compression Format", fontweight="bold")
        ax.set_ylabel("Speed (MB/s)",        fontweight="bold")
        ax.set_title("Compression vs Decompression Speed", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        return fig

    @staticmethod
    def create_resource_usage_chart(data: BenchmarkData) -> Figure:
        """Three-panel bar chart: CPU, peak RAM, and total I/O per format."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

        formats: Dict[str, Dict[str, List[float]]] = {}
        for r in data.results:
            if r["compression"]["success"] and "system_metrics" in r:
                fmt = r["format"]
                sm  = r["system_metrics"]
                d   = formats.setdefault(fmt, {"cpu": [], "ram": [], "io": []})
                d["cpu"].append(sm["cpu"]["avg_process_percent"])
                d["ram"].append(sm["memory"]["peak_mb"])
                d["io"].append(sm["io"]["total_mb"])

        if not formats:
            for ax, msg in zip(
                (ax1, ax2, ax3),
                ("No system metrics available",
                 "Enable resource monitoring",
                 "in benchmark settings"),
            ):
                ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            return fig

        names = list(formats)
        x     = np.arange(len(names))

        for ax, key, ylabel, title, color in [
            (ax1, "cpu", "CPU Usage (%)",   "Average CPU Usage",  "skyblue"),
            (ax2, "ram", "Peak RAM (MB)",   "Peak RAM Usage",     "lightcoral"),
            (ax3, "io",  "Total I/O (MB)",  "Total I/O",          "lightgreen"),
        ]:
            values = [np.mean(formats[f][key]) for f in names]
            ax.bar(x, values, alpha=0.8, edgecolor="black", color=color)
            ax.set_xlabel("Format",  fontweight="bold")
            ax.set_ylabel(ylabel,    fontweight="bold")
            ax.set_title(title,      fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    @staticmethod
    def create_scatter_ratio_vs_speed(data: BenchmarkData) -> Figure:
        """Scatter plot: compression speed (x) vs ratio (y) per format."""
        fig, ax = plt.subplots(figsize=(10, 8))

        formats: Dict[str, Dict[str, List[float]]] = {}
        for r in data.results:
            if r["compression"]["success"]:
                fmt = r["format"]
                d   = formats.setdefault(fmt, {"ratio": [], "speed": []})
                d["ratio"].append(r["compression"]["compression_ratio"])
                d["speed"].append(r["compression"]["compression_speed_mbps"])

        colors = plt.cm.tab10(np.linspace(0, 1, len(formats)))

        for (fmt, values), color in zip(formats.items(), colors):
            ratios = values["ratio"]
            speeds = values["speed"]
            avg_r  = np.mean(ratios)
            avg_s  = np.mean(speeds)

            ax.scatter(speeds, ratios, alpha=0.4, s=50, color=color)
            ax.scatter(
                avg_s, avg_r, s=200, marker="*",
                edgecolor="black", linewidth=2, color=color,
                label=fmt, zorder=10,
            )

        ax.set_xlabel("Compression Speed (MB/s)",          fontweight="bold")
        ax.set_ylabel("Compression Ratio (higher is better)", fontweight="bold")
        ax.set_title("Compression Ratio vs Speed Tradeoff", fontsize=14, fontweight="bold")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No compression")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Annotate ideal / poor corners.
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.text(xlim[1] * 0.95, ylim[1] * 0.95, "Ideal\n(Fast & High Ratio)",
                ha="right", va="top",    fontsize=10, alpha=0.5, style="italic")
        ax.text(xlim[0] * 1.05, ylim[0] * 1.05, "Poor\n(Slow & Low Ratio)",
                ha="left",  va="bottom", fontsize=10, alpha=0.5, style="italic")

        plt.tight_layout()
        return fig

    @staticmethod
    def create_detailed_performance_heatmap(data: BenchmarkData) -> Figure:
        """Heatmap: compression ratio for each (format, image) combination."""
        fig, ax = plt.subplots(figsize=(12, 8))

        formats: List[str] = []
        images:  List[str] = []

        for r in data.results:
            if r["compression"]["success"]:
                if r["format"] not in formats:
                    formats.append(r["format"])
                if r["image"] not in images:
                    images.append(r["image"])

        matrix = np.zeros((len(formats), len(images)))
        for r in data.results:
            if r["compression"]["success"]:
                fi = formats.index(r["format"])
                ii = images.index(r["image"])
                matrix[fi, ii] = r["compression"]["compression_ratio"]

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.8, vmax=1.2)

        ax.set_xticks(np.arange(len(images)))
        ax.set_yticks(np.arange(len(formats)))
        ax.set_xticklabels(images,   rotation=45, ha="right")
        ax.set_yticklabels(formats)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Compression Ratio", fontweight="bold")

        for i in range(len(formats)):
            for j in range(len(images)):
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="black", fontsize=8, fontweight="bold",
                )

        ax.set_title("Compression Ratio Heatmap (Format × Image)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Images",  fontweight="bold")
        ax.set_ylabel("Formats", fontweight="bold")
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

class VisualizationExporter:
    """Exports matplotlib figures to PDF, PNG, SVG, and benchmark data to CSV."""

    @staticmethod
    def export_to_pdf(figures: List[Tuple[str, Figure]], output_path: Path) -> None:
        """Save all figures to a single multi-page PDF report."""
        with PdfPages(output_path) as pdf:
            for _title, fig in figures:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            info = pdf.infodict()
            info["Title"]        = "Benchmark Visualization Report"
            info["Author"]       = "Image Compression Benchmark Tool"
            info["Subject"]      = "Performance Analysis"
            info["Keywords"]     = "Compression, Benchmark, Performance"
            info["CreationDate"] = datetime.now()

    @staticmethod
    def export_to_png(figure: Figure, output_path: Path, dpi: int = 300) -> None:
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight",
                       facecolor="white", edgecolor="none")
        plt.close(figure)

    @staticmethod
    def export_to_svg(figure: Figure, output_path: Path) -> None:
        figure.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def export_to_csv_detail(data: BenchmarkData, output_path: Path) -> None:
        """Export all results to CSV — one row per image × format."""
        fieldnames = [
            "image", "format", "success",
            "compression_ratio", "space_saving_percent",
            "original_size_bytes", "compressed_size_bytes",
            "compression_time_s", "decompression_time_s",
            "compression_speed_mbps", "decompression_speed_mbps",
            "cpu_avg_percent", "ram_peak_mb", "io_total_mb",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

            for result in data.results:
                comp = result.get("compression", {})
                sm   = result.get("system_metrics", {})
                writer.writerow({
                    "image":                    result.get("image",  ""),
                    "format":                   result.get("format", ""),
                    "success":                  comp.get("success",                  False),
                    "compression_ratio":        comp.get("compression_ratio",        ""),
                    "space_saving_percent":     comp.get("space_saving_percent",     ""),
                    "original_size_bytes":      comp.get("original_size",            ""),
                    "compressed_size_bytes":    comp.get("compressed_size",          ""),
                    "compression_time_s":       comp.get("compression_time",         ""),
                    "decompression_time_s":     comp.get("decompression_time",       ""),
                    "compression_speed_mbps":   comp.get("compression_speed_mbps",  ""),
                    "decompression_speed_mbps": comp.get("decompression_speed_mbps",""),
                    "cpu_avg_percent": sm.get("cpu",    {}).get("avg_process_percent", ""),
                    "ram_peak_mb":     sm.get("memory", {}).get("peak_mb",              ""),
                    "io_total_mb":     sm.get("io",     {}).get("total_mb",            ""),
                })

    @staticmethod
    def export_to_csv_summary(data: BenchmarkData, output_path: Path) -> None:
        """Export per-format averages to CSV — one row per format (successful runs only)."""
        fieldnames = [
            "format", "num_images", "success_count",
            "avg_compression_ratio", "avg_space_saving_percent",
            "avg_original_size_bytes", "avg_compressed_size_bytes",
            "avg_compression_time_s", "avg_decompression_time_s",
            "avg_compression_speed_mbps", "avg_decompression_speed_mbps",
            "avg_cpu_avg_percent", "avg_ram_peak_mb", "avg_io_total_mb",
        ]

        # ---- Accumulate per-format values ----
        formats_data: Dict[str, Dict[str, List]] = {}
        for result in data.results:
            fmt  = result.get("format", "")
            comp = result.get("compression", {})
            sm   = result.get("system_metrics", {})
            if fmt not in formats_data:
                formats_data[fmt] = {k: [] for k in [
                    "success",
                    "compression_ratio", "space_saving_percent",
                    "original_size", "compressed_size",
                    "compression_time", "decompression_time",
                    "compression_speed_mbps", "decompression_speed_mbps",
                    "cpu_avg_percent", "ram_peak_mb", "io_total_mb",
                ]}
            fd = formats_data[fmt]
            fd["success"].append(comp.get("success", False))
            if comp.get("success"):
                def _append(key, val):
                    if val != "" and val is not None:
                        fd[key].append(val)
                _append("compression_ratio",        comp.get("compression_ratio"))
                _append("space_saving_percent",     comp.get("space_saving_percent"))
                _append("original_size",            comp.get("original_size"))
                _append("compressed_size",          comp.get("compressed_size"))
                _append("compression_time",         comp.get("compression_time"))
                _append("decompression_time",       comp.get("decompression_time"))
                _append("compression_speed_mbps",   comp.get("compression_speed_mbps"))
                _append("decompression_speed_mbps", comp.get("decompression_speed_mbps"))
                _append("cpu_avg_percent", sm.get("cpu",    {}).get("avg_process_percent"))
                _append("ram_peak_mb",     sm.get("memory", {}).get("peak_mb"))
                _append("io_total_mb",     sm.get("io",     {}).get("total_mb"))

        def _avg(lst: List) -> str:
            return f"{np.mean(lst):.4f}" if lst else ""

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

            for fmt, fd in formats_data.items():
                success_count = sum(1 for s in fd["success"] if s)
                writer.writerow({
                    "format":                       fmt,
                    "num_images":                   len(fd["success"]),
                    "success_count":                success_count,
                    "avg_compression_ratio":        _avg(fd["compression_ratio"]),
                    "avg_space_saving_percent":     _avg(fd["space_saving_percent"]),
                    "avg_original_size_bytes":      _avg(fd["original_size"]),
                    "avg_compressed_size_bytes":    _avg(fd["compressed_size"]),
                    "avg_compression_time_s":       _avg(fd["compression_time"]),
                    "avg_decompression_time_s":     _avg(fd["decompression_time"]),
                    "avg_compression_speed_mbps":   _avg(fd["compression_speed_mbps"]),
                    "avg_decompression_speed_mbps": _avg(fd["decompression_speed_mbps"]),
                    "avg_cpu_avg_percent":          _avg(fd["cpu_avg_percent"]),
                    "avg_ram_peak_mb":              _avg(fd["ram_peak_mb"]),
                    "avg_io_total_mb":              _avg(fd["io_total_mb"]),
                })


# ---------------------------------------------------------------------------
# Visualization window
# ---------------------------------------------------------------------------

class VisualizationWindow:
    """
    Tkinter Toplevel window for interactive chart selection and export.

    Can be opened with pre-loaded data (auto-show mode) or with the
    user manually selecting a JSON file via the Load button.
    """

    def __init__(self, parent: tk.Widget, data: Optional[BenchmarkData] = None):
        self.window          = tk.Toplevel(parent)
        self.window.title("Benchmark Visualization")
        self.window.geometry("1200x800")

        self.data:            Optional[BenchmarkData] = None
        self.current_figure:  Optional[Figure]        = None

        self._build_widgets()

        if data:
            self.load_data(data)

    def _build_widgets(self) -> None:
        # ---- Top control bar ----
        ctrl = ttk.Frame(self.window, padding="10")
        ctrl.pack(fill=tk.X)

        ttk.Button(ctrl, text="Load Benchmark JSON", command=self._load_json).pack(side=tk.LEFT, padx=5)
        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(ctrl, text="Export:").pack(side=tk.LEFT, padx=5)

        for text, cmd in [
            ("PDF (All Charts)",   self._export_pdf),
            ("PNG (Current)",      self._export_png),
            ("SVG (Current)",      self._export_svg),
            ("CSV (All Data)",     self._export_csv_detail),
            ("CSV (Summary)",      self._export_csv_summary),
        ]:
            ttk.Button(ctrl, text=text, command=cmd).pack(side=tk.LEFT, padx=2)

        # ---- Chart type selector ----
        select = ttk.LabelFrame(self.window, text="Select Chart Type", padding="10")
        select.pack(fill=tk.X, padx=10, pady=5)

        self.chart_buttons: List[ttk.Button] = []
        chart_types = [
            ("Compression Ratio Comparison", self.show_compression_ratio),
            ("Speed Comparison",             self.show_speed_comparison),
            ("Resource Usage (CPU/RAM/IO)",  self.show_resource_usage),
            ("Ratio vs Speed Tradeoff",      self.show_scatter_plot),
            ("Performance Heatmap",          self.show_heatmap),
        ]

        for text, command in chart_types:
            btn = ttk.Button(select, text=text, command=command, width=30)
            btn.pack(side=tk.LEFT, padx=5)
            # Disable chart buttons until data is loaded.
            btn.config(state=tk.DISABLED)
            self.chart_buttons.append(btn)

        # ---- Matplotlib canvas area ----
        self.canvas_frame = ttk.Frame(self.window)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._show_placeholder()

    def _show_placeholder(self) -> None:
        """Display a hint message before any chart is shown."""
        ttk.Label(
            self.canvas_frame,
            text="Load a benchmark JSON file or select a chart type to begin",
            font=("Arial", 14),
            justify=tk.CENTER,
        ).place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # ---- Data management ----

    def load_data(self, data: BenchmarkData) -> None:
        """Load benchmark data and enable the chart buttons."""
        self.data = data
        for btn in self.chart_buttons:
            btn.config(state=tk.NORMAL)

    def _load_json(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select Benchmark JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if filename:
            data = BenchmarkDataLoader.load_from_file(Path(filename))
            if data:
                self.load_data(data)
                messagebox.showinfo("Success", f"Loaded: {Path(filename).name}")
            else:
                messagebox.showerror("Error", "Failed to load JSON file.")

    # ---- Chart display ----

    def _clear_canvas(self) -> None:
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

    def _display_figure(self, fig: Figure) -> None:
        self._clear_canvas()
        self.current_figure = fig

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_compression_ratio(self) -> None:
        if self.data:
            self._display_figure(ChartGenerator.create_compression_ratio_comparison(self.data))

    def show_speed_comparison(self) -> None:
        if self.data:
            self._display_figure(ChartGenerator.create_speed_comparison(self.data))

    def show_resource_usage(self) -> None:
        if self.data:
            self._display_figure(ChartGenerator.create_resource_usage_chart(self.data))

    def show_scatter_plot(self) -> None:
        if self.data:
            self._display_figure(ChartGenerator.create_scatter_ratio_vs_speed(self.data))

    def show_heatmap(self) -> None:
        if self.data:
            self._display_figure(ChartGenerator.create_detailed_performance_heatmap(self.data))

    # ---- Export actions ----

    def _export_png(self) -> None:
        if not self.current_figure:
            messagebox.showwarning("No Chart", "Please display a chart first.")
            return
        filename = filedialog.asksaveasfilename(
            title="Save as PNG", defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
        )
        if filename:
            VisualizationExporter.export_to_png(self.current_figure, Path(filename))
            messagebox.showinfo("Success", f"Exported to: {Path(filename).name}")

    def _export_svg(self) -> None:
        if not self.current_figure:
            messagebox.showwarning("No Chart", "Please display a chart first.")
            return
        filename = filedialog.asksaveasfilename(
            title="Save as SVG", defaultextension=".svg",
            filetypes=[("SVG files", "*.svg")],
        )
        if filename:
            VisualizationExporter.export_to_svg(self.current_figure, Path(filename))
            messagebox.showinfo("Success", f"Exported to: {Path(filename).name}")

    def _export_pdf(self) -> None:
        if not self.data:
            messagebox.showwarning("No Data", "Please load benchmark data first.")
            return
        filename = filedialog.asksaveasfilename(
            title="Save as PDF", defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if filename:
            figures = [
                ("Compression Ratio",   ChartGenerator.create_compression_ratio_comparison(self.data)),
                ("Speed Comparison",    ChartGenerator.create_speed_comparison(self.data)),
                ("Resource Usage",      ChartGenerator.create_resource_usage_chart(self.data)),
                ("Ratio vs Speed",      ChartGenerator.create_scatter_ratio_vs_speed(self.data)),
                ("Performance Heatmap", ChartGenerator.create_detailed_performance_heatmap(self.data)),
            ]
            VisualizationExporter.export_to_pdf(figures, Path(filename))
            messagebox.showinfo("Success", f"Exported {len(figures)} charts to PDF.")

    def _export_csv_detail(self) -> None:
        if not self.data:
            messagebox.showwarning("No Data", "Please load benchmark data first.")
            return
        filename = filedialog.asksaveasfilename(
            title="Save All Data as CSV", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if filename:
            try:
                VisualizationExporter.export_to_csv_detail(self.data, Path(filename))
                messagebox.showinfo(
                    "Success",
                    f"Exported {len(self.data.results)} rows to: {Path(filename).name}",
                )
            except Exception as exc:
                messagebox.showerror("Export Error", str(exc))

    def _export_csv_summary(self) -> None:
        if not self.data:
            messagebox.showwarning("No Data", "Please load benchmark data first.")
            return
        filename = filedialog.asksaveasfilename(
            title="Save Summary as CSV", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if filename:
            try:
                VisualizationExporter.export_to_csv_summary(self.data, Path(filename))
                messagebox.showinfo("Success", f"Exported summary to: {Path(filename).name}")
            except Exception as exc:
                messagebox.showerror("Export Error", str(exc))


# ---------------------------------------------------------------------------
# Module-level factory function
# ---------------------------------------------------------------------------

def open_visualization_window(
    parent: tk.Widget,
    data: Optional[BenchmarkData] = None,
) -> VisualizationWindow:
    """
    Create and return a VisualizationWindow.

    The caller receives the window object so it can call methods like
    show_compression_ratio() for auto-show behaviour.
    """
    return VisualizationWindow(parent, data)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    open_visualization_window(root)
    root.mainloop()