"""
GUI Widget Components
Reusable UI components for the benchmark application
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Callable


class ImageSelectionWidget:
    """Widget for selecting images to benchmark"""
    
    def __init__(self, parent, on_update: Callable = None):
        self.parent = parent
        self.on_update = on_update
        self.selected_images = []
        
        self.frame = ttk.LabelFrame(parent, text="Images", padding="10")
        self.create_widgets()
    
    def create_widgets(self):
        """Create image selection UI components"""
        # Button frame
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.add_images_btn = ttk.Button(
            btn_frame,
            text="Add Images..."
        )
        self.add_images_btn.pack(side=tk.LEFT, padx=5)
        
        self.add_folder_btn = ttk.Button(
            btn_frame,
            text="Add Folder..."
        )
        self.add_folder_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(
            btn_frame,
            text="Clear All"
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Listbox
        self.listbox = tk.Listbox(self.frame, height=6, cursor="hand2")
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind('<Double-Button-1>', self._on_image_double_click)
        
        # Count label
        self.count_label = ttk.Label(self.frame, text="0 images selected")
        self.count_label.pack(anchor=tk.W, pady=(5, 0))
    
    def add_image(self, path):
        """Add image to selection"""
        if path not in self.selected_images:
            self.selected_images.append(path)
            self.listbox.insert(tk.END, path.name)
            self.update_count()
    
    def clear_images(self):
        """Clear all selected images"""
        self.selected_images = []
        self.listbox.delete(0, tk.END)
        self.update_count()
    
    def update_count(self):
        """Update image count label"""
        count = len(self.selected_images)
        text = f"{count} image{'s' if count != 1 else ''} selected"
        self.count_label.config(text=text)
        
        if self.on_update:
            self.on_update()
    
    def _on_image_double_click(self, event):
        """Handle double-click on image to preview"""
        selection = self.listbox.curselection()
        if selection:
            index = selection[0]
            image_path = self.selected_images[index]
            self._show_image_preview(image_path)
    
    def _show_image_preview(self, image_path):
        """Show image preview in new window"""
        try:
            from PIL import Image, ImageTk
            
            preview_window = tk.Toplevel(self.parent)
            preview_window.title(f"Preview: {image_path.name}")
            preview_window.geometry("800x600")
            
            # Load and resize image
            img = Image.open(image_path)
            
            # Calculate scaling to fit window
            max_size = (750, 550)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Create label with image
            label = ttk.Label(preview_window, image=photo)
            label.image = photo  # Keep reference
            label.pack(padx=10, pady=10)
            
            # Image info
            info_text = f"File: {image_path.name}\nSize: {image_path.stat().st_size:,} bytes\nDimensions: {img.size[0]} x {img.size[1]}"
            info_label = ttk.Label(preview_window, text=info_text, font=("Arial", 9))
            info_label.pack(pady=5)
            
            # Close button
            ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=5)
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Preview Error", f"Cannot preview image:\n{str(e)}")
    
    def pack(self, **kwargs):
        """Pack the frame"""
        self.frame.pack(**kwargs)


class CompressorSelectionWidget:
    """Widget for selecting compressors"""
    
    def __init__(self, parent, compressors: List[str]):
        self.parent = parent
        self.compressors = compressors
        self.compressor_vars: Dict[str, tk.BooleanVar] = {}
        
        self.frame = ttk.LabelFrame(parent, text="Compressors", padding="10")
        self.create_widgets()
    
    def create_widgets(self):
        """Create compressor selection UI"""
        if not self.compressors:
            ttk.Label(
                self.frame,
                text="No compressors found. Check plugins folder.",
                foreground="red"
            ).pack()
            return
        
        # Create checkboxes in grid
        cols = 3
        rows = (len(self.compressors) + cols - 1) // cols
        
        for idx, comp_name in enumerate(sorted(self.compressors)):
            row = idx % rows
            col = idx // rows
            
            var = tk.BooleanVar(value=True)
            self.compressor_vars[comp_name] = var
            
            cb = ttk.Checkbutton(self.frame, text=comp_name, variable=var)
            cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
        
        # Select/Deselect buttons
        btn_frame = ttk.Frame(self.frame)
        btn_frame.grid(row=rows+1, column=0, columnspan=cols, pady=(10, 0))
        
        ttk.Button(
            btn_frame,
            text="Select All",
            command=self.select_all
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Deselect All",
            command=self.deselect_all
        ).pack(side=tk.LEFT, padx=5)
    
    def get_selected(self) -> List[str]:
        """Get list of selected compressors"""
        return [
            name for name, var in self.compressor_vars.items()
            if var.get()
        ]
    
    def select_all(self):
        """Select all compressors"""
        for var in self.compressor_vars.values():
            var.set(True)
    
    def deselect_all(self):
        """Deselect all compressors"""
        for var in self.compressor_vars.values():
            var.set(False)
    
    def pack(self, **kwargs):
        """Pack the frame"""
        self.frame.pack(**kwargs)


class LevelSelectionWidget:
    """Widget for selecting compression levels"""
    
    def __init__(self, parent, levels: List[tuple]):
        self.parent = parent
        self.levels = levels
        self.level_vars: Dict = {}
        
        self.frame = ttk.LabelFrame(
            parent,
            text="Compression Levels",
            padding="10"
        )
        self.create_widgets()
    
    def create_widgets(self):
        """Create level selection UI"""
        for idx, (name, level) in enumerate(self.levels):
            var = tk.BooleanVar(value=False)
            self.level_vars[level] = var
            
            ttk.Checkbutton(
                self.frame,
                text=name,
                variable=var
            ).grid(row=0, column=idx, padx=10)
        
        # Options frame
        options_frame = ttk.Frame(self.frame)
        options_frame.grid(row=1, column=0, columnspan=len(self.levels), pady=(10, 0))
        
        # Verification checkbox
        self.verify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Verify lossless compression (pixel-by-pixel comparison)",
            variable=self.verify_var
        ).pack(anchor=tk.W)
        
        # Strip metadata checkbox
        self.strip_metadata_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Strip metadata (EXIF, XMP, etc.) before compression",
            variable=self.strip_metadata_var
        ).pack(anchor=tk.W)
    
    def get_selected(self) -> List:
        """Get list of selected levels"""
        return [level for level, var in self.level_vars.items() if var.get()]
    
    def is_verification_enabled(self) -> bool:
        """Check if verification is enabled"""
        return self.verify_var.get()
    
    def is_strip_metadata_enabled(self) -> bool:
        """Check if metadata stripping is enabled"""
        return self.strip_metadata_var.get()
    
    def pack(self, **kwargs):
        """Pack the frame"""
        self.frame.pack(**kwargs)


class VerificationResultsWidget:
    """Widget for displaying verification results"""
    
    def __init__(self, parent, verification_results: Dict):
        self.window = tk.Toplevel(parent)
        self.window.title("Verification Results Summary")
        self.window.geometry("1000x600")
        
        self.verification_results = verification_results
        self.create_widgets()
    
    def create_widgets(self):
        """Create results table"""
        # Frame for treeview
        tree_frame = ttk.Frame(self.window, padding="10")
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Define columns
        columns = (
            "Image",
            "Compressor",
            "Result",
            "Max Difference",
            "Different Pixels",
            "Accuracy %"
        )
        
        # Create treeview
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        
        # Configure columns
        column_widths = {
            "Image": 200,
            "Compressor": 150,
            "Result": 100,
            "Max Difference": 120,
            "Different Pixels": 150,
            "Accuracy %": 120
        }
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths.get(col, 100))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(
            tree_frame,
            orient=tk.VERTICAL,
            command=self.tree.yview
        )
        self.tree.configure(yscroll=scrollbar.set)
        
        # Pack
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate data
        self.populate_results()
        
        # Summary
        self.create_summary()
        
        # Close button
        ttk.Button(
            self.window,
            text="Close",
            command=self.window.destroy
        ).pack(pady=10)
    
    def populate_results(self):
        """Populate tree with verification results"""
        for key, verification in self.verification_results.items():
            image_name, comp_name = key
            
            result = "LOSSLESS" if verification.is_lossless else "LOSSY"
            
            # Add color tags
            tag = "lossless" if verification.is_lossless else "lossy"
            
            self.tree.insert("", tk.END, values=(
                image_name,
                comp_name,
                result,
                f"{verification.max_difference:.2f}",
                f"{verification.different_pixels:,}",
                f"{verification.accuracy_percent:.4f}"
            ), tags=(tag,))
        
        # Configure tags
        self.tree.tag_configure("lossless", foreground="dark green")
        self.tree.tag_configure("lossy", foreground="dark red")
    
    def create_summary(self):
        """Create summary statistics"""
        if not self.verification_results:
            return
        
        lossless_count = sum(
            1 for v in self.verification_results.values()
            if v.is_lossless
        )
        total_count = len(self.verification_results)
        lossy_count = total_count - lossless_count
        
        summary_text = (
            f"Total Tests: {total_count}  |  "
            f"Lossless: {lossless_count} ({100*lossless_count/total_count:.1f}%)  |  "
            f"Lossy: {lossy_count} ({100*lossy_count/total_count:.1f}%)"
        )
        
        summary_label = ttk.Label(
            self.window,
            text=summary_text,
            font=("Arial", 10, "bold")
        )
        summary_label.pack(pady=5)