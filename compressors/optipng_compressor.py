"""
PNG Lossless Compressor Plugin - optipng
compressors/png_compressor.py
"""

from pathlib import Path
from tkinter import Image
from typing import Optional
import subprocess
import time
import platform
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent))
from main import ImageCompressor, CompressionMetrics, CompressionLevel, CompressorFactory
from image_size_calculator import ImageSizeCalculator


class PNGCompressor(ImageCompressor):
    """
    PNG Lossless compressor
    
    Uses OptiPNG CLI tool from local libs/png folder
    """
    
    def __init__(self, lib_path: Optional[Path] = None):
        self.optipng_bin_path = None
        super().__init__(lib_path)
    
    def _validate_dependencies(self) -> None:
        """Find and validate OptiPNG binary"""
        # Find the libs/optipng directory
        base_dir = Path(__file__).parent.parent
        optipng_dir = base_dir / "libs" /"png"
        
        if not optipng_dir.exists():
            raise RuntimeError(f"Folder with OptiPNG was not found: {optipng_dir}")
        
        # Determine binary name based on platform
        system = platform.system().lower()
        if system == "windows":
            optipng_exe = "optipng.exe"
        else:
            optipng_exe = "optipng"
        
        optipng_path = optipng_dir / optipng_exe
        
        if not optipng_path.exists():
            raise RuntimeError(f"OptiPNG nebyl nalezen: {optipng_path}")
        
        # Store the path for later use
        self.optipng_bin_path = optipng_dir
        
        # On non-Windows, check if binary is executable
        if system != "windows":
            import os
            if not os.access(optipng_path, os.X_OK):
                raise RuntimeError(f"OptiPNG nemá práva ke spuštění: {optipng_path}")
    
    @property
    def name(self) -> str:
        return "OptiPNG"
    
    @property
    def extension(self) -> str:
        return ".png"
    
    def compress(self, 
                 input_path: Path, 
                 output_path: Path,
                 level: CompressionLevel = CompressionLevel.BALANCED) -> CompressionMetrics:
        """Compress image to PNG lossless format"""
        
        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)
            
            start_time = time.perf_counter()
            
            self._compress_cli(input_path, output_path, level)
            
            compression_time = time.perf_counter() - start_time
            
            compressed_size = output_path.stat().st_size
            
            # Test decompression to measure time (PNG decompression is just reading)
            temp_decomp = output_path.parent / f"temp_decomp_{output_path.stem}.png"
            decompression_time = self.decompress(output_path, temp_decomp)
            
            # Cleanup
            if temp_decomp.exists():
                temp_decomp.unlink()
            
            return CompressionMetrics(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=original_size / compressed_size,
                compression_time=compression_time,
                decompression_time=decompression_time,
                success=True
            )
            
        except Exception as e:
            return CompressionMetrics(
                original_size=0,
                compressed_size=0,
                compression_ratio=0,
                compression_time=0,
                decompression_time=0,
                success=False,
                error_message=str(e)
            )
    
    def _compress_cli(self, input_path: Path, output_path: Path, level: CompressionLevel):
        """Compress using OptiPNG CLI"""
        # Determine correct binary name based on platform
        system = platform.system().lower()
        optipng_exe = "optipng.exe" if system == "windows" else "optipng"
        optipng_path = self.optipng_bin_path / optipng_exe

        # Map compression levels to OptiPNG -o settings (0–7)
        # o0 = fastest, o7 = best compression (slowest)
        level_map = {
            CompressionLevel.FASTEST: 0,
            CompressionLevel.BALANCED: 4,
            CompressionLevel.BEST: 7,
        }

        o_level = level_map.get(level, 4)

        # OptiPNG modifies files in-place, so we need to copy first
        if input_path != output_path:
            shutil.copy2(input_path, output_path)

        cmd = [
            str(optipng_path),
            f"-o{o_level}",       # optimization level (0-7)
            "-strip", "all",      # strip metadata for smaller files
            "-quiet",             # suppress output
            str(output_path)      # file to optimize (in-place)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"OptiPNG failed: {result.stderr}")

        return result.stdout
    
    def decompress(self, input_path: Path, output_path: Path) -> float:
        """Decompress PNG by actually decoding it"""
        start_time = time.perf_counter()
        
        # Actually decode the PNG
        img = Image.open(input_path)
        img.load()  # Force loading pixels into memory
        
        # Save as uncompressed or re-encode
        img.save(output_path, format='PNG')
        
        return time.perf_counter() - start_time


# Automatic registration
CompressorFactory.register("optipng", PNGCompressor)