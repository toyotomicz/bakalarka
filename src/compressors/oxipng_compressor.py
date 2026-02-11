"""
PNG Lossless Compressor Plugin - oxipng
compressors/png_compressor.py
"""

from pathlib import Path
from PIL import Image
from typing import Optional
import subprocess
import time
import platform

import sys
sys.path.append(str(Path(__file__).parent.parent))
from main import ImageCompressor, CompressionMetrics, CompressionLevel, CompressorFactory
from image_size_calculator import ImageSizeCalculator


class OxiPNGCompressor(ImageCompressor):
    """
    PNG Lossless compressor (Multi-threaded)
    
    Uses OxiPNG CLI tool from local libs/png folder
    """
    
    def __init__(self, lib_path: Optional[Path] = None):
        self.oxipng_bin_path = None
        super().__init__(lib_path)
    
    def _validate_dependencies(self) -> None:
        """Find and validate OxiPNG binary"""
        # Hledá se ve stejné složce jako předtím
        base_dir = Path(__file__).parent.parent
        oxipng_dir = base_dir / "libs" / "oxipng"
        
        if not oxipng_dir.exists():
            raise RuntimeError(f"Složka s nástroji pro PNG nebyla nalezena: {oxipng_dir}")
        
        # Determine binary name based on platform
        system = platform.system().lower()
        if system == "windows":
            oxipng_exe = "oxipng.exe"
        else:
            oxipng_exe = "oxipng"
        
        oxipng_path = oxipng_dir / oxipng_exe
        
        if not oxipng_path.exists():
            raise RuntimeError(f"OxiPNG nebyl nalezen: {oxipng_path}")
        
        # Store the path for later use
        self.oxipng_bin_path = oxipng_dir
        
        # On non-Windows, check if binary is executable
        if system != "windows":
            import os
            if not os.access(oxipng_path, os.X_OK):
                raise RuntimeError(f"OxiPNG nemá práva ke spuštění: {oxipng_path}")
    
    @property
    def name(self) -> str:
        return "OxiPNG"
    
    @property
    def extension(self) -> str:
        return ".png"
    
    def compress(self, 
                 input_path: Path, 
                 output_path: Path,
                 level: CompressionLevel = CompressionLevel.BALANCED) -> CompressionMetrics:
        """Compress image to PNG lossless format using OxiPNG"""
        
        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)
            
            start_time = time.perf_counter()
            
            self._compress_cli(input_path, output_path, level)
            
            compression_time = time.perf_counter() - start_time
            
            compressed_size = output_path.stat().st_size
            
            # Test decompression to measure time (PNG decompression is just reading)
            temp_decomp = output_path.parent / f"temp_decomp_{output_path.stem}.png"
            try:
                decompression_time = self.decompress(output_path, temp_decomp)
            finally:
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
        """Compress using OxiPNG CLI"""
        # Determine correct binary name based on platform
        system = platform.system().lower()
        oxipng_exe = "oxipng.exe" if system == "windows" else "oxipng"
        oxipng_path = self.oxipng_bin_path / oxipng_exe

        # Mapování úrovní komprese pro OxiPNG (1-6)
        # výchozí je 2, nejlepší 6.
        level_map = {
            CompressionLevel.FASTEST: 1,
            CompressionLevel.BALANCED: 3,
            CompressionLevel.BEST: 6,
        }

        o_level = level_map.get(level, 3)

        # Sestavení příkazu. OxiPNG má parametr --out, 
        # takže nepotřebujeme kopírovat soubor pomocí shutil předem.
        cmd = [
            str(oxipng_path),
            "-o", str(o_level),   # optimization level (1-6)
            "--strip", "all",     # strip metadata for smaller files
            "-q",                 # suppress output (quiet)
            "--out", str(output_path), # explicit output path
            str(input_path)       # input file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"OxiPNG failed: {result.stderr}")

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
CompressorFactory.register("oxipng", OxiPNGCompressor)