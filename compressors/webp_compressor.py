"""
WebP Lossless Compressor Plugin
compressors/webp_compressor.py
"""

from pathlib import Path
from typing import Optional
import subprocess
import time
import platform

import sys
sys.path.append(str(Path(__file__).parent.parent))
from main import ImageCompressor, CompressionMetrics, CompressionLevel, CompressorFactory


class WebPCompressor(ImageCompressor):
    """
    WebP Lossless compressor
    
    Uses cwebp/dwebp CLI tools from local libs/webp folder
    """
    
    def __init__(self, lib_path: Optional[Path] = None):
        self.webp_bin_path = None
        super().__init__(lib_path)
    
    def _validate_dependencies(self) -> None:
        """Find and validate cwebp/dwebp binaries"""
        # Find the libs/webp directory
        base_dir = Path(__file__).parent.parent
        webp_dir = base_dir / "libs" / "webp"
        
        if not webp_dir.exists():
            raise RuntimeError(f"File with WebP tools found: {webp_dir}")
        
        # Determine binary names based on platform
        system = platform.system().lower()
        if system == "windows":
            cwebp_exe = "cwebp.exe"
            dwebp_exe = "dwebp.exe"
        else:
            cwebp_exe = "cwebp"
            dwebp_exe = "dwebp"
        
        cwebp_path = webp_dir / cwebp_exe
        dwebp_path = webp_dir / dwebp_exe
        
        if not cwebp_path.exists():
            raise RuntimeError(f"cwebp nebyl nalezen: {cwebp_path}")
        if not dwebp_path.exists():
            raise RuntimeError(f"dwebp nebyl nalezen: {dwebp_path}")
        
        # Store the path for later use
        self.webp_bin_path = webp_dir
        
        # On non-Windows, check if binaries are executable
        if system != "windows":
            import os
            if not os.access(cwebp_path, os.X_OK):
                raise RuntimeError(f"cwebp does not have access: {cwebp_path}")
            if not os.access(dwebp_path, os.X_OK):
                raise RuntimeError(f"dwebp does not have access: {dwebp_path}")
    
    @property
    def name(self) -> str:
        return "WebP-Lossless"
    
    @property
    def extension(self) -> str:
        return ".webp"
    
    def compress(self, 
                 input_path: Path, 
                 output_path: Path,
                 level: CompressionLevel = CompressionLevel.BALANCED) -> CompressionMetrics:
        """Compress image to WebP lossless format"""
        
        try:
            original_size = input_path.stat().st_size
            
            start_time = time.perf_counter()
            
            self._compress_cli(input_path, output_path, level)
            
            compression_time = time.perf_counter() - start_time
            
            compressed_size = output_path.stat().st_size
            
            # Test decompression to measure time
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
        """Compress using cwebp CLI"""
        # Determine correct binary name based on platform
        system = platform.system().lower()
        cwebp_exe = "cwebp.exe" if system == "windows" else "cwebp"
        cwebp_path = self.webp_bin_path / cwebp_exe

        # Map compression levels to WebP -z settings (0–9)
        level_map = {
            CompressionLevel.FASTEST: 0,
            CompressionLevel.BALANCED: 6,
            CompressionLevel.BEST: 9,
        }

        z_level = level_map.get(level, 6)

        # In lossless mode, -q controls the tradeoff between speed and size.
        # -q 100 = maximum compression, slower, smaller output.
        q_value = 100 if level in [CompressionLevel.BEST, CompressionLevel.GOOD] else 75

        # -m controls time vs. quality (0–6)
        # higher method = slower, but better compression
        m_value = 6 if level in [CompressionLevel.GOOD, CompressionLevel.BEST] else 4

        cmd = [
            str(cwebp_path),
            "-lossless",          # lossless compression mode
            "-mt",                # multi-threading
            "-exact",             # preserve RGB values in transparent area
            "-q", str(q_value),   # quality for lossless (0-100)
            "-z", str(z_level),   # level of compression (0-9)
            "-m", str(m_value),   # method (0-6)
            "-alpha_q", "100",    # lossless alpha channel
            str(input_path),
            "-o", str(output_path)# output file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"cwebp failed: {result.stderr}")

        return result.stdout

    
    def decompress(self, input_path: Path, output_path: Path) -> float:
        """Decompression using dwebp"""
        start_time = time.perf_counter()
        
        # Determine correct binary name based on platform
        system = platform.system().lower()
        dwebp_exe = "dwebp.exe" if system == "windows" else "dwebp"
        dwebp_path = self.webp_bin_path / dwebp_exe
        
        cmd = [str(dwebp_path), str(input_path), "-o", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"dwebp failed: {result.stderr}")
        
        return time.perf_counter() - start_time


# Automatic registration
CompressorFactory.register("webp", WebPCompressor)