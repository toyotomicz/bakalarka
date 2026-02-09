"""
UNIVERSAL COMPRESSOR TEMPLATE - STANDARDIZED VERSION
Use this as base for all new compressors to ensure consistency

KEY FIXES:
1. Always use ImageSizeCalculator for original_size
2. Consistent RGBA handling (preserve alpha by default)
3. Proper metadata stripping
4. Standardized error handling
"""

from pathlib import Path
from typing import Optional
import time
import sys

from PIL import Image
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from main import ImageCompressor, CompressionMetrics, CompressionLevel, CompressorFactory
from image_size_calculator import ImageSizeCalculator


class StandardizedCompressor(ImageCompressor):
    """
    Template for standardized compressor implementation.
    All compressors should follow this pattern.
    """
    
    def __init__(self, lib_path: Optional[Path] = None):
        super().__init__(lib_path)
    
    def _validate_dependencies(self) -> None:
        """Validate that required libraries/tools are available"""
        # TODO: Implement dependency checking
        pass
    
    @property
    def name(self) -> str:
        return "StandardizedFormat"
    
    @property
    def extension(self) -> str:
        return ".std"
    
    def compress(self, 
                 input_path: Path, 
                 output_path: Path,
                 level: CompressionLevel = CompressionLevel.BALANCED) -> CompressionMetrics:
        """
        Compress image with standardized measurements.
        
        CRITICAL: Always use ImageSizeCalculator for fair comparison!
        """
        
        try:
            # ✅ STANDARD: Use ImageSizeCalculator for uncompressed size
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)
            
            # Load image
            img = Image.open(input_path)
            
            # ✅ STANDARD: Preserve RGBA by default, only convert problematic modes
            if img.mode not in ('RGB', 'RGBA', 'L', 'LA'):
                # Convert palette, CMYK, etc. to RGB
                img = img.convert('RGB')
            
            # Convert to numpy if needed
            image_data = np.array(img)
            
            # Start timing
            start_time = time.perf_counter()
            
            # TODO: Implement actual compression here
            # compressed_data = self._do_compression(image_data, level)
            
            # For template, just save with PIL
            img.save(output_path)
            
            compression_time = time.perf_counter() - start_time
            
            # ✅ STANDARD: Get compressed file size
            compressed_size = output_path.stat().st_size
            
            # ✅ STANDARD: Measure decompression time
            temp_decomp = output_path.parent / f"temp_decomp_{output_path.stem}.png"
            try:
                decompression_time = self.decompress(output_path, temp_decomp)
            finally:
                if temp_decomp.exists():
                    temp_decomp.unlink()
            
            # ✅ STANDARD: Return metrics
            return CompressionMetrics(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=original_size / compressed_size,
                compression_time=compression_time,
                decompression_time=decompression_time,
                success=True
            )
            
        except Exception as e:
            # ✅ STANDARD: Error handling
            return CompressionMetrics(
                original_size=0,
                compressed_size=0,
                compression_ratio=0,
                compression_time=0,
                decompression_time=0,
                success=False,
                error_message=str(e)
            )
    
    def decompress(self, input_path: Path, output_path: Path) -> float:
        """
        Decompress and measure time.
        
        ✅ STANDARD: Always save as PNG for verification
        """
        start_time = time.perf_counter()
        
        # TODO: Implement actual decompression
        img = Image.open(input_path)
        img.save(output_path, format='PNG')
        
        return time.perf_counter() - start_time


# Registration example
# CompressorFactory.register("standard", StandardizedCompressor)
