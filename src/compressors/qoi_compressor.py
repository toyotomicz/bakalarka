"""
QOI (Quite OK Image) Lossless Compressor Plugin
compressors/qoi_compressor.py

QOI is a fast, lossless image compression format.
Requires: pip install qoi
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


class QOICompressor(ImageCompressor):
    """
    QOI (Quite OK Image) lossless compressor
    
    QOI is a simple, fast lossless image format.
    It doesn't have compression levels - it's always the same algorithm.
    """
    
    def __init__(self, lib_path: Optional[Path] = None):
        self.qoi = None
        super().__init__(lib_path)
    
    def _validate_dependencies(self) -> None:
        """Check if qoi library is available"""
        try:
            import qoi as qoi_module
            self.qoi = qoi_module
        except ImportError:
            raise RuntimeError(
                "qoi library is not installed. "
                "Install it with: pip install qoi"
            )
    
    @property
    def name(self) -> str:
        return "QOI"
    
    @property
    def extension(self) -> str:
        return ".qoi"
    
    def compress(self, 
                 input_path: Path, 
                 output_path: Path,
                 level: CompressionLevel = CompressionLevel.BALANCED) -> CompressionMetrics:
        """Compress image to QOI format"""
        
        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)
            
            # Load image
            img = Image.open(input_path)
            
            # Convert to RGB or RGBA
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Convert to numpy array
            image_data = np.array(img)
            
            start_time = time.perf_counter()
            
            # Encode to QOI
            # QOI doesn't have compression levels - it's always the same
            qoi_data = self.qoi.encode(image_data)
            
            # Write to file
            with open(output_path, 'wb') as f:
                f.write(qoi_data)
            
            compression_time = time.perf_counter() - start_time
            compressed_size = output_path.stat().st_size
            
            # Test decompression
            temp_decomp = output_path.parent / f"temp_decomp_{output_path.stem}.png"
            try:
                decompression_time = self.decompress(output_path, temp_decomp)
            finally:
                if temp_decomp.exists():
                    temp_decomp.unlink()
            
            return CompressionMetrics(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=original_size / compressed_size if compressed_size > 0 else 0,
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
    
    def decompress(self, input_path: Path, output_path: Path) -> float:
        """Decompress QOI file"""
        start_time = time.perf_counter()
        
        # Read QOI file
        with open(input_path, 'rb') as f:
            qoi_data = f.read()
        
        # Decode
        image_array = self.qoi.decode(qoi_data)
        
        # Convert to PIL Image and save
        img = Image.fromarray(image_array)
        img.save(output_path, format='PNG')
        
        return time.perf_counter() - start_time


# Register QOI compressor
CompressorFactory.register("qoi", QOICompressor)