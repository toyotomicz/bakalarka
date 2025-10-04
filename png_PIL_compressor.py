"""
PNG Compressor Plugin using PIL/Pillow
Supports lossless PNG compression with various compression levels

"""

from pathlib import Path
from PIL import Image
import time
import sys

# Ensure parent directory is in sys.path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from main - this MUST work because PluginLoader adds parent to path
import main
ImageCompressor = main.ImageCompressor
CompressionMetrics = main.CompressionMetrics
CompressionLevel = main.CompressionLevel
CompressorFactory = main.CompressorFactory


class PNGCompressor(ImageCompressor):
    """PNG compressor using Pillow (PIL) library"""
    
    def _validate_dependencies(self) -> None:
        """Check if PIL/Pillow is available"""
        try:
            import PIL
            # Verify PNG support
            img = Image.new('RGB', (1, 1))
            # Test will pass if Pillow is properly installed
        except ImportError:
            raise RuntimeError(
                "Pillow is not installed. Install it with: pip install Pillow"
            )
        except Exception as e:
            raise RuntimeError(f"PIL/Pillow error: {e}")
    
    def compress(self, 
                input_path: Path, 
                output_path: Path,
                level: CompressionLevel = CompressionLevel.BALANCED) -> CompressionMetrics:
        """
        Compress image to PNG format
        
        PNG compression levels:
        - FASTEST (1): compress_level=1
        - FAST (3): compress_level=3
        - BALANCED (5): compress_level=6
        - GOOD (7): compress_level=8
        - BEST (9): compress_level=9
        """
        try:
            # Map compression level to PNG compress_level (0-9)
            compress_map = {
                CompressionLevel.FASTEST: 1,
                CompressionLevel.FAST: 3,
                CompressionLevel.BALANCED: 6,
                CompressionLevel.GOOD: 8,
                CompressionLevel.BEST: 9
            }
            png_level = compress_map[level]
            
            # Get original file size
            original_size = input_path.stat().st_size
            
            # Load image
            img = Image.open(input_path)
            
            # Convert to RGB if needed (for compatibility)
            if img.mode not in ('RGB', 'RGBA', 'L', 'LA'):
                if img.mode == 'P':
                    img = img.convert('RGBA' if 'transparency' in img.info else 'RGB')
                else:
                    img = img.convert('RGB')
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Compress and measure time
            start_time = time.perf_counter()
            img.save(
                output_path, 
                'PNG',
                compress_level=png_level,
                optimize=True
            )
            compression_time = time.perf_counter() - start_time
            
            # Get compressed file size
            compressed_size = output_path.stat().st_size
            
            # Measure decompression time
            decompression_time = self.decompress(output_path, None)
            
            # Calculate metrics
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            return CompressionMetrics(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
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
    
    def decompress(self, input_path: Path, output_path: Path = None) -> float:
        """
        Decompress PNG image (or just measure loading time)
        
        Args:
            input_path: Path to PNG file
            output_path: If provided, saves decompressed image. If None, just measures time.
        
        Returns:
            Decompression time in seconds
        """
        start_time = time.perf_counter()
        
        img = Image.open(input_path)
        # Force loading of image data
        img.load()
        
        if output_path:
            # Save as uncompressed format for testing
            img.save(output_path, 'BMP')
        
        return time.perf_counter() - start_time
    
    @property
    def name(self) -> str:
        return "PNG"
    
    @property
    def extension(self) -> str:
        return ".png"


# CRITICAL: Register this compressor with the factory
# This line is executed when the module is loaded
# Only register if not already registered (prevents double registration)
if "png" not in CompressorFactory._compressors:
    CompressorFactory.register("png", PNGCompressor)
    print(f"  → PNG compressor registered successfully")