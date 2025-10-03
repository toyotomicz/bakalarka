from pathlib import Path
import subprocess
import time
from typing import Optional
import shutil

# Import ze základního modulu
import sys
sys.path.append(str(Path(__file__).parent.parent))
from main import ImageCompressor, CompressionMetrics, CompressionLevel, CompressorFactory


class PNGCompressor(ImageCompressor):
    """
    PNG kompresor používající libpng přes CLI nástroje.
    
    Použití:
    - pngcrush pro kompresi
    - Alternativně: optipng, pngquant (ale to je lossy)
    """
    
    def __init__(self, lib_path: Optional[Path] = None):
        # Můžeš použít lib_path pokud chceš přímo linkovat libpng
        # Pro jednoduchost používáme CLI nástroje
        self.cli_tool = None
        super().__init__(lib_path)
    
    def _validate_dependencies(self) -> None:
        """Zkontroluje dostupnost pngcrush nebo optipng"""
        # Zkus najít pngcrush
        if shutil.which("pngcrush"):
            self.cli_tool = "pngcrush"
            return
        
        # Zkus optipng
        if shutil.which("optipng"):
            self.cli_tool = "optipng"
            return
        
        # Pokud nic nenajde, zkus použít Pillow jako fallback
        try:
            from PIL import Image
            self.cli_tool = "pillow"
            return
        except ImportError:
            pass
        
        raise RuntimeError(
            "PNG kompresor vyžaduje: pngcrush, optipng, nebo Pillow\n"
            "Instalace: sudo apt-get install pngcrush optipng\n"
            "nebo: pip install Pillow"
        )
    
    @property
    def name(self) -> str:
        return "PNG"
    
    @property
    def extension(self) -> str:
        return ".png"
    
    def compress(self, 
                 input_path: Path, 
                 output_path: Path,
                 level: CompressionLevel = CompressionLevel.BALANCED) -> CompressionMetrics:
        """Zkomprimuje obrázek do PNG"""
        
        try:
            original_size = input_path.stat().st_size
            
            # Měření času komprese
            start_time = time.perf_counter()
            
            if self.cli_tool == "pngcrush":
                self._compress_pngcrush(input_path, output_path, level)
            elif self.cli_tool == "optipng":
                self._compress_optipng(input_path, output_path, level)
            elif self.cli_tool == "pillow":
                self._compress_pillow(input_path, output_path, level)
            
            compression_time = time.perf_counter() - start_time
            
            # Změř velikost
            compressed_size = output_path.stat().st_size
            
            # Dekomprese (v tomto případě jen načtení)
            decompression_time = self.decompress(output_path, output_path.parent / "temp_decompressed.png")
            
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
    
    def _compress_pngcrush(self, input_path: Path, output_path: Path, level: CompressionLevel):
        """Komprese pomocí pngcrush"""
        # pngcrush nemá explicitní level, ale můžeme použít -brute pro nejlepší kompresi
        cmd = ["pngcrush"]
        
        if level == CompressionLevel.BEST:
            cmd.append("-brute")  # Velmi pomalé, ale nejlepší výsledek
        
        cmd.extend([str(input_path), str(output_path)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"pngcrush selhalo: {result.stderr}")
    
    def _compress_optipng(self, input_path: Path, output_path: Path, level: CompressionLevel):
        """Komprese pomocí optipng"""
        # optipng -o level (0-7)
        level_map = {
            CompressionLevel.FASTEST: 0,
            CompressionLevel.FAST: 2,
            CompressionLevel.BALANCED: 4,
            CompressionLevel.GOOD: 6,
            CompressionLevel.BEST: 7
        }
        
        opt_level = level_map.get(level, 4)
        
        # optipng modifikuje soubor in-place, tak nejdřív zkopírujeme
        shutil.copy2(input_path, output_path)
        
        cmd = ["optipng", f"-o{opt_level}", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"optipng selhalo: {result.stderr}")
    
    def _compress_pillow(self, input_path: Path, output_path: Path, level: CompressionLevel):
        """Komprese pomocí Pillow (fallback)"""
        from PIL import Image
        
        level_map = {
            CompressionLevel.FASTEST: 1,
            CompressionLevel.FAST: 3,
            CompressionLevel.BALANCED: 6,
            CompressionLevel.GOOD: 8,
            CompressionLevel.BEST: 9
        }
        
        compress_level = level_map.get(level, 6)
        
        img = Image.open(input_path)
        img.save(output_path, "PNG", compress_level=compress_level, optimize=True)
    
    def decompress(self, input_path: Path, output_path: Path) -> float:
        """
        Pro PNG není dekomprese potřeba (je přímo čitelný).
        Simulujeme načtení.
        """
        from PIL import Image
        
        start_time = time.perf_counter()
        img = Image.open(input_path)
        img.load()  # Force load do paměti
        decompression_time = time.perf_counter() - start_time
        
        return decompression_time


# Automatická registrace pluginu při importu
CompressorFactory.register("png", PNGCompressor)
