"""
Accurate calculation of true uncompressed image size (raw raster data)
for lossless compression benchmarking.

Definition:
Uncompressed size = raw pixel buffer only
(width × height × bits_per_pixel, rounded up to full bytes)

No headers, metadata, palettes, or container overhead included.
"""

from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageInfo:
    width: int
    height: int
    mode: str
    channels: int
    bits_per_channel: int
    bits_per_pixel: int
    uncompressed_size_bytes: int
    file_size_bytes: int

    @property
    def resolution(self) -> int:
        return self.width * self.height

    @property
    def megapixels(self) -> float:
        return self.resolution / 1_000_000


class ImageSizeCalculator:
    """
    Calculates true uncompressed image size as raw raster data.
    """

    # Explicit mapping of PIL modes to channel and bit depth semantics
    MODE_INFO = {
        # Binary / grayscale
        "1":      (1, 1),     # 1-bit bitmap
        "L":      (1, 8),     # 8-bit grayscale
        "LA":     (2, 8),     # grayscale + alpha

        # Palette (index only, palette excluded by definition)
        "P":      (1, 8),
        "PA":     (2, 8),

        # RGB family
        "RGB":    (3, 8),
        "RGBA":   (4, 8),
        "RGBX":   (4, 8),
        "RGBa":   (4, 8),

        # Other color spaces (treated as raw channels)
        "CMYK":   (4, 8),
        "YCbCr":  (3, 8),
        "LAB":    (3, 8),
        "HSV":    (3, 8),

        # Integer / float pixels
        "I":      (1, 32),    # signed int32
        "F":      (1, 32),    # float32

        # 16-bit grayscale
        "I;16":   (1, 16),
        "I;16L":  (1, 16),
        "I;16B":  (1, 16),
        "I;16N":  (1, 16),

        # Packed BGR formats (bits per pixel, not per channel!)
        "BGR;15": (1, 15),    # 5-5-5
        "BGR;16": (1, 16),    # 5-6-5
        "BGR;24": (3, 8),
        "BGR;32": (4, 8),
    }

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    @staticmethod
    def get_image_info(image_path: Path) -> Optional[ImageInfo]:
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode

                if mode in ImageSizeCalculator.MODE_INFO:
                    channels, bits_per_channel = ImageSizeCalculator.MODE_INFO[mode]
                else:
                    # Fallback: assume 8-bit channels
                    bands = img.getbands()
                    channels = len(bands)
                    bits_per_channel = 8

                # Special handling for packed formats
                if mode.startswith("BGR;") and bits_per_channel in (15, 16):
                    bits_per_pixel = bits_per_channel
                else:
                    bits_per_pixel = channels * bits_per_channel

                total_bits = width * height * bits_per_pixel
                uncompressed_size = ImageSizeCalculator._ceil_div(total_bits, 8)

                file_size = image_path.stat().st_size

                return ImageInfo(
                    width=width,
                    height=height,
                    mode=mode,
                    channels=channels,
                    bits_per_channel=bits_per_channel,
                    bits_per_pixel=bits_per_pixel,
                    uncompressed_size_bytes=uncompressed_size,
                    file_size_bytes=file_size,
                )

        except Exception as e:
            print(f"Error analyzing image {image_path}: {e}")
            return None

    @staticmethod
    def calculate_uncompressed_size(image_path: Path) -> int:
        info = ImageSizeCalculator.get_image_info(image_path)
        return info.uncompressed_size_bytes if info else image_path.stat().st_size

    @staticmethod
    def get_compression_baseline(image_path: Path) -> dict:
        info = ImageSizeCalculator.get_image_info(image_path)

        if not info or info.file_size_bytes == 0:
            return {
                "uncompressed_size": 0,
                "file_size": 0,
                "baseline_ratio": 0,
                "format": "unknown",
            }

        ratio = info.uncompressed_size_bytes / info.file_size_bytes

        return {
            "uncompressed_size": info.uncompressed_size_bytes,
            "file_size": info.file_size_bytes,
            "baseline_ratio": ratio,
            "format": image_path.suffix.lower(),
            "width": info.width,
            "height": info.height,
            "mode": info.mode,
            "channels": info.channels,
            "bits_per_pixel": info.bits_per_pixel,
            "megapixels": info.megapixels,
        }


def print_image_analysis(image_path: Path):
    info = ImageSizeCalculator.get_image_info(image_path)

    if not info:
        print(f"Could not analyze: {image_path}")
        return

    ratio = (
        info.uncompressed_size_bytes / info.file_size_bytes
        if info.file_size_bytes > 0
        else 0
    )

    print("\n" + "=" * 72)
    print(f"Image Analysis: {image_path.name}")
    print("=" * 72)
    print(f"Resolution:            {info.width} × {info.height} ({info.megapixels:.2f} MP)")
    print(f"Mode:                  {info.mode}")
    print(f"Channels:              {info.channels}")
    print(f"Bits per channel:      {info.bits_per_channel}")
    print(f"Bits per pixel:        {info.bits_per_pixel}")
    print(
        f"Uncompressed size:     {info.uncompressed_size_bytes:,} bytes "
        f"({info.uncompressed_size_bytes / (1024**2):.2f} MB)"
    )
    print(
        f"File size:             {info.file_size_bytes:,} bytes "
        f"({info.file_size_bytes / (1024**2):.2f} MB)"
    )
    print(f"Compression ratio:     {ratio:.2f}×")
    print(
        f"Space saving:          {(1 - info.file_size_bytes / info.uncompressed_size_bytes) * 100:.1f}%"
        if info.uncompressed_size_bytes > 0
        else "Space saving:          n/a"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for path_str in sys.argv[1:]:
            path = Path(path_str)
            if path.exists():
                print_image_analysis(path)
            else:
                print(f"File not found: {path}")
    else:
        print("Image Size Calculator")
        print("=" * 72)
        print("Calculates true uncompressed image size (raw raster data).")
        print("Usage: python image_size_calculator.py <image_file> [...]")
