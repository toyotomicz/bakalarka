"""
Accurate calculation of true uncompressed image size (raw raster data)

Used for lossless compression benchmarking to provide a consistent,
format independent baseline for all compressors.

Definition:
    Uncompressed size = raw pixel buffer only
    (width x height x bits_per_pixel, rounded up to whole bytes).

    No headers, metadata, palettes, or container overhead are included.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image


@dataclass
class ImageInfo:
    """
    Geometry and colour-depth information for a single image file.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        mode: Pillow colour mode string (e.g. 'RGB', 'RGBA', 'L').
        channels: Number of colour channels (1 = grayscale, 3 = RGB, 4 = RGBA).
        bits_per_channel: Bit depth per channel (typically 8 or 16).
        bits_per_pixel: Total bits per pixel across all channels.
        uncompressed_size_bytes: Raw pixel buffer size in bytes (no container overhead).
        file_size_bytes: Actual on-disk file size in bytes (including headers etc.).
    """

    width:                   int
    height:                  int
    mode:                    str
    channels:                int
    bits_per_channel:        int
    bits_per_pixel:          int
    uncompressed_size_bytes: int
    file_size_bytes:         int

    @property
    def resolution(self) -> int:
        """Total pixel count (width x height)."""
        return self.width * self.height

    @property
    def megapixels(self) -> float:
        """Resolution expressed in megapixels."""
        return self.resolution / 1_000_000


class ImageSizeCalculator:
    """
    Calculates true uncompressed image size as raw raster data.

    All compressor plugins must use calculate_uncompressed_size() rather than
    Path.stat().st_size so that every compressor reports a comparable baseline
    regardless of source format, metadata, or filesystem block alignment.
    """

    # Explicit mapping of Pillow mode strings to (channels, bits_per_channel)
    # Modes not listed here fall back to len(img.getbands()) channels at 8 bpp
    MODE_INFO = {
        # Binary / grayscale
        "1":      (1, 1),    # 1-bit bitmap
        "L":      (1, 8),    # 8-bit grayscale
        "LA":     (2, 8),    # 8-bit grayscale + alpha

        # Palette (index channel only; palette data itself is excluded by definition)
        "P":      (1, 8),
        "PA":     (2, 8),

        # RGB family
        "RGB":    (3, 8),
        "RGBA":   (4, 8),
        "RGBX":   (4, 8),
        "RGBa":   (4, 8),    # pre-multiplied alpha

        # Other colour spaces (treated as raw per-channel data)
        "CMYK":   (4, 8),
        "YCbCr":  (3, 8),
        "LAB":    (3, 8),
        "HSV":    (3, 8),

        # Integer / float single-channel pixels
        "I":      (1, 32),   # signed int32
        "F":      (1, 32),   # float32

        # 16-bit grayscale variants
        "I;16":   (1, 16),
        "I;16L":  (1, 16),
        "I;16B":  (1, 16),
        "I;16N":  (1, 16),

        # Packed BGR formats (bits_per_pixel, not bits_per_channel)
        "BGR;15": (1, 15),   # 5-5-5 packed
        "BGR;16": (1, 16),   # 5-6-5 packed
        "BGR;24": (3, 8),
        "BGR;32": (4, 8),
    }

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        """Integer ceiling division without floating point arithmetic."""
        return (a + b - 1) // b

    @staticmethod
    def get_image_info(image_path: Path) -> Optional[ImageInfo]:
        """
        Open an image and return its geometry and depth information.

        Args:
            image_path: Path to any Pillow-readable image file.

        Returns:
            ImageInfo on success, or None if the file cannot be opened or analysed 
            (an error message is printed to stdout).
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode

                if mode in ImageSizeCalculator.MODE_INFO:
                    channels, bits_per_channel = ImageSizeCalculator.MODE_INFO[mode]
                else:
                    # Fallback for unlisted modes: count bands and assume 8 bpp.
                    channels         = len(img.getbands())
                    bits_per_channel = 8

                # For packed BGR formats (BGR;15, BGR;16) the bits_per_channel
                # value already encodes the total bits per pixel, do not multiply.
                if mode.startswith("BGR;") and bits_per_channel in (15, 16):
                    bits_per_pixel = bits_per_channel
                else:
                    bits_per_pixel = channels * bits_per_channel

                total_bits        = width * height * bits_per_pixel
                uncompressed_size = ImageSizeCalculator._ceil_div(total_bits, 8)
                file_size         = image_path.stat().st_size

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

        except Exception as exc:
            print(f"Error analysing image {image_path}: {exc}")
            return None

    @staticmethod
    def calculate_uncompressed_size(image_path: Path) -> int:
        """
        Return the raw pixel buffer size in bytes for the given image.

        Falls back to the on-disk file size if the image cannot be analysed
        (ensures callers always receive a positive integer).

        Args:
            image_path: Path to any Pillow-readable image file.

        Returns:
            Uncompressed pixel buffer size in bytes.
        """
        info = ImageSizeCalculator.get_image_info(image_path)
        return info.uncompressed_size_bytes if info else image_path.stat().st_size

    @staticmethod
    def get_compression_baseline(image_path: Path) -> dict:
        """
        Return a summary dict comparing raw pixel size to on-disk size.

        Useful for quick sanity checks and for embedding in log output.

        Args:
            image_path: Path to any Pillow-readable image file.

        Returns:
            Dict with keys: uncompressed_size, file_size, baseline_ratio, format,
            and (when info is available) width, height, mode, channels,
            bits_per_pixel, megapixels.
        """
        info = ImageSizeCalculator.get_image_info(image_path)

        if not info or info.file_size_bytes == 0:
            return {
                "uncompressed_size": 0,
                "file_size":         0,
                "baseline_ratio":    0,
                "format":            "unknown",
            }

        ratio = info.uncompressed_size_bytes / info.file_size_bytes
        
        return {
            "uncompressed_size": info.uncompressed_size_bytes,
            "file_size":         info.file_size_bytes,
            "baseline_ratio":    ratio,
            "format":            image_path.suffix.lower(),
            "width":             info.width,
            "height":            info.height,
            "mode":              info.mode,
            "channels":          info.channels,
            "bits_per_pixel":    info.bits_per_pixel,
            "megapixels":        info.megapixels,
        }


def print_image_analysis(image_path: Path) -> None:
    """
    Print a formatted analysis of image geometry, depth, and size.

    Args:
        image_path: Path to any Pillow readable image file.
    """
    info = ImageSizeCalculator.get_image_info(image_path)

    if not info:
        print(f"Could not analyse: {image_path}")
        return

    ratio = (
        info.uncompressed_size_bytes / info.file_size_bytes
        if info.file_size_bytes > 0
        else 0.0
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
        f"({info.uncompressed_size_bytes / (1024 ** 2):.2f} MB)"
    )
    print(
        f"File size:             {info.file_size_bytes:,} bytes "
        f"({info.file_size_bytes / (1024 ** 2):.2f} MB)"
    )
    print(f"Compression ratio:     {ratio:.2f}×")
    if info.uncompressed_size_bytes > 0:
        saving = (1 - info.file_size_bytes / info.uncompressed_size_bytes) * 100
        print(f"Space saving:          {saving:.1f}%")
    else:
        print("Space saving:          n/a")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for path_str in sys.argv[1:]:
            p = Path(path_str)
            if p.exists():
                print_image_analysis(p)
            else:
                print(f"File not found: {p}")
    else:
        print("Image Size Calculator")
        print("=" * 72)
        print("Calculates true uncompressed image size (raw raster data).")
        print("Usage: python image_size_calculator.py <image_file> [...]")