"""
LibPNG Native Compressor Plugin
compressors/libpng_compressor.py

High-performance PNG lossless compression using the native libpng C library via
ctypes. This bypasses slower pure-Python implementations for the compression step
while keeping a clean, Pythonic interface.

Architecture:
  - LibPNGWriter  : Low-level ctypes wrapper around the libpng C API.
  - LibPNGCompressor : High-level plugin that satisfies the ImageCompressor interface.

Supported platform: Windows (DLL).
"""

import ctypes
import ctypes.util
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from main import CompressionLevel, CompressionMetrics, CompressorFactory, ImageCompressor
from utils.image_size_calculator import ImageSizeCalculator


# ---------------------------------------------------------------------------
# ctypes type aliases  (mirror libpng's C typedefs for readability)
# ---------------------------------------------------------------------------

png_byte    = ctypes.c_ubyte       # uint8  – a single PNG byte
png_bytep   = ctypes.POINTER(png_byte)   # uint8* – pointer to a row of bytes
png_bytepp  = ctypes.POINTER(png_bytep)  # uint8** – array of row pointers
png_structp = ctypes.c_void_p      # opaque png_struct*
png_infop   = ctypes.c_void_p      # opaque png_info*
png_voidp   = ctypes.c_voidp       # generic void*
png_uint_32 = ctypes.c_uint        # uint32
png_charp   = ctypes.c_char_p      # C string (char*)
png_FILE_p  = ctypes.c_void_p      # opaque FILE* from C stdio

# Callback types used by libpng for error / warning reporting.
# CFUNCTYPE creates a callable that C code can call directly.
PNG_ERROR_FUNC   = ctypes.CFUNCTYPE(None, png_structp, png_charp)
PNG_WARNING_FUNC = ctypes.CFUNCTYPE(None, png_structp, png_charp)


# ---------------------------------------------------------------------------
# libpng constants  (from libpng.h)
# ---------------------------------------------------------------------------

# The version string that png_create_write_struct verifies at runtime.
PNG_LIBPNG_VER_STRING = b"1.6.37"

PNG_COLOR_TYPE_RGB      = 2    # 3-channel colour image
PNG_COLOR_TYPE_RGBA     = 6    # 3-channel colour + alpha
PNG_INTERLACE_NONE      = 0    # Non-interlaced (progressive) output
PNG_COMPRESSION_TYPE_DEFAULT = 0
PNG_FILTER_TYPE_DEFAULT      = 0

# Individual row-filter flags (bitwise OR to combine)
PNG_FILTER_NONE  = 0x08
PNG_FILTER_SUB   = 0x10
PNG_FILTER_UP    = 0x20
PNG_FILTER_AVG   = 0x40
PNG_FILTER_PAETH = 0x80
PNG_ALL_FILTERS  = (
    PNG_FILTER_NONE | PNG_FILTER_SUB | PNG_FILTER_UP |
    PNG_FILTER_AVG  | PNG_FILTER_PAETH
)


# ---------------------------------------------------------------------------
# Platform-specific library discovery
# ---------------------------------------------------------------------------

def _find_libraries(lib_dir: Path) -> Tuple[str, str, str]:
    """
    Locate libpng, zlib, and the C runtime (Windows only).

    Args:
        lib_dir: Directory that contains the bundled libpng and zlib DLLs.

    Returns:
        Tuple of (libpng_path, zlib_path, libc_path) as strings suitable for
        ctypes.CDLL().

    Raises:
        FileNotFoundError: If any required library cannot be found.
    """
    libpng_path = lib_dir / "libpng16.dll"
    zlib_path   = lib_dir / "zlib1.dll"

    if not libpng_path.exists():
        raise FileNotFoundError(f"libpng not found: {libpng_path}")
    if not zlib_path.exists():
        raise FileNotFoundError(f"zlib not found: {zlib_path}")

    # msvcrt is the Microsoft C Runtime – always present on Windows.
    libc_path = ctypes.util.find_library("msvcrt")
    if not libc_path:
        raise FileNotFoundError("Microsoft C Runtime (msvcrt) not found.")

    return str(libpng_path), str(zlib_path), libc_path


# ---------------------------------------------------------------------------
# Error / warning callbacks (C-callable Python functions)
# ---------------------------------------------------------------------------

@PNG_ERROR_FUNC
def _py_png_error_handler(png_ptr: png_structp, message_ptr: png_charp):
    """
    Called by libpng on a fatal error instead of longjmp().
    Raises a Python exception so the interpreter stays alive.
    """
    message = ctypes.string_at(message_ptr).decode("utf-8", errors="ignore")
    raise RuntimeError(f"LibPNG fatal error: {message}")


@PNG_WARNING_FUNC
def _py_png_warning_handler(png_ptr: png_structp, message_ptr: png_charp):
    """Called by libpng for non-fatal warnings."""
    message = ctypes.string_at(message_ptr).decode("utf-8", errors="ignore")
    warnings.warn(f"LibPNG warning: {message}", RuntimeWarning)


# ---------------------------------------------------------------------------
# Section 1: Low-level ctypes wrapper
# ---------------------------------------------------------------------------

class LibPNGWriter:
    """
    Thin wrapper around the libpng shared library.

    Responsibilities:
      - Load libpng, zlib, and the C runtime via ctypes.
      - Declare correct argtypes / restypes for every function used.
      - Expose a single high-level write() method that encodes a NumPy array
        to a PNG file with configurable compression.
    """

    def __init__(self, libpng_path: str, zlib_path: str, libc_path: str):
        """Load the three required shared libraries."""
        try:
            # Add the DLL directory so Windows can resolve transitive dependencies
            # (e.g. libpng finding zlib) without requiring them to be on PATH.
            import os
            os.add_dll_directory(str(Path(libpng_path).parent))

            # zlib must be loaded first because libpng depends on it.
            ctypes.CDLL(zlib_path)
            self.libpng = ctypes.CDLL(libpng_path)
            self.libc   = ctypes.CDLL(libc_path)

        except OSError as exc:
            raise ImportError(
                f"Failed to load shared libraries (libpng / zlib / libc): {exc}"
            ) from exc

        self._define_prototypes()

    def _define_prototypes(self):
        """
        Declare argtypes and restypes for every libpng / libc function used.

        Correct type declarations are required for ctypes to marshal arguments
        safely, especially on 64-bit platforms where pointer size matters.
        """
        lib  = self.libpng
        libc = self.libc

        # -- write struct lifecycle --
        lib.png_create_write_struct.restype  = png_structp
        lib.png_create_write_struct.argtypes = [
            png_charp, png_voidp, png_voidp, png_voidp
        ]

        lib.png_create_info_struct.restype  = png_infop
        lib.png_create_info_struct.argtypes = [png_structp]

        lib.png_destroy_write_struct.argtypes = [
            ctypes.POINTER(png_structp), ctypes.POINTER(png_infop)
        ]

        # -- I/O and header --
        lib.png_init_io.argtypes = [png_structp, png_FILE_p]

        lib.png_set_IHDR.argtypes = [
            png_structp, png_infop,
            png_uint_32, png_uint_32,  # width, height
            ctypes.c_int,              # bit depth
            ctypes.c_int,              # color type
            ctypes.c_int,              # interlace method
            ctypes.c_int,              # compression type
            ctypes.c_int,              # filter method
        ]

        # -- compression and filter settings --
        lib.png_set_compression_level.argtypes = [png_structp, ctypes.c_int]
        lib.png_set_filter.argtypes             = [png_structp, ctypes.c_int, ctypes.c_int]

        # -- write pipeline --
        lib.png_write_info.argtypes  = [png_structp, png_infop]
        lib.png_write_image.argtypes = [png_structp, png_bytepp]
        lib.png_write_end.argtypes   = [png_structp, png_infop]

        # -- C stdio (needed because libpng writes to a FILE*) --
        libc.fopen.restype  = png_FILE_p
        libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        libc.fclose.argtypes = [png_FILE_p]

    def write(
        self,
        file_path: str,
        image_data: np.ndarray,
        compression_level: int,
        filter_type: int,
    ):
        """
        Encode a NumPy array to a PNG file using libpng.

        Args:
            file_path:         Destination PNG file path (str).
            image_data:        uint8 ndarray of shape (H, W, 3) or (H, W, 4).
            compression_level: zlib compression level 1–9 (1 = fastest, 9 = best).
            filter_type:       Bitwise OR of PNG_FILTER_* constants.

        Raises:
            ValueError:   If the image has an unsupported number of channels.
            IOError:      If the output file cannot be opened.
            RuntimeError: If any libpng call fails (via the error callback).
        """
        height, width, channels = image_data.shape

        if channels == 3:
            color_type = PNG_COLOR_TYPE_RGB
        elif channels == 4:
            color_type = PNG_COLOR_TYPE_RGBA
        else:
            raise ValueError(
                f"Expected 3 (RGB) or 4 (RGBA) channels, got {channels}."
            )

        # libpng expects a C-contiguous uint8 buffer.
        if not image_data.flags["C_CONTIGUOUS"] or image_data.dtype != np.uint8:
            image_data = np.ascontiguousarray(image_data, dtype=np.uint8)

        fp       = None
        png_ptr  = None
        info_ptr = None

        try:
            fp = self.libc.fopen(file_path.encode("utf-8"), b"wb")
            if not fp:
                raise IOError(f"Could not open file for writing: {file_path}")

            # Create the main libpng write struct, injecting our Python callbacks.
            png_ptr = self.libpng.png_create_write_struct(
                PNG_LIBPNG_VER_STRING,
                None,
                _py_png_error_handler,
                _py_png_warning_handler,
            )
            if not png_ptr:
                raise RuntimeError("png_create_write_struct() returned NULL.")

            info_ptr = self.libpng.png_create_info_struct(png_ptr)
            if not info_ptr:
                raise RuntimeError("png_create_info_struct() returned NULL.")

            # Wire up the FILE* and write the IHDR chunk.
            self.libpng.png_init_io(png_ptr, fp)
            self.libpng.png_set_IHDR(
                png_ptr, info_ptr,
                width, height,
                8,                           # bit depth
                color_type,
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT,
                PNG_FILTER_TYPE_DEFAULT,
            )
            self.libpng.png_set_compression_level(png_ptr, compression_level)
            self.libpng.png_set_filter(png_ptr, 0, filter_type)
            self.libpng.png_write_info(png_ptr, info_ptr)

            # Build the row-pointer array that png_write_image() requires.
            # Each element points to the first byte of the corresponding image row.
            row_pointers_type = png_bytep * height
            row_pointers      = row_pointers_type()
            base_address      = image_data.ctypes.data
            row_stride        = image_data.strides[0]   # bytes per row

            for i in range(height):
                row_pointers[i] = ctypes.cast(
                    base_address + i * row_stride, png_bytep
                )

            self.libpng.png_write_image(png_ptr, row_pointers)
            self.libpng.png_write_end(png_ptr, None)

        finally:
            # Always release C-level resources, even on exception.
            if png_ptr or info_ptr:
                png_ptr_ref  = ctypes.c_void_p(png_ptr)
                info_ptr_ref = ctypes.c_void_p(info_ptr)
                self.libpng.png_destroy_write_struct(
                    ctypes.byref(png_ptr_ref),
                    ctypes.byref(info_ptr_ref),
                )
            if fp:
                self.libc.fclose(fp)


# ---------------------------------------------------------------------------
# Section 2: High-level compressor plugin
# ---------------------------------------------------------------------------

class LibPNGCompressor(ImageCompressor):
    """
    PNG lossless compressor that uses the native libpng shared library.

    Compared to Pillow's PNG backend, this compressor gives precise control
    over libpng's compression level and row-filter strategy, which can yield
    smaller files at the cost of extra latency.

    Image loading for both compression and decompression is done via Pillow
    (PIL) to stay consistent with all other compressors in this project.
    """

    def __init__(self, lib_path: Optional[Path] = None):
        # Initialised to None; set by _validate_dependencies() via super().__init__().
        self.libpng_writer: Optional[LibPNGWriter] = None
        super().__init__(lib_path)

    def _validate_dependencies(self) -> None:
        """Locate bundled libpng / zlib DLLs and initialise LibPNGWriter."""
        base_dir   = Path(__file__).parent.parent
        libpng_dir = base_dir / "libs" / "libpng"

        if not libpng_dir.is_dir():
            raise RuntimeError(
                f"LibPNG library directory not found: {libpng_dir}"
            )

        libpng_path, zlib_path, libc_path = _find_libraries(libpng_dir)
        self.libpng_writer = LibPNGWriter(libpng_path, zlib_path, libc_path)

    # -- ImageCompressor interface --

    @property
    def name(self) -> str:
        return "LibPNG-Native"

    @property
    def extension(self) -> str:
        return ".png"

    def compress(
        self,
        input_path: Path,
        output_path: Path,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> CompressionMetrics:
        """
        Compress an image to PNG lossless format using native libpng.

        The uncompressed size is calculated from raw pixel data (width × height
        × channels × bytes-per-channel) so that all compressors report a
        comparable baseline.
        """
        if not self.libpng_writer:
            raise RuntimeError(
                "LibPNG writer is not initialised. "
                "Dependency validation may have failed."
            )

        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)

            # Load image via Pillow – consistent with all other compressors.
            img = Image.open(input_path)
            img.load()  # Force full decode; also needed before .info can be dropped

            # Preserve RGBA; convert exotic modes (palette, CMYK, …) to RGB.
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            # Drop any residual metadata (EXIF, ICC, XMP) that survived the
            # strip step or came from a non-stripped source.  Rebuild from raw
            # pixel data into a fresh Image with an empty .info dict.
            clean = Image.new(img.mode, img.size)
            clean.putdata(img.getdata())
            img = clean

            image_data = np.asarray(img, dtype=np.uint8)

            # Map the abstract CompressionLevel to concrete libpng parameters.
            level_params = {
                CompressionLevel.FASTEST:  (1, PNG_FILTER_NONE),
                CompressionLevel.BALANCED: (6, PNG_ALL_FILTERS),
                CompressionLevel.BEST:     (9, PNG_FILTER_PAETH),
            }
            compression_level, filter_type = level_params.get(
                level, level_params[CompressionLevel.BALANCED]
            )

            start_time = time.perf_counter()

            self.libpng_writer.write(
                str(output_path),
                image_data,
                compression_level=compression_level,
                filter_type=filter_type,
            )

            compression_time = time.perf_counter() - start_time
            compressed_size  = output_path.stat().st_size

            # Measure decompression time using a temporary file so it is
            # symmetrical with all other compressors in this project.
            temp_decomp = output_path.parent / f"temp_decomp_{output_path.stem}.png"
            try:
                decompression_time = self.decompress(output_path, temp_decomp)
            finally:
                if temp_decomp.exists():
                    temp_decomp.unlink()

            return CompressionMetrics(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=(
                    original_size / compressed_size if compressed_size > 0 else float("inf")
                ),
                compression_time=compression_time,
                decompression_time=decompression_time,
                success=True,
            )

        except Exception as exc:
            return CompressionMetrics(
                original_size=0,
                compressed_size=0,
                compression_ratio=0,
                compression_time=0,
                decompression_time=0,
                success=False,
                error_message=str(exc),
            )

    def decompress(self, input_path: Path, output_path: Path) -> float:
        """
        Decode a PNG file and save it as PNG, measuring wall-clock time.

        Using Pillow's img.load() forces the full pixel decode (not just header
        parsing), which gives a realistic decompression benchmark.
        """
        start_time = time.perf_counter()

        img = Image.open(input_path)
        img.load()          # Force full pixel decode into memory
        img.save(output_path, format="PNG")

        return time.perf_counter() - start_time


# Register so CompressorFactory.create("libpng") works.
CompressorFactory.register("libpng", LibPNGCompressor)