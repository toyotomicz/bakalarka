"""
CharLS (JPEG-LS) Lossless Compressor Plugin
compressors/charls_compressor.py

High-performance JPEG-LS lossless compression using the native CharLS library
loaded via ctypes. JPEG-LS is a lossless / near-lossless image compression
standard optimised for medical imaging (DICOM), offering better ratios than
PNG at comparable decode speeds.

Binary location: libs/charls/charls-3-x64.dll (Windows)
CharLS documentation: https://github.com/team-charls/charls
"""

import ctypes
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from main import CompressionLevel, CompressionMetrics, CompressorFactory, ImageCompressor
from image_size_calculator import ImageSizeCalculator


# ---------------------------------------------------------------------------
# CharLS C API constants and structures
# ---------------------------------------------------------------------------

class JpegLSError:
    """Error codes returned by CharLS API functions (charls_jpegls_errc enum)."""
    SUCCESS                      = 0
    INVALID_ARGUMENT             = 1
    DESTINATION_BUFFER_TOO_SMALL = 10
    SOURCE_BUFFER_TOO_SMALL      = 11
    INVALID_ENCODED_DATA         = 12


class InterleaveMode:
    """
    Pixel interleave mode for multi-channel images (charls_interleave_mode enum).

    NONE   (0): Planar layout – all R samples, then all G, then all B (RRR GGG BBB).
    LINE   (1): Interleaved by row.
    SAMPLE (2): Interleaved by pixel – RGB RGB RGB (standard for most images).
    """
    NONE   = 0
    LINE   = 1
    SAMPLE = 2


class FrameInfo(ctypes.Structure):
    """
    Maps to charls_frame_info in the CharLS C API.
    Describes the geometry and bit depth of a single image frame.
    """
    _fields_ = [
        ("width",           ctypes.c_uint32),  # Image width in pixels
        ("height",          ctypes.c_uint32),  # Image height in pixels
        ("bits_per_sample", ctypes.c_int32),   # Bit depth per channel (8 or 16)
        ("component_count", ctypes.c_int32),   # Number of colour channels (1, 3, or 4)
    ]


class CharLSError(Exception):
    """Raised when a CharLS API function returns a non-SUCCESS error code."""
    pass


# ---------------------------------------------------------------------------
# High-level compressor plugin
# ---------------------------------------------------------------------------

class CharLSCompressor(ImageCompressor):
    """
    JPEG-LS lossless compressor using the native CharLS shared library.

    CharLS always produces bit-exact lossless output (near-lossless parameter
    is hard-coded to 0). The CompressionLevel argument is accepted for
    interface compatibility but does not affect output quality or file size
    (JPEG-LS has no equivalent of PNG's deflate level dial).

    Supports 8-bit and 16-bit images in grayscale (L), RGB, and RGBA modes.
    RGBA is converted to RGB because JPEG-LS has no native alpha channel.
    """

    def __init__(self, lib_path: Optional[Path] = None):
        # Filled by _validate_dependencies(); kept as None until then.
        self._lib: Optional[ctypes.CDLL] = None
        super().__init__(lib_path)

    def _validate_dependencies(self) -> None:
        """Locate and load the CharLS shared library, then declare function prototypes."""
        base_dir  = Path(__file__).parent.parent
        charls_dir = base_dir / "libs" / "charls"
        lib_name  = _library_name()
        lib_path  = charls_dir / lib_name

        if not charls_dir.exists():
            raise RuntimeError(f"CharLS library directory not found: {charls_dir}")
        if not lib_path.exists():
            raise RuntimeError(f"CharLS library not found: {lib_path}")

        try:
            self._lib = ctypes.CDLL(str(lib_path))
        except OSError as exc:
            raise RuntimeError(f"Failed to load CharLS library: {exc}") from exc

        self._declare_prototypes()

    def _declare_prototypes(self) -> None:
        """
        Declare argtypes / restypes for all CharLS API functions used.

        Correct declarations are required for ctypes to marshal arguments
        safely on both 32-bit and 64-bit platforms.
        """
        lib = self._lib

        # -- Encoder lifecycle --
        lib.charls_jpegls_encoder_create.restype  = ctypes.c_void_p
        lib.charls_jpegls_encoder_destroy.argtypes = [ctypes.c_void_p]

        # -- Encoder configuration --
        lib.charls_jpegls_encoder_set_frame_info.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(FrameInfo)
        ]
        lib.charls_jpegls_encoder_set_frame_info.restype = ctypes.c_int

        lib.charls_jpegls_encoder_set_near_lossless.argtypes = [
            ctypes.c_void_p, ctypes.c_int32
        ]
        lib.charls_jpegls_encoder_set_near_lossless.restype = ctypes.c_int

        lib.charls_jpegls_encoder_set_interleave_mode.argtypes = [
            ctypes.c_void_p, ctypes.c_int
        ]
        lib.charls_jpegls_encoder_set_interleave_mode.restype = ctypes.c_int

        # -- Encoder I/O --
        lib.charls_jpegls_encoder_get_estimated_destination_size.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.charls_jpegls_encoder_get_estimated_destination_size.restype = ctypes.c_int

        lib.charls_jpegls_encoder_set_destination_buffer.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
        ]
        lib.charls_jpegls_encoder_set_destination_buffer.restype = ctypes.c_int

        lib.charls_jpegls_encoder_encode_from_buffer.argtypes = [
            ctypes.c_void_p,   # encoder
            ctypes.c_void_p,   # source buffer
            ctypes.c_size_t,   # source size in bytes
            ctypes.c_uint32,   # stride (0 = tightly packed)
        ]
        lib.charls_jpegls_encoder_encode_from_buffer.restype = ctypes.c_int

        lib.charls_jpegls_encoder_get_bytes_written.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.charls_jpegls_encoder_get_bytes_written.restype = ctypes.c_int

        # -- Decoder lifecycle --
        lib.charls_jpegls_decoder_create.restype  = ctypes.c_void_p
        lib.charls_jpegls_decoder_destroy.argtypes = [ctypes.c_void_p]

        # -- Decoder I/O --
        lib.charls_jpegls_decoder_set_source_buffer.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
        ]
        lib.charls_jpegls_decoder_set_source_buffer.restype = ctypes.c_int

        lib.charls_jpegls_decoder_read_header.argtypes = [ctypes.c_void_p]
        lib.charls_jpegls_decoder_read_header.restype  = ctypes.c_int

        lib.charls_jpegls_decoder_get_frame_info.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(FrameInfo)
        ]
        lib.charls_jpegls_decoder_get_frame_info.restype = ctypes.c_int

        lib.charls_jpegls_decoder_decode_to_buffer.argtypes = [
            ctypes.c_void_p,   # decoder
            ctypes.c_void_p,   # destination buffer
            ctypes.c_size_t,   # destination size in bytes
            ctypes.c_uint32,   # stride (0 = tightly packed)
        ]
        lib.charls_jpegls_decoder_decode_to_buffer.restype = ctypes.c_int

    # -- ImageCompressor interface --

    @property
    def name(self) -> str:
        return "CharLS-JPEGLS"

    @property
    def extension(self) -> str:
        return ".jls"

    def compress(
        self,
        input_path: Path,
        output_path: Path,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> CompressionMetrics:
        """
        Compress an image to JPEG-LS (.jls) format using CharLS.

        Note: JPEG-LS does not expose a compression-level dial comparable to
        PNG's deflate level; the 'level' parameter is ignored. Output is always
        bit-exact lossless (near_lossless = 0).
        """
        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)

            # Load and prepare the image.
            img = Image.open(input_path)

            # JPEG-LS supports 1-channel (L) and 3-channel (RGB) natively.
            # Convert everything else to RGB; RGBA → RGB (alpha discarded).
            if img.mode not in ("L", "RGB"):
                img = img.convert("RGB")

            image_data = np.array(img)

            # Grayscale images arrive as (H, W); add a channel dim for uniform handling.
            if image_data.ndim == 2:
                image_data = image_data[:, :, np.newaxis]

            # CharLS requires a C-contiguous buffer.
            if not image_data.flags["C_CONTIGUOUS"]:
                image_data = np.ascontiguousarray(image_data)

            start_time   = time.perf_counter()
            encoded_data = self._encode(image_data)
            output_path.write_bytes(encoded_data)
            compression_time = time.perf_counter() - start_time

            compressed_size = len(encoded_data)

            # Measure decompression time via a temporary decode.
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

    def _encode(self, image_data: np.ndarray) -> bytes:
        """
        Encode a NumPy array to a raw JPEG-LS byte string.

        Args:
            image_data: uint8 or uint16 array of shape (H, W, C) with C in {1, 3}.

        Returns:
            Raw JPEG-LS bytes (no file header, just the bitstream).

        Raises:
            CharLSError: If any CharLS API call fails.
            ValueError:  If the array dtype is not uint8 or uint16.
        """
        height, width = image_data.shape[:2]
        components    = image_data.shape[2] if image_data.ndim == 3 else 1

        if image_data.dtype == np.uint8:
            bpp = 8
        elif image_data.dtype == np.uint16:
            bpp = 16
        else:
            raise ValueError(
                f"Only uint8 and uint16 are supported; got {image_data.dtype}."
            )

        encoder = self._lib.charls_jpegls_encoder_create()
        if not encoder:
            raise CharLSError("charls_jpegls_encoder_create() returned NULL.")

        try:
            # Configure frame geometry.
            frame_info = FrameInfo(width, height, bpp, components)
            _check(self._lib.charls_jpegls_encoder_set_frame_info(
                encoder, ctypes.byref(frame_info)
            ), "set_frame_info")

            # near_lossless = 0 → bit-exact lossless compression.
            _check(self._lib.charls_jpegls_encoder_set_near_lossless(
                encoder, 0
            ), "set_near_lossless")

            # Multi-channel images use per-pixel interleaving (RGB RGB RGB …).
            if components > 1:
                _check(self._lib.charls_jpegls_encoder_set_interleave_mode(
                    encoder, InterleaveMode.SAMPLE
                ), "set_interleave_mode")

            # Ask CharLS for a safe upper bound on the compressed size.
            estimated = ctypes.c_size_t()
            _check(self._lib.charls_jpegls_encoder_get_estimated_destination_size(
                encoder, ctypes.byref(estimated)
            ), "get_estimated_destination_size")

            # Add a 20 % safety margin in case the estimate is tight.
            buffer_size = int(estimated.value * 1.2)
            dest_buffer = ctypes.create_string_buffer(buffer_size)

            # Set the destination buffer BEFORE calling encode.
            _check(self._lib.charls_jpegls_encoder_set_destination_buffer(
                encoder, dest_buffer, buffer_size
            ), "set_destination_buffer")

            # Perform the encoding.
            source_ptr  = image_data.ctypes.data_as(ctypes.c_void_p)
            source_size = image_data.nbytes
            _check(self._lib.charls_jpegls_encoder_encode_from_buffer(
                encoder, source_ptr, source_size, 0  # stride 0 = tightly packed
            ), "encode_from_buffer")

            # Query how many bytes were actually written.
            bytes_written = ctypes.c_size_t()
            _check(self._lib.charls_jpegls_encoder_get_bytes_written(
                encoder, ctypes.byref(bytes_written)
            ), "get_bytes_written")

            return dest_buffer.raw[: bytes_written.value]

        finally:
            # Always release the encoder, even on exception.
            self._lib.charls_jpegls_encoder_destroy(encoder)

    def decompress(self, input_path: Path, output_path: Path) -> float:
        """
        Decode a JPEG-LS (.jls) file and save it as PNG, measuring wall-clock time.

        The full byte read + decode + PNG save is included in the timing so the
        measurement is comparable with other compressors that use Pillow to save.
        """
        start_time = time.perf_counter()

        raw_data     = input_path.read_bytes()
        decoded      = self._decode(raw_data)

        # Remove the dummy channel dimension added for grayscale images.
        if decoded.ndim == 3 and decoded.shape[2] == 1:
            decoded = decoded[:, :, 0]

        img = Image.fromarray(decoded)
        img.save(output_path, format="PNG")

        return time.perf_counter() - start_time

    def _decode(self, source_data: bytes) -> np.ndarray:
        """
        Decode raw JPEG-LS bytes to a NumPy array.

        Args:
            source_data: Raw JPEG-LS bitstream bytes.

        Returns:
            uint8 or uint16 ndarray of shape (H, W, C) or (H, W) for grayscale.

        Raises:
            CharLSError: If any CharLS API call fails.
        """
        decoder = self._lib.charls_jpegls_decoder_create()
        if not decoder:
            raise CharLSError("charls_jpegls_decoder_create() returned NULL.")

        try:
            # Point the decoder at the source bytes.
            source_ptr = ctypes.c_char_p(source_data)
            _check(self._lib.charls_jpegls_decoder_set_source_buffer(
                decoder, source_ptr, len(source_data)
            ), "set_source_buffer")

            # Parse the JPEG-LS header to discover frame geometry.
            _check(self._lib.charls_jpegls_decoder_read_header(decoder), "read_header")

            frame_info = FrameInfo()
            _check(self._lib.charls_jpegls_decoder_get_frame_info(
                decoder, ctypes.byref(frame_info)
            ), "get_frame_info")

            # Allocate the destination array with the correct dtype.
            dtype = np.uint8 if frame_info.bits_per_sample <= 8 else np.uint16
            shape = (frame_info.height, frame_info.width)
            if frame_info.component_count > 1:
                shape += (frame_info.component_count,)

            dest_array = np.empty(shape, dtype=dtype)
            dest_ptr   = dest_array.ctypes.data_as(ctypes.c_void_p)
            _check(self._lib.charls_jpegls_decoder_decode_to_buffer(
                decoder, dest_ptr, dest_array.nbytes, 0  # stride 0 = tightly packed
            ), "decode_to_buffer")

            return dest_array

        finally:
            # Always release the decoder, even on exception.
            self._lib.charls_jpegls_decoder_destroy(decoder)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(return_code: int, operation: str) -> None:
    """
    Raise CharLSError if a CharLS API function returned a non-SUCCESS code.

    Args:
        return_code: Value returned by the CharLS function.
        operation:   Short human-readable name used in the error message.
    """
    if return_code != JpegLSError.SUCCESS:
        raise CharLSError(
            f"CharLS '{operation}' failed with error code {return_code}."
        )


def _library_name() -> str:
    """Return the CharLS shared library filename (Windows only)."""
    return "charls-3-x64.dll"


# Register so CompressorFactory.create("charls") works.
CompressorFactory.register("charls", CharLSCompressor)