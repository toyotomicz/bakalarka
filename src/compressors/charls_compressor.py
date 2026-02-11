"""
CharLS (JPEG-LS) Lossless Compressor Plugin
compressors/charls_compressor.py

High-performance JPEG-LS compression using native CharLS library via ctypes.
"""

from pathlib import Path
from typing import Optional
import ctypes
import time
import sys
import platform
import os

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from main import ImageCompressor, CompressionMetrics, CompressionLevel, CompressorFactory
from image_size_calculator import ImageSizeCalculator


# --- CharLS Error Codes ---
class JpegLSError:
    SUCCESS = 0
    INVALID_ARGUMENT = 1
    DESTINATION_BUFFER_TOO_SMALL = 10
    SOURCE_BUFFER_TOO_SMALL = 11
    INVALID_ENCODED_DATA = 12


# --- Interleave Mode ---
class InterleaveMode:
    NONE = 0    # Planar: RRR GGG BBB
    LINE = 1    # By line
    SAMPLE = 2  # By pixel: RGB RGB RGB


# --- Frame Info Structure ---
class FrameInfo(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("bits_per_sample", ctypes.c_int32),
        ("component_count", ctypes.c_int32),
    ]


class CharLSError(Exception):
    """Exception for CharLS library errors"""
    pass


class CharLSCompressor(ImageCompressor):
    """
    JPEG-LS lossless compressor using CharLS native library
    """
    
    def __init__(self, lib_path: Optional[Path] = None):
        self.charls_lib = None
        super().__init__(lib_path)
    
    def _validate_dependencies(self) -> None:
        """Find and load CharLS library"""
        base_dir = Path(__file__).parent.parent
        charls_dir = base_dir / "libs" / "charls"
        
        if not charls_dir.exists():
            raise RuntimeError(f"CharLS library directory not found: {charls_dir}")
        
        # Determine library name based on platform
        system = platform.system().lower()
        if system == "windows":
            lib_name = "charls-3-x64.dll"
        elif system == "linux":
            lib_name = "libcharls.so"
        
        lib_path = charls_dir / lib_name
        
        if not lib_path.exists():
            raise RuntimeError(f"CharLS library not found: {lib_path}")
        
        try:
            self.charls_lib = ctypes.CDLL(str(lib_path))
            self._setup_function_prototypes()
        except OSError as e:
            raise RuntimeError(f"Failed to load CharLS library: {e}")
    
    def _setup_function_prototypes(self):
        """Define function prototypes for CharLS library"""
        lib = self.charls_lib
        
        # Encoder functions
        lib.charls_jpegls_encoder_create.restype = ctypes.c_void_p
        lib.charls_jpegls_encoder_destroy.argtypes = [ctypes.c_void_p]
        
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
        
        lib.charls_jpegls_encoder_set_destination_buffer.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
        ]
        lib.charls_jpegls_encoder_set_destination_buffer.restype = ctypes.c_int
        
        lib.charls_jpegls_encoder_get_estimated_destination_size.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.charls_jpegls_encoder_get_estimated_destination_size.restype = ctypes.c_int
        
        lib.charls_jpegls_encoder_encode_from_buffer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32
        ]
        lib.charls_jpegls_encoder_encode_from_buffer.restype = ctypes.c_int
        
        lib.charls_jpegls_encoder_get_bytes_written.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.charls_jpegls_encoder_get_bytes_written.restype = ctypes.c_int
        
        # Decoder functions
        lib.charls_jpegls_decoder_create.restype = ctypes.c_void_p
        lib.charls_jpegls_decoder_destroy.argtypes = [ctypes.c_void_p]
        
        lib.charls_jpegls_decoder_set_source_buffer.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
        ]
        lib.charls_jpegls_decoder_set_source_buffer.restype = ctypes.c_int
        
        lib.charls_jpegls_decoder_read_header.argtypes = [ctypes.c_void_p]
        lib.charls_jpegls_decoder_read_header.restype = ctypes.c_int
        
        lib.charls_jpegls_decoder_get_frame_info.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(FrameInfo)
        ]
        lib.charls_jpegls_decoder_get_frame_info.restype = ctypes.c_int
        
        lib.charls_jpegls_decoder_decode_to_buffer.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32
        ]
        lib.charls_jpegls_decoder_decode_to_buffer.restype = ctypes.c_int
    
    @property
    def name(self) -> str:
        return "CharLS-JPEGLS"
    
    @property
    def extension(self) -> str:
        return ".jls"
    
    def compress(self, 
                 input_path: Path, 
                 output_path: Path,
                 level: CompressionLevel = CompressionLevel.BALANCED) -> CompressionMetrics:
        """Compress image to JPEG-LS format"""
        
        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)
            
            # Load image using PIL
            img = Image.open(input_path)
            
            # Convert RGBA to RGB if needed (JPEG-LS works better with RGB)
            if img.mode not in ('RGB', 'RGBA', 'L', 'LA'):
                img = img.convert('RGB')
            elif img.mode not in ('RGB', 'L'):
                # Convert other modes to RGB
                img = img.convert('RGB')
            
            image_data = np.array(img)
            
            # Ensure proper shape
            if image_data.ndim == 2:
                # Grayscale - add channel dimension
                image_data = image_data[:, :, np.newaxis]
            
            # Ensure C-contiguous
            if not image_data.flags['C_CONTIGUOUS']:
                image_data = np.ascontiguousarray(image_data)
            
            start_time = time.perf_counter()
            
            # Encode
            encoded_data = self._encode(image_data, level)
            
            # Write to file
            with open(output_path, 'wb') as f:
                f.write(encoded_data)
            
            compression_time = time.perf_counter() - start_time
            compressed_size = len(encoded_data)
            
            # Test decompression
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
    
    def _encode(self, image_data: np.ndarray, level: CompressionLevel) -> bytes:
        """Encode image data to JPEG-LS"""
        height, width = image_data.shape[:2]
        components = image_data.shape[2] if image_data.ndim == 3 else 1
        
        if image_data.dtype == np.uint8:
            bpp = 8
        elif image_data.dtype == np.uint16:
            bpp = 16
        else:
            raise ValueError("Only uint8 and uint16 are supported")
        
        # Always use truly lossless compression (near=0)
        # Compression level doesn't affect quality, only encoding speed/size tradeoffs
        near_lossless = 0
        
        # Create encoder
        encoder = self.charls_lib.charls_jpegls_encoder_create()
        if not encoder:
            raise CharLSError("Failed to create encoder")
        
        try:
            # Set frame info
            frame_info = FrameInfo(width, height, bpp, components)
            result = self.charls_lib.charls_jpegls_encoder_set_frame_info(
                encoder, ctypes.byref(frame_info)
            )
            if result != JpegLSError.SUCCESS:
                raise CharLSError(f"Failed to set frame info: {result}")
            
            # Set near-lossless parameter based on compression level
            result = self.charls_lib.charls_jpegls_encoder_set_near_lossless(
                encoder, near_lossless
            )
            if result != JpegLSError.SUCCESS:
                raise CharLSError(f"Failed to set near-lossless: {result}")
            
            # Set interleave mode for color images
            if components > 1:
                result = self.charls_lib.charls_jpegls_encoder_set_interleave_mode(
                    encoder, InterleaveMode.SAMPLE
                )
                if result != JpegLSError.SUCCESS:
                    raise CharLSError(f"Failed to set interleave mode: {result}")
            
            # Get estimated size
            estimated_size = ctypes.c_size_t()
            result = self.charls_lib.charls_jpegls_encoder_get_estimated_destination_size(
                encoder, ctypes.byref(estimated_size)
            )
            if result != JpegLSError.SUCCESS:
                raise CharLSError(f"Failed to get estimated size: {result}")
            
            # Add 20% safety margin to buffer size
            buffer_size = int(estimated_size.value * 1.2)
            dest_buffer = ctypes.create_string_buffer(buffer_size)
            
            # CRITICAL FIX: Set the destination buffer BEFORE encoding
            result = self.charls_lib.charls_jpegls_encoder_set_destination_buffer(
                encoder,
                dest_buffer,
                buffer_size
            )
            if result != JpegLSError.SUCCESS:
                raise CharLSError(f"Failed to set destination buffer: {result}")
            
            # Encode (now with only source buffer parameters)
            source_ptr = image_data.ctypes.data_as(ctypes.c_void_p)
            source_size = image_data.nbytes
            
            result = self.charls_lib.charls_jpegls_encoder_encode_from_buffer(
                encoder,
                source_ptr,
                source_size,
                0  # stride
            )
            if result != JpegLSError.SUCCESS:
                raise CharLSError(f"Encoding failed: {result}")
            
            # Get bytes written
            bytes_written = ctypes.c_size_t()
            result = self.charls_lib.charls_jpegls_encoder_get_bytes_written(
                encoder, ctypes.byref(bytes_written)
            )
            if result != JpegLSError.SUCCESS:
                raise CharLSError(f"Failed to get bytes written: {result}")
            
            return dest_buffer.raw[:bytes_written.value]
            
        finally:
            self.charls_lib.charls_jpegls_encoder_destroy(encoder)
    
    def decompress(self, input_path: Path, output_path: Path) -> float:
        """Decompress JPEG-LS file"""
        start_time = time.perf_counter()
        
        # Read compressed data
        with open(input_path, 'rb') as f:
            source_data = f.read()
        
        # Decode
        decoded_array = self._decode(source_data)
        
        # Save as PNG
        if decoded_array.ndim == 3 and decoded_array.shape[2] == 1:
            decoded_array = decoded_array[:, :, 0]
        
        img = Image.fromarray(decoded_array)
        img.save(output_path)
        
        return time.perf_counter() - start_time
    
    def _decode(self, source_data: bytes) -> np.ndarray:
        """Decode JPEG-LS data"""
        # Create decoder
        decoder = self.charls_lib.charls_jpegls_decoder_create()
        if not decoder:
            raise CharLSError("Failed to create decoder")
        
        try:
            # Set source buffer
            source_ptr = ctypes.c_char_p(source_data)
            result = self.charls_lib.charls_jpegls_decoder_set_source_buffer(
                decoder, source_ptr, len(source_data)
            )
            if result != JpegLSError.SUCCESS:
                raise CharLSError(f"Failed to set source buffer: {result}")
            
            # Read header
            result = self.charls_lib.charls_jpegls_decoder_read_header(decoder)
            if result != JpegLSError.SUCCESS:
                raise CharLSError(f"Failed to read header: {result}")
            
            # Get frame info
            frame_info = FrameInfo()
            result = self.charls_lib.charls_jpegls_decoder_get_frame_info(
                decoder, ctypes.byref(frame_info)
            )
            if result != JpegLSError.SUCCESS:
                raise CharLSError(f"Failed to get frame info: {result}")
            
            # Determine dtype
            if frame_info.bits_per_sample <= 8:
                dtype = np.uint8
            elif frame_info.bits_per_sample <= 16:
                dtype = np.uint16
            else:
                raise ValueError(f"Unsupported bit depth: {frame_info.bits_per_sample}")
            
            # Create output array
            shape = (frame_info.height, frame_info.width)
            if frame_info.component_count > 1:
                shape += (frame_info.component_count,)
            
            dest_array = np.empty(shape, dtype=dtype)
            dest_ptr = dest_array.ctypes.data_as(ctypes.c_void_p)
            dest_size = dest_array.nbytes
            
            # Decode
            result = self.charls_lib.charls_jpegls_decoder_decode_to_buffer(
                decoder, dest_ptr, dest_size, 0
            )
            if result != JpegLSError.SUCCESS:
                raise CharLSError(f"Decoding failed: {result}")
            
            return dest_array
            
        finally:
            self.charls_lib.charls_jpegls_decoder_destroy(decoder)


# Automatic registration
CompressorFactory.register("charls", CharLSCompressor)