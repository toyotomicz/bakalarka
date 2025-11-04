"""
Image Verification Module
Validates lossless compression by pixel-level comparison
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class VerificationResult:
    """Result of lossless verification"""
    is_lossless: bool
    max_difference: float
    different_pixels: int
    total_pixels: int
    error_message: Optional[str] = None
    
    @property
    def accuracy_percent(self) -> float:
        """Calculate accuracy as percentage of identical pixels"""
        if self.total_pixels == 0:
            return 0.0
        return ((self.total_pixels - self.different_pixels) / self.total_pixels) * 100
    
    @property
    def identical_pixels(self) -> int:
        """Number of pixels that match exactly"""
        return self.total_pixels - self.different_pixels


class ImageVerifier:
    """Validates that compression is truly lossless"""
    
    @staticmethod
    def verify_lossless(
        original_path: Path,
        compressed_path: Path,
        temp_dir: Optional[Path] = None
    ) -> VerificationResult:
        """
        Compare original and decompressed images pixel by pixel
        
        Args:
            original_path: Path to original image
            compressed_path: Path to compressed image (will be decoded first if needed)
            temp_dir: Directory for temporary decompressed file
            
        Returns:
            VerificationResult with detailed comparison metrics
        """
        try:
            # Load original image
            img_original = Image.open(original_path)
            
            # Try to load compressed image directly
            # PIL can handle many formats, but for custom formats we may need decompression
            try:
                img_compressed = Image.open(compressed_path)
            except Exception:
                # If PIL can't open it, we need to decompress it first
                # This should trigger for formats like .jls that PIL doesn't support
                decompressed_path = ImageVerifier._decompress_if_needed(
                    compressed_path, 
                    temp_dir or compressed_path.parent
                )
                
                if decompressed_path is None:
                    return VerificationResult(
                        is_lossless=False,
                        max_difference=0,
                        different_pixels=0,
                        total_pixels=0,
                        error_message=f"Cannot decompress {compressed_path.suffix} format"
                    )
                
                img_compressed = Image.open(decompressed_path)
                
                # Clean up temp file
                if decompressed_path != compressed_path:
                    try:
                        decompressed_path.unlink()
                    except:
                        pass
            
            # Convert to same mode if needed
            if img_original.mode != img_compressed.mode:
                img_compressed = img_compressed.convert(img_original.mode)
            
            # Validate dimensions
            if img_original.size != img_compressed.size:
                total_pixels = img_original.size[0] * img_original.size[1]
                return VerificationResult(
                    is_lossless=False,
                    max_difference=float('inf'),
                    different_pixels=total_pixels,
                    total_pixels=total_pixels,
                    error_message=f"Dimension mismatch: {img_original.size} vs {img_compressed.size}"
                )
            
            # Convert to numpy arrays
            arr_original = np.array(img_original)
            arr_compressed = np.array(img_compressed)
            
            # Handle dimension mismatch (grayscale vs RGB)
            if arr_original.ndim != arr_compressed.ndim:
                if arr_original.ndim == 2:
                    arr_original = np.expand_dims(arr_original, axis=-1)
                if arr_compressed.ndim == 2:
                    arr_compressed = np.expand_dims(arr_compressed, axis=-1)
            
            # Calculate pixel differences
            diff = np.abs(
                arr_original.astype(np.float32) - arr_compressed.astype(np.float32)
            )
            
            max_diff = np.max(diff)
            
            # Count different pixels
            if diff.ndim == 3:
                different_pixels = np.sum(np.any(diff > 0, axis=-1))
            else:
                different_pixels = np.sum(diff > 0)
            
            total_pixels = arr_original.shape[0] * arr_original.shape[1]
            is_lossless = (max_diff == 0)
            
            return VerificationResult(
                is_lossless=is_lossless,
                max_difference=float(max_diff),
                different_pixels=int(different_pixels),
                total_pixels=total_pixels
            )
            
        except Exception as e:
            return VerificationResult(
                is_lossless=False,
                max_difference=0,
                different_pixels=0,
                total_pixels=0,
                error_message=str(e)
            )
    
    @staticmethod
    def _decompress_if_needed(compressed_path: Path, temp_dir: Path) -> Optional[Path]:
        """
        Decompress file if it's in a format that needs explicit decompression
        
        Returns:
            Path to decompressed file or None if decompression failed
        """
        # Import compressor factory to get the right decompressor
        try:
            from main import CompressorFactory
            
            # Find the right compressor based on file extension
            available = CompressorFactory.list_available()
            
            for comp_name in available:
                try:
                    compressor = CompressorFactory.create(comp_name)
                    
                    # Check if this compressor handles this extension
                    if compressed_path.suffix == compressor.extension:
                        # Decompress to temporary file
                        temp_output = temp_dir / f"temp_verify_{compressed_path.stem}.png"
                        compressor.decompress(compressed_path, temp_output)
                        return temp_output
                except:
                    continue
            
            return None
            
        except Exception:
            return None
    
    @staticmethod
    def create_difference_map(
        original_path: Path,
        compressed_path: Path,
        temp_dir: Optional[Path] = None
    ) -> Optional[np.ndarray]:
        """
        Create a binary mask showing which pixels differ
        
        Returns:
            numpy array where True indicates different pixels, or None on error
        """
        try:
            img_original = Image.open(original_path)
            
            # Try to load or decompress compressed image
            try:
                img_compressed = Image.open(compressed_path)
            except Exception:
                decompressed_path = ImageVerifier._decompress_if_needed(
                    compressed_path,
                    temp_dir or compressed_path.parent
                )
                
                if decompressed_path is None:
                    return None
                
                img_compressed = Image.open(decompressed_path)
                
                # Clean up
                if decompressed_path != compressed_path:
                    try:
                        decompressed_path.unlink()
                    except:
                        pass
            
            if img_original.size != img_compressed.size:
                return None
            
            img_compressed = img_compressed.convert(img_original.mode)
            
            arr_original = np.array(img_original)
            arr_compressed = np.array(img_compressed)
            
            diff = np.abs(
                arr_original.astype(np.float32) - arr_compressed.astype(np.float32)
            )
            
            if diff.ndim == 3:
                diff_mask = np.any(diff > 0, axis=-1)
            else:
                diff_mask = diff > 0
            
            return diff_mask
            
        except Exception:
            return None