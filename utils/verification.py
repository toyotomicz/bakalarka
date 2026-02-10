"""
Image Verification Module
Validates lossless compression by pixel-level comparison.

Záměrně NEIMPORTUJE z main.py — závislost na CompressorFactory
je předávána jako parametr (dependency injection), čímž se zabraňuje
kruhovým importům a usnadňuje testování.
"""

from pathlib import Path
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass
import logging

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    # Pouze pro type checkery — za runtime se neimportuje.
    from main import CompressorFactory as CompressorFactoryType

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Výsledek pixel-level srovnání originálního a dekomprimovaného obrázku."""
    is_lossless: bool
    max_difference: float
    different_pixels: int
    total_pixels: int
    error_message: Optional[str] = None

    @property
    def accuracy_percent(self) -> float:
        """Procento pixelů, které se bit-přesně shodují."""
        if self.total_pixels == 0:
            return 0.0
        return ((self.total_pixels - self.different_pixels) / self.total_pixels) * 100.0

    @property
    def identical_pixels(self) -> int:
        return self.total_pixels - self.different_pixels


class ImageVerifier:
    """
    Ověřuje, zda je komprese skutečně bezztrátová.

    Příklad použití:
        result = ImageVerifier.verify_lossless(
            original_path=Path("foto.png"),
            compressed_path=Path("foto.jls"),
            compressor_factory=CompressorFactory,   # předáno zvenku
            temp_dir=Path("/tmp"),
        )
    """

    @staticmethod
    def verify_lossless(
        original_path: Path,
        compressed_path: Path,
        compressor_factory: Optional["CompressorFactoryType"] = None,
        temp_dir: Optional[Path] = None,
    ) -> VerificationResult:
        """
        Srovná originál a dekomprimovanou verzi pixel po pixelu.

        Args:
            original_path:       Cesta k originálnímu obrázku (stripped verze,
                                 stejná jako vstup kompresoru — viz bug #4).
            compressed_path:     Cesta ke komprimovanému souboru.
            compressor_factory:  Instance/třída CompressorFactory pro dekomprimaci
                                 formátů nepodporovaných PIL. Pokud None, PIL musí
                                 zvládnout otevřít compressed_path přímo.
            temp_dir:            Adresář pro dočasné dekomprimované soubory.

        Returns:
            VerificationResult s detailními metrikami.
        """
        try:
            img_original = Image.open(original_path)

            img_compressed, temp_path_to_cleanup = ImageVerifier._open_compressed(
                compressed_path=compressed_path,
                compressor_factory=compressor_factory,
                temp_dir=temp_dir or compressed_path.parent,
            )

            if img_compressed is None:
                return VerificationResult(
                    is_lossless=False,
                    max_difference=0.0,
                    different_pixels=0,
                    total_pixels=0,
                    error_message=f"Nelze otevřít nebo dekomprimovat {compressed_path.suffix}",
                )

            try:
                return ImageVerifier._compare(img_original, img_compressed)
            finally:
                # Uklidíme dočasný soubor, pokud byl vytvořen.
                if temp_path_to_cleanup is not None:
                    try:
                        temp_path_to_cleanup.unlink()
                    except OSError as e:
                        logger.debug("Nepodařilo se smazat temp soubor %s: %s",
                                     temp_path_to_cleanup, e)

        except Exception as e:
            logger.exception("verify_lossless selhalo pro %s", compressed_path)
            return VerificationResult(
                is_lossless=False,
                max_difference=0.0,
                different_pixels=0,
                total_pixels=0,
                error_message=str(e),
            )

    @staticmethod
    def create_difference_map(
        original_path: Path,
        compressed_path: Path,
        compressor_factory: Optional["CompressorFactoryType"] = None,
        temp_dir: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """
        Vrátí binární masku s True tam, kde se pixely liší.

        Returns:
            numpy bool array (H × W), nebo None při chybě.
        """
        try:
            img_original = Image.open(original_path)

            img_compressed, temp_path_to_cleanup = ImageVerifier._open_compressed(
                compressed_path=compressed_path,
                compressor_factory=compressor_factory,
                temp_dir=temp_dir or compressed_path.parent,
            )

            if img_compressed is None:
                return None

            try:
                if img_original.size != img_compressed.size:
                    return None

                img_compressed = img_compressed.convert(img_original.mode)
                arr_orig = np.array(img_original, dtype=np.float32)
                arr_comp = np.array(img_compressed, dtype=np.float32)
                diff = np.abs(arr_orig - arr_comp)

                return np.any(diff > 0, axis=-1) if diff.ndim == 3 else diff > 0

            finally:
                if temp_path_to_cleanup is not None:
                    try:
                        temp_path_to_cleanup.unlink()
                    except OSError as e:
                        logger.debug("Nepodařilo se smazat temp soubor %s: %s",
                                     temp_path_to_cleanup, e)

        except Exception:
            logger.exception("create_difference_map selhalo pro %s", compressed_path)
            return None

    # ------------------------------------------------------------------
    # Interní pomocné metody
    # ------------------------------------------------------------------

    @staticmethod
    def _open_compressed(
        compressed_path: Path,
        compressor_factory: Optional["CompressorFactoryType"],
        temp_dir: Path,
    ) -> tuple[Optional[Image.Image], Optional[Path]]:
        """
        Pokusí se otevřít komprimovaný soubor.

        Nejprve zkusí PIL přímo. Pokud selže a je předán compressor_factory,
        pokusí se soubor dekomprimovat přes příslušný kompresor.

        Returns:
            (PIL Image nebo None, cesta k temp souboru ke smazání nebo None)
        """
        # Pokus 1: PIL přímo
        try:
            return Image.open(compressed_path), None
        except Exception:
            pass

        # Pokus 2: dekomprimace přes CompresorFactory
        if compressor_factory is None:
            logger.debug("PIL nemůže otevřít %s a compressor_factory není k dispozici",
                         compressed_path)
            return None, None

        temp_output = ImageVerifier._decompress_via_factory(
            compressed_path=compressed_path,
            compressor_factory=compressor_factory,
            temp_dir=temp_dir,
        )

        if temp_output is None:
            return None, None

        try:
            img = Image.open(temp_output)
            return img, temp_output
        except Exception as e:
            logger.debug("Nelze otevřít dekomprimovaný temp soubor %s: %s", temp_output, e)
            try:
                temp_output.unlink()
            except OSError:
                pass
            return None, None

    @staticmethod
    def _decompress_via_factory(
        compressed_path: Path,
        compressor_factory: "CompressorFactoryType",
        temp_dir: Path,
    ) -> Optional[Path]:
        """
        Najde kompresor odpovídající příponě souboru a dekomprimuje do temp souboru.

        Returns:
            Cesta k dekomprimovanému temp souboru, nebo None při selhání.
        """
        extension = compressed_path.suffix.lower()

        for comp_name in compressor_factory.list_available():
            try:
                compressor = compressor_factory.create(comp_name)
                if compressor.extension.lower() != extension:
                    continue

                temp_output = temp_dir / f"_verify_{compressed_path.stem}.png"
                compressor.decompress(compressed_path, temp_output)
                logger.debug("Dekomprimováno přes %s → %s", comp_name, temp_output)
                return temp_output

            except Exception as e:
                logger.debug("Kompresor %s selhal při dekomprimaci %s: %s",
                             comp_name, compressed_path, e)
                continue

        logger.warning("Žádný kompresor nenalezen pro příponu %s", extension)
        return None

    @staticmethod
    def _compare(
        img_original: Image.Image,
        img_compressed: Image.Image,
    ) -> VerificationResult:
        """Provede pixel-level srovnání dvou PIL obrazů."""

        # Sjednoť barevný mód
        if img_original.mode != img_compressed.mode:
            img_compressed = img_compressed.convert(img_original.mode)

        # Ověř rozměry
        if img_original.size != img_compressed.size:
            total_pixels = img_original.size[0] * img_original.size[1]
            return VerificationResult(
                is_lossless=False,
                max_difference=float("inf"),
                different_pixels=total_pixels,
                total_pixels=total_pixels,
                error_message=(f"Nesoulad rozměrů: "
                               f"{img_original.size} vs {img_compressed.size}"),
            )

        arr_orig = np.array(img_original, dtype=np.float32)
        arr_comp = np.array(img_compressed, dtype=np.float32)

        # Normalizuj dimenze (grayscale 2D vs RGB 3D)
        if arr_orig.ndim == 2:
            arr_orig = arr_orig[:, :, np.newaxis]
        if arr_comp.ndim == 2:
            arr_comp = arr_comp[:, :, np.newaxis]

        diff = np.abs(arr_orig - arr_comp)
        max_diff = float(np.max(diff))
        different_pixels = int(np.sum(np.any(diff > 0, axis=-1)))
        total_pixels = arr_orig.shape[0] * arr_orig.shape[1]

        return VerificationResult(
            is_lossless=(max_diff == 0.0),
            max_difference=max_diff,
            different_pixels=different_pixels,
            total_pixels=total_pixels,
        )