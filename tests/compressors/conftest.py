"""
Sdílené fixtures pro unit testy kompresorů.
"""

import sys
import types
from pathlib import Path

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Fixture: dočasný PNG soubor
# ---------------------------------------------------------------------------

@pytest.fixture()
def rgb_png(tmp_path) -> Path:
    """Malý 4×4 RGB PNG soubor."""
    p = tmp_path / "rgb.png"
    Image.new("RGB", (4, 4), color=(100, 150, 200)).save(p, format="PNG")
    return p


@pytest.fixture()
def rgba_png(tmp_path) -> Path:
    """Malý 4×4 RGBA PNG soubor."""
    p = tmp_path / "rgba.png"
    Image.new("RGBA", (4, 4), color=(100, 150, 200, 128)).save(p, format="PNG")
    return p


@pytest.fixture()
def grayscale_png(tmp_path) -> Path:
    """Malý 4×4 grayscale PNG soubor."""
    p = tmp_path / "gray.png"
    Image.new("L", (4, 4), color=128).save(p, format="PNG")
    return p
