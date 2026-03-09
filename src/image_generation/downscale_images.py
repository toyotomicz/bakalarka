"""
Downscale images (>12 MPx) to 1, 2, 4, 8 and 12 MPx.

Input:
    image_datasets/images_by_megapixels_dataset/high_res/

Output:
    image_datasets/images_by_megapixels_dataset/<N>_mpx/

Usage:
    python downscale_to_sizes.py

Requirements:
    pip install Pillow
"""

from pathlib import Path
from PIL import Image, ImageOps
import math
import time

# ───────────────────────── Configuration ─────────────────────────

INPUT_DIR = Path("image_datasets/images_by_megapixels_dataset/original")
OUTPUT_BASE = Path("image_datasets/images_by_megapixels_dataset")

TARGET_MPX = [1, 2, 4, 8, 12]

SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

PNG_COMPRESS = 0
JPEG_QUALITY = 95
SAVE_AS_PNG = True

# ───────────────────────── Helpers ─────────────────────────

def mpx_label(w: int, h: int) -> str:
    return f"{(w*h)/1_000_000:.2f} MPx"


def compute_resolution(mpx: int, aspect: float):
    pixels = mpx * 1_000_000

    w = math.sqrt(pixels * aspect)
    h = w / aspect

    return int(round(w)), int(round(h))


def save_image(img: Image.Image, path: Path):

    path.parent.mkdir(parents=True, exist_ok=True)

    if SAVE_AS_PNG:
        path = path.with_suffix(".png")
        img.save(path, format="PNG", compress_level=PNG_COMPRESS)

    else:
        ext = path.suffix.lower()

        if ext in {".jpg", ".jpeg"}:
            img.save(path, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
        else:
            path = path.with_suffix(".png")
            img.save(path, format="PNG", compress_level=PNG_COMPRESS)

    return path


# ───────────────────────── Main ─────────────────────────

def downscale_dataset():

    print("=" * 60)
    print("Dynamic MPx Dataset Generator (1,2,4,8,12)")
    print("=" * 60)

    print("Input :", INPUT_DIR.resolve())
    print("Output:", OUTPUT_BASE.resolve())
    print()

    if not INPUT_DIR.exists():
        print("ERROR: input directory not found")
        return

    photos = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in SUPPORTED)

    if not photos:
        print("ERROR: no images found")
        return

    print("Found", len(photos), "source images\n")

    total_written = 0
    start = time.perf_counter()

    for photo in photos:

        print("Processing:", photo.name)

        try:
            with Image.open(photo) as img:
                src = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as e:
            print("  skipped:", e)
            continue

        sw, sh = src.size
        aspect = sw / sh

        print(f"  source: {sw}×{sh} ({mpx_label(sw,sh)})")

        for mpx in TARGET_MPX:

            w, h = compute_resolution(mpx, aspect)

            if w >= sw or h >= sh:
                print(f"  skip {mpx} MPx (larger than source)")
                continue

            out_dir = OUTPUT_BASE / f"{mpx}_mpx"
            out_path = out_dir / photo.name

            resized = src.resize((w, h), Image.Resampling.LANCZOS)

            saved = save_image(resized, out_path)

            size_kb = saved.stat().st_size / 1024

            print(
                f"  → {mpx:>2} MPx  {w}×{h}  "
                f"({mpx_label(w,h)})  {size_kb:,.0f} KB"
            )

            total_written += 1

        print()

    elapsed = time.perf_counter() - start

    print("-" * 60)
    print("Finished:", total_written, "images written")
    print("Time:", round(elapsed,1), "s")
    print("=" * 60)


if __name__ == "__main__":
    downscale_dataset()