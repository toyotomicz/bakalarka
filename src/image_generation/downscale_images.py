"""
Downscale 24 MPx (6000×4000) photos to multiple target resolutions.

Input:  datasets/images_by_megapixels_dataset/24_mpx/
Output: datasets/images_by_megapixels_dataset/<N>_mpx/

Usage:
    python downscale_to_sizes.py

Requirements:
    pip install Pillow
"""

from pathlib import Path
from PIL import Image
import time

# ─────────────────────────── Configuration ──────────────────────────────────

INPUT_DIR  = Path("image_datasets/images_by_megapixels_dataset/24_mpx")
OUTPUT_BASE = Path("image_datasets/images_by_megapixels_dataset")

# Target sizes: (width, height, folder_name)
# All keep the 3:2 aspect ratio of 6000×4000
TARGET_SIZES = [
    (4243, 2828, "12_mpx"),   # ~12 MPx
    (3464, 2309, "6_mpx"),    # ~6 MPx
    (2449, 1633, "3_mpx"),    # ~3 MPx
    (1224, 816,  "1_mpx"),    # ~1 MPx
    (800,  533, "0.4_mpx"),   # ~0.4 MPx (thumbnail)
]

SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
PNG_COMPRESS = 0      # lossless, no zlib compression
JPEG_QUALITY = 95     # used only if source is JPEG and you want JPEG output
SAVE_AS_PNG  = True   # True → always save PNG;  False → keep original format

# ─────────────────────────── Helpers ────────────────────────────────────────

def mpx_label(w: int, h: int) -> str:
    return f"{w * h / 1_000_000:.1f} MPx"


def save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_AS_PNG:
        img.save(path.with_suffix(".png"), format="PNG", compress_level=PNG_COMPRESS)
    else:
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            img.save(path, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
        else:
            img.save(path.with_suffix(".png"), format="PNG", compress_level=PNG_COMPRESS)


# ─────────────────────────── Main ───────────────────────────────────────────

def downscale_dataset() -> None:
    print("=" * 64)
    print("  24 MPx → Multi-Resolution Downscaler")
    print("=" * 64)
    print(f"  Input  : {INPUT_DIR.resolve()}")
    print(f"  Output : {OUTPUT_BASE.resolve()}")
    print(f"  Targets: {len(TARGET_SIZES)} sizes")
    print(f"  Output format: {'PNG (lossless)' if SAVE_AS_PNG else 'original format'}")
    print("-" * 64)

    if not INPUT_DIR.exists():
        print(f"  [ERROR] Input directory not found: {INPUT_DIR}")
        return

    photos = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in SUPPORTED)
    if not photos:
        print(f"  [ERROR] No supported images found in {INPUT_DIR}")
        return

    print(f"  Found {len(photos)} source image(s).\n")

    total_saved = 0
    t0 = time.perf_counter()

    for photo_path in photos:
        print(f"  ── {photo_path.name}")
        try:
            src = Image.open(photo_path).convert("RGB")
        except Exception as exc:
            print(f"     [WARN] Could not open: {exc}")
            continue

        src_w, src_h = src.size
        print(f"     Source: {src_w}×{src_h}  ({mpx_label(src_w, src_h)})")

        for (tw, th, folder) in TARGET_SIZES:
            if tw >= src_w or th >= src_h:
                print(f"     Skip  {folder:>8}  ({tw}×{th}) – not smaller than source")
                continue

            out_dir  = OUTPUT_BASE / folder
            out_path = out_dir / photo_path.name

            resized = src.resize((tw, th), Image.LANCZOS)
            save_image(resized, out_path)

            # report actual saved path (extension may have changed)
            actual = out_path.with_suffix(".png") if SAVE_AS_PNG else out_path
            kb = actual.stat().st_size / 1024
            print(f"     Saved {folder:>8}  {tw}×{th}  ({mpx_label(tw, th)})  →  {kb:,.0f} KB")
            total_saved += 1

        print()

    elapsed = time.perf_counter() - t0
    print("-" * 64)
    print(f"  Done!  {total_saved} files written in {elapsed:.1f}s")
    print("=" * 64)


if __name__ == "__main__":
    downscale_dataset()