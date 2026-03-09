"""
Synthetic Image Dataset Generator for Benchmarking Image Compression Algorithms.

Generates a mix of real photos and synthetic graphics (text, flat colour areas, charts).
Each image is approximately 50% photograph and 50% synthetic content.

"""

import random
import io
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────── Configuration ──────────────────────────────────

SEED         = 6
INPUT_DIR    = Path("image_datasets/kodak_dataset")
OUTPUT_DIR   = Path("image_datasets/generated_dataset")
IMAGE_SIZES  = [(1024, 1024)]
NUM_IMAGES   = 20
PNG_COMPRESS = 0   # 0 = no zlib compression – true lossless pixel data

# ─────────────────────────── Text content pools ─────────────────────────────

LOREM_PARAGRAPHS = [
    (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, "
        "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo."
    ),
    (
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore "
        "eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, "
        "sunt in culpa qui officia deserunt mollit anim id est laborum."
    ),
    (
        "Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, "
        "turpis molestie dictum semper, diam lectus fermentum ipsum. Pellentesque "
        "habitant morbi tristique senectus et netus et malesuada fames ac turpis."
    ),
    (
        "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere "
        "cubilia curae. Proin pharetra nonummy pede. Mauris et orci. Aenean nec lorem. "
        "In porttitor. Donec laoreet nonummy augue."
    ),
    (
        "Suspendisse dui purus, scelerisque at, vulputate vitae, pretium mattis, nunc. "
        "Mauris eget neque at sem venenatis eleifend. Ut nonummy. Fusce aliquet pede "
        "non pede. Suspendisse dapibus lorem pellentesque magna."
    ),
    (
        "Integer nulla. Donec blandit feugiat ligula. Donec hendrerit, felis et "
        "imperdiet euismod, purus ipsum pretium metus, in lacinia nulla nisl eget "
        "sapien. Donec ut est in lectus consequat consequat."
    ),
    (
        "Etiam eget dui. Aliquam erat volutpat. Nam dui mi, tincidunt quis, accumsan "
        "porttitor, facilisis luctus, metus. Phasellus ultrices nulla quis nibh. "
        "Quisque a lectus. Donec consectetuer ligula vulputate sem tristique cursus."
    ),
    (
        "Nam nulla quam, gravida non, commodo a, sodales sit amet, nisi. Nullam "
        "pede justo, fringilla vel, aliquet nec, vulputate eget, arcu. In enim justo, "
        "rhoncus ut, imperdiet a, venenatis vitae, justo."
    ),
    (
        "Nullam dictum felis eu pede mollis pretium. Integer tincidunt. Cras dapibus. "
        "Vivamus elementum semper nisi. Aenean vulputate eleifend tellus. Aenean leo "
        "ligula, porttitor eu, consequat vitae, eleifend ac, enim."
    ),
    (
        "Aliquam lorem ante, dapibus in, viverra quis, feugiat a, tellus. Phasellus "
        "viverra nulla ut metus varius laoreet. Quisque rutrum. Aenean imperdiet. "
        "Etiam ultricies nisi vel augue. Curabitur ullamcorper ultricies nisi."
    ),
]

HEADINGS = [
    "Executive Summary", "Key Findings", "Data Overview",
    "Analysis Results", "Performance Report", "Quarterly Update",
    "Technical Specifications", "Market Insights", "Project Status",
    "Methodology", "Introduction", "Conclusions & Next Steps",
    "Background", "Objectives", "Recommendations",
]

SUBHEADINGS = [
    "Overview", "Details", "Summary", "Notes", "Remarks",
    "Highlights", "Context", "Rationale", "Scope", "Deliverables",
    "Findings", "Discussion", "Approach", "Results", "Impact",
]

COLORS_FLAT = [
    (220,  53,  69),   # red
    ( 40, 167,  69),   # green
    (  0, 123, 255),   # blue
    (255, 193,   7),   # yellow
    ( 52,  58,  64),   # dark grey
    ( 23, 162, 184),   # cyan
    (111,  66, 193),   # purple
    (253, 126,  20),   # orange
    ( 32, 201, 151),   # teal
    (214,  51, 132),   # pink
    (248, 249, 250),   # off-white
    ( 33,  37,  41),   # near-black
    (173, 181, 189),   # mid-grey
    (  8,  79, 117),   # navy
]

# ─────────────────────────── Font helpers ───────────────────────────────────

def try_get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def try_get_regular_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

# ─────────────────────────── Photo helpers ──────────────────────────────────

def load_input_photos(directory: Path) -> list:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    photos = []
    if not directory.exists():
        print(f"  [WARN] '{directory}' not found – using generated gradients.")
        return photos
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in supported:
            try:
                img = Image.open(path).convert("RGB")
                photos.append(img)
                print(f"  Loaded: {path.name}  ({img.width}×{img.height})")
            except Exception as exc:
                print(f"  [WARN] Could not load {path.name}: {exc}")
    return photos


def make_gradient_photo(width: int, height: int, rng: random.Random) -> Image.Image:
    """Smooth diagonal gradient + noise as photo substitute."""
    c1 = np.array([rng.randint(20, 160), rng.randint(20, 160), rng.randint(20, 160)], dtype=np.float32)
    c2 = np.array([rng.randint(100, 255), rng.randint(100, 255), rng.randint(100, 255)], dtype=np.float32)
    xs = np.linspace(0, 1, width)
    ys = np.linspace(0, 1, height)
    xg, yg = np.meshgrid(xs, ys)
    t = np.clip((xg + yg) / 2, 0, 1)[..., np.newaxis]
    arr = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    noise = rng.randint(4, 14)
    arr = np.clip(
        arr.astype(np.int16) + np.random.randint(-noise, noise + 1, arr.shape),
        0, 255
    ).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def get_photo(photos: list, width: int, height: int, rng: random.Random) -> Image.Image:
    """Return a centre-cropped and resized photo (or gradient fallback)."""
    src = rng.choice(photos).copy() if photos else make_gradient_photo(width * 2, height * 2, rng)
    src_w, src_h = src.size
    target_ratio = width / height
    src_ratio    = src_w / src_h
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left  = (src_w - new_w) // 2
        src   = src.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top   = (src_h - new_h) // 2
        src   = src.crop((0, top, src_w, top + new_h))
    return src.resize((width, height), Image.LANCZOS)

# ─────────────────────────── Flat-colour shapes ─────────────────────────────

def draw_shapes(canvas: Image.Image, rng: random.Random, count: int = 3) -> None:
    """
    Draw a moderate number of solid opaque rectangles / circles.
    Plain RGB ImageDraw – no RGBA, no transparency whatsoever.
    """
    draw = ImageDraw.Draw(canvas)   # RGB mode only
    w, h = canvas.size
    for _ in range(count):
        color = rng.choice(COLORS_FLAT)           # plain (R, G, B)
        shape = rng.choice(["rect", "rect", "circle"])   # slightly more rects
        x0 = rng.randint(0, w - 1)
        y0 = rng.randint(0, h - 1)
        x1 = min(x0 + rng.randint(w // 14, w // 6), w - 1)
        y1 = min(y0 + rng.randint(h // 14, h // 6), h - 1)
        if shape == "rect":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=color)

# ─────────────────────────── Text block ─────────────────────────────────────

def draw_text_block(canvas: Image.Image, rng: random.Random,
                    x: int, y: int, block_w: int, block_h: int) -> None:
    """
    Render a rich text block: heading + subheadings + body paragraphs.
    Solid opaque background, no transparency.
    """
    draw = ImageDraw.Draw(canvas)

    bg = rng.choice(COLORS_FLAT)
    draw.rectangle([x, y, x + block_w, y + block_h], fill=bg)

    lum      = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    ink      = (10,  10,  10)  if lum > 160 else (245, 245, 245)
    ink_dim  = (60,  60,  60)  if lum > 160 else (190, 190, 190)
    ink_rule = (130, 130, 130) if lum > 160 else (110, 110, 110)

    scale    = block_w / 512
    title_sz = max(18, int(42 * scale))
    sub_sz   = max(13, int(22 * scale))
    body_sz  = max(10, int(17 * scale))
    pad      = max(10, int(20 * scale))
    ln_gap   = max(3,  int( 5 * scale))
    sec_gap  = max(8,  int(14 * scale))

    title_font = try_get_font(title_sz)
    sub_font   = try_get_font(sub_sz)
    body_font  = try_get_regular_font(body_sz)

    cy = y + pad

    # ── Main heading ──────────────────────────────────────────────────────
    draw.text((x + pad, cy), rng.choice(HEADINGS), fill=ink, font=title_font)
    cy += title_sz + ln_gap
    draw.line([(x + pad, cy), (x + block_w - pad, cy)],
              fill=ink_rule, width=max(1, int(2 * scale)))
    cy += ln_gap + sec_gap

    # ── Body sections ─────────────────────────────────────────────────────
    cpl   = max(25, int(block_w / (body_sz * 0.52)))   # chars per line
    paras = rng.sample(LOREM_PARAGRAPHS,
                       min(len(LOREM_PARAGRAPHS), rng.randint(5, 8)))

    for i, para in enumerate(paras):
        if cy + sub_sz >= y + block_h - pad:
            break

        # subheading before each section
        draw.text((x + pad, cy), rng.choice(SUBHEADINGS).upper(),
                  fill=ink_dim, font=sub_font)
        cy += sub_sz + ln_gap

        for line in textwrap.wrap(para, width=cpl):
            if cy + body_sz >= y + block_h - pad:
                break
            draw.text((x + pad, cy), line, fill=ink, font=body_font)
            cy += body_sz + ln_gap
        cy += sec_gap

# ─────────────────────────── Chart ──────────────────────────────────────────

def make_chart(width: int, height: int, rng: random.Random) -> Image.Image:
    """Render a solid-background matplotlib chart as an RGB Pillow image."""
    dpi   = 96
    fig_w = width  / dpi
    fig_h = height / dpi

    chart_type = rng.choice(["bar", "line"])
    n          = rng.randint(5, 10)
    labels     = list("ABCDEFGHIJ"[:n])
    v1         = [rng.uniform(10, 100) for _ in range(n)]
    v2         = [rng.uniform(10, 100) for _ in range(n)]
    palette    = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in COLORS_FLAT]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    if chart_type == "bar":
        xs = np.arange(n)
        ax.bar(xs - 0.2, v1, 0.4, label="Series A", color=rng.choice(palette))
        ax.bar(xs + 0.2, v2, 0.4, label="Series B", color=rng.choice(palette))
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(rng.choice(["Monthly Sales", "Category Breakdown",
                                  "Revenue by Segment", "Quarterly Results"]),
                     fontsize=11, fontweight="bold")
    else:
        xs = list(range(n))
        ax.plot(xs, v1, marker="o", linewidth=2, label="Series A", color=rng.choice(palette))
        ax.plot(xs, v2, marker="s", linewidth=2, label="Series B", color=rng.choice(palette))
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(rng.choice(["Trend Analysis", "Performance Over Time",
                                  "Growth Metrics", "Year-on-Year"]),
                     fontsize=11, fontweight="bold")

    ax.set_ylabel("Value")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((width, height), Image.LANCZOS)

# ─────────────────────────── Layout generators ──────────────────────────────

def layout_side_by_side(photos, size, rng):
    """Left = photo  |  Right = chart."""
    W, H = size
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    half = W // 2
    canvas.paste(get_photo(photos, half, H, rng), (0, 0))
    canvas.paste(make_chart(half, H, rng), (half, 0))
    draw_shapes(canvas, rng, count=rng.randint(2, 4))
    return canvas


def layout_top_bottom(photos, size, rng):
    """Top = photo  |  Bottom = large text block."""
    W, H = size
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    split = int(H * rng.uniform(0.40, 0.55))
    canvas.paste(get_photo(photos, W, split, rng), (0, 0))
    draw_text_block(canvas, rng, x=0, y=split, block_w=W, block_h=H - split)
    draw_shapes(canvas, rng, count=rng.randint(2, 4))
    return canvas


def layout_photo_with_sidebar(photos, size, rng):
    """Wide photo panel left, tall text column right."""
    W, H = size
    split = int(W * rng.uniform(0.55, 0.65))
    canvas = Image.new("RGB", (W, H), (240, 240, 240))
    canvas.paste(get_photo(photos, split, H, rng), (0, 0))
    draw_text_block(canvas, rng, x=split, y=0, block_w=W - split, block_h=H)
    draw_shapes(canvas, rng, count=rng.randint(2, 4))
    return canvas


def layout_photo_plus_chart(photos, size, rng):
    """Photo | chart | text in three equal columns."""
    W, H = size
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    third = W // 3
    canvas.paste(get_photo(photos, third, H, rng), (0, 0))
    canvas.paste(make_chart(third, H, rng), (third, 0))
    draw_text_block(canvas, rng, x=2 * third, y=0, block_w=W - 2 * third, block_h=H)
    draw_shapes(canvas, rng, count=rng.randint(2, 4))
    return canvas


def layout_grid(photos, size, rng):
    """2×2 grid: photo / chart / text block / photo."""
    W, H = size
    hw, hh = W // 2, H // 2
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(get_photo(photos, hw, hh, rng), (0,  0))
    canvas.paste(make_chart(hw, hh, rng),         (hw, 0))
    draw_text_block(canvas, rng, x=0, y=hh, block_w=hw, block_h=hh)
    canvas.paste(get_photo(photos, hw, hh, rng), (hw, hh))
    draw_shapes(canvas, rng, count=rng.randint(3, 5))
    return canvas


def layout_text_heavy(photos, size, rng):
    """Narrow photo strip, wide text area, small chart inset at bottom."""
    W, H = size
    canvas  = Image.new("RGB", (W, H), (250, 250, 250))
    strip_w = int(W * rng.uniform(0.28, 0.38))
    canvas.paste(get_photo(photos, strip_w, H, rng), (0, 0))
    text_w  = W - strip_w
    chart_h = int(H * 0.28)
    text_h  = H - chart_h
    draw_text_block(canvas, rng, x=strip_w, y=0, block_w=text_w, block_h=text_h)
    canvas.paste(make_chart(text_w, chart_h, rng), (strip_w, text_h))
    draw_shapes(canvas, rng, count=rng.randint(2, 4))
    return canvas


def layout_full_text(photos, size, rng):
    """
    Two text columns with a solid colour header bar spanning the top.
    No photo panel – maximises text + flat-colour area.
    """
    W, H     = size
    header_h = int(H * 0.10)
    col_w    = W // 2

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    hdr_color = rng.choice(COLORS_FLAT)
    draw.rectangle([0, 0, W, header_h], fill=hdr_color)

    lum  = 0.299 * hdr_color[0] + 0.587 * hdr_color[1] + 0.114 * hdr_color[2]
    ink  = (10, 10, 10) if lum > 160 else (245, 245, 245)
    sc   = W / 1024
    hfnt = try_get_font(max(20, int(44 * sc)))
    draw.text((int(24 * sc), int(header_h * 0.15)), rng.choice(HEADINGS),
              fill=ink, font=hfnt)

    body_h = H - header_h
    draw_text_block(canvas, rng, x=0,     y=header_h, block_w=col_w,     block_h=body_h)
    draw_text_block(canvas, rng, x=col_w, y=header_h, block_w=W - col_w, block_h=body_h)
    draw_shapes(canvas, rng, count=rng.randint(2, 3))
    return canvas


LAYOUTS = [
    layout_side_by_side,
    layout_top_bottom,
    layout_photo_with_sidebar,
    layout_photo_plus_chart,
    layout_grid,
    layout_text_heavy,
    layout_full_text,
]

# ─────────────────────────── Main ───────────────────────────────────────────

def generate_dataset(
    n_images: int = NUM_IMAGES,
    sizes: list  = IMAGE_SIZES,
    seed: int    = SEED,
) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  Synthetic Image Dataset Generator")
    print("=" * 62)
    print(f"  Input  directory : {INPUT_DIR}")
    print(f"  Output directory : {OUTPUT_DIR.resolve()}")
    print(f"  Images to create : {n_images}")
    print(f"  Sizes            : {sizes}")
    print(f"  PNG compress lvl : {PNG_COMPRESS}  (0 = none, raw pixels)")
    print(f"  Random seed      : {seed}")
    print("-" * 62)

    photos = load_input_photos(INPUT_DIR)
    print(f"  Loaded {len(photos)} photo(s)." if photos
          else "  No input photos found – using procedural gradients.")
    print("-" * 62)

    for idx in range(1, n_images + 1):
        size      = rng.choice(sizes)
        layout_fn = rng.choice(LAYOUTS)
        name      = layout_fn.__name__.replace("layout_", "")

        print(f"  [{idx:02d}/{n_images}]  {size[0]}×{size[1]}  {name} …",
              end=" ", flush=True)

        # Convert to RGB guarantees no alpha channel ever reaches the encoder
        img = layout_fn(photos, size, rng).convert("RGB")

        out_path = OUTPUT_DIR / f"image_{idx:03d}.png"
        img.save(out_path, format="PNG", compress_level=PNG_COMPRESS)

        kb = out_path.stat().st_size / 1024
        print(f"saved  ({kb:,.0f} KB)")

    print("-" * 62)
    total_mb = sum(p.stat().st_size for p in OUTPUT_DIR.glob("*.png")) / (1024 ** 2)
    print(f"  Done!  {n_images} images saved to '{OUTPUT_DIR}/'  ({total_mb:.1f} MB total)")
    print("=" * 62)


if __name__ == "__main__":
    generate_dataset()