#!/usr/bin/env python3
"""
Generátor sady PNG obrázků - novinový/letákový styl
Velikosti: 8, 4, 2 Mpx
5 obrázků v každé složce
"""

import os
import math
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "generated")

SIZES_MPX = [8, 4, 2]

def mpx_to_dims(mpx, ratio=(4, 3)):
    total = mpx * 1_000_000
    w = int(math.sqrt(total * ratio[0] / ratio[1]))
    h = int(total / w)
    return w, h

PALETTES = [
    {"bg": (245, 235, 215), "text": (30, 25, 20), "accent": (180, 60, 40), "secondary": (100, 130, 80)},
    {"bg": (230, 235, 245), "text": (20, 30, 50), "accent": (40, 100, 180), "secondary": (200, 160, 60)},
    {"bg": (255, 248, 240), "text": (40, 30, 30), "accent": (200, 30, 50), "secondary": (60, 150, 100)},
    {"bg": (240, 248, 235), "text": (25, 50, 25), "accent": (60, 140, 60), "secondary": (180, 120, 40)},
    {"bg": (35, 35, 45),   "text": (230, 225, 215), "accent": (255, 180, 40), "secondary": (100, 180, 220)},
]

HEADLINES = [
    "Velká změna přichází do našeho města",
    "Nová éra technologií: Co nás čeká?",
    "Rekordní úroda letošního roku",
    "Festival kultury otevírá brány",
    "Věda a příroda: Objev desetiletí",
    "Ekonomika roste navzdory krizi",
    "Místní trh slaví 100 let existence",
    "Architektura budoucnosti dnes",
]

SUBHEADLINES = [
    "Odborníci varují i radují se zároveň – situace si žádá okamžitou pozornost",
    "Reportáž přímo z místa událostí přináší nový pohled na celou kauzu",
    "Exkluzivní rozhovor s klíčovými účastníky odhaluje dosud neznámé skutečnosti",
    "Analýza situace v číslech a faktech ukazuje znepokojivé trendy",
    "Co říkají místní obyvatelé o změnách, které ovlivňují každý den?",
]

# Rich lorem ipsum body paragraphs
BODY_TEXTS = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",

    "Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Vestibulum tortor quam, feugiat vitae, ultricies eget, tempor sit amet, ante. Donec eu libero sit amet quam egestas semper. Aenean ultricies mi vitae est. Mauris placerat eleifend leo.",

    "Quisque sit amet est et sapien ullamcorper pharetra. Vestibulum erat wisi, condimentum sed, commodo vitae, ornare sit amet, wisi. Aenean fermentum, elit eget tincidunt condimentum, eros ipsum rutrum orci, sagittis tempus lacus enim ac dui. Donec non enim in turpis pulvinar facilisis.",

    "Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis molestie dictum ultricies, arcus lorem imperdiet orci, vel interdum erat urna non ex. Integer vulputate sem a nibh rutrum consequat. Nam congue semper tellus. Sed erat dolor, dapibus sit amet, venenatis ornare, ultrices ut, nisi.",

    "Aliquam erat volutpat. Nam dui mi, tincidunt quis, accumsan porttitor, facilisis luctus, metus. Phasellus ultrices nulla quis nibh. Quisque a lectus. Donec consectetuer ligula vulputate sem tristique cursus. Nam nulla quam, gravida non, commodo a, sodales sit amet, nisi.",

    "Nullam eu ante vel est convallis dignissim. Fusce suscipit, wisi nec facilisis facilisis, est dui fermentum leo, quis tempor ligula erat quis odio. Nunc porta vulputate tellus. Nunc rutrum turpis sed pede. Sed bibendum. Aliquam posuere. Nunc aliquet, augue nec adipiscing interdum, lacus tellus malesuada massa, quis varius mi purus non odio.",

    "Proin in tellus sit amet nibh dignissim sagittis. Vivamus in augue id justo porta ornare. Fusce sagittis, libero non molestie mollis, magna orci ultrices dolor, at vulputate neque nulla lacinia eros. Sed id ligula quis est convallis tempor. Integer porta accumsan mauris. Etiam id lacus interdum elit tincidunt interdum.",

    "Donec posuere augue in quam. Etiam vel tortor sodales tellus ultricies commodo. Suspendisse potenti. Aenean orci lacus, blandit eget, lobortis at, luctus vel, lorem. Sed convallis, quam non euismod iaculis, nibh augue convallis ante, ac fermentum est velit id nisi. Cras a nunc magna.",

    "Mauris blandit aliquet elit, eget tincidunt nibh pulvinar a. Cras nec ante pellentesque, volutpat diam ut, euismod quam. Vivamus magna. Cras in mi at felis aliquet congue. Ut a est eget ligula molestie gravida. Curabitur massa. Donec eleifend libero at lobortis mollis.",

    "Integer in mauris eu nibh euismod gravida dui. Phasellus vitae lacus sit amet sapien vulputate dapibus. Quisque diam lorem, interdum vitae, dapibus ac, scelerisque vitae, pede. Donec luctus, erat id tristique placerat, ipsum dui faucibus lorem, non pharetra est turpis vitae tortor.",

    "Sed lacinia, urna non tincidunt mattis, tortor neque adipiscing diam, a cursus ipsum ante quis turpis. Nulla facilisi. Ut fringilla. Suspendisse potenti. Nunc feugiat mi a tellus consequat imperdiet. Vestibulum sapien. Proin quam. Etiam ultrices. Suspendisse in justo eu magna luctus suscipit.",

    "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Proin vel ante a orci tempus eleifend ut et magna. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus luctus urna sed urna ultricies ac tempor dui sagittis. In condimentum facilisis porta.",
]

CAPTIONS = [
    "Pohled na centrum města z ptačí perspektivy během jarního odpoledne",
    "Účastníci akce v plném nasazení, foto: archiv redakce",
    "Místní krajina v proměnách ročních období, pohled od jihu",
    "Detail fascinujícího přírodního jevu zachycený v ranních hodinách",
    "Pracovní tým při dokumentaci průběhu historické události",
    "Výhled na panorama v ranním světle po vydatném dešti",
]

TAGS = ["AKTUÁLNĚ", "EXKLUZIVNĚ", "REPORTÁŽ", "ANALÝZA", "TÉMA DNE", "ROZHOVOR", "KOMENTÁŘ", "ZPRÁVY"]

PULL_QUOTES = [
    "„Toto je moment, který změní vše, co jsme dosud znali.",
    "„Nikdy v historii jsme neviděli takovou rychlost změn.",
    "„Výsledky překonaly i ty nejodvážnější předpoklady.",
    "„Komunita se semkla způsobem, který nikoho nepřekvapil.",
    "„Data hovoří jasně – jsme na prahu nové epochy.",
]

SECTION_HEADERS = [
    "Pozadí celé kauzy",
    "Co říkají odborníci",
    "Reakce veřejnosti",
    "Historický kontext",
    "Výhled do budoucna",
    "Klíčová fakta",
    "Analýza situace",
]


def get_font(size):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                pass
    return ImageFont.load_default()

def get_font_regular(size):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                pass
    return ImageFont.load_default()


def draw_decorative_image(draw, x, y, w, h, palette, seed):
    rng = random.Random(seed)
    style = rng.choice(["landscape", "abstract", "grid", "circles"])

    if style == "landscape":
        sky_color = palette["accent"]
        ground_color = palette["secondary"]
        for i in range(h):
            t = i / h
            r = int(sky_color[0] * (1-t) + ground_color[0] * t)
            g = int(sky_color[1] * (1-t) + ground_color[1] * t)
            b = int(sky_color[2] * (1-t) + ground_color[2] * t)
            draw.line([(x, y+i), (x+w, y+i)], fill=(r, g, b))
        sx = x + int(w * 0.75)
        sy = y + int(h * 0.2)
        sr = int(min(w, h) * 0.12)
        draw.ellipse([sx-sr, sy-sr, sx+sr, sy+sr],
                     fill=palette["text"] if palette["bg"][0] < 100 else (255, 220, 80))
        for _ in range(3):
            hx = x + rng.randint(0, w)
            hy = y + int(h * 0.5) + rng.randint(-h//6, h//6)
            hr = int(w * rng.uniform(0.2, 0.5))
            col = tuple(min(255, max(0, c + rng.randint(-30, 30))) for c in ground_color)
            draw.ellipse([hx-hr, hy-hr//2, hx+hr, hy+hr//2], fill=col)

    elif style == "abstract":
        draw.rectangle([x, y, x+w, y+h], fill=palette["bg"])
        for _ in range(rng.randint(8, 18)):
            cx = x + rng.randint(0, w)
            cy = y + rng.randint(0, h)
            cr = rng.randint(int(min(w,h)*0.05), int(min(w,h)*0.35))
            col = rng.choice([palette["accent"], palette["secondary"], palette["text"]])
            draw.ellipse([cx-cr, cy-cr, cx+cr, cy+cr], fill=col)

    elif style == "grid":
        cols = rng.randint(4, 10)
        rows = rng.randint(4, 8)
        cw = w // cols
        ch = h // rows
        for ci in range(cols):
            for ri in range(rows):
                col = rng.choice([palette["accent"], palette["secondary"], palette["bg"], palette["text"]])
                draw.rectangle([x+ci*cw, y+ri*ch, x+(ci+1)*cw-1, y+(ri+1)*ch-1], fill=col)

    else:
        draw.rectangle([x, y, x+w, y+h], fill=palette["secondary"])
        for _ in range(rng.randint(5, 12)):
            cx = x + rng.randint(0, w)
            cy = y + rng.randint(0, h)
            cr = rng.randint(int(min(w,h)*0.1), int(min(w,h)*0.4))
            col = rng.choice([palette["accent"], palette["bg"]])
            draw.ellipse([cx-cr, cy-cr, cx+cr, cy+cr], outline=col, width=max(2, cr//8))


def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_text_block(draw, x, y, text, font, max_width, color, max_y=None):
    """Draw wrapped text, return new y position."""
    lines = wrap_text(text, font, max_width, draw)
    for line in lines:
        if max_y and y + 20 > max_y:
            break
        draw.text((x, y), line, font=font, fill=color)
        lb = draw.textbbox((0, 0), line, font=font)
        y += (lb[3] - lb[1]) + 2
    return y


def draw_pull_quote(draw, x, y, w, text, font_bold, font_reg, palette, scale):
    """Draw a styled pull quote box, return new y."""
    pad = max(10, int(18 * scale))
    box_h = max(60, int(90 * scale))
    # background tint
    bg = tuple(min(255, c + 20) for c in palette["bg"]) if palette["bg"][0] > 100 else tuple(min(255, c + 15) for c in palette["bg"])
    draw.rectangle([x, y, x+w, y+box_h], fill=bg)
    draw.line([(x + pad//2, y + pad//3), (x + pad//2, y + box_h - pad//3)],
              fill=palette["accent"], width=max(4, int(6*scale)))
    # truncate quote to fit
    lines = wrap_text(text, font_bold, w - pad*2, draw)
    ty = y + pad // 2
    for line in lines[:3]:
        draw.text((x + pad + max(8, int(12*scale)), ty), line, font=font_bold, fill=palette["accent"])
        lb = draw.textbbox((0, 0), line, font=font_bold)
        ty += (lb[3] - lb[1]) + 2
    return y + box_h + max(8, int(14*scale))


def draw_section_header(draw, x, y, w, text, font, palette, scale):
    """Draw a small section header with underline, return new y."""
    draw.text((x, y), text.upper(), font=font, fill=palette["accent"])
    bb = draw.textbbox((0, 0), text.upper(), font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    draw.line([(x, y + th + 2), (x + w, y + th + 2)], fill=palette["accent"], width=max(1, int(1.5*scale)))
    return y + th + max(6, int(10*scale))


def generate_newspaper_image(width, height, palette, seed):
    rng = random.Random(seed)
    img = Image.new("RGB", (width, height), palette["bg"])
    draw = ImageDraw.Draw(img)

    margin = max(30, int(width * 0.03))
    col_w = width - 2 * margin
    scale = width / 2400  # base scale

    # Fonts
    f_tag      = get_font(max(14, int(26 * scale)))
    f_headline = get_font(max(22, int(80 * scale)))
    f_subhead  = get_font(max(14, int(38 * scale)))
    f_section  = get_font(max(12, int(22 * scale)))
    f_body     = get_font_regular(max(11, int(22 * scale)))
    f_caption  = get_font_regular(max(9, int(17 * scale)))
    f_header   = get_font(max(18, int(52 * scale)))
    f_pullq    = get_font(max(11, int(20 * scale)))

    y = margin

    # ── HEADER BAR ──────────────────────────────────────────────────────────
    header_h = int(height * 0.065)
    draw.rectangle([0, 0, width, header_h], fill=palette["accent"])
    header_text = rng.choice(["DENNÍ ZPRAVODAJ", "RANNÍ LISTY", "MAGAZÍN PLUS", "VĚSTNÍK REGIONU", "TÝDENÍK DNES"])
    hw = draw.textbbox((0, 0), header_text, font=f_header)
    hx = (width - (hw[2] - hw[0])) // 2
    draw.text((hx, (header_h - (hw[3]-hw[1])) // 2), header_text, font=f_header, fill=palette["bg"])

    # thin accent stripe
    stripe_y = header_h + max(2, int(4*scale))
    draw.line([(0, stripe_y), (width, stripe_y)], fill=palette["secondary"], width=max(2, int(4*scale)))

    # date / edition info
    date_text = (f"Ročník {rng.randint(5,40)}, číslo {rng.randint(1,52)}"
                 f"  |  Datum: {rng.randint(1,28)}.{rng.randint(1,12)}.{rng.randint(2020,2025)}"
                 f"  |  Cena {rng.randint(10,50)} Kč")
    draw.text((margin, stripe_y + max(4, int(7*scale))), date_text, font=f_caption, fill=palette["text"])

    y = int(header_h * 1.6) + max(4, int(8*scale))

    # ── TAG ─────────────────────────────────────────────────────────────────
    tag = rng.choice(TAGS)
    tag_pad = max(6, int(10*scale))
    tag_bbox = draw.textbbox((0, 0), tag, font=f_tag)
    tag_w = tag_bbox[2] - tag_bbox[0] + 2*tag_pad
    tag_h_px = tag_bbox[3] - tag_bbox[1] + tag_pad
    draw.rectangle([margin, y, margin+tag_w, y+tag_h_px], fill=palette["accent"])
    draw.text((margin+tag_pad, y+tag_pad//2), tag, font=f_tag, fill=palette["bg"])
    y += tag_h_px + max(8, int(14*scale))

    # ── HEADLINE ────────────────────────────────────────────────────────────
    headline = rng.choice(HEADLINES)
    hl_lines = wrap_text(headline, f_headline, col_w, draw)
    for line in hl_lines[:3]:
        draw.text((margin, y), line, font=f_headline, fill=palette["text"])
        bb = draw.textbbox((0, 0), line, font=f_headline)
        y += (bb[3]-bb[1]) + max(3, int(6*scale))

    y += max(6, int(10*scale))
    draw.line([(margin, y), (width-margin, y)], fill=palette["accent"], width=max(2, int(3*scale)))
    y += max(6, int(10*scale))

    # ── SUBHEADLINE ──────────────────────────────────────────────────────────
    subhead = rng.choice(SUBHEADLINES)
    sh_lines = wrap_text(subhead, f_subhead, col_w, draw)
    for line in sh_lines[:2]:
        draw.text((margin, y), line, font=f_subhead, fill=palette["accent"])
        bb = draw.textbbox((0, 0), line, font=f_subhead)
        y += (bb[3]-bb[1]) + max(2, int(4*scale))
    y += max(10, int(18*scale))

    # ── MAIN CONTENT ─────────────────────────────────────────────────────────
    remaining_h = height - y - margin - max(40, int(50*scale))  # leave footer space
    layout = rng.choice(["img_left", "img_right", "img_top", "three_col"])

    body_pool = list(BODY_TEXTS)
    rng.shuffle(body_pool)

    line_h_body = draw.textbbox((0,0), "Ag", font=f_body)[3] + max(2, int(4*scale))
    line_h_sec  = draw.textbbox((0,0), "Ag", font=f_section)[3] + max(6, int(10*scale))

    def fill_column(cx, cy, cw, max_cy, paragraphs, insert_pullquote=False, insert_section=True):
        """Fill a column with paragraphs, optional pull quote and section headers."""
        pq_inserted = False
        sh_inserted = 0
        para_count = 0
        for para in paragraphs:
            if cy + line_h_body > max_cy:
                break
            # Occasional section header
            if insert_section and para_count > 0 and para_count % 3 == 0 and sh_inserted < 3:
                if cy + line_h_sec + 10 < max_cy:
                    cy = draw_section_header(draw, cx, cy, cw,
                                             rng.choice(SECTION_HEADERS), f_section, palette, scale)
                    sh_inserted += 1
            # Pull quote in middle
            if insert_pullquote and not pq_inserted and para_count == len(paragraphs)//2:
                pq_h = max(60, int(90*scale))
                if cy + pq_h < max_cy:
                    cy = draw_pull_quote(draw, cx, cy, cw,
                                         rng.choice(PULL_QUOTES), f_pullq, f_caption, palette, scale)
                    pq_inserted = True
            # Body paragraph
            lines = wrap_text(para, f_body, cw, draw)
            for line in lines:
                if cy + line_h_body > max_cy:
                    break
                draw.text((cx, cy), line, font=f_body, fill=palette["text"])
                cy += line_h_body
            cy += max(8, int(14*scale))  # paragraph spacing
            para_count += 1
        return cy

    if layout == "img_left":
        img_w = int(col_w * 0.42)
        img_h = int(remaining_h * 0.48)
        draw_decorative_image(draw, margin, y, img_w, img_h, palette, seed+100)
        cap = rng.choice(CAPTIONS)
        cap_y = y + img_h + max(2, int(4*scale))
        draw.text((margin, cap_y), cap, font=f_caption, fill=palette["secondary"])

        # right column alongside image
        tx = margin + img_w + max(14, int(22*scale))
        tw = col_w - img_w - max(14, int(22*scale))
        fill_column(tx, y, tw, y + img_h, body_pool[:3])

        # full-width below
        y2 = y + img_h + max(25, int(38*scale))
        fill_column(margin, y2, col_w, height - margin - max(40,int(50*scale)),
                    body_pool[3:], insert_pullquote=True, insert_section=True)

    elif layout == "img_right":
        img_w = int(col_w * 0.40)
        img_h = int(remaining_h * 0.46)
        ix = margin + col_w - img_w
        draw_decorative_image(draw, ix, y, img_w, img_h, palette, seed+200)
        cap = rng.choice(CAPTIONS)
        draw.text((ix, y + img_h + max(2,int(4*scale))), cap, font=f_caption, fill=palette["secondary"])

        tw = col_w - img_w - max(14, int(22*scale))
        fill_column(margin, y, tw, y + img_h, body_pool[:3])

        y2 = y + img_h + max(25, int(38*scale))
        fill_column(margin, y2, col_w, height - margin - max(40,int(50*scale)),
                    body_pool[3:], insert_pullquote=True)

    elif layout == "img_top":
        img_h = int(remaining_h * 0.36)
        draw_decorative_image(draw, margin, y, col_w, img_h, palette, seed+300)
        cap = rng.choice(CAPTIONS)
        draw.text((margin, y + img_h + max(2,int(4*scale))), cap, font=f_caption, fill=palette["secondary"])
        y += img_h + max(22, int(36*scale))

        # two columns of text
        gap = max(16, int(26*scale))
        half = (col_w - gap) // 2
        div_x = margin + half + gap // 2
        draw.line([(div_x, y), (div_x, height - margin - max(40,int(50*scale)))],
                  fill=palette["accent"], width=max(1, int(2*scale)))

        pool_l = body_pool[:6]
        pool_r = body_pool[6:]
        fill_column(margin, y, half, height - margin - max(40,int(50*scale)),
                    pool_l, insert_pullquote=True)
        fill_column(margin + half + gap, y, half, height - margin - max(40,int(50*scale)),
                    pool_r, insert_section=True)

    else:  # three_col
        gap = max(14, int(22*scale))
        third = (col_w - 2*gap) // 3
        col_positions = [margin, margin + third + gap, margin + 2*(third + gap)]
        dividers = [col_positions[1] - gap//2, col_positions[2] - gap//2]
        max_y_col = height - margin - max(40, int(50*scale))

        for di in dividers:
            draw.line([(di, y), (di, max_y_col)],
                      fill=palette["accent"], width=max(1, int(2*scale)))

        # First col: image + text
        img_h = int(remaining_h * 0.38)
        draw_decorative_image(draw, col_positions[0], y, third, img_h, palette, seed+400)
        cap = rng.choice(CAPTIONS)
        draw.text((col_positions[0], y + img_h + max(2,int(4*scale))), cap, font=f_caption, fill=palette["secondary"])
        fill_column(col_positions[0], y + img_h + max(20, int(30*scale)), third, max_y_col, body_pool[:3])

        # Second col: text + pull quote
        fill_column(col_positions[1], y, third, max_y_col, body_pool[3:7], insert_pullquote=True)

        # Third col: section headers + text
        fill_column(col_positions[2], y, third, max_y_col, body_pool[7:], insert_section=True)

    # ── FOOTER ───────────────────────────────────────────────────────────────
    footer_y = height - margin - max(28, int(38*scale))
    draw.line([(margin, footer_y), (width-margin, footer_y)],
              fill=palette["accent"], width=max(1, int(2*scale)))
    footer = (f"www.zpravy-dnes.cz  |  redakce@novinky.cz"
              f"  |  tel: +420 {rng.randint(100,999)} {rng.randint(100,999)} {rng.randint(100,999)}"
              f"  |  © {rng.randint(2020,2025)} Všechna práva vyhrazena")
    draw.text((margin, footer_y + max(4, int(7*scale))), footer, font=f_caption, fill=palette["secondary"])

    return img


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    for mpx in SIZES_MPX:
        folder_name = f"{mpx}mpx"
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        width, height = mpx_to_dims(mpx)
        print(f"\n[{mpx} Mpx] Rozměry: {width}x{height}px")

        for i in range(5):
            seed = mpx * 100 + i * 37
            palette = PALETTES[i % len(PALETTES)]

            print(f"  Generuji obr. {i+1}/5 (seed={seed})...", end=" ", flush=True)
            img = generate_newspaper_image(width, height, palette, seed)

            fname = f"img_{i+1:02d}_{width}x{height}.png"
            fpath = os.path.join(folder_path, fname)
            img.save(fpath, format="PNG", compress_level=0)

            size_mb = os.path.getsize(fpath) / (1024*1024)
            print(f"OK → {size_mb:.1f} MB")
            total += 1

    print(f"\n✓ Vygenerováno {total} obrázků v {OUTPUT_DIR}")


if __name__ == "__main__":
    main()