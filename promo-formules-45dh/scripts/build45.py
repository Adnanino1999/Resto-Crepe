"""Resto Crêpes – 'Formules 45 DH' ad compositor. Native 4K vertical (2160x3840, 30 fps).

Design coordinates are 1080x1920 ("design px"); everything is drawn at x2.
Usage:
  build45.py --stills 1.0,5.2 --still-dir DIR     # review frames (PNG, half size)
  build45.py OUT.mp4 --audio MIX.wav              # full render
"""
import argparse
import functools
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from edit import V, TOTAL, FREEZE  # noqa: E402

S = os.environ.get("PROMO_WORK", os.path.dirname(HERE))
CACHE, FONTS, PHOTOS_DIR = f"{S}/cache4k", f"{S}/fonts", f"{S}/photos"
U = 2
W, H, FPS = 1080 * U, 1920 * U, 30
cv2.setNumThreads(4)

INK = (14, 9, 7)
CHOC = (30, 18, 11)
CREAM = (250, 243, 230)
WHITE = (255, 255, 255)
GOLD = (236, 178, 64)
GOLD_HI = (255, 228, 165)
GOLD_D = (158, 100, 24)
RED = (214, 34, 42)
GOLD_GRAD = (GOLD_HI, GOLD, GOLD_D)

GRADE = ("eq=contrast=1.06:saturation=1.14:brightness=0.004,"
         "colorbalance=rs=0.005:bs=-0.01:rm=0.0:bm=-0.005:rh=0.01:bh=-0.01,"
         "curves=m='0/0 0.06/0.045 0.25/0.235 0.75/0.79 1/1',"
         "unsharp=7:7:0.40:5:5:0.0")

# ------------------------------------------------------------------ timeline data
SEG = {k: dict(key=k, clip=c, ss=ss, dur=d, sp=sp, T0=T, T1=T + d) for k, c, ss, d, sp, T, _ in V}
SEG["p07"] = dict(key="p07", photo="photo_5764677286648025025_w.jpg", T0=12.35, T1=14.96, zx=(1.12, 1.20), py=-0.20, foc=(0.46, 0.27))
SEG["p12"] = dict(key="p12", photo="photo_6050655862986052165_w.jpg", T0=19.76, T1=21.75, zx=(1.10, 1.18), py=-0.10, foc=(0.50, 0.25))
SEG[FREEZE[0]] = dict(key=FREEZE[0], freeze_of=FREEZE[1], T0=FREEZE[2], T1=FREEZE[2] + FREEZE[3])
ORDER = sorted(SEG.values(), key=lambda s: s["T0"])

HEADS = {"v00", "v01", "v03", "v06", "v08", "v13", "v17a", "v17b", "v17c", "v17d", "v17g"}
SUPER = {k for k in SEG if k.startswith("v02")}
CAM = {  # key: (z0, z1, fx, fy, punch)
    "v00": (1.00, 1.05, 0.5, 0.45, 0.0), "v01": (1.03, 1.08, 0.5, 0.35, 0.07), "v03": (1.04, 1.07, 0.5, 0.35, 0.05),
    "v06": (1.02, 1.06, 0.5, 0.55, 0.04), "v08": (1.03, 1.08, 0.5, 0.35, 0.05), "v13": (1.03, 1.08, 0.5, 0.35, 0.05),
    "v14": (1.10, 1.14, 0.5, 0.5, 0.0), "v15": (1.02, 1.10, 0.5, 0.62, 0.0),
    "v16": (1.06, 1.10, 0.5, 0.5, 0.0), "v17a": (1.00, 1.04, 0.5, 0.4, 0.0), "v17b": (1.02, 1.07, 0.5, 0.35, 0.06),
    "v17c": (1.04, 1.08, 0.5, 0.35, 0.07), "v17d": (1.02, 1.06, 0.5, 0.35, 0.07), "v17g": (1.06, 1.10, 0.5, 0.4, 0.0),
    "v04c": (1.04, 1.12, 0.45, 0.42, 0.04), "v09d": (1.06, 1.14, 0.5, 0.5, 0.04), "v02c": (1.05, 1.13, 0.5, 0.62, 0.06),
}
WHIP_AT = [4.25, 12.35, 19.76, 24.25, 25.45, 41.80]
SUPER_CUTS = [4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25]
SLAMS = {"p07": 12.72, "p12": 20.25}
END_T, STEP_B, CTA_T = 41.80, 44.60, 47.30
SKIP0, SKIP1 = 24.25, 25.45
PHONE = "06 60 24 14 75"

SUBS_UNUSED = [  # French subtitles: dropped on request (Moroccan audience), kept for reference
    (0.28, 1.90, "Qu'est-ce que tu manges ?", "q"),
    (1.95, 3.10, "Frérot, Resto Crêpes a lancé…", "a"),
    (3.10, 4.22, "…une *promo de folie* !", "a"),
    (8.18, 8.78, "Y a quoi dedans ?", "q"),
    (8.80, 9.95, "Une *pizza 4 fromages*…", "a"),
    (9.95, 11.10, "…avec un *pancake*…", "a"),
    (11.10, 12.33, "…et une *boisson* !", "a"),
    (14.98, 16.44, "Et chez Resto Crêpes,|y a aussi…", "a"),
    (16.46, 17.76, "…un *tacos poulet*…", "a"),
    (17.76, 18.96, "…avec une *crêpe Nutella*…", "a"),
    (18.96, 19.74, "…et des *frites* !", "a"),
    (21.80, 22.62, "C'est où ça ?", "q"),
    (22.88, 24.20, "Chez *Resto Crêpes*, frérot !", "a"),
]
CHIPS = [  # (t_in, t_out, row, text, plus, offered)
    (8.62, 12.35, 0, "PIZZA 4 FROMAGES", False, False),
    (9.98, 12.35, 1, "PANCAKE", True, False),
    (11.13, 12.35, 2, "BOISSON", True, True),
    (16.50, 19.76, 0, "TACOS POULET", False, False),
    (17.80, 19.76, 1, "CRÊPE NUTELLA", True, False),
    (19.00, 19.76, 2, "FRITES", True, False),
    (35.90, 39.15, 0, "FORMULE TACOS  •  45 DH", False, False),
]
CARDS = {
    "p07": dict(title="FORMULE PIZZA", lines=["Pizza 4 fromages  +  Pancake", "+  Boisson offerte"]),
    "p12": dict(title="FORMULE TACOS", lines=["Tacos poulet  +  Frites  +  Crêpe Nutella", "+  Boisson offerte"]),
}
TOPBAR = [(0.30, 4.25), (7.75, 12.35), (14.96, 19.76), (21.75, 24.25), (25.50, 39.20)]


# ------------------------------------------------------------------ easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(p):
    p = clamp(p)
    return 1 - (1 - p) ** 3


def ease_io(p):
    p = clamp(p)
    return p * p * (3 - 2 * p)


def ease_back(p, s=1.7):
    p = clamp(p) - 1
    return p * p * ((s + 1) * p + s) + 1


def prog(t, start, dur):
    return clamp((t - start) / dur) if dur > 0 else float(t >= start)


def window(t, t0, t1, fin=0.14, fout=0.10):
    """0..1 visibility with fade in/out."""
    if t < t0 or t >= t1:
        return 0.0
    return min(clamp((t - t0) / fin), clamp((t1 - t) / fout))


# ------------------------------------------------------------------ fonts & sprites
FONT_FILES = {"anton": "Anton.ttf", "mont": "Montserrat.ttf", "kaushan": "Kaushan.ttf", "lalezar": "Lalezar.ttf"}
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"


@functools.lru_cache(None)
def font(name, size_px, var=None):
    f = ImageFont.truetype(f"{FONTS}/{FONT_FILES[name]}", size_px)
    if var:
        f.set_variation_by_name(var)
    return f


def _grad_fill(size, stops):
    h = size[1]
    y = np.linspace(0, 1, h)[:, None]
    if len(stops) == 2:
        col = np.array(stops[0]) * (1 - y) + np.array(stops[1]) * y
    else:
        a, b, c = (np.array(s, np.float32) for s in stops)
        col = np.where(y < 0.5, a * (1 - y * 2) + b * (y * 2), b * (1 - (y - 0.5) * 2) + c * ((y - 0.5) * 2))
    return np.broadcast_to(col[:, None, :], (h, size[0], 3)).astype(np.uint8)


def tsprite(txt, fname, size, fill=CREAM, var=None, tracking=0.0, stroke=0.0, stroke_fill=INK, shadow=0.0,
            shadow_blur=10.0, shadow_off=(0, 5), grad=None, glow=None):
    """Single-line text sprite (sizes in design px). Returns RGBA uint8 array, tightly padded."""
    f = font(fname, int(size * U), var)
    tr, sw = tracking * U, int(round(stroke * U))
    if tr:
        widths = [f.getlength(ch) for ch in txt]
        tw = sum(widths) + tr * (len(txt) - 1)
    else:
        tw = f.getlength(txt)
    asc, desc = f.getmetrics()
    th = asc + desc
    blur = shadow_blur * U if (shadow > 0 or glow) else 0
    pad = int(sw + blur * 3 + 8 * U)
    size_px = (int(math.ceil(tw)) + 2 * pad, th + 2 * pad)

    def mask(swidth):
        m = Image.new("L", size_px, 0)
        d = ImageDraw.Draw(m)
        if tr:
            x = pad
            for ch, w_ in zip(txt, widths):
                d.text((x, pad), ch, font=f, fill=255, stroke_width=swidth)
                x += w_ + tr
        else:
            d.text((pad, pad), txt, font=f, fill=255, stroke_width=swidth)
        return m

    m_in = mask(0)
    m_out = mask(sw) if sw else m_in
    img = Image.new("RGBA", size_px, (0, 0, 0, 0))
    if glow:
        gcol, gstr = glow
        gm = m_out.filter(ImageFilter.GaussianBlur(blur))
        gl = Image.new("RGBA", size_px, tuple(gcol) + (0,))
        gl.putalpha(Image.eval(gm, lambda v: int(min(255, v * gstr))))
        img.alpha_composite(gl)
    if shadow > 0:
        sm = m_out.filter(ImageFilter.GaussianBlur(blur))
        sl = Image.new("RGBA", size_px, (6, 3, 2, 0))
        sl.putalpha(Image.eval(sm, lambda v: int(min(255, v * 1.6) * shadow)))
        img.alpha_composite(sl, (int(shadow_off[0] * U), int(shadow_off[1] * U)))
    if sw:
        st = Image.new("RGBA", size_px, tuple(stroke_fill) + (0,))
        st.putalpha(m_out)
        img.alpha_composite(st)
    if grad:
        fg = Image.fromarray(np.dstack([_grad_fill(size_px, grad), np.array(m_in)]), "RGBA")
    else:
        fg = Image.new("RGBA", size_px, tuple(fill) + (0,))
        fg.putalpha(m_in)
    img.alpha_composite(fg)
    return np.array(img)


def rich_line(words, size, var=b"ExtraBold", stroke=2.2, shadow=0.85):
    """words: list of (text, colour) drawn on one line with a shared outline + soft shadow."""
    f = font("mont", int(size * U), var)
    sw = int(round(stroke * U))
    txt = "".join(w for w, _ in words)
    tw = f.getlength(txt)
    asc, desc = f.getmetrics()
    blur = 9 * U
    pad = int(sw + blur * 3 + 6 * U)
    size_px = (int(math.ceil(tw)) + 2 * pad, asc + desc + 2 * pad)
    m_out = Image.new("L", size_px, 0)
    ImageDraw.Draw(m_out).text((pad, pad), txt, font=f, fill=255, stroke_width=sw)
    img = Image.new("RGBA", size_px, (0, 0, 0, 0))
    sm = m_out.filter(ImageFilter.GaussianBlur(blur))
    sl = Image.new("RGBA", size_px, (4, 2, 1, 0))
    sl.putalpha(Image.eval(sm, lambda v: int(min(255, v * 1.5) * shadow)))
    img.alpha_composite(sl, (0, int(4 * U)))
    st = Image.new("RGBA", size_px, (10, 6, 4, 0))
    st.putalpha(Image.eval(m_out, lambda v: int(v * 0.9)))
    img.alpha_composite(st)
    x = pad
    for w_, col in words:
        m = Image.new("L", size_px, 0)
        ImageDraw.Draw(m).text((x, pad), w_, font=f, fill=255)
        fg = Image.new("RGBA", size_px, tuple(col) + (0,))
        fg.putalpha(m)
        img.alpha_composite(fg)
        x += f.getlength(w_)
    return np.array(img)


def rbox(w, h, fill, alpha=255, radius=None, outline=None, ow=1.5, oalpha=255, shadow=0.0):
    """Rounded box sprite (design px). Supersampled for smooth edges."""
    SS = 2
    wp, hp_, r = int(w * U), int(h * U), int((radius if radius is not None else h / 2) * U)
    pad = int(24 * U) if shadow else 2
    img = Image.new("RGBA", ((wp + 2 * pad) * SS, (hp_ + 2 * pad) * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if shadow:
        sh = Image.new("L", img.size, 0)
        ImageDraw.Draw(sh).rounded_rectangle([pad * SS, (pad + 6 * U) * SS, (pad + wp) * SS, (pad + hp_ + 6 * U) * SS], r * SS,
                                             fill=int(255 * shadow))
        sh = sh.filter(ImageFilter.GaussianBlur(14 * U * SS))
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        layer.putalpha(sh)
        img.alpha_composite(layer)
    d.rounded_rectangle([pad * SS, pad * SS, (pad + wp) * SS - 1, (pad + hp_) * SS - 1], r * SS, fill=tuple(fill) + (alpha,))
    if outline:
        d.rounded_rectangle([pad * SS, pad * SS, (pad + wp) * SS - 1, (pad + hp_) * SS - 1], r * SS,
                            outline=tuple(outline) + (oalpha,), width=max(1, int(ow * U * SS)))
    return np.array(img.resize((wp + 2 * pad, hp_ + 2 * pad), Image.LANCZOS))


def hline(w, h=1.5, color=GOLD, fade=True):
    wp, hp_ = int(w * U), max(1, int(h * U))
    a = np.ones(wp, np.float32)
    if fade:
        x = np.linspace(-1, 1, wp)
        a = np.clip(1.25 - np.abs(x) ** 2 * 1.25, 0, 1)
    arr = np.zeros((hp_, wp, 4), np.uint8)
    arr[..., :3] = color
    arr[..., 3] = (a * 255).astype(np.uint8)[None, :]
    return arr


def soft_dot(r_px):
    n = int(r_px * 2 + 2)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / r_px
    a = np.clip(1 - d, 0, 1) ** 1.8
    arr = np.zeros((n, n, 4), np.uint8)
    arr[..., :3] = GOLD_HI
    arr[..., 3] = (a * 255).astype(np.uint8)
    return arr


@functools.lru_cache(None)
def emoji(ch, size):
    """Colour emoji sprite (Noto Color Emoji is a 109 px bitmap font), scaled to `size` design px."""
    f = ImageFont.truetype(EMOJI_FONT, 109)
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
    im = im.crop(im.getbbox())
    px = int(size * U)
    im = im.resize((px, int(px * im.height / im.width)), Image.LANCZOS)
    return np.array(im)


def speed_lines(n=56, seed=3):
    """Manga-style radial speed lines (transparent centre)."""
    r = np.random.default_rng(seed)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    cx, cy = W / 2, H * 0.36
    R = math.hypot(W, H)
    for i in range(n):
        a = 2 * math.pi * i / n + r.uniform(-0.04, 0.04)
        w = r.uniform(0.006, 0.02)
        r0 = r.uniform(0.42, 0.6) * W
        pts = [(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + R * math.cos(a - w), cy + R * math.sin(a - w)),
               (cx + R * math.cos(a + w), cy + R * math.sin(a + w))]
        d.polygon(pts, fill=int(r.uniform(150, 255)))
    arr = np.zeros((H, W, 4), np.uint8)
    arr[..., :3] = 255
    arr[..., 3] = np.array(img.filter(ImageFilter.GaussianBlur(2)))
    return arr


def hcat(sprites, gap):
    """Concatenate RGBA sprites horizontally (vertically centred)."""
    hh = max(s.shape[0] for s in sprites)
    ww = sum(s.shape[1] for s in sprites) + int(gap * U) * (len(sprites) - 1)
    out = np.zeros((hh, ww, 4), np.uint8)
    x = 0
    for s in sprites:
        y = (hh - s.shape[0]) // 2
        blit_rgba(out, s, x, y)
        x += s.shape[1] + int(gap * U)
    return out


def blit_rgba(dst, src, x, y):
    x, y = int(round(x)), int(round(y))
    h, w = src.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255
    d = dst[y0:y1, x0:x1].astype(np.float32) / 255
    sa, da = s[..., 3:4], d[..., 3:4]
    oa = sa + da * (1 - sa)
    rgb = (s[..., :3] * sa + d[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-6)
    dst[y0:y1, x0:x1] = (np.dstack([rgb, oa]) * 255 + 0.5).astype(np.uint8)


def blit(frame, spr, x, y, alpha=1.0, scale=1.0, rot=0.0, anchor="c", additive=False):
    """Composite RGBA sprite on the RGB frame. (x, y) in DESIGN px, at the anchor point."""
    if alpha <= 0.003 or scale <= 0.01:
        return
    x, y = x * U, y * U
    sh, sw = spr.shape[:2]
    if scale != 1.0 or rot != 0.0:
        if rot:
            diag = int(math.ceil(math.hypot(sw, sh) * scale)) + 2
            M = cv2.getRotationMatrix2D((sw / 2, sh / 2), rot, scale)
            M[0, 2] += diag / 2 - sw / 2
            M[1, 2] += diag / 2 - sh / 2
            spr = cv2.warpAffine(spr, M, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        else:
            spr = cv2.resize(spr, (max(2, int(sw * scale)), max(2, int(sh * scale))),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        sh, sw = spr.shape[:2]
    if anchor == "c":
        x, y = x - sw / 2, y - sh / 2
    elif anchor == "l":
        y = y - sh / 2
    elif anchor == "r":
        x, y = x - sw, y - sh / 2
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(frame.shape[1], x + sw), min(frame.shape[0], y + sh)
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4].astype(np.float32) * (alpha / 255.0)
    reg = frame[y0:y1, x0:x1].astype(np.float32)
    if additive:
        reg = np.minimum(255, reg + s[..., :3].astype(np.float32) * a)
    else:
        reg = reg * (1 - a) + s[..., :3].astype(np.float32) * a
    frame[y0:y1, x0:x1] = reg.astype(np.uint8)


def shine(spr, p, width=0.16, strength=0.75):
    """Diagonal light sweep across a sprite (p: 0..1)."""
    if p <= 0 or p >= 1:
        return spr
    h, w = spr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c = (-0.3 + 1.6 * p) * (w + h * 0.4)
    band = np.clip(1 - np.abs((xx + yy * 0.4) - c) / (width * w), 0, 1) ** 1.5
    out = spr.copy()
    out[..., :3] = np.clip(spr[..., :3] + (band[..., None] * 255 * strength), 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------ image ops
def warp(frame, z, fx, fy, dx=0.0, dy=0.0, rot=0.0):
    if abs(z - 1) < 1e-4 and not (dx or dy or rot):
        return frame
    M = cv2.getRotationMatrix2D((fx * W, fy * H), rot, z)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def zoom_blur(frame, p, direction=1):
    """Radial zoom blur (computed at half res) for whip transitions."""
    if p <= 0.01:
        return frame
    small = cv2.resize(frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    acc = np.zeros(small.shape, np.float32)
    k = 7
    for i in range(k):
        z = 1 + direction * (0.5 * p * p) + 0.05 * p * i
        M = cv2.getRotationMatrix2D((W / 4, H / 4), 0, z)
        acc += cv2.warpAffine(small, M, (W // 2, H // 2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    bl = cv2.resize((acc / k).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
    a = clamp(p * 1.6)
    return cv2.addWeighted(frame, 1 - a, bl, a, 0)


def make_masks():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W * 0.80)) ** 2 + ((yy - H / 2) / (H * 0.74)) ** 2)
    vig = 1 - 0.38 * np.clip(r - 0.38, 0, 1) ** 1.5
    y = np.arange(H, dtype=np.float32) / U
    top = 1 - 0.42 * np.clip((360 - y) / 360, 0, 1) ** 1.5             # keeps the brand bar readable
    subs = 1 - 0.30 * np.clip((y - 1050) / 300, 0, 1) * np.clip((1700 - y) / 300, 0, 1)
    head = (vig * (top * subs)[:, None])[..., None]
    sup = (vig * 0.92 * (1 - 0.35 * np.exp(-((y - 960) / 520) ** 2))[:, None])[..., None]
    card = (np.clip(1 - 0.97 * np.clip((y - 760) / 560, 0, 1) ** 1.2, 0, 1) * (1 - 0.35 * np.clip((300 - y) / 300, 0, 1)))[:, None]
    card = (vig * card)[..., None]
    cta = (vig * ((1 - 0.32 * np.clip((y - 1080) / 480, 0, 1)) * top)[:, None])[..., None]
    f = lambda m: np.ascontiguousarray(np.broadcast_to(m, (H, W, 3)).astype(np.float32))
    return {"head": f(head), "super": f(sup), "card": f(card), "cta": f(cta), "vig": f(vig[..., None])}


def apply(frame, m):
    return cv2.multiply(frame, m, dtype=cv2.CV_8U)


class Grain:
    def __init__(self, n=6, amp=3.2, seed=9):
        r = np.random.default_rng(seed)
        self.pos, self.neg = [], []
        for _ in range(n):
            g = r.normal(0, amp, (H // 2, W // 2)).astype(np.float32)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
            g3 = np.repeat(g[..., None], 3, axis=2)
            self.pos.append(np.clip(g3, 0, 255).astype(np.uint8))
            self.neg.append(np.clip(-g3, 0, 255).astype(np.uint8))

    def apply(self, frame, i):
        k = i % len(self.pos)
        return cv2.subtract(cv2.add(frame, self.pos[k]), self.neg[k])


# ------------------------------------------------------------------ sources
def video_cmd(key, start_frame=0, nframes=None):
    cmd = ["ffmpeg", "-v", "error"]
    if start_frame:
        cmd += ["-ss", f"{start_frame / FPS:.4f}"]
    cmd += ["-i", f"{CACHE}/{key}.mp4", "-an", "-vf", GRADE]
    if nframes:
        cmd += ["-frames:v", str(nframes)]
    return cmd + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


class VideoReader:
    def __init__(self, key, start_frame=0):
        self.p = subprocess.Popen(video_cmd(key, start_frame), stdout=subprocess.PIPE, bufsize=W * H * 3)
        self.last = None

    def read(self):
        b = self.p.stdout.read(W * H * 3)
        if len(b) == W * H * 3:
            self.last = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        if self.last is None:
            self.last = np.zeros((H, W, 3), np.uint8)
        return self.last

    def close(self):
        try:
            self.p.stdout.close()
            self.p.kill()
            self.p.wait()
        except Exception:
            pass


_FROZEN = {}


def frozen_frame(seg):
    held = SEG[seg["freeze_of"]]
    if held["key"] not in _FROZEN:
        k = int(round((held["T1"] - held["T0"]) * FPS)) - 1
        r = VideoReader(held["key"], k)
        _FROZEN[held["key"]] = r.read().copy()
        r.close()
    return _FROZEN[held["key"]]


@functools.lru_cache(None)
def photo_canvas(name, py):
    """Photo graded with the same chain, width-fitted to 2160, shifted by py*H, fading into black below."""
    out = f"{S}/strip/graded_{name}.png"
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{PHOTOS_DIR}/{name}", "-vf",
                        f"scale={W}:-2:flags=lanczos,{GRADE}", out], check=True)
    im = cv2.cvtColor(cv2.imread(out), cv2.COLOR_BGR2RGB)
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:] = INK
    oy = int(py * H)
    y0, y1 = max(0, oy), min(H, oy + im.shape[0])
    canvas[y0:y1] = im[y0 - oy:y1 - oy]
    # soft fade from the photo's bottom edge into ink
    fe = oy + im.shape[0]
    n = int(260 * U)
    a = np.clip((np.arange(H) - (fe - n)) / n, 0, 1)[:, None, None]
    canvas = (canvas * (1 - a) + np.array(INK) * a).astype(np.uint8)
    return canvas


# ------------------------------------------------------------------ graphics
class GFX:
    def __init__(self):
        self.masks = make_masks()
        self.grain = Grain()
        self.sub_cache = {}
        self.dots = [soft_dot(r * U) for r in (6, 10, 16, 26)]
        rr = np.random.default_rng(4)
        self.parts = [(rr.uniform(40, 1040), rr.uniform(0, 1920), rr.integers(0, 4), rr.uniform(14, 48), rr.uniform(0, 6.28),
                       rr.uniform(0.25, 0.8)) for _ in range(46)]
        # brand bar
        self.brand = tsprite("RESTO CRÊPES", "mont", 25, CREAM, var=b"Black", tracking=9, shadow=0.6, shadow_blur=6)
        self.brand_rule = hline(300, 1.4, GOLD)
        self.brand_date = tsprite("OFFRE DU 22 AU 29 NOVEMBRE", "mont", 17, GOLD_HI, var=b"Bold", tracking=4.5, shadow=0.6, shadow_blur=5)
        # supercut typography
        self.sc_kicker = tsprite("NOUVEAU CHEZ RESTO CRÊPES", "mont", 30, GOLD_HI, var=b"Bold", tracking=8, shadow=0.7)
        self.sc_two = tsprite("2 FORMULES", "anton", 205, CREAM, tracking=3, shadow=0.8, shadow_blur=16)
        self.sc_la = tsprite("LA FORMULE COMPLÈTE", "mont", 34, CREAM, var=b"Bold", tracking=9, shadow=0.7)
        self.sc_45 = tsprite("45", "anton", 430, grad=GOLD_GRAD, shadow=0.8, shadow_blur=18, stroke=1.2, stroke_fill=GOLD_D)
        self.sc_dh = tsprite("DH", "anton", 150, grad=GOLD_GRAD, shadow=0.8, shadow_blur=14, stroke=1.0, stroke_fill=GOLD_D)
        self.sc_week = tsprite("1 SEMAINE", "anton", 200, CREAM, tracking=3, shadow=0.8, shadow_blur=16)
        self.sc_only = tsprite("SEULEMENT", "mont", 58, GOLD, var=b"Black", tracking=16, shadow=0.7)
        self.sc_dates = tsprite("DU 22 AU 29 NOVEMBRE", "mont", 38, CREAM, var=b"Bold", tracking=6, shadow=0.7)
        self.rule_w = hline(520, 1.6, GOLD)
        # chips
        self.chips = {}
        self.chip_label = tsprite("LA FORMULE", "mont", 20, GOLD_HI, var=b"Bold", tracking=7, shadow=0.7, shadow_blur=6)
        for _, _, _, txt, plus, off in CHIPS:
            self.chips[txt] = self.make_chip(txt, plus, off)
        # cards
        self.cards = {k: self.make_card(v) for k, v in CARDS.items()}
        self.card_45 = tsprite("45", "anton", 400, grad=GOLD_GRAD, shadow=0.9, shadow_blur=20, stroke=1.2, stroke_fill=GOLD_D)
        self.card_dh = tsprite("DH", "anton", 132, grad=GOLD_GRAD, shadow=0.9, shadow_blur=14, stroke=1.0, stroke_fill=GOLD_D)
        self.card_date = tsprite("OFFRE VALABLE DU 22 AU 29 NOVEMBRE", "mont", 21, CREAM, var=b"SemiBold", tracking=4, shadow=0.7)
        self.glow = self.make_glow(560)
        # time skip + comic freeze
        self.ts_text = tsprite("من بعد شي سوايع...", "lalezar", 118, CREAM, shadow=0.85, shadow_blur=12, glow=(GOLD, 0.55))
        self.ts_clock = emoji("⏰", 190)
        self.ts_ff = emoji("⏩", 74)
        self.lines = speed_lines()
        self.angry = emoji("😤", 230)
        yy = np.arange(H) % (6 * U)
        self.scan = np.ascontiguousarray(np.broadcast_to((1 - 0.16 * (yy < 2 * U))[:, None, None], (H, W, 3)).astype(np.float32))
        # end card
        self.make_end()

    # ---- builders
    def make_chip(self, txt, plus, offered):
        parts = []
        if plus:
            parts.append(tsprite("+", "mont", 40, GOLD, var=b"Black"))
        parts.append(tsprite(txt, "mont", 36, CREAM, var=b"ExtraBold", tracking=1.5))
        if offered:
            o = tsprite("OFFERTE", "mont", 21, WHITE, var=b"ExtraBold", tracking=3)
            box = rbox(o.shape[1] / U - 14, 40, RED, radius=20)
            blit_rgba(box, o, (box.shape[1] - o.shape[1]) / 2, (box.shape[0] - o.shape[0]) / 2)
            parts.append(box)
        row = hcat(parts, 6)
        wd, hd = row.shape[1] / U + 30, 74
        chip = rbox(wd, hd, INK, alpha=200, radius=18, outline=GOLD, ow=1.4, oalpha=210, shadow=0.55)
        pad = (chip.shape[0] - hd * U) / 2
        blit_rgba(chip, row, pad + 15 * U - 0, (chip.shape[0] - row.shape[0]) / 2)
        return chip

    def make_card(self, c):
        title = tsprite(c["title"], "mont", 34, GOLD, var=b"Black", tracking=12, shadow=0.7)
        lines = [tsprite(l, "mont", 37, CREAM, var=b"SemiBold", tracking=0.5, shadow=0.8) for l in c["lines"]]
        return dict(title=title, lines=lines, rule=hline(210, 1.5, GOLD))

    def make_glow(self, r):
        n = int(r * 2 * U)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        d = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n / 2)
        a = np.clip(1 - d, 0, 1) ** 2.2
        arr = np.zeros((n, n, 4), np.uint8)
        arr[..., :3] = (255, 190, 90)
        arr[..., 3] = (a * 255).astype(np.uint8)
        return arr

    def make_end(self):
        self.e_brand = tsprite("RESTO CRÊPES", "anton", 96, CREAM, tracking=8, shadow=0.8, shadow_blur=12)
        self.e_rule = hline(420, 1.6, GOLD)
        self.e_two = tsprite("2 FORMULES", "anton", 128, CREAM, tracking=2, shadow=0.8, shadow_blur=12)
        self.e_a = tsprite("à", "kaushan", 92, GOLD_HI, shadow=0.7)
        self.e_45 = tsprite("45 DH", "anton", 150, grad=GOLD_GRAD, shadow=0.85, shadow_blur=14, stroke=1.0, stroke_fill=GOLD_D)
        self.e_cards = [self.make_end_card("photo_6050655862986052168_w.jpg", (0.52, 0.33), "FORMULE TACOS",
                                           ["Tacos poulet + Frites", "Crêpe Nutella"]),
                        self.make_end_card("photo_5764677286648025027_w.jpg", (0.62, 0.66), "FORMULE PIZZA",
                                           ["Pizza 4 fromages", "Pancake"])]
        self.e_week = tsprite("OFFRE VALABLE 1 SEMAINE", "mont", 24, GOLD_HI, var=b"Bold", tracking=8, shadow=0.7)
        self.e_dates = tsprite("DU 22 AU 29 NOVEMBRE", "anton", 92, CREAM, tracking=3, shadow=0.85, shadow_blur=10)
        self.e_datebox = rbox(self.e_dates.shape[1] / U + 30, 132, INK, alpha=150, radius=20, outline=GOLD, ow=1.6, shadow=0.5)
        self.e_order = tsprite("POUR COMMANDER", "mont", 30, GOLD_HI, var=b"Bold", tracking=10, shadow=0.7)
        ph = hcat([emoji("📞", 84), tsprite(PHONE, "anton", 122, grad=GOLD_GRAD, tracking=3, stroke=0.8, stroke_fill=GOLD_D)], 22)
        self.e_phone_box = rbox(ph.shape[1] / U + 70, 172, INK, alpha=170, radius=30, outline=GOLD, ow=1.8, shadow=0.55)
        self.e_phone = ph
        self.e_deliv = hcat([emoji("🛵", 54), tsprite("Livraison  Rabat  •  Témara", "mont", 41, CREAM, var=b"Bold", shadow=0.8)], 16)
        self.e_addr = hcat([emoji("📍", 48), tsprite("Hay El Fath  –  Rabat", "mont", 41, CREAM, var=b"SemiBold", shadow=0.8)], 14)
        self.cta = tsprite("On vous attend !", "kaushan", 128, CREAM, shadow=0.9, shadow_blur=14, glow=(GOLD, 0.6))
        self.cta_brand = tsprite("RESTO CRÊPES", "anton", 70, CREAM, tracking=8, shadow=0.8, shadow_blur=10)
        self.cta_sub = tsprite("2 formules à 45 DH  •  du 22 au 29 novembre", "mont", 28, GOLD_HI, var=b"Bold", tracking=1, shadow=0.8)
        self.cta_phone = hcat([emoji("📞", 50), tsprite(PHONE, "anton", 70, CREAM, tracking=2, shadow=0.85, shadow_blur=10)], 14)

    def make_end_card(self, photo, focus, title, lines):
        cw, ch = 960, 250
        card = rbox(cw, ch, (24, 15, 10), alpha=228, radius=30, outline=GOLD, ow=1.3, oalpha=150, shadow=0.6)
        pad = (card.shape[0] - ch * U) // 2
        # square thumbnail with rounded corners
        im = Image.open(f"{PHOTOS_DIR}/{photo}").convert("RGB")
        side = int(min(im.size) * 0.62)
        cx, cy = int(im.size[0] * focus[0]), int(im.size[1] * focus[1])
        box = (max(0, cx - side // 2), max(0, cy - side // 2))
        box = (min(box[0], im.size[0] - side), min(box[1], im.size[1] - side))
        th = int(206 * U)
        thumb = im.crop((box[0], box[1], box[0] + side, box[1] + side)).resize((th, th), Image.LANCZOS)
        thumb = Image.fromarray(np.clip((np.array(thumb).astype(np.float32) - 128) * 1.08 + 131, 0, 255).astype(np.uint8))
        m = Image.new("L", (th * 2, th * 2), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, th * 2 - 1, th * 2 - 1], int(22 * U * 2), fill=255)
        thumb_rgba = thumb.convert("RGBA")
        thumb_rgba.putalpha(m.resize((th, th), Image.LANCZOS))
        blit_rgba(card, np.array(thumb_rgba), pad + 22 * U, pad + 22 * U)
        x0 = pad + (22 + 206 + 34) * U
        t = tsprite(title, "mont", 30, GOLD, var=b"Black", tracking=7)
        blit_rgba(card, t, x0 - 8 * U, pad + 28 * U)
        for i, l in enumerate(lines):
            s = tsprite(l, "mont", 31, CREAM, var=b"SemiBold")
            blit_rgba(card, s, x0 - 8 * U, pad + (84 + i * 46) * U)
        off = tsprite("+ BOISSON OFFERTE", "mont", 19, WHITE, var=b"ExtraBold", tracking=2.5)
        ob = rbox(off.shape[1] / U - 10, 36, RED, radius=18)
        blit_rgba(ob, off, (ob.shape[1] - off.shape[1]) / 2, (ob.shape[0] - off.shape[0]) / 2)
        blit_rgba(card, ob, x0, pad + 186 * U)
        price = tsprite("45", "anton", 112, grad=GOLD_GRAD, stroke=0.8, stroke_fill=GOLD_D)
        dh = tsprite("DH", "anton", 52, grad=GOLD_GRAD)
        pr = hcat([price, dh], -8)
        blit_rgba(card, pr, pad + (cw - 26) * U - pr.shape[1] + 30 * U, pad + (ch / 2) * U - pr.shape[0] / 2)
        return card

    # ---- subtitle
    def subtitle(self, frame, t):
        for t0, t1, txt, kind in SUBS_UNUSED:
            a = window(t, t0, t1, 0.10, 0.07)
            if a <= 0:
                continue
            key = txt
            if key not in self.sub_cache:
                self.sub_cache[key] = self.make_sub(txt, kind)
            spr = self.sub_cache[key]
            p = prog(t, t0, 0.16)
            sc = 0.94 + 0.06 * ease_back(p, 2.2)
            y = 1332 if t0 > 1.0 else 1300
            blit(frame, spr, 540, y, alpha=a, scale=sc)

    def make_sub(self, txt, kind):
        size = 60 if kind == "a" else 62
        base = CREAM if kind == "a" else GOLD_HI
        import re
        flat = []                      # [word, colour, space_before]
        prev_space = True
        for i, chunk in enumerate(txt.split("*")):
            col = GOLD if i % 2 == 1 else base
            for tok in re.findall(r"\s*\S+|\s+$", chunk):
                if not tok.strip():
                    prev_space = True
                    continue
                flat.append([tok.strip(), col, prev_space or tok[0].isspace()])
                prev_space = False
            if chunk.endswith(" "):
                prev_space = True
        f = font("mont", int(size * U), b"ExtraBold")
        def width(ws):
            return f.getlength("".join((" " if (s_ and j) else "") + x for j, (x, _, s_) in enumerate(ws)))

        items = [tuple(x) for x in flat]
        if any(x[0].endswith("|") or "|" in x[0] for x in items):   # manual break
            k = next(i for i, x in enumerate(items) if "|" in x[0])
            a_, b_ = items[k][0].split("|")
            first = items[:k] + ([(a_, items[k][1], items[k][2])] if a_ else [])
            second = ([(b_, items[k][1], False)] if b_ else []) + items[k + 1:]
            lines = [first, second]
        elif width(items) <= 900 * U:
            lines = [items]
        else:                          # balanced two-line split
            best = min(range(1, len(items)), key=lambda k: max(width(items[:k]), width(items[k:])))
            lines = [items[:best], items[best:]]
        sprites = []
        for ln in lines:
            segs = []
            for j, (w_, col, sp) in enumerate(ln):
                segs.append(((" " if (sp and j) else "") + w_, col))
            sprites.append(rich_line(segs, size))
        if len(sprites) == 1:
            return sprites[0]
        wd = max(s.shape[1] for s in sprites)
        gap = int(-38 * U)
        hh = sum(s.shape[0] for s in sprites) + gap * (len(sprites) - 1)
        out = np.zeros((hh, wd, 4), np.uint8)
        y = 0
        for s in sprites:
            blit_rgba(out, s, (wd - s.shape[1]) / 2, y)
            y += s.shape[0] + gap
        return out

    # ---- brand bar
    def topbar(self, frame, t):
        a = max(window(t, a0, a1, 0.25, 0.12) for a0, a1 in TOPBAR)
        if a <= 0:
            return
        blit(frame, self.brand, 540, 168, alpha=a)
        blit(frame, self.brand_rule, 540, 196, alpha=a * 0.9)
        blit(frame, self.brand_date, 540, 222, alpha=a * 0.95)

    # ---- chips
    def chips_draw(self, frame, t):
        first = [c for c in CHIPS if c[2] == 0 and c[0] - 0.05 <= t < c[1] and not c[3].startswith("FORMULE")]
        if first:
            a = window(t, first[0][0], first[0][1], 0.2, 0.15)
            blit(frame, self.chip_label, 62, 300, alpha=a * 0.95, anchor="l")
        for t_in, t_out, row, txt, plus, off in CHIPS:
            if not (t_in <= t < t_out):
                continue
            p = prog(t, t_in, 0.30)
            out = prog(t, t_out - 0.15, 0.15)
            x = 40 - 70 * (1 - ease_out(p)) - 40 * out
            y = 362 + row * 90
            blit(frame, self.chips[txt], x, y, alpha=min(1, p * 2.2) * (1 - out), anchor="l", scale=0.92 + 0.08 * ease_back(p))

    # ---- particles
    def particles(self, frame, t, strength=1.0, y_max=1920):
        if strength <= 0:
            return
        for (x, y, s, sp, ph, a) in self.parts:
            yy = (y - t * sp) % 1980 - 30
            if yy > y_max:
                continue
            xx = x + 14 * math.sin(t * 0.8 + ph)
            tw = 0.55 + 0.45 * math.sin(t * 2.2 + ph * 3)
            blit(frame, self.dots[s], xx, yy, alpha=a * tw * strength * 0.75, additive=True)

    # ---- supercut type
    def supercut(self, frame, t):
        lt = t - 4.25
        if 0 <= lt < 1.0:
            p = ease_out(prog(lt, 0.0, 0.45))
            a = min(1, lt / 0.12) * (1 - prog(lt, 0.9, 0.1))
            blit(frame, self.sc_kicker, 540, 820, alpha=a * ease_out(prog(lt, 0.15, 0.3)))
            blit(frame, self.rule_w, 540, 866, alpha=a * 0.9, scale=0.4 + 0.6 * p)
            blit(frame, self.sc_two, 540, 1000, alpha=a, scale=1.10 - 0.10 * p)
        elif 1.0 <= lt < 2.0:
            l2 = lt - 1.0
            p = ease_back(prog(l2, 0.0, 0.32), 1.6)
            a = min(1, l2 / 0.08) * (1 - prog(l2, 0.9, 0.1))
            blit(frame, self.glow, 540, 1000, alpha=0.45 * a * ease_out(prog(l2, 0, 0.3)), additive=True, scale=1.2)
            blit(frame, self.sc_la, 540, 760, alpha=a * ease_out(prog(l2, 0.1, 0.25)))
            sp45 = shine(self.sc_45, prog(l2, 0.35, 0.5))
            blit(frame, sp45, 470, 1010, alpha=a, scale=1.35 - 0.35 * p)
            blit(frame, self.sc_dh, 470 + 245, 1095, alpha=a * ease_out(prog(l2, 0.12, 0.2)), scale=1.2 - 0.2 * p, anchor="l")
        elif 2.0 <= lt < 3.5:
            l3 = lt - 2.0
            p = ease_out(prog(l3, 0.0, 0.45))
            a = min(1, l3 / 0.1) * (1 - prog(l3, 1.38, 0.12))
            blit(frame, self.sc_week, 540, 880, alpha=a, scale=1.08 - 0.08 * p)
            blit(frame, self.sc_only, 540, 1040, alpha=a * ease_out(prog(l3, 0.15, 0.3)))
            blit(frame, self.rule_w, 540, 1100, alpha=a * 0.9, scale=0.3 + 0.7 * ease_out(prog(l3, 0.3, 0.5)))
            blit(frame, self.sc_dates, 540, 1150, alpha=a * ease_out(prog(l3, 0.45, 0.35)))

    # ---- price card
    def card(self, frame, t, key):
        seg = SEG[key]
        lt = t - seg["T0"]
        slam = SLAMS[key] - seg["T0"]
        c = self.cards[key]
        out = 1 - prog(t, seg["T1"] - 0.12, 0.12)
        self.particles(frame, t, strength=0.8 * ease_out(prog(lt, 0.1, 0.5)) * out)
        a = ease_out(prog(lt, 0.05, 0.3)) * out
        blit(frame, c["title"], 540, 955, alpha=a)
        blit(frame, c["rule"], 540 - 330, 955, alpha=a * 0.9)
        blit(frame, c["rule"], 540 + 330, 955, alpha=a * 0.9)
        for i, l in enumerate(c["lines"]):
            ai = ease_out(prog(lt, 0.12 + 0.1 * i, 0.3)) * out
            blit(frame, l, 540, 1030 + i * 58 + 18 * (1 - ai), alpha=ai)
        # price slam
        if lt >= slam - 0.02:
            ls = lt - slam
            p = ease_back(prog(ls, 0, 0.30), 1.8)
            ap = min(1, ls / 0.06 + 0.2) * out
            blit(frame, self.glow, 540, 1330, alpha=0.55 * ap * (1 - 0.5 * prog(ls, 0.2, 0.8)), additive=True,
                 scale=0.6 + 0.8 * ease_out(prog(ls, 0, 0.5)))
            sp = shine(self.card_45, prog(ls, 0.30, 0.55))
            sc = 1.7 - 0.7 * p
            blit(frame, sp, 480, 1330, alpha=ap, scale=sc)
            blit(frame, self.card_dh, 480 + 228 * sc, 1408, alpha=ap * ease_out(prog(ls, 0.1, 0.2)), scale=sc, anchor="l")
            blit(frame, self.card_date, 540, 1535, alpha=ease_out(prog(ls, 0.35, 0.3)) * out)

    # ---- end card
    def end(self, frame, t):
        lt = t - END_T
        lb = t - STEP_B
        out = 1 - prog(t, CTA_T - 0.12, 0.12)
        self.particles(frame, t, strength=1.0 * out)

        def ap(t0, d=0.3):
            return ease_out(prog(lt, t0, d)) * out

        blit(frame, self.e_brand, 540, 232, alpha=ap(0.08), scale=1.06 - 0.06 * ap(0.08, 0.5))
        blit(frame, self.e_rule, 540, 296, alpha=ap(0.2) * 0.9, scale=0.3 + 0.7 * ap(0.2, 0.5))
        a2 = ap(0.30)
        blit(frame, self.e_two, 540, 410, alpha=a2, scale=1.05 - 0.05 * a2)
        a3 = ease_back(prog(lt, 0.45, 0.32), 1.8) * out
        blit(frame, self.e_a, 330, 572, alpha=min(1, a3 * 2))
        blit(frame, shine(self.e_45, prog(lt, 0.8, 0.6)), 600, 560, alpha=min(1, a3 * 2), scale=1.25 - 0.25 * a3)
        # step A: the two formulas
        gone = ease_io(prog(lb, -0.15, 0.3))
        for i, crd in enumerate(self.e_cards):
            p = ease_out(prog(lt, 0.60 + 0.30 * i, 0.38))
            x = 540 + 90 * (1 - p) * (1 if i % 2 else -1) - 700 * gone * (1 if i % 2 else -1)
            blit(frame, crd, x, 830 + i * 290, alpha=p * out * (1 - gone))
        # step B: how to order
        if lb >= 0:
            pb = ease_back(prog(lb, 0.05, 0.35), 1.6)
            blit(frame, self.e_order, 540, 780, alpha=ease_out(prog(lb, 0.0, 0.3)) * out)
            blit(frame, self.e_phone_box, 540, 900, alpha=min(1, pb * 2) * out, scale=0.85 + 0.15 * pb)
            blit(frame, shine(self.e_phone, prog(lb, 0.5, 0.6)), 540, 900, alpha=min(1, pb * 2) * out, scale=0.85 + 0.15 * pb)
            pd = ease_out(prog(lb, 0.35, 0.35))
            blit(frame, self.e_deliv, 540 - 40 * (1 - pd), 1080, alpha=pd * out)
            pa = ease_out(prog(lb, 0.6, 0.35))
            blit(frame, self.e_addr, 540 + 40 * (1 - pa), 1170, alpha=pa * out)
        p = ease_back(prog(lt, 1.35, 0.35), 1.6)
        blit(frame, self.e_week, 540, 1352, alpha=ap(1.35))
        blit(frame, self.e_datebox, 540, 1440, alpha=min(1, p * 2) * out, scale=0.9 + 0.1 * p)
        blit(frame, shine(self.e_dates, prog(lt, 1.8, 0.6)), 540, 1440, alpha=min(1, p * 2) * out, scale=0.9 + 0.1 * p)

    def cta_draw(self, frame, t):
        lt = t - CTA_T
        a = ease_out(prog(lt, 0.0, 0.3))
        blit(frame, self.cta_brand, 540, 250, alpha=a)
        blit(frame, self.e_rule, 540, 305, alpha=a * 0.9)
        p = ease_back(prog(lt, 0.2, 0.4), 2.0)
        blit(frame, self.cta, 540, 1180, alpha=min(1, p * 2), scale=0.6 + 0.4 * p, rot=4 * (1 - p) + 3)
        blit(frame, self.cta_phone, 540, 1325, alpha=ease_out(prog(lt, 0.45, 0.35)))
        blit(frame, self.cta_sub, 540, 1420, alpha=ease_out(prog(lt, 0.6, 0.35)))

    def timeskip(self, frame, t):
        lt = t - SKIP0
        a = min(1, lt / 0.15) * (1 - prog(t, SKIP1 - 0.12, 0.12))
        blit(frame, self.ts_ff, 120, 330, alpha=a * (0.6 + 0.4 * (int(lt * 6) % 2)))
        p = ease_back(prog(lt, 0.05, 0.35), 2.0)
        blit(frame, self.ts_clock, 540, 760, alpha=a, scale=0.6 + 0.4 * p, rot=12 * math.sin(lt * 40) * (1 - prog(lt, 0.4, 0.5)))
        blit(frame, self.ts_text, 540, 990, alpha=a * ease_out(prog(lt, 0.15, 0.3)), scale=0.92 + 0.08 * ease_out(prog(lt, 0.15, 0.4)))

    def freeze(self, frame, t):
        lt = t - FREEZE[2]
        blit(frame, self.lines, 540, 960, alpha=0.38 * ease_out(prog(lt, 0.02, 0.12)) * (0.85 + 0.15 * math.sin(lt * 50)))
        p = ease_back(prog(lt, 0.12, 0.3), 2.4)
        blit(frame, self.angry, 820, 470, alpha=min(1, p * 2), scale=0.3 + 0.7 * p, rot=-14 + 8 * math.sin(lt * 18))


# ------------------------------------------------------------------ per-frame compositing
def cam_for(seg, tl):
    key = seg["key"]
    dur = seg["T1"] - seg["T0"]
    if "photo" in seg:
        z0, z1 = seg["zx"]
        return z0 + (z1 - z0) * ease_io(tl / dur), seg["foc"][0], seg["foc"][1], 0.0
    if key in CAM:
        z0, z1, fx, fy, punch = CAM[key]
    elif key in SUPER:
        z0, z1, fx, fy, punch = 1.04, 1.12, 0.5, 0.55, 0.06
    else:
        z0, z1, fx, fy, punch = 1.04, 1.11, 0.5, 0.55, 0.04
    z = z0 + (z1 - z0) * (ease_out(tl / dur) if key not in HEADS else tl / dur)
    if punch and tl < 0.22:
        z *= 1 + punch * (1 - ease_out(tl / 0.22))
    return z, fx, fy, punch


def compose(g, seg, raw, t, fi):
    key = seg["key"]
    tl = t - seg["T0"]
    z, fx, fy, _ = cam_for(seg, tl)
    dx = dy = 0.0
    if "photo" in seg:
        ls = t - SLAMS[key]
        if 0 <= ls < 0.3:
            k = (1 - ls / 0.3) * 14 * U
            dx, dy = k * math.sin(ls * 90), k * math.cos(ls * 77)
    if key == FREEZE[0]:                                  # comic freeze frame: punch-in + shake
        z = 1.0 + 0.14 * ease_out(prog(tl, 0.0, 0.45))
        fx, fy = 0.42, 0.33
        if tl < 0.3:
            k = (1 - tl / 0.3) * 18 * U
            dx, dy = k * math.sin(tl * 110), k * math.cos(tl * 93)
    if key == "v17g":                                     # he shoves the camera: extra shake
        k = 10 * U
        dx, dy = k * math.sin(t * 61), k * math.cos(t * 47)
    frame = warp(raw, z, fx, fy, dx, dy)
    if key == "v16":                                      # fast-forward look: desaturated, scanlines, chroma split
        g_ = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 0.45, g_, 0.55, 0)
        frame = cv2.multiply(frame, g.scan, dtype=cv2.CV_8U)
        sh = int(9 * U)
        frame[:, sh:, 0] = frame[:, :-sh, 0]
        frame = cv2.addWeighted(frame, 0.62, np.full_like(frame, INK), 0.38, 0)
    if key == "v17g":
        r_ = np.random.default_rng(fi)
        sh = int(r_.uniform(8, 26) * U)
        frame[:, sh:, 0] = frame[:, :-sh, 0]
        frame[:, :-sh, 2] = frame[:, sh:, 2]
        for _ in range(4):
            y0 = int(r_.uniform(0, H - 200 * U))
            hh = int(r_.uniform(20, 120) * U)
            frame[y0:y0 + hh] = np.roll(frame[y0:y0 + hh], int(r_.uniform(-60, 60) * U), axis=1)
    if key == FREEZE[0]:
        g_ = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 0.7, g_, 0.3, 0)
        if tl < 0.08:
            frame = cv2.addWeighted(frame, 1.0, np.full_like(frame, 255), 0.6 * (1 - tl / 0.08), 0)
    if key == "v14":                                     # end card: soft, dark, warm backdrop
        sm = cv2.resize(frame, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), 7)
        frame = cv2.resize(sm, (W, H), interpolation=cv2.INTER_CUBIC)
        frame = cv2.addWeighted(frame, 0.42, np.full_like(frame, CHOC), 0.50, 0)
        frame = apply(frame, g.masks["vig"])
    # whips
    for tw in WHIP_AT:
        if 0 <= tw - t < 0.17:
            frame = zoom_blur(frame, 1 - (tw - t) / 0.17 + 0.15, direction=1)
        elif 0 <= t - tw < 0.2:
            frame = zoom_blur(frame, 1 - (t - tw) / 0.2, direction=-0.6)
    # looks
    if key in SUPER:
        frame = apply(frame, g.masks["super"])
        for tc in SUPER_CUTS[1:]:
            if 0 <= t - tc < 0.1:
                frame = cv2.addWeighted(frame, 1.0, frame, 0.35 * (1 - (t - tc) / 0.1), 0)
    elif "photo" in seg:
        frame = apply(frame, g.masks["card"])
    elif key == "v15":
        frame = apply(frame, g.masks["cta"])
    elif key in (FREEZE[0], "v16"):
        frame = apply(frame, g.masks["vig"])
    elif key != "v14":
        frame = apply(frame, g.masks["head"])
    # graphics
    if key in SUPER:
        g.supercut(frame, t)
    elif "photo" in seg:
        g.card(frame, t, key)
    elif key == "v14":
        g.end(frame, t)
    elif key == "v15":
        g.cta_draw(frame, t)
    elif key == "v16":
        g.timeskip(frame, t)
    elif key == FREEZE[0]:
        g.freeze(frame, t)
    g.topbar(frame, t)
    g.chips_draw(frame, t)
    # fade from / to black at the very edges
    if t > TOTAL - 0.35:
        frame = cv2.addWeighted(frame, max(0.0, (TOTAL - t) / 0.35), frame, 0, 0)
    return g.grain.apply(frame, fi)


def seg_at(t):
    for s in ORDER:
        if s["T0"] - 1e-6 <= t < s["T1"] - 1e-6:
            return s
    return ORDER[-1]


def render(out, audio, stills, still_dir, t0=0.0, t1=TOTAL):
    g = GFX()
    if stills:
        os.makedirs(still_dir, exist_ok=True)
        for ts in stills:
            seg = seg_at(ts)
            fi = int(round(ts * FPS))
            k = fi - int(round(seg["T0"] * FPS))
            if "photo" in seg:
                raw = photo_canvas(seg["photo"], seg["py"])
            elif "freeze_of" in seg:
                raw = frozen_frame(seg)
            else:
                r = VideoReader(seg["key"], max(0, k))
                raw = r.read().copy()
                r.close()
            fr = compose(g, seg, raw, fi / FPS, fi)
            small = cv2.resize(fr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f"{still_dir}/t{ts:05.2f}.png", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
            print("still", ts, seg["key"], flush=True)
        return
    nframes = int(round(TOTAL * FPS))
    f0, f1 = int(round(t0 * FPS)), min(nframes, int(round(t1 * FPS)))
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if audio:
        cmd += ["-ss", f"{t0:.3f}", "-i", audio]
    cmd += ["-map", "0:v"] + (["-map", "1:a"] if audio else [])
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "15", "-maxrate", "60M", "-bufsize", "120M", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "5.2",
            "-x264-params", "aq-mode=3:deblock=-1,-1", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-movflags", "+faststart", "-r", str(FPS)]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-shortest"]
    cmd += [out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for seg in ORDER:
        s0, s1 = int(round(seg["T0"] * FPS)), int(round(seg["T1"] * FPS))
        if s1 <= f0 or s0 >= f1:
            continue
        a0 = max(s0, f0)
        still = "photo" in seg or "freeze_of" in seg
        reader = None if still else VideoReader(seg["key"], a0 - s0)
        for fi in range(a0, min(s1, f1)):
            if reader is not None:
                raw = reader.read()
            elif "photo" in seg:
                raw = photo_canvas(seg["photo"], seg["py"])
            else:
                raw = frozen_frame(seg)
            fr = compose(g, seg, raw, fi / FPS, fi)
            enc.stdin.write(fr.tobytes())
            if fi % 30 == 0:
                print(f"frame {fi}/{nframes} {seg['key']}", flush=True)
        if reader:
            reader.close()
    enc.stdin.close()
    enc.wait()
    print("wrote", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="")
    ap.add_argument("--audio", default="")
    ap.add_argument("--stills", default="")
    ap.add_argument("--still-dir", default=f"{S}/stills")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=TOTAL)
    a = ap.parse_args()
    st = [float(x) for x in a.stills.split(",")] if a.stills else None
    render(a.out, a.audio, st, a.still_dir, a.t0, a.t1)
