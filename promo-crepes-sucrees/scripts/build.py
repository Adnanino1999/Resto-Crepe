"""Resto Crêpes – promo ad compositor (9:16, 1080x1920, 30 fps).

Usage: build.py OUT.mp4 [--t0 S] [--t1 S] [--scale F] [--stills DIR t,t,t]
Reads source clips with ffmpeg, composites graphics with numpy/OpenCV/PIL,
pipes raw frames to an ffmpeg encoder.
"""
import argparse
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
ROOT = "/home/user/work"
CLIPS = f"{ROOT}/clips/Crepe"
FONTS = f"{ROOT}/fonts"

CREAM = (247, 240, 226)
GOLD = (236, 186, 92)
GOLD_HI = (255, 222, 150)
CHOC = (30, 16, 10)
CHOC2 = (58, 32, 18)
PISTA = (168, 200, 88)
RED = (226, 64, 48)

GRADE = ("hqdn3d=1.5:1.5:4:4,"
         "eq=contrast=1.07:saturation=1.16:brightness=0.012,"
         "colorbalance=rm=0.025:bm=-0.035:rh=0.015:bh=-0.02,"
         "curves=m='0/0 0.25/0.225 0.75/0.79 1/1',"
         "unsharp=5:5:0.55:3:3:0.0")


# ------------------------------------------------------------------ easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(p):
    p = clamp(p)
    return 1 - (1 - p) ** 3


def ease_in(p):
    p = clamp(p)
    return p ** 3


def ease_io(p):
    p = clamp(p)
    return p * p * (3 - 2 * p)


def ease_back(p, s=1.9):
    p = clamp(p)
    p -= 1
    return p * p * ((s + 1) * p + s) + 1


def prog(t, start, dur):
    return clamp((t - start) / dur) if dur > 0 else (1.0 if t >= start else 0.0)


# ------------------------------------------------------------------ fonts / sprites
_fc = {}


def font(name, size):
    k = (name, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(f"{FONTS}/{name}.ttf", size)
    return _fc[k]


def text_size(txt, f, tracking=0):
    w = 0
    for ch in txt:
        w += f.getlength(ch) + tracking
    w -= tracking
    asc, desc = f.getmetrics()
    return int(math.ceil(w)), asc + desc


def text_sprite(txt, fname, size, fill, tracking=0, shadow=0.75, shadow_blur=10, shadow_off=(0, 5),
                stroke=0, stroke_fill=CHOC, gradient=None):
    """RGBA sprite of a single text line with soft drop shadow. Returns np.uint8 HxWx4."""
    f = font(fname, size)
    tw, th = text_size(txt, f, tracking)
    pad = shadow_blur * 3 + stroke + 8
    img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    mask = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(mask)
    x = pad
    for ch in txt:
        d.text((x, pad), ch, font=f, fill=255, stroke_width=stroke)
        x += f.getlength(ch) + tracking
    if shadow > 0:
        sh = mask.filter(ImageFilter.GaussianBlur(shadow_blur))
        sh = Image.eval(sh, lambda v: int(min(255, v * 1.6) * shadow))
        shadow_img = Image.new("RGBA", img.size, (12, 6, 3, 0))
        shadow_img.putalpha(sh)
        img.alpha_composite(shadow_img, shadow_off)
    if stroke:
        st = Image.new("RGBA", img.size, stroke_fill + (0,))
        st.putalpha(mask)
        img.alpha_composite(st)
        mask = Image.new("L", img.size, 0)
        d = ImageDraw.Draw(mask)
        x = pad
        for ch in txt:
            d.text((x, pad), ch, font=f, fill=255)
            x += f.getlength(ch) + tracking
    if gradient:
        top, bot = gradient
        g = np.linspace(0, 1, img.size[1])[:, None, None]
        col = (np.array(top)[None, None, :] * (1 - g) + np.array(bot)[None, None, :] * g)
        col = np.broadcast_to(col, (img.size[1], img.size[0], 3)).astype(np.uint8)
        fg = Image.fromarray(np.dstack([col, np.array(mask)]), "RGBA")
    else:
        fg = Image.new("RGBA", img.size, tuple(fill) + (0,))
        fg.putalpha(mask)
    img.alpha_composite(fg)
    return np.array(img), pad


def pill_sprite(txt, fname, size, fg, bg, tracking=3, padx=26, pady=12, radius=None, outline=None):
    f = font(fname, size)
    tw, th = text_size(txt, f, tracking)
    w, h = tw + 2 * padx, th + 2 * pady
    S = 3
    pad = 24
    big = Image.new("RGBA", ((w + 2 * pad) * S, (h + 2 * pad) * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    r = (radius if radius is not None else h // 2) * S
    # shadow
    sh = Image.new("L", big.size, 0)
    ImageDraw.Draw(sh).rounded_rectangle([pad * S, (pad + 5) * S, (pad + w) * S, (pad + h + 5) * S], r, fill=150)
    sh = sh.filter(ImageFilter.GaussianBlur(9 * S))
    shim = Image.new("RGBA", big.size, (10, 5, 2, 0))
    shim.putalpha(sh)
    big.alpha_composite(shim)
    d.rounded_rectangle([pad * S, pad * S, (pad + w) * S, (pad + h) * S], r, fill=tuple(bg) + (255,),
                        outline=(tuple(outline) + (255,)) if outline else None, width=3 * S)
    img = big.resize((w + 2 * pad, h + 2 * pad), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    x = pad + padx
    for ch in txt:
        d.text((x, pad + pady), ch, font=f, fill=tuple(fg) + (255,))
        x += f.getlength(ch) + tracking
    return np.array(img), pad


def badge_sprite(diam=300):
    """Scalloped gold promo seal with '−5 DH'."""
    S = 3
    pad = 30
    size = (diam + 2 * pad) * S
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    c = size / 2
    R = diam * S / 2
    pts = []
    n = 28
    for i in range(n * 2):
        a = math.pi * i / n
        r = R if i % 2 == 0 else R * 0.92
        pts.append((c + r * math.cos(a), c + r * math.sin(a)))
    sh = Image.new("L", img.size, 0)
    ImageDraw.Draw(sh).polygon([(x, y + 8 * S) for x, y in pts], fill=170)
    sh = sh.filter(ImageFilter.GaussianBlur(10 * S))
    shim = Image.new("RGBA", img.size, (10, 5, 2, 0))
    shim.putalpha(sh)
    img.alpha_composite(shim)
    # radial-ish gold gradient
    yy, xx = np.mgrid[0:size, 0:size]
    g = clamp_arr(((xx - c * 0.75) ** 2 + (yy - c * 0.7) ** 2) ** 0.5 / (R * 1.5))
    col = np.array(GOLD_HI)[None, None, :] * (1 - g[..., None]) + np.array((206, 148, 52))[None, None, :] * g[..., None]
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    disc = Image.fromarray(np.dstack([col.astype(np.uint8), np.array(m)]), "RGBA")
    img.alpha_composite(disc)
    d = ImageDraw.Draw(img)
    d.ellipse([c - R * 0.82, c - R * 0.82, c + R * 0.82, c + R * 0.82], outline=CHOC + (255,), width=int(2.2 * S))
    f1 = font("Montserrat-Black", int(diam * 0.40) * S)
    f2 = font("Montserrat-ExtraBold", int(diam * 0.17) * S)
    t1 = "−5"
    w1 = f1.getlength(t1)
    d.text((c - w1 / 2 - 4 * S, c - diam * 0.33 * S), t1, font=f1, fill=CHOC + (255,))
    t2 = "DH"
    w2 = f2.getlength(t2)
    d.text((c - w2 / 2, c + diam * 0.115 * S), t2, font=f2, fill=CHOC + (255,))
    img = img.resize((diam + 2 * pad, diam + 2 * pad), Image.LANCZOS)
    return np.array(img), pad


def clamp_arr(a):
    return np.clip(a, 0, 1)


def rect_sprite(w, h, color, radius=0, alpha=255):
    S = 3
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius * S, fill=tuple(color) + (alpha,))
    return np.array(img.resize((w, h), Image.LANCZOS))


def blit(frame, spr, x, y, alpha=1.0, scale=1.0, rot=0.0, anchor="tl", additive=False):
    """Alpha-blend RGBA sprite on RGB frame. (x, y) refers to the anchor point."""
    if alpha <= 0.003:
        return
    sh, sw = spr.shape[:2]
    if scale != 1.0 or rot != 0.0:
        nw, nh = max(2, int(round(sw * scale))), max(2, int(round(sh * scale)))
        if rot != 0.0:
            diag = int(math.ceil(math.hypot(nw, nh))) + 2
            M = cv2.getRotationMatrix2D((sw / 2, sh / 2), rot, scale)
            M[0, 2] += diag / 2 - sw / 2
            M[1, 2] += diag / 2 - sh / 2
            spr = cv2.warpAffine(spr, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
        else:
            spr = cv2.resize(spr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        sh, sw = spr.shape[:2]
    if anchor == "c":
        x -= sw / 2
        y -= sh / 2
    elif anchor == "tc":
        x -= sw / 2
    elif anchor == "tr":
        x -= sw
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + sw), min(frame.shape[0], y + sh)
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


# ------------------------------------------------------------------ sources
class VideoSrc:
    def __init__(self, clip, ss, speed, n, slow=False, extra=""):
        vf = []
        if speed != 1.0:
            vf.append(f"setpts=(PTS-STARTPTS)/{speed}")
        if slow and speed < 1.0:
            vf.append("minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
        else:
            vf.append("fps=30")
        vf.append("scale=1080:1934:flags=lanczos,crop=1080:1920")
        vf.append(GRADE)
        if extra:
            vf.append(extra)
        dur_src = n / FPS * speed + 0.6
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{dur_src:.3f}", "-i",
               f"{CLIPS}/IMG_{clip}.MOV", "-an", "-vf", ",".join(vf), "-frames:v", str(n),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
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
        except Exception:
            pass


class PhotoSrc:
    """Still photo placed on a canvas (scale s, offsets) – returns the same canvas each frame."""

    def __init__(self, path, s, offx, offy):
        im = Image.open(path).convert("RGB")
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
        canvas = Image.new("RGB", (W, H), CHOC)
        canvas.paste(im, (offx, offy))
        arr = np.array(canvas).astype(np.float32)
        # match the grade: warmth + contrast + saturation
        arr = (arr - 128) * 1.08 + 128 + 3
        g = arr.mean(axis=2, keepdims=True)
        arr = g + (arr - g) * 1.18
        arr[..., 0] *= 1.03
        arr[..., 2] *= 0.96
        self.img = np.clip(arr, 0, 255).astype(np.uint8)

    def read(self):
        return self.img

    def close(self):
        pass


def warp(frame, z, fx, fy, dx=0.0, dy=0.0, rot=0.0):
    """Zoom by z around focus (fx, fy in 0..1), then translate by (dx, dy) px."""
    cx, cy = fx * W, fy * H
    M = cv2.getRotationMatrix2D((cx, cy), rot, z)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def zoom_blur(frame, p, direction=1):
    """Radial zoom-blur for whip transitions. p in 0..1 strength."""
    if p <= 0.001:
        return frame
    acc = np.zeros(frame.shape, np.float32)
    k = 6
    for i in range(k):
        z = 1 + direction * (0.55 * p * p) + 0.045 * p * i
        acc += warp(frame, z, 0.5, 0.5)
    return (acc / k).astype(np.uint8)


# ------------------------------------------------------------------ masks
def make_masks():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W * 0.78)) ** 2 + ((yy - H / 2) / (H * 0.72)) ** 2)
    vig = 1 - 0.42 * np.clip(r - 0.35, 0, 1) ** 1.6
    y = np.arange(H, dtype=np.float32)
    bottom = 1 - 0.62 * np.clip((y - 980) / 520, 0, 1) ** 1.3
    top = 1 - 0.50 * np.clip((430 - y) / 430, 0, 1) ** 1.4
    prep = vig * (bottom * top)[:, None]
    hook = vig * (1 - 0.30 * np.clip((y - 300) / 500, 0, 1) * np.clip((1500 - y) / 400, 0, 1))[:, None] * top[:, None]
    return {"vig": vig[..., None], "prep": prep[..., None], "hook": hook[..., None]}


def apply_mask(frame, m):
    return cv2.multiply(frame, np.broadcast_to(m, frame.shape).astype(np.float32), dtype=cv2.CV_8U)


def panel(frame, y0, y1, color=CHOC, max_a=1.0):
    """Blend frame into a solid colour from y0 (0%) to y1 (max) and below."""
    y = np.arange(H, dtype=np.float32)
    a = np.clip((y - y0) / (y1 - y0), 0, 1)
    a = (a * a * (3 - 2 * a)) * max_a
    a = a[:, None, None]
    col = np.array(color, np.float32)[None, None, :]
    return (frame.astype(np.float32) * (1 - a) + col * a).astype(np.uint8)


def flash(frame, a, color=(255, 244, 222)):
    if a <= 0.003:
        return frame
    col = np.array(color, np.float32)[None, None, :]
    return (frame.astype(np.float32) * (1 - a) + col * a).astype(np.uint8)


# ------------------------------------------------------------------ sparkles
class Sparkles:
    def __init__(self, n=46, seed=5, region=(60, 1020, 250, 1500)):
        r = np.random.default_rng(seed)
        self.x = r.uniform(region[0], region[1], n)
        self.y = r.uniform(region[2], region[3], n)
        self.v = r.uniform(25, 90, n)
        self.ph = r.uniform(0, 6.28, n)
        self.fr = r.uniform(1.5, 4.5, n)
        self.sz = r.integers(0, 4, n)
        self.sw = r.uniform(-18, 18, n)
        self.region = region
        self.dots = []
        for d in (10, 16, 26, 40):
            yy, xx = np.mgrid[0:d, 0:d].astype(np.float32)
            g = np.exp(-(((xx - d / 2 + .5) ** 2 + (yy - d / 2 + .5) ** 2) / (2 * (d / 5.2) ** 2)))
            spr = np.zeros((d, d, 4), np.uint8)
            spr[..., 0], spr[..., 1], spr[..., 2] = 255, 214, 140
            spr[..., 3] = (g * 255).astype(np.uint8)
            self.dots.append(spr)

    def draw(self, frame, t, strength=1.0):
        if strength <= 0.01:
            return
        x0, x1, y0, y1 = self.region
        span = y1 - y0
        for i in range(len(self.x)):
            y = y1 - ((self.y[i] - y0 + self.v[i] * t) % span)
            x = self.x[i] + self.sw[i] * math.sin(t * 1.3 + self.ph[i])
            tw = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * self.fr[i] + self.ph[i])) ** 2
            edge = min(1.0, (y - y0) / 120, (y1 - y) / 120)
            blit(frame, self.dots[self.sz[i]], x, y, alpha=strength * tw * max(0, edge) * 0.85, anchor="c", additive=True)


# ------------------------------------------------------------------ timeline
# out_t0, out_t1, kind, params...  (zoom z0->z1 around focus)
SEGS = [
    # ---- HOOK
    dict(t0=0.0, t1=1.2, clip="9955", ss=2.6, speed=0.6, slow=True, z=(1.10, 1.22), f=(0.50, 0.70), look="hook"),
    dict(t0=1.2, t1=2.0, clip="9958", ss=6.3, speed=0.55, slow=True, z=(1.04, 1.12), f=(0.50, 0.72), look="hook"),
    # ---- KOUNAFA PISTACHE
    dict(t0=2.0, t1=3.0, clip="9980", ss=4.0, speed=1.2, z=(1.00, 1.12), f=(0.42, 0.40), look="prep", tin="whip"),
    dict(t0=3.0, t1=4.0, clip="9980", ss=11.6, speed=1.8, z=(1.06, 1.16), f=(0.62, 0.52), look="prep"),
    dict(t0=4.0, t1=5.0, clip="9949", ss=7.0, speed=1.4, z=(1.08, 1.20), f=(0.50, 0.58), look="prep"),
    dict(t0=5.0, t1=6.0, clip="9950", ss=7.9, speed=1.3, z=(1.08, 1.20), f=(0.50, 0.55), look="prep"),
    dict(t0=6.0, t1=7.0, clip="9952", ss=3.0, speed=1.6, z=(1.04, 1.18), f=(0.50, 0.66), look="prep"),
    dict(t0=7.0, t1=8.5, clip="9955", ss=4.4, speed=0.9, z=(1.06, 1.22), f=(0.50, 0.66), look="prep"),
    dict(t0=8.5, t1=9.5, clip="9957", ss=15.3, speed=1.4, z=(1.08, 1.20), f=(0.50, 0.72), look="prep"),
    dict(t0=9.5, t1=10.0, clip="9968", ss=0.45, speed=1.0, z=(1.12, 1.26), f=(0.42, 0.50), look="prep"),
    dict(t0=10.0, t1=12.0, photo="photo_6035243488808473032_w.jpg", ps=0.70, poff=(-264, -660),
         z=(1.00, 1.07), f=(0.50, 0.36), look="heroK", tin="flash"),
    # ---- 4 YOU
    dict(t0=12.0, t1=12.5, clip="9971", ss=0.3, speed=2.4, z=(1.04, 1.12), f=(0.50, 0.70), look="prep", tin="whip"),
    dict(t0=12.5, t1=13.0, clip="9975", ss=6.15, speed=2.0, z=(1.04, 1.12), f=(0.50, 0.65), look="prep"),
    dict(t0=13.0, t1=13.5, clip="9977", ss=12.6, speed=2.4, z=(1.04, 1.12), f=(0.55, 0.60), look="prep"),
    dict(t0=13.5, t1=14.5, clip="9982", ss=0.3, speed=2.0, z=(1.06, 1.18), f=(0.45, 0.62), look="prep"),
    dict(t0=14.5, t1=15.25, clip="9985", ss=45.0, speed=4.0, z=(1.05, 1.12), f=(0.50, 0.55), look="prep"),
    dict(t0=15.25, t1=16.0, clip="9988", ss=16.5, speed=4.0, z=(1.05, 1.14), f=(0.50, 0.70), look="prep"),
    dict(t0=16.0, t1=17.0, clip="9993", ss=0.7, speed=1.8, z=(1.06, 1.20), f=(0.45, 0.72), look="prep"),
    dict(t0=17.0, t1=18.5, clip="9994", ss=3.9, speed=0.5, slow=True, z=(1.02, 1.12), f=(0.50, 0.50), look="prep"),
    dict(t0=18.5, t1=20.0, clip="9992", ss=0.6, speed=1.1, z=(1.04, 1.16), f=(0.50, 0.70), look="prep"),
    dict(t0=20.0, t1=22.0, clip="9995", ss=0.95, speed=0.8, z=(1.00, 1.08), f=(0.50, 0.80), shift=(0, -800),
         look="heroF", tin="flash"),
    # ---- MENU + END CARD backgrounds
    dict(t0=22.0, t1=26.0, clip="9958", ss=8.2, speed=0.5, z=(1.10, 1.22), f=(0.50, 0.75), look="menu", tin="whip",
         extra="gblur=sigma=14"),
    dict(t0=26.0, t1=29.5, clip="9994", ss=4.7, speed=0.4, z=(1.04, 1.14), f=(0.50, 0.55), look="end", tin="flash",
         extra="gblur=sigma=5"),
]
TOTAL = 29.5

CAPTIONS = [  # start, end, text
    (2.0, 4.0, "PÂTE FINE, PRÉPARÉE MINUTE"),
    (4.0, 5.0, "KOUNAFA CROUSTILLANTE"),
    (5.0, 6.0, "PISTACHE GÉNÉREUSE"),
    (6.0, 7.0, "CHOCOLAT FONDANT"),
    (7.0, 8.5, "FINITION À LA MAIN"),
    (8.5, 10.0, "ÉCLATS DE PISTACHE"),
    (12.0, 13.5, "FRUITS FRAIS, COUPÉS MINUTE"),
    (13.5, 14.5, "CRÊPE DORÉE"),
    (14.5, 16.0, "DRESSAGE DU CHEF"),
    (16.0, 17.0, "FILET DE CHOCOLAT"),
    (17.0, 18.5, "LA TOUCHE FINALE"),
    (18.5, 20.0, "RECETTE DE LA MAISON"),
]


# ------------------------------------------------------------------ graphics
class GFX:
    def __init__(self):
        T = text_sprite
        self.masks = make_masks()
        self.sparks = Sparkles()
        self.sparks_end = Sparkles(n=60, seed=11, region=(40, 1040, 200, 1700))
        # persistent top
        self.brand_sm = T("RESTO CRÊPES", "Montserrat-ExtraBold", 30, CREAM, tracking=9, shadow=0.8, shadow_blur=6)
        self.top_pill = pill_sprite("−5 DH  •  JUSQU'AU 22 NOVEMBRE", "Montserrat-ExtraBold", 32, CHOC, GOLD,
                                    tracking=2, padx=30, pady=13)
        # hook
        self.hk_small = T("OFFRE EXCLUSIVE", "Montserrat-ExtraBold", 40, GOLD, tracking=12, shadow=0.9, shadow_blur=8)
        self.hk_l1 = T("Crêpes", "Playfair-Black", 190, CREAM, shadow=0.85, shadow_blur=16, shadow_off=(0, 8))
        self.hk_l2 = T("Sucrées", "Playfair-Black", 190, CREAM, shadow=0.85, shadow_blur=16, shadow_off=(0, 8))
        self.hk_sub = T("jusqu'au 22 novembre inclus", "Playfair-Italic", 54, CREAM, shadow=1.0, shadow_blur=10, shadow_off=(0, 3))
        self.badge = badge_sprite(310)
        self.bar = rect_sprite(8, 44, GOLD, 4)
        self.rule = rect_sprite(120, 4, GOLD, 2)
        # lower thirds
        self.tagK = pill_sprite("NOUVEAU", "Montserrat-ExtraBold", 28, CHOC, PISTA, tracking=5, padx=22, pady=9)
        self.tagF = pill_sprite("CHOIX DU CHEF", "Montserrat-ExtraBold", 28, CHOC, GOLD, tracking=5, padx=22, pady=9)
        self.titleK = T("Kounafa Pistache", "Playfair-Black", 104, CREAM, shadow=0.9, shadow_blur=14, shadow_off=(0, 6))
        self.titleF = T("Crêpe 4 You", "Playfair-Black", 116, CREAM, shadow=0.9, shadow_blur=14, shadow_off=(0, 6))
        self.caps = {c[2]: T(c[2], "Montserrat-Bold", 38, GOLD_HI, tracking=4, shadow=0.95, shadow_blur=7) for c in CAPTIONS}
        # hero price cards
        self.titleKc = T("Kounafa Pistache", "Playfair-Black", 112, CREAM, shadow=0.7, shadow_blur=14)
        self.titleFc = T("Crêpe 4 You", "Playfair-Black", 128, CREAM, shadow=0.7, shadow_blur=14)
        self.old = {p: T(f"{p} DH", "Montserrat-Bold", 70, (222, 210, 192), shadow=0.6, shadow_blur=8) for p in (38, 35)}
        self.new30 = T("30", "Montserrat-Black", 250, GOLD, shadow=0.8, shadow_blur=18, shadow_off=(0, 10),
                       gradient=(GOLD_HI, (224, 160, 60)))
        self.newDH = T("DH", "Montserrat-Black", 96, GOLD, shadow=0.8, shadow_blur=12, gradient=(GOLD_HI, (224, 160, 60)))
        self.aulieu = T("au lieu de", "Playfair-Italic", 40, (222, 210, 192), shadow=0.6, shadow_blur=6)
        self.strike = rect_sprite(250, 9, RED, 4)
        # menu
        self.mn_big = T("−5 DH", "Montserrat-Black", 210, GOLD, shadow=0.8, shadow_blur=18, shadow_off=(0, 10),
                        gradient=(GOLD_HI, (224, 160, 60)))
        self.mn_sub1 = T("SUR TOUTES NOS", "Montserrat-ExtraBold", 44, CREAM, tracking=10, shadow=0.7, shadow_blur=8)
        self.mn_sub2 = T("CRÊPES SUCRÉES", "Montserrat-ExtraBold", 44, CREAM, tracking=10, shadow=0.7, shadow_blur=8)
        self.tiers = []
        for new, old, names in ((15, 20, ["Nutella", "Caramel"]),
                                (20, 25, ["Banane", "Framboise", "Oreo", "KitKat"]),
                                (25, 30, ["Milka", "Snickers", "Twix", "Kinder", "Lotus", "Pistache", "Chocolat blanc"])):
            self.tiers.append(self.make_tier(new, old, names))
        # end card
        self.en_brand = T("RESTO CRÊPES", "Playfair-Black", 84, CREAM, tracking=6, shadow=0.8, shadow_blur=12)
        self.en_tag = T("crêpes  ·  tacos  ·  pizza  ·  glaces  ·  jus", "Playfair-Italic", 34, (226, 214, 196), shadow=0.7, shadow_blur=6)
        self.en_offre = T("OFFRE EXCLUSIVE", "Montserrat-ExtraBold", 50, GOLD, tracking=14, shadow=0.8, shadow_blur=8)
        self.en_jusq = T("JUSQU'AU", "Montserrat-ExtraBold", 58, CREAM, tracking=16, shadow=0.8, shadow_blur=8)
        self.en_date = T("22 Novembre", "Playfair-Black", 150, GOLD, shadow=0.85, shadow_blur=16, shadow_off=(0, 8),
                         gradient=(GOLD_HI, (224, 160, 60)))
        self.en_inclus = T("INCLUS !", "Montserrat-ExtraBold", 58, CREAM, tracking=16, shadow=0.8, shadow_blur=8)
        self.chipK = self.make_chip("Kounafa Pistache", 38)
        self.chipF = self.make_chip("Crêpe 4 You", 35)
        self.en_plus = T("+  −5 DH sur toutes les crêpes sucrées", "Montserrat-Bold", 36, CREAM, shadow=0.8, shadow_blur=6)
        self.en_loc = self.make_loc("Rabat  ·  Salé  ·  Témara")
        self.rule_l = rect_sprite(150, 3, GOLD, 1)
        self.hero_foot = T("jusqu'au 22 novembre inclus", "Playfair-Italic", 44, (232, 220, 202), shadow=0.6, shadow_blur=6)

    # -- composite sprites
    def make_tier(self, new, old, names):
        w, h = 920, 238
        pad = 20
        img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        S = 2
        big = Image.new("RGBA", (img.size[0] * S, img.size[1] * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(big)
        d.rounded_rectangle([pad * S, pad * S, (pad + w) * S, (pad + h) * S], 30 * S, fill=(250, 240, 222, 26),
                            outline=GOLD + (150,), width=2 * S)
        img.alpha_composite(big.resize(img.size, Image.LANCZOS))
        arr = np.array(img)
        # price block (left)
        p_new, pp = text_sprite(str(new), "Montserrat-Black", 128, GOLD, shadow=0.5, shadow_blur=10,
                                gradient=(GOLD_HI, (224, 160, 60)))
        p_dh, pd_ = text_sprite("DH", "Montserrat-Black", 50, GOLD, shadow=0.5, shadow_blur=8,
                                gradient=(GOLD_HI, (224, 160, 60)))
        p_old, po = text_sprite(f"{old} DH", "Montserrat-Bold", 38, (214, 200, 180), shadow=0.4, shadow_blur=5)
        x0 = pad + 34
        blit_rgba(arr, p_old, x0 - po, pad + 22 - po)
        ow = p_old.shape[1] - 2 * po
        oh = p_old.shape[0] - 2 * po
        line = rect_sprite(ow + 14, 5, RED, 2)
        blit_rgba(arr, line, x0 - 7, pad + 22 + int(oh * 0.56))
        blit_rgba(arr, p_new, x0 - pp - 6, pad + 58 - pp)
        nw = p_new.shape[1] - 2 * pp
        blit_rgba(arr, p_dh, x0 + nw + 6 - pd_, pad + 58 + 68 - pd_)
        # divider
        blit_rgba(arr, rect_sprite(3, h - 70, GOLD, 1, 150), pad + 306, pad + 35)
        # names (right, wrapped, auto-fit to 3 lines max)
        nx = 336
        maxw = w - nx - 30
        fs = 48
        while True:
            f = font("Playfair-Bold", fs)
            lines, cur = [], ""
            for nme in names:
                trial = (cur + "  ·  " + nme) if cur else nme
                if f.getlength(trial) <= maxw:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = nme
            lines.append(cur)
            if len(lines) <= 3 or fs <= 34:
                break
            fs -= 2
        lh = int(fs * 1.36)
        y = pad + (h - lh * len(lines)) // 2 - 4
        for ln in lines:
            s, sp = text_sprite(ln, "Playfair-Bold", fs, CREAM, shadow=0.5, shadow_blur=6)
            blit_rgba(arr, s, pad + nx - sp, y - sp)
            y += lh
        return arr, pad

    def make_chip(self, name, old):
        w, h = 452, 236
        pad = 20
        S = 2
        img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        big = Image.new("RGBA", (img.size[0] * S, img.size[1] * S), (0, 0, 0, 0))
        ImageDraw.Draw(big).rounded_rectangle([pad * S, pad * S, (pad + w) * S, (pad + h) * S], 28 * S,
                                              fill=(250, 240, 222, 30), outline=GOLD + (190,), width=2 * S)
        img.alpha_composite(big.resize(img.size, Image.LANCZOS))
        arr = np.array(img)
        size = 44
        while font("Playfair-Bold", size).getlength(name) > w - 44:
            size -= 2
        n, np_ = text_sprite(name, "Playfair-Bold", size, CREAM, shadow=0.5, shadow_blur=6)
        nw = n.shape[1] - 2 * np_
        blit_rgba(arr, n, pad + (w - nw) // 2 - np_, pad + 20 - np_)
        o, op = text_sprite(f"{old} DH", "Montserrat-Bold", 36, (214, 200, 180), shadow=0.4, shadow_blur=5)
        p, pp = text_sprite("30", "Montserrat-Black", 112, GOLD, shadow=0.5, shadow_blur=10, gradient=(GOLD_HI, (224, 160, 60)))
        dh, dp = text_sprite("DH", "Montserrat-Black", 46, GOLD, shadow=0.5, shadow_blur=8, gradient=(GOLD_HI, (224, 160, 60)))
        ow, oh = o.shape[1] - 2 * op, o.shape[0] - 2 * op
        pw = p.shape[1] - 2 * pp
        dw = dh.shape[1] - 2 * dp
        total = ow + 26 + pw + 8 + dw
        x = pad + (w - total) // 2
        yb = pad + 86
        blit_rgba(arr, o, x - op, yb + 58 - op)
        blit_rgba(arr, rect_sprite(ow + 12, 5, RED, 2), x - 6, yb + 58 + int(oh * 0.56))
        blit_rgba(arr, p, x + ow + 26 - pp, yb - pp)
        blit_rgba(arr, dh, x + ow + 26 + pw + 8 - dp, yb + 60 - dp)
        return arr, pad

    def make_loc(self, txt):
        t, tp = text_sprite(txt, "Montserrat-Bold", 40, CREAM, tracking=2, shadow=0.8, shadow_blur=6)
        tw, th = t.shape[1] - 2 * tp, t.shape[0] - 2 * tp
        pin_w = 34
        w, h = tw + pin_w + 18, max(th, 50)
        pad = tp
        arr = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.uint8)
        S = 4
        pin = Image.new("RGBA", (pin_w * S, 46 * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(pin)
        d.ellipse([2 * S, 0, (pin_w - 2) * S, (pin_w - 4) * S], fill=GOLD + (255,))
        d.polygon([(5 * S, 22 * S), ((pin_w - 5) * S, 22 * S), (pin_w * S // 2, 45 * S)], fill=GOLD + (255,))
        d.ellipse([11 * S, 9 * S, (pin_w - 11) * S, (pin_w - 13) * S], fill=CHOC + (255,))
        pin = np.array(pin.resize((pin_w, 46), Image.LANCZOS))
        blit_rgba(arr, pin, pad, pad + (h - 46) // 2 + 2)
        blit_rgba(arr, t, pin_w + 18, (h - th) // 2)
        return arr, pad

    # -- drawing helpers
    def S(self, frame, sp, x, y, **kw):
        spr, pad = sp
        anchor = kw.pop("anchor", "tl")
        if anchor == "tl":
            blit(frame, spr, x - pad, y - pad, **kw)
        elif anchor == "tc":
            blit(frame, spr, x, y - pad, anchor="tc", **kw)
        elif anchor == "c":
            blit(frame, spr, x, y, anchor="c", **kw)

    def top_bar(self, frame, t):
        a = prog(t, 2.05, 0.35) * (1 - prog(t, 21.8, 0.2))
        if a <= 0:
            return
        dy = (1 - ease_out(prog(t, 2.05, 0.4))) * -30
        self.S(frame, self.brand_sm, W / 2, 188 + dy, anchor="tc", alpha=a)
        pulse = 1 + 0.02 * math.sin(t * math.pi * 2)   # one pulse per bar-half, on the beat
        self.S(frame, self.top_pill, W / 2, 278 + dy, anchor="c", alpha=a, scale=pulse)

    def hook(self, frame, t):
        out = 1 - prog(t, 1.82, 0.16)
        if out <= 0:
            return
        x = 72
        a0 = ease_out(prog(t, 0.05, 0.3)) * out
        self.S(frame, self.hk_small, x, 470, alpha=a0)
        blit(frame, self.rule, x, 532, alpha=a0)
        for i, sp in enumerate((self.hk_l1, self.hk_l2)):
            p = ease_out(prog(t, 0.10 + 0.12 * i, 0.38))
            self.S(frame, sp, x - 8 + (1 - p) * -70, 548 + i * 196, alpha=p * out)
        p = ease_out(prog(t, 0.45, 0.3))
        self.S(frame, self.hk_sub, x, 962 + (1 - p) * 20, alpha=p * out)
        pb = prog(t, 0.42, 0.42)
        if pb > 0:
            sc = ease_back(pb, 2.6) * (1 + 0.035 * math.sin((t - 0.84) * math.pi * 4) * (1 if t > 0.84 else 0))
            self.S(frame, self.badge, 842, 1172, anchor="c", alpha=min(1, pb * 3) * out, scale=max(0.05, sc),
                   rot=-12 + (1 - ease_out(pb)) * 40)

    def lower_third(self, frame, t, t_in, t_out, tag, title):
        if t < t_in or t >= t_out:
            return
        p = ease_out(prog(t, t_in + 0.05, 0.4))
        out = 1 - prog(t, t_out - 0.15, 0.15)
        x = 70 + (1 - p) * -80
        self.S(frame, tag, x, 1118, alpha=p * out)
        self.S(frame, title, x - 6, 1172, alpha=p * out)
        for (c0, c1, txt) in CAPTIONS:
            if c0 <= t < c1:
                q = ease_out(prog(t, c0 + 0.03, 0.22))
                a = q * out * (1 - prog(t, c1 - 0.08, 0.08) * (0 if c1 == t_out else 1))
                blit(frame, self.bar, 70, 1330 + (1 - q) * 14, alpha=a)
                self.S(frame, self.caps[txt], 96, 1327 + (1 - q) * 22, alpha=a)

    def price_card(self, frame, tl, tag, title, old):
        """tl = local time since hero start."""
        cx = W / 2
        p = ease_back(prog(tl, 0.02, 0.34), 1.6)
        a = min(1, prog(tl, 0.02, 0.18))
        self.S(frame, tag, cx, 1078, anchor="c", alpha=a, scale=max(0.05, p))
        self.S(frame, title, cx, 1180, anchor="c", alpha=a, scale=max(0.05, 0.9 + 0.1 * p))
        fo = ease_out(prog(tl, 0.75, 0.35))
        blit(frame, self.rule_l, cx - 75, 1598, alpha=fo * 0.9)
        self.S(frame, self.hero_foot, cx, 1622 + (1 - fo) * 12, anchor="tc", alpha=fo * 0.9)
        # old price (left) + strike, new price (right)
        osp, opad = self.old[old]
        ow, oh = osp.shape[1] - 2 * opad, osp.shape[0] - 2 * opad
        nsp, npad = self.new30
        nw, nh = nsp.shape[1] - 2 * npad, nsp.shape[0] - 2 * npad
        dsp, dpad = self.newDH
        dw = dsp.shape[1] - 2 * dpad
        gap = 54
        total = ow + gap + nw + 14 + dw
        x0 = cx - total / 2
        base_y = 1270
        ao = ease_out(prog(tl, 0.18, 0.2))
        self.S(frame, self.aulieu, x0 + ow / 2, base_y + 50, anchor="tc", alpha=ao)
        self.S(frame, self.old[old], x0, base_y + 104, alpha=ao)
        ps = ease_io(prog(tl, 0.30, 0.18))
        if ps > 0:
            line = self.strike[:, :max(2, int((ow + 24) * ps))]
            blit(frame, line, x0 - 12, base_y + 104 + oh * 0.50, alpha=1.0, rot=0)
        pn = prog(tl, 0.45, 0.36)
        if pn > 0:
            sc = 1.9 - 0.9 * ease_back(pn, 2.2)
            sc *= 1 + 0.018 * math.sin(max(0, tl - 0.81) * math.pi * 4)
            an = min(1, pn * 4)
            ncx = x0 + ow + gap + (nw + 14 + dw) / 2
            ncy = base_y + 20 + nh / 2
            # scale "30 DH" as one group around its centre
            self.S(frame, self.new30, ncx + (-(nw + 14 + dw) / 2 + nw / 2) * sc, ncy, anchor="c", alpha=an, scale=sc)
            self.S(frame, self.newDH, ncx + ((nw + 14 + dw) / 2 - dw / 2) * sc, ncy + 46 * sc, anchor="c", alpha=an, scale=sc)

    def menu(self, frame, tl):
        cx = W / 2
        p = ease_back(prog(tl, 0.05, 0.4), 1.8)
        a = min(1, prog(tl, 0.05, 0.2))
        self.S(frame, self.mn_big, cx, 390, anchor="c", alpha=a, scale=max(0.05, 0.6 + 0.4 * p))
        q = ease_out(prog(tl, 0.22, 0.3))
        self.S(frame, self.mn_sub1, cx, 512 + (1 - q) * 20, anchor="tc", alpha=q)
        self.S(frame, self.mn_sub2, cx, 572 + (1 - q) * 20, anchor="tc", alpha=q)
        for i, sp in enumerate(self.tiers):
            ts = 0.48 + 0.8 * i
            r = ease_out(prog(tl, ts, 0.38))
            if r <= 0:
                continue
            bump = 1 + 0.04 * (1 - ease_out(prog(tl, ts + 0.02, 0.5)))
            y = 672 + i * 252
            spr, pad = sp
            blit(frame, spr, cx + (1 - r) * (90 if i % 2 == 0 else -90), y + 119, anchor="c", alpha=r, scale=bump)

    def end(self, frame, tl):
        cx = W / 2
        a = ease_out(prog(tl, 0.05, 0.35))
        self.S(frame, self.en_brand, cx, 262 + (1 - a) * -24, anchor="tc", alpha=a)
        blit(frame, self.rule_l, cx - 75, 384, alpha=a)
        self.S(frame, self.en_tag, cx, 400, anchor="tc", alpha=a * 0.95)
        b = ease_out(prog(tl, 0.2, 0.3))
        self.S(frame, self.en_offre, cx, 498 + (1 - b) * 18, anchor="tc", alpha=b)
        c = ease_out(prog(tl, 0.32, 0.3))
        self.S(frame, self.en_jusq, cx, 590 + (1 - c) * 18, anchor="tc", alpha=c)
        pd_ = prog(tl, 0.42, 0.42)
        if pd_ > 0:
            sc = (1.5 - 0.5 * ease_back(pd_, 2.0)) * (1 + 0.012 * math.sin(max(0, tl - 0.84) * math.pi * 2))
            self.S(frame, self.en_date, cx, 760, anchor="c", alpha=min(1, pd_ * 4), scale=sc)
        e = ease_out(prog(tl, 0.62, 0.3))
        self.S(frame, self.en_inclus, cx, 862 + (1 - e) * 18, anchor="tc", alpha=e)
        for i, chip in enumerate((self.chipK, self.chipF)):
            r = ease_out(prog(tl, 0.85 + 0.14 * i, 0.36))
            spr, pad = chip
            x = cx + (-238 if i == 0 else 238)
            blit(frame, spr, x, 1104 + (1 - r) * 50, anchor="c", alpha=r)
        g = ease_out(prog(tl, 1.25, 0.3))
        self.S(frame, self.en_plus, cx, 1252 + (1 - g) * 14, anchor="tc", alpha=g)
        hh = ease_out(prog(tl, 1.4, 0.3))
        self.S(frame, self.en_loc, cx, 1362 + (1 - hh) * 14, anchor="c", alpha=hh)


def blit_rgba(dst, src, x, y):
    """'over' composite RGBA src onto RGBA dst (in place)."""
    x, y = int(round(x)), int(round(y))
    h, w = src.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + w), min(dst.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255
    d = dst[y0:y1, x0:x1].astype(np.float32) / 255
    sa, da = s[..., 3:4], d[..., 3:4]
    oa = sa + da * (1 - sa)
    rgb = (s[..., :3] * sa + d[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-6)
    dst[y0:y1, x0:x1] = (np.dstack([rgb, oa]) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ render
def open_src(seg):
    n = int(round((seg["t1"] - seg["t0"]) * FPS))
    if "photo" in seg:
        return PhotoSrc(f"{CLIPS}/{seg['photo']}", seg["ps"], *seg["poff"])
    return VideoSrc(seg["clip"], seg["ss"], seg["speed"], n, slow=seg.get("slow", False), extra=seg.get("extra", ""))


def render(out, t0=0.0, t1=TOTAL, scale=1.0, stills=None, still_dir=None, audio=None):
    g = GFX()
    ow, oh = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    enc = None
    if out:
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
               "-i", "-"]
        if audio:
            cmd += ["-ss", f"{t0:.3f}", "-i", audio]
        vf = [] if scale == 1.0 else [f"scale={ow}:{oh}:flags=area"]
        vf.append("format=yuv420p")
        cmd += ["-vf", ",".join(vf), "-c:v", "libx264", "-preset", "slow" if scale == 1.0 else "veryfast",
                "-crf", "16" if scale == 1.0 else "24", "-profile:v", "high", "-level", "4.2",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
        if audio:
            cmd += ["-af", "loudnorm=I=-14:TP=-1.2:LRA=9", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-shortest"]
        cmd += [out]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    still_frames = set(int(round(s * FPS)) for s in (stills or []))
    nframes = int(round(TOTAL * FPS))
    f0, f1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    for si, seg in enumerate(SEGS):
        s0, s1 = int(round(seg["t0"] * FPS)), int(round(seg["t1"] * FPS))
        if s1 <= f0 or s0 >= f1:
            continue
        if stills is not None and not out and not any(s0 <= k < s1 for k in still_frames):
            continue
        src = open_src(seg)
        n = s1 - s0
        nxt = SEGS[si + 1] if si + 1 < len(SEGS) else None
        for k in range(n):
            fi = s0 + k
            raw = src.read()
            if fi < f0 or fi >= f1:
                continue
            if stills is not None and not out and fi not in still_frames:
                continue
            t = fi / FPS
            tl = t - seg["t0"]
            dur = seg["t1"] - seg["t0"]
            look = seg["look"]
            # ---- camera move
            pz = ease_out(tl / dur) if look in ("prep", "hook") else tl / dur
            z = seg["z"][0] + (seg["z"][1] - seg["z"][0]) * pz
            if look == "prep" and tl < 0.2:       # punch-in on every cut
                z *= 1 + 0.05 * (1 - ease_out(tl / 0.2))
            dx, dy = seg.get("shift", (0, 0))
            shake = 0.0
            if look in ("heroK", "heroF") and 0.45 <= tl < 0.75:
                shake = (1 - (tl - 0.45) / 0.3) * 9
            sx = shake * math.sin(tl * 95)
            sy = shake * math.cos(tl * 83)
            frame = warp(raw, z, seg["f"][0], seg["f"][1], dx + sx, dy + sy)
            # ---- whip transitions (in / out)
            tin = seg.get("tin")
            if tin == "whip" and tl < 0.2:
                frame = zoom_blur(frame, 1 - tl / 0.2, direction=-0.6)
            if nxt is not None and nxt.get("tin") == "whip" and dur - tl <= 0.2:
                frame = zoom_blur(frame, 1 - (dur - tl) / 0.2 + 1 / 6, direction=1)
            # ---- looks
            if look == "hook":
                frame = apply_mask(frame, g.masks["hook"])
            elif look == "prep":
                frame = apply_mask(frame, g.masks["prep"])
            elif look in ("heroK", "heroF"):
                frame = apply_mask(frame, g.masks["vig"])
                frame = panel(frame, 890, 1100, CHOC, 0.97)
                frame = apply_mask(frame, 1 - 0.45 * np.clip((430 - np.arange(H, dtype=np.float32)) / 430, 0, 1)[:, None, None] ** 1.4)
            elif look == "menu":
                frame = (frame.astype(np.float32) * 0.34 + np.array(CHOC, np.float32) * 0.62).astype(np.uint8)
                frame = apply_mask(frame, g.masks["vig"])
            elif look == "end":
                frame = (frame.astype(np.float32) * 0.40 + np.array(CHOC, np.float32) * 0.55).astype(np.uint8)
                frame = apply_mask(frame, g.masks["vig"])
            # ---- graphics
            if look == "hook":
                g.hook(frame, t)
            g.lower_third(frame, t, 2.0, 10.0, g.tagK, g.titleK)
            g.lower_third(frame, t, 12.0, 20.0, g.tagF, g.titleF)
            if look == "heroK":
                g.sparks.draw(frame, t, strength=prog(tl, 0.45, 0.3))
                g.price_card(frame, tl, g.tagK, g.titleKc, 38)
            if look == "heroF":
                g.sparks.draw(frame, t, strength=prog(tl, 0.45, 0.3))
                g.price_card(frame, tl, g.tagF, g.titleFc, 35)
            if look == "menu":
                g.sparks_end.draw(frame, t, strength=0.5)
                g.menu(frame, tl)
            if look == "end":
                g.sparks_end.draw(frame, t, strength=0.9)
                g.end(frame, tl)
            g.top_bar(frame, t)
            # ---- flashes
            if tin == "flash" and tl < 0.3:
                frame = flash(frame, 0.85 * (1 - tl / 0.3) ** 1.6)
            if look in ("heroK", "heroF") and 0.45 <= tl < 0.7:
                frame = flash(frame, 0.22 * (1 - (tl - 0.45) / 0.25))
            if t < 0.2:                                     # opening fade from warm white
                frame = flash(frame, 0.6 * (1 - t / 0.2))
            if t > TOTAL - 0.35:                            # closing fade
                frame = (frame.astype(np.float32) * (1 - 0.9 * prog(t, TOTAL - 0.35, 0.35))).astype(np.uint8)
            if enc:
                enc.stdin.write(frame.tobytes())
            if fi in still_frames and still_dir:
                Image.fromarray(frame).save(f"{still_dir}/f_{t:06.2f}.jpg", quality=90)
        src.close()
        print(f"seg {si + 1}/{len(SEGS)} done ({seg['t0']:.1f}-{seg['t1']:.1f})", flush=True)
    if enc:
        enc.stdin.close()
        enc.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=TOTAL)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--stills", default="")
    ap.add_argument("--still-dir", default="")
    ap.add_argument("--audio", default="")
    a = ap.parse_args()
    st = [float(x) for x in a.stills.split(",")] if a.stills else None
    if a.still_dir:
        os.makedirs(a.still_dir, exist_ok=True)
    render(a.out, a.t0, a.t1, a.scale, st, a.still_dir, a.audio or None)
