"""Resto Crêpes – promo ad v2 ("pop gourmand" look: red / yellow, sticker prices old -> new).

Same edit and cuts as build.py; new graphics, grade and music.
Usage: build2.py OUT.mp4 [--audio WAV] [--stills t,t --still-dir DIR] [--scale F]
"""
import argparse
import copy
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build as B  # noqa: E402  (engine: sources, warps, blits)
from build import (W, H, FPS, clamp, ease_out, ease_io, ease_back, prog, font, text_size, blit, blit_rgba,  # noqa: E402
                   warp, zoom_blur, apply_mask, flash, CAPTIONS, TOTAL)

INK = (26, 12, 8)
WHITE = (255, 255, 255)
YEL = (255, 208, 0)
YEL_HI = (255, 236, 110)
RED = (234, 26, 42)
RED_D = (132, 8, 20)
PISTA = (166, 220, 52)
PINK = (255, 56, 122)
GREY = (120, 104, 98)

# punchier, warmer, more saturated food grade
B.GRADE = ("hqdn3d=1.5:1.5:4:4,"
           "eq=contrast=1.10:saturation=1.30:brightness=0.02,"
           "colorbalance=rm=0.04:bm=-0.05:rh=0.02:bh=-0.03:rs=0.02,"
           "curves=m='0/0 0.25/0.23 0.75/0.81 1/1',"
           "unsharp=5:5:0.7:3:3:0.0")


# ------------------------------------------------------------------ sprites
def tsprite(txt, fname, size, fill, tracking=0, stroke=0, stroke_fill=INK, extrude=None, shadow=0.0,
            shadow_blur=10, shadow_off=(0, 6), maxw=None):
    """Text sprite: fill + outline + optional 3-D extrusion + optional soft shadow. Returns (rgba, pad)."""
    if maxw:
        while text_size(txt, font(fname, size), tracking)[0] + 2 * stroke > maxw and size > 20:
            size -= 2
    f = font(fname, size)
    tw, th = text_size(txt, f, tracking)
    ex, ey = (extrude[0], extrude[1]) if extrude else (0, 0)
    pad = stroke + max(abs(ex), abs(ey)) + (shadow_blur * 3 if shadow > 0 else 0) + 10
    size_px = (tw + 2 * pad, th + 2 * pad)
    img = Image.new("RGBA", size_px, (0, 0, 0, 0))

    def mask_of(sw):
        m = Image.new("L", size_px, 0)
        d = ImageDraw.Draw(m)
        x = pad
        for ch in txt:
            d.text((x, pad), ch, font=f, fill=255, stroke_width=sw)
            x += f.getlength(ch) + tracking
        return m

    m_out = mask_of(stroke)
    m_in = mask_of(0) if stroke else m_out
    if shadow > 0:
        sh = m_out.filter(ImageFilter.GaussianBlur(shadow_blur))
        sh = Image.eval(sh, lambda v: int(min(255, v * 1.7) * shadow))
        layer = Image.new("RGBA", size_px, (10, 4, 2, 0))
        layer.putalpha(sh)
        img.alpha_composite(layer, (shadow_off[0] + ex, shadow_off[1] + ey))
    if extrude:
        col = Image.new("RGBA", size_px, tuple(extrude[2]) + (0,))
        col.putalpha(m_out)
        steps = max(abs(ex), abs(ey))
        for i in range(steps, 0, -1):
            img.alpha_composite(col, (int(round(ex * i / steps)), int(round(ey * i / steps))))
    if stroke:
        st = Image.new("RGBA", size_px, tuple(stroke_fill) + (0,))
        st.putalpha(m_out)
        img.alpha_composite(st)
    fg = Image.new("RGBA", size_px, tuple(fill) + (0,))
    fg.putalpha(m_in)
    img.alpha_composite(fg)
    return np.array(img), pad


def inner(sp):
    """(w, h) of a sprite without its padding."""
    arr, pad = sp
    return arr.shape[1] - 2 * pad, arr.shape[0] - 2 * pad


def box_sprite(w, h, fill, radius=24, outline=None, ow=4, hard=(7, 9), hard_col=INK, alpha=255):
    """Rounded box with a hard offset shadow (sticker look). Returns (rgba, pad)."""
    S = 3
    pad = max(hard) + 8
    img = Image.new("RGBA", ((w + 2 * pad) * S, (h + 2 * pad) * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if hard != (0, 0):
        d.rounded_rectangle([(pad + hard[0]) * S, (pad + hard[1]) * S, (pad + w + hard[0]) * S, (pad + h + hard[1]) * S],
                            radius * S, fill=tuple(hard_col) + (255,))
    d.rounded_rectangle([pad * S, pad * S, (pad + w) * S, (pad + h) * S], radius * S, fill=tuple(fill) + (alpha,),
                        outline=(tuple(outline) + (255,)) if outline else None, width=ow * S)
    return np.array(img.resize((w + 2 * pad, h + 2 * pad), Image.LANCZOS)), pad


def label_sprite(txt, fname, size, fg, bg, tracking=3, padx=26, pady=12, radius=None, outline=None, ow=4,
                 hard=(5, 6), stroke=0, maxw=None):
    """Text inside a rounded sticker box."""
    t = tsprite(txt, fname, size, fg, tracking=tracking, stroke=stroke, maxw=maxw)
    tw, th = inner(t)
    w, h = tw + 2 * padx, th + 2 * pady
    box, pad = box_sprite(w, h, bg, radius if radius is not None else h // 2, outline, ow, hard)
    blit_rgba(box, t[0], pad + padx - t[1], pad + pady - t[1])
    return box, pad


def starburst(diam, fill=YEL, n=22, depth=0.90, outline=INK, ow=7, hard=(8, 10), ring=True):
    S = 3
    pad = max(hard) + 10
    size = (diam + 2 * pad) * S
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    c = size / 2
    R = diam * S / 2

    def pts(r_out, dx=0, dy=0):
        out = []
        for i in range(n * 2):
            a = math.pi * i / n
            r = r_out if i % 2 == 0 else r_out * depth
            out.append((c + dx + r * math.cos(a), c + dy + r * math.sin(a)))
        return out

    d = ImageDraw.Draw(img)
    d.polygon(pts(R, hard[0] * S, hard[1] * S), fill=INK + (255,))
    d.polygon(pts(R), fill=tuple(outline) + (255,))
    d.polygon(pts(R - ow * S), fill=tuple(fill) + (255,))
    if ring:
        rr = R * 0.80
        d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=(255, 255, 255, 150), width=int(3 * S))
    return np.array(img.resize((diam + 2 * pad, diam + 2 * pad), Image.LANCZOS)), pad


def arrow_sprite(w=120, h=96, fill=YEL, ow=7):
    S = 3
    pad = 16
    img = Image.new("RGBA", ((w + 2 * pad) * S, (h + 2 * pad) * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def poly(o, dx=0, dy=0):
        x0, y0 = (pad + dx) * S, (pad + dy) * S
        ws, hs = w * S, h * S
        sh = hs * 0.30
        p = [(x0 - o, y0 + sh - o), (x0 + ws * 0.52 - o * 0.4, y0 + sh - o), (x0 + ws * 0.52 - o * 0.4, y0 - o * 1.6),
             (x0 + ws + o * 1.4, y0 + hs / 2), (x0 + ws * 0.52 - o * 0.4, y0 + hs + o * 1.6),
             (x0 + ws * 0.52 - o * 0.4, y0 + hs - sh + o), (x0 - o, y0 + hs - sh + o)]
        return p

    d.polygon(poly(ow * S, 5, 7), fill=INK + (255,))
    d.polygon(poly(ow * S), fill=INK + (255,))
    d.polygon(poly(0), fill=tuple(fill) + (255,))
    return np.array(img.resize((w + 2 * pad, h + 2 * pad), Image.LANCZOS)), pad


def line_sprite(w, h, color, outline=INK, ow=3):
    S = 3
    pad = ow + 2
    img = Image.new("RGBA", ((w + 2 * pad) * S, (h + 2 * pad) * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([(pad - ow) * S, (pad - ow) * S, (pad + w + ow) * S, (pad + h + ow) * S], (h // 2 + ow) * S,
                        fill=tuple(outline) + (255,))
    d.rounded_rectangle([pad * S, pad * S, (pad + w) * S, (pad + h) * S], (h // 2) * S, fill=tuple(color) + (255,))
    return np.array(img.resize((w + 2 * pad, h + 2 * pad), Image.LANCZOS)), pad


def old_price_sprite(price, size, color, strike=RED, stroke=0, lw=None):
    """'38 DH' with a slanted strike-through, pre-composited (for static uses)."""
    t = tsprite(f"{price} DH", "Lilita", size, color, stroke=stroke)
    arr = t[0].copy()
    tw, th = inner(t)
    lw = lw or max(5, size // 9)
    ln, lp = line_sprite(tw + 22, lw, strike, ow=max(2, lw // 4))
    M = cv2.getRotationMatrix2D((ln.shape[1] / 2, ln.shape[0] / 2), 7, 1.0)
    dh = int(abs(math.sin(math.radians(7))) * ln.shape[1]) + ln.shape[0] + 4
    canvas = np.zeros((dh, ln.shape[1], 4), np.uint8)
    canvas[(dh - ln.shape[0]) // 2:(dh - ln.shape[0]) // 2 + ln.shape[0]] = ln
    M = cv2.getRotationMatrix2D((canvas.shape[1] / 2, canvas.shape[0] / 2), 7, 1.0)
    rot = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_LINEAR)
    blit_rgba(arr, rot, t[1] + tw / 2 - rot.shape[1] / 2, t[1] + th * 0.54 - rot.shape[0] / 2)
    return arr, t[1]


# ------------------------------------------------------------------ backgrounds / light
class Rays:
    """Rotating sunburst rays, blended toward a colour through an alpha map."""

    def __init__(self, cx, cy, n=14):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.theta = np.arctan2(yy - cy, xx - cx)
        self.n = n

    def draw(self, frame, rot, color, amap):
        a = np.clip(0.5 + 5.0 * np.sin(self.n * (self.theta + rot)), 0, 1) * amap
        a = a[..., None]
        col = np.array(color, np.float32)[None, None, :]
        return (frame.astype(np.float32) * (1 - a) + col * a).astype(np.uint8)


def bloom(frame, amount=0.20):
    small = cv2.resize(frame, (W // 6, H // 6), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 9)
    up = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    f = frame.astype(np.float32)
    return (255 - (255 - f) * (255 - up * amount) / 255).astype(np.uint8)


def vgrad(y0, y1, power=1.0):
    y = np.arange(H, dtype=np.float32)
    a = np.clip((y - y0) / (y1 - y0), 0, 1)
    return (a * a * (3 - 2 * a)) ** power


def make_masks():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W * 0.80)) ** 2 + ((yy - H / 2) / (H * 0.74)) ** 2)
    vig = 1 - 0.36 * np.clip(r - 0.38, 0, 1) ** 1.6
    y = np.arange(H, dtype=np.float32)
    bottom = 1 - 0.55 * np.clip((y - 1000) / 480, 0, 1) ** 1.3
    top = 1 - 0.45 * np.clip((430 - y) / 430, 0, 1) ** 1.4
    prep = vig * (bottom * top)[:, None]
    hookm = vig * (1 - 0.38 * np.clip((y - 300) / 350, 0, 1) * np.clip((1250 - y) / 300, 0, 1))[:, None] * top[:, None]
    return {"vig": vig[..., None], "prep": prep[..., None], "hook": hookm[..., None]}


class Sparks:
    def __init__(self, n=44, seed=5, region=(40, 1040, 180, 1700)):
        r = np.random.default_rng(seed)
        self.x = r.uniform(region[0], region[1], n)
        self.y = r.uniform(region[2], region[3], n)
        self.v = r.uniform(40, 130, n)
        self.ph = r.uniform(0, 6.28, n)
        self.fr = r.uniform(2.0, 6.0, n)
        self.sz = r.integers(0, 3, n)
        self.region = region
        self.spr = []
        for d in (26, 40, 60):     # 4-point star glints
            yy, xx = np.mgrid[0:d, 0:d].astype(np.float32)
            dx, dy = np.abs(xx - d / 2 + .5), np.abs(yy - d / 2 + .5)
            g = np.exp(-(dx * dx + dy * dy) / (2 * (d / 9) ** 2))
            star = np.exp(-dx / (d * 0.02 + 0.6)) * np.exp(-dy / (d * 0.22)) + np.exp(-dy / (d * 0.02 + 0.6)) * np.exp(-dx / (d * 0.22))
            a = np.clip(g + star * 0.9, 0, 1)
            s = np.zeros((d, d, 4), np.uint8)
            s[..., 0], s[..., 1], s[..., 2] = 255, 244, 190
            s[..., 3] = (a * 255).astype(np.uint8)
            self.spr.append(s)

    def draw(self, frame, t, strength=1.0):
        if strength <= 0.01:
            return
        x0, x1, y0, y1 = self.region
        span = y1 - y0
        for i in range(len(self.x)):
            y = y1 - ((self.y[i] - y0 + self.v[i] * t) % span)
            tw = max(0.0, math.sin(t * self.fr[i] + self.ph[i])) ** 3
            edge = min(1.0, (y - y0) / 100, (y1 - y) / 100)
            blit(frame, self.spr[self.sz[i]], self.x[i], y, alpha=strength * tw * max(0, edge), anchor="c",
                 additive=True, scale=0.6 + 0.6 * tw)


# ------------------------------------------------------------------ timeline (same cuts as v1)
SEGS = copy.deepcopy(B.SEGS)
for s in SEGS:
    if s["look"] == "heroK":
        s["poff"] = (-264, -776)
        s["f"] = (0.50, 0.30)
    if s["look"] == "heroF":
        s["shift"] = (0, -960)
    if s["look"] == "menu":
        s["bg"] = True
    if s["look"] == "end":
        s["extra"] = "gblur=sigma=4"


# ------------------------------------------------------------------ graphics
class GFX:
    def __init__(self):
        self.masks = make_masks()
        self.sparks = Sparks()
        self.rays_hero = Rays(770, 1360)
        self.rays_menu = Rays(W / 2, 420, n=16)
        self.panel_a = vgrad(790, 1000)[:, None]                         # hero panel alpha
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt((xx - W / 2) ** 2 + (yy - 520) ** 2) / 1500
        base = np.array((240, 40, 44), np.float32)[None, None, :] * (1 - np.clip(d, 0, 1))[..., None] ** 1.2 \
            + np.array(RED_D, np.float32)[None, None, :] * (1 - (1 - np.clip(d, 0, 1))[..., None] ** 1.2)
        self.menu_bg = base.astype(np.uint8)
        self.menu_ray_a = (0.30 * (1 - np.clip(d, 0, 1) * 0.7)).astype(np.float32)
        hero_d = np.sqrt((xx - 770) ** 2 + (yy - 1360) ** 2) / 1100
        self.hero_ray_a = (0.34 * (1 - np.clip(hero_d, 0, 1)) * self.panel_a).astype(np.float32)

        # ---- persistent top
        self.brand_sm = tsprite("RESTO CRÊPES", "Montserrat-Black", 30, WHITE, tracking=9, shadow=0.9, shadow_blur=6,
                                shadow_off=(0, 3))
        self.top_pill = label_sprite("PROMO  •  JUSQU'AU 22 NOVEMBRE", "Montserrat-Black", 31, WHITE, RED, tracking=2,
                                     padx=30, pady=13, outline=WHITE, ow=4, hard=(4, 6))
        # ---- hook
        self.hk_promo = tsprite("Promo !", "Kaushan", 170, YEL, stroke=8, extrude=(6, 8, INK), shadow=0.6)
        self.hk_l1 = tsprite("CRÊPES", "Lilita", 236, WHITE, stroke=10, extrude=(8, 11, RED), shadow=0.7, shadow_blur=14,
                             maxw=900)
        self.hk_l2 = tsprite("SUCRÉES", "Lilita", 236, WHITE, stroke=10, extrude=(8, 11, RED), shadow=0.7, shadow_blur=14,
                             maxw=900)
        self.hk_slab = label_sprite("JUSQU'AU 22 NOVEMBRE INCLUS", "Montserrat-Black", 40, WHITE, RED, tracking=2,
                                    padx=28, pady=16, radius=16, outline=WHITE, ow=4, hard=(6, 8))
        self.hk_sticker = self.make_from_sticker()
        # ---- lower thirds
        self.tag = {"K": label_sprite("NOUVEAU", "Montserrat-Black", 28, INK, YEL, tracking=5, padx=22, pady=9, hard=(4, 5)),
                    "F": label_sprite("CHOIX DU CHEF", "Montserrat-Black", 28, INK, YEL, tracking=5, padx=22, pady=9,
                                      hard=(4, 5))}
        self.slab = {"K": label_sprite("KOUNAFA PISTACHE", "Lilita", 88, INK, PISTA, tracking=1, padx=30, pady=10,
                                       radius=20, outline=INK, ow=5, hard=(8, 10), maxw=800),
                     "F": label_sprite("CRÊPE 4 YOU", "Lilita", 100, WHITE, PINK, tracking=2, padx=34, pady=10, radius=20,
                                       outline=INK, ow=5, hard=(8, 10), stroke=5, maxw=800)}
        self.caps = {c[2]: label_sprite(c[2], "Montserrat-ExtraBold", 34, WHITE, INK, tracking=2, padx=22, pady=11,
                                        radius=14, hard=(0, 0)) for c in CAPTIONS}
        # ---- hero price cards
        self.h_title = {"K": tsprite("KOUNAFA PISTACHE", "Lilita", 118, WHITE, stroke=8, extrude=(6, 8, INK), shadow=0.5,
                                     maxw=960),
                        "F": tsprite("CRÊPE 4 YOU", "Lilita", 140, WHITE, stroke=8, extrude=(6, 8, INK), shadow=0.5,
                                     maxw=960)}
        self.h_aulieu = tsprite("au lieu de", "Kaushan", 54, WHITE, stroke=4, shadow=0.5, shadow_blur=6)
        self.h_old = {p: tsprite(f"{p} DH", "Lilita", 124, WHITE, stroke=7, extrude=(4, 6, INK)) for p in (38, 35)}
        self.h_strike = line_sprite(330, 16, YEL, ow=5)
        self.h_arrow = arrow_sprite(104, 92)
        self.h_sticker = self.make_price_sticker(460, 30)
        self.h_seul = tsprite("seulement", "Kaushan", 62, YEL, stroke=5, extrude=(3, 4, INK))
        # ---- menu
        self.mn_promo = tsprite("Promo !", "Kaushan", 132, YEL, stroke=7, extrude=(5, 7, INK), shadow=0.5)
        self.mn_title1 = tsprite("TOUTES NOS", "Lilita", 86, WHITE, stroke=7, extrude=(5, 7, INK), tracking=2)
        self.mn_title2 = tsprite("CRÊPES SUCRÉES", "Lilita", 116, WHITE, stroke=8, extrude=(6, 8, INK), maxw=940)
        self.tiers = [self.make_tier(15, 20, ["Nutella", "Caramel"]),
                      self.make_tier(20, 25, ["Banane", "Framboise", "Oreo", "KitKat"]),
                      self.make_tier(25, 30, ["Milka", "Snickers", "Twix", "Kinder", "Lotus", "Pistache", "Chocolat blanc"])]
        # ---- end card
        self.en_brand = tsprite("RESTO CRÊPES", "Lilita", 104, WHITE, stroke=8, extrude=(6, 8, RED), shadow=0.6, tracking=2)
        self.en_offre = tsprite("Offre exclusive", "Kaushan", 100, YEL, stroke=7, extrude=(5, 7, INK), shadow=0.5)
        self.en_jusq = tsprite("JUSQU'AU", "Lilita", 86, WHITE, stroke=7, extrude=(5, 7, INK), tracking=4)
        self.en_date = tsprite("22 NOVEMBRE", "Lilita", 168, YEL, stroke=10, extrude=(8, 11, INK), shadow=0.6, maxw=980)
        self.en_inclus = label_sprite("INCLUS !", "Lilita", 70, WHITE, RED, tracking=4, padx=30, pady=6, radius=18,
                                      outline=WHITE, ow=5, hard=(6, 8))
        self.chip = {"K": self.make_chip("KOUNAFA PISTACHE", 38, PISTA, INK), "F": self.make_chip("CRÊPE 4 YOU", 35, PINK, WHITE)}
        self.en_all = label_sprite("Toutes nos crêpes sucrées en promo, dès 15 DH", "Montserrat-ExtraBold", 31, WHITE, INK,
                                   tracking=0, padx=24, pady=13, radius=16, hard=(0, 0))
        self.en_loc = self.make_loc("Rabat  ·  Salé  ·  Témara")

    # ---- composite sprites
    def make_from_sticker(self):
        """Hook sticker: 'dès 15 DH' with the old 20 DH struck."""
        st, pad = starburst(350)
        c = st.shape[0] / 2
        d = tsprite("DÈS", "Montserrat-Black", 40, INK, tracking=4)
        blit_rgba(st, d[0], c - d[0].shape[1] / 2, c - 132 - d[1])
        n = tsprite("15", "Lilita", 156, RED, stroke=5)
        dh = tsprite("DH", "Lilita", 64, RED, stroke=4)
        nw, dw = inner(n)[0], inner(dh)[0]
        x = c - (nw + 8 + dw) / 2
        blit_rgba(st, n[0], x - n[1], c - 98 - n[1])
        blit_rgba(st, dh[0], x + nw + 8 - dh[1], c - 98 + 78 - dh[1])
        o = old_price_sprite(20, 48, INK, strike=RED)
        blit_rgba(st, o[0], c - o[0].shape[1] / 2, c + 64 - o[1])
        return st, pad

    def make_price_sticker(self, diam, price):
        st, pad = starburst(diam, n=24)
        c = st.shape[0] / 2
        n = tsprite(str(price), "Lilita", int(diam * 0.50), RED, stroke=7, extrude=(4, 6, INK))
        dh = tsprite("DH", "Lilita", int(diam * 0.20), RED, stroke=5, extrude=(3, 4, INK))
        nw, nh = inner(n)
        dw = inner(dh)[0]
        x = c - (nw + 6 + dw) / 2
        y = c - nh / 2 - 8
        blit_rgba(st, n[0], x - n[1], y - n[1])
        blit_rgba(st, dh[0], x + nw + 6 - dh[1], y + nh * 0.50 - dh[1])
        return st, pad

    def make_tier(self, new, old, names):
        w, h = 930, 272
        box, pad = box_sprite(w, h, (255, 252, 244), 34, outline=INK, ow=6, hard=(9, 11))
        o = old_price_sprite(old, 84, GREY, strike=RED)
        a = arrow_sprite(74, 64, fill=YEL, ow=5)
        n = tsprite(str(new), "Lilita", 132, RED, stroke=6, extrude=(4, 5, INK))
        dh = tsprite("DH", "Lilita", 66, RED, stroke=5, extrude=(3, 4, INK))
        ow_, oh = inner(o)
        aw, ah = inner(a)
        nw, nh = inner(n)
        dw = inner(dh)[0]
        total = ow_ + 26 + aw + 30 + nw + 8 + dw
        x = pad + (w - total) / 2
        row = pad + 14
        blit_rgba(box, o[0], x - o[1], row + 34 - o[1])
        blit_rgba(box, a[0], x + ow_ + 26 - a[1], row + 40 - a[1])
        blit_rgba(box, n[0], x + ow_ + 26 + aw + 30 - n[1], row - 12 - n[1])
        blit_rgba(box, dh[0], x + ow_ + 26 + aw + 30 + nw + 8 - dh[1], row + 52 - dh[1])
        # names (centred, up to 2 lines)
        fs = 40
        while True:
            f = font("Montserrat-ExtraBold", fs)
            lines, cur = [], ""
            for nme in names:
                trial = (cur + "  ·  " + nme) if cur else nme
                if f.getlength(trial) <= w - 70:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = nme
            lines.append(cur)
            if len(lines) <= 2 or fs <= 26:
                break
            fs -= 2
        lh = int(fs * 1.32)
        y = pad + 146 + (h - 146 - 18 - lh * len(lines)) / 2
        for ln in lines:
            s = tsprite(ln, "Montserrat-ExtraBold", fs, INK)
            blit_rgba(box, s[0], pad + (w - inner(s)[0]) / 2 - s[1], y - s[1])
            y += lh
        return box, pad

    def make_chip(self, name, old, accent, name_col):
        w, h = 458, 250
        box, pad = box_sprite(w, h, (255, 252, 244), 30, outline=INK, ow=6, hard=(8, 10))
        hd, hp_ = box_sprite(w - 28, 70, accent, 18, outline=INK, ow=4, hard=(0, 0))
        blit_rgba(box, hd, pad + 14 - hp_, pad + 14 - hp_)
        t = tsprite(name, "Lilita", 44, name_col, stroke=4 if name_col == WHITE else 0, maxw=w - 60)
        blit_rgba(box, t[0], pad + (w - inner(t)[0]) / 2 - t[1], pad + 14 + (70 - inner(t)[1]) / 2 - t[1] - 2)
        o = old_price_sprite(old, 52, GREY, strike=RED)
        a = arrow_sprite(44, 40, fill=YEL, ow=4)
        n = tsprite("30", "Lilita", 118, RED, stroke=6, extrude=(4, 5, INK))
        dh = tsprite("DH", "Lilita", 54, RED, stroke=4, extrude=(3, 4, INK))
        ow_, aw, nw, dw = inner(o)[0], inner(a)[0], inner(n)[0], inner(dh)[0]
        total = ow_ + 14 + aw + 16 + nw + 6 + dw
        x = pad + (w - total) / 2
        yb = pad + 98
        blit_rgba(box, o[0], x - o[1], yb + 46 - o[1])
        blit_rgba(box, a[0], x + ow_ + 14 - a[1], yb + 50 - a[1])
        blit_rgba(box, n[0], x + ow_ + 14 + aw + 16 - n[1], yb - 4 - n[1])
        blit_rgba(box, dh[0], x + ow_ + 14 + aw + 16 + nw + 6 - dh[1], yb + 58 - dh[1])
        return box, pad

    def make_loc(self, txt):
        t = tsprite(txt, "Montserrat-Black", 38, WHITE, tracking=2)
        tw, th = inner(t)
        pin_w = 34
        w, h = tw + pin_w + 16 + 52, th + 26
        box, pad = box_sprite(w, h, INK, h // 2, outline=YEL, ow=4, hard=(0, 0))
        S = 4
        pin = Image.new("RGBA", (pin_w * S, 46 * S), (0, 0, 0, 0))
        d = ImageDraw.Draw(pin)
        d.ellipse([2 * S, 0, (pin_w - 2) * S, (pin_w - 4) * S], fill=YEL + (255,))
        d.polygon([(5 * S, 22 * S), ((pin_w - 5) * S, 22 * S), (pin_w * S // 2, 45 * S)], fill=YEL + (255,))
        d.ellipse([11 * S, 9 * S, (pin_w - 11) * S, (pin_w - 13) * S], fill=INK + (255,))
        pin = np.array(pin.resize((pin_w, 46), Image.LANCZOS))
        blit_rgba(box, pin, pad + 26, pad + (h - 46) / 2 + 1)
        blit_rgba(box, t[0], pad + 26 + pin_w + 16 - t[1], pad + 13 - t[1])
        return box, pad

    # ---- helpers
    def S(self, frame, sp, x, y, anchor="tl", **kw):
        spr, pad = sp
        if anchor == "tl":
            if kw.get("scale", 1.0) != 1.0 or kw.get("rot", 0.0) != 0.0:
                w, h = spr.shape[1], spr.shape[0]
                blit(frame, spr, x - pad + w / 2, y - pad + h / 2, anchor="c", **kw)
            else:
                blit(frame, spr, x - pad, y - pad, **kw)
        elif anchor == "tc":
            blit(frame, spr, x, y - pad, anchor="tc", **kw)
        else:
            blit(frame, spr, x, y, anchor="c", **kw)

    # ---- scenes
    def top_bar(self, frame, t):
        a = prog(t, 2.05, 0.3) * (1 - prog(t, 21.8, 0.2))
        if a <= 0:
            return
        dy = (1 - ease_back(prog(t, 2.05, 0.4))) * -60
        self.S(frame, self.brand_sm, W / 2, 186 + dy, anchor="tc", alpha=a)
        beat = (t % 0.5) / 0.5
        pulse = 1 + 0.045 * (1 - ease_out(beat * 2.2))          # bumps on every beat
        self.S(frame, self.top_pill, W / 2, 280 + dy, anchor="c", alpha=a, scale=pulse)

    def hook(self, frame, t):
        out = 1 - prog(t, 1.82, 0.16)
        if out <= 0:
            return
        p = prog(t, 0.04, 0.34)
        if p > 0:
            self.S(frame, self.hk_promo, 300, 430, anchor="c", alpha=min(1, p * 3) * out,
                   scale=max(0.05, ease_back(p, 2.4)), rot=7 + (1 - ease_out(p)) * -25)
        for i, sp in enumerate((self.hk_l1, self.hk_l2)):
            q = prog(t, 0.14 + 0.13 * i, 0.34)
            if q > 0:
                e = ease_back(q, 1.7)
                self.S(frame, sp, W / 2 + (1 - e) * (-260 if i == 0 else 260), 640 + i * 214, anchor="c",
                       alpha=min(1, q * 3) * out, scale=1.0)
        q = prog(t, 0.50, 0.3)
        if q > 0:
            self.S(frame, self.hk_slab, W / 2, 1010 + (1 - ease_back(q)) * 60, anchor="c", alpha=min(1, q * 3) * out,
                   rot=-2.5)
        pb = prog(t, 0.62, 0.4)
        if pb > 0:
            wob = math.sin(t * 6.0) * 4
            sc = (1.8 - 0.8 * ease_back(pb, 2.2)) * (1 + 0.03 * math.sin(t * math.pi * 4))
            self.S(frame, self.hk_sticker, 800, 1300, anchor="c", alpha=min(1, pb * 4) * out, scale=sc, rot=10 + wob)

    def lower_third(self, frame, t, t_in, t_out, key):
        if t < t_in or t >= t_out:
            return
        p = prog(t, t_in + 0.04, 0.42)
        e = ease_back(p, 1.5)
        out = 1 - prog(t, t_out - 0.15, 0.15)
        a = min(1, p * 3) * out
        x = 62 + (1 - e) * -420
        self.S(frame, self.tag[key], x, 1096, alpha=a, rot=2.5)
        self.S(frame, self.slab[key], x - 2, 1158, alpha=a, rot=2.5)
        for (c0, c1, txt) in CAPTIONS:
            if c0 <= t < c1:
                q = prog(t, c0 + 0.02, 0.24)
                eb = ease_back(q, 2.0)
                self.S(frame, self.caps[txt], 64 + (1 - eb) * -60, 1318, alpha=min(1, q * 3) * out)

    def price_card(self, frame, tl, key, old):
        cx = W / 2
        p = prog(tl, 0.02, 0.34)
        a = min(1, p * 3.5)
        e = ease_back(p, 1.8)
        self.S(frame, self.tag[key], cx, 968, anchor="c", alpha=a, scale=max(0.05, e))
        self.S(frame, self.h_title[key], cx, 1078, anchor="c", alpha=a, scale=max(0.05, 0.7 + 0.3 * e))
        # old price (left), arrow, sticker (right)
        ox, oy = 232, 1392
        ao = ease_out(prog(tl, 0.16, 0.2))
        self.S(frame, self.h_aulieu, ox, 1262, anchor="c", alpha=ao, rot=5)
        self.S(frame, self.h_old[old], ox, oy + (1 - ao) * 30, anchor="c", alpha=ao)
        ps = ease_io(prog(tl, 0.28, 0.17))
        if ps > 0:
            ln, lp = self.h_strike
            vis = ln[:, :max(4, int(ln.shape[1] * ps))]
            canvas = np.zeros_like(ln)
            canvas[:, :vis.shape[1]] = vis
            blit(frame, canvas, ox, oy + 6, anchor="c", rot=8)
        pa = prog(tl, 0.36, 0.26)
        if pa > 0:
            nudge = 8 * math.sin(tl * math.pi * 4)
            self.S(frame, self.h_arrow, 458 + nudge, oy + 4, anchor="c", alpha=min(1, pa * 3), scale=max(0.05, ease_back(pa, 2.5)))
        pn = prog(tl, 0.45, 0.34)
        if pn > 0:
            sc = (2.3 - 1.3 * ease_back(pn, 1.9)) * (1 + 0.035 * math.sin(max(0, tl - 0.79) * math.pi * 4))
            wob = 4 * math.sin(tl * 5.0)
            self.S(frame, self.h_sticker, 770, 1372, anchor="c", alpha=min(1, pn * 5), scale=sc,
                   rot=-8 + wob + (1 - ease_out(pn)) * 30)
            ps2 = prog(tl, 0.7, 0.3)
            if ps2 > 0:
                self.S(frame, self.h_seul, 588, 1196, anchor="c", alpha=min(1, ps2 * 3), scale=max(0.05, ease_back(ps2, 2.2)),
                       rot=8)

    def menu(self, frame, tl):
        cx = W / 2
        p = prog(tl, 0.03, 0.34)
        if p > 0:
            self.S(frame, self.mn_promo, 262, 268, anchor="c", alpha=min(1, p * 3), scale=max(0.05, ease_back(p, 2.3)), rot=7)
        q = prog(tl, 0.12, 0.34)
        if q > 0:
            e = ease_back(q, 1.6)
            self.S(frame, self.mn_title1, cx, 392 + (1 - e) * 50, anchor="c", alpha=min(1, q * 3))
            self.S(frame, self.mn_title2, cx, 508 + (1 - e) * 70, anchor="c", alpha=min(1, q * 3))
        for i, sp in enumerate(self.tiers):
            ts = 0.48 + 0.8 * i
            r = prog(tl, ts, 0.34)
            if r <= 0:
                continue
            e = ease_back(r, 1.7)
            y = 744 + i * 296
            self.S(frame, sp, cx + (1 - e) * (700 if i % 2 == 0 else -700), y, anchor="c", alpha=min(1, r * 4),
                   scale=1 + 0.06 * (1 - ease_out(prog(tl, ts + 0.3, 0.35))) * (1 if r >= 1 else 0),
                   rot=(-0.8, 0.8, -0.7)[i])

    def date_slab(self, frame, tl, start, y, scale=1.0):
        q = prog(tl, start, 0.3)
        if q > 0:
            self.S(frame, self.hk_slab, W / 2, y + (1 - ease_back(q)) * 70, anchor="c", alpha=min(1, q * 3), rot=-2.0,
                   scale=scale)

    def end(self, frame, tl):
        cx = W / 2
        a = prog(tl, 0.04, 0.3)
        if a > 0:
            self.S(frame, self.en_brand, cx, 300 + (1 - ease_back(a)) * -80, anchor="c", alpha=min(1, a * 3))
        b = prog(tl, 0.18, 0.3)
        if b > 0:
            self.S(frame, self.en_offre, cx, 452, anchor="c", alpha=min(1, b * 3), scale=max(0.05, ease_back(b, 2.2)), rot=5)
        c = prog(tl, 0.30, 0.28)
        if c > 0:
            self.S(frame, self.en_jusq, cx, 590 + (1 - ease_back(c)) * 40, anchor="c", alpha=min(1, c * 3))
        pd_ = prog(tl, 0.42, 0.36)
        if pd_ > 0:
            sc = (1.9 - 0.9 * ease_back(pd_, 1.9)) * (1 + 0.02 * math.sin(max(0, tl - 0.78) * math.pi * 4))
            self.S(frame, self.en_date, cx, 742, anchor="c", alpha=min(1, pd_ * 5), scale=sc, rot=-1.5)
        e = prog(tl, 0.62, 0.3)
        if e > 0:
            self.S(frame, self.en_inclus, cx + 250, 880, anchor="c", alpha=min(1, e * 3), scale=max(0.05, ease_back(e, 2.4)),
                   rot=-5)
        for i, key in enumerate(("K", "F")):
            r = prog(tl, 0.85 + 0.14 * i, 0.34)
            if r > 0:
                eb = ease_back(r, 1.7)
                self.S(frame, self.chip[key], cx + (-244 if i == 0 else 244), 1090 + (1 - eb) * 160, anchor="c",
                       alpha=min(1, r * 4), rot=(-1.5, 1.5)[i])
        g = prog(tl, 1.25, 0.3)
        if g > 0:
            self.S(frame, self.en_all, cx, 1280 + (1 - ease_out(g)) * 20, anchor="c", alpha=min(1, g * 3))
        hh = prog(tl, 1.4, 0.3)
        if hh > 0:
            self.S(frame, self.en_loc, cx, 1372 + (1 - ease_out(hh)) * 20, anchor="c", alpha=min(1, hh * 3))


# ------------------------------------------------------------------ render
def render(out, t0=0.0, t1=TOTAL, scale=1.0, stills=None, still_dir=None, audio=None):
    g = GFX()
    enc = None
    if out:
        ow, oh = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
               "-i", "-"]
        if audio:
            cmd += ["-ss", f"{t0:.3f}", "-i", audio]
        vf = [] if scale == 1.0 else [f"scale={ow}:{oh}:flags=area"]
        vf.append("format=yuv420p")
        cmd += ["-vf", ",".join(vf), "-c:v", "libx264", "-preset", "slow" if scale == 1.0 else "veryfast",
                "-crf", "17" if scale == 1.0 else "24", "-profile:v", "high", "-level", "4.2",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-movflags", "+faststart"]
        if audio:
            cmd += ["-af", "loudnorm=I=-13:TP=-1.0:LRA=7", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-shortest"]
        cmd += [out]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    still_frames = set(int(round(s * FPS)) for s in (stills or []))
    f0, f1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    for si, seg in enumerate(SEGS):
        s0, s1 = int(round(seg["t0"] * FPS)), int(round(seg["t1"] * FPS))
        if s1 <= f0 or s0 >= f1:
            continue
        if stills is not None and not out and not any(s0 <= k < s1 for k in still_frames):
            continue
        graphic_bg = seg.get("bg", False)
        src = None if graphic_bg else B.open_src(seg)
        nxt = SEGS[si + 1] if si + 1 < len(SEGS) else None
        for k in range(s1 - s0):
            fi = s0 + k
            raw = None if graphic_bg else src.read()
            if fi < f0 or fi >= f1:
                continue
            if stills is not None and not out and fi not in still_frames:
                continue
            t = fi / FPS
            tl = t - seg["t0"]
            dur = seg["t1"] - seg["t0"]
            look = seg["look"]
            if graphic_bg:
                frame = g.rays_menu.draw(g.menu_bg, t * 0.35, (255, 96, 70), g.menu_ray_a)
                frame = apply_mask(frame, g.masks["vig"])
            else:
                pz = ease_out(tl / dur) if look in ("prep", "hook") else tl / dur
                z = seg["z"][0] + (seg["z"][1] - seg["z"][0]) * pz
                if look == "prep" and tl < 0.2:
                    z *= 1 + 0.06 * (1 - ease_out(tl / 0.2))
                dx, dy = seg.get("shift", (0, 0))
                shake = 0.0
                if look in ("heroK", "heroF") and 0.45 <= tl < 0.8:
                    shake = (1 - (tl - 0.45) / 0.35) * 12
                frame = warp(raw, z, seg["f"][0], seg["f"][1], dx + shake * math.sin(tl * 95), dy + shake * math.cos(tl * 83))
                if look in ("prep", "hook", "heroK", "heroF"):
                    frame = bloom(frame, 0.20)
            tin = seg.get("tin")
            if tin == "whip" and tl < 0.2:
                frame = zoom_blur(frame, 1 - tl / 0.2, direction=-0.6)
            if nxt is not None and nxt.get("tin") == "whip" and dur - tl <= 0.2:
                frame = zoom_blur(frame, 1 - (dur - tl) / 0.2 + 1 / 6, direction=1)
            if look == "hook":
                frame = apply_mask(frame, g.masks["hook"])
            elif look == "prep":
                frame = apply_mask(frame, g.masks["prep"])
            elif look in ("heroK", "heroF"):
                frame = apply_mask(frame, g.masks["vig"])
                a = g.panel_a[..., None]
                frame = (frame.astype(np.float32) * (1 - a) + np.array((168, 12, 26), np.float32)[None, None, :] * a).astype(np.uint8)
                frame = g.rays_hero.draw(frame, t * 0.5, (255, 70, 60), g.hero_ray_a)
                frame = apply_mask(frame, 1 - 0.45 * np.clip((430 - np.arange(H, dtype=np.float32)) / 430, 0, 1)[:, None, None] ** 1.4)
            elif look == "end":
                tint = np.array((96, 6, 16), np.float32)
                frame = (frame.astype(np.float32) * 0.46 + tint * 0.52).astype(np.uint8)
                frame = apply_mask(frame, g.masks["vig"])
            # ---- graphics
            if look == "hook":
                g.hook(frame, t)
            g.lower_third(frame, t, 2.0, 10.0, "K")
            g.lower_third(frame, t, 12.0, 20.0, "F")
            if look == "heroK":
                g.price_card(frame, tl, "K", 38)
                g.date_slab(frame, tl, 0.85, 1716, 0.92)
                g.sparks.draw(frame, t, strength=prog(tl, 0.45, 0.2))
            if look == "heroF":
                g.price_card(frame, tl, "F", 35)
                g.date_slab(frame, tl, 0.85, 1716, 0.92)
                g.sparks.draw(frame, t, strength=prog(tl, 0.45, 0.2))
            if look == "menu":
                g.menu(frame, tl)
                g.date_slab(frame, tl, 2.75, 1600)
                g.sparks.draw(frame, t, strength=0.6)
            if look == "end":
                g.end(frame, tl)
                g.sparks.draw(frame, t, strength=0.9)
            g.top_bar(frame, t)
            # ---- flashes
            if tin == "flash" and tl < 0.3:
                frame = flash(frame, 0.9 * (1 - tl / 0.3) ** 1.6, (255, 248, 220))
            if look in ("heroK", "heroF") and 0.45 <= tl < 0.7:
                frame = flash(frame, 0.30 * (1 - (tl - 0.45) / 0.25), (255, 236, 150))
            if t < 0.2:
                frame = flash(frame, 0.6 * (1 - t / 0.2))
            if t > TOTAL - 0.3:
                frame = (frame.astype(np.float32) * (1 - 0.9 * prog(t, TOTAL - 0.3, 0.3))).astype(np.uint8)
            if enc:
                enc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if fi in still_frames and still_dir:
                Image.fromarray(frame).save(f"{still_dir}/f_{t:06.2f}.jpg", quality=90)
        if src:
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
