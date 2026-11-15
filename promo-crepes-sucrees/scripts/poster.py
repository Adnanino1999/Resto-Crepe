"""Resto Crêpes – printable promo poster (A3 portrait, 300 dpi; also prints as A4).

Usage: poster.py OUT_BASENAME      -> OUT_BASENAME.png and OUT_BASENAME.pdf
Same look as the video (v2): red / yellow, sticker prices "old -> new".
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build2 as B2  # noqa: E402
from build2 import (INK, WHITE, YEL, RED, RED_D, PISTA, PINK, GREY, tsprite, box_sprite, label_sprite, starburst,  # noqa: E402
                    arrow_sprite, old_price_sprite, inner)
from build import blit_rgba, font  # noqa: E402

PW, PH = 3508, 4961            # A3 at 300 dpi
CLIPS = "/home/user/work/clips/Crepe"
M = 170                        # outer margin (~14 mm)


# ------------------------------------------------------------------ helpers
def paste(canvas, sp, x, y, anchor="tl", rot=0.0):
    """Composite sprite (rgba, pad) on the RGBA canvas. (x, y) = top-left of the un-padded content, or centre."""
    arr, pad = sp
    if rot:
        h, w = arr.shape[:2]
        diag = int(math.ceil(math.hypot(w, h))) + 4
        Mx = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
        Mx[0, 2] += diag / 2 - w / 2
        Mx[1, 2] += diag / 2 - h / 2
        rotated = cv2.warpAffine(arr, Mx, (diag, diag), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
        if anchor == "c":
            blit_rgba(canvas, rotated, x - diag / 2, y - diag / 2)
        else:
            blit_rgba(canvas, rotated, x - pad + w / 2 - diag / 2, y - pad + h / 2 - diag / 2)
        return
    if anchor == "c":
        blit_rgba(canvas, arr, x - arr.shape[1] / 2, y - arr.shape[0] / 2)
    elif anchor == "tc":
        blit_rgba(canvas, arr, x - arr.shape[1] / 2, y - pad)
    else:
        blit_rgba(canvas, arr, x - pad, y - pad)


def grade(img):
    """Warm, saturated food grade for print."""
    a = np.asarray(img).astype(np.float32)
    a = (a - 128) * 1.10 + 128 + 6
    g = a.mean(axis=2, keepdims=True)
    a = g + (a - g) * 1.26
    a[..., 0] *= 1.04
    a[..., 2] *= 0.95
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def rounded(img, radius):
    S = 2
    m = Image.new("L", (img.width * S, img.height * S), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, m.width - 1, m.height - 1], radius * S, fill=255)
    out = img.convert("RGBA")
    out.putalpha(m.resize(img.size, Image.LANCZOS))
    return np.array(out)


def sharpest_frame(clip, t0, n=36):
    """Pick the sharpest of n consecutive video frames and denoise it lightly -> clean still for print."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0}", "-i", f"{CLIPS}/IMG_{clip}.MOV", "-frames:v", str(n),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 1920, 1072, 3)
    score = [cv2.Laplacian(cv2.cvtColor(f[1100:1850], cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in fr]
    best = int(np.argmax(score))
    print(f"sharpest frame {best}/{len(fr)} of {clip} (score {score[best]:.0f}, median {np.median(score):.0f})")
    return Image.fromarray(cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(fr[best]), None, 4, 6, 5, 15))


def photo(img, box, size, radius):
    """Crop box (x0, y0, x1, y1) from img, resize to size, sharpen, grade, round corners."""
    im = img.crop(box).resize(size, Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=4, percent=90, threshold=2))
    return rounded(grade(im), radius)


def price_row(old, new, s_old=150, s_new=250):
    """Sprite: struck old price -> arrow -> big red new price."""
    o = old_price_sprite(old, s_old, GREY, strike=RED)
    a = arrow_sprite(int(s_old * 0.78), int(s_old * 0.66), fill=YEL, ow=max(5, s_old // 16))
    n = tsprite(str(new), "Lilita", s_new, RED, stroke=max(6, s_new // 20), extrude=(max(4, s_new // 30), max(5, s_new // 24), INK))
    d = tsprite("DH", "Lilita", int(s_new * 0.5), RED, stroke=max(5, s_new // 26), extrude=(max(3, s_new // 40), max(4, s_new // 32), INK))
    ow_, oh = inner(o)
    aw, ah = inner(a)
    nw, nh = inner(n)
    dw, dh = inner(d)
    g1, g2, g3 = int(s_old * 0.22), int(s_old * 0.26), int(s_new * 0.05)
    w = ow_ + g1 + aw + g2 + nw + g3 + dw
    h = nh
    pad = 60
    arr = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.uint8)
    x = pad
    blit_rgba(arr, o[0], x - o[1], pad + (h - oh) / 2 + oh * 0.06 - o[1])
    x += ow_ + g1
    blit_rgba(arr, a[0], x - a[1], pad + (h - ah) / 2 + ah * 0.06 - a[1])
    x += aw + g2
    blit_rgba(arr, n[0], x - n[1], pad - n[1])
    x += nw + g3
    blit_rgba(arr, d[0], x - d[1], pad + nh - dh - nh * 0.11 - d[1])
    return arr, pad


def star(canvas, x, y, d, alpha=1.0):
    yy, xx = np.mgrid[0:d, 0:d].astype(np.float32)
    dx, dy = np.abs(xx - d / 2 + .5), np.abs(yy - d / 2 + .5)
    a = np.clip(np.exp(-(dx * dx + dy * dy) / (2 * (d / 9) ** 2)) + np.exp(-dx / (d * 0.02 + 0.6)) * np.exp(-dy / (d * 0.22))
                + np.exp(-dy / (d * 0.02 + 0.6)) * np.exp(-dx / (d * 0.22)), 0, 1) * alpha
    s = np.zeros((d, d, 4), np.uint8)
    s[..., 0], s[..., 1], s[..., 2] = 255, 246, 200
    s[..., 3] = (a * 255).astype(np.uint8)
    blit_rgba(canvas, s, x - d / 2, y - d / 2)


# ------------------------------------------------------------------ background
yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
cx, cy = PW / 2, 1000.0
d = np.clip(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / 4300, 0, 1)
k = ((1 - d) ** 1.2)[..., None]
bg = np.array((240, 40, 44), np.float32) * k + np.array(RED_D, np.float32) * (1 - k)
theta = np.arctan2(yy - cy, xx - cx)
rays = np.clip(0.5 + 6.0 * np.sin(18 * theta + 0.3), 0, 1) * 0.26 * (1 - d * 0.6)
bg = bg * (1 - rays[..., None]) + np.array((255, 96, 70), np.float32) * rays[..., None]
vig = 1 - 0.30 * np.clip(np.sqrt(((xx - PW / 2) / (PW * 0.8)) ** 2 + ((yy - PH / 2) / (PH * 0.75)) ** 2) - 0.4, 0, 1) ** 1.5
bg *= vig[..., None]
canvas = np.dstack([np.clip(bg, 0, 255).astype(np.uint8), np.full((PH, PW), 255, np.uint8)])
del yy, xx, d, k, theta, rays, vig, bg

# ------------------------------------------------------------------ header
brand = tsprite("RESTO CRÊPES", "Lilita", 240, WHITE, stroke=18, extrude=(14, 18, INK), tracking=10)
paste(canvas, brand, PW / 2, 150, anchor="tc")
title = tsprite("CRÊPES SUCRÉES", "Lilita", 470, WHITE, stroke=28, extrude=(22, 28, INK), maxw=3120)
paste(canvas, title, PW / 2, 1000, anchor="c")
promo = tsprite("Promo !", "Kaushan", 400, YEL, stroke=22, extrude=(15, 20, INK))
paste(canvas, promo, 860, 640, anchor="c", rot=7)
st, spad = starburst(540, n=24, ow=12, hard=(16, 20))
sc = st.shape[0] / 2
t1 = tsprite("DÈS", "Montserrat-Black", 64, INK, tracking=6)
blit_rgba(st, t1[0], sc - t1[0].shape[1] / 2, sc - 205 - t1[1])
n15 = tsprite("15", "Lilita", 240, RED, stroke=10, extrude=(6, 8, INK))
dh15 = tsprite("DH", "Lilita", 104, RED, stroke=7, extrude=(4, 6, INK))
nw_, dw_ = inner(n15)[0], inner(dh15)[0]
x15 = sc - (nw_ + 10 + dw_) / 2
blit_rgba(st, n15[0], x15 - n15[1], sc - 150 - n15[1])
blit_rgba(st, dh15[0], x15 + nw_ + 10 - dh15[1], sc - 150 + 122 - dh15[1])
o20 = old_price_sprite(20, 76, INK, strike=RED)
blit_rgba(st, o20[0], sc - o20[0].shape[1] / 2, sc + 104 - o20[1])
paste(canvas, (st, spad), 2890, 600, anchor="c", rot=10)

# ------------------------------------------------------------------ the two star offers
kou = Image.open(f"{CLIPS}/photo_6035243488808473032_w.jpg").convert("RGB")
four = sharpest_frame("9992", 2.75, 36)
CW, CH_ = 1540, 1500
PHW, PHH = 1480, 1090
cards = [
    dict(x=M, name="KOUNAFA PISTACHE", tag="NOUVEAU", accent=PISTA, name_col=INK, old=38, new=30,
         img=photo(kou, (410, 1385, 1920, 2497), (PHW, PHH), 46), rot=-1.0),
    dict(x=PW - M - CW, name="CRÊPE 4 YOU", tag="CHOIX DU CHEF", accent=PINK, name_col=WHITE, old=35, new=30,
         img=photo(four, (70, 1130, 1020, 1830), (PHW, PHH), 46), rot=1.0),
]
CY = 1340
for c in cards:
    card, pad = box_sprite(CW, CH_, (255, 252, 244), 76, outline=INK, ow=14, hard=(22, 28))
    blit_rgba(card, c["img"], pad + 30, pad + 30)
    # ink frame around the photo
    S = 2
    fr = Image.new("RGBA", (PHW * S, PHH * S), (0, 0, 0, 0))
    ImageDraw.Draw(fr).rounded_rectangle([0, 0, PHW * S - 1, PHH * S - 1], 46 * S, outline=INK + (255,), width=10 * S)
    blit_rgba(card, np.array(fr.resize((PHW, PHH), Image.LANCZOS)), pad + 30, pad + 30)
    name = label_sprite(c["name"], "Lilita", 150, c["name_col"], c["accent"], tracking=2, padx=54, pady=14, radius=34,
                        outline=INK, ow=10, hard=(14, 18), stroke=8 if c["name_col"] == WHITE else 0, maxw=1260)
    paste(card, name, pad + CW / 2, pad + 30 + PHH - 20, anchor="c", rot=2.0)
    row = price_row(c["old"], c["new"], 150, 260)
    paste(card, row, pad + CW / 2, pad + 30 + PHH + 232, anchor="c")
    tag = label_sprite(c["tag"], "Montserrat-Black", 70, INK, YEL, tracking=8, padx=44, pady=20, outline=INK, ow=8,
                       hard=(10, 12))
    paste(card, tag, pad + 40 + inner(tag)[0] / 2 + 10, pad + 70, anchor="c", rot=5)
    paste(canvas, (card, pad), c["x"], CY, rot=c["rot"])

# ------------------------------------------------------------------ full list
sec = tsprite("ET TOUTES NOS CRÊPES SUCRÉES EN PROMO", "Lilita", 168, WHITE, stroke=14, extrude=(10, 13, INK), maxw=3160)
paste(canvas, sec, PW / 2, 3010, anchor="c")
LX, LY, LW, LH = M, 3130, PW - 2 * M, 1150
panel, pad = box_sprite(LW, LH, (255, 252, 244), 70, outline=INK, ow=14, hard=(22, 28))
items = [("Nutella", 20, 15), ("Caramel", 20, 15), ("Banane", 25, 20), ("Framboise", 25, 20), ("Oreo", 25, 20),
         ("KitKat", 25, 20), ("Milka", 30, 25), ("Snickers", 30, 25), ("Twix", 30, 25), ("Kinder", 30, 25),
         ("Lotus", 30, 25), ("Pistache", 30, 25), ("Chocolat blanc", 30, 25)]
cols = [items[:7], items[7:]]
colw = (LW - 3 * 80) // 2
rowh = (LH - 2 * 44) / 7
pd_ = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
for ci, col in enumerate(cols):
    x0 = pad + 80 + ci * (colw + 80)
    for ri, (nm, old, new) in enumerate(col):
        yc = pad + 44 + rowh * (ri + 0.5)
        t = tsprite(nm, "Montserrat-ExtraBold", 92, INK)
        blit_rgba(panel, t[0], x0 - t[1], yc - inner(t)[1] / 2 - t[1])
        pr = price_row(old, new, 84, 124)
        prw = inner(pr)[0]
        blit_rgba(panel, pr[0], x0 + colw - prw - pr[1], yc - pr[0].shape[0] / 2 + 2)
        # dotted leader between name and prices
        xa, xb = x0 + inner(t)[0] + 36, x0 + colw - prw - 36
        for xd in range(int(xa), int(xb), 30):
            cv2.circle(panel, (xd, int(yc + 22)), 5, (190, 176, 168, 255), -1, cv2.LINE_AA)
        if ri < len(col) - 1 or (ci == 1 and ri == len(col) - 1):
            cv2.line(panel, (int(x0), int(yc + rowh / 2)), (int(x0 + colw), int(yc + rowh / 2)), (232, 222, 210, 255), 4)
cv2.line(panel, (pad + LW // 2, pad + 60), (pad + LW // 2, pad + LH - 60), INK + (255,), 6)
paste(canvas, (panel, pad), LX, LY)

# ------------------------------------------------------------------ footer
FW, FH = PW - 2 * M - 60, 430
foot, fpad = box_sprite(FW, FH, YEL, 60, outline=INK, ow=14, hard=(22, 28))
fo = tsprite("OFFRE EXCLUSIVE", "Montserrat-Black", 96, RED, tracking=22)
blit_rgba(foot, fo[0], fpad + (FW - inner(fo)[0]) / 2 - fo[1], fpad + 30 - fo[1])
ft = tsprite("JUSQU'AU 22 NOVEMBRE INCLUS !", "Lilita", 226, INK, maxw=FW - 160)
blit_rgba(foot, ft[0], fpad + (FW - inner(ft)[0]) / 2 - ft[1], fpad + 140 - ft[1])
paste(canvas, (foot, fpad), M + 30, 4400, rot=-1.0)

# sparkles
rng = np.random.default_rng(4)
for _ in range(46):
    x, y = rng.uniform(80, PW - 80), rng.uniform(80, PH - 80)
    if (M < x < PW - M and 1330 < y < 2880) or (M < x < PW - M and 3120 < y < 4300):
        continue
    star(canvas, x, y, int(rng.uniform(50, 150)), alpha=rng.uniform(0.5, 1.0))

out = sys.argv[1]
img = Image.fromarray(canvas[..., :3], "RGB")
img.save(out + ".png", dpi=(300, 300), optimize=True)
img.save(out + ".pdf", "PDF", resolution=300.0, quality=95)
img.resize((PW // 4, PH // 4), Image.LANCZOS).save(out + "_apercu.jpg", quality=88)
print("wrote", out, img.size)
