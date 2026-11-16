"""Resto Crêpes – QR code to the PDF menu + printable table card (A6) and an A4 sheet of 4 cards.

Usage: qr_menu.py OUT_DIR [URL]
The card follows the look of the menu itself: cream paper, serif titles, gold details.
"""
import os
import sys

import cv2
import numpy as np
import qrcode
import qrcode.image.svg
from PIL import Image, ImageDraw, ImageFont

FONTS = "/home/user/work/fonts"
OUT = sys.argv[1]
URL = sys.argv[2] if len(sys.argv) > 2 else "https://drive.google.com/file/d/15xqoeFGZX6tbDpYRWcMLqP6hUgMQ1pk0/view"
os.makedirs(OUT, exist_ok=True)

CREAM = (247, 241, 229)
INK = (34, 24, 18)
GOLD = (176, 132, 62)
GREY = (120, 106, 94)


def font(name, size):
    return ImageFont.truetype(f"{FONTS}/{name}.ttf", size)


def text_c(d, cx, y, txt, f, fill, tracking=0):
    """Draw centred text with optional letter-spacing; returns the text height."""
    w = sum(f.getlength(ch) + tracking for ch in txt) - tracking
    x = cx - w / 2
    for ch in txt:
        d.text((x, y), ch, font=f, fill=fill)
        x += f.getlength(ch) + tracking
    return w


# ------------------------------------------------------------------ QR code
qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_Q, box_size=40, border=4)
qr.add_data(URL)
qr.make(fit=True)
qr_img = qr.make_image(fill_color=INK, back_color="white").convert("RGB")
qr_img.save(f"{OUT}/qr_menu.png", dpi=(300, 300))
qrcode.make(URL, image_factory=qrcode.image.svg.SvgPathImage, error_correction=qrcode.constants.ERROR_CORRECT_Q,
            box_size=20, border=4).save(f"{OUT}/qr_menu.svg")
print("QR:", qr_img.size, "modules:", qr.modules_count, "->", URL)

# ------------------------------------------------------------------ table card, A6 portrait at 300 dpi
W, H = 1240, 1748
S = 2                                                       # supersampling for smooth lines
card = Image.new("RGB", (W * S, H * S), CREAM)
d = ImageDraw.Draw(card)
d.rectangle([44 * S, 44 * S, (W - 44) * S, (H - 44) * S], outline=GOLD, width=4 * S)
d.rectangle([60 * S, 60 * S, (W - 60) * S, (H - 60) * S], outline=GOLD, width=1 * S)
cx = W * S / 2
text_c(d, cx, 138 * S, "BIENVENUE CHEZ", font("Montserrat-Bold", 26 * S), GOLD, tracking=9 * S)
text_c(d, cx, 180 * S, "RESTO CRÊPES", font("Playfair-Bold", 112 * S), INK, tracking=3 * S)
# rule with a diamond
ry = 352 * S
d.line([cx - 190 * S, ry, cx - 22 * S, ry], fill=GOLD, width=2 * S)
d.line([cx + 22 * S, ry, cx + 190 * S, ry], fill=GOLD, width=2 * S)
d.polygon([(cx, ry - 10 * S), (cx + 10 * S, ry), (cx, ry + 10 * S), (cx - 10 * S, ry)], fill=GOLD)
text_c(d, cx, 384 * S, "LA CARTE", font("Playfair-Black", 104 * S), INK, tracking=20 * S)
# QR panel
PW_, PY = 800, 566
px0 = (W - PW_) // 2
d.rounded_rectangle([(px0 + 8) * S, (PY + 10) * S, (px0 + PW_ + 8) * S, (PY + PW_ + 10) * S], 36 * S, fill=(226, 214, 194))
d.rounded_rectangle([px0 * S, PY * S, (px0 + PW_) * S, (PY + PW_) * S], 36 * S, fill="white", outline=GOLD, width=5 * S)
card = card.resize((W, H), Image.LANCZOS)
QS = 716
card.paste(qr_img.resize((QS, QS), Image.NEAREST if QS % qr_img.width == 0 else Image.LANCZOS), ((W - QS) // 2, PY + (PW_ - QS) // 2))
d = ImageDraw.Draw(card)
text_c(d, W / 2, 1404, "Scannez-moi", font("Playfair-BoldItalic", 92), GOLD)
text_c(d, W / 2, 1530, "pour découvrir tout notre menu", font("Montserrat-Bold", 40), INK)
text_c(d, W / 2, 1596, "Ouvrez l'appareil photo de votre téléphone et visez le code", font("Montserrat-Medium", 26), GREY)
text_c(d, W / 2, 1646, "Good Food, Good Mood", font("Playfair-Italic", 32), GOLD)
card.save(f"{OUT}/carte_table_A6.png", dpi=(300, 300))
card.save(f"{OUT}/carte_table_A6.pdf", "PDF", resolution=300.0, quality=95)

# ------------------------------------------------------------------ A4 sheet with 4 cards and cut marks
A4W, A4H = 2480, 3508
sheet = Image.new("RGB", (A4W, A4H), "white")
for i in range(4):
    sheet.paste(card, ((i % 2) * W, (i // 2) * H + (A4H - 2 * H) // 2))
sd = ImageDraw.Draw(sheet)
y0 = (A4H - 2 * H) // 2
for y in (y0, y0 + H, y0 + 2 * H):                           # light cut guides in the margins only
    sd.line([0, y, A4W, y], fill=(200, 200, 200), width=1)
sd.line([W, 0, W, y0 - 1], fill=(160, 160, 160), width=2)
sd.line([W, y0 + 2 * H + 1, W, A4H], fill=(160, 160, 160), width=2)
sd.line([W, y0, W, y0 + 2 * H], fill=(215, 205, 190), width=1)
sheet.save(f"{OUT}/planche_A4_4_cartes.pdf", "PDF", resolution=300.0, quality=95)

# ------------------------------------------------------------------ verification: decode what was produced
det = cv2.QRCodeDetector()
for name, img in (("qr_menu.png", qr_img), ("carte_table_A6", card), ("carte at phone-photo size", card.resize((W // 3, H // 3), Image.LANCZOS)),
                  ("A4 sheet (reduced)", sheet.resize((A4W // 3, A4H // 3), Image.LANCZOS))):
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    val, pts, _ = det.detectAndDecode(arr)
    if not val:
        ok, vals, _, _ = det.detectAndDecodeMulti(arr)
        val = vals[0] if ok and vals else ""
    print(f"decode {name:28s}: {'OK' if val == URL else 'FAILED'}  {val[:70]}")
