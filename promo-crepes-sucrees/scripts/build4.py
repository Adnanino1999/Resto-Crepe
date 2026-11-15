"""Resto Crêpes – promo ad v4: v2 look + second Darija voice-over, 30 s."""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build2 as B2  # noqa: E402
from build2 import W, YEL, INK, ease_back, prog, tsprite  # noqa: E402

TOTAL = 30.0
B2.TOTAL = TOTAL
B2.SEGS[-1]["t1"] = TOTAL
B2.SEGS[-1]["speed"] = 0.4
B2.SEGS[-1]["z"] = (1.04, 1.20)

_end = B2.GFX.end


def end_with_cta(self, frame, tl):
    _end(self, frame, tl)
    if not hasattr(self, "en_cta"):
        self.en_cta = tsprite("On vous attend !", "Kaushan", 96, YEL, stroke=7, extrude=(5, 7, INK), shadow=0.5)
    p = prog(tl, 1.95, 0.4)                 # lands with the last spoken line
    if p > 0:
        sc = max(0.05, ease_back(p, 2.3)) * (1 + 0.03 * math.sin(max(0, tl - 2.35) * math.pi * 4))
        self.S(frame, self.en_cta, W / 2, 1500, anchor="c", alpha=min(1, p * 3), scale=sc, rot=4)


B2.GFX.end = end_with_cta

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="")
    ap.add_argument("--stills", default="")
    ap.add_argument("--still-dir", default="")
    ap.add_argument("--audio", default="")
    a = ap.parse_args()
    st = [float(x) for x in a.stills.split(",")] if a.stills else None
    if a.still_dir:
        os.makedirs(a.still_dir, exist_ok=True)
    B2.render(a.out, 0.0, TOTAL, 1.0, st, a.still_dir, a.audio or None)
