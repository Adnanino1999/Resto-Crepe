"""Original music v2 for the 'Formules 45 DH' ad: Moroccan-flavoured afro-house in D Hijaz (120 BPM).
Darbuka (maqsum), shaker, deep kick, oud-like Karplus-Strong plucks, reed lead, warm pad.
Comic section for the comeback scene, record-scratch stop on the freeze frame, final drop on the end card.
Fully synthesized. Usage: music_hijaz.py MUSIC_WAV FX_WAV
"""
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthlib as L  # noqa: E402
from synthlib import (SR, BEAT, STEP, BAR, midi, lp, hp, bp, kick, clap, crash, impact, whoosh, swish, pop, riser,  # noqa: E402
                      sub_drop, bell, reverb, brass)

TOTAL = 49.30
N = int(SR * (TOTAL + 2.0))
L.N = N
rng = np.random.default_rng(23)

# ---- edit events (must match build45.py)
ORIGIN = 4.25
SUPERCUT_CUTS = [4.75, 5.25, 5.75, 6.25, 6.75, 7.25]
BROLL_CUTS = [8.55, 9.00, 9.45, 10.55, 16.81, 17.11, 17.41, 18.26]
CHIP_POPS = [8.62, 9.98, 11.13, 16.50, 17.80, 19.00, 35.90]
CARD_IN = [12.35, 19.76]
SLAMS = [12.72, 20.25]
SKIP0, SKIP1 = 24.25, 25.45          # time skip
COME = 25.45                         # comeback scene grid origin
JUMPS = [30.95, 35.35]               # jump cuts in the comeback scene
UP = 37.60                           # he gets up
FREEZE0, FREEZE1 = 39.20, 40.10
END0, STEP_B, CTA = 41.80, 44.60, 47.30
END_POPS = [41.95, 42.15, 42.40, 42.70, 43.05, 44.75, 45.00, 45.25]


def buf():
    return np.zeros((N, 2))


def A(b, snd, t, g=1.0, pan=0.0):
    L.add(b, snd, t, g, pan)


# ---------------------------------------------------------------- new instruments
def ks(note, dur=0.5, damp=0.994, bright=0.55, body=True):
    """Karplus-Strong plucked string (oud / qanun colour)."""
    f = midi(note)
    P = max(2, int(round(SR / f)))
    n = int(dur * SR)
    out = np.zeros(n + P + 1)
    exc = rng.uniform(-1, 1, P)
    exc = lp(exc, 1500 + 6000 * bright)
    out[1:P + 1] = exc
    i = P + 1
    while i < n + P + 1:
        j = min(i + P, n + P + 1)
        out[i:j] = damp * 0.5 * (out[i - P:j - P] + out[i - P - 1:j - P - 1])
        i = j
    y = out[P + 1:P + 1 + n]
    if body:
        y = y + 0.6 * bp(y, 700, 2600)
    t = np.arange(n) / SR
    y *= np.minimum(1, (dur - t) / 0.03).clip(0, 1)
    return y / (np.abs(y).max() + 1e-9) * 0.5


def doum():
    n = int(0.32 * SR)
    t = np.arange(n) / SR
    f = 82 + 70 * np.exp(-t / 0.03)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)
    x += lp(rng.standard_normal(n), 900) * np.exp(-t / 0.012) * 0.4
    return np.tanh(x * 1.8) * 0.8


def tek(g=1.0):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    ring = np.sin(2 * np.pi * 820 * t) * np.exp(-t / 0.025) + 0.5 * np.sin(2 * np.pi * 1730 * t) * np.exp(-t / 0.015)
    nz = bp(rng.standard_normal(n), 2500, 9000) * np.exp(-t / 0.012)
    return (ring * 0.5 + nz * 0.7) * 0.55 * g


def ka():
    n = int(0.07 * SR)
    t = np.arange(n) / SR
    return bp(rng.standard_normal(n), 1800, 6000) * np.exp(-t / 0.008) * 0.3


def shaker(acc=1.0):
    n = int(0.07 * SR)
    t = np.arange(n) / SR
    e = np.minimum(1, t / 0.008) * np.exp(-t / 0.03)
    return hp(rng.standard_normal(n), 6000, 4) * e * 0.22 * acc


def reed(note, dur, prev=None, gain=1.0):
    """Ghaita-like reed lead: nasal saw/square blend, vibrato, small portamento."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f1 = midi(note)
    f0 = midi(prev) if prev is not None else f1
    f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    f = f * (1 + 0.007 * np.sin(2 * np.pi * 5.6 * t) * np.minimum(1, t / 0.15))
    ph = np.cumsum(f) / SR
    x = 0.6 * (2 * (ph % 1) - 1) + 0.4 * np.sign(np.sin(2 * np.pi * ph))
    x = bp(x, 700, 4200) + 0.5 * bp(x, 1300, 1900)
    e = np.minimum(1, t / 0.03) * np.minimum(1, (dur - t) / 0.05).clip(0, 1)
    return np.tanh(x * e * 1.4) * 0.28 * gain


def pad(notes, dur, cutoff=1500):
    n = int(dur * SR)
    t = np.arange(n) / SR
    l, r = np.zeros(n), np.zeros(n)
    for nn in notes:
        f = midi(nn)
        for k, d in enumerate((-8, -2, 3, 9)):
            s = 2 * ((t * f * 2 ** (d / 1200) + rng.random()) % 1) - 1
            if k % 2:
                l += s
                r += 0.35 * s
            else:
                r += s
                l += 0.35 * s
    x = np.stack([l, r], axis=1) / (3.0 * len(notes))
    e = np.minimum(1, t / 0.3) * np.minimum(1, (dur - t) / 0.35).clip(0, 1)
    return lp(x, cutoff) * e[:, None] * 0.55


def bass(note, dur, stacc=False):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = midi(note)
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) + 0.25 * (2 * ((t * f) % 1) - 1)
    x = lp(x, 700)
    e = np.minimum(1, t / 0.004) * (np.exp(-t / 0.09) if stacc else (0.65 + 0.35 * np.exp(-t / 0.12)))
    e *= np.minimum(1, (dur - t) / 0.01).clip(0, 1)
    return np.tanh(x * e * 1.8) * 0.6


def record_scratch():
    n = int(0.42 * SR)
    t = np.arange(n) / SR
    rate = np.interp(t, [0, 0.08, 0.16, 0.26, 0.42], [1.0, -1.6, 1.8, -1.2, 0.0])
    ph = np.cumsum(rate) / SR
    base = rng.standard_normal(int(0.6 * SR))
    idx = (np.abs(ph) * SR * 0.9).astype(int) % len(base)
    x = base[idx]
    fc = 600 + 2600 * np.abs(rate) / 1.8
    out = np.zeros(n)
    for i in range(0, n, 256):
        s = signal.butter(2, [max(200, fc[i] * 0.6), min(fc[i] * 1.8, SR * 0.45)], 'band', fs=SR, output='sos')
        out[i:i + 256] = signal.sosfilt(s, x[i:i + 256])
    return np.tanh(out * 3) * np.minimum(1, (0.42 - t) / 0.05).clip(0, 1) * 0.8


def boing():
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    f = 180 + 260 * (1 - np.exp(-t / 0.05)) + 60 * np.sin(2 * np.pi * 11 * t) * np.exp(-t / 0.25)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22) * 0.55


def fast_forward(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    whine = np.sin(2 * np.pi * np.cumsum(1800 + 2600 * t / dur) / SR) * 0.12
    gar = bp(rng.standard_normal(n), 500, 4000) * (0.5 + 0.5 * np.sin(2 * np.pi * 14 * t)) * 0.35
    e = np.minimum(1, t / 0.08) * np.minimum(1, (dur - t) / 0.12).clip(0, 1)
    return (whine + gar) * e


def tick():
    n = int(0.03 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 3200 * t) * 0.6 + hp(rng.standard_normal(n), 4000) * 0.4) * np.exp(-t / 0.004) * 0.4


def glitch(dur=0.12):
    n = int(dur * SR)
    sq = np.sign(np.sin(2 * np.pi * rng.uniform(300, 1400) * np.arange(n) / SR))
    sq *= (rng.random(n // 200 + 1).repeat(200)[:n] > 0.3)
    return sq * 0.18


# ---------------------------------------------------------------- harmony (D Hijaz)
PROG = [  # (bass root, pad chord, oud riff {step: note})
    (38, (62, 66, 69), {0: 62, 2: 63, 3: 66, 4: 67, 6: 66, 7: 63, 8: 62, 10: 69, 11: 67, 12: 66, 14: 63, 15: 62}),
    (39, (63, 67, 70), {0: 63, 2: 67, 3: 66, 4: 63, 6: 62, 8: 63, 10: 67, 11: 70, 12: 69, 14: 67, 15: 66}),
    (36, (60, 63, 67), {0: 72, 2: 70, 3: 69, 4: 67, 6: 66, 7: 67, 8: 63, 10: 66, 11: 67, 12: 69, 14: 70, 15: 69}),
    (38, (62, 66, 69), {0: 74, 2: 72, 3: 70, 4: 69, 6: 67, 7: 66, 8: 63, 10: 66, 12: 62}),
]
REED = {0: [(0, 74, 4), (4, 75, 2), (6, 78, 2), (8, 79, 4), (12, 78, 2), (14, 75, 2)],
        1: [(0, 75, 4), (4, 79, 4), (8, 82, 4), (12, 81, 4)],
        2: [(0, 84, 4), (4, 82, 2), (6, 81, 2), (8, 79, 4), (12, 78, 4)],
        3: [(0, 81, 4), (4, 79, 2), (6, 78, 2), (8, 75, 4), (12, 74, 4)]}
MAQSUM = {0: 'D', 2: 'T', 6: 'T', 8: 'D', 12: 'T', 4: 'k', 10: 'k', 14: 'k', 15: 'k'}
SNEAK = {0: [(0, 62), (4, 66), (8, 69), (12, 67), (14, 66)], 1: [(0, 63), (4, 67), (8, 70), (12, 69), (14, 67)],
         2: [(0, 60), (4, 63), (8, 67), (12, 66), (14, 63)], 3: [(0, 62), (2, 63), (4, 66), (6, 67), (8, 69), (12, 62)]}

drums, perc, bassb, oud, reedb, padb, mfx, fx = (buf() for _ in range(8))
K, C, CR = kick(), clap(), crash()
DOUM = doum()
pump_pts = []


def do_kick(t, g=1.0):
    A(drums, K, t, g)
    pump_pts.append(t)


def bar_groove(t0, bar, style, t_end=None):
    """One bar (2 s) of a given style starting at t0."""
    root, chord, riff = PROG[bar % 4]
    full = style in ('drop', 'final')
    if style in ('groove', 'hero', 'final', 'drop', 'intro', 'sneak', 'lite'):
        A(padb, pad(chord, BAR, 1100 if style in ('intro', 'sneak', 'lite', 'groove') else 1900), t0,
          0.5 if style != 'sneak' else 0.3)
    for s in range(16):
        ts = t0 + s * STEP + (0.018 if s % 2 else 0.0)          # light swing
        if t_end is not None and ts >= t_end - 1e-6:
            break
        b = s // 4
        # kick / clap
        if style in ('drop', 'final', 'groove', 'lite') and s % 4 == 0:
            do_kick(t0 + s * STEP, 1.0 if full else (0.7 if style == 'groove' else 0.55))
        if style in ('drop', 'final', 'groove') and s in (4, 12):
            A(drums, C, t0 + s * STEP, 0.8 if full else 0.45, pan=0.05)
        if style == 'hero' and s in (0, 6):
            do_kick(t0 + s * STEP, 1.0)
        # darbuka maqsum
        hit = MAQSUM.get(s)
        if hit and style != 'intro' or (style == 'intro' and hit in ('D', 'T')):
            g = {'drop': 1.0, 'final': 1.0, 'groove': 0.75, 'hero': 0.8, 'lite': 0.6, 'sneak': 0.55, 'intro': 0.5}.get(style, 0.6)
            if hit == 'D':
                A(perc, DOUM, ts, 0.9 * g, pan=-0.1)
            elif hit == 'T':
                A(perc, tek(), ts, 0.85 * g, pan=0.25)
            elif style in ('drop', 'final', 'groove'):
                A(perc, ka(), ts, 0.9 * g, pan=-0.3)
        # shaker
        if style in ('drop', 'final', 'groove', 'lite', 'sneak'):
            A(perc, shaker(1.0 if s % 2 else 0.55), ts, 0.9 if full else 0.6, pan=0.45)
        # bass
        if style in ('drop', 'final', 'groove', 'hero', 'lite'):
            pat = {0: (root, 0.35), 3: (root, 0.12), 6: (root + 7, 0.2), 8: (root, 0.3), 11: (root, 0.12), 14: (root + 1, 0.12)}
            if s in pat:
                nn, d = pat[s]
                A(bassb, bass(nn, d, stacc=d < 0.2), ts, 1.0 if full else 0.85)
        elif style == 'sneak' and s in (0, 4, 8, 12):
            A(bassb, bass(root + (0, 7, 12, 7)[b], 0.14, stacc=True), ts, 0.8)
        # oud
        if s in riff:
            if style in ('drop', 'final'):
                A(oud, ks(riff[s], 0.45, bright=0.7), ts, 0.75, pan=-0.2)
            elif style in ('intro',):
                A(oud, ks(riff[s], 0.4, bright=0.35, damp=0.99), ts, 0.45, pan=-0.2)
            elif style in ('groove', 'lite') and s in (0, 4, 8, 12):
                A(oud, ks(riff[s], 0.35, bright=0.4, damp=0.99), ts, 0.4, pan=0.2)
        if style == 'sneak':
            for st, nn in SNEAK[bar % 4]:
                if st == s:
                    A(oud, ks(nn + 12, 0.18, damp=0.97, bright=0.5), ts, 0.55, pan=(-0.3 if s % 8 else 0.3))
    # reed lead on the big sections
    if style in ('drop', 'final'):
        prev = None
        for st, nn, ln in REED[bar % 4]:
            if t_end is None or t0 + st * STEP < t_end:
                A(reedb, reed(nn, ln * STEP * 0.98, prev), t0 + st * STEP, 0.9)
            prev = nn
    if style == 'hero':
        A(reedb, reed(PROG[bar % 4][2][0] + 12, 1.2), t0, 0.8)
        A(padb, pad(chord + (chord[0] + 12,), 1.6, 2400), t0, 0.6)


def run(t_start, t_end, origin, style_fn):
    b0 = int(np.floor((t_start - origin) / BAR + 1e-6))
    b = b0
    while origin + b * BAR < t_end - 1e-6:
        t0 = origin + b * BAR
        st = style_fn(max(t0, t_start))
        if st:
            bar_groove(t0, b, st, t_end=t_end)
        b += 1


def style_a(t):
    if t < 2.25:
        return 'intro'
    if t < ORIGIN:
        return 'build'
    if t < 7.75:
        return 'drop'
    if t < 12.25:
        return 'groove'
    if t < 14.25:
        return 'hero'
    if t < 20.25:
        return 'groove'
    if t < 22.25:
        return 'hero'
    return 'lite'


# section A: 0 -> 24.25 (grid on the supercut drop)
for t0 in np.arange(ORIGIN - 2 * BAR, SKIP0 - 1e-6, BAR):
    st = style_a(max(t0, 0))
    if st == 'build':
        for s in range(16):
            ts = t0 + s * STEP
            if s % 4 == 0:
                do_kick(ts, 0.7)
            A(perc, shaker(0.8), ts, 0.4 + 0.03 * s, pan=0.4)
            if s in (0, 8):
                A(perc, DOUM, ts, 0.7)
        continue
    bar_groove(t0, int(round((t0 - ORIGIN) / BAR)), st, t_end=SKIP0 if t0 + BAR > SKIP0 else None)


def darbuka_roll(t0, dur, g0=0.2, g1=0.95):
    t = t0
    while t < t0 + dur - 1e-6:
        p = (t - t0) / dur
        A(perc, tek(0.6 + 0.6 * p), t, g0 + (g1 - g0) * p, pan=0.2 * np.sin(p * 30))
        t += BEAT / (2 if p < 0.35 else (4 if p < 0.7 else 8))


darbuka_roll(3.0, 1.2)
A(mfx, riser(2.0, 250, 9000), 2.25, 0.5)

# section C: comeback scene, comic 'sneak' groove, build when he gets up, cut on the freeze
run(COME, FREEZE0, COME, lambda t: 'sneak')
darbuka_roll(UP, FREEZE0 - UP, 0.15, 0.9)
A(mfx, riser(FREEZE0 - UP + 0.05, 200, 9000), UP, 0.55)

# section F: end card + CTA
run(END0, CTA, END0, lambda t: 'final')
darbuka_roll(STEP_B - 0.5, 0.5, 0.3, 0.8)
# final hit
do_kick(CTA, 1.0)
A(drums, C, CTA, 0.8)
A(bassb, bass(38, 1.8), CTA, 1.0)
A(padb, pad((62, 66, 69, 74), 2.0, 2600), CTA, 0.7)
A(reedb, reed(74, 1.3), CTA, 0.9)
for k, nn in enumerate((62, 66, 69, 74, 78)):
    A(oud, ks(nn, 1.2, damp=0.996), CTA + k * 0.06, 0.5, pan=-0.3 + 0.15 * k)

# ---------------------------------------------------------------- sound design (not ducked)
for t in SUPERCUT_CUTS:
    A(fx, swish(0.18), t - 0.18, 0.6, pan=rng.uniform(-0.5, 0.5))
    A(fx, pop(72), t, 0.3)
for t in BROLL_CUTS + JUMPS:
    A(fx, swish(0.14), t - 0.14, 0.3, pan=rng.uniform(-0.5, 0.5))
for k, t in enumerate(CHIP_POPS):
    A(fx, pop(76 + 2 * (k % 3)), t, 0.55)
    A(fx, bell(86 + 3 * (k % 3), 0.6, 0.6), t + 0.01, 0.35, pan=(-0.3, 0.3)[k % 2])
for t in CARD_IN:
    A(fx, whoosh(0.45), t - 0.3, 0.7)
for t in SLAMS:
    A(fx, impact(), t, 1.0)
    A(fx, CR, t, 0.8)
    A(fx, sub_drop(), t, 0.85)
    for k, nn in enumerate((86, 90, 93)):
        A(fx, bell(nn, 1.0), t + 0.05 + 0.1 * k, (0.75, 0.65, 0.45)[k], pan=(-0.15, 0.2, 0.0)[k])
for t in (ORIGIN, END0):
    A(fx, whoosh(0.6), t - 0.33, 0.95)
    A(fx, impact(), t, 0.8)
    A(fx, CR, t, 0.7)
    A(fx, sub_drop(), t, 0.6)
# time skip
A(fx, whoosh(0.5), SKIP0 - 0.3, 0.8)
A(fx, fast_forward(SKIP1 - SKIP0), SKIP0, 0.9)
tt = SKIP0 + 0.05
while tt < SKIP1 - 0.05:
    A(fx, tick(), tt, 0.7, pan=rng.uniform(-0.3, 0.3))
    tt += 0.11 - 0.05 * (tt - SKIP0) / (SKIP1 - SKIP0)
A(fx, pop(84), SKIP0 + 0.12, 0.5)
A(fx, whoosh(0.4), SKIP1 - 0.25, 0.6)
# freeze frame: record scratch, comic sting
A(fx, record_scratch(), FREEZE0 - 0.02, 1.0)
A(fx, impact(), FREEZE0, 0.7)
A(fx, brass((38, 45, 50), 0.35, 1800), FREEZE0 + 0.12, 0.9)
A(fx, brass((37, 44, 49), 0.6, 1600), FREEZE0 + 0.45, 0.9)
A(fx, boing(), FREEZE0 + 0.18, 0.7)
A(fx, pop(80), FREEZE0 + 0.15, 0.6)
for t in (FREEZE1 + 0.05, FREEZE1 + 0.42, FREEZE1 + 0.85):
    A(fx, glitch(0.1), t, 0.8, pan=rng.uniform(-0.6, 0.6))
for k, t in enumerate(END_POPS):
    A(fx, swish(0.15), t - 0.15, 0.35, pan=(-0.4, 0.4)[k % 2])
    A(fx, pop(74 + 2 * (k % 5)), t, 0.5)
A(fx, bell(86, 1.0), 43.05, 0.55)
A(fx, bell(90, 1.0), 44.75, 0.55)
A(fx, impact(), CTA, 1.0)
A(fx, CR, CTA, 0.9)
A(fx, sub_drop(), CTA, 0.8)
A(fx, bell(93, 1.6), CTA + 0.2, 0.6)

# ---------------------------------------------------------------- mixdown
pump = np.ones(N)
for t in pump_pts:
    i, n = int(t * SR), int(0.2 * SR)
    if i + n < N:
        pump[i:i + n] = np.minimum(pump[i:i + n], 0.25 + 0.75 * np.linspace(0, 1, n) ** 1.4)
bassb *= pump[:, None]
padb *= (0.5 + 0.5 * pump)[:, None]
oud *= (0.75 + 0.25 * pump)[:, None]
music = (np.tanh(hp(drums, 40) * 1.2) * 0.75 + hp(perc, 60) * 1.1 + hp(bassb, 35, 4) * 0.7 + reverb(oud, 1.2, 0.25, 2) * 1.5
         + reverb(reedb, 1.0, 0.22, 6) * 1.1 + reverb(padb, 1.8, 0.35, 7) * 0.8 + mfx * 0.7)

# tape-stop into the time skip, silence during it, hard stop on the freeze, silence until the end card
ts0 = int((SKIP0 - 0.45) * SR)
seg = music[ts0:int(SKIP0 * SR)].copy()
n = len(seg)
rate = np.linspace(1.0, 0.0, n) ** 1.3
pos = np.clip(np.cumsum(rate), 0, n - 1)
for c in range(2):
    music[ts0:ts0 + n, c] = np.interp(pos, np.arange(n), seg[:, c]) * np.linspace(1, 0.2, n)
gate = np.ones(N)


def silence(a, b, ramp=0.01):
    i, j, r = int(a * SR), int(b * SR), int(ramp * SR)
    gate[i:j] = 0
    gate[max(0, i - r):i] = np.minimum(gate[max(0, i - r):i], np.linspace(1, 0, i - max(0, i - r)))
    gate[j:j + r] = np.minimum(gate[j:j + r], np.linspace(0, 1, r))


silence(SKIP0, SKIP1)
silence(FREEZE0, END0)
for t in SLAMS + [ORIGIN, END0]:
    silence(t - 0.125, t)
music *= gate[:, None]


def master(x, fade=True):
    x = hp(x, 35, 4)
    x /= np.percentile(np.abs(x), 99.9) + 1e-9
    x = np.tanh(x * 1.4) / np.tanh(1.4)
    t = np.arange(len(x)) / SR
    if fade:
        x *= (np.clip((TOTAL - t) / 1.4, 0, 1) ** 1.3)[:, None]
    x[:int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))[:, None]
    x = x[:int(TOTAL * SR)]
    return x * 0.89 / np.abs(x).max()


mus = master(music)
fxm = reverb(hp(fx, 50), 1.3, 0.22, 4)[:int(TOTAL * SR)]
fxm *= 0.89 / (np.abs(fxm).max() + 1e-9)
wavfile.write(sys.argv[1], SR, (mus * 32767).astype(np.int16))
wavfile.write(sys.argv[2], SR, (fxm * 32767).astype(np.int16))
print("music v2 + fx written", TOTAL, "s")
