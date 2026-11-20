"""Original music bed + edit-synced sound design for the 'Formules 45 DH' ad (30.25 s, 120 BPM, A minor).
Fully synthesized (no third-party audio). Grid origin = 4.25 s (the drop into the supercut).
Usage: music45.py MUSIC_WAV FX_WAV
"""
import os
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthlib import *  # noqa: F401,F403  (SR, BEAT, STEP, instruments, CH, RIFF_STEPS, ...)
import synthlib as L

TOTAL = 30.25
N = int(SR * (TOTAL + 2.0))
L.N = N
ORIGIN = 4.25

# edit events (must match build45.py)
SUPERCUT_CUTS = [4.75, 5.25, 5.75, 6.25, 6.75, 7.25]
BROLL_CUTS = [8.55, 9.00, 9.45, 10.55, 16.81, 17.11, 17.41, 18.26]
CHIP_POPS = [8.62, 9.98, 11.13, 16.50, 17.80, 19.00]
CARD_IN = [12.35, 19.76]
SLAMS = [12.72, 20.25]
WHIPS = [4.25, 24.25]
END_POPS = [24.55, 24.85, 25.15, 25.60, 26.00]
FINAL = 28.25


def style(t):
    if t < 2.25:
        return 'intro'
    if t < ORIGIN:
        return 'build'
    if t < 7.75:
        return 'drop'
    if t < 12.75:
        return 'groove'
    if t < 14.75:
        return 'hero'
    if t < 20.25:
        return 'groove'
    if t < 21.75:
        return 'hero'
    if t < 24.25:
        return 'build2'
    if t < FINAL:
        return 'final'
    return 'outro'


PROG = ['Am', 'F', 'C', 'G']
drums, bassb, leadb, brassb, padb = (np.zeros((N, 2)) for _ in range(5))
music_fx, fx = np.zeros((N, 2)), np.zeros((N, 2))
K, C, Hh, OH, CB, CR = kick(), clap(), hat(), hat(True), cowbell(), crash()
pump_pts = []


def A(buf, snd, t, g=1.0, pan=0.0):
    L.add(buf, snd, t, g, pan)


def do_kick(t, g=1.0):
    A(drums, K, t, g)
    pump_pts.append(t)


def pad(notes, dur, cutoff=1800):
    n = int(dur * SR)
    t = np.arange(n) / SR
    l = np.zeros(n)
    r = np.zeros(n)
    for nn in notes:
        f = midi(nn)
        for k, d in enumerate((-9, -3, 4, 10)):
            s = osc_saw(f * 2 ** (d / 1200), n, rng.random())
            (l if k % 2 else r)[:] += s
            (r if k % 2 else l)[:] += 0.35 * s
    x = np.stack([l, r], axis=1) / (3.0 * len(notes))
    e = np.minimum(1, t / 0.25) * np.minimum(1, (dur - t) / 0.3).clip(0, 1)
    return lp(x, cutoff) * e[:, None] * 0.6


first_beat = int(np.floor((0.0 - ORIGIN) / BEAT))
last_beat = int(np.ceil((TOTAL - ORIGIN) / BEAT))
for b in range(first_beat, last_beat):
    tb = ORIGIN + b * BEAT
    if tb < -1e-6:
        continue
    st = style(tb)
    bar = b // 4
    bib = b % 4                      # beat in bar
    ch = PROG[bar % 4]
    root, triad, riff = CH[ch]
    if bib == 0 and st in ('groove', 'hero', 'final', 'intro', 'build', 'build2'):
        A(padb, pad(triad, BAR, 1400 if st in ('intro', 'groove') else 2200), tb, 0.55)
    if st == 'intro':
        # filtered riff teaser + heartbeat sub, very light (under the question)
        for k, s in enumerate(RIFF_STEPS):
            if bib * 4 <= s < bib * 4 + 4:
                A(leadb, lead_note(riff[k], 0.18, bright=0.18), tb + (s - bib * 4) * STEP, 0.35, pan=-0.15)
        if bib in (0, 2):
            A(drums, K, tb, 0.35)
        continue
    if st in ('build', 'build2'):
        do_kick(tb, 0.8)
        A(drums, C, tb + BEAT / 2, 0.18 + 0.1 * bib)
        for s in range(4):
            A(drums, Hh, tb + s * STEP, 0.2, pan=0.3)
        for i, s in enumerate((2, 3)):
            A(bassb, bass_note(root + 12 * i, STEP * 0.9), tb + s * STEP, 0.75)
        continue
    if st in ('drop', 'final', 'groove'):
        hard = st in ('drop', 'final')
        do_kick(tb, 1.0 if hard else 0.75)
        if bib in (1, 3):
            A(drums, C, tb, 0.9 if hard else 0.5, pan=0.04)
        A(drums, OH, tb + BEAT / 2, 0.55 if hard else 0.3, pan=-0.25)
        for s in range(4):
            A(drums, Hh, tb + s * STEP, (0.42 if s == 2 else 0.26) * (1.1 if hard else 0.8), pan=0.3)
        for i, s in enumerate((2, 3)):
            A(bassb, bass_note(root + 12 * i, STEP * 0.95), tb + s * STEP, 1.0 if hard else 0.8)
        for s in (0, 3):
            if hard or bib % 2 == 0:
                A(drums, CB, tb + s * STEP, 0.35 if hard else 0.18, pan=0.4)
        if hard:
            for k, s in enumerate(RIFF_STEPS):
                if bib * 4 <= s < bib * 4 + 4:
                    nn = riff[k] + (12 if (st == 'final' and k % 5 == 4) else 0)
                    snd = lead_note(nn, 0.2, bright=1.0)
                    A(leadb, snd, tb + (s - bib * 4) * STEP, 0.55, pan=-0.2)
                    A(leadb, snd, tb + (s - bib * 4) * STEP + 0.1875, 0.15, pan=0.5)
            if bib in (0, 2):
                A(brassb, brass(triad, 0.15), tb, 0.5)
                A(brassb, brass(triad, 0.15), tb + 3 * STEP, 0.45)
        else:
            # light pluck counter-melody, sits under the voice
            if bib in (0, 2):
                A(leadb, lead_note(riff[(bar * 2 + bib) % len(riff)] + 12, 0.14, bright=0.35), tb + 2 * STEP, 0.22, pan=0.25)
        continue
    if st == 'hero':
        # half-time, roomy: lets the price breathe
        if bib == 0:
            do_kick(tb, 1.0)
            A(bassb, bass_note(root, 0.9), tb, 0.95)
            A(brassb, brass(triad + (triad[0] + 12,), 0.5, 4200), tb, 0.75)
        if bib == 2:
            A(drums, C, tb, 0.8)
            do_kick(tb + 0.25, 0.7)
        for s in range(2):
            A(drums, Hh, tb + s * 2 * STEP, 0.3, pan=0.3)
        continue
    if st == 'outro':
        break

# ---- builds: rolls, risers, stops
def snare_roll(t0, dur, g0=0.2, g1=0.9):
    t = t0
    while t < t0 + dur - 1e-6:
        p = (t - t0) / dur
        A(drums, snare(190 + 170 * p), t, g0 + (g1 - g0) * p, pan=0.1 * np.sin(p * 20))
        t += BEAT / (2 if p < 0.4 else (4 if p < 0.75 else 8))


snare_roll(3.25, 0.95)
A(music_fx, riser(2.0, 250, 10000), 2.25, 0.55)
snare_roll(23.25, 0.95, 0.15, 0.8)
A(music_fx, riser(2.4, 250, 11000), 21.85, 0.5)

# final hit + ring out
do_kick(FINAL, 1.0)
A(drums, C, FINAL, 0.8)
A(bassb, bass_note(33, 1.6), FINAL, 1.0)
A(brassb, brass((57, 60, 64, 69, 72), 1.6, 3600), FINAL, 0.9)
for k, nn in enumerate((69, 72, 76, 81, 84)):
    A(leadb, lead_note(nn, 0.5), FINAL + k * 0.0625, 0.45, pan=-0.3 + 0.15 * k)
A(padb, pad((57, 60, 64, 69), 2.0, 2600), FINAL, 0.6)

# ---- sound design (separate stem: it is not ducked under the voice)
for t in SUPERCUT_CUTS:
    A(fx, swish(0.18), t - 0.18, 0.6, pan=rng.uniform(-0.5, 0.5))
    A(fx, pop(72), t, 0.3)
for t in BROLL_CUTS:
    A(fx, swish(0.14), t - 0.14, 0.28, pan=rng.uniform(-0.5, 0.5))
for k, t in enumerate(CHIP_POPS):
    A(fx, pop(76 + 2 * (k % 3)), t, 0.55)
    A(fx, bell(86 + 3 * (k % 3), 0.6, 0.6), t + 0.01, 0.35, pan=(-0.3, 0.3)[k % 2])
for t in CARD_IN:
    A(fx, whoosh(0.45), t - 0.3, 0.7)
for t in SLAMS:
    A(fx, impact(), t, 1.0)
    A(fx, CR, t, 0.8)
    A(fx, sub_drop(), t, 0.85)
    for k, nn in enumerate((88, 93, 100)):
        A(fx, bell(nn, 1.0), t + 0.05 + 0.1 * k, (0.75, 0.65, 0.45)[k], pan=(-0.15, 0.2, 0.0)[k])
for t in WHIPS:
    A(fx, whoosh(0.6), t - 0.33, 0.95)
    A(fx, impact(), t, 0.8)
    A(fx, CR, t, 0.7)
    A(fx, sub_drop(), t, 0.6)
for k, t in enumerate(END_POPS):
    A(fx, swish(0.15), t - 0.15, 0.35, pan=(-0.4, 0.4)[k % 2])
    A(fx, pop(74 + 3 * k), t, 0.5)
A(fx, bell(88, 1.0), 25.60, 0.6)
A(fx, impact(), FINAL, 1.0)
A(fx, CR, FINAL, 0.9)
A(fx, sub_drop(), FINAL, 0.8)
A(fx, bell(93, 1.6), FINAL + 0.2, 0.6)

# ---- side-chain, stops, master
pump = np.ones(N)
for t in pump_pts:
    i = int(t * SR)
    n = int(0.22 * SR)
    if i + n < N:
        pump[i:i + n] = np.minimum(pump[i:i + n], 0.2 + 0.8 * np.linspace(0, 1, n) ** 1.4)
bassb *= pump[:, None]
brassb *= (0.5 + 0.5 * pump)[:, None]
leadb *= (0.7 + 0.3 * pump)[:, None]
padb *= (0.45 + 0.55 * pump)[:, None]
music = (np.tanh(hp(drums, 45) * 1.2) * 0.8 + hp(bassb, 45, 4) * 0.62 + reverb(leadb, 1.0, 0.22, 2) * 1.6
         + reverb(brassb, 0.9, 0.18, 5) * 1.4 + reverb(padb, 1.6, 0.35, 7) * 0.9 + music_fx * 0.8)
gate = np.ones(N)
for t in SLAMS + [ORIGIN, 24.25]:            # 1/8-note stop before each big moment
    a, b = int((t - 0.125) * SR), int(t * SR)
    r = int(0.01 * SR)
    gate[a:b] = 0.0
    gate[a - r:a] = np.linspace(1, 0, r)
music *= gate[:, None]


def master(x, fade=True):
    x = hp(x, 40, 4)
    x /= np.percentile(np.abs(x), 99.9) + 1e-9
    x = np.tanh(x * 1.5) / np.tanh(1.5)
    t = np.arange(len(x)) / SR
    if fade:
        x *= (np.clip((TOTAL - t) / 1.4, 0, 1) ** 1.3)[:, None]
    x[:int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))[:, None]
    x = x[:int(TOTAL * SR)]
    return x * 0.89 / np.abs(x).max()


mus = master(music)
fxm = reverb(hp(fx, 55), 1.4, 0.25, 4)
fxm = fxm[:int(TOTAL * SR)] / (np.abs(fxm).max() + 1e-9) * 0.89
wavfile.write(sys.argv[1], SR, (mus * 32767).astype(np.int16))
wavfile.write(sys.argv[2], SR, (fxm * 32767).astype(np.int16))
print("music + fx written", TOTAL, "s")
