"""Original royalty-free music bed v4 (30 s, second voice-over cut) for the Resto Crêpes ad – energetic, cut-synced.
120 BPM (cuts land on beats), A minor.  Fully synthesized.  Output: stereo 48 kHz WAV.
"""
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
BEAT = 0.5
STEP = BEAT / 4            # 16th note = 0.125 s
BAR = 2.0
TOTAL = 30.0
N = int(SR * (TOTAL + 2.0))
rng = np.random.default_rng(11)

# video edit points
CUTS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.5, 9.5, 12.5, 13.0, 13.5, 14.5, 15.25, 16.0, 17.0, 18.5]
WHIPS = [2.0, 12.0, 22.0]
REVEALS = [10.0, 20.0]          # price cards (flash); price pops 0.45 s later
END_CARD = 26.0
MENU_POPS = [22.48, 23.28, 24.08]
FINAL_HIT = 28.0


def buf():
    return np.zeros((N, 2))


def add(dst, snd, t, gain=1.0, pan=0.0):
    i = int(round(t * SR))
    if i >= N:
        return
    if i < 0:
        snd = snd[-i:]
        i = 0
    n = min(len(snd), N - i)
    if n <= 0:
        return
    if snd.ndim == 1:
        l = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
        r = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
        dst[i:i + n, 0] += snd[:n] * gain * l
        dst[i:i + n, 1] += snd[:n] * gain * r
    else:
        dst[i:i + n] += snd[:n] * gain


def ex(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def sos(kind, fc, order=2):
    return signal.butter(order, fc, kind, fs=SR, output='sos')


def lp(x, fc, order=2):
    return signal.sosfilt(sos('low', min(fc, SR * 0.45), order), x, axis=0)


def hp(x, fc, order=2):
    return signal.sosfilt(sos('high', fc, order), x, axis=0)


def bp(x, f1, f2, order=2):
    return signal.sosfilt(sos('band', [f1, min(f2, SR * 0.45)], order), x, axis=0)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)


def sweep_filter(x, f_of_p, q=(0.65, 1.6), blk=512):
    """Band-pass with centre frequency following f_of_p(progress 0..1)."""
    n = len(x)
    out = np.zeros(n)
    zi = None
    for i in range(0, n, blk):
        fc = f_of_p(i / n)
        s = signal.butter(2, [fc * q[0], min(fc * q[1], SR * 0.45)], 'band', fs=SR, output='sos')
        if zi is None:
            zi = signal.sosfilt_zi(s) * 0
        seg, zi = signal.sosfilt(s, x[i:i + blk], zi=zi)
        out[i:i + blk] = seg
    return out


# ---------------------------------------------------------------- drums
def kick():
    n = int(0.34 * SR)
    t = np.arange(n) / SR
    f = 47 + 190 * np.exp(-t / 0.018)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.13)
    click = hp(rng.standard_normal(n), 3000) * np.exp(-t / 0.003) * 0.5
    return np.tanh((body + click) * 2.4) * 0.95


def clap():
    n = int(0.3 * SR)
    noise = bp(rng.standard_normal(n), 1000, 6500)
    out = np.zeros(n)
    for d, g, tail in ((0.0, 0.7, 0.008), (0.010, 0.8, 0.008), (0.021, 0.9, 0.008), (0.031, 1.0, 0.085)):
        i = int(d * SR)
        e = np.zeros(n)
        e[i:] = ex(n - i, tail)
        out += noise * e * g
    t = np.arange(n) / SR
    snare = np.sin(2 * np.pi * 205 * t) * np.exp(-t / 0.035) * 0.5 + hp(rng.standard_normal(n), 2000) * np.exp(-t / 0.06) * 0.35
    return np.tanh((out * 0.7 + snare) * 1.3) * 0.6


def snare(pitch=200.0):
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * pitch * t) * np.exp(-t / 0.035)
    noise = bp(rng.standard_normal(n), 1500, 8500) * np.exp(-t / 0.05)
    return np.tanh((tone * 0.6 + noise * 0.8) * 1.5) * 0.55


def hat(open_=False):
    n = int((0.20 if open_ else 0.05) * SR)
    x = hp(rng.standard_normal(n), 8000, 4)
    return x * ex(n, 0.06 if open_ else 0.011) * 0.4


def cowbell(f1=560.0):
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    x = np.sign(np.sin(2 * np.pi * f1 * t)) + np.sign(np.sin(2 * np.pi * f1 * 1.48 * t))
    x = bp(x, f1 * 0.9, f1 * 3.2)
    return x * (0.25 * np.exp(-t / 0.012) + 0.75 * np.exp(-t / 0.05)) * 0.22


def crash():
    n = int(1.8 * SR)
    t = np.arange(n) / SR
    x = hp(rng.standard_normal(n), 4500, 2) * np.exp(-t / 0.5)
    x += bp(rng.standard_normal(n), 2500, 9000) * np.exp(-t / 0.2) * 0.6
    return x * 0.3


# ---------------------------------------------------------------- tonal
def osc_saw(f, n, ph=0.0):
    t = np.arange(n) / SR
    return 2 * ((t * f + ph) % 1.0) - 1


def osc_sq(f, n, ph=0.0, pw=0.5):
    t = np.arange(n) / SR
    return np.where(((t * f + ph) % 1.0) < pw, 1.0, -1.0)


def lead_note(note, dur=0.2, bright=1.0):
    n = int(dur * SR)
    f = midi(note)
    t = np.arange(n) / SR
    x = 0.55 * osc_saw(f, n, rng.random()) + 0.35 * osc_sq(f * 1.004, n, rng.random(), 0.35) + 0.3 * osc_saw(f * 2.0, n)
    # falling filter envelope = pluck
    out = np.zeros(n)
    blk = 256
    zi = None
    for i in range(0, n, blk):
        fc = 700 + (5200 * bright) * np.exp(-(i / SR) / 0.07)
        s = signal.butter(2, min(fc, SR * 0.45), 'low', fs=SR, output='sos')
        if zi is None:
            zi = signal.sosfilt_zi(s) * 0
        seg, zi = signal.sosfilt(s, x[i:i + blk], zi=zi)
        out[i:i + blk] = seg
    e = np.exp(-t / 0.16)
    e[:int(0.003 * SR)] *= np.linspace(0, 1, int(0.003 * SR))
    rel = int(0.02 * SR)
    e[-rel:] *= np.linspace(1, 0, rel)
    return out * e * 0.5


def brass(notes, dur=0.16, cutoff=3400):
    n = int(dur * SR)
    l = np.zeros(n)
    r = np.zeros(n)
    for nn in notes:
        f = midi(nn)
        for k, d in enumerate((-14, -5, 5, 14)):
            s = osc_saw(f * 2 ** (d / 1200), n, rng.random())
            if k % 2:
                l += s
                r += 0.4 * s
            else:
                r += s
                l += 0.4 * s
    x = np.stack([l, r], axis=1) / (2.2 * len(notes))
    t = np.arange(n) / SR
    e = np.minimum(1, t / 0.012) * (0.6 + 0.4 * np.exp(-t / 0.05))
    rel = int(0.03 * SR)
    e[-rel:] *= np.linspace(1, 0, rel)
    return np.tanh(lp(x, cutoff) * e[:, None] * 2.2) * 0.5


def bass_note(note, dur):
    n = int(dur * SR)
    f = midi(note)
    t = np.arange(n) / SR
    x = 0.8 * np.sin(2 * np.pi * f * t) + 0.6 * osc_saw(f, n) + 0.25 * osc_sq(f * 2, n)
    x = lp(x, 1600, 2) + 0.35 * bp(osc_saw(f * 2, n), 300, 1800)
    e = np.minimum(1, t / 0.004) * (0.6 + 0.4 * np.exp(-t / 0.06))
    rel = int(0.015 * SR)
    e[-rel:] *= np.linspace(1, 0, rel)
    return np.tanh(x * e * 2.0) * 0.55


def sub_drop():
    n = int(1.1 * SR)
    t = np.arange(n) / SR
    f = 30 + 85 * np.exp(-t / 0.22)
    return np.tanh(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.4) * 1.8) * 0.9


def bell(note, dur=1.2, bright=1.0):
    n = int(dur * SR)
    f = midi(note)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for mult, g, tau in ((1, 1.0, 0.45), (2.0, 0.6, 0.3), (3.01, 0.4 * bright, 0.18), (4.2, 0.25 * bright, 0.12),
                         (5.43, 0.18 * bright, 0.08), (6.8, 0.12 * bright, 0.05)):
        x += g * np.sin(2 * np.pi * f * mult * t) * np.exp(-t / tau)
    return x * 0.3


def pop(note=84):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    f = midi(note) * (1 + 0.6 * np.exp(-t / 0.012))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.03) * 0.6


# ---------------------------------------------------------------- fx
def riser(dur, f0=250, f1=9000):
    n = int(dur * SR)
    x = sweep_filter(rng.standard_normal(n), lambda p: f0 * (f1 / f0) ** p)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * np.cumsum(180 * (4.0 ** (t / dur))) / SR) * 0.25
    e = np.linspace(0, 1, n) ** 2.0
    return (x * 0.6 + tone) * e


def swish(dur=0.2):
    """Short rising whoosh that ends exactly at its last sample (place it to end on a cut)."""
    n = int(dur * SR)
    x = sweep_filter(rng.standard_normal(n), lambda p: 900 + 7000 * p ** 1.5)
    e = np.linspace(0, 1, n) ** 2.5
    e[-int(0.006 * SR):] *= np.linspace(1, 0, int(0.006 * SR))
    return x * e * 0.55


def whoosh(dur=0.55):
    n = int(dur * SR)
    x = sweep_filter(rng.standard_normal(n), lambda p: 400 + 6500 * np.sin(np.pi * p) ** 1.4, q=(0.5, 1.9))
    e = np.sin(np.linspace(0, np.pi, n)) ** 2
    l = x * e * np.linspace(1.0, 0.3, n)
    r = x * e * np.linspace(0.3, 1.0, n)
    return np.stack([l, r], axis=1) * 0.75


def impact():
    n = int(1.3 * SR)
    t = np.arange(n) / SR
    mid = bp(rng.standard_normal(n), 150, 2500) * np.exp(-t / 0.07) * 0.8
    hi = hp(rng.standard_normal(n), 4000) * np.exp(-t / 0.25) * 0.25
    f = 40 + 120 * np.exp(-t / 0.03)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.3)
    return np.tanh((boom * 1.2 + mid + hi) * 1.5) * 0.9


def reverse(x):
    return x[::-1].copy()


def reverb(x, decay=1.2, wet=0.2, seed=3):
    r = np.random.default_rng(seed)
    n = int(decay * SR)
    t = np.arange(n) / SR
    ir = np.stack([r.standard_normal(n), r.standard_normal(n)], axis=1) * np.exp(-t / (decay / 5.5))[:, None]
    ir = hp(lp(ir, 7000), 300)
    ir /= np.sqrt((ir ** 2).sum(axis=0)).max()
    out = np.zeros_like(x)
    for c in range(2):
        out[:, c] = signal.fftconvolve(x[:, c], ir[:, c])[:len(x)]
    return x + wet * out


# ---------------------------------------------------------------- harmony
CH = {  # name: (bass root, brass triad, lead riff notes)
    'Am': (33, (57, 60, 64), (69, 69, 72, 76, 74, 72, 69, 72, 76, 79)),
    'F':  (29, (57, 60, 65), (69, 69, 72, 77, 76, 72, 69, 72, 77, 81)),
    'C':  (36, (55, 60, 64), (67, 67, 72, 76, 74, 72, 67, 72, 76, 79)),
    'G':  (31, (55, 59, 62), (67, 67, 71, 74, 74, 71, 67, 71, 74, 79)),
}
RIFF_STEPS = (0, 2, 3, 5, 6, 8, 10, 11, 13, 14)
BASS_STEPS = (2, 3, 6, 7, 10, 11, 14, 15)
STAB_STEPS = (0, 3, 6, 10)
BELL_STEPS = (0, 3, 6, 8, 11, 14)          # 3-3-2 feel

# bar -> (section, chord)
PLAN = {
    0: ('intro', 'Am'),
    1: ('drop', 'Am'), 2: ('drop', 'F'), 3: ('drop', 'C'), 4: ('build', 'G'),
    5: ('hero', 'Am'),
    6: ('drop2', 'Am'), 7: ('drop2', 'F'), 8: ('drop2', 'C'), 9: ('build', 'G'),
    10: ('hero', 'Am'),
    11: ('menu', 'F'), 12: ('menubuild', 'G'),
    13: ('final', 'Am'),
    14: ('outro', 'Am'),
}

drums, bassb, leadb, brassb, fx = buf(), buf(), buf(), buf(), buf()
K, C, H, OH, CB, CR = kick(), clap(), hat(), hat(True), cowbell(), crash()
pump_pts = []   # kick times for side-chain


def do_kick(t, g=1.0):
    add(drums, K, t, g)
    pump_pts.append(t)


def snare_roll(t0, dur, start_div=2, accel=True, g0=0.25, g1=0.9):
    """Accelerating roll with rising pitch from t0 to t0+dur."""
    t = t0
    end = t0 + dur
    while t < end - 1e-6:
        p = (t - t0) / dur
        add(drums, snare(190 + 160 * p), t, g0 + (g1 - g0) * p, pan=0.1 * np.sin(p * 20))
        if not accel:
            t += BEAT / start_div
        else:
            div = start_div if p < 0.4 else (start_div * 2 if p < 0.75 else start_div * 4)
            t += BEAT / div


for bar, (sec, ch) in PLAN.items():
    t0 = bar * BAR
    root, triad, riff = CH[ch]

    if sec == 'intro':
        # filtered riff teaser, claps, riser and roll into the drop
        for k, st in enumerate(RIFF_STEPS):
            add(leadb, lead_note(riff[k], 0.2, bright=0.25 + 0.06 * k), t0 + st * STEP, 0.55, pan=-0.15)
        do_kick(t0, 0.9)
        add(fx, impact(), t0, 0.5)
        add(drums, C, t0 + 0.5, 0.5)
        add(drums, C, t0 + 1.0, 0.6)
        snare_roll(t0 + 1.0, 1.0, start_div=2)
        add(fx, riser(1.95), t0 + 0.05, 0.7)
        continue

    if sec in ('drop', 'drop2', 'final', 'menu', 'menubuild'):
        hard = sec in ('drop2', 'final')
        for b in range(4):
            tb = t0 + b * BEAT
            do_kick(tb, 1.0)
            if b in (1, 3):
                add(drums, C, tb, 0.95 if sec != 'menu' else 0.8, pan=0.04)
            add(drums, OH, tb + BEAT / 2, 0.6, pan=-0.25)
            for s in range(4):
                add(drums, H, tb + s * STEP, (0.45 if s == 2 else 0.28) * (1.15 if hard else 1.0), pan=0.3)
        if hard:      # hat roll on the last beat for extra drive
            for s in range(8):
                add(drums, H, t0 + 3 * BEAT + s * STEP / 2, 0.22 + 0.03 * s, pan=-0.3)
        for st in BELL_STEPS:
            add(drums, CB, t0 + st * STEP, 0.55 if hard else 0.4, pan=0.4)
        for i, st in enumerate(BASS_STEPS):
            add(bassb, bass_note(root + (12 if i % 2 else 0), STEP * 0.95), t0 + st * STEP, 1.0)
        if sec != 'menu':
            for k, st in enumerate(RIFF_STEPS):
                nn = riff[k] + (12 if (hard and k % 5 == 4) else 0)
                s = lead_note(nn, 0.2, bright=1.0)
                add(leadb, s, t0 + st * STEP, 0.6, pan=-0.2)
                add(leadb, s, t0 + st * STEP + 0.1875, 0.16, pan=0.5)
            for st in STAB_STEPS:
                add(brassb, brass(triad if not hard else triad + (triad[0] + 12,), 0.15), t0 + st * STEP, 0.6)
        else:
            # menu: lighter melody so the price "pops" read clearly, still driving
            for k, st in enumerate((0, 3, 6, 8, 11, 14)):
                s = lead_note(riff[k % len(riff)] + 12, 0.16, bright=0.7)
                add(leadb, s, t0 + st * STEP, 0.4, pan=0.2)
            add(brassb, brass(triad, 0.2), t0, 0.5)
        if sec == 'menubuild':
            snare_roll(t0 + 1.0, 1.0, start_div=2, g0=0.2, g1=0.85)
            add(fx, riser(1.5), t0 + 0.5, 0.65)
        if sec in ('drop', 'drop2') and bar in (3, 8):      # fill into the build
            for s in range(4):
                add(drums, snare(210 + 25 * s), t0 + 1.5 + s * STEP, 0.45 + 0.1 * s)

    elif sec == 'build':
        # tension: thinning kick, accelerating roll, riser, then a 1/8-beat silence before the reveal
        for b in range(3):
            do_kick(t0 + b * BEAT, 0.95)
        for b in range(4):
            add(drums, C, t0 + b * BEAT + 0.0, 0.35 + 0.12 * b)
        snare_roll(t0 + 0.5, 1.3, start_div=2, g0=0.25, g1=0.95)
        for i, st in enumerate(BASS_STEPS[:6]):
            add(bassb, bass_note(root + (12 if i % 2 else 0), STEP * 0.95), t0 + st * STEP, 0.9)
        for k, st in enumerate(RIFF_STEPS[:8]):
            add(leadb, lead_note(riff[k] + 12 * (k // 4), 0.18, bright=0.6 + 0.08 * k), t0 + st * STEP, 0.55, pan=-0.2)
        add(fx, riser(1.8, 300, 11000), t0, 0.85)
        add(fx, reverse(crash())[-int(0.9 * SR):], t0 + 2.0 - 0.9, 0.9)

    elif sec == 'hero':
        # half-time, huge: price reveal
        do_kick(t0, 1.0)
        do_kick(t0 + 0.75, 0.9)
        do_kick(t0 + 1.25, 0.9)
        add(drums, C, t0 + 1.0, 1.0)
        add(drums, C, t0 + 1.75, 0.6)
        for s in range(16):
            add(drums, H, t0 + s * STEP, 0.25 if s % 2 else 0.4, pan=0.3)
        for s in (12, 13, 14, 15):
            add(drums, snare(200 + 20 * s), t0 + s * STEP, 0.3 + 0.08 * (s - 12))
        add(bassb, bass_note(root, 0.7), t0, 1.0)
        add(bassb, bass_note(root, 0.4), t0 + 0.75, 0.9)
        add(bassb, bass_note(root + 12, 0.6), t0 + 1.25, 0.9)
        add(brassb, brass(triad + (triad[0] + 12,), 0.45, 4200), t0, 0.9)
        add(brassb, brass(triad + (triad[0] + 12,), 0.2, 4200), t0 + 0.75, 0.7)
        add(brassb, brass(tuple(n + 12 for n in triad), 0.5, 4500), t0 + 1.0, 0.8)
        for k, st in enumerate((8, 10, 11, 13, 14)):
            add(leadb, lead_note(riff[5 + k % 5] + 12, 0.2), t0 + st * STEP, 0.5, pan=-0.2)

    elif sec == 'outro':
        do_kick(t0, 1.0)
        add(drums, C, t0, 0.8)
        add(fx, CR, t0, 1.0)
        add(fx, impact(), t0, 1.0)
        add(fx, sub_drop(), t0, 0.8)
        add(bassb, bass_note(root, 1.3), t0, 1.0)
        add(brassb, brass(triad + (triad[0] + 12, triad[1] + 12), 1.4, 3600), t0, 1.0)
        for k, nn in enumerate((69, 72, 76, 81, 84)):
            add(leadb, lead_note(nn, 0.5), t0 + k * 0.0625, 0.5, pan=-0.3 + 0.15 * k)
        add(fx, bell(93, 1.4), t0, 0.7)

# ---------------------------------------------------------------- edit-synced FX
for t in CUTS:                                   # every cut: swish in + tick on the frame
    add(fx, swish(0.2), t - 0.2, 0.75, pan=rng.uniform(-0.5, 0.5))
    add(fx, pop(72), t, 0.35)
for t in WHIPS:                                  # zoom-blur transitions
    add(fx, whoosh(0.6), t - 0.33, 1.0)
    add(fx, impact(), t, 0.85)
    add(fx, CR, t, 0.7)
    add(fx, sub_drop(), t, 0.6)
for t in REVEALS:                                # price cards
    add(fx, impact(), t, 1.0)
    add(fx, CR, t, 0.9)
    add(fx, sub_drop(), t, 0.9)
    add(fx, swish(0.18), t + 0.45 - 0.18, 0.6)   # price sticker slams in
    add(fx, pop(60), t + 0.45, 0.8)
    add(fx, bell(88, 1.0), t + 0.45, 0.9, pan=-0.15)
    add(fx, bell(93, 1.0), t + 0.55, 0.8, pan=0.2)
    add(fx, bell(100, 0.8), t + 0.65, 0.5, pan=0.0)
    add(fx, whoosh(0.5), t + 2.0 - 0.3, 0.9)     # exit of the card
for k, t in enumerate(MENU_POPS):                # menu rows popping in
    add(fx, swish(0.15), t - 0.15, 0.5, pan=(-0.4, 0.4, -0.4)[k])
    add(fx, pop(76 + 3 * k), t, 0.7)
    add(fx, bell(84 + 4 * k, 0.7, 0.7), t + 0.02, 0.6, pan=(-0.3, 0.3, -0.3)[k])
add(fx, impact(), END_CARD, 1.0)
add(fx, CR, END_CARD, 1.0)
add(fx, sub_drop(), END_CARD, 0.8)
for k, tt in enumerate((0.42, 0.85, 0.99)):      # date slam + the two offer chips
    add(fx, pop(72 + 5 * k), END_CARD + tt, 0.6)
add(fx, bell(88, 1.0), END_CARD + 0.42, 0.7)

# ---------------------------------------------------------------- side-chain, silences, master
pump = np.ones(N)
for t in pump_pts:
    i = int(t * SR)
    n = int(0.22 * SR)
    if i + n < N:
        pump[i:i + n] = np.minimum(pump[i:i + n], 0.18 + 0.82 * np.linspace(0, 1, n) ** 1.4)
bassb *= pump[:, None]
brassb *= (0.5 + 0.5 * pump)[:, None]
leadb *= (0.7 + 0.3 * pump)[:, None]

music = (np.tanh(hp(drums, 45) * 1.25) * 0.80 + hp(bassb, 45, 4) * 0.62 + reverb(leadb, 1.0, 0.22, 2) * 1.75
         + reverb(brassb, 0.9, 0.18, 5) * 1.55)
# dramatic 1/8-note stop right before each price reveal (FX keep ringing through it)
gate = np.ones(N)
for t in REVEALS:
    a, b = int((t - 0.125) * SR), int(t * SR)
    ramp = int(0.01 * SR)
    gate[a:b] = 0.0
    gate[a - ramp:a] = np.linspace(1, 0, ramp)
music *= gate[:, None]
# ring out after the final hit
i_end = int((FINAL_HIT + 0.02) * SR)
mix = music + reverb(hp(fx, 55), 1.6, 0.28, 4) * 0.8
mix = hp(mix, 40, 4)
mix /= np.percentile(np.abs(mix), 99.9) + 1e-9
mix = np.tanh(mix * 1.7) / np.tanh(1.7)
t = np.arange(N) / SR
mix *= (np.clip((TOTAL - t) / 1.2, 0, 1) ** 1.3)[:, None]
mix[:int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))[:, None]
mix = mix[:int(TOTAL * SR)]
mix *= 0.89 / np.abs(mix).max()
wavfile.write(sys.argv[1], SR, (mix * 32767).astype(np.int16))
print("wrote", sys.argv[1], len(mix) / SR, "s")
