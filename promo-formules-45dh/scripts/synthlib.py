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
TOTAL = 30.25
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

