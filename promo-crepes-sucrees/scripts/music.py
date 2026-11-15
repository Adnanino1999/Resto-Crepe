"""Original royalty-free music bed for the Resto Crêpes ad (fully synthesized).
120 BPM, 4/4, key of A minor / C major.  Output: stereo 48 kHz WAV.
"""
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
BPM = 120.0
BEAT = 60.0 / BPM          # 0.5 s
BAR = 4 * BEAT             # 2.0 s
TOTAL = 29.5
N = int(SR * (TOTAL + 1.5))
rng = np.random.default_rng(7)


def buf():
    return np.zeros((N, 2), dtype=np.float64)


def add(dst, snd, t, gain=1.0, pan=0.0):
    """Mix mono or stereo snd into dst at time t (s). pan in [-1, 1]."""
    i = int(round(t * SR))
    if i >= N or i < 0:
        return
    n = min(len(snd), N - i)
    if snd.ndim == 1:
        l = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
        r = np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
        dst[i:i + n, 0] += snd[:n] * gain * l
        dst[i:i + n, 1] += snd[:n] * gain * r
    else:
        dst[i:i + n] += snd[:n] * gain


def env_exp(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def lp(x, fc, order=2):
    sos = signal.butter(order, min(fc, SR * 0.45), 'low', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0)


def hp(x, fc, order=2):
    sos = signal.butter(order, fc, 'high', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0)


def bp(x, f1, f2, order=2):
    sos = signal.butter(order, [f1, f2], 'band', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)


# ---------------------------------------------------------------- drums
def kick():
    n = int(0.42 * SR)
    t = np.arange(n) / SR
    f = 42 + 130 * np.exp(-t / 0.028)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.16)
    click = hp(rng.standard_normal(n), 2500) * np.exp(-t / 0.004) * 0.35
    return np.tanh((body + click) * 1.6) * 0.9


def clap():
    n = int(0.35 * SR)
    out = np.zeros(n)
    noise = bp(rng.standard_normal(n), 900, 5200)
    for d, g in ((0.0, 0.7), (0.011, 0.8), (0.023, 0.9), (0.034, 1.0)):
        i = int(d * SR)
        e = np.zeros(n)
        tail = 0.012 if d < 0.03 else 0.10
        e[i:] = env_exp(n - i, tail)
        out += noise * e * g
    return out * 0.5


def hat(open_=False):
    n = int((0.22 if open_ else 0.06) * SR)
    x = hp(rng.standard_normal(n), 7500, 4)
    return x * env_exp(n, 0.07 if open_ else 0.014) * 0.35


def shaker():
    n = int(0.05 * SR)
    x = bp(rng.standard_normal(n), 4500, 9500)
    e = np.sin(np.linspace(0, np.pi, n)) ** 2
    return x * e * 0.2


def snare_roll_hit():
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.03)
    noise = bp(rng.standard_normal(n), 1200, 7000) * np.exp(-t / 0.045)
    return (tone * 0.5 + noise * 0.6) * 0.6


# ---------------------------------------------------------------- tonal
def saw(freq, n, detune=0.0, phase=0.0):
    t = np.arange(n) / SR
    f = freq * (2 ** (detune / 1200.0))
    return 2 * ((t * f + phase) % 1.0) - 1


def supersaw_note(note, dur, cutoff=2600, pluck=0.18, sustain=0.25):
    n = int(dur * SR)
    f = midi(note)
    l = np.zeros(n)
    r = np.zeros(n)
    for k, d in enumerate((-17, -9, -3, 3, 9, 17)):
        s = saw(f, n, d, rng.random())
        if k % 2 == 0:
            l += s
            r += s * 0.45
        else:
            r += s
            l += s * 0.45
    x = np.stack([l, r], axis=1) / 4.5
    e = sustain + (1 - sustain) * env_exp(n, pluck)
    rel = int(0.03 * SR)
    e[-rel:] *= np.linspace(1, 0, rel)
    att = int(0.004 * SR)
    e[:att] *= np.linspace(0, 1, att)
    return lp(x, cutoff, 2) * e[:, None]


def bass_note(note, dur):
    n = int(dur * SR)
    f = midi(note)
    t = np.arange(n) / SR
    x = 0.7 * np.sin(2 * np.pi * f * t) + 0.45 * saw(f, n) + 0.2 * np.sign(np.sin(2 * np.pi * f * t))
    x = lp(x, 420, 2)
    e = np.ones(n)
    att = int(0.006 * SR)
    e[:att] = np.linspace(0, 1, att)
    rel = int(0.04 * SR)
    e[-rel:] *= np.linspace(1, 0, rel)
    e *= 0.55 + 0.45 * env_exp(n, 0.12)
    return np.tanh(x * e * 1.4) * 0.6


def pluck_lead(note, dur=0.32):
    n = int(dur * SR)
    f = midi(note)
    t = np.arange(n) / SR
    x = (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.05)
         + 0.25 * np.sin(2 * np.pi * 3 * f * t) * np.exp(-t / 0.03))
    return x * np.exp(-t / 0.11) * 0.4


def bell(note, dur=1.6, bright=1.0):
    n = int(dur * SR)
    f = midi(note)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for mult, g, tau in ((1, 1.0, 0.55), (2.0, 0.55, 0.35), (3.01, 0.35 * bright, 0.22),
                         (4.2, 0.22 * bright, 0.14), (5.43, 0.15 * bright, 0.09)):
        x += g * np.sin(2 * np.pi * f * mult * t) * np.exp(-t / tau)
    return x * 0.3


# ---------------------------------------------------------------- fx
def riser(dur):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    out = np.zeros(n)
    # sweep a band-pass upward in short blocks
    blk = 1024
    zi = None
    for i in range(0, n, blk):
        p = i / n
        fc = 300 * (40 ** p)
        sos = signal.butter(2, [fc * 0.7, min(fc * 1.5, SR * 0.45)], 'band', fs=SR, output='sos')
        if zi is None:
            zi = signal.sosfilt_zi(sos) * 0
        seg, zi = signal.sosfilt(sos, x[i:i + blk], zi=zi)
        out[i:i + blk] = seg
    e = np.linspace(0, 1, n) ** 2.2
    return out * e * 0.5


def whoosh(dur=0.45):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    out = np.zeros(n)
    blk = 512
    zi = None
    for i in range(0, n, blk):
        p = i / n
        fc = 500 + 5500 * np.sin(np.pi * p) ** 1.5
        sos = signal.butter(2, [fc * 0.6, min(fc * 1.6, SR * 0.45)], 'band', fs=SR, output='sos')
        if zi is None:
            zi = signal.sosfilt_zi(sos) * 0
        seg, zi = signal.sosfilt(sos, x[i:i + blk], zi=zi)
        out[i:i + blk] = seg
    e = np.sin(np.linspace(0, np.pi, n)) ** 2
    return out * e * 0.5


def impact():
    n = int(1.6 * SR)
    t = np.arange(n) / SR
    f = 34 + 70 * np.exp(-t / 0.06)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.45)
    crash = hp(rng.standard_normal(n), 3000) * np.exp(-t / 0.5) * 0.28
    mid = bp(rng.standard_normal(n), 200, 1500) * np.exp(-t / 0.09) * 0.4
    return np.tanh(boom * 1.4) * 0.9 + crash + mid


def reverb(x, decay=1.4, wet=0.22, seed=3):
    r = np.random.default_rng(seed)
    n = int(decay * SR)
    t = np.arange(n) / SR
    ir = np.stack([r.standard_normal(n), r.standard_normal(n)], axis=1) * np.exp(-t / (decay / 5.5))[:, None]
    ir = hp(lp(ir, 6500), 250)
    ir /= np.sqrt((ir ** 2).sum(axis=0)).max()
    out = np.zeros_like(x)
    for c in range(2):
        out[:, c] = signal.fftconvolve(x[:, c], ir[:, c])[:len(x)]
    return x + wet * out


# ---------------------------------------------------------------- arrangement
# Bars (2 s each): 0 hook | 1-5 Kounafa | 6-10 4 You | 11-12 menu | 13-14 end card
CHORDS = {  # bar -> (bass root midi, chord notes)
    'Am': (45, (57, 60, 64, 67)),   # A C E G  (Am7)
    'F':  (41, (53, 57, 60, 64)),   # F A C E  (Fmaj7)
    'C':  (48, (55, 60, 64, 67)),   # G C E G
    'G':  (43, (55, 59, 62, 67)),   # G B D G
}
PROG = ['Am', 'F', 'C', 'G']
bars = 15


def chord_for(bar):
    if bar == 0:
        return 'Am'
    return PROG[(bar - 1) % 4]


drums = buf()
bassb = buf()
chords = buf()
lead = buf()
fx = buf()

K, C, H, OH, SH, SN = kick(), clap(), hat(), hat(True), shaker(), snare_roll_hit()

FULL = set(range(1, 11)) | {13, 14}
BREAK = {11, 12}

for bar in range(bars):
    t0 = bar * BAR
    name = chord_for(bar)
    root, notes = CHORDS[name]

    if bar == 0:
        # hook: filtered chord swell + snare build into bar 1
        for k, tt in enumerate((0.0, 0.75, 1.5)):
            for nn in notes:
                add(chords, supersaw_note(nn, 0.6, cutoff=900 + 500 * k, pluck=0.22), t0 + tt, 0.16)
        add(drums, K, t0, 0.95)
        add(fx, impact(), t0, 0.55)
        for k in range(8):       # 8th then 16th roll in the last beat
            add(drums, SN, t0 + 1.0 + k * 0.125, 0.25 + 0.07 * k)
        add(fx, riser(1.9), t0 + 0.1, 0.55)
        continue

    full = bar in FULL
    # --- drums
    for b in range(4):
        tb = t0 + b * BEAT
        if full or (bar in BREAK and b in (0, 2)):
            add(drums, K, tb, 1.0 if full else 0.75)
        if b in (1, 3):
            add(drums, C, tb, 0.8 if full else 0.5, pan=0.05)
        if full:
            add(drums, OH, tb + BEAT / 2, 0.55, pan=-0.2)
            for s in range(4):
                add(drums, H, tb + s * BEAT / 4, 0.32 if s % 2 else 0.2, pan=0.25)
        else:
            for s in range(2):
                add(drums, SH, tb + s * BEAT / 2, 0.8, pan=0.3)
    # fills at the end of each 4-bar phrase / before section changes
    if bar in (4, 9, 12):
        for k in range(4):
            add(drums, SN, t0 + 1.5 + k * 0.125, 0.3 + 0.1 * k)

    # --- bass (offbeat house pattern with octave jump)
    if full:
        pat = [(0.5, 0.22, 0), (1.0, 0.22, 0), (1.5, 0.22, 12), (2.5, 0.22, 0),
               (3.0, 0.22, 0), (3.5, 0.22, 7)]
        for (bt, dur, off) in pat:
            add(bassb, bass_note(root + off - 12 if root > 44 else root + off, dur), t0 + bt * BEAT, 0.9)
        add(bassb, bass_note(root - 12 if root > 44 else root, 0.3), t0, 0.5)
    else:
        add(bassb, bass_note(root - 12 if root > 44 else root, 1.6), t0, 0.55)

    # --- chord stabs (syncopated)
    if full:
        stabs = [(0.0, 0.3), (0.75, 0.22), (1.5, 0.3), (2.5, 0.22), (3.25, 0.3)]
        cut = 2400 if bar < 6 else 3200
        for (bt, dur) in stabs:
            for nn in notes:
                add(chords, supersaw_note(nn, dur, cutoff=cut), t0 + bt * BEAT, 0.15)
    else:
        for nn in notes:
            add(chords, supersaw_note(nn, 1.95, cutoff=1500, pluck=0.6, sustain=0.6), t0, 0.11)

    # --- lead motif (pentatonic, changes per section)
    PENT = [69, 72, 74, 76, 79, 81, 84]
    if 1 <= bar <= 5:
        motif = [(0, 3), (0.75, 4), (1.5, 3), (2.0, 2), (2.75, 1), (3.5, 2)]
    elif 6 <= bar <= 10:
        motif = [(0, 4), (0.5, 5), (1.0, 4), (1.75, 3), (2.5, 4), (3.0, 5), (3.5, 6)]
    elif bar in BREAK:
        motif = [(0, 1), (1.0, 3), (2.0, 2), (3.0, 4)]
    else:
        motif = [(0, 4), (0.75, 5), (1.5, 6), (2.0, 5), (2.75, 4), (3.5, 3)]
    shift = {'Am': 0, 'F': 0, 'C': 0, 'G': -1}[name]
    for (bt, idx) in motif:
        nn = PENT[max(0, min(len(PENT) - 1, idx + shift))]
        s = pluck_lead(nn)
        add(lead, s, t0 + bt * BEAT, 0.5, pan=-0.25)
        add(lead, s, t0 + bt * BEAT + 0.375, 0.2, pan=0.45)     # dotted-8th echo
        add(lead, s, t0 + bt * BEAT + 0.75, 0.09, pan=-0.45)

# --- transition FX (times match the video edit)
IMP = impact()
add(fx, IMP, 2.0, 0.8)                    # hook -> Kounafa
add(fx, riser(1.4), 8.55, 0.5)            # build to Kounafa price
add(fx, IMP, 10.0, 0.9)                   # Kounafa price reveal
add(fx, bell(88), 10.45, 0.8, pan=-0.1)
add(fx, bell(93), 10.57, 0.7, pan=0.15)
add(fx, whoosh(), 11.72, 0.8)             # -> 4 You
add(fx, IMP, 12.0, 0.7)
add(fx, riser(1.4), 18.55, 0.5)           # build to 4 You price
add(fx, IMP, 20.0, 0.9)
add(fx, bell(88), 20.45, 0.8, pan=-0.1)
add(fx, bell(93), 20.57, 0.7, pan=0.15)
add(fx, whoosh(), 21.72, 0.8)             # -> menu
add(fx, IMP, 22.0, 0.55)
for k, tt in enumerate((22.5, 23.3, 24.1)):   # menu tiers popping in
    add(fx, bell(84 + 2 * k, 0.8, 0.6), tt, 0.45, pan=(-0.3, 0.0, 0.3)[k])
add(fx, riser(1.2), 24.75, 0.55)
add(fx, IMP, 26.0, 1.0)                   # end card
add(fx, bell(81), 26.0, 0.6)
add(fx, bell(88), 26.25, 0.6, pan=0.2)
for tt in (2.5, 3.5, 4.5, 5.5, 6.5, 14.5, 16.0, 17.0):   # light whooshes on cuts
    add(fx, whoosh(0.3), tt - 0.15, 0.3)

# --- sidechain pump on bass / chords
pump = np.ones(N)
for bar in range(1, bars):
    if bar in FULL:
        for b in range(4):
            i = int((bar * BAR + b * BEAT) * SR)
            n = int(0.24 * SR)
            if i + n < N:
                pump[i:i + n] = np.minimum(pump[i:i + n], 0.25 + 0.75 * np.linspace(0, 1, n) ** 1.5)
bassb *= pump[:, None]
chords *= (0.45 + 0.55 * pump)[:, None]

# final chord: let the last bar ring instead of looping
end_t = 28.0
i_end = int(end_t * SR)
for b in (drums, bassb, lead):
    b[i_end:] *= np.exp(-np.arange(N - i_end) / (0.05 * SR))[:, None]
chords[i_end:] *= 0
for nn in (45, 57, 60, 64, 69, 72):
    add(chords, supersaw_note(nn, 2.6, cutoff=2600, pluck=0.9, sustain=0.5), end_t, 0.15)
add(drums, K, end_t, 1.0)
add(fx, IMP, end_t, 0.9)
add(fx, bell(93, 2.0), end_t, 0.6)

mix = (drums * 0.95 + hp(bassb, 30) * 0.9 + reverb(chords, 1.2, 0.25, 1) * 0.9
       + reverb(lead, 1.6, 0.35, 2) * 0.85 + reverb(fx, 1.8, 0.3, 4) * 0.8)
mix = hp(mix, 28)
# master: gentle glue + soft clip + fades
mix /= np.percentile(np.abs(mix), 99.9) + 1e-9
mix = np.tanh(mix * 1.5) / np.tanh(1.5)
t = np.arange(N) / SR
fade = np.clip((TOTAL + 0.0 - t) / 1.4, 0, 1) ** 1.5
mix *= fade[:, None]
mix[:int(0.005 * SR)] *= np.linspace(0, 1, int(0.005 * SR))[:, None]
mix = mix[:int(TOTAL * SR)]
mix *= 0.89 / np.abs(mix).max()
wavfile.write(sys.argv[1], SR, (mix * 32767).astype(np.int16))
print("wrote", sys.argv[1], len(mix) / SR, "s")
