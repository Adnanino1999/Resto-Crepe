"""Mix: voice on top, music ducks under it (side-chain on the voice envelope), sound design on its own level.
Usage: mix45.py VOICE MUSIC FX OUT
"""
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

vp, mp, fp, op = sys.argv[1:5]
sr, v = wavfile.read(vp)
_, m = wavfile.read(mp)
_, f = wavfile.read(fp)
v = v.astype(np.float64) / 32768
m = m.astype(np.float64) / 32768
f = f.astype(np.float64) / 32768
n = min(len(v), len(m), len(f))
v, m, f = v[:n], m[:n], f[:n]
env = np.convolve(np.abs(v), np.ones(int(0.03 * sr)) / int(0.03 * sr), mode="same")
active = (env > 0.015).astype(np.float64)
la = int(0.06 * sr)
active = np.concatenate([active[la:], np.zeros(la)])
att, rel = np.exp(-1 / (0.04 * sr)), np.exp(-1 / (0.45 * sr))
g = signal.lfilter([1 - rel], [1, -rel], active)          # slow release
g = np.maximum(g, signal.lfilter([1 - att], [1, -att], active))
g = np.clip(g, 0, 1)
FULL, DUCK = 0.62, 0.075
tt = np.arange(n) / sr
duck = np.where((tt > 25.4) & (tt < 39.2), 0.20, DUCK)      # comic groove stays audible in the comeback scene
mg = FULL + (duck - FULL) * g
sos = signal.butter(2, [900, 4500], 'bandstop', fs=sr, output='sos')
mc = signal.sosfilt(sos, m, axis=0)
mm = m * (1 - g)[:, None] + (0.55 * m + 0.45 * mc) * g[:, None]
mix = mm * mg[:, None] + v[:, None] * 1.0 + f * (0.55 - 0.3 * g)[:, None]
mix = np.tanh(mix * 1.1) / np.tanh(1.1)
mix *= 0.9 / np.abs(mix).max()
wavfile.write(op, sr, (mix * 32767).astype(np.int16))
print("mix written; voice-active share", round(float((g > 0.5).mean()), 2))
