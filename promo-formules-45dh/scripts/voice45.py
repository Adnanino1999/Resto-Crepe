"""Dialogue track: the live Darija exchange from IMG_0143, cut and placed on the edit (lip-sync kept: no tempo change).
Usage: voice45.py SRC_MOV OUT_WAV
"""
import subprocess
import sys

import noisereduce as nr
import numpy as np
from scipy.io import wavfile

SR = 48000
TOTAL = 30.25
SRC, OUT = sys.argv[1], sys.argv[2]
# (src_start, src_end, T_start)
PIECES = [
    (0.60, 2.50, 0.00),     # "Chnou katakoul ?"
    (12.80, 15.15, 1.90),   # "Khouya, Resto Crêpes dayrin wa7ed promo harba !"
    (15.15, 20.95, 7.75),   # "Chnou fiha ?" ... pizza 4 fromages, pancake, boisson, ghir b 45 dh
    (20.95, 26.97, 14.96),  # tacos poulet, crêpe Nutella, frites, ghir b 45 dh
    (26.97, 29.08, 21.75),  # "Fin had chi ?" "Khouya, f Resto Crêpes"
]
CHAIN = ("highpass=f=85,lowpass=f=15000,"
         "equalizer=f=280:t=q:w=1.0:g=-3,"
         "equalizer=f=3200:t=q:w=1.0:g=3,"
         "highshelf=f=9000:g=2.5,"
         "deesser=i=0.35,"
         "acompressor=threshold=-22dB:ratio=4:attack=6:release=90:makeup=5:knee=6,"
         "acompressor=threshold=-10dB:ratio=3:attack=2:release=50:makeup=1")


def run(cmd, data=None):
    return subprocess.run(cmd, check=True, capture_output=True, input=data).stdout


raw = run(["ffmpeg", "-v", "error", "-i", SRC, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"])
x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
w = int(0.02 * SR)
rms = np.array([np.sqrt((x[i:i + w] ** 2).mean()) for i in range(0, len(x) - w, w)])
cand = np.where(rms > 1e-4)[0]                       # skip the digitally-muted gaps
quiet = cand[np.argsort(rms[cand])[:max(10, len(cand) // 10)]]
noise = np.concatenate([x[i * w:(i + 1) * w] for i in sorted(quiet)])
clean = nr.reduce_noise(y=x, sr=SR, y_noise=noise, stationary=True, prop_decrease=0.6, n_fft=2048,
                        freq_mask_smooth_hz=200, time_mask_smooth_ms=40)
pcm = (np.clip(clean, -1, 1) * 32767).astype(np.int16).tobytes()
proc = run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-", "-af", CHAIN, "-f", "s16le", "-"], data=pcm)
y = np.frombuffer(proc, np.int16).astype(np.float64) / 32768

track = np.zeros(int(TOTAL * SR))
for a, b, t0 in PIECES:
    seg = y[int(a * SR):int(b * SR)].copy()
    f = int(0.012 * SR)
    seg[:f] *= np.linspace(0, 1, f)
    seg[-f:] *= np.linspace(1, 0, f)
    i = int(t0 * SR)
    track[i:i + len(seg)] += seg
act = np.abs(track) > 0.02
track *= 0.2 / np.sqrt(np.mean(track[act] ** 2))
track = np.tanh(track * 1.1) / np.tanh(1.1)
track *= 0.92 / np.abs(track).max()
wavfile.write(OUT, SR, (track * 32767).astype(np.int16))
print("voice written")
