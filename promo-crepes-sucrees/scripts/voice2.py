"""Voice-over track v2 for the Resto Crêpes ad (second recording).

Usage: voice2.py RAW_AUDIO OUT_WAV [TOTAL_SECONDS]
1. background-noise removal (spectral gating learned from the recording itself),
2. each spoken phrase is cut out, brightened and compressed for a lively ad delivery,
3. sped up slightly with a small pitch/formant lift (a brighter, more "smiling" colour),
4. placed on the edit so the price lines land on the price reveals.
"""
import subprocess
import sys

import noisereduce as nr
import numpy as np
from scipy.io import wavfile

SR = 48000
RAW, OUT = sys.argv[1], sys.argv[2]
TOTAL = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

# phrase: (start in recording, end in recording, start in video, tempo)
PHRASES = [
    ("accueil / promo",          0.10,  3.90,  0.12, 1.15),
    ("Kounafa Pistache",         4.56, 10.16,  3.75, 1.15),
    ("30 dh f blast 38",        10.82, 13.66, 10.08, 1.15),
    ("Crêpe 4 You",             14.64, 19.80, 12.95, 1.15),
    ("30 dh f blast 35",        20.30, 22.38, 20.20, 1.12),
    ("ga3 les crêpes f promo",  23.28, 27.44, 22.35, 1.18),
    ("7tal 22 novembre",        27.66, 31.72, 26.05, 1.18),
]
PITCH = 1.045   # ~ +0.75 semitone with formants following: brighter, more smiling colour

CHAIN = (
    "highpass=f=100,"
    "agate=threshold=0.012:ratio=3:attack=4:release=120:range=0.25,"   # push residual room noise down between words
    "equalizer=f=260:t=q:w=1.0:g=-3.5,"                                # less boxy
    "equalizer=f=1800:t=q:w=1.2:g=1.5,"
    "equalizer=f=4000:t=q:w=0.9:g=4,"                                  # presence / clarity
    "highshelf=f=9000:g=4,"                                            # air
    "deesser=i=0.4:m=0.5:f=0.5,"
    "aexciter=level_in=1:level_out=1:amount=1.5:drive=6:blend=0:freq=6000:ceil=15000,"
    "acompressor=threshold=-24dB:ratio=5:attack=5:release=80:makeup=6:knee=6,"   # punchy, even delivery
    "acompressor=threshold=-12dB:ratio=3:attack=2:release=60:makeup=1.5,"
    "alimiter=limit=0.89:attack=3:release=40:level=disabled"
)


def run(cmd, data=None):
    return subprocess.run(cmd, check=True, capture_output=True, input=data).stdout


def floor_db(sig, w):
    r = np.sort(np.array([np.sqrt((sig[i:i + w] ** 2).mean()) for i in range(0, len(sig) - w, w)]))
    return round(20 * np.log10(r[len(r) // 20] + 1e-9), 1)


# ---- 1. decode + denoise the whole take
raw = run(["ffmpeg", "-v", "error", "-i", RAW, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"])
x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
w = int(0.02 * SR)
rms = np.array([np.sqrt((x[i:i + w] ** 2).mean()) for i in range(0, len(x) - w, w)])
quiet = np.argsort(rms)[:max(10, len(rms) // 12)]            # quietest moments = noise profile
noise = np.concatenate([x[i * w:(i + 1) * w] for i in sorted(quiet)])
# settings chosen by measurement: floor -53 dB -> about -70 dB while the voice itself loses < 1 dB
clean = nr.reduce_noise(y=x, sr=SR, y_noise=noise, stationary=True, prop_decrease=0.88, n_fft=2048,
                        n_std_thresh_stationary=1.0, freq_mask_smooth_hz=200, time_mask_smooth_ms=30)
print("noise floor before / after (dB):", floor_db(x, w), "/", floor_db(clean, w))
pcm = (np.clip(clean, -1, 1) * 32767).astype(np.int16).tobytes()


def phrase_audio(a, b, tempo):
    af = (f"atrim={a}:{b},asetpts=PTS-STARTPTS,{CHAIN},"
          f"rubberband=tempo={tempo}:pitch={PITCH}:transients=crisp,"
          f"aecho=0.9:0.5:19|31:0.07|0.04,"
          f"afade=t=in:d=0.02,areverse,afade=t=in:d=0.06,areverse")
    out = run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-", "-af", af,
               "-f", "s16le", "-"], data=pcm)
    return np.frombuffer(out, np.int16).astype(np.float64) / 32768


track = np.zeros(int(TOTAL * SR))
for name, a, b, t0, tempo in PHRASES:
    y = phrase_audio(a, b, tempo)
    act = np.abs(y) > 0.02
    y *= 0.17 / (np.sqrt(np.mean(y[act] ** 2)) if act.any() else 1e-3)     # same loudness for every line
    i = int(t0 * SR)
    n = min(len(y), len(track) - i)
    track[i:i + n] += y[:n]
    print(f"{name:26s} video {t0:5.2f} -> {t0 + len(y) / SR:5.2f} s  (x{tempo})")
track = np.tanh(track * 1.15) / np.tanh(1.15)
track *= 0.92 / np.abs(track).max()
wavfile.write(OUT, SR, (track * 32767).astype(np.int16))
print("wrote", OUT)
