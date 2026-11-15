"""Voice-over track for the Resto Crêpes ad: clean, brighten, tighten, place on the edit timeline.

Usage: voice.py RAW_AUDIO OUT_WAV [TOTAL_SECONDS]
The recording is cut into its spoken phrases, each phrase is enhanced and slightly sped up
(pitch preserved, plus a small lift for a brighter, more smiling tone) and placed so that the
price lines land on the price reveals of the video.
"""
import subprocess
import sys

import numpy as np
from scipy.io import wavfile

SR = 48000
RAW, OUT = sys.argv[1], sys.argv[2]
TOTAL = float(sys.argv[3]) if len(sys.argv) > 3 else 33.0

# phrase: (start in recording, end in recording, start in video, tempo)
PHRASES = [
    ("accueil / promo",          0.14,  4.46,  0.12, 1.15),
    ("Kounafa Pistache",         5.84, 11.42,  3.98, 1.15),
    ("30 dh f blast 38",        12.04, 14.60, 10.02, 1.15),
    ("Crêpe 4 You",             15.92, 20.80, 12.95, 1.15),
    ("30 dh f blast 35",        21.26, 25.44, 19.32, 1.20),
    ("ga3 les crêpes f promo",  25.52, 30.26, 22.90, 1.20),
    ("mer7ba bikom",            30.32, 34.82, 26.98, 1.20),
    ("7tal 22 novembre",        35.48, 37.66, 30.90, 1.20),
]
PITCH = 1.03    # +0.5 semitone: brighter, livelier tone without sounding processed

# enhancement chain: rumble cut, light denoise, less boxiness, presence + air ("smile"), de-ess,
# harmonic exciter, firm compression for a punchy radio-ad delivery
CHAIN = (
    "highpass=f=95,"
    "afftdn=nr=12:nf=-50:tn=1,"
    "equalizer=f=280:t=q:w=1.0:g=-3,"
    "equalizer=f=3800:t=q:w=0.9:g=3.5,"
    "equalizer=f=1700:t=q:w=1.2:g=1.5,"
    "highshelf=f=9500:g=3,"
    "deesser=i=0.35:m=0.5:f=0.5,"
    "aexciter=level_in=1:level_out=1:amount=1.2:drive=5:blend=0:freq=6500:ceil=14000,"
    "acompressor=threshold=-22dB:ratio=4:attack=6:release=90:makeup=5:knee=6,"
    "acompressor=threshold=-12dB:ratio=3:attack=2:release=60:makeup=1.5,"
    "alimiter=limit=0.89:attack=3:release=40:level=disabled"
)


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def phrase_audio(a, b, tempo):
    af = (f"atrim={a}:{b},asetpts=PTS-STARTPTS,{CHAIN},"
          f"rubberband=tempo={tempo}:pitch={PITCH}:formant=preserved:transients=crisp,"
          f"afade=t=in:d=0.02,areverse,afade=t=in:d=0.05,areverse")
    raw = run(["ffmpeg", "-v", "error", "-i", RAW, "-ac", "1", "-ar", str(SR), "-af", af, "-f", "s16le", "-"])
    return np.frombuffer(raw, np.int16).astype(np.float64) / 32768


track = np.zeros(int(TOTAL * SR))
for name, a, b, t0, tempo in PHRASES:
    x = phrase_audio(a, b, tempo)
    # level each phrase to the same loudness so no line drops
    rms = np.sqrt(np.mean(x[np.abs(x) > 0.02] ** 2)) if np.any(np.abs(x) > 0.02) else 1e-3
    x *= 0.16 / rms
    i = int(t0 * SR)
    n = min(len(x), len(track) - i)
    track[i:i + n] += x[:n]
    print(f"{name:26s} video {t0:5.2f} -> {t0 + len(x) / SR:5.2f} s  (x{tempo})")
track = np.tanh(track * 1.1) / np.tanh(1.1)
track *= 0.92 / np.abs(track).max()
wavfile.write(OUT, SR, (track * 32767).astype(np.int16))
print("wrote", OUT)
