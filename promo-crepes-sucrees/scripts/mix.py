"""Mix voice-over and music: the music ducks under the voice and comes back up between lines."""
import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile

voice_p, music_p, out_p = sys.argv[1:4]
sr, v = wavfile.read(voice_p)
sr2, m = wavfile.read(music_p)
assert sr == sr2
v = v.astype(np.float64) / 32768
m = m.astype(np.float64) / 32768
n = min(len(v), len(m))
v, m = v[:n], m[:n]
# voice envelope -> ducking gain (fast attack, slow release, look-ahead 60 ms)
env = np.abs(v)
win = int(0.03 * sr)
env = np.convolve(env, np.ones(win) / win, mode="same")
active = (env > 0.012).astype(np.float64)
la = int(0.06 * sr)
active = np.concatenate([active[la:], np.zeros(la)])
att = np.exp(-1 / (0.04 * sr))
rel = np.exp(-1 / (0.35 * sr))
g = np.zeros(n)
cur = 0.0
for i in range(n):
    a = active[i]
    cur = att * cur + (1 - att) * a if a > cur else rel * cur + (1 - rel) * a
    g[i] = cur
MUSIC_FULL, MUSIC_DUCK = 0.60, 0.20          # -4.4 dB between lines, -14 dB under the voice
mg = MUSIC_FULL + (MUSIC_DUCK - MUSIC_FULL) * g
# carve a little room for the voice in the music's mids while it is ducked
sos = signal.butter(2, [900, 4500], 'bandstop', fs=sr, output='sos')
m_carved = signal.sosfilt(sos, m, axis=0)
mm = m * (1 - g)[:, None] + (0.6 * m + 0.4 * m_carved) * g[:, None]
mix = mm * mg[:, None] + (v * 1.0)[:, None]
mix = np.tanh(mix * 1.15) / np.tanh(1.15)
mix *= 0.85 / np.abs(mix).max()
wavfile.write(out_p, sr, (mix * 32767).astype(np.int16))
print("voice share of time:", round(float((g > 0.5).mean()), 2))
