"""Pre-render each timeline shot as a 4K (2160x3840) intermediate: denoise -> ESPCN x2 super-resolution -> fit."""
import os, subprocess, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edit import V
S = os.environ.get('PROMO_WORK', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f'{S}/cache4k'; os.makedirs(OUT, exist_ok=True)
cv2.setNumThreads(4)
sr = cv2.dnn_superres.DnnSuperResImpl_create(); sr.readModel(f'{S}/models/ESPCN_x2.pb'); sr.setModel('espcn', 2)
W4, H4 = 2160, 3840
only = set(sys.argv[1:])
for key, clip, ss, dur, sp, T, note in V:
    if only and key not in only:
        continue
    out = f'{OUT}/{key}.mp4'
    n = int(round(dur * 30)) + 3
    vf = []
    if sp < 1.0:
        vf.append(f"setpts=(PTS-STARTPTS)/{sp}")
        vf.append("minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    elif sp > 1.0:
        vf.append(f"tmix=frames={max(2, int(round(sp)))}")      # motion-blurred fast forward
        vf.append(f"setpts=(PTS-STARTPTS)/{sp}")
        vf.append("fps=30")
    else:
        vf.append("fps=30")
    vf.append("hqdn3d=1.2:1.2:3:3")
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{n / 30 * sp + 0.5:.3f}", "-i", f"{S}/videos/IMG_{clip}.MOV",
                            "-an", "-vf", ",".join(vf), "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W4}x{H4}", "-r", "30", "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "9", "-pix_fmt", "yuv444p", out], stdin=subprocess.PIPE)
    t0 = time.time(); k = 0
    while True:
        b = dec.stdout.read(1072 * 1920 * 3)
        if len(b) < 1072 * 1920 * 3:
            break
        f = np.frombuffer(b, np.uint8).reshape(1920, 1072, 3)
        up = sr.upsample(f) if key != "v14" else cv2.resize(f, (2144, 3840), interpolation=cv2.INTER_LANCZOS4)
        up = cv2.resize(up, (W4, 3869), interpolation=cv2.INTER_LANCZOS4)[14:14 + H4]
        enc.stdin.write(up.tobytes()); k += 1
    enc.stdin.close(); enc.wait(); dec.wait()
    print(f"{key} {clip} frames={k}/{n} {time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
