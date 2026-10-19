"""量化口播表现力：音域(半音)、语速、句尾下降、音量起伏。用法: python expressiveness.py a.wav [b.wav ...]"""
import sys
import numpy as np
import parselmouth as pm

st = lambda x: 12 * np.log2(x / 100)
print(f"{'文件':44s} {'F0中位':>6s} {'音域':>5s} {'句尾降':>6s} {'音量起伏':>6s} {'时长':>5s}")
for f in sys.argv[1:]:
    s = pm.Sound(f)
    p = s.to_pitch_ac(time_step=0.01, pitch_floor=70, pitch_ceiling=500)
    f0 = p.selected_array["frequency"]; v = f0[f0 > 0]
    it = s.to_intensity(time_step=0.01).values[0]
    loud = it[it > np.percentile(it, 30)]
    tail = v[-max(5, len(v) // 6):]; head = v[: len(v) // 2]
    print(f"{f[-44:]:44s} {np.median(v):6.0f} {st(np.percentile(v,95))-st(np.percentile(v,5)):5.1f} "
          f"{st(np.median(head))-st(np.median(tail)):6.1f} {np.std(loud):6.1f}dB {s.duration:5.2f}")
