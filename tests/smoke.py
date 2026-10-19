"""直接调用 IndexTTS-2.5 测速（不经过 MCP）。
用法：index-tts/.venv/bin/python tests/smoke.py <音色参考.wav> [情绪参考.wav]
环境变量 VS_DEVICE=cpu 可强制用 CPU。
"""
import os
import sys
import time

import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TTS = os.path.join(ROOT, "index-tts")
spk = os.path.abspath(sys.argv[1])
emo = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
os.chdir(TTS)
sys.path.insert(0, TTS)
from indextts.infer_v2_5 import IndexTTS2  # noqa: E402

t = time.time()
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", device=os.environ.get("VS_DEVICE"))
print(f"LOAD {time.time() - t:.1f}s device={tts.device}", file=sys.stderr)
text = "现在只要一句话，AI就能自己操作软件，把一整套房子的建筑图画出来。"
out = os.path.join(ROOT, "tests", "smoke.wav")
t = time.time()
kw = {"emo_audio_prompt": emo, "emo_alpha": 0.5} if emo else {}
tts.infer(spk_audio_prompt=spk, text=text, output_path=out, lang="ZH", **kw)
d = sf.info(out).duration
print(f"GEN {time.time() - t:.1f}s for {d:.2f}s audio  RTF={(time.time() - t) / d:.2f}  -> {out}", file=sys.stderr)
