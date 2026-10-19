"""逐字对齐：用 whisper-cli 识别生成的音频，取 token 时间，再按原稿逐字对位。

原稿是已知的，识别结果只用来取时间；识别出的字和原稿不一致的地方用前后插值补齐，
同时报告一致率，低了说明 TTS 可能漏读、读错。
"""
import difflib
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

WHISPER_CLI = os.environ.get("VS_WHISPER_CLI") or shutil.which("whisper-cli") or "whisper-cli"
WHISPER_MODEL = os.environ.get(
    "VS_WHISPER_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "models", "ggml-large-v3-turbo.bin"),
)

PUNCT = set("，。！？、；：,.!?;:\"'“”‘’（）()《》【】[]—…-·~～ \n\t")


def _norm(ch):
    return ch.lower()


def speech_bounds(wav, sr, thresh_db=-40.0, win=0.02):
    """按能量找口播起止（秒）。"""
    n = max(1, int(sr * win))
    frames = len(wav) // n
    if frames == 0:
        return 0.0, len(wav) / sr
    rms = np.sqrt((wav[: frames * n].reshape(frames, n) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms / (np.abs(wav).max() + 1e-9))
    idx = np.where(db > thresh_db)[0]
    if len(idx) == 0:
        return 0.0, len(wav) / sr
    return idx[0] * win, min(len(wav) / sr, (idx[-1] + 1) * win)


def _asr_chars(wav_path, prompt):
    """返回 [(字, 开始秒, 结束秒)]，多字 token 平均分配时间。"""
    with tempfile.TemporaryDirectory() as td:
        wav16 = os.path.join(td, "a.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
                        "-ac", "1", "-ar", "16000", wav16], check=True)
        base = os.path.join(td, "r")
        subprocess.run([WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav16, "-l", "zh",
                        "-ojf", "-of", base, "-dtw", "large.v3.turbo", "-nfa",
                        "--prompt", prompt[:200]],
                       check=True, capture_output=True)
        data = json.load(open(base + ".json", encoding="utf-8", errors="replace"))
    toks = []
    for seg in data.get("transcription", []):
        for tok in seg.get("tokens", []):
            text = tok["text"].replace("\ufffd", "")
            if tok["text"].startswith("[_") or not text.strip() or tok.get("t_dtw", -1) < 0:
                continue
            toks.append(([c for c in text if not c.isspace()], tok["t_dtw"] / 100))
    out = []
    for k, (chars, t) in enumerate(toks):
        # DTW 给的是 token 的时间点；token 结束取下一个 token 的时间点
        t_next = toks[k + 1][1] if k + 1 < len(toks) else t + 0.15 * len(chars)
        t_next = max(t_next, t)
        step = (t_next - t) / len(chars)
        for j, c in enumerate(chars):
            out.append((c, t + j * step, t + (j + 1) * step))
    return out


def align(wav_path, text):
    """返回 dict: charStart, charEnd（与 text 逐字等长，本段局部秒）, voStart, voEnd, match, asr_text。"""
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    vo0, vo1 = speech_bounds(wav, sr)
    asr = _asr_chars(wav_path, text)

    # 只拿非标点字参与匹配
    s_idx = [i for i, c in enumerate(text) if c not in PUNCT]
    a_idx = [i for i, (c, _, _) in enumerate(asr) if c not in PUNCT]
    sm = difflib.SequenceMatcher(None, [_norm(text[i]) for i in s_idx],
                                 [_norm(asr[j][0]) for j in a_idx], autojunk=False)
    start = [None] * len(text)
    end = [None] * len(text)
    matched = 0
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            i, j = s_idx[blk.a + k], a_idx[blk.b + k]
            start[i], end[i] = asr[j][1], asr[j][2]
            matched += 1

    # 未匹配的字：在相邻已知时间之间线性插值；两端用口播起止兜底
    known = [i for i in range(len(text)) if start[i] is not None]
    for i in range(len(text)):
        if start[i] is not None:
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        t_prev = end[prev] if prev is not None else vo0
        t_next = start[nxt] if nxt is not None else vo1
        lo = prev if prev is not None else -1
        hi = nxt if nxt is not None else len(text)
        frac0 = (i - lo - 1) / max(1, hi - lo - 1)
        frac1 = (i - lo) / max(1, hi - lo - 1)
        if text[i] in PUNCT:
            start[i] = end[i] = t_prev if prev is not None else vo0
        else:
            start[i] = t_prev + (t_next - t_prev) * frac0
            end[i] = t_prev + (t_next - t_prev) * frac1
    # DTW 时间整体偏晚，用能量检测到的口播起止做线性校准
    nz = [i for i in range(len(text)) if text[i] not in PUNCT]
    if nz:
        a0, a1 = start[nz[0]], end[nz[-1]]
        if a1 - a0 > 0.2 and abs(a0 - vo0) < 0.8 and abs(a1 - vo1) < 0.8:
            k = (vo1 - vo0) / (a1 - a0)
            start = [vo0 + (x - a0) * k for x in start]
            end = [vo0 + (x - a0) * k for x in end]
    # 保证单调
    for i in range(1, len(text)):
        start[i] = max(start[i], start[i - 1])
        end[i] = max(end[i], start[i])
    r3 = lambda xs: [round(float(x), 3) for x in xs]
    return {
        "charStart": r3(start), "charEnd": r3(end),
        "voStart": round(vo0, 3), "voEnd": round(vo1, 3),
        "match": round(matched / max(1, len(s_idx)), 3),
        "asr_text": "".join(c for c, _, _ in asr),
    }
