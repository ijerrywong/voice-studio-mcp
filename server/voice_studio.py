"""voice-studio MCP：用固定音色 + 情绪参考（IndexTTS-2.5）生成口播，并输出逐字时间轴。

输出与 huashu-art-motion 口播整片的 timing.js 同格式：
  window.TIMING = {total, seg: {id: {text, dur, voStart, voEnd, t0, charStart[], charEnd[]}}}
其中 charStart/charEnd/voStart/voEnd 是本段局部秒，t0 是本段在整条音轨上的起点，
dur 是本段占用的时长（音频 + 段后停顿）。

stdout 是 MCP 通道，IndexTTS 的 print 会污染它，所以启动时把 fd 1 改指 stderr，
MCP 用复制出来的原 stdout。
"""
import os
import sys

_MCP_STDOUT_FD = os.dup(1)
os.dup2(2, 1)

import io
import json
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import anyio
import numpy as np
import soundfile as sf
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "server"))
import align  # noqa: E402

# 三个目录都可用环境变量改：IndexTTS 仓库、数据（voices/ emotions/ presets.json）
TTS_DIR = os.path.abspath(os.path.expanduser(os.environ.get("VS_TTS_DIR", os.path.join(ROOT, "index-tts"))))
DATA = os.path.abspath(os.path.expanduser(os.environ.get("VS_DATA_DIR", ROOT)))
os.chdir(TTS_DIR)
sys.path.insert(0, TTS_DIR)
PRESETS = os.path.join(DATA, "presets.json")
SR = 22050

mcp = FastMCP("voice-studio")
_pool = ThreadPoolExecutor(max_workers=1)  # 模型不是线程安全的，串行跑
_jobs = {}
_tts = None
_tts_lock = threading.Lock()


# ---------- 预设 ----------

def _load_presets():
    if not os.path.exists(PRESETS):
        return {"default": None, "presets": {}}
    return json.load(open(PRESETS, encoding="utf-8"))


def _save_presets(p):
    json.dump(p, open(PRESETS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _preset(name):
    p = _load_presets()
    name = name or p.get("default")
    if not name or name not in p["presets"]:
        raise ValueError(f"没有预设「{name}」。现有：{list(p['presets'])}，默认：{p.get('default')}")
    return name, p["presets"][name]


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(DATA, path)


def _out_dir(path):
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        raise ValueError(f"out_dir 要写绝对路径：{path}")
    return path


# ---------- 模型 ----------

def _get_tts():
    global _tts
    with _tts_lock:
        if _tts is None:
            from indextts.infer_v2_5 import IndexTTS2
            device = os.environ.get("VS_DEVICE") or None
            _tts = IndexTTS2(cfg_path=os.path.join(TTS_DIR, "checkpoints", "config.yaml"),
                             model_dir=os.path.join(TTS_DIR, "checkpoints"), device=device)
        return _tts


def _synth(text, preset, out_path, duration_factor=None):
    tts = _get_tts()
    kw = dict(emo_alpha=preset.get("emo_alpha", 1.0))
    if preset.get("emotion_ref"):
        kw["emo_audio_prompt"] = _abs(preset["emotion_ref"])
    elif preset.get("emo_vector"):
        kw["emo_vector"] = preset["emo_vector"]
    tts.infer(spk_audio_prompt=_abs(preset["speaker_ref"]), text=text,
              output_path=out_path, lang=preset.get("lang", "ZH"),
              duration_factor=duration_factor or preset.get("duration_factor", 1.0),
              **kw)
    return out_path


def _wav_dur(path):
    info = sf.info(path)
    return info.frames / info.samplerate


# ---------- 拼装时间轴 ----------

def _build_outputs(out_dir, manifest):
    """按 manifest 里各段的 wav 和对齐结果，拼整条音轨，写 timing.json/js 和 srt。"""
    gap = manifest["gap_ms"] / 1000
    parts, t, seg = [], 0.0, {}
    srt, n = [], 1
    for i, s in enumerate(manifest["segments"]):
        wav, sr = sf.read(_abs_out(out_dir, s["wav"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != SR:
            raise RuntimeError(f"{s['wav']} 采样率 {sr}，应为 {SR}")
        g = s.get("gap_after_ms", manifest["gap_ms"]) / 1000 if i < len(manifest["segments"]) - 1 else 0.0
        parts += [wav, np.zeros(int(round(g * SR)), dtype="float32")]
        audio_dur = len(wav) / SR
        dur = audio_dur + g
        seg[s["id"]] = {"text": s["text"], "dur": round(dur, 4), "voStart": s["voStart"],
                        "voEnd": s["voEnd"], "t0": round(t, 4),
                        "charStart": s["charStart"], "charEnd": s["charEnd"]}
        srt.append(f"{n}\n{_srt_t(t + s['voStart'])} --> {_srt_t(t + s['voEnd'])}\n{s['text']}\n")
        n += 1
        t += dur
    track = np.concatenate(parts) if parts else np.zeros(0, dtype="float32")
    sf.write(os.path.join(out_dir, "narration.wav"), track, SR, subtype="PCM_16")
    timing = {"total": round(t, 4), "seg": seg}
    json.dump(timing, open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    with open(os.path.join(out_dir, "timing.js"), "w", encoding="utf-8") as f:
        f.write("// 由 voice-studio MCP 生成，勿手改。seg[id]: dur 段长（含段后停顿）；t0 段起点（整条音轨秒）；"
                "voStart/voEnd 口播起止；charStart/charEnd 逐字起止（本段局部秒）\n")
        f.write("window.TIMING = " + json.dumps(timing, ensure_ascii=False) + ";\n")
    open(os.path.join(out_dir, "narration.srt"), "w", encoding="utf-8").write("\n".join(srt))
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return timing


def _abs_out(out_dir, rel):
    return rel if os.path.isabs(rel) else os.path.join(out_dir, rel)


def _srt_t(x):
    ms = int(round(x * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _summary(out_dir, timing, manifest):
    segs = []
    for s in manifest["segments"]:
        tm = timing["seg"][s["id"]]
        item = {"id": s["id"], "t0": tm["t0"], "dur": tm["dur"],
                "voStart": tm["voStart"], "voEnd": tm["voEnd"], "match": s["match"]}
        if s["match"] < 0.9:
            item["warning"] = f"识别一致率 {s['match']}，可能漏读/读错，识别结果：{s['asr_text']}"
        segs.append(item)
    return {"out_dir": out_dir, "total": timing["total"], "preset": manifest["preset"],
            "files": {k: os.path.join(out_dir, k) for k in
                      ["narration.wav", "timing.js", "timing.json", "narration.srt", "manifest.json"]},
            "segments": segs}


def _gen_segment(out_dir, s, preset, duration_factor=None):
    rel = os.path.join("segments", f"{s['id']}.wav")
    path = os.path.join(out_dir, rel)
    if os.path.exists(path):
        os.remove(path)
    _synth(s["text"], preset, path, duration_factor)
    a = align.align(path, s["text"])
    return {"id": s["id"], "text": s["text"], "wav": rel, "audio_dur": round(_wav_dur(path), 4),
            "duration_factor": duration_factor or preset.get("duration_factor", 1.0), **a,
            **({"gap_after_ms": s["gap_after_ms"]} if "gap_after_ms" in s else {})}


def _narrate_job(job, segments, out_dir, preset_name, gap_ms):
    name, preset = _preset(preset_name)
    os.makedirs(os.path.join(out_dir, "segments"), exist_ok=True)
    manifest = {"preset": name, "preset_params": preset,
                "gap_ms": preset.get("gap_ms", 400) if gap_ms is None else gap_ms, "segments": []}
    job["status"] = "loading_model" if _tts is None else "running"
    _get_tts()
    job["status"] = "running"
    for i, s in enumerate(segments):
        job["progress"] = f"{i}/{len(segments)} 正在生成 {s['id']}"
        t0 = time.time()
        manifest["segments"].append(_gen_segment(out_dir, s, preset))
        job.setdefault("seconds_per_segment", []).append(round(time.time() - t0, 1))
    job["progress"] = f"{len(segments)}/{len(segments)} 完成"
    timing = _build_outputs(out_dir, manifest)
    return _summary(out_dir, timing, manifest)


def _run_job(fn, *args):
    jid = uuid.uuid4().hex[:8]
    job = {"id": jid, "status": "queued", "started": time.time()}
    _jobs[jid] = job

    def wrap():
        try:
            job["result"] = fn(job, *args)
            job["status"] = "done"
        except Exception as e:
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
            job["traceback"] = traceback.format_exc()[-2000:]
        job["elapsed"] = round(time.time() - job["started"], 1)

    _pool.submit(wrap)
    return jid


def _wait(jid, wait_seconds):
    end = time.time() + max(0, wait_seconds)
    while time.time() < end and _jobs[jid]["status"] not in ("done", "error"):
        time.sleep(0.5)
    j = dict(_jobs[jid])
    j.pop("started", None)
    return j


# ---------- MCP 工具 ----------

@mcp.tool()
def narrate(segments: list[dict], out_dir: str, preset: str | None = None,
            gap_ms: int | None = None, wait_seconds: int = 50) -> dict:
    """给视频生成口播（用户本人音色 + 预设里的情绪），并输出逐字时间轴。做视频需要口播时默认用它。

    segments: [{"id": "s01", "text": "这一段的口播稿", "gap_after_ms"?: 600}, ...]
              按画面段落切，一段对应一个镜头/场景；id 会成为 timing.js 里的键。
    out_dir:  输出目录（建议放在视频项目里，如 <项目>/口播/）。
    preset:   预设名，省略用默认预设。
    gap_ms:   段与段之间的停顿，省略用预设值。
    wait_seconds: 最多等多久；没跑完就返回 job_id，再用 job_status 取结果。

    产出：narration.wav（22.05kHz 单声道整条音轨）、segments/<id>.wav、
    timing.js（window.TIMING，可直接给 TM.cue 用）、timing.json、narration.srt、manifest.json。
    返回每段的 t0（整条音轨起点）、dur、voStart/voEnd，以及识别一致率（低于 0.9 会给 warning）。
    """
    ids = [s.get("id") for s in segments]
    if any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("每段都要有唯一的 id")
    out_dir = _out_dir(out_dir)
    jid = _run_job(_narrate_job, segments, out_dir, preset, gap_ms)
    return _wait(jid, wait_seconds)


@mcp.tool()
def job_status(job_id: str, wait_seconds: int = 50) -> dict:
    """查询 narrate/retime/audition 任务；最多等 wait_seconds 秒。"""
    if job_id not in _jobs:
        raise ValueError(f"没有任务 {job_id}")
    return _wait(job_id, wait_seconds)


def _retime_job(job, out_dir, seg_id, target_dur, text):
    manifest = json.load(open(os.path.join(out_dir, "manifest.json"), encoding="utf-8"))
    preset = manifest["preset_params"]
    idx = next((i for i, s in enumerate(manifest["segments"]) if s["id"] == seg_id), None)
    if idx is None:
        raise ValueError(f"manifest 里没有段 {seg_id}")
    cur = manifest["segments"][idx]
    s = {"id": seg_id, "text": text or cur["text"]}
    if "gap_after_ms" in cur:
        s["gap_after_ms"] = cur["gap_after_ms"]
    tries = []
    if target_dur is None:
        new = _gen_segment(out_dir, s, preset, cur.get("duration_factor"))
    else:
        factor = cur.get("duration_factor", 1.0) * target_dur / cur["audio_dur"]
        new = None
        for _ in range(3):
            factor = float(min(2.0, max(0.5, factor)))
            new = _gen_segment(out_dir, s, preset, factor)
            tries.append({"duration_factor": round(factor, 3), "audio_dur": new["audio_dur"]})
            if abs(new["audio_dur"] - target_dur) / target_dur < 0.03:
                break
            factor *= target_dur / new["audio_dur"]
    manifest["segments"][idx] = new
    timing = _build_outputs(out_dir, manifest)
    res = _summary(out_dir, timing, manifest)
    res["retime"] = {"id": seg_id, "target_dur": target_dur, "tries": tries,
                     "audio_dur": new["audio_dur"]}
    return res


@mcp.tool()
def retime(out_dir: str, seg_id: str, target_dur: float | None = None, text: str | None = None,
           wait_seconds: int = 50) -> dict:
    """重做某一段并更新整条音轨和时间轴。

    target_dur: 想要这段音频（不含段后停顿）接近的秒数，会调 duration_factor 重生成，最多 3 次，误差 3% 内停。
                IndexTTS-2.5 的精确时长控制尚未开放，所以只能逼近；duration_factor 限 0.5–2.0。
    text:       改稿时传新文字；不传就用原文重新生成（换一次随机结果）。
    """
    out_dir = _out_dir(out_dir)
    jid = _run_job(_retime_job, out_dir, seg_id, target_dur, text)
    return _wait(jid, wait_seconds)


def _audition_job(job, text, presets, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for name in presets:
        n, p = _preset(name)
        path = os.path.join(out_dir, f"{n}.wav")
        if os.path.exists(path):
            os.remove(path)
        _synth(text, p, path)
        out.append({"preset": n, "wav": path, "dur": round(_wav_dur(path), 3)})
    return {"text": text, "results": out}


@mcp.tool()
def audition(text: str, presets: list[str], out_dir: str, wait_seconds: int = 50) -> dict:
    """同一句话用几个预设各生成一遍，供用户试听比较。"""
    out_dir = _out_dir(out_dir)
    jid = _run_job(_audition_job, text, presets, out_dir)
    return _wait(jid, wait_seconds)


@mcp.tool()
def list_presets() -> dict:
    """列出口播预设（音色参考、情绪参考、情绪强度、语速、段间停顿）和默认预设。"""
    return _load_presets()


@mcp.tool()
def save_preset(name: str, speaker_ref: str, emotion_ref: str | None = None,
                emo_alpha: float = 0.8, duration_factor: float = 1.0, gap_ms: int = 400,
                note: str = "", set_default: bool = False) -> dict:
    """保存口播预设。speaker_ref/emotion_ref 是 wav 路径（会复制进数据目录的 voices/、emotions/；相对路径按数据目录解释）。

    speaker_ref: 用户本人音色参考，3–15 秒、干净无配乐。
    emotion_ref: 情绪/节奏参考（榜样博主的片段），10–15 秒；只取情绪，不取音色。不给则沿用音色参考的情绪。
    emo_alpha:   情绪强度 0–1。duration_factor: 整体语速，>1 更慢。gap_ms: 段间停顿。
    """
    p = _load_presets()

    def keep(src, sub):
        src = _abs(os.path.expanduser(src))
        if not os.path.exists(src):
            raise ValueError(f"文件不存在：{src}")
        if src.startswith(DATA + os.sep):
            return os.path.relpath(src, DATA)
        os.makedirs(os.path.join(DATA, sub), exist_ok=True)
        dst = os.path.join(sub, f"{name}_{os.path.basename(src)}")
        shutil.copyfile(src, os.path.join(DATA, dst))
        return dst

    p["presets"][name] = {"speaker_ref": keep(speaker_ref, "voices"),
                          "emotion_ref": keep(emotion_ref, "emotions") if emotion_ref else None,
                          "emo_alpha": emo_alpha, "duration_factor": duration_factor,
                          "gap_ms": gap_ms, "lang": "ZH", "note": note}
    if set_default or not p.get("default"):
        p["default"] = name
    _save_presets(p)
    return p


async def _main():
    out = anyio.wrap_file(io.TextIOWrapper(os.fdopen(_MCP_STDOUT_FD, "wb"), encoding="utf-8"))
    async with stdio_server(stdout=out) as (r, w):
        await mcp._mcp_server.run(r, w, mcp._mcp_server.create_initialization_options())


if __name__ == "__main__":
    if os.environ.get("VS_PRELOAD", "0") == "1":
        _pool.submit(_get_tts)  # 启动即后台加载模型
    anyio.run(_main)
