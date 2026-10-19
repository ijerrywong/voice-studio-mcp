# voice-studio-mcp

用**你自己的音色** + **榜样博主的情绪节奏**生成口播，并通过 MCP 交给 AI 出片：每段时长、每个字的起止时间都有，画面和字幕可以精确对位。

基于 [IndexTTS-2.5](https://github.com/index-tts/index-tts)（bilibili）。本仓库只包含我们写的 MCP 服务、逐字对齐和分析工具，不包含 IndexTTS 的代码、权重，也不包含任何声音样本。

## 解决什么问题

自己录的口播像念书：每句话音高起伏小、句尾不落、停顿一样长。普通的声音克隆会把这种平淡也一起学走。IndexTTS-2.5 能把**音色**和**情绪**分成两路输入：音色取自你的录音，情绪取自你想学的博主的片段。

实测（同一段稿子）：

| 版本 | 基础音高 | 音高起伏 |
|---|---|---|
| 本人真人录音 | 108–111 Hz | 8 半音 |
| 只用音色 | 117–123 Hz | 9 |
| 音色 + 博主情绪（emo_alpha 0.5） | 122–128 Hz | 11 |
| 音色 + 博主情绪（emo_alpha 0.8） | 139–144 Hz | 14–16 |
| 博主本人 | 150–165 Hz | 15–17 |

情绪强度越高越有起伏，但基础音高也越往博主靠，越不像你自己。我们最后选的是 0.5。

## 原理

```
你的录音（音色）──┐
                  ├─▶ IndexTTS-2.5 ──▶ 逐段 wav ──▶ whisper 逐字对齐 ──▶ timing.js / .srt / narration.wav
博主片段（情绪）──┘                                                          │
                                                                              ▼
                                                              AI 出片时按每个字的时间做画面
```

1. **音色和情绪分开**：`spk_audio_prompt` 传音色，`emo_audio_prompt` 传情绪，`emo_alpha` 调强度。
2. **逐段生成 + 逐字对齐**：IndexTTS 不输出时间信息。每段生成后用 whisper（DTW）识别一遍，和原稿逐字对位，再用能量检测出的口播起止做首尾校准。识别和原稿不一致会报警告，等于顺带检查了有没有读错。
3. **MCP**：Claude Code 做视频时调用 `narrate`，拿回音轨和时间表。

## 安装

需要：macOS 或 Linux、16GB 内存、约 15GB 硬盘、`git` `uv` `ffmpeg` `whisper-cli`（macOS：`brew install ffmpeg whisper-cpp`）。有 NVIDIA 显卡会快很多。

```bash
git clone https://github.com/ijerrywong/voice-studio-mcp.git
cd voice-studio-mcp
bash setup.sh
```

`setup.sh` 会把 IndexTTS 固定到我们验证过的版本，从 ModelScope 下载权重并校验 SHA256，再装 MCP 依赖和 whisper 模型。版本细节和手动安装方法见 [docs/VERSIONS.md](docs/VERSIONS.md)。

## 准备两段素材

| 素材 | 要求 |
|---|---|
| 音色参考 → `voices/` | 你自己的录音，3–15 秒，干净无配乐，挑读得最清楚的句子 |
| 情绪参考 → `emotions/` | 榜样博主的片段，10–15 秒，情绪典型。先用 Demucs 去掉背景音乐：`uvx --from demucs demucs --two-stems=vocals 原音频.wav`，再在停顿处剪 |

格式都用 WAV 单声道。然后改 `presets.json`（从 `presets.example.json` 复制而来）。

自检：

```bash
index-tts/.venv/bin/python tests/smoke.py voices/me.wav emotions/role-model.wav     # 直接推理测速
index-tts/.venv/bin/python tests/mcp_e2e.py voices/me.wav emotions/role-model.wav   # 走一遍 MCP
```

## 接入 Claude Code

```bash
claude mcp add -s user voice-studio -e VS_PRELOAD=0 -- \
  "$(pwd)/index-tts/.venv/bin/python" "$(pwd)/server/voice_studio.py"
```

想让 AI 做视频时自动用它，在 `~/.claude/CLAUDE.md` 加一句：

> 做视频需要口播时，默认用 voice-studio MCP 的 `narrate`；按镜头切段，画面时间读它产出的 `timing.js`。

## MCP 工具

| 工具 | 作用 |
|---|---|
| `narrate(segments, out_dir, preset?, gap_ms?)` | 逐段生成口播。`segments` = `[{"id":"s01","text":"…"}]`。生成慢，没跑完就返回 `job_id` |
| `job_status(job_id)` | 查询任务进度和结果 |
| `retime(out_dir, seg_id, target_dur?, text?)` | 重做某段：逼近目标时长（调 `duration_factor`，最多 3 次），或者改稿 |
| `audition(text, presets, out_dir)` | 同一句话用几个预设各生成一遍，供试听比较 |
| `save_preset` / `list_presets` | 管理预设（音色、情绪、强度、语速、段间停顿） |

输出目录里有：

- `narration.wav`：整条音轨，22.05kHz 单声道
- `segments/<id>.wav`：每段单独的音频
- `timing.js` / `timing.json`：逐字时间轴，格式见下
- `narration.srt`：字幕
- `manifest.json`：每段的参数和识别结果

```js
window.TIMING = { total: 12.4, seg: { s01: {
  text, dur,          // dur = 本段音频 + 段后停顿
  t0,                 // 本段在整条音轨上的起点（秒）
  voStart, voEnd,     // 口播起止（本段局部秒）
  charStart: [...], charEnd: [...]   // 与 text 逐字等长（本段局部秒）
}}}
```

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VS_TTS_DIR` | `./index-tts` | IndexTTS 仓库位置 |
| `VS_DATA_DIR` | 仓库根目录 | `voices/` `emotions/` `presets.json` 所在目录 |
| `VS_WHISPER_CLI` | `PATH` 里的 `whisper-cli` | |
| `VS_WHISPER_MODEL` | `./models/ggml-large-v3-turbo.bin` | |
| `VS_DEVICE` | 自动（CUDA → MPS → CPU） | 如 `cpu` |
| `VS_PRELOAD` | `0` | `1` = 服务启动就加载模型（占约 4GB 内存） |

## 已知限制

- **速度**：Mac M4 上生成 1 秒音频约需 6–8 秒，1 分钟口播约 7 分钟。CPU 和 MPS 差不多。
- **时长**：IndexTTS-2.5 官方暂未开放精确时长控制，`retime` 只能逼近。思路是让画面去适应声音。
- **英文词**：缩写和产品名可能读得不对，要试听。
- **`uv sync`**：在 `index-tts/` 里重跑会删掉 mcp 和 soundfile，需要重跑 `setup.sh`，或者执行它的第 2 步。

## 授权与伦理

- 只克隆你自己的声音，或者得到本人授权的声音。
- 情绪参考取自他人作品时，只在本地使用，不要分发原片段。`.gitignore` 已经挡掉所有音频文件。
- IndexTTS 用的是 bilibili Model Use License：个人和中小团队可以免费使用；月活超过 1 亿或年收入超过门槛的，需要向 bilibili 申请。
- 本仓库代码采用 [MIT](LICENSE) 协议。

更完整的过程记录（含踩坑）见 [docs/guide.html](docs/guide.html)。
