# 验证过的版本（下载推荐）

本仓库在下面这套组合上实测通过（2026-10-18，Mac M4 16GB）。`setup.sh` 会自动固定到这套版本并校验。

## IndexTTS 源码

| 项目 | 值 |
|---|---|
| 仓库 | https://github.com/index-tts/index-tts |
| commit | `d9e41aac89fd00b3d71497fddb287b7f24613712`（2026-09-30，"fix: 末尾加淡出…" #806） |
| 推理入口 | `indextts/infer_v2_5.py` 的 `IndexTTS2` |
| 许可 | bilibili Model Use License Agreement（不是 MIT；本仓库不包含它的任何代码或权重） |

手动安装：

```bash
git clone https://github.com/index-tts/index-tts.git
cd index-tts && git checkout d9e41aac89fd00b3d71497fddb287b7f24613712
uv sync            # 不要加 --all-extras：Mac 上 DeepSpeed 装不上
```

## 模型权重：IndexTeam/IndexTTS-2.5

国内推荐 **ModelScope**，HuggingFace 的大文件在国内经常卡住。

| 来源 | 地址 |
|---|---|
| ModelScope（推荐） | https://modelscope.cn/models/IndexTeam/IndexTTS-2.5 |
| HuggingFace | https://huggingface.co/IndexTeam/IndexTTS-2.5 |

```bash
uvx --from modelscope modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir checkpoints
cd checkpoints && shasum -a 256 -c ../../docs/checkpoints.sha256
```

ModelScope 上这个模型没有版本标签（只有 `master` 分支），所以用 SHA256 锁定版本。下表是我们用的文件，已和 ModelScope 2026-10-19 的 `master` 逐个比对一致：

| 文件 | 大小 | SHA256（前 16 位） |
|---|---|---|
| gpt.pth | 3.26 GB | `43a8f4c30eccdf20` |
| codec.pth | 607 MB | `d15cbed16a40f478` |
| s2mel.pth | 415 MB | `9b1b0003fc189c94` |
| config.yaml | 2.9 KB | `18adf417be3e8f5e` |

完整校验值见 `docs/checkpoints.sha256`。如果以后官方更新了权重、校验不过，代码多半还能用，但效果数据就不能保证和我们一样了。

首次运行时 IndexTTS 还会自动下载辅助模型，约 4GB：w2v-bert-2.0、campplus、BigVGAN、MaskGCT codec。代码会检测网络，必要时自动改走 ModelScope。

## 其他依赖

| 依赖 | 版本 / 说明 |
|---|---|
| Python | 3.10–3.11（IndexTTS 要求） |
| torch | 2.8（由 `uv sync` 装；Mac 上 CPU 和 MPS 速度差不多） |
| mcp | `>=1.10,<2`（2.x 把 FastMCP 改名了，本仓库用的是 1.x API） |
| whisper.cpp | `whisper-cli` + `ggml-large-v3-turbo.bin`（逐字对齐，需要 `-dtw` 支持） |
| ffmpeg | 任意较新版本 |
