#!/usr/bin/env bash
# 一键安装：IndexTTS（固定到验证过的版本）+ 权重（ModelScope）+ MCP 依赖 + whisper 模型
# 用法：bash setup.sh            （在仓库根目录运行）
#       VS_SKIP_WEIGHTS=1 bash setup.sh   跳过下载权重
set -euo pipefail
cd "$(dirname "$0")"

# 验证过的版本，见 docs/VERSIONS.md
INDEXTTS_COMMIT=d9e41aac89fd00b3d71497fddb287b7f24613712   # 2026-09-30
MODEL_ID=IndexTeam/IndexTTS-2.5
WHISPER_URL=${VS_WHISPER_URL:-https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin}

need() { command -v "$1" >/dev/null || { echo "缺少 $1：$2"; exit 1; }; }
need git    "请先安装 git"
need uv     "curl -LsSf https://astral.sh/uv/install.sh | sh"
need ffmpeg "macOS: brew install ffmpeg"
command -v whisper-cli >/dev/null || echo "提示：没找到 whisper-cli（macOS: brew install whisper-cpp），逐字对齐需要它"

echo "== 1/4 IndexTTS 源码（$INDEXTTS_COMMIT）"
if [ ! -d index-tts/.git ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/index-tts/index-tts.git index-tts
fi
git -C index-tts fetch --depth 1 origin "$INDEXTTS_COMMIT" 2>/dev/null || true
git -C index-tts checkout -q "$INDEXTTS_COMMIT"

echo "== 2/4 Python 依赖（不装 DeepSpeed 等 CUDA 扩展）"
( cd index-tts && uv sync )
# uv sync 会删掉额外装的包，所以每次 sync 之后都要重装这一行
uv pip install --python index-tts/.venv/bin/python "mcp>=1.10,<2" soundfile praat-parselmouth

if [ "${VS_SKIP_WEIGHTS:-0}" != "1" ]; then
  echo "== 3/4 模型权重（ModelScope，约 4.3GB）"
  uvx --from modelscope modelscope download --model "$MODEL_ID" --local_dir index-tts/checkpoints
  echo "-- 校验 SHA256"
  ( cd index-tts/checkpoints && shasum -a 256 -c ../../docs/checkpoints.sha256 )
fi

echo "== 4/4 whisper 模型（约 1.6GB）"
mkdir -p models
if [ ! -s models/ggml-large-v3-turbo.bin ]; then
  curl -L --fail -o models/ggml-large-v3-turbo.bin "$WHISPER_URL"
fi

mkdir -p voices emotions
[ -f presets.json ] || cp presets.example.json presets.json

cat <<EOF

安装完成。下一步：
  1. 把你的音色参考放进 voices/，情绪参考放进 emotions/，改 presets.json
  2. 自检：index-tts/.venv/bin/python tests/smoke.py
  3. 注册 MCP：
     claude mcp add -s user voice-studio -e VS_PRELOAD=0 -- \\
       $(pwd)/index-tts/.venv/bin/python $(pwd)/server/voice_studio.py
首次生成时 IndexTTS 还会自动下载约 4GB 辅助模型（国内会自动走 ModelScope）。
EOF
