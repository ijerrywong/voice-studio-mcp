"""端到端：启动 MCP 服务，存一个预设，生成两段口播，打印时间轴摘要。
用法：index-tts/.venv/bin/python tests/mcp_e2e.py <音色参考.wav> [情绪参考.wav]
"""
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "tests", "e2e_out")


async def main():
    spk = os.path.abspath(sys.argv[1])
    emo = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else None
    params = StdioServerParameters(command=sys.executable, args=[os.path.join(ROOT, "server", "voice_studio.py")],
                                   env={**os.environ})
    async with stdio_client(params, errlog=open(os.path.join(ROOT, "tests", "server.log"), "w")) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print("TOOLS", [t.name for t in (await s.list_tools()).tools])
            await s.call_tool("save_preset", {"name": "e2e-test", "speaker_ref": spk, "emotion_ref": emo,
                                              "emo_alpha": 0.5, "note": "tests/mcp_e2e.py 生成"})
            res = await s.call_tool("narrate", {"out_dir": OUT, "preset": "e2e-test", "wait_seconds": 1, "segments": [
                {"id": "s01", "text": "现在只要一句话，AI就能自己操作软件。"},
                {"id": "s02", "text": "画面和字幕，都可以对准每一个字。"}]})
            j = json.loads(res.content[0].text)
            while j["status"] not in ("done", "error"):
                j = json.loads((await s.call_tool("job_status", {"job_id": j["id"], "wait_seconds": 50})).content[0].text)
                print("..", j["status"], j.get("progress"))
            print(json.dumps(j.get("result") or j, ensure_ascii=False, indent=1))


asyncio.run(main())
