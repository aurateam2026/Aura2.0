#!/usr/bin/env python3
"""End-to-end check through the gateway page protocol (/ws), like a browser.

Sends camera frames, a push-to-talk question, a second spoken question (which
usually calls a tool right after a spoken reply) and a typed question, and
requires ASR text, reply text, audio and turn_done back.

  python smoke_test.py --url ws://127.0.0.1:9310/ws --assets <dir with q1_16k.wav q2_16k.wav>
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import time
from pathlib import Path

import websockets
from PIL import Image


def jpeg_b64(color: tuple[int, int, int]) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), color=color).save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def packet(sid: str, cid: int, mtype: str, data: dict, ts: int) -> str:
    return json.dumps(
        {
            "direction": "upstream",
            "type": mtype,
            "session_id": sid,
            "chunk_id": cid,
            "video_timestamp_ms": ts,
            "wall_time_ms": int(time.time() * 1000),
            "data": data,
        }
    )


async def turn(ws, timeout: float) -> dict:
    seen = {"asr": [], "text": [], "audio": 0, "tools": [], "errors": [], "done": False}
    t0 = time.time()
    first_audio = None
    while time.time() - t0 < timeout:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout - (time.time() - t0))
        except asyncio.TimeoutError:
            break
        msg = json.loads(raw)
        kind, data = msg.get("type"), msg.get("data") or {}
        if kind == "asr_text":
            seen["asr"].append(data.get("text"))
        elif kind == "text":
            seen["text"].append(data.get("text"))
        elif kind == "audio":
            seen["audio"] += 1
            first_audio = first_audio or round(time.time() - t0, 2)
        elif kind == "toolcall" and data.get("status") == "started":
            seen["tools"].append(data.get("name"))
        elif kind == "error":
            seen["errors"].append(data.get("message"))
        elif kind == "turn_done" and (seen["text"] or seen["errors"] or seen["audio"]):
            seen["done"] = True
            break
    seen["first_audio_s"] = first_audio
    return seen


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:9310/ws")
    p.add_argument("--assets", type=Path, help="Private directory containing q1_16k.wav and q2_16k.wav.")
    p.add_argument("--timeout", type=float, default=90)
    a = p.parse_args()
    sample = Path(__file__).resolve().parents[2] / "tests/assets/minicpmo_4_5/response_required_16k.wav"
    frames = [jpeg_b64((200, 40, 40)), jpeg_b64((40, 200, 40))]
    video = {"video_url": "data:video/jpeg;base64," + ",".join(frames), "frame_count": 2}

    def speech(name: str) -> dict:
        audio_path = a.assets / name if a.assets is not None else sample
        return {"audio_b64": base64.b64encode(audio_path.read_bytes()).decode(), **video}

    results = []
    async with websockets.connect(a.url, max_size=64 * 1024 * 1024, open_timeout=30) as ws:
        hello = json.loads(await asyncio.wait_for(ws.recv(), 30))
        if hello.get("type") != "session":
            print(f"unexpected first message: {hello}")
            return 1
        sid = hello["session_id"]
        for cid in (1, 2):
            await ws.send(packet(sid, cid, "video", video, cid * 1000))
            await asyncio.sleep(1)
        await turn(ws, 5)  # drain whatever vision-follow produced
        cases = [
            ("spoken question", packet(sid, 3, "multimodal", speech("q1_16k.wav"), 3000)),
            ("spoken question after a spoken reply", packet(sid, 4, "multimodal", speech("q2_16k.wav"), 3500)),
            ("typed question", packet(sid, 5, "text", {"text": "帮我算一下123乘以45"}, 4000)),
        ]
        for label, msg in cases:
            await ws.send(msg)
            r = await turn(ws, a.timeout)
            passed = bool(r["text"]) and r["audio"] > 0 and r["done"] and not r["errors"]
            results.append(passed)
            print(
                f"  [{'PASS' if passed else 'FAIL'}] {label}: asr={r['asr']} text={r['text']} "
                f"audio_chunks={r['audio']} tools={r['tools']} "
                f"first_audio={r['first_audio_s']}s errors={r['errors']}"
            )
    ok = all(results)
    print("SMOKE TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
