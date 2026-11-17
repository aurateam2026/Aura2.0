# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Background memory summarization for duplex AURA (experimental).

Long duplex sessions grow ``SessionHistory`` without bound. This module
compresses the non-recent prefix of the Stage1 history into one text memory
row using an external OpenAI-compatible VLM, so the prompt keeps early
context while staying short.

Ported from the AURA-qwen3-5 single-file demo (``aura_memory.py``), adapted to
this tree's ``SessionHistory``:

* ``messages`` is a flat list of ``user`` / ``assistant`` / ``tool_response``
  rows, where only ``user`` rows may carry a ``(array, metadata)`` video tuple;
* the loop runs on its own daemon thread and touches history only between
  turns: it skips while ``pending_user`` is set (a turn is in flight) and
  re-validates the snapshot before committing;
* tool rounds are never split and ``tool_response`` rows never reach the VLM.

Configuration is environment-only (see ``docs/DEPLOYMENT_NOTES.md``):

``AURA_MEMORY_ENABLE`` (0/1), ``AURA_MEMORY_VLM_URL``, ``AURA_MEMORY_VLM_MODEL``,
``AURA_MEMORY_INTERVAL_S``, ``AURA_MEMORY_MIN_ROUNDS``,
``AURA_MEMORY_KEEP_ROUNDS``, ``AURA_MEMORY_MAX_FRAMES``,
``AURA_MEMORY_TIMEOUT_S``.
"""

from __future__ import annotations

import base64
import os
import re
import threading
import traceback

from vllm.logger import init_logger

from vllm_omni.model_executor.models.aura_omni.duplex.history import (
    HISTORY_LOCK,
    SessionHistory,
    get_or_create_session_history,
)

logger = init_logger(__name__)

MEMORY_PREFIX = "我看到了"
SILENT = "<|silent|>"
_RESPOND = "<|respond|>"
_MEMORY_SUMMARY_PROMPT = (
    "你正在帮助一个实时视频助手压缩记忆。以下是一段时间内的视频帧和用户对话记录。\n"
    "请用简洁的中文总结你观察到的关键事件、出现的物体/人物、场景变化、"
    "以及与用户问题相关的重要事实。只输出总结，不要输出其他内容。"
)
_TOOL_XML_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|$)|<tool_response>.*?(?:</tool_response>|$)", re.DOTALL)

_LOOPS: dict[str, _MemoryLoop] = {}
_LOOPS_LOCK = threading.Lock()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("[memory] %s=%r is not an int; using %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("[memory] %s=%r is not a float; using %s", name, raw, default)
        return default


def _vlm_url() -> str:
    url = os.getenv("AURA_MEMORY_VLM_URL", "").strip() or "http://127.0.0.1:8003/v1/chat/completions"
    if not url.endswith("/chat/completions") and not url.endswith("/chat/completions/"):
        url = url.rstrip("/") + "/chat/completions"
    return url


def _vlm_model() -> str:
    return os.getenv("AURA_MEMORY_VLM_MODEL", "").strip() or "Qwen/Qwen3-VL-8B-Instruct"


def _clean_text(text: str) -> str:
    """Strip control markers the model may store, so the memory reads as prose."""
    text = _TOOL_XML_RE.sub("", text or "")
    for token in (SILENT, _RESPOND, "<think>", "</think>"):
        text = text.replace(token, "")
    return text.strip()


def _encode_frame(frame) -> str:
    try:
        import cv2
        import numpy as np

        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return ""


def _round_boundaries(messages: list[dict[str, object]]) -> list[tuple[int, int]]:
    """Split ``messages`` into (start, end) rounds; ``end`` is exclusive.

    A round starts at a ``user`` row and runs through its ``assistant`` reply
    plus any ``tool_response`` rows, so a tool round is never cut in half.
    """
    bounds: list[tuple[int, int]] = []
    index = 0
    while index < len(messages):
        if messages[index].get("role") != "user":
            index += 1
            continue
        start = index
        index += 1
        while index < len(messages) and messages[index].get("role") != "user":
            index += 1
        bounds.append((start, index))
    return bounds


def _snapshot(history: SessionHistory, keep_rounds: int, max_frames: int) -> dict[str, object] | None:
    """Read-only view of the prefix that may be summarized, or None."""
    if history.pending_user is not None:
        return None  # a turn is in flight
    bounds = _round_boundaries(history.messages)
    num = len(bounds) - keep_rounds
    if num <= 0:
        return None

    frames: list = []
    qa: list[dict[str, str]] = []
    first_sec = last_sec = None
    fps = 2.0
    for start, end in bounds[:num]:
        user = history.messages[start]
        assistant_text = ""
        for row in history.messages[start + 1 : end]:
            if row.get("role") == "assistant" and isinstance(row.get("content"), str):
                assistant_text = row["content"]
        video = user.get("video")
        if isinstance(video, tuple) and len(video) == 2:
            array, metadata = video
            metadata = metadata if isinstance(metadata, dict) else {}
            fps = float(metadata.get("fps") or fps)
            indices = metadata.get("frames_indices") or []
            if indices:
                lo, hi = float(min(indices)), float(max(indices) + 1)
                video_start = float(user.get("video_start_seconds", 0.0))
                lo_sec, hi_sec = video_start + lo / fps, video_start + hi / fps
                first_sec = lo_sec if first_sec is None else min(first_sec, lo_sec)
                last_sec = hi_sec if last_sec is None else max(last_sec, hi_sec)
            try:
                frames.extend(list(array))
            except TypeError:
                pass
        user_text = _clean_text(str(user.get("content") or ""))
        reply = _clean_text(assistant_text)
        if reply == SILENT:
            reply = ""
        if user_text or reply:
            qa.append({"user": user_text, "assistant": reply})

    if first_sec is None:
        return None

    if len(frames) > max_frames:
        step = (len(frames) - 1) / max(max_frames - 1, 1)
        frames = [frames[round(i * step)] for i in range(max_frames)]

    return {
        "rounds": num,
        "anchor": history.messages[bounds[0][0]],
        "frames": frames,
        "qa": qa,
        "first_sec": first_sec,
        "last_sec": last_sec,
    }


def _summarize(snapshot: dict[str, object]) -> str:
    import requests

    frames = snapshot["frames"] or []
    qa = snapshot["qa"] or []
    lines = []
    for pair in qa:  # type: ignore[union-attr]
        if pair["user"]:
            lines.append(f"用户: {pair['user']}")
        if pair["assistant"]:
            lines.append(f"助手: {pair['assistant']}")
    transcript = "\n".join(lines) or "(这段时间没有说话)"

    content: list[dict[str, object]] = [
        {"type": "text", "text": f"对话记录:\n{transcript}\n\n{_MEMORY_SUMMARY_PROMPT}"}
    ]
    for frame in frames:  # type: ignore[union-attr]
        b64 = _encode_frame(frame)
        if b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    response = requests.post(
        _vlm_url(),
        json={
            "model": _vlm_model(),
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512,
            "temperature": 0.0,
        },
        timeout=_env_float("AURA_MEMORY_TIMEOUT_S", 30.0),
    )
    response.raise_for_status()
    payload = response.json()
    text = payload["choices"][0]["message"]["content"]
    return _clean_text(str(text))


def _commit(history: SessionHistory, snapshot: dict[str, object], summary: str) -> bool:
    """Evict the summarized prefix and append one memory turn. Caller holds the lock."""
    anchor = snapshot["anchor"]
    messages = history.messages
    # A turn that started while the VLM ran moved the prefix: drop the summary.
    if not messages or messages[0] is not anchor:
        return False
    if history.pending_user is not None:
        return False

    bounds = _round_boundaries(messages)
    num = int(snapshot["rounds"])  # type: ignore[arg-type]
    if num > len(bounds) - 0:
        num = min(num, len(bounds))
    if num <= 0:
        return False

    cut = bounds[num - 1][1]
    history.messages = messages[cut:]
    first_sec = float(snapshot["first_sec"])  # type: ignore[arg-type]
    last_sec = float(snapshot["last_sec"])  # type: ignore[arg-type]
    memory_text = summary if summary.startswith(MEMORY_PREFIX) else f"{MEMORY_PREFIX}{summary}"
    history.messages.insert(
        0,
        {
            "role": "user",
            "content": f"在{first_sec:.1f}-{last_sec:.1f}秒时刻，你记住了什么？",
        },
    )
    history.messages.insert(1, {"role": "assistant", "content": memory_text})
    logger.info(
        "[memory] committed %d rounds (%.1f-%.1f s); history now %d rows",
        num,
        first_sec,
        last_sec,
        len(history.messages),
    )
    return True


class _MemoryLoop:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"aura-memory-{session_id[:8]}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        interval = max(_env_float("AURA_MEMORY_INTERVAL_S", 30.0), 1.0)
        min_rounds = _env_int("AURA_MEMORY_MIN_ROUNDS", 6)
        keep_rounds = _env_int("AURA_MEMORY_KEEP_ROUNDS", 4)
        max_frames = _env_int("AURA_MEMORY_MAX_FRAMES", 8)
        logger.info(
            "[memory] loop started session=%s interval=%.0fs min_rounds=%d keep_rounds=%d max_frames=%d vlm=%s",
            self.session_id,
            interval,
            min_rounds,
            keep_rounds,
            max_frames,
            _vlm_url(),
        )
        while not self._stop.wait(interval):
            try:
                history = get_or_create_session_history(self.session_id)
                with HISTORY_LOCK:
                    if len(_round_boundaries(history.messages)) < min_rounds:
                        continue
                    snapshot = _snapshot(history, keep_rounds, max_frames)
                if snapshot is None:
                    continue
                logger.info(
                    "[memory] summarizing %d rounds, %d frames, %.1f-%.1f s session=%s",
                    snapshot["rounds"],
                    len(snapshot["frames"]),  # type: ignore[arg-type]
                    snapshot["first_sec"],  # type: ignore[arg-type]
                    snapshot["last_sec"],  # type: ignore[arg-type]
                    self.session_id,
                )
                try:
                    summary = _summarize(snapshot)
                except Exception as exc:  # network errors retry next interval
                    logger.warning("[memory] VLM call failed session=%s: %s", self.session_id, exc)
                    continue
                if not summary:
                    logger.warning("[memory] empty summary session=%s", self.session_id)
                    continue
                with HISTORY_LOCK:
                    _commit(history, snapshot, summary)
            except Exception:  # keep the thread alive across unexpected errors
                logger.warning("[memory] unexpected error session=%s\n%s", self.session_id, traceback.format_exc())


def note_turn(session_id: str | None) -> None:
    """Lazily start this session's memory loop (no-op unless enabled)."""
    if not session_id or not _env_flag("AURA_MEMORY_ENABLE"):
        return
    with _LOOPS_LOCK:
        if session_id in _LOOPS:
            return
        loop = _MemoryLoop(session_id)
        _LOOPS[session_id] = loop
    loop.start()


def stop_session(session_id: str) -> None:
    """Stop a session's memory loop; safe to call for unknown sessions."""
    with _LOOPS_LOCK:
        loop = _LOOPS.pop(session_id, None)
    if loop is not None:
        loop.stop()


__all__ = ["MEMORY_PREFIX", "note_turn", "stop_session"]
