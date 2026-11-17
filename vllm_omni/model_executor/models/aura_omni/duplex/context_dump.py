# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-session request/response snapshots, with media replaced by metadata.

The input hook runs after history pruning and prompt assembly. The output hook
matches by request id (including session, epoch and turn), never by whichever
turn happens to be current when a response arrives. A single worker owns file
I/O and numbering so serving does not wait for the filesystem.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import regex as re
from vllm.logger import init_logger

logger = init_logger(__name__)
_CHAT_MESSAGE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>\n", re.DOTALL)
_STAGE_TURN = re.compile(r"\.r\.stage\d+-turn(\d+)$")


def _sanitize(value: object) -> object:
    """Copy JSON structure without copying pixels or retaining live history."""
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        if value.startswith("data:") and len(value) > 120:
            return f"[data_url: {len(value)} chars]"
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes):
        return f"[bytes: {len(value)}]"
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"media_omitted": True, "shape": list(shape), "dtype": str(getattr(value, "dtype", ""))}
    return f"[{type(value).__name__}]"


def _request_key(request_id: str) -> str:
    # Stage0 and Stage1 may use different resource ids for the same turn.
    return _STAGE_TURN.sub(r".r.turn\1", request_id)


class UserContextDumper:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aura-context-dump")
        # Accessed only by the writer thread.
        self._sequences: dict[str, int] = {}
        self._pending: dict[str, tuple[Path, dict[str, object]]] = {}

    def _submit(self, action: Callable[..., None], *args: object) -> None:
        self._worker.submit(self._run, action, args)

    @staticmethod
    def _run(action: Callable[..., None], args: tuple[object, ...]) -> None:
        try:
            action(*args)
        except (OSError, TypeError, ValueError):
            logger.exception("[user_context] could not save context")

    @staticmethod
    def _write(path: Path, record: dict[str, object]) -> None:
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)

    def record_input(
        self,
        *,
        session_id: str,
        request_id: str,
        prompt: str,
        tools: object,
        multi_modal_data: object,
        input_info: Mapping[str, object],
    ) -> None:
        matches = list(_CHAT_MESSAGE.finditer(prompt))
        payload = {
            "model": os.getenv("AURA_USER_CONTEXT_MODEL", ""),
            # These are the rendered ChatML messages, including tool-response
            # wrappers and empty think blocks. `prompt` is authoritative.
            "messages": [{"role": match[1], "content": match[2]} for match in matches],
            "prompt": prompt,
            "generation_prompt": prompt[matches[-1].end() :] if matches else prompt,
            "tools": _sanitize(tools),
            "multi_modal_data": _sanitize(multi_modal_data),
        }
        record = {
            "session_id": session_id,
            "request_id": request_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
            "payload": payload,
            "input": _sanitize(input_info),
            "message": None,
        }
        self._submit(self._record_input, session_id, _request_key(request_id), record)

    def _record_input(self, session_id: str, key: str, record: dict[str, object]) -> None:
        if key in self._pending:
            return
        token = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id).strip("._-") or "unknown"
        if token != session_id:
            token += "_" + hashlib.sha256(session_id.encode()).hexdigest()[:16]
        directory = self.directory / f"session_{token}"
        directory.mkdir(parents=True, exist_ok=True)
        if session_id not in self._sequences:
            self._sequences[session_id] = max(
                (int(path.stem) for path in directory.glob("*.json") if path.stem.isdigit()), default=0
            )
        seq = self._sequences[session_id] + 1
        self._sequences[session_id] = seq
        record["seq"] = seq
        path = directory / f"{seq}.json"
        self._pending[key] = (path, record)
        self._write(path, record)

    def record_output(self, request_id: str, text: str, raw_response: Mapping[str, object]) -> None:
        self._submit(self._record_output, _request_key(request_id), text, _sanitize(raw_response))

    def _record_output(self, key: str, text: str, raw_response: object) -> None:
        pending = self._pending.pop(key, None)
        if pending is None:
            return
        path, record = pending
        record.update(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            message={"role": "assistant", "content": text},
            raw_response=raw_response,
        )
        self._write(path, record)

    def close_session(self, session_id: str) -> None:
        self._submit(self._close_session, session_id)

    def _close_session(self, session_id: str) -> None:
        for key, (path, record) in list(self._pending.items()):
            if record["session_id"] == session_id:
                self._pending.pop(key)
                record["status"] = "incomplete"
                self._write(path, record)
        self._sequences.pop(session_id, None)

    def flush(self) -> None:
        """Wait for already submitted writes (diagnostics/tests only)."""
        self._worker.submit(lambda: None).result()

    def shutdown(self) -> None:
        self._worker.shutdown(wait=True)


_dumper: UserContextDumper | None = None
_dumper_lock = Lock()


def get_context_dumper() -> UserContextDumper | None:
    """Deployment opts in with AURA_USER_CONTEXT_DIR; generic serving is unchanged."""
    global _dumper
    directory = os.getenv("AURA_USER_CONTEXT_DIR", "").strip()
    if not directory:
        return None
    with _dumper_lock:
        if _dumper is None:
            _dumper = UserContextDumper(Path(directory))
    return _dumper
