# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import requests

from vllm_omni.model_executor.models.aura_omni.duplex.memory import _summarize

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_memory_summary_sends_transcript_and_frame(monkeypatch):
    monkeypatch.setenv("AURA_MEMORY_VLM_URL", "http://memory.test/v1/chat/completions")
    monkeypatch.setenv("AURA_MEMORY_VLM_MODEL", "memory-test-model")
    calls = []

    def post(url, *, json, timeout):
        calls.append(json)
        assert url == "http://memory.test/v1/chat/completions"
        assert json["model"] == "memory-test-model"
        content = json["messages"][0]["content"]
        assert "用户: 请记住红色盒子" in content[0]["text"]
        assert "助手: 收到" in content[0]["text"]
        assert "总结" in content[0]["text"]
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert timeout > 0
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "<|respond|>我看到了红色盒子"}}]},
        )

    monkeypatch.setattr(requests, "post", post)
    summary = _summarize(
        {
            "qa": [{"user": "请记住红色盒子", "assistant": "收到"}],
            "frames": [np.zeros((8, 8, 3), dtype=np.uint8)],
        }
    )
    assert summary == "我看到了红色盒子"
    assert len(calls) == 1


def test_memory_compression_preserves_session_video_timeline():
    from vllm_omni.model_executor.models.aura_omni.duplex.history import SessionHistory
    from vllm_omni.model_executor.models.aura_omni.duplex.memory import _commit, _snapshot

    history = SessionHistory("memory-clock", next_video_seconds=10.0)
    clip = (np.zeros((2, 8, 8, 3), dtype=np.uint8), {"fps": 2.0, "frames_indices": [0, 1], "duration": 1.0})
    for i in range(3):
        history.begin_user_turn("look", clip, video_duration=1.0, turn_key=str(i))
        history.commit_turn("answer")
    snapshot = _snapshot(history, keep_rounds=1, max_frames=8)
    assert snapshot is not None
    assert (snapshot["first_sec"], snapshot["last_sec"]) == (10.0, 12.0)
    assert _commit(history, snapshot, "saved scene")
    assert "10.0-12.0秒" in history.messages[0]["content"]
    assert history.retained_video_starts() == [12.0]
    history.begin_user_turn("look", clip, video_duration=1.0, turn_key="next")
    assert history.pending_video_start_seconds == 13.0
