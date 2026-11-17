# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Verify persisted context at the real prompt/output hooks, without inference."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.engine.duplex.contracts import DuplexFence, duplex_ephemeral_stage_request_id
from vllm_omni.model_executor.models.aura_omni.duplex import context_dump
from vllm_omni.model_executor.models.aura_omni.duplex.data_plane import AuraDataPlaneSession
from vllm_omni.model_executor.models.aura_omni.duplex.history import drop_session_history, get_or_create_session_history
from vllm_omni.model_executor.stage_input_processors.aura_omni import asr2aura

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def dumper(tmp_path, monkeypatch):
    writer = context_dump.UserContextDumper(tmp_path)
    monkeypatch.setenv("AURA_USER_CONTEXT_DIR", str(tmp_path))
    monkeypatch.setenv("AURA_USER_CONTEXT_MODEL", "test-checkpoint")
    monkeypatch.delenv("AURA_MEMORY_VLM_URL", raising=False)
    monkeypatch.setattr(context_dump, "_dumper", writer)
    yield writer
    writer.shutdown()
    for session_id in ("alice", "bob"):
        drop_session_history(session_id)


def _request_id(session_id, turn=1, stage=0, epoch=0):
    return duplex_ephemeral_stage_request_id(DuplexFence(session_id, epoch, turn), stage_id=stage)


def _input(session_id, text, *, turn=1, epoch=0, video=None, tools=None, followup=False):
    source = SimpleNamespace(
        request_id=_request_id(session_id, turn, epoch=epoch),
        outputs=[SimpleNamespace(text=text, cumulative_token_ids=[1, 2], multimodal_output={})],
    )
    original = {
        "multi_modal_data": {"video": [video]} if video is not None else {},
        "additional_information": {
            "session_id": session_id,
            "aura_duplex": True,
            "aura_two_token": True,
            "is_speech": True,
            "aura_tool_followup": followup,
            "aura_tools": tools or [],
            "epoch": epoch,
            "turn_id": turn,
        },
    }
    return asr2aura([source], prompt=[original])[0]


def _output(plane, session_id, text, *, turn=1, epoch=0, finished=True):
    result = SimpleNamespace(
        request_id=_request_id(session_id, turn, stage=1, epoch=epoch),
        stage_id=1,
        finished=finished,
        prompt_token_ids=[10, 20, 30],
        outputs=[
            SimpleNamespace(
                text=text,
                cumulative_text=text,
                token_ids=[40, 50],
                finished=finished,
                finish_reason="stop" if finished else None,
            )
        ],
    )
    return list(plane.project_output(result))


def _read(dumper, session_id, seq=1):
    dumper.flush()
    return json.loads((dumper.directory / f"session_{session_id}" / f"{seq}.json").read_text())


def test_interleaved_sessions_and_epochs_keep_their_own_context(dumper):
    plane = AuraDataPlaneSession(lambda *args: None)
    _input("alice", "明天需要带伞吗？")
    _input("bob", "我在北京")
    _input("alice", "新的对话", epoch=1)
    _output(plane, "alice", "新回复", epoch=1)
    _output(plane, "bob", "北京的回复")
    _output(plane, "alice", "查一下天气", finished=False)
    assert _read(dumper, "alice")["status"] == "pending"
    _output(plane, "alice", "查一下天气。")
    # A duplicate final completion cannot overwrite an already saved response.
    _output(plane, "alice", "重复回调")
    alice = _read(dumper, "alice")
    assert alice["message"]["content"] == "查一下天气。"
    assert "明天需要带伞吗？" in alice["payload"]["prompt"]
    assert "北京" not in alice["payload"]["prompt"]
    assert alice["raw_response"]["prompt_token_ids"] == [10, 20, 30]
    assert _read(dumper, "bob")["message"]["content"] == "北京的回复"
    assert _read(dumper, "alice", 2)["message"]["content"] == "新回复"


def test_tool_followup_saves_actual_prompt_and_frozen_video_metadata(dumper):
    plane = AuraDataPlaneSession(lambda *args: None)
    metadata = {"fps": 1.0, "frames_indices": [0, 1], "total_num_frames": 2}
    video = (np.zeros((2, 8, 8, 3), dtype=np.uint8), metadata)
    tools = [{"type": "function", "function": {"name": "get_current_location", "description": "查询当前地点"}}]
    first_input = _input("alice", "明天需要带伞吗？", video=video, tools=tools)
    call = "<|respond|><tool_call>\n<function=get_current_location>\n</function>\n</tool_call>"
    _output(plane, "alice", call)
    history = get_or_create_session_history("alice")
    history.commit_turn(call)
    history.append_tool_response('{"city":"香港"}')
    followup = _input("alice", "", turn=2, tools=tools, followup=True)
    _output(plane, "alice", "<|respond|>正在查询香港的天气。", turn=2)
    metadata["frames_indices"].append(99)
    tools[0]["function"]["description"] = "changed"
    record = _read(dumper, "alice")
    assert record["payload"]["prompt"] == first_input["prompt"]
    assert record["message"]["content"] == call
    assert record["payload"]["tools"][0]["function"]["description"] == "查询当前地点"
    saved_video = record["payload"]["multi_modal_data"]["video"][0]
    assert saved_video[0] == {"media_omitted": True, "shape": [2, 8, 8, 3], "dtype": "uint8"}
    assert saved_video[1]["frames_indices"] == [0, 1]
    followup_record = _read(dumper, "alice", 2)
    assert followup_record["payload"]["prompt"] == followup["prompt"]
    assert "<think>\n\n</think>\n\n" in followup_record["payload"]["prompt"]
    assert "<|video_pad|>" in followup_record["payload"]["prompt"]
    messages = followup_record["payload"]["messages"]
    assert '<tool_response>\n{"city":"香港"}\n</tool_response>' in messages[-1]["content"]
    assert followup_record["input"]["tool_followup"] is True
    assert followup_record["payload"]["generation_prompt"].startswith("<|im_start|>assistant")


def test_silent_and_interrupted_turns_are_saved_and_numbering_survives_reopen(dumper):
    plane = AuraDataPlaneSession(lambda *args: None)
    _input("alice", "看看周围")
    _output(plane, "alice", "<|silent|>")
    assert _read(dumper, "alice")["message"]["content"] == "<|silent|>"
    _input("alice", "尚未回复", turn=2)
    plane.close_session("alice")
    assert _read(dumper, "alice", 2)["status"] == "incomplete"
    _input("alice", "重新连接", turn=3)
    _output(plane, "alice", "已重新连接", turn=3)
    assert _read(dumper, "alice", 3)["seq"] == 3
    assert _read(dumper, "alice")["message"]["content"] == "<|silent|>"


def test_logging_disabled_without_environment(monkeypatch):
    monkeypatch.delenv("AURA_USER_CONTEXT_DIR", raising=False)
    assert context_dump.get_context_dumper() is None


def test_unsafe_session_names_do_not_escape_directory_or_collide(dumper):
    for session_id in ("../alice", "alice", "/alice"):
        dumper.record_input(
            session_id=session_id,
            request_id=_request_id(session_id),
            prompt="<|im_start|>user\nhi<|im_end|>\n",
            tools=[],
            multi_modal_data={},
            input_info={},
        )
    dumper.flush()
    records = [json.loads(path.read_text()) for path in dumper.directory.glob("session_*/*.json")]
    assert {record["session_id"] for record in records} == {"../alice", "alice", "/alice"}


def test_file_write_failure_does_not_interrupt_model_output(dumper, monkeypatch, caplog):
    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(dumper, "_write", fail)
    _input("alice", "正常输入")
    events = _output(AuraDataPlaneSession(lambda *args: None), "alice", "正常回复")
    dumper.flush()
    assert any(event.get("model_context_text") == "正常回复" for event in events)
    assert "could not save context" in caplog.text
