# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native AURA video expansion, including retained history and cached clips."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from transformers import BatchFeature, Qwen3VLProcessor
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor, _replace_video_token_placeholders

from vllm_omni.model_executor.models.aura_omni import qwen3_5 as aura_qwen3_5

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _clip(num_frames: int = 2) -> dict:
    return {
        "duration": num_frames / 2.0,
        "fps": 2.0,
        "frames_indices": list(range(num_frames)),
        "total_num_frames": num_frames,
    }


@pytest.fixture
def processing_info(monkeypatch: pytest.MonkeyPatch):
    info = aura_qwen3_5.AuraQwen3_5ProcessingInfo.__new__(aura_qwen3_5.AuraQwen3_5ProcessingInfo)
    monkeypatch.setattr(info, "get_video_processor", lambda: SimpleNamespace(temporal_patch_size=2))
    return info


def test_processor_keeps_clip_times_relative(processing_info) -> None:
    assert processing_info._get_video_second_idx(_clip(2)) == [0.25]
    assert processing_info._get_video_second_idx(_clip(4)) == [0.25, 1.25]
    assert processing_info._get_video_second_idx(_clip(2)) == [0.25]


def test_aura_registers_the_video_processor() -> None:
    assert issubclass(aura_qwen3_5.AuraQwen3VLMultiModalProcessor, Qwen3VLMultiModalProcessor)


@pytest.fixture
def processor(processing_info, monkeypatch: pytest.MonkeyPatch):
    hf_processor = Qwen3VLProcessor.__new__(Qwen3VLProcessor)
    hf_processor.image_token_id = 13
    config = SimpleNamespace(
        vision_start_token_id=10,
        video_token_id=11,
        vision_end_token_id=12,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
    monkeypatch.setattr(processing_info, "get_hf_processor", lambda **kwargs: hf_processor)
    monkeypatch.setattr(processing_info, "get_hf_config", lambda: config)
    monkeypatch.setattr(processing_info, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(processing_info, "get_image_processor", lambda **kwargs: SimpleNamespace(merge_size=2))
    monkeypatch.setattr(processing_info, "_merge_and_resolve_mm_processor_kwargs", lambda kwargs: {})
    processing_info.ctx = SimpleNamespace(
        get_mm_config=lambda: SimpleNamespace(get_video_pruning_spec=lambda: None),
        call_hf_processor=Mock(),
    )
    proc = aura_qwen3_5.AuraQwen3VLMultiModalProcessor.__new__(aura_qwen3_5.AuraQwen3VLMultiModalProcessor)
    proc.info = processing_info
    monkeypatch.setattr(proc, "_get_per_video_kwargs", lambda *args: {})
    return proc


@pytest.mark.parametrize("clip_count", [1, 4])
def test_uncached_video_expansion_has_one_pair_of_markers(processor, monkeypatch, clip_count) -> None:
    videos = [(np.zeros((2, 32, 32, 3), dtype=np.uint8), _clip()) for _ in range(clip_count)]
    monkeypatch.setattr(processor, "_get_hf_mm_inputs", lambda *args: ({"videos": videos}, {}, {}))
    monkeypatch.setattr(processor, "_finalize_hf_mm_data", lambda data, kwargs, passthrough, output: output)

    # Stub pixel processing only; execute vLLM's real timestamp calculation
    # and token expansion using the same wrapped placeholder as AURA history.
    processor.info.ctx.call_hf_processor.side_effect = [
        BatchFeature({"video_grid_thw": torch.tensor([[1, 2, 2]]), "pixel_values_videos": torch.zeros(4, 3)})
        for _ in videos
    ] + [BatchFeature({"input_ids": [[10, 11, 12] * clip_count]})]
    result = processor._apply_hf_processor_main(None, {})
    expected = []
    for _ in range(clip_count):
        expected.extend(list(b"<0.2 seconds>") + [10, 11, 12])
    assert result["input_ids"] == [expected]
    assert result["timestamps"] == [[0.25]] * clip_count


@pytest.mark.parametrize("clip_count", [1, 4])
def test_cached_video_replacement_consumes_outer_markers(processor, clip_count) -> None:
    out_mm_kwargs = {
        "video": [
            {
                "video_grid_thw": SimpleNamespace(data=torch.tensor([1, 2, 2])),
                "timestamps": SimpleNamespace(data=[0.25]),
            }
            for _ in range(clip_count)
        ]
    }
    updates = processor._get_prompt_updates(None, {}, out_mm_kwargs)
    video_update = next(update for update in updates if update.modality == "video")
    replacements = [list(video_update.replacement(i).full) for i in range(clip_count)]
    expanded = _replace_video_token_placeholders([10, 11, 12] * clip_count, video_update.target, replacements)
    assert expanded == (list(b"<0.2 seconds>") + [10, 11, 12]) * clip_count
    # The fix must preserve the image replacement contract.
    image_update = next(update for update in updates if update.modality == "image")
    assert image_update.target == [13]


@pytest.fixture
def cached_processor(processor, monkeypatch):
    from vllm.config import ModelConfig
    from vllm.config.multimodal import MultiModalConfig
    from vllm.multimodal.parse import MultiModalDataItems, VideoProcessorItems

    mm_config = MultiModalConfig(limit_per_prompt={"video": 8}, mm_processor_cache_gb=0.01)
    model_config = Mock(spec=ModelConfig, model="aura-timestamp-test")
    model_config.get_multimodal_config.return_value = mm_config
    processor.info.ctx.model_config = model_config
    processor.info.ctx.get_mm_config = lambda: mm_config

    def parse(data, **kwargs):
        return (
            MultiModalDataItems({"video": VideoProcessorItems(data["video"])})
            if data.get("video")
            else MultiModalDataItems({})
        )

    monkeypatch.setattr(processor.info, "parse_mm_data", parse)
    monkeypatch.setattr(
        processor,
        "_get_hf_mm_inputs",
        lambda items, kwargs: (
            {"videos": items["video"].get_all(), "text": [10, 11, 12] * len(items["video"])},
            kwargs,
            {},
        )
        if items.get("video")
        else ({}, kwargs, {}),
    )
    processed = []

    def call_hf(hf, data, kwargs):
        if "videos" in data:
            video = data["videos"][0][0]
            processed.append(int(video[0, 0, 0, 0]))
            frames = len(video) // 2
            return BatchFeature(
                {"video_grid_thw": torch.tensor([[frames, 2, 2]]), "pixel_values_videos": torch.zeros(frames * 4, 3)}
            )
        return BatchFeature({"input_ids": [data.get("text", [])]})

    processor.info.ctx.call_hf_processor.side_effect = call_hf
    return processor, processed


def _process_clips(processor, clips, starts, cache=None):
    from vllm.multimodal.processing import ProcessorInputs, TimingContext

    items = processor.info.parse_mm_data({"video": clips})
    return processor.apply(
        ProcessorInputs(
            prompt=[10, 11, 12] * len(clips),
            mm_data_items=items,
            hf_processor_mm_kwargs={"aura_video_start_seconds": starts},
            cache=cache,
        ),
        TimingContext(enabled=False),
    )


def _video(value, frames=2):
    return np.full((frames, 32, 32, 3), value, dtype=np.uint8), _clip(frames)


@pytest.mark.parametrize("cache_type", ["processor", "ipc"])
def test_timeline_matches_with_partial_and_full_cache_hits(cached_processor, cache_type):
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache
    from vllm.multimodal.cache.lru import LruKeyReplicatedSenderCache

    processor, processed = cached_processor
    a, b = _video(1), _video(2)
    cache_cls = MultiModalProcessorOnlyCache if cache_type == "processor" else LruKeyReplicatedSenderCache
    cache = cache_cls(processor.info.ctx.model_config)
    _process_clips(processor, [a], [0.0], cache)
    processed.clear()
    warm = _process_clips(processor, [a, b], [0.0, 1.0], cache)
    assert processed == [2], "A must reuse its actual preprocessing cache"
    processed.clear()
    full_hit = _process_clips(processor, [a, b], [0.0, 1.0], cache)
    assert processed == []
    cold = _process_clips(processor, [a, b], [0.0, 1.0])
    assert warm["prompt_token_ids"] == full_hit["prompt_token_ids"] == cold["prompt_token_ids"]
    assert warm["prompt_token_ids"] == list(b"<0.2 seconds>") + [10, 11, 12] + list(b"<1.2 seconds>") + [10, 11, 12]


def test_identical_pixels_reuse_cache_at_different_session_times(cached_processor):
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache

    processor, processed = cached_processor
    cache = MultiModalProcessorOnlyCache(processor.info.ctx.model_config)
    a = _video(1)
    first = _process_clips(processor, [a], [0.0], cache)
    processed.clear()
    later = _process_clips(processor, [a, a], [8.0, 9.0], cache)
    assert processed == []
    assert later["prompt_token_ids"] == list(b"<8.2 seconds>") + [10, 11, 12] + list(b"<9.2 seconds>") + [10, 11, 12]
    # Another session starts at zero, even though it shares the media cache.
    again = _process_clips(processor, [a], [0.0], cache)
    assert again["prompt_token_ids"] == first["prompt_token_ids"]
    assert again["mm_hashes"]["video"][0] == later["mm_hashes"]["video"][0]
    assert again["mm_kwargs"]["video"][0]["timestamps"].data == [0.25]


def test_timeline_keeps_precision_and_multiple_frame_groups(cached_processor):
    processor, _ = cached_processor
    a = _video(1, frames=4)
    result = _process_clips(processor, [a], [0.5])
    assert result["prompt_token_ids"] == list(b"<0.8 seconds>") + [10, 11, 12] + list(b"<1.8 seconds>") + [10, 11, 12]
    later = _process_clips(processor, [a], [10000.0])
    assert later["prompt_token_ids"] == list(b"<10000.2 seconds>") + [10, 11, 12] + list(b"<10001.2 seconds>") + [
        10,
        11,
        12,
    ]
    assert later["mm_placeholders"]["video"][0].get_num_embeds() == 2


@pytest.mark.parametrize("starts", [[], [1.0, 2.0], [float("nan")], [-1.0], [True]])
def test_timeline_rejects_invalid_positions(cached_processor, starts):
    processor, _ = cached_processor
    with pytest.raises(ValueError, match="start time"):
        _process_clips(processor, [_video(1)], starts)


def test_timeline_rejects_pruned_embeddings(cached_processor, monkeypatch):
    processor, processed = cached_processor
    monkeypatch.setattr(type(processor.info.ctx.get_mm_config()), "get_video_pruning_spec", lambda self: ("evs", 0.5))
    with pytest.raises(ValueError, match="video pruning"):
        _process_clips(processor, [_video(1)], [0.0])
    assert processed == []
