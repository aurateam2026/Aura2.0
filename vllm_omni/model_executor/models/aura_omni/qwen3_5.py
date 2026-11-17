# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AURA-compatible Qwen3.5-VL wrapper (AURA_v2 / qwen3_5 checkpoints).

Mirrors ``aura_omni/qwen3_vl.py``: keep stock vLLM Qwen3.5 multimodal logic while
accepting structurally compatible remote configs. Stage1 → TTS only needs finished
text / token ids, so full-prefix hidden D2H is left off.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from transformers import Qwen3_5Config
from transformers.processing_utils import ProcessorMixin
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import ProcessorInputs, PromptUpdateDetails, TimingContext
from vllm.multimodal.processing.processor import MultiModalProcessingResult, MultiModalPromptUpdates

from vllm_omni.model_executor.models.aura_omni.video_timeline import VIDEO_START_SECONDS_KEY


@dataclass
class _VideoPromptDetails(PromptUpdateDetails):
    # Cache full precision relative times, not the rounded text or a session's
    # offset. This also works with IPC caches which only retain prompt updates.
    relative_timestamps: tuple[float, ...] = ()


class AuraQwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_hf_config(self) -> Any:
        try:
            return self.ctx.get_hf_config(Qwen3_5Config)
        except TypeError:
            config = self.ctx.get_hf_config()
            if (
                getattr(config, "model_type", None) == "qwen3_5"
                and hasattr(config, "vision_config")
                and hasattr(config, "text_config")
            ):
                return config
            raise

    def get_hf_processor(self, **kwargs: object) -> Any:
        # Qwen3.5 multimodal still uses the Qwen3-VL processor class in vLLM.
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


class AuraQwen3VLMultiModalProcessor(Qwen3VLMultiModalProcessor):
    @staticmethod
    def _expands_only_video_token(hf_processor: ProcessorMixin) -> bool:
        # AURA's current and historical turns use the full wrapped placeholder.
        # vLLM's video replacement already supplies per-frame vision markers,
        # so consume the outer pair too, as in the native AURA demo. Apply this
        # to both HF processing and cached prompt updates, regardless of the
        # installed Transformers processor's replacement convention.
        return False

    def _get_mm_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> MultiModalPromptUpdates:
        updates = super()._get_mm_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        for index, group in enumerate(updates.get("video", [])):
            timestamps = tuple(float(t) for t in out_mm_kwargs["video"][index]["timestamps"].data)
            updates["video"][index] = [
                replace(
                    update,
                    content=_VideoPromptDetails(
                        full=update.content.full,
                        is_embed=update.content.is_embed,
                        relative_timestamps=timestamps,
                    ),
                )
                for update in group
            ]
        return updates

    def _cached_apply_hf_processor(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalProcessingResult:
        kwargs = dict(inputs.hf_processor_mm_kwargs)
        starts = kwargs.pop(VIDEO_START_SECONDS_KEY, None)
        if starts is None:
            return super()._cached_apply_hf_processor(inputs, timing_ctx)
        count = inputs.mm_data_items.get_count("video", strict=False)
        if (
            not isinstance(starts, (list, tuple))
            or len(starts) != count
            or any(
                isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in starts
            )
        ):
            raise ValueError("AURA requires one finite, non-negative start time per video")
        if self.info.ctx.get_mm_config().get_video_pruning_spec() is not None:
            # Pruned embeddings include timestamp text embeddings. Keeping their
            # encoder cache valid requires a separate integration, not just a
            # prompt update. AURA's current deployment does not use pruning.
            raise ValueError("AURA session video timestamps do not support video pruning")

        # Session positions must not enter HF kwargs or the media cache key.
        # Cache only context-independent pixels, grids and relative timestamps.
        result = super()._cached_apply_hf_processor(replace(inputs, hf_processor_mm_kwargs=kwargs), timing_ctx)
        config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()
        video_updates = []
        for start, group in zip(starts, result.prompt_updates.get("video", [])):
            shifted_group = []
            for update in group:
                details = update.content
                if not isinstance(details, _VideoPromptDetails):
                    raise TypeError("AURA video cache entry has no relative timestamps")
                full: list[int] = []
                cursor = 0
                for relative in details.relative_timestamps:
                    begin = details.full.index(config.vision_start_token_id, cursor)
                    end = details.full.index(config.vision_end_token_id, begin) + 1
                    full.extend(tokenizer.encode(f"<{start + relative:.1f} seconds>", add_special_tokens=False))
                    full.extend(details.full[begin:end])
                    cursor = end
                # Never edit cached details in place: identical pixels can
                # occur at different times, even within the same request.
                shifted_group.append(replace(update, content=replace(details, full=full)))
            video_updates.append(shifted_group)
        updates = {**result.prompt_updates, "video": video_updates}
        # Non-pruned visual embeddings depend only on pixels/grid. Their cached
        # metadata stays relative; the language model receives time via the
        # request-local text tokens and freshly calculated placeholder ranges.
        return result._replace(prompt_updates=updates)


@MULTIMODAL_REGISTRY.register_processor(
    AuraQwen3VLMultiModalProcessor,
    info=AuraQwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AuraQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    # Stage1 AURA → TTS only needs finished text / token ids.
    requires_full_prefix_cached_hidden_states = False
    # Qwen3.5 is hybrid (full attention + GDN), so the Omni stage-output cache
    # refuses it. Native KV + aligned recurrent-state caching still serves the
    # shared history prefix across turns.
    uses_native_kv_cache_only = True
