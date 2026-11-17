#!/usr/bin/env python3
"""Check that a directory is a servable AURA_v2 checkpoint before anything starts.

Exit 0 when usable (warnings allowed), 1 on a hard error. Standard library only.

  python check_ckpt.py /path/to/checkpoint
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ids the duplex pipeline relies on (vllm_omni .../aura_omni/control_tokens.py AURA_V2).
EXPECTED_IDS = {
    248070: "<|silent|>",
    248046: "<|im_end|>",
    248045: "<|im_start|>",
    74455: "assistant",
    248077: "<|respond|>",
}

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_ckpt.py <checkpoint dir>", file=sys.stderr)
        return 2
    ckpt = Path(sys.argv[1])
    errors: list[str] = []
    warnings: list[str] = []

    if not ckpt.is_dir():
        print(f"error: not a directory: {ckpt}")
        return 1

    config_path = ckpt / "config.json"
    if not config_path.is_file():
        errors.append("config.json is missing")
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_type = config.get("model_type")
        archs = config.get("architectures") or []
        if model_type != "qwen3_5":
            warnings.append(f"model_type is {model_type!r}, expected 'qwen3_5' (AURA_v2)")
        if "vision_config" not in config:
            errors.append("config.json has no vision_config: AURA needs the vision tower")
        print(f"  model_type={model_type} architectures={archs}")

    shards = sorted(ckpt.glob("*.safetensors"))
    if not shards:
        if (ckpt / "adapter_config.json").is_file():
            errors.append("only a LoRA adapter here; merge it into full weights first")
        else:
            errors.append("no *.safetensors weights found")
    else:
        size_gb = sum(p.stat().st_size for p in shards) / 1e9
        print(f"  weights: {len(shards)} safetensors file(s), {size_gb:.1f} GB")
        index = ckpt / "model.safetensors.index.json"
        if index.is_file():
            needed = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
            missing = sorted(name for name in needed if not (ckpt / name).is_file())
            if missing:
                errors.append(f"index lists missing shards: {missing[:3]}")

    tok_path = ckpt / "tokenizer.json"
    if not tok_path.is_file():
        errors.append("tokenizer.json is missing")
    else:
        tok = json.loads(tok_path.read_text(encoding="utf-8"))
        added = {item["id"]: item["content"] for item in tok.get("added_tokens", [])}
        vocab = {v: k for k, v in (tok.get("model", {}).get("vocab") or {}).items()}
        for tid, want in EXPECTED_IDS.items():
            got = added.get(tid, vocab.get(tid))
            if got != want:
                errors.append(f"token id {tid} is {got!r}, expected {want!r} (not an AURA_v2 tokenizer?)")
        if not errors:
            print("  control tokens: <|silent|>/<|respond|>/<|im_start|>/<|im_end|> ids match AURA_v2")

    for name in ("tokenizer_config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        if not (ckpt / name).is_file():
            warnings.append(f"{name} is missing (copy it from the base AURA_v2 / Qwen3.5 model if loading fails)")

    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"error: {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
