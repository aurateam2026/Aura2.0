# Third-Party Notices

This repository contains, and builds on, the following third-party software
and model assets.

## vLLM-Omni

- Source: https://github.com/vllm-project/vllm-omni
- License: Apache License 2.0 (see `LICENSE`)
- The tree in this repository is derived from the `vllm-project/vllm-omni`
  project, with AURA-specific changes maintained on the `AURA_production_v2_duplex`
  line of `https://github.com/NumberWan/vllm-omni`, plus the AURA changes
  distributed in this source tree.

## vLLM

- Source: https://github.com/vllm-project/vllm
- License: Apache License 2.0
- Not vendored here. Installed at deployment time as a prebuilt wheel
  (`vllm 0.33.0`, accelerator runtime 13 build).

## Qwen3-ASR, Qwen3-TTS, Qwen3-VL

- Source: https://huggingface.co/Qwen
- Licenses: see the model cards on Hugging Face (Apache License 2.0 unless
  stated otherwise on the model page).
- Not vendored here. Downloaded at deployment time by `deploy/aura.sh`.

## AURA model weights

- Not included in this repository. You must provide a checkpoint directory
  that you are licensed to use.

## Other bundled Python packages

Runtime dependencies are installed by `uv`/`pip` from PyPI and
`wheels.vllm.ai`; their licenses are declared in their own distributions.
