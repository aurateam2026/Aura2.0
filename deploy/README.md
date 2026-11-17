# AURA Deployment Guide (AURA_v2 + vllm-omni duplex)

[English](README.md) · [简体中文](README.zh-CN.md)

Deploy a real-time audio/video conversation interface on a Linux server with accelerators and an AURA_v2 checkpoint. Run all commands below from the repository root. For container deployment, see the [Docker guide](docker/README.md).

```bash
bash deploy/aura.sh up --aura /path/to/aura_v2_checkpoint
```

The script prints the page URL when ready (default: `http://<server>:9310/`). Host installation requires neither root access nor Docker.

## What the launcher does

| Step | Action |
|---|---|
| 1. Check the host | Check Linux x86_64, a CUDA-compatible accelerator driver, and disk space. |
| 2. Install uv | Reuse an existing installation, or install to `deploy/runtime/bin` without changing the system or `~/.bashrc`. |
| 3. Use the source | Use the vllm-omni source included in this repository, one directory above `deploy/`; no upstream repository is cloned. |
| 4. Accelerator runtime compatibility | The pinned vLLM 0.33 build targets driver 580 or newer. For older drivers such as 535, download the vendor-provided `cuda-compat-13-0` forward-compatibility libraries into `deploy/runtime/cuda-compat`. These libraries affect only this service and require a supported accelerator. |
| 5. Python environment | Use uv to install Python 3.12, create `deploy/runtime/venv`, install vLLM 0.33.0 (cu130) and editable vllm-omni, and verify that PyTorch can use the accelerator. |
| 5b. FlashInfer kernels | The AURA model uses FlashInfer attention. When the accelerator runtime compiler (`nvcc`) is unavailable, install and verify precompiled kernels with `vllm download-kernels` (`flashinfer-cubin` + `flashinfer-jit-cache`). |
| 6. Download ASR/TTS | Download `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (about 9 GB), or use local paths supplied through `--asr` / `--tts`. |
| 7. Validate AURA weights | Check `config.json`, complete safetensors shards (a LoRA-only directory is rejected), and AURA_v2 control-token IDs such as `<\|silent\|>` / `<\|respond\|>`. |
| 8. Select accelerators and generate configuration | Use the included `aura_omni.yaml` (two accelerators) or `aura_omni_1gpu.yaml` (one accelerator), replacing model paths and accelerator IDs. |
| 9. Start services | Start `vllm-omni serve` on port 8099 (four stages plus warmup; about 5–8 minutes on the first run and about 4 minutes afterward) and `gateway.duplex_app` on port 9310. |
| 10. Run a smoke test | Simulate a browser sending camera frames, two spoken questions (the second usually triggers a tool call), and one typed question; check ASR, text responses, and speech output. |

Repeated runs skip completed installation steps. Environment reuse is determined by the source commit and vLLM version; completed model downloads are marked for reuse.

## Hardware requirements

- The verified two-accelerator demo uses 80 GB accelerators. Support for other accelerators depends on the runtime build and the checkpoint's quantization format; see [validation scope](../docs/DEPLOYMENT_NOTES.md#validation). On drivers older than 580, the installed compatibility libraries work only with accelerators supported by the vendor's forward-compatibility libraries.
- Accelerator memory:

| Layout | Accelerators | Reserved memory per accelerator, as a fraction of total memory |
|---|---|---|
| Two accelerators (recommended) | 2 | About 90% on the AURA accelerator and 38% on the ASR/TTS accelerator |
| One accelerator | 1 × 80 GB | About 88%, with all four stages on the same accelerator |

- Disk: about 25 GB for the environment, 9 GB for ASR/TTS, and 18 GB for AURA weights; checkpoint sizes vary.
- Network: access to GitHub, PyPI, `wheels.vllm.ai`, `flashinfer.ai`, Hugging Face, and the accelerator vendor's download servers. If needed, configure `export https_proxy=...` or `export HF_ENDPOINT=<mirror>` before installation.

## Common commands

```bash
bash deploy/aura.sh up --aura /path/to/ckpt                 # Install, start, and test
bash deploy/aura.sh up --aura /path/to/ckpt --gpus 0,1      # AURA on accelerator 0; ASR/TTS on accelerator 1
bash deploy/aura.sh up --aura /path/to/ckpt --gpus 3        # Single-accelerator layout on accelerator 3
bash deploy/aura.sh status                                  # Processes, ports, and accelerator memory
bash deploy/aura.sh test                                    # Run the smoke test again
bash deploy/aura.sh logs                                    # Follow logs
bash deploy/aura.sh stop                                    # Stop services
bash deploy/aura.sh restart                                 # Reuse the previous weights and accelerator
bash deploy/aura.sh start --aura /path/to/another_ckpt      # Switch weights without reinstalling
```

Without `--gpus`, the launcher selects two idle accelerators if available (less than 1 GB already used per accelerator), otherwise one idle accelerator. If none are idle, it reports each accelerator's usage and exits. It also rejects explicitly selected accelerators with insufficient free memory; use `--force` to override that check when intended.

## Options

| Option | Default | Description |
|---|---|---|
| `--aura PATH` | Required on the first run | AURA_v2 checkpoint directory |
| `--gpus auto\|I\|I,J` | `auto` | Accelerator selection as described above |
| `--home DIR` | `deploy/runtime` | Installation, logs, and runtime state directory |
| `--port N` | 9310 | Browser gateway port |
| `--backend-port N` | 8099 | vllm-omni realtime backend port; only local access is needed |
| `--name NAME` | Checkpoint directory name | Model name displayed in the interface |
| `--asr PATH` / `--tts PATH` | Download automatically | Use local ASR / TTS weights |
| `--env-file FILE` | `deploy/runtime/tool_keys.env` | Keys for gateway tools, such as `DEEPSEEK_API_KEY=sk-...`; do not commit this file |
| `--speaker NAME` | Vivian | Qwen3-TTS CustomVoice preset: Vivian, Serena, Uncle_Fu, Ryan, Aiden, Ono_Anna, Sohee, Eric (Sichuan dialect), or Dylan (Beijing dialect). Requires gateway support for `AURA_TTS_SPEAKER`. |
| `--no-test` | | Skip the smoke test after startup |
| `--force` | | Start even if the accelerator memory check fails |

Environment variables are also supported: `AURA_HOME`, `GPUS`, `GATEWAY_PORT`, `BACKEND_PORT`, `VLLM_WHEEL`, `ASR_REPO`, `TTS_REPO`, and `HF_HUB_DISABLE_XET` (default: 1, to avoid stalled Xet downloads).

## Start the memory summary model with the demo

Memory is disabled by default. Configure it in **`deploy/runtime/memory.env`**, relative to the repository root. This is separate from `tool_keys.env`. With `--home DIR`, use `DIR/memory.env`. The Docker setup uses `docker-runtime/memory.env` on the host.

For initial setup, run the following commands. If the file already exists, edit it directly instead of copying over it:

```bash
# Run from the repository root
mkdir -p deploy/runtime
cp deploy/memory.env.example deploy/runtime/memory.env
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=1/' deploy/runtime/memory.env
bash deploy/aura.sh restart
```

The example configuration puts memory on accelerator 2 while the demo uses accelerators 0 and 1. The launcher requires a separate accelerator for the managed memory service.

`AURA_MEMORY_VLM_URL=http://127.0.0.1:8003/v1/chat/completions` points to the local Qwen3-VL summary service. It receives older dialogue and sampled video frames and returns a summary; no OpenAI API key is needed. Keep the URL port consistent with `MEMORY_VLM_PORT`, retain `export` on configuration lines, and restart after changes.

To disable memory, change `AURA_MEMORY_ENABLE=1` to `AURA_MEMORY_ENABLE=0` and restart. For model paths, all settings, and verification steps, see the [Memory guide](../docs/MEMORY.md).

For the container environment, persistent directories, configuration mounts, and lifecycle commands, see the [Docker guide](docker/README.md).

## User context files

After startup through `aura.sh start`, `up`, or `restart`, each AURA inference request saves its context to `deploy/runtime/user_context/session_<session_id>/1.json`, `2.json`, and so on. A custom `--home` places `user_context` under that directory instead.

Each file contains `session_id`, `seq`, `payload`, `message`, and, once completed, `raw_response`. `payload.messages` contains messages supplied to the input template. `payload.prompt` contains the template text constructed after history trimming, memory summaries, and tool results have been applied, retaining empty thinking sections and video placeholders. Video pixels, audio, and long data URLs are replaced with shapes, types, and video metadata; tool definitions and text results are retained. `input` distinguishes speech, visual-only input, and continuation after tool execution, and records both the original ASR transcript and the actual user text. Responses retain control markers and tool-call XML.

Inputs are initially written as `pending` and updated to `completed` when the full model response arrives. Requests still awaiting a response when a session closes are marked `incomplete`. A `pending` file left after a crash means no complete response was recorded. A background thread pairs records by request ID and replaces files atomically. Restarting does not delete existing records.

Video placeholders in the template text have not yet been expanded by the multimodal processor. When provided by the backend, `raw_response.prompt_token_ids` is saved for further comparison of the expanded input.

Duplex video time accumulates within each session. Every new clip receives a fixed start offset; its duration remains the frame count divided by 2, with input FPS kept at 2. The preprocessing cache stores only times relative to the clip. The session offset is added when assembling model input, so cache hits for historical clips, repeated frames, tool continuations, and history compression do not reset the timeline. A new session starts from zero, including a reconnection that creates a new session.

`input.video_start_seconds` records each clip's start offset in video order. For two-frame clips, the actual timestamp tokens are `<0.2 seconds>`, `<1.2 seconds>`, `<2.2 seconds>`, and so on: the first frame-group center is 0.25 seconds, formatted to one decimal place. This behavior applies to the current AURA Qwen3.5 deployment with video pruning disabled; enabling video pruning at the same time is unsupported.

When launching the backend directly, set `AURA_USER_CONTEXT_DIR` to enable the same recording feature. It is disabled if that variable is unset.

## Open the interface

Browsers allow camera and microphone access only over `https://` or on `localhost`.

Click “允许权限” (Allow permissions) and grant camera and microphone access to enter the call directly. The interface uses 720p video, sends two frames every second, enables tool calls by default, uses the system prompt with `<|respond|>` / `<|silent|>` markers, and selects the server's default model automatically. Hanging up releases the camera and microphone; click “再次通话” (Call again) to reconnect.

**For personal access: an SSH tunnel**

```bash
ssh -N -L 9310:127.0.0.1:9310 your-user@your-server
```

Then open `http://localhost:9310/`.

**For shared access: an HTTPS reverse proxy.** For example, [Caddy](https://caddyserver.com/) can obtain certificates automatically. Point a domain to the server and open ports 80 and 443:

```
aura.example.com {
    reverse_proxy 127.0.0.1:9310
}
```

Calls use WebSocket at `/ws`, which Caddy supports by default. With Nginx, include `proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";`.

## Directory layout

```
deploy/
├── aura.sh              Launcher
├── lib/common.sh        Shared logging and process-management helpers
├── lib/check_ckpt.py    AURA checkpoint validation (Python standard library only)
├── lib/make_yaml.py     Deployment configuration generated from included templates
├── lib/smoke_test.py    End-to-end smoke test
└── assets/              Two 16 kHz Chinese speech samples for the smoke test

deploy/runtime/          Created at runtime; change the location with --home
├── venv/                Python environment
├── models/              ASR / TTS weights
├── cuda-compat/         CUDA 13 compatibility libraries (drivers older than 580 only)
├── deploy.yaml          Active deployment configuration
├── aura.env             Previous settings, read by restart / status
├── user_context/session_<session_id>/*.json  Model context and response for each request
├── logs/backend.log  logs/gateway.log
└── run/*.pid
```

## Troubleshooting

| Symptom | Action |
|---|---|
| `torch ... cannot use the accelerator` | The driver may be too old for an accelerator without forward-compatibility support, or the compatibility libraries may not apply. Upgrade the driver to 580 or newer. |
| `no idle accelerator` | Select accelerators with `--gpus`, or free accelerator memory first. |
| `not enough free accelerator memory` | Choose another accelerator or use the two-accelerator layout. Use `--force` only if you intend to override the check. |
| `backend exited during startup` | Read the end of `deploy/runtime/logs/backend.log`; common causes are insufficient accelerator memory or an incorrect checkpoint path. |
| `FlashInfer backend is not available` | Install missing precompiled kernels with `deploy/runtime/venv/bin/vllm download-kernels`, or install the accelerator runtime 13 toolchain (`nvcc` + `ninja`). |
| The smoke test fails but services remain running | Inspect `deploy/runtime/logs/gateway.log` and rerun `bash deploy/aura.sh test`. |
| Downloads are slow or stalled | Configure an `HF_ENDPOINT` mirror, or download weights manually and pass `--asr` / `--tts`. |
| A port is already in use | Choose another port with `--port` / `--backend-port`. |

On a SLURM cluster, run on a **accelerator compute node allocated to your job** and keep the job alive. Some clusters clean up child processes when `srun` exits; in that case, run inside the job's interactive shell (for example, in `tmux`) or submit through `sbatch`.

## Default components

| Component | Version |
|---|---|
| vllm-omni | Source included in this repository, derived from `AURA_production_v2_duplex` at commit `9182429` plus local changes |
| vLLM | 0.33.0 (cu130), PyTorch 2.14 |
| ASR | Qwen3-ASR-1.7B |
| TTS | Qwen3-TTS-12Hz-1.7B-CustomVoice; voice selected with `--speaker`. Project examples use Serena with Chinese speech. |
| Interface | Included `examples/online_serving/aura_v2_gateway` |

TTS Talker uses `min_tokens=2`. When a complete sentence is sent to Talker, the generation limit is the smaller of the configured `max_tokens` (default: 1024) and `50 + 15 × number of text characters`. This is an audio-token limit; generation can end earlier at the end-of-sequence marker. Voice selection continues to use `--speaker`.

## Third-party components and licenses

This repository contains deployment scripts and the vllm-omni source tree, **without model weights**. Deployment downloads weights from their official sources to the user's server.

| Component | Source | License | Distributed in this repository? |
|---|---|---|---|
| vllm-omni (AURA branch) | Included here; derived from github.com/vllm-project/vllm-omni | Apache-2.0 | Yes |
| vLLM, FlashInfer, PyTorch | wheels.vllm.ai / flashinfer.ai / PyPI | Apache-2.0 / BSD | No; installed during deployment |
| Qwen3-ASR-1.7B | huggingface.co/Qwen | Apache-2.0 | No; downloaded during deployment |
| Qwen3-TTS-12Hz-1.7B-CustomVoice, including the Vivian preset | huggingface.co/Qwen | Apache-2.0 | No; downloaded during deployment. The code references voice names only. |
| Accelerator runtime 13 forward-compatibility libraries (`cuda-compat-13-0`) | Driver vendor package repository | Vendor software license | No; downloaded on the user's machine only for drivers older than 580 |
| uv | astral.sh | Apache-2.0 / MIT | No; installed automatically if unavailable |
| AURA_v2 weights | Supplied by the user | Determined by the weight publisher | No |

If redistributing model weights in a repository or image, follow the license of those weights. For Apache-2.0 weights, include the license text and retain the original copyright notices.
