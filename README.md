<img src="docs/assets/aura-2-logo.png" width="200" alt="AURA 2.0 logo" align="left">

<h1>Beyond Answering: Learning to Speak First and Act in Streaming Video</h1>

<a href="https://aurateam2026.github.io/Aura-2.0/"><img alt="Home Page" src="https://img.shields.io/badge/Home-Page-0A66C2?logo=googlechrome&logoColor=white" height="22"></a>
<a href="https://huggingface.co/aurateam/AURA_2"><img alt="Model: Aura2.0" src="https://img.shields.io/badge/Model-Aura2.0-FFD21E?logo=huggingface&amp;logoColor=black" height="22"></a>
<a href="https://huggingface.co/aurateam/AURA_2_BF8"><img alt="Model: Aura2.0 BF8" src="https://img.shields.io/badge/Model-Aura2.0_BF8-FFD21E?logo=huggingface&amp;logoColor=black" height="22"></a>
<a href="deploy/docker/README.md"><img alt="Docker deployment" src="https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker&logoColor=white" height="22"></a>
<a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/License-Apache_2.0-blue" height="22"></a>
<a href="https://arxiv.org/pdf/2611.17048"><img alt="Technical Report: arXiv" src="https://img.shields.io/badge/Technical_Report-arXiv-EC1C24?logo=arxiv&logoColor=white" height="22"></a>
<a href="https://aura.polarpanda.top"><img alt="Online Demo" src="https://img.shields.io/badge/Online%20Demo-007F73?logo=googlechrome&amp;logoColor=white" height="22"></a>

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

AURA 2.0 receives a continuous camera stream, decides when to respond, and answers through streaming speech. This repository includes the AURA duplex backend, browser interface, tool executor, and deployment scripts, together with the vLLM-Omni source they run on.

## Demo Deployment and Usage

### Requirements

| Component | Requirement / validation status |
| --- | --- |
| OS | Linux x86_64; development environment uses Ubuntu 22.04 |
| Verified accelerator layout | 2 × 80 GB accelerators; AURA on one, ASR and TTS on the other |
| Single-accelerator layout | Configuration provided for an 80 GB accelerator, approximately 70 GB free; see validation notes below |
| Runtime | Accelerator drivers and runtime must be compatible with the pinned dependencies in this repository |
| Disk | Plan for at least 50 GB free, including dependencies and model weights; caches and conversation dumps grow over time |
| Software | `bash`, `git`, `curl`, and `python3`; the script installs uv and Python 3.12 and checks accelerator availability |
| Network | Access to package/model providers for installation; network tools need internet access at runtime |
| Browser | A modern browser with camera/microphone access, using **HTTPS or localhost** |

The deployment uses a pinned vLLM 0.33 wheel and the bundled vLLM-Omni tree. Keep this combination when reproducing the demo. Validation is limited to the configuration above; compatibility and memory requirements for other accelerators need separate verification.

The defaults reserve approximately 90% of the AURA accelerator and 38% of the ASR/TTS accelerator in the two-accelerator layout, or 88% in the single-accelerator layout. Switching to quantized weights does not automatically reduce those reservations. See [deployment notes](docs/DEPLOYMENT_NOTES.md) for limits and verification scope.

## Models

| Model | How it is obtained |
| --- | --- |
| [AURA 2.0](https://huggingface.co/aurateam/AURA_2) | Download the checkpoint and pass its local directory through `--aura` |
| [AURA 2.0 BF8](https://huggingface.co/aurateam/AURA_2_BF8) | Download the checkpoint and pass its local directory through `--aura` |
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Automatically downloaded, or set `--asr /path/to/model` |
| [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | Automatically downloaded, or set `--tts /path/to/model` |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Optional memory summarizer; disabled by default |

## Quick start

Run these commands on the accelerator server. For an environment packaged in a container, use [Docker deployment](#docker-deployment).

### 1. Clone the repository

```bash
git clone https://github.com/aurateam2026/Aura2.0.git
cd Aura2.0
```

### 2. Select and check your checkpoint

Replace the example path with the directory containing your AURA 2.0 weights. The BF8 checkpoint uses the same command.

```bash
export AURA_CKPT=/absolute/path/to/aura-2-checkpoint
python3 deploy/lib/check_ckpt.py "$AURA_CKPT"
```

### 3. Start the demo

```bash
# Accelerator 0: AURA; accelerator 1: ASR + TTS. Explicitly select the Serena voice.
bash deploy/aura.sh up --aura "$AURA_CKPT" --gpus 0,1 --speaker Serena
```

The first run installs the Python environment and precompiled kernels, downloads ASR/TTS models, generates a deployment configuration, starts all four backend stages and the gateway, and runs a smoke test. Downloads and initial compilation can take several minutes. All runtime files are placed under `deploy/runtime/` by default.

For the provided single-accelerator configuration:

```bash
bash deploy/aura.sh up --aura "$AURA_CKPT" --gpus 0 --speaker Serena
```

Run one layout at a time. To change an existing deployment, stop it first and then run `up` with the new options. Check accelerator availability yourself when sharing a server; `--gpus` selects physical accelerator indices.

### 4. Open the browser

If the browser is on the accelerator server, open **<http://localhost:9310/>**.

For a browser on your laptop, run this command **on the laptop** to connect directly to the accelerator server running AURA:

```bash
ssh -N -L 9310:127.0.0.1:9310 your-user@your-server
```

Replace `your-user` with your SSH username and `your-server` with the accelerator server's IP address or hostname. Keep `127.0.0.1` unchanged: it refers to the accelerator server itself, where SSH connects to AURA on port `9310`. This assumes SSH connects to the server running AURA; if your cluster requires a login node, use it as an SSH jump host to reach the accelerator server.

Keep the SSH session running, then open **<http://localhost:9310/>** on the laptop. Here, `localhost` refers to your laptop; the SSH tunnel forwards its port `9310` to AURA on the accelerator server.

For a phone or public access, configure an HTTPS reverse proxy with WebSocket support and open its **HTTPS domain or server address**; see [external access](docs/DEPLOYMENT_NOTES.md#external-access). Plain HTTP on a remote server IP does not provide the secure context needed for camera/microphone access.

### 5. Talk to AURA

1. Click **Allow access** and grant camera/microphone access. The call starts automatically.
2. Once connected, hold **Hold to talk**, speak, and release to send. You can also type a question or draw on the frame.
3. Wait for the first video packet before sending a typed question.
4. Click **Hang up** to end the call and release the camera/microphone; **Call again** starts another call.

## Docker deployment

Build the environment from this repository; a prebuilt public AURA 2.0 image has not been published. The image contains Python 3.12, the pinned vLLM 0.33 wheel, the bundled vLLM-Omni code, and gateway dependencies. Model weights are mounted or downloaded separately.

Install Docker Engine and configure container access to the accelerators on the host. Then, from the repository root:

```bash
docker build -f deploy/docker/Dockerfile -t aura2:local .
export AURA_CKPT=/absolute/path/to/aura-2-checkpoint
mkdir -p docker-runtime

docker run -d --name aura2 --gpus all --shm-size=16g --stop-timeout=180 \
  -p 127.0.0.1:9310:9310 \
  --mount "type=bind,src=$AURA_CKPT,dst=/models/aura,readonly" \
  --mount "type=bind,src=$PWD/docker-runtime,dst=/var/lib/aura" \
  aura2:local --aura /models/aura --gpus 0,1 --speaker Serena

docker logs -f aura2
```

Docker's `--gpus all` makes accelerators visible; the AURA `--gpus 0,1` selects the two accelerators used by the demo. First startup downloads ASR/TTS weights and any missing kernels. The runtime directory persists models, caches, logs, and `memory.env`; Python dependencies remain in the image. Open `http://localhost:9310/`, using an SSH tunnel from a laptop as described above.

See the [complete Docker guide](deploy/docker/README.md) for host setup, environment contents, accelerator selection, offline model mounts, memory, restart/stop commands, and build validation status.

## Optional network tools

Local camera understanding, speech recognition, and speech generation do not require a DeepSeek key. The gateway can also use these tools:

| Tool | Credential / behavior |
| --- | --- |
| Current time | Public time service; falls back to the server clock; Asia/Shanghai timezone |
| Approximate location | Public IP geolocation of the **server's outbound IP**, not phone GPS |
| Weather | Open-Meteo; no key configured by this demo |
| DeepSeek text and image tools | `DEEPSEEK_API_KEY` and access to the requested provider models/features |

The model still sees the DeepSeek tool schemas without a key; calling those tools then returns a configuration error. Some provider model IDs/features may not be available to every account.

To configure the optional tools:

```bash
mkdir -p deploy/runtime
cp tools/tool_keys.env.example deploy/runtime/tool_keys.env
chmod 600 deploy/runtime/tool_keys.env
# Edit the file locally and fill DEEPSEEK_API_KEY; use NAME=value, without quotes or export.
bash deploy/aura.sh restart
```

The default file is loaded automatically. For a different location, pass `--env-file /absolute/path/to/tool_keys.env`. The current gateway does not use a Serper key. DeepSeek image tools send the selected frame and query to the configured provider.

## Manage the services

After installation, run `start` for a stopped service or `restart` for a running
service. Saved model, device and port settings are reused; no environment
activation is needed. `start` runs a smoke test unless `--no-test` is given;
`restart` does not, so run `test` afterwards.

```bash
bash deploy/aura.sh start
bash deploy/aura.sh status
bash deploy/aura.sh test
bash deploy/aura.sh logs
bash deploy/aura.sh stop
bash deploy/aura.sh restart
```

Use `--home /absolute/path/to/runtime` for a custom runtime directory and repeat that option for subsequent commands. `--port` changes the gateway port; `--backend-port` changes the realtime backend port. Logs are in `deploy/runtime/logs/`.

The launcher also writes per-turn context dumps under `deploy/runtime/user_context/`. These may contain user conversations, tool results and location information; keep the runtime directory private. The demo has no built-in user authentication—use an access-controlled reverse proxy when exposing it beyond your own machine.

## Optional memory summaries

Memory is **disabled by default**. It compresses older turns of the current session into text using a separate Qwen3-VL model. It is not cross-session storage. Turning it off keeps ordinary conversation history; it disables the background summary service.

**Where to configure:** create `deploy/runtime/memory.env` when running on the host. The startup script sources this shell file and exports its settings to the backend. This is separate from the gateway's `tool_keys.env`.

For a first configuration (edit the existing file instead if one already exists):

```bash
mkdir -p deploy/runtime
cp deploy/memory.env.example deploy/runtime/memory.env
# Enable memory. The example assigns accelerator 2 to memory and accelerators 0,1 to the demo.
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=1/' deploy/runtime/memory.env
bash deploy/aura.sh restart
```

After a successful initial deployment, `restart` reuses its saved checkpoint and accelerators. For a new deployment, use the `up --aura ... --gpus 0,1 --speaker Serena` command instead.

The important settings in that file are:

```bash
export AURA_MEMORY_ENABLE=1
export MEMORY_VLM_GPU=2
export MEMORY_VLM_PORT=8003
export AURA_MEMORY_VLM_URL=http://127.0.0.1:8003/v1/chat/completions
```

`AURA_MEMORY_VLM_URL` is the **chat-completions endpoint of the local memory summarizer**, not the demo web page or the main AURA model. The launcher starts Qwen3-VL on port 8003 and waits for it before starting AURA. The backend sends older dialogue and selected video frames there to request a summary. The local endpoint does not require an OpenAI key. Keep the port in this URL equal to `MEMORY_VLM_PORT`; keep the two model-name settings in the example equal.

The launcher requires a separate accelerator for memory: two demo accelerators plus one memory accelerator in this example. It does not automatically switch to a remote-only service when you change the URL.

To **disable** memory:

```bash
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=0/' deploy/runtime/memory.env
bash deploy/aura.sh restart
```

For **Docker**, the host file is `docker-runtime/memory.env`, mounted as `/var/lib/aura/memory.env`. Copy the same example there, set `AURA_MEMORY_ENABLE=1` or `0`, then run `docker restart aura2`. Both services run in the same container, so the default `127.0.0.1:8003` URL stays valid. With a custom host `--home`, use `<that directory>/memory.env` and pass the same `--home` when restarting.

[Full memory guide](docs/MEMORY.md): file locations, every setting, startup/verification, disabling, accelerator requirements, and troubleshooting. The feature is experimental.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Camera/microphone unavailable | Use HTTPS or localhost and allow browser permissions |
| Checkpoint validation fails | Use full AURA 2.0 weights with matching tokenizer/processor files |
| Accelerator memory check or allocation fails | Free the selected accelerators or use the two-accelerator layout; quantization alone does not change reservations |
| Runtime or FlashInfer initialization fails | Check driver compatibility and `deploy/runtime/logs/backend.log` |
| DeepSeek tool reports a missing key / unavailable model | Configure `DEEPSEEK_API_KEY` and verify your provider account has the necessary access |
| Weather uses the wrong city | Server IP geolocation is approximate; supply the intended city |
| Port already in use | Choose different ports with `--port` and `--backend-port` |

## Documentation and layout

- [中文说明](README.zh-CN.md)
- [Full deployment reference](deploy/README.md)
- [Docker environment and lifecycle](deploy/docker/README.md)
- [Memory configuration and verification](docs/MEMORY.md)
- [Ports, accelerator layouts, access and validation](docs/DEPLOYMENT_NOTES.md)
- [Architecture and session memory](docs/ARCHITECTURE.md)
- [Original vLLM-Omni README](README.upstream.md)

```text
vllm_omni/                         bundled serving backend
examples/online_serving/aura_v2_gateway/  browser and gateway
deploy/aura.sh                     install/start/stop/test entry point
deploy/lib/                        checkpoint/configuration/smoke-test helpers
deploy/assets/                     bundled smoke-test audio
tools/tool_keys.env.example        optional credential template
```

## License and acknowledgements

The source code is distributed under [Apache License 2.0](LICENSE), with existing third-party notices retained. This implementation builds on vLLM-Omni and vLLM and uses Qwen ASR/TTS models. Model weights are separate downloads with their own license terms. See [Third-Party Notices](THIRD_PARTY_NOTICES.md).

The AURA 2.0 logo and mascot come from the [AURA 2.0 project](https://github.com/aurateam2026/Aura-2.0/tree/main/logo).
