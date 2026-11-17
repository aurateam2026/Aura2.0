# Deploy AURA 2.0 with Docker

[English](README.md) · [简体中文](README.zh-CN.md)

Run all commands from the repository root. This directory provides a recipe for building the complete runtime environment. A public prebuilt AURA 2.0 image has not been published yet.

## What the image contains

| Component | Location or version |
|---|---|
| Linux / Python | Debian Bookworm / Python 3.12 |
| uv | 0.9.8 |
| vLLM | 0.33.0, using the same pinned wheel as the host launcher |
| PyTorch / accelerator runtime libraries | Installed with the pinned vLLM wheel's dependencies |
| vLLM-Omni / gateway | This repository's source, installed at `/opt/aura` |
| Python environment | `/opt/aura-venv`, selected by `AURA_VENV` and reused at startup |
| ffmpeg / curl / git / tini | Installed during the image build |
| FlashInfer precompiled kernels | Checked at startup and downloaded if missing |
| AURA weights | Mounted read-only by the user at `/models/aura` |
| ASR / TTS weights | Downloaded into the persistent runtime directory on first startup, or mounted from local directories |
| Memory weights | Qwen3-VL is downloaded when memory is enabled, or loaded from a configured local path |

The host needs a compatible accelerator driver and container runtime integration. The image supplies the user-space runtime libraries; the host supplies the driver. On older drivers, the launcher can install forward-compatibility libraries for supported data-center accelerators. See [deployment notes](../../docs/DEPLOYMENT_NOTES.md) for accelerator and quantization validation scope.

## 1. Configure accelerator access on the host

Install Docker Engine using the [official Docker instructions](https://docs.docker.com/engine/install/).
Install the container toolkit for your accelerator and configure it for Docker following the driver vendor's instructions. Restart Docker after changing its configuration.

Check that Docker can start a container with accelerator access:

```bash
docker run --rm --gpus all ubuntu:22.04 true
```

This command should exit successfully. If it reports `could not select device driver`, check that the container toolkit is installed and configured for Docker.

You do not need to create a separate Python environment on the host. Building the image does not require an accelerator, but it needs network access to the image registry, PyPI, and `wheels.vllm.ai`. Startup also needs access to model and kernel download services.

## 2. Build the environment image

```bash
docker build -f deploy/docker/Dockerfile -t aura2:local .
```

The first build downloads PyTorch and accelerator runtime dependencies. Allow disk space for the image, build cache, models, and runtime cache; at least 100 GB is recommended. The build ends with `uv pip check` to check dependency compatibility. Keep model weights and real API keys out of the image: `deploy/docker/Dockerfile.dockerignore` excludes runtime directories, credentials, model weights, and backup archives.

Rebuild after changing source code, then recreate the container to use the new image. `docker restart` continues to use the original image.

## 3. Mount your checkpoint and start

```bash
export AURA_CKPT=/absolute/path/to/aura-2-checkpoint
mkdir -p docker-runtime

docker run -d --name aura2 --gpus all --shm-size=16g --stop-timeout=180 \
  -p 127.0.0.1:9310:9310 \
  --mount "type=bind,src=$AURA_CKPT,dst=/models/aura,readonly" \
  --mount "type=bind,src=$PWD/docker-runtime,dst=/var/lib/aura" \
  aura2:local --aura /models/aura --gpus 0,1 --speaker Serena

docker logs -f aura2
```

- Docker's `--gpus all` makes the accelerators visible inside the container. The `--gpus 0,1` argument after the image name belongs to AURA: accelerator 0 runs the main model and accelerator 1 runs ASR/TTS. Exposing all accelerators in this example preserves consistent accelerator numbering.
- For the single-accelerator template, change the AURA argument to `--gpus 0`. That template targets 80 GB accelerators; see deployment notes for memory reservations and validation scope.
- `--shm-size=16g` reserves container shared memory for communication between processes.
- `--stop-timeout=180` allows time for the multi-stage services to shut down.
- The default container voice is Serena. There is no frontend setup screen; granting camera and microphone access starts the call automatically.
- Use a dedicated `docker-runtime/` directory for this container. Do not mount the `deploy/runtime/` directory of a running host demo. It contains process state, settings, model caches, and conversation logs.

The first startup downloads ASR/TTS weights and missing kernels, loads the models, starts the web page, and runs a smoke test. After `AURA is up` appears, open `http://localhost:9310/`. To access an accelerator server from your laptop, run this on the laptop:

```bash
ssh -N -L 9310:127.0.0.1:9310 your-user@your-server
```

For a phone or public access, configure an HTTPS reverse proxy on the host forwarding to `127.0.0.1:9310`, with WebSocket support. Only the web port is published here. The main model's port 8099 and memory port 8003 do not need to be exposed publicly.

## 4. Configure and toggle memory

The container reads `/var/lib/aura/memory.env`, which corresponds to **`docker-runtime/memory.env` on the host**. For the first configuration:

```bash
cp deploy/memory.env.example docker-runtime/memory.env
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=1/' docker-runtime/memory.env
docker restart aura2
```

If the file already exists, edit it directly to preserve your ports, accelerator selection, and local model paths. The default template places memory on accelerator 2 and AURA on accelerators 0 and 1, so accelerator 2 needs its own available memory budget. With two cards, you can use the single-accelerator demo template plus a separate memory accelerator; update `MEMORY_VLM_GPU` accordingly.

AURA and Qwen3-VL run in the same container. Keep:

```bash
export MEMORY_VLM_PORT=8003
export AURA_MEMORY_VLM_URL=http://127.0.0.1:8003/v1/chat/completions
```

This URL points to the **summary model endpoint**, not the web page or main AURA model. It does not require an OpenAI key.

To disable memory:

```bash
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=0/' docker-runtime/memory.env
docker restart aura2
```

Check the endpoint and summary logs:

```bash
docker exec aura2 curl -f http://127.0.0.1:8003/v1/models
docker exec aura2 bash deploy/aura.sh status
# Check after enough rounds of video conversation:
docker exec aura2 grep '\[memory\]' /var/lib/aura/logs/backend.log
```

See the [memory guide](../../docs/MEMORY.md) for the check interval, minimum rounds, retained rounds, and other settings.

## 5. Use local ASR/TTS weights

Add these two mounts before the image name in the `docker run` command above:

```text
--mount type=bind,src=/absolute/path/to/Qwen3-ASR-1.7B,dst=/models/asr,readonly
--mount type=bind,src=/absolute/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice,dst=/models/tts,readonly
```

Add these AURA arguments after the image name:

```text
--asr /models/asr --tts /models/tts
```

Mount local memory weights similarly and set `MEMORY_VLM_MODEL` in `memory.env` to the container path. Keep the two served model names identical. Local weights do not eliminate the potential first-start download of FlashInfer kernels; prepare these dependencies in advance for a fully offline deployment.

## 6. Optional tool keys

```bash
cp tools/tool_keys.env.example docker-runtime/tool_keys.env
chmod 600 docker-runtime/tool_keys.env
# Edit the file locally and enter your key as NAME=value, without export or quotes.
docker restart aura2
```

The files are read differently: `tool_keys.env` configures the gateway, while `memory.env` is sourced as a shell file by the launcher and requires `export`.

## 7. Manage, update, and troubleshoot the container

```bash
docker logs -f aura2
docker exec aura2 bash deploy/aura.sh status
docker exec aura2 bash deploy/aura.sh test
docker stop aura2
docker start aura2
docker restart aura2
```

To change the image or mount configuration, run `docker stop aura2`, then `docker rm aura2`, and issue a new `docker run` command using the same `docker-runtime/` directory. Removing the container does not delete that host directory. The entrypoint reuses the image environment, removes PID files from a previous container, starts the services, and stops them on exit. Use Docker's stop/restart commands instead of running `aura.sh stop/restart` inside the container.

| Symptom | Check |
|---|---|
| Accelerators are not visible | Repeat the container launch check above and verify the container toolkit configuration |
| Insufficient accelerator memory at startup | Check the selected cards and current usage; avoid interrupting other services |
| Camera or microphone unavailable | Use HTTPS or localhost, not plain HTTP on a remote IP |
| Memory service fails to start | Check the separate accelerator, `docker-runtime/logs/memory_vlm.log`, model path, and port |
| Downloads are unavailable | Check host/container networking and proxies; reuse persistent model and cache directories |
| Container exits | Check `docker logs aura2` and `docker-runtime/logs/`; a backend or gateway exit ends the container |

See [validation scope](../../docs/DEPLOYMENT_NOTES.md#validation). A successful build and dependency check are not equivalent to a full accelerator audio/video inference test.
