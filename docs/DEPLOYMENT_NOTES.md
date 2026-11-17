# Deployment notes

[English](DEPLOYMENT_NOTES.md) · [简体中文](DEPLOYMENT_NOTES.zh-CN.md)

## Ports

| Port | Process |
| --- | --- |
| 9310 | gateway page (`GATEWAY_PORT`) |
| 8099 | vllm-omni realtime backend (`BACKEND_PORT`) |
| 8003 | optional memory VLM (`AURA_MEMORY_VLM_URL`) |

## Accelerator layouts

| Layout | Cards | Notes |
| --- | --- | --- |
| 2 accelerators | AURA on the first, ASR + Talker + Code2Wav on the second | recommended |
| 1 accelerator | all four stages | needs ~70 GB free |
| memory VLM | a separate card, distinct from the demo accelerators | Managed Qwen3-VL-8B service; default `gpu_memory_utilization=0.3` |

## TTS voice

`--speaker` (or `AURA_TTS_SPEAKER`) selects a CustomVoice speaker: Vivian
(default), Serena, Uncle_Fu, Ryan, Aiden, Ono_Anna, Sohee, Eric (Sichuan),
Dylan (Beijing). `AURA_TTS_LANGUAGE` defaults to Chinese.

## External access

The page binds `0.0.0.0:9310`. Options, from simplest:

- Local browser: open `http://localhost:9310/`. Plain HTTP on a remote IP does not enable camera/microphone access.
- SSH tunnel: `ssh -N -L 9310:127.0.0.1:9310 <host>`.
- Direct HTTPS: place `cert.pem` and `key.pem` in
  `examples/online_serving/aura_v2_gateway/gateway/`, then start. The gateway
  detects the files and uses HTTPS automatically. Generate self-signed certs with
  `openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 3650 -subj "/CN=aura-gateway"`.
  Use a certificate trusted by your browser/device. For phone access, a domain with a trusted HTTPS reverse proxy is easier to configure.
- Public: put nginx or a frp/Cloudflare tunnel in front. Microphone capture
  requires a secure context, so use HTTPS/WSS or `localhost`.

## Memory configuration

Place exported variables in `deploy/runtime/memory.env` (or `<--home>/memory.env`).
For Docker, edit host `docker-runtime/memory.env`, mounted at `/var/lib/aura/memory.env`.
Restart the backend/container after changes. `AURA_MEMORY_VLM_URL` points to the
separate Qwen3-VL summary endpoint; it is not the web page or main model endpoint.
See [the full memory guide](MEMORY.md) and [example configuration](../deploy/memory.env.example).

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `AURA_TTS_SPEAKER` | `Vivian` | Talker voice |
| `AURA_TTS_LANGUAGE` | empty (Chinese) | Talker language |
| `AURA_GATEWAY_TRACE` | `0` | log every frame in the gateway |
| `AURA_WEATHER_FORECAST_URL` | `https://api.open-meteo.com/v1/forecast` | optional Open-Meteo-compatible current/forecast endpoint; set in the gateway environment or `--env-file` |
| `AURA_MEMORY_ENABLE` | `0` | background memory summarization |
| `AURA_MEMORY_VLM_URL` | `http://127.0.0.1:8003/v1/chat/completions` | summarizer VLM |
| `AURA_MEMORY_INTERVAL_S` | `30` | loop period |
| `AURA_MEMORY_MIN_ROUNDS` | `6` | start summarizing above this many rounds |
| `AURA_MEMORY_KEEP_ROUNDS` | `4` | recent rounds never summarized |
| `AURA_MEMORY_MAX_FRAMES` | `8` | frames sent per summary |

## Docker

[Docker deployment](../deploy/docker/README.md) includes the environment build,
accelerator container setup, model mounts, persistent runtime configuration,
Accelerator selection, memory and container lifecycle commands. `AURA_VENV` selects the
preinstalled image environment; host installs keep the default runtime venv.

## Validation

- Existing accelerator demo: Quantized AURA, ASR and TTS were exercised on two 80 GB accelerators.
- Docker recipe: the image was built and `uv pip check` passed. Inside the
  container, the install command reused the image environment, installed runtime
  compatibility libraries and FlashInfer kernels, and verified that PyTorch 2.14
  could see the host accelerators. Local ASR/TTS model mounts passed the install checks.
- Container lifecycle: stale PID cleanup, exclusive runtime-directory locking,
  `docker stop` cleanup, and container exit after a gateway failure were checked
  with isolated stand-in processes; these tests do not load model weights.
- A complete audio/video conversation through this new Docker image has not been
  tested. The available accelerators were already serving other workloads; no running
  demo was stopped or replaced for this documentation update.
- The single-accelerator template is provided for 80 GB cards. This update does not claim
  a complete single-accelerator deployment test or establish support for other accelerator models.

Weights are not bundled in the image or repository. Model repositories:
[AURA 2.0](https://huggingface.co/aurateam/AURA_2) and
[AURA 2.0 BF8](https://huggingface.co/aurateam/AURA_2_BF8).

## Weather connectivity

City geocoding and weather forecasts use different hosts. A successful city
lookup does not guarantee that `api.open-meteo.com` is reachable. If needed,
set `AURA_WEATHER_FORECAST_URL` to a reachable Open-Meteo-compatible endpoint
and restart the gateway; verify that it covers the requested date. Past dates
continue to use `archive-api.open-meteo.com`. Failed requests are reported as
errors to the page and model.

For a gateway using a private certificate authority, set
`SSL_CERT_FILE=/path/to/ca.crt` for `start` or `test`. Use HTTPS and a hostname
covered by the certificate when the gateway has TLS enabled.
