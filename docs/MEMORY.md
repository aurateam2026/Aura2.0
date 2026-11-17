# AURA 2.0 Memory Guide

[English](MEMORY.md) · [简体中文](MEMORY.zh-CN.md)

Memory is a background summarization feature for the current session, disabled by default. The main AURA model continues to process live video and current questions. A separate Qwen3-VL model summarizes older video and dialogue into text, which is added back to the same session's context.

This is not a database shared across users, and it does not automatically restore memories from a previous call into a new session. Ordinary conversation history remains available when summarization is disabled.

## Where to put the configuration

Paths below are relative to the repository root unless absolute:

| Deployment | File to edit | Apply changes |
|---|---|---|
| Default host deployment | `deploy/runtime/memory.env` | `bash deploy/aura.sh restart` |
| Host with custom `--home /srv/aura-runtime` | `/srv/aura-runtime/memory.env` | `bash deploy/aura.sh restart --home /srv/aura-runtime` |
| Docker setup in this repository | Host `docker-runtime/memory.env`, mounted as `/var/lib/aura/memory.env` inside the container | `docker restart aura2` |

`deploy/aura.sh` reads `memory.env` from the runtime directory before starting the backend.
Keep `export` on the configuration lines so the backend workers inherit the variables. Setting only `AURA_MEMORY_ENABLE=1` without exporting it can start the summary service without enabling summarization in the model workers.
Do not put these settings in `tool_keys.env`; that file configures the gateway's network tools.

## Enable memory: demo on accelerators 0 and 1, memory on accelerator 2

First complete a normal AURA deployment. The following commands create the configuration for the first time. If the file already exists, edit it directly to preserve your settings.

```bash
mkdir -p deploy/runtime
cp deploy/memory.env.example deploy/runtime/memory.env
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=1/' deploy/runtime/memory.env
bash deploy/aura.sh restart
```

See the complete template in [deploy/memory.env.example](../deploy/memory.env.example).
The launcher requires the memory accelerator to be separate from the demo accelerators selected by `--gpus`. Accelerator 2 is an example, not a fixed requirement; change `MEMORY_VLM_GPU` to suit your machine.

If the demo has not been started before, configure memory and use the initial startup command instead:

```bash
bash deploy/aura.sh up --aura /absolute/path/to/aura-2-checkpoint --gpus 0,1 --speaker Serena
```

The script starts Qwen3-VL, waits for its endpoint to become ready, and then starts the AURA backend and web page. The first run downloads Qwen3-VL weights. Its default memory budget is 30% of the selected accelerator; lowering this fraction does not guarantee that the model will fit on a smaller accelerator.

## What `AURA_MEMORY_VLM_URL` means

Default value:

```bash
export AURA_MEMORY_VLM_URL=http://127.0.0.1:8003/v1/chat/completions
```

- `127.0.0.1`: the machine running the AURA backend, or the same container in this repository's Docker setup.
- `8003`: the Qwen3-VL summary service port, matching `MEMORY_VLM_PORT`.
- `/v1/chat/completions`: an OpenAI-compatible endpoint that accepts text and images and returns a summary. This refers to the API format; the local service does not require an OpenAI account or key.

The AURA backend POSTs older dialogue and sampled video frames to this address. Port `9310` serves the web page and port `8099` serves the AURA backend; neither is the default summary endpoint.

When changing the port, update both settings, for example:

```bash
export MEMORY_VLM_PORT=8013
export AURA_MEMORY_VLM_URL=http://127.0.0.1:8013/v1/chat/completions
```

The current launcher always starts a local summary model when `AURA_MEMORY_ENABLE=1`. Changing the URL alone does not skip that step, so this guide uses the managed local service. The summary client also does not send an authorization header; entering a cloud endpoint that requires a key will not automatically configure authentication.

## Model paths and served model names

The default template uses:

```bash
export MEMORY_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
export MEMORY_VLM_SERVED_NAME=Qwen/Qwen3-VL-8B-Instruct
export AURA_MEMORY_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

`MEMORY_VLM_MODEL` selects the weights to load: a Hugging Face model ID or a local directory containing the full model.
`MEMORY_VLM_SERVED_NAME` is the name exposed by the service. `AURA_MEMORY_VLM_MODEL` is the name sent in the request JSON. These two names must match. When changing only the local weight path, you can leave both names unchanged.

In Docker, a local model path must exist inside the container through a mount. A host path that is not mounted into the container cannot be used directly.

## Other settings

| Variable | Default | Meaning |
|---|---|---|
| `AURA_MEMORY_ENABLE` | `0` | `1` enables background summaries; `0` disables them |
| `MEMORY_VLM_GPU` | `2` | Separate accelerator for the managed summary model |
| `MEMORY_VLM_PORT` | `8003` | Local summary service port |
| `MEMORY_VLM_GPU_UTIL` | `0.3` | vLLM accelerator memory budget as a fraction of total memory |
| `MEMORY_VLM_WAIT_S` | `1800` | Maximum time in seconds to wait for the summary service to start |
| `AURA_MEMORY_INTERVAL_S` | `30` | Interval in seconds between checks for work to summarize |
| `AURA_MEMORY_MIN_ROUNDS` | `6` | Minimum number of complete history rounds before checking for a summary |
| `AURA_MEMORY_KEEP_ROUNDS` | `4` | Number of recent rounds retained without summarization |
| `AURA_MEMORY_MAX_FRAMES` | `8` | Maximum number of sampled video frames per summary request |
| `AURA_MEMORY_TIMEOUT_S` | `30` | Timeout in seconds for each summary request |

Reaching six rounds does not trigger an immediate summary. The next check must run, older video must be available, and no user turn may be in flight. Tool rounds are not split in the middle. If the summary request fails or returns empty content, existing history is retained and the background loop retries in a later interval.

## Verify that memory is working

Host deployment:

```bash
bash deploy/aura.sh status
curl -f http://127.0.0.1:8003/v1/models
grep '\[memory\]' deploy/runtime/logs/backend.log
tail -n 50 deploy/runtime/logs/memory_vlm.log
```

Docker deployment:

```bash
docker exec aura2 bash deploy/aura.sh status
docker exec aura2 curl -f http://127.0.0.1:8003/v1/models
docker exec aura2 grep '\[memory\]' /var/lib/aura/logs/backend.log
```

Replace `8003` if you changed the port. A response from `/v1/models` containing the expected model name confirms that the service is ready; it does not by itself prove that a summary has been generated. After enough rounds of video conversation, look for `[memory] loop started` and `[memory] summarizing` in the backend log. If you see `[memory] VLM call failed`, check the URL, model name, network connection, and summary model log.

## Disable and re-enable memory

Set the switch to 0 in the configuration file and restart the backend. For a host deployment:

```bash
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=0/' deploy/runtime/memory.env
bash deploy/aura.sh restart
```

This stops the existing demo and managed memory service, then starts the demo with summarization disabled. Model files and caches do not need to be deleted.
To re-enable memory, change `0` back to `1` and restart again. In Docker, edit `docker-runtime/memory.env` and run `docker restart aura2`.

Restarting ends active calls and clears their in-memory session context, so reconnect afterward. Previously written context logs on disk are retained.
