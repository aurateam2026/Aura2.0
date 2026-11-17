# Architecture

[English](ARCHITECTURE.md) · [简体中文](ARCHITECTURE.zh-CN.md)

```text
Browser
  │  HTTPS page + WSS realtime stream (mic up, TTS down)
  ▼
reverse proxy / tunnel (optional: Cloudflare + frp, nginx, SSH -L)
  ▼
gateway.duplex_app        :9310   examples/online_serving/aura_v2_gateway
  │  page, /health, /models, /ws
  │  per-session DuplexActor: protocol translation, playback ack,
  │  vision locks, tool calls (native_tools.py), history injection
  ▼  ws://127.0.0.1:8099/v1/realtime?duplex=1
vllm-omni serve           :8099
  ├─ Stage 0  ASR        Qwen3-ASR-1.7B
  ├─ Stage 1  AURA_v2    duplex LLM, two-token <|silent|>/<|respond|>
  ├─ Stage 2  Talker     Qwen3-TTS text→codec
  └─ Stage 3  Code2Wav   Qwen3-TTS codec→24 kHz PCM, streamed back
```

## Why a gateway exists

The backend speaks the OpenAI realtime duplex protocol on `/v1/realtime`.
The browser page speaks a chunk-based format (`protocol.py`: `chunk_id`,
`video_timestamp_ms`, `wall_time_ms`). The gateway owns that translation plus
the product-side session policy:

- one backend WebSocket per page WebSocket, tied to a session id;
- audio down-sampling/up-sampling and WAV wrapping (`duplex_actor.py`);
- playback acknowledgement so TTS audio does not pile up in the browser;
- per-vision-turn tool budget and a vision lock, so overlapping tool rounds
  cannot interleave;
- `session.update` extra body: TTS speaker/language and the tool list;
- tool execution (`native_tools.py`): clock, location, weather, DeepSeek
  chat, visual search. Keys come from `--env-file`, never from the repo.

## Model-side session history

Stage 1 keeps a per-session `SessionHistory`
(`vllm_omni/model_executor/models/aura_omni/duplex/history.py`):
user/assistant rows plus optional video pixels, rendered into the ChatML
prefix on every turn. Video trimming is gateway-compatible (45 clips, drop the
oldest 30). Tool rounds are spliced as `tool_response` rows.

## Memory summarization (experimental)

`vllm_omni/model_executor/models/aura_omni/duplex/memory.py` compresses the
non-recent prefix of `SessionHistory.messages` into one text memory row using
an external VLM (default `Qwen/Qwen3-VL-8B-Instruct` served by vLLM at
`:8003`). Design rules, ported from the `AURA-qwen3-5` single-file demo:

- the background loop never mutates history during generation; snapshots are
  taken and re-validated under the history lock;
- the summarizer sees only complete user/assistant pairs, never a half tool
  round and never `<tool_response>` XML;
- after a commit, the prefix is evicted and one `(user, assistant)` memory
  pair is appended with the `我看到了` prefix, so the model reads it like a
  normal past turn;
- any reset (session drop) invalidates in-flight summaries via an epoch.
