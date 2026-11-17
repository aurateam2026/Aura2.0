# 架构

[English](ARCHITECTURE.md) · [简体中文](ARCHITECTURE.zh-CN.md)

```text
浏览器
  │  HTTPS 页面 + WSS 实时流（上传麦克风音频，接收 TTS 音频）
  ▼
反向代理 / 隧道（可选：Cloudflare + frp、nginx、SSH -L）
  ▼
gateway.duplex_app        :9310   examples/online_serving/aura_v2_gateway
  │  页面、/health、/models、/ws
  │  每个会话对应一个 DuplexActor：协议转换、播放确认、
  │  视觉锁、工具调用（native_tools.py）、历史上下文注入
  ▼  ws://127.0.0.1:8099/v1/realtime?duplex=1
vllm-omni serve           :8099
  ├─ Stage 0  ASR        Qwen3-ASR-1.7B
  ├─ Stage 1  AURA_v2    双工语言模型，使用 <|silent|>/<|respond|> 两个控制 token
  ├─ Stage 2  Talker     Qwen3-TTS，将文本转换为音频 codec token
  └─ Stage 3  Code2Wav   Qwen3-TTS，将 codec token 转为 24 kHz PCM 并流式返回
```

<a id="why-a-gateway-exists"></a>

## 为什么需要网关

后端在 `/v1/realtime` 上使用 OpenAI 实时双工协议。浏览器页面使用按块传输的数据格式（见 `protocol.py` 中的 `chunk_id`、`video_timestamp_ms`、`wall_time_ms`）。网关负责两种协议之间的转换，以及页面侧的会话管理：

- 每个页面 WebSocket 对应一个后端 WebSocket，并绑定到同一个会话 ID。
- 音频降采样、升采样和 WAV 封装（`duplex_actor.py`）。
- 播放确认，避免 TTS 音频在浏览器中不断积压。
- 每轮视觉交互的工具调用次数限制，以及防止多轮工具调用交错执行的视觉锁。
- 在 `session.update` 的扩展请求体中传入 TTS 音色、语言和工具列表。
- 执行工具（`native_tools.py`）：时间、位置、天气、DeepSeek 对话和视觉搜索。密钥从 `--env-file` 指定的文件读取，不随仓库提供。

<a id="model-side-session-history"></a>

## 模型侧的会话历史

Stage 1 为每个会话维护独立的 `SessionHistory`（`vllm_omni/model_executor/models/aura_omni/duplex/history.py`），保存用户和助手消息，以及可选的视频像素数据。每轮推理都会将这些历史内容渲染为 ChatML 前缀。

视频历史裁剪与网关兼容：按 45 个片段的阈值裁剪，一次移除最早的 30 个片段。工具调用结果以 `tool_response` 消息插入历史。

<a id="memory-summarization-experimental"></a>

## 记忆摘要（实验性功能）

`vllm_omni/model_executor/models/aura_omni/duplex/memory.py` 使用独立的 VLM，将 `SessionHistory.messages` 中较早的历史前缀压缩为文字记忆。默认使用由 vLLM 在 `8003` 端口提供服务的 `Qwen/Qwen3-VL-8B-Instruct`。设计沿用了 `AURA-qwen3-5` 单文件 Demo 的规则：

- 后台循环不会在模型生成期间修改历史；获取快照和提交前的再次校验都在历史锁保护下进行。
- 摘要模型只接收完整的用户/助手消息对，不接收未完成的工具调用轮次，也不接收 `<tool_response>` XML。
- 摘要提交后移除原有历史前缀，并加入一对带有“我看到了”前缀的 `(user, assistant)` 记忆消息，让主模型像读取普通历史对话一样读取摘要。
- 会话重置或结束时，通过递增的版本号（epoch）使尚未完成的摘要失效。
