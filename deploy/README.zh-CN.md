# AURA 一键部署（AURA_v2 + vllm-omni duplex）

[English](README.md) · [简体中文](README.zh-CN.md)

给定一份 AURA_v2 权重，在带 accelerators 的 Linux 服务器上部署实时音视频对话页面。以下命令均在仓库根目录执行。Docker 方式见 [Docker 部署](docker/README.zh-CN.md)。

```bash
bash deploy/aura.sh up --aura /path/to/aura_v2_checkpoint
```

跑完会打印页面地址（默认 `http://<服务器>:9310/`）。整个过程不需要 root，也不需要 Docker。

## 它做了什么

| 步骤 | 内容 |
| --- | --- |
| 1. 检查环境 | Linux x86_64、兼容 CUDA 的加速器驱动、磁盘空间 |
| 2. 安装 uv | 已有就用；没有就装到 `deploy/runtime/bin`（不改系统、不改 `~/.bashrc`） |
| 3. 代码 | 直接用本仓库自带的 vllm-omni 代码树（`deploy/` 的上一级），不 clone 任何上游仓库 |
| 4. 运行时兼容 | 固定的 vLLM 0.33 构建面向驱动 ≥ 580。驱动更老（如 535）时自动下载厂商提供的 `cuda-compat-13-0` 前向兼容库，解包到 `deploy/runtime/cuda-compat`，只对本服务生效 |
| 5. Python 环境 | uv 装 Python 3.12，建 `deploy/runtime/venv`，装 vLLM 0.33.0（cu130）、vllm-omni（可编辑安装），并确认 torch 能用 accelerators |
| 5b. FlashInfer 内核 | AURA 主模型用 FlashInfer 注意力。服务器上没有 运行时编译器（nvcc）时，用 `vllm download-kernels` 装上预编译内核（`flashinfer-cubin` + `flashinfer-jit-cache`），并确认可用 |
| 6. 下载 ASR/TTS | `Qwen/Qwen3-ASR-1.7B`、`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`（约 9 GB），也可以用 `--asr`/`--tts` 指定本地路径 |
| 7. 校验 AURA 权重 | `config.json`、完整的 safetensors 分片（拒绝只有 LoRA 的目录）、`<\|silent\|>`/`<\|respond\|>` 等控制 token 的 id 是否是 AURA_v2 的 |
| 8. 选 accelerators、生成配置 | 用分支自带的 `aura_omni.yaml`（双卡）或 `aura_omni_1gpu.yaml`（单卡），只替换模型路径和卡号 |
| 9. 启动 | 后端 `vllm-omni serve`（:8099，四个 stage + 预热；首次约 5～8 分钟，之后约 4 分钟）和页面 `gateway.duplex_app`（:9310） |
| 10. 冒烟测试 | 模拟浏览器：发摄像头画面、两次语音提问（第二次通常会触发工具调用）、一次打字提问，检查 ASR、回复文字、语音都正常 |

重复运行是安全的：已完成的步骤会跳过（环境按"代码提交 + vLLM 版本"判断是否要重装，模型下载完会留标记）。

## 硬件要求

- 已验证的双设备 demo 使用 80 GB accelerators；其他设备的可用性还取决于运行时和权重量化格式，见 [验证范围](../docs/DEPLOYMENT_NOTES.zh-CN.md#validation)。驱动 < 580 时，仅支持厂商前向兼容的 accelerators 可使用脚本安装的兼容库。
- 显存：

| 布局 | 卡 | 每张卡占用（按总显存比例） |
| --- | --- | --- |
| 双卡（推荐） | 2 张 | AURA 那张约 90%，ASR+TTS 那张约 38% |
| 单卡 | 1 张 80 GB | 约 88%（四个 stage 全在一张卡上） |

- 磁盘：环境约 25 GB + ASR/TTS 约 9 GB + AURA 权重（约 18 GB）。
- 网络：要能访问 GitHub、PyPI、`wheels.vllm.ai`、`flashinfer.ai`、HuggingFace、accelerators 厂商下载站。有代理就先 `export https_proxy=...`；HuggingFace 慢可以 `export HF_ENDPOINT=<镜像>`。

## 常用命令

```bash
bash deploy/aura.sh up --aura /path/to/ckpt                 # 一键：安装 + 启动 + 测试
bash deploy/aura.sh up --aura /path/to/ckpt --gpus 0,1      # 指定卡：AURA 用 0，ASR+TTS 用 1
bash deploy/aura.sh up --aura /path/to/ckpt --gpus 3        # 单卡布局用 accelerators 3
bash deploy/aura.sh status                                  # 进程、端口、显存
bash deploy/aura.sh test                                    # 再跑一次冒烟测试
bash deploy/aura.sh logs                                    # 跟踪日志
bash deploy/aura.sh stop                                    # 停止
bash deploy/aura.sh restart                                 # 重启（沿用上次的权重和卡）
bash deploy/aura.sh start --aura /path/to/another_ckpt      # 换一份权重启动（环境不重装）
```

不加 `--gpus` 时自动选：有 2 张空闲卡（已用 < 1 GB）就用双卡布局，只有 1 张就用单卡布局，都没有就报错并列出各卡占用。指定的卡显存不够时会拒绝启动，确认要强行启动加 `--force`。

## 选项

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--aura PATH` | （必填，首次） | AURA_v2 权重目录 |
| `--gpus auto\|I\|I,J` | `auto` | 见上 |
| `--home DIR` | `deploy/runtime` | 所有安装内容、日志、运行状态都在这里 |
| `--port N` | 9310 | 页面端口 |
| `--backend-port N` | 8099 | vllm-omni realtime 端口（只需本机访问） |
| `--name NAME` | 权重目录名 | 页面上显示的模型名 |
| `--asr PATH` / `--tts PATH` | 自动下载 | 使用本地的 ASR / TTS 权重 |
| `--env-file FILE` | `deploy/runtime/tool_keys.env` | 页面工具用的 key，如 `DEEPSEEK_API_KEY=sk-...`（不要提交到 git） |
| `--speaker NAME` | Vivian | TTS 音色（Qwen3-TTS CustomVoice 内置）：Vivian、Serena、Uncle_Fu、Ryan、Aiden、Ono_Anna、Sohee、Eric（四川话）、Dylan（北京话）。需要 vllm-omni 的 gateway 支持 `AURA_TTS_SPEAKER` |
| `--no-test` | | 启动后不跑冒烟测试 |
| `--force` | | 显存检查不通过也启动 |

也可以用同名环境变量设置：`AURA_HOME`、`GPUS`、`GATEWAY_PORT`、`BACKEND_PORT`、`VLLM_WHEEL`、`ASR_REPO`、`TTS_REPO`、`HF_HUB_DISABLE_XET`（默认 1，避开会卡住的 Xet 下载）。

## 随 demo 启动 Memory 摘要模型

Memory 默认关闭。配置文件是仓库根目录下的 **`deploy/runtime/memory.env`**，
不是 `tool_keys.env`。如果使用 `--home DIR`，则是 `DIR/memory.env`。
Docker 对应宿主机 `docker-runtime/memory.env`。

首次配置（已有文件时直接编辑）：

```bash
# 从仓库根目录执行
mkdir -p deploy/runtime
cp deploy/memory.env.example deploy/runtime/memory.env
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=1/' deploy/runtime/memory.env
bash deploy/aura.sh restart
```

默认模板将 memory 放在 accelerator 2，demo 使用 accelerator 0、1。脚本要求它使用独立 accelerator。
`AURA_MEMORY_VLM_URL=http://127.0.0.1:8003/v1/chat/completions` 指向本地 Qwen3-VL
摘要服务；它接收旧对话与抽样视频帧，返回摘要，不需要 OpenAI Key。
URL 端口与 `MEMORY_VLM_PORT` 保持一致，配置行保留 `export`，修改后必须重启。

关闭时把 `AURA_MEMORY_ENABLE=1` 改为 `AURA_MEMORY_ENABLE=0` 并重启。
完整解释、模型路径、全部参数和验证方法见 [Memory 使用说明](../docs/MEMORY.zh-CN.md)。

Docker 的环境安装、运行目录、配置挂载与启停命令见 [Docker 部署](docker/README.zh-CN.md)。

## 用户上下文文件

通过 `aura.sh start` / `up` / `restart` 启动后，每次 AURA 推理的上下文保存到
`deploy/runtime/user_context/session_<session_id>/1.json`、`2.json` 等文件。
自定义 `--home` 时使用对应目录下的 `user_context`。

文件包含 `session_id`、`seq`、`payload`、`message` 和完成后的 `raw_response`：
`payload.messages` 是输入模板中的消息，`payload.prompt` 是裁剪、记忆摘要和工具结果合并后
实际构造的模板文本，保留空 thinking 和视频占位符。视频像素、音频及长 data URL 不保存，
改为形状、类型及视频元数据；工具定义和文本结果保留。`input` 区分语音、纯视觉及工具续答，
记录 ASR 原文和实际用户文本。回复保留控制标记和工具调用 XML。

输入先写为 `pending`，收到模型完整回复后更新为 `completed`；会话关闭时尚未收到回复的
记录标为 `incomplete`。崩溃留下的 `pending` 只代表未记录到完整回复。
写入由后台线程按请求 ID 配对，文件原子替换，重启不会删除已有文件。
模板文本中的视频占位符尚未经过多模态处理器展开；`raw_response.prompt_token_ids`
在后端输出提供时保存，用于进一步核对展开后的输入。

Duplex 视频按会话累计时间：新片段记录固定起点，片段时长仍为帧数 / 2，输入 fps 保持 2。
预处理缓存只保存片段内的相对时间，组装模型输入时再加会话起点，因此历史片段命中缓存、
重复画面、工具续答和历史压缩不会重置时间轴。新会话从零开始；重连形成新会话时同样重置。
`input.video_start_seconds` 按视频顺序记录各片段起点。每段两帧时，实际时间 token 依次为
`<0.2 seconds>`、`<1.2 seconds>`、`<2.2 seconds>`（帧组中心为 0.25 秒，格式化到一位小数）。
此逻辑用于当前未启用 video pruning 的 AURA Qwen3.5 部署；不支持同时开启 video pruning。

直接启动后端时，设置 `AURA_USER_CONTEXT_DIR` 可启用同一功能；未设置则关闭。

## 打开页面

浏览器只在 `https://` 或 `localhost` 下才允许用摄像头和麦克风。

点击“允许权限”并授权摄像头和麦克风后，页面直接进入通话。发送设置固定为原来的
720p、每 1 秒发送 2 帧，默认启用工具调用，使用带 `<|respond|>` / `<|silent|>` 的
系统提示词，并自动使用服务提供的默认模型。挂断后释放摄像头和麦克风，点击“再次通话”即可重新接通。

**自己用：SSH 隧道**（最简单）

```bash
ssh -N -L 9310:127.0.0.1:9310 your-user@your-server
```

然后打开 `http://localhost:9310/`。

**给别人用：HTTPS 反向代理**。例如用 [Caddy](https://caddyserver.com/)（自动申请证书，需要域名指向这台服务器、开放 80/443）：

```text
aura.example.com {
    reverse_proxy 127.0.0.1:9310
}
```

页面的实时通话走 WebSocket（`/ws`），Caddy 默认就支持；用 Nginx 时要加 `proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";`。

## 目录结构

```text
deploy/
├── aura.sh              入口
├── lib/common.sh        日志、进程管理等公共函数
├── lib/check_ckpt.py    AURA 权重校验（只用标准库）
├── lib/make_yaml.py     从分支自带配置生成部署配置
├── lib/smoke_test.py    端到端冒烟测试
└── assets/              冒烟测试用的两段 16 kHz 中文语音

deploy/runtime/          （运行后生成，可用 --home 改位置）
├── venv/                Python 环境
├── models/              ASR / TTS 权重
├── cuda-compat/         CUDA 13 兼容库（只在驱动 < 580 时出现）
├── deploy.yaml          实际使用的部署配置
├── aura.env             上次的设置（restart / status 会读它）
├── user_context/session_<session_id>/*.json  每轮模型上下文与回复
├── logs/backend.log  logs/gateway.log
└── run/*.pid
```

## 排错

| 现象 | 处理 |
| --- | --- |
| `torch ... cannot use the accelerator` | 驱动太老且不是数据中心卡，或兼容库不适用：把驱动升到 ≥ 580 |
| `no idle accelerator` | 用 `--gpus` 指定，或先释放显存 |
| `not enough free accelerator memory` | 换卡，或双卡布局；确实要启动加 `--force` |
| `backend exited during startup` | 看 `deploy/runtime/logs/backend.log` 末尾，常见是显存不足或权重路径不对 |
| 日志里有 `FlashInfer backend is not available` | 预编译内核没装上：`deploy/runtime/venv/bin/vllm download-kernels`，或装 accelerators 运行时 13 工具链（nvcc + ninja） |
| 冒烟测试失败但服务在跑 | 看 `deploy/runtime/logs/gateway.log`；`bash deploy/aura.sh test` 可重跑 |
| 下载很慢或卡住 | 设 `HF_ENDPOINT` 镜像，或先手动下好再用 `--asr` / `--tts` |
| 端口被占用 | `--port` / `--backend-port` 换端口 |

在 SLURM 集群上：要在**分配到 accelerators 的计算节点**上运行，并保持作业存活；有的集群会在 `srun` 命令结束时清理它启动的进程，这种情况下请在作业的交互 shell（如 `tmux` 里）运行，或用 `sbatch` 提交。

## 默认用的组件

| 组件 | 版本 |
| --- | --- |
| vllm-omni | 本仓库自带的代码树（源自 `AURA_production_v2_duplex`，提交 `9182429` 加本地改动） |
| vLLM | 0.33.0（cu130），PyTorch 2.14 |
| ASR | Qwen3-ASR-1.7B |
| TTS | Qwen3-TTS-12Hz-1.7B-CustomVoice，音色由 `--speaker` 指定；本项目示例使用 Serena，中文 |
| 页面 | 分支自带的 `examples/online_serving/aura_v2_gateway` |

TTS Talker 的 `min_tokens` 为 2。整句送入 Talker 时，生成上限取配置中的
`max_tokens`（默认 1024）与 `50 + 15 × 朗读文本字符数` 的较小值。
这是音频 token 的上限；正常遇到结束标记仍会提前结束，音色由现有 `--speaker` 设置决定。

## 第三方组件与许可

本仓库包含部署脚本与 vllm-omni 代码树，**不包含任何模型权重**；权重在部署时从官方来源下载到用户自己的服务器。

| 组件 | 来源 | 许可 | 本仓库是否分发 |
| --- | --- | --- | --- |
| vllm-omni（AURA 分支） | 本仓库内分发（源自 github.com/vllm-project/vllm-omni） | Apache-2.0 | 是 |
| vLLM、FlashInfer、PyTorch | wheels.vllm.ai / flashinfer.ai / PyPI | Apache-2.0 / BSD | 否，部署时安装 |
| Qwen3-ASR-1.7B | huggingface.co/Qwen | Apache-2.0 | 否，部署时下载 |
| Qwen3-TTS-12Hz-1.7B-CustomVoice（含预设音色 Vivian） | huggingface.co/Qwen | Apache-2.0 | 否，部署时下载；代码里只引用音色名 |
| accelerators 运行时 13 前向兼容库（`cuda-compat-13-0`） | 驱动厂商软件源 | 厂商软件许可 | 否，仅在驱动 < 580 时由用户机器下载 |
| uv | astral.sh | Apache-2.0 / MIT | 否，没有时自动安装 |
| AURA_v2 权重 | 用户提供 | 由权重发布方决定 | 否 |

如果要把模型权重直接打包进仓库或镜像分发，请遵守该权重的许可。对于 Apache-2.0 权重，附上许可证文本并保留原始版权声明。
