<div align="center">

<img src="docs/assets/aura-2-logo.png" width="200" alt="AURA 2.0 logo" align="left">

<h1>Beyond Answering: Learning to Speak First and Act in Streaming Video</h1>

<a href="https://aurateam2026.github.io/Aura-2.0/"><img alt="Home Page" src="https://img.shields.io/badge/Home-Page-0A66C2?logo=googlechrome&logoColor=white" height="22"></a>
<a href="https://huggingface.co/aurateam/AURA_2"><img alt="Model: Aura2.0" src="https://img.shields.io/badge/Model-Aura2.0-FFD21E?logo=huggingface&amp;logoColor=black" height="22"></a>
<a href="https://huggingface.co/aurateam/AURA_2_BF8"><img alt="Model: Aura2.0 BF8" src="https://img.shields.io/badge/Model-Aura2.0_BF8-FFD21E?logo=huggingface&amp;logoColor=black" height="22"></a>
<a href="deploy/docker/README.zh-CN.md"><img alt="Docker deployment" src="https://img.shields.io/badge/Deploy-Docker-2496ED?logo=docker&logoColor=white" height="22"></a>
<a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/License-Apache_2.0-blue" height="22"></a>
<a href="https://arxiv.org/pdf/2611.17048"><img alt="Technical Report: arXiv" src="https://img.shields.io/badge/Technical_Report-arXiv-EC1C24?logo=arxiv&logoColor=white" height="22"></a>

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

AURA 2.0 持续接收摄像头画面，在需要时回应，通过流式语音与用户交流。本仓库自带 AURA duplex 后端、浏览器页面、工具执行器和部署脚本，以及运行所需的 vLLM-Omni 源码。

## Demo 部署与使用

### 环境要求

| 项目 | 要求及验证范围 |
| --- | --- |
| 系统 | Linux x86_64；开发环境为 Ubuntu 22.04 |
| 已验证配置 | 2 × 80 GB accelerators，分别运行 AURA 和 ASR/TTS |
| 单设备配置 | 提供面向 80 GB accelerators 的模板，约需 70 GB 空闲设备内存；验证范围见部署说明 |
| 运行环境 | accelerators 的驱动和运行时需与仓库固定的依赖版本兼容 |
| 磁盘 | 建议至少 50 GB 空闲，模型、缓存和会话日志还会继续占用空间 |
| 基础工具 | `bash`、`git`、`curl`、`python3`；脚本安装 uv 和 Python 3.12，并检查 accelerators 是否可用 |
| 网络 | 安装时访问依赖及模型下载站；联网工具运行时也需要网络 |
| 浏览器 | 支持摄像头和麦克风的现代浏览器，使用 **HTTPS 或 localhost** |

部署使用固定版本的 vLLM 0.33 wheel 和本仓库源码。验证范围限于上述配置；其他 accelerators 的兼容性和内存需求需要单独验证。

双设备模板分别预留 AURA 所用 accelerators 内存的约 90% 和 ASR/TTS 所用 accelerators 内存的约 38%；单设备模板约预留 88%。更换量化权重不会自动降低这些比例。更多信息见 [部署说明](docs/DEPLOYMENT_NOTES.zh-CN.md)。

## 准备模型

- [AURA 2.0](https://huggingface.co/aurateam/AURA_2)：下载后通过 `--aura` 指定本地目录。
- [AURA 2.0 BF8](https://huggingface.co/aurateam/AURA_2_BF8)：下载后通过 `--aura` 指定本地目录。
- [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)：自动下载，或使用 `--asr` 指定本地目录。
- [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)：自动下载，或使用 `--tts` 指定本地目录。
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)：可选记忆摘要模型，默认不开启。

## 快速开始

以下命令在 accelerators 服务器上运行。

### 1. 克隆仓库

```bash
git clone https://github.com/aurateam2026/Aura2.0.git
cd Aura2.0
```

### 2. 指定并校验权重

将下面的路径替换成自己的 AURA 2.0 权重目录。BF8 版本使用相同命令。

```bash
export AURA_CKPT=/absolute/path/to/aura-2-checkpoint
python3 deploy/lib/check_ckpt.py "$AURA_CKPT"
```

### 3. 启动服务

```bash
# Accelerators 0 运行 AURA，accelerators 1 运行 ASR/TTS；显式选择 Serena 音色。
bash deploy/aura.sh up --aura "$AURA_CKPT" --gpus 0,1 --speaker Serena
```

首次运行自动安装环境和预编译内核、下载 ASR/TTS、生成配置、启动后端与页面，并执行冒烟测试。首次下载与编译需要等待，文件默认放在 `deploy/runtime/`。

使用提供的单卡配置：

```bash
bash deploy/aura.sh up --aura "$AURA_CKPT" --gpus 0 --speaker Serena
```

两种布局选择一种运行。切换现有部署时先 `stop`，再用新参数执行 `up`。共享服务器上先确认 accelerators 可用；`--gpus` 使用物理 accelerators 编号。

### 4. 打开页面

浏览器与 accelerators 在同一台机器时，打开 **<http://localhost:9310/>**。

如果浏览器在笔记本上，在**笔记本终端**运行以下命令，通过 SSH 直接连接运行 AURA 的加速器服务器：

```bash
ssh -N -L 9310:127.0.0.1:9310 your-user@your-server
```

将 `your-user` 替换为 SSH 用户名，`your-server` 替换为加速器服务器的 IP 地址或主机名。保留 `127.0.0.1`：这里指加速器服务器自身，SSH 会连接该服务器上 AURA 的 `9310` 端口。此命令假设 SSH 连接的就是运行 AURA 的服务器；如果集群要求经过登录节点，请将登录节点作为 SSH 跳板，连接到加速器服务器。

保持 SSH 会话运行，然后在笔记本打开 **<http://localhost:9310/>**。这里的 `localhost` 指笔记本自身，SSH 隧道会将笔记本的 `9310` 端口转发到加速器服务器上的 AURA。

手机或公网访问需要配置支持 WebSocket 的 HTTPS 反向代理，然后打开对应的 **HTTPS 域名或服务器地址**，见 [远程访问说明](docs/DEPLOYMENT_NOTES.zh-CN.md#external-access)。直接访问远程 IP 的普通 HTTP 页面，不能获得摄像头和麦克风所需的安全上下文。

### 5. 使用页面

1. 点击 **Allow access**，授权摄像头和麦克风后自动接通。
2. 连接成功后，按住 **Hold to talk**，说完松开；也可以打字或圈画提问。
3. 首个视频包发出后再发送文字问题。
4. 点击 **Hang up** 结束并释放摄像头/麦克风；点击 **Call again** 重新接通。

## Docker 部署（包含运行环境）

仓库提供可构建的 Dockerfile，暂未发布可直接拉取的 AURA 2.0 公共镜像。镜像安装 Python 3.12、固定的 vLLM 0.33 wheel、本仓库 vLLM-Omni 和页面依赖；模型权重通过挂载或首次启动时下载。

宿主机先安装 Docker Engine，并配置容器的 accelerators 访问权限，在仓库根目录执行：

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

Docker 的 `--gpus all` 负责让容器看到 accelerators；后面的 AURA 参数 `--gpus 0,1` 决定实际使用哪两张卡。首次启动下载 ASR/TTS 和缺失的预编译内核。Python 环境位于镜像内；模型、缓存、日志和配置保存在宿主机 `docker-runtime/`，容器重建后可继续使用。

浏览器访问与本机部署相同：本机或 SSH 隧道使用 `http://localhost:9310/`，手机远程访问使用 HTTPS。

[完整 Docker 说明](deploy/docker/README.zh-CN.md) 包含宿主机配置、镜像环境清单、离线权重挂载、单/双卡选择、memory、启停和验证范围。

## 可选联网工具

本地视频理解、语音识别和语音生成不需要 DeepSeek Key。

| 工具 | 依赖与行为 |
| --- | --- |
| 当前时间 | 公共时间服务，失败时使用服务器时钟；时区为 Asia/Shanghai |
| 大致位置 | 根据**服务器出口公网 IP** 查询，使用的不是手机 GPS |
| 天气 | Open-Meteo，本 demo 不需要为它配置 Key |
| DeepSeek 文本与图像工具 | `DEEPSEEK_API_KEY`，以及相应模型/功能的服务商权限 |

没有配置 Key 时，DeepSeek 工具仍会出现在模型可见的列表里，调用时返回配置错误。服务商的部分模型或功能可能需要额外权限。

```bash
mkdir -p deploy/runtime
cp tools/tool_keys.env.example deploy/runtime/tool_keys.env
chmod 600 deploy/runtime/tool_keys.env
# 在本机编辑文件，填入 DEEPSEEK_API_KEY；格式为 NAME=value，不加引号或 export。
bash deploy/aura.sh restart
```

默认位置会自动加载，其他位置通过 `--env-file /absolute/path/to/tool_keys.env` 指定。当前 gateway 不使用 Serper Key。调用 DeepSeek 图像工具时，选中的画面和问题会发送给对应服务商。

## 日常管理

安装完成后，已停止的服务用 `start` 启动，正在运行的服务用 `restart` 重启。
脚本复用保存的模型、设备和端口设置，无需激活虚拟环境。`start` 默认执行冒烟测试，
可用 `--no-test` 跳过；`restart` 不执行冒烟测试，重启后可单独运行 `test`。

```bash
bash deploy/aura.sh start
bash deploy/aura.sh status
bash deploy/aura.sh test
bash deploy/aura.sh logs
bash deploy/aura.sh stop
bash deploy/aura.sh restart
```

`--home` 可指定独立运行目录，后续管理命令也要传入相同的 `--home`。`--port` 修改页面端口，`--backend-port` 修改后端端口。日志默认保存在 `deploy/runtime/logs/`。

启动脚本还会在 `deploy/runtime/user_context/` 保存逐轮上下文，可能包含用户对话、工具结果和位置信息，请将运行目录保存在本地。Demo 没有内置用户认证；向外提供访问时，应在反向代理层限制访问权限。

## Memory：在哪里配置、如何开关

Memory 默认关闭。它使用单独的 Qwen3-VL 模型，把**当前会话**中较早的对话和视频压缩成文字摘要；它不是跨会话永久记忆。关闭此功能仍有正常的会话历史。

**本机部署的配置文件是 `deploy/runtime/memory.env`。** `deploy/aura.sh` 会读取它，把导出的环境变量传给后端。不要把这些变量写进只供页面工具使用的 `tool_keys.env`。

首次配置执行下面的命令；如果已有 `memory.env`，直接编辑原文件，保留自己的 accelerators、端口设置：

```bash
mkdir -p deploy/runtime
cp deploy/memory.env.example deploy/runtime/memory.env
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=1/' deploy/runtime/memory.env
bash deploy/aura.sh restart
```

这里的 `restart` 适用于已经成功部署的 demo，会沿用此前的模型和 accelerators。首次部署请使用前面的 `up --aura ... --gpus 0,1 --speaker Serena`。

文件中的关键配置如下，`export` 需要保留：

```bash
export AURA_MEMORY_ENABLE=1
export MEMORY_VLM_GPU=2
export MEMORY_VLM_PORT=8003
export AURA_MEMORY_VLM_URL=http://127.0.0.1:8003/v1/chat/completions
```

`AURA_MEMORY_VLM_URL` 是**记忆摘要模型的接口地址**。本例启动脚本会先在本机 accelerators 2 上启动 Qwen3-VL，监听端口 8003，再启动 AURA。AURA 后端会将旧对话和抽样的视频帧发给该接口，取回摘要。它不指向 demo 网页，也不指向 AURA 主模型；本地服务无需 OpenAI Key。

URL 中的端口要与 `MEMORY_VLM_PORT` 一致，模板中的两个 served-model 名称也要一致。当前一键脚本要求 memory 独占一张不同于 demo 的 accelerators：本例共用三张卡。只改 URL 不会让脚本自动变成仅使用远端服务的模式。

**关闭 memory：**

```bash
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=0/' deploy/runtime/memory.env
bash deploy/aura.sh restart
```

**Docker 部署：**配置文件放在宿主机 `docker-runtime/memory.env`，容器中对应 `/var/lib/aura/memory.env`。复制同一个模板，修改开关后执行 `docker restart aura2`。AURA 与摘要服务在同一个容器，默认的 `127.0.0.1:8003` 不需要改。

若使用 `--home /some/runtime`，则文件应放在 `/some/runtime/memory.env`，重启命令也要带相同的 `--home`。修改配置后必须重启后端才能生效。

完整步骤、配置解释和验证方法见 [Memory 使用说明](docs/MEMORY.zh-CN.md)。

## 常见问题

| 问题 | 排查 |
| --- | --- |
| 无法授权摄像头/麦克风 | 使用 HTTPS 或 localhost，检查浏览器权限 |
| 权重校验失败 | 检查是否为完整 AURA 2.0 权重，以及 tokenizer/processor 是否匹配 |
| 设备内存检查失败或 OOM | 释放目标 accelerators 的内存或选择双设备布局；量化权重不会自动修改内存预留比例 |
| 运行时 / FlashInfer 加载失败 | 检查驱动兼容性和 `deploy/runtime/logs/backend.log` |
| DeepSeek 工具失败 | 检查 Key、网络，以及服务商模型/功能权限 |
| 查到的城市不对 | IP 定位对应服务器出口，查天气时明确提供城市 |
| 端口被占用 | 使用 `--port` / `--backend-port` 更换端口 |

## 文档与许可

- [完整部署参考](deploy/README.zh-CN.md)
- [Docker 环境与部署](deploy/docker/README.zh-CN.md)
- [Memory 配置与验证](docs/MEMORY.zh-CN.md)
- [端口、加速卡布局、远程访问与验证范围](docs/DEPLOYMENT_NOTES.zh-CN.md)
- [架构及会话记忆](docs/ARCHITECTURE.zh-CN.md)
- [上游 vLLM-Omni README](README.upstream.md)

源码采用 [Apache License 2.0](LICENSE)，保留上游及第三方声明。项目基于 vLLM-Omni/vLLM，并使用 Qwen ASR/TTS。模型权重单独下载，适用各自的许可证，详见 [第三方说明](THIRD_PARTY_NOTICES.md)。

新图标来自 [AURA 2.0 项目](https://github.com/aurateam2026/Aura-2.0/tree/main/logo)。
