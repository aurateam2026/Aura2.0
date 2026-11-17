# Docker 部署 AURA 2.0

[English](README.md) · [简体中文](README.zh-CN.md)

所有命令从仓库根目录执行。本目录提供完整环境的构建配方，当前没有已发布的公共 AURA 2.0 镜像地址。

## 镜像里有什么

| 内容 | 位置或版本 |
| --- | --- |
| Linux / Python | Debian Bookworm / Python 3.12 |
| uv | 0.9.8 |
| vLLM | 0.33.0，固定 wheel，与本机启动脚本一致 |
| PyTorch / accelerators 运行时依赖 | 随固定 vLLM wheel 的依赖安装 |
| vLLM-Omni / gateway | 本仓库源码，安装在镜像 `/opt/aura` |
| Python 环境 | `/opt/aura-venv`，由 `AURA_VENV` 指定，启动时复用 |
| ffmpeg / curl / git / tini | 构建时安装 |
| FlashInfer 预编译内核 | 首次启动检查，缺少时自动下载 |
| AURA 权重 | 用户只读挂载到 `/models/aura` |
| ASR / TTS 权重 | 首次启动下载到持久化目录，或通过本地挂载提供 |
| Memory 权重 | 开启 memory 后下载 Qwen3-VL，或配置本地路径 |

宿主机需要兼容的 accelerators 驱动和容器运行时集成。容器提供用户态运行库，驱动由宿主机提供。较旧驱动仅在支持前向兼容的数据中心 accelerators 上通过启动脚本安装兼容库。具体设备和量化格式的支持范围见 [部署说明](../../docs/DEPLOYMENT_NOTES.zh-CN.md)。

## 1. 配置宿主机 accelerators 容器环境

先按 [Docker 官方安装说明](https://docs.docker.com/engine/install/) 安装 Docker Engine。
按驱动厂商的说明安装加速器容器工具包，并配置 Docker 使用该工具包。修改配置后重启 Docker。

用以下命令检查 Docker 能否启动启用了加速器访问的容器：

```bash
docker run --rm --gpus all ubuntu:22.04 true
```

该命令应正常退出。如果提示 `could not select device driver`，请检查容器工具包是否已安装并为 Docker 配置。

无需在宿主机另外创建 Python 环境。镜像构建阶段无需 accelerators，但需要访问镜像仓库、PyPI 和 `wheels.vllm.ai`；启动时还需要访问模型与内核下载站。

## 2. 构建包含依赖的镜像

```bash
docker build -f deploy/docker/Dockerfile -t aura2:local .
```

首次构建下载 PyTorch 和 accelerators 运行时等大依赖，磁盘应同时容纳镜像、构建缓存、模型和运行缓存，建议准备至少 100 GB 空间。构建结尾执行 `uv pip check` 核对依赖。模型和真实 Key 不应放进镜像；`deploy/docker/Dockerfile.dockerignore` 排除运行目录、凭据、模型权重和备份归档。

修改源码后重新 build；重建容器才会使用新镜像，`docker restart` 仍使用原镜像。

## 3. 指定权重并启动

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

- Docker `--gpus all` 让容器看到 accelerators；镜像名后的 `--gpus 0,1` 是 AURA 启动参数：accelerators 0 跑主模型，accelerators 1 跑 ASR/TTS。这里采用全部可见的配置来保留一致的 accelerators 编号。
- 单卡模板把后一个参数改为 `--gpus 0`；该模板面向 80 GB accelerators，显存预留和验证范围见部署说明。
- `--shm-size=16g` 为多进程通信预留容器共享内存。
- `--stop-timeout=180` 给多阶段服务留出退出时间。
- 默认音色为 Serena。没有前端配置页，授权摄像头和麦克风后自动进入通话。
- `docker-runtime/` 必须是此容器专用目录，不要挂载正在运行的本机 demo 的 `deploy/runtime/`。它包含进程状态、设置、模型缓存和对话日志。

首次启动下载 ASR/TTS 和缺少的内核，然后加载模型、启动页面并做冒烟测试。看到 `AURA is up` 后，打开 `http://localhost:9310/`。从笔记本访问 accelerators 服务器时，在笔记本执行：

```bash
ssh -N -L 9310:127.0.0.1:9310 your-user@your-server
```

手机或公网访问需在宿主机使用 HTTPS 反向代理，转发到 `127.0.0.1:9310`，同时支持 WebSocket。这里只映射网页端口；8099 主模型端口和 8003 memory 端口无需映射到公网。

## 4. Memory 的配置文件与开关

容器读取 `/var/lib/aura/memory.env`，它对应宿主机的 **`docker-runtime/memory.env`**。首次配置：

```bash
cp deploy/memory.env.example docker-runtime/memory.env
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=1/' docker-runtime/memory.env
docker restart aura2
```

已有配置文件时直接编辑，不要覆盖自己设置的端口、accelerators 或本地模型路径。默认模板要求 memory 使用 accelerators 2，而 AURA 使用 accelerators 0、1；需额外预留 accelerators 2 的显存。只有两张卡时，可选择单卡 demo 加独立 memory 卡，并修改模板中的 `MEMORY_VLM_GPU`。

AURA 和 Qwen3-VL 在同一个容器，保持：

```bash
export MEMORY_VLM_PORT=8003
export AURA_MEMORY_VLM_URL=http://127.0.0.1:8003/v1/chat/completions
```

这个 URL 是**摘要模型接口**，不是网页地址，也不是 AURA 主模型地址。无需配置 OpenAI Key。

关闭：

```bash
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=0/' docker-runtime/memory.env
docker restart aura2
```

验证接口及摘要日志：

```bash
docker exec aura2 curl -f http://127.0.0.1:8003/v1/models
docker exec aura2 bash deploy/aura.sh status
# 完成足够轮数的视频对话后查看：
docker exec aura2 grep '\[memory\]' /var/lib/aura/logs/backend.log
```

每轮检查周期、最少轮数、保留轮数及其他说明见 [Memory 使用说明](../../docs/MEMORY.zh-CN.md)。

## 5. 使用本地 ASR/TTS 权重

在上述 `docker run` 的镜像名之前再添加两个挂载：

```text
--mount type=bind,src=/absolute/path/to/Qwen3-ASR-1.7B,dst=/models/asr,readonly
--mount type=bind,src=/absolute/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice,dst=/models/tts,readonly
```

在镜像名之后的 AURA 参数中再添加：

```text
--asr /models/asr --tts /models/tts
```

本地 memory 权重同样挂载到容器路径，并在 `memory.env` 中将 `MEMORY_VLM_MODEL` 改为该容器路径；两个 served-model 名称仍保持一致。使用本地权重不会自动消除首次下载 FlashInfer 内核的需要；完全离线部署应提前准备这些依赖。

## 6. 可选工具 Key

```bash
cp tools/tool_keys.env.example docker-runtime/tool_keys.env
chmod 600 docker-runtime/tool_keys.env
# 在本机编辑该文件，填入 Key，使用 NAME=value 格式，不加 export 或引号。
docker restart aura2
```

工具文件与 memory 文件读取方式不同：`tool_keys.env` 由 gateway 读取，`memory.env` 是启动脚本读取的 shell 文件，后者需要保留 `export`。

## 7. 启停、更新与排错

```bash
docker logs -f aura2
docker exec aura2 bash deploy/aura.sh status
docker exec aura2 bash deploy/aura.sh test
docker stop aura2
docker start aura2
docker restart aura2
```

更换镜像或挂载配置时，先 `docker stop aura2`、`docker rm aura2`，再用相同的 `docker-runtime/` 执行新的 `docker run`。移除容器不会删除这个宿主机目录。容器入口会复用镜像内的环境，清理上次容器留下的 PID，启动服务并在退出时停止其进程；不要在容器内部运行 `aura.sh stop/restart`，应使用 Docker 的对应命令。

| 现象 | 检查 |
| --- | --- |
| accelerators 不可见 | 重新执行上面的容器启动检查，并检查容器工具包配置 |
| 启动显存不足 | 检查选中的卡及当前占用；不要覆盖别的服务 |
| Camera / microphone 不可用 | 需要 HTTPS 或 localhost，而非远程 IP 的普通 HTTP |
| Memory 启动失败 | 检查独立 accelerators、`docker-runtime/logs/memory_vlm.log`、模型路径和端口 |
| 无法访问外部下载站 | 宿主机/容器网络与代理配置；缓存和模型可以持久化复用 |
| 容器退出 | `docker logs aura2` 与 `docker-runtime/logs/`；backend/gateway 退出会使容器结束 |

验证范围记录在 [部署说明](../../docs/DEPLOYMENT_NOTES.zh-CN.md#validation)。构建和依赖检查不等同于完整 accelerators 音视频推理验证。
