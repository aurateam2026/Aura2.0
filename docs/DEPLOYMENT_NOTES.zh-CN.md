# 部署说明

[English](DEPLOYMENT_NOTES.md) · [简体中文](DEPLOYMENT_NOTES.zh-CN.md)

<a id="ports"></a>

## 端口

| 端口 | 进程 |
| --- | --- |
| 9310 | 网关页面（`GATEWAY_PORT`） |
| 8099 | vllm-omni 实时后端（`BACKEND_PORT`） |
| 8003 | 可选的记忆摘要 VLM（`AURA_MEMORY_VLM_URL`） |

<a id="accelerator-layouts"></a>

## 加速卡布局

| 布局 | 设备分配 | 说明 |
| --- | --- | --- |
| 双卡 | 第一张卡运行 AURA，第二张卡运行 ASR + Talker + Code2Wav | 推荐 |
| 单卡 | 四个阶段运行在同一张卡上 | 需要约 70 GB 可用显存 |
| 记忆摘要 VLM | 使用一张独立于 Demo 加速卡的卡 | 由启动脚本管理 Qwen3-VL-8B 服务，默认 `gpu_memory_utilization=0.3` |

<a id="tts-voice"></a>

## TTS 音色

通过 `--speaker`（或 `AURA_TTS_SPEAKER`）选择 CustomVoice 音色：Vivian（默认）、Serena、Uncle_Fu、Ryan、Aiden、Ono_Anna、Sohee、Eric（四川话）、Dylan（北京话）。`AURA_TTS_LANGUAGE` 默认使用中文。

<a id="external-access"></a>

## 远程访问

页面监听 `0.0.0.0:9310`。可以按需选择以下访问方式：

- 本机浏览器：打开 `http://localhost:9310/`。通过远程 IP 访问普通 HTTP 页面无法使用摄像头和麦克风。
- SSH 隧道：运行 `ssh -N -L 9310:127.0.0.1:9310 <host>`。
- 直接使用 HTTPS：将 `cert.pem` 和 `key.pem` 放入 `examples/online_serving/aura_v2_gateway/gateway/`，然后启动服务。网关检测到这两个文件后会自动启用 HTTPS。可以用 `openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 3650 -subj "/CN=aura-gateway"` 生成自签名证书。实际访问时应使用浏览器或设备信任的证书。手机访问通常更适合使用域名和带可信证书的 HTTPS 反向代理。
- 公网访问：在网关前配置 nginx 或 frp/Cloudflare 隧道。麦克风采集需要安全上下文，因此应使用 HTTPS/WSS 或 `localhost`。

<a id="memory-configuration"></a>

## 记忆摘要配置

将带 `export` 的配置变量写入 `deploy/runtime/memory.env`（或 `--home` 指定目录中的 `memory.env`）。Docker 部署时，编辑宿主机的 `docker-runtime/memory.env`；它在容器内挂载为 `/var/lib/aura/memory.env`。

修改后重启后端或容器。`AURA_MEMORY_VLM_URL` 指向独立的 Qwen3-VL 摘要接口，不是网页地址或 AURA 主模型接口。完整步骤见 [Memory 使用说明](MEMORY.zh-CN.md) 和 [配置模板](../deploy/memory.env.example)。

<a id="environment-variables"></a>

## 环境变量

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `AURA_TTS_SPEAKER` | `Vivian` | Talker 音色 |
| `AURA_TTS_LANGUAGE` | 空（中文） | Talker 语言 |
| `AURA_GATEWAY_TRACE` | `0` | 是否在网关中记录每一帧 |
| `AURA_WEATHER_FORECAST_URL` | `https://api.open-meteo.com/v1/forecast` | 可选的 Open-Meteo 兼容实时天气/预报接口；在网关环境变量或 `--env-file` 指定的文件中设置 |
| `AURA_MEMORY_ENABLE` | `0` | 是否启用后台记忆摘要 |
| `AURA_MEMORY_VLM_URL` | `http://127.0.0.1:8003/v1/chat/completions` | 记忆摘要 VLM 接口 |
| `AURA_MEMORY_INTERVAL_S` | `30` | 后台循环检查间隔，单位为秒 |
| `AURA_MEMORY_MIN_ROUNDS` | `6` | 检查是否需要摘要的最低对话轮数 |
| `AURA_MEMORY_KEEP_ROUNDS` | `4` | 保留原文、不参与摘要的最近对话轮数 |
| `AURA_MEMORY_MAX_FRAMES` | `8` | 每次摘要最多发送的视频帧数 |

<a id="docker"></a>

## Docker

[Docker 部署说明](../deploy/docker/README.zh-CN.md) 包含镜像环境构建、宿主机加速卡容器配置、模型挂载、运行配置持久化、加速卡选择、记忆摘要和容器生命周期管理命令。`AURA_VENV` 用于选择镜像内预装的 Python 环境；宿主机安装默认使用运行目录中的虚拟环境。

<a id="validation"></a>

## 验证范围

- 已有的加速卡 Demo：量化 AURA、ASR 和 TTS 已在两张 80 GB 加速卡上运行验证。
- Docker 构建：镜像已构建，`uv pip check` 已通过。容器内的安装命令复用了镜像环境，安装了运行时兼容库和 FlashInfer 内核，并确认 PyTorch 2.14 能识别宿主机加速卡。本地 ASR/TTS 模型挂载通过了安装检查。
- 容器生命周期：使用独立的模拟进程检查了失效 PID 清理、运行目录互斥锁、`docker stop` 清理，以及网关失败后的容器退出行为；这些测试不加载模型权重。
- 尚未在这个新 Docker 镜像中完成完整的音视频对话测试。验证时可用加速卡正在运行其他任务，文档更新期间没有停止或替换运行中的 Demo。
- 单卡模板面向 80 GB 加速卡。本次更新不代表已经完成单卡部署的完整测试，也不代表已确认支持其他加速卡型号。

镜像和仓库均不包含模型权重。模型仓库：[AURA 2.0](https://huggingface.co/aurateam/AURA_2) 和 [AURA 2.0 BF8](https://huggingface.co/aurateam/AURA_2_BF8)。

<a id="weather-connectivity"></a>

## 天气接口连通性

城市地理编码查询和天气预报使用不同的主机。能够查到城市，不代表服务器能够访问 `api.open-meteo.com`。需要时，将 `AURA_WEATHER_FORECAST_URL` 设置为可访问的 Open-Meteo 兼容接口，并重启网关；同时确认该接口覆盖需要查询的日期。历史日期仍使用 `archive-api.open-meteo.com`。请求失败会以错误状态反馈给页面和模型。

如果网关证书由私有证书颁发机构签发，在执行 `start` 或 `test` 时设置 `SSL_CERT_FILE=/path/to/ca.crt`。网关启用 TLS 后，应通过 HTTPS 和证书覆盖的主机名访问。
