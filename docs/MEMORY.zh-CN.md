# AURA 2.0 Memory 使用说明

[English](MEMORY.md) · [简体中文](MEMORY.zh-CN.md)

Memory 是当前会话的后台摘要功能，默认关闭。AURA 主模型继续处理实时视频和当前问题；另一个 Qwen3-VL 模型将较早的视频及对话压缩成文字，再把摘要写回同一会话的上下文。

它不是跨用户共享的数据库，也不会自动把上一通电话的记忆恢复到新会话。关闭摘要功能后仍会保留正常的对话历史。

## 配置文件到底放在哪里

以下路径相对于仓库根目录：

| 部署方式 | 在哪里编辑 | 如何生效 |
|---|---|---|
| 本机默认部署 | `deploy/runtime/memory.env` | `bash deploy/aura.sh restart` |
| 本机自定义 `--home /srv/aura-runtime` | `/srv/aura-runtime/memory.env` | `bash deploy/aura.sh restart --home /srv/aura-runtime` |
| 本仓库 Docker 部署 | 宿主机 `docker-runtime/memory.env`，映射为容器 `/var/lib/aura/memory.env` | `docker restart aura2` |

`deploy/aura.sh` 在启动后端之前读取运行目录里的 `memory.env`。
文件中的 `export` 使变量能传入后端的工作进程；只写 `AURA_MEMORY_ENABLE=1` 而不导出，可能使脚本启动了摘要服务，但模型进程没有启用摘要。
不要写到 `tool_keys.env`：那个文件仅用于 gateway 的联网工具配置。

## 开启：以 demo 用 accelerators 0、1，memory 用 accelerators 2 为例

先完成一次正常的 AURA 部署。以下为首次创建配置文件的命令；已有文件时直接编辑它，不覆盖其他设置。

```bash
mkdir -p deploy/runtime
cp deploy/memory.env.example deploy/runtime/memory.env
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=1/' deploy/runtime/memory.env
bash deploy/aura.sh restart
```

模板完整内容可在 [deploy/memory.env.example](../deploy/memory.env.example) 查看。
一键脚本要求 memory accelerators 与 `--gpus` 指定的 demo accelerators 不重叠。accelerators 2 是例子，不是固定要求；根据自己的机器修改 `MEMORY_VLM_GPU`。

如果还没启动过 demo，配置完成后用首次启动命令：

```bash
bash deploy/aura.sh up --aura /absolute/path/to/aura-2-checkpoint --gpus 0,1 --speaker Serena
```

脚本会依次启动 Qwen3-VL、等待其接口就绪、启动 AURA 后端和网页。Qwen3-VL 首次需要下载权重，显存使用预算默认是所选 accelerators 的 30%；显存不足时不能靠降低比例保证模型能装下。

## `AURA_MEMORY_VLM_URL` 是什么

默认值：

```bash
export AURA_MEMORY_VLM_URL=http://127.0.0.1:8003/v1/chat/completions
```

- `127.0.0.1`：运行 AURA 后端的本机；本仓库 Docker 方案中就是同一个容器。
- `8003`：Qwen3-VL 摘要服务的端口，与 `MEMORY_VLM_PORT` 对应。
- `/v1/chat/completions`：接收文字和图像、返回摘要的 OpenAI 兼容接口路径。这里只是接口格式相同，本地服务不需要 OpenAI 账号或 Key。

AURA 后端把旧对话和抽样视频帧 POST 到这个地址。`9310` 是网页端口、`8099` 是 AURA 后端端口，两者都不是此默认摘要接口。

端口改变时同时修改两行，例如：

```bash
export MEMORY_VLM_PORT=8013
export AURA_MEMORY_VLM_URL=http://127.0.0.1:8013/v1/chat/completions
```

当前一键脚本在 `AURA_MEMORY_ENABLE=1` 时总会启动本地摘要模型。仅修改 URL 不会跳过这个步骤，因此本指南采用本地托管服务。底层客户端也没有发送鉴权请求头，不应把带 Key 的云服务地址直接填进来并期待它自动完成认证。

## 模型路径与模型名

默认模板：

```bash
export MEMORY_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
export MEMORY_VLM_SERVED_NAME=Qwen/Qwen3-VL-8B-Instruct
export AURA_MEMORY_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

`MEMORY_VLM_MODEL` 决定加载什么权重，可以是 Hugging Face ID，也可以是本地完整权重目录。
`MEMORY_VLM_SERVED_NAME` 是服务对外暴露的名字，`AURA_MEMORY_VLM_MODEL` 是请求 JSON 里填写的名字，两者必须一致。只修改本地权重路径时，可保持两个名字不变。

Docker 中本地模型路径必须是容器内已经挂载的路径，不能填容器无法访问的宿主机路径。

## 其他参数

| 变量 | 默认 | 含义 |
|---|---|---|
| `AURA_MEMORY_ENABLE` | `0` | `1` 开启后台摘要，`0` 关闭 |
| `MEMORY_VLM_GPU` | `2` | 托管摘要模型所用的独立 accelerators |
| `MEMORY_VLM_PORT` | `8003` | 本机摘要服务端口 |
| `MEMORY_VLM_GPU_UTIL` | `0.3` | vLLM 的 accelerators 显存预算比例 |
| `MEMORY_VLM_WAIT_S` | `1800` | 等待摘要服务启动的最长秒数 |
| `AURA_MEMORY_INTERVAL_S` | `30` | 每隔多少秒检查是否需要摘要 |
| `AURA_MEMORY_MIN_ROUNDS` | `6` | 至少多少个完整历史轮次才开始检查摘要 |
| `AURA_MEMORY_KEEP_ROUNDS` | `4` | 最近多少轮保留原内容 |
| `AURA_MEMORY_MAX_FRAMES` | `8` | 单次摘要最多发送多少张抽样视频帧 |
| `AURA_MEMORY_TIMEOUT_S` | `30` | 单次摘要请求的超时秒数 |

达到 6 轮不代表立即发生摘要：还需等检查周期、历史中有可摘要的视频片段，且当前没有尚未完成的用户轮次。工具轮不会从中间拆开。摘要接口报错或返回空内容时，已有历史保留，后续周期重试。

## 怎样确认真正生效

本机部署：

```bash
bash deploy/aura.sh status
curl -f http://127.0.0.1:8003/v1/models
grep '\[memory\]' deploy/runtime/logs/backend.log
tail -n 50 deploy/runtime/logs/memory_vlm.log
```

Docker 部署：

```bash
docker exec aura2 bash deploy/aura.sh status
docker exec aura2 curl -f http://127.0.0.1:8003/v1/models
docker exec aura2 grep '\[memory\]' /var/lib/aura/logs/backend.log
```

改过端口时，相应替换上面的 `8003`。`/v1/models` 返回正确模型名表示服务就绪；这并不能单独证明已经生成摘要。进行足够轮数的视频对话后，后端日志应出现 `[memory] loop started` 和 `[memory] summarizing`。如果出现 `[memory] VLM call failed`，继续检查地址、模型名、网络和摘要模型日志。

## 关闭与重新开启

将配置文件里的开关改为 0，重启后端；本机例子：

```bash
sed -i 's/^export AURA_MEMORY_ENABLE=.*/export AURA_MEMORY_ENABLE=0/' deploy/runtime/memory.env
bash deploy/aura.sh restart
```

这会先停止原来的 demo 和托管 memory 服务，再在关闭摘要的状态下启动 demo。无需删除模型或缓存。
重新开启将 `0` 改回 `1`，再次重启。Docker 对应修改 `docker-runtime/memory.env` 并执行 `docker restart aura2`。

重启会结束正在进行的通话和内存中的会话上下文，因此需重新接通。它不会删除此前写入磁盘的上下文日志。
