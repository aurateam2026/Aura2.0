#!/usr/bin/env bash
# Memory-summarization VLM: an OpenAI-compatible vLLM endpoint used by
# vllm_omni/.../duplex/memory.py to compress old conversation turns.
# Runs on its own accelerator; start it before the demo backend when
# AURA_MEMORY_ENABLE=1.  Requires the runtime venv from deploy/aura.sh.
#   bash deploy/memory_vlm.sh --gpu 2 [--port 8003] [--home DIR]
set -euo pipefail
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$KIT_DIR/lib/common.sh"

GPU="${GPU:-2}"
PORT="${MEMORY_VLM_PORT:-8003}"
MODEL="${MEMORY_VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
GPU_UTIL="${MEMORY_VLM_GPU_UTIL:-0.3}"
SERVED_NAME="${MEMORY_VLM_SERVED_NAME:-$MODEL}"
WAIT_S="${MEMORY_VLM_WAIT_S:-1800}"
AURA_HOME="${AURA_HOME:-$KIT_DIR/runtime}"
PY="${AURA_VENV:-$AURA_HOME/venv}/bin/python"
LOG_DIR="$AURA_HOME/logs"
RUN_DIR="$AURA_HOME/run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --home) AURA_HOME="$2"; PY="${AURA_VENV:-$AURA_HOME/venv}/bin/python"; LOG_DIR="$AURA_HOME/logs"; RUN_DIR="$AURA_HOME/run"; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
[[ -x $PY ]] || die "runtime python not found: $PY (run deploy/aura.sh install first)"

# Reuse the CUDA 13 compatibility libraries installed by aura.sh on older drivers.
compat="$AURA_HOME/cuda-compat/usr/local/cuda-13.0/compat"
if [[ -f $compat/libcuda.so.1 && ${LD_LIBRARY_PATH:-} != *"$compat"* ]]; then
  export LD_LIBRARY_PATH="$compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if pid_alive "$RUN_DIR/memory_vlm.pid"; then
  http_ok "http://127.0.0.1:$PORT/v1/models" || die "memory VLM process exists but is not ready; inspect $LOG_DIR/memory_vlm.log"
  ok "memory VLM already running on :$PORT"
  exit 0
fi
port_in_use "$PORT" && die "memory port $PORT is already occupied"
read -r free total < <(nvidia-smi -i "$GPU" --query-gpu=memory.free,memory.total --format=csv,noheader,nounits | tr ',' ' ')
need=$(awk -v total="$total" -v util="$GPU_UTIL" 'BEGIN {printf "%d", total * util}')
[[ $free -ge $need ]] || die "memory accelerator $GPU has $free MiB free; needs at least $need MiB"

export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$LOG_DIR" "$RUN_DIR"
log "memory VLM: model=$MODEL accelerator=$GPU port=$PORT util=$GPU_UTIL"
run_detached "$RUN_DIR/memory_vlm.pid" "$LOG_DIR/memory_vlm.log" \
  "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$SERVED_NAME" \
  --port "$PORT" --host 127.0.0.1 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len 16384 --max-num-seqs 2 \
  --dtype auto --trust-remote-code \
  --limit-mm-per-prompt '{"image": 32}'
for ((i = 0; i < WAIT_S; i += 2)); do
  http_ok "http://127.0.0.1:$PORT/v1/models" && { ok "memory VLM ready on :$PORT"; exit 0; }
  pid_alive "$RUN_DIR/memory_vlm.pid" || { tail -30 "$LOG_DIR/memory_vlm.log" >&2; die "memory VLM exited"; }
  (( i % 30 != 0 )) || log "memory VLM loading... ${i}s"
  sleep 2
done
die "memory VLM not ready after ${WAIT_S}s (log: $LOG_DIR/memory_vlm.log)"
