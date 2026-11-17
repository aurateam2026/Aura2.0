#!/usr/bin/env bash
# One-click AURA_v2 + vllm-omni (duplex) deployment.
#
#   bash aura.sh up --aura /path/to/aura_v2_checkpoint      # install + start + smoke test
#   bash aura.sh status | stop | restart | test | logs
#
# The serving tree is this repository itself (the script lives in deploy/).
# Everything else (uv, Python 3.12, venv, ASR/TTS weights, logs) lives under
# --home (default: ./runtime next to this script). No root and no Docker
# needed. See README.md.
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$KIT_DIR/lib/common.sh"

# ---------------------------------------------------------------- defaults
# The vllm-omni tree ships inside this repository; nothing is cloned.
OMNI_ROOT="$(cd "$KIT_DIR/.." && pwd)"
# vLLM 0.33.0 (cu130) wheel the branch is built against (docs/getting_started/installation/gpu/cuda.inc.md).
VLLM_WHEEL="${VLLM_WHEEL:-https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl}"
ASR_REPO="${ASR_REPO:-Qwen/Qwen3-ASR-1.7B}"
TTS_REPO="${TTS_REPO:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
MIN_DRIVER_FOR_CUDA13=580

AURA_HOME="${AURA_HOME:-$KIT_DIR/runtime}"
AURA_CKPT="${AURA_CKPT:-}"
ASR_MODEL="${ASR_MODEL:-}"
TTS_MODEL="${TTS_MODEL:-}"
GPUS="${GPUS:-auto}"
BACKEND_PORT="${BACKEND_PORT:-8099}"
GATEWAY_PORT="${GATEWAY_PORT:-9310}"
SERVED_NAME="${SERVED_NAME:-}"
TTS_SPEAKER="${TTS_SPEAKER:-}"
ENV_FILE="${ENV_FILE:-}"
RUN_TEST=1
FORCE=0
WAIT_S="${WAIT_S:-1800}"
DRIVER_MAJOR=""
LAYOUT=""
GPU_IDS=""
CLI_SET=()   # options given on this command line win over the saved state

usage() {
  cat <<EOF
Usage: bash aura.sh <command> [options]

Commands
  up         install (if needed) + start + smoke test   <- the one-click path
  install    set up uv, Python 3.12, vLLM 0.33, this tree, ASR/TTS weights
  start      start optional memory VLM, backend and web page
  stop       stop web page, backend and managed memory VLM
  restart    stop + start
  status     processes, health, accelerator memory
  test       end-to-end smoke test against the running page
  logs       follow backend + gateway logs

Options
  --aura PATH        AURA_v2 checkpoint directory (required for up/start the first time)
  --gpus auto|I|I,J  auto: 2 idle accelerators -> two-accelerator layout, else 1 idle 80 GB accelerator.
                     I    : all four stages on accelerator I (needs ~70 GB free)
                     I,J  : AURA on accelerator I, ASR + TTS on accelerator J
  --home DIR         install/runtime directory (default: $KIT_DIR/runtime)
  --port N           web page port (default 9310)
  --backend-port N   vllm-omni realtime port (default 8099)
  --name NAME        served model name (default: checkpoint directory name)
  --asr PATH         use a local Qwen3-ASR-1.7B instead of downloading
  --tts PATH         use a local Qwen3-TTS-12Hz-1.7B-CustomVoice instead of downloading
  --env-file FILE    extra env for the page's tools (e.g. DEEPSEEK_API_KEY=...)
  --speaker NAME     TTS voice: Vivian (default), Serena, Uncle_Fu, Ryan, Aiden,
                     Ono_Anna, Sohee, Eric (Sichuan), Dylan (Beijing)
  --no-test          skip the smoke test after start
  --force            start even if the chosen accelerators look too full
EOF
}

# ---------------------------------------------------------------- args
CMD="${1:-}"
[[ -z $CMD || $CMD == -h || $CMD == --help ]] && { usage; exit 0; }
shift
need_arg() { [[ $# -ge 2 && -n $2 ]] || die "$1 needs a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --aura) need_arg "$@"; CLI_SET+=("AURA_CKPT=$2"); shift 2 ;;
    --gpus) need_arg "$@"; CLI_SET+=("GPUS=$2"); shift 2 ;;
    --home) need_arg "$@"; AURA_HOME="$2"; shift 2 ;;
    --port) need_arg "$@"; CLI_SET+=("GATEWAY_PORT=$2"); shift 2 ;;
    --backend-port) need_arg "$@"; CLI_SET+=("BACKEND_PORT=$2"); shift 2 ;;
    --name) need_arg "$@"; CLI_SET+=("SERVED_NAME=$2"); shift 2 ;;
    --asr) need_arg "$@"; CLI_SET+=("ASR_MODEL=$2"); shift 2 ;;
    --tts) need_arg "$@"; CLI_SET+=("TTS_MODEL=$2"); shift 2 ;;
    --env-file) need_arg "$@"; CLI_SET+=("ENV_FILE=$2"); shift 2 ;;
    --speaker) need_arg "$@"; CLI_SET+=("TTS_SPEAKER=$2"); shift 2 ;;
    --no-test) RUN_TEST=0; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

mkdir -p "$AURA_HOME"
AURA_HOME="$(cd "$AURA_HOME" && pwd)"
STATE="$AURA_HOME/aura.env"
OMNI_DIR="$OMNI_ROOT"
VENV="${AURA_VENV:-$AURA_HOME/venv}"
PY="$VENV/bin/python"
LOG_DIR="$AURA_HOME/logs"
RUN_DIR="$AURA_HOME/run"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$AURA_HOME/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$AURA_HOME/python}"
export PATH="$AURA_HOME/bin:$PATH"
COMPAT_LIB=""
# The gateway serves HTTPS when cert.pem/key.pem sit next to duplex_app.py.
GW_URL="http://127.0.0.1:${GATEWAY_PORT}"
gw_scheme() {
  local gw="$OMNI_DIR/examples/online_serving/aura_v2_gateway/gateway"
  if [[ -f $gw/cert.pem && -f $gw/key.pem ]]; then echo https; else echo http; fi
}
ws_scheme() {
  if [[ $(gw_scheme) == https ]]; then echo wss; else echo ws; fi
}
page_ok() {
  local url
  url="$(gw_scheme)://127.0.0.1:$GATEWAY_PORT$1"
  if [[ $(gw_scheme) == https ]]; then
    curl -skf --max-time 3 "$url" >/dev/null 2>&1
  else
    curl -sf --max-time 3 "$url" >/dev/null 2>&1
  fi
}

# Settings from a previous `up/start` fill in anything not given now;
# options on this command line always win.
load_state() {
  if [[ -f $STATE ]]; then
    # shellcheck disable=SC1090
    source "$STATE"
  fi
  if [[ -f $AURA_HOME/memory.env ]]; then
    # Runtime memory settings must reach the backend's worker processes.
    # shellcheck disable=SC1091
    source "$AURA_HOME/memory.env"
  fi
  local kv
  for kv in "${CLI_SET[@]}"; do
    printf -v "${kv%%=*}" '%s' "${kv#*=}"
  done
  # A new checkpoint gets its own default name unless --name was given.
  if [[ " ${CLI_SET[*]} " == *" AURA_CKPT="* && " ${CLI_SET[*]} " != *" SERVED_NAME="* ]]; then
    SERVED_NAME=""
  fi
  return 0
}

compat_env() {
  # LD_LIBRARY_PATH for every process that touches CUDA.
  if [[ -n $COMPAT_LIB ]]; then
    echo "LD_LIBRARY_PATH=$COMPAT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  else
    echo "AURA_NO_COMPAT=1"
  fi
}

# ---------------------------------------------------------------- install steps
check_host() {
  step "Checking the host"
  [[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die "needs Linux x86_64 (found $(uname -s) $(uname -m))"
  have nvidia-smi || die "accelerator driver utility not found: install a compatible driver"
  have curl || die "curl is required"
  have git || die "git is required"
  local driver
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  DRIVER_MAJOR=${driver%%.*}
  ok "driver $driver, $(nvidia-smi -L | wc -l) accelerator(s) detected"
  local free_gb
  free_gb=$(df -Pk "$AURA_HOME" | awk 'NR==2 {print int($4/1024/1024)}')
  [[ $free_gb -ge 40 ]] || warn "only ${free_gb} GB free under $AURA_HOME (install needs ~25 GB, weights ~9 GB)"
}

ensure_uv() {
  if have uv; then ok "uv $(uv --version | awk '{print $2}')"; return; fi
  step "Installing uv into $AURA_HOME/bin"
  mkdir -p "$AURA_HOME/bin"
  retry 3 bash -c "curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR='$AURA_HOME/bin' INSTALLER_NO_MODIFY_PATH=1 UV_NO_MODIFY_PATH=1 sh" \
    || die "could not install uv (set https_proxy if the host has no direct internet)"
  have uv || die "uv installed but not on PATH"
  ok "uv $(uv --version | awk '{print $2}')"
}

ensure_repo() {
  step "vllm-omni tree"
  [[ -f $OMNI_DIR/pyproject.toml && -d $OMNI_DIR/vllm_omni ]] \
    || die "not a vllm-omni tree: $OMNI_DIR (deploy/ must live inside the repository)"
  local rev=nogit
  git -C "$OMNI_DIR" rev-parse --short HEAD >/dev/null 2>&1 \
    && rev=$(git -C "$OMNI_DIR" rev-parse --short HEAD)
  ok "$OMNI_DIR @ $rev (shipped with this repository)"
}

# vLLM 0.33 ships CUDA 13 (cu130) wheels, which need driver >= 580. On older
# datacenter drivers (e.g. 535) the vendor's forward-compatibility libraries make
# them work; unpack them under $AURA_HOME without root.
ensure_cuda_compat() {
  if [[ ${DRIVER_MAJOR:-0} -ge $MIN_DRIVER_FOR_CUDA13 ]]; then
    ok "driver supports CUDA 13 natively"
    return
  fi
  local dest="$AURA_HOME/cuda-compat" lib
  lib=$(find "$dest" -name 'libcuda.so.*' -path '*compat*' 2>/dev/null | head -1 || true)
  if [[ -z $lib ]]; then
    step "Driver $DRIVER_MAJOR < $MIN_DRIVER_FOR_CUDA13: fetching CUDA 13 forward-compatibility libraries"
    local distro base pkg
    distro=$(. /etc/os-release && echo "${ID}${VERSION_ID//./}")
    case "$distro" in
      ubuntu2004|ubuntu2204|ubuntu2404|debian12) base="https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64" ;;
      *) base="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
         warn "no driver package repository mapping for $distro; trying the ubuntu2204 package" ;;
    esac
    local listing
    listing=$(retry 3 curl -fsSL "$base/") || die "could not list $base (no internet / proxy?)"
    # e.g. cuda-compat-13-0_580.95.05-1ubuntu1_amd64.deb. Stay on the 13.0 (R580)
    # line: that is the combination verified on R535 hosts.
    pkg=$(grep -oE 'cuda-compat-13-0_[0-9.]+-[0-9A-Za-z.~+]+_amd64\.deb' <<<"$listing" | sort -uV | tail -1 || true)
    [[ -n $pkg ]] || die "cuda-compat-13-0 not found in $base; upgrade the driver to >= $MIN_DRIVER_FOR_CUDA13"
    mkdir -p "$dest"
    retry 3 curl -fsSL -o "$dest/$pkg" "$base/$pkg" || die "download failed: $base/$pkg"
    if have dpkg-deb; then
      dpkg-deb -x "$dest/$pkg" "$dest"
    else
      (cd "$dest" && ar x "$pkg" && tar xf data.tar.*) || die "cannot unpack $pkg (needs dpkg-deb or ar)"
    fi
    lib=$(find "$dest" -name 'libcuda.so.*' -path '*compat*' | head -1)
    [[ -n $lib ]] || die "unpacked $pkg but found no compat libcuda"
  fi
  COMPAT_LIB=$(dirname "$lib")
  ok "CUDA 13 compat libraries: $COMPAT_LIB (forward compatibility works on datacenter accelerators)"
}

ensure_venv() {
  local stamp="$AURA_HOME/.install-stamp" want revision
  revision=$(git -C "$OMNI_DIR" rev-parse HEAD 2>/dev/null || echo source-archive)
  want="ref=$revision vllm=$VLLM_WHEEL"
  if [[ -n ${AURA_VENV:-} ]]; then
    [[ -x $PY ]] || die "preinstalled AURA_VENV has no Python: $PY"
    ok "Using the preinstalled environment ($VENV)"
  elif [[ -x $PY && -f $stamp && $(cat "$stamp") == "$want" ]]; then
    ok "Python environment up to date ($VENV)"
  else
    step "Building the Python environment (vLLM 0.33 cu130 + vllm-omni); first time ~5-15 min"
    [[ -x $PY ]] || uv venv -q "$VENV" --python 3.12 --seed || die "uv venv failed"
    retry 3 uv pip install -q --python "$PY" --torch-backend=cu130 "vllm @ $VLLM_WHEEL" || die "installing vLLM failed"
    # Without git metadata visible to setuptools-scm the version falls back to an invalid 'dev'.
    retry 2 env VLLM_OMNI_VERSION_OVERRIDE=0.33.0.dev0 uv pip install -q --python "$PY" -e "$OMNI_DIR" || die "installing vllm-omni failed"
    echo "$want" >"$stamp"
  fi
  ensure_kernels
  local check
  check=$(env "$(compat_env)" PYTHONPATH= "$PY" -c "import torch, vllm, vllm_omni; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), vllm.__version__)" 2>/dev/null | tail -1) \
    || die "the environment does not import; rerun with a clean --home"
  read -r t_ver t_cuda t_ok t_n v_ver <<<"$check"
  [[ $t_ok == True ]] || die "torch $t_ver (CUDA $t_cuda) cannot use the accelerator. Driver $DRIVER_MAJOR: upgrade to >= $MIN_DRIVER_FOR_CUDA13 (forward compatibility only works on datacenter accelerators)"
  ok "torch $t_ver CUDA $t_cuda sees $t_n accelerator(s); vLLM $v_ver"
}

# Stage1 (AURA) uses the FlashInfer attention backend. Without a CUDA toolkit
# (nvcc + ninja) on the host, FlashInfer needs its precompiled kernels
# (flashinfer-cubin + flashinfer-jit-cache); `vllm download-kernels` installs
# exactly the versions that match this vLLM.
flashinfer_ready() {
  env "$(compat_env)" PYTHONPATH= "$PY" -c \
    "from vllm.utils.flashinfer import has_flashinfer; raise SystemExit(0 if has_flashinfer() else 1)" >/dev/null 2>&1
}

ensure_kernels() {
  if flashinfer_ready; then ok "FlashInfer kernels available"; return; fi
  step "Installing precompiled FlashInfer kernels (vllm download-kernels)"
  retry 2 env "$(compat_env)" PYTHONPATH= PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" "$VENV/bin/vllm" download-kernels \
    || die "vllm download-kernels failed (needs access to flashinfer.ai)"
  flashinfer_ready || die "FlashInfer kernels still unavailable; install a CUDA 13 toolkit (nvcc + ninja) or check flashinfer.ai access"
  ok "FlashInfer kernels available"
}

download_model() {
  local repo=$1 dir=$2 shift_files=$3
  if [[ -f $dir/.aura-complete ]]; then ok "$repo already downloaded ($dir)"; return; fi
  step "Downloading $repo -> $dir"
  local hf="$VENV/bin/hf"
  [[ -x $hf ]] || hf="$VENV/bin/huggingface-cli"
  # Xet transfers stall on some clusters; plain HTTP resumes reliably. HF_ENDPOINT mirrors are honoured.
  retry 3 env HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" "$hf" download "$repo" --local-dir "$dir" >/dev/null \
    || die "download failed: $repo (set HF_ENDPOINT for a mirror, or pass a local path)"
  local f
  for f in $shift_files; do
    [[ -s $dir/$f ]] || die "$repo: $f missing after download"
  done
  if find "$dir" -name '*.incomplete' | grep -q .; then die "$repo: incomplete files left in $dir"; fi
  touch "$dir/.aura-complete"
  ok "$repo ready"
}

ensure_models() {
  step "ASR / TTS weights"
  if [[ -z $ASR_MODEL ]]; then
    ASR_MODEL="$AURA_HOME/models/Qwen3-ASR-1.7B"
    download_model "$ASR_REPO" "$ASR_MODEL" "config.json"
  fi
  if [[ -z $TTS_MODEL ]]; then
    TTS_MODEL="$AURA_HOME/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    download_model "$TTS_REPO" "$TTS_MODEL" "config.json model.safetensors speech_tokenizer/model.safetensors"
  fi
  [[ -f $ASR_MODEL/config.json ]] || die "ASR model not found: $ASR_MODEL"
  [[ -f $TTS_MODEL/config.json ]] || die "TTS model not found: $TTS_MODEL"
  ok "ASR: $ASR_MODEL"
  ok "TTS: $TTS_MODEL"
}

check_aura() {
  [[ -n $AURA_CKPT ]] || die "--aura /path/to/checkpoint is required"
  AURA_CKPT="$(cd "$AURA_CKPT" 2>/dev/null && pwd)" || die "AURA checkpoint not found"
  step "Checking the AURA checkpoint"
  log "$AURA_CKPT"
  local py=python3
  [[ -x $PY ]] && py=$PY
  "$py" "$KIT_DIR/lib/check_ckpt.py" "$AURA_CKPT" || die "this checkpoint cannot be served (see errors above)"
  [[ -n $SERVED_NAME ]] || SERVED_NAME="$(basename "$AURA_CKPT")"
  ok "served model name: $SERVED_NAME"
}

do_install() {
  check_host
  ensure_uv
  ensure_repo
  ensure_cuda_compat
  ensure_venv
  ensure_models
  [[ -n $AURA_CKPT ]] && check_aura
  save_state
}

save_state() {
  save_env "$STATE" AURA_CKPT ASR_MODEL TTS_MODEL SERVED_NAME TTS_SPEAKER GPUS BACKEND_PORT GATEWAY_PORT ENV_FILE COMPAT_LIB LAYOUT GPU_IDS
}

# ---------------------------------------------------------------- start
pick_gpus() {
  local rows idle
  rows=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ')
  if [[ $GPUS == auto ]]; then
    idle=$(awk -F, '$2 < 1024 {print $1}' <<<"$rows" | paste -sd, -)
    local n
    n=$(awk -F, '{print NF}' <<<"${idle:-}")
    [[ -z $idle ]] && n=0
    if [[ $n -ge 2 ]]; then
      GPU_IDS=$(cut -d, -f1,2 <<<"$idle")
    elif [[ $n -eq 1 ]]; then
      GPU_IDS=$idle
    else
      printf '%s\n' "$rows" | awk -F, '{printf "  accelerator %s: %s / %s MiB used\n", $1, $2, $3}' >&2
      die "no idle accelerator; free one or pass --gpus I or --gpus I,J"
    fi
  else
    GPU_IDS=$GPUS
  fi
  local ids
  IFS=, read -r -a ids <<<"$GPU_IDS"
  case ${#ids[@]} in
    1) LAYOUT=1gpu ;;
    2) LAYOUT=2gpu ;;
    *) die "--gpus takes one or two accelerator indices" ;;
  esac
  # Memory each accelerator must have free, as a share of its total (upstream yaml fractions).
  local need_a need_b
  if [[ $LAYOUT == 1gpu ]]; then need_a=0.88; need_b=0; else need_a=0.90; need_b=0.38; fi
  local i id used total share free_need short=0
  for i in "${!ids[@]}"; do
    id=${ids[$i]}
    used=""; total=""
    read -r used total < <(awk -F, -v g="$id" '$1 == g {print $2, $3}' <<<"$rows") || true
    [[ -n $total ]] || die "accelerator $id does not exist (check the accelerator indices)"
    share=$([[ $i -eq 0 ]] && echo "$need_a" || echo "$need_b")
    free_need=$(awk -v t="$total" -v s="$share" 'BEGIN {printf "%d", t * s}')
    if [[ $((total - used)) -lt $free_need ]]; then
      warn "accelerator $id: $((total - used)) MiB free, needs ~$free_need MiB"
      short=1
    fi
    if [[ $LAYOUT == 1gpu && $total -lt 75000 ]]; then
      warn "accelerator $id has $total MiB; the single-accelerator layout is tuned for 80 GB cards"
    fi
  done
  [[ $short -eq 0 || $FORCE -eq 1 ]] || die "not enough free accelerator memory (use other accelerators, or --force)"
  if [[ $LAYOUT == 1gpu ]]; then
    ok "single-accelerator layout on accelerator $GPU_IDS (ASR + AURA + TTS)"
  else
    ok "two-accelerator layout: AURA on accelerator ${ids[0]}, ASR + TTS on accelerator ${ids[1]}"
  fi
}

do_start() {
  load_state
  [[ -x $PY ]] || die "not installed yet: run 'bash aura.sh up --aura ...' first"
  [[ -n $DRIVER_MAJOR ]] || DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [[ -n $COMPAT_LIB && ! -d $COMPAT_LIB ]]; then COMPAT_LIB=""; fi
  [[ -n $COMPAT_LIB || $DRIVER_MAJOR -ge $MIN_DRIVER_FOR_CUDA13 ]] || ensure_cuda_compat
  check_aura
  [[ -f $ASR_MODEL/config.json && -f $TTS_MODEL/config.json ]] || ensure_models
  if pid_alive "$RUN_DIR/backend.pid" || pid_alive "$RUN_DIR/gateway.pid"; then
    die "already running (bash aura.sh status / stop)"
  fi
  local p
  for p in "$BACKEND_PORT" "$GATEWAY_PORT"; do
    port_in_use "$p" && die "port $p is in use (choose another with --port / --backend-port)"
  done
  step "Choosing accelerators"
  pick_gpus
  save_state

  local base yaml="$AURA_HOME/deploy.yaml"
  if [[ $LAYOUT == 1gpu ]]; then
    base="$OMNI_DIR/vllm_omni/deploy/aura_omni_1gpu.yaml"
    "$PY" "$KIT_DIR/lib/make_yaml.py" "$base" "$yaml" --aura "$AURA_CKPT" --asr "$ASR_MODEL" --tts "$TTS_MODEL" --stage1-dev 0 --side-dev 0
  else
    base="$OMNI_DIR/vllm_omni/deploy/aura_omni.yaml"
    "$PY" "$KIT_DIR/lib/make_yaml.py" "$base" "$yaml" --aura "$AURA_CKPT" --asr "$ASR_MODEL" --tts "$TTS_MODEL" --stage1-dev 0 --side-dev 1
  fi
  ok "deploy config: $yaml (from $(basename "$base"))"

  if [[ ${AURA_MEMORY_ENABLE:-0} == 1 ]]; then
    [[ ,$GPU_IDS, != *,${MEMORY_VLM_GPU:-2},* ]] || die "memory accelerator must be separate from demo accelerators"
    step "Starting the memory VLM on accelerator ${MEMORY_VLM_GPU:-2}"
    env "$(compat_env)" bash "$KIT_DIR/memory_vlm.sh" \
      --home "$AURA_HOME" --gpu "${MEMORY_VLM_GPU:-2}" --port "${MEMORY_VLM_PORT:-8003}"
  fi

  step "Starting the backend (4 stages + warmup; first start ~5-8 min, later ~4 min)"
  mkdir -p "$AURA_HOME/user_context"
  run_detached "$RUN_DIR/backend.pid" "$LOG_DIR/backend.log" \
    env "$(compat_env)" PYTHONPATH= CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    AURA_USER_CONTEXT_DIR="$AURA_HOME/user_context" AURA_USER_CONTEXT_MODEL="$SERVED_NAME" \
    VLLM_AURA_TTS_TOKENIZER="$TTS_MODEL" VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "$VENV/bin/vllm-omni" serve "$AURA_CKPT" --omni --deploy-config "$yaml" \
    --served-model-name "$SERVED_NAME" --trust-remote-code --host 0.0.0.0 --port "$BACKEND_PORT"
  local t=0
  while true; do
    if ! pid_alive "$RUN_DIR/backend.pid"; then
      tail -40 "$LOG_DIR/backend.log" >&2
      die "backend exited during startup (full log: $LOG_DIR/backend.log)"
    fi
    if grep -q "Duplex warmup finished" "$LOG_DIR/backend.log" 2>/dev/null; then break; fi
    if grep -qE "Duplex warmup (failed|timed out)" "$LOG_DIR/backend.log" 2>/dev/null && http_ok "http://127.0.0.1:$BACKEND_PORT/v1/models"; then
      warn "warmup did not finish cleanly; the first turn may be slow"
      break
    fi
    [[ $t -ge $WAIT_S ]] && die "backend not ready after ${WAIT_S}s (log: $LOG_DIR/backend.log)"
    if (( t % 30 == 0 )); then log "loading... ${t}s"; fi
    sleep 5; t=$((t + 5))
  done
  ok "backend ready on :$BACKEND_PORT ($(grep -oE 'Duplex warmup finished in [0-9.]+ s' "$LOG_DIR/backend.log" | tail -1))"

  step "Starting the web page"
  local gw="$OMNI_DIR/examples/online_serving/aura_v2_gateway" envfile=${ENV_FILE:-$AURA_HOME/tool_keys.env}
  [[ -d $gw/gateway ]] || die "this tree has no examples/online_serving/aura_v2_gateway"
  local -a extra=()
  if [[ -f $envfile ]]; then
    while IFS= read -r line; do
      [[ $line =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] && extra+=("$line")
    done <"$envfile"
    ok "tool keys from $envfile (${#extra[@]} variable(s))"
  fi
  run_detached "$RUN_DIR/gateway.pid" "$LOG_DIR/gateway.log" \
    env -C "$gw" PYTHONPATH="$gw:$OMNI_DIR" AURA_DUPLEX_BACKEND="ws://127.0.0.1:$BACKEND_PORT" \
    AURA_DUPLEX_MODEL="$SERVED_NAME" GATEWAY_PORT="$GATEWAY_PORT" AURA_TTS_SPEAKER="$TTS_SPEAKER" "${extra[@]}" \
    "$PY" -m gateway.duplex_app
  for ((t = 0; t < 60; t++)); do
    page_ok /health && break
    pid_alive "$RUN_DIR/gateway.pid" || { tail -20 "$LOG_DIR/gateway.log" >&2; die "web page failed to start"; }
    sleep 1
  done
  page_ok /health || die "web page not answering on :$GATEWAY_PORT"
  ok "web page on :$GATEWAY_PORT ($(gw_scheme))"
}

# ---------------------------------------------------------------- other commands
do_stop() {
  step "Stopping"
  stop_pidfile "$RUN_DIR/gateway.pid" "web page"
  stop_pidfile "$RUN_DIR/backend.pid" "backend"
  stop_pidfile "$RUN_DIR/memory_vlm.pid" "memory VLM"
}

do_status() {
  load_state
  step "Status ($AURA_HOME)"
  local name
  for name in backend gateway; do
    if pid_alive "$RUN_DIR/$name.pid"; then ok "$name running (pid $(cat "$RUN_DIR/$name.pid"))"; else warn "$name not running"; fi
  done
  http_ok "http://127.0.0.1:$BACKEND_PORT/v1/models" && ok "backend :$BACKEND_PORT answers" || warn "backend :$BACKEND_PORT not answering"
  page_ok /health && ok "page :$GATEWAY_PORT answers ($(curl -sk --max-time 3 "$(gw_scheme)://127.0.0.1:$GATEWAY_PORT/models"))" || warn "page :$GATEWAY_PORT not answering"
  log "model: ${AURA_CKPT:-?} as ${SERVED_NAME:-?}; layout ${LAYOUT:-?} on accelerator ${GPU_IDS:-?}"
  [[ -n ${GPU_IDS:-} ]] && nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "$GPU_IDS" | sed 's/^/  accelerator /'
  if [[ ${AURA_MEMORY_ENABLE:-0} == 1 ]]; then
    if pid_alive "$RUN_DIR/memory_vlm.pid"; then
      ok "memory VLM running (pid $(cat "$RUN_DIR/memory_vlm.pid"), accelerator ${MEMORY_VLM_GPU:-2})"
    else
      warn "memory VLM not running"
    fi
    http_ok "http://127.0.0.1:${MEMORY_VLM_PORT:-8003}/v1/models" \
      && ok "memory :${MEMORY_VLM_PORT:-8003} answers; summarization enabled" \
      || warn "memory :${MEMORY_VLM_PORT:-8003} not answering"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "${MEMORY_VLM_GPU:-2}" | sed 's/^/  accelerator /'
  fi
  return 0
}

do_test() {
  load_state
  step "Smoke test (camera frames, two spoken questions, one typed question)"
  local assets_args=()
  if [[ -f $KIT_DIR/assets/q1_16k.wav && -f $KIT_DIR/assets/q2_16k.wav ]]; then
    assets_args=(--assets "$KIT_DIR/assets")
  fi
  "$PY" "$KIT_DIR/lib/smoke_test.py" --url "$(ws_scheme)://127.0.0.1:$GATEWAY_PORT/ws" "${assets_args[@]}"
}

print_access() {
  local host proto
  host=$(hostname -I 2>/dev/null | awk '{print $1}')
  proto=$(gw_scheme)
  cat <<EOF

${C_G}AURA is up.${C_0}
  Page (this machine):  $proto://127.0.0.1:$GATEWAY_PORT/
  Page (LAN):           $proto://${host:-<server-ip>}:$GATEWAY_PORT/
  Browsers only allow camera + microphone on https:// or localhost. From a laptop:
      ssh -N -L $GATEWAY_PORT:127.0.0.1:$GATEWAY_PORT <user>@<this-server>   then open http://localhost:$GATEWAY_PORT/
  or put an HTTPS reverse proxy in front (README.md, "Public HTTPS").
  Logs: $LOG_DIR    Stop: bash $KIT_DIR/aura.sh stop --home $AURA_HOME
EOF
}

case "$CMD" in
  install) load_state; do_install ;;
  up)
    load_state
    do_install
    do_start
    if [[ $RUN_TEST -eq 1 ]]; then do_test || warn "smoke test failed: check $LOG_DIR (the services keep running)"; fi
    print_access
    ;;
  start) do_start; [[ $RUN_TEST -eq 1 ]] && { do_test || warn "smoke test failed"; }; print_access ;;
  stop) do_stop ;;
  restart) do_stop; do_start; print_access ;;
  status) do_status ;;
  test) do_test ;;
  logs)
    logs=("$LOG_DIR/backend.log" "$LOG_DIR/gateway.log")
    [[ ! -f $LOG_DIR/memory_vlm.log ]] || logs+=("$LOG_DIR/memory_vlm.log")
    tail -n 50 -F "${logs[@]}"
    ;;
  *) usage; die "unknown command: $CMD" ;;
esac
