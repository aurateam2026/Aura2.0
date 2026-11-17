# shellcheck shell=bash
# Shared helpers for aura.sh. Sourced, not executed.

if [[ -t 1 ]]; then
  C_B=$'\033[1m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_R=$'\033[31m'; C_0=$'\033[0m'
else
  C_B=""; C_G=""; C_Y=""; C_R=""; C_0=""
fi

log()  { printf '%s[aura]%s %s\n' "$C_B" "$C_0" "$*"; }
ok()   { printf '%s[ ok ]%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '%s[warn]%s %s\n' "$C_Y" "$C_0" "$*" >&2; }
die()  { printf '%s[fail]%s %s\n' "$C_R" "$C_0" "$*" >&2; exit 1; }
step() { printf '\n%s==> %s%s\n' "$C_B" "$*" "$C_0"; }
have() { command -v "$1" >/dev/null 2>&1; }

# Run a command up to N times (network steps flake on some clusters).
retry() {
  local n=$1; shift
  local i
  for ((i = 1; i <= n; i++)); do
    "$@" && return 0
    [[ $i -lt $n ]] && { warn "attempt $i/$n failed: $*; retrying in 5 s"; sleep 5; }
  done
  return 1
}

# Start a long-running process detached from this shell; write its pid.
# usage: run_detached <pidfile> <logfile> <cmd...>
run_detached() {
  local pidfile=$1 logfile=$2; shift 2
  mkdir -p "$(dirname "$pidfile")" "$(dirname "$logfile")"
  setsid nohup "$@" >"$logfile" 2>&1 </dev/null &
  echo $! >"$pidfile"
}

pid_alive() { [[ -f $1 ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

# Stop the process group started by run_detached.
stop_pidfile() {
  local pidfile=$1 name=$2 pid i
  if ! pid_alive "$pidfile"; then
    rm -f "$pidfile"
    log "$name: not running"
    return 0
  fi
  pid=$(cat "$pidfile")
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for ((i = 0; i < 60; i++)); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    warn "$name did not exit after 60 s; sending SIGKILL"
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$pidfile"
  ok "$name stopped"
}

port_in_use() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&- 3<&-; return 0; }
  return 1
}

http_ok() { curl -sf --max-time 3 "$1" >/dev/null 2>&1; }

# Persist resolved settings so stop/status/test reuse them.
save_env() {
  local file=$1; shift
  : >"$file"
  local name
  for name in "$@"; do
    printf '%s=%q\n' "$name" "${!name-}" >>"$file"
  done
}
