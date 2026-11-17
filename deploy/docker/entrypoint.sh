#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$KIT_DIR/lib/common.sh"
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  exec bash "$KIT_DIR/aura.sh" --help
fi
AURA_HOME="${AURA_HOME:-/var/lib/aura}"
mkdir -p "$AURA_HOME/run"

# A runtime directory belongs to one container. PIDs from a stopped container
# are invalid in a new PID namespace, so remove them after acquiring the lock.
exec 9>"$AURA_HOME/.container.lock"
flock -n 9 || die "another container is using $AURA_HOME"
rm -f "$AURA_HOME/run/backend.pid" "$AURA_HOME/run/gateway.pid" "$AURA_HOME/run/memory_vlm.pid"
tail_pid=""
cleanup() {
  trap - EXIT
  [[ -z $tail_pid ]] || kill "$tail_pid" 2>/dev/null || true
  bash "$KIT_DIR/aura.sh" stop --home "$AURA_HOME" || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

bash "$KIT_DIR/aura.sh" up --speaker Serena "$@" --home "$AURA_HOME"
tail -n 30 -F "$AURA_HOME/logs/backend.log" "$AURA_HOME/logs/gateway.log" &
tail_pid=$!
while sleep 5; do
  for service in backend gateway; do
    pid_alive "$AURA_HOME/run/$service.pid" || die "$service exited; inspect $AURA_HOME/logs/$service.log"
  done
done
