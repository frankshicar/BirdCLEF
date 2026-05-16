#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG_PATH:-birdclef2026/config/local.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RUN_NAME="${RUN_NAME:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/train}"
PYTHON_BIN="${PYTHON_BIN:-}"
FORCE_CPU="${FORCE_CPU:-1}"

mkdir -p "$LOG_DIR" checkpoints

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in "/home/sbplab/anaconda3/bin/python" "python" "python3"; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" - <<'PY' >/dev/null 2>&1
import torch
PY
    then
      if [[ "$candidate" = /* ]]; then
        PYTHON_BIN="$candidate"
      else
        PYTHON_BIN="$(command -v "$candidate")"
      fi
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Could not find a Python interpreter with torch installed." >&2
  exit 1
fi

LOG_PATH="$LOG_DIR/${RUN_NAME}.log"
PID_PATH="$LOG_DIR/${RUN_NAME}.pid"

CMD=("$PYTHON_BIN" scripts/train.py --config "$CONFIG_PATH")
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  CMD+=(--resume "$RESUME_CHECKPOINT")
fi

echo "Root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Config: $CONFIG_PATH"
echo "Log: $LOG_PATH"
echo "FORCE_CPU: $FORCE_CPU"
echo "Command: FORCE_CPU=$FORCE_CPU ${CMD[*]}"

setsid bash -c '
  set +e
  echo "[$(date -u --iso-8601=seconds)] Starting training"
  echo "Command: FORCE_CPU=$0 ${@:2}"
  FORCE_CPU="$0" "${@:2}"
  status="$?"
  echo "[$(date -u --iso-8601=seconds)] Training exited with status $status"
  exit "$status"
' "$FORCE_CPU" "${CMD[@]}" > "$LOG_PATH" 2>&1 < /dev/null &
PID="$!"
echo "$PID" > "$PID_PATH"

echo "Started training in background."
echo "PID: $PID"
echo "PID file: $PID_PATH"
echo "Tail logs with: tail -f $LOG_PATH"
