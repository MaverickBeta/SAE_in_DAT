#!/usr/bin/env bash
set -euo pipefail

# Run FAB untargeted attack in 4-way sharded mode on 4 GPUs.
# This launcher itself stays in foreground briefly, but each shard runs in background via nohup.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/ad_generation_untar.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: Cannot find $PY_SCRIPT"
  exit 1
fi

# Tunables (override by exporting before running this script)
NUM_SHARDS="${NUM_SHARDS:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
ATTACK_FILTER="${ATTACK_FILTER:-fab_untar_sl2py}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
FAB_STEPS="${FAB_STEPS:-60}"
FAB_BATCH_SIZE="${FAB_BATCH_SIZE:-2}"
FAB_EPS="${FAB_EPS:-0.06274509803921569}"  # 16/255
ATTACK_MAX_CHUNK="${ATTACK_MAX_CHUNK:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Optional paths. Leave empty to use Python defaults.
SOURCE_DIR="${SOURCE_DIR:-}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/Data_share/hongyi/DAT/data/ImageNet/val_attacks_sl2py}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$SCRIPT_DIR/runs/fab_${TS}"
LOG_DIR="$RUN_DIR/logs"
PID_DIR="$RUN_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt "$NUM_SHARDS" ]]; then
  echo "ERROR: GPU_LIST has ${#GPUS[@]} GPUs but NUM_SHARDS=$NUM_SHARDS"
  exit 1
fi

echo "Run dir: $RUN_DIR"
echo "Launching $NUM_SHARDS shards with ATTACK_FILTER=$ATTACK_FILTER"

for ((i=0; i<NUM_SHARDS; i++)); do
  gpu="${GPUS[$i]}"
  out_root="$OUTPUT_ROOT_BASE/shard_$(printf "%02d" "$i")"
  log="$LOG_DIR/shard_${i}.log"
  pidfile="$PID_DIR/shard_${i}.pid"

  cmd=(
    env
    CUDA_VISIBLE_DEVICES="$gpu"
    ATTACK_FILTER="$ATTACK_FILTER"
    NUM_SHARDS="$NUM_SHARDS"
    SHARD_INDEX="$i"
    FORCE_REGENERATE="$FORCE_REGENERATE"
    FAB_STEPS="$FAB_STEPS"
    FAB_BATCH_SIZE="$FAB_BATCH_SIZE"
    FAB_EPS="$FAB_EPS"
    ATTACK_MAX_CHUNK="$ATTACK_MAX_CHUNK"
    NUM_WORKERS="$NUM_WORKERS"
    OUTPUT_ROOT="$out_root"
  )

  if [[ -n "$SOURCE_DIR" ]]; then
    cmd+=(SOURCE_DIR="$SOURCE_DIR")
  fi

  cmd+=(python "$PY_SCRIPT")

  echo "[shard $i] gpu=$gpu log=$log out=$out_root"
  nohup "${cmd[@]}" > "$log" 2>&1 &
  echo $! > "$pidfile"
done

cat > "$RUN_DIR/README.txt" <<EOF
FAB background run launched at: $TS
Launcher: $0
Python script: $PY_SCRIPT
NUM_SHARDS: $NUM_SHARDS
GPU_LIST: $GPU_LIST
ATTACK_FILTER: $ATTACK_FILTER
FORCE_REGENERATE: $FORCE_REGENERATE
FAB_STEPS: $FAB_STEPS
FAB_BATCH_SIZE: $FAB_BATCH_SIZE
FAB_EPS: $FAB_EPS
ATTACK_MAX_CHUNK: $ATTACK_MAX_CHUNK
NUM_WORKERS: $NUM_WORKERS
SOURCE_DIR: ${SOURCE_DIR:-<python-default>}
OUTPUT_ROOT_BASE: $OUTPUT_ROOT_BASE

Logs: $LOG_DIR
PIDs: $PID_DIR

Quick checks:
  tail -f $LOG_DIR/shard_0.log
  watch -n 1 nvidia-smi

Stop all shards for this run:
  for f in $PID_DIR/*.pid; do kill "\$(cat "\$f")" 2>/dev/null || true; done
EOF

echo
echo "Started."
echo "Logs: $LOG_DIR"
echo "PIDs: $PID_DIR"
echo "Readme: $RUN_DIR/README.txt"
