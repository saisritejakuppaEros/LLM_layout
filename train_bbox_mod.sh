#!/usr/bin/env bash
# Multi-GPU: uses physical GPUs listed in CUDA_VISIBLE_DEVICES (default 4 and 7).
# They appear as cuda:0 and cuda:1 inside each process; num_processes must match the count.
set -euo pipefail

cd "$(dirname "$0")"

# Physical GPU ids (comma-separated). Override: CUDA_VISIBLE_DEVICES=0,1 ./train_bbox_mod.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,7}"

# Number of processes = number of visible GPUs (default 2 for GPUs 4+7).
NUM_GPUS="${NUM_GPUS:-2}"

# Avoid port clashes if another training job uses the default port.
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29547}"

# Optional: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate not found; install with: pip install accelerate" >&2
  exit 1
fi

LAUNCH=(accelerate launch --num_machines 1 --num_processes "${NUM_GPUS}" --main_process_port "${MAIN_PROCESS_PORT}")
if [ "${NUM_GPUS}" -gt 1 ]; then
  LAUNCH+=(--multi_gpu)
fi

exec "${LAUNCH[@]}" train.py --config config.yaml
