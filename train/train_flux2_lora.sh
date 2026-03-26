#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL_DIR="/mnt/data0/teja/research_multiref/llm_based_layout/models/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"
CSV_PATH="/mnt/data0/teja/research_multiref/dataset_preparation/output/bbox_results/yolo26_detections.csv"
CANVAS_IMAGE_ROOT="/mnt/data0/teja/research_multiref/dataset_preparation/output/images"
# Stage3 multiview root (optional); leave empty to always use bbox crops from the original image.
MULTIVIEW_DIR="/mnt/data0/teja/research_multiref/dataset_preparation/output/multiview_out"
OUTPUT_DIR="./output/lora_checkpoints"

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"

# VRAM: FLUX.2 + canvas (main+cond latents) + long text fills ~140GB easily without checkpointing.
# Optional if you hit fragmentation: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
  --pretrained_model_name_or_path "$MODEL_DIR"
  --dataset_type canvas
  --csv_path "$CSV_PATH"
  --canvas_image_root "$CANVAS_IMAGE_ROOT"
  --canvas_conditioning bbox_multiview
  --canvas_bbox_min_side 0
  --canvas_multiview_match_min_side 200
  --canvas_multiview_dir "$MULTIVIEW_DIR"
  --canvas_multiview_prob 0.5
  --canvas_background black
  --canvas_column canvas_path
  --canvas_target_column image_path
  --canvas_prompt_column prompt
  --spatial_column None
  --cond_size 512
  --noise_size 1024
  # Unified 720p (1280×720, both /16) — lower VRAM than 1080p; compose is still 1920×1080 then scaled here (contain keeps pairing).
  --unified_train_width 1280
  --unified_train_height 720
  # Lower rank = fewer LoRA params / less optimizer state (raise e.g. 64 → 128 when you have a free GPU).
  --ranks 32
  --network_alphas 32
  --lora_num 1
  --output_dir "$OUTPUT_DIR"
  --logging_dir "$OUTPUT_DIR/logs"
  --mixed_precision bf16
  # Recompute activations during backward — large VRAM savings on long sequences (slower step).
  --gradient_checkpointing
  --learning_rate 1e-4
  --train_batch_size 1
  --gradient_accumulation_steps 1
  --num_train_epochs 1000
  --checkpointing_steps 1000
  --validation_steps 500
  --validation_num_samples 4
  --validation_inference_steps 28
  --validation_samples_subdir validation_samples
  --guidance_scale 1.0
  # Shorter text sequence = fewer joint-attention tokens (raise if prompts are truncated).
  --max_sequence_length 256
  --text_encoder_out_layers 10 20 30
)

# Single GPU, plain Python — one process, no NCCL/DDP barriers (avoids long stalls
# while ranks diverge during heavy CPU work, e.g. FluxTransformer2DModel.from_config).
export CUDA_VISIBLE_DEVICES=5
python train.py "${COMMON_ARGS[@]}"

# Multi-GPU (uncomment): set CUDA_VISIBLE_DEVICES=6,7 and NCCL env before launch.
# export CUDA_VISIBLE_DEVICES=6,7
# export NCCL_NVLS_ENABLE=0
# export NCCL_TREE_THRESHOLD=0
# export NCCL_NET_GDR_LEVEL=0
# export NCCL_P2P_LEVEL=SYS
# export NCCL_SHM_DISABLE=0
# export NCCL_ALGO=Ring
# export NCCL_TIMEOUT=1800
# export NCCL_DEBUG=WARN
# accelerate launch --config_file ./default_config.yaml train.py "${COMMON_ARGS[@]}"