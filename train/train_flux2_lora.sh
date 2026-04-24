#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL_DIR="/mnt/data0/teja/research_multiref/llm_based_layout/models/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"
CSV_PATH="/mnt/data0/teja/research_multiref/dataset_preparation/output/bbox_results/yolo26_detections.csv"
CANVAS_IMAGE_ROOT="/mnt/data0/teja/research_multiref/dataset_preparation/output/images"
DEPTH_IMAGE_ROOT="/mnt/data0/teja/research_multiref/dataset_preparation/output/depth"
# Per-sample prob to keep real depth / canvas (else black cond). Override: DEPTH_KEEP_PROB=1 CANVAS_KEEP_PROB=1 ./train_flux2_lora.sh
DEPTH_KEEP_PROB="${DEPTH_KEEP_PROB:-0.5}"
CANVAS_KEEP_PROB="${CANVAS_KEEP_PROB:-0.5}"
CAPTION_DIR="/mnt/data0/teja/research_multiref/dataset_preparation/output/image_captions"
MULTIVIEW_DIR="/mnt/data0/teja/research_multiref/dataset_preparation/output/multiview_out"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output/lora_checkpoints_v2"

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"

COMMON_ARGS=(
  --pretrained_model_name_or_path "$MODEL_DIR"
  --dataset_type canvas
  --csv_path "$CSV_PATH"
  --canvas_image_root "$CANVAS_IMAGE_ROOT"
  --depth_image_root "$DEPTH_IMAGE_ROOT"
  --depth_keep_prob "$DEPTH_KEEP_PROB"
  --canvas_keep_prob "$CANVAS_KEEP_PROB"
  --canvas_conditioning bbox_multiview
  --canvas_bbox_min_side 0
  --canvas_multiview_match_min_side 200
  --canvas_multiview_dir "$MULTIVIEW_DIR"
  --canvas_multiview_prob 0.5
  --canvas_background black
  --canvas_column canvas_path
  --canvas_target_column image_path
  --canvas_prompt_column prompt
  --caption_dir "$CAPTION_DIR"
  --prompt_warmup_steps 500
  --spatial_column None
  --cond_size 512
  --noise_size 1024
  --unified_train_width 1280
  --unified_train_height 720
  --canvas_unified_resize cover
  --ranks 32
  --network_alphas 32
  --lora_num 2
  --output_dir "$OUTPUT_DIR"
  --logging_dir "$OUTPUT_DIR/logs"
  --mixed_precision bf16
  --gradient_checkpointing
  --learning_rate 1e-4
  --train_batch_size 1
  --gradient_accumulation_steps 1
  --num_train_epochs 1000
  --checkpointing_steps 1000
  --validation_steps 100
  --validation_num_samples 4
  --validation_loss_micro_batch_size 1
  --validation_inference_steps 28
  --validation_samples_subdir validation_samples
  --guidance_scale 1.0
  --max_sequence_length 256
  --text_encoder_out_layers 10 20 30
)

export CUDA_VISIBLE_DEVICES=1
python -u train.py "${COMMON_ARGS[@]}"
