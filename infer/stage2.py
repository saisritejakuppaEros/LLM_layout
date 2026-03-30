#!/usr/bin/env python3
"""
Stage 2: layout-conditioned LoRA inference.

Reads Stage 1 outputs (scene image + layouts.json), builds a deterministic conditioning
canvas (no training augmentations). ``refs_masked`` uses a **black** canvas and pastes only
reference images into layout bboxes (splits duplicate bottle/bucket boxes). Saves
``bbox_layout.png`` for that mode. Loads FLUX.2 + canvas LoRA, denoises with optional
layout-prior timestep split (t/1000 > split → zero cond latents).

All new code lives under infer/; train/src is import-only via sys.path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG (override with CLI or env)
# ---------------------------------------------------------------------------
PRETRAINED_MODEL_NAME_OR_PATH = (
    "/mnt/data0/teja/research_multiref/llm_based_layout/models/models--black-forest-labs--FLUX.2-dev/"
    "snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"
)
# Default checkpoint (override with --lora-path or STAGE2_LORA_PATH)
_LORA_PARENT = Path(__file__).resolve().parent.parent / "train" / "output" / "lora_checkpoints"
LORA_SAFETENSORS_PATH = _LORA_PARENT / "checkpoint-34000" / "lora.safetensors"

UNIFIED_TRAIN_WIDTH = 1280
UNIFIED_TRAIN_HEIGHT = 720
GUIDANCE_SCALE = 1.0
NUM_INFERENCE_STEPS = 28
MAX_SEQUENCE_LENGTH = 256
TEXT_ENCODER_OUT_LAYERS = (10, 20, 30)
LORA_RANKS = [32]
LORA_NETWORK_ALPHAS = [32]
LORA_NUM = 1
# Training validation (_denoise_one in flux2_dataloader_validation.py) uses full canvas cond every step.
# Negative = disabled (same as train). If e.g. 0.8, zeros cond only while t/1000 > 0.8 (real cond for t_norm in (0,0.8]).
LAYOUT_PRIOR_T_SPLIT = -1.0
TORCH_DTYPE = "bf16"
# refs_masked: black canvas + refs only in bboxes + bbox_layout.png; scene/refs unchanged
DEFAULT_CANVAS_MODE = "refs_masked"
# ---------------------------------------------------------------------------

_INFER_ROOT = Path(__file__).resolve().parent
if str(_INFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_INFER_ROOT))


def _setup_train_path() -> Path:
    from utils.paths import TRAIN_PACKAGE_DIR

    td = str(TRAIN_PACKAGE_DIR.resolve())
    if td not in sys.path:
        sys.path.insert(0, td)
    return TRAIN_PACKAGE_DIR


def main() -> None:
    import os

    from PIL import Image

    from utils.layout_generation import DEFAULT_STAGE1_PROMPT
    from utils.paths import (
        DEFAULT_STAGE1_OUTPUT,
        DEFAULT_STAGE2_OUTPUT,
        REF_IMAGES_DIR,
        TRAIN_PACKAGE_DIR,
    )
    from utils.reference_images import default_reference_paths, validate_reference_images
    from utils.stage2_canvas import build_stage2_canvas
    from utils.stage2_lora_inference import append_jsonl, denoise_stage2, load_flux2_stage2_bundle, _resolve_dtype

    _setup_train_path()
    from src.jsonl_datasets import multiple_16

    default_model = os.environ.get("FLUX2_MODEL_DIR", PRETRAINED_MODEL_NAME_OR_PATH)
    default_lora = os.environ.get("STAGE2_LORA_PATH", str(LORA_SAFETENSORS_PATH))

    p = argparse.ArgumentParser(description="Stage 2 — LoRA scene generation from Stage 1 layout + canvas cond")
    p.add_argument("--name", "-n", type=str, default="person_bottle_bucket", help="Stage1/Stage2 run name")
    p.add_argument(
        "--stage1-output-dir",
        type=Path,
        default=DEFAULT_STAGE1_OUTPUT,
        help="Base dir containing stage1/<name>/",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_STAGE2_OUTPUT,
        help="Base output directory for stage2/<name>/",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=default_model,
        help="FLUX.2 local snapshot directory (or FLUX2_MODEL_DIR)",
    )
    p.add_argument(
        "--lora-path",
        type=Path,
        default=Path(default_lora),
        help="lora.safetensors (or STAGE2_LORA_PATH)",
    )
    p.add_argument("--prompt", type=str, default=None, help="Text prompt (default: same default as Stage 1)")
    p.add_argument(
        "--canvas-mode",
        type=str,
        choices=["scene", "refs", "refs_masked"],
        default=DEFAULT_CANVAS_MODE,
        help="scene=crops; refs=rect paste; refs_masked=black + refs in layout bboxes + bbox_layout",
    )
    p.add_argument(
        "--segmentation-masks-dir",
        type=Path,
        default=None,
        help="Optional; recorded in logs only (canvas does not use mask files). Default path under stage1 layouts/.",
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    p.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    p.add_argument("--max-sequence-length", type=int, default=MAX_SEQUENCE_LENGTH)
    p.add_argument(
        "--text-encoder-out-layers",
        type=int,
        nargs="+",
        default=list(TEXT_ENCODER_OUT_LAYERS),
    )
    p.add_argument("--ranks", type=int, nargs="+", default=LORA_RANKS)
    p.add_argument("--network-alphas", type=int, nargs="+", default=LORA_NETWORK_ALPHAS)
    p.add_argument("--lora-num", type=int, default=LORA_NUM)
    p.add_argument("--unified-width", type=int, default=UNIFIED_TRAIN_WIDTH)
    p.add_argument("--unified-height", type=int, default=UNIFIED_TRAIN_HEIGHT)
    p.add_argument(
        "--layout-prior-t-split",
        type=float,
        default=LAYOUT_PRIOR_T_SPLIT,
        help=(
            "Experimental: zero cond while t/1000 > split, else real canvas cond. "
            "Default -1 disables (matches train validation: cond every step). "
            "Use 0.8 to only explore above t_norm=0.8 (stronger ref/layout pull than 0.6)."
        ),
    )
    p.add_argument("--torch-dtype", type=str, default=TORCH_DTYPE, choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true", help="Verbose logs + extra LoRA load key samples")
    p.add_argument(
        "--debug-decode-every",
        type=int,
        default=None,
        help="If set, VAE-decode every k steps into debug/ (slow)",
    )
    p.add_argument(
        "--skip-ref-check",
        action="store_true",
        help="Allow refs canvas mode without infer/images files (will skip missing)",
    )
    args = p.parse_args()

    uw = multiple_16(args.unified_width)
    uh = multiple_16(args.unified_height)

    layout_split: float | None = args.layout_prior_t_split
    if layout_split is not None and layout_split < 0:
        layout_split = None

    prompt = (args.prompt or DEFAULT_STAGE1_PROMPT).strip()
    stage1_root = Path(args.stage1_output_dir).resolve() / args.name
    image_path = stage1_root / "images" / f"{args.name}.png"
    layouts_path = stage1_root / "layouts" / "layouts.json"
    if not image_path.is_file():
        raise FileNotFoundError(f"Stage 1 image not found: {image_path}")
    if not layouts_path.is_file():
        raise FileNotFoundError(f"layouts.json not found: {layouts_path}")

    out_root = Path(args.output_dir).resolve() / args.name
    debug_dir = out_root / "debug"
    out_root.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    refs = default_reference_paths()
    if args.canvas_mode in ("refs", "refs_masked") and not args.skip_ref_check:
        miss = validate_reference_images(refs)
        if miss:
            raise FileNotFoundError(f"Missing reference images for keys {miss} under {REF_IMAGES_DIR}")

    scene_pil = Image.open(image_path).convert("RGB")

    # S0
    s0 = {
        "tag": "S0_config",
        "stage1_root": str(stage1_root),
        "out_root": str(out_root),
        "model_dir": str(Path(args.model_dir).resolve()),
        "lora_path": str(Path(args.lora_path).resolve()),
        "canvas_mode": args.canvas_mode,
        "segmentation_masks_dir": str(
            Path(args.segmentation_masks_dir).resolve()
            if args.segmentation_masks_dir is not None
            else (stage1_root / "layouts" / "segmentation_masks")
        ),
        "segmentation_masks_used_for_canvas": False,
        "unified_wh": [uw, uh],
        "layout_prior_t_split": layout_split,
        "cond_matches_train_validation": layout_split is None,
        "steps": args.steps,
        "seed": args.seed,
        "prompt_excerpt": prompt[:200],
    }
    print(json.dumps(s0, indent=2))
    append_jsonl(debug_dir / "run_log.jsonl", s0)

    masks_log = Path(args.segmentation_masks_dir).resolve() if args.segmentation_masks_dir is not None else (stage1_root / "layouts" / "segmentation_masks").resolve()

    canvas_pil, bbox_layout_pil, layout_meta, filter_debug = build_stage2_canvas(
        mode=args.canvas_mode,
        scene_pil=scene_pil,
        layouts_path=layouts_path,
        refs=refs,
        canvas_w=uw,
        canvas_h=uh,
        run_name=args.name,
        seed=args.seed,
    )

    s1 = {
        "tag": "S1_layout",
        "rows": layout_meta,
        "filter_debug": filter_debug,
        "canvas_saved": str(out_root / "canvas.png"),
        "bbox_layout_saved": str(out_root / "bbox_layout.png") if bbox_layout_pil is not None else None,
        "segmentation_masks_dir": str(masks_log) if args.canvas_mode == "refs_masked" else None,
    }
    print(json.dumps(s1, indent=2))
    append_jsonl(debug_dir / "run_log.jsonl", s1)

    canvas_pil.save(out_root / "canvas.png")
    canvas_pil.save(debug_dir / "canvas.png")
    if bbox_layout_pil is not None:
        bbox_layout_pil.save(out_root / "bbox_layout.png")
        bbox_layout_pil.save(debug_dir / "bbox_layout.png")

    from src.canvas_dataset import pil_to_model_tensor, resize_unified_pil

    target_pil = resize_unified_pil(scene_pil, uw, uh, "contain")
    canvas_resized = resize_unified_pil(canvas_pil, uw, uh, "contain")
    pixel_values = pil_to_model_tensor(target_pil).unsqueeze(0)
    subject_pixel_values = pil_to_model_tensor(canvas_resized).unsqueeze(0)

    if not Path(args.lora_path).is_file():
        raise FileNotFoundError(f"LoRA not found: {args.lora_path}")

    weight_dtype = _resolve_dtype(args.torch_dtype)
    bundle, lora_stats = load_flux2_stage2_bundle(
        pretrained_model_name_or_path=args.model_dir,
        lora_safetensors_path=args.lora_path,
        train_dir=TRAIN_PACKAGE_DIR,
        device=args.device,
        weight_dtype=weight_dtype,
        cond_width=uw,
        cond_height=uh,
        ranks=list(args.ranks),
        network_alphas=list(args.network_alphas),
        lora_num=args.lora_num,
        debug=args.debug,
    )

    s2 = {"tag": "S2_weights", **lora_stats}
    print(json.dumps(s2, indent=2))
    append_jsonl(debug_dir / "run_log.jsonl", s2)

    s3 = {
        "tag": "S3_encode",
        "pixel_values_shape": list(pixel_values.shape),
        "subject_pixel_values_shape": list(subject_pixel_values.shape),
    }
    print(json.dumps(s3, indent=2))
    append_jsonl(debug_dir / "run_log.jsonl", s3)

    import torch

    gen = None
    if args.seed is not None:
        gen = torch.Generator(device=bundle.device)
        gen.manual_seed(int(args.seed))

    gen_pil, step_logs = denoise_stage2(
        bundle,
        pixel_values=pixel_values.to(device=bundle.device, dtype=torch.float32),
        subject_pixel_values=subject_pixel_values.to(device=bundle.device, dtype=torch.float32),
        prompt=prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        text_encoder_out_layers=tuple(args.text_encoder_out_layers),
        generator=gen,
        layout_prior_t_split=layout_split,
        debug_decode_every=args.debug_decode_every,
        debug_dir=debug_dir if args.debug_decode_every else None,
    )

    for row in step_logs:
        row["tag"] = "S4_denoise"
        append_jsonl(debug_dir / "run_log.jsonl", row)
        if args.debug:
            print(json.dumps(row))

    gen_pil.save(out_root / "gen.png")

    meta = {
        "prompt": prompt,
        "stage1_image": str(image_path),
        "layouts_json": str(layouts_path),
        "model_dir": str(Path(args.model_dir).resolve()),
        "lora_path": str(Path(args.lora_path).resolve()),
        "canvas_mode": args.canvas_mode,
        "segmentation_masks_dir": str(masks_log) if args.canvas_mode == "refs_masked" else None,
        "bbox_layout": str(out_root / "bbox_layout.png") if bbox_layout_pil is not None else None,
        "unified_wh": [uw, uh],
        "layout_prior_t_split": layout_split,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "layout_rows": layout_meta,
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Done.")
    print(f"  gen.png: {out_root / 'gen.png'}")
    print(f"  canvas.png: {out_root / 'canvas.png'}")
    if bbox_layout_pil is not None:
        print(f"  bbox_layout.png: {out_root / 'bbox_layout.png'}")
    print(f"  meta.json: {out_root / 'meta.json'}")
    print(f"  debug: {debug_dir}")


if __name__ == "__main__":
    main()
