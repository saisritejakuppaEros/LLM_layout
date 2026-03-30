#!/usr/bin/env python3
"""Stage 1: generate scene layout (image + bboxes) via FLUX.2 + attention extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_INFER_ROOT = Path(__file__).resolve().parent
if str(_INFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_INFER_ROOT))

from utils.layout_generation import DEFAULT_STAGE1_PROMPT, run_stage1_layout
from utils.paths import DEFAULT_STAGE1_OUTPUT


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 1 — diffusion layout for person / bottle / bucket")
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=f"Override default prompt (comma-separated clauses work best). Default uses three object clauses.",
    )
    p.add_argument("--name", "-n", type=str, default="person_bottle_bucket", help="Run name under output dir")
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_STAGE1_OUTPUT,
        help=f"Base output directory (default: {DEFAULT_STAGE1_OUTPUT})",
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--no-extraction", action="store_true", help="Only image + attention maps")
    p.add_argument("--attn-min-t", type=float, default=0.6)
    p.add_argument("--attn-max-t", type=float, default=0.8)
    p.add_argument("--torch-dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument(
        "--native-size",
        action="store_true",
        help="Do not pass width/height to the pipeline (model default resolution)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1280,
        help="FLUX.2 generation width (default 1280; paired with --height)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=720,
        help="FLUX.2 generation height (default 720)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed (cpu generator, matches typical Flux examples)",
    )
    p.add_argument(
        "--layout-preview-w",
        type=int,
        default=None,
        help="Rare: letterbox layout to this width (default: no resize — same as generated image)",
    )
    p.add_argument(
        "--layout-preview-h",
        type=int,
        default=None,
        help="Rare: letterbox layout to this height",
    )
    p.add_argument(
        "--models-cache",
        type=Path,
        default=None,
        help="HF/transformers cache for FLUX.2 (default: llm_based_layout/models)",
    )
    p.add_argument(
        "--skip-ref-check",
        action="store_true",
        help="Do not require infer/images person.jpg, bottle.jpg, bucket.jpg",
    )
    args = p.parse_args()

    gen_w = None if args.native_size else args.width
    gen_h = None if args.native_size else args.height

    result = run_stage1_layout(
        prompt=args.prompt,
        output_name=args.name,
        output_dir=args.output_dir,
        device=args.device,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        run_extraction=not args.no_extraction,
        attn_min_t=args.attn_min_t,
        attn_max_t=args.attn_max_t,
        torch_dtype=args.torch_dtype,
        width=gen_w,
        height=gen_h,
        seed=args.seed,
        layout_preview_w=args.layout_preview_w,
        layout_preview_h=args.layout_preview_h,
        models_cache=args.models_cache,
        require_reference_images=not args.skip_ref_check,
    )

    print("Done.")
    print(
        f"  Image: {result['image_path']} ({result.get('image_width', '?')}×{result.get('image_height', '?')})"
    )
    print(f"  Attention: {result['attn_dir']}")
    print(f"  Layouts: {result['layout_dir']}")
    if "num_objects" in result:
        print(f"  Objects: {result['num_objects']}")
    print(f"  Reference images: {result.get('reference_images', {})}")
    if args.prompt is None:
        print(f"  (default prompt — override with --prompt; current default excerpt: {DEFAULT_STAGE1_PROMPT[:80]}...)")


if __name__ == "__main__":
    main()
