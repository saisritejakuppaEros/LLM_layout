"""
Stage-1 layout generation: FLUX.2 image + attention maps + clause-based bboxes.

Delegates to ``diffusion_layout_generation`` (same repo). Reference JPEGs in
``infer/images`` are for later stages; stage 1 only needs a prompt whose
comma-separated clauses align with those objects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .paths import MODELS_CACHE_DIR, REPO_ROOT, DEFAULT_STAGE1_OUTPUT
from .reference_images import ReferenceImageSet, default_reference_paths, validate_reference_images

DEFAULT_STAGE1_PROMPT = (

"A cinematic outdoor daytime shot featuring at the center, a person, a bottle, a bucket"
)


def _ensure_repo_on_path() -> None:
    r = str(REPO_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def build_stage1_prompt(custom: str | None = None) -> str:
    return (custom or DEFAULT_STAGE1_PROMPT).strip()


def run_stage1_layout(
    prompt: str | None = None,
    output_name: str = "person_bottle_bucket",
    output_dir: str | Path | None = None,
    device: str = "cuda",
    num_inference_steps: int = 15,
    guidance_scale: float = 4.0,
    run_extraction: bool = True,
    attn_min_t: float = 0.6,
    attn_max_t: float = 0.8,
    torch_dtype: str = "fp16",
    width: int | None = 1280,
    height: int | None = 720,
    seed: int | None = None,
    layout_preview_w: int | None = None,
    layout_preview_h: int | None = None,
    models_cache: str | Path | None = None,
    require_reference_images: bool = True,
    reference_paths: ReferenceImageSet | None = None,
) -> dict[str, Any]:
    """
    Run diffusion layout generation and optionally assert reference assets exist.

    Returns the dict from ``generate_layout_from_prompt`` plus
    ``reference_images`` (resolved paths) when validation passes.
    """
    if require_reference_images:
        refs = reference_paths or default_reference_paths()
        miss = validate_reference_images(refs)
        if miss:
            raise FileNotFoundError(
                f"Missing reference images for: {miss} under {refs.person.parent}"
            )
    else:
        refs = reference_paths or default_reference_paths()

    _ensure_repo_on_path()
    from diffusion_layout_generation.generate_layout import generate_layout_from_prompt

    out = Path(output_dir) if output_dir is not None else DEFAULT_STAGE1_OUTPUT
    cache = Path(models_cache) if models_cache is not None else MODELS_CACHE_DIR

    result = generate_layout_from_prompt(
        prompt=build_stage1_prompt(prompt),
        output_name=output_name,
        output_dir=out,
        models_cache=str(cache),
        device=device,
        run_extraction=run_extraction,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        attn_min_t=attn_min_t,
        attn_max_t=attn_max_t,
        torch_dtype=torch_dtype,
        width=width,
        height=height,
        seed=seed,
        layout_preview_w=layout_preview_w,
        layout_preview_h=layout_preview_h,
    )
    result["reference_images"] = {k: str(p) for k, p in refs.as_dict().items()}
    run_root = Path(result["image_path"]).resolve().parent.parent
    manifest = run_root / "reference_manifest.json"
    manifest.write_text(json.dumps(result["reference_images"], indent=2), encoding="utf-8")
    result["reference_manifest_path"] = str(manifest)
    return result
