"""
Generate layout from a text prompt using FLUX.2-dev.

Pipeline:
  1. Load FLUX.2-dev with attention-map-diffusers hooks
  2. Generate image + capture attention maps
  3. Extract per-object layouts from attention (clause-based segmentation)

Usage:
    from diffusion_layout_generation import generate_layout_from_prompt
    result = generate_layout_from_prompt("A wide shot of a room with a lamp, a chair, and a table")

    # Or from CLI:
    python -m diffusion_layout_generation.generate_layout --prompt "Your prompt here"
"""

import argparse
import re
import sys
from pathlib import Path

# Add llm_based_layout and attention-map-diffusers to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LLM_LAYOUT = _REPO_ROOT / "llm_based_layout"
_ATTN_MAP_PKG = _LLM_LAYOUT / "attention-map-diffusers"
MODELS_CACHE = _LLM_LAYOUT / "models"

sys.path.insert(0, str(_LLM_LAYOUT))
sys.path.insert(0, str(_ATTN_MAP_PKG))

import numpy as np
import torch
from PIL import Image

# These imports require the paths above
from diffusers import Flux2Pipeline
from diffusers.pipelines.flux2.pipeline_flux2 import format_input
from attention_map_diffusers import attn_maps, init_pipeline, save_attention_maps


def _slugify(name: str) -> str:
    """Create filesystem-safe name from prompt or custom name."""
    s = re.sub(r"[^\w\s-]", "", name.lower())
    s = re.sub(r"[-\s]+", "_", s).strip("_")
    return s[:64] if s else "layout"


class Flux2TokenizerWrapper:
    """Wrapper so save_attention_maps gets token_ids from FLUX2 chat format."""

    def __init__(self, processor, system_message):
        self.processor = processor
        self.system_message = system_message
        self._tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

    def __call__(self, prompts):
        prompts = [prompts] if isinstance(prompts, str) else prompts
        messages = format_input(prompts, system_message=self.system_message)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        return {"input_ids": inputs["input_ids"]}

    def convert_ids_to_tokens(self, ids):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self._tokenizer.convert_ids_to_tokens(ids)


def _get_flux2_local_path() -> Path | None:
    """Return local FLUX.2-dev path if fully downloaded, else None."""
    snapshots = MODELS_CACHE / "models--black-forest-labs--FLUX.2-dev" / "snapshots"
    if not snapshots.exists():
        return None
    for snapshot in snapshots.iterdir():
        if snapshot.is_dir():
            return snapshot
    return None


def _load_pipeline(device: str = "cuda", cache_dir: str | Path | None = None):
    """Load FLUX.2-dev pipeline with attention capture hooks."""
    if cache_dir is None:
        cache_dir = str(MODELS_CACHE)
    # Prefer local path to avoid re-downloading
    local_path = _get_flux2_local_path()
    model_id = str(local_path) if local_path else "black-forest-labs/FLUX.2-dev"
    kwargs = {"torch_dtype": torch.bfloat16}
    if not local_path:
        kwargs["cache_dir"] = cache_dir
    pipe = Flux2Pipeline.from_pretrained(model_id, **kwargs)
    pipe.to(device)
    pipe = init_pipeline(pipe)
    return pipe


def generate_image_and_attention(
    prompt: str,
    pipe=None,
    num_inference_steps: int = 15,
    guidance_scale: float = 4.0,
    device: str = "cuda",
    models_cache: str | Path | None = None,
):
    """
    Generate image and capture attention maps for a prompt.

    Returns:
        (PIL.Image, attn_maps dict) - generated image and attention maps
    """
    if pipe is None:
        pipe = _load_pipeline(device=device, cache_dir=models_cache)

    attn_maps.clear()
    images = pipe(
        prompt=[prompt],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images

    return images[0], pipe


def save_attention_maps_for_prompt(
    attn_maps_dict,
    pipe,
    prompt: str,
    base_dir: str,
):
    """Save attention maps to disk using the pipeline's tokenizer."""
    tokenizer_for_save = Flux2TokenizerWrapper(
        pipe.tokenizer, pipe.system_message
    )
    save_attention_maps(
        attn_maps_dict,
        tokenizer_for_save,
        [prompt],
        base_dir=base_dir,
        unconditional=False,
    )


def generate_layout_from_prompt(
    prompt: str,
    output_name: str | None = None,
    output_dir: str | Path = "outputs",
    models_cache: str = "./models",
    device: str = "cuda",
    run_extraction: bool = True,
    num_inference_steps: int = 15,
    guidance_scale: float = 4.0,
):
    """
    Full pipeline: generate image + attention maps + extract layouts.

    Args:
        prompt: Text prompt for image generation
        output_name: Optional name for outputs (default: slug from prompt)
        output_dir: Base output directory (default: outputs)
        models_cache: Cache dir for FLUX.2 model
        device: Device for inference (cuda/cpu)
        run_extraction: Whether to run layout extraction from attention maps
        num_inference_steps: Diffusion steps
        guidance_scale: CFG scale

    Returns:
        dict with keys: image_path, attn_dir, layout_dir, layouts (if extraction run)
    """
    output_dir = Path(output_dir)
    name = output_name or _slugify(prompt[:80])
    attn_dir = output_dir / name / "attention_maps"
    layout_dir = output_dir / name / "layouts"
    images_dir = output_dir / name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pipeline and generate
    cache_dir = str(MODELS_CACHE) if models_cache is None else models_cache
    pipe = _load_pipeline(device=device, cache_dir=cache_dir)
    image, pipe = generate_image_and_attention(
        prompt,
        pipe=pipe,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        device=device,
    )

    img_path = images_dir / f"{name}.png"
    image.save(img_path)

    # 2. Save attention maps
    save_attention_maps_for_prompt(
        attn_maps, pipe, prompt, str(attn_dir)
    )

    result = {
        "image_path": str(img_path),
        "attn_dir": str(attn_dir),
        "layout_dir": str(layout_dir),
        "prompt": prompt,
    }

    # 3. Run layout extraction
    if run_extraction:
        sys.path.insert(0, str(_LLM_LAYOUT))
        from extract_object_layouts_v2 import (
            parse_prompt_into_clauses,
            extract_key_noun_from_clause,
            load_all_tokens,
            reconstruct_text_from_tokens,
            find_clause_token_range,
            aggregate_attention_for_tokens,
            compute_segmentation_mask,
            compute_bounding_box_from_mask,
            scale_mask_to_image,
            visualize_segmentation_masks,
            save_individual_segmentation_masks,
        )
        import json

        base_dir = Path(attn_dir)
        all_tokens = load_all_tokens(base_dir)
        full_text = reconstruct_text_from_tokens(all_tokens)
        clauses = parse_prompt_into_clauses(prompt)

        layouts = {}
        layout_dir = Path(layout_dir)
        layout_dir.mkdir(parents=True, exist_ok=True)

        for i, clause in enumerate(clauses, 1):
            obj_name = extract_key_noun_from_clause(clause)
            token_ids = find_clause_token_range(clause, all_tokens, full_text)
            if not token_ids:
                continue

            attn_map = aggregate_attention_for_tokens(
                base_dir, token_ids, min_t=0.6, max_t=0.8
            )
            seg_mask = compute_segmentation_mask(
                attn_map, threshold=0.5, percentile=98.0
            )
            bbox = compute_bounding_box_from_mask(seg_mask)

            layout_key = f"{i:02d}_{obj_name}"
            layouts[layout_key] = {
                "clause": clause,
                "object_name": obj_name,
                "token_ids": token_ids,
                "num_tokens": len(token_ids),
                "segmentation_mask": seg_mask,
                "bbox": bbox,
                "attention_map": attn_map,
            }

        if layouts:
            json_path = layout_dir / "layouts.json"
            json_data = {
                k: {
                    "clause": v["clause"],
                    "object_name": v["object_name"],
                    "num_tokens": v["num_tokens"],
                    "bbox": v["bbox"],
                }
                for k, v in layouts.items()
            }
            json_path.write_text(json.dumps(json_data, indent=2))

            image_path = Path(img_path)
            viz_path = layout_dir / "segmentation_overlay.png"
            visualize_segmentation_masks(
                layouts, viz_path, image_path=image_path
            )
            masks_dir = layout_dir / "segmentation_masks"
            save_individual_segmentation_masks(
                layouts, masks_dir, image_path=image_path
            )

        result["layouts"] = layouts
        result["num_objects"] = len(layouts)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate layout from a text prompt using FLUX.2-dev"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Output name (default: slug from prompt)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)",
    )
    parser.add_argument(
        "--models-cache",
        type=str,
        default=str(MODELS_CACHE),
        help="Model cache directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )
    parser.add_argument(
        "--no-extraction",
        action="store_true",
        help="Skip layout extraction (only generate image + attention maps)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale",
    )
    args = parser.parse_args()

    result = generate_layout_from_prompt(
        prompt=args.prompt,
        output_name=args.name,
        output_dir=args.output_dir,
        models_cache=args.models_cache,
        device=args.device,
        run_extraction=not args.no_extraction,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )

    print("\nDone.")
    print(f"  Image: {result['image_path']}")
    print(f"  Attention maps: {result['attn_dir']}")
    print(f"  Layouts: {result['layout_dir']}")
    if "num_objects" in result:
        print(f"  Extracted {result['num_objects']} object layouts")


if __name__ == "__main__":

    result = generate_layout_from_prompt(
            prompt="An underwater cinematic shot of a submerged room with a coral-covered brass lamp, a drifting wooden chair, a seaweed-covered table, a barnacle-encrusted bookshelf, a slowly rotating ceiling fan, a cracked wall clock with floating parts, a silt-covered floor rug, a broken window with light rays filtering through water, a glowing bedside lamp, a partially open wardrobe with clothes drifting out, a foggy mirror, a potted plant overtaken by algae, a waterlogged laptop, a scattered stack of soaked books, a floating coffee mug, a rusted table fan, a tipped-over shoe rack, a faded framed painting, a drifting pen holder, and a small storage box half-buried in sand",
            output_name="my_scene_underwater_1",
    )