"""
Reference-image conditioned generation with FLUX.2 (same model path as stage1_genimage.py).
"""
import gc
import os
from pathlib import Path

import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image

# Local FLUX.2-dev snapshot; override with FLUX2_MODEL (same as stage1_genimage.py / batch_run.sh).
flux2_path = os.environ.get(
    "FLUX2_MODEL",
    "/mnt/data0/teja/research_multiref/llm_based_layout/models/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
)

DEFAULT_REF_IMAGE = (
    "/mnt/data0/teja/research_multiref/lora_training_v2/infer/background_remove_imgs/srk.png"
)

# Same default scene prompt as stage1_genimage.py (text-to-image layout path).
ai_prompt = """
Use the person, pose, clothing style, and overall mood from the reference image.

A clean, well-composed scene featuring:
- an exotic candle stand (no candle),
- a transparent drinking glass,
- a classic glass Coca-Cola bottle,
- a table with four legs,
- a chair,
- a ceiling fan above,
- and a single adult person.

The person should resemble the reference style: a South Asian male in traditional attire (kurta and dhoti), standing casually with folded arms, leaning slightly, with a thoughtful expression. Maintain a natural, grounded pose similar to the reference.

Scene styling should follow the reference image:
- cinematic black-and-white or muted tones,
- soft diffused lighting with mild shadows,
- slightly vintage architectural elements (pillars, textured walls, or heritage-style setting),
- subtle depth and realism.

All objects must remain clearly visible and properly arranged:
- candle stand, glass, and Coca-Cola bottle placed on the table,
- realistic proportions and spacing,
- no clutter, balanced composition.

Camera/style:
- medium shot, slightly dramatic angle,
- shallow depth of field,
- high detail, realistic textures,
- cinematic composition.

Ensure the scene remains minimal, grounded, and physically plausible while blending the reference image’s mood and character styling.
"""

def generate_image_with_reference(
    prompt: str,
    ref_image_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 2.5,
    height: int = 1024,
    width: int = 1024,
    pretrained_path: str | None = None,
) -> Path:
    """Load FLUX.2 once, run img2img-style conditioning with a single reference image, save result."""
    device = "cuda"
    dtype = torch.bfloat16
    path = pretrained_path or flux2_path
    ref_image = load_image(str(ref_image_path))

    pipe = Flux2Pipeline.from_pretrained(path, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()

    try:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        result = pipe(
            prompt=prompt,
            image=[ref_image],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(out)
        print(f"Saved to {out}")
        return out
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    ref_image = os.environ.get("REF_IMAGE_PATH", DEFAULT_REF_IMAGE)
    output_path = os.environ.get("OUTPUT_IMAGE", "flux2_ref_output.png")

    prompt_env = os.environ.get("AI_PROMPT", "").strip()
    prompt = prompt_env or ai_prompt.strip()

    generate_image_with_reference(
        prompt,
        ref_image_path=ref_image,
        output_path=output_path,
        seed=int(os.environ.get("SEED", "42")),
        num_inference_steps=int(os.environ.get("NUM_INFERENCE_STEPS", "50")),
        guidance_scale=float(os.environ.get("GUIDANCE_SCALE", "2.5")),
        height=int(os.environ.get("HEIGHT", "1024")),
        width=int(os.environ.get("WIDTH", "1024")),
    )
