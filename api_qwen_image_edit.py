import base64
import io
from pathlib import Path

import torch
from diffusers import QwenImageEditPlusPipeline
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

EDIT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
CACHE_DIR = "./models/cache"
LOCAL_EDIT_MODEL_PATH = "./models/Qwen-Image-Edit-2511"

app = FastAPI()

print("Loading Qwen-Image-Edit...")

torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

edit_model_path = LOCAL_EDIT_MODEL_PATH if Path(LOCAL_EDIT_MODEL_PATH).exists() else EDIT_MODEL_ID
pipe = QwenImageEditPlusPipeline.from_pretrained(
    edit_model_path,
    torch_dtype=torch_dtype,
    cache_dir=CACHE_DIR if edit_model_path == EDIT_MODEL_ID else None,
)
pipe = pipe.to(device)

print("Image edit model loaded")


class Request(BaseModel):
    prompt: str
    image_path: str | None = None
    image_paths: list[str] | None = None
    width: int = 1536
    height: int = 1024
    negative_prompt: str = " "
    num_inference_steps: int = 40
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    seed: int | None = None


@app.post("/generate")
def generate(req: Request):
    if not req.image_path and not (req.image_paths and len(req.image_paths) > 0):
        raise HTTPException(422, "Either image_path or image_paths must be provided")

    paths = [req.image_path] if req.image_path else req.image_paths
    images = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise HTTPException(422, f"Image not found: {p}")
        images.append(Image.open(path).convert("RGB"))

    generator = torch.Generator(device=device).manual_seed(req.seed) if req.seed is not None else None

    with torch.inference_mode():
        output = pipe(
            image=images,
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            generator=generator,
            true_cfg_scale=req.true_cfg_scale,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            num_images_per_prompt=1,
        )

    buffer = io.BytesIO()
    output.images[0].save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"response": image_base64, "format": "png"}
