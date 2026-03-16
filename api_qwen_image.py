import base64
import io
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_ID = "Qwen/Qwen-Image"
CACHE_DIR = "./models/cache"
# Cached model path (snapshot from models/cache/models--Qwen--Qwen-Image)
LOCAL_MODEL_PATH = "./models/cache/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6"

# Positive prompt suffix for better quality (from Qwen-Image docs)
POSITIVE_MAGIC = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图.",
}

# Supported aspect ratios (width, height)
ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

app = FastAPI()

print("Loading Qwen-Image...")

torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = LOCAL_MODEL_PATH if Path(LOCAL_MODEL_PATH).exists() else MODEL_ID
pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch_dtype,
    cache_dir=CACHE_DIR if model_path == MODEL_ID else None,
)
pipe = pipe.to(device)

print("Qwen-Image loaded")


class Request(BaseModel):
    prompt: str
    aspect_ratio: str = "16:9"
    width: int | None = None
    height: int | None = None
    negative_prompt: str = " "
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    seed: int | None = None
    add_positive_magic: bool = True
    positive_magic_lang: str = "en"


@app.post("/generate")
def generate(req: Request):
    prompt = req.prompt
    if req.add_positive_magic:
        magic = POSITIVE_MAGIC.get(req.positive_magic_lang, POSITIVE_MAGIC["en"])
        prompt = prompt + magic

    if req.width is not None and req.height is not None:
        width, height = req.width, req.height
    else:
        width, height = ASPECT_RATIOS.get(req.aspect_ratio, ASPECT_RATIOS["16:9"])

    generator = torch.Generator(device=device).manual_seed(req.seed) if req.seed is not None else None

    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=req.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=req.num_inference_steps,
            true_cfg_scale=req.true_cfg_scale,
            generator=generator,
        )

    buffer = io.BytesIO()
    output.images[0].save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"response": image_base64, "format": "png"}
