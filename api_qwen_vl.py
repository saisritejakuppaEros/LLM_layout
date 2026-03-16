import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
CACHE_DIR = "./models/cache"

app = FastAPI()

print("Loading Qwen3-VL...")

processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    torch_dtype="auto",
    device_map="auto"
)

print("VL Model Loaded")


class Request(BaseModel):
    image_url: str | None = None
    image_urls: list[str] | None = None
    image_base64: str | None = None
    prompt: str = "Describe this image."
    max_tokens: int = 512


@app.post("/generate")
def generate(req: Request):
    content = []

    if req.image_urls and len(req.image_urls) > 0:
        for url in req.image_urls:
            content.append({"type": "image", "image": url})
    elif req.image_base64:
        image_input = f"data:image/png;base64,{req.image_base64}"
        content.append({"type": "image", "image": image_input})
    elif req.image_url:
        content.append({"type": "image", "image": req.image_url})
    else:
        raise HTTPException(422, "Either image_url, image_urls, or image_base64 must be provided")

    content.append({"type": "text", "text": req.prompt})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=req.max_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return {"response": output_text[0]}
