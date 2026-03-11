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
    image_base64: str | None = None
    prompt: str = "Describe this image."
    max_tokens: int = 512


@app.post("/generate")
def generate(req: Request):
    if req.image_base64:
        image_input = f"data:image/png;base64,{req.image_base64}"
    elif req.image_url:
        image_input = req.image_url
    else:
        raise HTTPException(422, "Either image_url or image_base64 must be provided")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_input},
                {"type": "text", "text": req.prompt},
            ],
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
