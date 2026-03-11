import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"
CACHE_DIR = "./models/cache"

app = FastAPI()

print("Loading Qwen3-8B...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    torch_dtype="auto",
    device_map="auto"
)

print("Model Loaded")

class Request(BaseModel):
    prompt: str
    max_tokens: int = 512


@app.post("/generate")
def generate(req: Request):

    messages = [{"role": "user", "content": req.prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens
    )

    ids = output[0][len(inputs.input_ids[0]):]

    response = tokenizer.decode(ids, skip_special_tokens=True)

    return {"response": response}
