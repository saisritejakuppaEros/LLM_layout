#!/usr/bin/env python3
"""
Long-running HTTP service: load FLUX.2 once, then answer generation requests.

Based on the same ``Flux2Pipeline`` usage as ``lora_training_v2/infer/cultural_edit/stage1_gen.py``:
text-to-image by default; if one or more reference images are posted, they are passed as
``image=[PIL, ...]`` so the pipeline packs and concatenates reference latents (native multi-ref).

Dependencies: fastapi, uvicorn, python-multipart, torch, diffusers, pillow, transformers (for FLUX text encoder).
Run::

  export FLUX2_MODEL=/path/to/FLUX.2-dev
  python server_flux.py --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import argparse
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional

import torch
import uvicorn
from diffusers import Flux2Pipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

# --- Defaults aligned with stage1_gen.py -------------------------------------------------
FLUX2_PATH = os.environ.get(
    "FLUX2_MODEL",
    "/mnt/data0/teja/research_multiref/llm_based_layout/models/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
)
GEN_IMAGE_WIDTH = 1280
GEN_IMAGE_HEIGHT = 720

_pipe: Optional[Flux2Pipeline] = None
_infer_lock = threading.Lock()
_model_error: Optional[str] = None


def _device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_flux2_t2i_pipeline(
    *,
    pretrained_path: str | None = None,
    torch_dtype=torch.bfloat16,
) -> Flux2Pipeline:
    path = pretrained_path or FLUX2_PATH
    if not os.path.isdir(path):
        raise FileNotFoundError(f"FLUX2 model path is not a directory: {path}")
    t0 = time.perf_counter()
    print(f"[flux_server] Loading Flux2Pipeline from {path!r} (dtype={torch_dtype}) …", flush=True)
    pipe = Flux2Pipeline.from_pretrained(path, torch_dtype=torch_dtype)
    print(f"[flux_server] from_pretrained done in {time.perf_counter() - t0:.1f}s; enabling CPU offload …", flush=True)
    pipe.enable_model_cpu_offload()
    print(f"[flux_server] Model ready. Total load {time.perf_counter() - t0:.1f}s", flush=True)
    return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe, _model_error
    _model_error = None
    print(f"[flux_server] Startup: device={_device_str()} FLUX2_MODEL={FLUX2_PATH!r}", flush=True)
    try:
        _pipe = load_flux2_t2i_pipeline()
        print("[flux_server] Ready: POST /generate, GET /health", flush=True)
    except Exception as e:  # noqa: BLE001 — surface any load failure on /health
        _model_error = f"{type(e).__name__}: {e}"
        _pipe = None
        print(f"[flux_server] Model load FAILED: {_model_error}", flush=True)
    yield
    print("[flux_server] Shutting down; releasing pipeline reference.", flush=True)
    _pipe = None


app = FastAPI(title="FLUX.2 multi-ref server", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _pipe is not None and _model_error is None else "unavailable",
        "model_loaded": _pipe is not None,
        "error": _model_error,
        "device": _device_str(),
    }


@app.post("/generate")
async def generate(
    caption: str = Form(..., description="Scene / generation prompt (caption)."),
    seed: int = Form(42),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(4.0),
    width: int = Form(GEN_IMAGE_WIDTH),
    height: int = Form(GEN_IMAGE_HEIGHT),
    caption_upsample_temperature: Optional[float] = Form(
        None, description="Optional; e.g. 0.15 to enable caption upsampling (uses ref images if any)."
    ),
    ref_images: Annotated[
        list[UploadFile],
        File(description="Zero or more reference images (RGB); repeat form field ref_images."),
    ] = [],
) -> Response:
    if _pipe is None:
        print("[flux_server] /generate rejected: model not loaded", flush=True)
        raise HTTPException(
            status_code=503,
            detail=_model_error or "Model not loaded",
        )

    pil_list: list[Image.Image] = []
    for uf in ref_images:
        if not uf.filename:
            continue
        data = await uf.read()
        if not data:
            continue
        try:
            pil_list.append(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid image {uf.filename!r}: {e}") from e

    dev = _device_str()
    gen = torch.Generator(device=dev).manual_seed(int(seed))

    call_kw: dict[str, Any] = {
        "prompt": caption,
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "height": int(height),
        "width": int(width),
        "generator": gen,
    }
    if pil_list:
        # List of PIL images → multiple ref latents concatenated (see diffusers Flux2 ``prepare_image_latents``).
        call_kw["image"] = pil_list
    if caption_upsample_temperature is not None:
        call_kw["caption_upsample_temperature"] = float(caption_upsample_temperature)

    cap_preview = (caption[:200] + "…") if len(caption) > 200 else caption
    cap_preview = cap_preview.replace("\n", " ")
    print(
        f"[flux_server] /generate: {width}×{height} steps={num_inference_steps} "
        f"guidance={guidance_scale} seed={seed} refs={len(pil_list)} "
        f"upsample={caption_upsample_temperature!r}",
        flush=True,
    )
    print(f"[flux_server]   prompt: {cap_preview!r}", flush=True)

    def _run() -> Image.Image:
        assert _pipe is not None
        return _pipe(**call_kw).images[0]

    t_inf = time.perf_counter()
    with _infer_lock:
        try:
            print("[flux_server]   inference: acquiring lock, running pipeline …", flush=True)
            image_out = _run()
        except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
            print(f"[flux_server]   inference: CUDA OOM: {e}", flush=True)
            raise HTTPException(status_code=507, detail=f"CUDA OOM: {e}") from e
        except Exception as e:  # noqa: BLE001
            print(f"[flux_server]   inference: ERROR {type(e).__name__}: {e}", flush=True)
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    elapsed = time.perf_counter() - t_inf
    buf = io.BytesIO()
    image_out.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    print(f"[flux_server]   done in {elapsed:.1f}s; PNG {len(png_bytes) // 1024} KiB", flush=True)
    return Response(content=png_bytes, media_type="image/png")


def main() -> None:
    parser = argparse.ArgumentParser(description="FLUX.2 HTTP server (multi-ref + caption).")
    parser.add_argument("--host", default=os.environ.get("FLUX_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FLUX_SERVER_PORT", "8765")))
    args = parser.parse_args()
    print(
        f"[flux_server] Uvicorn starting on {args.host}:{args.port} (model loads in startup; then /health = ready).",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
