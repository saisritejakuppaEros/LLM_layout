"""
Stage 2: monocular relative depth from a raster image using Depth Anything 3 (DA3-BASE).

Requires: pip install -r requirements-da3.txt
Hub: https://huggingface.co/depth-anything/DA3-BASE

Depth PNG format matches training consumption in ``train/src/canvas_dataset.py``:
``load_depth_image_as_rgb_pil`` — **16-bit grayscale** ``I;16`` with linear values in
``[0, 65535]`` (not false-color RGB; avoids Pillow ``L``/``I;16`` mishandling described there).

Environment (optional):
  INPUT_IMAGE   — input RGB path (default: flux2_output.png next to this file)
  OUTPUT_DEPTH  — output depth PNG (default: <stem>_depth.png beside input)
  DA3_MODEL     — Hugging Face repo id (default: depth-anything/DA3-BASE)
  SAVE_RAW_NPY  — if set to 1, also writes <stem>_depth_raw.npy (float32 [H,W])
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_INFER_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT = _INFER_DIR / "flux2_output.png"
_DEFAULT_MODEL = "depth-anything/DA3-BASE"


def _relative_depth_to_u16_hw(depth_hw: np.ndarray) -> np.ndarray:
    """Map relative depth (float) to uint16 [0, 65535] for ``I;16`` PNG (same semantics as training depth assets)."""
    d = depth_hw.astype(np.float32)
    lo, hi = float(np.percentile(d, 2.0)), float(np.percentile(d, 98.0))
    if hi <= lo + 1e-6:
        lo, hi = float(d.min()), float(d.max())
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    x = (d - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return np.round(x * 65535.0).astype(np.uint16)


def run_depth(
    input_path: Path,
    *,
    output_depth_png: Path | None = None,
    model_id: str = _DEFAULT_MODEL,
    save_raw_npy: bool = False,
    process_res: int = 504,
) -> Path:
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as e:
        print(
            "Missing package depth_anything_3. Install with:\n"
            f"  pip install -r {_INFER_DIR / 'requirements-da3.txt'}\n",
            file=sys.stderr,
        )
        raise e

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"INPUT_IMAGE not found: {input_path}")

    if output_depth_png is None:
        output_depth_png = input_path.with_name(f"{input_path.stem}_depth.png")
    else:
        output_depth_png = Path(output_depth_png).expanduser().resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DA3 model {model_id!r} on {device} …")
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)

    with Image.open(input_path) as pil_in:
        pil_in = pil_in.convert("RGB")
        orig_w, orig_h = pil_in.size

    prediction = model.inference(
        [str(input_path)],
        process_res=process_res,
        process_res_method="upper_bound_resize",
        export_dir=None,
    )
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    h, w = depth.shape

    depth_t = torch.from_numpy(depth)[None, None]
    depth_up = (
        F.interpolate(depth_t, size=(orig_h, orig_w), mode="bilinear", align_corners=False)[
            0, 0
        ]
        .cpu()
        .numpy()
    )

    u16 = _relative_depth_to_u16_hw(depth_up)
    out_img = Image.fromarray(u16, mode="I;16")
    output_depth_png.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(output_depth_png, compress_level=6)
    print(
        f"Saved depth PNG I;16 {orig_w}x{orig_h} (linear 0..65535, training-compatible) → {output_depth_png}"
    )

    if save_raw_npy:
        raw_path = input_path.with_name(f"{input_path.stem}_depth_raw.npy")
        np.save(raw_path, depth_up.astype(np.float32))
        print(f"Saved raw depth array to {raw_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_depth_png


def main() -> None:
    input_image = Path(os.environ.get("INPUT_IMAGE", str(_DEFAULT_INPUT))).expanduser()
    output_depth = os.environ.get("OUTPUT_DEPTH", "").strip()
    output_path = Path(output_depth) if output_depth else None
    model_id = os.environ.get("DA3_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    save_npy = os.environ.get("SAVE_RAW_NPY", "").strip() in ("1", "true", "yes")

    run_depth(
        input_image,
        output_depth_png=output_path,
        model_id=model_id,
        save_raw_npy=save_npy,
    )


if __name__ == "__main__":
    main()
