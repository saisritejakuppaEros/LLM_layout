# Ensure sibling ``layout_canvas_utils`` imports when cwd != this directory.
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from layout_canvas_utils import (
    OBJECT_DETECTION_QUERIES,
    compose_layout_from_bbox_csv,
    read_bbox_csv as _read_bbox_csv,
)

# Cutouts in ``bg_removed/`` (only these two assets). Keys must match
# ``CULTURAL_DETECTION_QUERIES``; phrases should match what Grounding DINO
# returns for your FLUX scene (substring match in ``resolve_label_to_object_key``).
BG_REMOVED_DIR = _SCRIPT_DIR / "bg_removed"
CULTURAL_OBJECTS_DICT: dict[str, str] = {
    "boy": "upanayanam.png",
    "celebrity": "srk_2.png",
}
CULTURAL_DETECTION_QUERIES: dict[str, str] = {
    "boy": "a young boy in traditional white clothing sitting cross legged",
    "celebrity": "a man wearing an elegant deep blue sherwani",
}

ai_prompt = """
A realistic Indian cultural ceremony scene set in a softly lit traditional indoor environment. In the center, a young boy sits cross-legged on the floor wearing traditional white attire, participating in a sacred thread ceremony (upanayanam). A middle-aged priest/father figure, bare-chested with a dhoti, is carefully guiding the ritual, holding and placing a sacred thread (yajnopavita) across the boy’s shoulder with focused attention. Ritual items like a small fire (havan), brass vessels, flowers, and offerings are placed neatly around them on a mat.

To the side, Shah Rukh Khan stands slightly behind and to the right, dressed in an elegant deep blue sherwani with black trousers. He is observing the ceremony with a calm, respectful expression, hands relaxed, slightly leaning forward as if attentively watching the ritual unfold.

Lighting is warm and natural, coming from the side, casting soft shadows. The composition is balanced, with depth and perspective maintained so all subjects feel naturally placed in the same environment. The background includes subtle traditional decor like muted walls, soft textures, and possibly a temple-like ambiance. Skin tones, shadows, and reflections are realistic, ensuring seamless blending of all individuals into a single cohesive scene.

Style: photorealistic, high detail, natural skin tones, cinematic lighting, 35mm lens feel, shallow depth of field.
"""



import csv
import gc
import os

import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import Flux2Pipeline
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Local FLUX.2-dev snapshot; override with FLUX2_MODEL (same as batch_run.sh / run.sh).
flux2_path = os.environ.get(
    "FLUX2_MODEL",
    "/mnt/data0/teja/research_multiref/llm_based_layout/models/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
)

# Reused across warm-batch layout jobs (see ``load_flux2_t2i_pipeline``).
_t2i_pipe: Flux2Pipeline | None = None


def load_flux2_t2i_pipeline(
    *,
    pretrained_path: str | None = None,
    torch_dtype=torch.bfloat16,
) -> Flux2Pipeline:
    """Load and cache a single ``Flux2Pipeline`` for repeated ``generate_image`` calls."""
    global _t2i_pipe
    if _t2i_pipe is not None:
        return _t2i_pipe
    path = pretrained_path or flux2_path
    pipe = Flux2Pipeline.from_pretrained(path, torch_dtype=torch_dtype)
    pipe.enable_model_cpu_offload()
    _t2i_pipe = pipe
    return _t2i_pipe


def unload_flux2_t2i_pipeline() -> None:
    """Release cached T2I pipeline to free VRAM before loading the LoRA inference stack."""
    global _t2i_pipe
    if _t2i_pipe is None:
        return
    del _t2i_pipe
    _t2i_pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_image(
    ai_prompt,
    image_path,
    *,
    pipe: Flux2Pipeline | None = None,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    height: int = 720,
    width: int = 1280,
):
    """
    Text-to-image with FLUX.2. If ``pipe`` is None, loads a one-off pipeline and unloads it.
    If ``pipe`` is provided (from ``load_flux2_t2i_pipeline``), reuses it for faster batch runs.
    """
    device = "cuda"
    dtype = torch.bfloat16
    own_pipe = pipe is None
    if own_pipe:
        pipe = Flux2Pipeline.from_pretrained(flux2_path, torch_dtype=dtype)
        pipe.enable_model_cpu_offload()

    try:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        image = pipe(
            prompt=ai_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        ).images[0]
        out = Path(image_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        print(f"Saved to {image_path}")
    finally:
        if own_pipe:
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_layout_stages(
    scene_prompt: str,
    *,
    image_path: str | Path,
    bbox_csv: str | Path,
    prompt_file: str | Path,
    pipe: Flux2Pipeline | None = None,
    layout_seed: int = 42,
    asset_dir: str | Path | None = None,
    objects_dict: dict[str, str] | None = None,
    detection_queries: dict[str, str] | None = None,
) -> Path:
    """
    Full layout path: T2I → bbox CSV → debug overlay → composite PNG → write ``prompt_file``.

    Returns path to the layout composite PNG (``*_layout_composite.png`` next to ``bbox_csv``).
    """
    image_path = Path(image_path)
    bbox_csv = Path(bbox_csv)
    prompt_file = Path(prompt_file)
    prompt_file.parent.mkdir(parents=True, exist_ok=True)

    generate_image(
        scene_prompt,
        str(image_path),
        pipe=pipe,
        seed=layout_seed,
    )
    get_bbox(
        str(image_path),
        output_csv=str(bbox_csv),
        detection_queries=detection_queries,
    )
    debug_png = image_path.with_name(f"{image_path.stem}_debug_bbox.png")
    debug_bboxes(str(image_path), str(bbox_csv), draw_output_path=str(debug_png), draw=True)
    compose_layout_from_bbox_csv(
        str(bbox_csv),
        reference_image_path=str(image_path),
        asset_dir=asset_dir,
        objects_dict=objects_dict,
        detection_queries=detection_queries,
    )
    prompt_file.write_text(scene_prompt.strip(), encoding="utf-8")
    print(f"Wrote {prompt_file} for LoRA --prompt_file")
    composite = bbox_csv.with_name(f"{bbox_csv.stem}_layout_composite.png")
    return composite


GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

_grounding_processor = None
_grounding_model = None


def _get_grounding_dino(device):
    global _grounding_processor, _grounding_model
    if _grounding_model is None:
        _grounding_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
        _grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_MODEL_ID
        ).to(device)
        _grounding_model.eval()
    return _grounding_processor, _grounding_model


def get_bbox(
    image_path,
    output_csv=None,
    threshold=0.4,
    text_threshold=0.3,
    detection_queries: dict[str, str] | None = None,
):
    """
    Run Grounding DINO on ``image_path`` and write detections to a CSV file.

    Columns: ``label``, ``score``, ``x_min``, ``y_min``, ``x_max``, ``y_max``.
    Default CSV path: ``<image_stem>_bbox.csv`` next to the image.
    """
    path = Path(image_path)
    if output_csv is None:
        output_csv = path.with_name(f"{path.stem}_bbox.csv")
    else:
        output_csv = Path(output_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = _get_grounding_dino(device)

    image = Image.open(path).convert("RGB")
    queries = detection_queries if detection_queries is not None else OBJECT_DETECTION_QUERIES
    text_labels = [list(queries.values())]

    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )

    result = results[0]
    rows = []
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        x_min, y_min, x_max, y_max = [round(float(x), 2) for x in box.tolist()]
        rows.append(
            {
                "label": label,
                "score": round(score.item(), 4),
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label", "score", "x_min", "y_min", "x_max", "y_max"]
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"Detected {row['label']} with confidence {row['score']} "
            f"at [{row['x_min']}, {row['y_min']}, {row['x_max']}, {row['y_max']}]"
        )
    print(f"Saved bbox CSV to {output_csv}")
    return rows, output_csv


def _font(size=16):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    )
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_bboxes_from_csv(
    image_path,
    csv_path,
    output_path=None,
    line_width=3,
    show_score=True,
):
    """
    Load ``csv_path`` (same format as ``get_bbox``) and draw rectangles with
    labels above each box on ``image_path``.

    Saves to ``output_path`` if given, else ``<image_stem>_annotated.png`` beside the image.
    Returns ``(annotated_image, output_path)``.
    """
    img_path = Path(image_path)
    rows = _read_bbox_csv(csv_path)
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    draw = ImageDraw.Draw(image)
    font = _font(18)

    palette = [
        (220, 50, 50),
        (50, 180, 80),
        (50, 120, 220),
        (200, 140, 40),
        (160, 60, 200),
        (40, 180, 180),
        (220, 120, 180),
    ]

    for i, row in enumerate(rows):
        x0 = max(0, min(w - 1, int(round(row["x_min"]))))
        y0 = max(0, min(h - 1, int(round(row["y_min"]))))
        x1 = max(0, min(w - 1, int(round(row["x_max"]))))
        y1 = max(0, min(h - 1, int(round(row["y_max"]))))
        color = palette[i % len(palette)]

        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

        cap = row["label"]
        if show_score:
            cap = f"{row['label']} {row['score']:.2f}"
        bbox_text = draw.textbbox((0, 0), cap, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        pad = 3
        tx = x0
        ty = y0 - th - 2 * pad
        if ty < 0:
            ty = y0 + pad
        bg = [tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad]
        draw.rectangle(bg, fill=(20, 20, 20))
        draw.text((tx + pad, ty + pad), cap, fill=(255, 255, 255), font=font)

    out = Path(output_path) if output_path else img_path.with_name(f"{img_path.stem}_annotated.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"Saved annotated image to {out}")
    return image, out


def debug_bboxes(image_path, csv_path, draw_output_path=None, draw=True):
    """
    Print bbox CSV diagnostics vs image size, clamping notes, and per-row checks.
    If ``draw`` is True, also runs ``draw_bboxes_from_csv`` and saves to
    ``draw_output_path`` or ``<stem>_debug_bbox.png``.
    """
    img_path = Path(image_path)
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    rows = _read_bbox_csv(csv_path)

    print(f"[debug_bboxes] image: {img_path} size={w}x{h}")
    print(f"[debug_bboxes] csv: {csv_path} rows={len(rows)}")
    if not rows:
        print("[debug_bboxes] no rows in CSV; nothing to draw.")
        return {"image_size": (w, h), "rows": [], "issues": ["empty CSV"], "annotated_path": None}

    issues = []
    for idx, row in enumerate(rows):
        raw = {k: row[k] for k in ("x_min", "y_min", "x_max", "y_max")}
        for name, val in raw.items():
            if val < 0:
                issues.append(f"row {idx} {name}={val} (negative)")
        if row["x_max"] <= row["x_min"] or row["y_max"] <= row["y_min"]:
            issues.append(f"row {idx} degenerate box: {raw}")
        for name, val in (("x_min", row["x_min"]), ("x_max", row["x_max"])):
            if val > w:
                issues.append(f"row {idx} {name}={val} beyond width {w}")
        for name, val in (("y_min", row["y_min"]), ("y_max", row["y_max"])):
            if val > h:
                issues.append(f"row {idx} {name}={val} beyond height {h}")
        print(
            f"  [{idx}] label={row['label']!r} score={row['score']:.4f} "
            f"box=({row['x_min']:.1f},{row['y_min']:.1f},{row['x_max']:.1f},{row['y_max']:.1f})"
        )

    if issues:
        print("[debug_bboxes] issues:")
        for msg in issues:
            print(f"  - {msg}")
    else:
        print("[debug_bboxes] no coordinate issues detected.")

    out_path = None
    if draw:
        default_out = img_path.with_name(f"{img_path.stem}_debug_bbox.png")
        out_path = Path(draw_output_path) if draw_output_path else default_out
        draw_bboxes_from_csv(image_path, csv_path, output_path=out_path)
        print(f"[debug_bboxes] annotated preview: {out_path}")

    return {"image_size": (w, h), "rows": rows, "issues": issues, "annotated_path": out_path}




if __name__ == "__main__":
    # run.sh sets these so layout artifacts live under outputs/layout_utils/.
    image_path = os.environ.get("IMAGE_PATH", "flux2_output.png")
    bbox_csv = os.environ.get("BBOX_CSV", "flux2_output_bbox.csv")

    prompt_path = os.environ.get("AI_PROMPT_FILE", "").strip()
    if prompt_path and Path(prompt_path).is_file():
        scene_prompt = Path(prompt_path).read_text(encoding="utf-8").strip()
    else:
        env_prompt = os.environ.get("AI_PROMPT", "").strip()
        scene_prompt = env_prompt or ai_prompt.strip()

    # Optional: full pipeline (FLUX generation + detection; needs GPU, slow).
    prompt_file = Path(os.environ.get("PROMPT_FILE", "scene_prompt.txt"))
    use_cultural = os.environ.get("USE_CULTURAL_CUTOUTS", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    asset_dir = BG_REMOVED_DIR if use_cultural else None
    objects_dict = CULTURAL_OBJECTS_DICT if use_cultural else None
    detection_queries = CULTURAL_DETECTION_QUERIES if use_cultural else None
    if use_cultural and not BG_REMOVED_DIR.is_dir():
        raise FileNotFoundError(f"Expected cutout directory: {BG_REMOVED_DIR}")
    for fname in (objects_dict or {}).values():
        p = (asset_dir or _SCRIPT_DIR) / fname
        if use_cultural and not p.is_file():
            raise FileNotFoundError(f"Missing reference cutout: {p}")

    composite = run_layout_stages(
        scene_prompt,
        image_path=image_path,
        bbox_csv=bbox_csv,
        prompt_file=prompt_file,
        pipe=None,
        layout_seed=42,
        asset_dir=asset_dir,
        objects_dict=objects_dict,
        detection_queries=detection_queries,
    )
    print(f"Wrote {prompt_file} for --prompt_file with flux2_lora_inference.py")
    print(
        "Next: python flux2_lora_inference.py --lora_path <.../checkpoint-*/lora.safetensors> "
        f"--canvas_image {composite} --prompt_file {prompt_file} --output multiref_gen.png"
    )