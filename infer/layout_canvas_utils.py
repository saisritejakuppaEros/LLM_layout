"""
Compose a black canvas by placing cutout assets into bbox regions from a CSV.

CSV format matches ``get_bbox`` output: ``label``, ``score``, ``x_min``, ``y_min``,
``x_max``, ``y_max``. Each object image is **cover**-fitted to its box (uniform
scale, center crop) so aspect ratio is preserved.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from PIL import Image

# Default asset filenames (keys match detector object ids / resolved labels).
# PNGs from mask_bg.py under ``background_remove_imgs`` (same basenames as images_set).
OBJECTS_DICT: dict[str, str] = {
    "candle_stand": "candle_stand.png",
    "glass": "cello_glass.png",
    "bottle": "coke_bottle.png",
    "table": "durain_table_frontview.png",
    "chair": "nilkamal_chair.png",
    "man": "srk.png",
    "ceiling_fan": "usha_ceiling_fan.png",
}

# Phrases Grounding DINO uses; values must match CSV ``label`` text when possible.
OBJECT_DETECTION_QUERIES: dict[str, str] = {
    "candle_stand": "a candle stand",
    "glass": "a drinking glass",
    "bottle": "a bottle",
    "table": "a wooden table",
    "chair": "a chair",
    "man": "a man",
    "ceiling_fan": "a ceiling fan",
}

DEFAULT_ASSET_DIR = Path(
    "/mnt/data0/teja/multiref_image/inference_data/ads/set_1/background_remove_imgs"
)
DEFAULT_FLUX_CANVAS_SIZE = (1280, 720)


def read_bbox_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """Load bbox rows from CSV (same columns as ``get_bbox``)."""
    path = Path(csv_path)
    rows: list[dict[str, Any]] = []
    required = {"label", "score", "x_min", "y_min", "x_max", "y_max"}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV must have columns {sorted(required)}, got {reader.fieldnames}"
            )
        for row in reader:
            rows.append(
                {
                    "label": str(row["label"]).strip(),
                    "score": float(row["score"]),
                    "x_min": float(row["x_min"]),
                    "y_min": float(row["y_min"]),
                    "x_max": float(row["x_max"]),
                    "y_max": float(row["y_max"]),
                }
            )
    return rows


def resolve_label_to_object_key(
    label: str,
    queries: dict[str, str] | None = None,
) -> str | None:
    """
    Map a CSV ``label`` (e.g. ``a wooden table``) to an ``OBJECTS_DICT`` key
    (e.g. ``table``).
    """
    queries = queries or OBJECT_DETECTION_QUERIES
    ln = label.strip().lower()
    if not ln:
        return None

    for key, phrase in queries.items():
        pl = phrase.strip().lower()
        if pl == ln or pl in ln or ln in pl:
            return key

    for key in queries:
        underscored = key.replace("_", " ")
        if underscored in ln or key in ln.replace(" ", "_").lower():
            return key

    return None


def _clamp_box_int(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    canvas_w: int,
    canvas_h: int,
) -> tuple[int, int, int, int]:
    x0 = max(0, min(canvas_w - 1, int(round(x_min))))
    y0 = max(0, min(canvas_h - 1, int(round(y_min))))
    x1 = max(0, min(canvas_w - 1, int(round(x_max))))
    y1 = max(0, min(canvas_h - 1, int(round(y_max))))
    if x1 <= x0:
        x1 = min(canvas_w - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(canvas_h - 1, y0 + 1)
    return x0, y0, x1, y1


def rgba_cover_to_box(src: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """
    Scale ``src`` uniformly so it fully covers ``box_w`` × ``box_h``, then
    center-crop to that size. Preserves aspect ratio (no distortion).
    """
    src = src.convert("RGBA")
    iw, ih = src.size
    if iw < 1 or ih < 1:
        raise ValueError(f"Invalid source size {iw}x{ih}")
    if box_w < 1 or box_h < 1:
        raise ValueError(f"Invalid box size {box_w}x{box_h}")

    scale = max(box_w / iw, box_h / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)

    left = (nw - box_w) // 2
    top = (nh - box_h) // 2
    right = left + box_w
    bottom = top + box_h
    # Clamp if rounding made crop larger than resized image
    if right > nw:
        left = max(0, nw - box_w)
        right = nw
    if bottom > nh:
        top = max(0, nh - box_h)
        bottom = nh
    return resized.crop((left, top, right, bottom))


def paste_object_to_canvas(canvas: Image.Image, src: Image.Image, bbox: list[float] | tuple[float, float, float, float]) -> Image.Image:
    """
    Paste a single object onto a canvas at the specified bbox, ensuring aspect ratio
    and proper clamping to canvas bounds.
    """
    canvas_w, canvas_h = canvas.size
    x0, y0, x1, y1 = _clamp_box_int(
        bbox[0], bbox[1], bbox[2], bbox[3],
        canvas_w, canvas_h
    )
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w < 1 or box_h < 1:
        return canvas

    fitted = rgba_cover_to_box(src, box_w, box_h)
    # Ensure canvas is in RGBA for transparency handling if needed, but we'll convert back if caller expects RGB
    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")
    
    canvas.paste(fitted, (x0, y0), mask=fitted.split()[3] if fitted.mode == "RGBA" else None)
    return canvas

def compose_layout_from_bbox_csv(
    bbox_csv_path: str | Path,
    *,
    asset_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    canvas_size: tuple[int, int] | None = None,
    reference_image_path: str | Path | None = None,
    objects_dict: dict[str, str] | None = None,
    detection_queries: dict[str, str] | None = None,
    layer_order: str = "area_desc",
) -> tuple[Image.Image, Path]:
    """
    Build a black RGB canvas, then paste each asset into the bbox from the CSV.

    Parameters
    ----------
    bbox_csv_path
        Path to ``*_bbox.csv`` (``label``, ``score``, ``x_min``, ``y_min``,
        ``x_max``, ``y_max``).
    asset_dir
        Directory containing cutout images (see ``OBJECTS_DICT`` filenames).
    output_path
        Where to save the composed PNG. Default: ``<csv_stem>_layout_composite.png``
        next to the CSV.
    canvas_size
        ``(width, height)`` in pixels. Ignored if ``reference_image_path`` is set.
    reference_image_path
        If set, canvas size matches this image (same coords as bbox detection).
    objects_dict
        Maps object keys to filenames. Defaults to :data:`OBJECTS_DICT`.
    detection_queries
        Maps object keys to detector label phrases. Defaults to
        :data:`OBJECT_DETECTION_QUERIES`.
    layer_order
        ``"csv"`` — paste in CSV row order. ``"area_desc"`` — larger boxes first
        (typical back-to-front). ``"area_asc"`` — smaller boxes last on top.

    Returns
    -------
    Composed PIL image and path written to disk.
    """
    csv_path = Path(bbox_csv_path)
    asset_dir = Path(asset_dir) if asset_dir is not None else DEFAULT_ASSET_DIR
    objects_dict = objects_dict or OBJECTS_DICT
    detection_queries = detection_queries or OBJECT_DETECTION_QUERIES

    if reference_image_path is not None:
        ref = Image.open(reference_image_path).convert("RGB")
        canvas_w, canvas_h = ref.size
    elif canvas_size is not None:
        canvas_w, canvas_h = int(canvas_size[0]), int(canvas_size[1])
    else:
        canvas_w, canvas_h = DEFAULT_FLUX_CANVAS_SIZE

    rows = read_bbox_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")

    indexed: list[tuple[int, dict[str, Any]]] = list(enumerate(rows))
    if layer_order == "csv":
        ordered = [r for _, r in indexed]
    elif layer_order == "area_desc":
        ordered = sorted(
            rows,
            key=lambda r: (r["x_max"] - r["x_min"]) * (r["y_max"] - r["y_min"]),
            reverse=True,
        )
    elif layer_order == "area_asc":
        ordered = sorted(
            rows,
            key=lambda r: (r["x_max"] - r["x_min"]) * (r["y_max"] - r["y_min"]),
        )
    else:
        raise ValueError("layer_order must be 'csv', 'area_desc', or 'area_asc'")

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))

    for row in ordered:
        key = resolve_label_to_object_key(row["label"], detection_queries)
        if key is None:
            print(f"[compose_layout] skip unknown label: {row['label']!r}")
            continue
        fname = objects_dict.get(key)
        if not fname:
            print(f"[compose_layout] no asset for key: {key!r}")
            continue

        asset_path = asset_dir / fname
        if not asset_path.is_file():
            print(f"[compose_layout] missing file: {asset_path}")
            continue

        x0, y0, x1, y1 = _clamp_box_int(
            row["x_min"],
            row["y_min"],
            row["x_max"],
            row["y_max"],
            canvas_w,
            canvas_h,
        )
        box_w = x1 - x0
        box_h = y1 - y0
        if box_w < 1 or box_h < 1:
            continue

        src = Image.open(asset_path)
        fitted = rgba_cover_to_box(src, box_w, box_h)
        canvas.paste(fitted, (x0, y0), mask=fitted.split()[3])

    out = (
        Path(output_path)
        if output_path
        else csv_path.with_name(f"{csv_path.stem}_layout_composite.png")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb_out = canvas.convert("RGB")
    rgb_out.save(out)
    print(f"Saved layout composite to {out}")
    return rgb_out, out
