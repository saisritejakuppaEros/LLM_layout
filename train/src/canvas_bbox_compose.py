"""Build 1920×1080 conditioning canvas: black (or scaled scene) + bbox crops / optional multiview.

Bboxes are mapped with **uniform scale + centered letterboxing** (same as ``contain``): the full frame keeps
aspect ratio inside the canvas; black bars fill the rest. This matches CSV coordinates in the **original**
image space without stretching non-16:9 sources.

Aligned with ``dataset_preparation/stage4_canva.py`` (same paste order, multiview layout, and letterbox math).

Training: ``CanvasSceneDataset`` letterboxes target + this canvas again into the unified /16 tensor so both
branches stay paired."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

CANVAS_W = 1920
CANVAS_H = 1080


def _f(x: Any) -> float:
    return float(x)


def clamp_xyxy(
    x1: float, y1: float, x2: float, y2: float, iw: int, ih: int
) -> Tuple[int, int, int, int]:
    x1i = max(0, min(int(round(x1)), iw - 1))
    y1i = max(0, min(int(round(y1)), ih - 1))
    x2i = max(x1i + 1, min(int(round(x2)), iw))
    y2i = max(y1i + 1, min(int(round(y2)), ih))
    return x1i, y1i, x2i, y2i


def letterbox_layout(iw: int, ih: int) -> Tuple[float, int, int, int, int]:
    """Uniform scale to fit ``iw×ih`` inside ``CANVAS_W×CANVAS_H``, centered. Returns
    ``scale, off_x, off_y, nw, nh`` where the letterboxed content occupies ``nw×nh`` at ``(off_x, off_y)``.
    """
    if iw < 1 or ih < 1:
        raise ValueError(f"letterbox_layout: invalid image size {iw}x{ih}")
    scale = min(CANVAS_W / iw, CANVAS_H / ih)
    nw = max(1, min(CANVAS_W, int(round(iw * scale))))
    nh = max(1, min(CANVAS_H, int(round(ih * scale))))
    off_x = (CANVAS_W - nw) // 2
    off_y = (CANVAS_H - nh) // 2
    return scale, off_x, off_y, nw, nh


def scale_box_to_canvas_letterbox(
    x1: int, y1: int, x2: int, y2: int, scale: float, off_x: int, off_y: int
) -> Tuple[int, int, int, int]:
    """Map bbox from source image coords to canvas coords using the same scale/offset as letterbox."""
    X1 = off_x + int(round(x1 * scale))
    Y1 = off_y + int(round(y1 * scale))
    X2 = off_x + int(round(x2 * scale))
    Y2 = off_y + int(round(y2 * scale))
    X1 = max(0, min(X1, CANVAS_W - 1))
    Y1 = max(0, min(Y1, CANVAS_H - 1))
    X2 = max(X1 + 1, min(X2, CANVAS_W))
    Y2 = max(Y1 + 1, min(Y2, CANVAS_H))
    return X1, Y1, X2, Y2


def list_multiview_pngs(multiview_dir: Path, stem: str, crop_no: int) -> List[Path]:
    d = multiview_dir / stem / str(crop_no)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".png")


def rng_for_image(rel: str, base_seed: Optional[int]) -> random.Random:
    h = int(hashlib.md5(rel.encode(), usedforsecurity=False).hexdigest()[:8], 16)
    return random.Random((base_seed if base_seed is not None else 0) ^ h)


def prepare_groups_from_flat_rows(
    rows: Sequence[Dict[str, str]],
    image_key: str,
    min_side: float,
    multiview_match_min_side: float,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """One entry per distinct scene image; rows sorted like stage4; adds _crop_no, _mv_crop_no."""
    from collections import defaultdict

    by: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        rel = (r.get(image_key) or "").strip()
        if not rel:
            continue
        by[rel].append(dict(r))

    groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    for rel, gs in by.items():
        cleaned: List[Dict[str, Any]] = []
        for r in gs:
            try:
                x1, y1, x2, y2 = _f(r["x1"]), _f(r["y1"]), _f(r["x2"]), _f(r["y2"])
            except (KeyError, ValueError, TypeError):
                continue
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            if min_side > 0 and (w <= min_side or h <= min_side):
                continue
            cleaned.append(dict(r))

        if not cleaned:
            continue

        cleaned.sort(
            key=lambda r: (
                -_f(r.get("confidence", 0)),
                _f(r["x1"]),
                _f(r["y1"]),
                _f(r["x2"]),
                _f(r["y2"]),
            )
        )
        mv_counter = 0
        for i, r in enumerate(cleaned):
            r["_crop_no"] = i
            w = _f(r["x2"]) - _f(r["x1"])
            h = _f(r["y2"]) - _f(r["y1"])
            if w > multiview_match_min_side and h > multiview_match_min_side:
                r["_mv_crop_no"] = mv_counter
                mv_counter += 1
            else:
                r["_mv_crop_no"] = None
        groups.append((rel, cleaned))

    return groups


def compose_bbox_multiview_canvas(
    rel: str,
    full: Image.Image,
    rows: Sequence[Dict[str, Any]],
    multiview_root: Optional[Path],
    multiview_prob: float,
    background: str,
    rng: random.Random,
    min_side: float,
) -> Image.Image:
    """Return RGB canvas CANVAS_W×CANVAS_H with crops pasted (stage4 order)."""
    full = full.convert("RGB")
    iw, ih = full.size
    stem = Path(rel).stem
    scale, off_x, off_y, nw, nh = letterbox_layout(iw, ih)

    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (0, 0, 0))
    if background == "scaled":
        bg = full.resize((nw, nh), Image.Resampling.LANCZOS).convert("RGB")
        canvas.paste(bg, (off_x, off_y))

    ordered = sorted(rows, key=lambda r: int(r["_crop_no"]))
    others = [r for r in ordered if int(r["_crop_no"]) != 0]
    primary = [r for r in ordered if int(r["_crop_no"]) == 0]
    paste_rows = others + primary

    for row in paste_rows:
        x1, y1, x2, y2 = clamp_xyxy(_f(row["x1"]), _f(row["y1"]), _f(row["x2"]), _f(row["y2"]), iw, ih)
        if min_side > 0 and ((x2 - x1) <= min_side or (y2 - y1) <= min_side):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        X1, Y1, X2, Y2 = scale_box_to_canvas_letterbox(x1, y1, x2, y2, scale, off_x, off_y)

        crop: Optional[Image.Image] = None
        mvn = row.get("_mv_crop_no")
        mv_paths: List[Path] = []
        if multiview_root is not None and mvn is not None:
            mv_paths = list_multiview_pngs(multiview_root, stem, int(mvn))
        if mv_paths and rng.random() < multiview_prob:
            p = rng.choice(mv_paths)
            try:
                crop = Image.open(p).convert("RGBA")
            except OSError:
                crop = None
        if crop is None:
            crop = full.crop((x1, y1, x2, y2)).convert("RGBA")

        tw, th = X2 - X1, Y2 - Y1
        if tw < 1 or th < 1:
            continue
        resized = crop.resize((tw, th), Image.Resampling.LANCZOS)
        if resized.mode == "RGBA":
            canvas.paste(resized, (X1, Y1), resized)
        else:
            canvas.paste(resized.convert("RGB"), (X1, Y1))

    return canvas


def prompt_for_group(
    group_rows: Sequence[Dict[str, Any]],
    prompt_column: str,
) -> str:
    """Use CSV prompt if any row has it; else join unique class_name."""
    texts = []
    for r in group_rows:
        t = (r.get(prompt_column) or "").strip()
        if t:
            texts.append(t)
    if texts:
        return texts[0]
    names = sorted({(r.get("class_name") or "").strip() for r in group_rows if (r.get("class_name") or "").strip()})
    if names:
        return "A scene containing: " + ", ".join(names) + "."
    return "A scene image."


# Helper: set rel on compose for multiview paths (cleaner than global)
def build_canvas_for_image_rel(
    rel: str,
    full: Image.Image,
    rows: Sequence[Dict[str, Any]],
    multiview_root: Optional[Path],
    multiview_prob: float,
    background: str,
    seed: Optional[int],
    min_side: float,
) -> Image.Image:
    rng = rng_for_image(rel, seed)
    return compose_bbox_multiview_canvas(
        rel,
        full,
        rows,
        multiview_root,
        multiview_prob,
        background,
        rng,
        min_side,
    )
