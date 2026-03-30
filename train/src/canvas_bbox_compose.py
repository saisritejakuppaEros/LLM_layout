"""Build conditioning canvas (default 1920×1080): letterboxed scene + bbox crops / optional multiview.

Bboxes use uniform scale + centered letterboxing (same as ``contain``). Canvas width/height are
configurable so compose matches ``unified_train_*`` (e.g. 1280×720) before ``resize_unified_pil``.

Aligned with ``dataset_preparation/stage4_canva.py`` (paste order, multiview layout, letterbox math)."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageEnhance

try:
    import torchvision.transforms.functional as TVF
except ImportError:  # pragma: no cover
    TVF = None  # type: ignore[misc, assignment]

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


def letterbox_layout(
    iw: int, ih: int, canvas_w: int, canvas_h: int
) -> Tuple[float, int, int, int, int]:
    """Uniform scale to fit ``iw×ih`` inside ``canvas_w×canvas_h``, centered."""
    if iw < 1 or ih < 1:
        raise ValueError(f"letterbox_layout: invalid image size {iw}x{ih}")
    if canvas_w < 1 or canvas_h < 1:
        raise ValueError(f"letterbox_layout: invalid canvas {canvas_w}x{canvas_h}")
    scale = min(canvas_w / iw, canvas_h / ih)
    nw = max(1, min(canvas_w, int(round(iw * scale))))
    nh = max(1, min(canvas_h, int(round(ih * scale))))
    off_x = (canvas_w - nw) // 2
    off_y = (canvas_h - nh) // 2
    return scale, off_x, off_y, nw, nh


def scale_box_to_canvas_letterbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    scale: float,
    off_x: int,
    off_y: int,
    canvas_w: int,
    canvas_h: int,
) -> Tuple[int, int, int, int]:
    """Map bbox from source image coords to canvas coords using the same scale/offset as letterbox."""
    X1 = off_x + int(round(x1 * scale))
    Y1 = off_y + int(round(y1 * scale))
    X2 = off_x + int(round(x2 * scale))
    Y2 = off_y + int(round(y2 * scale))
    X1 = max(0, min(X1, canvas_w - 1))
    Y1 = max(0, min(Y1, canvas_h - 1))
    X2 = max(X1 + 1, min(X2, canvas_w))
    Y2 = max(Y1 + 1, min(Y2, canvas_h))
    return X1, Y1, X2, Y2


def _apply_bbox_margin_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    iw: int,
    ih: int,
    min_side: float,
    rng: random.Random,
    margin_min: int,
    margin_max: int,
    expand_prob: float,
) -> Tuple[int, int, int, int]:
    m = rng.randint(margin_min, margin_max)
    if rng.random() < expand_prob:
        x1n, y1n = x1 - m, y1 - m
        x2n, y2n = x2 + m, y2 + m
    else:
        x1n, y1n = x1 + m, y1 + m
        x2n, y2n = x2 - m, y2 - m
    x1n, y1n, x2n, y2n = clamp_xyxy(x1n, y1n, x2n, y2n, iw, ih)
    if x2n <= x1n or y2n <= y1n:
        return x1, y1, x2, y2
    if min_side > 0 and ((x2n - x1n) <= min_side or (y2n - y1n) <= min_side):
        return x1, y1, x2, y2
    return x1n, y1n, x2n, y2n


def sample_brightness_factor(
    rng: random.Random,
    *,
    bimodal: bool,
    brightness_min: float,
    brightness_max: float,
    dim_min: float,
    dim_max: float,
    lit_min: float,
    lit_max: float,
) -> float:
    """Sample a brightness enhance factor: bimodal favors very dim vs heavily lit; else uniform in [min, max]."""
    if bimodal:
        if rng.random() < 0.5:
            return rng.uniform(dim_min, dim_max)
        return rng.uniform(lit_min, lit_max)
    return rng.uniform(brightness_min, brightness_max)


def _perspective_start_end(
    width: int,
    height: int,
    distortion_scale: float,
    rng: random.Random,
) -> Tuple[list[list[int]], list[list[int]]]:
    """Sample corners like ``torchvision.transforms.RandomPerspective.get_params`` (random.Random)."""
    half_height = height // 2
    half_width = width // 2
    dw = int(distortion_scale * half_width)
    dh = int(distortion_scale * half_height)
    dw = max(0, min(dw, max(0, half_width - 1)))
    dh = max(0, min(dh, max(0, half_height - 1)))

    def ri(lo: int, hi: int) -> int:
        if lo >= hi:
            return lo
        return rng.randint(lo, hi)

    topleft = [ri(0, dw), ri(0, dh)]
    topright = [ri(max(0, width - dw - 1), width - 1), ri(0, dh)]
    botright = [
        ri(max(0, width - dw - 1), width - 1),
        ri(max(0, height - dh - 1), height - 1),
    ]
    botleft = [ri(0, dw), ri(max(0, height - dh - 1), height - 1)]
    startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    endpoints = [topleft, topright, botright, botleft]
    return startpoints, endpoints


def _augment_crop_pil(
    crop: Image.Image,
    rng: random.Random,
    *,
    brightness_prob: float,
    brightness_bimodal: bool,
    brightness_min: float,
    brightness_max: float,
    brightness_dim_min: float,
    brightness_dim_max: float,
    brightness_lit_min: float,
    brightness_lit_max: float,
    perspective_prob: float,
    perspective_distortion_scale: float,
    shear_prob: float,
    shear_degrees: float,
    extreme_geom: bool,
) -> Image.Image:
    """Brightness always; geometry is strong (perspective + two-axis shear) only when ``extreme_geom``; else mild."""
    crop = crop.convert("RGBA")
    if rng.random() < brightness_prob:
        f = sample_brightness_factor(
            rng,
            bimodal=brightness_bimodal,
            brightness_min=brightness_min,
            brightness_max=brightness_max,
            dim_min=brightness_dim_min,
            dim_max=brightness_dim_max,
            lit_min=brightness_lit_min,
            lit_max=brightness_lit_max,
        )
        crop = ImageEnhance.Brightness(crop).enhance(f)
    if TVF is None:
        return crop

    t = TVF.pil_to_tensor(crop).float() / 255.0
    _, h, w = t.shape

    if extreme_geom:
        pp, pd = perspective_prob, perspective_distortion_scale
        sp, sd = shear_prob, shear_degrees
    else:
        pp, pd = 0.0, 0.0
        sp, sd = 0.3, 12.0

    if rng.random() < pp and h >= 8 and w >= 8 and pd > 0.0:
        spp, epp = _perspective_start_end(w, h, pd, rng)
        c = t.shape[0]
        fill_persp = [0.0] * c
        t = TVF.perspective(
            t,
            spp,
            epp,
            interpolation=TVF.InterpolationMode.BILINEAR,
            fill=fill_persp,
        )

    if rng.random() < sp:
        if extreme_geom:
            sx = rng.uniform(-sd, sd)
            sy = rng.uniform(-sd, sd)
            shear = [sx, sy]
        else:
            if rng.random() < 0.5:
                shear = [rng.uniform(-sd, sd), 0.0]
            else:
                shear = [0.0, rng.uniform(-sd, sd)]
        t = TVF.affine(
            t,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=shear,
            interpolation=TVF.InterpolationMode.BILINEAR,
        )

    crop = TVF.to_pil_image(t.clamp(0.0, 1.0))
    return crop


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
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
    *,
    extreme_geom_max_crops: int = 3,
    augment: bool = False,
    bbox_margin_prob: float = 1.0,
    bbox_margin_min: int = 10,
    bbox_margin_max: int = 30,
    bbox_expand_prob: float = 0.5,
    brightness_prob: float = 0.5,
    brightness_bimodal: bool = True,
    brightness_min: float = 0.45,
    brightness_max: float = 1.85,
    brightness_dim_min: float = 0.32,
    brightness_dim_max: float = 0.58,
    brightness_lit_min: float = 1.42,
    brightness_lit_max: float = 2.08,
    global_brightness_prob: float = 0.0,
    perspective_prob: float = 0.48,
    perspective_distortion_scale: float = 0.62,
    shear_prob: float = 0.55,
    shear_degrees: float = 26.0,
) -> Image.Image:
    """Return RGB canvas ``canvas_w×canvas_h`` with crops pasted (stage4 order)."""
    full = full.convert("RGB")
    if global_brightness_prob > 0.0 and rng.random() < global_brightness_prob:
        gf = sample_brightness_factor(
            rng,
            bimodal=brightness_bimodal,
            brightness_min=brightness_min,
            brightness_max=brightness_max,
            dim_min=brightness_dim_min,
            dim_max=brightness_dim_max,
            lit_min=brightness_lit_min,
            lit_max=brightness_lit_max,
        )
        full = ImageEnhance.Brightness(full).enhance(gf)
    iw, ih = full.size
    stem = Path(rel).stem
    scale, off_x, off_y, nw, nh = letterbox_layout(iw, ih, canvas_w, canvas_h)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    if background == "scaled":
        bg = full.resize((nw, nh), Image.Resampling.LANCZOS).convert("RGB")
        canvas.paste(bg, (off_x, off_y))

    ordered = sorted(rows, key=lambda r: int(r["_crop_no"]))
    others = [r for r in ordered if int(r["_crop_no"]) != 0]
    primary = [r for r in ordered if int(r["_crop_no"]) == 0]
    paste_rows = others + primary

    n_paste = len(paste_rows)
    if augment and extreme_geom_max_crops > 0 and n_paste > 0:
        k = min(extreme_geom_max_crops, n_paste)
        extreme_idx: set[int] = set(rng.sample(range(n_paste), k=k))
    else:
        extreme_idx = set()

    for i, row in enumerate(paste_rows):
        x1, y1, x2, y2 = clamp_xyxy(_f(row["x1"]), _f(row["y1"]), _f(row["x2"]), _f(row["y2"]), iw, ih)
        if min_side > 0 and ((x2 - x1) <= min_side or (y2 - y1) <= min_side):
            continue
        if x2 <= x1 or y2 <= y1:
            continue

        if augment and rng.random() < bbox_margin_prob:
            x1, y1, x2, y2 = _apply_bbox_margin_xyxy(
                x1,
                y1,
                x2,
                y2,
                iw,
                ih,
                min_side,
                rng,
                bbox_margin_min,
                bbox_margin_max,
                bbox_expand_prob,
            )

        X1, Y1, X2, Y2 = scale_box_to_canvas_letterbox(
            x1, y1, x2, y2, scale, off_x, off_y, canvas_w, canvas_h
        )

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

        if augment:
            crop = _augment_crop_pil(
                crop,
                rng,
                brightness_prob=brightness_prob,
                brightness_bimodal=brightness_bimodal,
                brightness_min=brightness_min,
                brightness_max=brightness_max,
                brightness_dim_min=brightness_dim_min,
                brightness_dim_max=brightness_dim_max,
                brightness_lit_min=brightness_lit_min,
                brightness_lit_max=brightness_lit_max,
                perspective_prob=perspective_prob,
                perspective_distortion_scale=perspective_distortion_scale,
                shear_prob=shear_prob,
                shear_degrees=shear_degrees,
                extreme_geom=i in extreme_idx,
            )

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


def build_canvas_for_image_rel(
    rel: str,
    full: Image.Image,
    rows: Sequence[Dict[str, Any]],
    multiview_root: Optional[Path],
    multiview_prob: float,
    background: str,
    seed: Optional[int],
    min_side: float,
    canvas_w: int = CANVAS_W,
    canvas_h: int = CANVAS_H,
    *,
    extreme_geom_max_crops: int = 3,
    augment: bool = False,
    bbox_margin_prob: float = 1.0,
    bbox_margin_min: int = 10,
    bbox_margin_max: int = 30,
    bbox_expand_prob: float = 0.5,
    brightness_prob: float = 0.5,
    brightness_bimodal: bool = True,
    brightness_min: float = 0.45,
    brightness_max: float = 1.85,
    brightness_dim_min: float = 0.32,
    brightness_dim_max: float = 0.58,
    brightness_lit_min: float = 1.42,
    brightness_lit_max: float = 2.08,
    global_brightness_prob: float = 0.0,
    perspective_prob: float = 0.48,
    perspective_distortion_scale: float = 0.62,
    shear_prob: float = 0.55,
    shear_degrees: float = 26.0,
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
        canvas_w,
        canvas_h,
        extreme_geom_max_crops=extreme_geom_max_crops,
        augment=augment,
        bbox_margin_prob=bbox_margin_prob,
        bbox_margin_min=bbox_margin_min,
        bbox_margin_max=bbox_margin_max,
        bbox_expand_prob=bbox_expand_prob,
        brightness_prob=brightness_prob,
        brightness_bimodal=brightness_bimodal,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        brightness_dim_min=brightness_dim_min,
        brightness_dim_max=brightness_dim_max,
        brightness_lit_min=brightness_lit_min,
        brightness_lit_max=brightness_lit_max,
        global_brightness_prob=global_brightness_prob,
        perspective_prob=perspective_prob,
        perspective_distortion_scale=perspective_distortion_scale,
        shear_prob=shear_prob,
        shear_degrees=shear_degrees,
    )
