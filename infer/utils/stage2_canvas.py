"""
Stage 2 canvas: Stage 1 layouts → conditioning canvas.

Inference only: no training-style augmentations.

- ``scene``: ``build_canvas_for_image_rel(..., augment=False)``.
- ``refs``: letterbox + paste reference JPEGs into bbox slots (rectangular).
- ``refs_masked``: **black** canvas only (no Stage 1 image), then paste **whole** reference
  images (resized to each bbox, opaque). Identical bboxes for
  multiple objects (e.g. bottle+bucket from Stage 1) are split **horizontally** (bottle left,
  bucket right, then person). Missing refs are stacked in the largest bbox. Saves
  ``bbox_layout.png`` for debugging.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFont

from .reference_images import OBJECT_KEYS, ReferenceImageSet

CanvasMode = Literal["scene", "refs", "refs_masked"]


def _bbox_area_ratio(bbox: list[float | int], iw: int, ih: int) -> float:
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / float(max(1, iw * ih))


def _bbox_area(bbox: list[int]) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def _ref_key_for_entry(object_name: str, clause: str) -> str | None:
    blob = f"{object_name} {clause}".lower()
    for key in OBJECT_KEYS:
        if key in blob:
            return key
    return None


def filter_layouts_to_rows(
    layouts: dict[str, Any],
    image_w: int,
    image_h: int,
    *,
    max_area_ratio: float = 0.85,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Legacy filter for ``scene`` / rectangular ``refs`` (drops huge bboxes)."""
    debug: list[dict[str, Any]] = []
    candidates: list[tuple[str, dict[str, Any]]] = []

    for key in sorted(layouts.keys()):
        entry = layouts[key]
        if not isinstance(entry, dict):
            continue
        bbox = entry.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        object_name = str(entry.get("object_name", ""))
        clause = str(entry.get("clause", ""))
        rk = _ref_key_for_entry(object_name, clause)
        if rk is None:
            debug.append({"layout_key": key, "skipped": "no_object_keyword", "object_name": object_name})
            continue
        ar = _bbox_area_ratio(bbox, image_w, image_h)
        if ar > max_area_ratio:
            debug.append({"layout_key": key, "skipped": "bbox_too_large", "area_ratio": ar, "object_name": object_name})
            continue
        candidates.append((key, {**entry, "_ref_key": rk, "_area_ratio": ar}))

    rows: list[dict[str, Any]] = []
    for i, (_k, e) in enumerate(candidates):
        bb = e["bbox"]
        rows.append(
            {
                "x1": int(bb[0]),
                "y1": int(bb[1]),
                "x2": int(bb[2]),
                "y2": int(bb[3]),
                "_crop_no": i,
                "_mv_crop_no": None,
                "_ref_key": e["_ref_key"],
            }
        )

    return rows, debug


def layout_entries_all_refs(
    layouts: dict[str, Any],
    image_w: int,
    image_h: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    All clauses that mention person / bottle / bucket — **no** max-area skip.

    Returns (entries, filter_debug) where each entry has layout_key, bbox ints, ref_key.
    """
    debug: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []

    # Preserve JSON key order for mask filename index (matches Stage 1 extraction).
    for layout_key in list(layouts.keys()):
        entry = layouts[layout_key]
        if not isinstance(entry, dict):
            continue
        bbox = entry.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        object_name = str(entry.get("object_name", ""))
        clause = str(entry.get("clause", ""))
        rk = _ref_key_for_entry(object_name, clause)
        if rk is None:
            debug.append({"layout_key": layout_key, "skipped": "no_object_keyword", "object_name": object_name})
            continue
        bb = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        ar = _bbox_area_ratio(bb, image_w, image_h)
        entries.append(
            {
                "layout_key": layout_key,
                "x1": bb[0],
                "y1": bb[1],
                "x2": bb[2],
                "y2": bb[3],
                "_ref_key": rk,
                "_area_ratio": ar,
            }
        )

    # Stable order: person, bottle, bucket first occurrence wins for duplicate ref_keys.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for e in entries:
        rk = e["_ref_key"]
        if rk in seen:
            debug.append({"layout_key": e["layout_key"], "skipped": "duplicate_ref_key", "ref_key": rk})
            continue
        seen.add(rk)
        deduped.append(e)

    for i, e in enumerate(deduped):
        e["_crop_no"] = i

    return deduped, debug


def split_entries_sharing_identical_bbox(entries: list[dict[str, Any]]) -> None:
    """
    Stage 1 sometimes assigns the same bbox to bottle and bucket. Split that rectangle
    into equal **horizontal** strips (left → right). Order: bottle, bucket, person so
    bottle is left and bucket is right when only those two collide.
    """
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        t = (int(e["x1"]), int(e["y1"]), int(e["x2"]), int(e["y2"]))
        groups[t].append(e)

    priority = {"bottle": 0, "bucket": 1, "person": 2}

    for box, group in groups.items():
        if len(group) < 2:
            continue
        x1, y1, x2, y2 = box
        w = x2 - x1
        if w < len(group) * 2:
            continue
        group.sort(key=lambda e: priority.get(str(e.get("_ref_key", "")), 99))
        n = len(group)
        for i, e in enumerate(group):
            nx1 = x1 + (i * w) // n
            nx2 = x1 + ((i + 1) * w) // n if i < n - 1 else x2
            e["x1"], e["x2"] = nx1, nx2


def load_layouts_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"layouts.json must be a JSON object, got {type(data)}")
    return data


def build_scene_canvas(
    *,
    rel_stem: str,
    full_image: Image.Image,
    rows: list[dict[str, Any]],
    canvas_w: int,
    canvas_h: int,
    seed: int | None,
) -> Image.Image:
    """Call training compose with augment=False (deterministic)."""
    _ensure_train_on_path()
    from src.canvas_bbox_compose import build_canvas_for_image_rel

    compose_rows = [
        {
            "x1": r["x1"],
            "y1": r["y1"],
            "x2": r["x2"],
            "y2": r["y2"],
            "_crop_no": r["_crop_no"],
            "_mv_crop_no": None,
        }
        for r in rows
    ]
    return build_canvas_for_image_rel(
        rel_stem,
        full_image.convert("RGB"),
        compose_rows,
        multiview_root=None,
        multiview_prob=0.0,
        background="black",
        seed=seed,
        min_side=0.0,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        augment=False,
    )


def build_refs_canvas(
    *,
    full_image: Image.Image,
    rows: list[dict[str, Any]],
    refs: ReferenceImageSet,
    canvas_w: int,
    canvas_h: int,
) -> Image.Image:
    """Letterbox + paste reference JPEGs into bbox slots (rectangular)."""
    _ensure_train_on_path()
    from src.canvas_bbox_compose import clamp_xyxy, letterbox_layout, scale_box_to_canvas_letterbox

    full = full_image.convert("RGB")
    iw, ih = full.size
    scale, off_x, off_y, nw, nh = letterbox_layout(iw, ih, canvas_w, canvas_h)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))

    ref_map = refs.as_dict()
    ordered = sorted(rows, key=lambda r: int(r["_crop_no"]))
    others = [r for r in ordered if int(r["_crop_no"]) != 0]
    primary = [r for r in ordered if int(r["_crop_no"]) == 0]
    paste_rows = others + primary

    for row in paste_rows:
        x1, y1, x2, y2 = clamp_xyxy(
            float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]), iw, ih
        )
        if x2 <= x1 or y2 <= y1:
            continue
        X1, Y1, X2, Y2 = scale_box_to_canvas_letterbox(x1, y1, x2, y2, scale, off_x, off_y, canvas_w, canvas_h)
        tw, th = X2 - X1, Y2 - Y1
        if tw < 1 or th < 1:
            continue
        rk = row.get("_ref_key")
        if not rk or rk not in ref_map:
            continue
        p = ref_map[rk]
        if not p.is_file():
            continue
        crop = Image.open(p).convert("RGBA")
        resized = crop.resize((tw, th), Image.Resampling.LANCZOS)
        if resized.mode == "RGBA":
            canvas.paste(resized, (X1, Y1), resized)
        else:
            canvas.paste(resized.convert("RGB"), (X1, Y1))

    return canvas


def _image_to_canvas_xyxy(
    x1: int, y1: int, x2: int, y2: int,
    iw: int, ih: int,
    scale: float, off_x: int, off_y: int,
    canvas_w: int, canvas_h: int,
) -> tuple[int, int, int, int]:
    _ensure_train_on_path()
    from src.canvas_bbox_compose import clamp_xyxy, scale_box_to_canvas_letterbox

    x1, y1, x2, y2 = clamp_xyxy(float(x1), float(y1), float(x2), float(y2), iw, ih)
    return scale_box_to_canvas_letterbox(x1, y1, x2, y2, scale, off_x, off_y, canvas_w, canvas_h)


def render_bbox_layout(
    *,
    scene_pil: Image.Image,
    entries: list[dict[str, Any]],
    canvas_w: int,
    canvas_h: int,
    outline_colors: dict[str, tuple[int, int, int]] | None = None,
) -> Image.Image:
    """BBox-only diagram: black canvas, dim letterbox frame, colored bbox outlines (no augmentations)."""
    full = scene_pil.convert("RGB")
    iw, ih = full.size
    _ensure_train_on_path()
    from src.canvas_bbox_compose import letterbox_layout

    scale, off_x, off_y, nw, nh = letterbox_layout(iw, ih, canvas_w, canvas_h)
    base = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(base)
    draw.rectangle(
        [off_x, off_y, off_x + nw - 1, off_y + nh - 1],
        outline=(48, 48, 48),
        width=2,
    )
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    colors = outline_colors or {
        "person": (80, 200, 255),
        "bottle": (255, 180, 80),
        "bucket": (180, 255, 120),
    }
    for e in entries:
        rk = e.get("_ref_key", "")
        col = colors.get(str(rk), (255, 255, 255))
        X1, Y1, X2, Y2 = _image_to_canvas_xyxy(
            e["x1"], e["y1"], e["x2"], e["y2"], iw, ih, scale, off_x, off_y, canvas_w, canvas_h
        )
        draw.rectangle([X1, Y1, X2 - 1, Y2 - 1], outline=col, width=4)
        if font is not None:
            draw.text((X1 + 4, Y1 + 4), str(rk), fill=col, font=font)

    return base


def _paste_ref_full_rect(canvas_rgba: Image.Image, ref_path: Path, canvas_xyxy: tuple[int, int, int, int]) -> None:
    """Resize the full reference image to the canvas bbox and paste opaque (no mask)."""
    X1, Y1, X2, Y2 = canvas_xyxy
    tw, th = X2 - X1, Y2 - Y1
    if tw < 2 or th < 2:
        return
    ref = Image.open(ref_path).convert("RGBA").resize((tw, th), Image.Resampling.LANCZOS)
    canvas_rgba.alpha_composite(ref, (X1, Y1))


def _largest_entry_canvas_box(
    entries: list[dict[str, Any]],
    iw: int, ih: int,
    scale: float, off_x: int, off_y: int,
    canvas_w: int, canvas_h: int,
) -> tuple[int, int, int, int] | None:
    if not entries:
        return None
    best = max(entries, key=lambda e: _bbox_area([e["x1"], e["y1"], e["x2"], e["y2"]]))
    return _image_to_canvas_xyxy(
        best["x1"], best["y1"], best["x2"], best["y2"],
        iw, ih, scale, off_x, off_y, canvas_w, canvas_h,
    )


def _paste_stacked_refs_vertical(
    canvas_rgba: Image.Image,
    box: tuple[int, int, int, int],
    ref_keys: list[str],
    ref_map: dict[str, Path],
) -> None:
    """Stack refs as horizontal bands inside ``box`` (bottom to top order in list)."""
    X1, Y1, X2, Y2 = box
    tw, th = X2 - X1, Y2 - Y1
    if tw < 2 or th < 2 or not ref_keys:
        return
    n = len(ref_keys)
    band = max(1, th // n)
    for i, rk in enumerate(ref_keys):
        p = ref_map.get(rk)
        if p is None or not p.is_file():
            continue
        y0 = Y1 + i * band
        y1 = min(Y2, y0 + band)
        h = y1 - y0
        if h < 2:
            continue
        ref = Image.open(p).convert("RGBA")
        ref = ref.resize((tw, h), Image.Resampling.LANCZOS)
        canvas_rgba.alpha_composite(ref, (X1, y0))


def build_refs_masked_canvas(
    *,
    scene_pil: Image.Image,
    entries: list[dict[str, Any]],
    refs: ReferenceImageSet,
    canvas_w: int,
    canvas_h: int,
) -> tuple[Image.Image, Image.Image]:
    """
    Bbox layout preview + conditioning canvas.

    1. ``bbox_layout``: black canvas + letterbox frame + colored bbox outlines (uses
       **post-split** boxes from ``entries``).
    2. Cond canvas: **solid black** (no generated scene); only reference images, resized
       to each layout bbox and pasted fully opaque.
    3. Missing person/bottle/bucket entries: stack vertically in the largest bbox (or
       the letterboxed content rectangle on black).
    """
    full = scene_pil.convert("RGB")
    iw, ih = full.size
    _ensure_train_on_path()
    from src.canvas_bbox_compose import letterbox_layout

    scale, off_x, off_y, _, _ = letterbox_layout(iw, ih, canvas_w, canvas_h)
    bbox_layout = render_bbox_layout(
        scene_pil=scene_pil, entries=entries, canvas_w=canvas_w, canvas_h=canvas_h
    )

    canvas_rgba = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))

    ref_map = refs.as_dict()
    placed: set[str] = set()

    order = {k: i for i, k in enumerate(OBJECT_KEYS)}
    sorted_entries = sorted(entries, key=lambda e: order.get(str(e.get("_ref_key", "")), 99))

    for e in sorted_entries:
        rk = e["_ref_key"]
        p = ref_map.get(rk)
        if p is None or not p.is_file():
            continue
        img_xy = (e["x1"], e["y1"], e["x2"], e["y2"])
        cs_xy = _image_to_canvas_xyxy(*img_xy, iw, ih, scale, off_x, off_y, canvas_w, canvas_h)
        _paste_ref_full_rect(canvas_rgba, p, cs_xy)
        placed.add(rk)

    missing = [k for k in OBJECT_KEYS if k not in placed]
    if missing:
        big = _largest_entry_canvas_box(entries, iw, ih, scale, off_x, off_y, canvas_w, canvas_h)
        if big is None:
            big = (off_x, off_y, off_x + max(1, int(round(iw * scale))), off_y + max(1, int(round(ih * scale))))
        _paste_stacked_refs_vertical(canvas_rgba, big, list(missing), ref_map)

    return bbox_layout, canvas_rgba.convert("RGB")


def _ensure_train_on_path() -> None:
    import sys

    from .paths import TRAIN_PACKAGE_DIR

    train_dir = str(TRAIN_PACKAGE_DIR.resolve())
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)


def build_stage2_canvas(
    *,
    mode: CanvasMode,
    scene_pil: Image.Image,
    layouts_path: Path,
    refs: ReferenceImageSet,
    canvas_w: int,
    canvas_h: int,
    run_name: str,
    seed: int | None,
) -> tuple[Image.Image, Image.Image | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns (cond_canvas, bbox_layout_or_none, layout_meta, filter_debug).

    ``bbox_layout`` is set for ``refs_masked``; otherwise None.
    """
    layouts = load_layouts_json(layouts_path)
    iw, ih = scene_pil.size

    if mode == "refs_masked":
        entries, filter_debug = layout_entries_all_refs(layouts, iw, ih)
        if not entries:
            raise ValueError(
                "No person/bottle/bucket entries in layouts.json for refs_masked. "
                f"filter_debug={filter_debug}"
            )
        split_entries_sharing_identical_bbox(entries)
        bbox_layout, canvas = build_refs_masked_canvas(
            scene_pil=scene_pil,
            entries=entries,
            refs=refs,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )
        meta = [
            {
                "layout_key": e["layout_key"],
                "_crop_no": e["_crop_no"],
                "_ref_key": e["_ref_key"],
                "bbox": [e["x1"], e["y1"], e["x2"], e["y2"]],
                "_area_ratio": _bbox_area_ratio([e["x1"], e["y1"], e["x2"], e["y2"]], iw, ih),
            }
            for e in entries
        ]
        return canvas, bbox_layout, meta, filter_debug

    rows, filter_debug = filter_layouts_to_rows(layouts, iw, ih)
    if not rows:
        raise ValueError(
            "No layout rows left after filtering (need person/bottle/bucket in object_name or clause, "
            f"and bbox area ratio ≤ 0.85). Debug: {filter_debug}"
        )

    if mode == "scene":
        canvas = build_scene_canvas(
            rel_stem=run_name,
            full_image=scene_pil,
            rows=rows,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            seed=seed,
        )
    else:
        canvas = build_refs_canvas(
            full_image=scene_pil,
            rows=rows,
            refs=refs,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )

    meta = [
        {"_crop_no": r["_crop_no"], "_ref_key": r.get("_ref_key"), "bbox": [r["x1"], r["y1"], r["x2"], r["y2"]]}
        for r in rows
    ]
    return canvas, None, meta, filter_debug
