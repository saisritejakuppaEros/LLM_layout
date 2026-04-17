"""Canvas + bbox dataset for bbox-mod training."""

from __future__ import annotations

import csv
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torchvision import transforms

from canvas_aug import build_canvas, pil_to_model_tensor, resize_unified_pil
from vendored.canvas_bbox_compose import (
    letterbox_layout,
    prepare_groups_from_flat_rows,
    prompt_for_group,
    scale_box_to_canvas_letterbox,
)
from vendored.jsonl_utils import load_image_safely, multiple_16, resolve_image_path
from vendored.metadata_prompt_lookup import load_metadata_prompt_map, lookup_prompt

Image.MAX_IMAGE_PIXELS = None


def _f(x: Any) -> float:
    return float(x)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _boxes_non_overlapping(boxes: List[Tuple[float, float, float, float]]) -> bool:
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            if _iou(boxes[i], boxes[j]) > 1e-6:
                return False
    return True


def map_source_boxes_to_unified(
    rows: List[Dict[str, Any]],
    iw: int,
    ih: int,
    uw: int,
    uh: int,
    resize_mode: str,
) -> List[Tuple[float, float, float, float]]:
    """Map CSV boxes from source image coords to unified tensor pixel coords."""
    scale, off_x, off_y, _, _ = letterbox_layout(iw, ih, uw, uh)
    out: List[Tuple[float, float, float, float]] = []
    for r in rows:
        try:
            x1, y1, x2, y2 = _f(r["x1"]), _f(r["y1"]), _f(r["x2"]), _f(r["y2"])
        except (KeyError, ValueError, TypeError):
            continue
        X1, Y1, X2, Y2 = scale_box_to_canvas_letterbox(
            int(x1), int(y1), int(x2), int(y2), scale, off_x, off_y, uw, uh
        )
        out.append((float(X1), float(Y1), float(X2), float(Y2)))
    return out


class CanvasBBoxDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.canvas_target_column = getattr(cfg, "canvas_target_column", "image_path")
        self.canvas_prompt_column = getattr(cfg, "canvas_prompt_column", "prompt")
        root = getattr(cfg, "canvas_image_root", None)
        self._image_root = root.strip() if isinstance(root, str) and root.strip() else None

        with open(cfg.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            flat_rows = [{k.strip(): (v or "").strip() for k, v in row.items()} for row in reader]

        mj = getattr(cfg, "canvas_metadata_jsonl", None) or ""
        self._metadata_prompt_map = load_metadata_prompt_map(mj.strip()) if mj.strip() else {}

        uw = int(getattr(cfg, "unified_train_width", 1280))
        uh = int(getattr(cfg, "unified_train_height", 720))
        self._unified_wh = (multiple_16(uw), multiple_16(uh))
        self._resize_mode = getattr(cfg, "canvas_unified_resize", "contain")
        self._bbox_min_side = float(getattr(cfg, "canvas_bbox_min_side", 0.0))
        self._mv_match_min = 200.0

        groups = prepare_groups_from_flat_rows(
            flat_rows,
            self.canvas_target_column,
            self._bbox_min_side,
            self._mv_match_min,
        )
        stage = int(getattr(cfg, "curriculum_stage", 3))
        self._groups = self._filter_curriculum(groups, stage)

        self._max_boxes = int(getattr(cfg, "max_boxes", 32))
        self._alpha_feather = float(getattr(cfg, "alpha_feather_px", 4.0))
        self._scale_jitter = float(getattr(cfg, "canvas_aug_scale_jitter", 0.15))
        self._translate_px = int(getattr(cfg, "canvas_aug_translate_px", 5))

    def _filter_curriculum(
        self,
        groups: List[Tuple[str, List[Dict[str, Any]]]],
        stage: int,
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        if stage >= 3:
            return groups
        out: List[Tuple[str, List[Dict[str, Any]]]] = []
        for rel, gr in groups:
            iw, ih = self._image_size(rel)
            uw, uh = self._unified_wh
            boxes = map_source_boxes_to_unified(gr, iw, ih, uw, uh, self._resize_mode)
            if not boxes:
                continue
            if stage == 1:
                if len(boxes) == 1:
                    out.append((rel, gr))
            elif stage == 2:
                if len(boxes) >= 2 and _boxes_non_overlapping(boxes):
                    out.append((rel, gr))
        return out if out else groups

    def _image_size(self, rel: str) -> Tuple[int, int]:
        p = resolve_image_path(rel, self._image_root)
        im = Image.open(p)
        return im.size

    def __len__(self) -> int:
        return len(self._groups)

    def _resolve_prompt(self, rel: str, rows: List[Dict[str, Any]]) -> str:
        for r in rows:
            t = (r.get(self.canvas_prompt_column) or "").strip()
            if t:
                return t
        meta = lookup_prompt(self._metadata_prompt_map, resolve_image_path(rel, self._image_root))
        if meta:
            return meta
        return prompt_for_group(rows, self.canvas_prompt_column)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rel, group_rows = self._groups[idx]
        full = load_image_safely(rel, 512, root_dir=self._image_root)
        iw, ih = full.size
        uw, uh = self._unified_wh

        rng = random.Random((hash(rel) ^ idx) & 0xFFFFFFFF)

        boxes_unified = map_source_boxes_to_unified(
            group_rows, iw, ih, uw, uh, self._resize_mode
        )
        if not boxes_unified:
            boxes_unified = [(0.0, 0.0, float(min(uw, 32)), float(min(uh, 32)))]

        canvas_pil = build_canvas(
            full,
            boxes_unified,
            rng,
            canvas_w=uw,
            canvas_h=uh,
            alpha_feather_px=self._alpha_feather,
            scale_jitter=self._scale_jitter,
            translate_px=self._translate_px,
        )

        pixel_values = pil_to_model_tensor(resize_unified_pil(full, uw, uh, self._resize_mode))
        canvas_tensor = pil_to_model_tensor(canvas_pil)

        K = self._max_boxes
        bb = torch.zeros(K, 4, dtype=torch.float32)
        valid = torch.zeros(K, dtype=torch.bool)
        n = min(len(boxes_unified), K)
        for i in range(n):
            bb[i] = torch.tensor(boxes_unified[i], dtype=torch.float32)
            valid[i] = True

        prompt = self._resolve_prompt(rel, group_rows)
        if random.random() < 0.1:
            prompt = " "

        return {
            "pixel_values": pixel_values,
            "canvas": canvas_tensor,
            "bboxes": bb,
            "box_valid": valid,
            "prompts": prompt,
            "unified_hw": torch.tensor([uw, uh], dtype=torch.long),
        }


def collate_fn_bbox(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples]).float()
    canvas = torch.stack([ex["canvas"] for ex in examples]).float()
    bboxes = torch.stack([ex["bboxes"] for ex in examples])
    box_valid = torch.stack([ex["box_valid"] for ex in examples])
    prompts = [ex["prompts"] for ex in examples]
    unified_hw = torch.stack([ex["unified_hw"] for ex in examples])
    return {
        "pixel_values": pixel_values,
        "canvas": canvas,
        "bboxes": bboxes,
        "box_valid": box_valid,
        "prompts": prompts,
        "unified_hw": unified_hw,
    }
