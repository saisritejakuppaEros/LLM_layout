"""CSV-driven dataset for canvas / scene training (Flux2)."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

from .canvas_bbox_compose import build_canvas_for_image_rel, prepare_groups_from_flat_rows, prompt_for_group
from .jsonl_datasets import get_random_resolution, load_image_safely, multiple_16

Image.MAX_IMAGE_PIXELS = None


def resize_cover_pil(im: Image.Image, out_w: int, out_h: int) -> Image.Image:
    """Scale so the image covers out_w×out_h, then center-crop to exact size (LANCZOS)."""
    im = im.convert("RGB")
    iw, ih = im.size
    if iw == out_w and ih == out_h:
        return im
    scale = max(out_w / iw, out_h / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - out_w) // 2)
    top = max(0, (nh - out_h) // 2)
    return im.crop((left, top, left + out_w, top + out_h))


def resize_contain_letterbox_pil(im: Image.Image, out_w: int, out_h: int, fill: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Scale uniformly so the image fits inside out_w×out_h, then center-pad to exact size (LANCZOS). No crop."""
    im = im.convert("RGB")
    iw, ih = im.size
    if iw == out_w and ih == out_h:
        return im
    scale = min(out_w / iw, out_h / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (out_w, out_h), fill)
    left = (out_w - nw) // 2
    top = (out_h - nh) // 2
    canvas.paste(im, (left, top))
    return canvas


def resize_unified_pil(im: Image.Image, out_w: int, out_h: int, mode: str) -> Image.Image:
    if mode == "cover":
        return resize_cover_pil(im, out_w, out_h)
    if mode == "contain":
        return resize_contain_letterbox_pil(im, out_w, out_h)
    raise ValueError(f"resize_unified_pil: unknown mode {mode!r} (use 'cover' or 'contain')")


def pil_to_model_tensor(pil: Image.Image) -> torch.Tensor:
    """RGB PIL → (3, H, W) in [-1, 1], matching legacy dataset normalization."""
    t = transforms.ToTensor()(pil)
    return transforms.Normalize([0.5], [0.5])(t)


def _make_subject_transform(cond_size: int, *, random_geom_aug: bool) -> transforms.Compose:
    """Resize (long side → cond_size, /16), optional flip/rotate, square pad, [-1,1] tensor."""
    steps: List[Any] = [
        transforms.Lambda(
            lambda img: img.resize(
                (
                    multiple_16(cond_size * img.size[0] / max(img.size)),
                    multiple_16(cond_size * img.size[1] / max(img.size)),
                ),
                resample=Image.BILINEAR,
            )
        ),
    ]
    if random_geom_aug:
        steps += [
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomRotation(degrees=20),
        ]
    steps += [
        transforms.Lambda(
            lambda img: transforms.Pad(
                padding=(
                    int((cond_size - img.size[0]) / 2),
                    int((cond_size - img.size[1]) / 2),
                    int((cond_size - img.size[0]) / 2),
                    int((cond_size - img.size[1]) / 2),
                ),
                fill=0,
            )(img)
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    return transforms.Compose(steps)


class CanvasSceneDataset(torch.utils.data.Dataset):
    """
    Two conditioning strategies (``args.canvas_conditioning``):

    - ``precomputed``: one sample per CSV row. Target = ``canvas_target_column`` image;
      optional ``canvas_column`` / ``subject_path`` file for subject pixels.

    - ``bbox_multiview``: one sample per distinct scene image (grouped by ``canvas_target_column``).
      Target = full scene; subject = composed canvas (internally 1920×1080 in ``canvas_bbox_compose``).

    **Unified resolution (default):** unless ``args.canvas_random_target_resolution`` is set, both target and
    canvas are resized to the same box: ``(unified_train_width, unified_train_height)`` (defaults **1920×1080**),
    snapped to multiples of 16 (1080→1088 for FLUX patchify). The bbox canvas from ``canvas_bbox_compose`` is
    **1920×1080**; pairing uses ``args.canvas_unified_resize``:

    - ``contain`` (default): scale to **fit inside** the box, **letterbox** with black — no crop; a 1920×1080
      canvas and a 16:9 full frame both pick up matching thin bars in a 1920×1088 tensor.
    - ``cover``: scale to **fill** the box, then **center-crop** — can cut edges when aspect ≠ box.

    Same latent grid for batched training in both modes.

    **Legacy random target:** set ``--canvas_random_target_resolution`` to restore random long-side
    ``noise_size`` for the target and ``cond_size``² letterboxing for the canvas (batch size 1 unless all
    targets share the same size).

    Training noise is applied in train.py to **main** latents only; subject/canvas latents stay clean (see train loop).
    """

    def __init__(
        self,
        csv_path: str,
        args: Any,
        canvas_column: str = "canvas_path",
        target_column: str = "image_path",
        prompt_column: str = "prompt",
    ):
        super().__init__()
        self.args = args
        self.canvas_column = canvas_column
        self.target_column = target_column
        self.prompt_column = prompt_column
        root = getattr(args, "canvas_image_root", None)
        self._image_root = root.strip() if isinstance(root, str) and root.strip() else None
        self.conditioning = getattr(args, "canvas_conditioning", "precomputed")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            flat_rows = [{k.strip(): (v or "").strip() for k, v in row.items()} for row in reader]

        size = args.cond_size
        self._noise_cap = args.noise_size
        self._random_target_res = bool(getattr(args, "canvas_random_target_resolution", False))
        uw = int(getattr(args, "unified_train_width", 1920))
        uh = int(getattr(args, "unified_train_height", 1080))
        self._unified_wh: Optional[Tuple[int, int]] = None
        if not self._random_target_res:
            self._unified_wh = (multiple_16(uw), multiple_16(uh))
        mode = getattr(args, "canvas_unified_resize", "contain")
        if mode not in ("cover", "contain"):
            raise ValueError(f"canvas_unified_resize must be 'cover' or 'contain', got {mode!r}")
        self._unified_resize_mode: str = mode
        # Precomputed file-based canvas: keep legacy random geom aug.
        self.subject_transform = _make_subject_transform(size, random_geom_aug=True)
        # Composed bbox canvas: preserve layout (only scale + letterbox to cond_size for VAE).
        self.subject_transform_layout_preserving = _make_subject_transform(size, random_geom_aug=False)

        if self.conditioning == "bbox_multiview":
            mv = getattr(args, "canvas_multiview_dir", "") or ""
            self._multiview_root = Path(mv).resolve() if mv.strip() else None
            if self._multiview_root is not None and not self._multiview_root.is_dir():
                self._multiview_root = None
            self._multiview_prob = float(getattr(args, "canvas_multiview_prob", 0.5))
            self._canvas_background = getattr(args, "canvas_background", "black")
            self._bbox_min_side = float(getattr(args, "canvas_bbox_min_side", 0.0))
            self._mv_match_min = float(getattr(args, "canvas_multiview_match_min_side", 200.0))
            self.groups: List[Tuple[str, List[Dict[str, Any]]]] = prepare_groups_from_flat_rows(
                flat_rows,
                target_column,
                self._bbox_min_side,
                self._mv_match_min,
            )
            if not self.groups:
                raise ValueError(
                    "canvas_conditioning=bbox_multiview: no valid image groups (need image_path + x1,y1,x2,y2)."
                )
            self.rows = []  # unused
        else:
            self.groups = []
            self.rows = flat_rows
            self._multiview_root = None

    def __len__(self):
        if self.conditioning == "bbox_multiview":
            return len(self.groups)
        return len(self.rows)

    def _target_transform(self, image: Image.Image, noise_size: int):
        tfm = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: img.resize(
                        (
                            multiple_16(noise_size * img.size[0] / max(img.size)),
                            multiple_16(noise_size * img.size[1] / max(img.size)),
                        ),
                        resample=Image.BILINEAR,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        return tfm(image)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.conditioning == "bbox_multiview":
            return self._getitem_bbox_multiview(idx)
        return self._getitem_precomputed(idx)

    def _getitem_bbox_multiview(self, idx: int) -> Dict[str, Any]:
        rel, group_rows = self.groups[idx]
        full = load_image_safely(rel, self.args.cond_size, root_dir=self._image_root)

        canvas_pil = build_canvas_for_image_rel(
            rel,
            full,
            group_rows,
            self._multiview_root,
            self._multiview_prob,
            self._canvas_background,
            getattr(self.args, "seed", None),
            self._bbox_min_side,
        )

        if self._unified_wh is not None:
            tw, th = self._unified_wh
            pixel_values = pil_to_model_tensor(resize_unified_pil(full, tw, th, self._unified_resize_mode))
            subject_pixel_values = pil_to_model_tensor(resize_unified_pil(canvas_pil, tw, th, self._unified_resize_mode))
        else:
            noise_size = get_random_resolution(max_size=self.args.noise_size)
            pixel_values = self._target_transform(full, noise_size)
            subject_pixel_values = self.subject_transform_layout_preserving(canvas_pil)

        prompt = prompt_for_group(group_rows, self.prompt_column)
        if random.random() < 0.1:
            prompt = " "

        return {
            "pixel_values": pixel_values,
            "prompts": prompt,
            "subject_pixel_values": subject_pixel_values,
        }

    def _getitem_precomputed(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        target = load_image_safely(row[self.target_column], self.args.cond_size, root_dir=self._image_root)

        if self._unified_wh is not None:
            tw, th = self._unified_wh
            pixel_values = pil_to_model_tensor(resize_unified_pil(target, tw, th, self._unified_resize_mode))
        else:
            noise_size = get_random_resolution(max_size=self.args.noise_size)
            pixel_values = self._target_transform(target, noise_size)

        prompt = row.get(self.prompt_column, "")
        if random.random() < 0.1:
            prompt = " "

        out: Dict[str, Any] = {"pixel_values": pixel_values, "prompts": prompt}

        canvas_key = self.canvas_column if self.canvas_column in row else "subject_path"
        if canvas_key in row and row[canvas_key]:
            canvas = load_image_safely(row[canvas_key], self.args.cond_size, root_dir=self._image_root)
            if self._unified_wh is not None:
                tw, th = self._unified_wh
                out["subject_pixel_values"] = pil_to_model_tensor(resize_unified_pil(canvas, tw, th, self._unified_resize_mode))
            else:
                out["subject_pixel_values"] = self.subject_transform(canvas)

        for k in ("x1", "y1", "x2", "y2", "confidence"):
            if k in row and row[k] != "":
                try:
                    out[k] = float(row[k])
                except ValueError:
                    out[k] = row[k]

        return out


def make_canvas_train_dataset(args, accelerator=None):
    ds = CanvasSceneDataset(
        args.csv_path,
        args,
        canvas_column=args.canvas_column,
        target_column=getattr(args, "canvas_target_column", "image_path"),
        prompt_column=getattr(args, "canvas_prompt_column", "prompt"),
    )
    return ds


def collate_fn_canvas(examples: List[Dict[str, Any]]):
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples]).float()
    prompts = [ex["prompts"] for ex in examples]
    batch: Dict[str, Any] = {"pixel_values": pixel_values, "prompts": prompts}
    if examples[0].get("subject_pixel_values") is not None:
        batch["subject_pixel_values"] = torch.stack([ex["subject_pixel_values"] for ex in examples]).float()
    return batch
