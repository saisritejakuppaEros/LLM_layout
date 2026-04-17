"""Build augmented canvas from GT + bboxes (GT never modified)."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image, ImageFilter

from vendored.jsonl_utils import multiple_16


def _feather_alpha_mask(
    mask_l: Image.Image, feather_px: float
) -> Image.Image:
    if feather_px <= 0:
        return mask_l
    return mask_l.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))


def build_canvas(
    full_rgb: Image.Image,
    bboxes_xyxy: Sequence[Tuple[float, float, float, float]],
    rng: random.Random,
    *,
    canvas_w: int,
    canvas_h: int,
    alpha_feather_px: float = 4.0,
    scale_jitter: float = 0.15,
    translate_px: int = 5,
) -> Image.Image:
    """Black canvas, same size as unified target; paste augmented crops with feathered alpha."""
    full_rgb = full_rgb.convert("RGB")
    iw, ih = full_rgb.size
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))

    scale, off_x, off_y, nw, nh = _letterbox_layout(iw, ih, canvas_w, canvas_h)

    for x1, y1, x2, y2 in bboxes_xyxy:
        X1, Y1, X2, Y2 = _scale_box_to_canvas(
            int(x1), int(y1), int(x2), int(y2), scale, off_x, off_y, canvas_w, canvas_h
        )
        crop = full_rgb.crop((int(x1), int(y1), int(x2), int(y2))).convert("RGBA")
        tw, th = X2 - X1, Y2 - Y1
        if tw < 2 or th < 2:
            continue

        # Augment crop (no hflip, no shear)
        crop = _augment_crop(
            crop,
            rng,
            scale_jitter=scale_jitter,
            translate_px=translate_px,
        )
        crop = crop.resize((tw, th), Image.Resampling.LANCZOS)

        alpha = crop.split()[-1] if crop.mode == "RGBA" else Image.new("L", crop.size, 255)
        if crop.mode != "RGBA":
            crop = crop.convert("RGBA")
        alpha = _feather_alpha_mask(alpha, alpha_feather_px)
        crop.putalpha(alpha)

        layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        layer.paste(crop, (X1, Y1), crop)
        canvas = Image.alpha_composite(canvas, layer)

    return canvas.convert("RGB")


def _letterbox_layout(
    iw: int, ih: int, canvas_w: int, canvas_h: int
) -> Tuple[float, int, int, int, int]:
    scale = min(canvas_w / iw, canvas_h / ih)
    nw = max(1, min(canvas_w, int(round(iw * scale))))
    nh = max(1, min(canvas_h, int(round(ih * scale))))
    off_x = (canvas_w - nw) // 2
    off_y = (canvas_h - nh) // 2
    return scale, off_x, off_y, nw, nh


def _scale_box_to_canvas(
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
    X1 = off_x + int(round(x1 * scale))
    Y1 = off_y + int(round(y1 * scale))
    X2 = off_x + int(round(x2 * scale))
    Y2 = off_y + int(round(y2 * scale))
    X1 = max(0, min(X1, canvas_w - 1))
    Y1 = max(0, min(Y1, canvas_h - 1))
    X2 = max(X1 + 1, min(X2, canvas_w))
    Y2 = max(Y1 + 1, min(Y2, canvas_h))
    return X1, Y1, X2, Y2


def _augment_crop(
    crop_rgba: Image.Image,
    rng: random.Random,
    *,
    scale_jitter: float,
    translate_px: int,
) -> Image.Image:
    w, h = crop_rgba.size
    t = torch.from_numpy(np.array(crop_rgba)).permute(2, 0, 1).float() / 255.0
    if t.shape[0] == 3:
        t = torch.cat([t, torch.ones(1, h, w)], dim=0)

    rgb = t[:3]
    a = t[3:4]

    # Color jitter (brightness/contrast/saturation/hue)
    brightness = 1.0 + rng.uniform(-0.3, 0.3)
    contrast = 1.0 + rng.uniform(-0.2, 0.2)
    saturation = 1.0 + rng.uniform(-0.2, 0.2)
    hue = rng.uniform(-0.05, 0.05)
    rgb = TVF.adjust_brightness(rgb, brightness)
    rgb = TVF.adjust_contrast(rgb, contrast)
    rgb = TVF.adjust_saturation(rgb, saturation)
    rgb = TVF.adjust_hue(rgb, hue)

    # Scale ±scale_jitter
    s = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    new_w = max(2, int(round(w * s)))
    new_h = max(2, int(round(h * s)))
    rgb = torch.nn.functional.interpolate(
        rgb.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    a = torch.nn.functional.interpolate(
        a.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)

    # Translate ±translate_px
    tx = rng.randint(-translate_px, translate_px)
    ty = rng.randint(-translate_px, translate_px)
    pad_x1 = max(0, tx)
    pad_y1 = max(0, ty)
    pad_x2 = max(0, -tx)
    pad_y2 = max(0, -ty)
    if pad_x1 + pad_x2 > 0 or pad_y1 + pad_y2 > 0:
        rgb = torch.nn.functional.pad(rgb, (pad_x1, pad_x2, pad_y1, pad_y2))
        a = torch.nn.functional.pad(a, (pad_x1, pad_x2, pad_y1, pad_y2))

    _, ch, cw = rgb.shape
    out = torch.zeros(4, ch, cw)
    out[:3] = rgb
    out[3:] = a

    arr = (out.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGBA")


def resize_cover_pil(im: Image.Image, out_w: int, out_h: int) -> Image.Image:
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


def resize_contain_letterbox_pil(
    im: Image.Image, out_w: int, out_h: int, fill: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
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
    raise ValueError(mode)


def pil_to_model_tensor(pil: Image.Image) -> torch.Tensor:
    from torchvision import transforms

    t = transforms.ToTensor()(pil.convert("RGB"))
    return transforms.Normalize([0.5], [0.5])(t)


def unified_wh(multiple: bool, uw: int, uh: int) -> Tuple[int, int]:
    return (multiple_16(uw), multiple_16(uh))
