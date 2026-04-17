"""Pixel↔latent bbox mapping and main-sequence token masks."""

from __future__ import annotations

from typing import Tuple

import torch


def aggregate_per_layer(
    shift_k: torch.Tensor,
    scale_k: torch.Tensor,
    token_mask_bkt: torch.Tensor,
    layer_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(B,K,D) after gate -> (B,T,D) with mask; apply layer_scale."""
    g = layer_scale
    s = torch.einsum("bkt,bkd->btd", token_mask_bkt.float(), shift_k) * g
    sc = torch.einsum("bkt,bkd->btd", token_mask_bkt.float(), scale_k) * g
    return s, sc


def pixel_xyxy_to_latent_roi(
    xyxy: torch.Tensor,
    Himg: int,
    Wimg: int,
    Hm: int,
    Wm: int,
) -> torch.Tensor:
    """xyxy in pixel unified space -> latent inclusive-exclusive [ly1, lx1, ly2, lx2]."""
    x1, y1, x2, y2 = xyxy.unbind(-1)
    lx1 = (x1 / float(Wimg) * Wm).floor().long().clamp(0, Wm - 1)
    lx2 = (x2 / float(Wimg) * Wm).ceil().long().clamp(1, Wm)
    ly1 = (y1 / float(Himg) * Hm).floor().long().clamp(0, Hm - 1)
    ly2 = (y2 / float(Himg) * Hm).ceil().long().clamp(1, Hm)
    lx2 = torch.maximum(lx2, lx1 + 1)
    ly2 = torch.maximum(ly2, ly1 + 1)
    return torch.stack([ly1, lx1, ly2, lx2], dim=-1)


def build_main_token_mask_bkt(
    Hm: int,
    Wm: int,
    rois_lat: torch.Tensor,
    box_valid: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    rois_lat: (B, K, 4) ly1,lx1,ly2,lx2
    box_valid: (B, K) bool
    Returns (B, K, T) bool, T = Hm*Wm
    """
    B, K, _ = rois_lat.shape
    T = Hm * Wm
    yi = torch.arange(Hm, device=device, dtype=torch.long).view(Hm, 1).expand(Hm, Wm)
    xi = torch.arange(Wm, device=device, dtype=torch.long).view(1, Wm).expand(Hm, Wm)
    flat_y = yi.reshape(-1)
    flat_x = xi.reshape(-1)

    m = torch.zeros(B, K, T, dtype=torch.bool, device=device)
    for b in range(B):
        for k in range(K):
            if not box_valid[b, k]:
                continue
            ly1, lx1, ly2, lx2 = rois_lat[b, k].tolist()
            inside = (
                (flat_x >= lx1)
                & (flat_x < lx2)
                & (flat_y >= ly1)
                & (flat_y < ly2)
            )
            m[b, k] = inside
    return m


def pool_canvas_latent_per_box(
    canvas_latent_bchw: torch.Tensor,
    rois_lat: torch.Tensor,
    box_valid: torch.Tensor,
) -> torch.Tensor:
    """Mean pool each bbox region; invalid boxes -> zeros. (B, K, C)"""
    B, C, Hm, Wm = canvas_latent_bchw.shape
    K = rois_lat.shape[1]
    out = torch.zeros(B, K, C, device=canvas_latent_bchw.device, dtype=canvas_latent_bchw.dtype)
    for b in range(B):
        for k in range(K):
            if not box_valid[b, k]:
                continue
            ly1, lx1, ly2, lx2 = [int(x) for x in rois_lat[b, k].tolist()]
            ly1 = max(0, ly1)
            lx1 = max(0, lx1)
            ly2 = min(Hm, ly2)
            lx2 = min(Wm, lx2)
            if ly2 <= ly1 or lx2 <= lx1:
                continue
            patch = canvas_latent_bchw[b : b + 1, :, ly1:ly2, lx1:lx2]
            out[b, k] = patch.mean(dim=(2, 3)).squeeze(0)
    return out


def piecewise_layer_scales(
    L_total: int,
    first_frac: float,
    mid_frac: float,
    first_scale: float,
    mid_scale: float,
    last_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """(L_total,) tensor of per-layer multipliers."""
    n1 = max(1, int(round(L_total * first_frac)))
    n2 = max(n1 + 1, int(round(L_total * mid_frac)))
    s = []
    for l in range(L_total):
        if l < n1:
            s.append(first_scale)
        elif l < n2:
            s.append(mid_scale)
        else:
            s.append(last_scale)
    return torch.tensor(s, device=device, dtype=dtype)
