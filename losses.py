"""Flow-matching diffusion loss + optional masked LPIPS / L2 on decoded RGB."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from diffusers.training_utils import compute_loss_weighting_for_sd3


def diffusion_loss_fm(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    sigmas: torch.Tensor,
    weighting_scheme: str,
) -> torch.Tensor:
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )
    return loss.mean()


def decode_latents_to_rgb(vae, latents_4d: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
    """latents_4d (B,C,H,W) after unpatchify + denorm BN -> VAE decode -> (B,3,H',W') [-1,1]."""
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents_4d.device, latents_4d.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latents_4d.device, latents_4d.dtype
    )
    latents_4d = latents_4d * latents_bn_std + latents_bn_mean
    latents_4d = Flux2Pipeline._unpatchify_latents(latents_4d)
    image = vae.decode(latents_4d.to(dtype=vae.dtype), return_dict=False)[0]
    return image


def x0_hat_from_noise_pred(noise: torch.Tensor, model_pred: torch.Tensor) -> torch.Tensor:
    """Training target is noise - x0; one-step x0 estimate = noise - pred."""
    return noise - model_pred


def aux_lpips_l2(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    union_mask_hw: torch.Tensor,
    bbox_w: float,
    bg_w: float,
    lpips_net,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pred_rgb, gt_rgb: (B, 3, H, W) in [-1, 1]
    union_mask_hw: (B, 1, H, W) float 0/1 inside union of boxes
    """
    m = union_mask_hw.expand_as(pred_rgb)
    out = 1.0 - m

    if lpips_net is not None:
        # LPIPS expects normalized [-1,1] typically
        lp = lpips_net(pred_rgb * m, gt_rgb * m).mean()
    else:
        lp = F.mse_loss(pred_rgb * m, gt_rgb * m)

    l2_bg = F.mse_loss(pred_rgb * out, gt_rgb * out)
    return bbox_w * lp, bg_w * l2_bg
