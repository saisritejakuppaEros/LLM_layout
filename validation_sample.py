"""BBox-mod flow-matching sampling for validation (matches training: canvas -> adapter -> bbox_mod, no cond tokens)."""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
from PIL import Image
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline, compute_empirical_mu, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor

from latent_bbox import aggregate_per_layer, build_main_token_mask_bkt, pixel_xyxy_to_latent_roi, pool_canvas_latent_per_box
from vendored.flux2_train_helpers import encode_flux2_latents
from vendored.prompt_helper import encode_prompts_flux2


def _flux2_image_processor(vae):
    vsf = 2 ** (len(vae.config.block_out_channels) - 1)
    return Flux2ImageProcessor(vae_scale_factor=vsf * 2)


def tensor_m11_chw_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().float().cpu().clamp(-1, 1)
    x = (x * 0.5 + 0.5) * 255.0
    x = x.byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def build_bbox_mod_list(
    adapter,
    pooled: torch.Tensor,
    box_valid: torch.Tensor,
    tok_mask: torch.Tensor,
    layer_scale_vec: torch.Tensor,
    L_total: int,
    weight_dtype: torch.dtype,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    _, layer_pairs = adapter(pooled, box_valid)
    bbox_mod = []
    for li in range(L_total):
        ds, dsc = layer_pairs[li]
        ls_c = float(layer_scale_vec[li].item())
        s_agg, sc_agg = aggregate_per_layer(ds, dsc, tok_mask, ls_c)
        bbox_mod.append((s_agg.to(weight_dtype), sc_agg.to(weight_dtype)))
    return bbox_mod


@torch.no_grad()
def sample_bbox_mod_one(
    *,
    vae,
    transformer,
    adapter,
    noise_scheduler_template,
    text_encoder,
    tokenizer,
    pixel_values: torch.Tensor,
    canvas: torch.Tensor,
    bboxes: torch.Tensor,
    box_valid: torch.Tensor,
    unified_hw: Tuple[int, int],
    prompts: List[str],
    weight_dtype: torch.dtype,
    device: torch.device,
    max_sequence_length: int,
    text_encoder_out_layers: Tuple[int, ...],
    guidance_scale: float,
    layer_scale_vec: torch.Tensor,
    L_total: int,
    num_inference_steps: int,
    generator: Optional[torch.Generator] = None,
) -> Image.Image:
    """Denoise one example (B=1) with bbox modulation; same conditioning path as training."""
    assert pixel_values.shape[0] == 1
    uw, uh = unified_hw

    pe, text_ids = encode_prompts_flux2(
        text_encoder,
        tokenizer,
        prompts,
        device,
        max_sequence_length,
        weight_dtype,
        text_encoder_out_layers,
    )
    pe = pe.to(dtype=weight_dtype)
    text_ids = text_ids.to(dtype=weight_dtype)

    model_input_shape_ref = encode_flux2_latents(vae, pixel_values.to(dtype=torch.float32), weight_dtype)
    canvas_latent = encode_flux2_latents(vae, canvas.to(dtype=torch.float32), weight_dtype)

    B, _C, Hm, Wm = model_input_shape_ref.shape
    rois = []
    for b in range(B):
        row = []
        for k in range(bboxes.shape[1]):
            if not box_valid[b, k]:
                row.append(torch.zeros(4, device=model_input_shape_ref.device, dtype=torch.long))
                continue
            xy = bboxes[b, k].to(device=model_input_shape_ref.device, dtype=torch.float32)
            r = pixel_xyxy_to_latent_roi(xy, uh, uw, Hm, Wm)
            row.append(r.long())
        rois.append(torch.stack(row, dim=0))
    rois_lat = torch.stack(rois, dim=0)

    tok_mask = build_main_token_mask_bkt(Hm, Wm, rois_lat, box_valid, model_input_shape_ref.device)
    pooled = pool_canvas_latent_per_box(canvas_latent, rois_lat, box_valid).float()

    adapter.eval()
    transformer.eval()
    bbox_mod = build_bbox_mod_list(
        adapter, pooled, box_valid, tok_mask, layer_scale_vec, L_total, weight_dtype
    )

    latents_4d = randn_tensor(
        model_input_shape_ref.shape,
        generator=generator,
        device=device,
        dtype=weight_dtype,
    )
    latent_ids_main = Flux2Pipeline._prepare_latent_ids(latents_4d).to(device=device)
    latents = Flux2Pipeline._pack_latents(latents_4d)
    main_seq = latents.shape[1]

    infer_scheduler = copy.deepcopy(noise_scheduler_template)
    infer_scheduler.set_begin_index(0)
    mu = compute_empirical_mu(image_seq_len=main_seq, num_steps=num_inference_steps)
    timesteps, _ = retrieve_timesteps(
        infer_scheduler,
        num_inference_steps,
        device,
        sigmas=None,
        mu=mu,
    )

    guidance = torch.full((1,), guidance_scale, device=device, dtype=weight_dtype)

    for t in timesteps:
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = transformer(
            hidden_states=latents.to(transformer.dtype),
            cond_hidden_states=None,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=pe,
            txt_ids=text_ids,
            img_ids=latent_ids_main,
            bbox_mod=bbox_mod,
            main_seq_len=main_seq,
            return_dict=False,
        )[0]
        if noise_pred.shape[1] != main_seq:
            noise_pred = noise_pred[:, :main_seq, :]
        noise_pred = noise_pred.to(dtype=latents.dtype)
        latents_dtype = latents.dtype
        latents = infer_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    latents_4d_out = Flux2Pipeline._unpack_latents_with_ids(latents, latent_ids_main)
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents_4d_out.device, latents_4d_out.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latents_4d_out.device, latents_4d_out.dtype
    )
    latents_4d_out = latents_4d_out * latents_bn_std + latents_bn_mean
    latents_4d_out = Flux2Pipeline._unpatchify_latents(latents_4d_out)
    image = vae.decode(latents_4d_out.to(dtype=vae.dtype), return_dict=False)[0]
    proc = _flux2_image_processor(vae)
    pil_list = proc.postprocess(image, output_type="pil")
    return pil_list[0]
