"""Validation using real training batches: flow-matching loss to TensorBoard, cond-aware sampling to disk.

By default, canvas conditioning matches training-time augmentation (see ``--validation_canvas_augment``).
Metrics are always appended to ``{output_dir}/validation_metrics.jsonl`` so you still get scalars if the
``tensorboard`` package is missing (Accelerate then has no TB tracker and ``accelerator.log`` is a no-op).
"""

from __future__ import annotations

import copy
import json
import os
import time
from argparse import Namespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline, compute_empirical_mu, retrieve_timesteps
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils.torch_utils import randn_tensor

from src.flux2_train_helpers import encode_flux2_latents, prepare_subject_latent_ids, unpack_main_latents
from src.prompt_helper import encode_prompts_flux2


def _tensor_m11_chw_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().float().cpu().clamp(-1, 1)
    x = (x * 0.5 + 0.5) * 255.0
    x = x.byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def _rgb_uint8_to_height(arr: np.ndarray, out_h: int) -> np.ndarray:
    """Resize H×W×3 uint8 to height ``out_h``, width scaled to preserve aspect ratio (LANCZOS)."""
    if arr.shape[0] == out_h:
        return arr
    pil = Image.fromarray(arr)
    out_w = max(1, int(round(pil.width * out_h / pil.height)))
    return np.array(pil.resize((out_w, out_h), Image.Resampling.LANCZOS))


def _flux2_image_processor(vae) -> Flux2ImageProcessor:
    vsf = 2 ** (len(vae.config.block_out_channels) - 1)
    return Flux2ImageProcessor(vae_scale_factor=vsf * 2)


def _get_sigmas(
    noise_scheduler_copy,
    timesteps: torch.Tensor,
    n_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


@torch.no_grad()
def _compute_flow_matching_loss_batch(
    *,
    vae,
    transformer,
    text_encoder,
    tokenizer,
    batch: Dict[str, Any],
    args: Namespace,
    noise_scheduler_copy,
    weight_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    prompts = batch["prompts"]
    prompt_embeds, text_ids = encode_prompts_flux2(
        text_encoder,
        tokenizer,
        prompts,
        device,
        args.max_sequence_length,
        weight_dtype,
        tuple(args.text_encoder_out_layers),
    )
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=device)
    text_ids = text_ids.to(dtype=weight_dtype, device=device)

    pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float32)
    model_input = encode_flux2_latents(vae, pixel_values, weight_dtype)
    latent_image_ids = Flux2Pipeline._prepare_latent_ids(model_input)

    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    u = compute_density_for_timestep_sampling(
        weighting_scheme=args.weighting_scheme,
        batch_size=bsz,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        mode_scale=args.mode_scale,
    )
    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

    sigmas = _get_sigmas(noise_scheduler_copy, timesteps, n_dim=model_input.ndim, dtype=model_input.dtype, device=device)
    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
    packed_noisy_model_input = Flux2Pipeline._pack_latents(noisy_model_input)

    latent_image_ids_to_concat = [latent_image_ids]
    packed_cond_parts: List[torch.Tensor] = []

    if batch.get("subject_pixel_values") is not None:
        subject_pixel_values = batch["subject_pixel_values"].to(device=device, dtype=torch.float32)
        subject_latents = encode_flux2_latents(vae, subject_pixel_values, weight_dtype)
        latent_image_ids_to_concat.append(prepare_subject_latent_ids(subject_latents))
        packed_cond_parts.append(Flux2Pipeline._pack_latents(subject_latents))

    if batch.get("cond_pixel_values") is not None:
        cond_pixel_values = batch["cond_pixel_values"].to(device=device, dtype=torch.float32)
        cond_latents = encode_flux2_latents(vae, cond_pixel_values, weight_dtype)
        latent_image_ids_to_concat.append(prepare_subject_latent_ids(cond_latents))
        packed_cond_parts.append(Flux2Pipeline._pack_latents(cond_latents))

    latent_image_ids = torch.cat(latent_image_ids_to_concat, dim=1)
    cond_packed = torch.cat(packed_cond_parts, dim=1) if packed_cond_parts else None

    guidance = torch.full((bsz,), args.guidance_scale, device=device, dtype=torch.float32)

    model_pred = transformer(
        hidden_states=packed_noisy_model_input,
        cond_hidden_states=cond_packed,
        timestep=timesteps / 1000,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]

    main_seq = packed_noisy_model_input.shape[1]
    main_latent_ids = latent_image_ids[:, :main_seq, :]
    model_pred = unpack_main_latents(model_pred, main_latent_ids)

    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
    target = noise - model_input
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )
    return loss.mean()


@torch.no_grad()
def _denoise_one(
    *,
    vae,
    transformer,
    scheduler_template,
    pixel_values: torch.Tensor,
    subject_pixel_values: Optional[torch.Tensor],
    prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    guidance_scale: float,
    weight_dtype: torch.dtype,
    device: torch.device,
    num_inference_steps: int,
    generator: Optional[torch.Generator],
) -> Image.Image:
    """Cond-aware denoise matching training (`cond_hidden_states` + joint `img_ids`)."""
    infer_scheduler = copy.deepcopy(scheduler_template)
    b1 = pixel_values.shape[0]
    assert b1 == 1
    model_input_shape_ref = encode_flux2_latents(vae, pixel_values.to(dtype=torch.float32), weight_dtype)
    latents_4d = randn_tensor(model_input_shape_ref.shape, generator=generator, device=device, dtype=weight_dtype)
    latent_ids_main = Flux2Pipeline._prepare_latent_ids(latents_4d).to(device=device)
    latents = Flux2Pipeline._pack_latents(latents_4d)

    cond_packed: Optional[torch.Tensor] = None
    latent_parts = [latent_ids_main]
    if subject_pixel_values is not None:
        sub = encode_flux2_latents(vae, subject_pixel_values.to(dtype=torch.float32), weight_dtype)
        latent_parts.append(prepare_subject_latent_ids(sub).to(device=device))
        cond_packed = Flux2Pipeline._pack_latents(sub)
    full_img_ids = torch.cat(latent_parts, dim=1)

    main_seq = latents.size(1)
    image_seq_len = main_seq
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    infer_scheduler.set_begin_index(0)
    timesteps, _ = retrieve_timesteps(
        infer_scheduler,
        num_inference_steps,
        device,
        sigmas=None,
        mu=mu,
    )

    guidance = torch.full((1,), guidance_scale, device=device, dtype=torch.float32)

    for t in timesteps:
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = transformer(
            hidden_states=latents.to(transformer.dtype),
            cond_hidden_states=cond_packed.to(transformer.dtype) if cond_packed is not None else None,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=full_img_ids,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, :main_seq, :]
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


def run_dataloader_validation(
    *,
    accelerator: Accelerator,
    args: Namespace,
    train_dataset: torch.utils.data.Dataset,
    collate_fn,
    vae,
    transformer,
    text_encoder,
    tokenizer,
    noise_scheduler_copy,
    weight_dtype: torch.dtype,
    global_step: int,
) -> None:
    n = min(int(args.validation_num_samples), len(train_dataset))
    if n < 1:
        return

    g = torch.Generator()
    if args.seed is not None:
        g.manual_seed(int(args.seed) + int(global_step))

    # Match training-time canvas augments in saved PNGs unless --no-validation_canvas_augment.
    want_clean = hasattr(train_dataset, "_canvas_augment_enabled") and not getattr(
        args, "validation_canvas_augment", True
    )
    if want_clean:
        _saved_canvas_aug = train_dataset._canvas_augment_enabled
        train_dataset._canvas_augment_enabled = False
    else:
        _saved_canvas_aug = None
    try:
        val_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=n,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            generator=g,
        )
        batch = next(iter(val_loader))
    finally:
        if _saved_canvas_aug is not None:
            train_dataset._canvas_augment_enabled = _saved_canvas_aug
    device = accelerator.device

    was_training = transformer.training
    subdir = getattr(args, "validation_samples_subdir", None) or "validation_samples"
    out_root = os.path.join(args.output_dir, subdir, f"step_{global_step:07d}")
    os.makedirs(out_root, exist_ok=True)

    gen_igen = torch.Generator(device=device)
    if args.seed is not None:
        gen_igen.manual_seed(int(args.seed) + int(global_step) + 17)

    t0 = time.perf_counter()
    transformer.eval()
    pv = batch["pixel_values"]
    subj = batch.get("subject_pixel_values")
    n_samp = pv.shape[0]
    n_steps = int(args.validation_inference_steps)
    print(
        f"[validation] batch: {n_samp} images, flow-matching loss forward …",
        flush=True,
    )
    try:
        loss_tensor = _compute_flow_matching_loss_batch(
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            batch=batch,
            args=args,
            noise_scheduler_copy=noise_scheduler_copy,
            weight_dtype=weight_dtype,
            device=device,
        )
        val_loss = float(loss_tensor.detach().item())
        print(f"[validation] loss={val_loss:.6f}; denoising {n_samp} sample(s) × {n_steps} steps each (slow at high res) …", flush=True)

        prompts = batch["prompts"]
        pe, tid = encode_prompts_flux2(
            text_encoder,
            tokenizer,
            prompts,
            device,
            args.max_sequence_length,
            weight_dtype,
            tuple(args.text_encoder_out_layers),
        )
        pe = pe.to(dtype=weight_dtype, device=device)
        tid = tid.to(dtype=weight_dtype, device=device)

        for i in range(pv.shape[0]):
            print(f"[validation] sample {i + 1}/{n_samp}: saving target/canvas, running DiT denoise …", flush=True)
            pv_i = pv[i : i + 1]
            sub_i = subj[i : i + 1] if subj is not None else None
            _tensor_m11_chw_to_pil(pv[i]).save(os.path.join(out_root, f"sample_{i:02d}_target.png"))
            if subj is not None:
                _tensor_m11_chw_to_pil(subj[i]).save(os.path.join(out_root, f"sample_{i:02d}_canvas.png"))

            gen_pil = _denoise_one(
                vae=vae,
                transformer=transformer,
                scheduler_template=noise_scheduler_copy,
                pixel_values=pv_i,
                subject_pixel_values=sub_i,
                prompt_embeds=pe[i : i + 1],
                text_ids=tid[i : i + 1],
                guidance_scale=float(args.guidance_scale),
                weight_dtype=weight_dtype,
                device=device,
                num_inference_steps=int(args.validation_inference_steps),
                generator=gen_igen,
            )
            gen_pil.save(os.path.join(out_root, f"sample_{i:02d}_gen.png"))

            with open(os.path.join(out_root, f"sample_{i:02d}_prompt.txt"), "w", encoding="utf-8") as f:
                p = prompts[i] if isinstance(prompts, list) else prompts
                f.write(str(p))
    finally:
        if was_training:
            transformer.train()

    dt_ms = (time.perf_counter() - t0) * 1000.0
    accelerator.log(
        {
            "validation/loss": val_loss,
            "validation/elapsed_ms": dt_ms,
        },
        step=global_step,
    )

    metrics_jsonl = os.path.join(args.output_dir, "validation_metrics.jsonl")
    with open(metrics_jsonl, "a", encoding="utf-8") as mf:
        mf.write(
            json.dumps(
                {
                    "global_step": global_step,
                    "validation/loss": val_loss,
                    "validation/elapsed_ms": dt_ms,
                    "samples_dir": out_root,
                }
            )
            + "\n"
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            # Grid: target | canvas (if any) | gen for each sample (stacked vertically in one strip per row)
            rows: List[np.ndarray] = []
            for i in range(pv.shape[0]):
                tgt = np.asarray(_tensor_m11_chw_to_pil(pv[i]))
                row_list = [tgt]
                if subj is not None:
                    row_list.append(np.asarray(_tensor_m11_chw_to_pil(subj[i])))
                gen_path = os.path.join(out_root, f"sample_{i:02d}_gen.png")
                row_list.append(np.asarray(Image.open(gen_path).convert("RGB")))
                # Do not use min(height): if ``gen`` is a few pixels shorter, target/canvas were squashed and
                # looked cropped / wrong aspect in TensorBoard. Align to the tallest panel instead.
                h_ref = max(x.shape[0] for x in row_list)
                scaled = [_rgb_uint8_to_height(x, h_ref) for x in row_list]
                w = sum(x.shape[1] for x in scaled)
                strip = np.zeros((h_ref, w, 3), dtype=np.uint8)
                x0 = 0
                for im in scaled:
                    strip[:, x0 : x0 + im.shape[1], :] = im
                    x0 += im.shape[1]
                rows.append(strip)
            if rows:
                max_w = max(r.shape[1] for r in rows)
                padded = []
                for r in rows:
                    if r.shape[1] < max_w:
                        pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                        r = np.concatenate([r, pad], axis=1)
                    padded.append(r)
                grid = np.concatenate(padded, axis=0)
                grid_f = grid.astype(np.float32) / 255.0
                tracker.writer.add_image("validation/dataloader_samples", grid_f.transpose(2, 0, 1), global_step)
                tracker.writer.flush()
