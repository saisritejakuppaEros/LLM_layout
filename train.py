#!/usr/bin/env python3
"""BBox canvas modulation training for FLUX.2 (run from model_training/)."""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import logzero
import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel as Flux2Transformer2DModelBase
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling
from diffusers.utils.torch_utils import is_compiled_module
from PIL import Image, ImageDraw
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapter import BBoxModulationAdapter
from dataset import CanvasBBoxDataset, collate_fn_bbox
from flux_patch import Flux2Transformer2DModelBBoxMod
from latent_bbox import (
    aggregate_per_layer,
    build_main_token_mask_bkt,
    piecewise_layer_scales,
    pixel_xyxy_to_latent_roi,
    pool_canvas_latent_per_box,
)
from losses import decode_latents_to_rgb, diffusion_loss_fm, x0_hat_from_noise_pred
from validation_sample import sample_bbox_mod_one, tensor_m11_chw_to_pil
from vendored.flux2_train_helpers import encode_flux2_latents, unpack_main_latents
from vendored.layers_flux2 import MultiDoubleStreamBlockFlux2LoraProcessor, MultiSingleStreamBlockFlux2LoraProcessor
from vendored.prompt_helper import encode_prompts_flux2

logzero.setup_default_logger(level=logging.INFO)
logger = logzero.logger


def load_config(path: str) -> SimpleNamespace:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return SimpleNamespace(**{k: _nested(v) for k, v in d.items()})


def _nested(v):
    if isinstance(v, dict):
        return SimpleNamespace(**{k: _nested(x) for k, x in v.items()})
    return v


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "config.yaml"))
    p.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    return p.parse_args()


LOSS_CSV_FIELDS = (
    "step",
    "empty_caption",
    "loss",
    "loss_diffusion",
    "lr",
    "mean_abs_delta_shift",
    "mean_abs_delta_scale",
    "mask_token_count",
    "gate_mean",
    "grad_norm_adapter",
    "loss_lpips",
    "loss_bg_l2",
)


def mean_abs_delta_shift_scale(layer_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[float, float]:
    """Mean |delta| over all layers (after gate). Adapter returns gated shift/scale per layer."""
    if not layer_pairs:
        return 0.0, 0.0
    all_ds = torch.cat([ds.detach().flatten() for ds, _ in layer_pairs])
    all_dsc = torch.cat([dsc.detach().flatten() for _, dsc in layer_pairs])
    return all_ds.abs().mean().item(), all_dsc.abs().mean().item()


def resolve_loss_csv_path(args) -> Path:
    p = getattr(args, "loss_csv_path", None)
    if p:
        return Path(p)
    return Path(args.output_dir) / getattr(args, "loss_csv_filename", "training_loss.csv")


def append_training_loss_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(LOSS_CSV_FIELDS), extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def attach_lora(transformer, args, accelerator, weight_dtype):
    import re

    if not getattr(args, "use_lora", True):
        for p in transformer.parameters():
            p.requires_grad = False
        if accelerator.is_main_process:
            logger.info("LoRA disabled (use_lora=false); only bbox adapter params train on the transformer.")
        return ()

    num_d = len(transformer.transformer_blocks)
    num_s = len(transformer.single_transformer_blocks)
    last_d = getattr(args, "lora_last_double_blocks", None)
    last_s = getattr(args, "lora_last_single_blocks", None)
    if last_d is None:
        last_d = num_d
    if last_s is None:
        last_s = num_s
    last_d = max(0, min(int(last_d), num_d))
    last_s = max(0, min(int(last_s), num_s))
    double_lora_idx = set(range(num_d - last_d, num_d)) if last_d > 0 else set()
    single_lora_idx = set(range(num_s - last_s, num_s)) if last_s > 0 else set()

    if accelerator.is_main_process:
        logger.info(
            "LoRA scope: %d/%d double blocks %s | %d/%d single blocks %s (last-N only; rest keep frozen base processors)",
            len(double_lora_idx),
            num_d,
            sorted(double_lora_idx) if len(double_lora_idx) <= 24 else f"{min(double_lora_idx)}…{max(double_lora_idx)}",
            len(single_lora_idx),
            num_s,
            sorted(single_lora_idx) if len(single_lora_idx) <= 24 else f"{min(single_lora_idx)}…{max(single_lora_idx)}",
        )

    dim = transformer.inner_dim
    sa0 = transformer.single_transformer_blocks[0].attn
    s_inner, s_mlp_h, s_mlp_mf = sa0.inner_dim, sa0.mlp_hidden_dim, sa0.mlp_mult_factor
    lora_w = [1.0 for _ in range(args.lora_num)]
    uw, uh = args.unified_train_width, args.unified_train_height
    lora_attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        match = re.search(r"\.(\d+)\.", name)
        layer_index = int(match.group(1)) if match else -1
        if name.startswith("transformer_blocks") and layer_index in double_lora_idx:
            lora_attn_procs[name] = MultiDoubleStreamBlockFlux2LoraProcessor(
                dim=dim,
                ranks=list(args.ranks),
                network_alphas=list(args.network_alphas),
                lora_weights=lora_w,
                device=accelerator.device,
                dtype=weight_dtype,
                cond_width=uw,
                cond_height=uh,
                n_loras=args.lora_num,
            )
        elif name.startswith("single_transformer_blocks") and layer_index in single_lora_idx:
            lora_attn_procs[name] = MultiSingleStreamBlockFlux2LoraProcessor(
                dim=dim,
                inner_dim=s_inner,
                mlp_hidden_dim=s_mlp_h,
                mlp_mult_factor=s_mlp_mf,
                ranks=list(args.ranks),
                network_alphas=list(args.network_alphas),
                lora_weights=lora_w,
                device=accelerator.device,
                dtype=weight_dtype,
                cond_width=uw,
                cond_height=uh,
                n_loras=args.lora_num,
            )
        else:
            lora_attn_procs[name] = attn_processor
    transformer.set_attn_processor(lora_attn_procs)
    lora_substrings = ("q_loras", "k_loras", "v_loras", "proj_loras", "qkv_mlp_loras")
    for n, param in transformer.named_parameters():
        if not any(s in n for s in lora_substrings):
            param.requires_grad = False
    return lora_substrings


def draw_overlay(
    img_t: torch.Tensor,
    bboxes: torch.Tensor,
    valid: torch.Tensor,
) -> Image.Image:
    """img_t (3,H,W) [-1,1]."""
    x = img_t.detach().float().cpu().clamp(-1, 1)
    x = (x * 0.5 + 0.5) * 255.0
    x = x.byte().permute(1, 2, 0).numpy()
    pil = Image.fromarray(x)
    dr = ImageDraw.Draw(pil)
    _, h, w = img_t.shape
    for i in range(bboxes.shape[0]):
        if not valid[i]:
            continue
        x1, y1, x2, y2 = bboxes[i].tolist()
        dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    return pil


def main():
    cli = parse_args()
    cfg_path = cli.config
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if cli.pretrained_model_name_or_path:
        raw["pretrained_model_name_or_path"] = cli.pretrained_model_name_or_path
    args = SimpleNamespace(**{k: _nested(v) for k, v in raw.items()})

    logger.info(
        "Starting bbox-mod training | config=%s | pretrained=%s",
        cfg_path,
        args.pretrained_model_name_or_path,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=args.logging_dir),
    )

    if accelerator.is_main_process:
        logger.info(
            "Accelerator ready | device=%s | mixed_precision=%s | world_size=%d | grad_accum=%d",
            str(accelerator.device),
            args.mixed_precision,
            accelerator.num_processes,
            args.gradient_accumulation_steps,
        )

    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    t0 = time.perf_counter()
    if accelerator.is_main_process:
        logger.info("Loading tokenizer from %s ...", os.path.join(args.pretrained_model_name_or_path, "tokenizer"))
    tokenizer = PixtralProcessor.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, "tokenizer"),
    )
    if accelerator.is_main_process:
        logger.info("Tokenizer loaded in %.1fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    if accelerator.is_main_process:
        logger.info(
            "Loading text_encoder (large; checkpoint shards — this can take minutes) from %s/text_encoder ...",
            args.pretrained_model_name_or_path,
        )
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    if accelerator.is_main_process:
        logger.info("Text encoder loaded in %.1fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    if accelerator.is_main_process:
        logger.info("Loading scheduler ...")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    if accelerator.is_main_process:
        logger.info("Scheduler loaded in %.1fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    if accelerator.is_main_process:
        logger.info("Loading VAE ...")
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    if accelerator.is_main_process:
        logger.info("VAE loaded in %.1fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    if accelerator.is_main_process:
        logger.info("Loading FLUX transformer weights ...")
    transformer_base = Flux2Transformer2DModelBase.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )
    if accelerator.is_main_process:
        logger.info(
            "Transformer checkpoint shards loaded in %.1fs. Next: BBoxMod wrapper — "
            "from_config() rebuilds the full transformer on CPU then swaps blocks (see flux_patch logs); "
            "then we load the pretrained state_dict again. This stretch can take several minutes with no tqdm.",
            time.perf_counter() - t0,
        )
    t_wrap = time.perf_counter()
    transformer = Flux2Transformer2DModelBBoxMod.from_config(transformer_base.config)
    if accelerator.is_main_process:
        logger.info("BBoxMod from_config() done in %.1fs; loading full pretrained state_dict into wrapper ...", time.perf_counter() - t_wrap)
    t_sd = time.perf_counter()
    sd = transformer_base.state_dict()
    transformer.load_state_dict(sd, strict=True)
    del sd
    del transformer_base
    if accelerator.is_main_process:
        logger.info("Pretrained state_dict applied to BBoxMod in %.1fs.", time.perf_counter() - t_sd)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if accelerator.is_main_process:
        logger.info("Moving VAE, transformer, text_encoder to %s (dtype=%s) ...", accelerator.device, weight_dtype)
    t0 = time.perf_counter()
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if accelerator.is_main_process:
        logger.info("Models on device in %.1fs", time.perf_counter() - t0)

    # flux_patch.py supports checkpointing with bbox via _ckpt_*_bbox; without this, 720p activations for all 56 layers OOM ~140GB.
    gc = bool(getattr(args, "gradient_checkpointing", True))
    transformer.gradient_checkpointing = gc
    if accelerator.is_main_process:
        logger.info(
            "Transformer gradient_checkpointing=%s (strongly recommended for 1280×720; trades compute for VRAM).",
            gc,
        )

    if accelerator.is_main_process:
        logger.info("Attaching LoRA processors (unified %dx%d) ...", args.unified_train_width, args.unified_train_height)
    t0 = time.perf_counter()
    lora_substrings = attach_lora(transformer, args, accelerator, weight_dtype)
    if accelerator.is_main_process:
        logger.info("LoRA attached in %.1fs", time.perf_counter() - t0)

    num_double = len(transformer.transformer_blocks)
    num_single = len(transformer.single_transformer_blocks)
    L_total = num_double + num_single
    inner_dim = transformer.inner_dim
    ls = args.layer_strength
    layer_scale_vec = piecewise_layer_scales(
        L_total,
        ls.first_frac,
        ls.mid_frac,
        ls.first_scale,
        ls.mid_scale,
        ls.last_scale,
        accelerator.device,
        torch.float32,
    )

    if accelerator.is_main_process:
        logger.info(
            "Building adapter | L_total=%d inner_dim=%d adapter_dim=%d max_boxes=%d",
            L_total,
            inner_dim,
            args.adapter_dim,
            args.max_boxes,
        )
    adapter = BBoxModulationAdapter(
        in_channels=128,
        adapter_dim=args.adapter_dim,
        inner_dim=inner_dim,
        num_injection_layers=L_total,
        mlp_gate_final_bias=args.mlp_gate_final_bias,
        use_bbox_index_embedding=False,
        max_boxes=args.max_boxes,
    ).to(accelerator.device, dtype=torch.float32)

    for p in adapter.parameters():
        p.requires_grad = True

    trainable_params = [p for p in list(transformer.parameters()) + list(adapter.parameters()) if p.requires_grad]
    if args.mixed_precision == "fp16":
        cast_training_params([transformer, adapter], dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if accelerator.is_main_process:
        logger.info("Building CanvasBBoxDataset (CSV/metadata scan may take a bit) ...")
    t0 = time.perf_counter()
    train_dataset = CanvasBBoxDataset(args)
    if accelerator.is_main_process:
        logger.info(
            "Dataset ready in %.1fs | len=%d | batch_size=%d | num_workers=%d",
            time.perf_counter() - t0,
            len(train_dataset),
            args.train_batch_size,
            args.dataloader_num_workers,
        )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn_bbox,
        num_workers=args.dataloader_num_workers,
    )

    n_batches = math.ceil(len(train_dataset) / max(1, args.train_batch_size * accelerator.num_processes))
    steps_per_epoch = math.ceil(n_batches / max(1, args.gradient_accumulation_steps))
    total_opt_steps = args.max_train_steps
    if total_opt_steps is None:
        total_opt_steps = steps_per_epoch * args.num_train_epochs
    total_opt_steps = max(1, int(total_opt_steps))

    if accelerator.is_main_process:
        logger.info(
            "Schedule | epochs=%d | steps/epoch≈%d | total optimizer steps=%d | aux every %d steps",
            args.num_train_epochs,
            steps_per_epoch,
            total_opt_steps,
            args.aux_loss_every_n_steps,
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=total_opt_steps * accelerator.num_processes,
    )

    if accelerator.is_main_process:
        logger.info(
            "Calling accelerator.prepare() (distributed wrap / dataloader — first worker spawn can pause briefly) ..."
        )
    t0 = time.perf_counter()
    transformer, adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, adapter, optimizer, train_dataloader, lr_scheduler
    )
    if accelerator.is_main_process:
        logger.info("accelerator.prepare() done in %.1fs", time.perf_counter() - t0)

    if accelerator.is_main_process:
        try:
            accelerator.init_trackers("bbox_flux2", config={"config_path": cfg_path})
            logger.info("TensorBoard trackers initialized (logging_dir=%s).", args.logging_dir)
        except Exception as ex:
            accelerator.init_trackers("bbox_flux2")
            logger.warning(
                "TensorBoard init fallback: %s — install tensorboard if you want TB logs.",
                ex,
            )

    global_step = 0
    progress_bar = tqdm(range(total_opt_steps), disable=not accelerator.is_local_main_process)

    lpips_net = None
    lpips_on_cpu = bool(getattr(args, "lpips_on_cpu", True))
    if accelerator.is_main_process:
        logger.info(
            "Loading optional LPIPS (VGG) on %s for aux loss ...",
            "CPU (frees GPU during DiT forward)" if lpips_on_cpu else str(accelerator.device),
        )
    try:
        import lpips as _lp

        t_lp = time.perf_counter()
        _lp_dev = torch.device("cpu") if lpips_on_cpu else accelerator.device
        lpips_net = _lp.LPIPS(net="vgg").to(_lp_dev)
        lpips_net.requires_grad_(False)
        if accelerator.is_main_process:
            logger.info("LPIPS ready in %.1fs on %s", time.perf_counter() - t_lp, _lp_dev)
    except Exception as e:
        if accelerator.is_main_process:
            logger.warning("LPIPS unavailable (%s); aux bbox term uses MSE on masked region.", e)

    if accelerator.is_main_process:
        logger.info("Entering training loop | total_opt_steps=%d", total_opt_steps)
        logger.info(
            "Empty-caption probability (text bypass, LoRA+adapter only): %.2f",
            float(getattr(args, "empty_caption_prob", 0.4)),
        )
        if getattr(args, "loss_csv_enabled", True):
            logger.info("Per-step loss CSV: %s", resolve_loss_csv_path(args))

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(args.num_train_epochs):
        if accelerator.is_main_process:
            logger.info("Epoch %d / %d starting", epoch + 1, args.num_train_epochs)
        transformer.train()
        adapter.train()
        for step, batch in enumerate(train_dataloader):
            if step == 0 and accelerator.is_main_process:
                logger.info(
                    "First batch arrived — encoding prompts & running forward (if this hangs, try dataloader_num_workers=0)."
                )
            with accelerator.accumulate([transformer, adapter]):
                prompts = batch["prompts"]
                if isinstance(prompts, str):
                    prompts_list = [prompts]
                else:
                    prompts_list = list(prompts)
                use_empty_caption = random.random() < float(getattr(args, "empty_caption_prob", 0.4))
                if use_empty_caption:
                    prompts_list = [""] * len(prompts_list)

                pe, text_ids = encode_prompts_flux2(
                    text_encoder,
                    tokenizer,
                    prompts_list,
                    accelerator.device,
                    args.max_sequence_length,
                    weight_dtype,
                    tuple(args.text_encoder_out_layers),
                )
                pe = pe.to(dtype=weight_dtype)
                text_ids = text_ids.to(dtype=weight_dtype)

                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=torch.float32)
                canvas = batch["canvas"].to(device=accelerator.device, dtype=torch.float32)
                bboxes = batch["bboxes"].to(device=accelerator.device)
                box_valid = batch["box_valid"].to(device=accelerator.device)
                uw, uh = int(batch["unified_hw"][0, 0]), int(batch["unified_hw"][0, 1])

                model_input = encode_flux2_latents(vae, pixel_values, weight_dtype)
                with torch.no_grad():
                    canvas_latent = encode_flux2_latents(vae, canvas, weight_dtype)

                B, C, Hm, Wm = model_input.shape
                rois = []
                for b in range(B):
                    row = []
                    for k in range(bboxes.shape[1]):
                        if not box_valid[b, k]:
                            row.append(torch.zeros(4, device=model_input.device, dtype=torch.long))
                            continue
                        xy = bboxes[b, k].to(device=model_input.device, dtype=torch.float32)
                        r = pixel_xyxy_to_latent_roi(xy, uh, uw, Hm, Wm)
                        row.append(r.long())
                    rois.append(torch.stack(row, dim=0))
                rois_lat = torch.stack(rois, dim=0)

                tok_mask = build_main_token_mask_bkt(Hm, Wm, rois_lat, box_valid, model_input.device)
                pooled = pool_canvas_latent_per_box(canvas_latent, rois_lat, box_valid)
                pooled = pooled.float()

                gate, layer_pairs = adapter(pooled, box_valid)
                mean_abs_shift, mean_abs_scale = mean_abs_delta_shift_scale(layer_pairs)
                mask_token_count = int(tok_mask.sum().item())

                bbox_mod: List[Tuple[torch.Tensor, torch.Tensor]] = []
                for li in range(L_total):
                    ds, dsc = layer_pairs[li]
                    ls_c = float(layer_scale_vec[li].item())
                    s_agg, sc_agg = aggregate_per_layer(ds, dsc, tok_mask, ls_c)
                    bbox_mod.append((s_agg.to(weight_dtype), sc_agg.to(weight_dtype)))

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
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                packed_noisy = Flux2Pipeline._pack_latents(noisy_model_input)
                latent_image_ids = Flux2Pipeline._prepare_latent_ids(model_input).to(device=model_input.device)
                main_seq = packed_noisy.shape[1]
                guidance = torch.full((bsz,), args.guidance_scale, device=accelerator.device, dtype=weight_dtype)

                model_pred = transformer(
                    hidden_states=packed_noisy,
                    cond_hidden_states=None,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=pe,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    bbox_mod=bbox_mod,
                    main_seq_len=main_seq,
                    return_dict=False,
                )[0]

                main_latent_ids = latent_image_ids[:, :main_seq, :]
                model_pred = unpack_main_latents(model_pred, main_latent_ids)
                target = noise - model_input
                loss_diff = diffusion_loss_fm(model_pred, target, sigmas, args.weighting_scheme)

                loss = loss_diff
                if global_step % args.aux_loss_every_n_steps == 0:
                    x0_hat = x0_hat_from_noise_pred(noise, model_pred).float()
                    pred_rgb = decode_latents_to_rgb(vae, x0_hat, torch.float32)
                    gt_rgb = decode_latents_to_rgb(vae, model_input.float(), torch.float32)
                    union = torch.zeros(B, 1, uh, uw, device=accelerator.device, dtype=pred_rgb.dtype)
                    for b in range(B):
                        for k in range(bboxes.shape[1]):
                            if not box_valid[b, k]:
                                continue
                            x1, y1, x2, y2 = bboxes[b, k].long()
                            union[b, 0, y1:y2, x1:x2] = 1.0
                    m = torch.nn.functional.interpolate(union, size=pred_rgb.shape[-2:], mode="nearest")
                    if lpips_net is not None:
                        if lpips_on_cpu:
                            lp = lpips_net(
                                (pred_rgb * m).detach().cpu(),
                                (gt_rgb * m).detach().cpu(),
                            ).mean().to(accelerator.device)
                        else:
                            lp = lpips_net(pred_rgb * m, gt_rgb * m).mean()
                    else:
                        lp = torch.nn.functional.mse_loss(pred_rgb * m, gt_rgb * m)
                    out = 1.0 - m
                    l2_bg = torch.nn.functional.mse_loss(pred_rgb * out, gt_rgb * out)
                    loss = loss + args.bbox_loss_weight * lp + args.bg_loss_weight * l2_bg
                else:
                    lp = torch.tensor(0.0, device=accelerator.device)
                    l2_bg = torch.tensor(0.0, device=accelerator.device)

                accelerator.backward(loss)

                grad_norm_adapter = float("nan")
                if accelerator.sync_gradients:
                    ad_for_norm = accelerator.unwrap_model(adapter)
                    ad_params = [p for p in ad_for_norm.parameters() if p.requires_grad]
                    grad_norm_adapter = torch.nn.utils.clip_grad_norm_(ad_params, float("inf")).item()
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "loss_diffusion": loss_diff.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "gate_mean": gate.detach().mean().item(),
                    "empty_caption": int(use_empty_caption),
                    "mean_abs_delta_shift": mean_abs_shift,
                    "mean_abs_delta_scale": mean_abs_scale,
                    "mask_token_count": mask_token_count,
                    "grad_norm_adapter": grad_norm_adapter,
                }
                if global_step % args.aux_loss_every_n_steps == 0:
                    logs["loss_lpips"] = lp.detach().item()
                    logs["loss_bg_l2"] = l2_bg.detach().item()
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**{k: logs[k] for k in ("loss",) if k in logs})

                if accelerator.is_main_process and getattr(args, "loss_csv_enabled", True):
                    csv_row = {
                        "step": global_step,
                        "empty_caption": logs["empty_caption"],
                        "loss": logs["loss"],
                        "loss_diffusion": logs["loss_diffusion"],
                        "lr": logs["lr"],
                        "mean_abs_delta_shift": logs["mean_abs_delta_shift"],
                        "mean_abs_delta_scale": logs["mean_abs_delta_scale"],
                        "mask_token_count": logs["mask_token_count"],
                        "gate_mean": logs["gate_mean"],
                        "grad_norm_adapter": logs["grad_norm_adapter"],
                        "loss_lpips": logs.get("loss_lpips", ""),
                        "loss_bg_l2": logs.get("loss_bg_l2", ""),
                    }
                    append_training_loss_row(resolve_loss_csv_path(args), csv_row)

                if accelerator.is_main_process:
                    log_every = max(1, getattr(args, "log_every_n_steps", 10))
                    if global_step == 1 or global_step % log_every == 0:
                        logger.info(
                            "step=%d empty_cap=%d |loss=%.5f loss_diff=%.5f lr=%.2e |"
                            "|delta|_s=%.3e |delta|_c=%.3e mask_tok=%d gate=%.4f "
                            "grad_adapt=%.3e",
                            global_step,
                            logs["empty_caption"],
                            logs["loss"],
                            logs["loss_diffusion"],
                            logs["lr"],
                            logs["mean_abs_delta_shift"],
                            logs["mean_abs_delta_scale"],
                            logs["mask_token_count"],
                            logs["gate_mean"],
                            logs["grad_norm_adapter"],
                        )
                        if global_step % args.aux_loss_every_n_steps == 0:
                            logger.info(
                                "  aux | lpips/mse_bbox=%.5f bg_l2=%.5f",
                                logs.get("loss_lpips", float("nan")),
                                logs.get("loss_bg_l2", float("nan")),
                            )

                if accelerator.is_main_process and global_step % args.validation_steps == 0:
                    out_dir = Path(args.output_dir) / args.validation_samples_subdir / f"step_{global_step:07d}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    pv = batch["pixel_values"][0].detach().cpu()
                    cv = batch["canvas"][0].detach().cpu()
                    bb = batch["bboxes"][0].cpu()
                    bv = batch["box_valid"][0].cpu()
                    pr = batch["prompts"]
                    prompt_text = pr[0] if isinstance(pr, (list, tuple)) else pr
                    (out_dir / "prompt.txt").write_text(str(prompt_text), encoding="utf-8")
                    tensor_m11_chw_to_pil(pv).save(out_dir / "gt.png")
                    draw_overlay(pv, bb, bv).save(out_dir / "gt_overlay.png")
                    t = (cv * 0.5 + 0.5).clamp(0, 1)
                    Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(out_dir / "canvas.png")

                    if getattr(args, "validation_save_inference", True):
                        infer_steps = int(getattr(args, "validation_inference_steps", 50))
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            uw_m = accelerator.unwrap_model(transformer)
                            ad_m = accelerator.unwrap_model(adapter)
                            tr_was, ad_was = uw_m.training, ad_m.training
                            uw_m.eval()
                            ad_m.eval()
                            with torch.autocast(
                                device_type="cuda",
                                enabled=accelerator.mixed_precision != "no",
                                dtype=weight_dtype,
                            ):
                                sample_pil = sample_bbox_mod_one(
                                    vae=vae,
                                    transformer=uw_m,
                                    adapter=ad_m,
                                    noise_scheduler_template=noise_scheduler_copy,
                                    text_encoder=text_encoder,
                                    tokenizer=tokenizer,
                                    pixel_values=batch["pixel_values"][:1].to(
                                        device=accelerator.device, dtype=torch.float32
                                    ),
                                    canvas=batch["canvas"][:1].to(device=accelerator.device, dtype=torch.float32),
                                    bboxes=batch["bboxes"][:1].to(device=accelerator.device),
                                    box_valid=batch["box_valid"][:1].to(device=accelerator.device),
                                    unified_hw=(
                                        int(batch["unified_hw"][0, 0]),
                                        int(batch["unified_hw"][0, 1]),
                                    ),
                                    prompts=[str(prompt_text)],
                                    weight_dtype=weight_dtype,
                                    device=accelerator.device,
                                    max_sequence_length=args.max_sequence_length,
                                    text_encoder_out_layers=tuple(args.text_encoder_out_layers),
                                    guidance_scale=float(args.guidance_scale),
                                    layer_scale_vec=layer_scale_vec,
                                    L_total=L_total,
                                    num_inference_steps=infer_steps,
                                    generator=None,
                                )
                            sample_pil.save(out_dir / "sample.png")
                            uw_m.train(tr_was)
                            ad_m.train(ad_was)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            logger.info(
                                "Wrote validation + %d-step sample to %s",
                                infer_steps,
                                out_dir,
                            )
                        except Exception as ex:
                            logger.warning(
                                "Validation sampling failed (gt.png, canvas.png, prompt.txt still saved): %s",
                                ex,
                            )
                            try:
                                accelerator.unwrap_model(transformer).train(True)
                                accelerator.unwrap_model(adapter).train(True)
                            except Exception:
                                pass
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    else:
                        logger.info("Wrote validation assets (no inference) to %s", out_dir)

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    uw_m = accelerator.unwrap_model(transformer)
                    ad_m = accelerator.unwrap_model(adapter)
                    sd = uw_m.state_dict()
                    lora_sd = {k: sd[k] for k in sd if any(s in k for s in lora_substrings)}
                    save_file(lora_sd, save_path / "lora.safetensors")
                    torch.save(ad_m.state_dict(), save_path / "adapter.pt")
                    logger.info("Saved checkpoint to %s", save_path)

            if args.max_train_steps and global_step >= args.max_train_steps:
                break
        if args.max_train_steps and global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        logger.info("Training finished | global_step=%d", global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
