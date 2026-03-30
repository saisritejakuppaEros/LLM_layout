"""
Load FLUX.2 + canvas LoRA (same wiring as train) and run cond-aware denoise.

Optional layout prior: when ``layout_prior_t_split`` is set, use zero cond latents for
steps with ``t/1000 > split``, then real cond for later steps (img_ids unchanged).
"""

from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel as Flux2Transformer2DModelBase
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline, compute_empirical_mu, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor

LORA_KEY_SUBSTRINGS = ("q_loras", "k_loras", "v_loras", "proj_loras", "qkv_mlp_loras")


def _ensure_train_on_path(train_dir: Path) -> None:
    s = str(train_dir.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _flux2_image_processor(vae) -> Flux2ImageProcessor:
    vsf = 2 ** (len(vae.config.block_out_channels) - 1)
    return Flux2ImageProcessor(vae_scale_factor=vsf * 2)


def _resolve_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def attach_flux2_lora_processors(
    transformer,
    *,
    device: torch.device,
    weight_dtype: torch.dtype,
    cond_width: int,
    cond_height: int,
    ranks: list[int],
    network_alphas: list[int],
    lora_num: int,
) -> None:
    from src.layers_flux2 import MultiDoubleStreamBlockFlux2LoraProcessor, MultiSingleStreamBlockFlux2LoraProcessor
    from src.transformer_flux import FluxTransformer2DModel

    assert isinstance(transformer, FluxTransformer2DModel)
    dim = transformer.inner_dim
    sa0 = transformer.single_transformer_blocks[0].attn
    s_inner, s_mlp_h, s_mlp_mf = sa0.inner_dim, sa0.mlp_hidden_dim, sa0.mlp_mult_factor
    double_blocks_idx = list(range(len(transformer.transformer_blocks)))
    single_blocks_idx = list(range(len(transformer.single_transformer_blocks)))
    lora_w = [1.0 for _ in range(lora_num)]

    lora_attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        match = re.search(r"\.(\d+)\.", name)
        layer_index = int(match.group(1)) if match else -1
        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
            lora_attn_procs[name] = MultiDoubleStreamBlockFlux2LoraProcessor(
                dim=dim,
                ranks=ranks,
                network_alphas=network_alphas,
                lora_weights=lora_w,
                device=device,
                dtype=weight_dtype,
                cond_width=cond_width,
                cond_height=cond_height,
                n_loras=lora_num,
            )
        elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
            lora_attn_procs[name] = MultiSingleStreamBlockFlux2LoraProcessor(
                dim=dim,
                inner_dim=s_inner,
                mlp_hidden_dim=s_mlp_h,
                mlp_mult_factor=s_mlp_mf,
                ranks=ranks,
                network_alphas=network_alphas,
                lora_weights=lora_w,
                device=device,
                dtype=weight_dtype,
                cond_width=cond_width,
                cond_height=cond_height,
                n_loras=lora_num,
            )
        else:
            lora_attn_procs[name] = attn_processor
    transformer.set_attn_processor(lora_attn_procs)


@dataclass
class Stage2ModelBundle:
    device: torch.device
    weight_dtype: torch.dtype
    tokenizer: Any
    text_encoder: Any
    vae: Any
    transformer: Any
    noise_scheduler_template: Any


def load_flux2_stage2_bundle(
    *,
    pretrained_model_name_or_path: str | Path,
    lora_safetensors_path: str | Path,
    train_dir: Path,
    device: str,
    weight_dtype: torch.dtype,
    cond_width: int,
    cond_height: int,
    ranks: list[int],
    network_alphas: list[int],
    lora_num: int,
    debug: bool = False,
) -> tuple[Stage2ModelBundle, dict[str, Any]]:
    _ensure_train_on_path(train_dir)

    from src.transformer_flux import FluxTransformer2DModel

    pretrained = str(pretrained_model_name_or_path)
    dev = torch.device(device)

    tokenizer_path = str(Path(pretrained) / "tokenizer")
    tokenizer = PixtralProcessor.from_pretrained(tokenizer_path, revision=None)
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        pretrained, subfolder="text_encoder", revision=None, variant=None
    )
    noise_scheduler_template = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained, subfolder="scheduler")
    vae = AutoencoderKLFlux2.from_pretrained(pretrained, subfolder="vae", revision=None, variant=None)

    transformer_base = Flux2Transformer2DModelBase.from_pretrained(
        pretrained, subfolder="transformer", revision=None, variant=None
    )
    transformer = FluxTransformer2DModel.from_config(transformer_base.config)
    transformer.load_state_dict(transformer_base.state_dict(), strict=True)
    del transformer_base

    vae.to(dev, dtype=weight_dtype)
    text_encoder.to(dev, dtype=weight_dtype)
    transformer.to(dev, dtype=weight_dtype)

    attach_flux2_lora_processors(
        transformer,
        device=dev,
        weight_dtype=weight_dtype,
        cond_width=cond_width,
        cond_height=cond_height,
        ranks=ranks,
        network_alphas=network_alphas,
        lora_num=lora_num,
    )

    lora_path = Path(lora_safetensors_path)
    sd = load_file(str(lora_path))
    incompat = transformer.load_state_dict(sd, strict=False)

    # ``missing_keys`` = full-model params not listed in ``sd`` (mostly base DiT weights
    # already loaded above). Only ``...lora...`` names here indicate a real load problem.
    missing_lora = [k for k in incompat.missing_keys if any(s in k for s in LORA_KEY_SUBSTRINGS)]
    ckpt_lora_keys = [k for k in sd if any(s in k for s in LORA_KEY_SUBSTRINGS)]

    lora_stats: dict[str, Any] = {
        "lora_path": str(lora_path.resolve()),
        "lora_tensors_in_checkpoint": len(sd),
        "lora_key_tensors_in_checkpoint": len(ckpt_lora_keys),
        "unexpected_keys_count": len(incompat.unexpected_keys),
        "missing_lora_param_keys_count": len(missing_lora),
        "missing_non_lora_keys_count": len(incompat.missing_keys) - len(missing_lora),
        "lora_load_ok": len(incompat.unexpected_keys) == 0 and len(missing_lora) == 0,
        "note": "High missing_non_lora_keys_count is normal: base transformer weights are not in lora.safetensors.",
    }
    if debug:
        lora_stats["missing_keys_sample_base_only"] = [
            k for k in incompat.missing_keys[:20] if not any(s in k for s in LORA_KEY_SUBSTRINGS)
        ]
        lora_stats["unexpected_keys_sample"] = incompat.unexpected_keys[:20]
        if missing_lora:
            lora_stats["missing_lora_keys_sample"] = missing_lora[:20]

    transformer.eval()
    vae.eval()
    text_encoder.eval()

    bundle = Stage2ModelBundle(
        device=dev,
        weight_dtype=weight_dtype,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        noise_scheduler_template=noise_scheduler_template,
    )
    return bundle, lora_stats


def _decode_latents_to_pil(vae, latent_ids_main, latents_packed, weight_dtype: torch.dtype) -> Image.Image:
    latents_4d_out = Flux2Pipeline._unpack_latents_with_ids(latents_packed, latent_ids_main)
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents_4d_out.device, latents_4d_out.dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        latents_4d_out.device, latents_4d_out.dtype
    )
    latents_4d_out = latents_4d_out * latents_bn_std + latents_bn_mean
    latents_4d_out = Flux2Pipeline._unpatchify_latents(latents_4d_out)
    image = vae.decode(latents_4d_out.to(dtype=vae.dtype), return_dict=False)[0]
    proc = _flux2_image_processor(vae)
    return proc.postprocess(image, output_type="pil")[0]


@torch.no_grad()
def denoise_stage2(
    bundle: Stage2ModelBundle,
    *,
    pixel_values: torch.Tensor,
    subject_pixel_values: torch.Tensor | None,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    max_sequence_length: int,
    text_encoder_out_layers: tuple[int, ...],
    generator: torch.Generator | None,
    layout_prior_t_split: float | None,
    debug_decode_every: int | None,
    debug_dir: Path | None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    from src.flux2_train_helpers import encode_flux2_latents, prepare_subject_latent_ids
    from src.prompt_helper import encode_prompts_flux2

    vae = bundle.vae
    transformer = bundle.transformer
    device = bundle.device
    weight_dtype = bundle.weight_dtype

    pe, text_ids = encode_prompts_flux2(
        bundle.text_encoder,
        bundle.tokenizer,
        prompt,
        device,
        max_sequence_length,
        weight_dtype,
        text_encoder_out_layers,
    )
    pe = pe.to(dtype=weight_dtype, device=device)
    text_ids = text_ids.to(dtype=weight_dtype, device=device)

    infer_scheduler = copy.deepcopy(bundle.noise_scheduler_template)
    b1 = pixel_values.shape[0]
    assert b1 == 1

    model_input_shape_ref = encode_flux2_latents(vae, pixel_values.to(dtype=torch.float32), weight_dtype)
    latents_4d = randn_tensor(model_input_shape_ref.shape, generator=generator, device=device, dtype=weight_dtype)
    latent_ids_main = Flux2Pipeline._prepare_latent_ids(latents_4d).to(device=device)
    latents = Flux2Pipeline._pack_latents(latents_4d)

    cond_packed: torch.Tensor | None = None
    latent_parts = [latent_ids_main]
    if subject_pixel_values is not None:
        sub = encode_flux2_latents(vae, subject_pixel_values.to(dtype=torch.float32), weight_dtype)
        latent_parts.append(prepare_subject_latent_ids(sub).to(device=device))
        cond_packed = Flux2Pipeline._pack_latents(sub)
    full_img_ids = torch.cat(latent_parts, dim=1)

    main_seq = latents.size(1)
    mu = compute_empirical_mu(image_seq_len=main_seq, num_steps=num_inference_steps)
    infer_scheduler.set_begin_index(0)
    timesteps, _ = retrieve_timesteps(
        infer_scheduler,
        num_inference_steps,
        device,
        sigmas=None,
        mu=mu,
    )

    guidance = torch.full((1,), guidance_scale, device=device, dtype=torch.float32)
    step_logs: list[dict[str, Any]] = []

    cond_real = cond_packed
    cond_zero = torch.zeros_like(cond_packed) if cond_packed is not None else None

    for step_idx, t in enumerate(timesteps):
        t_val = float(t.detach().float().mean().item()) if hasattr(t, "detach") else float(t)
        t_norm = t_val / 1000.0
        if layout_prior_t_split is None or cond_packed is None:
            phase = "full_cond"
            cond_use = cond_packed
        elif t_norm > layout_prior_t_split:
            phase = "explore"
            cond_use = cond_zero
        else:
            phase = "layout_prior"
            cond_use = cond_real

        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = transformer(
            hidden_states=latents.to(transformer.dtype),
            cond_hidden_states=cond_use.to(transformer.dtype) if cond_use is not None else None,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=pe,
            txt_ids=text_ids,
            img_ids=full_img_ids,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, :main_seq, :]
        latents_dtype = latents.dtype
        latents = infer_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

        step_logs.append(
            {
                "step": step_idx,
                "t": t_val,
                "t_norm": t_norm,
                "phase": phase,
            }
        )

        if (
            debug_decode_every is not None
            and debug_decode_every > 0
            and debug_dir is not None
            and ((step_idx + 1) % debug_decode_every == 0 or step_idx == len(timesteps) - 1)
        ):
            debug_dir.mkdir(parents=True, exist_ok=True)
            pil = _decode_latents_to_pil(vae, latent_ids_main, latents, weight_dtype)
            pil.save(debug_dir / f"step_{step_idx:04d}.png")

    out_pil = _decode_latents_to_pil(vae, latent_ids_main, latents, weight_dtype)
    return out_pil, step_logs


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
