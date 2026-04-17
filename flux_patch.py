# Fork: bbox modulation after AdaLN on main image tokens (see vendored/flux2_transformer_cond.py).
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import logzero
import torch
import torch.nn as nn

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_npu_available, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

from vendored.flux2_transformer_cond import (
    Flux2Modulation,
    Flux2SingleTransformerBlockCond,
    Flux2Transformer2DModelCond,
    Flux2TransformerBlockCond,
    _ckpt_double_block,
    _ckpt_single_block,
)

logger = logging.get_logger(__name__)


class Flux2TransformerBlockBBox(Flux2TransformerBlockCond):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cond_hidden_states: Optional[torch.Tensor],
        temb_mod_img: torch.Tensor,
        temb_mod_img_cond: Optional[torch.Tensor],
        temb_mod_txt: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        bbox_shift: Optional[torch.Tensor] = None,
        bbox_scale: Optional[torch.Tensor] = None,
    ):
        joint_attention_kwargs = joint_attention_kwargs or {}
        use_cond = cond_hidden_states is not None

        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(temb_mod_img, 2)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(
            temb_mod_txt, 2
        )

        norm_hidden_states = (1 + scale_msa) * self.norm1(hidden_states) + shift_msa
        if bbox_shift is not None and bbox_scale is not None:
            norm_hidden_states = norm_hidden_states * (1 + bbox_scale) + bbox_shift

        if use_cond:
            (cnd_shift_msa, cnd_scale_msa, cnd_gate_msa), (cnd_shift_mlp, cnd_scale_mlp, cnd_gate_mlp) = (
                Flux2Modulation.split(temb_mod_img_cond, 2)
            )
            norm_cond = (1 + cnd_scale_msa) * self.norm1(cond_hidden_states) + cnd_shift_msa
            norm_img_in = torch.cat([norm_hidden_states, norm_cond], dim=1)
            attn_kwargs = {**joint_attention_kwargs, "use_cond": True, "main_seq_len": hidden_states.shape[1]}
        else:
            cnd_gate_msa = cnd_shift_mlp = cnd_scale_mlp = cnd_gate_mlp = None
            norm_img_in = norm_hidden_states
            attn_kwargs = dict(joint_attention_kwargs)

        norm_encoder_hidden_states = (1 + c_scale_msa) * self.norm1_context(encoder_hidden_states) + c_shift_msa

        attention_outputs = self.attn(
            hidden_states=norm_img_in,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attn_kwargs,
        )

        if use_cond:
            attn_output, context_attn_output, cond_attn_output = attention_outputs
        else:
            attn_output, context_attn_output = attention_outputs
            cond_attn_output = None

        hidden_states = hidden_states + gate_msa * attn_output

        if use_cond:
            cond_hidden_states = cond_hidden_states + cnd_gate_msa * cond_attn_output
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
            hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)

            norm_cond = self.norm2(cond_hidden_states)
            norm_cond = norm_cond * (1 + cnd_scale_mlp) + cnd_shift_mlp
            cond_hidden_states = cond_hidden_states + cnd_gate_mlp * self.ff(norm_cond)
        else:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
            hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)

        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(norm_encoder_hidden_states)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states, cond_hidden_states


class Flux2SingleTransformerBlockBBox(Flux2SingleTransformerBlockCond):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_hidden_states: Optional[torch.Tensor],
        temb_mod: torch.Tensor,
        temb_mod_cond: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        bbox_shift: Optional[torch.Tensor] = None,
        bbox_scale: Optional[torch.Tensor] = None,
        num_txt_tokens: int = 0,
        main_seq_len: int = 0,
    ):
        use_cond = cond_hidden_states is not None
        joint_attention_kwargs = joint_attention_kwargs or {}

        mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]
        norm_tm = (1 + mod_scale) * self.norm(hidden_states) + mod_shift

        if bbox_shift is not None and bbox_scale is not None:
            sl = slice(num_txt_tokens, num_txt_tokens + main_seq_len)
            norm_tm[:, sl, :] = norm_tm[:, sl, :] * (1 + bbox_scale) + bbox_shift

        if use_cond:
            c_shift, c_scale, c_gate = Flux2Modulation.split(temb_mod_cond, 1)[0]
            norm_c = (1 + c_scale) * self.norm(cond_hidden_states) + c_shift
            combined = torch.cat([norm_tm, norm_c], dim=1)
            attn_kwargs = {**joint_attention_kwargs, "use_cond": True, "main_seq_len": hidden_states.shape[1]}
        else:
            combined = norm_tm
            c_gate = None
            attn_kwargs = dict(joint_attention_kwargs)

        attn_output = self.attn(
            hidden_states=combined,
            image_rotary_emb=image_rotary_emb,
            **attn_kwargs,
        )

        if use_cond:
            out_tm, out_c = attn_output
            hidden_states = hidden_states + mod_gate * out_tm
            cond_hidden_states = cond_hidden_states + c_gate * out_c
            return hidden_states, cond_hidden_states

        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states, None


def _ckpt_double_block_bbox(
    block,
    hidden_states,
    encoder_hidden_states,
    cond_hidden_states,
    d_img,
    d_img_cond,
    d_txt,
    rope,
    jkw,
    use_cond: bool,
    bbox_shift,
    bbox_scale,
):
    return block(
        hidden_states,
        encoder_hidden_states,
        cond_hidden_states if use_cond else None,
        d_img,
        d_img_cond if use_cond else None,
        d_txt,
        rope,
        jkw,
        bbox_shift=bbox_shift,
        bbox_scale=bbox_scale,
    )


def _ckpt_single_block_bbox(
    block,
    hidden_states,
    cond_hidden_states,
    s_mod,
    s_mod_c,
    rope,
    jkw,
    use_cond: bool,
    bbox_shift,
    bbox_scale,
    num_txt_tokens,
    main_seq_len,
):
    return block(
        hidden_states,
        cond_hidden_states if use_cond else None,
        s_mod,
        s_mod_c if use_cond else None,
        rope,
        jkw,
        bbox_shift=bbox_shift,
        bbox_scale=bbox_scale,
        num_txt_tokens=num_txt_tokens,
        main_seq_len=main_seq_len,
    )


class Flux2Transformer2DModelBBoxMod(Flux2Transformer2DModelCond):
    """Same weights as Cond model; blocks support optional per-layer bbox shift/scale on main tokens."""

    def __init__(self, *args, **kwargs):
        lz = logzero.logger
        t_all = time.perf_counter()
        lz.info(
            "Flux2Transformer2DModelBBoxMod: starting __init__ — building parent Cond from config "
            "(large alloc + init; no progress bar — often 1–10+ minutes depending on CPU/RAM)."
        )
        t_super = time.perf_counter()
        super().__init__(*args, **kwargs)
        lz.info(
            "Flux2Transformer2DModelBBoxMod: super() finished in %.1fs; swapping %d double + %d single blocks to BBox variants (per-block weight copy).",
            time.perf_counter() - t_super,
            len(self.transformer_blocks),
            len(self.single_transformer_blocks),
        )
        cfg = self.config
        inner_dim = self.inner_dim
        nh = cfg.num_attention_heads
        hd = cfg.attention_head_dim
        mr = cfg.mlp_ratio
        eps = cfg.eps

        t_swap = time.perf_counter()
        new_double = nn.ModuleList()
        for i, block in enumerate(self.transformer_blocks):
            nb = Flux2TransformerBlockBBox(
                dim=inner_dim,
                num_attention_heads=nh,
                attention_head_dim=hd,
                mlp_ratio=mr,
                eps=eps,
                bias=False,
            )
            nb.load_state_dict(block.state_dict())
            new_double.append(nb)
        self.transformer_blocks = new_double
        lz.info(
            "Flux2Transformer2DModelBBoxMod: double blocks swapped in %.1fs.",
            time.perf_counter() - t_swap,
        )

        t_single = time.perf_counter()
        new_single = nn.ModuleList()
        for block in self.single_transformer_blocks:
            nb = Flux2SingleTransformerBlockBBox(
                dim=inner_dim,
                num_attention_heads=nh,
                attention_head_dim=hd,
                mlp_ratio=mr,
                eps=eps,
                bias=False,
            )
            nb.load_state_dict(block.state_dict())
            new_single.append(nb)
        self.single_transformer_blocks = new_single
        lz.info(
            "Flux2Transformer2DModelBBoxMod: single blocks swapped in %.1fs; __init__ total %.1fs.",
            time.perf_counter() - t_single,
            time.perf_counter() - t_all,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        cond_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        bbox_mod: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        main_seq_len: int = 0,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        use_cond = cond_hidden_states is not None
        num_txt_tokens = encoder_hidden_states.shape[1]

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is None:
            guidance = torch.zeros_like(timestep)
        guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)
        zero_ts = torch.zeros_like(timestep)
        cond_temb = self.time_guidance_embed(zero_ts, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_img_cond = self.double_stream_modulation_img(cond_temb) if use_cond else None
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)
        single_stream_mod_cond = self.single_stream_modulation(cond_temb) if use_cond else None

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if use_cond:
            cond_hidden_states = self.x_embedder(cond_hidden_states)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        rope_dev = hidden_states.device
        img_ids = img_ids.to(device=rope_dev)
        txt_ids = txt_ids.to(device=rope_dev)

        if is_torch_npu_available():
            freqs_cos_image, freqs_sin_image = self.pos_embed(img_ids.cpu())
            image_rotary_emb = (freqs_cos_image.npu(), freqs_sin_image.npu())
            freqs_cos_text, freqs_sin_text = self.pos_embed(txt_ids.cpu())
            text_rotary_emb = (freqs_cos_text.npu(), freqs_sin_text.npu())
        else:
            image_rotary_emb = self.pos_embed(img_ids)
            text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        ckpt_kw = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

        li = 0
        for block in self.transformer_blocks:
            bs = bsc = None
            if bbox_mod is not None and li < len(bbox_mod):
                bs, bsc = bbox_mod[li]
            li += 1

            cond_for_ckpt = cond_hidden_states if use_cond else hidden_states[:, :0, :]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states, cond_hidden_states = torch.utils.checkpoint.checkpoint(
                    _ckpt_double_block_bbox,
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    cond_for_ckpt,
                    double_stream_mod_img,
                    double_stream_mod_img_cond,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                    use_cond,
                    bs,
                    bsc,
                    **ckpt_kw,
                )
                if not use_cond:
                    cond_hidden_states = None
            else:
                encoder_hidden_states, hidden_states, cond_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    cond_hidden_states if use_cond else None,
                    double_stream_mod_img,
                    double_stream_mod_img_cond,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                    bbox_shift=bs,
                    bbox_scale=bsc,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.single_transformer_blocks:
            bs = bsc = None
            if bbox_mod is not None and li < len(bbox_mod):
                bs, bsc = bbox_mod[li]
            li += 1

            cond_single_ckpt = cond_hidden_states if use_cond else hidden_states[:, :0, :]
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, cond_hidden_states = torch.utils.checkpoint.checkpoint(
                    _ckpt_single_block_bbox,
                    block,
                    hidden_states,
                    cond_single_ckpt,
                    single_stream_mod,
                    single_stream_mod_cond,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                    use_cond,
                    bs,
                    bsc,
                    num_txt_tokens,
                    main_seq_len,
                    **ckpt_kw,
                )
                if not use_cond:
                    cond_hidden_states = None
            else:
                hidden_states, cond_hidden_states = block(
                    hidden_states,
                    cond_hidden_states if use_cond else None,
                    single_stream_mod,
                    single_stream_mod_cond,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                    bbox_shift=bs,
                    bbox_scale=bsc,
                    num_txt_tokens=num_txt_tokens,
                    main_seq_len=main_seq_len,
                )

        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
