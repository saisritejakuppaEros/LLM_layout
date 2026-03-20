import os

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0
)
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

from .modules import *


def _register_flux2_processor_and_patch():
    """Patch Flux2AttnProcessor and Flux2Transformer2DModel for attention map capture."""
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2AttnProcessor,
        Flux2Transformer2DModel,
    )
    Flux2AttnProcessor.__call__ = flux2_attn_call

    # Inject height and timestep into joint_attention_kwargs in transformer forward
    _original_flux2_forward = Flux2Transformer2DModel.forward

    def _patched_flux2_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        joint_attention_kwargs=None,
        return_dict=True,
    ):
        joint_attention_kwargs = dict(joint_attention_kwargs or {})
        if img_ids is not None:
            ids = img_ids[0] if img_ids.ndim == 3 else img_ids
            joint_attention_kwargs["height"] = int(ids[:, 1].max().item()) + 1
        if timestep is not None:
            joint_attention_kwargs["timestep"] = timestep
        return _original_flux2_forward(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=return_dict,
        )

    Flux2Transformer2DModel.forward = _patched_flux2_forward


def hook_function(name, detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):

            timestep = module.processor.timestep

            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_maps[timestep][name] = module.processor.attn_map.cpu() if detach \
                else module.processor.attn_map
            
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(model, hook_function, target_name, flux2_transformer_blocks_only=False):
    for name, module in model.named_modules():
        if not name.endswith(target_name):
            continue

        # Flux2: only hook transformer_blocks.attn (Flux2Attention), not single_transformer_blocks.attn (Flux2ParallelSelfAttention)
        if flux2_transformer_blocks_only:
            if "single_transformer_blocks" in name:
                continue
            from diffusers.models.transformers.transformer_flux2 import Flux2AttnProcessor
            if isinstance(module.processor, Flux2AttnProcessor):
                module.processor.store_attn_map = True
                module.register_forward_hook(hook_function(name))
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, JointAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, FluxAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, FluxAttnProcessor):
            module.processor.store_attn_map = True
        else:
            continue

        hook = module.register_forward_hook(hook_function(name))
    
    return model


def replace_call_method_for_unet(model):
    if model.__class__.__name__ == 'UNet2DConditionModel':
        from diffusers.models.unets import UNet2DConditionModel
        model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'Transformer2DModel':
            from diffusers.models import Transformer2DModel
            layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)
        
        elif layer.__class__.__name__ == 'BasicTransformerBlock':
            from diffusers.models.attention import BasicTransformerBlock
            layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)
        
        replace_call_method_for_unet(layer)
    
    return model


# TODO: implement
# def replace_call_method_for_sana(model):
#     if model.__class__.__name__ == 'SanaTransformer2DModel':
#         from diffusers.models.transformers import SanaTransformer2DModel
#         model.forward = SanaTransformer2DModelForward.__get__(model, SanaTransformer2DModel)

#     for name, layer in model.named_children():
        
#         if layer.__class__.__name__ == 'SanaTransformerBlock':
#             from diffusers.models.transformers.sana_transformer import SanaTransformerBlock
#             layer.forward = SanaTransformerBlockForward.__get__(layer, SanaTransformerBlock)
        
#         replace_call_method_for_sana(layer)
    
#     return model


def replace_call_method_for_sd3(model):
    if model.__class__.__name__ == 'SD3Transformer2DModel':
        from diffusers.models.transformers import SD3Transformer2DModel
        model.forward = SD3Transformer2DModelForward.__get__(model, SD3Transformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'JointTransformerBlock':
            from diffusers.models.attention import JointTransformerBlock
            layer.forward = JointTransformerBlockForward.__get__(layer, JointTransformerBlock)
        
        replace_call_method_for_sd3(layer)
    
    return model


def replace_call_method_for_flux(model):
    if model.__class__.__name__ == 'FluxTransformer2DModel':
        from diffusers.models.transformers import FluxTransformer2DModel
        model.forward = FluxTransformer2DModelForward.__get__(model, FluxTransformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'FluxTransformerBlock':
            from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
            layer.forward = FluxTransformerBlockForward.__get__(layer, FluxTransformerBlock)
        
        replace_call_method_for_flux(layer)
    
    return model


def init_pipeline(pipeline):
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call2_0
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call2_0
    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'SD3Transformer2DModel':
            JointAttnProcessor2_0.__call__ = joint_attn_call2_0
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_sd3(pipeline.transformer)
        
        elif pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
            from diffusers import FluxPipeline
            FluxAttnProcessor2_0.__call__ = flux_attn_call2_0
            FluxAttnProcessor.__call__ = flux_attn_call2_0
            FluxPipeline.__call__ = FluxPipeline_call
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)

        elif pipeline.transformer.__class__.__name__ == 'Flux2Transformer2DModel':
            _register_flux2_processor_and_patch()
            pipeline.transformer = register_cross_attention_hook(
                pipeline.transformer, hook_function, 'attn', flux2_transformer_blocks_only=True
            )

        # TODO: implement
        # elif pipeline.transformer.__class__.__name__ == 'SanaTransformer2DModel':
        #     from diffusers import SanaPipeline
        #     SanaPipeline.__call__ == SanaPipeline_call
        #     pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn2')
        #     pipeline.transformer = replace_call_method_for_sana(pipeline.transformer)

    else:
        if pipeline.unet.__class__.__name__ == 'UNet2DConditionModel':
            pipeline.unet = register_cross_attention_hook(pipeline.unet, hook_function, 'attn2')
            pipeline.unet = replace_call_method_for_unet(pipeline.unet)


    return pipeline


def process_token(token, startofword):
    if '</w>' in token:
        token = token.replace('</w>', '')
        if startofword:
            token = '<' + token + '>'
        else:
            token = '-' + token + '>'
            startofword = True
    elif token not in ['<|startoftext|>', '<|endoftext|>']:
        if startofword:
            token = '<' + token + '-'
            startofword = False
        else:
            token = '-' + token + '-'
    return token, startofword


def _sanitize_filename(s):
    """Replace path-separator and other invalid filename chars with underscore."""
    return s.replace("/", "_").replace("\\", "_").replace(":", "_")


def save_attention_image(attn_map, tokens, batch_dir, to_pil):
    startofword = True
    for i, (token, a) in enumerate(zip(tokens, attn_map[:len(tokens)])):
        token, startofword = process_token(token, startofword)
        safe_token = _sanitize_filename(token)
        to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{safe_token}.png'))


def save_attention_maps(attn_maps, tokenizer, prompts, base_dir='attn_maps', unconditional=True):
    to_pil = ToPILImage()
    
    token_ids = tokenizer(prompts)['input_ids']
    if isinstance(token_ids, torch.Tensor):
        token_ids = [token_ids[i] for i in range(len(token_ids))]
    elif not (token_ids and isinstance(token_ids[0], list)):
        token_ids = [token_ids]
    total_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids]
    
    os.makedirs(base_dir, exist_ok=True)
    
    if not attn_maps:
        raise ValueError(f"No attention maps were captured. attn_maps is empty. Ensure the pipeline was initialized with init_pipeline() and that FluxAttnProcessor is properly patched.")
    first_timestep = next(iter(attn_maps.values()))
    if not first_timestep:
        raise ValueError(f"No attention maps were captured for any timestep. The layers dict is empty.")
    total_attn_map = list(first_timestep.values())[0].sum(1)
    if unconditional:
        total_attn_map = total_attn_map.chunk(2)[1]  # (batch, height, width, attn_dim)
    total_attn_map = total_attn_map.permute(0, 3, 1, 2)
    total_attn_map = torch.zeros_like(total_attn_map)
    total_attn_map_shape = total_attn_map.shape[-2:]
    total_attn_map_number = 0
    
    for timestep, layers in attn_maps.items():
        timestep_dir = os.path.join(base_dir, f'{timestep}')
        os.makedirs(timestep_dir, exist_ok=True)
        
        for layer, attn_map in layers.items():
            layer_dir = os.path.join(timestep_dir, f'{layer}')
            os.makedirs(layer_dir, exist_ok=True)
            
            attn_map = attn_map.sum(1).squeeze(1).permute(0, 3, 1, 2)
            if unconditional:
                attn_map = attn_map.chunk(2)[1]
            
            resized_attn_map = F.interpolate(attn_map, size=total_attn_map_shape, mode='bilinear', align_corners=False)
            total_attn_map += resized_attn_map
            total_attn_map_number += 1
            
            for batch, (tokens, attn) in enumerate(zip(total_tokens, attn_map)):
                batch_dir = os.path.join(layer_dir, f'batch-{batch}')
                os.makedirs(batch_dir, exist_ok=True)
                save_attention_image(attn, tokens, batch_dir, to_pil)
    
    total_attn_map /= total_attn_map_number
    for batch, (attn_map, tokens) in enumerate(zip(total_attn_map, total_tokens)):
        batch_dir = os.path.join(base_dir, f'batch-{batch}')
        os.makedirs(batch_dir, exist_ok=True)
        save_attention_image(attn_map, tokens, batch_dir, to_pil)