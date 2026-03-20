"""
Diffusion-based layout generation from text prompts.

Given a prompt, generates an image with FLUX.2-dev, captures attention maps,
and extracts per-object segmentation layouts from the attention.
"""

from .generate_layout import generate_layout_from_prompt

__all__ = ["generate_layout_from_prompt"]
