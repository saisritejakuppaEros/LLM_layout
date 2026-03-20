# Diffusion Layout Generation

Generate per-object layout segmentation from text prompts using FLUX.2-dev and attention maps.

## Pipeline

1. **Image generation** – FLUX.2-dev generates an image from the prompt
2. **Attention capture** – `attention-map-diffusers` (Flux2-modified) captures cross-attention during diffusion
3. **Layout extraction** – Prompt is split into comma-separated clauses; each clause’s tokens are used to aggregate attention and produce segmentation masks

## Usage

### CLI

```bash
cd /mnt/data0/teja/research_multiref
python -m diffusion_layout_generation.generate_layout --prompt "A wide shot of a room with a lamp, a chair, and a table"
```

Options:

- `--prompt`, `-p` – Text prompt (required)
- `--name`, `-n` – Output name (default: slug from prompt)
- `--output-dir`, `-o` – Output directory (default: `outputs`)
- `--no-extraction` – Only generate image + attention maps, skip layout extraction
- `--steps` – Inference steps (default: 15)
- `--guidance-scale` – CFG scale (default: 4.0)

### Python API

```python
from diffusion_layout_generation import generate_layout_from_prompt

result = generate_layout_from_prompt(
    prompt="A wide cinematic shot of a room with a brass lamp, a wooden chair, and a table",
    output_name="my_scene",
    output_dir="outputs",
)

# result contains:
# - image_path: path to generated image
# - attn_dir: path to attention maps
# - layout_dir: path to layouts (JSON, overlay PNG, per-object masks)
# - layouts: dict of object_name -> {segmentation_mask, bbox, ...}
```

## Dependencies

- `diffusers` (with Flux2Pipeline)
- `transformers`
- `attention-map-diffusers` (Flux2-modified, in `llm_based_layout/attention-map-diffusers`)
- `numpy`, `PIL`, `matplotlib`

Install from `llm_based_layout`:

```bash
cd llm_based_layout
pip install -e ./attention-map-diffusers
```

## Related Scripts

| Script | Purpose |
|--------|---------|
| `llm_based_layout/get_layout_directors_flux2.py` | Generate layouts for director prompts |
| `llm_based_layout/get_layout_moods_flux2.py` | Generate layouts for mood prompts |
| `llm_based_layout/batch_extract_layouts.py` | Batch extract layouts from existing attention maps |
| `llm_based_layout/generate_attn_viewer_flux2.py` | HTML viewer for attention maps |
| `llm_based_layout/extract_object_layouts_v2.py` | Extract layouts from attention maps (clause-based) |
