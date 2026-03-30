"""Repository and infer directory paths."""

from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parent
INFER_DIR = _UTILS_DIR.parent
V2_DIR = INFER_DIR.parent
LORA_TRAINING_DIR = V2_DIR.parent
REPO_ROOT = LORA_TRAINING_DIR.parent
# Python package root for `src.*` (train.py, layers_flux2, …)
TRAIN_PACKAGE_DIR = V2_DIR / "train"

REF_IMAGES_DIR = INFER_DIR / "images"
DEFAULT_STAGE1_OUTPUT = INFER_DIR / "outputs" / "stage1"
DEFAULT_STAGE2_OUTPUT = INFER_DIR / "outputs" / "stage2"

LLM_LAYOUT_DIR = REPO_ROOT / "llm_based_layout"
MODELS_CACHE_DIR = LLM_LAYOUT_DIR / "models"
DIFFUSION_LAYOUT_PKG = REPO_ROOT / "diffusion_layout_generation"
