"""Subset of jsonl_datasets helpers (copied; no dependency on training_model.train)."""

from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def multiple_16(num: float) -> int:
    return int(round(num / 16) * 16)


def resolve_image_path(image_path: str, root_dir: str | None) -> str:
    if not image_path or not str(image_path).strip():
        return image_path
    p = Path(image_path).expanduser()
    if p.is_absolute():
        return str(p)
    if root_dir and str(root_dir).strip():
        return str(Path(root_dir).expanduser() / p)
    return str(p)


def load_image_safely(image_path, size: int, root_dir=None):
    path = resolve_image_path(image_path, root_dir)
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        print("file error: " + path)
        with open("failed_images.txt", "a") as f:
            f.write(f"{path}\n")
        return Image.new("RGB", (size, size), (255, 255, 255))
