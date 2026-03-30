"""Reference object images under ``infer/images`` (person, bottle, bucket)."""

from dataclasses import dataclass
from pathlib import Path

from .paths import REF_IMAGES_DIR

OBJECT_KEYS = ("person", "bottle", "bucket")


@dataclass(frozen=True)
class ReferenceImageSet:
    """Paths to reference assets for the three objects."""

    person: Path
    bottle: Path
    bucket: Path

    def as_dict(self) -> dict[str, Path]:
        return {"person": self.person, "bottle": self.bottle, "bucket": self.bucket}


def default_reference_paths() -> ReferenceImageSet:
    return ReferenceImageSet(
        person=REF_IMAGES_DIR / "person.jpg",
        bottle=REF_IMAGES_DIR / "bottle.jpg",
        bucket=REF_IMAGES_DIR / "bucket.jpg",
    )


def validate_reference_images(refs: ReferenceImageSet | None = None) -> list[str]:
    """Return list of missing keys (empty if all files exist)."""
    refs = refs or default_reference_paths()
    missing: list[str] = []
    for key, path in refs.as_dict().items():
        if not path.is_file():
            missing.append(key)
    return missing
