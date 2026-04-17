"""Map training images to captions from dataset ``metadata.jsonl`` (``file_name`` + ``text``).

``file_name`` entries look like ``images/0003/VideoId.jpg``. Training paths like
``.../hd_1280x720/0003/VideoId.png`` are matched by **(parent folder name, file stem)**, e.g.
``("0003", "VideoId")``, so different extensions (.jpg vs .png) still align and duplicate video
IDs in different shards stay disambiguated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

Key = Tuple[str, str]


def load_metadata_prompt_map(jsonl_path: str) -> Dict[Key, str]:
    """Return mapping (shard_dir, stem) -> caption text. Later JSONL lines win on duplicate keys."""
    out: Dict[Key, str] = {}
    if not jsonl_path or not str(jsonl_path).strip():
        return out
    path = os.path.expanduser(str(jsonl_path).strip())
    if not os.path.isfile(path):
        print(f"[canvas dataset] canvas_metadata_jsonl not found (skipping): {path}")
        return out

    dups = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fn = (obj.get("file_name") or "").strip()
            text = (obj.get("text") or "").strip()
            if not fn or not text:
                continue
            p = Path(fn)
            stem, shard = p.stem, p.parent.name
            if not stem or not shard:
                continue
            if (shard, stem) in out:
                dups += 1
            out[(shard, stem)] = text

    print(
        f"[canvas dataset] loaded {len(out)} metadata prompts from {path}"
        + (f" ({dups} duplicate keys overwritten)" if dups else "")
    )
    return out


def lookup_prompt(map_: Optional[Dict[Key, str]], resolved_image_path: str) -> Optional[str]:
    """Resolve ``resolved_image_path`` to caption if present in map."""
    if not map_:
        return None
    try:
        p = Path(resolved_image_path).resolve()
    except OSError:
        p = Path(resolved_image_path)
    stem, shard = p.stem, p.parent.name
    if not stem or not shard:
        return None
    return map_.get((shard, stem))
