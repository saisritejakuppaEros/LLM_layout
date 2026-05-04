#!/usr/bin/env python3
"""
HTTP client for ``server_flux.py``: sends a caption and optional reference images, saves PNG output.

By default runs **4** generations with different seeds and writes files like
``{stem}_var01_seed42.png`` … ``{stem}_var04_seed45.png`` in the same directory as ``-o`` (see ``--variants`` / ``--single``).

Example::

  python client_flux.py \\
    --url http://127.0.0.1:8765 \\
    --caption "A cinematic scene matching the references." \\
    --ref ./a.png ./b.png \\
    -o out/run.png

  # → out/run_var01_seed42.png … (default 4 variants)

Requires: requests
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any

import requests


def main() -> int:
    p = argparse.ArgumentParser(description="Call FLUX server /generate (caption + optional refs).")
    p.add_argument("--url", default="http://127.0.0.1:8765", help="Server base URL (no trailing slash).")
    p.add_argument("--caption", required=True, help="Generation prompt / caption.")
    p.add_argument(
        "--ref",
        nargs="*",
        default=[],
        metavar="PATH",
        help="Optional reference images (0 or more).",
    )
    p.add_argument(
        "-o",
        "--output",
        default="flux_client_out.png",
        help="Output path. For multiple variants, each file is {stem}_varNN_seedS{suffix} in the same directory.",
    )
    p.add_argument(
        "--variants",
        type=int,
        default=4,
        metavar="N",
        help="Number of images to generate with different seeds (default: 4). Use 1 for a single file at -o.",
    )
    p.add_argument(
        "--single",
        action="store_true",
        help="Shortcut for --variants 1 (one image written exactly to -o).",
    )
    p.add_argument("--seed", type=int, default=42, help="Base seed for the first variant.")
    p.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Added to seed for each subsequent variant (seed, seed+step, seed+2*step, …).",
    )
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument(
        "--caption-upsample-temperature",
        type=float,
        default=None,
        help="Optional; e.g. 0.15 for caption upsampling on the server.",
    )
    p.add_argument("--health-only", action="store_true", help="GET /health and exit.")
    args = p.parse_args()

    if args.single:
        args.variants = 1

    if args.variants < 1:
        print("--variants must be >= 1", file=sys.stderr)
        return 2

    base = args.url.rstrip("/")

    if args.health_only:
        r = requests.get(f"{base}/health", timeout=30)
        print(r.status_code, r.text)
        return 0 if r.ok else 1

    for path in args.ref:
        if not Path(path).is_file():
            print(f"Missing ref image: {path}", file=sys.stderr)
            return 2

    # Read refs once; fresh BytesIO per request for multipart.
    ref_bytes: list[tuple[str, bytes]] = [
        (Path(p).name, Path(p).read_bytes()) for p in args.ref
    ]

    out_base = Path(args.output)
    out_dir = out_base.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = out_base.suffix or ".png"
    base_stem = out_base.stem or "flux_out"

    for vi in range(args.variants):
        use_seed = args.seed + vi * args.seed_step
        if args.variants == 1:
            out_path = out_base if out_base.suffix else out_dir / f"{base_stem}{ext}"
        else:
            # Distinct names per run: ordinal + seed (avoids any overwrite / confusion with -o).
            out_path = out_dir / f"{base_stem}_var{vi + 1:02d}_seed{use_seed}{ext}"

        data: dict[str, str | int | float] = {
            "caption": args.caption,
            "seed": use_seed,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "width": args.width,
            "height": args.height,
        }
        if args.caption_upsample_temperature is not None:
            data["caption_upsample_temperature"] = args.caption_upsample_temperature

        files: list[tuple[str, tuple[str, Any, str]]] | None
        if ref_bytes:
            files = []
            for name, blob in ref_bytes:
                files.append(("ref_images", (name, io.BytesIO(blob), "image/png")))
        else:
            files = None

        r = requests.post(
            f"{base}/generate",
            data=data,
            files=files,
            timeout=3600,
        )

        if not r.ok:
            print(f"HTTP {r.status_code} (variant {vi + 1}/{args.variants}, seed={use_seed}): {r.text}", file=sys.stderr)
            return 1

        out_path.write_bytes(r.content)
        print(f"Wrote {out_path.resolve()} ({len(r.content)} bytes) seed={use_seed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
