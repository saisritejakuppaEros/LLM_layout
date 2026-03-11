"""
Pass 3 — Delta Generator.

VLM suggests position deltas (dx, dy, dz) for failed objects.
Includes position table and explicit dx/dy/dz instructions so VLM uses all axes.
"""

import json
import sys
from stage2.config import SCENE_WIDTH, SCENE_HEIGHT, SCENE_DEPTH
from stage2.utils.api import call_vlm, parse_json_response
from stage2.utils.types import Question, DeltaResult

DELTA_PROMPT = """You are a 3D scene layout fixer.

This front-view diagram has layout issues. The following questions failed:
{failed_questions_text}

Current object positions (meters):
{position_table}

Coordinate system:
  X: left(-) to right(+), range [-{half_w:.1f}, {half_w:.1f}]
  Y: floor(0) to ceiling({height:.1f})
  Z: camera(0) to back wall({depth:.1f})

Object bounding boxes in pixel coords (x1, y1, x2, y2) — use to locate objects in the image:
{bbox_text}

Suggest position deltas (dx, dy, dz in meters) to fix the issues.
- dx: move left (negative) or right (positive)
- dy: move down (negative) or up (positive)
- dz: move forward (negative) or back (positive)

Use dx when object is too far left/right or overlapping horizontally.
Use dy when object is too high/low or overlapping vertically.
Use dz when depth placement is wrong.

Do NOT default to dz only. Consider dx and dy for lateral/vertical fixes.
Only suggest deltas for objects that clearly need to move. Use small values (0.1 to 0.5 meters).

Return ONLY a valid JSON object (no explanation, no markdown):
{{
  "deltas": {{
    "<object_name>": {{"dx": <float>, "dy": <float>, "dz": <float>}},
    ...
  }}
}}
"""


def generate_deltas(
    diagram_path: str,
    failed_questions: list[Question],
    object_bboxes_px: dict[str, tuple[int, int, int, int]],
    placements: dict,
) -> DeltaResult:
    """
    VLM suggests position deltas for objects that failed validation.
    placements: dict of {obj_name: {x, y, z, ...}} or ObjectPlacement
    """
    if not failed_questions:
        return DeltaResult(deltas={})

    failed_text = "\n".join(
        f"- {q.id}: {q.text}" for q in failed_questions
    )
    bbox_lines = [
        f"  {name}: ({x1},{y1},{x2},{y2})"
        for name, (x1, y1, x2, y2) in object_bboxes_px.items()
    ]
    bbox_text = "\n".join(bbox_lines)

    position_lines = []
    for name, p in placements.items():
        if hasattr(p, "x"):
            x, y, z = p.x, p.y, p.z
        else:
            x = p.get("x", 0)
            y = p.get("y", 0)
            z = p.get("z", 0)
        position_lines.append(f"  {name}: x={x:.2f}, y={y:.2f}, z={z:.2f}")
    position_table = "\n".join(position_lines)

    half_w = SCENE_WIDTH / 2.0
    prompt = DELTA_PROMPT.format(
        failed_questions_text=failed_text,
        position_table=position_table,
        half_w=half_w,
        height=SCENE_HEIGHT,
        depth=SCENE_DEPTH,
        bbox_text=bbox_text,
    )

    try:
        raw = call_vlm(diagram_path, prompt, max_tokens=1024)
        parsed = parse_json_response(raw)
    except Exception as e:
        print(f"[Delta-gen] Failed: {e}", file=sys.stderr)
        return DeltaResult(deltas={})

    deltas = {}
    for obj_id, d in parsed.get("deltas", {}).items():
        if not isinstance(d, dict):
            continue
        dx = float(d.get("dx", 0))
        dy = float(d.get("dy", 0))
        dz = float(d.get("dz", 0))
        deltas[str(obj_id)] = (dx, dy, dz)

    return DeltaResult(deltas=deltas)
