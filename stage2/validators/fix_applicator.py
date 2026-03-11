"""
Fix Applicator.

Takes VLMResult issues and applies algorithmic fixes to placements.
Only touches objects with critical/warning severity.
Minor issues are logged only.
"""

import math
from stage2.utils.types import ObjectPlacement, ObjectSize, VLMResult
from stage2.config import SCENE_WIDTH, SCENE_HEIGHT, FLOOR_Y


def apply_fixes(
    placements: dict[str, ObjectPlacement],
    vlm_result: VLMResult,
) -> dict[str, ObjectPlacement]:

    for issue in vlm_result.issues:
        if issue.severity == "minor":
            continue

        obj_name = issue.object
        hint     = issue.fix_hint

        if obj_name == "scene" or obj_name not in placements:
            # scene-level composition hints — handle separately
            _apply_scene_fix(placements, hint)
            continue

        p = placements[obj_name]
        s = p.size or ObjectSize(0.3, 0.4, 0.3)

        if hint == "shift_lateral":
            # shift away from scene center
            direction = 1 if p.x >= 0 else -1
            step = s.w * 0.6
            new_x = p.x + direction * step
            half = SCENE_WIDTH / 2.0 - s.w / 2.0
            p.x = max(-half, min(half, new_x))

        elif hint == "snap_to_surface":
            # snap Y to floor or parent surface
            if p.surface and p.surface.startswith("on:"):
                parent = placements.get(p.surface[3:])
                if parent and parent.size:
                    p.y = parent.y + parent.size.h / 2.0 + s.h / 2.0
                else:
                    p.y = FLOOR_Y + s.h / 2.0
            else:
                p.y = FLOOR_Y + s.h / 2.0

        elif hint == "push_apart":
            # push toward scene edge along X
            direction = 1 if p.x >= 0 else -1
            p.x += direction * (s.w + 0.1)
            half = SCENE_WIDTH / 2.0 - s.w / 2.0
            p.x = max(-half, min(half, p.x))

        elif hint == "move_toward_center":
            # move X toward 0
            p.x = p.x * 0.5

        elif hint == "increase_depth_separation":
            # push background objects further back
            from stage2.config import DEPTH_BANDS
            if p.layer in DEPTH_BANDS:
                z_min, z_max = DEPTH_BANDS[p.layer]
                p.z = (z_min + z_max) / 2.0 + (z_max - z_min) * 0.3

    return placements


def _apply_scene_fix(placements: dict[str, ObjectPlacement], hint: str):
    """Scene-level composition adjustments."""
    if hint == "increase_depth_separation":
        from stage2.config import DEPTH_BANDS
        for p in placements.values():
            if p.layer in DEPTH_BANDS:
                z_min, z_max = DEPTH_BANDS[p.layer]
                mid = (z_min + z_max) / 2.0
                p.z = mid + (p.z - mid) * 1.2   # spread outward from band midpoint
