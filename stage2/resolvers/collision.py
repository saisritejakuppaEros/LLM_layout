"""
DIoU3D Collision Checker + Push-Apart Solver.

Checks all object pairs for 3D overlap using Distance-IoU.
Push-apart: nudges lower-priority objects along XZ plane.
"""

import math
from stage2.config import DIOU_COLLISION_THRESH, DEPTH_BANDS, SCENE_WIDTH
from stage2.utils.types import ObjectPlacement, ObjectSize


PRIORITY_MAP = {"primary": 3, "secondary": 2, "tertiary": 1}


def _bbox(p: ObjectPlacement) -> tuple[float, float, float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max, z_min, z_max)."""
    s = p.size or ObjectSize(0.3, 0.4, 0.3)
    return (
        p.x - s.w / 2, p.x + s.w / 2,
        p.y - s.h / 2, p.y + s.h / 2,
        p.z - s.d / 2, p.z + s.d / 2,
    )


def _iou3d(a: ObjectPlacement, b: ObjectPlacement) -> float:
    ax1, ax2, ay1, ay2, az1, az2 = _bbox(a)
    bx1, bx2, by1, by2, bz1, bz2 = _bbox(b)

    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    iz = max(0.0, min(az2, bz2) - max(az1, bz1))
    inter = ix * iy * iz

    sa = a.size or ObjectSize(0.3, 0.4, 0.3)
    sb = b.size or ObjectSize(0.3, 0.4, 0.3)
    vol_a = sa.w * sa.h * sa.d
    vol_b = sb.w * sb.h * sb.d
    union = vol_a + vol_b - inter
    return inter / union if union > 0 else 0.0


def _center_dist_sq(a: ObjectPlacement, b: ObjectPlacement) -> float:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2


def _enclosing_diag_sq(a: ObjectPlacement, b: ObjectPlacement) -> float:
    ax1, ax2, ay1, ay2, az1, az2 = _bbox(a)
    bx1, bx2, by1, by2, bz1, bz2 = _bbox(b)
    dx = max(ax2, bx2) - min(ax1, bx1)
    dy = max(ay2, by2) - min(ay1, by1)
    dz = max(az2, bz2) - min(az1, bz1)
    return dx**2 + dy**2 + dz**2


def diou3d(a: ObjectPlacement, b: ObjectPlacement) -> float:
    iou = _iou3d(a, b)
    diag_sq = _enclosing_diag_sq(a, b)
    if diag_sq == 0:
        return iou
    return iou - _center_dist_sq(a, b) / diag_sq


def _push_apart(mover: ObjectPlacement, fixed: ObjectPlacement):
    """Push mover away from fixed along XZ plane."""
    dx = mover.x - fixed.x
    dz = mover.z - fixed.z
    dist = math.sqrt(dx**2 + dz**2)

    if dist < 1e-4:
        # directly on top — push right
        dx, dz = 1.0, 0.0
        dist = 1.0

    # required separation
    ms = mover.size or ObjectSize(0.3, 0.4, 0.3)
    fs = fixed.size or ObjectSize(0.3, 0.4, 0.3)
    min_sep = (ms.w + fs.w) / 2.0 + 0.05

    push = min_sep - dist
    if push > 0:
        mover.x += (dx / dist) * push
        mover.z += (dz / dist) * push

    # clamp X to scene bounds
    half_w = SCENE_WIDTH / 2.0
    mover.x = max(-half_w + (ms.w / 2), min(half_w - (ms.w / 2), mover.x))


def resolve_collisions(
    placements: dict[str, ObjectPlacement],
    priorities: dict[str, str],          # object_name → semantic_importance
    max_iters: int = 10,
) -> dict[str, ObjectPlacement]:
    names = list(placements.keys())

    for _ in range(max_iters):
        collision_found = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = placements[names[i]]
                b = placements[names[j]]

                # skip wall-mounted objects — they don't collide with floor objects
                if (a.surface == "wall") != (b.surface == "wall"):
                    continue

                # skip environment geometry (large planes: floor, carpet, curtains)
                if a.environment_geometry or b.environment_geometry:
                    continue

                score = diou3d(a, b)
                if score > DIOU_COLLISION_THRESH:
                    collision_found = True
                    pri_a = PRIORITY_MAP.get(priorities.get(a.name, "tertiary"), 1)
                    pri_b = PRIORITY_MAP.get(priorities.get(b.name, "tertiary"), 1)

                    # move the lower-priority object
                    if pri_a >= pri_b:
                        _push_apart(b, a)
                    else:
                        _push_apart(a, b)

        if not collision_found:
            break

    return placements
