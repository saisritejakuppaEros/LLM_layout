"""
Occlusion Budget Checker (front-view orthographic projection).

Projects all objects onto a 2D front canvas (X, Y).
Checks that deeper-layer objects retain >= OCCLUSION_MIN_VISIBLE of their projected area.
Shifts laterally if violated.
"""

from stage2.config import OCCLUSION_MIN_VISIBLE, SCENE_WIDTH, SCENE_HEIGHT
from stage2.utils.types import ObjectPlacement, ObjectSize

LAYER_ORDER = ["foreground", "midground", "background"]


def _proj_rect(p: ObjectPlacement) -> tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) in front-view projection."""
    s = p.size or ObjectSize(0.3, 0.4, 0.3)
    return (p.x - s.w / 2, p.x + s.w / 2, p.y - s.h / 2, p.y + s.h / 2)


def _rect_area(r: tuple) -> float:
    return max(0.0, r[1] - r[0]) * max(0.0, r[3] - r[2])


def _intersection_rect(a: tuple, b: tuple) -> tuple:
    return (max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3]))


def _visible_fraction(target: ObjectPlacement, occluders: list[ObjectPlacement]) -> float:
    """Approximate visible fraction of target given a list of occluders (closer layer)."""
    t_rect = _proj_rect(target)
    t_area = _rect_area(t_rect)
    if t_area <= 0:
        return 1.0

    # Approximate: subtract union of all occlusion overlaps (ignores partial multi-occluder cases)
    occluded = 0.0
    for occ in occluders:
        inter = _intersection_rect(t_rect, _proj_rect(occ))
        occluded += _rect_area(inter)

    occluded = min(occluded, t_area)
    return 1.0 - occluded / t_area


def _shift_to_reduce_occlusion(target: ObjectPlacement, occluders: list[ObjectPlacement]):
    """Shift target laterally until visibility improves or scene edge reached."""
    half_scene = SCENE_WIDTH / 2.0
    s = target.size or ObjectSize(0.3, 0.4, 0.3)
    step = s.w * 0.5
    best_x = target.x
    best_vis = _visible_fraction(target, occluders)

    for direction in [1, -1]:
        test_x = target.x
        for _ in range(8):
            test_x += direction * step
            if abs(test_x) + s.w / 2 > half_scene:
                break
            target.x = test_x
            vis = _visible_fraction(target, occluders)
            if vis > best_vis:
                best_vis = vis
                best_x = test_x
            if vis >= OCCLUSION_MIN_VISIBLE:
                return

    target.x = best_x


def resolve_occlusions(placements: dict[str, ObjectPlacement]) -> dict[str, ObjectPlacement]:
    # Group by layer
    by_layer: dict[str, list[ObjectPlacement]] = {l: [] for l in LAYER_ORDER}
    for p in placements.values():
        if p.layer in by_layer and p.surface != "wall":
            by_layer[p.layer].append(p)

    # Check midground against foreground occluders
    for target in by_layer["midground"]:
        vis = _visible_fraction(target, by_layer["foreground"])
        if vis < OCCLUSION_MIN_VISIBLE:
            _shift_to_reduce_occlusion(target, by_layer["foreground"])

    # Check background against foreground + midground occluders
    all_closer = by_layer["foreground"] + by_layer["midground"]
    for target in by_layer["background"]:
        vis = _visible_fraction(target, all_closer)
        if vis < OCCLUSION_MIN_VISIBLE:
            _shift_to_reduce_occlusion(target, all_closer)

    return placements
