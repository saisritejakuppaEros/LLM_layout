"""
Bounds Clamp.

After collision resolver, clamp all placements within scene bounds:
  - (x ± w/2) within ±SCENE_WIDTH/2
  - (y + h/2) within CEILING_Y
"""

from stage2.config import SCENE_WIDTH, CEILING_Y
from stage2.utils.types import ObjectPlacement, ObjectSize


def clamp_bounds(placements: dict[str, ObjectPlacement]) -> dict[str, ObjectPlacement]:
    """
    Clamp placement positions so that:
      - x - w/2 >= -SCENE_WIDTH/2  and  x + w/2 <= SCENE_WIDTH/2
      - y + h/2 <= CEILING_Y
    """
    half_width = SCENE_WIDTH / 2.0

    for p in placements.values():
        if p.environment_geometry:
            continue
        s = p.size or ObjectSize(0.3, 0.4, 0.3)
        half_w = s.w / 2.0
        half_h = s.h / 2.0

        x_min = -half_width + half_w
        x_max = half_width - half_w
        p.x = max(x_min, min(x_max, p.x))

        y_max = CEILING_Y - half_h
        if p.y > y_max:
            p.y = y_max

    return placements
