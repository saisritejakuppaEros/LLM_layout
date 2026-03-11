"""
Spatial Position Resolver.

Converts scene graph semantic anchors → concrete (x, y, z) positions.
Pure algorithmic — no LLM calls.

Coordinate system:
  X: left(-) to right(+), center=0
  Y: floor=0, up=+
  Z: camera=0, back_wall=SCENE_DEPTH
"""

import random
from stage2.config import (
    DEPTH_BANDS, FLOOR_Y, CEILING_Y, BACK_WALL_Z,
    SCENE_WIDTH, SCENE_HEIGHT, ENV_GEOMETRY_THRESHOLD,
)
from stage2.utils.types import ObjectPlacement, ObjectSize, SceneGraph


# Anchor token → normalized X in [-1, 1]
ANCHOR_X_MAP = {
    "center":        0.0,
    "center-left":  -0.25,
    "center-right":  0.25,
    "left":         -0.55,
    "right":         0.55,
    "corner-left":  -0.85,
    "corner-right":  0.85,
}

# Anchor token → normalized Y bias (applied on top of surface Y)
ANCHOR_Y_MAP = {
    "high":  0.75,
    "low":   0.15,
}


def _resolve_depth(layer: str, jitter: float = 0.15) -> float:
    z_min, z_max = DEPTH_BANDS[layer]
    mid = (z_min + z_max) / 2.0
    return mid + random.uniform(-jitter, jitter)


def _resolve_x(anchor: str, half_w: float) -> float:
    token = anchor.split(":")[0] if ":" in anchor else anchor
    norm = ANCHOR_X_MAP.get(token, 0.0)
    half_scene = SCENE_WIDTH / 2.0
    x = norm * half_scene
    # small jitter to avoid perfect grid
    x += random.uniform(-0.1, 0.1)
    return x


def _resolve_y_gravity(surface: str, size: ObjectSize, placements: dict[str, ObjectPlacement]) -> float:
    if surface == "floor":
        return FLOOR_Y + size.h / 2.0

    if surface == "wall":
        return SCENE_HEIGHT * 0.6   # default wall-mount height, overridden by anchor Y

    if surface == "ceiling":
        return CEILING_Y - size.h / 2.0

    if surface and surface.startswith("on:"):
        parent_name = surface[3:]
        parent = placements.get(parent_name)
        if parent and parent.size:
            return parent.y + parent.size.h / 2.0 + size.h / 2.0
        # parent not placed yet — floor fallback, will be corrected in post-pass
        return FLOOR_Y + size.h / 2.0

    return FLOOR_Y + size.h / 2.0


def _resolve_y_float(anchor: str, layer: str) -> float:
    """Y for gravity=false scenes."""
    norm = ANCHOR_Y_MAP.get(anchor, 0.5)
    return FLOOR_Y + norm * SCENE_HEIGHT


def _resolve_near(anchor: str, placements: dict[str, ObjectPlacement], half_w: float) -> float | None:
    """Return X offset for near:<object> anchors."""
    if anchor.startswith("near:"):
        ref_name = anchor[5:]
        ref = placements.get(ref_name)
        if ref:
            offset = (ref.size.w / 2.0 + half_w + 0.05) if ref.size else 0.35
            return ref.x + offset * random.choice([-1, 1])
    return None


def resolve_positions(graph: SceneGraph) -> dict[str, ObjectPlacement]:
    placements: dict[str, ObjectPlacement] = {}

    # Sort: process supporting objects first (floor items before surface items)
    def sort_key(rel):
        surface = rel.get("surface", "floor")
        if surface == "floor" or surface == "wall":
            return 0
        if surface.startswith("on:"):
            return 1
        return 2

    sorted_relations = sorted(graph.relations, key=sort_key)

    for rel in sorted_relations:
        name    = rel["object"]
        layer   = rel["layer"]
        anchor  = rel.get("anchor", "center")
        surface = rel.get("surface", "floor")
        wall    = rel.get("wall")
        facing  = rel.get("facing", "camera")
        sz      = rel.get("size_meters", {"w": 0.3, "h": 0.4, "d": 0.3})
        size    = ObjectSize(w=sz["w"], h=sz["h"], d=sz["d"])

        # --- Z (depth) ---
        if surface == "wall":
            z = BACK_WALL_Z if (wall and "back" in wall) else DEPTH_BANDS[layer][1]
        elif surface and surface.startswith("on:"):
            z = 0.0  # placeholder; second pass will set child.z = parent.z
        else:
            z = _resolve_depth(layer)

        # --- X ---
        near_x = _resolve_near(anchor, placements, size.w / 2.0)
        x = near_x if near_x is not None else _resolve_x(anchor, size.w / 2.0)

        # --- Y ---
        if not graph.gravity:
            y = _resolve_y_float(anchor, layer)
        else:
            y = _resolve_y_gravity(surface, size, placements)
            # wall-height override
            if surface == "wall" and anchor in ANCHOR_Y_MAP:
                y = FLOOR_Y + ANCHOR_Y_MAP[anchor] * SCENE_HEIGHT

        # --- Rotation ---
        rot_y = 0.0
        if facing == "camera":
            rot_y = 0.0
        elif facing == "left":
            rot_y = 90.0
        elif facing == "right":
            rot_y = -90.0

        env_geom = size.w > ENV_GEOMETRY_THRESHOLD or size.d > ENV_GEOMETRY_THRESHOLD
        placements[name] = ObjectPlacement(
            name=name, layer=layer,
            x=x, y=y, z=z,
            rot_y=rot_y,
            size=size,
            surface=surface,
            anchor_ref=anchor,
            floating=(not graph.gravity),
            environment_geometry=env_geom,
        )

    # Second pass: fix on:<parent> placements (Y and Z inheritance from parent)
    for name, p in placements.items():
        if p.surface and p.surface.startswith("on:"):
            parent_name = p.surface[3:]
            parent = placements.get(parent_name)
            if parent and parent.size and p.size:
                p.y = parent.y + parent.size.h / 2.0 + p.size.h / 2.0
                p.z = parent.z  # child inherits parent's depth

    return placements
