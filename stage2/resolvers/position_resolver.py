"""
Spatial Position Resolver.

Two-phase placement:
  Phase A: Root anchors (floor, wall, ceiling, floating) with zone budget
  Phase B: Surface children packed onto parents (z = parent.z, deterministic)

Coordinate system:
  X: left(-) to right(+), center=0
  Y: floor=0, up=+
  Z: camera=0, back_wall=SCENE_DEPTH
"""

import random
from stage2.config import (
    DEPTH_BANDS, FLOOR_Y, CEILING_Y, BACK_WALL_Z,
    SCENE_WIDTH, SCENE_HEIGHT, ENV_GEOMETRY_THRESHOLD,
    ZONE_NAMES, ZONE_X, ZONE_BUDGET,
)
from stage2.utils.types import ObjectPlacement, ObjectSize, SceneGraph


# Anchor token -> normalized X in [-1, 1]
ANCHOR_X_MAP = {
    "center":        0.0,
    "center-left":  -0.25,
    "center-right":  0.25,
    "left":         -0.55,
    "right":         0.55,
    "corner-left":  -0.85,
    "corner-right":  0.85,
}

ANCHOR_Y_MAP = {
    "high":  0.75,
    "low":   0.15,
}


def _resolve_depth(layer: str, jitter: float = 0.15) -> float:
    z_min, z_max = DEPTH_BANDS[layer]
    mid = (z_min + z_max) / 2.0
    return mid + random.uniform(-jitter, jitter)


def _is_root_surface(surface: str) -> bool:
    s = surface or "floor"
    return s in ("floor", "wall", "ceiling", "floating")


def enforce_zone_budget(root_relations: list[dict]) -> dict[str, str]:
    """
    Assign each root object to a zone. Distribute to avoid clustering.
    Returns {object_name: zone_name}.
    """
    zone_counts = {z: 0 for z in ZONE_NAMES}
    result = {}

    for rel in root_relations:
        obj_name = rel["object"]
        # Pick zone with fewest objects, cycling if all at budget
        best_zone = min(ZONE_NAMES, key=lambda z: zone_counts[z])
        result[obj_name] = best_zone
        zone_counts[best_zone] += 1

    return result


def _place_roots(
    graph: SceneGraph,
    root_relations: list[dict],
    zone_assign: dict[str, str],
) -> dict[str, ObjectPlacement]:
    """Phase A: Place root objects with zone budget."""
    placements: dict[str, ObjectPlacement] = {}
    half_scene = SCENE_WIDTH / 2.0

    for rel in root_relations:
        name = rel["object"]
        layer = rel["layer"]
        anchor = rel.get("anchor", "center")
        surface = rel.get("surface", "floor")
        wall = rel.get("wall")
        facing = rel.get("facing", "camera")
        sz = rel.get("size_meters", {"w": 0.3, "h": 0.4, "d": 0.3})
        size = ObjectSize(w=sz["w"], h=sz["h"], d=sz["d"])

        zone = zone_assign.get(name, "center")
        zone_center_x = ZONE_X.get(zone, 0.0)
        jitter = 0.15
        x = zone_center_x + random.uniform(-jitter, jitter)
        x = max(-half_scene + size.w / 2, min(half_scene - size.w / 2, x))

        if surface == "wall":
            z = BACK_WALL_Z if (wall and "back" in str(wall).lower()) else DEPTH_BANDS[layer][1]
        else:
            z = _resolve_depth(layer)

        if not graph.gravity:
            norm_y = ANCHOR_Y_MAP.get(anchor, 0.5)
            y = FLOOR_Y + norm_y * SCENE_HEIGHT
        else:
            if surface == "floor":
                y = FLOOR_Y + size.h / 2.0
            elif surface == "wall":
                y = SCENE_HEIGHT * 0.6
                if anchor in ANCHOR_Y_MAP:
                    y = FLOOR_Y + ANCHOR_Y_MAP[anchor] * SCENE_HEIGHT
            elif surface == "ceiling":
                y = CEILING_Y - size.h / 2.0
            elif surface == "floating":
                y = FLOOR_Y + 0.5 * SCENE_HEIGHT
            else:
                y = FLOOR_Y + size.h / 2.0

        rot_y = 0.0
        if facing == "left":
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

    return placements


def _surface_pack_offset(
    child_name: str,
    child_size: ObjectSize,
    parent: ObjectPlacement,
    sibling_count: int,
) -> float:
    """
    X offset for child on parent surface. Distribute children to avoid overlap.
    Returns delta-x from parent center.
    """
    if sibling_count == 0:
        return random.uniform(-0.1, 0.1)
    parent_w = (parent.size or ObjectSize(0.5, 0.5, 0.5)).w
    child_w = child_size.w
    max_offset = (parent_w / 2) - (child_w / 2) - 0.05
    if max_offset <= 0:
        return 0.0
    step = (2 * max_offset) / (sibling_count + 1)
    pos = -max_offset + step * (sibling_count + 1)
    return pos + random.uniform(-0.05, 0.05)


def _place_surface_children(
    graph: SceneGraph,
    child_relations: list[dict],
    placements: dict[str, ObjectPlacement],
) -> None:
    """Phase B: Place surface children onto parents. Modifies placements in place."""
    parent_child_count: dict[str, int] = {}

    for rel in child_relations:
        name = rel["object"]
        surface = rel.get("surface", "")
        if not surface.startswith("on:"):
            continue
        parent_name = surface[3:].strip()
        parent = placements.get(parent_name)
        if not parent or not parent.size:
            continue

        sz = rel.get("size_meters", {"w": 0.3, "h": 0.4, "d": 0.3})
        size = ObjectSize(w=sz["w"], h=sz["h"], d=sz["d"])
        layer = rel["layer"]
        facing = rel.get("facing", "camera")

        sibling_idx = parent_child_count.get(parent_name, 0)
        dx = _surface_pack_offset(name, size, parent, sibling_idx)
        parent_child_count[parent_name] = sibling_idx + 1

        x = parent.x + dx
        z = parent.z
        y = parent.y + parent.size.h / 2.0 + size.h / 2.0

        rot_y = 0.0
        if facing == "left":
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
            anchor_ref=rel.get("anchor"),
            floating=(not graph.gravity),
            environment_geometry=env_geom,
        )


def resolve_positions(graph: SceneGraph) -> dict[str, ObjectPlacement]:
    """Two-phase placement: roots with zone budget, then surface children."""
    root_relations = [r for r in graph.relations if _is_root_surface(r.get("surface", "floor"))]
    child_relations = [r for r in graph.relations if not _is_root_surface(r.get("surface", "floor"))]

    zone_assign = enforce_zone_budget(root_relations)
    placements = _place_roots(graph, root_relations, zone_assign)
    _place_surface_children(graph, child_relations, placements)

    return placements
