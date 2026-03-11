"""
Delta Applicator.

Applies position deltas (dx, dy, dz) to placements.
"""

from stage2.utils.types import ObjectPlacement


def apply_deltas(
    placements: dict[str, ObjectPlacement],
    deltas: dict[str, tuple[float, float, float]],
) -> dict[str, ObjectPlacement]:
    """
    Add deltas to placement positions.
    deltas: {obj_id: (dx, dy, dz)} in world coords (meters)
    """
    for obj_id, (dx, dy, dz) in deltas.items():
        if obj_id in placements:
            p = placements[obj_id]
            p.x += dx
            p.y += dy
            p.z += dz
    return placements
