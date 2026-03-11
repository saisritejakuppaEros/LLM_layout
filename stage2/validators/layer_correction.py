"""
Layer Correction Pass.

If surface=wall and wall contains "back", force layer=background.
"""

from stage2.utils.types import SceneGraph


def correct_layer_for_wall(graph: SceneGraph) -> SceneGraph:
    """
    For relations with surface=wall and wall containing "back", set layer=background.
    """
    relations = []
    for r in graph.relations:
        rel = r.copy()
        surface = rel.get("surface") or "floor"
        wall = rel.get("wall") or ""
        if surface == "wall" and "back" in str(wall).lower():
            rel["layer"] = "background"
        relations.append(rel)

    return SceneGraph(
        scene_type=graph.scene_type,
        gravity=graph.gravity,
        camera_facing=graph.camera_facing,
        relations=relations,
    )
