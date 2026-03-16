"""
Parent Existence Validator.

After build_scene_graph, validates that every on:<x> reference exists in the layout.
Invalid references are fixed via LLM: demote to floor or map to a valid alternative parent.
"""

import json
from stage2.utils.api import call_llm, parse_json_response
from stage2.utils.types import SceneGraph


def _normalize_name(name: str) -> str:
    return name.lower().strip().replace(" ", "_")


def _get_layout_objects(relations: list[dict]) -> set[str]:
    return {r["object"] for r in relations}


def _find_invalid_on_refs(relations: list[dict], layout_objects: set[str]) -> list[dict]:
    layout_normalized = {_normalize_name(o): o for o in layout_objects}
    invalid = []

    for rel in relations:
        surface = rel.get("surface") or "floor"
        if not surface.startswith("on:"):
            continue

        parent_ref = surface[3:].strip()
        parent_norm = _normalize_name(parent_ref)

        if parent_norm in layout_normalized:
            continue

        invalid.append({
            "object": rel["object"],
            "surface": surface,
            "parent_ref": parent_ref,
            "valid_parents": list(layout_objects),
        })

    return invalid


PARENT_FIX_PROMPT = """You are fixing invalid object placement references in a 3D scene graph.

Some objects have surface "on:<parent>" but the parent does not exist in the scene.
For each invalid reference below, either:
  1. Demote to floor: set surface to "floor"
  2. Map to valid parent: set surface to "on:<valid_parent>" if a valid parent from the list would make semantic sense

Invalid references:
{invalid_json}

Valid object names in the scene:
{valid_parents}

Respond ONLY with a valid JSON array (no explanation, no markdown). Each element:
{{"object": "<object name>", "surface": "<floor | on:<valid_parent>>"}}
"""


def _fix_invalid_refs_with_llm(invalid: list[dict], layout_objects: set[str]) -> list[dict]:
    if not invalid:
        return []

    valid_parents = invalid[0]["valid_parents"] if invalid else []
    prompt = PARENT_FIX_PROMPT.format(
        invalid_json=json.dumps(
            [{"object": i["object"], "surface": i["surface"], "parent_ref": i["parent_ref"]} for i in invalid],
            indent=2,
        ),
        valid_parents=json.dumps(valid_parents),
    )

    try:
        raw = call_llm(prompt, max_tokens=1024)
        parsed = parse_json_response(raw)
    except Exception:
        return []

    fixes = []
    if isinstance(parsed, list):
        fixes = parsed
    elif isinstance(parsed, dict) and "fixes" in parsed:
        fixes = parsed["fixes"]
    elif isinstance(parsed, dict) and "object" in parsed:
        fixes = [parsed]

    layout_normalized = {_normalize_name(o): o for o in layout_objects}
    result = []
    for f in fixes:
        if not isinstance(f, dict) or "object" not in f or "surface" not in f:
            continue
        surf = f["surface"]
        if surf.startswith("on:"):
            parent = surf[3:].strip()
            if _normalize_name(parent) not in layout_normalized:
                surf = "floor"
        result.append({"object": f["object"], "surface": surf})
    return result


def _enforce_depth_coherence(relations: list[dict], layout_objects: set[str]) -> None:
    """
    Ensure child.layer == parent.layer for surface=on:<parent>.
    Child must be in same depth band as parent.
    """
    obj_to_rel = {r["object"]: r for r in relations}
    layout_normalized = {_normalize_name(o): o for o in layout_objects}

    for rel in relations:
        surface = rel.get("surface") or "floor"
        if not surface.startswith("on:"):
            continue

        parent_ref = surface[3:].strip()
        parent_canonical = layout_normalized.get(_normalize_name(parent_ref))
        if not parent_canonical or parent_canonical not in obj_to_rel:
            continue

        parent_rel = obj_to_rel[parent_canonical]
        child_layer = rel.get("layer", "midground")
        parent_layer = parent_rel.get("layer", "midground")

        if child_layer != parent_layer:
            rel["layer"] = parent_layer


def validate_parent_existence(graph: SceneGraph) -> SceneGraph:
    """
    Validate every on:<x> reference exists in the layout.
    Invalid refs are fixed via LLM (demote to floor or map to valid parent).
    Add depth coherence: child.layer = parent.layer for on:<parent>.
    """
    relations = [r.copy() for r in graph.relations]
    layout_objects = _get_layout_objects(relations)
    invalid = _find_invalid_on_refs(relations, layout_objects)

    if invalid:
        for inv in invalid:
            print(f"  [parent_check] {inv['object']}: surface={inv['surface']} — parent '{inv['parent_ref']}' not in layout, fixing via LLM...")

        fixes = _fix_invalid_refs_with_llm(invalid, layout_objects)
        fix_by_object = {f["object"]: f["surface"] for f in fixes if isinstance(f, dict) and "object" in f and "surface" in f}

        for inv in invalid:
            if inv["object"] not in fix_by_object:
                fix_by_object[inv["object"]] = "floor"

        for rel in relations:
            obj_name = rel.get("object")
            if obj_name in fix_by_object:
                new_surface = fix_by_object[obj_name]
                if new_surface in ("floor", "wall", "ceiling", "floating") or new_surface.startswith("on:"):
                    rel["surface"] = new_surface

    # Depth coherence: child must match parent layer
    _enforce_depth_coherence(relations, layout_objects)

    return SceneGraph(
        scene_type=graph.scene_type,
        gravity=graph.gravity,
        camera_facing=graph.camera_facing,
        relations=relations,
    )
