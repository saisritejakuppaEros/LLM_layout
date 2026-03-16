"""
LayoutVLM-style relational self-consistency.

When VLM validation fails, instead of delta proposals:
1. VLM describes spatial relations from front + bird's-eye views
2. VLM checks consistency between layout coords and description
3. If inconsistent: extract relation corrections, re-run resolver
4. Fallback: return None to use delta-based fix
"""

import json
from stage2.utils.api import call_vlm, parse_json_response
from stage2.utils.types import SceneGraph, Question
from stage2.config import LAYOUT_VLM_CONSISTENCY_THRESH


DESCRIBE_PROMPT = """Describe this 3D scene layout's spatial relations in 2-4 sentences.

You see: (1) front view (XY), (2) bird's-eye view (XZ, camera at top, back at bottom).

Mention: which objects are on which surfaces (e.g. "cup on table"), depth ordering (e.g. "chair in foreground, clock in background"), left/right positions, and any obvious problems (e.g. "cup appears to float" or "overlap between X and Y").

Return ONLY plain text, no JSON."""


CONSISTENCY_PROMPT = """You see two views of a scene layout and a description of its spatial relations.

Layout coordinates (meters):
{layout_text}

Description: "{description}"

Does the layout match the description? Consider: support (objects on surfaces), depth (Z ordering), overlap. Score 0.0 (completely inconsistent) to 1.0 (fully consistent).

Return ONLY a valid JSON object: {{"score": <float 0-1>, "reason": "<brief reason>"}}"""


CORRECTIONS_PROMPT = """The layout has validation failures. The VLM described the layout as:
"{description}"

Failed questions: {failed_text}

Current relations: {relations_json}

Propose relation corrections to fix the issues. For each object that needs correction, suggest new surface and/or layer.
- surface: "floor" | "wall" | "on:<parent>" 
- layer: "foreground" | "midground" | "background"

Only include objects that need changes. Return ONLY valid JSON:
{{"corrections": [{{"object": "<name>", "surface": "<value>", "layer": "<value>"}}, ...]}}

If no corrections can fix this, return {{"corrections": []}}."""


def generate_spatial_description(
    front_path: str,
    birds_eye_path: str,
) -> str:
    """VLM describes spatial relations from both views."""
    try:
        raw = call_vlm(
            image_paths=[front_path, birds_eye_path],
            prompt=DESCRIBE_PROMPT,
            max_tokens=512,
        )
        return raw.strip() if raw else ""
    except Exception:
        return ""


def check_layout_consistency(
    layout_dict: dict,
    description: str,
    front_path: str,
    birds_eye_path: str,
) -> float:
    """VLM scores consistency between layout and description. Returns 0-1."""
    if not description:
        return 0.0

    layout_lines = []
    for name, obj in layout_dict.items():
        x = obj.get("x", 0)
        y = obj.get("y", 0)
        z = obj.get("z", 0)
        surface = obj.get("surface", "floor")
        layer = obj.get("layer", "midground")
        layout_lines.append(f"  {name}: x={x:.2f}, y={y:.2f}, z={z:.2f}, surface={surface}, layer={layer}")
    layout_text = "\n".join(layout_lines)

    prompt = CONSISTENCY_PROMPT.format(
        layout_text=layout_text,
        description=description,
    )
    try:
        raw = call_vlm(
            image_paths=[front_path, birds_eye_path],
            prompt=prompt,
            max_tokens=256,
        )
        parsed = parse_json_response(raw)
        return float(parsed.get("score", 0.0))
    except Exception:
        return 0.0


def extract_relation_corrections(
    description: str,
    failed_questions: list[Question],
    relations: list[dict],
) -> list[dict]:
    """VLM proposes relation corrections. Returns list of {object, surface?, layer?}."""
    failed_text = "\n".join(f"- {q.id}: {q.text}" for q in failed_questions)
    relations_json = json.dumps(relations, indent=2)

    prompt = CORRECTIONS_PROMPT.format(
        description=description,
        failed_text=failed_text,
        relations_json=relations_json,
    )
    try:
        from stage2.utils.api import call_llm
        raw = call_llm(prompt, max_tokens=1024)
        parsed = parse_json_response(raw)
        return parsed.get("corrections", [])
    except Exception:
        return []


def apply_corrections_to_graph(graph: SceneGraph, corrections: list[dict]) -> SceneGraph:
    """Merge corrections into graph relations."""
    if not corrections:
        return graph

    fix_by_obj = {c["object"]: c for c in corrections if isinstance(c, dict) and "object" in c}
    relations = []
    for r in graph.relations.copy():
        rel = r.copy()
        obj = rel.get("object")
        if obj in fix_by_obj:
            fix = fix_by_obj[obj]
            if "surface" in fix:
                rel["surface"] = fix["surface"]
            if "layer" in fix:
                rel["layer"] = fix["layer"]
        relations.append(rel)

    return SceneGraph(
        scene_type=graph.scene_type,
        gravity=graph.gravity,
        camera_facing=graph.camera_facing,
        relations=relations,
    )


def try_relational_resolve(
    front_path: str,
    birds_eye_path: str,
    layout_dict: dict,
    graph: SceneGraph,
    failed_questions: list[Question],
) -> tuple[bool, SceneGraph | None]:
    """
    Try LayoutVLM relational re-solving.
    Returns (success, modified_graph or None).
    If success, caller should re-run resolve_positions, collision, bounds, occlusion.
    """
    description = generate_spatial_description(front_path, birds_eye_path)
    if not description:
        return False, None

    score = check_layout_consistency(layout_dict, description, front_path, birds_eye_path)
    if score >= LAYOUT_VLM_CONSISTENCY_THRESH:
        return False, None  # Consistent enough, no need to re-solve

    corrections = extract_relation_corrections(
        description, failed_questions, graph.relations
    )
    if not corrections:
        return False, None

    modified_graph = apply_corrections_to_graph(graph, corrections)
    return True, modified_graph
