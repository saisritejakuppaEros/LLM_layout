"""
Pass 1 — Question Generator.

Text LLM generates validation questions from scene graph + layout.
Questions are constrained to: LAYOUT (position) and GRAVITY/FEASIBILITY (support).
"""

import json
from stage2.utils.api import call_llm, parse_json_response
from stage2.utils.types import Question, SceneGraph

QUESTION_PROMPT = """You are a 3D scene layout validator. Given the scene graph and layout below, generate yes/no questions to validate a front-view layout diagram.

Scene graph:
- gravity: {gravity}
- scene_type: {scene_type}
- relations: {relations_json}

Layout (current positions):
{layout_lines}

{composition_check}

Generate 8-12 questions. Each question MUST be about:
1. LAYOUT: Is object X positioned correctly in the scene? (overlap, depth, placement)
2. GRAVITY/FEASIBILITY: Given gravity={gravity}, is object X correctly supported? Is it on the right surface (floor/wall/on:parent)? Does the placement make physical sense?

Do NOT ask about: theme, lighting, narrative, mood, or anything not visible in a 2D layout diagram.

Return ONLY a valid JSON object (no explanation, no markdown):
{{
  "questions": [
    {{"id": "Q1", "text": "<question>", "object_ids": ["<object_name>"]}},
    ...
  ]
}}

For each question, include object_ids: the object name(s) the question references.
"""


def generate_questions(
    scene_graph: SceneGraph,
    layout: dict,
) -> list[Question]:
    """
    Generate validation questions from scene graph + layout.
    Questions focus on layout (position) and gravity/feasibility (support).
    """
    relations_json = json.dumps(scene_graph.relations, indent=2)
    layout_lines = []
    for name, obj in layout.items():
        x = obj.get("x", 0)
        y = obj.get("y", 0)
        z = obj.get("z", 0)
        surface = obj.get("surface", "floor")
        layer = obj.get("layer", "foreground")
        layout_lines.append(f"  {name}: x={x:.2f}, y={y:.2f}, z={z:.2f}, surface={surface}, layer={layer}")
    layout_text = "\n".join(layout_lines)

    # X distribution check: flag if >50% of objects clustered near center
    total = len(layout)
    n_centered = sum(1 for obj in layout.values() if abs(obj.get("x", 0)) <= 0.3)
    centered_ratio = n_centered / total if total > 0 else 0
    composition_check = (
        f"COMPOSITION CHECK: {n_centered} of {total} objects have x in [-0.3, +0.3] ({centered_ratio:.0%}).\n"
        "If >50%, you MUST include a composition question: "
        '"Are too many objects clustered near the center (x ≈ 0)? Should some be redistributed left/right?"'
    )

    prompt = QUESTION_PROMPT.format(
        gravity=scene_graph.gravity,
        scene_type=scene_graph.scene_type,
        relations_json=relations_json,
        layout_lines=layout_text,
        composition_check=composition_check,
    )

    raw = call_llm(prompt, max_tokens=2000)
    parsed = parse_json_response(raw)

    questions = []
    for i, q in enumerate(parsed.get("questions", [])):
        qid = q.get("id", f"Q{i+1}")
        text = q.get("text", "")
        obj_ids = q.get("object_ids")
        if isinstance(obj_ids, list):
            obj_ids = [str(o) for o in obj_ids]
        else:
            obj_ids = None
        questions.append(Question(id=qid, text=text, object_ids=obj_ids))

    return questions
