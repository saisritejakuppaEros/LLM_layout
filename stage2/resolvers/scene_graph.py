"""
LLM Scene Graph Builder.

Call A: Semantic only (no size_meters) — reduces LLM task load.
Call B: Size resolution via OBJECT_SIZE_PRIORS + clamp + LLM fallback.
"""

import json
from stage2.utils.api import call_llm, parse_json_response
from stage2.utils.types import SceneGraph
from stage2.data.size_priors import resolve_sizes


SCENE_GRAPH_PROMPT = """You are a 3D scene layout expert.
Given a scene description with objects grouped by layer (foreground/midground/background),
produce a structured scene graph with spatial semantics ONLY. Do NOT estimate sizes.

Scene JSON:
{scene_json}

Respond ONLY with a valid JSON object (no explanation, no markdown) following this exact schema:

{{
  "scene_type": "<indoor_room | outdoor | space | underwater | abstract>",
  "gravity": <true | false>,
  "camera_facing": "front",
  "relations": [
    {{
      "object": "<object name>",
      "layer": "<foreground | midground | background>",
      "surface": "<floor | wall | ceiling | floating | on:<object_name>>",
      "wall": "<back | back-left | back-right | left | right | null>",
      "anchor": "<center | center-left | center-right | left | right | corner-left | corner-right | high | low | near:<object_name> | on:<object_name>>",
      "facing": "<camera | left | right | null>",
      "supports": [<list of object names this object supports, or empty list>],
      "semantic_importance": "<primary | secondary | tertiary>"
    }}
  ]
}}

Rules:
- Objects with surface "wall" must have a wall direction.
- Objects on surfaces (e.g. cup on table) must have surface "on:table".
- supports lists which objects rest ON this object.
- semantic_importance: primary = mentioned first/prominently in theme, tertiary = background filler.
- If the scene theme suggests floating/zero-gravity, set gravity to false.
- Every object in the input must appear in relations exactly once.
- Do NOT include size_meters — sizes are resolved separately.
"""


def build_scene_graph(scene_data: dict) -> SceneGraph:
    prompt = SCENE_GRAPH_PROMPT.format(scene_json=json.dumps(scene_data, indent=2))
    raw = call_llm(prompt, max_tokens=3000)
    parsed = parse_json_response(raw)

    relations = parsed["relations"]

    # Call B: Resolve sizes from priors + clamp + LLM fallback
    relations = resolve_sizes(relations)

    return SceneGraph(
        scene_type=parsed["scene_type"],
        gravity=parsed["gravity"],
        camera_facing=parsed["camera_facing"],
        relations=relations,
    )
