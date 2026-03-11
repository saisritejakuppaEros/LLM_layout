"""
VLM Layout Validator.

Sends the rendered front-view diagram + scene context to the VLM.
Asks the full structured question bank.
Returns VLMResult with issues and composition score.
"""

import json
import sys
from stage2.utils.api import call_vlm, parse_json_response
from stage2.utils.types import VLMResult, VLMIssue

VLM_PROMPT = """You are a 3D scene layout quality inspector.

You are given a front-view projection diagram of a scene layout.
The scene theme is: "{theme}"
Gravity is: {gravity}

Inspect the diagram carefully and answer ALL of the following questions.
Return ONLY a valid JSON object — no explanation, no markdown.

Questions to evaluate:

GROUP 1 — Visibility & Occlusion:
- Q1: Which foreground objects are fully occluded (0% visible)?
- Q2: Which foreground objects are more than 70% occluded by another foreground object?
- Q3: Which midground objects are completely hidden behind foreground objects?
- Q4: Are any semantically critical objects (given the theme) not visible at all?

GROUP 2 — Physics / Gravity (only relevant if gravity=true):
- Q5: Which objects appear to be floating (no visible surface support)?
- Q6: Are any objects clipping into the floor (sunk below ground level)?
- Q7: Are any objects stacked implausibly (heavy object on unsupported edge)?

GROUP 3 — Collision & Spacing:
- Q8: Which pairs of objects appear to be physically intersecting?
- Q9: Are any objects too tightly clustered to be distinguishable?
- Q10: Is any object unreachably isolated from the rest of the scene?

GROUP 4 — Theme Coherence:
- Q11: Does the dominant light source (if any) match the theme's focal intent?
- Q12: Are background objects reading as background, or competing with foreground?
- Q13: Does the spatial arrangement support the scene narrative or contradict it?

GROUP 5 — Camera / Composition:
- Q14: Is the center of the frame occupied by something meaningful?
- Q15: Are foreground objects distributed across the full width or all clustered one side?
- Q16: Is the depth layering readable — can you distinguish fg/mid/bg as separate planes?

Return this exact JSON schema:
{{
  "issues": [
    {{
      "object": "<object name or 'scene'>",
      "group": "<occlusion | gravity | collision | theme | composition>",
      "severity": "<critical | warning | minor>",
      "description": "<one sentence>",
      "fix_hint": "<shift_lateral | snap_to_surface | push_apart | move_toward_center | increase_depth_separation | none>"
    }}
  ],
  "composition_score": <float 0.0 to 1.0>,
  "approved": <true | false>
}}

approved = true only if composition_score >= 0.75 AND no critical issues exist.
If there are no issues, return an empty issues list.
"""


def validate_with_vlm(
    diagram_path: str,
    scene_data: dict,
    gravity: bool,
) -> VLMResult:
    theme = scene_data.get("theme", "")
    prompt = VLM_PROMPT.format(theme=theme, gravity=gravity)

    for attempt in range(2):  # initial + 1 retry
        try:
            raw = call_vlm(diagram_path, prompt, max_tokens=1500)
            parsed = parse_json_response(raw)
            break
        except Exception as e:
            if attempt == 0:
                print(
                    f"[VLM] Attempt 1 failed: {e}. Retrying once...",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[VLM] Retry also failed: {e}",
                    file=sys.stderr,
                )
                return VLMResult(
                    issues=[],
                    composition_score=0.5,
                    approved=False,
                    parse_failed=True,
                )

    issues = [
        VLMIssue(
            object=i["object"],
            group=i["group"],
            severity=i["severity"],
            description=i["description"],
            fix_hint=i["fix_hint"],
        )
        for i in parsed.get("issues", [])
    ]

    return VLMResult(
        issues=issues,
        composition_score=float(parsed.get("composition_score", 0.0)),
        approved=bool(parsed.get("approved", False)),
    )
