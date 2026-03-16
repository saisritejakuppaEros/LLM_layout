"""
Pass 2 — Question Evaluator.

VLM evaluates each question on the diagram.
Answer based ONLY on what is visible: layout, physics, placement.
"""

import json
import sys
from stage2.utils.api import call_vlm, parse_json_response
from stage2.utils.diagram import get_name_to_box_id
from stage2.utils.types import Question, QuestionResult, QValidationResult

EVALUATE_PROMPT_SINGLE = """You are a 3D scene layout quality inspector.

Look at this front-view diagram of a scene layout. Answer based ONLY on what you see.
This is a 2D layout diagram (front view). Do NOT judge theme, lighting, or narrative.
Focus on: physics (support, floating), placement (overlap, position).

Diagram format: Each object has a number in its box. The legend at the bottom maps numbers to object names (e.g. "1: chair | 2: cup"). Brighter boxes with thicker borders are foreground; dimmer ones are background.

For each question, respond with "passed" (true/false) and a brief "observation".
passed=true only if the layout clearly satisfies the question. When in doubt, passed=false.

Questions:
{questions_text}

Return ONLY a valid JSON object (no explanation, no markdown):
{{
  "results": [
    {{"question_id": "Q1", "passed": true, "observation": "<brief observation>"}},
    ...
  ]
}}
"""

EVALUATE_PROMPT_DUAL = """You are a 3D scene layout quality inspector.

You see two views of the same scene layout:
(1) Front view: XY projection (what the camera sees)
(2) Bird's-eye view: XZ projection showing depth (camera at top, back wall at bottom)

Use BOTH views to answer. The bird's-eye view shows depth — objects at the same Z are at the same depth. A cup "on" a table should have the same Z as the table in the bird's-eye view. Do NOT judge theme, lighting, or narrative. Focus on: physics (support, floating), placement (overlap, position, depth).

Diagram format: Each object has a number in its box. The legend maps numbers to object names.

For each question, respond with "passed" (true/false) and a brief "observation".
passed=true only if the layout clearly satisfies the question. When in doubt, passed=false.

Questions:
{questions_text}

Return ONLY a valid JSON object (no explanation, no markdown):
{{
  "results": [
    {{"question_id": "Q1", "passed": true, "observation": "<brief observation>"}},
    ...
  ]
}}
"""


def evaluate_questions(
    diagram_path: str,
    questions: list[Question],
    placements: dict | None = None,
    birds_eye_path: str | None = None,
) -> QValidationResult:
    """
    VLM evaluates each question on the diagram(s).
    If birds_eye_path is provided, pass both views for depth-aware validation.
    Returns QValidationResult with results and failed_question_ids.
    """
    name_to_id = get_name_to_box_id(placements) if placements else {}

    def _get_box_num(obj_name: str) -> str | None:
        return str(name_to_id[obj_name]) if obj_name in name_to_id else None

    questions_text = "\n".join(
        f"- {q.id} [box #{_get_box_num(q.object_ids[0])}]: {q.text}"
        if q.object_ids and _get_box_num(q.object_ids[0])
        else f"- {q.id}: {q.text}"
        for q in questions
    )

    if birds_eye_path:
        prompt = EVALUATE_PROMPT_DUAL.format(questions_text=questions_text)
    else:
        prompt = EVALUATE_PROMPT_SINGLE.format(questions_text=questions_text)

    for attempt in range(2):
        try:
            if birds_eye_path:
                raw = call_vlm(image_paths=[diagram_path, birds_eye_path], prompt=prompt, max_tokens=1500)
            else:
                raw = call_vlm(image_path=diagram_path, prompt=prompt, max_tokens=1500)
            parsed = parse_json_response(raw)
            break
        except Exception as e:
            if attempt == 0:
                print(f"[Q-eval] Attempt 1 failed: {e}. Retrying...", file=sys.stderr)
            else:
                print(f"[Q-eval] Retry failed: {e}", file=sys.stderr)
                return QValidationResult(
                    results=[],
                    approved=False,
                    failed_question_ids=[q.id for q in questions],
                    parse_failed=True,
                )

    results = []
    for r in parsed.get("results", []):
        qid = r.get("question_id", "")
        passed = bool(r.get("passed", False))
        obs = str(r.get("observation", ""))
        results.append(QuestionResult(question_id=qid, passed=passed, observation=obs))

    answered_ids = {r.question_id for r in results}
    answer_rate = len(answered_ids) / len(questions) if questions else 0
    if answer_rate < 0.8:
        return QValidationResult(
            results=results,
            approved=False,
            failed_question_ids=[q.id for q in questions],
            parse_failed=True,
        )

    failed_ids = [r.question_id for r in results if not r.passed]
    for q in questions:
        if q.id not in answered_ids:
            failed_ids.append(q.id)

    return QValidationResult(
        results=results,
        approved=len(failed_ids) == 0,
        failed_question_ids=failed_ids,
    )
