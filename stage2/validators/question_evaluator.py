"""
Pass 2 — Question Evaluator.

VLM evaluates each question on the diagram.
Answer based ONLY on what is visible: layout, physics, placement.
"""

import json
import sys
from stage2.utils.api import call_vlm, parse_json_response
from stage2.utils.types import Question, QuestionResult, QValidationResult

EVALUATE_PROMPT = """You are a 3D scene layout quality inspector.

Look at this front-view diagram of a scene layout. Answer based ONLY on what you see.
This is a 2D layout diagram (front view). Do NOT judge theme, lighting, or narrative.
Focus on: physics (support, floating), placement (overlap, position).

Diagram format: Each object has a number in its box. The legend at the bottom maps numbers to object names (e.g. "1: chair | 2: cup"). Brighter boxes with thicker borders are foreground; dimmer ones are background.

For each question, respond with "passed" (true/false) and a brief "observation".
passed=true if the layout satisfies the question.
passed=false only if there is a clear fixable issue (e.g. object floating, overlapping, wrong surface).

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
) -> QValidationResult:
    """
    VLM evaluates each question on the diagram.
    Returns QValidationResult with results and failed_question_ids.
    """
    questions_text = "\n".join(
        f"- {q.id}: {q.text}" for q in questions
    )
    prompt = EVALUATE_PROMPT.format(questions_text=questions_text)

    for attempt in range(2):
        try:
            raw = call_vlm(diagram_path, prompt, max_tokens=1500)
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

    failed_ids = [r.question_id for r in results if not r.passed]
    answered_ids = {r.question_id for r in results}
    for q in questions:
        if q.id not in answered_ids:
            failed_ids.append(q.id)

    return QValidationResult(
        results=results,
        approved=len(failed_ids) == 0,
        failed_question_ids=failed_ids,
    )
