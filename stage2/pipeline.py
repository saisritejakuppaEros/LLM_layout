"""
Stage 2: Spatial Layout Resolution Pipeline.

Input:  stage1 JSON  (theme, locations fg/mid/bg, enhanced_prompt)
Output: per-object (x, y, z, rot_y, size) + final diagram PNG

Flow:
  1. LLM  → Scene Graph  (semantic relations + size estimates)
  2. LLM  → Parent existence check  (validate on:<x> refs exist; demote to floor if not)
  3. Algo → Layer correction  (surface=wall + back → layer=background)
  4. Algo → Position Resolver  (depth bands + anchor → XYZ)
  5. Algo → Collision Resolver  (DIoU3D push-apart)
  6. Algo → Bounds clamp  (x within ±SCENE_WIDTH/2, y+h/2 ≤ CEILING_Y)
  7. Algo → Occlusion Resolver  (front-view visibility budget)
  8. Pass 1: Text LLM → generate scene-specific questions
  9. Loop (max 5): Render → Pass 2 VLM (evaluate Qs) → if approved break
     else Pass 3 VLM (deltas) → apply_deltas → collision → bounds → occlusion
 10. Save final layout JSON + diagram
"""

import json
import os
import sys
from dataclasses import asdict

from stage2.config import MAX_VLM_ITERATIONS, OUTPUT_DIR
from stage2.resolvers.scene_graph    import build_scene_graph
from stage2.validators.parent_existence import validate_parent_existence
from stage2.validators.layer_correction import correct_layer_for_wall
from stage2.resolvers.position_resolver import resolve_positions
from stage2.resolvers.collision      import resolve_collisions
from stage2.resolvers.bounds         import clamp_bounds
from stage2.resolvers.occlusion      import resolve_occlusions
from stage2.utils.diagram            import render_front_view, compute_bboxes_from_placements
from stage2.validators.question_generator import generate_questions
from stage2.validators.question_evaluator import evaluate_questions
from stage2.validators.delta_generator    import generate_deltas
from stage2.validators.delta_applicator   import apply_deltas


def _extract_priorities(graph) -> dict[str, str]:
    return {r["object"]: r.get("semantic_importance", "tertiary") for r in graph.relations}


def _placements_to_dict(placements) -> dict:
    out = {}
    for name, p in placements.items():
        out[name] = {
            "x": round(p.x, 4),
            "y": round(p.y, 4),
            "z": round(p.z, 4),
            "rot_y": round(p.rot_y, 2),
            "layer": p.layer,
            "surface": p.surface,
            "floating": p.floating,
            "size": asdict(p.size) if p.size else None,
        }
    return out


def _write_debug_log(path: str, lines: list[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(lines))


def run_stage2(scene_data: dict, scene_id: str) -> dict:
    scene_out_dir = os.path.join(OUTPUT_DIR, scene_id)
    iterations_dir = os.path.join(scene_out_dir, "iterations")
    os.makedirs(scene_out_dir, exist_ok=True)
    os.makedirs(iterations_dir, exist_ok=True)

    run_log: list[str] = []

    def _log(msg: str) -> None:
        run_log.append(msg)
        print(msg)

    # ── Step 1: LLM → Scene Graph ─────────────────────────────────────────
    _log(f"[{scene_id}] Building scene graph via LLM...")
    graph = build_scene_graph(scene_data)

    # Parent existence check: validate every on:<x> reference exists; demote to floor if not
    graph = validate_parent_existence(graph)

    # Layer correction: surface=wall + wall contains "back" → layer=background
    graph = correct_layer_for_wall(graph)

    with open(os.path.join(scene_out_dir, "scene_graph.json"), "w") as f:
        json.dump({"scene_type": graph.scene_type, "gravity": graph.gravity,
                   "relations": graph.relations}, f, indent=2)

    priorities = _extract_priorities(graph)

    # ── Step 2: Position Resolver ─────────────────────────────────────────
    _log(f"[{scene_id}] Resolving positions...")
    placements = resolve_positions(graph)

    # ── Step 3: Collision Resolver ────────────────────────────────────────
    _log(f"[{scene_id}] Resolving collisions (DIoU3D)...")
    placements = resolve_collisions(placements, priorities)

    # Bounds clamp: (x ± w/2) within ±SCENE_WIDTH/2, (y + h/2) within CEILING_Y
    placements = clamp_bounds(placements)

    # ── Step 4: Occlusion Resolver ────────────────────────────────────────
    _log(f"[{scene_id}] Resolving occlusions...")
    placements = resolve_occlusions(placements)

    # ── Pass 1: Generate scene-specific questions (scene graph + layout) ───
    layout_dict = _placements_to_dict(placements)
    _log(f"[{scene_id}] Pass 1: Generating questions via Text LLM (scene graph + layout)...")
    questions = generate_questions(scene_graph=graph, layout=layout_dict)

    with open(os.path.join(scene_out_dir, "questions.json"), "w") as f:
        json.dump([
            {"id": q.id, "text": q.text, "object_ids": q.object_ids}
            for q in questions
        ], f, indent=2)

    questions_txt_lines = [
        f"=== Validation Questions (from scene graph) ===",
        f"Scene: {scene_id} | Gravity: {graph.gravity} | Objects: {len(placements)}",
        "",
    ]
    for q in questions:
        obj_ids = f"  object_ids: {q.object_ids}" if q.object_ids else ""
        questions_txt_lines.append(f"{q.id}: {q.text}")
        if obj_ids:
            questions_txt_lines.append(obj_ids)
        questions_txt_lines.append("")
    _write_debug_log(os.path.join(scene_out_dir, "questions.txt"), questions_txt_lines)
    _log(f"[{scene_id}]   Generated {len(questions)} questions → questions.json, questions.txt")

    # ── Q-based validation loop (max 5 iterations) ─────────────────────────
    q_result = None
    for iteration in range(MAX_VLM_ITERATIONS):
        diagram_path = os.path.join(iterations_dir, f"diagram_iter{iteration}.png")
        render_front_view(placements, scene_data, diagram_path)

        object_bboxes_px = compute_bboxes_from_placements(placements, diagram_path)

        debug_lines = [
            f"=== Iteration {iteration + 1}/{MAX_VLM_ITERATIONS} ===",
            "",
            f"Diagram: {diagram_path}",
            f"Placements: {len(placements)} objects",
            "",
            "--- Questions ---",
        ]
        for q in questions:
            debug_lines.append(f"  {q.id}: {q.text}")
        debug_lines.append("")

        _log(f"[{scene_id}] Pass 2: Evaluating questions — iteration {iteration + 1}/{MAX_VLM_ITERATIONS}...")
        q_result = evaluate_questions(diagram_path, questions)

        debug_lines.extend([
            "--- Q Validation Result ---",
            f"approved: {q_result.approved}",
            f"parse_failed: {q_result.parse_failed}",
            f"failed: {q_result.failed_question_ids}",
            "",
        ])

        if q_result.results:
            debug_lines.append("--- Results ---")
            for r in q_result.results:
                status = "PASS" if r.passed else "FAIL"
                debug_lines.append(f"  {r.question_id} [{status}]: {r.observation}")
            debug_lines.append("")

        with open(os.path.join(iterations_dir, f"question_result_iter{iteration}.json"), "w") as f:
            json.dump({
                "approved": q_result.approved,
                "parse_failed": q_result.parse_failed,
                "failed_question_ids": q_result.failed_question_ids,
                "results": [
                    {"question_id": r.question_id, "passed": r.passed, "observation": r.observation}
                    for r in q_result.results
                ],
            }, f, indent=2)

        _log(f"[{scene_id}]   approved={q_result.approved}  failed={len(q_result.failed_question_ids)}")

        if q_result.approved:
            debug_lines.extend([
                "--- Decision ---",
                "APPROVED: All questions passed",
            ])
            _write_debug_log(os.path.join(iterations_dir, f"debug_iter{iteration}.txt"), debug_lines)
            break

        if q_result.parse_failed:
            debug_lines.extend([
                "--- Decision ---",
                "Parse failed. Continuing to next iteration to retry.",
            ])
            _write_debug_log(os.path.join(iterations_dir, f"debug_iter{iteration}.txt"), debug_lines)
            continue

        # Pass 3: Generate deltas for failed questions (only those with object_ids)
        failed_questions = [q for q in questions if q.id in q_result.failed_question_ids]
        failed_with_objects = [q for q in failed_questions if q.object_ids]
        debug_lines.append("--- Failed Questions ---")
        for q in failed_questions:
            debug_lines.append(f"  {q.id}: {q.text}")
        debug_lines.append("")

        _log(f"[{scene_id}] Pass 3: Generating deltas for {len(failed_questions)} failed questions...")
        delta_result = generate_deltas(
            diagram_path,
            failed_with_objects if failed_with_objects else failed_questions,
            object_bboxes_px,
            placements,
        )

        with open(os.path.join(iterations_dir, f"delta_result_iter{iteration}.json"), "w") as f:
            json.dump({"deltas": {k: {"dx": v[0], "dy": v[1], "dz": v[2]} for k, v in delta_result.deltas.items()}}, f, indent=2)

        debug_lines.append("--- Deltas Applied ---")
        for obj_id, (dx, dy, dz) in delta_result.deltas.items():
            debug_lines.append(f"  {obj_id}: dx={dx:.4f} dy={dy:.4f} dz={dz:.4f}")
        debug_lines.append("")

        placements_before = _placements_to_dict(placements)
        placements = apply_deltas(placements, delta_result.deltas)
        placements = resolve_collisions(placements, priorities)
        placements = clamp_bounds(placements)
        placements = resolve_occlusions(placements)
        placements_after = _placements_to_dict(placements)

        debug_lines.extend([
            "--- Decision ---",
            "Not approved. Applying deltas, then collision + bounds + occlusion.",
            "",
            "--- Placements changed ---",
        ])
        for name in placements:
            before = placements_before.get(name, {})
            after = placements_after.get(name, {})
            if before != after:
                debug_lines.append(f"  {name}:")
                debug_lines.append(f"    before: x={before.get('x')} y={before.get('y')} z={before.get('z')}")
                debug_lines.append(f"    after:  x={after.get('x')} y={after.get('y')} z={after.get('z')}")
        debug_lines.append("")
        debug_lines.append(f"Next: iteration {iteration + 2}")

        _write_debug_log(os.path.join(iterations_dir, f"debug_iter{iteration}.txt"), debug_lines)

    # ── Save final outputs ─────────────────────────────────────────────────
    _write_debug_log(os.path.join(scene_out_dir, "run_log.txt"), run_log)

    final_layout = _placements_to_dict(placements)
    final_layout_path = os.path.join(scene_out_dir, "layout.json")

    # Run summary for debugging
    last_iter = iteration if q_result is not None else -1
    summary_lines = [
        f"Scene: {scene_id}",
        f"Total iterations: {last_iter + 1}",
        f"Q-based validation approved: {q_result.approved if q_result else False}",
        f"Iterations dir: {iterations_dir}",
        "",
        "Files: questions.json, questions.txt, run_log.txt, layout.json, diagram_final.png",
        "Per-iteration: diagram_iter{N}.png, debug_iter{N}.txt, question_result_iter{N}.json, delta_result_iter{N}.json",
    ]
    _write_debug_log(os.path.join(scene_out_dir, "debug_run_summary.txt"), summary_lines)
    with open(final_layout_path, "w") as f:
        json.dump({
            "scene_id":    scene_id,
            "theme":       scene_data.get("theme"),
            "gravity":     graph.gravity,
            "scene_type":  graph.scene_type,
            "layout":      final_layout,
            "q_approved":  q_result.approved if q_result else False,
        }, f, indent=2)

    # final diagram
    final_diagram = os.path.join(scene_out_dir, "diagram_final.png")
    render_front_view(placements, scene_data, final_diagram)

    _log(f"[{scene_id}] Done → {final_layout_path}")
    return final_layout


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m stage2.pipeline <path_to_locations_prompts.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    with open(input_path) as f:
        data = json.load(f)

    # Handle both single scene dict and list of scenes
    scenes = data if isinstance(data, list) else [data]

    for idx, scene in enumerate(scenes):
        scene_id = scene.get("scene_id", f"scene_{idx:04d}")
        run_stage2(scene, scene_id)


if __name__ == "__main__":
    main()
