#!/usr/bin/env python3
"""
Run Stage 1 and Stage 2 for all test scenarios.

Scenarios:
  - Basic: 3-6 objects — 10 indoor + 10 outdoor samples
  - Medium: 15-20 objects — 5 indoor + 5 outdoor samples
  - Hard: 30-40 objects — 5 indoor + 5 outdoor samples

Objects are picked randomly from indoor.json and outdoor.json.
Outputs saved under testing/outputs/<scenario>_<indoor|outdoor>_<idx>/
"""

import json
import os
import random
import sys
from pathlib import Path

# Add project root for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Must set STAGE2_OUT before importing stage2 (config reads it at import time)
OUTPUT_BASE = SCRIPT_DIR / "outputs"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
os.environ["STAGE2_OUT"] = str(OUTPUT_BASE)

from stage1.pipeline import get_the_grounds, enhance_scene_prompt
from stage2.pipeline import run_stage2

# Paths to object lists
INDOOR_JSON = SCRIPT_DIR / "stage1_2" / "indoor.json"
OUTDOOR_JSON = SCRIPT_DIR / "stage1_2" / "outdoor.json"

# Default theme (can override per sample)
DEFAULT_THEME = "a cozy living room scene with warm lighting."

# Scenario config: (min_objects, max_objects, num_indoor_samples, num_outdoor_samples)
SCENARIOS = {
    "basic": (3, 6, 10, 10),
    "medium": (15, 20, 5, 5),
    "hard": (30, 40, 5, 5),
}


def load_objects() -> tuple[list[str], list[str]]:
    """Load indoor and outdoor object lists from JSON files."""
    with open(INDOOR_JSON) as f:
        indoor = json.load(f)["indoor_objects"]
    with open(OUTDOOR_JSON) as f:
        outdoor = json.load(f)["outdoor_objects"]
    return indoor, outdoor


def sample_objects(
    pool: list[str],
    min_n: int,
    max_n: int,
) -> list[str]:
    """Randomly sample between min_n and max_n objects from pool."""
    n = random.randint(min_n, max_n)
    n = min(n, len(pool))
    return random.sample(pool, n)


def run_stage1(theme: str, objects: list[str], verbose: bool = False) -> dict:
    """Run stage 1: classify objects into layers and enhance prompt."""
    layers = get_the_grounds(theme=theme, objects=objects, verbose=verbose)
    enhanced = enhance_scene_prompt(theme, layers)
    return {
        "theme": theme,
        "locations": layers,
        "enhanced_prompt": enhanced,
    }


def process_sample(
    scene_id: str,
    objects: list[str],
    theme: str = DEFAULT_THEME,
    verbose: bool = False,
) -> dict | None:
    """Run stage 1 + stage 2 for one sample. Returns final layout or None on error."""
    out_dir = OUTPUT_BASE / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Stage 1
        print(f"\n[{scene_id}] Stage 1 — {len(objects)} objects...")
        scene_data = run_stage1(theme, objects, verbose=verbose)

        # Save stage 1 output
        stage1_path = out_dir / "stage1_locations_prompts.json"
        with open(stage1_path, "w") as f:
            json.dump(scene_data, f, indent=2)
        print(f"[{scene_id}] Stage 1 done → {stage1_path}")

        # Stage 2
        print(f"[{scene_id}] Stage 2...")
        layout = run_stage2(scene_data, scene_id)
        print(f"[{scene_id}] Stage 2 done")
        return layout

    except Exception as e:
        print(f"[{scene_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Stage 1 + Stage 2 for all scenarios")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default=None,
                        help="Run only this scenario (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose stage 1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    indoor_objs, outdoor_objs = load_objects()

    scenarios_to_run = [args.scenario] if args.scenario else list(SCENARIOS)
    results = []

    for scenario_name in scenarios_to_run:
        min_n, max_n, n_indoor, n_outdoor = SCENARIOS[scenario_name]

        for i in range(n_indoor):
            scene_id = f"{scenario_name}_indoor_{i:02d}"
            objects = sample_objects(indoor_objs, min_n, max_n)
            layout = process_sample(scene_id, objects, verbose=args.verbose)
            results.append((scene_id, layout is not None))

        for i in range(n_outdoor):
            scene_id = f"{scenario_name}_outdoor_{i:02d}"
            objects = sample_objects(outdoor_objs, min_n, max_n)
            layout = process_sample(scene_id, objects, verbose=args.verbose)
            results.append((scene_id, layout is not None))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ok = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Completed: {ok}/{total}")
    for scene_id, success in results:
        status = "OK" if success else "FAILED"
        print(f"  {scene_id}: {status}")
    print(f"\nOutputs saved under: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
