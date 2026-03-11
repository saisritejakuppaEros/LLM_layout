#!/usr/bin/env python3
"""
Classify objects into foreground, midground, and background using LLM.
Theme from user prompt guides placement. Cross-checks that all three layers
exist; if any layer is empty, those items go to background.
"""

import json
import re
import requests
from pathlib import Path

TEXT_API = "http://localhost:8001/generate"

OBJECTS = [
    "chair", "table", "book", "cup", "bottle", "lamp", "clock",
    "window", "door", "bag", "box", "plant", "picture frame",
    "notebook", "phone"
]

# User prompt (theme) - can be overridden
THEME = "a horror scene with a terrifying red light."

OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "stage1" / "locations_prompts.json"


def build_llm_prompt(theme: str, objects: list[str]) -> str:
    """Build the prompt that feeds into the LLM for foreground/midground/background classification."""
    objects_str = ", ".join(objects)
    return f"""Given the scene theme: "{theme}"

Classify each of these objects into exactly one layer based on where they would naturally appear in such a scene:
- foreground: closest to viewer, most prominent
- midground: middle depth, supporting elements
- background: farthest, atmospheric/setting elements

Objects to classify: {objects_str}

Respond with ONLY valid JSON in this exact format (no other text):
{{"foreground": ["obj1", "obj2"], "midground": ["obj3"], "background": ["obj4", "obj5"]}}

Every object must appear in exactly one layer. Use the exact object names as given."""


def call_llm(prompt: str, max_tokens: int = 512) -> str:
    """Call the text LLM API."""
    resp = requests.post(
        TEXT_API,
        json={"prompt": prompt, "max_tokens": max_tokens},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def parse_llm_response(response: str) -> dict[str, list[str]]:
    """Extract JSON from LLM response (handles markdown code blocks, thinking blocks, extra text)."""
    text = response.strip()
    # Strip <think>...</think> blocks (Qwen thinking format)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Try to find JSON object - match balanced braces for nested structures
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue
    # Fallback: try parsing whole response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def cross_check_and_fix(
    parsed: dict[str, list[str]],
    all_objects: list[str],
) -> dict[str, list[str]]:
    """
    Cross-check: foreground, midground, background must all be non-empty.
    If any layer is missing or empty, put those items in background.
    """
    fg = parsed.get("foreground") or []
    mg = parsed.get("midground") or []
    bg = parsed.get("background") or []

    # Normalize to lists of strings
    fg = [str(x).strip().lower() for x in fg if x]
    mg = [str(x).strip().lower() for x in mg if x]
    bg = [str(x).strip().lower() for x in bg if x]

    all_lower = [o.lower() for o in all_objects]
    assigned = set(fg + mg + bg)

    # Add any missing objects to background
    for obj in all_lower:
        if obj not in assigned:
            bg.append(obj)

    # Cross-check: if any layer is empty, move items from others to fill it
    # Rule: keep them in background if not all three are present
    if not fg or not mg or not bg:
        # Not all layers present -> put everything in background
        bg = list(set(fg + mg + bg))
        fg = []
        mg = []

    return {
        "foreground": sorted(set(fg)),
        "midground": sorted(set(mg)),
        "background": sorted(set(bg)),
    }


def build_enhancer_prompt(
    theme: str,
    layers: dict[str, list[str]],
) -> str:
    """Build prompt for LLM to create a detailed scene description."""
    fg = ", ".join(layers.get("foreground", []))
    mg = ", ".join(layers.get("midground", []))
    bg = ", ".join(layers.get("background", []))
    return f"""Scene theme: "{theme}"

Object layout:
- Foreground (closest): {fg}
- Midground (middle): {mg}
- Background (farthest): {bg}

Write a single, detailed scene caption (2-4 sentences) that:
- Focus on the MAJOR objects that convey the story (not every object)
- Describe lighting, shadows, and atmosphere
- Mention key positions (e.g. chair in foreground, lamp casting red light)
- Match and intensify the theme's mood with vivid sensory details
- Capture the overall composition and story

Output only the caption, no preamble."""


def enhance_scene_prompt(theme: str, layers: dict[str, list[str]]) -> str:
    """Use LLM to generate a detailed scene description from theme + layout."""
    prompt = build_enhancer_prompt(theme, layers)
    response = call_llm(prompt, max_tokens=256)
    return response.strip().strip('"')


def get_the_grounds(
    theme: str | None = None,
    objects: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, list[str]]:
    """
    Main entry: use LLM to classify objects into foreground, midground, background.
    Theme guides placement. Cross-checks that all three layers exist; otherwise
    keeps items in background.
    """
    theme = theme or THEME
    objects = objects or OBJECTS

    prompt = build_llm_prompt(theme, objects)
    response = call_llm(prompt)
    if verbose:
        print("--- LLM raw response ---")
        print(response[:2000] + ("..." if len(response) > 2000 else ""))
        print("--- end ---")
    parsed = parse_llm_response(response)
    if verbose and parsed:
        print("--- Parsed ---")
        print(json.dumps(parsed, indent=2))
    return cross_check_and_fix(parsed, objects)


if __name__ == "__main__":
    import sys

    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--verbose", "-v")]
    theme = args[0] if args else THEME

    # 1. Get object layout (foreground, midground, background)
    layers = get_the_grounds(theme=theme, verbose=verbose)

    # 2. Generate enhanced scene prompt
    enhanced = enhance_scene_prompt(theme, layers)

    # 3. Print object locations and enhanced prompt
    print("=== Object locations ===")
    print(json.dumps(layers, indent=2))
    print()
    print("=== Enhanced scene prompt ===")
    print(enhanced)

    # 4. Log to outputs/stage1/locations_prompts.json
    output_data = {
        "theme": theme,
        "locations": layers,
        "enhanced_prompt": enhanced,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")
