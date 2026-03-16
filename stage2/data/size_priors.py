"""
Object size priors and resolution.

Provides OBJECT_SIZE_PRIORS for known object categories, fuzzy matching,
clamping by category, and LLM fallback for unknown objects.
"""

from stage2.utils.types import ObjectSize
from stage2.utils.api import call_llm, parse_json_response
import json

# Canonical sizes (w, h, d) in meters for common object categories
# Format: normalized key -> (w, h, d)
OBJECT_SIZE_PRIORS: dict[str, tuple[float, float, float]] = {
    # Furniture
    "chair": (0.5, 0.9, 0.5),
    "table": (1.2, 0.75, 0.8),
    "coffee_table": (1.0, 0.45, 0.6),
    "dining_table": (1.6, 0.75, 1.0),
    "side_table": (0.5, 0.55, 0.4),
    "sofa": (2.0, 0.9, 0.9),
    "armchair": (0.9, 0.9, 0.9),
    "bed": (2.0, 0.5, 1.2),
    "mattress": (2.0, 0.2, 1.2),
    "stool": (0.4, 0.45, 0.4),
    "bench": (1.2, 0.5, 0.5),
    "desk": (1.2, 0.75, 0.8),
    "bookshelf": (1.0, 0.2, 0.4),
    "cabinet": (1.0, 1.8, 0.5),
    "wardrobe": (1.2, 2.0, 0.6),
    "nightstand": (0.5, 0.55, 0.4),
    "tv_stand": (1.2, 0.6, 0.5),
    "drawer": (0.6, 0.5, 0.5),
    # Lighting
    "lamp": (0.3, 0.4, 0.3),
    "floor_lamp": (0.4, 1.6, 0.4),
    "table_lamp": (0.25, 0.35, 0.25),
    "ceiling_light": (0.5, 0.2, 0.5),
    "wall_light": (0.3, 0.4, 0.2),
    "chandelier": (0.8, 0.6, 0.8),
    "street_lamp": (0.3, 3.0, 0.3),
    # Small objects
    "cup": (0.08, 0.1, 0.08),
    "coffee_mug": (0.09, 0.11, 0.09),
    "glass": (0.07, 0.15, 0.07),
    "bottle": (0.08, 0.25, 0.08),
    "water_bottle": (0.08, 0.28, 0.08),
    "book": (0.22, 0.03, 0.28),
    "books": (0.25, 0.15, 0.3),
    "notebook": (0.22, 0.02, 0.28),
    "magazine": (0.28, 0.02, 0.22),
    "phone": (0.08, 0.16, 0.01),
    "remote_control": (0.15, 0.04, 0.04),
    "clock": (0.3, 0.3, 0.05),
    "wall_clock": (0.35, 0.35, 0.05),
    "watch": (0.04, 0.01, 0.04),
    "vase": (0.15, 0.25, 0.15),
    "candle": (0.05, 0.15, 0.05),
    "candle_holder": (0.08, 0.12, 0.08),
    "plate": (0.28, 0.03, 0.28),
    "bowl": (0.2, 0.08, 0.2),
    "pen": (0.15, 0.01, 0.01),
    "keys": (0.08, 0.02, 0.02),
    "wallet": (0.1, 0.01, 0.07),
    "ashtray": (0.12, 0.08, 0.12),
    # Wall / decor
    "picture": (0.5, 0.4, 0.03),
    "picture_frame": (0.5, 0.4, 0.03),
    "painting": (0.6, 0.5, 0.03),
    "poster": (0.6, 0.9, 0.01),
    "mirror": (0.5, 0.8, 0.05),
    "window": (1.2, 1.2, 0.1),
    "door": (0.9, 2.1, 0.05),
    "curtains": (2.0, 2.5, 0.05),
    "window_blinds": (1.2, 1.2, 0.05),
    # Floor / large
    "carpet": (2.5, 0.02, 0.02),
    "rug": (1.8, 0.02, 0.02),
    "doormat": (0.6, 0.02, 0.4),
    # Electronics
    "television": (1.2, 0.7, 0.08),
    "tv": (1.2, 0.7, 0.08),
    "laptop": (0.35, 0.02, 0.25),
    "monitor": (0.5, 0.35, 0.05),
    "keyboard": (0.45, 0.02, 0.15),
    "mouse": (0.1, 0.04, 0.06),
    "printer": (0.4, 0.3, 0.35),
    "speaker": (0.2, 0.3, 0.2),
    "game_console": (0.3, 0.06, 0.25),
    "camera": (0.15, 0.1, 0.08),
    "microphone": (0.05, 0.2, 0.05),
    # Plants / containers
    "plant": (0.3, 0.5, 0.3),
    "indoor_plant": (0.3, 0.5, 0.3),
    "flower_pot": (0.2, 0.25, 0.2),
    "bag": (0.4, 0.45, 0.15),
    "backpack": (0.5, 0.55, 0.2),
    "suitcase": (0.6, 0.4, 0.3),
    "box": (0.3, 0.2, 0.3),
    "trash_bin": (0.35, 0.5, 0.35),
    "recycling_bin": (0.4, 0.6, 0.4),
    "laundry_basket": (0.5, 0.6, 0.4),
    # Kitchen / appliances
    "teapot": (0.2, 0.15, 0.15),
    "kettle": (0.2, 0.2, 0.2),
    "microwave": (0.5, 0.35, 0.45),
    "oven": (0.6, 0.6, 0.6),
    "stove": (0.6, 0.4, 0.6),
    "refrigerator": (1.0, 1.8, 0.8),
    "sink": (0.6, 0.5, 0.5),
    "dish_rack": (0.5, 0.4, 0.35),
    "cutlery": (0.25, 0.02, 0.02),
    # Outdoor
    "tree": (0.8, 4.0, 0.8),
    "bush": (0.8, 0.6, 0.8),
    "rock": (0.5, 0.3, 0.5),
    "boulder": (1.5, 1.0, 1.5),
    "fence": (4.0, 1.2, 0.1),
    "gate": (1.2, 1.2, 0.05),
    "mailbox": (0.3, 0.2, 0.3),
    "traffic_light": (0.5, 3.0, 0.3),
    "traffic_sign": (0.5, 1.5, 0.05),
    "stop_sign": (0.6, 0.6, 0.05),
    "barrel": (0.5, 0.7, 0.5),
    "traffic_cone": (0.3, 0.5, 0.3),
    "sculpture": (0.5, 0.8, 0.5),
    "decorative_sculpture": (0.4, 0.6, 0.4),
    "umbrella": (0.5, 0.1, 0.5),
    "coat_rack": (0.5, 1.5, 0.5),
    "clothes_hanger": (0.2, 0.08, 0.2),
    "charger": (0.08, 0.03, 0.05),
    "extension_cord": (0.2, 0.02, 0.02),
    "tripod": (0.1, 1.5, 0.1),
    "pillow": (0.5, 0.15, 0.5),
    "blanket": (1.5, 0.02, 1.2),
    "bedsheet": (2.0, 0.01, 1.5),
}

# Category -> (min_w, max_w, min_h, max_h, min_d, max_d) for clamping
SIZE_CLAMP_BY_CATEGORY: dict[str, tuple[float, float, float, float, float, float]] = {
    "cup": (0.05, 0.15, 0.06, 0.18, 0.05, 0.15),
    "coffee_mug": (0.06, 0.12, 0.08, 0.15, 0.06, 0.12),
    "bottle": (0.05, 0.15, 0.15, 0.5, 0.05, 0.15),
    "book": (0.1, 0.4, 0.01, 0.04, 0.15, 0.4),
    "phone": (0.06, 0.12, 0.12, 0.2, 0.005, 0.02),
    "carpet": (1.0, 5.0, 0.01, 0.05, 1.0, 5.0),  # carpet: w,d large, h tiny
    "rug": (0.8, 4.0, 0.01, 0.05, 0.8, 4.0),
    "lamp": (0.15, 0.5, 0.2, 0.6, 0.15, 0.5),
    "floor_lamp": (0.2, 0.6, 0.8, 2.5, 0.2, 0.6),
    "chair": (0.35, 0.8, 0.6, 1.2, 0.35, 0.8),
    "table": (0.6, 2.5, 0.4, 1.0, 0.5, 2.0),
    "sofa": (1.2, 3.0, 0.6, 1.2, 0.6, 1.5),
    "plant": (0.15, 0.8, 0.2, 1.5, 0.15, 0.8),
    "television": (0.5, 2.0, 0.3, 1.2, 0.03, 0.2),
    "tv": (0.5, 2.0, 0.3, 1.2, 0.03, 0.2),
    "window": (0.6, 2.5, 0.4, 2.5, 0.05, 0.15),
    "door": (0.7, 1.2, 1.8, 2.5, 0.03, 0.1),
}

# Fuzzy mapping: object name substring -> canonical key (longer matches first)
FUZZY_MATCH: dict[str, str] = {
    "coffee table": "coffee_table",
    "dining table": "dining_table",
    "side table": "side_table",
    "floor lamp": "floor_lamp",
    "table lamp": "table_lamp",
    "wall lamp": "wall_light",
    "coffee mug": "coffee_mug",
    "water bottle": "water_bottle",
    "picture frame": "picture_frame",
    "wall clock": "wall_clock",
    "indoor plant": "indoor_plant",
    "flower pot": "flower_pot",
    "tv stand": "tv_stand",
    "game console": "game_console",
    "television": "television",
}


def _normalize_name(name: str) -> str:
    """Normalize object name for lookup."""
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def _lookup_prior(obj_name: str) -> tuple[float, float, float] | None:
    """Look up size prior from OBJECT_SIZE_PRIORS with fuzzy matching."""
    norm = _normalize_name(obj_name)
    if norm in OBJECT_SIZE_PRIORS:
        return OBJECT_SIZE_PRIORS[norm]

    # Try fuzzy match first
    for substr, canonical in FUZZY_MATCH.items():
        if substr in obj_name.lower():
            return OBJECT_SIZE_PRIORS.get(canonical)

    # Try prefix match (e.g. "chair" matches "armchair" -> no, but "chair" matches "chair")
    for key, size in OBJECT_SIZE_PRIORS.items():
        if key in norm or norm in key:
            return size

    return None


def get_clamp_bounds(obj_name: str) -> tuple[float, float, float, float, float, float] | None:
    """Get clamp bounds for object category if defined."""
    norm = _normalize_name(obj_name)
    if norm in SIZE_CLAMP_BY_CATEGORY:
        return SIZE_CLAMP_BY_CATEGORY[norm]

    for substr, canonical in FUZZY_MATCH.items():
        if substr in obj_name.lower() and canonical in SIZE_CLAMP_BY_CATEGORY:
            return SIZE_CLAMP_BY_CATEGORY[canonical]

    return None


def clamp_size(obj_name: str, size: ObjectSize) -> ObjectSize:
    """Clamp size to category-specific bounds if defined."""
    bounds = get_clamp_bounds(obj_name)
    if not bounds:
        # Default bounds to prevent extreme values
        min_w, max_w = 0.05, 5.0
        min_h, max_h = 0.01, 5.0
        min_d, max_d = 0.05, 5.0
    else:
        min_w, max_w, min_h, max_h, min_d, max_d = bounds

    return ObjectSize(
        w=max(min_w, min(max_w, size.w)),
        h=max(min_h, min(max_h, size.h)),
        d=max(min_d, min(max_d, size.d)),
    )


SIZE_ESTIMATE_PROMPT = """Estimate real-world size in meters (width, height, depth) for these objects.
Return ONLY valid JSON (no explanation):
{{"sizes": {{"<object_name>": {{"w": <float>, "h": <float>, "d": <float>}}, ...}}}}

Objects: {objects}

Be realistic. Examples: cup ~0.08x0.1x0.08, chair ~0.5x0.9x0.5, table ~1.2x0.75x0.8."""


def _llm_estimate_sizes(unknown_objects: list[str]) -> dict[str, ObjectSize]:
    """LLM fallback for objects not in priors."""
    if not unknown_objects:
        return {}

    prompt = SIZE_ESTIMATE_PROMPT.format(objects=json.dumps(unknown_objects))
    try:
        raw = call_llm(prompt, max_tokens=1024)
        parsed = parse_json_response(raw)
        sizes = parsed.get("sizes", parsed)
        result = {}
        for name, sz in sizes.items():
            if isinstance(sz, dict) and "w" in sz and "h" in sz and "d" in sz:
                result[name] = ObjectSize(
                    w=float(sz["w"]),
                    h=float(sz["h"]),
                    d=float(sz["d"]),
                )
        return result
    except Exception:
        # Fallback: use generic small object
        return {name: ObjectSize(0.3, 0.4, 0.3) for name in unknown_objects}


def resolve_sizes(relations: list[dict]) -> list[dict]:
    """
    Resolve size_meters for each relation using priors + clamp + LLM fallback.
    Returns relations with size_meters filled.
    """
    unknown = []
    for rel in relations:
        obj_name = rel.get("object", "")
        prior = _lookup_prior(obj_name)
        if prior:
            w, h, d = prior
            size = ObjectSize(w=w, h=h, d=d)
        else:
            unknown.append(obj_name)
            continue

        size = clamp_size(obj_name, size)
        rel["size_meters"] = {"w": size.w, "h": size.h, "d": size.d}

    if unknown:
        llm_sizes = _llm_estimate_sizes(unknown)
        for rel in relations:
            obj_name = rel.get("object", "")
            if obj_name in llm_sizes:
                size = clamp_size(obj_name, llm_sizes[obj_name])
                rel["size_meters"] = {"w": size.w, "h": size.h, "d": size.d}
            elif "size_meters" not in rel:
                rel["size_meters"] = {"w": 0.3, "h": 0.4, "d": 0.3}

    # Ensure all have size_meters
    for rel in relations:
        if "size_meters" not in rel:
            rel["size_meters"] = {"w": 0.3, "h": 0.4, "d": 0.3}

    return relations
