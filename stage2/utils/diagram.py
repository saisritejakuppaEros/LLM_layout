"""
Scene Diagram Renderer.

Renders a front-view orthographic projection diagram (PNG) from current placements.
Uses numbered labels + legend to avoid overlap. Foreground objects are emphasized.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from stage2.config import SCENE_WIDTH, SCENE_HEIGHT
from stage2.utils.types import ObjectPlacement, ObjectSize

LAYER_COLORS = {
    "foreground":  "#e74c3c",
    "midground":   "#e67e22",
    "background":  "#3498db",
}
WALL_COLOR = "#95a5a6"


def _name_to_short(name: str, max_len: int = 8) -> str:
    """Abbreviate long names to reduce overlap."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "."


def render_front_view(
    placements: dict[str, ObjectPlacement],
    scene_data: dict,
    output_path: str,
) -> str:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-SCENE_WIDTH / 2, SCENE_WIDTH / 2)
    ax.set_ylim(0, SCENE_HEIGHT)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")

    ax.set_xlabel("X (left ← → right)", color="white", fontsize=9)
    ax.set_ylabel("Y (floor → ceiling)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    title = scene_data.get("theme", "Scene Layout")
    ax.set_title(f"Front-View Layout — {title}", color="white", fontsize=12, pad=10)

    # floor line
    ax.axhline(y=0, color="#555", linewidth=1.5, linestyle="--", label="floor")

    # draw objects sorted back to front (background first so fg renders on top)
    layer_order = ["background", "midground", "foreground"]
    ordered = sorted(
        placements.values(),
        key=lambda p: (layer_order.index(p.layer) if p.layer in layer_order else 0, -p.z)
    )

    # Build name->id map for legend
    name_to_id = {p.name: i + 1 for i, p in enumerate(ordered)}

    for p in ordered:
        s = p.size or ObjectSize(0.3, 0.4, 0.3)
        color = WALL_COLOR if p.surface == "wall" else LAYER_COLORS.get(p.layer, "#aaa")
        # Foreground: brighter, thicker border. Background: dimmer
        alpha = 0.45 if p.layer == "background" else 0.7 if p.layer == "midground" else 0.95
        lw = 1.0 if p.layer == "background" else 1.5 if p.layer == "midground" else 2.0

        rect = patches.FancyBboxPatch(
            (p.x - s.w / 2, p.y - s.h / 2),
            s.w, s.h,
            boxstyle="round,pad=0.02",
            linewidth=lw,
            edgecolor="white",
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect)

        # Small ID number in top-left corner of box (no overlap with other boxes)
        obj_id = name_to_id[p.name]
        ax.text(
            p.x - s.w / 2 + 0.08,
            p.y + s.h / 2 - 0.08,
            str(obj_id),
            ha="left", va="top",
            fontsize=8, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.7, edgecolor="white", linewidth=0.5),
        )

        # Short label above box (reduces overlap vs center labels)
        short = _name_to_short(p.name)
        ax.text(
            p.x, p.y + s.h / 2 + 0.12,
            short,
            ha="center", va="bottom",
            fontsize=6, color="white",
            fontweight="normal",
        )

    # Legend: ID -> full name (bottom, wrapped to avoid overflow)
    legend_items = [f"{name_to_id[p.name]}: {p.name}" for p in ordered]
    n_per_line = 5
    lines = []
    for i in range(0, len(legend_items), n_per_line):
        lines.append("  |  ".join(legend_items[i : i + n_per_line]))
    legend_text = "\n".join(lines)
    n_lines = len(lines)
    bottom_offset = 0.04 + 0.025 * n_lines
    ax.text(0.5, -bottom_offset, legend_text, transform=ax.transAxes,
            ha="center", va="top", fontsize=6, color="#aaa",
            family="monospace")

    # Layer legend (compact)
    legend_elements = [
        patches.Patch(facecolor=LAYER_COLORS["foreground"], label="FG", alpha=0.8),
        patches.Patch(facecolor=LAYER_COLORS["midground"],  label="MG", alpha=0.8),
        patches.Patch(facecolor=LAYER_COLORS["background"], label="BG", alpha=0.8),
        patches.Patch(facecolor=WALL_COLOR, label="Wall", alpha=0.8),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
              facecolor="#222", edgecolor="#555", labelcolor="white", ncol=2)

    plt.subplots_adjust(bottom=0.12)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path


def compute_bboxes_from_placements(
    placements: dict,
    image_path: str,
) -> dict[str, tuple[int, int, int, int]]:
    """
    Compute pixel bboxes for each object from placements.
    Returns: {obj_name: (x1, y1, x2, y2)} in pixel coords (top-left origin, y down)
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            width_px = img.width
            height_px = img.height
    except Exception:
        width_px = int(12 * 150)
        height_px = int(7 * 150)

    half_w = SCENE_WIDTH / 2.0
    x_min_world = -half_w
    x_max_world = half_w
    y_min_world = 0.0
    y_max_world = SCENE_HEIGHT

    result = {}
    for name, p in placements.items():
        if hasattr(p, "size"):
            s = p.size or ObjectSize(0.3, 0.4, 0.3)
            x, y = p.x, p.y
        else:
            sz = p.get("size")
            if isinstance(sz, dict):
                s = ObjectSize(sz.get("w", 0.3), sz.get("h", 0.4), sz.get("d", 0.3))
            else:
                s = ObjectSize(0.3, 0.4, 0.3)
            x = p.get("x", 0)
            y = p.get("y", 0)
        x1_w = x - s.w / 2
        x2_w = x + s.w / 2
        y1_w = y - s.h / 2
        y2_w = y + s.h / 2

        x1_n = (x1_w - x_min_world) / (x_max_world - x_min_world)
        x2_n = (x2_w - x_min_world) / (x_max_world - x_min_world)
        y1_n = (y1_w - y_min_world) / (y_max_world - y_min_world)
        y2_n = (y2_w - y_min_world) / (y_max_world - y_min_world)

        y1_px = int((1.0 - y2_n) * height_px)
        y2_px = int((1.0 - y1_n) * height_px)
        x1_px = int(x1_n * width_px)
        x2_px = int(x2_n * width_px)

        result[name] = (max(0, x1_px), max(0, y1_px), min(width_px, x2_px), min(height_px, y2_px))

    return result
