"""Global config — paths and API endpoints only."""

import os

TEXT_API = os.getenv("TEXT_API", "http://localhost:8001/generate")
VL_API   = os.getenv("VL_API",   "http://localhost:8002/generate")

# Scene volume (meters): camera looks down -Z, Y is up
SCENE_WIDTH  = 6.0   # X axis
SCENE_HEIGHT = 3.0   # Y axis
SCENE_DEPTH  = 6.0   # Z axis  (0=camera, DEPTH=back wall)

DEPTH_BANDS = {
    "foreground":  (0.3, 1.5),
    "midground":   (1.5, 3.5),
    "background":  (3.5, 5.8),
}

FLOOR_Y      = 0.0
CEILING_Y    = SCENE_HEIGHT
BACK_WALL_Z  = SCENE_DEPTH

# VLM loop
MAX_VLM_ITERATIONS  = 5
COMPOSITION_THRESH  = 0.75
OCCLUSION_MIN_VISIBLE = 0.30   # 30% of projected area must be visible

# DIoU3D collision
DIOU_COLLISION_THRESH = 0.10

# Environment geometry: objects with w or d > this are tagged, exempt from collision/bounds
ENV_GEOMETRY_THRESHOLD = 3.0

# Zone budget for Phase A placement (X distribution)
ZONE_NAMES = ["left", "center-left", "center", "center-right", "right"]
ZONE_X = {
    "left":         -SCENE_WIDTH * 0.4,
    "center-left":  -SCENE_WIDTH * 0.2,
    "center":       0.0,
    "center-right":  SCENE_WIDTH * 0.2,
    "right":         SCENE_WIDTH * 0.4,
}
ZONE_BUDGET = 3  # max objects per zone

# LayoutVLM consistency
LAYOUT_VLM_CONSISTENCY_THRESH = 0.8

# output
OUTPUT_DIR = os.getenv("STAGE2_OUT", "/mnt/data0/teja/research_multiref/llm_based_layout/outputs/stage2")
