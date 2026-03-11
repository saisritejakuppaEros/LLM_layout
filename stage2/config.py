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

# output
OUTPUT_DIR = os.getenv("STAGE2_OUT", "/mnt/data0/teja/research_multiref/llm_based_layout/outputs/stage2")
