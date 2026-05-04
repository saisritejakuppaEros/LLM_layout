#!/usr/bin/env bash
# Multi-ref FLUX client: two cutouts from bg_removed + the same scene prompt as stage1_gen.py (ai_prompt).
# Requires server_flux.py running (default http://127.0.0.1:8765). Override with FLUX_SERVER_URL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BG="/mnt/data0/teja/research_multiref/lora_inferenceing_v3/LLM_layout/infer/cultural_edit/bg_removed"
# Anchor to this directory so the script works when run from any cwd.
CLIENT="${ROOT}/client_flux.py"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/flux_client_run}"
OUT_BASE="${OUT_BASE:-cultural_scene.png}"

mkdir -p "${OUT_DIR}"

for f in "${BG}/upanayanam.png" "${BG}/srk_2.png"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing reference image: ${f}" >&2
    exit 1
  fi
done

if [[ ! -f "${CLIENT}" ]]; then
  echo "Missing client: ${CLIENT}" >&2
  exit 1
fi

# Same text as stage1_gen.py ``ai_prompt`` (lines 28–36).
PROMPT="$(cat <<'ENDPROMPT'
    "Use image 1 as the fixed base. Keep the man EXACTLY unchanged — same face, hairstyle, expression, pose, "
    "body proportions, and purple kurta with identical fabric details. He must remain standing in the same position "
    "and orientation at the corner of the frame. Do not modify or regenerate him.\n\n"

    "Perform localized inpainting around him to place him inside an upanayanam ceremony inspired by image 2.\n\n"

    "Add the following elements with correct spatial arrangement and scale:\n"
    "- A seated young boy and a priest performing the ritual in front/center\n"
    "- Sacred thread ceremony interaction (yajnopavita), with natural hand placement\n"
    "- Traditional pooja setup: plate, flowers, darbha grass, vessels\n"
    "- Floor seating arrangement and ceremonial environment\n\n"

    "Match the scene to the subject:\n"
    "- Lighting direction, intensity, and color temperature must match the man\n"
    "- Ensure realistic shadows and grounding (contact shadows under feet)\n"
    "- Maintain consistent perspective and camera angle\n"
    "- Ensure proper depth and occlusion (foreground/background layering)\n\n"

    "Strict constraints:\n"
    "- Do NOT alter the man's face, skin tone, clothing color, or texture\n"
    "- Do NOT change his pose, position, or proportions\n"
    "- Do NOT restyle into painting or illustration — keep fully photorealistic\n\n"

    "Goal: a seamless, realistic composite where the man from image 1 naturally appears present "
    "at the edge of an upanayanam ceremony scene from image 2."
ENDPROMPT
)"

export FLUX_SERVER_URL="${FLUX_SERVER_URL:-http://127.0.0.1:8765}"

python3 "${CLIENT}" \
  --url "${FLUX_SERVER_URL}" \
  --caption "${PROMPT}" \
  --ref "${BG}/upanayanam.png" "${BG}/srk_2.png" \
  --variants 4 \
  -o "${OUT_DIR}/${OUT_BASE}" \
  ${SEED:+--seed "${SEED}"} \
  "$@"

echo "Done. Outputs under ${OUT_DIR} (four files: ${OUT_BASE%.*}_var01_seed* … _var04_seed*)."
