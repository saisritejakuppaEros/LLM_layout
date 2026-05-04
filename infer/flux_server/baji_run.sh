#!/usr/bin/env bash
# Multi-ref FLUX client: Baji cutout + Theyyam reference (bg_removed_baji_theyyam), same flow as srk_run.sh / aish_run.sh.
# Requires server_flux.py running (default http://127.0.0.1:8765). Override with FLUX_SERVER_URL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BG="/mnt/data0/teja/research_multiref/lora_inferenceing_v3/LLM_layout/infer/cultural_edit/bg_removed_baji_theyyam"
CLIENT="${ROOT}/client_flux.py"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/flux_client_run_baji}"
OUT_BASE="${OUT_BASE:-cultural_scene_baji.png}"

mkdir -p "${OUT_DIR}"

for f in "${BG}/bg_removed_baji.png" "${BG}/bg_removed_theyyam.png"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing reference image: ${f}" >&2
    exit 1
  fi
done

if [[ ! -f "${CLIENT}" ]]; then
  echo "Missing client: ${CLIENT}" >&2
  exit 1
fi

PROMPT="$(cat <<'ENDPROMPT'
Use Image 1 as the immutable base reference.

PRIMARY SUBJECT (LOCKED):
- Preserve the man EXACTLY as in Image 1.
- Do not alter: face, identity, skin tone, shaved head, mustache, expression, pose, clothing, or jewelry.
- He must remain a normal human with no stylization.
- Absolutely no Kathakali makeup, face paint, costume elements, or headgear applied to him.

SCENE TRANSFORMATION:
- Place the man into a realistic Kathakali performance environment inspired by Image 2.
- Use localized inpainting/compositing only around him; do not modify his pixels.

SECONDARY SUBJECT:
- Add a separate Kathakali performer in the midground or background.
- Performer must have: green face makeup, elaborate headgear, red costume, and traditional styling.
- The performer should be in a dynamic dance pose with authentic mudras (hand gestures).
- Keep the performer clearly distinct from the man (no overlap, blending, or fusion).

COMPOSITION:
- The man remains in the foreground or off to one side as the primary subject.
- The Kathakali performer is positioned behind or beside him with correct scale and perspective.
- Maintain clear spatial separation and depth between subjects.

ENVIRONMENT:
- Add a stage or cultural performance setting with appropriate decor.
- Include realistic lighting consistent with the man's original light direction and intensity.
- Add shadows, depth, and grounding for both subjects.

RENDERING STYLE:
- Fully photorealistic.
- No painterly, illustrative, or stylized effects.
- Maintain natural textures, accurate lighting, and believable integration.

STRICT NEGATIVE CONSTRAINTS:
- Do not modify or regenerate the man's identity in any way.
- Do not transfer colors, textures, or costume elements onto the man.
- Do not merge subjects or distort proportions.
ENDPROMPT
)"

export FLUX_SERVER_URL="${FLUX_SERVER_URL:-http://127.0.0.1:8765}"

python3 "${CLIENT}" \
  --url "${FLUX_SERVER_URL}" \
  --caption "${PROMPT}" \
  --ref "${BG}/bg_removed_baji.png" "${BG}/bg_removed_theyyam.png" \
  --variants 4 \
  -o "${OUT_DIR}/${OUT_BASE}" \
  ${SEED:+--seed "${SEED}"} \
  "$@"

echo "Done. Outputs under ${OUT_DIR} (four files: ${OUT_BASE%.*}_var01_seed* … _var04_seed*)."
