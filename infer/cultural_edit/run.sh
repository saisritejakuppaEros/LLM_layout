#!/usr/bin/env bash
# Multi-ref FLUX client: two cutouts from bg_removed + the same scene prompt as stage1_gen.py (ai_prompt).
# Requires server_flux.py running (default http://127.0.0.1:8765). Override with FLUX_SERVER_URL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BG="${ROOT}/bg_removed"
CLIENT="${ROOT}/../flux_server/client_flux.py"
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
A realistic Indian cultural ceremony scene set in a softly lit traditional indoor environment. In the center, a young boy sits cross-legged on the floor wearing traditional white attire, participating in a sacred thread ceremony (upanayanam). A middle-aged priest/father figure, bare-chested with a dhoti, is carefully guiding the ritual, holding and placing a sacred thread (yajnopavita) across the boy’s shoulder with focused attention. Ritual items like a small fire (havan), brass vessels, flowers, and offerings are placed neatly around them on a mat.

To the side, Shah Rukh Khan stands slightly behind and to the right, dressed in an elegant deep blue sherwani with black trousers. He is observing the ceremony with a calm, respectful expression, hands relaxed, slightly leaning forward as if attentively watching the ritual unfold.

Lighting is warm and natural, coming from the side, casting soft shadows. The composition is balanced, with depth and perspective maintained so all subjects feel naturally placed in the same environment. The background includes subtle traditional decor like muted walls, soft textures, and possibly a temple-like ambiance. Skin tones, shadows, and reflections are realistic, ensuring seamless blending of all individuals into a single cohesive scene.

Style: photorealistic, high detail, natural skin tones, cinematic lighting, 35mm lens feel, shallow depth of field.
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
