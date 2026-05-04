#!/usr/bin/env bash
# Multi-ref FLUX client: Aish cutout + Kali Maa reference (bg_removed_aish_kalimaa), same flow as srk_run.sh.
# Requires server_flux.py running (default http://127.0.0.1:8765). Override with FLUX_SERVER_URL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BG="/mnt/data0/teja/research_multiref/lora_inferenceing_v3/LLM_layout/infer/cultural_edit/bg_removed_aish_kalimaa"
CLIENT="${ROOT}/client_flux.py"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/flux_client_run_aish}"
OUT_BASE="${OUT_BASE:-cultural_scene_aish.png}"

mkdir -p "${OUT_DIR}"

for f in "${BG}/bg_removed_aish.png" "${BG}/bg_removed_kali_maa.png"; do
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
    "Use image 1 as the fixed base. Keep the woman EXACTLY unchanged — same face, hairstyle, jewelry, expression, "
    "body proportions, pose, saree color, embroidery, and fabric details. Do not modify or regenerate her in any way.\n\n"

    "She must remain a normal human standing figure. Do NOT transform her into a deity, idol, or multi-armed form.\n\n"

    "Perform localized inpainting around her to integrate her into a traditional Kali puja setting inspired by image 2.\n\n"

    "Add the following elements while keeping clear separation from the woman:\n"
    "- A Kali idol placed centrally in the background with correct structure and symmetry\n"
    "- Ritual decorations: flower garlands, lamps, offerings, ornaments\n"
    "- Supporting figures or props only in background or midground (do not overlap or distort the woman)\n\n"

    "Composition constraints:\n"
    "- The woman remains in foreground or slightly off-center\n"
    "- The idol stays in the background as the main ritual focus\n"
    "- Maintain proper scale difference (idol larger, woman realistic human size)\n"
    "- No merging of limbs, jewelry, or textures between subject and idol\n\n"

    "Lighting and realism:\n"
    "- Match lighting direction, intensity, and color temperature to the woman\n"
    "- Add soft shadows and grounding to avoid cut-paste look\n"
    "- Ensure depth, perspective, and natural occlusion\n\n"

    "Strict constraints:\n"
    "- Do NOT alter the woman's face, hands, or clothing\n"
    "- Do NOT add extra limbs or accessories to her\n"
    "- Do NOT stylize into painting or sculpture — keep photorealistic\n\n"

    "Goal: a seamless, realistic scene where the woman from image 1 is naturally present at a Kali puja setting "
    "inspired by image 2, without any identity or structural distortion."
ENDPROMPT
)"

export FLUX_SERVER_URL="${FLUX_SERVER_URL:-http://127.0.0.1:8765}"

python3 "${CLIENT}" \
  --url "${FLUX_SERVER_URL}" \
  --caption "${PROMPT}" \
  --ref "${BG}/bg_removed_aish.png" "${BG}/bg_removed_kali_maa.png" \
  --variants 4 \
  -o "${OUT_DIR}/${OUT_BASE}" \
  ${SEED:+--seed "${SEED}"} \
  "$@"

echo "Done. Outputs under ${OUT_DIR} (four files: ${OUT_BASE%.*}_var01_seed* … _var04_seed*)."
