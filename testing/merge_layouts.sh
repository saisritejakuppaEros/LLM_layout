#!/bin/bash
# Merge all layout.json files from testing/outputs into a single JSON array.
# Output: testing/outputs/all_layouts.json

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUTS_DIR="${SCRIPT_DIR}/outputs"
OUT_FILE="${OUTPUTS_DIR}/all_layouts.json"

python3 -c "
import json
from pathlib import Path

outputs_dir = Path('${OUTPUTS_DIR}')
out_file = '${OUT_FILE}'

layouts = []
for f in sorted(outputs_dir.rglob('layout.json')):
    with open(f) as fp:
        layouts.append(json.load(fp))

with open(out_file, 'w') as fp:
    json.dump(layouts, fp, indent=2)

print(f'Merged {len(layouts)} layout.json files -> {out_file}')
"
