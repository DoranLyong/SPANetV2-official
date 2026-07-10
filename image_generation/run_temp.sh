#!/bin/bash
set -e

echo "Initializing environment..."

eval "$(mamba shell hook --shell bash)"
mamba activate spanetv2

echo "Running model analysis..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- ExSPAM DiT-S/2 분석 (파라미터, FLOPs, 속도):
python "${SCRIPT_DIR}/temp.py" --model DiT-S/2 --framework dit --image_size 256

# -- ExSPAM SiT-S/2 분석:
#python "${SCRIPT_DIR}/temp.py" --model SiT-S/2 --framework sit --image_size 256
