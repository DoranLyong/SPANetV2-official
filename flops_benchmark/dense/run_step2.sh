#!/bin/bash
# Step 2: each repository's own get_flops.py on every dense model (logs -> results/step2/).
#   bash run_step2.sh <COCO dir>      # COCO dir with annotations/instances_val2017.json and val2017/
# SPANetV2 scripts (also used for the constructed configs) run in env spanetv2-dense,
# DFFormer's script in env dfformer-mm0. GPU 0 is visible (the scripts move the model to CUDA when available).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
COCO=$(realpath "$1")
OUT=results/step2
mkdir -p $OUT
source "$(conda info --base)/etc/profile.d/conda.sh"
export CUDA_VISIBLE_DEVICES=0

list() {  # setting|name|config|source for one task
    python -c "
import sys; sys.argv=['x']
import measure_dense as D
for s, n, f, g, k, p, src, ref in D.MODELS:
    if D.SETTINGS[s]['task'] == '$1': print(f'{s}|{n}|{ref}|{src}')"
}

conda activate spanetv2-dense
DET_LIST=$(list det | tr ' ' '#')
SEG_LIST=$(list seg | tr ' ' '#')
for line in $DET_LIST; do
    IFS='|' read -r s n cfg src <<< "$line"
    echo "== $s $n"
    python repo_get_flops.py det "$cfg" --coco "$COCO" --num-images 3 > "$OUT/${s}__${n}.log" 2>&1 || echo "   FAILED (see log)"
    if grep -q OutOfMemoryError "$OUT/${s}__${n}.log"; then   # 24 GB is not enough for some Cascade S36 models
        echo "   GPU out of memory -> rerun on the CPU (the script uses CUDA only when it is available)"
        mv "$OUT/${s}__${n}.log" "$OUT/${s}__${n}.gpu_oom.log"
        CUDA_VISIBLE_DEVICES= python repo_get_flops.py det "$cfg" --coco "$COCO" --num-images 3 > "$OUT/${s}__${n}.log" 2>&1 || echo "   FAILED (see log)"
    fi
    tail -n 1 "$OUT/${s}__${n}.log" | cut -c1-200
done
for line in $SEG_LIST; do
    IFS='|' read -r s n cfg src <<< "$line"
    case "$n" in DFFormer*|CDFFormer*) continue ;; esac
    shape=$([ "$s" = upernet ] && echo "512 2048" || echo "512 512")
    echo "== $s $n ($shape)"
    python repo_get_flops.py seg "$cfg" --shape $shape > "$OUT/${s}__${n}.log" 2>&1 || echo "   FAILED (see log)"
    tail -n 1 "$OUT/${s}__${n}.log" | cut -c1-200
done

conda activate dfformer-mm0
for line in $SEG_LIST; do
    IFS='|' read -r s n cfg src <<< "$line"
    case "$n" in DFFormer*|CDFFormer*) ;; *) continue ;; esac
    echo "== $s $n (512 512, DFFormer tool)"
    python repo_get_flops.py dfformer "$cfg" --shape 512 512 > "$OUT/${s}__${n}.log" 2>&1 || echo "   FAILED (see log)"
    tail -n 1 "$OUT/${s}__${n}.log" | cut -c1-200
done
