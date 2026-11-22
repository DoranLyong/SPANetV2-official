#!/usr/bin/env bash
# Full re-run (inference only). Order used for report.md; each step skips work whose output already exists.
#   bash run_all.sh            (GPUs 0-3; ImageNet / ADE20K / COCO paths below)
set -e
cd "$(dirname "$0")"
PYC=/home/server/miniforge3/envs/spanetv2/bin/python          # torch 2.8, timm 0.6.13      (0, A, B, C)
PYD=/home/server/miniforge3/envs/spanetv2-dense/bin/python    # torch 2.1, mmcv 2.1, mmengine 0.10.7 (D, figures)
IMNET=/home/server/dataset/ImageNet2012/val

bash setup_third_party.sh
python3 download_weights.py all
[ -f cache/val_crop1.000.npy ] || $PYC make_cache.py --val $IMNET
mkdir -p cache/datasets/ade20k cache/datasets/coco
[ -d cache/datasets/ade20k/ADEChallengeData2016 ] || unzip -q /home/server/dataset/ade20k/ADEChallengeData2016.zip \
  'ADEChallengeData2016/images/validation/*' 'ADEChallengeData2016/annotations/validation/*' -d cache/datasets/ade20k
[ -d cache/datasets/coco/val2017 ] || unzip -q /home/server/dataset/coco2017/val2017.zip -d cache/datasets/coco
[ -f cache/datasets/coco/annotations/instances_val2017.json ] || unzip -q /home/server/dataset/coco2017/annotations_trainval2017.zip \
  'annotations/instances_val2017.json' -d cache/datasets/coco

python3 scheduler.py --gpus 0 1 2 --exps clean band fourier pgd fourier_sparse &   # 0, A, B, C, B' (stride-8 grid)
bash run_dense.sh 3                                                                  # D: ADE20K then COCO inference
wait
$PYD boundary_eval.py --select-examples
$PYD boundary_eval.py --models $(ls data/raw/seg_*.json | sed 's|.*/seg_||; s|\.json||')
$PYD boundary_ap.py --models $(ls data/raw/det_*.json | sed 's|.*/det_||; s|\.json||')
CUDA_VISIBLE_DEVICES= $PYD make_figs.py
$PYD write_conclusions.py
$PYD make_report.py
