#!/bin/bash
# Environment and third-party setup for the dense-prediction (COCO / ADE20K) re-measurement.
#   bash setup.sh            # clone third_party (../setup_third_party.sh) and apply the import patch
#   bash setup.sh envs       # also create the two conda environments
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
HERE=$(pwd)

bash ../setup_third_party.sh

# SPANet-official/object_detection/mmdet/models/backbones/spanet.py is written for mmcv 1.x / mmdet 2.x.
# Only its import lines are changed so that it loads with mmcv 2.x / mmengine / mmseg 1.x.
git -C ../third_party/SPANet-official apply --check "$HERE/patches/SPANet_det_backbone_mm2_imports.patch" 2>/dev/null \
    && git -C ../third_party/SPANet-official apply "$HERE/patches/SPANet_det_backbone_mm2_imports.patch" \
    && echo "patched SPANet-official/object_detection/mmdet/models/backbones/spanet.py" \
    || echo "SPANet import patch already applied"

[ "$1" = "envs" ] || exit 0
source "$(conda info --base)/etc/profile.d/conda.sh"

# (1) spanetv2-dense: every measurement in the tables (mmengine analyser + the convention handlers).
#     Versions follow object_detection/ and semantic_segmentation/ (pyproject.toml, INSTALL.md):
#     torch 2.1.0 (cu121), mmcv 2.1.0, timm 0.6.13, numpy<1.26; mmdet 3.3.0 / mmseg 1.2.2 are the copies
#     inside this repository (put on PYTHONPATH by the scripts).
conda create -y -q -n spanetv2-dense python=3.10
conda activate spanetv2-dense
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy>=1.24,<1.26" mmengine==0.10.7 opencv-python==4.8.0.74 timm==0.6.13 fvcore==0.1.5.post20221221 \
    pycocotools scipy shapely terminaltables prettytable ftfy regex matplotlib
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

# (2) dfformer-mm0: only to run DFFormer's own segmentation/tools/get_flops.py (mmcv 1.x / mmseg 0.x),
#     versions from dfformer/requirements.txt (torch 1.12.1, mmcv-full 1.6.2, mmsegmentation 0.29.1, timm 0.5.4).
conda create -y -q -n dfformer-mm0 python=3.10
conda activate dfformer-mm0
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install "numpy<1.24" timm==0.5.4 mmsegmentation==0.29.1 prettytable opencv-python==4.7.0.68 yapf==0.32.0
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
