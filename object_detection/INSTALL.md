# Installation 
The environment is tested on `PyTorch_2.1` with `CUDA_12.1`.

Prerequisite:
```bash 
#-- First, sync the uv env. to MMDetection dependenceis
pip install uv 
uv sync --index-strategy unsafe-best-match --upgrade
```

## MMdet Install 
Step 0. Install `MMCV-Full`

```bash 
#-- Run the uv env. 
source .venv/bin/activate 

#-- Next,
mim install mmcv-full
mim install mmcls
```



Step 1. Install [MMDetection](https://github.com/open-mmlab/mmdetection/blob/v2.28.2/docs/en/get_started.md/#Installation).

```bash
mim install mmdet==2.28.2
```

## Verifying the Installation 
You can verify that the mmdet package is installed by running the follows.

Step 1. We need to download config and checkpoint files.
```bash
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
```

Step 2. Verify the inference demo.
```bash
python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file ./demo/result.jpg
```
