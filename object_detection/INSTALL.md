# Installation 
The environment is tested on `PyTorch_2.1` with `CUDA_12.2`.

Prerequisite:
```bash 
pip install opencv-python>=4.8.0.74   #for DictValue of cv2.dnn
pip install ftfy
```

## MMdet Install 
Step 0. Install [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-on-linux) and [MMEngine](https://mmengine.readthedocs.io/en/latest/get_started/installation.html).

```bash 
pip install -U openmim
mim install mmengine
mim install mmcv
```


Step 1. Install [MMDetection](https://github.com/open-mmlab/mmdetection) from source.

```bash
git clone https://github.com/open-mmlab/mmdetection/tree/v3.3.0
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

## Verifying the Installation 
You can verify that the mmdet package is installed by running the follows.

If you install mmdetection from source, just run the following instructions:
* download the [rtmdet_tiny_8xb32-300e_coco](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) model.
* then, run the following command
```bash
python demo/image_demo.py demo/demo.jpg configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py --weights ./ckpt/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cuda:0 --out-dir ./demo/result
```
