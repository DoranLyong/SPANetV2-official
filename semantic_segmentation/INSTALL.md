# Installation 
The environment is tested on `PyTorch_2.1` with `CUDA_12.2`.

Prerequisite:
```bash 
pip install opencv-python==4.8.0.74   #for DictValue of cv2.dnn
pip install ftfy
```

## MMseg Install 
Step 0. Install [MMCV](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-on-linux) and [MMEngine](https://mmengine.readthedocs.io/en/latest/get_started/installation.html).

```bash 
pip install -U openmim
mim install mmengine
mim install mmcv
```


Step 1. Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) from source.

```bash
git clone https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2
cd mmsegmentation
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

## Verifying the Installation 
You can verify that the mmdet package is installed by running the follows.

If you install mmsegmentation from source, just run the following instructions:
* download the [pspnet_r50-d8_512x1024_40k_cityscapes](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet) model.
* then, run the following command
```bash
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py ./ckpt/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file ./demo/result.jpg
```
