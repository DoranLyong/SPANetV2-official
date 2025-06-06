# Applying SPANetV2 to Object Detection and Instance Segmentation


##  Environement Setup
Check the [INSTALL.md](./INSTALL.md).


## Data preparation

Prepare COCO 2017 according to the [guidelines](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/useful_tools.md#dataset-download) in MMDetection. If you have interest in this dataset, refer to below links:

* [mmdetection docs](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html)
* [COCO homepage](https://cocodataset.org/#download)


You can specify the path of the dataset in `./local_configs/_base_/datasets/coco_instance.py`.



## Training

To train SPANetV2 + {`Mask R-CNN` or `Cascade Mask-RCNN`} on a single node with 4 GPUs, run:

``` bash
# simple usage 
bash run_train.sh
```

Before training, make sure that the pretrained weights on ImageNet-1K are placed in `ckpt` directory. 



## Evaluation
To evaluate SPANetV2 + {`Mask R-CNN` or `Cascade Mask-RCNN`} on a single GPU, run:

``` bash
# simple usage 
bash run_eval.sh
```
