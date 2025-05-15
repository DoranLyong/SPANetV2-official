# Applying SPANetV2 to Object Detection and Instance Segmentation


##  Environement Setup
Check the [INSTALL.md](./INSTALL.md).


## Data preparation

Prepare COCO 2017 according to the [guidelines](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/useful_tools.md#dataset-download) in MMDetection. If you have interest in this dataset, refer to below links:

* [mmdetection docs](https://mmdetection.readthedocs.io/en/stable/1_exist_data_model.html#prepare-datasets)
* [COCO homepage](https://cocodataset.org/#download)



## Training

To train SPANetV2 + {`Mask R-CNN` or `Cascade Mask-RCNN`} on a single node with 4 GPUs, run:

``` bash
# simple usage 
bash run_train.sh
```


## Evaluation
To evaluate SPANetV2 + {`Mask R-CNN` or `Cascade Mask-RCNN`} on a single GPU, run:

``` bash
# simple usage 
bash run_eval.sh
```
