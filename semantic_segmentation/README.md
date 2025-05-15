# Applying SPANetV2 to Semantic Segmentation


##  Environement Setup
Check the [INSTALL.md](./INSTALL.md).

## Data preparation
Prepare ADE20K according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) in MMSegmentation. If you have interest in this dataset, refer to below links:

* [mmsegmentation docs](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets)
* [ADE20K homepage](https://ade20k.csail.mit.edu/)

You can specify the path of the dataset in `./local_configs/_base_/datasets/ade20k.py`.

## Training

To train SPANetV2 + {`UperNet` or `Semantic FPN`} on a single node with 4 GPUs, run:

``` bash
# simple usage 
bash run_train.sh
```

Before training, make sure that the pretrained weights on ImageNet-1K are placed in `ckpt` directory. 


## Evaluation
To evaluate SPANetV2 + {`UperNet` or `Semantic FPN`} on a single GPU, run:

``` bash
# simple usage 
bash run_eval.sh
```

