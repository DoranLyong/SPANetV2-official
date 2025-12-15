# SPANetV2 Official

<p align="left">
<a href="https://arxiv.org/abs/2503.23947" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2503.23947-b31b1b.svg?style=flat" /></a>
</p>

### 📜 History:
- ***arXiv2025***: Spectral-Adaptive Modulation Networks for Visual Perception
- ***ICCV2023***: [SPANet: Frequency-balancing Token Mixer using Spectral Pooling Aggregation Modulation](https://doranlyong.github.io/projects/spanet/)

  

### 🤖 It currently includes code and models for the following tasks:
- [x] [Image Classification](./image_classification)
- [x] [Object Detection](object_detection)
- [x] [Semantic Segmentation](semantic_segmentation)

## Main results on ImageNet-1K
Please see [image_classification](image_classification) for more details.

| Model      | Pretrain    | Resolution | Top-1 | #Params (M) | MACs (G)|
| ---------- | ----------- | ---------- | ----- | ------- | ----- |
| SPANeV2-S18-pure     | ImageNet-1K | 224x224 | 83.4 | 29  | 4.2 |
| SPANeV2-S18-hybrid   | ImageNet-1K | 224x224 | 83.9 | 27  | 4.2 |
| SPANeV2-S36-pure     | ImageNet-1K | 224x224 | 84.4 | 44  | 8.1 |
| SPANeV2-S36-hybrid   | ImageNet-1K | 224x224 | 84.7 | 41  | 8.1 |
| SPANeV2-M36-pure     | ImageNet-1K | 224x224 | 84.9 | 61  | 13.7 |
| SPANeV2-M36-hybrid   | ImageNet-1K | 224x224 | 85.3 | 58  | 13.6 |
| SPANeV2-B36-pure     | ImageNet-1K | 224x224 | 85.0 | 100 | 24.3 |
| SPANeV2-B36-hybrid   | ImageNet-1K | 224x224 | 85.6 | 100 | 23.9 |


With the scaled sigmoid for `SRF`
| Model      | Pretrain    | Resolution | Top-1 | #Params (M) | MACs (G)|
| ---------- | ----------- | ---------- | ----- | ------- | ----- |
| SPANeV2-S18-pure     | ImageNet-1K | 224x224 | 83.5 | 29  | 4.2 |
| SPANeV2-S18-hybrid   | ImageNet-1K | 224x224 | -.- | 27  | 4.2 |
| SPANeV2-S36-pure     | ImageNet-1K | 224x224 | -.- | 44  | 8.1 |
| SPANeV2-S36-hybrid   | ImageNet-1K | 224x224 | 84.9 | 41  | 8.1 |


## Main results on COCO object detection and instance segmentation 
Please see [object_detection](object_detection) for more details.


### Mask R-CNN 3x
|         Backbone          |box mAP | mask mAP | MACs (G)|
| :---------------          |  :-----  | :------  | :-----  | 
| SPANet-S18-pure           |   48.0   |   42.9   |   255   | 
| SPANet-S18-hybrid         |   49.6   |   44.3   |   251   | 


### Cascade Mask R-CNN 3x
|         Backbone          |  box mAP | mask mAP | MACs (G)|
| :---------------          |  :-----  | :------  | :-----  | 
| SPANet-S18-pure           |   51.6   |   44.7   |   734   | 
| SPANet-S18-hybrid         |   52.8   |   45.7   |   729   | 

## Main results on ADE20K semantice segmentation 
Please see [semantic_segmentation](semantic_segmentation) for more details.

### Semantic FPN
|         Backbone     |  mIoU | #params (M) | MACs (G) |
| :------------------- |  :--  | :-----  | :--- |
| SPANet-S18-pure      |  46.7 |   33   | 43   |
| SPANet-S18-hybrid    |  47.8 |   31   | 48   |
| SPANet-S36-pure      |  47.9 |   48   | 64   |
| SPANet-S36-hybrid    |  48.6 |   45   | 74   |

### UperNet
|         Backbone     |  mIoU | #params (M) | MACs (G) |
| :------------------- |  :--  | :----- | :--- |
| SPANet-S18-pure      |  48.7 |   60   | 930  |
| SPANet-S18-hybrid    |  49.1 |   58   | 925  |
| SPANet-S36-pure      |  49.8 |   75   | 1012 |
| SPANet-S36-hybrid    |  51.6 |   71   | 1004 |


## ⭐ Cite SPANetV2
If you find this repository useful, please give us stars and use the following BibTeX entry for citation.

```latex
@article{yun2025spectral,
  title={Spectral-Adaptive Modulation Networks for Visual Perception},
  author={Yun, Guhnoo and Yoo, Juhan and Kim, Kijung and Lee, Jeongho and Seo, Paul Hongsuck and Kim, Dong Hwan},
  journal={arXiv preprint arXiv:2503.23947},
  year={2025}
}
```
