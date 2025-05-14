# SPANetV2 Official (Coming Soon)

<p align="left">
<a href="https://arxiv.org/abs/2503.23947" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2503.23947-b31b1b.svg?style=flat" /></a>
</p>

### 💬 News:
- ***arXiv2025***: Spectral-Adaptive Modulation Networks for Visual Perception
- ***ICCV2023***: [SPANet: Frequency-balancing Token Mixer using Spectral Pooling Aggregation Modulation](https://doranlyong.github.io/projects/spanet/)

  

### 🤖 It currently includes code and models for the following tasks:
- [x] [Image Classification](./image_classification)
- [] [Object Detection](object_detection)
- [] [Semantic Segmentation](semantic_segmentation)

## Main results on ImageNet-1K
Please see [image_classification](image_classification) for more details.

| Model      | Pretrain    | Resolution | Top-1 | #Param. | FLOPs |
| ---------- | ----------- | ---------- | ----- | ------- | ----- |
| SPANet-S   | ImageNet-1K | 224x224    | 83.1  | 28.7M   | 4.6G |
| SPANet-M   | ImageNet-1K | 224x224    | 83.5  | 41.8M   | 6.8G |
| SPANet-MX   | ImageNet-1K | 224x224    | 83.8  | 54.9M   | 9.0G |
| SPANet-B   | ImageNet-1K | 224x224    | 84.0  | 75.9M   | 12.0G |
| SPANet-BX   | ImageNet-1K | 224x224    | 84.4  | 99.8 M   | 15.8G |


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
