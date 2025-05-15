# SPANetV2 Official (Ongoing)

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

| Model      | Pretrain    | Resolution | Top-1 | #Param. | MACs |
| ---------- | ----------- | ---------- | ----- | ------- | ----- |
| SPANeV2-S18-pure     | ImageNet-1K | 224x224 | 83.4 | 29M  | 4.2G |
| SPANeV2-S18-hybrid   | ImageNet-1K | 224x224 | 83.9 | 27M  | 4.2G |
| SPANeV2-S36-pure     | ImageNet-1K | 224x224 | 84.4 | 44M  | 8.1G |
| SPANeV2-S36-hybrid   | ImageNet-1K | 224x224 | 84.7 | 41M  | 8.1G |
| SPANeV2-M36-pure     | ImageNet-1K | 224x224 | 84.9 | 61M  | 13.7G |
| SPANeV2-M36-hybrid   | ImageNet-1K | 224x224 | 85.3 | 58M  | 13.6G |
| SPANeV2-B36-pure     | ImageNet-1K | 224x224 | 85.0 | 100M | 24.3G |
| SPANeV2-B36-hybrid   | ImageNet-1K | 224x224 | 85.6 | 100M | 23.9G |



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
