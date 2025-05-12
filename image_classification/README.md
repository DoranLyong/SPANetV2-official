# SPANetV2 for Classification

<p align="left">
<a href="https://arxiv.org/abs/2503.23947" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2503.23947-b31b1b.svg?style=flat" /></a>
</p>

### To do:
- [] Training and validation code 
- [x] SPANet checkpoints with demo
- [x] Visualization of features out of SPAM


## Requirements

torch>=1.7.0; torchvision>=0.8.0; pyyaml; [timm](https://github.com/rwightman/pytorch-image-models) (`pip install timm==0.6.11`)

Data preparation: ImageNet with the following folder structure, you can extract ImageNet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```


## Acknowledgement 
Our implementation is mainly based on [metaformer baseline](https://github.com/sail-sg/metaformer). We would like to thank for sharing your nice work!
