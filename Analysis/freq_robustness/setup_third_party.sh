#!/usr/bin/env bash
# Clone the official repositories used here at the commits the results were produced with (nothing in them is changed).
# MetaFormer and DFFormer come from ../../flops_benchmark/third_party (bash ../../flops_benchmark/setup_third_party.sh).
#   bash setup_third_party.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p third_party && cd third_party
clone() {  # url dir commit
  [ -d "$2" ] || git clone -q "$1" "$2"
  git -C "$2" fetch -q --depth 200 origin 2>/dev/null || true
  git -C "$2" checkout -q "$3"
  echo "$2 $(git -C "$2" rev-parse HEAD)"
}
clone https://github.com/facebookresearch/ConvNeXt.git                          ConvNeXt                               048efcea897d999aed302f2639b6270aedf8d4c8
clone https://github.com/microsoft/Swin-Transformer.git                         Swin-Transformer                       f82860bfb5225915aca09c3227159ee9e1df874d
clone https://github.com/facebookresearch/deit.git                              deit                                   7e160fe43f0252d17191b71cbb5826254114ea5b
clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation.git Swin-Transformer-Semantic-Segmentation 87e6f90577435c94f3e92c7db1d36edc234d91f6
clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git  Swin-Transformer-Object-Detection      c7b20110addde0f74b1fbf812b403d16a59a87a9
clone https://github.com/sail-sg/poolformer.git                                 poolformer                             a075575b8b3dc3c76ecc5c49985d156167372bfa
clone https://github.com/bowenc0221/boundary-iou-api.git                        boundary-iou-api                       37d25586a677b043ed585f10e5c42d4e80176ea9
