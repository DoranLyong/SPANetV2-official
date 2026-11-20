# 검출·분할 모델 연산량 재측정 (SPANet·SPANetV2와 비교 모델)

학위논문의 COCO 검출 표와 ADE20K 분할 표에 실린 MACs와 #params를 다시 쟀다. 규약은 분류 재측정(`../README.md`, `../measure.py`)과 같다. 측정 결과만 적는다.

- **측정 조건:** CPU, eval 모드, 무작위 초기화한 가중치(체크포인트를 읽지 않음), `torch.rand` 입력(시드 0), 배치 1.
- **단위와 표기:** 값은 모두 MACs(곱셈-덧셈 1회 = 1)다. G = 10^9, M = 10^6이다. 소수 셋째 자리까지, 반올림하기 전 값이다.
- **열의 뜻:** (가)는 mmengine 기본 분석이다. tools/analysis_tools/get_flops.py와 같은 함수다. +(a)부터는 (가)에 차례로 더한 값이다. "그중 backbone"은 같은 입력으로 backbone만 잰 합계다.
- **실행:** `python measure_dense.py det|seg|cls384` → `python report_dense.py`.

## 결과

### COCO Mask R-CNN 3× — 입력 1×3×1280×800

| 모델 | Params (M) | 논문 값 | (가) 기본 | +(a) | +(b) | +(c) | +(d) | 합계 | 그중 backbone |
|---|---|---|---|---|---|---|---|---|---|
| **S18** |  |  |  |  |  |  |  |  |  |
| ConvFormer-S18 | 43.838 | 251 | 250.814 | 250.814 | 250.814 | 251.538 | 251.538 | 251.538 | 81.422 |
| SPANetV2-S18-pure | 49.331 | 255 | 255.631 | 256.504 | 256.520 | 257.197 | 257.213 | 257.213 | 87.098 |
| CAFormer-S18 | 43.405 | 254 | 344.766 | 344.766 | 344.766 | 348.388 | 348.388 | 348.388 | 178.272 |
| SPANetV2-S18-hybrid | 47.160 | 251 | 346.185 | 346.750 | 346.759 | 350.365 | 350.374 | 350.374 | 180.259 |

- 모든 모델에서 RoI는 bbox head마다 1000개, mask head마다 그중 앞의 100개로 셌다 (2절 끝, 4절의 `TwoStageFixedRoI`). Params는 검출기 전체다.
- Params에는 forward에서 한 번도 호출되지 않는 모듈의 파라미터도 들어 있다(측정 도구가 세는 그대로): SPANetV2-S18-pure `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M; SPANetV2-S18-hybrid `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M.

### COCO Cascade Mask R-CNN 3× — 입력 1×3×1280×800

| 모델 | Params (M) | 논문 값 | (가) 기본 | +(a) | +(b) | +(c) | +(d) | 합계 | 그중 backbone |
|---|---|---|---|---|---|---|---|---|---|
| **S18** |  |  |  |  |  |  |  |  |  |
| ConvFormer-S18 | 81.673 | 729 | 729.366 | 729.366 | 729.366 | 730.090 | 730.090 | 730.090 | 81.422 |
| SPANetV2-S18-pure | 87.167 | 734 | 734.183 | 735.056 | 735.072 | 735.749 | 735.765 | 735.765 | 87.098 |
| CAFormer-S18 | 81.241 | 733 | 823.318 | 823.318 | 823.318 | 826.940 | 826.940 | 826.940 | 178.272 |
| SPANetV2-S18-hybrid | 84.996 | 729 | 824.737 | 825.302 | 825.311 | 828.917 | 828.926 | 828.926 | 180.259 |
| **S36** |  |  |  |  |  |  |  |  |  |
| ConvFormer-S36 | 94.911 | 805 | 805.149 | 805.149 | 805.149 | 806.495 | 806.495 | 806.495 | 157.827 |
| SPANetV2-S36-pure | 102.102 | 814 | 814.844 | 816.509 | 816.541 | 817.795 | 817.827 | 817.827 | 169.159 |
| CAFormer-S36 | 94.196 | 811 | 990.132 | 990.132 | 990.132 | 997.187 | 997.187 | 997.187 | 348.520 |
| SPANetV2-S36-hybrid | 98.673 | 806 | 993.756 | 994.837 | 994.856 | 1001.879 | 1001.897 | 1001.897 | 353.230 |

- 모든 모델에서 RoI는 bbox head마다 1000개, mask head마다 그중 앞의 100개로 셌다 (2절 끝, 4절의 `TwoStageFixedRoI`). Params는 검출기 전체다.
- Params에는 forward에서 한 번도 호출되지 않는 모듈의 파라미터도 들어 있다(측정 도구가 세는 그대로): SPANetV2-S18-pure `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M; SPANetV2-S18-hybrid `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M; SPANetV2-S36-pure `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M; SPANetV2-S36-hybrid `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M.

### ADE20K UperNet — 입력 1×3×512×2048

| 모델 | Params (M) | 논문 값 | (가) 기본 | +(a) | +(b) | +(c) | +(d) | 합계 | 그중 backbone |
|---|---|---|---|---|---|---|---|---|---|
| **S18** |  |  |  |  |  |  |  |  |  |
| ConvFormer-S18 | 36.158 | 925 (54) | 414.914 | 414.914 | 414.914 | 415.661 | 415.661 | 415.661 | 83.376 |
| SPANetV2-S18-pure | 41.651 | 930 (60) | 419.848 | 420.741 | 420.757 | 421.456 | 421.472 | 421.472 | 89.187 |
| CAFormer-S18 | 35.725 | 1024 (54) | 513.462 | 513.462 | 513.462 | 517.249 | 517.249 | 517.249 | 184.964 |
| SPANetV2-S18-hybrid | 39.480 | 925 (58) | 514.915 | 515.494 | 515.503 | 519.274 | 519.283 | 519.283 | 186.998 |
| **S36** |  |  |  |  |  |  |  |  |  |
| ConvFormer-S36 | 49.396 | 1003 (67) | 492.516 | 492.516 | 492.516 | 493.900 | 493.900 | 493.900 | 161.615 |
| SPANetV2-S36-pure | 56.587 | 1012 (75) | 502.444 | 504.149 | 504.181 | 505.471 | 505.503 | 505.503 | 173.218 |
| CAFormer-S36 | 48.681 | 1197 (67) | 686.545 | 686.545 | 686.545 | 693.918 | 693.918 | 693.918 | 361.633 |
| SPANetV2-S36-hybrid | 53.157 | 1004 (71) | 690.255 | 691.363 | 691.382 | 698.722 | 698.741 | 698.741 | 366.456 |

- get_flops.py처럼 auxiliary head를 떼고 쟀다. Params에도 들어 있지 않다 (auxiliary head 파라미터, M: ConvFormer-S18 0.776, SPANetV2-S18-pure 0.776, CAFormer-S18 0.776, SPANetV2-S18-hybrid 0.776, ConvFormer-S36 0.776, SPANetV2-S36-pure 0.776, CAFormer-S36 0.776, SPANetV2-S36-hybrid 0.776).
- Params에는 forward에서 한 번도 호출되지 않는 모듈의 파라미터도 들어 있다(측정 도구가 세는 그대로): SPANetV2-S18-pure `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M; SPANetV2-S18-hybrid `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M; SPANetV2-S36-pure `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M; SPANetV2-S36-hybrid `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M.

### ADE20K Semantic FPN — 입력 1×3×512×512

| 모델 | Params (M) | 논문 값 | (가) 기본 | +(a) | +(b) | +(c) | +(d) | 합계 | 그중 backbone |
|---|---|---|---|---|---|---|---|---|---|
| **S18** |  |  |  |  |  |  |  |  |  |
| SPANet-S | 32.360 | 46 (32) | 45.693 | 45.948 | 45.958 | 46.076 | 46.076 | 46.076 | 24.550 |
| DFFormer-S18 | 31.724 | 41 (35) | 41.345 | 41.727 | 41.736 | 41.993 | 41.993 | 41.993 | 20.467 |
| SPANetV2-S18-pure | 33.431 | 43 (33) | 43.412 | 43.603 | 43.607 | 43.787 | 43.791 | 43.791 | 22.265 |
| CDFFormer-S18 | 31.391 | 42 (35) | 41.788 | 41.919 | 41.922 | 42.179 | 42.179 | 42.179 | 20.653 |
| SPANetV2-S18-hybrid | 31.260 | 48 (31) | 48.455 | 48.581 | 48.583 | 48.946 | 48.948 | 48.948 | 27.422 |
| **S36** |  |  |  |  |  |  |  |  |  |
| SPANet-M | 45.442 | 57 (45) | 57.172 | 57.554 | 57.570 | 57.740 | 57.740 | 57.740 | 36.214 |
| DFFormer-S36 | 47.163 | 60 (50) | 59.952 | 60.679 | 60.695 | 61.173 | 61.173 | 61.173 | 39.647 |
| SPANetV2-S36-pure | 48.366 | 64 (48) | 64.061 | 64.424 | 64.432 | 64.760 | 64.768 | 64.768 | 43.242 |
| CDFFormer-S36 | 46.460 | 61 (50) | 60.838 | 61.086 | 61.093 | 61.571 | 61.571 | 61.571 | 40.045 |
| SPANetV2-S36-hybrid | 44.937 | 74 (45) | 74.171 | 74.410 | 74.415 | 75.103 | 75.108 | 75.108 | 53.582 |

- get_flops.py처럼 auxiliary head를 떼고 쟀다. Params에도 들어 있지 않다.
- Params에는 forward에서 한 번도 호출되지 않는 모듈의 파라미터도 들어 있다(측정 도구가 세는 그대로): SPANetV2-S18-pure `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M; SPANetV2-S18-hybrid `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M; SPANetV2-S36-pure `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M; SPANetV2-S36-hybrid `backbone.head` (MlpHead) 3.104M, `backbone.norm` (LayerNorm) 0.001M.

### ImageNet-1K 384×384 (분류) — 입력 1×3×384×384

| 모델 | Params (M) | 논문 값 | (가) 기본 | +(a) | +(b) | +(c) | +(d) | 합계 | 그중 backbone |
|---|---|---|---|---|---|---|---|---|---|
| **S18** |  |  |  |  |  |  |  |  |  |
| ConvFormer-S18 | 26.774 | 11.6 | 11.621 | 11.621 | 11.621 | 11.722 | 11.722 | 11.722 | - |
| SPANetV2-S18-pure | 29.163 | 12.3 | 12.315 | 12.422 | 12.425 | 12.519 | 12.522 | 12.522 | - |
| CAFormer-S18 | 26.342 | 13.4 | 13.412 | 13.412 | 13.412 | 13.563 | 13.563 | 13.563 | - |
| SPANetV2-S18-hybrid | 26.992 | 13.6 | 13.616 | 13.687 | 13.688 | 13.837 | 13.839 | 13.839 | - |
| **S36** |  |  |  |  |  |  |  |  |  |
| ConvFormer-S36 | 40.012 | 22.4 | 22.534 | 22.534 | 22.534 | 22.725 | 22.725 | 22.725 | - |
| SPANetV2-S36-pure | 44.099 | 23.9 | 23.930 | 24.134 | 24.139 | 24.317 | 24.321 | 24.321 | - |
| CAFormer-S36 | 39.297 | 26.0 | 26.073 | 26.073 | 26.073 | 26.364 | 26.364 | 26.364 | - |
| SPANetV2-S36-hybrid | 40.669 | 26.6 | 26.595 | 26.729 | 26.732 | 27.018 | 27.021 | 27.021 | - |

- 분류 모델 전체를 1×3×384×384로 쟀다. SPANetV2는 train_384.py처럼 기본 input_size(224)로 만들었다. "그중 backbone" 열은 해당하지 않는다.

## 1. 환경

표의 모든 값 (통합 측정, `measure_dense.py`, conda env `spanetv2-dense`):

| 항목 | 값 |
|---|---|
| 날짜 | 2026-11-20 19:18:04 KST |
| Python | 3.10.22 |
| PyTorch | 2.1.0+cu121 |
| mmengine | 0.10.7 |
| mmcv | 2.1.0 |
| mmdet | 3.3.0 (../../object_detection/mmdet) |
| mmseg | 1.2.2 (../../semantic_segmentation/mmseg) |
| fvcore | 0.1.5.post20221221 |
| timm | 0.6.13 |
| NumPy | 1.25.2 |
| 장치 | cpu (AMD Ryzen Threadripper PRO 5955WX 16-Cores, torch threads 16) |
| OS | Linux-6.17.0-35-generic-x86_64-with-glibc2.39 |
| Python 경로 | `/home/server/miniforge3/envs/spanetv2-dense/bin/python` |

2절의 저장소 도구 실행 (`repo_get_flops.py`, GPU 0):

| conda env | Python | PyTorch | mmcv | mmengine | mmdet | mmseg | timm | NumPy | CUDA | GPU, 드라이버 |
|---|---|---|---|---|---|---|---|---|---|---|
| spanetv2-dense | 3.10.22 | 2.1.0+cu121 | 2.1.0 | 0.10.7 | 3.3.0 (저장소 사본) | 1.2.2 (저장소 사본) | 0.6.13 | 1.25.2 | 12.1 | NVIDIA GeForce RTX 4090, 590.48.01 |
| dfformer-mm0 | 3.10.22 | 1.12.1+cu116 | 1.6.2 | - | - | 0.29.1 | 0.5.4 | 1.23.5 | 11.6 | NVIDIA GeForce RTX 4090, 590.48.01 |

- `dfformer-mm0`은 DFFormer의 requirements.txt 버전(torch 1.12.1, mmcv-full 1.6.2, mmsegmentation 0.29.1, timm 0.5.4)이다. DFFormer의 get_flops.py만 이 env에서 돌렸다. mmcv 1.x와 mmseg 0.x가 필요하기 때문이다.
- `spanetv2-dense`에서 mmdet·mmseg는 설치하지 않았다. 이 저장소에 들어 있는 사본(object_detection/mmdet 3.3.0, semantic_segmentation/mmseg 1.2.2)을 PYTHONPATH로 썼다. 버전은 object_detection·semantic_segmentation의 pyproject.toml과 INSTALL.md를 따랐다(torch 2.1.0 cu121, mmcv 2.1.0, timm 0.6.13, numpy<1.26). Python은 pyproject.toml의 3.10이다.

저장소 커밋 (측정 시점):

| 저장소 | 커밋 | remote | 변경 (추적 파일) |
|---|---|---|---|
| SPANetV2-official | `d1713e5c51e0172205536e8cfa08e8ebf28a0727` | https://github.com/DoranLyong/SPANetV2-official.git | clean |
| metaformer | `64c534ab8f65105fd21713b38e3393a449ca742d` | https://github.com/sail-sg/metaformer.git | clean |
| dfformer | `2b757c511a1bf1e66a12d494f3b0c42daa5b3636` | https://github.com/okojoalg/dfformer.git | clean |
| SPANet-official | `886ad24d4d69495401e93e90d2b0a52c6cb718b7` | https://github.com/DoranLyong/SPANet-official.git | M object_detection/mmdet/models/backbones/spanet.py |

- SPANetV2-official의 커밋은 이 폴더(flops_benchmark/)를 더한 d1713e5다. object_detection/, semantic_segmentation/, image_classification/models/spanetv2.py는 지시의 1b2cfee와 같다 (`git diff --stat 1b2cfee HEAD -- ...` 결과 없음).
- SPANet-official의 변경 한 건은 아래 import 수정이다.

### import 수정

계산 코드는 고치지 않았다. import가 깨진 파일은 하나다. SPANet-official/object_detection/mmdet/models/backbones/spanet.py(mmcv 1.x·mmdet 2.x용)이며, import 줄만 mmcv 2.x·mmengine·mmseg 1.x 경로로 바꿨다. `get_root_logger`는 mmdet 2.x의 함수 이름이다. mmengine에 같은 일을 하는 `MMLogger.get_current_instance`를 같은 이름에 묶었다. 이 이름은 사전 학습 가중치를 읽는 `init_weights` 안에서만 쓰이고, 측정에서는 불리지 않는다. 수정본은 `patches/SPANet_det_backbone_mm2_imports.patch`이고, `setup.sh`가 적용한다.

```diff
diff --git a/object_detection/mmdet/models/backbones/spanet.py b/object_detection/mmdet/models/backbones/spanet.py
index db08529..d7cc77e 100644
--- a/object_detection/mmdet/models/backbones/spanet.py
+++ b/object_detection/mmdet/models/backbones/spanet.py
@@ -28,13 +28,15 @@ from torch.nn.modules.utils import _pair as to_2tuple
 
 
 from mmcv.cnn import build_norm_layer
-from mmcv.runner import BaseModule, _load_checkpoint
+from mmengine.model import BaseModule
+from mmengine.runner.checkpoint import _load_checkpoint
 from mmcv.cnn.bricks import DropPath
-from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
+from mmengine.model.weight_init import (constant_init, normal_init,
                                         trunc_normal_init)
 
-from mmdet.models.builder import BACKBONES
-from mmdet.utils import get_root_logger
+from mmseg.models.builder import BACKBONES
+from mmengine.logging import MMLogger
+get_root_logger = MMLogger.get_current_instance
 
 
 class PatchEmbed(BaseModule):
```

- DFFormer의 segmentation/backbone/{dfformer,cdfformer}.py는 고치지 않고 import했다 (`from mmseg.models.builder import BACKBONES`가 mmseg 1.2.2에도 있다).
- metaformer_baselines.py는 고치지 않았다(감싼 코드는 6절).

## 2. 기존 방식 재현 — 각 저장소의 get_flops.py

각 저장소의 스크립트를 고치지 않고 돌렸다(`run_step2.sh` → `repo_get_flops.py`). launcher가 더한 것은 다음뿐이다.
- 난수 시드 0.
- config 덮어쓰기 `model.backbone.init_cfg=None`. config의 체크포인트 파일이 이 서버에 없고, 가중치는 필요 없다.
- 함수 감싸기. 스크립트가 부르는 `get_model_complexity_info`의 반환값(반올림 전 정수)을 기록한다.
- 검출기 forward hook. RoI 추출기에 들어간 RoI 수를 기록한다.

도구마다 따로 적을 점:
- **검출(SPANetV2 mmdet 3.x):** get_flops.py에는 입력 크기 인자가 없다. val dataloader로 COCO val2017 이미지를 읽어 `--num-images`장의 평균을 낸다. 그래서 test pipeline의 Resize를 `scale=(800, 1280), keep_ratio=False`로 바꿔, 모든 이미지가 1280×800(H×W)으로 들어가게 했다. 이미지 3장을 썼다(`--num-images 3`). 모델은 mmengine의 `tensor` 모드로 돈다. RPN이 실제로 낸 proposal이 RoI head에 들어가므로 RoI 수가 입력과 가중치에 따라 달라질 수 있다.
- **분할(SPANetV2 mmseg 1.x):** `--shape 512 2048`(UperNet), `--shape 512 512`(Semantic FPN). ConvFormer, CAFormer, SPANet의 새로 만든 config(6절)도 이 스크립트로 쟀다.
- **DFFormer(mmseg 0.x):** DFFormer 저장소의 segmentation/tools/get_flops.py는 mmcv 1.x의 `mmcv.cnn.get_model_complexity_info`를 쓴다. 이 함수는 trace가 아니라 module hook으로 센다. 모듈 종류별로 정해진 수식만 쓰고, 함수 호출(matmul, FFT, einsum 등)과 SyncBatchNorm은 세지 않는다. 이 env의 torch 1.12.1+cu116 cuFFT가 RTX 4090에서 `CUFFT_INTERNAL_ERROR`를 낸다. 그래서 모델은 스크립트대로 GPU에 두고, `torch.fft.rfft2/irfft2` 두 함수만 CPU에서 계산했다. 이 카운터는 module만 세므로 값은 같다.

### COCO Mask R-CNN 3×

| 모델 | 도구 | 출력 (그대로) | 반환값 (G) | Params (M) | 논문 값 | 논문과 차이 | 통합 측정 (가) | 이미지별 반환값 (G) | RoI: bbox head / mask head (단계마다) |
|---|---|---|---|---|---|---|---|---|---|
| ConvFormer-S18 | object_detection/tools/analysis_tools/get_flops.py (mmengine) | Use size divisor set input shape from (428, 640) to (1280, 800)<br>Input shape: (1280, 800)<br>Flops: 0.251T<br>Params: 43.838M | 250.814 | 43.838 | 251 | -0.186 (-0.07%) | 250.814 | 250.814, 250.814, 250.814 | 1000 / 100 |
| SPANetV2-S18-pure | object_detection/tools/analysis_tools/get_flops.py (mmengine) | Use size divisor set input shape from (428, 640) to (1280, 800)<br>Input shape: (1280, 800)<br>Flops: 0.256T<br>Params: 49.331M | 255.631 | 49.331 | 255 | +0.631 (+0.25%) | 255.631 | 255.631, 255.631, 255.631 | 1000 / 100 |
| CAFormer-S18 | object_detection/tools/analysis_tools/get_flops.py (mmengine) | Use size divisor set input shape from (428, 640) to (1280, 800)<br>Input shape: (1280, 800)<br>Flops: 0.345T<br>Params: 43.405M | 344.766 | 43.405 | 254 | +90.766 (+35.73%) | 344.766 | 344.766, 344.766, 344.766 | 1000 / 100 |
| SPANetV2-S18-hybrid | object_detection/tools/analysis_tools/get_flops.py (mmengine) | Use size divisor set input shape from (428, 640) to (1280, 800)<br>Input shape: (1280, 800)<br>Flops: 0.346T<br>Params: 47.16M | 346.185 | 47.160 | 251 | +95.185 (+37.92%) | 346.185 | 346.185, 346.185, 346.185 | 1000 / 100 |

### COCO Cascade Mask R-CNN 3×

| 모델 | 도구 | 출력 (그대로) | 반환값 (G) | Params (M) | 논문 값 | 논문과 차이 | 통합 측정 (가) | 이미지별 반환값 (G) | RoI: bbox head / mask head (단계마다) |
|---|---|---|---|---|---|---|---|---|---|
| ConvFormer-S18 | object_detection/tools/analysis_tools/get_flops.py (mmengine) | Use size divisor set input shape from (428, 640) to (1280, 800)<br>Input shape: (1280, 800)<br>Flops: 2.16T<br>Params: 81.673M | 2159.984 | 81.673 | 729 | +1430.984 (+196.29%) | 729.366 | 2159.984, 2159.984, 2159.984 | 1000 ×3 / 1000 ×3 |
| SPANetV2-S18-pure | object_detection/tools/analysis_tools/get_flops.py (mmengine) | Use size divisor set input shape from (428, 640) to (1280, 800)<br>Input shape: (1280, 800)<br>Flops: 2.165T<br>Params: 87.167M | 2164.802 | 87.167 | 734 | +1430.802 (+194.93%) | 734.183 | 2164.802, 2164.802, 2164.802 | 1000 ×3 / 1000 ×3 |
| CAFormer-S18 | object_detection/tools/analysis_tools/get_flops.py (mmengine) | Use size divisor set input shape from (428, 640) to (1280, 800)<br>Input shape: (1280, 800)<br>Flops: 2.254T<br>Params: 81.241M | 2253.936 | 81.241 | 733 | +1520.936 (+207.49%) | 823.318 | 2253.936, 2253.936, 2253.936 | 1000 ×3 / 1000 ×3 |
| SPANetV2-S18-hybrid | object_detection/tools/analysis_tools/get_flops.py (mmengine) | Use size divisor set input shape from (428, 640) to (1280, 800)<br>Input shape: (1280, 800)<br>Flops: 2.255T<br>Params: 84.996M | 2255.355 | 84.996 | 729 | +1526.355 (+209.38%) | 824.737 | 2255.355, 2255.355, 2255.355 | 1000 ×3 / 1000 ×3 |
| ConvFormer-S36 | object_detection/tools/analysis_tools/get_flops.py (mmengine) | Use size divisor set input shape from (428, 640) to (1280, 800)<br>Input shape: (1280, 800)<br>Flops: 2.236T<br>Params: 94.911M | 2235.767 | 94.911 | 805 | +1430.767 (+177.74%) | 805.149 | 2235.767, 2235.767, 2235.767 | 1000 ×3 / 1000 ×3 |
| SPANetV2-S36-pure | object_detection/tools/analysis_tools/get_flops.py (mmengine) | Use size divisor set input shape from (428, 640) to (1280, 800)<br>Input shape: (1280, 800)<br>Flops: 2.245T<br>Params: 0.102G | 2245.462 | 102.102 | 814 | +1431.462 (+175.86%) | 814.844 | 2245.462, 2245.462, 2245.462 | 1000 ×3 / 1000 ×3 |
| CAFormer-S36 | object_detection/tools/analysis_tools/get_flops.py (mmengine)<br>(GPU 메모리 부족 → CPU에서 다시 실행) | Use size divisor set input shape from (428, 640) to (1280, 800)<br>Input shape: (1280, 800)<br>Flops: 2.421T<br>Params: 94.196M | 2420.751 | 94.196 | 811 | +1609.751 (+198.49%) | 990.132 | 2420.751, 2420.751, 2420.751 | 1000 ×3 / 1000 ×3 |
| SPANetV2-S36-hybrid | object_detection/tools/analysis_tools/get_flops.py (mmengine)<br>(GPU 메모리 부족 → CPU에서 다시 실행) | Use size divisor set input shape from (428, 640) to (1280, 800)<br>Input shape: (1280, 800)<br>Flops: 2.424T<br>Params: 98.673M | 2424.374 | 98.673 | 806 | +1618.374 (+200.79%) | 993.756 | 2424.374, 2424.374, 2424.374 | 1000 ×3 / 1000 ×3 |

### ADE20K UperNet

| 모델 | 도구 | 출력 (그대로) | 반환값 (G) | Params (M) | 논문 값 | 논문과 차이 | 통합 측정 (가) |
|---|---|---|---|---|---|---|---|
| ConvFormer-S18 | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 2048)<br>Flops: 0.415T<br>Params: 36.158M | 414.914 | 36.158 | 925 | -510.086 (-55.14%) | 414.914 |
| SPANetV2-S18-pure | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 2048)<br>Flops: 0.42T<br>Params: 41.651M | 419.848 | 41.651 | 930 | -510.152 (-54.86%) | 419.848 |
| CAFormer-S18 | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 2048)<br>Flops: 0.513T<br>Params: 35.725M | 513.462 | 35.725 | 1024 | -510.538 (-49.86%) | 513.462 |
| SPANetV2-S18-hybrid | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 2048)<br>Flops: 0.515T<br>Params: 39.48M | 514.915 | 39.480 | 925 | -410.085 (-44.33%) | 514.915 |
| ConvFormer-S36 | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 2048)<br>Flops: 0.493T<br>Params: 49.396M | 492.516 | 49.396 | 1003 | -510.484 (-50.90%) | 492.516 |
| SPANetV2-S36-pure | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 2048)<br>Flops: 0.502T<br>Params: 56.587M | 502.444 | 56.587 | 1012 | -509.556 (-50.35%) | 502.444 |
| CAFormer-S36 | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 2048)<br>Flops: 0.687T<br>Params: 48.681M | 686.545 | 48.681 | 1197 | -510.455 (-42.64%) | 686.545 |
| SPANetV2-S36-hybrid | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 2048)<br>Flops: 0.69T<br>Params: 53.157M | 690.255 | 53.157 | 1004 | -313.745 (-31.25%) | 690.255 |

### ADE20K Semantic FPN

| 모델 | 도구 | 출력 (그대로) | 반환값 (G) | Params (M) | 논문 값 | 논문과 차이 | 통합 측정 (가) |
|---|---|---|---|---|---|---|---|
| SPANet-S | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 512)<br>Flops: 45.693G<br>Params: 32.36M | 45.693 | 32.360 | 46 | -0.307 (-0.67%) | 45.693 |
| DFFormer-S18 | dfformer/segmentation/tools/get_flops.py (mmcv 1.x) | Input shape: (3, 512, 512)<br>Flops: 41.3 GFLOPs<br>Params: 31.72 M | 41.305 | 31.724 | 41 | +0.305 (+0.74%) | 41.345 |
| SPANetV2-S18-pure | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 512)<br>Flops: 43.412G<br>Params: 33.431M | 43.412 | 33.431 | 43 | +0.412 (+0.96%) | 43.412 |
| CDFFormer-S18 | dfformer/segmentation/tools/get_flops.py (mmcv 1.x) | Input shape: (3, 512, 512)<br>Flops: 41.77 GFLOPs<br>Params: 31.39 M | 41.767 | 31.391 | 42 | -0.233 (-0.55%) | 41.788 |
| SPANetV2-S18-hybrid | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 512)<br>Flops: 48.455G<br>Params: 31.26M | 48.455 | 31.260 | 48 | +0.455 (+0.95%) | 48.455 |
| SPANet-M | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 512)<br>Flops: 57.172G<br>Params: 45.442M | 57.172 | 45.442 | 57 | +0.172 (+0.30%) | 57.172 |
| DFFormer-S36 | dfformer/segmentation/tools/get_flops.py (mmcv 1.x) | Input shape: (3, 512, 512)<br>Flops: 59.93 GFLOPs<br>Params: 47.16 M | 59.926 | 47.163 | 60 | -0.074 (-0.12%) | 59.952 |
| SPANetV2-S36-pure | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 512)<br>Flops: 64.061G<br>Params: 48.366M | 64.061 | 48.366 | 64 | +0.061 (+0.09%) | 64.061 |
| CDFFormer-S36 | dfformer/segmentation/tools/get_flops.py (mmcv 1.x) | Input shape: (3, 512, 512)<br>Flops: 60.85 GFLOPs<br>Params: 46.46 M | 60.850 | 46.460 | 61 | -0.150 (-0.25%) | 60.838 |
| SPANetV2-S36-hybrid | semantic_segmentation/tools/analysis_tools/get_flops.py (mmengine) | Input shape: (512, 512)<br>Flops: 74.171G<br>Params: 44.937M | 74.171 | 44.937 | 74 | +0.171 (+0.23%) | 74.171 |

- "반환값"은 카운터가 돌려준 정수를 10^9로 나눈 값이다(검출은 이미지 평균). "통합 측정 (가)"는 결과 표의 (가)다. 검출에서는 RoI 설정이 다르다(아래).
- 결과 표의 검출 값은 모든 모델에 같은 RoI 설정으로 쟀다. RPN 3×3 conv까지 세고, bbox head마다 RoI 1000개, mask head마다 그중 앞의 100개를 넣는다. mmdet 2.x `forward_dummy`와 같은 구조다. Mask R-CNN에서는 mmdet 3.x get_flops.py와 같다(StandardRoIHead의 tensor 모드가 mask head에 `rois[:100]`을 넣는다). Cascade Mask R-CNN에서는 다르다. mmdet 3.x의 CascadeRoIHead tensor 모드는 세 단계 mask head에 모든 RoI(위 표의 수)를 넣는다.

## 3. 빠진 연산 (unsupported ops)

mmengine 기본 분석에서 처리 함수가 없던 연산이다. 전체 모델 기준이다. 모듈 이름은 trace의 scope이고, 같은 모양의 모듈은 번호를 `*`로 묶었다.

<details><summary>ConvFormer-S18 — COCO Mask R-CNN 3×</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::add': 81, 'aten::mul': 66, 'aten::pow': 40, 'aten::mean': 8, 'aten::sub': 8, 'prim::PythonOp.RoIAlignFunction': 8, 'aten::sqrt': 6, 'aten::div': 6, 'aten::log2': 2, 'aten::max_pool2d': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::add | 3 | `neck` | FPN |
| aten::add | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::add | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::div | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::log2 | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::log2 | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::max_pool2d | 1 | `neck` | FPN |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::mul | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::mul | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::sqrt | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 2 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::sub | 2 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| prim::PythonOp.RoIAlignFunction | 4 | `roi_head.bbox_roi_extractor.roi_layers.*` | RoIAlign |
| prim::PythonOp.RoIAlignFunction | 4 | `roi_head.mask_roi_extractor.roi_layers.*` | RoIAlign |

</details>

<details><summary>SPANetV2-S18-pure — COCO Mask R-CNN 3×</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 138, 'aten::add': 135, 'aten::upsample_bicubic2d': 72, 'aten::sigmoid': 72, 'aten::fft_rfft2': 72, 'aten::fft_irfft2': 72, 'aten::gelu': 36, 'aten::pow': 22, 'aten::mean': 8, 'aten::sub': 8, 'prim::PythonOp.RoIAlignFunction': 8, 'aten::sqrt': 6, 'aten::div': 6, 'aten::log2': 2, 'aten::max_pool2d': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::add | 3 | `neck` | FPN |
| aten::add | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::add | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::div | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::fft_irfft2 | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 18 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 18 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::log2 | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::log2 | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::max_pool2d | 1 | `neck` | FPN |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 90 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::mul | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::mul | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::sqrt | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 2 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::sub | 2 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::upsample_bicubic2d | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| prim::PythonOp.RoIAlignFunction | 4 | `roi_head.bbox_roi_extractor.roi_layers.*` | RoIAlign |
| prim::PythonOp.RoIAlignFunction | 4 | `roi_head.mask_roi_extractor.roi_layers.*` | RoIAlign |

</details>

<details><summary>CAFormer-S18 — COCO Mask R-CNN 3×</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 78, 'aten::add': 69, 'aten::pow': 28, 'aten::softmax': 12, 'aten::mean': 8, 'aten::sub': 8, 'prim::PythonOp.RoIAlignFunction': 8, 'aten::sqrt': 6, 'aten::div': 6, 'aten::log2': 2, 'aten::max_pool2d': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 6 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::add | 3 | `neck` | FPN |
| aten::add | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::add | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::div | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::log2 | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::log2 | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::max_pool2d | 1 | `neck` | FPN |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 24 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::mul | 6 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::mul | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::mul | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 6 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::softmax | 12 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::sqrt | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 2 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::sub | 2 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| prim::PythonOp.RoIAlignFunction | 4 | `roi_head.bbox_roi_extractor.roi_layers.*` | RoIAlign |
| prim::PythonOp.RoIAlignFunction | 4 | `roi_head.mask_roi_extractor.roi_layers.*` | RoIAlign |

</details>

<details><summary>SPANetV2-S18-hybrid — COCO Mask R-CNN 3×</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 102, 'aten::add': 96, 'aten::upsample_bicubic2d': 24, 'aten::sigmoid': 24, 'aten::fft_rfft2': 24, 'aten::fft_irfft2': 24, 'aten::pow': 22, 'aten::gelu': 12, 'aten::softmax': 12, 'aten::mean': 8, 'aten::sub': 8, 'prim::PythonOp.RoIAlignFunction': 8, 'aten::sqrt': 6, 'aten::div': 6, 'aten::log2': 2, 'aten::max_pool2d': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::add | 9 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::add | 3 | `neck` | FPN |
| aten::add | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::add | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::div | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::fft_irfft2 | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 6 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 6 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::log2 | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::log2 | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::max_pool2d | 1 | `neck` | FPN |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 30 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::mul | 18 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::mul | 6 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::mul | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::mul | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::softmax | 9 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::softmax | 3 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 1 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::sqrt | 1 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 2 | `roi_head.bbox_roi_extractor` | SingleRoIExtractor |
| aten::sub | 2 | `roi_head.mask_roi_extractor` | SingleRoIExtractor |
| aten::upsample_bicubic2d | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| prim::PythonOp.RoIAlignFunction | 4 | `roi_head.bbox_roi_extractor.roi_layers.*` | RoIAlign |
| prim::PythonOp.RoIAlignFunction | 4 | `roi_head.mask_roi_extractor.roi_layers.*` | RoIAlign |

</details>

<details><summary>ConvFormer-S18 — COCO Cascade Mask R-CNN 3×</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::add': 85, 'aten::mul': 70, 'aten::pow': 40, 'prim::PythonOp.RoIAlignFunction': 24, 'aten::sub': 16, 'aten::sqrt': 10, 'aten::div': 10, 'aten::mean': 8, 'aten::log2': 6, 'aten::max_pool2d': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::add | 3 | `neck` | FPN |
| aten::add | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::add | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::log2 | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::log2 | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::max_pool2d | 1 | `neck` | FPN |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::mul | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::mul | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sqrt | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 6 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 6 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.bbox_roi_extractor.*.roi_layers.*` | RoIAlign |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.mask_roi_extractor.*.roi_layers.*` | RoIAlign |

</details>

<details><summary>SPANetV2-S18-pure — COCO Cascade Mask R-CNN 3×</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 142, 'aten::add': 139, 'aten::upsample_bicubic2d': 72, 'aten::sigmoid': 72, 'aten::fft_rfft2': 72, 'aten::fft_irfft2': 72, 'aten::gelu': 36, 'prim::PythonOp.RoIAlignFunction': 24, 'aten::pow': 22, 'aten::sub': 16, 'aten::sqrt': 10, 'aten::div': 10, 'aten::mean': 8, 'aten::log2': 6, 'aten::max_pool2d': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::add | 3 | `neck` | FPN |
| aten::add | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::add | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::fft_irfft2 | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 18 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 18 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::log2 | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::log2 | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::max_pool2d | 1 | `neck` | FPN |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 90 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::mul | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::mul | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sqrt | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 6 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 6 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::upsample_bicubic2d | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.bbox_roi_extractor.*.roi_layers.*` | RoIAlign |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.mask_roi_extractor.*.roi_layers.*` | RoIAlign |

</details>

<details><summary>CAFormer-S18 — COCO Cascade Mask R-CNN 3×</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 82, 'aten::add': 73, 'aten::pow': 28, 'prim::PythonOp.RoIAlignFunction': 24, 'aten::sub': 16, 'aten::softmax': 12, 'aten::sqrt': 10, 'aten::div': 10, 'aten::mean': 8, 'aten::log2': 6, 'aten::max_pool2d': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 6 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::add | 3 | `neck` | FPN |
| aten::add | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::add | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::log2 | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::log2 | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::max_pool2d | 1 | `neck` | FPN |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 24 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::mul | 6 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::mul | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::mul | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 6 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::softmax | 12 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sqrt | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 6 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 6 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.bbox_roi_extractor.*.roi_layers.*` | RoIAlign |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.mask_roi_extractor.*.roi_layers.*` | RoIAlign |

</details>

<details><summary>SPANetV2-S18-hybrid — COCO Cascade Mask R-CNN 3×</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 106, 'aten::add': 100, 'aten::upsample_bicubic2d': 24, 'aten::sigmoid': 24, 'aten::fft_rfft2': 24, 'aten::fft_irfft2': 24, 'prim::PythonOp.RoIAlignFunction': 24, 'aten::pow': 22, 'aten::sub': 16, 'aten::gelu': 12, 'aten::softmax': 12, 'aten::sqrt': 10, 'aten::div': 10, 'aten::mean': 8, 'aten::log2': 6, 'aten::max_pool2d': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::add | 9 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::add | 3 | `neck` | FPN |
| aten::add | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::add | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::fft_irfft2 | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 6 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 6 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::log2 | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::log2 | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::max_pool2d | 1 | `neck` | FPN |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 30 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::mul | 18 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::mul | 6 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::mul | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::mul | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::softmax | 9 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::softmax | 3 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sqrt | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 6 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 6 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::upsample_bicubic2d | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.bbox_roi_extractor.*.roi_layers.*` | RoIAlign |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.mask_roi_extractor.*.roi_layers.*` | RoIAlign |

</details>

<details><summary>ConvFormer-S36 — COCO Cascade Mask R-CNN 3×</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::add': 157, 'aten::mul': 124, 'aten::pow': 76, 'prim::PythonOp.RoIAlignFunction': 24, 'aten::sub': 16, 'aten::sqrt': 10, 'aten::div': 10, 'aten::mean': 8, 'aten::log2': 6, 'aten::max_pool2d': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 36 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::add | 3 | `neck` | FPN |
| aten::add | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::add | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::log2 | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::log2 | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::max_pool2d | 1 | `neck` | FPN |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 36 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::mul | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::mul | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 36 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sqrt | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 6 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 6 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.bbox_roi_extractor.*.roi_layers.*` | RoIAlign |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.mask_roi_extractor.*.roi_layers.*` | RoIAlign |

</details>

<details><summary>SPANetV2-S36-pure — COCO Cascade Mask R-CNN 3×</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 268, 'aten::add': 265, 'aten::upsample_bicubic2d': 144, 'aten::sigmoid': 144, 'aten::fft_rfft2': 144, 'aten::fft_irfft2': 144, 'aten::gelu': 72, 'aten::pow': 40, 'prim::PythonOp.RoIAlignFunction': 24, 'aten::sub': 16, 'aten::sqrt': 10, 'aten::div': 10, 'aten::mean': 8, 'aten::log2': 6, 'aten::max_pool2d': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::add | 3 | `neck` | FPN |
| aten::add | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::add | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::fft_irfft2 | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 36 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 36 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::log2 | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::log2 | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::max_pool2d | 1 | `neck` | FPN |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 180 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::mul | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::mul | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sqrt | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 6 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 6 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::upsample_bicubic2d | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.bbox_roi_extractor.*.roi_layers.*` | RoIAlign |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.mask_roi_extractor.*.roi_layers.*` | RoIAlign |

</details>

<details><summary>CAFormer-S36 — COCO Cascade Mask R-CNN 3×</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 145, 'aten::add': 136, 'aten::pow': 55, 'prim::PythonOp.RoIAlignFunction': 24, 'aten::softmax': 21, 'aten::sub': 16, 'aten::sqrt': 10, 'aten::div': 10, 'aten::mean': 8, 'aten::log2': 6, 'aten::max_pool2d': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 15 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::add | 3 | `neck` | FPN |
| aten::add | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::add | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::log2 | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::log2 | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::max_pool2d | 1 | `neck` | FPN |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 42 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::mul | 15 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::mul | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::mul | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 15 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::softmax | 21 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sqrt | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 6 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 6 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.bbox_roi_extractor.*.roi_layers.*` | RoIAlign |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.mask_roi_extractor.*.roi_layers.*` | RoIAlign |

</details>

<details><summary>SPANetV2-S36-hybrid — COCO Cascade Mask R-CNN 3×</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 205, 'aten::add': 199, 'aten::upsample_bicubic2d': 60, 'aten::sigmoid': 60, 'aten::fft_rfft2': 60, 'aten::fft_irfft2': 60, 'aten::pow': 40, 'aten::gelu': 30, 'prim::PythonOp.RoIAlignFunction': 24, 'aten::softmax': 21, 'aten::sub': 16, 'aten::sqrt': 10, 'aten::div': 10, 'aten::mean': 8, 'aten::log2': 6, 'aten::max_pool2d': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::add | 18 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::add | 3 | `neck` | FPN |
| aten::add | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::add | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::div | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::fft_irfft2 | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 15 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 15 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::log2 | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::log2 | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::max_pool2d | 1 | `neck` | FPN |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 75 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::mul | 36 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::mul | 6 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::mul | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::mul | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::softmax | 18 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::softmax | 3 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sqrt | 3 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 6 | `roi_head.bbox_roi_extractor.*` | SingleRoIExtractor |
| aten::sub | 6 | `roi_head.mask_roi_extractor.*` | SingleRoIExtractor |
| aten::upsample_bicubic2d | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.bbox_roi_extractor.*.roi_layers.*` | RoIAlign |
| prim::PythonOp.RoIAlignFunction | 12 | `roi_head.mask_roi_extractor.*.roi_layers.*` | RoIAlign |

</details>

<details><summary>ConvFormer-S18 — ADE20K UperNet</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::add': 79, 'aten::mul': 64, 'aten::pow': 40, 'aten::mean': 8, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |

</details>

<details><summary>SPANetV2-S18-pure — ADE20K UperNet</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 136, 'aten::add': 133, 'aten::upsample_bicubic2d': 72, 'aten::sigmoid': 72, 'aten::fft_rfft2': 72, 'aten::fft_irfft2': 72, 'aten::gelu': 36, 'aten::pow': 22, 'aten::mean': 8, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_irfft2 | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 18 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 18 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 90 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::upsample_bicubic2d | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |

</details>

<details><summary>CAFormer-S18 — ADE20K UperNet</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 76, 'aten::add': 67, 'aten::pow': 28, 'aten::softmax': 12, 'aten::mean': 8, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 6 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 24 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::mul | 6 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 6 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::softmax | 12 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |

</details>

<details><summary>SPANetV2-S18-hybrid — ADE20K UperNet</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 100, 'aten::add': 94, 'aten::upsample_bicubic2d': 24, 'aten::sigmoid': 24, 'aten::fft_rfft2': 24, 'aten::fft_irfft2': 24, 'aten::pow': 22, 'aten::gelu': 12, 'aten::softmax': 12, 'aten::mean': 8, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::add | 9 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_irfft2 | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 6 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 6 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 30 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::mul | 18 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::mul | 6 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::softmax | 9 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::softmax | 3 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::upsample_bicubic2d | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |

</details>

<details><summary>ConvFormer-S36 — ADE20K UperNet</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::add': 151, 'aten::mul': 118, 'aten::pow': 76, 'aten::mean': 8, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 36 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 36 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 36 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |

</details>

<details><summary>SPANetV2-S36-pure — ADE20K UperNet</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 262, 'aten::add': 259, 'aten::upsample_bicubic2d': 144, 'aten::sigmoid': 144, 'aten::fft_rfft2': 144, 'aten::fft_irfft2': 144, 'aten::gelu': 72, 'aten::pow': 40, 'aten::mean': 8, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_irfft2 | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 36 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 36 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 180 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::upsample_bicubic2d | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |

</details>

<details><summary>CAFormer-S36 — ADE20K UperNet</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 139, 'aten::add': 130, 'aten::pow': 55, 'aten::softmax': 21, 'aten::mean': 8, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 15 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 42 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::mul | 15 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 15 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::softmax | 21 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |

</details>

<details><summary>SPANetV2-S36-hybrid — ADE20K UperNet</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 199, 'aten::add': 193, 'aten::upsample_bicubic2d': 60, 'aten::sigmoid': 60, 'aten::fft_rfft2': 60, 'aten::fft_irfft2': 60, 'aten::pow': 40, 'aten::gelu': 30, 'aten::softmax': 21, 'aten::mean': 8, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::add | 18 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_irfft2 | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 15 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 15 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 75 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::mul | 36 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::mul | 6 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::softmax | 18 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::softmax | 3 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::upsample_bicubic2d | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |

</details>

<details><summary>SPANet-S — ADE20K Semantic FPN</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 128, 'aten::add': 102, 'aten::fft_fft2': 72, 'aten::fft_fftshift': 72, 'aten::fft_ifftshift': 72, 'aten::fft_ifft2': 72, 'aten::real': 72, 'aten::clone': 24, 'aten::gelu': 24, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 48 | `backbone.network.*.*` | SPANetBlock |
| aten::add | 48 | `backbone.network.*.*.token_mixer` | SPAttention |
| aten::add | 3 | `neck` | FPN |
| aten::clone | 24 | `backbone.network.*.*.token_mixer` | SPAttention |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_fft2 | 72 | `backbone.network.*.*.token_mixer.sps.*` | SpectralFilter |
| aten::fft_fftshift | 72 | `backbone.network.*.*.token_mixer.sps.*` | SpectralFilter |
| aten::fft_ifft2 | 72 | `backbone.network.*.*.token_mixer.sps.*` | SpectralFilter |
| aten::fft_ifftshift | 72 | `backbone.network.*.*.token_mixer.sps.*` | SpectralFilter |
| aten::gelu | 24 | `backbone.network.*.*.mlp.act` | GELU |
| aten::mul | 16 | `backbone.network.*.*.res_scale1` | Scale |
| aten::mul | 16 | `backbone.network.*.*.res_scale2` | Scale |
| aten::mul | 24 | `backbone.network.*.*.token_mixer` | SPAttention |
| aten::mul | 72 | `backbone.network.*.*.token_mixer.sps.*` | SpectralFilter |
| aten::real | 72 | `backbone.network.*.*.token_mixer.sps.*` | SpectralFilter |

</details>

<details><summary>DFFormer-S18 — ADE20K Semantic FPN</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 154, 'aten::add': 136, 'aten::mean': 98, 'aten::pow': 94, 'aten::sub': 40, 'aten::sqrt': 40, 'aten::div': 40, 'aten::softmax': 18, 'aten::fft_rfft2': 18, 'aten::upsample_bicubic2d': 18, 'aten::view_as_complex': 18, 'aten::fft_irfft2': 18, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 18 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::add | 18 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::add | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::add | 18 | `backbone.stages.*.*.token_mixer.reweight.act` | StarReLU |
| aten::add | 3 | `neck` | FPN |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 18 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::div | 18 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_irfft2 | 18 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::fft_rfft2 | 18 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mean | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::mean | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::mean | 18 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 18 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 36 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::mul | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::mul | 18 | `backbone.stages.*.*.token_mixer.reweight.act` | StarReLU |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 18 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::pow | 18 | `backbone.stages.*.*.token_mixer.reweight.act` | StarReLU |
| aten::softmax | 18 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 18 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::sqrt | 18 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 18 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::sub | 18 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::upsample_bicubic2d | 18 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::view_as_complex | 18 | `backbone.stages.*.*.token_mixer` | DynamicFilter |

</details>

<details><summary>SPANetV2-S18-pure — ADE20K Semantic FPN</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::add': 136, 'aten::mul': 136, 'aten::upsample_bicubic2d': 72, 'aten::sigmoid': 72, 'aten::fft_rfft2': 72, 'aten::fft_irfft2': 72, 'aten::gelu': 36, 'aten::pow': 22, 'aten::mean': 8, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::add | 3 | `neck` | FPN |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_irfft2 | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 18 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 18 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 90 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::upsample_bicubic2d | 72 | `backbone.stages.*.*.token_mixer` | ExSPAM |

</details>

<details><summary>CDFFormer-S18 — ADE20K Semantic FPN</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 136, 'aten::add': 130, 'aten::mean': 92, 'aten::pow': 88, 'aten::sub': 40, 'aten::sqrt': 40, 'aten::div': 40, 'aten::softmax': 12, 'aten::fft_rfft2': 12, 'aten::upsample_bicubic2d': 12, 'aten::view_as_complex': 12, 'aten::fft_irfft2': 12, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 18 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::add | 18 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::add | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::add | 12 | `backbone.stages.*.*.token_mixer.reweight.act` | StarReLU |
| aten::add | 3 | `neck` | FPN |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 18 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::div | 18 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_irfft2 | 12 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::fft_rfft2 | 12 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mean | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::mean | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::mean | 12 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 18 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 24 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::mul | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.token_mixer.reweight.act` | StarReLU |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 18 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::pow | 12 | `backbone.stages.*.*.token_mixer.reweight.act` | StarReLU |
| aten::softmax | 12 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 18 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::sqrt | 18 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 18 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::sub | 18 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::upsample_bicubic2d | 12 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::view_as_complex | 12 | `backbone.stages.*.*.token_mixer` | DynamicFilter |

</details>

<details><summary>SPANetV2-S18-hybrid — ADE20K Semantic FPN</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 100, 'aten::add': 97, 'aten::upsample_bicubic2d': 24, 'aten::sigmoid': 24, 'aten::fft_rfft2': 24, 'aten::fft_irfft2': 24, 'aten::pow': 22, 'aten::gelu': 12, 'aten::softmax': 12, 'aten::mean': 8, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::add | 9 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::add | 3 | `neck` | FPN |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_irfft2 | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 6 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 6 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 30 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::mul | 18 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::mul | 6 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::softmax | 9 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::softmax | 3 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::upsample_bicubic2d | 24 | `backbone.stages.*.*.token_mixer` | ExSPAM |

</details>

<details><summary>SPANet-M — ADE20K Semantic FPN</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 192, 'aten::add': 150, 'aten::fft_fft2': 108, 'aten::fft_fftshift': 108, 'aten::fft_ifftshift': 108, 'aten::fft_ifft2': 108, 'aten::real': 108, 'aten::clone': 36, 'aten::gelu': 36, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 72 | `backbone.network.*.*` | SPANetBlock |
| aten::add | 72 | `backbone.network.*.*.token_mixer` | SPAttention |
| aten::add | 3 | `neck` | FPN |
| aten::clone | 36 | `backbone.network.*.*.token_mixer` | SPAttention |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_fft2 | 108 | `backbone.network.*.*.token_mixer.sps.*` | SpectralFilter |
| aten::fft_fftshift | 108 | `backbone.network.*.*.token_mixer.sps.*` | SpectralFilter |
| aten::fft_ifft2 | 108 | `backbone.network.*.*.token_mixer.sps.*` | SpectralFilter |
| aten::fft_ifftshift | 108 | `backbone.network.*.*.token_mixer.sps.*` | SpectralFilter |
| aten::gelu | 36 | `backbone.network.*.*.mlp.act` | GELU |
| aten::mul | 24 | `backbone.network.*.*.res_scale1` | Scale |
| aten::mul | 24 | `backbone.network.*.*.res_scale2` | Scale |
| aten::mul | 36 | `backbone.network.*.*.token_mixer` | SPAttention |
| aten::mul | 108 | `backbone.network.*.*.token_mixer.sps.*` | SpectralFilter |
| aten::real | 108 | `backbone.network.*.*.token_mixer.sps.*` | SpectralFilter |

</details>

<details><summary>DFFormer-S36 — ADE20K Semantic FPN</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 298, 'aten::add': 262, 'aten::mean': 188, 'aten::pow': 184, 'aten::sub': 76, 'aten::sqrt': 76, 'aten::div': 76, 'aten::softmax': 36, 'aten::fft_rfft2': 36, 'aten::upsample_bicubic2d': 36, 'aten::view_as_complex': 36, 'aten::fft_irfft2': 36, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::add | 36 | `backbone.stages.*.*.token_mixer.reweight.act` | StarReLU |
| aten::add | 3 | `neck` | FPN |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::div | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_irfft2 | 36 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::fft_rfft2 | 36 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mean | 72 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::mean | 72 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::mean | 36 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::mul | 21 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 72 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::mul | 36 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::mul | 36 | `backbone.stages.*.*.token_mixer.reweight.act` | StarReLU |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::pow | 36 | `backbone.stages.*.*.token_mixer.reweight.act` | StarReLU |
| aten::softmax | 36 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::sqrt | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::sub | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::upsample_bicubic2d | 36 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::view_as_complex | 36 | `backbone.stages.*.*.token_mixer` | DynamicFilter |

</details>

<details><summary>SPANetV2-S36-pure — ADE20K Semantic FPN</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::add': 262, 'aten::mul': 262, 'aten::upsample_bicubic2d': 144, 'aten::sigmoid': 144, 'aten::fft_rfft2': 144, 'aten::fft_irfft2': 144, 'aten::gelu': 72, 'aten::pow': 40, 'aten::mean': 8, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::add | 3 | `neck` | FPN |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_irfft2 | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 36 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 36 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 180 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::upsample_bicubic2d | 144 | `backbone.stages.*.*.token_mixer` | ExSPAM |

</details>

<details><summary>CDFFormer-S36 — ADE20K Semantic FPN</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 253, 'aten::add': 247, 'aten::mean': 173, 'aten::pow': 169, 'aten::sub': 76, 'aten::sqrt': 76, 'aten::div': 76, 'aten::softmax': 21, 'aten::fft_rfft2': 21, 'aten::upsample_bicubic2d': 21, 'aten::view_as_complex': 21, 'aten::fft_irfft2': 21, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::add | 36 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::add | 21 | `backbone.stages.*.*.token_mixer.reweight.act` | StarReLU |
| aten::add | 3 | `neck` | FPN |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::div | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::div | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_irfft2 | 21 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::fft_rfft2 | 21 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mean | 72 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::mean | 72 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::mean | 21 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::mul | 21 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 42 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::mul | 36 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::mul | 21 | `backbone.stages.*.*.token_mixer.reweight.act` | StarReLU |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::pow | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.token_mixer.act1` | StarReLU |
| aten::pow | 21 | `backbone.stages.*.*.token_mixer.reweight.act` | StarReLU |
| aten::softmax | 21 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sqrt | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::sqrt | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 36 | `backbone.stages.*.*.norm1` | LayerNormGeneral |
| aten::sub | 36 | `backbone.stages.*.*.norm2` | LayerNormGeneral |
| aten::upsample_bicubic2d | 21 | `backbone.stages.*.*.token_mixer` | DynamicFilter |
| aten::view_as_complex | 21 | `backbone.stages.*.*.token_mixer` | DynamicFilter |

</details>

<details><summary>SPANetV2-S36-hybrid — ADE20K Semantic FPN</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 199, 'aten::add': 196, 'aten::upsample_bicubic2d': 60, 'aten::sigmoid': 60, 'aten::fft_rfft2': 60, 'aten::fft_irfft2': 60, 'aten::pow': 40, 'aten::gelu': 30, 'aten::softmax': 21, 'aten::mean': 8, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::feature_dropout': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 3 | `(root)` | EncoderDecoder |
| aten::add | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `backbone.stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::add | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::add | 18 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::add | 3 | `neck` | FPN |
| aten::div | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::feature_dropout | 1 | `decode_head.dropout` | Dropout2d |
| aten::fft_irfft2 | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 15 | `backbone.stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 15 | `backbone.stages.*.*.token_mixer.query.*` | GELU |
| aten::mean | 2 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `backbone.stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `backbone.stages.*.*.res_scale2` | Scale |
| aten::mul | 75 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::mul | 36 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::mul | 6 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::pow | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `backbone.stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |
| aten::softmax | 18 | `backbone.stages.*.*.token_mixer` | MixAttention |
| aten::softmax | 3 | `backbone.stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::sub | 1 | `backbone.downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `backbone.downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::upsample_bicubic2d | 60 | `backbone.stages.*.*.token_mixer` | ExSPAM |

</details>

<details><summary>ConvFormer-S18 — ImageNet-1K 384×384 (분류)</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::add': 76, 'aten::mul': 64, 'aten::pow': 40, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `stages.*.*.mlp.act` | StarReLU |
| aten::add | 18 | `stages.*.*.token_mixer.act1` | StarReLU |
| aten::div | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mean | 1 | `(root)` | MetaFormer |
| aten::mean | 2 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `stages.*.*.res_scale2` | Scale |
| aten::mul | 18 | `stages.*.*.token_mixer.act1` | StarReLU |
| aten::pow | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `stages.*.*.mlp.act` | StarReLU |
| aten::pow | 18 | `stages.*.*.token_mixer.act1` | StarReLU |
| aten::sqrt | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::square | 1 | `head.act` | SquaredReLU |
| aten::sub | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |

</details>

<details><summary>SPANetV2-S18-pure — ImageNet-1K 384×384 (분류)</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 136, 'aten::add': 130, 'aten::upsample_bicubic2d': 72, 'aten::sigmoid': 72, 'aten::fft_rfft2': 72, 'aten::fft_irfft2': 72, 'aten::gelu': 36, 'aten::pow': 22, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `stages.*.*.mlp.act` | StarReLU |
| aten::add | 72 | `stages.*.*.token_mixer` | ExSPAM |
| aten::div | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::fft_irfft2 | 72 | `stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 72 | `stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 18 | `stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 18 | `stages.*.*.token_mixer.query.*` | GELU |
| aten::mean | 1 | `(root)` | SPANetV2 |
| aten::mean | 2 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `stages.*.*.res_scale2` | Scale |
| aten::mul | 90 | `stages.*.*.token_mixer` | ExSPAM |
| aten::pow | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 72 | `stages.*.*.token_mixer` | ExSPAM |
| aten::sqrt | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::square | 1 | `head.act` | SquaredReLU |
| aten::sub | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::upsample_bicubic2d | 72 | `stages.*.*.token_mixer` | ExSPAM |

</details>

<details><summary>CAFormer-S18 — ImageNet-1K 384×384 (분류)</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 76, 'aten::add': 64, 'aten::pow': 28, 'aten::softmax': 12, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `stages.*.*.mlp.act` | StarReLU |
| aten::add | 6 | `stages.*.*.token_mixer.act1` | StarReLU |
| aten::div | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mean | 1 | `(root)` | MetaFormer |
| aten::mean | 2 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `stages.*.*.res_scale2` | Scale |
| aten::mul | 24 | `stages.*.*.token_mixer` | Attention |
| aten::mul | 6 | `stages.*.*.token_mixer.act1` | StarReLU |
| aten::pow | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `stages.*.*.mlp.act` | StarReLU |
| aten::pow | 6 | `stages.*.*.token_mixer.act1` | StarReLU |
| aten::softmax | 12 | `stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::square | 1 | `head.act` | SquaredReLU |
| aten::sub | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |

</details>

<details><summary>SPANetV2-S18-hybrid — ImageNet-1K 384×384 (분류)</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 100, 'aten::add': 91, 'aten::upsample_bicubic2d': 24, 'aten::sigmoid': 24, 'aten::fft_rfft2': 24, 'aten::fft_irfft2': 24, 'aten::pow': 22, 'aten::gelu': 12, 'aten::softmax': 12, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 36 | `stages.*.*` | MetaFormerBlock |
| aten::add | 18 | `stages.*.*.mlp.act` | StarReLU |
| aten::add | 24 | `stages.*.*.token_mixer` | ExSPAM |
| aten::add | 9 | `stages.*.*.token_mixer` | MixAttention |
| aten::div | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::fft_irfft2 | 24 | `stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 24 | `stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 6 | `stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 6 | `stages.*.*.token_mixer.query.*` | GELU |
| aten::mean | 1 | `(root)` | SPANetV2 |
| aten::mean | 2 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 18 | `stages.*.*.mlp.act` | StarReLU |
| aten::mul | 12 | `stages.*.*.res_scale1` | Scale |
| aten::mul | 12 | `stages.*.*.res_scale2` | Scale |
| aten::mul | 30 | `stages.*.*.token_mixer` | ExSPAM |
| aten::mul | 18 | `stages.*.*.token_mixer` | MixAttention |
| aten::mul | 6 | `stages.*.*.token_mixer` | Attention |
| aten::pow | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 18 | `stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 24 | `stages.*.*.token_mixer` | ExSPAM |
| aten::softmax | 9 | `stages.*.*.token_mixer` | MixAttention |
| aten::softmax | 3 | `stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::square | 1 | `head.act` | SquaredReLU |
| aten::sub | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::upsample_bicubic2d | 24 | `stages.*.*.token_mixer` | ExSPAM |

</details>

<details><summary>ConvFormer-S36 — ImageNet-1K 384×384 (분류)</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::add': 148, 'aten::mul': 118, 'aten::pow': 76, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `stages.*.*.mlp.act` | StarReLU |
| aten::add | 36 | `stages.*.*.token_mixer.act1` | StarReLU |
| aten::div | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mean | 1 | `(root)` | MetaFormer |
| aten::mean | 2 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `stages.*.*.res_scale2` | Scale |
| aten::mul | 36 | `stages.*.*.token_mixer.act1` | StarReLU |
| aten::pow | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `stages.*.*.mlp.act` | StarReLU |
| aten::pow | 36 | `stages.*.*.token_mixer.act1` | StarReLU |
| aten::sqrt | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::square | 1 | `head.act` | SquaredReLU |
| aten::sub | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |

</details>

<details><summary>SPANetV2-S36-pure — ImageNet-1K 384×384 (분류)</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 262, 'aten::add': 256, 'aten::upsample_bicubic2d': 144, 'aten::sigmoid': 144, 'aten::fft_rfft2': 144, 'aten::fft_irfft2': 144, 'aten::gelu': 72, 'aten::pow': 40, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `stages.*.*.mlp.act` | StarReLU |
| aten::add | 144 | `stages.*.*.token_mixer` | ExSPAM |
| aten::div | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::fft_irfft2 | 144 | `stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 144 | `stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 36 | `stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 36 | `stages.*.*.token_mixer.query.*` | GELU |
| aten::mean | 1 | `(root)` | SPANetV2 |
| aten::mean | 2 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `stages.*.*.res_scale2` | Scale |
| aten::mul | 180 | `stages.*.*.token_mixer` | ExSPAM |
| aten::pow | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 144 | `stages.*.*.token_mixer` | ExSPAM |
| aten::sqrt | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::square | 1 | `head.act` | SquaredReLU |
| aten::sub | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::upsample_bicubic2d | 144 | `stages.*.*.token_mixer` | ExSPAM |

</details>

<details><summary>CAFormer-S36 — ImageNet-1K 384×384 (분류)</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 139, 'aten::add': 127, 'aten::pow': 55, 'aten::softmax': 21, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `stages.*.*.mlp.act` | StarReLU |
| aten::add | 15 | `stages.*.*.token_mixer.act1` | StarReLU |
| aten::div | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mean | 1 | `(root)` | MetaFormer |
| aten::mean | 2 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `stages.*.*.res_scale2` | Scale |
| aten::mul | 42 | `stages.*.*.token_mixer` | Attention |
| aten::mul | 15 | `stages.*.*.token_mixer.act1` | StarReLU |
| aten::pow | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `stages.*.*.mlp.act` | StarReLU |
| aten::pow | 15 | `stages.*.*.token_mixer.act1` | StarReLU |
| aten::softmax | 21 | `stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::square | 1 | `head.act` | SquaredReLU |
| aten::sub | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |

</details>

<details><summary>SPANetV2-S36-hybrid — ImageNet-1K 384×384 (분류)</summary>

`FlopAnalyzer(model, inputs).unsupported_ops()` 출력:

```
Counter({'aten::mul': 199, 'aten::add': 190, 'aten::upsample_bicubic2d': 60, 'aten::sigmoid': 60, 'aten::fft_rfft2': 60, 'aten::fft_irfft2': 60, 'aten::pow': 40, 'aten::gelu': 30, 'aten::softmax': 21, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 수 | 모듈 (번호는 *) | 모듈 클래스 |
|---|---|---|---|
| aten::add | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::add | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::add | 72 | `stages.*.*` | MetaFormerBlock |
| aten::add | 36 | `stages.*.*.mlp.act` | StarReLU |
| aten::add | 60 | `stages.*.*.token_mixer` | ExSPAM |
| aten::add | 18 | `stages.*.*.token_mixer` | MixAttention |
| aten::div | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::div | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::fft_irfft2 | 60 | `stages.*.*.token_mixer` | ExSPAM |
| aten::fft_rfft2 | 60 | `stages.*.*.token_mixer` | ExSPAM |
| aten::gelu | 15 | `stages.*.*.token_mixer.proj.*` | GELU |
| aten::gelu | 15 | `stages.*.*.token_mixer.query.*` | GELU |
| aten::mean | 1 | `(root)` | SPANetV2 |
| aten::mean | 2 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mean | 6 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::mul | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::mul | 36 | `stages.*.*.mlp.act` | StarReLU |
| aten::mul | 21 | `stages.*.*.res_scale1` | Scale |
| aten::mul | 21 | `stages.*.*.res_scale2` | Scale |
| aten::mul | 75 | `stages.*.*.token_mixer` | ExSPAM |
| aten::mul | 36 | `stages.*.*.token_mixer` | MixAttention |
| aten::mul | 6 | `stages.*.*.token_mixer` | Attention |
| aten::pow | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::pow | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::pow | 36 | `stages.*.*.mlp.act` | StarReLU |
| aten::sigmoid | 60 | `stages.*.*.token_mixer` | ExSPAM |
| aten::softmax | 18 | `stages.*.*.token_mixer` | MixAttention |
| aten::softmax | 3 | `stages.*.*.token_mixer` | Attention |
| aten::sqrt | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sqrt | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::square | 1 | `head.act` | SquaredReLU |
| aten::sub | 1 | `downsample_layers.*.post_norm` | LayerNormGeneral |
| aten::sub | 3 | `downsample_layers.*.pre_norm` | LayerNormGeneral |
| aten::upsample_bicubic2d | 60 | `stages.*.*.token_mixer` | ExSPAM |

</details>

## 4. 하나의 규약으로 다시 잼

`../measure.py`의 `Convention` 처리 함수(분류 재측정과 같은 코드)를 mmengine의 `FlopAnalyzer`에 `set_op_handle`로 그대로 등록했다. 처리 함수의 인자 형식은 fvcore와 같다. (가)는 처리 함수 없이 `FlopAnalyzer(model, inputs).total()`이다. get_flops.py가 부르는 `get_model_complexity_info`의 flops가 이 값이다. 합계 = (가) + (a) + (b) + (c) + (d). 처리 함수를 등록한 분석의 total과 같은지 모델마다 확인했다(assert).

- **(a) FFT:** 한 번의 변환을 E·⌈log2 N⌉로 센다. E는 실수 영역 텐서의 원소 수다(rfft*는 입력, irfft*는 출력, 복소 fft*/ifft*는 복소 출력). N은 `dim` 인자의 축 크기를 곱한 값이다. fftshift/ifftshift는 0이다.
- **(b) 주파수 영역의 원소별 곱:** 복소 출력인 aten::mul, 출력 원소 하나에 1.
- **(c) 그 밖의 원소별 연산:** measure.py의 목록(add, sub, mul, div, pow, sqrt, gelu, sigmoid, softmax, mean 등), 출력 원소 하나에 1.
- **(d) 마스크·필터 크기 조정:** aten::upsample_bicubic2d, 출력 원소 하나에 1. SPANetV2는 SRF 마스크를 224 격자로 만들어 둔다. 입력이 만드는 격자가 다르면 ExSPAM.forward의 `resize_weight`가 bicubic으로 보간한다. DFFormer·CDFFormer의 분할 백본(`weight_resize=True`)은 `resize_complex_weight`가 복소 필터를 bicubic으로 보간한다. 크기가 같아도 보간한다. 이것도 (d)에 넣었다. 헤드의 보간(FPN의 nearest, UperNet·FPNHead의 bilinear)은 mmengine 기본 처리 함수가 이미 센다. 그래서 (가)에 들어 있다.
- 분류 재측정에 없던 연산을 이렇게 처리했다.
  - `aten::log2` → (c). mmdet RoI 추출기가 RoI를 FPN 단계에 배정할 때 쓴다(원소 1000개).
  - `aten::feature_dropout` → 0. eval 모드 Dropout2d로 항등이다. aten::dropout은 무시 목록에 있다.
  - `aten::max_pool2d` → 0. mmdet FPN의 다섯째 출력 단계가 `F.max_pool2d(x, 1, stride=2)`다. 창이 1×1이라 산술이 없다. 다른 창 크기가 나오면 측정을 멈추게 했다.
  - `prim::PythonOp.RoIAlignFunction`(mmcv RoIAlign) → 세지 않음. 원소별 연산이 아니고 규약에 정의가 없다. 호출 수와 출력 원소 수는 기록했다.
  - `aten::relu`/`relu_`는 분류 때처럼 0이다(mmengine 무시 목록). 원소 수만 기록했다.
- 위 목록에 없는 연산이 나오면 측정이 멈춘다. 모든 모델이 멈추지 않고 끝났다.

검출기는 아래 forward로 쟀다. 서브모듈은 검출기 자신의 모듈이고, 이름도 같다(backbone, neck, rpn_head, roi_head). RoI는 이미지 안의 고정 상자 1000개다(`make_rois`, 시드 0). 상자 값은 어떤 수도 바꾸지 않는다.

```python
class TwoStageFixedRoI(nn.Module):
    """Two-stage detector forward with a fixed RoI count, the same for every model.

    Same structure as mmdet 2.x `TwoStageDetector.forward_dummy` + `{Standard,Cascade}RoIHead.forward_dummy`
    (backbone -> neck -> RPN head convs -> every bbox head on all RoIs -> every mask head on the first 100 RoIs),
    written with the mmdet 3.x RoI-head methods `_bbox_forward` / `_mask_forward` of the built model.
    The submodules are the detector's own modules under their own names (backbone, neck, rpn_head, roi_head).
    """

    def __init__(self, det, num_mask_rois=NUM_MASK_ROIS):
        super().__init__()
        self.backbone, self.neck, self.rpn_head, self.roi_head = det.backbone, det.neck, det.rpn_head, det.roi_head
        self.num_mask_rois = num_mask_rois
        self.cascade = hasattr(det.roi_head, 'num_stages')

    def forward(self, img, rois):
        x = self.neck(self.backbone(img))
        cls_scores, bbox_preds = self.rpn_head(x)
        outs = list(cls_scores) + list(bbox_preds)
        mask_rois = rois[:self.num_mask_rois]
        if self.cascade:
            for i in range(self.roi_head.num_stages):
                r = self.roi_head._bbox_forward(i, x, rois)
                outs += [r['cls_score'], r['bbox_pred']]
            for i in range(self.roi_head.num_stages):
                outs.append(self.roi_head._mask_forward(i, x, mask_rois)['mask_preds'])
        else:
            r = self.roi_head._bbox_forward(x, rois)
            outs += [r['cls_score'], r['bbox_pred']]
            outs.append(self.roi_head._mask_forward(x, mask_rois)['mask_preds'])
        return tuple(outs)
```

### 처리 함수 확인

모델마다 FFT 연산의 종류별 첫 호출과 보간 연산의 첫 호출이다(전체 모델 측정). 인자의 None은 trace에서 상수가 아닌 값이다(forward에서 H, W로 계산). 실제 크기는 출력 shape에 있다.

**COCO Mask R-CNN 3×**

| 모델 | 연산 | 모듈 | 입력 shape, dtype | 출력 shape, dtype | 인자 | 계산 | 값 (MACs) |
|---|---|---|---|---|---|---|---|
| ConvFormer-S18 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S18-pure | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 320, 200] Float | [1, 16, 320, 101] ComplexFloat | dim=[2, 3], s=None | E=1,024,000 (input), N=64,000, ⌈log2 N⌉=16 | 16,384,000 |
| SPANetV2-S18-pure | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 320, 101] ComplexFloat | [1, 16, 320, 200] Float | dim=[2, 3], s=[None, None] | E=1,024,000 (output), N=64,000, ⌈log2 N⌉=16 | 16,384,000 |
| SPANetV2-S18-pure | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 320, 101] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 517,120 | 517,120 |
| CAFormer-S18 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S18-hybrid | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 320, 200] Float | [1, 16, 320, 101] ComplexFloat | dim=[2, 3], s=None | E=1,024,000 (input), N=64,000, ⌈log2 N⌉=16 | 16,384,000 |
| SPANetV2-S18-hybrid | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 320, 101] ComplexFloat | [1, 16, 320, 200] Float | dim=[2, 3], s=[None, None] | E=1,024,000 (output), N=64,000, ⌈log2 N⌉=16 | 16,384,000 |
| SPANetV2-S18-hybrid | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 320, 101] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 517,120 | 517,120 |

**COCO Cascade Mask R-CNN 3×**

| 모델 | 연산 | 모듈 | 입력 shape, dtype | 출력 shape, dtype | 인자 | 계산 | 값 (MACs) |
|---|---|---|---|---|---|---|---|
| ConvFormer-S18 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S18-pure | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 320, 200] Float | [1, 16, 320, 101] ComplexFloat | dim=[2, 3], s=None | E=1,024,000 (input), N=64,000, ⌈log2 N⌉=16 | 16,384,000 |
| SPANetV2-S18-pure | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 320, 101] ComplexFloat | [1, 16, 320, 200] Float | dim=[2, 3], s=[None, None] | E=1,024,000 (output), N=64,000, ⌈log2 N⌉=16 | 16,384,000 |
| SPANetV2-S18-pure | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 320, 101] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 517,120 | 517,120 |
| CAFormer-S18 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S18-hybrid | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 320, 200] Float | [1, 16, 320, 101] ComplexFloat | dim=[2, 3], s=None | E=1,024,000 (input), N=64,000, ⌈log2 N⌉=16 | 16,384,000 |
| SPANetV2-S18-hybrid | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 320, 101] ComplexFloat | [1, 16, 320, 200] Float | dim=[2, 3], s=[None, None] | E=1,024,000 (output), N=64,000, ⌈log2 N⌉=16 | 16,384,000 |
| SPANetV2-S18-hybrid | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 320, 101] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 517,120 | 517,120 |
| ConvFormer-S36 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S36-pure | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 320, 200] Float | [1, 16, 320, 101] ComplexFloat | dim=[2, 3], s=None | E=1,024,000 (input), N=64,000, ⌈log2 N⌉=16 | 16,384,000 |
| SPANetV2-S36-pure | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 320, 101] ComplexFloat | [1, 16, 320, 200] Float | dim=[2, 3], s=[None, None] | E=1,024,000 (output), N=64,000, ⌈log2 N⌉=16 | 16,384,000 |
| SPANetV2-S36-pure | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 320, 101] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 517,120 | 517,120 |
| CAFormer-S36 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S36-hybrid | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 320, 200] Float | [1, 16, 320, 101] ComplexFloat | dim=[2, 3], s=None | E=1,024,000 (input), N=64,000, ⌈log2 N⌉=16 | 16,384,000 |
| SPANetV2-S36-hybrid | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 320, 101] ComplexFloat | [1, 16, 320, 200] Float | dim=[2, 3], s=[None, None] | E=1,024,000 (output), N=64,000, ⌈log2 N⌉=16 | 16,384,000 |
| SPANetV2-S36-hybrid | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 320, 101] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 517,120 | 517,120 |

**ADE20K UperNet**

| 모델 | 연산 | 모듈 | 입력 shape, dtype | 출력 shape, dtype | 인자 | 계산 | 값 (MACs) |
|---|---|---|---|---|---|---|---|
| ConvFormer-S18 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S18-pure | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 512] Float | [1, 16, 128, 257] ComplexFloat | dim=[2, 3], s=None | E=1,048,576 (input), N=65,536, ⌈log2 N⌉=16 | 16,777,216 |
| SPANetV2-S18-pure | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 257] ComplexFloat | [1, 16, 128, 512] Float | dim=[2, 3], s=[None, None] | E=1,048,576 (output), N=65,536, ⌈log2 N⌉=16 | 16,777,216 |
| SPANetV2-S18-pure | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 128, 257] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 526,336 | 526,336 |
| CAFormer-S18 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S18-hybrid | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 512] Float | [1, 16, 128, 257] ComplexFloat | dim=[2, 3], s=None | E=1,048,576 (input), N=65,536, ⌈log2 N⌉=16 | 16,777,216 |
| SPANetV2-S18-hybrid | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 257] ComplexFloat | [1, 16, 128, 512] Float | dim=[2, 3], s=[None, None] | E=1,048,576 (output), N=65,536, ⌈log2 N⌉=16 | 16,777,216 |
| SPANetV2-S18-hybrid | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 128, 257] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 526,336 | 526,336 |
| ConvFormer-S36 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S36-pure | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 512] Float | [1, 16, 128, 257] ComplexFloat | dim=[2, 3], s=None | E=1,048,576 (input), N=65,536, ⌈log2 N⌉=16 | 16,777,216 |
| SPANetV2-S36-pure | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 257] ComplexFloat | [1, 16, 128, 512] Float | dim=[2, 3], s=[None, None] | E=1,048,576 (output), N=65,536, ⌈log2 N⌉=16 | 16,777,216 |
| SPANetV2-S36-pure | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 128, 257] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 526,336 | 526,336 |
| CAFormer-S36 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S36-hybrid | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 512] Float | [1, 16, 128, 257] ComplexFloat | dim=[2, 3], s=None | E=1,048,576 (input), N=65,536, ⌈log2 N⌉=16 | 16,777,216 |
| SPANetV2-S36-hybrid | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 257] ComplexFloat | [1, 16, 128, 512] Float | dim=[2, 3], s=[None, None] | E=1,048,576 (output), N=65,536, ⌈log2 N⌉=16 | 16,777,216 |
| SPANetV2-S36-hybrid | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 128, 257] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 526,336 | 526,336 |

**ADE20K Semantic FPN**

| 모델 | 연산 | 모듈 | 입력 shape, dtype | 출력 shape, dtype | 인자 | 계산 | 값 (MACs) |
|---|---|---|---|---|---|---|---|
| SPANet-S | aten::fft_fft2 | `backbone.network.0.0.token_mixer.sps.0` | [1, 22, 128, 128] Float | [1, 22, 128, 128] ComplexFloat | dim=[2, 3], s=None | E=360,448 (output), N=16,384, ⌈log2 N⌉=14 | 5,046,272 |
| SPANet-S | aten::fft_ifft2 | `backbone.network.0.0.token_mixer.sps.0` | [1, 22, 128, 128] ComplexFloat | [1, 22, 128, 128] ComplexFloat | dim=[2, 3], s=[None, None] | E=360,448 (output), N=16,384, ⌈log2 N⌉=14 | 5,046,272 |
| DFFormer-S18 | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 128, 128, 128] Float | [1, 128, 65, 128] ComplexFloat | dim=[1, 2], s=None | E=2,097,152 (input), N=16,384, ⌈log2 N⌉=14 | 29,360,128 |
| DFFormer-S18 | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 128, 65, 128] ComplexFloat | [1, 128, 128, 128] Float | dim=[1, 2], s=[None, None] | E=2,097,152 (output), N=16,384, ⌈log2 N⌉=14 | 29,360,128 |
| DFFormer-S18 | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 8, 128, 65] Float | [1, 8, 128, 65] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 66,560 | 66,560 |
| SPANetV2-S18-pure | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 128] Float | [1, 16, 128, 65] ComplexFloat | dim=[2, 3], s=None | E=262,144 (input), N=16,384, ⌈log2 N⌉=14 | 3,670,016 |
| SPANetV2-S18-pure | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 65] ComplexFloat | [1, 16, 128, 128] Float | dim=[2, 3], s=[None, None] | E=262,144 (output), N=16,384, ⌈log2 N⌉=14 | 3,670,016 |
| SPANetV2-S18-pure | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 128, 65] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 133,120 | 133,120 |
| CDFFormer-S18 | aten::fft_rfft2 | `backbone.stages.2.0.token_mixer` | [1, 32, 32, 640] Float | [1, 32, 17, 640] ComplexFloat | dim=[1, 2], s=None | E=655,360 (input), N=1,024, ⌈log2 N⌉=10 | 6,553,600 |
| CDFFormer-S18 | aten::fft_irfft2 | `backbone.stages.2.0.token_mixer` | [1, 32, 17, 640] ComplexFloat | [1, 32, 32, 640] Float | dim=[1, 2], s=[None, None] | E=655,360 (output), N=1,024, ⌈log2 N⌉=10 | 6,553,600 |
| CDFFormer-S18 | aten::upsample_bicubic2d | `backbone.stages.2.0.token_mixer` | [1, 8, 32, 17] Float | [1, 8, 32, 17] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 4,352 | 4,352 |
| SPANetV2-S18-hybrid | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 128] Float | [1, 16, 128, 65] ComplexFloat | dim=[2, 3], s=None | E=262,144 (input), N=16,384, ⌈log2 N⌉=14 | 3,670,016 |
| SPANetV2-S18-hybrid | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 65] ComplexFloat | [1, 16, 128, 128] Float | dim=[2, 3], s=[None, None] | E=262,144 (output), N=16,384, ⌈log2 N⌉=14 | 3,670,016 |
| SPANetV2-S18-hybrid | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 128, 65] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 133,120 | 133,120 |
| SPANet-M | aten::fft_fft2 | `backbone.network.0.0.token_mixer.sps.0` | [1, 22, 128, 128] Float | [1, 22, 128, 128] ComplexFloat | dim=[2, 3], s=None | E=360,448 (output), N=16,384, ⌈log2 N⌉=14 | 5,046,272 |
| SPANet-M | aten::fft_ifft2 | `backbone.network.0.0.token_mixer.sps.0` | [1, 22, 128, 128] ComplexFloat | [1, 22, 128, 128] ComplexFloat | dim=[2, 3], s=[None, None] | E=360,448 (output), N=16,384, ⌈log2 N⌉=14 | 5,046,272 |
| DFFormer-S36 | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 128, 128, 128] Float | [1, 128, 65, 128] ComplexFloat | dim=[1, 2], s=None | E=2,097,152 (input), N=16,384, ⌈log2 N⌉=14 | 29,360,128 |
| DFFormer-S36 | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 128, 65, 128] ComplexFloat | [1, 128, 128, 128] Float | dim=[1, 2], s=[None, None] | E=2,097,152 (output), N=16,384, ⌈log2 N⌉=14 | 29,360,128 |
| DFFormer-S36 | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 8, 128, 65] Float | [1, 8, 128, 65] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 66,560 | 66,560 |
| SPANetV2-S36-pure | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 128] Float | [1, 16, 128, 65] ComplexFloat | dim=[2, 3], s=None | E=262,144 (input), N=16,384, ⌈log2 N⌉=14 | 3,670,016 |
| SPANetV2-S36-pure | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 65] ComplexFloat | [1, 16, 128, 128] Float | dim=[2, 3], s=[None, None] | E=262,144 (output), N=16,384, ⌈log2 N⌉=14 | 3,670,016 |
| SPANetV2-S36-pure | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 128, 65] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 133,120 | 133,120 |
| CDFFormer-S36 | aten::fft_rfft2 | `backbone.stages.2.0.token_mixer` | [1, 32, 32, 640] Float | [1, 32, 17, 640] ComplexFloat | dim=[1, 2], s=None | E=655,360 (input), N=1,024, ⌈log2 N⌉=10 | 6,553,600 |
| CDFFormer-S36 | aten::fft_irfft2 | `backbone.stages.2.0.token_mixer` | [1, 32, 17, 640] ComplexFloat | [1, 32, 32, 640] Float | dim=[1, 2], s=[None, None] | E=655,360 (output), N=1,024, ⌈log2 N⌉=10 | 6,553,600 |
| CDFFormer-S36 | aten::upsample_bicubic2d | `backbone.stages.2.0.token_mixer` | [1, 8, 32, 17] Float | [1, 8, 32, 17] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 4,352 | 4,352 |
| SPANetV2-S36-hybrid | aten::fft_rfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 128] Float | [1, 16, 128, 65] ComplexFloat | dim=[2, 3], s=None | E=262,144 (input), N=16,384, ⌈log2 N⌉=14 | 3,670,016 |
| SPANetV2-S36-hybrid | aten::fft_irfft2 | `backbone.stages.0.0.token_mixer` | [1, 16, 128, 65] ComplexFloat | [1, 16, 128, 128] Float | dim=[2, 3], s=[None, None] | E=262,144 (output), N=16,384, ⌈log2 N⌉=14 | 3,670,016 |
| SPANetV2-S36-hybrid | aten::upsample_bicubic2d | `backbone.stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 128, 65] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 133,120 | 133,120 |

**ImageNet-1K 384×384 (분류)**

| 모델 | 연산 | 모듈 | 입력 shape, dtype | 출력 shape, dtype | 인자 | 계산 | 값 (MACs) |
|---|---|---|---|---|---|---|---|
| ConvFormer-S18 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S18-pure | aten::fft_rfft2 | `stages.0.0.token_mixer` | [1, 16, 96, 96] Float | [1, 16, 96, 49] ComplexFloat | dim=[2, 3], s=None | E=147,456 (input), N=9,216, ⌈log2 N⌉=14 | 2,064,384 |
| SPANetV2-S18-pure | aten::fft_irfft2 | `stages.0.0.token_mixer` | [1, 16, 96, 49] ComplexFloat | [1, 16, 96, 96] Float | dim=[2, 3], s=[None, None] | E=147,456 (output), N=9,216, ⌈log2 N⌉=14 | 2,064,384 |
| SPANetV2-S18-pure | aten::upsample_bicubic2d | `stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 96, 49] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 75,264 | 75,264 |
| CAFormer-S18 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S18-hybrid | aten::fft_rfft2 | `stages.0.0.token_mixer` | [1, 16, 96, 96] Float | [1, 16, 96, 49] ComplexFloat | dim=[2, 3], s=None | E=147,456 (input), N=9,216, ⌈log2 N⌉=14 | 2,064,384 |
| SPANetV2-S18-hybrid | aten::fft_irfft2 | `stages.0.0.token_mixer` | [1, 16, 96, 49] ComplexFloat | [1, 16, 96, 96] Float | dim=[2, 3], s=[None, None] | E=147,456 (output), N=9,216, ⌈log2 N⌉=14 | 2,064,384 |
| SPANetV2-S18-hybrid | aten::upsample_bicubic2d | `stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 96, 49] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 75,264 | 75,264 |
| ConvFormer-S36 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S36-pure | aten::fft_rfft2 | `stages.0.0.token_mixer` | [1, 16, 96, 96] Float | [1, 16, 96, 49] ComplexFloat | dim=[2, 3], s=None | E=147,456 (input), N=9,216, ⌈log2 N⌉=14 | 2,064,384 |
| SPANetV2-S36-pure | aten::fft_irfft2 | `stages.0.0.token_mixer` | [1, 16, 96, 49] ComplexFloat | [1, 16, 96, 96] Float | dim=[2, 3], s=[None, None] | E=147,456 (output), N=9,216, ⌈log2 N⌉=14 | 2,064,384 |
| SPANetV2-S36-pure | aten::upsample_bicubic2d | `stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 96, 49] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 75,264 | 75,264 |
| CAFormer-S36 | (FFT·보간 연산 없음) |  |  |  |  |  |  |
| SPANetV2-S36-hybrid | aten::fft_rfft2 | `stages.0.0.token_mixer` | [1, 16, 96, 96] Float | [1, 16, 96, 49] ComplexFloat | dim=[2, 3], s=None | E=147,456 (input), N=9,216, ⌈log2 N⌉=14 | 2,064,384 |
| SPANetV2-S36-hybrid | aten::fft_irfft2 | `stages.0.0.token_mixer` | [1, 16, 96, 49] ComplexFloat | [1, 16, 96, 96] Float | dim=[2, 3], s=[None, None] | E=147,456 (output), N=9,216, ⌈log2 N⌉=14 | 2,064,384 |
| SPANetV2-S36-hybrid | aten::upsample_bicubic2d | `stages.0.0.token_mixer` | [1, 16, 56, 29] Float | [1, 16, 96, 49] Float | output_size=[None, None], align_corners=True, scale_factors=None | 출력 원소 수 75,264 | 75,264 |

### 모델별 (a)(b)(c)(d) 내역

<details><summary>ConvFormer-S18 — COCO Mask R-CNN 3×</summary>

(a) 0 · (b) 0 · (c) 724,055,700 · (d) 0 MACs
backbone만: (가) 80,719,616,000 · (a) 0 · (b) 0 · (c) 702,544,000 · (d) 0 · 합계 81,422,160,000 MACs; 전체 모델 측정의 backbone 몫 81,422,160,000 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 21,504,000 | 21,504,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 128,000 | 128,000 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 168,000 | 168,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.bbox_roi_extractor` | 2 | 2,000 | 2,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.mask_roi_extractor` | 2 | 200 | 200 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 125,952,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 18 | 62,976,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `rpn_head` | 5 | 21,826,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.relu` | 2 | 2,048,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.convs.*.activate` | 4 | 20,070,400 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.relu` | 1 | 20,070,400 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.bbox_roi_extractor.roi_layers.*` | 4 | 12,544,000 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.mask_roi_extractor.roi_layers.*` | 4 | 5,017,600 | 0 |
| 0 (자료 이동·항등) | aten::max_pool2d | Float | `neck` | 1 | 66,560 | 0 |

</details>

<details><summary>SPANetV2-S18-pure — COCO Mask R-CNN 3×</summary>

(a) 872,448,000 · (b) 16,128,000 · (c) 677,207,772 · (d) 16,128,000 MACs
backbone만: (가) 85,537,280,000 · (a) 872,448,000 · (b) 16,128,000 · (c) 655,696,072 · (d) 16,128,000 · 합계 87,097,680,072 MACs; 전체 모델 측정의 backbone 몫 87,097,680,072 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 72 | 31,488,000 | 436,224,000 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 72 | 31,488,000 | 436,224,000 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 72 | 16,128,000 | 16,128,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 72 | 0 | 72 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 21,504,000 | 21,504,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 18 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 18 | 31,488,000 | 31,488,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 128,000 | 128,000 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 168,000 | 168,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 18 | 31,488,000 | 31,488,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 72 | 16,128,000 | 16,128,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.bbox_roi_extractor` | 2 | 2,000 | 2,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.mask_roi_extractor` | 2 | 200 | 200 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 72 | 16,128,000 | 16,128,000 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 125,952,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `rpn_head` | 5 | 21,826,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.relu` | 2 | 2,048,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.convs.*.activate` | 4 | 20,070,400 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.relu` | 1 | 20,070,400 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.bbox_roi_extractor.roi_layers.*` | 4 | 12,544,000 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.mask_roi_extractor.roi_layers.*` | 4 | 5,017,600 | 0 |
| 0 (자료 이동·항등) | aten::max_pool2d | Float | `neck` | 1 | 66,560 | 0 |

</details>

<details><summary>CAFormer-S18 — COCO Mask R-CNN 3×</summary>

(a) 0 · (b) 0 · (c) 3,621,719,712 · (d) 0 MACs
backbone만: (가) 174,672,128,000 · (a) 0 · (b) 0 · (c) 3,600,208,012 · (d) 0 · 합계 178,272,336,012 MACs; 전체 모델 측정의 backbone 몫 178,272,336,012 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 6 | 36,864,000 | 36,864,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 21,504,000 | 21,504,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 128,000 | 128,000 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 168,000 | 168,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 12 | 0 | 12 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 12 | 1,488,000,000 | 1,488,000,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 6 | 36,864,000 | 36,864,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 6 | 36,864,000 | 36,864,000 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 12 | 1,488,000,000 | 1,488,000,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.bbox_roi_extractor` | 2 | 2,000 | 2,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.mask_roi_extractor` | 2 | 200 | 200 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 125,952,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 6 | 36,864,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `rpn_head` | 5 | 21,826,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.relu` | 2 | 2,048,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.convs.*.activate` | 4 | 20,070,400 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.relu` | 1 | 20,070,400 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.bbox_roi_extractor.roi_layers.*` | 4 | 12,544,000 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.mask_roi_extractor.roi_layers.*` | 4 | 5,017,600 | 0 |
| 0 (자료 이동·항등) | aten::max_pool2d | Float | `neck` | 1 | 66,560 | 0 |

</details>

<details><summary>SPANetV2-S18-hybrid — COCO Mask R-CNN 3×</summary>

(a) 565,248,000 · (b) 9,338,880 · (c) 3,605,714,616 · (d) 9,338,880 MACs
backbone만: (가) 176,090,624,000 · (a) 565,248,000 · (b) 9,338,880 · (c) 3,584,202,916 · (d) 9,338,880 · 합계 180,258,752,676 MACs; 전체 모델 측정의 backbone 몫 180,258,752,676 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 24 | 18,432,000 | 282,624,000 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 24 | 18,432,000 | 282,624,000 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 24 | 9,338,880 | 9,338,880 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 24 | 0 | 24 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer` | 9 | 11,520,000 | 11,520,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 21,504,000 | 21,504,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 6 | 36,864,000 | 36,864,000 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 6 | 18,432,000 | 18,432,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 128,000 | 128,000 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 168,000 | 168,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 18 | 1,506,432,000 | 1,506,432,000 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 12 | 0 | 12 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 24 | 9,338,880 | 9,338,880 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 12 | 1,488,000,000 | 1,488,000,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.bbox_roi_extractor` | 1 | 1,000 | 1,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.mask_roi_extractor` | 1 | 100 | 100 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.bbox_roi_extractor` | 2 | 2,000 | 2,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.mask_roi_extractor` | 2 | 200 | 200 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 24 | 9,338,880 | 9,338,880 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 125,952,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `rpn_head` | 5 | 21,826,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.relu` | 2 | 2,048,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.convs.*.activate` | 4 | 20,070,400 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.relu` | 1 | 20,070,400 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.bbox_roi_extractor.roi_layers.*` | 4 | 12,544,000 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.mask_roi_extractor.roi_layers.*` | 4 | 5,017,600 | 0 |
| 0 (자료 이동·항등) | aten::max_pool2d | Float | `neck` | 1 | 66,560 | 0 |

</details>

<details><summary>ConvFormer-S18 — COCO Cascade Mask R-CNN 3×</summary>

(a) 0 · (b) 0 · (c) 724,071,100 · (d) 0 MACs
backbone만: (가) 80,719,616,000 · (a) 0 · (b) 0 · (c) 702,544,000 · (d) 0 · 합계 81,422,160,000 MACs; 전체 모델 측정의 backbone 몫 81,422,160,000 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 21,504,000 | 21,504,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 128,000 | 128,000 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 168,000 | 168,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.bbox_roi_extractor.*` | 6 | 6,000 | 6,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.mask_roi_extractor.*` | 6 | 600 | 600 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 125,952,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 18 | 62,976,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `rpn_head` | 5 | 21,826,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.relu` | 3 | 3,072,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.shared_convs.*.activate` | 12 | 150,528,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.convs.*.activate` | 12 | 60,211,200 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.relu` | 3 | 60,211,200 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.bbox_roi_extractor.*.roi_layers.*` | 12 | 37,632,000 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.mask_roi_extractor.*.roi_layers.*` | 12 | 15,052,800 | 0 |
| 0 (자료 이동·항등) | aten::max_pool2d | Float | `neck` | 1 | 66,560 | 0 |

</details>

<details><summary>SPANetV2-S18-pure — COCO Cascade Mask R-CNN 3×</summary>

(a) 872,448,000 · (b) 16,128,000 · (c) 677,223,172 · (d) 16,128,000 MACs
backbone만: (가) 85,537,280,000 · (a) 872,448,000 · (b) 16,128,000 · (c) 655,696,072 · (d) 16,128,000 · 합계 87,097,680,072 MACs; 전체 모델 측정의 backbone 몫 87,097,680,072 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 72 | 31,488,000 | 436,224,000 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 72 | 31,488,000 | 436,224,000 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 72 | 16,128,000 | 16,128,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 72 | 0 | 72 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 21,504,000 | 21,504,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 18 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 18 | 31,488,000 | 31,488,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 128,000 | 128,000 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 168,000 | 168,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 18 | 31,488,000 | 31,488,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 72 | 16,128,000 | 16,128,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.bbox_roi_extractor.*` | 6 | 6,000 | 6,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.mask_roi_extractor.*` | 6 | 600 | 600 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 72 | 16,128,000 | 16,128,000 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 125,952,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `rpn_head` | 5 | 21,826,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.relu` | 3 | 3,072,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.shared_convs.*.activate` | 12 | 150,528,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.convs.*.activate` | 12 | 60,211,200 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.relu` | 3 | 60,211,200 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.bbox_roi_extractor.*.roi_layers.*` | 12 | 37,632,000 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.mask_roi_extractor.*.roi_layers.*` | 12 | 15,052,800 | 0 |
| 0 (자료 이동·항등) | aten::max_pool2d | Float | `neck` | 1 | 66,560 | 0 |

</details>

<details><summary>CAFormer-S18 — COCO Cascade Mask R-CNN 3×</summary>

(a) 0 · (b) 0 · (c) 3,621,735,112 · (d) 0 MACs
backbone만: (가) 174,672,128,000 · (a) 0 · (b) 0 · (c) 3,600,208,012 · (d) 0 · 합계 178,272,336,012 MACs; 전체 모델 측정의 backbone 몫 178,272,336,012 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 6 | 36,864,000 | 36,864,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 21,504,000 | 21,504,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 128,000 | 128,000 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 168,000 | 168,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 12 | 0 | 12 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 12 | 1,488,000,000 | 1,488,000,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 6 | 36,864,000 | 36,864,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 6 | 36,864,000 | 36,864,000 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 12 | 1,488,000,000 | 1,488,000,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.bbox_roi_extractor.*` | 6 | 6,000 | 6,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.mask_roi_extractor.*` | 6 | 600 | 600 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 125,952,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 6 | 36,864,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `rpn_head` | 5 | 21,826,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.relu` | 3 | 3,072,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.shared_convs.*.activate` | 12 | 150,528,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.convs.*.activate` | 12 | 60,211,200 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.relu` | 3 | 60,211,200 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.bbox_roi_extractor.*.roi_layers.*` | 12 | 37,632,000 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.mask_roi_extractor.*.roi_layers.*` | 12 | 15,052,800 | 0 |
| 0 (자료 이동·항등) | aten::max_pool2d | Float | `neck` | 1 | 66,560 | 0 |

</details>

<details><summary>SPANetV2-S18-hybrid — COCO Cascade Mask R-CNN 3×</summary>

(a) 565,248,000 · (b) 9,338,880 · (c) 3,605,730,016 · (d) 9,338,880 MACs
backbone만: (가) 176,090,624,000 · (a) 565,248,000 · (b) 9,338,880 · (c) 3,584,202,916 · (d) 9,338,880 · 합계 180,258,752,676 MACs; 전체 모델 측정의 backbone 몫 180,258,752,676 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 24 | 18,432,000 | 282,624,000 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 24 | 18,432,000 | 282,624,000 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 24 | 9,338,880 | 9,338,880 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 62,976,000 | 62,976,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 24 | 0 | 24 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer` | 9 | 11,520,000 | 11,520,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 21,504,000 | 21,504,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 6 | 36,864,000 | 36,864,000 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 6 | 18,432,000 | 18,432,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 128,000 | 128,000 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 168,000 | 168,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 13,056,000 | 13,056,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 18 | 1,506,432,000 | 1,506,432,000 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 12 | 0 | 12 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 125,952,000 | 125,952,000 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 24 | 9,338,880 | 9,338,880 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 12 | 1,488,000,000 | 1,488,000,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.bbox_roi_extractor.*` | 6 | 6,000 | 6,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.mask_roi_extractor.*` | 6 | 600 | 600 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 24 | 9,338,880 | 9,338,880 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 125,952,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `rpn_head` | 5 | 21,826,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.relu` | 3 | 3,072,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.shared_convs.*.activate` | 12 | 150,528,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.convs.*.activate` | 12 | 60,211,200 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.relu` | 3 | 60,211,200 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.bbox_roi_extractor.*.roi_layers.*` | 12 | 37,632,000 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.mask_roi_extractor.*.roi_layers.*` | 12 | 15,052,800 | 0 |
| 0 (자료 이동·항등) | aten::max_pool2d | Float | `neck` | 1 | 66,560 | 0 |

</details>

<details><summary>ConvFormer-S36 — COCO Cascade Mask R-CNN 3×</summary>

(a) 0 · (b) 0 · (c) 1,346,151,100 · (d) 0 MACs
backbone만: (가) 156,502,784,000 · (a) 0 · (b) 0 · (c) 1,324,624,000 · (d) 0 · 합계 157,827,408,000 MACs; 전체 모델 측정의 backbone 몫 157,827,408,000 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 72 | 122,880,000 | 122,880,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 36 | 245,760,000 | 245,760,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 36 | 122,880,000 | 122,880,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 21,504,000 | 21,504,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 128,000 | 128,000 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 168,000 | 168,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 36 | 245,760,000 | 245,760,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 21 | 24,576,000 | 24,576,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 21 | 24,576,000 | 24,576,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 36 | 122,880,000 | 122,880,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 36 | 245,760,000 | 245,760,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 36 | 122,880,000 | 122,880,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.bbox_roi_extractor.*` | 6 | 6,000 | 6,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.mask_roi_extractor.*` | 6 | 600 | 600 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 36 | 245,760,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 36 | 122,880,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `rpn_head` | 5 | 21,826,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.relu` | 3 | 3,072,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.shared_convs.*.activate` | 12 | 150,528,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.convs.*.activate` | 12 | 60,211,200 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.relu` | 3 | 60,211,200 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.bbox_roi_extractor.*.roi_layers.*` | 12 | 37,632,000 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.mask_roi_extractor.*.roi_layers.*` | 12 | 15,052,800 | 0 |
| 0 (자료 이동·항등) | aten::max_pool2d | Float | `neck` | 1 | 66,560 | 0 |

</details>

<details><summary>SPANetV2-S36-pure — COCO Cascade Mask R-CNN 3×</summary>

(a) 1,665,024,000 · (b) 31,518,720 · (c) 1,254,789,964 · (d) 31,518,720 MACs
backbone만: (가) 166,198,016,000 · (a) 1,665,024,000 · (b) 31,518,720 · (c) 1,233,262,864 · (d) 31,518,720 · 합계 169,159,340,304 MACs; 전체 모델 측정의 backbone 몫 169,159,340,304 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 144 | 61,440,000 | 832,512,000 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 144 | 61,440,000 | 832,512,000 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 144 | 31,518,720 | 31,518,720 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 72 | 122,880,000 | 122,880,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 36 | 245,760,000 | 245,760,000 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 144 | 0 | 144 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 21,504,000 | 21,504,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 36 | 122,880,000 | 122,880,000 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 36 | 61,440,000 | 61,440,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 128,000 | 128,000 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 168,000 | 168,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 36 | 245,760,000 | 245,760,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 21 | 24,576,000 | 24,576,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 21 | 24,576,000 | 24,576,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 36 | 61,440,000 | 61,440,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 36 | 245,760,000 | 245,760,000 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 144 | 31,518,720 | 31,518,720 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.bbox_roi_extractor.*` | 6 | 6,000 | 6,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.mask_roi_extractor.*` | 6 | 600 | 600 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 144 | 31,518,720 | 31,518,720 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 36 | 245,760,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `rpn_head` | 5 | 21,826,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.relu` | 3 | 3,072,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.shared_convs.*.activate` | 12 | 150,528,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.convs.*.activate` | 12 | 60,211,200 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.relu` | 3 | 60,211,200 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.bbox_roi_extractor.*.roi_layers.*` | 12 | 37,632,000 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.mask_roi_extractor.*.roi_layers.*` | 12 | 15,052,800 | 0 |
| 0 (자료 이동·항등) | aten::max_pool2d | Float | `neck` | 1 | 66,560 | 0 |

</details>

<details><summary>CAFormer-S36 — COCO Cascade Mask R-CNN 3×</summary>

(a) 0 · (b) 0 · (c) 7,054,695,121 · (d) 0 MACs
backbone만: (가) 341,486,336,000 · (a) 0 · (b) 0 · (c) 7,033,168,021 · (d) 0 · 합계 348,519,504,021 MACs; 전체 모델 측정의 backbone 몫 348,519,504,021 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 72 | 122,880,000 | 122,880,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 36 | 245,760,000 | 245,760,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 15 | 73,728,000 | 73,728,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 21,504,000 | 21,504,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 128,000 | 128,000 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 168,000 | 168,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 36 | 245,760,000 | 245,760,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 21 | 24,576,000 | 24,576,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 21 | 24,576,000 | 24,576,000 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 21 | 0 | 21 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 21 | 2,928,000,000 | 2,928,000,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 15 | 73,728,000 | 73,728,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 36 | 245,760,000 | 245,760,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 15 | 73,728,000 | 73,728,000 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 21 | 2,928,000,000 | 2,928,000,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.bbox_roi_extractor.*` | 6 | 6,000 | 6,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.mask_roi_extractor.*` | 6 | 600 | 600 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 36 | 245,760,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 15 | 73,728,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `rpn_head` | 5 | 21,826,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.relu` | 3 | 3,072,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.shared_convs.*.activate` | 12 | 150,528,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.convs.*.activate` | 12 | 60,211,200 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.relu` | 3 | 60,211,200 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.bbox_roi_extractor.*.roi_layers.*` | 12 | 37,632,000 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.mask_roi_extractor.*.roi_layers.*` | 12 | 15,052,800 | 0 |
| 0 (자료 이동·항등) | aten::max_pool2d | Float | `neck` | 1 | 66,560 | 0 |

</details>

<details><summary>SPANetV2-S36-hybrid — COCO Cascade Mask R-CNN 3×</summary>

(a) 1,081,344,000 · (b) 18,739,200 · (c) 7,022,746,381 · (d) 18,739,200 MACs
backbone만: (가) 345,109,760,000 · (a) 1,081,344,000 · (b) 18,739,200 · (c) 7,001,219,281 · (d) 18,739,200 · 합계 353,229,801,681 MACs; 전체 모델 측정의 backbone 몫 353,229,801,681 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 60 | 36,864,000 | 540,672,000 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 60 | 36,864,000 | 540,672,000 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 60 | 18,739,200 | 18,739,200 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 72 | 122,880,000 | 122,880,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 36 | 245,760,000 | 245,760,000 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 60 | 0 | 60 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer` | 18 | 23,040,000 | 23,040,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 21,504,000 | 21,504,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::add | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::div | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 15 | 73,728,000 | 73,728,000 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 15 | 36,864,000 | 36,864,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::log2 | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 128,000 | 128,000 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 168,000 | 168,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 36 | 245,760,000 | 245,760,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 21 | 24,576,000 | 24,576,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 21 | 24,576,000 | 24,576,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 36 | 2,964,864,000 | 2,964,864,000 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 21 | 0 | 21 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::mul | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 36 | 245,760,000 | 245,760,000 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 60 | 18,739,200 | 18,739,200 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 21 | 2,928,000,000 | 2,928,000,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 64,000 | 64,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 84,000 | 84,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.bbox_roi_extractor.*` | 3 | 3,000 | 3,000 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `roi_head.mask_roi_extractor.*` | 3 | 300 | 300 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,096,000 | 4,096,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,424,000 | 7,424,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.bbox_roi_extractor.*` | 6 | 6,000 | 6,000 |
| (c) 그 밖의 원소별 | aten::sub | Float | `roi_head.mask_roi_extractor.*` | 6 | 600 | 600 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 60 | 18,739,200 | 18,739,200 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 36 | 245,760,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `rpn_head` | 5 | 21,826,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.relu` | 3 | 3,072,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.bbox_head.*.shared_convs.*.activate` | 12 | 150,528,000 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.convs.*.activate` | 12 | 60,211,200 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `roi_head.mask_head.*.relu` | 3 | 60,211,200 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.bbox_roi_extractor.*.roi_layers.*` | 12 | 37,632,000 | 0 |
| 0 (세지 않음, 참고) | prim::PythonOp.RoIAlignFunction | Float | `roi_head.mask_roi_extractor.*.roi_layers.*` | 12 | 15,052,800 | 0 |
| 0 (자료 이동·항등) | aten::max_pool2d | Float | `neck` | 1 | 66,560 | 0 |

</details>

<details><summary>ConvFormer-S18 — ADE20K UperNet</summary>

(a) 0 · (b) 0 · (c) 746,930,176 · (d) 0 MACs
backbone만: (가) 82,656,886,784 · (a) 0 · (b) 0 · (c) 719,405,056 · (d) 0 · 합계 83,376,291,840 MACs; 전체 모델 측정의 backbone 몫 83,376,291,840 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 27,525,120 | 27,525,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 64,487,424 | 64,487,424 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 128,974,848 | 128,974,848 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 64,487,424 | 64,487,424 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 131,072 | 131,072 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 172,032 | 172,032 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 128,974,848 | 128,974,848 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 13,369,344 | 13,369,344 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 13,369,344 | 13,369,344 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 64,487,424 | 64,487,424 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 128,974,848 | 128,974,848 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 64,487,424 | 64,487,424 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 128,974,848 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 18 | 64,487,424 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.fpn_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.lateral_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.bottleneck.activate` | 1 | 327,680 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.fpn_bottleneck.activate` | 1 | 20,971,520 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.psp_modules.*.*.activate` | 4 | 16,000 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 20,971,520 | 0 |

</details>

<details><summary>SPANetV2-S18-pure — ADE20K UperNet</summary>

(a) 893,386,752 · (b) 16,287,744 · (c) 698,730,568 · (d) 16,287,744 MACs
backbone만: (가) 87,590,174,720 · (a) 893,386,752 · (b) 16,287,744 · (c) 671,205,448 · (d) 16,287,744 · 합계 89,187,342,408 MACs; 전체 모델 측정의 backbone 몫 89,187,342,408 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 72 | 32,243,712 | 446,693,376 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 72 | 32,243,712 | 446,693,376 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 72 | 16,287,744 | 16,287,744 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 27,525,120 | 27,525,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 64,487,424 | 64,487,424 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 128,974,848 | 128,974,848 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 72 | 0 | 72 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 18 | 64,487,424 | 64,487,424 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 131,072 | 131,072 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 172,032 | 172,032 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 128,974,848 | 128,974,848 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 13,369,344 | 13,369,344 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 13,369,344 | 13,369,344 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 128,974,848 | 128,974,848 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 72 | 16,287,744 | 16,287,744 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 72 | 16,287,744 | 16,287,744 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 128,974,848 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.fpn_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.lateral_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.bottleneck.activate` | 1 | 327,680 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.fpn_bottleneck.activate` | 1 | 20,971,520 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.psp_modules.*.*.activate` | 4 | 16,000 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 20,971,520 | 0 |

</details>

<details><summary>CAFormer-S18 — ADE20K UperNet</summary>

(a) 0 · (b) 0 · (c) 3,787,276,300 · (d) 0 MACs
backbone만: (가) 181,204,680,704 · (a) 0 · (b) 0 · (c) 3,759,751,180 · (d) 0 · 합계 184,964,431,884 MACs; 전체 모델 측정의 backbone 몫 184,964,431,884 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 27,525,120 | 27,525,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 64,487,424 | 64,487,424 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 128,974,848 | 128,974,848 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 6 | 37,748,736 | 37,748,736 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 131,072 | 131,072 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 172,032 | 172,032 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 128,974,848 | 128,974,848 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 13,369,344 | 13,369,344 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 13,369,344 | 13,369,344 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 12 | 0 | 12 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 12 | 1,560,281,088 | 1,560,281,088 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 6 | 37,748,736 | 37,748,736 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 128,974,848 | 128,974,848 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 6 | 37,748,736 | 37,748,736 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 12 | 1,560,281,088 | 1,560,281,088 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 128,974,848 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 6 | 37,748,736 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.fpn_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.lateral_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.bottleneck.activate` | 1 | 327,680 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.fpn_bottleneck.activate` | 1 | 20,971,520 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.psp_modules.*.*.activate` | 4 | 16,000 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 20,971,520 | 0 |

</details>

<details><summary>SPANetV2-S18-hybrid — ADE20K UperNet</summary>

(a) 578,813,952 · (b) 9,486,336 · (c) 3,770,810,404 · (d) 9,486,336 MACs
backbone만: (가) 182,657,220,608 · (a) 578,813,952 · (b) 9,486,336 · (c) 3,743,285,284 · (d) 9,486,336 · 합계 186,998,292,516 MACs; 전체 모델 측정의 backbone 몫 186,998,292,516 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 24 | 18,874,368 | 289,406,976 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 24 | 18,874,368 | 289,406,976 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 24 | 9,486,336 | 9,486,336 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 27,525,120 | 27,525,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 64,487,424 | 64,487,424 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 128,974,848 | 128,974,848 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 24 | 0 | 24 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer` | 9 | 11,796,480 | 11,796,480 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 6 | 37,748,736 | 37,748,736 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 6 | 18,874,368 | 18,874,368 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 131,072 | 131,072 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 172,032 | 172,032 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 128,974,848 | 128,974,848 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 13,369,344 | 13,369,344 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 13,369,344 | 13,369,344 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 18 | 1,579,155,456 | 1,579,155,456 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 12 | 0 | 12 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 128,974,848 | 128,974,848 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 24 | 9,486,336 | 9,486,336 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 12 | 1,560,281,088 | 1,560,281,088 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 24 | 9,486,336 | 9,486,336 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 128,974,848 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.fpn_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.lateral_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.bottleneck.activate` | 1 | 327,680 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.fpn_bottleneck.activate` | 1 | 20,971,520 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.psp_modules.*.*.activate` | 4 | 16,000 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 20,971,520 | 0 |

</details>

<details><summary>ConvFormer-S36 — ADE20K UperNet</summary>

(a) 0 · (b) 0 · (c) 1,383,940,096 · (d) 0 MACs
backbone만: (가) 160,258,850,816 · (a) 0 · (b) 0 · (c) 1,356,414,976 · (d) 0 · 합계 161,615,265,792 MACs; 전체 모델 측정의 backbone 몫 161,615,265,792 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 27,525,120 | 27,525,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 72 | 125,829,120 | 125,829,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 36 | 251,658,240 | 251,658,240 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 36 | 125,829,120 | 125,829,120 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 131,072 | 131,072 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 172,032 | 172,032 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 36 | 251,658,240 | 251,658,240 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 21 | 25,165,824 | 25,165,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 21 | 25,165,824 | 25,165,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 36 | 125,829,120 | 125,829,120 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 36 | 251,658,240 | 251,658,240 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 36 | 125,829,120 | 125,829,120 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 36 | 251,658,240 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 36 | 125,829,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.fpn_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.lateral_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.bottleneck.activate` | 1 | 327,680 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.fpn_bottleneck.activate` | 1 | 20,971,520 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.psp_modules.*.*.activate` | 4 | 16,000 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 20,971,520 | 0 |

</details>

<details><summary>SPANetV2-S36-pure — ADE20K UperNet</summary>

(a) 1,704,984,576 · (b) 31,789,056 · (c) 1,289,900,176 · (d) 31,789,056 MACs
backbone만: (가) 170,186,768,384 · (a) 1,704,984,576 · (b) 31,789,056 · (c) 1,262,375,056 · (d) 31,789,056 · 합계 173,217,706,128 MACs; 전체 모델 측정의 backbone 몫 173,217,706,128 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 144 | 62,914,560 | 852,492,288 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 144 | 62,914,560 | 852,492,288 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 144 | 31,789,056 | 31,789,056 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 27,525,120 | 27,525,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 72 | 125,829,120 | 125,829,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 36 | 251,658,240 | 251,658,240 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 144 | 0 | 144 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 36 | 125,829,120 | 125,829,120 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 131,072 | 131,072 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 172,032 | 172,032 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 36 | 251,658,240 | 251,658,240 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 21 | 25,165,824 | 25,165,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 21 | 25,165,824 | 25,165,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 36 | 251,658,240 | 251,658,240 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 144 | 31,789,056 | 31,789,056 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 144 | 31,789,056 | 31,789,056 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 36 | 251,658,240 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.fpn_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.lateral_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.bottleneck.activate` | 1 | 327,680 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.fpn_bottleneck.activate` | 1 | 20,971,520 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.psp_modules.*.*.activate` | 4 | 16,000 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 20,971,520 | 0 |

</details>

<details><summary>CAFormer-S36 — ADE20K UperNet</summary>

(a) 0 · (b) 0 · (c) 7,373,406,229 · (d) 0 MACs
backbone만: (가) 354,287,353,856 · (a) 0 · (b) 0 · (c) 7,345,881,109 · (d) 0 · 합계 361,633,234,965 MACs; 전체 모델 측정의 backbone 몫 361,633,234,965 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 27,525,120 | 27,525,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 72 | 125,829,120 | 125,829,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 36 | 251,658,240 | 251,658,240 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 15 | 75,497,472 | 75,497,472 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 131,072 | 131,072 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 172,032 | 172,032 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 36 | 251,658,240 | 251,658,240 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 21 | 25,165,824 | 25,165,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 21 | 25,165,824 | 25,165,824 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 21 | 0 | 21 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 21 | 3,070,230,528 | 3,070,230,528 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 15 | 75,497,472 | 75,497,472 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 36 | 251,658,240 | 251,658,240 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 15 | 75,497,472 | 75,497,472 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 21 | 3,070,230,528 | 3,070,230,528 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 36 | 251,658,240 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 15 | 75,497,472 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.fpn_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.lateral_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.bottleneck.activate` | 1 | 327,680 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.fpn_bottleneck.activate` | 1 | 20,971,520 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.psp_modules.*.*.activate` | 4 | 16,000 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 20,971,520 | 0 |

</details>

<details><summary>SPANetV2-S36-hybrid — ADE20K UperNet</summary>

(a) 1,107,296,256 · (b) 18,997,248 · (c) 7,340,499,025 · (d) 18,997,248 MACs
backbone만: (가) 357,997,740,032 · (a) 1,107,296,256 · (b) 18,997,248 · (c) 7,312,973,905 · (d) 18,997,248 · 합계 366,456,004,689 MACs; 전체 모델 측정의 backbone 몫 366,456,004,689 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 60 | 37,748,736 | 553,648,128 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 60 | 37,748,736 | 553,648,128 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 60 | 18,997,248 | 18,997,248 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 27,525,120 | 27,525,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 72 | 125,829,120 | 125,829,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 36 | 251,658,240 | 251,658,240 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 60 | 0 | 60 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer` | 18 | 23,592,960 | 23,592,960 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 15 | 75,497,472 | 75,497,472 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 15 | 37,748,736 | 37,748,736 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 131,072 | 131,072 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 172,032 | 172,032 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 36 | 251,658,240 | 251,658,240 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 21 | 25,165,824 | 25,165,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 21 | 25,165,824 | 25,165,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 36 | 3,107,979,264 | 3,107,979,264 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 21 | 0 | 21 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 36 | 251,658,240 | 251,658,240 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 60 | 18,997,248 | 18,997,248 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 21 | 3,070,230,528 | 3,070,230,528 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 65,536 | 65,536 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 86,016 | 86,016 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 4,194,304 | 4,194,304 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 7,602,176 | 7,602,176 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 60 | 18,997,248 | 18,997,248 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 36 | 251,658,240 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.fpn_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `decode_head.lateral_convs.*.activate` | 3 | 27,525,120 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.bottleneck.activate` | 1 | 327,680 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.fpn_bottleneck.activate` | 1 | 20,971,520 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.psp_modules.*.*.activate` | 4 | 16,000 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 20,971,520 | 0 |

</details>

<details><summary>SPANet-S — ADE20K Semantic FPN</summary>

(a) 254,803,968 · (b) 10,747,904 · (c) 117,440,512 · (d) 0 MACs
backbone만: (가) 24,178,786,304 · (a) 254,803,968 · (b) 10,747,904 · (c) 105,644,032 · (d) 0 · 합계 24,549,982,208 MACs; 전체 모델 측정의 backbone 몫 24,549,982,208 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_fft2 | ComplexFloat | `backbone.network.*.*.token_mixer.sps.*` | 72 | 10,747,904 | 127,401,984 |
| (a) FFT | aten::fft_fftshift | ComplexFloat | `backbone.network.*.*.token_mixer.sps.*` | 72 | 10,747,904 | 0 |
| (a) FFT | aten::fft_ifft2 | ComplexFloat | `backbone.network.*.*.token_mixer.sps.*` | 72 | 10,747,904 | 127,401,984 |
| (a) FFT | aten::fft_ifftshift | ComplexFloat | `backbone.network.*.*.token_mixer.sps.*` | 72 | 10,747,904 | 0 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.network.*.*.token_mixer.sps.*` | 72 | 10,747,904 | 10,747,904 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.network.*.*` | 48 | 21,495,808 | 21,495,808 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.network.*.*.token_mixer` | 48 | 21,495,808 | 21,495,808 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 5,505,024 | 5,505,024 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.network.*.*.mlp.act` | 24 | 42,991,616 | 42,991,616 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.network.*.*.res_scale1` | 16 | 4,456,448 | 4,456,448 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.network.*.*.res_scale2` | 16 | 4,456,448 | 4,456,448 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.network.*.*.token_mixer` | 24 | 10,747,904 | 10,747,904 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.scale_heads.*.*.activate` | 7 | 3,964,928 | 0 |
| 0 (자료 이동·항등) | aten::clone | Float | `backbone.network.*.*.token_mixer` | 24 | 10,747,904 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 2,097,152 | 0 |
| 0 (자료 이동·항등) | aten::real | Float | `backbone.network.*.*.token_mixer.sps.*` | 72 | 10,747,904 | 0 |

</details>

<details><summary>DFFormer-S18 — ADE20K Semantic FPN</summary>

(a) 382,205,952 · (b) 8,392,704 · (c) 256,755,250 · (d) 292,992 MACs
backbone만: (가) 19,831,192,784.0 · (a) 382,205,952 · (b) 8,392,704 · (c) 244,958,770 · (d) 292,992 · 합계 20,467,043,202.0 MACs; 전체 모델 측정의 backbone 몫 20,467,043,202.0 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 18 | 16,121,856 | 191,102,976 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 18 | 16,121,856 | 191,102,976 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 18 | 8,392,704 | 8,392,704 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 16,121,856 | 16,121,856 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.norm1` | 18 | 71,424 | 71,424 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.norm2` | 18 | 71,424 | 71,424 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 16,121,856 | 16,121,856 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.reweight.act` | 18 | 1,248 | 1,248 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 5,505,024 | 5,505,024 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.stages.*.*.norm1` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.stages.*.*.norm2` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 32,768 | 32,768 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 43,008 | 43,008 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.stages.*.*.norm1` | 36 | 142,848 | 142,848 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.stages.*.*.norm2` | 36 | 142,848 | 142,848 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.stages.*.*.token_mixer` | 18 | 4,992 | 4,992 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.norm1` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.norm2` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 3,342,336 | 3,342,336 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 3,342,336 | 3,342,336 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 18 | 0 | 18 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 16,121,856 | 16,121,856 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.reweight.act` | 18 | 1,248 | 1,248 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.norm1` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.norm2` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 16,121,856 | 16,121,856 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.reweight.act` | 18 | 1,248 | 1,248 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 18 | 39,936 | 39,936 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.stages.*.*.norm1` | 18 | 71,424 | 71,424 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.stages.*.*.norm2` | 18 | 71,424 | 71,424 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.stages.*.*.norm1` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.stages.*.*.norm2` | 18 | 8,060,928 | 8,060,928 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 18 | 292,992 | 292,992 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 32,243,712 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 18 | 16,121,856 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.reweight.act.relu` | 18 | 1,248 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.scale_heads.*.*.activate` | 7 | 3,964,928 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 2,097,152 | 0 |
| 0 (자료 이동·항등) | aten::view_as_complex | ComplexFloat | `backbone.stages.*.*.token_mixer` | 18 | 146,496 | 0 |

</details>

<details><summary>SPANetV2-S18-pure — ADE20K Semantic FPN</summary>

(a) 191,102,976 · (b) 4,196,352 · (c) 179,722,312 · (d) 4,196,352 MACs
backbone만: (가) 21,897,543,680 · (a) 191,102,976 · (b) 4,196,352 · (c) 167,925,832 · (d) 4,196,352 · 합계 22,264,965,192 MACs; 전체 모델 측정의 backbone 몫 22,264,965,192 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 72 | 8,060,928 | 95,551,488 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 72 | 8,060,928 | 95,551,488 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 72 | 4,196,352 | 4,196,352 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 16,121,856 | 16,121,856 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 72 | 0 | 72 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 5,505,024 | 5,505,024 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 18 | 16,121,856 | 16,121,856 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 32,768 | 32,768 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 43,008 | 43,008 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 3,342,336 | 3,342,336 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 3,342,336 | 3,342,336 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 72 | 4,196,352 | 4,196,352 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 72 | 4,196,352 | 4,196,352 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 32,243,712 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.scale_heads.*.*.activate` | 7 | 3,964,928 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 2,097,152 | 0 |

</details>

<details><summary>CDFFormer-S18 — ADE20K Semantic FPN</summary>

(a) 130,547,712 · (b) 3,575,808 · (c) 256,749,628 · (d) 42,624 MACs
backbone만: (가) 20,274,209,060.0 · (a) 130,547,712 · (b) 3,575,808 · (c) 244,953,148 · (d) 42,624 · 합계 20,653,328,352.0 MACs; 전체 모델 측정의 backbone 몫 20,653,328,352.0 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 12 | 6,684,672 | 65,273,856 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 12 | 6,684,672 | 65,273,856 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 12 | 3,575,808 | 3,575,808 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 16,121,856 | 16,121,856 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.norm1` | 18 | 71,424 | 71,424 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.norm2` | 18 | 71,424 | 71,424 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 16,121,856 | 16,121,856 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.reweight.act` | 12 | 1,104 | 1,104 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 5,505,024 | 5,505,024 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.stages.*.*.norm1` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.stages.*.*.norm2` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 32,768 | 32,768 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 43,008 | 43,008 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.stages.*.*.norm1` | 36 | 142,848 | 142,848 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.stages.*.*.norm2` | 36 | 142,848 | 142,848 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.stages.*.*.token_mixer` | 12 | 4,416 | 4,416 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.norm1` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.norm2` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 3,342,336 | 3,342,336 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 3,342,336 | 3,342,336 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 12 | 0 | 12 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 16,121,856 | 16,121,856 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.reweight.act` | 12 | 1,104 | 1,104 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.norm1` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.norm2` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 18 | 16,121,856 | 16,121,856 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.reweight.act` | 12 | 1,104 | 1,104 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 12 | 35,328 | 35,328 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.stages.*.*.norm1` | 18 | 71,424 | 71,424 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.stages.*.*.norm2` | 18 | 71,424 | 71,424 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.stages.*.*.norm1` | 18 | 8,060,928 | 8,060,928 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.stages.*.*.norm2` | 18 | 8,060,928 | 8,060,928 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 12 | 42,624 | 42,624 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 32,243,712 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 18 | 16,121,856 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.reweight.act.relu` | 12 | 1,104 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.scale_heads.*.*.activate` | 7 | 3,964,928 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 2,097,152 | 0 |
| 0 (자료 이동·항등) | aten::view_as_complex | ComplexFloat | `backbone.stages.*.*.token_mixer` | 12 | 21,312 | 0 |

</details>

<details><summary>SPANetV2-S18-hybrid — ADE20K Semantic FPN</summary>

(a) 125,829,120 · (b) 2,408,448 · (c) 362,549,284 · (d) 2,408,448 MACs
backbone만: (가) 26,940,932,096 · (a) 125,829,120 · (b) 2,408,448 · (c) 350,752,804 · (d) 2,408,448 · 합계 27,422,330,916 MACs; 전체 모델 측정의 backbone 몫 27,422,330,916 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 24 | 4,718,592 | 62,914,560 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 24 | 4,718,592 | 62,914,560 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 24 | 2,408,448 | 2,408,448 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 36 | 16,121,856 | 16,121,856 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 24 | 0 | 24 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer` | 9 | 2,949,120 | 2,949,120 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 5,505,024 | 5,505,024 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 6 | 9,437,184 | 9,437,184 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 6 | 4,718,592 | 4,718,592 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 32,768 | 32,768 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 43,008 | 43,008 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 12 | 3,342,336 | 3,342,336 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 12 | 3,342,336 | 3,342,336 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 18 | 102,236,160 | 102,236,160 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 12 | 0 | 12 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 18 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 24 | 2,408,448 | 2,408,448 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 12 | 97,517,568 | 97,517,568 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 24 | 2,408,448 | 2,408,448 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 18 | 32,243,712 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.scale_heads.*.*.activate` | 7 | 3,964,928 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 2,097,152 | 0 |

</details>

<details><summary>SPANet-M — ADE20K Semantic FPN</summary>

(a) 382,205,952 · (b) 16,121,856 · (c) 170,262,528 · (d) 0 MACs
backbone만: (가) 35,657,547,776 · (a) 382,205,952 · (b) 16,121,856 · (c) 158,466,048 · (d) 0 · 합계 36,214,341,632 MACs; 전체 모델 측정의 backbone 몫 36,214,341,632 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_fft2 | ComplexFloat | `backbone.network.*.*.token_mixer.sps.*` | 108 | 16,121,856 | 191,102,976 |
| (a) FFT | aten::fft_fftshift | ComplexFloat | `backbone.network.*.*.token_mixer.sps.*` | 108 | 16,121,856 | 0 |
| (a) FFT | aten::fft_ifft2 | ComplexFloat | `backbone.network.*.*.token_mixer.sps.*` | 108 | 16,121,856 | 191,102,976 |
| (a) FFT | aten::fft_ifftshift | ComplexFloat | `backbone.network.*.*.token_mixer.sps.*` | 108 | 16,121,856 | 0 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.network.*.*.token_mixer.sps.*` | 108 | 16,121,856 | 16,121,856 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.network.*.*` | 72 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.network.*.*.token_mixer` | 72 | 32,243,712 | 32,243,712 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 5,505,024 | 5,505,024 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.network.*.*.mlp.act` | 36 | 64,487,424 | 64,487,424 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.network.*.*.res_scale1` | 24 | 6,684,672 | 6,684,672 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.network.*.*.res_scale2` | 24 | 6,684,672 | 6,684,672 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.network.*.*.token_mixer` | 36 | 16,121,856 | 16,121,856 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.scale_heads.*.*.activate` | 7 | 3,964,928 | 0 |
| 0 (자료 이동·항등) | aten::clone | Float | `backbone.network.*.*.token_mixer` | 36 | 16,121,856 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 2,097,152 | 0 |
| 0 (자료 이동·항등) | aten::real | Float | `backbone.network.*.*.token_mixer.sps.*` | 108 | 16,121,856 | 0 |

</details>

<details><summary>DFFormer-S36 — ADE20K Semantic FPN</summary>

(a) 726,663,168 · (b) 16,392,192 · (c) 477,757,396 · (d) 484,224 MACs
backbone만: (가) 38,437,971,272.0 · (a) 726,663,168 · (b) 16,392,192 · (c) 465,960,916 · (d) 484,224 · 합계 39,647,471,772.0 MACs; 전체 모델 측정의 backbone 몫 39,647,471,772.0 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 36 | 31,457,280 | 363,331,584 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 36 | 31,457,280 | 363,331,584 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 36 | 16,392,192 | 16,392,192 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 72 | 31,457,280 | 31,457,280 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.norm1` | 36 | 117,504 | 117,504 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.norm2` | 36 | 117,504 | 117,504 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 36 | 31,457,280 | 31,457,280 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.reweight.act` | 36 | 2,256 | 2,256 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 5,505,024 | 5,505,024 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.stages.*.*.norm1` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.stages.*.*.norm2` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 32,768 | 32,768 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 43,008 | 43,008 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.stages.*.*.norm1` | 72 | 235,008 | 235,008 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.stages.*.*.norm2` | 72 | 235,008 | 235,008 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.stages.*.*.token_mixer` | 36 | 9,024 | 9,024 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.norm1` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.norm2` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 21 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 21 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 36 | 0 | 36 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 36 | 31,457,280 | 31,457,280 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.reweight.act` | 36 | 2,256 | 2,256 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.norm1` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.norm2` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 36 | 31,457,280 | 31,457,280 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.reweight.act` | 36 | 2,256 | 2,256 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 36 | 72,192 | 72,192 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.stages.*.*.norm1` | 36 | 117,504 | 117,504 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.stages.*.*.norm2` | 36 | 117,504 | 117,504 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.stages.*.*.norm1` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.stages.*.*.norm2` | 36 | 15,728,640 | 15,728,640 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 36 | 484,224 | 484,224 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 36 | 62,914,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 36 | 31,457,280 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.reweight.act.relu` | 36 | 2,256 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.scale_heads.*.*.activate` | 7 | 3,964,928 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 2,097,152 | 0 |
| 0 (자료 이동·항등) | aten::view_as_complex | ComplexFloat | `backbone.stages.*.*.token_mixer` | 36 | 242,112 | 0 |

</details>

<details><summary>SPANetV2-S36-pure — ADE20K Semantic FPN</summary>

(a) 363,331,584 · (b) 8,196,096 · (c) 327,639,184 · (d) 8,196,096 MACs
backbone만: (가) 42,546,692,096 · (a) 363,331,584 · (b) 8,196,096 · (c) 315,842,704 · (d) 8,196,096 · 합계 43,242,258,576 MACs; 전체 모델 측정의 backbone 몫 43,242,258,576 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 144 | 15,728,640 | 181,665,792 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 144 | 15,728,640 | 181,665,792 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 144 | 8,196,096 | 8,196,096 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 72 | 31,457,280 | 31,457,280 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 144 | 0 | 144 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 5,505,024 | 5,505,024 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 36 | 31,457,280 | 31,457,280 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 32,768 | 32,768 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 43,008 | 43,008 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 21 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 21 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 144 | 8,196,096 | 8,196,096 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 144 | 8,196,096 | 8,196,096 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 36 | 62,914,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.scale_heads.*.*.activate` | 7 | 3,964,928 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 2,097,152 | 0 |

</details>

<details><summary>CDFFormer-S36 — ADE20K Semantic FPN</summary>

(a) 248,512,512 · (b) 6,709,248 · (c) 477,740,533 · (d) 81,792 MACs
backbone만: (가) 39,323,615,288.0 · (a) 248,512,512 · (b) 6,709,248 · (c) 465,944,053 · (d) 81,792 · 합계 40,044,862,893.0 MACs; 전체 모델 측정의 backbone 몫 40,044,862,893.0 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 21 | 12,582,912 | 124,256,256 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 21 | 12,582,912 | 124,256,256 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 21 | 6,709,248 | 6,709,248 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 72 | 31,457,280 | 31,457,280 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.norm1` | 36 | 117,504 | 117,504 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.norm2` | 36 | 117,504 | 117,504 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.act1` | 36 | 31,457,280 | 31,457,280 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer.reweight.act` | 21 | 1,824 | 1,824 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 5,505,024 | 5,505,024 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.stages.*.*.norm1` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.stages.*.*.norm2` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 32,768 | 32,768 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 43,008 | 43,008 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.stages.*.*.norm1` | 72 | 235,008 | 235,008 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.stages.*.*.norm2` | 72 | 235,008 | 235,008 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.stages.*.*.token_mixer` | 21 | 7,296 | 7,296 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.norm1` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.norm2` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 21 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 21 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 21 | 0 | 21 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.act1` | 36 | 31,457,280 | 31,457,280 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer.reweight.act` | 21 | 1,824 | 1,824 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.norm1` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.norm2` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.act1` | 36 | 31,457,280 | 31,457,280 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.token_mixer.reweight.act` | 21 | 1,824 | 1,824 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 21 | 58,368 | 58,368 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.stages.*.*.norm1` | 36 | 117,504 | 117,504 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.stages.*.*.norm2` | 36 | 117,504 | 117,504 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.stages.*.*.norm1` | 36 | 15,728,640 | 15,728,640 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.stages.*.*.norm2` | 36 | 15,728,640 | 15,728,640 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 21 | 81,792 | 81,792 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 36 | 62,914,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.act1.relu` | 36 | 31,457,280 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.token_mixer.reweight.act.relu` | 21 | 1,824 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.scale_heads.*.*.activate` | 7 | 3,964,928 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 2,097,152 | 0 |
| 0 (자료 이동·항등) | aten::view_as_complex | ComplexFloat | `backbone.stages.*.*.token_mixer` | 21 | 40,896 | 0 |

</details>

<details><summary>SPANetV2-S36-hybrid — ADE20K Semantic FPN</summary>

(a) 239,075,328 · (b) 4,841,472 · (c) 688,795,729 · (d) 4,841,472 MACs
backbone만: (가) 52,656,668,672 · (a) 239,075,328 · (b) 4,841,472 · (c) 676,999,249 · (d) 4,841,472 · 합계 53,582,426,193 MACs; 전체 모델 측정의 backbone 몫 53,582,426,193 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `backbone.stages.*.*.token_mixer` | 60 | 9,437,184 | 119,537,664 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `backbone.stages.*.*.token_mixer` | 60 | 9,437,184 | 119,537,664 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `backbone.stages.*.*.token_mixer` | 60 | 4,841,472 | 4,841,472 |
| (c) 그 밖의 원소별 | aten::add | Float | `(root)` | 3 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*` | 72 | 31,457,280 | 31,457,280 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.mlp.act` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::add | Long | `backbone.stages.*.*.token_mixer` | 60 | 0 | 60 |
| (c) 그 밖의 원소별 | aten::add | Float | `backbone.stages.*.*.token_mixer` | 18 | 5,898,240 | 5,898,240 |
| (c) 그 밖의 원소별 | aten::add | Float | `neck` | 3 | 5,505,024 | 5,505,024 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::div | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.proj.*` | 15 | 18,874,368 | 18,874,368 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `backbone.stages.*.*.token_mixer.query.*` | 15 | 9,437,184 | 9,437,184 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.post_norm` | 2 | 32,768 | 32,768 |
| (c) 그 밖의 원소별 | aten::mean | Float | `backbone.downsample_layers.*.pre_norm` | 6 | 43,008 | 43,008 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.mlp.act` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale1` | 21 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.res_scale2` | 21 | 6,291,456 | 6,291,456 |
| (c) 그 밖의 원소별 | aten::mul | Float | `backbone.stages.*.*.token_mixer` | 36 | 201,326,592 | 201,326,592 |
| (c) 그 밖의 원소별 | aten::mul | Long | `backbone.stages.*.*.token_mixer` | 21 | 0 | 21 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (c) 그 밖의 원소별 | aten::pow | Float | `backbone.stages.*.*.mlp.act` | 36 | 62,914,560 | 62,914,560 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `backbone.stages.*.*.token_mixer` | 60 | 4,841,472 | 4,841,472 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `backbone.stages.*.*.token_mixer` | 21 | 191,889,408 | 191,889,408 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.post_norm` | 1 | 16,384 | 16,384 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 21,504 | 21,504 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.post_norm` | 1 | 1,048,576 | 1,048,576 |
| (c) 그 밖의 원소별 | aten::sub | Float | `backbone.downsample_layers.*.pre_norm` | 3 | 1,900,544 | 1,900,544 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `backbone.stages.*.*.token_mixer` | 60 | 4,841,472 | 4,841,472 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `backbone.stages.*.*.mlp.act.relu` | 36 | 62,914,560 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu_ | Float | `decode_head.scale_heads.*.*.activate` | 7 | 3,964,928 | 0 |
| 0 (자료 이동·항등) | aten::feature_dropout | Float | `decode_head.dropout` | 1 | 2,097,152 | 0 |

</details>

<details><summary>ConvFormer-S18 — ImageNet-1K 384×384 (분류)</summary>

(a) 0 · (b) 0 · (c) 101,168,896 · (d) 0 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*` | 36 | 9,068,544 | 9,068,544 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.mlp.act` | 18 | 18,137,088 | 18,137,088 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.token_mixer.act1` | 18 | 9,068,544 | 9,068,544 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::mean | Float | `(root)` | 1 | 512 | 512 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.post_norm` | 2 | 18,432 | 18,432 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.pre_norm` | 6 | 24,192 | 24,192 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.mlp.act` | 18 | 18,137,088 | 18,137,088 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale1` | 12 | 1,880,064 | 1,880,064 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale2` | 12 | 1,880,064 | 1,880,064 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.token_mixer.act1` | 18 | 9,068,544 | 9,068,544 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::pow | Float | `stages.*.*.mlp.act` | 18 | 18,137,088 | 18,137,088 |
| (c) 그 밖의 원소별 | aten::pow | Float | `stages.*.*.token_mixer.act1` | 18 | 9,068,544 | 9,068,544 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::square | Float | `head.act` | 1 | 2,048 | 2,048 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `head.act.relu` | 1 | 2,048 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `stages.*.*.mlp.act.relu` | 18 | 18,137,088 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `stages.*.*.token_mixer.act1.relu` | 18 | 9,068,544 | 0 |

</details>

<details><summary>SPANetV2-S18-pure — ImageNet-1K 384×384 (분류)</summary>

(a) 107,495,424 · (b) 2,391,552 · (c) 94,491,976 · (d) 2,391,552 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `stages.*.*.token_mixer` | 72 | 4,534,272 | 53,747,712 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `stages.*.*.token_mixer` | 72 | 4,534,272 | 53,747,712 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `stages.*.*.token_mixer` | 72 | 2,391,552 | 2,391,552 |
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*` | 36 | 9,068,544 | 9,068,544 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.mlp.act` | 18 | 18,137,088 | 18,137,088 |
| (c) 그 밖의 원소별 | aten::add | Long | `stages.*.*.token_mixer` | 72 | 0 | 72 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `stages.*.*.token_mixer.proj.*` | 18 | 9,068,544 | 9,068,544 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `stages.*.*.token_mixer.query.*` | 18 | 4,534,272 | 4,534,272 |
| (c) 그 밖의 원소별 | aten::mean | Float | `(root)` | 1 | 512 | 512 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.post_norm` | 2 | 18,432 | 18,432 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.pre_norm` | 6 | 24,192 | 24,192 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.mlp.act` | 18 | 18,137,088 | 18,137,088 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale1` | 12 | 1,880,064 | 1,880,064 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale2` | 12 | 1,880,064 | 1,880,064 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.token_mixer` | 18 | 4,534,272 | 4,534,272 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::pow | Float | `stages.*.*.mlp.act` | 18 | 18,137,088 | 18,137,088 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `stages.*.*.token_mixer` | 72 | 2,391,552 | 2,391,552 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::square | Float | `head.act` | 1 | 2,048 | 2,048 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `stages.*.*.token_mixer` | 72 | 2,391,552 | 2,391,552 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `head.act.relu` | 1 | 2,048 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `stages.*.*.mlp.act.relu` | 18 | 18,137,088 | 0 |

</details>

<details><summary>CAFormer-S18 — ImageNet-1K 384×384 (분류)</summary>

(a) 0 · (b) 0 · (c) 151,598,860 · (d) 0 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*` | 36 | 9,068,544 | 9,068,544 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.mlp.act` | 18 | 18,137,088 | 18,137,088 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.token_mixer.act1` | 6 | 5,308,416 | 5,308,416 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::mean | Float | `(root)` | 1 | 512 | 512 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.post_norm` | 2 | 18,432 | 18,432 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.pre_norm` | 6 | 24,192 | 24,192 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.mlp.act` | 18 | 18,137,088 | 18,137,088 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale1` | 12 | 1,880,064 | 1,880,064 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale2` | 12 | 1,880,064 | 1,880,064 |
| (c) 그 밖의 원소별 | aten::mul | Long | `stages.*.*.token_mixer` | 12 | 0 | 12 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.token_mixer` | 12 | 30,855,168 | 30,855,168 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.token_mixer.act1` | 6 | 5,308,416 | 5,308,416 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::pow | Float | `stages.*.*.mlp.act` | 18 | 18,137,088 | 18,137,088 |
| (c) 그 밖의 원소별 | aten::pow | Float | `stages.*.*.token_mixer.act1` | 6 | 5,308,416 | 5,308,416 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `stages.*.*.token_mixer` | 12 | 30,855,168 | 30,855,168 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::square | Float | `head.act` | 1 | 2,048 | 2,048 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `head.act.relu` | 1 | 2,048 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `stages.*.*.mlp.act.relu` | 18 | 18,137,088 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `stages.*.*.token_mixer.act1.relu` | 6 | 5,308,416 | 0 |

</details>

<details><summary>SPANetV2-S18-hybrid — ImageNet-1K 384×384 (분류)</summary>

(a) 70,778,880 · (b) 1,363,968 · (c) 149,313,316 · (d) 1,363,968 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `stages.*.*.token_mixer` | 24 | 2,654,208 | 35,389,440 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `stages.*.*.token_mixer` | 24 | 2,654,208 | 35,389,440 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `stages.*.*.token_mixer` | 24 | 1,363,968 | 1,363,968 |
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*` | 36 | 9,068,544 | 9,068,544 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.mlp.act` | 18 | 18,137,088 | 18,137,088 |
| (c) 그 밖의 원소별 | aten::add | Long | `stages.*.*.token_mixer` | 24 | 0 | 24 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.token_mixer` | 9 | 1,658,880 | 1,658,880 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `stages.*.*.token_mixer.proj.*` | 6 | 5,308,416 | 5,308,416 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `stages.*.*.token_mixer.query.*` | 6 | 2,654,208 | 2,654,208 |
| (c) 그 밖의 원소별 | aten::mean | Float | `(root)` | 1 | 512 | 512 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.post_norm` | 2 | 18,432 | 18,432 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.pre_norm` | 6 | 24,192 | 24,192 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.mlp.act` | 18 | 18,137,088 | 18,137,088 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale1` | 12 | 1,880,064 | 1,880,064 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale2` | 12 | 1,880,064 | 1,880,064 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.token_mixer` | 18 | 33,509,376 | 33,509,376 |
| (c) 그 밖의 원소별 | aten::mul | Long | `stages.*.*.token_mixer` | 12 | 0 | 12 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::pow | Float | `stages.*.*.mlp.act` | 18 | 18,137,088 | 18,137,088 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `stages.*.*.token_mixer` | 24 | 1,363,968 | 1,363,968 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `stages.*.*.token_mixer` | 12 | 30,855,168 | 30,855,168 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::square | Float | `head.act` | 1 | 2,048 | 2,048 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `stages.*.*.token_mixer` | 24 | 1,363,968 | 1,363,968 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `head.act.relu` | 1 | 2,048 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `stages.*.*.mlp.act.relu` | 18 | 18,137,088 | 0 |

</details>

<details><summary>ConvFormer-S36 — ImageNet-1K 384×384 (분류)</summary>

(a) 0 · (b) 0 · (c) 190,748,416 · (d) 0 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*` | 72 | 17,694,720 | 17,694,720 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.mlp.act` | 36 | 35,389,440 | 35,389,440 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.token_mixer.act1` | 36 | 17,694,720 | 17,694,720 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::mean | Float | `(root)` | 1 | 512 | 512 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.post_norm` | 2 | 18,432 | 18,432 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.pre_norm` | 6 | 24,192 | 24,192 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.mlp.act` | 36 | 35,389,440 | 35,389,440 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale1` | 21 | 3,538,944 | 3,538,944 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale2` | 21 | 3,538,944 | 3,538,944 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.token_mixer.act1` | 36 | 17,694,720 | 17,694,720 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::pow | Float | `stages.*.*.mlp.act` | 36 | 35,389,440 | 35,389,440 |
| (c) 그 밖의 원소별 | aten::pow | Float | `stages.*.*.token_mixer.act1` | 36 | 17,694,720 | 17,694,720 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::square | Float | `head.act` | 1 | 2,048 | 2,048 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `head.act.relu` | 1 | 2,048 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `stages.*.*.mlp.act.relu` | 36 | 35,389,440 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `stages.*.*.token_mixer.act1.relu` | 36 | 17,694,720 | 0 |

</details>

<details><summary>SPANetV2-S36-pure — ImageNet-1K 384×384 (분류)</summary>

(a) 204,374,016 · (b) 4,672,512 · (c) 177,726,352 · (d) 4,672,512 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `stages.*.*.token_mixer` | 144 | 8,847,360 | 102,187,008 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `stages.*.*.token_mixer` | 144 | 8,847,360 | 102,187,008 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `stages.*.*.token_mixer` | 144 | 4,672,512 | 4,672,512 |
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*` | 72 | 17,694,720 | 17,694,720 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.mlp.act` | 36 | 35,389,440 | 35,389,440 |
| (c) 그 밖의 원소별 | aten::add | Long | `stages.*.*.token_mixer` | 144 | 0 | 144 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `stages.*.*.token_mixer.proj.*` | 36 | 17,694,720 | 17,694,720 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `stages.*.*.token_mixer.query.*` | 36 | 8,847,360 | 8,847,360 |
| (c) 그 밖의 원소별 | aten::mean | Float | `(root)` | 1 | 512 | 512 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.post_norm` | 2 | 18,432 | 18,432 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.pre_norm` | 6 | 24,192 | 24,192 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.mlp.act` | 36 | 35,389,440 | 35,389,440 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale1` | 21 | 3,538,944 | 3,538,944 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale2` | 21 | 3,538,944 | 3,538,944 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.token_mixer` | 36 | 8,847,360 | 8,847,360 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::pow | Float | `stages.*.*.mlp.act` | 36 | 35,389,440 | 35,389,440 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `stages.*.*.token_mixer` | 144 | 4,672,512 | 4,672,512 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::square | Float | `head.act` | 1 | 2,048 | 2,048 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `stages.*.*.token_mixer` | 144 | 4,672,512 | 4,672,512 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `head.act.relu` | 1 | 2,048 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `stages.*.*.mlp.act.relu` | 36 | 35,389,440 | 0 |

</details>

<details><summary>CAFormer-S36 — ImageNet-1K 384×384 (분류)</summary>

(a) 0 · (b) 0 · (c) 290,944,789 · (d) 0 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*` | 72 | 17,694,720 | 17,694,720 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.mlp.act` | 36 | 35,389,440 | 35,389,440 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.token_mixer.act1` | 15 | 10,616,832 | 10,616,832 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::mean | Float | `(root)` | 1 | 512 | 512 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.post_norm` | 2 | 18,432 | 18,432 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.pre_norm` | 6 | 24,192 | 24,192 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.mlp.act` | 36 | 35,389,440 | 35,389,440 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale1` | 21 | 3,538,944 | 3,538,944 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale2` | 21 | 3,538,944 | 3,538,944 |
| (c) 그 밖의 원소별 | aten::mul | Long | `stages.*.*.token_mixer` | 21 | 0 | 21 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.token_mixer` | 21 | 60,715,008 | 60,715,008 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.token_mixer.act1` | 15 | 10,616,832 | 10,616,832 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::pow | Float | `stages.*.*.mlp.act` | 36 | 35,389,440 | 35,389,440 |
| (c) 그 밖의 원소별 | aten::pow | Float | `stages.*.*.token_mixer.act1` | 15 | 10,616,832 | 10,616,832 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `stages.*.*.token_mixer` | 21 | 60,715,008 | 60,715,008 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::square | Float | `head.act` | 1 | 2,048 | 2,048 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `head.act.relu` | 1 | 2,048 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `stages.*.*.mlp.act.relu` | 36 | 35,389,440 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `stages.*.*.token_mixer.act1.relu` | 15 | 10,616,832 | 0 |

</details>

<details><summary>SPANetV2-S36-hybrid — ImageNet-1K 384×384 (분류)</summary>

(a) 134,479,872 · (b) 2,746,368 · (c) 286,392,145 · (d) 2,746,368 MACs

| 항목 | 연산 | dtype | 모듈 (번호는 *) | 호출 수 | 원소 수 | MACs |
|---|---|---|---|---|---|---|
| (a) FFT | aten::fft_irfft2 | Float | `stages.*.*.token_mixer` | 60 | 5,308,416 | 67,239,936 |
| (a) FFT | aten::fft_rfft2 | ComplexFloat | `stages.*.*.token_mixer` | 60 | 5,308,416 | 67,239,936 |
| (b) 주파수 영역 곱 | aten::mul | ComplexFloat | `stages.*.*.token_mixer` | 60 | 2,746,368 | 2,746,368 |
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::add | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*` | 72 | 17,694,720 | 17,694,720 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.mlp.act` | 36 | 35,389,440 | 35,389,440 |
| (c) 그 밖의 원소별 | aten::add | Long | `stages.*.*.token_mixer` | 60 | 0 | 60 |
| (c) 그 밖의 원소별 | aten::add | Float | `stages.*.*.token_mixer` | 18 | 3,317,760 | 3,317,760 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::div | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `stages.*.*.token_mixer.proj.*` | 15 | 10,616,832 | 10,616,832 |
| (c) 그 밖의 원소별 | aten::gelu | Float | `stages.*.*.token_mixer.query.*` | 15 | 5,308,416 | 5,308,416 |
| (c) 그 밖의 원소별 | aten::mean | Float | `(root)` | 1 | 512 | 512 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.post_norm` | 2 | 18,432 | 18,432 |
| (c) 그 밖의 원소별 | aten::mean | Float | `downsample_layers.*.pre_norm` | 6 | 24,192 | 24,192 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::mul | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.mlp.act` | 36 | 35,389,440 | 35,389,440 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale1` | 21 | 3,538,944 | 3,538,944 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.res_scale2` | 21 | 3,538,944 | 3,538,944 |
| (c) 그 밖의 원소별 | aten::mul | Float | `stages.*.*.token_mixer` | 36 | 66,023,424 | 66,023,424 |
| (c) 그 밖의 원소별 | aten::mul | Long | `stages.*.*.token_mixer` | 21 | 0 | 21 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::pow | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (c) 그 밖의 원소별 | aten::pow | Float | `stages.*.*.mlp.act` | 36 | 35,389,440 | 35,389,440 |
| (c) 그 밖의 원소별 | aten::sigmoid | Float | `stages.*.*.token_mixer` | 60 | 2,746,368 | 2,746,368 |
| (c) 그 밖의 원소별 | aten::softmax | Float | `stages.*.*.token_mixer` | 21 | 60,715,008 | 60,715,008 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.post_norm` | 1 | 9,216 | 9,216 |
| (c) 그 밖의 원소별 | aten::sqrt | Float | `downsample_layers.*.pre_norm` | 3 | 12,096 | 12,096 |
| (c) 그 밖의 원소별 | aten::square | Float | `head.act` | 1 | 2,048 | 2,048 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.post_norm` | 1 | 589,824 | 589,824 |
| (c) 그 밖의 원소별 | aten::sub | Float | `downsample_layers.*.pre_norm` | 3 | 1,069,056 | 1,069,056 |
| (d) 마스크·필터 크기 조정 | aten::upsample_bicubic2d | Float | `stages.*.*.token_mixer` | 60 | 2,746,368 | 2,746,368 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `head.act.relu` | 1 | 2,048 | 0 |
| 0 (mmengine 무시 목록, 참고) | aten::relu | Float | `stages.*.*.mlp.act.relu` | 36 | 35,389,440 | 0 |

</details>

## 5. 논문 값과 1% 넘게 다른 모델 — 설정 확인

논문 값과 비교했다. 차이 = (재측정 − 논문) / 논문이다. "반올림"은 논문 값의 자릿수로 반올림한 값이다(half up). MACs는 (가), 즉 각 저장소 도구와 같은 기본 분석으로 비교했다. 합계는 논문이 세지 않은 연산까지 더한 값이라 판정에 쓰지 않고 차이만 적었다. (가)나 Params가 1%를 넘게 다르면 굵게 적고, 아래에 설정을 적었다.

| 설정 | 모델 | 논문 MACs | (가) | (가) 반올림 | (가) 차이 | 합계 | 합계 차이 | 논문 Params | Params | Params 반올림 | Params 차이 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| COCO Mask R-CNN 3× | ConvFormer-S18 | 251 | 250.814 | 251 | -0.07% | 251.538 | +0.21% | - | 43.838 | - | - |
| COCO Mask R-CNN 3× | SPANetV2-S18-pure | 255 | 255.631 | 256 | +0.25% | 257.213 | +0.87% | - | 49.331 | - | - |
| COCO Mask R-CNN 3× | **CAFormer-S18** | 254 | 344.766 | 345 | +35.73% | 348.388 | +37.16% | - | 43.405 | - | - |
| COCO Mask R-CNN 3× | **SPANetV2-S18-hybrid** | 251 | 346.185 | 346 | +37.92% | 350.374 | +39.59% | - | 47.160 | - | - |
| COCO Cascade Mask R-CNN 3× | ConvFormer-S18 | 729 | 729.366 | 729 | +0.05% | 730.090 | +0.15% | - | 81.673 | - | - |
| COCO Cascade Mask R-CNN 3× | SPANetV2-S18-pure | 734 | 734.183 | 734 | +0.02% | 735.765 | +0.24% | - | 87.167 | - | - |
| COCO Cascade Mask R-CNN 3× | **CAFormer-S18** | 733 | 823.318 | 823 | +12.32% | 826.940 | +12.82% | - | 81.241 | - | - |
| COCO Cascade Mask R-CNN 3× | **SPANetV2-S18-hybrid** | 729 | 824.737 | 825 | +13.13% | 828.926 | +13.71% | - | 84.996 | - | - |
| COCO Cascade Mask R-CNN 3× | ConvFormer-S36 | 805 | 805.149 | 805 | +0.02% | 806.495 | +0.19% | - | 94.911 | - | - |
| COCO Cascade Mask R-CNN 3× | SPANetV2-S36-pure | 814 | 814.844 | 815 | +0.10% | 817.827 | +0.47% | - | 102.102 | - | - |
| COCO Cascade Mask R-CNN 3× | **CAFormer-S36** | 811 | 990.132 | 990 | +22.09% | 997.187 | +22.96% | - | 94.196 | - | - |
| COCO Cascade Mask R-CNN 3× | **SPANetV2-S36-hybrid** | 806 | 993.756 | 994 | +23.29% | 1001.897 | +24.30% | - | 98.673 | - | - |
| ADE20K UperNet | **ConvFormer-S18** | 925 | 414.914 | 415 | -55.14% | 415.661 | -55.06% | 54 | 36.158 | 36 | -33.04% |
| ADE20K UperNet | **SPANetV2-S18-pure** | 930 | 419.848 | 420 | -54.86% | 421.472 | -54.68% | 60 | 41.651 | 42 | -30.58% |
| ADE20K UperNet | **CAFormer-S18** | 1024 | 513.462 | 513 | -49.86% | 517.249 | -49.49% | 54 | 35.725 | 36 | -33.84% |
| ADE20K UperNet | **SPANetV2-S18-hybrid** | 925 | 514.915 | 515 | -44.33% | 519.283 | -43.86% | 58 | 39.480 | 39 | -31.93% |
| ADE20K UperNet | **ConvFormer-S36** | 1003 | 492.516 | 493 | -50.90% | 493.900 | -50.76% | 67 | 49.396 | 49 | -26.28% |
| ADE20K UperNet | **SPANetV2-S36-pure** | 1012 | 502.444 | 502 | -50.35% | 505.503 | -50.05% | 75 | 56.587 | 57 | -24.55% |
| ADE20K UperNet | **CAFormer-S36** | 1197 | 686.545 | 687 | -42.64% | 693.918 | -42.03% | 67 | 48.681 | 49 | -27.34% |
| ADE20K UperNet | **SPANetV2-S36-hybrid** | 1004 | 690.255 | 690 | -31.25% | 698.741 | -30.40% | 71 | 53.157 | 53 | -25.13% |
| ADE20K Semantic FPN | **SPANet-S** | 46 | 45.693 | 46 | -0.67% | 46.076 | +0.16% | 32 | 32.360 | 32 | +1.12% |
| ADE20K Semantic FPN | **DFFormer-S18** | 41 | 41.345 | 41 | +0.84% | 41.993 | +2.42% | 35 | 31.724 | 32 | -9.36% |
| ADE20K Semantic FPN | **SPANetV2-S18-pure** | 43 | 43.412 | 43 | +0.96% | 43.791 | +1.84% | 33 | 33.431 | 33 | +1.30% |
| ADE20K Semantic FPN | **CDFFormer-S18** | 42 | 41.788 | 42 | -0.50% | 42.179 | +0.43% | 35 | 31.391 | 31 | -10.31% |
| ADE20K Semantic FPN | SPANetV2-S18-hybrid | 48 | 48.455 | 48 | +0.95% | 48.948 | +1.98% | 31 | 31.260 | 31 | +0.84% |
| ADE20K Semantic FPN | SPANet-M | 57 | 57.172 | 57 | +0.30% | 57.740 | +1.30% | 45 | 45.442 | 45 | +0.98% |
| ADE20K Semantic FPN | **DFFormer-S36** | 60 | 59.952 | 60 | -0.08% | 61.173 | +1.96% | 50 | 47.163 | 47 | -5.67% |
| ADE20K Semantic FPN | SPANetV2-S36-pure | 64 | 64.061 | 64 | +0.09% | 64.768 | +1.20% | 48 | 48.366 | 48 | +0.76% |
| ADE20K Semantic FPN | **CDFFormer-S36** | 61 | 60.838 | 61 | -0.27% | 61.571 | +0.94% | 50 | 46.460 | 46 | -7.08% |
| ADE20K Semantic FPN | SPANetV2-S36-hybrid | 74 | 74.171 | 74 | +0.23% | 75.108 | +1.50% | 45 | 44.937 | 45 | -0.14% |
| ImageNet-1K 384×384 (분류) | ConvFormer-S18 | 11.6 | 11.621 | 11.6 | +0.18% | 11.722 | +1.05% | - | 26.774 | - | - |
| ImageNet-1K 384×384 (분류) | SPANetV2-S18-pure | 12.3 | 12.315 | 12.3 | +0.12% | 12.522 | +1.80% | - | 29.163 | - | - |
| ImageNet-1K 384×384 (분류) | CAFormer-S18 | 13.4 | 13.412 | 13.4 | +0.09% | 13.563 | +1.22% | - | 26.342 | - | - |
| ImageNet-1K 384×384 (분류) | SPANetV2-S18-hybrid | 13.6 | 13.616 | 13.6 | +0.12% | 13.839 | +1.75% | - | 26.992 | - | - |
| ImageNet-1K 384×384 (분류) | ConvFormer-S36 | 22.4 | 22.534 | 22.5 | +0.60% | 22.725 | +1.45% | - | 40.012 | - | - |
| ImageNet-1K 384×384 (분류) | SPANetV2-S36-pure | 23.9 | 23.930 | 23.9 | +0.13% | 24.321 | +1.76% | - | 44.099 | - | - |
| ImageNet-1K 384×384 (분류) | CAFormer-S36 | 26.0 | 26.073 | 26.1 | +0.28% | 26.364 | +1.40% | - | 39.297 | - | - |
| ImageNet-1K 384×384 (분류) | SPANetV2-S36-hybrid | 26.6 | 26.595 | 26.6 | -0.02% | 27.021 | +1.58% | - | 40.669 | - | - |

아래는 1%를 넘는 모델의 설정이다(`config_check_dense.py`, 만든 모델과 config에서 읽음). backbone (가)는 같은 입력으로 backbone만 잰 값을 연산자별로 나눈 것이다.

### COCO Mask R-CNN 3×

| 모델 | 입력 | RoI·neck·head 설정 | backbone | backbone Params (M) | backbone (가), 연산자별 G |
|---|---|---|---|---|---|
| CAFormer-S18 | 1x3x1280x800 | RoI {'bbox': 1000, 'mask': 100}; test_cfg rpn {'nms_pre': 1000, 'max_per_img': 1000}, rcnn max_per_img 100<br>neck {'type': 'FPN', 'in_channels': [64, 128, 320, 512], 'out_channels': 256, 'num_outs': 5}<br>rpn {'type': 'RPNHead', 'feat_channels': 256}<br>roi_head {'type': 'StandardRoIHead', 'num_stages': 1, 'bbox_heads': ['Shared2FCBBoxHead(convs=0, fcs=2, fc_out=1024)'], 'mask_heads': ['FCNMaskHead(convs=4, ch=256)']} | MetaFormerFeat depths [3, 3, 9, 3] dims [64, 128, 320, 512]<br>token mixer ['SepConv', 'SepConv', 'Attention', 'Attention'] | 23.238 | matmul 95.2, linear 72.5, conv 6.5, layer_norm 0.4 |
| SPANetV2-S18-hybrid | 1x3x1280x800 | RoI {'bbox': 1000, 'mask': 100}; test_cfg rpn {'nms_pre': 1000, 'max_per_img': 1000}, rcnn max_per_img 100<br>neck {'type': 'FPN', 'in_channels': [64, 128, 320, 512], 'out_channels': 256, 'num_outs': 5}<br>rpn {'type': 'RPNHead', 'feat_channels': 256}<br>roi_head {'type': 'StandardRoIHead', 'num_stages': 1, 'bbox_heads': ['Shared2FCBBoxHead(convs=0, fcs=2, fc_out=1024)'], 'mask_heads': ['FCNMaskHead(convs=4, ch=256)']} | spanetv2_s18_hybrid depths [3, 3, 9, 3] dims [64, 128, 320, 512]<br>token mixer ['ExSPAM', 'ExSPAM', 'MixAttention', 'Attention'] | 26.993 (이 중 forward에 안 쓰임: head 3.104, norm 0.001) | matmul 95.2, linear 71.0, conv 9.3, layer_norm 0.4, group_norm 0.2 |

### COCO Cascade Mask R-CNN 3×

| 모델 | 입력 | RoI·neck·head 설정 | backbone | backbone Params (M) | backbone (가), 연산자별 G |
|---|---|---|---|---|---|
| CAFormer-S18 | 1x3x1280x800 | RoI {'bbox': 1000, 'mask': 100}; test_cfg rpn {'nms_pre': 1000, 'max_per_img': 1000}, rcnn max_per_img 100<br>neck {'type': 'FPN', 'in_channels': [64, 128, 320, 512], 'out_channels': 256, 'num_outs': 5}<br>rpn {'type': 'RPNHead', 'feat_channels': 256}<br>roi_head {'type': 'CascadeRoIHead', 'num_stages': 3, 'bbox_heads': ['ConvFCBBoxHead(convs=4, fcs=1, fc_out=1024)', 'ConvFCBBoxHead(convs=4, fcs=1, fc_out=1024)', 'ConvFCBBoxHead(convs=4, fcs=1, fc_out=1024)'], 'mask_heads': ['FCNMaskHead(convs=4, ch=256)', 'FCNMaskHead(convs=4, ch=256)', 'FCNMaskHead(convs=4, ch=256)']} | MetaFormerFeat depths [3, 3, 9, 3] dims [64, 128, 320, 512]<br>token mixer ['SepConv', 'SepConv', 'Attention', 'Attention'] | 23.238 | matmul 95.2, linear 72.5, conv 6.5, layer_norm 0.4 |
| SPANetV2-S18-hybrid | 1x3x1280x800 | RoI {'bbox': 1000, 'mask': 100}; test_cfg rpn {'nms_pre': 1000, 'max_per_img': 1000}, rcnn max_per_img 100<br>neck {'type': 'FPN', 'in_channels': [64, 128, 320, 512], 'out_channels': 256, 'num_outs': 5}<br>rpn {'type': 'RPNHead', 'feat_channels': 256}<br>roi_head {'type': 'CascadeRoIHead', 'num_stages': 3, 'bbox_heads': ['ConvFCBBoxHead(convs=4, fcs=1, fc_out=1024)', 'ConvFCBBoxHead(convs=4, fcs=1, fc_out=1024)', 'ConvFCBBoxHead(convs=4, fcs=1, fc_out=1024)'], 'mask_heads': ['FCNMaskHead(convs=4, ch=256)', 'FCNMaskHead(convs=4, ch=256)', 'FCNMaskHead(convs=4, ch=256)']} | spanetv2_s18_hybrid depths [3, 3, 9, 3] dims [64, 128, 320, 512]<br>token mixer ['ExSPAM', 'ExSPAM', 'MixAttention', 'Attention'] | 26.993 (이 중 forward에 안 쓰임: head 3.104, norm 0.001) | matmul 95.2, linear 71.0, conv 9.3, layer_norm 0.4, group_norm 0.2 |
| CAFormer-S36 | 1x3x1280x800 | RoI {'bbox': 1000, 'mask': 100}; test_cfg rpn {'nms_pre': 1000, 'max_per_img': 1000}, rcnn max_per_img 100<br>neck {'type': 'FPN', 'in_channels': [64, 128, 320, 512], 'out_channels': 256, 'num_outs': 5}<br>rpn {'type': 'RPNHead', 'feat_channels': 256}<br>roi_head {'type': 'CascadeRoIHead', 'num_stages': 3, 'bbox_heads': ['ConvFCBBoxHead(convs=4, fcs=1, fc_out=1024)', 'ConvFCBBoxHead(convs=4, fcs=1, fc_out=1024)', 'ConvFCBBoxHead(convs=4, fcs=1, fc_out=1024)'], 'mask_heads': ['FCNMaskHead(convs=4, ch=256)', 'FCNMaskHead(convs=4, ch=256)', 'FCNMaskHead(convs=4, ch=256)']} | MetaFormerFeat depths [3, 12, 18, 3] dims [64, 128, 320, 512]<br>token mixer ['SepConv', 'SepConv', 'Attention', 'Attention'] | 36.193 | matmul 187.4, linear 145.1, conv 8.3, layer_norm 0.7 |
| SPANetV2-S36-hybrid | 1x3x1280x800 | RoI {'bbox': 1000, 'mask': 100}; test_cfg rpn {'nms_pre': 1000, 'max_per_img': 1000}, rcnn max_per_img 100<br>neck {'type': 'FPN', 'in_channels': [64, 128, 320, 512], 'out_channels': 256, 'num_outs': 5}<br>rpn {'type': 'RPNHead', 'feat_channels': 256}<br>roi_head {'type': 'CascadeRoIHead', 'num_stages': 3, 'bbox_heads': ['ConvFCBBoxHead(convs=4, fcs=1, fc_out=1024)', 'ConvFCBBoxHead(convs=4, fcs=1, fc_out=1024)', 'ConvFCBBoxHead(convs=4, fcs=1, fc_out=1024)'], 'mask_heads': ['FCNMaskHead(convs=4, ch=256)', 'FCNMaskHead(convs=4, ch=256)', 'FCNMaskHead(convs=4, ch=256)']} | spanetv2_s36_hybrid depths [3, 12, 18, 3] dims [64, 128, 320, 512]<br>token mixer ['ExSPAM', 'ExSPAM', 'MixAttention', 'Attention'] | 40.670 (이 중 forward에 안 쓰임: head 3.104, norm 0.001) | matmul 187.4, linear 141.2, conv 15.5, layer_norm 0.7, group_norm 0.4 |

### ADE20K UperNet

| 모델 | 입력 | RoI·neck·head 설정 | backbone | backbone Params (M) | backbone (가), 연산자별 G |
|---|---|---|---|---|---|
| ConvFormer-S18 | 1x3x512x2048 | decode_head {'type': 'UPerHead', 'in_channels': [64, 128, 320, 512], 'channels': 320, 'pool_scales': [1, 2, 3, 6]}<br>auxiliary_head (config) FCNHead(channels=256), 측정에서 제외 | MetaFormerFeat depths [3, 3, 9, 3] dims [64, 128, 320, 512]<br>token mixer ['SepConv', 'SepConv', 'SepConv', 'SepConv'] | 23.671 | linear 74.3, conv 8.0, layer_norm 0.4 |
| SPANetV2-S18-pure | 1x3x512x2048 | decode_head {'type': 'UPerHead', 'in_channels': [64, 128, 320, 512], 'channels': 320, 'pool_scales': [1, 2, 3, 6]}<br>auxiliary_head (config) FCNHead(channels=256), 측정에서 제외 | spanetv2_s18_pure depths [3, 3, 9, 3] dims [64, 128, 320, 512]<br>token mixer ['ExSPAM', 'ExSPAM', 'ExSPAM', 'ExSPAM'] | 29.164 (이 중 forward에 안 쓰임: head 3.104, norm 0.001) | linear 68.1, conv 18.8, layer_norm 0.4, group_norm 0.3 |
| CAFormer-S18 | 1x3x512x2048 | decode_head {'type': 'UPerHead', 'in_channels': [64, 128, 320, 512], 'channels': 320, 'pool_scales': [1, 2, 3, 6]}<br>auxiliary_head (config) FCNHead(channels=256), 측정에서 제외 | MetaFormerFeat depths [3, 3, 9, 3] dims [64, 128, 320, 512]<br>token mixer ['SepConv', 'SepConv', 'Attention', 'Attention'] | 23.238 | matmul 99.9, linear 74.3, conv 6.7, layer_norm 0.4 |
| SPANetV2-S18-hybrid | 1x3x512x2048 | decode_head {'type': 'UPerHead', 'in_channels': [64, 128, 320, 512], 'channels': 320, 'pool_scales': [1, 2, 3, 6]}<br>auxiliary_head (config) FCNHead(channels=256), 측정에서 제외 | spanetv2_s18_hybrid depths [3, 3, 9, 3] dims [64, 128, 320, 512]<br>token mixer ['ExSPAM', 'ExSPAM', 'MixAttention', 'Attention'] | 26.993 (이 중 forward에 안 쓰임: head 3.104, norm 0.001) | matmul 99.9, linear 72.7, conv 9.6, layer_norm 0.4, group_norm 0.2 |
| ConvFormer-S36 | 1x3x512x2048 | decode_head {'type': 'UPerHead', 'in_channels': [64, 128, 320, 512], 'channels': 320, 'pool_scales': [1, 2, 3, 6]}<br>auxiliary_head (config) FCNHead(channels=256), 측정에서 제외 | MetaFormerFeat depths [3, 12, 18, 3] dims [64, 128, 320, 512]<br>token mixer ['SepConv', 'SepConv', 'SepConv', 'SepConv'] | 36.908 | linear 148.6, conv 11.0, layer_norm 0.7 |
| SPANetV2-S36-pure | 1x3x512x2048 | decode_head {'type': 'UPerHead', 'in_channels': [64, 128, 320, 512], 'channels': 320, 'pool_scales': [1, 2, 3, 6]}<br>auxiliary_head (config) FCNHead(channels=256), 측정에서 제외 | spanetv2_s36_pure depths [3, 12, 18, 3] dims [64, 128, 320, 512]<br>token mixer ['ExSPAM', 'ExSPAM', 'ExSPAM', 'ExSPAM'] | 44.100 (이 중 forward에 안 쓰임: head 3.104, norm 0.001) | linear 136.2, conv 32.7, layer_norm 0.7, group_norm 0.6 |
| CAFormer-S36 | 1x3x512x2048 | decode_head {'type': 'UPerHead', 'in_channels': [64, 128, 320, 512], 'channels': 320, 'pool_scales': [1, 2, 3, 6]}<br>auxiliary_head (config) FCNHead(channels=256), 측정에서 제외 | MetaFormerFeat depths [3, 12, 18, 3] dims [64, 128, 320, 512]<br>token mixer ['SepConv', 'SepConv', 'Attention', 'Attention'] | 36.193 | matmul 196.5, linear 148.6, conv 8.5, layer_norm 0.7 |
| SPANetV2-S36-hybrid | 1x3x512x2048 | decode_head {'type': 'UPerHead', 'in_channels': [64, 128, 320, 512], 'channels': 320, 'pool_scales': [1, 2, 3, 6]}<br>auxiliary_head (config) FCNHead(channels=256), 측정에서 제외 | spanetv2_s36_hybrid depths [3, 12, 18, 3] dims [64, 128, 320, 512]<br>token mixer ['ExSPAM', 'ExSPAM', 'MixAttention', 'Attention'] | 40.670 (이 중 forward에 안 쓰임: head 3.104, norm 0.001) | matmul 196.5, linear 144.6, conv 15.9, layer_norm 0.7, group_norm 0.4 |

UperNet config(semantic_segmentation/local_configs/upernet/SPANetV2/, 커밋 37e7a83 하나에만 있음)의 `decode_head.channels`는 320이다. 확인용으로 이 값만 512로 바꿔 한 번 더 쟀다. 512는 mmsegmentation의 UperNet 기본 config(예: `_base_/models/upernet_swin.py`) 값이다. 다른 설정은 같다. 이 값은 결과 표에 쓰지 않았다.

| 모델 | 논문 값 MACs (Params) | config channels | Params (aux 제외 / 포함) | (가) (논문과 차이) | 합계 | channels=512 Params (aux 제외 / 포함) | channels=512 (가) (논문과 차이) | channels=512 합계 |
|---|---|---|---|---|---|---|---|---|
| ConvFormer-S18 | 925 (54) | 320 | 36.158 / 36.934 | 414.914 (-55.1%) | 415.661 | 53.382 / 54.159 | 925.921 (+0.10%) | 926.685 |
| SPANetV2-S18-pure | 930 (60) | 320 | 41.651 / 42.427 | 419.848 (-54.9%) | 421.472 | 58.875 / 59.652 | 930.855 (+0.09%) | 932.496 |
| CAFormer-S18 | 1024 (54) | 320 | 35.725 / 36.501 | 513.462 (-49.9%) | 517.249 | 52.949 / 53.726 | 1024.469 (+0.05%) | 1028.273 |
| SPANetV2-S18-hybrid | 925 (58) | 320 | 39.480 / 40.257 | 514.915 (-44.3%) | 519.283 | 56.704 / 57.481 | 1025.922 (+10.91%) | 1030.307 |
| ConvFormer-S36 | 1003 (67) | 320 | 49.396 / 50.172 | 492.516 (-50.9%) | 493.900 | 66.620 / 67.396 | 1003.523 (+0.05%) | 1004.924 |
| SPANetV2-S36-pure | 1012 (75) | 320 | 56.587 / 57.363 | 502.444 (-50.4%) | 505.503 | 73.811 / 74.587 | 1013.451 (+0.14%) | 1016.526 |
| CAFormer-S36 | 1197 (67) | 320 | 48.681 / 49.457 | 686.545 (-42.6%) | 693.918 | 65.905 / 66.681 | 1197.552 (+0.05%) | 1204.942 |
| SPANetV2-S36-hybrid | 1004 (71) | 320 | 53.157 / 53.934 | 690.255 (-31.2%) | 698.741 | 70.382 / 71.158 | 1201.262 (+19.65%) | 1209.764 |

- "aux 포함"은 auxiliary head(FCNHead)까지 더한 파라미터 수다. get_flops.py와 결과 표는 aux를 뺀다.

### ADE20K Semantic FPN

| 모델 | 입력 | RoI·neck·head 설정 | backbone | backbone Params (M) | backbone (가), 연산자별 G |
|---|---|---|---|---|---|
| SPANet-S | 1x3x512x512 | decode_head {'type': 'FPNHead', 'in_channels': [256, 256, 256, 256], 'channels': 128, 'pool_scales': None}<br>neck {'type': 'FPN', 'in_channels': [64, 128, 320, 512], 'out_channels': 256, 'num_outs': 4}<br>auxiliary_head (config) None, 측정에서 제외 | SPANet depths [4, 4, 12, 4] dims [64, 128, 320, 512]<br>token mixer ['SPAttention', 'SPAttention', 'SPAttention', 'SPAttention']<br>patch_dims [[128, 128], [64, 64], [32, 32], [16, 16]] | 28.093 | conv 24.1, group_norm 0.1 |
| DFFormer-S18 | 1x3x512x512 | decode_head {'type': 'FPNHead', 'in_channels': [256, 256, 256, 256], 'channels': 128, 'pool_scales': None}<br>neck {'type': 'FPN', 'in_channels': [64, 128, 320, 512], 'out_channels': 256, 'num_outs': 4}<br>auxiliary_head (config) None, 측정에서 제외 | SegDFFormerS18 depths [3, 3, 9, 3] dims [64, 128, 320, 512]<br>token mixer ['DynamicFilter', 'DynamicFilter', 'DynamicFilter', 'DynamicFilter'] | 27.457 | linear 18.6, conv 1.2, einsum 0.0, layer_norm 0.0 |
| SPANetV2-S18-pure | 1x3x512x512 | decode_head {'type': 'FPNHead', 'in_channels': [256, 256, 256, 256], 'channels': 128, 'pool_scales': None}<br>neck {'type': 'FPN', 'in_channels': [64, 128, 320, 512], 'out_channels': 256, 'num_outs': 4}<br>auxiliary_head (config) None, 측정에서 제외 | spanetv2_s18_pure depths [3, 3, 9, 3] dims [64, 128, 320, 512]<br>token mixer ['ExSPAM', 'ExSPAM', 'ExSPAM', 'ExSPAM'] | 29.164 (이 중 forward에 안 쓰임: head 3.104, norm 0.001) | linear 17.0, conv 4.7, layer_norm 0.1, group_norm 0.1 |
| CDFFormer-S18 | 1x3x512x512 | decode_head {'type': 'FPNHead', 'in_channels': [256, 256, 256, 256], 'channels': 128, 'pool_scales': None}<br>neck {'type': 'FPN', 'in_channels': [64, 128, 320, 512], 'out_channels': 256, 'num_outs': 4}<br>auxiliary_head (config) None, 측정에서 제외 | SegCDFFormerS18 depths [3, 3, 9, 3] dims [64, 128, 320, 512]<br>token mixer ['SepConv', 'SepConv', 'DynamicFilter', 'DynamicFilter'] | 27.125 | linear 18.6, conv 1.7, einsum 0.0, layer_norm 0.0 |
| DFFormer-S36 | 1x3x512x512 | decode_head {'type': 'FPNHead', 'in_channels': [256, 256, 256, 256], 'channels': 128, 'pool_scales': None}<br>neck {'type': 'FPN', 'in_channels': [64, 128, 320, 512], 'out_channels': 256, 'num_outs': 4}<br>auxiliary_head (config) None, 측정에서 제외 | SegDFFormerS36 depths [3, 12, 18, 3] dims [64, 128, 320, 512]<br>token mixer ['DynamicFilter', 'DynamicFilter', 'DynamicFilter', 'DynamicFilter'] | 42.896 | linear 37.2, conv 1.2, einsum 0.1, layer_norm 0.0 |
| CDFFormer-S36 | 1x3x512x512 | decode_head {'type': 'FPNHead', 'in_channels': [256, 256, 256, 256], 'channels': 128, 'pool_scales': None}<br>neck {'type': 'FPN', 'in_channels': [64, 128, 320, 512], 'out_channels': 256, 'num_outs': 4}<br>auxiliary_head (config) None, 측정에서 제외 | SegCDFFormerS36 depths [3, 12, 18, 3] dims [64, 128, 320, 512]<br>token mixer ['SepConv', 'SepConv', 'DynamicFilter', 'DynamicFilter'] | 42.193 | linear 37.2, conv 2.1, einsum 0.0, layer_norm 0.0 |

## 6. 새로 만든 config와 감싼 코드

이 서버와 저장소에서 다음 config를 찾지 못했다. 그래서 SPANetV2 config에서 backbone만 바꿔 만들었다. neck, head, test_cfg, 입력 크기는 그대로다.
- SPANetV2 Cascade Mask R-CNN S36(local_configs/SPANetV2에는 S18만 있다)
- ConvFormer·CAFormer의 Mask R-CNN, Cascade Mask R-CNN, UperNet
- SPANet-S/M의 Semantic FPN(SPANet-official에 분할 코드가 없다. README의 Semantic Segmentation 항목도 체크되지 않았고, semantic_segmentation 폴더가 없다)

모델별 config(측정 스크립트는 모두 `measure_dense.py`, 2절은 `repo_get_flops.py`):

| 설정 | 모델 | config (이 폴더 기준 경로) | 출처 | backbone type | 측정 때 덮어쓴 값 |
|---|---|---|---|---|---|
| COCO Mask R-CNN 3× | ConvFormer-S18 | `configs/det/mask-rcnn_convformer_s18_3x.py` | 새로 만듦 | convformer_s18_feat | - |
| COCO Mask R-CNN 3× | SPANetV2-S18-pure | `../../object_detection/local_configs/SPANetV2/mask-rcnn_spanetv2_s18_pure_3x.py` | 저장소 | spanetv2_s18_pure | model.backbone.init_cfg=None (no checkpoint is loaded) |
| COCO Mask R-CNN 3× | CAFormer-S18 | `configs/det/mask-rcnn_caformer_s18_3x.py` | 새로 만듦 | caformer_s18_feat | - |
| COCO Mask R-CNN 3× | SPANetV2-S18-hybrid | `../../object_detection/local_configs/SPANetV2/mask-rcnn_spanetv2_s18_hybrid_3x.py` | 저장소 | spanetv2_s18_hybrid | model.backbone.init_cfg=None (no checkpoint is loaded) |
| COCO Cascade Mask R-CNN 3× | ConvFormer-S18 | `configs/det/cascade-mask-rcnn_convformer_s18_fpn-3x_coco.py` | 새로 만듦 | convformer_s18_feat | - |
| COCO Cascade Mask R-CNN 3× | SPANetV2-S18-pure | `../../object_detection/local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_pure_fpn-3x_coco.py` | 저장소 | spanetv2_s18_pure | model.backbone.init_cfg=None (no checkpoint is loaded) |
| COCO Cascade Mask R-CNN 3× | CAFormer-S18 | `configs/det/cascade-mask-rcnn_caformer_s18_fpn-3x_coco.py` | 새로 만듦 | caformer_s18_feat | - |
| COCO Cascade Mask R-CNN 3× | SPANetV2-S18-hybrid | `../../object_detection/local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_hybrid_fpn-3x_coco.py` | 저장소 | spanetv2_s18_hybrid | model.backbone.init_cfg=None (no checkpoint is loaded) |
| COCO Cascade Mask R-CNN 3× | ConvFormer-S36 | `configs/det/cascade-mask-rcnn_convformer_s36_fpn-3x_coco.py` | 새로 만듦 | convformer_s36_feat | - |
| COCO Cascade Mask R-CNN 3× | SPANetV2-S36-pure | `configs/det/cascade-mask-rcnn_spanetv2_s36_pure_fpn-3x_coco.py` | 새로 만듦 | spanetv2_s36_pure | model.backbone.init_cfg=None (no checkpoint is loaded) |
| COCO Cascade Mask R-CNN 3× | CAFormer-S36 | `configs/det/cascade-mask-rcnn_caformer_s36_fpn-3x_coco.py` | 새로 만듦 | caformer_s36_feat | - |
| COCO Cascade Mask R-CNN 3× | SPANetV2-S36-hybrid | `configs/det/cascade-mask-rcnn_spanetv2_s36_hybrid_fpn-3x_coco.py` | 새로 만듦 | spanetv2_s36_hybrid | model.backbone.init_cfg=None (no checkpoint is loaded) |
| ADE20K UperNet | ConvFormer-S18 | `configs/seg/upernet_convformer_s18_ade20k-160k.py` | 새로 만듦 | convformer_s18_feat | - |
| ADE20K UperNet | SPANetV2-S18-pure | `../../semantic_segmentation/local_configs/upernet/SPANetV2/upernet_spanet_s18_pure_ade20k-160k.py` | 저장소 | spanetv2_s18_pure | model.backbone.init_cfg=None (no checkpoint is loaded) |
| ADE20K UperNet | CAFormer-S18 | `configs/seg/upernet_caformer_s18_ade20k-160k.py` | 새로 만듦 | caformer_s18_feat | - |
| ADE20K UperNet | SPANetV2-S18-hybrid | `../../semantic_segmentation/local_configs/upernet/SPANetV2/upernet_spanet_s18_hybrid_ade20k-160k.py` | 저장소 | spanetv2_s18_hybrid | model.backbone.init_cfg=None (no checkpoint is loaded) |
| ADE20K UperNet | ConvFormer-S36 | `configs/seg/upernet_convformer_s36_ade20k-160k.py` | 새로 만듦 | convformer_s36_feat | - |
| ADE20K UperNet | SPANetV2-S36-pure | `../../semantic_segmentation/local_configs/upernet/SPANetV2/upernet_spanet_s36_pure_ade20k-160k.py` | 저장소 | spanetv2_s36_pure | model.backbone.init_cfg=None (no checkpoint is loaded) |
| ADE20K UperNet | CAFormer-S36 | `configs/seg/upernet_caformer_s36_ade20k-160k.py` | 새로 만듦 | caformer_s36_feat | - |
| ADE20K UperNet | SPANetV2-S36-hybrid | `../../semantic_segmentation/local_configs/upernet/SPANetV2/upernet_spanet_s36_hybrid_ade20k-160k.py` | 저장소 | spanetv2_s36_hybrid | model.backbone.init_cfg=None (no checkpoint is loaded) |
| ADE20K Semantic FPN | SPANet-S | `configs/seg/fpn_spanet_s_ade20k-40k.py` | 새로 만듦 | SPANet | - |
| ADE20K Semantic FPN | DFFormer-S18 | `../third_party/dfformer/segmentation/configs/fpn/dfformer_s18_fpn.py` | 저장소 | SegDFFormerS18 | - |
| ADE20K Semantic FPN | SPANetV2-S18-pure | `../../semantic_segmentation/local_configs/fpn/SPANetV2/fpn_spanet_s18_pure_ade20k-40k.py` | 저장소 | spanetv2_s18_pure | model.backbone.init_cfg=None (no checkpoint is loaded) |
| ADE20K Semantic FPN | CDFFormer-S18 | `../third_party/dfformer/segmentation/configs/fpn/cdfformer_s18_fpn.py` | 저장소 | SegCDFFormerS18 | - |
| ADE20K Semantic FPN | SPANetV2-S18-hybrid | `../../semantic_segmentation/local_configs/fpn/SPANetV2/fpn_spanet_s18_hybrid_ade20k-40k.py` | 저장소 | spanetv2_s18_hybrid | model.backbone.init_cfg=None (no checkpoint is loaded) |
| ADE20K Semantic FPN | SPANet-M | `configs/seg/fpn_spanet_m_ade20k-40k.py` | 새로 만듦 | SPANet | - |
| ADE20K Semantic FPN | DFFormer-S36 | `../third_party/dfformer/segmentation/configs/fpn/dfformer_s36_fpn.py` | 저장소 | SegDFFormerS36 | - |
| ADE20K Semantic FPN | SPANetV2-S36-pure | `../../semantic_segmentation/local_configs/fpn/SPANetV2/fpn_spanet_s36_pure_ade20k-40k.py` | 저장소 | spanetv2_s36_pure | model.backbone.init_cfg=None (no checkpoint is loaded) |
| ADE20K Semantic FPN | CDFFormer-S36 | `../third_party/dfformer/segmentation/configs/fpn/cdfformer_s36_fpn.py` | 저장소 | SegCDFFormerS36 | - |
| ADE20K Semantic FPN | SPANetV2-S36-hybrid | `../../semantic_segmentation/local_configs/fpn/SPANetV2/fpn_spanet_s36_hybrid_ade20k-40k.py` | 저장소 | spanetv2_s36_hybrid | model.backbone.init_cfg=None (no checkpoint is loaded) |
| ImageNet-1K 384×384 (분류) | ConvFormer-S18 | - | ../zoo.py (분류) | - | - |
| ImageNet-1K 384×384 (분류) | SPANetV2-S18-pure | - | ../zoo.py (분류) | - | - |
| ImageNet-1K 384×384 (분류) | CAFormer-S18 | - | ../zoo.py (분류) | - | - |
| ImageNet-1K 384×384 (분류) | SPANetV2-S18-hybrid | - | ../zoo.py (분류) | - | - |
| ImageNet-1K 384×384 (분류) | ConvFormer-S36 | - | ../zoo.py (분류) | - | - |
| ImageNet-1K 384×384 (분류) | SPANetV2-S36-pure | - | ../zoo.py (분류) | - | - |
| ImageNet-1K 384×384 (분류) | CAFormer-S36 | - | ../zoo.py (분류) | - | - |
| ImageNet-1K 384×384 (분류) | SPANetV2-S36-hybrid | - | ../zoo.py (분류) | - | - |

<details><summary>configs/det/cascade-mask-rcnn_caformer_s18_fpn-3x_coco.py</summary>

```python
# Constructed for the re-measurement (not in any repository):
# SPANetV2-S18-hybrid Cascade Mask R-CNN 3x config with only the backbone replaced (neck, heads, test_cfg unchanged).
_base_ = '../../../../object_detection/local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_hybrid_fpn-3x_coco.py'
model = dict(backbone=dict(_delete_=True, type='caformer_s18_feat'))
```

</details>

<details><summary>configs/det/cascade-mask-rcnn_caformer_s36_fpn-3x_coco.py</summary>

```python
# Constructed for the re-measurement (not in any repository):
# SPANetV2-S18-hybrid Cascade Mask R-CNN 3x config with only the backbone replaced (neck, heads, test_cfg unchanged).
_base_ = '../../../../object_detection/local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_hybrid_fpn-3x_coco.py'
model = dict(backbone=dict(_delete_=True, type='caformer_s36_feat'))
```

</details>

<details><summary>configs/det/cascade-mask-rcnn_convformer_s18_fpn-3x_coco.py</summary>

```python
# Constructed for the re-measurement (not in any repository):
# SPANetV2-S18-pure Cascade Mask R-CNN 3x config with only the backbone replaced (neck, heads, test_cfg unchanged).
_base_ = '../../../../object_detection/local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_pure_fpn-3x_coco.py'
model = dict(backbone=dict(_delete_=True, type='convformer_s18_feat'))
```

</details>

<details><summary>configs/det/cascade-mask-rcnn_convformer_s36_fpn-3x_coco.py</summary>

```python
# Constructed for the re-measurement (not in any repository):
# SPANetV2-S18-pure Cascade Mask R-CNN 3x config with only the backbone replaced (neck, heads, test_cfg unchanged).
_base_ = '../../../../object_detection/local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_pure_fpn-3x_coco.py'
model = dict(backbone=dict(_delete_=True, type='convformer_s36_feat'))
```

</details>

<details><summary>configs/det/cascade-mask-rcnn_spanetv2_s36_hybrid_fpn-3x_coco.py</summary>

```python
# Constructed for the re-measurement: object_detection/local_configs/SPANetV2 has only the S18 Cascade configs.
# S18-hybrid Cascade Mask R-CNN 3x config with only the backbone name changed to S36
# (S36 has the same stage channels [64, 128, 320, 512] as S18; neck, heads, test_cfg unchanged).
_base_ = '../../../../object_detection/local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_hybrid_fpn-3x_coco.py'
model = dict(backbone=dict(type='spanetv2_s36_hybrid'), neck=dict(in_channels=[64, 128, 320, 512]))
```

</details>

<details><summary>configs/det/cascade-mask-rcnn_spanetv2_s36_pure_fpn-3x_coco.py</summary>

```python
# Constructed for the re-measurement: object_detection/local_configs/SPANetV2 has only the S18 Cascade configs.
# S18-pure Cascade Mask R-CNN 3x config with only the backbone name changed to S36
# (S36 has the same stage channels [64, 128, 320, 512] as S18; neck, heads, test_cfg unchanged).
_base_ = '../../../../object_detection/local_configs/SPANetV2/cascade-mask-rcnn_spanetv2_s18_pure_fpn-3x_coco.py'
model = dict(backbone=dict(type='spanetv2_s36_pure'), neck=dict(in_channels=[64, 128, 320, 512]))
```

</details>

<details><summary>configs/det/mask-rcnn_caformer_s18_3x.py</summary>

```python
# Constructed for the re-measurement (not in any repository):
# SPANetV2-S18-hybrid Mask R-CNN 3x config with only the backbone replaced (neck, heads, test_cfg unchanged).
_base_ = '../../../../object_detection/local_configs/SPANetV2/mask-rcnn_spanetv2_s18_hybrid_3x.py'
model = dict(backbone=dict(_delete_=True, type='caformer_s18_feat'))
```

</details>

<details><summary>configs/det/mask-rcnn_convformer_s18_3x.py</summary>

```python
# Constructed for the re-measurement (not in any repository):
# SPANetV2-S18-pure Mask R-CNN 3x config with only the backbone replaced (neck, heads, test_cfg unchanged).
_base_ = '../../../../object_detection/local_configs/SPANetV2/mask-rcnn_spanetv2_s18_pure_3x.py'
model = dict(backbone=dict(_delete_=True, type='convformer_s18_feat'))
```

</details>

<details><summary>configs/seg/fpn_spanet_m_ade20k-40k.py</summary>

```python
# Constructed for the re-measurement: SPANet-official has no segmentation code.
# SPANetV2-S36-pure Semantic FPN config with only the backbone replaced by SPANet-M (layers [6, 6, 18, 6]:
# object_detection/local_configs/spanet/S36 and image_classification/spanet.py: spanet_medium); see fpn_spanet_s.
_base_ = '../../../../semantic_segmentation/local_configs/fpn/SPANetV2/fpn_spanet_s36_pure_ade20k-40k.py'
model = dict(backbone=dict(
    _delete_=True, type='SPANet',
    layers=[6, 6, 18, 6], embed_dims=[64, 128, 320, 512],
    patch_dims=[512 // 2**2, 512 // 2**3, 512 // 2**4, 512 // 2**5],
    radius=[2**1, 2**1, 2**0, 2**0], mlp_ratios=[4, 4, 4, 4], downsamples=[True, True, True, True]))
```

</details>

<details><summary>configs/seg/fpn_spanet_s_ade20k-40k.py</summary>

```python
# Constructed for the re-measurement: SPANet-official has no segmentation code.
# SPANetV2-S18-pure Semantic FPN config with only the backbone replaced by SPANet-S, i.e. the SPANet
# detection backbone (SPANet-official/object_detection/mmdet/models/backbones/spanet.py, class SPANet) with
# layers/embed_dims/radius/mlp_ratios/downsamples of SPANet-S (object_detection/local_configs/_base_/models/
# mask_rcnn_spanet_fpn.py and image_classification/spanet.py: spanet_small) and patch_dims for a 512x512 input
# (the img_size // 2**(i+2) rule of spanet.py; the backbone file's own __main__ uses img_size = 512).
_base_ = '../../../../semantic_segmentation/local_configs/fpn/SPANetV2/fpn_spanet_s18_pure_ade20k-40k.py'
model = dict(backbone=dict(
    _delete_=True, type='SPANet',
    layers=[4, 4, 12, 4], embed_dims=[64, 128, 320, 512],
    patch_dims=[512 // 2**2, 512 // 2**3, 512 // 2**4, 512 // 2**5],
    radius=[2**1, 2**1, 2**0, 2**0], mlp_ratios=[4, 4, 4, 4], downsamples=[True, True, True, True]))
```

</details>

<details><summary>configs/seg/upernet_caformer_s18_ade20k-160k.py</summary>

```python
# Constructed for the re-measurement (not in any repository):
# SPANetV2-S18-hybrid UperNet config with only the backbone replaced (decode/auxiliary heads unchanged).
_base_ = '../../../../semantic_segmentation/local_configs/upernet/SPANetV2/upernet_spanet_s18_hybrid_ade20k-160k.py'
model = dict(backbone=dict(_delete_=True, type='caformer_s18_feat'))
```

</details>

<details><summary>configs/seg/upernet_caformer_s36_ade20k-160k.py</summary>

```python
# Constructed for the re-measurement (not in any repository):
# SPANetV2-S36-hybrid UperNet config with only the backbone replaced (decode/auxiliary heads unchanged).
_base_ = '../../../../semantic_segmentation/local_configs/upernet/SPANetV2/upernet_spanet_s36_hybrid_ade20k-160k.py'
model = dict(backbone=dict(_delete_=True, type='caformer_s36_feat'))
```

</details>

<details><summary>configs/seg/upernet_convformer_s18_ade20k-160k.py</summary>

```python
# Constructed for the re-measurement (not in any repository):
# SPANetV2-S18-pure UperNet config with only the backbone replaced (decode/auxiliary heads unchanged).
_base_ = '../../../../semantic_segmentation/local_configs/upernet/SPANetV2/upernet_spanet_s18_pure_ade20k-160k.py'
model = dict(backbone=dict(_delete_=True, type='convformer_s18_feat'))
```

</details>

<details><summary>configs/seg/upernet_convformer_s36_ade20k-160k.py</summary>

```python
# Constructed for the re-measurement (not in any repository):
# SPANetV2-S36-pure UperNet config with only the backbone replaced (decode/auxiliary heads unchanged).
_base_ = '../../../../semantic_segmentation/local_configs/upernet/SPANetV2/upernet_spanet_s36_pure_ade20k-160k.py'
model = dict(backbone=dict(_delete_=True, type='convformer_s36_feat'))
```

</details>

감싼 코드와 등록(`backbones.py`):

```python
"""
Backbones for the dense-prediction re-measurement that are not registered in SPANetV2's
mmdet 3.x / mmseg 1.x copies. No model code is changed; this file only wraps and registers.

  convformer_{s18,s36}_feat, caformer_{s18,s36}_feat
      metaformer_baselines.py (sail-sg/metaformer) has no detection / segmentation backbone.
      `MetaFormerFeat` builds the classifier with its own factory function (random init) and returns the
      features of the four stages (out_indices 0-3), the same way SPANetV2's dense backbone does
      (object_detection/mmdet/models/backbones/spanetv2.py, SPANetV2.forward_features):
          x = downsample_layers[i](x); x = stages[i](x); out_i = norm{i}(x).permute(0, 3, 1, 2)
      with norm{i} = LayerNormWithoutBias(dims[i], eps=1e-6) (SPANetV2's default `norm_layers`,
      taken from metaformer_baselines.py). The classifier's final norm and head are not kept.

  SPANet (SPANet-S / SPANet-M)
      SPANet-official has no segmentation code. Its detection backbone
      (object_detection/mmdet/models/backbones/spanet.py, class `SPANet`, four-stage outputs with a
      GroupNorm per output) is loaded by path. Only its import lines are changed
      (patches/SPANet_det_backbone_mm2_imports.patch) so that it registers into mmseg 1.x.

  SegDFFormerS18/S36, SegCDFFormerS18/S36
      dfformer/segmentation/backbone/{dfformer,cdfformer}.py, imported unchanged
      (`from mmseg.models.builder import BACKBONES` still exists in mmseg 1.2.2).
"""
import contextlib
import importlib.util
import io
import os
import sys

import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
FB = os.path.dirname(HERE)                      # flops_benchmark/
if FB not in sys.path:
    sys.path.insert(0, FB)
import zoo  # noqa: E402

TP = zoo.TP
DFFORMER_ROOT = os.path.join(TP, 'dfformer')
SPANET_DET_BACKBONE = os.path.join(TP, 'SPANet-official', 'object_detection', 'mmdet', 'models', 'backbones', 'spanet.py')

FEAT_ARCHS = ['convformer_s18', 'convformer_s36', 'caformer_s18', 'caformer_s36']


class MetaFormerFeat(nn.Module):
    """metaformer_baselines.<arch>() -> 4-stage feature backbone (see module docstring)."""

    def __init__(self, arch, **unused):
        super().__init__()
        mf = zoo.load('metaformer')
        net = getattr(mf, arch)()          # random init, factory defaults
        self.arch = arch
        self.num_stage = net.num_stage
        self.downsample_layers = net.downsample_layers
        self.stages = net.stages
        dims = [d.conv.out_channels for d in net.downsample_layers]
        for i, d in enumerate(dims):
            setattr(self, f'norm{i}', mf.LayerNormWithoutBias(d, eps=1e-6))
        self.ignored_cfg_keys = sorted(unused)   # e.g. drop_path_rate / head_dropout / init_cfg from the config

    def forward(self, x):
        outs = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(getattr(self, f'norm{i}')(x).permute(0, 3, 1, 2).contiguous())
        return outs


def _feat_factory(arch):
    def build(**kwargs):
        return MetaFormerFeat(arch, **kwargs)
    build.__name__ = f'{arch}_feat'
    return build


def register_metaformer(registry):
    for arch in FEAT_ARCHS:
        name = f'{arch}_feat'
        if name not in registry.module_dict:
            registry.register_module(name=name, module=_feat_factory(arch))


def register_spanet_v1():
    """Load the (import-patched) SPANet detection backbone; it registers `SPANet` into mmseg's MODELS."""
    name = '_dense_spanet_v1_backbone'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SPANET_DET_BACKBONE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # SPANet-official tracks __pycache__/spanet.*.pyc
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def register_dfformer_seg():
    """Import dfformer/segmentation/backbone/{dfformer,cdfformer}.py unchanged (registers SegDFFormer*/SegCDFFormer*)."""
    if DFFORMER_ROOT not in sys.path:
        sys.path.insert(0, DFFORMER_ROOT)
    import segmentation.backbone.dfformer  # noqa: F401
    import segmentation.backbone.cdfformer  # noqa: F401


@contextlib.contextmanager
def captured_stdout():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf
```

## 7. 파일과 재현

| 파일 | 내용 |
|---|---|
| `setup.sh` | third_party 받기, SPANet import 패치 적용, (`envs`) conda env 두 개 만들기 |
| `backbones.py` | ConvFormer·CAFormer 감싸기, SPANet·DFFormer 분할 백본 등록 |
| `configs/` | 새로 만든 config 14개 |
| `measure_dense.py` | 결과 표의 모든 측정 → `results/{det,seg,cls384}.json`, `results/env_*.json`, `results/measure_*.log` |
| `repo_get_flops.py`, `run_step2.sh` | 2절: 저장소 도구 실행 → `results/step2/*.log`, `results/step2/env_*.json` |
| `config_check_dense.py` | 5절 → `results/config_check_{det,seg}.json` |
| `tool_env.py` | env 버전 기록 |
| `report_dense.py` | 이 README |
| `patches/SPANet_det_backbone_mm2_imports.patch` | import 수정 |

```bash
bash setup.sh envs                     # third_party, 패치, conda env
conda activate spanetv2-dense
python measure_dense.py det && python measure_dense.py seg && python measure_dense.py cls384
python config_check_dense.py det && python config_check_dense.py seg
bash run_step2.sh <COCO dir>           # annotations/instances_val2017.json, val2017/ 가 있는 폴더
python report_dense.py
```
