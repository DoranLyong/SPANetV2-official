# 연산량 재측정: SPANet·SPANetV2와 MetaFormer·DFFormer 비교 모델

학위논문 표의 연산량은 기준이 섞여 있다. SPANet·SPANetV2는 각 저장소의 `FLOPs_count.ipynb`(fvcore)로 쟀고, 비교 모델은 원논문 보고값을 옮겼다. 이 폴더에서는 모든 모델을 **같은 환경, 같은 스크립트, 같은 규약**으로 다시 쟀다. 모델 코드와 논문 원고는 고치지 않았고, 측정값만 적는다.

- 입력 `1×3×224×224`, `model.eval()`, CPU, 무작위 초기화(가중치 불필요).
- 단위: Params는 M(10⁶), 연산량은 G(10⁹) MACs. 1 MAC = 곱셈-덧셈 1회.
- 수치는 정수 MACs 원값으로 계산한 뒤 소수 셋째 자리로 표시했다. 그래서 표시된 값끼리 더한 값과 끝자리가 1 다를 수 있다. 원값은 [results/results.json](results/results.json)에 있다.

## 결과 표

| 모델 | Params (M)<br>논문 → 재측정 | MACs (G)<br>논문 → 재측정(합계) | (가) fvcore 기본 | (나) 노트북 처리 함수 | +(a) | +(b) | +(c) | 합계 |
|---|---|---|---|---|---|---|---|---|
| **S18** | | | | | | | | |
| SPANet-S | 29 → 29 (±0) | 4.6 → 4.7 (+0.1) | 4.627 | 4.627 | 4.667 | 4.669 | 4.689 | 4.689 |
| SPANetV2-S18-pure | 29 → 29 (±0) | 4.2 → 4.3 (+0.1) | 4.192 | 4.217 | 4.223 | 4.224 | 4.256 | 4.256 |
| ConvFormer-S18 | 27 → 27 (±0) | 3.9 → 4.0 (+0.1) | 3.956 | - | 3.956 | 3.956 | 3.991 | 3.991 |
| DFFormer-S18 | 30 → 30 (±0) | 3.8 → 3.9 (+0.1) | 3.800 | - | 3.861 | 3.863 | 3.910 | 3.910 |
| SPANetV2-S18-hybrid | 27 → 27 (±0) | 4.2 → 4.2 (±0) | 4.192 | 4.206 | 4.212 | 4.213 | 4.250 | 4.250 |
| CAFormer-S18 | 26 → 26 (±0) | 4.1 → 4.2 (+0.1) | 4.122 | - | 4.122 | 4.122 | 4.160 | 4.160 |
| CDFFormer-S18 | 30 → 30 (±0) | 3.9 → 4.0 (+0.1) | 3.885 | - | 3.905 | 3.906 | 3.953 | 3.953 |
| **S36** | | | | | | | | |
| SPANet-M | 42 → 42 (±0) | 6.8 → 6.9 (+0.1) | 6.824 | 6.824 | 6.885 | 6.888 | 6.918 | 6.918 |
| SPANetV2-S36-pure | 44 → 44 (±0) | 8.1 → 8.3 (+0.2) | 8.145 | 8.193 | 8.202 | 8.204 | 8.265 | 8.265 |
| ConvFormer-S36 | 40 → 40 (±0) | 7.6 → 7.7 (+0.1) | 7.670 | - | 7.670 | 7.670 | 7.735 | 7.735 |
| DFFormer-S36 | 46 → 46 (±0) | 7.4 → 7.6 (+0.2) | 7.364 | - | 7.479 | 7.483 | 7.572 | 7.572 |
| SPANetV2-S36-hybrid | 41 → 41 (±0) | 8.1 → 8.3 (+0.2) | 8.179 | 8.207 | 8.218 | 8.219 | 8.289 | 8.289 |
| CAFormer-S36 | 39 → 39 (±0) | 8.0 → 8.1 (+0.1) | 8.002 | - | 8.002 | 8.002 | 8.073 | 8.073 |
| CDFFormer-S36 | 45 → 45 (±0) | 7.5 → 7.7 (+0.2) | 7.533 | - | 7.571 | 7.572 | 7.662 | 7.662 |
| **M36** | | | | | | | | |
| SPANet-B | 76 → 76 (±0) | 12.0 → 12.1 (+0.1) | 11.983 | 11.983 | 12.069 | 12.073 | 12.115 | 12.115 |
| SPANetV2-M36-pure | 61 → 63 (+2) | 13.7 → 13.9 (+0.2) | 13.725 | 13.795 | 13.805 | 13.807 | 13.890 | 13.890 |
| ConvFormer-M36 | 57 → 57 (±0) | 12.8 → 13.0 (+0.2) | 12.884 | - | 12.884 | 12.884 | 12.973 | 12.973 |
| DFFormer-M36 | 65 → 65 (±0) | 12.5 → 12.8 (+0.3) | 12.463 | - | 12.624 | 12.629 | 12.751 | 12.751 |
| SPANetV2-M36-hybrid | 58 → 58 (±0) | 13.6 → 13.8 (+0.2) | 13.676 | 13.722 | 13.734 | 13.736 | 13.830 | 13.830 |
| CAFormer-M36 | 56 → 56 (±0) | 13.2 → 13.4 (+0.2) | 13.282 | - | 13.282 | 13.282 | 13.380 | 13.380 |
| CDFFormer-M36 | 64 → 64 (±0) | 12.7 → 12.9 (+0.2) | 12.716 | - | 12.761 | 12.763 | 12.886 | 12.886 |
| **B36** | | | | | | | | |
| SPANetV2-B36-pure | 100 → 110 (+10) | 24.3 → 24.5 (+0.2) | 24.251 | 24.345 | 24.359 | 24.362 | 24.472 | 24.472 |
| ConvFormer-B36 | 100 → 100 (±0) | 22.6 → 22.8 (+0.2) | 22.685 | - | 22.685 | 22.685 | 22.804 | 22.804 |
| DFFormer-B36 | 115 → 115 (±0) | 22.1 → 22.5 (+0.4) | 22.128 | - | 22.343 | 22.349 | 22.512 | 22.512 |
| SPANetV2-B36-hybrid | 100 → 102 (+2) | 23.9 → 24.1 (+0.2) | 23.934 | 23.995 | 24.011 | 24.013 | 24.139 | 24.139 |
| CAFormer-B36 | 99 → 99 (±0) | 23.2 → 23.3 (+0.1) | 23.216 | - | 23.216 | 23.216 | 23.346 | 23.346 |
| CDFFormer-B36 | 113 → 113 (±0) | 22.5 → 22.7 (+0.2) | 22.465 | - | 22.525 | 22.527 | 22.691 | 22.691 |

- **논문 → 재측정** 논문 보고 수치 → 이번 재측정 수치, 괄호 안은 차이(재측정 − 논문). MACs의 재측정 수치는 4절 규약의 합계 (가)+(a)+(b)+(c)이다. 논문 표기에 맞춰 재측정 수치를 반올림했다: Params는 소수 첫째 자리에서, MACs는 소수 둘째 자리에서 반올림. 차이도 반올림한 값끼리 계산했다. 나머지 열은 반올림 전 값을 소수 셋째 자리까지 적었다.
- **(가)** fvcore 0.1.5 기본값. 노트북 6번 셀과 같다.
- **(나)** 노트북 7번 셀처럼 `rfft_flop_jit`을 `aten::fft_rfft2`, `aten::fft_irfft2`에 등록한 값. SPANet·SPANetV2만 적었다.
- **+(a), +(b), +(c)** (가)에 (a) FFT, (b) 주파수 영역의 원소별 곱, (c) 그 밖의 원소별 연산을 차례로 더한 누적값. **합계** = (가)+(a)+(b)+(c) 이므로 +(c)와 같다. 규약은 [4절](#4-하나의-규약으로-다시-잰-값)에 있다.
- Params는 `model.parameters()`의 원소 수 전체다. SPANet의 SPF 필터는 `requires_grad=False`인 `nn.Parameter`라서 포함된다 (S 0.055 M, M 0.082 M, B 0.082 M). 다른 모델은 학습 파라미터 수와 같다.
- SPANet-S / M / B는 크기가 가까운 S18 / S36 / M36 묶음의 pure 쪽에 두었다.

항목별 증가분 (MACs, 정수 원값). 마지막 열은 fvcore 무시 목록에 있어 어느 값에도 들어가지 않은 `aten::relu`의 출력 원소 수로, 참고용이다.

| 모델 | (a) FFT | (b) 주파수 영역 곱 | (c) 그 밖의 원소별 | (a)+(b)+(c) | 참고: relu 원소 수 |
|---|---|---|---|---|---|
| **S18** | | | | | |
| SPANet-S | 40,542,208 | 2,057,216 | 20,221,440 | 62,820,864 | 0 |
| SPANetV2-S18-pure | 30,406,656 | 838,656 | 32,180,048 | 63,425,360 | 6,173,696 |
| ConvFormer-S18 | 0 | 0 | 34,427,216 | 34,427,216 | 9,259,520 |
| DFFormer-S18 | 60,813,312 | 1,677,312 | 46,928,552 | 109,419,176 | 9,260,768 |
| SPANetV2-S18-hybrid | 20,471,808 | 473,088 | 36,965,372 | 57,910,268 | 6,173,696 |
| CAFormer-S18 | 0 | 0 | 37,734,140 | 37,734,140 | 7,980,032 |
| CDFFormer-S18 | 19,869,696 | 731,136 | 46,922,936 | 67,523,768 | 9,260,624 |
| **S36** | | | | | |
| SPANet-M | 60,813,312 | 3,085,824 | 30,331,904 | 94,231,040 | 0 |
| SPANetV2-S36-pure | 57,501,696 | 1,645,056 | 60,533,072 | 119,679,824 | 12,044,288 |
| ConvFormer-S36 | 0 | 0 | 64,909,136 | 64,909,136 | 18,065,408 |
| DFFormer-S36 | 115,003,392 | 3,290,112 | 89,261,528 | 207,555,032 | 18,067,664 |
| SPANetV2-S36-hybrid | 38,535,168 | 956,928 | 70,217,285 | 109,709,381 | 12,044,288 |
| CAFormer-S36 | 0 | 0 | 71,744,069 | 71,744,069 | 15,656,960 |
| CDFFormer-S36 | 37,933,056 | 1,376,256 | 89,244,680 | 128,553,992 | 18,067,232 |
| **M36** | | | | | |
| SPANet-B | 85,800,960 | 4,290,048 | 41,772,288 | 131,863,296 | 0 |
| SPANetV2-M36-pure | 80,494,848 | 2,257,920 | 83,160,976 | 165,913,744 | 16,598,016 |
| ConvFormer-M36 | 0 | 0 | 89,200,912 | 89,200,912 | 24,895,872 |
| DFFormer-M36 | 160,989,696 | 4,515,840 | 122,681,776 | 288,187,312 | 24,898,680 |
| SPANetV2-M36-hybrid | 57,802,752 | 1,435,392 | 94,790,545 | 154,028,689 | 16,598,016 |
| CAFormer-M36 | 0 | 0 | 97,419,409 | 97,419,409 | 22,017,024 |
| CDFFormer-M36 | 45,384,192 | 1,645,056 | 122,656,504 | 169,685,752 | 24,898,032 |
| **B36** | | | | | |
| SPANetV2-B36-pure | 107,326,464 | 3,010,560 | 110,871,632 | 221,208,656 | 22,130,688 |
| ConvFormer-B36 | 0 | 0 | 118,924,880 | 118,924,880 | 33,194,496 |
| DFFormer-B36 | 214,652,928 | 6,021,120 | 163,506,056 | 384,180,104 | 33,198,240 |
| SPANetV2-B36-hybrid | 77,070,336 | 1,913,856 | 126,377,717 | 205,361,909 | 22,130,688 |
| CAFormer-B36 | 0 | 0 | 129,882,869 | 129,882,869 | 29,356,032 |
| CDFFormer-B36 | 60,512,256 | 2,193,408 | 163,472,360 | 226,178,024 | 33,197,376 |

## 1. 환경

| 항목 | 값 |
|---|---|
| date | `2026-11-19 15:15:27 KST` |
| python | `3.11.15` |
| torch | `2.8.0+cu128` |
| fvcore | `0.1.5.post20221221` |
| timm | `0.6.13` |
| numpy | `2.4.3` |
| device | `cpu` |
| torch_threads | `16` |
| cpu | `AMD Ryzen Threadripper PRO 5955WX 16-Cores` |
| platform | `Linux-6.17.0-35-generic-x86_64-with-glibc2.39` |
| python_executable | `/home/server/miniforge3/envs/spanetv2/bin/python` |

| 저장소 | 측정 파일 | 측정 시 커밋 | 지정 커밋 | 측정 파일 상태 | remote |
|---|---|---|---|---|---|
| spanet | `image_classification/spanet.py` | `886ad24d4d69495401e93e90d2b0a52c6cb718b7` | `886ad24` (일치) | clean | https://github.com/DoranLyong/SPANet-official.git |
| spanetv2 | `image_classification/models/spanetv2.py` | `1b2cfeecc7d846c0a455c0d2de39922b2684607b` | `1b2cfee` (일치) | clean | https://github.com/DoranLyong/SPANetV2-official.git |
| metaformer | `metaformer_baselines.py` | `64c534ab8f65105fd21713b38e3393a449ca742d` | `64c534a` (일치) | clean | https://github.com/sail-sg/metaformer.git |
| dfformer | `models/dfformer.py` | `2b757c511a1bf1e66a12d494f3b0c42daa5b3636` | `2b757c5` (일치) | clean | https://github.com/okojoalg/dfformer.git |
| gfnet | `gfnet.py` | `2de48b626ea805de82a7f389fed930af1ae0340d` | `-` | M gfnet.py | https://github.com/raoyongming/GFNet.git |

- 모든 모델을 위의 한 파이썬 환경(`spanetv2` conda env)에서 쟀다. 저장소가 지정한 timm 버전은 0.6.11(SPANet, MetaFormer), 0.6.13(SPANetV2), 0.5.4(DFFormer)이지만 0.6.13으로 모두 import 되었다.
- 지정 커밋 네 개는 측정일(2026-11-19)에 각 저장소 기본 브랜치의 HEAD와 같았다. GFNet은 지정 커밋이 없어 HEAD를 썼다.
- SPANetV2는 이 저장소의 `image_classification/models/spanetv2.py` 파일만 경로로 불러왔다. `models/__init__.py`를 거치지 않으므로 `spanetv2_ver4.py`는 import 되지 않는다.
- GFNet의 측정 파일 상태 `M`은 아래 import 수정 한 줄이다.

### import 수정

SPANet, SPANetV2, MetaFormer, DFFormer는 고친 줄이 없다. GFNet `gfnet.py` 7번째 줄만 고쳤다. numpy 2.x에서 `numpy.lib.arraypad`가 비공개 모듈이 되어 import가 실패했고, `pad`라는 이름은 파일 안에서 쓰이지 않는다. 계산 코드는 그대로다 ([patches/GFNet_numpy_import.patch](patches/GFNet_numpy_import.patch)).

```diff
-from numpy.lib.arraypad import pad
+from numpy import pad  # import path fixed for numpy>=2 (was: from numpy.lib.arraypad import pad)
```

## 2. 기존 노트북 재현 (SPANet·SPANetV2)

각 저장소의 `image_classification/FLOPs_count.ipynb`를 그대로 따랐다(처리 함수인 2번 셀과 측정 코드인 6·7번 셀은 두 저장소에서 같고, 출력 문구만 다르다). 7번 셀의 시간 측정 부분은 CUDA가 필요하고 연산량과 무관해 뺐다. `rfft_flop_jit`은 노트북 2번 셀을 그대로 복사했다.

| 모델 | 논문 MACs | (가) | (나) | (가) 0.1 반올림 | (나) 0.1 반올림 | 처리 함수 호출 수 | 논문 값과 맞는 쪽 |
|---|---|---|---|---|---|---|---|
| SPANet-S | 4.6 | 4.627 | 4.627 | 4.6 | 4.6 | 0 | (가), (나) |
| SPANet-M | 6.8 | 6.824 | 6.824 | 6.8 | 6.8 | 0 | (가), (나) |
| SPANet-B | 12.0 | 11.983 | 11.983 | 12.0 | 12.0 | 0 | (가), (나) |
| SPANetV2-S18-pure | 4.2 | 4.192 | 4.217 | 4.2 | 4.2 | 144 | (가), (나) |
| SPANetV2-S18-hybrid | 4.2 | 4.192 | 4.206 | 4.2 | 4.2 | 48 | (가), (나) |
| SPANetV2-S36-pure | 8.1 | 8.145 | 8.193 | 8.1 | 8.2 | 288 | (가) |
| SPANetV2-S36-hybrid | 8.1 | 8.179 | 8.207 | 8.2 | 8.2 | 120 | 둘 다 다름 |
| SPANetV2-M36-pure | 13.7 | 13.725 | 13.795 | 13.7 | 13.8 | 288 | (가) |
| SPANetV2-M36-hybrid | 13.6 | 13.676 | 13.722 | 13.7 | 13.7 | 120 | 둘 다 다름 |
| SPANetV2-B36-pure | 24.3 | 24.251 | 24.345 | 24.3 | 24.3 | 288 | (가), (나) |
| SPANetV2-B36-hybrid | 23.9 | 23.934 | 23.995 | 23.9 | 24.0 | 120 | (가) |

0.1G 단위로 반올림해 비교하면, SPANet 3개는 (가) = (나)라서 둘 다 논문 값과 같다. SPANetV2 8개 중 (가)는 6개, (나)는 3개에서 논문 값과 같고, SPANetV2-S36-hybrid, SPANetV2-M36-hybrid는 둘 다 다르다.

- SPANet은 `torch.fft.fft2` / `ifft2`(`aten::fft_fft2`, `aten::fft_ifft2`)를 쓴다. 처리 함수는 `rfft2` / `irfft2`에만 등록되므로 한 번도 불리지 않고, (나) = (가)이다.
- SPANetV2는 `(B, C, H, W)` 텐서에 `dim=(2, 3)`으로 FFT를 한다. 처리 함수는 `inputs[0]`을 `B, H, W, C`로 풀기 때문에, 예를 들어 S18 stage 0의 `rfft2` 입력 (1, 16, 56, 56)은 H=16, W=56, C=56로 읽혀 501,760이 된다. 같은 호출을 4절 규약으로 세면 602,112이다. `irfft2`에는 반쪽 스펙트럼 (1, 16, 56, 29) (ComplexFloat)이 들어가 259,840이 된다. 모든 모델의 첫 호출은 아래 표에 있다.

| 모델 | 연산 | 모듈 | inputs[0] shape (dtype) | B, H, W, C 로 푼 값 | N = H·W | 노트북 함수 값 | 새 규약 값 (같은 호출) |
|---|---|---|---|---|---|---|---|
| SPANetV2-S18-pure | `aten::fft_rfft2` | `stages.0.0.token_mixer` | (1, 16, 56, 56) (Float) | 1, 16, 56, 56 | 896 | 501,760 | 602,112 |
| SPANetV2-S18-pure | `aten::fft_irfft2` | `stages.0.0.token_mixer` | (1, 16, 56, 29) (ComplexFloat) | 1, 16, 56, 29 | 896 | 259,840 | 602,112 |
| SPANetV2-S18-hybrid | `aten::fft_rfft2` | `stages.0.0.token_mixer` | (1, 16, 56, 56) (Float) | 1, 16, 56, 56 | 896 | 501,760 | 602,112 |
| SPANetV2-S18-hybrid | `aten::fft_irfft2` | `stages.0.0.token_mixer` | (1, 16, 56, 29) (ComplexFloat) | 1, 16, 56, 29 | 896 | 259,840 | 602,112 |
| SPANetV2-S36-pure | `aten::fft_rfft2` | `stages.0.0.token_mixer` | (1, 16, 56, 56) (Float) | 1, 16, 56, 56 | 896 | 501,760 | 602,112 |
| SPANetV2-S36-pure | `aten::fft_irfft2` | `stages.0.0.token_mixer` | (1, 16, 56, 29) (ComplexFloat) | 1, 16, 56, 29 | 896 | 259,840 | 602,112 |
| SPANetV2-S36-hybrid | `aten::fft_rfft2` | `stages.0.0.token_mixer` | (1, 16, 56, 56) (Float) | 1, 16, 56, 56 | 896 | 501,760 | 602,112 |
| SPANetV2-S36-hybrid | `aten::fft_irfft2` | `stages.0.0.token_mixer` | (1, 16, 56, 29) (ComplexFloat) | 1, 16, 56, 29 | 896 | 259,840 | 602,112 |
| SPANetV2-M36-pure | `aten::fft_rfft2` | `stages.0.0.token_mixer` | (1, 24, 56, 56) (Float) | 1, 24, 56, 56 | 1,344 | 827,904 | 903,168 |
| SPANetV2-M36-pure | `aten::fft_irfft2` | `stages.0.0.token_mixer` | (1, 24, 56, 29) (ComplexFloat) | 1, 24, 56, 29 | 1,344 | 428,736 | 903,168 |
| SPANetV2-M36-hybrid | `aten::fft_rfft2` | `stages.0.0.token_mixer` | (1, 24, 56, 56) (Float) | 1, 24, 56, 56 | 1,344 | 827,904 | 903,168 |
| SPANetV2-M36-hybrid | `aten::fft_irfft2` | `stages.0.0.token_mixer` | (1, 24, 56, 29) (ComplexFloat) | 1, 24, 56, 29 | 1,344 | 428,736 | 903,168 |
| SPANetV2-B36-pure | `aten::fft_rfft2` | `stages.0.0.token_mixer` | (1, 32, 56, 56) (Float) | 1, 32, 56, 56 | 1,792 | 1,103,872 | 1,204,224 |
| SPANetV2-B36-pure | `aten::fft_irfft2` | `stages.0.0.token_mixer` | (1, 32, 56, 29) (ComplexFloat) | 1, 32, 56, 29 | 1,792 | 571,648 | 1,204,224 |
| SPANetV2-B36-hybrid | `aten::fft_rfft2` | `stages.0.0.token_mixer` | (1, 32, 56, 56) (Float) | 1, 32, 56, 56 | 1,792 | 1,103,872 | 1,204,224 |
| SPANetV2-B36-hybrid | `aten::fft_irfft2` | `stages.0.0.token_mixer` | (1, 32, 56, 29) (ComplexFloat) | 1, 32, 56, 29 | 1,792 | 571,648 | 1,204,224 |

## 3. fvcore 기본값에서 빠진 연산

모델마다 fvcore 기본값의 `unsupported_ops()`를 그대로 옮겼다. 모듈 이름은 그 연산을 직접 부르는 가장 안쪽 모듈이며, 블록 번호 같은 숫자는 `*`로 묶고 클래스 이름과 호출 횟수를 붙였다. `unsupported_ops(모듈)`에서 자식 모듈 몫을 빼서 구했다.

전체 모델에서 나온 연산: `aten::add`, `aten::clone`, `aten::div`, `aten::fft_fft2`, `aten::fft_fftshift`, `aten::fft_ifft2`, `aten::fft_ifftshift`, `aten::fft_irfft2`, `aten::fft_rfft2`, `aten::gelu`, `aten::mean`, `aten::mul`, `aten::pow`, `aten::real`, `aten::sigmoid`, `aten::softmax`, `aten::sqrt`, `aten::square`, `aten::sub`, `aten::view_as_complex`.

fvcore가 따로 무시하는 연산(`aten::relu`, `aten::to`, `aten::copy_`, `aten::permute` 등 `_IGNORED_OPS`)은 `unsupported_ops()`에 나오지 않는다. fvcore 기본값은 정규화 층(`layer_norm`, `group_norm`)은 센다(입력 원소 수 × 5, affine이 없으면 × 4).

<details><summary><b>SPANet-S</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 128, 'aten::add': 96, 'aten::fft_fft2': 72, 'aten::fft_fftshift': 72, 'aten::fft_ifftshift': 72, 'aten::fft_ifft2': 72, 'aten::real': 72, 'aten::clone': 24, 'aten::gelu': 24, 'aten::mean': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 128 | `network.*.*.res_scale1` [Scale] 16; `network.*.*.res_scale2` [Scale] 16; `network.*.*.token_mixer` [SPAM] 24; `network.*.*.token_mixer.sps.*` [SPF] 72 |
| `aten::add` | 96 | `network.*.*` [SPANetBlock] 48; `network.*.*.token_mixer` [SPAM] 48 |
| `aten::fft_fft2` | 72 | `network.*.*.token_mixer.sps.*` [SPF] 72 |
| `aten::fft_fftshift` | 72 | `network.*.*.token_mixer.sps.*` [SPF] 72 |
| `aten::fft_ifft2` | 72 | `network.*.*.token_mixer.sps.*` [SPF] 72 |
| `aten::fft_ifftshift` | 72 | `network.*.*.token_mixer.sps.*` [SPF] 72 |
| `aten::real` | 72 | `network.*.*.token_mixer.sps.*` [SPF] 72 |
| `aten::clone` | 24 | `network.*.*.token_mixer` [SPAM] 24 |
| `aten::gelu` | 24 | `network.*.*.mlp.act` [GELU] 24 |
| `aten::mean` | 1 | `(root)` [SPANet] 1 |

</details>

<details><summary><b>SPANetV2-S18-pure</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 136, 'aten::sigmoid': 72, 'aten::fft_rfft2': 72, 'aten::fft_irfft2': 72, 'aten::add': 58, 'aten::gelu': 36, 'aten::pow': 22, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 136 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.res_scale1` [Scale] 12; `stages.*.*.res_scale2` [Scale] 12; `stages.*.*.token_mixer` [ExSPAM] 90 |
| `aten::fft_irfft2` | 72 | `stages.*.*.token_mixer` [ExSPAM] 72 |
| `aten::fft_rfft2` | 72 | `stages.*.*.token_mixer` [ExSPAM] 72 |
| `aten::sigmoid` | 72 | `stages.*.*.token_mixer` [ExSPAM] 72 |
| `aten::add` | 58 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 36; `stages.*.*.mlp.act` [StarReLU] 18 |
| `aten::gelu` | 36 | `stages.*.*.token_mixer.proj.*` [GELU] 18; `stages.*.*.token_mixer.query.*` [GELU] 18 |
| `aten::pow` | 22 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 18 |
| `aten::mean` | 9 | `(root)` [SPANetV2] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>ConvFormer-S18</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::add': 76, 'aten::mul': 64, 'aten::pow': 40, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::add` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 36; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.token_mixer.act1` [StarReLU] 18 |
| `aten::mul` | 64 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.res_scale1` [Scale] 12; `stages.*.*.res_scale2` [Scale] 12; `stages.*.*.token_mixer.act1` [StarReLU] 18 |
| `aten::pow` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.token_mixer.act1` [StarReLU] 18 |
| `aten::mean` | 9 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>DFFormer-S18</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 136, 'aten::add': 130, 'aten::mean': 99, 'aten::pow': 94, 'aten::sub': 40, 'aten::sqrt': 40, 'aten::div': 40, 'aten::softmax': 18, 'aten::fft_rfft2': 18, 'aten::view_as_complex': 18, 'aten::fft_irfft2': 18, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 136 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.norm1` [LayerNormGeneral] 18; `stages.*.*.norm2` [LayerNormGeneral] 18; `stages.*.*.res_scale1` [Scale] 12; `stages.*.*.res_scale2` [Scale] 12; `stages.*.*.token_mixer` [DynamicFilter] 18; `stages.*.*.token_mixer.act1` [StarReLU] 18; `stages.*.*.token_mixer.reweight.act` [StarReLU] 18 |
| `aten::add` | 130 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 36; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.norm1` [LayerNormGeneral] 18; `stages.*.*.norm2` [LayerNormGeneral] 18; `stages.*.*.token_mixer.act1` [StarReLU] 18; `stages.*.*.token_mixer.reweight.act` [StarReLU] 18 |
| `aten::mean` | 99 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer` [DynamicFilter] 18 |
| `aten::pow` | 94 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.norm1` [LayerNormGeneral] 18; `stages.*.*.norm2` [LayerNormGeneral] 18; `stages.*.*.token_mixer.act1` [StarReLU] 18; `stages.*.*.token_mixer.reweight.act` [StarReLU] 18 |
| `aten::div` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 18; `stages.*.*.norm2` [LayerNormGeneral] 18 |
| `aten::sqrt` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 18; `stages.*.*.norm2` [LayerNormGeneral] 18 |
| `aten::sub` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 18; `stages.*.*.norm2` [LayerNormGeneral] 18 |
| `aten::fft_irfft2` | 18 | `stages.*.*.token_mixer` [DynamicFilter] 18 |
| `aten::fft_rfft2` | 18 | `stages.*.*.token_mixer` [DynamicFilter] 18 |
| `aten::softmax` | 18 | `stages.*.*.token_mixer` [DynamicFilter] 18 |
| `aten::view_as_complex` | 18 | `stages.*.*.token_mixer` [DynamicFilter] 18 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>SPANetV2-S18-hybrid</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 100, 'aten::add': 67, 'aten::sigmoid': 24, 'aten::fft_rfft2': 24, 'aten::fft_irfft2': 24, 'aten::pow': 22, 'aten::gelu': 12, 'aten::softmax': 12, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 100 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.res_scale1` [Scale] 12; `stages.*.*.res_scale2` [Scale] 12; `stages.*.*.token_mixer` [ExSPAM] 30; `stages.*.*.token_mixer` [MixAttention] 18; `stages.*.*.token_mixer` [Attention] 6 |
| `aten::add` | 67 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 36; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.token_mixer` [MixAttention] 9 |
| `aten::fft_irfft2` | 24 | `stages.*.*.token_mixer` [ExSPAM] 24 |
| `aten::fft_rfft2` | 24 | `stages.*.*.token_mixer` [ExSPAM] 24 |
| `aten::sigmoid` | 24 | `stages.*.*.token_mixer` [ExSPAM] 24 |
| `aten::pow` | 22 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 18 |
| `aten::gelu` | 12 | `stages.*.*.token_mixer.proj.*` [GELU] 6; `stages.*.*.token_mixer.query.*` [GELU] 6 |
| `aten::softmax` | 12 | `stages.*.*.token_mixer` [MixAttention] 9; `stages.*.*.token_mixer` [Attention] 3 |
| `aten::mean` | 9 | `(root)` [SPANetV2] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>CAFormer-S18</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 76, 'aten::add': 64, 'aten::pow': 28, 'aten::softmax': 12, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.res_scale1` [Scale] 12; `stages.*.*.res_scale2` [Scale] 12; `stages.*.*.token_mixer` [Attention] 24; `stages.*.*.token_mixer.act1` [StarReLU] 6 |
| `aten::add` | 64 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 36; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.token_mixer.act1` [StarReLU] 6 |
| `aten::pow` | 28 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.token_mixer.act1` [StarReLU] 6 |
| `aten::softmax` | 12 | `stages.*.*.token_mixer` [Attention] 12 |
| `aten::mean` | 9 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>CDFFormer-S18</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::add': 124, 'aten::mul': 124, 'aten::mean': 93, 'aten::pow': 88, 'aten::sub': 40, 'aten::sqrt': 40, 'aten::div': 40, 'aten::softmax': 12, 'aten::fft_rfft2': 12, 'aten::view_as_complex': 12, 'aten::fft_irfft2': 12, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::add` | 124 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 36; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.norm1` [LayerNormGeneral] 18; `stages.*.*.norm2` [LayerNormGeneral] 18; `stages.*.*.token_mixer.act1` [StarReLU] 18; `stages.*.*.token_mixer.reweight.act` [StarReLU] 12 |
| `aten::mul` | 124 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.norm1` [LayerNormGeneral] 18; `stages.*.*.norm2` [LayerNormGeneral] 18; `stages.*.*.res_scale1` [Scale] 12; `stages.*.*.res_scale2` [Scale] 12; `stages.*.*.token_mixer` [DynamicFilter] 12; `stages.*.*.token_mixer.act1` [StarReLU] 18; `stages.*.*.token_mixer.reweight.act` [StarReLU] 12 |
| `aten::mean` | 93 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer` [DynamicFilter] 12 |
| `aten::pow` | 88 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 18; `stages.*.*.norm1` [LayerNormGeneral] 18; `stages.*.*.norm2` [LayerNormGeneral] 18; `stages.*.*.token_mixer.act1` [StarReLU] 18; `stages.*.*.token_mixer.reweight.act` [StarReLU] 12 |
| `aten::div` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 18; `stages.*.*.norm2` [LayerNormGeneral] 18 |
| `aten::sqrt` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 18; `stages.*.*.norm2` [LayerNormGeneral] 18 |
| `aten::sub` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 18; `stages.*.*.norm2` [LayerNormGeneral] 18 |
| `aten::fft_irfft2` | 12 | `stages.*.*.token_mixer` [DynamicFilter] 12 |
| `aten::fft_rfft2` | 12 | `stages.*.*.token_mixer` [DynamicFilter] 12 |
| `aten::softmax` | 12 | `stages.*.*.token_mixer` [DynamicFilter] 12 |
| `aten::view_as_complex` | 12 | `stages.*.*.token_mixer` [DynamicFilter] 12 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>SPANet-M</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 192, 'aten::add': 144, 'aten::fft_fft2': 108, 'aten::fft_fftshift': 108, 'aten::fft_ifftshift': 108, 'aten::fft_ifft2': 108, 'aten::real': 108, 'aten::clone': 36, 'aten::gelu': 36, 'aten::mean': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 192 | `network.*.*.res_scale1` [Scale] 24; `network.*.*.res_scale2` [Scale] 24; `network.*.*.token_mixer` [SPAM] 36; `network.*.*.token_mixer.sps.*` [SPF] 108 |
| `aten::add` | 144 | `network.*.*` [SPANetBlock] 72; `network.*.*.token_mixer` [SPAM] 72 |
| `aten::fft_fft2` | 108 | `network.*.*.token_mixer.sps.*` [SPF] 108 |
| `aten::fft_fftshift` | 108 | `network.*.*.token_mixer.sps.*` [SPF] 108 |
| `aten::fft_ifft2` | 108 | `network.*.*.token_mixer.sps.*` [SPF] 108 |
| `aten::fft_ifftshift` | 108 | `network.*.*.token_mixer.sps.*` [SPF] 108 |
| `aten::real` | 108 | `network.*.*.token_mixer.sps.*` [SPF] 108 |
| `aten::clone` | 36 | `network.*.*.token_mixer` [SPAM] 36 |
| `aten::gelu` | 36 | `network.*.*.mlp.act` [GELU] 36 |
| `aten::mean` | 1 | `(root)` [SPANet] 1 |

</details>

<details><summary><b>SPANetV2-S36-pure</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 262, 'aten::sigmoid': 144, 'aten::fft_rfft2': 144, 'aten::fft_irfft2': 144, 'aten::add': 112, 'aten::gelu': 72, 'aten::pow': 40, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 262 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [ExSPAM] 180 |
| `aten::fft_irfft2` | 144 | `stages.*.*.token_mixer` [ExSPAM] 144 |
| `aten::fft_rfft2` | 144 | `stages.*.*.token_mixer` [ExSPAM] 144 |
| `aten::sigmoid` | 144 | `stages.*.*.token_mixer` [ExSPAM] 144 |
| `aten::add` | 112 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36 |
| `aten::gelu` | 72 | `stages.*.*.token_mixer.proj.*` [GELU] 36; `stages.*.*.token_mixer.query.*` [GELU] 36 |
| `aten::pow` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36 |
| `aten::mean` | 9 | `(root)` [SPANetV2] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>ConvFormer-S36</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::add': 148, 'aten::mul': 118, 'aten::pow': 76, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::add` | 148 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36 |
| `aten::mul` | 118 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer.act1` [StarReLU] 36 |
| `aten::pow` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36 |
| `aten::mean` | 9 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>DFFormer-S36</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 262, 'aten::add': 256, 'aten::mean': 189, 'aten::pow': 184, 'aten::sub': 76, 'aten::sqrt': 76, 'aten::div': 76, 'aten::softmax': 36, 'aten::fft_rfft2': 36, 'aten::view_as_complex': 36, 'aten::fft_irfft2': 36, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 262 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [DynamicFilter] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 36 |
| `aten::add` | 256 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 36 |
| `aten::mean` | 189 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6; `stages.*.*.norm1` [LayerNormGeneral] 72; `stages.*.*.norm2` [LayerNormGeneral] 72; `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::pow` | 184 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 36 |
| `aten::div` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::sqrt` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::sub` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::fft_irfft2` | 36 | `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::fft_rfft2` | 36 | `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::softmax` | 36 | `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::view_as_complex` | 36 | `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>SPANetV2-S36-hybrid</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 199, 'aten::add': 130, 'aten::sigmoid': 60, 'aten::fft_rfft2': 60, 'aten::fft_irfft2': 60, 'aten::pow': 40, 'aten::gelu': 30, 'aten::softmax': 21, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 199 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [ExSPAM] 75; `stages.*.*.token_mixer` [MixAttention] 36; `stages.*.*.token_mixer` [Attention] 6 |
| `aten::add` | 130 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer` [MixAttention] 18 |
| `aten::fft_irfft2` | 60 | `stages.*.*.token_mixer` [ExSPAM] 60 |
| `aten::fft_rfft2` | 60 | `stages.*.*.token_mixer` [ExSPAM] 60 |
| `aten::sigmoid` | 60 | `stages.*.*.token_mixer` [ExSPAM] 60 |
| `aten::pow` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36 |
| `aten::gelu` | 30 | `stages.*.*.token_mixer.proj.*` [GELU] 15; `stages.*.*.token_mixer.query.*` [GELU] 15 |
| `aten::softmax` | 21 | `stages.*.*.token_mixer` [MixAttention] 18; `stages.*.*.token_mixer` [Attention] 3 |
| `aten::mean` | 9 | `(root)` [SPANetV2] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>CAFormer-S36</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 139, 'aten::add': 127, 'aten::pow': 55, 'aten::softmax': 21, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 139 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [Attention] 42; `stages.*.*.token_mixer.act1` [StarReLU] 15 |
| `aten::add` | 127 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer.act1` [StarReLU] 15 |
| `aten::pow` | 55 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer.act1` [StarReLU] 15 |
| `aten::softmax` | 21 | `stages.*.*.token_mixer` [Attention] 21 |
| `aten::mean` | 9 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>CDFFormer-S36</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::add': 241, 'aten::mul': 232, 'aten::mean': 174, 'aten::pow': 169, 'aten::sub': 76, 'aten::sqrt': 76, 'aten::div': 76, 'aten::softmax': 21, 'aten::fft_rfft2': 21, 'aten::view_as_complex': 21, 'aten::fft_irfft2': 21, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::add` | 241 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 21 |
| `aten::mul` | 232 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [DynamicFilter] 21; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 21 |
| `aten::mean` | 174 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6; `stages.*.*.norm1` [LayerNormGeneral] 72; `stages.*.*.norm2` [LayerNormGeneral] 72; `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::pow` | 169 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 21 |
| `aten::div` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::sqrt` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::sub` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::fft_irfft2` | 21 | `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::fft_rfft2` | 21 | `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::softmax` | 21 | `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::view_as_complex` | 21 | `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>SPANet-B</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 192, 'aten::add': 144, 'aten::fft_fft2': 108, 'aten::fft_fftshift': 108, 'aten::fft_ifftshift': 108, 'aten::fft_ifft2': 108, 'aten::real': 108, 'aten::clone': 36, 'aten::gelu': 36, 'aten::mean': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 192 | `network.*.*.res_scale1` [Scale] 24; `network.*.*.res_scale2` [Scale] 24; `network.*.*.token_mixer` [SPAM] 36; `network.*.*.token_mixer.sps.*` [SPF] 108 |
| `aten::add` | 144 | `network.*.*` [SPANetBlock] 72; `network.*.*.token_mixer` [SPAM] 72 |
| `aten::fft_fft2` | 108 | `network.*.*.token_mixer.sps.*` [SPF] 108 |
| `aten::fft_fftshift` | 108 | `network.*.*.token_mixer.sps.*` [SPF] 108 |
| `aten::fft_ifft2` | 108 | `network.*.*.token_mixer.sps.*` [SPF] 108 |
| `aten::fft_ifftshift` | 108 | `network.*.*.token_mixer.sps.*` [SPF] 108 |
| `aten::real` | 108 | `network.*.*.token_mixer.sps.*` [SPF] 108 |
| `aten::clone` | 36 | `network.*.*.token_mixer` [SPAM] 36 |
| `aten::gelu` | 36 | `network.*.*.mlp.act` [GELU] 36 |
| `aten::mean` | 1 | `(root)` [SPANet] 1 |

</details>

<details><summary><b>SPANetV2-M36-pure</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 262, 'aten::sigmoid': 144, 'aten::fft_rfft2': 144, 'aten::fft_irfft2': 144, 'aten::add': 112, 'aten::gelu': 72, 'aten::pow': 40, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 262 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [ExSPAM] 180 |
| `aten::fft_irfft2` | 144 | `stages.*.*.token_mixer` [ExSPAM] 144 |
| `aten::fft_rfft2` | 144 | `stages.*.*.token_mixer` [ExSPAM] 144 |
| `aten::sigmoid` | 144 | `stages.*.*.token_mixer` [ExSPAM] 144 |
| `aten::add` | 112 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36 |
| `aten::gelu` | 72 | `stages.*.*.token_mixer.proj.*` [GELU] 36; `stages.*.*.token_mixer.query.*` [GELU] 36 |
| `aten::pow` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36 |
| `aten::mean` | 9 | `(root)` [SPANetV2] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>ConvFormer-M36</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::add': 148, 'aten::mul': 118, 'aten::pow': 76, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::add` | 148 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36 |
| `aten::mul` | 118 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer.act1` [StarReLU] 36 |
| `aten::pow` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36 |
| `aten::mean` | 9 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>DFFormer-M36</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 262, 'aten::add': 256, 'aten::mean': 189, 'aten::pow': 184, 'aten::sub': 76, 'aten::sqrt': 76, 'aten::div': 76, 'aten::softmax': 36, 'aten::fft_rfft2': 36, 'aten::view_as_complex': 36, 'aten::fft_irfft2': 36, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 262 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [DynamicFilter] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 36 |
| `aten::add` | 256 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 36 |
| `aten::mean` | 189 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6; `stages.*.*.norm1` [LayerNormGeneral] 72; `stages.*.*.norm2` [LayerNormGeneral] 72; `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::pow` | 184 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 36 |
| `aten::div` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::sqrt` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::sub` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::fft_irfft2` | 36 | `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::fft_rfft2` | 36 | `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::softmax` | 36 | `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::view_as_complex` | 36 | `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>SPANetV2-M36-hybrid</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 199, 'aten::add': 130, 'aten::sigmoid': 60, 'aten::fft_rfft2': 60, 'aten::fft_irfft2': 60, 'aten::pow': 40, 'aten::gelu': 30, 'aten::softmax': 21, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 199 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [ExSPAM] 75; `stages.*.*.token_mixer` [MixAttention] 36; `stages.*.*.token_mixer` [Attention] 6 |
| `aten::add` | 130 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer` [MixAttention] 18 |
| `aten::fft_irfft2` | 60 | `stages.*.*.token_mixer` [ExSPAM] 60 |
| `aten::fft_rfft2` | 60 | `stages.*.*.token_mixer` [ExSPAM] 60 |
| `aten::sigmoid` | 60 | `stages.*.*.token_mixer` [ExSPAM] 60 |
| `aten::pow` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36 |
| `aten::gelu` | 30 | `stages.*.*.token_mixer.proj.*` [GELU] 15; `stages.*.*.token_mixer.query.*` [GELU] 15 |
| `aten::softmax` | 21 | `stages.*.*.token_mixer` [MixAttention] 18; `stages.*.*.token_mixer` [Attention] 3 |
| `aten::mean` | 9 | `(root)` [SPANetV2] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>CAFormer-M36</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 139, 'aten::add': 127, 'aten::pow': 55, 'aten::softmax': 21, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 139 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [Attention] 42; `stages.*.*.token_mixer.act1` [StarReLU] 15 |
| `aten::add` | 127 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer.act1` [StarReLU] 15 |
| `aten::pow` | 55 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer.act1` [StarReLU] 15 |
| `aten::softmax` | 21 | `stages.*.*.token_mixer` [Attention] 21 |
| `aten::mean` | 9 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>CDFFormer-M36</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::add': 241, 'aten::mul': 232, 'aten::mean': 174, 'aten::pow': 169, 'aten::sub': 76, 'aten::sqrt': 76, 'aten::div': 76, 'aten::softmax': 21, 'aten::fft_rfft2': 21, 'aten::view_as_complex': 21, 'aten::fft_irfft2': 21, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::add` | 241 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 21 |
| `aten::mul` | 232 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [DynamicFilter] 21; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 21 |
| `aten::mean` | 174 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6; `stages.*.*.norm1` [LayerNormGeneral] 72; `stages.*.*.norm2` [LayerNormGeneral] 72; `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::pow` | 169 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 21 |
| `aten::div` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::sqrt` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::sub` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::fft_irfft2` | 21 | `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::fft_rfft2` | 21 | `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::softmax` | 21 | `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::view_as_complex` | 21 | `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>SPANetV2-B36-pure</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 262, 'aten::sigmoid': 144, 'aten::fft_rfft2': 144, 'aten::fft_irfft2': 144, 'aten::add': 112, 'aten::gelu': 72, 'aten::pow': 40, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 262 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [ExSPAM] 180 |
| `aten::fft_irfft2` | 144 | `stages.*.*.token_mixer` [ExSPAM] 144 |
| `aten::fft_rfft2` | 144 | `stages.*.*.token_mixer` [ExSPAM] 144 |
| `aten::sigmoid` | 144 | `stages.*.*.token_mixer` [ExSPAM] 144 |
| `aten::add` | 112 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36 |
| `aten::gelu` | 72 | `stages.*.*.token_mixer.proj.*` [GELU] 36; `stages.*.*.token_mixer.query.*` [GELU] 36 |
| `aten::pow` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36 |
| `aten::mean` | 9 | `(root)` [SPANetV2] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>ConvFormer-B36</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::add': 148, 'aten::mul': 118, 'aten::pow': 76, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::add` | 148 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36 |
| `aten::mul` | 118 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer.act1` [StarReLU] 36 |
| `aten::pow` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36 |
| `aten::mean` | 9 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>DFFormer-B36</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 262, 'aten::add': 256, 'aten::mean': 189, 'aten::pow': 184, 'aten::sub': 76, 'aten::sqrt': 76, 'aten::div': 76, 'aten::softmax': 36, 'aten::fft_rfft2': 36, 'aten::view_as_complex': 36, 'aten::fft_irfft2': 36, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 262 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [DynamicFilter] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 36 |
| `aten::add` | 256 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 36 |
| `aten::mean` | 189 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6; `stages.*.*.norm1` [LayerNormGeneral] 72; `stages.*.*.norm2` [LayerNormGeneral] 72; `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::pow` | 184 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 36 |
| `aten::div` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::sqrt` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::sub` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::fft_irfft2` | 36 | `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::fft_rfft2` | 36 | `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::softmax` | 36 | `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::view_as_complex` | 36 | `stages.*.*.token_mixer` [DynamicFilter] 36 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>SPANetV2-B36-hybrid</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 199, 'aten::add': 130, 'aten::sigmoid': 60, 'aten::fft_rfft2': 60, 'aten::fft_irfft2': 60, 'aten::pow': 40, 'aten::gelu': 30, 'aten::softmax': 21, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 199 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [ExSPAM] 75; `stages.*.*.token_mixer` [MixAttention] 36; `stages.*.*.token_mixer` [Attention] 6 |
| `aten::add` | 130 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer` [MixAttention] 18 |
| `aten::fft_irfft2` | 60 | `stages.*.*.token_mixer` [ExSPAM] 60 |
| `aten::fft_rfft2` | 60 | `stages.*.*.token_mixer` [ExSPAM] 60 |
| `aten::sigmoid` | 60 | `stages.*.*.token_mixer` [ExSPAM] 60 |
| `aten::pow` | 40 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36 |
| `aten::gelu` | 30 | `stages.*.*.token_mixer.proj.*` [GELU] 15; `stages.*.*.token_mixer.query.*` [GELU] 15 |
| `aten::softmax` | 21 | `stages.*.*.token_mixer` [MixAttention] 18; `stages.*.*.token_mixer` [Attention] 3 |
| `aten::mean` | 9 | `(root)` [SPANetV2] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>CAFormer-B36</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 139, 'aten::add': 127, 'aten::pow': 55, 'aten::softmax': 21, 'aten::mean': 9, 'aten::sub': 4, 'aten::sqrt': 4, 'aten::div': 4, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 139 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [Attention] 42; `stages.*.*.token_mixer.act1` [StarReLU] 15 |
| `aten::add` | 127 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer.act1` [StarReLU] 15 |
| `aten::pow` | 55 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.token_mixer.act1` [StarReLU] 15 |
| `aten::softmax` | 21 | `stages.*.*.token_mixer` [Attention] 21 |
| `aten::mean` | 9 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6 |
| `aten::div` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sqrt` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::sub` | 4 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>CDFFormer-B36</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::add': 241, 'aten::mul': 232, 'aten::mean': 174, 'aten::pow': 169, 'aten::sub': 76, 'aten::sqrt': 76, 'aten::div': 76, 'aten::softmax': 21, 'aten::fft_rfft2': 21, 'aten::view_as_complex': 21, 'aten::fft_irfft2': 21, 'aten::square': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::add` | 241 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*` [MetaFormerBlock] 72; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 21 |
| `aten::mul` | 232 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.res_scale1` [Scale] 21; `stages.*.*.res_scale2` [Scale] 21; `stages.*.*.token_mixer` [DynamicFilter] 21; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 21 |
| `aten::mean` | 174 | `(root)` [MetaFormer] 1; `downsample_layers.*.post_norm` [LayerNormGeneral] 2; `downsample_layers.*.pre_norm` [LayerNormGeneral] 6; `stages.*.*.norm1` [LayerNormGeneral] 72; `stages.*.*.norm2` [LayerNormGeneral] 72; `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::pow` | 169 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.mlp.act` [StarReLU] 36; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36; `stages.*.*.token_mixer.act1` [StarReLU] 36; `stages.*.*.token_mixer.reweight.act` [StarReLU] 21 |
| `aten::div` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::sqrt` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::sub` | 76 | `downsample_layers.*.post_norm` [LayerNormGeneral] 1; `downsample_layers.*.pre_norm` [LayerNormGeneral] 3; `stages.*.*.norm1` [LayerNormGeneral] 36; `stages.*.*.norm2` [LayerNormGeneral] 36 |
| `aten::fft_irfft2` | 21 | `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::fft_rfft2` | 21 | `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::softmax` | 21 | `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::view_as_complex` | 21 | `stages.*.*.token_mixer` [DynamicFilter] 21 |
| `aten::square` | 1 | `head.act` [SquaredReLU] 1 |

</details>

<details><summary><b>GFNet-H-S</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 38, 'aten::add': 20, 'aten::fft_rfft2': 19, 'aten::view_as_complex': 19, 'aten::fft_irfft2': 19, 'aten::gelu': 19, 'aten::mean': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 38 | `blocks.*.*` [BlockLayerScale] 19; `blocks.*.*.filter` [GlobalFilter] 19 |
| `aten::add` | 20 | `(root)` [GFNetPyramid] 1; `blocks.*.*` [BlockLayerScale] 19 |
| `aten::fft_irfft2` | 19 | `blocks.*.*.filter` [GlobalFilter] 19 |
| `aten::fft_rfft2` | 19 | `blocks.*.*.filter` [GlobalFilter] 19 |
| `aten::gelu` | 19 | `blocks.*.*.mlp.act` [GELU] 19 |
| `aten::view_as_complex` | 19 | `blocks.*.*.filter` [GlobalFilter] 19 |
| `aten::mean` | 1 | `(root)` [GFNetPyramid] 1 |

</details>

<details><summary><b>GFNet-H-B</b></summary>

`fvcore.nn.FlopCountAnalysis(model, x).unsupported_ops()` 출력 원문:

```
Counter({'aten::mul': 72, 'aten::add': 37, 'aten::fft_rfft2': 36, 'aten::view_as_complex': 36, 'aten::fft_irfft2': 36, 'aten::gelu': 36, 'aten::mean': 1})
```

| 연산 | 호출 횟수 (전체) | 모듈 (이름 패턴 [클래스] 호출 횟수) |
|---|---|---|
| `aten::mul` | 72 | `blocks.*.*` [BlockLayerScale] 36; `blocks.*.*.filter` [GlobalFilter] 36 |
| `aten::add` | 37 | `(root)` [GFNetPyramid] 1; `blocks.*.*` [BlockLayerScale] 36 |
| `aten::fft_irfft2` | 36 | `blocks.*.*.filter` [GlobalFilter] 36 |
| `aten::fft_rfft2` | 36 | `blocks.*.*.filter` [GlobalFilter] 36 |
| `aten::gelu` | 36 | `blocks.*.*.mlp.act` [GELU] 36 |
| `aten::view_as_complex` | 36 | `blocks.*.*.filter` [GlobalFilter] 36 |
| `aten::mean` | 1 | `(root)` [GFNetPyramid] 1 |

</details>

## 4. 하나의 규약으로 다시 잰 값

[measure.py](measure.py)의 `Convention` 클래스가 fvcore op handle을 새로 등록해 모든 모델에 똑같이 적용한다. 노트북의 `rfft_flop_jit`은 쓰지 않는다. 등록한 처리 함수를 거치지 않은 연산이 하나라도 남으면 멈추게 했고, 모든 모델에서 남은 연산이 없었다. (가)+(a)+(b)+(c)는 처리 함수를 등록한 `FlopCountAnalysis.total()`과 모든 모델에서 같았다.

- **(a) FFT** — `fft_rfft2`, `fft_irfft2`, `fft_fft2`, `fft_ifft2`(및 `fftn`·1차원 변형). 변환 축은 연산의 `dim` 인자에서 읽는다. 실수 영역 텐서의 원소 수를 E(`rfft*`는 입력, `irfft*`는 출력; 복소 변환 `fft*`/`ifft*`는 복소 출력), 변환 축 크기의 곱을 N이라 하고 한 번의 변환을 **E·⌈log2 N⌉** MACs로 센다(기존 노트북과 같은 규약; 순변환·역변환 한 쌍이면 2·H·W·C·⌈log2(H·W)⌉). `fft_fftshift`, `fft_ifftshift`는 0. 변환마다 실수/복소를 적었다.
- **(b) 주파수 영역의 원소별 곱** — 출력이 복소수인 `aten::mul`. 출력 원소 하나를 1로 센다. SPANet의 SPF 마스크 곱, SPANetV2의 SRF 곱, DFFormer·CDFFormer·GFNet의 `x * weight`가 여기에 든다.
- **(c) 그 밖의 원소별 연산** — fvcore 기본값이 세지 않은 나머지 산술 연산(`add`, `sub`, 실수 `mul`, `div`, `pow`, `sqrt`, `square`, `gelu`, `sigmoid`, `softmax`, `mean`). 출력 원소 하나를 1로 센다. `mean`도 규약대로 출력 원소 수로 셌고, 입력 원소 수는 아래 상세 표에 따로 남겼다.
- **0으로 센 연산** — 자료 이동·형 변환: `clone`, `real`, `view_as_complex`. fvcore 무시 목록의 `relu`는 그대로 0이며, 원소 수만 참고로 남겼다.
- 각 연산의 원소 수와 자료형(Float / ComplexFloat)을 남겼으므로 다른 규약(예: 복소 곱 = 실수 곱 4회)으로 다시 계산할 수 있다.

### 처리 함수 확인

모델마다 FFT 연산 종류별로 첫 호출 하나를 골라, 입력 shape, `dim` 인자, 계산된 값을 출력했다. ConvFormer·CAFormer에는 FFT 연산이 없다.

| 모델 | 연산 | 변환 | 모듈 | 입력 shape (dtype) | 출력 shape (dtype) | dim | s | E 텐서 | E | N | ⌈log2 N⌉ | E·⌈log2 N⌉ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPANet-S | `aten::fft_fft2` | 복소 | `network.0.0.token_mixer.sps.0` | (1, 22, 56, 56) (Float) | (1, 22, 56, 56) (ComplexFloat) | [2, 3] | None | 출력 | 68,992 | 3,136 | 12 | 827,904 |
| SPANet-S | `aten::fft_ifft2` | 복소 | `network.0.0.token_mixer.sps.0` | (1, 22, 56, 56) (ComplexFloat) | (1, 22, 56, 56) (ComplexFloat) | [2, 3] | [None, None] | 출력 | 68,992 | 3,136 | 12 | 827,904 |
| SPANetV2-S18-pure | `aten::fft_rfft2` | 실수 | `stages.0.0.token_mixer` | (1, 16, 56, 56) (Float) | (1, 16, 56, 29) (ComplexFloat) | [2, 3] | None | 입력 | 50,176 | 3,136 | 12 | 602,112 |
| SPANetV2-S18-pure | `aten::fft_irfft2` | 실수 | `stages.0.0.token_mixer` | (1, 16, 56, 29) (ComplexFloat) | (1, 16, 56, 56) (Float) | [2, 3] | [None, None] | 출력 | 50,176 | 3,136 | 12 | 602,112 |
| DFFormer-S18 | `aten::fft_rfft2` | 실수 | `stages.0.0.token_mixer` | (1, 56, 56, 128) (Float) | (1, 56, 29, 128) (ComplexFloat) | [1, 2] | None | 입력 | 401,408 | 3,136 | 12 | 4,816,896 |
| DFFormer-S18 | `aten::fft_irfft2` | 실수 | `stages.0.0.token_mixer` | (1, 56, 29, 128) (ComplexFloat) | (1, 56, 56, 128) (Float) | [1, 2] | [None, None] | 출력 | 401,408 | 3,136 | 12 | 4,816,896 |
| SPANetV2-S18-hybrid | `aten::fft_rfft2` | 실수 | `stages.0.0.token_mixer` | (1, 16, 56, 56) (Float) | (1, 16, 56, 29) (ComplexFloat) | [2, 3] | None | 입력 | 50,176 | 3,136 | 12 | 602,112 |
| SPANetV2-S18-hybrid | `aten::fft_irfft2` | 실수 | `stages.0.0.token_mixer` | (1, 16, 56, 29) (ComplexFloat) | (1, 16, 56, 56) (Float) | [2, 3] | [None, None] | 출력 | 50,176 | 3,136 | 12 | 602,112 |
| CDFFormer-S18 | `aten::fft_rfft2` | 실수 | `stages.2.0.token_mixer` | (1, 14, 14, 640) (Float) | (1, 14, 8, 640) (ComplexFloat) | [1, 2] | None | 입력 | 125,440 | 196 | 8 | 1,003,520 |
| CDFFormer-S18 | `aten::fft_irfft2` | 실수 | `stages.2.0.token_mixer` | (1, 14, 8, 640) (ComplexFloat) | (1, 14, 14, 640) (Float) | [1, 2] | [None, None] | 출력 | 125,440 | 196 | 8 | 1,003,520 |
| SPANet-M | `aten::fft_fft2` | 복소 | `network.0.0.token_mixer.sps.0` | (1, 22, 56, 56) (Float) | (1, 22, 56, 56) (ComplexFloat) | [2, 3] | None | 출력 | 68,992 | 3,136 | 12 | 827,904 |
| SPANet-M | `aten::fft_ifft2` | 복소 | `network.0.0.token_mixer.sps.0` | (1, 22, 56, 56) (ComplexFloat) | (1, 22, 56, 56) (ComplexFloat) | [2, 3] | [None, None] | 출력 | 68,992 | 3,136 | 12 | 827,904 |
| SPANetV2-S36-pure | `aten::fft_rfft2` | 실수 | `stages.0.0.token_mixer` | (1, 16, 56, 56) (Float) | (1, 16, 56, 29) (ComplexFloat) | [2, 3] | None | 입력 | 50,176 | 3,136 | 12 | 602,112 |
| SPANetV2-S36-pure | `aten::fft_irfft2` | 실수 | `stages.0.0.token_mixer` | (1, 16, 56, 29) (ComplexFloat) | (1, 16, 56, 56) (Float) | [2, 3] | [None, None] | 출력 | 50,176 | 3,136 | 12 | 602,112 |
| DFFormer-S36 | `aten::fft_rfft2` | 실수 | `stages.0.0.token_mixer` | (1, 56, 56, 128) (Float) | (1, 56, 29, 128) (ComplexFloat) | [1, 2] | None | 입력 | 401,408 | 3,136 | 12 | 4,816,896 |
| DFFormer-S36 | `aten::fft_irfft2` | 실수 | `stages.0.0.token_mixer` | (1, 56, 29, 128) (ComplexFloat) | (1, 56, 56, 128) (Float) | [1, 2] | [None, None] | 출력 | 401,408 | 3,136 | 12 | 4,816,896 |
| SPANetV2-S36-hybrid | `aten::fft_rfft2` | 실수 | `stages.0.0.token_mixer` | (1, 16, 56, 56) (Float) | (1, 16, 56, 29) (ComplexFloat) | [2, 3] | None | 입력 | 50,176 | 3,136 | 12 | 602,112 |
| SPANetV2-S36-hybrid | `aten::fft_irfft2` | 실수 | `stages.0.0.token_mixer` | (1, 16, 56, 29) (ComplexFloat) | (1, 16, 56, 56) (Float) | [2, 3] | [None, None] | 출력 | 50,176 | 3,136 | 12 | 602,112 |
| CDFFormer-S36 | `aten::fft_rfft2` | 실수 | `stages.2.0.token_mixer` | (1, 14, 14, 640) (Float) | (1, 14, 8, 640) (ComplexFloat) | [1, 2] | None | 입력 | 125,440 | 196 | 8 | 1,003,520 |
| CDFFormer-S36 | `aten::fft_irfft2` | 실수 | `stages.2.0.token_mixer` | (1, 14, 8, 640) (ComplexFloat) | (1, 14, 14, 640) (Float) | [1, 2] | [None, None] | 출력 | 125,440 | 196 | 8 | 1,003,520 |
| SPANet-B | `aten::fft_fft2` | 복소 | `network.0.0.token_mixer.sps.0` | (1, 32, 56, 56) (Float) | (1, 32, 56, 56) (ComplexFloat) | [2, 3] | None | 출력 | 100,352 | 3,136 | 12 | 1,204,224 |
| SPANet-B | `aten::fft_ifft2` | 복소 | `network.0.0.token_mixer.sps.0` | (1, 32, 56, 56) (ComplexFloat) | (1, 32, 56, 56) (ComplexFloat) | [2, 3] | [None, None] | 출력 | 100,352 | 3,136 | 12 | 1,204,224 |
| SPANetV2-M36-pure | `aten::fft_rfft2` | 실수 | `stages.0.0.token_mixer` | (1, 24, 56, 56) (Float) | (1, 24, 56, 29) (ComplexFloat) | [2, 3] | None | 입력 | 75,264 | 3,136 | 12 | 903,168 |
| SPANetV2-M36-pure | `aten::fft_irfft2` | 실수 | `stages.0.0.token_mixer` | (1, 24, 56, 29) (ComplexFloat) | (1, 24, 56, 56) (Float) | [2, 3] | [None, None] | 출력 | 75,264 | 3,136 | 12 | 903,168 |
| DFFormer-M36 | `aten::fft_rfft2` | 실수 | `stages.0.0.token_mixer` | (1, 56, 56, 192) (Float) | (1, 56, 29, 192) (ComplexFloat) | [1, 2] | None | 입력 | 602,112 | 3,136 | 12 | 7,225,344 |
| DFFormer-M36 | `aten::fft_irfft2` | 실수 | `stages.0.0.token_mixer` | (1, 56, 29, 192) (ComplexFloat) | (1, 56, 56, 192) (Float) | [1, 2] | [None, None] | 출력 | 602,112 | 3,136 | 12 | 7,225,344 |
| SPANetV2-M36-hybrid | `aten::fft_rfft2` | 실수 | `stages.0.0.token_mixer` | (1, 24, 56, 56) (Float) | (1, 24, 56, 29) (ComplexFloat) | [2, 3] | None | 입력 | 75,264 | 3,136 | 12 | 903,168 |
| SPANetV2-M36-hybrid | `aten::fft_irfft2` | 실수 | `stages.0.0.token_mixer` | (1, 24, 56, 29) (ComplexFloat) | (1, 24, 56, 56) (Float) | [2, 3] | [None, None] | 출력 | 75,264 | 3,136 | 12 | 903,168 |
| CDFFormer-M36 | `aten::fft_rfft2` | 실수 | `stages.2.0.token_mixer` | (1, 14, 14, 768) (Float) | (1, 14, 8, 768) (ComplexFloat) | [1, 2] | None | 입력 | 150,528 | 196 | 8 | 1,204,224 |
| CDFFormer-M36 | `aten::fft_irfft2` | 실수 | `stages.2.0.token_mixer` | (1, 14, 8, 768) (ComplexFloat) | (1, 14, 14, 768) (Float) | [1, 2] | [None, None] | 출력 | 150,528 | 196 | 8 | 1,204,224 |
| SPANetV2-B36-pure | `aten::fft_rfft2` | 실수 | `stages.0.0.token_mixer` | (1, 32, 56, 56) (Float) | (1, 32, 56, 29) (ComplexFloat) | [2, 3] | None | 입력 | 100,352 | 3,136 | 12 | 1,204,224 |
| SPANetV2-B36-pure | `aten::fft_irfft2` | 실수 | `stages.0.0.token_mixer` | (1, 32, 56, 29) (ComplexFloat) | (1, 32, 56, 56) (Float) | [2, 3] | [None, None] | 출력 | 100,352 | 3,136 | 12 | 1,204,224 |
| DFFormer-B36 | `aten::fft_rfft2` | 실수 | `stages.0.0.token_mixer` | (1, 56, 56, 256) (Float) | (1, 56, 29, 256) (ComplexFloat) | [1, 2] | None | 입력 | 802,816 | 3,136 | 12 | 9,633,792 |
| DFFormer-B36 | `aten::fft_irfft2` | 실수 | `stages.0.0.token_mixer` | (1, 56, 29, 256) (ComplexFloat) | (1, 56, 56, 256) (Float) | [1, 2] | [None, None] | 출력 | 802,816 | 3,136 | 12 | 9,633,792 |
| SPANetV2-B36-hybrid | `aten::fft_rfft2` | 실수 | `stages.0.0.token_mixer` | (1, 32, 56, 56) (Float) | (1, 32, 56, 29) (ComplexFloat) | [2, 3] | None | 입력 | 100,352 | 3,136 | 12 | 1,204,224 |
| SPANetV2-B36-hybrid | `aten::fft_irfft2` | 실수 | `stages.0.0.token_mixer` | (1, 32, 56, 29) (ComplexFloat) | (1, 32, 56, 56) (Float) | [2, 3] | [None, None] | 출력 | 100,352 | 3,136 | 12 | 1,204,224 |
| CDFFormer-B36 | `aten::fft_rfft2` | 실수 | `stages.2.0.token_mixer` | (1, 14, 14, 1024) (Float) | (1, 14, 8, 1024) (ComplexFloat) | [1, 2] | None | 입력 | 200,704 | 196 | 8 | 1,605,632 |
| CDFFormer-B36 | `aten::fft_irfft2` | 실수 | `stages.2.0.token_mixer` | (1, 14, 8, 1024) (ComplexFloat) | (1, 14, 14, 1024) (Float) | [1, 2] | [None, None] | 출력 | 200,704 | 196 | 8 | 1,605,632 |
| GFNet-H-S | `aten::fft_rfft2` | 실수 | `blocks.0.0.filter` | (1, 56, 56, 96) (Float) | (1, 56, 29, 96) (ComplexFloat) | [1, 2] | None | 입력 | 301,056 | 3,136 | 12 | 3,612,672 |
| GFNet-H-S | `aten::fft_irfft2` | 실수 | `blocks.0.0.filter` | (1, 56, 29, 96) (ComplexFloat) | (1, 56, 56, 96) (Float) | [1, 2] | [56, 56] | 출력 | 301,056 | 3,136 | 12 | 3,612,672 |
| GFNet-H-B | `aten::fft_rfft2` | 실수 | `blocks.0.0.filter` | (1, 56, 56, 96) (Float) | (1, 56, 29, 96) (ComplexFloat) | [1, 2] | None | 입력 | 301,056 | 3,136 | 12 | 3,612,672 |
| GFNet-H-B | `aten::fft_irfft2` | 실수 | `blocks.0.0.filter` | (1, 56, 29, 96) (ComplexFloat) | (1, 56, 56, 96) (Float) | [1, 2] | [56, 56] | 출력 | 301,056 | 3,136 | 12 | 3,612,672 |

### 모델별 (a)(b)(c) 상세

<details><summary><b>SPANet-S</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_fft2` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 72 | 2,057,216 | 20,271,104 |  |
| (a) FFT | `aten::fft_fftshift` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 72 | 2,057,216 | 0 |  |
| (a) FFT | `aten::fft_ifft2` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 72 | 2,057,216 | 20,271,104 |  |
| (a) FFT | `aten::fft_ifftshift` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 72 | 2,057,216 | 0 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 72 | 2,057,216 | 2,057,216 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `network.*.*` | 48 | 4,114,432 | 4,114,432 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `network.*.*.token_mixer` | 48 | 4,114,432 | 4,114,432 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `network.*.*.mlp.act` | 24 | 8,228,864 | 8,228,864 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `network.*.*.res_scale1` | 16 | 852,992 | 852,992 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `network.*.*.res_scale2` | 16 | 852,992 | 852,992 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `network.*.*.token_mixer` | 24 | 2,057,216 | 2,057,216 |  |
| 0 (자료 이동) | `aten::clone` | Float | `network.*.*.token_mixer` | 24 | 2,057,216 | 0 |  |
| 0 (자료 이동) | `aten::real` | Float | `network.*.*.token_mixer.sps.*` | 72 | 2,057,216 | 0 |  |

합계: (a) 40,542,208 / (b) 2,057,216 / (c) 20,221,440 MACs

</details>

<details><summary><b>SPANetV2-S18-pure</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 72 | 1,542,912 | 15,203,328 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 72 | 1,542,912 | 15,203,328 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 72 | 838,656 | 838,656 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 36 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.proj.*` | 18 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.query.*` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 401,408 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 727,552 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 12 | 639,744 | 639,744 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 12 | 639,744 | 639,744 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::sigmoid` | Float | `stages.*.*.token_mixer` | 72 | 838,656 | 838,656 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,048 | 2,048 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 18 | 6,171,648 | 0 |  |

합계: (a) 30,406,656 / (b) 838,656 / (c) 32,180,048 MACs

</details>

<details><summary><b>ConvFormer-S18</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 36 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 18 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 401,408 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 727,552 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 12 | 639,744 | 639,744 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 12 | 639,744 | 639,744 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 18 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 18 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,048 | 2,048 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 18 | 6,171,648 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 18 | 3,085,824 | 0 |  |

합계: (a) 0 / (b) 0 / (c) 34,427,216 MACs

</details>

<details><summary><b>DFFormer-S18</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 18 | 3,085,824 | 30,406,656 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 18 | 3,085,824 | 30,406,656 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 18 | 1,677,312 | 1,677,312 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 36 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm1` | 18 | 13,671 | 13,671 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm2` | 18 | 13,671 | 13,671 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 18 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.reweight.act` | 18 | 1,248 | 1,248 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm1` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm2` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 401,408 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 727,552 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm1` | 36 | 27,342 | 27,342 | 3,085,824 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm2` | 36 | 27,342 | 27,342 | 3,085,824 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.token_mixer` | 18 | 4,992 | 4,992 | 1,542,912 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm1` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm2` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 12 | 639,744 | 639,744 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 12 | 639,744 | 639,744 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 18 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.reweight.act` | 18 | 1,248 | 1,248 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm1` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm2` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 18 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.reweight.act` | 18 | 1,248 | 1,248 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 18 | 39,936 | 39,936 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm1` | 18 | 13,671 | 13,671 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm2` | 18 | 13,671 | 13,671 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,048 | 2,048 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm1` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm2` | 18 | 1,542,912 | 1,542,912 |  |
| 0 (자료 이동) | `aten::view_as_complex` | ComplexFloat | `stages.*.*.token_mixer` | 18 | 28,896 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 18 | 6,171,648 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 18 | 3,085,824 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.reweight.act.relu` | 18 | 1,248 | 0 |  |

합계: (a) 60,813,312 / (b) 1,677,312 / (c) 46,928,552 MACs

</details>

<details><summary><b>SPANetV2-S18-hybrid</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 24 | 903,168 | 10,235,904 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 24 | 903,168 | 10,235,904 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 24 | 473,088 | 473,088 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 36 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer` | 9 | 564,480 | 564,480 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.proj.*` | 6 | 1,806,336 | 1,806,336 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.query.*` | 6 | 903,168 | 903,168 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 401,408 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 727,552 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 12 | 639,744 | 639,744 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 12 | 639,744 | 639,744 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer` | 18 | 4,475,856 | 4,475,856 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Long | `stages.*.*.token_mixer` | 12 | 0 | 12 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::sigmoid` | Float | `stages.*.*.token_mixer` | 24 | 473,088 | 473,088 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 12 | 3,572,688 | 3,572,688 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,048 | 2,048 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 18 | 6,171,648 | 0 |  |

합계: (a) 20,471,808 / (b) 473,088 / (c) 36,965,372 MACs

</details>

<details><summary><b>CAFormer-S18</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 36 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 6 | 1,806,336 | 1,806,336 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 401,408 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 727,552 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 12 | 639,744 | 639,744 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 12 | 639,744 | 639,744 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Long | `stages.*.*.token_mixer` | 12 | 0 | 12 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer` | 12 | 3,572,688 | 3,572,688 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 6 | 1,806,336 | 1,806,336 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 6 | 1,806,336 | 1,806,336 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 12 | 3,572,688 | 3,572,688 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,048 | 2,048 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 18 | 6,171,648 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 6 | 1,806,336 | 0 |  |

합계: (a) 0 / (b) 0 / (c) 37,734,140 MACs

</details>

<details><summary><b>CDFFormer-S18</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 12 | 1,279,488 | 9,934,848 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 12 | 1,279,488 | 9,934,848 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 12 | 731,136 | 731,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 36 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm1` | 18 | 13,671 | 13,671 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm2` | 18 | 13,671 | 13,671 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 18 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.reweight.act` | 12 | 1,104 | 1,104 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm1` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm2` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 401,408 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 727,552 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm1` | 36 | 27,342 | 27,342 | 3,085,824 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm2` | 36 | 27,342 | 27,342 | 3,085,824 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.token_mixer` | 12 | 4,416 | 4,416 | 639,744 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm1` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm2` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 12 | 639,744 | 639,744 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 12 | 639,744 | 639,744 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 18 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.reweight.act` | 12 | 1,104 | 1,104 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 18 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm1` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm2` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 18 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.reweight.act` | 12 | 1,104 | 1,104 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 12 | 35,328 | 35,328 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm1` | 18 | 13,671 | 13,671 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm2` | 18 | 13,671 | 13,671 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,048 | 2,048 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm1` | 18 | 1,542,912 | 1,542,912 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm2` | 18 | 1,542,912 | 1,542,912 |  |
| 0 (자료 이동) | `aten::view_as_complex` | ComplexFloat | `stages.*.*.token_mixer` | 12 | 4,368 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 18 | 6,171,648 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 18 | 3,085,824 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.reweight.act.relu` | 12 | 1,104 | 0 |  |

합계: (a) 19,869,696 / (b) 731,136 / (c) 46,922,936 MACs

</details>

<details><summary><b>SPANet-M</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_fft2` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 108 | 3,085,824 | 30,406,656 |  |
| (a) FFT | `aten::fft_fftshift` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 108 | 3,085,824 | 0 |  |
| (a) FFT | `aten::fft_ifft2` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 108 | 3,085,824 | 30,406,656 |  |
| (a) FFT | `aten::fft_ifftshift` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 108 | 3,085,824 | 0 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 108 | 3,085,824 | 3,085,824 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `network.*.*` | 72 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `network.*.*.token_mixer` | 72 | 6,171,648 | 6,171,648 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `network.*.*.mlp.act` | 36 | 12,343,296 | 12,343,296 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `network.*.*.res_scale1` | 24 | 1,279,488 | 1,279,488 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `network.*.*.res_scale2` | 24 | 1,279,488 | 1,279,488 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `network.*.*.token_mixer` | 36 | 3,085,824 | 3,085,824 |  |
| 0 (자료 이동) | `aten::clone` | Float | `network.*.*.token_mixer` | 36 | 3,085,824 | 0 |  |
| 0 (자료 이동) | `aten::real` | Float | `network.*.*.token_mixer.sps.*` | 108 | 3,085,824 | 0 |  |

합계: (a) 60,813,312 / (b) 3,085,824 / (c) 30,331,904 MACs

</details>

<details><summary><b>SPANetV2-S36-pure</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 144 | 3,010,560 | 28,750,848 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 144 | 3,010,560 | 28,750,848 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 144 | 1,645,056 | 1,645,056 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.proj.*` | 36 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.query.*` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 401,408 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 727,552 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::sigmoid` | Float | `stages.*.*.token_mixer` | 144 | 1,645,056 | 1,645,056 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,048 | 2,048 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 12,042,240 | 0 |  |

합계: (a) 57,501,696 / (b) 1,645,056 / (c) 60,533,072 MACs

</details>

<details><summary><b>ConvFormer-S36</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 36 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 401,408 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 727,552 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 36 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 36 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,048 | 2,048 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 12,042,240 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 36 | 6,021,120 | 0 |  |

합계: (a) 0 / (b) 0 / (c) 64,909,136 MACs

</details>

<details><summary><b>DFFormer-S36</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 36 | 6,021,120 | 57,501,696 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 36 | 6,021,120 | 57,501,696 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 36 | 3,290,112 | 3,290,112 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm1` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm2` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 36 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.reweight.act` | 36 | 2,256 | 2,256 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm1` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm2` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 401,408 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 727,552 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm1` | 72 | 44,982 | 44,982 | 6,021,120 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm2` | 72 | 44,982 | 44,982 | 6,021,120 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.token_mixer` | 36 | 9,024 | 9,024 | 3,010,560 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm1` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm2` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 36 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.reweight.act` | 36 | 2,256 | 2,256 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm1` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm2` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 36 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.reweight.act` | 36 | 2,256 | 2,256 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 36 | 72,192 | 72,192 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm1` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm2` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,048 | 2,048 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm1` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm2` | 36 | 3,010,560 | 3,010,560 |  |
| 0 (자료 이동) | `aten::view_as_complex` | ComplexFloat | `stages.*.*.token_mixer` | 36 | 48,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 12,042,240 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 36 | 6,021,120 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.reweight.act.relu` | 36 | 2,256 | 0 |  |

합계: (a) 115,003,392 / (b) 3,290,112 / (c) 89,261,528 MACs

</details>

<details><summary><b>SPANetV2-S36-hybrid</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 60 | 1,806,336 | 19,267,584 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 60 | 1,806,336 | 19,267,584 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 60 | 956,928 | 956,928 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer` | 18 | 1,128,960 | 1,128,960 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.proj.*` | 15 | 3,612,672 | 3,612,672 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.query.*` | 15 | 1,806,336 | 1,806,336 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 401,408 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 727,552 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer` | 36 | 8,836,464 | 8,836,464 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Long | `stages.*.*.token_mixer` | 21 | 0 | 21 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::sigmoid` | Float | `stages.*.*.token_mixer` | 60 | 956,928 | 956,928 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 21 | 7,030,128 | 7,030,128 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,048 | 2,048 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 12,042,240 | 0 |  |

합계: (a) 38,535,168 / (b) 956,928 / (c) 70,217,285 MACs

</details>

<details><summary><b>CAFormer-S36</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 15 | 3,612,672 | 3,612,672 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 401,408 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 727,552 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Long | `stages.*.*.token_mixer` | 21 | 0 | 21 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer` | 21 | 7,030,128 | 7,030,128 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 15 | 3,612,672 | 3,612,672 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 15 | 3,612,672 | 3,612,672 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 21 | 7,030,128 | 7,030,128 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,048 | 2,048 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 12,042,240 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 15 | 3,612,672 | 0 |  |

합계: (a) 0 / (b) 0 / (c) 71,744,069 MACs

</details>

<details><summary><b>CDFFormer-S36</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 21 | 2,408,448 | 18,966,528 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 21 | 2,408,448 | 18,966,528 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 21 | 1,376,256 | 1,376,256 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm1` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm2` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 36 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.reweight.act` | 21 | 1,824 | 1,824 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm1` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm2` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 512 | 512 | 25,088 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 401,408 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 727,552 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm1` | 72 | 44,982 | 44,982 | 6,021,120 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm2` | 72 | 44,982 | 44,982 | 6,021,120 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.token_mixer` | 21 | 7,296 | 7,296 | 1,204,224 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm1` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm2` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 36 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.reweight.act` | 21 | 1,824 | 1,824 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 12,042,240 | 12,042,240 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm1` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm2` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 36 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.reweight.act` | 21 | 1,824 | 1,824 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 21 | 58,368 | 58,368 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm1` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm2` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,048 | 2,048 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 200,704 | 200,704 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 363,776 | 363,776 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm1` | 36 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm2` | 36 | 3,010,560 | 3,010,560 |  |
| 0 (자료 이동) | `aten::view_as_complex` | ComplexFloat | `stages.*.*.token_mixer` | 21 | 8,400 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 12,042,240 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 36 | 6,021,120 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.reweight.act.relu` | 21 | 1,824 | 0 |  |

합계: (a) 37,933,056 / (b) 1,376,256 / (c) 89,244,680 MACs

</details>

<details><summary><b>SPANet-B</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_fft2` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 108 | 4,290,048 | 42,900,480 |  |
| (a) FFT | `aten::fft_fftshift` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 108 | 4,290,048 | 0 |  |
| (a) FFT | `aten::fft_ifft2` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 108 | 4,290,048 | 42,900,480 |  |
| (a) FFT | `aten::fft_ifftshift` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 108 | 4,290,048 | 0 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `network.*.*.token_mixer.sps.*` | 108 | 4,290,048 | 4,290,048 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `network.*.*` | 72 | 8,580,096 | 8,580,096 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `network.*.*.token_mixer` | 72 | 8,580,096 | 8,580,096 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `network.*.*.mlp.act` | 36 | 17,160,192 | 17,160,192 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 768 | 768 | 37,632 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `network.*.*.res_scale1` | 24 | 1,580,544 | 1,580,544 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `network.*.*.res_scale2` | 24 | 1,580,544 | 1,580,544 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `network.*.*.token_mixer` | 36 | 4,290,048 | 4,290,048 |  |
| 0 (자료 이동) | `aten::clone` | Float | `network.*.*.token_mixer` | 36 | 4,290,048 | 0 |  |
| 0 (자료 이동) | `aten::real` | Float | `network.*.*.token_mixer.sps.*` | 108 | 4,290,048 | 0 |  |

합계: (a) 85,800,960 / (b) 4,290,048 / (c) 41,772,288 MACs

</details>

<details><summary><b>SPANetV2-M36-pure</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 144 | 4,148,928 | 40,247,424 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 144 | 4,148,928 | 40,247,424 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 144 | 2,257,920 | 2,257,920 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.proj.*` | 36 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.query.*` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 576 | 576 | 28,224 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 602,112 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 1,053,696 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,439,424 | 1,439,424 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,439,424 | 1,439,424 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::sigmoid` | Float | `stages.*.*.token_mixer` | 144 | 2,257,920 | 2,257,920 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,304 | 2,304 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,304 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 16,595,712 | 0 |  |

합계: (a) 80,494,848 / (b) 2,257,920 / (c) 83,160,976 MACs

</details>

<details><summary><b>ConvFormer-M36</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 36 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 576 | 576 | 28,224 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 602,112 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 1,053,696 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,439,424 | 1,439,424 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,439,424 | 1,439,424 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 36 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 36 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,304 | 2,304 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,304 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 16,595,712 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 36 | 8,297,856 | 0 |  |

합계: (a) 0 / (b) 0 / (c) 89,200,912 MACs

</details>

<details><summary><b>DFFormer-M36</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 36 | 8,297,856 | 80,494,848 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 36 | 8,297,856 | 80,494,848 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 36 | 4,515,840 | 4,515,840 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm1` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm2` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 36 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.reweight.act` | 36 | 2,808 | 2,808 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm1` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm2` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 576 | 576 | 28,224 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 602,112 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 1,053,696 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm1` | 72 | 44,982 | 44,982 | 8,297,856 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm2` | 72 | 44,982 | 44,982 | 8,297,856 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.token_mixer` | 36 | 11,232 | 11,232 | 4,148,928 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm1` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm2` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,439,424 | 1,439,424 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,439,424 | 1,439,424 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 36 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.reweight.act` | 36 | 2,808 | 2,808 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm1` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm2` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 36 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.reweight.act` | 36 | 2,808 | 2,808 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 36 | 89,856 | 89,856 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm1` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm2` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,304 | 2,304 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm1` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm2` | 36 | 4,148,928 | 4,148,928 |  |
| 0 (자료 이동) | `aten::view_as_complex` | ComplexFloat | `stages.*.*.token_mixer` | 36 | 48,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,304 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 16,595,712 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 36 | 8,297,856 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.reweight.act.relu` | 36 | 2,808 | 0 |  |

합계: (a) 160,989,696 / (b) 4,515,840 / (c) 122,681,776 MACs

</details>

<details><summary><b>SPANetV2-M36-hybrid</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 60 | 2,709,504 | 28,901,376 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 60 | 2,709,504 | 28,901,376 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 60 | 1,435,392 | 1,435,392 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer` | 18 | 1,354,752 | 1,354,752 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.proj.*` | 15 | 5,419,008 | 5,419,008 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.query.*` | 15 | 2,709,504 | 2,709,504 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 576 | 576 | 28,224 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 602,112 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 1,053,696 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,439,424 | 1,439,424 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,439,424 | 1,439,424 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer` | 36 | 11,137,014 | 11,137,014 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Long | `stages.*.*.token_mixer` | 21 | 0 | 21 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::sigmoid` | Float | `stages.*.*.token_mixer` | 60 | 1,435,392 | 1,435,392 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 21 | 8,427,510 | 8,427,510 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,304 | 2,304 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,304 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 16,595,712 | 0 |  |

합계: (a) 57,802,752 / (b) 1,435,392 / (c) 94,790,545 MACs

</details>

<details><summary><b>CAFormer-M36</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 15 | 5,419,008 | 5,419,008 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 576 | 576 | 28,224 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 602,112 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 1,053,696 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,439,424 | 1,439,424 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,439,424 | 1,439,424 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Long | `stages.*.*.token_mixer` | 21 | 0 | 21 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer` | 21 | 8,427,510 | 8,427,510 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 15 | 5,419,008 | 5,419,008 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 15 | 5,419,008 | 5,419,008 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 21 | 8,427,510 | 8,427,510 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,304 | 2,304 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,304 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 16,595,712 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 15 | 5,419,008 | 0 |  |

합계: (a) 0 / (b) 0 / (c) 97,419,409 MACs

</details>

<details><summary><b>CDFFormer-M36</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 21 | 2,878,848 | 22,692,096 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 21 | 2,878,848 | 22,692,096 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 21 | 1,645,056 | 1,645,056 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm1` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm2` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 36 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.reweight.act` | 21 | 2,160 | 2,160 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm1` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm2` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 576 | 576 | 28,224 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 602,112 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 1,053,696 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm1` | 72 | 44,982 | 44,982 | 8,297,856 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm2` | 72 | 44,982 | 44,982 | 8,297,856 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.token_mixer` | 21 | 8,640 | 8,640 | 1,439,424 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm1` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm2` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,439,424 | 1,439,424 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,439,424 | 1,439,424 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 36 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.reweight.act` | 21 | 2,160 | 2,160 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 16,595,712 | 16,595,712 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm1` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm2` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 36 | 8,297,856 | 8,297,856 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.reweight.act` | 21 | 2,160 | 2,160 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 21 | 69,120 | 69,120 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm1` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm2` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 2,304 | 2,304 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 526,848 | 526,848 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm1` | 36 | 4,148,928 | 4,148,928 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm2` | 36 | 4,148,928 | 4,148,928 |  |
| 0 (자료 이동) | `aten::view_as_complex` | ComplexFloat | `stages.*.*.token_mixer` | 21 | 8,400 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 2,304 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 16,595,712 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 36 | 8,297,856 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.reweight.act.relu` | 21 | 2,160 | 0 |  |

합계: (a) 45,384,192 / (b) 1,645,056 / (c) 122,656,504 MACs

</details>

<details><summary><b>SPANetV2-B36-pure</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 144 | 5,531,904 | 53,663,232 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 144 | 5,531,904 | 53,663,232 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 144 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.proj.*` | 36 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.query.*` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 768 | 768 | 37,632 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 802,816 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 1,404,928 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,919,232 | 1,919,232 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,919,232 | 1,919,232 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::sigmoid` | Float | `stages.*.*.token_mixer` | 144 | 3,010,560 | 3,010,560 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 3,072 | 3,072 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 3,072 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 22,127,616 | 0 |  |

합계: (a) 107,326,464 / (b) 3,010,560 / (c) 110,871,632 MACs

</details>

<details><summary><b>ConvFormer-B36</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 36 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 768 | 768 | 37,632 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 802,816 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 1,404,928 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,919,232 | 1,919,232 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,919,232 | 1,919,232 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 36 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 36 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 3,072 | 3,072 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 3,072 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 22,127,616 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 36 | 11,063,808 | 0 |  |

합계: (a) 0 / (b) 0 / (c) 118,924,880 MACs

</details>

<details><summary><b>DFFormer-B36</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 36 | 11,063,808 | 107,326,464 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 36 | 11,063,808 | 107,326,464 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 36 | 6,021,120 | 6,021,120 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm1` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm2` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 36 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.reweight.act` | 36 | 3,744 | 3,744 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm1` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm2` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 768 | 768 | 37,632 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 802,816 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 1,404,928 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm1` | 72 | 44,982 | 44,982 | 11,063,808 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm2` | 72 | 44,982 | 44,982 | 11,063,808 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.token_mixer` | 36 | 14,976 | 14,976 | 5,531,904 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm1` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm2` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,919,232 | 1,919,232 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,919,232 | 1,919,232 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 36 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.reweight.act` | 36 | 3,744 | 3,744 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm1` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm2` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 36 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.reweight.act` | 36 | 3,744 | 3,744 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 36 | 119,808 | 119,808 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm1` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm2` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 3,072 | 3,072 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm1` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm2` | 36 | 5,531,904 | 5,531,904 |  |
| 0 (자료 이동) | `aten::view_as_complex` | ComplexFloat | `stages.*.*.token_mixer` | 36 | 48,048 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 3,072 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 22,127,616 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 36 | 11,063,808 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.reweight.act.relu` | 36 | 3,744 | 0 |  |

합계: (a) 214,652,928 / (b) 6,021,120 / (c) 163,506,056 MACs

</details>

<details><summary><b>SPANetV2-B36-hybrid</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 60 | 3,612,672 | 38,535,168 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 60 | 3,612,672 | 38,535,168 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 60 | 1,913,856 | 1,913,856 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer` | 18 | 1,806,336 | 1,806,336 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.proj.*` | 15 | 7,225,344 | 7,225,344 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `stages.*.*.token_mixer.query.*` | 15 | 3,612,672 | 3,612,672 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 768 | 768 | 37,632 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 802,816 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 1,404,928 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,919,232 | 1,919,232 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,919,232 | 1,919,232 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer` | 36 | 14,849,352 | 14,849,352 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Long | `stages.*.*.token_mixer` | 21 | 0 | 21 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::sigmoid` | Float | `stages.*.*.token_mixer` | 60 | 1,913,856 | 1,913,856 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 21 | 11,236,680 | 11,236,680 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 3,072 | 3,072 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 3,072 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 22,127,616 | 0 |  |

합계: (a) 77,070,336 / (b) 1,913,856 / (c) 126,377,717 MACs

</details>

<details><summary><b>CAFormer-B36</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 15 | 7,225,344 | 7,225,344 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 768 | 768 | 37,632 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 802,816 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 1,404,928 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,919,232 | 1,919,232 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,919,232 | 1,919,232 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Long | `stages.*.*.token_mixer` | 21 | 0 | 21 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer` | 21 | 11,236,680 | 11,236,680 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 15 | 7,225,344 | 7,225,344 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 15 | 7,225,344 | 7,225,344 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 21 | 11,236,680 | 11,236,680 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 3,072 | 3,072 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 3,072 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 22,127,616 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 15 | 7,225,344 | 0 |  |

합계: (a) 0 / (b) 0 / (c) 129,882,869 MACs

</details>

<details><summary><b>CDFFormer-B36</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `stages.*.*.token_mixer` | 21 | 3,838,464 | 30,256,128 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `stages.*.*.token_mixer` | 21 | 3,838,464 | 30,256,128 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `stages.*.*.token_mixer` | 21 | 2,193,408 | 2,193,408 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*` | 72 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm1` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.norm2` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.act1` | 36 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `stages.*.*.token_mixer.reweight.act` | 21 | 2,880 | 2,880 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm1` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::div` | Float | `stages.*.*.norm2` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 768 | 768 | 37,632 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.post_norm` | 2 | 6,272 | 6,272 | 802,816 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `downsample_layers.*.pre_norm` | 6 | 8,232 | 8,232 | 1,404,928 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm1` | 72 | 44,982 | 44,982 | 11,063,808 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.norm2` | 72 | 44,982 | 44,982 | 11,063,808 |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `stages.*.*.token_mixer` | 21 | 11,520 | 11,520 | 1,919,232 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm1` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.norm2` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale1` | 21 | 1,919,232 | 1,919,232 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.res_scale2` | 21 | 1,919,232 | 1,919,232 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.act1` | 36 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `stages.*.*.token_mixer.reweight.act` | 21 | 2,880 | 2,880 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.mlp.act` | 36 | 22,127,616 | 22,127,616 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm1` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.norm2` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.act1` | 36 | 11,063,808 | 11,063,808 |  |
| (c) 그 밖의 원소별 | `aten::pow` | Float | `stages.*.*.token_mixer.reweight.act` | 21 | 2,880 | 2,880 |  |
| (c) 그 밖의 원소별 | `aten::softmax` | Float | `stages.*.*.token_mixer` | 21 | 92,160 | 92,160 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.post_norm` | 1 | 3,136 | 3,136 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `downsample_layers.*.pre_norm` | 3 | 4,116 | 4,116 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm1` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::sqrt` | Float | `stages.*.*.norm2` | 36 | 22,491 | 22,491 |  |
| (c) 그 밖의 원소별 | `aten::square` | Float | `head.act` | 1 | 3,072 | 3,072 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.post_norm` | 1 | 401,408 | 401,408 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `downsample_layers.*.pre_norm` | 3 | 702,464 | 702,464 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm1` | 36 | 5,531,904 | 5,531,904 |  |
| (c) 그 밖의 원소별 | `aten::sub` | Float | `stages.*.*.norm2` | 36 | 5,531,904 | 5,531,904 |  |
| 0 (자료 이동) | `aten::view_as_complex` | ComplexFloat | `stages.*.*.token_mixer` | 21 | 8,400 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `head.act.relu` | 1 | 3,072 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.mlp.act.relu` | 36 | 22,127,616 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.act1.relu` | 36 | 11,063,808 | 0 |  |
| 0 (fvcore 무시 목록, 참고) | `aten::relu` | Float | `stages.*.*.token_mixer.reweight.act.relu` | 21 | 2,880 | 0 |  |

합계: (a) 60,512,256 / (b) 2,193,408 / (c) 163,472,360 MACs

</details>

<details><summary><b>GFNet-H-S</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `blocks.*.*.filter` | 19 | 2,220,288 | 22,052,352 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `blocks.*.*.filter` | 19 | 2,220,288 | 22,052,352 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `blocks.*.*.filter` | 19 | 1,204,224 | 1,204,224 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `(root)` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `blocks.*.*` | 19 | 2,220,288 | 2,220,288 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `blocks.*.*.mlp.act` | 19 | 8,881,152 | 8,881,152 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 768 | 768 | 37,632 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `blocks.*.*` | 19 | 2,220,288 | 2,220,288 |  |
| 0 (자료 이동) | `aten::view_as_complex` | ComplexFloat | `blocks.*.*.filter` | 19 | 1,204,224 | 0 |  |

합계: (a) 44,104,704 / (b) 1,204,224 / (c) 13,623,552 MACs

</details>

<details><summary><b>GFNet-H-B</b></summary>

| 항목 | 연산 | 출력 dtype | 모듈 (이름 패턴) | 호출 | 원소 수 (FFT는 ΣE) | 셈 값 (MACs) | reduction 입력 원소 수 |
|---|---|---|---|---|---|---|---|
| (a) FFT | `aten::fft_irfft2` | Float | `blocks.*.*.filter` | 36 | 3,499,776 | 32,288,256 |  |
| (a) FFT | `aten::fft_rfft2` | ComplexFloat | `blocks.*.*.filter` | 36 | 3,499,776 | 32,288,256 |  |
| (b) 주파수 영역 곱 | `aten::mul` | ComplexFloat | `blocks.*.*.filter` | 36 | 1,935,360 | 1,935,360 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `(root)` | 1 | 301,056 | 301,056 |  |
| (c) 그 밖의 원소별 | `aten::add` | Float | `blocks.*.*` | 36 | 3,499,776 | 3,499,776 |  |
| (c) 그 밖의 원소별 | `aten::gelu` | Float | `blocks.*.*.mlp.act` | 36 | 13,999,104 | 13,999,104 |  |
| (c) 그 밖의 원소별 | `aten::mean` | Float | `(root)` | 1 | 768 | 768 | 37,632 |
| (c) 그 밖의 원소별 | `aten::mul` | Float | `blocks.*.*` | 36 | 3,499,776 | 3,499,776 |  |
| 0 (자료 이동) | `aten::view_as_complex` | ComplexFloat | `blocks.*.*.filter` | 36 | 1,935,360 | 0 |  |

합계: (a) 64,576,512 / (b) 1,935,360 / (c) 21,300,480 MACs

</details>

## 5. 다른 도구로 교차 확인: `torch.utils.flop_counter.FlopCounterMode`

- torch 2.8.0 내장 도구다. 디스패치 단계에서 연산을 세며, 단위는 **FLOPs(곱셈-덧셈 1회 = 2)**이다. 그래서 2로 나눠 MACs로 비교했다.
- 세는 연산은 `flop_registry`에 등록된 `mm`, `addmm`, `bmm`, `baddbmm`, `convolution`, `_scaled_mm`, scaled-dot-product attention 계열뿐이다. **FFT, 원소별 곱, 정규화 층은 세지 않는다.** DFFormer의 복소 `einsum`은 `bmm`으로 분해되어 shape 기준으로 세어진다.
- DeepSpeed 프로파일러는 이 환경에 설치되어 있지 않아 쓰지 않았다.
- 아래 표의 `(가) − norm − FCM/2` 열은 (가)에서 fvcore가 센 정규화 층 몫과 FlopCounterMode/2를 뺀 값이다(정수 MACs). DFFormer·CDFFormer 외에는 0이다. 두 모델 계열에서 남는 수천 MACs 이하의 차이는 복소 `einsum`을 fvcore는 `np.einsum_path`의 추정값으로, FlopCounterMode는 `bmm` shape로 세서 생긴다.

| 모델 | (가) | FCM FLOPs | FCM FLOPs / 2 | (가)의 norm 몫 | (가) − norm − FCM/2 (MACs) | FCM 연산별 FLOPs (G) |
|---|---|---|---|---|---|---|
| **S18** | | | | | | |
| SPANet-S | 4.627 | 9.212 | 4.606 | 0.021 | 0 | convolution 9.211, addmm 0.001 |
| SPANetV2-S18-pure | 4.192 | 8.323 | 4.162 | 0.031 | 0 | convolution 1.800, mm 6.517, addmm 0.006 |
| ConvFormer-S18 | 3.956 | 7.882 | 3.941 | 0.015 | 0 | convolution 0.766, mm 7.110, addmm 0.006 |
| DFFormer-S18 | 3.800 | 7.601 | 3.800 | 0.000 | -48 | convolution 0.464, mm 7.118, bmm 0.013, addmm 0.006 |
| SPANetV2-S18-hybrid | 4.192 | 8.335 | 4.167 | 0.024 | 0 | convolution 0.916, mm 6.956, bmm 0.457, addmm 0.006 |
| CAFormer-S18 | 4.122 | 8.214 | 4.107 | 0.015 | 0 | convolution 0.641, mm 7.110, bmm 0.457, addmm 0.006 |
| CDFFormer-S18 | 3.885 | 7.770 | 3.885 | 0.000 | -144 | convolution 0.641, mm 7.117, bmm 0.006, addmm 0.006 |
| **S36** | | | | | | |
| SPANet-M | 6.824 | 13.586 | 6.793 | 0.031 | 0 | convolution 13.585, addmm 0.001 |
| SPANetV2-S36-pure | 8.145 | 16.169 | 8.085 | 0.060 | 0 | convolution 3.129, mm 13.035, addmm 0.006 |
| ConvFormer-S36 | 7.670 | 15.279 | 7.640 | 0.030 | 0 | convolution 1.054, mm 14.219, addmm 0.006 |
| DFFormer-S36 | 7.364 | 14.728 | 7.364 | 0.000 | -48 | convolution 0.464, mm 14.232, bmm 0.026, addmm 0.006 |
| SPANetV2-S36-hybrid | 8.179 | 16.262 | 8.131 | 0.048 | 0 | convolution 1.522, mm 13.834, bmm 0.900, addmm 0.006 |
| CAFormer-S36 | 8.002 | 15.943 | 7.972 | 0.030 | 0 | convolution 0.818, mm 14.219, bmm 0.900, addmm 0.006 |
| CDFFormer-S36 | 7.533 | 15.066 | 7.533 | 0.000 | -324 | convolution 0.818, mm 14.231, bmm 0.011, addmm 0.006 |
| **M36** | | | | | | |
| SPANet-B | 11.983 | 23.880 | 11.940 | 0.043 | 0 | convolution 23.879, addmm 0.002 |
| SPANetV2-M36-pure | 13.725 | 27.283 | 13.642 | 0.083 | 0 | convolution 5.220, mm 22.055, addmm 0.007 |
| ConvFormer-M36 | 12.884 | 25.685 | 12.842 | 0.042 | 0 | convolution 1.617, mm 24.060, addmm 0.007 |
| DFFormer-M36 | 12.463 | 24.926 | 12.463 | 0.000 | -2,460 | convolution 0.804, mm 24.079, bmm 0.036, addmm 0.007 |
| SPANetV2-M36-hybrid | 13.676 | 27.216 | 13.608 | 0.069 | 0 | convolution 2.936, mm 23.193, bmm 1.079, addmm 0.007 |
| CAFormer-M36 | 13.282 | 26.481 | 13.241 | 0.042 | 0 | convolution 1.335, mm 24.060, bmm 1.079, addmm 0.007 |
| CDFFormer-M36 | 12.716 | 25.432 | 12.716 | 0.000 | -324 | convolution 1.335, mm 24.077, bmm 0.013, addmm 0.007 |
| **B36** | | | | | | |
| SPANetV2-B36-pure | 24.251 | 48.281 | 24.141 | 0.111 | 0 | convolution 9.061, mm 39.210, addmm 0.011 |
| ConvFormer-B36 | 22.685 | 45.259 | 22.629 | 0.055 | 0 | convolution 2.474, mm 42.774, addmm 0.011 |
| DFFormer-B36 | 22.128 | 44.256 | 22.128 | 0.000 | -1,830 | convolution 1.390, mm 42.807, bmm 0.048, addmm 0.011 |
| SPANetV2-B36-hybrid | 23.934 | 47.685 | 23.843 | 0.091 | 0 | convolution 5.004, mm 41.233, bmm 1.438, addmm 0.011 |
| CAFormer-B36 | 23.216 | 46.321 | 23.160 | 0.055 | 0 | convolution 2.098, mm 42.774, bmm 1.438, addmm 0.011 |
| CDFFormer-B36 | 22.465 | 44.929 | 22.465 | 0.000 | 18 | convolution 2.098, mm 42.803, bmm 0.018, addmm 0.011 |
| **GFNet** | | | | | | |
| GFNet-H-S | 4.604 | 9.163 | 4.582 | 0.022 | 0 | convolution 0.376, addmm 8.788 |
| GFNet-H-B | 8.547 | 17.024 | 8.512 | 0.035 | 0 | convolution 0.376, addmm 16.649 |

## 6. (선택) GFNet-H

GFNet 공식 저장소(`raoyongming/GFNet`)의 `GFNetPyramid`를 `infer.py` / `main_gfnet.py`의 `gfnet-h-s`, `gfnet-h-b` 인자 그대로 만들었다. 공식 저장소에는 `gfnet-h-ti`, `gfnet-h-s`, `gfnet-h-b`만 있고 **GFNet-H-S++, GFNet-H-L++는 없어 재지 못했다.** 저장소 README의 보고값은 H-S 32M / 4.5G, H-B 54M / 8.4G이다.

| 모델 | Params (M)<br>학위논문 표 → 재측정 | MACs (G)<br>학위논문 표 → 재측정(합계) | (가) fvcore 기본 | (나) | +(a) | +(b) | +(c) | 합계 |
|---|---|---|---|---|---|---|---|---|
| GFNet-H-S | 32 → 32 (±0) | 4.6 → 4.7 (+0.1) | 4.604 | - | 4.648 | 4.649 | 4.663 | 4.663 |
| GFNet-H-B | 54 → 54 (±0) | 8.6 → 8.6 (±0) | 8.547 | - | 8.612 | 8.614 | 8.635 | 8.635 |

## 논문 값과의 차이

**MACs.** (가)와 논문 값의 차이는 모든 모델에서 0.1G 이하다. 차이가 큰 순서로 다섯 개:

- ConvFormer-B36: (가) 22.685 − 논문 22.6 = +0.085
- ConvFormer-M36: (가) 12.884 − 논문 12.8 = +0.084
- CAFormer-M36: (가) 13.282 − 논문 13.2 = +0.082
- SPANetV2-S36-hybrid: (가) 8.179 − 논문 8.1 = +0.079
- SPANetV2-M36-hybrid: (가) 13.676 − 논문 13.6 = +0.076

(나)는 다음 모델에서 논문 값과 0.1G 넘게 다르다.

- SPANetV2-S36-hybrid: (나) 8.207 − 논문 8.1 = +0.107
- SPANetV2-M36-hybrid: (나) 13.722 − 논문 13.6 = +0.122

**Params.** 1M 단위로 반올림한 값이 논문 값과 다른 모델:

- SPANetV2-M36-pure: 62.935 M (논문 61 M)
- SPANetV2-B36-pure: 109.558 M (논문 100 M)
- SPANetV2-B36-hybrid: 101.803 M (논문 100 M)

### 설정 확인

위 모델들에 대해, 코드가 실제로 만든 설정을 모델 객체에서 읽어 출력했다([config_check.py](config_check.py), [results/config_check.txt](results/config_check.txt)). 채널 수, 블록 수, token mixer, ExSPAM head 수와 커널 집합, SRF 필터 크기, attention head 수, 구성 요소별 Params이다. 같은 폭의 ConvFormer·CAFormer도 함께 실었다. 학위논문 원고가 이 폴더에 없어서, 원고의 설정표와의 대조는 하지 않았다.

SPANetV2 공통(코드 기준): ExSPAM `num_heads` 4, head별 depthwise 커널 3·5·7·9, SRF 필터 `(C/4, H, W//2+1)`(224 입력 기준 56×29, 28×15, 14×8, 7×4), proj 확장비 2, MLP 비 4(StarReLU), head `MlpHead`(비 4). hybrid는 stage 2 `MixAttention`(head_dim 32, 7×7 local conv), stage 3 `Attention`(head_dim 32). SPANet(비교용): stage마다 SPF 3개(λ 0.7, 0.8, 0.9), 1×7·7×1 depthwise conv, 반경은 코드 인자 `radius=[2, 2, 1, 1]`.

```
== SPANetV2-S36-hybrid: 40.669 M params
  stage 0: depth 3, dim 64, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(16, 56, 29), (16, 56, 29), (16, 56, 29), (16, 56, 29)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 64->256->64 (StarReLU)
  stage 1: depth 12, dim 128, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(32, 28, 15), (32, 28, 15), (32, 28, 15), (32, 28, 15)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 128->512->128 (StarReLU)
  stage 2: depth 18, dim 320, token_mixer MixAttention, num_heads 10, head_dim 32, local_conv kernel (7, 7), mlp 320->1280->320 (StarReLU)
  stage 3: depth 3, dim 512, token_mixer Attention, num_heads 16, head_dim 32, mlp 512->2048->512 (StarReLU)
  head: MlpHead 512->2048->1000
    downsample / stem                                1.928 M
    block norms                                      0.018 M
    token_mixer: spectral filters (SRF params)       0.957 M
    token_mixer: other                              11.939 M
    mlp                                             22.708 M
    res/layer scale                                  0.015 M
    final norm + head                                3.105 M


== SPANetV2-M36-pure: 62.935 M params
  stage 0: depth 3, dim 96, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(24, 56, 29), (24, 56, 29), (24, 56, 29), (24, 56, 29)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 96->384->96 (StarReLU)
  stage 1: depth 12, dim 192, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(48, 28, 15), (48, 28, 15), (48, 28, 15), (48, 28, 15)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 192->768->192 (StarReLU)
  stage 2: depth 18, dim 384, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(96, 14, 8), (96, 14, 8), (96, 14, 8), (96, 14, 8)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 384->1536->384 (StarReLU)
  stage 3: depth 3, dim 576, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(144, 7, 4), (144, 7, 4), (144, 7, 4), (144, 7, 4)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 576->2304->576 (StarReLU)
  head: MlpHead 576->2304->1000
    downsample / stem                                2.836 M
    block norms                                      0.022 M
    token_mixer: spectral filters (SRF params)       2.258 M
    token_mixer: other                              21.204 M
    mlp                                             32.956 M
    res/layer scale                                  0.017 M
    final norm + head                                3.640 M


== SPANetV2-M36-hybrid: 58.335 M params
  stage 0: depth 3, dim 96, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(24, 56, 29), (24, 56, 29), (24, 56, 29), (24, 56, 29)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 96->384->96 (StarReLU)
  stage 1: depth 12, dim 192, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(48, 28, 15), (48, 28, 15), (48, 28, 15), (48, 28, 15)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 192->768->192 (StarReLU)
  stage 2: depth 18, dim 384, token_mixer MixAttention, num_heads 12, head_dim 32, local_conv kernel (7, 7), mlp 384->1536->384 (StarReLU)
  stage 3: depth 3, dim 576, token_mixer Attention, num_heads 18, head_dim 32, mlp 576->2304->576 (StarReLU)
  head: MlpHead 576->2304->1000
    downsample / stem                                2.836 M
    block norms                                      0.022 M
    token_mixer: spectral filters (SRF params)       1.435 M
    token_mixer: other                              17.427 M
    mlp                                             32.956 M
    res/layer scale                                  0.017 M
    final norm + head                                3.640 M


== SPANetV2-B36-pure: 109.558 M params
  stage 0: depth 3, dim 128, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(32, 56, 29), (32, 56, 29), (32, 56, 29), (32, 56, 29)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 128->512->128 (StarReLU)
  stage 1: depth 12, dim 256, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(64, 28, 15), (64, 28, 15), (64, 28, 15), (64, 28, 15)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 256->1024->256 (StarReLU)
  stage 2: depth 18, dim 512, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(128, 14, 8), (128, 14, 8), (128, 14, 8), (128, 14, 8)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 512->2048->512 (StarReLU)
  stage 3: depth 3, dim 768, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(192, 7, 4), (192, 7, 4), (192, 7, 4), (192, 7, 4)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 768->3072->768 (StarReLU)
  head: MlpHead 768->3072->1000
    downsample / stem                                5.035 M
    block norms                                      0.030 M
    token_mixer: spectral filters (SRF params)       3.011 M
    token_mixer: other                              37.427 M
    mlp                                             58.589 M
    res/layer scale                                  0.023 M
    final norm + head                                5.443 M


== SPANetV2-B36-hybrid: 101.803 M params
  stage 0: depth 3, dim 128, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(32, 56, 29), (32, 56, 29), (32, 56, 29), (32, 56, 29)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 128->512->128 (StarReLU)
  stage 1: depth 12, dim 256, token_mixer ExSPAM, num_heads 4, local_conv kernels [(3, 3), (5, 5), (7, 7), (9, 9)], filters (C/heads, H, W//2+1) [(64, 28, 15), (64, 28, 15), (64, 28, 15), (64, 28, 15)], proj ['Conv2d', 'GroupNorm', 'GELU', 'Conv2d'] expand 2, mlp 256->1024->256 (StarReLU)
  stage 2: depth 18, dim 512, token_mixer MixAttention, num_heads 16, head_dim 32, local_conv kernel (7, 7), mlp 512->2048->512 (StarReLU)
  stage 3: depth 3, dim 768, token_mixer Attention, num_heads 24, head_dim 32, mlp 768->3072->768 (StarReLU)
  head: MlpHead 768->3072->1000
    downsample / stem                                5.035 M
    block norms                                      0.030 M
    token_mixer: spectral filters (SRF params)       1.914 M
    token_mixer: other                              30.768 M
    mlp                                             58.589 M
    res/layer scale                                  0.023 M
    final norm + head                                5.443 M


== ConvFormer-M36: 57.052 M params
    downsample / stem                                2.836 M
    block norms                                      0.022 M
    token_mixer: other                              17.579 M
    mlp                                             32.956 M
    res/layer scale                                  0.017 M
    final norm + head                                3.640 M


== ConvFormer-B36: 99.883 M params
    downsample / stem                                5.035 M
    block norms                                      0.030 M
    token_mixer: other                              30.762 M
    mlp                                             58.589 M
    res/layer scale                                  0.023 M
    final norm + head                                5.443 M


== CAFormer-M36: 56.205 M params
    downsample / stem                                2.836 M
    block norms                                      0.022 M
    token_mixer: other                              16.732 M
    mlp                                             32.956 M
    res/layer scale                                  0.017 M
    final norm + head                                3.640 M


== CAFormer-B36: 98.754 M params
    downsample / stem                                5.035 M
    block norms                                      0.030 M
    token_mixer: other                              29.633 M
    mlp                                             58.589 M
    res/layer scale                                  0.023 M
    final norm + head                                5.443 M
```

## 파일

| 파일 | 내용 |
|---|---|
| [setup_third_party.sh](setup_third_party.sh) | 비교 저장소를 측정 커밋으로 `third_party/`에 clone 하고 GFNet import 수정 적용 |
| [zoo.py](zoo.py) | 모델 생성. 모델 파일을 경로로 불러온다(`models` 패키지 이름 충돌 방지) |
| [measure.py](measure.py) | 1·2·3·4·5단계 측정. `results/results.json`, `results/env.json` 생성 |
| [config_check.py](config_check.py) | 설정과 구성 요소별 Params 출력 → `results/config_check.txt` |
| [report.py](report.py) | 이 README 생성 |
| [patches/GFNet_numpy_import.patch](patches/GFNet_numpy_import.patch) | GFNet import 경로 수정 |
| [results/](results/) | 측정 원값(`results.json`), 환경(`env.json`), 실행 로그(`measure_log.txt`), 설정 출력 |

## 재현

```bash
cd flops_benchmark
bash setup_third_party.sh
conda activate spanetv2   # torch 2.8.0, fvcore 0.1.5.post20221221, timm 0.6.13
python measure.py > results/measure_log.txt 2>&1
python config_check.py > results/config_check.txt
python report.py
```
