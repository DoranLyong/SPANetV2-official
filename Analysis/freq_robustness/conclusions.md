## A
저역 통과 11개 크기(S = 20–220)의 평균 top-1은 SPANetV2-S18-hybrid 69.25(9개 중 2위), SPANetV2-S18-pure 68.19(5위)이고, 가장 높은 값은 CAFormer-S18 69.45, 가장 낮은 값은 Swin-T 65.13이다. 고역 통과 6개 크기(S = 170–220)의 평균 top-1은 SPANetV2-S18-hybrid 30.71(2위), SPANetV2-S18-pure 21.91(8위)이고, 가장 높은 값은 ConvNeXt-T 31.71, 가장 낮은 값은 DeiT-S 20.00이다. 고역 통과 S = 220에서 SPANetV2-S18-pure는 59.54로 9개 중 9위, SPANetV2-S18-hybrid는 74.14로 1위이다.

## B
5,000장 Fourier heat map에서 저역(ρ ≤ 0.25) 평균 오류율은 SPANetV2-S18-hybrid 23.10%(9개 중 낮은 순 1위), SPANetV2-S18-pure 25.67%(3위)이고, 가장 높은 값은 ConvNeXt-T 31.92%이다. 고역(ρ > 0.5, stride 8 격자) 평균 오류율은 SPANetV2-S18-hybrid 21.57%(1위), SPANetV2-S18-pure 23.86%(4위)이고, 가장 높은 값은 Swin-T 32.75%이다. 같은 5,000장의 잡음 없는 오류율은 16.72–20.36%이다(SPANetV2-S18-hybrid 16.72%, SPANetV2-S18-pure 17.42%).

## C
제한 없는 L2 PGD ε = 1.0에서 top-1은 SPANetV2-S18-hybrid 25.10%(9개 중 1위), SPANetV2-S18-pure 11.02%(7위)이고, 가장 낮은 값은 ConvNeXt-T 3.98%이다. ε = 1.0에서 저역만 제한한 공격(M_l^56)은 SPANetV2-S18-hybrid 47.14%(2위; 1위 CAFormer-S18 48.56%), SPANetV2-S18-pure 39.12%(6위)이고, 고역만 제한한 공격은 SPANetV2-S18-hybrid 29.38%(1위), SPANetV2-S18-pure 14.82%(7위)이다. 마지막 [0,1] clamp 뒤 δ 에너지 가운데 대역 밖 비율의 평균은 ε = 1.0에서 모델·변형마다 0.08–0.28%이다.

## D1
UperNet 단일 스케일의 mIoU / Boundary IoU는 S18 짝에서 SPANetV2-S18-pure 48.72 / 35.59, SPANetV2-S18-hybrid 49.13 / 36.22, ConvNeXt-T 45.99 / 33.35, Swin-T 44.51 / 31.80이고, S36 짝에서 SPANetV2-S36-pure 49.76 / 37.26, SPANetV2-S36-hybrid 51.63 / 38.29, ConvNeXt-S 48.74 / 35.85, Swin-S 47.64 / 35.18이다. 여덟 모델의 mIoU 순위와 Boundary IoU 순위는 같다.

## D2
Semantic FPN의 mIoU / Boundary IoU는 SPANetV2-S18-pure 46.68 / 33.27, SPANetV2-S18-hybrid 47.83 / 34.64, PoolFormer-S24 40.35 / 27.61, SPANetV2-S36-pure 47.90 / 34.77, SPANetV2-S36-hybrid 48.57 / 34.64, PoolFormer-S36 41.80 / 28.94이다. 같은 크기 짝(S18 ↔ PoolFormer-S24, S36 ↔ PoolFormer-S36)에서 SPANetV2와 PoolFormer의 차이는 mIoU 6.09–7.47, Boundary IoU 5.66–7.03이다.

## D3
COCO val2017의 mask AP / Boundary AP는 Mask R-CNN 3×(S18 짝)에서 SPANetV2-S18-pure 42.9 / 28.27, SPANetV2-S18-hybrid 44.3 / 29.23, ConvNeXt-T 41.7 / 26.74, Swin-T 41.7 / 26.84이다. Cascade Mask R-CNN 3×에서는 S18 짝 SPANetV2-S18-pure 44.7 / 30.07, SPANetV2-S18-hybrid 45.7 / 30.64, ConvNeXt-T 43.6 / 28.92, Swin-T 43.7 / 28.97, S36 짝 SPANetV2-S36-pure 45.9 / 30.86, SPANetV2-S36-hybrid 46.4 / 31.01, ConvNeXt-S 45.0 / 30.03, Swin-S 45.0 / 30.21이다.
