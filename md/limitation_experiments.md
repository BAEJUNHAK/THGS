# THGS Limitation 분석 실험 계획

> THGS는 training-free라 "학습으로 오류를 보상할 기회가 없다". 따라서 한계점은 **각 단계의 결정적 오류가 다음 단계로 전파되며 ceiling을 만드는 구조**에서 나온다. 각 단계를 가설 → 관측 가능한 지표 → 실험 설계 순으로 정리한다.

---

## 0. 가설 지도

| # | 한계 원인 | 병목 단계 | Observable Symptom |
|---|---|---|---|
| H1 | 2DGS primitive 자체의 geometric 불일치 | Stage 0 | 얇은/투명 객체 IoU 급락 |
| H2 | SAM multi-scale mask의 granularity gap | Stage α | part-level 쿼리 recall 낮음 |
| H3 | CLIP의 object-centric bias (224 resize) | Stage α | 작은/긴 객체 유사도 왜곡 |
| H4 | FRNN K-NN의 K 고정 | Stage β-1 | 밀도 불균일 영역 over/under-connected |
| H5 | 랜덤 encoding trick의 stochasticity | Stage β-2, γ-1 | 실행마다 결과 분산 |
| H6 | τ=0.85 threshold의 hard cutoff | Stage β-2 | occluded/edge Gaussian 손실 |
| H7 | depth_weight `1/(1+ReLU(d-5))`의 scene-scale 의존성 | Stage β-2 | scene normalization 깨진 데이터 실패 |
| H8 | cut-pursuit regularization의 hyperparameter 민감도 | Stage β-3 | 장면마다 튜닝 필요, overfit |
| H9 | 3회 고정 merging + `[0.9,0.7,0.7]` schedule | Stage γ | 장면 복잡도에 non-adaptive |
| H10 | CLIP feature가 bag-of-masks weighted avg | Stage γ-5 | compositional/relational 쿼리 실패 |
| H11 | `level=[2,3]` + `topk=3` 고정 쿼리 | Stage δ | single-object vs group 쿼리 trade-off |
| H12 | 평가 이미지가 train set에서 옴 (llffhold=8 미사용) | Stage δ | novel-view generalization 미측정 |

---

## Exp 1. Upper-bound Oracle 실험 — "각 단계가 완벽했다면?"

각 단계를 oracle로 대체해 ceiling을 측정. 성능 상승이 큰 단계 = 그 단계의 limitation 기여도.

### 1-A. SAM-oracle
- `language_features/*_s.npy`를 LERF-OVS GT polygon → 4-scale mask로 덮어쓰기.
- `_f.npy`는 GT crop을 CLIP ViT-B-16으로 재인코딩.
- 이후 Stage β-2, β-3, γ, δ 그대로 실행.

### 1-B. CLIP-oracle
- `_f.npy`를 각 mask의 **GT category one-hot** 또는 GT category 텍스트의 CLIP 임베딩으로 교체.

### 1-C. Superpoint-oracle (가능한 데이터셋 한정)
- ScanNet/Replica처럼 GT 3D instance mask가 있을 때만 가능.
- `nag-l1.pt`를 GT로 치환 후 Stage γ 이후만 실행.

### 1-D. No-merge baseline
- `merge_proj.py` 생략, level-1만 사용.
- `test_lerf.py`에서 `level=[1]`로 쿼리.

**지표**: mIoU, Precision, Recall. "GT-SAM + 기존 이후"가 얼마나 올라가는지, "GT-CLIP + 기존 이후"가 얼마나 올라가는지 gap 비교.

**소요**: 1-A/1-B는 Stage α부터 재실행 → scene당 ~1.5h.

---

## Exp 2. SAM granularity sensitivity (H2, H3)

`configs/lerf.yml`의 `graph_weight.level: 1` (s-scale) 고정을 바꿈.

### 조건
- `level ∈ {0=default, 1=s, 2=m, 3=l}` 단일
- `level ∈ {[0,1,2], [1,2,3], [0,1,2,3]}` 멀티 (nargs='+' 지원)

### 평가 분리
- LERF-OVS prompt를 수동 분류: **object-level** vs **part-level**.
- 각 prompt 카테고리별 mIoU 분리 측정.

**기대**: "잘게 쪼갠 level은 part에 좋고 object에 나쁘다"는 trade-off 정량화.

**소요**: graph_weight + merge_proj + test 재실행, scene당 ~40m × 조건 수.

---

## Exp 3. 랜덤 encoding stochasticity (H5)

`graph_weight.py:40`의 `torch.normal(mean=0, std=1, size=(seg_num, 20))`와 `merge_proj.py:159`의 20-D 랜덤 인코딩의 seed 변화 측정.

### 3-A. Seed variance
- seed ∈ {42, 123, 456, 789, 2026} 5회 실행.
- 5회의 최종 mIoU 평균/표준편차.

### 3-B. Latent dim sweep
- `dim_latent ∈ {8, 20, 50, 128}` 비교. 20이 너무 낮아서 마스크 id 고유성 보장이 약한지 확인.

**의미**: 표준편차가 크면 "random label-passing"이 noisy proxy라는 증거.

**소요**: graph_weight + merge_proj + test, scene당 ~40m × 5회.

---

## Exp 4. τ hard-cutoff 영향 (H6)

`graph_weight.py:54` `sim_filter = sim_max > tau`, `tau=0.85`는 가장 critical한 hyperparameter 중 하나.

### 4-A. τ sweep
- `τ ∈ {0.5, 0.7, 0.85, 0.95}` 4조건.
- 각 조건에서 **버려지는 가우시안 비율** `(~sim_filter).float().mean()`, 최종 mIoU 로깅.

### 4-B. Soft-weight variant
- Hard cutoff 대신 `weight = sigmoid((sim_max - τ) / T)`로 smooth 가중치.
- T ∈ {0.05, 0.1, 0.2}.

**기대**: U자 곡선이 나오면 hard cutoff 한계 입증. Soft variant가 hard보다 좋으면 structural fix 제안 가능.

**소요**: graph_weight + β-3 + γ + δ, scene당 ~45m × 조건 수.

---

## Exp 5. depth_weight scene-scale 의존성 (H7)

`graph_weight.py:23`의 `f(x) = 1/(1+ReLU(x-5))`의 상수 `5`는 scene depth 분포 가정.

### 5-A. Scene-adaptive depth constant
- scene별 median depth `d_med` 측정 (학습 view의 렌더링 depth에서).
- `f(x) = 1/(1+ReLU(x - d_med))`로 교체.

### 5-B. Depth weight 제거
- `depth_weight = 1` 고정 baseline.

### 5-C. Sigmoid 대체
- `f(x) = 1/(1+exp((x-d_med)/s))`, s ∈ {1, 3, 5}.

**지표**: scene별 mIoU 분산. `waldo_kitchen`처럼 저자 결과에서 상대적으로 낮은 scene이 adaptive로 개선되는지.

**소요**: graph_weight + β-3 + γ + δ, scene당 ~45m × 조건 수.

---

## Exp 6. cut-pursuit hyperparameter sweep (H8)

`configs/lerf.yml`의 `pcp_regularization=0.1`, `pcp_spatial_weight=0.1`가 슈퍼포인트 개수를 지배.

### 6-A. 2D grid
- `reg ∈ {0.03, 0.1, 0.3, 1.0}` × `spatial ∈ {0.03, 0.1, 0.3, 1.0}` = 16 runs.
- 로깅: level-1 superpoint 수 `S`, 최종 mIoU.

### 6-B. Per-scene optimum gap
- scene별 best config vs 저자 default의 mIoU gap 측정.
- gap이 크면 "hyperparameter overfit" 증거.

### 6-C. S vs mIoU scatter
- 16 run 결과를 (S, mIoU) scatter plot.
- scene별 peak S가 다르면 adaptive 필요.

**소요**: β-3 + γ + δ, scene당 ~30m × 16 = 8h/scene (parallelizable).

---

## Exp 7. 계층 schedule 실험 (H9)

`merge_proj.py`의 `thres_connect = [0.9, 0.7, 0.7]` 3-step 고정.

### 조건
- 1-step: `[0.7]`
- 2-step: `[0.8, 0.7]`
- 3-step (기본): `[0.9, 0.7, 0.7]`
- 4-step: `[0.9, 0.8, 0.7, 0.6]`
- 역순: `[0.6, 0.7, 0.8]` — regularity 증가가 맞는지 검증

**지표**: step별 mIoU, 각 step 후 superpoint 수 변화 곡선.

**관찰**: step을 늘려도 saturate하면 "hierarchy의 semantic bottleneck은 CLIP" (H10과 연결).

**소요**: merge_proj + test, scene당 ~15m × 5 = 1.25h/scene.

---

## Exp 8. Feature aggregation 방식 (H10)

`merge_proj.py`의 `feat_assign=1` (`proj_gaussian_features`) vs `feat_assign=2` (`proj_gaussian_features_x`).

### 8-A. Swap 실험
- LERF(default=2)에 `feat_assign=1` 적용.
- 3DOVS(default=1)에 `feat_assign=2` 적용.
- 기존 default가 정말 dataset-specific optimum인지 검증.

### 8-B. 새로운 aggregation variant
- `max-pool`: 슈퍼포인트 내 가우시안 CLIP feature의 max.
- `attention`: `softmax(weight) @ feature`로 soft argmax.
- `median`: outlier에 robust.
- 구현 위치: `merge_proj.py`에 새 함수 추가 + `feat_assign=3,4,5` 케이스.

**소요**: merge_proj + test, scene당 ~15m × 조건 수.

---

## Exp 9. 쿼리 시 level/topk 영향 (H11)

`test_lerf.py:57`의 `snag.get_related_gaussian(sims, topk=3, level=[2,3])` 고정값.

### 9-A. Grid sweep
- `level ∈ {[1], [2], [3], [2,3], [1,2,3]}` × `topk ∈ {1, 3, 5, 10}` = 20 조합.

### 9-B. Prompt-specific optimum
- Object prompt vs part prompt 각각의 optimal (level, topk) 탐색.
- 둘이 다르면 "query-dependent hierarchy selection" 필요성 증명.

**의미**: `sai_nag.pt` 캐시된 상태에서 **test만 재실행하면 됨**. 거의 공짜 실험.

**소요**: test only, scene당 ~3m × 20 = 1h/scene.

---

## Exp 10. View sparsity 영향 (H1, H5)

Training view 개수 감소 시 Stage β-2의 multi-view contrastive signal이 약해지는지.

### 조건
- `scene/dataset_readers.py`에서 train 카메라 subsample.
- view 비율 ∈ {100%, 50%, 25%, 12.5%} (균등 간격).

### 측정
- 각 조건에서 full pipeline 재실행 → mIoU.
- view 수 대비 mIoU 곡선.

**의미**: few-view 시나리오(실세계 캡처)에서의 실패 모드 정량화.

**소요**: 전체 pipeline, scene당 ~2h × 4 = 8h/scene.

---

## Exp 11. Prompt 난이도 taxonomy 분석

**핵심 post-hoc 분석**: 기존 실행 결과만으로 가능.

### 분류 체계
LERF-OVS의 모든 prompt를 수동 분류:
- **Object**: "teapot", "apple", "bag of cookies"
- **Part**: "teapot handle", "chopstick tip"
- **Material**: "wooden", "ceramic"
- **Color+Object**: "red cup", "green apple"
- **Spatial relation**: "cup on the left", "book behind laptop"
- **Functional**: "something to eat", "place to sit"

### 분석
- 카테고리별 mIoU 분해.
- `eval_seg.py`를 수정해 prompt별 metric을 저장 후 후처리.

**기대**: overall mIoU 뒤에 카테고리별 큰 편차. 특히 relational/functional에서 CLIP bag-of-masks가 실패 → H10의 정성 증거.

**소요**: 기존 결과 post-hoc 분석, **0h**.

---

## Exp 12. Error localization heatmap

FP/FN 픽셀을 역투영해 3D 가우시안에 attribution → "어디서 틀렸는가"를 3D로 시각화.

### 절차
1. `test_lerf.py`의 `embd_sim` 렌더링에서 각 픽셀의 contributing Gaussian 기록 (render_point weight 재활용).
2. FP 픽셀 → contributing Gaussian → 해당 superpoint (level 2, 3) 카운트.
3. Top-K failure superpoint를 `<scene>saifine.ply`에 색칠해 시각화.

### 분류
- Boundary failure: 객체 경계 슈퍼포인트 → SAM mask edge 문제.
- Interior failure: 객체 내부 슈퍼포인트 → CLIP feature 문제.
- Background leakage: 배경으로 판정됐어야 할 영역 → τ/threshold 문제.

**의미**: 정량 지표로 보이지 않는 failure mode의 정성 진단.

**소요**: 결과 후처리 + 시각화 스크립트 작성 ~4h.

---

## 실험 우선순위

가장 많은 정보를 빠르게 주는 순서:

| 순위 | 실험 | 이유 |
|---|---|---|
| 1 | Exp 11 (prompt taxonomy) | 기존 결과 분석만, cost 0. 후속 실험 범위 결정. |
| 2 | Exp 9 (level/topk sweep) | test만 재실행, 캐시 활용. |
| 3 | Exp 1-A (SAM-oracle) | "CLIP이 병목인가 SAM이 병목인가" 핵심 분기. |
| 4 | Exp 4 (τ sweep) | 가장 critical한 단일 hyperparam의 민감도. |
| 5 | Exp 6 (pcp sweep) | config가 LERF에 overfit인지. |
| 6 | Exp 3 (seed 분산) | method 자체의 reproducibility ceiling. |
| 7 | Exp 2 (SAM granularity) | object vs part trade-off 정량화. |
| 8 | Exp 12 (error localization) | 정성 failure case. |
| 9 | Exp 7, 8, 5, 10 | 세부 분석. |

---

## 재실행 범위별 소요 시간 (1 scene 기준, T4 GPU 가정)

| Exp | 재실행 단계 | 1 config당 시간 |
|---|---|---|
| 1-A (SAM oracle) | Stage α부터 전부 | ~1.5h |
| 1-B (CLIP oracle) | Stage γ부터 | ~25m |
| 2 (level 변경) | graph_weight + merge_proj + test | ~40m |
| 3 (seed) | graph_weight + merge_proj + test | ~40m |
| 4 (τ) | graph_weight + β-3 + merge_proj + test | ~45m |
| 5 (depth) | graph_weight + β-3 + merge_proj + test | ~45m |
| 6 (pcp) | β-3 + merge_proj + test | ~30m |
| 7 (schedule) | merge_proj + test | ~15m |
| 8 (feat_assign) | merge_proj + test | ~15m |
| 9 (level/topk) | test only | ~3m |
| 10 (view sparsity) | 전체 | ~2h |
| 11 (taxonomy) | 후처리 | 0m |
| 12 (localization) | 후처리 + 시각화 | ~4h |

**주의**: 각 실험은 Drive 캐시를 활용해 **공통 부분은 재사용**. 예: Exp 7, 8, 9는 동일한 `nag-l1.pt`를 공유.

---

## 실험 로깅 템플릿

각 실험 run마다 저장할 metadata:

```yaml
exp_id: exp4_tau_sweep_figurines
scene: figurines
config:
  tau: 0.7
  other_params: default
metrics:
  mIoU: 0.xxx
  mAcc: 0.xxx
  per_prompt:
    teapot: {IoU: ..., P: ..., R: ...}
    ...
pipeline_stats:
  superpoint_count_l1: N
  superpoint_count_l2: N
  superpoint_count_l3: N
  dropped_gaussian_ratio: 0.xx  # τ로 버려진 비율
runtime:
  graph_weight: Xs
  partition: Xs
  merge_proj: Xs
seed: 42
git_sha: xxxxxx
```

저장 위치 제안: `output/experiments/<exp_id>/metadata.json` + `predictions/`.

---

## 예상 결론 프레임

> THGS의 성능 한계는 어느 단일 단계에서 오지 않고 **세 가지가 중첩**된다:
> (i) SAM의 multi-scale mask가 part-level 쿼리엔 coarse하고 object-level엔 fragmented하여 Stage α가 object-part trade-off를 강요,
> (ii) Stage β-2의 contrastive signal이 `τ`, `depth_weight`, view sparsity에 대해 **non-adaptive**여서 장면별 튜닝 없이 5–10 mIoU 손실,
> (iii) 최종 쿼리가 CLIP의 bag-of-masks-weighted-average에 의존하므로 **relational/compositional prompt에서 구조적으로 실패**.
> 이 중 training-free 제약 내에서 해결 가능한 것은 (ii)뿐. (i), (iii)은 pretrained 모델의 교체 또는 학습 도입이 필요.

이 결론이 실제로 뒷받침된다면 후속 연구의 motivation이 자연스럽게 도출된다.
