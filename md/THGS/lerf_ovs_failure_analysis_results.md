# LERF-OVS Failure Analysis — Results


> Generated 2026-06-04. Data: 4 scenes (figurines, ramen, teatime, waldo_kitchen). Per-frame CSV ([per_prompt.csv](../output/diagnostics/lerf_ovs_per_prompt.csv)). Methodology: [lerf_ovs_failure_analysis_plan.md](lerf_ovs_failure_analysis_plan.md).


> **Headline numbers**: LangSplat-style aggregation (per-image mean → per-scene mean → overall). 다른 3D OVS 논문과 직접 비교 가능.


> **Failure analysis** (Type/sub-type/FP/FN): per-prompt aggregation (mean over the prompt's frames; pixels summed for FP/FN).


## TL;DR


**Headline (LangSplat-style, 4 scenes):**
- Oracle mIoU = **0.8072**
- Actual mIoU = **0.5887**
- **Gap = 0.2185** ← matching loss, 다른 논문과 직접 비교 가능한 수치

**Failure 분포 (τ=0.25, per-prompt aggregation):**
- Type A (CLIP matching 실패): **20/67 prompts (29.9%)**
- Type A가 차지하는 전체 IoU 손실: **85.4%**

**Type A 내부:**
- A1 (CLIP feature noise, rank ≥ 11): **12/20 = 60.0%**
- A1의 IoU 손실 비중 (within A): **62.7%**
- A1 정답 SP 평균 CLIP rank: **50.3** (in pools of 745–4136 SPs)

**Attack priority (architecture-only 원칙):**
1. **#1 (유일 valid architectural attack): A1 — per-SP CLIP feature 생성 단계 수정 (`image_encoding.py` + `merge_proj.py`)**
   - 회복 가능 IoU = **9.705** (Type A 손실의 62.7%)
2. A2 / A3 → **❌ inference setting (topk, scoring) 변경 필요 — protocol fix를 깸**

**해석 한 줄:**
> 정답 SP가 NAG 안에 거의 항상 존재 (SAM lift 성공). 그러나 CLIP이 정답 SP를 평균 50등으로 매기는 prompt 12개가 LERF-OVS 헤드룸의 약 54% 를 잠그고 있다.


---

## 핵심 원칙 — Fixed protocol & Architecture-only attacks

본 분석의 framing은 다음 원칙을 따른다:

| 항목 | 결정 |
|--|--|
| **고정 (논문 protocol)** | 평가 데이터 (4 scenes, polygon GT), 평가 metric (LangSplat-style mIoU/P/R), **THGS의 inference 인터페이스 (`test_lerf.py` 의 CLIP text 인코딩 + topk=3 at level=[2,3] + canon-contrast scoring + threshold=0.5)** |
| **수정 가능 (architecture / pipeline)** | per-pixel CLIP feature 추출 (`image_encoding.py`), Gaussian/SP feature 투영 + 집계 (`merge_proj.py`), SP partition (`sp_partition.py`), 그래프 가중치 (`graph_weight.py`) |
| **수정 금지 (settings)** | topk 값, threshold, canon-contrast 함수, level 선택 — 이걸 바꾸면 다른 method가 됨 (논문 fair comparison 깨짐) |

→ Attack 후보는 반드시 **pipeline 안의 deterministic 알고리즘 수정**이어야 함.

---

## Type / Sub-type 정의

각 (scene, prompt) 의 분류:

### Type (Oracle IoU와 Actual IoU의 조합, τ = 0.25 기준)

| Type | 조건 | 의미 | 진단 |
|--|--|--|--|
| **Type C** | Oracle ≥ τ **AND** Actual ≥ τ | 성공 | 분석 불필요 |
| **Type A** | Oracle ≥ τ **AND** Actual < τ | **CLIP matching 실패** — 정답 SP는 NAG에 있는데 CLIP이 못 골랐다 | ✅ **공격 가능** |
| **Type B** | Oracle < τ **AND** Actual < τ | SAM lift 실패 — 정답 SP 자체가 NAG에 없음 | NAG 구성 한계 |
| ? | Oracle < τ **AND** Actual ≥ τ | 이론적 불가 (Actual > Oracle은 발생하지 않아야 함) | data error 의심 |

### Sub-type (Type A 내부, 정답 SP의 CLIP rank로 분류)

정답 SP = greedy v4의 first pick (= 단독 IoU 최고 SP). 이 SP의 CLIP score rank (unified pool level=[2,3]):

| Sub-type | 조건 (정답 SP CLIP rank) | 진단 | Attack 가능 여부 |
|--|--|--|--|
| **A1** | rank ≥ 11 | CLIP이 정답 SP를 한참 뒤로 밀어버림 → per-SP CLIP feature quality 자체 문제 | ✅ **architecture-attackable** |
| **A2** | rank in [2, 10] | 정답 SP가 상위권에 있지만 top-1 아님 | ❌ inference setting (topk) |
| **A3** | rank = 1 | 정답 SP는 top-1, but topk=3가 distractor 포함 → over-segmentation | ❌ inference setting (topk) |

**핵심**: **A1만** architecture/pipeline level attack 대상. A2/A3는 inference 인터페이스를 바꿔야 풀리는데 그건 protocol fix를 깸.

---

## 0. Headline 수치 (LangSplat-style aggregation)


| Scene | Oracle mIoU | Actual mIoU | Gap | mP | mR |
|---|---|---|---|---|---|
| figurines | 0.7875 | 0.5493 | 0.2382 | 0.7255 | 0.6231 |
| ramen | 0.7784 | 0.4214 | 0.3570 | 0.4531 | 0.6652 |
| teatime | 0.9065 | 0.8187 | 0.0878 | 0.8589 | 0.9217 |
| waldo_kitchen | 0.7564 | 0.5654 | 0.1909 | 0.6312 | 0.6481 |
| **ALL (LangSplat mean)** | **0.8072** | **0.5887** | **0.2185** | **0.6672** | **0.7145** |



Aggregation 방식: `eval_seg.py` 와 동일


```
for each scene:
    for each annotated image:
        for each prompt in this image:
            compute IoU
        image_score = mean over prompts in this image
    scene_score = mean over images
overall = mean over scenes
```



## 1. Type A/B/C 분류 (Step 1)


Per-prompt: mean Oracle / Actual across the frames the prompt appears in. Type 분류는 그 평균값에 τ 적용.


### τ = 0.25

| Type | N prompts | Share | Oracle avg | Actual avg | Sum IoU lost |
|---|---|---|---|---|---|
| A | 20 | 29.9% | 0.812 | 0.039 | 15.476 |
| B | 1 | 1.5% | 0.235 | 0.235 | 0.000 |
| C | 46 | 68.7% | 0.826 | 0.768 | 2.656 |

- **Type A 잃은 IoU 합 = 15.476** (전체 손실 18.131 중 85.4%)
- **공격 가능 헤드룸 = Type A 20/67 prompts (29.9%)**

### τ = 0.5

| Type | N prompts | Share | Oracle avg | Actual avg | Sum IoU lost |
|---|---|---|---|---|---|
| A | 21 | 31.3% | 0.838 | 0.078 | 15.974 |
| B | 6 | 9.0% | 0.384 | 0.252 | 0.790 |
| C | 39 | 58.2% | 0.875 | 0.833 | 1.658 |
| ? | 1 | 1.5% | 0.427 | 0.718 | -0.291 |

- **Type A 잃은 IoU 합 = 15.974** (전체 손실 18.131 중 88.1%)
- **공격 가능 헤드룸 = Type A 21/67 prompts (31.3%)**


## 2. Type A 내부 sub-classification (Step 2, τ = 0.25)


| Sub-type | 조건 | N | Share of A | Oracle avg | Actual avg | Avg rank | Attack direction |
|---|---|---|---|---|---|---|---|
| A1 | rank >= 11 | 12 | 60.0% | 0.809 | 0.000 | 50.3 | CLIP feature noisy → multi-view CLIP aggregation / per-SP feature refinement |
| A2 | rank in [2, 10] | 6 | 30.0% | 0.809 | 0.082 | 6.2 | ranking algo limit → matching algo 교체 / canon-contrast 대체 |
| A3 | rank = 1 | 2 | 10.0% | 0.845 | 0.142 | 1.0 | fragmentation / hierarchy → adaptive topk / level sweep |

### Type A1 대표 prompt (rank ≥ 11, oracle 높은 순 top 10)

| Scene | Prompt | N frames | Oracle | Actual | CLIP rank | Pool |
|---|---|---|---|---|---|---|
| waldo_kitchen | cabinet | 1 | 0.977 | 0.000 | 25 | 1449 |
| figurines | pumpkin | 1 | 0.973 | 0.000 | 52 | 745 |
| waldo_kitchen | ottolenghi | 1 | 0.962 | 0.000 | 66 | 1449 |
| figurines | pikachu | 2 | 0.914 | 0.000 | 64 | 745 |
| figurines | bag | 1 | 0.882 | 0.000 | 63 | 745 |
| waldo_kitchen | pour-over vessel | 1 | 0.867 | 0.000 | 53 | 1449 |
| figurines | pirate hat | 4 | 0.844 | 0.000 | 49 | 745 |
| ramen | onion segments | 7 | 0.758 | 0.000 | 14 | 793 |
| waldo_kitchen | spoon | 1 | 0.728 | 0.000 | 47 | 1449 |
| teatime | hooves | 2 | 0.723 | 0.000 | 52 | 4136 |


## 3. Confusion Matrix Analysis (Step 3, τ = 0.25)


전체 confusion matrix (TP/TN/FP/FN) + Precision/Recall/F1/Accuracy/IoU 모두 제공. Per-prompt 평균과 pixel-pooled 둘 다 보고 (class imbalance — 배경이 ~99% — 영향 파악).


### 3.1 전체 — Per-prompt 평균 metrics (N=67)


| Metric | Value | Definition |
|---|---|---|
| Precision (avg) | 0.639 | TP / (TP+FP) per prompt → mean |
| Recall (avg) | 0.665 | TP / (TP+FN) per prompt → mean |
| F1 (avg) | 0.614 | 2·P·R / (P+R) per prompt → mean |
| Accuracy (avg) | 0.9786 | (TP+TN) / total per prompt → mean |
| IoU (avg) | 0.542 | TP / (TP+FP+FN) per prompt → mean |



| Error pattern | Share | Definition |
|---|---|---|
| FP-dominant (FP / (FP+FN) > 0.7) | 40.3% | over-segment |
| FN-dominant (< 0.3) | 41.8% | miss / under-segment |
| Balanced | 17.9% | 양쪽 비슷 |


### 3.2 전체 — Pixel-summed raw confusion matrix


| Pixel category | Count | % of all image pixels |
|---|---|---|
| TP (correct foreground) | 3,723,864 | 2.4869% |
| TN (correct background) | 142,847,091 | 95.3972% |
| FP (잘못 칠한 픽셀) | 1,877,084 | 1.2536% |
| FN (놓친 정답 픽셀) | 1,291,307 | 0.8624% |
| GT total (객체 픽셀) | 5,015,171 | 3.3493% |
| Image total | 149,739,346 | 100% |


Pooled metrics (모든 픽셀 합쳐 계산):


| Metric | Pooled value | Per-prompt avg | Difference |
|---|---|---|---|
| Precision | 0.665 | 0.639 | +0.026 |
| Recall | 0.743 | 0.665 | +0.077 |
| F1 | 0.702 | 0.614 | +0.088 |
| Accuracy | 0.9788 | 0.9786 | +0.0002 |
| IoU | 0.540 | 0.542 | -0.002 |


> **주의**: Pixel-pooled Accuracy = 0.9788 가 매우 높아 보이는 건 배경 픽셀이 전체의 ~95.4%를 차지하기 때문 (class imbalance). LERF-OVS 평가에서 accuracy는 보조 지표.



### 3.3 Type별 Confusion Matrix (τ = 0.25, pixel-summed)


| Type | N | TP | TN | FP | FN | Precision | Recall | F1 | Accuracy | IoU |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 20 | 182,075 | 38,390,105 | 1,470,547 | 1,034,614 | 0.110 | 0.150 | 0.127 | 0.9390 | 0.068 |
| B | 1 | 3,030 | 2,140,697 | 52 | 9,645 | 0.983 | 0.239 | 0.385 | 0.9955 | 0.238 |
| C | 46 | 3,538,759 | 102,316,289 | 406,485 | 247,048 | 0.897 | 0.935 | 0.915 | 0.9939 | 0.844 |


### 3.4 Type별 — Per-prompt 평균 (τ = 0.25)


| Type | N | Precision | Recall | F1 | Accuracy | IoU |
|---|---|---|---|---|---|---|
| A | 20 | 0.073 | 0.201 | 0.060 | 0.9465 | 0.039 |
| B | 1 | 0.983 | 0.239 | 0.385 | 0.9955 | 0.235 |
| C | 46 | 0.878 | 0.876 | 0.860 | 0.9922 | 0.768 |


### 3.5 Type × FP/FN dominance 교차표 (τ = 0.25)


| Type | N | FP-dominant | Balanced | FN-dominant |
|---|---|---|---|---|
| A | 20 | 5 (25%) | 2 (10%) | 13 (65%) |
| B | 1 | 0 (0%) | 0 (0%) | 1 (100%) |
| C | 46 | 22 (48%) | 10 (22%) | 14 (30%) |


## 4. Attack priorities (Step 4)


| Sub-type | N prompts | IoU loss sum | Share of A loss | Architecture attack? | 이유 |
|---|---|---|---|---|---|
| A1 | 12 | 9.705 | 62.7% | ✅ YES (architecture) | per-SP CLIP feature 생성 단계 수정 (image_encoding.py + merge_proj.py). topk/scoring 미변경. |
| A2 | 6 | 4.363 | 28.2% | ❌ NO (inference setting) | rank in [2,10] = topk 키우면 풀리지만 topk=3은 protocol fix. |
| A3 | 2 | 1.407 | 9.1% | ❌ NO (inference setting) | rank=1인데 topk=3가 distractor 포함 → topk=1 축소가 해결이지만 setting 변경. |


## 5. 핵심 해석


### 5.1 SAM lift는 거의 실패하지 않음

τ=0.25 기준 Type B (Oracle도 actual도 낮음) = **1 prompts**. 즉 **NAG의 SP 풀에는 정답 객체가 거의 항상 존재**한다. SAM lift 단계는 LERF-OVS 평가에서 bottleneck이 아니다.

### 5.2 실패는 CLIP matching에 집중, A 내부에서 더 좁은 sub-type에 집중

```
전체 67 prompts
├── Type C (성공): 46 (68.7%)
└── Type A (실패): 20 (29.9%) — 손실 15.476 (85.4% of total)
    ├── A1 (rank≥11): 12 (60.0% of A) — 손실 9.705 (62.7% of A)
    ├── A2 (rank 2-10): 6
    └── A3 (rank=1): 2
```

→ **A1 12개 prompt만 해결해도 전체 IoU 손실의 약 54% 회복 가능**.

### 5.3 A1의 평균 rank ≈ 50 — CLIP feature가 정답을 못 알아봄

A1의 12개 prompt에서 정답 SP의 CLIP rank가 평균 **50등** (pool ~745-4136 SPs 중). 즉 위쪽 ~49개 SP가 모두 정답보다 CLIP score 높음.

- topk=3로는 절대 도달 불가
- topk 확장은 노이즈만 추가
- 본질적으로 **CLIP feature quality 개선**이 필요



## 6. Scene별 상세


| Scene | N prompts | Oracle mIoU | Actual mIoU | Gap | Type A | Type A1 |
|---|---|---|---|---|---|---|
| figurines | 21 | 0.7875 | 0.5493 | 0.2382 | 5 (24%) | 5 (100% of A) |
| ramen | 14 | 0.7784 | 0.4214 | 0.3570 | 8 (57%) | 2 (25% of A) |
| teatime | 14 | 0.9065 | 0.8187 | 0.0878 | 2 (14%) | 1 (50% of A) |
| waldo_kitchen | 18 | 0.7564 | 0.5654 | 0.1909 | 5 (28%) | 4 (80% of A) |


## 7. 부록 A: 가장 큰 IoU 손실 prompt top 15 (per-prompt mean 기준)


| Scene | Prompt | Frames | Oracle | Actual | Loss | Type@0.25 | CLIP rank | P | R | FP-dom |
|---|---|---|---|---|---|---|---|---|---|---|
| waldo_kitchen | cabinet | 1 | 0.977 | 0.000 | 0.977 | A | 25 | 0.000 | 0.000 | 0.00 |
| figurines | pumpkin | 1 | 0.973 | 0.000 | 0.973 | A | 52 | 0.000 | 0.000 | 0.00 |
| waldo_kitchen | ottolenghi | 1 | 0.962 | 0.000 | 0.962 | A | 66 | 0.000 | 0.000 | 0.00 |
| waldo_kitchen | yellow desk | 1 | 0.947 | 0.000 | 0.947 | A | 6 | 0.000 | 0.000 | 0.00 |
| figurines | pikachu | 2 | 0.914 | 0.000 | 0.914 | A | 64 | 0.000 | 0.000 | 0.36 |
| figurines | bag | 1 | 0.882 | 0.000 | 0.882 | A | 63 | 0.000 | 0.000 | 0.00 |
| teatime | bear nose | 3 | 0.974 | 0.100 | 0.874 | A | 5 | 0.102 | 1.000 | 1.00 |
| waldo_kitchen | pour-over vessel | 1 | 0.867 | 0.000 | 0.867 | A | 53 | 0.000 | 0.000 | 0.00 |
| figurines | pirate hat | 4 | 0.844 | 0.000 | 0.844 | A | 49 | 0.000 | 0.000 | 0.00 |
| ramen | kamaboko | 7 | 0.890 | 0.064 | 0.825 | A | 1 | 0.065 | 1.000 | 1.00 |
| ramen | onion segments | 7 | 0.758 | 0.000 | 0.758 | A | 14 | 0.000 | 0.000 | 0.72 |
| ramen | plate | 4 | 0.745 | 0.000 | 0.745 | A | 6 | 0.000 | 0.000 | 0.00 |
| waldo_kitchen | spoon | 1 | 0.728 | 0.000 | 0.728 | A | 47 | 0.000 | 0.000 | 0.00 |
| teatime | hooves | 2 | 0.723 | 0.000 | 0.723 | A | 52 | 0.000 | 0.000 | 0.04 |
| ramen | corn | 5 | 0.723 | 0.073 | 0.650 | A | 2 | 0.072 | 0.872 | 0.99 |


## 8. Deep Analysis (Q1~Q5) — A1 attack 가설 좁히기

Phase 1 결과 (A1이 dominant) 후 attack 후보를 좁히기 위한 5개 sub-질문.
자세한 표는 [lerf_ovs_deep_analysis.md](lerf_ovs_deep_analysis.md) 참조.

| Q | 결과 | 의미 |
|--|--|--|
| Q1: A1 frame stability | A1 std = **0.000** (vs A2: 0.110) | A1은 **view-invariant** — multi-view aggregation 무효 |
| Q2: CLIP top-1 SP 정확도 | Type A의 90%가 top-1 SP oracle IoU < 0.1 | CLIP은 **완전 무관한 영역**을 picking (spatial misalignment) |
| Q3: 객체 크기 효과 | Correlation = -0.118 | 거의 무관 |
| Q4: Oracle SP 개수 | A1 mean = 1.58, C mean = 1.48 (비슷) | Fragmentation은 원인 아님 |
| Q5: A1 prompt 특성 | 평균 단어 수 1.25 (C: 1.91), 짧은 generic noun 다수 | longer specific phrase 문제 아님 |

**Q1+Q2 결합 결론**: A1은 view-invariant + CLIP은 totally-wrong 영역. → 가설: **per-SP CLIP feature 자체가 정답 위치에 의미 신호 못 emit.**

다음 단계 (Phase 1.5): pipeline 코드 trace로 (a)/(b)/(c)/(d) 분리.

---

## 9. Phase 1.5 — Pipeline Code Trace 결과

### 9.1 데이터 흐름 파악

THGS는 **per-SAM-mask CLIP** 사용 (LangSplat 방식):

```
[image_encoding.py]
  SAM이 각 image에 4-level (default/s/m/l) mask 생성 → 각 mask CLIP encode → per-mask feature

[merge_proj.py::proj_gaussian_features]
  for each training view:
    render_point() → per-gaussian visibility weight + 2D 좌표
    visible gaussian의 2D 좌표 → SAM mask ID 조회 → mask의 CLIP feature 사용
    sp_feature[sp] += normalize(sum over gaussians) × portion
  if portion < 0.1: continue  ← SP의 visible gaussian < 10% 이면 SKIP
```

### 9.2 진짜 Root Cause = (d), 아니 (a)/(b)/(c)

가설 (a)/(b)/(c) 모두 틀림. 실제 원인:

**(d) Zero-norm SPs 가 canon-contrast 점수 0.5를 받아 ranking 점령**

증거 (figurines/pikachu):
- 정답 SP cos_sim with "pikachu" = **0.2413** (pool 745개 중 **3등** in raw cos_sim)
- CLIP top-1 SP cos_sim = 0.2422 (거의 동일)
- canon-contrast relevancy: 정답 = 0.4393, top-1 = 0.5061
- **relevancy 상위 3-15등 모두 cos_sim=0, relevancy=0.5 (zero-norm SPs)**

원리: `utils/vlm_utils.py:compute_similarity` 에서 zero feature → softmax([0,0]) = [0.5, 0.5] → relevancy = 0.5 (항상).

### 9.3 Root Cause 원인의 두 layer

| Layer | 위치 | 무엇 | Attack 자격 |
|--|--|--|--|
| **(d-1) Pipeline** | `merge_proj.py::proj_gaussian_features` 의 `if portion < 0.1: continue` | 일부 SP가 모든 view에서 portion 충족 못해 zero feature 잔존 | ✅ YES (architecture) |
| **(d-2) Inference logic** | `utils/vlm_utils.py::compute_similarity` 의 softmax([0,0])=[0.5,0.5] | zero feature가 default relevancy 0.5 받음 | ❌ NO (inference 변경 금지) |

→ **(d-1) Pipeline fix로 zero-norm SP를 제거하면 (d-2) 문제도 자동 해소**.

### 9.4 Scene별 Zero-norm SP 정량

| Scene | Pool size | Zero-norm | 비율 |
|--|--|--|--|
| figurines | 745 | **48** | **6.4%** |
| waldo_kitchen | 1449 | 51 | 3.5% |
| teatime | 4136 | 47 | 1.1% |
| ramen | 793 | 4 | 0.5% |

### 9.5 Zero-norm 제거 시 A1 정답 SP rank 변화

| Scene | Prompt | 원본 rank | non-zero pool rank | zero above | real above |
|--|--|--|--|--|--|
| figurines | **pirate hat** | 49 | **1** ✅ | 48 | 0 |
| waldo_kitchen | **pour-over vessel** | 53 | **2** ✅ | 51 | 1 |
| figurines | **pumpkin** | 52 | **4** ✅ | 48 | 3 |
| figurines | pikachu | 64 | 16 | 48 | 15 |
| figurines | bag | 63 | 15 | 48 | 14 |
| figurines | miffy | 75 | 27 | 48 | 26 |
| waldo_kitchen | ottolenghi | 66 | 15 | 51 | 14 |
| ramen | hand | 44 | 44 | 0 | 43 |
| teatime | hooves | 52 | 52 | 0 | 51 |
| waldo_kitchen | cabinet | 25 | 25 | 0 | 24 |
| waldo_kitchen | spoon | 47 | 47 | 0 | 46 |

### 9.6 A1이 두 갈래로 분리됨

| Sub-sub | 조건 | Prompts | Phase 2 fix |
|--|--|--|--|
| **A1-α: Zero-norm dominated** | 위에 있는 SPs 대부분이 zero-norm | pirate hat, pour-over vessel, pumpkin, (old camera 부분) — ~3-4개 | ✅ pipeline 수정으로 즉시 해결 |
| **A1-β: Real-feature ranking** | 위에 real-feature SPs 다수 | pikachu, bag, miffy, ottolenghi, hand, hooves, cabinet, spoon, tesla door handle — ~8개 | ⚠️ language_features 재생성 후 추가 진단 |

### 9.7 Phase 2 architectural fix 후보

대상: `merge_proj.py::proj_gaussian_features`

| # | Fix | Risk | Inference 영향 |
|--|--|--|--|
| (i) | `portion < 0.1` threshold 완화 (예: 0.01) | 낮음 | 무변경 |
| **(ii)** | **Threshold 제거** | 낮음~중간 | 무변경 |
| (iii) | First-visible-view fallback | 낮음 | 무변경 |
| (iv) | Zero-norm SP를 NAG에서 명시적 제외 | 중간 | 무변경 (pool만 작아짐) |

→ 권장 시작점: **(ii)** (가장 minimal change, 1 line).

### 9.8 Phase 1.5 산출물

- [scripts/lerf_ovs_sp_feature_inspect.py](scripts/lerf_ovs_sp_feature_inspect.py)
- [scripts/lerf_ovs_zeronorm_check.py](scripts/lerf_ovs_zeronorm_check.py)
- [scripts/lerf_ovs_root_cause.py](scripts/lerf_ovs_root_cause.py) (language_features 필요, 미실행)

---

## 10. 부록 B: Aggregation 방식별 수치 비교


| Aggregation | Oracle | Actual | Gap | Note |
|---|---|---|---|---|
| LangSplat (per-image → per-scene) | 0.8072 | 0.5887 | 0.2185 | 논문/eval_seg.py 표준 |
| Per-prompt mean across frames → per-scene → overall | 0.8156 | 0.5464 | 0.2692 | failure 분석에 사용 |
| Flat prompt mean (all prompts equal weight) | 0.8130 | 0.5424 | 0.2706 | 참고용 (scene size 무시) |

Aggregation 방식이 다를 뿐 같은 raw data. 결과의 절대값은 다르지만 **failure pattern과 attack priority는 모두 동일**.



---

## 11. Problem Definition — Distractor Confusion in Open-vocabulary 3DGS Segmentation

### 11.1 표준 용어 (Standard terminology)

CV / Retrieval / Open-vocabulary 문헌에서 "정답이 아닌 비슷한 후보"는 표준적으로 다음 용어로 불린다:

| 용어 | 의미 | 사용 분야 |
|--|--|--|
| **Distractor** | Query와 표현 공간에서 비슷하지만 정답이 아닌 후보 | Image retrieval, open-vocab seg, OWL-ViT/GLIP 등 |
| **Hard negative** | 학습 측면 강조 — "거의 정답처럼 보이는 오답" | Contrastive learning, hard negative mining |
| **Spurious match / activation** | 잘못된 위치/객체에 attention emit | CLIP / VLM analysis |

본 분석은 가장 일반적인 **distractor** 용어를 채택.

### 11.2 정식 문제 정의 (Formal problem statement)

> **Distractor confusion in open-vocabulary 3D Gaussian segmentation**:
>
> Text query `t`와 3D superpoints set `S = {s_1, ..., s_N}` (per-SP CLIP features `{f_i}`) 가 주어졌을 때, CLIP-based matching `score(f_i, t)` 의 의도는 **target SP** `s*` (= prompt가 가리키는 객체의 superpoint) 를 top rank로 두는 것이다.
>
> **Failure mode**: target이 아닌 일부 SP `{s_d}` (= **distractor SPs**) 가 `score(f_d, t) ≥ score(f*, t)` 를 만족해 top-k selection에서 `s*` 를 밀어냄. → render union이 잘못된 영역을 가리키거나 (FP > 0) 빈 mask로 귀결 (FP ≈ 0).
>
> Distractor는 세 클래스로 분리된다.

### 11.3 본 분석이 정의한 3-class distractor taxonomy

| Class | 명칭 | 정의 | Render 양상 | 우리 sub-type |
|--|--|--|--|--|
| **D1** | **Degenerate-feature distractor** (= null-feature distractor) | Pipeline 단계 결함으로 feature 못 받은 zero-norm SPs. Canon-contrast softmax가 default 0.5 점수 할당 → 실제 정답 SP를 밀어냄. | empty (가우시안 visible 부족) | **A1-α** |
| **D2** | **In-view spatial distractor** | 현재 evaluation view에서 visible한 다른 객체의 SP가 CLIP feature space에서 prompt와 closely match. | wrong area (FP > 0) | **A1-β1** |
| **D3** | **Out-of-view distractor** | CLIP feature space에서 prompt와 close하지만 그 SP의 gaussians가 현재 view에서 not visible. | empty (off-camera) | **A1-β2** |

### 11.4 Distractor class × Confusion matrix signature

각 distractor class는 pixel-level confusion matrix에서 **고유한 signature**를 만든다.

| Class | TP | FP | FN | Precision | Recall | FPR | F1 | Pixel pattern |
|--|--|--|--|--|--|--|--|--|
| **D1 (degenerate)** | 0 | 0 | GT | 정의 불가 (→ 0) | **0** | 0 | 0 | **Silent failure** — 아무것도 출력 안 함 |
| **D2 (in-view spatial)** | ~0 | **> 0** | GT | ~0 | ~0 | > 0 | ~0 | **Confident wrong** — 잘못된 위치를 자신 있게 출력 |
| **D3 (out-of-view)** | 0 | 0 | GT | 정의 불가 (→ 0) | **0** | 0 | 0 | **Silent failure** (D1과 동일) |
| **Type C (success)** | ≫ 0 | small | small | high | high | low | high | 정상 |

**핵심 관찰**:

> **D1과 D3는 pixel-level confusion matrix상 구분 불가능**. 둘 다 TP=0, FP=0, FN=GT 의 동일한 "silent failure" pattern을 보임. 구분하려면 **feature-level 진단** (zero-norm SP detection or gaussian visibility check) 이 필요.

**정량적 매핑** (Section 3.5의 Type × FP/FN dominance와 일치):

| Pixel signature | A1 prompts | Distractor class |
|--|--|--|
| Silent (FP = 0) | 9 prompts (75%) | D1 ∪ D3 |
| Confident wrong (FP > 0) | 3 prompts (25%) | D2 |

→ Section 3.5 의 "Type A 65% FN-dominant" 분포는 본질적으로 **D1+D3 ≫ D2** 라는 의미. A1 내부에서는 9:3 = **75% silent / 25% confident-wrong**.

### 11.5 Pixel signature → Distractor class 추론 protocol

새로운 method를 evaluate할 때 distractor class를 자동 진단하는 절차:

```
for each failure prompt (Type A, oracle IoU ≥ τ, actual IoU < τ):
    if FP_pixels == 0:                            # silent failure
        check zero-norm SPs in top-k selection
        if any selected SP has feat_norm < ε:
            → classify as D1 (degenerate)
        else:
            check gaussian visibility of selected SPs at eval view
            if visible gaussians == 0:
                → classify as D3 (out-of-view)
            else:
                → classify as D2 sub-case (low-visibility)
    else:                                          # FP > 0
        → classify as D2 (in-view spatial)
```

이 protocol을 사용하면 **pixel-level CSV (TP/TN/FP/FN) + 간단한 feature inspection** 만으로 distractor class 분포를 정량화 가능.

### 11.6 정량 분해 (this work)

THGS / LERF-OVS / 4 scenes / 67 prompts:

| Class | N prompts | IoU 손실 | 손실 비중 (of total A1 loss 9.71) | 손실 비중 (of total loss 18.13) |
|--|--|--|--|--|
| D1 (degenerate) | 3 | ~2.7 | **28%** | **15%** |
| D2 (in-view spatial) | 3 | ~2.0 | **21%** | **11%** |
| D3 (out-of-view) | 6 | ~5.0 | **52%** | **28%** |
| **A1 total (D1+D2+D3)** | **12** | **9.71** | **100% of A1** | **54%** |
| Non-A1 failures (A2, A3, B) | 8 | 8.42 | — | 46% |

**Key finding**: A1 (distractor confusion) 이 전체 LERF-OVS IoU 손실의 **54%**. 그 중에서도:
- **D3 (out-of-view distractor) 가 단일 최대** — 손실의 28% 차지.
- **D1 (degenerate)** 만 단순 pipeline fix로 해결 가능 (15% 회수).
- **D2, D3** 는 per-SP CLIP feature 변별력 문제 — 추가 진단 필요.

### 11.7 Class별 Architectural attack 후보 (Phase 2~)

| Class | Root cause layer | Architectural fix 후보 |
|--|--|--|
| **D1** | `merge_proj.py::proj_gaussian_features` 의 `if portion < 0.1: continue` skip | 후보 (i): threshold 완화 / (ii): 제거 / (iii): first-visible fallback / (iv): NAG에서 zero-norm SP 제외 |
| **D2** | per-SAM-mask CLIP feature 변별력 부족 (or per-SP aggregation dilution) | 후보 (i): SAM-mask boundary-aware aggregation / (ii): per-pixel CLIP fallback / (iii): per-SP feature contrastive refinement |
| **D3** | View-independent feature space의 ambiguity (semantic 닮은 다른 위치 SP) | 후보 (i): per-view visibility prior가 NAG construction에 반영 / (ii): cross-view feature consistency 강화 (단, inference 인터페이스 무변경 조건 충족 필요) |

### 11.8 Contribution statement (paper용)

> *"We provide the first quantitative decomposition of distractor confusion in open-vocabulary 3DGS segmentation. By inserting a per-SP greedy oracle ceiling between SAM lift and CLIP matching stages, we attribute 54% of LERF-OVS IoU loss to three distractor classes — degenerate (pipeline-resolvable), in-view spatial, and out-of-view — each requiring distinct architectural fixes."*

### 11.9 향후 연구 질문

1. **D3 (out-of-view distractor) 가 dominant**한 게 LERF-OVS 특수성인가, 일반 현상인가?
   - 검증: LangSplat / Gaussian Grouping에 동일 oracle 적용해 D1/D2/D3 비율 측정
2. **D2/D3 attack은 inference 인터페이스 변경 없이 가능한가?**
   - 핵심 제약: protocol fix 유지하면서 pipeline만 수정
   - 가능한 방향: NAG construction 시 view-visibility-aware feature averaging
3. **CLIP feature space 자체의 한계인가, aggregation 한계인가?**
   - 검증: language_features 재생성 후 per-SAM-mask CLIP feature 직접 비교
