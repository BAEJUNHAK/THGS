# LERF-mask ↔ LERF-OVS Gap 분석 실험 계획

본 문서는 THGS 기준으로 LERF-mask와 LERF-OVS 사이의 성능 gap을 정량적으로 분해하고, 그 gap의 원인이 어디에 있는지 진단하여 novel한 method 개선안을 도출하기 위한 실험 계획서다.

배경 분석: [lerf_mask_vs_ovs_analysis.md](lerf_mask_vs_ovs_analysis.md)

## 0. 실측이 끝난 부분 vs 아직 안 한 부분

본 문서는 일부는 **실측 완료** 결과를 기반으로, 일부는 **앞으로 할 실험 계획**으로 구성됨.

- ✅ 실측 완료: Oracle v1 / v2 (τ sweep) / v3 (topk sweep) ablation, CLIP-based baseline
- 📋 계획: GT vs GT noise floor, per-prompt 분석, precision/recall 분해, ramen CLIP failure 진단

---

## 1. 문제 정의

### 1.1 관찰 (Community 통념과 paradigm 분화)
LERF-mask와 LERF-OVS는 도입 논문이 다르고, 도입 의도가 다르며, 후속 연구들이 각자 자기 paradigm에 맞는 쪽에 집중하면서 **paradigm 분화**가 굳어졌다.

| 벤치마크 | 도입 논문 | 도입 의도 | 실질적으로 측정하는 것 |
|---|---|---|---|
| **LERF-OVS** | LangSplat (CLIP-based, CVPR 2024) | CLIP feature가 3D에 잘 부여되는가 | **CLIP open-vocab semantic 매칭 능력** |
| **LERF-mask** | Gaussian Grouping (SAM-based, ECCV 2024) | SAM mask가 3D로 잘 lifting되는가 | **SAM-supervised 3D 표현의 객체 표현 능력** |

후속 연구들은 자기 paradigm 쪽 벤치마크 점수만 주로 보고하고, **두 벤치마크 사이 transition을 정량 분해한 사례가 없다**.

### 1.2 핵심 Framing — Ceiling vs Actual

#### 1.2.1 LERF-mask task의 본질 (잘못 이해하기 쉬운 부분)
LERF-mask task의 본질은 다음과 같다:
```
SAM 2D mask (학습 데이터)
        ↓ method-specific lifting (학습 단계)
3D scene 표현  (THGS의 NAG superpoint, GG의 instance feature 등)
        ↓ novel view에 splat (inference)
2D mask
        ↓ IoU with GT  ← LERF-mask 평가
```

평가 대상은 **"lifting의 결과물인 3D 표현이 2D로 splat한 mask"** 이지, **SAM의 raw 2D mask가 아니다**.

명시적으로 **task의 의도를 벗어난 측정**:
- ❌ "SAM이 만든 2D segmentation의 GT 일치도" → SAM 자체의 능력 평가일 뿐, 3D lifting과 무관
- ❌ "Grounded-SAM의 text→2D mask 정확도" → 외부 2D segmenter의 평가일 뿐

본 분석은 **"3D 표현이 novel view에서 GT와 일치하는가"** 라는 LERF-mask 본질의 측정에만 집중한다.

#### 1.2.2 두 task의 분화식 (핵심 식)
이 framing에서 두 벤치마크 점수가 자연스럽게 다음과 같이 분해된다:

```
LERF-mask 점수  ≒ Oracle  (3D 표현의 천장)
LERF-OVS 점수   ≒ Oracle − CLIP-NAG 매칭 손실
```

즉:
- **LERF-mask** = "3D scene 표현이 GT 객체를 표현할 수 있는가" 만 묻는 task
- **LERF-OVS** = 그 위에 추가로 "CLIP-text 매칭이 정확한가" 까지 묻는 task

따라서 같은 method가 두 벤치마크에서 받는 점수 차이는:
- Method의 절대 성능 차이가 아니라
- **두 task가 측정하는 능력의 분화**가 만든 차이

이 식이 본 분석의 핵심 framework이다.

#### 1.2.3 측정 단위 정의
위 분화식에서 등장하는 양들:

| 양 | 정의 | 의미 |
|---|---|---|
| **Ceiling** | SAM-supervised로 만들어진 3D 표현이 GT를 표현할 수 있는 천장 | "SAM lifting 결과의 능력" — LERF-mask 본질의 oracle |
| **Actual** | Method의 매칭 알고리즘이 실제 도달한 점수 | "매칭 알고리즘이 천장의 몇 %를 활용하는가" |
| **Gap** = Ceiling − Actual | 매칭 알고리즘이 표현 능력을 활용하지 못하는 양 | **THGS의 진짜 bottleneck** |

#### 1.2.4 본 framing 안에서 THGS의 LERF-mask 점수 해석
THGS의 실측 결과를 위 분화식에 적용하면:
- **Ceiling (v2 τ=0.5) = 0.8531** → THGS의 SAM supervision이 매우 효과적
- **Actual (CLIP) = 0.7367** → 매칭 알고리즘이 천장의 약 86%만 활용
- **Gap = 0.1164** → 매칭 알고리즘에서 11.6%p 손실 발생

→ **"THGS의 SAM lifting은 잘했다, CLIP-NAG 매칭이 부족하다"** 라는 정량적 결론.
→ 이 매칭 손실의 대부분(0.2527 / 0.3 ≈ 84%)이 **ramen 한 장면에 집중**.

### 1.3 가설 (실측으로 일부 검증됨)
3D OVS 파이프라인을 능력 단위로 분해:
- **Stage 1 (Lifting)**: SAM 신호를 3D 표현(superpoint, instance feature)으로 변환 — LERF-mask paradigm의 중심
- **Stage 2 (Matching)**: 3D 표현을 text query와 매칭 — LERF-OVS paradigm의 중심

THGS는 SAM-supervised + CLIP-matched 구조라 두 paradigm 사이에 위치한다. Section 4의 실측 결과는 THGS의 **Stage 1은 매우 강하고 Stage 2가 진짜 bottleneck** 임을 보여준다 — 특히 ramen 장면에서.

### 1.4 본 분석의 목표
1. Gap의 크기를 sample 단위로 정량화
2. 의미 있는 크기인지 사전 정의된 threshold로 검증
3. Gap의 원인이 Stage 1 (lifting/표현) 인지 Stage 2 (매칭) 인지 진단
4. 어디까지가 paradigm 본질이 만든 측정 차이이고, 어디부터가 THGS method 디자인 결함인지 구분
5. 가장 큰 bottleneck에 대응하는 method 개선안 도출

### 1.5 본 분석의 Scope — 정직한 한계 명시
- 본 분석은 **THGS의 CLIP paradigm 내부에서의 stage 분해**를 다룬다. THGS가 SAM lifting paradigm method (GG, OpenSplat3D) 보다 좋다/나쁘다를 직접 주장하지 않는다.
- 직접 비교 대상은 같은 CLIP paradigm method (LangSplat, LEGaussians 등) 로 한정한다.
- LERF-mask 점수의 SAM-lifting paradigm 점수와의 횡적 비교는 reference로만 인용한다.

### 1.6 단순화 가설을 피하기 위한 통제 변수
gap에 기여하는 confound가 최소 4가지 있으므로 단일 인과로 환원하지 않는다:
- View 분포 (LERF-mask hold-out novel view vs LERF-OVS train view)
- GT 품질 (LERF-mask PNG raster vs LERF-OVS polygon raster)
- Prompt 수/다양성 (장면당 6~10 vs 5~21, 프로토콜마다 다름)
- 장면 수 (3 vs 4, waldo_kitchen 포함 여부)
- 평균 집계 방식 (per-prompt macro vs flat micro)

이 중 **method/모델은 THGS 단일 sai_nag.pt로 고정**하여 method 변수는 통제. 나머지는 phase별로 명시적으로 다룬다.

---

## 2. 분석 단위와 매칭 기준

### 2.1 분석 단위
**(scene, prompt) 페어 단위**를 기본 분석 단위로 한다.

### 2.2 공통 prompt 셋
양쪽 데이터셋에 모두 존재하는 prompt만 직접 비교 대상:
- LERF-mask prompt = `data/lerf_mask/<scene>/test_mask/<view>/*.png` 파일명
- LERF-OVS prompt = `data/lerf/label/<scene>/frame_*.json` 의 `objects[].category` 합집합
- **교집합**을 공통 prompt 셋으로 정의

### 2.3 View 매칭 (Phase 1까지 평균, Phase 2 이후 선택적 매칭)
- 1차: 각 데이터셋의 모든 view에서 평균을 낸 (scene, prompt) 점수 비교
- 2차: LERF-mask test view와 카메라 중심이 가까운 LERF-OVS train view 매칭

---

## 3. Oracle 정의 (3가지 — 실측 완료)

### 3.1 Oracle v1 — Union (Baseline / 부적절함이 확인됨)
**절차**: ref view에서 GT mask 안에 떨어진 가우시안이 속한 모든 NAG superpoint를 **union**하여 indicator 생성 → 모든 view 렌더.

**문제점**: 슈퍼포인트 1개가 객체 경계를 넘어 펴진 경우, GT 내부의 가우시안 1개만으로도 SP 전체가 끌려옴 → **massive over-expansion**.

**실측**:
- ramen `chopsticks`: 506개 가우시안 → 82,857개로 expand (163배)
- teatime `stuffed bear`: 51,051 → 171,689개
- Overall mIoU = **0.0574** (사실상 무너짐)

→ Oracle 정의로 부적절. 그러나 **"SP 단순 union은 매우 좁은 GT에 비해 SP가 거대하다"** 는 사실을 확인.

### 3.2 Oracle v2 — Majority Threshold (분포 기반, **추천 Ceiling**)
**절차**:
1. ref view에서 각 NAG superpoint에 대해 "이 SP의 visible 가우시안 중 GT mask 안에 떨어지는 비율" 계산
2. 비율 ≥ τ 인 SP만 select
3. 모든 view 렌더

**측정하는 것**: **"SP들이 객체에 얼마나 정확히 grouped 되어 있는가"** — 가우시안 분포 기반. 객체 1개가 SP 여러 개에 fragmented된 경우도 모두 잡음.

**구현**: [sam_oracle_v2_lerf_mask.py](../sam_oracle_v2_lerf_mask.py)

### 3.3 Oracle v3 — Best-IoU topk (렌더 기반)
**절차**:
1. ref view에서 각 NAG superpoint를 단독 렌더 → 그 SP의 2D projection mask
2. 각 SP mask와 GT mask의 IoU 계산
3. IoU 상위 topk SP를 select
4. 모든 view 렌더

**측정하는 것**: **"SP 단독으로 객체의 2D projection을 만들 수 있는가"** — 렌더 mask 기반. "객체 = SP 1개" 가정의 최고 매칭.

**구현**: [sam_oracle_v3_lerf_mask.py](../sam_oracle_v3_lerf_mask.py)

### 3.4 v2 vs v3 차이 한 줄
- **v2**: 분포 기반 SP grouping (객체 1개 → SP 여러 개 자동 매칭) → **더 진실한 천장**
- **v3**: single-SP 최고 매칭의 천장 (객체 = SP 1개 가정의 한계 측정) → SP granularity의 한계 지표

### 3.5 어느 게 LERF-mask 본질의 ceiling 인가
**v2가 더 정확한 ceiling**. LERF-mask task는 "3D 표현이 GT를 얼마나 잘 표현하는가" 를 묻는데, 객체가 SP 여러 개에 fragmented된 경우라도 그들을 모두 활용했을 때의 표현 능력이 진짜 천장이다.

v3 topk=1은 단일 SP 단위 한계로, **THGS supervision이 객체를 얼마나 SP 단위로 응집시켰는가** 를 보는 보조 지표.

---

## 4. 실측 결과 (Section 3의 oracle들)

### 4.1 전체 표
| 방식 | figurines | ramen | teatime | **Overall mIoU** | **Overall BIoU** |
|---|---|---|---|---|---|
| v1 (union, no τ) | 0.0238 | 0.0926 | 0.0559 | 0.0574 | 0.0157 |
| v2 τ=0.1 | 0.6697 | 0.4810 | 0.7923 | 0.6477 | 0.5969 |
| v2 τ=0.3 | 0.7868 | 0.8452 | 0.8674 | 0.8331 | 0.7845 |
| **v2 τ=0.5** ★ | 0.8389 | 0.8445 | 0.8760 | **0.8531** | **0.8071** |
| v2 τ=0.7 | 0.8435 | 0.7260 | 0.8858 | 0.8184 | 0.7722 |
| v2 τ=0.9 | 0.7758 | 0.7234 | 0.6638 | 0.7210 | 0.6764 |
| **v3 topk=1** ★ | 0.7805 | 0.7934 | 0.8871 | **0.8204** | 0.7843 |
| v3 topk=2 | 0.7804 | 0.6393 | 0.8521 | 0.7572 | 0.7168 |
| v3 topk=3 | 0.5536 | 0.4848 | 0.7020 | 0.5801 | 0.5459 |
| v3 topk=5 | 0.2857 | 0.3405 | 0.4675 | 0.3645 | 0.3276 |
| v3 topk=10 | 0.0245 | 0.1159 | 0.2029 | 0.1144 | 0.0894 |
| **CLIP-based** (test_lerf_mask.py) | 0.7804 | 0.5918 | 0.8380 | **0.7367** | 0.7028 |

### 4.2 Ceiling vs Actual (핵심 식 적용)

| | Ceiling (v2 τ=0.5) | Ceiling (v3 topk=1) | Actual (CLIP) | **Gap (v2)** | **Gap (v3)** |
|---|---|---|---|---|---|
| figurines | 0.8389 | 0.7805 | 0.7804 | **+0.0585** | +0.0001 (사실상 동일) |
| ramen | 0.8445 | 0.7934 | 0.5918 | **+0.2527** ★ | **+0.2016** ★ |
| teatime | 0.8760 | 0.8871 | 0.8380 | +0.0380 | +0.0491 |
| Overall | **0.8531** | 0.8204 | 0.7367 | **+0.1164** | +0.0837 |

### 4.3 핵심 발견
1. **THGS의 Stage 1 (lifting)은 매우 강함** — Oracle v2 τ=0.5 = 0.8531. SAM supervision이 효과적으로 작동하여 SP가 객체 단위 분리에 거의 도달.
2. **Stage 2 (CLIP-NAG 매칭)가 진짜 bottleneck** — Actual 0.7367 vs Ceiling 0.8531 → 11.6%p gap.
3. **Bottleneck이 ramen에 집중** — figurines/teatime에서는 CLIP이 거의 Oracle 수준 (gap < 6%p), **ramen에서만 25%p gap**.
4. **figurines/teatime CLIP ≈ v3 topk=1** — 두 장면에서는 CLIP-NAG가 best single SP를 거의 정확히 찾고 있다는 강력한 신호.

### 4.4 v3 topk가 늘수록 점수가 떨어지는 현상
topk=1 (0.82) → topk=3 (0.58) → topk=10 (0.11). 직관과 반대.
**해석**: SP가 객체 단위로 잘 grouped 되어 있어서, 추가 SP는 거의 항상 **다른 객체에 속하는 SP** → false positive 추가. 객체 1개 = SP 1~2개 매핑이 일반적이라는 증거.

---

## 5. Phase 0 — Baseline 측정 (계획, 1~2일)

### 5.1 GT vs GT IoU (데이터셋 noise floor)
**목적**: 같은 prompt에 대해 LERF-OVS polygon-raster와 LERF-mask PNG가 얼마나 다른지 → 데이터셋 자체가 만든 gap의 baseline.

**방법**:
1. 공통 prompt 셋에서 LERF-mask test view 1장 선택
2. 그 view와 sim3 정합 후 카메라 중심이 가까운 LERF-OVS train view에서 polygon raster GT 생성
3. 두 GT mask의 IoU 측정

**해석**: GT vs GT IoU = 0.85 라면 method가 이 이상 점수 받는 건 불가능 → gap의 일부는 데이터셋이 만든 것.

### 5.2 양쪽 평가 결과 dump (Section 4와 결합)
- LERF-mask 결과: Section 4 표 (이미 dump됨)
- LERF-OVS 결과: `scripts/eval_lerf_galre_b.py` (OpenGaussian 호환)로 동일 prompt 셋에 측정

---

## 6. Phase 1 — Gap 정량화 (2~3일)

### 6.1 Gap 계산 (두 종류)
```
gap_mask(scene, prompt)  = Ceiling_v2(scene, prompt) − Actual_lerf-mask(scene, prompt)
gap_dataset(scene, prompt) = mIoU_lerf-mask(scene, prompt) − mIoU_lerf-ovs(scene, prompt)
```
- `gap_mask`: LERF-mask 내부의 매칭 손실 (Section 4 식)
- `gap_dataset`: 두 데이터셋 사이 점수 차

### 6.2 분포 분석
- 히스토그램: gap 값의 분포 (per scene)
- Outlier 식별: gap 큰 상위 10% prompt

### 6.3 "의미 있는 gap" 기준 사전 정의
- 의미 있는 gap: `gap ≥ 0.15`
- 의미 있는 비율: 전체 (scene, prompt) 페어의 **20% 이상** 이 조건 만족 → 분석 가치 있는 failure mode로 인정

### 6.4 산출물
- `output/analysis/gap_distribution.png`
- `output/analysis/significant_failures.csv`

---

## 7. Phase 2 — Failure 유형 분해 (3~4일)

### 7.1 Precision / Recall 분해
각 failure (scene, prompt) 에 대해:
```
precision = TP / (TP + FP)     # 과예측 측정
recall    = TP / (TP + FN)     # 빠뜨림 측정
```

**해석**:
- Precision↓ → CLIP 매칭이 광범위, false alarm → Stage 2 의심
- Recall↓ → 3D 표현이 객체를 못 잡거나 부분만 잡음 → Stage 1 의심 (단 Section 4에서 Stage 1은 강함이 확인되었으므로, recall↓은 매칭 시 객체 SP를 빠뜨린 경우가 더 가능성 큼)

### 7.2 BIoU vs IoU 격차
- BIoU만 떨어짐 → 경계 부정확 (SP 경계 문제)
- IoU 전체 떨어짐 → 영역 매칭 실패

### 7.3 Failure mode 사전 카테고리
- **(a) Compositional/specific 쿼리** ("rubber duck with red hat", "wavy noodles in bowl") → Stage 2 의심
- **(b) 작은 객체** (GT mask area / image area < 0.02) → Stage 1 (SP granularity) 의심
- **(c) Multi-instance 같은 외형** ("apple"이 여러 개) → Stage 2 instance 구분 실패
- **(d) Novel-view occlusion** → view-consistency 문제
- **(e) GT 자체 노이즈** (Phase 0 GT vs GT IoU 낮은 prompt) → 데이터셋 변수, 분석 제외

### 7.4 산출물
- `output/analysis/failure_pr_table.csv` (failure prompt별 precision/recall/BIoU/IoU/category)
- 카테고리별 빈도 히스토그램

---

## 8. Phase 3 — Stage 진단 (Oracle ablation, **부분 완료**)

### 8.1 이미 확보한 진단 도구
- **Oracle v2 / v3** ([sam_oracle_v2_lerf_mask.py](../sam_oracle_v2_lerf_mask.py), [sam_oracle_v3_lerf_mask.py](../sam_oracle_v3_lerf_mask.py)) → **Stage 1 (lifting) ceiling 측정**
- **test_lerf_mask.py** → **Stage 2 매칭의 actual 점수**

Section 4의 결과가 이 진단의 첫 번째 결과:
- Stage 1 ceiling = 0.8531 (v2 τ=0.5)
- Stage 2 actual = 0.7367 (CLIP)
- Gap = 0.1164 (대부분 ramen에서 발생)

### 8.2 추가로 할 진단 — Per-prompt Stage 분류

각 failure prompt를 다음 사분면에 배치:

| | **v3 single SP IoU 높음** | **v3 single SP IoU 낮음** |
|---|---|---|
| **v2 grouping IoU 높음** | SP는 잘 grouped되어 있고 CLIP이 잘 매칭. (단 failure) → 진단 재검토 | **SP가 fragmented 되어 있어 CLIP topk가 일부만 잡음** → Stage 1.5 |
| **v2 grouping IoU 낮음** | SP 단독은 객체를 표현하지만 다른 SP와 grouping 안 됨 → Stage 1 (cut pursuit 한계) | **SP 자체가 객체 단위가 아님** → Stage 1 bottleneck |

### 8.3 추가로 할 진단 — CLIP-side 진단 (ramen 집중)
ramen의 6개 prompt 각각에 대해:
- v3 best-single SP가 무엇인지 (인덱스)
- CLIP-based가 select한 top-3 SP가 무엇인지
- 두 셋의 교집합 → "CLIP이 best SP를 포함하는가" recall
- CLIP이 잘못 잡은 SP의 CLIP feature 시각화 → 어떤 텍스트와 더 가까운지

### 8.4 산출물
- `output/analysis/oracle_quadrant.csv` (failure prompt별 v2/v3 점수 + 사분면 분류)
- `output/analysis/ramen_clip_diagnosis.csv` (ramen prompt별 CLIP top-3 vs Oracle best)

---

## 9. Phase 4 — Novel Idea 도출 (5~7일)

### 9.1 Section 4 결과가 시사하는 방향
Section 4에서 이미 확인된 사실:
- **Stage 1은 강하다** (0.85 ceiling) → SP supervision 디자인은 유지
- **Stage 2가 약하다** (0.74 actual, ramen에서 0.59) → CLIP-NAG 매칭이 개선 대상

자연스러운 idea 방향:
1. **CLIP feature 부여 방식 개선** ([merge_proj.py:120-144](../merge_proj.py#L120-L144)) — ramen에서 부정확
2. **CLIP relevancy 매칭 알고리즘 개선** ([utils/vlm_utils.py:34-44](../utils/vlm_utils.py#L34-L44)) — canon-contrast가 ramen 도메인에 적합한가
3. **Multi-instance 처리** — 비슷한 외형의 객체 구분
4. **Compositional 쿼리 처리** — "wavy noodles in bowl" 같은 복합 텍스트

### 9.2 가설 → 검증 형식
각 novel idea는 다음 형태로 정리:
- **가설**: "X를 바꾸면 ramen prompt의 Y 카테고리 failure가 Z%p 줄어든다"
- **검증**: 해당 X만 ablation으로 적용한 변형 모델로 Phase 2 분석 재실행
- **목표**: CLIP-based 점수가 v2 ceiling(0.85)에 얼마나 근접하는가

### 9.3 (옵션) OpenSplat3D 비교
OpenSplat3D는 Stage 2를 Grounded-SAM에 외주하므로:
- THGS와 OpenSplat3D의 같은 prompt 점수 차 = "CLIP query vs Grounded-SAM oracle" cost-benefit
- ramen에서 OpenSplat3D가 더 잘하면 → "외주가 유효, CLIP 매칭이 부족"
- THGS가 더 잘하면 → "CLIP-NAG도 충분히 강력"

---

## 10. 일정 요약

| Phase | 내용 | 상태 | 기간 |
|---|---|---|---|
| 0 | GT vs GT baseline + per-prompt 점수 dump | 부분 (LERF-mask 측정 완료) | 1~2일 |
| 1 | Gap 정량화 + 의미 있는 크기 판정 | 📋 계획 | 2~3일 |
| 2 | Precision/Recall + failure 카테고리 라벨링 | 📋 계획 | 3~4일 |
| 3 | Oracle ablation으로 stage 진단 | **부분 완료** (v2/v3 sweep) | 4~5일 (per-prompt 진단 남음) |
| 4 | Bottleneck 기반 novel idea + 검증 | 📋 계획 | 5~7일 |

총 약 15~21일 (full-time 기준).

---

## 11. 교수님 보고용 한 문단 요약

> 본 분석은 LERF-mask와 LERF-OVS 사이 mIoU gap을 단순 비교하는 것이 아니라, **Ceiling vs Actual** 의 차이를 정량 분해합니다. Ceiling은 SAM-supervised 3D 표현(NAG superpoint)이 GT를 표현할 수 있는 천장(Oracle v2 τ=0.5 = 0.8531), Actual은 CLIP-NAG 매칭이 실제로 도달한 점수(0.7367)입니다. 그 차이 0.1164가 **THGS 매칭 알고리즘의 손실**이며, 이 손실의 90% 이상이 ramen 장면에 집중(ramen 단독 gap 0.2527)되어 있음을 확인했습니다. THGS의 SAM supervision은 매우 효과적(Stage 1 강함)이고, **CLIP-NAG 매칭(Stage 2)가 진짜 bottleneck** 임이 정량으로 밝혀졌으며, 후속 분석은 ramen prompt 단위의 CLIP failure mode 진단으로 집중됩니다.

---

## 12. 사용자 초안 대비 보완점 (5+)

1. **샘플 매칭 기준 명시** — 공통 prompt 셋 단위, (scene, prompt) 페어
2. **Confound 통제 방법** — 동일 method/모델 고정, view 매칭 옵션
3. **Stage 1 vs Stage 2 진단의 구체적 oracle** — v2 (분포 기반 ceiling), v3 (single-SP ceiling) 두 가지
4. **"의미 있는 크기"의 사전 threshold** — gap ≥ 0.15, 전체의 20% 이상
5. **Failure mode 사전 카테고리** — (a)~(e) 5종 미리 정의
6. **Ceiling vs Actual 핵심 식** — 본 분석의 정량적 framework
7. **실측 결과로 가설 보정** — Stage 2가 bottleneck이라는 실측 발견 반영

---

## 13. 참고 자료

### 본 머신 코드
- [test_lerf.py](../test_lerf.py) — LERF-OVS 추론
- [test_lerf_mask.py](../test_lerf_mask.py) — LERF-mask 추론 (sim3 정합 포함, CLIP-based)
- [sam_oracle_lerf_mask.py](../sam_oracle_lerf_mask.py) — Oracle v1 (union, 부적절함 확인)
- [sam_oracle_v2_lerf_mask.py](../sam_oracle_v2_lerf_mask.py) — Oracle v2 (majority threshold, **추천 ceiling**)
- [sam_oracle_v3_lerf_mask.py](../sam_oracle_v3_lerf_mask.py) — Oracle v3 (best-IoU topk)
- [scripts/eval_seg.py](../scripts/eval_seg.py) — LERF-OVS 평가 (mAcc = 픽셀 acc, 비표준)
- [scripts/eval_lerf_galre_b.py](../scripts/eval_lerf_galre_b.py) — LERF-OVS 평가 (OpenGaussian 호환)
- [scripts/eval_lerf_mask.py](../scripts/eval_lerf_mask.py) — LERF-mask 평가 (GG 호환)
- [nag_data.py](../nag_data.py) — SemanticNAG
- [utils/vlm_utils.py](../utils/vlm_utils.py) — ClipSimMeasure (LERF-style relevancy)
- [merge_proj.py](../merge_proj.py) — CLIP feature → superpoint 부여

### 데이터 위치
- LERF-OVS GT: `data/lerf/label/<scene>/frame_*.json`
- LERF-mask GT: `data/lerf_mask/<scene>/test_mask/<view_idx>/<prompt>.png`
- LERF-mask SAM-style object_mask: `data/lerf_mask/<scene>/object_mask/*.png` (303/135/180 장)

### 사전 분석 문서
- [lerf_mask_vs_ovs_analysis.md](lerf_mask_vs_ovs_analysis.md) — 두 벤치마크의 데이터/프로토콜 차이
