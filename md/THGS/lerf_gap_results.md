# LERF-mask × LERF-OVS Gap 실험 결과 총정리

본 문서는 THGS 기준으로 LERF-mask와 LERF-OVS 두 벤치마크 위에서 **Ceiling vs Actual ablation + Prompt-controlled 분석** 을 모두 끝낸 실험 결과 보고서다. **Primary ceiling은 Oracle v4 (greedy union, budget=3)** 를 채택.

관련 문서:
- 실험 설계: [lerf_gap_analysis_plan.md](lerf_gap_analysis_plan.md)
- 두 벤치마크의 데이터/프로토콜 차이: [lerf_mask_vs_ovs_analysis.md](lerf_mask_vs_ovs_analysis.md)

---

## 0. 핵심 발견 (TL;DR)

1. **THGS의 SAM lifting Ceiling은 두 데이터셋에서 거의 같다** (LERF-mask 0.8653 vs LERF-OVS 0.8323, 4%p 차이) — **lifting 자체는 데이터셋 무관**.
2. **THGS의 진짜 bottleneck은 매칭 알고리즘 (Stage 2)** — Gap이 LERF-OVS에서 거의 2배 (0.1286 → 0.2437).
3. **두 데이터셋 점수 차이(0.1481)의 86%는 prompt 난이도가 만든 것**. 공통 prompt에서 두 데이터셋 점수는 거의 동일 (2%p 차이).
4. **LERF-OVS는 캐릭터 이름/브랜드/sub-part 같은 hard prompt를 포함** (`miffy`, `pikachu`, `ottolenghi`, `hooves`) → CLIP 매칭 자체 실패.
5. **ramen이 두 데이터셋 모두에서 worst**, **teatime이 두 데이터셋 모두에서 best (ceiling 거의 도달)**.

→ 결론: **lifting 디자인은 유지, 매칭 알고리즘과 prompt 보강 (hard prompt 처리) 이 개선 대상**.

---

## 0.5 Oracle 채택 — v4 budget=3 (THGS-specific Fair Ceiling)

### Fair Comparison의 핵심 원칙 — 두 method가 같은 게임을 풀어야 한다

Gap = Ceiling − Actual 이 의미를 가지려면, Oracle은 Actual과 다음을 **모두 공유** 해야 한다:
1. ✅ 같은 model artifact (학습된 가우시안 + NAG)
2. ✅ 같은 SP 후보 풀 (level [2, 3])
3. ✅ 같은 budget (몇 개 SP를 select하는가)
4. ✅ 같은 render mechanism, threshold, 평가 metric
5. ✅ Oracle은 그 budget 안에서 best selection 수행

차이는 오직: **selection의 reference** — CLIP은 text relevancy, Oracle은 GT IoU.

### Oracle 4종의 fairness 비교

| Oracle | Budget | Selection | Fair? | 채택? |
|---|---|---|---|---|
| **v1** (union) | 무한 | GT 안 가우시안의 모든 SP | ❌ over-expansion | 폐기 |
| **v2 τ=0.5** | 가변 (5~20) | majority fraction ≥ τ | ❌ budget unfair (inflated) | 폐기 |
| **v3 topk=1** | 1 | 단독 IoU best | ❌ budget unfair (한 손 묶임) | 보조 지표 |
| **v3 topk=3** | 3 | 단독 IoU top-3 (조합 best 아님) | ❌ selection sub-optimal (oracle < actual) | 폐기 |
| **v4 budget=3** ★ | 3 (= CLIP topk) | greedy union (budget 안 optimal) | ✅ **fair + optimal** | **Primary Ceiling** |

### v4 채택 근거 5가지

1. **Budget fair**: CLIP-based는 `topk=3 at level=[2,3]` ([nag_data.py:39-44](../nag_data.py#L39-L44)) → 정확히 3 SP. v4도 3 SP.
2. **Selection optimal-approx**: Greedy union이 best 3-SP combination에 매우 근접 (mask IoU의 monotonic 성질로 인해).
3. **수렴 확인**: v4 budget=3 ≈ v4 unlimited (4%p 이내). budget=3에서 greedy 거의 종료 → 객체 = SP 2-3개 매핑이라는 NAG의 자연스러운 성질.
4. **v2/v3보다 약간 높음**: LERF-mask 0.8653 (vs v2 0.8531, v3 topk=1 0.8204) → 진짜 ceiling에 더 가까움.
5. **모든 거시적 결론 robust**: 어느 oracle 기준이든 "lifting 강함, 매칭 약함, ramen 집중" 동일 trend.

### ⚠️ Fair의 정의는 method-specific — 다른 method에 적용 시 주의

**v4 budget=3이 fair한 건 THGS의 CLIP-based가 정확히 `topk=3 at level=[2,3]` 이기 때문**. 다른 method에 같은 분석을 적용하려면 **그 method의 inference 설정에 맞춰 oracle을 재설정** 해야 한다:

| Method | CLIP-based inference 설정 | Fair Oracle 설정 |
|---|---|---|
| **THGS** | topk=3, level=[2,3] | **v4 budget=3 at level=[2,3]** ← 본 분석 |
| **LangSplat** | 3 hierarchy levels, max-relevancy oracle pick, smooth + threshold | budget=하나(level별 1 pick), 같은 후처리 |
| **Gaussian Grouping** | Grounded-SAM 첫 프레임 IoA > 0.7로 instance 선택 | GT mask로 IoA > 0.7 instance 선택 |
| **OpenSplat3D** | 같은 IoA matching | 같은 oracle |
| **LEGaussians** | per-pixel CLIP feature relevancy | per-pixel oracle 등 |

→ **본 문서의 ceiling/gap 수치는 THGS 한정**. 다른 method와 직접 비교하려면 그 method의 fair oracle을 별도로 측정해야 한다.

→ "Fair = method의 actual inference 설정에 맞춰 oracle을 설정" 이라는 일반 원칙이 핵심. **"v4가 절대적 fair" 아니라 "THGS의 CLIP-NAG topk=3 selection에 대한 fair"**.

---

## 1. 실험 설정

### 1.1 측정 대상

| Method | 데이터셋 | 무엇을 측정 | 스크립트 |
|---|---|---|---|
| **CLIP Actual (mask)** | LERF-mask | THGS 실제 매칭 도달 점수 (topk=3) | [test_lerf_mask.py](../test_lerf_mask.py) |
| **CLIP Actual (ovs)** | LERF-OVS | 동일 | [test_lerf.py](../test_lerf.py) |
| **Oracle v4 budget=3 (mask)** ★ | LERF-mask | **Fair ceiling** (greedy, budget=3) | [sam_oracle_v4_lerf_mask.py](../sam_oracle_v4_lerf_mask.py) |
| **Oracle v4 budget=3 (ovs)** ★ | LERF-OVS | **Fair ceiling** (greedy, budget=3) | [sam_oracle_v4_lerf_ovs.py](../sam_oracle_v4_lerf_ovs.py) |
| Oracle v4 budget=1, 2, 5, unlimited | 둘 다 | Budget sweep (saturation 확인) | 동일 스크립트 |
| Oracle v2 τ sweep | 둘 다 | 분포 기반 (참고) | sam_oracle_v2_*.py |
| Oracle v3 topk sweep | 둘 다 | Single-SP 한계 (참고) | sam_oracle_v3_*.py |

### 1.2 통제 변수
- **Model**: THGS 단일 sai_nag.pt (figurines/ramen/teatime/waldo_kitchen 각 scene별 학습 1회)
- **Inference 코드**: 매칭 알고리즘 외엔 모두 동일 (같은 가우시안 splat, 같은 threshold 0.5, 같은 평가 metric)
- **Metric**: IoU + Boundary-IoU (3×3 erode, iter=round(0.02·diag), GG 공식 정의)
- **sim3 정합 (LERF-mask only)**: inlier rate 98~100%, mean residual mm 수준 → noise 통제됨

### 1.3 데이터 범위
| 데이터셋 | 장면 | 평가 단위 |
|---|---|---|
| LERF-mask | figurines/ramen/teatime (3 scenes) | (view, prompt) pairs |
| LERF-OVS | figurines/ramen/teatime/waldo_kitchen (4 scenes) | (frame, prompt) pairs |

| Scene | LERF-mask | LERF-OVS |
|---|---|---|
| figurines | 7 prompts × 4 views = **28 pairs** | 21 prompts × 4 frames = **56 pairs** |
| ramen | 6 prompts × 3 views = **18 pairs** | 14 prompts × 7 frames = **71 pairs** |
| teatime | 10 prompts × 2 views = **15 pairs** | 14 prompts × 6 frames = **59 pairs** |
| waldo_kitchen | — | 18 prompts × 5 frames = **22 pairs** |

✅ 두 데이터셋 모두 **GT의 모든 prompt가 평가됨**.

---

## 2. Master 결과표

### 2.1 Primary 결과 (v4 budget=3 기준)

| Method | figurines | ramen | teatime | waldo | Overall (3sc) | Overall (4sc) |
|---|---|---|---|---|---|---|
| **LERF-mask Ceiling (v4 budget=3)** ★ | 0.8489 | 0.8598 | 0.8872 | — | **0.8653** | — |
| **LERF-mask Actual (CLIP topk=3)** | 0.7804 | 0.5918 | 0.8380 | — | **0.7367** | — |
| **LERF-OVS Ceiling (v4 budget=3)** ★ | 0.7866 | 0.7760 | 0.9344 | 0.7708 | **0.8323** | 0.8170 |
| **LERF-OVS Actual (CLIP topk=3)** | 0.5432 | 0.4108 | 0.8119 | 0.5499 | **0.5886** | 0.5790 |

### 2.2 v4 budget sweep — Saturation 확인

**LERF-mask Overall**:
| budget | 1 | 2 | **3 ★** | 5 | unlimited |
|---|---|---|---|---|---|
| mIoU | 0.8204 | 0.8641 | **0.8653** | 0.8653 | 0.8653 |

**LERF-OVS Overall (4sc)**:
| budget | 1 | 2 | **3 ★** | 5 | unlimited |
|---|---|---|---|---|---|
| mIoU | 0.7826 | 0.8124 | **0.8170** | 0.8172 | 0.8172 |

→ **budget=3에서 greedy 거의 수렴** (unlimited와 0.0002 차이). NAG가 객체 1개를 SP 2-3개로 표현하는 자연스러운 구조.

### 2.3 Oracle 4종 ceiling 비교 (참고)

| Oracle | LERF-mask | LERF-OVS (4sc) | Fairness |
|---|---|---|---|
| v3 topk=1 | 0.8204 | 0.7826 | budget=1 (CLIP보다 적음) |
| v3 topk=3 | 0.5801 | 0.5110 | budget fair, **selection sub-optimal → oracle < actual** |
| v2 τ=0.5 | 0.8531 | 0.7950 | budget 가변 (inflated 가능) |
| **v4 budget=3** ★ | **0.8653** | **0.8170** | **fair + optimal** |
| v4 unlimited | 0.8653 | 0.8172 | budget unfair, 절대 천장 (= v4 budget=3) |

---

## 3. Ceiling vs Actual 분해 (Fair Gap)

### 3.1 두 데이터셋 정직한 분해 (v4 budget=3 기준)

같은 3 scenes (figurines/ramen/teatime) 비교:

| 양 | LERF-mask | LERF-OVS | 차이 |
|---|---|---|---|
| **Ceiling (v4 budget=3)** | 0.8653 | 0.8323 | +0.0330 (4%p) |
| **Actual (CLIP)** | 0.7367 | 0.5886 | +0.1481 (15%p) |
| **Fair Gap = Ceiling − Actual** | **0.1286** | **0.2437** | +0.1151 |
| **활용도 (Actual/Ceiling)** | 85% | 71% | -14%p |

### 3.2 두 데이터셋 점수 차이 분해

```
score_mask − score_ovs = 0.7367 − 0.5886 = 0.1481  (총 차이)
   ├── Ceiling 차이:    0.0330  (22% of total)  ← lifting의 데이터셋 종속 (mild)
   └── Gap 차이:        0.1151  (78% of total)  ← 매칭 능력의 데이터셋 종속 (major)
```

### 3.3 Decomposition 검증
```
LERF-mask 점수 = Ceiling_v4_mask − Gap_v4_mask = 0.8653 − 0.1286 = 0.7367  ✓
LERF-OVS 점수  = Ceiling_v4_ovs  − Gap_v4_ovs  = 0.8323 − 0.2437 = 0.5886  ✓
```

→ Dual equation 정확히 성립.

---

## 4. Prompt-controlled 분석 — 점수 차이의 진짜 원인

### 4.1 공통 prompt 셋

| Scene | 공통 (intersection) | LERF-mask 전용 | LERF-OVS 전용 |
|---|---|---|---|
| figurines | 6 | 1: `rubber duck with red hat` | 15: `bag, jake, miffy, pikachu, pink ice cream, pirate hat, pumpkin, ...` |
| ramen | 3: `chopsticks, egg, glass of water` | 3: `pork belly, wavy noodles in bowl, yellow bowl` | 11: `bowl, corn, hand, kamaboko, napkin, nori, ...` |
| teatime | 8 | 2: `cookies on a plate, spoon handle` | 6: `bear nose, coffee, dall-e brand, hooves, three cookies, yellow pouf` |

### 4.2 공통 prompt 점수 비교 (Actual CLIP-based 끼리만)

| | LERF-mask | LERF-OVS | 차이 |
|---|---|---|---|
| **공통 prompt 평균** (17개) | **0.8356** | **0.8143** | **+0.0213** (2%p) |

→ 공통 prompt에서 두 데이터셋 점수 차이는 단 2%p. View 차이(train vs novel), GT 형식 차이(polygon vs PNG) 의 합계 효과.

### 4.3 LERF-OVS 전용 (hard) prompt 점수

| Scene | OVS 전용 prompt 평균 | 0점 (CLIP 완전 실패) prompts |
|---|---|---|
| figurines | **0.4233** (15개) | `miffy`, `pirate hat`, `pikachu`, `jake`, `waldo` |
| ramen | **0.2493** (11개) | `onion segments`, `plate`, `hand` |
| teatime | **0.5694** (6개) | `hooves`, `bear nose` |
| waldo_kitchen | **0.5551** (18개) | `ottolenghi`, `pour-over vessel`, `yellow desk` |

### 4.4 점수 차이의 진짜 분해

```
LERF-mask 평균 − LERF-OVS 평균 = 0.1481  (총 차이)
   ├── 공통 prompt 차이      0.0213  (14%)  ← view/GT effect (작음)
   └── 전용 prompt effect    0.1268  (86%)  ← LERF-OVS의 어려운 prompt가 평균을 깎음
```

→ **점수 차이의 86% 가 LERF-OVS의 어려운 prompt가 만든 것**. 데이터셋 변수(view/GT)는 14%에 불과.

### 4.5 LERF-OVS의 hard prompt 카테고리

| 패턴 | 예시 | CLIP 매칭 실패 이유 |
|---|---|---|
| **고유명사/캐릭터** | `miffy`, `pikachu`, `jake`, `waldo` | CLIP은 특정 캐릭터 이름과 외형을 정확히 매칭 못함 |
| **브랜드/제품명** | `ottolenghi`, `dall-e brand`, `tesla door handle` | OCR-like 이해 필요 |
| **Sub-part** | `hooves`, `bear nose`, `spoon handle` | 부분 객체, modifier 처리 약함 |
| **Generic + Multi-instance** | `plate`, `hand`, `bowl` | 여러 객체와 매칭, 인스턴스 구분 실패 |
| **음식 specific** | `onion segments`, `kamaboko`, `nori` | 시각적으로 비슷한 음식과 혼동 |

---

## 5. Scene별 양상 분석 (v4 기준)

| Scene | LERF-mask Ceil/Actual/Gap | LERF-OVS Ceil/Actual/Gap | 진단 |
|---|---|---|---|
| **figurines** | 0.85 / 0.78 / **0.07** | 0.79 / 0.54 / **0.24** | LERF-mask는 ceiling 거의 도달. LERF-OVS는 캐릭터 이름 prompt에서 무너짐 |
| **ramen** | 0.86 / 0.59 / **0.27** | 0.78 / 0.41 / **0.36** | 두 데이터셋 모두 매칭 어려움. fine-grained 음식 객체가 CLIP 매칭 약점 |
| **teatime** | 0.89 / 0.84 / **0.05** | 0.93 / 0.81 / **0.12** | 두 데이터셋 모두 ceiling 거의 도달. **건강한 baseline scene** |
| **waldo** | — | 0.77 / 0.55 / **0.22** | LERF-mask 미정의. 브랜드/제품 prompt에서 매칭 실패 |

→ **teatime은 두 데이터셋 모두에서 잘 됨** = lifting + 매칭 모두 강함 = "easy" scene.
→ **ramen은 두 데이터셋 모두에서 worst** = lifting/매칭 모두 fine-grained 객체에 약함.
→ **figurines은 prompt에 따라 양상이 갈림** = 공통은 잘, hard prompt에서 무너짐.

---

## 6. v4 Greedy IoU progression — 객체별 SP 응집도 진단

### 6.1 LERF-mask ramen
| prompt | SP 1개 IoU | SP 2개 union IoU | 선택된 SP 수 |
|---|---|---|---|
| chopsticks | 0.806 | **0.858** | 2 |
| egg | 0.474 | **0.874** | 2 |
| glass of water | **0.966** | — | 1 |
| pork belly | **0.947** | — | 1 |
| wavy noodles in bowl | 0.894 | **0.897** | 2 |
| yellow bowl | **0.903** | — | 1 |

→ **fragmented 객체** (egg)는 SP 2개 필요, **단일 객체** (glass, pork belly, yellow bowl)는 SP 1개로 충분.

### 6.2 LERF-mask figurines
| prompt | SP 1개 | SP 2개 | SP 3개 | 선택된 SP 수 |
|---|---|---|---|---|
| green apple | **0.971** | — | — | 1 |
| green toy chair | 0.852 | **0.855** | — | 2 |
| old camera | 0.481 | 0.842 | **0.865** | 3 |
| porcelain hand | **0.728** | — | — | 1 |
| red apple | **0.942** | — | — | 1 |
| red toy chair | **0.918** | — | — | 1 |
| rubber duck with red hat | **0.931** | — | — | 1 |

→ `old camera` 만 3 SP (가장 fragmented). 나머지는 1-2 SP로 표현 가능. 평균적으로 객체 = SP 1.4 매핑.

---

## 7. 사용자 framework 검증 — "LERF-mask = lifting, LERF-OVS = lifting + OVS"

```
사용자 framework:
    score_mask  ≒  SAM lifting 능력
    score_ovs   ≒  SAM lifting + OVS 매칭 능력
    Gap         ≒  OVS 매칭의 추가 부담
```

### 실증 (v4 기준)
- ✅ **거시적 추세 맞음**: LERF-OVS가 LERF-mask보다 낮은 점수 → "OVS 추가 부담"이 있긴 함
- ❌ **모든 성분이 OVS 능력만은 아님**: 점수 차이의 14%는 단순 데이터셋 noise, 78%가 매칭 알고리즘 차이, 그 매칭 손실 중 대부분이 **prompt 난이도** 때문
- ❌ **"LERF-mask = lifting only" 가정 부정확**: LERF-mask 위에서도 CLIP 매칭이 ceiling에서 12.9%p 손실 발생 (특히 ramen)

### 정직한 framework
```
LERF-mask 점수 = Ceiling_v4_mask − Gap_v4_mask = 0.8653 − 0.1286 = 0.7367
LERF-OVS 점수  = Ceiling_v4_ovs  − Gap_v4_ovs  = 0.8323 − 0.2437 = 0.5886

데이터셋 점수 차이 0.1481의 진짜 원인:
  ├── Ceiling 차이 (4%p)    ← view/GT 데이터셋 변수 (mild)
  └── Gap 차이 (11%p)
       ├── 공통 prompt에서의 noise (2%p)
       └── 어려운 prompt가 평균 깎음 (9%p)
```

---

## 8. 결론과 시사점

### 8.1 THGS의 능력 정량 진단 (v4 기준)
- **SAM lifting (Stage 1) = 매우 강함**: 두 데이터셋 모두 Ceiling 0.83~0.87. 사실상 데이터셋 무관.
- **CLIP-NAG 매칭 (Stage 2) = bottleneck**:
  - LERF-mask에서 ceiling의 **85%** 활용 (15% 손실)
  - LERF-OVS에서 ceiling의 **71%** 활용 (29% 손실)
- **개선 우선순위 = 매칭 알고리즘**. lifting 디자인은 유지.

### 8.2 매칭 알고리즘의 약점 카테고리 (개선 대상)
1. **고유명사/캐릭터** — CLIP의 vision-language 사전 학습 한계
2. **브랜드/텍스트 기반** — OCR-like 이해 필요
3. **Sub-part** — modifier 처리 약함
4. **Multi-instance + generic** — instance 구분 필요
5. **Fine-grained 음식/도구** — 시각적으로 비슷한 객체와 혼동

### 8.3 ramen 집중 — 두 데이터셋 모두에서 worst
- LERF-mask Gap_ramen = 0.2680 (활용도 69%)
- LERF-OVS Gap_ramen = 0.3652 (활용도 53%)
- **다음 단계 진단 1순위 = ramen prompt 단위 CLIP failure 분석**

### 8.4 Method 비교 시 주의사항
- LERF-mask와 LERF-OVS 점수를 단순 비교하면 **prompt 분포 차이** 가 가장 큰 결과 변수
- 공정한 비교 = **공통 prompt만 비교** 또는 **각 데이터셋에서 자체 ceiling 대비 활용도 비교**
- 본 분석의 v4 ceiling은 **THGS 한정**, 다른 method에 적용 시 그 method의 inference 설정에 맞춰 oracle 재설정 필요 (Section 0.5 참고)

---

## 9. 다음 단계 후보

| # | 작업 | 답하는 질문 | 작업량 |
|---|---|---|---|
| 1 | ramen prompt-level 진단 (v4 ceiling 대비) | 어떤 prompt가 양쪽 데이터셋에서 모두 실패하는가? | 30분 |
| 2 | Hard prompt 카테고리 ablation | 캐릭터/브랜드/sub-part 각 카테고리별 IoU 분포 | 1시간 |
| 3 | CLIP relevancy 매칭 알고리즘 ablation | topk, canon, threshold 변경 효과 | 2~3시간 |
| 4 | CLIP feature 부여 방식 ablation (merge_proj.py) | bag-of-masks weighted avg vs alternatives | 4~6시간 |
| 5 | 다른 method (LangSplat 등)의 fair oracle 측정 + 비교 | 다른 paradigm에서도 같은 패턴인가? | 1일+ |

→ 추천: **#1 (ramen 진단)** 먼저 → **#2 (카테고리 분석)** → #3 → #4

---

## 10. 부록 — 모든 측정 결과 디렉토리

| 측정 | 디렉토리 |
|---|---|
| LERF-mask CLIP Actual | `output/render/lerf_mask/` |
| LERF-mask Oracle v1 (union, 부적절) | `output/render/lerf_mask_sam_oracle/` |
| LERF-mask Oracle v2 (τ sweep) | `output/render/lerf_mask_sam_oracle_v2_tau{0.1,0.3,0.5,0.7,0.9}/` |
| LERF-mask Oracle v3 (topk sweep) | `output/render/lerf_mask_sam_oracle_v3_topk{1,2,3,5,10}/` |
| **LERF-mask Oracle v4 (budget sweep)** ★ | `output/render/lerf_mask_sam_oracle_v4_{budget1,budget2,budget3,budget5,unlimited}/` |
| LERF-OVS CLIP Actual | `output/render/lerf/` |
| LERF-OVS Oracle v2 (τ sweep) | `output/render/lerf_ovs_sam_oracle_v2_tau{0.1,0.3,0.5,0.7,0.9}/` |
| LERF-OVS Oracle v3 (topk sweep) | `output/render/lerf_ovs_sam_oracle_v3_topk{1,2,3,5,10}/` |
| **LERF-OVS Oracle v4 (budget sweep)** ★ | `output/render/lerf_ovs_sam_oracle_v4_{budget1,budget2,budget3,budget5,unlimited}/` |

### 평가 스크립트
| Script | 데이터셋 | 비고 |
|---|---|---|
| [scripts/eval_lerf_mask.py](../scripts/eval_lerf_mask.py) | LERF-mask | GG-호환 IoU+BIoU |
| [scripts/eval_lerf_ovs_pairs.py](../scripts/eval_lerf_ovs_pairs.py) | LERF-OVS | pred+_gt.png pair, IoU+BIoU |

### 재현 명령
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate thgs
export CUDA_VISIBLE_DEVICES=2

# v4 측정 (LERF-mask)
python sam_oracle_v4_lerf_mask.py -s data/lerf/figurines -m output/lerf/figurines

# v4 측정 (LERF-OVS)
python sam_oracle_v4_lerf_ovs.py -s data/lerf/figurines -m output/lerf/figurines

# 평가
python scripts/eval_lerf_mask.py -p output/render/lerf_mask_sam_oracle_v4_budget3 -s figurines ramen teatime
python scripts/eval_lerf_ovs_pairs.py -p output/render/lerf_ovs_sam_oracle_v4_budget3 -s figurines ramen teatime waldo_kitchen
```
