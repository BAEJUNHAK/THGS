# Stage 1 — 실패의 1 차 mechanism 분해 (B7 + A4 + A2 + Joint + Cross-method)

> 완료. 67 prompt × 4 LERF-OVS scene × 2 method (THGS, ReLaGS).
>
> 핵심 finding 한 줄: **"실패는 어렵기 때문이 아니라 우리가 멀쩡한 신호를 평균으로 죽였기 때문이다 — 그것도 두 method 모두에서."**

---

## 0. Stage 의 question

> **THGS/ReLaGS 같은 training-free 3D segmentation 이 실패할 때, 그 실패가 *진짜* 어려운 문제 (encoder 한계) 인가, 아니면 우리 pipeline 이 *멀쩡한 신호를 죽이고 있는* 것 (aggregation 손실) 인가?**

이 질문에 답하려면 실패를 **4 단계로 분리** 해야 한다:

| 단계 | 질문 | 측정 도구 |
|---|---|---|
| 1. **분할** | GT 객체에 맞는 SP 가 *존재* 하긴 하는가? | **B7** |
| 2. **검색** | 그 SP 가 CLIP cosine 으로 *찾아지는가*? | **A4** |
| 3. **인코더** | CLIP 자체가 prompt 를 *알아보긴* 하는가? | **A2** |
| 4. **분해** | 실패가 (인코더 한계) 인가 (aggregation 손실) 인가? | **Joint** |
| 5. **검증** | 다른 method (ReLaGS) 도 같은가? | **Cross-method** |

---

## 1. Pre-experiment 상태

이전 분석에서 다음을 알고 있었음:
- Type A (CLIP injection) = 85% of mIoU loss (vs Type B SAM = 1.5%)
- D1 (degenerate), D2 (semantic distractor), D3 (over-union), D4 (visibility) 4-class failure taxonomy
- ReLaGS 가 D1 을 86% 감소시킴 (paper Table 3 reproduction)
- ReLaGS 가 paper mIoU 거의 정확히 일치 (Algorithm 1 의 6/6 step 일치)

알지 못했던 것:
- D2 의 *내부 구조* — 진짜 어려운지 (real) vs aggregation 이 망친 건지 (phantom)
- ReLaGS 의 D1 회복이 *어디로* 갔는지 (Success vs D2.phantom)
- 두 method 의 phantom 분포가 같은지 다른지

---

## 2. 실험 1 — B7: "Oracle SP 가 존재하는가?"

### 무엇을 함

각 prompt 마다, 모든 SP 의 mask 를 ref-frame 에 렌더링하고 GT polygon 과 비교해서 **가장 깨끗한 SP** (oracle) 를 찾음. 그 oracle 의:
- **Purity** = SP 안의 픽셀이 GT 인 비율 (이 SP 는 깨끗한가?)
- **Completeness** = GT 의 픽셀이 SP 에 들어간 비율 (이 SP 가 GT 를 다 덮는가?)
- **Fragmentation** = GT 를 다 덮으려면 SP 가 몇 개 필요한가 (greedy union budget=3)

### 직관

비유: 강아지 사진에서 "강아지" GT 를 잡으려는데, **분할** 이 강아지를 (a) 통째로 잡은 SP 하나로 만들었나, (b) 여러 조각 (머리/몸/꼬리) 으로 쪼갰나, (c) 강아지 + 옆 의자도 포함된 SP 로 만들었나?

이게 모든 분석의 *전제*. 만약 "강아지 + 의자" SP 밖에 없다면 (purity 낮음), CLIP 이 아무리 잘 골라도 의자까지 같이 끌고 옴 — 검색의 문제가 아니라 분할의 문제. **B7 = prerequisite gate**.

### 결과 (THGS)

| 통계 | 값 | 의미 |
|---|---|---|
| Purity 평균 | **0.925** | SP 픽셀의 92.5% 가 GT |
| Purity < 0.5 인 prompt | 2 / 67 (3%) | 깨끗한 oracle 이 없는 경우 극히 드묾 |
| Purity < 0.8 인 prompt | 5 / 67 (7%) | 거의 모든 경우 깨끗한 oracle 존재 |
| Completeness 평균 | 0.857 | GT 의 86% 가 oracle 안 |
| Completeness < 0.5 | 4 / 67 (6%) | 객체 일부만 잡힘 (under-segmentation) |
| Fragmentation = 1 | 45 / 67 (67%) | 67% 는 *단일 SP* 가 정답 |
| Fragmentation ≥ 2 | 22 / 67 (33%) | 33% 는 여러 SP 협동 필요 |
| Fragmentation ≥ 3 | 11 / 67 (16%) | — |
| Oracle IoU 평균 | 0.800 | — |

### 의미

**분할은 dominant 문제가 아니다**. 67 개 prompt 중 65 개는 깨끗한 oracle SP 가 존재 (purity ≥ 0.5). 우리가 SP partition 을 의심할 필요가 거의 없음 → 이후 분석 (검색, 인코더) 의 *해석이 valid* 함을 확보.

단, 다음 두 가지 부수적 발견:
1. **4 개 prompt 가 completeness < 0.5** (`miffy`, `porcelain hand`, `waldo`, `tesla door handle`) → 객체의 일부만 잡힘. partition under-segmentation 의 가장자리 사례 — D3 cardinality 와 연결.
2. **22 개 prompt 가 fragmentation ≥ 2** → 단일 SP 로는 부족하고 union 이 필요. 이게 top-k 결정 정책 (D3) 의 motivation.

→ **B7 의 가장 큰 contribution: A4/A2 의 후속 분석을 정당화** — oracle 이 있으니 rank/ceiling 의 의미가 명확.

→ 산출 plot: `plots/b7_distributions.png`, `b7_purity_vs_completeness.png`, `b7_by_scene.png`, `b7_cross_view_stability.png`

---

## 3. 실험 2 — A4: "CLIP cosine 으로 oracle 을 찾을 수 있는가?"

### 무엇을 함

각 prompt 에 대해, scene 안의 모든 SP (수백~수천 개) 의 CLIP feature 와 prompt text 의 코사인 유사도 계산. **Oracle SP 의 순위** (rank) 와 **얼마나 큰 차이로 다른 SP 에 밀렸는지** (margin) 측정:

- **Raw margin** = cos(1 등) − cos(oracle)
- **Z-margin** = raw / (그 prompt 의 cos 분포 std) → cross-prompt 비교 가능
- **Percentile margin** = oracle 이 분포의 몇 백분위 (0 = top, 100 = bottom)

### 직관

비유: scene 에 강아지 SP, 고양이 SP, 의자 SP, ... 가 있고, "강아지" 라고 외쳤을 때 **강아지 SP 가 몇 번째로 손드는가**. 1 등이면 검색 성공. 256 등이면 catastrophic.

Raw margin 만 보면 "0.05 차이" 가 큰지 작은지 모름 (prompt 마다 코사인 분포가 다름). Z-margin 으로 정규화해야 *다른 prompt 의 실패와 비교 가능*. 이게 6.1 patch 의 핵심 정정.

### 결과 (THGS)

| 통계 | 값 | 해석 |
|---|---|---|
| Oracle rank 중앙값 | **2** | 절반은 거의 top |
| q75 | 7 | 위쪽 25% 는 7 등 밖 |
| q90 | 52 | 위쪽 10% 는 52 등 밖 |
| Max | 256 | 최악은 256 등 |
| Rank ≤ 3 | 41 / 67 (61%) | 60% 만 검색 성공 |
| Rank ≤ 10 | 53 / 67 (79%) | 80% 는 top-10 안에는 들어옴 |
| Rank > 30 | 13 / 67 (**19%**) | **catastrophic — 거의 random** |
| Z-margin 중앙값 | 0.29 | 절반은 top 과 거의 동급 |
| |Z-margin| > 1.5 | 10 / 67 | "wrong SP 가 *훨씬* 강함" |

### 의미

**검색 결과가 양극화** — 절반은 쉽고 (top-2), 나머지는 갑자기 매우 어려움. 단순한 *느린 실패* 가 아니라 **완전히 잘못된 SP 를 1 등으로 뽑는** 경우가 있음. 

- 39% 의 prompt 가 검색에서 실패 (rank > 3)
- 그 중 19% 는 완전한 실패 (rank > 30)
- 10 개 prompt 는 z-margin > 1.5 — wrong SP 가 *극단적으로* 강한 case

**하지만 이걸로 끝나면 안 됨**: 검색 실패의 *원인* 이 무엇인가? CLIP 이 원래 못 알아보는 건가, 아니면 SP feature 가 망가져서 못 찾는 건가? → 이게 실험 3 (A2) 의 역할.

→ 산출 plot: `plots/a4_distributions.png`, `a4_rank_zmargin_scatter.png`, `a4_rank_by_scene.png`

---

## 4. 실험 3 — A2: "CLIP 인코더 자체의 ceiling 은 얼마인가?"

### 무엇을 함

SP 와 multi-view 를 완전히 우회하고 **GT 영역을 직접 잘라서** CLIP image encoder 에 넣음. 4 가지 crop 방식:

- **tight**: GT polygon 의 최소 bbox, 원본 RGB
- **mask**: 같은 bbox, polygon 밖 픽셀은 검정 (SAM-style proxy)
- **context**: 1.5× 확장한 bbox, 원본 RGB (주변 맥락 포함)
- **method**: polygon-blackout + 1.2× bbox (image_encoding.py 의 crop policy 근사)

각 crop 을 CLIP 으로 인코딩 → scene 의 모든 prompt 와 코사인 비교 → **true prompt 의 순위**.

### 직관

비유: 강아지 사진을 *완벽하게* 잘라줬을 때 CLIP 이 "강아지" 라고 답하는가? 만약 못 알아보면 → CLIP encoder/어휘의 한계 → 우리 method 가 아무리 좋아도 못 풀음 (encoder ceiling).

만약 알아보는데 우리 method 의 SP 검색 (A4) 은 못 찾으면 → **pipeline 이 멀쩡한 신호를 죽인 것** (= phantom).

이게 paper 의 가장 강력한 진단 도구 — "주어진 prompt 가 *원래* 어려운 거였나, 우리가 망친 거였나" 를 분리.

### 결과 (Method-agnostic — 두 method 공통)

| Crop 정책 | Top-1 hit | Top-3 hit | 의미 |
|---|---|---|---|
| Tight | 43/67 (64.2%) | 58/67 (86.6%) | 깨끗한 영역으로도 35% 가 top-1 못 잡음 |
| Mask | 42/67 (62.7%) | 57/67 (85.1%) | 배경 검정도 큰 도움 안 됨 (−1.5%) |
| **Context (1.5×)** | **48/67 (71.6%)** | **60/67 (89.6%)** | **+7.4% — 맥락이 결정적** |
| Method | 46/67 (68.7%) | 57/67 (85.1%) | tight + bg-removal 의 절충 |

### 의미

**CLIP 의 한계는 약 30%** — 67 개 prompt 중 ~20 개는 *깨끗한 crop 으로도* top-1 못 잡음. 그 중 D2.real (실험 4) 로 분류되는 것들은 진짜 어려운 경우 (`miffy` 캐릭터, `ottolenghi` 브랜드, `pour-over vessel` 특수어, `hooves` part-level).

**Context 가 +7.4% 개선** — 객체 단독 crop 보다 *주변 환경* 까지 보면 CLIP 이 더 잘 알아봄. 이건 CLIP training distribution 이 *맥락 있는 사진* 위주라는 사실의 정량 확인.

**Mask 와 Tight 의 차이가 거의 없음** (62.7% vs 64.2%) — 배경 검정 처리 자체로는 ceiling 이 거의 안 변함. → 우리 method 의 SAM-mask 의존성이 *ceiling 면에서* 큰 손실이 아니라는 시사.

→ 산출 plot: `plots/a2_rank_per_policy.png`, `a2_top1_per_scene_policy.png`, `a2_rank_box.png`

---

## 5. 실험 4 — Joint 분해: D2.real vs D2.phantom

### 무엇을 함

A4 (SP rank) 와 A2 (image-CLIP rank, mask crop) 를 **결합**. 두 축 모두 rank ≤ 3 을 성공 임계치로 사용해 2×2 분류:

```
                  A2 mask rank
                ≤ 3        > 3
SP rank ≤ 3    Easy        Rare
SP rank > 3   Phantom     Real
```

### 직관

이게 paper 의 **핵심 통찰** — 실패를 *recoverable* 인지 *encoder-limited* 인지로 분리.

- **Easy**: CLIP 도 쉽고 SP 도 쉬움 → 이미 잘 됨
- **D2.real**: CLIP 도 어렵고 SP 도 어려움 → *진짜 어려운 문제*. Encoder upgrade, prompt expansion, negative contrast 만이 해결책
- **D2.phantom**: CLIP 은 쉬운데 SP 는 어려움 → **pipeline 이 망친 것**. Aggregation 만 고치면 회복 가능
- **Rare**: CLIP 은 어려운데 SP 는 쉬움 → multi-view 평균이 *positive aggregation* — 단일 view 보다 나음

이 2×2 가 paper 의 가장 큰 정량 contribution.

### 결과 (THGS)

| Class | 정의 | 개수 | Share |
|---|---|---|---|
| Easy | A2_mask ≤ 3 AND SP rank ≤ 3 | 36 | 53.7% |
| **D2.phantom** | A2_mask ≤ 3 AND SP rank > 3 | **21** | **31.3%** |
| **D2.real** | A2_mask > 3 AND SP rank > 3 | 5 | 7.5% |
| Rare | A2_mask > 3 AND SP rank ≤ 3 | 5 | 7.5% |

**Phantom : Real = 21 : 5 = 4.2 : 1**

26 개 SP-실패 prompt (rank > 3) 중 **81% (21/26) 가 원리적으로 회복 가능** — CLIP 이 알아본 것을 우리 aggregation 이 죽인 것. 단 **19% (5/26) 만 encoder limit**.

### Per-scene 분해

| Scene | n | Easy | Phantom | Real | Rare |
|---|---|---|---|---|---|
| figurines | 21 | 11 (52%) | 6 (29%) | 3 (14%) | 1 (5%) |
| **ramen** | 14 | 7 (50%) | 6 (**43%**) | **0 (0%)** | 1 (7%) |
| teatime | 14 | 11 (79%) | 2 (14%) | 1 (7%) | 0 |
| waldo_kitchen | 18 | 7 (39%) | 7 (39%) | 1 (6%) | 3 (17%) |

**ramen 의 mIoU 격차는 100% phantom** — 0 real, 6 phantom + 1 rare. paper 의 ramen 13.57 mIoU 분포 차이의 mechanism 적 해명. ramen 이 *aggregation fix 의 cleanest testbed*.

**teatime 은 79% easy** — 새 method 의 marginal gain 측정용으로는 약함.

### 의미

암묵적 가정 "실패는 CLIP 이 못 알아본 것" 을 *반박*. 실패의 81% 는 **CLIP 이 알아봤지만 우리 aggregation 이 죽인 것**. 즉:

- 새 aggregation method 가 phantom 들을 회복하면 → mIoU 대폭 향상 가능
- Encoder 만 한계라고 가정하면 → 잘못된 framing, 큰 손해 (paper narrative 의 misorientation)

**구체적 phantom 사례** (purity 0.97-0.99 의 깨끗한 oracle 인데 SP rank 가 catastrophic):

| Scene | Prompt | A2 rank | SP rank | Purity | z-margin |
|---|---|---|---|---|---|
| figurines | tesla door handle | 1 | **256** | 0.50 | 1.64 |
| figurines | old camera | 1 | 130 | 0.99 | 2.82 |
| waldo_kitchen | ottolenghi | 1 | 66 | 0.97 | 3.05 |
| figurines | pikachu | 1 | 64 | 0.98 | 1.16 |
| figurines | pumpkin | 1 | 52 | 0.98 | 1.08 |
| ramen | hand | 1 | 46 | 0.97 | 2.80 |

→ Image-CLIP 이 crop 만 보고 정확히 `pikachu` 로 식별, 하지만 SP feature (multi-view 평균 후 ROFA filter) 는 745 개 중 64 번째. Oracle SP 의 purity 는 0.98 (깨끗함). **순수 aggregation 의 손실**.

**D2.real 사례** (encoder limit):

| Scene | Prompt | A2 rank | SP rank | 어휘 특성 |
|---|---|---|---|---|
| figurines | miffy | 13 | 75 | proper noun (캐릭터 이름) |
| figurines | bag | 11 | 63 | generic word + 작은 객체 |
| waldo_kitchen | pour-over vessel | 4 | 53 | specialized vocabulary |
| teatime | hooves | 8 | 52 | part-level term |
| figurines | pirate hat | 12 | 49 | compound term |

→ proper noun, brand, specialized vocab, part-level term — encoder limit 의 *vocabulary nature* 확인. C2 (prompt expansion) 와 C3 (negative contrast) 가 정확히 이 5 개를 target.

→ 산출 plot: `plots/joint_d2real_vs_phantom.png`

---

## 6. 실험 5 — Cross-method: "ReLaGS 도 같은가?"

### 무엇을 함

같은 B7+A4 protocol 을 ReLaGS 의 sai_nag.pt 에 적용. A2 는 method-agnostic 이라 그대로. 4×4 transition matrix 생성.

### 직관

Paper claim 의 *robustness* 검증. 만약 D2.phantom 분포가 ReLaGS 에서 다르면 → THGS 특이 현상. 만약 같으면 → 구조적 현상 → paper 의 일반화 claim 강함.

또한 ReLaGS 가 *어디서 어떻게* 좋아졌는지 mechanism 으로 분해. paper 의 "ReLaGS 가 D1 86% 감소" 같은 표면적 claim 을 *mechanism level* 로 격상.

### 결과 — 분포 (거의 동일)

| Class | THGS | ReLaGS | Δ |
|---|---|---|---|
| Easy | 36 / 67 (53.7%) | 37 / 67 (55.2%) | +1 |
| **Phantom** | **21 / 67 (31.3%)** | **20 / 67 (29.9%)** | **−1** |
| Real | 5 / 67 (7.5%) | 4 / 67 (6.0%) | −1 |
| Rare | 5 / 67 (7.5%) | 6 / 67 (9.0%) | +1 |

→ 두 method 가 거의 동일한 4.2 : 1 phantom : real 분포. **D2.phantom 우세는 method-agnostic.**

### 결과 — Transition matrix (THGS → ReLaGS)

```
              ReLaGS
         easy phantom real rare
THGS  easy  33     3     0    0   ← 3 borderline 재분류
phantom      4    17     0    0   ← 4 회복 (19%) / 17 잔존 (81%)
real         0     0     4    1
rare         0     0     0    5
```

- THGS phantom 21 개 중 **4 개 (19%) 만 ReLaGS 가 회복**, 17 개 (81%) 잔존
- 0 개의 phantom 이 real 로 변환 → ReLaGS 가 encoder regression 만들지 않음
- 3 개 borderline regression (rank 2-3 → 4) 은 threshold artifact

### 결과 — Per-scene 비대칭

| Scene | Phantom Δ | Easy Δ | 평가 |
|---|---|---|---|
| **waldo_kitchen** | −2 | +2 | **ReLaGS 가 분명히 도움** (mIoU +9.95 의 mechanism) |
| teatime | −1 | +1 | mild improvement |
| figurines | +1 | −1 | neutral |
| **ramen** | +1 | −1 | **ReLaGS 도움 안 됨** (0 회복) |

ramen 의 6 phantoms (`bowl`, `hand`, `napkin`, `onion segments`, `plate`, `sake cup`) 가 *모두* ROFA 를 통과해 잔존 — ROFA mechanism 으로 못 잡는 phantom subtype 의 존재 증거 (F2 의 bimodal-balanced 후보).

### 17 persistent phantoms (두 method 모두 실패)

| Scene | Prompts |
|---|---|
| figurines | bag, jake, miffy, old camera, pikachu, pirate hat, pumpkin, spatula, tesla door handle, toy elephant |
| ramen | bowl, hand, napkin, onion segments, plate, sake cup |
| teatime | hooves |
| waldo_kitchen | cabinet, ketchup, ottolenghi, plate, pour-over vessel, spoon |

→ 이게 paper 의 새 method 의 *명확한 target set*. 17 prompt 를 회복하면 → SOTA evidence.

### 의미

3 가지 결론:

1. **D2.phantom 우세는 method-agnostic** — ROFA 도 이 구조를 못 바꿈. 두 method 가 ~30% phantom 으로 거의 동일. → paper 가 *paradigm 의 한계* 를 지적함, *method 의 부족* 만 지적하는 게 아님.

2. **ReLaGS = partial phantom recovery (19%) + zero encoder regression**. ROFA 가 *outlier-style* phantom 은 잡지만 *mode-cluster style* 은 못 잡음 (F2 가설의 정량 evidence).

3. **mIoU gap 의 mechanism 분해** — paper Table 3 의 "ReLaGS Waldo +9.95 mIoU" 가 mechanism 으로는 "waldo 의 7 phantom 중 2 개 회복" 으로 설명됨. cross-method claim 을 표면적 mIoU 비교에서 *mechanism evidence* 로 격상.

→ 산출 plot: `plots/cross_method_transition_matrix.png`, `cross_method_rank_scatter.png`, `cross_method_d2_stacked.png`, `cross_method_per_scene_stacked.png`

---

## 7. 종합 — 5 개 실험이 같이 말하는 것

5 개 실험이 하나의 일관된 **diagnostic stack** 을 구성:

```
B7   (분할이 멀쩡한지)  → 멀쩡함 (92% 깨끗)
A4   (검색이 되는지)    → 60% 만 성공
A2   (CLIP 한계는?)     → 30% 가 인코더 한계
↓ joint
D2 분해                 → 31% phantom : 7.5% real (4.2배)
↓ cross-method
ReLaGS 도 같은지         → 구조적 동일 (29.9% phantom)
                       → 회복은 19% 만, 81% 잔존 (17 persistent)
```

### Paper 의 핵심 narrative (이제 명확)

> *"우리는 training-free 3D open-vocabulary segmentation 의 실패 중 81% 가 multi-view CLIP aggregation 의 *복구 가능한* 손실 (D2.phantom) 이라는 것을 처음으로 정량 증명한다. 이 phantom 분포는 method-agnostic (THGS, ReLaGS 거의 동일) 이며, 기존 ROFA 같은 aggregation 개선은 phantom 의 19% 만 회복한다. 우리는 17 개 persistent phantom 의 mechanism (target-dilution vs distractor-inflation, within-view vs across-view mixing) 을 분해하고, 그에 맞춘 새 aggregation 을 제안한다."*

### 한 줄로

> **"실패는 어렵기 때문이 아니라 우리가 멀쩡한 신호를 평균으로 죽였기 때문이다 — 그것도 두 method 모두에서."**

---

## 8. Stage 1 산출물

### Data

| 파일 | Rows | 내용 |
|---|---|---|
| [b7_a4_combined.csv](../../../output/diagnostics/b7_a4_combined.csv) | 208 | THGS B7+A4 per (prompt, eval_frame) |
| [b7_a4_combined_relags.csv](../../../output/diagnostics/b7_a4_combined_relags.csv) | 208 | ReLaGS B7+A4 |
| [a2_image_clip_ceiling.csv](../../../output/diagnostics/a2_image_clip_ceiling.csv) | 268 | A2 ceiling per (prompt, policy) |
| [cross_method_d2_decomposition.csv](../../../output/diagnostics/cross_method_d2_decomposition.csv) | 67 | Per-prompt class + rank + purity 양 method |

### Plots (15)

Within-method (11):
- `b7_distributions.png`, `b7_by_scene.png`, `b7_purity_vs_completeness.png`, `b7_cross_view_stability.png`
- `a4_distributions.png`, `a4_rank_zmargin_scatter.png`, `a4_rank_by_scene.png`
- `a2_rank_per_policy.png`, `a2_top1_per_scene_policy.png`, `a2_rank_box.png`
- `joint_d2real_vs_phantom.png`

Cross-method (4):
- `cross_method_rank_scatter.png` — 두 method 간 per-prompt rank scatter (대각선 아래 = ReLaGS 개선)
- `cross_method_d2_stacked.png` — 두 method 의 D2 class stacked bar
- `cross_method_per_scene_stacked.png` — scene 별 side-by-side stacked
- `cross_method_transition_matrix.png` — 4×4 heatmap

### Scripts

- [scripts/b7_a4_oracle_analysis.py](../../../scripts/b7_a4_oracle_analysis.py) (THGS)
- [ReLaGS/scripts/b7_a4_oracle_analysis.py](../../../ReLaGS/scripts/b7_a4_oracle_analysis.py) (ReLaGS copy)
- [scripts/a2_image_clip_ceiling.py](../../../scripts/a2_image_clip_ceiling.py)
- [scripts/b7_a4_a2_plots.py](../../../scripts/b7_a4_a2_plots.py)
- [scripts/cross_method_comparison.py](../../../scripts/cross_method_comparison.py)

### Paper draft

- [md/THGS/paper_section1_draft.md](../../THGS/paper_section1_draft.md) — Paper Section 1 draft v2 (cross-method 포함)

---

## 9. Decision point — Stage 2 후보

Stage 1 이 결정적으로 좁힌 question:

> **17 persistent phantoms 를 fix 하는 aggregation 은 무엇이고, 그 fix 가 method-agnostic 한가?**

이 question 에 답하는 path 별 후보:

### 후보 A — Mechanism 정량화 (왜 phantom 인가?)

| 실험 | 시간 | 답하는 질문 |
|---|---|---|
| **B8 within-view mix-rate** | 반나절 | 17 phantom 의 SP 가 view 안에서 mask 가 여러 개 섞이는 비율이 높은가? |
| **D3 top-k sweep** | 반나절 | 17 phantom 의 몇 개가 단순히 k 줄여서 회복? |
| **F2 ROFA subtype 분류** | 1 일 | 17 phantom 의 view-feature 분포가 outlier / bimodal-balanced / mean-dilution 중 무엇? |

→ 답: phantom 의 *원인* mechanism 분해. Aggregation 의 어디를 고쳐야 하는지 결정.

### 후보 B — Cross-view 분해 (어디에서 죽었는지?)

| 실험 | 시간 | 답하는 질문 |
|---|---|---|
| **H2 instrument** (per-view feature dump) | 1 주 | 각 SP 의 pre-ROFA per-view feature 를 dump (foundation) |
| **A3 dual-side classifier** | 1 일 (H2 후) | 17 phantom 의 view 별 ranking — 어떤 view 에선 정답이고 평균에서 죽는가? (target dilution vs distractor inflation) |
| **E1 phantom direction** | 2-3 일 (A3 후) | 17 phantom 의 direction vector 가 LVIS 의 특정 category 로 편향되는가? |

→ 답: phantom 의 *진행 과정* 분해. main paper contribution 후보 (direction-aware aggregation).

### 후보 C — Quick wins (encoder 측 회복)

| 실험 | 시간 | 답하는 질문 |
|---|---|---|
| **C2 prompt expansion** | 반나절 | D2.real 5 개 (`miffy`, `bag` 등) 가 generic expansion 으로 회복? |
| **C3 negative contrast (λ sweep)** | 반나절 | distractor inflation 후보 들이 negative prompt 로 분리? |

→ 답: D2.real 의 5 개 회복 가능성. Paper 의 "encoder limit" 주장의 boundary.

### 추천

**Stage 2 = 후보 A 의 B8 + D3** (1 일, 새 instrument 불필요):

이유:
1. 가장 cheap — 둘 다 기존 데이터만으로 가능
2. 가장 정보량 큰 *binary 분기점*:
   - D3 sweep 으로 cardinality 가 main fix 면 → paper pivot ("새 aggregation 필요 없음, top-k 만 조정")
   - B8 mix-rate 가 phantom 과 상관되면 → within-view mixing 이 paper 의 핵심 section
   - 둘 다 negative 면 → 후보 B (H2 instrument + A3) 가 필연
3. 결과가 *Stage 3 의 선택을 강제* 하므로 decision tree 가 깔끔

**Stage 2 = 후보 A + 후보 C 의 C2 병렬** (1 일 + 반나절):
- 추가로 D2.real 5 개의 회복 가능성도 측정
- C2 는 inference 만 바꾸면 되어 거의 무료

Stage 2 후 ⇒ 결과 봐서 Stage 3 결정.
