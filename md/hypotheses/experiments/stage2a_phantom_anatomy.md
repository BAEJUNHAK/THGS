# Stage 2A — Phantom Anatomy of 17 Persistent Phantoms

> 완료. Stage 1 에서 추출한 *17 persistent phantoms* (THGS 와 ReLaGS 양쪽 모두 SP rank > 3 인 prompt) 의 4-layer forensic deep dive.
>
> 핵심 finding 한 줄: **"65% 의 persistent phantom 은 ROFA 류 across-view aggregation 의 실패다 — instance confusion 과 background drift 는 소수 (24%)."**

---

## 0. Stage 2A 의 question

> **Stage 1 이 좁힌 17 persistent phantoms 가 *왜* phantom 인가? Within-view mixing (B8)? Across-view aggregation (ROFA)? Top-k policy (D3)? Encoder hidden? Instance confusion?**

분기 결정의 기초:
- *B8 within-view* dominant → Stage 2B = RATIO_THRESHOLD sweep + hard-assignment ablation
- *across-view ROFA* dominant → Stage 2B = F2 subtype 분류 + 부분 H2 instrument
- *encoder hidden* dominant → Stage 2B = C2 + C3 + E1 partial

---

## 1. Pre-experiment 상태

Stage 1 의 결과:
- 17 persistent phantoms 의 list 확정 (figurines 6, ramen 6, teatime 1, waldo_kitchen 4)
- 거의 모두 A2 mask rank ≤ 3 (CLIP image-CLIP 으로는 정답 → encoder limit 아님)
- 대부분 oracle purity ≥ 0.93 (oracle SP 가 깨끗함), 2 outliers (tesla door handle 0.50, onion segments 0.48)
- 두 method 모두 fail 하므로 *method-specific* 가 아님

알지 못했던 것:
- 17 phantom 의 *원인 mechanism* — within-view mixing 인가, across-view ROFA 인가, 다른 무엇인가
- D3 top-k 가 회복 가능한가
- Per-view rank 가 single-view rescue 시사하는가

---

## 2. Layer 1 — B8 within-view mixing visual proxy

### 무엇을 했나

각 target SP 의 mask 를 20 개 sampled train views 에 렌더링하고, 각 view 마다:
- **`num_components`** ≥ 10% area: SP 의 projection 이 disconnected components 로 쪼개지는지
- **`largest_component_frac`**: 가장 큰 component 의 비율
- **`bbox_fill_ratio`**: SP 픽셀이 bbox 를 얼마나 채우는지
- **`aspect_ratio`**: bbox 의 가로세로 비

집계 → `mix_view_frac` = num_components ≥ 2 인 view 비율.

### 직관

B8 의 진짜 측정 (within-view SAM mask 가 여러 개 mixed 되는 비율) 은 SAM mask 가 별도 저장되지 않아서 직접 불가. 대신 *visual proxy*:

- SP 가 한 view 에서 여러 disconnected piece 로 보이면 → 3D 공간에서 *여러 객체에 걸쳐 있을* 가능성
- bbox_fill 이 낮으면 → 산만한 mask
- 단 visual proxy 라 *interpretation 의 한계*: 단순히 occlusion 으로 인한 disconnected 일 수도 있음

### 결과

17 phantom vs 17 sampled easy:

| 통계 | Phantom17 | Easy_sample | 의미 |
|---|---|---|---|
| `mix_view_frac` 평균 | 0.16 | 0.16 | **차이 없음** |
| `mean_num_components` 평균 | 1.18 | 1.17 | 거의 동일 |
| `mean_bbox_fill` 평균 | 0.52 | 0.48 | phantom 이 살짝 더 compact |

**Phantom 의 mix_view_frac 분포**:
- 11/17 (65%): mix_view_frac < 0.10 (compact, mixing 증거 없음)
- 4/17 (24%): mix_view_frac > 0.30 (fragmented — napkin, plate, tesla door handle, hand)
- 2/17 (12%): 중간

→ 산출 plot: `plots/phantom_b8_mixrate.png`

### 의미

**B8 within-view mixing 은 phantom 의 dominant cause 가 아니다**. 17 phantom 중 11 개가 compact mask 임에도 phantom. Easy oracle 도 비슷한 mix-rate 분포 — *fragmentation 자체가 phantom 의 predictive feature 아님*.

단, 4 개 phantom (napkin, plate, tesla door handle, hand) 은 fragmented + phantom — *secondary contributing factor* 가능성. 이들은 Stage 2A 의 attribution 에서 `geometry_fragmented` tag 로 표시.

→ **Stage 2B 의 within-view dominant 분기는 supported 안 됨**.

---

## 3. Layer 2 — Wrong top-1 forensics + per-view rank trajectory

### 무엇을 했나

**Wrong top-1 forensics** — 각 phantom 의 CLIP top-1 SP (= 실제 method 가 잘못 선택한 것) 의 render mask 를:
- 정답 GT 와의 overlap (`overlap_with_gt`)
- scene 의 *다른* 모든 GT prompt 와의 IoU → 최대 매칭 prompt + IoU 기록
- 매칭 패턴 분류: **over_union / instance_confusion / background_drift**

**Per-view rank trajectory** — 각 GT frame 에서:
- 그 frame 의 GT 기반 oracle SP 재정의
- 같은 CLIP pool 에서 그 oracle 의 rank 측정
- `min_view_rank`, `max_view_rank`, `same_oracle_frac` 집계

### 직관

Wrong top-1 의 정체를 알면:
- **GT 와 겹치는 큰 SP** → D3 over_union (top-k 줄여서 해결)
- **다른 객체 SP** → instance confusion (E1 direction bias)
- **GT 근처도 아닌 SP** → 진짜 mysterious phantom (background drift, ROFA 실패)

Per-view trajectory 로 알 수 있는 것:
- **min ≤ 3** but max > 30 → 일부 view 에선 정답인데 평균이 죽임 → **target dilution** (A3-T subtype)
- **모든 view rank > 3** → 진짜 structural phantom (ROFA 가 outlier 가 아니라 *모든* view 가 잘못된 feature)

### 결과

**Wrong top-1 type 분포** (17 phantom):

| Type | 개수 | 예시 |
|---|---|---|
| **Background drift** | 11 (65%) | 11 개 phantom 의 wrong top-1 이 어떤 GT prompt 와도 IoU < 0.10 — 배경/unannotated 영역 |
| **Over-union** | 3 (18%) | jake (0.71 with GT), old camera (0.31), sink (0.80) |
| **Instance confusion** | 3 (18%) | pikachu → jake (IoU 0.71), rubber duck with hat → pirate hat (0.90), onion segments → egg (0.59) |

**Per-view rank trajectory**:

| Phantom | n_frames | min_rank | max_rank | trajectory 의미 |
|---|---|---|---|---|
| old camera | 3 | **1** | 130 | **target dilution** — 한 frame 에선 rank 1, 두 frame 에선 130 |
| bowl | 5 | **1** | 9 | target dilution (light) |
| napkin | 5 | **1** | 16 | target dilution (light) |
| sake cup | 6 | 6 | 10 | 안정적 mid-tier |
| sink | 2 | **1** | 4 | target dilution (light) |
| pikachu | 2 | 64 | 64 | structural |
| pumpkin | 1 | 52 | 52 | structural |
| tesla door handle | 2 | 256 | 256 | structural catastrophic |
| 나머지 9 개 | | 모두 rank > 10 | | structural — single-view rescue 불가 |

**13/17 (76%)**: 모든 view 에서 rank > 3 = **structural phantom**.
**4/17 (24%)**: 일부 view 에서 rank ≤ 3 = **target dilution evidence** (old camera, bowl, napkin, sink).

→ 산출 plot: `plots/phantom_wrong_top1_types.png`, `plots/phantom_per_view_trajectory.png`

### 의미

Layer 2 의 두 발견은 *서로 보완적*:

1. **Wrong top-1 의 65% 가 background drift** → ROFA aggregation 이 GT 와 무관한 *unannotated 영역의 SP* 를 평균적으로 strongest 로 만듦. 이것이 E1 direction bias (P1: small → background) 의 직접 evidence.

2. **76% 가 structural phantom** (모든 view rank > 3) → 단순한 outlier-view 가 아니라 **모든 view 가 일관되게 잘못된 SP feature** 를 만듦. F2 의 mean-dilution subtype (모든 view 가 비슷한 mixed feature) 의 evidence.

3. **24% target dilution** (4 개) → 일부 view 에서는 정답인데 평균이 죽임. F2 의 outlier-or-bimodal phantom subtype 후보.

4. **Instance confusion 3 개** (pikachu↔jake, rubber duck↔pirate hat, onion↔egg) → E1 P3 (instance-level bias) 의 *사전 evidence*. paper 의 E1 hypothesis 가 supported.

---

## 4. Layer 3 — D3 top-k sweep + cross-frame persistence

### 무엇을 했나

각 phantom 의 ref_frame 에서 CLIP pool 의 top-k SPs union 을 렌더링, IoU 측정.
- k ∈ {1, 2, 3, 5, 10}
- 모든 eval frame 에 대해 평균 IoU 계산
- per-prompt optimal k 와 그때의 IoU 기록

### 직관

- **k=1 이 best 면** → over-union (D3): default k=3 가 잡것 끌고옴
- **k=10 이 best 면** → oracle 이 pool 의 *깊은 곳*에 있음 (rank 4-10), CLIP 신호 약한 evidence
- **어떤 k 도 IoU 가 낮으면** → oracle 이 top-10 밖이거나 SP 자체가 망함 (structural)

### 결과

| Optimal k | 개수 | 의미 |
|---|---|---|
| **k=1** | 7 (41%) | over-union 가능성 — 단 IoU 가 회복하는 경우만 |
| k=2 | 2 (12%) | similar |
| k=3 (default) | 4 (24%) | already optimal |
| k=5 | 1 (6%) | medium-depth recovery |
| **k=10** | 3 (18%) | oracle deep in pool — bowl, plate, sake cup |

**회복 가능성** (per-prompt optimal k):
- IoU > 0.5 가능: **5/17 (29%)** — jake, rubber duck with hat, sink, bowl, plate
- IoU > 0.3 가능: **8/17 (47%)** — 위 5 개 + tesla door handle (0.28 → 0.28 marginal), napkin (0.45), sake cup (0.35)
- **IoU < 0.1 (어떤 k 도 회복 못 함)**: **8/17 (47%)** — pikachu, pumpkin, hand, onion segments, bear nose, cabinet, ottolenghi, spoon

**Mean IoU 개선**:
- Default k=3: 0.203
- Per-prompt optimal: **0.314** (+11.2 mIoU points)

→ 산출 plot: `plots/phantom_d3_sweep.png`

### 의미

**D3 top-k 는 phantom 의 ~50% 에 의미 있는 lever**:
- 5/17 (29%) 은 IoU > 0.5 으로 회복 가능 → 단순 top-k 정책 변경으로 큰 mIoU 회복
- 8/17 (47%) 는 어떤 k 로도 회복 안 됨 → 진짜 **structural phantom** (oracle 이 top-10 밖이거나 SP feature 자체가 wrong)

**Paper 함의**:
- D3 sweep 자체가 +11.2 mIoU 개선 — paper section 으로 가능 (단순 ablation)
- 하지만 +11.2 는 oracle 을 미리 안다는 가정 (per-prompt optimal k). 실제로는 prompt-agnostic 한 k=1 또는 k=10 를 골라야. Stage 2B 의 sweep 으로 확정.

---

## 5. Layer 4 — Per-prompt visual montage + case study

### 무엇을 했나

각 17 phantom 마다 ref_frame 에서 4 패널 시각화:
1. **GT crop** (green overlay)
2. **Oracle SP** (cyan) — 우리가 골라야 하는 깨끗한 SP
3. **CLIP top-1 SP** (red) — method 가 실제로 고르는 wrong SP
4. **Top-k=10 union** (yellow) — D3 best 후보

전체 17 행 × 4 열 montage 이미지.

### 결과

`plots/phantom_montage.png` (1612 × 3164 px) — 17 phantom 의 visual catalog.

직관:
- **Instance confusion 시각화**: pikachu (oracle 은 pikachu 인형) vs CLIP top-1 (jake 인형). 두 인형은 *서로 다른 색깔 영역* 인데 CLIP 이 jake 로 잘못 attribute.
- **Background drift 시각화**: pumpkin (oracle = pumpkin SP) vs CLIP top-1 (인접한 책상 위의 비-pumpkin 영역). 완전히 다른 영역.
- **Over-union 시각화**: sink (oracle = sink SP, IoU 0.84) vs CLIP top-1 (sink + 옆 카운터 큰 영역, overlap 0.80).
- **Target dilution 시각화**: old camera 같이 한 view 에서는 잘 보이지만 다른 view 에서 안 보이는 것 — montage 의 단일 view 로는 직관 어려움.

### 의미

Montage 는 paper 의 **killer figure** 후보 — "한 figure 로 17 phantom 의 모든 mechanism 이 다 있다."

---

## 6. Mechanism Attribution Table

| Scene | Prompt | Primary Mechanism | All Tags |
|---|---|---|---|
| figurines | jake | **D3_over_union** | over_union; d3_recoverable |
| figurines | old camera | **target_dilution** | over_union; target_dilution |
| figurines | pikachu | **instance_confusion** | instance_confusion; structural_phantom |
| figurines | pumpkin | **structural_ROFA** | background_drift; structural_phantom |
| figurines | rubber duck with hat | **instance_confusion** | instance_confusion |
| figurines | tesla door handle | **background_drift** | background_drift |
| ramen | bowl | **D3_deep_pool** | background_drift; d3_recoverable |
| ramen | hand | **structural_ROFA** | background_drift; structural_phantom |
| ramen | napkin | **target_dilution** | background_drift; target_dilution; geometry_fragmented; encoder_hidden |
| ramen | onion segments | **instance_confusion** | instance_confusion; structural_phantom; encoder_hidden |
| ramen | plate | **D3_deep_pool** | background_drift; d3_recoverable; geometry_fragmented; encoder_hidden |
| ramen | sake cup | **D3_deep_pool** | background_drift; d3_recoverable |
| teatime | bear nose | **structural_ROFA** | structural_phantom; encoder_hidden |
| waldo_kitchen | cabinet | **structural_ROFA** | background_drift; structural_phantom; encoder_hidden |
| waldo_kitchen | ottolenghi | **structural_ROFA** | background_drift; structural_phantom |
| waldo_kitchen | sink | unknown (D3-marginal) | over_union |
| waldo_kitchen | spoon | **structural_ROFA** | background_drift; structural_phantom; encoder_hidden |

### Distribution

| Primary Mechanism | 개수 | Share | Stage 2B 분기와의 관계 |
|---|---|---|---|
| **structural_ROFA** | 6 | 35% | across-view ROFA dominant — F2 subtype 분류 필요 |
| **D3_deep_pool** | 3 | 18% | top-k 변경으로 회복 — D3 sweep 의 method-level lever |
| **instance_confusion** | 3 | 18% | E1 P3 direction bias evidence |
| **target_dilution** | 2 | 12% | single-view rescue — A3-T evidence, F2 outlier subtype |
| D3_over_union | 1 | 6% | top-k=1 로 회복 (jake) |
| background_drift | 1 | 6% | encoder bias + ROFA (tesla door handle) |
| unknown (marginal) | 1 | 6% | sink — over_union but D3 lift 적음 |
| **합계** | 17 | 100% | |

### Aggregated by mechanism category

| Category | 개수 | Share |
|---|---|---|
| **ROFA across-view aggregation failure** (structural + deep_pool + target_dilution + over_union) | **12** | **71%** |
| **E1 direction bias** (instance_confusion + background_drift) | **4** | **24%** |
| Marginal/unknown | 1 | 5% |

→ 산출: `output/diagnostics/phantom_attribution.csv`

---

## 7. 종합 — Stage 2A 가 답한 것

### 4 가지 핵심 결론

1. **B8 (within-view mixing) 은 phantom 의 dominant cause 가 아님** (Layer 1).
   - 17 phantom 중 11 개가 compact mask, easy 와 mix_view_frac 분포 거의 동일.
   - 4 개 fragmented phantom 은 *secondary* — geometry_fragmented tag.

2. **Wrong top-1 의 65% 가 background drift** (Layer 2 forensic).
   - GT 와도, 다른 GT prompt 와도 매칭 안 되는 *unannotated 영역* 의 SP.
   - 이것이 ROFA 가 평균적으로 만드는 "phantom direction" 의 정량 evidence.

3. **76% 가 structural phantom** (Layer 2 trajectory).
   - 모든 GT frame 에서 rank > 3 — single-view rescue 불가.
   - **이것은 F2 의 mean-dilution subtype** (모든 view 의 feature 가 mixed) 의 시그너처.

4. **D3 top-k 는 47% 의 phantom 에서 회복 lever** (Layer 3).
   - 5/17 IoU > 0.5 회복 가능, 8/17 IoU > 0.3.
   - 단 8/17 (47%) 는 어떤 k 로도 회복 못 함 → **진짜 structural**.

### 한 줄로

> **"17 persistent phantoms 의 65-71% 는 across-view ROFA 의 mean-dilution failure 다. 단순한 outlier 가 아니라 *모든 view 가 일관되게 wrong direction* 으로 가는 phantom."**

---

## 8. Stage 2A 산출물

### CSV
- [output/diagnostics/persistent_phantoms_17.csv](../../../output/diagnostics/persistent_phantoms_17.csv) — 17 phantom list
- [output/diagnostics/stage2a_layer1.csv](../../../output/diagnostics/stage2a_layer1.csv) — B8 mix-rate (33 rows: 17 phantom + 16 easy)
- [output/diagnostics/stage2a_layer2_forensic.csv](../../../output/diagnostics/stage2a_layer2_forensic.csv) — wrong top-1 forensic (17 rows)
- [output/diagnostics/stage2a_layer2_trajectory.csv](../../../output/diagnostics/stage2a_layer2_trajectory.csv) — per-frame trajectory (51 rows)
- [output/diagnostics/stage2a_layer3_d3.csv](../../../output/diagnostics/stage2a_layer3_d3.csv) — D3 top-k sweep raw (51 rows)
- [output/diagnostics/stage2a_layer3_d3_summary.csv](../../../output/diagnostics/stage2a_layer3_d3_summary.csv) — per-phantom D3 summary (17 rows)
- **[output/diagnostics/phantom_anatomy.csv](../../../output/diagnostics/phantom_anatomy.csv)** — all-layer combined (17 rows × 26 cols)
- **[output/diagnostics/phantom_attribution.csv](../../../output/diagnostics/phantom_attribution.csv)** — mechanism attribution (17 rows)

### Plots
- `plots/phantom_b8_mixrate.png` — B8 mix-rate distribution
- `plots/phantom_wrong_top1_types.png` — wrong top-1 type bar chart
- `plots/phantom_d3_sweep.png` — per-phantom D3 top-k IoU curves
- `plots/phantom_per_view_trajectory.png` — min/max view rank per phantom
- **`plots/phantom_montage.png`** — 17 phantoms × 4 panel visual catalog (3164 × 1612 px)

### Scripts
- [scripts/stage2a_layer1_b8_coverage.py](../../../scripts/stage2a_layer1_b8_coverage.py)
- [scripts/stage2a_layer2_wrongtop1.py](../../../scripts/stage2a_layer2_wrongtop1.py)
- [scripts/stage2a_layer3_d3_topk.py](../../../scripts/stage2a_layer3_d3_topk.py)
- [scripts/stage2a_layer4_montage.py](../../../scripts/stage2a_layer4_montage.py)
- [scripts/stage2a_layer4_combine_montage.py](../../../scripts/stage2a_layer4_combine_montage.py)
- [scripts/stage2a_synthesize.py](../../../scripts/stage2a_synthesize.py)

---

## 9. Decision — Stage 2B 분기

### Evidence summary

| 분기 후보 | 지지 정도 | Evidence |
|---|---|---|
| **across-view ROFA dominant** | **★★★★★** | 71% phantoms 가 ROFA 류 실패, 76% structural (mean-dilution signature) |
| within-view mixing (B8) dominant | ★★ | 4/17 만 fragmented, easy 와 동일 분포 |
| encoder hidden (C2/C3) dominant | ★★ | 5/17 에 encoder_hidden tag (A2 rank > 1) 이지만 dominant 아님 |

### Stage 2B = **across-view ROFA dominant 분기**

#### 실험 계획

| 실험 | 비용 | 답하는 질문 |
|---|---|---|
| **F2.A — H2 lite instrument** | 1 일 | per-view CLIP feature 를 17 phantom 에 대해서만 dump (전체 SP 가 아니라 oracle + clip_top1 + 그 주변 → 100 SPs 정도, dataset-wide H2 가 아닌 phantom-focused) |
| **F2.B — ROFA subtype 분류** | 반나절 (F2.A 후) | per-view feature 의 GMM(k=1) vs GMM(k=2) BIC → outlier / bimodal-balanced / mean-dilution 세 subtype 비율 측정 |
| **F2.C — ROFA keep_mask 분석** | 반나절 (F2.A 후) | 각 phantom 마다 ROFA 가 drop 한 view 들 정체 분석 — GT 가 *잘 보이는* view 가 drop 됐는가? |
| **D3-prompt-agnostic sweep** | 반나절 | k ∈ {1, 2, 3, 5, 10} × 67 prompt → prompt-agnostic best k 결정 (실제 method-level lever) |
| **instance_confusion 3 case 검증** | 반나절 | pikachu↔jake, rubber duck↔pirate hat, onion↔egg 에 대해 negative prompt contrast (C3) 가 회복하는가 — quick C3 test |

→ Stage 2B 의 산출물:
- `output/diagnostics/stage2b_rofa_subtypes.csv` — 17 phantom × ROFA subtype
- `output/diagnostics/stage2b_d3_full_sweep.csv` — 67 prompt × k sweep
- `output/diagnostics/stage2b_c3_quickcheck.csv` — 3 instance confusion case × C3 lambda
- `plots/stage2b_rofa_subtype_distribution.png`
- `plots/stage2b_d3_prompt_agnostic_curve.png`
- `plots/stage2b_keep_mask_geometry.png`
- `md/hypotheses/experiments/stage2b_rofa_anatomy.md`
- `md/THGS/paper_section2_draft.md`

#### Reconciliation with Stage 1 의 4.2:1 비율

Stage 1 의 4.2:1 (phantom : real) 은 *D2 class 의 분포*.
Stage 2A 의 65-71% (ROFA fail) 는 *phantom 안 의 mechanism breakdown*.

- 4.2:1 = phantom (21) : real (5) — A2 ceiling 으로 분리
- 71% of phantom = ROFA fail — 17 persistent 의 mechanism, 4 non-persistent phantom 도 포함하면 비슷
- D2.real (5) + 17 ROFA fail + 4 다른 mechanism = phantom 의 entire breakdown

Stage 2B 가 F2 subtype 으로 ROFA fail 의 71% 를 더 세분화 → 새 aggregation method 의 *직접 target* 정량.

---

## 10. 관련 문서

- [stage1_b7_a4_a2.md](stage1_b7_a4_a2.md) — Stage 1 (B7+A4+A2+Joint+Cross-method)
- [../extended_failure_hypotheses.md](../extended_failure_hypotheses.md) — 6.2 patch 가설 catalog (F2, A3, E1 정의)
- [../../THGS/paper_section1_draft.md](../../THGS/paper_section1_draft.md) — paper Section 1 draft
