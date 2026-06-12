# Paper Section 2 (Draft) — Mechanism Analysis of Persistent Phantoms

> Stage 2A + Stage 2B 통합. Section 1 의 D2.phantom 21 prompts 중 17 persistent phantoms (THGS + ReLaGS 모두 fail) 에 대한 4-Layer forensic + ROFA subtype 분석 + D3 sweep + C3 quickcheck.
>
> Status: **Draft v1** — reproducible from `output/diagnostics/{phantom_anatomy.csv, phantom_attribution.csv, stage2b_*.csv}`.

---

## 1. Phantom Mechanism Question

Section 1 결과: D2.phantom 21 (31% of all prompts), 그 중 17 (81%) 가 THGS+ReLaGS 모두 fail = **persistent phantom**.

본 section 의 question:
> **이 17 persistent phantom 의 mechanism — within-view mixing? Across-view ROFA aggregation? Encoder limit? Top-k policy? Instance confusion?**

답을 위해 4-Layer forensic (Stage 2A) + ROFA anatomy (Stage 2B) 진행.

---

## 2. Methodology

### 2.1 Stage 2A — 4-Layer Anatomy

각 phantom 에 대해 4 차원으로 측정:

- **Layer 1 — Spatial origin**: SP 가 view 안에서 fragmented? `mix_view_frac` (≥ 2 connected components 의 view 비율)
- **Layer 2 — Forensic**: CLIP top-1 SP 가 무엇인가? GT 와 다른 prompt 와의 IoU. Per-frame rank trajectory
- **Layer 3 — D3 cardinality**: top-k ∈ {1, 2, 3, 5, 10} 의 IoU 변화
- **Layer 4 — Visual case study**: 17 phantoms × 4 panel montage

### 2.2 Stage 2B — ROFA Anatomy

- **F2.A H2 lite instrument**: 17 phantom × 30 sampled train views 의 *clean SP-mask CLIP image encoding* (배경 검정 후 CLIP image encoder, no SAM mixing)
- **F2.B subtype classification**: cos distribution 으로 outlier / bimodal / mean-dilution / strong_signal 분류
- **F2.C keep-mask analysis**: ROFA simulate, kept vs dropped views 의 cos 비교
- **D3 prompt-agnostic sweep**: 67 prompts × 6 k values
- **C3 quick check**: 3 instance confusion case × 6 lambda values

---

## 3. Findings

### 3.1 Spatial origin is not dominant (Layer 1)

| Stat | Phantom17 | Easy_sample |
|---|---|---|
| `mix_view_frac` 평균 | 0.16 | 0.16 |
| `mean_bbox_fill` 평균 | 0.52 | 0.48 |

17 phantom 중 **11 (65%) 가 compact SP mask** (`mix_view_frac` < 0.10). Easy 와 분포 거의 동일.

→ **Spatial fragmentation (visual proxy) 은 phantom 의 dominant cause 아님**.

### 3.2 Wrong top-1 은 65% 가 background drift (Layer 2 forensic)

| Wrong top-1 type | 개수 | Share |
|---|---|---|
| **Background drift** (어떤 GT prompt 와도 IoU < 0.10) | 11 | 65% |
| **Over-union** (GT 와 IoU ≥ 0.30) | 3 | 18% — `jake`, `old camera`, `sink` |
| **Instance confusion** (다른 GT object 와 IoU ≥ 0.30) | 3 | 18% — `pikachu→jake`, `rubber duck with hat→pirate hat`, `onion segments→egg` |

→ 65% phantom 에서 method 가 *GT 와 무관한 unannotated 영역의 SP* 를 1 등으로 뽑음. **Phantom direction bias 의 evidence** — paper 의 E1 hypothesis P1/P4 (background drift) 의 정량.

### 3.3 76% 가 structural phantom — single-view rescue 불가 (Layer 2 trajectory)

| Trajectory type | 개수 | Share |
|---|---|---|
| **Structural** (모든 view 에서 rank > 3) | 13 | 76% |
| **Target dilution** (일부 view 에서 rank ≤ 3) | 4 | 24% |

Target dilution 4 개: `old camera` (rank 1 in one frame, 130 in others), `bowl` (1-9), `napkin` (1-16), `sink` (1, 4).

→ **76% 는 단순 outlier-view 가 아니라 *모든 view 가 일관되게 잘못된 SP feature*** — F2 의 mean-dilution subtype 의 시그너처.

### 3.4 D3 top-k 는 47% 의 phantom 에서 recovery lever (Layer 3 + D3 full sweep)

**Per-phantom optimal k**:

| Optimal k | 개수 | 의미 |
|---|---|---|
| k=1 | 7 | over-union 회복 후보 |
| k=2 | 2 | 마찬가지 |
| k=3 (default) | 4 | already optimal |
| k=5 | 1 | medium recovery |
| k=10 | 3 | deep-pool oracle (bowl, plate, sake cup) |

**recovery 가능성** (per-prompt optimal k 가정 시):
- IoU > 0.5 회복: 5/17 (29%)
- IoU > 0.3 회복: 8/17 (47%)
- 어떤 k 도 안 됨 (`IoU < 0.10`): 8/17 (47%)

**Prompt-agnostic fixed k** (실제 method 의 lever):

| k | All 67 prompts mIoU | 17 phantoms mIoU |
|---|---|---|
| 1 | 0.464 | 0.155 |
| 2 | 0.526 | 0.166 |
| **3 (default)** | **0.542** | 0.203 |
| 5 | 0.469 | 0.185 |
| **10** | 0.483 | **0.290** |
| 20 | 0.448 | 0.281 |

→ **Trade-off**: k=10 이 phantom 만 보면 +8.7 mIoU 회복, 하지만 전체 mIoU 는 −5.9 점 감소. **Prompt-conditional adaptive k 가 필수**. 단순 fixed-k 변경은 free lunch 없음.

### 3.5 **Critical finding** — Within-view stage 가 신호 죽임 (Stage 2B F2.A)

17 phantom 의 oracle SP 를 *clean mask-blackout CLIP image encoding* 으로 인코딩 (no SAM mixing) 후 prompt cos:

| Subtype | 개수 | Share | 의미 |
|---|---|---|---|
| **strong_signal_phantom** (cos_mean ≥ 0.20) | **11** | **65%** | clean CLIP 이 prompt 인식 가능 ★ |
| mean_dilution (cos_mean < 0.20) | 4 | 24% | clean CLIP 도 약함 — `old camera`, `pumpkin`, `napkin`, `ottolenghi` |
| bimodal_balanced (cos IQR > 0.06) | 2 | 12% | view-distribution 분기 — `pikachu`, `sake cup` |

**파악**:
- 11 phantoms 의 *clean SP-mask CLIP encoding* 으로는 평균 cos 0.20–0.30 (강한 신호)
- 그런데 pipeline 의 final SP-CLIP feature 위에서는 rank > 3 (phantom)
- → **pipeline 의 within-view 단계 (SAM-based ratio mixing) 가 신호를 죽이고 있다**

```python
# pipeline 의 within-view feature 생성 (merge_proj.py:167-168):
sp_mask_mat[sp_mask_mat < RATIO_THRESHOLD] = 0      # RATIO_THRESHOLD = 0.3
sp_feat = sp_mask_mat @ view_level_feature           # weighted sum of multiple SAM mask features
```

이 mixing 이 **65% 의 phantom 의 dominant cause**. Clean CLIP 으로는 인식 가능한 prompt 가, multi-SAM-mask 의 ratio-weighted 합산을 거치면서 wrong direction 으로 끌려감.

### 3.6 ROFA 는 정상 작동 (Stage 2B F2.C)

| ROFA pathology | 개수 | Share |
|---|---|---|
| kept_cos > dropped_cos (정상) | 14 | 82% |
| dropped_cos > kept_cos (mechanism 실패) | 3 | 18% |

**14/17 phantom 에서 ROFA 는 의도대로 동작** — kept views 가 평균적으로 더 강한 cos. 그럼에도 phantom 결과. → **ROFA 가 잘못된 게 아니라 *이미 죽은 신호 위에서 동작***.

**3/17 (18%) 가 ROFA mechanism 실패**: `old camera`, `ottolenghi`, `sink` — 강한 cos view 들을 outlier 로 판정해 drop. ROFA 의 정량적 한계 evidence.

### 3.7 C3 negative contrast 는 instance confusion 회복 못 함 (Stage 2B)

3 instance confusion phantoms 의 λ sweep:

| Phantom | λ=0 rank | λ=1.0 rank | Δ |
|---|---|---|---|
| pikachu | 64 | 50 | −14 (22% marginal 개선) |
| rubber duck with hat | 5 | 6 | +1 |
| onion segments | 14 | 13 | −1 |

→ **Text-side contrast 로는 instance confusion 못 풀음**. Phantom 의 source 는 SP-feature aggregation 단계에 있음. 새 aggregation method 가 유일한 path.

---

## 4. Mechanism Attribution Table — 17 Persistent Phantoms

| Scene | Prompt | Primary Mechanism | F2 Subtype | D3 Optimal k |
|---|---|---|---|---|
| figurines | jake | D3 over_union | strong_signal | 2 |
| figurines | old camera | target_dilution + ROFA fail | mean_dilution | 1 |
| figurines | pikachu | instance_confusion | bimodal_balanced | 1 |
| figurines | pumpkin | structural ROFA | mean_dilution | 1 |
| figurines | rubber duck with hat | instance_confusion | strong_signal | 3 |
| figurines | tesla door handle | background_drift | strong_signal | 2 |
| ramen | bowl | D3 deep_pool | strong_signal | 10 |
| ramen | hand | structural ROFA | strong_signal | 1 |
| ramen | napkin | target_dilution + fragmented | mean_dilution | 3 |
| ramen | onion segments | instance_confusion | strong_signal | 5 |
| ramen | plate | D3 deep_pool | strong_signal | 10 |
| ramen | sake cup | D3 deep_pool | bimodal_balanced | 10 |
| teatime | bear nose | structural ROFA | strong_signal | 3 |
| waldo_kitchen | cabinet | structural ROFA | strong_signal | 1 |
| waldo_kitchen | ottolenghi | structural ROFA + ROFA fail | mean_dilution | 1 |
| waldo_kitchen | sink | ROFA fail + over_union | strong_signal | 3 |
| waldo_kitchen | spoon | structural ROFA | strong_signal | 1 |

### Aggregated by *root cause* mechanism

| Root Cause | 개수 | Share | Fix path |
|---|---|---|---|
| **Within-view SAM mixing killed signal** | 11 | 65% | New within-view aggregation (hard assignment, query-conditioned) |
| **Encoder-side mean dilution** | 4 | 24% | Encoder upgrade, prompt expansion (C2) |
| **ROFA mechanism fail** | 3 | 18% (overlap) | Better outlier detection (mode-cluster center) |
| **D3 over-union or deep-pool** | 5 | 29% (overlap) | Adaptive top-k policy |

(tags overlap — 한 phantom 이 여러 cause 가질 수 있음)

---

## 5. Reconciliation with Section 1

Section 1 의 발견:
- D2.phantom : D2.real = 21 : 5 = **4.2 : 1** (method-agnostic)

Section 2 가 그 21 phantom 의 *내부 구조* 분해:

```
67 prompts total
├── 36 (54%) easy            — both A2 and SP rank ≤ 3
├── 21 (31%) D2.phantom
│   ├── 17 persistent       
│   │   ├── 11 strong_signal — pipeline within-view stage killed
│   │   │   ├── 5 D3-recoverable (jake, bowl, plate, sake cup, tesla)
│   │   │   ├── 2 instance_confusion (rubber duck, onion segments)
│   │   │   └── 4 catastrophic (pikachu, hand, cabinet, spoon, sink, ...)
│   │   ├── 4 mean_dilution  — encoder-side limit
│   │   └── 2 bimodal_balanced — ROFA blind (pikachu, sake cup overlap)
│   └── 4 method-specific phantom
├── 5 (7.5%) D2.real         — pure encoder limit
└── 5 (7.5%) rare            — positive aggregation
```

**Section 1 narrative 의 보강**: 31% 의 phantom 중 **65% 는 within-view stage 가 dominant cause** — paper 의 새 method 의 정확한 target.

---

## 6. Implications for Paper Method (Section 3)

Stage 2A + 2B 의 mechanism evidence 가 새 method 의 design 을 *직접* motivate:

### 6.1 Within-view stage fix 가 first-order lever (65% phantoms 대상)

현재 pipeline:
```python
sp_feat = sp_mask_mat @ view_level_feature  # ratio-weighted sum of multi-SAM-mask features
```

문제: RATIO_THRESHOLD=0.3 이라 비등한 mask (0.45 + 0.40) 둘 다 통과 → mixed feature.

대안:
- **(A) Hard within-view assignment**: `sp_feat = view_level_feature[argmax(sp_mask_mat)]` (single best mask)
- **(B) Query-conditioned within-view selection**: query embedding 가까운 mask 만 select
- **(C) RATIO_THRESHOLD 상향**: 0.5 또는 0.7

→ Stage 3 에서 ablation.

### 6.2 ROFA 는 keep — 그 위에 새 within-view stage 만 추가

ROFA 가 82% 에서 정상 작동하므로 *기존 across-view stage 는 유지*. 새 method 는 **within-view 단계만 fix**.

### 6.3 Mean dilution 4 개는 encoder 측 fix 필요

`old camera`, `pumpkin`, `napkin`, `ottolenghi` 는 within-view fix 로 안 됨. C2 (prompt expansion) 또는 C3 (좀 더 큰 λ + 신중한 negative set) 의 후속 실험 필요.

### 6.4 Adaptive top-k 가 secondary lever

D3 default k=3 가 method-level optimal, 하지만 prompt-agnostic 회복 위해서는 *gap-based* adaptive (ReLaGS-style) 의 *세련화* 필요. Paper 의 minor section.

---

## 7. Key Takeaways

1. **Phantom 의 dominant root cause 는 within-view SAM mixing 단계의 신호 죽음** (65%).
2. **ROFA across-view aggregation 은 *정상 작동* (82%)** — 죽은 신호 위에서 동작하기에 회복 불가.
3. **17 persistent phantoms 의 24% 는 encoder-side mean dilution** — within-view fix 로 안 됨.
4. **Instance confusion 3 case 는 text-side contrast (C3) 로 회복 불가** — SP-feature aggregation 단계 fix 필요.
5. **D3 cardinality 는 phantom-conditional adaptive 정책 필요** — fixed-k 로는 trade-off.

→ Paper 의 **proposed method = new within-view aggregation policy** (Stage 3 에서 정량).

---

## 8. Artifacts

| File | 내용 |
|---|---|
| [output/diagnostics/phantom_anatomy.csv](../../output/diagnostics/phantom_anatomy.csv) | 17 phantoms × 26 columns (Stage 2A all-layer merged) |
| [output/diagnostics/phantom_attribution.csv](../../output/diagnostics/phantom_attribution.csv) | mechanism attribution 17 rows |
| [output/diagnostics/stage2b_rofa_subtypes.csv](../../output/diagnostics/stage2b_rofa_subtypes.csv) | F2 subtype (17) |
| [output/diagnostics/stage2b_rofa_keep_mask.csv](../../output/diagnostics/stage2b_rofa_keep_mask.csv) | ROFA keep-mask analysis |
| [output/diagnostics/stage2b_d3_full_sweep.csv](../../output/diagnostics/stage2b_d3_full_sweep.csv) | D3 sweep 67 prompts × 6 k |
| [output/diagnostics/stage2b_d3_full_sweep_agg.csv](../../output/diagnostics/stage2b_d3_full_sweep_agg.csv) | per-prompt aggregate |
| [output/diagnostics/stage2b_c3_quickcheck.csv](../../output/diagnostics/stage2b_c3_quickcheck.csv) | C3 lambda sweep on 3 instance confusion |
| [output/diagnostics/plots/phantom_montage.png](../../output/diagnostics/plots/phantom_montage.png) | 17-phantom × 4 panel visual catalog |
| [output/diagnostics/plots/stage2b_rofa_subtype_distribution.png](../../output/diagnostics/plots/stage2b_rofa_subtype_distribution.png) | F2 subtype bar |
| [output/diagnostics/plots/stage2b_keep_mask_geometry.png](../../output/diagnostics/plots/stage2b_keep_mask_geometry.png) | kept vs dropped cos |
| [output/diagnostics/plots/stage2b_per_view_cos_dists.png](../../output/diagnostics/plots/stage2b_per_view_cos_dists.png) | per-view cos distributions |
| [scripts/stage2a_*.py](../../scripts/) | Stage 2A 4 layers + synthesis (6 scripts) |
| [scripts/stage2b_*.py](../../scripts/) | Stage 2B F2 + D3 + C3 (4 scripts) |
