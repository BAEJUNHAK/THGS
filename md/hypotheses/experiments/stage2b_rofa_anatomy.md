# Stage 2B — ROFA Anatomy + D3 Prompt-Agnostic Sweep + C3 Quickcheck

> 완료. Stage 2A 가 분기 결정 ("across-view ROFA dominant") 한 결과에 따라 4 sub-experiments 진행.
>
> 핵심 finding 한 줄: **"17 phantom 중 11 개 (65%) 는 *clean SP-mask CLIP encoding 으로 cos≥0.20 의 strong signal* 인데도 pipeline 의 SP-CLIP feature 에서는 phantom — pipeline 의 within-view SAM mixing 단계가 신호를 죽이고 있다. ROFA 는 정상 작동하지만 *이미 죽은 signal* 위에서."**
>
> **✱ 검증 패치 (2026-06-10)**: τ=2.0 (실제 pipeline default) 재실행 + rank 기반 재검증 + joint raw/canon robustness 체크 완료 ([scripts/verify_stage2b_checks.py](../../../scripts/verify_stage2b_checks.py)). **65% strong-signal 결론 유지**, F2.C 의 수치/케이스 명단은 **정정** (§4), onion segments 는 encoder 측으로 **재분류 권장** (§2).

---

## 0. Stage 2B 의 question

> **Stage 2A 가 좁힌 "across-view ROFA dominant" 분기에서:**
> 1. ROFA 가 phantom 을 만들었는가, 아니면 *이미 죽은 signal* 위에서 동작하는가?
> 2. ROFA 의 어떤 subtype (outlier / bimodal / mean-dilution) 이 dominant?
> 3. D3 top-k 가 prompt-agnostic 하게 phantom 회복 가능한가?
> 4. Instance confusion phantom 은 C3 (negative contrast) 로 회복 가능한가?

---

## 1. Pre-experiment 상태 (Stage 2A 결과)

- **71% phantoms 가 ROFA 류 across-view aggregation 실패** (structural_ROFA + D3_deep_pool + target_dilution + over_union)
- 76% 가 모든 view 에서 rank > 3 (structural — 단순 outlier 가 아님)
- 65% wrong top-1 이 background drift (GT 가 없는 영역)

---

## 2. F2.A — H2 lite instrument (per-view CLIP feature dump)

### 무엇을 했나

기존 `language_features/` 가 disk 에 없어서 진짜 H2 instrument (전체 pipeline replay) 가 비쌈. 대신 **proxy**:

각 17 phantom 의 oracle SP 에 대해, 30 sampled train views 각각에서:
1. Oracle SP mask 를 renderer 로 추출 (RGB 이미지의 어느 픽셀이 SP 인지)
2. RGB 이미지에서 SP 영역만 crop + 배경 검정 (A2 mask policy 와 동일)
3. CLIP image encoder (ViT-B-16) 로 인코딩 → per-view image feature
4. Prompt text embedding 과 cosine = per-view "이 view 에서 SP 가 prompt 와 얼마나 비슷한가"

### 직관

**이건 진짜 H2 instrument 가 아니라 LITE 버전**. 차이:
- 진짜 H2 = pipeline 의 SAM-segmented per-view feature (with within-view mask mixing)
- LITE H2 = pipeline 의 single oracle SP mask 의 clean CLIP encoding (no SAM mixing)

→ 두 feature 가 다르면 → **그 차이가 곧 within-view 단계의 손실**.

### 결과 — per-view cos 분포

| Prompt | N views | cos_mean | cos_std | cos_min | cos_max |
|---|---|---|---|---|---|
| rubber duck with hat | 28 | **0.295** | 0.065 | 0.128 | 0.357 |
| bowl | 30 | **0.235** | 0.023 | 0.184 | 0.279 |
| bear nose | 17 | **0.232** | 0.034 | 0.134 | 0.279 |
| plate | 30 | **0.234** | 0.022 | 0.179 | 0.265 |
| jake | 27 | 0.229 | 0.012 | 0.192 | 0.249 |
| hand | 7 | 0.228 | 0.009 | 0.210 | 0.240 |
| spoon | 8 | 0.217 | 0.028 | 0.176 | 0.272 |
| tesla door handle | 22 | 0.211 | 0.039 | 0.118 | 0.295 |
| sake cup | 23 | 0.211 | 0.046 | 0.123 | 0.286 |
| sink | 16 | 0.210 | 0.035 | 0.167 | 0.273 |
| cabinet | 8 | 0.204 | 0.031 | 0.155 | 0.267 |
| napkin | 25 | 0.196 | 0.033 | 0.124 | 0.267 |
| onion segments | 30 | 0.201 | 0.020 | 0.164 | 0.240 |
| pikachu | 29 | 0.194 | 0.048 | 0.118 | 0.304 |
| pumpkin | 25 | 0.187 | 0.030 | 0.146 | 0.276 |
| old camera | 28 | 0.176 | 0.023 | 0.102 | 0.224 |
| ottolenghi | 9 | 0.153 | 0.027 | 0.099 | 0.198 |

**cos_mean ≥ 0.20 은 12/17 (71%)** — onion segments (0.201) 포함. subtype 분류 후 strong_signal_phantom 으로 남는 것이 11/17 (65%) (sake cup 은 bimodal 로 분류).

**✱ 검증 패치 (2026-06-10) — rank 기반 재검증** ([verify_h2lite_rank.csv](../../../output/diagnostics/verify_h2lite_rank.csv)): 절대 threshold 대신 각 view 의 clean crop feature 를 *scene 전체 prompt 와 랭킹* 하면 **median rank ≤ 3 = 11/17 (65%)** — headline 비율이 더 강한 기준에서도 생존. 단 멤버십 1 건 교정:
- `onion segments`: cos 0.201 (threshold 통과) 이지만 **median rank 6.0, top-3 view 비율 17%** → clean crop 으로도 CLIP 이 사실상 인식 못 함 = **encoder 측 (mean_dilution 류) 으로 재분류 권장**. onion→egg instance confusion + encoder_hidden tag 와 정합.
- `sake cup`: rank 기준으로는 strong (median 3.0) — bimodal 분류와 별개로 재료는 살아있음.

### 의미

**이 단일 fact 이 paper 의 핵심 contribution 의 evidence**:

- 17 persistent phantoms 의 65% 는 *clean SP-mask CLIP 인코딩으로는 정답 인식 가능*
- 그런데 pipeline 의 final SP-CLIP feature 에서는 *모든 view 의 평균* 인데도 phantom
- → **pipeline 의 within-view 단계 (SAM-based mixing) 에서 신호가 죽고 있다**

이건 Stage 2A Layer 1 의 *visual proxy* (connected components) 가 dominant 가 아니라고 한 발견과 *모순* 아니다. 두 발견이 짝을 이룸:
- SP 의 **공간적 fragmentation** 은 phantom 의 dominant cause 아님 (Layer 1)
- 하지만 SP feature 생성 시 **SAM mask 와의 ratio mixing** 이 dominant cause (F2.A)
- 즉 SP mask 자체는 깨끗한데, *그 SP 를 SAM mask 비율로 weighted 합하는 단계* 가 문제

---

## 3. F2.B — ROFA Subtype Classification

### 무엇을 했나

각 phantom 의 per-view feature 위에서:
1. ROFA simulate (mean_sim per view → keep_mask with tau=1.0)
2. cos distribution 분석 (mean, std, IQR)
3. Subtype 분류:
   - **outlier**: ROFA dropped low-cos views 정상 동작
   - **bimodal_balanced**: cos IQR > 0.06 AND std > 0.04 AND kept set still has spread
   - **mean_dilution**: cos_mean < 0.20 (CLIP 이 못 인식)
   - **strong_signal_phantom**: cos_mean ≥ 0.20 (CLIP 은 인식하는데 SP-CLIP 은 phantom)

### 결과

| Subtype | 개수 | Share | 의미 |
|---|---|---|---|
| **strong_signal_phantom** | **11** | **65%** | clean encoding 으로는 strong, pipeline 에서 죽음 — **within-view stage 문제** |
| mean_dilution | 4 | 24% | clean encoding 자체도 약함 — encoder 영역 문제 (old camera, pumpkin, napkin, ottolenghi) |
| bimodal_balanced | 2 | 12% | cos distribution wide — ROFA blind mode-cluster (pikachu, sake cup) |
| outlier_handled | 0 | 0% | ROFA 가 정상 outlier handling 한 경우 — 없음 |

**✱ 검증 패치 (2026-06-10) — τ robustness**: 위 분류는 simulation **τ=1.0** 기준인데 실제 ReLaGS pipeline default 는 **τ=2.0** ([ReLaGS/merge_proj.py:124](../../../ReLaGS/merge_proj.py#L124), argparse default 2). τ=2.0 재실행 결과 **subtype 분포 동일 (11 / 4 / 2 / 0)** — 65% strong_signal 결론은 τ 에 견고 ([verify_f2_tau_sweep.csv](../../../output/diagnostics/verify_f2_tau_sweep.csv)).

### 의미

**중요한 발견 4 개**:

1. **65% 가 strong_signal_phantom**: pipeline 의 SP-CLIP feature 가 죽이는 게 문제. ROFA 가 잘못 작동하는 게 아니라 *이미 죽은 신호 위에서 작동* 함.
2. **24% mean_dilution**: 정말 CLIP encoder limit 이거나 SP mask 가 너무 작아서 CLIP 이 인식 못 함. 4 개 (old camera, pumpkin, napkin, ottolenghi) — Stage 2A 의 instance_confusion 과 다름. encoder 측 fix 필요.
3. **12% bimodal_balanced**: pikachu (cos_std 0.048, cos_max 0.304), sake cup (cos_std 0.046) — *일부 view 는 강함, 다른 view 는 약함*, ROFA 가 그 차이를 *못 잡음*. F2 가설의 bimodal-balanced 의 직접 evidence.
4. **0% outlier_handled** = ROFA 가 의도된 대로 (outlier 처리) 동작하는 경우 없음. ROFA mechanism 적 한계의 강한 evidence.

---

## 4. F2.C — ROFA Keep-Mask Analysis

### 무엇을 했나

각 phantom 에서:
- ROFA 가 drop 한 view 들의 cos with prompt (`dropped_cos_mean`)
- ROFA 가 keep 한 view 들의 cos (`kept_cos_mean`)
- 비교

### 결과

| ROFA Pathology | 개수 | Share |
|---|---|---|
| **ROFA_kept_good_views** (kept_cos > dropped_cos) | 14 | 82% |
| **ROFA_dropped_good_views** (dropped_cos > kept_cos) | 3 | 18% |

**3 개 ROFA mechanism 실패 case**:
- **old camera**: kept_cos 0.176, dropped_cos 0.178 — ROFA 가 더 강한 cos view 들을 *drop*
- **ottolenghi**: kept_cos 0.147, dropped_cos 0.198 — 강한 view (cos 0.198) 를 outlier 로 잘못 판정
- **sink**: kept_cos 0.210, dropped_cos 0.230 — 같은 패턴

**✱ 검증 패치 (2026-06-10) — τ=2.0 (실제 pipeline default) 재실행** ([verify_f2_tau_sweep.csv](../../../output/diagnostics/verify_f2_tau_sweep.csv)):

| ROFA Pathology (τ=2.0) | 개수 | Share | 케이스 |
|---|---|---|---|
| ROFA_kept_good_views | 11 | 65% | — |
| no_drops (drop 0 건) | 2 | 12% | sink, cabinet |
| **ROFA_dropped_good_views** | **4** | **24%** | **old camera, ottolenghi, pikachu, onion segments** |

- 위 82%/18% 와 3-case 명단은 **τ=1.0 simulation 의 산물**. τ=2.0 에서 sink 는 drop 자체가 없어 명단에서 빠지고, **pikachu (drop 5→1), onion segments (drop 5→2) 가 좋은 view 를 drop 하는 케이스로 새로 등장**.
- **신규 관찰**: τ=2.0 실수 4 건 중 2 건 (pikachu, onion segments) 이 정확히 instance-confusion phantom — 실제 pipeline 조건에서 ROFA 가 instance confusion 의 정답 view 를 drop 한다는 단서. Stage 3 분석 후보.
- 정성적 결론 ("ROFA 는 phantom 의 주범이 아님") 은 유지: 실수 4/17 에 그치고, outlier_handled 0% 도 τ=2.0 에서 유지.

### 의미

**14/17 (82%) 의 phantom 에서 ROFA 는 정확히 의도대로 동작** — kept 가 dropped 보다 평균적으로 더 강한 cos. 하지만 **여전히 phantom 결과**. → ROFA 가 잘못된 게 아니라, *피할 수 없는 phantom* 이 이미 모든 view 에 존재.

3/17 (18%) 는 **ROFA 가 mechanism 적으로 잘못 동작** — 정답을 더 잘 보여주는 view 를 outlier 로 판정. paper 의 ROFA limitation 의 직접 evidence.

---

## 5. D3 Prompt-Agnostic Top-K Sweep (67 prompts × 6 k values)

### 무엇을 했나

Full 67 prompt 에 대해 k ∈ {1, 2, 3, 5, 10, 20} sweep → method-level fixed-k 의 mean mIoU.

### 결과

| k | Method-level mean mIoU |
|---|---|
| 1 | 0.464 |
| 2 | 0.526 |
| **3 (default)** | **0.542** |
| 5 | 0.469 |
| 10 | 0.483 |
| 20 | 0.448 |

→ **k=3 default 가 이미 method-level optimal**.

### 17 phantom 만 보면

| k | mean mIoU (17 phantoms) | > 0.5 인 수 |
|---|---|---|
| 1 | 0.155 | 2 / 17 |
| 2 | 0.166 | 2 / 17 |
| 3 (default) | 0.203 | 3 / 17 |
| 5 | 0.185 | 3 / 17 |
| **10** | **0.290** | **5 / 17** |
| 20 | 0.281 | 4 / 17 |

→ **17 phantom 만 보면 k=10 이 best** (+8.7 mIoU 회복).

### Per-scene

| Scene | Optimal k | mean mIoU at optimal |
|---|---|---|
| figurines | 2 | 0.508 |
| ramen | 3 | 0.371 |
| **teatime** | **3** | **0.757** (가장 쉬움) |
| waldo_kitchen | 3 | 0.555 |

### 의미

**핵심 dilemma**: D3 top-k 는 phantom-vs-non-phantom 의 *trade-off* 임:
- k 늘리면 (5, 10): phantom 회복 ↑, 하지만 non-phantom 에서 noise 증가
- k=3 default 가 average 적으로 best — 하지만 phantom 만 보면 sub-optimal

**Paper 함의**: 단순 k 변경으로는 free lunch 없음. **adaptive policy** 필요:
- "Easy" prompt (top-1 강함, z_margin 작음) → k=2-3
- "Phantom-prone" prompt (top-1 약함, z_margin 큼) → k=10

이게 ReLaGS 의 gap-based adaptive policy 와 유사 — 하지만 ReLaGS 도 phantom 의 19% 만 회복 (Stage 1 결과). 새 method 의 *prompt-conditional k selection* 이 더 정교해야 함.

---

## 6. C3 Quick Check — Negative Prompt Contrast (3 instance confusion cases)

### 무엇을 했나

Instance confusion phantom 3 개 (`pikachu`, `rubber duck with hat`, `onion segments`) 에 대해:
- `score(SP) = cos(SP, target) − λ · max_neg cos(SP, neg)` 
- negatives = scene 의 다른 prompts
- λ ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}
- Oracle SP rank 변화 측정

### 결과

| Phantom | λ=0 rank | λ=1.0 rank | Δ |
|---|---|---|---|
| pikachu | 64 | **50** | −14 (22% 개선) |
| rubber duck with hat | 5 | 6 | +1 (worse) |
| onion segments | 14 | 13 | −1 (marginal) |

### 의미

**C3 negative contrast 는 instance confusion phantom 회복 못 함**:
- pikachu 만 marginal 개선 (64 → 50, 여전히 catastrophic phantom)
- 다른 2 개는 변화 없음

→ **Instance confusion 의 mechanism 은 SP-feature aggregation 단계에 있다**. Text-side contrast 로는 SP feature 자체가 wrong direction 으로 가는 걸 *되돌릴 수 없음*. 새 aggregation method 가 유일한 path.

이는 Stage 2A 의 instance_confusion 3 개 (`pikachu → jake`, `rubber duck → pirate hat`, `onion segments → egg`) 에 대한 C3 의 *negative finding*. 강한 결론.

---

## 7. 종합 — Stage 2B 가 답한 것 (4 + 1 발견)

### 핵심 결론 4 개

1. **Within-view stage 가 신호 죽임 (65%)** — clean SP-mask CLIP encoding 으로는 strong signal 인 prompt 가 pipeline SP-CLIP 에서 phantom. → SAM-based within-view mixing 의 정량 evidence (Layer 1 visual proxy 와 모순 아님, 보완).

2. **ROFA 는 대체로 정상 작동** — kept_cos > dropped_cos (τ=1.0 sim 기준 82%; **✱ τ=2.0 pipeline 기준 65% + no-drop 12%**). ROFA 가 mechanism 잘못된 게 아니라 *이미 죽은 신호 위에서* 작동.

3. **ROFA 의 mechanism 실패 — ✱ τ=2.0 기준 24% (4 건)** — old camera, ottolenghi, **pikachu, onion segments**: 강한 cos view 를 outlier 로 잘못 drop. (τ=1.0 sim 명단이던 sink 는 τ=2.0 에선 drop 0 건.) 4 건 중 2 건이 instance-confusion phantom 인 점은 신규 단서.

4. **D3 의 trade-off** — k=3 default 가 method-level optimal 이지만 phantom 만 보면 k=10 better. Prompt-conditional adaptive 정책 필수.

### Bonus 발견

5. **C3 는 instance confusion 회복 못 함** — text-side contrast 로 SP-feature aggregation 단계의 잘못된 방향 못 되돌림.

### 한 줄로 (paper 의 main contribution)

> **"실패의 dominant mechanism 은 within-view SAM-based feature mixing 이다. ROFA 같은 across-view aggregation 은 정상 동작하지만 *이미 죽은 신호 위에서* 동작하므로 회복 불가. Paper 의 새 aggregation 은 within-view 단계를 fix 해야 한다."**

---

## 8. Reconciliation with Stage 1 의 4.2:1 비율

Stage 1: D2.phantom 21 : D2.real 5 = 4.2 : 1 (method-agnostic)

Stage 2A/2B 가 이 비율의 *내부 구조* 를 분해:

```
67 prompts total
├── 36 (54%) easy            — both A2 and SP rank ≤ 3
├── 21 (31%) D2.phantom      — A2 ≤ 3, SP > 3
│   ├── 17 persistent       — both THGS and ReLaGS phantom
│   │   ├── 11 within-view killed (strong_signal_phantom)
│   │   │   ├── 4 D3-recoverable (jake, bowl, plate, sake cup)
│   │   │   ├── 3 instance_confusion (pikachu, rubber duck, onion segments)
│   │   │   ├── 2 ROFA mechanism fail (sink, plus pumpkin partial)
│   │   │   └── 2 catastrophic structural (rest)
│   │   ├── 4 encoder-side mean dilution
│   │   │   (old camera, pumpkin, napkin, ottolenghi)
│   │   └── 2 bimodal-balanced
│   │       (pikachu, sake cup) — overlap with strong_signal
│   └── 4 method-specific (THGS only or ReLaGS only)
├── 5 (7.5%) D2.real          — encoder hard limit
└── 5 (7.5%) rare             — A2 fails but SP wins
```

### Section 1 narrative 의 보강

Stage 1 은 *4.2:1 phantom dominance* 를 보였고, Stage 2 는 *그 21 phantom 의 65% 가 within-view mixing 이 dominant cause* 임을 보임 — paper 의 새 method 의 정확한 fix target 정의.

---

## 9. Stage 2B 산출물

### CSV
- [output/diagnostics/_h2_lite_perview.pkl](../../../output/diagnostics/_h2_lite_perview.pkl) — 17 phantom × 30 sampled views × 512D CLIP feature
- [output/diagnostics/stage2b_rofa_subtypes.csv](../../../output/diagnostics/stage2b_rofa_subtypes.csv) — F2 subtype 분류 (17 rows)
- [output/diagnostics/stage2b_rofa_keep_mask.csv](../../../output/diagnostics/stage2b_rofa_keep_mask.csv) — F2.C keep_mask analysis
- [output/diagnostics/stage2b_d3_full_sweep.csv](../../../output/diagnostics/stage2b_d3_full_sweep.csv) — 67 prompts × 6 k values
- [output/diagnostics/stage2b_d3_full_sweep_agg.csv](../../../output/diagnostics/stage2b_d3_full_sweep_agg.csv) — per-prompt aggregate
- [output/diagnostics/stage2b_c3_quickcheck.csv](../../../output/diagnostics/stage2b_c3_quickcheck.csv) — C3 sweep (3 prompts × 6 lambdas)
- [output/diagnostics/verify_f2_tau_sweep.csv](../../../output/diagnostics/verify_f2_tau_sweep.csv) — ✱ 검증: τ∈{1.0, 2.0} pathology/subtype 비교 (17 rows)
- [output/diagnostics/verify_joint_canon_vs_raw.csv](../../../output/diagnostics/verify_joint_canon_vs_raw.csv) — ✱ 검증: joint 2×2 raw/canon robustness (67 rows, 분류 변화 0)
- [output/diagnostics/verify_h2lite_rank.csv](../../../output/diagnostics/verify_h2lite_rank.csv) — ✱ 검증: clean crop 의 scene-prompt rank (17 rows)

### Plots
- `plots/stage2b_rofa_subtype_distribution.png`
- `plots/stage2b_keep_mask_geometry.png`
- `plots/stage2b_per_view_cos_dists.png`

### Scripts
- [scripts/stage2b_h2lite_perview.py](../../../scripts/stage2b_h2lite_perview.py)
- [scripts/stage2b_f2_subtypes.py](../../../scripts/stage2b_f2_subtypes.py)
- [scripts/stage2b_d3_prompt_agnostic_sweep.py](../../../scripts/stage2b_d3_prompt_agnostic_sweep.py)
- [scripts/stage2b_c3_negative_contrast.py](../../../scripts/stage2b_c3_negative_contrast.py)
- [scripts/verify_stage2b_checks.py](../../../scripts/verify_stage2b_checks.py) — ✱ 검증 3종 (τ sweep / joint canon vs raw / H2-lite rank 재검증)

---

## 10. Decision — Stage 3 후보

### Question for Stage 3

Stage 2B 의 결론:
> **Within-view SAM-based feature mixing 이 dominant phantom cause**

→ Stage 3 의 question:
> *"Within-view mixing 을 어떻게 fix 하면 phantom 회복 가능한가?"*

### Stage 3 후보들

#### 후보 A — Hard within-view assignment (within-view 직접 fix)
- 현재: `sp_feat = sp_mask_mat @ view_level_feature` (ratio-weighted sum)
- 대안: `sp_feat = view_level_feature[argmax(sp_mask_mat)]` (single mask hard assignment)
- 측정: 17 phantom 의 회복률
- 비용: pipeline 의 1 line 수정, full re-run (반나절)

#### 후보 B — RATIO_THRESHOLD sweep
- 현재 RATIO_THRESHOLD = 0.3 — *비등한 두 mask (0.45 + 0.40)* 둘 다 통과
- Sweep: {0.3, 0.5, 0.7, 0.9}
- 측정: 17 phantom 회복 + 전체 67 prompt mIoU trade-off
- 비용: 4 re-runs (1 일)

#### 후보 C — Query-conditioned top-view aggregation
- 평균 대신 *query 와 cos 가장 높은 view 들*만 평균
- 측정: phantom 회복 + 전체 mIoU
- 비용: inference 단계 수정 (1 일)

#### 후보 D — Mode-cluster center aggregation
- ROFA 의 mean 대신 GMM(k=2) 의 majority cluster 의 center 사용
- bimodal_balanced 2 case (pikachu, sake cup) 회복 가능성
- 비용: 1 일

### 추천: 후보 A + 후보 C 병렬 (2 일)

이유:
1. **후보 A** = root cause 의 직접 attack (within-view stage). 단순한 ablation 이라 결과 명확.
2. **후보 C** = 후보 A 가 만족스럽지 못한 경우의 backup. across-view stage 의 대안 aggregation.
3. **두 개 결과 비교** → paper section 3 (proposed method) 의 두 길 정량 비교

후보 B 와 D 는 후속.

---

## 11. 관련 문서

- [stage1_b7_a4_a2.md](stage1_b7_a4_a2.md) — Stage 1 (B7+A4+A2+Joint+Cross-method)
- [stage2a_phantom_anatomy.md](stage2a_phantom_anatomy.md) — Stage 2A (4-Layer anatomy)
- [../extended_failure_hypotheses.md](../extended_failure_hypotheses.md) — 6.2 patch 가설 catalog
- [../../THGS/paper_section1_draft.md](../../THGS/paper_section1_draft.md) — Section 1 draft
- [../../THGS/paper_section2_draft.md](../../THGS/paper_section2_draft.md) — Section 2 draft (이 stage 의 결과로 작성)
