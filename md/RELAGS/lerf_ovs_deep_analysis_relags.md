# LERF-OVS Deep Failure Analysis

> Generated 2026-06-04. Builds on Phase 1 diagnostic. Answers 5 sub-questions to refine attack hypothesis before Phase 2.

Data source: `output/diagnostics/lerf_ovs_per_prompt.csv` (208 (scene, prompt, frame) rows, 67 unique prompts).

## Q1. Frame stability — view-dependent failure 인가?

Per-prompt: actual IoU std across that prompt's frames. Higher std = more view-dependent.

N prompts with ≥2 frames: 47 / 67

| Type | N (≥2 frames) | Actual IoU std avg | Actual IoU std max | Actual IoU (max-min) avg |
|---|---|---|---|---|
| A | 12 | 0.069 | 0.392 | 0.171 |
| C | 34 | 0.062 | 0.220 | 0.150 |
| B | 1 | 0.044 | 0.044 | 0.102 |

Type A 내부 sub-type 별 stability:

| Sub-type | N (≥2 frames) | Actual IoU std avg | Actual range avg | Failure pattern |
|---|---|---|---|---|
| A1 | 4 | 0.000 | 0.000 | consistent (low std) |
| A2 | 7 | 0.118 | 0.292 | fluctuating |
| A3 | 1 | 0.000 | 0.000 | consistent (low std) |

### Type A에서 가장 흔들리는 prompt top 10 (std 큰 순)

| Scene | Prompt | Frames | Oracle mean | Actual mean | Actual std | Actual range | Sub-type |
|---|---|---|---|---|---|---|---|
| ramen | napkin | 5 | 0.420 | 0.196 | 0.392 | [0.00, 0.98] | A2 |
| ramen | bowl | 5 | 0.752 | 0.155 | 0.195 | [0.00, 0.46] | A2 |
| waldo_kitchen | knife | 3 | 0.408 | 0.233 | 0.181 | [0.00, 0.44] | A2 |
| ramen | sake cup | 6 | 0.719 | 0.042 | 0.053 | [0.00, 0.14] | A2 |
| ramen | corn | 5 | 0.711 | 0.004 | 0.008 | [0.00, 0.02] | A2 |
| teatime | bear nose | 3 | 0.974 | 0.001 | 0.000 | [0.00, 0.00] | A2 |
| figurines | pikachu | 2 | 0.916 | 0.000 | 0.000 | [0.00, 0.00] | A1 |
| figurines | tesla door handle | 2 | 0.573 | 0.000 | 0.000 | [0.00, 0.00] | A1 |
| ramen | onion segments | 7 | 0.803 | 0.000 | 0.000 | [0.00, 0.00] | A1 |
| ramen | plate | 4 | 0.756 | 0.000 | 0.000 | [0.00, 0.00] | A2 |

## Q2. CLIP top-1 SP characterization — 틀린 pick은 얼마나 틀렸나?

Per Type: distribution of CLIP top-1 SP's oracle IoU (= "if you blindly trust CLIP top-1, what's your IoU?")

| Type | N | top1_oracle_iou mean | median | min | max | > 0.5 share | < 0.1 share |
|---|---|---|---|---|---|---|---|
| A | 18 | 0.079 | 0.000 | 0.000 | 0.872 | 6% | 89% |
| C | 48 | 0.575 | 0.761 | 0.000 | 0.982 | 65% | 27% |
| B | 1 | 0.054 | 0.054 | 0.054 | 0.054 | 0% | 100% |

**Type A의 top-1 SP의 oracle IoU 분포**:

- 0.0  : 15 prompts (83%) — totally wrong region
- 0.05-0.3: 1 prompts — barely overlapping
- 0.3-0.5: 1 prompts — semantic near-miss
- > 0.5: 1 prompts — top-1 is actually OK

**결론**: **CLIP top-1 가 완전 엉뚱한 위치를 picking** → spatial misalignment dominant


## Q3. 객체 크기 효과 — 작은 객체가 더 잘 실패하나?

gt_pixels (per prompt, summed across frames) quartiles: Q1=15716, Q2(median)=31259, Q3=65387

| Size bin | N | Oracle mean | Actual mean | Loss mean | Type A share | A1 share |
|---|---|---|---|---|---|---|
| XS | 16 | 0.704 | 0.520 | 0.184 | 4/16 (25%) | 3/16 (19%) |
| S | 17 | 0.852 | 0.644 | 0.207 | 3/17 (18%) | 2/17 (12%) |
| M | 17 | 0.814 | 0.486 | 0.328 | 8/17 (47%) | 3/17 (18%) |
| L | 17 | 0.861 | 0.689 | 0.171 | 3/17 (18%) | 1/17 (6%) |

Correlation(gt_pixels_total, iou_lost) = **-0.074**
→ 객체 크기와 실패 강도는 사실상 무관 (|corr| < 0.15).

## Q4. Oracle SP 개수 — 실패 prompt가 더 많은 SP를 필요로 하나?

| Type | N | SP count mean | 1 SP | 2 SP | 3 SP |
|---|---|---|---|---|---|
| A | 18 | 1.72 | 9 | 5 | 4 |
| B | 1 | 1.00 | 1 | 0 | 0 |
| C | 48 | 1.31 | 37 | 7 | 4 |

**A1 (=9 prompts) Oracle SP count distribution**: mean=1.67, 1 SP=4, 2 SP=4, 3 SP=1

## Q5. A1 prompt들의 공통점 — 단어 수, 길이 등

| Group | N | Word count mean | Char count mean | Word count distribution |
|---|---|---|---|---|
| A1 | 9 | 1.33 | 8.1 | {1: 7, 2: 1, 3: 1} |
| C (success) | 48 | 1.83 | 9.9 | {1: 21, 2: 17, 3: 7, 4: 3} |

**A1 prompts (12)**: ['bag', 'cabinet', 'hand', 'hooves', 'onion segments', 'ottolenghi', 'pikachu', 'spoon', 'tesla door handle']

**C prompts (samples)**: ['Stainless steel pots', 'apple', 'bag of cookies', 'chopsticks', 'coffee', 'dall-e brand', 'egg', 'frog cup', 'glass of water', 'green apple', 'green toy chair', 'jake', 'kamaboko', 'ketchup', 'miffy']...
