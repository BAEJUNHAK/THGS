# LERF-OVS Deep Failure Analysis

> Generated 2026-06-04. Builds on Phase 1 diagnostic. Answers 5 sub-questions to refine attack hypothesis before Phase 2.

Data source: `output/diagnostics/lerf_ovs_per_prompt.csv` (208 (scene, prompt, frame) rows, 67 unique prompts).

## Q1. Frame stability — view-dependent failure 인가?

Per-prompt: actual IoU std across that prompt's frames. Higher std = more view-dependent.

N prompts with ≥2 frames: 47 / 67

| Type | N (≥2 frames) | Actual IoU std avg | Actual IoU std max | Actual IoU (max-min) avg |
|---|---|---|---|---|
| A | 12 | 0.047 | 0.318 | 0.123 |
| C | 34 | 0.073 | 0.277 | 0.177 |
| B | 1 | 0.047 | 0.047 | 0.116 |

Type A 내부 sub-type 별 stability:

| Sub-type | N (≥2 frames) | Actual IoU std avg | Actual range avg | Failure pattern |
|---|---|---|---|---|
| A1 | 5 | 0.000 | 0.000 | consistent (low std) |
| A2 | 5 | 0.110 | 0.288 | fluctuating |
| A3 | 2 | 0.008 | 0.018 | consistent (low std) |

### Type A에서 가장 흔들리는 prompt top 10 (std 큰 순)

| Scene | Prompt | Frames | Oracle mean | Actual mean | Actual std | Actual range | Sub-type |
|---|---|---|---|---|---|---|---|
| ramen | sake cup | 6 | 0.716 | 0.170 | 0.318 | [0.00, 0.88] | A2 |
| ramen | bowl | 5 | 0.748 | 0.148 | 0.186 | [0.00, 0.44] | A2 |
| teatime | bear nose | 3 | 0.974 | 0.100 | 0.032 | [0.06, 0.14] | A2 |
| ramen | corn | 5 | 0.723 | 0.073 | 0.015 | [0.05, 0.09] | A2 |
| ramen | spoon | 2 | 0.801 | 0.219 | 0.013 | [0.21, 0.23] | A3 |
| ramen | kamaboko | 7 | 0.890 | 0.064 | 0.003 | [0.06, 0.07] | A3 |
| figurines | miffy | 2 | 0.476 | 0.000 | 0.000 | [0.00, 0.00] | A1 |
| figurines | pikachu | 2 | 0.914 | 0.000 | 0.000 | [0.00, 0.00] | A1 |
| figurines | pirate hat | 4 | 0.844 | 0.000 | 0.000 | [0.00, 0.00] | A1 |
| ramen | onion segments | 7 | 0.758 | 0.000 | 0.000 | [0.00, 0.00] | A1 |

## Q2. CLIP top-1 SP characterization — 틀린 pick은 얼마나 틀렸나?

Per Type: distribution of CLIP top-1 SP's oracle IoU (= "if you blindly trust CLIP top-1, what's your IoU?")

| Type | N | top1_oracle_iou mean | median | min | max | > 0.5 share | < 0.1 share |
|---|---|---|---|---|---|---|---|
| A | 20 | 0.085 | 0.000 | 0.000 | 0.874 | 10% | 90% |
| C | 46 | 0.612 | 0.774 | 0.000 | 0.979 | 63% | 20% |
| B | 1 | 0.289 | 0.289 | 0.289 | 0.289 | 0% | 0% |

**Type A의 top-1 SP의 oracle IoU 분포**:

- 0.0  : 18 prompts (90%) — totally wrong region
- 0.05-0.3: 0 prompts — barely overlapping
- 0.3-0.5: 0 prompts — semantic near-miss
- > 0.5: 2 prompts — top-1 is actually OK

**결론**: **CLIP top-1 가 완전 엉뚱한 위치를 picking** → spatial misalignment dominant


## Q3. 객체 크기 효과 — 작은 객체가 더 잘 실패하나?

gt_pixels (per prompt, summed across frames) quartiles: Q1=15716, Q2(median)=31259, Q3=65387

| Size bin | N | Oracle mean | Actual mean | Loss mean | Type A share | A1 share |
|---|---|---|---|---|---|---|
| XS | 16 | 0.718 | 0.373 | 0.344 | 7/16 (44%) | 6/16 (38%) |
| S | 17 | 0.852 | 0.681 | 0.171 | 2/17 (12%) | 1/17 (6%) |
| M | 17 | 0.819 | 0.468 | 0.351 | 7/17 (41%) | 4/17 (24%) |
| L | 17 | 0.857 | 0.637 | 0.220 | 4/17 (24%) | 1/17 (6%) |

Correlation(gt_pixels_total, iou_lost) = **-0.118**
→ 객체 크기와 실패 강도는 사실상 무관 (|corr| < 0.15).

## Q4. Oracle SP 개수 — 실패 prompt가 더 많은 SP를 필요로 하나?

| Type | N | SP count mean | 1 SP | 2 SP | 3 SP |
|---|---|---|---|---|---|
| A | 20 | 1.55 | 12 | 5 | 3 |
| B | 1 | 1.00 | 1 | 0 | 0 |
| C | 46 | 1.48 | 32 | 6 | 8 |

**A1 (=12 prompts) Oracle SP count distribution**: mean=1.58, 1 SP=7, 2 SP=3, 3 SP=2

## Q5. A1 prompt들의 공통점 — 단어 수, 길이 등

| Group | N | Word count mean | Char count mean | Word count distribution |
|---|---|---|---|---|
| A1 | 12 | 1.25 | 7.8 | {1: 9, 2: 3} |
| C (success) | 46 | 1.91 | 10.1 | {1: 18, 2: 17, 3: 8, 4: 3} |

**A1 prompts (12)**: ['bag', 'cabinet', 'hand', 'hooves', 'miffy', 'onion segments', 'ottolenghi', 'pikachu', 'pirate hat', 'pour-over vessel', 'pumpkin', 'spoon']

**C prompts (samples)**: ['Stainless steel pots', 'apple', 'bag of cookies', 'chopsticks', 'coffee', 'coffee mug', 'dall-e brand', 'dark cup', 'egg', 'frog cup', 'glass of water', 'green apple', 'green toy chair', 'jake', 'ketchup']...
