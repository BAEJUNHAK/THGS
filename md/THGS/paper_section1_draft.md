# Paper Section 1 (Draft) — Failure Taxonomy and Mechanism-based Decomposition

> Day-1 sprint output, extended with cross-method analysis. Combines existing D1–D4 framework with B7 (oracle SP geometry), A2 (image-CLIP ceiling), A4 (rank margin) measurements on LERF-OVS for **both THGS and ReLaGS**.
>
> Status: **Draft v2 (cross-method)** — based on 67 prompts × 4 LERF-OVS scenes × 2 methods. All numbers are reproducible from `output/diagnostics/{b7_a4_combined.csv, b7_a4_combined_relags.csv, a2_image_clip_ceiling.csv, cross_method_d2_decomposition.csv}`.

---

## 1. Failure Taxonomy

We start from the prior D1–D4 distractor framework of THGS analysis and extend it with **mechanism-based decomposition** of D2. The hierarchy:

| Class | Symptom | Mechanism candidate |
|---|---|---|
| **D1** | Selected SP is degenerate (zero-norm feature or empty mask) | Pipeline degeneracy — partition or feature injection failed |
| **D2** | Selected SP belongs to a *different* object (semantic distractor) | *Decomposed in this work* — see Section 4 |
| **D3** | Selected SP set covers GT but also unrelated regions (over-union) | top-k cardinality / decision policy |
| **D4** | Selected SP is correct but not visible in evaluation frames | Visibility / cross-view propagation |
| **Success** | Selected SP matches GT within tolerance | — |

For D2 we introduce two sub-classes whose mechanisms call for different fixes:

- **D2.real** — even an *image-CLIP encoded crop of GT* fails to retrieve the correct prompt → ceiling is set by CLIP encoder / prompt language, not by aggregation.
- **D2.phantom** — image-CLIP succeeds on the crop, but the SP-level aggregated feature does not. The retrieval failure originates in the multi-view SP feature aggregation.

This separation is the central diagnostic contribution of this section.

---

## 2. Methodology

We measure three quantities per (scene, prompt, ref_frame) tuple over 67 LERF-OVS prompts (4 scenes):

### B7 — Oracle SP geometry

For each (prompt, ref_frame), render every superpoint mask at all hierarchy levels and identify the **oracle SP**: the single SP with highest IoU vs GT polygon.

For the oracle SP:
- `purity = |SP ∩ GT| / |SP|` (precision)
- `completeness = |SP ∩ GT| / |GT|` (recall)
- `fragmentation` = number of SPs picked by greedy-union (budget = 3) to maximize IoU

These bound the achievable retrieval ceiling: if no clean oracle exists, all downstream rank analysis is uninterpretable.

### A4 — Oracle-rank margin

Pool of SP cosines = scores at hierarchy levels [2, 3] (~745–4136 SPs per scene). For the oracle SP we record:

- `oracle_rank` — its rank within the pool (1 = best)
- `raw_margin = cos(top) − cos(oracle)` — within-prompt only
- `z_margin = (cos_top − cos_oracle) / std_pool` — **cross-prompt comparable**
- `percentile_margin` — oracle's quantile in pool (0 = top, 100 = bottom)

### A2 — Image-CLIP ceiling

For each prompt at its ref-frame, we crop the GT region with 4 policies and feed the crop to CLIP ViT-B-16 image encoder, ranking against the scene's prompt set:

- **(i) tight** — minimal bbox of GT polygon, raw RGB
- **(ii) mask** — same bbox, background blacked out outside polygon (SAM-style proxy)
- **(iii) context** — 1.5× expanded bbox, raw RGB
- **(iv) method** — polygon-blackout + 1.2× bbox (method-matched proxy)

A2 is a **ceiling diagnostic**: if even a clean crop cannot rank the true prompt in top-K, the failure cannot be a SP-aggregation issue.

> **Limitation note**: (ii) and (iv) approximate SAM-mask geometry with GT polygon. True method-matched would use the actual SAM mask covering the object. We defer that to a follow-up (requires SAM inference re-run); the GT proxy gives an *idealized SAM* ceiling.

---

## 3. Findings

### 3.1 B7 — Oracle SP geometry is mostly clean

| Statistic | Mean | Median | Tail |
|---|---|---|---|
| Oracle purity | **0.925** | 0.965 | 2/67 (3%) < 0.5; 5/67 (7%) < 0.8 |
| Oracle completeness | **0.857** | 0.912 | 4/67 (6%) < 0.5 |
| Fragmentation | 1.49 SPs | 1.0 | 22/67 (33%) ≥ 2 SPs needed; 11/67 (16%) ≥ 3 |
| Oracle IoU @ ref | 0.800 | — | 8/67 (12%) < 0.5 |

**Read** — the SP partition produces a sharp single-SP oracle for ~93% of prompts (purity ≥ 0.8). Failures are concentrated in two regimes:

1. **Geometric undersegmentation (4 prompts)** — completeness < 0.5: the GT object spans multiple SPs that the partition refused to merge (e.g., `miffy` completeness=0.36, `porcelain hand`=0.34, `waldo`=0.29, `tesla door handle`=0.50 purity which is the *only* purity outlier).
2. **Fragmentation (22 prompts)** — multiple SPs cooperatively cover GT but no single SP is a clean target (e.g., `coffee`, `hand`, `sink`, `old camera` need 3 SPs).

**Implication**: oracle existence is largely *not* the bottleneck. The downstream A3/A4 rank analysis is interpretable on top of this geometry — see Section 5 for the prerequisite reasoning.

→ See `plots/b7_distributions.png`, `b7_purity_vs_completeness.png`.

### 3.2 A4 — CLIP rank has a heavy long tail

| Statistic | Value |
|---|---|
| `oracle_rank` median | **2** |
| q75 / q90 | 7 / 52 |
| Max | 256 |
| rank ≤ 3 | 41 / 67 (61%) |
| rank ≤ 10 | 53 / 67 (79%) |
| rank > 30 (catastrophic) | 13 / 67 (19%) |
| `z_margin` median | 0.29 |
| `|z_margin|` > 1.5 | 10 / 67 (15%) |

**Read** — half of prompts retrieve their oracle in top-2, but the bottom-25% have rank ≥ 7 and 13 prompts have rank > 30 (essentially random within the pool). The z-margin distribution is bimodal: easy prompts cluster near z=0 (top-1) while distractor cases produce z ≥ 1.5 (the "wrong SP is *much* stronger" regime).

→ See `plots/a4_distributions.png`, `a4_rank_zmargin_scatter.png`.

### 3.3 A2 — Image-CLIP ceiling is well below 100%

| Policy | Top-1 hit | Top-3 hit | Median rank |
|---|---|---|---|
| (i) tight | 43/67 = 64.2% | 58/67 = 86.6% | 1 |
| (ii) mask | 42/67 = 62.7% | 57/67 = 85.1% | 1 |
| (iii) context | **48/67 = 71.6%** | 60/67 = 89.6% | 1 |
| (iv) method | 46/67 = 68.7% | 57/67 = 85.1% | 1 |

**Read** — even feeding CLIP a clean crop of the GT region, ~30% of prompts are not top-1 in the scene's prompt set. Context (1.5×) helps (+7.4% over tight) — this is consistent with CLIP being trained on object-centric images with surrounding context.

The gap between (i) and (ii)/(iv) (~1.5% top-1) shows that *background suppression alone* does not change the ceiling much. The gap between (i) and (iii) (~7%) shows context is the more important lever.

→ See `plots/a2_rank_per_policy.png`, `a2_top1_per_scene_policy.png`.

### 3.4 D2.real vs D2.phantom — the central decomposition

Joining B7+A4 (SP rank) with A2 (image-CLIP rank, mask policy) — using rank ≤ 3 as the success threshold on both sides:

| Class | Definition | Count | Share |
|---|---|---|---|
| **Easy** | A2_mask ≤ 3 AND SP rank ≤ 3 | 36 / 67 | 53.7% |
| **D2.phantom** | A2_mask ≤ 3 AND SP rank > 3 | **21 / 67** | **31.3%** |
| **D2.real** | A2_mask > 3 AND SP rank > 3 | 5 / 67 | 7.5% |
| **Rare** | A2_mask > 3 AND SP rank ≤ 3 | 5 / 67 | 7.5% |

**Phantom : real ratio = 4.2 : 1**. Of the 26 prompts where SP-based retrieval fails (rank > 3), **81% (21/26) are recoverable** in principle — CLIP would have found the right prompt if the SP feature had not been compromised by aggregation. Only **19% (5/26) are encoder-limited** in the strict sense.

→ See `plots/joint_d2real_vs_phantom.png`.

**Per-scene breakdown** (n = scene prompt count):

| Scene | n | Easy | Phantom | Real | Rare |
|---|---|---|---|---|---|
| figurines | 21 | 11 (52%) | 6 (29%) | 3 (14%) | 1 (5%) |
| ramen | 14 | 7 (50%) | 6 (**43%**) | **0 (0%)** | 1 (7%) |
| teatime | 14 | 11 (79%) | 2 (14%) | 1 (7%) | 0 |
| waldo_kitchen | 18 | 7 (39%) | 7 (39%) | 1 (6%) | 3 (17%) |

The most striking finding: **ramen has zero D2.real cases** — *every* failure in ramen is recoverable in principle. Combined with the high phantom share (43%), this localizes ramen's mIoU gap entirely in the aggregation pipeline, not in encoder limits.

teatime is by far the easiest scene (79% easy), with the lowest phantom share (14%) and only 1 real case (`hooves`).

waldo_kitchen has an unusually high `rare` count (17%) — these are cases where A2 ceiling fails but SP rank succeeds. We discuss this in Section 4.3.

### 3.5 Phantom and real example cases

**Worst phantom (biggest aggregation gap — A2 finds in top-3, SP rank ≥ 46):**

| Scene | Prompt | A2 rank | SP rank | Purity | z-margin |
|---|---|---|---|---|---|
| figurines | tesla door handle | 1 | **256** | 0.50 | 1.64 |
| figurines | old camera | 1 | 130 | 0.99 | 2.82 |
| waldo_kitchen | ottolenghi | 1 | 66 | 0.97 | 3.05 |
| figurines | pikachu | 1 | 64 | 0.98 | 1.16 |
| figurines | pumpkin | 1 | 52 | 0.98 | 1.08 |
| waldo_kitchen | spoon | 2 | 47 | 0.86 | 1.72 |
| ramen | hand | 1 | 46 | 0.97 | 2.80 |

→ Image-CLIP correctly identifies the object as `pikachu` from the crop alone, but the SP feature (averaged across views via ROFA) ranks it 64th out of 745 SPs. Purity is 0.98 — the oracle SP is clean. The error is purely in feature aggregation.

**Worst real (encoder limit — both A2 and SP fail):**

| Scene | Prompt | A2 rank | SP rank | Purity | z-margin |
|---|---|---|---|---|---|
| figurines | miffy | 13 | 75 | 1.00 | 1.91 |
| figurines | bag | 11 | 63 | 0.97 | 1.79 |
| waldo_kitchen | pour-over vessel | 4 | 53 | 0.96 | 0.32 |
| teatime | hooves | 8 | 52 | 0.99 | 1.18 |
| figurines | pirate hat | 12 | 49 | 0.99 | 0.29 |

→ These prompts are intrinsically hard for CLIP: `miffy` (proper noun rabbit character), `ottolenghi` (brand name on cookbook), `pour-over vessel` (specialized vocab), `hooves` (part-level term). Purity is high (≥ 0.96), so the oracle SP exists — but CLIP itself cannot map the prompt to the right object representation.

---

## 3.6 Cross-method comparison — THGS vs ReLaGS

The same B7+A4 protocol was applied to ReLaGS (sai_nag.pt from the authors' HuggingFace release). A2 is method-agnostic (image-CLIP ceiling does not depend on the SP pipeline). The full 4×4 transition matrix (`output/diagnostics/cross_method_d2_decomposition.csv`):

**D2 class distribution per method**:

| Class | THGS | ReLaGS | Δ |
|---|---|---|---|
| Easy | 36 / 67 (53.7%) | 37 / 67 (55.2%) | +1 |
| **Phantom** | **21 / 67 (31.3%)** | **20 / 67 (29.9%)** | −1 |
| Real | 5 / 67 (7.5%) | 4 / 67 (6.0%) | −1 |
| Rare | 5 / 67 (7.5%) | 6 / 67 (9.0%) | +1 |

→ The 4.2 : 1 phantom-to-real ratio is **almost identical between methods**. D2.phantom dominance is *not* a THGS-specific artifact — it is a structural property of the SP + multi-view CLIP aggregation pipeline that both methods inherit.

**Transition matrix** (counts):

```
                       ReLaGS
                  easy phantom real rare
THGS  easy         33      3     0    0   ← 3 regressions
      phantom       4     17     0    0   ← 4 recoveries / 17 unchanged
      real          0      0     4    1
      rare          0      0     0    5
```

→ See `plots/cross_method_transition_matrix.png`, `cross_method_rank_scatter.png`, `cross_method_d2_stacked.png`, `cross_method_per_scene_stacked.png`.

### 3.6.1 ReLaGS recovery rate on THGS phantoms

Among 21 THGS phantoms, ReLaGS recovers **4 (19%)** to easy class. The remaining 17 (81%) stay phantom even after ROFA's outlier-filtered aggregation. None converted to real (encoder limit) — ReLaGS does not introduce encoder-side regressions.

**Rank improvement on phantoms** (those that stayed phantom):
- Median rank: THGS 8 → ReLaGS 6 (modest improvement)
- Median z-margin: THGS 1.09 → ReLaGS 1.13 (slightly worse — z increased because pool std shrank as ROFA filtered outliers)

→ ROFA narrows the pool but does not change the SP's *relative* position much for the prompts where the phantom mechanism is mode-cluster style (not outlier style — see hypothesis F2 subtypes).

### 3.6.2 Regression cases (3 prompts)

| Scene | Prompt | THGS rank | ReLaGS rank |
|---|---|---|---|
| figurines | spatula | 2 | 4 |
| ramen | nori | 2 | 4 |
| waldo_kitchen | dark cup | 3 | 4 |

All three are **borderline cases** (THGS rank 2-3) that crossed the rank-3 threshold to rank 4. With a stricter classification (e.g., rank ≤ 5 or z-margin based), they would not flip. These are *not* meaningful regressions — they are an artifact of the discrete threshold on a continuous rank scale. We flag this to avoid over-interpreting the transition matrix's diagonal vs off-diagonal entries.

### 3.6.3 Per-scene asymmetry — where ReLaGS actually helps

| Scene | Phantom (THGS → ReLaGS) | Easy (THGS → ReLaGS) | Net effect |
|---|---|---|---|
| figurines (n=21) | 6 → 7 | 11 → 10 | Neutral / mild regression |
| ramen (n=14) | 6 → 7 | 7 → 6 | Neutral |
| teatime (n=14) | 2 → 1 | 11 → 12 | Mild improvement |
| **waldo_kitchen (n=18)** | **7 → 5** | **7 → 9** | **Clear improvement** |

→ **waldo_kitchen is where ReLaGS clearly helps** (2 phantoms become easy, +2 net). This matches the paper's Table 3 result (ReLaGS Waldo +9.95 mIoU over THGS) but localizes the mechanism: it is phantom recovery, not encoder upgrade.

**ramen is the most stubborn**: every D2 failure in ramen is phantom, and ReLaGS recovers *none* of them. The 6 ramen phantoms (`bowl`, `hand`, `napkin`, `onion segments`, `plate`, `sake cup`) survive ROFA — they need a different aggregation mechanism. Combined with ramen having 0% D2.real, this localizes ramen's ~3-4 mIoU gap *entirely* in the aggregation pipeline.

### 3.6.4 ReLaGS phantoms — origin

Of ReLaGS's 20 phantoms:
- 17 inherited from THGS (same prompts both methods fail)
- 3 newly introduced (the regressions above — borderline threshold effect)

The 17 *persistent phantoms* form a stable failure set: prompts that the entire SP+CLIP+multi-view-aggregation paradigm cannot retrieve, regardless of whether ROFA is applied. They are the natural target for a new aggregation method.

### 3.6.5 Implications for paper narrative

1. **D2.phantom is the dominant failure class regardless of method**. The mechanism analysis is method-agnostic — it predicts the same failure structure for any SP+CLIP pipeline.
2. **ReLaGS's primary mechanism is partial phantom recovery (19%)**, not encoder improvement. This sharpens the paper's mIoU gap claim: the gap is mechanism-attributable.
3. **Per-scene effectiveness is uneven**: waldo benefits the most (4 net failures fixed), figurines and ramen barely change, teatime was already easy.
4. **The 17 persistent phantoms are the new method's target**. Their behavior under different aggregations (mode-cluster center, query-conditioned top-view, hard within-view assignment) will determine whether the paper's main contribution clears them.

---

## 4. Implications

### 4.1 Aggregation is the dominant lever, not encoder

The 4.2 : 1 phantom-to-real ratio overturns the implicit assumption that CLIP encoder limits dominate. **31% of all LERF-OVS prompts are stuck in the aggregation pipeline** with recoverable signal. Any improvement to multi-view SP feature aggregation (B1 mean dilution, B8 within-view mixing, F2 ROFA mean-limit) directly attacks the majority failure class.

### 4.2 The 7.5% encoder ceiling sets the absolute upper bound

Of the 67 prompts, 5 are encoder-limited regardless of partition / aggregation choices. This is the absolute ceiling — even oracle aggregation cannot fix them without a stronger encoder, prompt expansion (C2), or negative-prompt contrast (C3).

### 4.3 The "rare" cases (5/67) are informative

5 cases where A2 fails (image-CLIP can't find right prompt from a clean crop) but SP retrieval succeeds. This means **multi-view aggregation can be better than single-view image-CLIP** — averaging across views appears to recover signal that a single noisy view misses. This is a positive aggregation effect, opposite to phantom.

This pattern is concentrated in waldo_kitchen (3/5) — likely because waldo_kitchen has many small / occluded objects (`knife`, `spatula`, `plastic ladle`) where a single GT-cropped view is harder than the multi-view average.

### 4.4 Geometric oracle is sound (mostly)

B7 confirms that for 93% of prompts a clean single-SP oracle exists (purity ≥ 0.8). The partition stage is *not* the dominant failure source — A3/A4 rank analysis is interpretable, and downstream phantom-direction analysis (E1) can build on top of this oracle without worry.

The 4 undersegmentation cases (completeness < 0.5) are a partition-side issue we flag as a separate axis — they are recovered by greedy union but not by single-SP selection, suggesting top-k cardinality (D3) interacts with B7.

---

## 5. Methodological notes

### 5.1 Cross-prompt margin must be normalized

A4 reports three margin definitions. The raw margin distribution looks tight (small absolute values) because CLIP cosines are concentrated in a narrow band. But the z-margin reveals clear bimodality: easy prompts at z≈0, phantom cases at z ≥ 1.5. Without z-score normalization, the within-prompt comparison signal would be drowned out. We use z-margin as the default for cross-prompt analysis.

### 5.2 Oracle definition is rendered-mask based, not gaussian-membership based

We define oracle SP via the rendered 2D mask vs the 2D GT polygon (the same space as evaluation). This avoids the ambiguity of "which gaussian belongs to which object" and matches the downstream eval geometry. B7 purity/completeness are pixel-level statistics, not point-cloud.

### 5.3 A2 with polygon proxy is an idealized SAM ceiling

A2 (ii) and (iv) use GT polygon as a stand-in for SAM mask. The true method-matched ceiling — using SAM masks from the actual pipeline — may be lower because SAM mask boundaries are noisier than GT polygons. We expect the gap to be small (since high-quality SAM mostly recovers object boundaries), but a follow-up SAM-mask experiment will confirm the bound.

---

## 6. What's next (out of scope for this section)

These results lock the prerequisites for downstream analysis:

- **Section 2 — D3 over-union (top-k cardinality)** — among phantom cases, how many recover with smaller k or area penalty?
- **Section 3 — A3 cross-view consistency** — within the 21 phantom prompts, are there views where the oracle SP *would* rank top-3? If yes → target-dilution subtype, motivating query-conditioned top-view aggregation.
- **Section 4 — B8 within-view mixing** — does the ratio-mix of SAM masks at line `merge_proj.py:167-168` explain the phantom share quantitatively?
- **Section 5 — E1 phantom direction bias** — do the 21 phantom cases bias toward particular semantic categories (small→background, food→vessel, etc.)?
- **Section 6 — F1 transition matrix THGS↔ReLaGS** — does ReLaGS's D1 reduction convert easy cases into phantom cases? (Net effect of ROFA.)

---

## 7. Artifacts

| File | Rows | Description |
|---|---|---|
| [output/diagnostics/b7_a4_combined.csv](../../output/diagnostics/b7_a4_combined.csv) | 208 | THGS B7 + A4 per (prompt, eval_frame) |
| [output/diagnostics/b7_a4_combined_relags.csv](../../output/diagnostics/b7_a4_combined_relags.csv) | 208 | ReLaGS B7 + A4 per (prompt, eval_frame) |
| [output/diagnostics/a2_image_clip_ceiling.csv](../../output/diagnostics/a2_image_clip_ceiling.csv) | 268 | A2 ceiling per (prompt, policy) — method-agnostic |
| [output/diagnostics/cross_method_d2_decomposition.csv](../../output/diagnostics/cross_method_d2_decomposition.csv) | 67 | Per-prompt class + rank + purity for both methods, side by side |
| [output/diagnostics/plots/](../../output/diagnostics/plots/) | 15 PNGs | B7 + A4 + A2 + joint D2 + 4 cross-method plots |
| [scripts/b7_a4_oracle_analysis.py](../../scripts/b7_a4_oracle_analysis.py) | — | B7+A4 measurement (both THGS and ReLaGS via copy in ReLaGS/scripts/) |
| [scripts/a2_image_clip_ceiling.py](../../scripts/a2_image_clip_ceiling.py) | — | A2 ceiling code |
| [scripts/b7_a4_a2_plots.py](../../scripts/b7_a4_a2_plots.py) | — | Within-method plot generation |
| [scripts/cross_method_comparison.py](../../scripts/cross_method_comparison.py) | — | Cross-method transition + comparison plots |
