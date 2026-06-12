# 실험 결과 카탈로그 — 코드 기반 정의 + 결과 표 (living document)

> 각 실험 (= 각 script) 의 "*무엇을 입력으로 받고 무엇을 계산하고 무엇을 출력하는지*" 를 **실제 code 기준으로** 정의 + 결과 한 줄.
>
> [intuition.md](intuition.md) 가 *해석/직관* 이라면, 이 문서는 *원본 코드의 거울 + 결과 표*.
>
> 새 stage 가 끝날 때마다 (1) 맨 위 *전체 실험 한눈에 보기* 표에 row 추가, (2) 본문에 stage section 추가, (3) 업데이트 로그에 한 줄 추가.

**마지막 업데이트**: Stage 2B 완료 시점

---

## 📋 전체 실험 한눈에 보기

| Stage | Exp | Script | 한 줄 정의 | 핵심 결과 |
|---|---|---|---|---|
| 1 | **B7** | [b7_a4_oracle_analysis.py](../../../scripts/b7_a4_oracle_analysis.py) | 각 prompt 의 oracle SP 의 purity / completeness / fragmentation 측정 | purity 평균 0.925, 67% 가 fragmentation=1 |
| 1 | **A4** | (B7 와 같은 script) | oracle SP 의 CLIP cosine rank + raw/z/percentile margin | median rank 2, 19% rank>30 (catastrophic) |
| 1 | **A2** | [a2_image_clip_ceiling.py](../../../scripts/a2_image_clip_ceiling.py) | 4 crop policy (tight/mask/context/method) 로 image-CLIP top-K | tight 64.2%, context 71.6% top-1 |
| 1 | **Joint** | [b7_a4_a2_plots.py](../../../scripts/b7_a4_a2_plots.py) | A4×A2 (threshold rank=3) 2×2 분류 → D2.phantom vs D2.real | 31% phantom : 7.5% real = **4.2:1** |
| 1 | **Cross-method** | [cross_method_comparison.py](../../../scripts/cross_method_comparison.py) | THGS↔ReLaGS 4×4 transition matrix | ReLaGS 가 phantom 19% 회복, **17 persistent** |
| 2A | **L1 (B8 proxy)** | [stage2a_layer1_b8_coverage.py](../../../scripts/stage2a_layer1_b8_coverage.py) | oracle SP 의 render mask 가 disconnected components 로 쪼개지는 비율 | mix_view_frac phantom = easy = 0.16 |
| 2A | **L2 forensic** | [stage2a_layer2_wrongtop1.py](../../../scripts/stage2a_layer2_wrongtop1.py) | CLIP wrong top-1 SP 의 정체 분류 (over_union / instance_confusion / background_drift) | 65% background drift, 18%/18% |
| 2A | **L2 trajectory** | (L2 forensic 과 같은 script) | 각 eval frame 의 oracle rank 측정 → target dilution vs structural | 76% structural, 24% target dilution |
| 2A | **L3 (D3)** | [stage2a_layer3_d3_topk.py](../../../scripts/stage2a_layer3_d3_topk.py) | 17 phantom 의 ref_frame 에서 k ∈ {1,2,3,5,10} top-k union IoU | default 0.203 → optimal 0.314 (+11.2pt) |
| 2A | **L4 montage** | [stage2a_layer4_montage.py](../../../scripts/stage2a_layer4_montage.py) + [_combine](../../../scripts/stage2a_layer4_combine_montage.py) | 17 × 4-panel (GT / Oracle / wrong top-1 / top-10) 시각화 | killer figure (3164×1612) |
| 2A | **Synthesis** | [stage2a_synthesize.py](../../../scripts/stage2a_synthesize.py) | layer 1-3 결과 join + rule-based primary mechanism attribution | 35% structural_ROFA, 18% D3_deep_pool, 18% inst_conf, 12% target_dil |
| 2B | **F2.A H2 lite** | [stage2b_h2lite_perview.py](../../../scripts/stage2b_h2lite_perview.py) | 17 oracle SP × 30 views → mask crop → CLIP image enc → prompt cos | 12/17 cos≥0.20; ✱rank 재검증 11/17 (65%) (재료 살아있음) |
| 2B | **F2.B subtype** | [stage2b_f2_subtypes.py](../../../scripts/stage2b_f2_subtypes.py) | per-view cos 분포 → 4 subtype 분류 (strong/mean_dil/bimodal/outlier) | **65% strong_signal, 0% outlier_handled** |
| 2B | **F2.C keep-mask** | (F2.B 와 같은 script) | ROFA simulate (tau=1.0; ✱pipeline default 는 tau=2.0) 의 kept vs dropped view cos 비교 | τ=1: 82% 정직 / **✱τ=2: 65%+12% no-drop, 실수 4 case** |
| 2B | **D3 full** | [stage2b_d3_prompt_agnostic_sweep.py](../../../scripts/stage2b_d3_prompt_agnostic_sweep.py) | 67 prompt × k ∈ {1,2,3,5,10,20} 의 method-level mIoU | k=3 (전체 0.542), k=10 (phantom 0.290) |
| 2B | **C3 quick** | [stage2b_c3_negative_contrast.py](../../../scripts/stage2b_c3_negative_contrast.py) | 3 phantom × λ ∈ {0, 0.1, 0.3, 0.5, 0.7, 1.0} negative contrast | pikachu 64→50 marginal, 나머지 실패 |
| ✱검증 | **Robustness ×3** | [verify_stage2b_checks.py](../../../scripts/verify_stage2b_checks.py) | τ∈{1,2} sweep + joint raw/canon + H2-lite rank 재검증 (2026-06-10) | Stage 1 robust (0 변화); F2.C 수치/명단 정정; onion segments 재분류 |
| 3.1 | **B8 replay** | [stage3_b8_replay.py](../../../scripts/stage3_b8_replay.py) | within-view 단계 verbatim 재현 (33 SP × 전체 view), clean/mixed/hard 3-신호 + mix_count 최초 직접 측정, 충실도 gate | **충실도 33/33 (cos≥0.9935)**; strong gap **0.0038**; per-view medR 1-7 |
| 3.1 | **B8 판정** | [stage3_b8_analyze.py](../../../scripts/stage3_b8_analyze.py) | 사전 등록 H-B8a/b/c 판정 (paired gap, mix-rate 상관, hard 회복률) | **a✗ b✗ c→분기③: within-view 무죄, B1 재조준** |
| 3.2 | **B1.A 단일view+궤적** | [stage3_2_b1a_trajectory.py](../../../scripts/stage3_2_b1a_trajectory.py) | 각 view 단독/누적 feature 를 실제 SP pool 에 넣은 canon rank (진짜 A3-postB8) | best 단일 ≤3 = **15/17 (88%)**, 좋은 view 비율 18% vs easy 50% → **R1 ✅ averaging 인과** |
| 3.2 | **B1.B ablation** | [stage3_2_b1b_ablation.py](../../../scripts/stage3_2_b1b_ablation.py) | 같은 per-view set 으로 12 aggregation variants 의 pool rank | **query-cond top-k: 15/17 회복+reg 0**; 정적 top-k 6/17; GMM/ROFA/uniform 실패 → **R2 ✅** |
| 3.2 | **B1.C 경쟁 분해** | [stage3_2_b1c_compare.py](../../../scripts/stage3_2_b1c_compare.py) | wrong-top1 replay (--targets_csv) + oracle 과 paired coherence/cos/visibility | **R3 ✅ dispersion-defeat p=0.013**; wrong 의 평균 prompt-cos 가 oracle 보다 높음; pumpkin wrong = zero-norm (D1 잔존) |
| 3.3 | **full-pool rank** | [stage3_3_fullpool_rank.py](../../../scripts/stage3_3_fullpool_rank.py) | all-SP dump (--dump_all_sp, gate 98-100%) 로 동결 없는 전 pool query-top-5 재채점 | phantom +6 / other +5 / **easy −10** — rank 제로섬; "15/17" 환상 최종 확정 |
| 3.3 | **mask-IoU (R4)** | [stage3_3_mask_eval.py](../../../scripts/stage3_3_mask_eval.py) | 변형별 top-3 union 렌더 × 208 (prompt,frame) mIoU | **R4 FAIL**: phantom 0.203→**0.402** ↔ easy −9.8pt, full-67 −0.1pt. baseline 0.5424=stage2b 재현 |
| 3.3 | **impostor 법의학 (R5)** | [stage3_3_impostor_forensics.py](../../../scripts/stage3_3_impostor_forensics.py) | 승리 view 렌더 + ref_frame GT containment + montage | **R5: semantic confusion** (GT-포함 2/16) — 단 5/16 은 precision≥0.87 = **진실의 조각** (granularity) |
| 3.3 | ✱hybrid 탐색 | (inline, 사후) | α·mean + (1−α)·top5 rank sweep | **α=0.3: 순 +9** (phantom 4, other 7, easy −2) — Stage 4 출발점 |
| 3.4 | **친족 감사** | [stage3_4_nag_kinship.py](../../../scripts/stage3_4_nag_kinship.py) | wrong vs oracle 의 gaussian 집합 포함률 → child/parent/sibling/unrelated | granularity 4 확정 (jake≈same, tesla=1% 조각, sink=부모), sibling 2, unrelated 11 |
| 3.4 | **패자 해부** | [stage3_4_loser_anatomy.py](../../../scripts/stage3_4_loser_anatomy.py) | never_good+악화 5건의 두 체제 승자 top-10 신원분석 (렌더+GT IoU+친족) | **few-view opportunist 발견** (승자 nv 2-11, coh 0.92-0.99) + sake cup 의 rank-진단 맹점 (gt_iou 0.93 SP 가 mean rank 3) |
| 3.4 | **역행 해부** | [stage3_4_easy_regression.py](../../../scripts/stage3_4_easy_regression.py) | easy 역행 10건의 top5 신규 승자 신원 + 가드 신호 | **3/10 은 kin_same 가짜 역행**; g1 gap median +0.17, g2 60% — hybrid 가드 설계 데이터 |
| 3.4 | **multi-instance 감사** | [stage3_4_multi_instance.py](../../../scripts/stage3_4_multi_instance.py) | generic prompt 의 외부 승자 중 nv≥10 의심 수집 + montage | **R7 발동: 7 prompt 46건** (spoon 10, sake cup 10, cabinet 9, sink 8 …) |
| 3.4 | **taxonomy 종합** | [stage3_4_synthesize.py](../../../scripts/stage3_4_synthesize.py) | 36 케이스 rule-based 라벨 + R6/R7/R8 | **R6 PASS** (unknown 2: bowl, napkin) · **R8: 2B encoder-측 4/4 + Stage 1 D2.real 4/5 재분류 → encoder 잔여 3건** |
| 4 | **채점기 v1/v2** | [stage4_scorer.py](../../../scripts/stage4_scorer.py) / [_v2](../../../scripts/stage4_scorer_v2.py) | hybrid+Z/E/(g1→g2)/P grid (216→96) + 정합성 gate | gate ① 0/67 (±2) ② 0/67 (정확) — 양 극단 재현 |
| 4 | **LOSO+mask 평가** | [stage4_loso_eval.py](../../../scripts/stage4_loso_eval.py) | fold 별 calib-proxy shortlist → calib mIoU 선택 → held-out 평가 + ablation | **R9 PARTIAL (최종)**: v2 held-out **+3.74pt** (phantom +17.5 / other +13.3 / easy **−4.08** / multi-inst 제외 +4.53). v1 ablation 이 g1 결함 적발 (+8.7pt off 시); g2 margin 은 포화로 불활성 (41 easy 중 1 보호) |
| 5 | **ReLaGS replay** | [stage3_b8_replay.py](../../../scripts/stage3_b8_replay.py) `--agg relags --iteration 0` | ROFA-포함 충실도 재구성으로 ReLaGS 자체 파이프라인 dump (1.2GB) | gate 97.8~100%, 타깃 36/36; **유령 21 = "150→21 (86%)" 독립 재확인** |
| 5 | **G1 소수파 매장** | [stage5_g1_singleview.py](../../../scripts/stage5_g1_singleview.py) | ReLaGS pool 에서 단일 view rank + 좋은 view 비율 | **✅ 재현**: best 단일 ≤3 **95%** (THGS 88), **23% vs 51%** (THGS 18/50, p<1e-4) |
| 5 | **G2 제로섬** | [stage5_g2_fullpool.py](../../../scripts/stage5_g2_fullpool.py) | ReLaGS 전 pool query-top5 재채점 | **✅ 재현**: phantom **+8** / easy **−12** / other +3 (THGS +6/−10/+5) |
| 5 | **G3 승자 시그니처** | [stage5_g3_winners.py](../../../scripts/stage5_g3_winners.py) | 회복불능/역행 승자의 nv·coherence·GT-IoU·친족 | **✅**: phantom 승자 GT-무관 82% (coh 0.89, few-view 33%); 역행 승자 GT-무관 90% |
| 5 | **G4 ROFA 실측** | [stage5_g4_rofa.py](../../../scripts/stage5_g4_rofa.py) | 같은 dump 에서 ROFA on/off 재구성 rank 비교 | **phantom median 효과 0, 구출 1/20** — 2B 시뮬 ("outlier_handled 0%") 실제 확정 |

---

## 🧪 Stage 1 — 4 단계 진단

### B7 — Oracle SP existence (purity / completeness / fragmentation)

**Code**: [scripts/b7_a4_oracle_analysis.py](../../../scripts/b7_a4_oracle_analysis.py)
**ReLaGS copy**: [ReLaGS/scripts/b7_a4_oracle_analysis.py](../../../ReLaGS/scripts/b7_a4_oracle_analysis.py) — line 50 까지 byte-for-byte identical, 출력은 각 method 의 `sai_nag.pt` 에 의해 달라짐

**Input**:
- `sai_nag.pt` (= NAG hierarchical superpoint graph + per-level CLIP features)
- per-frame GT polygon JSON

**무엇을 함**: 각 prompt 에 대해 모든 SP mask 를 ref_frame 에 render → GT polygon 과 비교해 가장 깨끗한 SP (= oracle) 선택 → 그 oracle 의 geometric 품질 측정.

**핵심 formula** ([b7_a4_oracle_analysis.py:72-107](../../../scripts/b7_a4_oracle_analysis.py#L72-L107)):
```
purity        = |SP ∩ GT| / |SP|        # SP 픽셀 중 GT 비율
completeness  = |SP ∩ GT| / |GT|        # GT 픽셀 중 SP 비율
IoU           = |SP ∩ GT| / |SP ∪ GT|
fragmentation = GT 를 budget=3 까지 greedy-union 으로 덮는데 필요한 SP 수
```

**핵심 threshold**:
- `render_thresh = 0.5` (SP mask binarize)
- `levels = [2, 3]` (NAG 의 coarser hierarchy 만)
- `chunk = 200`, `topk_keep = 30`, `budget = 3`

**Output**: [output/diagnostics/b7_a4_combined.csv](../../../output/diagnostics/b7_a4_combined.csv) (THGS, 208 rows = (prompt, eval_frame))

**결과**:
| Metric | 값 |
|---|---|
| purity 평균 | 0.925 |
| purity < 0.5 | 2/67 (3%) |
| completeness 평균 | 0.857 |
| completeness < 0.5 | 4/67 (6%) |
| fragmentation = 1 | 45/67 (67%) |
| fragmentation ≥ 3 | 11/67 (16%) |

→ **분할은 dominant 문제 아님**. 65/67 가 purity ≥ 0.5.

---

### A4 — Oracle CLIP rank + margin (3 정의)

**Code**: 같은 [b7_a4_oracle_analysis.py](../../../scripts/b7_a4_oracle_analysis.py)
**Input**: `sai_nag.pt` 의 `nag_feat` (per-level CLIP features) + B7 의 oracle SP id

**무엇을 함**: scene 의 모든 SP CLIP feature vs prompt text embedding 의 cosine 계산 → oracle SP 가 몇 등인지 + top 과의 격차 3 가지 정의로 측정.

**핵심 formula** ([b7_a4_oracle_analysis.py:305-317](../../../scripts/b7_a4_oracle_analysis.py#L305-L317)):
```
oracle_rank      = rank of oracle SP by CLIP score (1-indexed)
raw_margin       = cos_top − cos_oracle
z_margin         = raw_margin / pool_std            # cross-prompt 비교용 (default)
percentile_margin = (rank − 1) / (pool_size − 1) × 100
pool_entropy     = softmax_entropy(pool_scores)
```

CLIP score (canon-contrast, [utils/vlm_utils.py:34-44](../../../utils/vlm_utils.py#L34-L44)):
```
score = min over canon ∈ {"object","things","stuff","texture"} of
        softmax(10 × [SP_feat @ prompt_feat, SP_feat @ canon_feat])[0]
```

**Output**: 같은 b7_a4_combined.csv 의 추가 column (`oracle_rank`, `raw_margin`, `z_margin`, `percentile_margin`, `cos_top`, `cos_oracle`, `pool_std`)

**결과**:
| Metric | 값 |
|---|---|
| oracle rank 중앙값 | 2 |
| q75 / q90 / max | 7 / 52 / 256 |
| rank ≤ 3 | 41/67 (61%) |
| rank ≤ 10 | 53/67 (79%) |
| **rank > 30 (catastrophic)** | **13/67 (19%)** |
| z_margin 중앙값 | 0.29 |
| \|z_margin\| > 1.5 | 10/67 |

→ 양극화 — 절반 (rank ≤ 2) 은 쉽고 19% 는 거의 random.

---

### A2 — Image-CLIP encoder ceiling (4 crop policy)

**Code**: [scripts/a2_image_clip_ceiling.py](../../../scripts/a2_image_clip_ceiling.py)
**Input**: per-frame GT polygon JSON + RGB 이미지

**무엇을 함**: SP/multi-view 우회. GT polygon 영역만 직접 잘라서 CLIP image encoder 에 넣고 67 prompt 중 정답이 몇 등으로 인식되는지 측정 = **CLIP 자체의 한계**.

**핵심 formula — 4 crop policy** ([a2_image_clip_ceiling.py:112-122](../../../scripts/a2_image_clip_ceiling.py#L112-L122)):
| Policy | 정의 |
|---|---|
| **tight** | GT polygon 의 minimum bbox, 원본 RGB |
| **mask** | 같은 bbox, polygon 밖 픽셀 = 검정 (SAM-mask proxy) |
| **context** | 1.5× 확장한 bbox, 원본 RGB (주변 맥락 포함) |
| **method** | polygon-blackout + 1.2× bbox (image_encoding.py 의 crop policy 근사) |

각 crop → CLIP image encoder → 67 prompt 별 canon-contrast score → 정답 prompt 의 rank.

**Output**: [a2_image_clip_ceiling.csv](../../../output/diagnostics/a2_image_clip_ceiling.csv) (268 rows = 67 prompt × 4 policy)

**결과**:
| Policy | Top-1 | Top-3 |
|---|---|---|
| tight | 43/67 = 64.2% | 86.6% |
| mask | 42/67 = 62.7% | 85.1% |
| **context (1.5×)** | **48/67 = 71.6%** | **89.6%** |
| method | 46/67 = 68.7% | 85.1% |

→ context +7.4% (맥락이 결정적). mask vs tight 차이 거의 없음 (배경 검정 자체는 ceiling 안 깎음).

---

### Joint — D2.phantom vs D2.real 분해 (2×2)

**Code**: [scripts/b7_a4_a2_plots.py](../../../scripts/b7_a4_a2_plots.py) (분류 logic), [cross_method_comparison.py:26-33](../../../scripts/cross_method_comparison.py#L26-L33) (canonical 정의)

**Input**: b7_a4_combined.csv + a2_image_clip_ceiling.csv 의 ref_frame rows merge on (scene, prompt)

**무엇을 함**: A4 (SP rank) 와 A2 (image-CLIP rank, mask policy) 를 threshold=3 으로 2×2 cross.

**핵심 logic**:
```
                  A2_mask_rank ≤ 3   > 3
   SP_rank ≤ 3    Easy             Rare
   SP_rank > 3   Phantom          Real
```

**결과** (THGS):
| Class | n | % | 의미 |
|---|---|---|---|
| Easy | 36 | 53.7% | CLIP & SP 둘 다 쉬움 |
| **D2.phantom** | **21** | **31.3%** | CLIP 알아봄, SP 가 죽임 (회복 가능) |
| D2.real | 5 | 7.5% | CLIP 도 못 알아봄 (encoder 한계) |
| Rare | 5 | 7.5% | 평균이 단일 view 보다 나음 |

**Phantom : Real = 21 : 5 = 4.2 : 1** — 실패 26 중 21 (81%) 회복 가능.

**✱ 검증 (2026-06-10)**: A2 쪽을 `canon_rank_true` 로 교체해도 **67 prompt 분류 변화 0 건** (THGS 21 phantom / ReLaGS 20 / persistent 17 동일, [verify_joint_canon_vs_raw.csv](../../../output/diagnostics/verify_joint_canon_vs_raw.csv)) — 이 분해는 A2 metric (raw vs canon) 선택에 완전 robust.

Per-scene:
| Scene | n | Easy | Phantom | Real | Rare |
|---|---|---|---|---|---|
| figurines | 21 | 11 | 6 | 3 | 1 |
| **ramen** | 14 | 7 | 6 | **0** | 1 |
| teatime | 14 | 11 | 2 | 1 | 0 |
| waldo_kitchen | 18 | 7 | 7 | 1 | 3 |

→ **ramen 0 real** = 100% 회복 가능 영역.

---

### Cross-method — THGS ↔ ReLaGS 4×4 transition

**Code**: [scripts/cross_method_comparison.py](../../../scripts/cross_method_comparison.py)
**Input**: b7_a4_combined.csv (THGS) + b7_a4_combined_relags.csv (ReLaGS) + a2_image_clip_ceiling.csv

**무엇을 함**: 같은 67 prompt 가 두 method 에서 어떻게 class 변하는지 4×4 transition counts.

**Output**: [cross_method_d2_decomposition.csv](../../../output/diagnostics/cross_method_d2_decomposition.csv) (67 rows × 두 method 별 class)

**결과** (분포):
| Class | THGS | ReLaGS | Δ |
|---|---|---|---|
| Easy | 36 (53.7%) | 37 (55.2%) | +1 |
| **Phantom** | **21 (31.3%)** | **20 (29.9%)** | −1 |
| Real | 5 (7.5%) | 4 (6.0%) | −1 |
| Rare | 5 (7.5%) | 6 (9.0%) | +1 |

Transition (THGS phantom 21 → ReLaGS):
- 4 easy (19% 회복), 17 phantom (81% 잔존), 0 real (regression 없음)

→ **17 persistent phantoms** (두 method 공통 실패) = paper 의 새 method 의 target set.

---

## 🔬 Stage 2A — 4-layer forensics on 17 persistent phantoms

### Layer 1 — B8 within-view mixing (visual proxy)

**Code**: [scripts/stage2a_layer1_b8_coverage.py](../../../scripts/stage2a_layer1_b8_coverage.py)
**Input**: sai_nag.pt + persistent_phantoms_17.csv

**무엇을 함**: 각 phantom 의 oracle SP 를 20 sampled train views 에 render → 각 view 에서 SP mask 의 disconnected components 분석.

**핵심 formula** ([stage2a_layer1_b8_coverage.py:48-72](../../../scripts/stage2a_layer1_b8_coverage.py#L48-L72)):
```
per (SP, view):
    num_components       = ≥ 10% area 의 connected comp 수
    largest_component_frac = max comp area / total SP pixels
    bbox_fill_ratio      = total pixels / bbox area
    aspect_ratio         = max(h, w) / min(h, w)

per SP (aggregate over views):
    mix_view_frac = #{view : num_components ≥ 2} / #views
```

**핵심 threshold**: `view_subsample = 20`, `min_area_frac = 0.10`, `easy_per_scene = 4` (control sample)

**Output**: [stage2a_layer1.csv](../../../output/diagnostics/stage2a_layer1.csv) (33 rows = 17 phantom + 16 easy sample)

**결과**:
| 통계 | Phantom17 | Easy_sample |
|---|---|---|
| mix_view_frac 평균 | **0.16** | **0.16** |
| mean_num_components | 1.18 | 1.17 |
| mean_bbox_fill | 0.52 | 0.48 |

Phantom 의 mix_view_frac 분포: 11/17 (65%) < 0.10, 4/17 (24%) > 0.30, 2/17 (12%) 중간.

→ **within-view 공간 fragmentation 은 phantom 의 dominant cause 아님** (phantom = easy 분포).

---

### Layer 2 — Wrong top-1 forensics + per-view rank trajectory

**Code**: [scripts/stage2a_layer2_wrongtop1.py](../../../scripts/stage2a_layer2_wrongtop1.py)
**Input**: sai_nag.pt + persistent_phantoms_17.csv + GT polygons

**Part A — Wrong top-1 forensics** ([stage2a_layer2_wrongtop1.py:193-237](../../../scripts/stage2a_layer2_wrongtop1.py#L193-L237))

**무엇을 함**: CLIP top-1 SP (method 가 실제 잘못 선택한 것) 을 ref_frame 에 render → 정답 GT 와의 overlap + scene 의 *다른* 모든 GT prompt 와의 IoU 매트릭스 → 가장 잘 매칭되는 prompt 찾음.

**분류 logic**:
| Type | 조건 |
|---|---|
| over_union | wrong_top1_overlap_with_gt ≥ 0.30 |
| instance_confusion | wrong_top1_best_match_iou (with OTHER prompt) ≥ 0.30 |
| background_drift | 둘 다 < 0.10 |

**Part B — Per-view rank trajectory** ([:242-296](../../../scripts/stage2a_layer2_wrongtop1.py#L242-L296))

**무엇을 함**: 각 eval frame 마다 그 frame 의 GT 로 oracle SP 재정의 → 같은 (ref_frame 의) CLIP pool 안의 그 oracle 의 rank → min/max/mean view rank, `same_as_ref_oracle` flag.

**분류**:
- min ≤ 3 AND max > 30 → **target dilution** (일부 view 정답)
- 모든 view rank > 3 → **structural** (mean-dilution signature)

**Output**:
- [stage2a_layer2_forensic.csv](../../../output/diagnostics/stage2a_layer2_forensic.csv) (17 rows)
- [stage2a_layer2_trajectory.csv](../../../output/diagnostics/stage2a_layer2_trajectory.csv) (51 rows per (phantom, eval_frame))

**결과**:
| Wrong top-1 type | n |
|---|---|
| background drift | 11 (65%) |
| over union | 3 (18%) |
| instance confusion | 3 (18%) |

| Trajectory | n |
|---|---|
| structural (all view rank > 3) | 13 (76%) |
| target dilution (min ≤ 3, max > 10) | 4 (24%) — old camera, bowl, napkin, sink |

→ Background drift 65% 가 E1 direction bias 의 evidence. 76% structural 이 mean-dilution signature.

---

### Layer 3 — D3 top-k sweep (17 phantom-only)

**Code**: [scripts/stage2a_layer3_d3_topk.py](../../../scripts/stage2a_layer3_d3_topk.py)
**Input**: sai_nag.pt + persistent_phantoms_17.csv + GT polygons

**무엇을 함**: 각 phantom 의 ref_frame 에서 k ∈ {1, 2, 3, 5, 10} 의 top-k SP union 을 render → 모든 eval frame 에 대해 mIoU 계산.

**핵심 formula** ([stage2a_layer3_d3_topk.py:199-207](../../../scripts/stage2a_layer3_d3_topk.py#L199-L207)):
```python
K_SWEEP = [1, 2, 3, 5, 10]
for k in K_SWEEP:
    pairs = top-k (level, sp_id) by CLIP score
    mask = render_union(pairs)
    iou = intersection / union vs GT
mean over eval_frames
```

**Output**:
- [stage2a_layer3_d3.csv](../../../output/diagnostics/stage2a_layer3_d3.csv) (51 rows raw)
- [stage2a_layer3_d3_summary.csv](../../../output/diagnostics/stage2a_layer3_d3_summary.csv) (17 per-phantom)

**결과**:
| Optimal k | n (per-phantom) |
|---|---|
| k=1 | 7 (41%) |
| k=2 | 2 (12%) |
| k=3 (default) | 4 (24%) |
| k=5 | 1 (6%) |
| k=10 | 3 (18%) |

**Recovery**:
- IoU > 0.5 가능: **5/17 (29%)**
- IoU > 0.3 가능: 8/17 (47%)
- IoU < 0.1 (어떤 k 도 회복 못 함): **8/17 (47%)** — structural

Mean IoU: default 0.203 → per-prompt optimal **0.314** (+11.2pt).

---

### Layer 4 — Visual montage

**Code**:
- [scripts/stage2a_layer4_montage.py](../../../scripts/stage2a_layer4_montage.py) — per-phantom 4-panel render
- [scripts/stage2a_layer4_combine_montage.py](../../../scripts/stage2a_layer4_combine_montage.py) — 17 row grid 결합

**무엇을 함** ([stage2a_layer4_montage.py:136-210](../../../scripts/stage2a_layer4_montage.py#L136-L210)):
각 phantom 의 ref_frame 에서 4 colored overlay:
1. GT (green, α=0.5)
2. Oracle SP (cyan)
3. CLIP top-1 SP (red)
4. Top-10 union (yellow)

Combined mask bbox 1.6× crop, panel height = 180px, labels = `scene:prompt`.

**Output**: `output/diagnostics/plots/phantom_montage.png` (3164 × 1612 px)

---

### Synthesis — Mechanism attribution

**Code**: [scripts/stage2a_synthesize.py](../../../scripts/stage2a_synthesize.py)
**Input**: Layer 1-3 + B7+A4+A2 join on (scene, prompt)

**무엇을 함**: 17 phantom 각각의 primary mechanism 을 rule-based 로 분류.

**핵심 attribution rules** ([stage2a_synthesize.py:124-158](../../../scripts/stage2a_synthesize.py#L124-L158)):
| Tag | 조건 |
|---|---|
| over_union | wrong_top1_overlap_with_gt ≥ 0.30 OR (best_k=1 with iou>0.3 and k3_iou < best_iou − 0.10) |
| instance_confusion | wrong_top1_best_match_iou ≥ 0.30 |
| background_drift | both overlaps < 0.10 |
| target_dilution | min_rank ≤ 3 AND max_rank > 10 |
| d3_recoverable | best_iou ≥ 0.30 AND iou_k3 + 0.10 ≤ best_iou |
| geometry_fragmented | mix_view_frac > 0.40 |
| encoder_hidden | a2_mask_rank > 1 |
| structural_ROFA | best_iou < 0.10 |

**Output**:
- [phantom_anatomy.csv](../../../output/diagnostics/phantom_anatomy.csv) (17 × 26 cols)
- [phantom_attribution.csv](../../../output/diagnostics/phantom_attribution.csv) (17 × primary_mechanism + all_tags)

**결과**:
| Primary mechanism | n | % |
|---|---|---|
| structural_ROFA (mean-dilution) | 6 | 35% |
| D3_deep_pool | 3 | 18% |
| instance_confusion | 3 | 18% |
| target_dilution | 2 | 12% |
| D3_over_union | 1 | 6% |
| background_drift | 1 | 6% |
| unknown / marginal | 1 | 6% |

카테고리:
- 🌊 **Across-view ROFA 류**: **12/17 (71%)** ← 주범
- 🧭 Direction bias (instance + background drift): 4 (24%)
- Marginal: 1 (5%)

---

## 🧬 Stage 2B — ROFA anatomy + alternative recovery probes

### F2.A — H2 lite per-view CLIP feature

**Code**: [scripts/stage2b_h2lite_perview.py](../../../scripts/stage2b_h2lite_perview.py)
**Input**: sai_nag.pt + persistent_phantoms_17.csv + RGB training images

**무엇을 함**: 각 17 phantom 의 oracle SP mask 를 30 sampled train views 에 render → 마스크 영역만 crop (배경 검정) → CLIP image encoder → text prompt 와 cosine. *진짜 H2 instrument 의 lite 버전* (per-SAM-mask CLIP 대신 oracle-SP-mask-only encoding).

**핵심 formula** ([stage2b_h2lite_perview.py:100-141](../../../scripts/stage2b_h2lite_perview.py#L100-L141)):
```python
mask = render_sp_mask(view, oracle_sp, thresh=0.5)
img2[~mask] = 0                                # blackout background
crop = bbox_crop_to_sp_region(img2)
img_feat = clip.encode_image(preprocess(PIL(crop)))
img_feat = img_feat / img_feat.norm()
cos_with_prompt = np.dot(img_feat, text_feat)  # both L2-normalized
```

**핵심 constant**:
- CLIP: **ViT-B-16, laion2b_s34b_b88k** ([:75](../../../scripts/stage2b_h2lite_perview.py#L75))
- view_subsample = 30, min_pixels = 50, render thresh = 0.5

**Output**: [_h2_lite_perview.pkl](../../../output/diagnostics/_h2_lite_perview.pkl) — `{(scene, prompt): [{view_idx, image_name, visibility_pixels, feature[512], cos_with_prompt}]}`

**결과** (per-phantom cos_mean):
| 강한 신호 (cos ≥ 0.20) | 약한 신호 (cos < 0.20) |
|---|---|
| **12/17 (71%)** | 5/17 (29%) |

대표 값: rubber duck with hat 0.295, bowl 0.235, plate 0.234, ottolenghi 0.153 (가장 낮음).

**✱ 정정 (2026-06-10 검증)**: threshold (cos≥0.20) 기준은 **12/17** (onion segments 0.201 포함) — 기존 문서들의 "11/17" 은 *subtype 분류 후 strong_signal_phantom 개수* (sake cup 이 bimodal 로 빠짐) 와 혼용된 표기. **rank 기반 재검증** ([verify_h2lite_rank.csv](../../../output/diagnostics/verify_h2lite_rank.csv) — clean crop feature 를 scene 전체 prompt 와 raw cosine 랭킹): **median rank ≤ 3 = 11/17 (65%)**. onion segments 는 median rank 6.0 (top-3 view 17%) 으로 약한 신호 → encoder 측 재분류 권장; sake cup 은 median 3.0 으로 강함.

→ **재료는 살아있음** — clean encoding 에선 강한데 pipeline 에선 phantom. (rank 기준에서도 65% 유지.)

---

### F2.B — ROFA subtype classification

**Code**: [scripts/stage2b_f2_subtypes.py](../../../scripts/stage2b_f2_subtypes.py)
**Input**: `_h2_lite_perview.pkl` (F2.A 출력)

**무엇을 함**: per-view CLIP feature 위에서 ROFA outlier filter simulate + cos 분포 분석 → 4 subtype 분류.

**핵심 formula — ROFA simulation** ([stage2b_f2_subtypes.py:31-41](../../../scripts/stage2b_f2_subtypes.py#L31-L41)):
```python
TAU = 1.0
cos_sim = features @ features.T              # N × N
mean_sim = (cos_sim.sum(axis=1) - 1) / (N - 1)
keep_mask = mean_sim > (μ − τ · σ)
```

**✱ 정정 (2026-06-10)**: script 의 `TAU = 1.0  # ROFA default` 주석과 달리 **실제 ReLaGS pipeline default 는 τ=2** ([ReLaGS/merge_proj.py:124](../../../ReLaGS/merge_proj.py#L124), argparse default 2). τ=2.0 재실행 시 **subtype 분포는 동일 (11/4/2/0)** — F2.B 결론은 τ-robust. 단 F2.C 의 keep/drop 수치는 τ 에 민감 (아래 F2.C 참조).

**✱ 주의 (code-mirror)**: 가설 문서 (6.1 patch) 가 명세한 **feature-space GMM BIC 검정은 미구현** — 분류는 cos-with-prompt 스칼라 분포 휴리스틱 (IQR/std threshold) 이며, script docstring 의 GMM 언급은 부정확 (sklearn `GaussianMixture` 는 import 만 되고 미사용). 페이퍼에 "GMM 기반" 으로 기술하면 코드와 불일치.

**핵심 formula — Subtype 분류** ([:44-82](../../../scripts/stage2b_f2_subtypes.py#L44-L82)):
| Subtype | 조건 |
|---|---|
| insufficient_evidence | N < 5 |
| **bimodal_balanced** | cos_iqr > 0.06 AND cos_std > 0.04 AND kept_std > 0.03 |
| **outlier_handled** | bimodal 조건 + (dropped_cos_mean < kept_cos_mean − 0.04) |
| **mean_dilution** | cos_mean < 0.20 |
| **strong_signal_phantom** | (위 어디에도 안 들어가면) |

**Output**: [stage2b_rofa_subtypes.csv](../../../output/diagnostics/stage2b_rofa_subtypes.csv) (17 rows × n_views, cos_mean/min/max/std/q25/q75, mean_sim_mu/sigma, n_dropped_by_rofa, subtype, notes)

**결과**:
| Subtype | n | % |
|---|---|---|
| **strong_signal_phantom** | **11** | **65%** |
| mean_dilution | 4 | 24% (old camera, pumpkin, napkin, ottolenghi) |
| bimodal_balanced | 2 | 12% (pikachu, sake cup) |
| **outlier_handled** | **0** | **0%** |

→ **0 outlier_handled** = ROFA 가 의도대로 (outlier 처리) 동작한 case 없음.

---

### F2.C — ROFA keep-mask analysis

**Code**: 같은 [stage2b_f2_subtypes.py](../../../scripts/stage2b_f2_subtypes.py) (별도 output)

**무엇을 함**: 각 phantom 에서 ROFA keep_mask 가 어떤 view 를 drop 했는지 → kept view 들의 cos_with_prompt 평균 vs dropped 들의 평균 비교.

**Output**: [stage2b_rofa_keep_mask.csv](../../../output/diagnostics/stage2b_rofa_keep_mask.csv) (scene, prompt, n_views, n_dropped, kept_cos_mean, dropped_cos_mean, **cos_delta**, **rofa_pathology**)

**rofa_pathology**:
- `ROFA_kept_good_views`: kept_cos > dropped_cos (정직)
- `ROFA_dropped_good_views`: dropped_cos > kept_cos (실수)

**결과** (simulation τ=1.0):
| Pathology | n | % |
|---|---|---|
| ROFA_kept_good_views | 14 | 82% |
| ROFA_dropped_good_views | 3 | 18% — **old camera, ottolenghi, sink** |

**✱ 정정 (2026-06-10) — τ=2.0 (실제 pipeline default) 재실행** ([verify_f2_tau_sweep.csv](../../../output/diagnostics/verify_f2_tau_sweep.csv)):
| Pathology (τ=2.0) | n | % |
|---|---|---|
| ROFA_kept_good_views | 11 | 65% |
| no_drops | 2 | 12% — sink, cabinet |
| ROFA_dropped_good_views | **4** | **24% — old camera, ottolenghi, pikachu, onion segments** |

τ=2 에서 sink 는 drop 0 건이 되어 "실수" 명단에서 빠지고, **pikachu (drop 5→1)·onion segments (drop 5→2) 가 새로 등장** — 실수 4 건 중 2 건이 instance-confusion phantom 이라는 신규 단서.

→ ROFA 는 대체로 무죄 (실수 4/17, outlier_handled 0% 유지). 받기 전에 신호가 죽어있었다는 결론 유지.

---

### D3 — Prompt-agnostic top-k sweep (full 67 prompts)

**Code**: [scripts/stage2b_d3_prompt_agnostic_sweep.py](../../../scripts/stage2b_d3_prompt_agnostic_sweep.py)
**Input**: sai_nag.pt + GT polygons (label/scene/frame.json)

**무엇을 함**: 67 prompt 전체에 대해 k ∈ {1, 2, 3, 5, 10, 20} 의 top-k union mIoU 측정 → method-level fixed-k 의 mean mIoU.

**핵심 formula** ([stage2b_d3_prompt_agnostic_sweep.py:144-161](../../../scripts/stage2b_d3_prompt_agnostic_sweep.py#L144-L161)):
```python
K_SWEEP = [1, 2, 3, 5, 10, 20]
for prompt, eval_frame:
    pool_scores = vlm.compute_similarity(snag.feat[levels=[2,3]])
    order = argsort(-pool_scores)
    for k in K_SWEEP:
        pairs = [(cache_lvl[i], cache_sp[i]) for i in order[:k]]
        mask = render_union(pairs)
        iou = intersection / max(union, 1)
```

**핵심 threshold**: `thresh = 0.5`, `levels = [2, 3]`

**Output**:
- [stage2b_d3_full_sweep.csv](../../../output/diagnostics/stage2b_d3_full_sweep.csv) (per (prompt, eval_frame))
- [stage2b_d3_full_sweep_agg.csv](../../../output/diagnostics/stage2b_d3_full_sweep_agg.csv) (per-prompt aggregate)

**결과** — Method-level mean mIoU:
| k | 전체 67 prompt | 17 phantom only | > 0.5 in 17 |
|---|---|---|---|
| 1 | 0.464 | 0.155 | 2/17 |
| 2 | 0.526 | 0.166 | 2/17 |
| **3 (default)** | **0.542** | 0.203 | 3/17 |
| 5 | 0.469 | 0.185 | 3/17 |
| **10** | 0.483 | **0.290** | **5/17** |
| 20 | 0.448 | 0.281 | 4/17 |

Per-scene optimal k: figurines 2, ramen/teatime/waldo_kitchen 3.

→ **Trade-off**: 전체 = k=3 best, phantom 만 = k=10 best (+8.7pt). Free lunch 없음. Prompt-conditional adaptive 필요.

---

### C3 — Negative prompt contrast (3 instance confusion)

**Code**: [scripts/stage2b_c3_negative_contrast.py](../../../scripts/stage2b_c3_negative_contrast.py)
**Input**: sai_nag.pt + GT polygons

**무엇을 함**: 3 instance confusion phantom 에 대해 score 에 negative-prompt contrast term 추가 → λ sweep 으로 oracle rank 변화.

**핵심 formula** ([stage2b_c3_negative_contrast.py:172](../../../scripts/stage2b_c3_negative_contrast.py#L172)):
```python
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
negatives = [p for p in scene_prompts if p != target]  # scene 의 다른 prompts
target_pool[i] = cos(SP_i, target_text)
max_neg[i]     = max(cos(SP_i, neg_text) for neg in negatives)
adjusted_score = target_pool − λ · max_neg
oracle_rank    = rank of oracle SP under adjusted_score
```

**3 tested prompts**: figurines/pikachu, figurines/rubber duck with hat, ramen/onion segments

**Output**: [stage2b_c3_quickcheck.csv](../../../output/diagnostics/stage2b_c3_quickcheck.csv) (3 prompts × 6 lambdas)

**결과**:
| Phantom | λ=0 rank | λ=1.0 rank | Δ |
|---|---|---|---|
| pikachu | 64 | **50** | −14 (22% 개선이지만 여전히 catastrophic) |
| rubber duck with hat | 5 | 6 | +1 (악화) |
| onion segments | 14 | 13 | −1 (marginal) |

→ **Text-side 로는 SP feature 의 wrong direction 못 되돌림**. Instance confusion 의 fix path = aggregation 단계 수정뿐.

---

## 🔁 검증 (2026-06-10) — Stage 1/2B robustness checks 3종

**Code**: [scripts/verify_stage2b_checks.py](../../../scripts/verify_stage2b_checks.py)
**Input**: `_h2_lite_perview.pkl` + `a2_image_clip_ceiling.csv` + `b7_a4_combined(.relags).csv` — 전부 기존 산출물 재사용, 신규 렌더링 없음

코드 재검토에서 나온 3 가지 의심 (τ 불일치 / metric 비대칭 / 절대-threshold 휴리스틱) 을 실험으로 확인.

### Check 1 — F2 τ sweep (simulation τ=1.0 vs pipeline default τ=2.0)

| | τ=1.0 (기존 문서) | τ=2.0 (pipeline) |
|---|---|---|
| subtype 분포 | strong 11 / mean_dil 4 / bimodal 2 / outlier 0 | **동일 (11/4/2/0)** |
| ROFA_kept_good_views | 14 (82%) | 11 (65%) |
| no_drops | 0 | 2 (sink, cabinet) |
| ROFA_dropped_good_views | 3 (old camera, ottolenghi, sink) | **4 (old camera, ottolenghi, pikachu, onion segments)** |

→ **65% strong_signal 은 τ-robust**. F2.C 수치/명단은 τ=2.0 기준으로 정정. 실수 4 건 중 2 건 = instance confusion (신규 단서). **Output**: [verify_f2_tau_sweep.csv](../../../output/diagnostics/verify_f2_tau_sweep.csv)

### Check 2 — Joint 2×2 의 A2 metric robustness (raw vs canon)

`raw_rank_true` → `canon_rank_true` 교체 시 **67 prompt 중 분류 변화 0 건** (THGS easy 36/phantom 21/real 5/rare 5, ReLaGS 37/20/4/6, persistent 17 모두 동일). → Stage 1 headline 은 metric 선택에 완전 robust. **Output**: [verify_joint_canon_vs_raw.csv](../../../output/diagnostics/verify_joint_canon_vs_raw.csv)

### Check 3 — H2-lite strong-signal 의 rank 재검증

절대 threshold (cos≥0.20) 대신 clean crop feature 를 scene 전체 prompt 와 raw cosine 랭킹:
- cos ≥ 0.20: **12/17** (기존 "11/17" 표기는 subtype 개수와 혼용 — threshold 기준은 onion segments 0.201 포함 12)
- **median rank ≤ 3: 11/17 (65%)** — headline 유지
- 멤버십 교정: **onion segments** 약함 (median rank 6.0, top-3 view 17%) → encoder 측 재분류; **sake cup** 은 강함 (median 3.0)

**Output**: [verify_h2lite_rank.csv](../../../output/diagnostics/verify_h2lite_rank.csv)

### 검증이 바꾼 것 / 안 바꾼 것

| 항목 | 판정 |
|---|---|
| Stage 1: 21 phantom, 17 persistent, 4.2:1 | ✅ 완전 robust (raw/canon 무관) |
| F2.B "65% strong_signal" | ✅ 유지 (τ·rank 기준 모두) — 멤버는 onion segments ↔ sake cup 교정 |
| F2.C "82% 정직, 실수 3건 (old camera·ottolenghi·sink)" | ⚠️ **정정**: τ=2.0 기준 65% + 12% no-drop, 실수 4건 (old camera·ottolenghi·pikachu·onion segments) |
| "0% outlier_handled" | ✅ τ=2.0 에서도 유지 |
| 신규 단서 | τ=2 의 ROFA 실수 4건 중 2건이 instance-confusion phantom — Stage 3 분석 후보 |

---

## 🧪 Stage 3.1 — B8 인과 replay

### B8 replay — within-view 단계 verbatim 재현

**Code**: [scripts/stage3_b8_replay.py](../../../scripts/stage3_b8_replay.py)
**Input**: `data/lerf_ovs/<scene>/language_features/` (재생성, ~9GB) + sai_nag.pt + persistent_phantoms_17.csv + b7_a4_combined.csv

**무엇을 함**: 17 phantom oracle SP + 16 easy control (RandomState(42), stage2a 와 동일) 에 대해 THGS production 경로 (`proj_gaussian_features_x`, `feat_assign=2`, WEIGHT_THRESHOLD=**0.0001**, RATIO_THRESHOLD=0.3) 를 전체 train view 에서 그대로 재현. per (SP, view): **mix_count** = `(ratio ≥ 0.3).sum()` (B8 최초 직접 측정), **mixed_feat** = `normalize(threshold(ratio) @ view_level_feature)`, **hard_feat** = `view_level_feature[argmax(ratio)]`.

**핵심 gate**: replay feature 를 pipeline aggregation 으로 재구성 → sai_nag.pt 실제 feature 와 비교: **33/33 PASS, cos 0.9935–0.9995**.

**Output**: [stage3_b8_replay_perview.pkl](../../../output/diagnostics/stage3_b8_replay_perview.pkl) (33 SP × 전체 view, feature fp16 포함 — **B1 해부의 foundation**)

### B8 판정 — 사전 등록 H-B8a/b/c

**Code**: [scripts/stage3_b8_analyze.py](../../../scripts/stage3_b8_analyze.py)
**Input**: replay pkl + `_h2_lite_perview.pkl` (phantom clean) + `_h2_lite_easy.pkl` (easy clean, 신규) — image_name 으로 paired join

**결과**:
| 판정 | 결과 |
|---|---|
| H-B8a (gap ≥ 0.05 & easy 대비 유의) | ❌ strong 11 mean gap **0.0038** (easy −0.0181; 차이는 유의 p=0.020 이나 크기 1/13) |
| H-B8b (mix-rate↔gap r > 0.4) | ❌ pearson 0.264 / spearman 0.357; phantom mix_rate 0.134 vs easy 0.081 |
| H-B8c | **분기 ③: gap 자체가 작음 → within-view 무죄, B1 재조준**. hard ≈ mixed (회복할 gap 없음) → Stage 3.2 hard-assignment fix 실행 전 기각 |

**결정적 관측**: per-view mixed median rank (text-side) 1–7 (strong 11 전부; 유일한 예외 ottolenghi 14 = encoder limit) vs 최종 A4 rank 4–256 (tesla 3→256, pikachu 5→64, old camera 4→130). strong 5 개 (bear nose, spoon, bowl, sink, cabinet) 는 **gap 음수** — mixing 이 신호를 개선.

**Output**: [stage3_b8_mix_stats.csv](../../../output/diagnostics/stage3_b8_mix_stats.csv), [stage3_b8_gap_summary.csv](../../../output/diagnostics/stage3_b8_gap_summary.csv)

상세: [stage3_b8_causal.md](stage3_b8_causal.md)

---

## 🧪 Stage 3.2 — B1 anatomy

**Code**: [stage3_2_common.py](../../../scripts/stage3_2_common.py) (pool-rank 인프라) + B1.A/B/C 스크립트 3종
**Input**: stage3_b8_replay_perview.pkl 재사용 (신규 GPU = wrong-top1 replay ~15분뿐)

**Pool rank 정의**: sai_nag levels [2,3] 전체 SP pool 에서 해당 SP entry 를 후보 feature 로 *교체* 후 ClipSimMeasure canon-contrast rank (A4 와 동일 universe/기준). **충실도 gate**: 재구성 baseline rank vs A4 oracle_rank — 26/33 strict (±2), 초과 7건은 deep-rank ±5-10% 상대오차, 질적 이탈은 jake (1↔6) 1건뿐 (회복 카운트에 병기).

**B1.A 결과**: phantom 의 best 단일 view pool rank ≤3 = 15/17 (대부분 rank 1), 그러나 좋은 view 비율 18% (easy 50%), 단일 view rank 중앙값 51 (easy 2). 곡선: sudden_drop 9 / gradual 3 / no_degradation 3 / **never_good 2 (spoon, cabinet)**. → **R1 ✅**

**B1.B 결과** (회복 = rank≤3, regression = easy baseline≤3 → >3):
| variant | 회복/17 | reg/15 |
|---|---|---|
| uniform | 1 | 8 |
| visweight (baseline) | 1 (jake=gate노이즈) | 0 |
| top-k portion (k=1/3) | 6 / 6 (jake 제외 5) | 0 |
| **top-k query-cos (k=5)** | **15** | **0** |
| GMM mode-cluster | 3 | 6 |
| ROFA τ=2 | 1 | 0 |
→ **R2 ✅, Stage 4 = query-aware**. visibility weighting 은 필수 성분 (uniform 개악), 6.1 의 GMM 제안은 실측 기각.

**B1.C 결과** (paired 15쌍): coherence O 0.870 < W 0.886 (**p=0.0128 → R3 ✅ dispersion-defeat**); **wrong 의 평균 prompt-cos 0.227 > oracle 0.220** (mean 게임 구조적 열세 — 평균 계열 fix 의 원리적 한계 증명); visibility 는 oracle 우위 (visibility-defeat 기각). pumpkin 의 wrong-top1 = **zero-norm degenerate** (canon(0)=0.5 승리, D1 잔존 — 필터 1줄 과제), ottolenghi 쌍 제외.

**✱ 적대적 검증 — 공정 결투** ([scripts/stage3_2_fair_duel.py](../../../scripts/stage3_2_fair_duel.py)): B1.B (d) 는 타깃만 보정 + 경쟁자 동결 측정. oracle 과 wrong-top1 **둘 다** query-top-5 보정 시: **oracle 단독 승리 3/15** (pikachu, hand, plate), fair-rank ≤3 = 13/15 (대부분 rank 2 — wrong 이 1등 유지, D2 실종 → D3 오염 전이). → R2 의 "15/17 회복" 은 상한으로 정정, Stage 4 는 결합 신호 설계. **Output**: [stage3_2_fair_duel.csv](../../../output/diagnostics/stage3_2_fair_duel.csv)

상세: [stage3_2_b1_anatomy.md](stage3_2_b1_anatomy.md) (§6.5 적대적 검증 패치 포함)

---

## 📂 데이터 / Plot 한눈에

### CSV ([output/diagnostics/](../../../output/diagnostics/))

| Stage | 파일 | Rows | 내용 |
|---|---|---|---|
| 1 | b7_a4_combined.csv | 208 | THGS B7+A4 per (prompt, eval_frame) |
| 1 | b7_a4_combined_relags.csv | 208 | ReLaGS B7+A4 |
| 1 | a2_image_clip_ceiling.csv | 268 | A2 per (prompt, policy) |
| 1 | cross_method_d2_decomposition.csv | 67 | Per-prompt cross-method class |
| 2A | persistent_phantoms_17.csv | 17 | persistent phantom list |
| 2A | stage2a_layer1.csv | 33 | B8 mix-rate (17 phantom + 16 easy) |
| 2A | stage2a_layer2_forensic.csv | 17 | wrong top-1 forensic |
| 2A | stage2a_layer2_trajectory.csv | 51 | per-frame trajectory |
| 2A | stage2a_layer3_d3.csv | 51 | D3 sweep raw |
| 2A | stage2a_layer3_d3_summary.csv | 17 | per-phantom D3 summary |
| 2A | phantom_anatomy.csv | 17 × 26 cols | all-layer combined |
| 2A | phantom_attribution.csv | 17 | primary_mechanism + all_tags |
| 2B | _h2_lite_perview.pkl | 17×30 | per-view CLIP feature (pkl) |
| 2B | stage2b_rofa_subtypes.csv | 17 | F2.B subtype |
| 2B | stage2b_rofa_keep_mask.csv | 17 | F2.C keep/drop cos |
| 2B | stage2b_d3_full_sweep.csv | 208 | D3 k sweep raw |
| 2B | stage2b_d3_full_sweep_agg.csv | 67 | D3 per-prompt aggregate |
| 2B | stage2b_c3_quickcheck.csv | 18 | C3 λ sweep |
| ✱검증 | verify_f2_tau_sweep.csv | 17 | τ∈{1.0, 2.0} pathology/subtype 비교 |
| ✱검증 | verify_joint_canon_vs_raw.csv | 67 | joint 2×2 raw/canon robustness (변화 0) |
| ✱검증 | verify_h2lite_rank.csv | 17 | clean crop 의 scene-prompt rank |
| 3.1 | stage3_b8_replay_perview.pkl | 33 SP × all views | mixed/hard feature (fp16) + mix stats + 충실도 |
| 3.1 | stage3_b8_mix_stats.csv | 33 | per-SP B8 mix 통계 (최초 직접 측정) |
| 3.1 | stage3_b8_gap_summary.csv | 33 | clean/mixed/hard gap + rank + verdict 입력 |
| 3.1 | _h2_lite_easy.pkl | 16 | easy control 의 clean encoding (신규) |
| 3.2 | stage3_2_singleview_rank.csv | 33 | 단일/누적 pool rank + gate + 곡선 분류 |
| 3.2 | stage3_2_trajectory.csv | per SP×k | 누적 궤적 raw |
| 3.2 | stage3_2_ablation.csv | 33 | 12 aggregation variants pool rank |
| 3.2 | stage3_2_wrongtop1_perview.pkl | 17 | wrong-top1 SP 의 per-view dump |
| 3.2 | stage3_2_competition.csv | 15 | oracle vs wrong paired 비교 |
| 3.2 | stage3_2_fair_duel.csv | 15 | ✱공정 결투 (둘 다 query 보정) |
| 3.3 | stage3_3_allsp_<scene>.pkl ×4 | 전 SP×view | per-view feature foundation (1.37GB) |
| 3.3 | stage3_3_fullpool_ranks.csv | 67 | 동결 없는 재채점 rank (p/z/c) |
| 3.3 | stage3_3_top3_selections.pkl | 67×6 | variant 별 top-3 (lvl,sp) |
| 3.3 | stage3_3_mask_iou.csv | 208 | variant 별 mask IoU (R4 근거) |
| 3.3 | stage3_3_impostor.csv | 16 | 사기꾼 GT-containment (R5 근거) |
| 3.4 | stage3_4_kinship.csv | 17 | wrong vs oracle NAG 친족 |
| 3.4 | stage3_4_losers.csv | ~95 | 패자부활 실패자의 승자 신원 |
| 3.4 | stage3_4_easy_regression.csv | 30 | 역행 신규 승자 + 가드 신호 |
| 3.4 | stage3_4_multi_instance.csv | ~57 | 미표기 인스턴스 의심 (montage 증거) |
| 3.4 | **stage3_4_taxonomy.csv** | 36 | **case→원인→처방 전수 매핑 (Stage 4 명세)** |
| 4 | stage4_grid_ranks{,_v2}.csv | 67×(216/96) | config 별 oracle rank |
| 4 | stage4_selections{,_v2}.pkl | 67×configs | top-3(+union) 선택 |
| 4 | stage4_loso_choice{,_v2}.csv | 4 folds | LOSO 선택 감사 추적 |
| 4 | stage4_mask_iou{,_v2}.csv | long | held-out/calib mask IoU (R9 근거) |
| 4 | stage4_ablation.csv | 5 | v1 신호별 기여 (g1 결함 적발) |
| 5 | stage5_relags_allsp_<scene>.pkl ×4 | 전 SP×view | ReLaGS per-view foundation (1.2GB, R10 용) |
| 5 | stage5_relags_targets{,_perview}.{csv,pkl} | 36 | ReLaGS phantom 20 + easy 16 |
| 5 | stage5_g1_singleview.csv | 36 | 단일 view rank + 좋은 view 비율 |
| 5 | stage5_g2_fullpool.csv | 67 | base vs top5 rank (제로섬 근거) |
| 5 | stage5_g3_winners.csv | 120 | 승자 시그니처 |
| 5 | stage5_g4_rofa.csv | 67 | ROFA on/off rank (실측 net 효과) |

### Plot ([output/diagnostics/plots/](../../../output/diagnostics/plots/)) — 23 PNG

| Stage | Plots |
|---|---|
| 1 (within-method, 11) | b7_distributions, b7_by_scene, b7_purity_vs_completeness, b7_cross_view_stability, a4_distributions, a4_rank_zmargin_scatter, a4_rank_by_scene, a2_rank_per_policy, a2_top1_per_scene_policy, a2_rank_box, joint_d2real_vs_phantom |
| 1 (cross-method, 4) | cross_method_rank_scatter, cross_method_d2_stacked, cross_method_per_scene_stacked, cross_method_transition_matrix |
| 2A (5) | phantom_b8_mixrate, phantom_wrong_top1_types, phantom_d3_sweep, phantom_per_view_trajectory, **phantom_montage** |
| 2B (3) | stage2b_rofa_subtype_distribution, stage2b_keep_mask_geometry, stage2b_per_view_cos_dists |

---

<!-- ====================================================================== -->
<!--  ⬇⬇⬇  새 Stage 끝나면: 위 summary 표에 row 추가 + 여기 아래에 section 추가  ⬇⬇⬇ -->
<!-- ====================================================================== -->

## 📝 업데이트 로그

| 날짜 | 추가/수정 | 한 줄 요약 |
|---|---|---|
| Stage 2B 완료 시점 | 초기 작성 | Stage 1 (5 exp) + Stage 2A (5 exp + synthesis) + Stage 2B (5 exp) 코드 grounded 정의 + 결과 표 |
| 2026-06-10 | ✱ 검증 패치 | 검증 3종 section 신설: (1) τ=2.0 재실행 — F2.B subtype τ-robust, F2.C 정직 65%+12% no-drop / 실수 4건으로 정정 (sink→out, pikachu·onion→in). (2) joint raw/canon — 67 prompt 변화 0, Stage 1 robust. (3) H2-lite rank 재검증 — threshold 기준은 12/17 로 표기 교정, rank 기준 11/17 (65%) 유지, onion segments encoder 측 재분류. F2.B 의 GMM 미구현 (cos 스칼라 휴리스틱) 명시. |
| 2026-06-10 | **Stage 3.1 section 추가** | B8 인과 replay: 충실도 33/33, H-B8a/b/c 모두 사전 분기 ③ → **within-view 무죄 확정, B1 재조준**. language_features 4 scene 재생성 (재사용 가능, CLAUDE.md), per-view feature dump 확보 (B1 해부 foundation). |
| 2026-06-11 | **Stage 3.2 section 추가** | B1 anatomy: R1 (averaging 인과, 88% 단일 view 회복 가능) + R2 (query-cond top-k 15/17, reg 0 → Stage 4 방향) + R3 (dispersion-defeat p=0.013, wrong 의 mean prompt-cos 우위). gate 26/33 strict + jake 예외 명시. pumpkin wrong-top1 = zero-norm D1 잔존. |
| 2026-06-11 | **Stage 3.3 section 추가 (확정 진단)** | all-SP dump + full-pool 재채점 + mask-IoU 208쌍 + impostor 법의학. **R4 FAIL (제로섬)**, **R5 semantic confusion (32% 는 granularity)**, hybrid α=0.3 탐색 (순 +9). baseline 0.5424 = stage2b 0.542 재현, zero-norm 150 전수 확인. |
| 2026-06-11 | **Stage 3.4 section 추가 (잔여 전수 분해)** | 36 케이스 taxonomy 완성: R6 PASS (unknown 2) / R7 발동 (multi-instance 7 prompt) / R8 전원 재분류 (encoder 잔여 3건으로 축소 — Stage 1 의 D2.real 프레임 수정). 신규 원인 few-view opportunist + 가드 신호 g1/g2. Stage 4 method 요구사항 명세 산출. |
| 2026-06-12 | **Stage 4 section 추가 (method LOSO 검증)** | 정합성 gate 0/67×2 → v1 PARTIAL (+0.76) → ablation 으로 g1 결함 적발 → 재탐색 1회 (v2: g1 제거+g2 추가) → **held-out +3.74pt, phantom +17.5pt — 단 easy −4.08pt 로 R9 PARTIAL 최종**. g2 margin 분리자 포화 부검 포함. Stage 5 = 가드 분리자 재설계. |
| 2026-06-12 | **Stage 5 section 추가 (ReLaGS 메커니즘 재현)** | ROFA-포함 gate 97.8~100% (유령 21 독립 재확인) → **G1·G2·G3 전부 재현** (95%·23/51%·+8/−12) + **G4: ROFA 실측 효과 phantom median 0 (구출 1/20)** → "소수파 매장 = paradigm-level, ROFA 도 못 막음" 확정. ReLaGS dump 확보로 R10 즉시 가능. |

---

## 🔗 참고

- [intuition.md](intuition.md) — 직관 해석 (이 문서의 짝꿍)
- [README.md](README.md) — stage 진행 index
- [stage1_b7_a4_a2.md](stage1_b7_a4_a2.md) — Stage 1 상세 분석
- [stage2a_phantom_anatomy.md](stage2a_phantom_anatomy.md) — Stage 2A 상세 분석
- [stage2b_rofa_anatomy.md](stage2b_rofa_anatomy.md) — Stage 2B 상세 분석
- [../extended_failure_hypotheses.md](../extended_failure_hypotheses.md) — 18 가설 catalog
