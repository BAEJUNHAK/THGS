# LERF-OVS Failure Analysis — 실험 개요 및 계획

> Last updated: 2026-06-04
> Status: Plan (실행 전)

---

## 0. 한 줄 요약

LERF-OVS 점수의 헤드룸 **Gap = 0.2437** (Oracle 0.8323 − Actual 0.5886, 3sc) 안을 **prompt 단위**로 분해해 공격 가능한 failure mode를 좁힌다.
모든 prompt에 대해 동시에 측정한다:

1. **Oracle IoU** — SAM lift 한계
2. **Actual IoU** — CLIP matching 결과
3. **CLIP rank of correct SP** — matching 실패 mechanism
4. **Pixel Precision / Recall / FP / FN** — error 유형 (모든 데이터에 일괄 적용)

→ Type A/B/C 분류 + Type A 내부 A1/A2/A3 sub-classify + 전체 FP/FN 통계.

---

## 1. 배경 (Why this analysis)

### 1.1 이전 분석에서 확정된 사실

- LERF-OVS는 (a) SAM lift, (b) CLIP semantic matching 두 단계로 분해 가능.
- Oracle v4 budget=3 (GT-guided 최적 SP 선택) 은 (a)만 isolate 하는 ceiling.
- 측정값 (3 scene 평균):

  | Stage 분리 | LERF-OVS 점수 (mIoU) |
  |--|--|
  | Oracle (SAM lift ceiling) | **0.8323** |
  | Actual (CLIP topk=3) | **0.5886** |
  | **Gap (CLIP matching 손실)** | **0.2437** |

- → Gap 0.2437은 NAG가 가진 SP 풀 안에 이미 정답이 있는데 CLIP이 못 골라서 잃는 점수. **공격 대상**.

### 1.2 분석의 핵심 질문

LERF-OVS Gap 0.2437이:

1. 몇 개의 prompt에서 발생하는가? (size)
2. 그 prompt들이 어떤 mechanism으로 실패하는가? (CLIP feature? Ranking algo? Hierarchy?)
3. Error가 어떤 유형인가? (FP-dominant? FN-dominant?)
4. 실패 prompt들이 어떤 의미적 특성을 갖는가? (brand? sub-part? character?)

→ 위 4가지를 정량 답하면 attack point가 결정된다.

---

## 2. 분석 Framework

### 2.1 Stage 분리

| Stage | 실험 | 측정 metric |
|--|--|--|
| SAM lift 단계 | Oracle (v4 greedy budget=3) | Oracle IoU |
| CLIP matching 단계 | Actual (test_lerf.py, topk=3) | Actual IoU |
| Matching 손실 | Oracle − Actual | per-prompt Gap |

같은 sai_nag.pt, 같은 SP 풀 (level=[2,3]), 같은 budget=3, 같은 render 메커니즘. **차이는 SP 선택 방식뿐**.

### 2.2 Failure type 분류 (per prompt)

| | Actual IoU ≥ τ_high | Actual IoU < τ_high |
|--|--|--|
| **Oracle IoU ≥ τ_high** | **C: 성공** | **A: CLIP matching 실패** ← 공격 대상 |
| **Oracle IoU < τ_high** | (이론상 불가능) | **B: SAM lift 실패** (NAG 한계) |

- τ_high 후보: 0.25 또는 0.5 (LERF-OVS 평가의 일반 기준).
- 두 threshold 모두 보고 (sensitivity 확인).

### 2.3 Type A 내부 sub-classification

Type A 중에서 정답 SP의 CLIP rank로 한 번 더 자른다 (정답 SP = oracle greedy v4의 first pick):

| sub-type | 조건 | 진단 | 후보 attack |
|--|--|--|--|
| **A1** | correct SP의 CLIP rank ≥ 10 | CLIP feature noisy | per-SP feature refinement, multi-view CLIP aggregation |
| **A2** | rank in [2, 10] | ranking algo (canon-contrast topk) 한계 | matching algo 교체, hard-prompt augmentation |
| **A3** | rank = 1 이지만 fragmentation으로 IoU 낮음 | topk=3 부족 / level 선택 | adaptive topk, hierarchy sweep |

### 2.4 Pixel-level FP/FN — 모든 데이터에 일괄 적용

Type 분류와 *별개의 축*. 모든 prompt × frame에 대해:

| Metric | 정의 |
|--|--|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| FP rate | FP / (TP + FP + FN) |
| FN rate | FN / (TP + FP + FN) |
| FP/FN ratio | FP / (FP + FN) → 1에 가까우면 over-segment, 0이면 under-segment |

→ 이를 Type 축과 cross-tab 하면:
- Type A 안에서도 FP-dominant prompts vs FN-dominant prompts가 갈림
- Type B는 거의 FN-dominant일 것 (정답 자체가 없으니 안 채워짐)
- Type C는 FP/FN 모두 낮을 것 (sanity check)

전체 prompt 평균 FP/FN은 다른 3D OVS 논문 표와 직접 비교 가능 (standard metric).

---

## 3. Metrics 정의 (스크립트가 출력해야 할 값)

per (scene, prompt, frame) 한 row:

| Column | 정의 | 사용처 |
|--|--|--|
| `scene` | 장면 이름 | grouping |
| `prompt` | 텍스트 쿼리 | grouping |
| `frame` | view 이름 | grouping |
| `is_ref_frame` | 이 frame이 prompt의 ref인지 | filter |
| `gt_pixels` | GT mask pixel 수 | normalization |
| `oracle_iou` | v4 greedy budget=3 union vs GT | Type 분류 |
| `actual_iou` | CLIP topk=3 union vs GT | Type 분류 |
| `correct_sp_count` | greedy가 선택한 SP 개수 (≤3) | reference |
| `correct_sp_clip_rank` | first-pick SP의 CLIP score 순위 (1=top) | A1/A2/A3 |
| `clip_top1_oracle_iou` | CLIP top-1 SP의 oracle-IoU | 추가 진단 |
| `precision` | pixel-level | FP/FN 축 |
| `recall` | pixel-level | FP/FN 축 |
| `fp_pixels` | Actual − GT | FP/FN |
| `fn_pixels` | GT − Actual | FP/FN |
| `type` | A / B / C (τ=0.25, τ=0.5 두 column) | 분류 |
| `sub_type` | A1 / A2 / A3 (Type A에 한정) | 분류 |
| `prompt_category` | object / brand / character / sub-part (manual) | 카테고리 분석 |

---

## 4. 실행 계획 — Step-by-step

| Step | 작업 | 산출물 | 예상시간 |
|--|--|--|--|
| 0 | 진단 데이터 생성 스크립트 작성 + 실행 | `output/diagnostics/lerf_ovs_per_prompt.csv` | 1.5h |
| 1 | Type A/B/C 분류 + 헤드룸 size 계산 | type 분포 표 | 20m |
| 2 | A1/A2/A3 sub-classify | sub-type 분포 표 | 20m |
| 3 | 전체 prompt FP/FN 통계 + cross-tab (Type × FP/FN) | FP-vs-FN dominance 표 | 30m |
| 4 | Attack point 결정 + ROI 표 | `attack_priorities.md` | 30m |

**합계: ~3시간** (모두 단일 머신 GPU 2)
**Prompt category 라벨링은 제외** — mechanism (A1/A2/A3)이 attack 설계에 충분하고 카테고리는 attack code path에 영향 없음. paper 단계에서 필요해지면 그때 추가.

### Step 0 상세 — 진단 스크립트

**파일**: `scripts/lerf_ovs_diagnostic.py` (신규)

**기능**:
1. Scene loop (figurines, ramen, teatime, [waldo_kitchen optional])
2. 각 prompt마다:
   - ref frame = prompt의 첫 GT frame (v4와 동일)
   - 모든 GT frame에서 측정
3. Per-SP at ref frame:
   - 각 SP 단독 mask 렌더링 (level=[2,3]) → cache
   - Per-SP oracle IoU vs GT@ref 계산
   - Per-SP CLIP relevancy score 계산 (test_lerf.py 동일 함수)
4. Per-prompt:
   - Oracle: greedy v4 selection at ref → SP set
   - Actual: CLIP top-3 SPs (level=[2,3]) → SP set
   - 각 frame에서 두 set union 렌더 → IoU, precision, recall, FP, FN
5. CSV 저장: `output/diagnostics/lerf_ovs_per_prompt.csv`

**재사용 가능한 코드**:
- `sam_oracle_v4_lerf_ovs.py::cache_sp_masks_at`, `greedy_union_select`
- `test_lerf.py`의 CLIP scoring 부분 (canon-contrast)
- `scripts/eval_lerf_ovs_pairs.py`의 IoU/BIoU 계산

**구조 (의사코드)**:
```python
for scene in scenes:
    load gaussians, NAG, CLIP text encoder
    for prompt in prompts:
        ref = prompt.first_gt_frame
        sp_masks = cache_sp_masks_at(ref)              # v4 동일
        sp_oracle_ious = [iou(m, gt_at_ref) for m in sp_masks]
        sp_clip_scores = clip_score(prompt, sp_features)  # test_lerf 동일
        
        oracle_set = greedy_union_select(sp_masks, gt_at_ref, budget=3)
        clip_set   = argsort(sp_clip_scores)[:3]
        
        first_pick_sp = oracle_set[0]
        correct_sp_clip_rank = rank_of(first_pick_sp, sp_clip_scores)
        
        for frame in prompt.gt_frames:
            oracle_mask = render(oracle_set @ frame)
            actual_mask = render(clip_set @ frame)
            gt_mask     = rasterize_polygon(prompt, frame)
            
            write_row(...)
```

### Step 1 상세 — Type 분류

`scripts/lerf_ovs_diag_analyze.py::classify_types`

```python
df = pd.read_csv("output/diagnostics/lerf_ovs_per_prompt.csv")
# ref frame만 사용 (Type 분류의 기준)
ref = df[df.is_ref_frame]
ref['type_25'] = classify(ref.oracle_iou, ref.actual_iou, tau=0.25)
ref['type_50'] = classify(ref.oracle_iou, ref.actual_iou, tau=0.50)

# 출력: Type별 prompt 개수, 잃은 IoU 총량
print(ref.groupby('type_25').agg(
    n_prompts=('prompt', 'count'),
    iou_lost=lambda x: (x.oracle_iou - x.actual_iou).sum(),
))
```

**확인 사항**:
- Type A의 prompt 개수와 비율 (전체 대비)
- Type A의 잃은 IoU 합 = 공격 가능 헤드룸
- Type B의 비율 → SAM lift 한계의 크기 (별도 라인)

### Step 2 상세 — A1/A2/A3

Step 1에서 Type A로 분류된 prompt만 대상:
```python
A = ref[ref.type_25 == 'A']
A['sub_type'] = A.correct_sp_clip_rank.apply(
    lambda r: 'A1' if r >= 10 else ('A2' if r >= 2 else 'A3'))
```

산출: A1/A2/A3 비율 + 각 sub-type의 평균 oracle_iou, actual_iou.

### Step 3 상세 — 전체 FP/FN 통계

**모든 prompt (Type 무관)** 에 대해:
```python
df['fp_rate'] = df.fp_pixels / (df.fp_pixels + df.tp_pixels + df.fn_pixels)
df['fn_rate'] = df.fn_pixels / (df.fp_pixels + df.tp_pixels + df.fn_pixels)
df['fp_dominance'] = df.fp_pixels / (df.fp_pixels + df.fn_pixels + 1e-6)
```

산출:
- 전체 평균 precision / recall (논문 표용)
- Type × FP-dominance 교차표:

  | | FP-dominant (>0.7) | balanced | FN-dominant (<0.3) |
  |--|--|--|--|
  | Type A | ? | ? | ? |
  | Type B | ~0 | ~0 | ~100% |
  | Type C | ~0 | balanced | ~0 |

### Step 4 상세 — Attack priorities

ROI 표:

| Attack | 영향 sub-type | 영향 받는 prompt 수 | 잠재 IoU 회복 | 구현 난이도 | ROI |
|--|--|--|--|--|--|
| Multi-view CLIP aggregation | A1 | ? | ? | medium | ? |
| Canon-contrast 대체 | A2 | ? | ? | medium | ? |
| Adaptive topk / level sweep | A3 | ? | ? | low | ? |

**가장 큰 sub-type에 자원 집중**.

---

## 5. 실행 명령 (사전 정의)

```bash
# 0. env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate thgs
export CUDA_VISIBLE_DEVICES=2

# Step 0: 진단 데이터 생성 (스크립트 작성 후)
for sc in figurines ramen teatime; do
    python scripts/lerf_ovs_diagnostic.py \
        -s data/lerf/$sc -m output/lerf/$sc \
        --out output/diagnostics/lerf_ovs_per_prompt.csv \
        --append
done

# Step 1-5: 분석
python scripts/lerf_ovs_diag_analyze.py \
    --csv output/diagnostics/lerf_ovs_per_prompt.csv \
    --out md/lerf_ovs_failure_analysis_results.md
```

---

## 6. Expected outcomes (가설)

작업 시작 전 가설 — 분석 후 검증:

1. **Type A가 전체 prompt의 50-70%** 를 차지 (가장 큰 헤드룸).
2. **Type B는 미세 객체(작은 sub-part) 에 집중** (NAG가 못 잡음).
3. **A1 (CLIP feature 문제) 이 A2/A3보다 큼** — 정답 SP rank가 일반적으로 10등 이후로 밀려있을 것.
4. **전체 FP/FN 통계는 FP-dominant** — CLIP이 distractor SP까지 끌어들이는 게 주된 error 양상.
→ 가설 1-2 검증되면: attack은 Type A에 집중.
→ 가설 3 검증되면: per-SP CLIP feature 개선이 1순위.
→ 가설 4 검증되면: distractor 억제(matching algo)도 우선순위.

---

## 7. 산출물 (deliverables)

| 파일 | 내용 |
|--|--|
| `scripts/lerf_ovs_diagnostic.py` | 진단 데이터 생성 스크립트 |
| `scripts/lerf_ovs_diag_analyze.py` | 분석 스크립트 |
| `output/diagnostics/lerf_ovs_per_prompt.csv` | row-level 진단 데이터 |
| `output/diagnostics/sp_clip_scores.json` | per-SP CLIP score (sanity check용) |
| `md/lerf_ovs_failure_analysis_results.md` | 분석 결과 (이 파일과 짝) |
| `md/attack_priorities.md` | Step 5 산출, 다음 phase 시작점 |

---

## 8. 이후 phase (이 분석 이후)

이 plan은 **failure mode 진단까지만**. 이후:

- **Phase 2**: 가장 큰 sub-type에 맞춘 ablation 실험 설계 (e.g., multi-view CLIP aggregation prototype)
- **Phase 3**: 다른 method (LangSplat, Gaussian Grouping) 에 동일 framework 적용 — 단, 각 method의 inference 설정에 맞춘 fair oracle 재정의 필요 (이전 결과 참조)
- **Phase 4**: novel method 설계 + 평가

이 plan의 결과가 명확하지 않으면 Phase 2 진입 전 plan 재검토.

---

## 9. 사전 결정 사항 (확정됨)

| # | 항목 | 결정 |
|--|--|--|
| 1 | Scene 범위 | **figurines, ramen, teatime, waldo_kitchen (4 scene 모두)** |
| 2 | τ_high (Type 분류 임계값) | **0.25 와 0.5 둘 다** (두 column 모두 출력, 주 분석은 0.25) |
| 3 | 측정 frame 범위 | **ALL annotated frames** (LangSplat 표준 매칭, 2026-06-04 갱신) |
| 4 | CLIP rank 풀 범위 | **level [2, 3] union** (Actual 매칭이 사용하는 풀과 동일) |
| 5 | Prompt category 라벨링 | **제외** (mechanism이 attack code path 결정에 충분) |
| 6 | Headline aggregation | **LangSplat-style** (per-image mean → per-scene mean → overall) |
| 7 | Failure analysis aggregation | **per-prompt** (mean over frames per prompt; pixels summed for FP/FN) |

→ 모든 결정 완료. 실행 완료 (Section 10 변경 로그 참조).

---

## 10. 변경 로그

- 2026-06-04: 초안 작성. Oracle vs Actual framing, Type A/B/C + A1/A2/A3 + FP/FN 전체 적용.
- 2026-06-04: Section 9 사전 결정 사항 확정 (4 scene, τ=0.25+0.5, ref-frame only, level [2,3] union pool, manual category).
- 2026-06-04: Prompt category 제외 결정 (mechanism이 충분).
- 2026-06-04: **ref-frame → all-frame 확장**. Headline aggregation = LangSplat 표준 (`eval_seg.py` 와 동일). Failure analysis = per-prompt aggregation. Diagnostic CSV가 이제 per (scene, prompt, frame) row로 출력.
- 2026-06-04: 실험 + 분석 완료. 결과 [lerf_ovs_failure_analysis_results.md](lerf_ovs_failure_analysis_results.md).
