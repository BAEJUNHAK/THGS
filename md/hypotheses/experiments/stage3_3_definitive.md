# Stage 3.3 — 확정 진단 (full-pool query-aware mask-IoU + 사기꾼 법의학)

> 완료 (2026-06-11). Stage 3.2 적대적 검증이 남긴 두 구멍 — "동결 없는 진짜 점수" 와 "사기꾼이 이기는 이유" — 을 닫는 확정 실험. proxy 없음: 동결 없는 전 pool 재채점 + 실제 mask mIoU + 67 prompt 전체.
>
> 핵심 finding 한 줄: **"query-aware top-k 는 phantom IoU 를 2배로 올리지만 (0.203→0.402) easy 를 거의 같은 양만큼 부수는 제로섬이다 (full-67 mIoU −0.1pt, R4 FAIL). 사기꾼의 정체는 frame-the-truth 가 아니라 진짜 semantic confusion (R5: GT-포함 2/16). 단 hybrid (α·mean + (1−α)·top-k, α=0.3) 는 rank 수준에서 순 +9 prompt — Stage 4 의 출발점."**

---

## 0. Question 과 사전 판정 (B1 §6.5, 실험 전 등록)

- **R4**: 어떤 training-free variant 가 *full-67 mIoU ≥ baseline+2pt AND phantom 개선 AND easy 손실 <1pt* → Stage 4 승격 / 미달 → query-aware 단독 한계 확정
- **R5**: 사기꾼 승리 view 의 GT-포함률 ≥50% → frame-the-truth (처방=tightness) / <50% → semantic confusion (처방=negative contrast·공간 분리)

---

## 1. 충실도 gate — all-SP dump

[stage3_b8_replay.py](../../../scripts/stage3_b8_replay.py) `--dump_all_sp` 확장: view 루프에서 이미 계산되는 전 SP feature 행렬을 저장만 추가 (4 scene, 1.37GB).

| Scene | lvl2 pass | lvl3 pass | sai_nag zero-norm SP |
|---|---|---|---|
| ramen | 98.2% | 98.8% | 4 |
| figurines | 99.6% | 100% | 48 |
| teatime | 98.8% | 99.3% | 47 |
| waldo_kitchen | 98.3% | 99.8% | 51 |

→ gate 통과. **부수 확인**: zero-norm 유령 SP 전수 **150개** — cross_method 문서의 "THGS D1 150개" 와 독립 경로로 정확히 일치. 추가 정합성: 아래 mask eval 의 baseline mIoU 0.5424 = stage2b_d3 의 0.542 재현.

---

## 2. 3.3-A — full-pool query-aware (동결 없음)

### Rank 수준 ([stage3_3_fullpool_rank.py](../../../scripts/stage3_3_fullpool_rank.py))

모든 SP 가 `canon(top-5-by-query-cos 가중합)` 으로 동시 재채점:

| 그룹 | baseline ≤3 | plain (p) | zero-filter (z) | coherence c0.1/0.3/0.5 |
|---|---|---|---|---|
| phantom 17 | 0 | **6** | 6 | 5/5/5 |
| easy 41 | 41 | **31 (−10!)** | 31 | 31/26/22 |
| other 9 | 0 | 5 | 5 | 5/5/5 |

- 극적 회복: pikachu 64→**1**, pumpkin 52→**1**, tesla 256→**4**, hand 46→**2**, old camera 130→17, ottolenghi 66→**3**
- 역행도 실재: sake cup 10→37, spoon 47→70, sink 4→10 — top-5 선택은 *모든* SP 의 운 좋은 view 도 증폭
- Stage 3.2 의 "15/17" 이 동결-경쟁자 환상이었음이 전 pool 에서 최종 확인. coherence 감점은 일관되게 무익.

### Mask-IoU 수준 — R4 확정 판정 ([stage3_3_mask_eval.py](../../../scripts/stage3_3_mask_eval.py), 208 (prompt,frame) 쌍, top-3 union 렌더)

| variant | full-67 mIoU (per-prompt mean) | phantom-17 Δ | easy Δ | **R4** |
|---|---|---|---|---|
| baseline | 0.5424 (= stage2b 0.542 ✓) | — | — | — |
| **plain top-5 (p)** | 0.5415 (−0.0009) | **+0.199 (0.203→0.402, 2배)** | **−0.098** | **FAIL** |
| zero-filter (z) | 0.5415 | +0.199 | −0.098 | FAIL |
| coherence λ=0.1/0.3/0.5 | 0.539/0.471/0.413 | +0.19/+0.11/+0.08 | −0.10/−0.17/−0.25 | FAIL |

> **R4 판정: 전 variant FAIL — "query-aware top-k 단독은 method 가 아니다" 확정.** phantom 이득 (+19.9pt × 17) 과 easy 손실 (−9.8pt × 41) 이 거의 정확히 상쇄되는 제로섬.

### ✱ 사후 탐색 (post-hoc, 사전 등록 아님) — hybrid 절충

`score = α·canon(mean) + (1−α)·canon(top-5)` rank 수준 sweep:

| α | phantom ≤3 | easy ≤3 | other ≤3 | 순변화 |
|---|---|---|---|---|
| 1.0 (=baseline) | 0 | 41 | 0 | — |
| **0.3** | **4** | **39 (−2)** | **7** | **+9** |
| 0.0 (=pure top-5) | 6 | 31 (−10) | 5 | +1 |

→ **절충이 순수 top-k 를 압도** (순 +9 vs +1). mean 의 안정성과 top-k 의 소수파 구조를 동시에 쓰는 방향 — Stage 4 의 첫 가설 (mask-IoU 검증 필요).

---

## 3. 3.3-B — 사기꾼 법의학, R5 판정

([stage3_3_impostor_forensics.py](../../../scripts/stage3_3_impostor_forensics.py), 16 impostor 측정 — pumpkin 의 wrong 은 zero-norm 유령이라 view 없음. GT 는 라벨 frame 에만 있어 containment 는 ref_frame 측정 — 한계 명시.)

| 분류 (ref_frame 의 wrong mask vs GT) | n | 사례 |
|---|---|---|
| **part-of-truth** (precision ≥ 0.87 — GT 내부 조각) | **5/16** | tesla (prec 1.00!), rubber duck 0.99, jake 0.98, old camera 0.98, sink 0.87 |
| **진짜 배경 confuser** (recall=precision=0.00) | **10/16** | ramen 6개 전부, cabinet, spoon, bear nose 등 |
| GT-포함률 ≥ 0.5 (R5 기준) | 2/16 (12.5%) | jake, sink |

> **R5 판정: frame-the-truth 기각 (12.5% < 50%) → semantic confusion 분기** — 처방은 negative contrast·공간 분리 계열.
>
> 단 세부 구조가 중요: **5/16 의 "사기꾼" 은 사실 진실의 조각** (tesla 의 wrong top-1 은 문 손잡이의 일부!) — 이들에게 baseline 의 실패는 confusion 이 아니라 granularity (부분만 잡음) 였고, query-aware 의 큰 회복 (tesla 256→4) 과 정합. 나머지 10/16 은 CLIP 의 눈에 진짜로 prompt 같아 보이는 배경 영역 (B1.C 의 "평균 prompt-cos 우위" 와 일치).

산출물: [stage3_3_impostor.csv](../../../output/diagnostics/stage3_3_impostor.csv), `plots/stage3_3_impostor_montage.png`

---

## 4. 종합 — 확정 진단 (Stage 1→3.3 의 최종 서사)

1. **실패의 81% 는 회복 가능한 phantom** (Stage 1, method-agnostic)
2. **within-view mixing 무죄** (3.1, 충실도 33/33)
3. **범인 = across-view 평균의 다수결**: 정답 view 는 소수파 (18% vs easy 50%), 경쟁자는 균질-그럴듯 (3.2)
4. **query-aware 단독은 제로섬** (3.3, R4 FAIL): phantom IoU 2배 ↔ easy −10pt 맞교환. "한 줄 fix" 는 존재하지 않음이 *확정*
5. **사기꾼의 32% 는 진실의 조각, 62% 는 진짜 confusion** (R5)
6. 잔여 한계 (정직): spoon·cabinet never_good, encoder-limit 5, zero-norm 유령 150개

### 한 줄로

> **"진단 끝: phantom 은 '소수파 매장' 이고, 만능 처방은 없다. mean 은 easy 를, top-k 는 phantom 을 지킨다 — Stage 4 는 이 둘의 결합 (hybrid α≈0.3, rank 순 +9) 을 mask 수준에서 입증하는 것."**

---

## 5. 산출물

- **Data**: stage3_3_allsp_<scene>.pkl ×4 (1.37GB — 전 SP per-view feature, 향후 모든 aggregation 실험의 foundation), [stage3_3_fullpool_ranks.csv](../../../output/diagnostics/stage3_3_fullpool_ranks.csv), [stage3_3_top3_selections.pkl](../../../output/diagnostics/stage3_3_top3_selections.pkl), [stage3_3_mask_iou.csv](../../../output/diagnostics/stage3_3_mask_iou.csv) (208), [stage3_3_impostor.csv](../../../output/diagnostics/stage3_3_impostor.csv)
- **Plots**: `plots/stage3_3_impostor_montage.png`
- **Scripts**: stage3_3_fullpool_rank.py / stage3_3_mask_eval.py / stage3_3_impostor_forensics.py + replay `--dump_all_sp`

---

## 6. Decision — Stage 4

**순수 query-aware 승격: NO-GO (R4 FAIL).** Stage 4 = **hybrid scoring 의 mask-수준 검증**:

1. `score = α·canon(mean) + (1−α)·canon(top-k)` 의 α×k grid 를 **mask mIoU** 로 (rank 아님) — 탐색의 α=0.3 (순 +9) 이 출발점. 기계는 전부 준비됨 (selections → mask_eval 재사용)
2. **calibration 규율**: α/k 를 scene-split 으로 튜닝 (leave-one-scene-out) — 67 prompt 에 직접 튜닝하면 overfitting
3. 결합 신호: + zero-norm 필터 (150 유령 제거 — 무해 확인됨) + part-of-truth 케이스용 계층 union (tesla 류는 parent SP 포함)
4. semantic confuser 10건에는 C3 negative contrast 재시도 — 단 이번엔 *hybrid score 위에서* (Stage 2B 의 C3 실패는 mean feature 위였음)
5. ReLaGS 에 동일 hybrid 적용 (method-agnostic claim) + 3DOVS 일반화

---

## 7. 관련 문서

- [stage3_2_b1_anatomy.md](stage3_2_b1_anatomy.md) (§6.5 적대적 검증 — 이 실험의 동기) / [stage3_b8_causal.md](stage3_b8_causal.md) / [../extended_failure_hypotheses.md](../extended_failure_hypotheses.md) B1 §6.5 설계+★★★ 결과
