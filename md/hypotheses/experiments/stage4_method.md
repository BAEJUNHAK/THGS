# Stage 4 — Taxonomy 기반 결합 method 의 LOSO mask-mIoU 검증

> 완료 (2026-06-12). 진단 (Stage 1→3.4) 이 명세한 부품들을 단일 채점기로 조립하고, **leave-one-scene-out (LOSO)** 규율의 mask-mIoU 로 검증. 사전 등록 R9 의 재탐색 1회까지 사용한 **최종 판정: PARTIAL** — headline 은 기준 통과 (+3.74pt ≥ +2.0), easy 보호 기준 (−4.08pt, 기준 <1.0) 미달.
>
> 핵심 finding 한 줄: **"hybrid 본체는 진짜다 — held-out 에서 full-67 +3.74pt, phantom +17.5pt, other +13.3pt (multi-instance 제외 시 +4.53pt). 그러나 easy 보호 가드는 아직 미완 — margin 기반 g2 는 canon 점수의 스케일 포화로 분리력이 없었다 (41 easy 중 1개만 보호). 처방의 코어는 입증, 가드는 다음 설계 question."**

---

## 0. Question 과 사전 판정 (B1 §6.7)

R9: held-out full-67 ≥ baseline+2pt AND easy 손실 <1pt → 확정 / +0.5~2pt → partial (재탐색 1회) / 미달 → 정직 보고. **재탐색 1회 사용 후 최종 PARTIAL.**

---

## 1. 채점기 (training-free, query-time)

`score = α·canon(mean_feat) + (1−α)·canon(topk_q_feat)` + 신호: **Z** zero-norm 필터 (유령 150 차단, 상시) / **E** min-evidence (nval<τ_v 면 top-k 항→mean, few-view opportunist 차단) / v1 **G** g1 전역 클램프 / v2 **g2** prompt-적응 (mean margin≥τ_c 면 순수 mean) / **P** parent-union (선택 SP 의 NAG 부모·자식 ε=0.05 이내 합류).

**정합성 gate (Phase A)**: ① α=1+신호off → baseline rank 재현 **0/67 불일치** ② α=0 → stage3_3 rank_p **0/67 (정확 일치)** — grid 측정 전체가 신뢰 가능.

**LOSO 규율 (Phase B)**: fold 마다 calib 3 scene 의 rank-proxy 로 shortlist 3 → calib mask-mIoU 로 최종 선택 → held-out 은 평가에만 사용. 선택 기록: [stage4_loso_choice{,_v2}.csv](../../../output/diagnostics/stage4_loso_choice_v2.csv).

---

## 2. v1 결과 (grid 216) — PARTIAL #1 + ablation 의 교훈

| | full-67 | phantom17 | easy | other |
|---|---|---|---|---|
| v1 held-out | +0.76pt | +13.95pt | **−5.40pt** | +3.89pt |

**Ablation 의 자백**: figurines fold 에서 **g1 을 끄면 +8.7pt** (0.533→0.619) — 전역 클램프는 사기꾼의 점프와 함께 **정답의 점프 (phantom 회복의 원천) 까지 억압**. E (min-evidence) 는 held-out 에서 중립. → 재탐색의 방향이 ablation 으로 정당화됨: g1 제거, 3.4-B 의 g2 (미구현이었음) 추가.

## 3. v2 결과 (재탐색 1회, grid 96) — 최종

| | full-67 | phantom17 | easy | other | multi-inst 제외 (n=59) |
|---|---|---|---|---|---|
| baseline | 0.5424 | 0.2029 | 0.7396 | 0.2856 | 0.5766 |
| **v2 held-out** | **0.5798 (+3.74pt)** | **0.3783 (+17.54pt)** | 0.6988 (**−4.08pt**) | **0.4183 (+13.28pt)** | **0.6219 (+4.53pt)** |

fold 선택: a0.5_k5_tv5_c0.1 (figurines·teatime) / a0.5_k3_tv5_c0.1 (ramen) / a0.3_k5_tv10_c0.1 (waldo) — α=0.5, k=3-5, τ_v=5-10 에서 안정적, **τ_c 는 전 fold 가 경계값 0.1 선택**.

### g2 사후 부검 (정직 보고)

- canon margin 의 스케일: easy median **0.013** (q25 0.005) → τ_c=0.1 기준 **41 easy 중 1개만 보호** — g2 는 사실상 불활성. **+3.74pt 는 hybrid 본체 (α·mean+top-k + Z/E) 의 힘.**
- margin 은 정답 확신의 나쁜 proxy: plastic ladle 은 baseline IoU 0.843 인데 margin 0.0002 → 미보호 → **ours 0.000 (−0.84, 최대 단일 손실)**. canon-contrast 점수의 포화 (softmax) 가 원인.
- 손실 분해: 보호된 easy 손실 합 −0.17 vs 미보호 −1.50 — **가드 개념의 방향은 맞고, 분리 신호가 틀렸다** (margin → 다른 confidence 지표 필요).

---

## 4. 최종 판정과 의미

> **R9 = PARTIAL (최종)** — headline 기준 통과 (+3.74 ≥ +2.0), easy 보호 기준 미달 (4.08 ≥ 1.0). 사전 등록상 재탐색 소진 → 이 결과를 그대로 보고.

**Paper 관점**:
1. **주장 가능**: "진단이 명세한 hybrid 채점은 LOSO held-out 에서 +3.74pt (phantom 2배, recoverable 그룹 +13~17pt); 모든 설계 선택이 측정된 원인에 1:1 대응; multi-instance 의심 트랙 분리 시 +4.53pt."
2. **한계로 명시**: easy −4.08pt — 정밀도 일부를 회복력과 교환. 가드의 *개념* 은 손실 분해로 지지되나 margin 분리자가 부적합 (포화) — **Stage 5 question**: 더 나은 prompt-수준 confidence (예: 두 체제 top-1 일치 여부, mean top-1 의 pool z-score, rank-stability).
3. 부수 확인: 진단의 케이스 예측과 회복 명단 일치 (pikachu·pumpkin·ottolenghi·tesla 류 회복 / spoon·cabinet·multi-instance 트랙은 예상대로 미회복 — 별도 트랙).

---

## 5. 산출물

- [stage4_grid_ranks{,_v2}.csv](../../../output/diagnostics/stage4_grid_ranks_v2.csv) (216/96 configs × 67) / stage4_selections{,_v2}.pkl
- [stage4_loso_choice{,_v2}.csv](../../../output/diagnostics/stage4_loso_choice_v2.csv) — fold 별 선택 근거 (LOSO 감사 추적)
- [stage4_mask_iou{,_v2}.csv](../../../output/diagnostics/stage4_mask_iou_v2.csv) / [stage4_ablation.csv](../../../output/diagnostics/stage4_ablation.csv)
- Scripts: [stage4_scorer.py](../../../scripts/stage4_scorer.py) / [stage4_scorer_v2.py](../../../scripts/stage4_scorer_v2.py) / [stage4_loso_eval.py](../../../scripts/stage4_loso_eval.py)

## 6. Decision — 다음

1. **Stage 5 (가드 재설계, 새 사전등록 필요)**: margin 대신 *체제 간 top-1 일치* / pool z-score / rank-stability 를 분리자로 — easy 손실 <1pt 달성 시 R9 PASS 격상 가능.
2. **R10 (ReLaGS 일반화)**: 동일 채점기를 ReLaGS 에 — method-agnostic claim (ReLaGS 용 all-SP dump 필요, 별도 충실도 gate).
3. **원저자 평가 경로 교차 확인**: 최종 숫자를 test_lerf.py+eval_seg.py 프로토콜로 재확인 (paper 방어).
