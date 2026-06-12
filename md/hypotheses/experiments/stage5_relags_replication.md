# Stage 5 — 메커니즘 일반화: ReLaGS 자체 파이프라인에서의 재현

> 완료 (2026-06-12). Stage 3 의 진단 (소수파 매장·제로섬·승자 시그니처) 을 **ReLaGS 의 자체 partition + ROFA 포함 파이프라인** 위에서 재현. 모든 측정은 ROFA 포함 충실도 gate (97.8~100%) 를 통과한 replay dump 에서.
>
> 핵심 finding 한 줄: **"메커니즘은 method-agnostic 으로 확정 — 소수파 매장 (best 단일 95%, 좋은 view 23% vs 51%) 과 제로섬 (+8/−12) 이 ReLaGS 에서 거의 같은 숫자로 재현됐고, ROFA 의 실측 net 효과는 phantom 에 median 0 (구출 1/20) — ROFA 는 유령(D1) 청소부일 뿐 dilution 은 전혀 못 막는다."**

---

## 0. Question 과 사전 판정 (B1 §6.8)

진단의 메커니즘 증거가 전부 THGS 한정이었음 → "paradigm-level 결함" 주장의 마지막 빈칸. 사전 판정 G1~G5.

---

## 1. Phase A — ReLaGS replay + dump (전제)

[stage3_b8_replay.py](../../../scripts/stage3_b8_replay.py) 에 `--agg relags` (충실도 재구성을 ReLaGS 의 ROFA τ=2 경로로 — [ReLaGS/merge_proj.py:96-130](../../../ReLaGS/merge_proj.py#L96-L130) verbatim) + `--iteration 0` (HF 릴리즈 패키징) 확장. ReLaGS sai_nag: `ReLaGS/output/lerf_hf/scenes/LeRF/<scene>/` (자체 gaussians 234K, labels [5081/2103/463/177] — THGS 와 전혀 다른 partition).

| Scene | all-SP 충실도 (lvl2/lvl3) | 유령 SP |
|---|---|---|
| figurines | **100% / 100%** | 1 |
| ramen | 99.0% / 98.8% | 1 |
| teatime | 98.5% / 99.2% | 13 |
| waldo | 97.8% / 98.0% | 6 |

타깃 36/36 통과. **유령 합계 = 21 — cross_method 의 "ReLaGS 가 zero-norm 150→21 (86% 감소)" 를 dump 에서 독립 재확인** (THGS dump 150 과 대조).

산출물: stage5_relags_allsp_<scene>.pkl ×4 (1.2GB — R10 의 foundation), stage5_relags_targets_perview.pkl

---

## 2. 시그니처 재현 결과 — THGS ↔ ReLaGS 나란히

| 시그니처 | THGS | **ReLaGS** | 판정 |
|---|---|---|---|
| **G1** best 단일 view pool rank ≤3 (phantom) | 88% (15/17) | **95% (19/20)** | ✅ 재현 (기준 ≥60%) |
| **G1** 좋은 view 비율: phantom vs easy | 18% vs 50% | **23% vs 51%** (MW p<0.0001) | ✅ 재현 |
| **G2** query-top5 전면 적용: phantom 회복 / easy 역행 | +6 / −10 | **+8 / −12** (other +3) | ✅ 제로섬 재현 |
| **G3** 회복불능 phantom 의 승자 | GT-무관 다수, coherent (+few-view) | **GT-무관 82%, coh med 0.89, few-view 33%** | ✅ 시그니처 존재 |
| **G3** easy 역행의 승자 | lucky-view jump 외부 SP | **GT-무관 90%, few-view 0%** (동일 패턴) | ✅ |
| **G4** ROFA 실측 net 효과 (ReLaGS 고유) | (2B 시뮬: outlier_handled 0%) | **phantom median +0 (helped 6/hurt 5/neutral 9), rank≤3 구출 1/20** | 시뮬 예측 확정 |

(plot: `plots/stage5_signature_comparison.png`)

### G4 의 의미 — ROFA 의 정확한 가치 확정

ROFA 는 **유령 (zero-norm D1) 제거에는 강력** (150→21) 하지만, **소수파 매장 (dilution) 에는 사실상 무력** — phantom rank 에 대한 실측 효과가 median 0, 20개 중 1개만 구출. Stage 2B 시뮬레이션 ("ROFA 무죄 — 잡을 outlier 가 없다") 이 실제 파이프라인 분포에서 확정됨. → paper 의 한 문장: *"robust averaging 은 degenerate feature 는 고치지만 dilution 은 못 고친다 — 문제는 outlier 가 아니라 다수결 그 자체이기 때문."*

---

## 3. G5 종합 판정

> **G1 ✅ · G2 ✅ · G3 ✅ (+G4 확정) → "소수파 매장 메커니즘과 그 따름정리들은 method-agnostic 이다. ROFA 라는 방어기제조차 이를 막지 못한다."** — paper Section 2 의 마지막 빈칸 닫힘.

부수 정합성: ReLaGS phantom 20 의 분포·유령 21·D1 86% 감소가 모두 기존 cross_method 의 표면 관찰과 일치 — 이제 그 관찰들의 **mechanism 버전**이 갖춰짐.

---

## 4. 산출물

- **Data**: stage5_relags_allsp_<scene>.pkl ×4 (1.2GB), stage5_relags_targets{,_perview}.{csv,pkl}, [stage5_g1_singleview.csv](../../../output/diagnostics/stage5_g1_singleview.csv) (36), [stage5_g2_fullpool.csv](../../../output/diagnostics/stage5_g2_fullpool.csv) (67), [stage5_g3_winners.csv](../../../output/diagnostics/stage5_g3_winners.csv) (120), [stage5_g4_rofa.csv](../../../output/diagnostics/stage5_g4_rofa.csv) (67)
- **Plot**: `plots/stage5_signature_comparison.png`
- **Scripts**: stage5_g1_singleview / g2_fullpool / g3_winners / g4_rofa.py + replay `--agg relags`/`--iteration` 확장

## 5. Decision — 다음

1. **R10 이 즉시 가능해짐**: ReLaGS dump 가 생겼으므로 Stage 4 채점기 (stage4_scorer_v2) 를 ReLaGS 에 그대로 적용 → "처방도 method-agnostic" 검증 (반나절)
2. Stage 5-가드 재설계 (easy 보호 분리자) — 이제 THGS+ReLaGS 합산 모집단 (easy 84, phantom 37) 으로 설계 가능
3. 원저자 평가 경로 교차 확인 (paper 최종 숫자)

## 6. 관련 문서

- [stage3_2_b1_anatomy.md](stage3_2_b1_anatomy.md) / [stage3_3_definitive.md](stage3_3_definitive.md) (THGS 원본 시그니처) / [stage4_method.md](stage4_method.md) / [../extended_failure_hypotheses.md](../extended_failure_hypotheses.md) B1 §6.8
