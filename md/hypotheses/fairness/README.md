# `fairness/` — 공정 비교 트랙 (THGS · ReLaGS · VALA)

> 생성 2026-06-18. `strategy/`·`experiments/`의 새 peer. 새로 오면 **여기부터** 읽는다.
>
> 트랙 질문 한 줄: **"세 method를 *공정하게* 비교하려면 무엇을 동일하게 맞춰야 하고, 그 위에서 누가 이기나(리더보드) / 무엇이 원인인가(메커니즘)?"**

---

## 🧭 3 레이어 / 2 주장 (한눈에)

```
레이어:  EVALUATION(평가) │ SUBSTRATE(기반) │ METHOD(방법)
주장 A 리더보드: EVALUATION만 동일 → 최종 mask를 단일 하네스로 동일 채점 (각자 best 임계)
주장 B 메커니즘: 1개 요소 빼고 전부 동일 → 공유 dump 위 aggregation만 교체
```
자세히: [protocol.md](protocol.md) · P1-VALA: [p1_vala_protocol.md](p1_vala_protocol.md) · 직관: [intuition.md](intuition.md)

## 🚫 하드 룰 (배너)

> **leaderboard/는 mechanism/ dump를 절대 안 읽고, mechanism/은 native cross-method를 절대 안 돌린다. `common/`은 코드만 공유, 결과는 안 공유.**

## 📋 트랙 상태 (단일 소스 — stage 상태는 여기, RF 예측/판정은 [preregistration_log.md](preregistration_log.md))

| Stage | 주장 | 질문 | 상태 | 핵심결과 | 산출물 | RF-ID |
|---|---|---|---|---|---|---|
| lb01 | A | 모든 method에 적용할 *동일* 임계 선택 규칙은? | ⏳ 설계 전 | — | — | RF-L1 |
| lb02 | A | 단일 하네스로 THGS/ReLaGS/VALA 재채점 시 순위는? | ⏳ 설계 전 | — | — | RF-L2 |
| mc01 | B | 공유 dump 위 aggregation만 바꾸면 metric이 어떻게? | ⏳ 설계 전 | — | — | RF-M1 |
| mc02 | B | phantom 실패를 aggregation에 귀속 가능한가(나머지 통제)? | ⏳ 설계 전 | — | — | RF-M2 |
| p1v-d1 | D | VALA official 2D mask를 우리 prompt/class taxonomy로 split하면? | ✅ 완료(retrospective) | scene-dependent: phantom이 uniform하게 해결되지 않음. 단 mechanism 주장은 불가 | `output/diagnostics/p1e_vala_official_2d_prompt_*.csv` | — |
| p1v-m1 | B precursor | VALA-native level/threshold oracle vs actual gap은? | ✅ 완료 | failed rows 73개 중 selection/calibration recoverable failure 22개(30.1%). RF-V1 지지 | `output/diagnostics/p1e_vala_native_2d_oracle_official_actual_*.csv` | RF-V1 |
| p1v-m2 | B | VALA 안에서 robust-gate vs mean/non-gated가 무엇을 고치나? | ⏳ GPU 필요 | — | robust/mean ablation 예정 | RF-V2 |

> 현재 생성됨: framework 문서(protocol·README·preregistration_log·intuition)와 P1-VALA 전용 protocol. leaderboard/·mechanism/·methods/ 서브트리, config, common harness는 **미생성** (다음 단계).

## 🔗 구성 (계획 — 아직 미생성)

- `leaderboard/` (Claim A): README · harness_spec · lb01 · lb02
- `mechanism/` (Claim B): README · dump_spec · mc01 · mc02
- `methods/`: thgs.md · relags.md · vala.md (method 정체성 spec sheet)
- `configs/fairness/`, `scripts/fairness/`, `output/diagnostics/fairness/` (코드·config·산출물; 다음 단계)

## 📌 이 트랙의 동기 (기존 발견들)

- VALA 재현 검증 최신 정정: ramen target은 0.604가 아니라 **0.4541**이고 `VALA official feature + ReferSplat RGB`에서 **0.4589**로 재현권. 현재 진짜 gap은 **Waldo Kitchen**: official 3D 0.4904 vs paper 0.5571, official 2D 0.5412@0.5 / 0.5575@0.4 vs paper 0.651 ([../strategy/vala문제.md](../strategy/vala문제.md)).
- THGS↔ReLaGS 진단은 *공정*(공유 A2 + per-method oracle); VALA regime-split은 *confounded*(라벨 전이) — [protocol.md §3](protocol.md).
- P1-VALA에서는 claim lane을 셋으로 분리한다: native descriptive audit(D), common-harness leaderboard(A), within-VALA mechanism(B) — [p1_vala_protocol.md](p1_vala_protocol.md).
- 그래서 표 B(성능 리더보드)는 **단일 하네스로 재채점하기 전엔 무효** — 이 트랙이 그걸 고침.

## 관련
- [protocol.md](protocol.md) · [p1_vala_protocol.md](p1_vala_protocol.md) · [preregistration_log.md](preregistration_log.md) · [intuition.md](intuition.md)
- [../strategy/roadmap.md](../strategy/roadmap.md) (표 A/표 B 위계) · [../../cross_method/](../../cross_method/) (기존 비교 narrative)
