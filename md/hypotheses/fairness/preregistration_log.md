# fairness 사전등록 로그 (RF-L* / RF-M* / RF-V*) — 단일 소스

> 생성 2026-06-18. 짝꿍: [protocol.md](protocol.md) · [README.md](README.md).
>
> 이 파일 = fairness 트랙의 모든 사전등록 예측·판정의 **단일 소스**. (기존 본선 R-ID는 R10~R13까지 사용 중 — 충돌 방지 위해 fairness는 **RF-L*/RF-M*** 새 네임스페이스.)
>
> **규율**: 각 RF-ID는 *드라이버 실행 전* 여기서 `예측`+`기준`+`고정일`로 태어난다. 같은 ID가 해당 stage 문서의 결과표(`사전예측 | 실측 | 판정`)에 다시 등장한다. 빗나가면 정직하게 보고 (THGS 본선의 R11/R13 관행 그대로). 이미 실행된 retrospective audit은 RF-ID를 소급 부여하지 않는다.

- **RF-L\*** = Fairness · Leaderboard 주장
- **RF-M\*** = Fairness · Mechanism 주장
- **RF-V\*** = Fairness · P1-VALA 전용 주장

---

## 사전등록 표

| RF-ID | 주장 | 예측 | 판정 기준 | 고정일 | 상태 | 결과 |
|---|---|---|---|---|---|---|
| **RF-L1** | A | method마다 *자기 best threshold* 를 동일 규칙으로 고르면 native 숫자보다 공정/안정적. (단 VALA는 이미 이진화 PNG라 re-export/re-render 필요할 수 있음) | 4 method 모두 threshold sweep에서 단일 best-rule 적용 가능 + export 경로 확보 | (미정) | ⏳ pending | — |
| **RF-L2** | A | 단일 하네스 재채점 시 THGS/ReLaGS/VALA 순위가 native-protocol 표와 *달라질* 수 있다 (특히 VALA가 자기 관대한 protocol에서 부풀려졌다면 하향) | 동일 GT·metric·집계·각자 best-thresh로 4 scene mIoU 산출, native 표와 차이 보고 | (미정) | ⏳ pending | — |
| **RF-M1** | B | 공유 dump 위 aggregation만 바꾸면(mean/median/top-k/hybrid) metric이 유의하게 변함 (mean=phantom 매장) | 동일 dump·동일 EVAL, rule만 교체 → per-rule Δ 측정 | (미정) | ⏳ pending | — |
| **RF-M2** | B | 나머지 전부 통제 시 phantom 실패율이 aggregation rule에 귀속됨 (mean에서 최대, 선택형에서 감소) | 같은 dump에서 rule별 phantom 회복/easy 역행 분리 | (미정) | ⏳ pending | — |
| **RF-V1** | B precursor | VALA official 2D feature map 안에서도 actual chosen level/threshold와 oracle 후보 사이에 non-trivial gap이 남는다. 즉 실패가 전부 representation 부재가 아니라 selection/calibration으로도 분해된다. | P1V-M1에서 `vala_selection_fail` 또는 `vala_calibration_fail` 비율이 failed rows(`actual_iou < 0.5`)의 25% 이상이면 지지. 10-25%면 약한 지지. 10% 미만이면 기각. | 2026-06-19 | ✅ supported | Failed rows 73개 중 selection/calibration 22개 = **30.1%**. 산출물: `p1e_vala_native_2d_oracle_official_actual_*.csv` |
| **RF-V2** | B | VALA robust-gate는 mean/non-gated 대비 overgrowth를 줄이지만, VALA-native selection/calibration failure를 완전히 제거하지 못한다. | 같은 RGB 3DGS/source feature/eval에서 robust-gate가 mean보다 precision 또는 area-ratio를 개선하고, P1V-M1의 `vala_selection_fail`/`vala_calibration_fail` subset 평균 IoU gap이 0.10 이상 남으면 지지. | 2026-06-19 | ❌ unsupported / partially inverted | mean/non-gated가 actual IoU에서 ramen **0.5936 > 0.5445**, waldo **0.6436 > 0.4702**로 robust-gate보다 높음. robust는 ramen area overgrowth만 줄였지만 precision/recall 개선 없음; waldo는 mean이 precision/recall/area-ratio 모두 우세. 단 recoverable failure total gap은 양쪽 조건 모두 0.29-0.39로 남음. 산출물: `p1e_vala_rf_v2_robust_vs_mean_*.csv`, `vala_rf_v2_robust_vs_mean.md` |
| **RF-V3** | A | common harness로 VALA/THGS/ReLaGS final mask를 재채점하면 native evaluator 숫자와 ranking/scene gap 해석이 달라진다. | 동일 GT·prompt·frame·aggregation·threshold policy로 산출한 4-scene 표가 native table과 scene 평균 0.03 mIoU 이상 차이나거나 순위가 바뀌면 지지. | 2026-06-19 | ⏳ pending | — |

> `pending` 항목은 초안 — 각 stage 설계 시 정밀화하고 `고정일`을 박은 뒤 드라이버를 돌린다. `supported`/`unsupported` 항목은 이미 실행된 preregistered 결과다.

## Retrospective records (RF 아님)

| ID | Lane | 실행일 | 내용 | 산출물 | 해석 제한 |
|---|---|---|---|---|---|
| **P1V-D1** | D | 2026-06-19 | VALA job84 official 2D saved `chosen_*.png`를 LERF-OVS GT로 per-prompt 재채점하고 THGS/ReLaGS class를 join | `output/diagnostics/p1e_vala_official_2d_prompt_detail.csv`, `output/diagnostics/p1e_vala_official_2d_prompt_agg.csv` | descriptive audit만 가능. THGS class 전이이므로 leaderboard/mechanism 주장 금지 |

## 비고

- 이 표는 **stage 상태**를 담지 않는다 (그건 [README.md](README.md)). 여기는 **예측·기준·판정**만.
- 본선 R-ID 시리즈(R1~R13)와는 별개 — 필요 시 [../experiments/README.md](../experiments/README.md) 및 [../strategy/](../strategy/) 참조.
