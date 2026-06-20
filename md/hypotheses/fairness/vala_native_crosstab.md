# VALA-native class cross-tab with THGS/ReLaGS classes

> 생성: 2026-06-19  
> 산출물:
> - `output/diagnostics/p1e_vala_native_class_crosstab_rows.csv`
> - `output/diagnostics/p1e_vala_native_class_crosstab_prompts.csv`
>
> 해석 제한: THGS/ReLaGS class는 VALA pipeline에서 나온 taxonomy가 아니므로 **descriptive bridge**로만 쓴다. Mechanism claim은 VALA 내부 ablation 또는 VALA-native oracle/actual gap에서만 주장한다.

---

## 1. 질문

P1V-M1은 VALA 자체 feature map에서 actual output과 level/threshold oracle 사이 gap을 만들었다. 이 문서는 그 VALA-native class가 THGS/ReLaGS에서 보던 `easy/phantom/rare/real` class와 어떻게 겹치는지 확인한다.

목적은 "VALA가 THGS phantom을 못 고쳤다"를 직접 주장하는 것이 아니다. 목적은 다음 descriptive bridge다.

> THGS/ReLaGS에서 어려웠던 prompt family가 VALA에서도 representation/selection/calibration 실패로 자주 남는가?

---

## 2. 방법

입력:

- `output/diagnostics/p1e_vala_native_2d_oracle_official_actual_detail.csv`

단위 두 개를 모두 계산했다.

| unit | 의미 |
|---|---|
| row | `(scene, frame, prompt)` 단위. P1V-M1과 같은 208 rows. |
| prompt | scene별 prompt로 평균한 단위. frame 중복의 영향을 줄인다. |

VALA-native class:

| class | 조건 |
|---|---|
| `vala_actual_ok` | actual IoU >= 0.5 |
| `vala_selection_fail` | actual < 0.5, fixed-threshold best level oracle >= 0.5 |
| `vala_calibration_fail` | level oracle < 0.5, threshold oracle >= 0.5 |
| `vala_representation_fail` | threshold oracle < 0.5 |

---

## 3. Overall row-level 결과

THGS class 기준:

| THGS class | VALA actual ok | selection fail | calibration fail | representation fail |
|---|---:|---:|---:|---:|
| easy | 104/124 = **83.9%** | 2/124 = 1.6% | 8/124 = 6.5% | 10/124 = 8.1% |
| phantom | 22/61 = **36.1%** | 3/61 = 4.9% | 8/61 = 13.1% | 28/61 = **45.9%** |
| rare | 9/13 = **69.2%** | 1/13 = 7.7% | 0/13 = 0.0% | 3/13 = 23.1% |
| real | 0/10 = 0.0% | 0/10 = 0.0% | 0/10 = 0.0% | 10/10 = **100.0%** |

ReLaGS class 기준도 거의 같은 패턴이다.

| ReLaGS class | VALA actual ok | selection fail | calibration fail | representation fail |
|---|---:|---:|---:|---:|
| easy | 103/124 = **83.1%** | 2/124 = 1.6% | 9/124 = 7.3% | 10/124 = 8.1% |
| phantom | 23/61 = **37.7%** | 3/61 = 4.9% | 7/61 = 11.5% | 28/61 = **45.9%** |
| rare | 9/17 = **52.9%** | 1/17 = 5.9% | 0/17 = 0.0% | 7/17 = 41.2% |
| real | 0/6 = 0.0% | 0/6 = 0.0% | 0/6 = 0.0% | 6/6 = **100.0%** |

핵심 관찰:

- `easy`는 대부분 VALA actual로도 해결된다.
- `phantom`은 actual ok가 36-38%뿐이고, representation fail이 45.9%로 크다.
- 하지만 `phantom` 안에서도 selection/calibration fail이 16-18% 정도 존재한다. 즉 일부는 representation 자체가 없기보다 level/threshold 선택이 놓친 recoverable evidence다.
- `real`은 row-level에서 전부 representation fail로 떨어진다. 이것은 작은 object/희귀 visible evidence가 VALA feature map family 안에서도 약하다는 신호지만, class transfer이므로 mechanism으로 단정하지 않는다.

---

## 4. Prompt-aggregated 결과

Prompt 단위로 평균해도 같은 방향이 유지된다.

THGS class 기준:

| THGS class | VALA actual ok | selection fail | calibration fail | representation fail |
|---|---:|---:|---:|---:|
| easy | 28/36 = **77.8%** | 3/36 = 8.3% | 1/36 = 2.8% | 4/36 = 11.1% |
| phantom | 6/21 = **28.6%** | 3/21 = 14.3% | 4/21 = 19.0% | 8/21 = **38.1%** |
| rare | 2/5 = 40.0% | 1/5 = 20.0% | 1/5 = 20.0% | 1/5 = 20.0% |
| real | 0/5 = 0.0% | 0/5 = 0.0% | 0/5 = 0.0% | 5/5 = **100.0%** |

ReLaGS class 기준:

| ReLaGS class | VALA actual ok | selection fail | calibration fail | representation fail |
|---|---:|---:|---:|---:|
| easy | 28/37 = **75.7%** | 3/37 = 8.1% | 2/37 = 5.4% | 4/37 = 10.8% |
| phantom | 6/20 = **30.0%** | 3/20 = 15.0% | 3/20 = 15.0% | 8/20 = **40.0%** |
| rare | 2/6 = 33.3% | 1/6 = 16.7% | 1/6 = 16.7% | 2/6 = 33.3% |
| real | 0/4 = 0.0% | 0/4 = 0.0% | 0/4 = 0.0% | 4/4 = **100.0%** |

Prompt 평균에서는 `phantom`의 selection+calibration 비중이 더 커진다.

- THGS phantom: 7/21 = **33.3%**가 selection/calibration.
- ReLaGS phantom: 6/20 = **30.0%**가 selection/calibration.

이는 P1의 핵심 문장과 잘 맞는다. VALA에는 두 종류의 잔여 실패가 공존한다.

1. feature/render mask family 안에 충분한 후보가 없는 representation failure
2. 후보는 있는데 level/threshold policy가 query-time에 놓치는 selection/calibration failure

---

## 5. P1에 주는 의미

이 cross-tab만으로 mechanism을 주장하면 안 된다. 그러나 P1 문제정의를 좁히는 데는 중요하다.

가능한 주장:

- VALA는 THGS/ReLaGS에서 쉬운 family를 대체로 안정화한다.
- THGS/ReLaGS에서 어려웠던 family는 VALA에서도 균일하게 해결되지 않고, representation failure와 recoverable selection/calibration failure로 갈라진다.
- 따라서 P1은 "noisy average를 고치면 끝"이 아니라, **query마다 recoverable evidence가 있는지, 있다면 어떤 level/threshold/evidence regime을 선택해야 하는지**를 묻는 문제로 가야 한다.

금지할 주장:

- "VALA가 THGS phantom을 못 고쳤다"를 leaderboard 성능처럼 말하면 안 된다.
- "phantom failure의 원인은 VALA robust aggregation이다"라고 말하면 안 된다. 그건 RF-V2 ablation에서만 가능하다.

요약하면, 이 표는 P1V-M1의 VALA-native 결과와 THGS/ReLaGS taxonomy를 연결해 주는 descriptive bridge다. Mechanism claim은 다음 단계인 robust-gate vs mean/non-gated VALA ablation으로 넘긴다.
