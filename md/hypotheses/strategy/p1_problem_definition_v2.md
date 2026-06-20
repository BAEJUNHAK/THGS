# P1 problem definition v2

> 생성: 2026-06-19  
> 목적: THGS/ReLaGS/VALA 분석을 거친 뒤, P1의 논문용 문제정의를 "noisy average"보다 더 구체적인 형태로 고정한다.

---

## 1. 한 문장 결론

VALA partially mitigates the symptom identified as noisy aggregation, but does not eliminate the underlying evidence-regime collapse: some failures lack recoverable representation, while others contain recoverable evidence missed by level/threshold selection.

한국어로 쓰면:

> VALA는 noisy aggregation/visibility leakage라는 증상을 일부 완화하지만, query마다 필요한 evidence regime을 고르는 문제는 해결하지 못한다. 어떤 실패는 feature family 안에 답이 없고, 어떤 실패는 답이 있는데 level/threshold selection이 놓친다.

---

## 2. 기존 문제정의와 우리의 문제정의

기존 paper들이 이미 강하게 말한 문제:

> 여러 view의 2D language feature를 평균하면 occlusion/background leakage와 view drift가 섞여 noisy representative feature가 된다.

VALA는 이 문제를 정면으로 다룬다.

- visibility-aware gate로 실제 ray contribution이 약한 Gaussian-view assignment를 줄인다.
- cosine-space streaming weighted geometric median으로 view drift에 robust한 representative language feature를 만든다.

그래서 우리가 "average가 noisy하다"만 말하면 novelty가 약하다. P1은 다음 질문으로 좁혀야 한다.

> 좋은 representative feature 하나를 만드는 것만으로 충분한가? 아니면 query마다 consensus evidence를 따라야 할 때와 minority/visible evidence를 복구해야 할 때가 달라서, method가 그 regime을 query-conditioned로 선택해야 하는가?

이것을 **query-conditioned evidence-regime collapse**라고 부른다.

---

## 3. 문제의 구조

한 query `q`에 대해 method는 보통 다음 후보들을 가진다.

- 여러 view에서 온 2D language evidence
- 여러 hierarchy/feature level
- 여러 threshold 또는 calibration choice
- foreground/background/occlusion이 섞인 Gaussian 또는 superpoint evidence

실패는 네 종류로 분해된다.

| class | 뜻 | 논문에서의 역할 |
|---|---|---|
| actual ok | native policy로 해결됨 | baseline success |
| selection fail | 좋은 level/evidence 후보가 있는데 score가 못 고름 | query-time regime selection 문제 |
| calibration fail | 후보는 있지만 threshold/calibration이 안 맞음 | guard/calibration 문제 |
| representation fail | 후보 family 안에 답이 약함 | feature preservation 또는 object discovery 문제 |

P1의 핵심은 selection/calibration과 representation을 섞지 않는 것이다. "VALA가 못 한다"가 아니라, **VALA가 고친 축 이후에도 다른 failure locus가 남는다**고 말해야 한다.

---

## 4. 증거 1: Waldo paper protocol forensic

문서: [../fairness/vala_paper_protocol_forensics.md](../fairness/vala_paper_protocol_forensics.md)

Waldo Kitchen 2D paper number `0.651`은 public fixed actual로 설명되지 않았다.

| protocol | mIoU |
|---|---:|
| official saved actual @0.5, official train/official SAM | 0.5412 |
| official saved actual @0.4, official train/official SAM | 0.5575 |
| scene-best fixed threshold | 0.5576 |
| VALA dynamic threshold | 0.3433 |
| per-row level oracle @0.5 | 0.5737 |
| per-row threshold+level oracle | 0.6865 |

P1V-M1에서 쓰는 `refersplat_3dgs_valafeat_full` 조건은 threshold+level oracle 0.6538로 paper number와 가깝지만, 이것은 GT oracle이므로 actual 성능으로 쓰면 안 된다.

따라서 안전한 해석:

- Waldo paper number는 public artifact/protocol mismatch flag를 단다.
- "paper가 oracle을 썼다"고 단정하지 않는다.
- leaderboard에는 actual output만 쓴다.
- oracle은 feature family 안의 recoverable evidence를 보는 diagnostic upper bound로만 쓴다.

---

## 5. 증거 2: VALA-native oracle gap

문서: [../fairness/p1_vala_protocol.md](../fairness/p1_vala_protocol.md)

P1V-M1은 VALA 자체 feature map에서 actual chosen output과 level/threshold oracle을 비교했다.

| scene | actual | level oracle | threshold oracle | level gap | threshold gap |
|---|---:|---:|---:|---:|---:|
| figurines | 0.5589 | 0.5762 | 0.6619 | 0.0174 | 0.0857 |
| ramen | 0.5445 | 0.5649 | 0.6391 | 0.0204 | 0.0742 |
| teatime | 0.6616 | 0.6846 | 0.7646 | 0.0230 | 0.0800 |
| waldo_kitchen | 0.4702 | 0.5323 | 0.6538 | 0.0623 | 0.1215 |

RF-V1 판정:

- failed rows(`actual_iou < 0.5`) 73개
- selection/calibration recoverable rows 22개
- 비율 **30.1%**
- 사전 기준 25% 초과라 supported

해석:

> VALA output의 모든 실패가 representation 부재는 아니다. 일부는 같은 VALA feature map 안에 recoverable 후보가 있는데 actual level/threshold policy가 놓친다.

---

## 6. 증거 3: THGS/ReLaGS descriptive bridge

문서: [../fairness/vala_native_crosstab.md](../fairness/vala_native_crosstab.md)

THGS/ReLaGS class를 VALA에 붙인 결과는 descriptive bridge로만 해석한다. 그래도 P1 방향을 잡는 데 유용하다.

Prompt-aggregated THGS class 기준:

| THGS class | VALA actual ok | selection fail | calibration fail | representation fail |
|---|---:|---:|---:|---:|
| easy | 28/36 = 77.8% | 3/36 = 8.3% | 1/36 = 2.8% | 4/36 = 11.1% |
| phantom | 6/21 = 28.6% | 3/21 = 14.3% | 4/21 = 19.0% | 8/21 = 38.1% |
| rare | 2/5 = 40.0% | 1/5 = 20.0% | 1/5 = 20.0% | 1/5 = 20.0% |
| real | 0/5 = 0.0% | 0/5 = 0.0% | 0/5 = 0.0% | 5/5 = 100.0% |

해석 제한:

- "VALA가 THGS phantom을 못 고쳤다"를 mechanism claim으로 쓰면 안 된다.
- 하지만 easy는 대체로 풀리고 phantom/real은 representation과 selection/calibration으로 갈라진다는 관찰은 P1 문제정의의 방향과 맞다.

---

## 7. 증거 4: RF-V2 robust-gate vs mean/non-gated

문서: [../fairness/vala_rf_v2_robust_vs_mean.md](../fairness/vala_rf_v2_robust_vs_mean.md)

같은 VALA pipeline에서 language aggregation checkpoint만 바꿨다.

| scene | condition | actual | threshold oracle | total oracle gap |
|---|---|---:|---:|---:|
| ramen | mean/non-gated | **0.5936** | 0.6788 | 0.0852 |
| ramen | robust-gate | 0.5445 | 0.6391 | 0.0946 |
| waldo_kitchen | mean/non-gated | **0.6436** | 0.7630 | 0.1194 |
| waldo_kitchen | robust-gate | 0.4702 | 0.6538 | 0.1837 |

RF-V2 판정:

- unsupported / partially inverted.
- mean/non-gated가 ramen과 waldo_kitchen 모두에서 actual IoU가 높다.
- robust-gate는 ramen에서 area overgrowth를 줄이지만 recall 손실로 IoU가 낮다.
- waldo_kitchen에서는 mean/non-gated가 precision/recall/area-ratio 모두 좋다.
- 그래도 recoverable selection/calibration failure의 total gap은 양쪽 조건 모두 0.29-0.39로 남는다.

이 결과는 P1을 이렇게 고정한다.

> 특정 robust aggregation rule 하나가 universal fix가 아니다. Aggregation choice는 evidence regime을 바꾸지만, query마다 어떤 regime을 선택해야 하는지는 여전히 별도의 문제다.

---

## 8. 가능한 주장

논문에서 가능한 주장:

1. VALA는 visibility leakage와 multi-view drift를 올바르게 문제화했고, noisy aggregation의 일부 증상을 완화한다.
2. Public fixed actual과 oracle diagnostic을 분리하면, VALA feature family 안에도 actual policy가 놓치는 recoverable evidence가 남는다.
3. THGS/ReLaGS에서 어려웠던 query family는 VALA에서도 uniform하게 해결되지 않고 representation/selection/calibration으로 갈라진다. 단 이것은 descriptive bridge다.
4. 같은 VALA pipeline의 one-factor ablation에서도 aggregation rule 변화가 output regime을 크게 바꾸지만, recoverable selection/calibration gap은 남는다.
5. 따라서 P1의 novelty는 "더 좋은 average"가 아니라 **query-conditioned guarded evidence selection**에 있다.

---

## 9. 불가능한 주장

논문에서 피해야 할 주장:

1. "VALA paper used oracle evaluation."  
   현재 증거는 public artifact/protocol mismatch이지 oracle 사용 단정이 아니다.

2. "VALA가 phantom을 못 고쳤다"를 leaderboard처럼 말하기.  
   THGS/ReLaGS class transfer는 descriptive bridge다.

3. "robust-gate가 mean보다 항상 낫다."  
   RF-V2에서 mean/non-gated가 actual IoU, precision, recall을 더 잘 냈다.

4. "P1은 aggregation noise를 처음 발견했다."  
   VALA 등 기존 paper가 이미 이 문제를 정의했다. P1은 그보다 더 구체적인 evidence-regime problem을 정의해야 한다.

---

## 10. 논문용 위치

VALA와의 관계는 공격이 아니라 refinement로 써야 한다.

> VALA correctly targets visibility leakage and multi-view drift, and its robust aggregation is a principled improvement over naive feature lifting. Our evidence shows that this is not the end of the problem: even after forming a representative feature, the system must decide which evidence regime a query requires. Some queries need consensus preservation; others need minority visible evidence or stricter calibration. This query-conditioned regime decision is the axis P1 isolates.

한 줄로 줄이면:

> P1 is not a better average; P1 is a framework for deciding when averaging is the wrong evidence regime for the query.
