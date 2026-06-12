# Stage 3.4 — 잔여 원인 전수 분해 (method 설계 전 최종 진단)

> 완료 (2026-06-11). 대다수 원인 (소수파 매장) 으로 설명 안 되던 잔여 케이스 전부에 증거 기반 원인 라벨을 부여 — **36 케이스 (phantom 17 + other 실패 9 + easy 역행 10) 의 완전한 taxonomy** 와 Stage 4 method 요구사항 명세가 산출물.
>
> 핵심 finding 한 줄: **"잔여를 해부하니 새 원인 2개가 나왔다 — ① few-view opportunist (view 2-11개짜리 미세 SP 는 평균 희석이 없어 mean 게임에서 구조적으로 유리; spoon/cabinet 의 진범) ② 벤치마크 자체의 multi-instance 미표기 의심 (7 prompt, 46건). 그리고 Stage 1 의 'D2.real = encoder 한계' 분류가 사실상 해체됐다 (5건 중 4건이 query-aware 로 회복 = 극단적 dilution 이었음)."**

---

## 0. Question 과 사전 판정 (B1 §6.6)

- **R6**: 36 케이스 전원 증거 기반 라벨 (unknown ≤ 2) → **PASS (unknown 2: bowl, napkin)**
- **R7**: multi-instance 의심 ≥3 → **발동 (7 prompt, 46건) — Stage 4 에 평가 보정 포함 필수**
- **R8**: 2B 의 encoder-측 분류 재감사 → **4/4 전원 재분류** (encoder limit 아님)

---

## 1. 3.4-C — NAG 친족 감사 ([stage3_4_nag_kinship.py](../../../scripts/stage3_4_nag_kinship.py))

17 phantom 의 wrong-top1 vs oracle 관계 (gaussian 집합 포함률 기반):

| 관계 | n | 사례 | 함의 |
|---|---|---|---|
| **child_of_b** (wrong ⊂ oracle) | 3 | jake (사실상 same, 1.00/0.98), rubber duck (모자 부분), tesla (1% 조각!) | **parent-union 1줄로 무해화** |
| **parent_of_b** (oracle ⊂ wrong) | 1 | sink (over-union) | 계층 채점 |
| **sibling** (같은 부모) | 2 | onion↔egg (같은 그릇 안!), bear nose | 공간 분리/negative contrast |
| unrelated | 11 | 진짜 외부 confuser | hybrid/가드 |

흥미: old camera 의 wrong 은 ref-frame 에서 precision 0.98 인데 NAG 상 **unrelated** — 3D 가우시안은 분리됐는데 2D 투영이 겹침 (깊이 방향 중첩).

---

## 2. 3.4-A — 패자부활 실패자 해부 ([stage3_4_loser_anatomy.py](../../../scripts/stage3_4_loser_anatomy.py))

### ★ 신규 원인 ①: few-view opportunist

spoon/cabinet (never_good) 을 mean 체제에서 이기는 승자들의 시그니처:

| | spoon 승자 10 | cabinet 승자 10 |
|---|---|---|
| n_valid_views | **4~11** | **2~6 (!)** |
| coherence | 0.84~0.95 | **0.92~0.99** |
| GT IoU | 전부 0.00 | 전부 0.00 |

**메커니즘**: view 2개짜리 SP 는 "평균" 이 곧 그 2 view 라 **희석이 일어나지 않는다** — 순수한 (그리고 우연히 prompt 같은) feature 를 유지. 잘 관찰된 진짜 물체 (oracle spoon nv≈54) 는 다수 평범 view 로 희석. **visibility-weighted mean 은 저관찰 SP 에게 구조적으로 유리한 게임** — B1 메커니즘의 새 따름정리. → 처방: **min-evidence prior** (관측량 하한/가중).

### sake cup 의 재분류: 진단 도구의 맹점 발견

sake cup [mean] 3등 SP = **gt_iou 0.93, recall 1.00 의 사실상 진짜 물체** (oracle 과 다른 SP). 즉 baseline top-3 union 은 이미 정답을 포함 → "oracle rank 10 = phantom" 분류는 **oracle 단일 SP 의 rank 만 보는 진단의 착시**. (다른 GT-커버 SP 의 존재를 무시.) → rank 기반 분류의 한계를 명시, mask 기준이 항상 우선.

---

## 3. 3.4-B — easy 역행 10건 해부 ([stage3_4_easy_regression.py](../../../scripts/stage3_4_easy_regression.py))

| 정체 | n | 의미 |
|---|---|---|
| **fake regression (kin_same)** | **3** (rubics cube, apple, yellow pouf) | top-k 의 새 1등이 **같은 물체의 다른 계층 항목** (gt_iou 0.92-0.98) — rank 통계의 착시, mask 무해 |
| lucky_view_jump victim | 7 | 외부 SP 가 운 좋은 5 view 로 점프해 승리 |

### ★ 가드 신호 (Stage 4 hybrid 의 설계 데이터)

- **g1 (lucky-view jump)**: 역행 승자의 (top5점수 − mean점수) median **+0.17**, 최대 +0.52 — "mean 에서 평범했다가 top-k 로 점프한 SP 를 의심하라"
- **g2**: 역행 케이스의 **60% 에서 oracle 이 mean 1등** — "mean 이 이미 확신하면 top-k 로 뒤집지 마라"
- g4: 역행 승자의 90% 가 GT 무관 외부 — 가드 필요성 확증

---

## 4. 3.4-D — multi-instance 감사 ([stage3_4_multi_instance.py](../../../scripts/stage3_4_multi_instance.py))

generic prompt 의 top-5 체제 상위 승자 중 "전 frame GT 와 무관 + 관측 충분 (nv≥10)" = 미표기 동종 인스턴스 의심:

| prompt | 의심 건수 |
|---|---|
| waldo spoon / sake cup | 10 / 10 |
| waldo cabinet | 9 |
| waldo sink | 8 |
| ramen bowl | 6 |
| ramen plate / napkin / waldo plate | 2 / 1 / 1 |

**R7 발동 (7 prompt, 46건)** — 주방/식탁 장면의 spoon·cabinet·bowl 류는 실물이 여러 개인데 GT 가 일부만 표기했을 개연성. 시각 증거: `plots/stage3_4_multi_instance_montage.png` (자동 판정은 보수적 — 최종 확인은 montage 육안). **함의**: ① 이들 prompt 의 "실패" 일부는 method 가 아니라 **벤치마크의 한계** ② Stage 4 평가에 보정 (의심 케이스 분리 보고) 필수 ③ paper 의 정직성 포인트.

---

## 5. 3.4-E — 종합 taxonomy ([stage3_4_taxonomy.csv](../../../output/diagnostics/stage3_4_taxonomy.csv), 36 케이스)

### Phantom 17 의 최종 원인 분포

| 원인 | n | 케이스 | 처방 |
|---|---|---|---|
| **minority_dilution(+partial)** | **6** | pikachu, pumpkin, hand, plate, ottolenghi (+old camera) | hybrid |
| **granularity** (child 3 + parent 1) | **4** | jake, rubber duck, tesla, sink | parent_union |
| sibling_confusion | 2 | onion↔egg, bear nose | negative contrast/공간 |
| **multi_instance_suspect** | 2 | spoon, cabinet | 평가 보정 + min-evidence |
| coherent_confuser | 1 | sake cup (+rank 진단 착시) | hybrid+negative |
| unknown | 2 | bowl (8→8, IoU 정체), napkin (이미 0.45 — 경계) | tbd |

IoU 증거 (baseline→query-top5): pikachu 0.00→**0.91**, ottolenghi 0.00→**0.94**, pumpkin 0.00→0.79, plate 0.00→0.74 / 손실: jake 0.55→0.32, old camera 0.39→0.01, sake cup 0.17→0.00 — **케이스별로 최적 체제가 다름** = hybrid 의 필요성을 케이스 수준에서 재확인.

### ★ R8 + Stage 1 프레임 수정 (중요)

- 2B 의 "mean_dilution = encoder 측" **4/4 전원 재분류**: old camera 130→17, pumpkin 52→1, napkin 7→5, ottolenghi 66→3 — encoder 한계가 아니라 **극단적 dilution**.
- Stage 1 의 **D2.real 5건 중 4건 회복** (miffy 도 rank≤3!): proper noun 조차 encoder 는 알고 있었고 dilution 이 가렸던 것. **encoder-limit 최종 잔여 = waldo plate, pot, yellow desk (3건, method-specific 군)** — "encoder 한계 7.5%" 는 과대 추정이었음.

### Stage 4 method 요구사항 (fix_signal 수요 집계)

| 신호 | 수요 | 근거 케이스 |
|---|---|---|
| **hybrid (α·mean + (1−α)·top-k)** | 12+1 | dilution 계열 전부 |
| **hybrid 가드** (g1 gap, g2 mean-confidence) | 7 | lucky_view_jump 차단 |
| **parent_union** (계층 인지 채점/union) | 4 | granularity 4건 |
| **min_evidence prior** (저관찰 SP 패널티) | 2+ | few-view opportunist |
| **zero_norm 필터** | 유령 150 | 전 scene |
| negative_contrast/공간 분리 | 3 | sibling/instance confusion |
| **평가 보정** (multi-instance 분리 보고) | 7 prompt | R7 |
| 해당 없음 (rank 착시 3 / encoder 3 / unknown 2) | 8 | 정직한 잔여 |

---

## 6. Decision — Stage 4 (이제 진짜 설계 가능)

Method = **가드된 hybrid 채점 + 계층 union + 사전 필터 2종**, 평가 = **multi-instance 분리 보고**:

```
score(SP, q) = α·canon(mean_feat) + (1−α)·canon(top-k_q feat)     [hybrid, α≈0.3]
  단, ① n_valid_views < τ_v 인 SP 는 top-k 항 비활성 (min-evidence)
      ② zero-norm SP 제외
      ③ (top-k 항 − mean 항) > τ_g 인 비-oracle 후보는 top-k 항 감쇠 (g1 가드)
선택:  top-3 에 뽑힌 SP 의 NAG 부모/자식이 인접 점수면 union (granularity)
평가:  67 전체 + multi-instance 의심 7 prompt 분리 트랙, scene-split calibration
```

기대 커버리지 (taxonomy 기준): 회복 가능 13~17 + 역행 차단 7+3 / 원리적 잔여 8 (encoder 3, unknown 2, eval 보정 대상은 별도 트랙).

---

## 7. 산출물 / 관련 문서

- CSV: stage3_4_kinship / losers / easy_regression / multi_instance / **taxonomy** (5종)
- Plots: loser_winners / easyreg / multi_instance montage (3종)
- Scripts: stage3_4_common.py + 5종
- 관련: [stage3_3_definitive.md](stage3_3_definitive.md) / [../extended_failure_hypotheses.md](../extended_failure_hypotheses.md) B1 §6.6
