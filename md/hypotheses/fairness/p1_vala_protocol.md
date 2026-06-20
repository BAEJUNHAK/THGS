# P1-VALA fairness protocol

> 생성 2026-06-19. 목적: VALA를 THGS/ReLaGS와 비교할 때 claim을 섞지 않고, P1 문제정의를 논문 novelty가 생기는 수준까지 좁히기 위한 실행 문서.
>
> 상위 문서: [protocol.md](protocol.md), [README.md](README.md), [../strategy/vala문제.md](../strategy/vala문제.md), [../strategy/p1e_external_code_study_plan.md](../strategy/p1e_external_code_study_plan.md).

---

## 0. VALA가 실제로 고친 문제

VALA 논문의 문제정의는 두 가지다.

1. **visibility leakage**: 한 pixel ray를 지나는 foreground/background/occluded Gaussian들이 같은 2D language feature를 받는다.
2. **multi-view drift**: view마다 CLIP/SAM language embedding이 흔들려 평균 feature가 noisy해진다.

VALA의 처방도 이에 맞춰져 있다.

- ray marginal contribution 기반 **visibility-aware gate**로 보이는 Gaussian에만 feature를 남긴다.
- cosine space **streaming weighted geometric median**으로 view별 noisy feature를 robust하게 합친다.

따라서 VALA는 "평균이 noisy하다"는 문제를 실제 method로 고친 강한 경쟁자다. 하지만 P1에서 우리가 물어야 할 질문은 여기서 한 단계 더 좁다.

> **P1 재정의**: 좋은 representative feature 하나를 만드는 것만으로 충분한가? 아니면 query마다 consensus evidence를 따라야 할 때와 minority evidence를 복구해야 할 때가 달라서, method가 그 regime을 query-conditioned로 선택해야 하는가?

이 차이가 없으면 우리는 VALA를 다시 구현한 논문이 된다. 이 차이를 보이면 "aggregation noise"보다 구체적인 문제정의를 만든다.

---

## 1. Claim lane을 셋으로 분리한다

| Lane | 이름 | 허용되는 주장 | 필요한 통제 | VALA에서의 예 |
|---|---|---|---|---|
| **D** | Descriptive/native audit | "VALA output을 우리 taxonomy로 보면 이런 패턴이 있다" | 평가 코드는 기록하되 완전 통제는 아님 | job84 official 2D mask를 THGS/ReLaGS class로 split |
| **A** | Leaderboard | "같은 벤치마크 자에서 누가 이긴다" | GT, prompt, frame, metric, aggregation, threshold policy 동일 | THGS/ReLaGS/VALA final mask를 common harness로 채점 |
| **B** | Mechanism | "이 실패의 원인은 aggregation/gating/selection이다" | 한 요소만 바꾸고 나머지 모두 동일 | VALA 안에서 robust-gate vs mean/non-gated ablation |

중요한 금지:

- Lane D 결과를 Lane A 리더보드로 쓰지 않는다.
- Lane D 결과를 Lane B 메커니즘 증명으로 쓰지 않는다.
- THGS/ReLaGS의 `phantom/easy` 라벨을 VALA에 붙이는 것은 **descriptive bridge**일 뿐이다. VALA-native oracle/actual class를 만들기 전에는 "VALA가 phantom을 고쳤다/못 고쳤다"라고 쓰지 않는다.

---

## 2. 공정성 위반을 피하는 방법

VALA와 THGS/ReLaGS는 substrate와 method가 다르다.

- THGS/ReLaGS: 2DGS + superpoint hierarchy + level/rank retrieval.
- VALA: 3DGS + per-Gaussian language feature + level별 Gaussian relevance mask.

따라서 다음 비교는 불공정하거나 claim이 약하다.

| 비교 | 왜 약한가 | 허용되는 해석 |
|---|---|---|
| VALA native mIoU vs THGS native mIoU | eval 코드, threshold, aggregation 순서가 다름 | 재현성/참고 숫자 |
| VALA mask를 THGS phantom class로 split | class가 다른 pipeline에서 나온 라벨임 | hypothesis generator |
| VALA robust output vs THGS mean-family dump | geometry와 query selection이 같이 바뀜 | family-level analogy |

공정한 비교는 claim별로 따로 만든다.

- **Leaderboard**: final mask만 모아서 common harness로 채점한다. method 차이는 비교 대상이므로 허용한다.
- **Mechanism**: 같은 VALA pipeline 안에서 checkpoint/feature assignment만 바꾼다. 또는 같은 THGS/ReLaGS dump 위에서 aggregation rule만 바꾸는 family-level proxy를 별도로 둔다.
- **Bridge**: THGS/ReLaGS class split은 "이 class가 VALA에서도 유사하게 취약한가?"를 묻는 관찰 실험으로만 사용한다.

---

## 3. 현재 확보된 VALA 산출물

### 3.1 재현성 상태

[../strategy/vala문제.md](../strategy/vala문제.md)의 최신 판정:

- Ramen 3D target은 0.604가 아니라 **0.4541**이다. `VALA official feature + ReferSplat RGB` 조건에서 **0.4589**로 재현권이다.
- 현재 재현성 갭의 중심은 **Waldo Kitchen**이다. official RGB train + official features도 3D **0.4904 vs paper 0.5571**, official 2D eval **0.5412@0.5 / 0.5575@0.4 vs paper 0.651**이다.

이 상태에서는 VALA를 "완벽 재현된 leaderboard baseline"으로 선언하면 안 된다. 대신:

- Ramen/Teatime/Figurines는 실험적 진단에 사용 가능.
- Waldo는 "public artifact reproducibility gap" flag를 달고 분석한다.

### 3.2 job84 official 2D prompt audit

이미 완료된 retrospective 산출물:

- summary: `output/diagnostics/p1e_vala_official_2d_eval_all_84.csv`
- detail: `output/diagnostics/p1e_vala_official_2d_prompt_detail.csv`
- aggregate: `output/diagnostics/p1e_vala_official_2d_prompt_agg.csv`
- script: `scripts/p1e_vala_official_2d_prompt_table.py`

`refersplat_3dgs_valafeat_full`의 official 2D flat mIoU:

| scene | @0.5 | @0.4 |
|---|---:|---:|
| figurines | 0.5589 | 0.5616 |
| ramen | 0.5445 | 0.5007 |
| teatime | 0.6616 | 0.6979 |
| waldo_kitchen | 0.4702 | 0.4839 |

THGS class로 split하면 scene-dependent 패턴이 나온다. 예를 들어 @0.5 per-prompt 기준:

| scene | easy | phantom | rare | real |
|---|---:|---:|---:|---:|
| figurines | 0.7720 | 0.4021 | 0.2794 | 0.0649 |
| ramen | 0.5843 | 0.2973 | 0.8474 | - |
| teatime | 0.6882 | 0.4493 | - | 0.0648 |
| waldo_kitchen | 0.5340 | 0.4580 | 0.4291 | 0.2862 |

해석:

- "VALA가 phantom을 전부 고쳤다"는 단순 서사는 아니다.
- 하지만 이 표는 THGS class 전이이므로 mechanism 결론은 아니다.
- 다음 단계는 VALA-native oracle/actual class를 만드는 것이다.

---

## 4. P1-VALA 실험 스택

| ID | Lane | 질문 | 상태 | 산출물 |
|---|---|---|---|---|
| **P1V-D1** | D | VALA official 2D mask를 우리 prompt/class taxonomy로 보면 어떤가? | 완료, retrospective | `p1e_vala_official_2d_prompt_detail/agg.csv` |
| **P1V-M1** | B precursor | VALA-native level/threshold oracle과 actual chosen mask 사이 gap이 있는가? | 완료. RF-V1 지지 | `p1e_vala_native_2d_oracle_official_actual_detail/agg.csv` |
| **P1V-M2** | B | 같은 VALA model에서 robust-gate feature와 mean/non-gated feature가 어떤 failure를 다르게 만드는가? | 완료. RF-V2 unsupported / partially inverted | `p1e_vala_rf_v2_robust_vs_mean_detail/agg.csv`, `vala_rf_v2_robust_vs_mean.md` |
| **P1V-L1** | A | common harness에서 THGS/ReLaGS/VALA final mask ranking은 무엇인가? | 설계 전 | `fairness/leaderboard` 산출물 |

---

## 5. P1V-M1: VALA-native oracle/actual table

목적: THGS class를 빌리지 않고 VALA 자체의 failure 위치를 정의한다.

입력:

- `external_methods/VALA/output/official_2d_eval_all_84/feat_dir/<case>/<scene>_{1,2,3}/train/ours_None/renders_npy/*.npy`
- `data/lerf_ovs/label/<scene>/frame_*.json`
- VALA scorer: `external_methods/VALA/eval/openclip_encoder.py`

driver:

- `scripts/p1e_vala_native_2d_oracle.py`
- `scripts/p1e_vala_reconcile_oracle_actual.py` (actual은 official saved `chosen_*.png` 점수로 교정)
- dry-run: 2026-06-19, `refersplat_3dgs_valafeat_full__figurines`, 1 frame, 1 prompt, CPU, smoothing off.
- dry-run result: `green apple` / `frame_00041`, chosen=oracle=level 3, IoU 0.9066.
- full run: 2026-06-19, Slurm reservation `cal_jhbae_rtxpro5000_192083786c86`, `--gres=gpu:1`, 208 rows.

결과:

| scene | actual | level oracle | threshold oracle | level gap | threshold gap |
|---|---:|---:|---:|---:|---:|
| figurines | 0.5589 | 0.5762 | 0.6619 | 0.0174 | 0.0857 |
| ramen | 0.5445 | 0.5649 | 0.6391 | 0.0204 | 0.0742 |
| teatime | 0.6616 | 0.6846 | 0.7646 | 0.0230 | 0.0800 |
| waldo_kitchen | 0.4702 | 0.5323 | 0.6538 | 0.0623 | 0.1215 |

VALA-native class counts:

| scene | actual ok | calibration fail | representation fail | selection fail |
|---|---:|---:|---:|---:|
| figurines | 36 | 2 | 17 | 1 |
| ramen | 40 | 8 | 22 | 1 |
| teatime | 48 | 4 | 6 | 1 |
| waldo_kitchen | 11 | 2 | 6 | 3 |

RF-V1 판정: **지지**. Failed rows(`actual_iou < 0.5`) 73개 중 `vala_selection_fail` 또는 `vala_calibration_fail`이 22개, **30.1%**다. 사전 기준 25%를 넘으므로 VALA feature map 안에도 actual selection/calibration이 놓치는 recoverable 후보가 남아 있다고 본다.

단위:

- per `(case_id, scene, frame, prompt)`.

계산:

1. 3개 level의 512-d feature map을 읽는다.
2. VALA `OpenCLIPNetwork.get_max_across()`와 같은 canon-contrast relevance를 계산한다.
3. VALA native actual:
   - `chosen_level = argmax(max_relevance(level, prompt))`
   - `actual_iou = IoU(mask(chosen_level, native_thresh), GT)`
4. VALA oracle:
   - `oracle_iou_level = max_level IoU(mask(level, native_thresh), GT)`
   - `oracle_iou_level_thresh = max_{level, threshold} IoU(mask(level, threshold), GT)`
5. gap:
   - `level_selection_gap = oracle_iou_level - actual_iou`
   - `threshold_gap = oracle_iou_level_thresh - oracle_iou_level`

VALA-native class:

| class | 조건 | 뜻 |
|---|---|---|
| **vala_actual_ok** | `actual_iou >= 0.5` | VALA native selection으로 해결됨 |
| **vala_selection_fail** | `oracle_iou_level >= 0.5` and `actual_iou < 0.5` | 좋은 level은 있는데 score가 못 고름 |
| **vala_calibration_fail** | `oracle_iou_level < 0.5`, `oracle_iou_level_thresh >= 0.5` | level은 부족하지 않지만 threshold policy가 안 맞음 |
| **vala_representation_fail** | `oracle_iou_level_thresh < 0.5` | feature/render mask family 안에 답이 약함 |

이 실험이 P1에서 중요한 이유:

- `vala_selection_fail`가 크면 VALA의 robust representative 이후에도 query-time regime/selection 문제가 남는다.
- `vala_calibration_fail`가 크면 "평균"보다 threshold/guard 문제가 더 큰 축이다.
- `vala_representation_fail`가 크면 visibility/median이 해당 query evidence를 feature map에 충분히 보존하지 못한 것이다.

---

## 6. P1V-M2: robust-gate vs mean/non-gated VALA ablation

목적: VALA가 제안한 aggregation 처방이 무엇을 고치는지 같은 VALA pipeline 안에서 분리한다.

실행:

- driver: `scripts/run_p1e_vala_rf_v2.sh`
- suffix-aware renderer: `scripts/p1e_vala_feature_map_renderer_suffix.py`
- pipeline: `scripts/p1e_vala_rf_v2_pipeline.py`
- finalize: `scripts/p1e_vala_rf_v2_finalize.py`
- output doc: [vala_rf_v2_robust_vs_mean.md](vala_rf_v2_robust_vs_mean.md)
- Slurm: reservation `cal_jhbae_rtxpro5000_192083786c86`, `--gres=gpu:1`.

공정한 ablation 통제:

| factor | robust-gate condition | mean/non-gated condition |
|---|---|---|
| RGB 3DGS | 동일 checkpoint | 동일 checkpoint |
| source language feature | 동일 `langsplat/language_features` | 동일 |
| train/test frames | 동일 | 동일 |
| prompt/eval | 동일 common or official evaluator | 동일 |
| changed element | visibility gate + geometric median | standard weighted average |

우선 scene:

- `ramen`: 재현성 안정, 잔여 query failure가 큼.
- `waldo_kitchen`: public reproducibility gap flag를 달고 별도 분석.

결과:

| scene | condition | actual | level oracle | threshold oracle | level gap | threshold gap |
|---|---|---:|---:|---:|---:|---:|
| ramen | mean/non-gated | **0.5936** | 0.6122 | 0.6788 | 0.0186 | 0.0666 |
| ramen | robust-gate | 0.5445 | 0.5649 | 0.6391 | 0.0204 | 0.0742 |
| waldo_kitchen | mean/non-gated | **0.6436** | 0.6722 | 0.7630 | 0.0286 | 0.0908 |
| waldo_kitchen | robust-gate | 0.4702 | 0.5323 | 0.6538 | 0.0623 | 0.1215 |

precision/recall/area:

| scene | condition | precision | recall | pred/gt area |
|---|---|---:|---:|---:|
| ramen | mean/non-gated | **0.6534** | **0.8026** | 1.7594 |
| ramen | robust-gate | 0.6509 | 0.6953 | **1.4272** |
| waldo_kitchen | mean/non-gated | **0.7723** | **0.7948** | **0.9821** |
| waldo_kitchen | robust-gate | 0.7185 | 0.6045 | 0.6162 |

RF-V2 판정:

- **unsupported / partially inverted**.
- mean/non-gated가 두 scene 모두 actual IoU, precision, recall에서 robust-gate보다 높다.
- robust-gate는 ramen에서 area overgrowth를 줄이지만 recall 손실로 IoU가 낮아진다.
- waldo_kitchen에서는 robust-gate가 under-coverage로 가고, mean/non-gated가 area-ratio까지 더 좋다.
- 그래도 selection/calibration failure subset의 threshold+level oracle gap은 양쪽 조건 모두 크게 남는다.

Recoverable failed rows:

| scene | condition | n | actual | threshold oracle | total gap |
|---|---|---:|---:|---:|---:|
| ramen | mean/non-gated | 7 | 0.3416 | 0.6461 | 0.3045 |
| ramen | robust-gate | 9 | 0.3557 | 0.6470 | 0.2912 |
| waldo_kitchen | mean/non-gated | 3 | 0.3591 | 0.7468 | 0.3877 |
| waldo_kitchen | robust-gate | 5 | 0.3410 | 0.7283 | 0.3873 |

해석:

- "robust-gate가 mean보다 항상 낫다"는 주장은 이번 artifact/evaluator에서는 성립하지 않는다.
- 하지만 "aggregation rule이 바뀌어도 query-time level/threshold evidence selection 문제가 남는다"는 P1 핵심은 성립한다.
- 그러므로 P1은 특정 robust aggregation의 우월성보다 **query-conditioned evidence-regime collapse**로 문제를 정의해야 한다.

---

## 7. 논문에서의 위치

VALA와 차별화되는 우리의 문장은 이렇게 가야 한다.

> VALA correctly identifies visibility leakage and multi-view drift, and improves the representative language feature. However, a single robust representative does not decide which evidence regime a query requires. Our P1 reframes the failure as a query-conditioned regime-selection problem: consensus-preserving queries and minority-evidence queries require different aggregation and guard behavior.

이 문장이 실험으로 성립하려면 필요한 표는 세 개다.

1. **Descriptive bridge table**: VALA official output도 THGS/ReLaGS taxonomy에서 uniform하게 해결되지 않는다. 이미 P1V-D1로 있음.
2. **VALA-native oracle table**: 좋은 level/threshold 후보가 있는데 actual selection이 놓치는 prompt-frame이 존재한다. P1V-M1.
3. **Within-VALA ablation table**: aggregation condition이 어떤 failure distribution을 바꾸고, 어떤 selection/calibration gap을 남기는지 분리한다. P1V-M2.

이 세 표가 모이면 "VALA는 틀렸다"가 아니라 "VALA가 맞게 고친 축과 아직 문제정의가 덜 된 축을 분리했다"가 된다. 이 톤이 top-tier rebuttal에도 가장 안전하다.
