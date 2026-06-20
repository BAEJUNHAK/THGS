# VALA RF-V2 robust-gate vs mean/non-gated ablation

> 생성: 2026-06-19  
> 산출물:
> - `output/diagnostics/p1e_vala_rf_v2_official_2d_summary.csv`
> - `output/diagnostics/p1e_vala_rf_v2_official_2d_prompt_detail.csv`
> - `output/diagnostics/p1e_vala_rf_v2_oracle_detail.csv`
> - `output/diagnostics/p1e_vala_rf_v2_robust_vs_mean_detail.csv`
> - `output/diagnostics/p1e_vala_rf_v2_robust_vs_mean_agg.csv`

---

## 1. 질문

RF-V2의 질문은 다음이었다.

> 같은 VALA pipeline에서 visibility-aware robust-gate aggregation이 mean/non-gated aggregation보다 overgrowth를 줄이고, 그래도 VALA-native selection/calibration failure가 남는가?

이 실험은 mechanism lane이다. 그래서 다음 요소를 고정했다.

| factor | robust-gate | mean/non-gated |
|---|---|---|
| RGB 3DGS | 동일 `refersplat_3dgs_valafeat_full` | 동일 |
| source language features | 동일 LERF-OVS `language_features` | 동일 |
| frames/prompts/GT | 동일 VALA official 2D evaluator input | 동일 |
| evaluator | 동일 `evaluate_iou_loc.py` + prompt table + native oracle adapter | 동일 |
| changed element | `chkpnt30000_langfeat_{1,2,3}_stochastic_gate.pth` | `chkpnt30000_langfeat_{1,2,3}.pth` |

실행은 Slurm reservation `cal_jhbae_rtxpro5000_192083786c86`에서 `--gres=gpu:1`로 진행했다. 원본 VALA hardcode를 직접 수정하지 않고 `scripts/p1e_vala_feature_map_renderer_suffix.py`로 checkpoint suffix만 선택했다.

---

## 2. Overall 결과

Leaderboard나 성능 주장은 official saved actual만 본다. Oracle은 upper bound diagnostic이다.

| scene | condition | actual IoU | level oracle | threshold+level oracle | level gap | threshold gap |
|---|---|---:|---:|---:|---:|---:|
| ramen | mean/non-gated | **0.5936** | 0.6122 | 0.6788 | 0.0186 | 0.0666 |
| ramen | robust-gate | 0.5445 | 0.5649 | 0.6391 | 0.0204 | 0.0742 |
| waldo_kitchen | mean/non-gated | **0.6436** | 0.6722 | 0.7630 | 0.0286 | 0.0908 |
| waldo_kitchen | robust-gate | 0.4702 | 0.5323 | 0.6538 | 0.0623 | 0.1215 |

Mean/non-gated가 두 scene 모두 actual IoU에서 robust-gate보다 높았다.

- ramen: +0.0491
- waldo_kitchen: +0.1735

따라서 RF-V2의 "robust-gate가 mean보다 성능/precision을 개선한다"는 강한 예측은 지지되지 않는다.

---

## 3. Precision, recall, area

| scene | condition | precision | recall | pred/gt area ratio | area-ratio abs error |
|---|---|---:|---:|---:|---:|
| ramen | mean/non-gated | **0.6534** | **0.8026** | 1.7594 | 0.7594 |
| ramen | robust-gate | 0.6509 | 0.6953 | **1.4272** | **0.4272** |
| waldo_kitchen | mean/non-gated | **0.7723** | **0.7948** | **0.9821** | **0.0179** |
| waldo_kitchen | robust-gate | 0.7185 | 0.6045 | 0.6162 | 0.3838 |

해석:

- ramen에서는 robust-gate가 mask area를 줄여 overgrowth는 완화한다. 하지만 precision은 거의 늘지 않고 recall이 크게 떨어져 IoU가 낮아진다.
- waldo_kitchen에서는 robust-gate가 area를 너무 줄여 under-coverage가 되고, mean/non-gated가 precision/recall/area-ratio 모두 더 좋다.

즉 robust-gate는 "면적을 줄이는 방향"의 효과는 보이지만, 이 artifact와 evaluator에서는 그것이 일관된 성능 개선으로 이어지지 않았다.

---

## 4. VALA-native failure class

| scene | condition | actual ok | calibration fail | representation fail | selection fail |
|---|---|---:|---:|---:|---:|
| ramen | mean/non-gated | 45 | 6 | 19 | 1 |
| ramen | robust-gate | 40 | 8 | 22 | 1 |
| waldo_kitchen | mean/non-gated | 16 | 3 | 3 | 0 |
| waldo_kitchen | robust-gate | 11 | 2 | 6 | 3 |

Mean/non-gated는 actual ok를 늘리고 representation fail을 줄였다. 특히 waldo_kitchen에서 robust-gate의 selection fail 3개와 representation fail 6개가 mean/non-gated에서는 각각 0개와 3개로 줄었다.

하지만 mean/non-gated에서도 calibration failure가 남는다. 즉 "robust가 이긴다"는 예측은 틀렸지만, "single aggregation output만으로 query-time evidence selection/calibration이 끝나지 않는다"는 P1 방향은 오히려 더 선명해졌다.

---

## 5. Recoverable failed rows

Selection/calibration failure subset만 보면, oracle upper bound까지의 gap은 여전히 크다.

| scene | condition | n | actual | threshold+level oracle | total gap |
|---|---|---:|---:|---:|---:|
| ramen | mean/non-gated | 7 | 0.3416 | 0.6461 | 0.3045 |
| ramen | robust-gate | 9 | 0.3557 | 0.6470 | 0.2912 |
| waldo_kitchen | mean/non-gated | 3 | 0.3591 | 0.7468 | 0.3877 |
| waldo_kitchen | robust-gate | 5 | 0.3410 | 0.7283 | 0.3873 |

이 표는 RF-V2의 두 번째 절반을 지지한다. aggregation condition이 바뀌어도, recoverable evidence가 있는데 actual level/threshold policy가 못 쓰는 사례가 남는다.

---

## 6. RF-V2 판정

**RF-V2는 사전등록 기준으로 unsupported / partially inverted.**

이유:

1. robust-gate가 mean/non-gated보다 actual IoU를 개선하지 못했다. 오히려 mean/non-gated가 ramen과 waldo_kitchen 모두에서 더 높다.
2. robust-gate가 precision을 개선하지 못했다. ramen은 거의 동률이고 waldo_kitchen은 더 낮다.
3. robust-gate는 ramen에서 area overgrowth를 줄였지만 recall 손실 때문에 IoU가 하락했다. waldo_kitchen에서는 area-ratio도 mean/non-gated가 더 좋다.
4. 그러나 두 condition 모두 selection/calibration recoverable failure에서는 total oracle gap이 0.29-0.39로 남는다.

따라서 논문에 쓸 수 있는 문장은 다음이다.

> In our public-artifact one-factor VALA ablation, robust-gated aggregation is not uniformly better than mean aggregation under the official 2D actual evaluator. Nevertheless, both aggregation regimes retain substantial query-level selection/calibration gaps, indicating that the unresolved axis is not merely how to form a robust representative feature, but how to choose the evidence regime required by each query.

쓰면 안 되는 문장:

> VALA's robust gate fixes overgrowth and therefore solves aggregation failure.

> VALA paper used oracle evaluation.

---

## 7. P1 문제정의에 주는 의미

RF-V2가 빗나간 것은 나쁜 소식이 아니다. 오히려 P1 문제정의를 더 안전하게 만든다.

- VALA의 visibility-aware robust aggregation은 중요한 처방이지만, 공개 artifact의 이 one-factor ablation에서는 "항상 mean보다 낫다"는 단순 명제가 성립하지 않았다.
- mean/non-gated가 더 좋은 scene도 있고, robust-gate가 area를 줄이다가 minority/visible evidence recall을 잃는 scene도 있다.
- 두 condition 모두 recoverable selection/calibration gap이 남으므로, 문제는 특정 aggregation rule 하나의 승패가 아니라 **query-conditioned evidence-regime collapse**로 정의해야 한다.

최종 문장:

> VALA partially mitigates the symptom identified as noisy aggregation, but does not eliminate the underlying evidence-regime collapse: some failures lack recoverable representation, while others contain recoverable evidence missed by level/threshold selection.
