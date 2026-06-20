# P1-E External Code Study Plan

> 생성 2026-06-15. 상위 문서: [p1_problem_experiments.md](p1_problem_experiments.md), [competitor_autopsy.md](competitor_autopsy.md).
>
> 목적: P1-A의 family-level 부검을 실제 공개 코드 기반의 **per-paper 실측**으로 격상한다. 우선 대상은 사용자가 관심을 보인 **VALA(C1)** 와 **Segment then Splat(C5)** 이다.

---

## 0. 현재 결론

나는 외부 레포를 받아와서 논문+코드를 같이 공부하는 것이 맞다고 본다. 단, 바로 두 방법을 모두 학습시키는 것보다 **VALA 먼저, Segment then Splat은 어댑터 설계 후**가 더 좋다.

- **VALA** 는 우리가 이미 부검한 robust-statistics family와 거의 정면으로 겹친다. 논문 주장도 "occluded/background leakage + multi-view drift"이고, 처방도 visibility gate + cosine-space streaming weighted geometric median이다. 따라서 성공하면 P1-A의 "근사 rule"을 실제 paper implementation으로 승격할 수 있다.
- **Segment then Splat(StS)** 은 더 큰 판을 건드린다. 언어 feature 평균을 고치는 것이 아니라 object-first reconstruction으로 문제를 우회한다. 그래서 직접 비교가 더 설득력 있을 수 있지만, THGS식 SP rank/B7/A4를 그대로 붙일 수 없어 **object-group diagnostic adapter**가 필요하다.
- 따라서 P1-E의 첫 논문용 산출물은 "VALA도 실제 코드로 돌리면 phantom/easy trade-off가 남는가"이고, 두 번째 산출물은 "StS는 phantom을 없애는가, 아니면 tracking/init/object-CLIP association failure로 실패 위치가 이동하는가"가 되어야 한다.

### 0.1 P1-E 의 진짜 질문

P1-E 는 "평균이 문제다"를 다시 증명하는 실험이 아니다. 그 문장은 이미 기존 논문들이 말했다. 우리가 해야 하는 일은 기존 문제정의를 더 구체화하는 것이다.

> **기존 정의**: multi-view average 가 noisy/occluded/drifting features 를 섞어서 language feature 를 망친다.
>
> **우리 정의**: query 별로 필요한 증거 체제가 다른데, 기존 method 들은 그 regime 을 판별하지 못한다. easy query 는 consensus 를 보존해야 하고, phantom query 는 query-conditioned minority evidence 를 복구해야 한다.

이 재정의가 있어야 top-conference novelty 가 생긴다. "더 좋은 average"가 아니라 **average 가 맞는 경우와 틀린 경우를 분리하는 진단-설계 framework**가 우리의 위치다.

### 0.2 기존 paper 와의 차별화 axis

| Axis | 기존 paper 의 형태 | 우리의 고도화 |
|---|---|---|
| 문제정의 | average/aggregation 이 noisy 하다 | query별로 consensus regime 과 minority-evidence regime 이 갈린다 |
| 처방 | robust representative, bag/top-k, object-first 등 하나의 주 처방 | 어떤 처방이 어떤 regime 을 살리고 어떤 regime 을 희생하는지 측정 |
| 평가 | 전체 mIoU 중심 | full/easy/phantom split + oracle/actual gap + retrieval/selection failure |
| 실패 해석 | 벤치마크 숫자가 낮거나 높음 | failure location: representative feature, object discovery, query selection, abstention/guard 부재 |
| method 요구사항 | noise 를 줄이는 aggregator | query-conditioned guarded aggregation |

### 0.3 먼저 할 것 — dump 기반 faithful regime split (GPU 0, real-VALA repro 전의 결정 게이트)

real-VALA repro 는 GPU-days + sm_120 빌드 + 논문 숫자 재현 리스크를 진다. 그런데 **VALA 의 mechanism 질문("robust aggregation 이 consensus regime 만 살리고 minority regime 은 못 살리는가")은 우리 자산으로 GPU 없이 거의 답이 나온다**:

- P1-A 의 CA-1 이 이미 *VALA-faithful weighted geometric median + gating* (gm_w/gm_g) 을 우리 per-view dump 위에서 돌렸고, per-prompt mask IoU 까지 `p1a_mask_iou.csv` 에 있다. 여기에 **consensus/minority regime split + regime map 만 얹으면** faithful-VALA 의 regime 거동이 즉시 나온다 (= `p1e_vala_regime_split.csv` 의 mechanism 내용, faithful-rule 버전).
- real-VALA 는 그 위에 "feature field 까지 포함한 full system 도 같은 trade-off 인가" 의 **per-paper 확증**을 더하는 것이다 — mechanism 자체는 dump 로 충분하다.

**그래서 P1-E 의 0번 실험은 GPU-free dump regime split (§6.0) 이고, 이것이 real-VALA 에 GPU 를 투입할지의 결정 게이트다**: faithful-VALA 가 깨끗이 갈리면 (easy 보존·phantom 무회복) → real-VALA 는 확증용이라 우선순위·예산을 낮춘다 / 안 갈리면 → VALA 의 feature field 가 mechanism 을 바꾸는 것이므로 real repro 가 필수가 된다.

### 0.4 2026-06-19 패치 — VALA P1 실험은 세 claim lane으로 재설계

VALA 재현과 job84 official 2D audit 이후, P1-E의 VALA 실험은 [../fairness/p1_vala_protocol.md](../fairness/p1_vala_protocol.md)를 따른다.

핵심 변경:

- **D/native descriptive**: VALA official output을 THGS/ReLaGS taxonomy로 split하는 실험은 가설 생성용이다. 이미 `scripts/p1e_vala_official_2d_prompt_table.py`로 `output/diagnostics/p1e_vala_official_2d_prompt_detail.csv`와 `..._agg.csv`를 생성했다. 이 결과는 scene-dependent라서 "VALA가 phantom을 일괄 해결했다"는 단순 서사를 지지하지 않지만, THGS class 전이이므로 mechanism 결론은 아니다.
- **B/mechanism precursor**: 다음 실험은 VALA 자체의 feature map에서 level/threshold oracle과 actual chosen mask를 비교하는 **VALA-native oracle table**이다. 이것이 있어야 "representative feature 부재", "level selection 실패", "threshold/calibration 실패"를 나눌 수 있다. Driver는 `scripts/p1e_vala_native_2d_oracle.py`로 추가했고, 1-row CPU dry-run은 통과했다.
- **B/mechanism ablation**: 진짜 원인 귀속은 같은 VALA pipeline 안에서 robust-gate checkpoint와 mean/non-gated checkpoint를 비교해야 한다. 현재 public runs에는 `_stochastic_gate.pth`만 있으므로 mean checkpoint 재생성은 GPU 예약이 필요하다.
- **A/leaderboard**: final mask ranking은 common harness로 따로 만든다. native VALA 2D/3D 숫자와 THGS/ReLaGS native 숫자는 한 표에 놓지 않는다.

P1 novelty 문장도 이에 맞춰 바뀐다: VALA는 visibility leakage와 multi-view drift를 올바르게 고쳤지만, P1은 그 이후에도 남는 **query-conditioned evidence-regime selection**을 문제로 정의한다.

2026-06-19 실행 결과:

- P1V-M1 완료. 산출물은 `output/diagnostics/p1e_vala_native_2d_oracle_official_actual_detail.csv`와 `..._agg.csv`.
- Actual은 official saved `chosen_*.png` 점수로 교정했고, oracle은 VALA feature map에서 level/threshold sweep으로 계산했다.
- Failed rows 73개 중 `vala_selection_fail` 또는 `vala_calibration_fail`이 22개(**30.1%**)라 RF-V1을 지지한다.
- scene 평균 actual → threshold oracle: figurines 0.5589→0.6619, ramen 0.5445→0.6391, teatime 0.6616→0.7646, waldo 0.4702→0.6538.

2026-06-19 RF-V2 실행 결과:

- P1V-M2 robust-gate vs mean/non-gated ablation 완료. 산출물은 `output/diagnostics/p1e_vala_rf_v2_robust_vs_mean_detail.csv`, `..._agg.csv`, 문서는 [../fairness/vala_rf_v2_robust_vs_mean.md](../fairness/vala_rf_v2_robust_vs_mean.md).
- 같은 RGB 3DGS/source language feature/frames/prompts/GT/evaluator에서 language aggregation checkpoint만 바꿨다.
- RF-V2의 강한 예측은 **unsupported / partially inverted**. mean/non-gated가 actual IoU에서 ramen **0.5936 > 0.5445**, waldo_kitchen **0.6436 > 0.4702**로 robust-gate보다 높았다.
- robust-gate는 ramen에서 pred/gt area ratio를 1.7594→1.4272로 줄였지만 recall도 0.8026→0.6953으로 낮아져 IoU가 하락했다. waldo_kitchen에서는 mean/non-gated가 precision/recall/area-ratio 모두 더 좋았다.
- 그러나 recoverable selection/calibration failure에는 두 조건 모두 큰 oracle gap이 남았다: ramen mean 0.3045, ramen robust 0.2912, waldo mean 0.3877, waldo robust 0.3873.
- 결론: VALA의 robust aggregation이 public artifact의 official 2D actual에서 항상 우세하다고 말할 수 없다. 하지만 어떤 aggregation condition에서도 query-time level/threshold evidence selection 문제가 남으므로, P1 문제정의는 "better robust average"가 아니라 **query-conditioned evidence-regime collapse**로 고정해야 한다.

---

## 1. 받아온 레포와 논문 링크

| Method | Local source | Paper / project | 현재 판정 |
|---|---|---|---|
| VALA | `external_methods/VALA` | arXiv 2509.05515, GitHub `changandao/VALA` | **P1-E 1순위** |
| Segment then Splat | `external_methods/Segment-then-Splat` | arXiv 2503.22204, project page, GitHub `luyr/Segment-then-Splat` | **P1-E 2순위** |

중요한 실무 관찰:

- 로컬에는 이미 `data/lerf_ovs`와 `data/lerf_ovs/label`이 있다. VALA가 기대하는 LERF-OVS 구조와 꽤 가깝다.
- StS는 README 기준 `train.txt/test.txt`와 `multiview_masks_*_merged`가 필요하다. 현재 로컬 `data/lerf_ovs`에는 이 전처리 산출물이 보이지 않으므로, 저자 제공 preprocessed LERF-OVS를 쓰는 쪽이 먼저다.
- 두 레포 모두 PyTorch 2.3 + CUDA 12.1 계열을 요구한다. 현재 장비/환경이 sm_120이면 native extension 빌드 호환성 리스크가 있다.

---

## 2. VALA 소스 리딩 메모

### 2.1 논문 주장

VALA가 보는 원인은 두 가지다.

1. ray에서 실제 기여가 거의 없는 background/occluded Gaussian이 foreground와 같은 language feature를 받는 leakage.
2. 여러 view에서 나온 language embedding noise/drift를 단순 평균하면 view-consistent feature가 망가지는 문제.

처방은 visibility-aware gate로 의미 있는 Gaussian-view assignment만 남기고, cosine space에서 streaming weighted geometric median으로 multi-view feature를 합치는 것이다.

우리 관점의 핵심: 이것은 P1-A의 **robust 통계 family + visibility gating** 실제 구현이다. 그러므로 질문은 "VALA가 평균보다 좋나?"가 아니라 **좋아진다면 어떤 regime 이 좋아지는가**다. robust/gating 이 easy consensus 를 안정화하지만 phantom minority evidence 를 못 살린다면, VALA는 average problem 의 한 부분만 해결한 것이다.

### 2.2 코드에서 확인한 파이프라인

대표 entrypoints:

- `external_methods/VALA/scripts/run_lerf_3d.sh`
- `external_methods/VALA/scripts/eval_lerf_ovs_3d.sh`
- `external_methods/VALA/gaussian_feature_extractor.py`
- `external_methods/VALA/scene/gaussian_model.py`
- `external_methods/VALA/eval/render_lerf_by_text_langsplat.py`
- `external_methods/VALA/eval/compute_lerf_iou.py`
- `external_methods/VALA/eval/openclip_encoder.py`

흐름:

1. `train.py`로 LERF-OVS scene별 3DGS를 30k iteration 학습.
2. `gaussian_feature_extractor.py --feature_level 1/2/3 --use_efficient`로 각 level의 Gaussian language feature를 추출.
3. `accumulate_gaussian_feature_per_view_robust()`가 per-view CLIP feature를 Gaussian에 누적한다.
4. significance 기반 gate를 적용하고, cosine update로 robust language feature를 갱신한다.
5. `finalize_gaussian_features_robust()`에서 low-weight Gaussian을 prune하고 normalize한다.
6. eval에서는 OpenCLIP ViT-B-16 + negatives `object/things/stuff/texture`를 사용한다.
7. prompt별로 3개 feature level 중 max score level을 고른 뒤, Gaussian relevance를 KNN smoothing하고 thresholding해서 mask를 렌더한다.
8. `compute_lerf_iou.py`가 LERF-OVS label과 `renders_silhouette` PNG를 비교해 mIoU/Acc를 낸다.

특히 중요한 접점:

- VALA도 score 함수는 THGS 계열과 같은 canon-contrast 구조를 쓴다. 따라서 P1-B에서 본 top1-conf/margin/absence 문제가 완전히 무관하지 않다.
- eval output이 prompt/frame별 silhouette PNG라서, 우리 67 prompt를 phantom/easy/control로 다시 쪼개는 것은 쉽다.
- 다만 VALA는 SP가 아니라 Gaussian-level mask를 직접 만든다. B7/A4를 그대로 재사용하기보다 **VALA-native oracle gap**을 정의해야 한다.

### 2.3 VALA에 붙일 진단 어댑터

최소 진단:

- native 전체 mIoU, Acc@0.25, Acc@0.5 재현.
- 우리 67 prompt taxonomy로 재집계: full / phantom / easy / other.
- prompt별 `selected_level`, max relevance score, mask area, IoU를 저장.

강한 진단:

- threshold sweep: prompt별 best possible threshold IoU와 native threshold IoU의 gap.
- level oracle: 3개 level 중 IoU가 가장 높은 level과 score가 고른 level의 gap.
- score diagnostic: top1-conf가 failure를 예측하는지 확인. P1-B의 "공짜 신호"가 VALA에서도 살아 있으면 아주 좋다.
- regime split: VALA gain 을 easy/phantom/other 로 나눠서 robust representative 가 consensus regime 을 강화한 것인지, minority-evidence regime 을 복구한 것인지 분리한다.
- mean/median/VALA-native 비교: 가능하면 native output 외에 같은 scene 에서 simple mean 또는 non-robust feature checkpoint 와 비교해 "robustification 의 실제 기여"를 분리한다.

B7/A4 대응:

- **B7 analogue**: "어느 threshold/level에서도 GT와 겹치는 Gaussian mask가 존재하는가"로 oracle recoverability를 정의한다.
- **A4 analogue**: "native score가 oracle level/threshold 후보를 top으로 고르는가"로 retrieval/ranking failure를 정의한다.
- 이렇게 하면 SP가 없어도 `oracle exists vs native retrieval succeeds` 2x2를 만들 수 있다.

### 2.4 VALA go/no-go

Go 조건:

- 한 scene(`ramen` 추천)에서 train+feature extraction+eval이 끝난다.
- `predictions_mask_0.6/renders_silhouette`이 생성된다.
- native result가 논문/README의 LERF-OVS 수준과 같은 order로 나온다. 정확 재현이 아니어도 pipeline이 정상이어야 한다.

No-go / 보류 조건:

- CUDA extension 빌드가 sm_120에서 막힌다.
- feature extraction이 기존 `data/lerf_ovs/language_features`와 호환되지 않아 SAM+CLIP feature를 전부 다시 뽑아야 하고, 그 비용이 P1-E 예산을 과도하게 잡아먹는다.
- native eval output을 만들 수 없으면 per-paper 실측으로 쓰기 어렵다. 이 경우 P1-A의 family-level 결과만 유지한다.

---

## 3. Segment then Splat 소스 리딩 메모

### 3.1 논문 주장

StS의 핵심은 "splat then segment"가 아니라 **segment then splat**이다. 기존 방법은 reconstruction 뒤에 language field나 2D feature map으로 query를 처리하기 때문에 Gaussian-object misalignment와 multi-view inconsistency가 생긴다고 본다.

StS는 먼저 object mask를 multi-view로 추적하고, COLMAP 초기 Gaussian에 object ID를 부여한 뒤, reconstruction 내내 object-specific Gaussian set을 유지한다. 마지막에 object group마다 CLIP embedding을 붙이고 text query는 object embedding과 cosine similarity로 고른다.

우리 관점의 핵심: StS는 평균 aggregation을 고친 방법이 아니라 **문제 위치를 바꾸는 구조 교체군**이다. 그래서 결과가 좋으면 "object-first는 실제로 paradigm escape인가?"를 봐야 하고, 실패하면 "aggregation phantom이 object tracking/init/association phantom으로 이동했는가?"를 봐야 한다. 특히 StS도 마지막에는 object embedding 과 text query 의 top1 association 을 한다. 따라서 average 를 피했더라도 **query selection/guard 문제**를 완전히 피한 것은 아닐 수 있다.

### 3.2 코드에서 확인한 파이프라인

대표 entrypoints:

- `external_methods/Segment-then-Splat/helpers/preprocess_mask.py`
- `external_methods/Segment-then-Splat/helpers/object_specific_initialization.py`
- `external_methods/Segment-then-Splat/train.py`
- `external_methods/Segment-then-Splat/render_objs.py`
- `external_methods/Segment-then-Splat/helpers/evaluation.py`
- `external_methods/Segment-then-Splat/gaussian_renderer/__init__.py`
- `external_methods/Segment-then-Splat/scene/gaussian_model.py`

흐름:

1. AutoSeg-SAM2 또는 preprocessed data로 large/middle/small multi-view masks를 만든다.
2. `preprocess_mask.py`가 overlapping masks를 정리한다.
3. `object_specific_initialization.py`가 COLMAP points를 mask에 투영해 `obj_id_default/middle/small`을 부여한다.
4. point가 없는 object는 dummy points로 보상하고, geometry+appearance distance로 duplicate/lost tracking object를 merge한다.
5. `train.py`가 normal RGB reconstruction loss에 object-specific render loss를 더한다.
6. stage1/2/3에서 small -> middle -> default 순서로 object supervision을 확장한다.
7. renderer는 `obj_level`, `obj_id`로 Gaussian subset만 렌더한다.
8. `render_objs.py`가 object별 test render를 만든다.
9. `helpers/evaluation.py`는 masked crop image를 CLIP ViT-L/14@336으로 encoding해서 object embedding을 만들고, text query와 cosine top1 object를 골라 IoU를 계산한다.

특히 중요한 접점:

- StS는 query-time에서 object top1 선택을 한다. 즉 실패가 나면 "object group은 있었는데 CLIP association이 틀렸는가"를 볼 수 있다.
- object group이 없는 경우도 중요하다. small object나 occluded object가 tracking/init에서 빠지면, StS는 애초에 query가 닿을 대상이 없다.
- GT가 2D visible mask인데 StS는 complete 3D object를 렌더할 수 있어, occlusion case에서는 IoU가 낮아도 failure 해석이 애매할 수 있다. 이건 논문에서도 인정하는 평가 mismatch라, 우리 진단에는 별도 flag가 필요하다.

### 3.3 StS에 붙일 진단 어댑터

StS는 SP rank가 아니라 object group retrieval이므로 다음 2x2가 더 자연스럽다.

| 축 | 질문 | 측정 |
|---|---|---|
| object oracle | GT prompt와 충분히 겹치는 object group이 어느 level에든 있는가 | per-prompt max IoU over all object renders |
| retrieval | CLIP top1이 그 oracle object를 고르는가 | top1 object IoU / oracle object IoU |
| granularity | small/middle/default 중 어느 level이 맞았는가 | oracle level vs chosen level |
| init/tracking | object group 자체가 없거나 너무 fragment/coarse한가 | max IoU ceiling, group area, duplicate groups |

실패 taxonomy:

- **S1 missing-object failure**: oracle max IoU가 낮다. object tracking/init가 대상 자체를 못 만들었다.
- **S2 association failure**: oracle max IoU는 높은데 CLIP top1이 다른 object를 고른다.
- **S3 granularity failure**: 맞는 object가 다른 level에는 있는데 native selection이 coarse/fine level을 잘못 고른다.
- **S4 evaluation-mismatch case**: complete 3D object render가 visible-only GT보다 커서 IoU가 낮다. 이건 실패로만 세면 불공정할 수 있다.
- **S5 regime-collapse case**: object-first 가 easy object 는 안정화하지만 small/occluded/minority prompt 에서 object discovery 또는 association 이 실패한다. 이 경우 StS 는 average 문제를 해결했다기보다 다른 bottleneck 으로 이동시킨 것이다.

### 3.4 StS go/no-go

Go 조건:

- 저자 제공 preprocessed LERF-OVS를 확보한다.
- 한 scene(`ramen` 또는 `waldo_kitchen`)에서 `render_objs.py --skip_train`까지 완료된다.
- object별 rendered masks를 모두 저장하고, `helpers/evaluation.py`의 top1 object choice를 prompt별로 기록할 수 있다.

보류 조건:

- preprocessed tracking dataset을 받지 못하면 AutoSeg-SAM2부터 재현해야 한다. README에도 해당 단계는 "To be verified"라고 되어 있어 P1-E 본 실험으로는 리스크가 크다.
- object별 render 수가 너무 많아 eval 비용이 폭증하면 우선 `ramen` 1 scene로 축소한다.

---

## 4. 실행 순서 제안

### Step 1. Source audit 확정

- VALA/StS README, paper, entrypoint, eval 코드를 계속 읽고 이 문서에 누적한다.
- 외부 레포는 third-party source로 취급하고, 논문 실험 전에는 최소 patch만 한다.

### Step 2. VALA one-scene feasibility

권장 scene: `ramen`.

1. `external_methods/VALA/dataset/3dgs/lerf_ovs`를 로컬 `data/lerf_ovs`에 연결하거나 path를 수정한다.
2. native script를 한 scene로 줄여 train 30k -> feature level 1/2/3 -> eval까지 실행한다.
3. 출력 silhouette으로 prompt별 IoU table을 만든다.
4. full/easy/phantom split으로 재집계한다.

성공하면:

- 4 scenes로 확장한다.
- threshold/level oracle adapter를 붙인다.
- P1-A 표에서 VALA row를 "family approximation"에서 "official code 실측"으로 승격한다.

### Step 3. StS data feasibility

1. 저자 제공 preprocessed `lerf_ovs_langsplat`를 확보한다.
2. one scene에서 `train.py` 또는 제공 checkpoint가 있으면 render/eval만 먼저 확인한다.
3. object render를 전부 저장하고, prompt별 top1 object와 oracle object를 비교하는 adapter를 작성한다.

성공하면:

- StS를 P1-E 두 번째 축으로 둔다.
- 논문에는 "object-first는 aggregation failure를 줄이는가, 아니면 missing/association/granularity failure로 이동시키는가"로 쓴다.

---

## 5. 사전등록 초안

아직 확정 전 초안이다. 실행 전 사용자와 리뷰 후 고정한다.

**operational 정의 고정 필수 (R11/R13 규율)**: V1/S1 실행 전에 ① "회복" = per-prompt mask IoU Δ ≥ +0.05 (rank 트랙은 oracle rank ≤3 복귀) ② "selection failure" = oracle-actual IoU gap ≥ +0.10 ③ regime 경계 = `thgs_class` (또는 V4 의 VALA-native class) — 이 셋을 숫자로 박은 뒤 돌린다. 사후 조정 금지.

### P1-E.VALA 예측

- **E-V1**: VALA는 전체 mIoU 또는 easy split 에서는 강할 수 있지만, phantom split 에서는 easy 만큼 안정적이지 않을 것이다. 즉 robust/gating 의 주효과는 minority recovery 보다 consensus stabilization 일 가능성이 높다.
- **E-V2**: threshold/level oracle gap이 큰 prompt가 남으면, VALA의 실패는 "feature aggregation만의 문제"가 아니라 query-time scoring/selection 문제로 해석된다.
- **E-V3**: P1-B의 top1-conf가 VALA failure/absence에도 살아 있으면, P2 guard의 method-agnostic 근거가 강화된다.
- **E-V4**: VALA가 phantom 을 크게 회복한다면, 어떤 subtype 을 회복했는지 분리한다. outlier/occlusion subtype 만 회복하고 coherent-plausible confuser subtype 이 남으면 우리의 regime-confusion 주장은 유지된다.

### P1-E.StS 예측

- **E-S1**: StS는 aggregation phantom 일부를 없앨 수 있지만, small/occluded object에서는 missing-object 또는 granularity failure가 남을 것이다.
- **E-S2**: object oracle은 높은데 CLIP top1이 틀리는 case가 나오면, "object-first도 최종 query selection은 zero-sum/ranking 문제를 피하지 못한다"는 주장이 가능하다.
- **E-S3**: visible-only GT와 complete-object render mismatch는 별도 flag로 분리한다. 이 case를 실패로만 세지 않는다.
- **E-S4**: StS가 전체 mIoU 를 높여도 phantom/easy split 에서 특정 prompt family 를 놓치면, "average 를 피한 구조 교체도 regime 판단 문제를 해결하지 못한다"는 증거가 된다.

---

## 6. 상세 실험 매트릭스

아래는 실제 실행할 실험 단위다. 원칙은 **native reproduction → regime split → oracle/actual gap → failure location** 순서다. 전체 mIoU 는 시작점일 뿐이고, 논문 기여는 2단계 이후에서 나온다.

### 6.0 P1-E.0 — dump 기반 faithful regime split (GPU 0, real-VALA 전에 먼저)

real-VALA repro 없이 기존 산출물만으로 regime-confusion 의 핵심 그림을 만든다. **real-VALA 에 GPU 를 투입할지의 결정 게이트** (§0.3).

**입력** (전부 `output/diagnostics/`): `p1a_mask_iou.csv` (208 = scene×prompt×eval_frame; `iou_{baseline,top5,qmax1,gm_u,gm_w,gm_g}`) · `cross_method_d2_decomposition.csv` (67; `thgs_class`/`relags_class`) · `persistent_phantoms_17.csv` · `ca1_gm_ranks_{thgs,relags}.csv` · `ca2_qmax_ranks_{thgs,relags}.csv` · `stage3_3_fullpool_ranks.csv` · `p1b_absence_scores.csv`.

**산출**:
1. `p1e0_regime_split.csv` — rule × regime 평균 mask IoU + Δ(vs baseline). rule = consensus family {baseline(mean), gm_u, gm_w, gm_g} ∪ selection family {top5, qmax1}. regime = easy / phantom / other (`thgs_class`). per-prompt IoU = eval_frame 평균.
2. `p1e0_regime_map.png` — per-prompt scatter: x = `iou_baseline` (consensus-regime IoU), y = `iou_top5` (minority-regime IoU), 색 = class. 대각선 기준 위/아래로 제로섬·regime 분리 시각화.
3. `p1e0_guard_auroc.csv` — easy-vs-phantom AUROC: `top1_mean`(=top1-conf) / `mean_margin` / `agree_mean_top5` (p1b 의 present 쿼리만, 양 method). present-vs-absent AUROC (top1-conf 0.84 기지) 와 병기.

**사전등록 (실행 전 고정)**:
- **G-A (regime confusion 시각화)**: robust family (gm_*) 는 easy Δ ≥ 0 AND phantom Δ ≤ 0, selection family (top5) 는 phantom Δ ≫ 0 AND easy Δ ≪ 0 — 거울상이 mask 수준 regime split 에서 재확인.
- **G-B (regime 분리 가능성)**: (`iou_top5 − iou_baseline`) 가 phantom 을 easy 에서 분리하는 AUROC ≥ 0.8 → 두 regime 이 *결과로* 분리됨.
- **G-C (guard 신호의 진짜 과제)**: top1-conf 의 easy-vs-phantom AUROC. 예측 (confident-impostor / 함정 2): **< 0.75 이고 자신의 absence AUROC 0.84 보다 낮다**. ≥ 0.8 → prompt-level top1-conf 가드 채택 / < 0.65 → structural 신호 (§6.0-b) 로 전환 필요.

**§6.0-b (후속, 여전히 GPU 0)**: structural 신호의 easy-vs-phantom AUROC — `stage3_3_allsp_*.pkl` / `stage5_relags_allsp_*.pkl` 를 **CPU 로** 로드해 mean-top1 SP 의 query-aligned view fraction · n_valid · coherence 계산 (R5 예측: structural ≫ score).

전부 pandas/numpy/matplotlib, CUDA 불필요 → SLURM 예약 없이 login 노드에서 실행.

**결과 (2026-06-15 실행, [scripts/p1e0_regime_split.py](../../../scripts/p1e0_regime_split.py); 산출물 `output/diagnostics/p1e0_{regime_split.csv,regime_map.png,guard_auroc.csv}`)**:

- **G-A ✅ PASS — mirror image 가 mask 수준에서 재확인**: robust `gm_g` easy **+1.3pt** / phantom **−1.3pt**, selection `top5` phantom **+10.3pt** / easy **−7.6pt**. strict-17 phantom: `top5` **+19.9pt** (stage3_3 정확 재현), `gm_g` **−1.8pt** (R11-d 재현). 각 family 가 한 regime 전용임이 mask Δ 로 확정 → A1(거울상)=A3(regime confusion)의 population 증거 성립.
- **G-C ✅ 예측 적중 — top1-conf 는 easy/phantom 가드로 부적합**: easy-vs-phantom AUROC **0.615**(THGS)/0.596(ReLaGS) ≪ 자신의 present-vs-absent **0.836/0.848**. confident-impostor 예측 그대로 — top1-conf 는 "있나/없나"는 알아도 "(mean top-1 이) 맞나"는 모른다. margin 0.578, agree 0.528 도 ≈random. → **prompt-level scalar 가드(후보1·후보3) 데이터로 기각, structural(§6.0-b)로 전환**.
- **G-B ❌ FAIL (중요) — regime 은 per-prompt 로 깨끗이 안 갈린다**: selection-benefit(`iou_top5−iou_baseline`)의 phantom-vs-easy AUROC **0.540**(strict-17 도 0.608, 둘 다 ≪0.8). 원인 = **phantom class 가 mask-outcome 상 bimodal**: 21 중 **7 은 +13~+94pt 회복 / 7 은 −9~−92pt 악화**(multi-instance·over-union 의심) / 나머지 무변. easy 도 median 0 + 손실 꼬리. **두 class median 모두 +0.0**. 일부는 rank-label vs mask-outcome 축 불일치(Stage 3.4 rank 착시·sake-cup 보수성)도 섞임.
- **함의 (P2 방향 재조정)**: regime-confusion 은 **population 수준에선 확정(G-A)** 이나 **per-prompt 단일 스칼라 가드는 원리적으로 어렵다(G-B·G-C 모두 ≈random)**. → P2 가드는 (a) **per-SP 단위**(lucky-view impostor 는 prompt 가 아니라 후보 SP 의 속성) + (b) **multivariate structural**(view-fraction·n_valid·coherence·rank-stability 조합) 로 가야 한다. easy 보호 목표(<1pt)는 단일 스칼라로는 미달 가능성 높음 — SP-level 로 내려가야.
- **real-VALA 게이트 판정**: faithful-VALA(gm_*)가 깨끗이 갈렸으므로(easy 보존·phantom 무회복) **real-VALA 는 확증용** → 우선순위 낮춤, P2 먼저(§0.3 게이트 통과).
- **다음**: §6.0-b structural 신호 AUROC (dump CPU 로드) + **multi-instance 7 prompt 제외 후 G-B 재측정** (bimodal 음수꼬리가 R7 set 인지 확인 — 맞으면 G-B 가 회복되고 "회복가능 phantom" subset 이 깨끗이 분리될 수 있음).

### 6.1 VALA 실험군

| ID | 실험 | 질문 | 산출물 | 판정 |
|---|---|---|---|---|
| **V0** | environment/import/build dry run | VALA를 이 장비에서 실행 가능한가? **+ mean/non-robust aggregation variant 를 코드에서 추출 가능한가 (V6 의 전제)** | import log, extension build log, mean-variant 추출 가능 여부 | 빌드 실패 시 P1-E.VALA 보류, P1-A family-level 결과 유지. mean-variant 추출 불가 시 mechanism 은 dump(§6.0)로만 주장 |
| **V1** | one-scene native reproduction (`ramen`) | 원저자 eval 경로가 정상 동작하는가? | `p1e_vala_native_ramen.csv`, native mIoU/Acc | silhouette PNG와 result JSON이 나오면 go |
| **V2** | 4-scene native reproduction | VALA 전체 성능이 논문/README order와 맞는가? | `p1e_vala_native_all.csv` | 정확 수치보다 protocol 정상성과 prompt별 output 확보가 핵심 |
| **V3** | regime split | VALA gain 은 easy/consensus 에서 오는가, phantom/minority 에서 오는가? | `p1e_vala_regime_split.csv` | easy gain >> phantom gain 이면 robust consensus 처방의 한계 확정 |
| **V4** | threshold/level oracle gap | 좋은 mask 후보는 있는데 native score/threshold가 못 고르는가? | `p1e_vala_oracle_gap.csv` | oracle IoU 높고 native IoU 낮으면 query-time selection failure |
| **V5** | score diagnostic | top1-conf, max relevance, area, selected level 이 failure 를 예측하는가? | `p1e_vala_score_diagnostics.csv` | top1-conf가 살아 있으면 P2 guard 의 method-agnostic 근거 강화 |
| **V6** | robustification ablation 가능성 점검 | VALA native robust feature 와 simple mean/non-gated variant 차이를 분리할 수 있는가? | `p1e_vala_ablation_probe.csv` | 가능하면 robust 처방의 실제 gain source 분리. 불가능하면 정직하게 native-only 로 제한 |

V3의 주 분석표:

| Group | n | THGS baseline IoU | VALA IoU | Δ | 해석 |
|---|---:|---:|---:|---:|---|
| easy/consensus | 41 | baseline | VALA | `VALA - baseline` | robust/gating 이 consensus 를 안정화하는지 |
| phantom/minority | 17 or 21 | baseline | VALA | `VALA - baseline` | minority evidence 를 복구하는지 |
| other | 나머지 | baseline | VALA | `VALA - baseline` | 부수 효과 |

V4의 핵심 분해:

```
native failure prompt
├── no oracle mask exists        → representation/geometry/feature field failure
├── oracle level exists          → level selection failure
├── oracle threshold exists      → threshold/score calibration failure
└── high score but low IoU       → score-mask misalignment / over-union
```

VALA에서 우리가 가장 보고 싶은 결과는 단순히 "mIoU가 낮다"가 아니다. **native는 실패했지만 threshold/level oracle은 성공하는 prompt**가 있으면, 기존 average-fix가 representative feature는 개선했어도 query-time decision을 못 고쳤다는 강한 증거가 된다.

> **방법론 주의 — mechanism 은 V6, V3 는 descriptive**: V3 의 Δ(THGS baseline → VALA) 는 aggregation regime 처리뿐 아니라 VALA 의 다른 파이프라인 (자체 3DGS·partition·SAM·feature field) 을 통째로 담으므로 **mechanism 주장에는 confounded** 하다. "robust agg 가 consensus 만 살린다" 의 mechanism 입증은 **같은 파이프라인에서 aggregation 만 토글하는 V6 (VALA-robust vs VALA-mean)** 가 해야 한다 (우리 Stage 3.2 B1.B 가 같은 dump 위에서 한 규율과 동일). 따라서 V6 는 optional 이 아니라 mechanism 의 본실험이고, 그 가능성(mean-variant 추출) 은 V0 에서 확인한다.
>
> **classification — V4 가 V3 의 분류기**: easy/phantom 라벨은 THGS-SP 기반이라 VALA 에 그대로 붙이면 method-혼선이 생긴다. V4(oracle gap) 로 **VALA-native easy/phantom** 을 정의하고 **THGS-class × VALA-native-class cross-tab** 을 보고한다 — 일치율 높으면 "regime 은 prompt 내재적" (Stage 1 의 THGS≈ReLaGS 를 VALA 로 확장, paradigm 주장 강화), 낮으면 regime 이 method 종속 (정직 보고). 이 cross-tab 자체가 paradigm 주장의 증거다.

### 6.2 Segment then Splat 실험군

| ID | 실험 | 질문 | 산출물 | 판정 |
|---|---|---|---|---|
| **S0** | preprocessed data availability | 저자 제공 `lerf_ovs_langsplat`를 확보했는가? | data manifest | 없으면 StS 본 실험 보류. AutoSeg-SAM2 재현은 별도 고비용 트랙 |
| **S1** | one-scene native reproduction (`ramen` or `waldo_kitchen`) | object render/eval 경로가 정상 동작하는가? | `p1e_sts_native_<scene>.csv` | object별 test render와 native mIoU가 나오면 go |
| **S2** | object oracle sweep | GT와 잘 겹치는 object group이 어느 level에든 존재하는가? | `p1e_sts_object_oracle.csv` | oracle 낮으면 object discovery/init failure |
| **S3** | retrieval gap | oracle object가 있는데 CLIP top1이 다른 object를 고르는가? | `p1e_sts_retrieval_gap.csv` | oracle high + top1 low = association failure |
| **S4** | granularity diagnosis | small/middle/default 중 어느 level이 맞고 native는 무엇을 고르는가? | `p1e_sts_granularity.csv` | 맞는 level이 따로 있으면 level-selection/granularity failure |
| **S5** | evaluation mismatch flag | complete-object render와 visible-only GT mismatch가 실패처럼 보이는가? | `p1e_sts_eval_mismatch.csv` | 이 케이스는 실패/성공 판정에서 별도 분리 |
| **S6** | regime split | StS가 easy object는 안정화하지만 small/occluded/minority prompt를 놓치는가? | `p1e_sts_regime_split.csv` | object-first 구조 교체의 놓친 regime 확인 |

StS 실패 taxonomy:

| Failure | 조건 | 의미 |
|---|---|---|
| **missing-object** | max object oracle IoU 낮음 | object tracking/init 가 target 을 만들지 못함 |
| **association** | oracle IoU 높음, CLIP top1 IoU 낮음 | object 는 있는데 query retrieval 이 틀림 |
| **granularity** | small/middle/default 중 oracle level과 chosen level 불일치 | 맞는 scale 은 있는데 선택 기준이 없음 |
| **over-complete render** | rendered complete object 가 visible GT보다 커서 IoU 하락 | 평가 mismatch. 실패로만 세면 불공정 |
| **regime-collapse** | easy는 강하지만 phantom/small/occluded에서 실패 | average 문제를 object bottleneck 으로 이동 |

StS는 VALA보다 더 늦게 하는 것이 맞다. 이유는 실패가 나오더라도 그 실패가 method 한계인지 data preprocessing 문제인지 분리해야 하기 때문이다. 그래서 **저자 제공 preprocessed data + native render/eval 성공**이 없으면 본 논문 실험으로 쓰지 않는다.

### 6.3 두 논문을 묶는 cross-paper 분석

최종 표는 method별 mIoU 순위표가 아니라 **problem-definition refinement table**이어야 한다.

| Method | Native metric | easy/consensus | phantom/minority | oracle/actual gap | failure location | 결론 |
|---|---:|---:|---:|---:|---|---|
| THGS | baseline | strong | weak | known | aggregation/selection | base failure |
| ReLaGS/ROFA | measured | stable | weak | known | outlier-removal insufficient | robust outlier fix 한계 |
| VALA | to measure | ? | ? | threshold/level | representative vs selection | robust+visibility fix 가 어느 regime 을 고치는지 |
| StS | to measure | ? | ? | object oracle/top1 | object discovery vs association | object-first 가 실패 위치를 옮기는지 |

이 표가 말해야 하는 최종 문장은 다음 중 하나다.

1. **예상 결과**: VALA는 consensus 를 안정화하고 StS는 object boundary 를 개선하지만, 둘 다 query-conditioned guard 가 없어 minority/regime 전환 문제를 남긴다.
2. **반대 결과**: VALA 또는 StS가 phantom/minority를 강하게 해결한다면, 그 성공 조건을 흡수해 P2의 설계를 수정한다.
3. **실행 불가 결과**: native code/data 제약으로 실행이 막히면, P1-A family-level 부검과 code-level source audit 으로만 주장 강도를 낮춘다.

### 6.4 우선순위와 중단 기준

실험 순서:

```
V0 → V1 → V3
   ↘ V4/V5
   ↘ V2 only if V1/V3가 유의미

S0 → S1 → S5 (eval-mismatch 비율 먼저 = validity gate)
   ↘ S2/S3 only if native object renders 확보 AND mismatch 비율이 해석 가능 수준
   ↘ S4/S6 after S2/S3
```

중단 기준:

- VALA V0/V1 실패: P1-E.VALA는 보류하고 P1-A의 CA-1 결과를 official-code source audit로 보강하는 선에서 멈춘다.
- VALA V3에서 easy/phantom 모두 크게 개선: 우리의 기존 taxonomy를 수정한다. 이 경우 VALA의 성공 조건을 P2 설계에 흡수한다.
- StS S0 실패: StS는 paper related-work/source-audit 포지셔닝으로만 사용한다.
- StS S1 성공 but S2/S3 실패: native number는 참고만 하고 mechanism claim에는 쓰지 않는다.
- **StS S5 에서 eval-mismatch 비율이 큼 (예: ≥30% prompt)**: visible-only GT 와 complete-object render 의 구조적 불일치가 IoU 를 지배 → LERF-OVS 에서 StS 는 *측정* 대상이 아니라 *포지셔닝/개념 대조* 대상으로 강등. S2/S3/S6 에 GPU 쓰기 전 이 게이트를 먼저 통과해야 함.

### 6.5 최소 산출물 세트

P1-E를 paper에 넣으려면 최소한 아래 3개가 필요하다.

| 산출물 | 필요 이유 |
|---|---|
| `p1e_vala_regime_split.csv` | VALA가 고친 regime과 놓친 regime을 보여주는 핵심 |
| `p1e_vala_oracle_gap.csv` | robust/gating 이후에도 query-time selection 문제가 남는지 증명 |
| `p1e_external_methods_summary.md` | VALA/StS/ReLaGS/THGS를 같은 language로 묶는 paper 표 초안 |

StS까지 포함하려면 추가로 아래가 필요하다.

| 산출물 | 필요 이유 |
|---|---|
| `p1e_sts_object_oracle.csv` | object-first representation이 target을 만들었는지 확인 |
| `p1e_sts_retrieval_gap.csv` | object는 있는데 CLIP association이 틀리는지 확인 |
| `p1e_sts_regime_split.csv` | object-first가 어떤 prompt family를 놓치는지 확인 |

---

## 7. 논문 기여로 바뀌는 조건

강한 positive:

- VALA official code에서도 phantom/easy split trade-off가 관찰된다.
- StS에서도 object oracle과 retrieval 사이 gap이 관찰된다.
- 그러면 우리 주장은 "THGS의 특수 버그"가 아니라 **open-vocabulary 3DGS pipeline들의 공통 failure interface**가 된다.

강한 negative:

- VALA가 phantom/easy를 동시에 크게 개선한다.
- StS가 missing/association failure 없이 phantom을 거의 제거한다.
- 이 경우 taxonomy를 수정해야 한다. 하지만 그래도 좋은 결과다. P2 method는 VALA/StS의 성공 요인을 흡수하거나, 우리 문제 정의를 더 좁혀야 한다.

가장 가능성 높은 중간 결과:

- VALA는 robust/gating으로 easy는 안정적이지만, few-view/minority/confuser case가 남는다.
- StS는 object boundary는 좋아지지만, object discovery와 CLIP association이 새 병목이 된다.
- 이 결과가 나오면 P1의 메시지는 아주 선명해진다: **기존 처방들은 실패 위치를 줄이거나 이동시키지만, query-specific risk를 측정하고 abstain/guard하는 장치가 없다.**

논문에서의 최종 문장 후보:

> **Existing works fix averaging; we characterize when averaging is the right inductive bias and when it is the failure mode.**

> **Robust aggregation preserves consensus but misses minority evidence; selection recovers minority evidence but breaks consensus. The missing component is not another aggregator, but a query-conditioned guard.**

---

## 8. 다음 액션

1. VALA one-scene dry run용 branch/script를 만든다.
2. `ramen` 기준으로 data symlink/path compatibility를 맞춘다.
3. CUDA extension import/build만 먼저 확인한다.
4. native output이 나오면 `output/diagnostics/p1e_vala_*.csv`로 prompt-level table을 만든다.
5. StS는 preprocessed dataset 확보 여부를 먼저 확인한다.

---

## 9. 업데이트 로그

| 날짜 | 변경 | 한 줄 |
|---|---|---|
| 2026-06-15 | 초기 source audit 작성 | VALA/StS 레포, 논문 주장, README, 핵심 코드 entrypoint, diagnostic adapter, go/no-go 기준 정리 |
| 2026-06-15 | **regime-confusion framing 반영** | P1-E 를 "외부 코드 재현"에서 **기존 average-fix 가 어떤 regime 을 고치고 어떤 regime 을 놓치는지 측정하는 실험**으로 격상. VALA 는 consensus stabilization vs minority recovery, StS 는 object-first escape vs missing/association/granularity failure 로 재정의 |
| 2026-06-15 | **상세 실험 매트릭스 추가** | VALA V0-V6, StS S0-S6, cross-paper summary, 중단 기준, 최소 산출물 세트를 추가해 실제 실행 계획으로 구체화 |
| 2026-06-15 | **리뷰 반영 (Claude)** | §0.3 + §6.0 **dump 기반 faithful regime split 신설** (GPU 0, real-VALA 투입의 결정 게이트). **V6 = mechanism 본실험 / V3 = descriptive(confounded)** 명시 + V0 에 mean-variant 추출 점검 추가. **V4 = V3 의 분류기** (THGS-class × VALA-native cross-tab = paradigm 증거). **StS S5(eval-mismatch) validity gate 를 S2/S3 앞으로**. §5 operational 정의 고정 규율 추가 |
