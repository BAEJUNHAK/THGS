# VALA Waldo paper-protocol forensic

> 생성: 2026-06-19  
> 산출물: `output/diagnostics/p1e_vala_waldo_protocol_forensics.csv`  
> 목적: Waldo Kitchen paper 2D number 약 0.651이 public fixed actual인지, tuned/adaptive threshold인지, oracle upper bound에 가까운지, 또는 artifact mismatch인지 분리한다.

---

## 1. 결론 먼저

현재 공개 artifact와 public fixed protocol만으로는 Waldo Kitchen 2D paper number `0.651`을 설명하지 못한다.

가장 paper-like한 공개 재현 조건인 `official_train_officialsam_waldo__waldo_kitchen`에서:

| protocol | GT threshold? | GT level? | mIoU | paper 0.651 대비 |
|---|---:|---:|---:|---:|
| official saved actual @0.5 | no | no | **0.5412** | -0.1098 |
| official saved actual @0.4 | no | no | **0.5575** | -0.0935 |
| scene-best fixed threshold | yes, scene-level | no | **0.5576** | -0.0934 |
| VALA dynamic threshold | no | no | **0.3433** | -0.3077 |
| per-row level oracle @0.5 | no | yes | **0.5737** | -0.0773 |
| per-row threshold+level oracle | yes, per-row | yes | **0.6865** | +0.0355 |

P1V-M1에서 쓰는 `refersplat_3dgs_valafeat_full__waldo_kitchen` 조건에서는:

| protocol | GT threshold? | GT level? | mIoU | paper 0.651 대비 |
|---|---:|---:|---:|---:|
| official saved actual @0.5 | no | no | **0.4702** | -0.1808 |
| official saved actual @0.4 | no | no | **0.4839** | -0.1671 |
| scene-best fixed threshold | yes, scene-level | no | **0.4871** | -0.1639 |
| VALA dynamic threshold | no | no | **0.2395** | -0.4115 |
| per-row level oracle @0.5 | no | yes | **0.5323** | -0.1187 |
| per-row threshold+level oracle | yes, per-row | yes | **0.6538** | +0.0028 |

따라서 forensic 판정은 다음이다.

1. `0.651`은 public fixed actual @0.5/@0.4로는 설명되지 않는다.
2. 단일 scene-level best fixed threshold도 paper number에 도달하지 못한다.
3. 공개 `compute_dynamic_threshold`를 LERF 2D feature map에 적용하면 오히려 낮다. 이 함수가 Waldo 2D paper number를 설명한다는 증거는 없다.
4. paper number와 가장 가까운 것은 per-frame/prompt threshold oracle이다. 그러나 이것은 GT를 쓰는 diagnostic upper bound이므로 leaderboard 성능으로 보고하면 안 된다.
5. 그러므로 현재 가장 안전한 결론은 **public artifact/protocol mismatch 또는 저자-side artifact 차이 가능성이 남는다**이다. “paper가 oracle을 썼다”라고 단정하면 안 된다.

---

## 2. Public code protocol 확인

`external_methods/VALA/eval/evaluate_iou_loc.py`의 LERF 2D evaluator는 각 level의 IoU를 계산한 뒤, GT IoU가 아니라 relevance score로 level을 고른다.

- level별 mask와 IoU 계산: `evaluate_iou_loc.py:127-143`
- level score 계산: `evaluate_iou_loc.py:145-148`
- chosen level: `evaluate_iou_loc.py:149`
- saved chosen mask: `evaluate_iou_loc.py:154-156`
- CLI default threshold는 `0.4`: `evaluate_iou_loc.py:320`

즉 public 2D code는 **GT level oracle이 아니다**. 다만 paper Appendix의 2D threshold가 `0.5`라고 기록되어 있어, 우리는 both @0.5와 @0.4를 모두 확인했다.

3D render code도 같은 방향이다.

- `_stochastic_gate.pth` checkpoint hardcode: `render_lerf_by_text_langsplat.py:238-239`, `265-266`
- prompt별 score stack argmax로 level 선택: `render_lerf_by_text_langsplat.py:243-248`

VALA의 dynamic threshold 함수는 `external_methods/VALA/eval/eval_utils.py:100-166`에 있다. 이 함수는 level별 normalized relevance map에서 threshold stability를 보고 level/threshold를 고른다. 하지만 LERF 2D public evaluator에는 연결되어 있지 않고, 이번 forensic에서도 Waldo 2D gap을 설명하지 못했다.

---

## 3. Protocol별 해석

### 3.1 Official saved actual

Leaderboard에 쓸 수 있는 숫자는 이것뿐이다.

- `official_train_officialsam_waldo`: 0.5412@0.5, 0.5575@0.4
- `refersplat_3dgs_valafeat_full`: 0.4702@0.5, 0.4839@0.4

둘 다 paper 0.651보다 낮다.

### 3.2 Scene-level best fixed threshold

Waldo scene 전체에 대해 threshold를 하나 고르는 GT-tuned diagnostic이다.

- `official_train_officialsam_waldo`: best fixed threshold 0.40, mIoU 0.5576
- `refersplat_3dgs_valafeat_full`: best fixed threshold 0.45, mIoU 0.4871

이것도 paper 0.651에 못 간다. 따라서 단순히 “paper는 0.4 threshold였나?”만으로는 갭을 설명하지 못한다.

### 3.3 VALA dynamic threshold

`compute_dynamic_threshold`는 GT-free라서 원칙적으로 actual protocol 후보가 될 수 있지만, 이번 Waldo LERF 2D 적용에서는 낮았다.

- `official_train_officialsam_waldo`: 0.3433
- `refersplat_3dgs_valafeat_full`: 0.2395

로그에서도 many prompts에 대해 stable region이 없다는 warning이 나왔다. 따라서 이 함수가 paper Waldo 2D number의 hidden protocol일 가능성은 현재 증거로 약하다.

### 3.4 Per-row oracle

Per-row oracle은 paper number와 가장 가깝다.

- `official_train_officialsam_waldo`: threshold+level oracle 0.6865
- `refersplat_3dgs_valafeat_full`: threshold+level oracle 0.6538

그러나 이 protocol은 frame/prompt마다 GT IoU를 보고 level과 threshold를 고른다. 따라서 이것은 **diagnostic upper bound**이지, 공정한 actual output이 아니다.

논문에서 말할 수 있는 형태:

> Public fixed VALA outputs are substantially below the reported Waldo 2D number. The reported number is closer to a per-query upper bound than to the public fixed-threshold protocol in our artifact, but this does not prove that the paper used oracle selection; it indicates an unresolved protocol/artifact mismatch.

논문에서 말하면 안 되는 형태:

> VALA paper used oracle evaluation.

---

## 4. P1 문제정의와의 연결

이 forensic은 VALA 공격용이 아니라 공정성 보호장치다.

우리는 VALA를 낮게 보이게 하려고 protocol을 고르는 것이 아니라, paper number와 public actual 사이의 gap을 분리했다.

P1에 필요한 결론은 다음이다.

- Leaderboard claim에는 public actual 또는 common harness actual만 사용한다.
- Oracle은 “VALA feature family 안에 recoverable evidence가 있었는가”를 보는 diagnostic으로만 사용한다.
- Waldo paper number는 현재 public artifact mismatch flag를 달아야 한다.
- `refersplat_3dgs_valafeat_full`의 threshold oracle 0.6538이 paper 0.651과 가까운 것은 흥미롭지만, 이 숫자는 GT oracle이므로 method 성능으로 쓰면 안 된다.

따라서 P1의 안전한 문장은 다음이다.

> VALA partially mitigates visibility/aggregation symptoms, but public actual outputs and VALA-native oracle gaps show that query-time level/threshold evidence selection remains a distinct unresolved axis. Waldo paper-number reproducibility remains artifact/protocol-sensitive and must be reported separately from P1 mechanism claims.
