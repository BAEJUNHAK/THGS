# 공정성 프로토콜 — fairness 트랙의 헌법 (living document)

> 생성 2026-06-18. 짝꿍: [README.md](README.md) (트랙 인덱스·상태) · [preregistration_log.md](preregistration_log.md) (RF 사전등록) · [intuition.md](intuition.md) (두 법정 비유).
>
> 이 문서의 역할: THGS·ReLaGS·VALA(및 향후 외부 method)를 **공정하게** 비교하기 위한 단일 기준. 모든 fairness stage 문서는 이 문서를 인용한다. 형식은 [extended_failure_hypotheses.md](../extended_failure_hypotheses.md)의 카탈로그 형(번호 섹션 + dated patch)을 따른다.

---

## §0. 3-레이어 모델 + 3-주장 유형

오픈보캡 3D 세그 method = 사슬:
```
데이터/COLMAP → geometry(2DGS/3DGS) → SAM마스크+CLIP(2D feature)
   → lifting/aggregation(3D에 feature 올리기) → query 선택(text→mask) → 평가(IoU vs GT)
```

이 사슬을 **3개 레이어**로 나눈다:

| 레이어 | 구성요소 | 본질 |
|---|---|---|
| **EVALUATION** | GT 폴리곤 · 프롬프트 세트 · metric 정의 · 집계 순서 | 채점 규칙. *어떤 비교든* 동일해야 함 |
| **SUBSTRATE** | 데이터/COLMAP · SAM 마스크 · CLIP 모델 | 공유 기반. 안 맞추면 backbone 덕인지 3D 방법 덕인지 모름 |
| **METHOD** | geometry repr · lifting/aggregation · query selection · rasterizer | 비교 대상(독립변수). 당연히 다름 |

**핵심 원리: "공정"은 하나가 아니라 *주장*에 종속된다.** 세 주장 유형:

- **Claim D — DESCRIPTIVE / NATIVE AUDIT** ("외부 method output을 우리 taxonomy로 보면 어떤 패턴이 있나"): 완전한 공정 비교나 원인 귀속이 아니라 **가설 생성용 관찰**. native evaluator, native output, cross-method class transfer를 쓸 수 있지만, 리더보드/메커니즘 결론으로 승격 금지. → VALA job84 prompt-level split이 여기에 속한다.
- **Claim A — LEADERBOARD** ("우리가 벤치마크에서 이긴다"): **EVALUATION 레이어만** 동일하면 됨. METHOD 차이는 정당(그게 비교 대상). SUBSTRATE는 "3D 방법의 우위"를 주장하려면 동일해야. → 모든 method의 *최종 렌더 mask*를 **단일 하네스**로 동일 채점, 단 임계는 *각 method 자기 best*(동일 규칙으로 선택).
- **Claim B — MECHANISM** ("평균/aggregation이 phantom 실패의 원인이다"): **연구 대상 1개 요소만 빼고 전부** 동일해야 함. → **공유 per-view dump** 위에서 *aggregation rule만* 교체. native cross-method 비교로는 **불가**.

---

## §1. 레이어 매트릭스 — 무엇을 맞춰야 하나

| 레이어 | Claim D (native audit) 동일 필요? | Claim A (리더보드) 동일 필요? | Claim B (메커니즘) 동일 필요? |
|---|---|---|---|
| **EVALUATION** (GT·프롬프트·metric·집계) | 기록 필수, 동일 불필요 | ✅ **필수** | ✅ **필수** |
| **SUBSTRATE** (데이터·SAM·CLIP) | 기록 필수, 동일 불필요 | △ "3D 방법 우위" 주장 시 필수 | ✅ **필수** |
| **METHOD** (geometry·lifting·aggregation·selection) | 기록 필수, 동일 불필요 | ❌ 달라도 됨 (비교 대상) | 연구 대상 1개 빼고 ✅ 필수 |

→ **현재 우리 상태**: SUBSTRATE의 CLIP은 셋 다 동일(ViT-B-16/laion2b_s34b_b88k/512d). 데이터/COLMAP 동일(`data/lerf_ovs`). METHOD는 다름(THGS·ReLaGS=2DGS+superpoint, VALA=3DGS+per-Gaussian) — 이건 *정당*. **유일한 실제 위반 = EVALUATION이 method마다 다른 코드**(§3).

---

## §2. 공유 EVALUATION 사양 (동결 대상)

단일 하네스가 모든 method에 적용할 *하나의* 평가:

- **GT**: `data/lerf_ovs/label/{scene}/*.json` (LangSplat LERF-OVS 폴리곤), `cv2.fillPoly(...,1)`로 래스터화. 한 프롬프트에 객체 여럿이면 union.
- **프롬프트**: GT JSON의 `category` 필드에서 추출 (VALA의 하드코딩 `scene_texts` 금지). **67개** (figurines 21 / ramen 14 / teatime 14 / waldo_kitchen 18).
- **프레임**: 각 scene의 label 디렉토리 *전체* 라벨 프레임 (VALA의 하드코딩 5-7 프레임 금지).
- **mIoU**: `tp / (tp + fp + fn + 1e-6)` (per-(prompt,frame) → 집계).
- **mAcc**: 정의 *하나* 고정 — 후보 ① 픽셀 정확도 `(tp+tn)/(tp+tn+fp+fn)` ② per-mask balanced. 택1(또는 둘 다 병기), 한 번 정하면 §4에 기록.
- **집계 순서**: *하나* 고정 (권장: per-image 평균 → per-scene 평균). VALA식 flat per-(frame,object) 평균과 섞지 말 것.
- **임계 정책**: 각 method를 *자기 best/calibrated threshold*에서 평가 (모두 동일 규칙으로 선택). ⚠️ **"모두에게 같은 임계 강요"는 그 자체로 불공정** — relevance 스케일이 method마다 다르므로.

---

## §3. 기록된 위반 — 현재 method별 평가의 *진짜* 숫자

(소스 코드 직접 확인. 이전 구두 요약의 "VALA 0.6"은 부정확했음 — 아래가 정확.)

| method | 이진화 | 집계 | mAcc/Acc | 프레임·프롬프트 | 코드 |
|---|---|---|---|---|---|
| **THGS** | 렌더 mask `>0.5`(test_lerf) → PNG `>128`(eval_seg) | per-image 평균 → per-scene 평균 | 픽셀 `(tp+tn)/전체` | label 전체 / JSON category | `test_lerf.py`, `scripts/eval_seg.py:10-31,62` |
| **ReLaGS** | 진단 기본 thresh 0.5 | (진단별) | — | label 전체 | `ReLaGS/scripts/lerf_ovs_diagnostic_native.py` |
| **VALA native 3D** | Gaussian relevance `mask_thresh` (paper/script 0.6) → silhouette PNG `load_image_as_binary(threshold=10)` | **flat per-(frame,object) `np.mean`** | **Acc@0.5 = count(IoU>0.5)/total** + Acc@0.25 | official `scene_texts` + test split | `external_methods/VALA/eval/render_lerf_by_text_langsplat.py`, `compute_lerf_iou.py` |
| **VALA official 2D** | feature-map relevance `mask_thresh` (paper Appendix: 0.5; code default 0.4) | **flat per-(frame,object) `np.mean`** | localization accuracy 별도 | JSON labels, our job84는 label frames/prompts 전체로 adapter | `external_methods/VALA/eval/evaluate_iou_loc.py`, `scripts/p1e_vala_official_2d_eval_all.py` |

**결론**: 세 method가 *서로 다른 자*로 잰다 — 이진화·집계·mAcc 정의·프레임/프롬프트 출처가 다름. (IoU 공식 자체는 동일.) → **단일 하네스로 재채점하기 전엔 한 표(리더보드)에 못 놓는다.** 이게 fairness 트랙이 고치는 위반.

**참고 — 이미 공정한 부분**: THGS↔ReLaGS *진단* 비교(`scripts/cross_method_comparison.py`, `b7_a4_oracle_analysis.py`)는 per-method oracle/rank + 공유 A2 ceiling + 동일 canon-contrast라 **공정**. VALA regime-split([../strategy/vala문제.md](../strategy/vala문제.md))과 job84 prompt split은 THGS 라벨 전이 + cross-pipeline이라 **Claim D descriptive audit**까지만 허용된다. mechanism 주장은 [p1_vala_protocol.md](p1_vala_protocol.md)의 VALA-native oracle/ablation을 요구한다.

---

## §4. 개정 로그 + EVAL-레이어 변경 로그

> `scripts/fairness/common/metrics.py`·`thresh_policy.py` 수정 시 여기에 기록 + MANIFEST의 `metric_version` 갱신 → 옛 CSV가 어느 metric으로 나왔는지 추적 가능.

| 날짜 | 변경 | metric_version |
|---|---|---|
| 2026-06-18 | 프로토콜 초안 (3층·2주장·§2 EVAL 사양·§3 위반 진짜숫자). 코드/config 미생성. | (미정) |
| 2026-06-19 | P1-VALA를 위해 Claim D(native descriptive audit)를 추가하고 VALA native 3D / official 2D 평가를 분리. | (미정) |

---

## §5. 두 불변식 (배너)

> 🚫 **leaderboard는 mechanism dump를 절대 읽지 않는다.** mechanism은 native cross-method 렌더를 절대 돌리지 않는다.
> 🔗 **`common/`은 코드만 공유한다 — 결과는 절대 공유하지 않는다.**
> ⚖️ 리더보드 표에서 끌어온 메커니즘 결론, 또는 native cross-method dump에서 끌어온 리더보드 주장 = **무효(mistrial)**.

---

## §6. 관련 문서

- [../strategy/vala문제.md](../strategy/vala문제.md) — VALA 재현 + 기각 가설 (이 트랙의 동기)
- [../strategy/p1e_external_code_study_plan.md](../strategy/p1e_external_code_study_plan.md) — P1-E 외부 코드 실측
- [../strategy/competitor_autopsy.md](../strategy/competitor_autopsy.md) — 경쟁 처방 부검(family rule)
- [../../cross_method/](../../cross_method/) — 기존 method 비교 narrative (fairness/는 *공정한* 리더보드/메커니즘 주장의 정본)
