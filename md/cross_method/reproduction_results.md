# Reproduction Results — THGS / ReLaGS LERF-OVS 완벽 재현 + 재현성 분석

> 생성 2026-06-18. 골: 각 논문이 명시한 프로토콜 + 각 논문 공식 코드로 재현 → 논문 수치 대조 → 큰 격차 prompt의 GT/프롬프트 문제 분석.
> 상위 문서: [paper_deep_analysis.md](paper_deep_analysis.md) §9 (재현성 정밀 분석), [reproduction_plan.md](reproduction_plan.md).
> 입력은 전부 기존 산출물(피처/체크포인트/NAG, 정본 검증 완료) 재사용 — 재학습/재인코딩 금지.

---

## Phase 0 — 프로토콜 공부 (논문 명시 vs 코드 실제동작)

세 평가경로를 코드로 직접 검증. 핵심: **세 method가 서로 다른 자(채점기)로 잰다.**

### 0.1 세 평가경로 대조표

| 축 | THGS 릴리즈 (`test_lerf.py`+`eval_seg.py`) | LangSplat 원본 ("LERF protocol", `evaluate_iou_loc.py`) | ReLaGS (`lerf_ovs_diagnostic_native.py`) |
|---|---|---|---|
| **출력 표현** | superpoint presence mask 렌더 (binary) | per-Gaussian CLIP feature, **3 SAM scale** relevancy map | superpoint presence mask 렌더 (in-memory) |
| **level/scale 선택** | NAG level `[2,3]` **union** + 전역 **top-3** SP ([test_lerf.py:57](../../test_lerf.py#L57), [nag_data.py:21-52](../../nag_data.py#L21-L52)) | 3 scale 중 **peak relevancy argmax** (IoU-blind, [eval...iou_loc.py:145-152](../../external_methods/VALA/eval/evaluate_iou_loc.py#L145)) | hierarchical **root→child descent** + topk=5 (Algorithm 1) |
| **mask threshold** | 렌더 mask `>0.5` → PNG `>128` ([test_lerf.py:62](../../test_lerf.py#L62), [eval_seg.py:62](../../scripts/eval_seg.py#L62)) | normalize→`>0.4` + **smoothing**(avg filter scale30, `0.5*(avg+orig)`) ([…:111,134,320](../../external_methods/VALA/eval/evaluate_iou_loc.py#L134)) | render `>thresh` (diagnostic 기본 0.5) |
| **집계 순서** | per-prompt→**per-image 평균**→per-scene→4-scene 평균 ([eval_seg.py:67-78](../../scripts/eval_seg.py#L67)) | **flat per-query mean** (전 scene·frame·prompt 인스턴스 동일가중, [...:284](../../external_methods/VALA/eval/evaluate_iou_loc.py#L284)) | **per-prompt** (prompt가 등장한 frame 평균→prompt 동일가중) |
| **IoU 식** | `tp/(tp+fp+fn+1e-6)` | `∩/∪` | `∩/∪` |
| **mAcc/acc 정의** | **픽셀** `(tp+tn)/전체` (TN 포함→과대) | **point-in-bbox localization** (max-relevancy 점이 GT bbox 안?) | LERF엔 미사용 (ScanNet 전용) |
| **GT** | `data/lerf_ovs/label/<scene>/*.json` category 폴리곤, multi-instance union | 동일 json, category, polygon+bbox, 동일 label은 stack | 동일 label |
| **mask_thresh 기본** | 0.5 | **0.4** | 0.5 |

### 0.2 논문이 *명시한* 것 vs 코드 (미명시 항목)

- **THGS 논문 §3.5**: "follow LERF protocol", Eq.(12) canon-contrast, "top-ranked superpoints", "single or jointly across multiple levels", "thresholded at 0.5". → **명시**: scoring식·0.5 threshold·multi-level 가능성. **미명시**: 정확히 `level=[2,3]`인지·`topk=3`인지·집계순서(per-image)·mAcc 정의. ⇒ 릴리즈 코드의 상수가 paper Table 1 생성 설정과 같은지 **논문에 근거 없음** (이번 재현의 핵심 검증대상).
- **ReLaGS 논문**: Algorithm 1 (root top-K=5 → child descent → <1% parent filter → max-gap cut). "follow LangSplat ... OpenCLIP ViT-B/16" (encoder 한정). → **LERF-OVS 평가 프로토콜/라벨/코드를 THGS·LangSplat과 동일하게 쓴다는 명시 없음**; Table 3에 THGS 공개수치를 그대로 인용. **공식 eval 코드 미공개**([ReLaGS/README.md:21](../../ReLaGS/README.md) "published soon").
- **LangSplat**: 위 코드가 정본. 3-scale peak 선택 + flat-per-query + 0.4 + smoothing + localization-acc.

### 0.3 Phase 0 결론 (재현 전 사전 진단)

1. **THGS의 릴리즈 평가는 자기가 인용한 LangSplat "LERF protocol"과 4개 축(level선택·threshold·집계·acc정의·smoothing)에서 다르다.** THGS는 superpoint presence mask를 렌더하므로 `evaluate_iou_loc.py`(3-scale per-pixel feature 기대)를 **그대로 통과시킬 수 없음** — 즉 THGS는 자체 평가(test_lerf+eval_seg)가 정본 경로. "LERF protocol"은 scoring/threshold 아이디어 차용 수준.
2. **THGS↔ReLaGS는 현재 다른 채점기**(per-image vs per-prompt)로 측정됨 → 한 표 비교 불가. Phase 3(단일 하네스)이 필수.
3. **ReLaGS 공식 eval 미공개** → Phase 2는 (a) ReLaGS Algorithm 1 inference + (b) THGS의 test_lerf 렌더 골격 + eval_seg 채점으로 단일 하네스 재현(가장 충실 대체).

> 다음: Phase 1 (THGS 재현 + level/topk/threshold/집계 ablation, GPU) — job 84 종료 후 예약 큐에서 자동 실행.

---

## 실행 상태 (2026-06-18, 예약 cal_jhbae_rtxpro5000_c3a3b6da300d)

SLURM 잡 체인 (예약에 큐잉, 사용자 job 84 종료 후 자동 실행):

| JobID | 이름 | 상태 | 내용 | 산출물 |
|---|---|---|---|---|
| 84 | vala-2d-all | RUNNING (사용자) | — GPU 점유중 | — |
| **86** | repro-thgs | PENDING(Resources) | Phase 1: `scripts/repro/render_thgs_ablate.py` (6 config: L23/L2/L3 × k3, L23×k1/5/10) → soft mask → `eval_thgs_ablate.py` (thresh×집계 sweep) | `output/render/repro/thgs/`, `output/diagnostics/repro_thgs_ablation.csv` |
| **87** | repro-relags | PENDING(afterok:86) | Phase 2/3: `ReLaGS/scripts/render_relags_eval.py` (faithful Algorithm-1 level_until=1 topk=5) → soft mask → `eval_soft.py` 단일 하네스 (ReLaGS vs 64.4, THGS vs 54.94 동일 채점기) | `output/render/repro/relags/`, `repro_{relags,thgs}_singleharness.csv` |

검증 목표:
- **Phase 1**: THGS released-default(L23_k3, thresh0.5, per-image)가 58.87 재현하는지(sanity) + **54.94를 내는 (level,topk,thresh,집계) 설정이 존재하는가**. teatime +13 격차가 어느 손잡이에서 닫히는지.
- **Phase 3**: THGS·ReLaGS를 **동일 evaluator(soft→thresh→IoU→집계)**로 재채점 → apples-to-apples 표. per-image/per-prompt 병기.
- 재현 즉시 Phase 4 (큰 격차 prompt의 GT/프롬프트 진단) + 본 문서에 재현표·판정 기록.

**완료 (2026-06-18, jobs 90+91, 노드 jhsong-TRX50-AI-TOP).** 버그 2개 수정 후 성공: ① 스크립트가 서브폴더라 `PYTHONPATH=repo루트` 필요 ② HF 캐시가 점유불가 `/home/meaabebe`로 가던 것 → NAS `.valahome/hf` + `HF_HUB_OFFLINE=1`. 결과 ↓.

---

## Phase 1 — THGS 재현 + (level, topk) × threshold × 집계 ablation

### 1.1 config sweep (per-image, thresh 0.5) vs 논문 Table 1

| config | overall | figurines | ramen | teatime | waldo | 비고 |
|---|---|---|---|---|---|---|
| **논문 THGS** | **.5494** | .5730 | .4346 | .6833 | .5065 | (저자 미공개 빌드) |
| **L23_k3 (릴리즈 기본)** | **.5887** | .5493 | .4214 | **.8187** | .5654 | released 정확 재현, **Δpaper +3.93** |
| L2_k3 | .5347 | .4366 | .4434 | .7280 | .5306 | overall 근접하나 per-scene 패턴 불일치 |
| L3_k3 | .4872 | .5270 | .2537 | .6487 | .5192 | teatime는 paper 근접, ramen 붕괴 |
| L23_k1 | .4800 | .5130 | .2847 | .6604 | .4617 | |
| L23_k5 | .4928 | .5078 | .2429 | .6742 | .5462 | |
| L23_k10 | .5086 | .4955 | .3165 | .6511 | .5710 | teatime paper 근접, fig/ramen 미달 |

- **threshold 효과 미미** (0.3~0.6에서 overall ±1~2pt). **집계: per-image > per-prompt ≈ +4pt** (예: L23_k3 per-prompt=.5464 ≈ 논문 overall .5494 — 단 이는 fig 손해 ↔ teatime 이득의 **상쇄 우연**, per-scene은 안 맞음). flat은 둘 사이.

### 1.2 판정 — **54.94 재현 설정은 존재하지 않음 (released ≠ paper)**

- 릴리즈 기본(L23_k3)은 **fig/ramen은 논문에 근접**(−2.4/−1.3)하지만 **teatime을 +13.5, waldo를 +5.9 과잉공급**. teatime을 논문값(.683)에 맞추려면(L3_k3 .649 / L23_k10 .651) **fig/ramen이 동시에 붕괴** → **어떤 (level,topk,thresh,집계) 조합도 4 scene을 논문 패턴으로 동시 재현 못 함**.
- ⇒ **paper Table 1은 현재 공개된 체크포인트+공개 inference 상수로 도달 불가**. 우리 측 평가/aggregation 문제가 아니라 (Phase 0에서 집계불변·scene-directional 확인) **THGS 자신의 released-vs-paper 재현 격차**로 기계적 확정. (LangSplat 원본 `evaluate_iou_loc.py`는 3-scale per-pixel feature를 기대 → superpoint presence mask인 THGS엔 직접 적용 불가, 그래서 THGS-side analog인 level/topk sweep으로 검증.)
- **teatime 과잉은 GT 문제 아님** — 14 prompt 중 12개가 IoU ≥0.78 (stuffed bear .97 / sheep .96 / yellow pouf .97 / plate .93 / paper napkin .94…). 릴리즈 체크포인트가 teatime(크고 잘 분리된 객체)을 level[2,3] union으로 깨끗이 회수. = released 빌드가 paper 빌드보다 teatime에서 강함.

---

## Phase 3 — 단일 하네스 (THGS vs ReLaGS, **동일 evaluator**: soft→thresh0.5→IoU→per-image)

| method | overall | figurines | ramen | teatime | waldo |
|---|---|---|---|---|---|
| **THGS** (L23_k3) | **.5887** | .5493 | .4214 | **.8187** | .5654 |
| **ReLaGS** (faithful Alg-1, level_until=1 topk=5) | **.6165** | **.6467** | .4742 | .7390 | **.6062** |
| **Δ (ReLaGS − THGS)** | **+.0278** | +.097 | +.053 | **−.080** | +.041 |

- ReLaGS = **단일 하네스에서 .6165 = 기존 진단 61.65 정확 재현**; **fig .6467 / waldo .6062 = 논문 64.7/60.6 소수점 일치**(충실 anchor), ramen/teatime만 미달(−3.8/−6.6 = phantom).
- **⚠ 이전 우려 정정**: "THGS 58.87(per-image) vs ReLaGS 61.65"는 apples-to-oranges가 **아니었음** — ReLaGS도 per-image였고(per-prompt는 .5807), 이번에 **완전히 동일한 코드 경로로 재확인**. 단 두 method가 *서로 다른 스크립트 경로*(test_lerf+eval_seg vs in-memory diagnostic)였던 것을 이번에 단일 하네스로 통일 → 비교 유효.
- **ReLaGS > THGS (+2.78)**, 단 **teatime은 THGS가 +8.0 우세**(릴리즈 THGS teatime의 이상 강세 때문). 논문 방향(ReLaGS>THGS, paper +9.4)과 일치하나 격차는 released THGS teatime 인플레로 축소.
- **표 B base (출처-clean, 동일 하네스)**: THGS .5887 / ReLaGS .6165. drop-in 이득은 "이 base 위 Δ"로.

---

## Phase 4 — 큰 격차 데이터의 prompt/GT 진단

### 4.1 Multi-instance generic prompt (GT=N 인스턴스 union, retrieval=1 → 구조적 저-IoU)

| scene | prompt | #GT인스턴스 | THGS | ReLaGS |
|---|---|---|---|---|
| ramen | plate | 2 | .000 | .000 |
| ramen | bowl | 2 | .148 | .155 |
| ramen | sake cup | 2 | .170 | .042 |
| ramen | spoon | 2 | .219 | .336 |
| ramen | napkin | 2 | .445 | .196 |
| waldo | knife | **5** | .274 | .233 |
| waldo | spoon | 2 | .000 | .000 |
| teatime | hooves | 3 | .000 | .000 |
| teatime | three cookies | 2 | .916 | .920 |
| figurines | pink ice cream | 2 | .914 | .889 |

- **ramen은 14 prompt 중 5개가 multi-instance** — ramen이 양 method·양 논문에서 낮은 **구조적 원인**. generic 단어("plate/bowl/spoon/sake cup")가 여러 인스턴스에 매핑되는데 GT는 union, 채점은 한 클러스터만 → recall 상한이 ~1/N. (단 three cookies/pink ice cream처럼 인스턴스가 인접·동질이면 union이 한 덩어리라 영향 작음 → IoU 높음.)
- **hooves(3), spoon(waldo,2)**: 양 method 0 — 분산된 동일 카테고리 다수 인스턴스를 prompt 하나로 잡아야 하는 ambiguous case.

### 4.2 TINY GT (작은 객체) — method 의존

- figurines **pirate hat**(area 0.23%): THGS .000 / ReLaGS .899 — THGS-specific phantom (GT 문제 아님). **miffy**(0.22%) THGS .000/ReLaGS .476, **pumpkin** THGS .000/ReLaGS .721 — figurines에서 ReLaGS 우세(64.7>54.9)의 원천.
- 반례: waldo ketchup(0.39%)/plastic ladle(0.5%)/spatula(0.34%)는 THGS도 .83~.90 → **작다고 항상 실패 아님**. ⇒ TINY 자체보다 phantom(소수파 매장)이 원인.

### 4.3 method 간 큰 prompt 격차 (= 메커니즘 증거, GT 문제 아님)

- ReLaGS 압승: ramen **kamaboko** (THGS .064→ReLaGS .882), waldo **yellow desk** (.000→.939), **pour-over vessel** (.000→.870), figurines pirate hat/pumpkin/miffy.
- THGS 우세: teatime **coffee mug** (.784 vs ReLaGS .000), waldo **dark cup** (.825 vs .154), figurines tesla door handle (.275 vs .000).
- ⇒ 두 method의 partition/ROFA 차이로 phantom 회수 대상이 다름. **재현 실패가 아니라 method 차이** (thesis의 phantom 증거).

---

## 최종 판정

1. **THGS**: 릴리즈 체크포인트(byte-identical) + 릴리즈 inference(L23_k3/0.5/per-image)로 **overall .5887 재현(=released)**. **논문 .5494는 공개 artifact로 재현 불가** — teatime을 맞추는 어떤 설정도 fig/ramen을 깨므로 per-scene 동시 재현 불가 ⇒ **THGS released≠paper-table 빌드 격차로 확정**. 인용 규칙: "publicly released THGS, official protocol = **58.87** (paper 54.94 = unreleased build)".
2. **ReLaGS**: 단일 하네스에서 **.6165 재현**, fig/waldo 논문 소수점 일치(충실), ramen/teatime 미달은 **phantom**(재현 실패 아님).
3. **단일 하네스 비교 확정**: THGS .5887 / ReLaGS .6165 (+2.78), 동일 evaluator. 표 B base로 사용 가능.
4. **prompt 진단**: 큰 격차의 GT-side 원인 = **ramen 다수의 multi-instance generic prompt**(구조적 저-IoU, 벤치마크 한계). teatime 과잉·figurines 격차는 GT 문제가 아니라 **체크포인트/메커니즘** 차이.

### 산출물
- `output/diagnostics/repro_thgs_ablation.csv`, `repro_thgs_singleharness.csv`, `repro_relags_singleharness.csv`
- 렌더: `output/render/repro/{thgs,relags}/`
- 스크립트: `scripts/repro/{render_thgs_ablate,eval_thgs_ablate,eval_soft}.py`, `ReLaGS/scripts/render_relags_eval.py`, `scripts/repro/job_{thgs,relags}_repro.sh`

---

## Phase 5 — 공정 비교 (동일 2DGS substrate) — 2026-06-19, job 95

> 동기: THGS vs ReLaGS 절대 비교가 "다른 2DGS" 때문에 불공정할 수 있다는 우려 → **THGS의 학습된 2DGS 하나 위에 두 파이프라인을 다 돌려** substrate를 통제. ReLaGS 파이프라인(prune→partition→graph_weight→partition→ROFA merge)은 전부 비학습 후처리라 가능. 스크립트 `scripts/repro/job_fair_relags_on_thgs.sh` (모델 `output/repro_fair/relags_on_thgs/<scene>`, THGS 릴리즈 무손상). ⚠ ReLaGS vendored SPT는 PyG2.3 `keys` property 미패치라 5곳 수술 패치(ext/spt 4 + sp_partition.py).

### 5.1 ★ 2DGS는 공유였다 — prune 후 가우시안 4/4 정확 일치

| scene | THGS 2DGS | prune(0.0005) 후 | ReLaGS 릴리즈 | 일치 |
|---|---|---|---|---|
| figurines | 369,120 | **234,534** | 234,534 | ✅ |
| ramen | 400,401 | **372,270** | 372,270 | ✅ |
| teatime | 1,093,781 | **958,840** | 958,840 | ✅ |
| waldo | 1,062,260 | **905,012** | 905,012 | ✅ |

→ **ReLaGS-2DGS = THGS-2DGS를 contribution 0.0005로 prune한 것**으로 기계적 확정. **두 method는 같은 2DGS geometry를 공유**한다. ReLaGS의 가우시안 차이는 *더 나은 backbone*이 아니라 *자기 method의 pruning 단계*. ⇒ **기존 published 비교(THGS 58.87 vs ReLaGS 61.65)는 이미 동일-2DGS 공정 비교였음** (이제 증명됨). NAG 구조도 재현: figurines 재실행 SP/level [5052,2097,456,185] ≈ 릴리즈 [5081,2103,463,177].

### 5.2 ⚠ 그러나 파이프라인은 비결정적 (released = 단일 샘플)

`graph_weight.py:42`가 SAM 세그먼트별 20-dim 인코딩을 **`torch.normal(0,1)`로 seed 없이 랜덤 init** → edge 가중 → cut-pursuit 파티션 → NAG가 **매 실행마다 달라짐**. (THGS도 동일 코드 계보 → 동일 비결정성.)

단일 하네스(per-image, t0.5) 비교:

| arm | 2DGS | overall | fig | ramen | teatime | waldo |
|---|---|---|---|---|---|---|
| THGS-pipe (released) | THGS 369k | **.5887** | .549 | .421 | .819 | .565 |
| ReLaGS-pipe (released) | =THGS prune 234k | **.6165** | .647 | .474 | .739 | .606 |
| **ReLaGS-pipe (내 재실행)** | =THGS prune 234k | **.5667** | .542 | .411 | .732 | .583 |

- 가우시안은 릴리즈와 정확히 같은데(234k) 재실행 mIoU는 **−5.0pt(.5667 vs .6165)**, figurines는 **−10.5pt**(.542 vs .647). 원인 = graph_weight 랜덤 init + cut-pursuit stochasticity (문서화된 run_lerf.sh 파라미터 그대로 사용). **THGS released≠paper(58.87 vs 54.94)와 같은 종류의 "released ≠ faithful 재실행" 격차** — 양 method 모두 released 숫자는 비결정 파이프라인의 (유리한) 단일 draw.

### 5.3 판정 + 권고

1. **공정성 우려 해소**: 2DGS 공유 확정(4/4 exact). published THGS vs ReLaGS는 동일 substrate 비교가 맞고, ReLaGS 우위는 geometry가 아니라 **pipeline(prune+ROFA+partition) 기여**.
2. **단, 절대 숫자는 단일 샘플**: 파이프라인 비결정성 때문에 한 번의 재실행으로 released를 대체하면 안 됨. **표 B의 fair number는 두 method의 released artifact를 단일 하네스로(THGS .5887 / ReLaGS .6165) 쓰되 "동일 THGS-2DGS 기반, ReLaGS는 prune" 캡션 + 비결정성 caveat** 명시.
3. **진짜 rigorous fair 비교(논문용)**: 같은 THGS-2DGS(369k)에서 **두 파이프라인을 각 K seed로 재실행 → mean±std 비교**가 substrate·run-variance 둘 다 통제하는 gold standard. (다음 단계 후보, GPU 필요.)

> **열린 결정 (사람 판단 대기, 2026-06-19)**: 표 B를 (a) released-artifact 공정표(THGS .5887 / ReLaGS .6165 + 비결정성 caveat, 지금 사용 가능) 로 갈지, (b) K-seed 분포 실험(두 파이프라인 × K seed × 동일 THGS-2DGS → mean±std)을 추가로 돌릴지. (b)는 graph_weight 비결정성까지 통제하는 gold standard이며 예약 GPU 필요. ReLaGS 쪽 인프라는 준비됨(job_fair_relags_on_thgs.sh), THGS 파이프라인의 동일-2DGS K-seed 재실행 잡만 추가하면 대칭 완성.

### 5.4 ★ 엄밀한 seeded 공정 비교 (2026-06-19, job 99) — 핵심 결과

**설계**: 같은 THGS-2DGS 위에서 **두 파이프라인을 모두 재실행** (이전 5.2는 ReLaGS만 재실행한 비대칭 → 교정), graph_weight의 unseeded `torch.normal`에 `PIPELINE_SEED` 주입해 seed{0,1,2}로 **동일 seed**에서 양쪽 실행. 각 codebase 문서화 호출(THGS launcher / ReLaGS run_lerf.sh) 미러링. 단일 하네스(per-image t0.5).

| | overall | figurines | ramen | teatime | waldo |
|---|---|---|---|---|---|
| **THGS-pipe** | **.5696** ±.0024 | .531 | .393 | .787 | .568 |
| **ReLaGS-pipe** | **.5701** ±.0030 | .541 | .425 | .732 | .583 |

**Paired Δ (ReLaGS−THGS) seed별**: −.0029 / +.0017 / +.0027 → **Δ평균 +.0005 ± .0024 (통계적 동률)**.

**판정 (이전 5.2 stochasticity 가설 교정)**:
1. **공정 조건(같은 2DGS+같은 seed+같은 채점기+문서화 config)에서 THGS ≈ ReLaGS** — 논문 +9.4도, released +2.78도 **공정 비교에서 ~0으로 소멸**. "ReLaGS 우위"는 method 본질이 아니라 substrate/build/protocol 비대칭의 산물.
2. **seed std ≈ .002–.003 (거의 0)** → 파이프라인은 결과적으로 **안정적**. 5.2의 단일 재실행 .5667은 "나쁜 draw"가 아니라 정상값(seeded .568/.568/.574). ⇒ **5.2의 "graph_weight stochasticity가 5pt 격차 원인" 주장은 틀렸음, 본 절로 교체.**
3. 진짜 격차 = **문서화된 공개 config 재실행(~.570) vs 저자 released 빌드(THGS .589 / ReLaGS .617)** = 둘 다 released가 공개 config보다 높은 **체계적** 격차(ReLaGS −4.6 > THGS −1.9). THGS released≠paper(54.94)와 동종의 "released≠공개재현".
4. per-scene 무역: ReLaGS가 ramen(+3.2)/figurines(+1.0)/waldo(+1.5), THGS가 teatime(+5.5) → 상쇄. ReLaGS pruning이 ramen엔 +, teatime엔 −.

**논문 함의**: 표 B의 "fair leaderboard"는 **동일 2DGS·동일 seed·문서화 config 재현**으로 보고하면 THGS·ReLaGS 동률 → 우리 method의 drop-in Δ가 base 차이 노이즈에 묻히지 않고 순수하게 드러남. published 절대값(ReLaGS 64.4 등)은 "release-build, 공개 config 미재현"으로 명시.

### 산출물 (Phase 5)
- seeded: `output/render/repro/seedrun/{THGS,RELAGS}_s{0,1,2}/`, 모델 `output/repro_fair/seedrun/{thgs,relags}_s{seed}/`, 잡 `scripts/repro/job_seedrun.sh`, 집계 `scripts/repro/aggregate_seedrun.py`, 렌더 `scripts/repro/render_thgs_single.py`. seed 주입: `{,ReLaGS/}graph_weight.py` (PIPELINE_SEED).
- `output/diagnostics/repro_fair_relags_on_thgs2dgs.csv`, 렌더 `output/render/repro/relags_on_thgs2dgs/`, 모델 `output/repro_fair/relags_on_thgs/<scene>/sai_nag_3.0.pt`
- 잡 `scripts/repro/job_fair_relags_on_thgs.sh` (자체완결: 셋업+5단계+render+eval, 완료 scene skip)
- 패치: `ReLaGS/ext/spt/{data/data,transforms/data,transforms/partition,transforms/point}.py` + `ReLaGS/sp_partition.py` (PyG2.3 keys property)

