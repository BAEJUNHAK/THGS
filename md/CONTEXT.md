# CONTEXT — 프로젝트 컨텍스트 원장 (새 레포 진입점 초안)

> 목적: 새 레포(A)로 옮긴 뒤 **AI/사람이 5분 안에 맥락을 잡게** 하는 단일 진입점.
> 규율: 이 문서는 **지도(index)** — 상세는 링크된 문서가 진실의 원천. 실험 1개 끝날 때마다 §4에 1칸 추가.
> 표기: ✅확정 / 🔬가설·미완 / ⚠️정정·주의. 최종 갱신 2026-06-20.

---

## 0. TL;DR (먼저 읽기)

- **무엇**: 학습 없이(training-free) 2DGS 장면에서 open-vocabulary 계층적 3D 분할. **목표 = top-conference 논문.**
- **Thesis**: OV-3DGS 실패의 핵심은 **소수파 매장(minority burial)** — 정답 CLIP 신호가 소수 view에만 강한데 multi-view 평균이 그걸 묻고, 경쟁 슈퍼포인트는 "coherent-plausible"해서 이김. 한 칸 고도화 = **regime confusion**: easy/consensus 쿼리는 평균이 맞고 phantom/minority-evidence 쿼리는 query-조건부 selection이 필요한데, **기존 method 전부 이 둘을 구분 못 함.**
- **Method (우리 기여)**: query-time **guarded hybrid** 채점기(어떤 training-free pipeline에도 drop-in). Stage4에서 held-out +3.74pt(phantom +17.5)지만 easy −4.08pt(R9 PARTIAL). 🔬 **P2 = 가드 재설계(미완, 다음 단계).**
- **부수 기여(이번 세션에서 강화)**: **재현성/공정성 감사** — published 리더보드 숫자는 공개물로 재현 불가, 공정 비교하면 THGS ≈ ReLaGS.
- **지금 단계**: Stage1~5 + P1 완료 → 재현/공정비교 완료 → **다음 = P2 가드(핵심 승부처)** 또는 fairness 트랙 정식화. venue/deadline 미정.
- **진실의 원천**: `md/hypotheses/` (가설·실험), `md/cross_method/reproduction_results.md` (재현·공정), `md/hypotheses/strategy/roadmap.md` (전략).

---

## 1. 용어집 (jargon)

| 용어 | 뜻 |
|---|---|
| **phantom** | CLIP은 아는데 multi-view aggregation이 신호를 죽인 실패. 회복 가능. 실패의 ~31%. |
| **real** | CLIP 자체 한계(encoder ceiling). ~7.5%. (phantom:real ≈ 4.2:1) |
| **easy** | 평균 체제가 이미 잘 맞는 쿼리. |
| **2DGS** | 가우시안 기하 모델(`point_cloud.ply`): 위치·스케일·회전·불투명도·SH. 학습/최적화 대상. |
| **sai_nag.pt** | 파이프라인 산출물 = **슈퍼포인트 계층(nag) + 슈퍼포인트별 CLIP 피처(nag_feat)**. 기하 아님(2DGS를 인덱스로 참조). test_lerf/eval이 로드하는 대상. |
| **canon-contrast** | 쿼리 점수 = `softmax(10·[sp·query, sp·canon])` vs `{object,things,stuff,texture}` 중 min. 전 paper 공유. 10×라 margin이 포화됨. |
| **세 법칙** | ① 소수파 매장 ② coherent-plausible 경쟁자(→robust 통계 원리적 불가) ③ 제로섬(view선택은 phantom 살리면 easy 죽임) |
| **ROFA** | ReLaGS의 z-score outlier 제거 aggregation. 실측: phantom median 효과 0, 구출 1/20(유령 청소부일 뿐). |
| **single-harness** | 우리 통일 평가: 선택 SP presence mask 렌더 → `>0.5` → IoU(per prompt,frame) → per-image → per-scene. `scripts/repro/eval_soft.py`. |
| **regime confusion** | 문제정의 고도화: "평균이 문제"가 아니라 "어떤 쿼리에 평균/selection이 맞는지 구분 못 함". |

---

## 2. 자산 레지스트리 (데이터·체크포인트·메소드·env — "뭘 썼나")

### 데이터 / GT
- **LERF-OVS 4 scene**: `data/lerf_ovs/<scene>` (figurines/ramen/teatime/waldo_kitchen). GT = LangSplat 폴리곤 `data/lerf_ovs/label/<scene>/*.json` (`cv2.fillPoly`). **67 prompt**(21/14/14/18), label frame 4/7/6/5.
- ⚠️ **language_features 캐시** `data/lerf_ovs/<scene>/language_features` (SAM seg `*_s.npy` + CLIP `*_f.npy`): **scene당 GPU ~4h, 재생성 금지.** HF 백업 `JUNHAKBAE/THGS-lerf-ovs-language-features`(비공개). 재생성 시 표준 SAM 아님 — langsplat SAM 변형 필요.

### 체크포인트 (출처·재현성 ★)
| method | 위치 | 출처 | single-harness mIoU | 논문 | 주석 |
|---|---|---|---|---|---|
| **THGS** | `output/lerf/<scene>/sai_nag.pt` | 공식 Google Drive ready-to-use **byte-identical(md5 4/4)** | **0.5887** | 0.5494 | ⚠️ **released≠paper**(released가 +3.9 높음, teatime +13.5). 우리 빌드 아님 |
| **ReLaGS** | `ReLaGS/output/lerf_hf/scenes/LeRF/<scene>` | 저자 HuggingFace `dfki-av/ReLaGS` | **0.6165** | 0.644 | fig/waldo 소수점 일치(anchor). ramen/teatime 미달=phantom |
- ✅ **2DGS는 공유**: ReLaGS-2DGS = THGS-2DGS를 `max_weight_pruning`(contribution 0.0005)한 것 (prune 후 가우시안 **4/4 정확 일치**: fig 234534/ramen 372270/teatime 958840/waldo 905012).
- ⚠️ **인용 규칙**: THGS=54.94 인용 금지. "공개 릴리즈+공식 프로토콜=58.87, 논문 54.94는 미공개 빌드"로 명시. 비교 baseline은 **released artifact**(THGS .589/ReLaGS .617), 재실행값(.57) 아님.

### 메소드 (재클론 대상 C — repo + 정체)
| method | repo | 정체 | env 주의 |
|---|---|---|---|
| THGS | github.com/Atrovast/THGS (현 루트) | base. 2DGS+superpoint, training-free | — |
| ReLaGS | github.com/dfki-av/ReLaGS (현 `ReLaGS/`) | THGS 파생 + pruning + ROFA + scene graph. 공식 eval 미공개 | ⚠️ **PyG2.3 패치 필요**(아래) |
| VALA | github.com/changandao/VALA (`external_methods/VALA`) | per-Gaussian, geometric median + visibility gating | sm_120 PTX-JIT, `envs/vala-port` |
| Segment-then-Splat | vulab-ai.github.io/Segment-then-Splat (`external_methods/`) | object-first 분할 후 1 embedding | 고비용 |
| OpenSplat3D | github.com/VisualComputingInstitute/opensplat3d (`opensplat3d/`) | visibility-top5 view MasQCLIP | 셋업 완료 |
| LangSplat | github.com/minghanqin/LangSplat | LERF-OVS 벤치 정의·원본 eval(`evaluate_iou_loc.py`) | — |

⚠️ **재클론 시 필수 패치 (이 env에서 돌리려면)**:
- ReLaGS vendored SPT의 PyG2.3 호환: `ReLaGS/ext/spt/{data/data.py,transforms/data.py,transforms/partition.py,transforms/point.py}` + `ReLaGS/sp_partition.py`에서 `X.keys()`→`X.keys` (keys가 method→property로 바뀜; THGS는 이미 패치됨).
- seed 재현: `{,ReLaGS/}graph_weight.py`에 `PIPELINE_SEED` 주입(graph_weight가 `torch.normal` unseeded).

### env / GPU
- **env**: `source /mnt/pilab_nas/projects/THGS/envs/thgs/bin/activate` (torch 2.11+cu128, py3.10, CUDA 확장 검증). HF 캐시 `.valahome/hf` + `HF_HUB_OFFLINE=1`. ⚠️ 시스템 주입 CLAUDE.md의 conda/CUDA_VISIBLE_DEVICES는 stale — `AGENTS.md`가 정본.
- **GPU = SLURM 예약제**: Google Calendar(pilgpu47@gmail.com) 예약 → Slack RID → `srun -p gpu --reservation=<RID> --gres=gpu:1`. 계정 `jhbae`. 노드 jhsong-TRX50-AI-TOP(RTXPRO5000). 스크립트는 repo 루트 `PYTHONPATH` 필요, ReLaGS는 `PYTHONPATH=$ROOT/ReLaGS`.

### 하네스 (단일 채점기)
- 선택 SP → presence mask 렌더 → `>0.5` → IoU → **per-image** 집계. `scripts/repro/eval_soft.py`(soft mask + threshold sweep), `scripts/eval_seg.py`(공식). THGS 선택=`get_related_gaussian(topk=3, level=[2,3])`, ReLaGS=`search_matched_superpoint_in_mhtree`(Alg-1, level_until=1 topk=5).
- ⚠️ 세 method가 원래 다른 채점기였음(THGS eval_seg per-image / ReLaGS diagnostic per-prompt / VALA flat per-frame) → fairness 트랙이 단일 하네스로 통일. `md/hypotheses/fairness/protocol.md`.

---

## 3. 실험 arc (시간순 요약 — 상세는 §4)

```
Stage1 실패 taxonomy → Stage2~3 mechanism(소수파 매장) → Stage4 method(guarded hybrid)
   → Stage5 ReLaGS 재현(method-agnostic) → P1 문제 구체화(경쟁 부검·부재쿼리·E1·E2)
   → [이번 세션] 재현성 감사(released≠paper) → 공정 비교(2DGS 공유·seeded 동률)
   → 다음: P2 가드 재설계 / fairness 정식화
```

---

## 4. 실험 원장 (phase별)

### [~2026-06-12] Stage 1–5 — 실패 taxonomy → mechanism → method → 일반화
- **질문**: 분할/검색/encoder 중 dominant 실패는? 회복 가능한가? method로 변환되나? 일반화되나?
- **한 것**: D1–D4 framework, oracle-rank 진단, all-SP per-view dump 재채점, hybrid 채점기 LOSO, ReLaGS 재현.
- **자산**: THGS/ReLaGS sai_nag, all-SP dump(THGS 1.37GB / ReLaGS 1.2GB), language_features, envs/thgs.
- **결과 ✅**: D2.phantom 31% vs real 7.5%(4.2:1, method-agnostic). **세 법칙**(소수파 매장 88/95%, coherent p=0.013, 제로섬 top5 +19.9/−9.8). **Stage4 hybrid α=0.3: held-out +3.74pt(phantom +17.5), easy −4.08pt = R9 PARTIAL.** **Stage5: ReLaGS서 전부 재현, ROFA phantom median 0.**
- **상세**: `md/hypotheses/experiments/` (stage1~5), `extended_failure_hypotheses.md`.

### [2026-06-12] P1 — 기존 paper 문제점 구체화 (경쟁 부검·부재쿼리·방향성·계층)
- **질문**: 경쟁 처방은 왜 실패? score 함수가 '없음'을 아나? phantom 방향이 systematic? hierarchy가 phantom 증폭?
- **한 것**: CA-1 geometric median + CA-2 bag(R11), 부재 쿼리(R13), E1 방향성, E2 계층.
- **자산**: all-SP dump(THGS/ReLaGS), LVIS 어휘, mask eval.
- **결과 ✅**: R11-d 적중(경쟁 전 변형 R9 bar FAIL — robust=easy만 gm_g +1.2/ph−1.8 vs selection=phantom만 top5 +19.9/easy−9.8 **거울상**). R13 c-분기: **top1-conf AUROC 0.836/0.848**(미사용 신호), 부재 유령 top1 42%/29%, margin 포화. E1 negative(modality-gap). E2 전파 37%≥세척 32%, 증폭 17%.
- **상세**: `md/hypotheses/strategy/{competitor_autopsy,p1_problem_experiments}.md`.

### [2026-06-17~18] 재현성 감사 — THGS/ReLaGS released vs paper
- **질문**: 공개 체크포인트+공식 프로토콜로 논문 수치 재현되나?
- **한 것**: 공식 eval_seg.py / single-harness로 released artifact 채점. level/topk/threshold/집계 ablation. prompt별 GT 진단.
- **자산**: THGS released(byte-verified), ReLaGS HF, single-harness.
- **결과 ✅**: THGS released **0.5887**(논문 0.5494 재현 불가 — 어떤 (level,topk,thr,집계)도 per-scene 동시 일치 못 함, teatime +13.5). ReLaGS **0.6165**(fig/waldo 논문 일치). **둘 다 released≠paper.** prompt 진단: ramen 14중 5개 multi-instance(구조적 저-IoU); teatime GT quirk(coffee≈coffee mug 0.89, bear nose⊂bear)는 점수를 *낮춤* → teatime 과잉은 GT 아니라 released 체크포인트가 쉬운 객체를 잘 함.
- **상세**: `md/cross_method/reproduction_results.md` §Phase0–4.

### [2026-06-19] 공정 비교 — 같은 2DGS 위 두 파이프라인 재실행(seeded)
- **질문**: THGS vs ReLaGS 비교가 다른 2DGS 때문에 불공정한가? ReLaGS 우위가 진짜인가?
- **한 것**: (a) ReLaGS pipeline을 THGS 2DGS 위 재실행 → 2DGS 공유 확인. (b) seed{0,1,2}로 **양 method** 모두 같은 THGS-2DGS+같은 seed 재실행 → mean±std.
- **자산**: THGS released 2DGS(`output/lerf`), language_features, envs/thgs, GPU jhsong, `scripts/repro/{job_seedrun,aggregate_seedrun,render_thgs_single}.py`, graph_weight seed 패치.
- **결과 ✅**: **2DGS 공유 확정**(prune 4/4 일치). **seeded fair: THGS .5696±.0024 ≈ ReLaGS .5701±.0030 → Δ+.0005 (동률).** seed std ≈ .002–.003(거의 0). per-scene: ReLaGS가 ramen+3.2/fig+1.0/waldo+1.5, THGS가 teatime+5.5로 상쇄.
- **산출물**: `output/render/repro/seedrun/`, `output/diagnostics/repro_fair_*.csv`.
- **상세**: `md/cross_method/reproduction_results.md` §Phase5.
- ⚠️ **캐비엇**: faithful 재실행(~.570)은 released(.589/.617)보다 낮음(ReLaGS −4.6 > THGS −1.9) = "공개 config ≠ release 빌드". **비교 baseline은 released artifact(ReLaGS 0.617), 재실행 0.57 아님.**

---

## 5. 확정 결론 / 열린 스레드 / 정정 (★ AI가 실수 반복 안 하게)

### ✅ 확정
- 소수파 매장 = paradigm-level, method-agnostic(THGS·ReLaGS 둘 다). robust 통계(median/ROFA) 원리적 불가, 순수 view-선택은 제로섬.
- ReLaGS-2DGS = THGS-2DGS prune(4/4 exact) → 두 method 같은 geometry 공유.
- 공정 조건(같은 2DGS+seed+하네스+공개 config)에서 **THGS ≈ ReLaGS**. published "ReLaGS +9.4"는 공정 비교에서 소멸.
- 양 method released≠paper(THGS released 0.589>paper 0.549; ReLaGS 0.617<0.644).

### 🔬 열린 / 다음
- **P2 가드 재설계(핵심 승부처)**: easy −4.08pt를 <1pt로 줄이며 phantom 회복 유지. P1-E.0 결과상 prompt-level 단일 스칼라 가드는 기각 → per-SP multivariate structural로 가야 함.
- **−5pt 재현 갭 미해결**: faithful 재실행이 released를 못 따라잡는 정확한 원인(미공개 튜닝? language_features 출처? 미식별 파라미터?) — 미규명.
- venue/deadline 미정. fairness 트랙(RF-L/RF-M) 정식 stage 미실행.

### ⚠️ 정정 (이전 주장 폐기 — 다시 꺼내지 말 것)
- ❌ "graph_weight stochasticity가 재실행 −5pt의 원인" → **틀림**. seed std가 .003로 거의 0. 진짜 원인은 "공개 config ≠ release 빌드"(체계적).
- ❌ 재실행 0.57을 비교 baseline으로 사용 → **금지**. faithfulness 미확정 + ReLaGS 과소평가. baseline = released 0.617.
- ❌ "THGS-vs-ReLaGS 58.87 vs 61.65가 apples-to-oranges(per-image vs per-prompt)" → **틀림**. 둘 다 per-image, 단일 하네스로 재확인.

---

## 6. 포인터 (상세 문서 지도)
- 전략: `md/hypotheses/strategy/{roadmap,competitor_autopsy,p1_problem_experiments,p2_next_stage_thoughts}.md`
- 실험: `md/hypotheses/experiments/` (stage1~5, README, intuition)
- 가설 카탈로그: `md/hypotheses/extended_failure_hypotheses.md`
- 재현·공정: `md/cross_method/{reproduction_results,reproduction_plan,paper_deep_analysis}.md`
- 공정성 헌법: `md/hypotheses/fairness/{protocol,intuition,preregistration_log}.md`
- VALA/외부: `md/hypotheses/strategy/vala문제.md`, `md/hypotheses/strategy/p1e_external_code_study_plan.md`
- GPU/운영: `AGENTS.md` (시스템 CLAUDE.md는 stale)
