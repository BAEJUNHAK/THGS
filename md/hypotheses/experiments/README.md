# `experiments/` — Stage-by-stage experimental progression

가설 문서 ([../extended_failure_hypotheses.md](../extended_failure_hypotheses.md), 6.2 patch) 에서 정의한 가설들을 단계별로 검증해 가는 과정의 기록.

각 stage 가 다음 stage 의 question 을 결정 — 결과를 보고 *다음에 답해야 할 가장 중요한 질문* 으로 분기.

---

## 📋 Stage 진행 상황

| Stage | 질문 | 실험 | 상태 | 핵심 결과 |
|---|---|---|---|---|
| **1** | 분할/검색/encoder 중 무엇이 dominant 실패 원인인가? | B7 + A4 + A2 + Joint + Cross-method | ✅ **완료** | **D2.phantom 31% vs D2.real 7.5% (4.2:1 비율)** — method-agnostic |
| **2A** | 17 persistent phantoms 의 mechanism (4-layer forensic) | B8 mix-rate + wrong-top1 + per-view trajectory + D3 top-k + montage | ✅ **완료** | 65% wrong-top1 = background drift, 76% structural, 71% ROFA류 fail |
| **2B** | Within-view vs across-view vs encoder 중 dominant 는? | H2 lite + F2 subtype + ROFA keep_mask + D3 67-prompt + C3 | ✅ **완료** | **65% phantoms 가 within-view stage 에서 신호 죽음** (strong_signal_phantom); ROFA 대체로 정상 (✱τ=2 검증: 실수 4/17); C3 instance confusion 회복 불가 |
| **검증** | Stage 1/2B 결론이 구현 디테일 (τ, metric, threshold) 에 robust 한가? | τ∈{1,2} sweep + joint raw/canon + H2-lite rank ([verify_stage2b_checks.py](../../../scripts/verify_stage2b_checks.py)) | ✅ **완료 (2026-06-10)** | Stage 1 robust (변화 0); 65% strong-signal 유지; F2.C 정정 (실수 4건, 그중 2건 instance-confusion); onion segments encoder 측 재분류 |
| **3.1** | within-view mixing 이 *정말* 신호를 죽이는가? (인과 replay) | B8 직접 측정 + clean/mixed/hard 3-신호 통제 비교 + 충실도 gate 33/33 | ✅ **완료 (2026-06-10)** | **B8 무죄** — strong_signal gap 0.004 (기준의 1/13), per-view mixed rank 1-7 인데 최종 rank 4-256 → **살해 지점 = across-view 누적 (B1)**. 사전 분기 ③ 발동, hard-assignment fix 는 실행 전 기각 |
| **3.2** | B1 (across-view 누적) 의 어디서/왜 신호가 죽는가? | B1.A 단일view+궤적 / B1.B ablation 12종 / B1.C 경쟁 분해 / ✱공정 결투 (dump 재사용) | ✅ **완료 (2026-06-11)** | **R1·R3 확정**: phantom = 소수(18%) 탁월 view 가 평균에 묻힘; 경쟁 wrong SP 는 coherent-plausible. **R2 정정 (적대적 검증)**: query top-k 15/17 은 동결-경쟁자 상한 — 공정 결투에선 단독 승리 3/15, **후보권 복귀 13/15 (대부분 rank 2)** → Stage 4 = query-aware + 결합 신호 (D1 필터·coherence prior 등) |
| **3.3** | 확정 진단: 동결 없는 전 pool 재채점이 *실제 mIoU* 를 올리는가? 사기꾼의 정체는? | all-SP dump (1.37GB) + full-pool rank + **mask-IoU 208쌍** + impostor 법의학 | ✅ **완료 (2026-06-11)** | **R4 FAIL** — query-top-k 단독은 제로섬 (phantom IoU **2배** ↔ easy −9.8pt, full-67 −0.1pt). **R5** — semantic confusion (단 5/16 은 진실의 조각 = granularity). ✱탐색: **hybrid α=0.3 이 rank 순 +9** → Stage 4 출발점. zero-norm 유령 150개 전수 확인 |
| **3.4** | 대다수 원인 밖의 잔여 케이스들의 정확한 원인은? (method 설계 전 최종 진단) | 친족 감사 + 승자 신원분석 + 가드 신호 + multi-instance 감사 → **36 케이스 taxonomy** | ✅ **완료 (2026-06-11)** | **R6 PASS** (unknown 2) · **R7 발동** (multi-instance 7 prompt — 평가 보정 필수) · **R8 전원 재분류** (2B encoder-측 4/4 + **Stage 1 D2.real 4/5 도 dilution 이었음** → encoder 잔여 3건). 신규 원인: **few-view opportunist** (mean 은 저관찰 SP 에 유리) + 가드 신호 g1/g2 확보 |
| **4** | taxonomy 가 명세한 결합 method 가 LOSO mask mIoU 를 올리는가? | 채점기 grid (216+96) + 정합성 gate (0/67×2) + LOSO + held-out mask eval + ablation | ✅ **완료 (2026-06-12)** | **R9 PARTIAL (최종)** — held-out **full-67 +3.74pt** (phantom **+17.5**, other +13.3, multi-inst 제외 +4.53) 로 headline 기준 통과, 단 **easy −4.08pt** 로 보호 기준 미달. ablation 이 g1 결함 적발, g2 margin 분리자는 포화로 불활성 → **Stage 5 = 가드 분리자 재설계** |
| **5** | 메커니즘이 ReLaGS (자체 partition + ROFA) 에서도 재현되는가? | ROFA-포함 replay (gate 97.8~100%, 유령 21=86% 감소 재확인) + G1~G4 | ✅ **완료 (2026-06-12)** | **G5: 전부 재현** — G1 95%·23%/51%, G2 +8/−12 제로섬, G3 동일 승자 시그니처, **G4: ROFA 의 phantom 실측 효과 median 0 (구출 1/20)** → **"소수파 매장은 method-agnostic, ROFA 도 못 막음" 확정** |
| **6 (=P1)** | **기존 paper 들의 문제점 구체화 및 확정** — 경쟁 처방은 왜 실패하나 (A) / score 함수가 '없음' 을 아나 (B) / phantom 방향이 systematic 한가 (C) / hierarchy 가 phantom 을 증폭하나 (D) | A 부검 CA-1/2 (R11) + B 부재쿼리 (R13) + C E1 + D E2 — gate 0 mismatch ×2 + mask 재현 차 0.0 | ✅ **완료 (2026-06-12)** | **R11-d 적중: 경쟁 전 변형 R9 bar FAIL** — robust=easy만 (gm_g +1.2/ph −1.8) vs selection=phantom만 (top5 +19.9/easy −9.8) **거울상 확정**; R11-b/c 빗나감은 전부 "예측보다 나쁨" 방향. **R13 c-분기 발동**: top1-conf AUROC 0.84 (P2 가드 입력) + **부재 쿼리 유령 top-1 42%/29%** + margin 포화 재확인. **E1 negative** (modality-gap 지배). **E2 전파 37%≥세척 32% 적중**, 증폭 17%<20% (tesla 만 양 method 매장). 결과: [../strategy/competitor_autopsy.md](../strategy/competitor_autopsy.md) §4-5 + [../strategy/p1_problem_experiments.md](../strategy/p1_problem_experiments.md) §6 |
| **7 (=P2)** | method 제작 — 가드 재설계 (R12, top1-conf 신호 채택) + R10 (ReLaGS drop-in) + 가드 이식 | P1 리뷰 후 사전등록 | ⏳ 다음 | P1-E (VALA·StS 코드 직접 실험) go/no-go 도 사람 결정 대기 |

---

## 🎯 Stage 1 — 실패의 1 차 mechanism 분해

상세 분석: **[stage1_b7_a4_a2.md](stage1_b7_a4_a2.md)**

핵심 발견 한 줄: *"THGS/ReLaGS 실패의 81% 는 multi-view CLIP aggregation 이 만든 회복 가능한 손실 (D2.phantom) 이며, 이 분포는 method-agnostic 이다."*

## 🎯 Stage 2A — Phantom Anatomy (17 persistent phantoms 의 4-Layer forensic)

상세 분석: **[stage2a_phantom_anatomy.md](stage2a_phantom_anatomy.md)**

핵심 발견 한 줄: *"65% 의 persistent phantom 은 ROFA 류 across-view aggregation 의 실패다 — instance confusion 과 background drift 는 소수 (24%)."*

## 🎯 Stage 2B — ROFA Anatomy + D3 Sweep + C3 Quickcheck

상세 분석: **[stage2b_rofa_anatomy.md](stage2b_rofa_anatomy.md)**

핵심 발견 한 줄: *"17 phantom 중 11 개 (65%) 는 clean SP-mask CLIP encoding 으로 cos≥0.20 의 strong signal 인데도 phantom — pipeline 의 within-view SAM mixing 단계가 신호를 죽이고 있다. ROFA 는 정상 작동하지만 이미 죽은 signal 위에서."*

## 🎯 Stage 3.1 — B8 인과 replay (within-view 직접 검증)

상세 분석: **[stage3_b8_causal.md](stage3_b8_causal.md)**

핵심 발견 한 줄: *"B8 (within-view mixing) 은 무죄 — per-view mixed 신호는 rank 1-7 로 살아있다 (clean 대비 손실 0.004, easy 는 오히려 −0.018 개선). 신호는 across-view 누적 (B1) 에서 죽는다. Stage 2B 의 결론은 이 실험으로 수정됨."*

## 🎯 Stage 3.2 — B1 anatomy (averaging 의 해부 + method 방향 확정)

상세 분석: **[stage3_2_b1_anatomy.md](stage3_2_b1_anatomy.md)**

핵심 발견 한 줄: *"평균은 다수결이다 — phantom 의 정답 view 는 소수파(18%)라 지고, 경쟁자는 '균질하게 그럴듯해서' 이긴다 (평균 prompt-cos 도 oracle 보다 높음!). query 가 지명하는 top-k view 선택이 17 중 15 를 회복 (regression 0) — Stage 4 의 method 사양이 정해졌다."* (⚠ "15/17" 은 이후 3.3 에서 동결-경쟁자 환상으로 정정)

## 🎯 Stage 3.3 — 확정 진단 (proxy 전부 제거)

상세 분석: **[stage3_3_definitive.md](stage3_3_definitive.md)**

핵심 발견 한 줄: *"query-aware top-k 단독은 제로섬이다 — phantom IoU 2배 (0.203→0.402) 와 easy −9.8pt 가 정확히 맞교환 (R4 FAIL). 만능 한 줄 fix 는 없음이 확정. mean 은 easy 를, top-k 는 phantom 을 지키므로 Stage 4 = 둘의 hybrid (α=0.3, rank 순 +9)."*

## 🎯 Stage 3.4 — 잔여 원인 전수 분해 (36 케이스 taxonomy)

상세 분석: **[stage3_4_residual_causes.md](stage3_4_residual_causes.md)**

핵심 발견 한 줄: *"잔여를 해부하니 새 원인 2개 — few-view opportunist (view 2-11개 SP 는 희석이 없어 mean 게임에서 유리: spoon/cabinet 의 진범) 와 multi-instance 미표기 의심 (7 prompt) — 가 나왔고, Stage 1 의 'encoder 한계 7.5%' 는 과대 추정이었음이 드러났다 (D2.real 5 중 4 가 dilution: miffy 조차 회복). Stage 4 의 요구사항 명세 완성."*

## 🎯 Stage 4 — Method 검증 (LOSO mask-mIoU)

상세 분석: **[stage4_method.md](stage4_method.md)**

핵심 발견 한 줄: *"hybrid 본체는 진짜 — held-out +3.74pt (phantom 2배). 그러나 easy 보호 가드는 미완 (−4.08pt): margin 분리자가 canon 점수 포화로 불활성. R9 PARTIAL 로 정직 마감, 가드 재설계가 Stage 5 의 단일 question."*

## 🎯 Stage 5 — 메커니즘 일반화 (ReLaGS 재현)

상세 분석: **[stage5_relags_replication.md](stage5_relags_replication.md)**

핵심 발견 한 줄: *"소수파 매장 (95%, 23% vs 51%) 과 제로섬 (+8/−12) 이 ReLaGS 의 자체 파이프라인에서 거의 같은 숫자로 재현 — 그리고 ROFA 의 실측 효과는 phantom 에 median 0 (구출 1/20): ROFA 는 유령 청소부일 뿐 다수결의 비극은 못 막는다. paradigm-level 결함 확정."*

Paper Section 2 draft: **[../../THGS/paper_section2_draft.md](../../THGS/paper_section2_draft.md)** (⚠ Stage 3.1 결과 반영 전 — within-view 범인 지목 부분은 수정 필요)

산출물:
- [output/diagnostics/b7_a4_combined.csv](../../../output/diagnostics/b7_a4_combined.csv) — THGS B7+A4 (208 rows)
- [output/diagnostics/b7_a4_combined_relags.csv](../../../output/diagnostics/b7_a4_combined_relags.csv) — ReLaGS B7+A4 (208 rows)
- [output/diagnostics/a2_image_clip_ceiling.csv](../../../output/diagnostics/a2_image_clip_ceiling.csv) — A2 (268 rows)
- [output/diagnostics/cross_method_d2_decomposition.csv](../../../output/diagnostics/cross_method_d2_decomposition.csv) — 67 prompts × 두 method side-by-side
- [output/diagnostics/plots/](../../../output/diagnostics/plots/) — 15 PNG (within-method 11 + cross-method 4)
- [md/THGS/paper_section1_draft.md](../../THGS/paper_section1_draft.md) — Paper Section 1 draft v2

스크립트:
- [scripts/b7_a4_oracle_analysis.py](../../../scripts/b7_a4_oracle_analysis.py) (THGS 본체)
- [ReLaGS/scripts/b7_a4_oracle_analysis.py](../../../ReLaGS/scripts/b7_a4_oracle_analysis.py) (ReLaGS 용 copy)
- [scripts/a2_image_clip_ceiling.py](../../../scripts/a2_image_clip_ceiling.py)
- [scripts/b7_a4_a2_plots.py](../../../scripts/b7_a4_a2_plots.py)
- [scripts/cross_method_comparison.py](../../../scripts/cross_method_comparison.py)

---

## 📜 Stage 작성 규칙

각 stage 파일은 다음 sections 을 포함:

1. **Stage 의 question** — 무엇에 답하려고 했나
2. **Pre-experiment 상태** — 이전 stage 까지 알고 있던 것
3. **실험 list + 각 실험의 (목적 / 방법 / 직관 / 결과 / 의미)**
4. **종합 finding** — 실험들이 같이 말하는 것
5. **산출물** — 재현 가능한 CSV/plot/script 목록
6. **Decision point** — 결과를 본 다음 단계 후보들과 선택

이렇게 정리하면 paper writing 시 각 section 마다 *어떤 실험으로 그 claim 이 지지되는지* 추적 가능.

---

## 🔗 관련 문서

- [../extended_failure_hypotheses.md](../extended_failure_hypotheses.md) — 6.2 patch 가설 catalog (17 hypotheses, Tier 0/1/2/3)
- [../../README.md](../../README.md) — md 폴더 전체 index
- [../../THGS/lerf_ovs_failure_analysis_results.md](../../THGS/lerf_ovs_failure_analysis_results.md) — D1-D4 framework (출발점)
- [../../cross_method/lerf_ovs_thgs_vs_relags.md](../../cross_method/lerf_ovs_thgs_vs_relags.md) — 13 cross-invariant + D1 86% 감소 (prior)
