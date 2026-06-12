# Stage 3.2 — B1 Anatomy (across-view 누적의 해부)

> 완료 (2026-06-11). Stage 3.1 이 좁힌 유일 용의자 — across-view visibility-weighted 누적 (B1) — 의 mechanism 을 replay dump 재사용으로 해부 (신규 GPU 작업: wrong-top1 replay ~15분뿐).
>
> 핵심 finding 한 줄: **"phantom 은 '소수의 탁월한 view 가 다수의 평범한 view 에 평균으로 묻히는' 현상이다 (15/17 이 단일 view 로 pool rank 1 가능, 그러나 좋은 view 비율이 18% vs easy 50%). 경쟁에서 이기는 wrong SP 는 '균질하게 그럴듯한' SP 다 (coherence 우위 p=0.013, 평균 prompt-cos 까지 oracle 보다 높음). query-conditioned top-k view 선택이 17 중 15 를 rank 1-3 으로 회복 (easy regression 0) — Stage 4 method 방향 확정."**

---

## 0. Stage 3.2 의 question

> Stage 3.1: per-view mixed 신호는 살아있는데 (text-side rank 1-7) 최종 SP feature 만 죽는다 (pool rank 4-256). **across-view 누적의 정확히 무엇이 신호를 죽이는가? (i) 어디서 죽나 (ii) 어떻게 합치면 사나 (iii) 누구에게 왜 지나**

사전 판정 규칙 (B1 §6.4, 실험 전 등록): R1 (averaging 인과성), R2 (회복 가능한 aggregation 존재), R3 (dispersion-defeat mechanism).

---

## 1. Pre-experiment 상태

- Stage 3.1: B8 기각, 살해 구간 = [per-view mixed → 최종 feature] 사이 = B1. 단 per-view rank 는 *text-side* 경쟁이라 pool 경쟁과의 다리 필요 (§4 해석 한계).
- 가용 데이터: stage3_b8_replay_perview.pkl (33 SP × 전체 view mixed feature + portion)

---

## 2. 공통 인프라 + 충실도 gate

**Pool rank** ([scripts/stage3_2_common.py](../../../scripts/stage3_2_common.py)): pool = sai_nag 의 levels [2,3] 전체 SP feature (figurines 745 = 기존 "rank 64/745" 와 동일 universe). 후보 feature 가 해당 SP 의 pool entry 를 *교체*한 뒤 ClipSimMeasure canon-contrast 로 rank (A4 와 동일 기준). raw-cos rank 병기.

**충실도 gate**: pkl 로 재구성한 visibility-weighted baseline 의 pool rank vs b7_a4 oracle_rank (±2):
- **26/33 strict pass.** 7건 초과는 전부 deep-rank 영역 (a4 46-256 에서 ±4-19, 상대오차 ~5-10% — fp16 + 동점처리 누적): tesla 237↔256, hand 36↔46, ottolenghi 62↔66, spoon 44↔47, old camera 153↔130, pikachu 68↔64.
- **유일한 질적 이탈: jake (재구성 1 vs A4 6)** — rank≤3 경계를 넘는 단 한 건. 이하 R2 의 회복 카운트는 jake 포함/제외 병기.
- 판정에 쓰는 임계 (rank ≤ 3 vs ≫3) 기준으로는 분석 유효.

---

## 3. B1.A — 단일 view pool rank (진짜 A3-postB8) + 누적 궤적

**Code**: [scripts/stage3_2_b1a_trajectory.py](../../../scripts/stage3_2_b1a_trajectory.py)

### 결과 — phantom 의 정체가 한 표로

| 통계 | Phantom 17 | Easy 16 |
|---|---|---|
| best 단일 view pool rank = 1 | **15/17** (spoon 6, cabinet 9 만 예외) | 16/16 |
| best 단일 view rank ≤ 3 | **15/17 (88%)** | 16/16 |
| **view 중 rank ≤ 3 인 비율 (평균)** | **18%** | **50%** |
| 단일 view rank 의 중앙값(의 중앙값) | **51** | **2** |
| 최종 (전부 누적) rank ≤ 3 | **1/17** | 15/16 |

→ **phantom 과 easy 의 차이는 '탁월한 view 의 존재' 가 아니라 '탁월한 view 의 비율'.** 둘 다 rank-1 짜리 view 를 갖고 있다. easy 는 절반이 좋아서 평균이 살아남고, phantom 은 18% 만 좋아서 평균이 묻는다.

### 누적 궤적 분류 (17 phantom)

| 곡선 형태 | n | 의미 |
|---|---|---|
| sudden_drop | 9 | 특정 누적 구간에서 rank 급락 (나쁜 view 군집 유입) |
| gradual_dilution | 3 | 점진 희석 (rubber duck, tesla, bowl) |
| no_degradation | 3 | 처음부터 끝까지 비슷 (이미 경계선) |
| **never_good_in_pool** | **2** | **spoon, cabinet — 어떤 단일 view 도 pool top-3 불가** → aggregation 으로 회복 불가능한 진짜 잔여 |

---

## 4. B1.B — Aggregation ablation (re-run 없는 counterfactual fix)

**Code**: [scripts/stage3_2_b1b_ablation.py](../../../scripts/stage3_2_b1b_ablation.py) — 6.1 이 명세했던 feature-space GMM 도 (e) 에서 최초 실제 구현.

| Aggregation | phantom 회복 (rank≤3) /17 | easy regression /15 |
|---|---|---|
| (a) uniform mean | 1 | **8** ← visibility weighting 은 오히려 **필수적** |
| (b) visibility-weighted [baseline] | 1 (jake, gate 노이즈) | 0 |
| (c) top-k by portion, k=1 / 3 | **6 / 6** (jake 제외 시 5) | **0 / 0** |
| (d) **top-k by query-cos, k=5** | **15** (jake 제외 14) | **0** |
| (e) mode-cluster (GMM k=2 majority) | 3 | 6 |
| (f) ROFA τ=2 | 1 | 0 |

핵심:
- **(d) query-conditioned 가 압도적**: k∈{1,3,5,10} 모두 14-15/17 회복 + regression 0. 단 query 를 쓰므로 *오프라인 사전계산 feature 로는 불가* — **inference-time aggregation** 으로의 방법 전환이 필요 (leakage 아님: query 는 inference 입력).
- (c) 정적 top-k 도 R2 기준 (≥6, ≤1) 을 **턱걸이 통과** — 정적 fix 의 한계선.
- (e) GMM majority cluster 는 실패 (F2 의 mode-cluster 제안 기각), (f) ROFA 무력 재확인, (a) uniform 은 개악.

---

## 5. B1.C — Pool 경쟁 분해 (oracle vs wrong-top1, paired 15쌍)

**Code**: [scripts/stage3_2_b1c_compare.py](../../../scripts/stage3_2_b1c_compare.py). wrong-top1 17 SP 를 같은 replay 로 dump ([--targets_csv 확장](../../../scripts/stage3_b8_replay.py)) — 충실도 15/17 (예외 ① pumpkin 의 wrong-top1 = **zero-norm degenerate SP** — canon(0)=0.5 로 1등이 되는 **D1 메커니즘이 phantom 내부에서 재발견**됨 ② ottolenghi 쌍 분석 제외).

| Wilcoxon paired (one-sided) | oracle med | wrong med | p | 판정 |
|---|---|---|---|---|
| **coherence: oracle < wrong** | 0.870 | 0.886 | **0.0128** | ✅ **dispersion-defeat 확정** |
| prompt-cos 평균: oracle > wrong | 0.220 | 0.227 | 1.000 | ❌ — **wrong 이 평균적으로 더 높음!** |
| visibility 합: oracle < wrong | 67.3 | 27.0 | 0.982 | ❌ — oracle 이 오히려 더 잘 보임 |

**놀라운 발견**: wrong-top1 SP 들은 '의미 없는 배경' 이 아니라 — per-view 평균 prompt-cos 가 oracle 보다 *높은*, **균질하게 그럴듯한 (coherent-plausible) 경쟁자** 다. Stage 2A 의 "background drift 65%" 명명은 GT-overlap 기준의 분류였고, CLIP 의 눈에는 이들이 평균적으로 더 prompt 같아 보인다.

→ **mean-vs-mean 게임에서는 oracle 이 구조적으로 진다** (평균조차 열세). oracle 의 우위는 오직 *best views* (rank 1 짜리 소수) 에 있다 → 평균 기반 개선 (더 나은 평균, outlier 제거, cluster 중심) 으로는 원리적으로 회복 불가, **view 선택 (top-k by query)** 만이 oracle 의 비교우위를 쓰는 길. B1.B 의 (d) 압승과 정확히 일치.

---

## 6. 사전 등록 판정

| 규칙 | 판정 |
|---|---|
| **R1** | ✅ **averaging 인과 확정** — best 단일 view ≤3 = 88% (기준 50%), 최종 ≤3 = 6% |
| **R2** | ✅ 정적 (c) top-1/3-by-portion 이 기준 통과 (6/17, reg 0; jake 제외 시 5 로 경계) / **query-conditioned (d) 가 14-15/17, reg 0** → **Stage 4 = query-aware aggregation 으로 설계** |
| **R3** | ✅ **dispersion-defeat 확정** (p=0.013) + 강화 발견: 경쟁자는 평균 prompt-cos 도 우위 → 평균 게임 자체가 oracle 에게 불리 |

### 한 줄로

> **"평균은 다수결이다. phantom 의 정답 view 는 소수파(18%)라 다수결에서 지고, 상대는 만장일치(높은 coherence)로 그럴듯하다. 해법은 더 나은 다수결이 아니라 — query 가 지명하는 소수 정예 (top-k by query-cos): 17 중 15 회복, 부작용 0."**

---

## 6.5 ✱ 적대적 검증 패치 (2026-06-11) — "15/17 은 동결-경쟁자 상한이었다"

위 R2 의 15/17 은 **타깃 SP 만 query-보정하고 경쟁자 전원은 pipeline 평균 feature 로 동결**한 측정 — 실제 query-aware method 에서는 모든 SP 가 재채점된다. **공정 결투** ([scripts/stage3_2_fair_duel.py](../../../scripts/stage3_2_fair_duel.py), [stage3_2_fair_duel.csv](../../../output/diagnostics/stage3_2_fair_duel.csv)): oracle 과 그 wrong-top1 *둘 다* query-top-5 보정 후 canon 대결:

| | 결과 |
|---|---|
| **oracle 결투 승리** | **3/15 뿐** (pikachu, hand, plate) — wrong SP 도 best view 를 쓰면 12/15 에서 여전히 승리 |
| oracle fair-rank ≤ 3 (둘 다 보정, 나머지 동결) | 13/15 — 단 대부분 **rank 2** (wrong 이 1등 유지) |

**해석 정정**:
1. query-aware 의 진짜 효과 = phantom 을 "완전 실종 (rank 4-256)" 에서 **"coherent 사기꾼 뒤의 rank 2"** 로 복귀시키는 것. top-3 union 에는 포함되지만 wrong SP 도 같이 뽑힘 → **문제가 D2 (실종) 에서 D3/over-union (오염) 으로 전이**. mask 수준 이득은 wrong SP 의 GT-overlap 에 따라 다름 (over-union 류 wrong 은 무해할 수 있고, background-drift 류는 유해).
2. wrong SP 의 top-5 cos 가 oracle 보다 높은 경우가 다수 (old camera 0.318 vs 0.259 등) — **결정적 분리는 query-cos 단독으론 불가**, 추가 신호 필요: (i) zero-norm/D1 필터 (pumpkin 류 즉사) (ii) coherence prior 의 역이용 (iii) 공간/mask 정합 (iv) negative contrast. Stage 4 설계는 "top-k 채점" 단독이 아니라 **결합 신호**로.
3. 이 결투조차 낙관적 상한 — full-pool 재채점 시 dark horse 부상 가능. Stage 4 full eval 이 최종 심판.

**유지되는 것**: R1 (진단 — 소수파 매장, selection-free 18% vs 50%), R3 (coherent-plausible 경쟁자 — 결투 결과로 오히려 강화), B1.B 의 서열 (query-aware ≫ 정적 top-k > 평균 계열). **정정되는 것**: "15/17 회복" → "후보권 복귀 13/15 (대부분 rank 2), 단독 승리 3/15".

---

## 7. 산출물

### Data
- [stage3_2_singleview_rank.csv](../../../output/diagnostics/stage3_2_singleview_rank.csv) (33) / [stage3_2_trajectory.csv](../../../output/diagnostics/stage3_2_trajectory.csv) (per SP per k)
- [stage3_2_ablation.csv](../../../output/diagnostics/stage3_2_ablation.csv) (33 × 12 variants)
- [stage3_2_wrongtop1_perview.pkl](../../../output/diagnostics/stage3_2_wrongtop1_perview.pkl) + [stage3_2_wrongtop1_targets.csv](../../../output/diagnostics/stage3_2_wrongtop1_targets.csv) / [stage3_2_competition.csv](../../../output/diagnostics/stage3_2_competition.csv) (15쌍)

### Plots
- `plots/stage3_2_trajectory_curves.png` — 17 phantom 누적 궤적 (log)
- `plots/stage3_2_ablation_recovery.png` — 12 variants 회복/역행
- `plots/stage3_2_coherence_paired.png` — coherence paired scatter

### Scripts
- [stage3_2_common.py](../../../scripts/stage3_2_common.py) / [stage3_2_b1a_trajectory.py](../../../scripts/stage3_2_b1a_trajectory.py) / [stage3_2_b1b_ablation.py](../../../scripts/stage3_2_b1b_ablation.py) / [stage3_2_b1c_compare.py](../../../scripts/stage3_2_b1c_compare.py)
- [stage3_b8_replay.py](../../../scripts/stage3_b8_replay.py) `--targets_csv` 확장

---

## 8. Decision — Stage 4

**방향: query-aware aggregation — 단, ✱ 적대적 검증 (§6.5) 반영하여 "top-k 채점 단독" 이 아니라 결합 신호로 설계.**

설계 스펙 (이 stage 의 숫자가 직접 결정):
1. **표현**: SP 당 단일 512-vector 대신 **per-view feature 의 보존/압축** — (i) 전량 fp16 (~200MB/scene) 또는 (ii) SP 당 K 개 prototype (k-means / 방향 클러스터, ~수 MB) — bimodal/소수파 view 보존이 목적이므로 prototype 방식이 자연
2. **Inference**: query 입력 시 `score(SP) = top-k mean over prototypes/views of canon-contrast` (B1.B 의 (d), k=5 부근)
3. **기대 성능**: phantom 15/17 회복 + easy regression 0 (B1.B 의 upper bound) — mIoU 환산은 Stage 4 에서 full eval
4. **회복 불가 잔여의 정직한 처리**: spoon·cabinet (never_good_in_pool) + encoder-limit (D2.real 5) 는 이 방법 밖 — paper 의 limitation
5. **검증 설계**: 67 prompt full eval (phantom 만이 아니라) + 3DOVS 일반화 + ReLaGS 에 같은 query-aware 적용 (method-agnostic claim)

부수 과제: pumpkin 의 wrong-top1 이 zero-norm (D1) 인 사례 — D1 필터 (zero-norm 제외) 1줄로 pumpkin 류 회복 가능성, Stage 4 에서 같이 처리.

---

## 9. 관련 문서

- [stage3_b8_causal.md](stage3_b8_causal.md) — 직전 stage (B8 기각)
- [../extended_failure_hypotheses.md](../extended_failure_hypotheses.md) — B1 §6.4 설계 + ★★ 결과
- [intuition.md](intuition.md) / [experiment_results.md](experiment_results.md) / [README.md](README.md)
