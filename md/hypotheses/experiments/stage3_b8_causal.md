# Stage 3.1 — B8 인과 replay (within-view mixing 직접 검증)

> 완료 (2026-06-10). Stage 2B 가 지목한 "within-view SAM mixing = 범인" 가설을 **proxy 가 아닌 실제 파이프라인 replay** 로 직접 검증.
>
> 핵심 finding 한 줄: **"B8 (within-view ratio mixing) 은 무죄다. 17 phantom 의 per-view mixed 신호는 rank 1-7 로 살아있고 (clean 대비 손실 0.004), 신호는 그 *다음* 단계 — across-view visibility-weighted 누적 (B1) — 에서 죽는다. 사전 판정 분기 ③: B1 재조준."**

---

## 0. Stage 3.1 의 question

> **Stage 2B 의 결론 ("65% strong-signal phantom 은 within-view SAM mixing 이 죽인다") 은 H2-lite 의 *간접 대비* 에 기반했다. 실제 파이프라인의 within-view 단계를 그대로 재현하면, 그 단계에서 정말 신호가 죽는가?**

Stage 2B 의 evidence 체인에는 구멍이 있었다: H2-lite(clean) vs pipeline(phantom) 의 대비에는 within-view mixing 외에 SAM mask 기하 / crop 정책 / visibility weighting 차이도 섞여 있었고, **B8 의 직접 측정치 (`sp_mask_mat` mix count) 는 한 번도 측정된 적이 없었다** (`language_features/` 부재). Stage 3.1 = 이 인과 고리를 닫는 실험.

사전 판정 규칙 ([../extended_failure_hypotheses.md](../extended_failure_hypotheses.md) B8 §6.3, 실험 *전* 등록):
- ① strong_signal 11 에서 mean(clean−mixed) ≥ 0.05 AND easy 대비 유의 → B8 손실 확인
- ② mix-rate ↔ gap 상관 r > 0.4 → 가설 predict 적중
- ③ hard 가 gap 의 ≥70% 회복 → Stage 3.2 GO / <30% → B2 pivot / **gap 자체가 작으면 → within-view 무죄, B1 재조준**

---

## 1. Pre-experiment 상태

- Stage 2B: 17 persistent phantom 중 11 (65%) 가 strong_signal (clean crop 으로 CLIP 인식 가능한데 pipeline 에서 phantom) → "within-view mixing 이 범인" 추정
- 검증 패치 (2026-06-10): τ=2.0 재실행·rank 재검증으로 65% 헤드라인은 유지, 단 모든 evidence 가 **간접** (proxy) 임을 명시
- B8 mix count: **사상 미측정**

---

## 2. 실험 설계 — 3 신호의 통제된 비교

### Phase A — language features 재생성 (선행)

`data/lerf_ovs/<scene>/language_features/` 4 scene 재생성 (per-view SAM seg map + per-mask CLIP feature, 총 ~9GB, GPU ~4h). 함정: **표준 segment-anything 으로는 불가** — LangSplat 변형 필요 (CLAUDE.md 참조). `image_encoding.py` 의 bare-except 무한 재시도 버그 발견 → traceback 출력 + 재시도 상한 패치. `mask_nms` O(n²) 루프는 벡터화 (30회 동치성 검증 후 적용; 단 병목은 SAM generate 자체였음).

### Phase B — replay ([scripts/stage3_b8_replay.py](../../../scripts/stage3_b8_replay.py))

17 phantom oracle SP + scene 당 4 easy control (같은 RandomState(42) → stage2a 와 동일 샘플) 에 대해, **THGS production 경로** (`proj_gaussian_features_x`, configs/lerf.yml `feat_assign: 2`, WEIGHT_THRESHOLD=0.0001, RATIO_THRESHOLD=0.3) 를 모든 train view 에서 verbatim 재현. (주의: cross_method 문서의 "THGS=0.01" 은 미사용 경로 `feat_assign=1` 의 값.)

같은 (SP, view) 에서 세 신호:
| 신호 | 정의 | 의미 |
|---|---|---|
| **clean** | oracle SP mask crop 의 CLIP encoding (기존 H2-lite) | 섞기 전 재료 |
| **mixed** | `normalize(threshold(ratio row) @ view_level_feature)` | **B8 직후** — 파이프라인 연산 그대로 |
| **hard** | `view_level_feature[argmax(ratio row)]` | hard-assignment fix 의 per-view 미리보기 (counterfactual) |

+ **mix_count** = `(ratio ≥ 0.3).sum()` — **B8 의 최초 직접 측정**.

### 충실도 gate (실험 유효성의 전제)

replay 된 per-view feature 를 파이프라인 aggregation 그대로 누적 → `sai_nag.pt` 의 실제 SP feature 와 비교:

> **33/33 PASS — cos(recon, sai_nag) = 0.9935 ~ 0.9995**

→ replay 는 실제 파이프라인의 사실상 완벽한 재현. 이하 모든 측정은 실제 파이프라인에 귀속 가능.

### Phase C — 분석 ([scripts/stage3_b8_analyze.py](../../../scripts/stage3_b8_analyze.py))

clean(H2-lite pkl, easy 는 신규 `_h2_lite_easy.pkl`) 과 image_name 으로 paired join. gap = clean − mixed (paired views). rank = scene 전체 prompt 와의 raw-cos 랭킹 (verify_h2lite_rank 와 동일 기준).

---

## 3. 결과

### 3.1 사전 등록 판정 (17 phantom + 16 easy)

| 판정 | 결과 | 상세 |
|---|---|---|
| **H-B8a** | ❌ **NOT confirmed** | strong_signal (n=11) mean gap = **0.0038** — threshold 0.05 의 1/13. (easy 와의 *차이* 는 유의: easy mean −0.0181, MW p=0.020 — 방향은 존재하나 크기가 phantom 설명에 ~13배 부족) |
| **H-B8b** | ❌ **NOT confirmed** | pearson r=0.264 (p=0.137), spearman ρ=0.357 (p=0.041) — 가설 predict r>0.4 미달. phantom mix_rate 0.134 vs easy 0.081 |
| **H-B8c** | **분기 ③** | gap 자체가 작음 → **"within-view 무죄, B1 (across-view visibility-weighted accumulation) 재조준"** |

### 3.2 결정적 표 — per-view mixed rank vs 최종 pipeline rank

B8 직후의 per-view 신호 (median rank, scene prompts 경쟁) 와 최종 SP feature 의 A4 rank (SP pool 경쟁):

| Phantom | subtype | per-view medR (mixed) | **최종 A4 rank** | gap (clean−mixed) |
|---|---|---|---|---|
| tesla door handle | strong | **3** | **256** | +0.019 |
| old camera | mean_dil | 4 | 130 | +0.006 |
| ottolenghi | mean_dil | 14 | 66 | +0.014 |
| pikachu | bimodal | **5** | **64** | +0.010 |
| pumpkin | mean_dil | 5 | 52 | +0.018 |
| spoon (waldo) | strong | **2** | **47** | **−0.014** |
| hand | strong | **1** | **46** | +0.006 |
| cabinet | strong | **2** | **38** | −0.002 |
| onion segments | strong | 7 | 14 | +0.006 |
| sake cup | bimodal | 4 | 10 | +0.006 |
| bowl | strong | 1 | 8 | −0.008 |
| napkin | mean_dil | 7 | 7 | +0.010 |
| jake | strong | 1 | 6 | +0.007 |
| plate | strong | 3 | 6 | +0.008 |
| rubber duck with hat | strong | 1 | 5 | +0.044 |
| bear nose | strong | 1 | 5 | −0.021 |
| sink | strong | 1 | 4 | −0.003 |

읽는 법:
- **strong_signal 11 개 전부 per-view medR ≤ 7, 그중 8 개는 ≤ 3, 5 개는 rank 1** — B8 직후에도 신호 멀쩡
- 유일한 두 자리 per-view rank = ottolenghi (14) — 진짜 encoder limit (Stage 2B rank 재검증과 일관)
- **strong_signal 5 개 (bear nose, spoon, bowl, sink, cabinet) 는 gap 이 음수** — mixing 이 오히려 신호를 *개선*
- easy control 16 개 mean gap = **−0.018** — mixing 은 평균적으로 *도움* (맥락 추가 효과; A2 의 context+7.4%p 발견과 일관)
- tesla: mix_rate 47% (최고) 인데도 per-view rank 3 — 심한 mixing 도 신호를 안 죽임. 그런데 최종 rank 256.

### 3.3 hard-assignment 미리보기 (Stage 3.2 후보 A 의 운명)

hard ≈ mixed (recovery 의미 없음, 일부는 hard 가 더 나쁨). **gap 자체가 없으므로 hard-assignment fix 는 회복할 것이 없다** — Stage 3.2 (full re-run) 는 실행 전에 기각됨. 인과 실험을 fix 보다 먼저 한 설계의 직접적 보상.

---

## 4. 종합 — Stage 3.1 이 답한 것

1. **B8 은 무죄 (확정)**: clean−mixed gap 0.004 (phantom), −0.018 (easy). within-view ratio mixing 은 신호를 죽이지 않으며 평균적으로는 돕는다.
2. **B8 mix-rate 는 phantom 의 predictor 가 아님**: r=0.26 (<0.4), 최고 mix-rate SP (tesla 47%) 도 per-view 신호 건재.
3. **살해 지점이 좁혀짐**: per-view mixed (rank 1-7) → [across-view 누적 + 정규화 + pool 경쟁] → 최종 rank 4-256. **B1 구간이 유일한 용의자.**
4. **Stage 2B 의 결론 수정**: "within-view SAM mixing 이 범인" → 기각. H2-lite proxy 의 gap 은 crop 기하/인코딩 차이였지 mixing 손실이 아니었음.
5. **주의 (해석 한계)**: per-view rank 는 *text-side* 경쟁 (scene prompts 중 몇 등), 최종 A4 rank 는 *SP-pool* 경쟁 — 다른 게임. 따라서 "B1 이 범인" 은 아직 소거법 (between per-view and final, B1 이 유일한 변환). B1 내부의 mechanism (방향 회전인가, 경쟁 SP 의 상대적 강화인가, canon-contrast 효과인가) 은 Stage 3.2-B1 의 question.

### 한 줄로

> **"재료도 멀쩡하고 (clean), 1차 가공도 멀쩡하다 (mixed, rank 1-7). 요리를 망치는 것은 마지막 합치기 (across-view 누적) 다."**

---

## 5. 산출물

### Data
- [output/diagnostics/stage3_b8_replay_perview.pkl](../../../output/diagnostics/stage3_b8_replay_perview.pkl) — 33 SP × 전체 view 의 mixed/hard feature (fp16) + mix stats + 충실도. **B1 해부의 foundation — 추가 GPU 작업 없이 진짜 A3/B1 분석 가능**
- [output/diagnostics/stage3_b8_mix_stats.csv](../../../output/diagnostics/stage3_b8_mix_stats.csv) — per-SP mix 통계 (33 rows)
- [output/diagnostics/stage3_b8_gap_summary.csv](../../../output/diagnostics/stage3_b8_gap_summary.csv) — per-SP gap/recovery/rank (33 rows)
- [output/diagnostics/_h2_lite_easy.pkl](../../../output/diagnostics/_h2_lite_easy.pkl) — easy control 16 개의 clean encoding (신규)
- `data/lerf_ovs/<scene>/language_features/` — 재생성 완료 (~9GB, 재사용 가능 — CLAUDE.md 참조)

### Plots
- `plots/stage3_clean_vs_mixed_vs_hard.png` — 33 SP 의 3-신호 비교
- `plots/stage3_mixrate_vs_gap.png` — mix-rate vs gap scatter (H-B8b)

### Scripts
- [scripts/stage3_b8_replay.py](../../../scripts/stage3_b8_replay.py) — Phase B (replay + 충실도 gate)
- [scripts/stage3_b8_analyze.py](../../../scripts/stage3_b8_analyze.py) — Phase C (사전 판정 적용)
- [scripts/stage3_status.sh](../../../scripts/stage3_status.sh) — 실시간 상태판 (`watch -n 10 bash scripts/stage3_status.sh`)

---

## 6. Decision — Stage 3.2 분기

사전 규칙 분기 ③ 발동: **B1 재조준.** Stage 3.2 = B1 anatomy (이미 가진 per-view dump 만으로 가능, GPU 재작업 불필요):

| 실험 | 질문 | 비용 |
|---|---|---|
| **B1.A — 누적 궤적 추적** | per-view mixed 를 하나씩 누적하며 aggregated feature 의 prompt-cos / pool-rank 가 *언제* 무너지는지 (view 수 함수로) | 반나절, dump 재사용 |
| **B1.B — weighting ablation** | visibility-weighted vs uniform vs top-k-cos view 만 평균 — 어떤 누적이 신호 보존? (가설 문서 B1 의 (a)/(e) ablation) | 반나절, dump 재사용 |
| **B1.C — pool 경쟁 분해** | phantom SP 가 지는 것이 *자기 feature 약화* 때문인가, *경쟁 SP (background drift 의 wrong top-1) 의 상대적 강화* 때문인가 — wrong top-1 SP 도 같은 replay 로 dump 해 비교 | 반나절 + wrong-top1 replay (GPU ~10분) |
| B1.D — canon-contrast 민감도 | raw cos rank vs canon-contrast rank 의 차이가 phantom 에서 큰가 | 2시간, dump 재사용 |

추천: **B1.A + B1.C 병렬** — A 가 "언제 죽는지", C 가 "누구에게 지는지". 둘이 합쳐지면 새 aggregation 의 설계 사양이 나온다.

---

## 7. 관련 문서

- [stage2b_rofa_anatomy.md](stage2b_rofa_anatomy.md) — 직전 stage (이 실험으로 결론 수정됨)
- [../extended_failure_hypotheses.md](../extended_failure_hypotheses.md) — B8 §6.3 설계 + 사전 판정 규칙
- [experiment_results.md](experiment_results.md) — 코드 기반 정의 카탈로그
- [intuition.md](intuition.md) — 직관 요약
