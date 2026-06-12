# Extended Failure Hypotheses — Mechanism-based Taxonomy (6차 revision)

> Generated 2026-06-08. **6차 revision** 2026-06-08 (code-grounded gap audit 통합).
>
> **Framing 진화 흐름**:
> 1. **1차**: D1–D4 너머 가설 (H1–H7)
> 2. **2차**: SAM(1.5%) 아닌 CLIP(85%) 이 핵심 → CLIP-stage 중심 (H1–H13)
> 3. **3차**: Visibility ⊥ mechanism, D2/D3 merge + **mechanism axis** 도입
> 4. **4차**: 6-category (A/B/C/D/E/F) 역할 기반 reorganization
> 5. **5차**: A1=risk-signal vs A3=classifier 분리, A4/B7/C3/D4 추가, A3 dual-side, E1 specific predictions, E4 counterfactual probe
> 6. **6차**: **B8 신설 (within-view mixing)**, F2 test statistic 구체화, E1 vocabulary/null 명시, **B7 Tier 0 격상**, F1 full 5×5 transition, A2 method-matched crop 추가, A3 subtype→fix mapping, C3 λ sweep, E1 검정력 경고
> 7. **6.1 patch (코드 재검토 — critical)**: F2 의 GMM 적용 공간을 **feature vector space** 로 명시 (ROFA 는 mean_sim 스칼라 위에서 filter), B1 의 baseline 을 **visibility-magnitude-weighted mean** 으로 정정 (uniform mean 아님), A3 측정을 **pre-B8 / post-B8 두 변형** 으로 분리 (within-view mixing effect 분리), E1 의 "phantom feature" 를 **direction vector** (aggregated − target text) 로 재정의
> 8. **6.2 patch — measurement 정밀화**: B7 에 **post-ROFA effective purity** 추가 (raw geometric purity 와 별도), B3 sweep 을 **threshold × ROFA on/off 2×2 design** 으로 (raw effect 와 ROFA mitigation 분리), A4 margin 을 **raw / z-score / percentile** 세 정의 (cross-prompt normalization), E1 의 **P3 (instance-level bias) 는 LVIS 대신 spatial-proximity metric** 으로 분리 측정
> 9. **현재 (6.3 patch — 2026-06-10, Stage 1–2B 구현 검증 + B8 인과 실험 설계)**: Stage 1 joint 분해의 **raw/canon robustness 검증 완료** (67 prompt 변화 0), F2 keep-mask 수치를 **실제 pipeline τ=2.0 으로 정정** (실수 4건 — instance confusion 2건 포함), H2-lite 를 **rank 기반으로 재검증** (onion segments 는 encoder 측 재분류, 65% 유지), F2 구현이 6.1 의 feature-space GMM 검정을 **미구현** (cos 스칼라 휴리스틱) 임을 명시, **B8 직접 측정 + 인과 replay (Stage 3.1) 설계 추가** (B8 section 참조)

---

## 6차 revision 의 변경 사항 (code-grounded gap audit)

| Item | 변경 | Why |
|---|---|---|
| **B8 신설** | "Within-view ratio mixing" — `sp_mask_mat @ view_level_feature` 의 단일-view pre-mixing | 코드상 mixing 이 2 단계 (within-view + across-view) 인데 기존 B1/B2/B4 어디에도 within-view stage 가 없었음 |
| **F2 구체화** | bimodal-balanced 정의: GMM(k=2) BIC < GMM(k=1) BIC AND minority cluster ≥ 30% | 기존 "low variance, unimodal-but-shifted" 정의가 모호; ROFA 실제 실패 모드는 *cluster 간 분산이 작아 보이는 다봉* |
| **E1 vocabulary** | scene-disjoint vocab (LVIS 1203) + permutation null 명시 | "open vocab" 만 적혀 있어 순환 위험; null distribution 미정의 |
| **B7 Tier 0** | Tier 1 → Tier 0 (A3/A4 의 prerequisite) | impure / fragmented oracle 위에서 "oracle rank" 해석이 약해짐 |
| **F1 5×5** | D1 (THGS) → 4 classes (ReLaGS) → **full 5×5 양방향** transition | THGS Success 가 ReLaGS 에서 망가진 경우를 못 잡으면 net effect 주장 약화 |
| **A2 +1 crop** | (i)(ii)(iii) + **(iv) method-matched crop** 추가 | method 와 다른 crop 으로 측정한 ceiling 은 method 비교의 valid baseline 이 아님 |
| **A3 mapping** | 두 phantom subtype 의 fix path 를 명시 표로 | subtype 정의만 있고 fix 와의 연결이 누락되어 있었음 |
| **C3 λ** | λ ∈ {0.1, 0.3, 0.5, 0.7, 1.0} sweep, calibration split 으로 결정 | λ 미정의 → 결과 재현/비교 불가 |
| **E1 power note** | section 6.6 추가 — 검정력 한계 + 3DOVS 합산 권고 | 67 prompt 를 4 pattern 분할 시 underpowered |

---

## 0. Main Question — paper 의 한 문장

> **THGS/ReLaGS 의 남은 D2 (semantic distractor) 실패는 단순 CLIP 오답인가, 아니면 multi-view SP feature aggregation 이 만든 phantom 과 실제 encoder/text ceiling 이 섞여 있는가?**
>
> **우리는 cross-view consistency (A3) + rank margin (A4) 로 이 둘을 분리하고, phantom 의 발생 조건 (B), 방향성 (E1), 계층 전파 (E2), ReLaGS 의 한계 (F) 를 정량화한다.**

### D2 의 두 sub-mechanism

| Subtype | 정의 | 직관 | Fix path |
|---|---|---|---|
| **D2.real** | GT crop 단독 + 단일-view evidence 에서도 정답 못 고름 | CLIP/text 자체가 헷갈리는 진짜 어려운 문제 | encoder upgrade, prompt 보강 |
| **D2.phantom** | 어떤 view 에서는 정답 가능하지만 평균/aggregation 후 실패 | feature aggregation 이 만든 가짜 distractor | 새 aggregation 방법 |

### Mixing stage 의 분리 (6차 신설 관점)

코드를 다시 보면 ReLaGS 의 feature mixing 은 **2 단계**로 발생한다:

1. **Within-view mixing** (B8): `sp_feat[view] = sp_mask_mat @ view_level_feature` — 한 view 안에서 SP 가 여러 SAM mask 에 걸치면 ratio-weighted sum 으로 *이미 mixed*
2. **Across-view mixing** (B1, F2): ROFA 가 view 별 (이미 mixed 된) feature 를 평균

기존 가설들은 (2) 만 다뤘으나 (1) 이 phantom 의 *원료* 일 수 있어 분리한다.

---

## 1. Methodological Refinements

### 1.1 A1 은 classifier 가 아니라 **risk signal**
- Per-view feature variance 가 높다고 무조건 phantom 이 아님 — occlusion, 작은 객체, 진짜 multi-view appearance 변화도 variance ↑
- 진짜 판정은 A3 (cross-view consistency) 가 한다
- A1 의 역할: phantom-prone SP 의 *후보 식별*

### 1.2 A2 는 ground truth 가 아니라 **CLIP encoder/text ceiling diagnostic**
- GT crop image-CLIP 도 CLIP 자체의 한계 + crop policy + context 제거 문제를 가짐
- "real 의 ground truth" 가 아니라 "image-CLIP 단독도 못 풀면 encoder-limit" 라는 *upper-bound diagnostic*
- 6차 보강: **method-matched crop (iv) 까지 4 개 비교** — method 의 실제 input policy 와 일치시켜야 valid baseline

### 1.3 A3 는 **dual-side** (target + distractor 둘 다)
- A3-target: oracle SP 의 어떤 view 라도 query rank 가 좋은가?
- A3-distractor: wrong SP 가 평균 feature 에서만 강한가, 단일 view 에서도 강한가?
- 6차 보강: subtype 별 fix path mapping 추가 (A3 본문 참조)

### 1.4 D3 (top-k cardinality) 는 semantic axis 와 **직교한 independent mechanism**
- D2 는 "*누가* 1 등인가?" 의 semantic feature 문제
- D3 는 "*몇 명을* 뽑을 것인가?" 의 decision policy 문제
- → 별도 axis 로 유지. paper 의 "retrieval score 가 맞아도 selection policy 가 망칠 수 있다" narrative 의 기둥

### 1.5 E1 (phantom direction bias) 은 high-risk/high-reward + **재정의**
- Systematic 하게 발견되면 main contribution / random 이면 negative finding
- 6차 보강: ① vocabulary 를 **scene-disjoint** (LVIS 1203) 로 고정 — 순환 logic 차단 ② **permutation null** 로 random 정의 ③ 검정력 부족 시 3DOVS 까지 prompt 합산

### 1.6 F1 (ReLaGS transition) 의 표현 — 공격적이지 않게 + **full 5×5**
- ❌ "ReLaGS 가 D1 을 fake fix 했다"
- ✅ "ReLaGS substantially reduces degenerate retrieval. Our mechanism analysis reveals **whether the recovered cases become true successes or transition into semantic distractor failures**"
- 6차 보강: D1→ReLaGS 한 방향이 아니라 **5×5 양방향 transition matrix** — THGS Success 가 ReLaGS 에서 어디로 갔는지도 포함해야 net effect 가 valid

### 1.7 Mixing stage 의 분리 (NEW)
- Within-view mixing (B8) vs Across-view mixing (B1, F2) 은 *서로 다른 lever*
- RATIO_THRESHOLD, aggregation scheme, ROFA tau 가 각각 다른 단계에 작용
- B8 이 phantom 의 *원료* 를 만들고 B1/F2 가 *증폭/잔존* 시키는 구조

---

## 2. New 6-category Catalog (18 가설)

각 가설마다 **직관 — operationalization — predict — cost** 4 줄로 압축.

---

### A. Measurement — mechanism classifier 의 도구 (4 개)

#### A1. Per-view variance — **risk signal**

- **직관**: SP 의 view 별 feature 가 흔들리는가?
- **측정**: per-SP per-view CLIP feature → view-간 cosine variance, eigenvalue spread, GMM K=1/2 BIC
- **Predict**: variance 가 phantom prone-ness 와 상관 (causal 아님)
- **Cost**: H2 instrument 필요 (1주, classifier 의 prerequisite)

#### A2. Image-CLIP ceiling — **upper-bound diagnostic** (not ground truth) ★ 6차 보강

- **직관**: 깨끗한 GT crop 을 CLIP 에 넣어도 prompt 를 맞히는가?
- **측정**: **4 가지** crop policy 비교
  - (i) **GT polygon tight crop** — pixel-perfect upper bound
  - (ii) **SAM-mask crop** — SAM 의 mask 한도 내 measure
  - (iii) **context-1.5× crop** — context 가 도움 되는지
  - (iv) **method-matched crop** (NEW) — `scripts/image_encoding.py` 의 실제 crop policy 와 동일하게. method 비교의 valid baseline
- **Predict**:
  - 모든 4 crop 으로 image-CLIP 실패 → **D2.real.encoder-limit** (13 cross-invariant 의 N%)
  - (i)–(iii) 성공이지만 (iv) 실패 → **method 의 crop policy 가 ceiling 을 깎는 case** → 별도 fix
  - (iv) 도 성공이면 pipeline aggregation 문제 (phantom 후보)
- **Cost**: 4 시간
- **★ 검증 (6.3)**: joint 2×2 분류는 A2 rank 의 raw/canon 선택에 완전 robust — 67 prompt 분류 변화 0 ([verify_joint_canon_vs_raw.csv](../../output/diagnostics/verify_joint_canon_vs_raw.csv))

#### A3. Cross-view consistency — **dual-side classifier** ★★ 6차 보강

- **직관**: 어떤 단일 view 에서는 정답인데 평균 후 망하는가?
- **★ 정정 (6.1) — "per-view feature" 가 이미 mixed**: 코드상 view-level feature 는 [merge_proj.py:167-168](ReLaGS/merge_proj.py#L167-L168) 의 `sp_mask_mat @ view_level_feature` 결과 — 즉 단일 view 의 feature 도 이미 multi-mask ratio-mix. A3 측정이 B8 (within-view mixing) 의 effect 와 across-view aggregation 의 effect 를 conflate. → **두 변형으로 분리 측정**:
  - **A3-postB8**: 코드 그대로 ratio-weighted view-feature (현 default)
  - **A3-preB8**: 같은 view 에서 **argmax mask 하나만** 사용한 hard-assignment view-feature (within-view mixing 제거)
  - **두 결과의 차이 = within-view mixing (B8) 의 phantom 기여분** → A3 가 B8 와 across-view 의 effect 를 자동으로 분해
- **측정 (target-side, A3-T)**: oracle SP 의 per-view CLIP feature 별로 ranking 재계산 (**A3-postB8 와 A3-preB8 둘 다**). min_view_rank, max_view_rank, mean_view_rank
- **측정 (distractor-side, A3-D)**: 선택된 wrong SP 의 per-view ranking — 평균에서만 1 등인지 일부 view 에서도 1 등인지 (**A3-postB8 와 A3-preB8 둘 다**)
- **분류 logic**:
  ```
  A3-T.min ≤ k (target 어떤 view 에선 top-k) AND A3-D 가 평균 only → target dilution phantom
  A3-T.min ≤ k AND A3-D 도 일부 view 1 등 → distractor inflation phantom
  A3-T.min > k → D2.real
  ```
- **★ Subtype → fix path mapping** (6차 신설):

  | Subtype | Mechanism | 직접 fix path | 검증 가설 |
  |---|---|---|---|
  | **Target dilution** | 정답 view 가 평균에서 희석 | Query-conditioned top-view / max-view aggregation, mode-cluster center | B1, B8, F2 |
  | **Distractor inflation** | wrong SP 가 일부 view 에서 진짜로 강함 (occlusion, viewpoint coincidence) | Visibility gating, negative-prompt contrast, area penalty | D4, C3, D3 |
  | **D2.real** | 모든 view 에서 약함 | Encoder upgrade, prompt expansion, negative contrast | A2, C1, C2, C3 |

- **Predict**: D2 의 X% 가 phantom (target dilution + distractor inflation), Y% 가 real. **Phantom 안에서도 subtype 비율 분포** 가 fix priority 결정
- **Cost**: A1 (H2 instrument) 이후 즉시 계산 가능

#### A4. Oracle-rank margin — **failure difficulty 의 기준 통계** ★

- **직관**: 정답 SP 가 몇 등이고, top distractor 와 점수 차이가 얼마나 나는가?
- **측정**: 각 (prompt, frame) 에서 oracle SP rank, top distractor cosine, cosine 분포 entropy. **per-frame min/mean/max 세 aggregate 동시 보고** (A3 와 일관성)
- **★ Margin definition (6.2 — cross-prompt normalization)**: CLIP cosine 분포는 prompt-dependent (rare word baseline 과 frequent word baseline 다름) → raw margin 으로 cross-prompt 비교 시 같은 값이 다른 ranking 의미. **세 정의 동시 계산**:
  - **raw margin** = `cos(top) − cos(oracle)` — within-prompt 분석에만
  - **z-score margin** = `(cos_top − cos_oracle) / std_within_prompt` — cross-prompt 비교용 (default)
  - **percentile margin** = oracle 이 per-prompt cosine 분포의 몇 percentile (0 = best, 100 = worst) — robust, scale-free
  - "rank-margin 결합 분포" plot, 분류 보조 threshold 모두 **z-score 또는 percentile** 위에서. raw 는 within-prompt drill-down 에만.
- **분류 보조 (z-score margin 기준)**:
  - rank ≤ 3, z-margin 작음 (|z| < 0.5) → calibration / top-k 문제 (D3 후보)
  - rank ≤ 3, z-margin 큼 (|z| > 1.5) → 우리가 못 본 retrieval 성공
  - rank 30+, z-margin 큼 → feature 자체 크게 틀림 (D2.real 가능성 ↑)
  - rank 4–10 → marginal case, A3 로 진단 필요
- **Predict**: rank × z-margin 결합 분포가 D2.real vs D2.phantom 분포를 시각적으로 분리 (raw 로는 cross-prompt 비교 노이즈로 분리 약함)
- **Cost**: 기존 진단 CSV + per-view feature 만으로 즉시. **모든 분석의 기본 통계**

---

### B. Phantom Genesis — phantom 의 발원지 (8 개)

#### B1. Mean aggregation dilution ★ (across-view stage)

- **직관**: 평균 내면서 정답 signal 이 흐려지는가?
- **★ 정정 (6.1) — baseline 의 정확한 정의**: 코드 baseline 은 *uniform mean* 이 아니라 **visibility-magnitude-weighted mean of L2-normalized features, after ROFA cosine outlier filter** ([merge_proj.py:175-176](ReLaGS/merge_proj.py#L175-L176), [merge_proj.py:129-130](ReLaGS/merge_proj.py#L129-L130)):
  - 각 view-feature 가 `visibility_portion = sp_gau_count[view] / sp_total_count` 로 magnitude-scaled 됨
  - ROFA 의 keep_mask 는 cosine 기반이라 magnitude-invariant — visibility 가 keep 결정에는 영향 없음
  - 하지만 `filtered_feats.mean(dim=0)` 의 direction 은 visibility-weighted (high-visibility view 가 더 기여)
  - → 6차의 "mean" 표현은 부정확. visibility 는 *implicit weight* 로 이미 들어가 있음
- **측정**: aggregation scheme ablation — **(default) visibility-weighted mean + ROFA** vs (a) **uniform mean (no visibility weight)** / (b) max / (c) median / (d) **mode-cluster center** / (e) **query-conditioned top-view** / (f) **visibility-only (ROFA off)**
- **Predict**:
  - (d), (e) 가 default 보다 D2.phantom ↓ (target dilution subtype 에서)
  - (a) vs default 의 차이 = **visibility weighting 의 net contribution** — 이게 0 이 아니면 visibility 가 *implicit lever* 임을 입증
  - (f) vs default 의 차이 = ROFA filter 의 net contribution
- **Cost**: H2 instrument 후 1 주
- **Stage**: across-view (cf. B8 = within-view)
- **★★ Stage 3.2 — B1 anatomy 설계 (6.4, 2026-06-11)**: Stage 3.1 이 B8 을 기각하고 B1 을 유일 용의자로 지목 ("per-view mixed rank 1-7 → 최종 rank 4-256"). replay dump ([stage3_b8_replay_perview.pkl](../../output/diagnostics/stage3_b8_replay_perview.pkl): 33 SP × 전체 view 의 mixed feature + portion) 재사용으로 GPU 거의 불필요. **3 실험**:
  - **B1.A — 누적 궤적 + 진짜 A3 (어디서 죽나)**: 각 phantom 의 per-view mixed feature 를 pipeline 순서로 하나씩 누적 (`agg_k = Σ_{i≤k} normalize(f_i)·w_i` 후 normalize) → k 별로 (i) prompt 와 raw cos, (ii) **pool rank** — sai_nag 의 전체 SP pool (levels [2,3]) 에서 해당 SP 의 feature 만 agg_k 로 교체하고 canon-contrast (ClipSimMeasure, A4 와 동일) 로 rank 재계산. **+ 단일 view pool rank** (= A3-postB8 의 진짜 측정): 각 view 의 mixed feature *단독* 을 pool 에 넣었을 때 rank — text-side rank 1-7 이 pool 경쟁에서도 유지되는지가 분기점.
  - **B1.B — aggregation ablation (counterfactual fix, re-run 없이)**: 같은 per-view set 으로 (a) uniform mean (b) visibility-weighted [= baseline, 충실도 gate] (c) top-k by portion (d) **query-conditioned top-k by cos** (k∈{1,3,5,10}, leakage 주의 — upper-bound diagnostic) (e) mode-cluster center (feature-space GMM k=2, majority cluster mean — 6.1 의 GMM 을 여기서 처음 실제 구현) (f) ROFA τ=2 → 각각의 **pool rank** 와 17 phantom 회복 수 + easy 16 regression 수.
  - **B1.C — pool 경쟁 분해 (누구에게 왜 지나)**: 17 phantom 의 wrong top-1 SP (b7_a4 의 clip_top1_lvl/sp_id) 도 같은 replay 로 dump (GPU ~15분) → oracle vs wrong-top1 의 (i) **coherence** = `‖Σ normalize(f_i)·w_i‖ / Σ w_i` (단위벡터 평균의 resultant length — 분산되면 작아짐) (ii) per-view cos-to-prompt 분포 (iii) portion 프로필 (iv) raw-cos rank vs canon-contrast rank 차이 (canon 민감도). 가설: oracle 은 시점별 외형 변화로 분산↑ → 평균 방향이 generic 쪽으로 drift, background SP 는 균질해서 coherent → canon-contrast 경쟁에서 승리.
  - **충실도 gate**: B1.B 의 (b) baseline pool rank 가 b7_a4 의 oracle_rank 와 일치해야 (±2 허용, fp16) 이후 비교가 유효.
  - **사전 판정 규칙 (6.4)**:
    - **R1**: phantom 의 ≥50% 에서 *best 단일 view 의 pool rank ≤ 3* 인데 누적 rank 가 악화 → **averaging 이 인과** (dilution/drift) → fix 방향 = view 선택/클러스터 aggregation. 반대로 best 단일 view 도 pool rank > 3 이 다수면 → text-side rank 는 착시, 문제는 **pool 경쟁/canon-contrast/경쟁 SP 강도** → R3 가 결정.
    - **R2**: ablation 중 query 를 안 쓰는 변형 (a/c/e/f) 이 **17 중 ≥6 을 pool rank ≤ 3 으로 회복 AND easy 16 regression ≤ 1** → Stage 4 method 후보로 채택 (그때 full re-run). query-conditioned (d) 만 회복하면 → method 는 query-aware aggregation 으로 설계.
    - **R3**: oracle coherence < wrong-top1 coherence 가 paired 로 유의 (Wilcoxon p<0.05) → "dispersion-defeat" mechanism 확정 → mode-cluster/query-conditioned 의 이론적 근거. 유의하지 않으면 → canon-contrast (iv) 또는 경쟁 SP visibility 가 lever.
- **★★ 결과 (2026-06-11, Stage 3.2 완료 — B1 = 확정 범인, mechanism 분해 완료)**: 충실도 gate 26/33 strict (초과 7건은 deep-rank 의 ±5-10% 상대오차; 질적 이탈은 jake 1↔6 단 1건 — 회복 카운트에 병기). **R1 ✅** — phantom 의 88% (15/17) 가 *단일 view* 로 pool rank ≤3 가능 (대부분 rank 1) 인데 최종은 1/17 → averaging 인과 확정. 단 **좋은 view 의 비율이 18% (easy 는 50%)** — phantom = 소수 탁월 view 가 다수에 묻히는 현상. **R2 ✅** — 정적 top-k-by-portion 6/17 (reg 0, 경계 통과); **query-conditioned top-k-by-cos 가 14-15/17 회복 + easy regression 0** → Stage 4 = query-aware aggregation. GMM mode-cluster (e) 는 3/17 + reg 6 으로 기각, uniform mean 은 개악 (reg 8 — visibility weighting 은 필수 성분), ROFA τ=2 는 1/17 (무력 재확인). **R3 ✅** — dispersion-defeat 확정 (coherence O 0.870 < W 0.886, Wilcoxon p=0.0128). **강화 발견**: wrong-top1 의 per-view 평균 prompt-cos 가 oracle 보다 *높음* (0.227 vs 0.220, p=1.0 역방향) — 경쟁자는 '균질하게 그럴듯한' SP 라 **mean-vs-mean 게임은 oracle 에게 구조적으로 불리** → 평균 계열 개선 (B1 의 (a)-(f) 중 mean 기반 전부) 으로는 원리적 회복 불가, view 선택만이 길. 부수 발견: pumpkin 의 wrong-top1 = zero-norm degenerate (canon(0)=0.5 로 승리 — **D1 메커니즘이 phantom 내부에 잔존**, zero-norm 필터 1줄 과제). 잔여 한계: spoon·cabinet 은 never_good_in_pool (어떤 단일 view 도 top-3 불가) — aggregation 으로 회복 불가. 상세: [experiments/stage3_2_b1_anatomy.md](experiments/stage3_2_b1_anatomy.md). **✱ 적대적 검증 정정 (같은 날)**: 15/17 은 *동결-경쟁자 상한* — 공정 결투 (wrong-top1 도 query-top-5 보정, [stage3_2_fair_duel.csv](../../output/diagnostics/stage3_2_fair_duel.csv)) 에서 **oracle 단독 승리는 3/15**, fair-rank ≤3 은 13/15 (대부분 rank 2 — wrong 이 1등 유지) → query-aware 는 phantom 을 "실종→후보권 복귀" 시키는 것이고 결정적 분리는 결합 신호 (D1 필터·coherence prior·공간 정합·negative contrast) 필요. R1/R3 진단은 유지·강화, R2 의 회복 수치만 하향.
- **★★★ Stage 3.3 — 확정 진단 설계 (6.5, 2026-06-11)**: 남은 구멍 2개를 닫는다.
  - **3.3-A — full-pool query-aware + mask-IoU 확정 평가**: (1) stage3_b8_replay 의 view 루프에서 이미 계산되는 전체-SP feature 행렬 (`sp_feat`, levels 2·3) 을 **저장하도록 확장** (`--dump_all_sp`, fp16, ~300MB/scene, 런타임 동일) → 4 scene 전체 SP per-view dump. (2) **전 pool 동시 query-aware 재채점**: 각 prompt 에 대해 모든 SP 의 `score = canon(topk_agg(views, query, k=5))` — 동결 없는 진짜 랭킹. variants: plain / +zero-norm 필터 / +coherence penalty (`score − λ·coherence`, 사기꾼의 균질성 감점) / λ sweep. (3) **mask-level 확정**: 각 variant 의 top-3 union 을 렌더링해 **67 prompt 전체 mIoU** (stage2b_d3 機構 재사용) — baseline 0.542 와 비교. **R4 판정 (사전 등록)**: 어떤 training-free variant 가 *full-67 mIoU ≥ baseline + 2pt AND phantom-17 mean IoU 개선 AND easy mIoU 손실 < 1pt* → **Stage 4 method 로 승격** / 미달 → query-aware 단독 한계 확정, 결합 신호 재설계.
  - **3.3-B — 사기꾼 승리 view 법의학**: 각 phantom 의 wrong-top1 이 이긴 top-5 query-cos view 에서 (i) wrong mask ∩ GT 투영 overlap (객체 포함 여부) (ii) crop 면적비 (wrong/oracle) (iii) oracle vs wrong top-5 view 의 4-panel montage. **R5 판정 (사전 등록)**: wrong 의 승리 view 중 **GT-포함률 ≥ 50%** → **"frame-the-truth" mechanism** 확정 → 처방 = tightness/purity-aware scoring (area-normalized, complement-contrast) / **< 50%** → 진짜 semantic confusion → 처방 = negative contrast·공간 분리. (Stage 2A 의 background_drift 65% 명명을 mechanism 수준에서 재검증하는 실험이기도 함.)
  - 비용: 3.3-A dump ~10분 GPU + 채점 ~수십분, mask eval ~30분 GPU; 3.3-B 렌더 ~20분. 합계 반나절.
- **★★★ 결과 (2026-06-11, Stage 3.3 완료 — R4 FAIL, R5 semantic confusion, hybrid 가 Stage 4 출발점)**: gate — all-SP dump 충실도 98.2~100%, baseline mIoU 0.5424 = stage2b 0.542 재현, **zero-norm 유령 150개 전수 확인** (cross_method 의 D1 150 과 독립 일치). **R4 ❌ 전 variant FAIL**: plain query-top-5 가 phantom IoU 를 **0.203→0.402 (+19.9pt, 2배)** 올리지만 easy −9.8pt → full-67 mIoU −0.1pt **제로섬**. rank 수준: phantom +6 / other +5 / easy **−10** — Stage 3.2 의 "15/17" 은 동결-경쟁자 환상으로 최종 확정. coherence 감점 무익. **R5: frame-the-truth 기각 (GT-포함 2/16)** → semantic confusion 분기 — 단 5/16 은 *진실의 조각* (precision ≥0.87, tesla 의 wrong = 손잡이 일부 → granularity 문제), 10/16 이 진짜 배경 confuser. **✱ 사후 탐색**: hybrid `α·mean + (1−α)·top5`, α=0.3 이 rank 순 +9 (phantom 4, other 7, easy −2) 로 순수 top-k (+1) 압도 → **Stage 4 = hybrid 의 mask-수준 검증 (scene-split calibration 필수) + zero-norm 필터 + 계층 union (part-of-truth 용) + hybrid 위 C3 재시도**. 상세: [experiments/stage3_3_definitive.md](experiments/stage3_3_definitive.md).
- **★★★★ Stage 3.4 — 잔여 원인 전수 분해 설계 (6.6, 2026-06-11, method 설계 전 최종 진단)**: 대다수 원인 (소수파 매장) 으로 설명 안 되는 잔여를 전수 분류해 **case→원인→처방 신호의 완전한 매핑 표**를 만든다. 전부 기존 dump (stage3_3_allsp_*, selections, wrongtop1) 재사용 — GPU 는 승자 렌더링 (~30분) 뿐.
  - **3.4-A — 패자부활 실패자 해부**: never_good 2 (spoon, cabinet) + query-aware 악화 5 (sake cup, sink, onion, cabinet, spoon) 에 대해, **두 채점 체제 각각의 top-10 승자 SP** 를 식별 → 렌더링 → 분류: (i) **같은 카테고리의 다른 인스턴스** (multi-instance — GT 미표기 실물; 시각 montage + 전 frame GT 와의 대조) (ii) oracle 의 NAG 부모/자식/형제 (iii) 무관 배경. 
  - **3.4-B — easy 역행 10건 해부**: query-top-k 에서 easy 를 새로 이긴 SP 들의 정체 + "왜 mean 에선 못 이겼나" (coherence/n_views/topk-cos 분포) → **hybrid 의 가드 신호 설계 데이터** (예: mean-score 와 topk-score 의 괴리, 승자의 view 수 등).
  - **3.4-C — NAG 계층 감사 (granularity 5건)**: tesla·rubber duck·jake·old camera·sink 의 wrong SP 가 oracle 의 **자식/부모/형제인지 labels 계층에서 직접 확인** → 자식이면 "parent-union 1줄" 처방 확정.
  - **3.4-D — 다중 인스턴스/annotation 감사**: generic prompt (spoon, plate, bowl, cabinet, napkin 류) 에 대해 scene 내 같은 카테고리 후보 SP 들을 렌더해 GT 가 일부 인스턴스만 표기했는지 정량화 — 일부 "실패" 가 benchmark artifact 인지 판정 (paper 의 정직성 + 새 method 평가의 공정성).
  - **3.4-E — 종합 taxonomy**: 26 실패 prompt (phantom 21 + real 5) + easy 역행 10 전부에 primary cause 라벨 + "어떤 신호가 고치나" (hybrid / zero-norm / parent-union / negative-contrast / 평가보정 / 불가) 매핑 → **Stage 4 method 요구사항 명세서**.
  - **사전 판정**: **R6** — 17 phantom + 10 역행 전원이 증거 기반 cause 라벨을 받아야 완료 (unknown ≤ 2 허용). **R7** — multi-instance 비율 정량 (≥3건이면 평가 프로토콜 보정을 Stage 4 에 포함). **R8** — 2B 의 mean_dilution/encoder_hidden 분류 중 query-aware 로 회복된 케이스는 재분류 (encoder limit 의 최종 명단 확정).
- **★★★★ 결과 (2026-06-11, Stage 3.4 완료 — R6 PASS / R7 발동 / R8 전원 재분류)**: 36 케이스 전수 라벨 완료 (unknown 2: bowl, napkin). **신규 원인 ① few-view opportunist**: spoon/cabinet 의 mean-체제 승자들은 전부 GT-무관 외부인데 **n_views 2-11, coherence 0.92-0.99** — view 가 적으면 평균 희석이 없어 순수 feature 유지 → **visibility-weighted mean 은 저관찰 SP 에게 구조적으로 유리** (B1 의 새 따름정리; 처방 = min-evidence prior). **신규 원인 ② multi-instance 의심 (R7 발동)**: 7 prompt 46건 — 주방류 generic prompt 의 "실패" 일부는 벤치마크 미표기 가능성 (montage 증거, Stage 4 평가 보정 필수). **3.4-C 친족**: granularity 4건 확정 (jake≈same, rubber duck·tesla=자식, sink=부모 → parent-union 처방), sibling 2 (onion↔egg 같은 그릇!), unrelated 11. **3.4-B 가드 신호**: 역행 승자의 top5−mean gap median +0.17 (g1), 역행의 60% 는 oracle 이 mean 1등 (g2), **역행 10 중 3 은 kin_same 의 가짜 역행** (rank 착시, mask 무해). **R8**: 2B "encoder 측" 4/4 전원 재분류 (pumpkin 52→1, ottolenghi 66→3 등 — 극단적 dilution 이었음) + **Stage 1 의 D2.real 5 중 4 도 회복** (miffy 포함!) → **encoder-limit 최종 잔여 = 3건 (waldo plate, pot, yellow desk)** — "encoder 한계 7.5%" 는 과대 추정이었음. 부수: sake cup 케이스에서 oracle-rank 진단의 맹점 발견 (다른 GT-커버 SP 가 rank 3 — rank 기반 phantom 분류는 보수적 과대 카운트 가능). **Stage 4 요구사항 확정**: 가드된 hybrid + parent_union + min-evidence + zero-norm 필터 + multi-instance 분리 평가. 상세: [experiments/stage3_4_residual_causes.md](experiments/stage3_4_residual_causes.md).
- **★★★★★ Stage 4 — method 검증 설계 (6.7, 2026-06-12)**: taxonomy 가 명세한 결합 채점기를 구현하고 **mask mIoU + leave-one-scene-out (LOSO)** 으로 입증한다. 전부 기존 인프라 재사용 (all-SP dump → 채점 → selections → mask eval).
  - **채점기 (training-free, query-time)**:
    `score(SP,q) = α·canon(mean_feat) + (1−α)·canon(topk_q_feat)` 에 결합 신호 4종:
    ① **zero-norm 필터**: 유효 view 0 인 SP 제외 ② **min-evidence**: `n_valid_views < τ_v` 인 SP 는 top-k 항 비활성 (mean 항만) ③ **g1 가드**: `(topk항 − mean항) > τ_g` 인 SP 의 top-k 항을 mean 항으로 클램프 (lucky-view 점프 차단) ④ **선택 후 parent-union**: top-3 선택 SP 의 NAG 부모/자식이 점수 ε 이내면 union 에 포함 (granularity).
  - **Grid**: α ∈ {0.2, 0.3, 0.4, 0.5} × k ∈ {3, 5, 10} × τ_v ∈ {0, 5, 10} × g1 on/off × parent-union on/off (g1 의 τ_g 는 calibration scene 에서 역행-차단/회복-보존 trade-off 로 결정).
  - **Calibration 규율 (필수)**: **LOSO** — 3 scene 으로 hyperparam 선택, 남은 1 scene 으로만 평가, 4회 회전 → held-out mIoU 의 평균이 유일한 headline. 67 prompt 전체에 직접 튜닝 금지.
  - **평가**: (i) full-67 held-out mIoU vs baseline 0.5424 (ii) 그룹별 (phantom17 / easy / other) (iii) **multi-instance 의심 7 prompt 분리 트랙** (포함/제외 양쪽 보고) (iv) 변형별 ablation 표 (각 신호의 기여 분해 — taxonomy 의 케이스 예측과 대조).
  - **사전 판정 R9**: held-out full-67 mIoU ≥ **baseline + 2pt** AND easy 그룹 손실 < 1pt → **method 확정 (paper Section 3 진입)** / +0.5~2pt → partial (신호 조합 재탐색 1회 허용) / 미달 → 정직 보고 + 잔여 설계 question 도출.
  - **사전 판정 R10 (확장, optional)**: 동일 채점기를 ReLaGS sai_nag 에 적용 (ReLaGS 용 all-SP dump 필요 — ReLaGS 경로 replay 는 ROFA 포함이므로 별도 검증 gate) → method-agnostic claim. 시간 없으면 Stage 5 로 이월.
  - 비용: 채점 grid (GPU-경량, ~1h) + mask eval (선두 3-4 config × 208쌍, ~1.5h) + 문서화. ReLaGS 확장 시 +2-3h.
- **★★★★★ 결과 (2026-06-12, Stage 4 완료 — R9 PARTIAL 최종, 재탐색 1회 소진)**: 정합성 gate 완벽 (① 0/67 ② 0/67 정확일치). **v1** (+0.76pt; phantom +13.9, easy −5.4) → ablation 이 g1 전역 클램프의 설계 결함 적발 (끄면 +8.7pt — 정답의 점프까지 억압) → **재탐색 1회 (v2)**: g1 제거 + 3.4-B 의 g2 (prompt-적응 mean-confidence 가드) 추가 → **held-out full-67 +3.74pt (0.5424→0.5798, 기준 +2.0 통과)**, phantom **+17.5pt (0.203→0.378)**, other **+13.3pt**, multi-instance 트랙 제외 시 **+4.53pt**. 단 **easy −4.08pt 로 보호 기준 (<1pt) 미달 → R9 PARTIAL 확정**. g2 사후 부검: canon margin 스케일 포화 (easy median 0.013) 로 41 중 1개만 보호 — 가드 *개념* 은 손실 분해 (보호 −0.17 vs 미보호 −1.50) 로 지지되나 **margin 은 부적합한 분리자** (plastic ladle: IoU 0.84 인데 margin 0.0002 → 0.00 으로 전멸). → **Stage 5 question = 가드 분리자 재설계** (체제 간 top-1 일치 / pool z-score / rank-stability), R10 (ReLaGS) 과 원저자 평가 경로 교차확인 대기. 상세: [experiments/stage4_method.md](experiments/stage4_method.md).
- **★★★★★★ Stage 5 — 메커니즘 일반화 설계 (6.8, 2026-06-12): Stage 3 의 진단 실험을 ReLaGS 자체 파이프라인 위에서 재현**. 동기: 메커니즘 증거 (소수파 매장·제로섬·승자 시그니처) 가 전부 THGS 한정 — "paradigm-level 결함" 주장의 마지막 빈칸. ReLaGS 는 자체 partition + **ROFA 방어기제**를 가진 파이프라인이라, 재현되면 "ROFA 조차 못 막는다" 까지 입증됨. method 고도화 (가드 재설계) 는 이 뒤로 — ReLaGS 케이스가 가드 설계의 모집단을 2배로 만들기 때문.
  - **전제 (Phase A) — ReLaGS replay + dump**: ReLaGS 의 sai_nag (자체 partition — THGS dump 재사용 불가; 위치·로딩은 [ReLaGS/scripts/b7_a4_oracle_analysis.py](../../ReLaGS/scripts/b7_a4_oracle_analysis.py) 의 호출 방식과 [md/RELAGS/relags_setup_verification.md](../RELAGS/relags_setup_verification.md) 참조) 에 대해 stage3_b8_replay 를 이식. **ReLaGS 경로의 차이**: per-view feat 는 동일 (within-view ratio mixing, RATIO 0.3, WEIGHT 0.0001) 하나 **최종 합치기 = ROFA (τ=2, [ReLaGS/merge_proj.py:96-130](../../ReLaGS/merge_proj.py#L96-L130), keep 후 visibility-scaled mean)** → **충실도 gate 도 ROFA 포함 재구성**으로 ReLaGS sai_nag feat 와 cos≥0.95 (all-SP 통과율 + zero-norm 수 보고 — cross_method 의 "D1 86% 감소 (~21 유령)" 와 대조). all-SP dump (levels [2,3]) 산출.
  - **G1 — 소수파 매장 재현**: ReLaGS 의 phantom 명단 (cross_method_d2_decomposition.csv 의 relags_class=phantom, ~20) + easy control 에 대해 단일 view pool rank (ReLaGS 자체 pool) + 좋은 view 비율. **판정**: best 단일 ≤3 비율 ≥60% AND 좋은-view-비율의 phantom<easy 격차 유의 (THGS: 88%, 18% vs 50%) → 재현.
  - **G2 — 제로섬 재현**: full-pool query-top5 재채점 (rank 수준 primary; 시간 되면 mask 까지) → phantom 회복과 easy 역행이 동시 발생하는가, net 은 어느 쪽인가 (THGS: +6/−10, mask ±0.1pt).
  - **G3 — 승자 시그니처 재현**: ReLaGS 의 회복불능 케이스 승자들의 nv/coherence/GT-IoU (THGS: few-view opportunist nv 2-11 / coherent confuser).
  - **G4 — ROFA 의 실측 net 효과 (ReLaGS 고유, 신규)**: 같은 dump 에서 ROFA on/off 재구성 → phantom/easy rank 에 대한 ROFA 의 실제 기여 정량 — Stage 2B 의 시뮬레이션 결론 ("무죄, outlier_handled 0%") 을 실제 파이프라인 분포에서 확정.
  - **판정 G5 (종합)**: G1·G2·G3 모두 재현 → **"메커니즘은 method-agnostic — ROFA 도 못 막는다" 확정** (paper Section 2 완결) / 부분 재현 → 차이를 THGS-특이 요소로 명시 / 비재현 → 진단의 범위 축소를 정직 보고.
  - 비용: language features 공유 (재생성 불필요), replay+dump ~30분 GPU, 분석은 stage3_2/3_3 기계의 경로 일반화 재사용 — 합계 반나절.
- **★★★★★★ 결과 (2026-06-12, Stage 5 완료 — G5: 전부 재현, method-agnostic 확정)**: Phase A gate — ReLaGS ROFA-포함 재구성 충실도 97.8~100% (타깃 36/36), **유령 21 = cross_method 의 "150→21 (86% 감소)" 독립 재확인**. **G1 ✅** — best 단일 view ≤3 = **95%** (THGS 88%), 좋은 view 비율 **23% vs 51%** (THGS 18%/50%, p<0.0001). **G2 ✅** — query-top5 전면 적용 시 phantom **+8** / easy **−12** (THGS +6/−10) — 제로섬 구조 재현. **G3 ✅** — 회복불능 phantom 승자: GT-무관 82%, coherence med 0.89, few-view 33%; easy 역행 승자: GT-무관 90% (lucky-view jump 동일 패턴). **G4 (신규 확정)** — ROFA 의 실측 net 효과: phantom rank 에 **median 0** (helped 6/hurt 5/neutral 9), rank≤3 구출 **1/20** — 2B 시뮬 ("outlier_handled 0%") 의 실제 파이프라인 확정판. **종합: "소수파 매장은 paradigm-level — ROFA 는 유령(D1) 청소부일 뿐 dilution 은 못 막는다."** 부산물: ReLaGS all-SP dump 1.2GB → **R10 (채점기의 ReLaGS 적용) 즉시 가능**, 가드 재설계의 모집단 2배 확보. 상세: [experiments/stage5_relags_replication.md](experiments/stage5_relags_replication.md).

#### B2. SAM multi-mask contamination

- **직관**: SAM mask 가 옆 객체/배경 feature 를 같이 먹는가?
- **측정**: small/medium/large mask 별 contamination rate (mask 가 GT object boundary 를 얼마나 넘는지)
- **Predict**: large mask 가 contamination 의 주범 → SP 가 large 의 영향 많이 받을수록 phantom rate ↑
- **Cost**: 반나절 (0 추가 학습)

#### B3. WEIGHT_THRESHOLD over-rescue

- **직관**: 거의 안 보이는 view 까지 평균에 들어가서 오염시키는가?
- **★ 측정 (6.2 — 2×2 design)**: threshold ∈ {0.01, 0.001, 0.0001, 0.00001} × **ROFA on/off** 2 차원 sweep × mIoU + D2.phantom rate:

  | | ROFA on (default) | ROFA off |
  |---|---|---|
  | **Low** (0.0001 = ReLaGS default) | net effect — ROFA 가 잡으면 phantom ↓, 못 잡으면 ↑ | B3 raw effect — phantom ↑ 의 magnitude |
  | **High** (0.01 = THGS default) | baseline | B3 baseline (ROFA 영향 거의 없음) |

  - 1D sweep 만 하면 B3 의 raw effect 와 ROFA 의 mitigation 이 conflate. 2D 가 분리해줌.
  - **note**: `render_point` 의 weight 만 캐싱하면 pipeline 재실행 없이 threshold 만 re-apply 가능 — cost 가 시간 단위로 떨어짐
- **★ Method comparison confound (6.2)**: THGS default=0.01 vs ReLaGS default=0.0001 — 이 차이 자체가 method 비교의 confound. F1 (transition matrix) 해석 전에 **fixed threshold (e.g., 0.001) 에서 두 method 다시 평가** 권장. 안 그러면 ReLaGS 의 D1 86% 감소가 ROFA 의 contribution 인지 단순 threshold 차이의 결과인지 분리 안 됨.
- **Predict**:
  - (Low, ROFA off): D1 큰 폭 감소 + D2.phantom 큰 폭 증가 → B3 raw effect magnitude 정량
  - (Low, ROFA on): D2.phantom 증가 폭이 (ROFA off) 보다 작아야 ROFA 의 mitigation 입증
  - (Low, ROFA on) vs (Low, ROFA off) gap 이 작으면 → ROFA 가 거의 모든 phantom 잡음 / 크면 → ROFA blindness 의 magnitude (F2 와 cross-check)
- **Cost**: weight 캐시 가능 시 1 일, 아니면 1 주

#### B4. Background bleed via `zero_scale`

- **직관**: crop padding / 배경 픽셀이 작은 객체 feature 를 잡아먹는가?
- **측정**: zero_scale ∈ {0.0, 0.1, 0.2 (default), 0.5} 객체 면적 bin 별 mIoU
- **Predict**: 작은 객체 prompt 에서 zero_scale=0 이 mIoU 개선 → 배경 phantom 입증
- **Cost**: 1 일

#### B5. Small crop noise

- **직관**: 너무 작은 SAM crop 은 CLIP encoding 이 불안정한가?
- **측정**: per-mask crop size vs CLIP feature entropy / rank 안정성
- **Predict**: crop size < N 픽셀이면 entropy ↑, contribution 가중치 down-weight 시 phantom 감소
- **Cost**: 3 일 (encoding 단계 instrument)

#### B6. View-count sparsity ★

- **직관**: 적은 view 에서만 보인 SP 가 더 phantom-prone 한가?
- **측정**: SP 별 visible-view 수 distribution → A3 phantom rate 와의 상관관계
- **의미**: partition geometry 와 semantic reliability 의 연결 — 단순 semantic 문제가 아니라 *3D coverage 의 함수*
- **Cost**: 1 일 (분석만)

#### B7. Oracle SP purity & completeness ★ **6차 Tier 0 격상**

- **직관**: 정답 SP 가 정말 깨끗한 후보인가, 아니면 GT + distractor 가 섞인 후보인가?
- **측정**:
  - **purity (geometric)** = `|SP ∩ GT| / |SP|` — SP 의 GT 비중
  - **completeness** = `|SP ∩ GT| / |GT|` — GT 의 SP 비중
  - **fragmentation** = GT 를 covering 하는 SP 의 개수
  - **★ post-ROFA effective purity (6.2 추가)**: per-(SP, query) 의 ROFA-kept view subset 이 만드는 effective feature 의 implicit purity — kept view 들의 SAM mask 가 GT 와 얼마나 잘 맞는지. 정의: `effective_purity = mean_over_kept_views(|mask_view ∩ GT_proj_view| / |mask_view|)`. **Raw geometric purity 와 별도 axis**.
- **Predict 4 axis (6.2 — axis 1 개 추가)**:
  - purity 낮음 → SP 안에 다른 객체 섞임 → feature contamination 의 root
  - completeness 낮음 → object 일부만 → feature 약함
  - fragmentation 높음 → top-k/cardinality (D3) 로 연결
  - **★ raw purity − effective purity 의 gap 큼** (6.2): ROFA 가 *wrong direction 으로* outlier 정의 (GT-aligned view 를 drop, distractor view 를 keep) → ROFA-induced phantom 의 hidden source. F2 의 mean-dilution subtype 과 같은 SP 인지 cross-check
- **의미**: **oracle existence ≠ oracle retrievability**. A3/A4 의 "rank" 해석이 oracle 의 purity 가정 위에 서 있으므로 **B7 은 A3/A4 의 prerequisite** — 6차에서 Tier 0 로 격상. **post-ROFA effective purity 는 A3-postB8 결과의 baseline**.
- **Cost**: 반나절 (0 추가 학습)

#### B8. Within-view ratio mixing ★ NEW (6차 신설)

- **직관**: 한 view 안에서 SP 가 여러 SAM mask 에 걸치면, 그 view 의 contribution 이 *이미* CLIP mixed feature 다. 평균 전부터 오염이 시작된다.
- **Code 근거** [merge_proj.py:163-168](ReLaGS/merge_proj.py#L163-L168):
  ```python
  sp_mask_mat = self._get_superpoint_mask_ratio(gau2sp, pt_sp_label[view_idx], seg_num)
  sp_mask_mat[sp_mask_mat < RATIO_THRESHOLD] = 0    # 0.3
  sp_feat = sp_mask_mat @ view_level_feature        # ← ratio-weighted SUM
  ```
  RATIO_THRESHOLD=0.3 은 잡것은 거르지만 **비등한 두 후보 (0.45 + 0.40) 는 둘 다 통과**. 한 view 가 두 mask 의 CLIP 가중합으로 ROFA 에 들어감.
- **측정**:
  - per-(SP, view) 별 `(sp_mask_mat[sp] ≥ 0.3).sum()` 의 분포
  - "view-level mix count" = 1 이면 깨끗, ≥ 2 이면 view 단위로 이미 혼합
  - **mix-rate** = 한 SP 의 전체 view 중 mix count ≥ 2 인 view 비율
- **Predict**:
  - SP 의 mix-rate 와 D2.phantom rate 의 정상관 (correlation > 0.4 예상)
  - RATIO_THRESHOLD ∈ {0.3, 0.5, 0.7} sweep:
    - 0.3 → 0.5: phantom ↓, 일부 view 가 contribution 잃음 → D1 borderline ↑
    - 0.5 → 0.7: phantom 추가 ↓, D1 추가 ↑ (trade-off curve 측정)
  - **Hard assignment 비교**: 비율 mix 대신 argmax mask 만 사용한 ablation 이 phantom 을 얼마나 줄이는지
- **B1/B2 와의 분리**:
  - B1 = across-view stage (이미 view-level feature 가 있다는 가정 하에 어떻게 합칠지)
  - B2 = SAM mask 자체의 boundary 오류
  - B8 = SP 와 SAM 의 *비대칭 partition* 으로 인한 within-view mixing — partition mismatch 의 결과
- **의미**: F2 (ROFA mean-limit) 의 motivation 을 한 단계 깊게. ROFA 입력 자체가 이미 mixed 상태이므로 ROFA 가 잡을 수 없는 phantom 의 source
- **Cost**: 반나절 (기존 데이터만)
- **★ 측정 현황 (6.3, 2026-06-10)**: 위 "측정" 은 아직 **한 번도 직접 수행되지 않음** — Stage 2A Layer 1 은 connected-component *시각 proxy* (공간 fragmentation, B8 의 feature mixing 과 다른 것) 였고, Stage 2B 의 evidence 는 H2-lite 의 간접 대비 (clean 강함 vs pipeline 죽음) 뿐. 그 대비에는 SAM mask 기하 / crop 정책 / visibility weighting 차이도 섞여 있어 **B8 의 인과는 미확정**. `language_features/` 부재가 원인이었음.
- **★ Stage 3.1 — B8 인과 replay 설계 (6.3)**: 17 persistent phantom 의 oracle SP (+ scene 당 4 easy control) × 30 sampled views 에서 merge_proj 의 within-view 단계를 그대로 재현, 같은 view 에서 세 신호를 비교:
  1. **mix_count** = `(sp_mask_mat[sp] ≥ 0.3).sum()` — **B8 의 최초 직접 측정** (mix-rate = mix_count ≥ 2 인 view 비율)
  2. **mixed_cos** = `normalize(sp_mask_mat[sp] @ view_level_feature)` 와 prompt 의 raw cos — post-B8 신호
  3. **hard_cos** = `view_level_feature[argmax(sp_mask_mat[sp])]` 의 cos — *hard-assignment fix (Stage 3.2 후보 A) 의 per-view 미리보기, full re-run 없이*
  - H2-lite 의 clean_cos 와 view 단위 join → **clean − mixed gap = within-view 손실의 직접 정량**
  - **충실도 체크 (필수 gate)**: replay 된 per-view feature 를 pipeline aggregation 으로 재구성 → `sai_nag.pt` 의 실제 SP feature 와 cos ≥ 0.95. 미달이면 replay 가 pipeline 과 다른 것 — 결과 무효.
  - **사전 판정 규칙**:
    - ① strong_signal 11 에서 mean(clean − mixed) ≥ 0.05 AND easy control 대비 유의 → **B8 손실 확인**
    - ② mix-rate ↔ gap 상관 r > 0.4 → 본 가설의 predict 적중
    - ③ hard_cos 가 gap 의 **≥ 70% 회복** → Stage 3.2 (hard-assignment full re-run) **GO** / **< 30%** → 범인은 ratio-sum 이 아니라 argmax mask 자체의 오염 = **B2 로 pivot** / gap 자체가 작으면 → within-view 무죄, **B1 (visibility weighting) 재조준**
- **★★ 결과 (2026-06-10, Stage 3.1 완료 — 가설 기각)**: 충실도 gate 33/33 (cos 0.9935–0.9995). **① 기각** — strong_signal 11 mean gap = 0.0038 (threshold 의 1/13; easy 는 −0.0181 로 mixing 이 오히려 개선). **② 기각** — pearson r=0.264 (phantom mix_rate 0.134 vs easy 0.081). **③ 분기 ③ 발동** — gap 자체가 없어 hard-assignment 는 회복할 것이 없음. **결정적 관측**: 17 phantom 의 per-view mixed rank 1–7 (ottolenghi 14 만 예외 = encoder limit) vs 최종 A4 rank 4–256 → **B8 은 무죄, 살해 지점은 across-view 누적 (B1)**. 상세: [experiments/stage3_b8_causal.md](experiments/stage3_b8_causal.md). → **B1 이 Tier 0 급 최우선 가설로 격상** (replay dump 로 GPU 재작업 없이 검증 가능).

---

### C. Real Ceiling — real competition 의 한계 (3 개)

#### C1. CLIP encoder OOD ceiling

- **직관**: CLIP 이 kamaboko, ottolenghi 같은 prompt 를 애초에 아는가?
- **측정**: 67 prompt 를 rare-word / brand-name / object-name / generic 으로 분류 → 각 category 별 image-CLIP top-K
- **Predict**: rare/brand 가 D2.real.encoder-limit 의 majority
- **Cost**: 4 시간 (A2 와 함께)

#### C2. Short prompt under-discrimination ★

- **직관**: "bowl" 같은 한 단어 prompt 가 scene 내 후보 구분에 너무 약한가?
- **측정 (주의: leakage)**: 두 가지로 분리
  - **Generic expansion** (method-safe): "bowl" → "a bowl, a round container used for food"
  - **Image-aware expansion** (diagnostic only): GT crop 본 뒤 "yellow ceramic bowl" — **evaluation leakage 위험, upper bound diagnostic 로만 사용**
- **Predict**: generic expansion 으로도 mIoU 일부 회복 → C2 가 method 후보; image-aware 만 회복하면 encoder limit
- **Cost**: 반나절 (text-side 만)

#### C3. Negative prompt contrast ★ **6차 보강**

- **직관**: "bowl vs cup vs mug" 헷갈리는 후보를 text-side contrast 로 분리 가능한가?
- **측정**:
  ```
  score(SP, query) = cos(SP, text_query) − λ · max_i cos(SP, text_negative_i)
  ```
  - negatives = scene-local 카테고리 (현 scene 의 다른 prompts, query 자기 자신 제외)
  - **λ sweep**: λ ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
  - **Calibration**: scene 별 prompt 절반으로 λ tune → 나머지 절반으로 평가 (leave-one-scene-out 또는 80/20 split)
- **Predict**: D2.real 의 일부가 negative contrast 로 회복 (특히 distractor inflation subtype). λ=0.3–0.5 부근이 sweet spot 예상 (contrast 효과 vs noise amplification trade-off)
- **Cost**: 반나절 (inference 시 score 만 바꿈)

---

### D. Inference — matching 단계 (4 개)

#### D1. Hierarchical level mismatch

- **직관**: query 가 원하는 object/part level 과 선택 level 이 맞는가?
- **측정**: 67 prompt × {L1, L2, L3, L1+L2, L2+L3, L1+L2+L3} oracle. Prompt 를 **object / part / material / attribute** 로 분류 (2 annotator + Cohen's κ ≥ 0.7 의 agreement) 해서 level 적합성 확인
- **Predict**: 작은 객체 (egg yolk) 는 L1, 큰 객체 (table) 는 L3 가 optimal — granularity 가 prompt-dependent
- **Cost**: 반나절

#### D2. Score calibration

- **직관**: cosine 폭이 좁아 ranking 이 noise 에 흔들리는가?
- **측정**: cosine 분포 [min, max, std], score margin (top-1 − top-2), rank stability across frames
- **Predict**: A2 (rank 2–10 prompts) 가 calibration 으로 일부 회복; temperature scaling / isotonic 으로 마진 확장
- **Cost**: 반나절 (0 추가 학습)

#### D3. Top-k cardinality — **independent decision policy axis** ★

- **직관**: 정답도 골랐지만 k 때문에 잡것을 같이 끌고 오는가?
- **측정**: k ∈ {1, 2, 3, 5, ROFA-adaptive} sweep × score-gap threshold × area-penalty. Per-prompt optimal k
- **분류 logic**:
  - oracle rank ≤ 3 이지만 mIoU 낮음 → D3.over-union (정답+잡것)
  - top-k 줄여도 정답 못 잡으면 → D2 axis 의 문제
- **Predict**: D3 가 D2 와 독립적인 mIoU 손실 source — "정답 retrieval 은 맞아도 selection policy 가 망친" 비율 정량
- **Cost**: 반나절 (test_lerf 의 k 변수만 sweep)

#### D4. Visibility gating — **attribute as mitigation** ★

- **직관**: 보이지 않는 wrong SP 를 visibility 조건으로 줄일 수 있는가?
- **측정**: inference 시 "current-view visible SP 만 ranking" 또는 "visibility weight 로 score modulation"
- **Predict**:
  - invisible D2 → 줄어듦 (mitigation)
  - visible D2 → 그대로
  - D3 over-union → 일부 변화
- **의미**: visibility 가 root cause 가 아니라 **manifestation-level mitigation** 임을 실험적으로 증명. Method 로 살릴지 diagnostic 으로 둘지 결정
- **Cost**: 반나절

---

### E. Novel Mechanism Insights — mechanism framing 의 contribution (4 개)

#### E1. Phantom direction bias — **paper 의 가장 큰 risk/reward** ★★ **6차 보강**

- **직관**: phantom feature 가 random noise 인가, 특정 semantic 방향으로 치우치는가?
- **측정**:
  - **Vocabulary 선택**: **scene-disjoint LVIS 1203 category** (또는 COCO 80) — 67 LERF prompt 와 분리. *순환 logic 차단의 핵심*
  - **★ 정정 (6.1) — "phantom feature" 의 정의**: "direction bias" 라는 단어와 일치하는 자연 정의는 *틀어진 방향 벡터* 이다. 세 가지 candidate:
    - (a) **Aggregated SP feature** 자체 → "결과 위치" (덜 informative — 절대 위치는 query 마다 다른 baseline)
    - (b) **View-mixed feature** (post-B8, pre-ROFA) → "원인 위치" — phantom genesis 의 source 진단용
    - (c) **Direction vector** = `f_aggregated − f_target_text` (L2-normalize 후) → **"틀어진 방향"** ← **E1 의 default**
    - → P1–P4 의 pattern test 는 (c) direction vector 위에서. (a)/(b) 는 robustness check 로 병행.
  - 각 D2.phantom SP 의 **direction vector (c)** → LVIS text embedding 과 cosine top-K → nearest category 분포 수집 (**P1/P2/P4 에만 적용**)
  - **★ P3 별도 metric (6.2)**: P1/P2/P4 는 **category-level** bias (small → background **category** 등) — LVIS top-K 가 자연. **P3 는 *instance-level* bias** (figurine A → figurine B, 같은 LVIS bin "figure/toy" 안) → LVIS category 분포로는 P3 가 안 잡힘. P3 는 *spatial-proximity 기반* 별도 metric:
    - per-pair (figurine_i, figurine_j) 의 **3D centroid Euclidean distance** × **confusion frequency** (phantom direction 이 j 와 cosine top-1 인 횟수)
    - **"phantom 이 가장 가까운 다른 figurine 으로 가는 확률"** vs **random-pairing baseline**
    - permutation null = figurine instance 간 random pairing (N=1000 회 셔플)
    - alternative: rank correlation (Spearman) between 3D distance rank 와 confusion rank
  - **Pattern grouping**: 위 4 P1–P4 pattern 별로 분포 집계 (P1/P2/P4 는 **LVIS category 분포**, **P3 는 spatial-proximity 통계** — 분리)
  - **Robustness check**: (a)/(b)/(c) 셋 다 측정 후 P1–P4 결론의 일관성. 셋이 다른 결론 → 그 자체가 finding (어떤 stage 가 direction bias 의 진짜 source 인지)
- **Null distribution (NEW)**:
  - CLIP feature 공간은 anisotropic (modality gap, frequency bias) 이라 *uniform* 이 null 이 아님
  - → **permutation test**: phantom SP 의 label 을 N=1000 회 셔플 → null 분포 추출
  - Observed pattern 의 chi-square / KL-divergence 가 null 의 99%-tile 위면 significant
- **★ Specific testable predictions** (paper 가 발견할 수 있는 pattern):
  1. **Small object phantom** → background/table direction bias
  2. **Food object phantom** → bowl/plate direction bias
  3. **Character/figure phantom** → nearby figurine direction bias
  4. **Transparent object phantom** → background/empty direction bias
- **Decision rule (사전 약속)**:
  - 위 4 pattern 중 2+ 가 permutation-test significant (p < 0.05) AND effect size > 0.3 → **main contribution: direction-aware aggregation**
  - 1 pattern significant → moderate contribution, paper section 으로
  - 모두 null 내 → **negative finding, appendix only**, paper 방향을 E2/B1 으로 전환
- **검정력 한계 (6차 추가, section 6.6 참조)**: 67 prompt 를 4 pattern 분할 시 pattern 당 ~17 prompt → effect size 0.3 detect 어려움. **3DOVS 까지 합산해 prompt 수 늘리거나 effect size threshold 를 0.5 로 상향** 권장
- **Cost**: 2–3 일 (A3 결과 후)

#### E2. Hierarchical phantom propagation

- **직관**: L1 phantom 이 L2/L3 까지 전파되나, 평균으로 씻기나, 증폭되나?
- **측정**: L1 SP feature 의 phantom signature → L2 (그 L1 들의 parent) 의 feature drift, L3 까지 추적
- **★ 3 가지 가능한 결과**:
  - **Propagation**: L1 오염 → L2/L3 도 오염 → hierarchy 가 도움 안 됨
  - **Wash-out**: 상위 merge 가 phantom 희석 → hierarchy 자체가 implicit fix
  - **Amplification**: 오염된 child 가 parent score 지배 → hierarchy 가 phantom 을 *증폭*
- **Predict**: 가장 임팩트는 amplification — THGS/ReLaGS hierarchy 자체의 한계 증명
- **Cost**: 1 주 (A3 후)

#### E3. Frame-conditional phantom emergence

- **직관**: 같은 SP 가 prompt A 에선 phantom, prompt B 에선 real 일 수 있나?
- **측정**: (SP, query) pair 별 mechanism — phantom 이 SP 의 absolute 속성인지 conditional 인지
- **Predict**: phantom 은 SP × query 의 함수 → "bad SP 제거" 만으론 부족. **query-conditioned reliability** 필요
- **Cost**: A3 결과 후 자연스럽게 따라옴

#### E4. Counterfactual phantom injection ★

- **직관**: 관찰이 아닌 *개입* 으로 mechanism 검증. 깨끗한 SP feature 에 인위적 phantom 을 주입하면 D2.phantom 의 failure pattern 이 재현되는가?
- **측정**:
  1. Type C (성공) prompt 의 oracle SP 선택
  2. 그 SP 의 view-별 feature 중 일부를 random 객체 view 로 교체 (artificial contamination)
  3. Re-aggregation → ranking 재계산 → failure 발생 여부 + pattern 측정
- **Predict**: artificial phantom 이 자연 D2.phantom 의 pixel signature 와 rank pattern 을 재현 → **mechanism 의 causal evidence**
- **의미**: E1–E3 가 *관찰* 이라면 E4 는 *intervention*. 인과 증명의 표준 도구
- **Cost**: A3 instrument 후 1 주

---

### F. Cross-method Validation — ReLaGS 에 framework 적용 (2 개)

#### F1. Method transition matrix — **carefully framed + full 5×5** ★ **6차 보강**

- **직관**: ReLaGS 가 THGS 의 어느 failure 를 어떻게 reclassify 했는가? (양방향)
- **측정**: 67 prompt × **full 5×5 transition matrix**:
  ```
                     ReLaGS class
                     ┌─────────┬─────┬──────────┬───────────┬─────┐
                     │ Success │ D1  │ D2.real  │ D2.phantom│ D3  │
  THGS ─────────────┼─────────┼─────┼──────────┼───────────┼─────┤
       Success      │   (a)   │ (b) │   (c)    │    (d)    │ (e) │
       D1           │   (f)   │ (g) │   (h)    │    (i)    │ (j) │
       D2.real      │   ...   │ ... │   ...    │    ...    │ ... │
       D2.phantom   │         │     │          │           │     │
       D3           │         │     │          │           │     │
                     └─────────┴─────┴──────────┴───────────┴─────┘
  ```
  - **Forward (THGS → ReLaGS)** 만 보면: ReLaGS 가 무엇을 *고쳤나* 만 보임 (한 면)
  - **Reverse (THGS Success → ReLaGS 실패)** 도 같이 봐야 ROFA over-smoothing 이나 새 aggregation 이 만든 regression 을 잡음 → **net effect 의 valid 근거**
- **표현 주의** (공격적 ❌ → 정확한 ✅):
  > "ReLaGS substantially reduces degenerate retrieval. Our mechanism analysis reveals whether the recovered cases become **true successes or transition into semantic distractor failures, and whether previously successful cases are preserved**."
- **Predict**:
  - 대각선 우측 (D1 → Success) 비율이 majority 이면 net fix
  - D1 → D2.phantom 이 상당 비율이면 partial reclassification
  - **Success → D2.phantom 이 0 이 아니면 ROFA regression** — ReLaGS 한계의 정량 증거
- **Cost**: H2 + A3 + ReLaGS 측 instrument 후 1 주

#### F2. ROFA mean-limit — **bimodal-balanced phantom** ★ **6차 구체화**

- **직관**: ROFA 의 outlier filter 가 못 잡는 phantom 의 정량 정의
- **Code 분석 (왜 ROFA 가 못 잡나)** [merge_proj.py:117-124](ReLaGS/merge_proj.py#L117-L124):
  - ROFA 는 `mean_sim` (각 view 의 *스칼라* 평균 코사인) 분포 위에서 filter: `keep_mask = mean_sim > μ − τσ`
  - 5:5 balanced bimodal 이면 각 view 의 mean_sim 이 비슷 → σ 작음 → keep_mask 전원 통과 → mean = midpoint
  - **★ 정정 (6.1) — GMM 적용 공간**: ROFA blind phantom 을 잡으려면 **feature vector space** (단위구 위의 N×D matrix) 의 cluster 구조를 봐야 한다. mean_sim (스칼라 N-vector) 위의 unimodality 와 feature space 의 multimodality 는 *서로 다른 분포* — 6차의 "GMM(k=2) BIC < GMM(k=1) BIC" 표현이 어디서 fit 하는지 모호했음. **명시: 모든 GMM 은 per-view L2-normalized feature vector (N×D) 위에서 fit, *not* on mean_sim scalar (N)**
- **GMM 안정성 가드 (6.1 추가)**:
  - View N < 10 인 SP 는 GMM(k=2) fit 불안정 (covariance 추정 noisy) → **"insufficient evidence" box 로 별도 분류**
  - View N ≥ 10 인 SP 만 3 subtype 분류 진행 → fraction 보고 시 분모 (분류 가능 SP 수) 명시
- **3 가지 phantom subtype (정량 정의 — 6.1 패치)**:

  | Subtype | 정의 | 판정 statistic |
  |---|---|---|
  | **Outlier phantom** | 한 두 view 가 튀어서 평균 오염 | feature-space `GMM(k=1) BIC` 최소 (or k=2 BIC 와 유의 차이 없음) AND ∃ view with `mean_sim < μ − τσ` |
  | **Bimodal-balanced phantom** | 두 cluster 가 비등하게 갈려 ROFA 통과 | **feature-space** `GMM(k=2) BIC < GMM(k=1) BIC` AND **minority cluster ≥ 30%** AND mean_sim 분포의 σ < τ × overall-σ (ROFA blindness condition) |
  | **Mean-dilution phantom** | 모든 view 가 비슷한 mixed feature (B8 결과) | feature-space `GMM(k=1) BIC` 최소 AND cluster center 가 wrong direction (text query 후보 중 어느 것과도 cos 차이 < 0.05) |

- **Predict**:
  - ReLaGS D2.phantom 의 **bimodal-balanced + mean-dilution 비율 > 50%** → ROFA mechanism 적 한계 증거
  - Outlier phantom < 30% → ROFA 이미 잘 처리한 부분
- **Aggregation 대안 별 phantom 처리 비교**:
  - mean (ROFA): outlier 만 잡음
  - mode-cluster center: bimodal-balanced 해결
  - query-conditioned top-view: target dilution + bimodal 둘 다 해결
  - B8 의 hard-assignment + 위: mean-dilution 까지 해결
- **Cost**: F1 와 함께. **새 aggregation 방법의 motivation 직접 제공**
- **★ 구현/검증 현황 (6.3, 2026-06-10)**: Stage 2B 의 실제 구현 ([scripts/stage2b_f2_subtypes.py](../../scripts/stage2b_f2_subtypes.py)) 은 위 feature-space GMM 검정을 **구현하지 않음** — cos-with-prompt 스칼라 휴리스틱 (IQR/std threshold) 으로 대체됨 (6.1 의 명세는 여전히 미이행 과제). simulation τ=1.0 vs 실제 pipeline default **τ=2.0** 불일치도 검증으로 정정: subtype 분포는 τ-robust (strong 11 / mean_dil 4 / bimodal 2 / outlier 0), keep-mask pathology 는 **정직 65% + no-drop 12% / 실수 4건 (old camera, ottolenghi, pikachu, onion segments)**. 실수 4건 중 2건이 instance-confusion phantom 인 점이 신규 단서. **outlier_handled 0% 는 τ=2.0 에서도 유지 → "ROFA 가 잡을 outlier 가 애초에 없다" 는 본 가설의 핵심 predict 는 강화됨**. 상세: [experiments/experiment_results.md](experiments/experiment_results.md) 의 🔁 검증 section.

---

## 3. E1 specific predictions — paper 의 핵심 risk

E1 의 발견 여부가 paper narrative 를 좌우. **사전 명시** 가 적절한 과학:

### 발견하면 main contribution 인 4 pattern

> **검증 방법 dispatch (6.2)**: P1/P2/P4 = **LVIS category-level** (chi-square + permutation null). P3 = **spatial-proximity instance-level** (별도 metric — 아래 row 참조).

| Pattern | 예측 | 검증 방법 |
|---|---|---|
| **P1. Small-object → background** | 작은 객체의 phantom direction 이 table/background category 로 bias | LVIS category 분포: object 면적 vs phantom direction 의 correlation, permutation chi-square |
| **P2. Food → vessel** | 음식 객체의 phantom 이 bowl/plate/container 로 bias | LVIS food vs vessel category 분리 (ramen scene 특화), permutation chi-square |
| **P3. Character → nearby figure** ★ | figurines scene 의 phantom 이 인접 figurine 으로 bias (**instance-level — LVIS 로 안 잡힘**) | **별도 metric** (6.2): 3D centroid distance × confusion frequency, "nearest other figurine" 확률 vs random-pairing permutation null (N=1000), Spearman (distance rank, confusion rank) |
| **P4. Transparent → background** | "glass cup" 등의 phantom 이 배경 category 로 bias | LVIS material/transparency category 분류 (2 annotator 합의), permutation chi-square |

### Decision rule (사전 약속)

- **2+ pattern significant** (permutation p < 0.05, effect size > 0.3) → **main contribution + direction-aware aggregation 제안**
- **1 pattern significant** → moderate contribution, paper section 으로
- **0 pattern significant** (random direction) → **negative finding** 으로 appendix, paper 방향을 E2 (hierarchical propagation) 나 B1 (aggregation ablation) 로 전환

이렇게 사전에 정의하면 cherry-picking 위험 없음. 검토자에게도 신뢰성 ↑.

**검정력 보강 (6차)**: 67 prompt × 4 pattern 분할 시 pattern 당 17 prompt → effect size 0.3 underpowered. 보강책:
1. LERF + 3DOVS 합산 (총 prompt 수 ↑)
2. Effect size threshold 0.5 상향 (보수적 판정)
3. Pattern aggregation: P1 + P4 (둘 다 background bias) 합쳐서 검정

---

## 4. 수정된 Prioritization (Tier 0/1/2/3) — 6차 재구성

### Tier 0 — Classifier 정의 (모든 분석의 prerequisite)

| 순위 | 가설 | Cost | 이유 |
|---|---|---|---|
| **1** | **B7** Oracle SP purity/completeness/fragmentation | 반나절 | A3/A4 의 prerequisite (impure oracle 위에서 rank 해석 불가) — **6차 격상** |
| **2** | **A4** Oracle-rank margin | 1 일 | 모든 failure 난이도 해석의 기준, 0 추가 학습 |
| **3** | **A2** Image-CLIP ceiling (4 crop) | 4 h | encoder-limit 분리, method-matched crop 포함 |
| **4** | **A3** Cross-view consistency (dual-side + subtype mapping) | A1 후 즉시 | real vs phantom 핵심 분리 |
| **5** | **A1** Per-view variance | H2 instrument (1 주) | risk signal, classifier 의 보조 |

→ B7 + A4 + A2 먼저 (1 일+반나절+4h), 그 후 A1 (1 주) → A3 (자동).

### Tier 1 — 빠른 paper-worthy 결과 (1 일–반나절)

| 순위 | 가설 | Cost | 이유 |
|---|---|---|---|
| **1** | **D3** top-k cardinality sweep | 반나절 | over-union 의 mechanism, 기존 D4 직접 설명 |
| **2** | **C2** prompt expansion (generic) | 반나절 | text-side under-discrimination |
| **3** | **D1** hierarchy mismatch | 반나절 | granularity 진단 |
| **4** | **D2** calibration | 반나절 | A2/D3 prompts 회복 |
| **5** | **D4** visibility gating | 반나절 | mitigation 검증 |
| **6** | **C3** negative prompt contrast (λ sweep) | 반나절 | distractor inflation subtype 회복 후보 |

→ Tier 1 만으로도 paper Section 1–2 (taxonomy + 진단) 완성 가능.

### Tier 2 — phantom genesis 정량화 (1 주)

| 순위 | 가설 | Cost | 단계 |
|---|---|---|---|
| **1** | **B8** Within-view ratio mixing | 반나절 | within-view — **6차 신설** |
| **2** | **B1** mean dilution + aggregation ablation | 1 주 | across-view |
| **3** | **B6** view-count sparsity | 1 일 | partition geometry |
| **4** | **B2** SAM contamination | 반나절 | mask quality |
| **5** | **B3** WEIGHT_THRESHOLD sweep | 1 일 (cache) ~ 1 주 | rendering threshold |
| **6** | **B4** zero_scale | 1 일 | background bleed |
| **7** | **B5** crop size | 3 일 | CLIP encoding |

### Tier 3 — paper contribution 후보 (mechanism framing 만이 가능하게 한 것)

| 순위 | 가설 | Cost | 가치 |
|---|---|---|---|
| **1** | **E1** phantom direction bias (LVIS vocab + permutation null) | 2–3 일 | high-risk/high-reward, paper 의 가장 큰 novelty |
| **2** | **F2** ROFA mean-limit (3 subtype 분리) | F1 와 함께 | ReLaGS 한계의 mechanism 적 이유 |
| **3** | **E2** hierarchical propagation | 1 주 | THGS/ReLaGS hierarchy 본질 한계 |
| **4** | **E4** counterfactual injection | 1 주 | mechanism causal evidence |
| **5** | **E3** frame-conditional emergence | A3 후 자동 | taxonomy 정교화 |
| **6** | **F1** Method transition matrix (full 5×5) | 1 주 | ReLaGS improvement 의 정확한 성격 |
| **7** | **C1** CLIP OOD | A2 와 함께 | encoder limit boundary |

---

## 5. 최종 Narrative — Phantom 해부 + 방법론적 엄밀성

### Paper 의 한 단락 요약

> *"우리는 open-vocabulary 3DGS segmentation 의 실패를 **mechanism-based taxonomy** 로 처음 정량 해부한다. (1) **Oracle SP purity (B7) + Cross-view consistency (A3) + rank-margin (A4)** 로 D2 semantic distractor 를 D2.real, D2.phantom-target-dilution, D2.phantom-distractor-inflation 세 subtype 으로 분리한다. (2) **Within-view mixing (B8)** 과 **across-view ROFA mean-limit (F2)** 의 2-stage phantom genesis 모델을 제시한다. (3) Phantom 의 **systematic direction bias (E1)** 4 가지 pattern 을 사전 정의하고 scene-disjoint LVIS vocabulary + permutation null 로 검증한다. (4) ReLaGS 의 D1 86% 감소를 **full 5×5 transition matrix (F1)** 로 분해해, 그 중 N% 가 D2.phantom 으로의 reclassification 임을 보인다. (5) ROFA 의 mean-limit 를 **outlier / bimodal-balanced / mean-dilution** 세 phantom subtype 으로 정량화해 새 aggregation 의 필요성을 motivate 한다. (6) **Counterfactual injection (E4)** 으로 mechanism 의 causal evidence 를 제공한다."*

### Paper title 후보

> *"Beyond the Distractor: Mechanism-based Decomposition of Failure in Training-Free 3DGS Open-Vocabulary Segmentation"*

---

## 6. Methodological cautions — paper 의 reviewer 가 짚을 점

### 6.1 Evaluation leakage (C2)
- **Generic expansion** = method (safe)
- **Image-aware expansion** = diagnostic only (leakage 위험), paper 에 명시 필요

### 6.2 Transition matrix framing (F1)
- "fake fix" 같은 공격적 표현 ❌
- "transition between failure classes" 중립 표현 ✅
- **양방향 명시** 필수 (THGS Success → ReLaGS regression 도 포함)

### 6.3 E1 의 사전 명시 + vocabulary independence
- 발견 가능한 pattern + decision rule 을 *사전* 에 정의 → cherry-picking 위험 차단
- Vocabulary 는 **scene-disjoint LVIS** 사용 (LERF prompt 와 분리) → 순환 logic 차단
- Null distribution 은 **permutation test** (theory uniform 이 아님)

### 6.4 A1 vs A3 의 역할 분리
- A1 = risk signal (high variance → phantom-prone 후보)
- A3 = classifier (실제 판정)
- A1 alone 으로 phantom 결론 금지

### 6.5 A2 의 한계 + crop-policy 일치
- "GT crop 으로 CLIP 도 못 풀면 encoder limit" 까지 안전
- 단, method 와 다른 crop policy 로 측정한 ceiling 은 method 비교의 valid baseline 이 아님 → **(iv) method-matched crop 필수**
- "GT crop 으로 CLIP 이 풀면 우리 method 가 풀어야 한다" 는 다른 (더 강한) 가정 필요

### 6.6 통계 검정력 (NEW)
- 67 prompt 를 4 pattern (small / food / character / transparent) 으로 분할 시 pattern 당 ~17 prompt
- effect size 0.3 + p < 0.05 detect 에는 unpowered (Cohen's d=0.3, two-sample, α=0.05, power=0.8 → 약 n=175 필요)
- **대응**:
  - LERF + 3DOVS prompt 합산 → 약 150+ prompt
  - Effect size threshold 0.5 상향 (보수적 detection)
  - Pattern aggregation (e.g., P1+P4 → "background bias")
  - 만약 통계 power 못 채우면 → "preliminary findings" 로 framing, full validation 을 future work 로 명시

### 6.7 Oracle SP definition stability (NEW)
- A3/A4 가 의존하는 "oracle SP" 가 B7 의 purity/fragmentation 에 따라 ambiguous
- 권장: oracle SP 후보가 multiple 일 때 (best purity & completeness 의 trade-off) 모두에 대해 A3 계산 → robustness check

---

## 7. 다음 step — 최소 1 일짜리 sprint

> **★ 6.3 갱신 (2026-06-10)**: 아래 Day 1 / Week 1 sprint 는 완료 (Stage 1–2B, [experiments/](experiments/) 참조 — B7/A4/A2/D3/C3/B8-proxy/F2 수행, 구현 검증 패치까지).
>
> **★ Stage 3.1 완료 (2026-06-10)**: B8 인과 replay 수행 → **B8 기각** (B8 section 의 ★★ 결과 참조). hard-assignment fix 는 실행 전 기각. **현재 next step = Stage 3.2 — B1 anatomy**: replay dump ([stage3_b8_replay_perview.pkl](../../output/diagnostics/stage3_b8_replay_perview.pkl)) 재사용으로 GPU 재작업 없이 (A) 누적 궤적 추적 — view 를 하나씩 더하며 신호가 *언제* 무너지는지, (C) pool 경쟁 분해 — 자기 약화 vs 경쟁 SP (wrong top-1) 상대 강화. B1 의 aggregation ablation (uniform / top-k-cos / mode-cluster) 이 그 뒤를 따름.

**Day 1 sprint** (모든 후속 작업의 baseline):

1. **B7 (반나절)** — oracle SP purity/completeness/fragmentation → D2/D3 경계 명확화 + A3/A4 prerequisite 확인 — **6차에서 Day 1 으로 이동**
2. **A4 (4 h)** — rank-margin 통계 → 67 prompt 별 분포 plot (B7 결과로 stratify)
3. **A2 (4 h)** — image-CLIP ceiling 측정 (4 crop 비교) → 13 cross-invariant 분류
4. **D3 (반나절)** — top-k sweep → over-union 의 mechanism 확인

→ 1 일 끝에 paper section 1 (taxonomy + initial findings) 의 draft 가능.

**Week 1 sprint** (B8 + Tier 1 완료):

5. **B8 (반나절)** — within-view mixing rate 분포 + RATIO_THRESHOLD sweep — **6차 신설**
6. **C2 + D1 + D2 + D4 + C3** (각 반나절) — Tier 1 완료
7. **B2 + B6** (각 반나절) — Tier 2 일부

**Week 2–3** (Classifier 완성 + E1 go/no-go):

8. **H2 instrument** + **A1 + A3** — classifier 완성 (B7 result 로 oracle stratification)
9. **E1** — direction bias 측정 (LVIS vocab + permutation null) → main contribution 의 go/no-go

**Week 4+** (Cross-method + causal):

10. **F1 + F2** — full 5×5 transition + 3-subtype ROFA limit
11. **E4** — counterfactual validation
12. **E2** — hierarchical propagation

---

## 8. 관련 문서

- [md/THGS/lerf_ovs_failure_analysis_results.md](../THGS/lerf_ovs_failure_analysis_results.md) — D1–D4 framework 정식 정의 (출발점)
- [md/THGS/lerf_ovs_deep_analysis.md](../THGS/lerf_ovs_deep_analysis.md) — Q1–Q5, A4 의 prerequisite 데이터
- [md/cross_method/lerf_ovs_thgs_vs_relags.md](../cross_method/lerf_ovs_thgs_vs_relags.md) — 13 cross-invariant, F1 의 transition matrix 데이터
- [md/cross_method/paper_deep_analysis.md](../cross_method/paper_deep_analysis.md) — 두 paper 원리 cross-check
- [md/RELAGS/relags_setup_verification.md](../RELAGS/relags_setup_verification.md) — ReLaGS 모델 출처 검증
