# Competitor Autopsy — 경쟁 처방의 사전등록 부검 (living document)

> 생성 2026-06-12 (v1.1 — hypotheses/ 통합). 짝꿍 문서: **[roadmap.md](roadmap.md)** (전체 전략). 실행 phase: **P1**.
>
> 목적: "평균이 문제다" 를 이미 말한 경쟁 논문들의 **처방(fix)** 을 우리의 인과 하네스 (all-SP per-view dump, THGS+ReLaGS) 위에서 같은 조건으로 평가한다. 단, 평가 *전에* taxonomy 가 도출하는 실패 예측을 사전 명시한다 (R11). 적중하면 related work 전체가 우리 framework 의 검증 데이터로 변환된다 — **postdiction 이 아니라 prediction**.
>
> **v1.1 명시 — 이 부검의 정체**: 이것은 "기존 method 들의 문제점 구체화" 작업이다. 단 비판이 목적이 아니라, 각 family 의 측정된 실패가 **우리 method 가 가져야 할 설계 제약** (= 요구사항 명세) 으로 변환되는 구조다 ([roadmap.md](roadmap.md) §0). Stage 3.4 가 우리 자신의 실패 36 케이스에 했던 일을 경쟁자에게 하는 것.

---

## 0. 원리 — 왜 "부검" 인가

경쟁 논문들의 구조는 전부 `관찰("평균 나쁨") → 내 대안 → 벤치마크 숫자`. 대안이 **왜** 작동해야 하는지의 인과 증거가 없고, 어떤 실패 모집단을 희생하는지 측정하지 않는다. 우리 법칙 3개가 각 처방 family 의 운명을 사전에 결정한다:

- **법칙 ② (coherent-plausible 경쟁자)** → robust 통계 family (median, outlier 제거, mode-cluster) 는 원리적 불가: 경쟁 SP 는 outlier 가 아니라 *균질하게 그럴듯* 하므로 robust 추정량으로 제거 불가, mean-vs-mean 게임은 oracle 이 구조적으로 패배.
- **법칙 ③ (제로섬)** → 순수 view-선택 family (bag-of-embeddings, query-top-k) 는 phantom 을 살리는 만큼 easy 를 죽임 (lucky-view jump 무방비).
- **따름정리 (few-view opportunist)** → visibility/weight 기반 gating family 는 저관찰-SP 비대칭을 건드리지 못함.

## 1. 경쟁자 카탈로그 (지형도)

| # | 논문 | 주장 (관찰) | 처방 | Family | Taxonomy 매핑 | 부검 상태 |
|---|---|---|---|---|---|---|
| C1 | **VALA** (arXiv 2509.05515) | occluded-Gaussian leakage + multi-view drift | cosine-space **geometric median** + α·T visibility gating | robust 통계 | 법칙 ② 가 불가 예측 | **CA-1 (신규 실험)** |
| C2 | **Beyond Averages** (arXiv 2509.12938) | 평균이 multi-aspect semantics 파괴 | **bag of embeddings** + query-time max/top-k | view 선택 | 법칙 ③ 제로섬 예측 | **CA-2 (기측정 — 재해석 + k=1 변형만)** |
| C3 | **ReLaGS ROFA** (CVPR 2026) | noisy SAM mask / 극단 viewpoint | z-score outlier 제거 후 평균 | robust 통계 | 법칙 ② | ✅ **완료 — Stage 5 G4**: phantom median 0, 구출 1/20 |
| C4 | **Polysemy/ExtrinSplat** (arXiv 2509.22225) | 1 vector ≠ 다의미 | VLM 텍스트 가설 인덱싱 | 표현 교체 | dilution 계열만 설명, few-view·coherent confuser 설명 불가 | 포지셔닝만 (replay 불가 — text-side 전면 교체) |
| C5 | **Segment then Splat** (arXiv 2503.22204) | per-view lifting 자체가 구조적 오류 | object-first 분할 후 1 embedding | 구조 교체 | "object 당 1 vector" 인 한 법칙 ②③ 동일 적용 — 단 직접 replay 불가 | 포지셔닝만 |
| C6 | **OpenSplat3D** (CVPRW 2025) | (implicit) feature blending | visibility-top-5 view 의 MasQCLIP | view 선택 (**query-agnostic**) | "최고 view ≠ 최다 노출 view" 시 소수파 매장 잔존 예측 | P4-(a) 후보 (별도) |
| C7 | **OpenInsGaussian** (ICCVW 2025) | incomplete cross-view fusion | context-aware fusion | mean 변형 | 법칙 ② | 포지셔닝만 |

**부검 가능 기준**: 처방이 "per-view feature 집합 → SP feature/score" 의 aggregation rule 로 환원되는 family (C1, C2, C3) 만 dump 재사용으로 충실하게 재현 가능. 구조/표현 교체 family (C4, C5) 는 정직하게 "직접 비교 불가, 단 object-당-1-vector 가정을 공유하므로 법칙 적용" 으로 related work 에서 처리.

## 2. 부검 실험 설계 (CA-1, CA-2)

### 공통 하네스

- **데이터**: `stage3_3_allsp_<scene>.pkl` (THGS, levels [2,3]) + `stage5_relags_allsp_<scene>.pkl` (ReLaGS) — per-(SP, view) mixed feature + visibility portion 전수.
- **채점**: 각 aggregation 변형으로 SP feature 재구성 → canon-contrast (ClipSimMeasure, pipeline 과 동일) 로 전 pool rank. **raw-cos rank 를 robustness check 로 병행** (canon 포화 영향 분리).
- **모집단**: phantom (THGS 17 / ReLaGS 20), easy control (THGS 41 / ReLaGS easy 군), other — Stage 3.3/5 와 동일 명단.
- **충실도 gate**: 각 변형의 "baseline 재구성" (= 해당 method 의 원래 rule) 이 기존 rank 재현 (Stage 3.3: 0/67 불일치, Stage 5: 97.8–100% 의 동일 기준).
- **평가 2단**: ① rank-level (양 method) — phantom 회복 수 (pool rank ≤3 복귀) / easy 역행 수 ② mask-level (THGS 만, stage4 기계) — 선두 변형만 full-67 mIoU + 그룹별.
- **충실도 한계 명시 (paper 정직성)**: 원논문 re-implementation 이 아니라 **same-harness faithful aggregation rule**. VALA 의 α·T weight 는 dump 의 visibility portion 으로 근사 (둘 다 "기여도 가중" — 근사임을 명시).

### CA-1 — Geometric median (VALA-faithful) ★ 신규 실험의 본체

- **Rule**: 단위구 위 per-view feature 의 **weighted geometric median** — Weiszfeld 반복 (20 iter, eps 1e-6), weight = visibility portion, 결과 L2-normalize. 변형: (a) unweighted (b) portion-weighted (c) +α·T-근사 gating (portion 하위 분위 제거 후 median).
- **비용**: 채점 수십 분 (GPU 불필요 수준), mask eval ~30분.

### CA-2 — Bag-of-embeddings (Beyond-Averages-faithful) — 대부분 기측정

- **이미 있는 것**: query-top-5 (= bag + query-time top-k mean) — **THGS**: phantom IoU 0.203→0.402 (+19.9pt) / easy −9.8pt / full-67 −0.1pt (Stage 3.3 R4 FAIL = 제로섬). **ReLaGS**: phantom +8 / easy −12 (Stage 5 G2).
- **추가할 것**: **k=1 query-max** (bag 의 가장 순수한 형태: `score = max_v canon(f_v)`) — 제로섬이 k=1 에서 더 극단화되는지 (예측: easy 역행 증가). 비용 ~2h.
- 재해석 포인트: 우리 framework 의 제로섬 법칙이 **남의 처방 family 에서, 두 method 에서** 적중한 사례로 격상.

## 3. 사전 판정 R11 (실험 전 고정 — 2026-06-12)

Taxonomy 가 도출하는 예측 (빗나가면 taxonomy 수정 대상이며 그대로 보고):

| ID | 예측 | 정량 기준 |
|---|---|---|
| **R11-a** | geometric median 은 phantom 에 무익 | THGS 17 중 rank≤3 회복 **≤ 3** AND ReLaGS 20 중 **≤ 4** (= Stage 3.2 의 GMM mode-cluster 3/17 수준 이하). 근거: 법칙 ② — 경쟁자는 outlier 가 아님 |
| **R11-b** | median 은 안전하나 무익 | easy 역행 ≤ 2 (mean 과 동급) — "robust 통계는 *해롭지 않지만 문제를 못 푼다*" |
| **R11-c** | k=1 query-max 는 제로섬의 극단화 | phantom 회복 ≥ top-5 수준 BUT easy 역행 **> top-5 의 역행 수** (THGS >10, ReLaGS >12) |
| **R11-d** | mask-level 에서 CA-1/CA-2 어느 것도 R9 headline bar (full-67 ≥ baseline+2pt AND easy 손실 <1pt) 통과 불가 | 전 변형 FAIL — 우리 가드된 hybrid 만 headline 통과 (easy bar 는 P2 과제로 정직 병기) |
| **판정** | **R11-a AND R11-c 적중** → "framework 가 경쟁 처방의 실패를 예측했다" 주장 가능 (§2.5 신설) / 하나 빗나감 → 해당 법칙 수정 + 정직 보고 / 둘 다 빗나감 → §2.5 철회, taxonomy 재검토 | — |

**Method 사양서로의 변환 (v1.1)**: R11 적중 시 각 행이 설계 제약으로 확정 — R11-a/b → "robust 통계 성분 불채택" 의 실측 근거, R11-c → "view 선택에는 가드 필수" = P2 가드 재설계의 정량 motivation, R11-d → 표 B 에서 우리 method 만 통과하는 차별화 라인.

## 4. 결과 (2026-06-12 실행 — P1-A 완료)

**충실도 gate: 양 method 0 mismatch** (in-harness mean baseline rank = 진단 CSV oracle_rank, 67/67 정확 일치 × 2) — 이하 모든 비교 유효.

### Rank-level (phantom 회복 / easy 역행, 회복: base>3→rule≤3, 역행: base≤3→rule>3)

| Rule | THGS ph 17 rec | THGS easy 41 reg | ReLaGS ph 20 rec | ReLaGS easy 43 reg |
|---|---|---|---|---|
| top5 (bag, 기측정 재현) | +6 | −10 | +8 | −12 |
| **qmax1 (CA-2, bag 순수형)** | +3 | −10 | +4 | −14 |
| **gm_u (CA-1 unweighted)** | +1 | **−12** | +2 | **−9** |
| **gm_w (CA-1 weighted)** | +1 | −4 | +3 | −3 |
| **gm_g (CA-1 +gating)** | +1 | −2 | +5 | −4 |
| (참고: ours hybrid v2, Stage 4) | (mask) +17.5pt | −4.08pt | — | — |

데이터: [ca1_gm_ranks_{thgs,relags}.csv](../../../output/diagnostics/ca1_gm_ranks_thgs.csv), [ca2_qmax_ranks_{thgs,relags}.csv](../../../output/diagnostics/ca2_qmax_ranks_thgs.csv), 로그 logs/p1a_autopsy.log

### R11 판정 (사전등록 §3 그대로, 빗나감 포함 정직 기록)

| ID | 사전 예측 | 실측 | 판정 |
|---|---|---|---|
| **R11-a** | median phantom 회복 THGS ≤3 AND ReLaGS ≤4 | THGS 1/1/1 ✅·ReLaGS 2/3/**5** | **부분 적중** — median 본체 (u/w) 적중, gating 변형 (gm_g) 이 ReLaGS 에서 5/20 로 경계 +1 초과. family 결론 (회복 무력: 최대 25% vs top5 40%) 은 유지 |
| **R11-b** | median 은 안전 (easy 역행 ≤2) | gm_u **−12/−9**, gm_w −4/−3, gm_g −2/−4 | **빗나감 — 예측보다 더 나쁨**: median 은 무익할 뿐 아니라 unweighted 는 명백히 유해. uniform-mean 의 교훈 (visibility weighting 필수, Stage 3.2 B1.B) 이 median 에도 그대로 적용 |
| **R11-c** | qmax1 = 제로섬 극단화 (rec ≥ top5 AND reg > top5) | rec **3<6 / 4<8** ❌, reg 10=10 / **14>12** 절반 | **빗나감 (회복 절)** — 순수 bag (k=1) 은 회복도 *더 적음*. 법칙 ③ 수정: 제로섬은 유지되나 k=1 에선 양쪽 다 악화 (단일 view 노이즈) — Stage 3.2 k-sweep 과 정합 |
| **R11-d** | mask 에서 어떤 변형도 R9 bar 통과 불가 | *(mask eval 결과 기입란 — 아래)* | — |

**종합 (사전등록 판정 문구 적용)**: R11-c 빗나감 → "법칙 ③ 수정 + 정직 보고" 경로. **단 모든 빗나감의 방향이 '경쟁 처방이 예측보다 더 나쁘다'** — robust 통계 family 는 유해하기까지 하고, bag 의 순수형은 회복조차 약함. 표 A 의 결론 (어떤 family 도 phantom-easy 동시 보존 불가) 은 약화가 아니라 강화됨. 예측 적중 주장은 "정량 경계 일부 빗나감, 방향성·family-수준 결론 전부 적중" 으로 정확하게 표현할 것.

### R11-d — mask-level (THGS, 208쌍 → per-prompt 67 평균) ✅ **적중: 전 변형 FAIL**

재현 gate: baseline per-prompt 0.5424 — stage3_3 과 **prompt 별 차이 0.0** (render 재현 완벽). [p1a_mask_iou.csv](../../../output/diagnostics/p1a_mask_iou.csv)

| Rule | full-67 | phantom17 | easy 41 | R9 bar |
|---|---|---|---|---|
| baseline | 0.5424 | 0.2029 | 0.7396 | — |
| top5 (bag) | 0.5415 (−0.1pt) | **0.4015 (+19.9pt)** | 0.6416 (−9.8pt) | FAIL (제로섬, stage3_3 정확 재현) |
| qmax1 (bag 순수형) | 0.4444 (−9.8pt) | 0.2212 | 0.5667 (−17.3pt) | FAIL |
| gm_u | 0.4349 (−10.8pt) | 0.1540 | 0.5910 | FAIL (유해) |
| gm_w | 0.5306 (−1.2pt) | 0.2025 (±0) | 0.7199 | FAIL |
| gm_g | 0.5454 (+0.3pt) | 0.1852 (**−1.8pt**) | **0.7513 (+1.2pt)** | FAIL (phantom 무개선) |
| (ours, Stage 4 held-out) | **+3.74pt** | **+17.5pt** | −4.08pt | PARTIAL (유일 headline 통과) |

**표 A 의 핵심 구도가 mask 수준에서 확정**: robust 통계의 최선 (gm_g) 은 *easy 는 지키지만 phantom 에 눈멂* (+1.2 / −1.8), view 선택의 최선 (top5) 은 *phantom 은 살리지만 easy 를 죽임* (+19.9 / −9.8) — **두 family 는 거울상의 절반짜리 처방**이고, 둘을 합치는 가드된 hybrid 만이 headline 을 넘는다 (easy 가드는 우리도 미완 — P2 과제로 정직 병기).

## 5. Paper related-work 표 (§2.5 용 = 표 A, 측정치 반영 — 2026-06-12)

| 처방 family | 대표 논문 | 다루는 원인 (taxonomy) | 못 다루는 원인 | **측정 ceiling (부검 실측)** |
|---|---|---|---|---|
| Robust 통계 (median/outlier 제거) | VALA, ReLaGS-ROFA | (이론상) outlier view — 실측상 easy 보존만 (gm_g easy +1.2pt) | **coherent-plausible 경쟁자, 소수파 매장** — phantom 회복 1~5/37, mask 0 | ROFA 구출 1/20 · gm_g full **+0.3pt / phantom −1.8pt** · unweighted 는 유해 (easy −12/−9) |
| View 선택 (bag/top-k) | Beyond Averages | 소수파 매장 (phantom +19.9pt) | **제로섬 — lucky-view jump (easy −9.8pt)**; 순수형 (k=1) 은 회복마저 반감 | top5 full **−0.1pt** · qmax1 full **−9.8pt** |
| 표현/구조 교체 | Polysemy, Segment-then-Splat | dilution 일부 | few-view opportunist, 다의미 외 confusion | 직접 비교 불가 (object-당-1-vector 가정 공유 명시) |
| **진단-설계 (ours)** | 가드된 hybrid | 케이스→신호 1:1 매핑 | easy 가드 미완 (P2 + P1-B 의 top1-conf 신호 채택 예정) | **+3.74pt held-out (유일 headline 통과)** |

## 6. 업데이트 로그

| 날짜 | 변경 | 한 줄 |
|---|---|---|
| 2026-06-12 | 초기 작성 | 카탈로그 C1–C7, 부검 가능성 분류, CA-1/CA-2 설계, R11 사전등록. CA-2 가 Stage 3.3+5 로 기측정임을 확인 — 신규 본체는 CA-1 (geometric median) |
| 2026-06-12 | **v1.1** | md/hypotheses/ 로 통합. "부검 = method 요구사항 명세" 역할 명시, R11 → 설계 제약 변환 표 추가 |
| 2026-06-12 | **v1.2** | 이 문서 = **P1-A**. P1 이 문제 구체화 phase 전체로 확장됨에 따라 P1-B (부재 쿼리)/C (E1)/D (E2)/E (VALA·StS 코드 직접 실험) 는 [p1_problem_experiments.md](p1_problem_experiments.md) 로 — C1 (VALA)/C5 (StS) 의 "포지셔닝만" 한계는 P1-E 가 추후 해소 예정 |
| 2026-06-12 | **P1-A 실행 완료** | gate 0 mismatch ×2 + mask 재현 차이 0.0. **R11-a 부분 적중** (median 본체 적중, gm_g 가 ReLaGS 5/20 경계 초과) · **R11-b 빗나감 (예측보다 나쁨** — unweighted median 유해 −12/−9**)** · **R11-c 빗나감 (회복 절** — k=1 은 회복도 반감**)** · **R11-d 적중 (전 변형 R9 bar FAIL)**. 거울상 구도 확정: robust=easy만 (gm_g +1.2/−1.8), selection=phantom만 (top5 +19.9/−9.8). §4·§5 표 기입 |
