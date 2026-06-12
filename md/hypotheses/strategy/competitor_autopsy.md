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

## 4. 결과 (실험 후 기입)

*(P1 실행 후 업데이트 — rank 표, mask 표, R11 판정, plot 링크)*

## 5. Paper related-work 표 초안 (§2.5 용 = 표 A)

| 처방 family | 대표 논문 | 다루는 원인 (taxonomy) | 못 다루는 원인 | 측정 ceiling (부검) |
|---|---|---|---|---|
| Robust 통계 (median/outlier 제거) | VALA, ReLaGS-ROFA | (이론상) outlier view | **coherent-plausible 경쟁자, 소수파 매장** | ROFA: 구출 1/20 (실측) / median: CA-1 |
| View 선택 (bag/top-k) | Beyond Averages | 소수파 매장 (phantom) | **제로섬 — lucky-view jump (easy)** | top-5: +19.9 ↔ −9.8 (실측) |
| 표현/구조 교체 | Polysemy, Segment-then-Splat | dilution 일부 | few-view opportunist, 다의미 외 confusion | 직접 비교 불가 (명시) |
| **진단-설계 (ours)** | 가드된 hybrid | 케이스→신호 1:1 매핑 | easy 가드 미완 (P2) | +3.74pt held-out |

## 6. 업데이트 로그

| 날짜 | 변경 | 한 줄 |
|---|---|---|
| 2026-06-12 | 초기 작성 | 카탈로그 C1–C7, 부검 가능성 분류, CA-1/CA-2 설계, R11 사전등록. CA-2 가 Stage 3.3+5 로 기측정임을 확인 — 신규 본체는 CA-1 (geometric median) |
| 2026-06-12 | **v1.1** | md/hypotheses/ 로 통합. "부검 = method 요구사항 명세" 역할 명시, R11 → 설계 제약 변환 표 추가 |
| 2026-06-12 | **v1.2** | 이 문서 = **P1-A**. P1 이 문제 구체화 phase 전체로 확장됨에 따라 P1-B (부재 쿼리)/C (E1)/D (E2)/E (VALA·StS 코드 직접 실험) 는 [p1_problem_experiments.md](p1_problem_experiments.md) 로 — C1 (VALA)/C5 (StS) 의 "포지셔닝만" 한계는 P1-E 가 추후 해소 예정 |
