# P1 문제 구체화 실험군 — 설계 + 사전등록 (living document)

> 생성 2026-06-12. 상위 문서: **[roadmap.md](roadmap.md)** (P1 정의). 짝꿍: **[competitor_autopsy.md](competitor_autopsy.md)** (= P1-A 의 상세).
>
> **P1 의 정체성 (사용자 확정)**: P1 = **기존 paper 들의 문제점 구체화 및 확정**. P2 (method 제작) 는 P1 결과 리뷰 후에만 시작. 이 문서는 P1-B/C/D 의 설계와 사전등록, P1-E (외부 코드 직접 실험) 의 계획을 담는다. 결과는 실험 후 이 문서 + experiments/README stage 표에 기록.

---

## P1 전체 구성

| 파트 | 실험 | 구체화하는 문제 | 대상 | 사전등록 | 상태 |
|---|---|---|---|---|---|
| **P1-A** | 경쟁 처방 부검 (CA-1 median, CA-2 bag k=1) | 경쟁 *처방* 이 왜 실패하는가 | VALA·Beyond Averages·ROFA | R11 ([competitor_autopsy.md](competitor_autopsy.md)) | 설계 완료 |
| **P1-B** | 부재 쿼리 (absence query) | 모두가 상속한 **score 함수 (canon-contrast) 가 '없음' 을 모름** | LERF→전 paper 공통 + 벤치마크 (positives-only) | **R13 (본 문서)** | 설계 완료 |
| **P1-C** | E1 phantom direction bias | phantom 이 random 인가 systematic 인가 — aggregation 손상의 *방향성* | paradigm 공통 | 카탈로그 §3 decision rule (기등록) + 검정력 보강 | 설계 완료 |
| **P1-D** | E2 계층 전파 | THGS/ReLaGS 의 셀링포인트인 **hierarchy 가 phantom 을 전파/증폭하는가** | THGS·ReLaGS | 본 문서 (예측 3분기) | 설계 완료 |
| **P1-E** | 외부 코드 전체 파이프라인 실험 | family-근사가 아닌 **per-paper 실측** — VALA·Segment-then-Splat 실제 코드를 우리 프로토콜로 | VALA, StS (공개 코드) | P1-A~D 리뷰 후 별도 등록 | **future** (아래 §5) |

---

## 1. P1-B — 부재 쿼리 (Absence Query) ★ score 함수 결함의 첫 정량화

### 문제 (왜 이게 "문제점 구체화" 인가)

- LERF-OVS 는 **positives-only** — "scene 에 없는 물체를 물으면?" 을 구조적으로 측정 불가. 어떤 paper 도 이걸 보고한 적 없음 (survey 의 미해결 #1).
- 전 paper (LERF→LangSplat→THGS→ReLaGS→VALA) 가 canon-contrast (`{"object","things","stuff","texture"}` 4 generic 단어와의 min-pairwise-softmax) 를 무비판 상속 — **"있다/없다" 를 분리할 장치가 없음**.
- 우리가 이미 가진 파편들이 전부 이 문제의 증상: canon(0)=0.5 (zero-norm 유령 150 이 0.5점으로 경쟁), easy margin median 0.013 (포화), plastic ladle (IoU 0.84 인데 margin 0.0002).

### 설계

- **부재 어휘**: scene 당 ~30개 — LVIS 1203 에서 추출하되 ① 해당 scene GT 카테고리/prompt 와 어간 중복 제거 ② 실내 장면에 명백히 부재한 카테고리 우선 (zebra, surfboard 류) + 헷갈리는 근접 카테고리 소수 포함 (예: ramen 에 fork). 부재 판정은 GT annotation 기준 + 상위 hit 의 montage 육안 점검 (정직성 노트로 명시).
- **측정**: 각 부재 쿼리에 대해 [rule: visibility-weighted mean (baseline) / query-top-5 / geometric median (CA-1 재사용) / Stage 4 hybrid] × [THGS, ReLaGS dump] 로:
  1. top-1 canon score 와 per-prompt 분포 (present easy 쿼리의 top-1 분포와 중첩도)
  2. **confident-hit rate** = 부재 쿼리 중 top-1 score 가 *present-easy top-1 의 25th percentile* 을 넘는 비율 (즉 "쉬운 진짜 물체만큼 자신 있는 가짜")
  3. 기존 confidence 후보 신호들 (margin, mean-체제 confidence, 체제 간 top-1 일치) 의 present/absent **query-level AUROC**
- 비용: dump + text encoding 만, GPU 거의 0. 렌더 불필요 (rank/score 수준).

### 사전등록 R13 (2026-06-12 고정)

| ID | 예측 | 기준 |
|---|---|---|
| **R13-a** | 전 rule 이 부재 쿼리에 자신 있음 — score 함수는 abstain 불능 | baseline mean rule 의 confident-hit rate ≥ **40%** (양 method) |
| **R13-b** | view-선택 family 는 부재에서 *더* 자신 있어짐 (lucky-view 는 부재 쿼리에도 적용) | query-top-5 의 confident-hit rate > mean 의 것 (양 method) |
| **R13-c** | 기존 신호로는 있/없 분리 불가 — P2 가드가 풀어야 할 문제의 정량 정의 | margin·단일-체제 confidence 의 AUROC < **0.8** (모두) |
| 판정 | a+b 적중 → "score 함수 결함" section 확정 (표 A 의 2층). c 적중 → P2 가드 설계의 요구사항으로 직결. 빗나감 → 해당 신호가 이미 충분하다는 뜻이므로 그대로 P2 에 채택 (어느 쪽이든 이득) | — |

---

## 2. P1-C — E1 phantom direction bias (카탈로그 Tier 3 의 실행)

### 문제

phantom 의 틀어진 feature 가 **random noise 인가, systematic 방향인가** — systematic 이면 aggregation 손상이 예측/보정 가능한 구조라는 뜻 (main contribution 후보), random 이면 negative finding 으로 정직 보고 (appendix).

### 설계 (카탈로그 [extended_failure_hypotheses.md](../extended_failure_hypotheses.md) E1 section 의 6.1/6.2 패치 그대로)

- **Direction vector (default)**: `d = normalize(f_aggregated − f_target_text)` — robustness check 로 (a) aggregated 자체 (b) view-mixed (post-B8) 도 병행.
- **모집단**: THGS phantom 21 + ReLaGS phantom 20 (pooled 주분석 + method 별 병기).
- **Vocabulary**: scene-disjoint **LVIS 1203** text embedding (lvis 패키지/json 에서 카테고리명 로드; 오프라인 실패 시 COCO-80 fallback 으로 명시) → d 와 cosine top-K 카테고리 분포.
- **Pattern test**: P1 small→background / P2 food→vessel / P4 transparent→background 는 LVIS category-level chi-square + **permutation null (label shuffle N=1000)**. P3 character→nearby-figurine 은 **spatial-proximity metric** (3D centroid 거리 × confusion 빈도 vs random-pairing null N=1000, Spearman 병행).
- **Decision rule (카탈로그 §3 기등록 그대로)**: 2+ pattern significant (perm p<0.05, effect>0.3) → main contribution / 1 → moderate / 0 → negative finding appendix. **검정력 보강**: pattern 당 n 부족하므로 ① P1+P4 합산 ("background bias") ② effect threshold 0.5 상향 판정 병기 ③ 미달 시 "preliminary" framing — 전부 카탈로그 §6.6 의 기등록 완화책.

비용: 반나절 (dump + text embedding).

---

## 3. P1-D — E2 계층 전파 (hierarchy 자체의 문제 구체화)

### 문제

THGS/ReLaGS 의 셀링포인트인 **계층 (NAG)** 이 phantom 앞에서 무엇을 하는가 — 전파 (오염 유지) / 세척 (상위 merge 가 희석) / **증폭** (오염된 child 가 parent 점수 지배). 증폭이면 "hierarchy 가 도움" 이라는 두 paper 공통 주장의 반례.

### 설계

- 각 phantom 의 oracle SP 에 대해 NAG 사슬 (L1 조상/자손 포함 L1·L2·L3) 의 per-level feature 로 prompt-cos 와 **per-level pool rank** 측정 (sai_nag 의 nag_feat 직접 사용 — replay 불필요).
- 분류: 사슬에서 rank 가 level ↑ 따라 개선 (세척) / 유지 (전파) / 악화 (증폭). easy control 대비.
- 보조: Stage 3.4 의 granularity 4건 (jake·rubber duck·tesla·sink) 이 "증폭/전파" 의 기존 사례인지 재해석.

### 예측 (사전 — 2026-06-12)

- 주 가설: **전파 ≥ 세척** (계층은 phantom 을 *못 고친다* — 상위도 같은 평균 기계이므로). 증폭 사례가 phantom 의 ≥20% 면 "hierarchy 역효과" section 승격, 세척이 다수면 "hierarchy = implicit fix" 로 반대 방향 정직 보고 (이 경우 parent-union 처방의 근거가 강화되는 부수 이득).

비용: 반나절.

---

## 4. P1 실행 순서와 산출물

```
P1-A (CA-2 k=1 ~2h → CA-1 median 반나절 → mask eval 30m → R11 판정)
P1-B (부재 어휘 구축 → 4 rule × 2 method 채점 → R13 판정)      ← A 와 병렬 가능
P1-C (E1: direction + LVIS + permutation → 카탈로그 rule 판정)
P1-D (E2: NAG 사슬 rank → 3분기 판정)
→ 종합: 표 A 를 2층으로 (A1층 = 경쟁 처방 실패 / A2층 = 공유 score 함수·계층·방향성 결함)
→ 사람 리뷰 → P1-E go/no-go + P2 (method) 진입
```

산출물 명명: `output/diagnostics/ca1_*.csv`, `ca2_*.csv`, `p1b_absence_*.csv`, `p1c_e1_*.csv`, `p1d_e2_*.csv` + plots. 문서 기록: 본 문서 §6 결과 + [../experiments/README.md](../experiments/README.md) stage 표.

## 5. P1-E — 외부 코드 전체 파이프라인 실험 (future, P1-A~D 리뷰 후)

- **무엇**: VALA 와 Segment-then-Splat 의 **공개 코드를 직접 받아** LERF-OVS 4 scene 에서 전체 파이프라인 실행 → ① 원저자 프로토콜 숫자 재현 ② 그들의 출력 위에 **우리 진단 스택 (B7/A4/joint 2×2)** 적용 → per-paper phantom 비율 실측.
- **왜**: P1-A 는 family-수준 faithful rule 부검 — P1-E 는 이를 **per-paper 전체 파이프라인 실측**으로 격상. "VALA 도 ~30% phantom" 이 실측되면 paradigm 주장이 3-4 method 로 확장.
- **비용/리스크**: env 구축 + 학습/최적화 (VALA 는 feature field 최적화, StS 는 per-object 최적화 — GPU 일 단위), sm_120 호환성 리스크 (opensplat3d 셋업 경험 재사용). **P1-A~D 결과 리뷰 후 별도 사전등록으로 진행.**

## 6. 결과 (실험 후 기입)

*(P1-A 는 competitor_autopsy.md §4 에, P1-B/C/D 는 여기에 — rank/score 표, R13 판정, E1 pattern 표, E2 3분기, plot 링크)*

## 7. 업데이트 로그

| 날짜 | 변경 | 한 줄 |
|---|---|---|
| 2026-06-12 | 초기 작성 | P1 을 A–E 로 확장 (사용자 확정: P1 = 문제점 구체화 및 확정, P2 = method 제작). R13 사전등록, E1/E2 설계 고정, P1-E (VALA·StS 코드 직접 실험) future 등록 |
