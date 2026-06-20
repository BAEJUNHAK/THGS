# Strategy Roadmap — top-conference 격차 전략 (living document)

> 생성 2026-06-12 (v1.1 — hypotheses/ 통합 + 목적 위계 명문화). 짝꿍 문서: **[competitor_autopsy.md](competitor_autopsy.md)** (경쟁 처방 부검의 상세 설계/결과).
>
> 이 문서의 역할: Stage 1–5 로 완성된 진단 자산을 **"top-conference 를 통과하는 paper"** 로 변환하기 위한 phase 계획과 사전 판정. 실험 진행에 따라 이 문서와 짝꿍 문서를 계속 업데이트한다 ([../experiments/](../experiments/) 의 stage 문서 규율과 동일).

---

## 0. 목적 위계 — 무엇이 수단이고 무엇이 목적인가

> **최종 목적 = 경쟁자들보다 성능이 좋은 method.**
> **현 단계 = 기존 method 들의 문제점을 *측정 가능한 수준으로* 구체화하는 것 — 단, 그 구체화의 산출물이 곧 method 의 요구사항 명세가 되도록.**

이 위계가 작동하는 구조 (Stage 3.4 가 우리 자신의 실패 36 케이스를 method 부품 목록으로 바꿨던 것과 동일한 패턴을 경쟁자에게 적용):

```
경쟁 처방의 실패를 측정 (P1)  ──→  실패 이유 = 우리 method 가 가져야 할 설계 제약
  median 실패 (법칙 ②)         ──→  robust 통계 방향은 폐기 (이미 반영됨)
  bag/top-k 제로섬 (법칙 ③)     ──→  가드가 유일한 차별화 지점 → P2 의 가치 = 격차 그 자체
  ROFA 무력 (G4)               ──→  outlier 제거 아닌 view 선택 + 가드 (이미 반영됨)
```

→ **P1 은 "비판을 위한 비판" 이 아니라 method 의 사양서 작성이다.** 경쟁자 전원이 못 푸는 지점 (easy 보호 + phantom 회복의 동시 달성) 이 정확히 P2 의 과제이고, P1 이 그걸 정량으로 못박는다.

### 한 문장 전략

> **"'평균이 문제다' 는 이미 등장한 관찰이므로 양보한다 (VALA, Beyond Averages, ReLaGS-ROFA, Segment then Splat 등). 우리는 문제정의를 한 칸 고도화한다: 실패의 핵심은 average 자체가 아니라, *easy query 에는 consensus preservation 이 필요하고 phantom query 에는 query-conditioned minority evidence recovery 가 필요한데 기존 method 들이 이 regime 을 구분하지 못하는 것*이다. P1 은 이 regime-confusion 을 기존 처방 위에서 측정하고, P2 는 그 구분기를 method 로 만든다."**

### Paper 의 승부처 = 두 개의 킬러 테이블

| | 내용 | 만드는 phase |
|---|---|---|
| **표 A (문제점)** | 처방 family × taxonomy 원인 × **측정된 ceiling** × 예측 적중 여부 — "왜 기존 수선들로는 안 되는가" 의 정량 종결 | **P1** |
| **표 B (성능)** | 두 base (THGS, ReLaGS) × {baseline, 경쟁 rule 들, **가드된 hybrid (ours)**} 의 mIoU — 같은 하네스 + **원저자 평가 프로토콜** | **P2 + R10 + P3-③** |

표 A 가 표 B 의 *이유* 를 설명하고, 표 B 가 표 A 의 *결론* 을 입증하는 상호 지지 구조.

---

## 1. 자산 현황 (Stage 1–5 완료, 2026-06-12 기준)

| 자산 | 내용 | 출처 |
|---|---|---|
| **실패 분해** | D2.phantom 31% vs real 7.5% (4.2:1), method-agnostic | Stage 1 |
| **법칙 ①: 소수파 매장** | 좋은 view 18–23% vs easy 50%, best 단일 view 로 88–95% 회복 가능 | Stage 3.2, 5 (G1) |
| **법칙 ②: coherent-plausible 경쟁자** | wrong SP 가 coherence 우위 (p=0.013) + 평균 prompt-cos 도 우위 → **mean-vs-mean 게임은 구조적으로 oracle 패배 → mean-family 수선 전부 원리적 불가** | Stage 3.2 (R3) |
| **법칙 ③: 제로섬** | 단일 투표 체제 (mean 또는 top-k) 는 한쪽 모집단을 반드시 희생 (+19.9 ↔ −9.8) | Stage 3.3 (R4), 5 (G2) |
| **따름정리: few-view opportunist** | visibility-weighted mean 은 저관찰 SP 에 구조적으로 유리 | Stage 3.4 |
| **encoder 신화 해체** | "CLIP 한계 7.5%" 는 과대 — 최종 잔여 3건, miffy 조차 dilution 이었음 | Stage 3.4 (R8) |
| **ROFA 무력 실측** | phantom median 효과 0, 구출 1/20 | Stage 5 (G4) |
| **paradigm-level 재현** | 위 전부가 ReLaGS (자체 partition + ROFA) 에서 거의 같은 숫자 | Stage 5 (G5) |
| **Method** | 가드된 hybrid: LOSO held-out +3.74pt (phantom +17.5) / easy −4.08pt → **R9 PARTIAL** | Stage 4 |
| **벤치마크 감사** | multi-instance 미표기 7 prompt 46건 (R7), mAcc 프로토콜 비호환, zero-norm 유령 150/21 | Stage 3.4, cross_method |
| **인프라** | THGS all-SP dump 1.37GB + ReLaGS all-SP dump 1.2GB (per-view feature 전수) — **모든 aggregation 변형을 GPU 없이 평가 가능** | Stage 3.3, 5 |

남은 구멍 (paper 관점):
1. ~~경쟁 처방 대비 포지셔닝~~ → **P1 (표 A)**
2. easy −4.08pt (method 의 유일한 약점) + **가장 강한 base (ReLaGS) 위에서의 성능 미측정** → **P2 (표 B)**
3. ROFA 유령-청소 귀속 미분리, 원저자 평가 경로 미확인 → **P3**
4. 단일 벤치마크 (LERF-OVS 67 prompts) → **P4 (optional)**

---

## 2. Phase 계획

### P1 — 문제점 구체화 및 확정 ★ 최우선 → 표 A (2층)

> **정체성 (v1.3, 사용자 확정)**: P1 = **기존 paper 들의 문제점을 구체화·확정하는 phase 전체**. P2 (method 제작) 는 P1 결과 리뷰 후에만 시작. 상세 설계/사전등록: **[competitor_autopsy.md](competitor_autopsy.md)** (P1-A) + **[p1_problem_experiments.md](p1_problem_experiments.md)** (P1-B/C/D/E).

| 파트 | 실험 | 구체화하는 문제 | 사전등록 | 비용 |
|---|---|---|---|---|
| **P1-A** | 경쟁 처방 부검 — CA-1 geometric median (VALA-faithful), CA-2 bag k=1 (Beyond-Averages; top-5 는 기측정) | 경쟁 *처방* 의 실패 이유 (법칙 ②③의 예측 적중) | R11 | ~1일 |
| **P1-B** | **부재 쿼리 (absence query)** | 전 paper 공유 **score 함수 (canon-contrast) 가 '없음' 을 모름** + positives-only 벤치마크의 맹점 | R13 | 반나절 |
| **P1-C** | **E1 phantom direction bias** | phantom 손상의 *방향성* — random 인가 systematic 인가 (main contribution 후보 / negative 면 appendix) | 카탈로그 §3 기등록 rule | 반나절 |
| **P1-D** | **E2 계층 전파** | 두 paper 의 셀링포인트 **hierarchy** 가 phantom 을 전파/세척/증폭하는가 | 3분기 예측 (p1 문서) | 반나절 |
| **P1-E** | VALA·Segment-then-Splat **공개 코드 직접 실험** — 전체 파이프라인 + method별 diagnostic adapter | family-근사 → **per-paper 실측** 격상. 단순히 "VALA 도 ~30% phantom?" 이 아니라 **기존 average-fix 가 어떤 regime 은 고치고 어떤 regime 은 놓치는지** 측정 | [p1e_external_code_study_plan.md](p1e_external_code_study_plan.md) 초안 | source audit 시작 |

- 표 A 가 **3층**이 됨: **A1층** = 경쟁 처방의 실패 (P1-A, 추후 P1-E 로 실측 격상) / **A2층** = 모두가 공유하는 구조적 결함 (score 함수 P1-B · 방향성 P1-C · 계층 P1-D) / **A3층** = 문제정의 고도화 (**average problem → regime confusion: consensus 가 필요한 query 와 minority evidence 가 필요한 query 를 구분하지 못함**).
- P1-B 는 P2 가드 설계의 *입력*이기도 함 (R13-c: 기존 confidence 신호의 present/absent AUROC < 0.8 이면 가드가 풀어야 할 문제의 정량 정의).
- 정직성 명시: P1-A 는 per-paper 재현이 아니라 mechanism family 수준 faithful rule (C4/C5 는 포지셔닝 처리) — P1-E 가 이 한계를 解消.

### P2 — Method 본선: 가드 재설계 + R10 (ReLaGS 적용) ★ 성능 우위의 승부처, 수일 → 표 B

> **v1.1 격상**: R10 을 P3 마감 게이트에서 P2 본선으로 이동. 이유 — "경쟁자보다 좋은 성능" 의 산수: 우리 hybrid 는 THGS base (0.5424) 위 +3.74 = 0.580. 외부 비교 대상은 ReLaGS 64.4 (paper), OpenSplat3D 59.7, VALA 58.0 — **THGS base 만으로는 leaderboard 를 못 이긴다. 가장 강한 base (ReLaGS) 위에 우리 scorer 를 얹는 R10 이 헤드라인 숫자의 원천.** 우리 method 는 query-time layer 이므로 "어떤 training-free pipeline 에도 drop-in +Xpt" 가 셀링 포인트이자 일반화 주장.

**P2 = 차별화 실험 라인업** — 부검 (표 A) 이 "경쟁자가 실패한다" 를 보이면, P2 는 "우리만 할 수 있는 것" 을 분리 입증한다. 각 실험이 어떤 경쟁자 대비 무엇을 증명하는지:

| 실험 | 내용 | 증명하는 차별점 | 겨냥 대상 |
|---|---|---|---|
| **P2-a 가드 재설계** | 분리자 후보: ① **체제 간 top-1 일치 (+NAG 친족 보정** — kin_same 가짜 역행 3건 때문에 필수**)** ← 1순위 ② within-prompt pool z-score (canon 이전 raw-cos 공간 — canon 포화 회피) ③ rank-stability. 모집단 THGS+ReLaGS 합산 (easy 84 / phantom 37), **cross-method held-out** (THGS 설계 → ReLaGS 검증). 사전등록 R12 실험 직전 작성 | **제로섬을 깨는 것** — view-선택 family 가 원리적으로 못 하는 easy 보호 (<1pt) 의 달성 | Beyond Averages, 모든 top-k 계열 |
| **P2-b R10 (drop-in)** | stage4_scorer (+새 가드) 를 ReLaGS dump 에 적용 → ReLaGS base 위 mask mIoU. ReLaGS mask render 경로 구축 포함 (+반나절~1일) | **"어느 training-free pipeline 에도 drop-in +Xpt"** — 전부 자기 pipeline 전용인 경쟁자 중 누구도 못 하는 주장 + headline 숫자 (ReLaGS 64.4 위 +α) | leaderboard 전체 |
| **P2-c 가드 이식 (guard transplant)** ★ 신규 | **우리 가드를 경쟁 rule 에 이식** — Beyond-Averages식 query-top-k + 우리 가드 → easy 역행 소멸 여부 (CA-1 median 에도 동일 적용 가능). 같은 하네스에서 rule 교체뿐, 거의 무료 | **기여의 본체 = 진단이 설계한 가드 (transferable)** — "hyperparameter 조합 아니냐" 공격 차단. ablation (끄기) 보다 강한 증명: 남의 method 를 *고쳐주는* 실험 | reviewer 의 novelty 공격 |

- 성공 기준: P2-a+b 에서 R9 bar (full-67 ≥ +2pt AND easy 손실 <1pt) 를 **두 base 모두**에서. P2-c 는 "이식 후 easy 역행 ≥ 절반 감소 AND phantom 회복 유지" 면 transferability 입증 (정확한 수치는 R12 에 사전등록).
- 판정: 전부 PASS → "진단-설계 method 의 성능 우위 + method-agnostic + transferable" 완결 / ReLaGS 측만 미달 → base-종속성 정직 보고.
- 실험 외 차별점 (paper 가 받쳐줄 것): ④ 사전등록-예측 구조 자체 (R1–R12 — 이 문헌 전례 없음) ⑤ 벤치마크 감사 (multi-instance 분리 트랙 — 경쟁자들은 오염된 7 prompt 위에서 숫자 경쟁 중).

### P3 — 마감 게이트 2종 (각 반나절)

| 게이트 | 내용 | 왜 필수 |
|---|---|---|
| **ROFA 귀속 확인** | G4 스크립트의 ROFA-off 재구성에서 **유령 수** 카운트 — 유령 150→21 감소가 ROFA 공인지 WEIGHT_THRESHOLD (0.01→0.0001) 공인지 분리 (B3 confound, 미실행) | "ROFA = 유령 청소부" 문장의 정확성. ROFA-off 에서도 유령 21 이면 ROFA 고유 기여 ≈ 0 으로 오히려 강화 |
| **원저자 평가 교차확인** | 최종 숫자를 test_lerf.py + eval_seg.py 프로토콜로 재확인 | **표 B 의 전제** — 외부 비교 (ReLaGS 64.4 등) 는 원저자 프로토콜 숫자라야 같은 표에 놓을 수 있음. reviewer 1순위 공격 지점 |

### P4 — 3번째 증거 지점 (optional, venue 일정에 따라)

두 후보 중 택1 (P1–P3 완료 후 결정):

- **(a) OpenSplat3D 자연 실험**: env 셋업 완료. visibility-top-5 는 **query-top-k 가 아니므로** "최고 view ≠ 최다 노출 view 일 때 소수파 매장 잔존" 이라는 falsifiable prediction 자동 도출. 비용 ~2-3일.
- **(b) 3DOVS 확장**: E1 검정력 문제도 동시 해결. 비용: language_features 신규 생성 (GPU 수 시간/scene) + 진단 스택 이식.

---

## 3. Paper section 매핑 (phase → 기여)

| Paper section | 내용 | 지지 phase/stage |
|---|---|---|
| §1 Taxonomy | phantom 31% / 4.2:1 / method-agnostic 분포 | Stage 1 (완) |
| §2 Mechanism — 법칙 3개 + paradigm 재현 | 소수파 매장, coherent 경쟁자, 제로섬, ROFA 무력 | Stage 3.x, 5 (완) |
| **§2.5 Why existing fixes cannot work (표 A)** ★ 신설 | 경쟁 처방 부검 + 예측 적중 | **P1** |
| **§3 Method (표 B)** | 가드된 hybrid, 두 base, cross-method held-out, 원저자 프로토콜 | Stage 4 (완) + **P2, P3-②** |
| §4 Benchmark 감사 | multi-instance, mAcc 비호환, 유령 | Stage 3.4 (완) + P3-① |
| Appendix | 사전등록 판정 R1–R12 전체 기록, 적대적 검증 패치 이력 | 전 stage |

## 4. 리스크 / 미정

- **P1 예측이 빗나가는 경우**: taxonomy 수정 대상이지만 사전등록 덕분에 정직 보고 가능 — Stage 3.2 의 mean-family 전멸 데이터상 확률 낮음.
- **R10 이 ReLaGS 에서 약한 경우**: ReLaGS 의 phantom 20 중 회복 가능 비율이 THGS 와 다를 수 있음 (G2 의 +8 은 가드 없는 상한) — 그 경우 "THGS +3.74 + ReLaGS +α" 로 정직 보고하고 일반화 주장의 강도 조절.
- **단일 벤치마크**: P4 없이 투고 시 일반화 공격 예상 → 2-method 재현 + family-level 부검으로 방어, P4 는 rebuttal 카드.
- **venue/deadline 미정** ← 결정 필요. CVPR 2027 (2026-11 추정) 이면 P4 포함 가능, 더 이른 마감이면 P1–P3 컷.
- ReLaGS mask-level 평가는 ReLaGS render 경로 필요 — P2-b 에서 구축 (G2 는 rank-level 이었음). 이게 P2-b 의 숨은 비용 (+반나절~1일).

## 5. 진행 순서 요약

```
P1 문제 구체화 (A 부검 R11 + B 부재쿼리 R13 + C E1 + D E2, ~2일)
   → 사람 리뷰 (P1-E go/no-go: VALA·StS 코드 직접 실험)
   → P2 method 제작 (가드 R12 + R10 + 이식, 표 B) → P3 게이트 2종 (1일)
                                        ↘ P4 (optional, venue 결정 후)
```

## 6. 업데이트 로그

| 날짜 | 변경 | 한 줄 |
|---|---|---|
| 2026-06-12 | 초기 작성 | Stage 1–5 자산 정리, P1–P4 phase 계획, "한 칸 위" 전략 명문화. P1 비용 1일로 절감 (bag-family 기측정) |
| 2026-06-12 | **v1.1** | md/hypotheses/ 로 통합. **목적 위계 명문화** (최종 목적 = 성능 우위 method, P1 = 문제점 구체화 → method 사양서). **R10 을 P3 → P2 본선으로 격상** (ReLaGS base 위 적용이 헤드라인 숫자의 원천). 두 개의 킬러 테이블 (표 A 문제점 / 표 B 성능) 구조 도입 |
| 2026-06-12 | **v1.2** | **P2 를 차별화 실험 라인업으로 재정의** — P2-a 가드 (제로섬을 깨는 것), P2-b R10 (drop-in), **P2-c 가드 이식 신설** (기여 본체 = transferable 가드의 분리 입증, 거의 무료). 실험 외 차별점 ④⑤ 병기 |
| 2026-06-12 | **v1.3** | **P1 을 "문제점 구체화 및 확정" phase 로 확장 (사용자 확정)** — P1-A 부검 + **P1-B 부재 쿼리 (R13 신설)** + **P1-C E1** + **P1-D E2** + **P1-E VALA·StS 공개 코드 직접 실험 (future)**. 표 A 2층 구조 (A1 경쟁 처방 실패 / A2 공유 구조 결함). P2 = method 제작, P1 리뷰 후 시작으로 명시. 상세: [p1_problem_experiments.md](p1_problem_experiments.md) |
| 2026-06-12 | **P1-A~D 실행 완료** | 표 A 2층 수치 확보. **A1층**: R11-d 적중 (전 변형 R9 bar FAIL) — robust family 는 easy 만 (gm_g +1.2/phantom −1.8), selection family 는 phantom 만 (top5 +19.9/easy −9.8), **거울상의 절반 처방** 확정; 빗나간 예측 (R11-b/c) 은 전부 "예측보다 더 나쁨" 방향. **A2층**: 부재 쿼리 유령 top-1 42%/29% + margin 포화 (0.52) + **미사용 분리 신호 top1-conf AUROC 0.84 (P2 가드 입력)**; E1 = negative (modality-gap); E2 = 계층은 phantom 못 고침 (전파 37%≥세척 32%), 증폭 17% (<20% 미달, tesla 양 method 매장). 사람 결정 대기: P1-E go/no-go, P2 진입 |
| 2026-06-15 | **v1.4 문제정의 고도화** | 한 문장 전략을 **average problem → regime confusion** 으로 업데이트. 표 A 를 3층 구조로 확장: A1 경쟁 처방 실패, A2 공유 구조 결함, A3 consensus regime 과 minority-evidence regime 을 구분하지 못하는 문제정의 고도화. P1-E 를 VALA/StS 의 "고친 regime / 놓친 regime" 실측으로 재정의 |
