# 직관 요약 — 실험 결과를 한눈에 (living document)

> 이 문서는 [stage1_b7_a4_a2.md](stage1_b7_a4_a2.md), [stage2a_phantom_anatomy.md](stage2a_phantom_anatomy.md) 같은 *상세 stage 문서* 의 **직관적 거울** 이다.
>
> 새 stage 가 끝날 때마다 여기에 한 section 씩 추가 → 시간이 지나도 "지금까지 어디까지 왔는지" 한 페이지로 본다.
>
> **마지막 업데이트**: Stage 3.1 완료 시점 (2026-06-10)

---

## 🎯 한 문장 결론 (현재까지)

> **"Stage 5 로 진단이 완전체가 됐다: 소수파 매장과 제로섬이 ReLaGS 에서 거의 같은 숫자로 재현 (95%, 23/51%, +8/−12) — 그리고 ROFA 의 실측 효과는 phantom 에 median 0 (구출 1/20). '다수결의 비극' 은 paradigm-level 이며, robust averaging 조차 못 막는다. method 는 +3.74pt (R9 PARTIAL) — 남은 공학 question 은 easy 를 알아보는 눈."**
>
> (2B "within-view 범인" → 3.1 기각 → 3.2 소수파 매장 → 3.3 제로섬 → 3.4 전수 이름표 → 4 method 본체 합격 → **5 method-agnostic 확정**)

---

## 🧩 비유: "교실에서 강아지 찾기"

선생님이 67장 사진을 교실에 흩뿌리고 "**강아지**!" 라고 외친다. 슈퍼포인트(SP) 들이 각각 "저요!" 손드는데, 정답 SP가 몇 등으로 선택되는지가 게임의 본질.

이 게임이 망가지는 단계는 4가지가 있는데, **각각이 진짜 문제인지** 확인하려고 실험을 단계별로 했다.

---

## 📋 Stage 1 — "범인 찾기 4단계"

상세: [stage1_b7_a4_a2.md](stage1_b7_a4_a2.md)

| 의심 | 실험 | 비유 | 결과 |
|---|---|---|---|
| **분할이 망가졌나?** | B7 | 강아지 SP가 *존재* 하긴 하나? (강아지 + 의자가 합쳐진 SP만 있으면 끝) | ✅ **분할은 멀쩡** (92.5% 깨끗) |
| **검색이 망가졌나?** | A4 | 강아지 SP가 몇 등으로 손드나? | ⚠️ 절반(rank=2)만 잘됨, **19%는 30등 밖** |
| **CLIP이 모르나?** | A2 | 강아지 사진만 *정확히* 잘라줘도 CLIP이 "강아지" 라고 답하나? | **30%는 CLIP도 모름** (encoder ceiling) |
| **그래서 무슨 실패?** | Joint | A4와 A2를 2x2로 교차 | 👇 |

### 🎯 핵심 발견 — 2x2 분류

```
                     CLIP이 알아보나?
                     예          아니오
   SP가 찾았나?  ┌─────────┬──────────┐
   예          │  Easy   │   Rare   │  ← 멀쩡한 케이스
                ├─────────┼──────────┤
   아니오      │ Phantom │   Real   │  ← 실패한 케이스
                └─────────┴──────────┘
```

**67개 prompt 중**:
- 😊 **Easy** (54%): CLIP도 알고 우리도 잘 찾음
- 😱 **Phantom** (31%): **CLIP은 알아봤는데 우리가 죽임** ← 살릴 수 있는 실패!
- 😞 **Real** (7.5%): CLIP도 모르는 진짜 어려운 문제
- 🤔 **Rare** (7.5%): CLIP보다 우리가 더 잘함 (가끔 평균이 도움됨)

**Phantom : Real = 21 : 5 = 4.2 : 1**

> **실패 26개 중 21개(81%)는 살릴 수 있다.** 이건 페이퍼의 정량 contribution.

### 📍 장면별 색깔이 다름

- **ramen**: 6 phantom + **0 real** → "100% 살릴 수 있는 실패만 있는 깨끗한 실험장"
- **teatime**: 79% easy → "이미 잘 됨, 새 방법 시험 의미 적음"
- **waldo_kitchen**: 7 phantom → "여기서 살리면 mIoU 크게 오름"

### 🔄 ReLaGS도 같은 결과

같은 분석을 ReLaGS(경쟁 method)에 했더니 **거의 동일 분포** (29.9% phantom). 즉 phantom 문제는 **THGS 만의 문제가 아니라 paradigm 의 한계**.

ReLaGS 가 phantom 회복 시도해도: **21개 중 4개만 살림 (19%)** → **17개는 두 방법 모두 실패** (= persistent phantoms).

---

## 🔬 Stage 2A — "17명 환자 정밀 부검"

상세: [stage2a_phantom_anatomy.md](stage2a_phantom_anatomy.md)

17 persistent phantoms 를 4개의 forensic layer 로 해부.

### Layer 1 — "조각조각 났나?" (B8 within-view mixing)

비유: 강아지 SP를 한 사진에 그려봤을 때 강아지 모양 한 덩어리인가, 아니면 머리/몸/꼬리 세 조각인가?

→ **phantom과 easy의 차이 없음** (0.16 vs 0.16)
→ "조각남 자체는 범인이 아님"

### Layer 2 — "그럼 누가 1등 됐나?" (Wrong top-1 forensics)

CLIP이 잘못 고른 SP의 정체:

| 정체 | 비율 | 직관 |
|---|---|---|
| 🌫️ **Background drift** | 65% | "강아지 옆 *아무것도 아닌 책상* 을 1등으로 뽑음" |
| 🤝 Over-union | 18% | "강아지 + 의자 합친 SP를 1등" (D3 top-k 문제) |
| 🐱 Instance confusion | 18% | "강아지 대신 *옆 고양이* 를 1등" (pikachu↔jake 같이) |

### Layer 2.5 — "view 별로 보면?"

각 evaluation frame 에서 정답 rank 를 따로 측정:

- **76% structural** = *모든* view 에서 rank > 3 → 어떤 view 도 정답 못 줌 → **mean-dilution** (모든 view 가 일관되게 잘못된 feature)
- **24% target dilution** = 일부 view 는 rank ≤ 3 (정답!) 인데 평균이 죽임 → 단일 view 로 복구 가능

### Layer 3 — "top-k 만 바꾸면?" (D3 sweep)

top-1만? top-10 union? per-prompt 최적 k 를 찾으면:

- mean IoU 0.203 → **0.314 (+11.2pt)** 회복
- 단 **47% (8개) 는 어떤 k 로도 회복 안 됨** = 진짜 structural

### Layer 4 — 시각 montage

17 phantom × 4 panel (GT / Oracle / 잘못된 top-1 / top-10 union) — 페이퍼의 **killer figure**.

---

## 🎯 Stage 2A 가 결정한 것 — phantom 의 정체

17명 환자의 **primary 진단명** 분포:

| 진단 | 개수 | 비율 | 의미 |
|---|---|---|---|
| 🌪️ **Structural ROFA** (mean-dilution) | 6 | 35% | 모든 view 가 잘못된 방향 — 평균이 못 살림 |
| 🎯 D3 deep pool | 3 | 18% | top-k 바꾸면 살아남 |
| 🐱 Instance confusion | 3 | 18% | 다른 인스턴스로 헷갈림 |
| 💧 Target dilution | 2 | 12% | 일부 view 정답, 평균이 죽임 |
| 기타 | 3 | 17% | over_union, background_drift, marginal |

**카테고리로 묶으면**:

- **🌊 Across-view aggregation 실패 (ROFA류)**: **71% (12개)** ← 주범
- **🧭 Direction bias (E1)**: 24% (4개)
- 기타: 5%

---

## 🎯 Stage 2A 완료 시점 — 한 문장 결론

> **"우리 방법이 못 푸는 게 아니라, 우리가 정답을 평균 내서 죽이고 있다."**

---

## 💡 Stage 2A 완료 시점 — 직관 한 페이지 요약

```
┌──────────────────────────────────────────────────────────┐
│  Stage 1 가 답한 것:                                       │
│    실패의 31% 는 CLIP 이 알아봤는데 우리가 죽인 것.       │
│    7.5% 만 진짜 어려운 문제. 비율 4.2:1.                  │
│    ReLaGS 도 같음 = 구조적 한계.                          │
│                                                            │
│  Stage 2A 가 답한 것:                                      │
│    17 명의 끈질긴 환자 중 71% 는                           │
│    "평균 내는 과정에서 죽었다"                              │
│    (ROFA 가 outlier 만 잡고 평균이 잘못된 방향이면 못 잡음) │
│                                                            │
│  다음 (Stage 2B):                                          │
│    - F2: ROFA mean-dilution 의 3 subtype 분류              │
│    - D3: prompt-agnostic top-k sweep                      │
│    - 새 aggregation method 의 motivation 정량화           │
└──────────────────────────────────────────────────────────┘
```

---

## 🎓 Stage 2A 완료 시점 — 페이퍼의 메인 주장

> "우리는 training-free 3D segmentation 의 실패가 **encoder 한계가 아니라 aggregation 의 자해**임을 처음 정량 증명했다. 81% 가 회복 가능한 phantom 이고, 이건 method 와 무관한 paradigm 의 구조적 한계다. 우리는 그 phantom 의 mechanism (mean-dilution) 을 분해하고, 새 aggregation 으로 17개를 살리는 방법을 제시한다."

---

<!-- ====================================================================== -->
<!--  ⬇⬇⬇  새 Stage 끝나면 여기 아래에 section 추가  ⬇⬇⬇                    -->
<!-- ====================================================================== -->

## 🧪 Stage 2B — "ROFA는 무죄, 진짜 범인은 within-view mixing"

상세: [stage2b_rofa_anatomy.md](stage2b_rofa_anatomy.md)

### 비유 업데이트

이제까지: "강아지 SP 가 몇 등 손드는가" 게임.

Stage 2B 가 새로 한 것: **강아지 SP 를 깨끗하게 잘라서** CLIP 에 직접 보여주기. 그랬더니 **17 중 11 (65%) 가 정답**으로 알아본다. 그런데 같은 SP 를 **파이프라인 안에서 추출하면** 죽는다.

→ **재료(강아지 이미지)는 멀쩡한데, 우리 요리 과정이 죽인다.**

요리 과정의 두 단계:
1. **Within-view**: 한 view 안에서 SP 가 여러 SAM mask 에 걸치면 → 비율로 weighted sum (← **여기서 죽음**)
2. **Across-view (ROFA)**: 여러 view 의 mixed feature 를 outlier 거르고 평균 (← **이건 정상 작동**)

### Layer 1 — 신호는 살아있다 (F2.A: H2 lite)

17 phantom 의 oracle SP 를 깨끗하게 crop → CLIP 직접 인코딩 → prompt 와 cos:

| | 비율 |
|---|---|
| 강한 신호 (clean crop median rank ≤ 3) | **11 / 17 (65%)** |
| 약한 신호 | 6 / 17 (35%) |

> ✱ 검증 패치 (2026-06-10): 원래 기준 (cos_mean ≥ 0.20) 으로는 12/17 (onion segments 0.201 턱걸이 포함). scene 전체 prompt 와의 **rank 재검증**으로 onion segments 는 약한 신호로 재분류 (median rank 6.0, top-3 view 17%) — headline **11/17 (65%) 는 유지**, 기준만 더 강해짐. [verify_h2lite_rank.csv](../../../output/diagnostics/verify_h2lite_rank.csv)

**충격적 사실**: `rubber duck with hat` 은 cos_mean 0.295 (매우 강함) 인데 pipeline 에선 phantom. `plate` 도 0.234 인데 phantom. → **재료 ≠ 결과물**.

### Layer 2 — Phantom 의 4 가지 모양 (F2.B: subtype 분류)

| Subtype | 개수 | 비유 | Fix path |
|---|---|---|---|
| 💪 **Strong-signal phantom** | **11 (65%)** | "재료 강한데 가공이 죽임" | within-view fix |
| 🌫️ Mean-dilution | 4 (24%) | "재료도 약함" (old camera, pumpkin, napkin, ottolenghi) | encoder 측 (C2/C3) |
| ⚖️ Bimodal-balanced | 2 (12%) | "절반 강, 절반 약 — ROFA 못 잡음" (pikachu, sake cup) | mode-cluster aggregation |
| 🎯 Outlier-handled | **0 (0%)** | ROFA 가 의도대로 작동한 케이스 | — |

> **0% outlier-handled** 가 결정적. ROFA 가 만든 'robust mean' 이 17 phantom 중 **단 하나도 살리지 못함**. ROFA 가 잘못된 게 아니라, **잡을 outlier 가 애초에 없는 phantom 위에서 작동**.

### Layer 3 — ROFA 는 정직했다 (F2.C: keep-mask)

각 phantom 에서 ROFA 가 keep 한 view 들 vs drop 한 view 들의 cos 비교:

- **82% (14/17)**: kept_cos > dropped_cos → ROFA 가 *정직하게* 더 정답에 가까운 view 를 골랐다 *(simulation τ=1.0 기준)*
- **18% (3/17)**: ROFA 실수 — **old camera, ottolenghi, sink** (강한 view 를 outlier 로 잘못 drop) *(simulation τ=1.0 기준)*

> ✱ 검증 패치 (2026-06-10) — **실제 pipeline default 는 τ=2.0** ([ReLaGS/merge_proj.py:124](../../../ReLaGS/merge_proj.py#L124)). 재실행 결과: 정직 65% (11/17) + drop 없음 12% (2/17 — sink, cabinet), **실수 24% (4/17) = old camera, ottolenghi, pikachu, onion segments**. sink 는 명단에서 빠지고 instance-confusion 2 건 (pikachu, onion) 이 들어옴 — "실제 τ 에서 ROFA 가 instance confusion 의 정답 view 를 drop 한다"는 새 단서. 결론 자체 ("ROFA 는 주범 아님", "0% outlier_handled") 는 유지. [verify_f2_tau_sweep.csv](../../../output/diagnostics/verify_f2_tau_sweep.csv)

→ "ROFA 가 폭력적으로 정답을 죽인" 게 아니라 **'더 정답에 가까운 view 만 모았는데도 phantom'** → 받기 전에 이미 다 죽어있었다.

### Layer 4 — D3 top-k 는 free lunch 아님

| k | 전체 67 prompt mIoU | 17 phantom mIoU |
|---|---|---|
| 1 | 0.464 | 0.155 |
| 3 (default) | **0.542** | 0.203 |
| **10** | 0.483 | **0.290** |

전체 보면 k=3, phantom 만 보면 k=10 — trade-off. → **prompt-conditional adaptive k** 필요 (ReLaGS gap-cut 도 이 시도지만 phantom 19% 만 살림).

### Layer 5 — Text-side trick (C3) 도 못 살림

Instance confusion 3 개에 negative contrast `score = cos(target) − λ · max cos(neg)`:

- pikachu: rank 64 → 50 (marginal, 여전히 catastrophic)
- rubber duck: 5 → 6 (악화)
- onion segments: 14 → 13 (marginal)

→ **SP feature 가 wrong direction 으로 가는 건 text-side 로 되돌릴 수 없음**. instance confusion = 반드시 aggregation 단계 수정 필요.

---

## 🌳 31% phantom 의 내부 구조 (이제 완전 분해됨)

```
67 prompts (4.2:1 phantom:real)
├── 36 Easy (54%)
├── 21 D2.phantom (31%)
│   ├── 17 persistent (THGS + ReLaGS 모두 실패)
│   │   ├── 💪 11 strong_signal (within-view killed)  ← 새 method 의 main target
│   │   │   ├── 4 D3-recoverable     (k=10 으로 부분 회복)
│   │   │   ├── 3 instance_confusion (C3 negative 못 살림 → aggregation fix 필수)
│   │   │   ├── 2 ROFA mechanism fail (old camera, ottolenghi, sink)
│   │   │   └── 2 structural catastrophic
│   │   ├── 🌫️ 4 mean_dilution (encoder 측 fix 필요)
│   │   └── ⚖️ 2 bimodal_balanced (mode-cluster aggregation)
│   └── 4 method-specific phantom
├── 5 D2.real (7.5%)  ← CLIP 한계
└── 5 Rare (7.5%)
```

> ✱ 검증 패치 (2026-06-10): 위 트리의 "ROFA mechanism fail" 멤버는 τ=1.0 simulation 기준. **실제 pipeline τ=2.0 기준 ROFA 실수 = old camera, ottolenghi, pikachu, onion segments (4건)** 이고 sink 는 drop 0건. 또한 onion segments 는 rank 재검증 결과 strong_signal 이 아니라 encoder 측 (clean crop median rank 6.0) — 트리 재구성은 Stage 3 전에 반영 예정.

---

## 📈 페이퍼 narrative 의 진화 3 단계

| Stage | 주장 | 정밀도 |
|---|---|---|
| Stage 1 | "31% phantom : 7.5% real = 4.2:1, method-agnostic" | 거시 분포 |
| Stage 2A | "17 persistent 중 71% 가 across-view ROFA 실패" | 진단명 분포 |
| **Stage 2B** | "그 71% 의 *진짜 범인* 은 ROFA 가 아니라 within-view SAM mixing — 65% 의 phantom 이 clean encoding 으로는 강함" | **mechanism 위치 확정** |

→ 페이퍼 contribution 의 칼끝이 **"within-view SAM mixing fix"** 로 좁혀짐. 막연한 "더 좋은 aggregation" 이 아니라 **파이프의 어느 1 line 을 어떻게 고칠지** 가 정해짐:

```python
# 현재 (ReLaGS merge_proj.py:167-168)
sp_mask_mat[sp_mask_mat < 0.3] = 0
sp_feat = sp_mask_mat @ view_level_feature   # ratio-weighted sum ← 여기서 죽음

# 후보 (Stage 3)
# A. Hard assignment: sp_feat = view_level_feature[argmax(sp_mask_mat)]
# B. RATIO_THRESHOLD ↑ (0.5, 0.7, 0.9)
# C. Query-conditioned top-view 만 평균
# D. Mode-cluster center (bimodal 2 case 회복용)
```

---

## 🎯 Stage 2B 완료 시점 — 한 문장 결론

> **"ROFA 같은 평균 단계는 무죄. 진짜 범인은 평균 *전* 의 within-view SAM mixing 단계다."**

---

## 💡 Stage 2B 완료 시점 — 직관 한 페이지 요약

```
┌────────────────────────────────────────────────────────────┐
│  Stage 1: "실패의 81% 는 회복 가능 (phantom)"               │
│           "ReLaGS 도 같음 = 구조적 한계"                   │
│                                                              │
│  Stage 2A: "17 환자 중 71% 는 across-view 단계 실패"       │
│            "wrong top-1 의 65% 가 background drift"          │
│                                                              │
│  Stage 2B: "사실 across-view ROFA 는 대체로 무죄"           │
│            "(τ=2 검증: 실수 4/17, outlier_handled 0%)"      │
│            "범인은 within-view SAM mixing — 11/17 (65%)"    │
│            "재료(clean CLIP)는 cos≥0.20 강한데 가공이 죽임"  │
│            "ROFA 가 의도대로 작동한 케이스: 0/17 (0%)"      │
│            "D3, C3 도 phantom 못 살림 → aggregation 만이 길" │
│                                                              │
│  다음 (Stage 3): "within-view 단계의 1 line fix 4가지 비교"  │
│    A. hard assignment   B. RATIO_THRESHOLD ↑               │
│    C. query-conditioned D. mode-cluster                     │
└────────────────────────────────────────────────────────────┘
```

---

## 🎓 Stage 2B 완료 시점 — 페이퍼의 메인 주장

> **"ROFA 같은 robust aggregation 은 *깨끗한 재료* 위에선 잘 작동한다. 하지만 우리 파이프라인은 평균 *전* 단계 (SAM ratio mixing) 에서 이미 재료를 잃었기 때문에, 더 나은 평균이 아니라 *재료를 살리는 가공* 이 필요하다."**

이전 (Stage 2A) 의 주장이 *across-view ROFA dominant* 라고 했다면, Stage 2B 후의 주장은 **파이프의 1 line 까지 정확히 지목** ([ReLaGS/merge_proj.py:167-168](../../../ReLaGS/merge_proj.py#L167-L168) 의 `sp_mask_mat @ view_level_feature`).

---

## 📝 업데이트 로그

| 날짜 | 추가/수정 | 한 줄 요약 |
|---|---|---|
| Stage 2A 완료 시점 | 초기 작성 | Stage 1 + Stage 2A 직관 요약, Stage 2B placeholder |
| Stage 2B 완료 시점 | Stage 2B section 채움 | within-view SAM mixing 이 진짜 범인 (65% strong-signal phantom), ROFA 는 무죄 (82% 정직, 0% outlier-handled) 로 reframe. 한 문장 결론 + 페이퍼 메인 주장 sharp 화. |
| 2026-06-10 | ✱ 검증 패치 | (1) F2 시뮬을 실제 pipeline τ=2.0 으로 재실행 — subtype 분포 동일 (65% 유지), F2.C 는 정직 65%+12% no-drop / 실수 4건 (sink→out, pikachu·onion→in) 으로 정정. (2) joint 2×2 raw/canon robustness — 67 prompt 분류 변화 0 (Stage 1 견고). (3) H2-lite rank 재검증 — onion segments 를 encoder 측으로 재분류, 11/17 (65%) 유지. |
| 2026-06-10 | **Stage 3.1 완료** | B8 인과 replay (충실도 33/33): **within-view mixing 무죄 확정** (gap 0.004, per-view rank 1-7), 한 문장 결론을 "범인은 across-view 누적 (B1)" 으로 교체. hard-assignment fix 는 실행 전 기각. Stage 3.2 = B1 anatomy (dump 재사용). |
| 2026-06-11 | **Stage 3.2 완료** | B1 해부 (R1·R2·R3 전부 확정): phantom = 소수(18%) 탁월 view 가 다수결(평균)에 묻힘; 경쟁 wrong SP 는 coherent-plausible (평균 prompt-cos 도 우위 → mean 게임 구조적 열세); **query-cond top-k 15/17 회복 + regression 0** → Stage 4 = query-aware aggregation. 보너스: pumpkin 의 wrong-top1 = zero-norm (D1 잔존), spoon·cabinet = never_good (정직한 잔여). |
| 2026-06-11 | **✱ 적대적 검증 패치** | "15/17" 은 동결-경쟁자 상한이었음 — 공정 결투 (wrong 도 query-top-5 보정) 에서 oracle 단독 승리 3/15, 후보권 복귀 13/15 (대부분 rank 2, 문제가 D2 실종 → D3 오염으로 전이). R1/R3 은 유지·강화. Stage 4 는 query-aware + 결합 신호 (zero-norm 필터, coherence prior, 공간 정합, negative contrast) 로 설계 격상. |
| 2026-06-11 | **Stage 3.3 완료 (확정 진단)** | proxy 전부 제거: all-SP dump (gate 98-100%) + 동결 없는 full-pool 재채점 + mask-IoU 208쌍. **R4 FAIL** — query-top-k 제로섬 (phantom 2배 ↔ easy −9.8pt). **R5** — semantic confusion 62% / 진실의 조각 32%. coherence 감점 무익. zero-norm 150 전수. ✱사후 탐색: hybrid α=0.3 = rank 순 +9 → **Stage 4 = hybrid mask 검증 (scene-split calibration)**. |
| 2026-06-11 | **Stage 3.4 완료 (잔여 전수 분해)** | 36 케이스 이름표 완성 (R6 PASS, unknown 2). 신규 원인: **few-view opportunist** (mean 은 저관찰 SP 에 유리) + **multi-instance 미표기 의심 7 prompt (R7 발동)**. 가짜 역행 3건 (rank 착시). **R8: encoder 한계 프레임 해체** — D2.real 4/5 + 2B encoder-측 4/4 가 전부 dilution, 최종 잔여 3건. Stage 4 부품 목록 확정. |
| 2026-06-12 | **Stage 4 완료 (method LOSO 검증)** | **R9 PARTIAL 최종**: held-out +3.74pt (phantom +17.5, other +13.3, multi-inst 제외 +4.53) — headline 합격, easy −4.08 로 보호 기준 미달. g1 결함 적발 (ablation) + g2 margin 포화 부검. 한 문장 결론 갱신: "심장은 뛴다, 남은 건 easy 를 알아보는 눈". |
| 2026-06-12 | **Stage 5 완료 (ReLaGS 재현)** | **G5: 전부 재현** — 두 법정 같은 판결 (95%·23/51%·+8/−12), ROFA 실측 = 유령 청소부일 뿐 (phantom median 0, 구출 1/20). "다수결의 비극 = paradigm-level" 확정, Section 2 완결. ReLaGS dump 1.2GB 확보 → R10 즉시 가능. |

<!-- 새 stage 추가 시 위 표에 한 줄 추가하고, 본문에는 새 section 을 "Stage 2A 가 결정한 것" 과 "Stage 2B" 사이에 넣을 것 -->

---

## 🔬 Stage 3.1 — "주방 재현 실험: 1차 가공도 무죄였다"

상세: [stage3_b8_causal.md](stage3_b8_causal.md)

### 비유 업데이트

Stage 2B 까지: "재료(clean crop)는 멀쩡한데 요리 과정이 죽인다 — 아마 1차 가공(within-view mixing)에서."

Stage 3.1 이 한 것: **주방에 직접 들어가서 그 1차 가공을 똑같이 재현**하고 (충실도: 재현 요리가 실제 요리와 cos 0.994-0.9995 일치, 33/33), 가공 직전/직후/다르게-가공한 맛을 같은 자리에서 비교.

→ **1차 가공 직후에도 맛이 멀쩡했다** (17 phantom 전부 per-view rank 1-7, easy 는 가공이 오히려 맛을 *개선*). 그런데 같은 재료로 만든 *최종* 요리만 망한다 (rank 4-256). **망치는 단계는 마지막 합치기 (across-view 누적, B1) 하나만 남았다.**

### 숫자 3개로 요약

| | 값 | 의미 |
|---|---|---|
| strong_signal 11 의 clean−mixed gap | **0.004** (기준 0.05 의 1/13) | within-view 손실 사실상 0 |
| per-view mixed rank → 최종 rank | **1-7 → 4-256** (tesla: 3→256) | 살해 지점은 그 사이 = B1 |
| easy control 의 gap | **−0.018** | mixing 은 평균적으로 *도움* (맥락 효과) |

### 사전 등록 판정의 가치가 증명된 순간

실험 *전에* 적어둔 분기 ③ ("gap 자체가 작으면 → within-view 무죄, B1 재조준") 이 그대로 발동. 만약 인과 실험 없이 바로 Stage 3.2 (hard-assignment full re-run) 로 갔다면 — hard 미리보기가 보여주듯 **회복할 gap 자체가 없어서** 하루를 버렸을 것. fix 보다 인과를 먼저 닫은 설계의 직접 보상.

### 다음 (Stage 3.2 — B1 anatomy)

replay 가 33 SP × 전체 view 의 per-view feature 를 dump 해 둠 → **GPU 재작업 없이** B1 해부 가능:
- **B1.A**: view 를 하나씩 누적하며 신호가 *언제* 무너지는지 궤적 추적
- **B1.C**: phantom 이 지는 게 자기 약화인가, 경쟁 SP (background drift) 의 상대 강화인가

---

## 🎯 Stage 3.1 완료 시점 — 한 문장 결론

> **"재료도, 1차 가공도 무죄. 범인은 마지막 합치기 (across-view 누적) 다 — 이제 용의자가 단 하나 남았다."**

---

## ⚖️ Stage 3.2 — "다수결의 비극, 그리고 소수 정예"

상세: [stage3_2_b1_anatomy.md](stage3_2_b1_anatomy.md)

### 비유: 목격자 투표

진술서 100장을 평균내는 것 = **다수결 투표**다. Stage 3.2 가 투표를 해부했다:

1. **phantom 의 정답 진술은 소수파였다** (B1.A): 17 phantom 중 15 개는 *단 한 장의 진술서만으로도* 전체 경쟁(pool)에서 1등이 가능하다. 그런데 그런 탁월한 진술서가 **전체의 18%뿐** (easy 는 50%). 다수결에서 묻힌다.
2. **이기는 상대는 '균질하게 그럴듯한' 후보였다** (B1.C): wrong SP 의 진술서들은 만장일치에 가깝고 (coherence 우위, p=0.013), 심지어 **평균 점수도 oracle 보다 높다**. 즉 mean-vs-mean 게임은 *원리적으로* oracle 이 진다 — 더 나은 평균(outlier 제거, cluster 중심, uniform)으로는 못 이긴다. 실제로 ablation 에서 평균 계열 전부 실패 (GMM 3/17+부작용 6, ROFA 1/17, uniform 은 개악).
3. **해법: query 가 지명하는 소수 정예** (B1.B): "피카츄" 라는 질문과 가장 맞는 top-5 진술서만 쓰면 — **17 중 15 회복, easy 부작용 0**. oracle 의 비교우위는 평균이 아니라 *최고의 몇 장* 에 있기 때문.

### 보너스 발견 2개

- pumpkin 을 이긴 SP 는 **feature 가 0 인 유령** (zero-norm) — canon-contrast 가 0 벡터에 0.5점을 줘서 1등이 됨. 옛 D1 메커니즘의 잔존. zero-norm 필터 1줄 과제.
- spoon·cabinet 은 어떤 단일 진술서도 top-3 불가 (never_good_in_pool) — 이 2개 + encoder-limit 5개가 method 가 못 구하는 정직한 잔여.

### Stage 4 로 가는 설계 사양 (실험 숫자가 직접 결정)

- SP 당 단일 벡터 → **per-view prototype 몇 개** 로 표현 교체 (소수파 보존)
- inference 에서 **query-conditioned top-k (k≈5) 채점**
- 기대 상한: phantom 15/17 회복 + regression 0 → full 67-prompt mIoU 로 검증

---

## 🎯 Stage 3.2 완료 시점 — 한 문장 결론

> **"phantom 의 원인은 '나쁜 feature' 가 아니라 '나쁜 투표 제도' 였다. 정답은 평균을 고치는 게 아니라, query 에게 투표권을 주는 것."**

---

## 🏛️ Stage 3.3 — "확정 진단: 두 개의 헌법"

상세: [stage3_3_definitive.md](stage3_3_definitive.md)

### 비유의 완결

3.2 는 "query 에게 투표권을 주자" 였다. 3.3 이 그 제도를 **전국 단위로 시행**해 봤다 (동결 없는 전 pool 재채점 + 실제 mask mIoU):

- **phantom 선거구**: 대성공 — IoU 0.203 → **0.402 (2배)**. pikachu 64→1, tesla 256→4.
- **easy 선거구**: 재앙 — 41곳 중 10곳에서 운 좋은 소수파(엉뚱한 SP 의 우연한 view 5장)가 당선. −9.8pt.
- **전국 합산: ±0.1pt 제로섬. R4 FAIL.**

→ **두 개의 헌법이 필요하다는 게 확정**: mean (다수결) 은 easy 를 지키는 헌법, top-k (소수정예) 는 phantom 을 지키는 헌법. 어느 하나로 통일하면 반대쪽이 무너진다. 사후 탐색에서 **절충 헌법 (α·mean + (1−α)·top-k, α=0.3)** 이 rank 수준 순 +9 (phantom 4, other 7, easy −2) — Stage 4 는 이걸 mask 수준에서 입증하는 것.

### 사기꾼의 정체 (R5)

- **62% (10/16)**: 진짜 confusion — CLIP 눈에 정말 prompt 같아 보이는 배경 (GT 와 겹침 0)
- **32% (5/16)**: 알고 보니 **진실의 조각** — tesla 의 "사기꾼" 은 문 손잡이의 일부였다 (precision 1.00). 이들의 실패는 confusion 이 아니라 granularity.
- 보너스: zero-norm 유령 SP 전수 150개 확인 (옛 D1, canon(0)=0.5 로 경쟁) — 필터 1줄 거리.

---

## 🎯 Stage 3.3 완료 시점 — 한 문장 결론

> **"진단은 완결됐다: 소수파 매장 (원인) + 제로섬 (단일 처방의 불가능성) + 사기꾼 분류 (62% confusion / 32% granularity). 남은 것은 처방의 공학 — hybrid 의 mask 검증."**

---

## 🧾 Stage 3.4 — "잔여 전수 분해: 마지막 수수께끼들의 정체"

상세: [stage3_4_residual_causes.md](stage3_4_residual_causes.md)

### 풀린 수수께끼들

1. **spoon·cabinet 은 왜 어떤 view 로도 안 됐나** → 이기는 놈들이 전부 **view 2~11개짜리 미세 SP** (coherence 0.92-0.99). view 가 2개면 평균을 내도 희석이 없다 — **mean 게임은 '적게 관찰된 놈'에게 구조적으로 유리**하다는 새 메커니즘. 게다가 이들 prompt 는 **주방에 실물이 여러 개** (multi-instance 미표기 의심 7 prompt 46건) — 일부 "실패" 는 벤치마크 책임.
2. **"진실의 조각" 사기꾼들** → NAG 친족 감사로 확정: jake 의 wrong 은 사실상 같은 물체 (포함률 1.00/0.98), tesla 의 wrong 은 1% 조각, sink 는 부모 — **parent-union 한 줄 처방**.
3. **easy 역행 10건** → 3건은 **가짜** (같은 물체의 다른 계층 항목이 뽑힘 — rank 착시, mask 무해). 진짜 7건의 범인은 "mean 에서 평범하다가 top-k 로 점프한" 외부 SP (gap +0.17~+0.52) → **가드 신호 2개 확보** (점프 폭 g1, mean 확신도 g2).
4. **가장 큰 반전 — encoder 한계는 거의 없었다**: Stage 1 이 "CLIP 도 모르는 진짜 어려움" 이라던 D2.real 5건 중 **4건이 query-aware 로 회복** (miffy 64→≤3 포함!), 2B 의 "encoder 측" 4건도 전원 회복 (pumpkin 52→1, ottolenghi 66→3). **encoder 핑계의 최종 잔여는 단 3건** — 나머지는 전부 우리가 평균으로 가렸던 것.

### 🎯 Stage 3.4 완료 시점 — 한 문장 결론

> **"수수께끼는 다 풀렸다. 남은 실패 36건 전원에게 이름표가 붙었고 (unknown 2), 그 이름표들이 곧 method 의 부품 목록이다: 가드된 hybrid + 계층 union + 최소관측 prior + 유령 필터 + 벤치마크 보정."**

---

## 🏗️ Stage 4 — "처방의 첫 임상시험: 본체는 합격, 가드는 재수강"

상세: [stage4_method.md](stage4_method.md)

### 무엇을 했나

부품들을 조립한 채점기를 만들어 (조립이 정확하다는 증명: 양 극단 재현 0/67 불일치), **공정한 임상시험**을 했다 — 4개 장면 중 3개로만 설정을 고르고 나머지 1개로 채점 (4회 회전, LOSO). 답안지를 보고 튜닝하는 것이 원천 불가능한 구조.

### 결과 (held-out, 시험 2회차 = 허용된 재탐색 1회 사용)

```
전체 67문제:   0.542 → 0.580  (+3.74pt)  ← 합격선(+2.0) 통과
유령 17문제:   0.203 → 0.378  (+17.5pt)  ← 거의 2배
기타 9문제:    0.286 → 0.418  (+13.3pt)
쉬운 41문제:   0.740 → 0.699  (−4.08pt)  ← 보호 기준(<1pt) 미달 → R9 PARTIAL
```

### 배운 것 2개 (둘 다 ablation/부검이 자백)

1. **v1 의 g1 가드(전역 클램프)는 설계 결함** — 사기꾼의 점프를 누르려다 정답의 점프(유령 회복의 원천)까지 눌렀다. 끄니 +8.7pt.
2. **v2 의 g2 가드(margin 기반)는 불활성** — canon 점수가 포화돼 margin 이 모두 0.01 언저리라 41개 easy 중 1개만 보호됨. "mean 이 확신하면 보호한다" 는 개념 자체는 손실 분해가 지지 (보호된 쪽 −0.17 vs 미보호 −1.50) — **분리 신호를 잘못 골랐을 뿐**.

### 한 문장 결론

> **"처방의 심장 (hybrid) 은 뛴다 — 회복력 +17.5pt 가 증명. 남은 건 단 하나의 공학 question: easy 를 알아보는 더 좋은 눈 (가드 분리자). 그게 Stage 5 다."**

---

## 🌍 Stage 5 — "다른 법정에서도 같은 판결: 다수결의 비극은 보편 법칙"

상세: [stage5_relags_replication.md](stage5_relags_replication.md)

### 비유

지금까지의 재판은 전부 THGS 라는 한 법정에서 열렸다. ReLaGS 는 **다른 배심원단 (자체 partition) + 전속 경호원 (ROFA)** 을 가진 법정이다. 거기서 같은 사건을 다시 재판했다 (경호원까지 포함한 재현이 진짜임을 충실도 97.8~100% 로 증명).

**판결: 전부 동일.**

| | THGS 법정 | ReLaGS 법정 |
|---|---|---|
| 정답의 최고 진술서는 1등감인가 | 88% | **95%** |
| 정답의 좋은 진술 비율 (vs easy) | 18% vs 50% | **23% vs 51%** |
| 소수정예 전면 도입 시 (회복/역행) | +6/−10 | **+8/−12** |

### 경호원 (ROFA) 의 정체도 확정

ROFA 가 실제로 하는 일을 켜고-끄고 비교해 보니: **유령 (feature 0 짜리) 청소는 탁월** (150→21 마리) — 그런데 **다수결의 비극 앞에서는 무력** (phantom 순위에 median 효과 0, 20명 중 1명 구출). 경호원은 침입자는 막지만, 배심원단의 투표 제도 자체는 못 바꾸기 때문.

### 🎯 Stage 5 완료 시점 — 한 문장 결론

> **"이제 말할 수 있다: 실패의 원인은 특정 모델의 버그가 아니라 'CLIP 화살표를 평균내어 물체당 1개로 만든다' 는 paradigm 그 자체다. 두 법정, 같은 판결, 경호원도 무력 — 페이퍼 Section 2 완결."**

---

## 🔗 참고

- [experiment_results.md](experiment_results.md) — **코드 기반 실험 정의 + 결과 표** (이 문서의 짝꿍)
- [README.md](README.md) — stage 진행 index
- [../extended_failure_hypotheses.md](../extended_failure_hypotheses.md) — 18 가설 catalog (6.2 patch)
- [stage1_b7_a4_a2.md](stage1_b7_a4_a2.md) — Stage 1 상세
- [stage2a_phantom_anatomy.md](stage2a_phantom_anatomy.md) — Stage 2A 상세
- [stage2b_rofa_anatomy.md](stage2b_rofa_anatomy.md) — Stage 2B 상세
