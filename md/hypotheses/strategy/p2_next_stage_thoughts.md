# P2 Next Stage Thoughts — guard 재설계와 method 본선 초안

> 작성: 2026-06-15, Codex working draft.
> 상위 문서: [roadmap.md](roadmap.md), P1 결과: [competitor_autopsy.md](competitor_autopsy.md), [p1_problem_experiments.md](p1_problem_experiments.md).
>
> 목적: P1 이 만든 "기존 처방들의 실패 명세"를 P2 의 실험 가능한 method 설계로 바꾸기.

---

## 0. 한 문장

> **P2 의 핵심은 더 강한 view-selection 이 아니라, view-selection 이 언제 위험한지 아는 guard 를 만드는 것이다.**

Stage 4 에서 이미 본체는 살아 있음이 보였다. Query-aware hybrid 는 held-out full-67 을 +3.74pt 올렸고 phantom 을 +17.5pt 살렸다. 문제는 easy -4.08pt. 즉 "phantom 을 살리는 힘"은 확보했고, 남은 질문은 **easy 를 보호하는 분리자**다.

P1 은 그 분리자 설계를 위해 중요한 힌트를 줬다:

- `margin` 은 버린다. Stage 4 g2 실패와 P1-B absence AUROC 0.52~0.62 가 같은 결론이다.
- `top1-conf` 는 채택 후보 1순위다. P1-B 에서 present/absent AUROC 0.836/0.848.
- robust statistic family 는 본체가 아니다. gm_g 는 easy +1.2pt 이지만 phantom -1.8pt.
- selection family 는 본체이지만 guard 없이는 제로섬이다. top5 는 phantom +19.9pt, easy -9.8pt.

따라서 P2 는 "새로운 aggregation 을 더 복잡하게 만들기"보다 **언제 mean 을 믿고, 언제 query-topk 를 믿을지 판정하는 작은 decision layer** 로 가는 게 맞다.

---

## 1. 내가 보는 현재 병목

### 병목 A — easy 와 phantom 의 목표가 다르다

easy 는 mean 체제가 이미 잘 맞는 케이스다. 여기서 top-k view selection 을 켜면 lucky-view 외부 SP 가 튀어 올라 easy 가 깨진다.

phantom 은 mean 체제가 정답 view 를 다수결로 묻어버린 케이스다. 여기서는 top-k view selection 을 켜야 정답이 후보권에 돌아온다.

즉 하나의 scoring rule 로 모두를 처리하려 하면 제로섬이 생긴다.

### 병목 B — 기존 confidence 는 잘못된 confidence 다

Canon margin 은 포화되어 있다. easy 에서도 top1-top2 gap 이 작고, 좋은 mask IoU 를 내는 prompt 도 margin 이 거의 0 인 경우가 있다. 그러니 "margin 이 크면 mean 보호" 같은 guard 는 거의 작동하지 않는다.

반대로 top1 absolute confidence 는 부재 쿼리에서 유의미한 분리 신호를 보였다. 내 해석은 이렇다:

> margin 은 "1등과 2등의 상대 차이"라서 crowded pool 에 취약하고, top1-conf 는 "이 scene 안에 정말 이 query 를 받을 만한 것이 있는가"에 더 가깝다.

### 병목 C — rank regression 과 mask regression 이 다르다

Stage 3.4 에서 easy 역행 10 중 3 은 kin_same 가짜 역행이었다. rank 로는 틀린 SP 가 올라왔지만 mask 로는 같은 물체/부모/자식이라 피해가 작다.

따라서 P2 guard 는 단순히 "top1 이 바뀌면 위험"이 아니라, **바뀐 후보가 NAG 친족인지**를 봐야 한다.

---

## 2. P2 의 설계 원칙

1. **Selection 은 유지한다.**
   Phantom 회복의 동력은 query-topk 쪽에 있다. 이걸 약하게 만들면 Stage 4 의 +17.5pt 를 잃는다.

2. **Guard 는 prompt-level 과 SP-level 을 분리한다.**
   Prompt-level guard 는 "이 query 는 mean 체제를 믿어도 되는가"를 판정한다.
   SP-level guard 는 "이 새 승자는 위험한 lucky-view jump 인가"를 판정한다.

3. **Canon margin 대신 raw/canon 복수 신호를 쓴다.**
   Canon score 는 inference 와 일치하므로 최종 score 에 필요하지만, guard feature 로는 raw-cos z-score, top1-conf, rank stability 를 같이 봐야 한다.

4. **NAG kinship 은 guard 의 예외 규칙으로 둔다.**
   parent/child/same 은 rank regression 이라도 mask regression 이 아닐 수 있다. 이걸 막지 않으면 guard 가 지나치게 보수적이 된다.

5. **THGS 에서 만든 guard 는 ReLaGS 에서 검증한다.**
   P2 의 가장 강한 문장은 "drop-in + transferable" 이다. 그러려면 guard 가 THGS 전용 hyperparameter 가 아니어야 한다.

---

## 3. R12 사전등록 초안

### Question

> P1/P4 가 찾은 guard 신호로 query-aware hybrid 의 easy regression 을 1pt 이내로 줄이면서 phantom 회복을 유지할 수 있는가?

### 모집단

- THGS: full 67 prompt, category = phantom17 / easy / other
- ReLaGS: cross-method class 기준 phantom/easy/other
- 학습/선택: THGS LOSO 또는 THGS train scenes
- 검증: held-out THGS scene + ReLaGS cross-method held-out

### 후보 신호

| Signal | 직관 | 기대 |
|---|---|---|
| `top1_conf_mean` | mean 체제의 top-1 절대 confidence | present/easy 보호, absence ghost 억제 |
| `regime_agree` | mean top-1 과 hybrid/topk top-1 이 같은가 | 같으면 안전, 다르면 guard 후보 |
| `kinship(mean_top1, topk_top1)` | 바뀐 후보가 same/parent/child 인가 | 가짜 역행 면제 |
| `raw_z_top1` | raw-cos pool z-score | canon 포화 우회 |
| `rank_stability` | k=3/5/10 또는 view bootstrap 에서 top 후보가 안정적인가 | lucky-view jump 탐지 |
| `n_valid_views` | few-view opportunist 감점 | 저관찰 외부 SP 억제 |
| `top_minus_mean_gap` | topk 항이 mean 항보다 비정상적으로 튀는가 | Stage 3.4 g1 의 개선판 |

### Guard 형태

우선 복잡한 학습기는 피하고, threshold rule 로 간다.

```text
if mean_is_confident(prompt) and not topk_candidate_is_kinship_safe:
    use mean / raise alpha
else:
    use hybrid
```

두 번째 후보는 SP-level clamp:

```text
score = alpha * mean + (1-alpha) * topk
if candidate is few-view opportunist or unstable lucky-view:
    topk contribution is clamped or downweighted
```

### 성공 기준

주 성공:

- held-out full-67 >= baseline +2pt
- easy loss < 1pt
- phantom gain >= +10pt

강한 성공:

- 위 기준을 THGS 와 ReLaGS 둘 다에서 만족
- guard transplant: top5+guard 가 top5 대비 easy 손실을 절반 이상 줄이고 phantom gain 의 70% 이상 유지

실패 판정:

- easy loss 가 계속 >2pt 이면 guard feature 재검토
- phantom gain 이 +5pt 이하로 떨어지면 guard 가 과보호
- THGS 에서만 통하면 base-specific method 로 주장 강도 축소

---

## 4. 내가 추천하는 실행 순서

### Step 1 — Guard feature table 만들기

먼저 mask render 없이 모든 prompt/SP 후보에 대해 feature table 을 만든다.

입력:

- `stage3_3_allsp_<scene>.pkl`
- `stage5_relags_allsp_<scene>.pkl`
- `sai_nag.pt`
- `stage4_grid_ranks_v2.csv`
- `stage3_4_*` kinship/taxonomy outputs
- P1-B absence score 결과

출력:

- `output/diagnostics/p2_guard_features_thgs.csv`
- `output/diagnostics/p2_guard_features_relags.csv`

필수 column:

```text
scene, prompt, category,
baseline_iou, stage4_iou,
mean_top1, topk_top1, hybrid_top1,
oracle_pair,
top1_conf_mean, top1_conf_topk,
mean_top1_score, mean_top2_score,
canon_margin, raw_z_top1,
regime_agree_mean_topk,
kinship_mean_topk, kinship_topk_oracle,
n_valid_topk, coherence_topk,
rank_k3, rank_k5, rank_k10,
rank_stability,
is_easy_regression, is_phantom_recovery
```

이 테이블이 P2 의 현미경이다. 만들고 나면 guard 후보는 대부분 pandas 로 검증 가능하다.

### Step 2 — 단변량 분리력 보기

각 feature 에 대해 easy regression vs phantom recovery 분리력을 본다.

- AUROC
- threshold sweep
- precision/recall
- scene별 안정성
- THGS threshold 를 ReLaGS 에 그대로 적용했을 때 성능

여기서 탈락할 신호는 빨리 버린다. 특히 `canon_margin` 은 negative control 로 넣되 기대하지 않는다.

### Step 3 — 2-rule guard 만 먼저 시도

처음부터 복잡한 조합으로 가지 말고, 두 줄짜리 rule 을 먼저 만들자.

후보 1:

```text
if top1_conf_mean >= tau_conf and kinship(mean_top1, topk_top1) is unsafe:
    choose mean
else:
    choose hybrid
```

후보 2:

```text
if regime_agree is false and topk_top1 has low n_valid or low stability:
    clamp topk
else:
    choose hybrid
```

후보 3:

```text
if top1_conf_mean high:
    alpha = high
else:
    alpha = low
```

내 직감은 후보 1 + kinship 예외가 제일 먼저 볼 만하다.

### Step 4 — Mask-level LOSO

rank proxy 로 shortlist 를 만들고, Stage 4 와 같은 방식으로 mask-level LOSO 를 한다.

중요: P2 는 Stage 4 보다 과적합 공격을 더 세게 받을 수 있다. 따라서 문서에 아래 규율을 명시해야 한다.

- guard feature 선정은 P1/Stage3.4 에서 나온 신호만 사용
- threshold 는 calibration scenes 에서만 선택
- held-out scene 의 mask IoU 는 마지막에 한 번만 본다
- THGS 에서 고른 rule 을 ReLaGS 에 그대로 또는 최소 조정으로 적용

### Step 5 — Guard transplant

이 실험은 novelty 방어에 좋다.

비교:

- top5
- top5 + our guard
- gm_g
- gm_g + our guard
- our hybrid
- our hybrid + guard

기대 문장:

> "Our contribution is not just a tuned hybrid score; the diagnosed guard repairs the failure mode of view-selection families themselves."

즉 남의 처방도 우리가 고쳐주는 그림이다.

---

## 5. P2 에서 조심할 함정

### 함정 1 — easy 보호만 하다 phantom 을 죽이는 것

Stage 4 의 실패를 보면 guard 를 세게 걸수록 쉬운 케이스는 보호되지만 phantom 회복이 사라질 위험이 크다. 그래서 threshold 선택 기준은 easy loss 만 보면 안 되고, 반드시 phantom gain lower bound 를 둬야 한다.

추천 proxy:

```text
score = phantom_recovered + 0.5 * other_recovered - 2 * easy_regressed
```

단 최종은 mask mIoU 로만 판단.

### 함정 2 — top1-conf 가 absence 에서 좋았다고 easy/phantom 에도 좋으리라 믿는 것

Absence AUROC 0.84 는 강한 힌트지만, P2 target 은 present prompt 안의 easy/phantom 분리다. top1-conf 는 먼저 단변량으로 검증해야 한다.

가능한 결과:

- present/absent 에는 좋지만 easy/phantom 에는 약함
- easy 보호에는 좋지만 phantom 회복을 과하게 막음
- ReLaGS 에서 calibration shift 발생

셋 다 충분히 가능하다.

### 함정 3 — kinship 계산을 과신하는 것

NAG parent/child 관계는 mask 안전성의 proxy 지, semantic correctness 자체는 아니다. parent-union 은 granularity에는 좋지만 over-union을 만들 수도 있다. kinship 은 "면제" 신호로 쓰되, 최종 선택 score 를 완전히 대체하면 위험하다.

### 함정 4 — ReLaGS headline 숫자와 프로토콜

R10 이 중요하다. THGS base 위 +3.74 만으로는 외부 leaderboard 와 싸우기 어렵다. ReLaGS base 위 drop-in 성능이 표 B 의 헤드라인이 될 가능성이 높다.

그러나 ReLaGS mask render/eval 경로가 원저자 프로토콜과 맞는지 먼저 교차확인해야 한다. 이게 틀리면 숫자 전체가 공격받는다.

---

## 6. 가장 좋은 P2 결과의 형태

내가 생각하는 이상적인 결과 표는 이렇다.

| Method | Full | Phantom | Easy | Other | Comment |
|---|---:|---:|---:|---:|---|
| baseline mean | base | base | base | base | stable easy, misses phantom |
| top5 / bag | ~0 | large + | large - | + | zero-sum |
| robust median | small/0 | 0 | small + | 0 | safe but blind |
| Stage4 hybrid | +3.7 | large + | -4.1 | + | body works, guard weak |
| **P2 guarded hybrid** | **+?** | **large +** | **>-1pt** | + | desired |
| top5 + our guard | +? | retained | reduced loss | + | transplant proof |
| ReLaGS + guarded hybrid | +? | + | protected | + | drop-in proof |

논문에서 가장 강한 문장은 이 조합이다:

> Robust fixes preserve easy cases but cannot recover phantoms; view-selection fixes recover phantoms but regress easy cases. A guard derived from our failure taxonomy is the missing ingredient, and it transfers across both base pipelines and competing selection rules.

---

## 7. 내 우선순위

나는 다음 순서가 가장 좋다고 본다.

1. **P2 guard feature table 작성**
2. **top1-conf / regime-agree / kinship / raw-z / stability 단변량 분석**
3. **R12 사전등록 문서화**
4. **2-rule guard 로 rank-level shortlist**
5. **THGS LOSO mask eval**
6. **ReLaGS drop-in mask eval (R10)**
7. **guard transplant**
8. **원저자 eval protocol 교차확인**

P1-E, 즉 VALA/StS 전체 코드 직접 실험은 P2-a/b/c 의 결과가 어느 정도 나온 뒤가 낫다. 지금은 guard 가 논문의 성능 승부처라서, 외부 코드 셋업에 며칠 쓰는 것보다 P2 의 표 B 를 먼저 세우는 편이 더 큰 이득이다.

---

## 8. 결론

P2 는 "새 모델"을 만드는 단계라기보다, 지금까지의 진단이 정말 method 를 낳는지 증명하는 단계다.

Stage 4 가 보여준 것은:

> query-aware selection 은 약이 맞다. 하지만 용량 조절을 못 하면 독이 된다.

P1 이 보여준 것은:

> 기존 경쟁 처방들은 이 용량 조절 장치가 없다.

그래서 P2 의 논문적 가치는 guard 에 있다. 이 guard 가 성공하면 우리의 contribution 은 단순한 score 조합이 아니라, **failure taxonomy 에서 직접 도출된 transferable decision rule** 이 된다.

