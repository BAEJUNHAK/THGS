# THGS CLIP 호출 절감 설계

> 목표: 파이프라인 결과 품질(mIoU)을 유지하면서 **CLIP image encoder 호출 횟수**를 줄이는 알고리즘 설계.
> 파이프라인 전체 흐름은 [THGS_pipeline_analysis.md](THGS_pipeline_analysis.md) 참조. 본 문서는 그 위에서 "CLIP을 덜 부르는 법"만 다룬다.

---

## 0. 결론 먼저

THGS에서 CLIP **image** 인코딩(`encode_image`)은 **오직 전처리(`scripts/image_encoding.py`) 한 곳**에서만 일어난다. 그리고 그 비용 모델은:

```
총 CLIP 호출(crop 수) ≈ Σ_(모든 뷰) (그 뷰의 4입도 SAM 마스크 수)
                    ≈ V × (대략 150~300 masks/이미지)
```

LERF 한 장면(V≈150~300장)이면 **2만~8만 회**의 224×224 ViT-B/16 forward. 이게 전체 전처리 시간의 지배 항이다.

**핵심 관찰**: 전처리 이후 모든 단계 — `sp_partition`(기하), `graph_weight`(SAM+랜덤 임베딩), `merge_proj`의 병합(`extract_gaussian_features`, SAM+랜덤 임베딩) — 는 **CLIP을 전혀 쓰지 않는다.** CLIP 피처는 `merge_proj`의 **맨 마지막 한 단계**(`proj_gaussian_features` / `_x`)에서 `view.semantic["sem"]`(전처리에서 저장된 `_f.npy`)을 *읽기만* 한다.

즉 현재 구조는 **"어떤 3D 영역이 라벨을 필요로 하는지 알기도 전에, 모든 뷰의 모든 마스크를 미리 CLIP으로 인코딩"** 하는 eager 방식이고, 인코딩된 crop의 대부분은 결국 어떤 슈퍼포인트의 피처로도 쓰이지 않는다(proj 단계가 overlap<0.3, portion<0.1 마스크를 버린다). → **여기에 큰 낭비가 있다.**

---

## 1. CLIP 호출 지점 검증 (코드 근거)

| 위치 | 호출 종류 | 빈도 |
|---|---|---|
| `scripts/image_encoding.py:188` `model.encode_image(tiles)` | **image** | 뷰 × 4입도 × 마스크. **지배 비용** |
| `scripts/encode_fastsam_clip.py:259` | image (FastSAM 대체본) | 동일 |
| `utils/vlm_utils.py` `encode_text` | text | 쿼리 프롬프트당 1회. 무시 가능 |
| `gui/main.py:45` | text(쿼리) | GUI 입력당. 무시 가능 |
| `merge_proj.py:71, 125` | **호출 아님** — `view.semantic["sem"]` 읽기만 | CLIP-free |

CLIP-free 검증:
- `graph_weight.py`: 마스크별 임베딩이 `torch.normal(...)` **랜덤 20-d**. CLIP 아님.
- `merge_proj.py extract_gaussian_features`: 동일하게 랜덤 20-d 임베딩으로 ray trace → argmax. CLIP 아님.
- CLIP 피처는 `proj_gaussian_features*`에서만 소비되고, 그것도 사전저장본을 읽음.

→ **CLIP image 인코딩을 "마지막 피처 할당" 직전까지 미룰 수 있다.** 이것이 모든 절감 아이디어의 토대.

---

## 2. 절감 기법 (효과 vs 위험 순)

### Tier 1 — 구조 재배치: "슈퍼포인트 우선, CLIP 지연(lazy)" ★가장 큰 효과

현재: `모든 마스크 CLIP 인코딩` → `슈퍼포인트 구축` → `피처 할당`
변경: `SAM만으로 슈퍼포인트 구축(CLIP 0회)` → `최종 슈퍼포인트마다 best-view top-k 선택` → `그 (슈퍼포인트, 뷰) crop만 CLIP 인코딩`

- **비용**: `O(S_final × k)`. `S_final` = 최종 레벨 슈퍼포인트 수(수백~수천), `k` = 슈퍼포인트당 뷰 수(3~5).
  - 예: 1,000 슈퍼포인트 × 4뷰 = **4,000회** vs 현재 2만~8만회 → **약 5~20× 절감**.
- **best view 선택**: 슈퍼포인트의 투영 면적/가시성/정면성 최대화. `render_point`가 주는 `weight`, `means2D`로 CLIP 없이 계산 가능.
- **인코딩 crop**: 그 뷰에서 슈퍼포인트가 투영된 영역의 bbox, 또는 그 영역과 가장 겹치는 SAM 마스크 1개(LangSplat식 crop 유지). 단일 crop 인코딩 → k뷰 평균.
- **레벨 한정**: 쿼리는 `test_lerf.py`에서 `level=[2,3]`만 쓴다. → **레벨 2,3 슈퍼포인트에만** CLIP을 돌리면 추가 절감.
- **주의**: 현재 `proj_gaussian_features_x`는 슈퍼포인트당 여러 SAM 마스크 피처를 overlap 가중 혼합한다. lazy 버전은 dominant 마스크(들)만 인코딩하므로 피처 의미가 약간 달라진다 → mIoU로 검증 필요.

### Tier 2 — 뷰 서브샘플링 / 키프레임 선택 (drop-in, 곱셈 효과)

모든 V뷰가 아니라 **커버리지 기반 대표 부분집합**만 전처리.
- pose 공간 farthest-point-sampling 또는 greedy set-cover로 "모든 가우시안(또는 슈퍼포인트)이 ≥k뷰에서 보이도록" 최소 뷰 선택.
- 최종 피처가 **멀티뷰 평균/투표**이므로 모든 뷰는 불필요하고 **각도 커버리지**만 충분하면 된다. 파이프라인이 이미 `merge_proj.py:244`의 상위 레벨에서 `[::3]`을 신뢰한다 — 이를 전처리로 일반화.
- **구현**: `image_encoding.py`의 `data_listi`(전체 이미지)를 선택 부분집합으로 교체. `merge_proj`도 같은 부분집합으로 `getTrainCameras()` 제한.
- **절감**: 서브샘플 비율에 선형. `[::3]`이면 3×, 커버리지 기반이면 2~5×. **위험**: 커버 부족 영역 피처 손실 → set-cover로 ≥k 보장하여 완화.

### Tier 3 — 입도 간 마스크 중복 제거 (이미지 내, CLIP 전)

4입도(default/s/m/l)는 심하게 겹친다. CLIP 전에:
- 4입도 마스크를 합쳐 더 공격적인 IoU-NMS → **유니크 crop만 1회 인코딩** → 그 피처를 4채널 seg_map의 해당 인덱스들에 매핑.
- **절감**: 겹침 비율에 따라 1.5~3×(s/m/l 공유 마스크 많음).
- **거의 무손실**: 동일한 4채널 seg_map과 인덱스별 피처를 그대로 산출, 중복 crop 재인코딩만 회피.

### Tier 4 — 어차피 버려질 마스크 사전 필터 (저위험)

proj 단계가 버리는 것: 슈퍼포인트-마스크 overlap<0.3, 가우시안 portion<0.1, 신뢰도 sim<0.85. 작은/슬리버/저안정 마스크는 어떤 피처도 안 됨.
- 면적 백분위 하위 또는 stability_score 하위 마스크는 CLIP 스킵.
- **절감**: 10~40%, 구조 변경 없음, 위험 낮음. mIoU 안 떨어지게 임계 튜닝.

### Tier 5 — 3D 마스크 캐시 (Tier 1의 온라인/점진 버전)

`(최근접-슈퍼포인트-id, 시야각 버킷)` 키 캐시. 새 2D 마스크가 가우시안 깊이로 재투영되어 **이미 비슷한 각도에서 인코딩된 3D 영역**에 닿으면 캐시 재사용, 아니면 인코딩.
- eager 전체 인코딩을 "새로운 (영역,각도) 조합만 인코딩"으로 전환. Tier 1을 파이프라인 재배치 없이 달성. 구현 복잡도는 더 높음.

### Tier 6 — (참고) 호출당 비용 절감 ≠ 호출 횟수 절감

작은/빠른 CLIP, fp16(이미 적용), 입도 일괄 배치, crop 해상도↓. **호출 수가 아니라 호출당 비용**을 줄임. 사용자가 요청한 "덜 호출"과는 다르지만 시간 단축엔 보완적.

---

## 3. 권장 조합

**1단계(저위험·즉시): Tier 2 + Tier 3 + Tier 4** — 모두 `image_encoding.py` drop-in이고 곱셈으로 쌓여 정확도 위험 최소로 **5~10× 절감** 기대.

**2단계(재설계·최대효과): Tier 1** — 비용을 마스크 수가 아닌 **라벨이 필요한 3D 영역 수**에 묶는 원리적 해법. 기하 파이프라인을 CLIP 없이 먼저 돌리고(이미 가능: `graph_weight`/병합이 CLIP 미사용), 슈퍼포인트별 best-view 단일 crop만 인코딩. 쿼리에 쓰는 레벨(2,3)에만 적용.

---

## 3-bis. Tier 1 적용 시 전처리 흐름 변화

**한 줄 요약**: CLIP 인코딩이 *전처리 맨 앞(마스크 단위·전체 뷰, eager)* → *merge_proj 맨 뒤(슈퍼포인트 단위·best-k 뷰, lazy)* 로 이동. `_f.npy`가 사라지고 전처리는 `_s.npy`(SAM seg_map)만 생성.

### 현재 (eager CLIP)
```
[전처리] image_encoding.py (뷰마다)
   SAM 4입도 마스크 → 마스크 crop → CLIP encode_image → _f.npy (M×512)  ← CLIP 전부 여기
                                  → seg_map           → _s.npy (4×H×W)
[Step1-3] 기하 (SAM seg_map만, CLIP 미사용) → nag-l1.pt
[Step4]  merge_proj: region growing(SAM만) + proj_gaussian_features(_f.npy 읽기) → sai_nag.pt
```

### Tier 1 (lazy CLIP)
```
[전처리A · 경량]  SAM만 (뷰마다)
   SAM 마스크 → seg_map → _s.npy        ← CLIP 인코딩 없음, _f.npy 생성 안 함
[Step1-3] 기하 (불변, SAM seg_map만) → nag-l1.pt
[Step4-a] merge_proj region growing (불변, SAM만) → 최종 NAG 슈퍼포인트 레벨 확정
[전처리B · 지연 CLIP, 신규]  슈퍼포인트 확정 후, 쿼리 레벨(2,3)의 각 슈퍼포인트마다:
   1) best-view top-k 선택        (render_point weight/portion으로, CLIP 없이)
   2) 그 뷰에서 SP와 가장 겹치는 SAM 마스크 crop
   3) CLIP encode_image           ← CLIP은 여기서만, 총 S×k회
   4) k뷰 평균 → SP 512-d  → nag_feat
→ sai_nag.pt
```

### 파일별 변경점
| 파일 | 현재 | Tier 1 |
|---|---|---|
| `image_encoding.py` | SAM+CLIP, `_s.npy`+`_f.npy` | **CLIP 제거**, SAM seg_map만 → `_s.npy` |
| `dataset_readers.py` | `semantic={seg_map,fg_mask,sem}` 로드 | `sem`(=`_f.npy`) 로드 불필요, seg_map/fg_mask만 |
| `merge_proj.py` | `proj_gaussian_features*`가 `_f.npy` 읽기 | **CLIP 모델 로드 + lazy 인코딩**으로 재작성, 원본 RGB 필요 |
| 산출물 | `_f.npy` (모든 마스크) | `_f.npy` 없음 (피처는 sai_nag.pt에만) |

**비용**: `O(Σ_뷰 마스크수)`(eager) → `O(S_쿼리레벨 × k)`(lazy). 기하 파이프라인은 원래 CLIP-free라 무변경.

## 4. 검증 방법

- 측정 지표: **CLIP 호출 횟수** + **전처리 wall-clock** + **mIoU/mAcc**(`scripts/eval_seg.py`, LERF: figurines/ramen/teatime/waldo_kitchen).
- `scripts/encode_fastsam_clip.py`는 이미 타이밍을 CSV로 로깅 → 같은 방식으로 호출 카운터 추가.
- A/B: 기준(전체) vs 각 Tier 적용본의 (호출수, 시간, mIoU) 표.
- 합격선: mIoU 하락 ≤ ~1%p 이내에서 호출수 최대 절감 지점 채택.

---

## 5. Semantic 추출·적용 원리 (코드 기반 데이터 흐름)

adaptive-k 설계의 토대. SAM 마스크 → CLIP → 가우시안 → 슈퍼포인트 → 쿼리로 의미가 흐르는 경로를 정확히 추적한다.

```
[1] SAM 마스크          image_encoding.py:194  get_seg_img
      mask['segmentation'] 영역만 남기고 bbox crop → pad → 224² 
[2] CLIP per-mask       image_encoding.py:188  model.encode_image(tiles) → 512-d, L2정규화
      _f.npy = (M_masks, 512)   ← CLIP 호출은 오직 여기
      _s.npy = (4, H, W) seg_map: 픽셀 → _f.npy 마스크 인덱스(-1=배경)
[3] 가우시안 투영       merge_proj.py:64  render_point(view) → weight(가우시안 기여도), means2D(투영 픽셀)
      gau_mask = weight > 0.01                         (이 뷰에서 보이는 가우시안)
      gt_batch_seg = seg_map[y,x]                      (각 가우시안이 떨어진 SAM 마스크)
      seen_gau_sem = significance · fg · clip[seg]     (가우시안별: 자기 마스크 CLIP × 렌더 기여도)
[4] 슈퍼포인트 집계     merge_proj.py:83-88
      portion[s,v] = (이 뷰에서 보이는 s의 가우시안 수) / (s 전체 가우시안 수)   ← 가시 비율
      if portion < 0.1: skip
      sp_feature[s] += normalize(Σ_{g∈s} seen_gau_sem[g]) · portion             ← portion 가중
      (모든 뷰 누적 후 최종 normalize)
[5] 쿼리                test_lerf.py  level=[2,3]
      text CLIP ↔ sp_feature 코사인 → top-k 슈퍼포인트 → 가우시안 렌더 → IoU
```

### 핵심 사실 (adaptive-k가 의존하는 불변식)
1. **(슈퍼포인트, 뷰) 단위 view-feature** = `normalize(Σ_{보이는 멤버 g} sig_g · clip[mask(g)])`. 뷰 간 결합은 **`portion[s,v]` 가중합**.
   → 즉 최종 SP 피처는 이미 **portion-가중 멀티뷰 평균**이다. 잘 안 보이는 뷰는 자동으로 기여가 작다.
2. **필요한 CLIP은 "선택된 (s,v)가 참조하는 마스크"뿐**. 뷰를 솎으면 참조 마스크 집합이 줄고, 그만큼만 인코딩하면 된다.
3. **crop은 `_s.npy`만으로 복원 가능** — `region = (seg_map[level]==idx)`, 그 bbox로 원본 RGB를 잘라 224² 인코딩. SAM 마스크 객체를 따로 보관할 필요 없음.
4. `portion[s,v]`, dominant-mask, purity는 **전부 `render_point`+`_s.npy`로 계산** → CLIP 없이 best-view 점수화 가능.

---

## 6. Adaptive-k 알고리즘 상세 설계

### 6.1 설계 원칙 (왜 고정 k가 아니라 adaptive인가)
이전 논의 결론: **선택(selection)과 평균(averaging)은 서로 다른 노이즈를 잡는다.**
- 선택 = 뷰 *사이*의 오염(mask bleeding·폐색·oblique) 제거 → best-view 점수 `q`로 처리. k와 무관.
- 평균 = good 뷰 *내부*의 잔여 분산(조명·프레이밍·CLIP 시점 노이즈) 제거 → `k≥2`로 처리.
- 큰/멀티파트 SP의 공간 커버리지 → set-cover로 처리 (한 뷰가 전체를 못 담음).

→ SP마다 "얼마나 크고/깨끗하고/잘 보이는가"가 다르므로 k도 SP별로 적응해야 한다.

### 6.2 CLIP-free 사전 신호 (geometry 1-pass, 뷰마다 render_point 1회)
각 (슈퍼포인트 s, 뷰 v)에 대해:
| 신호 | 정의 | 의미 |
|---|---|---|
| `portion[s,v]` | 보이는 멤버 수 / 전체 멤버 수 | 가시 비율(coverage) |
| `conf[s,v]`    | 보이는 멤버의 significance(weight) 평균 | 정면성/근접도(품질) |
| `purity[s,v]`  | dominant 마스크의 significance 점유율 | 단일객체성(청결도). 낮음=마스크 straddle |
| `vis_set[s,v]` | 보이는 멤버 가우시안 집합 | set-cover용 |
| `mask(s,v)`    | significance 가중 최빈 SAM 마스크 인덱스 | 인코딩할 crop |

전부 `_s.npy`(seg_map)와 `weight`/`means2D`로 계산. CLIP 0회.

### 6.3 뷰 품질 점수
```
q[s,v] = portion[s,v] · conf[s,v] · purity[s,v]^β      (β=1 기본; 클수록 straddle 뷰 강하게 배제)
```

### 6.4 선택 규칙 (SP s, 레벨 ℓ)
```
C = { v : portion[s,v] ≥ p_min }                  # p_min=0.1 (기존 필터 그대로)
if C == ∅: return [ argmax_v portion[s,v] ]        # 폴백 1뷰

covered = ∅;  selected = []
while True:
    # 커버리지 가중 greedy: 새로 덮는 멤버가 많고 q 높은 뷰 우선
    v* = argmax_{v∈C\selected}  q[s,v] · |vis_set[s,v] \ covered|
    selected += [v*];  covered ∪= vis_set[s,v*]

    cover_ratio = |covered| / |s|
    # 종료: k_min 충족 + 커버리지 충족
    if len(selected) ≥ k_min(s) and cover_ratio ≥ τ_cov(ℓ): break
    if len(selected) ≥ k_max(ℓ):                              break
    if selected == C:                                         break

# 모호성 bump: 선택 뷰들이 계속 마스크를 straddle하면(평균 신뢰 낮음) 뷰 추가
if mean(purity over selected) < δ_pure and len(selected) < k_max(ℓ):
    add next highest-q views until purity 안정 or k_max
```

**k_min(s) (평균 보험):**
- 기본 `k_min = 2`.
- **예외(사용자 직관 반영)**: top 뷰가 `portion ≥ 0.95` *and* `purity ≥ 0.9` → 작고 완전히·깨끗하게 보이는 SP → `k_min = 1` 허용 (낭비 안 함).

**레벨별 캡 (쿼리는 level 2,3):**
| 레벨 | τ_cov | k_max | 비고 |
|---|---|---|---|
| 1 (작은 SP) | 0.70 | 3 | k=1~2로 충분 |
| 2 | 0.85 | 6 | 쿼리 레벨, 넉넉히 |
| 3 (큰 SP) | 0.90 | 8 | 멀티파트, 가장 넉넉히 |

### 6.5 두 단계 실행
```
[Pass 1 · geometry, CLIP 0회]
  for v in views: render_point(v) → 6.2 신호 누적 (sparse 테이블)
  for s in 쿼리레벨 SP: 6.4로 selected_views[s], referenced_masks[s] 결정

[Pass 2 · lazy CLIP]
  to_encode = ∪_s ∪_{v∈selected[s]} { masks(s,v) : significance share ≥ ε }   # ε=0이면 정확히 baseline 부분집합
  for (v, mask_idx) in dedup(to_encode):                                       # 뷰·마스크 공유시 1회만
      crop = bbox(seg_map[ℓ]==mask_idx)에서 원본 RGB → 224²
      clip_feat[(v,mask_idx)] = CLIP.encode_image(crop)
  # 기존 집계식 그대로, 단 뷰 루프를 selected[s]로 제한
  sp_feature[s] = normalize( Σ_{v∈selected[s]} portion[s,v]·normalize(Σ_{g∈s,v} sig_g·clip_feat[mask(g)]) )
```

### 6.6 baseline 환원 보장 (핵심)
`k_max=∞, τ_cov=1, p_min=0.1, ε=0, 모든 레벨 활성` 으로 두면 **현재 `proj_gaussian_features`와 수식이 완전히 동일**.
→ adaptive-k는 baseline의 **strict generalization**. 최악의 경우 = baseline. A/B 노브가 깨끗하다.

### 6.7 파라미터 요약 (sweep 대상)
| 파라미터 | 기본 | 역할 |
|---|---|---|
| `k_min` | 2 (조건부 1) | 평균 보험 하한 |
| `k_max(ℓ)` | 3/6/8 | 비용 상한 |
| `τ_cov(ℓ)` | .70/.85/.90 | 공간 커버리지 목표 |
| `p_min` | 0.1 | 후보 뷰 최소 가시비(기존값) |
| `β` | 1.0 | straddle 뷰 배제 강도 |
| `δ_pure` | 0.6 | 모호성 bump 발동 임계 |
| `ε` | 0.05 | sliver 마스크 인코딩 컷(0=무손실) |
| `QUERY_LEVELS` | {2,3} | CLIP 돌릴 레벨(저레벨 스킵=추가 절감) |

### 6.8 ramen sweep 계획
`k ∈ {1, 3, 5, all(baseline)}` + `adaptive` 5-way로 (CLIP 호출수, wall-clock, mIoU) 곡선.
- 곡선이 평평하면 작은 k 채택, 큰 SP에서만 꺾이면 adaptive의 이득 확인.
- 합격선: mIoU 하락 ≤ 1%p 내 최대 절감점.
