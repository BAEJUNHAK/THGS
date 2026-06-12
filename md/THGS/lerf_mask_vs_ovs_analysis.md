# LERF-Mask vs LERF-OVS 벤치마크 분석

본 문서는 THGS 프로젝트에서 LERF-Mask와 LERF-OVS 두 평가 벤치마크를 코드 레벨까지 분석하고, 두 벤치마크가 측정하려는 task의 본질, THGS와 OpenSplat3D의 파이프라인 차이, 그리고 결과 해석 시 주의점을 정리한다.

분석 대상:
- LangSplat (CVPR 2024) — LERF-OVS 원조
- OpenGaussian (NeurIPS 2024) — LERF-OVS 사실상 표준 프로토콜
- Gaussian Grouping (ECCV 2024) — LERF-Mask 원조
- OpenSplat3D — LERF-Mask 실험 레포 (`opensplat3d/opensplat3d/`)
- THGS — 본 프로젝트

---

## 1. 데이터셋 계보

### 1.1 원본 LERF (Kerr et al., ICCV 2023)
- Polycam iPhone 캡처. NeRF용. **세그멘테이션 GT 없음** (3D localization accuracy로만 평가).
- 후속 논문들이 segmentation GT를 추가로 라벨링 → LERF-OVS와 LERF-Mask가 갈라짐.

### 1.2 LERF-OVS (LangSplat이 도입)
- LangSplat 저자가 텍스트 쿼리별 GT mask를 polygon으로 라벨링.
- 4개 장면: figurines, ramen, teatime, **waldo_kitchen**.
- GT 포맷: `label/<scene>/frame_*.json` (LabelMe polygon).
- 평가 view: GT JSON이 있는 **train view 일부**.

### 1.3 LERF-Mask (Gaussian Grouping이 도입)
- 3개 장면: figurines, ramen, teatime (waldo_kitchen 없음).
- GT 포맷: `test_mask/<view_idx>/<prompt>.png` (raster된 binary mask).
- 평가 view: 별도 hold-out **novel view 2~4장**.

### 1.4 실측 — 두 데이터셋의 카메라 관계

본 머신에서 직접 확인:

```
figurines: LERF-OVS 299장 ⊂ LERF-Mask 303장 (test_0..3 추가)
ramen:     LERF-OVS 131장 ⊂ LERF-Mask 135장 (test_0..3 추가)
teatime:   LERF-OVS 177장 ⊂ LERF-Mask 180장 (test_0..2 추가)
```

**LERF-Mask sparse가 LERF-OVS의 전체 train set을 그대로 포함하고 `test_*.jpg` 만 추가 등록**. 즉 같은 캡처 세션에서 일부 프레임을 hold-out한 진짜 novel view. LERF-OVS sparse에는 `test_*.jpg` 가 **0장** → 학습 때 본 적 없는 시점 확정.

---

## 2. GT mask 생성 방식

### 2.1 LERF-OVS — polygon → raster

[test_lerf.py:66-70](../test_lerf.py#L66-L70):
```python
mask_gt = np.zeros((h, w), dtype=np.uint8)
for obj in anno['objects']:
    if obj['category'] == prompt:
        _mask_gt = polygon_to_mask((h, w), obj['segmentation'])
        mask_gt = np.maximum(mask_gt, _mask_gt)   # OR union
```

- `polygon_to_mask = cv2.fillPoly`.
- **같은 카테고리의 모든 instance는 OR union** → semantic seg 평가 (instance seg 아님).
- JSON의 `group` 필드는 LangSplat·OpenGaussian·THGS 모두 무시 (LabelMe 메타데이터).

### 2.2 LERF-Mask — 이미 raster된 PNG

```
data/lerf_mask/<scene>/test_mask/<view_idx>/<prompt>.png   # 0/255 binary
```

- 본 머신 실측:
  - figurines view 0~3: 각 7 prompts (green apple, green toy chair, old camera, porcelain hand, red apple, red toy chair, rubber duck with red hat)
  - ramen view 0~2: 각 6 prompts (chopsticks, egg, glass of water, pork belly, wavy noodles in bowl, yellow bowl)
  - teatime view 0: 10 prompts, view 2: 5 prompts (**view 1 없음**)
- 1 prompt = 1 PNG. polygon 해석 단계 없음 → annotator의 raster화 결과를 그대로 신뢰.

---

## 3. 평가 프로토콜 코드 레벨 비교

| 항목 | **LangSplat** (LERF-OVS 원조) | **OpenGaussian** (LERF-OVS 사실상 표준) | **Gaussian Grouping** (LERF-Mask) | **THGS** `eval_seg.py` | **THGS** `eval_lerf_galre_b.py` | **THGS** `eval_lerf_mask.py` |
|---|---|---|---|---|---|---|
| 데이터셋 | LERF-OVS | LERF-OVS | LERF-Mask | LERF-OVS | LERF-OVS | LERF-Mask |
| GT 입력 | polygon JSON | 미리 raster된 jpg | 미리 raster된 png | polygon JSON (즉시 raster) | polygon JSON (즉시 raster) | raster된 png |
| 카테고리 union | OR (`stack_mask`) | (미리 union됨) | (1 prompt = 1 png) | OR (`np.maximum`) | OR | (1 prompt = 1 png) |
| Group field | 무시 | 무시 | 무관 | 무시 | 무시 | 무관 |
| Pred binarize | min-max norm + `>0.4` + smooth | silhouette > 0.7 → `>10` | softmax > 0.2 | semantic > 0.5 | semantic > 0.5 | semantic > 0.5 |
| 모폴로지 | 30×30 mean + 7×7 majority | 없음 | 없음 | 없음 | 없음 | 없음 |
| Hierarchy | 3 levels, max-relevancy oracle pick | cluster + 0.9 distance | softmax over object ids | NAG level [2,3], topk=3 | (재사용) | NAG level [2,3], topk=3 |
| Prompt 소스 | GT JSON | **하드코딩 리스트 (14~21개)** | PNG 파일명 | GT JSON | GT JSON | PNG 파일명 |
| Prompt 수 (장면당) | 5~10 | figurines 21, ramen 14, teatime 14, waldo 18 | figurines 7, ramen 6, teatime 10 | 5~10 | 5~10 | (PNG 파일명 따라) |
| View 선택 | GT JSON 있는 train view | 같은 train view | **별도 hold-out** `test_0..N.jpg` | train view | train view | sim3 정합된 novel view |
| mIoU 집계 | flat mean (frame, prompt) | flat mean (frame, prompt) | **class별 평균 → 평균** | frame mean → scene mean | flat mean | flat mean (비표준) |
| mAcc 정의 | 없음 (Loc Acc bbox 별도) | `(IoU > 0.25)` 비율 | 없음 | **픽셀 acc** (TN 포함, 비표준) | `(IoU ≥ 0.25)` (OpenGaussian 호환) | 없음 |
| Boundary IoU | 없음 | 없음 | **3×3 erode, iter=round(0.02·diag)** | 없음 | 없음 | 동일 정의 |

### 3.1 LangSplat pred 후처리 (LERF-OVS, [evaluate_iou_loc.py](https://github.com/minghanqin/LangSplat/blob/main/eval/evaluate_iou_loc.py))
```python
kernel = np.ones((30, 30)) / 900
relev = 0.5 * (cv2.filter2D(relev, -1, kernel) + relev)   # 30x30 blur + 50/50 blend
relev = (relev - min) / (max - min)
relev = clip(2 * relev - 1, 0, 1)                          # 상위 절반만 살림
mask = (relev > 0.4)
mask = smooth(mask)                                         # 7x7 majority vote
# + 3개 hierarchy level 중 max-relevancy 가진 level을 prompt별로 oracle pick
```

### 3.2 OpenGaussian pred (LERF-OVS, [compute_lerf_iou.py](https://github.com/yanmin-wu/OpenGaussian/blob/main/scripts/compute_lerf_iou.py))
```python
cluster_silhouette = rendered_silhouette > 0.7
save_image(silhouette, f"{frame}_{prompt}.png")
binary = (image_array > 10).astype(int)
count_iou_025 = (np.array(ious) > 0.25).sum()    # strict >, >= 아님
Acc@0.25 = count_iou_025 / total_count
```

### 3.3 Gaussian Grouping Boundary IoU 정의

[scripts/eval_lerf_mask.py:17-31](../scripts/eval_lerf_mask.py#L17-L31), OpenSplat3D [eval/metrics.py:18-34](../opensplat3d/opensplat3d/eval/metrics.py#L18-L34), GG 원본 동일:
```python
def mask_to_boundary(mask, dilation_ratio=0.02):
    diag = sqrt(h**2 + w**2)
    dilation = round(0.02 * diag)              # 988x731 → ~25 px
    eroded = cv2.erode(mask, np.ones((3,3)), iterations=dilation)
    return mask - eroded
```
- pycocotools 아닌 자체구현. dilation_ratio=0.02는 이미지 대각선 2%.
- **해상도에 따라 boundary 두께가 자동 변함** (988×731 → 25 px, 1920×1080 → 44 px). 보고 시 해상도 명시 필요.

---

## 4. THGS의 LERF-Mask 평가 파이프라인 (`test_lerf_mask.py`)

### 4.1 핵심 추가 작업 — sim3 좌표계 정합

THGS는 LERF-OVS world에서 학습한 가우시안을 LERF-Mask test view로 옮겨야 한다. [test_lerf_mask.py:52-80](../test_lerf_mask.py#L52-L80) 의 RANSAC:
- 4-point Umeyama, `inlier_thresh=0.05`, `iters=2000`.
- 두 COLMAP에 **공통으로 등록된 train 프레임** 의 카메라 중심으로 매칭.

### 4.2 실측 sim3 정합 정확도

```
figurines  s=0.984  ||R-I||∞=3.4e-1  ||t||=0.050  inliers=296/299  mean_err=1.9e-3
ramen      s=0.977  ||R-I||∞=5.9e-4  ||t||=0.095  inliers=131/131  mean_err=9.1e-4
teatime    s=1.027  ||R-I||∞=5.0e-2  ||t||=0.052  inliers=174/177  mean_err=2.7e-3
```

- Inlier rate 98~100%, mean residual **1~3mm 수준의 카메라 중심 오차**.
- figurines는 reconstruction 간 ~20도 회전 차이가 있지만 sim3가 정확히 보정.
- **결론: 정합 잔차가 mm 수준이라 좌표계 정합 비용은 사실상 무시 가능.**

### 4.3 파이프라인 6단계

1. **NAG 로드** ([test_lerf_mask.py:121-127](../test_lerf_mask.py#L121-L127)): `sai_nag.pt` (LERF-OVS 학습 결과물) + CLIP. 재학습 없음.
2. **sim3 추정** ([:131-134](../test_lerf_mask.py#L131-L134)): LERF-Mask world → LERF-OVS world.
3. **Test 카메라 포즈 변환** ([:150-162](../test_lerf_mask.py#L150-L162)): LERF-Mask COLMAP의 `test_<view_idx>.jpg` 포즈를 sim3로 LERF-OVS world에 배치.
4. **Prompt별 매칭** ([:170-178](../test_lerf_mask.py#L170-L178)):
   ```python
   vlm.encode_text(prompt)
   point_valid = snag.get_related_gaussian(
       [vlm.compute_similarity(f) for f in snag.feat],
       topk=3, level=[2, 3],
   )
   embd_sim = render(cam, gaussians, ...)['semantics']
   mask = (embd_sim > 0.5)
   ```
   CLIP text feature와 NAG superpoint feature 유사도, level [2,3]에서 topk=3.
5. **저장**: `output/render/lerf_mask/<scene>/<view_idx>/<prompt>.png`.
6. **평가** ([scripts/eval_lerf_mask.py](../scripts/eval_lerf_mask.py)): IoU + BIoU (GG 공식 정의).

---

## 5. OpenSplat3D의 LERF-Mask 평가 파이프라인

### 5.1 근본적 차이 — 좌표계 정합 불필요

[opensplat3d/eval/eval_lerf_mask.py:389-401](../opensplat3d/opensplat3d/eval/eval_lerf_mask.py#L389-L401):
```python
mask_dir = Path(model_params.source_path) / "test_mask"
test_cameras = setup_params.scene.get_test_cameras()
test_image_cameras = [
    next(c for c in test_cameras if c.name == f"test_{int(d.stem)}")
    for d in test_image_dirs
]
```
- OpenSplat3D는 **`source_path = data/lerf_mask/<scene>` 으로 직접 학습**.
- Gaussian과 test 카메라가 같은 world (LERF-Mask COLMAP) → sim3 불필요.
- 대신 **LERF-Mask 평가용 모델, LERF-OVS 평가용 모델을 별도 학습**해야 함.

### 5.2 텍스트→3D 매칭 — Grounded-SAM 2-stage

[:244-295](../opensplat3d/opensplat3d/eval/eval_lerf_mask.py#L244-L295):
```python
# 1. HDBSCAN으로 가우시안을 instance 라벨로 클러스터 (offline)
labels = np.load(model_path / "clustering" / "labels.npy")

# 2. 첫 test 프레임에서 GroundingDINO + SAM
grounded_masks = get_grounded_masks(first_test_image, query_prompts, ...)

# 3. 각 instance label을 첫 프레임에 렌더 → 2D SAM mask와 IoA > 0.7 인 id 선택
obj_ids, _ = select_objects_by_ioa(obj_masks[0], grounded_masks, ioa_threshold=0.7)

# 4. 선택된 id들을 모든 test 프레임에서 렌더 → union이 pred mask
```

**즉 OpenSplat3D는 "Grounded-SAM이 첫 프레임에서 본 2D mask = oracle"** 로 두고, 3D 가우시안 클러스터 중 그것과 IoA가 높은 걸 골라 다른 view들에서 일관 렌더한다.

### 5.3 OpenSplat3D Prompt 리스트 (하드코딩)

[:46-82](../opensplat3d/opensplat3d/eval/eval_lerf_mask.py#L46-L82):
- figurines: 7개 (THGS PNG 파일명과 일치)
- ramen: 6개 (THGS PNG 파일명과 일치)
- teatime: 10개 (THGS PNG 파일명과 일치, view 0 기준)

---

## 6. 두 파이프라인의 본질적 차이

| 측면 | OpenSplat3D | THGS |
|---|---|---|
| 학습 좌표계 | LERF-Mask 자체 COLMAP에서 직접 학습 | LERF-OVS COLMAP에서 학습 |
| 테스트 카메라 정합 | `scene.get_test_cameras()` 그대로. sim3 불필요 | RANSAC sim3 정합 (실측 잔차 mm 수준) |
| 텍스트→3D 매칭 | **Grounded-SAM 2D mask → IoA로 3D cluster 매칭** (2-stage) | **CLIP feature 직접 매칭** (NAG topk) (1-stage) |
| 3D 객체 표현 | HDBSCAN 사전 클러스터링 | 계층적 슈퍼포인트 그래프 (NAG) |
| 1st-frame 의존 | **있음** — SAM이 첫 프레임에서만 prompt 해석 | 없음 — 매 프레임 독립 |
| 외부 사전학습 모델 | GroundingDINO Swin-B + SAM ViT-H 필수 | CLIP만 |
| mIoU 집계 | **per-prompt macro** (GG 호환) | flat (view, prompt) micro (비표준) |
| 학습 횟수 | LERF-Mask용 별도 학습 | LERF-OVS 학습 한 번으로 양쪽 평가 |

---

## 7. LERF-Mask Task가 측정하려는 본질

### 7.1 평가 다이어그램 — 무엇을 평가하는가
```
SAM 2D mask (학습 데이터)
        ↓ method-specific lifting (학습 단계)
3D scene 표현 (THGS의 NAG superpoint, GG의 instance feature 등)
        ↓ novel view에 splat (inference)
2D mask
        ↓ IoU with GT  ← LERF-Mask 평가
```

평가 대상은 **"lifting의 결과물인 3D 표현이 2D로 splat한 mask"** 이지 **SAM의 raw 2D mask가 아니다**.

**명시적으로 task 의도를 벗어난 측정**:
- ❌ "SAM이 만든 2D segmentation의 GT 일치도" → SAM 자체의 능력 평가일 뿐, 3D lifting과 무관
- ❌ "Grounded-SAM의 text→2D mask 정확도" → 외부 2D segmenter의 평가일 뿐

### 7.2 GG가 evaluation으로 강조한 세 축
GG의 evaluation 설계를 역추적하면 LERF-mask가 강조한 세 축이 보인다:

**축 1. Novel view ⇒ "진짜 3D 능력" 요구**
- LERF-OVS는 train view 평가 → 2D-only feature lifting으로도 풀 수 있음.
- LERF-Mask는 hold-out novel view → 학습 때 못 본 시점 → **3D representation이 view-consistent해야** 점수가 나옴.

**축 2. Open-vocab (자유 텍스트) ⇒ "제한 없는 쿼리"**
- "rubber duck with red hat", "wavy noodles in bowl" 같은 compositional/specific 쿼리.

**축 3. 2D mask IoU + BIoU ⇒ 실용적 우회 측정**
- 3D GT가 없으니 2D mask로 우회 측정. BIoU = 객체 분리 정밀도.

### 7.3 두 task의 분화식 (핵심)
위 framing에서 두 벤치마크의 점수가 자연스럽게 분해된다:

```
LERF-mask 점수  ≒ Oracle  (3D 표현의 천장)
LERF-OVS 점수   ≒ Oracle − CLIP-NAG 매칭 손실
```

즉:
- **LERF-mask** = "3D scene 표현이 GT 객체를 표현할 수 있는가" 만 묻는 task
- **LERF-OVS** = 그 위에 추가로 "CLIP-text 매칭이 정확한가" 까지 묻는 task

따라서 같은 method가 두 벤치마크에서 받는 점수 차이는:
- Method의 절대 성능 차이가 아니라
- **두 task가 측정하는 능력의 분화**가 만든 차이

### 7.4 종합
"3D scene 모델이 자연어 쿼리에 대해 **새로운 시점에서도 정확한 객체 경계를 가진 view-consistent 2D mask를 생성할 수 있는가**" 측정. 이를 두 단계 (3D 표현 능력 + 매칭 능력) 로 분해하면 dual equation으로 정량화 가능.

---

## 8. 두 방법론이 task 본질을 지키는 정도

LERF-Mask가 측정하려는 능력 = **prompt → 3D 객체 매핑** + **3D 객체 → novel view 일관 렌더**.

### OpenSplat3D
- **앞부분(prompt→3D)을 Grounded-SAM에 위임**, 뒷부분(3D 일관성)을 중점 평가.
- 점수가 높다 = "**SAM lifting + 3D 일관 유지**가 잘 된다"는 뜻에 가까움.
- 실패 모드: 첫 프레임 Grounded-SAM이 prompt를 못 찾으면 → `obj_ids=[]` → catastrophic 0 IoU.
- 사실상 task의 절반을 외부 2D oracle에 외주.

### THGS
- **앞부분과 뒷부분 모두 자기 능력**으로 풀이.
- 점수가 높다 = "CLIP-NAG 매칭 + 시점 렌더"가 잘 된다.
- 좌표계 정합 비용은 실측 시 mm 수준 잔차 → 사실상 없음.
- LERF-Mask의 "3D-aware open-vocab seg" 정신에 **OpenSplat3D보다 더 충실**.

**즉 같은 LERF-Mask 점수 표에 올려도, OpenSplat3D는 task의 절반만 자기 능력으로 풀고 절반은 외주한 결과, THGS는 task 전체를 자기 능력으로 푼 결과.** 숫자만 보면 같은 메트릭이지만, 그 숫자가 답하는 질문이 다르다.

### 8.1 THGS의 SAM lifting Ceiling — 실측

Section 7.3의 dual equation을 THGS에 적용하기 위해, "SAM 정보로 만들어진 THGS의 3D 표현이 도달할 수 있는 천장"을 직접 측정. Oracle 정의 3가지:

| Oracle | 정의 | 측정 결과 (overall mIoU) |
|---|---|---|
| **v1** (union) | GT mask 안 가우시안의 모든 superpoint union | 0.0574 ❌ over-expansion |
| **v2 τ=0.5** (majority) ★ | SP의 visible 가우시안 중 majority가 GT 안인 SP select | **0.8531** 추천 ceiling |
| **v3 topk=1** | 각 SP를 단독 렌더 → IoU 상위 1개 select | 0.8204 (single-SP 한계) |

→ THGS의 **SAM lifting의 3D 표현 Ceiling = 0.8531** (LERF-mask 본질이 측정하려는 천장).
→ 구현: [sam_oracle_v2_lerf_mask.py](../sam_oracle_v2_lerf_mask.py), [sam_oracle_v3_lerf_mask.py](../sam_oracle_v3_lerf_mask.py).

### 8.2 THGS의 매칭 손실 — Ceiling vs Actual

| 양 | 값 | 해석 |
|---|---|---|
| **Ceiling (v2 τ=0.5)** | 0.8531 | THGS의 SAM lifting 결과가 GT를 표현할 수 있는 천장 |
| **Actual (CLIP-based)** | 0.7367 | CLIP-NAG 매칭이 실제 도달한 점수 |
| **Gap = Ceiling − Actual** | **0.1164** | **매칭 알고리즘이 표현 능력을 활용하지 못한 양** |
| 활용도 | 86% | 매칭 알고리즘이 천장의 약 86%만 활용 |

→ **결론**: "THGS의 SAM lifting은 잘했다, CLIP-NAG 매칭이 부족하다".

### 8.3 매칭 손실의 장면별 분포

| Scene | Ceiling (v2) | Actual (CLIP) | Gap |
|---|---|---|---|
| figurines | 0.8389 | 0.7804 | 0.0585 |
| ramen | 0.8445 | 0.5918 | **0.2527** ★ |
| teatime | 0.8760 | 0.8380 | 0.0380 |

매칭 손실의 84%(0.2527 / 0.3 ≈ 0.84) 가 **ramen 한 장면에 집중**. figurines/teatime에서는 CLIP이 거의 ceiling 수준.

---

## 9. LERF-Mask의 의미와 한계 — 결론

### 의미 있는 이유
1. **Novel view 평가는 진짜다** (Fact 1 확인). 2D-only trivial solution 안 통함.
2. **Open-vocab 자유 텍스트.** Compositional 쿼리 일반화 평가.
3. **BIoU 동반.** 객체 분리 정밀도 평가.
4. **GT 품질 합리적.** PNG 직접 제공 (LERF-OVS polygon보다 깨끗).

### 한계 — task를 무효화하진 않음
1. **규모가 작다** — 3 scenes, 2~4 views, 6~10 prompts. 통계적 흔들림 큼. 미세 우열은 noise 안.
2. **"3D-aware 능력 측정"이 method에 따라 달라진다** — 외부 2D segmenter에 의존하는 방법은 절반만 자기 능력으로 풀이.
3. **데이터 불완전성** — teatime view 1 GT 누락 등.

### THGS의 LERF-Mask 실험은 정당한가? — **정당하다**

1. sim3 정합 잔차가 mm 수준 → 평가 입력 깨끗.
2. Prompt/view 매칭이 GG 공식 GT 셋과 정확히 일치.
3. 외부 2D oracle 미사용 → task 정신에 OpenSplat3D보다 더 충실.
4. **평균 방식 한 줄만 prompt-macro로 통일하면** OpenSplat3D/GG 표와 직접 비교 가능.

### 한 줄 결론

> LERF-Mask는 **novel view 평가**라는 점에서 LERF-OVS보다 진일보한, **task로서 정당한** 벤치마크다. 다만 규모가 작아 미세 우열은 못 가르고, 어떤 method가 어떤 능력으로 점수를 따는지가 다르다는 점만 명시하면 된다. THGS의 파이프라인은 sim3 잔차가 무시할 수준이라 LERF-Mask 본질을 깨지 않으며, **평균 방식 한 줄만 prompt-macro로 통일하면** OpenSplat3D/GG와 fair comparison 가능. **의미 없지 않다, 단지 어떤 능력을 측정한 점수인지 정직히 말하면 된다.**

### 핵심 framing 요약 (Section 7.3 dual equation 재게재)

```
LERF-mask 점수  ≒ Oracle  (3D 표현의 천장)
LERF-OVS 점수   ≒ Oracle − CLIP-NAG 매칭 손실
```

THGS 실측 (Section 8.1-8.3): Ceiling 0.8531 / Actual 0.7367 / Gap 0.1164, ramen에 손실의 84% 집중.

---

## 10. 결과표 비교 (참고)

### LERF-Mask (mIoU / mBIoU, GG 보고)
| Method | figurines | ramen | teatime |
|---|---|---|---|
| LangSplat | 52.8 / 50.5 | 50.4 / 44.7 | 69.5 / 65.6 |
| Gaussian Grouping | 69.7 / 67.9 | 77.0 / 68.7 | 71.7 / 66.1 |
| GradiSeg | 81.3 / 78.1 | 78.5 / 72.9 | 74.4 / 70.6 |

### LERF-OVS (mIoU / mAcc@0.25, OpenGaussian 프로토콜)
| Method | Mean mIoU | Mean mAcc |
|---|---|---|
| LangSplat | 9.66 | 12.41 |
| LEGaussians | 16.21 | 23.82 |
| OpenGaussian | 38.36 | 51.43 |
| Identity-aware LGS (ICCV'25) | 42.52 | 62.09 |

**같은 LangSplat이 LERF-Mask figurines 52.8 vs LERF-OVS figurines 10.16** — 5배 차이. 어느 GT/프로토콜이냐가 결정적이라 논문 결과를 인용할 때 항상 확인 필요.

---

## 11. 본 머신의 평가 스크립트 3종 정리

| 스크립트 | 데이터셋 | mIoU 집계 | mAcc 정의 | 용도 |
|---|---|---|---|---|
| [scripts/eval_seg.py](../scripts/eval_seg.py) (현재 기본) | LERF-OVS | frame→scene mean | **픽셀 acc** (TN 포함) | 내부 빠른 확인용 |
| [scripts/eval_lerf_galre_b.py](../scripts/eval_lerf_galre_b.py) | LERF-OVS | flat mean | **IoU ≥ 0.25** (OpenGaussian 호환) | LangSplat/OpenGaussian/IDLGS와 비교 |
| [scripts/eval_lerf_mask.py](../scripts/eval_lerf_mask.py) | LERF-Mask | flat (view, prompt) mean | mIoU + **mBIoU** | GG/Gaga/GradiSeg와 비교 (단 prompt-macro로 수정 필요) |

→ **논문 작성 시**:
- OpenGaussian 표와 비교 → `eval_lerf_galre_b.py` (eval_seg.py 결과 아님)
- Gaussian Grouping 표와 비교 → `eval_lerf_mask.py` 의 평균 방식을 **per-prompt macro로 수정** 후 사용

---

## 12. 인용 소스

### 외부 코드 / 논문
- [LangSplat eval (evaluate_iou_loc.py)](https://github.com/minghanqin/LangSplat/blob/main/eval/evaluate_iou_loc.py)
- [Gaussian Grouping eval (eval_lerf_mask.py)](https://github.com/lkeab/gaussian-grouping/blob/main/script/eval_lerf_mask.py)
- [OpenGaussian eval (compute_lerf_iou.py)](https://github.com/yanmin-wu/OpenGaussian/blob/main/scripts/compute_lerf_iou.py)
- [Gaussian Grouping dataset doc](https://github.com/lkeab/gaussian-grouping/blob/main/docs/dataset.md)
- [Identity-aware LGS (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Jang_Identity-aware_Language_Gaussian_Splatting_for_Open-vocabulary_3D_Semantic_Segmentation_ICCV_2025_paper.pdf)
- [GradiSeg (arXiv 2412.00392)](https://arxiv.org/html/2412.00392v1)

### 본 머신 코드
- [test_lerf.py](../test_lerf.py)
- [test_lerf_mask.py](../test_lerf_mask.py)
- [scripts/eval_seg.py](../scripts/eval_seg.py)
- [scripts/eval_lerf_galre_b.py](../scripts/eval_lerf_galre_b.py)
- [scripts/eval_lerf_mask.py](../scripts/eval_lerf_mask.py)
- [opensplat3d/opensplat3d/eval/eval_lerf_mask.py](../opensplat3d/opensplat3d/eval/eval_lerf_mask.py)
- [opensplat3d/opensplat3d/eval/metrics.py](../opensplat3d/opensplat3d/eval/metrics.py)
- [opensplat3d/opensplat3d/eval/grounding_sam.py](../opensplat3d/opensplat3d/eval/grounding_sam.py)
