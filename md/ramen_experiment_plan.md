# Ramen 실험 계획 (THGS Limitation 검증)

> 이 md는 `q.md`의 참고 쿼리 구조를 **로컬 LERF-OVS GT 실측 결과**에 맞춰 재정리한 **우리 실험 전용 설계안**이다. `q.md`는 외부 참조이므로 그대로 두고, 이 파일을 프로젝트 기준으로 쓴다.

---

## 0. 실험 대상

- **Scene**: `ramen` (LERF-OVS)
- **Base**: 저자 pretrained 2DGS + 자체 생성한 THGS `sai_nag.pt`
- **평가 프레임 수**: 7 (`frame_00006, 00024, 00060, 00065, 00081, 00119, 00128`)
- **GT 소스**: `data/lerf-ovs/label/ramen/*.json` (LangSplat 공식판)

---

## 1. 실제 GT 카테고리 요약 (md/ramen_gt_categories.md 근거)

| Category | 전체 polygon | 프레임 커버 | q.md Level 0 포함 |
|---|---|---|---|
| `bowl` | 7 | 5/7 | ✓ |
| `chopsticks` | 7 | 7/7 | ✓ |
| `corn` | 5 | 5/7 | ✓ |
| `egg` | 7 | 7/7 | ✓ (q.md의 `eggs`도 같은 GT) |
| `glass of water` | 2 | 2/7 | ✓ |
| `hand` | 1 | 1/7 | ✓ |
| `kamaboko` | 7 | 7/7 | ✓ |
| `napkin` | 7 | 5/7 | ✗ (q.md 누락) |
| `nori` | 6 | 6/7 | ✓ |
| `onion segments` | 7 | 7/7 | ✗ (q.md 누락) |
| `plate` | 6 | 4/7 | ✓ |
| `sake cup` | 8 | 6/7 | ✓ |
| `spoon` | 3 | 2/7 | ✓ |
| `wavy noodles` | 7 | 7/7 | ✓ |

**결정**: 
- 우리 실험에서는 **q.md의 13개 Level 0 쿼리만 사용**한다 (`napkin`, `onion segments`는 q.md 설계 범위 밖).
- `eggs`는 GT가 `egg`와 동일하므로 **동일 GT 재사용**으로 처리 (CLIP의 단/복수 민감도 측정).

---

## 2. 6-Level 실험 쿼리 세트 (총 70개)

### Level 0 — Base Retrieval (13개, 정량)
GT 카테고리 단어 그대로.

```
bowl, chopsticks, corn, egg, eggs, glass of water, hand,
kamaboko, nori, plate, sake cup, spoon, wavy noodles
```

### Level 1 — 속성/어순 변화 (13개, 정량: L0 GT 재사용)
```
yellow bowl         → bowl
wooden chopsticks   → chopsticks
yellow corn         → corn
halved egg          → egg
two eggs            → eggs (=egg GT)
water glass         → glass of water
person hand         → hand
pink kamaboko       → kamaboko
dark nori           → nori
metal plate         → plate
metal cup           → sake cup
black spoon         → spoon
curly noodles       → wavy noodles
```

### Level 2 — 세부 파트 (14개, 정성만)
GT 없음. 시각화 + 수작업 라벨링.
```
bowl rim, chopstick tip, egg yolk, egg white, glass rim,
fingertips, kamaboko swirl, nori edge, plate rim, cup rim,
spoon handle, noodle strand, water, bottle
```

### Level 3 — 인스턴스 구분 (4개, 정성)
동일 카테고리 내 여러 polygon이 존재하는 프레임 기준. 
```
left chopstick, left egg, both eggs, right nori
```
**참고**: LERF-OVS ramen GT는 chopsticks/egg/nori가 프레임당 대체로 1 polygon → 자동 정량 불가. 정성 평가로만 의미. 대안: `left bowl/right bowl/left sake cup/right sake cup` (frame_00081, 00128은 bowl×2, sake cup×2 존재 → 엄격 instance 평가는 이것이 적합).

### Level 4 — 공간 관계 (13개, 정량: L0 GT 재사용 + 관계무시 overlap 측정)
```
bowl on plate               → bowl
chopsticks beside bowl      → chopsticks
corn near egg               → corn
egg above noodles           → egg
eggs above noodles          → eggs (=egg GT)
glass beside bowl           → glass of water
hand behind bowl            → hand
kamaboko below eggs         → kamaboko
nori beside eggs            → nori
plate under bowl            → plate
cup near chopsticks         → sake cup
spoon behind bowl           → spoon
noodles below eggs          → wavy noodles
```

### Level 5 — 설명문 (13개, 정량: L0 GT 재사용)
q.md의 `sentence` 그대로. 프롬프트 길이가 긺. `bowl` → `"An object containing noodles and various toppings on a table with sloping edges."` 등.

---

## 3. 평가 전략

### 정량 평가 (4 Level)
- **Level 0**: prompt → 3D 마스크 → render → `_pred.png`. GT polygon → `_gt.png`. IoU/Precision/Recall/F1/Acc (모두 `eval_seg.binary_mask_metrics` 사용).
- **Level 1**: 같은 GT로 L0 대비 **IoU drop**(= `IoU_L0 - IoU_L1`) 계산. 드롭이 크면 **속성 robustness 약함**.
- **Level 4**: 
  - L0 GT 대비 IoU (관계 쿼리가 올바른 주체 객체를 집는지)
  - **추가 핵심 지표**: `IoU(pred_L4, pred_L0)` — 둘의 예측 마스크가 얼마나 같은가. 크면(>0.85) **THGS가 공간관계를 무시**하고 주 명사만 본다는 증거.
- **Level 5**: L0 GT 대비 IoU. L0 대비 drop 측정. 문장형에서의 CLIP 성능.

### 정성 평가 (2 Level)
- **Level 2**: 파트 쿼리. 시각화 그리드에 색상별 overlay. 3단계 수동 점수(GOOD/PARTIAL/WRONG) 또는 4V auto-eval.
- **Level 3**: 인스턴스 분리. 시각화 + (왼쪽/오른쪽 예측이 **서로 다른지** 측정). `IoU(left_X_pred, right_X_pred) > 0.8`이면 instance 분리 실패.

---

## 4. 노트북 실행 순서

노트북 `THGS_Pipeline_ramen.ipynb`의 Cell 30~35에서 순차 실행:

1. **Cell 30 (md)**: 이 파일(`md/ramen_experiment_plan.md`) 요약 표시
2. **Cell 31**: 쿼리 리스트 정의 (70개, Level 0~5)
3. **Cell 32**: 배치 예측 실행
   - 입력: `sai_nag.pt`, `scene` 7 프레임 카메라
   - 각 (프레임, 쿼리) 쌍에 대해 CLIP text → similarity → topk=3 at level=[2,3] → render → 이진 마스크 PNG
   - 저장: `output/render/lerf_exp/ramen/<frame>/L<level>_<query>.png` + `L<level>_<query>_gt.png`
4. **Cell 33**: 정량 평가
   - L0, L1, L4 (L0 GT 대비), L5 대해 IoU/P/R/F1/Acc 계산
   - 추가로 L4의 pred↔L0 pred overlap IoU 계산
   - 결과 DataFrame → `output/render/lerf_exp/ramen_metrics.csv`
5. **Cell 34**: 정성 시각화
   - Level 0~5 각각 grid (행=쿼리, 열=7 프레임). `matplotlib.pyplot` subplot
   - 원본 이미지 + pred mask overlay + (L0/L1/L4/L5는 GT overlay도)
6. **Cell 35**: 종합 리포트
   - Level별 mIoU bar chart
   - L0 vs {L1, L4, L5} per-category IoU drop scatter
   - L4 관계 무시 scatter (x=`IoU(pred_L4, GT_L0)`, y=`IoU(pred_L4, pred_L0)`)
   - Level 3 left/right overlap matrix
   - Level 2는 sample 이미지 grid만

---

## 5. 예상 결과와 해석 프레임

| 레벨 | 예상 | 해석 |
|---|---|---|
| L0 | mIoU 0.3~0.5 (LERF 재현값 수준) | baseline 재현 검증 |
| L1 | L0 대비 5~15 mIoU 하락 | CLIP attribute grounding의 약점 |
| L2 | 대부분 실패 (GOOD < 20%) | SAM s-scale 단일 사용의 한계 (H2) |
| L3 | left/right overlap > 0.9 | **instance 분리 불가 결정적 증거** (H10) |
| L4 | `IoU(pred_L4, pred_L0) > 0.85` | **CLIP이 공간관계를 완전히 무시**하는 증거 (H10) |
| L5 | L0 대비 10~20 mIoU 하락 | CLIP 문장 인코딩의 dilution 효과 |

이 5개 결과를 합치면 "training-free가 가진 4가지 구조적 한계"가 한 실험에서 나란히 증명된다.

---

## 6. 필요한 산출물

- `output/render/lerf_exp/ramen/<frame>/L{0..5}_<query>.png` — 예측 이진 마스크
- `output/render/lerf_exp/ramen/<frame>/L{0,1,4,5}_<query>_gt.png` — GT mask (같은 것을 L1/L4/L5에 복사)
- `output/render/lerf_exp/ramen_metrics.csv` — (level, query, frame, IoU, P, R, F1, Acc)
- `output/render/lerf_exp/summary_*.png` — 종합 차트들
- 노트북 하단에 in-line 시각화

---

## 7. 실험 실행 전 사전 조건

1. Cell 20까지 실행해 `output/lerf/ramen/sai_nag.pt` 존재
2. `scene.getTrainCameras()`가 131장 전체를 반환 (이 중 7 프레임이 평가 대상)
3. `nag_data.SemanticNAG`, `utils.vlm_utils.ClipSimMeasure` 모두 import 가능
4. Colab T4 / L4 GPU 기준 (로컬 Mac에선 파이프라인 실행 불가, 이 md는 실행 설계서이며 Colab에서 Cell 30~35 실행)

Colab에서 sai_nag.pt 생성이 완료되면 Cell 30 이후를 순서대로 돌리면 된다.
