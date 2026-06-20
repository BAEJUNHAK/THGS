# ReferSplat HuggingFace 3DGS 체크포인트 → VALA 호환성 분석

> **결론: ReferSplat의 HuggingFace `*chkpnt30000.pth` 파일은 VALA의 RGB 3DGS 체크포인트로 그대로 재사용 가능하다.**
> Ref-LERF 데이터셋과 짝지어 쓰면 VALA의 `train.py` 30k RGB 학습 단계를 통째로 건너뛸 수 있다.
> 아래는 실제 다운로드·로드·수치 검증으로 입증한 결과다 (추정이 아니라 직접 실험).

- 검증 일자: 2026-06-15
- 검증 대상: ReferSplat HF `FudanCVL/RefSplat` 의 `ramenchkpnt30000.pth` (396MB) + Ref-LERF `FudanCVL/Ref-Lerf` ramen 씬 COLMAP
- 검증 환경: CPU only (Apple M4 Max, CUDA 불필요 — 정합은 수치 증명으로 확인)

---

## 1. 대상 자산

### ReferSplat 논문 / 레포
- 논문: *ReferSplat: Referring Segmentation in 3D Gaussian Splatting* (ICML 2025 Oral), arXiv 2508.08252
- 코드: https://github.com/heshuting555/ReferSplat
- 체크포인트(HF): https://huggingface.co/FudanCVL/RefSplat
- 데이터셋(HF): https://huggingface.co/datasets/FudanCVL/Ref-Lerf

### HF `FudanCVL/RefSplat` 내용물 (총 4.69GB)

| 파일 | 크기 | 정체 | VALA 사용 |
|---|---|---|---|
| `figurineschkpnt30000.pth` | 669MB | **원본 3DGS RGB 체크포인트** (12-tuple) | ✅ 사용 가능 |
| `ramenchkpnt30000.pth` | 415MB | 〃 | ✅ |
| `teatimechkpnt30000.pth` | 1.4GB | 〃 | ✅ |
| `kitchenchkpnt30000.pth` | 1.2GB | 〃 (= waldo_kitchen 로 추정, 확인 권장) | ✅ |
| `ramen.pth` | 263MB | ReferSplat **학습 완료** 모델 (17-tuple, language_feature + mlp + cross_attention) | ❌ 불필요/비호환 |
| `waldo_kitchen.pth` | 744MB | 〃 | ❌ |

- `*chkpnt30000.pth` = ReferSplat README가 `--start_checkpoint`로 요구하는 **"원본 3DGS pretrained Gaussians"**.
  → VALA의 2단계(`train.py`)가 직접 만드는 산출물과 동일한 종류.
- `ramen.pth` / `waldo_kitchen.pth` = ReferSplat 전용 17-tuple (language feature + MLP + cross-attention 포함). VALA와 무관.

---

## 2. 검증 결과 요약

| 조건 | 상태 | 근거 |
|---|---|---|
| ① 포맷 호환 | ✅ 증명됨 | 12-tuple, VALA `restore_rgb`와 요소·순서 완전 일치 |
| ① 포즈/COLMAP 정합 | ✅ 결정적 증명 | `spatial_lr_scale` 재계산 6.915804 == 저장값 6.915804 (ratio 1.00000) |
| ② pth→ply 변환 | ✅ 실행 완료 | 136MB ply, f_rest 45채널, VALA `load_ply` 호환, xyz round-trip 일치 |
| ③ 씬 이름 | ✅ ramen 확인 / ⚠️ kitchen→waldo_kitchen 확인 권장 | — |

---

## 3. 조건 ① — 포맷 호환성 (증명됨)

`torch.load`로 직접 디코드한 `ramenchkpnt30000.pth` 구조:

```
top-level = (model_params, 30000)        # (tuple, first_iter)
model_params: 12-tuple
   0 active_sh_degree   = 3
   1 _xyz               Tensor (576847, 3)   float32
   2 _features_dc       Tensor (576847, 1, 3)
   3 _features_rest     Tensor (576847, 15, 3)   # (sh_degree+1)^2 - 1 = 15
   4 _scaling           Tensor (576847, 3)
   5 _rotation          Tensor (576847, 4)
   6 _opacity           Tensor (576847, 1)
   7 max_radii2D        Tensor (576847,)
   8 xyz_gradient_accum Tensor (576847, 1)
   9 denom              Tensor (576847, 1)
  10 optimizer_state    dict {state, param_groups}
  11 spatial_lr_scale   = 6.915803527832032
```

VALA `scene/gaussian_model.py`의 `restore_rgb()`가 언패킹하는 12-tuple과 **순서·타입·shape가 완전히 동일**하다 (둘 다 inria 원본 3DGS에서 파생):

```python
# VALA: scene/gaussian_model.py  restore_rgb()
(self.active_sh_degree, self._xyz, self._features_dc, self._features_rest,
 self._scaling, self._rotation, self._opacity, self.max_radii2D,
 xyz_gradient_accum, denom, opt_dict, self.spatial_lr_scale) = model_args
```

- sh_degree=3 → VALA 기본값(`dataset.sh_degree`)과 일치.
- 활성화 함수도 동일: scaling=`exp`, opacity=`sigmoid`, rotation=`normalize`, SH convention 동일(graphdeco).
- ⚠️ 래스터라이저는 다름 — ReferSplat은 `diff_gaussian_rasterization`(원본 CUDA), VALA는 `gsplat`. 그러나 Gaussian 파라미터 규약이 같아 **호환**되며(gsplat은 3DGS 호환 설계), antialiasing/타일 블렌딩에서 수치 미세차이만 존재. 포맷 비호환 없음.

→ **코드 수정 없이 `restore_rgb`로 그대로 로드된다.**

---

## 4. 조건 ① — COLMAP/포즈 정합 (결정적 증명)

가장 중요한 리스크였던 "체크포인트와 카메라 포즈가 같은 좌표계인가". 체크포인트에 저장된 `spatial_lr_scale`(= 카메라 extent 반경)을 Ref-LERF ramen COLMAP의 131개 카메라로 재계산:

```
재계산 camera extent (= spatial_lr_scale): 6.915804
체크포인트 저장값                         : 6.915804
ratio (재계산/저장)                       : 1.00000   ← 유효숫자 6자리 일치
```

`spatial_lr_scale`은 카메라 위치들로부터 유도되는 부동소수 스칼라다. 6자리까지 일치한다는 것은 **카메라 집합·COLMAP 포즈·좌표 프레임이 동일**하다는 뜻 — 즉 **HF 체크포인트는 바로 이 Ref-LERF COLMAP 재구성으로 학습되었다**는 거의 암호학적 증명이다.

보조 증거:
- 카메라 중심 131개 **100%가 Gaussian bbox 내부**
- COLMAP seed points 29,746개가 densify되어 576,847 Gaussians로 성장 (정상적 3DGS 관계)

```
camera   bbox: min [-2.92, -2.92, -4.65]   max [4.40, 4.73, 3.76]
Gaussian bbox: min [-15.12, -6.92, -18.06] max [18.55, 34.67, 19.01]
COLMAP pts3D : min [-43.4, -17.28, -18.24] max [21.26, 37.59, 32.08]  (29,746 pts)
```

재계산 로직(VALA `scene/dataset_readers.py` `getNerfppNorm` 재현):

```python
# 카메라별 W2C → C2W → 중심 C, 평균중심에서 최대거리 * 1.1 = radius = spatial_lr_scale
W2C = getWorld2View2(R, T)               # R = qvec2rotmat(qvec).T,  T = tvec
C   = np.linalg.inv(W2C)[:3, 3]
radius = max_dist(camera_centers, mean_center) * 1.1
```

> **핵심:** 앞서 경고했던 "다른 lerf_ovs 변형과 섞으면 Gaussian과 뷰가 어긋나 feature가 깨진다"는 리스크는,
> **Ref-LERF 자체의 `sparse/0`를 함께 사용하면 완벽 정합**으로 해소된다.

---

## 5. 조건 ② — pth → point_cloud.ply 변환 (실행 완료)

VALA의 `Scene(load_iteration=30000)`은 `model_path/point_cloud/iteration_30000/point_cloud.ply`를 먼저 `load_ply`로 읽는다. HF에는 `.ply`가 없으므로(=.pth만 존재) 변환이 필요하다. 단, 그 직후 `restore_rgb`가 Gaussian 텐서를 통째로 덮어쓰므로 ply는 형식만 맞으면 된다 (가장 깔끔한 방법은 체크포인트를 복원해 `save_ply`).

변환 결과 (VALA `save_ply` 로직 재현):
```
WROTE ply: vertices = 576847,  total_fields = 62
  f_dc 3 + f_rest 45 + scale 3 + rot 4 + (xyz, normal, opacity)
  f_rest 채널 = 45 = 3*(sh_degree+1)^2 - 3   ← VALA load_ply 기대값과 정확히 일치
  xyz round-trip equal: True
```

변환 스크립트:
```python
import torch, numpy as np, os
from plyfile import PlyData, PlyElement

mp, _ = torch.load("ramenchkpnt30000.pth", map_location="cpu", weights_only=False)
mp = [x.detach() if torch.is_tensor(x) else x for x in mp]
xyz, fdc, frest, scaling, rotation, opacity = mp[1], mp[2], mp[3], mp[4], mp[5], mp[6]

l = ['x','y','z','nx','ny','nz']
for i in range(fdc.shape[1]*fdc.shape[2]):   l.append(f'f_dc_{i}')
for i in range(frest.shape[1]*frest.shape[2]): l.append(f'f_rest_{i}')
l.append('opacity')
for i in range(scaling.shape[1]):  l.append(f'scale_{i}')
for i in range(rotation.shape[1]): l.append(f'rot_{i}')

xyz_n   = xyz.numpy(); normals = np.zeros_like(xyz_n)
f_dc    = fdc.transpose(1,2).flatten(1).contiguous().numpy()
f_rest  = frest.transpose(1,2).flatten(1).contiguous().numpy()
attrs   = np.concatenate((xyz_n, normals, f_dc, f_rest,
                          opacity.numpy(), scaling.numpy(), rotation.numpy()), axis=1)
elements = np.empty(xyz_n.shape[0], dtype=[(a,'f4') for a in l])
elements[:] = list(map(tuple, attrs))
os.makedirs("point_cloud/iteration_30000", exist_ok=True)
PlyData([PlyElement.describe(elements,'vertex')]).write("point_cloud/iteration_30000/point_cloud.ply")
```

---

## 6. 권장 사용 절차 (VALA에서 ReferSplat 체크포인트 재사용)

1. HF에서 `FudanCVL/RefSplat`의 `<scene>chkpnt30000.pth` 4개 + `FudanCVL/Ref-Lerf` 데이터셋 다운로드.
2. 배치:
   - 데이터셋(이미지 + `sparse/0`): `dataset/3dgs/lerf_ovs/<scene>/`
   - 체크포인트: `output/3dgs/lerf_ovs/<scene>/chkpnt30000.pth`
3. 위 변환 스크립트로 `output/3dgs/lerf_ovs/<scene>/point_cloud/iteration_30000/point_cloud.ply` 생성.
4. (정합 sanity check) 한 뷰를 RGB로 렌더해 원본 이미지와 정렬되는지 확인 — `spatial_lr_scale` 일치로 이미 수치 증명됐으나, 시각 확인은 추가 안전장치.
5. `python gaussian_feature_extractor.py -m output/3dgs/lerf_ovs/<scene> --iteration 30000 ...` 로 **언어 feature 부착 단계부터 진행** (VALA `train.py` 30k RGB 학습 스킵).

> 단, 5단계 이후(feature 부착·렌더·평가)는 **gsplat(CUDA) 필요** — GPU 환경에서 실행해야 한다.

---

## 7. THGS/VALA ramen 결정 실험 결과 (2026-06-17)

위 호환성 분석을 실제 THGS VALA 파이프라인에 적용해 결정 실험을 수행했다.

### 실험 질문

**저장된 2D language_features를 그대로 고정하고 RGB 3DGS만 ReferSplat pretrained checkpoint로 교체하면, VALA ramen 실패가 회복되는가?**

이 질문은 "ramen 문제가 그냥 3DGS 학습 문제인가?"를 가르는 목적이다.

### 실행 조건

- GPU 예약: `cal_jhbae_rtxpro5000_fc0c78ca11ab`
- 서버: `jhsong-TRX50-AI-TOP`, RTX PRO 5000 Blackwell
- checkpoint: `external_methods/_refersplat_hf/ramenchkpnt30000.pth`
- VALA model dir: `external_methods/VALA/output/refersplat_3dgs/lerf_ovs/ramen`
- 변환 스크립트: `scripts/convert_vala_chkpnt_to_ply.py`
- 실행 스크립트: `scripts/run_vala_refersplat_ramen.sh`
- 실행 로그: `external_methods/_vala_refersplat_ramen_67.log`
- 기존 저장 feature 사용:
  - `data/lerf_ovs/ramen/langsplat/language_features -> ../language_features`

sanity:

```
checkpoint gaussians       : 576,847
checkpoint spatial_lr_scale: 6.915803528
local COLMAP radius        : 6.916664791
ratio                      : 1.000124536
```

→ checkpoint와 현재 ramen COLMAP은 좌표계가 사실상 정합한다.

### 결과

| variant | RGB 3DGS | final mIoU | Acc@0.25 | Acc@0.5 |
|---|---|---:|---:|---:|
| 기존 VALA self-train | 217,508 Gaussians | 0.3580 | 0.6197 | 0.3099 |
| VALA dense | 588,826 Gaussians | 0.3575 | 0.5493 | 0.2817 |
| **ReferSplat pretrained RGB** | 576,847 Gaussians | **0.3636** | 0.6197 | 0.2817 |

failure-locus 비교:

| variant | final IoU | precision | recall | area ratio | final ok | 2D semantic fail | 3D overgrowth |
|---|---:|---:|---:|---:|---:|---:|---:|
| VALA self-train | 0.3580 | 0.4269 | 0.7517 | 5.5003 | 22/71 | 32 | 7 |
| VALA dense | 0.3575 | 0.4256 | 0.7082 | 3.7290 | 20/71 | 32 | 8 |
| **ReferSplat RGB** | **0.3636** | 0.4492 | 0.7102 | 3.7469 | 20/71 | 32 | 7 |

threshold sweep:

| mask_thresh | mIoU | Acc@0.25 | Acc@0.5 |
|---:|---:|---:|---:|
| 0.4 | 0.3349 | 0.4789 | 0.3239 |
| 0.5 | 0.3610 | 0.5775 | 0.3099 |
| 0.6 | **0.3636** | **0.6197** | 0.2817 |
| 0.7 | 0.3396 | 0.4930 | **0.3380** |
| 0.8 | 0.2378 | 0.3803 | 0.2113 |

### 판정

**ReferSplat pretrained RGB 3DGS로 교체해도 ramen mIoU는 0.3580 → 0.3636으로 +0.56pt만 오른다.**

threshold sweep에서도 best가 0.3636에 머문다. 따라서 현재 THGS/VALA ramen gap은 **우리 3DGS 학습 품질 단독 문제로 보기 어렵다.** 실패 위치도 기존과 거의 같고, 2D semantic selection failure가 32/71로 그대로 유지된다. 즉 이 실험은 다음 결론을 지지한다.

> ramen 문제는 RGB 3DGS가 아니라, 저장된 2D feature의 semantic ambiguity와 VALA의 3D lifting/render selection 한계에서 주로 발생한다.

---

## 7.1 후속 결정 실험: VALA official feature 재생성 (2026-06-17)

위 실험은 저장된 THGS language_features를 고정했기 때문에, 남은 핵심 질문은 **저자 방식의 VALA/LangSplat feature를 새로 만들면 paper reported ramen 0.604가 회복되는가**였다.

### 실행 조건

- GPU 예약: `cal_jhbae_rtxpro5000_fc0c78ca11ab`
- feature 생성: VALA official `run_sam.py --use_langsplat --get_semantic`
- source 격리 루트: `external_methods/VALA_repro_root/dataset/3dgs/lerf_ovs/ramen`
- feature output: `external_methods/VALA_repro_root/dataset/3dgs/lerf_ovs/ramen/langsplat/language_features`
- model dir: `external_methods/VALA/output/refersplat_3dgs_valafeat/lerf_ovs/ramen`
- scripts:
  - `scripts/run_vala_official_feature_refersplat_ramen.sh`
  - `scripts/run_vala_official_feature_refersplat_ramen_thresh_sweep.sh`
  - `scripts/postprocess_vala_official_feature_refersplat_ramen.sh`

sanity:

```
generated feature files: 262 = 131 images * 2
frame_00006_f.npy: (197, 512), float16
frame_00006_s.npy: (4, 731, 988), int32
```

### 결과

| condition | final mIoU | Acc@0.25 | Acc@0.5 |
|---|---:|---:|---:|
| saved THGS feature + ReferSplat RGB | 0.3636 | 0.6197 | 0.2817 |
| **VALA official feature + ReferSplat RGB** | **0.4589** | **0.6761** | **0.4507** |
| VALA paper ramen | 0.604 | - | - |

threshold sweep:

| mask_thresh | mIoU | Acc@0.25 | Acc@0.5 |
|---:|---:|---:|---:|
| 0.4 | 0.3850 | 0.6056 | 0.3803 |
| 0.5 | 0.4565 | **0.7324** | 0.4366 |
| 0.6 | **0.4589** | 0.6761 | **0.4507** |
| 0.7 | 0.4025 | 0.5915 | 0.3944 |
| 0.8 | 0.2620 | 0.4225 | 0.1972 |

failure-locus:

| condition | final IoU | precision | recall | source IoU | sem-top IoU | final ok | 2D semantic fail | 3D overgrowth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| saved THGS feature + ReferSplat RGB | 0.3636 | 0.4492 | 0.7102 | 0.8550 | 0.4050 | 20/71 | 32 | 7 |
| **VALA official feature + ReferSplat RGB** | **0.4589** | **0.5282** | 0.6856 | 0.8550 | **0.4445** | **32/71** | **23** | **2** |

### 판정

**VALA official feature 재생성은 ramen을 유의미하게 회복하지만 paper reported 0.604는 재현하지 못한다.**

따라서 ReferSplat RGB checkpoint 호환성은 계속 유효하고, ramen 재현성 갭의 현재 분해는 다음과 같다.

1. RGB 3DGS 품질/출처 단독 원인: 기각.
2. 기존 저장 feature 재사용 mismatch: 일부 원인. 0.3636 → 0.4589 회복.
3. 남은 0.4589 → 0.604 갭: threshold가 아니라 eval/protocol/query/data bundle 또는 특정 ramen prompt의 semantic/lifting failure.

---

## 8. 주의사항

- **씬 이름 매핑**: HF의 `kitchenchkpnt30000.pth`(1.2GB)가 VALA의 `waldo_kitchen`에 대응하는지 다운로드 후 Gaussian 수/렌더로 확인 권장. ramen은 완전 검증됨.
- **라이선스**: 원본 3DGS(inria)의 **연구·비상업** 라이선스를 계승. ReferSplat 인용 필요:
  ```bibtex
  @inproceedings{ReferSplat,
    title={{ReferSplat}: Referring Segmentation in 3D Gaussian Splatting},
    author={He, Shuting and Jie, Guangquan and Wang, Changshuo and Zhou, Yun and Hu, Shuming and Li, Guanbin and Ding, Henghui},
    booktitle={International Conference on Machine Learning (ICML)}, year={2025}
  }
  ```
- **데이터 정합이 유일한 실질 리스크**: 코드/포맷이 아니라 "동일 COLMAP" 한 가지. Ref-LERF 자체 데이터셋을 쓰면 해결됨 (위 4절 증명).

---

## 9. 한 줄 요약

ReferSplat HF `<scene>chkpnt30000.pth` = 표준 12-tuple 3DGS RGB 체크포인트이며 VALA `restore_rgb`와 바이트 구조 동일.
Ref-LERF COLMAP과 `spatial_lr_scale`이 6자리까지 일치 → 동일 좌표계 확정.
pth→ply 변환 한 번이면 VALA 파이프라인의 RGB 학습 단계를 건너뛰고 곧바로 언어 feature 부착부터 시작 가능.
실제 ramen 결정 실험에서는 ReferSplat RGB 3DGS 교체 후에도 VALA mIoU가 0.3636에 머물러, ramen gap의 주원인이 단순 3DGS 학습 실패가 아님을 확인했다.
VALA official feature까지 재생성하면 0.4589로 회복되지만 논문값 0.604에는 여전히 미달하여, 남은 재현성 갭은 feature 재사용만으로 설명되지 않는다.
