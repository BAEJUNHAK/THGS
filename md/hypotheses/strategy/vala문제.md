# VALA 재현 문제 — 핸드오프 문서 (2026-06-15)

> 목적: P1-E (외부 method 실측)의 일환으로 **VALA** (Visibility-Aware Language Aggregation, arXiv 2509.05515, 3DV 2026) 공식 코드를 우리 Blackwell 클러스터에서 재현. 이 문서는 **다른 AI/사람이 맥락을 즉시 파악**하도록 환경·파이프라인·결과·기각 가설·gotcha·재현 명령을 자기완결적으로 정리한다.
>
> 상위 문서: [p1e_external_code_study_plan.md](p1e_external_code_study_plan.md), [roadmap.md](roadmap.md).

---

## 0. 한 문장 요약

> **2026-06-18 정정 / 최신 판정**
>
> 이전 기록의 `ramen 논문값 0.604`는 잘못 읽은 값이다. VALA Table 1의 **3D evaluation**에서 `60.38`은 **Figurines**, `45.41`이 **Ramen**, `55.71`이 **Waldo Kitchen**이다. 즉, ramen은 `VALA official feature + ReferSplat RGB` 조건에서 **0.4589**로 논문 3D target **0.4541**을 사실상 재현했다. 현재 진짜 문제는 **Waldo Kitchen**이다: official VALA 코드로 RGB를 처음부터 30k 학습하고 official feature assignment를 다시 해도 **3D mIoU 0.4904 vs 논문 0.5571**, official 2D eval도 **0.5412@0.5 / 0.5575@0.4 vs 논문 0.651**에 머문다.
>
> threshold도 확인 완료: 논문 Appendix B는 LERF-OVS **2D evaluation threshold 0.5**, **3D evaluation Gaussian relevancy threshold 0.6**을 명시한다. 공식 `scripts/eval_lerf_ovs_3d.sh`도 `mask_thresh=0.6`을 쓴다. 따라서 Waldo 갭은 단순 threshold 선택 문제가 아니라, **공개 코드/공개 데이터/저자-side checkpoint 또는 feature bundle 사이의 재현성 갭**으로 보는 것이 맞다.

> 2026-06-17 당시에는 `0.604`를 ramen 값으로 오독해 ramen을 미재현 scene으로 추적했다. 이 문서의 ramen 진단은 feature mismatch와 failure-locus 분석 기록으로 남겨두되, 2026-06-18 이후 재현성 문제의 중심 scene은 **Waldo Kitchen**이다.

---

## 1. 핵심 결과표

| scene | 모델 | 가우시안 | 우리 mIoU | 논문 | 판정 |
|---|---|---:|---:|---:|---|
| **teatime** | A: LangSplat pretrained | 2,211,209 | **0.685** | 0.7061 | ✅ 재현 |
| **teatime** | B: 우리 자체학습 | 749,489 | **0.694** | 0.7061 | ✅ 재현 |
| **ramen** | 우리 자체학습 (densify 0.0004) | 217,508 | **0.358** | 0.4541 | ❌ −9.6pt |
| **ramen** | 우리 자체학습 (densify 0.0002) | 588,826 | **0.3575** | 0.4541 | ❌ 동일 |
| **ramen** | ReferSplat HF pretrained RGB 3DGS | 576,847 | **0.3636** | 0.4541 | ❌ 동일 |
| **ramen** | ReferSplat RGB + VALA 공식 feature 재생성 | 576,847 | **0.4589** | 0.4541 | ✅ 재현권 |
| **waldo_kitchen** | VALA official RGB train 30k + official features | - | **0.4904** | 0.5571 | ❌ −6.7pt (3D) |
| **waldo_kitchen** | VALA official RGB train 30k + official 2D eval | - | **0.5412** | 0.6510 | ❌ −11.0pt (2D@0.5) |

→ **teatime A/B**: 밀도 3배 차이(2.2M vs 749k)에도 mIoU 동일 → **밀도 무관, 파이프라인 정상**.
→ **ramen**: 밀도 2.7배(217k→589k) 늘려도 0.358 불변 → **under-densification 아님**.
→ **ramen ReferSplat**: 원저자급 pretrained RGB 3DGS로 교체해도 0.3636 → **우리 3DGS 학습 실패 단독 원인 아님**.
→ **ramen official feature**: VALA 방식으로 feature를 재생성하면 0.4589까지 회복 → **논문 3D target 0.4541 재현권**. 기존 feature 재사용 mismatch는 일부 원인이었다.
→ **waldo_kitchen**: official train/feature/eval을 다시 태워도 2D/3D 모두 논문값보다 낮음 → **현재 재현성 문제의 중심**.

(official-feature ramen per-prompt: `egg` 0.82, `kamaboko` 0.84, `nori` 0.80, `chopsticks` 0.73은 정상권으로 회복. 반면 `corn` 0.02, `onion segments` 0.01, `spoon` 0.12, `sake cup` 0.24, `plate` 0.25는 계속 실패 — 특정 query 잔여 갭이 크다.)

---

## 2. 환경 (이게 절반의 작업이었음 — 그대로 재사용 가능)

### 2.1 클러스터 사실
- **GPU**: RTX PRO 5000 Blackwell **sm_120** (cap 12.0, 48GB). SLURM 예약제.
- **노드 이질성 (중요)**: login `pilab-gpu-b` = python3.10 + `/usr/local/cuda-12.4` + GPU 보유. 예약 노드 `jhsong-TRX50-AI-TOP` = **python3.12 + CUDA toolkit 없음**.
- **CUDA 12.8 toolkit이 어디에도 없음** → sm_120 네이티브 빌드 불가 → **cuda-12.4 + `TORCH_CUDA_ARCH_LIST="9.0+PTX"` forward-compat (PTX-JIT)** 로 빌드. simple_knn·gsplat 모두 이 방식으로 sm_120에서 실행 검증됨.
- 계정 `jhbae`, 예약 ID 패턴 `cal_jhbae_rtxpro5000_*` (`scontrol show reservation -o | grep jhbae`로 조회, 매번 바뀜).

### 2.2 portable env = `envs/vala-port` (모든 예약 노드에서 재사용)
- **thgs의 conda-pack python3.10을 base로** 생성 (자체 python을 품어 노드 이질성 무관):
  ```bash
  envs/thgs/bin/python -m venv --system-site-packages envs/vala-port   # torch 2.11.0+cu128 상속
  # get-pip 부트스트랩(시스템 python엔 ensurepip 없음), setuptools<75 + wheel 설치
  # VALA requirements.txt 설치 (open-clip-torch 3.3.0, numpy<2.0 등)
  ```
- **jhsong 예약 노드에서 동작 검증됨** (torch 2.11+cu128, cap (12,0)).
- 시스템 python venv는 **금지** — jhsong이 python3.12라 깨짐. 반드시 thgs-base.

### 2.3 빌드 (한 번이면 영구)
- 환경변수: `CUDA_HOME=/usr/local/cuda-12.4`, `PATH=$CUDA_HOME/bin:$PATH`, `TORCH_CUDA_ARCH_LIST="9.0+PTX"`, `SETUPTOOLS_USE_DISTUTILS=stdlib`, `MAX_JOBS=8`.
- `pip install --no-build-isolation -e ./submodules/simple_knn` (simple_knn에 `__init__.py` 추가 필요 — 원래 없음).
- `pip install --no-build-isolation -e ./submodules/gsplat` (gsplat 1.4.0, .cu 25개, ~10분).
- **diff_gaussian_rasterization 불필요** (optimizer_type='sparse_adam'일 때만 쓰임; 기본 Adam).

### 2.4 sitecustomize 호환 shim
`envs/vala-port/lib/python3.10/site-packages/sitecustomize.py` 에 **`torch.load` 기본 `weights_only=False` 복원** (VALA는 torch<2.6 기준; 우리는 2.11). 이거 없으면 모든 chkpnt 로드가 깨짐.

---

## 3. 파이프라인 (전부 검증됨)

VALA = **OccamLGS 기반** (built on LangSplat + 3DGS + gsplat). 4단계:

```
1. train.py            : vanilla RGB 3DGS 학습 (gsplat, 30k iter)
2. gaussian_feature_extractor.py --use_efficient --eval : robust Weiszfeld로 512-d CLIP을 가우시안에 lift (level 1/2/3)
3. eval/render_lerf_by_text_langsplat : open_clip 텍스트로 relevance 렌더 → silhouette mask
4. eval/compute_lerf_iou : GT와 비교 → mIoU
```

### 3.1 데이터 재사용 (재생성 불필요)
- `data/lerf_ovs/<scene>/{images, sparse/0}` = VALA 기대 구조와 일치 (해상도 988×731).
- **language_features 재사용**: THGS의 `data/lerf_ovs/<scene>/language_features/` 를 VALA가 기대하는 `<scene>/langsplat/language_features` 로 심링크. THGS·VALA가 **동일 CLIP** (OpenCLIPNetwork, ViT-B-16, laion2b_s34b_b88k) 이라 호환. seg `[4,H,W]` + feat `[N,512]` 포맷 일치.
- **test.txt 생성 필요**: `data/lerf_ovs/<scene>/sparse/0/test.txt` = label 프레임 stem 목록 (VALA `--eval` split). 4 scene 생성 완료.

### 3.2 재현 명령 (jhsong 예약, 검증된 형태)
```bash
RID=cal_jhbae_rtxpro5000_XXXX   # 실행 시점 조회
VP=/mnt/pilab_nas/projects/THGS/envs/vala-port/bin/python
S=/mnt/pilab_nas/projects/THGS/data/lerf_ovs/teatime
M=output/3dgs/lerf_ovs/teatime
srun -p gpu --reservation=$RID --gres=gpu:1 -t 60 bash -lc "
  cd /mnt/pilab_nas/projects/THGS/external_methods/VALA
  export HOME=/mnt/pilab_nas/projects/THGS/.valahome MPLCONFIGDIR=/tmp/mpl HF_HOME=/mnt/pilab_nas/projects/THGS/.valahome/hf
  $VP train.py -s $S -m $M --iterations 30000
  for L in 1 2 3; do $VP gaussian_feature_extractor.py -s $S -m $M --iteration 30000 --feature_level \$L --use_efficient --eval; done
  $VP -m eval.render_lerf_by_text_langsplat -s $S -m $M --iteration 30000 --mask_thresh 0.6 --scene_name teatime --dataset_name lerf_ovs --ae_ckpt_dir output/3dgs --base_dir $M --output_dir $M --eval --skip_train
  $VP -m eval.compute_lerf_iou --scene_name teatime --gt_dir output/3dgs/lerf_ovs --pred_dir output/3dgs/lerf_ovs --output_dir output/3dgs/lerf_ovs --ablation_type none --mask_thresh 0.6 --iteration 30000 --json_dir $S/../label
"
```

---

## 4. 기각된 가설 (ramen 갭의 원인 *아님* — 전부 실측)

| 가설 | 기각 근거 |
|---|---|
| **pretrained 3DGS 써야** | 존재 안 함. VALA·OccamLGS·LangSplat·ReferSplat 전부 "train.py로 직접 학습"이 정해진 워크플로. LangSplat Drive엔 LERF 중 **teatime+sofa만** 공개 (ramen/figurines/waldo 없음). |
| **밀도(under-densification)** | ramen densify 0.0002로 217k→589k(teatime급)로 늘려도 mIoU 0.358 불변. teatime A(2.2M)/B(749k) 동일 mIoU. |
| **CLIP 불일치** | THGS image_encoding.py와 VALA run_sam.py가 **동일** OpenCLIPNetwork (ViT-B-16/laion2b/512d, config 클래스 글자까지 동일). |
| **해상도 불일치** | 이미지·seg map 모두 988×731. 다운샘플 없음. |
| **파이프라인 버그** | teatime가 논문 수준 재현 (0.694 vs 0.706). |
| **3DGS 출처/포맷** | LangSplat chkpnt는 **13-tuple** ([7]에 (N,3) 추가 = LangSplat 3-dim 언어 feature → 언어모델), 우리는 **12-tuple** 순수 RGB. 하지만 우리가 RGB .ply만 추출해 변환 → teatime A/B 둘 다 재현. 차이 무영향. |

---

## 5. ramen-특이 진단 기록 (현재 재현성 중심은 Waldo)

- 2026-06-17 당시에는 `0.604`를 ramen target으로 오독해 ramen을 미재현 scene으로 추적했다.
- Table 1 정정 후 ramen 3D target은 **0.4541**이며, `VALA official feature + ReferSplat RGB`의 **0.4589**는 재현권이다.
- 아래 진단은 "왜 기존 저장 feature + self-train/ReferSplat RGB 조건에서는 ramen이 0.36대였는가"를 설명하는 기록으로 유지한다.
- 관측: ramen PSNR 27.71 (teatime 30.04), ramen SAM 마스크 123-215/frame (teatime 300-420). 단 ramen이 단순 씬이라 마스크 적은 건 정상일 수 있음 — **결정적 결함 미발견**.
- **완료된 결정 테스트**: VALA `run_sam.py --use_langsplat`로 ramen language_features를 **저자 방식대로 재생성** → ReferSplat RGB와 결합해 재eval. 결과는 0.3636 → **0.4589**로 개선되어, 우리 재사용(THGS) feature가 ramen에서 일부 갭을 만든 것은 맞고 official feature 재생성 후에는 논문 3D target을 재현한 것으로 본다.
- 보조 후보: ramen GT/test 프레임 특이, ramen 다중인스턴스 (sake cup/plate/napkin 등 — THGS 분석이 ramen을 multi-instance 다발로 지목한 바 있음).

### 5.1 추가 진단 업데이트 — "그냥 3DGS 학습 문제인가?" 에 대한 현재 답

2026-06-15 추가로 GPU 없이 가능한 진단을 수행했다. 결론부터 말하면 **3DGS 학습 품질은 증폭 요인일 수 있지만, 현재 관측을 단독으로 설명하지 못한다.** 이유는 네 가지다.

1. **2D source mask 자체는 ramen에서도 충분히 좋다.**
   - `scripts/p1e_vala_source_diagnostics.py`
   - 산출물: `output/diagnostics/p1e_vala_2d_source_oracle.csv`
   - best 2D mask oracle IoU: ramen **0.8550**, teatime **0.9166**
   - IoU >= 0.5 비율: ramen **0.958**, teatime **0.966**
   - 즉 ramen object에 겹치는 SAM/LangSplat source mask는 대부분 존재한다. "mask가 없어서" 깨진 것은 아니다.

2. **ramen은 2D semantic top 선택부터 더 자주 틀어진다.**
   - `scripts/p1e_vala_2d_semantic_diagnostics.py`
   - 산출물: `output/diagnostics/p1e_vala_2d_semantic_selection.csv`
   - 2D semantic top IoU: ramen **0.4050**, teatime **0.5616**
   - source oracle은 높은데 top semantic 선택이 틀리는 대표 prompt:
     - `corn`: sem-top 0.0000, oracle 0.8170
     - `onion segments`: sem-top 0.0000, oracle 0.8559
     - `nori`: sem-top 0.0690, oracle 0.9597
     - `kamaboko`: sem-top 0.1312, oracle 0.9132
   - 이건 ramen에서 CLIP/text-to-mask association이 이미 취약하다는 뜻이다. 3DGS 이전 병목이 있다.

3. **최종 VALA 실패는 high-recall/low-precision overgrowth가 크다.**
   - `scripts/p1e_vala_mask_diagnostics.py`
   - 산출물: `output/diagnostics/p1e_vala_mask_diagnostics.csv`
   - final mask 통계:
     - ramen: IoU **0.3580**, precision **0.4269**, recall **0.7517**, mean area ratio **5.5003**
     - teatime: IoU **0.6945**, precision **0.7266**, recall **0.9208**, mean area ratio **2.0739**
   - ramen은 정답을 어느 정도 덮지만 너무 넓게 퍼지는 경우가 많다. 특히 `corn`, `sake cup`, `spoon`은 recall은 높은데 area ratio가 10배 이상으로 커진다.
   - PNG threshold sweep도 기존 저장 feature 조건의 ramen을 크게 살리지 못했다: pixel threshold를 바꿔도 ramen best는 약 **0.378** 수준이었다.

4. **Gaussian 선택 비율 자체는 ramen-only 폭주가 아니다.**
   - `scripts/p1e_vala_3d_relevance_diagnostics.py`
   - 산출물:
     - `output/diagnostics/p1e_vala_3d_relevance_nosmooth_chosen.csv`
     - `output/diagnostics/p1e_vala_3d_relevance_smooth_chosen.csv`
   - VALA와 같은 KNN smoothing 후 threshold 0.6에서 선택 Gaussian 비율:
     - ramen mean **0.0148**, median **0.0145**, max **0.0358**
     - teatime mean **0.0260**, median **0.0086**, max **0.1240**
   - 따라서 "ramen은 3D relevance가 전역적으로 너무 많이 켜져서 망한다"는 단순 가설도 약하다. overgrowth는 선택 개수보다는 **선택된 Gaussian의 공간/투영 위치와 semantic association 오류가 결합된 결과**로 보는 쪽이 맞다.

#### failure locus 분해

`scripts/p1e_vala_failure_locus.py` 로 source oracle / 2D semantic top / final mask를 합쳐 query별 병목을 분류했다.

산출물:
- `output/diagnostics/p1e_vala_failure_locus.csv`
- `output/diagnostics/p1e_vala_failure_locus_by_scene.csv`
- `output/diagnostics/p1e_vala_failure_locus_by_prompt.csv`
- dense ramen 비교: `output/diagnostics/p1e_vala_dense_failure_locus_by_scene.csv`

| scene | n | final ok | 2D source missing | 2D semantic selection | 3D overgrowth | 3D undercoverage | 3D lifting degradation | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ramen | 71 | **22** | 1 | **32** | 7 | 3 | 5 | 1 |
| teatime | 59 | **49** | 2 | 7 | 1 | 0 | 0 | 0 |

dense ramen도 거의 동일하다:

| model | n | final IoU | final ok | 2D semantic selection | 3D overgrowth |
|---|---:|---:|---:|---:|---:|
| ramen basic | 71 | 0.3580 | 22 | 32 | 7 |
| ramen dense | 71 | 0.3575 | 20 | 32 | 8 |

→ 밀도 증가가 실패 위치를 바꾸지 못했다. 이건 "3DGS가 sparse해서 깨졌다"는 가설을 약하게 만든다.

#### 현재 원인 모델

현재 가장 그럴듯한 chain은 다음이다.

```
ramen prompt 중 일부는 2D CLIP/text association이 이미 ambiguous
        ↓
VALA의 robust lifting은 source mask oracle을 쓰는 것이 아니라 선택된/누적된 semantic evidence를 3D Gaussian에 입힌다
        ↓
ramen의 낮은 RGB/geometry 품질(PSNR 27.7 vs teatime 30.0)과 다중/작은 object 구조가 leakage를 증폭한다
        ↓
최종 렌더 mask는 recall은 유지하되 precision이 크게 떨어지는 overgrowth로 나타난다
```

따라서 지금 답은:

> **3DGS 학습 문제만은 아니다. 3DGS는 leakage를 키우는 증폭기이고, 더 앞단에는 ramen의 2D semantic association 취약성이 있다.**

당시 남은 결정 테스트는 두 가지였고, 2026-06-17에 둘 다 실행했다.

1. **저자 방식으로 ramen language_features 재생성**: `run_sam.py --use_langsplat`를 돌려 source feature 차이를 제거했다. 결과는 final mIoU **0.4589**로 회복되어 정정된 논문 3D target **0.4541**과 일치했다.
2. **GPU threshold sweep**: official feature 조건에서 `mask_thresh=0.4~0.8`을 sweep했다. best는 **0.6 / 0.4589**였다. 즉 ramen에서는 threshold보다 feature 재생성이 결정적이었다.

### 5.2 결정 실험 — ReferSplat pretrained RGB 3DGS 교체

사용자 예약 GPU `cal_jhbae_rtxpro5000_fc0c78ca11ab`로 2026-06-17 실행.

목적: **저장된 2D language_features는 그대로 고정하고 RGB 3DGS만 ReferSplat HF pretrained checkpoint로 교체**했을 때 ramen이 회복되는지 확인한다.

입력/산출물:
- HF checkpoint: `external_methods/_refersplat_hf/ramenchkpnt30000.pth`
- VALA model dir: `external_methods/VALA/output/refersplat_3dgs/lerf_ovs/ramen`
- 변환 스크립트: `scripts/convert_vala_chkpnt_to_ply.py`
- Slurm script: `scripts/run_vala_refersplat_ramen.sh`
- 실행 로그: `external_methods/_vala_refersplat_ramen_67.log`
- 결과: `external_methods/VALA/output/refersplat_3dgs/lerf_ovs/ramen/none/predictions_mask_0.6/result.txt`
- 진단:
  - `output/diagnostics/p1e_vala_refersplat_mask_diagnostics.csv`
  - `output/diagnostics/p1e_vala_refersplat_failure_locus_by_scene.csv`
  - `output/diagnostics/p1e_vala_refersplat_decision_summary.csv`

sanity:
- checkpoint 12-tuple 로드 성공.
- Gaussians: **576,847**
- checkpoint `spatial_lr_scale`: **6.915803528**
- local ramen COLMAP radius: **6.916664791**
- ratio: **1.000124536** → 좌표계 정합 OK.
- 기존 저장 feature 사용: `data/lerf_ovs/ramen/langsplat/language_features -> ../language_features`

결과:

| variant | Gaussians | final IoU | precision | recall | area ratio | final ok | 2D semantic fail | 3D overgrowth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VALA self-train | 217,508 | 0.3580 | 0.4269 | 0.7517 | 5.5003 | 22/71 | 32 | 7 |
| VALA dense | 588,826 | 0.3575 | 0.4256 | 0.7082 | 3.7290 | 20/71 | 32 | 8 |
| ReferSplat RGB | 576,847 | **0.3636** | 0.4492 | 0.7102 | 3.7469 | 20/71 | 32 | 7 |

threshold sweep:

| mask_thresh | mIoU | Acc@0.25 | Acc@0.5 |
|---:|---:|---:|---:|
| 0.4 | 0.3349 | 0.4789 | 0.3239 |
| 0.5 | 0.3610 | 0.5775 | 0.3099 |
| 0.6 | **0.3636** | **0.6197** | 0.2817 |
| 0.7 | 0.3396 | 0.4930 | **0.3380** |
| 0.8 | 0.2378 | 0.3803 | 0.2113 |

판정:

> **ReferSplat pretrained RGB 3DGS로 교체해도 mIoU는 +0.0056만 오른다. 실패 위치도 거의 동일하고, threshold sweep에서도 best가 0.3636이다. 따라서 ramen gap은 "우리 3DGS 학습이 나빠서"가 아니라, 저장된 2D feature의 semantic ambiguity와 VALA 3D lifting/render selection의 한계에서 온다.**

주의:
- `render.py` RGB sanity render는 VALA의 LLFF hold filename mismatch 때문에 test render가 `0it`로 끝났다. 그러나 결정 실험의 본체인 `gaussian_feature_extractor.py`와 `render_lerf_by_text_langsplat.py`는 `include_feature=True` 경로로 `sparse/0/test.txt` stem 목록을 정상 사용했고, 7개 label frame에 대해 mask render/eval이 완료됐다.

### 5.3 결정 실험 — VALA official `run_sam.py --use_langsplat` feature 재생성

사용자 예약 GPU `cal_jhbae_rtxpro5000_fc0c78ca11ab`로 2026-06-17 실행.

목적: **RGB 3DGS는 ReferSplat HF pretrained checkpoint로 고정하고, 2D language_features만 VALA 공식 `run_sam.py --use_langsplat --get_semantic` 방식으로 재생성**했을 때 ramen이 정정된 논문 3D target에 가까워지는지 확인한다.

기존 feature를 덮어쓰지 않기 위해 격리 루트를 사용했다.

- 격리 source: `external_methods/VALA_repro_root/dataset/3dgs/lerf_ovs/ramen`
- image/sparse: 원본 `data/lerf_ovs/ramen/{images,sparse}` symlink
- 새 feature: `external_methods/VALA_repro_root/dataset/3dgs/lerf_ovs/ramen/langsplat/language_features`
- model dir: `external_methods/VALA/output/refersplat_3dgs_valafeat/lerf_ovs/ramen`
- 실행 script: `scripts/run_vala_official_feature_refersplat_ramen.sh`
- 실행 log: `external_methods/_vala_official_feature_refersplat_ramen_69.log`
- threshold sweep script/log:
  - `scripts/run_vala_official_feature_refersplat_ramen_thresh_sweep.sh`
  - `external_methods/_vala_official_feature_refersplat_ramen_sweep_70.log`
- 후처리 script: `scripts/postprocess_vala_official_feature_refersplat_ramen.sh`

sanity:

- `run_sam.py`가 131 image 전체 처리 완료.
- 생성 feature: **262 `.npy` files** (`131 * {frame_f, frame_s}`)
- shape examples:
  - `frame_00006_f.npy`: `(197, 512)`, `float16`; `frame_00006_s.npy`: `(4, 731, 988)`, `int32`
  - `frame_00024_f.npy`: `(186, 512)`; `frame_00128_f.npy`: `(217, 512)`
- level 1/2/3 language checkpoint 생성 완료:
  - `none/chkpnt30000_langfeat_1_stochastic_gate.pth`
  - `none/chkpnt30000_langfeat_2_stochastic_gate.pth`
  - `none/chkpnt30000_langfeat_3_stochastic_gate.pth`

결과:

| condition | final mIoU | Acc@0.25 | Acc@0.5 |
|---|---:|---:|---:|
| 기존 저장 feature + ReferSplat RGB | 0.3636 | 0.6197 | 0.2817 |
| **VALA official regenerated feature + ReferSplat RGB** | **0.4589** | **0.6761** | **0.4507** |
| 논문 reported ramen 3D | 0.4541 | - | - |

threshold sweep:

| mask_thresh | mIoU | Acc@0.25 | Acc@0.5 |
|---:|---:|---:|---:|
| 0.4 | 0.3850 | 0.6056 | 0.3803 |
| 0.5 | 0.4565 | **0.7324** | 0.4366 |
| 0.6 | **0.4589** | 0.6761 | **0.4507** |
| 0.7 | 0.4025 | 0.5915 | 0.3944 |
| 0.8 | 0.2620 | 0.4225 | 0.1972 |

failure-locus:

| condition | final IoU | precision | recall | area ratio | source IoU | sem-top IoU | final ok | 2D semantic fail | 3D overgrowth | undercoverage | lifting degradation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 기존 저장 feature + ReferSplat RGB | 0.3636 | 0.4492 | 0.7102 | 3.7469 | 0.8550 | 0.4050 | 20/71 | 32 | 7 | 4 | 7 |
| **VALA official feature + ReferSplat RGB** | **0.4589** | **0.5282** | 0.6856 | 3.4508 | 0.8550 | **0.4445** | **32/71** | **23** | **2** | 8 | 5 |

prompt별 잔여 실패:

| prompt | n | final IoU | 주요 locus |
|---|---:|---:|---|
| `corn` | 5 | 0.0191 | 2D semantic fail 5/5 |
| `onion segments` | 7 | 0.0051 | 2D semantic fail 6/7 |
| `sake cup` | 6 | 0.2373 | 2D semantic fail 4/6, undercoverage 2/6 |
| `napkin` | 5 | 0.2853 | 2D semantic fail 3/5 |
| `plate` | 4 | 0.2464 | undercoverage 3/4 |
| `spoon` | 2 | 0.1202 | 3D overgrowth 2/2 |

판정:

> **VALA official feature 재생성은 ramen 갭을 닫는다.** mIoU가 0.3636 → 0.4589로 +9.5pt 오르고, final ok도 20/71 → 32/71로 증가한다. 정정된 논문 3D target이 0.4541이므로 ramen은 이 조건에서 재현권이다. 따라서 기존 THGS 저장 feature 재사용이 VALA paper reproduction을 낮춘 요인인 것은 맞다.
>
> 다만 per-prompt로는 `corn`, `onion segments`, `sake cup`, `napkin`, `plate/spoon` 같은 취약 query가 계속 남는다. 이 분석은 P1 문제정의에는 여전히 유용하지만, VALA 재현성 gap의 중심은 이제 ramen이 아니라 Waldo Kitchen이다.

---

### 5.4 2026-06-18 정정 실험 — Waldo Kitchen이 실제 미재현 scene

논문 Table 1 재확인:

| metric | Figurines | Ramen | Teatime | Waldo Kitchen |
|---|---:|---:|---:|---:|
| VALA 2D mIoU | 59.9 | 51.5 | 70.2 | 65.1 |
| VALA 3D mIoU | 60.38 | 45.41 | 70.61 | 55.71 |

중요 정정:

- `0.604`는 ramen이 아니라 **Figurines 3D 60.38**이다.
- ramen official-feature 조건은 **0.4589**, 논문 3D target **0.4541**과 사실상 일치한다.
- 다른 논문들이 인용하는 Waldo `65.1`은 VALA의 **2D evaluation** 수치다. 3D target은 `55.71`.

threshold 근거:

- 논문 Appendix B: LERF-OVS 2D evaluation은 LERF protocol을 따라 **relevancy map threshold 0.5**.
- 논문 Appendix B: LERF-OVS 3D evaluation은 OpenGaussian protocol을 따라 **Gaussian-text relevancy threshold 0.6**.
- 공식 코드: `external_methods/VALA/scripts/eval_lerf_ovs_3d.sh`가 `mask_thresh=0.6`으로 고정.
- 공식 2D 코드: `external_methods/VALA/eval/evaluate_iou_loc.py`의 activation threshold는 `0.5`가 실질 기준이며 CLI default `0.4`는 README/script에서 명시 실행되지 않아 주의.

Waldo Kitchen 결정 실험:

| condition | eval | threshold | mIoU | target | gap |
|---|---|---:|---:|---:|---:|
| ReferSplat RGB + VALA official SAM/features | 3D | 0.6 | 0.4380 | 0.5571 | -0.1191 |
| ReferSplat RGB + VALA official SAM/features | 2D | 0.5 | 0.4820 | 0.6510 | -0.1690 |
| VALA official RGB train 30k + official features | 3D | 0.6 | 0.4904 | 0.5571 | -0.0667 |
| VALA official RGB train 30k + official features, in-memory 2D eval | 2D | 0.5 | 0.5412 | 0.6510 | -0.1098 |
| VALA official RGB train 30k + official features, in-memory 2D eval | 2D | 0.4 | 0.5575 | 0.6510 | -0.0935 |

Waldo 3D threshold sweep, official RGB train + official features:

| `mask_thresh` | mIoU | Acc@0.25 | Acc@0.5 |
|---:|---:|---:|---:|
| 0.35 | 0.3683 | 0.5000 | 0.3182 |
| 0.40 | 0.3838 | 0.5909 | 0.3182 |
| 0.45 | 0.4006 | 0.5909 | 0.3182 |
| 0.50 | 0.4228 | 0.7273 | 0.3636 |
| 0.55 | 0.4495 | 0.8182 | 0.4091 |
| 0.60 | **0.4904** | **0.8182** | **0.4545** |

Official 2D byte-level reproduction (2026-06-18, Slurm job 83):

목적: 앞의 2D 결과가 우리 in-memory evaluator 때문인지 배제하기 위해, VALA public repo의 official 2D 경로를 직접 실행했다.

- feature rendering: `external_methods/VALA/feature_map_renderer.py`
- 2D eval: `external_methods/VALA/eval/evaluate_iou_loc.py`
- model: `external_methods/VALA/output/official_train_officialsam_waldo/lerf_ovs/waldo_kitchen`
- source: `external_methods/VALA_officialsam_waldo_runs/76/dataset/3dgs/lerf_ovs/waldo_kitchen`
- batch script: `scripts/run_vala_official_2d_eval_waldo.sh`
- renderer wrapper: `scripts/p1e_run_vala_feature_map_renderer_safe.py`
- log: `output/diagnostics/logs/vala_waldo_2d_official_83.{log,err}`
- official eval output: `external_methods/VALA/output/official_2d_eval_waldo_83/eval/`

wrapper/adapter의 범위:

- public `feature_map_renderer.py`는 LERF test camera 이름(`frame_00053`)을 확장자 없이 `torchvision.utils.save_image`에 넘겨 PIL `ValueError: unknown file extension`으로 중단된다. wrapper는 이 PNG side-product 저장 경로에만 `.png`를 붙인다. `.npy` feature map 생성 로직은 수정하지 않았다.
- public `evaluate_iou_loc.py`는 `feat_dir/<scene>_{1,2,3}/train/ours_None/renders_npy/<frame_number-1>.npy` 숫자 파일 구조를 기대한다. official renderer 출력은 `test/ours_30000_langfeat_<level>_stochastic_gate/renders_npy/frame_*.npy` 구조라, 평가 입력 경로만 symlink로 맞췄다. eval 코드는 수정하지 않았다.
- 따라서 이번 실험에서 method/eval core는 public VALA 그대로이고, 추가한 것은 official code가 서로 맞물려 실행되게 하는 I/O compatibility layer다.

결과:

| eval path | threshold | mIoU | target | gap | localization |
|---|---:|---:|---:|---:|---:|
| official `evaluate_iou_loc.py` | 0.5 | **0.5412** | 0.6510 | -0.1098 | 0.8636 |
| official `evaluate_iou_loc.py` | 0.4 | **0.5575** | 0.6510 | -0.0935 | 0.8636 |

official eval의 `chosen_lvl`은 0.5와 0.4에서 동일했다:

```text
[1, 2, 2, 2, 1, 2, 0, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 1, 2]
```

판정:

> Waldo 갭은 **threshold를 낮춰서 해결되는 문제가 아니다**. 논문/공식 스크립트 기준 threshold인 `0.6`이 sweep에서도 best다. official RGB training으로 ReferSplat 대비 +5.2pt 회복되지만 논문 3D target까지는 여전히 -6.7pt이고, 논문/타논문에서 보이는 2D `65.1`까지는 -9~11pt 남는다.
>
> official 2D eval protocol을 직접 태워도 in-memory 2D eval과 같은 값이 나온다. 따라서 낮은 Waldo 2D 점수는 우리 evaluator 구현 문제가 아니다. 현재 가장 강한 결론은 **public VALA code + public/재생성 feature + official RGB train 조건으로는 paper 2D Waldo 65.1을 재현하지 못한다**는 것이다. 남은 후보는 저자-side checkpoint/feature bundle/data split/label preprocessing 차이다.

---

## 6. gotcha 모음 (다른 AI가 반복 안 하도록)

1. **srun은 매번 cwd를 리셋** → `bash -lc "cd <VALA> && export ... && python ..."` 형태로 cd를 *맨 앞에* 박을 것. (이 실수 다발.)
2. **jhsong HOME 권한 없음** → `HOME=/tmp` 또는 NAS, `MPLCONFIGDIR=/tmp/mpl` 설정. matplotlib/HF 캐시 경고/오류 방지.
3. **CLIP 다운로드**: `HF_HOME`을 NAS 경로로 (jhsong 인터넷 됨, 캐시 재사용). eval(render)이 ViT-B-16 laion2b 자동 다운로드.
4. **eval `--ae_ckpt_dir` 필수 인자** (기본 None이면 `os.path.join` 터짐). **dummy 경로** 아무거나 넘기면 됨 (AE는 코드상 주석처리, 실제 로드 안 함).
5. **eval은 chkpnt 이름에 iteration 30000 하드코딩** (`chkpnt30000_langfeat_*_stochastic_gate.pth`). 30k로 학습해야 자연히 맞음.
6. **compute_iou 경로**: `pred_dir/<scene_name>/<ablation>/predictions_mask_<thresh>`. **모델 dir 이름 = scene_name** 이어야 함. 다른 모델은 별도 부모 dir에 scene명으로 둘 것 (예: `output/3dgs_A/lerf_ovs/teatime`).
7. **gsplat `packed=True`는 sm_120에서 illegal memory access** → `packed=False` 필수. VALA 코드는 이미 packed=False 사용 (우리 테스트만 default라 한 번 터졌음).
8. **LangSplat chkpnt 직접 사용 불가** (13-tuple). RGB .ply를 VALA chkpnt로 변환: `scripts/convert_ply_to_vala_chkpnt.py` (load_ply → 모든 param `.cuda()` → training_setup → capture_rgb → save). cuda 강제 안 하면 prune_points에서 device 오류.
9. **simple_knn에 `__init__.py` 없음** → editable 설치 시 import 안 됨. 빈 `__init__.py` 추가.
10. **학습을 스킵한 모델 dir에는 `cfg_args`를 직접 만들어야 함**. 없으면 `get_combined_args()`가 `FileNotFoundError`로 죽고, 빈 `Namespace()`만 넣으면 sentinel defaults 때문에 `depths=None`, `resolution=None`이 되어 `depth_params.json` 또는 camera downscale에서 다시 죽는다. ReferSplat 실험에서는 기존 VALA `cfg_args` 형식에 맞춰 `depths=''`, `resolution=-1`, `images='images'` 등을 명시했다.
11. **VALA `render.py` RGB sanity는 LERF `test.txt` stem split을 쓰지 않음**. `include_feature=False` 경로에서 LLFF hold list가 `.jpg` 이름으로 만들어져 `image_name` stem과 mismatch되어 test render가 `0it`로 끝날 수 있다. 반면 `gaussian_feature_extractor.py`/`render_lerf_by_text_langsplat.py`는 `include_feature=True`라 `sparse/0/test.txt` stem 목록을 정상 사용한다.
12. **official 2D eval은 public code끼리 I/O가 바로 맞지 않는다**. `feature_map_renderer.py`는 extensionless `frame_00053` PNG 저장에서 죽고, `evaluate_iou_loc.py`는 numeric `<frame_number-1>.npy` 입력을 기대한다. `scripts/p1e_run_vala_feature_map_renderer_safe.py`는 PNG side-product 확장자만 보정하고, `scripts/run_vala_official_2d_eval_waldo.sh`는 symlink로 경로/인덱스만 맞춘다. core feature/eval logic은 수정하지 않는다.

---

## 7. 산출물 / 경로

| 경로 | 내용 |
|---|---|
| `envs/vala-port/` | portable env (모든 노드 재사용) |
| `external_methods/VALA/` | VALA 레포 (changandao/VALA) |
| `external_methods/VALA/output/3dgs/lerf_ovs/{ramen,teatime}/` | 우리 자체학습 모델 + langfeat + eval |
| `external_methods/VALA/output/3dgs_A/lerf_ovs/teatime/` | LangSplat dense(2.2M) 변환 모델 (A) |
| `external_methods/VALA/output/3dgs_dense/lerf_ovs/ramen/` | ramen densify 0.0002 (589k) |
| `_langsplat_dl/teatime_unzip/teatime/teatime_{1,2,3}/` | LangSplat pretrained teatime (다운로드) |
| `external_methods/_vala_*.log` | 학습/feature/eval 로그 |
| `scripts/convert_ply_to_vala_chkpnt.py` | LangSplat .ply → VALA chkpnt 변환기 |
| `scripts/p1e_vala_regime_split.py` | VALA per-prompt IoU → easy/phantom regime split |
| `output/diagnostics/p1e_vala_regime_ramen.csv` | ramen regime split (단 0.358 broken 모델 기반) |
| `scripts/p1e_vala_source_diagnostics.py` | LERF GT 대비 2D SAM/LangSplat source mask oracle |
| `scripts/p1e_vala_2d_semantic_diagnostics.py` | 2D mask feature에서 CLIP text top 선택이 GT mask를 고르는지 진단 |
| `scripts/p1e_vala_mask_diagnostics.py` | 최종 VALA silhouette의 IoU/precision/recall/area ratio 진단 |
| `scripts/p1e_vala_failure_locus.py` | source oracle + 2D semantic top + final mask를 합쳐 실패 위치 분류 |
| `scripts/p1e_vala_3d_relevance_diagnostics.py` | VALA 3D language checkpoint의 prompt별 relevance/selected Gaussian 비율 진단 |
| `output/diagnostics/p1e_vala_*diagnostics*.csv` | 위 CPU 진단들의 주요 산출물 |
| `output/diagnostics/p1e_vala_failure_locus*.csv` | query/prompt/scene별 실패 위치 분해표 |
| `output/diagnostics/p1e_vala_3d_relevance_*chosen.csv` | prompt별 chosen level의 3D relevance 분포 |
| `output/diagnostics/p1e_vala_failure_locus_by_prompt_with_3d_relevance.csv` | prompt별 실패 위치 + chosen level/selected Gaussian 비율 결합표 |
| `scripts/convert_vala_chkpnt_to_ply.py` | ReferSplat/VALA 12-tuple RGB checkpoint → `point_cloud.ply` 변환 |
| `scripts/run_vala_refersplat_ramen.sh` | ReferSplat RGB 3DGS 결정 실험 Slurm batch script |
| `scripts/run_vala_refersplat_ramen_thresh_sweep.sh` | ReferSplat RGB 3DGS의 VALA `mask_thresh` sweep batch script |
| `output/diagnostics/p1e_vala_refersplat_decision_summary.csv` | self-train/dense/ReferSplat RGB 비교 요약 |
| `output/diagnostics/p1e_vala_refersplat_threshold_sweep.csv` | ReferSplat RGB 3DGS `mask_thresh` sweep 요약 |
| `external_methods/VALA_repro_root/dataset/3dgs/lerf_ovs/ramen/langsplat/language_features/` | VALA official `run_sam.py --use_langsplat`로 재생성한 ramen feature (원본 feature와 격리) |
| `external_methods/VALA/output/refersplat_3dgs_valafeat/lerf_ovs/ramen/` | ReferSplat RGB + official regenerated feature 조건의 VALA model/eval output |
| `scripts/run_vala_official_feature_refersplat_ramen.sh` | official feature 재생성 + feature extraction + eval batch script |
| `scripts/run_vala_official_feature_refersplat_ramen_thresh_sweep.sh` | official feature 조건의 `mask_thresh` sweep batch script |
| `scripts/postprocess_vala_official_feature_refersplat_ramen.sh` | official feature 조건의 source/semantic/final/failure-locus 후처리 |
| `output/diagnostics/p1e_vala_official_feature_2d_source_oracle.csv` | official feature 기준 2D source oracle |
| `output/diagnostics/p1e_vala_official_feature_2d_semantic_selection.csv` | official feature 기준 2D semantic top 진단 |
| `output/diagnostics/p1e_vala_official_feature_refersplat_mask_diagnostics.csv` | official feature + ReferSplat RGB final mask 진단 |
| `output/diagnostics/p1e_vala_official_feature_refersplat_failure_locus*.csv` | official feature + ReferSplat RGB failure locus |
| `output/diagnostics/p1e_vala_official_feature_refersplat_threshold_sweep.csv` | official feature + ReferSplat RGB threshold sweep 요약 |
| `scripts/run_vala_official_train_waldo_decision.sh` | Waldo official RGB train 30k + official feature assignment + 3D/2D eval batch script |
| `scripts/run_vala_waldo_3d_thresh_sweep.sh` | Waldo official train 조건의 3D `mask_thresh` sweep |
| `scripts/p1e_vala_2d_eval_inmemory.py` | VALA 2D evaluation을 in-memory로 재현한 진단 스크립트 |
| `scripts/run_vala_official_2d_eval_waldo.sh` | Waldo official 2D `feature_map_renderer.py` + `evaluate_iou_loc.py` reproduction batch script |
| `scripts/p1e_run_vala_feature_map_renderer_safe.py` | official renderer의 extensionless PNG side-product 저장 버그만 우회하는 wrapper |
| `external_methods/VALA/output/official_train_officialsam_waldo/lerf_ovs/waldo_kitchen/` | Waldo official RGB train 30k + langfeat model/eval output |
| `external_methods/VALA/output/official_2d_eval_waldo_83/eval/` | official 2D eval output (`thresh_0.5`, `thresh_0.4`) |
| `output/diagnostics/logs/vala_waldo_2d_official_83.{log,err}` | official 2D reproduction Slurm job 83 로그 |

---

## 8. 연구적 의미 (P1-E 관점)

- **달성**: "외부 method(VALA)를 이 클러스터에서 실제 실행해 per-paper 실측" 이 가능함을 입증 (teatime 재현). portable env·빌드 레시피는 다음 외부 method(Segment-then-Splat 등)에 그대로 재사용.
- **regime split (ramen, 0.358 broken 모델)**: VALA가 phantom +14pt / easy −12pt (vs THGS) — 단 ① cross-pipeline confound ② broken ramen 기반이라 **신뢰 보류**. ramen 재현이 고쳐진 뒤 재측정 필요. 진짜 mechanism 검증은 V6 (같은 VALA 위 robust vs mean, [p1e_external_code_study_plan.md](p1e_external_code_study_plan.md) §6.1).
- **현재 결론**: ramen은 Table 1 정정 후 3D target 0.4541을 사실상 재현한 것으로 본다. 진짜 미재현 scene은 Waldo Kitchen이다. official RGB train + official feature + official 2D eval까지 태워도 2D 0.5412@0.5 / 0.5575@0.4로 paper 0.651과 -9~11pt 차이가 남는다. 다음 재현성 추적은 ① 저자-side checkpoint/feature bundle 존재 여부 ② data split/label preprocessing 차이 ③ 다른 논문들이 사용한 VALA Waldo 평가 protocol 역추적으로 좁혀야 한다.

---

## 9. 진행 로그

| 날짜 | 사건 |
|---|---|
| 2026-06-15 | portable env 구축, sm_120 PTX-JIT 빌드 검증, V0 PASS, teatime 재현 성공(A/B), ramen 갭 발견, pretrained·밀도·CLIP·해상도·포맷 전부 기각. ramen-특이 문제로 미해결 마감. |
| 2026-06-15 | CPU 추가 진단 실행: 2D source oracle은 ramen 0.855로 충분, 2D semantic top은 ramen 0.405로 낮음, final mask는 low-precision overgrowth, dense ramen도 동일. 결론을 "단순 3DGS 학습 문제"가 아니라 "2D semantic ambiguity + 3D lifting/render leakage 증폭"으로 업데이트. |
| 2026-06-17 | ReferSplat HF `ramenchkpnt30000.pth`로 RGB 3DGS만 교체하는 결정 실험 실행. mIoU 0.3636으로 기존 0.3580과 동일권, failure locus도 동일. "우리 3DGS 학습 실패 단독 원인" 가설 사실상 기각. |
| 2026-06-17 | VALA official `run_sam.py --use_langsplat --get_semantic`로 ramen feature를 격리 루트에 재생성. ReferSplat RGB와 결합해 mIoU **0.4589** 달성(+9.5pt), threshold sweep best도 0.4589. 2026-06-18 Table 1 정정 후에는 ramen 3D target 0.4541 재현권으로 판정. |
| 2026-06-18 | VALA Table 1을 재확인해 `0.604`가 ramen이 아니라 Figurines 3D 60.38 오독임을 정정. ramen 3D target은 45.41이므로 official feature + ReferSplat RGB 0.4589는 재현권. 실제 미재현 scene을 Waldo Kitchen으로 전환. |
| 2026-06-18 | Waldo official RGB train 30k + official feature assignment 조건에서 3D **0.4904@0.6** 확인. 3D threshold sweep도 0.6이 best라 threshold 가설 기각. |
| 2026-06-18 | public VALA official 2D path(`feature_map_renderer.py` + `evaluate_iou_loc.py`)를 I/O compatibility layer만 붙여 재실행. 결과 **0.5412@0.5 / 0.5575@0.4**, in-memory eval과 동일. 우리 evaluator 문제가 아니라 public code/data 재현성 갭으로 판정. |
