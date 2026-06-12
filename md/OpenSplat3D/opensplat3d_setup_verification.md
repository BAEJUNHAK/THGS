# OpenSplat3D 환경 setup 검증 (THGS 머신 / Blackwell sm_120)

> Generated 2026-06-08. OpenSplat3D (CVPRW 2025, arXiv:2506.07697) 의 LERF-OVS 실험을 우리 머신에서 돌리기 위한 환경 setup 결과.
>
> **본 문서는 환경 setup 검증만 다룬다. 실험(학습/diagnostic)은 별도 진행.**

---

## 0. TL;DR — 환경 setup 완료 (별도 conda env 분리 구축)

| 항목 | 결과 |
|--|--|
| 가상환경 | **`opensplat3d` conda env** (thgs env에서 clone 후 의존성 추가) |
| Python | 3.10.13 (thgs와 동일) |
| PyTorch | **2.11.0+cu128** (sm_120 Blackwell native 지원) |
| 내부 nvcc | 12.8 (cu128 매칭) |
| GPU | NVIDIA RTX PRO 5000 Blackwell (sm_120) ✅ |
| CUDA extension | dgr / simple_knn / fused_ssim — sm_90+PTX 빌드, sm_120 JIT 동작 ✅ |
| 외부 의존성 | segment-anything 1.0, cuml-cu12 25.4, opencv 4.13, clip (OpenAI git) 모두 OK ✅ |
| OpenSplat3D import | `gaussian_renderer.render`, `eval.eval_lerf_ovs`, `language.LanguageModel`, `cluster.hdbscan` 모두 OK ✅ |
| GPU op | image-tensor permute + `GaussianRasterizationSettings/Rasterizer` 로드 OK ✅ |
| **thgs env 분리** | **thgs env는 그대로 보존 — spt/grid_graph/nag_data 모두 동작, numpy 1.26.4 유지** ✅ |

---

## 1. 왜 OpenSplat3D 공식 환경(uv venv)이 안 됐는가

OpenSplat3D 공식 `uv sync --extra compile` 은 **3가지 호환성 문제**가 겹쳐 실패.

### 1.1 시스템 CUDA = 12.4, Blackwell GPU = sm_120

- nvcc 12.4는 **`compute_120` arch를 모름** (`nvcc fatal: Unsupported gpu architecture 'compute_120'`).
- sm_120 native 빌드를 하려면 nvcc 12.6+ 필요.
- 우회: `TORCH_CUDA_ARCH_LIST="8.6 8.9 9.0+PTX"` 로 빌드 → sm_90 PTX가 sm_120 GPU에서 JIT compile 됨.

### 1.2 PyTorch 2.5–2.7 stable은 sm_120 미지원

- OpenSplat3D `pyproject.toml`은 torch 2.5.1 pin → sm_90까지가 한계, segment_anything 첫 forward에서 `RuntimeError: CUDA error: no kernel image is available`.
- PyTorch 2.7.1+cu126도 sm_120 native 미지원, 단순 permute에서도 같은 에러.
- PyTorch 2.8+ stable이 sm_120 native 지원하나 cu126 stable에서 사용 가능.

### 1.3 cu126/cu128/cu130 wheel의 strict CUDA version check

- torch 2.12.0+cu130 → `_check_cuda_version` 이 시스템 12.4 vs 13.0 major mismatch 거부.
- torch 2.8.0+cu124 wheel 부재.

→ **OpenSplat3D `.venv` 안에서 어떤 torch 조합으로도 sm_120 동작이 안 됨.**

---

## 2. 우회 전략 — thgs env를 base로 별도 conda env 구축

CLAUDE.md에 명시된 `thgs` env는 다음을 만족:

| 검증 항목 | 값 |
|--|--|
| Python | 3.10.13 |
| torch | 2.11.0+cu128 |
| nvcc | **12.8** (env 내부, cu128 매칭) |
| sm_120 GPU op | ✅ 동작 (CLAUDE.md ramen 평가 검증) |

→ **thgs를 직접 수정하지 않고 clone해서 별도 `opensplat3d` env 구축.**

### 2.1 thgs env 영향 없음 (분리 검증)

```python
# opensplat3d env: numpy 2.0.2 + opencv 4.13 + cuml + dgr + segment-anything + clip ...
# thgs env: numpy 1.26.4 + opencv 4.7.0 (원본 보존)
```

→ 두 env는 완전 독립. opensplat3d env 작업이 thgs env에 영향 0.

---

## 3. 재현 명령 (opensplat3d env 구축)

```bash
source ~/miniforge3/etc/profile.d/conda.sh

# 1. thgs를 clone해서 동일한 torch/cuda/sm_120 동작 base 확보 (~12분, 15GB NAS copy)
conda create --name opensplat3d --clone thgs

# 2. opensplat3d env 활성화
conda activate opensplat3d

# 3. Python deps (cuml-cu12가 numpy 1.26 → 2.0으로 다운그레이드 — 이 env에서만 영향)
pip install 'cuml-cu12==25.4.*' segment-anything tabulate omegaconf accelerate viser \
    sentencepiece 'open-clip-torch' munch wandb open3d pykeops \
    'transformers>=4.51.3' rerun-sdk \
    'clip @ git+https://github.com/openai/CLIP.git' \
    --extra-index-url https://pypi.nvidia.com/

# 4. opencv 업그레이드 (numpy 2.0 호환)
pip install --upgrade 'opencv-python>=4.10'

# 5. CUDA extensions 재빌드 (clone된 numpy 1.x ABI → numpy 2.0 ABI로 갱신)
export TORCH_CUDA_ARCH_LIST="8.6 8.9 9.0 12.0+PTX"
pip install --no-build-isolation --no-cache-dir --force-reinstall --no-deps \
    /mnt/pilab_nas/projects/THGS/opensplat3d/submodules/diff-gaussian-rasterization \
    /mnt/pilab_nas/projects/THGS/opensplat3d/submodules/simple-knn \
    'fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@30fb258c8a38fe61e640c382f891f14b2e8b0b5a'
```

### 외부 체크포인트 (이미 다운로드 완료)

| 파일 | 경로 | 크기 |
|--|--|--|
| SAM ViT-H | `opensplat3d/ckpts/sam_vit_h_4b8939.pth` | 2.4 GB |
| MasQCLIP base_novel | `opensplat3d/ckpts/MasQCLIP/base_novel.pth` | ~1.5 GB |

---

## 4. 검증 결과 (Final sanity check)

### 4.1 opensplat3d env

```bash
conda activate opensplat3d
cd /mnt/pilab_nas/projects/THGS/opensplat3d
PYTHONPATH=. python -c "..."
```

```
=== opensplat3d env verification ===
torch: 2.11.0+cu128
cuda: True | device: NVIDIA RTX PRO 5000 Blackwell
dgr OK
simple_knn OK
fused_ssim OK
segment_anything, clip, cuml HDBSCAN OK
=== ALL OPENSPLAT3D IMPORTS OK ===
Blackwell sm_120 GPU op OK: torch.Size([3, 720, 988])
numpy: 2.0.2
opencv: 4.13.0
```

### 4.2 thgs env (분리 확인, 손상 없음)

```
=== THGS core (spt+grid_graph+nag_data+vlm_utils) OK ===
numpy: 1.26.4 | opencv: 4.7.0 | torch: 2.11.0+cu128
sm_120 GPU: torch.Size([3, 10])
```

→ 두 env 완전 분리. 한쪽 작업이 다른 쪽에 영향 0.

---

## 5. 사용 방법

```bash
# OpenSplat3D 작업 (학습/평가)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate opensplat3d
export CUDA_VISIBLE_DEVICES=2
cd /mnt/pilab_nas/projects/THGS/opensplat3d
export PYTHONPATH=$PYTHONPATH:$PWD

# 예: 학습 (figurines)
python opensplat3d/train.py \
    model.source_path=/mnt/pilab_nas/projects/THGS/data/lerf/figurines \
    model.model_path=/mnt/pilab_nas/projects/THGS/opensplat3d/output/scenes/figurines \
    --config configs/lerf.yaml

# 예: 네이티브 평가
LERF_OVS_LABEL_PATH=/mnt/pilab_nas/projects/THGS/data/lerf/label \
python opensplat3d/eval/eval_lerf_ovs.py output/scenes/figurines

# 예: Oracle/Actual diagnostic
LERF_OVS_LABEL_PATH=/mnt/pilab_nas/projects/THGS/data/lerf/label \
python scripts/lerf_ovs_diagnostic_native.py \
    output/scenes/figurines \
    --out-csv /mnt/pilab_nas/projects/THGS/output/diagnostics/lerf_ovs_opensplat3d_native_h.csv

# THGS 작업 (기존 그대로)
conda activate thgs
cd /mnt/pilab_nas/projects/THGS
# ramen 평가, sam_oracle 등 그대로 동작
```

---

## 6. 환경별 패키지 차이 (요약)

| 패키지 | thgs env | opensplat3d env |
|--|--|--|
| Python | 3.10.13 | 3.10.13 |
| torch | 2.11.0+cu128 | 2.11.0+cu128 |
| nvcc | 12.8 | 12.8 |
| numpy | **1.26.4** | **2.0.2** |
| opencv-python | **4.7.0.72** | **4.13.0.92** |
| pandas | 2.3.3 | 2.2.3 |
| numba | 0.65.1 | 0.60.0 |
| simple_knn | THGS 빌드 (numpy 1.x ABI) | 재빌드 (numpy 2.0 ABI) |
| diff_surfel_rasterization | ✅ (THGS, 2DGS용) | ❌ (불필요) |
| diff_gaussian_rasterization | ❌ | ✅ (OpenSplat3D, 3DGS용) |
| fused_ssim | ❌ | ✅ |
| cuml-cu12 | ❌ | ✅ 25.4 |
| segment-anything | ❌ | ✅ 1.0 |
| clip (OpenAI) | ❌ | ✅ |
| spt + grid_graph + nag_data | ✅ (THGS 핵심) | ❌ (불필요) |

---

## 7. 산출물

| 파일 | 상태 |
|--|--|
| conda env `opensplat3d` (18GB) | ✅ 검증 완료 |
| [opensplat3d/scripts/lerf_ovs_diagnostic_native.py](../../opensplat3d/scripts/lerf_ovs_diagnostic_native.py) | ✅ paper Algorithm 6 step 일치 native diagnostic |
| [opensplat3d/ckpts/sam_vit_h_4b8939.pth](../../opensplat3d/ckpts/sam_vit_h_4b8939.pth) | ✅ 다운로드 |
| [opensplat3d/ckpts/MasQCLIP/base_novel.pth](../../opensplat3d/ckpts/MasQCLIP/base_novel.pth) | ✅ 다운로드 |
| `data/lerf/figurines/sam/` | ⏳ (실험 보류) |
| `output/scenes/<scene>/` | ⏳ (학습 보류) |
| `output/diagnostics/lerf_ovs_opensplat3d_native_h.csv` | ⏳ (실험 보류) |

---

## 8. 다음 단계 (사용자 승인 시)

1. SAM mask extraction (4 scenes)
2. OpenSplat3D 학습 (train.py + 자동 HDBSCAN + MasQCLIP)
3. Native LERF-OVS 평가 (Actual baseline)
4. Oracle/Actual diagnostic 실행 → CSV
5. cross_method 분석: `md/cross_method/lerf_ovs_thgs_vs_relags_vs_opensplat3d.md`
