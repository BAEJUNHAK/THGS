# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

THGS(Training-Free Hierarchical Scene Understanding for Gaussian Splatting with Superpoint Graphs)는 학습 없이 2DGS 장면에서 open-vocabulary 계층적 3D 분할을 수행한다. 2D Gaussian Splatting 장면 위에 슈퍼포인트 그래프를 계층적으로 구축하고, 각 레벨에 시맨틱 피처를 할당한다.

## 환경 설정

```bash
conda env create -f environment.yml
conda activate thgs
pip install pyg_lib torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
python scripts/setup_dependencies.py build_ext
```

CUDA 11.8, PyTorch 2.2, Python 3.10 필요. 기본 설정은 24GB GPU 기준.

## 주요 명령어

### 전체 파이프라인 실행 (config 내 모든 장면)
```bash
bash scripts/run.sh configs/lerf.yml
```

### 특정 장면만 실행
```bash
bash scripts/run.sh configs/lerf.yml figurines ramen
```

### 개별 파이프라인 단계 (launcher 경유)
```bash
python scripts/launcher.py -f sp_partition.py -cf configs/lerf.yml         # 1. 인접 그래프 구축
python scripts/launcher.py -f graph_weight.py -cf configs/lerf.yml         # 2. SAM 기반 엣지 가중치 조정
python scripts/launcher.py -f sp_partition.py -cf configs/lerf.yml -k      # 3. 슈퍼포인트 분할 (-k 플래그)
python scripts/launcher.py -f merge_proj.py -cf configs/lerf.yml           # 4. 계층적 병합 + 피처 재투영
```

### 장면별 언어 피처 생성
```bash
python scripts/image_encoding.py --source_path data/lerf/figurines
```

### 평가 (LERF-OVS)
```bash
for sc in figurines ramen teatime waldo_kitchen; do
    python test_lerf.py -s data/lerf/$sc -m output/lerf/$sc --path_pred output/render/lerf
done
python scripts/eval_seg.py --dataset lerf --scene_list figurines ramen teatime waldo_kitchen \
    --path_pred output/render/lerf --path_gt data/lerf/label
```

### GUI 시각화
```bash
python gui/main.py --config gui/configs.yaml
```

## 아키텍처

### 파이프라인 단계 (`scripts/run.sh`가 순서대로 실행)

1. **`sp_partition.py`** - FRNN(Fast Radius Nearest Neighbors)으로 가우시안 중심점 인접 그래프를 구축한 뒤, `ext/spt/`의 parallel cut pursuit 그래프 컷 알고리즘으로 슈퍼포인트를 분할한다. `-k` 없이 한 번 실행(그래프 구축), `-k` 붙여서 한 번 실행(분할).

2. **`graph_weight.py`** - SAM 기반 대조적 단서로 그래프 엣지 가중치를 재조정한다. 각 뷰에서 가우시안 피처를 렌더링하고, SAM 마스크를 positive/negative supervision으로 사용해 인접 가우시안 간 유사도를 계산한다.

3. **`merge_proj.py`** - 슈퍼포인트를 점진적으로 병합해 계층적 다중 레벨 슈퍼포인트 그래프(`sai_nag.pt`)를 생성하고, CLIP 기반 시맨틱 피처를 각 레벨에 재투영한다. SAI3D에서 파생.

### 핵심 모듈

- **`scene/`** - 장면 로딩 및 가우시안 모델. `gaussian_model.py`가 메인 3DGS 모델이고, `semantic_model.py`가 시맨틱 임베딩을 추가한다.
- **`gaussian_renderer/`** - 미분 가능한 서펠 래스터라이저(`submodules/diff-surfel-rasterization` 경유). `render`, `render_point`, `trace` 함수 제공.
- **`arguments/`** - CLI 인자 파싱. `ModelParams`, `PipelineParams`, `OptimizationParams` 파라미터 그룹 정의.
- **`ext/spt/`** - 포인트 클라우드 분할용 Superpoint Transform(SPT) 라이브러리. Hydra 설정, PyTorch Geometric, parallel cut pursuit 사용.
- **`nag_data.py`** - 추론/테스트 시 계층적 슈퍼포인트 그래프 데이터를 로드하는 `SemanticNAG` 클래스.
- **`utils/vlm_utils.py`** - open-vocabulary 쿼리를 위한 CLIP 유사도 측정.
- **`utils/sai3d_utils.py`** - 슈퍼포인트 병합 유틸리티(SAI3D 유래).

### 설정 파일 (`configs/`)

YAML 파일로 데이터셋 경로, 장면 목록, 모듈별 파라미터(`graph_weight`, `merge_proj`, `spt` 섹션)를 정의한다. LERF-OVS(`lerf.yml`)와 3DOVS(`3dovs.yml`) 두 데이터셋을 지원.

### 서브모듈

- `submodules/diff-surfel-rasterization` - 2D Gaussian Splatting용 커스텀 CUDA 래스터라이저
- `submodules/simple-knn` - KNN 구현
- `ext/spt/dependencies/FRNN` - Fast Radius Nearest Neighbors (CUDA)
- `ext/spt/dependencies/parallel_cut_pursuit` - 슈퍼포인트 분할용 그래프 컷 알고리즘

### 데이터 흐름

장면 데이터: `data/<dataset>/<scene>/` -> 모델 출력: `output/<save_folder>/<scene>/`
파이프라인의 최종 산출물은 `sai_nag.pt`(시맨틱 피처가 포함된 계층적 슈퍼포인트 그래프)이며, 모델 출력 디렉토리에 저장된다. `test_lerf.py`와 GUI가 이 파일을 로드해 쿼리를 수행한다.
