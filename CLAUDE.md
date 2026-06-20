# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 현재 머신 (pilab Blackwell 클러스터) — SLURM 예약제, 환경 셋업됨

랩 GPU는 **Calendar 예약 → Slurm 실행** 방식으로 운영된다 (공식 "PILAB GPU 사용 가이드", 2026-06-14). 흐름: **Google Calendar 일정 생성 → (최대 5분) Slurm Reservation 자동 생성 → Slack DM으로 Reservation ID 수신 → srun/sbatch 실행 → NAS 저장**. **직접 점유(`CUDA_VISIBLE_DEVICES=N` 고정) 방식은 쓰지 말 것** — 반드시 예약을 통해 GPU를 받는다.

사용 가능 GPU: RTXPRO5000 5대, RTXA6000 2대 (+ ABYSS 4~7, B200-8은 캘린더 등록 후 개별 사용). 본 클러스터(`gpu` 파티션)의 RTXPRO5000 노드는 `pilab-gpu-b`, `jinyoung-ROMED8-2T`, `jhsong-TRX50-AI-TOP` — 전부 RTX PRO 5000 Blackwell sm_120 (cap 12.0, 48GB).

### 예약 만들기 (Google Calendar)
- 캘린더: **pilgpu47@gmail.com** (PILAB GPU 예약 캘린더)
- 제목 형식: `[<SLURM_USER>] <GPU_TYPE> <GPU_COUNT>` — 예: `[jhbae] RTXPRO5000 1`. GPU_TYPE은 `RTXPRO5000`/`RTXA6000` (띄어쓰기 없이), GPU_COUNT는 `1` 또는 `2`.
- **Description 필수** (예약 사유/작업 내용). 비어 있으면 `[REJECTED]` 처리되어 reservation이 안 생긴다.
- 한 이벤트는 GPU 타입 1종만. 정상 생성 시 Slack DM으로 `cal_<user>_<gputype>_<date>_<time>_<id>` 형식 Reservation ID가 온다.

### 실험 시작 절차 (반드시 이 방식)
0. **현재 예약 자동 확인** — Reservation ID는 매번 바뀌므로 문서에 박지 말고 실행 시점에 조회한다. jhbae의 ACTIVE 예약을 한 줄로:
   ```bash
   RID=$(scontrol show reservation -o | grep 'Users=.*jhbae' | grep 'State=ACTIVE' | grep -oP 'ReservationName=\K\S+' | head -1); echo "$RID"
   ```
   `$RID`가 비어 있으면 활성 예약이 없는 것 → Google Calendar로 예약부터 만들 것(아래 "예약 만들기"). 배정 노드/종료시각은 `scontrol show reservation=$RID`로 확인.
1. **Reservation ID 확보** — 위 `$RID` (또는 Slack DM 값)을 아래 명령의 `--reservation`에 사용.
2. **예약 노드로 진입** — 예약마다 배정 노드가 다를 수 있다 (현재 로그인 노드와 다를 수 있음). GPU 종류는 Calendar 제목으로 결정되고, 실행 시엔 `--gres=gpu:<개수>`로 **개수만** 요청한다(예약한 개수와 반드시 일치):
   ```bash
   # 인터랙티브
   srun -p gpu --reservation=<RESERVATION_ID> --gres=gpu:1 --pty bash
   # 배치
   sbatch -p gpu --reservation=<RESERVATION_ID> --gres=gpu:1 job.sh
   ```
3. **NAS 공유 env 활성화** (모든 노드에서 동일):
   ```bash
   source /mnt/pilab_nas/projects/THGS/envs/thgs/bin/activate
   cd /mnt/pilab_nas/projects/THGS
   ```
   - `CUDA_VISIBLE_DEVICES`를 수동 설정하지 말 것 — SLURM이 `--gres`로 자동 할당한다.
   - 코드(`/mnt/pilab_nas/projects/THGS`)와 env(`.../THGS/envs/thgs`)는 모두 NFS 공유라 어느 노드에서든 동일하게 동작. env는 torch 2.11.0+cu128 / Python 3.10, CUDA 확장(diff-surfel-rasterization, simple-knn, torch_scatter/cluster) 검증 완료.
4. **작업 종료** — shell `exit` / job은 `squeue -u $USER`로 확인, `scancel <job_id>`로 취소. 안 쓸 예약은 Calendar에서 수정·삭제해 남에게 양보.

> ⚠️ 이전의 `source ~/miniforge3/...; conda activate thgs`는 **로컬 디스크 env**라 예약 노드에서는 없다. 위 NAS 경로 activate만 사용할 것.

### 랩 운영 방침 (공식 PILAB GPU 가이드 기준)
- **1인 동시 사용 최대 2 GPU** (타 연구원 일정에 따라 유동적).
- **예약 기간 최대 7일 이내** (종일 예약 가능). 7일 초과 시 `[REJECTED]`.
- 예약 시 **Description 필수**, 안 쓸 예약은 즉시 정리.
- (참고: 상위 Notion 초안에 적힌 "1회 12시간 / 연속예약 금지 / 월 균등분배"는 **공식 가이드에는 없는** 초안 문구다. 공식 기준은 위 3가지 — 동시 2개, 최대 7일, Description 필수.)

### 계정 & 데이터 정책 (관리자 확인 2026-06-15)
- **정식 계정 = `jhbae`** (NAS/LDAP). 로컬 `meaabebe` 계정은 시범기간 후 **약 2026-06-22 삭제 예정** — SSH 설정을 jhbae 기준으로 옮겨둘 것. NAS의 코드·env(`/mnt/pilab_nas/projects/THGS`, uid 1024 소유)는 계정 삭제와 무관하게 유지된다.
- **데이터는 NAS 중앙 저장 원칙**: 예약 노드가 매번 바뀌므로 원본은 NAS에 두고, 필요 시 로컬로 복사해 사용한다. ("전부 NAS에서만 실행"이 아니라 "원본 NAS + 작업 로컬복사 OK".)

ramen 평가는 검증 완료 (mIoU 0.42141, mAcc 0.96094 — 로컬 결과와 0.16% 이내 일치). env를 NAS에 재배포해야 하면 `envs/thgs.tar.gz`(conda-pack 산출물)를 풀고 `conda-unpack` 하면 된다. 아래 "환경 설정" 섹션은 **로컬 RTX 4060 Ti(cu118)용**이므로 이 클러스터에는 적용하지 말 것.

### language_features (LERF-OVS 4 scene) — 재생성 불필요, 재사용할 것

`data/lerf_ovs/<scene>/language_features/` 에 per-view SAM seg map (`*_s.npy`) + per-mask CLIP feature (`*_f.npy`) 가 **생성 완료되어 있다** (2026-06-10, scene당 이미지 수 × 2 파일, 총 ~9GB). 이건 `merge_proj.py`/Stage 3 replay 등 파이프라인 재실행의 필수 입력이며 **한 번 만드는 데 GPU ~4시간**이 걸리므로 지우거나 다시 만들지 말 것.

재생성이 정말 필요하면 (`python -u scripts/image_encoding.py --source_path data/lerf_ovs/<scene>`):
- **표준 `segment-anything` 으로는 안 된다** — `generate()` 가 4-level 튜플을 반환하는 LangSplat 변형이 필요: `pip install --force-reinstall --no-deps --no-build-isolation "git+https://github.com/minghanqin/segment-anything-langsplat.git"`
- SAM ckpt 는 `ckpts/sam_vit_h_4b8939.pth` (→ `opensplat3d/ckpts/` 로의 심링크, 이미 존재)
- 진행 로그는 `output/diagnostics/logs/encode_<scene>.log` 패턴 사용 (스크립트의 bare except 가 에러를 삼키는 버그는 traceback 출력 + 재시도 3회 상한으로 패치되어 있음)

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
