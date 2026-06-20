# 정전 후 pilab-gpu-b 서버 재접속 가이드

> 이 문서의 원본은 Notion에도 복사돼 있다 (서버가 꺼져 있으면 이 파일을 못 보므로).

## ⚡ 빠른 버전 — 평소 쓰던 컴퓨터로 다시 접속할 때 (대부분 이 경우)

SSH 설정은 내 컴퓨터에 이미 저장돼 있으므로 아래 5단계가 전부다:

1. 연구실에서 **서버(pilab-gpu-b) 전원 버튼** 누르기 → 1~2분 대기
2. **VSCode 열기** → 왼쪽 아래 파란 `><` 아이콘 (또는 `F1` → "Remote-SSH: Connect to Host...") → `pilab-gpu-b` 선택
3. **비밀번호 입력**
4. 폴더는 보통 자동으로 다시 열림. 안 열리면 File → Open Recent → `THGS [SSH: pilab-gpu-b]`
5. 터미널에서 `claude --continue` → 클로드 코드 이전 대화 이어서 (대화 내역은 서버 디스크에 저장돼 있어 정전에도 안 사라짐)

아래의 상세 단계(확장 설치, config 추가)는 **새 컴퓨터에서 처음 접속할 때만** 필요하다.

---

## 0. 서버 기본 정보

| 항목 | 값 |
|---|---|
| 서버 이름 | `pilab-gpu-b` (연구실 GPU 서버 B, RTX PRO 5000 ×3) |
| 연구실 내부 IP | `192.168.0.100` |
| SSH 포트 | `7100` (기본 22 아님) |
| 계정 | `meaabebe` |
| Tailscale IP (외부 접속용) | `100.64.0.4` (장치명 `pild`) |
| 작업 폴더 | `/mnt/pilab_nas/projects/THGS` (NAS `192.168.0.203:/volume1/projects` NFS 마운트) |

## 1. 서버 전원 켜기

정전 후 자동으로 안 켜질 수 있음. 연구실에서 pilab-gpu-b 본체 전원 버튼을 직접 누른다.
켜지고 1~2분 후 SSH 접속 가능.

## 2. VSCode에서 접속 (처음 설정할 때 / 새 컴퓨터에서만)

1. VSCode 실행 → 확장에서 "Remote - SSH" 설치
2. `F1` → "Remote-SSH: Open SSH Configuration File..." → `~/.ssh/config` 선택, 추가:

   ```
   Host pilab-gpu-b
       HostName 192.168.0.100
       Port 7100
       User meaabebe
   ```

3. `F1` → "Remote-SSH: Connect to Host..." → `pilab-gpu-b` → Linux 선택 → 비밀번호 입력
4. File → Open Folder → `/mnt/pilab_nas/projects/THGS`

연구실 밖: 노트북에 Tailscale 켜고 `HostName 100.64.0.4` 로 접속.

## 3. 접속 후 확인

```bash
ls /mnt/pilab_nas/projects/THGS          # NAS autofs 마운트 트리거 + 확인

source ~/miniforge3/etc/profile.d/conda.sh
conda activate thgs
export CUDA_VISIBLE_DEVICES=2            # GPU 2 고정 (0,1은 dorong이 사용)

nvidia-smi
```

## 4. 문제 해결

- "Could not establish connection" → 서버가 아직 안 켜짐. 전원 버튼부터.
- 비밀번호가 계속 거부됨 → 포트 7100인지 확인.
- THGS 폴더가 빈 것처럼 보임 → NAS(192.168.0.203, Synology)가 안 켜진 것. NAS 전원을 켜면 autofs가 자동으로 다시 마운트.
- 복구용 클라우드 백업:
  - 코드: https://github.com/BAEJUNHAK/THGS (`blackwell-research` 브랜치)
  - language_features (9GB): https://huggingface.co/datasets/JUNHAKBAE/THGS-lerf-ovs-language-features
  - 실험 산출물 (output/lerf, output/diagnostics, ReLaGS/output): https://huggingface.co/datasets/JUNHAKBAE/THGS-experiment-outputs
  - 복원: 레포 루트에서 `hf download <repo> --repo-type dataset --local-dir <대상>`
