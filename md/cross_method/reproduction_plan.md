# Reproduction Plan — THGS / ReLaGS LERF-OVS 출처-깨끗한 재현

> 생성 2026-06-17. 상위: [paper_deep_analysis.md](paper_deep_analysis.md) §9 (재현성 정밀 분석). 동기: 표 B(성능)를 신뢰 가능하게 만들려면 **출처가 확정된 체크포인트 위에서 공식 eval(test_lerf.py + eval_seg.py)** 로 잰 절대 수치가 필요하다.
>
> 핵심 한 줄: **ReLaGS 는 저자 HF 체크포인트라 이미 충실(anchor). THGS 는 출처 불명 2025-07 빌드라 절대 mIoU 인용 불가 → 공식 Google Drive "ready-to-use scenes" 로 재현해 출처를 확정한다.**

---

## 0. 확정 사실 (Phase 0 완료, 2026-06-17)

- 공식 `eval_seg.py` 를 기존 렌더(`output/render/lerf`, test_lerf.py 산출)에 실행 (GPU-free):
  **THGS overall mIoU 0.5887 / mAcc 0.9780** (fig 0.5493 / ramen 0.4214 / teatime **0.8187** / waldo 0.5654).
- teatime 0.8187 = 공식 eval 의 진짜 출력 → **paper(0.6833) 대비 +13.5 갭은 eval 방법이 아니라 체크포인트 차이** (확정).
- mAcc 는 0.978 vs paper 0.980 으로 잘 재현 → 차이는 mIoU(특히 teatime/waldo)에 국한.

→ 그러므로 재현 작업의 본질 = **THGS 체크포인트 출처 확정**. ReLaGS 는 fig/waldo 소수점 일치라 그대로 anchor.

---

## 1. Phase 1 — THGS 출처 확정 ✅ **완료 (2026-06-17): byte-identical**

THGS 공식 repo(github.com/Atrovast/THGS): pretrained 는 "TODO" 이나 **ready-to-use scenes (2DGS + semantic field) 를 Google Drive 제공** (folder `1b3bXy8XENhvpWh4nLzu06UPBZEZNX6fG`, scene당 ~270M zip × 4).

**실행 결과**: 4 scene zip 다운로드 → unzip → `sai_nag.pt` md5 비교 vs 우리 `output/lerf/*`:

| scene | md5 | 판정 |
|---|---|---|
| figurines | `86f678e0da5102c74af4a1cd6ce419e2` | **IDENTICAL** |
| ramen | `c87fd2c851405b38b638287a282718ad` | **IDENTICAL** |
| teatime | `03db4c9273af683cfcdf10ceaa2f2d44` | **IDENTICAL** |
| waldo_kitchen | `a03956400806fad0dca95ad9397961ea` | **IDENTICAL** |

cfg_args 도 완전 동일(`/data/dsh/...` = 저자 경로). → **우리 THGS 체크포인트 = THGS 공식 Drive 릴리즈 그 자체 (우리가 빌드한 게 아님).**

**R-repro-1 판정**: 첫 분기 적중 — **우리 현 체크포인트 = 공식 릴리즈**. 단 그 릴리즈가 paper Table(54.94)을 재현하지 못하고 official eval 로 **58.87** 을 냄. → 처방 확정: paper 에 "publicly released THGS checkpoint, official protocol = 58.87 (paper's 54.94 is an unreleased build)" 로 정직 명시. THGS base = 58.87 (출처 byte-clean). **Phase 2 불필요.**

(staging: `repro_staging/` — sai_nag 가 output/lerf 와 byte-동일이라 redundant. 정리 가능.)

---

## 2. Phase 2 — 전체 파이프라인 재빌드 (~~Phase 1 불충분 시에만~~ **불필요: Phase 1 이 byte-identical 로 출처 확정**, GPU-heavy)

> Phase 1 에서 우리 체크포인트 = 공식 릴리즈 byte-identical 확정 → 재빌드 불필요. 아래는 만약 "paper Table 수치 자체"를 재현하려 할 때(저자 미공개 빌드 추적)만 의미 있으나, 공개된 적 없으므로 비권장. 참고용으로만 보존.

공식 2DGS scene 위에서 THGS 파이프라인 직접 실행:
```bash
bash scripts/run.sh configs/lerf.yml figurines ramen teatime waldo_kitchen
# = sp_partition → graph_weight → sp_partition -k → merge_proj → sai_nag.pt
for sc in figurines ramen teatime waldo_kitchen; do
  python test_lerf.py -s data/lerf_ovs/$sc -m <model>/$sc --path_pred output/render/lerf_repro
done
python scripts/eval_seg.py -d lerf --scene_list figurines ramen teatime waldo_kitchen \
  --path_pred output/render/lerf_repro --path_gt data/lerf_ovs/label
```
- **language_features 캐시 재사용** (재생성 금지 — GPU 4h). merge_proj 가 `_f.npy` 를 읽음.
- SLURM 예약 필요 (FRNN + cut-pursuit + multi-view trace). scene당 수십 분~시간.

---

## 3. Phase 3 — ReLaGS 공식-경로 parity

ReLaGS 는 공식 eval 미공개("published soon"). 현재 native diagnostic = 61.65 (Algorithm-1 hierarchical).
- THGS 의 test_lerf.py 골격에 ReLaGS Algorithm-1 inference 를 얹어 **PNG mask 렌더** → eval_seg.py 로 동일 하네스 절대 수치.
- 목표: ReLaGS 도 eval_seg.py 기준 canonical 숫자 확보 (현 native 진단과 ±? 확인). fig/waldo 가 paper 와 일치하는지 재확인.
- 비용: 중간 (렌더 GPU). ReLaGS render 경로는 P2-b 에서도 필요하므로 공유.

---

## 4. Phase 4 — 표 B base 확정

- THGS·ReLaGS 각각 **출처-확정 + 공식 eval** 절대 수치를 표 B base 로 고정.
- P2 drop-in 이득은 전부 **"이 base 위 Δ"** 로 보고 (절대값 핸디캡 무관화).
- 외부 비교(leaderboard): ReLaGS Table 3 의 published 숫자(VALA 61.7, LAGA 64.0, ReLaGS 64.4)와 나란히.

---

## 5. 실행 순서 + 의사결정 트리

```
[Phase 0 ✅] 공식 eval = 0.5887 (체크포인트 차이 확정, eval 버그 아님)
[Phase 1 ✅] 공식 Drive 다운로드 + md5 = 4/4 byte-identical
   → 우리 체크포인트 = 공식 릴리즈. THGS base=58.87 출처-clean 확정. Phase 2 불필요.
[Phase 3 ⏳] ReLaGS 공식-경로 parity (optional — fig/waldo 이미 paper 일치)
[Phase 4 ⏳] 표 B base 고정: THGS 58.87 / ReLaGS 61.65 (둘 다 출처-clean), Δ 프레이밍
```

## 6. 영향 범위 (안심 포인트)

- **메커니즘/진단 결론(Stage 1~5, P1)은 전부 rank 기반 → 체크포인트 교체에 robust** (paper_deep_analysis §5.3). 재현 작업은 **표 B(절대 성능)** 의 신뢰도만 건드림.
- ReLaGS 는 이미 anchor 라 paradigm 주장은 흔들리지 않음.

## 7. 업데이트 로그

| 날짜 | 변경 | 한 줄 |
|---|---|---|
| 2026-06-17 | 초기 작성 | Phase 0(공식 eval=0.5887, 체크포인트 차이 확정) 완료. Phase 1(Drive 출처 확정)~4(표 B base) 계획 + R-repro-1 사전등록 |
| 2026-06-17 | **Phase 1 완료 (byte-identical)** | 공식 Drive 4 scene 다운로드 → md5 4/4 IDENTICAL. **우리 THGS = 공식 릴리즈 그 자체**, R-repro-1 첫 분기 적중. released≠paper-table(58.87 vs 54.94) = THGS reproducibility gap. THGS base=58.87 출처-clean 확정, Phase 2 불필요. 남은 건 Phase 3(ReLaGS parity, optional) + Phase 4(표 B) |
