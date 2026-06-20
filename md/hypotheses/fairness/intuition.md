# 직관 — 공정성의 "두 법정" (1 페이지)

> [protocol.md](protocol.md)의 직관적 거울. 엄밀한 버전은 protocol을 봐.

---

## ⚖️ 두 개의 법정

세 method(THGS·ReLaGS·VALA)를 비교하는 건 *어떤 법정*이냐에 따라 규칙이 달라:

### 🏆 법정 A — 리더보드 ("누가 이겼나")
- **판결 = 최종 점수판.** 각 선수(method)가 *자기 방식*으로 경기하되(2DGS든 3DGS든, top-3든 per-pixel이든 — 그게 실력), **심판(평가)은 한 명**이어야 함.
- 맞춰야 할 것: **평가뿐** — 같은 GT, 같은 metric 정의, 같은 집계. (그리고 "3D 방법이 이겼다"고 말하려면 CLIP/SAM도 같아야 — backbone빨 아니란 걸 보이려고.)
- 임계는? 각 선수의 *최적 컨디션*(자기 best threshold)에서 — 모두에게 같은 임계를 강요하면 스케일 다른 선수가 억울.

### 🔬 법정 B — 메커니즘 ("무엇이 원인인가")
- **통제 실험실.** "평균이 phantom을 죽인다"를 증명하려면 — **딱 한 개(aggregation)만 바꾸고 나머지 전부 동결.**
- 같은 per-view feature dump 위에서 aggregation rule만 mean→median→top-k로 갈아끼움. 다른 3DGS·다른 selection이 섞이면 = 오염.

---

## 🚫 섞으면 = 오심(mistrial)

- 리더보드 점수표 보고 "평균이 원인"이라 결론 → ❌ (선수마다 파이프라인이 달라서 원인 못 가림)
- native cross-method dump로 "우리가 이긴다" 주장 → ❌ (평가/기반이 안 맞음)

→ 그래서 폴더가 `leaderboard/`와 `mechanism/`로 갈리고, **서로의 데이터를 절대 안 읽음.** 공유하는 건 *심판 코드(common/)* 뿐.

---

## 🍜 우리 사례 한 줄

teatime은 어느 법정에서도 깨끗(0.69≈논문 0.71)인데, **지금 표 B는 THGS=eval_seg, VALA=compute_lerf_iou로 *심판이 둘*이라 무효.** fairness 트랙 = 심판을 한 명으로 통일하는 것 (법정 A) + 원인 규명용 통제 실험실 따로 두는 것 (법정 B).

→ 엄밀히: [protocol.md](protocol.md)
