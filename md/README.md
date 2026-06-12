# `md/` — Analysis Documents Index

분석 문서들을 method/주제별로 정리.

```
md/
├── THGS/             — THGS 단독 분석 (failure analysis, pipeline 등)
├── RELAGS/           — ReLaGS 단독 분석 (CVPR 2026, native inference)
├── cross_method/     — THGS vs ReLaGS 비교 + paper 검증
├── hypotheses/       — 확장 가설 + experiments/ (Stage 검증) + strategy/ (paper 전략, Phase 5)
├── ramen/            — ramen scene 특화 분석
├── _archive/         — 폐기된 잘못된 분석 (보존용)
└── misc/             — 기타
```

## 📌 핵심 문서 (현재 분석 흐름)

### 1. THGS Distractor Analysis (Phase 1)
- **[THGS/lerf_ovs_failure_analysis_results.md](THGS/lerf_ovs_failure_analysis_results.md)** ← THGS의 4-class distractor taxonomy 메인 결과
- [THGS/lerf_ovs_failure_analysis_plan.md](THGS/lerf_ovs_failure_analysis_plan.md) — 실험 계획
- [THGS/lerf_ovs_deep_analysis.md](THGS/lerf_ovs_deep_analysis.md) — Q1-Q5 deep analysis

### 2. ReLaGS Analysis (Phase 2 = Same framework applied)
- **[RELAGS/lerf_ovs_failure_analysis_relags_native_h.md](RELAGS/lerf_ovs_failure_analysis_relags_native_h.md)** ← ReLaGS hierarchical mode 결과
- [RELAGS/lerf_ovs_deep_analysis_relags.md](RELAGS/lerf_ovs_deep_analysis_relags.md) — ReLaGS Q1-Q5
- [RELAGS/relags_setup_verification.md](RELAGS/relags_setup_verification.md) — paper Algorithm 1과 호환성 검증
- [RELAGS/lerf_ovs_failure_analysis_relags_native.md](RELAGS/lerf_ovs_failure_analysis_relags_native.md) — root-only 모드 (참고용)

### 3. Cross-method Comparison ⭐ (가장 중요)
- **[cross_method/lerf_ovs_thgs_vs_relags.md](cross_method/lerf_ovs_thgs_vs_relags.md)** ← THGS vs ReLaGS distractor 분포 비교 (메인)
- **[cross_method/paper_deep_analysis.md](cross_method/paper_deep_analysis.md)** ← 두 논문 직접 인용 + 우리 측정과 cross-check

### 4. Extended Hypotheses (Phase 3) 🔬 — Mechanism-based Taxonomy (6.2 patch)
- **[hypotheses/extended_failure_hypotheses.md](hypotheses/extended_failure_hypotheses.md)** ← **6.2 patch (코드 재검토 + measurement 정밀화)**.
  - **Framing 진화**: ① D1–D4 너머 (H1–H7) → ② CLIP-injection 중심 → ③ Visibility ⊥ mechanism, D2/D3 merge → ④ 6-category (A/B/C/D/E/F) 재정렬 → ⑤ A4/B7/C3/D4 추가, E4 (counterfactual) 신설 → ⑥ B8 (within-view mixing) 신설, F2 GMM 공간 명시 → **⑦ 6.1 patch (코드 grounded)** + **⑧ 6.2 patch (measurement 정밀화)**.
  - **새 taxonomy**: D1 (degenerate) / **D2 (semantic distractor: D2.real vs D2.phantom)** / D3 (over-union).
  - **18 가설** (Tier 0/1/2/3) — B7+A4 prerequisite, A3 dual-side classifier, E1 phantom direction main contribution candidate.

### 5. Experiments (Phase 4) 🧪 — Stage-by-stage 검증 + 결과 기록
- **[hypotheses/experiments/README.md](hypotheses/experiments/README.md)** ← Stage 진행 index
- **[hypotheses/experiments/stage1_b7_a4_a2.md](hypotheses/experiments/stage1_b7_a4_a2.md)** ← ✅ **Stage 1 완료** — B7+A4+A2+Joint+Cross-method
  - **핵심 결과**: D2.phantom 31% vs D2.real 7.5% (4.2:1 비율), **method-agnostic**
  - **17 persistent phantoms** = 두 method 모두 실패하는 prompt set → 새 method 의 target
  - Paper Section 1 draft v2: [THGS/paper_section1_draft.md](THGS/paper_section1_draft.md)

### 6. Strategy (Phase 5) 🎯 — paper 격차 전략 + 경쟁 부검
- **[hypotheses/strategy/roadmap.md](hypotheses/strategy/roadmap.md)** ← phase 계획 (P1 경쟁 부검 → **P2 method 본선 (가드 + R10)** → P3 게이트 → P4 optional) + 목적 위계 (성능 우위 method 가 최종 목적, P1 = 문제점 구체화 → method 사양서) + 두 킬러 테이블 (표 A 문제점 / 표 B 성능)
- **[hypotheses/strategy/competitor_autopsy.md](hypotheses/strategy/competitor_autopsy.md)** ← 경쟁 처방 (VALA·Beyond Averages·ROFA 등) 의 사전등록 부검 (R11)

## 📊 데이터 파일

- [../output/diagnostics/lerf_ovs_per_prompt.csv](../output/diagnostics/lerf_ovs_per_prompt.csv) — THGS 진단 (208 rows)
- [../output/diagnostics/lerf_ovs_relags_native_h.csv](../output/diagnostics/lerf_ovs_relags_native_h.csv) — **ReLaGS 진단 (hierarchical, MAIN)**
- [../output/diagnostics/lerf_ovs_relags_native.csv](../output/diagnostics/lerf_ovs_relags_native.csv) — ReLaGS 진단 (root-only, 참고용)
- [../output/diagnostics/_archive/](../output/diagnostics/_archive/) — 폐기된 백업 CSV

## 🔧 핵심 스크립트

- [../scripts/lerf_ovs_diagnostic.py](../scripts/lerf_ovs_diagnostic.py) — THGS native diagnostic
- [../ReLaGS/scripts/lerf_ovs_diagnostic_native.py](../ReLaGS/scripts/lerf_ovs_diagnostic_native.py) — ReLaGS native diagnostic
- [../scripts/lerf_ovs_diag_analyze.py](../scripts/lerf_ovs_diag_analyze.py) — Distractor analysis (D1-D4 taxonomy)
- [../scripts/lerf_ovs_deep_analyze.py](../scripts/lerf_ovs_deep_analyze.py) — Q1-Q5 deep analysis

## 📁 폴더별 README 권장 읽기 순서

### 처음 보는 사람을 위해
1. **이 README** (개요)
2. [cross_method/lerf_ovs_thgs_vs_relags.md](cross_method/lerf_ovs_thgs_vs_relags.md) (핵심 비교 결과)
3. [cross_method/paper_deep_analysis.md](cross_method/paper_deep_analysis.md) (paper 검증)
4. THGS/RELAGS 각자 단독 분석 (세부 확인)

### Distractor framework 이해
1. [THGS/lerf_ovs_failure_analysis_results.md](THGS/lerf_ovs_failure_analysis_results.md) Section 11 (Distractor taxonomy 정식 정의)
2. [cross_method/lerf_ovs_thgs_vs_relags.md](cross_method/lerf_ovs_thgs_vs_relags.md) (적용 예시)
3. [hypotheses/extended_failure_hypotheses.md](hypotheses/extended_failure_hypotheses.md) (framework 의 blind spot + 확장 가설 H1–H7)

## ⚠️ Archive 알림

[_archive/](_archive/) 폴더의 *_WRONG.md.bak 파일은 **잘못된 분석 (THGS inference로 ReLaGS 측정함)** 의 백업. 참고하지 말 것.
