"""
P1-E.0 — dump-free faithful regime split (GPU 0, no SLURM).

Builds the regime-confusion evidence from existing diagnostic CSVs only:
  1) p1e0_regime_split.csv  — rule x regime mean mask IoU + delta vs baseline
  2) p1e0_regime_map.png    — per-prompt scatter (consensus vs minority IoU)
  3) p1e0_guard_auroc.csv   — easy-vs-phantom AUROC of guard signals

Pre-registration (md/hypotheses/strategy/p1e_external_code_study_plan.md §6.0):
  G-A regime confusion (mirror image at mask level)
  G-B regime separability (AUROC of selection-benefit, phantom vs easy)
  G-C guard signal (top1-conf easy-vs-phantom AUROC; predicted < 0.75 and < its
       present-vs-absent AUROC 0.84)

Read-only on inputs. Writes only the 3 outputs above.
"""

import os
import numpy as np
import pandas as pd

D = 'output/diagnostics'
RULES = {  # display name -> p1a_mask_iou column
    'baseline': 'iou_baseline',   # consensus / mean
    'gm_u':     'iou_gm_u',       # VALA-faithful: unweighted geometric median
    'gm_w':     'iou_gm_w',       # VALA-faithful: visibility-weighted geom median
    'gm_g':     'iou_gm_g',       # VALA-faithful: weighted + gating
    'top5':     'iou_top5',       # selection: bag query-top-5
    'qmax1':    'iou_qmax1',      # selection: bag k=1
}
CONSENSUS = ['baseline', 'gm_u', 'gm_w', 'gm_g']
SELECTION = ['top5', 'qmax1']


def auroc(pos, neg):
    """P(score(pos) > score(neg)) + 0.5 * P(tie). NaNs dropped."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def regime_of(cls):
    return 'easy' if cls == 'easy' else 'phantom' if cls == 'phantom' else 'other'


def main():
    # ---------- load ----------
    mask = pd.read_csv(f'{D}/p1a_mask_iou.csv')
    cross = pd.read_csv(f'{D}/cross_method_d2_decomposition.csv')
    strict17 = pd.read_csv(f'{D}/persistent_phantoms_17.csv')
    strict_set = set(map(tuple, strict17[['scene', 'prompt']].values))

    # per-prompt IoU = mean over eval_frame
    pp = mask.groupby(['scene', 'prompt'], as_index=False).mean(numeric_only=True)
    pp = pp.merge(cross[['scene', 'prompt', 'thgs_class']], on=['scene', 'prompt'], how='left')
    missing = pp['thgs_class'].isna().sum()
    if missing:
        print(f"[warn] {missing} prompts had no thgs_class (dropped from regime stats)")
    pp = pp.dropna(subset=['thgs_class']).copy()
    pp['regime'] = pp['thgs_class'].map(regime_of)
    pp['is_strict17'] = [(s, p) in strict_set for s, p in zip(pp.scene, pp.prompt)]

    n_by_reg = pp['regime'].value_counts().to_dict()
    print("=" * 78)
    print(f"per-prompt rows: {len(pp)}   regimes: {n_by_reg}   "
          f"strict17: {int(pp.is_strict17.sum())}")
    print("=" * 78)

    # ---------- 1) regime split ----------
    rows = []
    for rule, col in RULES.items():
        for reg in ['easy', 'phantom', 'other']:
            g = pp[pp.regime == reg]
            mean_iou = g[col].mean()
            base = g['iou_baseline'].mean()
            row = {'rule': rule, 'regime': reg, 'n': len(g),
                   'mean_iou': round(mean_iou, 4),
                   'delta_vs_baseline': round(mean_iou - base, 4),
                   'mean_iou_strict17': '', 'delta_strict17': ''}
            if reg == 'phantom':
                gs = pp[pp.is_strict17]
                ms = gs[col].mean(); bs = gs['iou_baseline'].mean()
                row['mean_iou_strict17'] = round(ms, 4)
                row['delta_strict17'] = round(ms - bs, 4)
            rows.append(row)
    split = pd.DataFrame(rows)
    split.to_csv(f'{D}/p1e0_regime_split.csv', index=False)

    # console pivot: delta vs baseline
    print("\nREGIME SPLIT — Δ mask IoU vs baseline (pt = x100), per-prompt mean")
    piv = split.pivot(index='rule', columns='regime', values='delta_vs_baseline')
    piv = piv.reindex(CONSENSUS + SELECTION)[['easy', 'phantom', 'other']]
    abs_iou = split.pivot(index='rule', columns='regime', values='mean_iou')
    abs_iou = abs_iou.reindex(CONSENSUS + SELECTION)[['easy', 'phantom', 'other']]
    hdr = f"  {'rule':10s} | {'easy Δ':>8s} {'phan Δ':>8s} {'othr Δ':>8s} |  (abs: easy/phan)"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for rule in CONSENSUS + SELECTION:
        d = piv.loc[rule]; a = abs_iou.loc[rule]
        fam = 'C' if rule in CONSENSUS else 'S'
        print(f"  {rule:10s} | {d.easy*100:+8.1f} {d.phantom*100:+8.1f} "
              f"{d.other*100:+8.1f} |  {a.easy:.3f}/{a.phantom:.3f}  [{fam}]")
    strict_phn = split[(split.regime == 'phantom')][['rule', 'delta_strict17']]
    print("  strict-17 phantom Δ: " + "  ".join(
        f"{r.rule}={float(r.delta_strict17)*100:+.1f}" for _, r in strict_phn.iterrows()))

    # ---------- 2) regime map ----------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'easy': '#2a9d8f', 'phantom': '#e63946', 'real': '#6a4c93', 'rare': '#f4a261'}
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for cls, c in colors.items():
        g = pp[pp.thgs_class == cls]
        if len(g):
            ax.scatter(g['iou_baseline'], g['iou_top5'], s=42, c=c, alpha=0.75,
                       edgecolors='k', linewidths=0.4, label=f"{cls} (n={len(g)})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
    ax.set_xlabel('consensus regime IoU  (baseline / mean)')
    ax.set_ylabel('minority regime IoU  (query-top-5)')
    ax.set_title('P1-E.0 regime map — per-prompt (THGS base)\n'
                 'below diag = selection hurts (easy) · above = selection helps (phantom)')
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.02, 1.0)
    ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f'{D}/p1e0_regime_map.png', dpi=150)
    plt.close(fig)

    # G-B: selection-benefit separability (phantom pos vs easy neg)
    pp['sel_benefit'] = pp['iou_top5'] - pp['iou_baseline']
    gb = auroc(pp[pp.regime == 'phantom']['sel_benefit'],
               pp[pp.regime == 'easy']['sel_benefit'])

    # ---------- 3) guard AUROC ----------
    p1b = pd.read_csv(f'{D}/p1b_absence_scores.csv')
    cls_map_thgs = dict(((s, p), c) for s, p, c in
                        cross[['scene', 'prompt', 'thgs_class']].values)
    cls_map_rel = dict(((s, p), c) for s, p, c in
                       cross[['scene', 'prompt', 'relags_class']].values)
    SIGNALS = ['top1_mean', 'mean_margin', 'agree_mean_top5']
    grows = []
    for method in ['thgs', 'relags']:
        m = p1b[p1b.method == method].copy()
        cmap = cls_map_thgs if method == 'thgs' else cls_map_rel
        present = m[m.is_absent == 0].copy()
        present['cls'] = [cmap.get((s, q)) for s, q in zip(present['scene'], present['query'])]
        easy = present[present.cls == 'easy']
        phan = present[present.cls == 'phantom']
        absent = m[m.is_absent == 1]
        for sig in SIGNALS:
            grows.append({
                'method': method, 'signal': sig,
                'n_easy': len(easy), 'n_phantom': len(phan),
                'auroc_easy_vs_phantom': round(auroc(easy[sig], phan[sig]), 3),
                'auroc_present_vs_absent': round(auroc(present[sig], absent[sig]), 3),
            })
    guard = pd.DataFrame(grows)
    guard.to_csv(f'{D}/p1e0_guard_auroc.csv', index=False)
    print("\nGUARD SIGNAL AUROC (present 쿼리, easy=pos vs phantom=neg)")
    print(guard.to_string(index=False))

    # ---------- pre-registered verdicts ----------
    print("\n" + "=" * 78)
    print("PRE-REGISTERED VERDICTS (G-A / G-B / G-C)")
    print("=" * 78)

    gm_g = piv.loc['gm_g']; top5 = piv.loc['top5']
    ga_robust = (gm_g.easy >= 0) and (gm_g.phantom <= 0)
    ga_select = (top5.phantom > 0) and (top5.easy < 0)
    print(f"[G-A] mirror image (regime confusion):")
    print(f"      robust gm_g : easy {gm_g.easy*100:+.1f}pt (≥0?), "
          f"phantom {gm_g.phantom*100:+.1f}pt (≤0?)  -> {'PASS' if ga_robust else 'FAIL'}")
    print(f"      select top5 : phantom {top5.phantom*100:+.1f}pt (≫0?), "
          f"easy {top5.easy*100:+.1f}pt (≪0?)  -> {'PASS' if ga_select else 'FAIL'}")
    print(f"      => {'PASS — 거울상 재확인 (각 family 한 regime 전용)' if ga_robust and ga_select else 'PARTIAL/FAIL'}")

    print(f"\n[G-B] regime separability (selection-benefit, phantom vs easy):")
    print(f"      AUROC = {gb:.3f}  (기준 ≥0.80)  -> {'PASS' if gb >= 0.8 else 'FAIL'}")

    t1 = guard[(guard.method == 'thgs') & (guard.signal == 'top1_mean')].iloc[0]
    ep = t1['auroc_easy_vs_phantom']; pa = t1['auroc_present_vs_absent']
    pred_ok = (ep < 0.75) and (ep < pa)
    if ep >= 0.8:
        branch = "≥0.8 → prompt-level top1-conf 가드 채택 신호 (예측 빗나감 = 이득)"
    elif ep < 0.65:
        branch = "<0.65 → structural 신호(§6.0-b)로 전환 필요"
    else:
        branch = "0.65–0.8 → 경계, structural 신호와 비교 필요"
    print(f"\n[G-C] top1-conf 의 진짜 과제 (THGS, easy-vs-phantom):")
    print(f"      easy/phantom AUROC = {ep:.3f}  vs  present/absent AUROC = {pa:.3f}")
    print(f"      예측 (confident-impostor): <0.75 AND <absent  -> "
          f"{'PASS (예측 적중)' if pred_ok else 'MISS'}")
    print(f"      분기: {branch}")

    print("\n" + "=" * 78)
    print("OUTPUTS:")
    for f in ['p1e0_regime_split.csv', 'p1e0_regime_map.png', 'p1e0_guard_auroc.csv']:
        print(f"  {D}/{f}")
    print("=" * 78)


if __name__ == '__main__':
    main()
