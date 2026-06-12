"""
Stage 3.4-E — synthesis: complete cause taxonomy of all residual cases.

Joins: fullpool ranks, NAG kinship, loser anatomy, easy-regression anatomy,
multi-instance audit, mask IoU, 2B subtypes -> one row per case with
primary_cause + fix_signal + evidence. Rule-based (rules documented inline),
mirroring stage2a_synthesize.py's attribution style.

Cases (36): 17 persistent phantoms + 9 'other' failures + 10 easy regressions.

Verdicts:
  R6: every case labeled, unknown <= 2
  R7: multi-instance suspects count (>=3 -> eval correction goes into Stage 4)
  R8: 2B mean_dilution/encoder_hidden cases recovered by query-aware are
      reclassified -> final encoder-limit list

Output: output/diagnostics/stage3_4_taxonomy.csv
"""

import os
import numpy as np
import pandas as pd

D = 'output/diagnostics'


def main():
    fr = pd.read_csv(f'{D}/stage3_3_fullpool_ranks.csv')
    kin = pd.read_csv(f'{D}/stage3_4_kinship.csv').set_index(['scene', 'prompt'])
    losers = pd.read_csv(f'{D}/stage3_4_losers.csv')
    ereg = pd.read_csv(f'{D}/stage3_4_easy_regression.csv')
    multi = pd.read_csv(f'{D}/stage3_4_multi_instance.csv')
    mask = pd.read_csv(f'{D}/stage3_3_mask_iou.csv')
    sub = pd.read_csv(f'{D}/stage2b_rofa_subtypes.csv').set_index(['scene', 'prompt'])

    mIoU = mask.groupby(['scene', 'prompt'])[['iou_baseline', 'iou_p']].mean()
    suspects = multi[multi.grade == 'suspect'].groupby(['scene', 'prompt']).size()

    def winner_stats(df, sc, p, regime=None):
        d = df[(df.scene == sc) & (df.prompt == p)]
        if regime is not None and 'regime' in d.columns:
            d = d[d.regime == regime]
        if len(d) == 0:
            return None
        return {'med_nv': float(d['n_valid_views'].median()),
                'frac_external': float((d['class'] == 'external').mean()),
                'frac_sibling': float((d['kinship'] == 'sibling').mean())}

    rows = []

    # ---------------- 17 phantoms
    for _, r in fr[fr.category == 'phantom17'].iterrows():
        sc, p = r['scene'], r['prompt']
        k = kin.loc[(sc, p)]
        rel = k['relation_wrong_vs_oracle']
        rp, rb = int(r['rank_p']), int(r['baseline_rank'])
        m = mIoU.loc[(sc, p)] if (sc, p) in mIoU.index else None
        ls = winner_stats(losers, sc, p, 'mean')
        n_sus = int(suspects.get((sc, p), 0))
        ev = (f"base={rb}->p={rp}; kin={rel}; "
              f"IoU {m['iou_baseline']:.2f}->{m['iou_p']:.2f}" if m is not None else "")

        # attribution rules (priority order, evidence-based)
        if rel in ('child_of_b', 'parent_of_b', 'same'):
            cause, fix = 'granularity_' + ('child' if 'child' in rel else 'parent'), 'parent_union'
        elif ls is not None and ls['med_nv'] <= 15 and ls['frac_external'] >= 0.8:
            cause, fix = 'few_view_opportunist', 'min_evidence_prior'
            if n_sus >= 3:
                cause, fix = 'multi_instance_suspect', 'eval_correction+min_evidence_prior'
        elif rel == 'sibling' and rp > rb:
            cause, fix = 'sibling_confusion', 'negative_contrast/spatial'
        elif rp <= 3:
            cause, fix = 'minority_dilution', 'hybrid'
        elif rp < rb * 0.5:
            cause, fix = 'minority_dilution_partial', 'hybrid'
        elif ls is not None and ls['frac_external'] >= 0.6:
            cause, fix = 'coherent_confuser', 'hybrid+negative_contrast'
        else:
            cause, fix = 'unknown', 'tbd'
        rows.append({'group': 'phantom17', 'scene': sc, 'prompt': p,
                     'primary_cause': cause, 'fix_signal': fix, 'evidence': ev,
                     'multi_instance_suspects': n_sus})

    # ---------------- 9 'other' failures
    for _, r in fr[fr.category == 'other'].iterrows():
        sc, p = r['scene'], r['prompt']
        rp, rb = int(r['rank_p']), int(r['baseline_rank'])
        if rp <= 3:
            cause, fix = 'minority_dilution', 'hybrid'
        elif rp < rb * 0.5:
            cause, fix = 'minority_dilution_partial', 'hybrid'
        else:
            cause, fix = 'encoder_limit_candidate', 'prompt_side(C2)/none'
        rows.append({'group': 'other_fail', 'scene': sc, 'prompt': p,
                     'primary_cause': cause, 'fix_signal': fix,
                     'evidence': f"base={rb}->p={rp}", 'multi_instance_suspects': 0})

    # ---------------- 10 easy regressions
    for (sc, p), g in ereg.groupby(['scene', 'prompt']):
        top1 = g[g.winner_rank == 1].iloc[0]
        same_in_top = (g['kinship'].isin(['same']) & (g['best_gt_iou'] >= 0.5)).any()
        if same_in_top:
            cause, fix = 'fake_regression_same_object', 'none(rank_artifact)'
        else:
            cause, fix = 'lucky_view_jump_victim', 'hybrid_guard(g1_gap,g2_mean_confidence)'
        rows.append({'group': 'easy_regression', 'scene': sc, 'prompt': p,
                     'primary_cause': cause, 'fix_signal': fix,
                     'evidence': (f"p_rank={int(top1['p_rank'])}; top1_gap="
                                  f"{top1['g1_winner_gap_top5_minus_mean']:+.2f}; "
                                  f"oracle_mean_rank={int(top1['oracle_mean_rank'])}"),
                     'multi_instance_suspects': 0})

    df = pd.DataFrame(rows)
    df.to_csv(f'{D}/stage3_4_taxonomy.csv', index=False)

    print("=" * 86)
    print("STAGE 3.4 TAXONOMY")
    print("=" * 86)
    for _, r in df.iterrows():
        print(f"  {r['group']:15s} {r['scene'][:6]:6s} {r['prompt']:26s} "
              f"{r['primary_cause']:28s} fix={r['fix_signal']}")
    print("\n=== primary_cause distribution ===")
    print(df.groupby(['group', 'primary_cause']).size().to_string())
    print("\n=== fix_signal demand ===")
    print(df['fix_signal'].value_counts().to_string())

    # R6
    n_unknown = int((df['primary_cause'] == 'unknown').sum())
    print(f"\n[R6] unknown labels: {n_unknown} (허용 <= 2) -> "
          f"{'PASS' if n_unknown <= 2 else 'FAIL'}")
    # R7
    n_sus_total = int(df['multi_instance_suspects'].sum())
    n_sus_prompts = int((df['multi_instance_suspects'] > 0).sum())
    print(f"[R7] multi-instance suspects: {n_sus_total} across {n_sus_prompts} prompts "
          f"-> {'eval correction REQUIRED in Stage 4' if n_sus_prompts >= 3 else 'optional'}")
    # R8
    print("\n[R8] 2B mean_dilution/encoder_hidden 재분류:")
    for (sc, p), s in sub.iterrows():
        if s['subtype'] != 'mean_dilution':
            continue
        row = fr[(fr.scene == sc) & (fr.prompt == p)]
        if len(row) == 0:
            continue
        rp, rb = int(row.iloc[0]['rank_p']), int(row.iloc[0]['baseline_rank'])
        verdict = ('NOT encoder limit — 극단적 dilution (query-aware 회복)'
                   if rp <= 5 or rp < rb * 0.3 else 'encoder limit 유지')
        print(f"  {sc}/{p}: base={rb}->p={rp} -> {verdict}")
    print(f"\nWrote {D}/stage3_4_taxonomy.csv")


if __name__ == '__main__':
    main()
