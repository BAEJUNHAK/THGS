"""
Stage 3.1 — B8 causal replay (Phase C analysis).

Joins:
  - replay pkl  (stage3_b8_replay_perview.pkl)   : mixed/hard per-view feats + mix stats
  - clean pkls  (_h2_lite_perview.pkl phantoms, _h2_lite_easy.pkl controls)
  - subtype CSV (stage2b_rofa_subtypes.csv)      : strong_signal membership

Per (scene, prompt) computes paired per-view (clean_cos, mixed_cos, hard_cos)
on the intersection of views (join by image_name), plus B8 mix statistics,
then applies the PRE-REGISTERED decision rules from
md/hypotheses/extended_failure_hypotheses.md B8 §6.3:

  H-B8a: strong_signal 11 — mean(clean - mixed) >= 0.05 AND significantly
         larger than easy controls (Mann-Whitney U, one-sided).
  H-B8b: corr(mix_rate, gap) r > 0.4 (and phantom-vs-easy mix_rate contrast).
  H-B8c: recovery = (hard - mixed) / (clean - mixed):
         >= 70% -> Stage 3.2 GO | < 30% -> B2 pivot | small gap -> B1 re-aim.

Outputs:
  output/diagnostics/stage3_b8_mix_stats.csv      (per SP per-view aggregate)
  output/diagnostics/stage3_b8_gap_summary.csv    (per SP gaps + group verdicts)
  output/diagnostics/plots/stage3_clean_vs_mixed_vs_hard.png
  output/diagnostics/plots/stage3_mixrate_vs_gap.png
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FID_GATE = 0.95


def load_text_feats(a2_csv, prompts_needed, device):
    import torch
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k",
        precision="fp16" if device == "cuda" else "fp32")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    a2 = pd.read_csv(a2_csv)
    scene_prompts = {sc: sorted(g['prompt'].unique()) for sc, g in a2.groupby('scene')}
    all_prompts = sorted(set(p for ps in scene_prompts.values() for p in ps) | prompts_needed)
    with __import__('torch').no_grad():
        tok = tokenizer(all_prompts).to(device)
        tf = model.encode_text(tok).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
    text = {p: tf[i].cpu().numpy() for i, p in enumerate(all_prompts)}
    return text, scene_prompts


def rank_of(true_cos, all_cos):
    return int((all_cos > true_cos).sum()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay_pkl", default="output/diagnostics/stage3_b8_replay_perview.pkl")
    ap.add_argument("--clean_pkl", default="output/diagnostics/_h2_lite_perview.pkl")
    ap.add_argument("--clean_easy_pkl", default="output/diagnostics/_h2_lite_easy.pkl")
    ap.add_argument("--a2_csv", default="output/diagnostics/a2_image_clip_ceiling.csv")
    ap.add_argument("--subtypes_csv", default="output/diagnostics/stage2b_rofa_subtypes.csv")
    ap.add_argument("--out_mix_csv", default="output/diagnostics/stage3_b8_mix_stats.csv")
    ap.add_argument("--out_gap_csv", default="output/diagnostics/stage3_b8_gap_summary.csv")
    ap.add_argument("--plot_dir", default="output/diagnostics/plots")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.replay_pkl, "rb") as f:
        replay = pickle.load(f)
    clean = {}
    for pth in (args.clean_pkl, args.clean_easy_pkl):
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                clean.update(pickle.load(f))
    sub = pd.read_csv(args.subtypes_csv)
    subtype = {(r['scene'], r['prompt']): r['subtype'] for _, r in sub.iterrows()}

    prompts_needed = set(p for (_, p) in replay.keys())
    text, scene_prompts = load_text_feats(args.a2_csv, prompts_needed, device)

    # ---------------- fidelity gate report
    fids = [(k, v['fidelity_cos']) for k, v in sorted(replay.items())]
    n_pass = sum(1 for _, f_ in fids if f_ >= FID_GATE)
    print("=" * 80)
    print(f"FIDELITY GATE: {n_pass}/{len(fids)} targets with cos(recon, sai_nag) >= {FID_GATE}")
    for (sc, p), f_ in fids:
        if f_ < FID_GATE:
            print(f"  FAIL {sc}/{p}: {f_:.4f}")
    print("=" * 80)

    mix_rows, gap_rows = [], []
    perview_store = {}  # for plots

    for (sc, prompt), rec in sorted(replay.items()):
        tfeat = text[prompt]
        sps = scene_prompts.get(sc, [])
        sp_mat = np.stack([text[p] for p in sps], axis=0) if sps else None
        true_idx = sps.index(prompt) if prompt in sps else -1

        clean_views = {v['image_name']: v for v in clean.get((sc, prompt), [])}

        rows_v = []
        for v in rec['views']:
            if v['mixed_feat'] is None:
                continue
            mixed = v['mixed_feat'].astype(np.float32)
            mixed_cos = float(mixed @ tfeat)
            hard_cos = (float(v['hard_feat'].astype(np.float32) @ tfeat)
                        if v['hard_feat'] is not None else np.nan)
            mixed_rank = hard_rank = np.nan
            if sp_mat is not None and true_idx >= 0:
                mc = mixed @ sp_mat.T
                mixed_rank = rank_of(mc[true_idx], mc)
                if v['hard_feat'] is not None:
                    hc = v['hard_feat'].astype(np.float32) @ sp_mat.T
                    hard_rank = rank_of(hc[true_idx], hc)
            cv = clean_views.get(v['image_name'])
            clean_cos = float(cv['cos_with_prompt']) if cv is not None else np.nan
            rows_v.append({
                'image_name': v['image_name'], 'visible': v['visible'],
                'portion': v['portion'], 'mix_count': v['mix_count'],
                'top1_ratio': v['top1_ratio'], 'top2_ratio': v['top2_ratio'],
                'clean_cos': clean_cos, 'mixed_cos': mixed_cos, 'hard_cos': hard_cos,
                'mixed_rank': mixed_rank, 'hard_rank': hard_rank,
            })
        dfv = pd.DataFrame(rows_v)
        perview_store[(sc, prompt)] = dfv
        if len(dfv) == 0:
            print(f"  [warn] {sc}/{prompt}: no usable views")
            continue

        vis = dfv[dfv['visible']]
        mix_rate = float((vis['mix_count'] >= 2).mean()) if len(vis) else np.nan
        paired = dfv.dropna(subset=['clean_cos'])
        paired_h = paired.dropna(subset=['hard_cos'])

        gap_mixed = float((paired['clean_cos'] - paired['mixed_cos']).mean()) if len(paired) else np.nan
        gap_hard = float((paired_h['clean_cos'] - paired_h['hard_cos']).mean()) if len(paired_h) else np.nan
        recovery = np.nan
        if len(paired_h) and not np.isnan(gap_mixed) and gap_mixed > 1e-4:
            num = float((paired_h['hard_cos'] - paired_h['mixed_cos']).mean())
            recovery = num / gap_mixed

        cat = rec['category']
        st = subtype.get((sc, prompt), 'easy' if cat == 'easy_sample' else 'unknown')

        mix_rows.append({
            'scene': sc, 'prompt': prompt, 'category': cat, 'subtype': st,
            'oracle_lvl': rec['oracle_lvl'], 'oracle_sp_id': rec['oracle_sp_id'],
            'fidelity_cos': rec['fidelity_cos'],
            'n_views_recorded': len(dfv), 'n_views_visible': int(dfv['visible'].sum()),
            'n_views_paired': len(paired),
            'mix_rate': mix_rate,
            'mean_mix_count': float(vis['mix_count'].mean()) if len(vis) else np.nan,
            'mean_top1_ratio': float(vis['top1_ratio'].mean()) if len(vis) else np.nan,
            'mean_top2_ratio': float(vis['top2_ratio'].mean()) if len(vis) else np.nan,
        })
        gap_rows.append({
            'scene': sc, 'prompt': prompt, 'category': cat, 'subtype': st,
            'fidelity_cos': rec['fidelity_cos'], 'mix_rate': mix_rate,
            'clean_cos_mean': float(paired['clean_cos'].mean()) if len(paired) else np.nan,
            'mixed_cos_mean': float(paired['mixed_cos'].mean()) if len(paired) else np.nan,
            'hard_cos_mean': float(paired_h['hard_cos'].mean()) if len(paired_h) else np.nan,
            'gap_mixed': gap_mixed, 'gap_hard': gap_hard, 'recovery_frac': recovery,
            'mixed_rank_median': float(dfv['mixed_rank'].median()),
            'hard_rank_median': float(dfv['hard_rank'].median()),
        })

    mix_df = pd.DataFrame(mix_rows)
    gap_df = pd.DataFrame(gap_rows)
    os.makedirs(os.path.dirname(args.out_mix_csv), exist_ok=True)
    mix_df.to_csv(args.out_mix_csv, index=False)
    gap_df.to_csv(args.out_gap_csv, index=False)

    # ---------------- per-target table
    print(f"\n{'cat':10s} {'subtype':22s} {'scene/prompt':38s} {'fid':>5s} {'mixrate':>7s} "
          f"{'clean':>6s} {'mixed':>6s} {'hard':>6s} {'gap':>6s} {'recov':>6s} {'medR_m':>6s} {'medR_h':>6s}")
    for _, r in gap_df.sort_values(['category', 'gap_mixed'], ascending=[True, False]).iterrows():
        print(f"{r['category'][:10]:10s} {str(r['subtype'])[:22]:22s} "
              f"{(r['scene'] + '/' + r['prompt'])[:38]:38s} {r['fidelity_cos']:5.2f} "
              f"{r['mix_rate']:7.2f} {r['clean_cos_mean']:6.3f} {r['mixed_cos_mean']:6.3f} "
              f"{r['hard_cos_mean']:6.3f} {r['gap_mixed']:6.3f} "
              f"{r['recovery_frac'] if not np.isnan(r['recovery_frac']) else float('nan'):6.2f} "
              f"{r['mixed_rank_median']:6.1f} {r['hard_rank_median']:6.1f}")

    # ---------------- pre-registered verdicts
    strong = gap_df[gap_df['subtype'] == 'strong_signal_phantom'].dropna(subset=['gap_mixed'])
    easy = gap_df[gap_df['category'] == 'easy_sample'].dropna(subset=['gap_mixed'])
    phant = gap_df[gap_df['category'] == 'phantom17'].dropna(subset=['gap_mixed'])

    print("\n" + "=" * 80)
    print("PRE-REGISTERED VERDICTS (extended_failure_hypotheses.md B8 §6.3)")
    print("=" * 80)

    # H-B8a
    g_s = strong['gap_mixed'].values
    g_e = easy['gap_mixed'].values
    mw = stats.mannwhitneyu(g_s, g_e, alternative='greater') if len(g_s) and len(g_e) else None
    print(f"\n[H-B8a] strong_signal (n={len(g_s)}) mean gap_mixed = {np.mean(g_s):.4f} "
          f"(threshold 0.05) | easy (n={len(g_e)}) mean = {np.mean(g_e):.4f}")
    if mw:
        print(f"        Mann-Whitney one-sided (strong > easy): U={mw.statistic:.0f}, p={mw.pvalue:.4f}")
    b8a = bool(len(g_s) and np.mean(g_s) >= 0.05 and mw and mw.pvalue < 0.05)
    print(f"        VERDICT: {'CONFIRMED — B8 손실 존재' if b8a else 'NOT confirmed'}")

    # H-B8b
    allv = gap_df.dropna(subset=['gap_mixed', 'mix_rate'])
    if len(allv) >= 5:
        pr = stats.pearsonr(allv['mix_rate'], allv['gap_mixed'])
        sr = stats.spearmanr(allv['mix_rate'], allv['gap_mixed'])
        is_ph = (allv['category'] == 'phantom17').astype(float)
        pb = stats.pearsonr(allv['mix_rate'], is_ph)
        print(f"\n[H-B8b] corr(mix_rate, gap_mixed): pearson r={pr[0]:.3f} (p={pr[1]:.3f}), "
              f"spearman ρ={sr[0]:.3f} (p={sr[1]:.3f})")
        print(f"        point-biserial mix_rate vs phantom: r={pb[0]:.3f} (p={pb[1]:.3f})")
        print(f"        phantom mean mix_rate={phant['mix_rate'].mean():.3f}, "
              f"easy={easy['mix_rate'].mean():.3f}")
        b8b = bool(pr[0] > 0.4 and pr[1] < 0.05)
        print(f"        VERDICT: {'CONFIRMED — 가설 predict (r>0.4) 적중' if b8b else 'NOT confirmed (r<=0.4)'}")

    # H-B8c — only meaningful if a real gap exists (pre-registered branch 3:
    # "gap 자체가 작으면 → within-view 무죄, B1 재조준")
    if len(g_s) and np.mean(g_s) < 0.05:
        print(f"\n[H-B8c] N/A — strong_signal mean gap ({np.mean(g_s):.4f}) < 0.05: "
              f"within-view 손실 자체가 작음")
        print("        VERDICT: B1 RE-AIM — within-view 무죄, across-view "
              "(visibility-weighted accumulation) 재조준")
    else:
        rec_s = strong.dropna(subset=['recovery_frac'])['recovery_frac'].values
        if len(rec_s):
            med_rec = float(np.median(rec_s))
            mean_rec = float(np.mean(rec_s))
            print(f"\n[H-B8c] strong_signal recovery: median={med_rec:.2f}, mean={mean_rec:.2f} "
                  f"(per-SP: {[f'{x:.2f}' for x in sorted(rec_s)]})")
            if med_rec >= 0.7:
                verdict_c = "GO — Stage 3.2 hard-assignment full re-run"
            elif med_rec < 0.3:
                verdict_c = "B2 PIVOT — argmax mask 자체 오염 (ratio-sum 무죄)"
            else:
                verdict_c = "PARTIAL — 30-70%: hard-assignment는 부분 해법, 혼합 전략 검토"
            print(f"        VERDICT: {verdict_c}")

    # ---------------- plots
    os.makedirs(args.plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    plot_df = gap_df.dropna(subset=['clean_cos_mean']).sort_values(
        ['category', 'clean_cos_mean'], ascending=[True, False]).reset_index(drop=True)
    x = np.arange(len(plot_df))
    ax.scatter(x, plot_df['clean_cos_mean'], marker='o', s=60, label='clean (H2-lite)', color='tab:green')
    ax.scatter(x, plot_df['mixed_cos_mean'], marker='s', s=60, label='mixed (post-B8)', color='tab:red')
    ax.scatter(x, plot_df['hard_cos_mean'], marker='^', s=60, label='hard (argmax preview)', color='tab:blue')
    for i, r in plot_df.iterrows():
        ax.plot([i, i], [r['mixed_cos_mean'], r['clean_cos_mean']], color='gray', lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['scene'][:4]}/{r['prompt'][:18]}\n[{str(r['subtype'])[:12]}]"
                        for _, r in plot_df.iterrows()], rotation=90, fontsize=7)
    ax.set_ylabel("cos with prompt (paired views)")
    ax.set_title("Stage 3.1 — clean vs mixed(post-B8) vs hard(argmax) per target SP")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "stage3_clean_vs_mixed_vs_hard.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for cat, col, mk in (('phantom17', 'tab:red', 'o'), ('easy_sample', 'tab:green', 's')):
        d = gap_df[gap_df['category'] == cat].dropna(subset=['gap_mixed', 'mix_rate'])
        ax.scatter(d['mix_rate'], d['gap_mixed'], c=col, marker=mk, s=70, label=cat, alpha=0.8)
        for _, r in d.iterrows():
            ax.annotate(r['prompt'][:12], (r['mix_rate'], r['gap_mixed']), fontsize=6, alpha=0.7)
    if len(allv) >= 5:
        ax.set_title(f"mix_rate vs gap_mixed (pearson r={pr[0]:.2f})")
    ax.set_xlabel("B8 mix_rate (frac. views with mix_count >= 2)")
    ax.set_ylabel("gap = clean_cos - mixed_cos")
    ax.axhline(0, color='black', lw=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "stage3_mixrate_vs_gap.png"), dpi=120)
    plt.close(fig)

    print(f"\nWrote {args.out_mix_csv}, {args.out_gap_csv}")
    print(f"Plots in {args.plot_dir}/")


if __name__ == "__main__":
    main()
