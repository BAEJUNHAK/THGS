"""
Stage 2B robustness verification — 3 checks (read-only on prior artifacts).

Check 1 — F2.B/C tau sweep: re-run ROFA simulation + subtype/pathology
  classification at tau in {1.0, 2.0}. tau=1.0 reproduces the original
  stage2b_rofa_subtypes.csv; tau=2.0 matches the actual ReLaGS pipeline
  default (merge_proj.py --tau default=2).

Check 2 — Joint 2x2 classification metric robustness: re-classify
  easy/phantom/real/rare using A2 canon_rank_true instead of raw_rank_true
  (mask policy), for both THGS and ReLaGS; recompute the persistent
  phantom set and diff against persistent_phantoms_17.csv.

Check 3 — H2-lite rank-based strong-signal: for each phantom's per-view
  clean SP-crop CLIP feature, rank the true prompt among ALL scene prompts
  (raw cosine, same semantics as A2 raw rank). Stronger evidence than the
  absolute cos>=0.20 threshold.

Outputs:
  output/diagnostics/verify_f2_tau_sweep.csv
  output/diagnostics/verify_joint_canon_vs_raw.csv
  output/diagnostics/verify_h2lite_rank.csv
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- Check 1
def simulate_rofa(features, tau):
    N = features.shape[0]
    if N == 1:
        return np.array([True]), np.array([1.0]), 1.0, 0.0
    cos_sim = features @ features.T
    mean_sim = (cos_sim.sum(axis=1) - 1) / (N - 1)
    mu = mean_sim.mean()
    sigma = mean_sim.std()
    keep_mask = mean_sim > (mu - tau * sigma)
    return keep_mask, mean_sim, mu, sigma


def classify_subtype(features, cos_with_prompt, kept_mask):
    """Verbatim logic from stage2b_f2_subtypes.py:classify_subtype."""
    N = features.shape[0]
    if N < 5:
        return 'insufficient_evidence'
    cos_mean = float(np.mean(cos_with_prompt))
    cos_std = float(np.std(cos_with_prompt))
    cos_iqr = float(np.percentile(cos_with_prompt, 75) - np.percentile(cos_with_prompt, 25))
    kept_cos_mean = float(cos_with_prompt[kept_mask].mean()) if kept_mask.any() else np.nan
    dropped_cos_mean = float(cos_with_prompt[~kept_mask].mean()) if (~kept_mask).any() else np.nan

    if cos_iqr > 0.06 and cos_std > 0.04:
        kept_std = float(np.std(cos_with_prompt[kept_mask])) if kept_mask.sum() > 1 else 0.0
        if kept_std > 0.03:
            return 'bimodal_balanced'
        if not np.isnan(dropped_cos_mean) and dropped_cos_mean < kept_cos_mean - 0.04:
            return 'outlier_handled'
        return 'outlier_partial'
    if cos_mean < 0.20:
        return 'mean_dilution'
    return 'strong_signal_phantom'


def check1_tau_sweep(pkl_path, out_csv):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    rows = []
    for (sc, prompt), views in sorted(data.items()):
        if not views:
            continue
        features = np.stack([v['feature'] for v in views], axis=0)
        cos = np.array([v['cos_with_prompt'] for v in views])
        row = {'scene': sc, 'prompt': prompt, 'n_views': len(views),
               'cos_mean': float(cos.mean())}
        for tau in (1.0, 2.0):
            kept, mean_sim, mu, sigma = simulate_rofa(features, tau)
            n_drop = int((~kept).sum())
            kept_cos = float(cos[kept].mean()) if kept.any() else np.nan
            drop_cos = float(cos[~kept].mean()) if (~kept).any() else np.nan
            if np.isnan(drop_cos):
                pathology = 'no_drops'
            elif drop_cos > kept_cos:
                pathology = 'ROFA_dropped_good_views'
            else:
                pathology = 'ROFA_kept_good_views'
            subtype = classify_subtype(features, cos, kept)
            t = f'tau{tau:.0f}'
            row[f'{t}_n_dropped'] = n_drop
            row[f'{t}_kept_cos'] = round(kept_cos, 4)
            row[f'{t}_dropped_cos'] = round(drop_cos, 4) if not np.isnan(drop_cos) else np.nan
            row[f'{t}_pathology'] = pathology
            row[f'{t}_subtype'] = subtype
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("=" * 78)
    print("CHECK 1 — F2.B/C tau sweep (tau=1.0 original sim vs tau=2.0 pipeline default)")
    print("=" * 78)
    for t in ('tau1', 'tau2'):
        print(f"\n--- {t} pathology distribution ---")
        print(df[f'{t}_pathology'].value_counts().to_string())
        print(f"--- {t} subtype distribution ---")
        print(df[f'{t}_subtype'].value_counts().to_string())
    changed = df[(df['tau1_pathology'] != df['tau2_pathology']) |
                 (df['tau1_subtype'] != df['tau2_subtype'])]
    print(f"\n--- per-phantom changes (tau1 -> tau2): {len(changed)} rows ---")
    for _, r in changed.iterrows():
        print(f"  {r['scene']:14s} {r['prompt']:25s} "
              f"drop {r['tau1_n_dropped']}->{r['tau2_n_dropped']}  "
              f"pathology {r['tau1_pathology']} -> {r['tau2_pathology']}  "
              f"subtype {r['tau1_subtype']} -> {r['tau2_subtype']}")
    print(f"\nWrote {out_csv}")
    return df


# ---------------------------------------------------------------- Check 2
def classify_2x2(a2_rank, sp_rank):
    if a2_rank <= 3 and sp_rank <= 3:
        return 'easy'
    if a2_rank <= 3 and sp_rank > 3:
        return 'phantom'
    if a2_rank > 3 and sp_rank > 3:
        return 'real'
    return 'rare'


def check2_canon_vs_raw(thgs_csv, relags_csv, a2_csv, persistent_csv, out_csv):
    a2 = pd.read_csv(a2_csv)
    a2m = a2[a2['policy'] == 'mask'][['scene', 'prompt', 'raw_rank_true', 'canon_rank_true']].copy()
    for c in ('raw_rank_true', 'canon_rank_true'):
        a2m[c] = pd.to_numeric(a2m[c], errors='coerce').fillna(99).astype(float)

    frames = {}
    for name, path in (('thgs', thgs_csv), ('relags', relags_csv)):
        b = pd.read_csv(path)
        ref = b[b['is_ref_frame'] == 1][['scene', 'prompt', 'oracle_rank']].copy()
        j = ref.merge(a2m, on=['scene', 'prompt'], how='left')
        j['raw_rank_true'] = j['raw_rank_true'].fillna(99)
        j['canon_rank_true'] = j['canon_rank_true'].fillna(99)
        j[f'{name}_class_raw'] = [classify_2x2(a, s) for a, s in zip(j['raw_rank_true'], j['oracle_rank'])]
        j[f'{name}_class_canon'] = [classify_2x2(a, s) for a, s in zip(j['canon_rank_true'], j['oracle_rank'])]
        frames[name] = j.rename(columns={'oracle_rank': f'{name}_sp_rank'})

    m = frames['thgs'].merge(
        frames['relags'][['scene', 'prompt', 'relags_sp_rank', 'relags_class_raw', 'relags_class_canon']],
        on=['scene', 'prompt'], how='inner')
    m.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print("CHECK 2 — Joint 2x2: A2 raw_rank_true (original) vs canon_rank_true")
    print("=" * 78)
    order = ['easy', 'phantom', 'real', 'rare']
    for name in ('thgs', 'relags'):
        print(f"\n--- {name.upper()} class counts (n={len(m)}) ---")
        for cls in order:
            n_raw = int((m[f'{name}_class_raw'] == cls).sum())
            n_can = int((m[f'{name}_class_canon'] == cls).sum())
            mark = '' if n_raw == n_can else '   <-- changed'
            print(f"  {cls:8s} raw={n_raw:3d}  canon={n_can:3d}{mark}")

    moved = m[m['thgs_class_raw'] != m['thgs_class_canon']]
    print(f"\n--- THGS prompts whose class changes under canon ({len(moved)}) ---")
    for _, r in moved.iterrows():
        print(f"  {r['scene']:14s} {r['prompt']:25s} "
              f"a2 raw={r['raw_rank_true']:.0f} canon={r['canon_rank_true']:.0f} "
              f"sp_rank={r['thgs_sp_rank']:.0f}  {r['thgs_class_raw']} -> {r['thgs_class_canon']}")

    pers_raw = set(map(tuple, m[(m['thgs_class_raw'] == 'phantom') &
                                (m['relags_class_raw'] == 'phantom')][['scene', 'prompt']].values))
    pers_can = set(map(tuple, m[(m['thgs_class_canon'] == 'phantom') &
                                (m['relags_class_canon'] == 'phantom')][['scene', 'prompt']].values))
    pers_file = set(map(tuple, pd.read_csv(persistent_csv)[['scene', 'prompt']].values))
    print(f"\npersistent phantom set: file={len(pers_file)}  raw-recomputed={len(pers_raw)}  canon={len(pers_can)}")
    print(f"  file vs raw-recomputed diff : {sorted(pers_file ^ pers_raw) or 'none'}")
    print(f"  canon-only additions        : {sorted(pers_can - pers_raw) or 'none'}")
    print(f"  canon-dropped               : {sorted(pers_raw - pers_can) or 'none'}")
    print(f"\nWrote {out_csv}")
    return m


# ---------------------------------------------------------------- Check 3
def check3_h2lite_rank(pkl_path, a2_csv, out_csv):
    import torch
    import open_clip
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n" + "=" * 78)
    print("CHECK 3 — H2-lite rank of true prompt among scene prompts (per view)")
    print("=" * 78)
    print(f"Loading CLIP ViT-B-16 laion2b_s34b_b88k on {device} ...")
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='laion2b_s34b_b88k',
        precision='fp16' if device == 'cuda' else 'fp32')
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-16')

    a2 = pd.read_csv(a2_csv)
    scene_prompts = {sc: sorted(g['prompt'].unique()) for sc, g in a2.groupby('scene')}

    text_feats = {}
    with torch.no_grad():
        for sc, prompts in scene_prompts.items():
            tok = tokenizer(prompts).to(device)
            tf = model.encode_text(tok).float()
            tf = tf / tf.norm(dim=-1, keepdim=True)
            text_feats[sc] = tf.cpu().numpy()

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    rows = []
    for (sc, prompt), views in sorted(data.items()):
        if not views or sc not in scene_prompts:
            continue
        prompts = scene_prompts[sc]
        if prompt not in prompts:
            print(f"  [warn] {sc}/{prompt} not in A2 prompt list, skipping")
            continue
        true_idx = prompts.index(prompt)
        feats = np.stack([v['feature'] for v in views], axis=0)   # (V, 512)
        sims = feats @ text_feats[sc].T                            # (V, P)
        ranks = (sims > sims[:, true_idx:true_idx + 1]).sum(axis=1) + 1
        cos = np.array([v['cos_with_prompt'] for v in views])
        rows.append({
            'scene': sc, 'prompt': prompt, 'n_views': len(views),
            'n_scene_prompts': len(prompts),
            'cos_mean': round(float(cos.mean()), 4),
            'rank_mean': round(float(ranks.mean()), 2),
            'rank_median': float(np.median(ranks)),
            'rank_min': int(ranks.min()), 'rank_max': int(ranks.max()),
            'frac_views_rank1': round(float((ranks == 1).mean()), 3),
            'frac_views_rank_le3': round(float((ranks <= 3).mean()), 3),
            'strong_by_cos020': bool(cos.mean() >= 0.20),
            'strong_by_medrank_le3': bool(np.median(ranks) <= 3),
        })
    df = pd.DataFrame(rows).sort_values('rank_median')
    df.to_csv(out_csv, index=False)

    print(f"\n{'scene':14s} {'prompt':25s} {'cos':>6s} {'medR':>5s} {'mean':>5s} "
          f"{'%R1':>5s} {'%R<=3':>6s}  cos>=.20 medR<=3")
    for _, r in df.iterrows():
        print(f"{r['scene']:14s} {r['prompt']:25s} {r['cos_mean']:6.3f} "
              f"{r['rank_median']:5.1f} {r['rank_mean']:5.1f} "
              f"{r['frac_views_rank1']*100:4.0f}% {r['frac_views_rank_le3']*100:5.0f}%  "
              f"{'Y' if r['strong_by_cos020'] else '.':>7s} {'Y' if r['strong_by_medrank_le3'] else '.':>7s}")

    n_cos = int(df['strong_by_cos020'].sum())
    n_rank = int(df['strong_by_medrank_le3'].sum())
    both = int((df['strong_by_cos020'] & df['strong_by_medrank_le3']).sum())
    print(f"\nstrong by cos>=0.20      : {n_cos}/{len(df)}")
    print(f"strong by median rank<=3 : {n_rank}/{len(df)}")
    print(f"agree (both)             : {both}/{len(df)}")
    disagree = df[df['strong_by_cos020'] != df['strong_by_medrank_le3']]
    if len(disagree):
        print("disagreements:")
        for _, r in disagree.iterrows():
            print(f"  {r['scene']}/{r['prompt']}: cos={r['cos_mean']:.3f} medRank={r['rank_median']}")
    print(f"\nWrote {out_csv}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='output/diagnostics/_h2_lite_perview.pkl')
    ap.add_argument('--a2_csv', default='output/diagnostics/a2_image_clip_ceiling.csv')
    ap.add_argument('--thgs_csv', default='output/diagnostics/b7_a4_combined.csv')
    ap.add_argument('--relags_csv', default='output/diagnostics/b7_a4_combined_relags.csv')
    ap.add_argument('--persistent_csv', default='output/diagnostics/persistent_phantoms_17.csv')
    ap.add_argument('--out_dir', default='output/diagnostics')
    ap.add_argument('--checks', default='1,2,3')
    args = ap.parse_args()

    checks = set(args.checks.split(','))
    if '1' in checks:
        check1_tau_sweep(args.pkl, os.path.join(args.out_dir, 'verify_f2_tau_sweep.csv'))
    if '2' in checks:
        check2_canon_vs_raw(args.thgs_csv, args.relags_csv, args.a2_csv,
                            args.persistent_csv,
                            os.path.join(args.out_dir, 'verify_joint_canon_vs_raw.csv'))
    if '3' in checks:
        check3_h2lite_rank(args.pkl, args.a2_csv,
                           os.path.join(args.out_dir, 'verify_h2lite_rank.csv'))


if __name__ == '__main__':
    main()
