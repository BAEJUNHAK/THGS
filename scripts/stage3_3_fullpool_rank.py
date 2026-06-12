"""
Stage 3.3-A2 — full-pool query-aware re-scoring (NO frozen competitors).

For every (prompt, SP) at levels [2,3]:
    agg(SP)  = normalize( sum_{v in top-5 views by query-cos} f_v * portion_v )
    score    = canon-contrast(agg, prompt)        [ClipSimMeasure, same as A4]

Variants:
  p  plain        : SPs with no valid view keep agg=0 -> canon(0)=0.5 competes
                    (mirrors the pipeline's D1-ghost behavior)
  z  zero-filter  : SPs with no valid view are excluded from the pool
  c  coherence    : z  +  score - lambda * coherence(SP), lambda in {0.1,0.3,0.5}
                    coherence = ||sum(f_v * w_v)|| / sum(w_v) over valid views
                    (the impostor signature from B1.C — homogeneity penalty)

Outputs:
  output/diagnostics/stage3_3_fullpool_ranks.csv   (per prompt x variant ranks)
  output/diagnostics/stage3_3_top3_selections.pkl  (top-3 (lvl,sp) per prompt x
                                                    variant — input to mask eval)
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from stage3_2_common import make_vlm

LEVELS = [2, 3]
LAMBDAS = [0.1, 0.3, 0.5]
TOPK = 5


def load_dump(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


@torch.no_grad()
def per_level_arrays(dump, lvl, device):
    d = dump['levels'][lvl]
    feats = torch.from_numpy(d['feats']).to(device)          # (V,S,512) fp16
    por = torch.from_numpy(d['portions']).to(device).float() # (V,S)
    valid = (por > 0) & (feats.float().abs().sum(-1) > 1e-6) # (V,S)
    # coherence over ALL valid views (B1.C definition)
    fw = feats.float() * por.unsqueeze(-1) * valid.unsqueeze(-1)
    resultant = fw.sum(dim=0)                                # (S,512)
    wsum = (por * valid).sum(dim=0).clamp_min(1e-9)          # (S,)
    coherence = resultant.norm(dim=-1) / wsum                # (S,)
    n_valid = valid.sum(dim=0)                               # (S,)
    return feats, por, valid, coherence.cpu().numpy(), n_valid.cpu().numpy()


@torch.no_grad()
def query_topk_agg(feats, por, valid, text, k=TOPK):
    """(V,S,512),(V,S),(V,S),(512,) -> (S,512) float32 normalized."""
    V, S, D = feats.shape
    qcos = torch.einsum('vsd,d->vs', feats.float(), text)    # (V,S)
    qcos = qcos.masked_fill(~valid, -1e4)
    kk = min(k, V)
    top_idx = qcos.topk(kk, dim=0).indices                   # (kk,S)
    f_sv = feats.float().permute(1, 0, 2)                    # (S,V,512)
    p_sv = (por * valid).permute(1, 0)                       # (S,V)
    idx_sv = top_idx.permute(1, 0)                           # (S,kk)
    g_f = torch.gather(f_sv, 1, idx_sv.unsqueeze(-1).expand(S, kk, D))
    g_p = torch.gather(p_sv, 1, idx_sv)
    agg = (g_f * g_p.unsqueeze(-1)).sum(dim=1)               # (S,512)
    agg = torch.nn.functional.normalize(agg, p=2, dim=-1)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump_tpl', default='output/diagnostics/stage3_3_allsp_{}.pkl')
    ap.add_argument('--b7_csv', default='output/diagnostics/b7_a4_combined.csv')
    ap.add_argument('--persistent_csv', default='output/diagnostics/persistent_phantoms_17.csv')
    ap.add_argument('--out_csv', default='output/diagnostics/stage3_3_fullpool_ranks.csv')
    ap.add_argument('--out_sel', default='output/diagnostics/stage3_3_top3_selections.pkl')
    args = ap.parse_args()

    device = 'cuda'
    b7 = pd.read_csv(args.b7_csv)
    ref = b7[b7['is_ref_frame'] == 1]
    ph = pd.read_csv(args.persistent_csv)
    phantom_set = set(map(tuple, ph[['scene', 'prompt']].values))

    vlm = make_vlm()
    rows, selections = [], {}

    for scene in ['figurines', 'ramen', 'teatime', 'waldo_kitchen']:
        dump = load_dump(args.dump_tpl.format(scene))
        lv = {lvl: per_level_arrays(dump, lvl, device) for lvl in LEVELS}
        entries = []
        for lvl in LEVELS:
            S = lv[lvl][0].shape[1]
            entries += [(lvl, i) for i in range(S)]
        index = {e: i for i, e in enumerate(entries)}
        coher = np.concatenate([lv[lvl][3] for lvl in LEVELS])
        nval = np.concatenate([lv[lvl][4] for lvl in LEVELS])

        sc_ref = ref[ref['scene'] == scene]
        print(f"\n=== {scene}: {len(sc_ref)} prompts, pool={len(entries)} ===", flush=True)
        for _, r in sc_ref.iterrows():
            prompt = r['prompt']
            o_key = (int(r['oracle_lvl']), int(r['oracle_sp_id']))
            vlm.encode_text(prompt)
            text = vlm.text_feature[0].float()

            aggs, canon = [], []
            for lvl in LEVELS:
                feats, por, valid, _, _ = lv[lvl]
                agg = query_topk_agg(feats, por, valid, text)
                aggs.append(agg)
                canon.append(vlm.compute_similarity(agg).cpu().numpy())
            scores_p = np.concatenate(canon)
            has_view = nval > 0

            def rank_and_top3(scores, mask=None):
                s = scores.copy()
                if mask is not None:
                    s[~mask] = -1e9
                order = np.argsort(-s)
                oi = index[o_key]
                rank = int(np.where(order == oi)[0][0]) + 1
                top3 = [entries[i] for i in order[:3]]
                return rank, top3

            cat = ('phantom17' if (scene, prompt) in phantom_set
                   else 'easy' if int(r['oracle_rank']) <= 3 else 'other')
            row = {'scene': scene, 'prompt': prompt, 'category': cat,
                   'baseline_rank': int(r['oracle_rank']),
                   'oracle_n_valid_views': int(nval[index[o_key]])}
            sel = {}
            row['rank_p'], sel['p'] = rank_and_top3(scores_p)
            row['rank_z'], sel['z'] = rank_and_top3(scores_p, has_view)
            for lam in LAMBDAS:
                sc_c = scores_p - lam * coher
                row[f'rank_c{lam}'], sel[f'c{lam}'] = rank_and_top3(sc_c, has_view)
            rows.append(row)
            selections[(scene, prompt)] = sel
            print(f"  {cat[:7]:7s} {prompt:26s} base={row['baseline_rank']:4d} "
                  f"p={row['rank_p']:4d} z={row['rank_z']:4d} "
                  + " ".join(f"c{lam}={row[f'rank_c{lam}']:4d}" for lam in LAMBDAS),
                  flush=True)
        del lv
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    with open(args.out_sel, 'wb') as f:
        pickle.dump(selections, f)

    print("\n" + "=" * 78)
    print("FULL-POOL QUERY-AWARE — rank summary (no frozen competitors)")
    print("=" * 78)
    variants = ['baseline_rank', 'rank_p', 'rank_z'] + [f'rank_c{l}' for l in LAMBDAS]
    for grp in ['phantom17', 'easy', 'other']:
        g = df[df['category'] == grp]
        line = f"  {grp:9s} (n={len(g):2d})  " + "  ".join(
            f"{v.replace('rank_', '').replace('baseline_rank', 'base')}:"
            f"{int((g[v] <= 3).sum()):2d}<=3" for v in variants)
        print(line)
    print(f"\nWrote {args.out_csv}, {args.out_sel}")


if __name__ == '__main__':
    main()
