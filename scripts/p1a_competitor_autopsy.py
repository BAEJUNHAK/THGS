"""
P1-A — Competitor autopsy (pre-registered R11, see
md/hypotheses/strategy/competitor_autopsy.md — read-only registration).

Evaluates competitor prescription FAMILIES as faithful aggregation rules on the
same all-SP per-view dumps used by Stage 3.3 (THGS) and Stage 5 (ReLaGS):

  CA-1  geometric median (VALA-faithful, robust-statistics family)
        gm_u : unweighted Weiszfeld on the unit sphere      (20 it, eps 1e-6)
        gm_w : visibility-portion-weighted Weiszfeld
        gm_g : drop bottom-25%-portion valid views, then weighted Weiszfeld
               (alpha*T visibility-gating approximation)
  CA-2  bag-of-embeddings (Beyond-Averages-faithful, view-selection family)
        qmax1: score = max over valid views of canon(f_v)   (purest bag form)
        top5 : query-top-5 portion-weighted agg (recomputed in-harness;
               equals stage3_3 'rank_p' / stage5_g2 'rank_top5')

Fidelity gate: in-harness mean baseline (canon over sai_nag feats, levels
[2,3]) must reproduce the diagnostic CSV oracle_rank (mismatches reported;
results marked invalid if gate fails badly).

Outputs:
  output/diagnostics/ca1_gm_ranks_{method}.csv
  output/diagnostics/ca2_qmax_ranks_{method}.csv
  output/diagnostics/p1a_selections_{method}.pkl   (top-3 per prompt x rule)
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage3_3_fullpool_rank import load_dump, per_level_arrays, query_topk_agg, LEVELS
from stage3_2_common import make_vlm

SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']

METHODS = {
    'thgs': dict(
        dump_tpl='output/diagnostics/stage3_3_allsp_{}.pkl',
        ref_csv='output/diagnostics/b7_a4_combined.csv',
        nag_tpl='output/lerf/{}/sai_nag.pt',
        phantom='persistent17',
    ),
    'relags': dict(
        dump_tpl='output/diagnostics/stage5_relags_allsp_{}.pkl',
        ref_csv='output/diagnostics/b7_a4_combined_relags.csv',
        nag_tpl='ReLaGS/output/lerf_hf/scenes/LeRF/{}/sai_nag.pt',
        phantom='relags_class',
    ),
}


@torch.no_grad()
def geometric_median(feats, por, valid, weighted, gate_q=None, iters=20, eps=1e-6):
    """Weiszfeld on (V,S,512) -> (S,512), L2-normalized. No-view SPs -> 0."""
    f = feats.float()
    if gate_q is not None:
        porq = por.clone()
        porq[~valid] = float('nan')
        q = torch.nanquantile(porq, gate_q, dim=0, keepdim=True)
        valid = valid & (por >= q)
    w_base = (por if weighted else torch.ones_like(por)) * valid
    y = (f * w_base.unsqueeze(-1)).sum(0)
    y = F.normalize(y, p=2, dim=-1)
    for _ in range(iters):
        d = (f - y.unsqueeze(0)).norm(dim=-1).clamp_min(eps)
        w = w_base / d
        y = (f * w.unsqueeze(-1)).sum(0) / w.sum(0).clamp_min(1e-9).unsqueeze(-1)
    y = F.normalize(y, p=2, dim=-1)
    y[w_base.sum(0) == 0] = 0
    return y


@torch.no_grad()
def qmax_scores(feats, valid, vlm, chunk=200_000):
    """canon for EVERY (view, SP) feature, then max over valid views.
    Returns (S,) scores and (S,) argmax view index."""
    V, S, D = feats.shape
    flat = feats.float().reshape(V * S, D)
    out = torch.empty(V * S, device=flat.device)
    for i in range(0, V * S, chunk):
        out[i:i + chunk] = vlm.compute_similarity(flat[i:i + chunk])
    c = out.reshape(V, S).masked_fill(~valid, -1e9)
    best, arg = c.max(dim=0)
    best = best.masked_fill(valid.sum(0) == 0, 0.5)  # ghost: canon(0)=0.5 mirror
    return best.cpu().numpy(), arg.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=['thgs', 'relags'])
    args = ap.parse_args()
    vlm = make_vlm()

    for method in args.methods:
        cfg = METHODS[method]
        ref = pd.read_csv(cfg['ref_csv'])
        ref = ref[ref['is_ref_frame'] == 1]
        if cfg['phantom'] == 'persistent17':
            ph = pd.read_csv('output/diagnostics/persistent_phantoms_17.csv')
            pset = set(map(tuple, ph[['scene', 'prompt']].values))
        else:
            cm = pd.read_csv('output/diagnostics/cross_method_d2_decomposition.csv')
            pset = set(map(tuple, cm[cm.relags_class == 'phantom'][['scene', 'prompt']].values))

        rows, selections, gate_bad = [], {}, []
        for scene in SCENES:
            dump = load_dump(cfg['dump_tpl'].format(scene))
            lv = {l: per_level_arrays(dump, l, 'cuda') for l in LEVELS}
            entries = [(l, i) for l in LEVELS for i in range(lv[l][0].shape[1])]
            index = {e: i for i, e in enumerate(entries)}
            nval = np.concatenate([lv[l][4] for l in LEVELS]).astype(int)
            nag = torch.load(cfg['nag_tpl'].format(scene))
            base_feat = torch.cat([
                F.normalize(nag['nag_feat'][l - 1].cuda().float(), p=2, dim=-1)
                for l in LEVELS])

            # query-independent aggregations once per scene
            gm = {}
            for name, (wt, gq) in dict(gm_u=(False, None), gm_w=(True, None),
                                       gm_g=(True, 0.25)).items():
                gm[name] = torch.cat([
                    geometric_median(*lv[l][:3], weighted=wt, gate_q=gq)
                    for l in LEVELS])

            sc_ref = ref[ref['scene'] == scene]
            print(f"\n=== [{method}] {scene}: {len(sc_ref)} prompts, "
                  f"pool={len(entries)} ===", flush=True)
            for _, r in sc_ref.iterrows():
                prompt = r['prompt']
                okey = (int(r['oracle_lvl']), int(r['oracle_sp_id']))
                oi = index[okey]
                vlm.encode_text(prompt)
                text = vlm.text_feature[0].float()

                s_mean = vlm.compute_similarity(base_feat).cpu().numpy()
                r_base = int((s_mean > s_mean[oi]).sum()) + 1
                csv_rank = int(r['oracle_rank'])
                if r_base != csv_rank:
                    gate_bad.append((scene, prompt, csv_rank, r_base))

                scores = {}
                top5 = torch.cat([query_topk_agg(*lv[l][:3], text, k=5)
                                  for l in LEVELS])
                scores['top5'] = vlm.compute_similarity(top5).cpu().numpy()
                qs, qarg = [], []
                off = 0
                for l in LEVELS:
                    s_, a_ = qmax_scores(lv[l][0], lv[l][2], vlm)
                    qs.append(s_)
                    qarg.append(a_)
                    off += lv[l][0].shape[1]
                scores['qmax1'] = np.concatenate(qs)
                for name in gm:
                    scores[name] = vlm.compute_similarity(gm[name]).cpu().numpy()

                cat = ('phantom' if (scene, prompt) in pset
                       else 'easy' if csv_rank <= 3 else 'other')
                row = {'scene': scene, 'prompt': prompt, 'category': cat,
                       'baseline_rank': r_base, 'csv_rank': csv_rank,
                       'oracle_n_views': int(nval[oi])}
                sel = {}
                for name, s in scores.items():
                    rank = int((s > s[oi]).sum()) + 1
                    row[f'rank_{name}'] = rank
                    order = np.argsort(-s)[:3]
                    sel[name] = [entries[i] for i in order]
                rows.append(row)
                selections[(scene, prompt)] = sel
                print(f"  {cat:7s} {prompt:26s} base={r_base:4d} "
                      + " ".join(f"{n}={row[f'rank_{n}']:4d}"
                                 for n in ['top5', 'qmax1', 'gm_u', 'gm_w', 'gm_g']),
                      flush=True)
            del lv, base_feat, gm
            torch.cuda.empty_cache()

        df = pd.DataFrame(rows)
        gm_cols = ['rank_gm_u', 'rank_gm_w', 'rank_gm_g']
        df[['scene', 'prompt', 'category', 'baseline_rank', 'csv_rank',
            'oracle_n_views'] + gm_cols].to_csv(
            f'output/diagnostics/ca1_gm_ranks_{method}.csv', index=False)
        df[['scene', 'prompt', 'category', 'baseline_rank', 'csv_rank',
            'oracle_n_views', 'rank_top5', 'rank_qmax1']].to_csv(
            f'output/diagnostics/ca2_qmax_ranks_{method}.csv', index=False)
        with open(f'output/diagnostics/p1a_selections_{method}.pkl', 'wb') as fp:
            pickle.dump(selections, fp)

        print("\n" + "=" * 78)
        print(f"[{method}] FIDELITY GATE: {len(gate_bad)} mismatch "
              f"(in-harness mean rank vs CSV oracle_rank)")
        for g in gate_bad[:10]:
            print(f"    {g}")
        print(f"[{method}] SUMMARY (recover: base>3 -> rule<=3 / "
              f"regress: base<=3 -> rule>3)")
        for name in ['top5', 'qmax1', 'gm_u', 'gm_w', 'gm_g']:
            line = f"  {name:6s}"
            for grp in ['phantom', 'easy', 'other']:
                g = df[df.category == grp]
                rec = int(((g.baseline_rank > 3) & (g[f'rank_{name}'] <= 3)).sum())
                reg = int(((g.baseline_rank <= 3) & (g[f'rank_{name}'] > 3)).sum())
                n3 = int((g[f'rank_{name}'] <= 3).sum())
                line += f" | {grp} n={len(g)}: <=3 {n3:2d} (rec {rec:+d}/reg -{reg})"
            print(line)
        print("=" * 78, flush=True)


if __name__ == '__main__':
    main()
