"""
Stage 3.2 — adversarial check on B1.B's query-conditioned result.

B1.B's 15/17 recovery upgraded ONLY the target SP while freezing all
competitors at their pipeline mean features. A real query-aware method would
re-score every SP. This script runs the FAIR DUEL for the known killer
competitor: both the oracle SP and its wrong-top1 SP get query-conditioned
top-5 aggregation; rest of the pool frozen.

Measures per phantom pair:
  - duel winner (canon-contrast score, both upgraded)
  - O_top5cos / W_top5cos (raw top-5 mean prompt-cos per side)
  - fair_rank: oracle rank with BOTH entries upgraded (rest frozen)

Caveats (still optimistic): only the one known competitor is upgraded —
full-pool re-scoring could surface dark horses; rank<=3 means inclusion in the
top-3 union, not exclusive victory (wrong SP stays rank 1 in lost duels).

Output: output/diagnostics/stage3_2_fair_duel.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stage3_2_common import load_replay, visible_views, ScenePool, make_vlm


def topk_agg(views, text, k=5):
    feats = np.stack([v['mixed_feat'].astype(np.float32) for v in views])
    por = np.array([v['portion'] for v in views])
    qcos = feats @ text
    idx = np.argsort(-qcos)[:k]
    acc = (feats[idx].astype(np.float64) * por[idx, None]).sum(0)
    n = np.linalg.norm(acc)
    return (acc / n).astype(np.float32), float(np.sort(qcos)[-k:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oracle_pkl', default='output/diagnostics/stage3_b8_replay_perview.pkl')
    ap.add_argument('--wrong_pkl', default='output/diagnostics/stage3_2_wrongtop1_perview.pkl')
    ap.add_argument('--persistent_csv', default='output/diagnostics/persistent_phantoms_17.csv')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out_csv', default='output/diagnostics/stage3_2_fair_duel.csv')
    args = ap.parse_args()

    oracle = load_replay(args.oracle_pkl)
    wrong = load_replay(args.wrong_pkl)
    ph = pd.read_csv(args.persistent_csv)
    vlm = make_vlm()
    pools = {}
    rows = []
    for _, r in ph.iterrows():
        sc, prompt = r['scene'], r['prompt']
        if (sc, prompt) not in oracle or (sc, prompt) not in wrong:
            continue
        if sc not in pools:
            pools[sc] = ScenePool(f"output/lerf/{sc}", vlm)
        pool = pools[sc]
        o_rec, w_rec = oracle[(sc, prompt)], wrong[(sc, prompt)]
        ov, wv = visible_views(o_rec), visible_views(w_rec)
        if not ov or not wv:
            rows.append({'scene': sc, 'prompt': prompt, 'note': 'skip_empty_views'})
            continue
        _, _, text = pool.prompt_scores(prompt)
        o_agg, o_t5 = topk_agg(ov, text, args.k)
        w_agg, w_t5 = topk_agg(wv, text, args.k)
        oc, _ = pool.cand_scores(prompt, o_agg[None])
        wc, _ = pool.cand_scores(prompt, w_agg[None])
        canon_pool, _, _ = pool.prompt_scores(prompt)
        i_o = pool.index[(o_rec['oracle_lvl'], o_rec['oracle_sp_id'])]
        i_w = pool.index[(w_rec['oracle_lvl'], w_rec['oracle_sp_id'])]
        scores = canon_pool.copy()
        scores[i_o] = oc[0]
        scores[i_w] = wc[0]
        fair_rank = int((scores > scores[i_o]).sum()) + 1
        rows.append({'scene': sc, 'prompt': prompt,
                     'oracle_wins_duel': bool(oc[0] > wc[0]),
                     'o_top5_cos': round(o_t5, 4), 'w_top5_cos': round(w_t5, 4),
                     'o_canon': float(oc[0]), 'w_canon': float(wc[0]),
                     'fair_rank_oracle': fair_rank, 'note': ''})
        print(f"  {sc:13s} {prompt:24s} {'ORACLE' if oc[0] > wc[0] else 'wrong ':6s} "
              f"O={o_t5:.3f} W={w_t5:.3f} fair_rank={fair_rank}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    d = df[df['note'] == '']
    print(f"\nORACLE wins duel: {int(d['oracle_wins_duel'].sum())}/{len(d)}")
    print(f"fair_rank <= 3 : {int((d['fair_rank_oracle'] <= 3).sum())}/{len(d)}")
    print(f"Wrote {args.out_csv}")


if __name__ == '__main__':
    main()
