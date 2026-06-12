"""
Stage 3.4-C — NAG kinship: relation between two (lvl, sp) nodes via gaussian
membership sets (snag.labels are per-gaussian assignments per level).

relation(a, b):
  containment_a_in_b = |G_a ∩ G_b| / |G_a|
  - 'same'      : a == b (same gaussian set)
  - 'child'     : a ⊂ b (containment_a_in_b >= 0.8)   [a is part of b]
  - 'parent'    : b ⊂ a (containment_b_in_a >= 0.8)
  - 'overlap'   : 0.2 <= max containment < 0.8
  - 'sibling'   : disjoint-ish but share the same parent SP at the next level up
  - 'unrelated' : otherwise

Applied to: the 5 "part-of-truth" impostor pairs (+ all wrong-top1 pairs).
Output: output/diagnostics/stage3_4_kinship.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))


def load_labels(scene):
    nag = torch.load(f"output/lerf/{scene}/sai_nag.pt")
    return [l.long().cpu().numpy() for l in nag['nag']]


def gset(labels, lvl, sp):
    return np.where(labels[lvl] == sp)[0]


def relation(labels, a, b):
    """a, b = (lvl, sp). Returns (relation, cont_a_in_b, cont_b_in_a)."""
    ga, gb = gset(labels, *a), gset(labels, *b)
    if len(ga) == 0 or len(gb) == 0:
        return 'empty', 0.0, 0.0
    inter = len(np.intersect1d(ga, gb, assume_unique=True))
    ca = inter / len(ga)   # how much of a is inside b
    cb = inter / len(gb)
    if ca >= 0.99 and cb >= 0.99:
        return 'same', ca, cb
    if ca >= 0.8:
        return 'child_of_b', ca, cb     # a is a part of b
    if cb >= 0.8:
        return 'parent_of_b', ca, cb    # b is a part of a
    if max(ca, cb) >= 0.2:
        return 'overlap', ca, cb
    # sibling: same parent at the next coarser level
    max_lvl = len(labels) - 1
    la, lb = a[0], b[0]
    if la < max_lvl and lb < max_lvl:
        pa = np.unique(labels[la + 1][ga])
        pb = np.unique(labels[lb + 1][gb])
        if len(np.intersect1d(pa, pb)) > 0:
            return 'sibling', ca, cb
    return 'unrelated', ca, cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b7_csv', default='output/diagnostics/b7_a4_combined.csv')
    ap.add_argument('--persistent_csv', default='output/diagnostics/persistent_phantoms_17.csv')
    ap.add_argument('--out_csv', default='output/diagnostics/stage3_4_kinship.csv')
    args = ap.parse_args()

    b7 = pd.read_csv(args.b7_csv)
    ref = b7[b7['is_ref_frame'] == 1]
    ph = pd.read_csv(args.persistent_csv)
    pset = set(map(tuple, ph[['scene', 'prompt']].values))

    labels_cache = {}
    rows = []
    for _, r in ref.iterrows():
        if (r['scene'], r['prompt']) not in pset:
            continue
        sc = r['scene']
        if sc not in labels_cache:
            labels_cache[sc] = load_labels(sc)
        labels = labels_cache[sc]
        oracle = (int(r['oracle_lvl']), int(r['oracle_sp_id']))
        wrong = (int(r['clip_top1_lvl']), int(r['clip_top1_sp_id']))
        rel, ca, cb = relation(labels, wrong, oracle)
        rows.append({'scene': sc, 'prompt': r['prompt'],
                     'oracle': f"{oracle[0]}.{oracle[1]}",
                     'wrong': f"{wrong[0]}.{wrong[1]}",
                     'relation_wrong_vs_oracle': rel,
                     'cont_wrong_in_oracle': round(ca, 3),
                     'cont_oracle_in_wrong': round(cb, 3)})
        print(f"  {sc:13s} {r['prompt']:24s} wrong={wrong} vs oracle={oracle}: "
              f"{rel} (w_in_o={ca:.2f}, o_in_w={cb:.2f})")
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nrelation distribution:\n{df['relation_wrong_vs_oracle'].value_counts().to_string()}")
    print(f"Wrote {args.out_csv}")


if __name__ == '__main__':
    main()
