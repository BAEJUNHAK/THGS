"""
P1-D — E2 hierarchical phantom propagation (pre-registered predictions in
md/hypotheses/strategy/p1_problem_experiments.md §3).

For each phantom oracle SP (THGS thgs_class phantom n=21 / ReLaGS
relags_class phantom n=20) and each easy control:
  chain members at level l in {1,2,3} = SPs whose gaussian overlap with the
  oracle's gaussian set satisfies (|G_m ∩ G_o|/|G_o| >= 0.3  [ancestor/major]
  or |G_m ∩ G_o|/|G_m| >= 0.5 [child mostly inside oracle]).
  best_rank_l = best within-level canon rank among chain members.

Classification per case:
  amplification : some lower level achieves rank<=3 but a higher level loses it
  wash_out      : only achieved at a higher level (lower levels all fail)
  propagation   : no level achieves (failure preserved up the chain)
  all_good      : every level achieves (typical easy)

Pre-registered: propagation >= wash_out expected; amplification >= 20% of
phantoms promotes the "hierarchy backfires" section.

Output: output/diagnostics/p1d_e2_hierarchy.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage3_2_common import make_vlm
from p1a_competitor_autopsy import METHODS, SCENES

CHAIN_LEVELS = [1, 2, 3]
GRANULARITY4 = {'jake', 'rubber duck with hat', 'tesla door handle', 'sink'}


def load_nag(cfg, scene):
    nag = torch.load(cfg['nag_tpl'].format(scene))
    labs = [np.asarray(l.long().cpu().numpy()) for l in nag['nag']]
    n = labs[0].shape[0]
    assert all(l.shape[0] == n for l in labs)
    # map level l (1..3) -> per-gaussian label array
    if len(labs) >= 4:
        lab = {l: labs[l] for l in CHAIN_LEVELS}
    else:
        lab = {l: labs[l - 1] for l in CHAIN_LEVELS}
    feat = {l: F.normalize(nag['nag_feat'][l - 1].cuda().float(), p=2, dim=-1)
            for l in CHAIN_LEVELS}
    return lab, feat


def chain_members(lab, lvl_o, sp_o, l):
    g0 = np.where(lab[lvl_o] == sp_o)[0]
    ids, cnt = np.unique(lab[l][g0], return_counts=True)
    members = []
    for i, c in zip(ids, cnt):
        share_o = c / len(g0)
        size_m = (lab[l] == i).sum()
        share_m = c / max(size_m, 1)
        if share_o >= 0.3 or share_m >= 0.5:
            members.append(int(i))
    return members


@torch.no_grad()
def main():
    vlm = make_vlm()
    cm = pd.read_csv('output/diagnostics/cross_method_d2_decomposition.csv')
    rows = []
    for method in ['thgs', 'relags']:
        cfg = METHODS[method]
        ref = pd.read_csv(cfg['ref_csv'])
        ref = ref[ref['is_ref_frame'] == 1]
        cls_col = 'thgs_class' if method == 'thgs' else 'relags_class'
        pset = set(map(tuple, cm[cm[cls_col] == 'phantom'][['scene', 'prompt']].values))

        for scene in SCENES:
            sc = ref[ref.scene == scene]
            lab, feat = load_nag(cfg, scene)
            scores_cache = {}
            for _, r in sc.iterrows():
                prompt = r['prompt']
                cat = ('phantom' if (scene, prompt) in pset
                       else 'easy' if int(r['oracle_rank']) <= 3 else 'other')
                if cat == 'other':
                    continue
                if prompt not in scores_cache:
                    vlm.encode_text(prompt)
                    scores_cache[prompt] = {
                        l: vlm.compute_similarity(feat[l]).cpu().numpy()
                        for l in CHAIN_LEVELS}
                s = scores_cache[prompt]
                lvl_o, sp_o = int(r['oracle_lvl']), int(r['oracle_sp_id'])
                row = {'method': method, 'scene': scene, 'prompt': prompt,
                       'category': cat, 'oracle_lvl': lvl_o,
                       'granularity4': int(prompt in GRANULARITY4)}
                ach = {}
                for l in CHAIN_LEVELS:
                    mem = chain_members(lab, lvl_o, sp_o, l)
                    if not mem:
                        row[f'best_rank_l{l}'] = -1
                        ach[l] = False
                        continue
                    ranks = [int((s[l] > s[l][m]).sum()) + 1 for m in mem]
                    row[f'best_rank_l{l}'] = int(min(ranks))
                    row[f'n_members_l{l}'] = len(mem)
                    ach[l] = min(ranks) <= 3
                lv_ok = [l for l in CHAIN_LEVELS if ach[l]]
                if not lv_ok:
                    cls = 'propagation'
                elif all(ach[l] for l in CHAIN_LEVELS if row[f'best_rank_l{l}'] != -1):
                    cls = 'all_good'
                elif any((l2 > l1) and ach[l1] and not ach[l2]
                         for l1 in CHAIN_LEVELS for l2 in CHAIN_LEVELS
                         if row[f'best_rank_l{l1}'] != -1 and row[f'best_rank_l{l2}'] != -1):
                    cls = 'amplification'
                else:
                    cls = 'wash_out'
                row['class'] = cls
                rows.append(row)
            del feat
            torch.cuda.empty_cache()
        print(f"[{method}] done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('output/diagnostics/p1d_e2_hierarchy.csv', index=False)
    print("\n" + "=" * 78)
    for method in ['thgs', 'relags']:
        for cat in ['phantom', 'easy']:
            g = df[(df.method == method) & (df.category == cat)]
            dist = g['class'].value_counts().to_dict()
            print(f"[{method}] {cat:7s} n={len(g):2d}  {dist}")
    ph = df[df.category == 'phantom']
    amp = float((ph['class'] == 'amplification').mean())
    prop = float((ph['class'] == 'propagation').mean())
    wash = float((ph['class'] == 'wash_out').mean())
    print(f"\n[E2 verdict] pooled phantom n={len(ph)}: amplification "
          f"{amp*100:.0f}% / propagation {prop*100:.0f}% / wash_out {wash*100:.0f}%")
    promote = 'PROMOTE hierarchy-backfires' if amp >= 0.2 else 'no promotion'
    print(f"  pre-reg: propagation>=wash_out {'OK' if prop >= wash else 'MISS'}; "
          f"amplification>=20% -> {promote}")
    g4 = df[(df.granularity4 == 1) & (df.category == 'phantom')]
    if len(g4):
        print("\n[granularity4 reinterpretation]")
        print(g4[['method', 'scene', 'prompt', 'best_rank_l1', 'best_rank_l2',
                  'best_rank_l3', 'class']].to_string(index=False))


if __name__ == '__main__':
    main()
