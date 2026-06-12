"""
Phase 1.5b — verify hypothesis (d): zero-norm SPs poison relevancy ranking.

For each scene + each A1 prompt:
  1. Count zero-norm SPs in pool (level [2, 3])
  2. Compute correct SP rank with all SPs vs only non-zero-norm SPs
  3. Show how many of the SPs ranked above correct are zero-norm
"""
import os
import sys
import csv
import torch
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict

from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from utils.vlm_utils import ClipSimMeasure
from arguments import ModelParams, PipelineParams, OptimizationParams


@torch.no_grad()
def main(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f"\n=== Phase 1.5b Zero-norm check: {scene_name} ===\n")

    gaussians = GaussianModel(dataset.sh_degree, 20)
    Scene(dataset, gaussians, 30000, load_sem=False, shuffle=False)
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    vlm = ClipSimMeasure()
    vlm.load_model()

    # Compute zero-norm SP stats per level
    pool_meta = []
    for sp_lvl in [2, 3]:
        feat = snag.feat[sp_lvl - 1]
        n = feat.shape[0]
        norms = feat.norm(dim=1).cpu().numpy()
        zero = int((norms < 0.01).sum())
        print(f"Level {sp_lvl}: {n} SPs, {zero} zero-norm ({100*zero/n:.1f}%)")
        for sp_id in range(n):
            pool_meta.append({
                'lvl': sp_lvl, 'sp_id': sp_id,
                'feat_norm': float(norms[sp_id]),
                'is_zero': float(norms[sp_id]) < 0.01,
            })
    total = len(pool_meta)
    total_zero = sum(1 for p in pool_meta if p['is_zero'])
    print(f"\nPool total (level [2,3]): {total}, zero-norm: {total_zero} ({100*total_zero/total:.1f}%)\n")

    # Load CSV and get A1 prompts for this scene
    rows = list(csv.DictReader(open(args.csv)))
    pp_data = defaultdict(list)
    for r in rows:
        pp_data[(r['scene'], r['prompt'])].append(r)

    a1_prompts = []
    for (sc, p), grp in pp_data.items():
        if sc != scene_name:
            continue
        rank = int(grp[0]['correct_sp_clip_rank'])
        if rank >= 11:
            ref_row = next((x for x in grp if x['is_ref_frame'] == '1'), grp[0])
            a1_prompts.append({
                'prompt': p,
                'rank': rank,
                'correct_sp_lvl': int(ref_row['correct_sp_lvl']),
                'correct_sp_id': int(ref_row['correct_sp_id']),
                'oracle_iou': float(ref_row['oracle_iou']),
            })
    a1_prompts.sort(key=lambda x: -x['oracle_iou'])

    # Per prompt: compute correct SP rank in (all SPs by relevancy) vs (non-zero SPs by relevancy)
    print(f"{'prompt':>22} {'corr lvl':>9} {'corr id':>8} {'orig rank':>10} "
          f"{'rank in non-zero pool':>22} {'zero SPs above':>15} {'real SPs above':>15}")
    for ap in a1_prompts:
        vlm.encode_text(ap['prompt'])
        sim_per_level = [vlm.compute_similarity(f) for f in snag.feat]
        # Build pool with relevancy
        pool = []
        for sp_lvl in [2, 3]:
            n = snag.feat[sp_lvl - 1].shape[0]
            rels = sim_per_level[sp_lvl - 1].cpu().numpy()
            norms = snag.feat[sp_lvl - 1].norm(dim=1).cpu().numpy()
            for sp_id in range(n):
                pool.append({
                    'lvl': sp_lvl, 'sp_id': sp_id,
                    'relevancy': float(rels[sp_id]),
                    'is_zero': float(norms[sp_id]) < 0.01,
                    'is_correct': (sp_lvl == ap['correct_sp_lvl']
                                   and sp_id == ap['correct_sp_id']),
                })
        # Sort by relevancy descending
        pool_by_rel = sorted(pool, key=lambda x: -x['relevancy'])
        orig_rank = next(i for i, p in enumerate(pool_by_rel) if p['is_correct']) + 1

        # Filter to non-zero
        non_zero_pool = [p for p in pool if not p['is_zero']]
        nz_by_rel = sorted(non_zero_pool, key=lambda x: -x['relevancy'])
        nz_rank = next(i for i, p in enumerate(nz_by_rel) if p['is_correct']) + 1

        # Count zero/real SPs above correct
        above_correct = pool_by_rel[:orig_rank - 1]
        zero_above = sum(1 for p in above_correct if p['is_zero'])
        real_above = sum(1 for p in above_correct if not p['is_zero'])

        print(f"{ap['prompt']:>22} {ap['correct_sp_lvl']:>9} {ap['correct_sp_id']:>8} "
              f"{orig_rank:>10} {nz_rank:>22} {zero_above:>15} {real_above:>15}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--csv", type=str,
                        default="output/diagnostics/lerf_ovs_per_prompt.csv")
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)
    main(lp.extract(args), pp.extract(args), args)
