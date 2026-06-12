"""
Phase 1.5a — per-SP feature inspection (uses only sai_nag.pt, no language_features needed).

For each A1 prompt: inspect the per-SP CLIP feature of:
  - the CORRECT SP (greedy first pick)
  - the CLIP top-1 SP (wrong SP CLIP chose)
  - distribution across the SP pool at level [2, 3]

Compute:
  - cos_sim of each SP feature with prompt text (raw, before canon-contrast)
  - canon-contrast relevancy (= scoring used by test_lerf.py)
  - feature norm (sanity: should be ~1 after L2-norm)
  - top-k SPs by both metrics

Verdict diagnostic:
  - If correct SP feature is L2-normalized but cos_sim is near 0 → feature direction is wrong
    → either (a) per-mask features were never aligned with prompt OR (b) aggregation killed signal
  - If many SPs have high cos_sim (> 0.2) → CLIP picks among them, correct SP just unlucky
  - If correct SP cos_sim is reasonable (>0.2) but relevancy is low → canon-contrast is hostile
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
def inspect_prompt(snag, vlm, prompt, correct_sp_lvl, correct_sp_id,
                   clip_top1_lvl, clip_top1_id, top_k=15):
    vlm.encode_text(prompt)
    text_pos = vlm.text_feature[0]
    text_pos_n = text_pos / (text_pos.norm() + 1e-8)

    # Compute scores at each level
    sim_per_level = [vlm.compute_similarity(f) for f in snag.feat]  # canon-contrast relevancy

    # Build unified pool (level 2 + level 3)
    pool = []  # list of dicts with metadata
    for sp_lvl in [2, 3]:
        feat = snag.feat[sp_lvl - 1]  # per-SP feature tensor
        n_sps = feat.shape[0]
        feat_n = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)
        cos_sims = (feat_n @ text_pos_n).cpu().numpy()
        rels = sim_per_level[sp_lvl - 1].cpu().numpy()
        norms = feat.norm(dim=1).cpu().numpy()
        for sp_id in range(n_sps):
            pool.append({
                'lvl': sp_lvl, 'sp_id': sp_id,
                'cos_sim': float(cos_sims[sp_id]),
                'relevancy': float(rels[sp_id]),
                'feat_norm': float(norms[sp_id]),
                'is_correct': (sp_lvl == correct_sp_lvl and sp_id == correct_sp_id),
                'is_clip_top1': (sp_lvl == clip_top1_lvl and sp_id == clip_top1_id),
            })

    print(f"\n=== Inspection for prompt = '{prompt}' ===")
    print(f"Pool size = {len(pool)}, levels {[snag.feat[i].shape[0] for i in range(len(snag.feat))]}\n")

    # Sort by relevancy (= what test_lerf.py uses)
    pool_by_rel = sorted(pool, key=lambda x: -x['relevancy'])
    pool_by_cos = sorted(pool, key=lambda x: -x['cos_sim'])

    # Find correct SP rank in both
    correct_rank_rel = next(i for i, p in enumerate(pool_by_rel) if p['is_correct']) + 1
    correct_rank_cos = next(i for i, p in enumerate(pool_by_cos) if p['is_correct']) + 1

    correct_entry = next(p for p in pool if p['is_correct'])
    top1_entry = next(p for p in pool if p['is_clip_top1'])

    print(f"CORRECT SP (lvl={correct_sp_lvl}, id={correct_sp_id}):")
    print(f"  raw cos_sim with prompt = {correct_entry['cos_sim']:.4f}")
    print(f"  canon-contrast relevancy = {correct_entry['relevancy']:.4f}")
    print(f"  feature norm = {correct_entry['feat_norm']:.4f}")
    print(f"  rank in cos_sim ordering = {correct_rank_cos}/{len(pool)}")
    print(f"  rank in relevancy (test_lerf.py) ordering = {correct_rank_rel}/{len(pool)}")

    print(f"\nCLIP TOP-1 SP (lvl={clip_top1_lvl}, id={clip_top1_id}):")
    print(f"  raw cos_sim with prompt = {top1_entry['cos_sim']:.4f}")
    print(f"  canon-contrast relevancy = {top1_entry['relevancy']:.4f}")
    print(f"  feature norm = {top1_entry['feat_norm']:.4f}")

    print(f"\nTop-{top_k} SPs by relevancy (= test_lerf.py picks from these):")
    print(f"  {'rank':>4} {'lvl':>3} {'sp_id':>6} {'cos_sim':>9} {'relevancy':>10} {'note':>20}")
    for i, p in enumerate(pool_by_rel[:top_k]):
        note = ''
        if p['is_correct']: note = '<-- CORRECT'
        elif p['is_clip_top1']: note = '<-- CLIP TOP-1'
        print(f"  {i+1:>4} {p['lvl']:>3} {p['sp_id']:>6} {p['cos_sim']:>9.4f} "
              f"{p['relevancy']:>10.4f} {note:>20}")

    print(f"\nTop-{top_k} SPs by raw cos_sim (alternative ranking):")
    print(f"  {'rank':>4} {'lvl':>3} {'sp_id':>6} {'cos_sim':>9} {'relevancy':>10} {'note':>20}")
    for i, p in enumerate(pool_by_cos[:top_k]):
        note = ''
        if p['is_correct']: note = '<-- CORRECT'
        elif p['is_clip_top1']: note = '<-- CLIP TOP-1'
        print(f"  {i+1:>4} {p['lvl']:>3} {p['sp_id']:>6} {p['cos_sim']:>9.4f} "
              f"{p['relevancy']:>10.4f} {note:>20}")

    # Distribution stats
    cos_arr = np.array([p['cos_sim'] for p in pool])
    rel_arr = np.array([p['relevancy'] for p in pool])
    norm_arr = np.array([p['feat_norm'] for p in pool])

    print(f"\nDistribution stats:")
    print(f"  cos_sim:   mean={cos_arr.mean():.4f}, std={cos_arr.std():.4f}, "
          f"min={cos_arr.min():.4f}, max={cos_arr.max():.4f}, p95={np.percentile(cos_arr, 95):.4f}")
    print(f"  relevancy: mean={rel_arr.mean():.4f}, std={rel_arr.std():.4f}, "
          f"min={rel_arr.min():.4f}, max={rel_arr.max():.4f}, p95={np.percentile(rel_arr, 95):.4f}")
    print(f"  feat_norm: mean={norm_arr.mean():.4f}, std={norm_arr.std():.4f}, "
          f"min={norm_arr.min():.4f}, max={norm_arr.max():.4f}")

    # Verdict
    print(f"\n--- VERDICT ---")
    correct_cos = correct_entry['cos_sim']
    correct_rel = correct_entry['relevancy']
    top1_cos = top1_entry['cos_sim']
    top1_rel = top1_entry['relevancy']

    print(f"  Cos_sim: correct SP = {correct_cos:.4f}, CLIP top-1 = {top1_cos:.4f} "
          f"(diff = {top1_cos - correct_cos:+.4f})")
    print(f"  Relevancy: correct SP = {correct_rel:.4f}, CLIP top-1 = {top1_rel:.4f} "
          f"(diff = {top1_rel - correct_rel:+.4f})")

    if correct_cos < 0.15 and top1_cos < 0.15:
        print(f"  → Both feature scores LOW: CLIP feature 자체가 prompt와 안 align (전체 pool에서 prompt-match가 약함)")
    elif correct_cos < 0.15:
        print(f"  → Correct SP feature only LOW: aggregation이 정답 SP feature를 망친 신호 (b 의심)")
    elif top1_cos > correct_cos + 0.05:
        print(f"  → CLIP top-1 SP가 더 높은 cos_sim → 단순 ranking 실수 아님, 다른 SP가 진짜로 prompt와 더 닮음")
    return correct_entry, top1_entry, pool


@torch.no_grad()
def main(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f"\n=== Phase 1.5a SP Feature Inspection: {scene_name} ===\n")

    gaussians = GaussianModel(dataset.sh_degree, 20)
    Scene(dataset, gaussians, 30000, load_sem=False, shuffle=False)
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    vlm = ClipSimMeasure()
    vlm.load_model()
    print(f"NAG level sizes: {[snag.feat[i].shape[0] for i in range(len(snag.feat))]}")

    # Load CSV and inspect each A1 prompt
    rows = list(csv.DictReader(open(args.csv)))
    # Get unique (scene, prompt) with A1 sub-type
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
                'clip_top1_lvl': int(ref_row['clip_top1_lvl']),
                'clip_top1_sp_id': int(ref_row['clip_top1_sp_id']),
                'oracle_iou': float(ref_row['oracle_iou']),
            })
    a1_prompts.sort(key=lambda x: -x['oracle_iou'])
    print(f"A1 prompts in {scene_name}: {len(a1_prompts)}")
    for ap in a1_prompts:
        print(f"  {ap['prompt']}: rank={ap['rank']}, oracle={ap['oracle_iou']:.3f}")

    for ap in a1_prompts:
        inspect_prompt(snag, vlm, ap['prompt'],
                       ap['correct_sp_lvl'], ap['correct_sp_id'],
                       ap['clip_top1_lvl'], ap['clip_top1_sp_id'])
        print("\n" + "=" * 80)


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
