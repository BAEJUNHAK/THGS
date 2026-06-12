"""
Stage 3.1 — B8 causal replay (Phase B).

Replays the THGS within-view mixing stage (merge_proj.py proj_gaussian_features_x,
the production path: configs/lerf.yml feat_assign=2) for the 17 persistent phantom
oracle SPs + 4 easy-control SPs per scene, over ALL train views.

Per (target SP, view) records:
  - mix stats on the RAW ratio row: mix_count = (ratio >= 0.3).sum(),
    top1/top2 ratio values, n_nonzero masks
  - mixed_feat = normalize(sp_mask_mat[sp] @ view_level_feature)   (post-B8 signal)
  - hard_feat  = normalize(view_level_feature[argmax(ratio row)])  (fix-A preview)
  - visibility portion (weight > 0.0001 gaussians / total)

Fidelity gate (MUST pass before Phase C):
  Reconstruct the pipeline-final SP feature by accumulating
  normalize(mixed_raw) * portion over ALL views, normalize, and compare with
  sai_nag.pt's snag.feat[lvl-1][sp_id]. Report cos per target; expect >= 0.95.

Faithful-replica notes (vs merge_proj.py):
  - extract_gaussian_features / get_superpoint_mask_ratio copied verbatim
    (random enc trick is label-preserving; argmax recovers seg ids).
  - WEIGHT_THRESHOLD = 0.0001, RATIO_THRESHOLD = 0.3 (proj_gaussian_features_x).
  - gau2sp = snag.labels[lvl] (0-based), feature_level = lvl,
    pt_sp_label = extract_gaussian_features(..., feature_level=lvl).
  - Aggregation order-invariant -> shuffle=False is safe and matches the
    h2lite view ordering (train_cams[::step][:30]) for the clean-cos join.

Output: pickle keyed by (scene, prompt) ->
  {category, oracle_lvl, oracle_sp_id, fidelity_cos, n_views_total,
   views: [{view_idx, image_name, portion, visible,
            mix_count, n_nonzero, top1_ratio, top2_ratio, argmax_col,
            mixed_feat(fp16, post-threshold), hard_feat(fp16)}]}
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from argparse import ArgumentParser

from gaussian_renderer import render_point, trace
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from arguments import ModelParams, PipelineParams, OptimizationParams

WEIGHT_THRESHOLD = 0.0001   # merge_proj.py:123 (proj_gaussian_features_x)
RATIO_THRESHOLD = 0.3       # merge_proj.py:133


def build_target_list(b7_csv, persistent_csv, easy_per_scene=4):
    """Verbatim from stage2a_layer1_b8_coverage.py (same RandomState -> same controls)."""
    b7 = pd.read_csv(b7_csv)
    ref = b7[b7['is_ref_frame'] == 1].copy()
    phantoms = pd.read_csv(persistent_csv)
    phantom_set = set((r['scene'], r['prompt']) for _, r in phantoms.iterrows())
    targets = []
    for _, r in ref.iterrows():
        if (r['scene'], r['prompt']) in phantom_set:
            targets.append({
                'scene': r['scene'], 'prompt': r['prompt'], 'category': 'phantom17',
                'oracle_lvl': int(r['oracle_lvl']), 'oracle_sp_id': int(r['oracle_sp_id']),
            })
    rng = np.random.RandomState(42)
    for sc, g in ref.groupby('scene'):
        easy = g[g['oracle_rank'] <= 3]
        n = min(easy_per_scene, len(easy))
        if n == 0:
            continue
        idx = rng.choice(len(easy), size=n, replace=False)
        for i in idx:
            r = easy.iloc[i]
            targets.append({
                'scene': r['scene'], 'prompt': r['prompt'], 'category': 'easy_sample',
                'oracle_lvl': int(r['oracle_lvl']), 'oracle_sp_id': int(r['oracle_sp_id']),
            })
    return targets


@torch.no_grad()
def extract_gaussian_features(gaussians, views, feature_level, pipeline, background):
    """Verbatim replica of merge_proj.py:147-173 (per-gaussian per-view SAM mask label)."""
    labels = []
    for i, view in enumerate(views):
        seg_map = view.semantic["seg_map"][feature_level].clone()
        img_mask = view.semantic['fg_mask'][feature_level]
        seg_min, seg_max = seg_map[img_mask].min().item(), seg_map.max().item()
        seg_num = seg_max - seg_min + 2
        seg_map -= seg_min - 1
        seg_map[~img_mask] = 0
        enc = torch.normal(mean=0, std=1, size=(seg_num, 20), device="cuda")
        enc = torch.nn.functional.normalize(enc, p=2, dim=-1)

        feature_map = enc[seg_map]
        render_pkg = trace(view, gaussians, feature_map, None, pipeline, background)
        gau_sem = render_pkg["gaussian_semantics"]
        gau_sem = torch.nn.functional.normalize(gau_sem, p=2, dim=-1)
        sim = torch.matmul(gau_sem, enc.T)
        gau_label = sim.argmax(dim=-1)
        sim_filter = sim.max(dim=-1)[0] > 0.85
        gau_label[~sim_filter] = 0
        labels.append(gau_label.to(torch.int32).cpu())
        del feature_map, render_pkg, gau_sem, sim
    torch.cuda.empty_cache()
    return torch.stack(labels, dim=1)  # (N, M) int32 cpu


def get_superpoint_mask_ratio(point_to_super, point_to_mask, num_masks):
    """Verbatim replica of merge_proj.py:100-119 (softmax=False path)."""
    num_superpoints = point_to_super.max().item() + 1
    indices = point_to_super * num_masks + point_to_mask
    super_mask_count_flat = torch.bincount(indices, minlength=num_superpoints * num_masks)
    super_mask_count = super_mask_count_flat.view(num_superpoints, num_masks)
    if point_to_mask.min() == 0:
        super_mask_count = super_mask_count[:, 1:]
    else:
        print("Warning: mask min is not 0, check the data")
    super_mask_sum = super_mask_count.sum(dim=1, keepdim=True)
    return super_mask_count.float() / (super_mask_sum + 1e-8)


def aggregate_views(scaled_feats, agg):
    """Final SP feature from per-view (normalized_feat * portion) vectors.

    agg='thgs'  : sum -> normalize                       (merge_proj.py:140-143)
    agg='relags': ROFA tau=2 keep-mask -> mean -> norm   (ReLaGS/merge_proj.py:96-130)
    scaled_feats: torch (N, 512) on cuda, zero rows already excluded.
    """
    if scaled_feats.shape[0] == 0:
        return torch.zeros(512, device='cuda')
    if agg == 'thgs':
        return F.normalize(scaled_feats.sum(dim=0), p=2, dim=-1)
    if scaled_feats.shape[0] == 1:
        return F.normalize(scaled_feats[0], p=2, dim=-1)
    cos = F.cosine_similarity(scaled_feats.unsqueeze(1),
                              scaled_feats.unsqueeze(0), dim=-1)
    mean_sim = (cos.sum(dim=1) - 1) / (cos.shape[0] - 1)
    mu, sigma = mean_sim.mean(), mean_sim.std()
    keep = mean_sim > (mu - 2.0 * sigma)
    if keep.sum() == 0:
        keep[mean_sim.argmax()] = True
    return F.normalize(scaled_feats[keep].mean(dim=0), p=2, dim=-1)


def load_targets_csv(path):
    """Arbitrary targets: CSV with scene,prompt,category,oracle_lvl,oracle_sp_id."""
    df = pd.read_csv(path)
    return [{'scene': r['scene'], 'prompt': r['prompt'],
             'category': r.get('category', 'custom'),
             'oracle_lvl': int(r['oracle_lvl']), 'oracle_sp_id': int(r['oracle_sp_id'])}
            for _, r in df.iterrows()]


@torch.no_grad()
def run(dataset, pipe_args, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    if args.targets_csv:
        all_targets = load_targets_csv(args.targets_csv)
    else:
        all_targets = build_target_list(args.b7_csv, args.persistent_csv,
                                        args.easy_per_scene)
    targets = [t for t in all_targets if t['scene'] == scene_name]
    if not targets:
        print(f"[skip] no targets for {scene_name}")
        return
    print(f"\n=== {scene_name}: {len(targets)} targets "
          f"({sum(t['category']=='phantom17' for t in targets)} phantom, "
          f"{sum(t['category']=='easy_sample' for t in targets)} easy) ===", flush=True)

    # merge_proj-style gaussians (sem dim 0) + semantic data loaded
    gaussians = GaussianModel(3, 0)
    iter_arg = args.iteration if args.iteration > 0 else -1
    scene = Scene(dataset, gaussians, load_iteration=iter_arg, load_sem=True, shuffle=False)
    if args.iteration == 0:
        ply_path = os.path.join(dataset.model_path, "point_cloud",
                                "iteration_0", "point_cloud.ply")
        if os.path.exists(ply_path):
            gaussians.load_ply(ply_path)
    pipeline = PipelineParams(ArgumentParser())
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    views = scene.getTrainCameras()
    M = len(views)
    print(f"  Gaussians={snag.gaussian_num}, views={M}", flush=True)

    levels = sorted(set(t['oracle_lvl'] for t in targets))
    if args.dump_all_sp:
        levels = sorted(set(levels) | {2, 3})
    results = {}
    allsp_dump = {'scene': scene_name, 'levels': {}}

    for lvl in levels:
        lvl_targets = [t for t in targets if t['oracle_lvl'] == lvl]
        target_sps = sorted(set(t['oracle_sp_id'] for t in lvl_targets))
        print(f"\n  --- level {lvl}: {len(lvl_targets)} targets, "
              f"SPs {target_sps} ---", flush=True)

        t0 = time.time()
        pt_sp_label = extract_gaussian_features(
            gaussians, views, lvl, pipeline, background)  # (N, M) cpu int32
        print(f"  extract_gaussian_features lvl={lvl}: {time.time()-t0:.1f}s", flush=True)

        gau2sp = snag.labels[lvl].long().cuda()
        sp_uni, sp_gnum = torch.unique(gau2sp, return_counts=True, sorted=True)
        sp_total = {int(s.item()): int(c.item()) for s, c in zip(sp_uni, sp_gnum)}

        # per-target accumulators for fidelity reconstruction
        agg = {sp: torch.zeros(512, device="cuda") for sp in target_sps}
        per_view_records = {sp: [] for sp in target_sps}

        num_sp = int(gau2sp.max().item()) + 1
        sp_total_vec = torch.zeros(num_sp, device="cuda")
        sp_total_vec[sp_uni] = sp_gnum.float()
        dump_feats, dump_portions, dump_names = [], [], []

        for vi, view in enumerate(views):
            render_pkg = render_point(view, gaussians, pipeline, background)
            weight = render_pkg["weight"]
            gau_mask = weight > WEIGHT_THRESHOLD

            gt_feature = view.semantic["sem"].cuda().float()
            seg_map = view.semantic["seg_map"][lvl]
            img_mask = view.semantic['fg_mask'][lvl]
            seg_min = seg_map[img_mask].min().item()
            seg_max = seg_map[img_mask].max().item()
            view_level_feature = gt_feature[seg_min:seg_max + 1]
            seg_num = seg_max - seg_min + 2

            ratio = get_superpoint_mask_ratio(
                gau2sp, pt_sp_label[:, vi].long().cuda(), seg_num)

            sp_mask_mat = ratio.clone()
            sp_mask_mat[sp_mask_mat < RATIO_THRESHOLD] = 0
            sp_feat = sp_mask_mat @ view_level_feature  # (num_sp, 512)

            sp_index = gau2sp[gau_mask]
            unique_sp, sp_gau_count = torch.unique(sp_index, return_counts=True)
            vis_count = {int(s.item()): int(c.item())
                         for s, c in zip(unique_sp, sp_gau_count)}

            if args.dump_all_sp:
                portion_vec = torch.zeros(num_sp, device="cuda")
                portion_vec[unique_sp] = sp_gau_count.float()
                portion_vec = portion_vec / sp_total_vec.clamp_min(1)
                dump_feats.append(F.normalize(sp_feat, p=2, dim=-1)
                                  .half().cpu().numpy())
                dump_portions.append(portion_vec.half().cpu().numpy())
                dump_names.append(view.image_name)

            for sp in target_sps:
                row = ratio[sp]
                n_nonzero = int((row > 0).sum().item())
                mix_count = int((row >= RATIO_THRESHOLD).sum().item())
                if n_nonzero > 0:
                    top_vals, top_idx = torch.topk(row, min(2, row.shape[0]))
                    top1_ratio = float(top_vals[0].item())
                    top2_ratio = float(top_vals[1].item()) if top_vals.shape[0] > 1 else 0.0
                    argmax_col = int(top_idx[0].item())
                    hard_feat = F.normalize(view_level_feature[argmax_col], p=2, dim=-1)
                else:
                    top1_ratio = top2_ratio = 0.0
                    argmax_col = -1
                    hard_feat = None

                mixed_raw = sp_feat[sp]
                mixed_norm = F.normalize(mixed_raw, p=2, dim=-1)
                visible = sp in vis_count
                portion = vis_count.get(sp, 0) / sp_total[sp]

                # pipeline accumulation (merge_proj.py:140-142): only visible SPs
                if visible:
                    agg[sp] += mixed_norm * portion

                # record only when SP contributes or is at least labeled in view
                if visible or n_nonzero > 0:
                    per_view_records[sp].append({
                        'view_idx': vi,
                        'image_name': view.image_name,
                        'visible': bool(visible),
                        'portion': float(portion),
                        'mix_count': mix_count,
                        'n_nonzero': n_nonzero,
                        'top1_ratio': top1_ratio,
                        'top2_ratio': top2_ratio,
                        'argmax_col': argmax_col,
                        'mixed_feat': (mixed_norm.cpu().numpy().astype(np.float16)
                                       if float(mixed_raw.abs().sum().item()) > 0 else None),
                        'hard_feat': (hard_feat.cpu().numpy().astype(np.float16)
                                      if hard_feat is not None else None),
                    })
            del render_pkg, weight, gau_mask, gt_feature, view_level_feature
            del ratio, sp_mask_mat, sp_feat
            if vi % 50 == 0:
                torch.cuda.empty_cache()
                print(f"    view {vi}/{M}", flush=True)

        # fidelity per target at this level (reconstruction honors --agg)
        for t in lvl_targets:
            sp = t['oracle_sp_id']
            contrib = [v for v in per_view_records[sp]
                       if v['visible'] and v['mixed_feat'] is not None]
            if contrib:
                scaled = torch.stack([
                    torch.from_numpy(v['mixed_feat'].astype(np.float32)).cuda()
                    * v['portion'] for v in contrib])
                recon = aggregate_views(scaled, args.agg)
            else:
                recon = F.normalize(agg[sp], p=2, dim=-1)
            actual = snag.feat[lvl - 1][sp].cuda().float()
            actual = F.normalize(actual, p=2, dim=-1)
            fid = float((recon @ actual).item())
            results[(scene_name, t['prompt'])] = {
                'category': t['category'],
                'oracle_lvl': lvl,
                'oracle_sp_id': sp,
                'fidelity_cos': fid,
                'n_views_total': M,
                'recon_feat': recon.cpu().numpy().astype(np.float16),
                'actual_feat': actual.cpu().numpy().astype(np.float16),
                'views': per_view_records[sp],
            }
            print(f"  [fidelity] {t['category']:12s} {t['prompt']:25s} "
                  f"lvl={lvl} sp={sp}  cos(recon, sai_nag)={fid:.4f} "
                  f"{'PASS' if fid >= 0.95 else '** FAIL **'}", flush=True)
        if args.dump_all_sp:
            feats_arr = np.stack(dump_feats, axis=0)        # (V, S, 512) fp16
            por_arr = np.stack(dump_portions, axis=0)       # (V, S) fp16
            # fidelity over ALL SPs: reconstruction honors --agg
            if args.agg == 'thgs':
                recon = (feats_arr.astype(np.float32) *
                         por_arr.astype(np.float32)[:, :, None]).sum(axis=0)
                rn = np.linalg.norm(recon, axis=1, keepdims=True)
                recon = recon / np.clip(rn, 1e-9, None)
            else:  # relags: per-SP ROFA reconstruction
                S = feats_arr.shape[1]
                recon = np.zeros((S, 512), dtype=np.float32)
                ft = torch.from_numpy(feats_arr.astype(np.float32)).cuda()
                pt = torch.from_numpy(por_arr.astype(np.float32)).cuda()
                nz = (ft.abs().sum(-1) > 1e-6) & (pt > 0)   # (V, S)
                for s_ in range(S):
                    m = nz[:, s_]
                    if not m.any():
                        continue
                    scaled = ft[m, s_, :] * pt[m, s_].unsqueeze(-1)
                    recon[s_] = aggregate_views(scaled, 'relags').cpu().numpy()
                del ft, pt, nz
            actual = snag.feat[lvl - 1].cpu().numpy().astype(np.float32)
            an = np.linalg.norm(actual, axis=1, keepdims=True)
            actual_n = actual / np.clip(an, 1e-9, None)
            cos_all = (recon * actual_n).sum(axis=1)
            nonzero = (an.squeeze() > 1e-6)
            pass_frac = float((cos_all[nonzero] >= 0.95).mean())
            print(f"  [dump lvl={lvl}] all-SP fidelity: "
                  f"{(cos_all[nonzero] >= 0.95).sum()}/{int(nonzero.sum())} "
                  f"({pass_frac*100:.1f}%) pass; zero-norm SPs in sai_nag: "
                  f"{int((~nonzero).sum())}", flush=True)
            allsp_dump['levels'][lvl] = {
                'image_names': dump_names,
                'feats': feats_arr,
                'portions': por_arr,
                'fidelity_cos': cos_all.astype(np.float32),
                'sai_nag_zero_norm': (~nonzero),
            }
        del pt_sp_label, agg
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    if os.path.exists(args.out_pkl) and args.append:
        with open(args.out_pkl, "rb") as f:
            existing = pickle.load(f)
        existing.update(results)
        results = existing
    with open(args.out_pkl, "wb") as f:
        pickle.dump(results, f)

    fids = [(k[1], v['fidelity_cos']) for k, v in results.items() if k[0] == scene_name]
    n_pass = sum(1 for _, f_ in fids if f_ >= 0.95)
    print(f"\n[{scene_name}] fidelity: {n_pass}/{len(fids)} pass (cos >= 0.95)")
    print(f"Wrote {len(results)} total targets to {args.out_pkl}")

    if args.dump_all_sp:
        with open(args.dump_all_sp, "wb") as f:
            pickle.dump(allsp_dump, f)
        sz = os.path.getsize(args.dump_all_sp) / 1e6
        print(f"Wrote all-SP dump to {args.dump_all_sp} ({sz:.0f} MB)")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--b7_csv", default="output/diagnostics/b7_a4_combined.csv")
    parser.add_argument("--persistent_csv", default="output/diagnostics/persistent_phantoms_17.csv")
    parser.add_argument("--out_pkl", default="output/diagnostics/stage3_b8_replay_perview.pkl")
    parser.add_argument("--easy_per_scene", type=int, default=4)
    parser.add_argument("--targets_csv", type=str, default="",
                        help="optional CSV (scene,prompt,category,oracle_lvl,oracle_sp_id) "
                             "overriding the default phantom+easy target list")
    parser.add_argument("--dump_all_sp", type=str, default="",
                        help="if set, dump ALL SPs' per-view normalized features + "
                             "portions for levels [2,3] to this pkl path "
                             "(Stage 3.3 full-pool analysis)")
    parser.add_argument("--agg", choices=["thgs", "relags"], default="thgs",
                        help="aggregation used for the FIDELITY reconstruction: "
                             "thgs = visibility-weighted sum (merge_proj feat_assign=2), "
                             "relags = ROFA tau=2 then mean (ReLaGS merge_proj)")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="0 = load point_cloud/iteration_0 (ReLaGS HF release)")
    parser.add_argument("--append", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("Done.")
