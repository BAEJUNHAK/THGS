"""
Stage 4 / Phase B+C — LOSO calibration + mask-IoU final evaluation + R9.

B: per held-out fold, shortlist 3 configs by a rank proxy computed ONLY on the
   3 calibration scenes (phantom/other recovered minus 2x easy regressions),
   then pick the winner among the shortlist by CALIBRATION-scene mask mIoU.
   Nothing from the held-out scene is used for selection.
C: evaluate each fold's chosen config on its held-out scene (mask top-3(+union)
   render), pool the 4 held-out scenes -> headline full-67 mIoU vs the
   stage3_3 baseline (iou_baseline, same machinery). Plus group breakdown,
   multi-instance separate track, and per-signal ablation (E/G/P off variants,
   which exist inside the grid).

Outputs:
  stage4_loso_choice.csv   per-fold shortlist + chosen config + reasons
  stage4_mask_iou.csv      long format (scene,prompt,frame,config,iou)
  stage4_ablation.csv      held-out ablation per fold
  + printed R9 verdict
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from collections import defaultdict
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(__file__))
from gaussian_renderer import render
from scene import Scene, GaussianModel
from nag_data import SemanticNAG
from arguments import ModelParams, PipelineParams, OptimizationParams

SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']
D = 'output/diagnostics'


def polygon_to_mask(img_shape, pts):
    m = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(m, [np.asarray(pts, dtype=np.int32)], 1)
    return m


class SceneEval:
    def __init__(self, scene_name):
        parser = ArgumentParser()
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        a = parser.parse_args(['-s', f'data/lerf_ovs/{scene_name}',
                               '-m', f'output/lerf/{scene_name}'])
        ds = lp.extract(a)
        self.pipe = pp.extract(a)
        self.g = GaussianModel(ds.sh_degree, 20)
        sc = Scene(ds, self.g, 30000, load_sem=False, shuffle=False)
        self.bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        nag = torch.load(f'output/lerf/{scene_name}/sai_nag.pt')
        self.snag = SemanticNAG(nag['nag'], nag['nag_feat'])
        cams = {c.image_name: c for c in sc.getTrainCameras()}
        label_dir = f'data/lerf_ovs/label/{scene_name}'
        self.frames = {}
        for im in sorted(os.listdir(label_dir)):
            if not im.endswith('.jpg'):
                continue
            n = im.split('.')[0]
            if n not in cams:
                continue
            anno = json.load(open(os.path.join(label_dir, n + '.json')))
            self.frames[n] = (cams[n], anno['objects'])

    def gt(self, prompt, frame):
        cam, objs = self.frames[frame]
        m = np.zeros((cam.image_height, cam.image_width), np.uint8)
        for o in objs:
            if o['category'] == prompt:
                m = np.maximum(m, polygon_to_mask(m.shape, o['segmentation']))
        return m > 0

    @torch.no_grad()
    def union_iou(self, prompt, frame, pairs, thresh=0.5):
        gt = self.gt(prompt, frame)
        if not gt.any():
            return None
        pv = torch.zeros(self.snag.gaussian_num, dtype=torch.float32)
        for lvl, sp in pairs:
            pv[(self.snag.labels[lvl].long() == sp).cpu()] = 1.0
        self.g._semantics = pv.unsqueeze(-1).expand(-1, 20).cuda()
        e = render(self.frames[frame][0], self.g, self.pipe, self.bg)["semantics"]
        m = (e.reshape(20, -1)[0] > thresh).reshape(gt.shape[0], gt.shape[1])
        gt_t = torch.from_numpy(gt).cuda()
        inter = (m & gt_t).sum().item()
        union = (m | gt_t).sum().item()
        return inter / max(union, 1)


def ablation_variants(cfg):
    """config name -> dict of variant_name -> config name with one signal off."""
    import re
    m = re.match(r'a([\d.]+)_k(\d+)_tv(\d+)_g(off|[\d.]+)_pu(on|off)', cfg)
    if m is None:        # v2 naming (g2/tau_c) — ablation covered by v1 run
        return {}
    a, k, tv, g, pu = m.groups()
    mk = lambda tv_, g_, pu_: f"a{a}_k{k}_tv{tv_}_g{g_}_pu{pu_}"
    out = {}
    if tv != '0':
        out['E_off'] = mk('0', g, pu)
    if g != 'off':
        out['G_off'] = mk(tv, 'off', pu)
    if pu != 'off':
        out['P_off'] = mk(tv, g, 'off')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shortlist_n', type=int, default=3)
    ap.add_argument('--suffix', default='',
                    help="e.g. '_v2' to read/write *_v2 artifacts")
    args = ap.parse_args()
    sfx = args.suffix

    grid = pd.read_csv(f'{D}/stage4_grid_ranks{sfx}.csv')
    with open(f'{D}/stage4_selections{sfx}.pkl', 'rb') as f:
        selections = pickle.load(f)
    base_mask = pd.read_csv(f'{D}/stage3_3_mask_iou.csv')
    base_pp = base_mask.groupby(['scene', 'prompt'])['iou_baseline'].mean()
    configs = [c[5:] for c in grid.columns if c.startswith('rank_')]

    # ---------- Phase B1: rank-proxy shortlist per fold
    def proxy(cdf, cfg):
        r = cdf[f'rank_{cfg}']
        ph = (cdf.category == 'phantom17') & (r <= 3)
        ot = (cdf.category == 'other') & (r <= 3)
        ez = (cdf.category == 'easy') & (cdf.baseline_rank <= 3) & (r > 3)
        return int(ph.sum()) + int(ot.sum()) - 2 * int(ez.sum())

    folds = {}
    for held in SCENES:
        calib = grid[grid.scene != held]
        scored = sorted(((proxy(calib, c), c) for c in configs), reverse=True)
        folds[held] = [c for _, c in scored[:args.shortlist_n]]
        print(f"[fold {held}] shortlist by calib proxy: "
              + ", ".join(f"{c} ({s:+d})" for s, c in scored[:args.shortlist_n]),
              flush=True)

    # ---------- work plan: (scene -> set of configs to mask-eval)
    plan = defaultdict(set)
    for held, sl in folds.items():
        for c in sl:
            for sc in SCENES:
                if sc != held:
                    plan[sc].add(c)          # calibration evals

    # ---------- mask eval pass 1 (calibration)
    iou_rows = []

    def eval_scene_configs(scene, cfgs):
        ev = SceneEval(scene)
        prompts = sorted({p for (s, p) in selections if s == scene})
        for prompt in prompts:
            sel = selections[(scene, prompt)]
            for frame in ev.frames:
                if not ev.gt(prompt, frame).any():
                    continue
                for cfg in cfgs:
                    iou = ev.union_iou(prompt, frame, sel[cfg])
                    if iou is not None:
                        iou_rows.append({'scene': scene, 'prompt': prompt,
                                         'frame': frame, 'config': cfg,
                                         'iou': iou})
            torch.cuda.empty_cache()
        print(f"  [mask] {scene}: {len(cfgs)} configs done", flush=True)

    for sc in SCENES:
        if plan[sc]:
            eval_scene_configs(sc, sorted(plan[sc]))

    idf = pd.DataFrame(iou_rows)
    pp1 = idf.groupby(['scene', 'prompt', 'config'])['iou'].mean().reset_index()

    # ---------- Phase B2: choose per fold by calibration mIoU
    choice_rows = []
    chosen = {}
    for held, sl in folds.items():
        best, best_v = None, -1
        for c in sl:
            v = pp1[(pp1.scene != held) & (pp1.config == c)]['iou'].mean()
            if v > best_v:
                best, best_v = c, v
        chosen[held] = best
        choice_rows.append({'held_out': held, 'shortlist': ';'.join(sl),
                            'chosen': best, 'calib_mIoU': round(best_v, 4)})
        print(f"[fold {held}] chosen by calib mIoU: {best} ({best_v:.4f})", flush=True)
    pd.DataFrame(choice_rows).to_csv(f'{D}/stage4_loso_choice{sfx}.csv', index=False)

    # ---------- Phase C: held-out eval (+ablation variants)
    for held, cfg in chosen.items():
        need = {cfg} | set(ablation_variants(cfg).values())
        done = set(idf[idf.scene == held]['config'].unique()) if len(idf) else set()
        eval_scene_configs(held, sorted(need - done))
    idf = pd.DataFrame(iou_rows)
    idf.to_csv(f'{D}/stage4_mask_iou{sfx}.csv', index=False)
    pp = idf.groupby(['scene', 'prompt', 'config'])['iou'].mean().reset_index()

    # held-out per-prompt IoU under each fold's chosen config
    held_pp = []
    for held, cfg in chosen.items():
        d = pp[(pp.scene == held) & (pp.config == cfg)].copy()
        d['fold_config'] = cfg
        held_pp.append(d)
    hdf = pd.concat(held_pp)
    hdf = hdf.merge(grid[['scene', 'prompt', 'category']].drop_duplicates(),
                    on=['scene', 'prompt'])
    hdf['iou_baseline'] = [base_pp.loc[(s, p)] for s, p in
                           zip(hdf.scene, hdf.prompt)]

    multi = pd.read_csv(f'{D}/stage3_4_multi_instance.csv')
    mi_set = set(map(tuple, multi[multi.grade == 'suspect']
                     [['scene', 'prompt']].drop_duplicates().values))
    hdf['multi_instance'] = [(s, p) in mi_set for s, p in zip(hdf.scene, hdf.prompt)]

    print("\n" + "=" * 78)
    print("STAGE 4 — LOSO HELD-OUT RESULTS (full-67, pooled over 4 folds)")
    print("=" * 78)
    base = hdf['iou_baseline'].mean()
    ours = hdf['iou'].mean()
    print(f"  full-67 mIoU : baseline {base:.4f} -> ours {ours:.4f}  "
          f"({(ours - base) * 100:+.2f}pt)")
    for grp in ['phantom17', 'easy', 'other']:
        g = hdf[hdf.category == grp]
        print(f"  {grp:9s} (n={len(g):2d}): {g['iou_baseline'].mean():.4f} -> "
              f"{g['iou'].mean():.4f} ({(g['iou'].mean() - g['iou_baseline'].mean()) * 100:+.2f}pt)")
    ex = hdf[~hdf.multi_instance]
    print(f"  excl. multi-instance track (n={len(ex)}): "
          f"{ex['iou_baseline'].mean():.4f} -> {ex['iou'].mean():.4f} "
          f"({(ex['iou'].mean() - ex['iou_baseline'].mean()) * 100:+.2f}pt)")
    mi = hdf[hdf.multi_instance]
    if len(mi):
        print(f"  multi-instance track only (n={len(mi)}): "
              f"{mi['iou_baseline'].mean():.4f} -> {mi['iou'].mean():.4f}")

    # ablation
    ab_rows = []
    for held, cfg in chosen.items():
        for name, vcfg in ablation_variants(cfg).items():
            d = pp[(pp.scene == held) & (pp.config == vcfg)]
            full = pp[(pp.scene == held) & (pp.config == cfg)]
            ab_rows.append({'held_out': held, 'config': cfg, 'ablate': name,
                            'mIoU_full': round(full['iou'].mean(), 4),
                            'mIoU_ablated': round(d['iou'].mean(), 4)})
    abdf = pd.DataFrame(ab_rows)
    abdf.to_csv(f'{D}/stage4_ablation{sfx}.csv', index=False)
    print("\nABLATION (held-out, per fold):")
    if len(abdf):
        print(abdf.to_string(index=False))

    # R9
    delta = (ours - base) * 100
    ez = hdf[hdf.category == 'easy']
    ez_loss = (ez['iou_baseline'].mean() - ez['iou'].mean()) * 100
    print("\n" + "=" * 78)
    if delta >= 2.0 and ez_loss < 1.0:
        v = "R9 PASS — method 확정 (paper Section 3 진입)"
    elif delta >= 0.5:
        v = "R9 PARTIAL — 신호 조합 재탐색 1회 허용"
    else:
        v = "R9 FAIL — 정직 보고 + 잔여 설계 question 도출"
    print(f"[R9] full-67 Δ={delta:+.2f}pt (기준 +2.0), easy 손실={ez_loss:.2f}pt "
          f"(기준 <1.0) → {v}")
    print("=" * 78)


if __name__ == '__main__':
    main()
