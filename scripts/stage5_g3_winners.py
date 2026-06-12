"""
Stage 5 / G3 — winner-signature replication on ReLaGS.

For ReLaGS phantoms NOT recovered by query-top5 (from G2) and easy
regressions: identify the SPs beating the oracle under each regime and
report their signatures — n_valid_views / coherence / max IoU vs the
prompt's GT across labeled frames / NAG kinship vs oracle.

THGS reference signatures to compare against:
  few-view opportunist (winners nv 2-11, coh 0.92-0.99, GT-IoU 0)
  coherent confuser    (winners nv high, coh > oracle, GT-IoU 0)

Output: output/diagnostics/stage5_g3_winners.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(__file__))
from stage3_3_fullpool_rank import load_dump, per_level_arrays, query_topk_agg, LEVELS
from stage3_4_nag_kinship import relation
from stage3_2_common import make_vlm

from gaussian_renderer import render
from scene import Scene, GaussianModel
from nag_data import SemanticNAG
from arguments import ModelParams, PipelineParams, OptimizationParams

SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']


def polygon_to_mask(shape, pts):
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(m, [np.asarray(pts, dtype=np.int32)], 1)
    return m


class RelagsScene:
    def __init__(self, scene_name, model_root):
        parser = ArgumentParser()
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        a = parser.parse_args(['-s', f'data/lerf_ovs/{scene_name}',
                               '-m', f'{model_root}/{scene_name}'])
        ds = lp.extract(a)
        self.pipe = pp.extract(a)
        self.g = GaussianModel(ds.sh_degree, 20)
        sc = Scene(ds, self.g, -1, load_sem=False, shuffle=False)
        ply = os.path.join(ds.model_path, 'point_cloud', 'iteration_0',
                           'point_cloud.ply')
        if os.path.exists(ply):
            self.g.load_ply(ply)
        self.bg = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
        nag = torch.load(f'{model_root}/{scene_name}/sai_nag.pt')
        self.snag = SemanticNAG(nag['nag'], nag['nag_feat'])
        self.labels_np = [l.long().cpu().numpy() for l in nag['nag']]
        cams = {c.image_name: c for c in sc.getTrainCameras()}
        label_dir = f'data/lerf_ovs/label/{scene_name}'
        self.gt = {}
        for im in sorted(os.listdir(label_dir)):
            if not im.endswith('.jpg'):
                continue
            n = im.split('.')[0]
            if n not in cams:
                continue
            anno = json.load(open(os.path.join(label_dir, n + '.json')))
            cam = cams[n]
            for o in anno['objects']:
                p = o['category']
                self.gt.setdefault(p, {})
                pm = polygon_to_mask((cam.image_height, cam.image_width),
                                     o['segmentation']) > 0
                self.gt[p][n] = self.gt[p].get(n, np.zeros_like(pm)) | pm
        self.cams = cams

    @torch.no_grad()
    def sp_mask(self, cam, lvl, sp):
        pv = (self.snag.labels[lvl].long() == sp).float()
        self.g._semantics = pv.unsqueeze(-1).expand(-1, 20).cuda()
        e = render(cam, self.g, self.pipe, self.bg)['semantics']
        return (e.reshape(20, -1)[0] > 0.5).reshape(
            cam.image_height, cam.image_width).cpu().numpy()

    def gt_iou_max(self, prompt, lvl, sp):
        best = 0.0
        for frame, gt in self.gt.get(prompt, {}).items():
            m = self.sp_mask(self.cams[frame], lvl, sp)
            iou = (m & gt).sum() / max((m | gt).sum(), 1)
            best = max(best, float(iou))
        return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_root', default='ReLaGS/output/lerf_hf/scenes/LeRF')
    ap.add_argument('--dump_tpl', default='output/diagnostics/stage5_relags_allsp_{}.pkl')
    ap.add_argument('--g2_csv', default='output/diagnostics/stage5_g2_fullpool.csv')
    ap.add_argument('--out_csv', default='output/diagnostics/stage5_g3_winners.csv')
    args = ap.parse_args()

    g2 = pd.read_csv(args.g2_csv)
    rb = pd.read_csv('output/diagnostics/b7_a4_combined_relags.csv')
    ref = rb[rb['is_ref_frame'] == 1].set_index(['scene', 'prompt'])
    cases = pd.concat([
        g2[(g2.category == 'phantom') & (g2.rank_top5 > 3)].assign(kind='unrescued_phantom'),
        g2[(g2.category == 'easy') & (g2.rank_base_mean <= 3) & (g2.rank_top5 > 3)]
        .assign(kind='easy_regression'),
    ])
    print(f"cases: {len(cases)} ({cases.kind.value_counts().to_dict()})")

    vlm = make_vlm()
    rows = []
    for scene in SCENES:
        cs = cases[cases.scene == scene]
        if len(cs) == 0:
            continue
        dump = load_dump(args.dump_tpl.format(scene))
        lv = {l: per_level_arrays(dump, l, 'cuda') for l in LEVELS}
        entries = [(l, i) for l in LEVELS for i in range(lv[l][0].shape[1])]
        index = {e: i for i, e in enumerate(entries)}
        coher = np.concatenate([lv[l][3] for l in LEVELS])
        nval = np.concatenate([lv[l][4] for l in LEVELS]).astype(int)
        nag = torch.load(f'{args.model_root}/{scene}/sai_nag.pt')
        base_feat = torch.cat([
            torch.nn.functional.normalize(nag['nag_feat'][l - 1].cuda().float(),
                                          p=2, dim=-1) for l in LEVELS])
        rsc = RelagsScene(scene, args.model_root)
        for _, c in cs.iterrows():
            prompt = c['prompt']
            o = ref.loc[(scene, prompt)]
            okey = (int(o['oracle_lvl']), int(o['oracle_sp_id']))
            oi = index[okey]
            vlm.encode_text(prompt)
            text = vlm.text_feature[0].float()
            regime = 'top5' if c['kind'] == 'easy_regression' else 'mean'
            if regime == 'mean':
                s = vlm.compute_similarity(base_feat).cpu().numpy()
            else:
                aggs = torch.cat([query_topk_agg(*lv[l][:3], text, k=5)
                                  for l in LEVELS])
                s = vlm.compute_similarity(aggs).cpu().numpy()
            order = np.argsort(-s)
            winners = [entries[i] for i in order if i != oi][:5]
            for wrank, w in enumerate(winners, 1):
                wi = index[w]
                rel, ca, cb = relation(rsc.labels_np, w, okey)
                giou = rsc.gt_iou_max(prompt, *w)
                rows.append({'scene': scene, 'prompt': prompt, 'kind': c['kind'],
                             'regime': regime, 'winner_rank': wrank,
                             'winner': f'{w[0]}.{w[1]}',
                             'n_views': int(nval[wi]),
                             'coherence': round(float(coher[wi]), 3),
                             'gt_iou_max': round(giou, 3), 'kinship': rel})
            top = [r for r in rows if r['scene'] == scene and r['prompt'] == prompt]
            print(f"  {c['kind'][:16]:16s} {prompt:22s} winners: " +
                  " ".join(f"nv={t['n_views']}/coh={t['coherence']:.2f}"
                           f"/iou={t['gt_iou_max']:.2f}" for t in top[:3]), flush=True)
        del lv, base_feat, rsc
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print("\n" + "=" * 70)
    for kind in df['kind'].unique():
        d = df[df.kind == kind]
        few = (d['n_views'] <= 15).mean()
        ext = (d['gt_iou_max'] < 0.1).mean()
        print(f"[G3] {kind}: winners n={len(d)}  few-view(<=15v) {few*100:.0f}%  "
              f"GT-무관 {ext*100:.0f}%  coh median {d['coherence'].median():.2f}  "
              f"kinship: {d['kinship'].value_counts().to_dict()}")
    print(f"Wrote {args.out_csv}")


if __name__ == '__main__':
    main()
