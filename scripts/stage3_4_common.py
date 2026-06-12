"""
Stage 3.4 shared machinery — winner identification & identity analysis.

- regime scores per (scene, prompt): mean (pipeline) and query-top5, over the
  full [2,3] pool (reuses stage3_3 dump + sai_nag)
- winners_above(): SPs ranked above the oracle under a regime
- identity report per winner SP: max IoU vs the prompt's GT across ALL labeled
  frames, NAG kinship vs oracle, coherence / n_valid_views from the dump,
  RGB crop at the frame where the winner is most visible.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from stage3_3_fullpool_rank import load_dump, per_level_arrays, query_topk_agg, LEVELS
from stage3_4_nag_kinship import load_labels, relation
from stage3_2_common import make_vlm

from gaussian_renderer import render
from scene import Scene, GaussianModel
from nag_data import SemanticNAG
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


class SceneCtx:
    """Everything needed for one scene: dump scores, renderer, GT, kinship."""

    def __init__(self, scene_name, vlm):
        self.scene = scene_name
        self.vlm = vlm
        dump = load_dump(f'output/diagnostics/stage3_3_allsp_{scene_name}.pkl')
        self.lv = {l: per_level_arrays(dump, l, 'cuda') for l in LEVELS}
        self.entries = [(l, i) for l in LEVELS for i in range(self.lv[l][0].shape[1])]
        self.index = {e: i for i, e in enumerate(self.entries)}
        self.coher = np.concatenate([self.lv[l][3] for l in LEVELS])
        self.nval = np.concatenate([self.lv[l][4] for l in LEVELS])
        nagd = torch.load(f'output/lerf/{scene_name}/sai_nag.pt')
        self.base_feat = torch.cat([
            torch.nn.functional.normalize(nagd['nag_feat'][l - 1].cuda().float(),
                                          p=2, dim=-1) for l in LEVELS])
        self.labels = load_labels(scene_name)

        # renderer
        parser = ArgumentParser()
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        args = parser.parse_args(['-s', f'data/lerf_ovs/{scene_name}',
                                  '-m', f'output/lerf/{scene_name}'])
        ds = lp.extract(args)
        self.pipe = pp.extract(args)
        self.gaussians = GaussianModel(ds.sh_degree, 20)
        sc = Scene(ds, self.gaussians, 30000, load_sem=False, shuffle=False)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.snag = SemanticNAG(nagd['nag'], nagd['nag_feat'])
        self.cams = {c.image_name: c for c in sc.getTrainCameras()}
        self.image_root = f'data/lerf_ovs/{scene_name}/images'

        # GT frames
        label_dir = f'data/lerf_ovs/label/{scene_name}'
        self.gt = {}   # prompt -> {frame: bool mask}
        for im in sorted(os.listdir(label_dir)):
            if not im.endswith('.jpg'):
                continue
            name = im.split('.')[0]
            if name not in self.cams:
                continue
            anno = json.load(open(os.path.join(label_dir, name + '.json')))
            cam = self.cams[name]
            for o in anno['objects']:
                p = o['category']
                self.gt.setdefault(p, {})
                m = self.gt[p].get(name)
                pm = polygon_to_mask((cam.image_height, cam.image_width),
                                     o['segmentation']) > 0
                self.gt[p][name] = pm if m is None else (m | pm)

        self._score_cache = {}

    @torch.no_grad()
    def scores(self, prompt):
        """(mean_scores, top5_scores) over the pool, canon-contrast."""
        if prompt in self._score_cache:
            return self._score_cache[prompt]
        self.vlm.encode_text(prompt)
        text = self.vlm.text_feature[0].float()
        s_mean = self.vlm.compute_similarity(self.base_feat).cpu().numpy()
        aggs = torch.cat([query_topk_agg(*self.lv[l][:3], text) for l in LEVELS])
        s_top = self.vlm.compute_similarity(aggs).cpu().numpy()
        self._score_cache[prompt] = (s_mean, s_top)
        return s_mean, s_top

    def winners_above(self, prompt, oracle, regime, topn=10):
        s_mean, s_top = self.scores(prompt)
        s = s_mean if regime == 'mean' else s_top
        oi = self.index[oracle]
        order = np.argsort(-s)
        out = []
        for i in order:
            if i == oi:
                break
            out.append((self.entries[i], float(s[i])))
            if len(out) >= topn:
                break
        return out, int((s > s[oi]).sum()) + 1

    @torch.no_grad()
    def render_sp(self, cam, lvl, sp, thresh=0.5):
        pv = (self.snag.labels[lvl].long() == sp).float()
        self.gaussians._semantics = pv.unsqueeze(-1).expand(-1, 20).cuda()
        embd = render(cam, self.gaussians, self.pipe, self.background)["semantics"]
        return (embd.reshape(20, -1)[0] > thresh).reshape(
            cam.image_height, cam.image_width).cpu().numpy()

    def identity(self, prompt, oracle, winner):
        """Identity report of winner (lvl,sp) for this prompt."""
        lvl, sp = winner
        best_iou, best_recall, best_frame = 0.0, 0.0, ''
        for frame, gt in self.gt.get(prompt, {}).items():
            m = self.render_sp(self.cams[frame], lvl, sp)
            inter = (m & gt).sum()
            iou = inter / max((m | gt).sum(), 1)
            rec = inter / max(gt.sum(), 1)
            if iou > best_iou:
                best_iou, best_frame = iou, frame
            best_recall = max(best_recall, rec)
        rel, ca, cb = relation(self.labels, winner, oracle)
        wi = self.index[winner]
        return {'winner': f'{lvl}.{sp}', 'best_gt_iou': round(best_iou, 3),
                'best_gt_recall': round(best_recall, 3), 'best_gt_frame': best_frame,
                'kinship': rel, 'cont_w_in_o': round(ca, 2),
                'coherence': round(float(self.coher[wi]), 3),
                'n_valid_views': int(self.nval[wi])}

    def crop(self, prompt, winner, pad=6):
        """RGB blackout-crop of winner at the GT frame where it is largest."""
        lvl, sp = winner
        best, best_area = None, 0
        frames = list(self.gt.get(prompt, {}).keys()) or list(self.cams.keys())[:1]
        for frame in frames:
            m = self.render_sp(self.cams[frame], lvl, sp)
            if m.sum() > best_area:
                best_area, best = m.sum(), (frame, m)
        if best is None or best_area < 25:
            return None
        frame, m = best
        img = cv2.imread(os.path.join(self.image_root, frame + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cam = self.cams[frame]
        if img.shape[0] != cam.image_height:
            img = cv2.resize(img, (cam.image_width, cam.image_height))
        ys, xs = np.where(m)
        y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
        img2 = img.copy()
        img2[~m] = 0
        return img2[y0:y1, x0:x1]


def classify_winner(ident, gt_iou_thr=0.3):
    """Coarse identity class for taxonomy."""
    if ident['kinship'] in ('child_of_b', 'parent_of_b', 'same'):
        return 'kin_' + ident['kinship']
    if ident['best_gt_iou'] >= gt_iou_thr or ident['best_gt_recall'] >= 0.5:
        return 'overlaps_gt'
    if ident['n_valid_views'] == 0:
        return 'zero_norm_ghost'
    return 'external'
