"""
P1-A step 4 — mask-level mIoU of autopsy variants (THGS, R11-d).

Same machinery as stage3_3_mask_eval.py, but reads p1a_selections_thgs.pkl
(rule -> top-3 (lvl, sp)) for VARIANTS = top5/qmax1/gm_u/gm_w/gm_g.
Baseline column recomputed identically to stage3_3 (canon top-3, levels [2,3]).

Output: output/diagnostics/p1a_mask_iou.csv
"""

import os
import sys
import csv
import json
import pickle
import numpy as np
import torch
import cv2
from argparse import ArgumentParser
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from utils.vlm_utils import ClipSimMeasure
from arguments import ModelParams, PipelineParams, OptimizationParams

VARIANTS = ['top5', 'qmax1', 'gm_u', 'gm_w', 'gm_g']


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


@torch.no_grad()
def render_union(cam, gaussians, pipe, background, snag, pairs, thresh):
    point_valid = torch.zeros(snag.gaussian_num, dtype=torch.float32)
    for lvl, sp_id in pairs:
        point_valid[(snag.labels[lvl].long() == sp_id).cpu()] = 1.0
    gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
    embd_sim = render(cam, gaussians, pipe, background)["semantics"]
    return (embd_sim.reshape(20, -1)[0] > thresh).reshape(cam.image_height, cam.image_width)


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    print(f"\n=== {scene_name} (P1-A mask eval) ===", flush=True)
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, args.iteration if args.iteration > 0 else -1,
                  load_sem=False, shuffle=False)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    vlm = ClipSimMeasure()
    vlm.load_model()

    with open(args.selections, 'rb') as f:
        selections = pickle.load(f)

    data_path = os.path.join(os.path.dirname(dataset.source_path), 'label', scene_name)
    img_list = sorted([f for f in os.listdir(data_path) if f.endswith('.jpg')])
    train_cams_by_name = {cam.image_name: cam for cam in scene.getTrainCameras()}
    frame_data = {}
    for im in img_list:
        image_name = im.split('.')[0]
        anno = json.load(open(os.path.join(data_path, image_name + '.json')))
        if image_name not in train_cams_by_name:
            continue
        cam = train_cams_by_name[image_name]
        frame_data[image_name] = {'cam': cam, 'objects': anno['objects'],
                                  'h': cam.image_height, 'w': cam.image_width}
    prompt_to_frames = defaultdict(list)
    for fr, fd in frame_data.items():
        for p in set(o['category'] for o in fd['objects']):
            prompt_to_frames[p].append(fr)

    def get_gt_for(prompt, frame_name):
        fd = frame_data[frame_name]
        mask = np.zeros((fd['h'], fd['w']), dtype=np.uint8)
        for obj in fd['objects']:
            if obj['category'] == prompt:
                mask = np.maximum(mask, polygon_to_mask((fd['h'], fd['w']),
                                                        obj['segmentation']))
        return mask > 0

    rows = []
    for prompt in sorted(prompt_to_frames.keys()):
        if (scene_name, prompt) not in selections:
            print(f"  [skip] no selections for {prompt}")
            continue
        frames = sorted(prompt_to_frames[prompt])
        vlm.encode_text(prompt)
        sims = {lvl: vlm.compute_similarity(snag.feat[lvl - 1]).cpu().numpy()
                for lvl in (2, 3)}
        flat = [(lvl, i, sims[lvl][i]) for lvl in (2, 3)
                for i in range(len(sims[lvl]))]
        flat.sort(key=lambda x: -x[2])
        sel_all = {'baseline': [(l, i) for l, i, _ in flat[:3]]}
        for v in VARIANTS:
            sel_all[v] = [tuple(x) for x in selections[(scene_name, prompt)][v]]

        for eval_frame in frames:
            gt_np = get_gt_for(prompt, eval_frame)
            if not gt_np.any():
                continue
            gt = torch.from_numpy(gt_np).cuda()
            cam = frame_data[eval_frame]['cam']
            row = [scene_name, prompt, eval_frame]
            for v in ['baseline'] + VARIANTS:
                m = render_union(cam, gaussians, pipe, background, snag,
                                 sel_all[v], args.thresh)
                inter = (m & gt).sum().item()
                union = (m | gt).sum().item()
                row.append(f"{inter / max(union, 1):.4f}")
                del m
            rows.append(row)
            del gt
        torch.cuda.empty_cache()
        print(f"  {prompt}: {len(frames)} frames done", flush=True)
    return rows


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--selections", default="output/diagnostics/p1a_selections_thgs.pkl")
    parser.add_argument("--out_csv", default="output/diagnostics/p1a_mask_iou.csv")
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--append", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    new_h = (args.append == 0 or not os.path.exists(args.out_csv))
    f = open(args.out_csv, 'w' if args.append == 0 else 'a', newline='')
    w = csv.writer(f)
    if new_h:
        w.writerow(['scene', 'prompt', 'eval_frame', 'iou_baseline'] +
                   [f'iou_{v}' for v in VARIANTS])
    pipe = pp.extract(args)
    for r in run(lp.extract(args), pipe, args):
        w.writerow(r)
    f.close()
    print("Done.")
