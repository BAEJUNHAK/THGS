"""
P1-B audit montage — render the top-1 SP for the highest-confidence ABSENT
queries (THGS, mean rule) and overlay on the scene's first label frame.
Run per scene: python scripts/p1b_absence_montage.py -s data/lerf_ovs/<scene>
               -m output/lerf/<scene> --iteration 30000
Panels accumulate into output/diagnostics/plots/p1b_absence_montage_<scene>.png
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import cv2
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from arguments import ModelParams, PipelineParams, OptimizationParams

N_TOP = 6


@torch.no_grad()
def main(dataset, pipe, args):
    import pandas as pd
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    df = pd.read_csv('output/diagnostics/p1b_absence_scores.csv')
    sub = df[(df.method == 'thgs') & (df.scene == scene_name)
             & (df.is_absent == 1)].nlargest(N_TOP, 'top1_mean')
    if len(sub) == 0:
        print('no absent rows for scene')
        return
    with open('output/diagnostics/p1b_top1_selections.pkl', 'rb') as f:
        sel = pickle.load(f)

    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, args.iteration, load_sem=False, shuffle=False)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])

    label_dir = os.path.join(os.path.dirname(dataset.source_path), 'label', scene_name)
    frame = sorted([f for f in os.listdir(label_dir) if f.endswith('.jpg')])[0]
    image_name = frame.split('.')[0]
    cam = {c.image_name: c for c in scene.getTrainCameras()}[image_name]
    base_img = cv2.imread(os.path.join(label_dir, frame))
    base_img = cv2.resize(base_img, (cam.image_width, cam.image_height))

    panels = []
    for _, r in sub.iterrows():
        lvl, sp = sel[('thgs', scene_name, r['query'], 1)]['mean']
        pv = torch.zeros(snag.gaussian_num, dtype=torch.float32)
        pv[(snag.labels[lvl].long() == sp).cpu()] = 1.0
        gaussians._semantics = pv.unsqueeze(-1).expand(-1, 20).cuda()
        sem = render(cam, gaussians, pipe, background)["semantics"]
        m = (sem.reshape(20, -1)[0] > 0.5).reshape(
            cam.image_height, cam.image_width).cpu().numpy()
        ov = base_img.copy()
        ov[m] = (0.4 * ov[m] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.putText(ov, f"ABSENT: {r['query']} ({r['top1_mean']:.3f})",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        panels.append(cv2.resize(ov, (640, int(640 * ov.shape[0] / ov.shape[1]))))
    rows_im = [np.hstack(panels[i:i + 3]) for i in range(0, len(panels), 3)]
    out = np.vstack(rows_im) if len(rows_im) > 1 else rows_im[0]
    os.makedirs('output/diagnostics/plots', exist_ok=True)
    path = f'output/diagnostics/plots/p1b_absence_montage_{scene_name}.png'
    cv2.imwrite(path, out)
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)
    main(lp.extract(args), pp.extract(args), args)
