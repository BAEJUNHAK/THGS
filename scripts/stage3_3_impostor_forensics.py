"""
Stage 3.3-B — impostor forensics: WHY do wrong-top1 SPs out-score oracles?

For each phantom's wrong-top1 SP:
  - top-5 "winning" views by query-cos (from its per-view replay dump)
  - crop AREA ratio wrong/oracle in shared winning views (context-size effect)
  - GT containment at the ref_frame (the only frame with GT):
      recall_gt  = |wrong_mask ∩ GT| / |GT|     (does it contain the object?)
      precision  = |wrong_mask ∩ GT| / |wrong|  (is it mostly the object?)
    NOTE: GT exists only on labeled frames -> containment measured at ref_frame,
    not at the winning views themselves (limitation, stated in the report).
  - montage: per phantom, top-3 winning views, oracle crop vs wrong crop.

R5 (pre-registered): fraction of phantoms with recall_gt >= 0.5
  >= 50% -> "frame-the-truth" mechanism / < 50% -> true semantic confusion.

Output: output/diagnostics/stage3_3_impostor.csv
        output/diagnostics/plots/stage3_3_impostor_montage.png
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

from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from arguments import ModelParams, PipelineParams, OptimizationParams

sys.path.insert(0, os.path.dirname(__file__))
from stage3_2_common import load_replay, visible_views, make_vlm


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


@torch.no_grad()
def render_sp_mask(cam, gaussians, pipe, background, snag, level, sp_id, thresh=0.5):
    point_valid = (snag.labels[level].long() == sp_id).float()
    gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
    embd = render(cam, gaussians, pipe, background)["semantics"]
    return (embd.reshape(20, -1)[0] > thresh).reshape(cam.image_height, cam.image_width)


def crop_of(img_rgb, mask_np, pad=4):
    ys, xs = np.where(mask_np)
    if len(ys) == 0:
        return None, 0
    y0, y1 = max(0, ys.min() - pad), min(img_rgb.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img_rgb.shape[1], xs.max() + pad)
    img2 = img_rgb.copy()
    img2[~mask_np] = 0
    return img2[y0:y1, x0:x1], int(mask_np.sum())


@torch.no_grad()
def run(dataset, pipe, args, panels, rows):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    wrong = load_replay(args.wrong_pkl)
    oracle = load_replay(args.oracle_pkl)
    keys = [k for k in wrong if k[0] == scene_name]
    if not keys:
        return
    print(f"\n=== {scene_name}: {len(keys)} impostors ===", flush=True)

    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 30000, load_sem=False, shuffle=False)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    vlm = make_vlm()
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    image_root = os.path.join(dataset.source_path, 'images')
    label_dir = os.path.join(os.path.dirname(dataset.source_path), 'label', scene_name)

    for (sc, prompt) in sorted(keys):
        w_rec, o_rec = wrong[(sc, prompt)], oracle.get((sc, prompt))
        wv = visible_views(w_rec)
        if not wv or o_rec is None:
            rows.append([sc, prompt, '', '', '', '', 'skip_empty'])
            continue
        vlm.encode_text(prompt)
        text = vlm.text_feature[0].cpu().numpy()
        qcos = np.array([v['mixed_feat'].astype(np.float32) @ text for v in wv])
        win = [wv[i] for i in np.argsort(-qcos)[:5]]

        # GT containment at ref frame (first labeled frame containing prompt)
        recall_gt = precision = np.nan
        ref_name = None
        for im in sorted(os.listdir(label_dir)):
            if not im.endswith('.jpg'):
                continue
            name = im.split('.')[0]
            anno = json.load(open(os.path.join(label_dir, name + '.json')))
            if any(o['category'] == prompt for o in anno['objects']) and name in cams:
                ref_name = name
                cam = cams[name]
                gt = np.zeros((cam.image_height, cam.image_width), np.uint8)
                for o in anno['objects']:
                    if o['category'] == prompt:
                        gt = np.maximum(gt, polygon_to_mask(gt.shape, o['segmentation']))
                gt = gt > 0
                wm = render_sp_mask(cam, gaussians, pipe, background, snag,
                                    w_rec['oracle_lvl'], w_rec['oracle_sp_id']).cpu().numpy()
                inter = (wm & gt).sum()
                recall_gt = inter / max(gt.sum(), 1)
                precision = inter / max(wm.sum(), 1)
                break

        # area ratio + montage panels on top winning views
        ratios = []
        panel_row = []
        for v in win[:3]:
            if v['image_name'] not in cams:
                continue
            cam = cams[v['image_name']]
            img = cv2.imread(os.path.join(image_root, cam.image_name + '.jpg'))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != cam.image_height:
                img = cv2.resize(img, (cam.image_width, cam.image_height))
            wm = render_sp_mask(cam, gaussians, pipe, background, snag,
                                w_rec['oracle_lvl'], w_rec['oracle_sp_id']).cpu().numpy()
            om = render_sp_mask(cam, gaussians, pipe, background, snag,
                                o_rec['oracle_lvl'], o_rec['oracle_sp_id']).cpu().numpy()
            wc, wa = crop_of(img, wm)
            oc, oa = crop_of(img, om)
            if wa and oa:
                ratios.append(wa / oa)
            if wc is not None and oc is not None and len(panel_row) < 2:
                panel_row.append((oc, wc))
        area_ratio = float(np.median(ratios)) if ratios else np.nan
        rows.append([sc, prompt, f"{recall_gt:.3f}", f"{precision:.3f}",
                     f"{area_ratio:.2f}", ref_name or '', ''])
        if panel_row:
            panels.append((f"{sc[:4]}/{prompt}", panel_row[0]))
        print(f"  {prompt:24s} recall_gt={recall_gt:.2f} precision={precision:.2f} "
              f"area_ratio(w/o)={area_ratio:.2f}", flush=True)
        torch.cuda.empty_cache()


def save_montage(panels, out_png, h=160):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.2 * n))
    if n == 1:
        axes = axes[None, :]
    for i, (label, (oc, wc)) in enumerate(panels):
        axes[i, 0].imshow(oc); axes[i, 0].set_title(f"{label}\nORACLE crop", fontsize=7)
        axes[i, 1].imshow(wc); axes[i, 1].set_title("WRONG-top1 crop (winning view)", fontsize=7)
        for j in (0, 1):
            axes[i, j].axis('off')
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--wrong_pkl", default="output/diagnostics/stage3_2_wrongtop1_perview.pkl")
    parser.add_argument("--oracle_pkl", default="output/diagnostics/stage3_b8_replay_perview.pkl")
    parser.add_argument("--out_csv", default="output/diagnostics/stage3_3_impostor.csv")
    parser.add_argument("--out_png", default="output/diagnostics/plots/stage3_3_impostor_montage.png")
    parser.add_argument("--panels_pkl", default="output/diagnostics/_stage3_3_panels.pkl")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--append", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)

    panels, rows = [], []
    if args.append and os.path.exists(args.panels_pkl):
        with open(args.panels_pkl, 'rb') as f:
            panels = pickle.load(f)
    run(lp.extract(args), pp.extract(args), args, panels, rows)
    with open(args.panels_pkl, 'wb') as f:
        pickle.dump(panels, f)

    new_h = (args.append == 0 or not os.path.exists(args.out_csv))
    with open(args.out_csv, 'w' if args.append == 0 else 'a', newline='') as f:
        w = csv.writer(f)
        if new_h:
            w.writerow(['scene', 'prompt', 'recall_gt', 'precision',
                        'area_ratio_wrong_over_oracle', 'ref_frame', 'note'])
        for r in rows:
            w.writerow(r)
    save_montage(panels, args.out_png)
    print("Done.")
