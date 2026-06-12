"""Compare my diagnostic Actual mechanism vs test_lerf.py's get_related_gaussian.

Renders both methods at ramen/frame_00024 for 'chopsticks' and prints
selected SPs + IoU.
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from argparse import ArgumentParser

from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from utils.vlm_utils import ClipSimMeasure
from arguments import ModelParams, PipelineParams, OptimizationParams


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


@torch.no_grad()
def main(dataset, pipe):
    scene = Scene(dataset, GaussianModel(dataset.sh_degree, 20), 30000,
                  load_sem=False, shuffle=False)
    gaussians = scene.gaussians
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])

    vlm = ClipSimMeasure()
    vlm.load_model()

    frame_name = 'frame_00024'
    prompt = 'chopsticks'
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    data_path = os.path.join(os.path.dirname(dataset.source_path), 'label', scene_name)

    js_file = os.path.join(data_path, frame_name + '.json')
    anno = json.load(open(js_file))
    cam = None
    for c in scene.getTrainCameras():
        if c.image_name == frame_name:
            cam = c
            break
    h, w = cam.image_height, cam.image_width

    # GT
    gt = np.zeros((h, w), dtype=np.uint8)
    for obj in anno['objects']:
        if obj['category'] == prompt:
            gt = np.maximum(gt, polygon_to_mask((h, w), obj['segmentation']))
    gt = torch.from_numpy(gt > 0).cuda()

    # Method A: test_lerf.py path
    vlm.encode_text(prompt)
    sims = [vlm.compute_similarity(f) for f in snag.feat]
    pv_A = snag.get_related_gaussian(sims, topk=3, level=[2, 3])
    pv_A = pv_A.expand(-1, 20).cuda()
    gaussians._semantics = pv_A
    out_A = render(cam, gaussians, pipe, background)["semantics"]
    mask_A = (out_A.reshape(20, -1)[0] > 0.5).reshape(h, w)
    iou_A = (mask_A & gt).sum().item() / max((mask_A | gt).sum().item(), 1)
    print(f"\n=== Method A: test_lerf.py path ===")
    print(f"  pv_A non-zero gaussians: {pv_A[:, 0].sum().item()}")
    print(f"  rendered mask pixel sum: {mask_A.sum().item()}")
    print(f"  IoU vs GT: {iou_A:.4f}")

    # Method B: my diagnostic path
    # Build cache_lvl/cache_sp order (same as diagnostic)
    levels = [2, 3]
    cache_lvl, cache_sp = [], []
    for lvl in levels:
        for sp_id in snag.labels[lvl].long().unique().tolist():
            cache_lvl.append(lvl)
            cache_sp.append(int(sp_id))
    # Per-SP CLIP score using sims
    pool_scores = []
    for lvl, sp_id in zip(cache_lvl, cache_sp):
        score = sims[lvl - 1][sp_id].item()
        pool_scores.append(score)
    pool_scores_arr = np.array(pool_scores)
    order = np.argsort(-pool_scores_arr)
    actual_sel = order[:3].tolist()
    selected_pairs_B = [(cache_lvl[i], cache_sp[i]) for i in actual_sel]
    print(f"\n=== Method B: my diagnostic path ===")
    print(f"  selected (lvl, sp_id): {selected_pairs_B}")
    print(f"  pool scores (top 5): {sorted(pool_scores, reverse=True)[:5]}")

    pv_B = torch.zeros(snag.gaussian_num, dtype=torch.float32, device='cuda')
    for lvl, sp_id in selected_pairs_B:
        picked = snag.labels[lvl].long() == sp_id
        pv_B[picked] = 1.0
    pv_B = pv_B.unsqueeze(-1).expand(-1, 20).cuda()
    gaussians._semantics = pv_B
    out_B = render(cam, gaussians, pipe, background)["semantics"]
    mask_B = (out_B.reshape(20, -1)[0] > 0.5).reshape(h, w)
    iou_B = (mask_B & gt).sum().item() / max((mask_B | gt).sum().item(), 1)
    print(f"  pv_B non-zero gaussians: {pv_B[:, 0].sum().item()}")
    print(f"  rendered mask pixel sum: {mask_B.sum().item()}")
    print(f"  IoU vs GT: {iou_B:.4f}")

    # Method A's selected SPs
    print(f"\n=== Method A SPs (re-derived from get_related_gaussian internals) ===")
    related_sp_lvl = []
    for i in [1, 2]:
        sim_array = sims[i]
        sim_val, indices = torch.topk(sim_array, 3)
        for j in range(3):
            related_sp_lvl.append((i + 1, sim_val[j].item(), indices[j].item()))
    related_sp_lvl.sort(key=lambda x: x[1], reverse=True)
    related_sp_lvl = related_sp_lvl[:3]
    print(f"  selected (lvl, sim, sp_id): {related_sp_lvl}")

    # Compare existing saved render
    pr_path = f'output/render/lerf/{scene_name}/{frame_name}/{prompt}.png'
    saved_mask = cv2.imread(pr_path, cv2.IMREAD_GRAYSCALE) > 128
    iou_saved = np.logical_and(saved_mask, gt.cpu().numpy()).sum() / max(np.logical_or(saved_mask, gt.cpu().numpy()).sum(), 1)
    print(f"\n=== Saved output (May 28) ===")
    print(f"  pixel sum: {saved_mask.sum()}")
    print(f"  IoU: {iou_saved:.4f}")

    # Diff masks
    print(f"\n=== Diff masks ===")
    diff_AB = (mask_A.cpu().numpy() != mask_B.cpu().numpy()).sum()
    diff_A_saved = (mask_A.cpu().numpy() != saved_mask).sum()
    diff_B_saved = (mask_B.cpu().numpy() != saved_mask).sum()
    print(f"  A vs B differing pixels: {diff_AB}")
    print(f"  A vs saved differing pixels: {diff_A_saved}")
    print(f"  B vs saved differing pixels: {diff_B_saved}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)
    main(lp.extract(args), pp.extract(args))
