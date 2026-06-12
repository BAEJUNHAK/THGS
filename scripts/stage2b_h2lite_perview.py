"""
Stage 2B — F2.A: H2 lite instrument.

For each 17 phantom oracle SP, compute per-view CLIP image features:
  1. Render the oracle SP mask at each train view
  2. Crop the RGB image to the SP region (mask-blackout style, matches A2 method)
  3. CLIP image-encode the crop → per-view feature
  4. Save (scene, prompt, view_idx, visibility, feature[512])

Then F2.B (subtype classification) runs on this dump.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import cv2
from argparse import ArgumentParser
import pickle

import open_clip
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from nag_data import SemanticNAG
from arguments import ModelParams, PipelineParams, OptimizationParams


@torch.no_grad()
def render_sp_mask(cam, gaussians, pipe, background, snag, level, sp_id, thresh):
    point_valid = (snag.labels[level].long() == sp_id).float()
    gaussians._semantics = point_valid.unsqueeze(-1).expand(-1, 20).cuda()
    embd_sim = render(cam, gaussians, pipe, background)["semantics"]
    return (embd_sim.reshape(20, -1)[0] > thresh).reshape(cam.image_height, cam.image_width)


def bbox_of_mask(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


@torch.no_grad()
def run(dataset, pipe, args):
    scene_name = os.path.basename(os.path.normpath(dataset.source_path))
    b7 = pd.read_csv(args.b7_csv)
    ref = b7[b7['is_ref_frame'] == 1].copy()
    phantoms = pd.read_csv(args.persistent_csv)
    phantom_set = set((r['scene'], r['prompt']) for _, r in phantoms.iterrows())
    targets = ref[ref.apply(lambda r: (r['scene'], r['prompt']) in phantom_set, axis=1)]
    targets = targets[targets['scene'] == scene_name]
    if len(targets) == 0:
        print(f"[skip] no targets for {scene_name}"); return
    print(f"\n=== {scene_name}: {len(targets)} phantoms ===", flush=True)

    gaussians = GaussianModel(dataset.sh_degree, 20)
    iter_arg = args.iteration if args.iteration > 0 else -1
    scene = Scene(dataset, gaussians, iter_arg, load_sem=False, shuffle=False)
    if args.iteration == 0:
        ply_path = os.path.join(dataset.model_path, "point_cloud", "iteration_0", "point_cloud.ply")
        if os.path.exists(ply_path):
            gaussians.load_ply(ply_path)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])

    # Load CLIP
    device = torch.device("cuda")
    print("Loading CLIP ViT-B-16...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp16")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    image_root = os.path.join(dataset.source_path, 'images')
    train_cams = scene.getTrainCameras()
    # Subsample for efficiency
    step = max(1, len(train_cams) // args.view_subsample)
    sampled_cams = train_cams[::step][:args.view_subsample]
    print(f"  {len(train_cams)} train cams -> sampling {len(sampled_cams)}", flush=True)

    # encode all targets' prompt texts once
    prompts = list(targets['prompt'].unique())
    tok = tokenizer(prompts).to(device)
    text_feats = model.encode_text(tok).float()
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    prompt_to_text_idx = {p: i for i, p in enumerate(prompts)}

    results = {}  # (scene, prompt) -> list of (view_idx, image_name, visibility, feature, cos_with_prompt)

    for _, r in targets.iterrows():
        prompt = r['prompt']
        oracle_lvl = int(r['oracle_lvl']); oracle_sp = int(r['oracle_sp_id'])
        per_view_features = []
        for view_idx, cam in enumerate(sampled_cams):
            mask = render_sp_mask(cam, gaussians, pipe, background, snag,
                                  oracle_lvl, oracle_sp, args.thresh)
            n_pix = int(mask.sum().item())
            if n_pix < args.min_pixels:
                del mask; continue
            mask_np = mask.cpu().numpy()
            bbox = bbox_of_mask(mask_np)
            if bbox is None:
                del mask; continue
            x0, y0, x1, y1 = bbox

            # Load image
            img_path = os.path.join(image_root, cam.image_name + '.jpg')
            if not os.path.exists(img_path):
                del mask; continue
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img_rgb.shape[0] != cam.image_height or img_rgb.shape[1] != cam.image_width:
                img_rgb = cv2.resize(img_rgb, (cam.image_width, cam.image_height))

            # blackout-bg + bbox crop (same as A2 mask policy)
            img2 = img_rgb.copy()
            img2[~mask_np] = 0
            crop = img2[y0:y1, x0:x1]
            if crop.shape[0] < 4 or crop.shape[1] < 4:
                del mask; continue
            from PIL import Image
            pil = Image.fromarray(crop)
            pre = preprocess(pil).unsqueeze(0).to(device).half()
            img_feat = model.encode_image(pre).float()
            img_feat = (img_feat / img_feat.norm(dim=-1, keepdim=True)).squeeze(0).cpu().numpy()

            # cos with this target prompt
            cos = float(np.dot(img_feat, text_feats[prompt_to_text_idx[prompt]].cpu().numpy()))
            per_view_features.append({
                'view_idx': view_idx,
                'image_name': cam.image_name,
                'visibility_pixels': n_pix,
                'feature': img_feat.astype(np.float32),
                'cos_with_prompt': cos,
            })
            del mask
        torch.cuda.empty_cache()

        if len(per_view_features) > 0:
            results[(scene_name, prompt)] = per_view_features
            cos_arr = np.array([d['cos_with_prompt'] for d in per_view_features])
            print(f"  {prompt:25s} n_views={len(per_view_features):2d} "
                  f"cos_mean={cos_arr.mean():.3f} cos_min={cos_arr.min():.3f} "
                  f"cos_max={cos_arr.max():.3f} cos_std={cos_arr.std():.3f}", flush=True)
        else:
            print(f"  {prompt}: no visible views");

    # save dump
    out_pkl = args.out_pkl
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    if os.path.exists(out_pkl) and args.append:
        with open(out_pkl, "rb") as f:
            existing = pickle.load(f)
        existing.update(results)
        results = existing
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"\nWrote {len(results)} prompts to {out_pkl}")


if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--b7_csv", default="output/diagnostics/b7_a4_combined.csv")
    parser.add_argument("--persistent_csv", default="output/diagnostics/persistent_phantoms_17.csv")
    parser.add_argument("--out_pkl", default="output/diagnostics/_h2_lite_perview.pkl")
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--view_subsample", type=int, default=30)
    parser.add_argument("--min_pixels", type=int, default=50)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--append", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    safe_state(False)
    run(lp.extract(args), pp.extract(args), args)
    print("Done.")
