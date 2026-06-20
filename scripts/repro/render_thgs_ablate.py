"""
Phase 1 — THGS reproduction ablation renderer.
Mirrors test_lerf.py exactly (same scoring/selection/render) but:
  - sweeps a grid of (level, topk) selection configs,
  - saves the SOFT presence mask (uint8 = round(soft*255)) so threshold can be
    swept on CPU later without re-rendering,
  - saves the GT mask once per (frame,prompt).
Inputs are the existing released checkpoint (output/lerf/<scene>/sai_nag.pt) — no retrain.

Usage (per scene, same args style as test_lerf.py):
  python scripts/repro/render_thgs_ablate.py -s data/lerf_ovs/<scene> -m output/lerf/<scene> \
      --path_pred output/render/repro/thgs
Output layout:
  <path_pred>/<config>/<scene>/<frame>/<prompt>.png       (soft, uint8)
  <path_pred>/<config>/<scene>/<frame>/<prompt>_gt.png    (binary 0/255)
config string: L<lvls>_k<topk>  e.g. L23_k3, L2_k3, L23_k1
"""
import os, sys, json
import torch
import numpy as np
import cv2
from argparse import ArgumentParser
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.vlm_utils import ClipSimMeasure
from nag_data import SemanticNAG

# (level_list, topk) grid. level_list passed to get_related_gaussian as `level`.
CONFIGS = [
    ([2, 3], 3),   # released default -> must reproduce 58.87
    ([2],    3),
    ([3],    3),
    ([2, 3], 1),
    ([2, 3], 5),
    ([2, 3], 10),
]

def cfg_name(lvls, topk):
    return f"L{''.join(str(l) for l in lvls)}_k{topk}"

def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask

@torch.no_grad()
def run(dataset, pipe, path_pred):
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 30000, load_sem=False)
    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    vlm = ClipSimMeasure(); vlm.load_model()
    snag = SemanticNAG(nag['nag'], nag['nag_feat'])

    scene_name = dataset.source_path.rstrip('/').split('/')[-1]
    data_path = os.path.join(os.path.dirname(dataset.source_path.rstrip('/')), 'label', scene_name)
    img_list = [f for f in os.listdir(data_path) if f.endswith('.jpg')]

    for im in img_list:
        image_name = im.split('.')[0]
        anno = json.load(open(os.path.join(data_path, image_name + '.json')))
        cam = None
        for c in scene.getTrainCameras():
            if c.image_name == image_name:
                cam = c; break
        if cam is None:
            print(f"[WARN] no camera for {image_name}"); continue
        h, w = cam.image_height, cam.image_width
        prompt_list = list(set(obj['category'] for obj in anno['objects']))

        for prompt in prompt_list:
            vlm.encode_text(prompt)
            sim = [vlm.compute_similarity(f) for f in snag.feat]  # per-level, config-independent
            # GT once
            mask_gt = np.zeros((h, w), dtype=np.uint8)
            for obj in anno['objects']:
                if obj['category'] == prompt:
                    mask_gt = np.maximum(mask_gt, polygon_to_mask((h, w), obj['segmentation']))
            pp = prompt.replace(' ', '_')
            for lvls, topk in CONFIGS:
                point_valid = snag.get_related_gaussian(sim, topk=topk, level=list(lvls))
                gaussians._semantics = point_valid.expand(-1, 20).cuda()
                embd = render(cam, gaussians, pipe, background)["semantics"]
                soft = embd.reshape(20, -1)[0].reshape(h, w).clamp(0, 1).cpu().numpy()
                out_dir = os.path.join(path_pred, cfg_name(lvls, topk), scene_name, image_name)
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, pp + '.png'), np.round(soft * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(out_dir, pp + '_gt.png'), mask_gt * 255)
        print(f"[{scene_name}] {image_name} done ({len(prompt_list)} prompts x {len(CONFIGS)} cfgs)")

if __name__ == "__main__":
    parser = ArgumentParser()
    lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
    parser.add_argument('--path_pred', type=str, default='output/render/repro/thgs')
    args = parser.parse_args(sys.argv[1:])
    safe_state(True)
    run(lp.extract(args), pp.extract(args), args.path_pred)
    print("render_thgs_ablate done.")
