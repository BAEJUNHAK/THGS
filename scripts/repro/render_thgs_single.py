"""THGS single-config render (L23_k3 = released default) for seeded fair comparison.
Loads a THGS-pipeline sai_nag.pt, selects via THGS get_related_gaussian(topk=3, level=[2,3]),
saves SOFT presence mask (uint8) in <path_pred>/<scene>/<frame>/<prompt>.png for eval_soft.
Run from THGS root (PYTHONPATH=repo root)."""
import os, sys, json
import torch, numpy as np, cv2
from argparse import ArgumentParser
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.vlm_utils import ClipSimMeasure
from nag_data import SemanticNAG

def polygon_to_mask(shape, pts):
    m = np.zeros(shape, dtype=np.uint8); cv2.fillPoly(m, [np.asarray(pts, dtype=np.int32)], 1); return m

@torch.no_grad()
def run(dataset, pipe, path_pred):
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 30000, load_sem=False)
    bg = [1,1,1] if dataset.white_background else [0,0,0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")
    nag = torch.load(os.path.join(dataset.model_path, "sai_nag.pt"))
    vlm = ClipSimMeasure(); vlm.load_model()
    snag = SemanticNAG(nag['nag'], nag['nag_feat'])
    scene_name = dataset.source_path.rstrip('/').split('/')[-1]
    data_path = os.path.join(os.path.dirname(dataset.source_path.rstrip('/')), 'label', scene_name)
    for im in [f for f in os.listdir(data_path) if f.endswith('.jpg')]:
        image_name = im.split('.')[0]
        anno = json.load(open(os.path.join(data_path, image_name+'.json')))
        cam = next((c for c in scene.getTrainCameras() if c.image_name == image_name), None)
        if cam is None:
            print(f"[WARN] no cam {image_name}"); continue
        h, w = cam.image_height, cam.image_width
        out_dir = os.path.join(path_pred, scene_name, image_name); os.makedirs(out_dir, exist_ok=True)
        for prompt in set(o['category'] for o in anno['objects']):
            vlm.encode_text(prompt)
            sim = [vlm.compute_similarity(f) for f in snag.feat]
            pv = snag.get_related_gaussian(sim, topk=3, level=[2,3])
            gaussians._semantics = pv.expand(-1,20).cuda()
            soft = render(cam, gaussians, pipe, background)["semantics"].reshape(20,-1)[0].reshape(h,w).clamp(0,1).cpu().numpy()
            gt = np.zeros((h,w), dtype=np.uint8)
            for o in anno['objects']:
                if o['category']==prompt: gt = np.maximum(gt, polygon_to_mask((h,w), o['segmentation']))
            pp = prompt.replace(' ','_')
            cv2.imwrite(os.path.join(out_dir, pp+'.png'), np.round(soft*255).astype(np.uint8))
            cv2.imwrite(os.path.join(out_dir, pp+'_gt.png'), gt*255)
        print(f"[{scene_name}] {image_name} done")

if __name__ == "__main__":
    parser = ArgumentParser(); lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
    parser.add_argument('--path_pred', required=True)
    args = parser.parse_args(sys.argv[1:]); safe_state(True)
    run(lp.extract(args), pp.extract(args), args.path_pred)
    print("render_thgs_single done.")
