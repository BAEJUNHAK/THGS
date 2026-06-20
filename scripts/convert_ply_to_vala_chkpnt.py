"""Convert a plain-RGB 3DGS point_cloud.ply into a VALA-native chkpnt30000.pth.

Needed because VALA's feature_extractor restores the gaussian state from
chkpnt30000.pth (12-tuple via restore_rgb), but LangSplat's released chkpnt has
an incompatible (longer) tuple. We load the dense .ply into VALA's GaussianModel,
run training_setup to build a matching optimizer, then capture_rgb() and save.
"""
import os, sys, argparse
VALA = '/mnt/pilab_nas/projects/THGS/external_methods/VALA'
os.chdir(VALA)
sys.path.insert(0, VALA)
os.environ.setdefault('HOME', '/tmp')
import torch
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams
from scene.gaussian_model import GaussianModel

ap = argparse.ArgumentParser()
ap.add_argument('--ply', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--iter', type=int, default=30000)
a = ap.parse_args()

p = ArgumentParser(); ModelParams(p); op = OptimizationParams(p); PipelineParams(p)
opt = op.extract(p.parse_args([]))

g = GaussianModel(3)
g.load_ply(a.ply)
# force all gaussian params onto cuda (load_ply may leave them on cpu, which
# breaks the cuda-indexed prune_points during feature extraction)
import torch.nn as nn
for attr in ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']:
    t = getattr(g, attr)
    setattr(g, attr, nn.Parameter(t.detach().cuda().requires_grad_(True)))
g.max_radii2D = torch.zeros(g._xyz.shape[0], device='cuda')
g.spatial_lr_scale = 5.0          # nonzero so training_setup builds lr schedule; value irrelevant (no training)
g.training_setup(opt)
n = g._xyz.shape[0]
if os.path.islink(a.out) or os.path.exists(a.out):
    os.remove(a.out)              # drop any existing symlink (don't write through it)
torch.save((g.capture_rgb(), a.iter), a.out)
print(f"saved VALA chkpnt: {n} gaussians -> {a.out}")
