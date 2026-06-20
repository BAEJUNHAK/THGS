"""P1-E.VALA V3 — per-prompt IoU regime split for VALA (real code) on a scene.

Uses VALA's own load_image_as_binary/calculate_iou on the rendered silhouettes
vs the GT masks compute_lerf_iou extracted, groups IoU by prompt, then splits
by our thgs_class (easy/phantom/other) and compares to the THGS baseline.
"""
import os, sys, argparse
import numpy as np, pandas as pd
from collections import defaultdict

VALADIR = '/mnt/pilab_nas/projects/THGS/external_methods/VALA'
sys.path.insert(0, VALADIR)
os.environ.setdefault('HOME', '/tmp')
from eval.compute_lerf_iou import load_image_as_binary, calculate_iou

ap = argparse.ArgumentParser()
ap.add_argument('--scene', default='ramen')
ap.add_argument('--mask_thresh', default='0.6')
args = ap.parse_args()
S = args.scene
base = f'{VALADIR}/output/3dgs/lerf_ovs/{S}'
PRED = f'{base}/none/predictions_mask_{args.mask_thresh}/renders_silhouette'
GT = f'{base}/gt'

q_ious = defaultdict(list)
for fr in sorted(os.listdir(GT)):
    gtf = os.path.join(GT, fr)
    if not os.path.isdir(gtf):
        continue
    for f in os.listdir(gtf):
        if not f.endswith('.jpg'):
            continue
        q = os.path.splitext(f)[0]
        predp = os.path.join(PRED, fr, q + '.png')
        if not os.path.exists(predp):
            q_ious[q].append(0.0); continue
        mg = load_image_as_binary(os.path.join(gtf, f))
        mp = load_image_as_binary(predp, is_png=True)
        q_ious[q].append(calculate_iou(mg, mp))
vala = {q: float(np.mean(v)) for q, v in q_ious.items()}

cross = pd.read_csv('output/diagnostics/cross_method_d2_decomposition.csv')
cr = cross[cross.scene == S].set_index('prompt')['thgs_class'].to_dict()
thgs = pd.read_csv('output/diagnostics/p1a_mask_iou.csv')
th = thgs[thgs.scene == S].groupby('prompt')['iou_baseline'].mean().to_dict()

rows = []
for q in sorted(vala):
    cls = cr.get(q, '?')
    rows.append({'prompt': q, 'thgs_class': cls,
                 'vala_iou': round(vala[q], 3),
                 'thgs_base_iou': round(th[q], 3) if q in th else np.nan})
df = pd.DataFrame(rows)
df['regime'] = df.thgs_class.map(lambda c: 'easy' if c == 'easy' else 'phantom' if c == 'phantom' else 'other')
print(df.to_string(index=False))
print(f"\nmatched to thgs_class: {(df.thgs_class!='?').sum()}/{len(df)}")
print(f"\n=== {S} regime split — VALA(real) vs THGS baseline ===")
for reg in ['easy', 'phantom', 'other']:
    g = df[df.regime == reg]
    if len(g):
        print(f"  {reg:8s} n={len(g):2d}  VALA {g.vala_iou.mean():.3f}   THGS_base {g.thgs_base_iou.mean():.3f}   Δ {(g.vala_iou.mean()-g.thgs_base_iou.mean())*100:+.1f}pt")
df.to_csv(f'output/diagnostics/p1e_vala_regime_{S}.csv', index=False)
print(f"\nwrote output/diagnostics/p1e_vala_regime_{S}.csv  | overall VALA mIoU {df.vala_iou.mean():.4f}")
