"""Aggregate seeded fair-comparison runs -> mean±std per method.
Evals each output/render/repro/seedrun/<METHOD>_s<seed>/RUN with single harness
(per-image, t=0.5) and reports mean±std over seeds + paired per-seed Δ.
CPU only."""
import os, json, cv2, numpy as np, collections
from argparse import ArgumentParser
SCENES = ['figurines','ramen','teatime','waldo_kitchen']
def iou(p,g):
    p=p>127; g=g>127; tp=np.sum(p&g); return tp/(tp+np.sum(p&~g)+np.sum(~p&g)+1e-6)
def eval_run(run_dir, gt_root, t=0.5):
    per_scene={}
    for sc in SCENES:
        d=os.path.join(gt_root,sc);
        if not os.path.isdir(d): continue
        frames=[f[:-4] for f in os.listdir(d) if f.endswith('.jpg')]; img=[]
        for fr in frames:
            objs=json.load(open(os.path.join(d,fr+'.json')))['objects']; vals=[]
            for p in sorted(set(o['category'] for o in objs)):
                pp=p.replace(' ','_'); base=os.path.join(run_dir,sc,fr)
                g=cv2.imread(os.path.join(base,pp+'_gt.png'),0)
                if g is None: continue
                s=cv2.imread(os.path.join(base,pp+'.png'),0)
                if s is None: s=np.zeros_like(g)
                vals.append(iou(s,g))  # iou() thresholds >127 internally (t=0.5)
            if vals: img.append(np.mean(vals))
        per_scene[sc]=float(np.mean(img)) if img else float('nan')
    per_scene['overall']=float(np.nanmean([per_scene[s] for s in SCENES]))
    return per_scene
if __name__=="__main__":
    ap=ArgumentParser(); ap.add_argument('--base',default='output/render/repro/seedrun')
    ap.add_argument('--gt',default='data/lerf_ovs/label'); ap.add_argument('--seeds',default='0,1,2')
    a=ap.parse_args(); seeds=[int(x) for x in a.seeds.split(',')]
    res={}  # method -> seed -> per_scene
    for method in ['THGS','RELAGS']:
        res[method]={}
        for s in seeds:
            rd=os.path.join(a.base,f'{method}_s{s}','RUN')
            if os.path.isdir(rd): res[method][s]=eval_run(rd,a.gt)
            else: print(f"[miss] {rd}")
    cols=SCENES+['overall']
    print("\n=== seeded fair comparison (same THGS-2DGS, per-image t0.5) ===")
    for method in ['THGS','RELAGS']:
        print(f"\n[{method}]  (seeds {sorted(res[method])})")
        for c in cols:
            vv=[res[method][s][c] for s in res[method]]
            if vv: print(f"  {c:14} mean {np.mean(vv):.4f}  std {np.std(vv):.4f}  vals {[round(x,4) for x in vv]}")
    # paired delta per seed
    common=sorted(set(res['THGS'])&set(res['RELAGS']))
    print("\n[ReLaGS - THGS] paired Δ per seed (overall):")
    for s in common:
        d=res['RELAGS'][s]['overall']-res['THGS'][s]['overall']
        print(f"  seed {s}: ReLaGS {res['RELAGS'][s]['overall']:.4f} - THGS {res['THGS'][s]['overall']:.4f} = {d:+.4f}")
    if common:
        ds=[res['RELAGS'][s]['overall']-res['THGS'][s]['overall'] for s in common]
        print(f"  Δ mean {np.mean(ds):+.4f}  std {np.std(ds):.4f}")
