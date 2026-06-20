"""
Phase 1 — CPU eval over the soft masks rendered by render_thgs_ablate.py.
Sweeps threshold x aggregation for each (level,topk) config and compares to paper Table 1.
No GPU. Reads <pred>/<config>/<scene>/<frame>/<prompt>.png (soft uint8) + _gt.png.

Aggregations:
  per-image  : prompt->mean over prompts in frame->mean over frames->mean over scenes (eval_seg.py)
  per-prompt : prompt = mean over its frames -> mean over prompts (per scene) -> mean over scenes
  flat       : flat mean over all (prompt,frame) instances (LangSplat-style)
"""
import os, json, cv2, csv, numpy as np, collections
from argparse import ArgumentParser

PAPER = {'figurines': .5730, 'ramen': .4346, 'teatime': .6833, 'waldo_kitchen': .5065, 'overall': .5494}
SCENES = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']

def iou(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = np.sum(pred & gt); fp = np.sum(pred & ~gt); fn = np.sum(~pred & gt)
    return tp / (tp + fp + fn + 1e-6)

def eval_config(cfg_dir, gt_label_root, thresholds):
    # returns dict[thresh][agg][scene] and overall
    out = {t: {'per_image': {}, 'per_prompt': {}, 'flat': {}} for t in thresholds}
    flat_pool = {t: [] for t in thresholds}
    for sc in SCENES:
        d = os.path.join(gt_label_root, sc)
        if not os.path.isdir(d):
            continue
        frames = [f[:-4] for f in os.listdir(d) if f.endswith('.jpg')]
        # per-thresh accumulators
        per_image = {t: [] for t in thresholds}
        prompt_ious = {t: collections.defaultdict(list) for t in thresholds}
        for fr in frames:
            objs = json.load(open(os.path.join(d, fr + '.json')))['objects']
            ps = sorted(set(o['category'] for o in objs))
            frame_vals = {t: [] for t in thresholds}
            for p in ps:
                pp = p.replace(' ', '_')
                base = os.path.join(cfg_dir, sc, fr)
                gtp = os.path.join(base, pp + '_gt.png')
                sfp = os.path.join(base, pp + '.png')
                gt = cv2.imread(gtp, 0)
                if gt is None:
                    continue
                soft = cv2.imread(sfp, 0)
                if soft is None:
                    soft = np.zeros_like(gt)
                for t in thresholds:
                    v = iou(soft > (t * 255), gt > 128)
                    frame_vals[t].append(v); prompt_ious[t][p].append(v); flat_pool[t].append(v)
            for t in thresholds:
                if frame_vals[t]:
                    per_image[t].append(np.mean(frame_vals[t]))
        for t in thresholds:
            out[t]['per_image'][sc] = float(np.mean(per_image[t])) if per_image[t] else 0.0
            out[t]['per_prompt'][sc] = (float(np.mean([np.mean(v) for v in prompt_ious[t].values()]))
                                        if prompt_ious[t] else 0.0)
    for t in thresholds:
        for agg in ('per_image', 'per_prompt'):
            vals = [out[t][agg][s] for s in SCENES if s in out[t][agg]]
            out[t][agg]['overall'] = float(np.mean(vals)) if vals else 0.0
        out[t]['flat']['overall'] = float(np.mean(flat_pool[t])) if flat_pool[t] else 0.0
    return out

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('--pred', default='output/render/repro/thgs')
    ap.add_argument('--gt', default='data/lerf_ovs/label')
    ap.add_argument('--thresholds', default='0.3,0.4,0.5,0.6')
    ap.add_argument('--csv', default='output/diagnostics/repro_thgs_ablation.csv')
    args = ap.parse_args()
    thr = [float(x) for x in args.thresholds.split(',')]
    configs = sorted([c for c in os.listdir(args.pred) if os.path.isdir(os.path.join(args.pred, c))])
    rows = []
    print(f"paper THGS: fig {PAPER['figurines']:.4f} ramen {PAPER['ramen']:.4f} "
          f"teatime {PAPER['teatime']:.4f} waldo {PAPER['waldo_kitchen']:.4f} overall {PAPER['overall']:.4f}\n")
    for cfg in configs:
        res = eval_config(os.path.join(args.pred, cfg), args.gt, thr)
        for t in thr:
            for agg in ('per_image', 'per_prompt', 'flat'):
                r = res[t][agg]
                ov = r.get('overall', 0.0)
                line = {'config': cfg, 'thresh': t, 'agg': agg, 'overall': round(ov, 4),
                        'd_paper': round(ov - PAPER['overall'], 4)}
                for s in SCENES:
                    line[s] = round(r.get(s, float('nan')), 4) if agg != 'flat' else ''
                rows.append(line)
                tag = '  <== released-default-baseline' if (cfg == 'L23_k3' and abs(t-0.5) < 1e-6 and agg == 'per_image') else ''
                print(f"{cfg:8} t={t} {agg:10} overall={ov:.4f} (Δpaper {ov-PAPER['overall']:+.4f}){tag}")
        print()
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['config', 'thresh', 'agg', 'overall', 'd_paper'] + SCENES)
        wr.writeheader(); wr.writerows(rows)
    print(f"wrote {args.csv}")
