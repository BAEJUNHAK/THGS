"""
Text-encoder-agnostic 2D source quality evaluator.

Goal: 텍스트 인코더가 다른 arm (A/B vs C/D 등) 도 같은 무대에 올릴 수 있도록,
"피처가 인스턴스를 얼마나 잘 분리하는가" 만 측정한다. 텍스트 쿼리는 사용하지 않는다.

Inputs (any arm with the same A-format <frame>_s.npy + <frame>_f.npy):
    s_map: (4, H, W) int — 4-level mask ids (-1 = bg)
    feat:  (M, D)    fp16/fp32 — per-mask feature

Per-frame procedure:
  1) level=1 (s) 의 mask id 별로 pixel set 과 feature vector 수집
  2) GT polygon mask (per category) 로드 → 각 mask 에 best-IoU GT category 라벨 할당
     (IoU < min_iou 인 mask 는 'background' 로 분류)
  3) GT 라벨이 붙은 mask 들에 대해 cosine 거리로 agglomerative clustering
     (K = 해당 frame 내 GT category 수)
  4) ARI / NMI / cluster purity / Hungarian-matched mIoU 계산
  5) (옵션) pixel feature PCA→RGB 그림 + mask feature t-SNE 산점도 저장

Output: per-frame metric CSV (+ optional PNG 디렉토리)

Usage:
    python scripts/eval_cluster_quality.py \
        --feat_dir data/lerf-ovs/ramen/language_features \
        --gt_dir   data/lerf-ovs/label/ramen \
        --out_csv  output/exp_2d_source/cluster_A.csv \
        --viz_dir  output/exp_2d_source/cluster_viz_A \
        --arm A --level 1 \
        --eval_frames frame_00006 frame_00024 ...
"""
import os
import json
import argparse
import csv
from typing import Dict, List, Tuple

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment


# ---- IO helpers (eval_2d_source 와 동일 포맷) ----

def polygon_to_mask(h, w, points):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
    return mask


def load_gt_masks_per_category(gt_dir, frame, target_hw=None):
    jpath = os.path.join(gt_dir, frame + '.json')
    if not os.path.exists(jpath):
        return {}, None
    anno = json.load(open(jpath))
    W, H = anno['info']['width'], anno['info']['height']
    masks = {}
    for obj in anno['objects']:
        c = obj['category']
        m = polygon_to_mask(H, W, obj['segmentation'])
        if c in masks:
            masks[c] = np.maximum(masks[c], m)
        else:
            masks[c] = m
    if target_hw is not None and (H, W) != target_hw:
        H2, W2 = target_hw
        masks = {c: cv2.resize(m, (W2, H2), interpolation=cv2.INTER_NEAREST)
                 for c, m in masks.items()}
        H, W = H2, W2
    return masks, (H, W)


def load_features(feat_dir, frame):
    s_path = os.path.join(feat_dir, frame + '_s.npy')
    f_path = os.path.join(feat_dir, frame + '_f.npy')
    if not (os.path.exists(s_path) and os.path.exists(f_path)):
        return None, None
    seg_map = np.load(s_path).astype(np.int64)
    feat = np.load(f_path).astype(np.float32)
    return seg_map, feat


def load_pixel_feature(feat_dir, frame):
    """Load (Hp, Wp, D) pixel feature map saved by encode_openseg_fastsam."""
    p_path = os.path.join(feat_dir, frame + '_p.npy')
    if not os.path.exists(p_path):
        return None
    return np.load(p_path).astype(np.float32)


# ---- Mask → GT label assignment ----

def assign_gt_labels(level_seg_hw: np.ndarray,
                     gt_masks_by_cat: Dict[str, np.ndarray],
                     min_iou: float = 0.2):
    """
    각 mask id 에 대해 best-IoU GT category 라벨을 할당. min_iou 미만이면 'background'.

    Returns:
        mask_ids       : list of int — level_seg_hw 에 등장하는 unique mask id (>=0)
        labels         : list of str — 같은 길이, 카테고리명 또는 'background'
        best_ious      : list of float
    """
    H, W = level_seg_hw.shape
    unique_ids = np.unique(level_seg_hw)
    unique_ids = unique_ids[unique_ids >= 0].tolist()
    cats = list(gt_masks_by_cat.keys())
    if len(cats) == 0:
        return unique_ids, ['background'] * len(unique_ids), [0.0] * len(unique_ids)

    labels, best_ious = [], []
    for mid in unique_ids:
        m = (level_seg_hw == mid)
        m_area = m.sum()
        if m_area == 0:
            labels.append('background'); best_ious.append(0.0); continue
        best_iou, best_cat = 0.0, 'background'
        for c in cats:
            gt = gt_masks_by_cat[c].astype(bool)
            inter = np.logical_and(m, gt).sum()
            union = np.logical_or(m, gt).sum()
            if union == 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou = float(iou); best_cat = c
        if best_iou < min_iou:
            best_cat = 'background'
        labels.append(best_cat); best_ious.append(best_iou)
    return unique_ids, labels, best_ious


# ---- Clustering metrics ----

def cluster_purity(labels_true, labels_pred):
    """각 cluster 가 한 GT class 에 얼마나 집중되었는가 (가중 평균)."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    total = len(labels_true)
    if total == 0:
        return 0.0
    purity = 0
    for c in np.unique(labels_pred):
        idx = labels_pred == c
        if idx.sum() == 0:
            continue
        most_common = np.bincount(labels_true[idx]).max()
        purity += most_common
    return float(purity / total)


def hungarian_mask_miou(mask_ids: List[int],
                        labels_pred: np.ndarray,
                        labels_true_str: List[str],
                        level_seg_hw: np.ndarray,
                        gt_masks_by_cat: Dict[str, np.ndarray]):
    """
    각 cluster -> 픽셀 union mask, 각 GT category -> 픽셀 union mask.
    Hungarian 매칭으로 cluster ↔ GT category 1:1 (or partial) 최적 매칭 후 평균 IoU.
    """
    H, W = level_seg_hw.shape
    cluster_ids = sorted(set(int(x) for x in labels_pred))
    cluster_pixmasks = []
    for c in cluster_ids:
        union = np.zeros((H, W), dtype=bool)
        for mid, lbl_pred in zip(mask_ids, labels_pred):
            if int(lbl_pred) == c:
                union |= (level_seg_hw == mid)
        cluster_pixmasks.append(union)

    gt_cats = list(gt_masks_by_cat.keys())
    if len(gt_cats) == 0 or len(cluster_pixmasks) == 0:
        return 0.0, []

    # cost matrix: -IoU (Hungarian minimizes)
    K, G = len(cluster_pixmasks), len(gt_cats)
    cost = np.zeros((K, G), dtype=np.float32)
    for i, cm in enumerate(cluster_pixmasks):
        for j, gc in enumerate(gt_cats):
            gt = gt_masks_by_cat[gc].astype(bool)
            inter = np.logical_and(cm, gt).sum()
            union = np.logical_or(cm, gt).sum()
            iou = inter / union if union > 0 else 0.0
            cost[i, j] = -iou

    rows, cols = linear_sum_assignment(cost)
    matched_ious = [-cost[r, c] for r, c in zip(rows, cols)]
    miou = float(np.mean(matched_ious)) if matched_ious else 0.0
    matches = [(cluster_ids[r], gt_cats[c], float(-cost[r, c]))
               for r, c in zip(rows, cols)]
    return miou, matches


# ---- Visualization ----

def viz_raw_pixel_pca(pixfeat_HpWpD: np.ndarray, out_path: str,
                      target_hw=None):
    """C-arm 등 진짜 pixel feature map (Hp,Wp,D) 의 PCA(3) → RGB."""
    Hp, Wp, D = pixfeat_HpWpD.shape
    flat = pixfeat_HpWpD.reshape(-1, D)
    flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9)
    pca = PCA(n_components=3)
    proj = pca.fit_transform(flat)
    proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-9)
    rgb = proj.reshape(Hp, Wp, 3)
    if target_hw is not None and (Hp, Wp) != target_hw:
        rgb = cv2.resize(rgb, (target_hw[1], target_hw[0]),
                         interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_path, (rgb[..., ::-1] * 255).astype(np.uint8))
    return True


def viz_pixel_feature_pca(level_seg_hw: np.ndarray, feat: np.ndarray, out_path: str):
    """level_seg_hw 의 각 픽셀에 해당 mask 의 feature 를 채워 (H,W,D) 만든 뒤 PCA(3)→RGB 저장."""
    H, W = level_seg_hw.shape
    unique_ids = np.unique(level_seg_hw)
    unique_ids = unique_ids[unique_ids >= 0]
    if len(unique_ids) == 0:
        return False
    valid_ids = unique_ids[unique_ids < len(feat)]
    if len(valid_ids) == 0:
        return False
    sub_feat = feat[valid_ids]                     # (k, D)
    n = sub_feat.shape[0]
    if n < 3:
        return False
    pca = PCA(n_components=3)
    proj = pca.fit_transform(sub_feat)             # (k, 3)
    proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-9)
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    id_to_color = {int(mid): proj[i] for i, mid in enumerate(valid_ids)}
    for mid, color in id_to_color.items():
        rgb[level_seg_hw == mid] = color
    cv2.imwrite(out_path, (rgb[..., ::-1] * 255).astype(np.uint8))  # BGR
    return True


def viz_mask_tsne(feat: np.ndarray, mask_ids: List[int], labels_str: List[str],
                  out_path: str):
    """mask feature 를 PCA(2) (또는 t-SNE 가 있으면 t-SNE) 로 2D 산점도, GT 카테고리로 색칠."""
    valid_ids = [m for m in mask_ids if 0 <= m < len(feat)]
    valid_idx = [i for i, m in enumerate(mask_ids) if 0 <= m < len(feat)]
    if len(valid_idx) < 3:
        return False
    sub_feat = feat[np.asarray(valid_ids)]
    sub_lbl = [labels_str[i] for i in valid_idx]
    try:
        from sklearn.manifold import TSNE
        if len(sub_feat) >= 5:
            xy = TSNE(n_components=2, perplexity=min(5, len(sub_feat) - 1),
                      init='pca', random_state=0).fit_transform(sub_feat)
        else:
            xy = PCA(n_components=2).fit_transform(sub_feat)
    except Exception:
        xy = PCA(n_components=2).fit_transform(sub_feat)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cats = sorted(set(sub_lbl))
    cmap = plt.cm.get_cmap('tab20', len(cats))
    fig, ax = plt.subplots(figsize=(7, 6))
    for ci, c in enumerate(cats):
        idx = [i for i, l in enumerate(sub_lbl) if l == c]
        ax.scatter(xy[idx, 0], xy[idx, 1], s=40, c=[cmap(ci)],
                   label=c, edgecolors='k', linewidths=0.4)
    ax.legend(fontsize=7, loc='best', markerscale=0.7)
    ax.set_title(f'mask feature 2D (n={len(sub_feat)})')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


# ---- Main loops ----

def evaluate_mask(feat_dir, gt_dir, out_csv, arm, level=1,
                  eval_frames=None, min_iou=0.2, viz_dir=None,
                  include_background=False):
    if eval_frames is None:
        eval_frames = sorted([os.path.splitext(f)[0] for f in os.listdir(gt_dir)
                              if f.endswith('.json')])
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)

    rows = []
    for frame in tqdm(eval_frames, desc=f"cluster[{arm}]"):
        seg_map, feat = load_features(feat_dir, frame)
        if seg_map is None:
            print(f"skip {frame}: no features"); continue
        target_hw = seg_map.shape[1:]
        gt_masks, _ = load_gt_masks_per_category(gt_dir, frame, target_hw=target_hw)
        if len(gt_masks) == 0:
            print(f"skip {frame}: no GT"); continue

        level_seg = seg_map[level]
        mask_ids, labels_str, best_ious = assign_gt_labels(
            level_seg, gt_masks, min_iou=min_iou)

        if include_background:
            sel_idx = list(range(len(mask_ids)))
        else:
            sel_idx = [i for i, l in enumerate(labels_str) if l != 'background']
        if len(sel_idx) < 2:
            print(f"skip {frame}: <2 valid masks"); continue

        sel_mask_ids = [mask_ids[i] for i in sel_idx]
        sel_labels = [labels_str[i] for i in sel_idx]
        cats_unique = sorted(set(sel_labels))
        cat_to_int = {c: i for i, c in enumerate(cats_unique)}
        true_int = np.array([cat_to_int[l] for l in sel_labels])

        valid_idx = [i for i, m in enumerate(sel_mask_ids) if 0 <= m < len(feat)]
        if len(valid_idx) < 2:
            print(f"skip {frame}: feat range mismatch"); continue
        sub_mask_ids = [sel_mask_ids[i] for i in valid_idx]
        sub_true = true_int[valid_idx]
        sub_labels = [sel_labels[i] for i in valid_idx]
        sub_feat = feat[np.asarray(sub_mask_ids)]
        sub_feat = sub_feat / (np.linalg.norm(sub_feat, axis=1, keepdims=True) + 1e-9)

        K = max(2, len(cats_unique))
        if len(sub_feat) < K:
            K = len(sub_feat)
        try:
            cluster = AgglomerativeClustering(
                n_clusters=K, metric='cosine', linkage='average')
            pred = cluster.fit_predict(sub_feat)
        except TypeError:
            # older sklearn: 'affinity' instead of 'metric'
            cluster = AgglomerativeClustering(
                n_clusters=K, affinity='cosine', linkage='average')
            pred = cluster.fit_predict(sub_feat)

        ari = float(adjusted_rand_score(sub_true, pred))
        nmi = float(normalized_mutual_info_score(sub_true, pred))
        purity = cluster_purity(sub_true, pred)
        h_miou, matches = hungarian_mask_miou(
            sub_mask_ids, pred, sub_labels, level_seg, gt_masks)

        rows.append(dict(
            arm=arm, frame=frame, level=level,
            n_masks_total=len(mask_ids),
            n_masks_labeled=len(sub_mask_ids),
            n_gt_categories=len(cats_unique),
            n_clusters=int(K),
            ARI=ari, NMI=nmi, purity=purity, hung_mIoU=h_miou,
            mean_assignment_iou=float(np.mean([best_ious[i] for i in sel_idx if i in valid_idx])
                                      if valid_idx else 0.0),
        ))

        if viz_dir:
            # raw pixel feature 가 있으면 우선 사용 (C arm), 없으면 mask-broadcast
            raw_p = load_pixel_feature(feat_dir, frame)
            if raw_p is not None:
                viz_raw_pixel_pca(raw_p,
                                  os.path.join(viz_dir, f"{frame}_pixfeat_pca.png"),
                                  target_hw=level_seg.shape)
            else:
                viz_pixel_feature_pca(level_seg, feat,
                                      os.path.join(viz_dir, f"{frame}_pixfeat_pca.png"))
            viz_mask_tsne(feat, sub_mask_ids, sub_labels,
                          os.path.join(viz_dir, f"{frame}_mask_tsne.png"))

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None
        fieldnames = ['arm', 'frame', 'level', 'n_masks_total', 'n_masks_labeled',
                      'n_gt_categories', 'n_clusters',
                      'ARI', 'NMI', 'purity', 'hung_mIoU', 'mean_assignment_iou']
        with open(out_csv, 'w', newline='') as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {out_csv} ({len(rows)} rows)")
    return rows


def evaluate_pixel(feat_dir, gt_dir, out_csv, arm,
                   eval_frames=None, viz_dir=None,
                   subsample=20000, seed=0):
    """
    Pixel-level clustering on (Hp, Wp, D) feature map saved as <frame>_p.npy.
    GT masks are downsized to (Hp, Wp) and used to:
      - generate per-pixel GT category labels (overlapping pixels: last-wins)
      - drop unlabeled pixels for ARI/NMI
      - compute Hungarian-mIoU on full predicted-cluster pixel maps
    """
    if eval_frames is None:
        eval_frames = sorted([os.path.splitext(f)[0] for f in os.listdir(gt_dir)
                              if f.endswith('.json')])
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    rows = []
    for frame in tqdm(eval_frames, desc=f"pix-cluster[{arm}]"):
        pixfeat = load_pixel_feature(feat_dir, frame)
        if pixfeat is None:
            print(f"skip {frame}: no _p.npy")
            continue
        Hp, Wp, D = pixfeat.shape
        gt_masks, _ = load_gt_masks_per_category(gt_dir, frame, target_hw=(Hp, Wp))
        if len(gt_masks) == 0:
            print(f"skip {frame}: no GT")
            continue

        cats = list(gt_masks.keys())
        cat_to_int = {c: i for i, c in enumerate(cats)}
        gt_pix = -np.ones((Hp, Wp), dtype=np.int32)
        # last-cat-wins on overlap (rare for distinct objects)
        for c, m in gt_masks.items():
            gt_pix[m > 0] = cat_to_int[c]

        feat_flat = pixfeat.reshape(-1, D)
        feat_flat = feat_flat / (np.linalg.norm(feat_flat, axis=1, keepdims=True) + 1e-9)
        labels_flat = gt_pix.flatten()
        lab_idx = np.where(labels_flat >= 0)[0]
        if len(lab_idx) < 10:
            print(f"skip {frame}: <10 labeled pixels"); continue

        if len(lab_idx) > subsample:
            sel = rng.choice(lab_idx, subsample, replace=False)
        else:
            sel = lab_idx
        feat_lab = feat_flat[sel]
        labels_lab = labels_flat[sel]

        K = max(2, len(cats))
        try:
            kmeans = MiniBatchKMeans(n_clusters=K, random_state=seed,
                                     n_init=3, batch_size=2048)
        except TypeError:
            kmeans = MiniBatchKMeans(n_clusters=K, random_state=seed,
                                     batch_size=2048)
        kmeans.fit(feat_lab)
        pred_lab = kmeans.predict(feat_lab)

        ari = float(adjusted_rand_score(labels_lab, pred_lab))
        nmi = float(normalized_mutual_info_score(labels_lab, pred_lab))
        purity = cluster_purity(labels_lab, pred_lab)

        # Hungarian-mIoU at full pixel res
        full_pred = kmeans.predict(feat_flat).reshape(Hp, Wp)
        cost = np.zeros((K, len(cats)), dtype=np.float32)
        for i in range(K):
            cm = (full_pred == i)
            for j, c in enumerate(cats):
                gt = gt_masks[c].astype(bool)
                inter = (cm & gt).sum()
                union = (cm | gt).sum()
                cost[i, j] = -((inter / union) if union > 0 else 0.0)
        rs, cs = linear_sum_assignment(cost)
        ious = [-cost[r, c] for r, c in zip(rs, cs)]
        h_miou = float(np.mean(ious)) if ious else 0.0

        rows.append(dict(arm=arm, frame=frame, mode='pixel',
                         Hp=int(Hp), Wp=int(Wp),
                         n_pixels=int(Hp * Wp),
                         n_pixels_labeled=int(len(lab_idx)),
                         n_pixels_used=int(len(sel)),
                         n_gt_categories=len(cats),
                         n_clusters=int(K),
                         ARI=ari, NMI=nmi, purity=purity, hung_mIoU=h_miou))

        if viz_dir:
            viz_raw_pixel_pca(pixfeat,
                              os.path.join(viz_dir, f"{frame}_rawpixfeat_pca.png"))
            # pixel-cluster colored map
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                cmap = plt.cm.get_cmap('tab20', K)
                col = (cmap(full_pred)[..., :3] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(viz_dir, f"{frame}_pixclusters.png"),
                            col[..., ::-1])
            except Exception:
                pass

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None
        fieldnames = ['arm', 'frame', 'mode', 'Hp', 'Wp', 'n_pixels',
                      'n_pixels_labeled', 'n_pixels_used',
                      'n_gt_categories', 'n_clusters',
                      'ARI', 'NMI', 'purity', 'hung_mIoU']
        with open(out_csv, 'w', newline='') as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {out_csv} ({len(rows)} rows)")
    return rows


def evaluate(feat_dir, gt_dir, out_csv, arm, mode='mask', **kwargs):
    """Dispatcher for backward compat."""
    if mode == 'mask':
        return evaluate_mask(feat_dir, gt_dir, out_csv, arm, **kwargs)
    elif mode == 'pixel':
        return evaluate_pixel(feat_dir, gt_dir, out_csv, arm,
                              eval_frames=kwargs.get('eval_frames'),
                              viz_dir=kwargs.get('viz_dir'),
                              subsample=kwargs.get('subsample', 20000))
    else:
        raise ValueError(f"unknown mode {mode}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--feat_dir', required=True)
    p.add_argument('--gt_dir', required=True)
    p.add_argument('--out_csv', required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--mode', default='mask', choices=['mask', 'pixel'],
                   help='mask: per-mask agglomerative; pixel: per-pixel MiniBatch K-means '
                        '(needs <frame>_p.npy)')
    p.add_argument('--level', type=int, default=1, help='[mask mode] seg_map level (0..3); default 1 = s')
    p.add_argument('--min_iou', type=float, default=0.2,
                   help='[mask mode] mask 가 GT category 로 라벨되기 위한 최소 IoU')
    p.add_argument('--subsample', type=int, default=20000,
                   help='[pixel mode] 라벨된 픽셀 중 ARI/NMI 계산용 최대 표본 수')
    p.add_argument('--viz_dir', default=None, help='PCA-RGB / t-SNE 그림 저장')
    p.add_argument('--eval_frames', nargs='*', default=None)
    p.add_argument('--include_background', action='store_true',
                   help='[mask mode] IoU < min_iou 인 mask 도 별도 cluster 로 포함')
    args = p.parse_args()

    if args.mode == 'mask':
        evaluate_mask(args.feat_dir, args.gt_dir, args.out_csv, args.arm,
                      level=args.level, eval_frames=args.eval_frames,
                      min_iou=args.min_iou, viz_dir=args.viz_dir,
                      include_background=args.include_background)
    else:
        evaluate_pixel(args.feat_dir, args.gt_dir, args.out_csv, args.arm,
                       eval_frames=args.eval_frames, viz_dir=args.viz_dir,
                       subsample=args.subsample)
