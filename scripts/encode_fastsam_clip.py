"""
FastSAM 4-scale + OpenCLIP ViT-B/16 224-crop encoder.
Drop-in replacement for scripts/image_encoding.py with mask proposer changed to FastSAM.

Key contract (must match image_encoding.py output):
- Saves <frame>_s.npy : (4, H, W) int32 seg_map (cumulative-offset indices, -1 = bg)
- Saves <frame>_f.npy : (M_total, 512) fp16 L2-normalized OpenCLIP features
- 4 channels in seg_map correspond to 4 confidence thresholds [0.4, 0.5, 0.6, 0.7]
  (matches A's [default, s, m, l] semantic: lower conf -> more/smaller masks)
- Same mask_nms(iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
- Same mask2segmap (bbox crop -> square pad -> 224 resize -> CLIP encode -> L2-norm)

Additionally writes per-image timing log to: <save_folder>/../timing_fastsam.csv
columns: frame,t_mask_ms,t_clip_ms,t_total_ms,n_masks,peak_mem_mb,resolution
"""
import os
import csv
import time
import random
import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Tuple, Type

import numpy as np
import torch
import torchvision
from torch import nn
from tqdm import tqdm
import cv2

try:
    import open_clip
except ImportError:
    raise ImportError("open_clip is not installed: pip install open-clip-torch")

# FastSAM via ultralytics (recommended; supports CUDA + simple API)
try:
    from ultralytics import FastSAM
    _HAS_FASTSAM = True
except ImportError:
    _HAS_FASTSAM = False


# ---- A's NMS / utils -- copied verbatim from image_encoding.py to keep parity ----

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation'] == 0] = np.array([0, 0, 0], dtype=np.uint8)
    x, y, w, h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img


def pad_img(img):
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad


def filter_masks(keep, masks_result):
    keep = keep.int().cpu().numpy()
    return [m for i, m in enumerate(masks_result) if i in keep]


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """A's mask_nms verbatim."""
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union if union > 0 else torch.tensor(0.0)
            iou_matrix[i, j] = iou
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    if keep_conf.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_conf[index] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_inner_u[index] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(min(3, len(scores))).indices
        keep_inner_l[index] = True
    keep = keep * keep_conf * keep_inner_u * keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def masks_update(*args, **kwargs):
    masks_new = ()
    for masks_lvl in args:
        if len(masks_lvl) == 0:
            masks_new += (masks_lvl,)
            continue
        seg_pred = torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0)).cuda()
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0)).cuda()
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0)).cuda()
        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter_masks(keep_mask_nms, masks_lvl)
        masks_new += (masks_lvl,)
    return masks_new


# ---- OpenCLIP wrapper (same as A) ----

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,
            pretrained=self.config.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


# ---- FastSAM wrapper: 4 conf thresholds -> 4 scale-equivalent levels ----

CONF_LEVELS = [0.4, 0.5, 0.6, 0.7]   # default <- s <- m <- l (low conf -> many small masks)
LEVEL_NAMES = ['default', 's', 'm', 'l']


def fastsam_to_sam_format(results_obj, image_hw):
    """
    Convert ultralytics FastSAM results to SAM-style mask dicts:
    [{segmentation: HxW bool, bbox: [x,y,w,h], predicted_iou: float, stability_score: float, area: int}]
    """
    H, W = image_hw
    out = []
    if results_obj is None or len(results_obj) == 0:
        return out
    r = results_obj[0]
    if r.masks is None or r.masks.data is None:
        return out
    masks_data = r.masks.data.cpu().numpy()  # (n, h', w') already resized to image
    if r.boxes is not None and r.boxes.conf is not None:
        confs = r.boxes.conf.cpu().numpy()
    else:
        confs = np.ones(len(masks_data), dtype=np.float32)

    for i in range(len(masks_data)):
        m = masks_data[i].astype(bool)
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        if m.sum() < 50:  # area filter (analogous to A's min_mask_region_area=100)
            continue
        ys, xs = np.where(m)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        out.append({
            'segmentation': m,
            'bbox': [int(x0), int(y0), int(bw), int(bh)],
            'predicted_iou': float(confs[i]),     # FastSAM uses detection conf as proxy
            'stability_score': float(confs[i]),    # same proxy (no separate stability)
            'area': int(m.sum()),
        })
    return out


def fastsam_4scale(model, image_bgr):
    """Run FastSAM at 4 conf thresholds. Returns dict {'default','s','m','l'} -> list of mask dicts."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    imgsz = max(H, W) if max(H, W) <= 1024 else 1024
    out = {}
    for name, conf in zip(LEVEL_NAMES, CONF_LEVELS):
        results = model(
            image_rgb,
            device='cuda',
            retina_masks=True,
            imgsz=imgsz,
            conf=conf,
            iou=0.7,
            verbose=False,
        )
        out[name] = fastsam_to_sam_format(results, (H, W))
    return out


# ---- Per-image encoding pipeline ----

def mask2segmap(masks, image_bgr_hwc, model_clip):
    """Identical to A's mask2segmap: crop -> pad -> 224 resize -> CLIP -> L2-norm."""
    seg_img_list = []
    seg_map = -np.ones(image_bgr_hwc.shape[:2], dtype=np.int32)
    for i, m in enumerate(masks):
        seg_img = get_seg_img(m, image_bgr_hwc)
        if seg_img.shape[0] == 0 or seg_img.shape[1] == 0:
            continue
        pad_seg = cv2.resize(pad_img(seg_img), (224, 224))
        seg_img_list.append(pad_seg)
        seg_map[m['segmentation']] = len(seg_img_list) - 1  # ensures consecutive ids

    if len(seg_img_list) == 0:
        return torch.zeros((0, 512), dtype=torch.float16), seg_map

    seg_imgs = np.stack(seg_img_list, axis=0)  # (B, 224, 224, 3) BGR
    seg_imgs = torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0
    seg_imgs = seg_imgs.to('cuda')
    with torch.no_grad():
        emb = model_clip.encode_image(seg_imgs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.detach().cpu().half(), seg_map


def encode_one_image(image_bgr, fastsam_model, model_clip):
    """
    Returns:
        feats_per_level: dict {level_name: tensor (n, 512)}
        seg_per_level:   dict {level_name: ndarray (H, W) int32}
        timing: dict
    """
    H, W = image_bgr.shape[:2]
    timing = {}

    # ---- Mask proposer (FastSAM 4-scale) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    masks_per_level = fastsam_4scale(fastsam_model, image_bgr)
    torch.cuda.synchronize()
    timing['t_mask_ms'] = (time.perf_counter() - t0) * 1000

    # NMS (same as A)
    masks_d, masks_s, masks_m, masks_l = masks_per_level['default'], masks_per_level['s'], masks_per_level['m'], masks_per_level['l']
    masks_d, masks_s, masks_m, masks_l = masks_update(
        masks_d, masks_s, masks_m, masks_l,
        iou_thr=0.8, score_thr=0.7, inner_thr=0.5,
    )

    # ---- CLIP per mask (224-crop, A-identical) ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    feats, seg_maps = {}, {}
    for name, masks in zip(LEVEL_NAMES, [masks_d, masks_s, masks_m, masks_l]):
        if len(masks) == 0:
            feats[name] = torch.zeros((0, 512), dtype=torch.float16)
            seg_maps[name] = -np.ones((H, W), dtype=np.int32)
            continue
        emb, sm = mask2segmap(masks, image_bgr, model_clip)
        feats[name] = emb
        seg_maps[name] = sm
    torch.cuda.synchronize()
    timing['t_clip_ms'] = (time.perf_counter() - t0) * 1000
    timing['t_total_ms'] = timing['t_mask_ms'] + timing['t_clip_ms']
    timing['n_masks'] = sum(f.shape[0] for f in feats.values())

    return feats, seg_maps, timing


def save_one(save_path, feats_per_level, seg_per_level):
    """Pack 4 levels into single (4,H,W) seg_map + concatenated (M,512) feat using cumulative offsets."""
    embed_size = 512
    H, W = next(iter(seg_per_level.values())).shape
    lengths = [feats_per_level[n].shape[0] for n in LEVEL_NAMES]
    total = sum(lengths)

    feat_concat = torch.zeros((total, embed_size), dtype=torch.float16) if total > 0 else torch.zeros((1, embed_size), dtype=torch.float16)
    seg_stack = np.zeros((4, H, W), dtype=np.int32)

    cum = 0
    for j, name in enumerate(LEVEL_NAMES):
        sm = seg_per_level[name].copy()
        f = feats_per_level[name]
        n = lengths[j]
        if n > 0:
            feat_concat[cum:cum+n] = f
            if j > 0:
                # Match A's cumulative offset: lengths_cumsum[j-1] = sum(lengths[0..j-1]) = cum here
                sm[sm != -1] += cum
        seg_stack[j] = sm
        cum += n

    np.save(save_path + '_s.npy', seg_stack)
    np.save(save_path + '_f.npy', feat_concat[:max(total, 1)].numpy())


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def main():
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True,
                        help='e.g. data/lerf-ovs/ramen')
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--fastsam_ckpt', type=str, default='ckpts/FastSAM-x.pt',
                        help='FastSAM-x.pt (~138MB) or FastSAM-s.pt (~24MB)')
    parser.add_argument('--feat_subdir', type=str, default='language_features_fastsam',
                        help='subdir under source_path to write features')
    parser.add_argument('--timing_csv', type=str, default=None,
                        help='timing CSV path (default: <source_path>/<feat_subdir>/../timing_fastsam.csv)')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    if not _HAS_FASTSAM:
        raise RuntimeError("ultralytics not installed: pip install ultralytics")

    img_folder = os.path.join(args.source_path, 'images')
    save_folder = os.path.join(args.source_path, args.feat_subdir)
    os.makedirs(save_folder, exist_ok=True)

    if args.timing_csv is None:
        args.timing_csv = os.path.join(args.source_path, 'timing_fastsam.csv')

    data_listi = sorted(os.listdir(img_folder))

    # Load CLIP
    model_clip = OpenCLIPNetwork(OpenCLIPNetworkConfig())
    # Load FastSAM
    fastsam = FastSAM(args.fastsam_ckpt)

    # CSV header
    write_header = not os.path.exists(args.timing_csv)
    timing_fp = open(args.timing_csv, 'a', newline='')
    timing_writer = csv.writer(timing_fp)
    if write_header:
        timing_writer.writerow(['frame', 't_mask_ms', 't_clip_ms', 't_total_ms',
                                'n_masks', 'peak_mem_mb', 'resolution_w', 'resolution_h'])

    WARNED = False
    for data_path in tqdm(data_listi, desc="FastSAM+CLIP encoding"):
        frame_name = os.path.splitext(data_path)[0]
        out_s = os.path.join(save_folder, frame_name + '_s.npy')
        out_f = os.path.join(save_folder, frame_name + '_f.npy')
        if os.path.exists(out_s) and os.path.exists(out_f):
            continue

        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"skip (cannot read): {image_path}")
            continue
        orig_w, orig_h = image.shape[1], image.shape[0]

        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[INFO] >1080P, rescaling to 1080P.")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
        scale = float(global_down)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        image = cv2.resize(image, resolution)

        torch.cuda.reset_peak_memory_stats()
        feats, seg_maps, timing = encode_one_image(image, fastsam, model_clip)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        save_path = os.path.join(save_folder, frame_name)
        save_one(save_path, feats, seg_maps)

        timing_writer.writerow([
            frame_name,
            f"{timing['t_mask_ms']:.2f}",
            f"{timing['t_clip_ms']:.2f}",
            f"{timing['t_total_ms']:.2f}",
            timing['n_masks'],
            f"{peak_mb:.1f}",
            resolution[0], resolution[1],
        ])
        timing_fp.flush()

    timing_fp.close()
    print(f"\nDone. Saved features to {save_folder}, timing to {args.timing_csv}")


if __name__ == '__main__':
    main()
