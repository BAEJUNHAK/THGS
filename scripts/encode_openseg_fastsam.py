"""
C arm encoder: OpenSeg pixel features + FastSAM mask proposer.

OUTPUT (drop-in compatible with image_encoding.py / encode_fastsam_clip.py):
    {feat_subdir}/{frame}_s.npy  (4, H, W) int32   FastSAM 4-scale mask ids (-1 = bg)
    {feat_subdir}/{frame}_f.npy  (M, 768) fp16     per-mask avg-pooled OpenSeg feature
    {feat_subdir}/{frame}_p.npy  (Hp, Wp, 768) fp16  OpenSeg pixel feature map
                                                   (only saved for frames in --pixel_frames,
                                                    or all if --pixel_all; downsized to max 256)

BACKENDS:
    --backend dense_clip  (DEFAULT, recommended)
                          PyTorch-only. OpenCLIP ViT-B/16 visual tower → patch token grid
                          (21×21×512 @ 336 input). NOT a literal OpenSeg port, but
                          produces dense CLIP-aligned features without TF dependency.
                          Use this when the original Google TF Hub OpenSeg model is
                          unavailable (Google removed `openseg-aligned/0` from TF Hub).
    --backend tfhub       Loads OpenSeg via tensorflow_hub.load(--openseg_url).
                          The default URL is dead; supply --openseg_url to your own
                          mirror or local SavedModel directory.
    --backend stub        Random RGB→768 projection (deterministic). Plumbing test only.

APPROXIMATION NOTES (read before treating numbers as canonical):
  * The mask proposer is FastSAM (B-arm parity). The OpenSeg paper uses its own
    class-agnostic mask head. We deliberately swap that out so A/B/C/D differ ONLY
    in the feature side.
  * Mask feature = simple average pool of OpenSeg pixel embeddings inside the mask.
    The original OpenSeg uses learned grouping; this is a faithful but simpler
    pooling commonly used in followups (LERF, OpenScene, EmbodiedSplat).
  * Text encoder is NOT exercised here. Use scripts/eval_cluster_quality.py for
    text-encoder-agnostic evaluation.

Per-image timing CSV is written to <source_path>/timing_openseg.csv.
"""
import os
import csv
import time
import argparse
from typing import Optional

import numpy as np
import torch
import cv2
from tqdm import tqdm

from encode_fastsam_clip import (  # noqa: E402  - sibling module reuse
    fastsam_4scale, masks_update, LEVEL_NAMES, seed_everything,
    _HAS_FASTSAM,
)
try:
    from ultralytics import FastSAM
except ImportError:
    FastSAM = None


PIXFEAT_MAX_DIM = 256   # save *_p.npy at <=256 longest side to bound disk usage


# ---- OpenSeg backend wrappers ----

class OpenSegStub:
    """Deterministic RGB→768 projection. For plumbing only; numbers are meaningless."""
    def __init__(self, dim=768, seed=0):
        rng = np.random.RandomState(seed)
        self.proj = rng.randn(3, dim).astype(np.float32) / np.sqrt(3.0)
        self.dim = dim

    def __call__(self, rgb_uint8_HW3):
        H, W, _ = rgb_uint8_HW3.shape
        x = rgb_uint8_HW3.astype(np.float32) / 255.0
        feat = x @ self.proj                  # (H, W, 768)
        feat /= (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-9)
        return feat


class OpenSegTFHub:
    """OpenSeg TF Hub wrapper. Tries to load the standard signature; raises with a
    clear message if unavailable. Returns (Hp, Wp, 768) np.float32 feature."""
    def __init__(self, url='https://tfhub.dev/google/openseg-aligned/0'):
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as e:
            raise ImportError(
                "tfhub backend requires tensorflow + tensorflow_hub. "
                "Install: pip install tensorflow tensorflow_hub"
            ) from e
        # Allow GPU memory growth so TF doesn't grab all GPU mem (PyTorch needs some too).
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        self.tf = tf
        print(f"[OpenSegTFHub] loading {url} ...")
        self.model = hub.load(url)
        print("[OpenSegTFHub] loaded.")

    def __call__(self, rgb_uint8_HW3):
        tf = self.tf
        # OpenSeg expects float32 in [0,1] or uint8; the TF Hub variant we target
        # accepts uint8 RGB (B,H,W,3). Output dict has 'image_embedding_feat'.
        x = tf.convert_to_tensor(rgb_uint8_HW3[None, ...])
        sigs = getattr(self.model, 'signatures', None)
        if sigs and 'serving_default' in sigs:
            out = sigs['serving_default'](x)
        else:
            out = self.model(x)
        # Pick the first 4D output if dict has unknown keys
        if isinstance(out, dict):
            for k in ['image_embedding_feat', 'embedding', 'features']:
                if k in out:
                    feat = out[k].numpy()
                    break
            else:
                feat = next(iter(out.values())).numpy()
        else:
            feat = out.numpy()
        if feat.ndim == 4:
            feat = feat[0]
        return feat.astype(np.float32)         # (Hp, Wp, D)


class MaskCLIPDense:
    """PyTorch-only dense CLIP-aligned feature extractor.

    Replaces external OpenSeg with OpenCLIP ViT visual tower's patch tokens.
    NOT a literal OpenSeg port — but it produces dense pixel-aligned features
    in the CLIP space, which is the property we actually need for arm C.

    For ViT-B/16 @ 336 input: returns (21, 21, 512) per image.
    """
    def __init__(self, model_name='ViT-B-16',
                 pretrained='laion2b_s34b_b88k', input_size=336):
        import torch
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, precision='fp16',
        )
        model.eval()
        self.model = model.to('cuda')
        self.visual = self.model.visual
        self.input_size = input_size
        self.torch = torch
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                 device='cuda').view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                device='cuda').view(1, 3, 1, 1)

    def __call__(self, rgb_uint8_HW3):
        torch = self.torch
        F = torch.nn.functional
        x = torch.from_numpy(rgb_uint8_HW3).permute(2, 0, 1).unsqueeze(0)
        x = x.float().to('cuda') / 255.0
        x = F.interpolate(x, size=(self.input_size, self.input_size),
                          mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        x = x.half()
        with torch.no_grad():
            v = self.visual
            tokens = v.conv1(x)                       # (1, D, hp, wp)
            B, D, hp, wp = tokens.shape
            tokens = tokens.flatten(2).transpose(1, 2)  # (1, hp*wp, D)
            cls = v.class_embedding.to(tokens.dtype) \
                   .unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            n_tok = tokens.shape[1]
            pos = v.positional_embedding[:n_tok].to(tokens.dtype)
            tokens = tokens + pos
            tokens = v.ln_pre(tokens)
            # OpenCLIP transformer expects (L, N, D) by default unless batch_first=True
            if hasattr(v.transformer, 'batch_first') and v.transformer.batch_first:
                tokens = v.transformer(tokens)
            else:
                tokens = tokens.permute(1, 0, 2)
                tokens = v.transformer(tokens)
                tokens = tokens.permute(1, 0, 2)
            tokens = v.ln_post(tokens)
            patch_tokens = tokens[:, 1:, :]               # drop CLS
            if v.proj is not None:
                patch_tokens = patch_tokens @ v.proj.to(tokens.dtype)
        feat = patch_tokens[0].reshape(hp, wp, -1).float().cpu().numpy()
        return feat


def make_openseg_backend(backend: str, model_url: Optional[str]):
    if backend == 'dense_clip':
        return MaskCLIPDense()
    elif backend == 'tfhub':
        return OpenSegTFHub(model_url or 'https://tfhub.dev/google/openseg-aligned/0')
    elif backend == 'stub':
        return OpenSegStub(dim=768)
    else:
        raise ValueError(f"unknown backend {backend}")


# ---- Mask average pooling on (H, W, D) feature map ----

def avg_pool_per_mask(feat_HpWpD: np.ndarray, masks_list, target_HW):
    """For each SAM-style mask, return the L2-normalized average of dense pixel
    features inside the mask.

    Instead of upsampling the high-dim feature to image resolution (cv2.resize is
    limited to 4 channels and torch upsampling on (H, W, 768) costs ~1GB), we map
    each mask pixel to its containing feature cell via integer division and average
    feat values weighted by mask-pixel count per cell — equivalent to nearest-neighbor
    upsampling of feat followed by mean over mask pixels.

    Args:
        feat_HpWpD: (Hp, Wp, D) low-res dense feature map (e.g. 10×10×768 from ConvNeXt-L)
        masks_list: SAM-style mask dicts with 'segmentation' (H, W) bool
        target_HW : (H, W) image resolution

    Returns: (n, D) torch.float16 + seg_map (H, W) int32 (consecutive ids, -1 bg).
    """
    H, W = target_HW
    Hp, Wp, D = feat_HpWpD.shape
    feat_flat = feat_HpWpD.reshape(-1, D)            # (Hp*Wp, D)
    seg_map = -np.ones((H, W), dtype=np.int32)
    embs = []
    for m in masks_list:
        seg = m['segmentation']
        if seg.shape != (H, W):
            seg = cv2.resize(seg.astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        if seg.sum() == 0:
            continue
        ys, xs = np.where(seg)
        yp = (ys.astype(np.float32) * Hp / H).astype(np.int32)
        xp = (xs.astype(np.float32) * Wp / W).astype(np.int32)
        np.clip(yp, 0, Hp - 1, out=yp)
        np.clip(xp, 0, Wp - 1, out=xp)
        idx = yp * Wp + xp
        v = feat_flat[idx].mean(axis=0)
        n = np.linalg.norm(v) + 1e-9
        embs.append((v / n).astype(np.float16))
        seg_map[seg] = len(embs) - 1
    if len(embs) == 0:
        return torch.zeros((0, D), dtype=torch.float16), seg_map
    return torch.from_numpy(np.stack(embs, axis=0)), seg_map


# ---- Per-image pipeline ----

def encode_one_image(image_bgr, fastsam_model, openseg):
    """Returns feats_per_level, seg_per_level, pixel_feat (Hp,Wp,D), timing."""
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    timing = {}

    # ---- Mask proposer ----
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    masks_per_level = fastsam_4scale(fastsam_model, image_bgr)
    torch.cuda.synchronize()
    timing['t_mask_ms'] = (time.perf_counter() - t0) * 1000

    masks_d, masks_s, masks_m, masks_l = (masks_per_level[k] for k in LEVEL_NAMES)
    masks_d, masks_s, masks_m, masks_l = masks_update(
        masks_d, masks_s, masks_m, masks_l,
        iou_thr=0.8, score_thr=0.7, inner_thr=0.5,
    )

    # ---- OpenSeg pixel features (H, W, D) ----
    t0 = time.perf_counter()
    pixel_feat = openseg(image_rgb)            # (Hp, Wp, D)
    timing['t_feat_ms'] = (time.perf_counter() - t0) * 1000

    # ---- Mask average pooling ----
    t0 = time.perf_counter()
    feats, seg_maps = {}, {}
    D = pixel_feat.shape[-1]
    for name, masks in zip(LEVEL_NAMES, [masks_d, masks_s, masks_m, masks_l]):
        if len(masks) == 0:
            feats[name] = torch.zeros((0, D), dtype=torch.float16)
            seg_maps[name] = -np.ones((H, W), dtype=np.int32)
            continue
        emb, sm = avg_pool_per_mask(pixel_feat, masks, target_HW=(H, W))
        feats[name] = emb
        seg_maps[name] = sm
    timing['t_pool_ms'] = (time.perf_counter() - t0) * 1000
    timing['t_total_ms'] = (timing['t_mask_ms'] + timing['t_feat_ms']
                            + timing['t_pool_ms'])
    timing['n_masks'] = sum(f.shape[0] for f in feats.values())
    return feats, seg_maps, pixel_feat, timing


def downsize_pixel_feat(feat_HWD, max_dim=PIXFEAT_MAX_DIM):
    H, W, D = feat_HWD.shape
    if max(H, W) <= max_dim:
        return feat_HWD
    scale = max_dim / max(H, W)
    Hn, Wn = int(round(H * scale)), int(round(W * scale))
    return cv2.resize(feat_HWD, (Wn, Hn), interpolation=cv2.INTER_LINEAR)


def save_one(save_path, feats_per_level, seg_per_level, pixel_feat=None,
             save_pixel=False, embed_size=768):
    """A-format packing using cumulative offsets (matches encode_fastsam_clip.save_one)."""
    H, W = next(iter(seg_per_level.values())).shape
    lengths = [feats_per_level[n].shape[0] for n in LEVEL_NAMES]
    total = sum(lengths)

    if total > 0:
        feat_concat = torch.zeros((total, embed_size), dtype=torch.float16)
    else:
        feat_concat = torch.zeros((1, embed_size), dtype=torch.float16)
    seg_stack = np.zeros((4, H, W), dtype=np.int32)

    cum = 0
    for j, name in enumerate(LEVEL_NAMES):
        sm = seg_per_level[name].copy()
        f = feats_per_level[name]
        n = lengths[j]
        if n > 0:
            feat_concat[cum:cum+n] = f
            if j > 0:
                sm[sm != -1] += cum
        seg_stack[j] = sm
        cum += n

    np.save(save_path + '_s.npy', seg_stack)
    np.save(save_path + '_f.npy', feat_concat[:max(total, 1)].numpy())
    if save_pixel and pixel_feat is not None:
        small = downsize_pixel_feat(pixel_feat).astype(np.float16)
        np.save(save_path + '_p.npy', small)


# ---- Main ----

def main():
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True,
                        help='e.g. data/lerf-ovs/ramen')
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--fastsam_ckpt', type=str, default='ckpts/FastSAM-x.pt')
    parser.add_argument('--feat_subdir', type=str, default='language_features_openseg')
    parser.add_argument('--timing_csv', type=str, default=None)
    parser.add_argument('--backend', choices=['dense_clip', 'tfhub', 'stub'],
                        default='dense_clip',
                        help='dense_clip: PyTorch OpenCLIP ViT patch tokens (default, no extra deps); '
                             'tfhub: real OpenSeg via TF Hub (requires --openseg_url to a working mirror); '
                             'stub: random projection (plumbing only)')
    parser.add_argument('--openseg_url', type=str, default=None,
                        help='URL or local SavedModel dir for tfhub backend')
    parser.add_argument('--pixel_frames', nargs='*', default=None,
                        help='Save (downsized) pixel feature *_p.npy only for these frames '
                             '(by stem name like frame_00006). Use --pixel_all for everything.')
    parser.add_argument('--pixel_all', action='store_true')
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    if not _HAS_FASTSAM:
        raise RuntimeError("ultralytics not installed: pip install ultralytics")

    img_folder = os.path.join(args.source_path, 'images')
    save_folder = os.path.join(args.source_path, args.feat_subdir)
    os.makedirs(save_folder, exist_ok=True)

    if args.timing_csv is None:
        args.timing_csv = os.path.join(args.source_path, 'timing_openseg.csv')

    pixel_frames_set = set(args.pixel_frames) if args.pixel_frames else set()

    data_listi = sorted(os.listdir(img_folder))
    fastsam = FastSAM(args.fastsam_ckpt)
    openseg = make_openseg_backend(args.backend, args.openseg_url)

    write_header = not os.path.exists(args.timing_csv)
    timing_fp = open(args.timing_csv, 'a', newline='')
    timing_writer = csv.writer(timing_fp)
    if write_header:
        timing_writer.writerow(['frame', 't_mask_ms', 't_feat_ms', 't_pool_ms',
                                't_total_ms', 'n_masks', 'peak_mem_mb',
                                'resolution_w', 'resolution_h', 'pixel_saved'])

    WARNED = False
    for data_path in tqdm(data_listi, desc=f"OpenSeg+FastSAM ({args.backend})"):
        frame_name = os.path.splitext(data_path)[0]
        out_s = os.path.join(save_folder, frame_name + '_s.npy')
        out_f = os.path.join(save_folder, frame_name + '_f.npy')
        if os.path.exists(out_s) and os.path.exists(out_f):
            continue

        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"skip (cannot read): {image_path}"); continue
        orig_w, orig_h = image.shape[1], image.shape[0]

        if args.resolution == -1:
            global_down = orig_h / 1080 if orig_h > 1080 else 1
            if orig_h > 1080 and not WARNED:
                print("[INFO] >1080P, rescaling to 1080P."); WARNED = True
        else:
            global_down = orig_w / args.resolution
        scale = float(global_down)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        image = cv2.resize(image, resolution)

        torch.cuda.reset_peak_memory_stats()
        feats, seg_maps, pixel_feat, timing = encode_one_image(image, fastsam, openseg)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        save_pixel = args.pixel_all or (frame_name in pixel_frames_set)
        save_path = os.path.join(save_folder, frame_name)
        save_one(save_path, feats, seg_maps, pixel_feat=pixel_feat,
                 save_pixel=save_pixel, embed_size=pixel_feat.shape[-1])

        timing_writer.writerow([
            frame_name,
            f"{timing['t_mask_ms']:.2f}",
            f"{timing['t_feat_ms']:.2f}",
            f"{timing['t_pool_ms']:.2f}",
            f"{timing['t_total_ms']:.2f}",
            timing['n_masks'],
            f"{peak_mb:.1f}",
            resolution[0], resolution[1],
            int(save_pixel),
        ])
        timing_fp.flush()

    timing_fp.close()
    print(f"\nDone. Features: {save_folder}, timing: {args.timing_csv}")


if __name__ == '__main__':
    main()
