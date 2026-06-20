"""2D semantic selection diagnostic for VALA/LangSplat language_features.

For each GT object in LERF-OVS eval frames, this checks whether the existing
per-mask CLIP features would choose a good 2D mask before any 3DGS lifting.
It loads only the CLIP text encoder on CPU, then scores saved mask features.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch


LEVEL_NAMES = ["default", "s", "m", "l"]
NEGATIVES = ["object", "things", "stuff", "texture"]


def polygon_to_mask(shape: tuple[int, int], points: list[list[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def frame_gt_masks(json_path: Path) -> dict[str, tuple[np.ndarray, int]]:
    data = json.loads(json_path.read_text())
    h = int(data["info"]["height"])
    w = int(data["info"]["width"])
    out: dict[str, tuple[np.ndarray, int]] = {}
    for obj in data["objects"]:
        prompt = obj["category"]
        mask, n = out.get(prompt, (np.zeros((h, w), dtype=bool), 0))
        mask |= polygon_to_mask((h, w), obj["segmentation"])
        out[prompt] = (mask, n + 1)
    return out


def canon_scores(feats: np.ndarray, prompt_feat: np.ndarray, neg_feats: np.ndarray) -> np.ndarray:
    pos = feats @ prompt_feat
    neg = feats @ neg_feats.T
    # min over canon negatives of softmax probability assigned to positive.
    logits_pos = 10.0 * pos[:, None]
    logits_neg = 10.0 * neg
    mx = np.maximum(logits_pos, logits_neg)
    exp_pos = np.exp(logits_pos - mx)
    exp_neg = np.exp(logits_neg - mx)
    probs = exp_pos / (exp_pos + exp_neg)
    return probs.min(axis=1)


def iou_stats(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float, int]:
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    gt_area = int(gt.sum())
    pred_area = int(pred.sum())
    iou = inter / max(union, 1)
    recall = inter / max(gt_area, 1)
    area_ratio = pred_area / max(gt_area, 1)
    return float(iou), float(recall), float(area_ratio), pred_area


def norm_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


@torch.no_grad()
def encode_texts(prompts: list[str]) -> dict[str, np.ndarray]:
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        precision="fp32",
        device="cpu",
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    toks = tokenizer(prompts)
    feats = model.encode_text(toks).float()
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return {p: feats[i].cpu().numpy().astype(np.float32) for i, p in enumerate(prompts)}


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HOME", str(Path(args.hf_home).resolve()))
    data_root = Path(args.data_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_prompts = set(NEGATIVES)
    for scene in args.scenes:
        for json_path in (data_root / "label" / scene).glob("frame_*.json"):
            for prompt in frame_gt_masks(json_path):
                all_prompts.add(prompt)
    print(f"encoding {len(all_prompts)} texts on CPU...")
    text = encode_texts(sorted(all_prompts))
    neg_feats = np.stack([text[n] for n in NEGATIVES], axis=0)

    rows = []
    for scene in args.scenes:
        label_dir = data_root / "label" / scene
        feat_dir = data_root / scene / args.feature_dir
        for json_path in sorted(label_dir.glob("frame_*.json")):
            frame = json_path.stem
            seg_path = feat_dir / f"{frame}_s.npy"
            feat_path = feat_dir / f"{frame}_f.npy"
            if not seg_path.exists() or not feat_path.exists():
                continue
            seg = np.load(seg_path)
            feats = norm_rows(np.load(feat_path))
            masks = frame_gt_masks(json_path)
            for prompt, (gt, n_inst) in masks.items():
                pfeat = text[prompt]
                scores = canon_scores(feats, pfeat, neg_feats)

                best_score = -1.0
                best = None
                best_oracle_iou = -1.0
                best_oracle = None

                for li, lname in enumerate(LEVEL_NAMES):
                    ids = np.unique(seg[li])
                    ids = ids[ids >= 0].astype(np.int64)
                    if ids.size == 0:
                        continue
                    id_scores = scores[ids]
                    top_id = int(ids[int(np.argmax(id_scores))])
                    pred = seg[li] == top_id
                    top_iou, top_recall, top_area_ratio, top_area = iou_stats(pred, gt)
                    top_score = float(scores[top_id])
                    if top_score > best_score:
                        best_score = top_score
                        best = (lname, top_id, top_iou, top_recall, top_area_ratio, top_area)

                    for mid in ids:
                        pred2 = seg[li] == int(mid)
                        oiou, orec, oarea_ratio, oarea = iou_stats(pred2, gt)
                        if oiou > best_oracle_iou:
                            order = np.argsort(-id_scores)
                            rank = int(np.where(ids[order] == mid)[0][0] + 1)
                            best_oracle_iou = oiou
                            best_oracle = (
                                lname,
                                int(mid),
                                oiou,
                                orec,
                                oarea_ratio,
                                oarea,
                                float(scores[int(mid)]),
                                rank,
                            )

                if best is None or best_oracle is None:
                    continue
                rows.append(
                    {
                        "scene": scene,
                        "frame": frame,
                        "prompt": prompt,
                        "n_instances": n_inst,
                        "sem_top_level": best[0],
                        "sem_top_mask_id": best[1],
                        "sem_top_score": f"{best_score:.6f}",
                        "sem_top_iou": f"{best[2]:.6f}",
                        "sem_top_recall": f"{best[3]:.6f}",
                        "sem_top_area_ratio": f"{best[4]:.6f}",
                        "sem_top_area": best[5],
                        "oracle_level": best_oracle[0],
                        "oracle_mask_id": best_oracle[1],
                        "oracle_iou": f"{best_oracle[2]:.6f}",
                        "oracle_recall": f"{best_oracle[3]:.6f}",
                        "oracle_area_ratio": f"{best_oracle[4]:.6f}",
                        "oracle_area": best_oracle[5],
                        "oracle_score": f"{best_oracle[6]:.6f}",
                        "oracle_score_rank_in_level": best_oracle[7],
                    }
                )

    fieldnames = [
        "scene",
        "frame",
        "prompt",
        "n_instances",
        "sem_top_level",
        "sem_top_mask_id",
        "sem_top_score",
        "sem_top_iou",
        "sem_top_recall",
        "sem_top_area_ratio",
        "sem_top_area",
        "oracle_level",
        "oracle_mask_id",
        "oracle_iou",
        "oracle_recall",
        "oracle_area_ratio",
        "oracle_area",
        "oracle_score",
        "oracle_score_rank_in_level",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for scene in args.scenes:
        scene_rows = [r for r in rows if r["scene"] == scene]
        top = np.asarray([float(r["sem_top_iou"]) for r in scene_rows], dtype=np.float32)
        oracle = np.asarray([float(r["oracle_iou"]) for r in scene_rows], dtype=np.float32)
        rank = np.asarray([int(r["oracle_score_rank_in_level"]) for r in scene_rows], dtype=np.float32)
        print(
            f"{scene}: n={len(scene_rows)} sem_top_iou={top.mean():.4f} "
            f"oracle_iou={oracle.mean():.4f} oracle_rank_med={np.median(rank):.1f} "
            f"top>=0.5={(top >= 0.5).mean():.3f}"
        )
        for r in sorted(scene_rows, key=lambda x: float(x["sem_top_iou"]))[:8]:
            print(
                f"  low {r['prompt']:<16s} {r['frame']} "
                f"top_iou={float(r['sem_top_iou']):.3f} "
                f"oracle={float(r['oracle_iou']):.3f} "
                f"oracle_rank={r['oracle_score_rank_in_level']} "
                f"top_area={float(r['sem_top_area_ratio']):.2f}"
            )
    print(f"wrote {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/lerf_ovs")
    parser.add_argument("--feature_dir", default="language_features")
    parser.add_argument("--scenes", nargs="+", default=["ramen", "teatime"])
    parser.add_argument("--hf_home", default=".valahome/hf")
    parser.add_argument(
        "--out_csv",
        default="output/diagnostics/p1e_vala_2d_semantic_selection.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
