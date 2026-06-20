#!/usr/bin/env python3
"""CPU diagnostic for VALA's 3D language relevance field.

VALA renders a query mask by scoring each language-lifted Gaussian against the
text prompt, smoothing scores over 3D neighbors, min-max normalizing per prompt,
and thresholding the normalized relevance. This script reproduces the scoring
side on CPU and reports how broad each query's selected Gaussian set becomes.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
from sklearn.neighbors import NearestNeighbors


NEGATIVES = ["object", "things", "stuff", "texture"]
SCENE_TEXTS = {
    "ramen": [
        "nori",
        "sake cup",
        "kamaboko",
        "corn",
        "spoon",
        "egg",
        "onion segments",
        "plate",
        "napkin",
        "bowl",
        "glass of water",
        "hand",
        "chopsticks",
        "wavy noodles",
    ],
    "teatime": [
        "sheep",
        "yellow pouf",
        "stuffed bear",
        "coffee mug",
        "tea in a glass",
        "apple",
        "coffee",
        "hooves",
        "bear nose",
        "dall-e brand",
        "plate",
        "paper napkin",
        "three cookies",
        "bag of cookies",
    ],
}


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


def canon_relevance(
    feats: np.ndarray,
    prompt_feats: np.ndarray,
    neg_feats: np.ndarray,
    chunk: int,
) -> np.ndarray:
    """Match VALA OpenCLIPNetwork.get_relevancy for many prompts."""
    n = feats.shape[0]
    out = np.zeros((prompt_feats.shape[0], n), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        emb = feats[start:stop]
        pos = emb @ prompt_feats.T
        neg = emb @ neg_feats.T
        for j in range(prompt_feats.shape[0]):
            logits_pos = 10.0 * pos[:, j : j + 1]
            logits_neg = 10.0 * neg
            mx = np.maximum(logits_pos, logits_neg)
            exp_pos = np.exp(logits_pos - mx)
            exp_neg = np.exp(logits_neg - mx)
            probs = exp_pos / (exp_pos + exp_neg)
            out[j, start:stop] = probs.min(axis=1)
    return out


def smooth_scores(xyz: np.ndarray, relevance: np.ndarray, k: int) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k).fit(xyz)
    _, indices = nbrs.kneighbors(xyz)
    smoothed = np.empty_like(relevance)
    for i in range(relevance.shape[0]):
        smoothed[i] = 0.5 * (relevance[i] + relevance[i, indices].mean(axis=1))
    return smoothed


def normalize_like_vala(score: np.ndarray) -> np.ndarray:
    x = score - score.min()
    x = x / (x.max() + 1e-9)
    x = (2.0 * x) - 1.0
    return np.clip(x, 0.0, 1.0)


def load_checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray]:
    model_params, _ = torch.load(path, map_location="cpu")
    xyz = model_params[1].detach().cpu().numpy().astype(np.float32)
    feat = model_params[7].detach().cpu().numpy().astype(np.float32)
    return xyz, norm_rows(feat)


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HOME", str(Path(args.hf_home).resolve()))
    model_root = Path(args.model_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_prompts = sorted(set(NEGATIVES) | {p for s in args.scenes for p in SCENE_TEXTS[s]})
    text = encode_texts(all_prompts)
    neg_feats = np.stack([text[n] for n in NEGATIVES], axis=0)

    rows: list[dict[str, str]] = []
    chosen_rows: list[dict[str, str]] = []
    thresholds = [float(x) for x in args.thresholds]

    for scene in args.scenes:
        prompts = SCENE_TEXTS[scene]
        prompt_feats = np.stack([text[p] for p in prompts], axis=0)
        level_max_scores: list[np.ndarray] = []
        level_rows: list[list[dict[str, str]]] = []

        for level in (1, 2, 3):
            ckpt = (
                model_root
                / scene
                / args.ablation_type
                / f"chkpnt30000_langfeat_{level}_stochastic_gate.pth"
            )
            print(f"loading {ckpt}")
            xyz, feats = load_checkpoint(ckpt)
            relevance = canon_relevance(feats, prompt_feats, neg_feats, args.chunk)
            if args.smooth:
                relevance = smooth_scores(xyz, relevance, args.knn)
            level_max_scores.append(relevance.max(axis=1))

            cur_rows: list[dict[str, str]] = []
            for pi, prompt in enumerate(prompts):
                raw = relevance[pi]
                norm = normalize_like_vala(raw)
                row = {
                    "scene": scene,
                    "prompt": prompt,
                    "level": str(level),
                    "n_gaussians": str(feats.shape[0]),
                    "score_min": f"{raw.min():.6f}",
                    "score_mean": f"{raw.mean():.6f}",
                    "score_p90": f"{np.quantile(raw, 0.90):.6f}",
                    "score_p99": f"{np.quantile(raw, 0.99):.6f}",
                    "score_max": f"{raw.max():.6f}",
                    "norm_mean": f"{norm.mean():.6f}",
                    "norm_p90": f"{np.quantile(norm, 0.90):.6f}",
                    "norm_p99": f"{np.quantile(norm, 0.99):.6f}",
                }
                for thr in thresholds:
                    row[f"selected_frac_t{thr:g}"] = f"{(norm > thr).mean():.6f}"
                    row[f"selected_count_t{thr:g}"] = str(int((norm > thr).sum()))
                rows.append(row)
                cur_rows.append(row)
            level_rows.append(cur_rows)

        chosen = np.argmax(np.stack(level_max_scores, axis=0), axis=0)
        for pi, prompt in enumerate(prompts):
            row = dict(level_rows[int(chosen[pi])][pi])
            row["chosen_level"] = row.pop("level")
            chosen_rows.append(row)

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    chosen_csv = out_csv.with_name(out_csv.stem + "_chosen.csv")
    chosen_fields = list(chosen_rows[0].keys())
    with chosen_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=chosen_fields)
        writer.writeheader()
        writer.writerows(chosen_rows)

    for scene in args.scenes:
        scene_rows = [r for r in chosen_rows if r["scene"] == scene]
        print(scene)
        for thr in thresholds:
            vals = np.asarray([float(r[f"selected_frac_t{thr:g}"]) for r in scene_rows])
            print(
                f"  t={thr:g} selected_frac mean={vals.mean():.4f} "
                f"median={np.median(vals):.4f} max={vals.max():.4f}"
            )
    print(f"wrote {out_csv} and {chosen_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", default="external_methods/VALA/output/3dgs/lerf_ovs")
    parser.add_argument("--ablation_type", default="none")
    parser.add_argument("--scenes", nargs="+", default=["ramen", "teatime"])
    parser.add_argument("--thresholds", nargs="+", default=["0.4", "0.5", "0.6", "0.7", "0.8"])
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=65536)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--hf_home", default=".valahome/hf")
    parser.add_argument("--out_csv", default="output/diagnostics/p1e_vala_3d_relevance.csv")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
