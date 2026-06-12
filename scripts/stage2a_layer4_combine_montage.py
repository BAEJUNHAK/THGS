"""Combine pickled per-prompt panels into a montage figure."""
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels_pkl", default="output/diagnostics/_phantom_panels.pkl")
    parser.add_argument("--persistent_csv", default="output/diagnostics/persistent_phantoms_17.csv")
    parser.add_argument("--out", default="output/diagnostics/plots/phantom_montage.png")
    args = parser.parse_args()

    with open(args.panels_pkl, "rb") as f:
        panels = pickle.load(f)
    print(f"Loaded {len(panels)} panels")

    ordering = []
    df = pd.read_csv(args.persistent_csv)
    for _, r in df.iterrows():
        ordering.append((r['scene'], r['prompt']))

    rows = []
    for (sc, p) in ordering:
        if (sc, p) not in panels:
            print(f"  missing: {sc} {p}")
            continue
        gt, oracle, top1, top10 = panels[(sc, p)]
        max_w = max(gt.shape[1], oracle.shape[1], top1.shape[1], top10.shape[1])
        def pad_w(im):
            h, w = im.shape[:2]
            if w == max_w:
                return im
            pad = np.zeros((h, max_w - w, 3), dtype=im.dtype)
            return np.concatenate([im, pad], axis=1)
        gt, oracle, top1, top10 = pad_w(gt), pad_w(oracle), pad_w(top1), pad_w(top10)

        # gap between panels
        gap = np.zeros((gt.shape[0], 8, 3), dtype=np.uint8)
        row_imgs = np.concatenate([gt, gap, oracle, gap, top1, gap, top10], axis=1)

        # add label strip on left
        label_h = row_imgs.shape[0]
        label_w = 180
        label_strip = np.full((label_h, label_w, 3), 35, dtype=np.uint8)
        text = f"{sc}: {p}"
        cv2.putText(label_strip, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        row_full = np.concatenate([label_strip, row_imgs], axis=1)
        rows.append(row_full)

    if not rows:
        print("No panels to montage")
        return

    max_w = max(r.shape[1] for r in rows)
    norm_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=r.dtype)
            r = np.concatenate([r, pad], axis=1)
        norm_rows.append(r)

    # column header
    header_h = 40
    header = np.full((header_h, max_w, 3), 60, dtype=np.uint8)
    # 4 columns aligned with panels: left label_w + col x widths
    panel_w = (max_w - 180 - 24) // 4
    cv2.putText(header, "label", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    labels = ["GT (green)", "Oracle SP (cyan)", "CLIP top-1 (red)", "top-k=10 (yellow)"]
    for i, lbl in enumerate(labels):
        x = 180 + i * (panel_w + 8) + 8
        cv2.putText(header, lbl, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # separator rows between
    sep_rows = []
    for i, r in enumerate(norm_rows):
        sep_rows.append(r)
        if i < len(norm_rows) - 1:
            sep_rows.append(np.zeros((4, max_w, 3), dtype=np.uint8))

    montage = np.concatenate([header] + sep_rows, axis=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print(f"Wrote {montage.shape} montage to {args.out}")


if __name__ == "__main__":
    main()
