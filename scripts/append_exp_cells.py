"""Append the 24 experiment cells (level-by-level viz + miou) to the ramen notebook.

정상 ipynb 소스 포맷: source는 list[str]이고 각 문자열 끝에 \n 포함 (마지막 줄 제외).
"""
import json
import uuid

PATH = '/Users/baejunhak/labatory/THGS/THGS_Pipeline_ramen.ipynb'


def src_lines(text: str) -> list:
    """multi-line string → ipynb source list with trailing \n preserved."""
    lines = text.split("\n")
    out = [l + "\n" for l in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def code_cell(text: str):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src_lines(text),
    }


def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": src_lines(text),
    }


new = []

new.append(md_cell(
"""### Level별 시각화 & mIoU

Level당 **2개 셀 세트**:
- **시각화**: 각 (query × frame) 쌍마다 `RGB overlay`(원본 + 예측 빨강 + GT 초록) + `mask on black`(검은 배경, 예측 흰색, GT 경계 초록)
- **mIoU**: per-query 평균 IoU 테이블 (GT가 있는 L0/L1/L4/L5만)

L2/L3는 GT 없으므로 시각화만, mIoU 셀 생략.
아래 먼저 공통 헬퍼 정의."""
))

new.append(code_cell(
"""# [Cell] 시각화 공통 헬퍼
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# PRED_RECORDS, df, pair_df, gt_cache, QUERIES, EVAL_FRAMES, EXP_OUT,
# DATA_PATH, SCENE_NAME 은 앞 셀에서 정의됨.

def _load_frame_image(frame_name):
    for root in [f"{DATA_PATH}/{SCENE_NAME}/images", f"{DATA_PATH}/label/{SCENE_NAME}"]:
        p = os.path.join(root, f"{frame_name}.jpg")
        if os.path.exists(p):
            return cv2.imread(p)
    return None

frame_imgs = {f: _load_frame_image(f) for f in EVAL_FRAMES}

def _read_mask(path):
    if not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (m > 128).astype(np.uint8) if m is not None else None

def _overlay_rgb(img_bgr, mask, color_bgr=(0, 0, 255), alpha=0.45):
    out = img_bgr.copy()
    if mask is not None and mask.sum() > 0:
        colored = np.zeros_like(out)
        colored[mask.astype(bool)] = color_bgr
        out = cv2.addWeighted(out, 1.0, colored, alpha, 0)
    return out

def _mask_on_black(shape_hw, pred_mask, gt_mask=None):
    canvas = np.zeros((shape_hw[0], shape_hw[1], 3), dtype=np.uint8)
    if pred_mask is not None and pred_mask.sum() > 0:
        canvas[pred_mask.astype(bool)] = (255, 255, 255)
    if gt_mask is not None and gt_mask.sum() > 0:
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
    return canvas

def _safe_name(prompt):
    return prompt.replace(" ", "_").replace(".", "").replace(",", "")[:80]

def visualize_level(level_name, show=True, save=True):
    '''Level별 시각화: 각 (query, frame) -> [RGB overlay | mask on black] 2 panel.'''
    items = QUERIES[level_name]
    rows = len(items)
    cols = len(EVAL_FRAMES) * 2
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 2.0 * rows))
    if rows == 1:
        axes = axes[None, :]
    fig.suptitle(
        f"Level {level_name} — Ramen  |  각 frame: [RGB overlay | mask-on-black]",
        fontsize=12,
    )
    for i, it in enumerate(items):
        prompt = it["q"]
        gt_cat = it.get("gt")
        for j, frm in enumerate(EVAL_FRAMES):
            pred_p = os.path.join(EXP_OUT, frm, f"{level_name}_{_safe_name(prompt)}.png")
            pred = _read_mask(pred_p)
            img = frame_imgs[frm]
            gt = gt_cache[frm].get(gt_cat) if gt_cat else None

            vis_rgb = _overlay_rgb(img, pred, (0, 0, 255), 0.45)
            if gt is not None:
                vis_rgb = _overlay_rgb(vis_rgb, gt, (0, 255, 0), 0.30)
            axes[i, j * 2].imshow(cv2.cvtColor(vis_rgb, cv2.COLOR_BGR2RGB))

            mob = _mask_on_black(img.shape[:2], pred, gt)
            axes[i, j * 2 + 1].imshow(cv2.cvtColor(mob, cv2.COLOR_BGR2RGB))

            for ax in (axes[i, j * 2], axes[i, j * 2 + 1]):
                ax.set_xticks([])
                ax.set_yticks([])
            if i == 0:
                axes[i, j * 2].set_title(frm.replace("frame_", "f"), fontsize=8)
                axes[i, j * 2 + 1].set_title("mask", fontsize=8)
            if j == 0:
                label = prompt[:35] + ("…" if len(prompt) > 35 else "")
                axes[i, 0].set_ylabel(
                    label, fontsize=8, rotation=0, ha="right", va="center", labelpad=90
                )
    plt.tight_layout()
    if save:
        out_path = os.path.join(EXP_OUT, f"viz_{level_name}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        print(f"saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def miou_for_level(level_name):
    '''GT 있는 Level만 per-query mIoU + 전체 mIoU 출력.'''
    d = df[(df["level"] == level_name) & df["IoU"].notna()]
    if len(d) == 0:
        print(f"[{level_name}] 정량 가능 데이터 없음 (GT 없음).")
        return None
    per_q = d.groupby("query").agg(
        mIoU=("IoU", "mean"), mP=("P", "mean"), mR=("R", "mean"),
        mF1=("F1", "mean"), n=("IoU", "count"),
    ).round(4).sort_values("mIoU", ascending=False)
    print(f"=== Level {level_name} — per-query mIoU (n={len(d)} 쌍) ===")
    print(per_q.to_string())
    print()
    print(f"[{level_name}] 전체 평균 mIoU = {d['IoU'].mean():.4f}  "
          f"(P={d['P'].mean():.4f}, R={d['R'].mean():.4f})")
    return per_q


print("헬퍼 로드 완료. visualize_level('L0') / miou_for_level('L0') 호출 가능.")
"""
))

# Level별 2셀(또는 1셀) 세트
for lvl, has_gt in [
    ("L0", True), ("L1", True), ("L2", False),
    ("L3", False), ("L4", True), ("L5", True),
]:
    new.append(md_cell(f"#### Level {lvl} — 시각화"))
    new.append(code_cell(f"visualize_level('{lvl}')"))
    if has_gt:
        new.append(md_cell(f"#### Level {lvl} — mIoU"))
        if lvl == "L4":
            new.append(code_cell(
"""miou_for_level('L4')

# L4 '관계 무시' 정도 측정
_ov = df[(df.level == 'L4') & df.overlap_vs_L0.notna()]['overlap_vs_L0']
if len(_ov):
    print()
    print(f"[L4] 관계 무시 정도 — IoU(pred_L4, pred_L0) "
          f"평균={_ov.mean():.4f}, median={_ov.median():.4f}")
    print(f"    >0.85 비율 = {(_ov > 0.85).sum()}/{len(_ov)} "
          f"쌍 ({(_ov > 0.85).mean():.2%}) — CLIP이 공간관계를 무시하는 증거")
"""
            ))
        else:
            new.append(code_cell(f"miou_for_level('{lvl}')"))

# 종합 리포트
new.append(md_cell("### 종합 리포트 — Level 간 비교"))
new.append(code_cell(
"""# [Cell 종합] Level 간 비교 차트 + 텍스트 요약
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 10))

# (1) Level별 mIoU bar (GT 있는 Level만)
ax1 = fig.add_subplot(2, 2, 1)
sub = (
    df[df["IoU"].notna()]
    .groupby("level")["IoU"].mean()
    .reindex(["L0", "L1", "L4", "L5"]).dropna()
)
colors = ["#4C72B0", "#DD8452", "#55A467", "#C44E52"][: len(sub)]
ax1.bar(sub.index, sub.values, color=colors)
for i, v in enumerate(sub.values):
    ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
ax1.set_title("Level별 평균 IoU (GT 있는 Level만)")
ax1.set_ylabel("mIoU")
ax1.set_ylim(0, max(0.6, sub.max() * 1.2) if len(sub) else 1.0)

# (2) per-(category, frame) L0 vs 다른 Level IoU scatter
ax2 = fig.add_subplot(2, 2, 2)
pivot = df[df["IoU"].notna()].pivot_table(
    index=["gt_category", "frame"], columns="level", values="IoU"
)
for lvl, color in zip(["L1", "L4", "L5"], ["#DD8452", "#55A467", "#C44E52"]):
    if "L0" in pivot.columns and lvl in pivot.columns:
        x = pivot["L0"].values
        y = pivot[lvl].values
        m = ~(np.isnan(x) | np.isnan(y))
        ax2.scatter(x[m], y[m], alpha=0.6, label=lvl, color=color, s=40)
ax2.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax2.set_xlabel("L0 IoU")
ax2.set_ylabel("L1/L4/L5 IoU")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.legend()
ax2.set_title("L0 대비 각 Level IoU (대각선 아래 = L0 보다 나쁨)")

# (3) 관계·속성·문장 무시 histogram
ax3 = fig.add_subplot(2, 2, 3)
for lvl, color in zip(["L1", "L4", "L5"], ["#DD8452", "#55A467", "#C44E52"]):
    d = df[(df["level"] == lvl) & df["overlap_vs_L0"].notna()]
    ax3.hist(
        d["overlap_vs_L0"].values, bins=np.linspace(0, 1, 21),
        alpha=0.55, label=f"{lvl} (n={len(d)})", color=color,
    )
ax3.axvline(0.85, color="red", linestyle=":", lw=1)
ax3.text(0.86, ax3.get_ylim()[1] * 0.9, "임계=0.85", color="red", fontsize=8)
ax3.set_xlabel("IoU(pred_X, pred_L0)")
ax3.set_ylabel("count")
ax3.set_title("예측이 L0과 얼마나 동일한가 (높을수록 변형 무시)")
ax3.legend()

# (4) L3 pair overlap
ax4 = fig.add_subplot(2, 2, 4)
if len(pair_df) > 0:
    grp = pair_df.groupby("q")["iou"].mean().sort_values(ascending=False)
    ax4.barh(grp.index, grp.values, color="#8172B3")
    for i, v in enumerate(grp.values):
        ax4.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax4.axvline(0.8, color="red", linestyle=":", lw=1)
    ax4.set_xlim(0, 1.05)
    ax4.set_title("L3 left/right pair overlap (높을수록 instance 분리 실패)")
else:
    ax4.text(0.5, 0.5, "L3 pair 데이터 없음", ha="center", va="center")

plt.tight_layout()
summary_path = os.path.join(EXP_OUT, "summary_report.png")
plt.savefig(summary_path, dpi=140, bbox_inches="tight")
plt.show()
print(f"saved: {summary_path}")

# 텍스트 리포트
print()
print("=" * 70)
print("종합 텍스트 리포트")
print("=" * 70)
print()
print("[Level별 평균 mIoU]")
print(sub.round(4).to_string())

if "L0" in pivot.columns:
    print()
    print("[L0 대비 IoU drop]")
    for lvl in ["L1", "L4", "L5"]:
        if lvl in pivot.columns:
            drop = (pivot["L0"] - pivot[lvl]).dropna()
            print(f"  L0 → {lvl}: mean={drop.mean():+.4f}  "
                  f"median={drop.median():+.4f}  worst={drop.max():+.4f}")

print()
print("[L1/L4/L5 vs L0 예측 overlap]")
for lvl in ["L1", "L4", "L5"]:
    d = df[(df.level == lvl) & df.overlap_vs_L0.notna()]["overlap_vs_L0"]
    if len(d):
        print(f"  {lvl}: mean={d.mean():.4f}  median={d.median():.4f}  "
              f">0.85 비율={(d > 0.85).mean():.2%}")

if len(pair_df) > 0:
    high = (pair_df["iou"] > 0.8).mean()
    print()
    print("[L3 instance 분리]")
    print(f"  평균 left/right overlap = {pair_df['iou'].mean():.4f}")
    print(f"  >0.8 비율 = {high:.2%} (instance 분리 실패)")

print()
print("L2 (파트)는 viz_L2.png 참조. GT 없으므로 mIoU 없음.")
print("L3 (인스턴스)도 GT 없으므로 mIoU 없음. pair overlap이 그 자리를 대체.")
"""
))

nb = json.load(open(PATH))
nb['cells'].extend(new)
json.dump(nb, open(PATH, 'w'), indent=1)
print(f'appended {len(new)} cells, total now {len(nb["cells"])}')
