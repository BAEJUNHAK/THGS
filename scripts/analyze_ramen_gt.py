"""
Ramen scene GT / frame / camera 구조 분석.
산출물 (md/ 아래):
  - ramen_frame_inventory.md
  - ramen_gt_categories.md
  - ramen_query_gt_matrix.md
  - ramen_camera_distribution.md
  - ramen_visualization/<frame>_all_overlay.png
"""
from __future__ import annotations

import json
import os
import struct
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "lerf-ovs"
IMAGES_DIR = DATA_ROOT / "ramen" / "images"
LABEL_DIR = DATA_ROOT / "label" / "ramen"
SPARSE_DIR = DATA_ROOT / "ramen" / "sparse" / "0"
MD_DIR = ROOT / "md"
VIS_DIR = MD_DIR / "ramen_visualization"

Q_LEVEL0 = [
    "bowl", "chopsticks", "corn", "egg", "eggs",
    "glass of water", "hand", "kamaboko", "nori",
    "plate", "sake cup", "spoon", "wavy noodles",
]


def load_label_jsons():
    files = sorted(glob(str(LABEL_DIR / "*.json")))
    data = {}
    for p in files:
        with open(p) as f:
            data[Path(p).stem] = json.load(f)
    return data


def write_frame_inventory(labels):
    all_imgs = sorted(glob(str(IMAGES_DIR / "*.jpg")))
    eval_frames = sorted(labels.keys())
    sizes = {}
    for p in all_imgs:
        with Image.open(p) as im:
            sizes.setdefault(im.size, []).append(Path(p).name)

    lines = [
        "# Ramen Frame Inventory",
        "",
        f"- 전체 학습용 이미지 수: **{len(all_imgs)}장** (`frame_00001.jpg` ~ `frame_{len(all_imgs):05d}.jpg`)",
        f"- GT 평가 프레임 수: **{len(eval_frames)}장**",
        f"- 이미지 저장 위치: `{IMAGES_DIR.relative_to(ROOT)}`",
        f"- GT 라벨 저장 위치: `{LABEL_DIR.relative_to(ROOT)}`",
        "",
        "## 해상도 분포",
        "",
        "| (W x H) | 프레임 수 |",
        "|---|---|",
    ]
    for size, files in sizes.items():
        lines.append(f"| {size[0]} x {size[1]} | {len(files)} |")

    lines += ["", "## GT 평가 프레임 목록", "", "| frame | image size (w,h) | GT bbox count | JSON filesize |", "|---|---|---|---|"]
    for frm in eval_frames:
        info = labels[frm]["info"]
        n_obj = len(labels[frm]["objects"])
        jpath = LABEL_DIR / f"{frm}.json"
        lines.append(f"| {frm} | {info['width']} x {info['height']} | {n_obj} | {jpath.stat().st_size} bytes |")

    lines += [
        "",
        "## q.md와의 일치성",
        "",
        "q.md가 상정한 7개 프레임: `frame_00006, 00024, 00060, 00065, 00081, 00119, 00128`",
        "",
    ]
    q_frames = ["frame_00006", "frame_00024", "frame_00060", "frame_00065",
                "frame_00081", "frame_00119", "frame_00128"]
    match = all(f in eval_frames for f in q_frames)
    lines.append(f"**결과**: {'완전 일치 ✓' if match else '불일치 ✗'} — LERF-OVS 공식 GT에 q.md의 7 프레임이 {'모두 있음' if match else '일부 없음'}.")
    lines.append("")
    lines.append("> q.md가 Ref-LERF 기반이라고 명시했지만, LERF-OVS 공식 zip에 이미 동일한 7 프레임이 들어있어 Ref-LERF 없이도 동일 평가가 가능.")

    (MD_DIR / "ramen_frame_inventory.md").write_text("\n".join(lines))


def write_gt_categories(labels):
    cat_per_frame = {}
    cat_total_instances = Counter()
    cat_frame_presence = defaultdict(set)

    for frm, data in labels.items():
        cats_in_frame = [o["category"] for o in data["objects"]]
        cat_per_frame[frm] = cats_in_frame
        for c in cats_in_frame:
            cat_total_instances[c] += 1
            cat_frame_presence[c].add(frm)

    all_cats = sorted(cat_total_instances.keys())

    lines = [
        "# Ramen GT 카테고리 전수 조사",
        "",
        f"- 전체 GT 프레임: **{len(labels)}** 장",
        f"- unique category 수: **{len(all_cats)}** 개",
        f"- 총 polygon instance 수: **{sum(cat_total_instances.values())}** 개",
        "",
        "## 전체 카테고리 리스트 (alphabetical)",
        "",
        "| # | Category | 전체 등장 횟수 (polygon) | 프레임 커버리지 |",
        "|---|---|---|---|",
    ]
    n_frames = len(labels)
    for i, c in enumerate(all_cats, 1):
        cnt = cat_total_instances[c]
        pres = len(cat_frame_presence[c])
        lines.append(f"| {i} | `{c}` | {cnt} | {pres}/{n_frames} |")

    lines += [
        "",
        "## 프레임별 카테고리",
        "",
    ]
    for frm in sorted(labels.keys()):
        cats = cat_per_frame[frm]
        dup = Counter(cats)
        pretty = ", ".join(f"`{k}`" + (f"×{v}" if v > 1 else "") for k, v in dup.items())
        lines.append(f"- **{frm}** ({len(cats)} objects): {pretty}")

    lines += [
        "",
        "## q.md Level 0 카테고리 대조",
        "",
        "| q.md 쿼리 | LERF-OVS GT 존재? | 전체 등장 | 프레임 커버리지 |",
        "|---|---|---|---|",
    ]
    q_set = set(Q_LEVEL0)
    gt_set = set(all_cats)
    for q in Q_LEVEL0:
        if q in gt_set:
            lines.append(f"| `{q}` | ✓ | {cat_total_instances[q]} | {len(cat_frame_presence[q])}/{n_frames} |")
        else:
            lines.append(f"| `{q}` | ✗ (없음) | — | 0/{n_frames} |")

    missing_from_gt = sorted(q_set - gt_set)
    extra_in_gt = sorted(gt_set - q_set)
    lines += [
        "",
        "### 요약",
        "",
        f"- q.md Level 0 {len(Q_LEVEL0)}개 중 실제 GT 있음: **{len(q_set & gt_set)}**개",
        f"- q.md에 있으나 GT 없음: **{missing_from_gt}**",
        f"- GT에 있으나 q.md에 없음: **{extra_in_gt}**",
    ]

    (MD_DIR / "ramen_gt_categories.md").write_text("\n".join(lines))
    return cat_per_frame, cat_frame_presence


def write_query_gt_matrix(cat_frame_presence, labels):
    frames = sorted(labels.keys())
    all_cats_in_gt = sorted(cat_frame_presence.keys())

    lines = [
        "# Query × Frame GT 매트릭스",
        "",
        "각 셀: 해당 프레임에 해당 카테고리 polygon GT가 **존재**(✓) / **없음**(·).",
        "이것으로 per-(category, frame) 정량 평가 가능 여부 확정.",
        "",
        "## 1. q.md Level 0 (13개) 쿼리 기준",
        "",
    ]
    header = "| query | " + " | ".join(f.replace("frame_", "f") for f in frames) + " | 합계 |"
    sep = "|---|" + "|".join(["---"] * (len(frames) + 1)) + "|"
    lines += [header, sep]
    for q in Q_LEVEL0:
        pres_frames = cat_frame_presence.get(q, set())
        cells = ["✓" if f in pres_frames else "·" for f in frames]
        total = len(pres_frames)
        lines.append(f"| `{q}` | " + " | ".join(cells) + f" | {total}/{len(frames)} |")

    lines += [
        "",
        "## 2. LERF-OVS에 실제 존재하는 모든 카테고리 기준",
        "",
        header,
        sep,
    ]
    for q in all_cats_in_gt:
        pres_frames = cat_frame_presence.get(q, set())
        cells = ["✓" if f in pres_frames else "·" for f in frames]
        total = len(pres_frames)
        lines.append(f"| `{q}` | " + " | ".join(cells) + f" | {total}/{len(frames)} |")

    q_set = set(Q_LEVEL0)
    gt_set = set(all_cats_in_gt)
    lines += [
        "",
        "## 3. 정량 평가 가능 쿼리·프레임 쌍 총수",
        "",
        f"- Level 0 (q.md) 유효 쌍: **{sum(len(cat_frame_presence.get(q, set())) for q in q_set & gt_set)}**",
        f"- GT 전체 유효 쌍: **{sum(len(cat_frame_presence[c]) for c in all_cats_in_gt)}**",
        "",
        "## 4. Level별 평가 가능성 요약",
        "",
        "| Level | q.md 쿼리 수 | GT mask 공유 가능 여부 | 실제 정량 가능 쿼리 수 |",
        "|---|---|---|---|",
        f"| Level 0 (base) | 13 | O | {len(q_set & gt_set)} (GT에 존재하는 것만) |",
        "| Level 1 (속성) | 13 | O (Level 0 GT 재사용) | Level 0과 동수 |",
        "| Level 2 (파트) | 14 | X (정성만) | 0 |",
        "| Level 3 (인스턴스) | 4 | 부분적 (L0 GT의 개별 polygon 활용) | 0 자동, 수작업 가능 |",
        "| Level 4 (관계) | 13 | O (L0 GT 재사용, 관계 무시 검증) | Level 0과 동수 |",
        "| Level 5 (문장) | 13 | O (Level 0 GT 재사용) | Level 0과 동수 |",
    ]
    (MD_DIR / "ramen_query_gt_matrix.md").write_text("\n".join(lines))


def visualize_gt_overlays(labels):
    """각 평가 프레임에 모든 GT polygon을 색상별로 overlay."""
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (0, 128, 255),
        (128, 255, 0), (255, 0, 128), (0, 255, 128),
        (200, 200, 200), (128, 128, 0), (0, 128, 128),
        (64, 64, 192),
    ]

    category_to_color = {}
    all_cats = set()
    for data in labels.values():
        for o in data["objects"]:
            all_cats.add(o["category"])
    for i, c in enumerate(sorted(all_cats)):
        category_to_color[c] = palette[i % len(palette)]

    for frm, data in labels.items():
        img_path = LABEL_DIR / f"{frm}.jpg"
        if not img_path.exists():
            img_path = IMAGES_DIR / f"{frm}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        overlay = img.copy()
        legend_h = 30 * len(set(o["category"] for o in data["objects"])) + 10
        legend = np.full((legend_h, img.shape[1], 3), 255, dtype=np.uint8)
        seen_cats = {}
        for o in data["objects"]:
            cat = o["category"]
            color = category_to_color[cat]
            pts = np.array(o["segmentation"], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
            seen_cats.setdefault(cat, color)

        blended = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

        for i, (cat, color) in enumerate(sorted(seen_cats.items())):
            y = 25 + i * 30
            cv2.rectangle(legend, (10, y - 18), (40, y + 2), color, -1)
            cv2.putText(legend, cat, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        combined = np.vstack([blended, legend])
        out_path = VIS_DIR / f"{frm}_all_overlay.png"
        cv2.imwrite(str(out_path), combined)
        print(f"  saved {out_path.name}: {len(data['objects'])} polys, {len(seen_cats)} unique cats")


def parse_cameras_bin():
    """
    COLMAP cameras.bin 읽어서 (camera_id -> (model, w, h, params)) 반환.
    """
    path = SPARSE_DIR / "cameras.bin"
    if not path.exists():
        return None
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            model_names = {
                0: ("SIMPLE_PINHOLE", 3),
                1: ("PINHOLE", 4),
                2: ("SIMPLE_RADIAL", 4),
                3: ("RADIAL", 5),
                4: ("OPENCV", 8),
                5: ("OPENCV_FISHEYE", 8),
                6: ("FULL_OPENCV", 12),
                7: ("FOV", 5),
                8: ("SIMPLE_RADIAL_FISHEYE", 4),
                9: ("SIMPLE_PINHOLE_FISHEYE", 4),
                10: ("RADIAL_FISHEYE", 5),
                11: ("THIN_PRISM_FISHEYE", 12),
            }
            name, nparam = model_names.get(model_id, ("UNKNOWN", 0))
            params = struct.unpack(f"<{nparam}d", f.read(8 * nparam))
            cameras[cam_id] = (name, w, h, params)
    return cameras


def parse_images_bin():
    """
    COLMAP images.bin 파싱. (image_id -> dict) 반환.
    dict 키: qw,qx,qy,qz, tx,ty,tz, camera_id, name, xys(무시), point3D_ids(무시).
    """
    path = SPARSE_DIR / "images.bin"
    if not path.exists():
        return None
    images = {}
    with open(path, "rb") as f:
        num_reg = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_reg):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch.decode("utf-8", errors="replace")
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            # skip 2D correspondences
            f.read(num_points2D * 24)
            images[image_id] = {
                "q": (qw, qx, qy, qz),
                "t": (tx, ty, tz),
                "camera_id": cam_id,
                "name": name,
            }
    return images


def qvec_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def write_camera_distribution():
    cameras = parse_cameras_bin()
    images = parse_images_bin()
    if cameras is None or images is None:
        (MD_DIR / "ramen_camera_distribution.md").write_text(
            "# Camera Distribution\n\nCOLMAP bin 파일을 읽을 수 없음 (파일 없음).\n"
        )
        return

    # camera centers in world coordinates: C = -R^T t
    centers = {}
    for iid, info in images.items():
        R = qvec_to_R(info["q"])
        t = np.array(info["t"])
        C = -R.T @ t
        centers[info["name"]] = C

    all_names = sorted(centers.keys())
    eval_names = [
        "frame_00006.jpg", "frame_00024.jpg", "frame_00060.jpg", "frame_00065.jpg",
        "frame_00081.jpg", "frame_00119.jpg", "frame_00128.jpg",
    ]

    # pairwise angular spread of eval cameras (center vectors from scene centroid)
    all_C = np.array([centers[n] for n in all_names])
    centroid = all_C.mean(axis=0)
    eval_C = np.array([centers[n] for n in eval_names if n in centers])

    def angle_deg(v1, v2):
        v1 = v1 / (np.linalg.norm(v1) + 1e-9)
        v2 = v2 / (np.linalg.norm(v2) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(v1 @ v2, -1, 1))))

    pair_angles = []
    for i in range(len(eval_C)):
        for j in range(i + 1, len(eval_C)):
            a = angle_deg(eval_C[i] - centroid, eval_C[j] - centroid)
            pair_angles.append((eval_names[i], eval_names[j], a))

    # camera intrinsics 요약
    cam_lines = []
    for cid, (name, w, h, params) in cameras.items():
        cam_lines.append(f"  - cam_id={cid} model={name} {w}x{h} params={params}")

    lines = [
        "# Ramen Camera Distribution",
        "",
        f"- 전체 등록된 카메라 수: **{len(images)}**",
        f"- images 파일의 name 샘플: {all_names[:3]} ... {all_names[-3:]}",
        "",
        "## Intrinsics",
        "",
    ]
    lines += cam_lines
    lines += [
        "",
        "## 평가 7 프레임의 카메라 중심 좌표 (월드 좌표계)",
        "",
        "| frame | Cx | Cy | Cz |",
        "|---|---|---|---|",
    ]
    for n in eval_names:
        if n in centers:
            c = centers[n]
            lines.append(f"| {n} | {c[0]:+.3f} | {c[1]:+.3f} | {c[2]:+.3f} |")
        else:
            lines.append(f"| {n} | (카메라 bin에 없음) | | |")

    lines += [
        "",
        "## 평가 프레임 간 viewpoint 각도 (scene centroid 기준)",
        "",
        "| frame A | frame B | 각도 (deg) |",
        "|---|---|---|",
    ]
    for a, b, ang in pair_angles:
        lines.append(f"| {a} | {b} | {ang:.1f} |")

    lines += [
        "",
        "## 해석",
        "",
        f"- 평가 프레임 간 평균 시점 각도: **{np.mean([a for _,_,a in pair_angles]):.1f}°**",
        f"- 최소: **{np.min([a for _,_,a in pair_angles]):.1f}°**, 최대: **{np.max([a for _,_,a in pair_angles]):.1f}°**",
        "",
        "→ 각도가 크면 서로 다른 시점 커버, 작으면 비슷한 각도에서 본 프레임이 섞여 있음.",
    ]
    (MD_DIR / "ramen_camera_distribution.md").write_text("\n".join(lines))


def main():
    MD_DIR.mkdir(exist_ok=True)
    VIS_DIR.mkdir(exist_ok=True)
    labels = load_label_jsons()
    print(f"[1/5] frame inventory ...")
    write_frame_inventory(labels)
    print(f"[2/5] gt categories ...")
    cat_per_frame, cat_frame_presence = write_gt_categories(labels)
    print(f"[3/5] query matrix ...")
    write_query_gt_matrix(cat_frame_presence, labels)
    print(f"[4/5] visualization overlays ...")
    visualize_gt_overlays(labels)
    print(f"[5/5] camera distribution ...")
    write_camera_distribution()
    print("done. 산출물:")
    for p in sorted(MD_DIR.glob("ramen_*.md")):
        print(f"  - {p.relative_to(ROOT)}")
    print(f"  - {VIS_DIR.relative_to(ROOT)}/ ({len(list(VIS_DIR.glob('*.png')))} images)")


if __name__ == "__main__":
    main()
