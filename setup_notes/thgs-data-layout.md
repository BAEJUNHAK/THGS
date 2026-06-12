---
name: thgs-data-layout
description: "Where THGS scenes, datasets and GT live locally, and the ready-to-use vs dataset split"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3dc409c0-3955-46c9-93b9-b33c5c495196
---

Local THGS data layout (LERF-OVS):

- **`output/lerf/<scene>/`** — author's *ready-to-use* scenes (2DGS + semantic field). Each has `point_cloud/iteration_30000/point_cloud.ply`, `cameras.json`, `cfg_args`, `input.ply`, and **`sai_nag.pt`** (the pipeline's final hierarchical superpoint graph + features). Because `sai_nag.pt` is already present, eval/query can run WITHOUT re-running the pipeline. Downloaded from the README "ready-to-use scenes" Google Drive folder.
- **`data/lerf_ovs/`** — LERF-OVS dataset (from LangSplat Drive file): per-scene COLMAP source (`images/`, `distorted/`, `sparse/`, `stereo/`) + **`label/<scene>/`** GT (frame_*.jpg + frame_*.json polygons). `data/lerf-ovs` and `data/lerf` are symlinks to `data/lerf_ovs` (config uses `data/lerf-ovs`, README examples use `data/lerf`).

Scenes: figurines, ramen, teatime, waldo_kitchen.

`test_lerf.py` derives the GT path as `dirname(source_path)/label/<scene>`, so it needs `-s data/lerf-ovs/<scene>`. Eval cmd:
`python test_lerf.py -s data/lerf-ovs/<sc> -m output/lerf/<sc> --path_pred output/render/lerf`
then `scripts/eval_seg.py --dataset lerf --scene_list ... --path_pred output/render/lerf --path_gt data/lerf-ovs/label`.

The 712MB dataset zip is kept at `data/lerf_ovs.zip`; scene zips staged under `_dl_scenes/` (can be deleted to reclaim space). See [[local-gpu-setup]].
