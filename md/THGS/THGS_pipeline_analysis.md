# THGS 파이프라인 완전 해부

> *Training-Free Hierarchical Scene Understanding for Gaussian Splatting with Superpoint Graphs* (Dai et al., arXiv:2504.13153, 2025) 의 README + 학습 코드 + 추론 코드 전수 분석

---

## 0. 큰 그림

THGS는 **학습 없이** 2DGS(2D Gaussian Splatting) 장면 위에 open-vocabulary 계층적 3D 분할을 얹는 방법이다. "Training-Free"라는 말이 핵심이다. THGS 자체는 어떤 신경망도 학습시키지 않는다. 사용하는 학습된 모델은 외부에서 가져오기만 한다:

| 외부 모델 | 역할 | 주체 |
|---|---|---|
| 2DGS | RGB 장면 재구성(서펠 가우시안) | 사용자가 별도로 사전학습 |
| SAM (ViT-H) | 2D 마스크 생성기 | 공식 체크포인트 |
| OpenCLIP (ViT-B/16, laion2b) | 시맨틱 피처 인코더 | 공식 체크포인트 |
| Parallel Cut Pursuit | 슈퍼포인트 분할 그래프 컷 | SPT 라이브러리 |

THGS의 **"학습 파이프라인"** 이라 부르는 것은 위 모델들의 출력을 조합해 `sai_nag.pt`(계층적 슈퍼포인트 그래프 + 시맨틱 피처)를 **빌드(build)** 하는 과정이며, gradient descent 학습이 아니라 결정론적 후처리에 가깝다.

전체 흐름:

```
[Step 0] 2DGS 장면 학습 (외부) ─┐
                               ├─→ point_cloud.ply (가우시안 N개)
[Step 0] image_encoding.py ────┘     + language_features/*_s.npy, *_f.npy
                                          (SAM seg_maps + CLIP feats)
                ↓
[Step 1] sp_partition.py (-k 없음): FRNN K-NN → neighbor.pt
                ↓
[Step 2] graph_weight.py: SAM 단서로 edge 재가중 → neighbor_new.pt
                ↓
[Step 3] sp_partition.py (-k 사용): Cut Pursuit → nag-l1.pt
                ↓
[Step 4] merge_proj.py: SAI3D 병합 + CLIP 재투영 → sai_nag.pt
                ↓
[추론] test_lerf.py / gui/main.py: 텍스트 쿼리 → 가우시안 마스크 → 렌더링
```

---

## 1. Step 0: 전처리

### 1-A. 2DGS 학습 (외부)

`README` Step 1: 사용자가 [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) 레포에서 별도로 학습한다. 결과물 `output/<scene>/point_cloud/iteration_30000/point_cloud.ply` 를 본 프로젝트의 `output/<save_folder>/<scene>/point_cloud/iteration_30000/` 에 복사한다.

PLY가 가지는 속성: `x, y, z`, `f_dc_0..2`, `f_rest_*` (SH 계수, sh_degree=3 → 15개×3채널), `rot_0..3` (사원수). `sp_partition.load_ply()` 가 이걸 읽어 `xyz`, `colors_precomp`(SH→RGB), `normal`(rotation의 third row)을 만든다. 즉 **"학습 코드"가 만드는 점군의 입력 표현은 위치+RGB+법선 3종**이며 이게 슈퍼포인트 분할의 feature가 된다.

`scene/dataset_readers.readColmapCameras()` 는 추가로 `language_features/*_s.npy`(SAM seg_map), `*_f.npy`(CLIP feat)를 읽어 각 카메라의 `semantic = {'seg_map': (4,H,W), 'fg_mask': (4,H,W), 'sem': (N,512)}` 에 채운다.

### 1-B. SAM + CLIP 언어 피처 (`scripts/image_encoding.py`)

LangSplat과 호환되는 포맷. 한 이미지당 4개 입도(default/s/m/l)의 SAM 마스크를 만들고, 각 마스크 영역을 잘라 224×224 패딩 → OpenCLIP ViT-B/16 으로 512차원 임베딩.

핵심 흐름:

1. `SamAutomaticMaskGenerator(points_per_side=32, pred_iou_thresh=0.7, stability_score_thresh=0.85, crop_n_layers=1, ...)` → 4입도 마스크
2. `mask_nms(iou_thr=0.8, score_thr=0.7, inner_thr=0.5)` 로 중복/포함 마스크 NMS
3. `mask2segmap`: 마스크별 bbox crop → 정사각 패딩 → 224 리사이즈 → CLIP image encoder → L2 정규화
4. 각 픽셀에 마스크 인덱스를 기록한 `seg_map` (default/s/m/l 각각 (H,W) int32, -1은 배경)
5. 4 레벨을 stack → `(4, H, W)` 텐서, 인덱스를 cumulative offset 으로 재매핑 (level 1+의 인덱스가 level 0의 끝에 이어 붙도록)
6. 저장: `<image_name>_s.npy` (seg_maps, (4,H,W)), `<image_name>_f.npy` (features, (N_total, 512))

따라서 `seg_map[level]`의 각 정수 ID는 **해당 레벨 내에서 마스크 ID**가 아니라 **누적 ID**이며, `feature_map[id]`로 바로 CLIP 피처를 참조할 수 있다.

이 단계가 한 번 끝나면 `data/<dataset>/<scene>/language_features/` 에 이미지 수의 2배 만큼의 `.npy`가 생긴다.

---

## 2. Step 1: 인접 그래프 구축 (`sp_partition.py` / `-k` 없음)

`pre_knn(data, ...)` 함수:

1. `load_ply` 로 `data.pos`(xyz), `data.rgb`, `data.normal` 채운 `Data` 객체 생성. `pos_center=[0,-1,3.5]` 가 기본 (장면 중심), `--centered` 면 [0,0,0].
2. SPT의 `cfg=init_config(overrides=['experiment=semantic/scannet'])` 로 ScanNet semantic 실험 cfg 로드 → `instantiate_datamodule_transforms(cfg.datamodule)` 가 두 변환 dict를 반환:
   - `knn_transform`: KNN 그래프 구축
   - `cut_transform`: 그래프 컷
3. `nag = transforms_dict['knn_transform'](data)` 가 `data.pos` 에 대해 **FRNN(Fast Radius Nearest Neighbors)** 으로 K-NN 인접 정보를 계산.
4. 결과 `neighbors`(N×K long), `distances`(N×K float) 를 `<model_path>/neighbor.pt` 로 저장.

여기서 K는 SPT cfg에 따라 보통 25 정도. 음수 인덱스가 발생하면 경고를 출력 (radius 안에 K개가 안 잡힌 경우).

이 그래프는 **유클리드 거리 기반**이라 가우시안끼리 공간적으로 가까우면 같은 슈퍼포인트로 묶이게 되어 객체 경계를 넘는 잘못된 병합이 생긴다. Step 2가 이걸 SAM 단서로 깎는다.

---

## 3. Step 2: SAM-Guided 엣지 재가중 (`graph_weight.py`)

핵심 아이디어: **같은 SAM 마스크에 속하는 가우시안 사이 거리는 줄이고, 다른 마스크에 속하면 거리를 늘린다.** 그래야 cut pursuit이 마스크 경계에서 자르도록 유도된다.

### 3-A. 입력
- `<model_path>/neighbor.pt` → `knn`(N×K), `ori_distance`(N×K)
- 학습 카메라 + 각 카메라의 SAM `seg_map[feature_level]`, `fg_mask[feature_level]`
- 기본 `feature_level=1` (s 입도, 가장 작은 마스크) — 가장 세밀한 분할에 맞춤

### 3-B. 한 뷰당 처리 (`extract_gaussian_features`)
1. 해당 뷰의 `seg_map`(H,W)을 받고, 배경 픽셀(`~fg_mask`)을 마지막 인덱스(`seg_num-1`)로 처리. 마스크 개수가 `seg_num`이라 하자.
2. **무작위 직교 임베딩** `enc ~ N(0,1)`, shape `(seg_num, 20)`, L2 정규화. 마지막(배경) 행에 `zero_scale=0.2`을 곱해 영향력을 줄임.
3. `feature_map = enc[seg_map]` → (H,W,20). 각 픽셀이 자기 마스크의 unit vector를 가짐.
4. `trace(view, gaussians, feature_map, None, ...)` 호출 — diff-surfel-rasterizer의 **역방향 ray-tracing** 모드. 픽셀의 20-D 피처를 카메라 ray를 따라 가우시안에 누적. 결과:
   - `gau_sem`: (N,20), 각 가우시안이 받은 가중평균 피처
   - `gau_depth`: (N,) 가시 깊이
   - `num_ray`: (N,) 그 가우시안을 본 픽셀 수
5. `gau_sem`을 L2 정규화 → `enc.T`와 코사인 유사도 → argmax 로 가우시안 별로 **이 뷰에서 자기가 속한 SAM 마스크 ID** 결정. `sim_max < tau(0.85)` 인 가우시안은 신뢰 안 함 → 마지막 인덱스로 처리.
6. **valid_gau** = 본 적이 있고(`num_ray>0`), 신뢰 가능하고(`sim_filter`), 배경이 아닌 가우시안.
7. KNN 엣지 검사: `knn_label = gau_sem[knn[valid_gau]]`(N×K), 비교:
   - 라벨이 다르면 → `neg_dist += depth_weight × 1`
   - 라벨이 같으면 → `pos_dist += depth_weight × 1`
   - `depth_weight = 1/(1+ReLU(depth-5))` (가까운 가우시안일수록 가중치 큼, 멀어지면 1/(1+(d-5)) 로 감쇠)

여러 뷰에 걸쳐 누적된 `neg_dist`, `pos_dist`는 각 엣지의 "다르다는 증거"와 "같다는 증거"의 합이다.

### 3-C. 거리 업데이트
```python
neg_dist = clamp(neg_dist, 0, neg_b=25)
pos_dist = clamp(pos_dist, 0, pos_b=25)
new_distance = ori_distance + neg_dist*neg_w(0.1) - pos_dist*pos_w(0.02)
new_distance = clamp(new_distance, min=0)
torch.save({'neighbors': knn, 'distances': new_distance}, 'neighbor_new.pt')
```

가중치 `neg_w=0.1`이 `pos_w=0.02`보다 5배 큰 점에 주목: **엣지를 끊는 쪽이 묶는 쪽보다 5배 강함**. 즉 "다르다는 증거"가 누적되면 빠르게 큰 distance가 되어 cut pursuit 이 그 엣지를 자른다.

`tau`, `zero_scale`, `neg_b/pos_b`, `neg_w/pos_w`, `level`은 `configs/lerf.yml`의 `graph_weight` 섹션에서 조절. LERF 기본은 tau=0.85, zero_scale=0.2; 3DOVS는 tau=0.5, zero_scale=1.0 으로 더 관대하다.

---

## 4. Step 3: 슈퍼포인트 분할 (`sp_partition.py` / `-k neighbor_new.pt`)

`partition()` 함수:

1. `neighbor_new.pt` 로드 → `data.neighbor_index`, `data.neighbor_distance` 에 주입.
2. `transforms_dict['cut_transform'](data)` → SPT의 `instantiate_datamodule_transforms`가 빌드한 변환 체인. 내부에서 **Parallel Cut Pursuit** 알고리즘이 돈다.
   - `pcp_regularization` (LERF 0.1, 3DOVS 0.3): 클수록 정규화 강해 슈퍼포인트 적어짐
   - `pcp_spatial_weight` (LERF 1e-1, 3DOVS 2e-1): 작을수록 공간 가중치 줄여 슈퍼포인트 적어짐
3. SPT는 다단계 NAG(Nested Adjacency Graph) 를 만든다 — level 0(원래 가우시안), level 1, level 2, level 3 의 슈퍼포인트가 hierarchical 하게 나옴.
4. `nag.get_super_index(1, 0)` → level 0(가우시안)이 속한 level 1 슈퍼포인트 ID — 이걸 `nag-l1.pt`로 저장.
5. `--verbose` 면 level 2, 3 도 저장.

여기서 **첫 번째 계층**만 만든다. Level 2, 3은 Step 4에서 SAI3D 방식으로 다시 구축된다 (graph cut의 위 계층은 사용 안 함).

---

## 5. Step 4: 계층적 병합 + 시맨틱 재투영 (`merge_proj.py`)

이 스크립트가 가장 길고 복잡하다. SAI3D ([arXiv:2312.11557](https://arxiv.org/abs/2312.11557)) 기반의 progressive region-growing 을 가우시안에 맞게 변형했다.

### 5-A. 초기화 (`SAI3D.init_data`)
- `self.scene` 로 가우시안 + 카메라 로드
- `self.seg_ids = torch.load('nag-l1.pt')` (가우시안 → level-1 SP)
- `self.points = gaussians._xyz`
- KDTree 8-NN: `self.points_neighbors[N, 8]` — Step 1의 FRNN과 별개, 가우시안 기준 좁은 KNN
- `print("Points num:", N, "views num:", M, "sp num:", seg_ids.max()+1)`

### 5-B. 메인 루프 (`assign_label`)

`thres_connect = [0.9, 0.7, 0.7]` (LERF 기본). 즉 **3단계 병합** 으로 level 1→2→3→4 까지 만든다 (nag 길이 4: [lvl0, lvl1, lvl2, lvl3] — 코드에서는 nag=[seg_ids] 로 시작해 매 iter마다 append).

각 iter `i`에서:

#### (1) 가우시안별 다중 뷰 라벨 추출 (`extract_gaussian_features`)
- feature_level = `i+1` (i=0 → level 1 즉 's' 입도, i=1 → 'm', i=2 → 'l')
- 다시 `trace`로 SAM 마스크 ID를 가우시안에 매핑. `sim > 0.85` 임계로 필터.
- 출력: `pt_sp_label : (N, M_views)` — 각 가우시안이 각 뷰에서 본 SAM 마스크 ID.
- `seg_enhance` 옵션 (3DOVS) 이면 첫 iter에 추가로 `[::3]` 뷰들로 더 큰 입도(level i+2)를 뽑아 concat → 라벨 다양성 증대.

#### (2) 슈퍼포인트 인접 그래프 (`get_seg_data_torch`)
- `seg_members`: 각 SP에 속한 가우시안 인덱스
- `seg_direct_neighbors`: KDTree-8NN 따라 이웃하는 SP 쌍 (대칭화)
- `seg_indirect_neighbors[d, S, S]`: distance ≤ d+1 만큼 떨어진 SP 쌍 (d=0..max_neighbor_distance-1, 기본 max=2 → 직접+간접 2단계)

#### (3) 유사도 행렬 (`get_seg_adjacency`)
- 각 뷰 m, 각 SP 쌍 (i,j) 에 대해:
  - `seglabels[s, l]` = SP s 안에서 라벨 l을 가진 가우시안 비율 (정규화). `2-norm` 메트릭이라 cosine similarity와 등가
  - `similar(i,j) = (2 - ||seglabels[i] - seglabels[j]||²) / 2`
  - confidence = `seg_seen[i,m] × seg_seen[j,m]` (각 SP가 그 뷰에서 얼마나 많이 보였는지의 곱)
- 모든 뷰에 걸쳐 가중 평균: `adj[i,j] = Σ_m similar*confidence / Σ_m confidence`. (`get_seg_adjacency_from_similar_confidence_torch`)
- 최종 adj은 대칭화 (max(adj[i,j], adj[j,i])).

#### (4) Region growing (`assign_seg_label_torch`)
- BFS with deque. 미할당 SP에서 시작해 같은 region label로 인접 SP를 흡수.
- `judge_connect_torch_opt`: 후보 이웃 j에 대해
  - 가중치 `weight = decay^d`, `decay=0.5` — 직접이웃은 1, 1단계 멀면 0.5, 2단계 멀면 0.25
  - score = `(weight × adj[j, neighbors_in_region] × member_count).sum() / (weight × member_count).sum()`
  - score ≥ `thres_connect[i]` 면 j를 같은 region에 합친다.
- 결과: `seg_labels[S]` — 각 SP가 어느 region(=상위 계층 SP) 에 속하는지.

#### (5) 작은 region 병합 (마지막 iter만, `merge_small_segs_torch`)
- region이 가진 SP 수 ≤ 2 이고 전체 가우시안 수가 `thres_merge=20` 미만이면 "작은 region"으로 표시
- 각 작은 region을 인접도 높은 큰 region으로 흡수 (반복 수렴까지)
- 흡수 못 한 건 라벨 0(invalid)으로

#### (6) 가우시안에 region 라벨 전파
- `pt_prim_label[N]`: 각 가우시안이 속한 새 region. `nag.append(pt_prim_label - 1)` (label은 1부터 시작하므로 -1 해서 NAG에 저장)
- `self.seg_ids = pt_prim_label` 로 갱신 → 다음 iter는 이 새 SP 위에서 다시 병합

#### (7) CLIP 시맨틱 피처 재투영
두 가지 방식 (config의 `feat_assign`):

##### feat_assign = 1 (`proj_gaussian_features`)
- 각 뷰에서 `render_point` (가우시안 중심을 카메라에 투영해 2D 픽셀 좌표 + weight 획득)
- `means2D[gau_mask]` 의 픽셀 위치에서 SAM seg_map 의 라벨 lookup → 그 SAM 마스크의 CLIP 피처를 가져옴 → `seen_gau_sem[gau] = significance × fg_mask × CLIP_feat[seg_id]`
- 슈퍼포인트별로 `gau_count/sp_size` 비율(<10% 면 무시)로 가중평균, 누적
- 최종 sp_feature[S, 512]

##### feat_assign = 2 (`proj_gaussian_features_x`, LERF 기본)
- 슈퍼포인트 × SAM 마스크 인접도 행렬을 만든다: `sp_mask_mat[s, m]` = SP s 안에서 SAM 마스크 m 픽셀 비율 (0.3 이하는 0으로 clip)
- `sp_feat = sp_mask_mat @ view_level_feature` — SP의 피처 = 자기에 가장 많이 겹친 SAM 마스크 피처들의 가중 합
- 가시성 비율로 누적 → 최종 L2 정규화

LERF는 방식 2, 3DOVS는 방식 1 사용. 방식 2가 픽셀-라벨 분포 기반이라 더 안정적이라는 게 코드 흐름의 의도.

각 iter 끝에 `sp_feature[S_lvl, 512]`를 `nag_features` 에 append.

### 5-C. 저장
```python
torch.save({'nag': nag, 'nag_feat': nag_feat}, 'sai_nag.pt')
utils.save_label_ply(points, labels_fine_global, '<scene>saifine.ply')
```

`sai_nag.pt`의 구조:
- `nag`: `[lvl0_label, lvl1_label, lvl2_label, lvl3_label]` — 길이 4. 각 텐서는 (N,) shape으로 가우시안이 그 레벨의 어느 SP에 속하는지.
- `nag_feat`: `[lvl1_feat, lvl2_feat, lvl3_feat]` — 길이 3. 각 (S_lvl, 512). lvl0(개별 가우시안)에는 피처 없음 (CLIP은 SP 단위로만 의미 있음).

이게 THGS의 **최종 산출물** 이며 추론 시 이 한 파일만 있으면 된다.

---

## 6. 추론 파이프라인

### 6-A. 정량 평가 (`test_lerf.py`)

```
input: <model_path>/sai_nag.pt, <data_path>/label/<scene>/*.json
loop scene → loop image → loop prompt:
  1) vlm.encode_text(prompt) → CLIP text feat (+ canon negatives)
  2) for each level feat in nag_feat: cos sim
  3) snag.get_related_gaussian(sims, topk=3, level=[2,3])
  4) gaussians._semantics = mask.expand(20)
  5) render(cam) → 2D semantic map
  6) mask = sem[0] > 0.5 → reshape (H,W) → save .png
  7) GT polygon → mask → save _gt.png
```

핵심은 **`SemanticNAG`** (`nag_data.py`):

#### `build_nag_from_multilevel_labels`
- 입력 `[lvl1, lvl2, lvl3]` (len=3 in inference, but loaded NAG has 4 entries — top level은 dummy)
- SPT의 `Cluster(pointers, points, dense=False)` (CSR-style 클러스터링)을 단계별로 빌드:
  - lvl0→lvl1: 모든 가우시안이 자기 SP에 속한다는 mapping을 정렬 → bincount 로 cluster size → cumsum 으로 pointer
  - lvl_i → lvl_{i+1}: lvl_i 의 unique label 별로 lvl_{i+1} label을 추출, 다시 정렬+bincount+cumsum
- 결과: `NAG(data_list)` — 양방향 탐색 가능한 hierarchical graph

#### `get_related_gaussian(sims, topk=3, level=[2,3])`
- `sims = [sim_lvl1, sim_lvl2, sim_lvl3]` — 각 SP의 텍스트 유사도
- `level=[2,3]` 이면 lvl2, lvl3 의 SP만 후보
- 각 레벨에서 `topk(sim, k=topk)` 로 후보 SP 모음 → 모든 후보를 sim 내림차순 정렬 후 상위 topk 선택 (level 섞어가며)
- 선택된 (level, sp_index) 쌍에 대해 `self.labels[level] == index` 인 가우시안에 1 부여
- 출력: `(N, 1)` 0/1 마스크

#### CLIP 유사도 (`vlm_utils.ClipSimMeasure.compute_similarity`)
- text feature = [positive] + canon negatives ("object", "things", "stuff", "texture") 5개
- semantic_feature(SP feat 512-d) @ text_features.T → (S, 5) logit
- positive vs 각 negative 쌍을 softmax → positive prob 가장 낮은 negative 선택 → 그 쌍의 positive prob 반환
- = LangSplat / LERF 의 "softmax-with-canonicals" 점수, 분포가 sharp 해서 0/1 임계가 잘 작동

#### Render
- `gaussians._semantics = point_valid.expand(-1, 20)` — binary 마스크를 20채널로 broadcast
- `render(cam, gaussians, pipe, bg)` 의 `rendered_sem`(20,H,W) 의 첫 채널만 사용 → > 0.5 임계로 binary mask
- diff-surfel-rasterizer 가 Gaussian의 semantic 채널을 자체 알파 누적 — 따라서 occlusion 자동 처리

### 6-B. 평가 (`scripts/eval_seg.py`)
- 픽셀 기준 IoU/Dice/Precision/Recall/F1/Acc 를 prompt 단위로 계산 → 이미지 평균 → 씬 평균 → 데이터셋 평균.
- LERF: GT 폴리곤 JSON, 3DOVS: GT PNG.

### 6-C. GUI (`gui/main.py`)
- DearPyGui 기반 인터랙티브 뷰어. `OrbitCamera` 로 자유 시점, 텍스트 입력 → CLIP 인코딩 → 같은 SemanticNAG 흐름.
- 추가로 DBSCAN 후처리, point-prompt 분할, 비디오 생성 기능 일부.
- **주의**: README의 TBD 표시처럼 인터랙티브 분할 기능은 미완성.

---

## 7. 데이터 흐름 한눈에 정리

| 산출물 | 위치 | 단계 | 형태 |
|---|---|---|---|
| `point_cloud.ply` | `output/.../point_cloud/iteration_30000/` | Step 0 (외부) | (N, 3+45+4) |
| `*_s.npy` `*_f.npy` | `data/.../language_features/` | Step 0 image_encoding | (4,H,W) int + (M,512) fp16 |
| `neighbor.pt` | `output/.../<scene>/` | Step 1 sp_partition | dict{neighbors:(N,K), distances:(N,K)} |
| `neighbor_new.pt` | 동일 | Step 2 graph_weight | 동일 구조, 거리만 갱신 |
| `nag-l1.pt` | 동일 | Step 3 sp_partition -k | (N,) long, level-1 SP id |
| `sai_nag.pt` | 동일 | Step 4 merge_proj | dict{nag:[4×(N,)], nag_feat:[3×(S_lvl,512)]} |
| `<scene>saifine.ply` | 동일 | Step 4 (디버그용) | 색칠된 ply |
| 예측 마스크 PNG | `output/render/lerf/<scene>/<image>/<prompt>.png` | 추론 test_lerf | (H,W) uint8 |

---

## 8. 핵심 하이퍼파라미터 요약

### LERF (configs/lerf.yml)
```yaml
graph_weight:  tau=0.85  zero_scale=0.2  neg_w=0.1  pos_w=0.02  neg_b=pos_b=25  level=1
merge_proj:    thres_connect=0.9,0.7,0.7  thres_merge=20  feat_assign=2
spt:           pcp_regularization=0.1  pcp_spatial_weight=1e-1  aligned_normal=True
```

### 3DOVS (configs/3dovs.yml)
```yaml
graph_weight:  tau=0.5   zero_scale=1.0  나머지 동일
merge_proj:    thres_connect=0.9,0.7,0.7  thres_merge=100  seg_enhance=True  feat_assign=1
spt:           pcp_regularization=0.3  pcp_spatial_weight=2e-1
```

3DOVS는 LERF보다 장면이 단순해 더 강한 정규화(`pcp_regularization` ↑) + 더 큰 작은-region 병합 임계(`thres_merge` 100 vs 20) + SAM 단서를 더 관대하게(tau ↓, zero_scale ↑) 사용. 추가로 `seg_enhance=True` 로 다중 입도 라벨링.

추론에서는 `topk=3`, `level=[2,3]` 이 하드코딩 (test_lerf.py L57). lvl 1은 너무 잘게, lvl 0은 가우시안 단위라 의미 없음. 실험적으로 lvl 2-3 이 객체/파트 입도에 잘 맞는 것으로 보임.

---

## 9. "Training-Free"라는 주장의 정확한 의미

THGS는 다음을 **하지 않는다**:
- 어떤 MLP/feature head 도 학습하지 않음
- gradient descent 루프 없음
- per-scene 최적화 없음 (SH 최적화는 외부 2DGS 단계)

THGS가 **하는 것**:
- 사전학습된 모델(SAM, CLIP, 2DGS) 의 출력을 결정론적 알고리즘(graph cut, region growing)으로 fuse
- 결과를 `.pt` 파일에 저장
- 추론 시 이 파일을 읽어 텍스트 쿼리에 응답

→ 학습 시간이 거의 없고(2DGS 단계 제외하면 분 단위), 새로운 prompt 도 즉시 처리 가능. 반대로 SAM/CLIP 의 한계(경계 흐림, 작은 객체 누락)를 그대로 물려받음.

---

## 10. 자주 헷갈리는 포인트

1. **"Step 1 sp_partition" 과 "Step 3 sp_partition" 은 같은 스크립트, 다른 모드**. `-k` 플래그가 전부를 가른다. 없으면 KNN 만, 있으면 cut pursuit 까지.
2. **`nag-l1.pt` 는 SPT-기반 슈퍼포인트, 그 위 계층(lvl 2-3) 은 SAI3D 기반**. 두 알고리즘의 역할이 분리돼 있다. SPT는 색·법선 기반 oversegmentation, SAI3D는 시맨틱 기반 region growing.
3. **`graph_weight.py` 의 zero_scale 은 배경에만 적용**. 배경 픽셀의 영향력을 줄이는 트릭. 3DOVS는 1.0 으로 두는데, 3DOVS는 배경이 의미 있는 wall/floor 클래스라 그렇다.
4. **`diff_surfel_rasterizer` 의 `trace` 는 forward render 가 아니라 inverse projection**. 픽셀 피처를 가우시안에 누적하는 함수. THGS만의 기능이라 submodule 도 `submodules/diff-surfel-rasterization` (포크 버전).
5. **추론 시 `gaussians._semantics` 는 binary mask 의 broadcast**. 진짜 시맨틱 임베딩이 아니다. 단지 rasterizer 의 alpha-composite 능력으로 occlusion-aware 2D mask 를 얻기 위한 트릭.

---

## 11. 한 줄 요약

> THGS = 2DGS 점군 위에 SPT(공간 cut pursuit) 로 작은 슈퍼포인트를 만들고, SAI3D(다중뷰 SAM-라벨 분포 region growing) 로 계층 병합하면서 CLIP 피처를 재투영해, 추론 시 텍스트 쿼리 → 코사인 유사도 → topk SP → 가우시안 마스크 → 2D 렌더링 으로 open-vocabulary 분할을 해내는 학습 없는 파이프라인.
