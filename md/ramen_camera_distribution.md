# Ramen Camera Distribution

- 전체 등록된 카메라 수: **131**
- images 파일의 name 샘플: ['frame_00001.jpg', 'frame_00002.jpg', 'frame_00003.jpg'] ... ['frame_00129.jpg', 'frame_00130.jpg', 'frame_00131.jpg']

## Intrinsics

  - cam_id=1 model=PINHOLE 988x731 params=(777.7830824211188, 779.205588654199, 494.0, 365.5)

## 평가 7 프레임의 카메라 중심 좌표 (월드 좌표계)

| frame | Cx | Cy | Cz |
|---|---|---|---|
| frame_00006.jpg | +0.735 | -2.621 | +3.065 |
| frame_00024.jpg | +0.059 | -0.578 | +0.410 |
| frame_00060.jpg | +0.046 | -0.653 | -0.565 |
| frame_00065.jpg | -0.593 | -2.142 | +1.669 |
| frame_00081.jpg | +1.626 | +2.744 | -3.067 |
| frame_00119.jpg | -0.627 | +1.379 | -0.711 |
| frame_00128.jpg | -0.629 | +2.298 | -1.696 |

## 평가 프레임 간 viewpoint 각도 (scene centroid 기준)

| frame A | frame B | 각도 (deg) |
|---|---|---|
| frame_00006.jpg | frame_00024.jpg | 22.8 |
| frame_00006.jpg | frame_00060.jpg | 82.1 |
| frame_00006.jpg | frame_00065.jpg | 25.4 |
| frame_00006.jpg | frame_00081.jpg | 152.0 |
| frame_00006.jpg | frame_00119.jpg | 152.0 |
| frame_00006.jpg | frame_00128.jpg | 167.1 |
| frame_00024.jpg | frame_00060.jpg | 63.0 |
| frame_00024.jpg | frame_00065.jpg | 9.7 |
| frame_00024.jpg | frame_00081.jpg | 157.1 |
| frame_00024.jpg | frame_00119.jpg | 143.3 |
| frame_00024.jpg | frame_00128.jpg | 155.5 |
| frame_00060.jpg | frame_00065.jpg | 68.4 |
| frame_00060.jpg | frame_00081.jpg | 99.1 |
| frame_00060.jpg | frame_00119.jpg | 109.8 |
| frame_00060.jpg | frame_00128.jpg | 105.9 |
| frame_00065.jpg | frame_00081.jpg | 166.1 |
| frame_00065.jpg | frame_00119.jpg | 135.0 |
| frame_00065.jpg | frame_00128.jpg | 148.8 |
| frame_00081.jpg | frame_00119.jpg | 53.6 |
| frame_00081.jpg | frame_00128.jpg | 38.3 |
| frame_00119.jpg | frame_00128.jpg | 15.5 |

## 해석

- 평가 프레임 간 평균 시점 각도: **98.6°**
- 최소: **9.7°**, 최대: **167.1°**

→ 각도가 크면 서로 다른 시점 커버, 작으면 비슷한 각도에서 본 프레임이 섞여 있음.