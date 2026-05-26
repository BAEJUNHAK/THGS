# Ramen Frame Inventory

- 전체 학습용 이미지 수: **131장** (`frame_00001.jpg` ~ `frame_00131.jpg`)
- GT 평가 프레임 수: **7장**
- 이미지 저장 위치: `data/lerf-ovs/ramen/images`
- GT 라벨 저장 위치: `data/lerf-ovs/label/ramen`

## 해상도 분포

| (W x H) | 프레임 수 |
|---|---|
| 988 x 731 | 131 |

## GT 평가 프레임 목록

| frame | image size (w,h) | GT bbox count | JSON filesize |
|---|---|---|---|
| frame_00006 | 988 x 731 | 10 | 109397 bytes |
| frame_00024 | 988 x 731 | 8 | 138422 bytes |
| frame_00060 | 988 x 731 | 8 | 76257 bytes |
| frame_00065 | 988 x 731 | 12 | 126047 bytes |
| frame_00081 | 988 x 731 | 16 | 133491 bytes |
| frame_00119 | 988 x 731 | 8 | 79328 bytes |
| frame_00128 | 988 x 731 | 18 | 179050 bytes |

## q.md와의 일치성

q.md가 상정한 7개 프레임: `frame_00006, 00024, 00060, 00065, 00081, 00119, 00128`

**결과**: 완전 일치 ✓ — LERF-OVS 공식 GT에 q.md의 7 프레임이 모두 있음.

> q.md가 Ref-LERF 기반이라고 명시했지만, LERF-OVS 공식 zip에 이미 동일한 7 프레임이 들어있어 Ref-LERF 없이도 동일 평가가 가능.