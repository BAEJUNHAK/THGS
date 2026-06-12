# Ramen GT 카테고리 전수 조사

- 전체 GT 프레임: **7** 장
- unique category 수: **14** 개
- 총 polygon instance 수: **80** 개

## 전체 카테고리 리스트 (alphabetical)

| # | Category | 전체 등장 횟수 (polygon) | 프레임 커버리지 |
|---|---|---|---|
| 1 | `bowl` | 7 | 5/7 |
| 2 | `chopsticks` | 7 | 7/7 |
| 3 | `corn` | 5 | 5/7 |
| 4 | `egg` | 7 | 7/7 |
| 5 | `glass of water` | 2 | 2/7 |
| 6 | `hand` | 1 | 1/7 |
| 7 | `kamaboko` | 7 | 7/7 |
| 8 | `napkin` | 7 | 5/7 |
| 9 | `nori` | 6 | 6/7 |
| 10 | `onion segments` | 7 | 7/7 |
| 11 | `plate` | 6 | 4/7 |
| 12 | `sake cup` | 8 | 6/7 |
| 13 | `spoon` | 3 | 2/7 |
| 14 | `wavy noodles` | 7 | 7/7 |

## 프레임별 카테고리

- **frame_00006** (10 objects): `chopsticks`, `egg`, `nori`, `bowl`, `napkin`, `sake cup`, `wavy noodles`, `kamaboko`, `plate`, `onion segments`
- **frame_00024** (8 objects): `bowl`, `chopsticks`, `egg`, `nori`, `wavy noodles`, `kamaboko`, `onion segments`, `corn`
- **frame_00060** (8 objects): `chopsticks`, `egg`, `sake cup`, `napkin`, `wavy noodles`, `kamaboko`, `corn`, `onion segments`
- **frame_00065** (12 objects): `bowl`, `egg`, `chopsticks`, `sake cup`, `wavy noodles`, `nori`, `napkin`×2, `kamaboko`, `plate`, `corn`, `onion segments`
- **frame_00081** (16 objects): `bowl`×2, `chopsticks`, `sake cup`×2, `nori`, `egg`, `wavy noodles`, `glass of water`, `kamaboko`, `spoon`, `napkin`×2, `plate`×2, `onion segments`
- **frame_00119** (8 objects): `nori`, `egg`, `sake cup`, `chopsticks`, `wavy noodles`, `kamaboko`, `corn`, `onion segments`
- **frame_00128** (18 objects): `glass of water`, `nori`, `egg`, `sake cup`×2, `bowl`×2, `chopsticks`, `wavy noodles`, `spoon`×2, `corn`, `onion segments`, `hand`, `plate`×2, `kamaboko`, `napkin`

## q.md Level 0 카테고리 대조

| q.md 쿼리 | LERF-OVS GT 존재? | 전체 등장 | 프레임 커버리지 |
|---|---|---|---|
| `bowl` | ✓ | 7 | 5/7 |
| `chopsticks` | ✓ | 7 | 7/7 |
| `corn` | ✓ | 5 | 5/7 |
| `egg` | ✓ | 7 | 7/7 |
| `eggs` | ✗ (없음) | — | 0/7 |
| `glass of water` | ✓ | 2 | 2/7 |
| `hand` | ✓ | 1 | 1/7 |
| `kamaboko` | ✓ | 7 | 7/7 |
| `nori` | ✓ | 6 | 6/7 |
| `plate` | ✓ | 6 | 4/7 |
| `sake cup` | ✓ | 8 | 6/7 |
| `spoon` | ✓ | 3 | 2/7 |
| `wavy noodles` | ✓ | 7 | 7/7 |

### 요약

- q.md Level 0 13개 중 실제 GT 있음: **12**개
- q.md에 있으나 GT 없음: **['eggs']**
- GT에 있으나 q.md에 없음: **['napkin', 'onion segments']**