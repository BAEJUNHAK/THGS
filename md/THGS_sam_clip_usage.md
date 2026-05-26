# THGS — SAM / CLIP 사용률(뷰 커버리지) 분석

> 질문: "CLIP(과 SAM)은 전체 뷰 중 몇 %를 쓰도록 알고리즘이 짜여 있나?"
> 대상: **ramen / LERF 설정**(`configs/lerf.yml`). 코드 라인 근거를 모두 명시한다.

---

## 0. TL;DR (핵심 결론)

1. **ramen 실제 설정에서는 SAM도 CLIP도, 등장하는 모든 단계에서 100% 뷰를 쓴다.** 뷰를 줄이는 로직이 활성화된 곳이 **하나도 없다.**
2. 뷰 서브샘플링 인자 2개(`view_freq=5`, `[::3]`)는 **둘 다 ramen에서 비활성/dead**다.
3. ⚠️ **정정**: 앞 대화에서 "SAM이 한 군데서 33%(`[::3]`)만 쓴다"고 한 것은 **`seg_enhance=True`일 때만** 해당한다. `lerf.yml`에는 `seg_enhance` 키 자체가 없어(launcher가 `hasattr` 체크) 그 분기는 **실행되지 않는다.** → ramen에선 SAM도 100%.
4. 비용의 본질: **SAM·CLIP 둘 다 전처리에서 단 한 번 100% 뷰로 계산**되고, 이후 단계는 그 출력을 **싸게 재사용(읽기)**할 뿐이다. 그중 **CLIP 인코딩(`encode_image`)이 호출 횟수·비용 모두 지배항**이다.

---

## 1. 파이프라인 단계별 SAM/CLIP/뷰 매트릭스 (핵심 표)

| # | 단계 | 스크립트(핵심 라인) | SAM | CLIP | 뷰 사용률 | 동작 |
|---|---|---|:---:|:---:|:---:|---|
| 0 | **전처리** | `image_encoding.py:361,389` (전체 이미지)<br>`:365-375` SAM, `:188` `encode_image` | ✅ **생성** | ✅ **인코딩** | **100%** | 뷰마다 SAM 4입도 마스크 → 마스크별 CLIP. **비용 전부 여기** |
| 1 | Step1 인접그래프 | `sp_partition.py:21` FRNN, `:145` `semantic=None` | ❌ | ❌ | (뷰 루프 없음) | 가우시안 xyz로 k-NN 그래프. 순수 기하 |
| 2 | Step2 엣지 재가중 | `graph_weight.py:77` `getTrainCameras()` | ✅ **재사용** | ❌ | **100%** | seg_map(1입도)→랜덤 20-d→ray trace. CLIP 안 씀 |
| 3 | Step3 분할(-k) | `sp_partition.py:34` cut pursuit | ❌ | ❌ | (뷰 루프 없음) | 재가중 거리로 그래프 컷. 순수 기하 |
| 4 | Step4a region growing | `merge_proj.py:242` `extract_gaussian_features` | ✅ **재사용** | ❌ | **100%** ×3레벨 | seg_map→랜덤 20-d→ray trace로 멀티뷰 라벨. CLIP 안 씀 |
| 5 | Step4b 피처 투영 | `merge_proj.py:283` `proj_gaussian_features_x` | ✅ seg_map 사용 | ✅ **읽기** | **100%** ×3레벨 | `_f.npy`(사전 CLIP) 읽어 SP에 집계. **encode 0회** |

**요약**:
- **SAM 등장 단계** = 전처리(생성) + Step2 + Step4a + Step4b(seg_map) → 전부 **100% 뷰**.
- **CLIP 등장 단계** = 전처리(인코딩) + Step4b(읽기) → 전부 **100% 뷰**.
- 기하 단계(Step1, Step3)는 SAM/CLIP 모두 미사용.

---

## 2. "사용률"의 두 가지 의미 (혼동 주의)

같은 100%라도 의미가 다르다:

| 의미 | 어디서 | 비용 | 뷰 |
|---|---|---|---|
| **(a) 생성/인코딩** | 전처리(`image_encoding.py`)에서만 | 💸 **무겁다** (SAM mask-gen, CLIP encode) | 100% (1회) |
| **(b) 출력 재사용(읽기)** | Step2/4a/4b | 싸다 (ray-trace 인덱싱·matmul) | 100% (여러 번) |

- SAM 마스크(`_s.npy`)는 **(a) 전처리 1회 생성** 후, Step2(1×) + Step4a(3×) + Step4b(3×)에서 **(b) ray-trace로 읽힘**. 재생성 없음.
- CLIP 피처(`_f.npy`)는 **(a) 전처리 1회 인코딩** 후, Step4b(3×)에서 **(b) 인덱싱으로 읽힘**. 재인코딩 없음.

→ **줄여야 할 것은 (a)의 CLIP encode**다. (b)의 100%는 싼 연산이라 의미 없음.

---

## 3. 뷰 서브샘플링 인자 — 왜 둘 다 dead인가 (정정 상세)

### 3-1. `view_freq` (= 5, "5뷰당 1뷰")
- `merge_proj.py:36` `self.view_freq = args.view_freq` 로 **저장만** 되고, 코드 어디에서도 사용되지 않음(`grep`로 확인). **dead parameter.**

### 3-2. `[::3]` (33%) — seg_enhance 전용, ramen에선 미실행
```python
# merge_proj.py:243-244
if args.seg_enhance and i == 0:
    upper_label = self.extract_gaussian_features(..., self.scene.getTrainCameras()[::3], i+2)
```
- 이 `[::3]`는 **`args.seg_enhance`가 True**여야 실행된다.
- launcher 게이트: `launcher.py:30` `if hasattr(cfg.merge_proj, 'seg_enhance') and cfg.merge_proj.seg_enhance:` 일 때만 `--seg_enhance` 전달.
- `configs/lerf.yml`의 `merge_proj` 섹션엔 `seg_enhance` 키가 **없음** → `hasattr=False` → 플래그 미전달 → `args.seg_enhance=False`(argparse 기본) → **분기 실행 안 됨.**
- 게다가 이 분기는 **CLIP 투영이 아니라 SAM 기반 region growing 라벨 추출**(상위레벨 보강)이라, 설령 켜져도 CLIP 비용과는 무관.

→ **결론**: ramen에서 SAM의 실효 뷰 사용률도 **100%**. "33%"는 다른 설정(seg_enhance on)에만 해당하는 예외였다.

---

## 4. SAM vs CLIP 비용 비대칭 (왜 타깃이 CLIP인가)

전처리에서 둘 다 100% 뷰지만, **호출 단위가 다르다**:

| | 호출 단위 | 장면당 횟수(V≈150~300뷰) |
|---|---|---|
| **SAM mask generation** | **뷰당 1회** (ViT-H AutomaticMaskGenerator) | ≈ V (150~300) |
| **CLIP `encode_image`** | **마스크당 1회** (뷰 × 4입도 × 마스크 수) | ≈ V × 150~300 = **2만~8만** |

- SAM은 뷰당 1번(무겁지만 V회), CLIP은 마스크당 1번 → **호출 수가 2~3 자릿수 더 많다.**
- 그래서 "뷰를 줄인다"의 실익이 가장 큰 쪽이 **CLIP**이고, 우리 절감 설계(Tier 1 + adaptive-k)의 표적이다.

---

## 5. "실효 사용률" — 100% 중 실제로 기여하는 비율

알고리즘은 100% 뷰를 돌지만, **한 슈퍼포인트 입장에서 실제 기여하는 뷰는 극히 일부**다. 게이트(`proj_gaussian_features_x`):

```python
gau_mask = weight > 0.0001          # :123  occlusion (가려진 가우시안 제외)
sp_mask_mat[sp_mask_mat < 0.3] = 0  # :133  overlap < 0.3 마스크 → 기여 0
portion = sp_gau_count / sp_gnum[sp]# :141  안 보이는 뷰일수록 가중치 ≈ 0
```

→ 어떤 SP의 512-d는 사실상 그 SP가 **잘 보이는 소수 뷰**의 가중평균으로 수렴. 나머지 뷰의 CLIP 인코딩은 **비용만 내고 0에 가깝게 희석/폐기**됨.

**측정 제안(Pass-1 첫 지표)**: SP별 `portion≥0.1`을 만족하는 뷰 수의 분포 → "100% 중 실효 기여 평균 N뷰 / 전체 V뷰 = M%". 이 M이 작을수록 낭비 가설이 강해진다. *(아직 ramen에서 측정 전 — 숫자는 미정.)*

---

## 6. 우리 절감 실험과의 연결

```
현재 baseline   : CLIP encode = 전처리 100% 뷰 × 모든 마스크   (감축 로직 0)
                  → 그중 실효 기여는 SP당 소수 뷰
목표(Tier1+adk) : encode를 merge_proj 뒤로 지연 + SP별 best-k 뷰만
                  → encode 호출 O(쿼리레벨 SP × k), 나머지 뷰는 애초에 인코딩 안 함
검증            : k ∈ {1,3,5,all} + adaptive 로 (호출수·시간·mIoU) A/B
```

- 핵심 빈틈: **CLIP 뷰 감축이 baseline에 전혀 없다(100%)** → 줄일 여지가 곧 0%→k뷰만큼 전부.
- 상세 알고리즘은 [THGS_clip_efficiency.md](THGS_clip_efficiency.md) §5–6 참조.

---

## 7. 논문(arXiv:2504.13153)은 이 문제를 다루는가 — "SR Time" 프레이밍

> 결론: **다루지 않는다.** 논문의 효율성 주장과 우리 절감 표적은 **서로 다른 축**이며, 논문의 시간 측정 방식(SR Time)이 우리가 줄이려는 단계를 **측정에서 제외**한다.

### 7-1. 논문이 안 다루는 것 (WebFetch로 본문 확인)
- ❌ CLIP image 인코딩 비용 (per-mask vs per-view, 타이밍 없음). 인코더로 "OpenCLIP ViT-B/16"만 명시.
- ❌ 피처 재투영에 전체 뷰를 쓰는지 일부만 쓰는지.
- ❌ 마스크 인코딩 중복(redundancy).
- ❌ best-view / keyframe 선택.

논문이 말하는 "efficient reprojection"(§3.3.2)은 **transmittance-weighted aggregation** — 즉 우리가 분석한 `merge_proj.py`의 `weight`/`portion` 가중 집계다. 이는 "학습(iterative optimization)을 안 한다"는 뜻이지 "CLIP을 덜 부른다"가 아니다.

### 7-2. 두 효율성 축 비교
| | 논문이 최적화 | 우리가 표적 |
|---|---|---|
| 대상 단계 | semantic field **구축**(병합+재투영) | CLIP **전처리 인코딩** |
| 방법 | 반복 학습 회피 → 결정론적 집계 | best-view만 인코딩(lazy + adaptive-k) |
| 효과 | **30×** (LERF: 85분→90초) | CLIP 호출 5~20× |
| 관계 | — | **경쟁 아님, 합쳐짐** (training-free + encode-free) |

### 7-3. "SR Time에서 뺀다"의 의미 (스톱워치 시점)
THGS 전체 = **① 전처리(SAM+CLIP)** → **② 구축(sp_partition→graph_weight→merge_proj)**.
논문이 보고하는 시간(**SR Time = Semantic Reconstruction Time**)은 **②만** 잰다. `_f.npy`/`_s.npy`가 디스크에 이미 만들어진 *다음*부터 스톱워치 ON.

```
[ ① 전처리 SAM+CLIP ]        →   [ ② 구축 ]
  ████ 무거움 (CLIP 2만~8만회)     ⏱ SR Time = 여기만 (90초/<2분, RTX3090)
  ↑ 시계 OFF (측정 제외)           ↑ 시계 ON
```

**왜 빼나(논문 입장)**: 비교군(LangSplat, LEGaussians)도 ①은 공통으로 필요. 차이는 ②(LangSplat 반복학습 85분 vs THGS 집계 90초). 공통 항 ①을 빼고 **자기 기여 ②만** 비교 → "30×"가 깔끔. 학술적으로 정당한 프레이밍.

| 방법 | ① 전처리 | ② 그 다음 |
|---|---|---|
| LangSplat | SAM+CLIP (공통) | 반복 학습 **85분** |
| LEGaussians | SAM+CLIP (공통) | 학습 **65분** |
| THGS | SAM+CLIP (공통) | 집계 **90초** = SR Time |
| 3DOVS·LangSplat | (공통) | 90분 / THGS **25초** |

### 7-4. 우리에게 주는 함의 — 측정 기준을 바꿔야 한다
우리 최적화는 **①(전처리) 안**에 있는데, 논문 스톱워치(SR Time)는 ①을 안 잰다.
```
CLIP 5~20× 절감 → ① 빨라짐 → 그러나 ②(SR Time)는 그대로 90초
                            → SR Time 지표만 보면 "변화 없음"으로 보임 ❌
```
→ 우리 이득을 드러내려면 보고 지표를 **(a) 전처리 시간(①) 단독** 또는 **(b) total = ①+② wall-clock** 으로 잡아야 한다. SR Time만으로는 기여가 안 보인다.

**측정 설계 메모**: ramen A/B에서 `① 전처리시간`, `CLIP encode 호출수`, `mIoU`, `(②는 거의 불변이라 참고용)`을 같이 로깅. 합격선은 mIoU −1%p 이내에서 ① 최대 절감.

---

## 부록 — 한눈에

```
                    뷰 사용률(ramen 실제)
전처리  SAM mask  ████████████ 100%  (생성, 무거움)
전처리  CLIP enc  ████████████ 100%  (인코딩, ★지배 비용)
Step1   기하       ─ (뷰 안 씀)
Step2   SAM trace ████████████ 100%  (재사용, 쌈)
Step3   기하       ─ (뷰 안 씀)
Step4a  SAM trace ████████████ 100% ×3  (재사용, 쌈)
Step4b  CLIP read ████████████ 100% ×3  (읽기, 쌈 / encode 0)

서브샘플링:  view_freq=5 → dead,  [::3] → seg_enhance off라 dead
```
