# HaMeR Pipeline Debug Notes

## 목적

이 브랜치의 목적은 기존 `entry_point.py` 파이프라인은 유지하면서, HaMeR 손 추적 결과가 튀거나 잘못 합쳐지는 문제를 별도로 실험하고 디버깅하는 것이다. 새 실험 엔트리포인트는 `entry_point_hamer.py`로 두었고, 손 파라미터 후처리와 raw HaMeR 결과 검증 도구를 추가했다.

핵심 문제는 크게 세 가지였다.

- HaMeR가 손이 아닌 위치를 손으로 잘못 잡아 mesh를 생성하는 경우
- raw HaMeR에서는 손이 잘 펴져 보이는데 merged SMPL-X 결과에서는 손이 계속 쥐어진 것처럼 보이는 경우
- 손이 잠깐 안 보이거나 잘못 보일 때 MANO 파라미터가 프레임 사이에서 크게 튀는 경우

## 현재 파이프라인 개요

기본 흐름은 다음과 같다.

```text
input video
-> GVHMR body 추정
-> HaMeR hand 추정
-> HaMeR MANO hand pose를 SMPL-X hand pose에 병합
-> hand postprocess
-> merged SMPL-X 렌더링
```

원본 엔트리포인트인 `entry_point.py`는 최대한 유지하고, 실험용 흐름은 `entry_point_hamer.py`에서 돌린다. HaMeR 출력은 기존 결과와 섞이지 않도록 `hamer_out_clean`을 사용한다.

주요 결과물 예시는 다음과 같다.

```text
outputs/cap_pipeline/back_2/smplx_merged_hamer_post.pt
outputs/cap_pipeline/back_2/smplx_merged_hamer_post.json
outputs/cap_pipeline/back_2/smplx_merged_hamer_incam.mp4
outputs/cap_pipeline/back_2/hamer_out_clean/
outputs/cap_pipeline/back_2/frames_hamer_raw/
```

## 새 엔트리포인트

추가한 파일:

```text
entry_point_hamer.py
```

주요 옵션:

```text
--hand-post-mode {smooth,outlier-only}
--hand-temporal-gate / --no-hand-temporal-gate
--hand-jump-thresh
--hand-jump-confirm-frames
--hand-smooth-sigma
--hand-long-gap-default-weight
--hand-outlier-z
--hand-outlier-min-residual
--hand-plot-params
--save-hamer-crops
```

현재 권장 모드는 `outlier-only`다. 이 모드는 정상 raw HaMeR pose는 최대한 보존하고, missing frame, temporal gate reject, outlier frame만 앞뒤 정상 프레임으로 보간하거나 유지한다. 즉 전체 손 움직임을 흐리게 만드는 강한 Gaussian smoothing이 아니라, 문제가 있는 프레임만 교체하는 방식이다.

예시 실행:

```bash
python -u /workspace/CAP/entry_point_hamer.py \
  --video /workspace/CAP/DATA/CAPSTONE_DATASET/back/back_2.mp4 \
  --output-root /workspace/CAP/outputs/cap_pipeline \
  --hamer-root /workspace/CAP/external/hamer \
  --auto-person \
  --no-interactive \
  --hand-post-mode outlier-only \
  --hand-jump-thresh 0.55 \
  --hand-jump-confirm-frames 3 \
  --hand-plot-params \
  --force
```

## merged 결과에서 손이 쥐어지던 원인

raw HaMeR 렌더링에서는 손가락이 잘 펴져 있는데, merged SMPL-X 렌더링에서는 양손이 계속 쥐어진 것처럼 보이는 문제가 있었다.

확인 결과 원인은 SMPL-X hand mean offset이었다. 현재 `make_smplx("supermotion_fullhands")`가 `flat_hand_mean=False`인 SMPL-X 모델을 쓰고 있어서, SMPL-X 내부에서 `left_hand_mean`, `right_hand_mean`이 hand pose에 더해진다.

그런데 HaMeR에서 가져온 값은 이미 full hand pose에 가까운 값이라, 그대로 SMPL-X에 넣으면 다음처럼 된다.

```text
rendered hand = HaMeR full hand pose + SMPL-X hand mean
```

이 때문에 손이 과하게 쥐어진 형태로 보였다.

수정한 방식:

```text
SMPL-X에 저장할 hand_pose = HaMeR full hand pose - SMPL-X hand mean
```

즉 병합 시에는 HaMeR 값을 SMPL-X residual 형태로 바꿔 넣는다. 손목과 팔 위치는 GVHMR 결과를 그대로 유지하고, HaMeR에서 가져오는 것은 45D finger pose만이다.

## HaMeR raw 렌더링

raw HaMeR가 실제로 프레임마다 어떤 손 mesh를 냈는지 확인하기 위해 렌더링 스크립트를 사용한다.

```text
scripts/render_hamer_hands.py
```

`back_2`에 대해 실행한 명령:

```bash
/opt/conda/bin/python /workspace/CAP/scripts/render_hamer_hands.py \
  /workspace/CAP/outputs/cap_pipeline/back_2 \
  --hamer-out /workspace/CAP/outputs/cap_pipeline/back_2/hamer_out_clean \
  --out-dir /workspace/CAP/outputs/cap_pipeline/back_2/frames_hamer_raw \
  --force
```

출력:

```text
/workspace/CAP/outputs/cap_pipeline/back_2/frames_hamer_raw
```

`back_2` 결과:

```text
frames: 1608
rendered_frames: 1608
hands: 3170
camera_sources: cam_t_full
```

이 렌더링은 merged 결과가 아니라 HaMeR 단독 결과를 보는 용도다. 따라서 merged에서 문제가 생겼는지, HaMeR 자체가 잘못 추정했는지를 분리해서 확인할 수 있다.

## 손 파라미터 spike 리포트

추가한 파일:

```text
scripts/report_hand_spikes.py
```

이 스크립트는 raw HaMeR의 45D MANO finger pose를 프레임별로 이어서, 파라미터가 갑자기 튀는 프레임을 찾는다.

계산하는 값:

```text
jump_prev = 현재 pose와 이전 pose의 RMS 차이
jump_next = 다음 pose와 현재 pose의 RMS 차이
residual  = 현재 pose가 앞뒤 프레임 평균에서 벗어난 정도
score     = 위 값들 중 최댓값
```

실행:

```bash
/opt/conda/bin/python /workspace/CAP/scripts/report_hand_spikes.py \
  /workspace/CAP/outputs/cap_pipeline/back_2 \
  --top-k 80 \
  --jump-thresh 0.55 \
  --residual-thresh 0.55
```

출력:

```text
/workspace/CAP/outputs/cap_pipeline/back_2/hand_spikes.json
/workspace/CAP/outputs/cap_pipeline/back_2/hand_spikes.csv
```

`back_2`에서 threshold `0.55` 기준으로 확실히 잡힌 프레임:

```text
left: 1080, 1081
right: 443, 444, 452, 453
```

threshold 아래지만 의심 후보로 볼 수 있는 프레임도 CSV에 점수순으로 남는다. 예를 들어 right hand에서는 `111`, `112`, `643`, `644` 등이 top 후보에 들어왔다.

## 손 mesh 크기 outlier 리포트

추가한 파일:

```text
scripts/report_hamer_size_outliers.py
```

111번 프레임처럼 HaMeR가 등/어깨 쪽에 말도 안 되는 손 mesh를 붙이는 경우가 있었다. 이 경우 finger pose만 보면 문제를 완전히 설명하기 어렵고, hand mesh가 이미지 위에서 어떻게 투영되는지 봐야 한다.

이 스크립트는 HaMeR npz에 저장된 값을 사용한다.

```text
vertices
cam_t_full
focal_length
is_right
```

이를 full image plane으로 다시 투영해서 2D bbox를 계산한다.

계산하는 값:

```text
area_frac  = projected hand bbox area / frame area
diag_frac  = projected hand bbox diagonal / frame diagonal
side_frac  = projected hand bbox max side / frame max side
area_jump  = 같은 손의 직전 detection 대비 area_frac 증가율
size_score = threshold 대비 outlier 강도
```

실행:

```bash
/opt/conda/bin/python /workspace/CAP/scripts/report_hamer_size_outliers.py \
  /workspace/CAP/outputs/cap_pipeline/back_2 \
  --top-k 80
```

출력:

```text
/workspace/CAP/outputs/cap_pipeline/back_2/hamer_size_outliers.json
/workspace/CAP/outputs/cap_pipeline/back_2/hamer_size_outliers.csv
```

`back_2` 결과:

```text
detections: 3170
outliers: 62
outlier frame examples: 3, 26, 64, 90, 98, 111, 114, 118, 119, 131, ...
```

111번 프레임은 절대 크기 자체가 가장 큰 프레임은 아니지만, right hand의 `area_jump`가 약 `3.15x`로 튀어서 outlier로 잡혔다. 즉 이 케이스는 단순히 "손이 너무 크다"가 아니라 "이전 프레임 대비 갑자기 크기/위치가 이상해졌다"로 보는 것이 더 맞다.

## 111번 프레임 분석

111번 프레임에서 raw HaMeR 렌더링을 보면 손 mesh가 실제 손 위치가 아니라 등/어깨 근처에 크게 붙는다. 이는 MANO pose smoothing으로 고칠 문제가 아니다.

이런 경우는 다음처럼 처리하는 것이 맞다.

```text
HaMeR detection 자체를 invalid 처리
-> 해당 프레임의 HaMeR hand pose를 버림
-> 이전 정상 pose 유지 또는 앞뒤 정상 프레임 보간
```

이 케이스는 다음 두 리포트에서 모두 의심 후보로 잡힌다.

```text
hand_spikes: right 111/112 근처 후보
hamer_size_outliers: frame 111 outlier
```

## 현재 추천 invalid 조건

HaMeR 결과를 사용할지 말지 결정할 때 다음 조건을 조합하는 것이 좋다.

```text
pose spike outlier
OR size outlier
OR size jump outlier
OR future wrist proximity outlier
```

현재 구현/리포트된 것은 다음이다.

```text
pose spike outlier: scripts/report_hand_spikes.py
size/size jump outlier: scripts/report_hamer_size_outliers.py
```

추가하면 좋은 것은 wrist proximity gate다.

```text
GVHMR/SMPL-X 손목 3D를 이미지로 projection
HaMeR hand bbox center 또는 mesh center와 거리 비교
너무 멀면 invalid
```

111번 같은 케이스는 크기 jump로도 잡히지만, wrist proximity gate가 있으면 더 직접적으로 걸러질 가능성이 높다.

## 스무딩 전략 정리

처음 논의한 아이디어는 두 종류였다.

1. 손이 탐지되지 않은 구간을 기본 자세나 주변 프레임으로 생성하고 Gaussian smoothing
2. 손이 탐지되었지만 부정확한 구간을 그래프/미분/극값 기반으로 찾아 smoothing

현재 판단은 다음과 같다.

- 전체 Gaussian smoothing을 강하게 걸면 raw HaMeR가 잘 맞춘 손가락 움직임까지 뭉개질 수 있다.
- 그립은 어느 정도 유지되는 편이라, 정상 구간은 최대한 보존하는 것이 좋다.
- 문제 프레임만 invalid 처리하고 앞뒤 정상 프레임으로 보간하는 `outlier-only` 방식이 현재 문제에 더 적합하다.

권장 방식:

```text
raw HaMeR pose는 기본적으로 사용
-> pose spike / size outlier / missing detection만 invalid
-> invalid 구간만 interpolation 또는 hold
```

라켓을 잡은 손은 짧은 invalid 구간에서 이전 정상 grip을 유지하는 편이 자연스럽다. 손을 펴는 동작처럼 실제 손가락 변화가 있는 구간은 앞뒤 정상 프레임 보간이 더 적합하다.

## front/global 렌더링

merged 결과를 고정된 front/global view에서 확인하기 위해 다음 스크립트를 추가했다.

```text
scripts/render_merged_front.py
```

실행:

```bash
/opt/conda/bin/python /workspace/CAP/scripts/render_merged_front.py \
  /workspace/CAP/outputs/cap_pipeline/back_2 \
  --force
```

출력:

```text
/workspace/CAP/outputs/cap_pipeline/back_2/smplx_merged_hamer_front.mp4
```

이 스크립트는 기본적으로 `BASE/0_input_video.mp4`에서 FPS를 추론하도록 수정되어, 출력 영상 프레임 속도가 원본 영상과 맞도록 했다.

## FPS / cached video 관련 수정

`front_1.mp4`에서 PyAV가 `59.894` 같은 FPS를 처리하다가 overflow를 내는 문제가 있었다.

수정 파일:

```text
hmr4d/utils/video_io_utils.py
```

수정 내용:

```text
fps를 Fraction(fps).limit_denominator(1000)로 정규화해서 writer에 전달
```

또한 이전 실패로 깨진 `0_input_video.mp4`가 남아 있으면 `moov atom not found`가 발생했다.

수정 파일:

```text
tools/demo/demo.py
```

수정 내용:

```text
cached 0_input_video.mp4를 읽다가 실패하면 경고 후 삭제하고 다시 copy/rebuild
```

## 주요 디버깅 파일 위치

`back_2` 기준:

```text
/workspace/CAP/outputs/cap_pipeline/back_2/frames_hamer_raw/
/workspace/CAP/outputs/cap_pipeline/back_2/hand_spikes.json
/workspace/CAP/outputs/cap_pipeline/back_2/hand_spikes.csv
/workspace/CAP/outputs/cap_pipeline/back_2/hamer_size_outliers.json
/workspace/CAP/outputs/cap_pipeline/back_2/hamer_size_outliers.csv
/workspace/CAP/outputs/cap_pipeline/back_2/smplx_merged_hamer_post.pt
/workspace/CAP/outputs/cap_pipeline/back_2/smplx_merged_hamer_incam.mp4
/workspace/CAP/outputs/cap_pipeline/back_2/smplx_merged_hamer_front.mp4
```

## 다음 단계

아직 리포트 단계인 size outlier와 pose spike를 `entry_point_hamer.py`의 실제 invalid mask에 연결하면 된다.

권장 적용 순서:

```text
1. report_hand_spikes.py의 pose spike 조건을 pipeline 내부 invalid mask로 연결
2. report_hamer_size_outliers.py의 size/area_jump 조건을 pipeline 내부 invalid mask로 연결
3. invalid frame은 HaMeR pose를 버리고 앞뒤 정상 pose로 interpolation
4. wrist proximity gate 추가
5. 최종적으로 hand plot, raw HaMeR render, merged render를 같이 비교
```

새 모델 도입은 그 다음 단계로 보는 것이 좋다. 현재는 HaMeR를 대체할 모델보다, "이 HaMeR 결과를 믿어도 되는지"를 판별하는 detector/confidence gate가 더 중요하다.

후보:

```text
MediaPipe Hands
ViTPose hand keypoints
RTMDet/hand detector
```

이 모델들은 손 mesh를 만들기 위한 것이 아니라, HaMeR 결과를 쓸지 버릴지 결정하는 confidence source로 쓰는 것이 적합하다.
