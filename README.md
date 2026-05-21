# CAP Deploy

이 폴더는 개발/학습용 파일을 제외하고, 서비스 추론에 필요한 코드와 체크포인트 배치 구조만 분리한 배포용 프로젝트입니다.

## 포함한 것

- `entry_point.py`: 단일 영상 추론 CLI
- `batch_entry_point.py`: 폴더 단위 배치 실행 CLI
- `service.py`: 업로드 기반 FastAPI 서비스
- `hmr4d/`, `tools/demo/`: GVHMR 추론 런타임 코드
- `inputs/checkpoints/`: 배포 환경에서 체크포인트를 넣을 자리
- `external/hamer/`: HaMeR 패키지를 둘 자리
- `scripts/check_assets.py`: 필수 모델 파일 확인 스크립트

## 체크포인트 배치

아래 파일들은 라이선스 때문에 이 배포 폴더에 자동 복사하지 않았습니다. 배포 서버에서는 같은 경로로 배치하세요.

```text
inputs/checkpoints/
├── body_models/
│   ├── smpl/SMPL_NEUTRAL.pkl
│   └── smplx/SMPLX_NEUTRAL.npz
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
└── yolo/yolov8x.pt
```

DPVO를 쓸 경우에만 `inputs/checkpoints/dpvo/dpvo.pth`도 필요합니다.

HaMeR는 `external/hamer` 아래에 설치합니다. 설치 후 `external/hamer/hamer` 패키지 디렉터리가 있어야 합니다.

## 설치

```bash
cd cap_deploy
conda create -y -n cap-deploy python=3.10
conda activate cap-deploy
pip install -r requirements.txt
pip install -e .
```

HaMeR는 별도로 클론 및 설치한 뒤 `external/hamer`에 두거나, 실행 시 `--hamer-root`로 경로를 넘기세요.

```bash
python scripts/check_assets.py
```

## CLI 실행

```bash
python entry_point.py \
  --video /path/to/input.mp4 \
  --output-root outputs/cap_pipeline \
  --hamer-root external/hamer \
  --auto-person \
  --no-interactive \
  --skip-result-video
```

결과는 `outputs/cap_pipeline/<video_name>/smplx_merged_hamer.pt`에 생성됩니다. 렌더링 영상까지 만들려면 `--skip-result-video`를 빼세요.

## API 서비스 실행

```bash
uvicorn service:app --host 0.0.0.0 --port 8000
```

요청 예시:

```bash
curl -F "video=@/path/to/input.mp4" \
  -F "skip_result_video=true" \
  http://localhost:8000/runs
```

상태 확인:

```bash
curl http://localhost:8000/runs/<job_id>
```

결과 다운로드:

```bash
curl -O http://localhost:8000/runs/<job_id>/artifacts/merged
curl -O http://localhost:8000/runs/<job_id>/artifacts/report
curl -O http://localhost:8000/runs/<job_id>/artifacts/log
```

## 배포 메모

- GPU/CUDA가 필요합니다. 현재 GVHMR 추론 코드는 CUDA가 없으면 실패하도록 되어 있습니다.
- 서비스에서는 사람 선택 프롬프트를 띄우지 않기 위해 기본적으로 `--auto-person --no-interactive`를 사용합니다.
- 인터넷이 막힌 서버에서는 HaMeR 체크포인트 자동 다운로드가 실패할 수 있으므로 `CAP_HAMER_CHECKPOINT` 환경변수나 API form의 `hamer_checkpoint`로 로컬 파일을 지정하세요.
