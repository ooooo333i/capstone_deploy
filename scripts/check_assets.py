from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
    "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
    "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
    "inputs/checkpoints/yolo/yolov8x.pt",
    "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
]

OPTIONAL_FILES = [
    "inputs/checkpoints/dpvo/dpvo.pth",
]

REQUIRED_DIRS = [
    "external/hamer/hamer",
]


def check_path(path: Path) -> bool:
    ok = path.exists()
    marker = "OK" if ok else "MISSING"
    print(f"{marker:8} {path.relative_to(PROJECT_ROOT)}")
    return ok


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    print("\nRequired files")
    required_file_results = [check_path(PROJECT_ROOT / item) for item in REQUIRED_FILES]

    print("\nRequired directories")
    required_dir_results = [check_path(PROJECT_ROOT / item) for item in REQUIRED_DIRS]

    print("\nOptional files")
    for item in OPTIONAL_FILES:
        check_path(PROJECT_ROOT / item)

    required_ok = all(required_file_results) and all(required_dir_results)
    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
