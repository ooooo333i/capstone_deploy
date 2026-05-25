#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report HaMeR detections with unusually large projected hand mesh bboxes.")
    parser.add_argument("base_dir", type=Path, help="CAP run directory, e.g. outputs/cap_pipeline/back_2")
    parser.add_argument("--frame-dir", type=Path, default=None, help="Defaults to BASE/hamer_frames")
    parser.add_argument("--hamer-out", type=Path, default=None, help="Defaults to BASE/hamer_out_clean, then BASE/hamer_out")
    parser.add_argument("--out", type=Path, default=None, help="Defaults to BASE/hamer_size_outliers.json")
    parser.add_argument("--csv", type=Path, default=None, help="Defaults to BASE/hamer_size_outliers.csv")
    parser.add_argument("--area-frac-thresh", type=float, default=0.06, help="Projected bbox area / frame area threshold.")
    parser.add_argument("--diag-frac-thresh", type=float, default=0.45, help="Projected bbox diagonal / frame diagonal threshold.")
    parser.add_argument("--side-frac-thresh", type=float, default=0.45, help="Projected bbox max side / frame max side threshold.")
    parser.add_argument("--area-jump-ratio", type=float, default=3.0, help="Flag if area jumps this much from the previous same-side detection.")
    parser.add_argument("--top-k", type=int, default=80)
    return parser.parse_args()


def frame_id_from_npz(path: Path) -> str | None:
    match = re.match(r"^(\d+)_", path.stem)
    return match.group(1) if match else None


def resolve_hamer_out(base_dir: Path, arg_path: Path | None) -> Path:
    if arg_path is not None:
        return arg_path.resolve()
    clean = base_dir / "hamer_out_clean"
    if clean.is_dir():
        return clean.resolve()
    return (base_dir / "hamer_out").resolve()


def group_hamer_outputs(hamer_out: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for npz_path in sorted(hamer_out.glob("*.npz")):
        frame_id = frame_id_from_npz(npz_path)
        if frame_id is not None:
            grouped[frame_id].append(npz_path)
    return grouped


def load_frame_sizes(frame_dir: Path) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    for frame_path in sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png")):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        height, width = img.shape[:2]
        sizes[frame_path.stem] = (width, height)
    return sizes


def project_bbox(npz_path: Path, width: int, height: int) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    vertices = np.asarray(data["vertices"], dtype=np.float32).copy()
    is_right = int(np.asarray(data["is_right"]).reshape(-1)[0]) if "is_right" in data.files else 1
    vertices[:, 0] = (2 * is_right - 1) * vertices[:, 0]

    if "cam_t_full" in data.files:
        cam_t = np.asarray(data["cam_t_full"], dtype=np.float32).reshape(3)
    elif "full_cam_t" in data.files:
        cam_t = np.asarray(data["full_cam_t"], dtype=np.float32).reshape(3)
    else:
        cam_t = np.asarray(data["cam_t"], dtype=np.float32).reshape(3)

    focal = float(np.asarray(data["focal_length"]).reshape(-1)[0]) if "focal_length" in data.files else 5000.0
    verts_cam = vertices + cam_t
    z = np.maximum(verts_cam[:, 2], 1e-6)
    projected = np.empty((verts_cam.shape[0], 2), dtype=np.float32)
    projected[:, 0] = focal * verts_cam[:, 0] / z + width / 2.0
    projected[:, 1] = focal * verts_cam[:, 1] / z + height / 2.0
    finite = np.isfinite(projected).all(axis=1)
    if not finite.any():
        raise ValueError(f"No finite projected vertices: {npz_path}")

    pts = projected[finite]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bbox_w = float(x2 - x1)
    bbox_h = float(y2 - y1)
    area_frac = float((bbox_w * bbox_h) / max(width * height, 1))
    diag_frac = float(np.hypot(bbox_w, bbox_h) / max(np.hypot(width, height), 1e-6))
    side_frac = float(max(bbox_w / max(width, 1), bbox_h / max(height, 1)))
    box_size = float(np.asarray(data["box_size"]).reshape(-1)[0]) if "box_size" in data.files else None
    crop_frac = None if box_size is None else float(box_size / max(width, height))

    return {
        "side": "right" if is_right else "left",
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "area_frac": area_frac,
        "diag_frac": diag_frac,
        "side_frac": side_frac,
        "crop_frac": crop_frac,
    }


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    frame_dir = (args.frame_dir or base_dir / "hamer_frames").resolve()
    hamer_out = resolve_hamer_out(base_dir, args.hamer_out)
    out_path = (args.out or base_dir / "hamer_size_outliers.json").resolve()
    csv_path = (args.csv or base_dir / "hamer_size_outliers.csv").resolve()

    frame_sizes = load_frame_sizes(frame_dir)
    grouped = group_hamer_outputs(hamer_out)
    rows: list[dict[str, Any]] = []
    previous_area_by_side: dict[str, float] = {}

    for frame_id in sorted(grouped):
        if frame_id not in frame_sizes:
            continue
        width, height = frame_sizes[frame_id]
        for npz_path in grouped[frame_id]:
            row = project_bbox(npz_path, width, height)
            side = row["side"]
            prev_area = previous_area_by_side.get(side)
            area_jump = None
            if prev_area is not None and prev_area > 1e-8:
                area_jump = float(row["area_frac"] / prev_area)
            previous_area_by_side[side] = row["area_frac"]
            jump_score = 0.0 if area_jump is None else area_jump / max(args.area_jump_ratio, 1e-8)
            size_score = max(
                row["area_frac"] / max(args.area_frac_thresh, 1e-8),
                row["diag_frac"] / max(args.diag_frac_thresh, 1e-8),
                row["side_frac"] / max(args.side_frac_thresh, 1e-8),
                jump_score,
            )

            row.update(
                {
                    "frame": int(frame_id),
                    "npz": str(npz_path),
                    "width": width,
                    "height": height,
                    "area_jump": area_jump,
                    "size_score": float(size_score),
                    "is_outlier": (
                        row["area_frac"] >= args.area_frac_thresh
                        or row["diag_frac"] >= args.diag_frac_thresh
                        or row["side_frac"] >= args.side_frac_thresh
                        or (area_jump is not None and area_jump >= args.area_jump_ratio)
                    ),
                }
            )
            rows.append(row)

    top = sorted(rows, key=lambda item: item["size_score"], reverse=True)
    outliers = [row for row in rows if row["is_outlier"]]
    report = {
        "base_dir": str(base_dir),
        "frame_dir": str(frame_dir),
        "hamer_out": str(hamer_out),
        "thresholds": {
            "area_frac": args.area_frac_thresh,
            "diag_frac": args.diag_frac_thresh,
            "side_frac": args.side_frac_thresh,
            "area_jump_ratio": args.area_jump_ratio,
        },
        "detections": len(rows),
        "outliers": len(outliers),
        "outlier_frames": sorted({int(row["frame"]) for row in outliers}),
        "top": top[: args.top_k],
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fieldnames = [
        "frame",
        "side",
        "area_frac",
        "diag_frac",
        "side_frac",
        "area_jump",
        "size_score",
        "crop_frac",
        "bbox_w",
        "bbox_h",
        "is_outlier",
        "npz",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in top:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"detections={len(rows)}, outliers={len(outliers)}")
    print("outlier frames:", ", ".join(str(frame) for frame in report["outlier_frames"][:40]))
    print(f"saved: {out_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
