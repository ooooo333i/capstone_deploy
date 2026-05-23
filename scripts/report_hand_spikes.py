#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import entry_point as cap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report frames where raw HaMeR hand pose parameters spike.")
    parser.add_argument("base_dir", type=Path, help="CAP run directory, e.g. outputs/cap_pipeline/back_2")
    parser.add_argument("--hamer-out", type=Path, default=None, help="Defaults to BASE/hamer_out_clean, then BASE/hamer_out")
    parser.add_argument("--out", type=Path, default=None, help="Defaults to BASE/hand_spikes.json")
    parser.add_argument("--csv", type=Path, default=None, help="Defaults to BASE/hand_spikes.csv")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--jump-thresh", type=float, default=0.55, help="RMS frame-to-frame pose jump threshold.")
    parser.add_argument("--residual-thresh", type=float, default=0.55, help="RMS deviation from neighbor interpolation threshold.")
    parser.add_argument("--num-frames", type=int, default=None, help="Override frame count. Otherwise infer from merged or GVHMR result.")
    return parser.parse_args()


def resolve_hamer_out(base_dir: Path, arg_path: Path | None) -> Path:
    if arg_path is not None:
        return arg_path.resolve()
    clean = base_dir / "hamer_out_clean"
    if clean.is_dir():
        return clean.resolve()
    return (base_dir / "hamer_out").resolve()


def infer_num_frames(base_dir: Path, override: int | None) -> int:
    if override is not None:
        return int(override)
    import torch

    for name in ("smplx_merged_hamer_post.pt", "smplx_merged_hamer.pt", "hmr4d_results.pt"):
        path = base_dir / name
        if not path.is_file():
            continue
        pred = cap.torch_load_file(torch, path, map_location="cpu")
        params = pred.get("smpl_params_global") or pred.get("smpl_params_incam")
        if params is not None:
            return cap.infer_num_frames_from_smpl_params(params)
    frame_dir = base_dir / "hamer_frames"
    return len(sorted(frame_dir.glob("*.jpg")))


def build_sequence(detections: dict[int, Any], num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    seq = np.full((num_frames, 45), np.nan, dtype=np.float32)
    detected = np.zeros(num_frames, dtype=bool)
    for frame_idx, pose in detections.items():
        if 0 <= frame_idx < num_frames:
            seq[frame_idx] = np.asarray(pose, dtype=np.float32).reshape(45)
            detected[frame_idx] = True
    return seq, detected


def compute_scores(seq: np.ndarray, detected: np.ndarray) -> dict[str, np.ndarray]:
    num_frames = len(seq)
    jump_prev = np.full(num_frames, np.nan, dtype=np.float32)
    jump_next = np.full(num_frames, np.nan, dtype=np.float32)
    residual = np.full(num_frames, np.nan, dtype=np.float32)
    dim_scale = np.sqrt(seq.shape[1])

    for idx in range(1, num_frames):
        if detected[idx] and detected[idx - 1]:
            jump_prev[idx] = float(np.linalg.norm(seq[idx] - seq[idx - 1]) / dim_scale)
    for idx in range(num_frames - 1):
        if detected[idx] and detected[idx + 1]:
            jump_next[idx] = float(np.linalg.norm(seq[idx + 1] - seq[idx]) / dim_scale)
    for idx in range(1, num_frames - 1):
        if detected[idx - 1] and detected[idx] and detected[idx + 1]:
            expected = 0.5 * (seq[idx - 1] + seq[idx + 1])
            residual[idx] = float(np.linalg.norm(seq[idx] - expected) / dim_scale)

    combined = np.nanmax(np.stack([jump_prev, jump_next, residual]), axis=0)
    combined[~detected] = np.nan
    return {"jump_prev": jump_prev, "jump_next": jump_next, "residual": residual, "score": combined}


def side_report(side: str, detections: dict[int, Any], num_frames: int, args: argparse.Namespace) -> dict[str, Any]:
    seq, detected = build_sequence(detections, num_frames)
    scores = compute_scores(seq, detected)
    spike_mask = (
        np.nan_to_num(scores["jump_prev"], nan=0.0) >= args.jump_thresh
    ) | (
        np.nan_to_num(scores["jump_next"], nan=0.0) >= args.jump_thresh
    ) | (
        np.nan_to_num(scores["residual"], nan=0.0) >= args.residual_thresh
    )
    spike_frames = np.flatnonzero(spike_mask).astype(int).tolist()
    order = np.argsort(np.nan_to_num(scores["score"], nan=-1.0))[::-1]
    top = []
    for idx in order[: args.top_k]:
        if not detected[idx]:
            continue
        top.append(
            {
                "frame": int(idx),
                "score": None if np.isnan(scores["score"][idx]) else float(scores["score"][idx]),
                "jump_prev": None if np.isnan(scores["jump_prev"][idx]) else float(scores["jump_prev"][idx]),
                "jump_next": None if np.isnan(scores["jump_next"][idx]) else float(scores["jump_next"][idx]),
                "residual": None if np.isnan(scores["residual"][idx]) else float(scores["residual"][idx]),
            }
        )

    return {
        "side": side,
        "detected_frames": int(detected.sum()),
        "spike_frames": spike_frames,
        "top_spikes": top,
    }


def write_csv(csv_path: Path, report: dict[str, Any]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["side", "frame", "score", "jump_prev", "jump_next", "residual"])
        writer.writeheader()
        for side in ("left", "right"):
            for row in report[side]["top_spikes"]:
                writer.writerow({"side": side, **row})


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    hamer_out = resolve_hamer_out(base_dir, args.hamer_out)
    out_path = (args.out or base_dir / "hand_spikes.json").resolve()
    csv_path = (args.csv or base_dir / "hand_spikes.csv").resolve()
    num_frames = infer_num_frames(base_dir, args.num_frames)

    detections = cap.load_hamer_hand_detections(hamer_out, num_frames)
    report = {
        "base_dir": str(base_dir),
        "hamer_out": str(hamer_out),
        "num_frames": num_frames,
        "jump_thresh": args.jump_thresh,
        "residual_thresh": args.residual_thresh,
        "left": side_report("left", detections["left"], num_frames, args),
        "right": side_report("right", detections["right"], num_frames, args),
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(csv_path, report)

    for side in ("left", "right"):
        print(f"{side}: detected={report[side]['detected_frames']}, spikes={len(report[side]['spike_frames'])}")
        print("  top:", ", ".join(str(item["frame"]) for item in report[side]["top_spikes"][:10]))
    print(f"saved: {out_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
