from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import entry_point as cap


@dataclass
class HandPostprocessConfig:
    smooth_sigma: float = 2.0
    gap_max_interp: int = 8
    long_gap_default_weight: float = 0.85
    outlier_z: float = 3.5
    outlier_min_residual: float = 0.35
    plot_params: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GVHMR + HaMeR pipeline with temporal hand-pose postprocessing."
    )
    parser.add_argument("input_video", nargs="?", help="Input video path.")
    parser.add_argument("--video", type=str, default=None, help="Input video path. If omitted, you will be prompted.")
    parser.add_argument("--output-root", type=str, default=str(cap.DEFAULT_OUTPUT_ROOT), help="Root output directory.")
    parser.add_argument("--static-cam", action="store_true", help="Tell GVHMR the camera is static.")
    parser.add_argument("--use-dpvo", action="store_true", help="Use DPVO instead of GVHMR SimpleVO.")
    parser.add_argument("--f-mm", type=int, default=None, help="Full-frame focal length in mm for GVHMR.")
    parser.add_argument("--auto-person", action="store_true", help="Use the largest detected person track.")
    parser.add_argument("--person-select-ui", choices=("auto", "window", "terminal"), default="auto")
    parser.add_argument("--person-track-id", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Save preprocessing/debug overlays.")
    parser.add_argument("--render-preview", dest="skip_gvhmr_render", action="store_false")
    parser.add_argument("--skip-gvhmr-render", dest="skip_gvhmr_render", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true", help="Recompute generated outputs when possible.")
    parser.add_argument("--hamer-root", type=str, default=str(cap.DEFAULT_HAMER_ROOT), help="Local HaMeR repository root.")
    parser.add_argument("--hamer-checkpoint", type=str, default=None, help="Optional HaMeR checkpoint path.")
    parser.add_argument("--hamer-batch-size", type=int, default=1, help="HaMeR inference batch size.")
    parser.add_argument("--hamer-rescale-factor", type=float, default=2.5, help="HaMeR hand crop padding factor.")
    parser.add_argument("--hand-min-conf", type=float, default=0.35, help="Minimum GVHMR/VitPose wrist confidence.")
    parser.add_argument("--skip-result-video", action="store_true", help="Skip the final merged mp4 render.")
    parser.add_argument("--no-interactive", action="store_true", help="Fail instead of prompting for missing values.")

    parser.add_argument("--hand-smooth-sigma", type=float, default=2.0, help="Gaussian sigma for hand pose smoothing.")
    parser.add_argument("--hand-gap-max-interp", type=int, default=8, help="Max missing gap length to interpolate directly.")
    parser.add_argument(
        "--hand-long-gap-default-weight",
        type=float,
        default=0.85,
        help="How strongly long missing gaps move toward the default pose.",
    )
    parser.add_argument("--hand-outlier-z", type=float, default=3.5, help="Robust z threshold for hand-pose spikes.")
    parser.add_argument(
        "--hand-outlier-min-residual",
        type=float,
        default=0.35,
        help="Minimum adjacent-frame residual before a detected hand pose can be considered an outlier.",
    )
    parser.add_argument("--hand-plot-params", action="store_true", help="Save per-hand parameter plots next to merged output.")
    parser.set_defaults(skip_gvhmr_render=True)
    return parser.parse_args()


def gaussian_kernel1d(sigma: float) -> Any:
    import numpy as np

    if sigma <= 0:
        return np.asarray([1.0], dtype=np.float32)
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).astype(np.float32)


def smooth_sequence(seq: Any, sigma: float) -> Any:
    import numpy as np

    kernel = gaussian_kernel1d(sigma)
    if kernel.size == 1:
        return seq.astype(np.float32, copy=True)
    pad = kernel.size // 2
    padded = np.pad(seq, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(seq, dtype=np.float32)
    for dim in range(seq.shape[1]):
        out[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return out


def contiguous_ranges(mask: Any) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if bool(value) and start is None:
            start = idx
        elif not bool(value) and start is not None:
            ranges.append((start, idx))
            start = None
    if start is not None:
        ranges.append((start, len(mask)))
    return ranges


def detect_pose_outliers(seq: Any, detected: Any, cfg: HandPostprocessConfig) -> Any:
    import numpy as np

    outlier = np.zeros(seq.shape[0], dtype=bool)
    residuals: list[tuple[int, float]] = []
    for idx in range(1, seq.shape[0] - 1):
        if not (detected[idx - 1] and detected[idx] and detected[idx + 1]):
            continue
        expected = 0.5 * (seq[idx - 1] + seq[idx + 1])
        residual = float(np.linalg.norm(seq[idx] - expected) / np.sqrt(seq.shape[1]))
        residuals.append((idx, residual))

    if not residuals:
        return outlier

    values = np.asarray([value for _, value in residuals], dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_scale = max(1.4826 * mad, 1e-6)
    threshold = max(cfg.outlier_min_residual, median + cfg.outlier_z * robust_scale)
    for idx, residual in residuals:
        if residual > threshold:
            outlier[idx] = True
    return outlier


def fill_missing_hand_pose(seq: Any, valid: Any, default_pose: Any, cfg: HandPostprocessConfig) -> Any:
    import numpy as np

    filled = seq.astype(np.float32, copy=True)
    missing_ranges = contiguous_ranges(~valid)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return np.repeat(default_pose.reshape(1, -1), seq.shape[0], axis=0).astype(np.float32)

    first_valid = int(valid_indices[0])
    last_valid = int(valid_indices[-1])
    for start, end in missing_ranges:
        length = end - start
        left_idx = start - 1
        right_idx = end
        has_left = left_idx >= 0 and valid[left_idx]
        has_right = right_idx < seq.shape[0] and valid[right_idx]

        if has_left and has_right:
            left = filled[left_idx]
            right = filled[right_idx]
            direct_interp = length <= cfg.gap_max_interp
            for offset, frame_idx in enumerate(range(start, end), start=1):
                alpha = offset / float(length + 1)
                interp = (1.0 - alpha) * left + alpha * right
                if direct_interp:
                    filled[frame_idx] = interp
                else:
                    center_weight = np.sin(np.pi * alpha) * cfg.long_gap_default_weight
                    filled[frame_idx] = (1.0 - center_weight) * interp + center_weight * default_pose
        elif end <= first_valid:
            target = filled[first_valid]
            for frame_idx in range(start, end):
                distance = first_valid - frame_idx
                alpha = max(0.0, 1.0 - distance / float(max(cfg.gap_max_interp, 1) + 1))
                filled[frame_idx] = (1.0 - alpha) * default_pose + alpha * target
        elif start > last_valid:
            target = filled[last_valid]
            for frame_idx in range(start, end):
                distance = frame_idx - last_valid
                alpha = max(0.0, 1.0 - distance / float(max(cfg.gap_max_interp, 1) + 1))
                filled[frame_idx] = (1.0 - alpha) * default_pose + alpha * target
        else:
            filled[start:end] = default_pose
    return filled


def postprocess_hand_sequence(
    detections: dict[int, Any],
    fallback_pose: Any,
    cfg: HandPostprocessConfig,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    num_frames = fallback_pose.shape[0]
    default_pose = np.zeros((45,), dtype=np.float32)
    raw = np.zeros((num_frames, 45), dtype=np.float32)
    detected = np.zeros(num_frames, dtype=bool)
    for frame_idx, pose in detections.items():
        if 0 <= frame_idx < num_frames:
            raw[frame_idx] = np.asarray(pose, dtype=np.float32).reshape(45)
            detected[frame_idx] = True

    outlier = detect_pose_outliers(raw, detected, cfg)
    valid = detected & ~outlier
    filled = fill_missing_hand_pose(raw, valid, default_pose, cfg)
    smoothed = smooth_sequence(filled, cfg.smooth_sigma)

    missing = ~detected
    meta = {
        "detected_frames": int(detected.sum()),
        "valid_frames_after_outlier_filter": int(valid.sum()),
        "missing_frames": int(missing.sum()),
        "outlier_frames": [int(idx) for idx in np.flatnonzero(outlier).tolist()],
        "missing_ranges": [[int(a), int(b)] for a, b in contiguous_ranges(missing)],
        "filled_ranges": [[int(a), int(b)] for a, b in contiguous_ranges(~valid)],
    }
    return smoothed.astype(np.float32), meta


def plot_hand_params(out_dir: Path, side: str, raw: Any, processed: Any, detected: Any) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[CAP] Warning: could not plot {side} hand params: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(processed.shape[0])
    fig, axes = plt.subplots(15, 1, figsize=(18, 24), sharex=True)
    for joint_idx, ax in enumerate(axes):
        cols = slice(joint_idx * 3, joint_idx * 3 + 3)
        for local_dim, color in enumerate(("tab:red", "tab:green", "tab:blue")):
            dim = cols.start + local_dim
            ax.plot(frames, processed[:, dim], color=color, linewidth=0.9)
            raw_values = raw[:, dim].copy()
            raw_values[~detected] = np.nan
            ax.plot(frames, raw_values, color=color, linewidth=0.4, alpha=0.35)
        ax.set_ylabel(f"j{joint_idx:02d}", rotation=0, labelpad=18)
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"{side} HaMeR hand pose params: processed lines, faint raw detections")
    fig.tight_layout()
    fig.savefig(out_dir / f"{side}_hand_params.png", dpi=140)
    plt.close(fig)


def merge_hamer_hands_into_gvhmr_postprocessed(
    gvhmr_results: Path,
    hamer_out_dir: Path,
    output_path: Path,
    cfg: HandPostprocessConfig,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    import numpy as np
    import torch

    pred = cap.torch_load_file(torch, gvhmr_results, map_location="cpu")
    merged = copy.deepcopy(pred)

    base_params = merged.get("smpl_params_global") or merged.get("smpl_params_incam")
    if base_params is None:
        raise KeyError("GVHMR results do not contain smpl_params_global or smpl_params_incam.")

    num_frames = cap.infer_num_frames_from_smpl_params(base_params)
    detections = cap.load_hamer_hand_detections(hamer_out_dir, num_frames)

    processed_by_side: dict[str, Any] = {}
    side_meta: dict[str, Any] = {}
    plot_cache: dict[str, tuple[Any, Any, Any]] = {}
    for side, hand_key in (("left", "left_hand_pose"), ("right", "right_hand_pose")):
        fallback_tensor = cap.ensure_hand_pose_tensor(base_params, hand_key, num_frames, torch.float32, torch)
        fallback = fallback_tensor.detach().cpu().numpy().reshape(num_frames, 45)
        processed, meta = postprocess_hand_sequence(detections[side], fallback, cfg)
        processed_by_side[side] = processed
        side_meta[side] = meta

        raw = np.zeros((num_frames, 45), dtype=np.float32)
        detected = np.zeros(num_frames, dtype=bool)
        for frame_idx, pose in detections[side].items():
            raw[frame_idx] = np.asarray(pose, dtype=np.float32).reshape(45)
            detected[frame_idx] = True
        plot_cache[side] = (raw, processed, detected)

    for param_key in ("smpl_params_global", "smpl_params_incam"):
        if param_key not in merged:
            continue
        params = merged[param_key]
        dtype = params["body_pose"].dtype if "body_pose" in params else torch.float32
        params["left_hand_pose"] = torch.from_numpy(processed_by_side["left"]).to(dtype=dtype)
        params["right_hand_pose"] = torch.from_numpy(processed_by_side["right"]).to(dtype=dtype)

    report = {
        "gvhmr_results": str(gvhmr_results),
        "hamer_out_dir": str(hamer_out_dir),
        "merged_results": str(output_path),
        "num_frames": num_frames,
        "left_hand_frames": side_meta["left"]["detected_frames"],
        "right_hand_frames": side_meta["right"]["detected_frames"],
        "hand_postprocess": {
            "smooth_sigma": cfg.smooth_sigma,
            "gap_max_interp": cfg.gap_max_interp,
            "long_gap_default_weight": cfg.long_gap_default_weight,
            "outlier_z": cfg.outlier_z,
            "outlier_min_residual": cfg.outlier_min_residual,
            "left": side_meta["left"],
            "right": side_meta["right"],
        },
    }
    if extra_meta:
        report.update(extra_meta)
    merged["cap_merge_meta"] = report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if cfg.plot_params:
        plot_dir = output_path.parent / "hand_param_plots"
        for side, (raw, processed, detected) in plot_cache.items():
            plot_hand_params(plot_dir, side, raw, processed, detected)

    return output_path, report


def main() -> None:
    total_tic = time.perf_counter()
    launch_cwd = Path.cwd()
    args = cap.complete_interactive_args(parse_args())
    cap.ensure_project_on_path()

    video = cap.normalize_input_path(args.video, launch_cwd)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    output_root = cap.normalize_input_path(args.output_root, launch_cwd)
    hamer_root = cap.normalize_input_path(args.hamer_root, launch_cwd)
    hamer_checkpoint = cap.normalize_input_path(args.hamer_checkpoint, launch_cwd) if args.hamer_checkpoint else None
    pp_cfg = HandPostprocessConfig(
        smooth_sigma=args.hand_smooth_sigma,
        gap_max_interp=args.hand_gap_max_interp,
        long_gap_default_weight=args.hand_long_gap_default_weight,
        outlier_z=args.hand_outlier_z,
        outlier_min_residual=args.hand_outlier_min_residual,
        plot_params=args.hand_plot_params,
    )

    os.chdir(cap.PROJ_ROOT)

    gvhmr_run = cap.run_gvhmr(
        video=video,
        output_root=output_root,
        static_cam=args.static_cam,
        use_dpvo=args.use_dpvo,
        f_mm=args.f_mm,
        verbose=args.verbose,
        render=not args.skip_gvhmr_render,
        force=args.force,
        auto_person=args.auto_person,
        person_track_id=args.person_track_id,
        no_interactive=args.no_interactive,
        selection_ui=args.person_select_ui,
    )

    frame_dir = gvhmr_run.output_dir / "hamer_frames"
    hamer_out_dir = gvhmr_run.output_dir / "hamer_out_clean"
    frame_extract_tic = time.perf_counter()
    frame_count = cap.extract_video_frames(gvhmr_run.video_path, frame_dir, args.force)
    frame_extract_sec = time.perf_counter() - frame_extract_tic
    print(f"[CAP] Prepared {frame_count} frames for HaMeR: {frame_dir}")

    hamer_tic = time.perf_counter()
    cap.run_hamer_from_gvhmr_keypoints(
        frame_dir=frame_dir,
        vitpose_path=gvhmr_run.vitpose_path,
        out_dir=hamer_out_dir,
        hamer_root=hamer_root,
        checkpoint=hamer_checkpoint,
        batch_size=args.hamer_batch_size,
        rescale_factor=args.hamer_rescale_factor,
        min_conf=args.hand_min_conf,
        force=args.force,
        verbose=args.verbose,
    )
    hamer_sec = time.perf_counter() - hamer_tic

    merged_path = gvhmr_run.output_dir / "smplx_merged_hamer_post.pt"
    total_elapsed_before_render = time.perf_counter() - total_tic
    runtime_meta = {
        **gvhmr_run.runtime_report,
        "frame_extract_sec": frame_extract_sec,
        "hamer_sec": hamer_sec,
        "input_frames": frame_count,
        "gvhmr_preprocess_fps": frame_count / gvhmr_run.runtime_report["gvhmr_preprocess_sec"]
        if gvhmr_run.runtime_report["gvhmr_preprocess_sec"] > 0
        else 0.0,
        "gvhmr_predict_fps": frame_count / gvhmr_run.runtime_report["gvhmr_predict_sec"]
        if gvhmr_run.runtime_report["gvhmr_predict_sec"] > 0
        else 0.0,
        "frame_extract_fps": frame_count / frame_extract_sec if frame_extract_sec > 0 else 0.0,
        "hamer_fps": frame_count / hamer_sec if hamer_sec > 0 else 0.0,
        "pipeline_sec_before_render": total_elapsed_before_render,
    }

    merged_path, report = merge_hamer_hands_into_gvhmr_postprocessed(
        gvhmr_run.hmr4d_results,
        hamer_out_dir,
        merged_path,
        pp_cfg,
        extra_meta=runtime_meta,
    )

    result_video = None
    if not args.skip_result_video:
        render_result_tic = time.perf_counter()
        result_video = cap.render_merged_result_video(gvhmr_run, merged_path, args.force)
        render_result_sec = time.perf_counter() - render_result_tic
        report["result_render_sec"] = render_result_sec
        report["result_render_fps"] = frame_count / render_result_sec if render_result_sec > 0 else 0.0
    else:
        report["result_render_sec"] = 0.0
        report["result_render_fps"] = 0.0

    report["pipeline_total_sec"] = time.perf_counter() - total_tic
    report["pipeline_total_fps"] = frame_count / report["pipeline_total_sec"] if report["pipeline_total_sec"] > 0 else 0.0
    merged_path, report = merge_hamer_hands_into_gvhmr_postprocessed(
        gvhmr_run.hmr4d_results,
        hamer_out_dir,
        merged_path,
        pp_cfg,
        extra_meta=report,
    )

    print("[CAP] Done")
    print(f"[CAP] Postprocessed merged SMPL-X params: {merged_path}")
    if result_video is not None:
        print(f"[CAP] Merged result video: {result_video}")
    print(f"[CAP] Left hand frames: {report['left_hand_frames']} / {report['num_frames']}")
    print(f"[CAP] Right hand frames: {report['right_hand_frames']} / {report['num_frames']}")
    print(f"[CAP] Runtime total: {report['pipeline_total_sec']:.2f}s ({report['pipeline_total_fps']:.2f} frames/s)")


if __name__ == "__main__":
    main()
