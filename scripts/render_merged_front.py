#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from hmr4d.utils.geo.hmr_cam import create_camera_sensor
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import get_video_fps, get_writer
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points


SMPLX_ALLOWED_KEYS = {
    "betas",
    "body_pose",
    "global_orient",
    "transl",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "expression",
}


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def filter_smplx_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k in SMPLX_ALLOWED_KEYS}


def move_to_start_point_face_z(verts: torch.Tensor, joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    verts = verts.clone()
    joints = joints.clone()
    offset = joints[0, 0].clone()
    offset[1] = verts[:, :, 1].min()
    verts = verts - offset
    joints = joints - offset
    transform = compute_T_ayfz2ay(joints[[0]], inverse=True)
    verts = apply_T_on_points(verts, transform)
    joints = apply_T_on_points(joints, transform)
    return verts, joints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render merged SMPL-X animation from a fixed front/global camera.")
    parser.add_argument("base_dir", type=Path, help="CAP run directory, e.g. outputs/cap_pipeline/back_2")
    parser.add_argument("--merged", type=Path, default=None, help="Merged .pt path. Defaults to BASE/smplx_merged_hamer_post.pt")
    parser.add_argument("--out", type=Path, default=None, help="Output mp4. Defaults to BASE/smplx_merged_hamer_front.mp4")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS. Defaults to BASE/0_input_video.mp4 FPS.")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--f-mm", type=float, default=35.0, help="Virtual camera focal length in mm.")
    parser.add_argument("--vec-rot", type=float, default=0.0, help="Yaw angle for camera around subject. 0 is front after face-z alignment.")
    parser.add_argument("--beta", type=float, default=2.0, help="Camera distance multiplier.")
    parser.add_argument("--cam-height-degree", type=float, default=8.0)
    parser.add_argument("--target-center-height", type=float, default=1.0)
    parser.add_argument("--no-ground", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    merged_path = (args.merged or base_dir / "smplx_merged_hamer_post.pt").resolve()
    out_path = (args.out or base_dir / "smplx_merged_hamer_front.mp4").resolve()

    if out_path.exists() and not args.force:
        Log.info(f"[Render Front] Reusing {out_path}")
        return
    if out_path.exists():
        out_path.unlink()

    input_video = base_dir / "0_input_video.mp4"
    render_fps = float(args.fps) if args.fps is not None else get_video_fps(input_video)

    pred = torch_load(merged_path)
    if "smpl_params_global" not in pred:
        raise KeyError(f"{merged_path} has no smpl_params_global")

    smplx = make_smplx("supermotion_fullhands").cuda()
    smpl_params = to_cuda(filter_smplx_params(pred["smpl_params_global"]))
    with torch.no_grad():
        smplx_out = smplx(**smpl_params)
    verts, joints = move_to_start_point_face_z(smplx_out.vertices, smplx_out.joints)

    global_R, global_T, lights = get_global_cameras_static(
        verts.cpu(),
        beta=args.beta,
        cam_height_degree=args.cam_height_degree,
        target_center_height=args.target_center_height,
        vec_rot=args.vec_rot,
    )

    _, _, K = create_camera_sensor(args.width, args.height, args.f_mm)
    renderer = Renderer(args.width, args.height, device="cuda", faces=smplx.faces, K=K, bin_size=0)
    if not args.no_ground:
        scale, cx, cz = get_ground_params_from_points(joints[:, 0], verts)
        renderer.set_ground(scale * 1.5, cx, cz)

    color = torch.ones(3, device="cuda") * 0.8
    writer = get_writer(out_path, fps=render_fps, crf=23)
    try:
        for i in tqdm(range(len(verts)), desc="Rendering Front"):
            cameras = renderer.create_camera(global_R[i], global_T[i])
            if args.no_ground:
                frame = renderer.render_mesh(verts[i].cuda(), background=None, colors=[0.8, 0.8, 0.8])
            else:
                frame = renderer.render_with_ground(verts[[i]], color[None], cameras, lights)
            writer.write_frame(frame)
    finally:
        writer.close()

    Log.info(f"[Render Front] Saved {out_path} at {render_fps:.3f} FPS")


if __name__ == "__main__":
    main()
