#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
WATERTIGHT_FACES = np.array(
    [
        [92, 38, 234],
        [234, 38, 239],
        [38, 122, 239],
        [239, 122, 279],
        [122, 118, 279],
        [279, 118, 215],
        [118, 117, 215],
        [215, 117, 214],
        [117, 119, 214],
        [214, 119, 121],
        [119, 120, 121],
        [121, 120, 78],
        [120, 108, 78],
        [78, 108, 79],
    ],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render HaMeR MANO meshes directly onto hamer_frames. "
            "The base directory should contain hamer_frames/ and hamer_out/ (or frame_out/)."
        )
    )
    parser.add_argument("base_dir", type=Path, help="CAP run directory, e.g. outputs/cap_pipeline/side_2")
    parser.add_argument("--frame-dir", type=Path, default=None, help="Frame image directory. Defaults to BASE/hamer_frames.")
    parser.add_argument("--hamer-out", type=Path, default=None, help="HaMeR npz directory. Defaults to BASE/hamer_out, then BASE/frame_out.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to BASE/frames.")
    parser.add_argument("--mano-model", type=Path, default=None, help="MANO_RIGHT.pkl. Defaults to external/hamer/_DATA/data/mano/MANO_RIGHT.pkl.")
    parser.add_argument("--focal-length", type=float, default=5000.0, help="Fallback focal length when npz has no focal_length.")
    parser.add_argument("--image-size", type=float, default=224.0, help="HaMeR crop size for scaled full-frame fallback focal length.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Mesh opacity multiplier from 0 to 1.")
    parser.add_argument("--mesh-color", type=float, nargs=3, default=LIGHT_BLUE, help="RGB mesh color in 0..1.")
    parser.add_argument("--copy-missing", action=argparse.BooleanOptionalAction, default=True, help="Write original frames when no hand npz exists.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output images.")
    parser.add_argument("--backend", choices=("auto", "pyrender", "software"), default="auto", help="Rendering backend. auto uses pyrender when available, otherwise OpenCV software rasterization.")
    parser.add_argument("--max-frames", type=int, default=None, help="Render only the first N frame images. Useful for smoke tests.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    base = args.base_dir.resolve()
    frame_dir = (args.frame_dir or base / "hamer_frames").resolve()
    hamer_out = args.hamer_out.resolve() if args.hamer_out else base / "hamer_out"
    if not hamer_out.exists() and args.hamer_out is None and (base / "frame_out").exists():
        hamer_out = base / "frame_out"
    hamer_out = hamer_out.resolve()
    out_dir = (args.out_dir or base / "frames").resolve()
    default_mano = Path(__file__).resolve().parents[1] / "external" / "hamer" / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl"
    mano_model = (args.mano_model or default_mano).resolve()

    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    if not hamer_out.is_dir():
        raise FileNotFoundError(f"HaMeR output directory not found: {hamer_out}")
    if not mano_model.is_file():
        raise FileNotFoundError(f"MANO model file not found: {mano_model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return frame_dir, hamer_out, out_dir, mano_model


def load_mano_faces(mano_model: Path) -> np.ndarray:
    try:
        with mano_model.open("rb") as f:
            data = pickle.load(f, encoding="latin1")
    except ModuleNotFoundError as exc:
        missing_name = exc.name
        if missing_name not in {"scipy", "chumpy"}:
            raise RuntimeError(
                f"Could not load {mano_model} because Python package {missing_name!r} is missing."
            ) from exc
        data = load_mano_faces_with_dummy_modules(mano_model)
    except Exception as exc:
        raise RuntimeError(f"Could not load MANO faces from {mano_model}: {exc}") from exc

    if "f" not in data:
        raise KeyError(f"MANO file does not contain an 'f' face array: {mano_model}")
    faces = np.asarray(data["f"], dtype=np.int64)
    return np.concatenate([faces, WATERTIGHT_FACES], axis=0)


def load_mano_faces_with_dummy_modules(mano_model: Path) -> dict[str, Any]:
    import sys
    import types

    class DummyPickleClass:
        def __new__(cls, *args: Any, **kwargs: Any):
            return object.__new__(cls)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __setstate__(self, state: Any) -> None:
            self.__dict__["_state"] = state

    previous = {
        name: sys.modules.get(name)
        for name in ("scipy", "scipy.sparse", "scipy.sparse.csc", "chumpy", "chumpy.ch", "chumpy.reordering")
    }
    try:
        for name in previous:
            sys.modules[name] = types.ModuleType(name)
        sys.modules["scipy.sparse.csc"].csc_matrix = DummyPickleClass
        sys.modules["chumpy.ch"].Ch = DummyPickleClass
        sys.modules["chumpy.reordering"].Select = DummyPickleClass
        with mano_model.open("rb") as f:
            return pickle.load(f, encoding="latin1")
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def frame_id_from_npz(path: Path) -> str | None:
    match = re.match(r"^(\d+)_", path.stem)
    return match.group(1) if match else None


def group_hamer_outputs(hamer_out: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for npz_path in sorted(hamer_out.glob("*.npz")):
        frame_id = frame_id_from_npz(npz_path)
        if frame_id is not None:
            grouped[frame_id].append(npz_path)
    return grouped


def load_hand(npz_path: Path, width: int, height: int, fallback_focal: float, image_size: float) -> dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    vertices = np.asarray(data["vertices"], dtype=np.float32).copy()
    is_right = int(np.asarray(data["is_right"]).reshape(-1)[0]) if "is_right" in data.files else 1

    # Match HaMeR's full-frame demo convention before rendering multiple hands together.
    vertices[:, 0] = (2 * is_right - 1) * vertices[:, 0]

    camera_source = "cam_t_full"
    if "cam_t_full" in data.files:
        cam_t = np.asarray(data["cam_t_full"], dtype=np.float32).reshape(3)
    elif "full_cam_t" in data.files:
        cam_t = np.asarray(data["full_cam_t"], dtype=np.float32).reshape(3)
        camera_source = "full_cam_t"
    else:
        cam_t = np.asarray(data["cam_t"], dtype=np.float32).reshape(3)
        camera_source = "cam_t_crop_fallback"

    if "focal_length" in data.files:
        focal = float(np.asarray(data["focal_length"]).reshape(-1)[0])
    elif camera_source == "cam_t_crop_fallback":
        focal = fallback_focal
    else:
        focal = fallback_focal / image_size * max(width, height)

    return {"vertices": vertices, "cam_t": cam_t, "is_right": is_right, "focal": focal, "camera_source": camera_source}



class SoftwareHandRenderer:
    def __init__(self, faces: np.ndarray, mesh_color: tuple[float, float, float]):
        import cv2

        self.cv2 = cv2
        self.faces = faces
        self.faces_left = faces[:, [0, 2, 1]]
        self.mesh_color = np.asarray(mesh_color, dtype=np.float32)

    def render_rgba(self, hands: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
        rgba = np.zeros((height, width, 4), dtype=np.float32)
        triangles: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
        for hand in hands:
            verts_cam = hand["vertices"] + hand["cam_t"]
            z = verts_cam[:, 2]
            valid = z > 1e-6
            focal = float(hand["focal"])
            projected = np.empty((verts_cam.shape[0], 2), dtype=np.float32)
            projected[:, 0] = focal * verts_cam[:, 0] / np.maximum(z, 1e-6) + width / 2.0
            projected[:, 1] = focal * verts_cam[:, 1] / np.maximum(z, 1e-6) + height / 2.0
            faces = self.faces if hand["is_right"] else self.faces_left
            for face in faces:
                if not bool(valid[face].all()):
                    continue
                pts = projected[face]
                if not np.isfinite(pts).all():
                    continue
                if pts[:, 0].max() < 0 or pts[:, 0].min() >= width or pts[:, 1].max() < 0 or pts[:, 1].min() >= height:
                    continue
                tri3d = verts_cam[face]
                normal = np.cross(tri3d[1] - tri3d[0], tri3d[2] - tri3d[0])
                norm = float(np.linalg.norm(normal))
                if norm > 1e-8:
                    normal = normal / norm
                    shade = 0.45 + 0.55 * abs(float(np.dot(normal, np.array([0.0, 0.0, -1.0], dtype=np.float32))))
                else:
                    shade = 0.7
                color = tuple(int(np.clip(c * shade, 0, 1) * 255) for c in self.mesh_color)
                triangles.append((float(np.mean(z[face])), pts.astype(np.int32), color))

        # Larger z is farther from the camera in this projection, so paint it first.
        triangles.sort(key=lambda item: item[0], reverse=True)
        color_img = np.zeros((height, width, 3), dtype=np.uint8)
        alpha = np.zeros((height, width), dtype=np.uint8)
        for _, pts, color in triangles:
            self.cv2.fillConvexPoly(color_img, pts, color, lineType=self.cv2.LINE_AA)
            self.cv2.fillConvexPoly(alpha, pts, 255, lineType=self.cv2.LINE_AA)
        rgba[:, :, :3] = color_img.astype(np.float32) / 255.0
        rgba[:, :, 3] = alpha.astype(np.float32) / 255.0
        return rgba


class HamerFrameRenderer:
    def __init__(self, faces: np.ndarray, mesh_color: tuple[float, float, float]):
        import pyrender
        import trimesh

        self.pyrender = pyrender
        self.trimesh = trimesh
        self.faces = faces
        self.faces_left = faces[:, [0, 2, 1]]
        self.mesh_color = mesh_color

    def _mesh(self, vertices: np.ndarray, cam_t: np.ndarray, is_right: int):
        faces = self.faces if is_right else self.faces_left
        vertex_colors = np.array([(*self.mesh_color, 1.0)] * vertices.shape[0])
        mesh = self.trimesh.Trimesh(vertices.copy() + cam_t.copy(), faces.copy(), vertex_colors=vertex_colors, process=False)
        rot = self.trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return self.pyrender.Mesh.from_trimesh(mesh, smooth=True)

    def render_rgba(self, hands: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
        pyrender = self.pyrender
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        for idx, hand in enumerate(hands):
            scene.add(self._mesh(hand["vertices"], hand["cam_t"], hand["is_right"]), f"hand_{idx}")

        focal = float(np.mean([hand["focal"] for hand in hands]))
        camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0, zfar=1e12)
        camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
        scene.add_node(camera_node)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=np.eye(4))
        for xyz in ([0, -1, 1], [0, 1, 1], [1, 1, 2], [-1, -1, 2]):
            pose = np.eye(4)
            pose[:3, 3] = np.asarray(xyz, dtype=np.float32)
            scene.add(pyrender.PointLight(color=np.ones(3), intensity=1.0), pose=pose)

        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        return color.astype(np.float32) / 255.0


def overlay_rgba(frame_bgr: np.ndarray, rgba: np.ndarray, alpha_scale: float) -> np.ndarray:
    frame_rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
    alpha = np.clip(rgba[:, :, 3:4] * alpha_scale, 0.0, 1.0)
    out_rgb = rgba[:, :, :3] * alpha + frame_rgb * (1.0 - alpha)
    return np.clip(out_rgb[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    frame_dir, hamer_out, out_dir, mano_model = resolve_paths(args)

    import cv2

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **_: x

    faces = load_mano_faces(mano_model)
    if args.backend == "software":
        renderer = SoftwareHandRenderer(faces, tuple(args.mesh_color))
    else:
        try:
            renderer = HamerFrameRenderer(faces, tuple(args.mesh_color))
        except ModuleNotFoundError:
            if args.backend == "pyrender":
                raise
            print("[WARN] pyrender/trimesh is not available; using OpenCV software rasterizer.")
            renderer = SoftwareHandRenderer(faces, tuple(args.mesh_color))
    grouped = group_hamer_outputs(hamer_out)
    frame_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]

    stats = {"frames": 0, "rendered_frames": 0, "copied_frames": 0, "hands": 0, "camera_sources": defaultdict(int)}
    for frame_path in tqdm(frame_paths, desc="Rendering HaMeR hands"):
        out_path = out_dir / f"{frame_path.stem}.jpg"
        if out_path.exists() and not args.force:
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        hand_paths = grouped.get(frame_path.stem, [])
        stats["frames"] += 1

        if not hand_paths:
            if args.copy_missing:
                cv2.imwrite(str(out_path), frame)
                stats["copied_frames"] += 1
            continue

        hands = [load_hand(p, width, height, args.focal_length, args.image_size) for p in hand_paths]
        rgba = renderer.render_rgba(hands, width, height)
        out = overlay_rgba(frame, rgba, args.alpha)
        cv2.imwrite(str(out_path), out)
        stats["rendered_frames"] += 1
        stats["hands"] += len(hands)
        for hand in hands:
            stats["camera_sources"][hand["camera_source"]] += 1

    stats["camera_sources"] = dict(stats["camera_sources"])
    stats["frame_dir"] = str(frame_dir)
    stats["hamer_out"] = str(hamer_out)
    stats["out_dir"] = str(out_dir)
    (out_dir / "render_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    if stats["camera_sources"].get("cam_t_crop_fallback"):
        print("[WARN] Some npz files did not contain cam_t_full/full_cam_t; used crop cam_t fallback, so hand placement may be centered or inaccurate.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
