#!/usr/bin/env python3
"""Estimate global metric scale from FoundationGeo ground depth.

FoundationGeo predicts metric depth for uniformly sampled RGB frames. Each
depth map is backprojected with the matching upstream camera intrinsics, then
the same gravity-aligned floor detector used by the real-depth route estimates
camera height. The global scale is the robust ratio between metric camera
height and the aligned VGGT trajectory height.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--timestamps-txt", required=True, type=Path)
    parser.add_argument("--head-trajectory-in", required=True, type=Path)
    parser.add_argument("--intrinsics-path", required=True, type=Path)
    parser.add_argument("--foundationgeo-model-pt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--depth-scale-cluster-rel-tol", type=float, default=0.08)
    parser.add_argument("--depth-scale-min-support-frames", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when no reliable FoundationGeo ground scale is found.",
    )
    return parser.parse_args()


def load_head_trajectory(path: Path) -> np.ndarray:
    poses = np.asarray(np.load(path), dtype=np.float64)
    if poses.ndim == 2 and poses.shape == (4, 4):
        poses = poses[None]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{path} must have shape [N,4,4] or [4,4], got {poses.shape}")
    return poses


def get_R_yup_to_zup() -> np.ndarray:
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )


def read_timestamps(path: Path) -> list[dict[str, Any]]:
    table = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")
    if table.shape == ():
        table = np.asarray([table])
    names = set(table.dtype.names or [])
    rows = []
    for idx, row in enumerate(table):
        frame_id = int(row["frame_id"]) if "frame_id" in names else idx
        rows.append({
            "frame_id": frame_id,
            "src_frame_id": int(row["src_frame_id"]) if "src_frame_id" in names else frame_id,
            "timestamp": float(row["timestamp"]) if "timestamp" in names else float(idx),
        })
    return rows


def natural_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def list_images(frames_dir: Path) -> list[Path]:
    return sorted(
        (p for p in frames_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: natural_key(p.name),
    )


def image_id_index(images: list[Path]) -> dict[int, Path]:
    mapping = {}
    for path in images:
        numbers = re.findall(r"\d+", path.stem)
        if numbers:
            mapping.setdefault(int(numbers[-1]), path)
    return mapping


def find_image_for_pose(
    pose_index: int,
    timestamps: list[dict[str, Any]],
    images: list[Path],
    id_to_image: dict[int, Path],
) -> Path | None:
    if len(images) == len(timestamps) and pose_index < len(images):
        return images[pose_index]
    if pose_index < len(timestamps):
        row = timestamps[pose_index]
        for key in ("frame_id", "src_frame_id"):
            image = id_to_image.get(int(row[key]))
            if image is not None:
                return image
    return id_to_image.get(pose_index)


def load_intrinsics_for_pose(
    path: Path,
    pose_index: int,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, int]:
    values = np.asarray(np.loadtxt(path, comments="#"), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(
            f"{path} must contain one fx fy cx cy source_width source_height row per pose, "
            f"got shape={values.shape}"
        )
    if not 0 <= pose_index < len(values):
        raise IndexError(f"No intrinsics row for pose {pose_index}: {path} has {len(values)} rows")
    fx, fy, cx, cy, source_width, source_height = values[pose_index]
    if source_width <= 0.0 or source_height <= 0.0:
        raise ValueError(f"Invalid intrinsics source resolution: {(source_width, source_height)}")
    scale_x = float(image_width) / float(source_width)
    scale_y = float(image_height) / float(source_height)
    fx, cx = fx * scale_x, cx * scale_x
    fy, cy = fy * scale_y, cy * scale_y
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if not np.isfinite(K).all() or K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError(f"Invalid intrinsics for pose {pose_index}: {K.tolist()}")
    return K, int(pose_index)


def horizontal_fov_deg(K: np.ndarray, width: int) -> float:
    return float(np.degrees(2.0 * np.arctan(width / (2.0 * float(K[0, 0])))))


def load_image_rgb(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_floor_helpers():
    path = Path(__file__).resolve().with_name("04_prepare_real_sensor_inputs.py")
    spec = importlib.util.spec_from_file_location("_real_depth_floor_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import floor helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_foundationgeo_model(model_path: Path, device: str, fp16: bool):
    root = os.getenv("FOUNDATIONGEO_ROOT")
    if root:
        sys.path.insert(0, str(Path(root).expanduser().resolve()))
    import torch
    from foundationgeo.model.v1 import FoundationGeo

    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"FoundationGeo model not found: {model_path}")
    model = FoundationGeo.from_pretrained(str(model_path)).to(torch.device(device)).eval()
    return model.half() if fp16 else model


def run_foundationgeo_depth(
    model,
    image_rgb: np.ndarray,
    *,
    device: str,
    fp16: bool,
    resolution_level: int,
    num_tokens: int | None,
    fov_x: float,
) -> dict[str, Any]:
    import torch

    tensor = torch.from_numpy(image_rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
    force_fp32 = bool(getattr(model, "_foundationgeo_force_fp32", False))
    kwargs: dict[str, Any] = {
        "resolution_level": int(resolution_level),
        "use_fp16": bool(fp16) and not force_fp32,
        "fov_x": float(fov_x),
        "apply_mask": False,
    }
    if num_tokens is not None and num_tokens > 0:
        kwargs["num_tokens"] = int(num_tokens)
    with torch.inference_mode():
        output = model.infer(tensor, **kwargs)

    fp32_fallback_used = False
    depth = output["depth_metric"]
    if fp16 and not np.isfinite(depth.detach().float().cpu().numpy()).any():
        model.float()
        setattr(model, "_foundationgeo_force_fp32", True)
        kwargs["use_fp16"] = False
        with torch.inference_mode():
            output = model.infer(tensor, **kwargs)
        depth = output["depth_metric"]
        fp32_fallback_used = True

    return {
        "depth": depth.detach().float().cpu().numpy(),
        "mask": output["mask"].detach().cpu().numpy().astype(bool),
        "fp32_fallback_used": fp32_fallback_used,
    }


def choose_sample_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if sample_count <= 0 or sample_count >= frame_count:
        return list(range(frame_count))
    return sorted(set(np.linspace(0, frame_count - 1, sample_count).round().astype(int).tolist()))


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir = args.output_dir.expanduser().resolve()
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.frames_dir.expanduser().resolve())
    timestamps = read_timestamps(args.timestamps_txt.expanduser().resolve())
    head_poses = load_head_trajectory(args.head_trajectory_in.expanduser().resolve())
    if not images:
        raise FileNotFoundError(f"No images found under {args.frames_dir}")
    frame_count = min(len(head_poses), max(len(images), len(timestamps)))
    sample_indices = choose_sample_indices(frame_count, int(args.num_samples))
    if not sample_indices:
        raise ValueError("No frames available for FoundationGeo ground scale estimation")

    floor = load_floor_helpers()
    model = load_foundationgeo_model(args.foundationgeo_model_pt, args.device, args.fp16)
    id_to_image = image_id_index(images)
    frame_results: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    print(f"[FoundationGeo ground scale] sampled_frames={len(sample_indices)}", flush=True)

    for pose_index in sample_indices:
        frame: dict[str, Any] = {"pose_index": int(pose_index), "status": "failed"}
        try:
            image_path = find_image_for_pose(pose_index, timestamps, images, id_to_image)
            if image_path is None:
                frame["reason"] = "image_not_found"
                frame_results.append(frame)
                continue
            image_rgb = load_image_rgb(image_path)
            height, width = image_rgb.shape[:2]
            K, intrinsics_row = load_intrinsics_for_pose(
                args.intrinsics_path.expanduser().resolve(),
                pose_index,
                width,
                height,
            )
            fov_x = horizontal_fov_deg(K, width)
            inference = run_foundationgeo_depth(
                model,
                image_rgb,
                device=args.device,
                fp16=args.fp16,
                resolution_level=args.resolution_level,
                num_tokens=args.num_tokens,
                fov_x=fov_x,
            )
            depth = np.asarray(inference["depth"], dtype=np.float32)
            mask = np.asarray(inference["mask"], dtype=bool)
            if depth.shape != (height, width) or mask.shape != depth.shape:
                raise ValueError(
                    f"FoundationGeo depth/mask shapes {depth.shape}/{mask.shape} do not match {(height, width)}"
                )
            valid = mask & np.isfinite(depth) & (depth > 0.1) & (depth < 10.0)
            depth_path = depth_dir / f"{pose_index:06d}_depth_metric.npy"
            np.save(depth_path, depth)
            points_cam = floor.backproject_depth_to_camera_metric(
                np.where(valid, depth, np.nan),
                K,
                np.eye(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            )
            up_cam = floor.camera_frame_up(head_poses[pose_index], up_world)
            candidates = floor.detect_floor_candidates_along_gravity(points_cam, up_cam)
            aligned_height = float(head_poses[pose_index, 2, 3])
            if not np.isfinite(aligned_height) or aligned_height <= 1e-8:
                raise ValueError(f"Invalid aligned camera height: {aligned_height}")

            frame_candidates = []
            for candidate_index, candidate in enumerate(candidates):
                metric_height = float(-candidate["level"])
                scale = metric_height / aligned_height
                candidate_info = {
                    "candidate_index": int(candidate_index),
                    "floor_level": float(candidate["level"]),
                    "metric_camera_height": metric_height,
                    "aligned_vggt_height": aligned_height,
                    "scale": float(scale),
                    "support": int(candidate["support"]),
                    "coverage_m2": float(candidate["coverage_m2"]),
                    "residual": float(candidate["residual"]),
                    "height_rank": float(candidate["height_rank"]),
                    "score": float(candidate["score"]),
                    "selected_scale_cluster": False,
                }
                frame_candidates.append(candidate_info)
                if np.isfinite(scale) and scale > 0.0:
                    candidate_records.append({"pose_index": int(pose_index), **candidate_info})

            frame.update({
                "status": "ok" if frame_candidates else "failed",
                "reason": "depth_floor_candidates" if frame_candidates else "no_floor_candidate",
                "image_path": str(image_path),
                "depth_path": str(depth_path),
                "valid_depth_pixels": int(valid.sum()),
                "backprojected_points": int(len(points_cam)),
                "intrinsics_row": intrinsics_row,
                "intrinsics": K.tolist(),
                "fov_x_deg": fov_x,
                "aligned_vggt_height": aligned_height,
                "fp32_fallback_used": bool(inference["fp32_fallback_used"]),
                "candidates": frame_candidates,
            })
        except Exception as exc:
            frame.update({
                "reason": "exception",
                "exception": repr(exc),
                "traceback": traceback.format_exc(limit=5),
            })
        frame_results.append(frame)

    frames_with_candidates = len({int(record["pose_index"]) for record in candidate_records})
    base = {
        "source": "foundationgeo",
        "scale_method": "foundationgeo_depth_floor_implied_scale_cluster",
        "num_sampled_frames": int(len(sample_indices)),
        "num_valid_frames": int(frames_with_candidates),
        "sample_indices": [int(i) for i in sample_indices],
        "elapsed_sec": round(time.time() - started, 3),
        "config": {
            "num_samples": int(args.num_samples),
            "depth_scale_cluster_rel_tol": float(args.depth_scale_cluster_rel_tol),
            "depth_scale_min_support_frames": int(args.depth_scale_min_support_frames),
            "depth_backproject_stride": int(floor.DEPTH_BACKPROJECT_STRIDE),
            "device": str(args.device),
            "fp16": bool(args.fp16),
            "resolution_level": int(args.resolution_level),
            "num_tokens": args.num_tokens,
        },
        "inputs": {
            "frames_dir": str(args.frames_dir),
            "timestamps_txt": str(args.timestamps_txt),
            "head_trajectory_in": str(args.head_trajectory_in),
            "intrinsics_path": str(args.intrinsics_path),
            "foundationgeo_model_pt": str(args.foundationgeo_model_pt),
        },
        "outputs": {"depth_dir": str(depth_dir)},
        "frames": frame_results,
    }
    if frames_with_candidates < int(floor.FLOOR_MIN_VALID_FRAMES):
        return {
            **base,
            "status": "failed",
            "reason": "not_enough_frames_with_floor_candidates",
            "scale": None,
        }

    try:
        selected, cluster = floor.select_largest_median_scale_cluster(
            candidate_records,
            relative_tolerance=float(args.depth_scale_cluster_rel_tol),
            min_support_frames=int(args.depth_scale_min_support_frames),
        )
        cluster["selection_policy"] = "largest_median_implied_scale_cluster"
    except ValueError as exc:
        return {
            **base,
            "status": "failed",
            "reason": "implied_scale_clustering_failed",
            "scale": None,
            "exception": repr(exc),
        }

    selected_keys = {
        (int(record["pose_index"]), int(record["candidate_index"]))
        for record in selected
    }
    for frame in frame_results:
        for candidate in frame.get("candidates", []):
            candidate["selected_scale_cluster"] = (
                int(frame["pose_index"]), int(candidate["candidate_index"])
            ) in selected_keys
    scale, scale_info = floor.robust_scale_from_matched_frames(selected)
    if scale is None or not np.isfinite(scale) or scale <= 0.0:
        return {
            **base,
            "status": "failed",
            "reason": "invalid_ground_scale",
            "scale": None,
            "cluster": cluster,
            "scale_info": scale_info,
        }

    return {
        **base,
        "status": "ok",
        "reason": "gravity_floor_matched_vggt_frame",
        "scale": float(scale),
        "relative_mad": scale_info.get("relative_mad"),
        "num_selected_frames": int(len(selected)),
        "selected_frame_indices": sorted(int(record["pose_index"]) for record in selected),
        "metric_camera_height_median_m": float(
            np.median([float(record["metric_camera_height"]) for record in selected])
        ),
        "cluster": cluster,
        "scale_info": scale_info,
    }


def main() -> int:
    args = parse_args()
    try:
        result = estimate(args)
    except Exception as exc:
        result = {
            "status": "failed",
            "reason": "exception",
            "scale": None,
            "source": "foundationgeo",
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
        }
    write_json(args.output_json.expanduser().resolve(), result)
    return 2 if args.strict and result.get("status") != "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())
