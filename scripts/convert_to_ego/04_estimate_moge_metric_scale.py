#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate one global metric scale using MoGe metric depth.

This script is intentionally YAML-free. run_body_mesh.py owns the YAML config
and passes every option as explicit CLI arguments.

Inputs are the already gravity-aligned, but not yet metric-scaled, trajectory and
scene point cloud. For each sampled frame:
  1) run MoGe on the RGB frame and get metric depth in meters;
  2) project the current unscaled scene point cloud into that camera;
  3) estimate a robust scale = median(depth_moge / depth_recon);
  4) aggregate per-frame scales by median.

Failures are written to --output-json with status="failed" and exit code 0 by
default, so the main pipeline can fall back to hand-based scale or no scale.
Use --strict to turn failures into non-zero exit codes.
"""
from __future__ import annotations
import os
import sys

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import time
import traceback
from typing import Any

import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate metric scale from MoGe metric depth.")

    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--timestamps-txt", required=True, type=Path)
    parser.add_argument("--head-trajectory-in", required=True, type=Path)
    parser.add_argument("--point-cloud-in", required=True, type=Path)
    parser.add_argument("--intrinsics-path", required=True, type=Path)
    parser.add_argument("--moge-model-pt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)

    # Frame selection: first uniformly sample candidate frames, then run a CPU-only
    # image-quality check and select the best --num-samples frames for MoGe.
    parser.add_argument("--candidate-samples", type=int, default=48)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--min-quality-brightness", type=float, default=25.0)
    parser.add_argument("--max-quality-brightness", type=float, default=235.0)
    parser.add_argument("--max-quality-dark-ratio", type=float, default=0.45)
    parser.add_argument("--max-quality-bright-ratio", type=float, default=0.45)
    parser.add_argument("--min-quality-laplacian-var", type=float, default=20.0)
    parser.add_argument("--min-quality-texture-std", type=float, default=8.0)
    parser.add_argument("--min-quality-edge-density", type=float, default=0.01)
    parser.add_argument("--quality-edge-threshold", type=float, default=20.0)
    parser.add_argument("--min-valid-frames", type=int, default=6)
    parser.add_argument("--min-valid-pixels", type=int, default=1000)
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=8.0)
    parser.add_argument("--min-scale", type=float, default=0.05)
    parser.add_argument("--max-scale", type=float, default=20.0)
    parser.add_argument("--max-frame-relative-mad", type=float, default=0.35)
    parser.add_argument("--max-final-relative-mad", type=float, default=0.35)
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--max-saved-match-pixels", type=int, default=200000)

    parser.add_argument("--projection-quality-enabled", type=str2bool, default=True)
    parser.add_argument("--min-projection-area-ratio", type=float, default=0.01)
    parser.add_argument("--min-depth-range-m", type=float, default=0.25)
    parser.add_argument("--min-depth-layer-ratio", type=float, default=0.03)
    parser.add_argument("--min-depth-layer-count", type=int, default=2)
    parser.add_argument("--max-center-region-ratio", type=float, default=0.90)
    parser.add_argument("--min-static-background-score", type=float, default=0.35)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--use-intrinsics-fov", type=str2bool, default=True)

    parser.add_argument("--save-depth-maps", type=str2bool, default=True)
    parser.add_argument("--save-projection-matches", type=str2bool, default=True)
    parser.add_argument("--save-sampled-frames", type=str2bool, default=False)

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if MoGe scale estimation fails. Default is exit 0 for fallback.",
    )
    return parser.parse_args()


def load_head_trajectory(path: Path) -> np.ndarray:
    poses = np.asarray(np.load(path), dtype=np.float64)
    if poses.ndim == 2 and poses.shape == (4, 4):
        poses = poses[None, :, :]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{path} must have shape [N,4,4] or [4,4], got {poses.shape}")
    return poses


def read_timestamps(path: Path) -> list[dict[str, Any]]:
    table = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")
    if table.shape == ():
        table = np.asarray([table])
    names = set(table.dtype.names or [])
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(table):
        frame_id = int(row["frame_id"]) if "frame_id" in names else idx
        src_frame_id = int(row["src_frame_id"]) if "src_frame_id" in names else frame_id
        timestamp = float(row["timestamp"]) if "timestamp" in names else float(idx)
        rows.append({"frame_id": frame_id, "src_frame_id": src_frame_id, "timestamp": timestamp})
    return rows


def list_images(frames_dir: Path) -> list[Path]:
    paths = [p for p in frames_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(paths, key=lambda p: natural_key(p.name))


def natural_key(text: str) -> list[Any]:
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", text)]


def image_id_index(images: list[Path]) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for p in images:
        nums = re.findall(r"\d+", p.stem)
        if not nums:
            continue
        # Usually the frame id is the last numeric group in names like frame_000123.
        key = int(nums[-1])
        mapping.setdefault(key, p)
    return mapping


def find_image_for_pose(
    pose_index: int,
    timestamp_rows: list[dict[str, Any]],
    sorted_images: list[Path],
    id_to_image: dict[int, Path],
) -> Path | None:
    candidates: list[int] = []
    if pose_index < len(timestamp_rows):
        row = timestamp_rows[pose_index]
        candidates.extend([int(row.get("src_frame_id", pose_index)), int(row.get("frame_id", pose_index))])
    candidates.append(int(pose_index))

    for idx in candidates:
        if idx in id_to_image:
            return id_to_image[idx]

    if 0 <= pose_index < len(sorted_images):
        return sorted_images[pose_index]
    return None


def load_intrinsics(path: Path, *, width: int, height: int) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        flat = arr.reshape(-1)
        if flat.size >= 9:
            K = flat[:9].reshape(3, 3)
        elif flat.size >= 4:
            fx, fy, cx, cy = flat[:4]
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        else:
            raise ValueError(f"Cannot parse intrinsics from {path}: shape={arr.shape}")
    elif arr.shape == (3, 3):
        K = arr
    else:
        flat = arr.reshape(-1)
        if flat.size >= 9:
            K = flat[:9].reshape(3, 3)
        elif flat.size >= 4:
            fx, fy, cx, cy = flat[:4]
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        else:
            raise ValueError(f"Cannot parse intrinsics from {path}: shape={arr.shape}")

    K = K.astype(np.float64, copy=True)
    # If intrinsics look normalized, convert to pixel units.
    if abs(K[0, 0]) <= 10.0 and abs(K[1, 1]) <= 10.0 and abs(K[0, 2]) <= 2.0 and abs(K[1, 2]) <= 2.0:
        K[0, 0] *= width
        K[0, 2] *= width
        K[1, 1] *= height
        K[1, 2] *= height
    return K


def horizontal_fov_deg_from_K(K: np.ndarray, width: int) -> float | None:
    fx = finite_float(K[0, 0])
    if fx is None or fx <= 1e-8:
        return None
    return float(np.degrees(2.0 * np.arctan(width / (2.0 * fx))))


def load_ply_points(path: Path) -> np.ndarray:
    try:
        from plyfile import PlyData

        ply = PlyData.read(str(path))
        v = ply["vertex"].data
        pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        return pts[np.isfinite(pts).all(axis=1)]
    except Exception:
        return load_ascii_ply_points(path)


def load_ascii_ply_points(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{path} is not a PLY file")
    vertex_count = None
    header_end = None
    props: list[str] = []
    in_vertex = False
    for idx, line in enumerate(lines):
        s = line.strip()
        if s.startswith("format ") and "ascii" not in s:
            raise ValueError(f"{path} is binary PLY and plyfile is not available")
        if s.startswith("element vertex "):
            vertex_count = int(s.split()[-1])
            in_vertex = True
            continue
        if s.startswith("element ") and not s.startswith("element vertex "):
            in_vertex = False
        if in_vertex and s.startswith("property "):
            props.append(s.split()[-1])
        if s == "end_header":
            header_end = idx
            break
    if vertex_count is None or header_end is None:
        raise ValueError(f"Cannot parse PLY header: {path}")
    for key in ("x", "y", "z"):
        if key not in props:
            raise ValueError(f"{path} vertex properties must contain x/y/z")
    xi, yi, zi = props.index("x"), props.index("y"), props.index("z")
    pts = []
    for line in lines[header_end + 1 : header_end + 1 + vertex_count]:
        parts = line.split()
        if len(parts) <= max(xi, yi, zi):
            continue
        pts.append([float(parts[xi]), float(parts[yi]), float(parts[zi])])
    arr = np.asarray(pts, dtype=np.float64)
    return arr[np.isfinite(arr).all(axis=1)]


def deterministic_sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).round().astype(np.int64)
    return points[idx]


def load_image_rgb(path: Path) -> np.ndarray:
    try:
        import cv2

        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"cv2.imread failed: {path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))



def _grayscale_float(image_rgb: np.ndarray) -> np.ndarray:
    img = np.asarray(image_rgb, dtype=np.float32)
    if img.ndim == 2:
        gray = img
    else:
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return np.clip(gray, 0.0, 255.0)


def laplacian_variance(gray: np.ndarray) -> float:
    gray = np.asarray(gray, dtype=np.float32)
    try:
        import cv2

        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except Exception:
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return 0.0
        lap = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        return float(np.var(lap))


def edge_density(gray: np.ndarray, threshold: float) -> float:
    gray = np.asarray(gray, dtype=np.float32)
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(mag > float(threshold)))


def image_quality_metrics(image_rgb: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    """CPU-only image-quality metrics used before running MoGe.

    The goal is not to predict MoGe quality perfectly. It only removes obvious bad
    frames: too dark/overexposed, too blurry, or too textureless. The final MoGe
    depth/projection/MAD checks still decide whether a selected frame contributes
    a valid metric scale.
    """
    gray = _grayscale_float(image_rgb)
    mean_brightness = float(np.mean(gray))
    dark_ratio = float(np.mean(gray <= 5.0))
    bright_ratio = float(np.mean(gray >= 250.0))
    texture_std = float(np.std(gray))
    lap_var = laplacian_variance(gray)
    edges = edge_density(gray, args.quality_edge_threshold)

    reasons: list[str] = []
    if mean_brightness < args.min_quality_brightness:
        reasons.append("too_dark")
    if mean_brightness > args.max_quality_brightness:
        reasons.append("too_bright")
    if dark_ratio > args.max_quality_dark_ratio:
        reasons.append("dark_ratio_too_high")
    if bright_ratio > args.max_quality_bright_ratio:
        reasons.append("bright_ratio_too_high")
    if lap_var < args.min_quality_laplacian_var:
        reasons.append("too_blurry")
    if texture_std < args.min_quality_texture_std:
        reasons.append("texture_too_low")
    if edges < args.min_quality_edge_density:
        reasons.append("edge_density_too_low")

    # Scores are deliberately soft. A frame can fail a hard threshold but still be
    # selected when there are not enough good frames; MoGe/post-projection checks
    # remain the authoritative filter.
    center = 0.5 * (float(args.min_quality_brightness) + float(args.max_quality_brightness))
    half_range = max(1.0, 0.5 * (float(args.max_quality_brightness) - float(args.min_quality_brightness)))
    brightness_score = float(np.clip(1.0 - abs(mean_brightness - center) / half_range, 0.0, 1.0))
    exposure_score = float(
        np.clip(1.0 - max(
            dark_ratio / max(float(args.max_quality_dark_ratio), 1e-6),
            bright_ratio / max(float(args.max_quality_bright_ratio), 1e-6),
        ), 0.0, 1.0)
    )
    blur_score = float(np.clip(lap_var / max(float(args.min_quality_laplacian_var), 1e-6), 0.0, 1.0))
    texture_score = float(np.clip(texture_std / max(float(args.min_quality_texture_std), 1e-6), 0.0, 1.0))
    edge_score = float(np.clip(edges / max(float(args.min_quality_edge_density), 1e-6), 0.0, 1.0))
    quality_score = float(
        0.25 * brightness_score
        + 0.15 * exposure_score
        + 0.25 * blur_score
        + 0.20 * texture_score
        + 0.15 * edge_score
    )

    return {
        "quality_score": quality_score,
        "quality_passed": len(reasons) == 0,
        "quality_reasons": reasons,
        "brightness_mean": mean_brightness,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "laplacian_var": lap_var,
        "texture_std": texture_std,
        "edge_density": edges,
        "brightness_score": brightness_score,
        "exposure_score": exposure_score,
        "blur_score": blur_score,
        "texture_score": texture_score,
        "edge_score": edge_score,
    }


def select_frames_by_cpu_quality(
    candidate_indices: list[int],
    *,
    timestamps: list[dict[str, Any]],
    images: list[Path],
    id_to_image: dict[int, Path],
    args: argparse.Namespace,
) -> tuple[list[int], list[dict[str, Any]], str | None]:
    """Evaluate CPU image quality on candidate frames and select frames for MoGe.

    Selection policy:
      1. uniformly sampled candidates are scored on CPU;
      2. if at least --num-samples pass all quality thresholds, select the best
         passing frames;
      3. otherwise select the best available frames anyway and leave final
         acceptance to MoGe depth/projection/MAD checks.
    """
    records: list[dict[str, Any]] = []
    for pose_index in candidate_indices:
        rec: dict[str, Any] = {
            "pose_index": int(pose_index),
            "status": "failed",
            "selected_for_moge": False,
        }
        try:
            image_path = find_image_for_pose(pose_index, timestamps, images, id_to_image)
            if image_path is None:
                rec.update({"reason": "image_not_found", "quality_score": float("-inf"), "quality_passed": False})
                records.append(rec)
                continue
            image_rgb = load_image_rgb(image_path)
            h, w = image_rgb.shape[:2]
            metrics = image_quality_metrics(image_rgb, args)
            rec.update({
                "status": "ok",
                "reason": "cpu_quality_checked",
                "image_path": str(image_path),
                "image_width": int(w),
                "image_height": int(h),
                **metrics,
            })
        except Exception as exc:
            rec.update({
                "reason": "quality_check_exception",
                "exception": repr(exc),
                "quality_score": float("-inf"),
                "quality_passed": False,
            })
        records.append(rec)

    usable = [r for r in records if np.isfinite(float(r.get("quality_score", float("-inf"))))]
    if not usable:
        return [], records, "no_quality_usable_frames"

    passed = [r for r in usable if bool(r.get("quality_passed", False))]
    select_pool = passed if len(passed) >= int(args.num_samples) else usable
    warning = None
    if len(passed) < int(args.num_samples):
        warning = (
            f"Only {len(passed)} / {len(candidate_indices)} candidate frames passed CPU quality thresholds; "
            f"selected the best {min(int(args.num_samples), len(usable))} available frames and will rely on MoGe/MAD checks."
        )

    top = sorted(select_pool, key=lambda r: float(r.get("quality_score", float("-inf"))), reverse=True)[: int(args.num_samples)]
    selected_ids = {int(r["pose_index"]) for r in top}
    rank_by_id = {int(r["pose_index"]): rank for rank, r in enumerate(top, start=1)}
    for r in records:
        pid = int(r["pose_index"])
        if pid in selected_ids:
            r["selected_for_moge"] = True
            r["quality_rank"] = int(rank_by_id[pid])
    return sorted(selected_ids), records, warning


def add_moge_root_to_path() -> None:
    moge_root = os.getenv("MOGE_ROOT")
    if not moge_root:
        return
    p = Path(moge_root).expanduser().resolve()
    if p.exists():
        sys.path.insert(0, str(p))


def load_moge_model(model_pt: Path, device: str):
    add_moge_root_to_path()
    import torch
    from moge.model.v2 import MoGeModel

    model_pt = model_pt.expanduser().resolve()
    if not model_pt.is_file():
        raise FileNotFoundError(f"MoGe model.pt not found: {model_pt}")
    model = MoGeModel.from_pretrained(str(model_pt)).to(torch.device(device)).eval()
    return model


def run_moge_depth(model, image_rgb: np.ndarray, *, device: str, fp16: bool, resolution_level: int, num_tokens: int | None, fov_x: float | None) -> dict[str, np.ndarray]:
    import torch

    tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=torch.device(device)).permute(2, 0, 1)
    kwargs: dict[str, Any] = {
        "resolution_level": int(resolution_level),
        "use_fp16": bool(fp16),
    }
    if num_tokens is not None and int(num_tokens) > 0:
        kwargs["num_tokens"] = int(num_tokens)
    if fov_x is not None and math.isfinite(float(fov_x)):
        kwargs["fov_x"] = float(fov_x)

    with torch.inference_mode():
        output = model.infer(tensor, **kwargs)

    result: dict[str, np.ndarray] = {}
    for key in ("depth", "mask", "intrinsics"):
        if key in output and output[key] is not None:
            result[key] = output[key].detach().cpu().numpy()
    if "depth" not in result:
        raise KeyError("MoGe output does not contain 'depth'")
    return result


def project_point_cloud_depth(points_world: np.ndarray, T_world_cam: np.ndarray, K: np.ndarray, width: int, height: int, min_depth: float, max_depth: float) -> np.ndarray:
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    # T_world_cam is assumed to map camera coordinates to world coordinates.
    points_cam = (points_world - t.reshape(1, 3)) @ R
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > min_depth) & (z < max_depth)
    if not np.any(valid):
        return np.full((height, width), np.inf, dtype=np.float32)

    pc = points_cam[valid]
    z = pc[:, 2]
    u = K[0, 0] * (pc[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (pc[:, 1] / z) + K[1, 2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not np.any(inside):
        return np.full((height, width), np.inf, dtype=np.float32)

    flat_idx = vi[inside] * width + ui[inside]
    z_inside = z[inside].astype(np.float32)
    zbuf = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(zbuf, flat_idx, z_inside)
    return zbuf.reshape(height, width)


def projection_quality_metrics(
    valid_mask: np.ndarray,
    depth_moge: np.ndarray,
    depth_recon: np.ndarray,
    *,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    valid_mask = np.asarray(valid_mask, dtype=bool)
    valid_pixels = int(np.count_nonzero(valid_mask))
    image_pixels = max(1, int(width) * int(height))
    out: dict[str, Any] = {
        "projection_quality_enabled": bool(args.projection_quality_enabled),
        "projection_quality_passed": True,
        "projection_quality_reasons": [],
        "valid_pixel_ratio": float(valid_pixels / image_pixels),
        "projection_area_ratio": 0.0,
        "projection_bbox_width_ratio": 0.0,
        "projection_bbox_height_ratio": 0.0,
        "depth_range_m": None,
        "depth_p05_m": None,
        "depth_p50_m": None,
        "depth_p95_m": None,
        "depth_near_ratio": 0.0,
        "depth_mid_ratio": 0.0,
        "depth_far_ratio": 0.0,
        "depth_layer_count": 0,
        "center_region_ratio": None,
        "over_center_concentrated": None,
        "static_background_score": None,
        "static_background_like": None,
    }
    if valid_pixels == 0:
        out["projection_quality_passed"] = False
        out["projection_quality_reasons"] = ["no_valid_projection_points"]
        return out

    vv, uu = np.nonzero(valid_mask)
    u_min, u_max = int(np.min(uu)), int(np.max(uu))
    v_min, v_max = int(np.min(vv)), int(np.max(vv))
    bbox_w = u_max - u_min + 1
    bbox_h = v_max - v_min + 1
    area_ratio = float((bbox_w * bbox_h) / image_pixels)
    width_ratio = float(bbox_w / max(1, int(width)))
    height_ratio = float(bbox_h / max(1, int(height)))

    depths = np.asarray(depth_moge[valid_mask], dtype=np.float64)
    depths = depths[np.isfinite(depths)]
    if depths.size:
        p05, p50, p95 = np.percentile(depths, [5, 50, 95])
        depth_range = float(p95 - p05)
    else:
        p05 = p50 = p95 = None
        depth_range = None

    recon_depths = np.asarray(depth_recon[valid_mask], dtype=np.float64)
    recon_depths = recon_depths[np.isfinite(recon_depths)]
    if recon_depths.size and depth_range is not None and depth_range > 1e-8:
        layer_min, layer_max = np.percentile(recon_depths, [5, 95])
        near_cut = float(layer_min + (layer_max - layer_min) / 3.0)
        far_cut = float(layer_min + 2.0 * (layer_max - layer_min) / 3.0)
        near_ratio = float(np.mean(recon_depths < near_cut))
        mid_ratio = float(np.mean((recon_depths >= near_cut) & (recon_depths < far_cut)))
        far_ratio = float(np.mean(recon_depths >= far_cut))
    else:
        near_ratio = mid_ratio = far_ratio = 0.0
    layer_ratios = [near_ratio, mid_ratio, far_ratio]
    layer_count = int(sum(r >= float(args.min_depth_layer_ratio) for r in layer_ratios))

    u_norm = (uu.astype(np.float64) + 0.5) / max(1.0, float(width))
    v_norm = (vv.astype(np.float64) + 0.5) / max(1.0, float(height))
    center_region = (np.abs(u_norm - 0.5) <= 0.25) & (np.abs(v_norm - 0.5) <= 0.25)
    center_region_ratio = float(np.mean(center_region))
    over_center = bool(center_region_ratio > float(args.max_center_region_ratio))

    coverage_score = float(np.clip(area_ratio / max(float(args.min_projection_area_ratio), 1e-6), 0.0, 1.0))
    depth_score = float(np.clip((depth_range or 0.0) / max(float(args.min_depth_range_m), 1e-6), 0.0, 1.0))
    layer_score = float(np.clip(layer_count / max(1, int(args.min_depth_layer_count)), 0.0, 1.0))
    spread_score = float(np.clip(1.0 - center_region_ratio / max(float(args.max_center_region_ratio), 1e-6), 0.0, 1.0))
    mid_far_ratio = float(mid_ratio + far_ratio)
    static_background_score = float(
        0.30 * coverage_score
        + 0.25 * depth_score
        + 0.20 * layer_score
        + 0.15 * spread_score
        + 0.10 * np.clip(mid_far_ratio, 0.0, 1.0)
    )

    reasons: list[str] = []
    if area_ratio < float(args.min_projection_area_ratio):
        reasons.append("projection_area_too_small")
    if depth_range is None or depth_range < float(args.min_depth_range_m):
        reasons.append("depth_range_too_small")
    if layer_count < int(args.min_depth_layer_count):
        reasons.append("insufficient_depth_layers")
    if over_center:
        reasons.append("projection_over_center_concentrated")
    if static_background_score < float(args.min_static_background_score):
        reasons.append("static_background_support_too_low")

    out.update({
        "projection_quality_passed": (not bool(args.projection_quality_enabled)) or len(reasons) == 0,
        "projection_quality_reasons": reasons,
        "projection_area_ratio": area_ratio,
        "projection_bbox_width_ratio": width_ratio,
        "projection_bbox_height_ratio": height_ratio,
        "depth_range_m": depth_range,
        "depth_p05_m": None if p05 is None else float(p05),
        "depth_p50_m": None if p50 is None else float(p50),
        "depth_p95_m": None if p95 is None else float(p95),
        "depth_near_ratio": near_ratio,
        "depth_mid_ratio": mid_ratio,
        "depth_far_ratio": far_ratio,
        "depth_layer_count": layer_count,
        "center_region_ratio": center_region_ratio,
        "over_center_concentrated": over_center,
        "static_background_score": static_background_score,
        "static_background_like": bool(static_background_score >= float(args.min_static_background_score)),
    })
    return out


def relative_mad(values: np.ndarray, center: float | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    med = float(np.median(values)) if center is None else float(center)
    denom = max(abs(med), 1e-8)
    return float(np.median(np.abs(values - med)) / denom)


def choose_sample_indices(num_frames: int, num_samples: int) -> list[int]:
    if num_frames <= 0:
        return []
    if num_samples <= 0 or num_samples >= num_frames:
        return list(range(num_frames))
    return sorted(set(np.linspace(0, num_frames - 1, num_samples).round().astype(int).tolist()))


def estimate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = args.frames_dir.expanduser().resolve()
    timestamps = read_timestamps(args.timestamps_txt.expanduser().resolve())
    head_poses = load_head_trajectory(args.head_trajectory_in.expanduser().resolve())
    points = load_ply_points(args.point_cloud_in.expanduser().resolve())
    points = deterministic_sample_points(points, args.max_points)

    images = list_images(frames_dir)
    if not images:
        raise FileNotFoundError(f"No images found under frames_dir: {frames_dir}")
    id_to_image = image_id_index(images)

    num_frames = min(len(head_poses), max(len(timestamps), len(images)))
    candidate_count = int(args.candidate_samples) if int(args.candidate_samples) > 0 else int(args.num_samples)
    candidate_count = max(candidate_count, int(args.num_samples))
    candidate_indices = choose_sample_indices(num_frames, candidate_count)
    if not candidate_indices:
        raise ValueError("No frames available for MoGe scale estimation")

    sample_indices, quality_candidates, quality_warning = select_frames_by_cpu_quality(
        candidate_indices,
        timestamps=timestamps,
        images=images,
        id_to_image=id_to_image,
        args=args,
    )
    if not sample_indices:
        raise ValueError("No candidate frames survived CPU quality preselection")

    print(
        f"[MoGe quality] candidates={len(candidate_indices)}, "
        f"quality_passed={sum(1 for r in quality_candidates if r.get('quality_passed'))}, "
        f"selected_for_moge={len(sample_indices)}",
        flush=True,
    )
    if quality_warning:
        print(f"[MoGe quality][Warning] {quality_warning}", flush=True)

    # Load the heavy MoGe model only after CPU preselection.
    model = load_moge_model(args.moge_model_pt, args.device)

    depth_dir = output_dir / "moge_depth_maps"
    matches_dir = output_dir / "projection_matches"
    sampled_dir = output_dir / "sampled_frames"
    if args.save_depth_maps:
        depth_dir.mkdir(parents=True, exist_ok=True)
    if args.save_projection_matches:
        matches_dir.mkdir(parents=True, exist_ok=True)
    if args.save_sampled_frames:
        sampled_dir.mkdir(parents=True, exist_ok=True)

    frame_results: list[dict[str, Any]] = []
    valid_scales: list[float] = []

    quality_by_pose = {int(r["pose_index"]): r for r in quality_candidates}

    for pose_index in sample_indices:
        qrec = quality_by_pose.get(int(pose_index), {})
        frame_info: dict[str, Any] = {
            "pose_index": int(pose_index),
            "status": "failed",
            "quality_score": finite_float(qrec.get("quality_score"), None),
            "quality_passed": bool(qrec.get("quality_passed", False)),
            "quality_rank": qrec.get("quality_rank"),
            "quality_reasons": qrec.get("quality_reasons", []),
            "brightness_mean": finite_float(qrec.get("brightness_mean"), None),
            "laplacian_var": finite_float(qrec.get("laplacian_var"), None),
            "texture_std": finite_float(qrec.get("texture_std"), None),
            "edge_density": finite_float(qrec.get("edge_density"), None),
        }
        try:
            image_path = find_image_for_pose(pose_index, timestamps, images, id_to_image)
            if image_path is None:
                frame_info["reason"] = "image_not_found"
                frame_results.append(frame_info)
                continue

            image_rgb = load_image_rgb(image_path)
            height, width = image_rgb.shape[:2]
            K = load_intrinsics(args.intrinsics_path.expanduser().resolve(), width=width, height=height)
            fov_x = horizontal_fov_deg_from_K(K, width) if args.use_intrinsics_fov else None

            moge = run_moge_depth(
                model,
                image_rgb,
                device=args.device,
                fp16=args.fp16,
                resolution_level=args.resolution_level,
                num_tokens=args.num_tokens,
                fov_x=fov_x,
            )
            depth_moge = np.asarray(moge["depth"], dtype=np.float32)
            if depth_moge.shape != (height, width):
                raise ValueError(f"MoGe depth shape {depth_moge.shape} does not match image {(height, width)}")
            mask_moge = np.asarray(moge.get("mask", np.isfinite(depth_moge)), dtype=bool)
            if mask_moge.shape != depth_moge.shape:
                mask_moge = np.isfinite(depth_moge)

            depth_recon = project_point_cloud_depth(
                points,
                head_poses[pose_index],
                K,
                width=width,
                height=height,
                min_depth=args.min_depth_m,
                max_depth=args.max_depth_m,
            )

            valid = (
                mask_moge
                & np.isfinite(depth_moge)
                & np.isfinite(depth_recon)
                & (depth_moge > args.min_depth_m)
                & (depth_moge < args.max_depth_m)
                & (depth_recon > args.min_depth_m)
                & (depth_recon < args.max_depth_m)
            )
            ratio_all = depth_moge[valid].astype(np.float64) / depth_recon[valid].astype(np.float64)
            ratio_ok_flat = np.isfinite(ratio_all) & (ratio_all >= args.min_scale) & (ratio_all <= args.max_scale)
            ratio_valid = np.zeros_like(valid, dtype=bool)
            valid_idx_all = np.flatnonzero(valid)
            ratio_valid.flat[valid_idx_all[ratio_ok_flat]] = True
            projection_metrics = projection_quality_metrics(
                ratio_valid,
                depth_moge,
                depth_recon,
                width=width,
                height=height,
                args=args,
            )
            ratio = ratio_all[ratio_ok_flat]
            valid_pixels = int(ratio.size)
            if valid_pixels < args.min_valid_pixels:
                frame_info.update({
                    "reason": "not_enough_valid_pixels",
                    "image_path": str(image_path),
                    "valid_pixels": valid_pixels,
                    **projection_metrics,
                })
                frame_results.append(frame_info)
                continue
            if not projection_metrics["projection_quality_passed"]:
                frame_info.update({
                    "reason": "projection_quality_rejected",
                    "image_path": str(image_path),
                    "valid_pixels": valid_pixels,
                    **projection_metrics,
                })
                frame_results.append(frame_info)
                continue

            scale = float(np.median(ratio))
            rel_mad = relative_mad(ratio, scale)
            if rel_mad > args.max_frame_relative_mad:
                frame_info.update({
                    "reason": "frame_relative_mad_too_large",
                    "image_path": str(image_path),
                    "scale": scale,
                    "valid_pixels": valid_pixels,
                    "relative_mad": rel_mad,
                    **projection_metrics,
                })
                frame_results.append(frame_info)
                continue

            frame_info.update({
                "status": "ok",
                "reason": "moge_depth_scene_ratio",
                "image_path": str(image_path),
                "frame_id": int(timestamps[pose_index]["frame_id"]) if pose_index < len(timestamps) else int(pose_index),
                "src_frame_id": int(timestamps[pose_index]["src_frame_id"]) if pose_index < len(timestamps) else int(pose_index),
                "timestamp": float(timestamps[pose_index]["timestamp"]) if pose_index < len(timestamps) else float(pose_index),
                "image_width": int(width),
                "image_height": int(height),
                "fov_x_deg": fov_x,
                "scale": scale,
                "valid_pixels": valid_pixels,
                "relative_mad": rel_mad,
                **projection_metrics,
            })
            frame_results.append(frame_info)
            valid_scales.append(scale)

            stem = f"frame_{pose_index:06d}"
            if args.save_depth_maps:
                np.savez_compressed(
                    depth_dir / f"{stem}_moge_depth.npz",
                    depth=depth_moge.astype(np.float32),
                    mask=mask_moge.astype(np.uint8),
                    intrinsics=np.asarray(moge.get("intrinsics", K), dtype=np.float32),
                    K_recon=K.astype(np.float32),
                    scale=np.float32(scale),
                )
            if args.save_projection_matches:
                valid_idx = np.flatnonzero(ratio_valid)
                if valid_idx.size > args.max_saved_match_pixels:
                    keep = np.linspace(0, valid_idx.size - 1, args.max_saved_match_pixels).round().astype(np.int64)
                    valid_idx = valid_idx[keep]
                vv, uu = np.unravel_index(valid_idx, (height, width))
                recon = depth_recon[vv, uu].astype(np.float32)
                md = depth_moge[vv, uu].astype(np.float32)
                rr = (md.astype(np.float64) / recon.astype(np.float64)).astype(np.float32)
                np.savez_compressed(
                    matches_dir / f"{stem}_matches.npz",
                    u=uu.astype(np.int32),
                    v=vv.astype(np.int32),
                    depth_moge=md,
                    depth_recon=recon,
                    ratio=rr,
                    scale=np.float32(scale),
                )
            if args.save_sampled_frames:
                shutil.copy2(image_path, sampled_dir / image_path.name)

        except Exception as exc:
            frame_info.update({
                "reason": "exception",
                "exception": repr(exc),
                "traceback": traceback.format_exc(limit=5),
            })
            frame_results.append(frame_info)

    valid_scales_arr = np.asarray(valid_scales, dtype=np.float64)
    if valid_scales_arr.size < int(args.min_valid_frames):
        status = "failed"
        reason = "not_enough_valid_frames"
        final_scale = None
        final_rel_mad = None
    else:
        final_scale_f = float(np.median(valid_scales_arr))
        final_rel_mad_f = relative_mad(valid_scales_arr, final_scale_f)
        if final_rel_mad_f > args.max_final_relative_mad:
            status = "failed"
            reason = "final_relative_mad_too_large"
            final_scale = None
            final_rel_mad = final_rel_mad_f
        else:
            status = "ok"
            reason = "moge_depth_scene_ratio"
            final_scale = final_scale_f
            final_rel_mad = final_rel_mad_f

    return {
        "status": status,
        "reason": reason,
        "scale": final_scale,
        "relative_mad": final_rel_mad,
        "num_candidate_frames": int(len(candidate_indices)),
        "num_quality_passed_frames": int(sum(1 for r in quality_candidates if r.get("quality_passed"))),
        "num_sampled_frames": int(len(sample_indices)),
        "num_valid_frames": int(valid_scales_arr.size),
        "candidate_indices": [int(i) for i in candidate_indices],
        "sample_indices": [int(i) for i in sample_indices],
        "quality_selection_warning": quality_warning,
        "elapsed_sec": round(time.time() - started, 3),
        "config": {
            "candidate_samples": int(args.candidate_samples),
            "num_samples": int(args.num_samples),
            "min_quality_brightness": float(args.min_quality_brightness),
            "max_quality_brightness": float(args.max_quality_brightness),
            "max_quality_dark_ratio": float(args.max_quality_dark_ratio),
            "max_quality_bright_ratio": float(args.max_quality_bright_ratio),
            "min_quality_laplacian_var": float(args.min_quality_laplacian_var),
            "min_quality_texture_std": float(args.min_quality_texture_std),
            "min_quality_edge_density": float(args.min_quality_edge_density),
            "quality_edge_threshold": float(args.quality_edge_threshold),
            "min_valid_frames": int(args.min_valid_frames),
            "min_valid_pixels": int(args.min_valid_pixels),
            "min_depth_m": float(args.min_depth_m),
            "max_depth_m": float(args.max_depth_m),
            "min_scale": float(args.min_scale),
            "max_scale": float(args.max_scale),
            "max_frame_relative_mad": float(args.max_frame_relative_mad),
            "max_final_relative_mad": float(args.max_final_relative_mad),
            "max_points": int(args.max_points),
            "projection_quality_enabled": bool(args.projection_quality_enabled),
            "min_projection_area_ratio": float(args.min_projection_area_ratio),
            "min_depth_range_m": float(args.min_depth_range_m),
            "min_depth_layer_ratio": float(args.min_depth_layer_ratio),
            "min_depth_layer_count": int(args.min_depth_layer_count),
            "max_center_region_ratio": float(args.max_center_region_ratio),
            "min_static_background_score": float(args.min_static_background_score),
            "device": str(args.device),
            "fp16": bool(args.fp16),
            "resolution_level": int(args.resolution_level),
            "num_tokens": int(args.num_tokens) if args.num_tokens is not None else None,
            "use_intrinsics_fov": bool(args.use_intrinsics_fov),
        },
        "inputs": {
            "frames_dir": str(args.frames_dir),
            "timestamps_txt": str(args.timestamps_txt),
            "head_trajectory_in": str(args.head_trajectory_in),
            "point_cloud_in": str(args.point_cloud_in),
            "intrinsics_path": str(args.intrinsics_path),
            "moge_model_pt": str(args.moge_model_pt),
        },
        "outputs": {
            "output_dir": str(args.output_dir),
            "depth_maps_dir": str(depth_dir) if args.save_depth_maps else None,
            "projection_matches_dir": str(matches_dir) if args.save_projection_matches else None,
            "sampled_frames_dir": str(sampled_dir) if args.save_sampled_frames else None,
        },
        "quality_candidates": quality_candidates,
        "frames": frame_results,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = estimate(args)
    except Exception as exc:
        result = {
            "status": "failed",
            "reason": "exception",
            "scale": None,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "inputs": {
                "frames_dir": str(args.frames_dir),
                "timestamps_txt": str(args.timestamps_txt),
                "head_trajectory_in": str(args.head_trajectory_in),
                "point_cloud_in": str(args.point_cloud_in),
                "intrinsics_path": str(args.intrinsics_path),
                "moge_model_pt": str(args.moge_model_pt),
            },
        }

    write_json(args.output_json.expanduser().resolve(), result)
    print("==== MoGe metric scale estimate ====")
    console_result = {
        k: v
        for k, v in result.items()
        if k not in {"frames", "quality_candidates"}
    }
    print(json.dumps(console_result, ensure_ascii=False, indent=2))
    if result.get("status") != "ok" and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
