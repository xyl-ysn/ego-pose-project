#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate gravity alignment rotation matrix for DA3 output.

Input:
    combined_pcd.ply
    camera_poses.txt

Output:
    R_align.npy

Assumptions:
    1. camera_poses.txt stores c2w poses.
    2. If a pose row is translation + quaternion, the quaternion order is qx qy qz qw.
    3. EgoAllo aligned world uses:
           +Y = up
           -Y = gravity
           X-Z = horizontal floor plane

This script only estimates and saves R_align:

    P_aligned = R_align @ P_da3_world

It does not modify the input point cloud or poses.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.optimize import minimize


# ============================================================
# Fixed internal parameters
# ============================================================

RANDOM_SEED = 0
MAX_ESTIMATION_POINTS = 300000
MAX_NN_POINTS = 50000
N_PLANES = 8
RANSAC_N = 3
RANSAC_ITERATIONS = 2500
MIN_INLIER_RATIO = 0.006
MIN_INLIERS_ABS = 300
REFINE_ITERS = 4

# Scale-adaptive ratios. These avoid hard-coded meter thresholds.
VOXEL_RATIO = 0.0025
RANSAC_DIST_RATIO = 0.0040
HEIGHT_MARGIN_RATIO = 0.0050
REFINE_THRESH_RATIOS = [0.0100, 0.0070, 0.0045, 0.0030]

# GeoCalib fusion parameters.
GEOCALIB_SWITCH_ANGLE_DEG = 10.0
GEOCALIB_OUTLIER_ANGLE_DEG = 25.0
GEOCALIB_PRIOR_MAX_ANGLE_DEG = 35.0

# When GeoCalib disagrees with point-cloud RANSAC, search near the lower
# envelope along GeoCalib up instead of only reweighting global plane candidates.
LOW_PRIOR_PERCENTILES = [8.0, 12.0, 18.0, 25.0, 35.0]
LOW_PRIOR_MAX_NORMAL_ANGLE_DEG = 18.0
LOW_PRIOR_MIN_INLIERS_ABS = 80
LOW_PRIOR_MIN_INLIER_RATIO = 0.015
LOW_PRIOR_PLANES_PER_BAND = 3
LOW_PRIOR_ACCEPT_MAX_HEIGHT_RANK = 0.35
LOW_PRIOR_REPLACE_HEIGHT_RANK_DELTA = 0.08
TABLETOP_RISK_HEIGHT_RANK = 0.35
MAX_BELOW_FLOOR_RATIO = 0.05
HAND_ABOVE_HEAD_MAX_RATIO = 0.5

# Real-IMU alignment parameters.
# With a real IMU the measured vertical drives the floor search: candidate planes
# are looked for in the low region along that axis instead of anywhere in the cloud.
# Connected patches are refined with the normal constrained to the configured
# IMU cone; the non-IMU path retains its original covariance refinement.
IMU_LOW_PERCENTILES = [2.0, 5.0, 8.0, 12.0, 18.0]
IMU_PLANES_PER_BAND = 8
IMU_FLOOR_MIN_INLIERS_ABS = 60
IMU_FLOOR_MIN_INLIER_RATIO = 0.004
IMU_FLOOR_CONNECTIVITY_VOXELS = 3.0
IMU_FLOOR_MIN_AREA_VOXELS = 64.0
IMU_FLOOR_MIN_RETAINED_RATIO = 0.7
IMU_MAX_REFINE_ANGLE_DEG = 10.0
IMU_MIN_CAMERA_ABOVE_HAND_RATIO = 0.5
IMU_MAX_HEAD_HEIGHT_REL_SPREAD = 0.75
IMU_MAX_POINTS_BELOW_FLOOR_RATIO = 0.05
IMU_MIN_HEAD_SPAN_RATIO = 0.15
IMU_MAX_HEAD_SPAN_RATIO = 2.50




class EmptyGeoCalibVectors(ValueError):
    """GeoCalib output exists but intentionally contains zero usable vectors."""

# ============================================================
# Basic utils
# ============================================================

def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n


def robust_percentile_range(x, low=5.0, high=95.0):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, high) - np.percentile(x, low))


def rotation_from_vectors(a, b):
    """
    Return R such that:

        R @ a = b
    """
    a = normalize(a)
    b = normalize(b)

    c = float(np.dot(a, b))

    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)

    if c < -1.0 + 1e-10:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(a, tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        axis = normalize(np.cross(a, tmp))
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)

    v = np.cross(a, b)
    s = np.linalg.norm(v)

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)

    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s ** 2))


def make_plane_basis(n):
    """
    Return two orthonormal vectors spanning the plane whose normal is n.
    """
    n = normalize(n)
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(n, tmp)) > 0.9:
        tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    u = normalize(np.cross(n, tmp))
    v = normalize(np.cross(n, u))
    return u, v


def angle_deg(a, b):
    a = normalize(a)
    b = normalize(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def robust_average_directions(vectors, outlier_angle_deg=GEOCALIB_OUTLIER_ANGLE_DEG):
    """
    Robustly average unit directions on the sphere.

    Returns:
        mean_direction, stats
    """
    V = np.asarray(vectors, dtype=np.float64)
    V = V[np.isfinite(V).all(axis=1)]
    if len(V) == 0:
        raise RuntimeError("No finite GeoCalib vectors.")

    V = np.array([normalize(v) for v in V], dtype=np.float64)
    initial = V.mean(axis=0)
    if np.linalg.norm(initial) < 1e-8:
        initial = V[0]
    initial = normalize(initial)

    # Guard against occasional sign-flipped estimates.
    V = np.array([v if np.dot(v, initial) >= 0 else -v for v in V], dtype=np.float64)
    initial = normalize(V.mean(axis=0))

    angles = np.array([angle_deg(v, initial) for v in V], dtype=np.float64)
    median_angle = float(np.median(angles))
    mad = float(np.median(np.abs(angles - median_angle)))
    adaptive_threshold = max(5.0, median_angle + 3.0 * 1.4826 * mad)
    threshold = min(float(outlier_angle_deg), adaptive_threshold)
    keep = angles <= threshold
    if np.sum(keep) < max(2, len(V) // 4):
        keep = angles <= float(outlier_angle_deg)
    if np.sum(keep) == 0:
        keep = np.ones(len(V), dtype=bool)

    mean_dir = normalize(V[keep].mean(axis=0))
    kept_angles = np.array([angle_deg(v, mean_dir) for v in V[keep]], dtype=np.float64)

    stats = {
        "num_vectors": int(len(V)),
        "num_kept": int(np.sum(keep)),
        "outlier_angle_deg": float(outlier_angle_deg),
        "median_angle_to_initial_deg": median_angle,
        "max_kept_angle_deg": float(np.max(kept_angles)) if len(kept_angles) else None,
        "mean_kept_angle_deg": float(np.mean(kept_angles)) if len(kept_angles) else None,
    }
    return mean_dir, stats


def _extract_vectors_from_json_object(obj):
    vector_keys = [
        "gravity",
        "gravity_cam",
        "gravity_camera",
        "g",
        "g_cam",
        "up",
        "up_cam",
        "up_camera",
        "vector",
    ]

    if isinstance(obj, dict):
        for key in vector_keys:
            if key in obj:
                arr = np.asarray(obj[key], dtype=np.float64)
                if arr.shape[-1] == 3:
                    return arr.reshape(-1, 3)
        for value in obj.values():
            try:
                out = _extract_vectors_from_json_object(value)
                if out is not None and len(out) > 0:
                    return out
            except Exception:
                pass

    if isinstance(obj, list):
        vectors = []
        for item in obj:
            out = _extract_vectors_from_json_object(item)
            if out is not None and len(out) > 0:
                vectors.append(out)
        if vectors:
            return np.concatenate(vectors, axis=0)

        arr = np.asarray(obj, dtype=np.float64)
        if arr.ndim >= 1 and arr.shape[-1] == 3:
            return arr.reshape(-1, 3)

    return None


def load_geocalib_vectors(path):
    """
    Load GeoCalib per-frame vectors in camera coordinates.

    Supported formats:
        .npy: [N,3] or [3]
        .npz: keys such as gravity, gravity_cam, up, up_cam
        .json: dict/list containing one of the keys above
        .txt/.csv: 3 columns, or more columns with the last 3 as vector

    If a txt/csv table has 4+ columns, the first column is treated as frame id.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    frame_ids = None

    if suffix == ".npy":
        vectors = np.load(path)

    elif suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            keys = list(data.keys())
            vector_key = None
            for key in [
                "gravity",
                "gravity_cam",
                "gravity_camera",
                "g",
                "g_cam",
                "up",
                "up_cam",
                "up_camera",
                "vector",
            ]:
                if key in data:
                    vector_key = key
                    break
            if vector_key is None:
                raise ValueError(f"No GeoCalib vector key found in {path}. Keys: {keys}")
            vectors = data[vector_key]
            for key in ["frame_ids", "frame_idx", "indices", "idx"]:
                if key in data:
                    frame_ids = np.asarray(data[key], dtype=np.int64).reshape(-1)
                    break

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        vectors = _extract_vectors_from_json_object(obj)
        if vectors is None:
            raise ValueError(f"Could not find GeoCalib vectors in JSON: {path}")

    else:
        try:
            table = np.loadtxt(path, delimiter=",", comments="#")
        except Exception:
            table = np.loadtxt(path, comments="#")
        table = np.asarray(table, dtype=np.float64)
        if table.ndim == 1:
            table = table[None, :]
        if table.shape[1] < 3:
            raise ValueError(f"GeoCalib table should have at least 3 columns: {path}")
        if table.shape[1] >= 4:
            frame_ids = table[:, 0].astype(np.int64)
        vectors = table[:, -3:]

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    if vectors.shape[-1] != 3:
        raise ValueError(f"GeoCalib vectors should have shape [N,3], got {vectors.shape}")

    vectors = vectors.reshape(-1, 3)
    vectors = vectors[np.isfinite(vectors).all(axis=1)]
    if len(vectors) == 0:
        skip_reason = ""
        if suffix == ".npz":
            try:
                with np.load(path, allow_pickle=True) as data:
                    if "skip_reason" in data:
                        skip_reason = str(np.asarray(data["skip_reason"]).item())
            except Exception:
                skip_reason = ""
        detail = f": {skip_reason}" if skip_reason else ""
        raise EmptyGeoCalibVectors(f"No finite GeoCalib vectors loaded from {path}{detail}")

    if frame_ids is not None and len(frame_ids) != len(vectors):
        print(
            "[Warning] GeoCalib frame id count does not match vector count; "
            "ignoring frame ids."
        )
        frame_ids = None

    return vectors, frame_ids


def estimate_up_from_geocalib(
    geocalib_path,
    Ts_c2w,
    vector_type="up",
    outlier_angle_deg=GEOCALIB_OUTLIER_ANGLE_DEG,
):
    """
    Convert GeoCalib camera-frame vectors to a DA3-world up direction.

    vector_type:
        gravity/down: input vector points along gravity, so up = -vector
        up:           input vector already points upward. This is the default
                      for the current GeoCalib output used in this pipeline.
    """
    vectors_cam, frame_ids = load_geocalib_vectors(geocalib_path)

    if frame_ids is not None:
        valid = (frame_ids >= 0) & (frame_ids < len(Ts_c2w))
        if not np.any(valid):
            raise ValueError("GeoCalib frame ids do not overlap pose indices.")
        vectors_cam = vectors_cam[valid]
        pose_indices = frame_ids[valid]
    elif len(vectors_cam) == len(Ts_c2w):
        pose_indices = np.arange(len(Ts_c2w), dtype=np.int64)
    else:
        pose_indices = np.linspace(0, len(Ts_c2w) - 1, len(vectors_cam))
        pose_indices = np.round(pose_indices).astype(np.int64)
        print(
            "[Warning] GeoCalib vector count does not match pose count and no frame ids "
            "were provided. Matching vectors to poses by uniform sampling."
        )

    vector_type = str(vector_type).lower()
    if vector_type in ["gravity", "down", "g"]:
        up_cam = -vectors_cam
    elif vector_type in ["up", "upright"]:
        up_cam = vectors_cam
    else:
        raise ValueError(f"Unsupported GeoCalib vector type: {vector_type}")

    up_world = []
    for v_cam, idx in zip(up_cam, pose_indices):
        R_world_cam = Ts_c2w[int(idx), :3, :3]
        up_world.append(R_world_cam @ normalize(v_cam))
    up_world = np.asarray(up_world, dtype=np.float64)

    up_geo, stats = robust_average_directions(up_world, outlier_angle_deg)
    stats.update({
        "path": str(Path(geocalib_path).expanduser().resolve()),
        "vector_type": vector_type,
        "num_pose_matched": int(len(pose_indices)),
        "pose_index_min": int(np.min(pose_indices)),
        "pose_index_max": int(np.max(pose_indices)),
    })
    return up_geo, stats


# ============================================================
# Pose loading: fixed c2w, fixed qx qy qz qw
# ============================================================

def quat_xyzw_to_rot(q):
    """
    Convert qx qy qz qw to rotation matrix.
    """
    q = np.asarray(q, dtype=np.float64)
    qx, qy, qz, qw = q
    return o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])


def load_da3_poses_c2w(path):
    """
    Load DA3 camera_poses.txt as c2w poses.

    Supported row formats:
        timestamp tx ty tz qx qy qz qw
        frame_id  tx ty tz qx qy qz qw
        tx ty tz qx qy qz qw
        3x4 matrix, 12 values
        id + 3x4 matrix, 13 values
        4x4 matrix, 16 values
        id + 4x4 matrix, 17 values

    No w2c inversion is performed.
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]

    Ts = []

    for row in data:
        row = np.asarray(row, dtype=np.float64)
        n = row.size

        if n == 16:
            T = row.reshape(4, 4)

        elif n == 17:
            T = row[1:].reshape(4, 4)

        elif n == 12:
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = row.reshape(3, 4)

        elif n == 13:
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = row[1:].reshape(3, 4)

        elif n >= 7:
            vals = row[-7:]
            t = vals[:3]
            q = vals[3:7]

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = quat_xyzw_to_rot(q)
            T[:3, 3] = t

        else:
            raise ValueError(f"Unsupported pose row with {n} values: {row}")

        Ts.append(T)

    return np.stack(Ts, axis=0)


def load_timestamps_txt(path):
    """Load frame_id/timestamp columns from timestamps.txt."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    try:
        table = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")
    except Exception:
        return None
    if table is None or "timestamp" not in table.dtype.names or "frame_id" not in table.dtype.names:
        return None
    return {
        "frame_ids": np.asarray(table["frame_id"], dtype=np.int64).reshape(-1),
        "timestamps": np.asarray(table["timestamp"], dtype=np.float64).reshape(-1),
    }


def load_hand_frames(path):
    """Load per-frame world-space hand keypoints from a HaWoR hands JSONL file."""
    if path is None:
        return []
    path = Path(path)
    if not path.is_file():
        print(f"[Warning] hands jsonl not found: {path}; skipping head/hand height check", flush=True)
        return []

    frames = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            frame_keypoints = []
            for hand in rec.get("hands", []):
                keypoints = hand.get("world_keypoints_3d")
                if keypoints is None:
                    continue
                arr = np.asarray(keypoints, dtype=np.float64).reshape(-1, 3)
                arr = arr[np.isfinite(arr).all(axis=1)]
                if arr.shape[0] > 0:
                    frame_keypoints.append(arr)

            if frame_keypoints:
                frames.append(
                    {
                        "timestamp": float(rec.get("timestamp", float("nan"))),
                        "frame_id": rec.get("frame_id"),
                        "keypoints": np.concatenate(frame_keypoints, axis=0),
                    }
                )
    return frames


def match_head_index(hand_frame, cam_centers, head_timestamps):
    n = len(cam_centers)
    frame_id = hand_frame.get("frame_id")

    if head_timestamps is not None and frame_id is not None:
        frame_ids = np.asarray(head_timestamps.get("frame_ids", []), dtype=np.int64)
        matches = np.flatnonzero(frame_ids == int(frame_id))
        if matches.size > 0:
            return int(matches[0])

    ts = hand_frame.get("timestamp")
    if head_timestamps is not None and np.isfinite(ts):
        timestamps = np.asarray(head_timestamps.get("timestamps", []), dtype=np.float64)
        if len(timestamps) == n:
            return int(np.argmin(np.abs(timestamps - ts)))

    if frame_id is not None:
        idx = int(frame_id)
        if 0 <= idx < n:
            return idx
    return None


def hand_above_head_stats(hand_frames, cam_centers, head_timestamps, floor_center, up):
    """Measure whether hands sit above their own frame's head/camera."""
    if not hand_frames or cam_centers is None or len(cam_centers) == 0:
        return {"available": False, "hand_above_head_ratio": None, "num_hand_frames": 0}

    cam_centers = np.asarray(cam_centers, dtype=np.float64)
    floor_center = np.asarray(floor_center, dtype=np.float64).reshape(3)
    up = normalize(up)

    per_frame_ratios = []
    matched_frames = 0
    for hand_frame in hand_frames:
        head_idx = match_head_index(hand_frame, cam_centers, head_timestamps)
        if head_idx is None:
            continue
        head_height = float((cam_centers[head_idx] - floor_center) @ up)
        hand_height = (hand_frame["keypoints"] - floor_center) @ up
        per_frame_ratios.append(float(np.mean(hand_height > head_height)))
        matched_frames += 1

    if not per_frame_ratios:
        return {"available": False, "hand_above_head_ratio": None, "num_hand_frames": 0}

    hand_above = float(np.mean(per_frame_ratios))
    return {
        "available": True,
        "hand_above_head_ratio": hand_above,
        "num_hand_frames": matched_frames,
        "per_frame_ratio_mean": hand_above,
    }


def validate_head_above_hands(hand_stats):
    if not hand_stats.get("available", False):
        return True, hand_stats.get("reason", "no_hand_height_data")
    ratio = float(hand_stats.get("hand_above_head_ratio", 1.0))
    if ratio >= HAND_ABOVE_HEAD_MAX_RATIO:
        return (
            False,
            f"hand_above_head_ratio_too_high: {ratio:.6g} >= {HAND_ABOVE_HEAD_MAX_RATIO:.6g}",
        )
    return True, "valid"


def resolve_imu_up_sign_with_hands(
    axis_up,
    hand_frames,
    cam_centers,
    head_timestamps,
    min_camera_above_hand_ratio,
):
    """Flip IMU up only when matched hands are mostly above the camera."""
    axis_up = normalize(axis_up)
    if not hand_frames or cam_centers is None or len(cam_centers) == 0:
        return axis_up, {
            "resolved": True,
            "reason": "no_hand_data_keep_device_convention",
            "flipped_from_input": False,
            "camera_above_hand_ratio": None,
            "threshold": float(min_camera_above_hand_ratio),
            "num_hand_frames": 0,
        }

    cam_centers = np.asarray(cam_centers, dtype=np.float64)
    per_frame_ratios = []
    matched_frames = 0
    total_keypoints = 0
    for hand_frame in hand_frames:
        head_idx = match_head_index(hand_frame, cam_centers, head_timestamps)
        if head_idx is None:
            continue
        head_height = float(cam_centers[head_idx] @ axis_up)
        hand_height = np.asarray(hand_frame["keypoints"], dtype=np.float64) @ axis_up
        hand_height = hand_height[np.isfinite(hand_height)]
        if len(hand_height) == 0:
            continue
        per_frame_ratios.append(float(np.mean(head_height > hand_height)))
        matched_frames += 1
        total_keypoints += int(len(hand_height))

    if not per_frame_ratios:
        return axis_up, {
            "resolved": True,
            "reason": "no_matched_hand_frames_keep_device_convention",
            "flipped_from_input": False,
            "camera_above_hand_ratio": None,
            "threshold": float(min_camera_above_hand_ratio),
            "num_hand_frames": 0,
        }

    ratio = float(np.mean(per_frame_ratios))
    flipped = ratio < min_camera_above_hand_ratio
    return normalize(-axis_up if flipped else axis_up), {
        "resolved": True,
        "reason": "camera_above_hands_failed_flip" if flipped else "camera_above_hands_pass",
        "flipped_from_input": bool(flipped),
        "camera_above_hand_ratio": ratio,
        "hand_above_camera_ratio": float(1.0 - ratio),
        "threshold": float(min_camera_above_hand_ratio),
        "num_hand_frames": int(matched_frames),
        "num_hand_keypoints": int(total_keypoints),
    }


# ============================================================
# Point cloud loading and scale-adaptive parameters
# ============================================================

def load_point_cloud(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {pcd_path}")

    P = np.asarray(pcd.points, dtype=np.float64)
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) == 0:
        raise RuntimeError("No finite points in point cloud.")
    return P


def estimate_nn_scale(P):
    """
    Estimate local point spacing from a random subset.
    Used only as a safety lower bound for thresholds.
    """
    if len(P) > MAX_NN_POINTS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(P), size=MAX_NN_POINTS, replace=False)
        P_nn = P[idx]
    else:
        P_nn = P

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P_nn)
    d = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float64)
    d = d[np.isfinite(d)]
    d = d[d > 0]
    if len(d) == 0:
        return 1e-4
    return float(np.median(d))


def compute_auto_params(P):
    bbox_min = P.min(axis=0)
    bbox_max = P.max(axis=0)
    bbox_range = bbox_max - bbox_min
    scene_extent = float(np.linalg.norm(bbox_range))
    if scene_extent <= 0:
        raise RuntimeError("Invalid point cloud extent.")

    nn = estimate_nn_scale(P)

    voxel_size = max(scene_extent * VOXEL_RATIO, 2.0 * nn, 1e-5)
    distance_threshold = max(scene_extent * RANSAC_DIST_RATIO, 3.0 * nn, 1e-5)
    height_margin = max(scene_extent * HEIGHT_MARGIN_RATIO, 3.0 * nn, 1e-5)
    refine_thresholds = [max(scene_extent * r, 2.0 * nn, 1e-5) for r in REFINE_THRESH_RATIOS]

    return {
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_range": bbox_range,
        "scene_extent": scene_extent,
        "nn_median": nn,
        "voxel_size": voxel_size,
        "distance_threshold": distance_threshold,
        "height_margin": height_margin,
        "refine_thresholds": refine_thresholds,
    }


def downsample_for_estimation(P, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd = pcd.voxel_down_sample(voxel_size)

    P_ds = np.asarray(pcd.points, dtype=np.float64)
    P_ds = P_ds[np.isfinite(P_ds).all(axis=1)]

    if len(P_ds) == 0:
        raise RuntimeError("No points after voxel downsampling.")

    if len(P_ds) > MAX_ESTIMATION_POINTS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(P_ds), size=MAX_ESTIMATION_POINTS, replace=False)
        P_ds = P_ds[idx]

    return P_ds


# ============================================================
# Plane fitting and candidate scoring
# ============================================================

def fit_normal_by_covariance(P_floor, prefer_dir=None):
    """
    Fit plane normal using only a 3x3 covariance matrix.
    """
    if len(P_floor) < 3:
        raise RuntimeError("Not enough points to fit a plane.")

    center = P_floor.mean(axis=0)
    X = P_floor - center
    C = (X.T @ X) / max(len(X) - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(C)
    n = normalize(eigvecs[:, np.argmin(eigvals)])

    if prefer_dir is not None:
        prefer_dir = normalize(prefer_dir)
        if np.dot(n, prefer_dir) < 0:
            n = -n

    residual = np.abs((P_floor - center) @ n)
    rms = float(np.sqrt(np.mean(residual ** 2)))

    return n, center, rms


def plane_inlier_area(P_in, center, normal):
    """
    Approximate robust 2D area of plane inliers.
    Useful to avoid selecting small dense table patches.
    """
    u, v = make_plane_basis(normal)
    Q = P_in - center
    cu = Q @ u
    cv = Q @ v
    ru = robust_percentile_range(cu, 5.0, 95.0)
    rv = robust_percentile_range(cv, 5.0, 95.0)
    return float(max(ru, 0.0) * max(rv, 0.0))


def detect_plane_candidates(P, params, cam_centers=None):
    """
    Fit up to N_PLANES valid floor candidates.

    A candidate is invalid when the chosen up direction leaves at least one
    camera center below the plane. Invalid candidates are removed from the
    working point cloud and do not consume one of the N_PLANES slots, so the
    loop keeps fitting until N_PLANES valid candidates are found or the point
    cloud is exhausted.
    """
    pcd_work = o3d.geometry.PointCloud()
    pcd_work.points = o3d.utility.Vector3dVector(P)

    total_points = len(P)
    min_inliers = max(MIN_INLIERS_ABS, int(total_points * MIN_INLIER_RATIO))
    candidates = []

    while len(candidates) < N_PLANES and len(pcd_work.points) >= min_inliers:
        if len(pcd_work.points) < min_inliers:
            break

        try:
            model, inliers = pcd_work.segment_plane(
                distance_threshold=params["distance_threshold"],
                ransac_n=RANSAC_N,
                num_iterations=RANSAC_ITERATIONS,
            )
        except RuntimeError:
            break

        if len(inliers) < min_inliers:
            break

        inlier_pcd = pcd_work.select_by_index(inliers)
        P_in = np.asarray(inlier_pcd.points, dtype=np.float64)

        n_ransac = normalize(np.asarray(model[:3], dtype=np.float64))
        n_refined, center, rms = fit_normal_by_covariance(P_in, prefer_dir=n_ransac)
        area = plane_inlier_area(P_in, center, n_refined)

        plane = {
            "idx": len(candidates),
            "normal": n_refined,
            "center": center,
            "inliers": int(len(inliers)),
            "ratio": float(len(inliers) / total_points),
            "rms": rms,
            "area": area,
        }

        up, _, _ = choose_up_direction(P, plane, cam_centers, params)
        invalid_camera = (
            cam_centers is not None
            and len(cam_centers) > 0
            and np.any((cam_centers - center) @ up < 0.0)
        )
        if invalid_camera:
            print(
                "[Warning] Rejecting plane candidate: camera centers are below "
                f"the fitted plane along the chosen up direction (inliers={len(inliers)}). "
                "This plane is not counted towards the valid plane budget.",
                flush=True,
            )
            pcd_work = pcd_work.select_by_index(inliers, invert=True)
            continue

        candidates.append(plane)

        pcd_work = pcd_work.select_by_index(inliers, invert=True)

    if len(candidates) == 0:
        raise RuntimeError("No valid plane candidates found.")

    return candidates


def choose_up_direction(P, plane, cam_centers, params, up_prior=None, prior_weight=0.0):
    """
    For one plane, choose normal sign using:
        1. most scene points should be above floor
        2. few scene points should be below floor
        3. camera centers should be above floor
    """
    n = normalize(plane["normal"])
    p0 = plane["center"]
    margin = params["height_margin"]
    extent = params["scene_extent"]

    best_up = None
    best_score = -1e18
    best_stats = None

    for sign in [1.0, -1.0]:
        up = sign * n
        h = (P - p0) @ up

        above = h > margin
        below = h < -margin

        above_ratio = float(np.mean(above))
        below_ratio = float(np.mean(below))

        below_depth = float(np.mean(np.abs(h[below])) / extent) if np.any(below) else 0.0
        above_height_median = float(np.median(h[above]) / extent) if np.any(above) else 0.0

        # Floor is expected to be a lower envelope of the reconstructed scene.
        score = 7.0 * above_ratio
        score -= 18.0 * below_ratio
        score -= 8.0 * below_depth
        score += 2.0 * np.tanh(above_height_median / 0.03)

        cam_height_median = None
        cam_height_std = None
        cam_above_ratio = None
        cam_below_ratio = None

        if cam_centers is not None and len(cam_centers) > 0:
            hc = (cam_centers - p0) @ up
            cam_above_ratio = float(np.mean(hc > margin))
            cam_below_ratio = float(np.mean(hc < -margin))
            cam_height_median = float(np.median(hc))
            cam_height_std = float(np.std(hc))

            score += 18.0 * cam_above_ratio
            score -= 30.0 * cam_below_ratio
            score += 5.0 * np.tanh(cam_height_median / (0.04 * extent + 1e-12))
            score -= 2.0 * min(cam_height_std / (extent + 1e-12), 1.0)

        if up_prior is not None and prior_weight > 0.0:
            score += float(prior_weight) * float(np.dot(up, normalize(up_prior)))

        if score > best_score:
            best_score = score
            best_up = up
            best_stats = {
                "above_ratio": above_ratio,
                "below_ratio": below_ratio,
                "below_depth_norm": below_depth,
                "cam_above_ratio": cam_above_ratio,
                "cam_below_ratio": cam_below_ratio,
                "cam_height_median": cam_height_median,
                "cam_height_std": cam_height_std,
                "sign_score": score,
            }

    return normalize(best_up), best_score, best_stats


def select_floor_plane(
    P,
    candidates,
    cam_centers,
    params,
    up_prior=None,
    strong_prior=False,
    reject_camera_below_floor=False,
    hand_frames=None,
    head_timestamps=None,
):
    """
    Select the most likely floor among multiple RANSAC planes.
    """
    best = None
    best_score = -1e18
    extent2 = params["scene_extent"] ** 2 + 1e-12
    sign_prior_weight = 12.0 if up_prior is not None else 0.0
    plane_prior_weight = 20.0 if up_prior is not None else 0.0

    for plane in candidates:
        up, sign_score, stats = choose_up_direction(
            P,
            plane,
            cam_centers,
            params,
            up_prior=up_prior,
            prior_weight=sign_prior_weight,
        )
        if (
            reject_camera_below_floor
            and cam_centers is not None
            and len(cam_centers) > 0
            and np.any((cam_centers - plane["center"]) @ up < 0.0)
        ):
            continue

        # Inlier count helps, but log avoids a large wall dominating everything.
        inlier_score = 1.2 * np.log1p(plane["inliers"])

        # Large planar extent is more floor-like than a small dense tabletop patch.
        area_score = 4.0 * np.tanh(plane["area"] / (0.08 * extent2))

        # Penalize noisy planes.
        rms_norm = plane["rms"] / (params["distance_threshold"] + 1e-12)
        rms_penalty = 2.0 * min(rms_norm, 5.0)

        score = sign_score + inlier_score + area_score - rms_penalty
        angle_to_prior = None
        if up_prior is not None:
            angle_to_prior = angle_deg(up, up_prior)
            score += plane_prior_weight * np.cos(np.deg2rad(angle_to_prior))
            if strong_prior and angle_to_prior > GEOCALIB_PRIOR_MAX_ANGLE_DEG:
                score -= 4.0 * (angle_to_prior - GEOCALIB_PRIOR_MAX_ANGLE_DEG)

        hand_stats = hand_above_head_stats(hand_frames, cam_centers, head_timestamps, plane["center"], up)
        hand_height_valid, hand_height_reason = validate_head_above_hands(hand_stats)
        if not hand_height_valid:
            print(
                "[Warning] Rejecting plane candidate: head/hand height check failed: "
                f"{hand_height_reason}",
                flush=True,
            )
            continue

        item = dict(plane)
        item.update({
            "up": up,
            "score": float(score),
            "sign_stats": stats,
            "angle_to_prior_deg": angle_to_prior,
            "hand_above_head": hand_stats,
            "hand_height_valid": bool(hand_height_valid),
            "hand_height_reason": str(hand_height_reason),
        })

        if score > best_score:
            best_score = score
            best = item

    if best is None:
        raise RuntimeError("No floor candidate passed camera/head/hand height checks.")
    return best


def detect_low_prior_plane_candidates(P, params, up_prior):
    """
    Search for floor candidates only in low points along the GeoCalib up axis.

    This is used when image-based gravity and global RANSAC disagree. In that
    case, floor points may be sparse, so large walls or tabletops should not win
    just because they have more inliers. The search first restricts points by
    height h = P dot up_prior, then accepts only planes whose normal is close to
    up_prior.
    """
    up_prior = normalize(up_prior)
    heights = P @ up_prior
    total_points = len(P)
    candidates = []

    for band_i, percentile in enumerate(LOW_PRIOR_PERCENTILES):
        cutoff = float(np.percentile(heights, percentile))
        low_mask = heights <= cutoff
        P_low = P[low_mask]

        min_inliers = max(
            LOW_PRIOR_MIN_INLIERS_ABS,
            int(len(P_low) * LOW_PRIOR_MIN_INLIER_RATIO),
        )
        if len(P_low) < max(min_inliers, RANSAC_N):
            continue

        pcd_work = o3d.geometry.PointCloud()
        pcd_work.points = o3d.utility.Vector3dVector(P_low)

        for local_idx in range(LOW_PRIOR_PLANES_PER_BAND):
            if len(pcd_work.points) < max(min_inliers, RANSAC_N):
                break

            try:
                model, inliers = pcd_work.segment_plane(
                    distance_threshold=params["distance_threshold"],
                    ransac_n=RANSAC_N,
                    num_iterations=RANSAC_ITERATIONS,
                )
            except RuntimeError:
                break

            if len(inliers) < min_inliers:
                break

            inlier_pcd = pcd_work.select_by_index(inliers)
            P_in = np.asarray(inlier_pcd.points, dtype=np.float64)

            n_ransac = normalize(np.asarray(model[:3], dtype=np.float64))
            n_refined, center, rms = fit_normal_by_covariance(
                P_in,
                prefer_dir=up_prior,
            )
            angle_to_prior = angle_deg(n_refined, up_prior)

            if angle_to_prior <= LOW_PRIOR_MAX_NORMAL_ANGLE_DEG:
                area = plane_inlier_area(P_in, center, n_refined)
                center_height = float(center @ up_prior)
                height_rank = float(np.mean(heights < center_height))

                candidates.append({
                    "idx": 1000 + band_i * 10 + local_idx,
                    "normal": n_refined,
                    "center": center,
                    "inliers": int(len(inliers)),
                    "ratio": float(len(inliers) / total_points),
                    "rms": rms,
                    "area": area,
                    "low_percentile": float(percentile),
                    "height_rank": height_rank,
                    "angle_to_prior_deg": angle_to_prior,
                    "selection_source": "geocalib_low_prior",
                })

            pcd_work = pcd_work.select_by_index(inliers, invert=True)

    return candidates


def select_low_prior_floor_plane(P, cam_centers, params, up_prior, hand_frames=None, head_timestamps=None):
    """
    Pick the lowest plausible plane whose normal agrees with GeoCalib up.
    """
    candidates = detect_low_prior_plane_candidates(P, params, up_prior)
    if len(candidates) == 0:
        return None, {
            "num_candidates": 0,
            "method": "geocalib_low_prior",
            "fallback_reason": "no low prior plane candidates",
        }

    up_prior = normalize(up_prior)
    extent2 = params["scene_extent"] ** 2 + 1e-12
    best = None
    best_score = -1e18

    for plane in candidates:
        up, sign_score, stats = choose_up_direction(
            P,
            plane,
            cam_centers,
            params,
            up_prior=up_prior,
            prior_weight=20.0,
        )
        angle_to_prior = angle_deg(up, up_prior)
        height_rank = float(plane.get("height_rank", 1.0))

        # Height dominates here: among planes consistent with gravity, the floor
        # should be the lower envelope of the scene.
        low_score = 32.0 * (1.0 - height_rank)
        prior_score = 18.0 * np.cos(np.deg2rad(angle_to_prior))
        inlier_score = 0.9 * np.log1p(plane["inliers"])
        area_score = 3.0 * np.tanh(plane["area"] / (0.06 * extent2))
        rms_norm = plane["rms"] / (params["distance_threshold"] + 1e-12)
        rms_penalty = 2.0 * min(rms_norm, 5.0)

        score = (
            low_score
            + prior_score
            + sign_score
            + inlier_score
            + area_score
            - rms_penalty
        )

        hand_stats = hand_above_head_stats(hand_frames, cam_centers, head_timestamps, plane["center"], up)
        hand_height_valid, hand_height_reason = validate_head_above_hands(hand_stats)
        if not hand_height_valid:
            continue

        item = dict(plane)
        item.update({
            "up": up,
            "score": float(score),
            "sign_stats": stats,
            "angle_to_prior_deg": angle_to_prior,
            "min_refine_inliers": LOW_PRIOR_MIN_INLIERS_ABS,
            "hand_above_head": hand_stats,
            "hand_height_valid": bool(hand_height_valid),
            "hand_height_reason": str(hand_height_reason),
        })

        if score > best_score:
            best_score = score
            best = item

    if best is None:
        return None, {
            "num_candidates": int(len(candidates)),
            "method": "geocalib_low_prior",
            "fallback_reason": "no low prior plane candidates passed head/hand height checks",
        }

    stats = {
        "num_candidates": int(len(candidates)),
        "method": "geocalib_low_prior",
        "selected_low_percentile": float(best.get("low_percentile", np.nan)),
        "selected_height_rank": float(best.get("height_rank", np.nan)),
        "selected_angle_to_prior_deg": float(best.get("angle_to_prior_deg", np.nan)),
        "selected_inliers": int(best.get("inliers", 0)),
        "selected_area": float(best.get("area", np.nan)),
    }
    return best, stats


# ============================================================
# Floor refinement and validation
# ============================================================

def refine_floor_coarse_to_fine(P, selected, params, min_inliers=MIN_INLIERS_ABS):
    """
    Coarse-to-fine refinement of the selected floor plane.

    The threshold shrinks each iteration. This reduces tilt caused by wall feet,
    table legs, body points, and other non-floor points near the initial plane.
    """
    up = normalize(selected["up"])
    center = selected["center"]
    floor_mask = None

    for threshold in params["refine_thresholds"][:REFINE_ITERS]:
        dist = (P - center) @ up
        mask = np.abs(dist) < threshold
        P_floor = P[mask]

        if len(P_floor) < min_inliers:
            continue

        up_new, center_new, _ = fit_normal_by_covariance(P_floor, prefer_dir=up)
        up = up_new
        center = center_new
        floor_mask = mask

    if floor_mask is None:
        dist = (P - center) @ up
        floor_mask = np.abs(dist) < params["distance_threshold"]

    return normalize(up), center, floor_mask


def final_sign_check(P, up, floor_center, cam_centers, params, up_prior=None):
    """
    Final up/down check after refinement.
    """
    up = normalize(up)
    if up_prior is not None:
        up_prior = normalize(up_prior)
        if np.dot(up, up_prior) < 0:
            up = -up
        return normalize(up)

    margin = params["height_margin"]

    if cam_centers is not None and len(cam_centers) > 0:
        cam_h = (cam_centers - floor_center) @ up
        if float(np.median(cam_h)) < 0:
            up = -up
        return normalize(up)

    h = (P - floor_center) @ up
    above_ratio = float(np.mean(h > margin))
    below_ratio = float(np.mean(h < -margin))
    if below_ratio > above_ratio:
        up = -up
    return normalize(up)


def validate_alignment(P, up, floor_center, R_align, params):
    """
    Validate floor flatness after alignment.
    This does not change the result; it only prints diagnostics.
    """
    threshold = max(params["refine_thresholds"][-1], params["distance_threshold"] * 0.75)
    dist = (P - floor_center) @ up
    floor_mask = np.abs(dist) < threshold
    P_floor = P[floor_mask]

    if len(P_floor) < MIN_INLIERS_ABS:
        return {
            "floor_points": int(len(P_floor)),
            "floor_y_std": None,
            "floor_y_range_p95_p5": None,
        }

    P_aligned_floor = (R_align @ P_floor.T).T
    y = P_aligned_floor[:, 1]

    return {
        "floor_points": int(len(P_floor)),
        "floor_y_std": float(np.std(y)),
        "floor_y_range_p95_p5": robust_percentile_range(y, 5.0, 95.0),
    }




def camera_height_stats(cam_centers, floor_center, up, params):
    """
    Compute scale-adaptive camera/head height diagnostics above the selected floor.

    Heights are measured along the candidate up direction before metric scaling.
    Therefore validation should use ratios against params["height_margin"], not
    fixed meter thresholds.
    """
    if cam_centers is None or len(cam_centers) == 0:
        return {
            "available": False,
            "reason": "no_camera_centers",
        }

    up = normalize(up)
    h = (np.asarray(cam_centers, dtype=np.float64) - np.asarray(floor_center, dtype=np.float64).reshape(1, 3)) @ up
    h = h[np.isfinite(h)]
    if len(h) == 0:
        return {
            "available": False,
            "reason": "no_finite_camera_heights",
        }

    margin = float(params.get("height_margin", 1e-5))
    return {
        "available": True,
        "count": int(len(h)),
        "height_margin": margin,
        "min": float(np.min(h)),
        "p05": float(np.percentile(h, 5.0)),
        "p10": float(np.percentile(h, 10.0)),
        "median": float(np.median(h)),
        "p90": float(np.percentile(h, 90.0)),
        "max": float(np.max(h)),
        "above_margin_ratio": float(np.mean(h > margin)),
        "above_zero_ratio": float(np.mean(h > 0.0)),
        "below_minus_margin_ratio": float(np.mean(h < -margin)),
    }


def validate_camera_height_stats(
    stats,
    *,
    min_valid_camera_height_margin_factor=2.0,
    min_camera_above_floor_ratio=0.75,
    min_camera_p10_height_margin_factor=-2.0,
):
    """
    Validate a floor/up proposal without assuming true metric scale.

    A valid proposal should put most camera centers above the selected floor.
    Since this stage is pre-metric-scale, thresholds are multiples of the
    scale-adaptive height_margin estimated from point spacing and scene extent.
    """
    if not stats.get("available", False):
        return False, stats.get("reason", "camera_height_stats_unavailable")

    margin = float(stats["height_margin"])
    min_median = float(min_valid_camera_height_margin_factor) * margin
    min_p10 = float(min_camera_p10_height_margin_factor) * margin
    min_ratio = float(min_camera_above_floor_ratio)

    median = float(stats["median"])
    p10 = float(stats["p10"])
    above_ratio = float(stats["above_margin_ratio"])

    if not np.isfinite(median) or not np.isfinite(p10) or not np.isfinite(above_ratio):
        return False, "non_finite_camera_height_stat"
    if median <= min_median:
        return False, f"median_height_too_low: {median:.6g} <= {min_median:.6g}"
    if above_ratio < min_ratio:
        return False, f"above_margin_ratio_too_low: {above_ratio:.6g} < {min_ratio:.6g}"
    if p10 <= min_p10:
        return False, f"p10_height_too_low: {p10:.6g} <= {min_p10:.6g}"
    return True, "valid"


def floor_position_stats(P, floor_center, up, params):
    """
    Estimate whether a candidate floor sits near the lower envelope.

    height_rank is the fraction of points lower than the candidate plane along
    the chosen up direction. A high value means the plane is probably elevated
    above a substantial part of the scene, e.g. a tabletop or platform.
    """
    up = normalize(up)
    heights = np.asarray(P, dtype=np.float64) @ up
    heights = heights[np.isfinite(heights)]
    if len(heights) == 0:
        return {
            "available": False,
            "reason": "no_finite_point_heights",
        }

    center_height = float(np.asarray(floor_center, dtype=np.float64).reshape(3) @ up)
    margin = float(params.get("height_margin", 1e-5))
    signed = heights - center_height
    height_rank = float(np.mean(heights < center_height))
    below_margin_ratio = float(np.mean(signed < -margin))
    low_position_valid = (
        height_rank <= LOW_PRIOR_ACCEPT_MAX_HEIGHT_RANK
        and below_margin_ratio <= MAX_BELOW_FLOOR_RATIO
    )
    tabletop_risk = below_margin_ratio > MAX_BELOW_FLOOR_RATIO

    return {
        "available": True,
        "center_height": center_height,
        "height_rank": height_rank,
        "below_margin_ratio": below_margin_ratio,
        "low_position_valid": bool(low_position_valid),
        "tabletop_risk": bool(tabletop_risk),
        "max_valid_height_rank": LOW_PRIOR_ACCEPT_MAX_HEIGHT_RANK,
        "tabletop_risk_height_rank": TABLETOP_RISK_HEIGHT_RANK,
        "max_below_margin_ratio": MAX_BELOW_FLOOR_RATIO,
    }


def should_replace_with_low_prior_floor(
    ransac_floor_stats,
    low_floor_stats,
    low_height_valid,
):
    if not low_height_valid:
        return False, "low_prior_camera_height_invalid"
    if not low_floor_stats.get("available", False):
        return False, low_floor_stats.get("reason", "low_prior_floor_stats_unavailable")
    if not low_floor_stats.get("low_position_valid", False):
        return False, "low_prior_floor_not_low_enough"

    if not ransac_floor_stats.get("available", False):
        return True, "ransac_floor_stats_unavailable"

    ransac_rank = float(ransac_floor_stats.get("height_rank", 1.0))
    low_rank = float(low_floor_stats.get("height_rank", 1.0))
    if bool(ransac_floor_stats.get("tabletop_risk", False)):
        return True, "ransac_tabletop_risk"
    if low_rank + LOW_PRIOR_REPLACE_HEIGHT_RANK_DELTA < ransac_rank:
        return True, "low_prior_floor_is_lower"

    return False, "ransac_floor_already_low_enough"


# ============================================================
# Real-IMU trusted alignment
# ============================================================

def estimate_up_from_imu(imu_path, Ts_c2w, vector_type="up", outlier_angle_deg=180.0):
    """
    Convert the real-IMU vertical vector from camera frame to a DA3-world axis.

    04_prepare_real_sensor_inputs.py writes a single row "frame_id gx gy gz" with
    the median accelerometer reading rotated into the camera frame. For a mostly
    stationary head this is the upward specific-force direction, so the default
    vector type here is "up" rather than the GeoCalib "gravity" convention.
    """
    up_imu, stats = estimate_up_from_geocalib(
        imu_path,
        Ts_c2w,
        vector_type=vector_type,
        outlier_angle_deg=outlier_angle_deg,
    )
    stats = dict(stats)
    stats["imu_path"] = stats.pop("path", str(imu_path))
    stats["source"] = "real_imu"
    return up_imu, stats


def fit_plane_with_imu_constraint(P, up, max_angle_deg):
    """Minimize orthogonal plane error inside the IMU normal cone, not after it."""
    if not np.isfinite(max_angle_deg) or not 0 <= max_angle_deg < 90:
        raise ValueError("IMU plane angle limit must be finite and in [0, 90)")
    up = normalize(up)
    free_normal, center, _ = fit_normal_by_covariance(P, prefer_dir=up)
    if angle_deg(free_normal, up) <= max_angle_deg:
        return free_normal, center
    if max_angle_deg == 0:
        return up, center
    u, v = make_plane_basis(up)
    basis = np.stack([u, v, up], axis=1)
    local = (P - center) @ basis
    covariance = local.T @ local
    covariance /= max(float(np.trace(covariance)), 1e-12)
    radius = np.tan(np.deg2rad(max_angle_deg)) * (1.0 - 1e-7)

    def objective(slope):
        q = np.r_[-slope, 1.0]
        cq = covariance @ q
        norm2 = 1.0 + slope @ slope
        value = q @ cq
        gradient = -2.0 * cq[:2] / norm2 - 2.0 * slope * value / norm2 ** 2
        return value / norm2, gradient

    result = minimize(objective, np.zeros(2), jac=True, method="SLSQP",
        constraints={"type": "ineq", "fun": lambda s: radius ** 2 - s @ s,
                     "jac": lambda s: -2.0 * s},
        options={"ftol": 1e-12, "maxiter": 100})
    normal = normalize(basis @ np.r_[-result.x, 1.0])
    if not result.success or angle_deg(normal, up) > max_angle_deg:
        raise ValueError("IMU-constrained plane solve did not converge within the angle limit")
    return normal, center


def refine_imu_floor_patch(P, up, params, max_angle_deg, min_inliers):
    """A bounded normal is useful only when it still explains a broad patch."""
    mask = np.ones(len(P), dtype=bool)
    minimum = max(min_inliers, int(np.ceil(len(P) * IMU_FLOOR_MIN_RETAINED_RATIO)))
    for threshold in params["refine_thresholds"][:REFINE_ITERS]:
        normal, center = fit_plane_with_imu_constraint(P[mask], up, max_angle_deg)
        mask = np.abs((P - center) @ normal) < min(threshold, params["distance_threshold"])
        if int(mask.sum()) < minimum:
            raise ValueError("constrained_plane_loses_support")
    support = P[mask]
    rms = float(np.sqrt(np.mean(((support - center) @ normal) ** 2)))
    u, v = make_plane_basis(normal)
    widths = [robust_percentile_range(support @ axis) for axis in (u, v)]
    area = float(np.prod(widths))
    voxel = params["voxel_size"]
    if min(widths) < 3.0 * voxel or area < IMU_FLOOR_MIN_AREA_VOXELS * voxel ** 2:
        raise ValueError("connected_patch_has_insufficient_coverage")
    if rms > 0.65 * params["distance_threshold"]:
        raise ValueError("constrained_plane_residual_too_large")
    return normal, center, mask, rms, area


def find_low_floor_candidates_along_imu_axis(
    P, up, params, cam_centers=None, *, max_refine_angle_deg=IMU_MAX_REFINE_ANGLE_DEG,
):
    """
    Collect floor candidates in the low region of the cloud along the IMU axis.

    Scan bands from bottom to top. Within each band, peel off each fitted plane
    before looking for the next one, so a rejected plane cannot monopolize the
    search. Connected patches are fitted inside the IMU normal cone; insufficient
    coverage, residual or retained support excludes a patch from the candidates.
    Both fitting and refinement stay inside the candidate's support.
    Search stops before a band reaches the minimum camera height.

    Returns:
        candidates (percentile, then supported plane height), search_stats
    """
    up = normalize(up)
    P = np.asarray(P, dtype=np.float64)
    heights = P @ up
    cam_height_limit = None
    if cam_centers is not None and len(cam_centers) > 0:
        cam_heights = np.asarray(cam_centers, dtype=np.float64) @ up
        cam_heights = cam_heights[np.isfinite(cam_heights)]
        if len(cam_heights) > 0:
            cam_height_limit = float(np.min(cam_heights))
    total_points = len(P)
    extent2 = params["scene_extent"] ** 2 + 1e-12
    connection_radius = IMU_FLOOR_CONNECTIVITY_VOXELS * params["voxel_size"]

    found = []
    tried = []
    o3d.utility.random.seed(RANDOM_SEED)

    for percentile in IMU_LOW_PERCENTILES:
        cutoff = float(np.percentile(heights, percentile))
        if cam_height_limit is not None and cutoff >= cam_height_limit:
            tried.append({
                "low_percentile": float(percentile),
                "status": "stopped_at_camera_min_height",
                "cutoff": float(cutoff),
                "min_camera_height": float(cam_height_limit),
            })
            break
        remaining = np.flatnonzero(heights <= cutoff)
        min_inliers = max(IMU_FLOOR_MIN_INLIERS_ABS,
                          int(len(remaining) * IMU_FLOOR_MIN_INLIER_RATIO))
        if len(remaining) < min_inliers:
            tried.append({
                "low_percentile": float(percentile),
                "status": "band_too_small",
                "band_points": int(len(remaining)),
            })
            continue

        for plane_index in range(IMU_PLANES_PER_BAND):
            if len(remaining) < max(min_inliers, RANSAC_N):
                break
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(P[remaining])
            _, local_inliers = pcd.segment_plane(
                distance_threshold=params["distance_threshold"],
                ransac_n=RANSAC_N,
                num_iterations=RANSAC_ITERATIONS,
            )
            if len(local_inliers) < min_inliers:
                tried.append({
                    "low_percentile": float(percentile),
                    "plane_in_band": plane_index + 1,
                    "status": "too_few_inliers",
                    "inliers": len(local_inliers),
                })
                break

            plane_indices = remaining[local_inliers]
            keep = np.ones(len(remaining), dtype=bool)
            keep[local_inliers] = False
            remaining = remaining[keep]
            pcd.points = o3d.utility.Vector3dVector(P[plane_indices])
            labels = np.asarray(pcd.cluster_dbscan(eps=connection_radius, min_points=3))
            component_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
            supported = component_ids[counts >= min_inliers]
            if not len(supported):
                tried.append({"low_percentile": float(percentile), "plane_in_band": plane_index + 1,
                              "status": "no_connected_patch_with_enough_points", "inliers": len(plane_indices)})
            for component_id in supported:
                indices = plane_indices[labels == component_id]
                try:
                    normal, center, mask, rms, area = refine_imu_floor_patch(
                        P[indices], up, params, max_refine_angle_deg, min_inliers)
                except ValueError as exc:
                    tried.append({"low_percentile": float(percentile), "plane_in_band": plane_index + 1,
                                  "component": int(component_id), "status": str(exc), "inliers": len(indices)})
                    continue
                floor_mask = np.zeros(total_points, dtype=bool)
                floor_mask[indices[mask]] = True
                level = float(center @ up)
                height_rank = float(np.mean(heights < level))
                inliers = int(mask.sum())
                score = 32.0 * (1.0 - height_rank) + float(np.log1p(inliers))
                score += 3.0 * float(np.tanh(area / (0.06 * extent2)))
                score -= 2.0 * rms / params["distance_threshold"]
                item = {
                    "low_percentile": float(percentile), "plane_in_band": plane_index + 1,
                    "component": int(component_id), "cutoff": cutoff, "status": "ok",
                    "level": level, "inliers": inliers, "inlier_ratio": float(inliers / max(total_points, 1)),
                    "area": area, "rms": rms, "height_rank": height_rank, "score": score,
                    "normal": normal.tolist(), "method": "imu_low_connected_constrained_plane",
                    "min_inliers": int(min_inliers), "retained_support_ratio": float(mask.mean()),
                    "angle_to_imu_deg": angle_deg(normal, up),
                }
                tried.append(dict(item))
                item.update(mask=floor_mask, center=center)
                found.append(item)

    candidates = sorted(found, key=lambda c: (c["low_percentile"], c["level"], -c["inliers"]))

    search_stats = {
        "num_candidates": int(len(candidates)),
        "num_raw_candidates": int(len(found)),
        "min_inliers": IMU_FLOOR_MIN_INLIERS_ABS,
        "support_count_policy": "low_band_fraction_and_absolute_minimum",
        "connection_radius": connection_radius,
        "min_patch_area": IMU_FLOOR_MIN_AREA_VOXELS * params["voxel_size"] ** 2,
        "max_refine_angle_deg": float(max_refine_angle_deg),
        "max_planes_per_band": IMU_PLANES_PER_BAND,
        "bands": tried,
        "candidate_levels": [float(c["level"]) for c in candidates],
        "min_camera_height": cam_height_limit,
    }
    return candidates, search_stats


def make_imu_display_reference(P, up, params):
    """Choose a noise-resistant low reference, not a validated physical floor.

    Gravity stays exactly on the measured IMU axis. The reference only defines
    an origin for consistent exports when the normal floor search has failed.
    """
    up = normalize(up)
    candidates, search = find_low_floor_candidates_along_imu_axis(
        P, up, params, cam_centers=None, max_refine_angle_deg=0.0,
    )
    if candidates:
        candidate = dict(min(candidates, key=lambda c: c["level"]))
        mask = candidate["mask"].copy()
        method = "imu_fixed_axis_low_supported_plane"
        level = float(np.median(P[mask] @ up))
        denoise_status = "connected_plane_support"
    else:
        # Never let one extreme point define the display origin. This last
        # resort is explicitly a lower-envelope reference, not a fitted floor.
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(P)
        retained = np.arange(len(P))
        if len(P) > 20:
            _, kept = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            if len(kept) >= 3:
                retained = np.asarray(kept, dtype=int)
        denoise_status = "statistical_outlier_filter" if len(retained) < len(P) else "percentile_only"
        heights = P[retained] @ up
        low_cutoff = float(np.percentile(heights, 2.0))
        level = float(np.median(heights[heights <= low_cutoff]))
        mask = np.zeros(len(P), dtype=bool)
        mask[retained[np.abs(heights - level) <= params["distance_threshold"]]] = True
        if not mask.any():
            mask[retained[int(np.argmin(np.abs(heights - level)))]] = True
        method = "imu_denoised_lower_envelope_reference"
        candidate = {"low_percentile": 2.0, "plane_in_band": 0, "score": 0.0,
                     "min_inliers": IMU_FLOOR_MIN_INLIERS_ABS}

    center = np.median(P[mask], axis=0)
    center = center + (level - float(center @ up)) * up
    stats = refined_floor_stats(P, mask, center, up, params)
    candidate.update({
        "normal": up.tolist(), "center": center, "mask": mask,
        "method": method, "level": level, "inliers": stats["inliers"],
        "inlier_ratio": stats["inlier_ratio"], "rms": stats["rms"], "area": stats["area"],
        "height_rank": float(np.mean(P @ up < level)),
        "reference_only": True, "denoise_status": denoise_status,
    })
    return candidate, search


def refined_floor_stats(P, floor_mask, floor_center, up, params):
    """Support statistics for a floor plane after coarse-to-fine refinement."""
    up = normalize(up)
    P = np.asarray(P, dtype=np.float64)
    floor_center = np.asarray(floor_center, dtype=np.float64).reshape(3)

    if floor_mask is None:
        floor_mask = np.abs((P - floor_center) @ up) < float(params["distance_threshold"])

    P_in = P[floor_mask]
    inliers = int(len(P_in))
    if inliers < 3:
        return {
            "inliers": inliers,
            "inlier_ratio": float(inliers / max(len(P), 1)),
            "area": 0.0,
            "rms": float("nan"),
        }

    signed = (P_in - floor_center) @ up
    return {
        "inliers": inliers,
        "inlier_ratio": float(inliers / max(len(P), 1)),
        "area": float(plane_inlier_area(P_in, floor_center, up)),
        "rms": float(np.sqrt(np.mean(signed ** 2))),
    }


def imu_head_height_stats(P, cam_centers, floor_center, up, params):
    """
    Head-height diagnostics for the IMU-trusted floor, in scale-free form.

    This stage runs before metric scaling, so absolute meters are not available.
    The extra fields are ratios: head-height spread relative to the median head
    height, and median head height relative to the vertical span of the scene.
    """
    stats = camera_height_stats(cam_centers, floor_center, up, params)
    if not stats.get("available", False):
        return stats

    up = normalize(up)
    heights = np.asarray(P, dtype=np.float64) @ up
    heights = heights[np.isfinite(heights)]
    span = float(np.percentile(heights, 99.0) - np.percentile(heights, 1.0)) if len(heights) else 0.0

    median = float(stats["median"])
    spread = float(stats["p90"] - stats["p10"])

    stats["head_height_spread_p90_p10"] = spread
    stats["head_height_rel_spread"] = float(spread / abs(median)) if abs(median) > 1e-12 else float("inf")
    stats["scene_vertical_span"] = span
    stats["head_span_ratio"] = float(median / span) if span > 1e-12 else float("inf")
    return stats


def evaluate_imu_floor_candidate(
    P,
    cam_centers,
    params,
    up_imu,
    candidate,
    *,
    max_refine_angle_deg,
    **head_height_thresholds,
):
    """
    Evaluate a connected candidate already refined inside the IMU normal cone.
    Head height is validated on the refined plane, and the refined normal is
    compared against the IMU axis.
    """
    selected = {
        "idx": 2000,
        "normal": np.asarray(candidate["normal"], dtype=np.float64),
        "center": np.asarray(candidate["center"], dtype=np.float64),
        "up": np.asarray(candidate["normal"], dtype=np.float64),
        "inliers": int(candidate["inliers"]),
        "ratio": float(candidate["inlier_ratio"]),
        "rms": float(candidate["rms"]),
        "area": float(candidate["area"]),
        "score": float(candidate["score"]),
        "low_percentile": float(candidate["low_percentile"]),
        "min_refine_inliers": int(candidate["min_inliers"]),
        "selection_source": str(candidate["method"]),
    }

    up_refined = np.asarray(candidate["normal"], dtype=np.float64)
    floor_center = np.asarray(candidate["center"], dtype=np.float64)
    floor_mask = candidate["mask"].copy()
    up_refined = final_sign_check(
        P,
        up_refined,
        floor_center,
        cam_centers,
        params,
        up_prior=up_imu,
    )

    refine_angle_deg = angle_deg(up_refined, up_imu)
    angle_ok = refine_angle_deg <= float(max_refine_angle_deg)
    refine_stats = refined_floor_stats(P, floor_mask, floor_center, up_refined, params)
    refine_stats.update({
        "angle_from_imu_axis_deg": float(refine_angle_deg),
        "max_angle_from_imu_axis_deg": float(max_refine_angle_deg),
        "angle_ok": bool(angle_ok),
    })

    floor_position = floor_position_stats(P, floor_center, up_refined, params)
    head_stats = imu_head_height_stats(P, cam_centers, floor_center, up_refined, params)
    height_valid, height_reason, height_checks = validate_imu_head_height(
        head_stats,
        floor_position,
        **head_height_thresholds,
    )

    return {
        "selected": selected,
        "up": up_refined,
        "floor_center": floor_center,
        "floor_mask": floor_mask,
        "candidate": {k: v for k, v in candidate.items() if k not in ("center", "mask")},
        "refine": refine_stats,
        "refine_angle_deg": float(refine_angle_deg),
        "angle_ok": bool(angle_ok),
        "floor_position": floor_position,
        "head_height": head_stats,
        "head_height_checks": height_checks,
        "height_valid": bool(height_valid),
        "height_reason": str(height_reason),
    }


def validate_imu_head_height(
    head_stats,
    floor_position,
    *,
    min_valid_camera_height_margin_factor=2.0,
    min_camera_above_floor_ratio=0.75,
    min_camera_p10_height_margin_factor=-2.0,
    max_head_height_rel_spread=IMU_MAX_HEAD_HEIGHT_REL_SPREAD,
    max_points_below_floor_ratio=IMU_MAX_POINTS_BELOW_FLOOR_RATIO,
    min_head_span_ratio=IMU_MIN_HEAD_SPAN_RATIO,
    max_head_span_ratio=IMU_MAX_HEAD_SPAN_RATIO,
):
    """
    Decide whether the IMU-trusted floor produces a plausible head trajectory.

    All criteria are scale-free because the point cloud is not metric yet:
        1. the head must stay above the fitted floor (shared height validation)
        2. head height must be stable, i.e. p90-p10 small against the median
        3. almost nothing may sit below the fitted plane, so a tabletop or a
           platform cannot pass as the floor
        4. head height must be a sane fraction of the scene vertical span

    Criterion 3 counts points more than one height_margin below the plane rather
    than the raw height rank, so a scene dominated by floor points is not
    rejected just because the floor slab holds most of the cloud.
    """
    checks = []

    valid, reason = validate_camera_height_stats(
        head_stats,
        min_valid_camera_height_margin_factor=min_valid_camera_height_margin_factor,
        min_camera_above_floor_ratio=min_camera_above_floor_ratio,
        min_camera_p10_height_margin_factor=min_camera_p10_height_margin_factor,
    )
    checks.append({"name": "head_above_floor", "passed": bool(valid), "detail": str(reason)})
    if not valid:
        return False, f"head_above_floor_failed: {reason}", checks

    rel_spread = float(head_stats.get("head_height_rel_spread", float("inf")))
    spread_ok = np.isfinite(rel_spread) and rel_spread <= float(max_head_height_rel_spread)
    checks.append({
        "name": "head_height_stability",
        "passed": bool(spread_ok),
        "value": rel_spread,
        "threshold": float(max_head_height_rel_spread),
    })
    if not spread_ok:
        return (
            False,
            f"head_height_unstable: rel_spread={rel_spread:.6g} > {float(max_head_height_rel_spread):.6g}",
            checks,
        )

    available = bool(floor_position.get("available", False))
    below_ratio = float(floor_position.get("below_margin_ratio", 1.0)) if available else 1.0
    height_rank = float(floor_position.get("height_rank", 1.0)) if available else 1.0
    below_ok = below_ratio <= float(max_points_below_floor_ratio)
    checks.append({
        "name": "floor_is_lower_envelope",
        "passed": bool(below_ok),
        "value": below_ratio,
        "threshold": float(max_points_below_floor_ratio),
        "height_rank": height_rank,
    })
    if not below_ok:
        return (
            False,
            (
                f"imu_floor_not_low_enough: points_below_floor_ratio={below_ratio:.6g} > "
                f"{float(max_points_below_floor_ratio):.6g} (height_rank={height_rank:.6g})"
            ),
            checks,
        )

    span_ratio = float(head_stats.get("head_span_ratio", float("inf")))
    span_ok = (
        np.isfinite(span_ratio)
        and float(min_head_span_ratio) <= span_ratio <= float(max_head_span_ratio)
    )
    checks.append({
        "name": "head_vs_scene_span",
        "passed": bool(span_ok),
        "value": span_ratio,
        "min_threshold": float(min_head_span_ratio),
        "max_threshold": float(max_head_span_ratio),
    })
    if not span_ok:
        return (
            False,
            (
                f"head_span_ratio_out_of_range: {span_ratio:.6g} not in "
                f"[{float(min_head_span_ratio):.6g}, {float(max_head_span_ratio):.6g}]"
            ),
            checks,
        )

    return True, "valid", checks


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", required=True, help="Input DA3 combined_pcd.ply")
    parser.add_argument("--poses", required=True, help="Input DA3 camera_poses.txt, fixed as c2w")
    parser.add_argument(
        "--hands-jsonl",
        default=None,
        help="Optional HaWoR hands JSONL in the same world frame as --poses, for the head/hand height check.",
    )
    parser.add_argument(
        "--timestamps",
        default=None,
        help="Optional timestamps.txt aligned with --poses, used to match hand frames to head frames.",
    )
    parser.add_argument("--out", default="R_align.npy", help="Output rotation matrix .npy")
    parser.add_argument(
        "--out_meta",
        default="alignment_transform.npz",
        help="Output metadata .npz containing floor plane and GeoCalib diagnostics",
    )
    parser.add_argument(
        "--geocalib_gravity",
        default=None,
        help="Optional GeoCalib gravity/up vectors in camera frame (.npy/.npz/.txt/.csv/.json)",
    )
    parser.add_argument(
        "--geocalib_vector_type",
        default="gravity",
        choices=["gravity", "down", "g", "up", "upright"],
        help=(
            "Whether GeoCalib vectors point down/gravity or up. "
            "Default is gravity because run_geocalib.py saves gravity/gravity_cam vectors."
        ),
    )
    parser.add_argument(
        "--geocalib_switch_angle_deg",
        type=float,
        default=GEOCALIB_SWITCH_ANGLE_DEG,
        help="If RANSAC up and GeoCalib up differ more than this, reselect floor with GeoCalib prior",
    )
    parser.add_argument(
        "--geocalib_outlier_angle_deg",
        type=float,
        default=GEOCALIB_OUTLIER_ANGLE_DEG,
        help="Per-frame GeoCalib world-up outlier rejection threshold",
    )
    parser.add_argument(
        "--imu_gravity",
        default=None,
        help=(
            "Real IMU vertical vector in camera frame (tmp/imu_gravity.txt). "
            "When given, the IMU axis is trusted: the floor normal is locked to it "
            "and only the floor offset is fitted from low scene points. GeoCalib "
            "options are ignored in this mode."
        ),
    )
    parser.add_argument(
        "--imu_vector_type",
        default="up",
        choices=["gravity", "down", "g", "up", "upright"],
        help=(
            "How to read imu_gravity.txt. Default is up because the file stores the "
            "median accelerometer specific force, which points upward at rest. The "
            "device convention gives the initial sign; matched hand data may flip it "
            "when the camera is not above the hands."
        ),
    )
    parser.add_argument(
        "--imu_min_camera_above_hand_ratio",
        type=float,
        default=IMU_MIN_CAMERA_ABOVE_HAND_RATIO,
        help=(
            "In real-IMU mode, flip the IMU up sign when matched camera/head height "
            "is above hand keypoints in less than this ratio."
        ),
    )
    parser.add_argument(
        "--imu_outlier_angle_deg",
        type=float,
        default=180.0,
        help="Per-sample IMU outlier rejection threshold. The file normally holds one vector.",
    )
    parser.add_argument(
        "--imu_max_refine_angle_deg",
        type=float,
        default=IMU_MAX_REFINE_ANGLE_DEG,
        help=(
            "Max angle between the IMU axis and the floor normal re-estimated from the "
            "point cloud. Beyond this the IMU result is marked rejected so the main "
            "pipeline can skip EgoGVAE inference."
        ),
    )
    parser.add_argument(
        "--imu_max_head_height_rel_spread",
        type=float,
        default=IMU_MAX_HEAD_HEIGHT_REL_SPREAD,
        help="Max (p90-p10)/median of head height above the IMU floor.",
    )
    parser.add_argument(
        "--imu_max_points_below_floor_ratio",
        type=float,
        default=IMU_MAX_POINTS_BELOW_FLOOR_RATIO,
        help=(
            "Max fraction of scene points more than one height_margin below the IMU "
            "floor plane. Keeps a tabletop or a platform from passing as the floor."
        ),
    )
    parser.add_argument(
        "--imu_min_head_span_ratio",
        type=float,
        default=IMU_MIN_HEAD_SPAN_RATIO,
        help="Min median head height as a fraction of the scene vertical span.",
    )
    parser.add_argument(
        "--imu_max_head_span_ratio",
        type=float,
        default=IMU_MAX_HEAD_SPAN_RATIO,
        help="Max median head height as a fraction of the scene vertical span.",
    )
    parser.add_argument(
        "--geocalib_precheck_height",
        action="store_true",
        help=(
            "Before using GeoCalib as a prior, check GeoCalib up against the "
            "RANSAC floor center. If camera heights are invalid, keep RANSAC."
        ),
    )
    parser.add_argument(
        "--fallback_to_ransac_on_invalid_height",
        action="store_true",
        help=(
            "After GeoCalib-prior floor fitting/refinement, validate camera/head "
            "heights and fall back to point-cloud-only RANSAC if invalid."
        ),
    )
    parser.add_argument(
        "--min_valid_camera_height_margin_factor",
        type=float,
        default=2.0,
        help="Minimum median camera height as a multiple of auto height_margin.",
    )
    parser.add_argument(
        "--min_camera_above_floor_ratio",
        type=float,
        default=0.75,
        help="Minimum ratio of camera centers above height_margin.",
    )
    parser.add_argument(
        "--min_camera_p10_height_margin_factor",
        type=float,
        default=-2.0,
        help="Minimum p10 camera height as a multiple of auto height_margin.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.imu_min_camera_above_hand_ratio <= 1.0:
        raise ValueError("--imu_min_camera_above_hand_ratio must be in [0, 1]")

    P_full = load_point_cloud(args.pcd)
    params = compute_auto_params(P_full)
    P = downsample_for_estimation(P_full, params["voxel_size"])

    Ts = load_da3_poses_c2w(args.poses)
    cam_centers = Ts[:, :3, 3]
    hand_frames = load_hand_frames(args.hands_jsonl)
    head_timestamps = load_timestamps_txt(args.timestamps)

    imu_mode = args.imu_gravity is not None

    # Stage 1: original DA3 point-cloud-only RANSAC selection. It is only used
    # when no real IMU is available; real-IMU mode must not precompute or fall
    # back to this global point-cloud direction.
    candidates = None
    selected_ransac = None
    up_ransac = None
    floor_center_ransac = None
    floor_mask_ransac = None
    ransac_error = None
    if not imu_mode:
        try:
            candidates = detect_plane_candidates(P, params, cam_centers)
            selected_ransac = select_floor_plane(
                P,
                candidates,
                cam_centers,
                params,
                reject_camera_below_floor=True,
                hand_frames=hand_frames,
                head_timestamps=head_timestamps,
            )
            up_ransac, floor_center_ransac, floor_mask_ransac = refine_floor_coarse_to_fine(
                P,
                selected_ransac,
                params,
            )
            up_ransac = final_sign_check(P, up_ransac, floor_center_ransac, cam_centers, params)
        except RuntimeError:
            raise
    else:
        ransac_error = "skipped_in_real_imu_mode"

    up_geocalib = None
    up_geocalib_raw = None
    geocalib_stats = {}
    geocalib_angle_deg = None
    geocalib_raw_angle_deg = None
    geocalib_flipped_180 = False
    geocalib_flip_precheck_height_stats = {}
    geocalib_flip_precheck_valid = None
    geocalib_flip_precheck_reason = None
    geocalib_direction_consistent = None
    used_geocalib_prior = False
    low_prior_stats = {}
    ransac_floor_position_stats = {}
    final_floor_position_stats = {}

    selected = selected_ransac
    up = up_ransac
    floor_center = floor_center_ransac
    floor_mask = floor_mask_ransac

    if up_ransac is not None:
        ransac_camera_height_stats = camera_height_stats(
            cam_centers,
            floor_center_ransac,
            up_ransac,
            params,
        )
        ransac_height_valid, ransac_height_reason = validate_camera_height_stats(
            ransac_camera_height_stats,
            min_valid_camera_height_margin_factor=args.min_valid_camera_height_margin_factor,
            min_camera_above_floor_ratio=args.min_camera_above_floor_ratio,
            min_camera_p10_height_margin_factor=args.min_camera_p10_height_margin_factor,
        )
        ransac_floor_position_stats = floor_position_stats(
            P,
            floor_center_ransac,
            up_ransac,
            params,
        )
    else:
        ransac_camera_height_stats = {
            "available": False,
            "reason": f"ransac_baseline_unavailable: {ransac_error}",
        }
        ransac_height_valid = False
        ransac_height_reason = f"ransac_baseline_unavailable: {ransac_error}"
        ransac_floor_position_stats = {
            "available": False,
            "reason": f"ransac_baseline_unavailable: {ransac_error}",
        }
    final_floor_position_stats = ransac_floor_position_stats

    geocalib_precheck_height_stats = {}
    geocalib_precheck_valid = None
    geocalib_precheck_reason = None
    final_camera_height_stats = ransac_camera_height_stats
    final_height_valid = ransac_height_valid
    final_height_reason = ransac_height_reason
    fallback_to_ransac = False
    fallback_reason = None
    decision_reason = "no_geocalib_prior_use_ransac"

    up_imu = None
    up_imu_raw = None
    imu_stats = {}
    used_imu_gravity = False

    # Stage 2A: real IMU. The measured vertical is trusted as-is. The point cloud
    # only decides where the floor sits along that axis, and a head-height check
    # decides whether the result is kept. GeoCalib is not involved in this mode.
    if imu_mode:
        decision_reason = "imu_mode_no_decision_recorded"
        print("[IMU] Real IMU gravity provided: running IMU-trusted alignment.", flush=True)
        print(
            f"[IMU] vector file: {args.imu_gravity} (vector_type={args.imu_vector_type})",
            flush=True,
        )
        try:
            up_imu_raw, imu_vector_stats = estimate_up_from_imu(
                args.imu_gravity,
                Ts,
                vector_type=args.imu_vector_type,
                outlier_angle_deg=args.imu_outlier_angle_deg,
            )
        except (EmptyGeoCalibVectors, FileNotFoundError, ValueError, RuntimeError) as exc:
            imu_stats = {
                "status": "unusable",
                "reason": str(exc),
                "imu_path": str(args.imu_gravity),
                "vector_type": str(args.imu_vector_type),
            }
            print(f"[IMU] IMU vertical unusable: {exc}", flush=True)
            raise RuntimeError(
                f"IMU vertical unusable ({exc}); real-IMU mode does not use "
                "point-cloud-only RANSAC fallback."
            ) from exc
        else:
            up_imu, imu_sign_stats = resolve_imu_up_sign_with_hands(
                up_imu_raw,
                hand_frames,
                cam_centers,
                head_timestamps,
                args.imu_min_camera_above_hand_ratio,
            )
            imu_sign_stats["vector_type"] = str(args.imu_vector_type)
            print(
                f"[IMU] measured vertical in DA3 world: "
                f"{np.array2string(np.asarray(up_imu, dtype=np.float64), precision=6)}",
                flush=True,
            )
            print(
                f"[IMU] hand-height sign check: resolved={imu_sign_stats.get('resolved')} "
                f"flipped_from_file={imu_sign_stats.get('flipped_from_input')} "
                f"camera_above_hand_ratio={imu_sign_stats.get('camera_above_hand_ratio')} "
                f"threshold={imu_sign_stats.get('threshold')} "
                f"reason={imu_sign_stats.get('reason')}",
                flush=True,
            )
            # Search along the IMU axis and fit connected patches within its cone.
            imu_floor_candidates, imu_floor_search = find_low_floor_candidates_along_imu_axis(
                P,
                up_imu,
                params,
                cam_centers=cam_centers,
                max_refine_angle_deg=args.imu_max_refine_angle_deg,
            )
            print(
                f"[IMU] low plane search along IMU axis: "
                f"candidates={imu_floor_search['num_candidates']} "
                f"min_inliers={imu_floor_search['min_inliers']}",
                flush=True,
            )
            imu_head_thresholds = {
                "min_valid_camera_height_margin_factor": args.min_valid_camera_height_margin_factor,
                "min_camera_above_floor_ratio": args.min_camera_above_floor_ratio,
                "min_camera_p10_height_margin_factor": args.min_camera_p10_height_margin_factor,
                "max_head_height_rel_spread": args.imu_max_head_height_rel_spread,
                "max_points_below_floor_ratio": args.imu_max_points_below_floor_ratio,
                "min_head_span_ratio": args.imu_min_head_span_ratio,
                "max_head_span_ratio": args.imu_max_head_span_ratio,
            }

            def evaluate_candidate(index):
                candidate = imu_floor_candidates[index]
                evaluation = evaluate_imu_floor_candidate(
                    P,
                    cam_centers,
                    params,
                    up_imu,
                    candidate,
                    max_refine_angle_deg=args.imu_max_refine_angle_deg,
                    **imu_head_thresholds,
                )
                head = evaluation["head_height"]
                print(
                    f"[IMU] candidate {index + 1}/{len(imu_floor_candidates)}: "
                    f"method={candidate['method']} "
                    f"band_percentile={candidate['low_percentile']:.6g} "
                    f"plane_in_band={candidate['plane_in_band']} "
                    f"level={candidate['level']:.6g} "
                    f"inliers={candidate['inliers']} "
                    f"height_rank={candidate['height_rank']:.6g}",
                    flush=True,
                )
                print(
                    "[IMU]   head height above refined floor: "
                    f"{'pass' if evaluation['height_valid'] else 'FAIL'} "
                    f"median={head.get('median', float('nan')):.6g} "
                    f"p10={head.get('p10', float('nan')):.6g} "
                    f"p90={head.get('p90', float('nan')):.6g} "
                    f"above_floor_ratio={head.get('above_margin_ratio', float('nan')):.6g} "
                    f"rel_spread={head.get('head_height_rel_spread', float('nan')):.6g} "
                    f"scene_span_ratio={head.get('head_span_ratio', float('nan')):.6g} "
                    f"reason={evaluation['height_reason']}",
                    flush=True,
                )
                for check in evaluation["head_height_checks"]:
                    verdict = "pass" if check.get("passed") else "FAIL"
                    extra = {k: v for k, v in check.items() if k not in ("name", "passed")}
                    print(f"[IMU]     check {check['name']}: {verdict} {extra}", flush=True)
                print(
                    "[IMU]   refined normal vs IMU axis: "
                    f"{'pass' if evaluation['angle_ok'] else 'FAIL'} "
                    f"angle={evaluation['refine_angle_deg']:.6f} deg "
                    f"(max {float(args.imu_max_refine_angle_deg):.6g}) "
                    f"inliers={evaluation['refine']['inliers']} "
                    f"rms={evaluation['refine']['rms']:.6g}",
                    flush=True,
                )
                return evaluation

            chosen = None
            imu_reject_reason = None
            imu_attempts = []
            for index in range(len(imu_floor_candidates)):
                evaluation = evaluate_candidate(index)
                imu_attempts.append(evaluation)
                if not evaluation["angle_ok"]:
                    print(
                        "[IMU] refined normal outside the IMU angle limit; "
                        "trying the next lower-camera-bounded candidate.",
                        flush=True,
                    )
                    continue
                if not evaluation["height_valid"]:
                    print(
                        "[IMU] refined normal matches the IMU axis; height checks are "
                        "recorded as diagnostics only in real-IMU mode.",
                        flush=True,
                    )
                chosen = evaluation
                break

            if chosen is None:
                imu_reject_reason = (
                    "no_floor_candidate" if not imu_attempts
                    else "no_low_plane_within_refine_angle"
                )
                reference, reference_search = make_imu_display_reference(P, up_imu, params)
                best_effort = evaluate_imu_floor_candidate(
                    P, cam_centers, params, up_imu, reference,
                    max_refine_angle_deg=args.imu_max_refine_angle_deg,
                    **imu_head_thresholds,
                )

            imu_valid = chosen is not None
            evaluation = chosen if chosen is not None else best_effort
            imu_up = evaluation["up"]
            imu_floor_center = evaluation["floor_center"]
            imu_floor_mask = evaluation["floor_mask"]
            imu_selected = evaluation["selected"]
            imu_floor_stats = evaluation["candidate"]
            imu_floor_refine_stats = evaluation["refine"]
            imu_floor_position_stats = evaluation["floor_position"]
            imu_height_stats = evaluation["head_height"]
            imu_height_checks = evaluation["head_height_checks"]
            imu_height_valid = evaluation["height_valid"]
            imu_height_reason = evaluation["height_reason"]
            imu_refine_angle_deg = evaluation["refine_angle_deg"]
            imu_refine_angle_ok = evaluation["angle_ok"]

            imu_stats = {
                "status": "ok",
                "vector": imu_vector_stats,
                "sign_check": imu_sign_stats,
                "floor_search": imu_floor_search,
                "floor_fit": imu_floor_stats,
                "floor_refine": imu_floor_refine_stats,
                "floor_position": imu_floor_position_stats,
                "head_height": imu_height_stats,
                "head_height_checks": imu_height_checks,
                "head_height_valid": bool(imu_height_valid),
                "head_height_reason": str(imu_height_reason),
                "refine_angle_ok": bool(imu_refine_angle_ok),
                "accepted": bool(imu_valid),
                "failed_checks_reason": str(imu_reject_reason) if imu_reject_reason else None,
                "reject_reason": None if imu_valid else str(imu_reject_reason),
                "num_candidates_tried": int(len(imu_attempts)),
                "attempts": [
                    {
                        "candidate": attempt["candidate"],
                        "refine_angle_deg": attempt["refine_angle_deg"],
                        "angle_ok": attempt["angle_ok"],
                        "height_valid": attempt["height_valid"],
                        "height_reason": attempt["height_reason"],
                    }
                    for attempt in imu_attempts
                ],
                "thresholds": {
                    "min_valid_camera_height_margin_factor": float(args.min_valid_camera_height_margin_factor),
                    "min_camera_above_floor_ratio": float(args.min_camera_above_floor_ratio),
                    "min_camera_p10_height_margin_factor": float(args.min_camera_p10_height_margin_factor),
                    "max_head_height_rel_spread": float(args.imu_max_head_height_rel_spread),
                    "max_points_below_floor_ratio": float(args.imu_max_points_below_floor_ratio),
                    "min_head_span_ratio": float(args.imu_min_head_span_ratio),
                    "max_head_span_ratio": float(args.imu_max_head_span_ratio),
                    "max_refine_angle_deg": float(args.imu_max_refine_angle_deg),
                },
            }

            if imu_valid:
                used_imu_gravity = True
                up = normalize(imu_up)
                floor_center = imu_floor_center
                floor_mask = imu_floor_mask
                selected = dict(imu_selected)
                selected.update({
                    "normal": np.asarray(imu_up, dtype=np.float64),
                    "center": np.asarray(imu_floor_center, dtype=np.float64),
                    "up": np.asarray(imu_up, dtype=np.float64),
                    "inliers": int(imu_floor_refine_stats["inliers"]),
                    "ratio": float(imu_floor_refine_stats["inlier_ratio"]),
                    "rms": float(imu_floor_refine_stats["rms"]),
                    "area": float(imu_floor_refine_stats["area"]),
                    "angle_to_prior_deg": float(imu_refine_angle_deg),
                    "height_rank": float(imu_floor_position_stats.get("height_rank", np.nan)),
                    "selection_source": f"{imu_floor_stats['method']}_refined",
                })
                final_camera_height_stats = imu_height_stats
                final_floor_position_stats = imu_floor_position_stats

                final_height_valid = bool(imu_height_valid)
                final_height_reason = str(imu_height_reason)
                decision_reason = (
                    "imu_low_plane_refined_accepted: "
                    f"method={imu_floor_stats['method']}; "
                    f"candidate={len(imu_attempts)}/{imu_floor_search['num_candidates']}; "
                    f"refine_angle_from_imu={imu_refine_angle_deg:.6f} deg"
                )
                print(
                    f"[IMU] accepted on candidate {len(imu_attempts)}: floor found below "
                    "the minimum camera height along the IMU axis, final up re-estimated "
                    "from the point cloud.",
                    flush=True,
                )
            else:
                print(f"[IMU] rejected: {imu_reject_reason}", flush=True)
                used_imu_gravity = True
                imu_stats["status"] = "fallback_alignment_only"
                imu_stats["fallback_reference_search"] = reference_search
                imu_stats["metric_scale_trusted"] = False
                imu_stats["reference_only"] = True
                up = normalize(imu_up)
                floor_center = imu_floor_center
                floor_mask = imu_floor_mask
                selected = dict(imu_selected)
                selected.update({
                    "normal": np.asarray(imu_up, dtype=np.float64),
                    "center": np.asarray(imu_floor_center, dtype=np.float64),
                    "up": np.asarray(imu_up, dtype=np.float64),
                    "inliers": int(imu_floor_refine_stats["inliers"]),
                    "ratio": float(imu_floor_refine_stats["inlier_ratio"]),
                    "rms": float(imu_floor_refine_stats["rms"]),
                    "area": float(imu_floor_refine_stats["area"]),
                    "angle_to_prior_deg": float(imu_refine_angle_deg),
                    "height_rank": float(imu_floor_position_stats.get("height_rank", np.nan)),
                    "selection_source": f"{imu_floor_stats['method']}_failed_diagnostic",
                })
                final_camera_height_stats = imu_height_stats
                final_floor_position_stats = imu_floor_position_stats
                final_height_valid = False
                final_height_reason = f"unvalidated_display_reference: {imu_reject_reason}"
                decision_reason = f"imu_fallback_alignment_only: {imu_reject_reason}"
                print(
                    "[IMU] Floor estimation failed; exporting fixed-IMU alignment with "
                    f"reference={reference['method']} level={reference['level']:.6g}. "
                    "accepted=false; skip metric scale and body inference.",
                    flush=True,
                )

    # Stage 2B: optional GeoCalib appearance-based gravity prior.
    elif args.geocalib_gravity is not None:
        try:
            up_geocalib, geocalib_stats = estimate_up_from_geocalib(
                args.geocalib_gravity,
                Ts,
                vector_type=args.geocalib_vector_type,
                outlier_angle_deg=args.geocalib_outlier_angle_deg,
            )
            up_geocalib_raw = np.asarray(up_geocalib, dtype=np.float64).copy()
        except EmptyGeoCalibVectors as exc:
            print(f"[Warning] GeoCalib prior skipped: {exc}")
            geocalib_stats = {
                "skipped": True,
                "skip_reason": str(exc),
                "path": str(Path(args.geocalib_gravity).expanduser().resolve()),
            }
            up_geocalib = None
            decision_reason = "empty_geocalib_vectors_use_ransac"
        else:
            geocalib_angle_deg = angle_deg(up_ransac, up_geocalib)
            geocalib_raw_angle_deg = geocalib_angle_deg
            geocalib_direction_consistent = geocalib_angle_deg <= args.geocalib_switch_angle_deg
            print(f"RANSAC vs GeoCalib angle: {geocalib_angle_deg:.6f} deg")

            if geocalib_angle_deg > args.geocalib_switch_angle_deg:
                # Optional pre-check: GeoCalib up should put the camera trajectory above
                # the RANSAC floor even before using it to search a new floor.
                precheck_ok = True
                if args.geocalib_precheck_height:
                    geocalib_precheck_height_stats = camera_height_stats(
                        cam_centers,
                        floor_center_ransac,
                        up_geocalib,
                        params,
                    )
                    geocalib_precheck_valid, geocalib_precheck_reason = validate_camera_height_stats(
                        geocalib_precheck_height_stats,
                        min_valid_camera_height_margin_factor=args.min_valid_camera_height_margin_factor,
                        min_camera_above_floor_ratio=args.min_camera_above_floor_ratio,
                        min_camera_p10_height_margin_factor=args.min_camera_p10_height_margin_factor,
                    )
                    precheck_ok = bool(geocalib_precheck_valid)
                    if not precheck_ok:
                        flipped_up_geocalib = -normalize(up_geocalib)
                        geocalib_flip_precheck_height_stats = camera_height_stats(
                            cam_centers,
                            floor_center_ransac,
                            flipped_up_geocalib,
                            params,
                        )
                        geocalib_flip_precheck_valid, geocalib_flip_precheck_reason = validate_camera_height_stats(
                            geocalib_flip_precheck_height_stats,
                            min_valid_camera_height_margin_factor=args.min_valid_camera_height_margin_factor,
                            min_camera_above_floor_ratio=args.min_camera_above_floor_ratio,
                            min_camera_p10_height_margin_factor=args.min_camera_p10_height_margin_factor,
                        )
                        if geocalib_flip_precheck_valid:
                            up_geocalib = flipped_up_geocalib
                            geocalib_flipped_180 = True
                            geocalib_angle_deg = angle_deg(up_ransac, up_geocalib)
                            geocalib_precheck_height_stats = geocalib_flip_precheck_height_stats
                            geocalib_precheck_valid = True
                            geocalib_precheck_reason = (
                                "flipped_180_valid_after_original_failed: "
                                f"original={geocalib_precheck_reason}"
                            )
                            precheck_ok = True
                            print(
                                "[Warning] GeoCalib precheck failed in original direction; "
                                f"using 180-degree flipped GeoCalib direction. "
                                f"new_angle={geocalib_angle_deg:.6f} deg"
                            )
                        else:
                            decision_reason = (
                                "geocalib_precheck_failed_use_ransac: "
                                f"original={geocalib_precheck_reason}; "
                                f"flipped_180={geocalib_flip_precheck_reason}"
                            )
                            print(f"[Warning] {decision_reason}")

                if precheck_ok:
                    if geocalib_angle_deg <= args.geocalib_switch_angle_deg:
                        # If image gravity and point-cloud geometry agree, keep the geometry result.
                        decision_reason = (
                            "geocalib_agrees_with_ransac_use_ransac: "
                            f"angle={geocalib_angle_deg:.6f} <= {args.geocalib_switch_angle_deg:.6f}"
                        )
                    else:
                        # GeoCalib is only used as a prior to re-search/refine a floor plane;
                        # the selected plane and final up are still estimated from point cloud
                        # density, inliers, area, lower-envelope score, and local refinement.
                        selected_low, low_prior_stats = select_low_prior_floor_plane(
                            P,
                            cam_centers,
                            params,
                            up_geocalib,
                            hand_frames=hand_frames,
                            head_timestamps=head_timestamps,
                        )

                        if selected_low is not None:
                            selected = selected_low
                        else:
                            selected = select_floor_plane(
                                P,
                                candidates,
                                cam_centers,
                                params,
                                up_prior=up_geocalib,
                                strong_prior=True,
                                hand_frames=hand_frames,
                                head_timestamps=head_timestamps,
                            )
                            low_prior_stats["fallback_selected_source"] = "global_prior_rescore"

                        up, floor_center, floor_mask = refine_floor_coarse_to_fine(
                            P,
                            selected,
                            params,
                            min_inliers=int(selected.get("min_refine_inliers", MIN_INLIERS_ABS)),
                        )
                        up = final_sign_check(
                            P,
                            up,
                            floor_center,
                            cam_centers,
                            params,
                            up_prior=up_geocalib,
                        )

                        final_camera_height_stats = camera_height_stats(
                            cam_centers,
                            floor_center,
                            up,
                            params,
                        )
                        final_height_valid, final_height_reason = validate_camera_height_stats(
                            final_camera_height_stats,
                            min_valid_camera_height_margin_factor=args.min_valid_camera_height_margin_factor,
                            min_camera_above_floor_ratio=args.min_camera_above_floor_ratio,
                            min_camera_p10_height_margin_factor=args.min_camera_p10_height_margin_factor,
                        )
                        final_floor_position_stats = floor_position_stats(
                            P,
                            floor_center,
                            up,
                            params,
                        )
                        floor_position_valid = bool(
                            final_floor_position_stats.get("low_position_valid", False)
                        ) and not bool(final_floor_position_stats.get("tabletop_risk", False))
                        if floor_position_valid:
                            floor_position_reason = "floor_position_valid"
                        else:
                            floor_position_reason = (
                                "geocalib_prior_floor_position_invalid: "
                                f"height_rank={final_floor_position_stats.get('height_rank', float('nan')):.6g} "
                                f"below_margin_ratio={final_floor_position_stats.get('below_margin_ratio', float('nan')):.6g} "
                                f"max_below_margin_ratio={final_floor_position_stats.get('max_below_margin_ratio', MAX_BELOW_FLOOR_RATIO):.6g} "
                                f"low_position_valid={final_floor_position_stats.get('low_position_valid', False)} "
                                f"tabletop_risk={final_floor_position_stats.get('tabletop_risk', False)}"
                            )
                        fallback_reason_candidate = None
                        if not final_height_valid:
                            fallback_reason_candidate = (
                                f"geocalib_prior_height_validation_failed: {final_height_reason}"
                            )
                        elif not floor_position_valid:
                            fallback_reason_candidate = floor_position_reason

                        if args.fallback_to_ransac_on_invalid_height and fallback_reason_candidate is not None:
                            fallback_to_ransac = True
                            fallback_reason = fallback_reason_candidate
                            print(f"[Warning] {fallback_reason}")
                            print("[Warning] Falling back to point-cloud-only RANSAC alignment.")

                            selected = selected_ransac
                            up = up_ransac
                            floor_center = floor_center_ransac
                            floor_mask = floor_mask_ransac
                            used_geocalib_prior = False
                            final_camera_height_stats = ransac_camera_height_stats
                            final_height_valid = ransac_height_valid
                            final_height_reason = ransac_height_reason
                            final_floor_position_stats = ransac_floor_position_stats
                            low_prior_stats["fallback_to_ransac"] = True
                            low_prior_stats["fallback_reason"] = fallback_reason
                            decision_reason = fallback_reason
                        else:
                            used_geocalib_prior = True
                            decision_reason = (
                                "geocalib_disagrees_with_ransac_prior_refined_and_accepted: "
                                f"angle={geocalib_angle_deg:.6f} > {args.geocalib_switch_angle_deg:.6f}; "
                                f"height_valid={final_height_valid}; floor_position_valid={floor_position_valid}; "
                                f"reason={final_height_reason if not final_height_valid else floor_position_reason}"
                            )
            else:
                # Direction agreement is not enough: still check whether the
                # selected RANSAC plane is too high, and search lower-envelope
                # planes along the same up direction.
                selected_low, low_prior_stats = select_low_prior_floor_plane(
                    P,
                    cam_centers,
                    params,
                    up_geocalib,
                    hand_frames=hand_frames,
                    head_timestamps=head_timestamps,
                )
                low_prior_stats["direction_consistent"] = True
                low_prior_stats["ransac_floor_position_stats"] = ransac_floor_position_stats

                if selected_low is not None:
                    low_up, low_floor_center, low_floor_mask = refine_floor_coarse_to_fine(
                        P,
                        selected_low,
                        params,
                        min_inliers=int(selected_low.get("min_refine_inliers", MIN_INLIERS_ABS)),
                    )
                    low_up = final_sign_check(
                        P,
                        low_up,
                        low_floor_center,
                        cam_centers,
                        params,
                        up_prior=up_geocalib,
                    )
                    low_camera_height_stats = camera_height_stats(
                        cam_centers,
                        low_floor_center,
                        low_up,
                        params,
                    )
                    low_height_valid, low_height_reason = validate_camera_height_stats(
                        low_camera_height_stats,
                        min_valid_camera_height_margin_factor=args.min_valid_camera_height_margin_factor,
                        min_camera_above_floor_ratio=args.min_camera_above_floor_ratio,
                        min_camera_p10_height_margin_factor=args.min_camera_p10_height_margin_factor,
                    )
                    low_floor_position_stats = floor_position_stats(
                        P,
                        low_floor_center,
                        low_up,
                        params,
                    )
                    replace_with_low, replace_reason = should_replace_with_low_prior_floor(
                        ransac_floor_position_stats,
                        low_floor_position_stats,
                        low_height_valid,
                    )

                    low_prior_stats.update({
                        "low_camera_height_stats": low_camera_height_stats,
                        "low_height_valid": bool(low_height_valid),
                        "low_height_reason": str(low_height_reason),
                        "low_floor_position_stats": low_floor_position_stats,
                        "replace_with_low_prior": bool(replace_with_low),
                        "replace_reason": str(replace_reason),
                    })

                    if replace_with_low:
                        selected = selected_low
                        up = low_up
                        floor_center = low_floor_center
                        floor_mask = low_floor_mask
                        used_geocalib_prior = True
                        final_camera_height_stats = low_camera_height_stats
                        final_height_valid = low_height_valid
                        final_height_reason = low_height_reason
                        final_floor_position_stats = low_floor_position_stats
                        decision_reason = (
                            "geocalib_agrees_with_ransac_low_floor_selected: "
                            f"angle={geocalib_angle_deg:.6f} <= {args.geocalib_switch_angle_deg:.6f}; "
                            f"reason={replace_reason}; height_valid={low_height_valid}"
                        )
                    else:
                        final_floor_position_stats = ransac_floor_position_stats
                        decision_reason = (
                            "geocalib_agrees_with_ransac_low_floor_checked_keep_ransac: "
                            f"angle={geocalib_angle_deg:.6f} <= {args.geocalib_switch_angle_deg:.6f}; "
                            f"reason={replace_reason}; low_height_reason={low_height_reason}"
                        )
                else:
                    final_floor_position_stats = ransac_floor_position_stats
                    decision_reason = (
                        "geocalib_agrees_with_ransac_no_low_floor_candidate_keep_ransac: "
                        f"angle={geocalib_angle_deg:.6f} <= {args.geocalib_switch_angle_deg:.6f}; "
                        f"ransac_floor_position={ransac_floor_position_stats}"
                    )

    final_hand_height_stats = {}
    final_hand_height_valid = True
    final_hand_height_reason = "not_checked_in_real_imu_mode" if imu_mode else "valid"
    if not imu_mode:
        final_hand_height_stats = hand_above_head_stats(
            hand_frames,
            cam_centers,
            head_timestamps,
            floor_center,
            up,
        )
        final_hand_height_valid, final_hand_height_reason = validate_head_above_hands(
            final_hand_height_stats
        )
        if not final_height_valid:
            raise RuntimeError(f"No-IMU final head height check failed: {final_height_reason}")
        if not final_hand_height_valid:
            raise RuntimeError(f"No-IMU final head/hand height check failed: {final_hand_height_reason}")

    target_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    R_align = rotation_from_vectors(up, target_up)
    np.save(args.out, R_align.astype(np.float64))

    angle_error = np.degrees(
        np.arccos(np.clip(np.dot(R_align @ up, target_up), -1.0, 1.0))
    )
    validation = validate_alignment(P, up, floor_center, R_align, params)

    np.savez(
        args.out_meta,
        R_align=R_align.astype(np.float64),
        floor_center_da3=np.asarray(floor_center, dtype=np.float64),
        up_final_da3=np.asarray(up, dtype=np.float64),
        up_ransac_da3=(
            np.asarray(up_ransac, dtype=np.float64)
            if up_ransac is not None
            else np.full(3, np.nan, dtype=np.float64)
        ),
        up_geocalib_da3=(
            np.asarray(up_geocalib, dtype=np.float64)
            if up_geocalib is not None
            else np.full(3, np.nan, dtype=np.float64)
        ),
        up_geocalib_raw_da3=(
            np.asarray(up_geocalib_raw, dtype=np.float64)
            if up_geocalib_raw is not None
            else np.full(3, np.nan, dtype=np.float64)
        ),
        geocalib_angle_deg=(
            np.array(geocalib_angle_deg, dtype=np.float64)
            if geocalib_angle_deg is not None
            else np.array(np.nan, dtype=np.float64)
        ),
        geocalib_raw_angle_deg=(
            np.array(geocalib_raw_angle_deg, dtype=np.float64)
            if geocalib_raw_angle_deg is not None
            else np.array(np.nan, dtype=np.float64)
        ),
        geocalib_flipped_180=np.array(geocalib_flipped_180, dtype=bool),
        geocalib_direction_consistent=(
            np.array(geocalib_direction_consistent, dtype=bool)
            if geocalib_direction_consistent is not None
            else np.array(False, dtype=bool)
        ),
        used_geocalib_prior=np.array(used_geocalib_prior, dtype=bool),
        imu_mode=np.array(imu_mode, dtype=bool),
        used_imu_gravity=np.array(used_imu_gravity, dtype=bool),
        up_imu_da3=(
            np.asarray(up_imu, dtype=np.float64)
            if up_imu is not None
            else np.full(3, np.nan, dtype=np.float64)
        ),
        up_imu_raw_da3=(
            np.asarray(up_imu_raw, dtype=np.float64)
            if up_imu_raw is not None
            else np.full(3, np.nan, dtype=np.float64)
        ),
        imu_stats_json=np.array(json.dumps(imu_stats), dtype=object),
        fallback_to_ransac=np.array(fallback_to_ransac, dtype=bool),
        fallback_reason=np.array(str(fallback_reason or ""), dtype=object),
        decision_reason=np.array(str(decision_reason), dtype=object),
        ransac_camera_height_stats_json=np.array(json.dumps(ransac_camera_height_stats), dtype=object),
        geocalib_precheck_height_stats_json=np.array(json.dumps(geocalib_precheck_height_stats), dtype=object),
        final_camera_height_stats_json=np.array(json.dumps(final_camera_height_stats), dtype=object),
        final_hand_height_stats_json=np.array(json.dumps(final_hand_height_stats), dtype=object),
        ransac_floor_position_stats_json=np.array(json.dumps(ransac_floor_position_stats), dtype=object),
        final_floor_position_stats_json=np.array(json.dumps(final_floor_position_stats), dtype=object),
        height_validation_json=np.array(json.dumps({
            "ransac_height_valid": bool(ransac_height_valid),
            "ransac_height_reason": str(ransac_height_reason),
            "geocalib_precheck_valid": geocalib_precheck_valid,
            "geocalib_precheck_reason": geocalib_precheck_reason,
            "geocalib_flipped_180": bool(geocalib_flipped_180),
            "geocalib_flip_precheck_valid": geocalib_flip_precheck_valid,
            "geocalib_flip_precheck_reason": geocalib_flip_precheck_reason,
            "geocalib_raw_angle_deg": None if geocalib_raw_angle_deg is None else float(geocalib_raw_angle_deg),
            "geocalib_final_angle_deg": None if geocalib_angle_deg is None else float(geocalib_angle_deg),
            "geocalib_direction_consistent": geocalib_direction_consistent,
            "final_height_valid": bool(final_height_valid),
            "final_height_reason": str(final_height_reason),
            "final_hand_height_valid": bool(final_hand_height_valid),
            "final_hand_height_reason": str(final_hand_height_reason),
            "min_valid_camera_height_margin_factor": float(args.min_valid_camera_height_margin_factor),
            "min_camera_above_floor_ratio": float(args.min_camera_above_floor_ratio),
            "min_camera_p10_height_margin_factor": float(args.min_camera_p10_height_margin_factor),
        }), dtype=object),
        geocalib_flip_precheck_height_stats_json=np.array(
            json.dumps(geocalib_flip_precheck_height_stats),
            dtype=object,
        ),
        selected_plane_idx=np.array(selected["idx"], dtype=np.int64),
        selected_plane_score=np.array(selected["score"], dtype=np.float64),
        selected_plane_inliers=np.array(selected["inliers"], dtype=np.int64),
        selected_plane_area=np.array(selected["area"], dtype=np.float64),
        selected_plane_angle_to_prior_deg=np.array(
            selected.get("angle_to_prior_deg", np.nan),
            dtype=np.float64,
        ),
        selected_plane_height_rank=np.array(
            selected.get("height_rank", np.nan),
            dtype=np.float64,
        ),
        selected_plane_low_percentile=np.array(
            selected.get("low_percentile", np.nan),
            dtype=np.float64,
        ),
        selected_plane_source=np.array(
            selected.get("selection_source", "global_ransac"),
            dtype=object,
        ),
        geocalib_stats_json=np.array(json.dumps(geocalib_stats), dtype=object),
        low_prior_stats_json=np.array(json.dumps(low_prior_stats), dtype=object),
        params_json=np.array(
            json.dumps(
                {
                    "voxel_size": params["voxel_size"],
                    "distance_threshold": params["distance_threshold"],
                    "height_margin": params["height_margin"],
                    "refine_thresholds": params["refine_thresholds"],
                    "scene_extent": params["scene_extent"],
                    "nn_median": params["nn_median"],
                }
            ),
            dtype=object,
        ),
    )

    if imu_mode:
        floor_search = {
            k: v for k, v in imu_stats.get("floor_search", {}).items() if k != "bands"
        }
        print("[IMU] Saved rotation:", args.out)
        print("[IMU] Saved metadata:", args.out_meta)
        if imu_stats.get("reference_only"):
            alignment_source = "imu_fallback_alignment_only"
        elif used_imu_gravity:
            alignment_source = "imu_guided_low_floor_refined"
        else:
            alignment_source = "imu_failed_no_ransac_fallback"
        print("[IMU] Alignment source:", alignment_source)
        print("[IMU] Final up in DA3 world:", up)
        print("[IMU] Measured IMU vertical after hand-height sign check:", up_imu)
        print("[IMU] Raw IMU vertical from file:", up_imu_raw)
        print("[IMU] Point-cloud RANSAC baseline:", "not run in real-IMU mode")
        print("[IMU] Decision:", decision_reason)
        if fallback_reason:
            print("[IMU] Fallback reason:", fallback_reason)
        print("[IMU] Low plane search along IMU axis:", floor_search)
        print("[IMU] Candidates tried:", imu_stats.get("attempts", []))
        print("[IMU] Chosen low plane:", imu_stats.get("floor_fit", {}))
        print("[IMU] Refined floor from point cloud:", imu_stats.get("floor_refine", {}))
        print("[IMU] Head height stats:", final_camera_height_stats)
        print("[IMU] Head height valid:", final_height_valid, "reason:", final_height_reason)
        print("[IMU] Floor position stats:", final_floor_position_stats)
        print("[IMU] RANSAC head height stats:", "not run in real-IMU mode")
        print("[IMU] Selected floor source:", selected.get("selection_source", "global_ransac"))
        print("[IMU] Selected floor inliers:", selected["inliers"], "area:", selected["area"])
        print("[IMU] Check R_align @ up:", R_align @ up)
        print("[IMU] Angle to +Y after alignment:", angle_error, "deg")
        print(
            "[IMU] Auto params:",
            {
                "voxel_size": params["voxel_size"],
                "distance_threshold": params["distance_threshold"],
                "height_margin": params["height_margin"],
                "scene_extent": params["scene_extent"],
            },
        )
        print(
            "[IMU] Validation:",
            {
                "floor_points": validation["floor_points"],
                "floor_y_std": validation["floor_y_std"],
                "floor_y_range_p95_p5": validation["floor_y_range_p95_p5"],
            },
        )
        return

    print("Saved:", args.out)
    print("Saved metadata:", args.out_meta)
    print("Estimated up in DA3 world:", up)
    print("Original RANSAC up in DA3 world:", up_ransac)
    print("GeoCalib up in DA3 world:", up_geocalib)
    print("Raw GeoCalib up in DA3 world:", up_geocalib_raw)
    print("GeoCalib flipped 180:", geocalib_flipped_180)
    print("Raw RANSAC vs GeoCalib angle:", geocalib_raw_angle_deg)
    print("RANSAC vs GeoCalib angle:", geocalib_angle_deg)
    print("GeoCalib direction consistent:", geocalib_direction_consistent)
    print("Used GeoCalib prior:", used_geocalib_prior)
    print("Fallback to RANSAC:", fallback_to_ransac)
    print("Decision reason:", decision_reason)
    print("RANSAC camera height stats:", ransac_camera_height_stats)
    print("GeoCalib precheck height stats:", geocalib_precheck_height_stats)
    print("GeoCalib flip precheck height stats:", geocalib_flip_precheck_height_stats)
    print("Final camera height stats:", final_camera_height_stats)
    print("Final head/hand height stats:", final_hand_height_stats)
    print("Final head/hand height valid:", final_hand_height_valid, "reason:", final_hand_height_reason)
    print("RANSAC floor position stats:", ransac_floor_position_stats)
    print("Final floor position stats:", final_floor_position_stats)
    print("Final height valid:", final_height_valid, "reason:", final_height_reason)
    print("Check R_align @ up:", R_align @ up)
    print("Angle to +Y after alignment:", angle_error, "deg")
    print("Selected plane index:", selected["idx"])
    print("Selected plane score:", selected["score"])
    print("Selected plane inliers:", selected["inliers"])
    print("Selected plane area:", selected["area"])
    print("Selected plane source:", selected.get("selection_source", "global_ransac"))
    print("Selected plane height_rank:", selected.get("height_rank", None))
    print("Selected plane low_percentile:", selected.get("low_percentile", None))
    print("GeoCalib low-prior stats:", low_prior_stats)
    print("Auto voxel_size:", params["voxel_size"])
    print("Auto distance_threshold:", params["distance_threshold"])
    print("Auto height_margin:", params["height_margin"])
    print("Auto refine_thresholds:", params["refine_thresholds"])
    print("Validation floor points:", validation["floor_points"])
    print("Validation floor_y_std:", validation["floor_y_std"])
    print("Validation floor_y_range_p95_p5:", validation["floor_y_range_p95_p5"])


if __name__ == "__main__":
    main()
