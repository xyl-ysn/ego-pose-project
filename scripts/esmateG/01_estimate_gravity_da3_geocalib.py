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


def detect_plane_candidates(P, params):
    pcd_work = o3d.geometry.PointCloud()
    pcd_work.points = o3d.utility.Vector3dVector(P)

    total_points = len(P)
    min_inliers = max(MIN_INLIERS_ABS, int(total_points * MIN_INLIER_RATIO))
    candidates = []

    for plane_idx in range(N_PLANES):
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

        candidates.append({
            "idx": plane_idx,
            "normal": n_refined,
            "center": center,
            "inliers": int(len(inliers)),
            "ratio": float(len(inliers) / total_points),
            "rms": rms,
            "area": area,
        })

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

        item = dict(plane)
        item.update({
            "up": up,
            "score": float(score),
            "sign_stats": stats,
            "angle_to_prior_deg": angle_to_prior,
        })

        if score > best_score:
            best_score = score
            best = item

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


def select_low_prior_floor_plane(P, cam_centers, params, up_prior):
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

        item = dict(plane)
        item.update({
            "up": up,
            "score": float(score),
            "sign_stats": stats,
            "angle_to_prior_deg": angle_to_prior,
            "min_refine_inliers": LOW_PRIOR_MIN_INLIERS_ABS,
        })

        if score > best_score:
            best_score = score
            best = item

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
    low_position_valid = height_rank <= LOW_PRIOR_ACCEPT_MAX_HEIGHT_RANK
    tabletop_risk = height_rank > TABLETOP_RISK_HEIGHT_RANK and below_margin_ratio > 0.05

    return {
        "available": True,
        "center_height": center_height,
        "height_rank": height_rank,
        "below_margin_ratio": below_margin_ratio,
        "low_position_valid": bool(low_position_valid),
        "tabletop_risk": bool(tabletop_risk),
        "max_valid_height_rank": LOW_PRIOR_ACCEPT_MAX_HEIGHT_RANK,
        "tabletop_risk_height_rank": TABLETOP_RISK_HEIGHT_RANK,
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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", required=True, help="Input DA3 combined_pcd.ply")
    parser.add_argument("--poses", required=True, help="Input DA3 camera_poses.txt, fixed as c2w")
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

    P_full = load_point_cloud(args.pcd)
    params = compute_auto_params(P_full)
    P = downsample_for_estimation(P_full, params["voxel_size"])

    Ts = load_da3_poses_c2w(args.poses)
    cam_centers = Ts[:, :3, 3]

    candidates = detect_plane_candidates(P, params)

    # Stage 1: original DA3 point-cloud-only RANSAC selection.
    selected_ransac = select_floor_plane(P, candidates, cam_centers, params)
    up_ransac, floor_center_ransac, floor_mask_ransac = refine_floor_coarse_to_fine(
        P,
        selected_ransac,
        params,
    )
    up_ransac = final_sign_check(P, up_ransac, floor_center_ransac, cam_centers, params)

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

    # Stage 2: optional GeoCalib appearance-based gravity prior.
    if args.geocalib_gravity is not None:
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

                        if args.fallback_to_ransac_on_invalid_height and not final_height_valid:
                            fallback_to_ransac = True
                            fallback_reason = f"geocalib_prior_height_validation_failed: {final_height_reason}"
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
                            final_floor_position_stats = floor_position_stats(
                                P,
                                floor_center,
                                up,
                                params,
                            )
                            decision_reason = (
                                "geocalib_disagrees_with_ransac_prior_refined_and_accepted: "
                                f"angle={geocalib_angle_deg:.6f} > {args.geocalib_switch_angle_deg:.6f}; "
                                f"height_valid={final_height_valid}; reason={final_height_reason}"
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
        up_ransac_da3=np.asarray(up_ransac, dtype=np.float64),
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
        fallback_to_ransac=np.array(fallback_to_ransac, dtype=bool),
        fallback_reason=np.array(str(fallback_reason or ""), dtype=object),
        decision_reason=np.array(str(decision_reason), dtype=object),
        ransac_camera_height_stats_json=np.array(json.dumps(ransac_camera_height_stats), dtype=object),
        geocalib_precheck_height_stats_json=np.array(json.dumps(geocalib_precheck_height_stats), dtype=object),
        final_camera_height_stats_json=np.array(json.dumps(final_camera_height_stats), dtype=object),
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
