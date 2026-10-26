#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import argparse
import json
import pickle
import re
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np


def load_camera_poses(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] == 16:
        return data.reshape(-1, 4, 4)
    if data.shape[1] == 17:
        return data[:, 1:].reshape(-1, 4, 4)
    if data.shape[1] == 12:
        out = np.tile(np.eye(4), (len(data), 1, 1))
        out[:, :3, :4] = data.reshape(-1, 3, 4)
        return out
    if data.shape[1] == 13:
        out = np.tile(np.eye(4), (len(data), 1, 1))
        out[:, :3, :4] = data[:, 1:].reshape(-1, 3, 4)
        return out
    raise ValueError(f"Unsupported camera pose shape {data.shape}: {path}")


def read_timestamps(path):
    if path is None:
        return {}
    table = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")
    if table.shape == ():
        table = np.asarray([table])
    required = {"frame_id", "timestamp", "src_frame_id"}
    if not required.issubset(set(table.dtype.names or [])):
        raise ValueError(f"{path} must have header: frame_id timestamp src_frame_id")
    return {
        int(row["frame_id"]): {
            "timestamp": float(row["timestamp"]),
            "timestamp_ns": int(round(float(row["timestamp"]) * 1e9)),
            "src_frame_id": int(row["src_frame_id"]),
        }
        for row in table
    }


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            hands = []
            for source_hand in rec.get("hands", []):
                hand = dict(source_hand)
                for canonical, upstream in (
                    ("world_keypoints_3d", "keypoints_3d"),
                    ("world_mesh_vertices_3d", "mesh_vertices_3d"),
                ):
                    if hand.get(canonical) is None and upstream in hand:
                        hand[canonical] = hand[upstream]
                hands.append(hand)
            rec["hands"] = hands
            rec["_line_no"] = line_no
            yield rec


def world_points_to_camera(points_world, T_world_cam):
    points_world = np.asarray(points_world, dtype=np.float64)
    R_world_cam = T_world_cam[:3, :3]
    t_world_cam = T_world_cam[:3, 3]
    return (points_world - t_world_cam[None, :]) @ R_world_cam


def require_points(data, key, context):
    if key not in data or data[key] is None:
        raise KeyError(f"{context}: missing {key}")
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{context}: {key} must have shape (N,3), got {arr.shape}")
    return arr


def optional_points(data, key):
    if key not in data or data[key] is None:
        return None
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


EXPLICIT_SOFT_ALPHA_KEYS = (
    "constraint_weight",
    "visualization_weight",
    "visualization_alpha",
    "soft_start_weight",
    "soft_start_alpha",
    "soft_end_weight",
    "soft_end_alpha",
)

# These are detector confidence fields, not soft-start fields. They are only used
# when explicitly requested via --soft-alpha-field confidence/score/etc.
DETECTION_SCORE_KEYS = (
    "confidence",
    "score",
    "hand_score",
    "det_score",
)


def finite_scalar(value, default=None):
    """Convert a scalar/list/np array to a finite float."""
    if value is None:
        return default
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return out if np.isfinite(out) else default
    if arr.size == 0:
        return default
    out = float(arr[0])
    return out if np.isfinite(out) else default


def clip01(value):
    value = finite_scalar(value, default=1.0)
    return float(np.clip(value, 0.0, 1.0))


def pick_hand_soft_alpha(hand, requested_key="auto", default=1.0):
    """Read soft-start alpha from a prepared hand record.

    Important compatibility rule:
    - auto mode only reads explicit soft-start fields written by the updated 02 script.
    - old detector confidence/score fields are NOT treated as soft-start weights in auto mode,
      because doing so would change old behavior.
    - to intentionally use detector confidence as alpha, pass --soft-alpha-field confidence.
    """
    if requested_key in (None, "", "none", "None"):
        return float(default)

    if requested_key == "auto":
        keys = EXPLICIT_SOFT_ALPHA_KEYS
    else:
        keys = (requested_key,)

    for key in keys:
        if key not in hand:
            continue
        value = finite_scalar(hand.get(key), default=None)
        if value is None:
            continue
        return float(np.clip(value, 0.0, 1.0))
    return float(default)


def records_have_explicit_soft_alpha(records):
    """Return True if the prepared JSONL contains fields from the soft-start 02 script."""
    for rec in records:
        for hand in rec.get("hands", []):
            if any(key in hand for key in EXPLICIT_SOFT_ALPHA_KEYS):
                return True
    return False


def add_soft_weight_fields(det, soft_alpha, hand, constraint_mode="full"):
    """Attach normalized per-residual weights to one EgoAllo hand detection dict."""
    weight = np.asarray([float(np.clip(soft_alpha, 0.0, 1.0))], dtype=np.float32)
    is_wrist_only = str(constraint_mode) == "wrist_only"
    det["constraint_weight"] = weight.copy()
    det["hand_pose_weight"] = np.zeros_like(weight) if is_wrist_only else weight.copy()
    det["hand_reproj_weight"] = np.zeros_like(weight) if is_wrist_only else weight.copy()
    det["wrist_pose_weight"] = weight.copy()
    det["visualization_alpha"] = weight.copy()
    det["visualization_weight"] = weight.copy()
    det["render_alpha"] = weight.copy()
    det["soft_alpha"] = weight.copy()
    det["confidence"] = weight.copy()
    det["score"] = weight.copy()
    det["constraint_mode"] = np.asarray([str(constraint_mode)], dtype=object)
    det["wrist_only"] = np.asarray([is_wrist_only], dtype=np.bool_)

    raw_conf = finite_scalar(hand.get("raw_confidence", None), default=None)
    if raw_conf is not None:
        det["raw_confidence"] = np.asarray([raw_conf], dtype=np.float32)
    return det


def validate_rotation_matrix(R, name):
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3,3), got {R.shape}")
    det = float(np.linalg.det(R))
    orth_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    if abs(det - 1.0) > 1e-3 or orth_err > 1e-3:
        raise ValueError(
            f"{name} is not a valid rotation matrix: det={det:.6f}, orth_err={orth_err:.6g}"
        )
    return R.astype(np.float32)


def estimate_hawor_to_vggt_camera_rotation(records, min_soft_alpha=0.0, soft_alpha_field="auto"):
    """Estimate column-vector rotation: p_vggt_cam ~= R @ p_hawor_cam."""
    src_vecs = []
    dst_vecs = []
    for rec in records:
        for hand in rec.get("hands", []):
            if pick_hand_soft_alpha(hand, soft_alpha_field, default=1.0) < min_soft_alpha:
                continue
            hawor = optional_points(hand, "hawor_camera_keypoints_3d")
            vggt = optional_points(hand, "camera_keypoints_3d")
            if hawor is None or vggt is None or len(hawor) != len(vggt) or len(hawor) < 2:
                continue
            hawor_centered = hawor - hawor[0:1]
            vggt_centered = vggt - vggt[0:1]
            valid = (
                np.linalg.norm(hawor_centered, axis=1) > 1e-6
            ) & (
                np.linalg.norm(vggt_centered, axis=1) > 1e-6
            )
            src_vecs.append(hawor_centered[valid])
            dst_vecs.append(vggt_centered[valid])

    if not src_vecs:
        raise ValueError(
            "Cannot estimate HaWoR-camera -> VGGT-camera rotation. "
            "prepared_hands_jsonl must contain both hawor_camera_keypoints_3d and camera_keypoints_3d, "
            "or pass --camera-frame-rotation-npy, or use --mano-global-orient-frame vggt_camera."
        )

    src = np.concatenate(src_vecs, axis=0)
    dst = np.concatenate(dst_vecs, axis=0)
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    # H uses row-wise points, so V U^T is the column-vector rotation.
    # Its transpose is used only when applying it to row-wise points.
    R_col = Vt.T @ U.T
    if np.linalg.det(R_col) < 0.0:
        Vt[-1, :] *= -1.0
        R_col = Vt.T @ U.T

    pred = src @ R_col.T
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    src_norm = float(np.sqrt(np.mean(np.sum(src ** 2, axis=1))))
    dst_norm = float(np.sqrt(np.mean(np.sum(dst ** 2, axis=1))))
    scale = dst_norm / src_norm if src_norm > 1e-8 else float("nan")
    R_col = validate_rotation_matrix(R_col, "estimated camera-frame rotation")
    angle = np.degrees(
        np.arccos(np.clip((np.trace(R_col.astype(np.float64)) - 1.0) * 0.5, -1.0, 1.0))
    )
    print(
        "Estimated HaWoR-camera -> VGGT-camera rotation: "
        f"angle={angle:.3f} deg, scale_ratio={scale:.6f}, keypoint_rmse={rmse:.6g}"
    )
    return R_col


def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def estimate_R_cam_wrist(keypoints, side):
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.shape[0] < 18:
        return np.eye(3, dtype=np.float32)
    wrist = kp[0]
    index_mcp = kp[5]
    middle_mcp = kp[9]
    pinky_mcp = kp[17]

    x_axis = normalize(middle_mcp - wrist)
    z_raw = normalize(index_mcp - pinky_mcp)
    if side == "right":
        x_axis = -x_axis
    y_axis = normalize(-np.cross(x_axis, z_raw))
    if np.linalg.norm(y_axis) < 1e-6:
        return np.eye(3, dtype=np.float32)
    z_axis = normalize(np.cross(x_axis, y_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)


def rodrigues(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float32)
    axis = rotvec / theta
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R.astype(np.float32)


def normalize_pose_array(value):
    arr = np.asarray(value, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.shape == (15, 3, 3):
        return arr
    if arr.shape == (45,):
        arr = arr.reshape(15, 3)
    if arr.shape == (15, 3):
        return np.stack([rodrigues(v) for v in arr], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported MANO hand pose shape: {arr.shape}")


def normalize_orient_array(value):
    arr = np.asarray(value, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.shape == (3, 3):
        return arr.astype(np.float32)
    if arr.shape == (3,):
        return rodrigues(arr).astype(np.float32)
    raise ValueError(f"Unsupported MANO global orient shape: {arr.shape}")


def normalize_betas(value):
    arr = np.asarray(value, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.shape == (10,):
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported MANO betas shape: {arr.shape}")


def pick_first_key(data, keys):
    for key in keys:
        if key in data:
            return data[key]
    return None


def infer_frame_ids_from_file(path, count):
    numbers = [int(x) for x in re.findall(r"\d+", path.stem)]
    if not numbers:
        return None
    if len(numbers) >= 2 and count > 1:
        start = numbers[0]
        return list(range(start, start + count))
    return [numbers[0]] if count == 1 else None


def add_mano_record(mapping, side, frame_id, root_orient, hand_pose, betas, source_track=None):
    mapping.setdefault((side, int(frame_id)), []).append({
        "mano_hand_global_orient": normalize_orient_array(root_orient),
        "mano_hand_pose": normalize_pose_array(hand_pose),
        "mano_hand_betas": normalize_betas(betas),
        "source_track": source_track,
    })


@lru_cache(maxsize=2)
def load_mano_layer(path, side):
    from smplx import MANOLayer
    layer = MANOLayer(str(path), is_rhand=side == "right", flat_hand_mean=True).eval()
    if side == "left":
        right_path = Path(path).with_name("MANO_RIGHT.pkl")
        if right_path.is_file():
            right = load_mano_layer(str(right_path), "right")
            left_x = layer.shapedirs[:, 0, :].detach().cpu().numpy()
            right_x = right.shapedirs[:, 0, :].detach().cpu().numpy()
            # Some MANO_LEFT releases retained the right-hand X shape basis.
            # Mirror it only when the paired models identify that known bug;
            # already corrected or unrelated bases must remain untouched.
            if (left_x.shape == right_x.shape and np.linalg.norm(right_x) > 0
                    and np.linalg.norm(left_x - right_x) <= 1e-3 * np.linalg.norm(right_x)):
                layer.shapedirs[:, 0, :] *= -1
    return layer


def select_mano_record(mapping, side, frame_id, keypoints_cam=None, mano_path=None):
    candidates = mapping.get((side, int(frame_id)), [])
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    if keypoints_cam is None or mano_path is None:
        return None

    # Overlapping chunks can contain different hands at the same frame. Match
    # wrist-relative geometry in the HaWoR camera frame, without using depth.
    target = np.asarray(keypoints_cam, dtype=np.float32)
    if target.shape != (21, 3) or not np.isfinite(target).all():
        return None
    target = target[[0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]]
    target = target - target[:1]
    norm = np.linalg.norm(target)
    if norm < 1e-6:
        return None
    import torch
    layer = load_mano_layer(str(mano_path), side)
    tensor = lambda key: torch.from_numpy(np.stack([c[key] for c in candidates]))
    with torch.no_grad():
        predicted = layer(betas=tensor("mano_hand_betas"),
                          global_orient=tensor("mano_hand_global_orient")[:, None],
                          hand_pose=tensor("mano_hand_pose")).joints.numpy()
    predicted = predicted - predicted[:, :1]
    scale = (predicted * target).sum((1, 2)) / np.maximum((predicted ** 2).sum((1, 2)), 1e-12)
    errors = np.linalg.norm((predicted * scale[:, None, None] - target).reshape(len(candidates), -1), axis=1) / norm
    errors[(scale <= 0) | ~np.isfinite(errors)] = np.inf
    best, runner_up = np.argsort(errors)[:2]
    if errors[best] > 0.15 or errors[runner_up] < max(errors[best] * 1.5, errors[best] + 0.03):
        return None
    return candidates[best]


def load_cam_space_json(path, side, frame_key):
    data = json.load(open(path, "r", encoding="utf-8"))
    records = []

    if isinstance(data, list):
        # Many HaWoR/HaMeR exports are chunk files named by source-frame ids,
        # e.g. 18_19.json, and the inner records may not carry src_frame_id.
        # When frame_key is src_frame_id, prefer those filename-derived ids over
        # a generic inner frame_id so MANO lookup matches the video source frame.
        inferred_frame_ids = infer_frame_ids_from_file(path, len(data))
        for i, item in enumerate(data):
            if isinstance(item, dict):
                rec = dict(item)
                if inferred_frame_ids is not None and i < len(inferred_frame_ids):
                    rec["_filename_frame_id"] = int(inferred_frame_ids[i])
                    if frame_key not in rec:
                        rec[frame_key] = int(inferred_frame_ids[i])
                records.append(rec)
    elif isinstance(data, dict):
        root = pick_first_key(data, ["init_root_orient", "root_orient", "mano_hand_global_orient"])
        pose = pick_first_key(data, ["init_hand_pose", "hand_pose", "mano_hand_pose"])
        betas = pick_first_key(data, ["init_betas", "betas", "mano_hand_betas"])
        if root is None or pose is None or betas is None:
            return []

        root_arr = np.asarray(root)
        pose_arr = np.asarray(pose)
        betas_arr = np.asarray(betas)

        if root_arr.ndim >= 3 and root_arr.shape[-2:] == (3, 3):
            count = root_arr.reshape(-1, 3, 3).shape[0]
        elif root_arr.ndim >= 2 and root_arr.shape[-1] == 3:
            count = root_arr.reshape(-1, 3).shape[0]
        else:
            count = 1

        # Prefer the requested key. If it is absent, prefer source-frame ids
        # inferred from the filename (e.g. 18_19.json) before falling back to
        # generic frame_ids/frames fields, which may refer to extracted indices.
        frame_ids = pick_first_key(data, [frame_key])
        if frame_ids is None:
            frame_ids = infer_frame_ids_from_file(path, count)
        if frame_ids is None:
            frame_ids = pick_first_key(data, ["frame_ids", "frames", "valid_frames"])
        if frame_ids is None:
            return []
        frame_ids = list(np.asarray(frame_ids).reshape(-1).astype(int))

        root_seq = root_arr.reshape(count, *root_arr.shape[-2:]) if root_arr.shape[-2:] == (3, 3) else root_arr.reshape(count, 3)
        if pose_arr.shape[-2:] == (3, 3):
            pose_seq = pose_arr.reshape(count, 15, 3, 3)
        elif pose_arr.shape[-1] == 45:
            pose_seq = pose_arr.reshape(count, 45)
        else:
            pose_seq = pose_arr.reshape(count, 15, 3)
        betas_seq = betas_arr.reshape(count, 10)

        for i, frame_id in enumerate(frame_ids[:count]):
            records.append(
                {
                    "frame_id": int(frame_id),
                    "root": root_seq[i],
                    "pose": pose_seq[i],
                    "betas": betas_seq[i],
                }
            )
    return records


def load_cam_space_dirs(left_dir, right_dir, frame_key):
    mapping = {}
    for side, directory in [("left", left_dir), ("right", right_dir)]:
        if directory is None:
            continue
        directory = Path(directory)
        if not directory.exists():
            print(f"[HaWoR] cam_space {side} dir not found; skipping MANO records: {directory}")
            continue
        for path in sorted(directory.glob("*.json")):
            for rec in load_cam_space_json(path, side, frame_key):
                frame_id = rec.get(frame_key, rec.get("_filename_frame_id", rec.get("frame_id")))
                root = rec.get("root", pick_first_key(rec, ["init_root_orient", "root_orient", "mano_hand_global_orient"]))
                pose = rec.get("pose", pick_first_key(rec, ["init_hand_pose", "hand_pose", "mano_hand_pose"]))
                betas = rec.get("betas", pick_first_key(rec, ["init_betas", "betas", "mano_hand_betas"]))
                if frame_id is None or root is None or pose is None or betas is None:
                    continue
                add_mano_record(mapping, side, frame_id, root, pose, betas, str(path))
    return mapping


def load_mano_timeline(path, frame_key="frame_id"):
    """Read the original camera timebase, before 5 Hz -> 30 Hz resampling."""
    import pandas as pd

    table = pd.read_parquet(path)
    id_column = "source_frame_idx" if frame_key == "src_frame_id" else "frame_id"
    required = {id_column, "timestamp_ns", "T_world_from_camera"}
    if not required.issubset(table.columns):
        raise ValueError(f"MANO timeline {path} requires columns {sorted(required)}")
    table = table.sort_values("timestamp_ns")
    times = table.timestamp_ns.to_numpy(dtype=np.int64)
    ids = table[id_column].to_numpy(dtype=np.int64)
    if not len(times) or np.any(np.diff(times) <= 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("MANO timeline must have unique frame IDs and increasing timestamps")
    rotations = np.stack([np.asarray(x, dtype=np.float64).reshape(4, 4)[:3, :3]
                          for x in table.T_world_from_camera])
    if not np.isfinite(rotations).all():
        raise ValueError("Non-finite source camera rotations in MANO timeline")
    return dict(times_ns=times, frame_ids=ids, camera_rotations=rotations)


def discover_mano_timeline(left_dir, right_dir):
    """Only auto-discover the original camera table beside a cam_space directory."""
    paths = {Path(directory).parent.parent / "camera.parquet"
             for directory in (left_dir, right_dir) if directory is not None
             and Path(directory).parent.name == "cam_space"}
    paths = {path.resolve() for path in paths if path.is_file()}
    if len(paths) > 1:
        raise ValueError("Left and right MANO directories refer to different camera timelines")
    return next(iter(paths)) if paths else None


def interpolate_rotations(a, b, fraction):
    """Shortest-arc rotation interpolation, including a stack of finger joints."""
    from scipy.spatial.transform import Rotation

    shape = np.shape(a)
    ra = Rotation.from_matrix(np.asarray(a).reshape(-1, 3, 3))
    rb = Rotation.from_matrix(np.asarray(b).reshape(-1, 3, 3))
    relative = (ra.inv() * rb).as_rotvec()
    return (ra * Rotation.from_rotvec(relative * fraction)).as_matrix().reshape(shape).astype(np.float32)


def select_mano_at_timestamp(mapping, side, timestamp_ns, timeline, *,
                             keypoints_cam=None, mano_path=None,
                             R_vggtcam_haworcam=None, R_world_target_cam=None):
    """Match source MANO by time, never by a resampled frame number.

    Interpolate only adjacent source camera frames from the same hand chunk.
    No extrapolation or filling missing MANO anchors is permitted. Global hand
    orientations are interpolated in world coordinates, then returned in the
    target HaWoR-camera convention expected by the existing converter.
    Source anchors allow 1 ns timestamp rounding tolerance.
    """
    times = timeline["times_ns"]
    position = int(np.searchsorted(times, timestamp_ns))
    if position < len(times) and abs(int(times[position]) - timestamp_ns) <= 1:
        lower = upper = position
    elif position > 0 and abs(int(times[position - 1]) - timestamp_ns) <= 1:
        lower = upper = position - 1
    elif position == 0 or position == len(times):
        return None
    else:
        lower, upper = position - 1, position
        # An irregular timeline must not silently bridge missing source frames.
        nominal_period = float(np.median(np.diff(times)))
        if times[upper] - times[lower] > 1.5 * nominal_period:
            return None
    first = select_mano_record(mapping, side, timeline["frame_ids"][lower], keypoints_cam, mano_path)
    last = (first if lower == upper else select_mano_record(
        mapping, side, timeline["frame_ids"][upper], keypoints_cam, mano_path))
    if first is None or last is None:
        return None
    if lower != upper and first.get("source_track") != last.get("source_track"):
        return None
    fraction = 0. if lower == upper else (timestamp_ns - times[lower]) / float(times[upper] - times[lower])
    camera_change = np.eye(3) if R_vggtcam_haworcam is None else np.asarray(R_vggtcam_haworcam)
    target_camera = (timeline["camera_rotations"][lower] if R_world_target_cam is None
                     else np.asarray(R_world_target_cam))
    first_world = timeline["camera_rotations"][lower] @ camera_change @ first["mano_hand_global_orient"]
    last_world = timeline["camera_rotations"][upper] @ camera_change @ last["mano_hand_global_orient"]
    root = camera_change.T @ target_camera.T @ interpolate_rotations(first_world, last_world, fraction)
    return {
        "mano_hand_global_orient": root.astype(np.float32),
        "mano_hand_pose": interpolate_rotations(first["mano_hand_pose"], last["mano_hand_pose"], fraction),
        "mano_hand_betas": ((1. - fraction) * first["mano_hand_betas"]
                            + fraction * last["mano_hand_betas"]).astype(np.float32),
        "source_track": first.get("source_track"),
    }


def validate_hand_records(records, timestamp_map):
    """Require hand frame IDs and timestamps to use the target camera timebase."""
    for rec in records:
        frame_id = int(rec["frame_id"])
        if frame_id not in timestamp_map:
            raise KeyError(f"frame_id {frame_id} not found in timestamps table")
        target_ns = int(timestamp_map[frame_id]["timestamp_ns"])
        if "timestamp_ns" in rec:
            record_ns = int(rec["timestamp_ns"])
        elif "timestamp" in rec:
            record_ns = int(round(float(rec["timestamp"]) * 1e9))
        else:
            raise ValueError("Hand input requires a timestamp for every record")
        if abs(record_ns - target_ns) > 1:
            raise ValueError(f"Hand input timestamp mismatch at frame_id={frame_id}")


def make_placeholder_mano(keypoints_cam, side):
    return {
        "mano_hand_pose": np.tile(np.eye(3, dtype=np.float32), (15, 1, 1)),
        "mano_hand_betas": np.zeros(10, dtype=np.float32),
        "mano_hand_global_orient": estimate_R_cam_wrist(keypoints_cam, side),
    }


def patch_numpy_aliases_for_chumpy():
    # Old MANO pickle files may import chumpy, which expects NumPy aliases
    # removed in recent NumPy versions.
    aliases = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def load_faces(npy_path, mano_pkl_path, side):
    if npy_path is not None:
        faces = np.load(npy_path)
        source = npy_path
    elif mano_pkl_path is not None:
        patch_numpy_aliases_for_chumpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            with open(mano_pkl_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
        if "f" not in data:
            raise KeyError(f"{mano_pkl_path} does not contain MANO face field 'f'")
        faces = data["f"]
        source = mano_pkl_path
    else:
        raise ValueError(
            f"Missing faces for {side} hand. Pass --mano-faces-{side}-npy or --mano-{side}-pkl."
        )

    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"{source}: MANO faces must have non-empty shape (F,3), got {faces.shape}")
    if faces.min() < 0 or faces.max() >= 778:
        raise ValueError(
            f"{source}: MANO face indices look invalid, min={faces.min()}, max={faces.max()}"
        )
    print(f"Loaded {side} MANO faces: {faces.shape} from {source}")
    return faces


def _arg_supplied(*names):
    for token in sys.argv[1:]:
        for name in names:
            if token == name or token.startswith(name + "="):
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-hands-jsonl", required=True, type=Path)
    parser.add_argument("--camera-poses-txt", required=True, type=Path)
    parser.add_argument("--timestamps-txt", required=True, type=Path)
    parser.add_argument("--output-pkl", required=True, type=Path)
    parser.add_argument("--debug-camera-jsonl", type=Path, default=None)
    parser.add_argument(
        "--soft-alpha-field",
        default="auto",
        help=(
            "Prepared-hand field used as soft-start alpha. Use 'auto' to try "
            "constraint_weight, visualization_weight, visualization_alpha, "
            "soft_start_weight, soft_start_alpha, soft_end_weight and soft_end_alpha. "
            "Use 'none' to disable soft alpha handling, or explicitly choose confidence/score."
        ),
    )
    parser.add_argument(
        "--min-soft-alpha-to-keep",
        type=float,
        default=0.0,
        help=(
            "Skip a hand detection when its soft alpha is below this value. "
            "Default 0 keeps detections and lets EgoAllo consume soft alpha through constraint_weight. "
            "Use a positive value only for explicit hard filtering."
        ),
    )
    parser.add_argument(
        "--auto-soft-alpha-threshold",
        type=float,
        default=0.0,
        help=(
            "Threshold used only when --min-soft-alpha-to-keep is negative and the "
            "prepared JSONL contains explicit soft-start fields. Default 0.25 skips "
            "the first very low-weight frames of a newly appearing hand."
        ),
    )
    parser.add_argument(
        "--visualization-safe-mode",
        action="store_true",
        default=False,
        help=(
            "For visualizers/IK paths that still snap to hand geometry even with soft weights, "
            "hard-skip detections below --visualization-safe-alpha unless --min-soft-alpha-to-keep is explicitly set."
        ),
    )
    parser.add_argument(
        "--visualization-safe-alpha",
        type=float,
        default=0.35,
        help="Alpha threshold used by --visualization-safe-mode. Default: 0.35.",
    )
    parser.add_argument(
        "--write-soft-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Deprecated compatibility flag. Normalized constraint weights are now always written "
            "because EgoAllo consumes hand_pose_weight/hand_reproj_weight/wrist_pose_weight."
        ),
    )
    parser.add_argument(
        "--camera-rotation-min-soft-alpha",
        type=float,
        default=0.0,
        help=(
            "When estimating HaWoR-camera -> VGGT-camera rotation, ignore hands "
            "whose soft alpha is below this threshold. Default 0 keeps old behavior."
        ),
    )
    parser.add_argument("--cam-space-left-dir", type=Path, default=None)
    parser.add_argument("--cam-space-right-dir", type=Path, default=None)
    parser.add_argument("--mano-frame-key", default="src_frame_id")
    parser.add_argument(
        "--mano-timeline-parquet", type=Path, default=None,
        help="Original camera.parquet mapping cam_space frame IDs to episode timestamps; "
             "automatically discovered beside cam_space when available. Interpolates MANO "
             "only within valid source chunks, without changing input wrist positions.",
    )
    parser.add_argument(
        "--skip-missing-mano",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip hand detections without per-frame MANO params. Disabled by default: if wrist/keypoints are valid, "
            "missing MANO detections are downgraded to wrist-only constraints."
        ),
    )
    parser.add_argument("--require-mano", action="store_true")
    parser.add_argument(
        "--allow-placeholder-mano-for-interpolated",
        action="store_true",
        help=(
            "When --require-mano is set, still allow interpolated hand detections "
            "to use a wrist-based placeholder MANO record. Raw detections remain strict."
        ),
    )
    parser.add_argument("--mano-faces-left-npy", type=Path, default=None)
    parser.add_argument("--mano-faces-right-npy", type=Path, default=None)
    parser.add_argument("--mano-left-pkl", type=Path, default=None)
    parser.add_argument("--mano-right-pkl", type=Path, default=None)
    parser.add_argument(
        "--mano-global-orient-frame",
        choices=("hawor_camera", "vggt_camera"),
        default="hawor_camera",
        help=(
            "Coordinate frame of cam_space MANO global orientation. "
            "Default assumes HaWoR camera frame and converts it to VGGT camera frame."
        ),
    )
    parser.add_argument(
        "--camera-frame-rotation-npy",
        type=Path,
        default=None,
        help="Optional 3x3 rotation mapping HaWoR camera vectors to VGGT camera vectors.",
    )
    parser.add_argument(
        "--pose-index-field",
        choices=("frame_id", "src_frame_id"),
        default="frame_id",
        help="Which record id indexes rows in camera_poses.txt.",
    )
    # Inverse of T_CAMERA_CPF in 05_run_egogvae_from_head.py: camera x/y
    # are opposite to SMPL-H CPF x/y. Hand arrays remain in camera coordinates.
    parser.add_argument(
        "--T-cpf-cam",
        nargs=7,
        type=float,
        default=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        metavar=("qw", "qx", "qy", "qz", "tx", "ty", "tz"),
        help="Camera-to-CPF transform (wxyz, xyz); defaults to the EgoGVAE camera convention.",
    )
    args = parser.parse_args()

    if args.visualization_safe_mode and not _arg_supplied("--min-soft-alpha-to-keep"):
        args.min_soft_alpha_to_keep = float(args.visualization_safe_alpha)
        print(
            "Visualization-safe mode: "
            f"hard-skipping detections with soft alpha < {args.min_soft_alpha_to_keep:.3f}."
        )

    if not args.write_soft_weights:
        print("[Warning] --no-write-soft-weights is deprecated; normalized constraint weights are still written.")
    print("Normalized soft weights will be written into EgoAllo hand detections.")
    if args.skip_missing_mano:
        print("Missing per-frame MANO detections will be skipped.")
    else:
        print("Missing per-frame MANO detections will become wrist-only constraints when wrist/keypoints are valid.")
    print(
        f"MANO records will be loaded with mano_frame_key={args.mano_frame_key!r} "
        "and matched by episode timestamp."
    )

    poses = load_camera_poses(args.camera_poses_txt)
    timestamp_map = read_timestamps(args.timestamps_txt)
    records = list(iter_jsonl(args.prepared_hands_jsonl))
    validate_hand_records(records, timestamp_map)
    has_explicit_soft_alpha = records_have_explicit_soft_alpha(records)
    if args.min_soft_alpha_to_keep < 0:
        if args.write_soft_weights:
            # True soft-start path: keep low-alpha detections in the pickle and
            # let EgoAllo multiply their residuals by constraint_weight.
            effective_min_soft_alpha_to_keep = 0.0
        else:
            # Backward-compatible path for unmodified EgoAllo: skip very low-alpha
            # detections because the official HaMeR reader otherwise treats every
            # non-None detection as a hard constraint.
            effective_min_soft_alpha_to_keep = (
                float(args.auto_soft_alpha_threshold)
                if has_explicit_soft_alpha and args.soft_alpha_field not in (None, "", "none", "None")
                else 0.0
            )
    else:
        effective_min_soft_alpha_to_keep = float(args.min_soft_alpha_to_keep)
    effective_min_soft_alpha_to_keep = float(np.clip(effective_min_soft_alpha_to_keep, 0.0, 1.0))

    if has_explicit_soft_alpha:
        print(
            "Detected explicit soft-start fields in prepared hands. "
            f"Using min_soft_alpha_to_keep={effective_min_soft_alpha_to_keep:.3f}."
        )
    elif effective_min_soft_alpha_to_keep > 0:
        print(
            "[Warning] min_soft_alpha_to_keep > 0 but no explicit soft-start fields were found. "
            f"Field '{args.soft_alpha_field}' will be used if present; otherwise alpha defaults to 1."
        )
    else:
        print("No explicit soft-start fields found; keeping old hand-detection behavior.")

    mano_map = load_cam_space_dirs(
        args.cam_space_left_dir,
        args.cam_space_right_dir,
        args.mano_frame_key,
    )
    mano_timeline_path = args.mano_timeline_parquet or discover_mano_timeline(
        args.cam_space_left_dir, args.cam_space_right_dir)
    mano_timeline = (load_mano_timeline(mano_timeline_path, args.mano_frame_key)
                     if mano_timeline_path is not None else None)
    if mano_map and mano_timeline is None:
        raise ValueError("MANO timestamp matching requires --mano-timeline-parquet or an auto-discovered "
                         "source camera.parquet; resampled frame numbers cannot identify source MANO frames")
    if mano_timeline is not None:
        print(f"MANO records matched by episode timestamp using {mano_timeline_path}; "
              "source frame numbers are not resampled frame numbers.")
    faces_left = load_faces(args.mano_faces_left_npy, args.mano_left_pkl, "left")
    faces_right = load_faces(args.mano_faces_right_npy, args.mano_right_pkl, "right")

    if args.mano_global_orient_frame == "vggt_camera":
        R_vggtcam_haworcam = np.eye(3, dtype=np.float32)
    elif args.camera_frame_rotation_npy is not None:
        R_vggtcam_haworcam = validate_rotation_matrix(
            np.load(args.camera_frame_rotation_npy),
            "camera-frame rotation",
        )
        print(f"Loaded HaWoR-camera -> VGGT-camera rotation from {args.camera_frame_rotation_npy}")
    else:
        R_vggtcam_haworcam = estimate_hawor_to_vggt_camera_rotation(
            records,
            min_soft_alpha=max(args.camera_rotation_min_soft_alpha, effective_min_soft_alpha_to_keep),
            soft_alpha_field=args.soft_alpha_field,
        )

    detections_left = {}
    detections_right = {}
    for frame_id, info in timestamp_map.items():
        detections_left[info["timestamp_ns"]] = None
        detections_right[info["timestamp_ns"]] = None

    debug_f = None
    if args.debug_camera_jsonl is not None:
        args.debug_camera_jsonl.parent.mkdir(parents=True, exist_ok=True)
        debug_f = open(args.debug_camera_jsonl, "w", encoding="utf-8")

    num_hands = 0
    num_missing_mano = 0
    num_skipped_missing_mano = 0
    num_placeholder_mano = 0
    num_wrist_only = 0
    num_full_mano = 0
    num_skipped_low_alpha = 0
    soft_alpha_values = []
    keypoint_consistency_errors = []
    for rec in records:
        frame_id = int(rec["frame_id"])
        if frame_id not in timestamp_map:
            raise KeyError(f"frame_id {frame_id} not found in timestamps table")
        src_frame_id = int(rec.get("src_frame_id", timestamp_map[frame_id]["src_frame_id"]))
        pose_index = frame_id if args.pose_index_field == "frame_id" else src_frame_id
        if pose_index >= len(poses):
            raise IndexError(
                f"{args.pose_index_field} {pose_index} out of range for {len(poses)} camera poses"
            )

        T_world_cam = poses[pose_index]
        timestamp_ns = int(timestamp_map[frame_id]["timestamp_ns"])
        debug_hands = []

        for hand_idx, hand in enumerate(rec.get("hands", [])):
            side = str(hand["side"]).lower()
            context = f"frame_id={frame_id}, hand={hand_idx}, side={side}"
            soft_alpha = pick_hand_soft_alpha(
                hand,
                requested_key=args.soft_alpha_field,
                default=1.0,
            )
            soft_alpha_values.append(soft_alpha)
            if soft_alpha < effective_min_soft_alpha_to_keep:
                num_skipped_low_alpha += 1
                if debug_f is not None:
                    debug_hands.append(
                        {
                            "side": side,
                            "src_frame_id": src_frame_id,
                            "soft_alpha": soft_alpha,
                            "skipped": True,
                            "skip_reason": "low_soft_alpha",
                        }
                    )
                continue

            keypoints_cam_input = require_points(hand, "camera_keypoints_3d", context)
            keypoints_world = require_points(hand, "world_keypoints_3d", context)
            verts_world = require_points(hand, "world_mesh_vertices_3d", context)

            # World points are the canonical upstream geometry. Reproject them
            # together so keypoints and mesh cannot drift into different frames.
            keypoints_cam = world_points_to_camera(
                keypoints_world,
                T_world_cam,
            ).astype(np.float32)
            verts_cam = world_points_to_camera(verts_world, T_world_cam).astype(np.float32)

            err = np.linalg.norm(keypoints_cam - keypoints_cam_input, axis=1)
            keypoint_consistency_errors.extend(err.tolist())

            selection_args = dict(
                keypoints_cam=hand.get("hawor_camera_keypoints_3d"),
                mano_path=args.mano_left_pkl if side == "left" else args.mano_right_pkl)
            if mano_timeline is None:
                mano = None
            else:
                mano = select_mano_at_timestamp(
                    mano_map, side, timestamp_ns, mano_timeline, **selection_args,
                    R_vggtcam_haworcam=R_vggtcam_haworcam,
                    R_world_target_cam=T_world_cam[:3, :3])
            has_mano = mano is not None
            constraint_mode = "full"
            if mano is None:
                num_missing_mano += 1
                if args.skip_missing_mano:
                    num_skipped_missing_mano += 1
                    if debug_f is not None:
                        debug_hands.append(
                            {
                                "side": side,
                                "src_frame_id": src_frame_id,
                                "soft_alpha": soft_alpha,
                                "skipped": True,
                                "skip_reason": "missing_mano",
                            }
                        )
                    continue

                allow_interpolated_placeholder = (
                    args.allow_placeholder_mano_for_interpolated
                    and bool(hand.get("interpolated", False))
                )
                if args.require_mano and not allow_interpolated_placeholder:
                    raise KeyError(
                        f"Missing MANO params for side={side}, timestamp_ns={timestamp_ns}. "
                        "Pass matching cam-space MANO dirs and their original camera timeline, "
                        "or disable --require-mano to downgrade valid wrist detections to wrist-only constraints."
                    )
                placeholder_keypoints = optional_points(
                    hand,
                    "hawor_camera_keypoints_3d",
                )
                if placeholder_keypoints is not None:
                    placeholder_keypoints = (
                        R_vggtcam_haworcam @ placeholder_keypoints.T
                    ).T.astype(np.float32)
                else:
                    placeholder_keypoints = keypoints_cam
                mano = make_placeholder_mano(placeholder_keypoints, side)
                num_placeholder_mano += 1
                num_wrist_only += 1
                constraint_mode = "wrist_only"
                mano_global_orient = mano["mano_hand_global_orient"]
            else:
                num_full_mano += 1
                mano_global_orient = (
                    R_vggtcam_haworcam @ mano["mano_hand_global_orient"]
                ).astype(np.float32)

            det = {
                "verts": verts_cam[None].astype(np.float32),
                "keypoints_3d": keypoints_cam[None].astype(np.float32),
                "mano_hand_pose": mano["mano_hand_pose"][None].astype(np.float32),
                "mano_hand_betas": mano["mano_hand_betas"][None].astype(np.float32),
                "mano_hand_global_orient": mano_global_orient[None].astype(np.float32),
            }
            det = add_soft_weight_fields(det, soft_alpha, hand, constraint_mode=constraint_mode)

            if side == "left":
                detections_left[timestamp_ns] = det
            elif side == "right":
                detections_right[timestamp_ns] = det
            else:
                raise ValueError(f"Bad side: {side}")

            num_hands += 1
            if debug_f is not None:
                debug_hands.append(
                    {
                        "side": side,
                        "src_frame_id": src_frame_id,
                        "soft_alpha": soft_alpha,
                        "constraint_weight": hand.get("constraint_weight"),
                        "visualization_alpha": hand.get("visualization_alpha"),
                        "camera_keypoints_3d": keypoints_cam.tolist(),
                        "camera_keypoints_input_3d": keypoints_cam_input.tolist(),
                        "camera_keypoints_from_world_3d": keypoints_cam.tolist(),
                        "camera_mesh_vertices_3d": verts_cam.tolist(),
                        "has_mano": has_mano,
                        "constraint_mode": constraint_mode,
                    }
                )

        if debug_f is not None:
            debug_f.write(
                json.dumps(
                    {
                        "frame_id": frame_id,
                        "src_frame_id": src_frame_id,
                        "pose_index": pose_index,
                        "timestamp": float(timestamp_map[frame_id]["timestamp"]),
                        "timestamp_ns": timestamp_ns,
                        "hands": debug_hands,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if debug_f is not None:
        debug_f.close()

    if keypoint_consistency_errors:
        errors = np.asarray(keypoint_consistency_errors, dtype=np.float64)
        print(
            "VGGT world/camera keypoint consistency: "
            f"mean={errors.mean():.6g}, max={errors.max():.6g}"
        )
        if errors.max() > 1e-3:
            print(
                "[Warning] Input camera_keypoints_3d and world_keypoints_3d transformed by "
                "camera_poses.txt are not identical. The output uses the world-derived "
                "camera keypoints so its keypoints and mesh stay spatially consistent."
            )

    outputs = {
        "mano_faces_right": faces_right,
        "mano_faces_left": faces_left,
        "detections_right_wrt_cam": detections_right,
        "detections_left_wrt_cam": detections_left,
        "T_device_cam": np.asarray(args.T_cpf_cam, dtype=np.float32),
        "T_cpf_cam": np.asarray(args.T_cpf_cam, dtype=np.float32),
    }
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(outputs, f)

    print(f"Saved: {args.output_pkl}")
    print(f"Frames in timestamps: {len(timestamp_map)}")
    print(f"Hands converted: {num_hands}")
    print(f"Hands skipped by soft alpha: {num_skipped_low_alpha} (threshold={effective_min_soft_alpha_to_keep:.3f})")
    if soft_alpha_values:
        alpha_arr = np.asarray(soft_alpha_values, dtype=np.float64)
        print(
            "Soft alpha stats: "
            f"min={alpha_arr.min():.3f}, "
            f"p10={np.percentile(alpha_arr, 10):.3f}, "
            f"p50={np.percentile(alpha_arr, 50):.3f}, "
            f"p90={np.percentile(alpha_arr, 90):.3f}, "
            f"max={alpha_arr.max():.3f}"
        )
    print(f"MANO records loaded: {sum(len(candidates) for candidates in mano_map.values())}")
    print(f"Full MANO constraints: {num_full_mano}")
    print(f"Wrist-only constraints: {num_wrist_only}")
    print(f"Hands missing MANO: {num_missing_mano}")
    print(f"Hands skipped by missing MANO: {num_skipped_missing_mano}")
    print(f"Hands using placeholder MANO: {num_placeholder_mano}")
    if num_placeholder_mano:
        print(
            "[Warning] Missing MANO params were filled with placeholder MANO arrays, "
            "but their hand_pose_weight and hand_reproj_weight are zero; only wrist_pose_weight is active."
        )
    if args.debug_camera_jsonl is not None:
        print(f"Saved debug camera-frame hands: {args.debug_camera_jsonl}")


if __name__ == "__main__":
    main()
