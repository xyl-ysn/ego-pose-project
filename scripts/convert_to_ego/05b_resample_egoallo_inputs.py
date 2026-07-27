#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resample EgoAllo inputs to target frame rate.

This script is inserted between:
  05_apply_metric_scale.py
  05b_resample_egoallo_inputs.py (this script)
  05_run_egoallo.sh

It keeps the official EgoAllo pkl schema unchanged and rewrites these files in-place
(via atomic replacement):

  * egoallo_inputs/head_trajectory.npy
  * egoallo_inputs/timestamps.txt
  * egoallo_inputs/hamer_outputs.pkl

It also writes temporal resampling metadata and mapping logs.
"""

from __future__ import annotations
import os
import sys

import argparse
import copy
import json
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None
try:
    import smplx
except Exception:  # pragma: no cover - optional dependency
    smplx = None


STATUS_OBSERVED = "observed"
STATUS_INTERPOLATED = "interpolated"
STATUS_MISSING = "missing"

HAND_TRACK_PREDICTIONS = (
    "mano_hand_global_orient",
    "mano_hand_pose",
    "mano_hand_betas",
)
HAND_META_STATUS_FIELDS = (
    "is_observed",
    "is_interpolated",
    "interpolation_method",
    "left_source_index",
    "right_source_index",
    "left_source_frame_id",
    "right_source_frame_id",
    "alpha",
    "gap_sec",
    "frame_gap_sec",
)


def _cfg_float(cfg: Any, key: str, default: float = 0.0) -> float:
    value = _cfg_get(cfg, key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _cfg_int(cfg: Any, key: str, default: int = 0) -> int:
    value = _cfg_get(cfg, key, default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _cfg_get(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a YAML mapping: {path}")
    return data


def _resolve_output_path(root: Path, value: Any, default: str) -> Path:
    p = Path(str(default if value is None else value))
    if not p.is_absolute():
        p = root / p
    return p


def _load_timestamps_txt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")
    if table.size == 0:
        raise ValueError(f"Empty timestamps file: {path}")
    if table.shape == ():
        table = np.asarray([table])

    required = {"frame_id", "timestamp", "src_frame_id"}
    names = set(table.dtype.names or [])
    if not required.issubset(names):
        raise ValueError(f"{path} must have columns: frame_id timestamp src_frame_id")

    return (
        np.asarray(table["timestamp"], dtype=np.float64),
        np.asarray(table["frame_id"], dtype=np.int64),
        np.asarray(table["src_frame_id"], dtype=np.int64),
    )


def _write_timestamps_txt(path: Path, timestamps: np.ndarray, frame_ids: np.ndarray, src_frame_ids: np.ndarray) -> None:
    arr = np.column_stack(
        [
            np.asarray(frame_ids, dtype=np.int64),
            np.asarray(timestamps, dtype=np.float64),
            np.asarray(src_frame_ids, dtype=np.int64),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, fmt=["%d", "%.9f", "%d"], header="frame_id timestamp src_frame_id", comments="")


def _load_head(path: Path) -> tuple[np.ndarray, np.dtype]:
    arr = np.asarray(np.load(path), dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (4, 4):
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"{path} must be [N,4,4] or [4,4], got {arr.shape}")
    return arr.astype(np.float64), arr.dtype


def _load_hamer(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a dict payload")
    required = {
        "detections_left_wrt_cam",
        "detections_right_wrt_cam",
        "mano_faces_left",
        "mano_faces_right",
        "T_cpf_cam",
        "T_device_cam",
    }
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"{path} missing required keys: {sorted(missing)}")
    for k in ("detections_left_wrt_cam", "detections_right_wrt_cam"):
        if not isinstance(data[k], dict):
            raise ValueError(f"{path}:{k} must be dict[nanoseconds]->detection")
    return data


def _atomic_replace(path: Path, write_fn) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=path.suffix, dir=str(path.parent))
    tmp_path = Path(tmp_name)
    os.close(fd)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _safe_copy(src: Path, dst: Path) -> None:
    _atomic_replace(dst, lambda p: shutil.copy2(src, p))


def _load_pchip():
    try:
        from scipy.interpolate import PchipInterpolator

        return PchipInterpolator
    except Exception:
        return None


def _normalize_timestamps(ts: np.ndarray) -> None:
    if ts.size <= 1:
        return
    if np.any(~np.isfinite(ts)):
        raise ValueError("timestamps contain non-finite values")
    diffs = np.diff(ts)
    if np.any(diffs <= 0):
        bad = int(np.where(diffs <= 0)[0][0])
        raise ValueError(f"timestamps must be strictly increasing; bad index={bad}")


def _generate_target_timestamps_by_src_frames(
    source_timestamps: np.ndarray,
    source_src_frame_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if source_timestamps.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
        )
    if source_timestamps.size == 1:
        return (
            np.array([int(source_src_frame_ids[0])], dtype=np.int64),
            source_timestamps.astype(np.float64).copy(),
        )

    src_ids = np.asarray(source_src_frame_ids, dtype=np.int64).astype(np.float64)
    if np.any(np.diff(src_ids) < 0):
        bad = int(np.where(np.diff(src_ids) < 0)[0][0])
        raise ValueError(f"src_frame_id must be non-decreasing; bad index={bad}")

    # 压缩重复 src_frame_id（若存在），避免插值报错。
    unique_mask = np.r_[True, np.diff(src_ids) > 0]
    src_ids_u = src_ids[unique_mask]
    source_timestamps_u = np.asarray(source_timestamps, dtype=np.float64)[unique_mask]
    if src_ids_u.size == 1:
        return (
            np.array([int(src_ids_u[0])], dtype=np.int64),
            np.array([float(source_timestamps_u[0])], dtype=np.float64),
        )

    start_id = int(src_ids_u[0])
    end_id = int(src_ids_u[-1])
    if end_id < start_id:
        raise ValueError("Invalid src_frame_id range for resampling")

    target_src_ids = np.arange(start_id, end_id + 1, dtype=np.int64)
    target_timestamps = np.interp(
        target_src_ids.astype(np.float64),
        src_ids_u,
        source_timestamps_u,
    ).astype(np.float64)
    return target_src_ids, target_timestamps


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got {R.shape}")
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] >= R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm([w, x, y, z])
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = np.array([w, x, y, z], dtype=np.float64) / n
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz = x * y, x * z
    yz = y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _ensure_contiguous_quats(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    if q.ndim == 1:
        q = q[None, :]
    if q.shape[-1] != 4:
        raise ValueError(f"Quaternions must have shape [...,4], got {q.shape}")
    if q.size == 0:
        return q
    q[0] = _quat_normalize(q[0])
    for i in range(1, q.shape[0]):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] = -q[i]
        q[i] = _quat_normalize(q[i])
    return q


def _ensure_contiguous_quat_sequence(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    if q.ndim == 2 and q.shape[-1] == 4:
        if q.shape[0] == 0:
            return q
        q[0] = _quat_normalize(q[0])
        for i in range(1, q.shape[0]):
            if float(np.dot(q[i - 1], q[i])) < 0.0:
                q[i] = -q[i]
            q[i] = _quat_normalize(q[i])
        return q
    if q.ndim != 3 or q.shape[-1] != 4:
        raise ValueError(f"Quaternion sequence must have shape [T,J,4], got {q.shape}")
    for j in range(q.shape[1]):
        q[:, j, :] = _ensure_contiguous_quats(q[:, j, :])
    return q


def _axis_angle_to_quat(v: np.ndarray) -> np.ndarray:
    aa = np.asarray(v, dtype=np.float64).reshape(-1)
    if aa.shape != (3,):
        raise ValueError(f"Axis-angle vector expected 3, got {aa.shape}")
    theta = np.linalg.norm(aa)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = aa / theta
    h = 0.5 * theta
    s = np.sin(h)
    return _quat_normalize(np.array([np.cos(h), *(axis * s)], dtype=np.float64))


def _quat_from_mano_pose(pose: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(pose, dtype=np.float64))
    if arr.shape == (15, 3, 3):
        return np.array([_quat_from_matrix(x) for x in arr], dtype=np.float64)
    if arr.shape == (15, 3):
        return np.array([_axis_angle_to_quat(v) for v in arr], dtype=np.float64)
    if arr.shape == (15, 4):
        return np.array([_quat_normalize(v) for v in arr], dtype=np.float64)
    if arr.shape == (4, 15):
        return np.array([_quat_normalize(v) for v in arr.T], dtype=np.float64)
    if arr.shape == (1, 15, 4):
        return np.array([_quat_normalize(v) for v in arr[0]], dtype=np.float64)
    if arr.shape == (45,):
        return np.array([_axis_angle_to_quat(v) for v in arr.reshape(15, 3)], dtype=np.float64)
    raise ValueError(f"Unsupported mano pose shape: {arr.shape}")


def _axis_angle_from_quat(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(np.asarray(q, dtype=np.float64).reshape(-1))
    if q.shape != (4,):
        raise ValueError(f"Quaternion shape must be (4,), got {q.shape}")
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 1e-12))
    if s < 1e-12 or angle < 1e-12:
        return np.zeros(3, dtype=np.float64)
    xyz = q[1:] / s
    return xyz * angle


def _axis_angle_from_matrix(R: np.ndarray) -> np.ndarray:
    return _axis_angle_from_quat(_quat_from_matrix(R))


def _axis_angle_to_matrix(v: np.ndarray) -> np.ndarray:
    aa = np.asarray(v, dtype=np.float64).reshape(-1)
    if aa.shape != (3,):
        raise ValueError(f"Axis-angle shape must be (3,), got {aa.shape}")
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = aa / theta
    x, y, z = axis.tolist()
    h = 0.5 * theta
    s = np.sin(h)
    c = np.cos(h)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _pose_to_axis_angles(pose: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(pose, dtype=np.float64))
    if arr.shape == (15, 3, 3):
        return np.array([_axis_angle_from_matrix(x) for x in arr], dtype=np.float64)
    if arr.shape == (15, 3):
        return np.asarray(arr, dtype=np.float64)
    if arr.shape == (45,):
        return np.asarray(arr, dtype=np.float64).reshape(15, 3)
    if arr.shape == (1, 45):
        return np.asarray(arr, dtype=np.float64).reshape(15, 3)
    if arr.shape == (15, 4):
        return np.array([_axis_angle_from_quat(v) for v in arr], dtype=np.float64)
    if arr.shape == (4, 15):
        return np.array([_axis_angle_from_quat(v) for v in arr.T], dtype=np.float64)
    raise ValueError(f"Unsupported mano pose shape: {arr.shape}")


def _get_dict_value(obj: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            return obj.get(key)
    return default


def _to_float_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).any():
            return None
        return arr
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 0 or not np.isfinite(arr[0]):
            return float(default)
        return float(arr[0])
    except Exception:
        return float(default)


def _select_padded_float_array(
    target: np.ndarray | None,
    fallback: np.ndarray | None,
) -> np.ndarray | None:
    if target is not None:
        return target
    return None if fallback is None else fallback.copy()


def _get_confidence(rec: dict[str, Any] | None) -> float:
    if rec is None:
        return 1.0
    for key in ("confidence", "score", "hand_confidence", "pose_confidence"):
        v = _to_float_array(rec.get(key) if isinstance(rec, dict) else None)
        if v is not None and v.size > 0 and np.isfinite(v[0]):
            return float(np.clip(v[0], 0.0, 1.0))
    return 1.0


def _track_id_from_record(rec: dict[str, Any] | None) -> str | None:
    if rec is None:
        return None
    for k in ("track_id", "track", "hand_track_id"):
        v = rec.get(k)
        if v is None:
            continue
        try:
            return str(v)
        except Exception:
            return None
    return None


def _handedness_from_record(rec: dict[str, Any] | None) -> str | None:
    if rec is None:
        return None
    for k in ("handedness", "side", "is_right_hand", "is_right"):
        if k not in rec:
            continue
        if k in {"is_right_hand", "is_right"}:
            try:
                return "right" if bool(rec[k]) else "left"
            except Exception:
                continue
        return str(rec[k]).lower()
    return None


def _keypoints_from_record(rec: dict[str, Any] | None) -> np.ndarray | None:
    if rec is None:
        return None
    arr = _to_float_array(rec.get("keypoints_3d"))
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 3:
        return arr
    return None


def _normalize_hand_detection_record(
    rec: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if rec is None:
        return None
    if not isinstance(rec, dict):
        return None

    out = copy.deepcopy(rec)

    verts = _to_float_array(out.get("verts"))
    keypoints = _to_float_array(out.get("keypoints_3d"))
    orient = _to_float_array(out.get("mano_hand_global_orient"))
    pose = _to_float_array(out.get("mano_hand_pose"))
    betas = _to_float_array(out.get("mano_hand_betas"))

    if verts is not None and verts.ndim == 2 and verts.shape[1] == 3:
        verts = verts.reshape(1, verts.shape[0], 3)
    if keypoints is not None and keypoints.ndim == 2 and keypoints.shape[1] == 3:
        keypoints = keypoints.reshape(1, keypoints.shape[0], 3)
    if orient is not None and orient.ndim == 2 and orient.shape == (3, 3):
        orient = orient.reshape(1, 3, 3)

    if (
        verts is not None
        and keypoints is not None
        and verts.ndim == 3
        and keypoints.ndim == 3
        and verts.shape[0] > 0
        and keypoints.shape[0] > 0
        and verts.shape[2] == 3
        and keypoints.shape[2] == 3
    ):
        n = min(int(verts.shape[0]), int(keypoints.shape[0]))
        if n > 0:
            verts = verts[:n]
            keypoints = keypoints[:n]

    if verts is not None:
        out["verts"] = np.asarray(verts, dtype=np.float32)
    if keypoints is not None:
        out["keypoints_3d"] = np.asarray(keypoints, dtype=np.float32)
    if orient is not None and orient.size > 0:
        if orient.ndim == 3 and orient.shape[-1] == 3 and orient.shape[-2] == 3:
            out["mano_hand_global_orient"] = np.asarray(orient, dtype=np.float32)
    if pose is not None and pose.size > 0:
        out["mano_hand_pose"] = np.asarray(pose, dtype=np.float32)
    if betas is not None and betas.size > 0:
        out["mano_hand_betas"] = np.asarray(betas, dtype=np.float32)

    return out


def _wrist_position_from_record(rec: dict[str, Any] | None, fallback_from_keypoints: bool = True) -> np.ndarray | None:
    if rec is None:
        return None
    trans = _extract_translation(rec, fallback_from_wrist=fallback_from_keypoints)
    if trans is not None and trans.size == 3 and np.isfinite(trans).all():
        return trans.reshape(3)
    kps = _keypoints_from_record(rec)
    if kps is not None and kps.size > 0 and kps.shape[1] > 0 and np.isfinite(kps[0, 0]).all():
        return np.asarray(kps[0, 0], dtype=np.float64).reshape(3)
    return None


def _palm_lengths(keypoints: np.ndarray) -> np.ndarray | None:
    if keypoints.ndim != 3 or keypoints.shape[1] <= 17 or keypoints.shape[2] != 3:
        return None
    p = keypoints[0]
    if p.shape[0] <= 17:
        return None
    idx_pairs = ((0, 9), (5, 17))
    lens = []
    for i, j in idx_pairs:
        a = p[i]
        b = p[j]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return None
        lens.append(float(np.linalg.norm(b - a)))
    return np.asarray(lens, dtype=np.float64)


def _palm_ratio(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 1.0
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return 1.0
    if np.any(a <= 0.0) or np.any(b <= 0.0):
        return 1.0
    r = np.abs(a - b) / np.clip(np.maximum(a, b), 1e-6, np.inf)
    return float(np.max(r))


def _pad_or_clip_keypoints(
    keypoints: np.ndarray,
    target_shape: tuple[int, int, int] | None,
) -> np.ndarray:
    if target_shape is None:
        return keypoints.astype(np.float32).copy()
    out = np.zeros(target_shape, dtype=np.float64)
    if keypoints.ndim != 3 or keypoints.shape[2] != 3:
        return out.astype(np.float32)
    n_src = keypoints.shape[1]
    n_tgt = target_shape[1]
    n = min(n_src, n_tgt)
    out[:, :n, :] = keypoints[:, :n, :]
    if n_tgt > n:
        out[:, n:, :] = keypoints[:, -1:, :][:, : (n_tgt - n), :]
    return out.astype(np.float32)


def _interpolated_confidence(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    alpha: float,
    gap_sec: float,
    *,
    max_gap: float,
    mid_strength: float,
    base_scale: float,
    motion_quality: float = 1.0,
    enable_gap_decay: bool = True,
    enable_motion_decay: bool = True,
) -> float:
    cl = _get_confidence(left)
    cr = _get_confidence(right)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    endpoint = float(np.power(cl, 1.0 - alpha) * np.power(cr, alpha))
    if enable_gap_decay:
        gap_quality = float(np.exp(-((gap_sec / max(max_gap, 1e-9)) ** 2)))
    else:
        gap_quality = 1.0
    midpoint_penalty = float(1.0 - mid_strength * 4.0 * alpha * (1.0 - alpha))
    if enable_motion_decay:
        motion_q = float(np.clip(motion_quality, 0.0, 1.0))
    else:
        motion_q = 1.0
    return float(np.clip(endpoint * gap_quality * midpoint_penalty * motion_q * base_scale, 0.0, 1.0))


def _compute_motion_quality(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    gap_sec: float,
) -> float:
    if left is None or right is None:
        return 1.0
    v_left = _wrist_position_from_record(left, fallback_from_keypoints=True)
    v_right = _wrist_position_from_record(right, fallback_from_keypoints=True)
    if (
        v_left is None
        or v_right is None
        or gap_sec <= 0
        or not np.isfinite(v_left).all()
        or not np.isfinite(v_right).all()
    ):
        return 1.0
    speed = float(np.linalg.norm(v_right - v_left) / max(float(gap_sec), 1e-9))
    speed_q = float(np.exp(-0.1 * speed))
    q_left = _quat_from_matrix(np.asarray(left["mano_hand_global_orient"], dtype=np.float64).reshape(3, 3))
    q_right = _quat_from_matrix(np.asarray(right["mano_hand_global_orient"], dtype=np.float64).reshape(3, 3))
    if q_left.shape != (4,) or q_right.shape != (4,):
        return speed_q
    rot_deg = float(_quat_distance_deg(q_left, q_right) / max(float(gap_sec), 1e-9))
    rot_q = float(np.exp(-0.01 * rot_deg))
    l_kp = _keypoints_from_record(left)
    r_kp = _keypoints_from_record(right)
    ratio = _palm_ratio(_palm_lengths(l_kp), _palm_lengths(r_kp))
    palm_q = float(np.exp(-2.0 * ratio))
    return float(np.clip(speed_q * rot_q * palm_q, 0.0, 1.0))


def _resolve_path(cfg_root: Path, rel_or_abs: Any) -> Path:
    if rel_or_abs is None:
        return Path("")
    p = Path(str(rel_or_abs))
    if p.is_absolute():
        return p
    candidates = [cfg_root, cfg_root.parent]
    for c in candidates:
        cp = c / p
        if cp.exists():
            return cp
    return cfg_root.parent / p


@dataclass
class _ManoLayerRuntime:
    side: str
    layer: Any
    device: str


def _load_mano_layers(
    left_model: Path | None,
    right_model: Path | None,
    *,
    device: str,
) -> dict[str, _ManoLayerRuntime]:
    if smplx is None or torch is None:
        raise RuntimeError("smplx and torch are required for hand resampling MANO forward")
    runtimes: dict[str, _ManoLayerRuntime] = {}
    model_map = {"left": left_model, "right": right_model}
    for side, path in model_map.items():
        if path is None:
            continue
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"MANO model path not found for {side}: {p}")
        runtimes[side] = _ManoLayerRuntime(
            side=side,
            layer=smplx.create(
                model_path=str(p),
                model_type="mano",
                is_rhand=(side == "right"),
                use_pca=False,
                flat_hand_mean=False,
            ).to(device),
            device=device,
        )
    return runtimes


def _resolve_mano_runtimes(
    cfg_root: Path,
    cfg: dict[str, Any],
    *,
    device: str,
) -> dict[str, _ManoLayerRuntime]:
    model_cfg = _cfg_get(cfg, "model_weights", {}) or {}
    left_model = _resolve_path(cfg_root, _cfg_get(model_cfg, "mano_left_pkl", None))
    if str(left_model) == ".":
        left_model = None
    right_model = _resolve_path(cfg_root, _cfg_get(model_cfg, "mano_right_pkl", None))
    if str(right_model) == ".":
        right_model = None
    if (left_model is None or not left_model.exists()) and (right_model is None or not right_model.exists()):
        return {}
    return _load_mano_layers(left_model, right_model, device=device)


def _mano_forward_state(
    runtime: _ManoLayerRuntime | None,
    *,
    global_orient: np.ndarray,
    pose: np.ndarray,
    betas: np.ndarray,
    trans: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if runtime is None:
        return None
    if global_orient.shape != (3, 3):
        raise ValueError(f"global_orient must be (3,3), got {global_orient.shape}")
    if trans is None:
        trans = np.zeros(3, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64).reshape(3)
    try:
        aa_orient = _axis_angle_from_matrix(np.asarray(global_orient, dtype=np.float64))
        aa_pose = _pose_to_axis_angles(np.asarray(pose, dtype=np.float64)).reshape(45)
        bb = np.asarray(betas, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if bb.size < 10:
        return None
    with torch.no_grad():
        out = runtime.layer(
            global_orient=torch.from_numpy(aa_orient.reshape(1, 3)).float().to(runtime.device),
            hand_pose=torch.from_numpy(aa_pose.reshape(1, 45)).float().to(runtime.device),
            betas=torch.from_numpy(bb[:10].reshape(1, 10)).float().to(runtime.device),
            transl=torch.from_numpy(trans.reshape(1, 3)).float().to(runtime.device),
            return_verts=True,
        )
    v = out.vertices[0].cpu().numpy().astype(np.float64)
    j = out.joints[0].cpu().numpy().astype(np.float64)
    return v, j


def _extract_translation(
    rec: dict[str, Any] | None,
    *,
    fallback_from_wrist: bool,
) -> np.ndarray | None:
    trans, _ = _extract_translation_with_source(rec, fallback_from_wrist=fallback_from_wrist)
    return trans


def _extract_translation_with_source(
    rec: dict[str, Any] | None,
    *,
    fallback_from_wrist: bool,
) -> tuple[np.ndarray | None, str | None]:
    if rec is None:
        return None, None
    for key in ("hand_translation", "wrist_translation", "root_translation", "translation"):
        if key in rec:
            v = rec[key]
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.size == 3 and np.all(np.isfinite(arr)):
                return arr.astype(np.float64), key
    if not fallback_from_wrist:
        return None, None
    kps = np.asarray(rec.get("keypoints_3d", []), dtype=np.float64)
    if kps.ndim == 3 and kps.shape[0] >= 1 and kps.shape[1] >= 1 and kps.shape[-1] == 3:
        if np.all(np.isfinite(kps[0, 0])):
            return kps[0, 0].astype(np.float64), "wrist_keypoint"
    return None, None


def _pose_translation_gap(
    frame_a: dict[str, Any],
    frame_b: dict[str, Any],
    *,
    dt: float,
    max_speed_mps: float,
) -> bool:
    if dt <= 0.0:
        return False
    p0 = _wrist_position_from_record(frame_a)
    p1 = _wrist_position_from_record(frame_b)
    if p0 is None or p1 is None or p0.size != 3 or p1.size != 3:
        return False
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        return False
    speed = float(np.linalg.norm(p1 - p0) / dt)
    return speed > max_speed_mps


def _quat_distance_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = _quat_normalize(np.asarray(q0, dtype=np.float64))
    q1 = _quat_normalize(np.asarray(q1, dtype=np.float64))
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    c = np.clip(float(np.dot(q0, q1)), -1.0, 1.0)
    return float(2.0 * np.degrees(np.arccos(c)))


def _extract_mano_state(
    rec: dict[str, Any] | None,
    *,
    fallback_from_wrist: bool = True,
) -> dict[str, Any] | None:
    if rec is None:
        return None
    orient = _to_float_array(_get_dict_value(rec, "mano_hand_global_orient"))
    pose = _to_float_array(_get_dict_value(rec, "mano_hand_pose"))
    betas = _to_float_array(_get_dict_value(rec, "mano_hand_betas"))
    if orient is None or pose is None or betas is None:
        return None
    try:
        orient_m = np.asarray(orient, dtype=np.float64).reshape(3, 3)
    except Exception:
        return None
    try:
        pose_aa = _pose_to_axis_angles(pose)
    except Exception:
        return None
    trans = _extract_translation(rec, fallback_from_wrist=fallback_from_wrist)
    if trans is None:
        return None
    return {
        "mano_pose_shape": tuple(np.asarray(rec["mano_hand_pose"]).shape),
        "mano_hand_global_orient": orient_m.astype(np.float64),
        "mano_hand_pose": pose_aa.astype(np.float64),
        "mano_hand_betas": np.asarray(betas, dtype=np.float64).reshape(-1),
        "translation": trans.astype(np.float64),
        "verts": _to_float_array(_get_dict_value(rec, "verts")),
        "keypoints_3d": _keypoints_from_record(rec),
    }


def _build_track_ids(
    obs_records: list[dict[str, Any] | None],
    obs_times: np.ndarray,
    *,
    max_gap_sec: float,
    split_on_track_id_change: bool,
    split_on_handedness_change: bool,
    max_wrist_speed_mps: float,
    max_global_angular_speed_deg_s: float,
    max_palm_bone_change_ratio: float,
    max_beta_difference: float,
    translation_fallback_from_wrist: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    n = len(obs_records)
    if n == 0:
        return np.zeros((0,), dtype=np.int64), {}

    track_ids = [0]
    reason_counts: dict[str, int] = defaultdict(int)

    for i in range(1, n):
        prev_rec = obs_records[i - 1]
        cur_rec = obs_records[i]
        dt = float(obs_times[i] - obs_times[i - 1])
        reasons: list[str] = []

        if dt > max_gap_sec:
            reasons.append("gap")

        if split_on_track_id_change and _track_id_from_record(prev_rec) != _track_id_from_record(cur_rec):
            if _track_id_from_record(prev_rec) is not None or _track_id_from_record(cur_rec) is not None:
                reasons.append("track_id")

        if split_on_handedness_change and _handedness_from_record(prev_rec) != _handedness_from_record(cur_rec):
            if _handedness_from_record(prev_rec) is not None or _handedness_from_record(cur_rec) is not None:
                reasons.append("handedness")

        if dt > 0.0:
            p0 = _wrist_position_from_record(prev_rec, fallback_from_keypoints=translation_fallback_from_wrist)
            p1 = _wrist_position_from_record(cur_rec, fallback_from_keypoints=translation_fallback_from_wrist)
            if p0 is not None and p1 is not None and np.isfinite(p0).all() and np.isfinite(p1).all():
                speed = float(np.linalg.norm(p1 - p0) / dt)
                if speed > max_wrist_speed_mps:
                    reasons.append("wrist_speed")

            q0 = _quat_from_matrix(np.asarray(_get_dict_value(prev_rec, "mano_hand_global_orient"), dtype=np.float64).reshape(3, 3))
            q1 = _quat_from_matrix(np.asarray(_get_dict_value(cur_rec, "mano_hand_global_orient"), dtype=np.float64).reshape(3, 3))
            if q0.size == 4 and q1.size == 4:
                if (_quat_distance_deg(q0, q1) / max(dt, 1e-9)) > max_global_angular_speed_deg_s:
                    reasons.append("angular_speed")

            b0 = _to_float_array(_get_dict_value(prev_rec, "mano_hand_betas"))
            b1 = _to_float_array(_get_dict_value(cur_rec, "mano_hand_betas"))
            if b0 is not None and b1 is not None and b0.size == b1.size:
                if np.linalg.norm(b0 - b1) > max_beta_difference:
                    reasons.append("beta_jump")

            kp0 = _keypoints_from_record(prev_rec)
            kp1 = _keypoints_from_record(cur_rec)
            l0 = _palm_lengths(kp0 if kp0 is not None else np.empty((1, 0, 3)))
            l1 = _palm_lengths(kp1 if kp1 is not None else np.empty((1, 0, 3)))
            if l0 is not None and l1 is not None:
                if _palm_ratio(l0, l1) > max_palm_bone_change_ratio:
                    reasons.append("palm_bone_jump")

        if reasons:
            reason_counts["|".join(sorted(set(reasons)))] += 1
            track_ids.append(track_ids[-1] + 1)
        else:
            track_ids.append(track_ids[-1])

    return np.asarray(track_ids, dtype=np.int64), reason_counts


def _betas_for_tracks(
    obs_records: list[dict[str, Any] | None],
    obs_track_ids: np.ndarray,
    num_tracks: int,
    beta_shape: int,
    *,
    freeze_scope: str,
) -> tuple[np.ndarray, np.ndarray]:
    betas: list[np.ndarray | None] = []
    valid_mask: list[bool] = []
    for rec in obs_records:
        s = _extract_mano_state(rec)
        if s is None or s["mano_hand_betas"].size < beta_shape:
            valid_mask.append(False)
            betas.append(None)
            continue
        arr = s["mano_hand_betas"].astype(np.float64).reshape(-1)[:beta_shape]
        valid_mask.append(True)
        betas.append(arr)

    valid_mask_arr = np.asarray(valid_mask, dtype=bool)
    if num_tracks <= 0:
        return np.zeros((1, beta_shape), dtype=np.float64), valid_mask_arr
    if not valid_mask_arr.any():
        return np.zeros((num_tracks, beta_shape), dtype=np.float64), valid_mask_arr

    beta_mat = np.asarray(
        [b if b is not None else np.full((beta_shape,), np.nan) for b in betas],
        dtype=np.float64,
    )

    if freeze_scope == "per_track":
        out = []
        for t in range(num_tracks):
            m = obs_track_ids == t
            if (valid_mask_arr & m).any():
                out.append(np.nanmedian(beta_mat[valid_mask_arr & m], axis=0))
            else:
                out.append(np.nanmedian(beta_mat[valid_mask_arr], axis=0))
        return np.asarray(out, dtype=np.float64), valid_mask_arr

    if freeze_scope == "per_sequence_per_side":
        if not valid_mask_arr.any():
            return np.tile(np.zeros((beta_shape,), dtype=np.float64)[None, :], (num_tracks, 1)), valid_mask_arr
        sequence_betas = np.nanmedian(beta_mat[valid_mask_arr], axis=0)
        return np.tile(sequence_betas[None, :], (num_tracks, 1)), valid_mask_arr

    # 默认按序列侧冻结。
    global_betas = np.nanmedian(beta_mat[valid_mask_arr], axis=0)
    return np.tile(global_betas[None, :], (num_tracks, 1)), valid_mask_arr


def _restore_pose_shape(pose_aa: np.ndarray, target_shape: tuple[Any, ...] | list[int] | None) -> np.ndarray:
    pose_aa = np.asarray(pose_aa, dtype=np.float32)
    if target_shape is None:
        return pose_aa.astype(np.float32)
    target_shape = tuple(target_shape)
    if target_shape == (15, 3):
        return pose_aa.reshape(15, 3).astype(np.float32)
    if target_shape == (15, 3, 3):
        mats = np.array(
            [_axis_angle_to_matrix(v) for v in pose_aa.reshape(-1, 3)],
            dtype=np.float32,
        )
        return mats.reshape(15, 3, 3)
    if target_shape == (1, 15, 3, 3):
        mats = np.array(
            [_axis_angle_to_matrix(v) for v in pose_aa.reshape(-1, 3)],
            dtype=np.float32,
        )
        return mats.reshape(1, 15, 3, 3)
    if target_shape == (45,):
        return pose_aa.reshape(45).astype(np.float32)
    if target_shape == (1, 45):
        return pose_aa.reshape(1, 45).astype(np.float32)
    if target_shape == (1, 15, 3):
        return pose_aa.reshape(1, 15, 3).astype(np.float32)
    try:
        return pose_aa.reshape(target_shape).astype(np.float32)
    except Exception:
        return pose_aa.astype(np.float32)


def _resample_hand(
    source_timestamps: np.ndarray,
    source_records: list[dict[str, Any] | None],
    target_times: np.ndarray,
    *,
    side: str,
    mano_runtime: _ManoLayerRuntime | None,
    max_interpolation_gap_sec: float,
    split_on_track_id_change: bool,
    split_on_handedness_change: bool,
    max_wrist_speed_mps: float,
    max_global_angular_speed_deg_s: float,
    max_palm_bone_change_ratio: float,
    max_beta_difference: float,
    translation_method: str,
    interpolated_constraint_scale: float,
    interpolated_visualization_scale: float,
    beta_scope: str,
    geometry_generation: str,
    mano_missing_fallback: str,
    pose_confidence_scale: float,
    interp_midpoint_penalty_strength: float,
    gap_tau_sec: float,
    apply_gap_decay: bool,
    apply_motion_decay: bool,
    do_extrapolate: bool,
    timestamp_tolerance_sec: float,
    source_frame_ids: np.ndarray | None = None,
    template_keypoints_shape: tuple[int, int, int] | None = None,
) -> tuple[list[dict[str, Any] | None], dict[str, int], list[int | None], list[int | None], list[float | None], dict[str, Any]]:
    obs = [(i, rec) for i, rec in enumerate(source_records) if _has_required_hand_fields(rec)]
    status = [STATUS_MISSING] * target_times.size
    src_left = [None] * target_times.size
    src_right = [None] * target_times.size
    alpha = [None] * target_times.size

    if not obs:
        return [None] * target_times.size, {"observed": 0, "interpolated": 0, "missing": int(target_times.size)}, src_left, src_right, alpha, {
            "num_valid_source_detections": 0,
            "num_missing_source_records": int(len(source_records)),
            "track_count": 0,
            "track_sizes": [],
            "track_split_reasons": {},
            "mano_forward_count": 0,
            "mano_forward_fail_count": 0,
            "mano_fallback_count": 0,
            "max_palm_ratio": 0.0,
        }

    obs_indices = np.array([i for i, _ in obs], dtype=np.int64)
    obs_records = [r for _, r in obs]
    obs_times = source_timestamps[obs_indices]
    src_ids_arr = (
        np.asarray(source_frame_ids, dtype=np.int64)
        if source_frame_ids is not None and len(source_frame_ids) == len(source_records)
        else None
    )

    track_ids, split_reason_counts = _build_track_ids(
        obs_records,
        obs_times,
        max_gap_sec=max_interpolation_gap_sec,
        split_on_track_id_change=split_on_track_id_change,
        split_on_handedness_change=split_on_handedness_change,
        max_wrist_speed_mps=max_wrist_speed_mps,
        max_global_angular_speed_deg_s=max_global_angular_speed_deg_s,
        max_palm_bone_change_ratio=max_palm_bone_change_ratio,
        max_beta_difference=max_beta_difference,
        translation_fallback_from_wrist=True,
    )
    num_tracks = int(np.max(track_ids) + 1) if track_ids.size else 0

    mano_pose_shape = tuple(np.asarray(obs_records[0]["mano_hand_pose"]).shape)
    beta_shape = int(np.asarray(obs_records[0]["mano_hand_betas"]).size)
    track_betas, _ = _betas_for_tracks(
        obs_records,
        track_ids,
        num_tracks,
        beta_shape=beta_shape,
        freeze_scope=str(beta_scope),
    )

    obs_orient_quat = np.stack(
        [
            _quat_from_matrix(np.asarray(rec["mano_hand_global_orient"], dtype=np.float64).reshape(3, 3))
            for rec in obs_records
        ],
        axis=0,
    )
    obs_pose_quat = np.stack(
        [_quat_from_mano_pose(np.asarray(rec["mano_hand_pose"], dtype=np.float64)) for rec in obs_records],
        axis=0,
    )
    obs_orient_quat = _ensure_contiguous_quats(obs_orient_quat)
    obs_pose_quat = _ensure_contiguous_quat_sequence(obs_pose_quat)
    obs_trans_list: list[np.ndarray] = []
    obs_trans_source: list[str] = []
    for rec in obs_records:
        trans, trans_source = _extract_translation_with_source(rec, fallback_from_wrist=True)
        if trans is None:
            kp = _keypoints_from_record(rec)
            if kp is not None and kp.size > 0 and np.isfinite(kp[0, 0]).all():
                trans = np.asarray(kp[0, 0], dtype=np.float64)
                trans_source = "wrist_keypoint"
        if trans is None or not np.isfinite(trans).all():
            trans = np.zeros(3, dtype=np.float64)
            trans_source = "zero_fill"
        elif trans_source is None:
            trans_source = "unknown"
        obs_trans_list.append(np.asarray(trans, dtype=np.float64).reshape(3))
        obs_trans_source.append(str(trans_source))
    obs_trans = np.stack(obs_trans_list, axis=0)

    Pchip = _load_pchip()
    pchip_trans = None
    if translation_method == "pchip" and Pchip is not None and len(obs_records) >= 3:
        try:
            pchip_trans = [Pchip(obs_times, obs_trans[:, i]) for i in range(3)]
        except Exception:
            pchip_trans = None

    pos = np.searchsorted(obs_times, target_times, side="right") - 1
    out: list[dict[str, Any] | None] = [None] * target_times.size
    mano_forward_count = 0
    mano_forward_fail_count = 0
    mano_fallback_count = 0
    max_palm_ratio = 0.0

    for ti, t in enumerate(target_times):
        j = int(pos[ti])
        if j < 0:
            continue

        if abs(float(t) - float(obs_times[j])) <= float(timestamp_tolerance_sec):
            idx = int(obs_indices[j])
            out_item = copy.deepcopy(obs_records[j])
            src_frame_id = int(src_ids_arr[idx]) if src_ids_arr is not None else idx
            out_item["is_observed"] = True
            out_item["is_interpolated"] = False
            out_item["interpolation_method"] = "observed"
            out_item["left_source_index"] = int(idx)
            out_item["right_source_index"] = int(idx)
            out_item["frame_gap_sec"] = 0.0
            out_item["left_source_frame_id"] = src_frame_id
            out_item["right_source_frame_id"] = src_frame_id
            out_item["alpha"] = 0.0
            out_item["status"] = STATUS_OBSERVED
            out[ti] = out_item
            status[ti] = STATUS_OBSERVED
            src_left[ti] = idx
            src_right[ti] = idx
            alpha[ti] = 0.0
            continue

        if j + 1 >= len(obs_times):
            continue
        if not do_extrapolate and (float(target_times[ti]) < float(obs_times[j]) or float(target_times[ti]) > float(obs_times[j + 1])):
            continue

        gap = float(obs_times[j + 1] - obs_times[j])
        if gap <= 0 or gap > float(max_interpolation_gap_sec):
            continue

        a = (float(t) - float(obs_times[j])) / gap
        if a < 0.0 or a > 1.0:
            continue

        l_rec = obs_records[j]
        r_rec = obs_records[j + 1]
        if track_ids[j] != track_ids[j + 1]:
            continue

        left_idx = int(obs_indices[j])
        right_idx = int(obs_indices[j + 1])
        left_src_id = int(src_ids_arr[left_idx]) if src_ids_arr is not None else left_idx
        right_src_id = int(src_ids_arr[right_idx]) if src_ids_arr is not None else right_idx
        t_id = int(track_ids[j])
        out_item: dict[str, Any] = {
            "side": side,
            "handedness": side,
            "track_id": str(int(track_ids[j])),
        }
        for key in (
            "confidence",
            "score",
            "pose_confidence",
            "visualization_alpha",
            "visualization_weight",
            "constraint_weight",
            "soft_start",
            "soft_end",
            "soft_start_alpha",
            "soft_end_alpha",
            "soft_start_weight",
            "soft_end_weight",
        ):
            if key in l_rec:
                out_item[key] = copy.deepcopy(l_rec[key])

        q_orient = _slerp(obs_orient_quat[j], obs_orient_quat[j + 1], a)
        q_pose = _slerp_array(
            obs_pose_quat[j],
            obs_pose_quat[j + 1],
            np.full((obs_pose_quat.shape[1],), a, dtype=np.float64),
        )
        pose_aa = _pose_to_axis_angles(_mano_pose_from_quats(q_pose))
        if pchip_trans is not None:
            trans_out = np.array([f(float(t)) for f in pchip_trans], dtype=np.float64)
            trans_source = "pchip"
        else:
            trans_out = (1.0 - a) * obs_trans[j] + a * obs_trans[j + 1]
            trans_source = "linear"
            if left_idx < len(obs_trans_source):
                trans_source = obs_trans_source[left_idx]
            if right_idx < len(obs_trans_source):
                trans_source = f"{trans_source}+{obs_trans_source[right_idx]}"

        if geometry_generation == "linear_geometry":
            v0 = _to_float_array(l_rec.get("verts"))
            v1 = _to_float_array(r_rec.get("verts"))
            k0 = _to_float_array(l_rec.get("keypoints_3d"))
            k1 = _to_float_array(r_rec.get("keypoints_3d"))
            has_geometry = False
            if v0 is None or v1 is None or k0 is None or k1 is None:
                if mano_missing_fallback == "missing":
                    out[ti] = None
                    continue
            else:
                out_item["verts"] = ((1.0 - a) * v0 + a * v1).astype(np.float32)
                out_item["keypoints_3d"] = ((1.0 - a) * k0 + a * k1).astype(np.float32)
                has_geometry = True
        else:
            betas_out = np.asarray(track_betas[t_id], dtype=np.float64)
            has_geometry = False
            if mano_runtime is not None:
                geom = _mano_forward_state(
                    mano_runtime,
                    global_orient=_matrix_from_quat(q_orient),
                    pose=pose_aa,
                    betas=betas_out,
                    trans=trans_out,
                )
                if geom is not None:
                    verts_out, kpts_out = geom
                    out_item["verts"] = verts_out.astype(np.float32)
                    kpts_out = np.asarray(kpts_out, dtype=np.float64).reshape(1, -1, 3)
                    out_item["keypoints_3d"] = _pad_or_clip_keypoints(kpts_out, template_keypoints_shape)
                    mano_forward_count += 1
                    has_geometry = True
                else:
                    mano_forward_fail_count += 1
            else:
                mano_forward_fail_count += 1

            if not has_geometry and mano_missing_fallback != "missing":
                v0 = _to_float_array(l_rec.get("verts"))
                k0 = _to_float_array(l_rec.get("keypoints_3d"))
                v1 = _to_float_array(r_rec.get("verts"))
                k1 = _to_float_array(r_rec.get("keypoints_3d"))
                if v0 is not None and v1 is not None and v0.shape == v1.shape:
                    out_item["verts"] = ((1.0 - a) * v0 + a * v1).astype(np.float32)
                if k0 is not None and k1 is not None and k0.shape == k1.shape:
                    out_item["keypoints_3d"] = ((1.0 - a) * k0 + a * k1).astype(np.float32)
                if "verts" in out_item and "keypoints_3d" in out_item:
                    mano_fallback_count += 1
                    has_geometry = True

        if not has_geometry:
            out[ti] = None
            continue

        status[ti] = STATUS_INTERPOLATED
        src_left[ti] = left_idx
        src_right[ti] = right_idx
        alpha[ti] = float(a)

        out_item["mano_hand_global_orient"] = _matrix_from_quat(q_orient)[None, :, :].astype(np.float32)
        out_item["mano_hand_pose"] = _restore_pose_shape(pose_aa, mano_pose_shape)
        out_item["mano_hand_betas"] = np.asarray(track_betas[t_id], dtype=np.float32)[None, :]

        gap_q = float(gap)
        conf = _interpolated_confidence(
            l_rec,
            r_rec,
            float(a),
            gap_q,
            max_gap=max(float(gap_tau_sec), 1e-9),
            mid_strength=float(interp_midpoint_penalty_strength),
            base_scale=float(interpolated_constraint_scale),
            motion_quality=_compute_motion_quality(l_rec, r_rec, gap_q),
            enable_gap_decay=bool(apply_gap_decay),
            enable_motion_decay=bool(apply_motion_decay),
        )
        out_item["constraint_weight"] = np.asarray([min(conf * pose_confidence_scale, 1.0)], dtype=np.float32)
        out_item["interpolation_confidence"] = float(conf)
        out_item["visualization_weight"] = np.asarray(
            [min(_safe_pair_weight(l_rec, r_rec, "visualization_weight") * interpolated_visualization_scale, 1.0)],
            dtype=np.float32,
        )
        out_item["visualization_alpha"] = np.asarray(
            [min(_safe_pair_weight(l_rec, r_rec, "visualization_alpha") * interpolated_visualization_scale, 1.0)],
            dtype=np.float32,
        )
        out_item["is_observed"] = False
        out_item["is_interpolated"] = True
        out_item["interpolation_method"] = f"{side}_mano_parameter_slerp"
        out_item["left_source_index"] = left_idx
        out_item["right_source_index"] = right_idx
        out_item["frame_gap_sec"] = float(gap_q)
        out_item["left_source_frame_id"] = left_src_id
        out_item["right_source_frame_id"] = right_src_id
        out_item["alpha"] = float(a)
        out_item["translation_source"] = trans_source
        out_item["status"] = STATUS_INTERPOLATED
        out_item["geometry_generation"] = geometry_generation

        k0 = _keypoints_from_record(l_rec)
        k1 = _keypoints_from_record(r_rec)
        ratio = _palm_ratio(_palm_lengths(k0), _palm_lengths(k1))
        if ratio > max_palm_ratio:
            max_palm_ratio = float(ratio)

        out[ti] = out_item

    counts = {
        "observed": int(sum(1 for s in status if s == STATUS_OBSERVED)),
        "interpolated": int(sum(1 for s in status if s == STATUS_INTERPOLATED)),
        "missing": int(sum(1 for s in status if s == STATUS_MISSING)),
    }
    info = {
        "num_valid_source_detections": int(len(obs_records)),
        "num_missing_source_records": int(len(source_records) - len(obs_records)),
        "track_count": int(num_tracks),
        "track_sizes": [int(np.sum(track_ids == i)) for i in range(num_tracks)],
        "track_split_reasons": {str(k): int(v) for k, v in split_reason_counts.items()},
        "track_max_gap_sec": float(np.max(np.diff(obs_times))) if obs_times.size > 1 else 0.0,
        "mano_forward_count": int(mano_forward_count),
        "mano_forward_fail_count": int(mano_forward_fail_count),
        "mano_fallback_count": int(mano_fallback_count),
        "max_palm_ratio": float(max_palm_ratio),
        "geometry_generation": geometry_generation,
    }
    if obs_times.size > 1:
        info["max_observation_gap_sec"] = float(np.max(np.diff(obs_times)))
    return out, counts, src_left, src_right, alpha, info

def _mano_pose_from_quats(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Pose quaternions must be [J,4], got {q.shape}")
    return np.array([_matrix_from_quat(x) for x in q], dtype=np.float64)


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    q0 = _quat_normalize(np.asarray(q0, dtype=np.float64))
    q1 = _quat_normalize(np.asarray(q1, dtype=np.float64))
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    dot = float(np.dot(q0, q1))
    if dot > 0.9995:
        return _quat_normalize((1.0 - a) * q0 + a * q1)
    dot = np.clip(dot, -1.0, 1.0)
    theta0 = np.arccos(dot)
    sin0 = np.sin(theta0)
    if sin0 < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    theta = theta0 * float(a)
    sin_t = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_t / sin0
    s1 = sin_t / sin0
    return _quat_normalize(s0 * q0 + s1 * q1)


def _slerp_array(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if q0.shape != q1.shape:
        raise ValueError(f"slerp shape mismatch {q0.shape} vs {q1.shape}")
    a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if a.size != q0.shape[0]:
        if a.size != 1:
            raise ValueError("alpha size must match first dim")
        a = np.full((q0.shape[0],), float(a[0]), dtype=np.float64)
    out = np.empty_like(q0)
    for i in range(q0.shape[0]):
        out[i] = _slerp(q0[i], q1[i], float(a[i]))
    return out


def _interpolate_head(
    source_times: np.ndarray,
    source_pose: np.ndarray,
    target_times: np.ndarray,
    translation_method: str,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, list[str], list[int | None], list[int | None], list[float | None]]:
    n_src = source_pose.shape[0]
    n_tgt = target_times.size
    source_trans = source_pose[:, :3, 3]
    source_rot = _ensure_contiguous_quats(np.array([_quat_from_matrix(R[:3, :3]) for R in source_pose]))

    if translation_method == "pchip":
        Pchip = _load_pchip()
        if Pchip is not None and n_src >= 3:
            trans_out = np.stack(
                [Pchip(source_times, source_trans[:, i])(target_times) for i in range(3)],
                axis=1,
            ).astype(np.float64)
        else:
            trans_out = np.stack(
                [np.interp(target_times, source_times, source_trans[:, i]) for i in range(3)],
                axis=1,
            ).astype(np.float64)
    else:
        trans_out = np.stack(
            [np.interp(target_times, source_times, source_trans[:, i]) for i in range(3)],
            axis=1,
        ).astype(np.float64)

    quat_out = np.zeros((n_tgt, 4), dtype=np.float64)
    status = [STATUS_MISSING] * n_tgt
    src_left = [None] * n_tgt
    src_right = [None] * n_tgt
    alpha = [None] * n_tgt

    pos = np.searchsorted(source_times, target_times, side="right") - 1
    for i, t in enumerate(target_times):
        j = int(pos[i])
        if j < 0 or j >= n_src:
            continue

        if abs(float(t) - float(source_times[j])) <= tolerance:
            status[i] = STATUS_OBSERVED
            src_left[i] = j
            src_right[i] = j
            alpha[i] = 0.0
            trans_out[i] = source_trans[j]
            quat_out[i] = source_rot[j]
            continue

        if j + 1 >= n_src:
            continue

        t0 = float(source_times[j])
        t1 = float(source_times[j + 1])
        gap = t1 - t0
        if gap <= 0:
            continue
        a = (float(t) - t0) / gap
        if a < 0.0 or a > 1.0:
            continue

        status[i] = STATUS_INTERPOLATED
        src_left[i] = j
        src_right[i] = j + 1
        alpha[i] = float(a)
        trans_out[i] = (1.0 - a) * source_trans[j] + a * source_trans[j + 1]
        quat_out[i] = _slerp(source_rot[j], source_rot[j + 1], a)

    pose_out = np.repeat(np.eye(4, dtype=np.float64)[None], n_tgt, axis=0)
    pose_out[:, :3, :3] = np.array([_matrix_from_quat(q) for q in quat_out], dtype=np.float64)
    pose_out[:, :3, 3] = trans_out
    return pose_out, status, src_left, src_right, alpha


def _has_required_hand_fields(det: dict[str, Any] | None) -> bool:
    if det is None:
        return False
    try:
        verts = np.asarray(det["verts"], dtype=np.float64)
        keypoints = np.asarray(det["keypoints_3d"], dtype=np.float64)
        orient = np.asarray(det["mano_hand_global_orient"], dtype=np.float64)
        betas = np.asarray(det["mano_hand_betas"], dtype=np.float64)
    except Exception:
        return False
    if verts.ndim != 3 or verts.shape[0] != 1 or verts.shape[-1] != 3:
        return False
    if keypoints.ndim != 3 or keypoints.shape[0] != 1 or keypoints.shape[-1] != 3:
        return False
    if orient.shape not in {(1, 3, 3), (3, 3)}:
        return False
    if betas.ndim != 2:
        return False
    try:
        pose = _pose_to_axis_angles(np.asarray(det["mano_hand_pose"]))
    except Exception:
        return False
    if pose.shape != (15, 3):
        return False
    return True


def _safe_weight(det: dict[str, Any] | None, key: str) -> float:
    if det is None:
        return 1.0
    value = det.get(key)
    if value is None:
        return 1.0
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr[0]):
        return 1.0
    return float(np.clip(arr[0], 0.0, 1.0))


def _safe_pair_weight(left: dict[str, Any] | None, right: dict[str, Any] | None, key: str) -> float:
    return min(_safe_weight(left, key), _safe_weight(right, key))



def _check_hand_motion(name: str, records: list[dict[str, Any] | None], times: np.ndarray, *, max_wrist_speed_mps: float, max_wrist_rotation_deg: float) -> list[str]:
    warnings: list[str] = []
    prev_rec: dict[str, Any] | None = None
    prev_idx = -1
    for i, rec in enumerate(records):
        if rec is None:
            continue
        if prev_rec is not None and prev_idx >= 0:
            dt = float(times[i] - times[prev_idx])
            if dt > 0:
                p0 = np.asarray(prev_rec["keypoints_3d"], dtype=np.float64).reshape(-1, 3)
                p1 = np.asarray(rec["keypoints_3d"], dtype=np.float64).reshape(-1, 3)
                speed = float(np.linalg.norm(p1[0] - p0[0]) / dt)
                if speed > float(max_wrist_speed_mps):
                    warnings.append(f"{name}: frame={i}, wrist_speed={speed:.6f} > {max_wrist_speed_mps:.3f}")

                q0 = _quat_from_matrix(np.asarray(prev_rec["mano_hand_global_orient"], dtype=np.float64).reshape(3, 3))
                q1 = _quat_from_matrix(np.asarray(rec["mano_hand_global_orient"], dtype=np.float64).reshape(3, 3))
                if np.dot(q0, q1) < 0:
                    q1 = -q1
                rot = float(2.0 * np.degrees(np.arccos(np.clip(np.dot(q0, q1), -1.0, 1.0))))
                if rot > float(max_wrist_rotation_deg):
                    warnings.append(f"{name}: frame={i}, wrist_rotation_deg={rot:.3f} > {max_wrist_rotation_deg:.3f}")

    prev_rec = rec
    prev_idx = i
    return warnings


def _first_keypoints_shape(records: list[dict[str, Any] | None]) -> tuple[int, int, int] | None:
    for rec in records:
        kp = _keypoints_from_record(rec)
        if kp is None:
            continue
        if kp.ndim == 3 and kp.shape[0] > 0 and kp.shape[-1] == 3:
            return tuple(kp.shape)
    return None


def _build_hands_resample_config(
    hands_cfg: dict[str, Any],
    valid_cfg: dict[str, Any],
) -> dict[str, Any]:
    interpolation_cfg = _cfg_get(hands_cfg, "interpolation", {}) or {}
    if not isinstance(interpolation_cfg, dict):
        interpolation_cfg = {}

    cfg: dict[str, Any] = {}

    cfg["max_interpolation_gap_sec"] = float(_cfg_get(interpolation_cfg, "max_interpolation_gap_sec", 0.30))
    cfg["geometry_generation"] = str(_cfg_get(interpolation_cfg, "geometry_generation", "mano_forward"))
    cfg["translation_method"] = str(_cfg_get(interpolation_cfg, "translation_method", "pchip"))
    cfg["split_on_track_id_change"] = _cfg_bool(_cfg_get(interpolation_cfg, "split_on_track_id_change", True))
    cfg["split_on_handedness_change"] = _cfg_bool(_cfg_get(interpolation_cfg, "split_on_handedness_change", True))
    cfg["max_wrist_speed_mps"] = float(_cfg_get(interpolation_cfg, "max_wrist_speed_mps", _cfg_get(valid_cfg, "max_wrist_speed_mps", 4.0)))
    cfg["max_global_angular_speed_deg_s"] = float(
        _cfg_get(interpolation_cfg, "max_global_angular_speed_deg_s", _cfg_get(valid_cfg, "max_wrist_rotation_deg", 150.0))
    )
    cfg["max_palm_bone_change_ratio"] = float(_cfg_get(interpolation_cfg, "max_palm_bone_change_ratio", 0.15))
    cfg["max_beta_difference"] = float(_cfg_get(interpolation_cfg, "max_beta_difference", 2.0))
    cfg["interpolated_constraint_scale"] = float(_cfg_get(interpolation_cfg, "interpolated_constraint_scale", 0.35))
    cfg["pose_confidence_scale"] = float(_cfg_get(interpolation_cfg, "pose_confidence_scale", 1.0))
    cfg["interpolated_visualization_scale"] = float(_cfg_get(interpolation_cfg, "interpolated_visualization_scale", 1.0))
    cfg["gap_tau_sec"] = float(_cfg_get(interpolation_cfg, "gap_tau_sec", 0.30))
    cfg["interp_midpoint_penalty_strength"] = float(
        _cfg_get(interpolation_cfg, "interp_midpoint_penalty_strength", 0.50)
    )
    cfg["apply_gap_decay"] = _cfg_bool(_cfg_get(interpolation_cfg, "apply_gap_decay", True))
    cfg["apply_motion_decay"] = _cfg_bool(_cfg_get(interpolation_cfg, "apply_motion_decay", True))
    cfg["do_extrapolate"] = _cfg_bool(_cfg_get(interpolation_cfg, "do_extrapolate", False), default=False)
    cfg["mano_missing_fallback"] = str(_cfg_get(interpolation_cfg, "mano_missing_fallback", "missing"))
    cfg["beta_scope"] = str(_cfg_get(interpolation_cfg, "beta_scope", "per_sequence_per_side"))
    return cfg


def _to_json_compatible(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_compatible(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_json_lines(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_json_compatible(row), ensure_ascii=False) + "\n")


def _write_json_obj(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_json_compatible(data), f, ensure_ascii=False, indent=2)


def _write_pickle(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _timestamps_to_ns(times: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(times, dtype=np.float64) * 1e9).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample EgoAllo input timeline to target FPS.")
    parser.add_argument("--head-trajectory", required=True, type=Path)
    parser.add_argument("--timestamps-txt", required=True, type=Path)
    parser.add_argument("--hamer-pkl", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tmp-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    head_path = args.head_trajectory.expanduser().resolve()
    timestamps_txt = args.timestamps_txt.expanduser().resolve()
    hamer_path = args.hamer_pkl.expanduser().resolve()
    cfg = _load_yaml(args.config)
    tmp_dir = args.tmp_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    tr_cfg = _cfg_get(cfg, "temporal_resampling", {}) or {}
    if not _cfg_bool(_cfg_get(tr_cfg, "enabled", False), default=False):
        print("[TEMPORAL RESAMPLE] disabled (temporal_resampling.enabled=false)")
        return
    if _cfg_get(tr_cfg, "stage", "after_metric_scale") != "after_metric_scale":
        print("[TEMPORAL RESAMPLE] skipped due to stage != after_metric_scale")
        return

    source_times, source_frame_ids, source_src_frame_ids = _load_timestamps_txt(timestamps_txt)
    _normalize_timestamps(source_times)
    source_count = int(source_times.size)

    head_poses, head_dtype = _load_head(head_path)
    if head_poses.shape[0] != source_count:
        raise ValueError(f"head_trajectory rows ({head_poses.shape[0]}) != timestamps rows ({source_count})")

    hamer_data = _load_hamer(hamer_path)

    target_src_frame_ids, target_times = _generate_target_timestamps_by_src_frames(
        source_times,
        source_src_frame_ids,
    )
    target_fps = float((len(target_src_frame_ids) - 1) / max(target_times[-1] - target_times[0], 1e-12)) if target_times.size > 1 else 0.0

    target_count = int(target_times.size)

    source_delta = np.diff(source_times)
    source_fps = float(source_count / (source_times[-1] - source_times[0])) if source_count > 1 else 0.0
    source_max_gap = float(np.max(source_delta)) if source_delta.size else 0.0
    head_cfg = _cfg_get(tr_cfg, "head", {}) or {}
    hands_cfg = _cfg_get(tr_cfg, "hands", {}) or {}
    valid_cfg = _cfg_get(tr_cfg, "validation", {}) or {}
    output_cfg = _cfg_get(tr_cfg, "output", {}) or {}
    max_gap = float(_cfg_get(head_cfg, "max_interpolation_gap_sec", 0.40))
    source_target_same_timing = (
        source_times.shape == target_times.shape
        and np.allclose(source_times, target_times, rtol=0.0, atol=1e-9)
    )

    if source_count == target_count and source_target_same_timing:
        target_duration = float(target_times[-1] - target_times[0]) if target_times.size > 1 else 0.0
        print("[TEMPORAL RESAMPLE] skipped (already at source frame rate)")
        print("[TEMPORAL RESAMPLE] source_fps:", float(source_fps))
        print("[TEMPORAL RESAMPLE] source_frame_count:", source_count)
        print("[TEMPORAL RESAMPLE] target_frame_count:", target_count)
        print("[TEMPORAL RESAMPLE] duration_sec:", float(target_duration))
        print("[TEMPORAL RESAMPLE] head obs/interp/missing:", f"{source_count}/0/0")
        print(
            "[TEMPORAL RESAMPLE] left hand obs/interp/missing:",
            f"{sum(1 for rec in hamer_data['detections_left_wrt_cam'].values() if _has_required_hand_fields(rec))}/0/"
            f"{sum(1 for rec in hamer_data['detections_left_wrt_cam'].values() if not _has_required_hand_fields(rec))}",
        )
        print(
            "[TEMPORAL RESAMPLE] right hand obs/interp/missing:",
            f"{sum(1 for rec in hamer_data['detections_right_wrt_cam'].values() if _has_required_hand_fields(rec))}/0/"
            f"{sum(1 for rec in hamer_data['detections_right_wrt_cam'].values() if not _has_required_hand_fields(rec))}",
        )
        print("[TEMPORAL RESAMPLE] script exit: no output rewritten")
        return

    hands_interp_cfg = _build_hands_resample_config(hands_cfg, valid_cfg)

    fail_on_gap = bool(_cfg_get(valid_cfg, "fail_on_large_head_gap", True))
    if source_delta.size and source_max_gap > max_gap:
        msg = f"Source head gap {source_max_gap:.6f}s exceeds max_interpolation_gap_sec {max_gap:.6f}s"
        if fail_on_gap:
            raise ValueError(msg)
        print(f"[TEMPORAL RESAMPLE][WARN] {msg}")

    if _cfg_bool(_cfg_get(output_cfg, "backup_sparse_inputs", True), default=True):
        backup_dir = tmp_dir / "temporal_resampling"
        backup_dir.mkdir(parents=True, exist_ok=True)
        _safe_copy(head_path, backup_dir / "source_head_trajectory.npy")
        _safe_copy(timestamps_txt, backup_dir / "source_timestamps.txt")
        _safe_copy(hamer_path, backup_dir / "source_hamer_outputs.pkl")

    # Head.
    res_head, head_status, head_left, head_right, head_alpha = _interpolate_head(
        source_times,
        head_poses,
        target_times,
        translation_method=str(_cfg_get(head_cfg, "translation_method", "pchip")),
    )

    # Hands.
    cfg_root = args.config.parent.resolve()
    mano_runtimes = _resolve_mano_runtimes(
        cfg_root,
        cfg,
        device="cuda" if (torch is not None and torch.cuda.is_available()) else "cpu",
    )
    left_map = {int(ts): copy.deepcopy(det) for ts, det in (hamer_data["detections_left_wrt_cam"].items())}
    right_map = {int(ts): copy.deepcopy(det) for ts, det in (hamer_data["detections_right_wrt_cam"].items())}

    left_source_records: list[dict[str, Any] | None] = [
        _normalize_hand_detection_record(left_map.get(int(np.rint(float(t) * 1e9))))
        for t in source_times
    ]
    right_source_records: list[dict[str, Any] | None] = [
        _normalize_hand_detection_record(right_map.get(int(np.rint(float(t) * 1e9))))
        for t in source_times
    ]
    left_template_shape = _first_keypoints_shape(left_source_records)
    right_template_shape = _first_keypoints_shape(right_source_records)

    hands_enabled = _cfg_bool(_cfg_get(hands_cfg, "enabled", True), default=True)
    hand_max_interpolation_gap = float(hands_interp_cfg["max_interpolation_gap_sec"]) if hands_enabled else 0.0

    res_left, left_counts, left_l, left_r, left_a, left_info = _resample_hand(
        source_times,
        left_source_records,
        target_times,
        side="left",
        mano_runtime=mano_runtimes.get("left"),
        max_interpolation_gap_sec=hand_max_interpolation_gap,
        split_on_track_id_change=bool(hands_interp_cfg["split_on_track_id_change"]),
        split_on_handedness_change=bool(hands_interp_cfg["split_on_handedness_change"]),
        max_wrist_speed_mps=float(hands_interp_cfg["max_wrist_speed_mps"]),
        max_global_angular_speed_deg_s=float(hands_interp_cfg["max_global_angular_speed_deg_s"]),
        max_palm_bone_change_ratio=float(hands_interp_cfg["max_palm_bone_change_ratio"]),
        max_beta_difference=float(hands_interp_cfg["max_beta_difference"]),
        translation_method=str(hands_interp_cfg["translation_method"]),
        interpolated_constraint_scale=float(hands_interp_cfg["interpolated_constraint_scale"]),
        pose_confidence_scale=float(hands_interp_cfg["pose_confidence_scale"]),
        interpolated_visualization_scale=float(hands_interp_cfg["interpolated_visualization_scale"]),
        beta_scope=str(hands_interp_cfg["beta_scope"]),
        geometry_generation=str(hands_interp_cfg["geometry_generation"]),
        mano_missing_fallback=str(hands_interp_cfg["mano_missing_fallback"]),
        interp_midpoint_penalty_strength=float(hands_interp_cfg["interp_midpoint_penalty_strength"]),
        gap_tau_sec=float(hands_interp_cfg["gap_tau_sec"]),
        apply_gap_decay=bool(hands_interp_cfg["apply_gap_decay"]),
        apply_motion_decay=bool(hands_interp_cfg["apply_motion_decay"]),
        do_extrapolate=bool(hands_interp_cfg["do_extrapolate"]),
        timestamp_tolerance_sec=float(_cfg_get(valid_cfg, "timestamp_tolerance_sec", 0.005)),
        source_frame_ids=source_src_frame_ids,
        template_keypoints_shape=left_template_shape,
    )

    res_right, right_counts, right_l, right_r, right_a, right_info = _resample_hand(
        source_times,
        right_source_records,
        target_times,
        side="right",
        mano_runtime=mano_runtimes.get("right"),
        max_interpolation_gap_sec=hand_max_interpolation_gap,
        split_on_track_id_change=bool(hands_interp_cfg["split_on_track_id_change"]),
        split_on_handedness_change=bool(hands_interp_cfg["split_on_handedness_change"]),
        max_wrist_speed_mps=float(hands_interp_cfg["max_wrist_speed_mps"]),
        max_global_angular_speed_deg_s=float(hands_interp_cfg["max_global_angular_speed_deg_s"]),
        max_palm_bone_change_ratio=float(hands_interp_cfg["max_palm_bone_change_ratio"]),
        max_beta_difference=float(hands_interp_cfg["max_beta_difference"]),
        translation_method=str(hands_interp_cfg["translation_method"]),
        interpolated_constraint_scale=float(hands_interp_cfg["interpolated_constraint_scale"]),
        pose_confidence_scale=float(hands_interp_cfg["pose_confidence_scale"]),
        interpolated_visualization_scale=float(hands_interp_cfg["interpolated_visualization_scale"]),
        beta_scope=str(hands_interp_cfg["beta_scope"]),
        geometry_generation=str(hands_interp_cfg["geometry_generation"]),
        mano_missing_fallback=str(hands_interp_cfg["mano_missing_fallback"]),
        interp_midpoint_penalty_strength=float(hands_interp_cfg["interp_midpoint_penalty_strength"]),
        gap_tau_sec=float(hands_interp_cfg["gap_tau_sec"]),
        apply_gap_decay=bool(hands_interp_cfg["apply_gap_decay"]),
        apply_motion_decay=bool(hands_interp_cfg["apply_motion_decay"]),
        do_extrapolate=bool(hands_interp_cfg["do_extrapolate"]),
        timestamp_tolerance_sec=float(_cfg_get(valid_cfg, "timestamp_tolerance_sec", 0.005)),
        source_frame_ids=source_src_frame_ids,
        template_keypoints_shape=right_template_shape,
    )

    hamer_out = dict(hamer_data)
    target_ns = _timestamps_to_ns(target_times)
    hamer_out["detections_left_wrt_cam"] = {
        int(ts): (None if rec is None else _normalize_hand_detection_record(rec))
        for ts, rec in zip(target_ns, res_left)
    }
    hamer_out["detections_right_wrt_cam"] = {
        int(ts): (None if rec is None else _normalize_hand_detection_record(rec))
        for ts, rec in zip(target_ns, res_right)
    }

    warnings: list[str] = []
    warnings.extend(
        _check_hand_motion(
            "left",
            [None if r is None else r for r in res_left],
            target_times,
            max_wrist_speed_mps=float(_cfg_get(valid_cfg, "max_wrist_speed_mps", 4.0)),
            max_wrist_rotation_deg=float(_cfg_get(valid_cfg, "max_wrist_rotation_deg", 150.0)),
        )
    )
    warnings.extend(
        _check_hand_motion(
            "right",
            [None if r is None else r for r in res_right],
            target_times,
            max_wrist_speed_mps=float(_cfg_get(valid_cfg, "max_wrist_speed_mps", 4.0)),
            max_wrist_rotation_deg=float(_cfg_get(valid_cfg, "max_wrist_rotation_deg", 150.0)),
        )
    )

    mapping_path = _resolve_output_path(output_dir, _cfg_get(output_cfg, "mapping_file"), "tmp/temporal_resampling_map.jsonl")
    metadata_path = _resolve_output_path(output_dir, _cfg_get(output_cfg, "metadata_file"), "tmp/temporal_resampling_metadata.json")

    mapping_rows: list[dict[str, Any]] = []
    for i, t in enumerate(target_times):
        if left_l[i] is not None and left_a[i] == 0.0:
            ls = STATUS_OBSERVED
        elif left_l[i] is not None:
            ls = STATUS_INTERPOLATED
        else:
            ls = STATUS_MISSING

        if right_l[i] is not None and right_a[i] == 0.0:
            rs = STATUS_OBSERVED
        elif right_l[i] is not None:
            rs = STATUS_INTERPOLATED
        else:
            rs = STATUS_MISSING

        a = (
            float(head_alpha[i])
            if head_alpha[i] is not None
            else float(left_a[i]) if left_a[i] is not None else float(right_a[i]) if right_a[i] is not None else None
        )

        mapping_rows.append(
            {
                "target_frame_id": int(i),
                "timestamp": float(t),
                "left_source_index": None if left_l[i] is None else int(left_l[i]),
                "right_source_index": None if right_l[i] is None else int(right_l[i]),
                "alpha": None if a is None else float(a),
                "head_status": head_status[i],
                "left_hand_status": ls,
                "right_hand_status": rs,
            }
        )

    source_duration = float(source_times[-1] - source_times[0]) if source_times.size > 1 else 0.0
    target_duration = float(target_times[-1] - target_times[0]) if target_times.size > 1 else 0.0
    metadata = {
        "enabled": True,
        "source_fps": float(source_fps),
        "source_frame_count": int(source_count),
        "target_frame_count": int(target_count),
        "duration_sec": float(target_duration),
        "head_observed_count": int(sum(1 for s in head_status if s == STATUS_OBSERVED)),
        "head_interpolated_count": int(sum(1 for s in head_status if s == STATUS_INTERPOLATED)),
        "left_hand_observed_count": int(left_counts["observed"]),
        "left_hand_interpolated_count": int(left_counts["interpolated"]),
        "left_hand_missing_count": int(left_counts["missing"]),
        "right_hand_observed_count": int(right_counts["observed"]),
        "right_hand_interpolated_count": int(right_counts["interpolated"]),
        "right_hand_missing_count": int(right_counts["missing"]),
        "max_source_gap_sec": float(source_max_gap),
        "left_source_frame_count": int(len(left_source_records)),
        "right_source_frame_count": int(len(right_source_records)),
        "source_duration_sec": float(source_duration),
        "target_duration_sec": float(target_duration),
        "source": {
            "head": {
                "max_interpolation_gap_sec": float(max_gap),
            },
            "hands": {
                "left": left_info,
                "right": right_info,
            },
        },
        "validation": {
            "warnings": warnings,
            "warnings_count": int(len(warnings)),
        },
    }

    # Atomic replace outputs.
    _atomic_replace(head_path, lambda p: np.save(p, res_head.astype(head_dtype, copy=False)))
    target_frame_ids = np.arange(target_count, dtype=np.int64)
    _atomic_replace(
        timestamps_txt,
        lambda p: _write_timestamps_txt(p, target_times, target_frame_ids, target_src_frame_ids),
    )
    _atomic_replace(hamer_path, lambda p: _write_pickle(p, hamer_out))
    _atomic_replace(mapping_path, lambda p: _write_json_lines(p, mapping_rows))
    _atomic_replace(metadata_path, lambda p: _write_json_obj(p, metadata))

    print("[TEMPORAL RESAMPLE] source_fps:", float(source_fps))
    print("[TEMPORAL RESAMPLE] source_frame_count:", source_count)
    print("[TEMPORAL RESAMPLE] target_frame_count:", target_count)
    print("[TEMPORAL RESAMPLE] duration_sec:", float(target_duration))
    print(
        "[TEMPORAL RESAMPLE] head obs/interp/missing:",
        f"{sum(1 for s in head_status if s == STATUS_OBSERVED)}/"
        f"{sum(1 for s in head_status if s == STATUS_INTERPOLATED)}/"
        f"{sum(1 for s in head_status if s == STATUS_MISSING)}",
    )
    print(
        "[TEMPORAL RESAMPLE] left hand obs/interp/missing:",
        f"{left_counts['observed']}/{left_counts['interpolated']}/{left_counts['missing']}",
    )
    print(
        "[TEMPORAL RESAMPLE] right hand obs/interp/missing:",
        f"{right_counts['observed']}/{right_counts['interpolated']}/{right_counts['missing']}",
    )
    if warnings:
        print(f"[TEMPORAL RESAMPLE][WARN] {len(warnings)} validation warnings; see {metadata_path}")


if __name__ == "__main__":
    main()
