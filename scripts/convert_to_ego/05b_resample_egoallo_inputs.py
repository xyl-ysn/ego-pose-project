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
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


STATUS_OBSERVED = "observed"
STATUS_INTERPOLATED = "interpolated"
STATUS_MISSING = "missing"


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
    target_fps: float | None = None,
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
    if target_fps is not None and float(target_fps) > 0.0:
        print(
            "[TEMPORAL RESAMPLE][WARN] target_fps is ignored; "
            "target frames are generated from contiguous src_frame_id range.",
            flush=True,
        )
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


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _quat_normalize(q0 + float(alpha) * (q1 - q0))

    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    if abs(sin_theta_0) < 1e-12:
        return q0
    theta = theta_0 * float(alpha)
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(np.sin(theta) / sin_theta_0)
    return _quat_normalize(s0 * q0 + s1 * q1)


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


def _interpolate_head(
    source_times: np.ndarray,
    source_pose: np.ndarray,
    target_times: np.ndarray,
    translation_method: str,
    max_interpolation_gap_sec: float,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, list[str], list[int | None], list[int | None], list[float | None]]:
    n_src = source_pose.shape[0]
    n_tgt = target_times.size
    source_trans = source_pose[:, :3, 3]
    source_rot = _ensure_contiguous_quats(np.array([_quat_from_matrix(R[:3, :3]) for R in source_pose]))

    method = str(translation_method).lower()
    Pchip = _load_pchip()
    if method == "pchip" and Pchip is not None and n_src >= 3:
        try:
            pchip = [Pchip(source_times, source_trans[:, axis]) for axis in range(3)]
            trans_out = np.stack([np.asarray(f(target_times), dtype=np.float64) for f in pchip], axis=1)
        except Exception:
            trans_out = np.stack(
                [
                    np.interp(target_times, source_times, source_trans[:, axis])
                    for axis in range(3)
                ],
                axis=1,
            )
    else:
        trans_out = np.stack(
            [
                np.interp(target_times, source_times, source_trans[:, axis])
                for axis in range(3)
            ],
            axis=1,
        )

    quat_out = np.repeat(source_rot[:1], n_tgt, axis=0)
    status = [STATUS_MISSING] * n_tgt
    src_left = [None] * n_tgt
    src_right = [None] * n_tgt
    alpha = [None] * n_tgt

    pos = np.searchsorted(source_times, target_times, side="right") - 1
    for i, t in enumerate(target_times):
        j = int(pos[i])
        if j < 0:
            quat_out[i] = source_rot[0]
            continue

        if j >= n_src:
            quat_out[i] = source_rot[-1]
            continue

        if abs(float(t) - float(source_times[j])) <= tolerance:
            status[i] = STATUS_OBSERVED
            src_left[i] = j
            src_right[i] = j
            alpha[i] = 0.0
            quat_out[i] = source_rot[j]
            continue

        if j + 1 >= n_src:
            quat_out[i] = source_rot[-1]
            continue

        t0 = float(source_times[j])
        t1 = float(source_times[j + 1])
        gap = t1 - t0
        if gap <= 0:
            quat_out[i] = source_rot[j]
            continue

        a = (float(t) - t0) / gap
        if a < 0.0 or a > 1.0:
            quat_out[i] = source_rot[j]
            continue

        status[i] = STATUS_INTERPOLATED
        src_left[i] = j
        src_right[i] = j + 1
        alpha[i] = float(a)
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


def _copy_observed_hands_to_target(
    source_timestamps: np.ndarray,
    source_records: list[dict[str, Any] | None],
    target_times: np.ndarray,
    *,
    timestamp_tolerance_sec: float,
    source_frame_ids: np.ndarray | None = None,
) -> tuple[list[dict[str, Any] | None], dict[str, int], list[int | None], list[float | None]]:
    source_timestamps = np.asarray(source_timestamps, dtype=np.float64).reshape(-1)
    target_times = np.asarray(target_times, dtype=np.float64).reshape(-1)
    src_ids_arr = (
        np.asarray(source_frame_ids, dtype=np.int64)
        if source_frame_ids is not None and len(source_frame_ids) == len(source_records)
        else None
    )

    out: list[dict[str, Any] | None] = [None] * int(target_times.size)
    src_indices: list[int | None] = [None] * int(target_times.size)
    alpha: list[float | None] = [None] * int(target_times.size)
    observed = 0

    for ti, t in enumerate(target_times):
        pos = int(np.searchsorted(source_timestamps, float(t)))
        candidates: list[int] = []
        if pos < source_timestamps.size:
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            continue
        idx = min(candidates, key=lambda i: abs(float(source_timestamps[i]) - float(t)))
        if abs(float(source_timestamps[idx]) - float(t)) > float(timestamp_tolerance_sec):
            continue
        rec = _normalize_hand_detection_record(source_records[idx])
        if not _has_required_hand_fields(rec):
            continue

        out_item = copy.deepcopy(rec)
        src_frame_id = int(src_ids_arr[idx]) if src_ids_arr is not None else int(idx)
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
        src_indices[ti] = int(idx)
        alpha[ti] = 0.0
        observed += 1

    return out, {
        "observed": int(observed),
        "interpolated": 0,
        "missing": int(target_times.size) - int(observed),
    }, src_indices, alpha


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

    configured_target_fps = _cfg_float(tr_cfg, "target_fps", 30.0)
    target_src_frame_ids, target_times = _generate_target_timestamps_by_src_frames(
        source_times,
        source_src_frame_ids,
        target_fps=configured_target_fps,
    )
    target_fps = float((len(target_src_frame_ids) - 1) / max(target_times[-1] - target_times[0], 1e-12)) if target_times.size > 1 else 0.0

    target_count = int(target_times.size)

    source_delta = np.diff(source_times)
    source_fps = float(source_count / (source_times[-1] - source_times[0])) if source_count > 1 else 0.0
    source_max_gap = float(np.max(source_delta)) if source_delta.size else 0.0
    head_cfg = _cfg_get(tr_cfg, "head", {}) or {}
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

    if source_delta.size and source_max_gap > max_gap:
        msg = f"Source head gap {source_max_gap:.6f}s exceeds max_interpolation_gap_sec {max_gap:.6f}s"
        if _cfg_bool(_cfg_get(head_cfg, "fail_on_large_head_gap", True), default=True):
            raise ValueError(msg)
        print(f"[TEMPORAL RESAMPLE][WARN] {msg}; continuing with legacy global head interpolation")

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
        max_interpolation_gap_sec=float(_cfg_get(head_cfg, "max_interpolation_gap_sec", 0.40)),
    )

    # Hands are no longer interpolated. Keep only real observations that align
    # with the resampled timestamp grid; target frames between observations stay missing.
    left_map = {int(ts): det for ts, det in (hamer_data["detections_left_wrt_cam"].items())}
    right_map = {int(ts): det for ts, det in (hamer_data["detections_right_wrt_cam"].items())}

    left_source_records: list[dict[str, Any] | None] = [
        _normalize_hand_detection_record(left_map.get(int(np.rint(float(t) * 1e9))))
        for t in source_times
    ]
    right_source_records: list[dict[str, Any] | None] = [
        _normalize_hand_detection_record(right_map.get(int(np.rint(float(t) * 1e9))))
        for t in source_times
    ]
    res_left, left_counts, left_l, left_a = _copy_observed_hands_to_target(
        source_times,
        left_source_records,
        target_times,
        timestamp_tolerance_sec=0.005,
        source_frame_ids=source_src_frame_ids,
    )
    res_right, right_counts, right_l, right_a = _copy_observed_hands_to_target(
        source_times,
        right_source_records,
        target_times,
        timestamp_tolerance_sec=0.005,
        source_frame_ids=source_src_frame_ids,
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

    mapping_path = _resolve_output_path(output_dir, _cfg_get(output_cfg, "mapping_file"), "tmp/temporal_resampling_map.jsonl")
    metadata_path = _resolve_output_path(output_dir, _cfg_get(output_cfg, "metadata_file"), "tmp/temporal_resampling_metadata.json")

    mapping_rows: list[dict[str, Any]] = []
    for i, t in enumerate(target_times):
        if left_l[i] is not None and left_a[i] == 0.0:
            ls = STATUS_OBSERVED
        else:
            ls = STATUS_MISSING

        if right_l[i] is not None and right_a[i] == 0.0:
            rs = STATUS_OBSERVED
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
        "target_fps": float(configured_target_fps),
        "estimated_target_fps": float(target_fps),
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
                "fail_on_large_head_gap": _cfg_bool(_cfg_get(head_cfg, "fail_on_large_head_gap", True), default=True),
                "translation_method": str(_cfg_get(head_cfg, "translation_method", "pchip")),
                "translation_fallback": "linear",
                "rotation_method": "slerp",
            },
            "hands": {
                "mode": "observed_only",
                "timestamp_tolerance_sec": 0.005,
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
