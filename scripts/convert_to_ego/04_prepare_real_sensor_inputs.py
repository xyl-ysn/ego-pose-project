#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

# Stationary-window search for the IMU gravity direction.
# A median over the whole segment mixes every head orientation together, so the
# gravity direction is taken from the most stationary window instead, and the
# frame whose pose matches that window is written next to it.
IMU_STATIONARY_WINDOW_SEC = 0.5
IMU_STATIONARY_MIN_SAMPLES = 10
IMU_STATIONARY_MAX_GYRO = 0.25
IMU_STATIONARY_MAX_DIR_SPREAD_DEG = 3.0
IMU_STATIONARY_MAX_NORM_REL_DEV = 0.05

# Gravity-floor metric scale from real depth.
# These thresholds are in metric units because real depth is metric after
# --depth-unit-scale. The depth camera is sampled with a pixel stride before
# backprojection to keep the per-frame floor search fast.
DEPTH_BACKPROJECT_STRIDE = 4
FLOOR_HEIGHT_BIN = 0.03
FLOOR_MIN_SUPPORT = 2000
FLOOR_MIN_COVERAGE_M2 = 0.20
FLOOR_MAX_RESIDUAL = 0.08
FLOOR_MAX_HEIGHT_RANK = 0.35
FLOOR_MIN_VALID_FRAMES = 5
SCALE_OUTLIER_MAD = 3.0
DEPTH_FLOOR_MIN_SCALE = 0.05
DEPTH_FLOOR_MAX_SCALE = 20.0
DEPTH_SCALE_CLUSTER_REL_TOL = 0.08
DEPTH_SCALE_MIN_SUPPORT_FRAMES = 3
DEPTH_SCALE_LOCAL_WINDOW = 15
DEPTH_SCALE_LOCAL_MIN_SUPPORT_FRAMES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real IMU/calibration/depth inputs for BodyMesh.")
    parser.add_argument("--episode-dir", required=True, type=Path)
    parser.add_argument("--camera-stream-id", default="ego_depth")
    parser.add_argument("--intrinsics-out", type=Path, default=None)
    parser.add_argument("--imu-gravity-out", type=Path, default=None)
    parser.add_argument("--scale-json", type=Path, default=None)
    parser.add_argument("--timestamps-txt", type=Path, default=None)
    parser.add_argument("--head-trajectory-in", type=Path, default=None)
    parser.add_argument("--point-cloud-r-align", type=Path, default=None)
    parser.add_argument("--point-cloud-align-meta", type=Path, default=None)
    parser.add_argument("--depth-unit-scale", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument(
        "--depth-scale-cluster-rel-tol",
        type=float,
        default=DEPTH_SCALE_CLUSTER_REL_TOL,
        help="Relative scale tolerance used to group the same horizontal surface across frames.",
    )
    parser.add_argument(
        "--depth-scale-local-window",
        type=int,
        default=DEPTH_SCALE_LOCAL_WINDOW,
        help="Number of raw depth frames searched forward/backward around each seed frame.",
    )
    parser.add_argument(
        "--depth-scale-local-min-support-frames",
        type=int,
        default=DEPTH_SCALE_LOCAL_MIN_SUPPORT_FRAMES,
        help="Minimum merged forward/backward raw depth frames required to verify a scale candidate.",
    )
    parser.add_argument(
        "--depth-scale-min-support-frames",
        type=int,
        default=DEPTH_SCALE_MIN_SUPPORT_FRAMES,
        help="Minimum distinct depth frames required before a low plane can define metric scale.",
    )
    parser.add_argument("--imu-stationary-window-sec", type=float, default=IMU_STATIONARY_WINDOW_SEC)
    parser.add_argument("--imu-stationary-max-gyro", type=float, default=IMU_STATIONARY_MAX_GYRO)
    parser.add_argument("--imu-stationary-max-dir-spread-deg", type=float, default=IMU_STATIONARY_MAX_DIR_SPREAD_DEG)
    parser.add_argument("--imu-stationary-max-norm-rel-dev", type=float, default=IMU_STATIONARY_MAX_NORM_REL_DEV)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON must be an object: {path}")
    return data


def resolve_uri(root: Path, uri: str) -> Path:
    path = Path(uri).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def open_depth_frames(root: Path, stream: dict[str, Any]) -> dict[str, Any]:
    """Open the new episode zarr depth stream as a frame source."""
    raw_uri = str(stream.get("uri") or "").strip()
    if not raw_uri:
        raise KeyError(f"Depth stream missing uri: {stream.get('stream_id')}")
    path = resolve_uri(root, raw_uri)
    array_name = str(stream.get("array") or "depth").strip() or "depth"

    try:
        import zarr
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError("zarr is required to read episode depth zarr") from exc

    root_zarr = zarr.open(str(path), mode="r")
    try:
        array = root_zarr[array_name]
    except (KeyError, TypeError):
        if hasattr(root_zarr, "shape"):
            array = root_zarr
        else:
            raise KeyError(f"Depth zarr missing array '{array_name}': {path}")

    if not hasattr(array, "shape") or len(array.shape) < 3:
        raise ValueError(f"Depth zarr array must have shape (frames, height, width): {path}")

    return {
        "kind": "zarr",
        "path": path,
        "array_name": array_name,
        "array": array,
        "frame_count": int(array.shape[0]),
        "shape": [int(v) for v in array.shape],
    }


def read_depth_frame(depth_source: dict[str, Any], frame_id: int, unit_scale: float) -> np.ndarray:
    if depth_source.get("kind") != "zarr":
        raise ValueError(f"Unsupported depth source: {depth_source.get('kind')}")
    frame_id = int(frame_id)
    if frame_id < 0 or frame_id >= int(depth_source["frame_count"]):
        raise IndexError(f"Depth frame {frame_id} out of range {depth_source['frame_count']}")
    depth = np.asarray(depth_source["array"][frame_id], dtype=np.float32)
    depth = depth * float(unit_scale)
    depth[depth <= 0.0] = np.nan
    return depth


def depth_frame_index_for_pose(
    pose_index: int,
    timestamps: list[dict[str, Any]],
    frame_count: int,
) -> int | None:
    candidates: list[tuple[int, str]] = []
    if pose_index < len(timestamps):
        row = timestamps[pose_index]
        candidates.extend([
            (int(row["src_frame_id"]), "src_frame_id"),
            (int(row["frame_id"]), "frame_id"),
        ])
    candidates.append((int(pose_index), "pose_index"))
    for idx, _source in candidates:
        if 0 <= idx < int(frame_count):
            return idx
    return None


def stream_by(segment: dict[str, Any], *, modality: str | None = None, stream_id: str | None = None) -> dict[str, Any]:
    for stream in segment.get("streams", []):
        if not isinstance(stream, dict):
            continue
        if stream_id is not None and stream.get("stream_id") == stream_id:
            return stream
        if modality is not None and stream.get("modality") == modality:
            return stream
    available = [
        {"stream_id": stream.get("stream_id"), "modality": stream.get("modality")}
        for stream in segment.get("streams", [])
        if isinstance(stream, dict)
    ]
    raise KeyError(f"Stream not found: modality={modality!r}, stream_id={stream_id!r}; available={available}")


def load_segment(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    segment = load_json(root / "episode.json")
    calib_uri = segment.get("calibration_uri")
    if not calib_uri:
        raise KeyError(f"{root / 'episode.json'} missing calibration_uri")
    calib_uri = str(calib_uri)
    return segment, load_json(resolve_uri(root, calib_uri))


def calibration_cameras(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    return [camera for camera in calibration.get("cameras", []) if isinstance(camera, dict)]


def camera_calibration(calibration: dict[str, Any], stream_id: str) -> dict[str, Any]:
    for camera in calibration_cameras(calibration):
        if isinstance(camera, dict) and camera.get("stream_id") == stream_id:
            return camera
    available = [camera.get("stream_id") for camera in calibration_cameras(calibration)]
    raise KeyError(f"Camera calibration not found for stream_id={stream_id!r}; available={available}")


def camera_calibration_for_stream(
    calibration: dict[str, Any],
    stream: dict[str, Any],
) -> dict[str, Any]:
    """Return calibration for a stream, falling back to its frame_id."""
    stream_id = str(stream.get("stream_id") or "")
    try:
        return camera_calibration(calibration, stream_id)
    except KeyError:
        frame_id = str(stream.get("frame_id") or "")
        if frame_id:
            for camera in calibration_cameras(calibration):
                if camera.get("frame_id") == frame_id or camera.get("source_frame_id") == frame_id:
                    return camera
    raise KeyError(
        f"Camera calibration not found for stream_id={stream_id!r}, frame_id={stream.get('frame_id')!r}"
    )


def resolve_camera_stream_id(
    calibration: dict[str, Any],
    requested_stream_id: str | None,
) -> tuple[str, dict[str, Any]]:
    cameras = calibration_cameras(calibration)
    camera_ids = {str(camera.get("stream_id")) for camera in cameras if camera.get("stream_id") is not None}
    requested = str(requested_stream_id or "auto")
    if requested != "auto" and requested in camera_ids:
        return requested, {
            "requested_stream_id": requested,
            "resolved_stream_id": requested,
            "reason": "requested_camera_found",
            "available_camera_stream_ids": sorted(camera_ids),
        }

    non_depth_cameras = [
        camera for camera in cameras
        if "depth" not in f"{camera.get('stream_id', '')} {camera.get('frame_id', '')}".lower()
    ]
    candidates = non_depth_cameras or cameras
    if candidates:
        stream_id = str(candidates[0]["stream_id"])
        reason = (
            "auto_first_non_depth_calibrated_camera"
            if requested == "auto"
            else "requested_missing_use_first_non_depth_calibrated_camera"
        )
        return stream_id, {
            "requested_stream_id": requested,
            "resolved_stream_id": stream_id,
            "reason": reason,
            "available_camera_stream_ids": sorted(camera_ids),
        }

    raise KeyError("No camera calibration entries found in calibration file.")


def write_intrinsics(
    calibration: dict[str, Any],
    stream_id: str,
    out: Path,
    timestamps_txt: Path | None,
) -> dict[str, Any]:
    if timestamps_txt is None:
        raise ValueError("--intrinsics-out requires --timestamps-txt")
    camera = camera_calibration(calibration, stream_id)
    geometry = camera.get("prepared_image_geometry") or camera
    intr = geometry.get("intrinsics") or {}
    resolution = geometry.get("resolution")
    if isinstance(resolution, Mapping):
        resolution = [resolution["width"], resolution["height"]]
    resolution = np.asarray(resolution, dtype=np.float64)
    if resolution.shape != (2,) or not np.isfinite(resolution).all() or np.any(resolution <= 0):
        raise ValueError(f"Camera {stream_id} requires a positive [width, height] resolution")
    row = [float(intr[k]) for k in ("fx", "fy", "cx", "cy")]
    row.extend(resolution.tolist())
    frame_count = len(read_timestamps(timestamps_txt))
    if frame_count <= 0:
        raise ValueError(f"No timestamps found: {timestamps_txt}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out,
        np.repeat(np.asarray(row, dtype=np.float64)[None], frame_count, axis=0),
        fmt="%.10f",
        header="fx fy cx cy source_width source_height",
    )
    return {
        "stream_id": stream_id,
        "source": "prepared_image_geometry" if geometry is not camera else "camera",
        "intrinsics": dict(zip(("fx", "fy", "cx", "cy"), row[:4])),
        "resolution": dict(zip(("width", "height"), row[4:])),
    }


def read_parquet_table(path: Path):
    try:
        import pandas as pd

        return pd.read_parquet(path)
    except Exception as pandas_exc:
        try:
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pydict()
        except Exception as arrow_exc:
            raise RuntimeError(
                f"Reading parquet requires pandas+pyarrow/fastparquet or pyarrow: {path}; "
                f"pandas_error={pandas_exc!r}; pyarrow_error={arrow_exc!r}"
            ) from arrow_exc


def table_columns(table) -> list[str]:
    return list(table.keys()) if isinstance(table, Mapping) else list(table.columns)


def column_values(table, name: str) -> list[Any]:
    if isinstance(table, Mapping):
        return list(table[name])
    series = table[name]
    return series.to_list() if hasattr(series, "to_list") else list(series)


def columns_to_numpy(table, cols: list[str]) -> np.ndarray:
    return np.column_stack([np.asarray(column_values(table, c), dtype=np.float64) for c in cols])


def stacked_vector_column(df, names: list[str]) -> np.ndarray | None:
    columns = table_columns(df)
    for cols in (
        names,
        [f"linear_acceleration_{axis}" for axis in "xyz"],
        [f"linear_acceleration.{axis}" for axis in "xyz"],
        [f"accel_{axis}" for axis in "xyz"],
        [f"acc_{axis}" for axis in "xyz"],
    ):
        if all(c in columns for c in cols):
            return columns_to_numpy(df, cols)

    for col in ("linear_acceleration", "acceleration", "accel", "acc"):
        if col not in columns:
            continue
        values = []
        for v in column_values(df, col):
            if isinstance(v, dict):
                xyz = [v.get(k) for k in ("x", "y", "z")]
                values.append(xyz if all(x is not None for x in xyz) else list(v.values())[:3])
            else:
                values.append(np.asarray(v, dtype=np.float64).reshape(-1)[:3])
        return np.vstack(values)
    return None


def gyro_vector_column(df) -> np.ndarray | None:
    """Angular velocity, used only to tell whether a window is stationary."""
    columns = table_columns(df)
    for cols in (
        [f"angular_velocity_{axis}" for axis in "xyz"],
        [f"angular_velocity.{axis}" for axis in "xyz"],
        ["wx", "wy", "wz"],
        [f"gyro_{axis}" for axis in "xyz"],
        ["gx", "gy", "gz"],
    ):
        if all(c in columns for c in cols):
            return columns_to_numpy(df, cols)
    return None


def imu_timestamps_sec(df) -> np.ndarray | None:
    """Segment-relative IMU timestamps in seconds."""
    columns = table_columns(df)
    for name, scale in (
        ("timestamp_ns", 1e-9),
        ("timestamp_us", 1e-6),
        ("timestamp_ms", 1e-3),
        ("timestamp_sec", 1.0),
        ("timestamp", 1.0),
        ("time", 1.0),
    ):
        if name in columns:
            return np.asarray(column_values(df, name), dtype=np.float64) * scale
    return None


def select_stationary_imu_window(
    times: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray | None,
    *,
    window_sec: float,
    max_gyro: float,
    max_dir_spread_deg: float,
    max_norm_rel_dev: float,
    min_samples: int = IMU_STATIONARY_MIN_SAMPLES,
) -> tuple[slice | None, dict[str, Any]]:
    """
    Find the window where the headset is closest to standing still.

    While the head rotates, the accelerometer direction in the camera frame
    rotates with it, so a median over the whole segment is not the gravity
    direction at any single frame. A window is judged stationary when the
    accelerometer direction barely moves, its magnitude stays at the segment
    gravity magnitude, and the gyroscope is quiet.

    Returns:
        window_slice, stats. window_slice is None when no window has enough
        samples, in which case the caller should use the whole segment.
    """
    norms = np.linalg.norm(acc, axis=1)
    unit = acc / np.maximum(norms, 1e-12)[:, None]
    gravity_magnitude = float(np.median(norms))
    gyro_norms = np.linalg.norm(gyro, axis=1) if gyro is not None else None

    candidates = []
    for start in range(len(times)):
        stop = int(np.searchsorted(times, times[start] + float(window_sec), side="right"))
        if stop - start < min_samples:
            continue

        u = unit[start:stop]
        mean_dir = u.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_dir))
        if mean_norm < 1e-9:
            continue
        mean_dir = mean_dir / mean_norm

        spread_deg = float(
            np.percentile(np.degrees(np.arccos(np.clip(u @ mean_dir, -1.0, 1.0))), 90.0)
        )
        norm_rel_dev = (
            abs(float(np.median(norms[start:stop])) - gravity_magnitude)
            / max(gravity_magnitude, 1e-9)
        )
        gyro_median = float(np.median(gyro_norms[start:stop])) if gyro_norms is not None else 0.0

        # Normalized so each criterion contributes on the same scale.
        score = (
            spread_deg / max(float(max_dir_spread_deg), 1e-9)
            + norm_rel_dev / max(float(max_norm_rel_dev), 1e-9)
            + gyro_median / max(float(max_gyro), 1e-9)
        )
        passed = (
            spread_deg <= float(max_dir_spread_deg)
            and norm_rel_dev <= float(max_norm_rel_dev)
            and (gyro_norms is None or gyro_median <= float(max_gyro))
        )
        candidates.append({
            "start": int(start),
            "stop": int(stop),
            "samples": int(stop - start),
            "start_sec": float(times[start]),
            "end_sec": float(times[stop - 1]),
            "dir_spread_p90_deg": spread_deg,
            "norm_rel_dev": float(norm_rel_dev),
            "gyro_median": gyro_median,
            "score": float(score),
            "passed": bool(passed),
        })

    if not candidates:
        return None, {
            "status": "no_window",
            "reason": f"no window of {float(window_sec):.3g}s holds {min_samples} samples",
            "num_windows": 0,
            "gravity_magnitude": gravity_magnitude,
            "gyro_available": bool(gyro is not None),
        }

    passing = [c for c in candidates if c["passed"]]
    best = min(passing or candidates, key=lambda c: c["score"])
    stats = {
        "status": "stationary" if passing else "relaxed_best_effort",
        "num_windows": int(len(candidates)),
        "num_passing_windows": int(len(passing)),
        "gravity_magnitude": gravity_magnitude,
        "gyro_available": bool(gyro is not None),
        "window": {k: v for k, v in best.items() if k not in ("start", "stop")},
        "thresholds": {
            "window_sec": float(window_sec),
            "max_gyro": float(max_gyro),
            "max_dir_spread_deg": float(max_dir_spread_deg),
            "max_norm_rel_dev": float(max_norm_rel_dev),
            "min_samples": int(min_samples),
        },
    }
    return slice(best["start"], best["stop"]), stats


def match_frame_index(timestamps_txt: Path | None, window_time_sec: float) -> dict[str, Any]:
    """
    Map a segment-relative IMU time to the pose row that shares that time.

    The gravity vector lives in the camera frame, so the alignment step rotates
    it with the pose of this frame. Picking a window in the middle of the
    segment is only correct if the matching pose goes with it.
    """
    if timestamps_txt is None:
        return {"frame_index": 0, "source": "no_timestamps_use_first_pose"}
    path = Path(timestamps_txt).expanduser()
    if not path.is_file():
        return {"frame_index": 0, "source": "timestamps_missing_use_first_pose", "path": str(path)}

    try:
        rows = read_timestamps(path)
    except Exception as exc:
        return {"frame_index": 0, "source": "timestamps_unreadable_use_first_pose", "error": repr(exc)}
    if not rows:
        return {"frame_index": 0, "source": "timestamps_empty_use_first_pose"}

    frame_times = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
    index = int(np.argmin(np.abs(frame_times - float(window_time_sec))))
    return {
        "frame_index": int(min(max(index, 0), len(rows) - 1)),
        "source": "nearest_frame_timestamp",
        "window_time_sec": float(window_time_sec),
        "matched_frame_time_sec": float(frame_times[index]),
        "time_error_sec": float(abs(frame_times[index] - float(window_time_sec))),
        "frame_time_span_sec": float(frame_times[-1] - frame_times[0]),
        "num_frames": int(len(rows)),
    }


def transform_imu_to_camera(vec: np.ndarray, calibration: dict[str, Any], camera_stream_id: str) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float64).reshape(3)
    imu_ext = calibration.get("imu_extrinsics") or {}
    if "rotation" in imu_ext:
        out = np.asarray(imu_ext["rotation"], dtype=np.float64).reshape(3, 3) @ out

    to_frame = str(imu_ext.get("to_frame", "")).lower()
    camera = camera_calibration(calibration, camera_stream_id)
    camera_frame = str(camera.get("frame_id", "")).lower()
    if to_frame and camera_frame and to_frame != camera_frame:
        rectification = _find_transform(calibration, from_frame=camera_frame, to_frame=to_frame)
        if rectification is not None:
            out = np.asarray(rectification, dtype=np.float64).reshape(3, 3).T @ out
    if "depth" in to_frame and "depth" not in camera_frame and "depth_to_color" in calibration:
        out = np.asarray(calibration["depth_to_color"]["rotation"], dtype=np.float64).reshape(3, 3) @ out
    return out


def _find_transform(
    calibration: dict[str, Any],
    *,
    from_frame: str,
    to_frame: str,
) -> Any | None:
    from_frame = from_frame.lower()
    to_frame = to_frame.lower()
    for transform in calibration.get("transforms", []):
        if not isinstance(transform, dict):
            continue
        if (
            str(transform.get("from_frame", "")).lower() == from_frame
            and str(transform.get("to_frame", "")).lower() == to_frame
        ):
            if "rotation" in transform:
                return transform["rotation"]
            matrix = np.asarray(transform.get("matrix"), dtype=np.float64)
            if matrix.shape == (4, 4):
                return matrix[:3, :3]
    return None


def write_imu_gravity(
    root: Path,
    segment: dict[str, Any],
    calibration: dict[str, Any],
    camera_stream_id: str,
    out: Path,
    *,
    timestamps_txt: Path | None = None,
    window_sec: float = IMU_STATIONARY_WINDOW_SEC,
    max_gyro: float = IMU_STATIONARY_MAX_GYRO,
    max_dir_spread_deg: float = IMU_STATIONARY_MAX_DIR_SPREAD_DEG,
    max_norm_rel_dev: float = IMU_STATIONARY_MAX_NORM_REL_DEV,
) -> dict[str, Any]:
    """
    Write the gravity direction taken from the most stationary IMU window.

    Output row is "frame_index gx gy gz" in the camera frame. frame_index is the
    pose that shares the window's time, because the downstream alignment rotates
    this camera-frame vector into the world with that pose.
    """
    imu_stream = stream_by(segment, stream_id="ego_imu")
    imu_path = resolve_uri(root, str(imu_stream["uri"]))
    df = read_parquet_table(imu_path)
    acc = stacked_vector_column(df, ["ax", "ay", "az"])
    if acc is None:
        raise KeyError(f"Cannot find linear acceleration columns in {imu_path}: {table_columns(df)}")

    gyro = gyro_vector_column(df)
    times = imu_timestamps_sec(df)

    finite = np.isfinite(acc).all(axis=1)
    if gyro is not None:
        finite &= np.isfinite(gyro).all(axis=1)
    if times is not None:
        finite &= np.isfinite(times)
    acc = acc[finite]
    gyro = gyro[finite] if gyro is not None else None
    times = times[finite] if times is not None else None
    if len(acc) == 0:
        raise ValueError(f"No finite IMU acceleration rows in {imu_path}")

    whole_segment_dir = np.median(acc, axis=0)
    whole_segment_norm = float(np.linalg.norm(whole_segment_dir))
    if np.isfinite(whole_segment_norm) and whole_segment_norm > 1e-8:
        whole_segment_dir = whole_segment_dir / whole_segment_norm
    else:
        whole_segment_dir = None

    window = None
    stationary: dict[str, Any]
    if times is None:
        stationary = {"status": "no_timestamps_use_whole_segment"}
    else:
        order = np.argsort(times, kind="stable")
        times = times[order]
        acc = acc[order]
        gyro = gyro[order] if gyro is not None else None
        window, stationary = select_stationary_imu_window(
            times,
            acc,
            gyro,
            window_sec=window_sec,
            max_gyro=max_gyro,
            max_dir_spread_deg=max_dir_spread_deg,
            max_norm_rel_dev=max_norm_rel_dev,
        )

    if window is None:
        acc_used = acc
        frame_match = {"frame_index": 0, "source": "whole_segment_use_first_pose"}
        stationary.setdefault("status", "no_window_use_whole_segment")
    else:
        acc_used = acc[window]
        window_center_sec = 0.5 * (
            float(stationary["window"]["start_sec"]) + float(stationary["window"]["end_sec"])
        )
        frame_match = match_frame_index(timestamps_txt, window_center_sec)

    gravity_imu = np.median(acc_used, axis=0)
    gravity = transform_imu_to_camera(gravity_imu, calibration, camera_stream_id)
    norm = float(np.linalg.norm(gravity))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"Invalid IMU gravity vector from {imu_path}: {gravity}")
    gravity = gravity / norm

    angle_to_whole_segment_deg = None
    if whole_segment_dir is not None:
        used_dir = gravity_imu / max(float(np.linalg.norm(gravity_imu)), 1e-12)
        angle_to_whole_segment_deg = float(
            np.degrees(np.arccos(np.clip(float(used_dir @ whole_segment_dir), -1.0, 1.0)))
        )

    frame_index = int(frame_match["frame_index"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out,
        np.asarray([[float(frame_index), *gravity.tolist()]], dtype=np.float64),
        fmt="%.10f",
    )

    summary = {
        "imu_path": str(imu_path),
        "rows": int(len(acc)),
        "rows_used": int(len(acc_used)),
        "gravity_camera": gravity.tolist(),
        "frame_index": frame_index,
        "frame_match": frame_match,
        "stationary": stationary,
        "angle_to_whole_segment_median_deg": angle_to_whole_segment_deg,
    }
    print(
        f"[Real sensor] IMU gravity from {stationary.get('status')} window: "
        f"rows_used={len(acc_used)}/{len(acc)} frame_index={frame_index} "
        f"({frame_match.get('source')}) "
        f"angle_to_whole_segment_median="
        f"{'n/a' if angle_to_whole_segment_deg is None else format(angle_to_whole_segment_deg, '.3f')} deg",
        flush=True,
    )
    return summary


def read_timestamps(path: Path) -> list[dict[str, Any]]:
    table = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")
    if table.shape == ():
        table = np.asarray([table])
    names = set(table.dtype.names or [])
    rows = []
    for idx, row in enumerate(table):
        rows.append({
            "frame_id": int(row["frame_id"]) if "frame_id" in names else idx,
            "src_frame_id": int(row["src_frame_id"]) if "src_frame_id" in names else idx,
            "timestamp": float(row["timestamp"]) if "timestamp" in names else float(idx),
        })
    return rows


def choose_sample_indices(num_frames: int, num_samples: int) -> list[int]:
    if num_frames <= 0:
        return []
    if num_samples <= 0 or num_samples >= num_frames:
        return list(range(num_frames))
    return sorted(set(np.linspace(0, num_frames - 1, num_samples).round().astype(int).tolist()))


def fill_missing_raw_pose_indices(raw_id_to_pose_index: dict[int, int], ordered_depth_ids: list[int]) -> None:
    if not raw_id_to_pose_index:
        return
    known_raw_ids = np.asarray(sorted(raw_id_to_pose_index), dtype=np.int64)
    known_pose_indices = np.asarray([raw_id_to_pose_index[int(raw_id)] for raw_id in known_raw_ids], dtype=np.int64)
    for raw_id in ordered_depth_ids:
        raw_id = int(raw_id)
        if raw_id in raw_id_to_pose_index:
            continue
        insert = int(np.searchsorted(known_raw_ids, raw_id))
        if insert <= 0:
            nearest = 0
        elif insert >= len(known_raw_ids):
            nearest = len(known_raw_ids) - 1
        elif raw_id - int(known_raw_ids[insert - 1]) <= int(known_raw_ids[insert]) - raw_id:
            nearest = insert - 1
        else:
            nearest = insert
        raw_id_to_pose_index[raw_id] = int(known_pose_indices[nearest])


def load_foundation_scale_helpers():
    path = Path(__file__).resolve().with_name("04_estimate_foundationgeo_metric_scale.py")
    spec = importlib.util.spec_from_file_location("_foundation_scale_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_alignment_up(helpers, r_align_path: Path | None, align_meta_path: Path | None) -> np.ndarray:
    """Return the estimated gravity-up direction in EgoAllo Z-up world."""
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if r_align_path is None or align_meta_path is None:
        return up

    r_align_path = Path(r_align_path).expanduser().resolve()
    align_meta_path = Path(align_meta_path).expanduser().resolve()
    if not r_align_path.is_file() or not align_meta_path.is_file():
        return up

    try:
        R_align = np.load(r_align_path).astype(np.float64)
        if R_align.shape == (3, 3):
            R_final = helpers.get_R_yup_to_zup() @ R_align
            with np.load(align_meta_path, allow_pickle=True) as data:
                for key in ("up_final_da3", "up_imu_da3"):
                    if key not in data:
                        continue
                    candidate = np.asarray(data[key], dtype=np.float64).reshape(3)
                    candidate = R_final @ candidate
                    if np.isfinite(candidate).all() and np.linalg.norm(candidate) > 1e-8:
                        up = candidate
                        break
    except Exception as exc:
        print(f"[Real depth] could not read alignment up direction: {exc}; using EgoAllo Z axis", flush=True)

    norm = float(np.linalg.norm(up))
    if not np.isfinite(norm) or norm <= 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        norm = 1.0
    up = up / norm
    if np.dot(up, np.array([0.0, 0.0, 1.0], dtype=np.float64)) < 0.0:
        up = -up
    return up


def depth_to_rgb_transform(calibration: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw = calibration.get("depth_to_color") or {}
    rotation = np.asarray(raw.get("rotation", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]), dtype=np.float64).reshape(3, 3)
    translation = np.asarray(raw.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    unit = str(raw.get("translation_unit", "mm")).strip().lower()
    unit_scale = 0.001 if unit.startswith("mm") else 1.0
    return rotation, translation * unit_scale


def camera_frame_up(T_world_rgb: np.ndarray, up_world: np.ndarray) -> np.ndarray:
    """Rotate the world up direction into the RGB camera frame."""
    up_world = np.asarray(up_world, dtype=np.float64).reshape(3)
    up_world = up_world / max(float(np.linalg.norm(up_world)), 1e-12)
    R_world_rgb = np.asarray(T_world_rgb[:3, :3], dtype=np.float64)
    up_cam = R_world_rgb.T @ up_world
    return up_cam / max(float(np.linalg.norm(up_cam)), 1e-12)


def backproject_depth_to_camera_metric(
    depth: np.ndarray,
    K_depth: np.ndarray,
    R_d2c: np.ndarray,
    t_d2c: np.ndarray,
    *,
    stride: int = DEPTH_BACKPROJECT_STRIDE,
) -> np.ndarray:
    """Backproject metric depth to metric RGB-camera 3D points.

    This intentionally does not use the unscaled trajectory translation, so the
    returned coordinates are in meters and are ready for gravity-based floor
    fitting in the camera frame.
    """
    height, width = depth.shape
    stride = max(int(stride), 1)
    vv, uu = np.mgrid[0:height:stride, 0:width:stride]
    d = np.asarray(depth, dtype=np.float64)[vv, uu]
    finite = np.isfinite(d) & (d > 0.0)
    u = uu[finite].astype(np.float64)
    v = vv[finite].astype(np.float64)
    d = d[finite]
    if d.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    fx = float(K_depth[0, 0])
    fy = float(K_depth[1, 1])
    cx = float(K_depth[0, 2])
    cy = float(K_depth[1, 2])
    p_depth = np.stack(
        [(u - cx) * d / fx, (v - cy) * d / fy, d],
        axis=1,
    )
    R_d2c = np.asarray(R_d2c, dtype=np.float64).reshape(3, 3)
    t_d2c = np.asarray(t_d2c, dtype=np.float64).reshape(3)
    return (R_d2c @ p_depth.T).T + t_d2c.reshape(1, 3)


def make_plane_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(up, tmp))) > 0.9:
        tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = np.cross(up, tmp)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(up, u)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    return u, v


def robust_spread(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return 1.4826 * mad


def detect_floor_candidates_along_gravity(
    points_world: np.ndarray,
    up: np.ndarray,
    *,
    height_bin: float = FLOOR_HEIGHT_BIN,
    min_support: int = FLOOR_MIN_SUPPORT,
    min_coverage_m2: float = FLOOR_MIN_COVERAGE_M2,
    max_residual: float = FLOOR_MAX_RESIDUAL,
    max_height_rank: float = FLOOR_MAX_HEIGHT_RANK,
) -> list[dict[str, Any]]:
    """Find plausible floor levels in one depth frame.

    The normal is fixed by the already-estimated gravity/up direction, so each
    candidate is a level along that axis. Candidate scoring combines support,
    spatial coverage, low-position prior, and plane residual.
    """
    points_world = np.asarray(points_world, dtype=np.float64)
    if points_world.size == 0 or len(points_world) < int(min_support):
        return []
    points_world = points_world[np.isfinite(points_world).all(axis=1)]
    if len(points_world) < int(min_support):
        return []

    up = up / max(float(np.linalg.norm(up)), 1e-12)
    heights = points_world @ up
    heights = heights[np.isfinite(heights)]
    if len(heights) < int(min_support):
        return []

    bin_width = max(float(height_bin), 1e-3)
    hmin = float(heights.min())
    hmax = float(heights.max())
    span = hmax - hmin
    if span <= bin_width:
        bins = np.array([hmin - 0.5 * bin_width, hmax + 0.5 * bin_width], dtype=np.float64)
    else:
        bins = np.arange(hmin - 0.5 * bin_width, hmax + bin_width, bin_width)
        if bins[-1] <= hmax + 0.5 * bin_width:
            bins = np.append(bins, bins[-1] + bin_width)

    counts, edges = np.histogram(heights, bins=bins)
    if counts.size < 3:
        peak_indices = [int(np.argmax(counts))]
    else:
        smoothed = np.convolve(counts.astype(np.float64), np.array([1.0, 2.0, 1.0]), mode="same")
        peak_indices = []
        for idx in range(smoothed.size):
            left = smoothed[idx - 1] if idx > 0 else -np.inf
            right = smoothed[idx + 1] if idx + 1 < smoothed.size else -np.inf
            if smoothed[idx] >= left and smoothed[idx] > right:
                peak_indices.append(int(idx))
        if not peak_indices:
            peak_indices = [int(np.argmax(smoothed))]
        peak_indices.sort(key=lambda i: float(counts[i]), reverse=True)

    u_axis, v_axis = make_plane_basis(up)
    candidates: list[dict[str, Any]] = []
    for peak_idx in peak_indices:
        level = 0.5 * float(edges[peak_idx] + edges[peak_idx + 1])
        inlier = np.abs(heights - level) <= 1.5 * bin_width
        support = int(np.count_nonzero(inlier))
        if support < int(min_support):
            continue

        P_in = points_world[inlier]
        residual = robust_spread(P_in @ up - level)
        if not np.isfinite(residual) or residual > float(max_residual):
            continue

        u_coords = P_in @ u_axis
        v_coords = P_in @ v_axis
        q10u, q90u = np.percentile(u_coords, 10.0), np.percentile(u_coords, 90.0)
        q10v, q90v = np.percentile(v_coords, 10.0), np.percentile(v_coords, 90.0)
        coverage = float(abs(q90u - q10u) * abs(q90v - q10v))
        if coverage < float(min_coverage_m2):
            continue

        # Points clearly below the candidate are evidence that this is not the
        # low main floor. The exact fitted level naturally splits its own noisy
        # inliers, so use a margin rather than the raw signed height.
        height_rank = float(np.mean(heights < level - 1.5 * bin_width))
        if height_rank > float(max_height_rank):
            continue
        low_penalty = 0.0 if height_rank <= float(max_height_rank) else 20.0 * (height_rank - float(max_height_rank))
        support_score = 1.2 * float(np.log1p(support))
        coverage_score = 4.0 * float(np.tanh(coverage / 0.75))
        residual_penalty = 2.0 * min(residual / max(bin_width, 1e-3), 5.0)
        score = support_score + coverage_score - residual_penalty - low_penalty

        candidates.append({
            "level": level,
            "support": support,
            "coverage_m2": coverage,
            "residual": residual,
            "height_rank": height_rank,
            "score": float(score),
        })

    candidates.sort(key=lambda c: float(c["score"]), reverse=True)
    return candidates


def select_largest_median_scale_cluster(
    records: list[dict[str, Any]],
    *,
    relative_tolerance: float = DEPTH_SCALE_CLUSTER_REL_TOL,
    min_support_frames: int = DEPTH_SCALE_MIN_SUPPORT_FRAMES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the largest median implied-scale cluster among qualified floors.

    A plane's camera-space level is not constant while the camera moves. Its
    implied metric scale is: real camera height / aligned trajectory height. The
    same physical plane should therefore form a stable scale cluster over time.

    Each pose contributes at most one vote. A cluster must contain
    ``min_support_frames`` distinct poses. Among all qualified clusters the one
    with the largest median implied scale is selected.
    
    """
    relative_tolerance = float(relative_tolerance)
    min_support_frames = max(int(min_support_frames), 1)
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("depth scale cluster relative tolerance must be positive")

    valid = [
        record
        for record in records
        if np.isfinite(record.get("scale", np.nan)) and float(record["scale"]) > 0.0
    ]
    if not valid:
        raise ValueError("No positive finite implied scales found")

    log_tolerance = float(np.log1p(relative_tolerance))
    remaining = list(valid)
    clusters: list[dict[str, Any]] = []
    cluster_votes: list[list[dict[str, Any]]] = []

    while remaining:
        # Seed from the largest scale not assigned to a prior cluster. A fixed
        # seed avoids chaining nearby but physically different surfaces together.
        seed_record = max(remaining, key=lambda record: float(record["scale"]))
        seed_scale = float(seed_record["scale"])
        log_seed = float(np.log(seed_scale))
        per_pose: dict[int, tuple[tuple[float, float, float, int], dict[str, Any]]] = {}
        for record in remaining:
            scale = float(record["scale"])
            log_distance = abs(float(np.log(scale)) - log_seed)
            if log_distance > log_tolerance:
                continue
            pose_index = int(record["pose_index"])
            key = (
                log_distance,
                -float(record.get("score", 0.0)),
                -float(record.get("support", 0.0)),
                int(record.get("candidate_index", 0)),
            )
            current = per_pose.get(pose_index)
            if current is None or key < current[0]:
                per_pose[pose_index] = (key, record)

        votes = [item[1] for item in per_pose.values()]
        if not votes:
            # The seed itself is always finite and positive, so this is only a
            # defensive guard against future changes to the filtering above.
            votes = [seed_record]
        members = [
            record
            for record in remaining
            if abs(float(np.log(float(record["scale"]))) - log_seed) <= log_tolerance
        ]
        member_ids = {id(record) for record in members}
        remaining = [record for record in remaining if id(record) not in member_ids]

        vote_scales = np.asarray([float(record["scale"]) for record in votes], dtype=np.float64)
        median_scale = float(np.median(vote_scales))
        rel_mad = float(
            np.median(np.abs(vote_scales - median_scale)) / max(abs(median_scale), 1e-12)
        )
        pose_indices = sorted(int(record["pose_index"]) for record in votes)
        cluster = {
            "cluster_index": int(len(clusters)),
            "seed_scale": float(seed_scale),
            "median_scale": median_scale,
            "min_scale": float(np.min(vote_scales)),
            "max_scale": float(np.max(vote_scales)),
            "relative_mad": rel_mad,
            "supporting_frame_count": int(len(votes)),
            "supporting_frames": pose_indices,
            "candidate_record_count": int(len(members)),
            "inlier_support_sum": int(sum(int(record.get("support", 0)) for record in votes)),
            "eligible": bool(len(votes) >= min_support_frames),
            "selected": False,
        }
        clusters.append(cluster)
        cluster_votes.append(votes)

    eligible_indices = [
        index for index, cluster in enumerate(clusters) if bool(cluster["eligible"])
    ]
    if not eligible_indices:
        raise ValueError(
            f"No qualified implied-scale cluster meets {min_support_frames} distinct supporting frames"
        )

    # Among all qualified clusters, choose the largest median implied scale.
    # Use support count and lower relative MAD only as tie-breakers.
    #
    selected_index = max(
        eligible_indices,
        key=lambda index: (
            float(clusters[index]["median_scale"]),
            int(clusters[index]["supporting_frame_count"]),
            -float(clusters[index]["relative_mad"]),
        ),
    )
    clusters[selected_index]["selected"] = True
    selected_votes = cluster_votes[selected_index]
    for record in selected_votes:
        record["selected_scale_cluster"] = True

    return selected_votes, {
        "selection_policy": "largest_median_implied_scale_cluster_locally_verified",
        "relative_tolerance": relative_tolerance,
        "min_support_frames": min_support_frames,
        "num_valid_candidate_records": int(len(valid)),
        "num_clusters": int(len(clusters)),
        "selected_cluster_index": int(selected_index),
        "selected_cluster": clusters[selected_index],
        "clusters": clusters,
    }


def robust_scale_from_matched_frames(records: list[dict[str, Any]]) -> tuple[float | None, dict[str, Any]]:
    """Robustly aggregate per-frame real-depth/VGGT-height scale ratios."""
    scales = np.asarray([float(r["scale"]) for r in records if np.isfinite(r.get("scale"))], dtype=np.float64)
    if scales.size == 0:
        return None, {"reason": "no_finite_matched_frame_scales", "count": 0}

    med = float(np.median(scales))
    mad = float(np.median(np.abs(scales - med)))
    threshold = SCALE_OUTLIER_MAD * 1.4826 * mad
    keep = np.abs(scales - med) <= max(threshold, 1e-12)
    if int(np.count_nonzero(keep)) < max(3, int(scales.size // 4)):
        q25, q75 = np.percentile(scales, 25.0), np.percentile(scales, 75.0)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        keep = (scales >= lower) & (scales <= upper)
    if not np.any(keep):
        keep = np.ones(scales.size, dtype=bool)

    robust_scale = float(np.median(scales[keep]))
    return robust_scale, {
        "count": int(scales.size),
        "kept_count": int(np.count_nonzero(keep)),
        "median": med,
        "robust_scale": robust_scale,
        "relative_mad": float(mad / max(abs(med), 1e-12)),
        "min": float(np.min(scales)),
        "max": float(np.max(scales)),
    }


def collect_contiguous_local_support(
    seed_id: int,
    target_scale: float,
    local_log_tol: float,
    *,
    raw_id_to_order: dict[int, int],
    ordered_depth_ids: list[int],
    local_window: int,
    eval_raw,
) -> tuple[dict[int, dict[str, Any]], set[int], dict[str, int]]:
    """Find same-scale high-quality planes around one sampled raw depth frame."""
    support: dict[int, dict[str, Any]] = {}
    merged_ids: set[int] = set()
    counts = {"seed": 0, "backward": 0, "forward": 0, "merged": 0}

    seed_id = int(seed_id)
    seed_record = eval_raw(seed_id, target_scale, local_log_tol)
    if seed_record is not None:
        support[seed_id] = seed_record
        merged_ids.add(seed_id)
        counts["seed"] = 1

    order = raw_id_to_order.get(seed_id)
    if order is None:
        counts["merged"] = int(len(merged_ids))
        return support, merged_ids, counts

    for direction, name in ((-1, "backward"), (1, "forward")):
        for step in range(1, int(local_window) + 1):
            neighbor_index = order + direction * step
            if not 0 <= neighbor_index < len(ordered_depth_ids):
                break
            neighbor_id = int(ordered_depth_ids[neighbor_index])
            neighbor_record = eval_raw(neighbor_id, target_scale, local_log_tol)
            if neighbor_record is None:
                break
            support[neighbor_id] = neighbor_record
            merged_ids.add(neighbor_id)
            counts[name] += 1

    counts["merged"] = int(len(merged_ids))
    return support, merged_ids, counts


def write_real_depth_scale(root: Path, segment: dict[str, Any], calibration: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    try:
        stream_by(segment, stream_id=args.camera_stream_id)
    except KeyError:
        payload = {
            "status": "skipped",
            "reason": "no_depth_stream",
            "scale": None,
            "source": "real_depth",
        }
        if args.scale_json is not None:
            args.scale_json.parent.mkdir(parents=True, exist_ok=True)
            with args.scale_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    payload = write_real_depth_scale_from_floor(root, segment, calibration, args)

    if args.scale_json is not None:
        args.scale_json.parent.mkdir(parents=True, exist_ok=True)
        with args.scale_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def compact_summary_for_stdout(summary: dict[str, Any], detail_path: Path | None) -> dict[str, Any]:
    if detail_path is None or not isinstance(summary.get("metric_scale"), dict):
        return summary

    out = dict(summary)
    metric = dict(summary["metric_scale"])
    detail_log: dict[str, Any] = {"written_to": str(detail_path)}
    for key in ("frames", "matched_frames"):
        value = metric.pop(key, None)
        if isinstance(value, list):
            detail_log[f"{key}_count"] = int(len(value))
    value = metric.pop("selected_frame_indices", None)
    if isinstance(value, list):
        detail_log["selected_frame_indices_count"] = int(len(value))

    cluster = metric.get("cluster")
    if isinstance(cluster, dict) and isinstance(cluster.get("clusters"), list):
        cluster = dict(cluster)
        selected_cluster = cluster.get("selected_cluster")
        if isinstance(selected_cluster, dict) and isinstance(selected_cluster.get("supporting_frames"), list):
            selected_cluster = dict(selected_cluster)
            selected_cluster["supporting_frames"] = {
                "omitted_count": int(len(selected_cluster["supporting_frames"])),
                "written_to": str(detail_path),
            }
            cluster["selected_cluster"] = selected_cluster
        verified = cluster.get("verified_candidate")
        if isinstance(verified, dict):
            verified = dict(verified)
            for key in ("seed_raw_depth_frame_ids", "verification_support_raw_depth_frame_ids"):
                value = verified.get(key)
                if isinstance(value, list):
                    verified[key] = {
                        "omitted_count": int(len(value)),
                        "written_to": str(detail_path),
                    }
            cluster["verified_candidate"] = verified
        cluster["clusters"] = {
            "omitted_count": int(len(cluster["clusters"])),
            "written_to": str(detail_path),
        }
        metric["cluster"] = cluster

    metric["detail_log"] = detail_log
    out["metric_scale"] = metric
    return out


def write_real_depth_scale_from_floor(root: Path, segment: dict[str, Any], calibration: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    missing = [
        name for name in ("timestamps_txt", "head_trajectory_in", "point_cloud_r_align", "point_cloud_align_meta")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError("--scale-json requires " + ", ".join("--" + x.replace("_", "-") for x in missing))

    depth_stream = stream_by(segment, stream_id=args.camera_stream_id)
    depth_source = open_depth_frames(root, depth_stream)
    depth_frame_count = int(depth_source["frame_count"])
    depth_source_info = {
        "kind": depth_source["kind"],
        "path": str(depth_source["path"]),
        "array_name": depth_source["array_name"],
        "shape": depth_source["shape"],
        "frame_count": depth_frame_count,
    }

    helpers = load_foundation_scale_helpers()
    head_poses = helpers.load_head_trajectory(args.head_trajectory_in)
    timestamps = read_timestamps(args.timestamps_txt)
    up = load_alignment_up(helpers, args.point_cloud_r_align, args.point_cloud_align_meta)

    intr = camera_calibration_for_stream(calibration, depth_stream)["intrinsics"]
    K_depth = np.array(
        [[float(intr["fx"]), 0.0, float(intr["cx"])], [0.0, float(intr["fy"]), float(intr["cy"])], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    R_d2c, t_d2c = depth_to_rgb_transform(calibration)

    ordered_depth_ids = list(range(depth_frame_count))
    raw_id_to_order = {int(raw_id): idx for idx, raw_id in enumerate(ordered_depth_ids)}
    raw_id_to_pose_index = {}
    for pose_index in range(min(len(head_poses), max(len(timestamps), depth_frame_count))):
        matched_id = depth_frame_index_for_pose(pose_index, timestamps, depth_frame_count)
        if matched_id is not None and int(matched_id) not in raw_id_to_pose_index:
            raw_id_to_pose_index[int(matched_id)] = int(pose_index)
    fill_missing_raw_pose_indices(raw_id_to_pose_index, ordered_depth_ids)

    def eval_raw(raw_id, target_scale, local_log_tol):
        pose_index = raw_id_to_pose_index.get(int(raw_id))
        if pose_index is None or pose_index >= len(head_poses):
            return None
        try:
            depth_real = read_depth_frame(depth_source, raw_id, args.depth_unit_scale)
            points_cam = backproject_depth_to_camera_metric(depth_real, K_depth, R_d2c, t_d2c)
            up_cam = camera_frame_up(head_poses[pose_index], up)
            aligned_height = float(np.asarray(head_poses[pose_index][:3, 3], dtype=np.float64) @ up)
            if not np.isfinite(aligned_height) or aligned_height <= 1e-8:
                return None
            candidates = detect_floor_candidates_along_gravity(points_cam, up_cam)
            if not candidates:
                return None
            best = None
            best_key = None
            for candidate_index, candidate in enumerate(candidates):
                real_height = float(-candidate["level"])
                scale = float(real_height / aligned_height)
                if not np.isfinite(scale) or scale <= 0.0:
                    continue
                log_dist = abs(float(np.log(scale)) - float(np.log(target_scale)))
                if log_dist > local_log_tol:
                    continue
                key = (log_dist, -float(candidate.get("score", 0.0)))
                if best is None or key < best_key:
                    best = {
                        "pose_index": int(pose_index),
                        "matched_depth_frame_id": int(raw_id),
                        "candidate_index": int(candidate_index),
                        "floor_level": float(candidate["level"]),
                        "real_camera_height": real_height,
                        "aligned_vggt_height": aligned_height,
                        "scale": scale,
                        "support": int(candidate["support"]),
                        "coverage_m2": float(candidate["coverage_m2"]),
                        "residual": float(candidate["residual"]),
                        "height_rank": float(candidate["height_rank"]),
                        "score": float(candidate["score"]),
                    }
                    best_key = key
            return best
        except Exception:
            return None
    sample_indices = choose_sample_indices(
        min(len(head_poses), max(len(timestamps), depth_frame_count)),
        int(args.num_samples),
    )
    frames: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []

    for pose_index in sample_indices:
        frame: dict[str, Any] = {"pose_index": int(pose_index), "status": "failed"}
        try:
            matched_depth_frame_id = depth_frame_index_for_pose(
                pose_index, timestamps, depth_frame_count
            )
            if matched_depth_frame_id is None:
                frame["reason"] = "depth_not_found"
                frames.append(frame)
                continue

            depth_real = read_depth_frame(depth_source, matched_depth_frame_id, args.depth_unit_scale)
            points_cam = backproject_depth_to_camera_metric(
                depth_real,
                K_depth,
                R_d2c,
                t_d2c,
            )
            up_cam = camera_frame_up(head_poses[pose_index], up)
            frame["depth_finite_pixels"] = int(np.isfinite(depth_real).sum())
            frame["backprojected_points"] = int(len(points_cam))
            frame["matched_depth_frame_id"] = int(matched_depth_frame_id)

            candidates = detect_floor_candidates_along_gravity(points_cam, up_cam)
            if not candidates:
                frame["reason"] = "no_floor_candidate"
                frames.append(frame)
                continue

            aligned_vggt_height = float(np.asarray(head_poses[pose_index][:3, 3], dtype=np.float64) @ up)
            if not np.isfinite(aligned_vggt_height) or aligned_vggt_height <= 1e-8:
                frame["reason"] = "invalid_aligned_vggt_height"
                frames.append(frame)
                continue

            frame_candidates = []
            for candidate_index, candidate in enumerate(candidates):
                real_camera_height = float(-candidate["level"])
                implied_scale = float(real_camera_height / aligned_vggt_height)
                candidate_info = {
                    "candidate_index": int(candidate_index),
                    "level": float(candidate["level"]),
                    "support": int(candidate["support"]),
                    "coverage_m2": float(candidate["coverage_m2"]),
                    "residual": float(candidate["residual"]),
                    "height_rank": float(candidate["height_rank"]),
                    "score": float(candidate["score"]),
                    "real_camera_height": real_camera_height,
                    "implied_scale": implied_scale,
                    "selected_scale_cluster": False,
                }
                frame_candidates.append(candidate_info)
                if np.isfinite(implied_scale) and implied_scale > 0.0:
                    candidate_records.append({
                        "pose_index": int(pose_index),
                        "matched_depth_frame_id": int(matched_depth_frame_id),
                        "candidate_index": int(candidate_index),
                        "scale": implied_scale,
                        "level": float(candidate["level"]),
                        "support": int(candidate["support"]),
                        "coverage_m2": float(candidate["coverage_m2"]),
                        "residual": float(candidate["residual"]),
                        "height_rank": float(candidate["height_rank"]),
                        "score": float(candidate["score"]),
                    })

            frame.update({
                "status": "ok",
                "reason": "depth_floor_candidates",
                "depth_path": str(depth_source["path"]),
                "depth_match": "zarr_frame_index",
                "aligned_vggt_height": aligned_vggt_height,
                "candidates": frame_candidates,
            })
            frames.append(frame)
        except Exception as exc:
            frame.update({"reason": "exception", "exception": repr(exc)})
            frames.append(frame)

    base_inputs = {
        "episode_dir": str(root),
        "depth_source": depth_source_info,
    }
    num_frames_with_candidates = len({int(record["pose_index"]) for record in candidate_records})
    if num_frames_with_candidates < FLOOR_MIN_VALID_FRAMES:
        return {
            "status": "failed",
            "reason": "not_enough_frames_with_floor_candidates",
            "scale": None,
            "source": "real_depth",
            "scale_method": "depth_floor_implied_scale_cluster",
            "num_sampled_frames": int(len(sample_indices)),
            "num_frames_with_candidates": int(num_frames_with_candidates),
            "min_frames_with_candidates": int(FLOOR_MIN_VALID_FRAMES),
            "cluster": {
                "selection_policy": "largest_median_implied_scale_cluster_locally_verified",
                "relative_tolerance": float(args.depth_scale_cluster_rel_tol),
                "min_support_frames": int(args.depth_scale_min_support_frames),
                "num_valid_candidate_records": int(len(candidate_records)),
                "num_clusters": 0,
                "selected_cluster_index": None,
                "selected_cluster": None,
                "clusters": [],
            },
            "elapsed_sec": round(time.time() - started, 3),
            "inputs": base_inputs,
            "frames": frames,
        }

    remaining_candidate_records = list(candidate_records)
    selected_cluster_records = []
    cluster_info = {
        "selection_policy": "largest_median_implied_scale_cluster_locally_verified",
        "relative_tolerance": float(args.depth_scale_cluster_rel_tol),
        "min_support_frames": int(args.depth_scale_min_support_frames),
        "num_valid_candidate_records": int(len(candidate_records)),
        "num_clusters": 0,
        "selected_cluster_index": None,
        "selected_cluster": None,
        "clusters": [],
        "verified_candidate": None,
    }
    local_window = max(1, int(args.depth_scale_local_window))
    min_local_support = max(1, int(args.depth_scale_local_min_support_frames))
    local_log_tol = float(np.log1p(float(args.depth_scale_cluster_rel_tol)))

    while remaining_candidate_records:
        try:
            selected_records, candidate_cluster_info = select_largest_median_scale_cluster(
                remaining_candidate_records,
                relative_tolerance=float(args.depth_scale_cluster_rel_tol),
                min_support_frames=int(args.depth_scale_min_support_frames),
            )
        except ValueError as exc:
            cluster_info["selection_policy"] = "largest_median_implied_scale_cluster_locally_verified"
            cluster_info["reason"] = f"implied_scale_clustering_failed: {exc}"
            break

        candidate = candidate_cluster_info["selected_cluster"]
        target_scale = float(candidate["median_scale"])
        seed_ids = sorted({
            int(record["matched_depth_frame_id"])
            for record in selected_records
        })
        local_support: dict[int, dict[str, Any]] = {}
        verified = False
        verified_seed_id = None
        verified_support_ids: list[int] = []
        verified_support_counts = None

        for seed_id in seed_ids:
            seed_support, merged_ids, support_counts = collect_contiguous_local_support(
                seed_id,
                target_scale,
                local_log_tol,
                raw_id_to_order=raw_id_to_order,
                ordered_depth_ids=ordered_depth_ids,
                local_window=local_window,
                eval_raw=eval_raw,
            )
            local_support.update(seed_support)
            if len(merged_ids) >= min_local_support:
                verified = True
                verified_seed_id = int(seed_id)
                verified_support_ids = sorted(int(raw_id) for raw_id in merged_ids)
                verified_support_counts = support_counts
                break

        if verified:
            selected_cluster_records = list(selected_records)
            cluster_info = candidate_cluster_info
            cluster_info["selection_policy"] = "largest_median_implied_scale_cluster_locally_verified"
            cluster_info["verified_candidate"] = {
                "median_scale": target_scale,
                "seed_raw_depth_frame_ids": seed_ids,
                "verified_seed_raw_depth_frame_id": verified_seed_id,
                "verification_support_raw_depth_frame_ids": verified_support_ids,
                "verification_support_counts": verified_support_counts,
                "num_verification_support_frames": int(len(verified_support_ids)),
                "num_scale_aggregation_frames": int(len(selected_cluster_records)),
            }
            break

        failed_poses = set(candidate["supporting_frames"])
        failed_log_tol = float(np.log1p(float(args.depth_scale_cluster_rel_tol)))
        remaining_candidate_records = [
            record
            for record in remaining_candidate_records
            if not (
                int(record["pose_index"]) in failed_poses
                and abs(float(np.log(float(record["scale"]))) - float(np.log(target_scale))) <= failed_log_tol
            )
        ]
        cluster_info = candidate_cluster_info
        cluster_info["selection_policy"] = "largest_median_implied_scale_cluster_locally_verified"
        cluster_info["verified_candidate"] = None

    if not selected_cluster_records:
        return {
            "status": "failed",
            "reason": "no_scale_candidate_verified_locally",
            "scale": None,
            "source": "real_depth",
            "scale_method": "depth_floor_implied_scale_cluster",
            "num_sampled_frames": int(len(sample_indices)),
            "num_frames_with_candidates": int(num_frames_with_candidates),
            "num_candidate_records": int(len(candidate_records)),
            "cluster": cluster_info,
            "elapsed_sec": round(time.time() - started, 3),
            "inputs": base_inputs,
            "frames": frames,
        }

    selected_keys = {
        (int(record["pose_index"]), int(record.get("candidate_index", 0)))
        for record in selected_cluster_records
    }
    for frame in frames:
        pose_index = int(frame.get("pose_index", -1))
        for candidate in frame.get("candidates", []):
            candidate["selected_scale_cluster"] = (
                pose_index,
                int(candidate.get("candidate_index", 0)),
            ) in selected_keys

    matched_scales = []
    for record in selected_cluster_records:
        pose_index = int(record["pose_index"])
        matched_scales.append({
            "pose_index": pose_index,
            "matched_depth_frame_id": int(record["matched_depth_frame_id"]),
            "candidate_index": int(record.get("candidate_index", 0)),
            "floor_level": float(record.get("floor_level", record.get("level", 0.0))),
            "real_camera_height": float(record.get("real_camera_height", -record.get("level", 0.0))),
            "aligned_vggt_height": float(np.asarray(head_poses[pose_index][:3, 3], dtype=np.float64) @ up),
            "scale": float(record["scale"]),
            "support": int(record.get("support", 0)),
            "coverage_m2": float(record.get("coverage_m2", 0.0)),
            "score": float(record.get("score", 0.0)),
        })

    scale, scale_info = robust_scale_from_matched_frames(matched_scales)
    if scale is None:
        return {
            "status": "failed",
            "reason": scale_info.get("reason", "no_valid_matched_frames"),
            "scale": None,
            "source": "real_depth",
            "scale_method": "depth_floor_implied_scale_cluster",
            "num_sampled_frames": int(len(sample_indices)),
            "num_frames_with_candidates": int(num_frames_with_candidates),
            "num_candidate_records": int(len(candidate_records)),
            "elapsed_sec": round(time.time() - started, 3),
            "inputs": base_inputs,
            "frames": frames,
        }

    if not np.isfinite(scale) or scale <= DEPTH_FLOOR_MIN_SCALE or scale >= DEPTH_FLOOR_MAX_SCALE:
        return {
            "status": "failed",
            "reason": "floor_scale_out_of_range",
            "scale": None if not np.isfinite(scale) else float(scale),
            "source": "real_depth",
            "scale_method": "depth_floor_implied_scale_cluster",
            "num_sampled_frames": int(len(sample_indices)),
            "num_frames_with_candidates": int(num_frames_with_candidates),
            "num_candidate_records": int(len(candidate_records)),
            "cluster": {
                "selection_policy": "largest_median_implied_scale_cluster_locally_verified",
                "relative_tolerance": float(args.depth_scale_cluster_rel_tol),
                "min_support_frames": int(args.depth_scale_min_support_frames),
                "num_valid_candidate_records": int(len(candidate_records)),
                "num_clusters": 0,
                "selected_cluster_index": None,
                "selected_cluster": None,
                "clusters": [],
            },
            "scale_info": scale_info,
            "elapsed_sec": round(time.time() - started, 3),
            "inputs": base_inputs,
            "frames": frames,
        }

    matched_by_pose = {int(item["pose_index"]): item for item in matched_scales}
    for frame in frames:
        pose_index = int(frame.get("pose_index"))
        matched = matched_by_pose.get(pose_index)
        if matched is None:
            frame["selected_floor_level"] = None
            frame["real_camera_height"] = None
            frame["frame_scale"] = None
        else:
            frame["selected_floor_level"] = float(matched["floor_level"])
            frame["real_camera_height"] = float(matched["real_camera_height"])
            frame["aligned_vggt_height"] = float(matched["aligned_vggt_height"])
            frame["frame_scale"] = float(matched["scale"])

    return {
        "status": "ok",
        "reason": "gravity_floor_matched_vggt_frame",
        "scale": float(scale),
        "relative_mad": float(scale_info.get("relative_mad")) if scale_info.get("relative_mad") is not None else None,
        "source": "real_depth",
        "scale_method": "depth_floor_implied_scale_cluster",
        "scale_info": scale_info,
        "cluster": cluster_info,
        "num_sampled_frames": int(len(sample_indices)),
        "num_frames_with_candidates": int(num_frames_with_candidates),
        "num_candidate_records": int(len(candidate_records)),
        "num_matched_frames": int(len(matched_scales)),
        "selected_frame_indices": sorted(int(item["pose_index"]) for item in matched_scales),
        "elapsed_sec": round(time.time() - started, 3),
        "config": {
            "camera_stream_id": args.camera_stream_id,
            "depth_unit_scale": float(args.depth_unit_scale),
            "num_samples": int(args.num_samples),
            "depth_scale_local_window": int(args.depth_scale_local_window),
            "depth_scale_local_min_support_frames": int(args.depth_scale_local_min_support_frames),
            "depth_backproject_stride": int(DEPTH_BACKPROJECT_STRIDE),
            "floor_height_bin": float(FLOOR_HEIGHT_BIN),
            "floor_min_support": int(FLOOR_MIN_SUPPORT),
            "floor_min_coverage_m2": float(FLOOR_MIN_COVERAGE_M2),
            "floor_max_residual": float(FLOOR_MAX_RESIDUAL),
            "floor_max_height_rank": float(FLOOR_MAX_HEIGHT_RANK),
            "depth_scale_cluster_rel_tol": float(args.depth_scale_cluster_rel_tol),
            "depth_scale_min_support_frames": int(args.depth_scale_min_support_frames),
            "floor_min_valid_frames": int(FLOOR_MIN_VALID_FRAMES),
        },
        "inputs": base_inputs,
        "frames": frames,
        "matched_frames": matched_scales,
    }


def main() -> None:
    args = parse_args()
    root = args.episode_dir.expanduser().resolve()
    segment, calibration = load_segment(root)
    if args.scale_json is not None and args.intrinsics_out is None and args.imu_gravity_out is None:
        camera_stream_id = str(args.camera_stream_id)
        camera_selection = {
            "requested_stream_id": camera_stream_id,
            "resolved_stream_id": camera_stream_id,
            "reason": "depth_scale_uses_requested_depth_stream",
        }
    else:
        camera_stream_id, camera_selection = resolve_camera_stream_id(
            calibration,
            args.camera_stream_id,
        )
    args.camera_stream_id = camera_stream_id
    if camera_selection["reason"] != "requested_camera_found":
        print(
            "[Real sensor] camera stream auto-selected: "
            f"requested={camera_selection['requested_stream_id']!r} "
            f"resolved={camera_stream_id!r} "
            f"reason={camera_selection['reason']}",
            flush=True,
        )
    summary: dict[str, Any] = {
        "episode_dir": str(root),
        "camera_stream": camera_selection,
    }

    if args.intrinsics_out is not None:
        summary["intrinsics"] = write_intrinsics(
            calibration,
            camera_stream_id,
            args.intrinsics_out,
            args.timestamps_txt,
        )
    if args.imu_gravity_out is not None:
        try:
            stream_by(segment, stream_id="ego_imu")
        except KeyError:
            summary["imu_gravity"] = {"status": "skipped", "reason": "no_ego_imu_stream"}
        else:
            summary["imu_gravity"] = write_imu_gravity(
                root,
                segment,
                calibration,
                camera_stream_id,
                args.imu_gravity_out,
                timestamps_txt=args.timestamps_txt,
                window_sec=args.imu_stationary_window_sec,
                max_gyro=args.imu_stationary_max_gyro,
                max_dir_spread_deg=args.imu_stationary_max_dir_spread_deg,
                max_norm_rel_dev=args.imu_stationary_max_norm_rel_dev,
            )
    if args.scale_json is not None:
        summary["metric_scale"] = write_real_depth_scale(root, segment, calibration, args)

    print(json.dumps(compact_summary_for_stdout(summary, args.scale_json), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
