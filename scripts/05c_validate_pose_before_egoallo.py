#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate head height above floor before EgoAllo inference."
    )
    parser.add_argument("--head-trajectory", required=True, type=Path)
    parser.add_argument("--timestamps-txt", required=True, type=Path)
    parser.add_argument("--hamer-pkl", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--scene-ply", type=Path, default=None)
    parser.add_argument("--write-empty-on-fail", action="store_true")
    parser.add_argument("--floor-z", type=float, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    return parser.parse_args()


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.expanduser().resolve().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a YAML mapping: {path}")
    return data


def get_cfg(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def as_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def load_timestamps(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64).reshape(-1)

    table = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")
    if getattr(table, "dtype", None) is not None and table.dtype.names and "timestamp" in table.dtype.names:
        if table.shape == ():
            return np.asarray([table["timestamp"]], dtype=np.float64)
        return np.asarray(table["timestamp"], dtype=np.float64).reshape(-1)
    return np.empty((0,), dtype=np.float64)


def infer_fps_from_timestamps(timestamps: np.ndarray) -> float | None:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size < 2:
        return None
    dts = np.diff(ts)
    dts = dts[np.isfinite(dts) & (dts > 0.0)]
    if dts.size == 0:
        return None
    median_dt = float(np.median(dts))
    return float(1.0 / median_dt) if median_dt > 0.0 else None


def percentile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, q))


def nearest_index(timestamp_ns: int, timestamps_ns: np.ndarray, tolerance_ns: int) -> int | None:
    if timestamps_ns.size == 0:
        return None
    pos = int(np.searchsorted(timestamps_ns, int(timestamp_ns)))
    candidates = []
    if pos < timestamps_ns.size:
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(int(timestamps_ns[i]) - int(timestamp_ns)))
    if abs(int(timestamps_ns[best]) - int(timestamp_ns)) > tolerance_ns:
        return None
    return int(best)


def iter_detection_items(detections: Any):
    if not isinstance(detections, dict):
        return
    for timestamp_ns, det in detections.items():
        if det is None:
            continue
        if isinstance(det, list):
            det = det[0] if det else None
        if isinstance(det, dict):
            yield timestamp_ns, det


def resolve_scene_ply(args_scene_ply: Path | None, output_npz: Path, cfg: dict[str, Any]) -> Path | None:
    if args_scene_ply is not None:
        return args_scene_ply.expanduser().resolve()
    scene_name = str(get_cfg(cfg, "pose_validation.tabletop_floor.scene_ply_name", "final_scene.ply"))
    if not scene_name:
        return None
    return (output_npz.parent / scene_name).expanduser().resolve()


def load_scene_heights(
    scene_ply: Path,
    *,
    floor_z: float,
    max_points: int,
) -> np.ndarray:
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(f"open3d is required to read scene ply: {exc}") from exc

    pcd = o3d.io.read_point_cloud(str(scene_ply))
    if pcd.is_empty():
        return np.empty((0,), dtype=np.float64)
    points = np.asarray(pcd.points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(points.shape[0], size=max_points, replace=False)]
    return points[:, 2] - float(floor_z)


def collect_hand_floor_contact_heights(
    *,
    hamer_path: Path,
    timestamps: np.ndarray,
    T: np.ndarray,
    floor_z: float,
    timestamp_tolerance_sec: float,
) -> dict[str, np.ndarray]:
    if timestamps.shape[0] != T.shape[0]:
        return {
            "lowest": np.empty((0,), dtype=np.float64),
            "palm": np.empty((0,), dtype=np.float64),
            "wrist": np.empty((0,), dtype=np.float64),
        }

    with hamer_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        return {
            "lowest": np.empty((0,), dtype=np.float64),
            "palm": np.empty((0,), dtype=np.float64),
            "wrist": np.empty((0,), dtype=np.float64),
        }

    timestamps_ns = np.round(np.asarray(timestamps, dtype=np.float64) * 1e9).astype(np.int64)
    tolerance_ns = int(round(timestamp_tolerance_sec * 1e9))
    lowest_heights: list[float] = []
    palm_heights: list[float] = []
    wrist_heights: list[float] = []
    for side_key in ("detections_left_wrt_cam", "detections_right_wrt_cam"):
        for timestamp_ns_raw, det in iter_detection_items(data.get(side_key)):
            try:
                timestamp_ns = int(timestamp_ns_raw)
            except Exception:
                continue
            idx = nearest_index(timestamp_ns, timestamps_ns, tolerance_ns)
            if idx is None:
                continue
            keypoints = det.get("keypoints_3d")
            if keypoints is None:
                continue
            try:
                keypoints_np = np.asarray(keypoints, dtype=np.float64).reshape(-1, 21, 3)[0]
            except Exception:
                continue
            wrist = keypoints_np[0]
            palm = keypoints_np[[0, 5, 9, 13, 17]]
            if np.isfinite(keypoints_np).all():
                lowest_heights.append(float(np.min(keypoints_np[:, 2]) - floor_z))
            if np.isfinite(palm).all():
                palm_heights.append(float(np.mean(palm[:, 2]) - floor_z))
            if np.isfinite(wrist).all():
                wrist_heights.append(float(wrist[2] - floor_z))
    return {
        "lowest": np.asarray(lowest_heights, dtype=np.float64),
        "palm": np.asarray(palm_heights, dtype=np.float64),
        "wrist": np.asarray(wrist_heights, dtype=np.float64),
    }


def validate_tabletop_floor(
    *,
    scene_ply: Path | None,
    hamer_path: Path | None,
    T: np.ndarray,
    timestamps: np.ndarray,
    floor_z: float,
    cfg: dict[str, Any],
    failures: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    enabled = as_bool(get_cfg(cfg, "pose_validation.tabletop_floor.enabled", True), True)
    scene_near_floor_height_m = as_float(
        get_cfg(cfg, "pose_validation.tabletop_floor.scene_near_floor_height_m", 0.20),
        0.20,
    )
    scene_below_floor_tolerance_m = as_float(
        get_cfg(cfg, "pose_validation.tabletop_floor.scene_below_floor_tolerance_m", 0.05),
        0.05,
    )
    scene_near_floor_ratio_min = as_float(
        get_cfg(cfg, "pose_validation.tabletop_floor.scene_near_floor_ratio_min", 0.65),
        0.65,
    )
    hand_lowest_near_floor_height_m = as_float(
        get_cfg(cfg, "pose_validation.tabletop_floor.hand_lowest_near_floor_height_m", 0.08),
        0.08,
    )
    hand_palm_near_floor_height_m = as_float(
        get_cfg(cfg, "pose_validation.tabletop_floor.hand_palm_near_floor_height_m", 0.20),
        0.20,
    )
    hand_near_floor_ratio_min = as_float(
        get_cfg(cfg, "pose_validation.tabletop_floor.hand_near_floor_ratio_min", 0.50),
        0.50,
    )
    min_hand_check_count = as_int(
        get_cfg(cfg, "pose_validation.tabletop_floor.min_hand_check_count", 10),
        10,
    )
    timestamp_tolerance_sec = as_float(
        get_cfg(cfg, "pose_validation.tabletop_floor.timestamp_tolerance_sec", 0.02),
        0.02,
    )
    max_scene_points = as_int(
        get_cfg(cfg, "pose_validation.tabletop_floor.max_scene_points", 150000),
        150000,
    )
    summary: dict[str, Any] = {
        "enabled": enabled,
        "scene_ply": str(scene_ply) if scene_ply is not None else None,
        "thresholds": {
            "scene_near_floor_height_m": scene_near_floor_height_m,
            "scene_below_floor_tolerance_m": scene_below_floor_tolerance_m,
            "scene_near_floor_ratio_min": scene_near_floor_ratio_min,
            "hand_lowest_near_floor_height_m": hand_lowest_near_floor_height_m,
            "hand_palm_near_floor_height_m": hand_palm_near_floor_height_m,
            "hand_near_floor_ratio_min": hand_near_floor_ratio_min,
            "min_hand_check_count": min_hand_check_count,
            "timestamp_tolerance_sec": timestamp_tolerance_sec,
            "max_scene_points": max_scene_points,
        },
    }
    if not enabled:
        return summary

    if scene_ply is None or not scene_ply.is_file():
        warnings.append(
            {
                "code": "tabletop_scene_ply_missing",
                "message": "Skip tabletop floor check: aligned scene ply is missing.",
                "scene_ply": str(scene_ply) if scene_ply is not None else None,
            }
        )
        summary["status"] = "skipped"
        return summary
    if hamer_path is None or not hamer_path.expanduser().resolve().is_file():
        warnings.append(
            {
                "code": "tabletop_hamer_pkl_missing",
                "message": "Skip tabletop floor check: hamer pkl is missing.",
            }
        )
        summary["status"] = "skipped"
        return summary

    try:
        scene_heights = load_scene_heights(scene_ply, floor_z=floor_z, max_points=max_scene_points)
    except Exception as exc:
        warnings.append(
            {
                "code": "tabletop_scene_read_failed",
                "message": "Skip tabletop floor check: failed to read aligned scene ply.",
                "error": repr(exc),
            }
        )
        summary["status"] = "skipped"
        return summary

    scene_heights = scene_heights[np.isfinite(scene_heights)]
    if scene_heights.size == 0:
        warnings.append(
            {
                "code": "tabletop_scene_empty",
                "message": "Skip tabletop floor check: scene has no finite points.",
            }
        )
        summary["status"] = "skipped"
        return summary

    hand_heights = collect_hand_floor_contact_heights(
        hamer_path=hamer_path.expanduser().resolve(),
        timestamps=timestamps,
        T=T,
        floor_z=floor_z,
        timestamp_tolerance_sec=timestamp_tolerance_sec,
    )
    lowest_heights = hand_heights["lowest"][np.isfinite(hand_heights["lowest"])]
    palm_heights = hand_heights["palm"][np.isfinite(hand_heights["palm"])]
    wrist_heights = hand_heights["wrist"][np.isfinite(hand_heights["wrist"])]
    checked_hand_count = int(max(lowest_heights.size, palm_heights.size))
    if checked_hand_count < min_hand_check_count:
        warnings.append(
            {
                "code": "tabletop_hand_insufficient_data",
                "message": "Skip tabletop floor failure: not enough matched hand detections.",
                "checked_count": checked_hand_count,
                "min_hand_check_count": min_hand_check_count,
            }
        )
        summary["status"] = "insufficient_data"
        summary["checked_hand_count"] = checked_hand_count
        return summary

    scene_near_mask = (
        (scene_heights >= -scene_below_floor_tolerance_m)
        & (scene_heights <= scene_near_floor_height_m)
    )
    lowest_near_mask = (
        (lowest_heights >= -scene_below_floor_tolerance_m)
        & (lowest_heights <= hand_lowest_near_floor_height_m)
    )
    palm_near_mask = (
        (palm_heights >= -scene_below_floor_tolerance_m)
        & (palm_heights <= hand_palm_near_floor_height_m)
    )
    scene_near_ratio = float(np.mean(scene_near_mask))
    hand_lowest_near_floor_ratio = float(np.mean(lowest_near_mask)) if lowest_heights.size else 0.0
    hand_palm_near_floor_ratio = float(np.mean(palm_near_mask)) if palm_heights.size else 0.0
    hand_near_ratio = max(hand_lowest_near_floor_ratio, hand_palm_near_floor_ratio)
    scene_p95_height_m = percentile(scene_heights, 95)
    scene_height_range_p95_p05_m = None
    scene_p05 = percentile(scene_heights, 5)
    if scene_p95_height_m is not None and scene_p05 is not None:
        scene_height_range_p95_p05_m = float(scene_p95_height_m - scene_p05)

    scene_body_near_floor = scene_near_ratio >= scene_near_floor_ratio_min
    hand_long_term_near_floor = hand_near_ratio >= hand_near_floor_ratio_min
    tabletop_suspected = bool(hand_long_term_near_floor and scene_body_near_floor)

    summary.update(
        {
            "status": "checked",
            "scene_point_count": int(scene_heights.size),
            "checked_hand_count": checked_hand_count,
            "scene_near_floor_ratio": scene_near_ratio,
            "hand_near_floor_ratio": hand_near_ratio,
            "hand_lowest_near_floor_ratio": hand_lowest_near_floor_ratio,
            "hand_palm_near_floor_ratio": hand_palm_near_floor_ratio,
            "scene_p95_height_m": scene_p95_height_m,
            "scene_height_range_p95_p05_m": scene_height_range_p95_p05_m,
            "scene_body_near_floor": bool(scene_body_near_floor),
            "hand_long_term_near_floor": bool(hand_long_term_near_floor),
            "tabletop_suspected": tabletop_suspected,
            "hand_lowest_height_above_floor_m": {
                "min": percentile(lowest_heights, 0),
                "median": percentile(lowest_heights, 50),
                "p95": percentile(lowest_heights, 95),
                "max": percentile(lowest_heights, 100),
            },
            "hand_palm_height_above_floor_m": {
                "min": percentile(palm_heights, 0),
                "median": percentile(palm_heights, 50),
                "p95": percentile(palm_heights, 95),
                "max": percentile(palm_heights, 100),
            },
            "hand_wrist_height_above_floor_m": {
                "min": percentile(wrist_heights, 0),
                "median": percentile(wrist_heights, 50),
                "p95": percentile(wrist_heights, 95),
                "max": percentile(wrist_heights, 100),
            },
            "scene_height_above_floor_m": {
                "min": percentile(scene_heights, 0),
                "median": percentile(scene_heights, 50),
                "p95": scene_p95_height_m,
                "max": percentile(scene_heights, 100),
            },
        }
    )

    if tabletop_suspected:
        failures.append(
            {
                "code": "tabletop_floor_alignment_suspected",
                "message": (
                    "Hand lowest/palm points stay close to the floor plane while the "
                    "scene is concentrated near the floor; the selected floor is likely "
                    "a tabletop."
                ),
                "scene_near_floor_ratio": scene_near_ratio,
                "scene_near_floor_ratio_min": scene_near_floor_ratio_min,
                "hand_near_floor_ratio": hand_near_ratio,
                "hand_near_floor_ratio_min": hand_near_floor_ratio_min,
                "hand_lowest_near_floor_ratio": hand_lowest_near_floor_ratio,
                "hand_palm_near_floor_ratio": hand_palm_near_floor_ratio,
                "scene_p95_height_m": scene_p95_height_m,
                "checked_hand_count": checked_hand_count,
                "scene_point_count": int(scene_heights.size),
            }
        )
    return summary


def write_empty_egoallo_npz(path: Path, *, num_samples: int, fps: float | None, reason: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ns = max(int(num_samples), 1)
    payload: dict[str, np.ndarray] = {
        "Ts_world_cpf": np.empty((0, 7), dtype=np.float32),
        "Ts_world_root": np.empty((ns, 0, 7), dtype=np.float32),
        "body_quats": np.empty((ns, 0, 21, 4), dtype=np.float32),
        "left_hand_quats": np.empty((ns, 0, 15, 4), dtype=np.float32),
        "right_hand_quats": np.empty((ns, 0, 15, 4), dtype=np.float32),
        "betas": np.empty((ns, 0, 16), dtype=np.float32),
        "contacts": np.empty((ns, 0, 21), dtype=np.float32),
        "frame_nums": np.empty((0,), dtype=np.int64),
        "timestamps_ns": np.empty((0,), dtype=np.int64),
        "merge_segment_boundaries": np.empty((0,), dtype=np.int64),
        "merge_segment_lengths": np.empty((0,), dtype=np.int64),
        "merge_is_smoothed": np.asarray(False),
        "merge_boundary_blend_frames": np.asarray(0, dtype=np.int64),
        "merge_translation_smooth_window": np.asarray(0, dtype=np.int64),
        "merge_quat_lowpass_alpha": np.asarray(0.0, dtype=np.float32),
        "timestamps": np.empty((0,), dtype=np.float64),
        "fps": np.asarray(float(fps) if fps is not None else np.nan, dtype=np.float32),
        "egoallo_skipped": np.asarray(True),
        "egoallo_skip_reason": np.asarray(str(reason)),
    }
    np.savez(path, **payload)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    floor_z = float(args.floor_z) if args.floor_z is not None else as_float(
        get_cfg(cfg, "pose_validation.floor_z", get_cfg(cfg, "egoallo.floor_z", 0.0)),
        0.0,
    )
    head_p01_min_m = as_float(get_cfg(cfg, "pose_validation.head_height.p01_min_m", -0.05), -0.05)
    head_p99_max_m = as_float(get_cfg(cfg, "pose_validation.head_height.p99_max_m", 3.0), 3.0)
    head_p95_min_m = as_float(get_cfg(cfg, "pose_validation.head_height.p95_min_m", 0.60), 0.60)
    num_samples = int(args.num_samples) if args.num_samples is not None else as_int(
        get_cfg(cfg, "egoallo.num_samples", 1),
        1,
    )

    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    head_path = args.head_trajectory.expanduser().resolve()
    timestamps_path = args.timestamps_txt.expanduser().resolve()
    output_npz = args.output_npz.expanduser().resolve()
    report_json = args.report_json.expanduser().resolve()
    scene_ply = resolve_scene_ply(args.scene_ply, output_npz, cfg)

    T = np.load(head_path)
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError(f"head_trajectory.npy should be [N,4,4], got {T.shape}")

    heights = np.asarray(T[:, 2, 3], dtype=np.float64) - floor_z
    finite_heights = heights[np.isfinite(heights)]
    max_head_height_m = float(np.max(finite_heights)) if finite_heights.size else None
    min_head_height_m = float(np.min(finite_heights)) if finite_heights.size else None
    head_height_p01_m = percentile(heights, 1)
    head_height_p05_m = percentile(heights, 5)
    head_height_median_m = percentile(heights, 50)
    head_height_p95_m = percentile(heights, 95)
    head_height_p99_m = percentile(heights, 99)
    nonfinite_count = int(heights.size - finite_heights.size)
    if nonfinite_count > 0:
        failures.append(
            {
                "code": "nonfinite_head_height",
                "message": "Head height above floor contains NaN or Inf.",
                "nonfinite_count": nonfinite_count,
            }
        )
    if max_head_height_m is None:
        failures.append(
            {
                "code": "no_finite_head_height",
                "message": "No finite head height above floor was found.",
            }
        )

    if head_height_p01_m is not None and head_height_p01_m < head_p01_min_m:
        failures.append(
            {
                "code": "head_p01_height_too_low",
                "message": "P01 head height above floor is below the allowed minimum.",
                "head_height_p01_m": head_height_p01_m,
                "p01_min_m": head_p01_min_m,
            }
        )
    if head_height_p99_m is not None and head_height_p99_m > head_p99_max_m:
        failures.append(
            {
                "code": "head_p99_height_too_large",
                "message": "P99 head height above floor is above the allowed maximum.",
                "head_height_p99_m": head_height_p99_m,
                "p99_max_m": head_p99_max_m,
            }
        )
    if head_height_p95_m is not None and head_height_p95_m < head_p95_min_m:
        failures.append(
            {
                "code": "head_p95_height_too_low",
                "message": "P95 head height above floor is below the minimum required value.",
                "head_height_p95_m": head_height_p95_m,
                "p95_min_m": head_p95_min_m,
            }
        )
    metric_scale_trusted_cfg = get_cfg(cfg, "pose_validation.metric_scale_trusted", None)
    if metric_scale_trusted_cfg is None:
        metric_scale_trusted = bool(
            nonfinite_count == 0
            and head_height_p01_m is not None
            and head_height_p99_m is not None
            and head_height_p95_m is not None
            and head_height_p01_m >= head_p01_min_m
            and head_height_p99_m <= head_p99_max_m
            and head_height_p95_m >= head_p95_min_m
        )
        metric_scale_trust_source = "auto_head_height"
    else:
        metric_scale_trusted = as_bool(metric_scale_trusted_cfg, False)
        metric_scale_trust_source = "config"

    timestamps = load_timestamps(timestamps_path)
    fps = infer_fps_from_timestamps(timestamps)
    tabletop_floor_summary = validate_tabletop_floor(
        scene_ply=scene_ply,
        hamer_path=args.hamer_pkl,
        T=T,
        timestamps=timestamps,
        floor_z=floor_z,
        cfg=cfg,
        failures=failures,
        warnings=warnings,
    )
    should_run_egoallo = len(failures) == 0
    empty_written = False
    if not should_run_egoallo and args.write_empty_on_fail:
        skip_reason = str(failures[0].get("code", "pose_validation_failed")) if failures else "pose_validation_failed"
        write_empty_egoallo_npz(
            output_npz,
            num_samples=num_samples,
            fps=fps,
            reason=skip_reason,
        )
        empty_written = True

    report = {
        "status": "ok" if should_run_egoallo else "failed",
        "should_run_egoallo": should_run_egoallo,
        "failures": failures,
        "warnings": warnings,
        "check": "head_height_and_tabletop_floor",
        "head": {
            "path": str(head_path),
            "num_poses": int(T.shape[0]),
            "floor_z": floor_z,
            "height_above_floor_m": {
                "min": percentile(heights, 0),
                "p01": head_height_p01_m,
                "p05": head_height_p05_m,
                "median": head_height_median_m,
                "p95": head_height_p95_m,
                "p99": head_height_p99_m,
                "max": max_head_height_m,
                "nonfinite_count": nonfinite_count,
            },
            "metric_scale_trusted": metric_scale_trusted,
            "metric_scale_trust_source": metric_scale_trust_source,
        },
        "tabletop_floor": tabletop_floor_summary,
        "thresholds": {
            "head_height": {
                "p01_min_m": head_p01_min_m,
                "p99_max_m": head_p99_max_m,
                "p95_min_m": head_p95_min_m,
            },
        },
        "output_npz": str(output_npz),
        "empty_output_written": empty_written,
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("Pose validation:", report["status"])
    print(f"  check: {report['check']}")
    print(f"  max_head_height_m: {max_head_height_m}")
    print(f"  head_height_p95_m: {head_height_p95_m}")
    print(f"  head_height_p95_min_m: {head_p95_min_m}")
    print(f"  metric_scale_trusted: {metric_scale_trusted} ({metric_scale_trust_source})")
    if tabletop_floor_summary.get("enabled"):
        print(
            "  tabletop_floor_status: "
            f"{tabletop_floor_summary.get('status')} suspected: {tabletop_floor_summary.get('tabletop_suspected')}"
        )
    print(f"  should_run_egoallo: {should_run_egoallo}")
    if empty_written:
        print(f"  wrote empty EgoAllo output: {output_npz}")
    print(f"Saved pose validation report: {report_json}")


if __name__ == "__main__":
    main()
