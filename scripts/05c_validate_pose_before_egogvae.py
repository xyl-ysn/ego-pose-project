#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate head height above floor before EgoGVAE inference."
    )
    parser.add_argument("--head-trajectory", required=True, type=Path)
    parser.add_argument("--timestamps-txt", required=True, type=Path)
    parser.add_argument("--hamer-pkl", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
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


def write_empty_egogvae_npz(path: Path, *, num_samples: int, fps: float | None, reason: str) -> None:
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
        "egogvae_skipped": np.asarray(True),
        "egogvae_skip_reason": np.asarray(str(reason)),
    }
    np.savez(path, **payload)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    floor_z = float(args.floor_z) if args.floor_z is not None else as_float(
        get_cfg(cfg, "pose_validation.floor_z", get_cfg(cfg, "egogvae.floor_z", 0.0)),
        0.0,
    )
    head_p01_min_m = as_float(get_cfg(cfg, "pose_validation.head_height.p01_min_m", -0.05), -0.05)
    head_p99_max_m = as_float(get_cfg(cfg, "pose_validation.head_height.p99_max_m", 3.0), 3.0)
    head_p95_min_m = as_float(get_cfg(cfg, "pose_validation.head_height.p95_min_m", 0.60), 0.60)
    num_samples = int(args.num_samples) if args.num_samples is not None else as_int(
        get_cfg(cfg, "egogvae.num_samples", 1),
        1,
    )

    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    head_path = args.head_trajectory.expanduser().resolve()
    timestamps_path = args.timestamps_txt.expanduser().resolve()
    output_npz = args.output_npz.expanduser().resolve()
    report_json = args.report_json.expanduser().resolve()

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
    should_run_egogvae = len(failures) == 0
    empty_written = False
    if not should_run_egogvae and args.write_empty_on_fail:
        skip_reason = str(failures[0].get("code", "pose_validation_failed")) if failures else "pose_validation_failed"
        write_empty_egogvae_npz(
            output_npz,
            num_samples=num_samples,
            fps=fps,
            reason=skip_reason,
        )
        empty_written = True

    report = {
        "status": "ok" if should_run_egogvae else "failed",
        "should_run_egogvae": should_run_egogvae,
        "failures": failures,
        "warnings": warnings,
        "check": "head_height",
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
    print(f"  should_run_egogvae: {should_run_egogvae}")
    if empty_written:
        print(f"  wrote empty EgoGVAE output: {output_npz}")
    print(f"Saved pose validation report: {report_json}")


if __name__ == "__main__":
    main()
