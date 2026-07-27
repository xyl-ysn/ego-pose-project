#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge EgoAllo segmented NPZ outputs.

The expected input is <output_dir>/ego_tmp, where each .npz contains the
non-overlapping target crop produced by overlap inference + center crop.

By default this script only concatenates those crops and reports boundary
quality.  When --boundary-blend-frames is positive, it additionally applies a
local interpolation window around each segment boundary.  The blend is limited
to the requested boundary window so it does not globally smooth the motion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_OUTPUT_KEYS = (
    "Ts_world_cpf",
    "Ts_world_root",
    "body_quats",
    "left_hand_quats",
    "right_hand_quats",
    "betas",
    "frame_nums",
    "timestamps_ns",
)

SAMPLE_TIME_KEYS = {
    "Ts_world_root",
    "body_quats",
    "left_hand_quats",
    "right_hand_quats",
    "betas",
    "contacts",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def is_egoallo_output_npz(path: Path) -> bool:
    try:
        with np.load(path) as d:
            return all(k in d for k in EXPECTED_OUTPUT_KEYS)
    except Exception:
        return False


def start_frame(path: Path) -> int:
    with np.load(path) as d:
        return int(np.asarray(d["frame_nums"]).reshape(-1)[0])


def normalize_quat(q, eps=1e-8):
    q = np.asarray(q, dtype=np.float64)
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), eps, None)


def quat_angle_deg(q1, q2):
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), -1.0, 1.0)
    return np.degrees(2 * np.arccos(dot))


def slerp_quat(q0, q1, t, eps=1e-8):
    q0 = normalize_quat(q0, eps=eps)
    q1 = normalize_quat(q1, eps=eps)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)

    t_arr = np.asarray(t, dtype=np.float64)
    while t_arr.ndim < q0.ndim:
        t_arr = np.expand_dims(t_arr, axis=-1)

    linear = dot > 0.9995
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t_arr
    sin_theta = np.sin(theta)

    s0 = np.where(linear, 1.0 - t_arr, np.cos(theta) - dot * sin_theta / np.clip(sin_theta_0, eps, None))
    s1 = np.where(linear, t_arr, sin_theta / np.clip(sin_theta_0, eps, None))
    return normalize_quat(s0 * q0 + s1 * q1, eps=eps).astype(np.float32)


def _odd_window(window: int) -> int:
    window = int(window)
    if window <= 1:
        return 0
    return window if window % 2 == 1 else window + 1


def _window_bounds(i: int, n: int, window: int) -> tuple[int, int]:
    half = window // 2
    lo = max(0, i - half)
    hi = min(n, i + half + 1)
    return lo, hi


def moving_average_time(arr: np.ndarray, axis: int, window: int) -> np.ndarray:
    window = _odd_window(window)
    if window <= 1:
        return arr
    moved = np.moveaxis(np.asarray(arr, dtype=np.float64), axis, 0)
    out = np.empty_like(moved)
    n = moved.shape[0]
    for i in range(n):
        lo, hi = _window_bounds(i, n, window)
        out[i] = np.mean(moved[lo:hi], axis=0)
    return np.moveaxis(out, 0, axis).astype(arr.dtype, copy=False)


def median_filter_time(arr: np.ndarray, axis: int, window: int) -> np.ndarray:
    window = _odd_window(window)
    if window <= 1:
        return arr
    moved = np.moveaxis(np.asarray(arr), axis, 0)
    out = np.empty_like(moved)
    n = moved.shape[0]
    for i in range(n):
        lo, hi = _window_bounds(i, n, window)
        out[i] = np.median(moved[lo:hi], axis=0)
    return np.moveaxis(out, 0, axis).astype(arr.dtype, copy=False)


def smooth_quat_time(arr: np.ndarray, axis: int, window: int) -> np.ndarray:
    window = _odd_window(window)
    if window <= 1 or arr.shape[-1] != 4:
        return arr
    moved = np.moveaxis(normalize_quat(arr), axis, 0)
    out = np.empty_like(moved)
    n = moved.shape[0]
    for i in range(n):
        lo, hi = _window_bounds(i, n, window)
        qs = moved[lo:hi]
        ref = moved[i]
        signs = np.where(np.sum(qs * np.expand_dims(ref, axis=0), axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
        out[i] = normalize_quat(np.mean(qs * signs, axis=0))
    return np.moveaxis(out.astype(np.float32), 0, axis)


def lowpass_quat_time(arr: np.ndarray, axis: int, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    if alpha <= 0.0 or arr.shape[-1] != 4:
        return arr
    alpha = min(alpha, 1.0)
    moved = np.moveaxis(normalize_quat(arr), axis, 0)
    out = np.empty_like(moved)
    out[0] = moved[0]
    for i in range(1, moved.shape[0]):
        out[i] = slerp_quat(out[i - 1], moved[i], alpha)
    return np.moveaxis(out.astype(np.float32), 0, axis)


def _time_axis_for_array(key: str, arr: np.ndarray, num_samples: int) -> int | None:
    if key in SAMPLE_TIME_KEYS and arr.ndim >= 2 and arr.shape[0] == num_samples:
        return 1
    if key == "Ts_world_cpf" and arr.ndim >= 2:
        return 0
    return None


def _time_length(arr: np.ndarray, axis: int) -> int:
    return int(arr.shape[axis])


def _take_time(arr: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(arr, index, axis=axis)


def _assign_time_range(arr: np.ndarray, axis: int, lo: int, hi: int, values: np.ndarray) -> None:
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(lo, hi)
    arr[tuple(sl)] = values


def _interpolate_quat_time_range(arr: np.ndarray, axis: int, lo: int, hi: int) -> bool:
    if arr.shape[-1] != 4 or hi - lo < 2:
        return False
    q0 = _take_time(arr, axis, lo)
    q1 = _take_time(arr, axis, hi - 1)
    n = hi - lo
    alpha = np.linspace(0.0, 1.0, n, dtype=np.float64)
    shape = [1] * arr.ndim
    shape[axis] = n
    alpha = alpha.reshape(shape[:-1])
    values = slerp_quat(np.expand_dims(q0, axis=axis), np.expand_dims(q1, axis=axis), alpha)
    _assign_time_range(arr, axis, lo, hi, values)
    return True


def _interpolate_pose7_time_range(arr: np.ndarray, axis: int, lo: int, hi: int) -> bool:
    if arr.shape[-1] != 7 or hi - lo < 2:
        return False
    p0 = _take_time(arr, axis, lo)
    p1 = _take_time(arr, axis, hi - 1)
    n = hi - lo
    alpha_1d = np.linspace(0.0, 1.0, n, dtype=np.float64)
    quat_shape = [1] * arr.ndim
    quat_shape[axis] = n
    alpha_quat = alpha_1d.reshape(quat_shape[:-1])
    trans_shape = [1] * arr.ndim
    trans_shape[axis] = n
    alpha_trans = alpha_1d.reshape(trans_shape)

    q = slerp_quat(
        np.expand_dims(p0[..., :4], axis=axis),
        np.expand_dims(p1[..., :4], axis=axis),
        alpha_quat,
    )
    t = (1.0 - alpha_trans) * np.expand_dims(p0[..., 4:7], axis=axis) + alpha_trans * np.expand_dims(p1[..., 4:7], axis=axis)
    _assign_time_range(arr, axis, lo, hi, np.concatenate([q, t.astype(np.float32)], axis=-1))
    return True


def apply_boundary_blend(out: dict[str, np.ndarray], boundaries: list[int], blend_frames: int) -> list[dict[str, Any]]:
    if blend_frames <= 0 or not boundaries:
        return []

    num_samples = int(out["body_quats"].shape[0]) if "body_quats" in out and out["body_quats"].ndim >= 2 else 1
    blend_keys = (
        "Ts_world_root",
        "body_quats",
        "left_hand_quats",
        "right_hand_quats",
    )
    reports: list[dict[str, Any]] = []
    for boundary in boundaries:
        item: dict[str, Any] = {"boundary_index": int(boundary), "requested_frames": int(blend_frames), "applied": {}}
        for key in blend_keys:
            arr = out.get(key)
            if arr is None:
                continue
            axis = _time_axis_for_array(key, arr, num_samples)
            if axis is None:
                continue
            T = _time_length(arr, axis)
            radius = min(int(blend_frames), int(boundary), int(T - boundary))
            if radius <= 0:
                item["applied"][key] = {"frames": 0, "reason": "boundary_too_close_to_sequence_edge"}
                continue
            lo = int(boundary - radius)
            hi = int(boundary + radius)
            if arr.shape[-1] == 7:
                changed = _interpolate_pose7_time_range(arr, axis, lo, hi)
            elif arr.shape[-1] == 4:
                changed = _interpolate_quat_time_range(arr, axis, lo, hi)
            else:
                changed = False
            item["applied"][key] = {"frames": int(hi - lo) if changed else 0, "range": [lo, hi]}
        reports.append(item)
    return reports


def apply_global_smoothing(
    out: dict[str, np.ndarray],
    *,
    translation_smooth_window: int,
    quat_lowpass_alpha: float,
) -> dict[str, Any]:
    num_samples = int(out["body_quats"].shape[0]) if "body_quats" in out and out["body_quats"].ndim >= 2 else 1
    report: dict[str, Any] = {
        "translation_smooth_window": int(translation_smooth_window),
        "quat_lowpass_alpha": float(quat_lowpass_alpha),
        "applied": {},
    }

    root = out.get("Ts_world_root")
    if root is not None:
        axis = _time_axis_for_array("Ts_world_root", root, num_samples)
        if axis is not None and root.shape[-1] == 7:
            changed = False
            smoothed = root.copy()
            if _odd_window(translation_smooth_window) > 1:
                trans = np.take(smoothed, indices=range(4, 7), axis=smoothed.ndim - 1)
                smoothed[..., 4:7] = moving_average_time(trans, axis, translation_smooth_window)
                changed = True
            if quat_lowpass_alpha > 0.0:
                smoothed[..., :4] = lowpass_quat_time(smoothed[..., :4], axis, quat_lowpass_alpha)
                changed = True
            if changed:
                out["Ts_world_root"] = smoothed.astype(root.dtype, copy=False)
            report["applied"]["Ts_world_root"] = {
                "changed": bool(changed),
                "time_axis": int(axis),
                "frames": int(_time_length(root, axis)),
            }

    body = out.get("body_quats")
    if body is not None:
        axis = _time_axis_for_array("body_quats", body, num_samples)
        changed = axis is not None and quat_lowpass_alpha > 0.0
        if changed:
            out["body_quats"] = lowpass_quat_time(body, axis, quat_lowpass_alpha).astype(body.dtype, copy=False)
        report["applied"]["body_quats"] = {"changed": bool(changed), "alpha": float(quat_lowpass_alpha)}

    for key in ("left_hand_quats", "right_hand_quats"):
        arr = out.get(key)
        if arr is None:
            continue
        axis = _time_axis_for_array(key, arr, num_samples)
        changed = axis is not None and quat_lowpass_alpha > 0.0
        if changed:
            out[key] = lowpass_quat_time(arr, axis, quat_lowpass_alpha).astype(arr.dtype, copy=False)
        report["applied"][key] = {"changed": bool(changed), "alpha": float(quat_lowpass_alpha)}

    return report


def load_segments(input_dir: Path):
    paths = [p for p in input_dir.glob("*.npz") if is_egoallo_output_npz(p)]
    if not paths:
        raise FileNotFoundError(f"No EgoAllo output npz files in {input_dir}")
    paths = sorted(paths, key=lambda p: (start_frame(p), p.name))

    segs: list[dict[str, np.ndarray]] = []
    segment_info = []
    for p in paths:
        with np.load(p) as d:
            seg = {k: np.asarray(d[k]).copy() for k in d.files}
        frames = np.asarray(seg["frame_nums"]).reshape(-1)
        if frames.size == 0:
            raise ValueError(f"Empty frame_nums in {p}")
        segs.append(seg)
        segment_info.append(
            {
                "path": str(p),
                "name": p.name,
                "start_frame": int(frames[0]),
                "end_frame_inclusive": int(frames[-1]),
                "end_frame_exclusive": int(frames[-1]) + 1,
                "length": int(frames.size),
                "infer_start_index": int(np.asarray(seg.get("infer_start_index", frames[0])).reshape(-1)[0]),
                "infer_end_index": int(np.asarray(seg.get("infer_end_index", frames[-1] + 1)).reshape(-1)[0]),
                "crop_offset": int(np.asarray(seg.get("crop_offset", 0)).reshape(-1)[0]),
            }
        )
    return segs, paths, segment_info


def concat_segments(segs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not segs:
        raise ValueError("No segments to concatenate")
    num_samples = segs[0]["body_quats"].shape[0]
    out: dict[str, np.ndarray] = {}
    all_keys = set().union(*(s.keys() for s in segs))
    for k in sorted(all_keys):
        if not all(k in s for s in segs):
            continue
        arrs = [s[k] for s in segs]
        # Scalar metadata such as infer_start_index cannot be concatenated usefully.
        if arrs[0].ndim == 0:
            continue
        axis = 1 if (k in SAMPLE_TIME_KEYS and arrs[0].ndim >= 2 and arrs[0].shape[0] == num_samples) else 0
        out[k] = np.concatenate(arrs, axis=axis)
    return out


def boundaries_from_segment_info(segment_info: list[dict[str, Any]]) -> list[int]:
    boundaries = []
    total = 0
    for info in segment_info[:-1]:
        total += int(info["length"])
        boundaries.append(total)
    return boundaries


def frame_continuity_report(segment_info: list[dict[str, Any]]) -> dict[str, Any]:
    issues = []
    for prev, cur in zip(segment_info[:-1], segment_info[1:]):
        prev_end = int(prev["end_frame_exclusive"])
        cur_start = int(cur["start_frame"])
        if cur_start != prev_end:
            issues.append(
                {
                    "prev_segment": prev["name"],
                    "next_segment": cur["name"],
                    "prev_end_exclusive": prev_end,
                    "next_start": cur_start,
                    "gap_or_overlap": cur_start - prev_end,
                }
            )
    return {
        "continuous": len(issues) == 0,
        "issues": issues,
    }


def boundary_quality(out: dict[str, np.ndarray], boundaries: list[int]) -> list[dict[str, Any]]:
    reports = []
    root = out.get("Ts_world_root")
    cpf = out.get("Ts_world_cpf")
    body = out.get("body_quats")
    left = out.get("left_hand_quats")
    right = out.get("right_hand_quats")

    for b in boundaries:
        item: dict[str, Any] = {"boundary_index": int(b), "transition": f"{b-1}->{b}"}
        if root is not None and 0 < b < (root.shape[1] if root.ndim == 3 else root.shape[0]):
            a, c = (root[0, b - 1], root[0, b]) if root.ndim == 3 else (root[b - 1], root[b])
            item["root_translation_jump_m"] = float(np.linalg.norm(c[4:7] - a[4:7]))
            item["root_rotation_jump_deg"] = float(quat_angle_deg(a[:4], c[:4]))
        if cpf is not None and 0 < b < cpf.shape[0]:
            a, c = cpf[b - 1], cpf[b]
            item["cpf_translation_jump_m"] = float(np.linalg.norm(c[4:7] - a[4:7]))
            item["cpf_rotation_jump_deg"] = float(quat_angle_deg(a[:4], c[:4]))
        for name, arr in (("body", body), ("left_hand", left), ("right_hand", right)):
            if arr is None:
                continue
            T = arr.shape[1] if arr.ndim == 4 else arr.shape[0]
            if not (0 < b < T):
                continue
            a, c = (arr[0, b - 1], arr[0, b]) if arr.ndim == 4 else (arr[b - 1], arr[b])
            jumps = quat_angle_deg(a, c)
            item[f"{name}_quat_jump_deg_mean"] = float(np.mean(jumps))
            item[f"{name}_quat_jump_deg_p90"] = float(np.percentile(jumps, 90))
            item[f"{name}_quat_jump_deg_max"] = float(np.max(jumps))
        reports.append(item)
    return reports


def infer_fps_from_timestamps_ns(timestamps_ns: np.ndarray) -> float | None:
    ts = np.asarray(timestamps_ns, dtype=np.float64).reshape(-1)
    if ts.size < 2:
        return None
    dts = np.diff(ts)
    if np.any(dts <= 0):
        return None
    dts_sec = dts / 1e9
    median_dt = float(np.median(dts_sec))
    if median_dt <= 0.0:
        return None
    return float(1.0 / median_dt)


def print_report(segment_info, continuity, boundary_reports):
    print("Loaded segments:")
    for info in segment_info:
        print(
            f"  {info['name']}: {info['start_frame']}->{info['end_frame_inclusive']}, "
            f"len={info['length']}, infer={info['infer_start_index']}-{info['infer_end_index']}, "
            f"crop_offset={info['crop_offset']}"
        )
    if continuity["continuous"]:
        print("Frame continuity: OK")
    else:
        print("[Warning] Frame continuity issues:")
        for issue in continuity["issues"]:
            print("  ", issue)
    if boundary_reports:
        print("Boundary quality:")
        for r in boundary_reports:
            msg = f"  {r['transition']}"
            if "root_translation_jump_m" in r:
                msg += f" | root Δt={r['root_translation_jump_m']:.4f}m ΔR={r['root_rotation_jump_deg']:.2f}deg"
            if "body_quat_jump_deg_mean" in r:
                msg += f" | body mean/max={r['body_quat_jump_deg_mean']:.2f}/{r['body_quat_jump_deg_max']:.2f}deg"
            print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--output-name", default="merged_smoothed_vis.npz")
    ap.add_argument("--report-json", type=Path, default=None)
    ap.add_argument("--print-stats", action="store_true")
    ap.add_argument("--strict-frame-continuity", action="store_true")

    ap.add_argument("--boundary-blend-frames", type=int, default=0)
    ap.add_argument("--translation-smooth-window", type=int, default=0)
    ap.add_argument("--quat-lowpass-alpha", type=float, default=0.0)
    args = ap.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    segs, paths, segment_info = load_segments(input_dir)
    out = concat_segments(segs)
    boundaries = boundaries_from_segment_info(segment_info)
    continuity = frame_continuity_report(segment_info)
    boundary_reports_before = boundary_quality(out, boundaries)
    boundary_blend_report = apply_boundary_blend(out, boundaries, int(args.boundary_blend_frames))
    global_smoothing_report = apply_global_smoothing(
        out,
        translation_smooth_window=int(args.translation_smooth_window),
        quat_lowpass_alpha=float(args.quat_lowpass_alpha),
    )
    boundary_reports_after = boundary_quality(out, boundaries)

    if args.print_stats:
        print_report(segment_info, continuity, boundary_reports_after)

    if args.strict_frame_continuity and not continuity["continuous"]:
        raise ValueError(f"Frame continuity check failed: {continuity['issues']}")

    out["merge_segment_boundaries"] = np.asarray(boundaries, dtype=np.int64)
    out["merge_segment_lengths"] = np.asarray([int(x["length"]) for x in segment_info], dtype=np.int64)
    any_global_smoothing = _odd_window(int(args.translation_smooth_window)) > 1 or float(args.quat_lowpass_alpha) > 0.0
    out["merge_is_smoothed"] = np.asarray(int(args.boundary_blend_frames) > 0 or any_global_smoothing)
    out["merge_boundary_blend_frames"] = np.asarray(int(args.boundary_blend_frames), dtype=np.int64)
    out["merge_translation_smooth_window"] = np.asarray(int(args.translation_smooth_window), dtype=np.int64)
    out["merge_quat_lowpass_alpha"] = np.asarray(float(args.quat_lowpass_alpha), dtype=np.float32)

    if "timestamps_ns" in out:
        timestamps_ns = np.asarray(out["timestamps_ns"], dtype=np.int64).reshape(-1)
        out["timestamps"] = timestamps_ns.astype(np.float64) / 1e9
        fps = infer_fps_from_timestamps_ns(timestamps_ns)
        if fps is not None:
            out["fps"] = np.asarray(float(fps), dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.output_name
    np.savez(out_path, **out)

    report = {
        "status": "ok" if continuity["continuous"] else "warning",
        "mode": "boundary_blend_global_smoothing"
        if int(args.boundary_blend_frames) > 0 and any_global_smoothing
        else "global_smoothing"
        if any_global_smoothing
        else "boundary_blend"
        if int(args.boundary_blend_frames) > 0
        else "merge_only_no_smoothing",
        "input_dir": str(input_dir),
        "output_npz": str(out_path),
        "num_segments": len(segment_info),
        "segments": segment_info,
        "frame_continuity": continuity,
        "boundaries": boundaries,
        "boundary_quality_before": boundary_reports_before,
        "boundary_quality": boundary_reports_after,
        "boundary_blend": {
            "boundary_blend_frames": int(args.boundary_blend_frames),
            "applied": boundary_blend_report,
        },
        "global_smoothing": global_smoothing_report,
    }

    report_json = args.report_json
    if report_json is not None:
        report_json = report_json.expanduser().resolve()
        report_json.parent.mkdir(parents=True, exist_ok=True)
        with report_json.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(report), f, ensure_ascii=False, indent=2)
        print(f"Saved merge quality report: {report_json}")

    print(f"Saved merged EgoAllo output: {out_path}")
    if int(args.boundary_blend_frames) > 0 or any_global_smoothing:
        print(
            "Mode: "
            f"boundary_blend={int(args.boundary_blend_frames)}, "
            f"translation_window={int(args.translation_smooth_window)}, "
            f"quat_lowpass_alpha={float(args.quat_lowpass_alpha):.3f}"
        )
    else:
        print("Mode: merge-only; no smoothing was applied.")


if __name__ == "__main__":
    main()
