#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare HaWoR hand detections for visualization.

Compared with the original 02_extract_hawor_world_hands.py, this version adds:
  1) soft-start / soft-end weights for hand segments,
  2) optional geometry soft-start from a previous same-side hand anchor,
  3) scale-free wrist-jump filtering for non-metric VGGT / HaWoR coordinates,
  4) segment-level cleanup and translation-only smoothing for visualization.

The output JSONL keeps the same top-level structure:
  frame_id, timestamp, timestamp_ns, src_frame_id, hands

Each output hand keeps the original array fields and additionally contains:
  raw_confidence
  visualization_alpha / visualization_weight
  constraint_weight
  soft_start_alpha / soft_start_weight
  soft_end_alpha / soft_end_weight
  segment_start_frame / segment_end_frame / segment_length / segment_frame_index

Important:
  For true soft-start behavior, the downstream visualizer/fusion code should use
  visualization_alpha / visualization_weight / constraint_weight, or at least the
  updated confidence field. If downstream ignores all confidence/weight fields,
  no JSONL-only preprocessing can make the first ever hand segment fully soft;
  in that case use --min-output-alpha or --trim-segment-edges to omit early frames.
"""

import sys

import argparse
import copy
import json
from pathlib import Path

import numpy as np


ARRAY_FIELDS = (
    "camera_keypoints_3d",
    "hawor_camera_keypoints_3d",
    "world_keypoints_3d",
    "world_mesh_vertices_3d",
)

# Each tuple: reference keypoint field, fields living in the same coordinate frame.
# Translation-only smoothing and anchor-based soft-start use these groups so that
# hand shape/orientation is preserved while the global wrist trajectory is smoothed.
COORD_GROUPS = (
    ("world_keypoints_3d", ("world_keypoints_3d", "world_mesh_vertices_3d")),
    ("camera_keypoints_3d", ("camera_keypoints_3d",)),
    ("hawor_camera_keypoints_3d", ("hawor_camera_keypoints_3d",)),
)


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
        int(row["frame_id"]): (float(row["timestamp"]), int(row["src_frame_id"]))
        for row in table
    }


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_line_no"] = line_no
            yield rec


def pick_frame_id(rec, frame_key):
    # Default frame_key is extracted_idx. Fallback keeps compatibility with old files.
    if frame_key in rec:
        return int(rec[frame_key])
    for key in ("extracted_idx", "frame_id", "idx"):
        if key in rec:
            return int(rec[key])
    raise KeyError(f"Line {rec.get('_line_no')}: cannot find frame id field")


def pick_src_frame_id(rec, frame_id, timestamp_map):
    if "original_frame_idx" in rec:
        return int(rec["original_frame_idx"])
    if "src_frame_id" in rec:
        return int(rec["src_frame_id"])
    if frame_id in timestamp_map:
        return int(timestamp_map[frame_id][1])
    return frame_id


def pick_timestamp(rec, frame_id, timestamp_map):
    if "timestamp" in rec:
        return float(rec["timestamp"])
    if "timestamp_sec" in rec:
        return float(rec["timestamp_sec"])
    if "timestamp_ns" in rec:
        return float(rec["timestamp_ns"]) / 1e9
    if frame_id in timestamp_map:
        return float(timestamp_map[frame_id][0])
    raise KeyError(
        f"Line {rec.get('_line_no')}: cannot find timestamp; pass --timestamps-txt"
    )


def require_points(hand, key, hand_idx, line_no):
    if key not in hand:
        raise KeyError(f"Line {line_no}, hand {hand_idx}: missing {key}")
    arr = np.asarray(hand[key], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Line {line_no}, hand {hand_idx}: {key} must have shape (N,3), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Line {line_no}, hand {hand_idx}: {key} contains NaN/Inf")
    return arr


def optional_points(hand, key, hand_idx, line_no):
    if key not in hand or hand[key] is None:
        return None
    return require_points(hand, key, hand_idx, line_no)


def finite_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def hand_confidence(hand):
    for key in ("confidence", "score", "hand_score", "det_score"):
        if key in hand:
            return finite_float(hand[key], default=0.0)
    return 0.0


def raw_confidence_for_output(hand):
    raw = finite_float(hand.get("raw_confidence", hand.get("confidence", 1.0)), default=1.0)
    # Some HaWoR exports have no score. Keep alpha meaningful in that case.
    if raw <= 0.0:
        return 1.0
    return raw


def select_one_hand_per_side(hands):
    selected = {}
    for hand in hands:
        side = hand["side"]
        if side not in selected or hand_confidence(hand) > hand_confidence(selected[side]):
            selected[side] = hand
    return selected


def copy_hand(hand):
    out = {}
    for key, value in hand.items():
        if key in ARRAY_FIELDS and value is not None:
            out[key] = np.asarray(value, dtype=np.float32).copy()
        else:
            out[key] = copy.deepcopy(value)
    return out


def valid_points(value):
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] == 3 and np.all(np.isfinite(arr)):
        return arr
    return None


def wrist_position_from_field(hand, field):
    if hand is None:
        return None
    arr = valid_points(hand.get(field))
    if arr is None:
        return None
    return arr[0]


def wrist_position(hand):
    if hand is None:
        return None
    for field in ("world_keypoints_3d", "camera_keypoints_3d"):
        wrist = wrist_position_from_field(hand, field)
        if wrist is not None:
            return wrist
    return None


def hand_size_from_keypoints(points, eps=1e-8):
    points = valid_points(points)
    if points is None:
        return None
    wrist = points[0]
    # Common 21-keypoint convention: 9 is middle MCP. This is a stable scale proxy.
    if points.shape[0] > 9:
        size = float(np.linalg.norm(points[9] - wrist))
        if size > eps:
            return size
    # Fallback: robust median wrist-to-joint distance.
    if points.shape[0] <= 1:
        return None
    dists = np.linalg.norm(points[1:] - wrist[None, :], axis=1)
    dists = dists[dists > eps]
    if len(dists) == 0:
        return None
    return float(np.median(dists))


def wrist_and_hand_size(hand):
    """Return wrist and hand scale in the same coordinate frame."""
    if hand is None:
        return None, None
    for field in ("world_keypoints_3d", "camera_keypoints_3d"):
        arr = valid_points(hand.get(field))
        if arr is None:
            continue
        size = hand_size_from_keypoints(arr)
        if size is not None:
            return arr[0], size
    return None, None


def iter_segments(track):
    idx = 0
    n = len(track)
    while idx < n:
        if track[idx] is None:
            idx += 1
            continue
        start = idx
        while idx < n and track[idx] is not None:
            idx += 1
        yield start, idx - 1


def drop_isolated_detections(track):
    dropped = 0
    original = list(track)
    for idx, hand in enumerate(original):
        if hand is None:
            continue
        has_prev = idx > 0 and original[idx - 1] is not None
        has_next = idx + 1 < len(original) and original[idx + 1] is not None
        if not has_prev and not has_next:
            track[idx] = None
            dropped += 1
    return dropped


def drop_short_segments(track, min_track_len):
    if min_track_len <= 1:
        return 0
    dropped = 0
    for start, end in list(iter_segments(track)):
        if end - start + 1 >= min_track_len:
            continue
        for idx in range(start, end + 1):
            if track[idx] is not None:
                track[idx] = None
                dropped += 1
    return dropped


def trim_segment_edges(track, trim):
    if trim <= 0:
        return 0
    dropped = 0
    for start, end in list(iter_segments(track)):
        length = end - start + 1
        if length <= 2 * trim:
            lo, hi = start, end
            ranges = [(lo, hi)]
        else:
            ranges = [(start, start + trim - 1), (end - trim + 1, end)]
        for lo, hi in ranges:
            for idx in range(lo, hi + 1):
                if track[idx] is not None:
                    track[idx] = None
                    dropped += 1
    return dropped


def drop_large_wrist_jumps_raw(track, max_wrist_step):
    if max_wrist_step <= 0:
        return 0
    dropped = 0
    prev_idx = None
    prev_wrist = None
    for idx, hand in enumerate(track):
        wrist = wrist_position(hand)
        if wrist is None:
            continue
        if prev_wrist is not None:
            frame_gap = max(1, idx - prev_idx)
            if float(np.linalg.norm(wrist - prev_wrist)) > max_wrist_step * frame_gap:
                track[idx] = None
                dropped += 1
                continue
        prev_idx = idx
        prev_wrist = wrist
    return dropped


def drop_large_wrist_jumps_relative(track, max_wrist_step_hand_ratio):
    if max_wrist_step_hand_ratio <= 0:
        return 0
    dropped = 0
    prev_idx = None
    prev_wrist = None
    prev_size = None
    for idx, hand in enumerate(track):
        wrist, size = wrist_and_hand_size(hand)
        if wrist is None or size is None:
            continue
        if prev_wrist is not None and prev_size is not None:
            frame_gap = max(1, idx - prev_idx)
            ref_size = max(1e-8, 0.5 * (float(size) + float(prev_size)))
            jump_ratio_per_frame = float(np.linalg.norm(wrist - prev_wrist)) / ref_size / frame_gap
            if jump_ratio_per_frame > max_wrist_step_hand_ratio:
                track[idx] = None
                dropped += 1
                continue
        prev_idx = idx
        prev_wrist = wrist
        prev_size = size
    return dropped


def collect_wrist_jump_ratios(track):
    ratios = []
    prev_idx = None
    prev_wrist = None
    prev_size = None
    for idx, hand in enumerate(track):
        wrist, size = wrist_and_hand_size(hand)
        if wrist is None or size is None:
            continue
        if prev_wrist is not None and prev_size is not None:
            frame_gap = max(1, idx - prev_idx)
            ref_size = max(1e-8, 0.5 * (float(size) + float(prev_size)))
            ratios.append(float(np.linalg.norm(wrist - prev_wrist)) / ref_size / frame_gap)
        prev_idx = idx
        prev_wrist = wrist
        prev_size = size
    return ratios


def endpoint_jump_ratio_per_frame(left, right, gap):
    lw, ls = wrist_and_hand_size(left)
    rw, rs = wrist_and_hand_size(right)
    if lw is None or rw is None or ls is None or rs is None:
        return None
    ref_size = max(1e-8, 0.5 * (float(ls) + float(rs)))
    return float(np.linalg.norm(rw - lw)) / ref_size / max(1, gap + 1)


def interpolate_hand(left, right, alpha, gap_size):
    left_conf = raw_confidence_for_output(left)
    right_conf = raw_confidence_for_output(right)
    out = {
        "side": left["side"],
        "confidence": float((1.0 - alpha) * left_conf + alpha * right_conf),
        "raw_confidence": float((1.0 - alpha) * left_conf + alpha * right_conf),
        "interpolated": True,
        "gap_size": int(gap_size),
    }
    for field in ARRAY_FIELDS:
        a = left.get(field)
        b = right.get(field)
        if a is None or b is None:
            out[field] = None
            continue
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.shape != b.shape:
            out[field] = None
            continue
        out[field] = ((1.0 - alpha) * a + alpha * b).astype(np.float32)
    return out


def fill_short_gaps(track, max_gap_fill, max_endpoint_jump_ratio=0.0):
    if max_gap_fill <= 0:
        return 0
    filled = 0
    valid_indices = [i for i, hand in enumerate(track) if hand is not None]
    for left_idx, right_idx in zip(valid_indices[:-1], valid_indices[1:]):
        gap = right_idx - left_idx - 1
        if gap <= 0 or gap > max_gap_fill:
            continue
        left = track[left_idx]
        right = track[right_idx]
        if left["side"] != right["side"]:
            continue
        if max_endpoint_jump_ratio > 0:
            ratio = endpoint_jump_ratio_per_frame(left, right, gap)
            if ratio is not None and ratio > max_endpoint_jump_ratio:
                continue
        for offset in range(1, gap + 1):
            alpha = offset / float(gap + 1)
            track[left_idx + offset] = interpolate_hand(left, right, alpha, gap)
            filled += 1
    return filled


def weighted_average(arrays, weights):
    w = np.asarray(weights, dtype=np.float32)
    s = float(np.sum(w))
    if s <= 0.0:
        return np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
    w /= s
    return np.tensordot(w, np.stack(arrays, axis=0), axes=(0, 0)).astype(np.float32)


def smooth_track_full(track, smooth_window):
    if smooth_window <= 1:
        return 0
    radius = smooth_window // 2
    smoothed = [copy_hand(hand) if hand is not None else None for hand in track]
    changed = 0
    for idx, hand in enumerate(track):
        if hand is None:
            continue
        lo = max(0, idx - radius)
        hi = min(len(track), idx + radius + 1)
        for field in ARRAY_FIELDS:
            value = hand.get(field)
            if value is None:
                continue
            shape = np.asarray(value).shape
            arrays = []
            weights = []
            for nbr_idx in range(lo, hi):
                nbr = track[nbr_idx]
                if nbr is None or nbr.get(field) is None:
                    continue
                arr = np.asarray(nbr[field], dtype=np.float32)
                if arr.shape == shape and np.all(np.isfinite(arr)):
                    arrays.append(arr)
                    weights.append(radius + 1 - abs(nbr_idx - idx))
            if len(arrays) > 1:
                smoothed[idx][field] = weighted_average(arrays, weights)
                changed += 1
    track[:] = smoothed
    return changed


def smooth_track_translation(track, smooth_window):
    """Smooth only global wrist translation; preserve local hand shape/orientation."""
    if smooth_window <= 1:
        return 0
    radius = smooth_window // 2
    smoothed = [copy_hand(hand) if hand is not None else None for hand in track]
    changed = 0
    for idx, hand in enumerate(track):
        if hand is None:
            continue
        lo = max(0, idx - radius)
        hi = min(len(track), idx + radius + 1)
        for ref_field, target_fields in COORD_GROUPS:
            ref_wrist = wrist_position_from_field(hand, ref_field)
            if ref_wrist is None:
                continue
            wrists = []
            weights = []
            for nbr_idx in range(lo, hi):
                nbr = track[nbr_idx]
                if nbr is None:
                    continue
                nbr_wrist = wrist_position_from_field(nbr, ref_field)
                if nbr_wrist is None:
                    continue
                wrists.append(nbr_wrist)
                weights.append(radius + 1 - abs(nbr_idx - idx))
            if len(wrists) <= 1:
                continue
            smooth_wrist = weighted_average(wrists, weights)
            delta = smooth_wrist - ref_wrist
            if not np.all(np.isfinite(delta)):
                continue
            for field in target_fields:
                arr = valid_points(smoothed[idx].get(field))
                if arr is None:
                    continue
                smoothed[idx][field] = (arr + delta[None, :]).astype(np.float32)
                changed += 1
    track[:] = smoothed
    return changed


def smooth_track(track, smooth_window, smooth_mode):
    if smooth_mode == "off" or smooth_window <= 1:
        return 0
    if smooth_mode == "translation":
        return smooth_track_translation(track, smooth_window)
    if smooth_mode == "full":
        return smooth_track_full(track, smooth_window)
    raise ValueError(f"Unsupported smooth mode: {smooth_mode}")


def smoothstep01(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def ramp_up_weight(local_idx, ramp_frames):
    if ramp_frames <= 1:
        return 1.0
    if local_idx >= ramp_frames - 1:
        return 1.0
    return smoothstep01(local_idx / float(ramp_frames - 1))


def ramp_down_weight(remaining_idx, ramp_frames):
    if ramp_frames <= 1:
        return 1.0
    if remaining_idx >= ramp_frames - 1:
        return 1.0
    return smoothstep01(remaining_idx / float(ramp_frames - 1))


def find_previous_anchor(track, start_idx, max_gap):
    prev_idx = start_idx - 1
    while prev_idx >= 0 and track[prev_idx] is None:
        prev_idx -= 1
    if prev_idx < 0:
        return None, None, None
    missing_gap = start_idx - prev_idx - 1
    if max_gap >= 0 and missing_gap > max_gap:
        return None, None, missing_gap
    return prev_idx, track[prev_idx], missing_gap


def shift_hand_from_anchor(hand, anchor, alpha):
    """
    Translate current hand from previous anchor wrist toward current detection wrist.
    Local hand shape and orientation are preserved; only global translation changes.
    """
    shifted_groups = 0
    for ref_field, target_fields in COORD_GROUPS:
        cur_wrist = wrist_position_from_field(hand, ref_field)
        anchor_wrist = wrist_position_from_field(anchor, ref_field)
        if cur_wrist is None or anchor_wrist is None:
            continue
        target_wrist = (1.0 - alpha) * anchor_wrist + alpha * cur_wrist
        delta = target_wrist - cur_wrist
        if not np.all(np.isfinite(delta)):
            continue
        for field in target_fields:
            arr = valid_points(hand.get(field))
            if arr is None:
                continue
            hand[field] = (arr + delta[None, :]).astype(np.float32)
            shifted_groups += 1
    return shifted_groups


def set_soft_fields(hand, alpha, start_alpha, end_alpha, start, end, local_idx, anchor_label):
    raw_conf = raw_confidence_for_output(hand)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    start_alpha = float(np.clip(start_alpha, 0.0, 1.0))
    end_alpha = float(np.clip(end_alpha, 0.0, 1.0))
    hand["raw_confidence"] = float(raw_conf)
    hand["visualization_alpha"] = alpha
    hand["visualization_weight"] = alpha
    hand["constraint_weight"] = alpha
    hand["soft_start_alpha"] = start_alpha
    hand["soft_start_weight"] = start_alpha
    hand["soft_end_alpha"] = end_alpha
    hand["soft_end_weight"] = end_alpha
    hand["soft_start"] = bool(start_alpha < 0.999)
    hand["soft_end"] = bool(end_alpha < 0.999)
    hand["soft_start_anchor"] = anchor_label
    hand["segment_start_frame"] = int(start)
    hand["segment_end_frame"] = int(end)
    hand["segment_length"] = int(end - start + 1)
    hand["segment_frame_index"] = int(local_idx)
    # Backward-compatible fallback: downstream that already uses confidence
    # will get soft-start behavior without reading new fields.
    hand["confidence"] = float(raw_conf * alpha)


def apply_soft_start_and_end(
    track,
    soft_start_frames,
    soft_end_frames,
    soft_start_geometry=True,
    soft_start_anchor_max_gap=45,
):
    stats = {
        "soft_weighted_hands": 0,
        "soft_started_hands": 0,
        "soft_ended_hands": 0,
        "soft_start_geometry_shifted_hands": 0,
        "soft_start_weight_only_hands": 0,
    }
    n = len(track)
    for start, end in list(iter_segments(track)):
        starts_after_missing = start > 0
        ends_before_missing = end < n - 1
        _, anchor, missing_gap = find_previous_anchor(track, start, soft_start_anchor_max_gap)
        if not starts_after_missing:
            anchor_label = "none"
        elif anchor is not None:
            anchor_label = "previous_hand"
        elif missing_gap is None:
            anchor_label = "first_segment_weight_only"
        else:
            anchor_label = "gap_too_large_weight_only"

        for local_idx, idx in enumerate(range(start, end + 1)):
            hand = track[idx]
            if hand is None:
                continue

            start_alpha = 1.0
            if starts_after_missing:
                start_alpha = ramp_up_weight(local_idx, soft_start_frames)
                if local_idx < soft_start_frames:
                    stats["soft_started_hands"] += 1
                    if soft_start_geometry and anchor is not None:
                        shifted = shift_hand_from_anchor(hand, anchor, start_alpha)
                        if shifted > 0:
                            stats["soft_start_geometry_shifted_hands"] += 1
                        else:
                            stats["soft_start_weight_only_hands"] += 1
                    else:
                        stats["soft_start_weight_only_hands"] += 1

            remaining_idx = end - idx
            end_alpha = 1.0
            if ends_before_missing:
                end_alpha = ramp_down_weight(remaining_idx, soft_end_frames)
                if remaining_idx < soft_end_frames:
                    stats["soft_ended_hands"] += 1

            alpha = min(start_alpha, end_alpha)
            set_soft_fields(hand, alpha, start_alpha, end_alpha, start, end, local_idx, anchor_label)
            stats["soft_weighted_hands"] += 1
    return stats


def drop_low_alpha_hands(track, min_output_alpha):
    if min_output_alpha <= 0:
        return 0
    dropped = 0
    for idx, hand in enumerate(track):
        if hand is None:
            continue
        alpha = finite_float(hand.get("visualization_alpha", 1.0), default=1.0)
        if alpha < min_output_alpha:
            track[idx] = None
            dropped += 1
    return dropped


def arrays_to_lists(hand):
    out = {}
    for key, value in hand.items():
        if key in ARRAY_FIELDS and value is not None:
            out[key] = np.asarray(value, dtype=np.float32).tolist()
        else:
            out[key] = value
    return out


def print_jump_stats(side, track):
    ratios = collect_wrist_jump_ratios(track)
    if not ratios:
        print(f"{side} wrist jump ratio: no valid adjacent detections")
        return
    arr = np.asarray(ratios, dtype=np.float64)
    print(
        f"{side} wrist jump ratio per frame, normalized by hand size: "
        f"p50={np.percentile(arr, 50):.3f}, "
        f"p90={np.percentile(arr, 90):.3f}, "
        f"p95={np.percentile(arr, 95):.3f}, "
        f"p99={np.percentile(arr, 99):.3f}, "
        f"max={arr.max():.3f}"
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Prepare HaWoR hand JSONL for stable visualization with soft-start."
    )
    parser.add_argument("--hands-jsonl", required=True, type=Path)
    parser.add_argument("--timestamps-txt", type=Path, default=None)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument(
        "--egoallo-soft-constraint-mode",
        "--egoallo-mode",
        dest="egoallo_soft_constraint_mode",
        action="store_true",
        default=False,
        help=(
            "Use safer defaults for EgoAllo when downstream consumes soft weights: "
            "no gap fill, no smoothing, keep all soft-start frames as weighted constraints. "
            "Any explicitly supplied command-line values still take precedence."
        ),
    )
    parser.add_argument(
        "--visualization-safe-mode",
        action="store_true",
        default=False,
        help=(
            "Use safer defaults for visualization when the renderer/IK still snaps to hand geometry: "
            "disable gap fill/smoothing and hard-drop very low-alpha soft-start frames. "
            "This avoids showing the first frames of a suddenly appearing hand at the full detected wrist location."
        ),
    )
    parser.add_argument(
        "--frame-key",
        default="extracted_idx",
        help="Frame id field in the input HaWoR JSONL. Default: extracted_idx.",
    )
    parser.add_argument("--keypoints-field", default="keypoints_3d")
    parser.add_argument("--camera-keypoints-field", default="camera_keypoints_3d")
    parser.add_argument("--hawor-camera-keypoints-field", default="hawor_camera_keypoints_3d")
    parser.add_argument("--verts-field", default="mesh_vertices_3d")

    # Cleanup / stability.
    parser.add_argument(
        "--drop-isolated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop one-frame same-side detections. Enabled by default; use --no-drop-isolated to disable.",
    )
    parser.add_argument(
        "--min-track-len",
        type=int,
        default=3,
        help="Drop same-side hand segments shorter than this many frames. Use 0/1 to disable. Default: 3.",
    )
    parser.add_argument(
        "--trim-segment-edges",
        type=int,
        default=0,
        help="Hard-trim this many frames from each segment boundary. Usually keep 0 when soft-start is used.",
    )
    parser.add_argument(
        "--max-wrist-step",
        type=float,
        default=0.0,
        help="Raw-coordinate wrist jump threshold per frame. Use 0 to disable. Kept for compatibility.",
    )
    parser.add_argument(
        "--max-wrist-step-hand-ratio",
        type=float,
        default=0.0,
        help="Scale-free jump threshold: wrist motion per frame / hand size. Use 0 to disable. Try 1.0-2.0.",
    )

    # Gap fill.
    parser.add_argument(
        "--max-gap-fill",
        type=int,
        default=3,
        help="Fill missing detections only for gaps at most this many frames. Use 0 to disable. Default: 3.",
    )
    parser.add_argument(
        "--max-gap-endpoint-hand-ratio",
        type=float,
        default=1.5,
        help="When filling a gap, require endpoint wrist motion per step <= this many hand sizes. Use 0 to disable.",
    )

    # Smoothing.
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Temporal smoothing window. Use 1 to disable. Default: 5.",
    )
    parser.add_argument(
        "--smooth-mode",
        choices=("translation", "full", "off"),
        default="translation",
        help="translation preserves local hand shape/orientation; full averages all arrays; off disables smoothing.",
    )

    # Soft start/end.
    parser.add_argument(
        "--soft-start-frames",
        type=int,
        default=10,
        help="Fade in a segment over this many frames. First frame has alpha 0. Use 0/1 to disable. Default: 10.",
    )
    parser.add_argument(
        "--soft-end-frames",
        type=int,
        default=4,
        help="Fade out a segment over this many frames before disappearance. Use 0/1 to disable. Default: 4.",
    )
    parser.add_argument(
        "--soft-start-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If a previous same-side hand segment exists nearby, translate early frames from that anchor. Enabled by default.",
    )
    parser.add_argument(
        "--soft-start-anchor-max-gap",
        type=int,
        default=45,
        help="Use previous same-side hand as geometry anchor only if the missing gap is <= this many frames. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--min-output-alpha",
        type=float,
        default=0.0,
        help=(
            "Drop hands whose final visualization_alpha is below this value. "
            "Useful if downstream ignores alpha/confidence. Default: 0 keeps all soft-start frames."
        ),
    )
    parser.add_argument(
        "--print-jump-stats",
        action="store_true",
        help="Print scale-free wrist jump statistics before and after filtering.",
    )
    return parser


def _arg_supplied(*names):
    for token in sys.argv[1:]:
        for name in names:
            if token == name or token.startswith(name + "="):
                return True
    return False


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.egoallo_soft_constraint_mode:
        # For EgoAllo soft-constraint mode, do not fabricate missing hands and
        # do not change world/camera geometry by smoothing unless the user
        # explicitly requests it. The soft-start/soft-end fields are still kept.
        if not _arg_supplied("--max-gap-fill"):
            args.max_gap_fill = 0
        if not _arg_supplied("--smooth-mode"):
            args.smooth_mode = "off"
        if not _arg_supplied("--smooth-window"):
            args.smooth_window = 1
        if not _arg_supplied("--min-output-alpha"):
            args.min_output_alpha = 0.0
        if not _arg_supplied("--trim-segment-edges"):
            args.trim_segment_edges = 0
        print(
            "EgoAllo soft-constraint mode: "
            f"max_gap_fill={args.max_gap_fill}, "
            f"smooth_mode={args.smooth_mode}, "
            f"smooth_window={args.smooth_window}, "
            f"min_output_alpha={args.min_output_alpha}, "
            f"trim_segment_edges={args.trim_segment_edges}"
        )

    if args.visualization_safe_mode:
        # Visualization-safe mode is stricter than soft-constraint mode because
        # many visualizers/IK steps use the detection geometry immediately even when
        # alpha is low. Therefore low-alpha frames are omitted instead of merely weighted.
        if not _arg_supplied("--max-gap-fill"):
            args.max_gap_fill = 0
        if not _arg_supplied("--smooth-mode"):
            args.smooth_mode = "off"
        if not _arg_supplied("--smooth-window"):
            args.smooth_window = 1
        if not _arg_supplied("--min-output-alpha"):
            args.min_output_alpha = 0.35
        if not _arg_supplied("--soft-start-frames"):
            args.soft_start_frames = 15
        if not _arg_supplied("--soft-end-frames"):
            args.soft_end_frames = 6
        if not _arg_supplied("--min-track-len"):
            args.min_track_len = max(args.min_track_len, 6)
        print(
            "Visualization-safe mode: "
            f"max_gap_fill={args.max_gap_fill}, "
            f"smooth_mode={args.smooth_mode}, "
            f"smooth_window={args.smooth_window}, "
            f"min_output_alpha={args.min_output_alpha}, "
            f"soft_start_frames={args.soft_start_frames}, "
            f"soft_end_frames={args.soft_end_frames}, "
            f"min_track_len={args.min_track_len}"
        )

    timestamp_map = read_timestamps(args.timestamps_txt)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    items_by_frame = {}
    raw_frame_count = 0
    raw_hand_count = 0
    for rec in iter_jsonl(args.hands_jsonl):
        frame_id = pick_frame_id(rec, args.frame_key)
        timestamp = pick_timestamp(rec, frame_id, timestamp_map)
        src_frame_id = pick_src_frame_id(rec, frame_id, timestamp_map)

        out_hands = []
        for hand_idx, hand in enumerate(rec.get("hands", [])):
            side = str(hand.get("side", "")).lower()
            if side not in {"left", "right"}:
                raise ValueError(
                    f"Line {rec.get('_line_no')}, hand {hand_idx}: side must be left/right"
                )
            conf = hand_confidence(hand)
            out_hands.append(
                {
                    "side": side,
                    "confidence": conf,
                    "raw_confidence": conf,
                    "interpolated": False,
                    "camera_keypoints_3d": require_points(
                        hand,
                        args.camera_keypoints_field,
                        hand_idx,
                        rec.get("_line_no"),
                    ),
                    "hawor_camera_keypoints_3d": optional_points(
                        hand,
                        args.hawor_camera_keypoints_field,
                        hand_idx,
                        rec.get("_line_no"),
                    ),
                    "world_keypoints_3d": require_points(
                        hand,
                        args.keypoints_field,
                        hand_idx,
                        rec.get("_line_no"),
                    ),
                    "world_mesh_vertices_3d": require_points(
                        hand,
                        args.verts_field,
                        hand_idx,
                        rec.get("_line_no"),
                    ),
                }
            )

        items_by_frame[frame_id] = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "timestamp_ns": int(round(timestamp * 1e9)),
            "src_frame_id": src_frame_id,
            "hands": list(select_one_hand_per_side(out_hands).values()),
        }
        raw_frame_count += 1
        raw_hand_count += len(out_hands)

    frame_ids = sorted(timestamp_map.keys()) if timestamp_map else sorted(items_by_frame.keys())
    items = []
    for frame_id in frame_ids:
        if frame_id in items_by_frame:
            items.append(items_by_frame[frame_id])
        else:
            timestamp, src_frame_id = timestamp_map[frame_id]
            items.append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "timestamp_ns": int(round(timestamp * 1e9)),
                    "src_frame_id": src_frame_id,
                    "hands": [],
                }
            )

    tracks = {"left": [None] * len(items), "right": [None] * len(items)}
    for idx, item in enumerate(items):
        by_side = select_one_hand_per_side(item.get("hands", []))
        for side in ("left", "right"):
            if side in by_side:
                tracks[side][idx] = by_side[side]

    stats = {
        "dropped_isolated": 0,
        "dropped_large_jumps_raw": 0,
        "dropped_large_jumps_relative": 0,
        "filled_gaps": 0,
        "dropped_short_segments_before_fill": 0,
        "dropped_short_segments_after_fill": 0,
        "trimmed_segment_edges": 0,
        "smoothed_values": 0,
        "soft_weighted_hands": 0,
        "soft_started_hands": 0,
        "soft_ended_hands": 0,
        "soft_start_geometry_shifted_hands": 0,
        "soft_start_weight_only_hands": 0,
        "dropped_low_alpha": 0,
    }

    for side in ("left", "right"):
        if args.print_jump_stats:
            print_jump_stats(f"{side} before filtering", tracks[side])

        stats["dropped_large_jumps_raw"] += drop_large_wrist_jumps_raw(
            tracks[side],
            args.max_wrist_step,
        )
        stats["dropped_large_jumps_relative"] += drop_large_wrist_jumps_relative(
            tracks[side],
            args.max_wrist_step_hand_ratio,
        )
        if args.drop_isolated:
            stats["dropped_isolated"] += drop_isolated_detections(tracks[side])
        stats["dropped_short_segments_before_fill"] += drop_short_segments(
            tracks[side],
            args.min_track_len,
        )
        stats["filled_gaps"] += fill_short_gaps(
            tracks[side],
            args.max_gap_fill,
            args.max_gap_endpoint_hand_ratio,
        )
        stats["dropped_short_segments_after_fill"] += drop_short_segments(
            tracks[side],
            args.min_track_len,
        )
        stats["trimmed_segment_edges"] += trim_segment_edges(
            tracks[side],
            args.trim_segment_edges,
        )
        stats["smoothed_values"] += smooth_track(
            tracks[side],
            args.smooth_window,
            args.smooth_mode,
        )
        soft_stats = apply_soft_start_and_end(
            tracks[side],
            args.soft_start_frames,
            args.soft_end_frames,
            args.soft_start_geometry,
            args.soft_start_anchor_max_gap,
        )
        for key, value in soft_stats.items():
            stats[key] += value
        stats["dropped_low_alpha"] += drop_low_alpha_hands(
            tracks[side],
            args.min_output_alpha,
        )

        if args.print_jump_stats:
            print_jump_stats(f"{side} after filtering", tracks[side])

    num_hands = 0
    num_interpolated = 0
    num_soft_started_out = 0
    num_soft_ended_out = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for idx, item in enumerate(items):
            out_hands = []
            for side in ("left", "right"):
                hand = tracks[side][idx]
                if hand is None:
                    continue
                out_hands.append(arrays_to_lists(hand))
                num_hands += 1
                if hand.get("interpolated", False):
                    num_interpolated += 1
                if hand.get("soft_start", False):
                    num_soft_started_out += 1
                if hand.get("soft_end", False):
                    num_soft_ended_out += 1
            item_out = {
                "frame_id": int(item["frame_id"]),
                "timestamp": float(item["timestamp"]),
                "timestamp_ns": int(item["timestamp_ns"]),
                "src_frame_id": int(item["src_frame_id"]),
                "hands": out_hands,
            }
            out.write(json.dumps(item_out, ensure_ascii=False) + "\n")

    print(f"Saved: {args.output_jsonl}")
    print(f"Raw frames: {raw_frame_count}")
    print(f"Output frames: {len(items)}")
    print(f"Raw hands: {raw_hand_count}")
    print(f"Output hands: {num_hands}")
    print(f"Interpolated hands: {num_interpolated}")
    print(f"Output soft-start hands: {num_soft_started_out}")
    print(f"Output soft-end hands: {num_soft_ended_out}")
    print(f"Dropped raw wrist jumps: {stats['dropped_large_jumps_raw']}")
    print(f"Dropped relative wrist jumps: {stats['dropped_large_jumps_relative']}")
    print(f"Dropped isolated detections: {stats['dropped_isolated']}")
    print(f"Dropped short segments before fill: {stats['dropped_short_segments_before_fill']}")
    print(f"Gap-filled hands: {stats['filled_gaps']}")
    print(f"Dropped short segments after fill: {stats['dropped_short_segments_after_fill']}")
    print(f"Trimmed segment-edge hands: {stats['trimmed_segment_edges']}")
    print(f"Smoothed values/groups: {stats['smoothed_values']}")
    print(f"Soft-weighted hands: {stats['soft_weighted_hands']}")
    print(f"Soft-started hands: {stats['soft_started_hands']}")
    print(f"Soft-ended hands: {stats['soft_ended_hands']}")
    print(f"Soft-start geometry shifted hands: {stats['soft_start_geometry_shifted_hands']}")
    print(f"Soft-start weight-only hands: {stats['soft_start_weight_only_hands']}")
    print(f"Dropped low-alpha hands: {stats['dropped_low_alpha']}")
    print("Soft fields: visualization_alpha, visualization_weight, constraint_weight, raw_confidence")


if __name__ == "__main__":
    main()
