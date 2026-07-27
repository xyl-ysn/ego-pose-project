#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conservative HaWoR hand-track stabilizer for EgoAllo when Aria wrist/palm is not available.

Input: JSONL produced by 02_extract_hawor_world_hands.py
Output: JSONL with the same schema, but with fewer / more stable hand constraints.

What this script does:
  1. correct obvious left/right side flips using wrist-track continuity,
  2. remove duplicated left/right detections at nearly the same wrist location,
  3. drop large wrist jumps and short hand segments,
  4. optionally trim segment boundaries,
  5. recompute soft-start alpha after filtering, and optionally hard-drop low-alpha frames for visualization.

It deliberately does NOT interpolate missing hands by default. Interpolated hands often do not
have matching MANO params and can become placeholder MANO constraints in 03.
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
SOFT_ALPHA_KEYS = (
    "constraint_weight",
    "visualization_weight",
    "visualization_alpha",
    "soft_start_weight",
    "soft_start_alpha",
    "soft_end_weight",
    "soft_end_alpha",
)
CONF_KEYS = ("confidence", "score", "hand_score", "det_score", "raw_confidence")


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_line_no"] = line_no
            yield rec


def finite_float(value, default=0.0):
    if value is None:
        return default
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return default
        out = float(arr[0])
    except (TypeError, ValueError):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
    return out if np.isfinite(out) else default


def confidence(hand):
    for key in CONF_KEYS:
        if key in hand:
            return finite_float(hand.get(key), 0.0)
    return 0.0


def soft_alpha(hand, default=1.0):
    for key in SOFT_ALPHA_KEYS:
        if key in hand:
            return float(np.clip(finite_float(hand.get(key), default), 0.0, 1.0))
    return float(default)


def valid_points(value):
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] == 3 and np.all(np.isfinite(arr)):
        return arr
    return None


def points_field(hand):
    for field in ("world_keypoints_3d", "camera_keypoints_3d", "hawor_camera_keypoints_3d"):
        arr = valid_points(hand.get(field))
        if arr is not None:
            return field, arr
    return None, None


def wrist(hand):
    _, arr = points_field(hand)
    if arr is None:
        return None
    return arr[0]


def hand_size(hand, eps=1e-8):
    _, arr = points_field(hand)
    if arr is None:
        return None
    w = arr[0]
    if arr.shape[0] > 9:
        s = float(np.linalg.norm(arr[9] - w))
        if s > eps:
            return s
    if arr.shape[0] <= 1:
        return None
    d = np.linalg.norm(arr[1:] - w[None, :], axis=1)
    d = d[d > eps]
    if len(d) == 0:
        return None
    return float(np.median(d))


def normalized_distance(a, b, frame_gap=1):
    wa, wb = wrist(a), wrist(b)
    sa, sb = hand_size(a), hand_size(b)
    if wa is None or wb is None or sa is None or sb is None:
        return None
    denom = max(1e-8, 0.5 * (sa + sb)) * max(1, frame_gap)
    return float(np.linalg.norm(wa - wb) / denom)


def copy_hand(hand):
    out = {}
    for k, v in hand.items():
        if k in ARRAY_FIELDS and v is not None:
            out[k] = np.asarray(v, dtype=np.float32).copy().tolist()
        else:
            out[k] = copy.deepcopy(v)
    return out


def select_one_hand_per_side(hands):
    selected = {}
    for hand in hands:
        side = str(hand.get("side", "")).lower()
        if side not in ("left", "right"):
            continue
        if side not in selected or confidence(hand) > confidence(selected[side]):
            selected[side] = hand
    return selected


def fix_side_flips(items, max_flip_dist_ratio, flip_margin_ratio):
    """Fix obvious single-frame or short left/right identity flips.

    A detection is flipped when it is much closer to the previous opposite-side wrist
    than to the previous same-side wrist. Distances are normalized by hand size, so
    this works for both metric and arbitrary VGGT scale.
    """
    last = {"left": None, "right": None}
    last_idx = {"left": None, "right": None}
    fixed = 0

    for idx, item in enumerate(items):
        hands = [copy_hand(h) for h in item.get("hands", [])]
        # Stronger detections update the state first when two candidates are present.
        hands.sort(key=confidence, reverse=True)

        for hand in hands:
            side = str(hand.get("side", "")).lower()
            if side not in ("left", "right"):
                continue
            other = "right" if side == "left" else "left"
            d_same = None
            d_other = None
            if last[side] is not None:
                d_same = normalized_distance(hand, last[side], idx - last_idx[side])
            if last[other] is not None:
                d_other = normalized_distance(hand, last[other], idx - last_idx[other])

            should_flip = False
            if d_other is not None and d_other <= max_flip_dist_ratio:
                if d_same is None:
                    # Same side has no recent anchor, but the opposite side does.
                    should_flip = True
                elif d_other + flip_margin_ratio < d_same:
                    should_flip = True

            if should_flip:
                hand["original_side"] = side
                hand["side"] = other
                hand["side_corrected"] = True
                fixed += 1

        by_side = select_one_hand_per_side(hands)
        item["hands"] = list(by_side.values())
        for side, hand in by_side.items():
            last[side] = hand
            last_idx[side] = idx
    return fixed


def drop_left_right_duplicates(items, duplicate_dist_ratio, mode="drop_both"):
    dropped = 0
    for item in items:
        by_side = select_one_hand_per_side(item.get("hands", []))
        if "left" not in by_side or "right" not in by_side:
            item["hands"] = list(by_side.values())
            continue
        ratio = normalized_distance(by_side["left"], by_side["right"], frame_gap=1)
        if ratio is None or ratio > duplicate_dist_ratio:
            item["hands"] = list(by_side.values())
            continue
        if mode == "keep_best":
            best = by_side["left"] if confidence(by_side["left"]) >= confidence(by_side["right"]) else by_side["right"]
            best["duplicate_lr_removed_other"] = True
            item["hands"] = [best]
            dropped += 1
        else:
            item["hands"] = []
            dropped += 2
    return dropped


def tracks_from_items(items):
    tracks = {"left": [None] * len(items), "right": [None] * len(items)}
    for i, item in enumerate(items):
        by_side = select_one_hand_per_side(item.get("hands", []))
        for side in ("left", "right"):
            if side in by_side:
                tracks[side][i] = by_side[side]
    return tracks


def iter_segments(track):
    i = 0
    n = len(track)
    while i < n:
        if track[i] is None:
            i += 1
            continue
        start = i
        while i < n and track[i] is not None:
            i += 1
        yield start, i - 1


def drop_short_segments(track, min_len):
    if min_len <= 1:
        return 0
    dropped = 0
    for start, end in list(iter_segments(track)):
        if end - start + 1 >= min_len:
            continue
        for i in range(start, end + 1):
            if track[i] is not None:
                track[i] = None
                dropped += 1
    return dropped


def trim_segment_edges(track, trim):
    if trim <= 0:
        return 0
    dropped = 0
    for start, end in list(iter_segments(track)):
        length = end - start + 1
        if length <= 2 * trim:
            ranges = [(start, end)]
        else:
            ranges = [(start, start + trim - 1), (end - trim + 1, end)]
        for lo, hi in ranges:
            for i in range(lo, hi + 1):
                if track[i] is not None:
                    track[i] = None
                    dropped += 1
    return dropped


def drop_large_jumps(track, max_ratio):
    if max_ratio <= 0:
        return 0
    dropped = 0
    prev = None
    prev_idx = None
    for i, hand in enumerate(track):
        if hand is None:
            continue
        if prev is not None:
            ratio = normalized_distance(hand, prev, i - prev_idx)
            if ratio is not None and ratio > max_ratio:
                track[i] = None
                dropped += 1
                continue
        prev = hand
        prev_idx = i
    return dropped


def drop_low_alpha(track, min_alpha):
    if min_alpha <= 0:
        return 0
    dropped = 0
    for i, hand in enumerate(track):
        if hand is None:
            continue
        if soft_alpha(hand, default=1.0) < min_alpha:
            track[i] = None
            dropped += 1
    return dropped


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


def recompute_soft_fields(track, soft_start_frames, soft_end_frames):
    """Recompute soft-start/end fields after 03 filtering changes segment boundaries.

    This is important: after side-flip fixes, duplicate removal, jump filtering, and
    short-segment removal, a frame that used to be in the middle of a segment may become
    the first visible frame. If we keep the old alpha, the visualizer/IK can still snap.
    """
    changed = 0
    n = len(track)
    for start, end in list(iter_segments(track)):
        starts_after_missing = start > 0
        ends_before_missing = end < n - 1
        for local_idx, idx in enumerate(range(start, end + 1)):
            hand = track[idx]
            if hand is None:
                continue
            start_alpha = ramp_up_weight(local_idx, soft_start_frames) if starts_after_missing else 1.0
            remaining_idx = end - idx
            end_alpha = ramp_down_weight(remaining_idx, soft_end_frames) if ends_before_missing else 1.0
            alpha = float(np.clip(min(start_alpha, end_alpha), 0.0, 1.0))
            raw = finite_float(hand.get("raw_confidence", hand.get("confidence", 1.0)), 1.0)
            if raw <= 0.0:
                raw = 1.0
            hand["raw_confidence"] = float(raw)
            hand["visualization_alpha"] = alpha
            hand["visualization_weight"] = alpha
            hand["constraint_weight"] = alpha
            hand["soft_start_alpha"] = float(start_alpha)
            hand["soft_start_weight"] = float(start_alpha)
            hand["soft_end_alpha"] = float(end_alpha)
            hand["soft_end_weight"] = float(end_alpha)
            hand["soft_start"] = bool(start_alpha < 0.999)
            hand["soft_end"] = bool(end_alpha < 0.999)
            hand["segment_start_frame"] = int(start)
            hand["segment_end_frame"] = int(end)
            hand["segment_length"] = int(end - start + 1)
            hand["segment_frame_index"] = int(local_idx)
            hand["confidence"] = float(raw * alpha)
            changed += 1
    return changed


def _arg_supplied(*names):
    for token in sys.argv[1:]:
        for name in names:
            if token == name or token.startswith(name + "="):
                return True
    return False


def items_from_tracks(template_items, tracks):
    out = []
    for i, item in enumerate(template_items):
        rec = {k: v for k, v in item.items() if not k.startswith("_") and k != "hands"}
        hands = []
        for side in ("left", "right"):
            if tracks[side][i] is not None:
                hands.append(tracks[side][i])
        rec["hands"] = hands
        out.append(rec)
    return out


def count_hands(items):
    return sum(len(item.get("hands", [])) for item in items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True, type=Path)
    ap.add_argument("--output-jsonl", required=True, type=Path)
    ap.add_argument("--max-flip-dist-ratio", type=float, default=1.5,
                    help="Flip side if a hand is within this normalized distance to previous opposite-side track.")
    ap.add_argument("--flip-margin-ratio", type=float, default=0.35,
                    help="Opposite-side match must beat same-side match by this normalized margin.")
    ap.add_argument("--duplicate-dist-ratio", type=float, default=0.75,
                    help="Drop left/right detections in the same frame if their wrist distance / hand size is below this.")
    ap.add_argument("--duplicate-mode", choices=("drop_both", "keep_best"), default="drop_both")
    ap.add_argument("--max-wrist-step-hand-ratio", type=float, default=2.0,
                    help="Drop per-side wrist jumps larger than this many hand sizes per frame. Use 0 to disable.")
    ap.add_argument("--min-track-len", type=int, default=8,
                    help="Drop per-side continuous segments shorter than this many frames.")
    ap.add_argument("--trim-segment-edges", type=int, default=0,
                    help="Hard-drop N frames at start/end of every remaining segment.")
    ap.add_argument("--min-alpha-to-keep", type=float, default=0.0,
                    help=(
                        "Hard-drop soft-start frames below this alpha. Default 0 keeps low-alpha frames "
                        "because the patched EgoAllo downstream uses constraint_weight. Set e.g. 0.35 "
                        "for visualization-safe output when the renderer/IK still snaps to hand geometry."
                    ))
    ap.add_argument("--recompute-soft-fields", action=argparse.BooleanOptionalAction, default=True,
                    help="Recompute soft-start/end fields after filtering. Enabled by default.")
    ap.add_argument("--soft-start-frames", type=int, default=12,
                    help="Soft-start ramp length used when recomputing segment alpha. Default: 12.")
    ap.add_argument("--soft-end-frames", type=int, default=5,
                    help="Soft-end ramp length used when recomputing segment alpha. Default: 5.")
    ap.add_argument("--visualization-safe-mode", action="store_true", default=False,
                    help=(
                        "For visualization/IK paths that still snap to hand geometry: recompute alpha with a longer ramp "
                        "and hard-drop low-alpha frames by default."
                    ))
    args = ap.parse_args()

    if args.visualization_safe_mode:
        if not _arg_supplied("--min-alpha-to-keep"):
            args.min_alpha_to_keep = 0.35
        if not _arg_supplied("--soft-start-frames"):
            args.soft_start_frames = 15
        if not _arg_supplied("--soft-end-frames"):
            args.soft_end_frames = 6
        if not _arg_supplied("--min-track-len"):
            args.min_track_len = max(args.min_track_len, 8)
        print(
            "Visualization-safe mode: "
            f"min_alpha_to_keep={args.min_alpha_to_keep}, "
            f"soft_start_frames={args.soft_start_frames}, "
            f"soft_end_frames={args.soft_end_frames}, "
            f"min_track_len={args.min_track_len}"
        )

    if args.min_alpha_to_keep <= 0:
        print("Soft-alpha frames are kept; downstream EgoAllo is expected to use constraint_weight.")
    else:
        print(f"Hard-dropping hands with soft alpha < {args.min_alpha_to_keep:.3f}.")

    items = list(iter_jsonl(args.input_jsonl))
    raw_hands = count_hands(items)

    stats = {}
    stats["fixed_side_flips"] = fix_side_flips(items, args.max_flip_dist_ratio, args.flip_margin_ratio)
    stats["dropped_lr_duplicates"] = drop_left_right_duplicates(items, args.duplicate_dist_ratio, args.duplicate_mode)

    tracks = tracks_from_items(items)
    stats["dropped_large_jumps"] = 0
    stats["dropped_short_segments"] = 0
    stats["trimmed_segment_edges"] = 0
    stats["recomputed_soft_fields"] = 0
    stats["dropped_low_alpha"] = 0
    for side in ("left", "right"):
        stats["dropped_large_jumps"] += drop_large_jumps(tracks[side], args.max_wrist_step_hand_ratio)
        stats["dropped_short_segments"] += drop_short_segments(tracks[side], args.min_track_len)
        stats["trimmed_segment_edges"] += trim_segment_edges(tracks[side], args.trim_segment_edges)
        if args.recompute_soft_fields:
            stats["recomputed_soft_fields"] += recompute_soft_fields(
                tracks[side],
                args.soft_start_frames,
                args.soft_end_frames,
            )
        stats["dropped_low_alpha"] += drop_low_alpha(tracks[side], args.min_alpha_to_keep)
        # Re-run short segment cleanup after hard alpha/edge drops.
        stats["dropped_short_segments"] += drop_short_segments(tracks[side], args.min_track_len)

    out_items = items_from_tracks(items, tracks)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for rec in out_items:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved: {args.output_jsonl}")
    print(f"Frames: {len(out_items)}")
    print(f"Input hands: {raw_hands}")
    print(f"Output hands: {count_hands(out_items)}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
