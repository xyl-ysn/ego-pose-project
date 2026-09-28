#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_line_no"] = line_no
            records.append(rec)
    if not records:
        raise RuntimeError(f"No records found in {path}")
    return records


def sort_records(records, sort_key):
    missing = [rec.get("_line_no") for rec in records if sort_key not in rec]
    if missing:
        raise KeyError(f"Missing sort key {sort_key!r} in JSONL lines: {missing[:10]}")
    return sorted(records, key=lambda rec: rec[sort_key])


def pose_to_c2w(rec):
    if "R" not in rec or "t" not in rec:
        raise KeyError(f"Line {rec.get('_line_no')}: missing R or t")

    R = np.asarray(rec["R"], dtype=np.float64)
    t = np.asarray(rec["t"], dtype=np.float64).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"Line {rec.get('_line_no')}: R has shape {R.shape}, expected (3, 3)")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    coordinate = rec.get("coordinate", "camera_to_world")
    if coordinate == "camera_to_world":
        return T
    if coordinate == "world_to_camera":
        return np.linalg.inv(T)
    raise ValueError(
        f"Line {rec.get('_line_no')}: unsupported coordinate={coordinate!r}; "
        "expected camera_to_world or world_to_camera"
    )


def intrinsics_to_row(rec):
    if "intrinsics" not in rec:
        raise KeyError(f"Line {rec.get('_line_no')}: missing intrinsics field")
    K = np.asarray(rec["intrinsics"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Line {rec.get('_line_no')}: intrinsics has shape {K.shape}, expected (3, 3)")
    depth_shape_raw = rec.get("depth_shape")
    if depth_shape_raw is None:
        raise KeyError(f"Line {rec.get('_line_no')}: missing depth_shape")
    depth_shape = np.asarray(depth_shape_raw, dtype=np.int64).reshape(-1)
    if depth_shape.shape != (2,) or np.any(depth_shape <= 0):
        raise ValueError(
            f"Line {rec.get('_line_no')}: depth_shape must contain positive [height, width]"
        )
    height, width = depth_shape
    return [K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height]


def main():
    parser = argparse.ArgumentParser(description="Export camera poses/timestamps/intrinsics from JSONL")
    # 必填输入
    parser.add_argument("--jsonl", required=True, type=Path, help="Input JSONL file with camera data")
    # 排序参数（保留，控制pose行顺序）
    parser.add_argument(
        "--sort-key",
        default="extracted_idx",
        choices=["extracted_idx", "original_frame_idx", "timestamp"],
        help="Order used for camera_poses.txt rows (default: extracted_idx)",
    )
    # 三个可选输出路径（核心修改点）
    parser.add_argument("--camera-poses-path", type=Path, help="Output path for camera_poses.txt")
    parser.add_argument("--timestamps-path", type=Path, help="Output path for timestamps.txt")
    parser.add_argument("--intrinsics-path", type=Path, help="Output path for intrinsic.txt")

    args = parser.parse_args()

    # 校验：至少指定一个输出文件
    if not any([args.camera_poses_path, args.timestamps_path, args.intrinsics_path]):
        parser.error("At least one output path (--camera-poses-path / --timestamps-path / --intrinsics-path) must be specified")

    # 加载并排序数据
    records = sort_records(load_jsonl(args.jsonl), args.sort_key)

    # 1. 生成 camera_poses.txt（如果指定了路径）
    if args.camera_poses_path:
        pose_path = args.camera_poses_path.resolve()
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        poses = np.stack([pose_to_c2w(rec) for rec in records], axis=0)
        np.savetxt(pose_path, poses.reshape(len(poses), 16), fmt="%.10f")
        print(f"Saved camera poses: {pose_path}")

    # 2. 生成 timestamps.txt（如果指定了路径）
    if args.timestamps_path:
        ts_path = args.timestamps_path.resolve()
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp_rows = []
        for i, rec in enumerate(records):
            frame_id = int(rec.get("extracted_idx", i))
            timestamp = float(rec.get("timestamp", i))
            src_frame_id = int(rec.get("original_frame_idx", frame_id))
            timestamp_rows.append([frame_id, timestamp, src_frame_id])
        np.savetxt(
            ts_path,
            np.asarray(timestamp_rows, dtype=np.float64),
            fmt=["%d", "%.9f", "%d"],
            header="frame_id timestamp src_frame_id",
            comments="",
        )
        print(f"Saved timestamps: {ts_path}")

    # 3. 生成 intrinsic.txt（如果指定了路径）
    if args.intrinsics_path:
        intr_path = args.intrinsics_path.resolve()
        intr_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [intrinsics_to_row(rec) for rec in records]
        np.savetxt(
            intr_path,
            np.asarray(rows, dtype=np.float64),
            fmt="%.10f",
            header="fx fy cx cy source_width source_height",
            comments="# ",
        )
        print(f"Saved intrinsics: {intr_path}")

    print(f"Total processed frames: {len(records)}")


if __name__ == "__main__":
    main()
