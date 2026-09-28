#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


WORLD_FIELDS = {
    "keypoints_3d",
    "mesh_vertices_3d",
    "world_keypoints_3d",
    "world_mesh_vertices_3d",
}

CAMERA_FIELDS = {
    "camera_keypoints_3d",
    "camera_mesh_vertices_3d",
    "camera_keypoints_from_world_3d",
    "hawor_camera_keypoints_3d",
    "hawor_camera_mesh_vertices_3d",
}

FRONTEND_COORDINATE_SYSTEM = {
    "name": "egoallo_world",
    "up_axis": "z",
    "floor_z": 0.0,
    "units": "meters",
}


def _coerce_json_friendly(obj, _seen=None):
    if _seen is None:
        _seen = set()

    if isinstance(obj, dict):
        obj_id = id(obj)
        if obj_id in _seen:
            return "<circular_reference>"
        _seen.add(obj_id)
        try:
            out = {str(k): _coerce_json_friendly(v, _seen) for k, v in obj.items()}
        finally:
            _seen.discard(obj_id)
        return out

    if isinstance(obj, list):
        obj_id = id(obj)
        if obj_id in _seen:
            return ["<circular_reference>"]
        _seen.add(obj_id)
        try:
            return [_coerce_json_friendly(v, _seen) for v in obj]
        finally:
            _seen.discard(obj_id)

    if isinstance(obj, tuple):
        return [_coerce_json_friendly(v, _seen) for v in obj]
    if isinstance(obj, set):
        return [_coerce_json_friendly(v, _seen) for v in obj]
    if isinstance(obj, np.ndarray):
        return _coerce_json_friendly(obj.tolist(), _seen)
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)

    return str(obj)


def load_head_trajectory(path: Path) -> np.ndarray:
    poses = np.asarray(np.load(path), dtype=np.float64)
    if poses.ndim == 2 and poses.shape == (4, 4):
        poses = poses[None, :, :]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{path} must have shape [N,4,4] or [4,4], got {poses.shape}")
    return poses


def scale_world_points(points: np.ndarray, scale: float, floor_z: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    anchor = np.array([0.0, 0.0, float(floor_z)], dtype=np.float64)
    return anchor + float(scale) * (points - anchor)


def scale_local_points(points: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) * float(scale)


def scale_pose_translations(poses: np.ndarray, scale: float, floor_z: float) -> np.ndarray:
    out = np.array(poses, dtype=np.float64, copy=True)
    out[:, :3, 3] = scale_world_points(out[:, :3, 3], scale, floor_z)
    return out


def get_R_yup_to_zup() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def load_vggt_to_egoallo_transform(R_align_path: Path, align_meta_path: Path) -> tuple[np.ndarray, np.ndarray]:
    R_align = np.load(R_align_path).astype(np.float64)
    if R_align.shape != (3, 3):
        raise ValueError(f"{R_align_path} must contain a 3x3 matrix, got {R_align.shape}")

    R_final = get_R_yup_to_zup() @ R_align
    det = float(np.linalg.det(R_final))
    if abs(det - 1.0) > 1e-3:
        raise ValueError(f"Invalid final rotation det={det:.6f}; expected close to 1")

    with np.load(align_meta_path, allow_pickle=True) as data:
        keys = set(data.keys())
        if "floor_center_da3" in keys:
            floor_center_da3 = np.asarray(data["floor_center_da3"], dtype=np.float64).reshape(3)
            floor_z = float((R_final @ floor_center_da3)[2])
        elif "floor_points_aligned" in keys:
            floor_points = np.asarray(data["floor_points_aligned"], dtype=np.float64)
            if floor_points.ndim != 2 or floor_points.shape[1] != 3:
                raise ValueError(
                    f"{align_meta_path}: floor_points_aligned must be [N,3], got {floor_points.shape}"
                )
            floor_points = floor_points[np.isfinite(floor_points).all(axis=1)]
            if len(floor_points) == 0:
                raise ValueError(f"{align_meta_path}: no finite floor_points_aligned")
            floor_z = float(np.median(floor_points[:, 2]))
        else:
            raise KeyError(
                f"{align_meta_path} has no supported floor metadata. "
                "Expected floor_center_da3 or floor_points_aligned."
            )

    return R_final, np.array([0.0, 0.0, -floor_z], dtype=np.float64)


def transform_vggt_world_to_egoallo(points: np.ndarray, R_final: np.ndarray, t_floor: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected points with last dimension 3, got {points.shape}")
    flat = points.reshape(-1, 3)
    out = (R_final @ flat.T).T + t_floor.reshape(1, 3)
    return out.reshape(points.shape)


def save_head_trajectory(input_path: Path, output_path: Path, scale: float, floor_z: float) -> dict:
    poses = load_head_trajectory(input_path)
    out = scale_pose_translations(poses, scale, floor_z).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, out)
    heights = out[:, 2, 3] - float(floor_z)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "count": int(len(out)),
        "height_min": float(np.min(heights)),
        "height_median": float(np.median(heights)),
        "height_max": float(np.max(heights)),
    }


def load_camera_poses_txt(path: Path) -> tuple[np.ndarray, np.ndarray | None, str]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]

    ids = None
    if data.shape[1] == 16:
        poses = data.reshape(-1, 4, 4)
        fmt = "16"
    elif data.shape[1] == 17:
        ids = data[:, 0:1]
        poses = data[:, 1:].reshape(-1, 4, 4)
        fmt = "17"
    elif data.shape[1] == 12:
        poses = np.tile(np.eye(4), (len(data), 1, 1))
        poses[:, :3, :4] = data.reshape(-1, 3, 4)
        fmt = "12"
    elif data.shape[1] == 13:
        ids = data[:, 0:1]
        poses = np.tile(np.eye(4), (len(data), 1, 1))
        poses[:, :3, :4] = data[:, 1:].reshape(-1, 3, 4)
        fmt = "13"
    else:
        raise ValueError(f"Unsupported camera pose shape {data.shape}: {path}")

    return poses.astype(np.float64), ids, fmt


def save_camera_poses_txt(input_path: Path, output_path: Path, scale: float, floor_z: float) -> dict:
    poses, ids, fmt = load_camera_poses_txt(input_path)
    out = scale_pose_translations(poses, scale, floor_z)

    if fmt == "16":
        data = out.reshape(len(out), 16)
    elif fmt == "17":
        data = np.concatenate([ids, out.reshape(len(out), 16)], axis=1)
    elif fmt == "12":
        data = out[:, :3, :4].reshape(len(out), 12)
    elif fmt == "13":
        data = np.concatenate([ids, out[:, :3, :4].reshape(len(out), 12)], axis=1)
    else:
        raise AssertionError(fmt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, data, fmt="%.9f")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "count": int(len(out)),
        "format": fmt,
        "translation_min": out[:, :3, 3].min(axis=0).tolist(),
        "translation_max": out[:, :3, 3].max(axis=0).tolist(),
    }


def scale_ascii_ply(input_path: Path, output_path: Path, scale: float, floor_z: float) -> dict:
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines or not lines[0].strip() == "ply":
        raise ValueError(f"{input_path} is not a PLY file")

    vertex_count = None
    header_end = None
    props = []
    in_vertex = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("format ") and "ascii" not in stripped:
            raise ValueError(
                f"{input_path} is not ascii PLY. Install/use plyfile support or export ascii PLY first."
            )
        if stripped.startswith("element vertex "):
            vertex_count = int(stripped.split()[-1])
            in_vertex = True
            continue
        if stripped.startswith("element ") and not stripped.startswith("element vertex "):
            in_vertex = False
        if in_vertex and stripped.startswith("property "):
            props.append(stripped.split()[-1])
        if stripped == "end_header":
            header_end = idx
            break

    if vertex_count is None or header_end is None:
        raise ValueError(f"Cannot parse PLY header: {input_path}")
    for key in ("x", "y", "z"):
        if key not in props:
            raise ValueError(f"{input_path} vertex properties must contain x/y/z")

    x_idx = props.index("x")
    y_idx = props.index("y")
    z_idx = props.index("z")
    out_lines = list(lines[: header_end + 1])
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    start = header_end + 1
    end = start + vertex_count
    for line in lines[start:end]:
        parts = line.rstrip("\n").split()
        xyz = np.array(
            [float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])],
            dtype=np.float64,
        )
        xyz = scale_world_points(xyz[None, :], scale, floor_z)[0]
        parts[x_idx] = f"{xyz[0]:.9f}"
        parts[y_idx] = f"{xyz[1]:.9f}"
        parts[z_idx] = f"{xyz[2]:.9f}"
        mins = np.minimum(mins, xyz)
        maxs = np.maximum(maxs, xyz)
        out_lines.append(" ".join(parts) + "\n")

    out_lines.extend(lines[end:])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "vertex_count": int(vertex_count),
        "min": mins.tolist(),
        "max": maxs.tolist(),
    }


def try_scale_ply_with_plyfile(input_path: Path, output_path: Path, scale: float, floor_z: float) -> dict:
    try:
        from plyfile import PlyData, PlyElement
    except Exception:
        return scale_ascii_ply(input_path, output_path, scale, floor_z)

    ply = PlyData.read(str(input_path))
    vertex = ply["vertex"].data.copy()
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    xyz = scale_world_points(xyz, scale, floor_z)
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]

    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(vertex, "vertex"))
        else:
            elements.append(element)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(output_path))
    return {
        "input": str(input_path),
        "output": str(output_path),
        "vertex_count": int(len(vertex)),
        "min": xyz.min(axis=0).tolist(),
        "max": xyz.max(axis=0).tolist(),
    }


def transform_and_scale_point_cloud(
    input_path: Path,
    output_path: Path,
    *,
    scale: float,
    floor_z: float,
    input_space: str,
    r_align_path: Path | None,
    align_meta_path: Path | None,
    output_voxel_size: float,
) -> dict:
    input_space = str(input_space or "egoallo").strip().lower()
    if input_space not in {"egoallo", "raw_vggt"}:
        raise ValueError(f"Unsupported --point-cloud-space: {input_space}")

    if input_space == "egoallo" and float(output_voxel_size) <= 0:
        return try_scale_ply_with_plyfile(input_path, output_path, scale, floor_z)

    try:
        import open3d as o3d
    except Exception as exc:
        if input_space == "egoallo":
            return try_scale_ply_with_plyfile(input_path, output_path, scale, floor_z)
        raise RuntimeError("open3d is required for --point-cloud-space raw_vggt") from exc

    pcd = o3d.io.read_point_cloud(str(input_path))
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {input_path}")

    points_before = int(len(pcd.points))
    if float(output_voxel_size) < 0:
        raise ValueError("--final-scene-output-voxel-size must be >= 0")
    if float(output_voxel_size) > 0:
        pcd = pcd.voxel_down_sample(float(output_voxel_size))

    points = np.asarray(pcd.points, dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64) if pcd.has_normals() else None

    R_final = None
    t_floor = None
    if input_space == "raw_vggt":
        if r_align_path is None or align_meta_path is None:
            raise ValueError("--point-cloud-space raw_vggt requires --point-cloud-r-align and --point-cloud-align-meta")
        R_final, t_floor = load_vggt_to_egoallo_transform(r_align_path, align_meta_path)
        points = transform_vggt_world_to_egoallo(points, R_final, t_floor)
        if normals is not None:
            normals = (R_final @ normals.T).T

    points = scale_world_points(points, scale, floor_z)

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    if pcd.has_colors():
        pcd_out.colors = pcd.colors
    if normals is not None:
        pcd_out.normals = o3d.utility.Vector3dVector(normals)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd_out)

    out_points = np.asarray(pcd_out.points, dtype=np.float64)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_space": input_space,
        "vertex_count_before": points_before,
        "vertex_count": int(len(out_points)),
        "output_voxel_size": float(output_voxel_size),
        "R_final": None if R_final is None else R_final.tolist(),
        "t_floor": None if t_floor is None else t_floor.tolist(),
        "min": out_points.min(axis=0).tolist() if len(out_points) else None,
        "max": out_points.max(axis=0).tolist() if len(out_points) else None,
    }


def scale_value_for_json(key: str, value, scale: float, floor_z: float):
    if key in WORLD_FIELDS:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim >= 2 and arr.shape[-1] == 3:
            flat = arr.reshape(-1, 3)
            scaled = scale_world_points(flat, scale, floor_z).reshape(arr.shape)
            return scaled.astype(np.float32).tolist()
    if key in CAMERA_FIELDS:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim >= 2 and arr.shape[-1] == 3:
            return scale_local_points(arr, scale).astype(np.float32).tolist()
    return scale_json_obj(value, scale, floor_z)


def scale_json_obj(obj, scale: float, floor_z: float):
    if isinstance(obj, dict):
        return {key: scale_value_for_json(key, value, scale, floor_z) for key, value in obj.items()}
    if isinstance(obj, list):
        return [scale_json_obj(value, scale, floor_z) for value in obj]
    return obj


def scale_hands_jsonl(input_path: Path, output_path: Path, scale: float, floor_z: float) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    hands = 0
    with open(input_path, "r", encoding="utf-8") as src, open(
        output_path, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out = scale_json_obj(rec, scale, floor_z)
            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            frames += 1
            hands += len(rec.get("hands", []))
    return {
        "input": str(input_path),
        "output": str(output_path),
        "frames": int(frames),
        "hands": int(hands),
    }


def read_timestamps(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    table = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")
    if table.shape == ():
        table = np.asarray([table])
    required = {"frame_id", "timestamp", "src_frame_id"}
    if not required.issubset(set(table.dtype.names or [])):
        raise ValueError(f"{path} must have header: frame_id timestamp src_frame_id")
    return [
        {
            "frame_id": int(row["frame_id"]),
            "timestamp": float(row["timestamp"]),
            "src_frame_id": int(row["src_frame_id"]),
        }
        for row in table
    ]


def record_frame_id(rec: dict, fallback: int) -> int:
    for key in ("frame_id", "extracted_idx", "idx"):
        if key in rec:
            return int(rec[key])
    return int(fallback)


def record_src_frame_id(rec: dict, frame_id: int) -> int:
    for key in ("src_frame_id", "original_frame_idx"):
        if key in rec:
            return int(rec[key])
    return int(frame_id)


def record_timestamp(rec: dict, fallback: float) -> float:
    if "timestamp" in rec:
        return float(rec["timestamp"])
    if "timestamp_sec" in rec:
        return float(rec["timestamp_sec"])
    if "timestamp_ns" in rec:
        return float(rec["timestamp_ns"]) / 1e9
    return float(fallback)


def hand_confidence(hand: dict):
    for key in ("confidence", "score", "hand_score", "det_score"):
        if key in hand:
            try:
                return float(hand[key])
            except (TypeError, ValueError):
                return None
    return None


def pick_world_hand_points(hand: dict, keys: tuple[str, ...]) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        if key not in hand:
            continue
        arr = np.asarray(hand[key], dtype=np.float64)
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 3:
            arr = arr[0]
        if arr.ndim >= 2 and arr.shape[-1] == 3:
            return arr, key
        return None, key
    return None, None


def save_frontend_head_jsonl(head_trajectory_path: Path, output_path: Path, timestamps_path: Path | None) -> dict:
    poses = load_head_trajectory(head_trajectory_path).astype(np.float32)
    timestamp_rows = read_timestamps(timestamps_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, T in enumerate(poses):
            meta = (
                timestamp_rows[i]
                if timestamp_rows is not None and i < len(timestamp_rows)
                else {"frame_id": i, "timestamp": float(i), "src_frame_id": i}
            )
            rec = {
                "frame_id": int(meta["frame_id"]),
                "timestamp": float(meta["timestamp"]),
                "src_frame_id": int(meta["src_frame_id"]),
                "coordinate_system": FRONTEND_COORDINATE_SYSTEM,
                "head": {
                    "position": T[:3, 3].astype(np.float32).tolist(),
                    "rotation": T[:3, :3].astype(np.float32).tolist(),
                    "matrix": T.astype(np.float32).tolist(),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"output": str(output_path), "frames": int(len(poses))}


def _rotation_to_quaternion_wxyz(R: np.ndarray) -> list[float]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / np.linalg.norm(q)
    return q.tolist()


def save_frontend_objects_parquet(
    objects_parquet_in: Path,
    output_path: Path,
    R_final: np.ndarray,
    t_floor: np.ndarray,
    scale: float,
    floor_z: float,
) -> dict:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError("pandas/pyarrow is required to export objects parquet") from exc

    table = pd.read_parquet(objects_parquet_in)
    rows: list[dict[str, Any]] = []
    anchor = np.array([0.0, 0.0, float(floor_z)], dtype=np.float64)

    for _, row in table.iterrows():
        frame_id = int(row["frame_id"])
        common = {
            "timestamp_ns": int(row["timestamp_ns"]),
            "frame": frame_id,
            "timestamp": float(row["timestamp_ns"]) / 1e9,
            "src_frame_id": int(row["source_frame_idx"]),
            "track_id": str(row.get("track_id") or ""),
            "category": str(row.get("category") or ""),
            "valid": bool(row.get("valid", False)),
            "confidence": float(row.get("confidence") or 0.0),
            "coordinate_system": "egoallo_world",
            "length_unit": "meter",
        }

        matrix = row.get("T_world_from_object")
        if matrix is None or not bool(row.get("valid", False)):
            common.update({
                "T_world_from_object": None,
                "translation": None,
                "quaternion_q_wxyz": None,
            })
        else:
            T = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
            T_out = np.eye(4, dtype=np.float64)
            T_out[:3, :3] = R_final @ T[:3, :3]
            T_out[:3, 3] = R_final @ T[:3, 3] + t_floor
            T_out[:3, 3] = anchor + float(scale) * (T_out[:3, 3] - anchor)
            common.update({
                "T_world_from_object": T_out.flatten().tolist(),
                "translation": T_out[:3, 3].tolist(),
                "quaternion_q_wxyz": _rotation_to_quaternion_wxyz(T_out[:3, :3]),
            })
        rows.append(common)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False, engine="pyarrow")
    return {
        "input": str(objects_parquet_in),
        "output": str(output_path),
        "frames": int(len(rows)),
        "valid_objects": int(sum(1 for r in rows if r["valid"])),
    }


def scale_object_mesh_glb(input_path: Path, output_path: Path, scale: float) -> dict:
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError("trimesh is required to process object GLB meshes") from exc

    scene = trimesh.load(input_path, force="scene")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))
    return {
        "input": str(input_path),
        "output": str(output_path),
        "scale_applied": False,
        "global_scale": float(scale),
    }


def save_frontend_hand_and_wrist_jsonl(
    hands_vggt_jsonl: Path,
    hands_output_path: Path,
    R_final: np.ndarray,
    t_floor: np.ndarray,
    scale: float,
    floor_z: float,
    include_mesh: bool,
) -> dict:
    hands_output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = 0
    hands_count = 0
    with open(hands_vggt_jsonl, "r", encoding="utf-8") as src, open(
        hands_output_path, "w", encoding="utf-8"
    ) as hands_dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            rec_in = json.loads(line)
            frame_id = record_frame_id(rec_in, frames)
            timestamp = record_timestamp(rec_in, float(frames))
            src_frame_id = record_src_frame_id(rec_in, frame_id)

            hands_out = []
            for hand_idx, hand in enumerate(rec_in.get("hands", [])):
                side = str(hand.get("side", "")).lower()
                if side not in ("left", "right"):
                    raise ValueError(
                        f"{hands_vggt_jsonl}:{line_no} hand {hand_idx}: side must be left/right, got {side!r}"
                    )
                keypoints, keypoints_key = pick_world_hand_points(hand, ("keypoints_3d", "world_keypoints_3d"))
                if keypoints is None:
                    missing = "keypoints_3d/world_keypoints_3d" if keypoints_key is None else f"valid {keypoints_key}"
                    raise KeyError(f"{hands_vggt_jsonl}:{line_no} hand {hand_idx}: missing {missing}")

                keypoints_ego = transform_vggt_world_to_egoallo(
                    keypoints,
                    R_final,
                    t_floor,
                )
                wrist_ego = keypoints_ego[0]
                wrist_scaled = scale_world_points(
                    wrist_ego.reshape(1, 3),
                    scale,
                    floor_z,
                ).reshape(3)
                keypoints_scaled = (
                    wrist_scaled + (keypoints_ego - wrist_ego)
                ).astype(np.float32)
                wrist = wrist_scaled.astype(np.float32).tolist()
                conf = hand_confidence(hand)

                hand_out = {
                    "side": side,
                    "wrist": wrist,
                    "keypoints_3d": keypoints_scaled.tolist(),
                }
                if conf is not None:
                    hand_out["confidence"] = conf
                mesh, _ = pick_world_hand_points(hand, ("mesh_vertices_3d", "world_mesh_vertices_3d"))
                if include_mesh and mesh is not None:
                    mesh_ego = transform_vggt_world_to_egoallo(
                        mesh,
                        R_final,
                        t_floor,
                    )
                    mesh_scaled = (
                        wrist_scaled + (mesh_ego - wrist_ego)
                    ).astype(np.float32)
                    hand_out["mesh_vertices_3d"] = mesh_scaled.tolist()

                hands_out.append(hand_out)
                hands_count += 1

            base = {
                "frame_id": int(frame_id),
                "timestamp": float(timestamp),
                "src_frame_id": int(src_frame_id),
                "coordinate_system": FRONTEND_COORDINATE_SYSTEM,
            }
            hands_dst.write(
                json.dumps({**base, "hands": hands_out}, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            frames += 1

    return {
        "hands_output": str(hands_output_path),
        "frames": int(frames),
        "hands": int(hands_count),
    }


def reject_distant_hamer_hands(data: dict, max_wrist_distance_m: float) -> dict:
    if not np.isfinite(max_wrist_distance_m) or max_wrist_distance_m <= 0:
        raise ValueError("max_wrist_distance_m must be finite and positive")
    report = {"max_wrist_distance_m": float(max_wrist_distance_m)}
    for side in ("left", "right"):
        counts = dict(input=0, kept=0, rejected_far=0, rejected_invalid=0)
        detections = data[f"detections_{side}_wrt_cam"]
        for timestamp, det in detections.items():
            if det is None:
                continue
            points = np.asarray(det["keypoints_3d"])
            if points.ndim != 3 or points.shape[1:] != (21, 3):
                raise ValueError(f"Invalid {side} keypoints shape at {timestamp}: {points.shape}")
            distance = np.linalg.norm(points[:, 0], axis=1)
            valid = np.isfinite(points).all(axis=(1, 2)) & (points[:, 0, 2] > 0)
            keep = valid & (distance <= max_wrist_distance_m)
            counts["input"] += len(points)
            counts["kept"] += int(keep.sum())
            counts["rejected_far"] += int((valid & ~keep).sum())
            counts["rejected_invalid"] += int((~valid).sum())
            if not keep.any():
                # Retain timestamp slots; absence is not a zero-weight observation.
                detections[timestamp] = None
            elif not keep.all():
                detections[timestamp] = {
                    field: value[keep] if isinstance(value, np.ndarray)
                    and value.ndim > 0 and value.shape[0] == len(points) else value
                    for field, value in det.items()
                }
        report[side] = counts
    return report


def scale_hamer_pkl(
    input_path: Path,
    output_path: Path,
    scale: float,
    max_wrist_distance_m: float = 0.0,
    hand_geometry_scale: float | None = None,
) -> dict:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Hand metric scale must be finite and positive")
    if not np.isfinite(max_wrist_distance_m) or max_wrist_distance_m < 0:
        raise ValueError("max_wrist_distance_m must be finite and nonnegative")
    geometry_scale = float(scale if hand_geometry_scale is None else hand_geometry_scale)
    if not np.isfinite(geometry_scale) or geometry_scale <= 0:
        raise ValueError("hand_geometry_scale must be finite and positive")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    scaled_detections = 0
    for key in ("detections_left_wrt_cam", "detections_right_wrt_cam"):
        detections = data.get(key, {})
        for det in detections.values():
            if det is None:
                continue
            keypoints = np.asarray(det["keypoints_3d"], dtype=np.float32)
            if keypoints.ndim != 3 or keypoints.shape[2] != 3:
                raise ValueError(f"Invalid keypoints_3d shape: {keypoints.shape}")
            wrist = keypoints[:, 0, :]
            target_wrist = wrist * float(scale)
            for field in ("verts", "keypoints_3d"):
                if field in det and det[field] is not None:
                    points = np.asarray(det[field], dtype=np.float32)
                    if points.ndim != 3 or points.shape[0] != len(wrist) or points.shape[2] != 3:
                        raise ValueError(f"Invalid {field} shape: {points.shape}")
                    det[field] = (
                        target_wrist[:, None, :]
                        + geometry_scale * (points - wrist[:, None, :])
                    ).astype(np.float32)
            scaled_detections += 1

    distance_filter = {"enabled": max_wrist_distance_m > 0}
    if distance_filter["enabled"]:
        distance_filter.update(reject_distant_hamer_hands(data, max_wrist_distance_m))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "scene_translation_scale": float(scale),
        "hand_geometry_scale": geometry_scale,
        "scaled_detections": int(scaled_detections),
        "distance_filter": distance_filter,
    }


def add_stem_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix + path.suffix)


def default_output(output_dir: Path | None, input_path: Path) -> Path:
    if output_dir is not None:
        return output_dir / input_path.name
    return add_stem_suffix(input_path, "_scaled")


def infer_demo_name(head_trajectory_path: Path) -> str:
    parent = head_trajectory_path.expanduser().parent.name
    return parent if parent else "demo"


def safe_output_filename(name: str, *, default: str) -> str:
    """Validate a user-configurable JSONL output name.

    The pipeline passes these names from YAML through run_body_mesh.py. They
    must be plain filenames, not absolute paths or nested relative paths, so all
    front-end JSONL files remain inside --frontend-output-dir.
    """
    value = str(name or default).strip()
    if not value:
        value = default

    path = Path(value)
    if path.is_absolute() or path.name != value:
        raise ValueError(f"Output JSONL name must be a filename, not a path: {value!r}")
    if not value.endswith(".jsonl"):
        raise ValueError(f"Output JSONL name must end with .jsonl: {value!r}")
    return value


DEFAULT_HAND_SCALE_BONES = [
    {"name": "wrist_middle_mcp", "i": 0, "j": 9, "length_m": 0.095},
    {"name": "index_pinky_mcp", "i": 5, "j": 17, "length_m": 0.075},
]


def cap_scale_by_head_p90(
    scale: float,
    info: dict,
    head_poses: np.ndarray,
    floor_z: float,
    max_p90_head_height: float | None,
) -> tuple[float, dict]:
    info = dict(info)
    info["head_height_sanity_policy"] = "cap_high_p90_only"
    info["max_p90_head_height"] = None if max_p90_head_height is None else float(max_p90_head_height)
    info["capped_by_head_p90"] = False
    info["scale_before_head_p90_cap"] = float(scale)

    if max_p90_head_height is None or float(max_p90_head_height) <= 0.0:
        info["head_height_sanity_policy"] = "disabled"
        return float(scale), info

    max_p90 = float(max_p90_head_height)
    if not np.isfinite(max_p90):
        info["head_p90_cap_warning"] = "invalid_max_p90_head_height"
        return float(scale), info

    heights = _finite_head_heights(head_poses, floor_z)
    original_p90 = _safe_percentile(heights, 90)
    info["original_height_p90_for_cap"] = original_p90
    if original_p90 is None or not np.isfinite(original_p90) or float(original_p90) <= 1e-8:
        info["head_p90_cap_warning"] = "invalid_original_head_p90"
        return float(scale), info

    scaled_p90 = float(original_p90) * float(scale)
    info["scaled_height_p90_before_cap"] = scaled_p90
    if scaled_p90 <= max_p90:
        return float(scale), info

    info.update({
        "capped_by_head_p90": True,
        "scaled_height_p90_after_cap": float(max_p90),
    })

    capped_scale = max_p90 / float(original_p90)
    info.update({
        "scale": float(capped_scale),
        "head_p90_cap_warning": (
            f"head p90 cap applied: "
            f"scaled p90 head height {scaled_p90:.6f}m exceeds {max_p90:.6f}m, "
            f"scale changed {float(scale):.9f} -> {float(capped_scale):.9f}."
        ),
    })
    return float(capped_scale), info


def relative_mad(values: np.ndarray, center: float | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    med = float(np.median(values)) if center is None else float(center)
    denom = max(abs(med), 1e-8)
    return float(np.median(np.abs(values - med)) / denom)


def _pick_hand_side(hand: dict) -> str | None:
    side = str(hand.get("side", "")).strip().lower()
    return side if side in ("left", "right") else None


def _pick_hand_keypoints(hand: dict) -> tuple[np.ndarray | None, str | None]:
    # Priority: explicit hand frame keypoints -> world -> camera variants.
    # Existing hand JSONL may only contain world/camera fields; keep a fallback
    # path for compatibility.
    for key in ("keypoints_3d", "world_keypoints_3d", "camera_keypoints_3d", "hawor_camera_keypoints_3d"):
        if key not in hand:
            continue
        arr = np.asarray(hand[key], dtype=np.float64)
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 3:
            arr = arr[0]
        if arr.ndim == 1 and arr.size > 0 and arr.shape[0] % 3 == 0:
            arr = arr.reshape(-1, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None, key
        return arr, key
    return None, None


def _filter_bones_to_palm(
    bones: list[dict],
    kps: np.ndarray,
) -> list[dict]:
    # Keep wrist + 4 MCP indices commonly used by HaWoR: 0,5,9,13,17.
    allowed = {0, 5, 9, 13, 17}
    if kps.shape[0] <= max(allowed):
        return []
    out = []
    for bone in bones:
        i = int(bone["i"])
        j = int(bone["j"])
        if i in allowed and j in allowed:
            out.append({
                "name": bone["name"],
                "i": i,
                "j": j,
                "length_m": float(bone["length_m"]),
            })
    return out


def _iter_hands_by_side(
    hands_jsonl: Path,
    *,
    r_align: Path,
    align_meta: Path,
) -> tuple[list[tuple[float, np.ndarray, float, str, int, int]], list[tuple[float, np.ndarray, float, str, int, int]], int, int, list[str], list[str]]:
    """
    Parse hand JSONL and return per-side candidate lists and summary stats.
    Returned lists are tuples of (timestamp, keypoints, confidence, source, line_no, hand_index).
    """
    R_final, t_floor = load_vggt_to_egoallo_transform(Path(r_align), Path(align_meta))
    hands: dict[str, list[tuple[float, np.ndarray, float, str, int, int]]] = {
        "left": [],
        "right": [],
    }
    total_frames = 0
    total_hands = 0
    warnings: list[str] = []
    errors: list[str] = []

    with Path(hands_jsonl).expanduser().resolve().open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_frames += 1
            try:
                rec = json.loads(line)
            except Exception as exc:
                errors.append(f"line{line_no}: json_error={exc!r}")
                continue

            ts = record_timestamp(rec, float(line_no))
            hands_in_frame = rec.get("hands", [])
            if not isinstance(hands_in_frame, list):
                continue

            # Keep one detection per hand side in each source frame; pick the
            # highest-confidence one so each side contributes at most one scale
            # sample per frame.
            per_frame_best: dict[str, tuple[float, np.ndarray, float, str, int, int] | None] = {
                "left": None,
                "right": None,
            }

            for hand_idx, hand in enumerate(hands_in_frame):
                if not isinstance(hand, dict):
                    continue
                side = _pick_hand_side(hand)
                if side is None:
                    continue

                hand_kps, used_key = _pick_hand_keypoints(hand)
                if hand_kps is None:
                    warnings.append(f"line{line_no}: hand{hand_idx} side={side}:missing/invalid keypoints")
                    continue
                if not np.isfinite(hand_kps).all():
                    warnings.append(f"line{line_no}: hand{hand_idx} side={side}:hand_keypoints_contains_nan_or_inf")
                    continue

                hand_kps = transform_vggt_world_to_egoallo(hand_kps, R_final, t_floor)
                conf = hand_confidence(hand)
                if conf is None or not np.isfinite(conf):
                    conf = 1.0
                else:
                    conf = float(conf)

                current = per_frame_best[side]
                if current is None or float(conf) > float(current[2]):
                    per_frame_best[side] = (float(ts), hand_kps, float(conf), used_key, int(line_no), int(hand_idx))
                total_hands += 1

            for side, chosen in per_frame_best.items():
                if chosen is not None:
                    hands[side].append(chosen)

    return hands["left"], hands["right"], total_frames, total_hands, warnings, errors


def _estimate_scale_from_bone_set(
    hand_kps: np.ndarray,
    bones: list[dict],
    min_scale: float,
    max_scale: float,
    max_relative_mad: float,
    min_valid_bones: int,
) -> tuple[float | None, dict]:
    # Each frame contributes one scale; reject bad frames by bone ratio stability.
    per_bone_scales: list[float] = []
    per_bone_meta: dict[str, float] = {}
    for bone in bones:
        i, j, target_len = int(bone["i"]), int(bone["j"]), float(bone["length_m"])
        if i < 0 or j < 0 or i >= len(hand_kps) or j >= len(hand_kps):
            continue
        if target_len <= 0:
            continue
        observed = float(np.linalg.norm(hand_kps[i] - hand_kps[j]))
        if not np.isfinite(observed) or observed <= 1e-8:
            continue
        scale = target_len / observed
        if not np.isfinite(scale) or scale < min_scale or scale > max_scale:
            continue
        per_bone_scales.append(float(scale))
        per_bone_meta[f"{bone.get('name', f'{i}_{j}')}_scale"] = float(scale)

    arr = np.asarray(per_bone_scales, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < int(max(1, min_valid_bones)):
        return None, {
            "status": "invalid",
            "num_valid_bones": int(arr.size),
            "reason": "not_enough_bones_in_frame",
            "per_bone": per_bone_meta,
        }

    frame_scale = float(np.median(arr))
    rel = relative_mad(arr, frame_scale)
    if rel > float(max_relative_mad):
        return None, {
            "status": "invalid",
            "num_valid_bones": int(arr.size),
            "relative_mad": float(rel),
            "reason": "frame_relative_mad_too_large",
            "per_bone": per_bone_meta,
            "frame_scale": frame_scale,
        }

    return frame_scale, {
        "status": "ok",
        "num_valid_bones": int(arr.size),
        "relative_mad": float(rel),
        "frame_scale": float(frame_scale),
        "per_bone": per_bone_meta,
        "bone_scales_median": float(frame_scale),
        "bone_scales": [float(x) for x in arr.tolist()],
    }


def _collect_hand_scale_samples(
    side_records: list[tuple[float, np.ndarray, float, str, int, int]],
    *,
    bones: list[dict],
    min_scale: float,
    max_scale: float,
    max_relative_mad: float,
    min_valid_bones: int,
) -> tuple[list[tuple[float, float, float]], dict[str, int], list[dict]]:
    frame_scales: list[tuple[float, float, float]] = []
    per_bone_per_scale: list[dict] = []
    summary = {"frame_total": 0, "frame_valid": 0}
    for ts, hand_kps, conf, used_key, _line_no, _hand_idx in side_records:
        summary["frame_total"] += 1
        if used_key is None:
            continue
        frame_bones = _filter_bones_to_palm(bones, hand_kps)
        if not frame_bones:
            continue
        frame_min_bones = max(1, min(int(min_valid_bones), len(frame_bones)))
        frame_scale, scale_meta = _estimate_scale_from_bone_set(
            hand_kps=hand_kps,
            bones=frame_bones,
            min_scale=float(min_scale),
            max_scale=float(max_scale),
            max_relative_mad=float(max_relative_mad),
            min_valid_bones=int(frame_min_bones),
        )
        if frame_scale is None:
            continue
        summary["frame_valid"] += 1
        if conf is None or not np.isfinite(conf):
            conf = 1.0
        frame_scales.append((float(ts), float(frame_scale), float(conf)))
        per_bone_per_scale.append(scale_meta)
    return frame_scales, summary, per_bone_per_scale


def _finite_head_heights(head_poses: np.ndarray, floor_z: float) -> np.ndarray:
    heights = np.asarray(head_poses[:, 2, 3], dtype=np.float64) - float(floor_z)
    return heights[np.isfinite(heights)]


def _safe_percentile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, float(q)))


def height_diagnostics(head_poses: np.ndarray, floor_z: float, scale: float) -> dict:
    heights = _finite_head_heights(head_poses, floor_z)
    if heights.size == 0:
        return {
            "floor_z": float(floor_z),
            "original_height_min": None,
            "original_height_median": None,
            "original_height_mean": None,
            "original_height_max": None,
            "original_height_p75": None,
            "original_height_p90": None,
            "original_height_p95": None,
            "scale": float(scale),
            "scaled_height_min": None,
            "scaled_height_median": None,
            "scaled_height_mean": None,
            "scaled_height_max": None,
            "scaled_height_p75": None,
            "scaled_height_p90": None,
            "scaled_height_p95": None,
            "invalid_median_height": True,
        }
    scaled = heights * float(scale)
    median_height = float(np.median(heights))
    scaled_median = float(np.median(scaled))
    return {
        "floor_z": float(floor_z),
        "original_height_min": float(np.min(heights)),
        "original_height_median": median_height,
        "original_height_mean": float(np.mean(heights)),
        "original_height_max": float(np.max(heights)),
        "original_height_p75": _safe_percentile(heights, 75),
        "original_height_p90": _safe_percentile(heights, 90),
        "original_height_p95": _safe_percentile(heights, 95),
        "scale": float(scale),
        "scaled_height_min": float(np.min(scaled)),
        "scaled_height_median": scaled_median,
        "scaled_height_mean": float(np.mean(scaled)),
        "scaled_height_max": float(np.max(scaled)),
        "scaled_height_p75": _safe_percentile(scaled, 75),
        "scaled_height_p90": _safe_percentile(scaled, 90),
        "scaled_height_p95": _safe_percentile(scaled, 95),
        "invalid_median_height": bool((not np.isfinite(median_height)) or median_height <= 1e-8),
    }


def try_depth_scale(path: Path | None) -> tuple[float | None, dict]:
    info: dict = {
        "source": "scale_json",
        "status": "failed",
        "scale": None,
    }
    if path is None:
        info["reason"] = "scale_json_not_provided"
        return None, info
    path = path.expanduser().resolve()
    info["path"] = str(path)
    if not path.is_file():
        info["reason"] = "scale_json_not_found"
        return None, info
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        source = str(data.get("source") or "metric_depth")
        info["source"] = source
        info["raw"] = {
            "path": str(path),
            "status": data.get("status"),
            "scale": data.get("scale"),
            "source": source,
        }
        status = data.get("status")
        if status != "ok":
            info["reason"] = f"scale_json_status_{status or 'missing'}"
            return None, info
        if "scale" not in data:
            info["reason"] = "scale_json_scale_missing"
            return None, info
        scale = float(data["scale"])
        if not np.isfinite(scale) or scale <= 0.0:
            info["reason"] = "invalid_scale_json_scale"
            return None, info
        info.update({
            "status": "ok",
            "reason": data.get("reason", "metric_depth_floor_scale"),
            "scale": float(scale),
            "num_valid_frames": data.get("num_valid_frames"),
            "relative_mad": data.get("relative_mad"),
        })
        return float(scale), info
    except Exception as exc:
        info["reason"] = "exception"
        info["exception"] = repr(exc)
        return None, info


def load_hand_bones(path: Path | None) -> list[dict]:
    if path is None:
        return list(DEFAULT_HAND_SCALE_BONES)
    path = path.expanduser().resolve()
    if not path.is_file():
        return list(DEFAULT_HAND_SCALE_BONES)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("bones", [])
    if not isinstance(data, list) or not data:
        return list(DEFAULT_HAND_SCALE_BONES)
    bones = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            bones.append({
                "name": str(item.get("name", f"{int(item['i'])}_{int(item['j'])}")),
                "i": int(item["i"]),
                "j": int(item["j"]),
                "length_m": float(item["length_m"]),
            })
        except Exception:
            continue
    return bones or list(DEFAULT_HAND_SCALE_BONES)


def try_hand_scale(
    *,
    hands_jsonl: Path | None,
    r_align: Path | None,
    align_meta: Path | None,
    bones_json: Path | None,
    min_valid_bones: int,
    max_relative_mad: float,
    min_scale: float,
    max_scale: float,
) -> tuple[float | None, dict]:
    info: dict = {
        "source": "hands",
        "status": "failed",
        "scale": None,
    }
    try:
        missing = []
        for name, value in (
            ("hands_jsonl", hands_jsonl),
            ("R_align", r_align),
            ("align_meta", align_meta),
        ):
            if value is None:
                missing.append(name)
            elif not Path(value).expanduser().is_file():
                missing.append(f"{name}:{value}")
        if missing:
            info["reason"] = "missing_inputs"
            info["missing"] = missing
            return None, info

        bones = load_hand_bones(bones_json)
        if not bones:
            info["reason"] = "no_valid_hand_bones_config"
            return None, info

        left_raw, right_raw, total_frames, total_hands, warnings, parse_errors = _iter_hands_by_side(
            Path(hands_jsonl),
            r_align=Path(r_align),
            align_meta=Path(align_meta),
        )

        left_scales, left_summary, left_bone_meta = _collect_hand_scale_samples(
            left_raw,
            bones=bones,
            min_scale=min_scale,
            max_scale=max_scale,
            max_relative_mad=max_relative_mad,
            min_valid_bones=int(min_valid_bones),
        )
        right_scales, right_summary, right_bone_meta = _collect_hand_scale_samples(
            right_raw,
            bones=bones,
            min_scale=min_scale,
            max_scale=max_scale,
            max_relative_mad=max_relative_mad,
            min_valid_bones=int(min_valid_bones),
        )

        all_scales = left_scales + right_scales
        if not all_scales:
            info["status"] = "failed"
            info["reason"] = "no_valid_frame_scale"
            info["parse_warnings"] = warnings
            info["parse_errors"] = parse_errors
            info["frames"] = int(total_frames)
            info["hands"] = int(total_hands)
            info["left_frames_total"] = int(left_summary["frame_total"])
            info["left_frames_valid"] = int(left_summary["frame_valid"])
            info["right_frames_total"] = int(right_summary["frame_total"])
            info["right_frames_valid"] = int(right_summary["frame_valid"])
            info["per_bone_scale_meta"] = left_bone_meta + right_bone_meta
            info["bones"] = bones
            return None, info

        if (len(all_scales) < int(max(1, min_valid_bones))):
            info["status"] = "failed"
            info["reason"] = "not_enough_valid_frames"
            info["min_valid_bones"] = int(min_valid_bones)
            info["valid_frames"] = int(len(all_scales))
            info["parse_warnings"] = warnings
            info["parse_errors"] = parse_errors
            info["frames"] = int(total_frames)
            info["hands"] = int(total_hands)
            info["left_frames_total"] = int(left_summary["frame_total"])
            info["left_frames_valid"] = int(left_summary["frame_valid"])
            info["right_frames_total"] = int(right_summary["frame_total"])
            info["right_frames_valid"] = int(right_summary["frame_valid"])
            info["per_bone_scale_meta"] = left_bone_meta + right_bone_meta
            return None, info

        scales = np.asarray([x[1] for x in all_scales], dtype=np.float64)
        scale = float(np.median(scales))
        conf = float(np.mean([x[2] for x in all_scales], dtype=np.float64))
        rel = relative_mad(scales, scale)

        if rel > float(max_relative_mad):
            info["status"] = "failed"
            info["reason"] = "global_relative_mad_too_large"
            info["scale"] = None
            info["relative_mad"] = float(rel)
            info["median_scale"] = float(scale)
            info["min_valid_bones"] = int(min_valid_bones)
            info["min_scale"] = float(min_scale)
            info["max_scale"] = float(max_scale)
            info["max_relative_mad"] = float(max_relative_mad)
            info["parse_warnings"] = warnings
            info["parse_errors"] = parse_errors
            info["left_frames_total"] = int(left_summary["frame_total"])
            info["left_frames_valid"] = int(left_summary["frame_valid"])
            info["right_frames_total"] = int(right_summary["frame_total"])
            info["right_frames_valid"] = int(right_summary["frame_valid"])
            info["per_bone_scale_meta"] = left_bone_meta + right_bone_meta
            return None, info

        info.update({
            "status": "ok",
            "reason": "global_hand_scale_median",
            "scale": float(scale),
            "selected_scale": float(scale),
            "confidence": float(conf),
            "relative_mad": float(rel),
            "num_valid_scales": int(scales.size),
            "min_valid_bones": int(min_valid_bones),
            "max_relative_mad": float(max_relative_mad),
            "min_scale": float(min_scale),
            "max_scale": float(max_scale),
            "left_frames_total": int(left_summary["frame_total"]),
            "left_frames_valid": int(left_summary["frame_valid"]),
            "right_frames_total": int(right_summary["frame_total"]),
            "right_frames_valid": int(right_summary["frame_valid"]),
            "parse_warnings": warnings,
            "parse_errors": parse_errors,
            "bones": bones,
            "frames": int(total_frames),
            "hands": int(total_hands),
            "per_bone_scale_meta": left_bone_meta + right_bone_meta,
        })
        return float(scale), info

    except Exception as exc:
        info["reason"] = "exception"
        info["exception"] = repr(exc)
        return None, info


def decide_metric_scale(args: argparse.Namespace, head_poses: np.ndarray) -> tuple[float, dict]:
    attempts: list[dict] = []
    candidates: list[tuple[str, float, dict]] = []

    depth_scale, depth_info = try_depth_scale(args.depth_scale_json)
    attempts.append(depth_info)
    if depth_scale is not None:
        source = str(depth_info.get("source") or "metric_depth")
        candidates.append((source, float(depth_scale), depth_info))

    if bool(args.hand_scale_enabled):
        hand_scale, hand_info = try_hand_scale(
            hands_jsonl=args.hands_vggt_jsonl_for_scale or args.hands_vggt_jsonl_in,
            r_align=args.r_align_for_scale or args.r_align,
            align_meta=args.align_meta_for_scale or args.align_meta,
            bones_json=args.hand_scale_bones_json,
            min_valid_bones=args.hand_scale_min_valid_bones,
            max_relative_mad=args.hand_scale_max_relative_mad,
            min_scale=args.hand_scale_min_scale,
            max_scale=args.hand_scale_max_scale,
        )
    else:
        hand_scale = None
        hand_info = {"source": "hands", "status": "failed", "reason": "disabled", "scale": None}
    attempts.append(hand_info)
    if hand_scale is not None:
        candidates.append(("hands", float(hand_scale), hand_info))

    if candidates:
        selected_source, raw_scale, selected_info = max(candidates, key=lambda x: x[1])
        selected_info = dict(selected_info)
        selected_source = str(selected_info.get("source") or selected_source)
        selected_info["selected_before_head_p90_cap"] = True
        selected_info["raw_selected_scale"] = float(raw_scale)
        scale, selected_info = cap_scale_by_head_p90(
            raw_scale,
            selected_info,
            head_poses,
            args.floor_z,
            args.max_p90_head_height,
        )
        for i, attempt in enumerate(attempts):
            if attempt.get("source") == selected_source:
                attempts[i] = selected_info
                break
        if selected_source == depth_info.get("source"):
            depth_info = selected_info
        elif selected_source == "hands":
            hand_info = selected_info

        diag = height_diagnostics(head_poses, args.floor_z, scale)
        diag.update({
            "reason": "metric_depth_hands_max",
            "selected_source": selected_source,
            "scale_method": "max_metric_depth_hands_then_p90_cap",
            "attempts": attempts,
            "depth": depth_info,
            "hands": hand_info,
            "candidate_scales": [
                {"source": source, "scale": float(scale_value)}
                for source, scale_value, _info in candidates
            ],
            "raw_selected_scale": float(raw_scale),
            "max_p90_head_height": float(args.max_p90_head_height),
            "warning": selected_info.get("head_p90_cap_warning"),
        })
        if selected_info.get("head_p90_cap_warning"):
            print(f"[Warning] {selected_info.get('head_p90_cap_warning')}", flush=True)
        return float(scale), diag

    scale = 1.0
    diag = height_diagnostics(head_poses, args.floor_z, scale)
    diag.update({
        "reason": "none",
        "selected_source": "none",
        "scale_method": "no_scale_warning",
        "attempts": attempts,
        "warning": "Metric depth and hand scale both failed; continue with scale=1.0.",
    })
    print(f"[Warning] {diag['warning']}", flush=True)
    return scale, diag


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply metric scale correction and export pipeline outputs. "
            "This version is intended to be called only by run_body_mesh.py; "
            "all YAML/config values are resolved there and passed as CLI args."
        )
    )

    # Required pipeline inputs.
    parser.add_argument("--head-trajectory-in", required=True, type=Path)
    parser.add_argument("--point-cloud-in", required=True, type=Path)
    parser.add_argument("--hamer-pkl-in", required=True, type=Path)
    parser.add_argument(
        "--point-cloud-space",
        choices=("egoallo", "raw_vggt"),
        default="egoallo",
        help="Coordinate space of --point-cloud-in. raw_vggt is aligned inside this script before final output.",
    )
    parser.add_argument("--point-cloud-r-align", type=Path, default=None)
    parser.add_argument("--point-cloud-align-meta", type=Path, default=None)
    parser.add_argument(
        "--final-scene-output-voxel-size",
        type=float,
        default=0.0,
        help="Voxel size for final_scene.ply downsampling; 0 disables downsampling.",
    )

    # Required pipeline outputs.
    parser.add_argument("--head-trajectory-out", required=True, type=Path)
    parser.add_argument("--point-cloud-out", required=True, type=Path)
    parser.add_argument("--hamer-pkl-out", required=True, type=Path)
    parser.add_argument(
        "--hand-max-camera-distance-m", type=float, default=0.0,
        help="Reject camera-to-wrist distances above this after metric scaling; 0 disables.",
    )
    parser.add_argument(
        "--metadata-out",
        required=True,
        type=Path,
        help="Write metric_scale_metadata.json. In the pipeline this is <output_dir>/tmp/metric_scale_metadata.json.",
    )

    # Metric-scale parameters. These are normally read from YAML by run_body_mesh.py.
    parser.add_argument("--floor-z", type=float, default=0.0)
    parser.add_argument("--depth-scale-json", type=Path, default=None)
    parser.add_argument("--hand-scale-enabled", type=int, default=1)
    parser.add_argument("--hand-scale-bones-json", type=Path, default=None)
    parser.add_argument("--hand-scale-min-valid-bones", type=int, default=20)
    parser.add_argument("--hand-scale-max-relative-mad", type=float, default=0.35)
    parser.add_argument("--hand-scale-min-scale", type=float, default=0.05)
    parser.add_argument("--hand-scale-max-scale", type=float, default=20.0)
    parser.add_argument(
        "--max-p90-head-height",
        type=float,
        default=1.60,
        help="Cap the selected scale so p90 head height does not exceed this value; 0 disables.",
    )

    parser.add_argument("--hands-vggt-jsonl-for-scale", type=Path, default=None)
    parser.add_argument("--R-align-for-scale", dest="r_align_for_scale", type=Path, default=None)
    parser.add_argument("--align-meta-for-scale", type=Path, default=None)

    # Front-end JSONL export. The filenames are passed by run_body_mesh.py
    # from YAML via --frontend-head-jsonl-name / --frontend-hands-jsonl-name.
    parser.add_argument("--export-frontend-jsonl", action="store_true")
    parser.add_argument(
        "--frontend-output-dir",
        type=Path,
        default=None,
        help="Directory to write front-end JSONL files. Pipeline uses <output_dir>/egogvae_outputs.",
    )
    parser.add_argument(
        "--frontend-head-jsonl-name",
        default="head_trajectory_egoallo.jsonl",
        help="Filename for exported head trajectory JSONL, not a path.",
    )
    parser.add_argument(
        "--frontend-hands-jsonl-name",
        default="hand_keypoints_egoallo.jsonl",
        help="Filename for exported hand keypoints JSONL, not a path.",
    )
    parser.add_argument(
        "--timestamps-txt",
        type=Path,
        default=None,
        help="timestamps.txt for front-end JSONL metadata. Pipeline uses <output_dir>/egogvae_inputs/timestamps.txt.",
    )
    parser.add_argument(
        "--hands-vggt-jsonl-in",
        type=Path,
        default=None,
        help="Original HaWoR/VGGT-world hands JSONL for hand_keypoints_egoallo.jsonl export.",
    )
    parser.add_argument(
        "--R-align",
        dest="r_align",
        type=Path,
        default=None,
        help="R_align.npy from 01_estimate_gravity_da3_geocalib.py.",
    )
    parser.add_argument(
        "--align-meta",
        type=Path,
        default=None,
        help="alignment_transform.npz from 01_estimate_gravity_da3_geocalib.py.",
    )
    parser.add_argument(
        "--frontend-include-mesh",
        action="store_true",
        help="Also export scaled mesh_vertices_3d in hand_keypoints_egoallo.jsonl. This can be very large.",
    )
    parser.add_argument("--objects-parquet-in", type=Path, default=None)
    parser.add_argument("--objects-parquet-out", type=Path, default=None)
    parser.add_argument("--object-mesh-in", type=Path, action="append", default=None)
    parser.add_argument("--object-mesh-out", type=Path, action="append", default=None)
    parser.add_argument(
        "--probe-scale-only",
        action="store_true",
        help="Only decide metric scale and write metadata; do not write scaled outputs.",
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    head_poses = load_head_trajectory(args.head_trajectory_in)
    scale, scale_info = decide_metric_scale(args, head_poses)

    metadata = {
        "scale_info": scale_info,
        "scale_source": scale_info.get("selected_source", scale_info.get("reason")),
        "scale_method": scale_info.get("scale_method", scale_info.get("reason")),
        "scale_warning": scale_info.get("warning"),
        "inputs": {
            "head_trajectory_in": str(args.head_trajectory_in),
            "point_cloud_in": str(args.point_cloud_in),
            "hamer_pkl_in": str(args.hamer_pkl_in),
            "timestamps_txt": str(args.timestamps_txt) if args.timestamps_txt is not None else None,
            "hands_vggt_jsonl_in": str(args.hands_vggt_jsonl_in) if args.hands_vggt_jsonl_in is not None else None,
            "R_align": str(args.r_align) if args.r_align is not None else None,
            "align_meta": str(args.align_meta) if args.align_meta is not None else None,
            "point_cloud_space": str(args.point_cloud_space),
            "point_cloud_r_align": str(args.point_cloud_r_align) if args.point_cloud_r_align is not None else None,
            "point_cloud_align_meta": str(args.point_cloud_align_meta) if args.point_cloud_align_meta is not None else None,
            "final_scene_output_voxel_size": float(args.final_scene_output_voxel_size),
            "depth_scale_json": str(args.depth_scale_json) if args.depth_scale_json is not None else None,
            "hand_scale_bones_json": str(args.hand_scale_bones_json) if args.hand_scale_bones_json is not None else None,
            "hands_vggt_jsonl_for_scale": str(args.hands_vggt_jsonl_for_scale) if args.hands_vggt_jsonl_for_scale is not None else None,
            "R_align_for_scale": str(args.r_align_for_scale) if args.r_align_for_scale is not None else None,
            "align_meta_for_scale": str(args.align_meta_for_scale) if args.align_meta_for_scale is not None else None,
        },
        "outputs": {},
    }

    print("==== Metric scale correction ====")
    print(f"head_trajectory_in: {args.head_trajectory_in}")
    print(f"point_cloud_in: {args.point_cloud_in}")
    print(f"hamer_pkl_in: {args.hamer_pkl_in}")
    print(f"floor_z: {args.floor_z:.6f}")
    orig_med = scale_info.get("original_height_median")
    if orig_med is not None:
        print(f"original median head height: {float(orig_med):.6f}")
    else:
        print("original median head height: None")
    print(f"scale reason: {scale_info['reason']}")
    print(f"selected scale source: {scale_info.get('selected_source', scale_info.get('reason'))}")
    print(f"selected scale method: {scale_info.get('scale_method', scale_info.get('reason'))}")
    if scale_info.get("warning"):
        print(f"[Warning] {scale_info.get('warning')}")
    print(f"scale: {scale:.9f}")
    scaled_med = scale_info.get("scaled_height_median")
    scaled_p90 = scale_info.get("scaled_height_p90")
    if scaled_med is not None:
        print(f"scaled median head height: {float(scaled_med):.6f}")
    else:
        print("scaled median head height: None")
    if scaled_p90 is not None:
        print(f"scaled p90 head height: {float(scaled_p90):.6f}")

    if args.probe_scale_only:
        metadata["outputs"]["probe_scale_only"] = True
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_json = _coerce_json_friendly(metadata)
        with args.metadata_out.open("w", encoding="utf-8") as f:
            json.dump(metadata_json, f, ensure_ascii=False, indent=2)
        print(f"Probe scale only; wrote metadata: {args.metadata_out}")
        return

    if args.dry_run:
        print("Dry run enabled; no files written.")
        return

    metadata["outputs"]["head_trajectory"] = save_head_trajectory(
        args.head_trajectory_in,
        args.head_trajectory_out,
        scale,
        args.floor_z,
    )
    print(f"Saved scaled head trajectory: {args.head_trajectory_out}")

    metadata["outputs"]["point_cloud"] = transform_and_scale_point_cloud(
        args.point_cloud_in,
        args.point_cloud_out,
        scale=scale,
        floor_z=args.floor_z,
        input_space=args.point_cloud_space,
        r_align_path=args.point_cloud_r_align,
        align_meta_path=args.point_cloud_align_meta,
        output_voxel_size=args.final_scene_output_voxel_size,
    )
    print(f"Saved scaled point cloud: {args.point_cloud_out}")

    hand_distance_limit = args.hand_max_camera_distance_m
    if scale_info.get("selected_source") == "none" and hand_distance_limit > 0:
        print("[Hand filter] metric distance skipped: no valid metric scale", flush=True)
        hand_distance_limit = 0.0
    hand_geometry_scale = scale
    hand_scale_info = scale_info.get("hands")
    if isinstance(hand_scale_info, dict) and hand_scale_info.get("status") == "ok":
        candidate = hand_scale_info.get("scale")
        if candidate is not None and np.isfinite(float(candidate)) and float(candidate) > 0:
            hand_geometry_scale = float(candidate)
    metadata["outputs"]["hamer_pkl"] = scale_hamer_pkl(
        args.hamer_pkl_in,
        args.hamer_pkl_out,
        scale,
        max_wrist_distance_m=hand_distance_limit,
        hand_geometry_scale=hand_geometry_scale,
    )
    print("[Hand filter] metric: " + json.dumps(metadata["outputs"]["hamer_pkl"]["distance_filter"]))
    print(f"Saved scaled hamer pkl: {args.hamer_pkl_out}")

    if args.export_frontend_jsonl:
        missing = []
        for name, value in (
            ("--frontend-output-dir", args.frontend_output_dir),
            ("--timestamps-txt", args.timestamps_txt),
            ("--hands-vggt-jsonl-in", args.hands_vggt_jsonl_in),
            ("--R-align", args.r_align),
            ("--align-meta", args.align_meta),
        ):
            if value is None:
                missing.append(name)
        if missing:
            raise ValueError("--export-frontend-jsonl requires " + ", ".join(missing))

        args.frontend_output_dir.mkdir(parents=True, exist_ok=True)
        frontend_head_out = args.frontend_output_dir / safe_output_filename(
            args.frontend_head_jsonl_name,
            default="head_trajectory_egoallo.jsonl",
        )
        frontend_hands_out = args.frontend_output_dir / safe_output_filename(
            args.frontend_hands_jsonl_name,
            default="hand_keypoints_egoallo.jsonl",
        )

        R_final, t_floor = load_vggt_to_egoallo_transform(args.r_align, args.align_meta)
        metadata["outputs"]["frontend_head_jsonl"] = save_frontend_head_jsonl(
            args.head_trajectory_out,
            frontend_head_out,
            args.timestamps_txt,
        )
        metadata["outputs"]["frontend_hand_jsonl"] = save_frontend_hand_and_wrist_jsonl(
            args.hands_vggt_jsonl_in,
            frontend_hands_out,
            R_final,
            t_floor,
            scale,
            args.floor_z,
            args.frontend_include_mesh,
        )
        print(f"Saved front-end head JSONL: {frontend_head_out}")
        print(f"Saved front-end hand keypoints JSONL: {frontend_hands_out}")

    if args.objects_parquet_in is not None or args.objects_parquet_out is not None:
        missing = []
        for name, value in (
            ("--objects-parquet-in", args.objects_parquet_in),
            ("--objects-parquet-out", args.objects_parquet_out),
            ("--R-align", args.r_align),
            ("--align-meta", args.align_meta),
        ):
            if value is None:
                missing.append(name)
        if missing:
            raise ValueError("--objects-parquet-out requires " + ", ".join(missing))

        R_final, t_floor = load_vggt_to_egoallo_transform(args.r_align, args.align_meta)
        metadata["outputs"]["frontend_objects_parquet"] = save_frontend_objects_parquet(
            args.objects_parquet_in,
            args.objects_parquet_out,
            R_final,
            t_floor,
            scale,
            args.floor_z,
        )
        print(f"Saved front-end objects parquet: {args.objects_parquet_out}")

    object_mesh_ins = list(args.object_mesh_in or [])
    object_mesh_outs = list(args.object_mesh_out or [])
    if object_mesh_ins or object_mesh_outs:
        if len(object_mesh_ins) != len(object_mesh_outs):
            raise ValueError("--object-mesh-in and --object-mesh-out must have the same count")
        metadata["outputs"]["object_meshes"] = [
            scale_object_mesh_glb(mesh_in, mesh_out, scale)
            for mesh_in, mesh_out in zip(object_mesh_ins, object_mesh_outs)
        ]
        print(f"Saved scaled object meshes: {len(object_mesh_outs)}")

    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = _coerce_json_friendly(metadata)
    with args.metadata_out.open("w", encoding="utf-8") as f:
        json.dump(metadata_json, f, ensure_ascii=False, indent=2)
    print(f"Saved metric scale metadata: {args.metadata_out}")


if __name__ == "__main__":
    main()
