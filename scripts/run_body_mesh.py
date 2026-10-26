#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import yaml

sys.dont_write_bytecode = True

# 本文件固定放在 project/scripts/run_body_mesh.py
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from runtime_threads import BLAS_THREAD_ENV, preprocessing_env, thread_count

CONVERT_TO_EGO_DIR = SCRIPTS_DIR / "convert_to_ego"
ESMATEG_DIR = SCRIPTS_DIR / "esmateG"

SCRIPT_01_CAMERA_POSES = CONVERT_TO_EGO_DIR / "01_convert_jsonl_to_camera_poses.py"
SCRIPT_02_EXTRACT_HANDS = CONVERT_TO_EGO_DIR / "02_extract_hawor_world_hands.py"
SCRIPT_03_STABILIZE_HANDS = CONVERT_TO_EGO_DIR / "03_stabilize_hawor_hands_for_egoallo.py"
SCRIPT_EXTRACT_ROT = CONVERT_TO_EGO_DIR / "extract_hawor_to_vggt_rotation.py"
SCRIPT_04_BUILD_HAMER = CONVERT_TO_EGO_DIR / "04_build_egoallo_hamer_from_hawor_world.py"
SCRIPT_PREPARE_REAL_SENSOR = CONVERT_TO_EGO_DIR / "04_prepare_real_sensor_inputs.py"
SCRIPT_ESTIMATE_FOUNDATIONGEO_SCALE = CONVERT_TO_EGO_DIR / "04_estimate_foundationgeo_metric_scale.py"
SCRIPT_05_APPLY_SCALE = CONVERT_TO_EGO_DIR / "05_apply_metric_scale.py"

SCRIPT_RUN_GEOCALIB = ESMATEG_DIR / "run_geocalib.py"
SCRIPT_ESTIMATE_GRAVITY = ESMATEG_DIR / "01_estimate_gravity_da3_geocalib.py"
SCRIPT_APPLY_ALIGNMENT = ESMATEG_DIR / "02_apply_alignment_da3_geocalib.py"

SCRIPT_VALIDATE_POSE_BEFORE_EGOGVAE = SCRIPTS_DIR / "05c_validate_pose_before_egogvae.py"
SCRIPT_RUN_EGOGVAE = SCRIPTS_DIR / "05_run_egogvae_from_head.py"
SCRIPT_RUN_EGOALLO_REFINE = SCRIPTS_DIR / "07_run_egoallo_refine_with_prior.py"
FINAL_MOTION_NAME = "merged_smoothed_vis.npz"

# Set CPU thread defaults in preprocessing child processes. The dedicated model
# runners retain their inference environment and scope CPU operations separately.
CPU_PREPROCESS_SCRIPTS = frozenset({
    SCRIPT_01_CAMERA_POSES, SCRIPT_02_EXTRACT_HANDS, SCRIPT_03_STABILIZE_HANDS,
    SCRIPT_EXTRACT_ROT, SCRIPT_04_BUILD_HAMER, SCRIPT_PREPARE_REAL_SENSOR,
    SCRIPT_05_APPLY_SCALE, SCRIPT_ESTIMATE_GRAVITY, SCRIPT_APPLY_ALIGNMENT,
    SCRIPT_VALIDATE_POSE_BEFORE_EGOGVAE,
})


@dataclass(frozen=True)
class UpstreamInputs:
    episode_dir: Path
    downstream_dir: Path
    frames_dir: Path
    cam_space_dir: Path
    objects_parquet: Path
    object_meshes: tuple[Path, ...]
    hawor_hands_vggt_world: Path
    hands_parquet: Path
    hawor_to_vggt_alignment: Path
    vggt_head_pose: Path
    scene_points_ply: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BodyMesh/EgoGVAE pipeline from episode/downstream inputs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--downstream-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
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


def finite_positive_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0.0 else None


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file '{name}' not found: {path}")


def require_dir(path: Path, name: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory '{name}' not found: {path}")


def _prepare_upstream_inputs(
    *,
    episode_dir: Path,
    downstream_dir: Path,
    prepared_dir: Path,
    frame_count: int | None = None,
) -> UpstreamInputs:
    manifest = _read_downstream_manifest(downstream_dir)

    frames_dir = prepared_dir / "frames"
    cam_space_dir = downstream_dir / "poses" / "cam_space"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    camera_parquet = (downstream_dir / "poses" / "camera_30fps.parquet").resolve()
    hands_parquet = (downstream_dir / "poses" / "hands_30fps.parquet").resolve()
    objects_parquet = (downstream_dir / "poses" / "objects_30fps.parquet").resolve()
    alignment_json = _downstream_dataset_path(downstream_dir, manifest, "alignment")
    scene_points_ply = _downstream_dataset_path(downstream_dir, manifest, "static_pointcloud")

    vggt_head_pose = prepared_dir / "vggt_head_pose.jsonl"
    hawor_hands_vggt_world = prepared_dir / "hawor_hands_vggt_world.jsonl"

    object_meshes: list[Path] = []
    for entry in manifest.get("object_meshes", []):
        mesh_path = str(entry.get("mesh") or "").strip()
        if not mesh_path:
            continue
        path = Path(mesh_path).expanduser()
        path = path.resolve() if path.is_absolute() else (downstream_dir / path).resolve()
        object_meshes.append(path)

    _extract_rgb_frames(
        episode_dir=episode_dir,
        camera_parquet=camera_parquet,
        frames_dir=frames_dir,
        frame_count=frame_count,
    )
    _write_camera_jsonl(camera_parquet, vggt_head_pose)
    _write_hands_jsonl(hands_parquet, hawor_hands_vggt_world)

    require_file(alignment_json, "downstream alignment.json")
    require_file(scene_points_ply, "downstream static_pointcloud")

    return UpstreamInputs(
        episode_dir=episode_dir,
        downstream_dir=downstream_dir,
        frames_dir=frames_dir,
        cam_space_dir=cam_space_dir,
        objects_parquet=objects_parquet,
        object_meshes=tuple(object_meshes),
        hawor_hands_vggt_world=hawor_hands_vggt_world,
        hands_parquet=hands_parquet,
        hawor_to_vggt_alignment=alignment_json,
        vggt_head_pose=vggt_head_pose,
        scene_points_ply=scene_points_ply,
    )


def _read_downstream_manifest(downstream_dir: Path) -> dict[str, Any]:
    manifest_path = downstream_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise TypeError(f"downstream manifest must be a JSON object: {manifest_path}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        raise KeyError(f"downstream manifest missing 'datasets': {manifest_path}")
    return manifest


def _object_mesh_output_paths(meshes: tuple[Path, ...], downstream_dir: Path, output_dir: Path) -> list[Path]:
    """Keep transformed assets under the output, including isolated reruns."""
    paths = []
    for index, mesh in enumerate(meshes):
        try:
            relative = mesh.resolve().relative_to(downstream_dir.resolve())
        except ValueError:
            # Manifests may reference external meshes; never write beside them.
            relative = Path('meshes') / 'external' / str(index) / mesh.name
        paths.append(output_dir / 'assets' / relative.parent / f'{mesh.stem}_egoallo.glb')
    return paths


def _downstream_dataset_path(
    downstream_dir: Path,
    manifest: dict[str, Any],
    key: str,
) -> Path:
    dataset = manifest.get("datasets", {}).get(key)
    if not isinstance(dataset, dict):
        raise KeyError(f"downstream manifest missing dataset '{key}'")
    raw_path = dataset.get("path")
    if not raw_path:
        raise KeyError(f"downstream dataset '{key}' missing path")
    path = Path(str(raw_path)).expanduser()
    path = path.resolve() if path.is_absolute() else (downstream_dir / path).resolve()
    return path


def _episode_stream_path(
    episode_dir: Path,
    *,
    modality: str,
    stream_id: str | None = None,
) -> Path:
    episode_path = episode_dir / "episode.json"
    with episode_path.open("r", encoding="utf-8") as f:
        episode = json.load(f)
    for stream in episode.get("streams", []):
        if not isinstance(stream, dict):
            continue
        if stream_id is not None and stream.get("stream_id") != stream_id:
            continue
        if modality is not None and stream.get("modality") != modality:
            continue
        uri = str(stream.get("uri") or "").strip()
        if not uri:
            continue
        path = Path(uri).expanduser()
        path = path.resolve() if path.is_absolute() else (episode_dir / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Episode stream file not found: {path}")
        return path
    raise KeyError(
        f"Episode stream not found: modality={modality!r}, stream_id={stream_id!r}"
    )


def _preferred_rgb_stream_id(episode_dir: Path) -> str:
    episode_path = episode_dir / "episode.json"
    with episode_path.open("r", encoding="utf-8") as f:
        episode = json.load(f)

    first_rgb: str | None = None
    for stream in episode.get("streams", []):
        if not isinstance(stream, dict):
            continue
        if stream.get("modality") != "rgb":
            continue
        uri = str(stream.get("uri") or "").strip()
        if not uri:
            continue
        stream_id = str(stream.get("stream_id") or "").strip()
        if not stream_id:
            continue
        if stream_id == "camera_left":
            return stream_id
        if first_rgb is None:
            first_rgb = stream_id

    if first_rgb is None:
        raise KeyError("No RGB stream with uri found in episode.json")
    return first_rgb


def _read_parquet(path: Path):
    try:
        import pandas as pd

        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - environment diagnostics
        raise RuntimeError(f"Reading parquet requires pandas/pyarrow: {path}") from exc


def _episode_video_frame_indices(
    episode_dir: Path, stream_id: str, timestamps_ns: list[int],
    *, frame_ids: list[int] | None = None,
    video_fps: float, video_frame_count: int,
) -> list[int]:
    """Resolve downstream frame IDs to local prepared-video frame indices.

    The prepared-episode standard defines ``origin.sample_map_uri`` as the
    authoritative relationship between output frames and the prepared MP4.
    ``frame_ids`` are downstream output frame IDs and therefore map to the
    sample map's ``output_frame_index``. Timestamps are retained for alignment
    and checked against the sample map, but are not used to infer local frame
    indices. The timestamp-based path remains for callers that use this helper
    directly without frame IDs and older episode metadata.
    """
    episode = read_json_file(episode_dir / "episode.json")
    stream = next(
        s for s in episode.get("streams", [])
        if isinstance(s, dict) and s.get("stream_id") == stream_id
    )
    timing = stream.get("time") or {}
    if timing.get("sampling") != "cfr":
        raise ValueError(f"RGB extraction requires sampling='cfr': {stream_id}")
    if timing.get("clock_id") != "episode":
        raise ValueError(f"RGB extraction requires clock_id='episode': {stream_id}")

    def integer_ns(value, name):
        try:
            valid = not isinstance(value, (bool, str)) and np.isfinite(value) and int(value) == value
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"Invalid integer nanoseconds for {name}: {value!r}")
        return int(value)

    start_ns = integer_ns(timing.get("start_ns"), "RGB time.start_ns")
    offset_ns = integer_ns(timing.get("video_pts_offset_ns", start_ns), "RGB video_pts_offset_ns")
    end_ns = timing.get("end_ns")
    if end_ns is not None:
        end_ns = integer_ns(end_ns, "RGB time.end_ns")
        if end_ns <= start_ns:
            raise ValueError("RGB time.end_ns must exceed time.start_ns")
    try:
        fps = float(timing.get("rate_hz"))
    except (TypeError, ValueError) as exc:
        raise ValueError("RGB time.rate_hz must specify a positive CFR frame rate") from exc
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(video_fps) or video_fps <= 0:
        raise ValueError(f"Invalid RGB frame rate: metadata={fps}, video={video_fps}")
    if not np.isclose(fps, video_fps, rtol=1e-4, atol=1e-6):
        raise ValueError(f"RGB frame rate mismatch: metadata={fps}, video={video_fps}")
    if video_frame_count <= 0:
        raise ValueError(f"RGB video frame count must be known and positive: {video_frame_count}")
    if len(timestamps_ns) == 0:
        raise ValueError("Camera trajectory has no timestamps for RGB extraction")
    if frame_ids is not None and len(frame_ids) != len(timestamps_ns):
        raise ValueError(
            f"Camera frame_id/timestamp length mismatch: frame_ids={len(frame_ids)}, "
            f"timestamps={len(timestamps_ns)}"
        )

    def stream_sample_map_path() -> Path:
        origin = stream.get("origin") or {}
        raw_uri = str(origin.get("sample_map_uri") or "").strip()
        if not raw_uri:
            raise ValueError(
                f"RGB stream is missing origin.sample_map_uri required for output-frame mapping: {stream_id}"
            )
        path = Path(raw_uri).expanduser()
        path = path.resolve() if path.is_absolute() else (episode_dir / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RGB sample map not found: {path}")
        return path

    if frame_ids is not None:
        sample_map = _read_parquet(stream_sample_map_path())
        required_columns = {"output_frame_index", "output_timestamp_ns"}
        missing_columns = sorted(required_columns.difference(sample_map.columns))
        if missing_columns:
            raise ValueError(
                f"RGB sample map missing required columns for {stream_id}: {missing_columns}"
            )
        sample_map = sample_map.sort_values("output_frame_index", kind="stable")

        output_indices = []
        output_timestamps = []
        for row_index, row in sample_map.iterrows():
            output_indices.append(
                integer_ns(row["output_frame_index"], f"RGB sample-map output_frame_index at row {row_index}")
            )
            output_timestamps.append(
                integer_ns(row["output_timestamp_ns"], f"RGB sample-map output_timestamp_ns at row {row_index}")
            )
        if len(output_indices) != video_frame_count:
            raise ValueError(
                f"RGB sample map/video frame count mismatch: stream={stream_id}, "
                f"sample_map_rows={len(output_indices)}, video_frame_count={video_frame_count}"
            )
        expected_indices = list(range(video_frame_count))
        if output_indices != expected_indices:
            raise ValueError(
                f"RGB sample map output_frame_index must be contiguous 0..{video_frame_count - 1}: "
                f"stream={stream_id}, first={output_indices[:3]}, last={output_indices[-3:]}"
            )
        timestamp_by_output_index = dict(zip(output_indices, output_timestamps))

        indices = []
        for row_index, (frame_id_value, timestamp_value) in enumerate(zip(frame_ids, timestamps_ns)):
            frame_id = integer_ns(frame_id_value, f"camera frame_id at row {row_index}")
            timestamp = integer_ns(timestamp_value, f"camera timestamp_ns at row {row_index}")
            if timestamp < start_ns - 1 or (end_ns is not None and timestamp >= end_ns):
                raise ValueError(
                    f"Camera timestamp outside RGB episode time: row={row_index}, "
                    f"timestamp_ns={timestamp}, range=[{start_ns}, {end_ns})"
                )
            expected_timestamp = timestamp_by_output_index.get(frame_id)
            if expected_timestamp is None:
                raise ValueError(
                    f"Camera frame_id is absent from RGB sample map: row={row_index}, "
                    f"frame_id={frame_id}, stream={stream_id}"
                )
            if abs(timestamp - expected_timestamp) > 1:
                raise ValueError(
                    f"Camera timestamp does not match RGB sample map: row={row_index}, "
                    f"frame_id={frame_id}, timestamp_ns={timestamp}, "
                    f"sample_map_timestamp_ns={expected_timestamp}"
                )
            indices.append(frame_id)
        print(
            f"[Frames] sample-map output frame -> local CFR video frame: stream={stream_id}, "
            f"fps={fps:g}, rows={len(indices)}; timestamps validated",
            flush=True,
        )
        return indices

    indices = []
    for row_index, value in enumerate(timestamps_ns):
        timestamp = integer_ns(value, f"camera timestamp_ns at row {row_index}")
        # Subtract integer times before floating-point conversion. Allow only
        # 1ns start-boundary rounding (e.g. 30Hz ticks represented as integers).
        delta_ns = timestamp - offset_ns
        if timestamp < start_ns - 1 or delta_ns < -1 or (end_ns is not None and timestamp >= end_ns):
            raise ValueError(f"Camera timestamp outside RGB episode time: row={row_index}, timestamp_ns={timestamp}, range=[{start_ns}, {end_ns})")
        frame_index = int(round(delta_ns * fps / 1e9))
        if frame_index < 0 or frame_index >= video_frame_count:
            raise ValueError(f"Video frame out of range: row={row_index}, timestamp_ns={timestamp}, video_frame_idx={frame_index}, video_frame_count={video_frame_count}")
        indices.append(frame_index)
    print(f"[Frames] episode timestamps -> local CFR video frames: stream={stream_id}, "
          f"fps={fps:g}, first_video_episode_ns={offset_ns}, rows={len(indices)}; legacy timestamp mapping", flush=True)
    return indices


def _extract_rgb_frames(
    *,
    episode_dir: Path,
    camera_parquet: Path,
    frames_dir: Path,
    frame_count: int | None = None,
) -> None:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError("OpenCV is required to extract episode RGB frames") from exc

    rgb_stream_id = _preferred_rgb_stream_id(episode_dir)
    video_path = _episode_stream_path(
        episode_dir,
        modality="rgb",
        stream_id=rgb_stream_id,
    )
    table = _read_parquet(camera_parquet)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open episode RGB video: {video_path}")
    try:
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        table = table.assign(
            video_frame_idx=_episode_video_frame_indices(
                episode_dir, rgb_stream_id, table["timestamp_ns"].tolist(),
                frame_ids=table["frame_id"].tolist(),
                video_fps=float(cap.get(cv2.CAP_PROP_FPS)), video_frame_count=video_frame_count,
            )
        )
        # Validate ALL trajectory times before subsampling or writing images.
        rows = list(table.iterrows())
        if frame_count is not None and 0 < int(frame_count) < len(rows):
            indices = sorted(set(np.linspace(0, len(rows) - 1, int(frame_count)).round().astype(int).tolist()))
            rows = [rows[i] for i in indices]
        for _, row in rows:
            frame_id = int(row["frame_id"])
            source_frame_idx = int(row["source_frame_idx"])
            video_frame_idx = int(row["video_frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Failed to read video frame: frame_id={frame_id}, "
                    f"source_frame_idx={source_frame_idx}, video_frame_idx={video_frame_idx}, "
                    f"video_frame_count={video_frame_count}: {video_path}"
                )
            output_path = frames_dir / f"{frame_id:06d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Failed to write extracted frame: {output_path}")
    finally:
        cap.release()

    print(
        f"[Frames] extracted {len(list(frames_dir.glob('*.jpg')))} RGB frames "
        f"from stream_id={rgb_stream_id}: {video_path}",
        flush=True,
    )


def _write_camera_jsonl(camera_parquet: Path, output_path: Path) -> None:
    require_file(camera_parquet, "downstream camera_30fps.parquet")
    table = _read_parquet(camera_parquet)

    with output_path.open("w", encoding="utf-8") as f:
        for _, row in table.iterrows():
            t_world_from_camera = np.asarray(row["T_world_from_camera"], dtype=np.float64).reshape(-1)
            if t_world_from_camera.size != 16:
                raise ValueError(
                    f"T_world_from_camera must contain 16 values, got {t_world_from_camera.size}"
                )
            matrix = t_world_from_camera.reshape(4, 4)
            intrinsics = np.asarray(row["intrinsics"], dtype=np.float64).reshape(-1)
            if intrinsics.size != 9:
                raise ValueError(f"intrinsics must contain 9 values, got {intrinsics.size}")

            record = {
                "extracted_idx": int(row["frame_id"]),
                "original_frame_idx": int(row["source_frame_idx"]),
                "timestamp": float(row["timestamp_ns"]) / 1e9,
                "coordinate": "camera_to_world",
                "R": matrix[:3, :3].tolist(),
                "t": matrix[:3, 3].tolist(),
                "intrinsics": intrinsics.reshape(3, 3).tolist(),
                "depth_shape": [int(row["depth_height"]), int(row["depth_width"])],
                "length_unit": str(row.get("length_unit") or "meter"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _vector_to_nested(value: Any, width: int, field_name: str) -> list[list[float]]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0 or arr.size % width != 0:
        raise ValueError(f"{field_name} must contain a multiple of {width} values")
    return arr.reshape(-1, width).tolist()


def _optional_vector_to_nested(value: Any, width: int) -> list[list[float]] | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return _vector_to_nested(arr, width, "hawor_camera_keypoints")


def _write_hands_jsonl(hands_parquet: Path, output_path: Path) -> None:
    require_file(hands_parquet, "downstream hands_30fps.parquet")
    table = _read_parquet(hands_parquet)

    by_frame: dict[int, dict[str, Any]] = {}
    for _, row in table.iterrows():
        frame_id = int(row["frame_id"])
        source_frame_idx = int(row["source_frame_idx"])
        timestamp = float(row["timestamp_ns"]) / 1e9
        hand = {
            "side": str(row["side"]),
            "confidence": float(row["confidence"]),
            "track_id": str(row.get("track_id") or ""),
            "keypoints_3d": _vector_to_nested(row["keypoints_world"], 3, "keypoints_world"),
            "mesh_vertices_3d": _vector_to_nested(row["vertices_world"], 3, "vertices_world"),
            "camera_keypoints_3d": _vector_to_nested(
                row["keypoints_camera"], 3, "keypoints_camera"
            ),
            "hawor_camera_keypoints_3d": _optional_vector_to_nested(
                row.get("hawor_camera_keypoints"), 3
            ),
        }

        by_frame.setdefault(
            frame_id,
            {
                "frame_id": frame_id,
                "src_frame_id": source_frame_idx,
                "extracted_idx": frame_id,
                "original_frame_idx": source_frame_idx,
                "timestamp": timestamp,
                "timestamp_ns": int(row["timestamp_ns"]),
                "hands": [],
            },
        )["hands"].append(hand)

    with output_path.open("w", encoding="utf-8") as f:
        for frame_id in sorted(by_frame):
            f.write(json.dumps(by_frame[frame_id], ensure_ascii=False) + "\n")


def resolve_real_sensor_input(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not (path / "episode.json").is_file():
        print(f"[Real sensor] no episode.json found; use default pipeline: {path}", flush=True)
        return None
    imu_path = episode_imu_path(path)
    has_depth = episode_has_depth_stream(path)
    has_calibration = episode_calibration_path(path) is not None
    if imu_path is None and not has_depth and not has_calibration:
        print(f"[Real sensor] no IMU/depth/calibration found; use default pipeline: {path}", flush=True)
        return None
    print(
        "[Real sensor] episode.json inputs found: "
        f"imu={'yes' if imu_path else 'no'}, depth={'yes' if has_depth else 'no'}, "
        f"calibration={'yes' if has_calibration else 'no'}",
        flush=True,
    )
    return path


def episode_imu_path(episode_dir: Path) -> Path | None:
    uri = episode_stream_uri(episode_dir, modality="imu", stream_id="ego_imu")
    if not uri:
        return None
    path = Path(uri).expanduser()
    path = path.resolve() if path.is_absolute() else (episode_dir / path).resolve()
    return path if path.is_file() else None


def episode_stream_uri(
    episode_dir: Path,
    *,
    modality: str | None = None,
    stream_id: str | None = None,
) -> str | None:
    episode_json = episode_dir / "episode.json"
    if not episode_json.is_file():
        return None
    with episode_json.open("r", encoding="utf-8") as f:
        episode = json.load(f)
    for stream in episode.get("streams", []):
        if not isinstance(stream, dict):
            continue
        if stream_id is not None and stream.get("stream_id") != stream_id:
            continue
        if modality is not None and stream.get("modality") != modality:
            continue
        return str(stream.get("uri") or "").strip() or None
    return None


def episode_stream_id(episode_dir: Path, modality: str) -> str | None:
    episode_json = episode_dir / "episode.json"
    if not episode_json.is_file():
        return None
    with episode_json.open("r", encoding="utf-8") as f:
        episode = json.load(f)
    for stream in episode.get("streams", []):
        if isinstance(stream, dict) and stream.get("modality") == modality:
            value = str(stream.get("stream_id") or "").strip()
            return value or None
    return None


def episode_calibration_path(episode_dir: Path) -> Path | None:
    episode_json = episode_dir / "episode.json"
    if not episode_json.is_file():
        return None
    with episode_json.open("r", encoding="utf-8") as f:
        episode = json.load(f)
    uri = str(episode.get("calibration_uri") or "").strip()
    if not uri:
        return None
    path = Path(uri).expanduser()
    path = path.resolve() if path.is_absolute() else (episode_dir / path).resolve()
    return path if path.is_file() else None


def episode_has_depth_stream(episode_dir: Path) -> bool:
    return episode_stream_uri(episode_dir, modality="depth") is not None


def read_imu_alignment_decision(align_meta: Path) -> dict[str, Any]:
    """Summarize the real-IMU alignment outcome for pipeline_run_log.json."""
    if not align_meta.is_file():
        return {"status": "missing", "align_meta": str(align_meta)}
    try:
        import numpy as np

        with np.load(align_meta, allow_pickle=True) as data:
            def item(key: str, default: Any = None) -> Any:
                return data[key].item() if key in data else default

            height_validation = json.loads(str(item("height_validation_json", "{}")))
            imu_stats = json.loads(str(item("imu_stats_json", "{}")))
            return {
                "status": "ok",
                "align_meta": str(align_meta),
                "imu_mode": bool(item("imu_mode", False)),
                "used_imu_gravity": bool(item("used_imu_gravity", False)),
                "accepted": imu_stats.get("accepted"),
                "reference_only": bool(imu_stats.get("reference_only", False)),
                "ground_plane_trusted": item("ground_plane_trusted", None),
                "metric_scale_trusted": item("metric_scale_trusted", None),
                "fallback_reference": imu_stats.get("floor_fit") if imu_stats.get("reference_only") else None,
                "reject_reason": imu_stats.get("reject_reason"),
                "num_candidates_tried": imu_stats.get("num_candidates_tried"),
                "decision_reason": str(item("decision_reason", "")),
                "head_height_valid": height_validation.get("final_height_valid"),
                "head_height_reason": height_validation.get("final_height_reason"),
            }
    except Exception as exc:  # diagnostics only, never fail the pipeline here
        return {"status": "unreadable", "align_meta": str(align_meta), "error": repr(exc)}


def mark_alignment_reference_only(path: Path, reason: str) -> None:
    """Keep the exact display transform, but revoke physical-floor trust."""
    with np.load(path, allow_pickle=True) as data:
        values = {key: data[key].copy() for key in data.files}
    stats = json.loads(str(values.get("imu_stats_json", "{}")))
    stats.setdefault("geometric_plane_accepted", stats.get("accepted"))
    stats.update(accepted=False, reference_only=True, ground_plane_trusted=False,
                 metric_scale_trusted=False, reject_reason=reason)
    values["imu_stats_json"] = np.asarray(json.dumps(stats))
    values.setdefault("geometric_decision_reason", values.get("decision_reason", np.asarray("")))
    values.update(reference_only=np.asarray(True), ground_plane_trusted=np.asarray(False),
                  metric_scale_trusted=np.asarray(False), decision_reason=np.asarray(reason))
    np.savez_compressed(path, **values)


def configure_reference_only_export(cmd: list[str], reason: str, *, output_dir: Path,
                                    head_name: str, hands_name: str, timestamps: Path,
                                    hands: Path, rotation: Path, alignment: Path) -> None:
    """Use normal shared-coordinate exporters even if frontend export was disabled."""
    cmd.extend(["--alignment-only", "--alignment-only-reason", reason])
    if "--export-frontend-jsonl" not in cmd:
        cmd.extend(["--export-frontend-jsonl", "--frontend-output-dir", str(output_dir),
                    "--frontend-head-jsonl-name", head_name, "--frontend-hands-jsonl-name", hands_name,
                    "--timestamps-txt", str(timestamps), "--hands-vggt-jsonl-in", str(hands),
                    "--R-align", str(rotation), "--align-meta", str(alignment)])


def add_flag(cmd: list[str], flag: str, value: Any) -> None:
    if isinstance(value, bool) and value:
        cmd.append(flag)


def prepend_pythonpath(env: dict[str, str], paths: list[Path]) -> None:
    valid_paths = [str(p) for p in paths if p is not None and p.exists()]
    if not valid_paths:
        return
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(valid_paths + ([old_pythonpath] if old_pythonpath else []))


def build_subprocess_env(cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["SCRIPTS_DIR"] = str(SCRIPTS_DIR)
    env["PYTHON_EXECUTABLE"] = sys.executable
    for key, name in (("preprocess_threads", "EGO_UNI_PREPROCESS_THREADS"),
                      ("torch_cpu_threads", "EGO_UNI_TORCH_CPU_THREADS")):
        value = env.get(name, get_cfg(cfg, f"runtime.{key}", 1))
        env[name] = str(thread_count(value, f"runtime.{key}"))

    # Only project scripts are added here. Model source roots are passed through
    # env/config so each subprocess owns its import path.
    prepend_pythonpath(env, [PROJECT_ROOT, SCRIPTS_DIR])

    egogvae_root = resolve_project_path(get_cfg(cfg, "model_sources.egogvae_root"))
    geocalib_root = resolve_project_path(get_cfg(cfg, "model_sources.geocalib_root"))
    foundationgeo_root = resolve_project_path(get_cfg(cfg, "model_sources.foundationgeo_root"))
    if egogvae_root is not None:
        env["EGOGVAE_ROOT"] = str(egogvae_root)
    if geocalib_root is not None:
        env["GEOCALIB_ROOT"] = str(geocalib_root)
    if foundationgeo_root is not None:
        # 只作为源码路径传给下游脚本，实际源码/conda优先级由下游脚本自行控制。
        env["FOUNDATIONGEO_ROOT"] = str(foundationgeo_root)

    env_path_map = {
        "EGOGVAE_CHECKPOINT_PATH": "model_weights.egogvae_checkpoint_path",
        "GEOCALIB_PINHOLE_TAR": "model_weights.geocalib_pinhole_tar",
        "SMPLH_NPZ_PATH": "model_weights.smplh_npz_path",
        "VPOSER_DIR": "model_weights.vposer_dir",
        "FOUNDATIONGEO_MODEL_PT": "model_weights.foundationgeo_model_pt",
    }
    for env_name, cfg_key in env_path_map.items():
        path = resolve_project_path(get_cfg(cfg, cfg_key))
        if path is not None:
            env[env_name] = str(path)

    return env


def _cfg_bool(value: Any, default: bool = False) -> bool:
    """Parse bool-like YAML/env values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def cleanup_flag(cfg: dict[str, Any], name: str, *, legacy_key: str | None = None, default: bool = False) -> bool:
    """Read output_layout.cleanup.<name>, with optional legacy fallback."""
    value = get_cfg(cfg, f"output_layout.cleanup.{name}", None)
    if value is None and legacy_key is not None:
        value = get_cfg(cfg, legacy_key, None)
    return _cfg_bool(value, default=default)


def cfg_list_to_csv(value: Any, default: list[str]) -> str:
    if value is None:
        value = default
    if isinstance(value, (list, tuple)):
        return ",".join(str(x).strip() for x in value if str(x).strip())
    return str(value)


def write_failed_scale_json(path: Path, *, source: str, reason: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "status": "failed",
        "source": source,
        "reason": reason,
        "scale": None,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_empty_egogvae_npz(path: Path, *, num_samples: int, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ns = max(int(num_samples), 1)
    np.savez(
        path,
        Ts_world_cpf=np.empty((0, 7), dtype=np.float32),
        Ts_world_root=np.empty((ns, 0, 7), dtype=np.float32),
        body_quats=np.empty((ns, 0, 21, 4), dtype=np.float32),
        left_hand_quats=np.empty((ns, 0, 15, 4), dtype=np.float32),
        right_hand_quats=np.empty((ns, 0, 15, 4), dtype=np.float32),
        betas=np.empty((ns, 0, 16), dtype=np.float32),
        contacts=np.empty((ns, 0, 21), dtype=np.float32),
        frame_nums=np.empty((0,), dtype=np.int64),
        timestamps_ns=np.empty((0,), dtype=np.int64),
        merge_segment_boundaries=np.empty((0,), dtype=np.int64),
        merge_segment_lengths=np.empty((0,), dtype=np.int64),
        merge_is_smoothed=np.asarray(False),
        merge_boundary_blend_frames=np.asarray(0, dtype=np.int64),
        merge_translation_smooth_window=np.asarray(0, dtype=np.int64),
        merge_quat_lowpass_alpha=np.asarray(0.0, dtype=np.float32),
        egogvae_skipped=np.asarray(True),
        egogvae_skip_reason=np.asarray(str(reason)),
    )


def _assets_pointclouds_dir(output_dir: Path, cfg: dict[str, Any]) -> Path:
    return output_dir / str(get_cfg(cfg, "output_layout.assets_pointclouds_dir_name", "assets/pointclouds"))


def _assets_poses_dir(output_dir: Path, cfg: dict[str, Any]) -> Path:
    return output_dir / str(get_cfg(cfg, "output_layout.assets_poses_dir_name", "assets/poses"))


def _json_friendly(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_friendly(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_friendly(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_friendly(value.tolist())
    if hasattr(value, "item"):
        return _json_friendly(value.item())
    return value


def _write_pointcloud_manifest_parquet(path: Path, pointcloud_name: str, *, reference_only: bool = False) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "path": pointcloud_name,
                "coordinate_system": "egoallo_world",
                "up_axis": "z",
                "floor_z": 0.0,
                "length_unit": "input_units" if reference_only else "meter",
                "reference_only": reference_only,
                "ground_plane_trusted": False if reference_only else None,
                "metric_scale_trusted": False if reference_only else None,
            }
        ]
    ).to_parquet(path, index=False, engine="pyarrow")


def _write_jsonl_parquet(src: Path, dst: Path, *, kind: str) -> None:
    import pandas as pd

    require_file(src, f"{kind} JSONL")
    rows: list[dict[str, Any]] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            timestamp_ns = int(round(float(rec.get("timestamp", 0.0)) * 1e9))
            frame_id = int(rec.get("frame_id", 0))
            common = {
                "timestamp_ns": timestamp_ns,
                "frame": frame_id,
                "timestamp": float(rec.get("timestamp", 0.0)),
                "src_frame_id": int(rec.get("src_frame_id", frame_id)),
                "coordinate_system": str(rec.get("coordinate_system", {}).get("name", "egoallo_world")),
                "length_unit": "input_units" if rec.get("coordinate_system", {}).get("units") == "input_units" else "meter",
                "reference_only": rec.get("coordinate_system", {}).get("reference_only", False),
                "ground_plane_trusted": rec.get("coordinate_system", {}).get("ground_plane_trusted"),
                "metric_scale_trusted": rec.get("coordinate_system", {}).get("metric_scale_trusted"),
            }
            if kind == "head":
                head = rec.get("head", {})
                rows.append(
                    {
                        **common,
                        "position": _json_friendly(head.get("position")),
                        "rotation": _json_friendly(head.get("rotation")),
                        "matrix": _json_friendly(head.get("matrix")),
                    }
                )
            elif kind == "hands":
                for hand in rec.get("hands", []):
                    row = {
                        **common,
                        "side": str(hand.get("side", "")),
                        "confidence": _json_friendly(hand.get("confidence")),
                        "wrist": _json_friendly(hand.get("wrist")),
                        "keypoints_3d": _json_friendly(hand.get("keypoints_3d")),
                    }
                    if "mesh_vertices_3d" in hand:
                        row["mesh_vertices_3d"] = _json_friendly(hand.get("mesh_vertices_3d"))
                    rows.append(row)
            else:
                raise ValueError(f"Unsupported JSONL parquet kind: {kind}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        columns = [
            "timestamp_ns",
            "frame",
            "timestamp",
            "src_frame_id",
            "coordinate_system",
            "length_unit",
        ]
        if kind == "head":
            columns.extend(["position", "rotation", "matrix"])
        elif kind == "hands":
            columns.extend(["side", "confidence", "wrist", "keypoints_3d", "mesh_vertices_3d"])
        frame = pd.DataFrame(columns=columns)
    frame.to_parquet(dst, index=False, engine="pyarrow")


def _write_motion_parquet(src: Path, dst: Path) -> None:
    import numpy as np
    import pandas as pd

    require_file(src, "final motion NPZ")
    data = np.load(src, allow_pickle=False)
    if "frame_nums" not in data or "timestamps_ns" not in data:
        raise KeyError(f"Motion NPZ must contain frame_nums and timestamps_ns: {src}")

    frame_nums = np.asarray(data["frame_nums"]).reshape(-1)
    timestamps_ns = np.asarray(data["timestamps_ns"]).reshape(-1)
    if frame_nums.size != timestamps_ns.size:
        raise ValueError(f"frame_nums/timestamps_ns length mismatch in {src}")

    rows: list[dict[str, Any]] = []
    for i in range(frame_nums.size):
        row: dict[str, Any] = {
            "timestamp_ns": int(timestamps_ns[i]),
            "frame": int(frame_nums[i]),
        }
        if "timestamps" in data and np.asarray(data["timestamps"]).reshape(-1).size == frame_nums.size:
            row["timestamp"] = float(np.asarray(data["timestamps"]).reshape(-1)[i])

        for key in sorted(data.files):
            if key in {"frame_nums", "timestamps_ns", "timestamps"}:
                continue
            arr = np.asarray(data[key])
            if arr.ndim == 0:
                continue
            if arr.shape[0] == frame_nums.size:
                value = arr[i]
            elif arr.ndim >= 2 and arr.shape[1] == frame_nums.size:
                value = arr[0, i]
            else:
                continue
            row[key] = _json_friendly(value)
        rows.append(row)

    scalar_columns: dict[str, Any] = {}
    for key in sorted(data.files):
        arr = np.asarray(data[key])
        if arr.ndim == 0:
            scalar_columns[key] = _json_friendly(arr)

    frame = pd.DataFrame(rows)
    for key, value in scalar_columns.items():
        frame[key] = value
    dst.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dst, index=False, engine="pyarrow")


def _export_assets(
    *,
    output_dir: Path,
    cfg: dict[str, Any],
    final_scene_ply: Path,
    frontend_head_jsonl: Path,
    frontend_hands_jsonl: Path,
    objects_egoallo_parquet: Path,
    final_motion_npz: Path,
    reference_only: bool = False,
) -> dict[str, Any]:
    pointclouds_dir = _assets_pointclouds_dir(output_dir, cfg)
    poses_dir = _assets_poses_dir(output_dir, cfg)
    pointclouds_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)

    final_scene_name = final_scene_ply.name
    final_scene_asset = pointclouds_dir / final_scene_name
    if final_scene_ply.resolve() != final_scene_asset.resolve():
        shutil.copy2(final_scene_ply, final_scene_asset)
    pointcloud_manifest = pointclouds_dir / "pointclouds.parquet"
    _write_pointcloud_manifest_parquet(pointcloud_manifest, final_scene_name, reference_only=reference_only)

    head_parquet = poses_dir / (frontend_head_jsonl.stem + ".parquet")
    hands_parquet = poses_dir / (frontend_hands_jsonl.stem + ".parquet")
    objects_parquet_out = poses_dir / objects_egoallo_parquet.name
    motion_npz = poses_dir / final_motion_npz.name
    _write_jsonl_parquet(frontend_head_jsonl, head_parquet, kind="head")
    _write_jsonl_parquet(frontend_hands_jsonl, hands_parquet, kind="hands")
    shutil.copy2(objects_egoallo_parquet, objects_parquet_out)
    shutil.copy2(final_motion_npz, motion_npz)

    return {
        "pointclouds_dir": str(pointclouds_dir),
        "final_scene": str(final_scene_asset),
        "pointcloud_manifest": str(pointcloud_manifest),
        "poses_dir": str(poses_dir),
        "head_parquet": str(head_parquet),
        "hands_parquet": str(hands_parquet),
        "objects_parquet": str(objects_parquet_out),
        "motion_npz": str(motion_npz),
    }


def remove_dir_if_requested(name: str, path: Path, enabled: bool) -> dict[str, Any]:
    result = {"name": name, "path": str(path), "requested": bool(enabled), "deleted": False}
    if not enabled:
        result["reason"] = "disabled"
        return result
    if not path.exists():
        result["reason"] = "not_exists"
        return result
    if not path.is_dir():
        result["reason"] = "not_a_directory"
        return result
    shutil.rmtree(path)
    result["deleted"] = True
    result["reason"] = "deleted_after_full_pipeline_success"
    return result


def cleanup_existing_scripts_pycache(enabled: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested": bool(enabled),
        "root": str(SCRIPTS_DIR),
        "deleted": [],
    }
    if not enabled:
        result["reason"] = "disabled"
        return result

    for pycache_dir in sorted(SCRIPTS_DIR.rglob("__pycache__")):
        if not pycache_dir.is_dir():
            continue
        shutil.rmtree(pycache_dir)
        result["deleted"].append(str(pycache_dir))

    result["reason"] = "deleted_existing_scripts_pycache" if result["deleted"] else "not_found"
    return result


def configure_python_pycache(env: dict[str, str], cfg: dict[str, Any], tmp_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    cleanup_result = cleanup_existing_scripts_pycache(
        _cfg_bool(get_cfg(cfg, "python_cache.cleanup_existing_scripts_pycache", True), default=True)
    )
    enabled = _cfg_bool(get_cfg(cfg, "python_cache.redirect_pycache_to_tmp", True), default=True)
    if not enabled:
        env.pop("PYTHONPYCACHEPREFIX", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return None, cleanup_result

    cache_subdir = str(get_cfg(cfg, "python_cache.pycache_subdir", "pycache"))
    cache_dir = (tmp_dir / cache_subdir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env["PYTHONPYCACHEPREFIX"] = str(cache_dir)
    print(f"[Python] pycache redirected to: {cache_dir}", flush=True)
    return cache_dir, cleanup_result


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _visible_cuda_devices() -> dict[str, Any]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is None or cuda_visible_devices.strip() == "":
        visible_tokens: list[str] | None = None
    elif cuda_visible_devices.strip() in {"-1", "none", "None"}:
        visible_tokens = []
    else:
        visible_tokens = [token.strip() for token in cuda_visible_devices.split(",") if token.strip()]

    detected_gpus: list[dict[str, str]] = []
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader,nounits"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        proc = None
    if proc is not None and proc.returncode == 0:
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 1)]
            if len(parts) == 2 and parts[0] != "":
                detected_gpus.append({"index": parts[0], "name": parts[1]})

    if visible_tokens is None:
        visible_gpus = detected_gpus
    else:
        by_index = {gpu["index"]: gpu for gpu in detected_gpus}
        visible_gpus = [by_index[token] for token in visible_tokens if token in by_index]

    return {
        "visible_gpu_count": len(visible_gpus) if visible_tokens is None or visible_gpus else len(visible_tokens or []),
        "visible_gpu_models": [gpu["name"] for gpu in visible_gpus],
        "detected_gpu_models": [gpu["name"] for gpu in detected_gpus],
    }


def _cfg_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_egoallo_refine_bucket_t(cfg: dict[str, Any]) -> int:
    value = get_cfg(cfg, "egoallo_refine.bucket_t", "auto")
    target_segment_length = _cfg_int(get_cfg(cfg, "egoallo_refine.target_segment_length", 1024), 1024)
    context_frames = _cfg_int(get_cfg(cfg, "egoallo_refine.context_frames", 36), 36)
    if isinstance(value, str) and value.strip().lower() == "auto":
        return int(target_segment_length) + 2 * int(context_frames)
    return _cfg_int(value, 0)


def _coerce_egoallo_refine_detection_bucket(cfg: dict[str, Any]) -> int:
    value = _cfg_int(get_cfg(cfg, "egoallo_refine.detection_bucket", 0), 0)
    bucket_t = _coerce_egoallo_refine_bucket_t(cfg)
    return bucket_t if value <= 0 and bucket_t > 0 else value


def _jax_compilation_cache_signature(cfg: dict[str, Any], cache_dir: Path | None = None) -> dict[str, Any]:
    bucket_t = _coerce_egoallo_refine_bucket_t(cfg)
    detection_bucket = _coerce_egoallo_refine_detection_bucket(cfg)
    return {
        "signature_schema": 17,
        "compile_tags": {
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir) if cache_dir is not None else None,
            "jaxlib_version": _package_version("jaxlib"),
            "jax_version": _package_version("jax"),
            "cuda": _visible_cuda_devices(),
            # Torch input references/postprocessing do not change this JAX guidance graph.
            "pipeline": "native_egoallo_contact_only_legs",
            "guidance_mode": get_cfg(cfg, "egoallo_refine.guidance_mode", "hamer_reproj2"),
            "target_segment_length": get_cfg(cfg, "egoallo_refine.target_segment_length", 1024),
            "context_frames": get_cfg(cfg, "egoallo_refine.context_frames", 36),
            "guidance_inner": True,
            "guidance_post": True,
            "inner_max_iters": get_cfg(cfg, "egoallo_refine.inner_max_iters", 2),
            "post_max_iters": get_cfg(cfg, "egoallo_refine.post_max_iters", 16),
            "egoallo_initialization": "full_noise_head_conditioned_egoallo",
            "wrist_guidance": True,
            "hand_pose_guidance": True,
            "guidance_shape": "mean_native_betas_excluding_padding",
            "window_size": get_cfg(cfg, "egoallo_refine.window_size", 128),
            "ddim_eta": 0.8,
            "bucket_t": bucket_t,
            "detection_bucket": detection_bucket,
            "hand_detection_pytree": {
                "root_keys": ["T_cpf_cam", "detections_left_concat", "detections_right_concat"],
                "side_keys": [
                    "keypoints_3d",
                    "mano_hand_global_orient",
                    "single_hand_quats",
                    "constraint_weight",
                    "hand_pose_weight",
                    "hand_reproj_weight",
                    "wrist_pose_weight",
                    "indices",
                ],
                "float_dtype": "float32",
                "index_dtype": "int64",
                "per_side_padded_detection_count": detection_bucket,
                "left_right_detection_counts_fixed_by_padding": True,
            },
        },
        "runtime_tags": {
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", ""),
            "hamer_abspos_weight": get_cfg(cfg, "egoallo_refine.hamer_abspos_weight", 20.0),
        },
        "egoallo_refine": get_cfg(cfg, "egoallo_refine", {}),
    }


def _jax_compilation_cache_compile_key(signature: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(signature, dict):
        return None
    compile_tags = signature.get("compile_tags")
    if not isinstance(compile_tags, dict):
        compile_tags = {}
    compile_tag_keys = (
        "JAX_COMPILATION_CACHE_DIR",
        "jaxlib_version",
        "jax_version",
        "pipeline",
        "guidance_mode",
        "target_segment_length",
        "context_frames",
        "guidance_inner",
        "guidance_post",
        "inner_max_iters",
        "post_max_iters",
        "egoallo_initialization",
        "wrist_guidance",
        "hand_pose_guidance",
        "ddim_eta",
        "guidance_shape",
        "window_size",
        "bucket_t",
        "detection_bucket",
        "hand_detection_pytree",
    )
    return {
        "signature_schema": signature.get("signature_schema"),
        "compile_tags": {key: compile_tags.get(key) for key in compile_tag_keys},
    }


def _is_safe_cache_dir_to_clear(cache_dir: Path, project_root: Path) -> tuple[bool, str]:
    cache_dir = cache_dir.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    forbidden = {
        Path("/").resolve(),
        Path.home().resolve(),
        project_root,
        project_root.parent.resolve(),
    }
    if cache_dir in forbidden:
        return False, "cache_dir_is_forbidden_root"
    if len(cache_dir.parts) < 3:
        return False, "cache_dir_too_shallow"
    return True, "ok"


def _maybe_reset_jax_cache_for_signature(cache_dir: Path, cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    reset_enabled = _cfg_bool(
        get_cfg(cfg, "jax_guidance.compilation_cache.reset_on_signature_mismatch", True),
        default=True,
    )
    signature_file_name = str(
        get_cfg(cfg, "jax_guidance.compilation_cache.signature_file", ".egoallo_jax_cache_signature.json")
    )
    signature = _jax_compilation_cache_signature(cfg, cache_dir)
    result = {
        "enabled": bool(reset_enabled),
        "cache_dir": str(cache_dir),
        "signature_file": str(cache_dir / signature_file_name),
        "reset": False,
        "reuse": False,
        "regenerate": False,
        "reason": "disabled",
    }
    if not reset_enabled:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with (cache_dir / signature_file_name).open("w", encoding="utf-8") as f:
            json.dump(signature, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        return result

    signature_path = cache_dir / signature_file_name
    old_signature = None
    cache_exists = cache_dir.exists()
    if signature_path.exists():
        try:
            with signature_path.open("r", encoding="utf-8") as f:
                old_signature = json.load(f)
        except Exception:
            old_signature = None

    if cache_exists and _jax_compilation_cache_compile_key(old_signature) == _jax_compilation_cache_compile_key(signature):
        result["reuse"] = True
        result["reason"] = "signature_match"
        return result

    if cache_exists:
        safe, reason = _is_safe_cache_dir_to_clear(cache_dir, project_root)
        if not safe:
            raise RuntimeError(f"Refuse to clear unsafe JAX cache directory: {cache_dir} ({reason})")
        shutil.rmtree(cache_dir)
        result["reset"] = True
        result["reason"] = "signature_mismatch" if old_signature is not None else "missing_or_invalid_signature"
    else:
        result["reason"] = "cache_missing"

    result["regenerate"] = True
    cache_dir.mkdir(parents=True, exist_ok=True)
    with signature_path.open("w", encoding="utf-8") as f:
        json.dump(signature, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return result


def configure_jax_compilation_cache(
    env: dict[str, str],
    cfg: dict[str, Any],
    tmp_dir: Path,
    project_root: Path,
) -> Path | None:
    enabled = _cfg_bool(get_cfg(cfg, "jax_guidance.compilation_cache.enabled", True), default=True)
    if not enabled:
        return None

    configured_cache_dir = get_cfg(cfg, "jax_guidance.compilation_cache.output_path", None)
    if configured_cache_dir is None:
        configured_cache_dir = get_cfg(cfg, "jax_guidance.compilation_cache.cache_dir", None)
    if configured_cache_dir:
        cache_dir = Path(str(configured_cache_dir)).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = project_root / cache_dir
        cache_dir = cache_dir.resolve()
    else:
        cache_subdir = str(get_cfg(cfg, "jax_guidance.compilation_cache.cache_subdir", "jax_worker_cache"))
        cache_dir = (tmp_dir / cache_subdir).expanduser().resolve()

    reset_result = _maybe_reset_jax_cache_for_signature(cache_dir, cfg, project_root)
    if reset_result["reset"]:
        print(f"[JAX] Persistent compilation cache reset: {cache_dir} reason={reset_result['reason']}", flush=True)
    elif reset_result["reuse"]:
        print(f"[JAX] Persistent compilation cache reused: {cache_dir}", flush=True)
    elif reset_result["regenerate"]:
        print(f"[JAX] Persistent compilation cache initialized: {cache_dir} reason={reset_result['reason']}", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env["JAX_ENABLE_COMPILATION_CACHE"] = "true"
    env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = str(
        get_cfg(cfg, "jax_guidance.compilation_cache.min_compile_time_secs", 0)
    )
    env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = str(
        get_cfg(cfg, "jax_guidance.compilation_cache.min_entry_size_bytes", -1)
    )
    env["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] = str(
        get_cfg(cfg, "jax_guidance.compilation_cache.enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    )

    if _cfg_bool(get_cfg(cfg, "jax_guidance.compilation_cache.explain_cache_misses", False)):
        env["JAX_EXPLAIN_CACHE_MISSES"] = "true"
    if _cfg_bool(get_cfg(cfg, "jax_guidance.compilation_cache.debug_log_modules", False)):
        env["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"
        env["JAX_LOGGING_LEVEL"] = "DEBUG"

    env["EGOALLO_JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    env["EGOALLO_JAX_CACHE_LOG"] = str(tmp_dir / "jax_cache_hits.log")
    env["EGOALLO_JAX_CACHE_RESET_REASON"] = str(reset_result["reason"])
    env["EGOALLO_JAX_CACHE_SIGNATURE_FILE"] = str(reset_result["signature_file"])
    env["EGOALLO_JAX_CACHE_REUSED"] = "1" if reset_result["reuse"] else "0"
    env["EGOALLO_JAX_CACHE_REGENERATED"] = "1" if reset_result["regenerate"] else "0"

    print(f"[JAX] Persistent compilation cache enabled: {cache_dir}", flush=True)
    return cache_dir


PIPELINE_RECORDER: "PipelineRunRecorder | None" = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class PipelineRunRecorder:
    """Collect every subprocess command and its full stdout/stderr into one JSON file."""

    def __init__(self, path: Path, *, metadata: dict[str, Any] | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started_at = _now_iso()
        self.metadata = metadata or {}
        self.steps: list[dict[str, Any]] = []
        self.write(status="running")

    def add_step(self, step: dict[str, Any]) -> None:
        self.steps.append(step)
        self.write(status="running")

    def write(
        self,
        *,
        status: str,
        error: str | None = None,
        final_summary: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "status": status,
            "started_at": self.started_at,
            "updated_at": _now_iso(),
            "metadata": self.metadata,
            "num_steps": len(self.steps),
            "steps": self.steps,
        }
        if error is not None:
            payload["error"] = error
        if final_summary is not None:
            payload["final_summary"] = final_summary
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)



def run(cmd: list[str], *, cwd: Path = PROJECT_ROOT, env: dict[str, str] | None = None) -> None:
    global PIPELINE_RECORDER

    if len(cmd) > 1 and Path(cmd[1]).resolve() in CPU_PREPROCESS_SCRIPTS:
        env = preprocessing_env(os.environ if env is None else env)

    command_str = " ".join(str(x) for x in cmd)
    print("\n[CMD]", command_str, flush=True)

    started = time.time()
    started_at = _now_iso()
    output_chunks: list[str] = []

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_chunks.append(line)

    return_code = process.wait()
    elapsed_sec = time.time() - started
    combined_output = "".join(output_chunks)

    step_record = {
        "index": len(PIPELINE_RECORDER.steps) if PIPELINE_RECORDER is not None else None,
        "started_at": started_at,
        "finished_at": _now_iso(),
        "elapsed_sec": round(elapsed_sec, 3),
        "cwd": str(cwd),
        "command": [str(x) for x in cmd],
        "command_str": command_str,
        "return_code": int(return_code),
        "thread_environment": {name: env[name] for name in BLAS_THREAD_ENV if env and name in env},
        "stdout_stderr": combined_output,
    }
    if PIPELINE_RECORDER is not None:
        PIPELINE_RECORDER.add_step(step_record)

    if return_code != 0:
        error = f"Command failed with exit code {return_code}: {command_str}"
        if PIPELINE_RECORDER is not None:
            PIPELINE_RECORDER.write(status="failed", error=error)
        raise subprocess.CalledProcessError(return_code, cmd, output=combined_output)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def read_json_file(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return data


def _redact_metric_scale_result_for_log(metric_scale_result: Any) -> Any:
    if isinstance(metric_scale_result, dict):
        out = {}
        for k, v in metric_scale_result.items():
            if k == "per_bone_scale_meta":
                continue
            out[k] = _redact_metric_scale_result_for_log(v)
        return out
    if isinstance(metric_scale_result, list):
        return [_redact_metric_scale_result_for_log(v) for v in metric_scale_result]
    return metric_scale_result



def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    py = sys.executable
    env = build_subprocess_env(cfg)

    for script, name in [
        (SCRIPT_01_CAMERA_POSES, "01_convert_jsonl_to_camera_poses.py"),
        (SCRIPT_02_EXTRACT_HANDS, "02_extract_hawor_world_hands.py"),
        (SCRIPT_03_STABILIZE_HANDS, "03_stabilize_hawor_hands_for_egoallo.py"),
        (SCRIPT_EXTRACT_ROT, "extract_hawor_to_vggt_rotation.py"),
        (SCRIPT_04_BUILD_HAMER, "04_build_egoallo_hamer_from_hawor_world.py"),
        (SCRIPT_PREPARE_REAL_SENSOR, "04_prepare_real_sensor_inputs.py"),
        (SCRIPT_05_APPLY_SCALE, "05_apply_metric_scale.py"),
        (SCRIPT_RUN_GEOCALIB, "run_geocalib.py"),
        (SCRIPT_ESTIMATE_GRAVITY, "01_estimate_gravity_da3_geocalib.py"),
        (SCRIPT_APPLY_ALIGNMENT, "02_apply_alignment_da3_geocalib.py"),
        (SCRIPT_VALIDATE_POSE_BEFORE_EGOGVAE, "05c_validate_pose_before_egogvae.py"),
        (SCRIPT_RUN_EGOGVAE, "05_run_egogvae_from_head.py"),
        (SCRIPT_RUN_EGOALLO_REFINE, "07_run_egoallo_refine_with_prior.py"),
    ]:
        require_file(script, name)
    episode_dir = args.episode_dir.expanduser().resolve()
    downstream_dir = args.downstream_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = output_dir / "tmp" / "upstream_converted"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    upstream = _prepare_upstream_inputs(
        episode_dir=episode_dir,
        downstream_dir=downstream_dir,
        prepared_dir=prepared_dir,
        frame_count=int(get_cfg(cfg, "geocalib.num_samples", 30)),
    )
    frames_dir = upstream.frames_dir
    cam_space_dir = upstream.cam_space_dir
    hawor_hands_vggt_world = upstream.hawor_hands_vggt_world
    hawor_to_vggt_alignment = upstream.hawor_to_vggt_alignment
    vggt_head_pose = upstream.vggt_head_pose
    scene_points_ply = upstream.scene_points_ply
    objects_parquet = upstream.objects_parquet
    object_meshes = upstream.object_meshes
    real_sensor_input = resolve_real_sensor_input(episode_dir)
    real_sensor_has_imu = real_sensor_input is not None and episode_imu_path(real_sensor_input) is not None
    real_sensor_has_depth = real_sensor_input is not None and episode_has_depth_stream(real_sensor_input)
    real_sensor_rgb_stream_id = None
    real_sensor_depth_stream_id = None
    if real_sensor_input is not None:
        configured_rgb_stream = str(get_cfg(cfg, "real_sensor.rgb_camera_stream_id", "auto"))
        real_sensor_rgb_stream_id = (
            episode_stream_id(real_sensor_input, "rgb")
            if configured_rgb_stream == "auto"
            else configured_rgb_stream
        )
        if real_sensor_rgb_stream_id is None:
            raise ValueError("No RGB stream found in episode.json")
        if real_sensor_has_depth:
            configured_depth_stream = str(get_cfg(cfg, "real_sensor.depth_camera_stream_id", "auto"))
            real_sensor_depth_stream_id = (
                episode_stream_id(real_sensor_input, "depth")
                if configured_depth_stream == "auto"
                else configured_depth_stream
            )
            if real_sensor_depth_stream_id is None:
                raise ValueError("No depth stream found in episode.json")

    require_dir(frames_dir, "frames_dir")
    require_file(vggt_head_pose, "vggt_head_pose")
    require_file(scene_points_ply, "scene_points_ply")

    tmp_dir = output_dir / str(get_cfg(cfg, "output_layout.tmp_dir_name", "tmp"))
    egogvae_inputs_dir = output_dir / str(get_cfg(cfg, "output_layout.egogvae_inputs_dir_name", "egogvae_inputs"))
    egogvae_outputs_dir = output_dir / str(get_cfg(cfg, "output_layout.egogvae_outputs_dir_name", "egogvae_outputs"))
    egogvae_tmp_dir = output_dir / str(get_cfg(cfg, "output_layout.egogvae_tmp_dir_name", "egogvae_tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    egogvae_inputs_dir.mkdir(parents=True, exist_ok=True)
    egogvae_outputs_dir.mkdir(parents=True, exist_ok=True)
    egogvae_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Keep Python bytecode cache out of project source directories.
    python_pycache_dir, python_pycache_cleanup = configure_python_pycache(env, cfg, tmp_dir)
    jax_compilation_cache_dir = configure_jax_compilation_cache(env, cfg, tmp_dir, PROJECT_ROOT)

    global PIPELINE_RECORDER
    pipeline_log_json = tmp_dir / str(get_cfg(cfg, "output_layout.pipeline_log_json_name", "pipeline_run_log.json"))
    PIPELINE_RECORDER = PipelineRunRecorder(
        pipeline_log_json,
        metadata={
            "project_root": str(PROJECT_ROOT),
            "config_path": str(args.config.expanduser().resolve()),
            "config_snapshot": cfg,
            "output_dir": str(output_dir),
            "tmp_dir": str(tmp_dir),
            "egogvae_inputs_dir": str(egogvae_inputs_dir),
            "egogvae_outputs_dir": str(egogvae_outputs_dir),
            "egogvae_tmp_dir": str(egogvae_tmp_dir),
            "python_executable": sys.executable,
            "python_pycache_dir": str(python_pycache_dir) if python_pycache_dir is not None else None,
            "python_pycache_cleanup": python_pycache_cleanup,
            "jax_compilation_cache_dir": str(jax_compilation_cache_dir) if jax_compilation_cache_dir is not None else None,
            "inputs": {
                "frames_dir": str(frames_dir),
                "cam_space_dir": str(cam_space_dir),
                "hawor_hands_vggt_world": str(hawor_hands_vggt_world),
                "egoallo_hand_input_parquet": str(upstream.hands_parquet),
                "hawor_to_vggt_alignment": str(hawor_to_vggt_alignment),
                "vggt_head_pose": str(vggt_head_pose),
                "scene_points_ply": str(scene_points_ply),
                "real_sensor_input": str(real_sensor_input) if real_sensor_input is not None else None,
            },
        },
    )

    camera_poses_txt = tmp_dir / "camera_poses.txt"
    timestamps_txt = egogvae_inputs_dir / "timestamps.txt"
    intrinsics_txt = tmp_dir / "intrinsics.txt"

    prepared_hands_jsonl = tmp_dir / "hawor_hands_for_egoallo_world.jsonl"
    stable_hands_jsonl = tmp_dir / "hawor_hands_for_egoallo_world_stable.jsonl"
    R_hawor_to_vggt_npy = tmp_dir / "R_hawor_to_vggt.npy"
    hamer_outputs_pkl_tmp = tmp_dir / "hamer_outputs.pkl"
    empty_hawor_hands_vggt_world = tmp_dir / "empty_hawor_hands_vggt_world.jsonl"

    geocalib_gravity_npz = tmp_dir / "geocalib_gravity.npz"
    imu_gravity_txt = tmp_dir / "imu_gravity.txt"
    R_align_npy = tmp_dir / "R_align.npy"
    alignment_transform_npz = tmp_dir / "alignment_transform.npz"
    aligned_scene_ply = tmp_dir / "aligned_scene.ply"
    head_trajectory_tmp = tmp_dir / "head_trajectory.npy"

    head_trajectory_out = egogvae_inputs_dir / "head_trajectory.npy"
    hamer_outputs_pkl_out = egogvae_inputs_dir / "hamer_outputs.pkl"
    final_scene_name = str(get_cfg(cfg, "output_layout.final_scene_name", "final_scene.ply"))
    final_scene_ply = _assets_pointclouds_dir(output_dir, cfg) / final_scene_name
    egogvae_motion_name = str(get_cfg(cfg, "output_layout.egogvae_motion_name", "egogvae_motion.npz"))
    egogvae_motion_npz = egogvae_outputs_dir / egogvae_motion_name
    final_smoothed_npz = egogvae_outputs_dir / FINAL_MOTION_NAME
    frontend_head_name = str(
        get_cfg(cfg, "metric_scale.frontend.head_jsonl_name", "head_trajectory_egoallo.jsonl")
    )
    frontend_hands_name = str(
        get_cfg(cfg, "metric_scale.frontend.hands_jsonl_name", "hand_keypoints_egoallo.jsonl")
    )
    frontend_head_jsonl = egogvae_outputs_dir / frontend_head_name
    frontend_hands_jsonl = egogvae_outputs_dir / frontend_hands_name
    objects_egoallo_parquet = egogvae_outputs_dir / "objects_egoallo.parquet"
    object_mesh_outputs = _object_mesh_output_paths(object_meshes, downstream_dir, output_dir)
    motion_quality_report_json = tmp_dir / str(get_cfg(cfg, "egogvae.report_json_name", "motion_quality_report.json"))
    egoallo_refine_report_json = tmp_dir / str(get_cfg(cfg, "egoallo_refine.report_json_name", "egoallo_refine_report.json"))
    egoallo_hawor_cache_npz = tmp_dir / str(get_cfg(cfg, "egoallo_refine.cache_npz_name", "egoallo_hawor_final_cache.npz"))
    pose_validation_report_json = tmp_dir / str(get_cfg(cfg, "pose_validation.report_json_name", "pose_validation_report.json"))
    pose_validation_result: dict[str, Any] = {
        "enabled": False,
        "status": "disabled",
        "should_run_egogvae": True,
    }
    egoallo_refine_result: dict[str, Any] = {
        "enabled": _cfg_bool(get_cfg(cfg, "egoallo_refine.enabled", True), default=True),
        "status": "not_run",
    }
    egogvae_inference_skipped = False
    metric_scale_metadata_json = tmp_dir / "metric_scale_metadata.json"
    real_depth_scale_json = tmp_dir / "real_depth_scale_estimate.json"
    real_imu_alignment_failed = False
    reference_only_reason: str | None = None

    mano_left = resolve_project_path(get_cfg(cfg, "model_weights.mano_left_pkl"))
    mano_right = resolve_project_path(get_cfg(cfg, "model_weights.mano_right_pkl"))
    if mano_left is None or mano_right is None:
        raise ValueError("model_weights.mano_left_pkl and model_weights.mano_right_pkl are required in YAML")
    require_file(mano_left, "mano_left_pkl")
    require_file(mano_right, "mano_right_pkl")

    # 01: VGGT head pose jsonl -> camera_poses / timestamps / intrinsics
    run([
        py,
        str(SCRIPT_01_CAMERA_POSES),
        "--jsonl", str(vggt_head_pose),
        "--camera-poses-path", str(camera_poses_txt),
        "--timestamps-path", str(timestamps_txt),
        "--intrinsics-path", str(intrinsics_txt),
    ], env=env)

    if real_sensor_input is not None:
        # timestamps.txt lets the IMU step attach its stationary window to the pose
        # captured at the same time, instead of always using the first pose.
        real_sensor_prepare_cmd = [
            py,
            str(SCRIPT_PREPARE_REAL_SENSOR),
            "--episode-dir", str(real_sensor_input),
            "--camera-stream-id", str(real_sensor_rgb_stream_id),
            "--intrinsics-out", str(intrinsics_txt),
            "--imu-gravity-out", str(imu_gravity_txt),
            "--timestamps-txt", str(timestamps_txt),
            "--imu-stationary-window-sec", str(get_cfg(cfg, "real_sensor.imu_stationary_window_sec", 0.5)),
            "--imu-stationary-max-gyro", str(get_cfg(cfg, "real_sensor.imu_stationary_max_gyro", 0.25)),
            "--imu-stationary-max-dir-spread-deg", str(get_cfg(cfg, "real_sensor.imu_stationary_max_dir_spread_deg", 3.0)),
            "--imu-stationary-max-norm-rel-dev", str(get_cfg(cfg, "real_sensor.imu_stationary_max_norm_rel_dev", 0.05)),
        ]
        run(real_sensor_prepare_cmd, env=env)

    if hawor_hands_vggt_world.is_file():
        hands_jsonl_for_pipeline = hawor_hands_vggt_world
    else:
        empty_hawor_hands_vggt_world.write_text("", encoding="utf-8")
        hands_jsonl_for_pipeline = empty_hawor_hands_vggt_world
        print(
            f"[HaWoR] No hand JSONL found at {hawor_hands_vggt_world}; "
            f"using empty hand detections: {empty_hawor_hands_vggt_world}",
            flush=True,
        )

    if not cam_space_dir.is_dir():
        print(
            f"[HaWoR] cam_space_dir not found: {cam_space_dir}; "
            "continuing without MANO cam_space records.",
            flush=True,
        )

    # 02: HaWoR world hands -> prepared hand records for scale/frontend outputs.
    cmd = [
        py,
        str(SCRIPT_02_EXTRACT_HANDS),
        "--hands-jsonl", str(hands_jsonl_for_pipeline),
        "--timestamps-txt", str(timestamps_txt),
        "--output-jsonl", str(prepared_hands_jsonl),
        "--soft-start-frames", str(get_cfg(cfg, "hawor_extract.soft_start_frames", 15)),
        "--soft-end-frames", str(get_cfg(cfg, "hawor_extract.soft_end_frames", 6)),
        "--min-output-alpha", str(get_cfg(cfg, "hawor_extract.min_output_alpha", 0.0)),
        "--max-wrist-step-hand-ratio", str(get_cfg(cfg, "hawor_extract.max_wrist_step_hand_ratio", 2.0)),
        "--min-track-len", str(get_cfg(cfg, "hawor_extract.min_track_len", 3)),
        "--trim-segment-edges", str(get_cfg(cfg, "hawor_extract.trim_segment_edges", 0)),
        "--max-gap-fill", str(get_cfg(cfg, "hawor_extract.max_gap_fill", 0)),
        "--max-gap-endpoint-hand-ratio", str(get_cfg(cfg, "hawor_extract.max_gap_endpoint_hand_ratio", 1.5)),
        "--smooth-window", str(get_cfg(cfg, "hawor_extract.smooth_window", 5)),
        "--smooth-mode", str(get_cfg(cfg, "hawor_extract.smooth_mode", "translation")),
        "--soft-start-anchor-max-gap", str(get_cfg(cfg, "hawor_extract.soft_start_anchor_max_gap", 45)),
        "--local-mad-window", str(get_cfg(cfg, "hawor_extract.local_mad_window", 7)),
        "--local-mad-multiplier", str(get_cfg(cfg, "hawor_extract.local_mad_multiplier", 6.0)),
        "--local-mad-min-samples", str(get_cfg(cfg, "hawor_extract.local_mad_min_samples", 4)),
    ]
    hand_filter_enabled = bool(get_cfg(cfg, "hand_filter.enabled", False))
    if hand_filter_enabled:
        cmd.extend([
            "--near-hand-filter",
            "--near-hand-distance-ratio", str(get_cfg(cfg, "hand_filter.near_distance_ratio", 3.0)),
            "--near-hand-min-support", str(get_cfg(cfg, "hand_filter.min_support", 3)),
            "--near-hand-max-gap-seconds", str(get_cfg(cfg, "hand_filter.max_gap_seconds", 0.6)),
            "--near-hand-max-speed-hand-lengths", str(get_cfg(cfg, "hand_filter.max_speed_hand_lengths", 20.0)),
            "--near-hand-max-distance-change-ratio", str(get_cfg(cfg, "hand_filter.max_distance_change_ratio", 1.5)),
            "--near-hand-ambiguity-ratio", str(get_cfg(cfg, "hand_filter.ambiguity_ratio", 1.25)),
        ])
    add_flag(cmd, "--drop-isolated", get_cfg(cfg, "hawor_extract.drop_isolated", True))
    if not bool(get_cfg(cfg, "hawor_extract.drop_isolated", True)):
        cmd.append("--no-drop-isolated")
    add_flag(cmd, "--soft-start-geometry", get_cfg(cfg, "hawor_extract.soft_start_geometry", True))
    if not bool(get_cfg(cfg, "hawor_extract.soft_start_geometry", True)):
        cmd.append("--no-soft-start-geometry")
    add_flag(cmd, "--print-jump-stats", get_cfg(cfg, "hawor_extract.print_jump_stats", True))
    run(cmd, env=env)

    # 03: stabilize hands for scale estimation. EgoAllo uses upstream geometry.
    run([
        py,
        str(SCRIPT_03_STABILIZE_HANDS),
        "--input-jsonl", str(prepared_hands_jsonl),
        "--output-jsonl", str(stable_hands_jsonl),
        "--max-flip-dist-ratio", str(get_cfg(cfg, "hawor_stabilize.max_flip_dist_ratio", 1.5)),
        "--max-wrist-step-hand-ratio", str(get_cfg(cfg, "hawor_stabilize.max_wrist_step_hand_ratio", 2.0)),
        "--local-mad-window", str(get_cfg(cfg, "hawor_stabilize.local_mad_window", 7)),
        "--local-mad-multiplier", str(get_cfg(cfg, "hawor_stabilize.local_mad_multiplier", 6.0)),
        "--local-mad-min-samples", str(get_cfg(cfg, "hawor_stabilize.local_mad_min_samples", 4)),
        "--min-track-len", str(get_cfg(cfg, "hawor_stabilize.min_track_len", 4)),
        "--soft-start-frames", str(get_cfg(cfg, "hawor_stabilize.soft_start_frames", 15)),
        "--soft-end-frames", str(get_cfg(cfg, "hawor_stabilize.soft_end_frames", 6)),
        "--min-alpha-to-keep", str(get_cfg(cfg, "hawor_stabilize.min_alpha_to_keep", 0.0)),
        "--duplicate-mode", str(get_cfg(cfg, "hawor_stabilize.duplicate_mode", "keep_best")),
    ], env=env)

    # EgoAllo takes every upstream 30 FPS hand record, without the scale
    # estimation branch's filtering, smoothing or gap filling.
    egoallo_hands_jsonl = hawor_hands_vggt_world

    # sim3.R maps column vectors from HaWoR camera to scene camera; do not
    # transpose it when transforming the MANO global orientation.
    # If no hands were detected upstream, the alignment
    # file may be absent; use identity so the empty-hand path can still finish.
    run([
        py,
        str(SCRIPT_EXTRACT_ROT),
        "--alignment-json", str(hawor_to_vggt_alignment),
        "--json-key", "sim3.R",
        "--output-npy", str(R_hawor_to_vggt_npy),
        "--missing-is-identity",
    ], env=env)

    # 04: MANO chunks retain original sampled camera IDs. Interpolate their
    # parameters onto the 30 FPS hand timeline using episode timestamps.
    hamer_build_cmd = [
        py,
        str(SCRIPT_04_BUILD_HAMER),
        "--prepared-hands-jsonl", str(egoallo_hands_jsonl),
        "--camera-poses-txt", str(camera_poses_txt),
        "--timestamps-txt", str(timestamps_txt),
        "--output-pkl", str(hamer_outputs_pkl_tmp),
        "--cam-space-left-dir", str(cam_space_dir / str(get_cfg(cfg, "hamer_build.cam_space_left_subdir", "0"))),
        "--cam-space-right-dir", str(cam_space_dir / str(get_cfg(cfg, "hamer_build.cam_space_right_subdir", "1"))),
        "--mano-frame-key", str(get_cfg(cfg, "hamer_build.mano_frame_key", "frame_id")),
        "--mano-left-pkl", str(mano_left),
        "--mano-right-pkl", str(mano_right),
        "--camera-frame-rotation-npy", str(R_hawor_to_vggt_npy),
    ]
    mano_timeline = cam_space_dir.parent / "camera.parquet"
    if mano_timeline.is_file():
        hamer_build_cmd.extend(["--mano-timeline-parquet", str(mano_timeline)])
    run(hamer_build_cmd, env=env)

    if not real_sensor_has_imu:
        # GeoCalib gravity：可选参数从 YAML 的 geocalib 段读取。
        geocalib_cmd = [
            py,
            str(SCRIPT_RUN_GEOCALIB),
            "--img-dir", str(frames_dir),
            "--output", str(geocalib_gravity_npz),
            "--device", str(get_cfg(cfg, "geocalib.device", "auto")),
            "--num-samples", str(get_cfg(cfg, "geocalib.num_samples", 30)),
            "--anchor-policy", str(get_cfg(cfg, "geocalib.anchor_policy", "top_k_best_uncertainty")),
            "--top-k-anchors", str(get_cfg(cfg, "geocalib.top_k_anchors", 3)),
            "--empty-policy", str(get_cfg(cfg, "geocalib.empty_policy", "skip")),
        ]
        geocalib_min_confidence = get_cfg(cfg, "geocalib.min_confidence", None)
        if geocalib_min_confidence is not None:
            geocalib_cmd.extend(["--min-confidence", str(geocalib_min_confidence)])

        if bool(get_cfg(cfg, "geocalib.viz_enabled", False)):
            viz_dir_name = str(get_cfg(cfg, "geocalib.viz_dir_name", "geocalib_viz"))
            geocalib_viz_dir = tmp_dir / viz_dir_name
            geocalib_cmd.extend(["--viz-dir", str(geocalib_viz_dir)])
            if bool(get_cfg(cfg, "geocalib.viz_save_all_sampled", True)):
                geocalib_cmd.append("--save-all-sampled-viz")
            if bool(get_cfg(cfg, "geocalib.viz_save_selected", False)):
                geocalib_cmd.append("--save-selected-viz")

        geocalib_root = env.get("GEOCALIB_ROOT")
        if geocalib_root:
            geocalib_cmd.extend(["--geocalib-root", geocalib_root])
        if env.get("GEOCALIB_PINHOLE_TAR"):
            geocalib_cmd.extend(["--geocalib-pinhole-weight", env["GEOCALIB_PINHOLE_TAR"]])

        run(geocalib_cmd, env=env)

    # Gravity alignment
    # Two different modes inside 01_estimate_gravity_da3_geocalib.py:
    #
    # 1. Real IMU (real_sensor_has_imu): the measured vertical, taken from
    #    the most stationary IMU window, restricts the floor search to the low region
    #    along that axis. The final up still comes from refine_floor_coarse_to_fine,
    #    which must stay within imu_max_refine_angle_deg of the IMU axis. If every
    #    candidate misses that limit, downstream inputs are still exported and EgoGVAE
    #    inference is skipped.
    #
    # 2. No IMU: point-cloud-only RANSAC baseline first. GeoCalib is optional and is
    #    used only as a prior when it disagrees with RANSAC beyond the switch angle,
    #    and the resulting floor/up proposal must pass camera-height validation or it
    #    will fall back to RANSAC.
    estimate_gravity_cmd = [
        py,
        str(SCRIPT_ESTIMATE_GRAVITY),
        "--pcd", str(scene_points_ply),
        "--poses", str(camera_poses_txt),
        "--hands-jsonl", str(stable_hands_jsonl),
        "--timestamps", str(timestamps_txt),
        "--out", str(R_align_npy),
        "--out_meta", str(alignment_transform_npz),
    ]

    if real_sensor_has_imu:
        estimate_gravity_cmd.extend([
            "--imu_gravity", str(imu_gravity_txt),
            "--imu_vector_type", str(get_cfg(cfg, "real_sensor.imu_vector_type", "up")),
            "--imu_min_camera_above_hand_ratio", str(get_cfg(cfg, "real_sensor.imu_min_camera_above_hand_ratio", 0.5)),
            "--imu_outlier_angle_deg", str(get_cfg(cfg, "real_sensor.imu_outlier_angle_deg", 180.0)),
            "--imu_max_refine_angle_deg", str(get_cfg(cfg, "real_sensor.imu_max_refine_angle_deg", 10.0)),
            "--imu_max_head_height_rel_spread", str(get_cfg(cfg, "real_sensor.imu_max_head_height_rel_spread", 0.75)),
            "--imu_max_points_below_floor_ratio", str(get_cfg(cfg, "real_sensor.imu_max_points_below_floor_ratio", 0.05)),
            "--imu_min_head_span_ratio", str(get_cfg(cfg, "real_sensor.imu_min_head_span_ratio", 0.15)),
            "--imu_max_head_span_ratio", str(get_cfg(cfg, "real_sensor.imu_max_head_span_ratio", 2.50)),
        ])
    else:
        # GeoCalib-only switches. In IMU mode they have no effect, so they are not passed.
        if bool(get_cfg(cfg, "alignment.use_geocalib_prior", True)):
            estimate_gravity_cmd.extend([
                "--geocalib_gravity", str(geocalib_gravity_npz),
                "--geocalib_vector_type", str(get_cfg(cfg, "alignment.geocalib_vector_type", "gravity")),
                "--geocalib_switch_angle_deg", str(get_cfg(cfg, "alignment.geocalib_switch_angle_deg", 10.0)),
                "--geocalib_outlier_angle_deg", str(get_cfg(cfg, "alignment.geocalib_outlier_angle_deg", 15.0)),
            ])

        if bool(get_cfg(cfg, "alignment.geocalib_precheck_height", True)):
            estimate_gravity_cmd.append("--geocalib_precheck_height")

        if bool(get_cfg(cfg, "alignment.fallback_to_ransac_on_invalid_height", True)):
            estimate_gravity_cmd.append("--fallback_to_ransac_on_invalid_height")

    estimate_gravity_cmd.extend([
        "--min_valid_camera_height_margin_factor",
        str(get_cfg(cfg, "alignment.min_valid_camera_height_margin_factor", 2.0)),
        "--min_camera_above_floor_ratio",
        str(get_cfg(cfg, "alignment.min_camera_above_floor_ratio", 0.75)),
        "--min_camera_p10_height_margin_factor",
        str(get_cfg(cfg, "alignment.min_camera_p10_height_margin_factor", -2.0)),
    ])

    run(estimate_gravity_cmd, env=env)

    alignment_decision: dict[str, Any] = {}
    if real_sensor_has_imu:
        alignment_decision = read_imu_alignment_decision(alignment_transform_npz)
        print(
            "[IMU] alignment outcome: "
            f"used_imu={alignment_decision.get('used_imu_gravity')} "
            f"accepted={alignment_decision.get('accepted')} "
            f"candidates={alignment_decision.get('num_candidates_tried')} "
            f"reason={alignment_decision.get('decision_reason')}",
            flush=True,
        )
        if alignment_decision.get("accepted") is False:
            real_imu_alignment_failed = True
            reference_only_reason = "imu_floor_estimation_failed"
            mark_alignment_reference_only(alignment_transform_npz, reference_only_reason)
            alignment_decision = read_imu_alignment_decision(alignment_transform_npz)
            print(
                "[IMU] floor estimation failed; exporting IMU-aligned inputs/scene "
                "at input scale. Metric scale and body inference will be skipped.",
                flush=True,
            )

    run([
        py,
        str(SCRIPT_APPLY_ALIGNMENT),
        "--poses", str(camera_poses_txt),
        "--R", str(R_align_npy),
        "--align_meta", str(alignment_transform_npz),
        "--out_poses", str(head_trajectory_tmp),
        "--poses-only",
    ], env=env)

    # Optional FoundationGeo metric-scale estimation from gravity-aligned ground depth.
    # 05_apply_metric_scale.py takes max(FoundationGeo, hands), then caps p90 head height.
    foundationgeo_scale_json: Path | None = None
    foundationgeo_output_dir: Path | None = None
    metric_scale_enabled = bool(get_cfg(cfg, "metric_scale.enabled", True))
    real_depth_scale_ok = False
    real_depth_scale_reason = "not_run"
    foundationgeo_enabled = (
        metric_scale_enabled
        and not real_imu_alignment_failed
        and bool(get_cfg(cfg, "metric_scale.foundationgeo.enabled", False))
    )
    foundationgeo_model_pt: Path | None = None
    if foundationgeo_enabled:
        require_file(SCRIPT_ESTIMATE_FOUNDATIONGEO_SCALE, "04_estimate_foundationgeo_metric_scale.py")
        foundationgeo_output_dir = tmp_dir / str(get_cfg(cfg, "metric_scale.foundationgeo.output_dir_name", "foundationgeo_metric_scale"))
        foundationgeo_scale_json = foundationgeo_output_dir / "foundationgeo_scale_estimate.json"
        foundationgeo_model_pt = resolve_project_path(get_cfg(cfg, "model_weights.foundationgeo_model_pt", env.get("FOUNDATIONGEO_MODEL_PT")))

    def run_foundationgeo_metric_scale_if_needed() -> None:
        if not foundationgeo_enabled or foundationgeo_scale_json is None:
            return
        if foundationgeo_model_pt is None:
            write_failed_scale_json(
                foundationgeo_scale_json,
                source="foundationgeo",
                reason="model_weights.foundationgeo_model_pt_not_set",
            )
            print(f"[FoundationGeo] Skip FoundationGeo scale: model_weights.foundationgeo_model_pt is not set. Wrote {foundationgeo_scale_json}", flush=True)
        elif not foundationgeo_model_pt.is_file():
            write_failed_scale_json(
                foundationgeo_scale_json,
                source="foundationgeo",
                reason="foundationgeo_model_pt_not_found",
                extra={"foundationgeo_model_pt": str(foundationgeo_model_pt)},
            )
            print(f"[FoundationGeo] Skip FoundationGeo scale: model.pt not found: {foundationgeo_model_pt}. Wrote {foundationgeo_scale_json}", flush=True)
        else:
            foundationgeo_cmd = [
                py,
                str(SCRIPT_ESTIMATE_FOUNDATIONGEO_SCALE),
                "--frames-dir", str(frames_dir),
                "--timestamps-txt", str(timestamps_txt),
                "--head-trajectory-in", str(head_trajectory_tmp),
                "--intrinsics-path", str(intrinsics_txt),
                "--foundationgeo-model-pt", str(foundationgeo_model_pt),
                "--output-dir", str(foundationgeo_output_dir),
                "--output-json", str(foundationgeo_scale_json),
                "--num-samples", str(get_cfg(cfg, "metric_scale.foundationgeo.num_samples", 32)),
                "--depth-scale-cluster-rel-tol", str(get_cfg(cfg, "metric_scale.foundationgeo.depth_scale_cluster_rel_tol", 0.08)),
                "--depth-scale-min-support-frames", str(get_cfg(cfg, "metric_scale.foundationgeo.depth_scale_min_support_frames", 3)),
                "--device", str(get_cfg(cfg, "metric_scale.foundationgeo.device", "cuda")),
                "--fp16", str(get_cfg(cfg, "metric_scale.foundationgeo.fp16", False)),
                "--resolution-level", str(get_cfg(cfg, "metric_scale.foundationgeo.resolution_level", 9)),
            ]
            foundationgeo_num_tokens = get_cfg(cfg, "metric_scale.foundationgeo.num_tokens", None)
            if foundationgeo_num_tokens is not None:
                foundationgeo_cmd.extend(["--num-tokens", str(foundationgeo_num_tokens)])
            if bool(get_cfg(cfg, "metric_scale.foundationgeo.strict", False)):
                foundationgeo_cmd.append("--strict")
            run(foundationgeo_cmd, env=env)

    hand_scale_bones_json = tmp_dir / "hand_scale_bones.json"

    metric_scale_cmd = [
        py,
        str(SCRIPT_05_APPLY_SCALE),
        "--head-trajectory-in", str(head_trajectory_tmp),
        "--point-cloud-in", str(scene_points_ply),
        "--hamer-pkl-in", str(hamer_outputs_pkl_tmp),
        "--point-cloud-space", "raw_vggt",
        "--point-cloud-r-align", str(R_align_npy),
        "--point-cloud-align-meta", str(alignment_transform_npz),
        "--final-scene-output-voxel-size", str(get_cfg(cfg, "metric_scale.final_scene_output_voxel_size", 0.0)),
        "--head-trajectory-out", str(head_trajectory_out),
        "--point-cloud-out", str(final_scene_ply),
        "--objects-parquet-in", str(objects_parquet),
        "--objects-parquet-out", str(objects_egoallo_parquet),
        "--hamer-pkl-out", str(hamer_outputs_pkl_out),
        "--metadata-out", str(metric_scale_metadata_json),
        "--floor-z", str(get_cfg(cfg, "metric_scale.floor_z", 0.0)),
        "--hand-scale-enabled", "1" if bool(get_cfg(cfg, "metric_scale.hands.enabled", True)) and not real_sensor_has_depth else "0",
        "--hand-scale-bones-json", str(hand_scale_bones_json),
        "--hand-scale-min-valid-bones", str(get_cfg(cfg, "metric_scale.hands.min_valid_bones", 20)),
        "--hand-scale-max-relative-mad", str(get_cfg(cfg, "metric_scale.hands.max_relative_mad", 0.35)),
        "--hand-scale-min-scale", str(get_cfg(cfg, "metric_scale.hands.min_scale", 0.05)),
        "--hand-scale-max-scale", str(get_cfg(cfg, "metric_scale.hands.max_scale", 20.0)),
        "--max-p90-head-height", str(get_cfg(cfg, "metric_scale.max_p90_head_height", 1.60)),
        "--hands-vggt-jsonl-for-scale", str(stable_hands_jsonl),
        "--R-align-for-scale", str(R_align_npy),
        "--align-meta-for-scale", str(alignment_transform_npz),
    ]

    for mesh_in, mesh_out in zip(object_meshes, object_mesh_outputs):
        metric_scale_cmd.extend([
            "--object-mesh-in", str(mesh_in),
            "--object-mesh-out", str(mesh_out),
        ])

    if hand_filter_enabled:
        metric_scale_cmd.extend([
            "--hand-max-camera-distance-m", str(get_cfg(cfg, "hand_filter.max_camera_distance_m", 1.2)),
        ])
    if real_imu_alignment_failed or bool(get_cfg(cfg, "metric_scale.frontend.enabled", True)):
        metric_scale_cmd.extend([
            "--export-frontend-jsonl",
            "--frontend-output-dir", str(egogvae_outputs_dir),
            "--frontend-head-jsonl-name", frontend_head_name,
            "--frontend-hands-jsonl-name", frontend_hands_name,
            "--timestamps-txt", str(timestamps_txt),
            "--hands-vggt-jsonl-in", str(hawor_hands_vggt_world),
            "--R-align", str(R_align_npy),
            "--align-meta", str(alignment_transform_npz),
        ])
        add_flag(
            metric_scale_cmd,
            "--frontend-include-mesh",
            get_cfg(cfg, "metric_scale.frontend.include_mesh", False),
        )

    # Metric scale
    # 05_apply_metric_scale.py no longer reads YAML directly. run_body_mesh.py is the single
    # config owner: it reads metric_scale.* from YAML and passes concrete CLI args here.
    if real_imu_alignment_failed:
        real_depth_scale_reason = "skipped_imu_floor_estimation_failed"
        # Reuse all normal geometry exporters, but never call a scale estimator
        # or apply a height cap / metric-distance filter on this fallback path.
        metric_scale_cmd.extend(["--alignment-only", "--alignment-only-reason", reference_only_reason])
        run(metric_scale_cmd, env=env)
    elif metric_scale_enabled:
        hand_scale_bones_json = tmp_dir / "hand_scale_bones.json"
        with hand_scale_bones_json.open("w", encoding="utf-8") as f:
            json.dump(
                get_cfg(cfg, "metric_scale.hands.bones", []),
                f,
                ensure_ascii=False,
                indent=2,
            )

        if real_sensor_has_depth:
            real_depth_cmd = [
                py,
                str(SCRIPT_PREPARE_REAL_SENSOR),
                "--episode-dir", str(real_sensor_input),
                "--camera-stream-id", str(real_sensor_depth_stream_id),
                "--scale-json", str(real_depth_scale_json),
                "--timestamps-txt", str(timestamps_txt),
                "--head-trajectory-in", str(head_trajectory_tmp),
                "--point-cloud-r-align", str(R_align_npy),
                "--point-cloud-align-meta", str(alignment_transform_npz),
                "--depth-cache-mb", str(get_cfg(cfg, "real_sensor.depth_cache_mb", 512)),
                "--num-samples", str(get_cfg(cfg, "real_sensor.depth_scale_num_samples", 96)),
                "--depth-confidence-min", str(get_cfg(cfg, "real_sensor.depth_confidence_min", 0.5)),
                "--depth-min-disparity-px", str(get_cfg(cfg, "real_sensor.depth_min_disparity_px", 0.0)),
                "--floor-max-histogram-bins", str(get_cfg(cfg, "real_sensor.depth_floor_max_histogram_bins", 4096)),
                "--depth-scale-cluster-rel-tol", str(get_cfg(cfg, "real_sensor.depth_scale_cluster_rel_tol", 0.08)),
                "--depth-scale-min-support-frames", str(get_cfg(cfg, "real_sensor.depth_scale_min_support_frames", 3)),
                "--depth-scale-local-window", str(get_cfg(cfg, "real_sensor.depth_scale_local_window", 15)),
                "--depth-scale-local-min-support-frames", str(get_cfg(cfg, "real_sensor.depth_scale_local_min_support_frames", 5)),
            ]
            for key, option in (("depth_unit_scale", "--depth-unit-scale"),
                                ("depth_min_m", "--depth-min-m"), ("depth_max_m", "--depth-max-m"),
                                ("depth_floor_max_abs_height_m", "--floor-max-abs-height-m")):
                value = get_cfg(cfg, f"real_sensor.{key}", None)
                if value is not None:
                    real_depth_cmd.extend([option, str(value)])
            for quality in ("validity", "confidence", "disparity"):
                array_name = get_cfg(cfg, f"real_sensor.depth_{quality}_array", None)
                if array_name:
                    real_depth_cmd.extend([f"--depth-{quality}-array", str(array_name)])
            invalid_values = get_cfg(cfg, "real_sensor.depth_invalid_values", [])
            if invalid_values:
                real_depth_cmd.extend(["--depth-invalid-values", *[str(v) for v in invalid_values]])
            run(real_depth_cmd, env=env)
            try:
                with real_depth_scale_json.open("r", encoding="utf-8") as f:
                    real_depth_data = json.load(f)
                real_depth_scale = finite_positive_float(real_depth_data.get("scale"))
                real_depth_scale_ok = (
                    str(real_depth_data.get("status", "")).lower() in {"ok", "success", "valid"}
                    and real_depth_scale is not None
                )
                real_depth_scale_reason = str(real_depth_data.get("reason") or real_depth_data.get("status") or "unknown")
            except Exception as exc:
                real_depth_scale_ok = False
                real_depth_scale_reason = f"read_failed: {exc!r}"

            if not real_depth_scale_ok:
                reference_only_reason = f"real_depth_floor_evidence_failed: {real_depth_scale_reason}"
                mark_alignment_reference_only(alignment_transform_npz, reference_only_reason)
                alignment_decision = read_imu_alignment_decision(alignment_transform_npz)
                configure_reference_only_export(
                    metric_scale_cmd, reference_only_reason, output_dir=egogvae_outputs_dir,
                    head_name=frontend_head_name, hands_name=frontend_hands_name, timestamps=timestamps_txt,
                    hands=hawor_hands_vggt_world, rotation=R_align_npy, alignment=alignment_transform_npz,
                )
                print(f"[Metric scale] {reference_only_reason}; preserve display alignment and input scale; skip body inference.", flush=True)

        if foundationgeo_enabled and not real_depth_scale_ok and reference_only_reason is None:
            run_foundationgeo_metric_scale_if_needed()
            if foundationgeo_scale_json is not None:
                metric_scale_cmd.extend(["--depth-scale-json", str(foundationgeo_scale_json)])

        if real_sensor_has_depth and real_depth_scale_ok:
            metric_scale_cmd.extend(["--depth-scale-json", str(real_depth_scale_json)])

        run(metric_scale_cmd, env=env)
    else:
        shutil.copy2(head_trajectory_tmp, head_trajectory_out)
        shutil.copy2(hamer_outputs_pkl_tmp, hamer_outputs_pkl_out)
        if hand_filter_enabled:
            print("[Hand filter] metric distance skipped: metric scale is disabled", flush=True)
        run([
            py,
            str(SCRIPT_APPLY_ALIGNMENT),
            "--pcd", str(scene_points_ply),
            "--poses", str(camera_poses_txt),
            "--R", str(R_align_npy),
            "--align_meta", str(alignment_transform_npz),
            "--out_pcd", str(final_scene_ply),
            "--out_poses", str(tmp_dir / "head_trajectory_metric_disabled.npy"),
            "--output_voxel_size", str(get_cfg(cfg, "metric_scale.final_scene_output_voxel_size", 0.01)),
        ], env=env)

    # Body inference and exports share the camera_30fps-derived inputs after
    # the existing scene alignment/scale steps; no body-only interpolation.
    pose_validation_enabled = _cfg_bool(get_cfg(cfg, "pose_validation.enabled", True), default=True)
    if reference_only_reason is not None:
        pose_validation_result = {
            "enabled": pose_validation_enabled, "status": "skipped",
            "should_run_egogvae": False, "metric_scale_trusted": False,
            "ground_plane_trusted": False, "reference_only": True,
            "egogvae_skip_reason": reference_only_reason,
        }
        write_summary(pose_validation_report_json, pose_validation_result)
        egoallo_refine_result.update(status="skipped", reason=reference_only_reason)
    elif pose_validation_enabled:
        pose_validation_floor_z = get_cfg(cfg, "pose_validation.floor_z", None)
        if pose_validation_floor_z is None:
            pose_validation_floor_z = get_cfg(cfg, "egogvae.floor_z", None)
        if pose_validation_floor_z is None:
            pose_validation_floor_z = get_cfg(cfg, "metric_scale.floor_z", 0.0)
        pose_validation_cmd = [
            py,
            str(SCRIPT_VALIDATE_POSE_BEFORE_EGOGVAE),
            "--head-trajectory", str(head_trajectory_out),
            "--timestamps-txt", str(timestamps_txt),
            "--hamer-pkl", str(hamer_outputs_pkl_out),
            "--config", str(args.config.expanduser().resolve()),
            "--report-json", str(pose_validation_report_json),
            "--output-npz", str(final_smoothed_npz),
            "--floor-z", str(pose_validation_floor_z),
            "--num-samples", "1",
        ]
        if _cfg_bool(get_cfg(cfg, "pose_validation.empty_npz_on_fail", True), default=True):
            pose_validation_cmd.append("--write-empty-on-fail")
        run(pose_validation_cmd, env=env)
        pose_validation_result = read_json_file(pose_validation_report_json)
        pose_validation_result["enabled"] = True
    else:
        print("[Pose validation] disabled; continue to EgoGVAE inference.", flush=True)

    should_run_egogvae = bool(pose_validation_result.get("should_run_egogvae", True))
    if reference_only_reason is not None:
        should_run_egogvae = False
        pose_validation_result["should_run_egogvae"] = False
        pose_validation_result["egogvae_skip_reason"] = reference_only_reason
        write_empty_egogvae_npz(
            final_smoothed_npz,
            num_samples=int(get_cfg(cfg, "egoallo_refine.num_samples", 1)),
            reason=reference_only_reason,
        )
        # Replace any prior inference output as well as the published final NPZ.
        write_empty_egogvae_npz(
            egogvae_motion_npz,
            num_samples=int(get_cfg(cfg, "egoallo_refine.num_samples", 1)),
            reason=reference_only_reason,
        )
        print(
            "[Floor evidence] wrote empty EgoGVAE output for reference-only alignment: "
            f"{final_smoothed_npz}",
            flush=True,
        )

    if should_run_egogvae:
        egogvae_root = resolve_project_path(get_cfg(cfg, "model_sources.egogvae_root", env.get("EGOGVAE_ROOT")))
        egoallo_root = resolve_project_path(get_cfg(cfg, "model_sources.egoallo_root", "third_party/egoallo"))
        egogvae_checkpoint = resolve_project_path(get_cfg(cfg, "model_weights.egogvae_checkpoint_path", env.get("EGOGVAE_CHECKPOINT_PATH")))
        egoallo_checkpoint_dir = resolve_project_path(
            get_cfg(
                cfg,
                "model_weights.egoallo_checkpoint_dir",
                "third_party/egoallo/egoallo_checkpoint_april13/checkpoints_3000000",
            )
        )
        smplh_npz_path = resolve_project_path(get_cfg(cfg, "model_weights.smplh_npz_path", env.get("SMPLH_NPZ_PATH")))
        if egogvae_root is None:
            raise ValueError("model_sources.egogvae_root is required")
        if egoallo_root is None:
            raise ValueError("model_sources.egoallo_root is required")
        if egogvae_checkpoint is None:
            raise ValueError("model_weights.egogvae_checkpoint_path is required")
        if egoallo_checkpoint_dir is None:
            raise ValueError("model_weights.egoallo_checkpoint_dir is required")
        if smplh_npz_path is None:
            raise ValueError("model_weights.smplh_npz_path is required")
        require_dir(egoallo_root, "egoallo_root")
        require_file(egoallo_checkpoint_dir / "model.safetensors", "egoallo_checkpoint_dir/model.safetensors")

        egogvae_cmd = [
            py,
            str(SCRIPT_RUN_EGOGVAE),
            "--traj-root",
            str(egogvae_inputs_dir),
            "--output-dir",
            str(egogvae_outputs_dir),
            "--output-name",
            egogvae_motion_name,
            "--report-json",
            str(motion_quality_report_json),
            "--egogvae-root",
            str(egogvae_root),
            "--checkpoint-path",
            str(egogvae_checkpoint),
            "--smplh-npz-path",
            str(smplh_npz_path),
            "--device",
            str(get_cfg(cfg, "egogvae.device", get_cfg(cfg, "runtime.device", "cuda"))),
            "--window-size",
            str(get_cfg(cfg, "egogvae.window_size", 128)),
            "--window-stride",
            str(get_cfg(cfg, "egogvae.window_stride", 32)),
            "--start-index",
            str(get_cfg(cfg, "egogvae.start_index", 0)),
            "--floor-z",
            str(get_cfg(cfg, "egogvae.floor_z", get_cfg(cfg, "metric_scale.floor_z", 0.0))),
            "--latent-mode",
            str(get_cfg(cfg, "egogvae.latent_mode", "mean")),
            "--seed",
            str(get_cfg(cfg, "egogvae.seed", 2025)),
            "--window-pos-center",
            str(get_cfg(cfg, "egogvae.window_pos_center", 64.0)),
            "--window-pos-trusted-lo",
            str(get_cfg(cfg, "egogvae.window_pos_trusted_lo", 32.0)),
            "--window-pos-trusted-hi",
            str(get_cfg(cfg, "egogvae.window_pos_trusted_hi", 96.0)),
            "--window-pos-weight",
            str(get_cfg(cfg, "egogvae.window_pos_weight", 1.0)),
            "--contact-confidence-weight",
            str(get_cfg(cfg, "egogvae.contact_confidence_weight", 0.0)),
            "--rotation-weight",
            str(get_cfg(cfg, "egogvae.rotation_weight", 1.0)),
            "--foot-position-weight",
            str(get_cfg(cfg, "egogvae.foot_position_weight", 50.0)),
            "--contact-jump-weight",
            str(get_cfg(cfg, "egogvae.contact_jump_weight", 2.0)),
            "--switch-penalty",
            str(get_cfg(cfg, "egogvae.switch_penalty", 5.0)),
            "--same-frame-rotation-weight",
            str(get_cfg(cfg, "egogvae.same_frame_rotation_weight", 1.0)),
            "--same-frame-foot-weight",
            str(get_cfg(cfg, "egogvae.same_frame_foot_weight", 50.0)),
            "--same-frame-contact-weight",
            str(get_cfg(cfg, "egogvae.same_frame_contact_weight", 2.0)),
            "--switch-rotation-threshold-deg",
            str(get_cfg(cfg, "egogvae.switch_rotation_threshold_deg", 15.0)),
            "--switch-foot-threshold-m",
            str(get_cfg(cfg, "egogvae.switch_foot_threshold_m", 0.12)),
            "--transition-region-min",
            str(get_cfg(cfg, "egogvae.transition_region_min", 16)),
            "--transition-region-max",
            str(get_cfg(cfg, "egogvae.transition_region_max", 32)),
        ]
        end_index = get_cfg(cfg, "egogvae.end_index", None)
        if end_index is not None:
            egogvae_cmd.extend(["--end-index", str(end_index)])
        add_flag(egogvae_cmd, "--allow-tf32", get_cfg(cfg, "egogvae.allow_tf32", True))
        run(egogvae_cmd, env=env)

        if _cfg_bool(get_cfg(cfg, "egoallo_refine.enabled", True), default=True):
            egoallo_refine_cmd = [
                py, str(SCRIPT_RUN_EGOALLO_REFINE),
                "--reference-motion-npz", str(egogvae_motion_npz),
                "--hamer-outputs-path", str(hamer_outputs_pkl_out),
                "--output-dir", str(egogvae_outputs_dir),
                "--report-json", str(egoallo_refine_report_json),
                "--egoallo-root", str(egoallo_root),
                "--checkpoint-dir", str(egoallo_checkpoint_dir),
                "--smplh-npz-path", str(smplh_npz_path),
                "--floor-z", str(get_cfg(cfg, "egogvae.floor_z", get_cfg(cfg, "metric_scale.floor_z", 0.0))),
                "--egoallo-cache-npz", str(egoallo_hawor_cache_npz),
            ]
            refine_defaults = {
                "guidance_mode": "hamer_reproj2", "device": "cuda",
                "num_samples": 1, "random_seed": 2025,
                "lower_body_reference_strength": 1.0, "lower_body_reference_seed": 7819,
                "target_segment_length": 1024, "context_frames": 36,
                "window_size": 128, "overlap_size": 32,
                "hamer_abspos_weight": 20.0, "bucket_t": "auto", "detection_bucket": 0,
                "hand_reachability_fade_ratio": 0.35,
                "hand_reachability_min_weight": 0.05,
                "inner_max_iters": 2, "inner_last_steps": 0, "inner_step_frequency": 1,
                "post_max_iters": 16, "post_last_steps": 0, "post_step_frequency": 1,
                "body_smoothing_max_deg": 5.0, "leg_max_iters": 250, "contact_source": "egogvae",
                "grounding_max_iters": 200, "grounding_spine_max_deg": 8.0,
                "grounding_leg_max_deg": 12.0, "grounding_smoothing_seconds": 0.08,
                "grounding_foot_xy_max_m": 0.05, "grounding_foot_rotation_max_deg": 8.0,
                "leg_max_correction_deg": 3.0,
                "hand_refinement_max_deg": 0.0,
            }
            for name, default in refine_defaults.items():
                egoallo_refine_cmd.extend(["--" + name.replace("_", "-"), str(get_cfg(cfg, "egoallo_refine." + name, default))])
            add_flag(egoallo_refine_cmd, "--verbose", get_cfg(cfg, "egoallo_refine.verbose", False))
            prewarm_guidance = _cfg_bool(
                get_cfg(cfg, "jax_guidance.compilation_cache.prewarm_guidance", True), default=True)
            egoallo_refine_cmd.append("--prewarm-guidance" if prewarm_guidance else "--skip-guidance-prewarm")
            egoallo_refine_cmd.extend(["--guidance-warmup-mode", str(
                get_cfg(cfg, "jax_guidance.compilation_cache.warmup_mode", "compile"))])
            run(egoallo_refine_cmd, env=env)
            egoallo_refine_result = read_json_file(egoallo_refine_report_json)
            egoallo_refine_result["enabled"] = True
        else:
            shutil.copy2(egogvae_motion_npz, final_smoothed_npz)
            egoallo_refine_result = {
                "enabled": False,
                "status": "disabled",
                "output_npz": str(final_smoothed_npz),
            }
    else:
        egogvae_inference_skipped = True
        if reference_only_reason is not None:
            print(
                "[Floor evidence] ground/scale untrusted; skip EgoGVAE inference and keep display exports.",
                flush=True,
            )
        else:
            print(
                "[Pose validation] failed; skip EgoGVAE inference and keep empty output: "
                f"{final_smoothed_npz}",
                flush=True,
            )

    metric_scale_result = None
    if metric_scale_metadata_json.is_file():
        try:
            with metric_scale_metadata_json.open("r", encoding="utf-8") as f:
                metric_scale_result = json.load(f)
            metric_scale_result = _redact_metric_scale_result_for_log(metric_scale_result)
        except Exception as exc:
            metric_scale_result = {"status": "metadata_read_failed", "error": repr(exc)}

    assets = _export_assets(
        output_dir=output_dir,
        cfg=cfg,
        final_scene_ply=final_scene_ply,
        frontend_head_jsonl=frontend_head_jsonl,
        frontend_hands_jsonl=frontend_hands_jsonl,
        objects_egoallo_parquet=objects_egoallo_parquet,
        final_motion_npz=final_smoothed_npz,
        reference_only=reference_only_reason is not None,
    )

    summary = {
        "status": "completed",
        "completion_reason": (
            "imu_fallback_alignment_only_skip_scale_and_body"
            if real_imu_alignment_failed
            else "depth_fallback_alignment_only_skip_scale_and_body" if reference_only_reason else "completed"
        ),
        "reference_only": reference_only_reason is not None,
        "reference_only_reason": reference_only_reason,
        "ground_plane_trusted": False if reference_only_reason else None,
        "metric_scale_trusted": False if reference_only_reason else None,
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(output_dir),
        "tmp_dir": str(tmp_dir),
        "egogvae_inputs_dir": str(egogvae_inputs_dir),
        "egogvae_outputs_dir": str(egogvae_outputs_dir),
        "egogvae_tmp_dir": str(egogvae_tmp_dir),
        "assets": assets,
        "runtime_threads": {
            "preprocess_threads": env.get("EGO_UNI_PREPROCESS_THREADS"),
            "torch_cpu_threads": env.get("EGO_UNI_TORCH_CPU_THREADS"),
            "preprocess_explicit_library_settings_take_precedence": True,
        },
        "python_cache": {
            "pycache_prefix": env.get("PYTHONPYCACHEPREFIX"),
            "cleanup_existing_scripts_pycache": python_pycache_cleanup,
        },
        "jax_guidance": {
            "compilation_cache_enabled": env.get("JAX_ENABLE_COMPILATION_CACHE") == "true",
            "compilation_cache_dir": env.get("JAX_COMPILATION_CACHE_DIR"),
            "compilation_cache_signature_file": env.get("EGOALLO_JAX_CACHE_SIGNATURE_FILE"),
            "compilation_cache_status": env.get("EGOALLO_JAX_CACHE_RESET_REASON"),
            "compilation_cache_reused": env.get("EGOALLO_JAX_CACHE_REUSED") == "1",
            "compilation_cache_regenerated": env.get("EGOALLO_JAX_CACHE_REGENERATED") == "1",
            "min_compile_time_secs": env.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"),
            "min_entry_size_bytes": env.get("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"),
            "enable_xla_caches": env.get("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"),
        },
        "model_sources": {
            "egogvae_root": env.get("EGOGVAE_ROOT"),
            "geocalib_root_fallback": env.get("GEOCALIB_ROOT"),
            "foundationgeo_root_fallback": env.get("FOUNDATIONGEO_ROOT"),
        },
        "model_weights": {
            "egogvae_checkpoint_path": env.get("EGOGVAE_CHECKPOINT_PATH"),
            "egoallo_checkpoint_dir": str(
                resolve_project_path(
                    get_cfg(
                        cfg,
                        "model_weights.egoallo_checkpoint_dir",
                        "third_party/egoallo/egoallo_checkpoint_april13/checkpoints_3000000",
                    )
                )
            ),
            "geocalib_pinhole_tar": env.get("GEOCALIB_PINHOLE_TAR"),
            "mano_left_pkl": str(mano_left),
            "mano_right_pkl": str(mano_right),
            "smplh_npz_path": env.get("SMPLH_NPZ_PATH"),
            "vposer_dir": env.get("VPOSER_DIR"),
            "foundationgeo_model_pt": env.get("FOUNDATIONGEO_MODEL_PT"),
        },
        "foundationgeo_metric_scale": {
            "enabled": bool(foundationgeo_enabled and not real_depth_scale_ok and reference_only_reason is None),
            "output_dir": str(foundationgeo_output_dir) if foundationgeo_output_dir is not None else None,
            "scale_json": str(foundationgeo_scale_json) if foundationgeo_scale_json is not None else None,
        },
        "real_sensor": {
            "enabled": real_sensor_input is not None,
            "input": str(real_sensor_input) if real_sensor_input is not None else None,
            "has_imu": bool(real_sensor_has_imu),
            "has_depth": bool(real_sensor_has_depth),
            "rgb_camera_stream_id": real_sensor_rgb_stream_id,
            "depth_camera_stream_id": real_sensor_depth_stream_id,
            "imu_gravity": str(imu_gravity_txt) if real_sensor_has_imu else None,
            "depth_scale_json": str(real_depth_scale_json)
            if real_sensor_has_depth and not real_imu_alignment_failed else None,
            "depth_scale_ok": bool(real_depth_scale_ok),
            "depth_scale_reason": real_depth_scale_reason,
            "imu_alignment": alignment_decision or None,
        },
        "metric_scale_result": metric_scale_result,
        "pose_validation": pose_validation_result,
        "pose_validation_report": str(pose_validation_report_json) if pose_validation_enabled else None,
        "egogvae_inference_skipped": bool(egogvae_inference_skipped),
        "egogvae_motion_npz": str(egogvae_motion_npz),
        "final_smoothed_npz": str(final_smoothed_npz),
        "motion_quality_report": str(motion_quality_report_json) if not egogvae_inference_skipped else None,
        "egoallo_refine_report": str(egoallo_refine_report_json)
        if bool(egoallo_refine_result.get("enabled")) and not egogvae_inference_skipped
        else None,
        "egoallo_hawor_cache_npz": str(egoallo_hawor_cache_npz)
        if bool(egoallo_refine_result.get("enabled")) and not egogvae_inference_skipped
        else None,
        "egoallo_refine": egoallo_refine_result,
        "egoallo_hand_input": {
            "source": "hands_30fps.parquet",
            "source_parquet": str(upstream.hands_parquet),
            "prepared_jsonl": str(egoallo_hands_jsonl),
            "frontend_source_parquet": str(upstream.hands_parquet),
        },
        "egogvae": {
            "device": str(get_cfg(cfg, "egogvae.device", get_cfg(cfg, "runtime.device", "cuda"))),
            "window_size": int(get_cfg(cfg, "egogvae.window_size", 128)),
            "window_stride": int(get_cfg(cfg, "egogvae.window_stride", 32)),
            "start_index": int(get_cfg(cfg, "egogvae.start_index", 0)),
            "end_index": get_cfg(cfg, "egogvae.end_index", None),
            "latent_mode": str(get_cfg(cfg, "egogvae.latent_mode", "mean")),
            "python": py,
        },
        "inputs": {
            "episode_dir": str(episode_dir),
            "downstream_dir": str(downstream_dir),
            "frames_dir": str(frames_dir),
            "cam_space_dir": str(cam_space_dir),
            "hawor_hands_vggt_world": str(hawor_hands_vggt_world),
            "hawor_to_vggt_alignment": str(hawor_to_vggt_alignment),
            "vggt_head_pose": str(vggt_head_pose),
            "scene_points_ply": str(scene_points_ply),
        },
    }

    # IMPORTANT:
    # Cleanup happens only after the whole pipeline has completed successfully.
    # At this point, the following stages have all finished:
    #   1) convert_to_ego preprocessing
    #   2) GeoCalib / gravity alignment
    #   3) metric scale correction and front-end JSONL export
    #   4) pose validation, which may write an empty final NPZ and skip EgoGVAE
    #   5) EgoGVAE inference into egogvae_outputs_dir, if validation passed
    # If any previous subprocess raises an error, execution will stop before this block,
    # and all intermediate directories will be kept for debugging.
    cleanup_results: dict[str, Any] = {}

    cleanup_results["egogvae_inputs"] = remove_dir_if_requested(
        "egogvae_inputs",
        egogvae_inputs_dir,
        cleanup_flag(cfg, "egogvae_inputs", default=False),
    )
    cleanup_results["egogvae_tmp"] = remove_dir_if_requested(
        "egogvae_tmp",
        egogvae_tmp_dir,
        cleanup_flag(cfg, "egogvae_tmp", default=False),
    )
    cleanup_results["egoallo_outputs"] = remove_dir_if_requested(
        "egoallo_outputs",
        egogvae_outputs_dir,
        cleanup_flag(cfg, "egoallo_outputs", default=False),
    )
    cleanup_results["assets_pointclouds"] = remove_dir_if_requested(
        "assets_pointclouds",
        _assets_pointclouds_dir(output_dir, cfg),
        cleanup_flag(cfg, "assets_pointclouds", default=False),
    )
    cleanup_results["assets_poses"] = remove_dir_if_requested(
        "assets_poses",
        _assets_poses_dir(output_dir, cfg),
        cleanup_flag(cfg, "assets_poses", default=False),
    )

    cleanup_tmp_enabled = cleanup_flag(
        cfg,
        "tmp",
        legacy_key="output_layout.cleanup_tmp",
        default=False,
    )
    cleanup_results["tmp"] = {
        "name": "tmp",
        "path": str(tmp_dir),
        "requested": bool(cleanup_tmp_enabled),
        "deleted": False,
        "reason": "deferred_until_pipeline_log_written" if cleanup_tmp_enabled else "disabled",
    }

    summary["cleanup"] = cleanup_results

    # Write the final JSON log before deleting tmp. If tmp cleanup is enabled,
    # pipeline_run_log.json will be deleted together with tmp immediately after this.
    if PIPELINE_RECORDER is not None:
        PIPELINE_RECORDER.write(status="completed", final_summary=summary)

    if cleanup_tmp_enabled:
        cleanup_results["tmp"] = remove_dir_if_requested("tmp", tmp_dir, True)
        summary["cleanup"] = cleanup_results

    print("\nBodyMesh pipeline completed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if PIPELINE_RECORDER is not None:
            PIPELINE_RECORDER.write(status="failed", error=traceback.format_exc())
        raise
