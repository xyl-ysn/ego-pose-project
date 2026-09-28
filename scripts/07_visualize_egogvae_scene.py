#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize EgoGVAE body motion over the reconstructed point cloud.")
    parser.add_argument("--motion-npz", required=True, type=Path)
    parser.add_argument("--scene-ply", required=True, type=Path)
    parser.add_argument(
        "--smplh-npz-path",
        default=os.environ.get(
            "SMPLH_NPZ_PATH",
            "../exp/EgoGVAE_release/weights/smplh/neutral/model.npz",
        ),
        type=Path,
    )
    parser.add_argument(
        "--egogvae-root",
        default=os.environ.get("EGOGVAE_ROOT", "../exp/EgoGVAE_release"),
        type=Path,
    )
    parser.add_argument(
        "--egoallo-root",
        default=os.environ.get("EGOALLO_ROOT", "../exp/egoallo"),
        type=Path,
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--hamer-outputs-pkl",
        type=Path,
        help="Optional hamer_outputs.pkl to expose in the Viser 'Show HaMeR hands' toggle.",
    )
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--floor-z", type=float, default=0.0)
    parser.add_argument("--show-joints", action="store_true")
    parser.add_argument("--diagnostic-samples", type=int, default=80)
    parser.add_argument("--objects-parquet", type=Path, default=None)
    parser.add_argument("--object-mesh", type=Path, default=None)
    return parser.parse_args()


def load_points_xyzrgb(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.PointCloud):
        points = np.asarray(loaded.vertices, dtype=np.float32)
        colors = np.asarray(getattr(loaded, "colors", None))
    elif isinstance(loaded, trimesh.Trimesh):
        points = np.asarray(loaded.vertices, dtype=np.float32)
        colors = np.asarray(getattr(loaded.visual, "vertex_colors", None))
    else:
        raise TypeError(f"Unsupported point cloud type from {path}: {type(loaded)!r}")

    if colors.ndim != 2 or colors.shape[0] != points.shape[0] or colors.shape[1] < 3:
        colors = np.full((points.shape[0], 3), 180, dtype=np.uint8)
    else:
        colors = colors[:, :3]
        if colors.dtype.kind == "f" and colors.size and float(np.nanmax(colors)) <= 1.0:
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        else:
            colors = np.clip(colors, 0, 255).astype(np.uint8)

    if max_points > 0 and points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[idx]
        colors = colors[idx]
    return points, colors


def install_optional_plyfile_stub() -> None:
    try:
        import plyfile  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("plyfile")

        class PlyData:
            @staticmethod
            def read(*_args, **_kwargs):
                raise RuntimeError("plyfile is not installed; pass ordinary PLY point clouds via --scene-ply.")

        module.PlyData = PlyData
        sys.modules["plyfile"] = module


def as_sample_time(array: np.ndarray, sample_index: int, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim >= 2 and array.shape[0] > sample_index and name != "Ts_world_cpf":
        return array[sample_index]
    return array


def load_motion(npz_path: Path, sample_index: int, device: torch.device):
    from egoallo.network import EgoDenoiseTraj
    from egoallo.transforms import SO3

    with np.load(npz_path) as data:
        Ts_world_cpf = np.asarray(data["Ts_world_cpf"], dtype=np.float32)
        Ts_world_root = as_sample_time(np.asarray(data["Ts_world_root"], dtype=np.float32), sample_index, "Ts_world_root")
        body_quats = as_sample_time(np.asarray(data["body_quats"], dtype=np.float32), sample_index, "body_quats")
        betas = as_sample_time(np.asarray(data["betas"], dtype=np.float32), sample_index, "betas")
        contacts = as_sample_time(np.asarray(data["contacts"], dtype=np.float32), sample_index, "contacts")
        left = as_sample_time(np.asarray(data["left_hand_quats"], dtype=np.float32), sample_index, "left_hand_quats")
        right = as_sample_time(np.asarray(data["right_hand_quats"], dtype=np.float32), sample_index, "right_hand_quats")
        if "timestamps" in data.files:
            timestamps = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
        elif "timestamps_ns" in data.files:
            timestamps = np.asarray(data["timestamps_ns"], dtype=np.float64).reshape(-1) / 1e9
        else:
            timestamps = None

    if body_quats.ndim != 3 or body_quats.shape[1:] != (21, 4):
        raise ValueError(f"body_quats should be [T,21,4] after sample selection, got {body_quats.shape}")
    timesteps = body_quats.shape[0]
    if Ts_world_cpf.shape != (timesteps, 7):
        raise ValueError(f"Ts_world_cpf should be [{timesteps},7], got {Ts_world_cpf.shape}")
    if Ts_world_root.shape != (timesteps, 7):
        raise ValueError(f"Ts_world_root should be [{timesteps},7], got {Ts_world_root.shape}")
    if betas.ndim == 1:
        betas = np.broadcast_to(betas[None, :], (timesteps, betas.shape[0]))
    if betas.shape != (timesteps, 16):
        raise ValueError(f"betas should be [T,16], got {betas.shape}")
    if contacts.shape != (timesteps, 21):
        contacts = np.zeros((timesteps, 21), dtype=np.float32)
    if timestamps is not None and timestamps.shape != (timesteps,):
        raise ValueError(f"timestamps should have {timesteps} rows, got {timestamps.shape}")

    hand_quats = np.concatenate([left, right], axis=1) if left.shape[:2] == (timesteps, 15) and right.shape[:2] == (timesteps, 15) else None
    if hand_quats is None:
        hand_quats = np.zeros((timesteps, 30, 4), dtype=np.float32)
        hand_quats[..., 0] = 1.0

    body_q = torch.from_numpy(body_quats).to(device=device, dtype=torch.float32).unsqueeze(0)
    hand_q = torch.from_numpy(hand_quats).to(device=device, dtype=torch.float32).unsqueeze(0)
    traj = EgoDenoiseTraj(
        betas=torch.from_numpy(betas).to(device=device, dtype=torch.float32).unsqueeze(0),
        body_rotmats=SO3(body_q).as_matrix(),
        contacts=torch.from_numpy(contacts).to(device=device, dtype=torch.float32).unsqueeze(0),
        hand_rotmats=SO3(hand_q).as_matrix(),
    )
    return (
        torch.from_numpy(Ts_world_cpf).to(device=device, dtype=torch.float32),
        torch.from_numpy(Ts_world_root).to(device=device, dtype=torch.float32).unsqueeze(0),
        traj,
        None if timestamps is None else tuple(float(x) for x in timestamps),
    )


def print_motion_diagnostics(body_model, Ts_world_cpf, Ts_world_root, traj, floor_z: float, max_samples: int) -> None:
    from egoallo.transforms import SO3

    timesteps = int(Ts_world_cpf.shape[0])
    if timesteps <= 0 or max_samples <= 0:
        return
    idx = torch.linspace(0, timesteps - 1, min(max_samples, timesteps), dtype=torch.long)
    mesh_min_z = []
    for frame_idx in idx.tolist():
        local_quats = SO3.from_matrix(
            torch.cat(
                [
                    traj.body_rotmats[:, frame_idx : frame_idx + 1],
                    traj.hand_rotmats[:, frame_idx : frame_idx + 1],
                ],
                dim=2,
            )
        ).wxyz
        posed = body_model.with_shape(traj.betas[:, frame_idx : frame_idx + 1]).with_pose(
            Ts_world_root[:, frame_idx : frame_idx + 1],
            local_quats,
        )
        mesh_min_z.append(float(posed.lbs().verts[..., 2].min().detach().cpu()) - float(floor_z))

    values = np.asarray(mesh_min_z, dtype=np.float64)
    print(
        "[EgoGVAE diagnostic] mesh_min_z_above_floor sampled "
        f"n={values.size}: min={values.min():.3f}m "
        f"p50={np.percentile(values, 50):.3f}m "
        f"p95={np.percentile(values, 95):.3f}m "
        f"max={values.max():.3f}m"
    )


def add_object_assets(server, objects_parquet: Path, object_mesh: Path) -> None:
    import pandas as pd

    objects = pd.read_parquet(objects_parquet)
    scene = trimesh.load(object_mesh, force="scene")

    mesh_parts = []
    for geometry in scene.geometry.values():
        if isinstance(geometry, trimesh.Trimesh):
            mesh_parts.append(geometry)
    if not mesh_parts:
        raise ValueError(f"No triangle mesh found in {object_mesh}")

    count = 0
    for _, row in objects.iterrows():
        matrix = row.get("T_world_from_object")
        if matrix is None:
            continue
        T = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        for part in mesh_parts:
            vertices = (T[:3, :3] @ np.asarray(part.vertices, dtype=np.float64).T).T + T[:3, 3]
            server.scene.add_mesh_simple(
                f"/objects/{row.get('track_id') or 'object'}/{int(row['frame'])}",
                vertices=vertices,
                faces=np.asarray(part.faces),
                color=(210, 140, 70),
                flat_shading=False,
            )
        count += 1
    print(f"[Objects] added {count} object meshes from {objects_parquet} + {object_mesh}")


def main() -> None:
    args = parse_args()
    egogvae_root = resolve_project_path(args.egogvae_root)
    egoallo_root = resolve_project_path(args.egoallo_root)
    smplh_npz = resolve_project_path(args.smplh_npz_path)
    motion_npz = resolve_project_path(args.motion_npz)
    scene_ply = resolve_project_path(args.scene_ply)
    hamer_outputs_pkl = resolve_project_path(args.hamer_outputs_pkl) if args.hamer_outputs_pkl is not None else None
    objects_parquet = resolve_project_path(args.objects_parquet) if args.objects_parquet is not None else None
    object_mesh = resolve_project_path(args.object_mesh) if args.object_mesh is not None else None
    if not egogvae_root.is_dir():
        raise FileNotFoundError(f"EgoGVAE root not found: {egogvae_root}")
    if not (egoallo_root / "src" / "egoallo").is_dir():
        raise FileNotFoundError(f"EgoAllo source not found: {egoallo_root / 'src' / 'egoallo'}")
    if not smplh_npz.is_file():
        raise FileNotFoundError(f"SMPL-H model not found: {smplh_npz}")
    if hamer_outputs_pkl is not None and not hamer_outputs_pkl.is_file():
        raise FileNotFoundError(f"HaMeR outputs not found: {hamer_outputs_pkl}")
    if (objects_parquet is None) != (object_mesh is None):
        raise ValueError("--objects-parquet and --object-mesh must be provided together")
    if objects_parquet is not None and not objects_parquet.is_file():
        raise FileNotFoundError(f"Objects parquet not found: {objects_parquet}")
    if object_mesh is not None and not object_mesh.is_file():
        raise FileNotFoundError(f"Object mesh not found: {object_mesh}")
    sys.path.insert(0, str(egoallo_root / "src"))
    install_optional_plyfile_stub()

    import viser
    from egoallo import fncsmpl
    from egoallo.hand_detection_structs import CorrespondedHamerDetections
    from egoallo.vis_helpers import visualize_traj_and_hand_detections

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    points, _ = load_points_xyzrgb(scene_ply, int(args.max_points))
    Ts_world_cpf, Ts_world_root, traj, timestamps = load_motion(motion_npz, int(args.sample_index), device)
    playback_fps = 30.0
    if timestamps is not None and len(timestamps) > 1:
        intervals = np.diff(timestamps)
        if not np.isfinite(intervals).all() or np.any(intervals <= 0):
            raise ValueError("Motion timestamps must be finite and strictly increasing")
        playback_fps = float(1.0 / np.median(intervals))
    hamer_detections = None
    if hamer_outputs_pkl is not None:
        if timestamps is None:
            raise KeyError("--hamer-outputs-pkl requires timestamps or timestamps_ns in --motion-npz")
        hamer_detections = CorrespondedHamerDetections.load(hamer_outputs_pkl, timestamps).to(device)
    body_model = fncsmpl.SmplhModel.load(smplh_npz).to(device)
    print_motion_diagnostics(body_model, Ts_world_cpf, Ts_world_root, traj, args.floor_z, int(args.diagnostic_samples))

    old_cwd = Path.cwd()
    os.chdir(egogvae_root)
    try:
        server = viser.ViserServer(host=args.host, port=args.port)
        if objects_parquet is not None and object_mesh is not None:
            add_object_assets(server, objects_parquet, object_mesh)
        loop_cb = visualize_traj_and_hand_detections(
            server,
            Ts_world_cpf=Ts_world_cpf,
            traj=traj,
            body_model=body_model,
            Ts_world_root_override=Ts_world_root,
            hamer_detections=hamer_detections,
            points_data=points,
            floor_z=float(args.floor_z),
            show_joints=bool(args.show_joints),
            playback_fps=playback_fps,
        )
        print(f"Open the Viser viewer at http://{args.host}:{server.get_port()}", flush=True)
        while True:
            loop_cb()
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
