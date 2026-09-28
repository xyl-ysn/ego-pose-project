#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import sys
import time

TOP_LEVEL_IMPORT_STARTED = time.perf_counter()

from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch

TOP_LEVEL_IMPORT_SEC = time.perf_counter() - TOP_LEVEL_IMPORT_STARTED

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

T_CAMERA_CPF = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EgoGVAE and select a continuity-aware trajectory.")
    parser.add_argument("--traj-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", default="egogvae_motion.npz")
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--egogvae-root", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--smplh-npz-path", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--window-stride", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--floor-z", type=float, default=0.0)
    parser.add_argument("--latent-mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--allow-tf32", action="store_true")

    parser.add_argument("--window-pos-center", type=float, default=64.0)
    parser.add_argument("--window-pos-trusted-lo", type=float, default=32.0)
    parser.add_argument("--window-pos-trusted-hi", type=float, default=96.0)
    parser.add_argument("--window-pos-weight", type=float, default=1.0)
    parser.add_argument("--contact-confidence-weight", type=float, default=0.0)
    parser.add_argument("--rotation-weight", type=float, default=1.0)
    parser.add_argument("--foot-position-weight", type=float, default=50.0)
    parser.add_argument("--contact-jump-weight", type=float, default=2.0)
    parser.add_argument("--switch-penalty", type=float, default=5.0)
    parser.add_argument("--same-frame-rotation-weight", type=float, default=1.0)
    parser.add_argument("--same-frame-foot-weight", type=float, default=50.0)
    parser.add_argument("--same-frame-contact-weight", type=float, default=2.0)
    parser.add_argument("--switch-rotation-threshold-deg", type=float, default=15.0)
    parser.add_argument("--switch-foot-threshold-m", type=float, default=0.12)
    parser.add_argument("--transition-region-min", type=int, default=16)
    parser.add_argument("--transition-region-max", type=int, default=32)
    return parser.parse_args()


def read_timestamps(path: Path, expected: int) -> np.ndarray:
    table = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")
    if getattr(table, "dtype", None) is not None and table.dtype.names and "timestamp" in table.dtype.names:
        values = np.asarray([table["timestamp"]] if table.shape == () else table["timestamp"], dtype=np.float64)
    else:
        values = np.loadtxt(path, dtype=np.float64, ndmin=2)[:, 0]
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} timestamps, expected {expected}")
    return values.reshape(-1)


def add_egogvae_to_path(root: Path) -> None:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"EgoGVAE root not found: {root}")
    sys.path.insert(0, str(root))


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_head_features(T_world_cpf: torch.Tensor, floor_z: float = 0.0) -> torch.Tensor:
    from model import utils_rotation as ru
    from utils.from_egoallo.transforms import SE3, SO3

    if T_world_cpf.ndim != 3 or T_world_cpf.shape[1:] != (4, 4):
        raise ValueError(f"T_world_cpf should be [N,4,4], got {tuple(T_world_cpf.shape)}")
    if T_world_cpf.shape[0] < 2:
        raise ValueError("Need at least 2 head poses for EgoGVAE")

    T_prev = SE3.from_matrix(T_world_cpf[:-1])
    T_next = SE3.from_matrix(T_world_cpf[1:])
    T_delta = (T_prev.inverse() @ T_next).as_matrix()
    delta_r6 = ru.matrix_to_rotation_6d(T_delta[..., :3, :3])
    delta_t = T_delta[..., :3, 3]
    height = T_world_cpf[1:, 2:3, 3] - float(floor_z)

    R_world_cpf = SE3.from_matrix(T_world_cpf[1:]).rotation().wxyz
    forward_cpf = R_world_cpf.new_tensor([0.0, 0.0, 1.0])
    forward_world = SO3(R_world_cpf) @ forward_cpf
    R_canonical_world = SO3.from_z_radians(-torch.arctan2(forward_world[..., 1], forward_world[..., 0]))
    yaw_r6 = ru.matrix_to_rotation_6d((R_canonical_world @ SO3(R_world_cpf)).as_matrix())
    return torch.cat([delta_r6, delta_t, height, yaw_r6], dim=-1).unsqueeze(0)


def forward_test(model: torch.nn.Module, head: torch.Tensor, *, print_shape: bool, latent_mode: str) -> dict[str, torch.Tensor]:
    if latent_mode == "sample":
        if print_shape:
            return model.forward_test(head)
        with redirect_stdout(io.StringIO()):
            return model.forward_test(head)

    from utils.quat import matrix_to_quaternion, rotation_6d_to_matrix

    if print_shape:
        print(head.shape)
    batch, timesteps, _ = head.shape
    slice_token = model.learnable_tokens[:timesteps, ...]
    head_tokens = model.head_tokenizer(head)
    head_tokens = torch.cat((head_tokens, slice_token.expand(batch, -1, -1)), dim=-1)
    head_tokens = model.concat_mixer(head_tokens)
    dist_head, feat_head = model.head_encoder(head_tokens)
    z_head = dist_head.loc
    head_out = model.motion_decoder(model.head_feat_dropout(feat_head), z_head)
    head_betas = model.head_betas_layer(head_out)
    head_contacts = model.head_contacts_layer(head_out)
    head_theta = model.head_theta_layer(head_out)
    pose_quat = matrix_to_quaternion(rotation_6d_to_matrix(head_theta.reshape(batch, timesteps, -1, 6)))
    return {
        "head_betas": head_betas,
        "head_contacts": head_contacts,
        "head_theta": head_theta,
        "head_dist": dist_head,
        "head_z": z_head,
        "pose_quat": pose_quat,
    }


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    from model.network import Head2Motion

    model = Head2Motion(head_dim=16, motion_dim=135, out_dim=126 + 16 + 21).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    return model.eval()


def run_all_windows(
    model: torch.nn.Module,
    head: torch.Tensor,
    window_size: int,
    stride: int,
    latent_mode: str,
) -> dict[int, dict[str, Any]]:
    _, total, _ = head.shape
    if stride <= 0 or window_size <= 0:
        raise ValueError("window_size and stride must be positive")
    if total <= window_size:
        starts = [0]
    else:
        starts = list(range(0, total - window_size + 1, stride))
        if starts[-1] + window_size < total:
            starts.append(total - window_size)
        starts = sorted(set(starts))

    windows: dict[int, dict[str, Any]] = {}
    for ws in starts:
        we = min(ws + window_size, total)
        with torch.inference_mode():
            out = forward_test(model, head[:, ws:we, :], print_shape=False, latent_mode=latent_mode)
        windows[ws] = {
            "start": ws,
            "end": we,
            "pose_quat": out["pose_quat"].detach().cpu().numpy().astype(np.float32),
            "head_contacts": out["head_contacts"].detach().cpu().numpy().astype(np.float32),
            "head_betas": out["head_betas"].detach().cpu().numpy().astype(np.float32),
        }
    return windows


def save_output(
    *,
    output_npz: Path,
    report_json: Path | None,
    smplh_npz_path: Path,
    body_model: Any,
    T_world_cpf: torch.Tensor,
    timestamps: np.ndarray,
    selected: dict[str, Any],
    start_index: int,
    window_size: int,
    stride: int,
    latent_mode: str,
    inference_device: str,
    elapsed_sec: float,
    timings_sec: dict[str, float],
    total_started: float,
) -> None:
    from utils.from_egoallo.fncsmpl_extensions import get_T_world_root_from_cpf_pose
    from utils.from_egoallo.transforms import SE3, SO3

    save_started = time.perf_counter()
    device = body_model.v_template.device
    dtype = body_model.v_template.dtype
    T_world_cpf = T_world_cpf.to(device=device, dtype=dtype)
    body_quats = torch.from_numpy(selected["body_quats"]).unsqueeze(0).to(device=device, dtype=dtype)
    betas = torch.from_numpy(selected["betas"]).unsqueeze(0).to(device=device, dtype=dtype)
    contacts = torch.from_numpy(selected["contacts"]).unsqueeze(0).to(device=device, dtype=dtype)
    identity_hands = torch.zeros((1, body_quats.shape[1], 30, 4), device=device, dtype=dtype)
    identity_hands[..., 0] = 1.0

    posed = body_model.with_shape(betas).with_pose(
        T_world_root=SE3.identity(device, dtype).wxyz_xyz,
        local_quats=torch.cat([body_quats, identity_hands], dim=2),
    )
    Ts_world_cpf = SE3.from_matrix(T_world_cpf[1:]).parameters()
    Ts_world_root = get_T_world_root_from_cpf_pose(posed, Ts_world_cpf)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(body_quats.shape[1])
    frame_nums = np.arange(start_index, start_index + frame_count, dtype=np.int64)
    timestamps_ns = (timestamps[1:] * 1e9).astype(np.int64)

    np.savez(
        output_npz,
        Ts_world_cpf=Ts_world_cpf.detach().cpu().numpy().astype(np.float32),
        Ts_world_root=Ts_world_root.detach().cpu().numpy().astype(np.float32),
        body_quats=body_quats.detach().cpu().numpy().astype(np.float32),
        left_hand_quats=identity_hands[..., :15, :].detach().cpu().numpy().astype(np.float32),
        right_hand_quats=identity_hands[..., 15:, :].detach().cpu().numpy().astype(np.float32),
        betas=betas.detach().cpu().numpy().astype(np.float32),
        contacts=contacts.detach().cpu().numpy().astype(np.float32),
        frame_nums=frame_nums,
        timestamps_ns=timestamps_ns,
        timestamps=timestamps[1:].astype(np.float64),
        fps=np.asarray(1.0 / np.median(np.diff(timestamps)) if timestamps.size > 1 else np.nan, dtype=np.float32),
        egogvae_runtime_sec=np.asarray(elapsed_sec, dtype=np.float32),
        egogvae_window_size=np.asarray(window_size, dtype=np.int64),
        egogvae_window_stride=np.asarray(stride, dtype=np.int64),
        egogvae_latent_mode=np.asarray(latent_mode),
        egogvae_device=np.asarray(inference_device),
        window_start=selected["window_start"].astype(np.int64),
        local_idx=selected["local_idx"].astype(np.int64),
    )
    sync_if_cuda(device)
    timings_sec["save_output_sec"] = time.perf_counter() - save_started
    timings_sec["total_main_sec"] = time.perf_counter() - total_started

    if report_json is not None:
        report = {
            "status": "ok",
            "output_npz": str(output_npz),
            "frames": frame_count,
            "start_index": start_index,
            "end_index": start_index + frame_count,
            "elapsed_sec": round(elapsed_sec, 3),
            "latent_mode": latent_mode,
            "device": inference_device,
            "window_size": window_size,
            "window_stride": stride,
            "num_source_switches": len(selected["switches"]),
            "gate_relaxation": selected["gate_relaxation"],
            "switches": selected["switches"],
            "window_compatibility": selected["compatibility"],
            "smplh_npz_path": str(smplh_npz_path),
            "timings_sec": {key: round(float(value), 3) for key, value in timings_sec.items()},
        }
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    total_started = time.perf_counter()
    timings_sec: dict[str, float] = {"top_level_import_sec": TOP_LEVEL_IMPORT_SEC}
    args = parse_args()
    imports_started = time.perf_counter()
    add_egogvae_to_path(args.egogvae_root)
    sys.path.insert(0, str(SCRIPT_DIR))

    from utils.from_egoallo import fncsmpl
    from utils.from_egoallo.transforms import SE3, SO3
    from egogvae_selection import select_and_blend
    timings_sec["egogvae_import_sec"] = time.perf_counter() - imports_started

    traj_root = args.traj_root.expanduser().resolve()
    head_path = traj_root / "head_trajectory.npy"
    timestamps_path = traj_root / "timestamps.txt"
    if not head_path.is_file():
        raise FileNotFoundError(head_path)
    if not timestamps_path.is_file():
        raise FileNotFoundError(timestamps_path)

    checkpoint_path = resolve_project_path(args.checkpoint_path)
    smplh_npz_path = resolve_project_path(args.smplh_npz_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"EgoGVAE checkpoint not found: {checkpoint_path}")
    if not smplh_npz_path.is_file():
        raise FileNotFoundError(f"SMPL-H npz not found: {smplh_npz_path}")

    runtime_started = time.perf_counter()
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed(int(args.seed))
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    timings_sec["runtime_setup_sec"] = time.perf_counter() - runtime_started
    if device.type == "cuda":
        cuda_started = time.perf_counter()
        torch.empty((1,), device=device)
        sync_if_cuda(device)
        timings_sec["cuda_init_sec"] = time.perf_counter() - cuda_started

    input_started = time.perf_counter()
    T_world_head_np = np.load(head_path).astype(np.float32)
    all_timestamps = read_timestamps(timestamps_path, T_world_head_np.shape[0])
    max_output_frames = T_world_head_np.shape[0] - 1
    start = int(args.start_index)
    end = max_output_frames if args.end_index is None else min(int(args.end_index), max_output_frames)
    if start < 0 or start >= max_output_frames or end <= start:
        raise ValueError(f"Invalid frame range [{start}, {end}) for {max_output_frames} output frames")

    T_world_head = torch.from_numpy(T_world_head_np[start : end + 1]).to(device=device, dtype=torch.float32)
    T_camera_cpf = torch.from_numpy(T_CAMERA_CPF).to(device=device, dtype=torch.float32)
    T_world_cpf = torch.matmul(T_world_head, T_camera_cpf)
    timestamps = all_timestamps[start : end + 1]
    sync_if_cuda(device)
    timings_sec["input_load_sec"] = time.perf_counter() - input_started

    model_started = time.perf_counter()
    model = load_model(checkpoint_path, device)
    sync_if_cuda(device)
    timings_sec["model_load_sec"] = time.perf_counter() - model_started

    feature_started = time.perf_counter()
    head = build_head_features(T_world_cpf, floor_z=float(args.floor_z))
    sync_if_cuda(device)
    timings_sec["feature_build_sec"] = time.perf_counter() - feature_started

    started = time.perf_counter()
    inference_started = time.perf_counter()
    windows = run_all_windows(model, head, int(args.window_size), int(args.window_stride), str(args.latent_mode))
    sync_if_cuda(device)
    timings_sec["window_inference_sec"] = time.perf_counter() - inference_started

    smplh_started = time.perf_counter()
    body_model = fncsmpl.SmplhModel.load(smplh_npz_path)
    timings_sec["smplh_load_sec"] = time.perf_counter() - smplh_started

    params = {
        "window_pos_center": float(args.window_pos_center),
        "window_pos_trusted_lo": float(args.window_pos_trusted_lo),
        "window_pos_trusted_hi": float(args.window_pos_trusted_hi),
        "window_pos_weight": float(args.window_pos_weight),
        "contact_confidence_weight": float(args.contact_confidence_weight),
        "rotation_weight": float(args.rotation_weight),
        "foot_position_weight": float(args.foot_position_weight),
        "contact_jump_weight": float(args.contact_jump_weight),
        "switch_penalty": float(args.switch_penalty),
        "same_frame_rotation_weight": float(args.same_frame_rotation_weight),
        "same_frame_foot_weight": float(args.same_frame_foot_weight),
        "same_frame_contact_weight": float(args.same_frame_contact_weight),
        "switch_rotation_threshold_deg": float(args.switch_rotation_threshold_deg),
        "switch_foot_threshold_m": float(args.switch_foot_threshold_m),
        "transition_region_min": int(args.transition_region_min),
        "transition_region_max": int(args.transition_region_max),
    }
    selection_started = time.perf_counter()
    selected = select_and_blend(windows, body_model, SO3, SE3, params)
    timings_sec["selection_sec"] = time.perf_counter() - selection_started
    elapsed = time.perf_counter() - started

    save_output(
        output_npz=args.output_dir.expanduser().resolve() / args.output_name,
        report_json=args.report_json.expanduser().resolve() if args.report_json else None,
        smplh_npz_path=smplh_npz_path,
        body_model=body_model,
        T_world_cpf=T_world_cpf,
        timestamps=timestamps,
        selected=selected,
        start_index=start,
        window_size=int(args.window_size),
        stride=int(args.window_stride),
        latent_mode=str(args.latent_mode),
        inference_device=str(device),
        elapsed_sec=elapsed,
        timings_sec=timings_sec,
        total_started=total_started,
    )
    timing_text = " ".join(f"{key}={value:.3f}s" for key, value in timings_sec.items())
    print(f"[EgoGVAE timing] frames={selected['body_quats'].shape[0]} windows={len(windows)} {timing_text}", flush=True)
    print(f"Saved EgoGVAE output: {args.output_dir / args.output_name}")


if __name__ == "__main__":
    main()
