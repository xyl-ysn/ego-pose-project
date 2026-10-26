#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from runtime_threads import torch_cpu_threads

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_DEFAULT_JAX_CACHE_DIR = Path(os.path.expanduser("~/.cache/egoallo/jax_compilation_cache")).resolve()


def _bootstrap_jax_compilation_cache() -> Path:
    cache_dir = Path(
        os.getenv(
            "EGOALLO_JAX_COMPILATION_CACHE_DIR",
            os.getenv(
                "JAX_COMPILATION_CACHE_DIR",
                str(_DEFAULT_JAX_CACHE_DIR),
            ),
        )
    ).expanduser().resolve()

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[JAX CACHE] unable to create cache dir {cache_dir!s}: {e}")

    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "true")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
    os.environ.setdefault(
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )

    if os.getenv("EGOALLO_JAX_CACHE_DEBUG", "0") not in {"0", "false", "False"}:
        print(
            "[JAX CACHE] JAX persistent compilation cache configured before import; "
            f"dir={os.environ['JAX_COMPILATION_CACHE_DIR']}",
        )

    return cache_dir


_BOOTSTRAPPED_JAX_CACHE_DIR = _bootstrap_jax_compilation_cache()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FINAL_MOTION_NAME = "merged_smoothed_vis.npz"


@dataclasses.dataclass(frozen=True)
class SegmentTask:
    segment_index: int
    target_start: int
    target_end: int
    infer_start: int
    infer_end: int

    @property
    def target_length(self) -> int:
        return self.target_end - self.target_start

    @property
    def infer_length(self) -> int:
        return self.infer_end - self.infer_start

    @property
    def crop_start(self) -> int:
        return self.target_start - self.infer_start

    @property
    def crop_end(self) -> int:
        return self.crop_start + self.target_length


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_torch_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly")
        # JAX compilation can change the current CUDA device after model loading.
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EgoAllo with a noise-matched EgoGVAE leg reference and bounded leg stabilization.")
    for name in ("reference-motion-npz", "hamer-outputs-path", "output-dir", "egoallo-root", "checkpoint-dir", "smplh-npz-path"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--egoallo-cache-npz", type=Path)
    parser.add_argument("--guidance-mode", choices=("hamer_wrist", "hamer_reproj2"), default="hamer_reproj2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=2025)
    parser.add_argument("--lower-body-reference-strength", type=float, default=1.0)
    parser.add_argument("--lower-body-reference-seed", type=int, default=7819)
    parser.add_argument("--target-segment-length", type=int, default=1024)
    parser.add_argument("--context-frames", type=int, default=36)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--overlap-size", type=int, default=32)
    parser.add_argument("--hamer-abspos-weight", type=float, default=20.0)
    parser.add_argument("--hand-reachability-fade-ratio", type=float, default=0.35,
                        help="Fade spatial hand residuals to their minimum weight over this fraction of arm length beyond reach.")
    parser.add_argument("--hand-reachability-min-weight", type=float, default=0.05,
                        help="Minimum multiplier for existing spatial hand residual weights; must be in (0,1].")
    parser.add_argument("--inner-max-iters", type=int, default=2)
    parser.add_argument("--inner-last-steps", type=int, default=0)
    parser.add_argument("--inner-step-frequency", type=int, default=1)
    parser.add_argument("--post-max-iters", type=int, default=16)
    parser.add_argument("--post-last-steps", type=int, default=0)
    parser.add_argument("--post-step-frequency", type=int, default=1)
    parser.add_argument("--bucket-t", default="auto")
    parser.add_argument("--detection-bucket", type=int, default=0)
    parser.add_argument("--body-smoothing-max-deg", type=float, default=5.0)
    parser.add_argument("--grounding-max-iters", type=int, default=200)
    parser.add_argument("--grounding-spine-max-deg", type=float, default=8.0)
    parser.add_argument("--grounding-leg-max-deg", type=float, default=12.0)
    parser.add_argument("--grounding-smoothing-seconds", type=float, default=0.08)
    parser.add_argument("--grounding-foot-xy-max-m", type=float, default=0.05)
    parser.add_argument("--grounding-foot-rotation-max-deg", type=float, default=8.0)
    parser.add_argument("--leg-max-correction-deg", type=float, default=3.0)
    parser.add_argument("--leg-max-iters", type=int, default=250)
    parser.add_argument("--contact-source", choices=("egoallo", "egogvae"), default="egogvae")
    parser.add_argument("--hand-refinement-max-deg", type=float, default=0.0)
    parser.add_argument("--floor-z", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    warmup = parser.add_mutually_exclusive_group()
    warmup.add_argument("--prewarm-guidance", dest="prewarm_guidance", action="store_true",
                        help="Prepare guidance executables before sampling (default).")
    warmup.add_argument("--skip-guidance-prewarm", dest="prewarm_guidance", action="store_false",
                        help="Skip discarded warmup solves; faster, but downstream numerical equivalence is not established.")
    parser.set_defaults(prewarm_guidance=True)
    parser.add_argument("--guidance-warmup-mode", choices=("compile", "solve"), default="compile",
                        help="compile prepares cached executables; solve also runs discarded legacy solves.")
    return parser.parse_args()


def add_egoallo_paths(root: Path) -> None:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in (root / "src", root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_timestamps(payload: dict[str, np.ndarray], timesteps: int) -> tuple[float, ...]:
    if "timestamps" in payload:
        values = np.asarray(payload["timestamps"], dtype=np.float64)
    elif "timestamps_ns" in payload:
        values = np.asarray(payload["timestamps_ns"], dtype=np.float64) / 1e9
    else:
        raise ValueError("Motion must contain timestamps")
    if values.shape != (timesteps,) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("Timestamps must be finite, strictly increasing and match motion length")
    return tuple(values.tolist())


def build_segment_tasks(timesteps: int, target_segment_length: int, context_frames: int, save_only_center: bool) -> list[SegmentTask]:
    if timesteps <= 0:
        raise ValueError("init motion is empty")
    if target_segment_length <= 0:
        target_segment_length = timesteps
    if context_frames < 0:
        raise ValueError(f"context_frames must be >= 0, got {context_frames}")

    tasks: list[SegmentTask] = []
    start = 0
    while start < timesteps:
        target_start = start
        target_end = min(timesteps, target_start + target_segment_length)
        if save_only_center and context_frames > 0:
            infer_start = max(0, target_start - context_frames)
            infer_end = min(timesteps, target_end + context_frames)
        else:
            infer_start = target_start
            infer_end = target_end
        tasks.append(
            SegmentTask(
                segment_index=len(tasks),
                target_start=target_start,
                target_end=target_end,
                infer_start=infer_start,
                infer_end=infer_end,
            )
        )
        start = target_end
    return tasks


def cpf_sampling_slice(Ts_world_cpf: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if start <= 0:
        return torch.cat([Ts_world_cpf[:1], Ts_world_cpf[start:end]], dim=0)
    return Ts_world_cpf[start - 1 : end]


def build_lower_body_reference(reference, device, seed):
    """Use reference sample zero and one clip-wide noise tensor across segment overlaps."""
    from egoallo import network
    from egoallo.transforms import SO3

    length = len(reference['Ts_world_cpf'])
    tensors = {}
    for key, tail in [('betas', (16,)), ('contacts', (21,)), ('body_quats', (21, 4)),
                      ('left_hand_quats', (15, 4)), ('right_hand_quats', (15, 4))]:
        value = np.asarray(reference[key])
        if value.ndim != len(tail) + 2 or value.shape[0] < 1 or value.shape[1:] != (length, *tail):
            raise ValueError(f'Invalid EgoGVAE reference shape for {key}: {value.shape}')
        if not np.isfinite(value[0]).all():
            raise ValueError(f'Non-finite EgoGVAE reference: {key}')
        if key.endswith('_quats') and np.any(np.linalg.norm(value[0], axis=-1) < 1e-6):
            raise ValueError(f'Zero quaternion in EgoGVAE reference: {key}')
        tensors[key] = torch.as_tensor(value[:1], dtype=torch.float32, device=device)
    trajectory = network.EgoDenoiseTraj(
        betas=tensors['betas'], body_rotmats=SO3(tensors['body_quats']).as_matrix(),
        contacts=tensors['contacts'], hand_rotmats=SO3(torch.cat(
            [tensors['left_hand_quats'], tensors['right_hand_quats']], dim=2)).as_matrix())
    packed = trajectory.pack()
    noise = torch.randn(packed.shape, device=device, dtype=packed.dtype,
                        generator=torch.Generator(device=device).manual_seed(seed))
    return packed, noise


def detection_count(hamer_detections: Any) -> dict[str, int]:
    out = {}
    for side in ("left", "right"):
        concat = getattr(hamer_detections, f"detections_{side}_concat", None)
        out[side] = 0 if concat is None else int(concat.indices.numel())
    return out


def build_hamer_wrist_targets_world(
    hamer_detections: Any,
    Ts_world_cpf: np.ndarray,
    SE3: Any,
) -> tuple[np.ndarray, np.ndarray]:
    timesteps = int(Ts_world_cpf.shape[0])
    targets = np.zeros((timesteps, 2, 3), dtype=np.float32)
    weights = np.zeros((timesteps, 2), dtype=np.float32)
    cpf = torch.from_numpy(np.asarray(Ts_world_cpf, dtype=np.float32))
    T_cpf_cam = SE3(hamer_detections.T_cpf_cam.detach().cpu().to(torch.float32))
    for side_index, side in enumerate(("left", "right")):
        concat = getattr(hamer_detections, f"detections_{side}_concat", None)
        if concat is None:
            continue
        indices = concat.indices.detach().cpu().to(torch.long)
        valid = (indices >= 0) & (indices < timesteps)
        indices = indices[valid]
        if indices.numel() == 0:
            continue
        wrists_cam = concat.keypoints_3d.detach().cpu()[valid, 0, :].to(torch.float32)
        wrists_world = (SE3(cpf[indices]) @ T_cpf_cam) @ wrists_cam
        targets[indices.numpy(), side_index] = wrists_world.numpy(force=True)
        weights[indices.numpy(), side_index] = (
            concat.wrist_pose_weight.detach().cpu()[valid].clamp(0.0, 1.0).numpy(force=True)
        )
    return targets, weights


@torch.no_grad()
def downweight_unreachable_hands(
    detections: Any, reference_joints: torch.Tensor, Ts_world_cpf: np.ndarray,
    fade_ratio: float = 0.35, min_weight: float = 0.05,
) -> tuple[Any, dict[str, Any]]:
    """Use fixed reference shoulders so bending the torso cannot restore weight."""
    from egoallo.transforms import SE3

    if not np.isfinite(fade_ratio) or fade_ratio <= 0:
        raise ValueError("hand-reachability-fade-ratio must be positive and finite")
    if not np.isfinite(min_weight) or not 0 < min_weight <= 1:
        raise ValueError("hand-reachability-min-weight must be finite and in (0,1]")
    if (reference_joints.shape != (len(Ts_world_cpf), 51, 3)
            or not torch.isfinite(reference_joints).all()):
        raise ValueError("Hand reachability requires finite reference joints [T,51,3]")
    for side in ("left", "right"):
        concat = getattr(detections, f"detections_{side}_concat")
        if concat is not None:
            idx = concat.indices
            if bool(((idx < 0) | (idx >= len(Ts_world_cpf))).any()) or len(torch.unique(idx)) != len(idx):
                raise ValueError("Hand observation indices must be unique and within the reference motion")
    targets, _ = build_hamer_wrist_targets_world(detections, Ts_world_cpf, SE3)
    targets = torch.as_tensor(targets, device=reference_joints.device, dtype=reference_joints.dtype)
    updates = {}
    report: dict[str, Any] = dict(reference="egogvae_sample_0", fade_ratio=fade_ratio, min_weight=min_weight)
    for side_index, side in enumerate(("left", "right")):
        concat = getattr(detections, f"detections_{side}_concat")
        if concat is None or not len(concat.indices):
            report[side] = dict(detections=0, downweighted=0, disabled=0, at_min_weight=0)
            continue
        idx = concat.indices.to(reference_joints.device)
        shoulder = reference_joints[idx, 15 + side_index]
        elbow = reference_joints[idx, 17 + side_index]
        wrist = reference_joints[idx, 19 + side_index]
        arm_length = (elbow - shoulder).norm(dim=-1) + (wrist - elbow).norm(dim=-1)
        distance = (targets[idx, side_index] - shoulder).norm(dim=-1)
        if not torch.isfinite(distance).all() or bool((arm_length <= 0).any()):
            raise ValueError(f"Non-finite {side} wrist target or invalid reference arm length")
        excess_ratio = (distance / arm_length - 1).clamp(min=0)
        fraction = (excess_ratio / fade_ratio).clamp(max=1)
        multiplier = min_weight + (1 - min_weight) * (1 - fraction).square() * (1 + 2 * fraction)
        # Both residuals move the full body. Local finger poses remain useful
        # even when an observation's absolute depth is incompatible with reach.
        updates[f"detections_{side}_concat"] = dataclasses.replace(
            concat,
            wrist_pose_weight=concat.wrist_pose_weight * multiplier.to(concat.wrist_pose_weight),
            hand_reproj_weight=concat.hand_reproj_weight * multiplier.to(concat.hand_reproj_weight),
        )
        active = (concat.wrist_pose_weight > 0) | (concat.hand_reproj_weight > 0)
        active = active.to(multiplier.device)
        report[side] = dict(
            detections=len(idx), downweighted=int(((multiplier < 1) & active).sum()),
            disabled=int(((multiplier == 0) & active).sum()),
            at_min_weight=int(((multiplier <= min_weight) & active).sum()),
            median_excess_m=float((distance - arm_length).clamp(min=0).median()),
            median_residual_multiplier=float(multiplier.median()),
        )
    return dataclasses.replace(detections, **updates), report


def resolve_guidance_buckets(args: argparse.Namespace, max_infer: int, max_detections: int) -> tuple[int, int]:
    if str(args.bucket_t).strip().lower() == "auto":
        if args.target_segment_length <= 0:
            raise ValueError("bucket-t=auto requires a positive target-segment-length")
        bucket_t = args.target_segment_length + 2 * args.context_frames
    else:
        bucket_t = int(args.bucket_t)
    if bucket_t < 0 or args.detection_bucket < 0 or args.context_frames < 0:
        raise ValueError("Buckets and context-frames must be nonnegative")
    if bucket_t > 0 and bucket_t < max_infer:
        raise ValueError("bucket-t is smaller than an inference segment")
    # Keep both array axes independent of the current video's observation counts.
    detection_bucket = args.detection_bucket or bucket_t or max_infer
    if max_detections > detection_bucket:
        raise ValueError(f"Need detection-bucket >= {max_detections}; no observations were discarded")
    return bucket_t, detection_bucket


def max_segment_detection_count(all_hamer: Any, tasks: list[SegmentTask]) -> int:
    out = 0
    for task in tasks:
        assert task.infer_start < task.infer_end  # Match CorrespondedHamerDetections.slice.
        for side in ("left", "right"):
            concat = getattr(all_hamer, f"detections_{side}_concat", None)
            if concat is not None:
                indices = concat.indices
                # Match slice's half-open mask without copying vertices, poses,
                # and weights merely to count the retained observations.
                count = torch.count_nonzero((indices >= task.infer_start) & (indices < task.infer_end))
                out = max(out, int(count.item()))
    return out


def slice_hamer_for_segment(all_hamer: Any, task: SegmentTask, device: torch.device) -> Any:
    with torch_cpu_threads():
        segment_hamer = all_hamer.slice(task.infer_start, task.infer_end)
    return segment_hamer.to(device)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return np.asarray(value).tolist()
    except Exception:
        return repr(value)


def set_reproducible_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def guidance_schedule_kwargs(args: argparse.Namespace) -> dict[str, int]:
    kwargs = {}
    for phase in ("inner", "post"):
        iterations = getattr(args, phase + "_max_iters")
        last_steps = getattr(args, phase + "_last_steps")
        frequency = getattr(args, phase + "_step_frequency")
        if iterations < 0 or last_steps < 0 or frequency < 1:
            raise ValueError(f"{phase}: max_iters/last_steps must be >= 0 and step_frequency >= 1")
        kwargs["constraint_optimization_" + phase + "_last_steps"] = last_steps
        kwargs["constraint_optimization_" + phase + "_step_frequency"] = frequency
    return kwargs


def track_jax_cache_events() -> dict[str, int | float]:
    """Observe actual JAX cache events, independently of directory contents."""
    from jax import monitoring

    stats: dict[str, int | float] = dict(requests=0, hits=0, miss_writes=0, load_seconds=0.0)
    names = {
        "/jax/compilation_cache/compile_requests_use_cache": "requests",
        "/jax/compilation_cache/cache_hits": "hits",
        "/jax/compilation_cache/cache_misses": "miss_writes",
    }

    def event(name, **metadata):
        if name in names:
            stats[names[name]] += 1

    def duration(name, seconds, **metadata):
        if name == "/jax/compilation_cache/cache_retrieval_time_sec":
            stats["load_seconds"] += seconds

    monitoring.register_event_listener(event)
    monitoring.register_event_duration_secs_listener(duration)
    return stats


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.lower_body_reference_strength) or not 0 <= args.lower_body_reference_strength <= 1:
        raise ValueError('lower-body-reference-strength must be between 0 and 1')
    schedule_kwargs = guidance_schedule_kwargs(args)
    if args.num_samples < 1 or args.window_size <= args.overlap_size or args.overlap_size < 1:
        raise ValueError("Invalid sample count or sampling window")
    if args.leg_max_iters < 0 or args.body_smoothing_max_deg < 0 or args.hamer_abspos_weight < 0:
        raise ValueError("Optimization budgets and weights must be nonnegative")
    if not np.isfinite(args.hand_reachability_fade_ratio) or args.hand_reachability_fade_ratio <= 0:
        raise ValueError("hand-reachability-fade-ratio must be positive and finite")
    if not np.isfinite(args.hand_reachability_min_weight) or not 0 < args.hand_reachability_min_weight <= 1:
        raise ValueError("hand-reachability-min-weight must be finite and in (0,1]")
    if not np.isfinite(args.leg_max_correction_deg) or not 0 <= args.leg_max_correction_deg <= 10:
        raise ValueError("leg-max-correction-deg must be between 0 and 10")
    if not np.isfinite(args.hand_refinement_max_deg) or not 0 <= args.hand_refinement_max_deg <= 5:
        raise ValueError("hand-refinement-max-deg must be between 0 and 5")
    add_egoallo_paths(resolve_project_path(args.egoallo_root))
    sys.path.insert(0, str(SCRIPT_DIR))
    from egoallo import fncsmpl, fncsmpl_extensions
    from egoallo.guidance_optimizer_jax import JaxGuidanceParams
    from egoallo.hand_detection_structs import CorrespondedHamerDetections
    from egoallo.inference_utils import load_denoiser
    from egoallo.sampling import run_sampling_with_stitching
    from egoallo.transforms import SO3
    from egoallo_stability import (contact_probabilities, reference_contacts, smooth_body, solve_legs,
                                  viewer_motion, ground_below_chest, validate_grounding_options)
    from egoallo_hand_refinement import refine_hands

    jax_cache_events = track_jax_cache_events()
    print("[JAX CACHE] startup_mode=" + (args.guidance_warmup_mode if args.prewarm_guidance else "on_demand")
          + " persistent_cache=" + os.environ.get("JAX_COMPILATION_CACHE_DIR", ""), flush=True)

    validate_grounding_options(args.grounding_spine_max_deg, args.grounding_leg_max_deg,
                               args.grounding_max_iters, args.grounding_smoothing_seconds, floor_z=args.floor_z)
    if (not np.isfinite([args.grounding_foot_xy_max_m, args.grounding_foot_rotation_max_deg]).all()
            or not 0 < args.grounding_foot_xy_max_m <= .05 or not 0 < args.grounding_foot_rotation_max_deg <= 8):
        raise ValueError('Support correction limits must be positive and at most 0.05 m / 8 degrees')
    device = resolve_torch_device(args.device)
    reference_path = resolve_project_path(args.reference_motion_npz)
    with np.load(reference_path, allow_pickle=False) as file:
        reference = {key: file[key] for key in file.files}
    cpf = np.asarray(reference["Ts_world_cpf"], dtype=np.float32)
    if cpf.ndim != 2 or cpf.shape[1] != 7 or len(cpf) < 2 or not np.isfinite(cpf).all():
        raise ValueError("Expected at least two finite CPF poses [T,7]")
    timestamps = load_timestamps(reference, len(cpf))
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch_cpu_threads():
        all_hamer = CorrespondedHamerDetections.load(resolve_project_path(args.hamer_outputs_path), timestamps)
        body_model = fncsmpl.SmplhModel.load(resolve_project_path(args.smplh_npz_path))
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32)
        reference_q = tensor(np.concatenate(
            [reference[key][0] for key in ("body_quats", "left_hand_quats", "right_hand_quats")], axis=1))
        reference_pose = body_model.with_shape(tensor(reference["betas"][0])).with_pose(
            tensor(reference["Ts_world_root"][0]), reference_q)
        all_hamer, hand_reachability = downweight_unreachable_hands(
            all_hamer, reference_pose.Ts_world_joint[..., 4:], cpf,
            args.hand_reachability_fade_ratio, args.hand_reachability_min_weight)
        del reference_pose, reference_q
        net = load_denoiser(resolve_project_path(args.checkpoint_dir))
    print("[EgoAllo] hand reachability " + json.dumps(hand_reachability), flush=True)
    body_model = body_model.to(device)
    net = net.to(device).eval()
    if not net.config.include_hands:
        raise ValueError("Checkpoint must include hands")
    tasks = build_segment_tasks(len(cpf), args.target_segment_length, args.context_frames, True)
    max_infer = max(task.infer_length for task in tasks)
    bucket_t, detection_bucket = resolve_guidance_buckets(
        args, max_infer, max_segment_detection_count(all_hamer, tasks))
    params = {phase: dataclasses.replace(JaxGuidanceParams.defaults(args.guidance_mode, phase),
              hamer_abspos_weight=args.hamer_abspos_weight,
              max_iters=getattr(args, phase + "_max_iters")) for phase in ("inner", "post")}
    schedule = {phase + "_" + name: getattr(args, phase + "_" + name)
                for phase in ("inner", "post") for name in ("max_iters", "last_steps", "step_frequency")}
    cpf_t = torch.as_tensor(cpf, device=device)
    reference_packed, reference_noise = (build_lower_body_reference(reference, device, args.lower_body_reference_seed)
                                        if args.lower_body_reference_strength > 0 else (None, None))
    parts, segment_reports = [], []
    started = time.perf_counter()
    for task in tasks:
        set_reproducible_seed(args.random_seed + task.segment_index)
        print(f"[EgoAllo] segment {task.segment_index + 1}/{len(tasks)} frames={task.infer_length}", flush=True)
        sample_cpf = cpf_sampling_slice(cpf_t, task.infer_start, task.infer_end)
        traj, timing = run_sampling_with_stitching(
            net, body_model=body_model, guidance_mode=args.guidance_mode,
            guidance_post=True, guidance_inner=True, Ts_world_cpf=sample_cpf,
            floor_z=args.floor_z, hamer_detections=slice_hamer_for_segment(all_hamer, task, device),
            aria_detections=None, num_samples=args.num_samples, device=device,
            guidance_verbose=args.verbose, window_size=args.window_size, overlap_size=args.overlap_size,
            guidance_params_inner=params["inner"], guidance_params_post=params["post"],
            constraint_optimization_bucket_t=bucket_t,
            # Prepare the original JAX executables without executing dummy solves.
            constraint_optimization_warmup_before_segments=args.prewarm_guidance,
            constraint_optimization_warmup_mode=args.guidance_warmup_mode,
            constraint_optimization_detection_bucket=detection_bucket, return_timing=True,
            lower_body_reference=None if reference_packed is None else reference_packed[:, task.infer_start:task.infer_end],
            lower_body_reference_noise=None if reference_noise is None else reference_noise[:, task.infer_start:task.infer_end],
            lower_body_reference_strength=args.lower_body_reference_strength,
            **schedule_kwargs)
        with torch.no_grad():
            # Guidance uses the valid segment's mean shape. Export that same body
            # before deriving its world root from the observed CPF trajectory.
            traj = dataclasses.replace(traj, betas=traj.betas.mean(1, keepdim=True).expand_as(traj.betas))
            pose = traj.apply_to_body(body_model)
            root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(pose, sample_cpf[1:])
            body = SO3.from_matrix(traj.body_rotmats).wxyz
            hands = SO3.from_matrix(traj.hand_rotmats).wxyz
            values = dict(body_quats=body, left_hand_quats=hands[:, :, :15], right_hand_quats=hands[:, :, 15:],
                          betas=traj.betas, contacts=traj.contacts, Ts_world_root=root,
                          body_rotmats=traj.body_rotmats, hand_rotmats=traj.hand_rotmats,
                          left_hand_rotmats=traj.hand_rotmats[:, :, :15], right_hand_rotmats=traj.hand_rotmats[:, :, 15:])
            parts.append({key: value[:, task.crop_start:task.crop_end].cpu().numpy() for key, value in values.items()})
        cache_snapshot = dict(jax_cache_events)
        print("[JAX CACHE] source=jax.monitoring " + json.dumps(cache_snapshot), flush=True)
        segment_reports.append(dict(segment=dataclasses.asdict(task), timing=jsonable(timing),
                                    jax_cache_events=cache_snapshot))
    raw = {key: np.concatenate([part[key] for part in parts], axis=1) for key in parts[0]}
    raw.update(Ts_world_cpf=cpf, timestamps=np.asarray(timestamps),
               timestamps_ns=np.asarray(reference.get("timestamps_ns", np.rint(np.asarray(timestamps) * 1e9)), dtype=np.int64),
               frame_nums=np.asarray(reference.get("frame_nums", np.arange(len(cpf))), dtype=np.int64),
               guidance_mode=np.asarray(args.guidance_mode), random_seed=np.asarray(args.random_seed),
               lower_body_reference_strength=np.asarray(args.lower_body_reference_strength),
               lower_body_reference_seed=np.asarray(args.lower_body_reference_seed),
               wrist_position_weight=np.asarray(args.hamer_abspos_weight),
               refinement_mode=np.asarray("egoallo_noise_matched_leg_reference_raw" if reference_packed is not None
                                           else "native_egoallo_soft_guidance_no_fusion_no_extra_hand_ik"))
    raw.update({name: np.asarray(value) for name, value in schedule.items()})
    for key in parts[0]:
        if not np.isfinite(raw[key]).all():
            raise FloatingPointError(f"Non-finite EgoAllo output: {key}")
    raw_path = output_dir / "egoallo_hawor_motion.npz"
    np.savez(raw_path, **raw)
    if args.egoallo_cache_npz is not None:
        cache = resolve_project_path(args.egoallo_cache_npz)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **raw)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    postprocessing_started = time.perf_counter()
    contact_reference = None
    if args.contact_source == "egogvae" and (args.leg_max_iters or args.grounding_max_iters):
        contact_reference = reference_contacts(reference, body_model, device, timestamps, args.floor_z)
    samples, stability_reports, hand_reports, grounding_reports = [], [], [], []
    for index in range(args.num_samples):
        single = dict(raw)
        for key in parts[0]:
            single[key] = raw[key][index:index + 1]
        single = smooth_body(single, body_model, device, args.body_smoothing_max_deg)
        with torch_cpu_threads():
            single = viewer_motion(single)
        support_reference = single
        contact = contact_probabilities(single, device)
        grounding_contact = ((contact - 0.6) / 0.25).clamp(0, 1)
        if contact_reference is not None:
            grounding_contact = torch.maximum(grounding_contact, contact_reference)
            contact = torch.maximum(contact * 0.5, contact_reference)
        if args.grounding_max_iters:
            print(f"[EgoAllo] grounding sample={index} iterations={args.grounding_max_iters}", flush=True)
        single, grounding_report = ground_below_chest(
            single, body_model, device, grounding_contact,
            spine_limit=args.grounding_spine_max_deg, leg_limit=args.grounding_leg_max_deg,
            iterations=args.grounding_max_iters, correction_sigma_seconds=args.grounding_smoothing_seconds,
            floor_z=args.floor_z, foot_xy_limit=args.grounding_foot_xy_max_m,
            foot_angle_limit=args.grounding_foot_rotation_max_deg)
        result, report = solve_legs(single, body_model, device, contact, timestamps, args.floor_z,
                                   args.leg_max_iters, args.leg_max_correction_deg,
                                   support_reference=support_reference, foot_xy_limit=args.grounding_foot_xy_max_m,
                                   foot_angle_limit=args.grounding_foot_rotation_max_deg)
        result, hand_report = refine_hands(result, body_model, all_hamer, device, args.hand_refinement_max_deg)
        samples.append(result)
        stability_reports.append(report)
        hand_reports.append(hand_report)
        grounding_reports.append(grounding_report)
    final = dict(raw)
    for key in parts[0]:
        final[key] = np.concatenate([sample[key] for sample in samples], axis=0)
    final.update(refinement_mode=np.asarray("egoallo_support_guarded_grounding" if args.grounding_max_iters else
                 "egoallo_noise_matched_leg_reference" if reference_packed is not None else "egoallo_full_body_contact_only_legs"),
                 body_smoothing_max_deg=np.asarray(args.body_smoothing_max_deg),
                 grounding_max_iters=np.asarray(args.grounding_max_iters),
                 grounding_spine_max_deg=np.asarray(args.grounding_spine_max_deg),
                 grounding_leg_max_deg=np.asarray(args.grounding_leg_max_deg),
                 grounding_smoothing_seconds=np.asarray(args.grounding_smoothing_seconds),
                 grounding_foot_xy_max_m=np.asarray(args.grounding_foot_xy_max_m),
                 grounding_foot_rotation_max_deg=np.asarray(args.grounding_foot_rotation_max_deg),
                 leg_max_correction_deg=np.asarray(args.leg_max_correction_deg),
                 hand_refinement_max_deg=np.asarray(args.hand_refinement_max_deg),
                 extra_hand_ik=np.asarray(any(r['any_accepted'] for r in hand_reports)),
                 shape_policy=np.asarray("guidance_segment_mean"),
                 egogvae_contact_reference=np.asarray(contact_reference is not None))
    final_path = output_dir / FINAL_MOTION_NAME
    np.savez(final_path, **final)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    postprocessing_elapsed_sec = time.perf_counter() - postprocessing_started
    report = dict(status="ok", output_npz=str(final_path), egoallo_compare_npz=str(raw_path),
                  postprocessing_elapsed_sec=postprocessing_elapsed_sec,
                  reference_motion_npz=str(reference_path), body_source="EgoAllo with noise-matched EgoGVAE leg reference"
                  if reference_packed is not None else "EgoAllo native shape and pose",
                  lower_body_reference_strength=args.lower_body_reference_strength,
                  lower_body_reference_seed=args.lower_body_reference_seed, reference_sample_index=0,
                  shape_policy="guidance_segment_mean",
                  guidance_weights={phase: dataclasses.asdict(value) for phase, value in params.items()},
                  hand_reachability=hand_reachability,
                  guidance_schedule=schedule,
                  guidance_dummy_prewarm=args.prewarm_guidance and args.guidance_warmup_mode == "solve",
                  guidance_warmup_mode=args.guidance_warmup_mode if args.prewarm_guidance else "off",
                  jax_cache_events=dict(jax_cache_events),
                  guidance_buckets=dict(bucket_t=bucket_t, detection_bucket=detection_bucket),
                  body_smoothing_max_deg=args.body_smoothing_max_deg, contact_source=args.contact_source,
                  reference_contact_shape="saved_per_frame",
                  grounding_foot_xy_max_m=args.grounding_foot_xy_max_m,
                  grounding_foot_rotation_max_deg=args.grounding_foot_rotation_max_deg,
                  leg_max_correction_deg=args.leg_max_correction_deg,
                  extra_hand_ik=any(r['any_accepted'] for r in hand_reports),
                  hand_refinement=hand_reports, copied_egogvae_pose=False, forced_egogvae_shape=False,
                  grounding=grounding_reports, leg_stability=stability_reports,
                  segments=segment_reports, detections=detection_count(all_hamer),
                  output_frames=len(cpf), elapsed_sec=time.perf_counter() - started)
    report_path = resolve_project_path(args.report_json) if args.report_json else output_dir / "egoallo_refine_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[EgoAllo] saved {final_path}", flush=True)


if __name__ == "__main__":
    main()
