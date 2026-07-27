#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import inspect
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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

# 本脚本固定放在 project/scripts/05_run_egoallo_from_head.py。
# 优先尝试从源码路径导入 egoallo；仅当源码导入失败时才回退到 conda/site-packages。
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _env_path(name: str, default: Path | str | None = None) -> Path | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        value = default
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _normalize_checkpoint_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    if (path / "model.safetensors").is_file():
        return path
    candidate = path / "checkpoints_3000000"
    if (candidate / "model.safetensors").is_file():
        return candidate
    return path


# 可选：优先使用 EGOALLO_ROOT 对应源码目录作为主逻辑；仅当源码导入失败时，再回退到 conda/site-packages。
EGOALLO_REPO = _env_path("EGOALLO_ROOT")
EGOALLO_CHECKPOINT_DIR = _normalize_checkpoint_dir(_env_path("EGOALLO_CHECKPOINT_DIR"))
SMPLH_NPZ_PATH = _env_path("SMPLH_NPZ_PATH", _env_path("EGOALLO_MODEL_NPZ"))
EGOALLO_MODEL_CONFIG_PATH = _env_path("EGOALLO_MODEL_CONFIG_PATH")
EGOALLO_EGO_CONFIG_PATH = _env_path("EGOALLO_EGO_CONFIG_PATH")
EGOALLO_SOURCE_LABEL = "conda/site-packages"

import numpy as np
import torch
import yaml


def _import_egoallo_modules():
    from egoallo import __file__ as _egoallo_file
    from egoallo import fncsmpl as _fncsmpl
    from egoallo import fncsmpl_extensions as _fncsmpl_extensions
    from egoallo.guidance_optimizer_jax import GuidanceMode as _GuidanceMode
    from egoallo.hand_detection_structs import CorrespondedHamerDetections as _CorrespondedHamerDetections
    from egoallo.sampling import run_sampling_with_stitching as _run_sampling_with_stitching
    from egoallo.transforms import SE3 as _SE3
    from egoallo.transforms import SO3 as _SO3
    return (
        _egoallo_file,
        _fncsmpl,
        _fncsmpl_extensions,
        _GuidanceMode,
        _CorrespondedHamerDetections,
        _run_sampling_with_stitching,
        _SE3,
        _SO3,
    )


def _remove_from_sys_path(path: str) -> None:
    while True:
        try:
            idx = sys.path.index(path)
        except ValueError:
            return
        del sys.path[idx]


def _clear_egoallo_modules() -> None:
    for name in list(sys.modules):
        if name == "egoallo" or name.startswith("egoallo."):
            del sys.modules[name]


def _egoallo_import_candidate_paths(repo: Path) -> list[str]:
    paths: list[str] = []
    for candidate in [repo / "src", repo]:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in paths:
                paths.append(candidate_str)
    return paths


def _module_under_path(module_file: str | None, candidate_path: str) -> bool:
    if not module_file:
        return False
    try:
        module_real = Path(module_file).resolve()
        candidate_real = Path(candidate_path).resolve()
        return str(module_real).startswith(str(candidate_real))
    except Exception:
        return False


def _load_egoallo_modules_with_fallback():
    global EGOALLO_SOURCE_LABEL
    source_exc: Exception | None = None
    source_label: str = "conda/site-packages"
    if EGOALLO_REPO is not None:
        if not EGOALLO_REPO.exists():
            raise FileNotFoundError(f"EGOALLO_ROOT does not exist: {EGOALLO_REPO}")
        for candidate in _egoallo_import_candidate_paths(EGOALLO_REPO):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            try:
                modules = _import_egoallo_modules()
                egoallo_file = modules[0]
                if _module_under_path(egoallo_file, candidate):
                    EGOALLO_SOURCE_LABEL = f"source {EGOALLO_REPO} (via {candidate})"
                    return modules
                raise RuntimeError(
                    f"Imported egoallo from {egoallo_file}, not under expected candidate path {candidate}"
                )
            except Exception as source_error:
                source_exc = source_error
                source_label = f"source {EGOALLO_REPO} (via {candidate})"
                _remove_from_sys_path(candidate)
            _clear_egoallo_modules()

    try:
        modules = _import_egoallo_modules()
        EGOALLO_SOURCE_LABEL = "conda/site-packages"
        return modules
    except Exception as second_exc:
        if source_exc is None:
            raise ModuleNotFoundError(
                "Cannot import egoallo from current conda/site-packages environment."
            ) from second_exc
        raise ModuleNotFoundError(
            f"Cannot import egoallo from {source_label} and conda/site-packages."
        ) from source_exc


(
    _egoallo_file,
    fncsmpl,
    fncsmpl_extensions,
    GuidanceMode,
    CorrespondedHamerDetections,
    run_sampling_with_stitching,
    SE3,
    SO3,
) = _load_egoallo_modules_with_fallback()

import egoallo
import egoallo.fncsmpl_jax as fncsmpl_jax
import jax


class GuidanceRuntime:
    def __init__(self, body_model):
        self.jax_body_model = fncsmpl_jax.SmplhModel(
            faces=jax.device_put(body_model.faces.cpu().numpy()),
            J_regressor=jax.device_put(body_model.J_regressor.cpu().numpy()),
            parent_indices=jax.device_put(np.asarray(body_model.parent_indices)),
            weights=jax.device_put(body_model.weights.cpu().numpy()),
            posedirs=jax.device_put(body_model.posedirs.cpu().numpy()),
            v_template=jax.device_put(body_model.v_template.cpu().numpy()),
            shapedirs=jax.device_put(body_model.shapedirs.cpu().numpy()),
        )


@dataclasses.dataclass
class Args:
    traj_root: Path
    """Custom trajectory directory containing head_trajectory.npy and timestamps.npy/timestamps.txt."""

    script_dir: Path = SCRIPT_DIR
    project_root: Path = PROJECT_ROOT

    checkpoint_dir: Path | None = EGOALLO_CHECKPOINT_DIR
    smplh_npz_path: Path | None = SMPLH_NPZ_PATH

    output_dir: Path | None = None
    """Directory for EgoAllo segment files (.npz and _args.yaml). In this pipeline it is <output_dir>/ego_tmp."""

    hamer_outputs_path: Path | None = None

    glasses_x_angle_offset: float = 0.0
    traj_length: int = 128
    """Target output segment length. With context_frames>0, each inference window is longer, but only the center target range is saved."""

    context_frames: int = 0
    """Extra frames before/after each target segment used only as inference context."""

    save_only_center: bool = True
    """When context_frames>0, save only the target center crop and discard context frames."""

    start_index: int = 0
    end_index: int | None = None
    segment_index_offset: int = 0
    num_samples: int = 1

    guidance_mode: GuidanceMode = "no_hands"
    guidance_inner: bool = False
    guidance_post: bool = True

    floor_z: float = 0.0
    model_config_path: Path | None = EGOALLO_MODEL_CONFIG_PATH
    ego_config_path: Path | None = EGOALLO_EGO_CONFIG_PATH
    ego_compile_model: bool = False

    save_traj: bool = True
    save_args: bool = True
    empty_cache_each_segment: bool = False
    allow_tf32: bool = False

    expected_head_height_min: float = 1.0
    expected_head_height_max: float = 2.2


T_CAMERA_CPF = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def load_custom_transforms(traj_root: Path, device: torch.device):
    head_path = traj_root / "head_trajectory.npy"
    timestamps_npy_path = traj_root / "timestamps.npy"
    timestamps_txt_path = traj_root / "timestamps.txt"

    if not head_path.exists():
        raise FileNotFoundError(head_path)

    T_np = np.load(head_path).astype(np.float32)

    if timestamps_npy_path.exists():
        timestamps_np = np.load(timestamps_npy_path).astype(np.float64)
    elif timestamps_txt_path.exists():
        table = np.genfromtxt(
            timestamps_txt_path,
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        if table.shape == ():
            timestamps_np = np.array([table["timestamp"]], dtype=np.float64)
        else:
            timestamps_np = np.asarray(table["timestamp"], dtype=np.float64)
    else:
        raise FileNotFoundError(
            f"Expected either {timestamps_npy_path} or {timestamps_txt_path}"
        )

    if T_np.ndim != 3 or T_np.shape[1:] != (4, 4):
        raise ValueError(f"head_trajectory.npy should be [N,4,4], got {T_np.shape}")
    if len(T_np) != len(timestamps_np):
        raise ValueError(f"num poses != num timestamps: {len(T_np)} vs {len(timestamps_np)}")

    T_torch = torch.from_numpy(T_np).to(device)
    timestamps_torch = torch.from_numpy(timestamps_np.astype(np.float32)).to(device)
    return T_torch, timestamps_torch, timestamps_np


def load_denoiser_with_config(
    checkpoint_dir: Path,
    model_config_path: Path | None = EGOALLO_MODEL_CONFIG_PATH,
    compile_model: bool = False,
):
    from egoallo.inference_utils import load_denoiser as original_load_denoiser

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if model_config_path is None:
        model_config_path = checkpoint_dir.parent / "model_config.yaml"
    else:
        model_config_path = Path(model_config_path).expanduser().resolve()

    model_weight_path = checkpoint_dir / "model.safetensors"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"EgoAllo checkpoint dir not found: {checkpoint_dir}")
    if not model_weight_path.is_file():
        raise FileNotFoundError(f"EgoAllo weight not found: {model_weight_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    supported_params = set(inspect.signature(original_load_denoiser).parameters.keys())
    call_kwargs = {}
    if "model_config_path" in supported_params:
        call_kwargs["model_config_path"] = model_config_path
    if "compile_model" in supported_params:
        call_kwargs["compile_model"] = compile_model

    default_model_config = checkpoint_dir.parent / "model_config.yaml"
    restore_backup = None
    replaced_default_config = False
    if model_config_path != default_model_config and "model_config_path" not in supported_params:
        restore_backup = default_model_config.with_suffix(default_model_config.suffix + ".egoop_backup")
        if restore_backup.exists():
            raise FileExistsError(f"Cannot temporarily replace model config; backup path already exists: {restore_backup}")

        if default_model_config.exists():
            shutil.move(default_model_config, restore_backup)
            restore_backup = restore_backup
            replaced_default_config = True
        else:
            restore_backup = None

        shutil.copy2(model_config_path, default_model_config)
        replaced_default_config = True

    try:
        return original_load_denoiser(checkpoint_dir, **call_kwargs)
    finally:
        if "model_config_path" not in supported_params and replaced_default_config:
            if default_model_config.exists():
                default_model_config.unlink()
            if restore_backup is not None:
                restore_backup_path = default_model_config.with_suffix(default_model_config.suffix + ".egoop_backup")
                if restore_backup_path.exists():
                    shutil.move(restore_backup_path, default_model_config)


def print_head_height_stats_once(T_world_cpf: torch.Tensor, floor_z: float, expected_min: float, expected_max: float) -> None:
    heights = T_world_cpf[:, 2, 3].detach().cpu().numpy() - float(floor_z)
    if heights.size == 0:
        return
    h_min = float(np.min(heights))
    h_med = float(np.median(heights))
    h_max = float(np.max(heights))
    print("\n[CPF/head height check: full sequence]")
    print(f"  floor_z: {float(floor_z):.6f}")
    print(f"  height above floor min/median/max: {h_min:.6f} / {h_med:.6f} / {h_max:.6f}")
    if h_med < expected_min or h_med > expected_max:
        print(
            "  [Warning] Median head height is outside the expected range "
            f"[{expected_min:.2f}, {expected_max:.2f}] meters."
        )


def maybe_slice_hamer(all_hamer, hamer_path: Path, pose_timesteps: torch.Tensor, start: int, length: int, device: torch.device):
    # Preferred path: load HaMeR once for the whole sequence on CPU and slice per segment,
    # then move only the current segment to CUDA. This preserves speed while avoiding
    # keeping all detections in GPU memory for long videos.
    if all_hamer is not None and hasattr(all_hamer, "slice"):
        return all_hamer.slice(start, start + length).to(device)

    # Fallback for older/custom EgoAllo structs without .slice(). This still works,
    # but loses the main HaMeR-loading speedup.
    pose_timesteps_np = pose_timesteps.detach().cpu().numpy()
    segment_ts = tuple(float(x) for x in pose_timesteps_np[start + 1 : start + length + 1])
    return CorrespondedHamerDetections.load(hamer_path, segment_ts).to(device)


def load_ego_config(config_path: Path | None) -> dict[str, Any]:
    defaults = {
        "sampling": {
            "num_steps": 30,
            "schedule": "quadratic",
            "window_size": 128,
            "overlap_size": 32,
            "keep_intermediate_states": False,
            "cache_clear_every_steps": 0,
            "use_expanded_window_tiles": False,
        },
        "guidance": {
            "inner_repeat": 1,
            "post_repeat": 1,
            "early_stop_grad_norm": None,
            "inner": {},
            "post": {},
        },
        "constraint_optimization": {
            "enabled": True,
            "inner_max_iters": None,
            "post_max_iters": None,
            "inner_last_steps": 0,
            "inner_step_frequency": 1,
            "post_last_steps": 0,
            "post_step_frequency": 1,
            "post_timing_repeat": 0,
            "bucket_t": 0,
            "detection_bucket": 0,
            "warmup_before_segments": False,
            "enable_hand_constraints": True,
            "enable_foot_constraints": False,
            "enable_collision_constraints": False,
            "enable_body_regularization": False,
        },
        "inference": {
            "use_torch_compile": False,
        },
        "denoiser": {
            "compile_model": False,
        },
    }
    if config_path is None:
        raise ValueError(
            "EGOALLO_EGO_CONFIG_PATH is required. Please set it in env or pass --ego-config-path."
        )
    if not config_path.exists():
        raise FileNotFoundError(f"Ego config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        return defaults
    if not isinstance(raw, dict):
        raise ValueError(f"Ego config must be a YAML mapping: {config_path}")

    merged = defaults.copy()
    # body_mesh.yaml passes the EgoAllo config via:
    #   config:
    #     egoallo:
    #       config:
    #         ...
    # Keep compatibility with legacy direct egoallo config by falling back to root-level sections.
    direct_block = raw
    config_block = raw.get("config")
    if isinstance(config_block, dict):
        direct_block = config_block

    legacy_egoallo_cfg = direct_block.get("egoallo")
    if isinstance(legacy_egoallo_cfg, dict):
        nested_block = legacy_egoallo_cfg.get("config")
        if isinstance(nested_block, dict):
            direct_block = nested_block

    for section, section_defaults in defaults.items():
        section_values = direct_block.get(section, {})
        if isinstance(section_values, dict):
            merged[section] = {**section_defaults, **section_values}
        else:
            raise ValueError(f"Section '{section}' must be a mapping in {config_path}")
    return merged


def _plain_namespace(ns: Any) -> Any:
    if isinstance(ns, SimpleNamespace):
        return {k: _plain_namespace(v) for k, v in vars(ns).items()}
    if isinstance(ns, dict):
        return {str(k): _plain_namespace(v) for k, v in ns.items()}
    if isinstance(ns, list):
        return [_plain_namespace(v) for v in ns]
    return ns


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Expected int or null, got {value!r}") from e
    return value_int


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Expected float or null, got {value!r}") from e
    return value_float


def _extract_debug_timesteps(num_steps: int, schedule: str) -> Any:
    num_steps_i = int(num_steps)
    if schedule != "quadratic":
        return np.arange(max(0, num_steps_i), dtype=np.int64)

    ts_builder = getattr(run_sampling_with_stitching, "__globals__", {}).get("quadratic_ts")
    if callable(ts_builder):
        try:
            return np.asarray(ts_builder(num_steps_i))
        except Exception:
            pass

    if num_steps_i <= 0:
        return np.array([], dtype=np.int64)
    if num_steps_i == 1:
        return np.array([1], dtype=np.int64)
    return np.arange(num_steps_i - 1, -1, -1, dtype=np.int64)


@contextmanager
def timer(name: str):
    yield


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Expected int, got {value!r}") from e


def _coerce_guidance_bucket_t(
    value: Any,
    target_segment_length: int,
    context_frames: int,
) -> int:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return int(target_segment_length) + 2 * int(context_frames)
    return _coerce_int(value, 0)


def block_until_ready(tree: Any) -> Any:
    if tree is None:
        return tree
    if hasattr(tree, "block_until_ready"):
        return tree.block_until_ready()

    try:
        jax_tree_fn = getattr(jax, "tree", None)
        if hasattr(jax_tree_fn, "map"):
            return jax.tree.map(block_until_ready, tree)
        if hasattr(jax, "tree_map"):
            return jax.tree_map(block_until_ready, tree)
    except Exception:
        pass
    return tree


def _build_effective_ego_config_view(
    ego_config: dict[str, Any],
    target_segment_length: int = 0,
    context_frames: int = 0,
):
    sampling_cfg = ego_config.get("sampling", {})
    if not isinstance(sampling_cfg, dict):
        sampling_cfg = {}
    guidance_cfg = ego_config.get("guidance", {})
    if not isinstance(guidance_cfg, dict):
        guidance_cfg = {}
    guidance_inner_cfg = guidance_cfg.get("inner", {})
    if not isinstance(guidance_inner_cfg, dict):
        guidance_inner_cfg = {}
    guidance_post_cfg = guidance_cfg.get("post", {})
    if not isinstance(guidance_post_cfg, dict):
        guidance_post_cfg = {}
    constraint_cfg = ego_config.get("constraint_optimization", {})
    if not isinstance(constraint_cfg, dict):
        constraint_cfg = {}
    denoiser_cfg = ego_config.get("denoiser", {})
    if not isinstance(denoiser_cfg, dict):
        denoiser_cfg = {}
    inference_cfg = ego_config.get("inference", {})
    if not isinstance(inference_cfg, dict):
        inference_cfg = {}

    return SimpleNamespace(
        sampling=SimpleNamespace(
            num_steps=_coerce_int(sampling_cfg.get("num_steps"), 30),
            window_size=_coerce_int(sampling_cfg.get("window_size"), 128),
            overlap_size=_coerce_int(sampling_cfg.get("overlap_size"), 32),
            cache_clear_every_steps=_coerce_int(
                sampling_cfg.get("cache_clear_every_steps"), 0
            ),
            use_expanded_window_tiles=bool(
                sampling_cfg.get("use_expanded_window_tiles", False)
            ),
            keep_intermediate_states=bool(
                sampling_cfg.get("keep_intermediate_states", False)
            ),
        ),
        guidance=SimpleNamespace(
            inner=SimpleNamespace(
                max_iters=_coerce_optional_int(guidance_inner_cfg.get("max_iters"))
            ),
            post=SimpleNamespace(
                max_iters=_coerce_optional_int(guidance_post_cfg.get("max_iters"))
            ),
            inner_repeat=_coerce_int(guidance_cfg.get("inner_repeat"), 1),
            post_repeat=_coerce_int(guidance_cfg.get("post_repeat"), 1),
            early_stop_grad_norm=guidance_cfg.get("early_stop_grad_norm", None),
        ),
        constraint_optimization=SimpleNamespace(
            enabled=bool(constraint_cfg.get("enabled", True)),
            inner_max_iters=_coerce_optional_int(
                constraint_cfg.get("inner_max_iters")
            ),
            post_max_iters=_coerce_optional_int(
                constraint_cfg.get("post_max_iters")
            ),
            inner_last_steps=_coerce_int(
                constraint_cfg.get("inner_last_steps"), 0
            ),
            inner_step_frequency=_coerce_int(
                constraint_cfg.get("inner_step_frequency"), 1
            ),
            post_last_steps=_coerce_int(
                constraint_cfg.get("post_last_steps"), 0
            ),
            post_step_frequency=_coerce_int(
                constraint_cfg.get("post_step_frequency"), 1
            ),
            post_timing_repeat=_coerce_int(
                constraint_cfg.get("post_timing_repeat"), 0
            ),
            bucket_t=_coerce_guidance_bucket_t(
                constraint_cfg.get("bucket_t"), target_segment_length, context_frames
            ),
            detection_bucket=_coerce_int(
                constraint_cfg.get("detection_bucket"), 0
            ),
            warmup_before_segments=bool(
                constraint_cfg.get("warmup_before_segments", False)
            ),
            enable_hand_constraints=bool(
                constraint_cfg.get("enable_hand_constraints", True)
            ),
            enable_foot_constraints=bool(
                constraint_cfg.get("enable_foot_constraints", False)
            ),
            enable_collision_constraints=bool(
                constraint_cfg.get("enable_collision_constraints", False)
            ),
            enable_body_regularization=bool(
                constraint_cfg.get("enable_body_regularization", False)
            ),
        ),
        inference=SimpleNamespace(
            use_torch_compile=bool(
                inference_cfg.get("use_torch_compile", denoiser_cfg.get("compile_model", False))
            )
        ),
    )


def _write_ego_runtime_report(
    output_dir: Path,
    args: Args,
    ego_config_path: Path | None,
    ego_config_raw: dict[str, Any],
    ego_config_effective: SimpleNamespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "egoallo_runtime_ego_config.json"
    resolved_model_config = None
    if args.model_config_path is not None:
        resolved_model_config = str(Path(args.model_config_path).expanduser().resolve())
    report = {
        "resolved_paths": {
            "ego_root_repo": str(EGOALLO_REPO) if EGOALLO_REPO is not None else None,
            "egoallo_source": EGOALLO_SOURCE_LABEL,
            "ego_config_path": str(ego_config_path) if ego_config_path is not None else None,
            "model_config_path": resolved_model_config,
            "project_root": str(args.project_root),
            "output_dir": str(output_dir),
        },
        "environment": {
            "EGOALLO_ROOT": os.getenv("EGOALLO_ROOT"),
            "EGOALLO_EGO_CONFIG_PATH": os.getenv("EGOALLO_EGO_CONFIG_PATH"),
            "EGOALLO_MODEL_CONFIG_PATH": os.getenv("EGOALLO_MODEL_CONFIG_PATH"),
            "EGOALLO_CHECKPOINT_DIR": os.getenv("EGOALLO_CHECKPOINT_DIR"),
        },
        "ego_config_raw": _plain_namespace(ego_config_raw),
        "ego_config_used": _plain_namespace(ego_config_effective),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


@dataclasses.dataclass(frozen=True)
class SegmentTask:
    segment_index: int
    target_start: int
    target_end: int
    infer_start: int
    infer_end: int
    crop_offset: int

    @property
    def target_length(self) -> int:
        return int(self.target_end - self.target_start)

    @property
    def infer_length(self) -> int:
        return int(self.infer_end - self.infer_start)


def build_segment_tasks(args: Args, max_output_frames: int) -> tuple[list[SegmentTask], int]:
    final_end = max_output_frames if args.end_index is None else min(args.end_index, max_output_frames)
    if final_end <= args.start_index:
        raise ValueError(f"end_index must be > start_index, got {final_end} <= {args.start_index}")

    tasks: list[SegmentTask] = []
    segment_index = int(args.segment_index_offset)
    start = int(args.start_index)
    while start < final_end:
        target_start = start
        target_length = min(int(args.traj_length), final_end - target_start)
        target_end = target_start + target_length

        if args.save_only_center and args.context_frames > 0:
            infer_start = max(args.start_index, target_start - args.context_frames)
            infer_end = min(final_end, target_end + args.context_frames)
        else:
            infer_start = target_start
            infer_end = target_end

        tasks.append(SegmentTask(
            segment_index=segment_index,
            target_start=target_start,
            target_end=target_end,
            infer_start=infer_start,
            infer_end=infer_end,
            crop_offset=target_start - infer_start,
        ))

        start = target_end
        segment_index += 1

    return tasks, final_end


def _slice_time_tensor(x: torch.Tensor, start: int, end: int, infer_len: int) -> torch.Tensor:
    """Slice a tensor along its time axis when that axis is identifiable."""
    if x.ndim >= 2 and x.shape[1] == infer_len:
        return x[:, start:end, ...]
    if x.ndim >= 1 and x.shape[0] == infer_len:
        return x[start:end, ...]
    return x


def _slice_time_numpy(x: np.ndarray, start: int, end: int, infer_len: int) -> np.ndarray:
    if x.ndim >= 2 and x.shape[1] == infer_len:
        return x[:, start:end, ...]
    if x.ndim >= 1 and x.shape[0] == infer_len:
        return x[start:end, ...]
    return x


@dataclasses.dataclass
class EgoAlloRuntime:
    args: Args
    device: torch.device
    T_world_head: torch.Tensor
    pose_timesteps: torch.Tensor
    T_world_cpf_all: torch.Tensor
    runtime_report_path: Path
    checkpoint_dir: Path
    smplh_npz_path: Path
    ego_config: dict[str, Any]
    cfg: SimpleNamespace
    denoiser_network: Any
    body_model: Any
    guidance_runtime: GuidanceRuntime | None
    all_hamer: CorrespondedHamerDetections | None
    tasks: list[SegmentTask]


def create_egoallo_runtime(args: Args) -> EgoAlloRuntime:
    if args.traj_length <= 0:
        raise ValueError(f"traj_length must be positive, got {args.traj_length}")
    if args.context_frames < 0:
        raise ValueError(f"context_frames must be >= 0, got {args.context_frames}")

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    T_world_head, pose_timesteps, _ = load_custom_transforms(args.traj_root, device)
    num_poses = T_world_head.shape[0]
    max_output_frames = num_poses - 1
    if args.start_index < 0 or args.start_index >= max_output_frames:
        raise ValueError(f"start_index must be in [0, {max_output_frames - 1}], got {args.start_index}")
    final_end = max_output_frames if args.end_index is None else min(args.end_index, max_output_frames)
    if final_end <= args.start_index:
        raise ValueError(f"end_index must be > start_index, got {final_end} <= {args.start_index}")

    tasks, final_end_for_log = build_segment_tasks(args, max_output_frames)
    print(f"{T_world_head.shape=}")
    print(f"total output frames={max_output_frames}, run range=[{args.start_index}, {final_end_for_log})")
    print(f"target segment length={args.traj_length}")
    print(f"context frames={args.context_frames}, save_only_center={args.save_only_center}")

    T_camera_cpf = torch.from_numpy(T_CAMERA_CPF).to(device)
    T_world_cpf_all = T_world_head @ T_camera_cpf
    print_head_height_stats_once(
        T_world_cpf_all,
        floor_z=args.floor_z,
        expected_min=args.expected_head_height_min,
        expected_max=args.expected_head_height_max,
    )

    checkpoint_dir = _normalize_checkpoint_dir(args.checkpoint_dir)
    if checkpoint_dir is None:
        raise ValueError("EgoAllo checkpoint is required. Set EGOALLO_CHECKPOINT_DIR or pass --checkpoint-dir.")
    model_config_path = args.model_config_path
    if model_config_path is None:
        model_config_path = EGOALLO_MODEL_CONFIG_PATH

    if args.smplh_npz_path is None:
        raise ValueError("SMPL-H model is required. Set SMPLH_NPZ_PATH/EGOALLO_MODEL_NPZ or pass --smplh-npz-path.")
    smplh_npz_path = Path(args.smplh_npz_path).expanduser().resolve()
    if not smplh_npz_path.is_file():
        raise FileNotFoundError(f"SMPL-H model not found: {smplh_npz_path}")

    ego_config = load_ego_config(args.ego_config_path)
    cfg = _build_effective_ego_config_view(
        ego_config,
        target_segment_length=args.traj_length,
        context_frames=args.context_frames,
    )

    output_root = args.output_dir if args.output_dir is not None else args.traj_root / "ego_tmp"
    runtime_report_path = _write_ego_runtime_report(output_root, args, args.ego_config_path, ego_config, cfg)

    print("\n[Load denoiser/body model once]")
    compile_model_cfg = bool(ego_config["denoiser"].get("compile_model", False))
    denoiser_network = load_denoiser_with_config(
        checkpoint_dir,
        model_config_path,
        compile_model=bool(args.ego_compile_model) or compile_model_cfg,
    ).to(device).eval()
    body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)

    inner_max_iters = _coerce_optional_int(cfg.constraint_optimization.inner_max_iters)
    post_max_iters = _coerce_optional_int(cfg.constraint_optimization.post_max_iters)
    build_guidance_runtime = (
        bool(cfg.constraint_optimization.enabled)
        and args.guidance_mode != "off"
        and (
            bool(args.guidance_inner)
            and (inner_max_iters is None or inner_max_iters > 0)
            or bool(args.guidance_post)
            and (post_max_iters is None or post_max_iters > 0)
        )
    )
    guidance_runtime = GuidanceRuntime(body_model) if build_guidance_runtime else None

    if args.hamer_outputs_path is not None:
        if not args.hamer_outputs_path.is_file():
            raise FileNotFoundError(args.hamer_outputs_path)
        if args.guidance_mode == "no_hands":
            raise ValueError("--hamer-outputs-path was provided, but --guidance-mode=no-hands would ignore it.")
        print("\n[Load HaMeR detections once on CPU]")
        pose_timesteps_np = pose_timesteps.detach().cpu().numpy()
        all_pose_ts = tuple(float(x) for x in pose_timesteps_np[1:])
        all_hamer = CorrespondedHamerDetections.load(args.hamer_outputs_path, all_pose_ts)
    else:
        if args.guidance_mode in ("hamer_wrist", "hamer_reproj2", "aria_hamer"):
            raise ValueError(f"--guidance-mode={args.guidance_mode} requires --hamer-outputs-path.")
        all_hamer = None
        print("No HaMeR/HaWoR detections provided.")

    return EgoAlloRuntime(
        args=args,
        device=device,
        T_world_head=T_world_head,
        pose_timesteps=pose_timesteps,
        T_world_cpf_all=T_world_cpf_all,
        runtime_report_path=runtime_report_path,
        checkpoint_dir=checkpoint_dir,
        smplh_npz_path=smplh_npz_path,
        ego_config=ego_config,
        cfg=cfg,
        denoiser_network=denoiser_network,
        body_model=body_model,
        guidance_runtime=guidance_runtime,
        all_hamer=all_hamer,
        tasks=tasks,
    )


def run_egoallo_segments(
    runtime: EgoAlloRuntime,
    *,
    task_indices: tuple[int, int] | None = None,
    override_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args = dataclasses.replace(runtime.args)
    if override_args:
        for key, value in override_args.items():
            setattr(args, key, value)

    segment_indices = runtime.tasks
    if task_indices is not None:
        lo, hi = task_indices
        segment_indices = [task for task in runtime.tasks if lo <= task.segment_index < hi]

    start_t = time.time()
    for task in segment_indices:
        target_start = task.target_start
        target_end = task.target_end
        target_length = task.target_length
        infer_start = task.infer_start
        infer_end = task.infer_end
        infer_length = task.infer_length
        crop_offset = task.crop_offset
        segment_index = task.segment_index
        print("\n========================================")
        print(
            f"Segment {segment_index} | target={target_start}-{target_end} "
            f"target_length={target_length} | infer={infer_start}-{infer_end} "
            f"infer_length={infer_length} crop_offset={crop_offset}"
        )
        print("========================================")

        T_world_cpf_window = runtime.T_world_cpf_all[infer_start : infer_start + infer_length + 1]
        Ts_world_cpf = (
            SE3.from_matrix(T_world_cpf_window)
            @ SE3.from_rotation(SO3.from_x_radians(runtime.T_world_head.new_tensor(args.glasses_x_angle_offset)))
        ).parameters()
        pose_timestamps_sec = runtime.pose_timesteps[infer_start + 1 : infer_start + infer_length + 1]

        hamer_detections = None
        if args.hamer_outputs_path is not None:
            hamer_detections = maybe_slice_hamer(
                runtime.all_hamer,
                args.hamer_outputs_path,
                runtime.pose_timesteps,
                infer_start,
                infer_length,
                runtime.device,
            )

        with torch.inference_mode():
            cfg = runtime.cfg
            sampling_kwargs = {
                "body_model": runtime.body_model,
                "guidance_mode": args.guidance_mode,
                "guidance_inner": args.guidance_inner,
                "guidance_post": args.guidance_post,
                "Ts_world_cpf": Ts_world_cpf,
                "hamer_detections": hamer_detections,
                "aria_detections": None,
                "num_samples": args.num_samples,
                "device": runtime.device,
                "floor_z": args.floor_z,
                "guidance_verbose": False,
                "jax_body_model": None if runtime.guidance_runtime is None else runtime.guidance_runtime.jax_body_model,
            }
            sampling_kwargs.update(
                {
                    "num_steps": int(cfg.sampling.num_steps),
                    "schedule": str(runtime.ego_config.get("sampling", {}).get("schedule", "quadratic")),
                    "window_size": int(cfg.sampling.window_size),
                    "overlap_size": int(cfg.sampling.overlap_size),
                    "keep_intermediate_states": bool(cfg.sampling.keep_intermediate_states),
                    "guidance_inner_max_iters": _coerce_optional_int(cfg.constraint_optimization.inner_max_iters),
                    "guidance_post_max_iters": _coerce_optional_int(cfg.constraint_optimization.post_max_iters),
                    "constraint_optimization_enabled": bool(cfg.constraint_optimization.enabled),
                    "constraint_optimization_inner_last_steps": _coerce_int(cfg.constraint_optimization.inner_last_steps, 0),
                    "constraint_optimization_inner_step_frequency": _coerce_int(cfg.constraint_optimization.inner_step_frequency, 1),
                    "constraint_optimization_post_last_steps": _coerce_int(cfg.constraint_optimization.post_last_steps, 0),
                    "constraint_optimization_post_step_frequency": _coerce_int(cfg.constraint_optimization.post_step_frequency, 1),
                    "constraint_optimization_post_timing_repeat": 0,
                    "constraint_optimization_bucket_t": _coerce_int(cfg.constraint_optimization.bucket_t, 0),
                    "constraint_optimization_detection_bucket": _coerce_int(cfg.constraint_optimization.detection_bucket, 0),
                    "constraint_optimization_warmup_before_segments": bool(cfg.constraint_optimization.warmup_before_segments),
                    "constraint_optimization_enable_hand_constraints": bool(cfg.constraint_optimization.enable_hand_constraints),
                    "constraint_optimization_enable_foot_constraints": bool(cfg.constraint_optimization.enable_foot_constraints),
                    "constraint_optimization_enable_collision_constraints": bool(cfg.constraint_optimization.enable_collision_constraints),
                    "constraint_optimization_enable_body_regularization": bool(cfg.constraint_optimization.enable_body_regularization),
                    "guidance_inner_repeat": _coerce_optional_int(cfg.guidance.inner_repeat),
                    "guidance_post_repeat": _coerce_optional_int(cfg.guidance.post_repeat),
                    "guidance_early_stop_grad_norm": _coerce_optional_float(cfg.guidance.early_stop_grad_norm),
                    "return_timing": False,
                }
            )
            supported_sampling_params = set(inspect.signature(run_sampling_with_stitching).parameters.keys())
            sampling_kwargs = {key: value for key, value in sampling_kwargs.items() if key in supported_sampling_params}
            sampled = run_sampling_with_stitching(runtime.denoiser_network, **sampling_kwargs)
            block_until_ready(sampled)

            if isinstance(sampled, tuple):
                traj, _stage_timing = sampled
            else:
                traj = sampled

        save_segment(
            args,
            traj,
            runtime.body_model,
            Ts_world_cpf,
            pose_timestamps_sec,
            infer_start,
            infer_length,
            target_start,
            target_length,
            crop_offset,
            segment_index,
        )

        del traj, hamer_detections
        if args.empty_cache_each_segment:
            torch.cuda.empty_cache()

    return {
        "status": "done",
        "segments": len(segment_indices),
        "elapsed_sec": round(time.time() - start_t, 3),
    }


def warmup_egoallo_runtime(runtime: EgoAlloRuntime) -> dict[str, Any]:
    if not runtime.tasks:
        return {
            "status": "done",
            "segments": 0,
            "bucket_t": int(runtime.cfg.constraint_optimization.bucket_t),
            "detection_bucket": int(runtime.cfg.constraint_optimization.detection_bucket),
        }

    first_index = int(runtime.args.segment_index_offset)
    warmup_result = run_egoallo_segments(
        runtime,
        task_indices=(first_index, first_index + 1),
        override_args={"save_traj": False, "save_args": False},
    )
    return {
        "status": "done",
        "segments": warmup_result["segments"],
        "bucket_t": int(runtime.cfg.constraint_optimization.bucket_t),
        "detection_bucket": int(runtime.cfg.constraint_optimization.detection_bucket),
    }

def save_segment(
    args: Args,
    traj,
    body_model,
    Ts_world_cpf: torch.Tensor,
    pose_timestamps_sec: torch.Tensor,
    infer_start: int,
    infer_length: int,
    target_start: int,
    target_length: int,
    crop_offset: int,
    segment_index: int,
) -> None:
    """Save only the target center crop from a possibly longer inference window.

    EgoAllo uses `infer_start:infer_start+infer_length` for sampling.  When
    context_frames > 0, that window contains extra context before/after the
    target segment.  The saved .npz keeps only frames
    `target_start:target_start+target_length`, so the final segment files do not
    overlap and can be concatenated without smoothing.
    """
    if not args.save_traj:
        return

    out_root = args.output_dir if args.output_dir is not None else args.traj_root / "ego_tmp"
    out_root.mkdir(parents=True, exist_ok=True)

    crop_start = int(crop_offset)
    crop_end = int(crop_offset + target_length)
    if crop_start < 0 or crop_end > infer_length:
        raise ValueError(
            f"Invalid center crop: crop=[{crop_start},{crop_end}) infer_length={infer_length} "
            f"infer_start={infer_start} target_start={target_start} target_length={target_length}"
        )

    target_end = target_start + target_length
    save_name = f"{segment_index}_{target_start}-{target_end}"

    if args.save_args:
        payload = dataclasses.asdict(args)
        payload.update({
            "target_start_index": int(target_start),
            "target_end_index": int(target_end),
            "target_length": int(target_length),
            "infer_start_index": int(infer_start),
            "infer_end_index": int(infer_start + infer_length),
            "infer_length": int(infer_length),
            "crop_offset": int(crop_offset),
            "segment_index": int(segment_index),
            "saved_frame_range": [int(target_start), int(target_end)],
        })
        (out_root / (save_name + "_args.yaml")).write_text(yaml.dump(payload), encoding="utf-8")

    posed = traj.apply_to_body(body_model)
    Ts_world_root_full = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
        posed, Ts_world_cpf[..., 1:, :]
    )

    # Ts_world_cpf has infer_length+1 entries.  Output frames are Ts_world_cpf[1:].
    Ts_world_cpf_frames = Ts_world_cpf[1:, :]
    Ts_world_cpf_saved = Ts_world_cpf_frames[crop_start:crop_end, :]
    pose_timestamps_saved = pose_timestamps_sec[crop_start:crop_end]

    body_quats_full = posed.local_quats[..., :21, :]
    left_hand_quats_full = posed.local_quats[..., 21:36, :]
    right_hand_quats_full = posed.local_quats[..., 36:51, :]

    Ts_world_root_saved = _slice_time_tensor(Ts_world_root_full, crop_start, crop_end, infer_length)
    body_quats_saved = _slice_time_tensor(body_quats_full, crop_start, crop_end, infer_length)
    left_hand_quats_saved = _slice_time_tensor(left_hand_quats_full, crop_start, crop_end, infer_length)
    right_hand_quats_saved = _slice_time_tensor(right_hand_quats_full, crop_start, crop_end, infer_length)
    contacts_saved = _slice_time_tensor(traj.contacts, crop_start, crop_end, infer_length)
    betas_saved = _slice_time_tensor(traj.betas, crop_start, crop_end, infer_length)

    print(
        f"Saving target crop {target_start}-{target_end} from inference window "
        f"{infer_start}-{infer_start + infer_length} to {out_root / (save_name + '.npz')}...",
        end="",
        flush=True,
    )
    np.savez(
        out_root / (save_name + ".npz"),
        Ts_world_cpf=Ts_world_cpf_saved.detach().cpu().numpy(),
        Ts_world_root=Ts_world_root_saved.detach().cpu().numpy(),
        body_quats=body_quats_saved.detach().cpu().numpy(),
        left_hand_quats=left_hand_quats_saved.detach().cpu().numpy(),
        right_hand_quats=right_hand_quats_saved.detach().cpu().numpy(),
        contacts=contacts_saved.detach().cpu().numpy(),
        betas=betas_saved.detach().cpu().numpy(),
        frame_nums=np.arange(target_start, target_end),
        timestamps_ns=(pose_timestamps_saved.detach().cpu().numpy() * 1e9).astype(np.int64),
        infer_start_index=np.asarray(infer_start, dtype=np.int64),
        infer_end_index=np.asarray(infer_start + infer_length, dtype=np.int64),
        crop_offset=np.asarray(crop_offset, dtype=np.int64),
    )
    print("saved!", flush=True)


def main(args: Args) -> None:
    runtime = create_egoallo_runtime(args)
    run_egoallo_segments(runtime)
    print("\nDone.")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
