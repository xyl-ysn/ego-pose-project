#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import yaml

sys.dont_write_bytecode = True

# 本文件固定放在 project/scripts/run_body_mesh.py
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

CONVERT_TO_EGO_DIR = SCRIPTS_DIR / "convert_to_ego"
ESMATEG_DIR = SCRIPTS_DIR / "esmateG"

SCRIPT_01_CAMERA_POSES = CONVERT_TO_EGO_DIR / "01_convert_jsonl_to_camera_poses.py"
SCRIPT_02_EXTRACT_HANDS = CONVERT_TO_EGO_DIR / "02_extract_hawor_world_hands.py"
SCRIPT_03_STABILIZE_HANDS = CONVERT_TO_EGO_DIR / "03_stabilize_hawor_hands_for_egoallo.py"
SCRIPT_EXTRACT_ROT = CONVERT_TO_EGO_DIR / "extract_hawor_to_vggt_rotation.py"
SCRIPT_04_BUILD_HAMER = CONVERT_TO_EGO_DIR / "04_build_egoallo_hamer_from_hawor_world.py"
SCRIPT_ESTIMATE_MOGE_SCALE = CONVERT_TO_EGO_DIR / "04_estimate_moge_metric_scale.py"
SCRIPT_05_APPLY_SCALE = CONVERT_TO_EGO_DIR / "05_apply_metric_scale.py"
SCRIPT_05B_RESAMPLE = CONVERT_TO_EGO_DIR / "05b_resample_egoallo_inputs.py"

SCRIPT_RUN_GEOCALIB = ESMATEG_DIR / "run_geocalib.py"
SCRIPT_ESTIMATE_GRAVITY = ESMATEG_DIR / "01_estimate_gravity_da3_geocalib.py"
SCRIPT_APPLY_ALIGNMENT = ESMATEG_DIR / "02_apply_alignment_da3_geocalib.py"

SCRIPT_RUN_EGOALLO = SCRIPTS_DIR / "05_run_egoallo.sh"
SCRIPT_SMOOTH_EGOALLO = SCRIPTS_DIR / "06_smooth_egoallo_segments.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BodyMesh/EgoAllo pipeline from keypoint_tool cache.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--cam-space-dir", type=Path, required=True)
    parser.add_argument("--hawor-hands-vggt-world", type=Path, required=True)
    parser.add_argument("--hawor-to-vggt-alignment", type=Path, required=True)
    parser.add_argument("--vggt-head-pose", type=Path, required=True)
    parser.add_argument("--scene-points-ply", type=Path, required=True)
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

    # 只把项目脚本路径加入 PYTHONPATH。EgoAllo/GeoCalib 源码路径不在这里提前加入，
    # 对应子脚本内部会自行决定 import 优先级；这里保留源码/conda 的配置路径并透传给子脚本。
    prepend_pythonpath(env, [PROJECT_ROOT, SCRIPTS_DIR])

    egoallo_root = resolve_project_path(get_cfg(cfg, "model_sources.egoallo_root"))
    geocalib_root = resolve_project_path(get_cfg(cfg, "model_sources.geocalib_root"))
    moge_root = resolve_project_path(get_cfg(cfg, "model_sources.moge_root"))
    if egoallo_root is not None:
        env["EGOALLO_ROOT"] = str(egoallo_root)
    if geocalib_root is not None:
        env["GEOCALIB_ROOT"] = str(geocalib_root)
    if moge_root is not None:
        # 只作为源码路径传给下游脚本，实际源码/conda优先级由下游脚本自行控制。
        env["MOGE_ROOT"] = str(moge_root)

    env_path_map = {
        "EGOALLO_CHECKPOINT_DIR": "model_weights.egoallo_checkpoint_dir",
        "EGOALLO_MODEL_CONFIG_PATH": "model_weights.egoallo_model_config_path",
        "EGOALLO_MODEL_NPZ": "model_weights.egoallo_model_npz",
        "GEOCALIB_PINHOLE_TAR": "model_weights.geocalib_pinhole_tar",
        "SMPLH_NPZ_PATH": "model_weights.smplh_npz_path",
        "VPOSER_DIR": "model_weights.vposer_dir",
        "MOGE_MODEL_PT": "model_weights.moge_model_pt",
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


def configure_jax_compilation_cache(env: dict[str, str], cfg: dict[str, Any], tmp_dir: Path) -> Path | None:
    """Enable JAX persistent compilation cache under <output_dir>/tmp.

    This is intentionally configured after tmp_dir is known, because the user wants
    the cache to live inside the per-task output tmp directory rather than in a
    global /app cache. It must be set before EgoAllo/JAX guidance is launched.
    """
    enabled = _cfg_bool(
        get_cfg(cfg, "jax_guidance.compilation_cache.enabled", True),
        default=True,
    )
    if not enabled:
        return None

    cache_subdir = str(
        get_cfg(
            cfg,
            "jax_guidance.compilation_cache.cache_subdir",
            "jax_worker_cache",
        )
    )
    cache_dir = (tmp_dir / cache_subdir).expanduser().resolve()
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
        get_cfg(
            cfg,
            "jax_guidance.compilation_cache.enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
    )

    if _cfg_bool(get_cfg(cfg, "jax_guidance.compilation_cache.explain_cache_misses", False)):
        env["JAX_EXPLAIN_CACHE_MISSES"] = "true"

    if _cfg_bool(get_cfg(cfg, "jax_guidance.compilation_cache.debug_log_modules", False)):
        env["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"
        env["JAX_LOGGING_LEVEL"] = "DEBUG"

    # Export to both env keys for legacy import paths in EgoAllo code.
    env["EGOALLO_JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)

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
        (SCRIPT_05_APPLY_SCALE, "05_apply_metric_scale.py"),
        (SCRIPT_05B_RESAMPLE, "05b_resample_egoallo_inputs.py"),
        (SCRIPT_RUN_GEOCALIB, "run_geocalib.py"),
        (SCRIPT_ESTIMATE_GRAVITY, "01_estimate_gravity_da3_geocalib.py"),
        (SCRIPT_APPLY_ALIGNMENT, "02_apply_alignment_da3_geocalib.py"),
        (SCRIPT_RUN_EGOALLO, "05_run_egoallo.sh"),
        (SCRIPT_SMOOTH_EGOALLO, "06_smooth_egoallo_segments.py"),
    ]:
        require_file(script, name)

    frames_dir = args.frames_dir.expanduser().resolve()
    cam_space_dir = args.cam_space_dir.expanduser().resolve()
    hawor_hands_vggt_world = args.hawor_hands_vggt_world.expanduser().resolve()
    hawor_to_vggt_alignment = args.hawor_to_vggt_alignment.expanduser().resolve()
    vggt_head_pose = args.vggt_head_pose.expanduser().resolve()
    scene_points_ply = args.scene_points_ply.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    require_dir(frames_dir, "frames_dir")
    require_file(vggt_head_pose, "vggt_head_pose")
    require_file(scene_points_ply, "scene_points_ply")

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / str(get_cfg(cfg, "output_layout.tmp_dir_name", "tmp"))
    egoallo_inputs_dir = output_dir / str(get_cfg(cfg, "output_layout.egoallo_inputs_dir_name", "egoallo_inputs"))
    egoallo_outputs_dir = output_dir / str(get_cfg(cfg, "output_layout.egoallo_outputs_dir_name", "egoallo_outputs"))
    ego_tmp_dir = output_dir / str(get_cfg(cfg, "output_layout.ego_tmp_dir_name", "ego_tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    egoallo_inputs_dir.mkdir(parents=True, exist_ok=True)
    egoallo_outputs_dir.mkdir(parents=True, exist_ok=True)
    ego_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Keep Python bytecode cache out of project source directories.
    python_pycache_dir, python_pycache_cleanup = configure_python_pycache(env, cfg, tmp_dir)

    # JAX guidance persistent compilation cache lives under <output_dir>/tmp.
    # It is configured before any subprocess can launch EgoAllo/JAX guidance.
    jax_compilation_cache_dir = configure_jax_compilation_cache(env, cfg, tmp_dir)

    global PIPELINE_RECORDER
    pipeline_log_json = tmp_dir / str(get_cfg(cfg, "output_layout.pipeline_log_json_name", "pipeline_run_log.json"))
    PIPELINE_RECORDER = PipelineRunRecorder(
        pipeline_log_json,
        metadata={
            "project_root": str(PROJECT_ROOT),
            "config_path": str(args.config.expanduser().resolve()),
            "output_dir": str(output_dir),
            "tmp_dir": str(tmp_dir),
            "egoallo_inputs_dir": str(egoallo_inputs_dir),
            "egoallo_outputs_dir": str(egoallo_outputs_dir),
            "ego_tmp_dir": str(ego_tmp_dir),
            "python_executable": sys.executable,
            "python_pycache_dir": str(python_pycache_dir) if python_pycache_dir is not None else None,
            "python_pycache_cleanup": python_pycache_cleanup,
            "jax_compilation_cache_dir": str(jax_compilation_cache_dir) if jax_compilation_cache_dir is not None else None,
            "inputs": {
                "frames_dir": str(frames_dir),
                "cam_space_dir": str(cam_space_dir),
                "hawor_hands_vggt_world": str(hawor_hands_vggt_world),
                "hawor_to_vggt_alignment": str(hawor_to_vggt_alignment),
                "vggt_head_pose": str(vggt_head_pose),
                "scene_points_ply": str(scene_points_ply),
            },
        },
    )

    camera_poses_txt = tmp_dir / "camera_poses.txt"
    timestamps_txt = egoallo_inputs_dir / "timestamps.txt"
    intrinsics_txt = tmp_dir / "intrinsics.txt"

    prepared_hands_jsonl = tmp_dir / "hawor_hands_for_egoallo_world.jsonl"
    stable_hands_jsonl = tmp_dir / "hawor_hands_for_egoallo_world_stable.jsonl"
    R_hawor_to_vggt_npy = tmp_dir / "R_hawor_to_vggt.npy"
    hamer_outputs_pkl_tmp = tmp_dir / "hamer_outputs.pkl"
    empty_hawor_hands_vggt_world = tmp_dir / "empty_hawor_hands_vggt_world.jsonl"

    geocalib_gravity_npz = tmp_dir / "geocalib_gravity.npz"
    R_align_npy = tmp_dir / "R_align.npy"
    alignment_transform_npz = tmp_dir / "alignment_transform.npz"
    aligned_scene_ply = tmp_dir / "aligned_scene.ply"
    head_trajectory_tmp = tmp_dir / "head_trajectory.npy"

    head_trajectory_out = egoallo_inputs_dir / "head_trajectory.npy"
    hamer_outputs_pkl_out = egoallo_inputs_dir / "hamer_outputs.pkl"
    final_scene_name = str(get_cfg(cfg, "output_layout.final_scene_name", "final_scene.ply"))
    final_scene_ply = egoallo_outputs_dir / final_scene_name
    metric_scale_metadata_json = tmp_dir / "metric_scale_metadata.json"

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

    # 02: HaWoR world hands -> EgoAllo prepared hands
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
        "--max-gap-fill", str(get_cfg(cfg, "hawor_extract.max_gap_fill", 3)),
        "--max-gap-endpoint-hand-ratio", str(get_cfg(cfg, "hawor_extract.max_gap_endpoint_hand_ratio", 1.5)),
        "--smooth-window", str(get_cfg(cfg, "hawor_extract.smooth_window", 5)),
        "--smooth-mode", str(get_cfg(cfg, "hawor_extract.smooth_mode", "translation")),
        "--soft-start-anchor-max-gap", str(get_cfg(cfg, "hawor_extract.soft_start_anchor_max_gap", 45)),
    ]
    add_flag(cmd, "--drop-isolated", get_cfg(cfg, "hawor_extract.drop_isolated", True))
    if not bool(get_cfg(cfg, "hawor_extract.drop_isolated", True)):
        cmd.append("--no-drop-isolated")
    add_flag(cmd, "--soft-start-geometry", get_cfg(cfg, "hawor_extract.soft_start_geometry", True))
    if not bool(get_cfg(cfg, "hawor_extract.soft_start_geometry", True)):
        cmd.append("--no-soft-start-geometry")
    add_flag(cmd, "--print-jump-stats", get_cfg(cfg, "hawor_extract.print_jump_stats", True))
    run(cmd, env=env)

    # 03: stabilize hands
    run([
        py,
        str(SCRIPT_03_STABILIZE_HANDS),
        "--input-jsonl", str(prepared_hands_jsonl),
        "--output-jsonl", str(stable_hands_jsonl),
        "--max-wrist-step-hand-ratio", str(get_cfg(cfg, "hawor_stabilize.max_wrist_step_hand_ratio", 2.0)),
        "--min-track-len", str(get_cfg(cfg, "hawor_stabilize.min_track_len", 4)),
        "--soft-start-frames", str(get_cfg(cfg, "hawor_stabilize.soft_start_frames", 15)),
        "--soft-end-frames", str(get_cfg(cfg, "hawor_stabilize.soft_end_frames", 6)),
        "--min-alpha-to-keep", str(get_cfg(cfg, "hawor_stabilize.min_alpha_to_keep", 0.0)),
        "--duplicate-mode", str(get_cfg(cfg, "hawor_stabilize.duplicate_mode", "drop_both")),
    ], env=env)

    # HaWoR -> VGGT rotation. If no hands were detected upstream, the alignment
    # file may be absent; use identity so the empty-hand path can still finish.
    run([
        py,
        str(SCRIPT_EXTRACT_ROT),
        "--alignment-json", str(hawor_to_vggt_alignment),
        "--output-npy", str(R_hawor_to_vggt_npy),
        "--missing-is-identity",
    ], env=env)

    # 04: build EgoAllo hamer pkl from HaWoR world hands
    run([
        py,
        str(SCRIPT_04_BUILD_HAMER),
        "--prepared-hands-jsonl", str(stable_hands_jsonl),
        "--camera-poses-txt", str(camera_poses_txt),
        "--timestamps-txt", str(timestamps_txt),
        "--output-pkl", str(hamer_outputs_pkl_tmp),
        "--cam-space-left-dir", str(cam_space_dir / str(get_cfg(cfg, "hamer_build.cam_space_left_subdir", "0"))),
        "--cam-space-right-dir", str(cam_space_dir / str(get_cfg(cfg, "hamer_build.cam_space_right_subdir", "1"))),
        "--mano-frame-key", str(get_cfg(cfg, "hamer_build.mano_frame_key", "frame_id")),
        "--mano-lookup-field", str(get_cfg(cfg, "hamer_build.mano_lookup_field", "frame_id")),
        "--mano-left-pkl", str(mano_left),
        "--mano-right-pkl", str(mano_right),
        "--camera-frame-rotation-npy", str(R_hawor_to_vggt_npy),
    ], env=env)

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
        "--follow-frames", str(get_cfg(cfg, "geocalib.follow_frames", 5)),
        "--empty-policy", str(get_cfg(cfg, "geocalib.empty_policy", "skip")),
    ]
    geocalib_min_confidence = get_cfg(cfg, "geocalib.min_confidence", None)
    if geocalib_min_confidence is not None:
        geocalib_cmd.extend(["--min-confidence", str(geocalib_min_confidence)])

    if bool(get_cfg(cfg, "geocalib.viz_enabled", False)):
        viz_dir_name = str(get_cfg(cfg, "geocalib.viz_dir_name", "geocalib_viz"))
        # GeoCalib 可视化图保存到 tmp 下。
        # 注意：如果 output_layout.cleanup_tmp=true，则完整 pipeline 成功后 tmp 会被删除，
        # 这些可视化图片也会一起删除；调试时可把 cleanup_tmp 设为 false。
        geocalib_viz_dir = tmp_dir / viz_dir_name
        geocalib_cmd.extend(["--viz-dir", str(geocalib_viz_dir)])
        if bool(get_cfg(cfg, "geocalib.viz_save_all_sampled", True)):
            geocalib_cmd.append("--save-all-sampled-viz")
        if bool(get_cfg(cfg, "geocalib.viz_save_selected", False)):
            geocalib_cmd.append("--save-selected-viz")

    # geocalib_root 透传给 01_estimate_gravity_da3_geocalib.py，由其决定源码/conda回退逻辑。
    geocalib_root = env.get("GEOCALIB_ROOT")
    if geocalib_root:
        geocalib_cmd.extend(["--geocalib-root", geocalib_root])
    if env.get("GEOCALIB_PINHOLE_TAR"):
        geocalib_cmd.extend(["--geocalib-pinhole-weight", env["GEOCALIB_PINHOLE_TAR"]])

    run(geocalib_cmd, env=env)

    # Gravity alignment
    # GeoCalib is optional. When enabled, 01_estimate_gravity_da3_geocalib.py
    # still computes a point-cloud-only RANSAC baseline first. GeoCalib is only
    # used as a prior when it disagrees with RANSAC beyond the switch angle, and
    # the resulting floor/up proposal must pass camera-height validation or it
    # will fall back to RANSAC.
    estimate_gravity_cmd = [
        py,
        str(SCRIPT_ESTIMATE_GRAVITY),
        "--pcd", str(scene_points_ply),
        "--poses", str(camera_poses_txt),
        "--out", str(R_align_npy),
        "--out_meta", str(alignment_transform_npz),
    ]

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

    run([
        py,
        str(SCRIPT_APPLY_ALIGNMENT),
        "--poses", str(camera_poses_txt),
        "--R", str(R_align_npy),
        "--align_meta", str(alignment_transform_npz),
        "--out_poses", str(head_trajectory_tmp),
        "--poses-only",
    ], env=env)

    # Optional MoGe metric-scale estimation.
    # MoGe is expensive, so it is deferred until after a lightweight hands-only
    # scale probe. If hands are selected successfully, MoGe is skipped.
    moge_scale_json: Path | None = None
    moge_output_dir: Path | None = None
    metric_scale_enabled = bool(get_cfg(cfg, "metric_scale.enabled", True))
    moge_enabled = metric_scale_enabled and bool(get_cfg(cfg, "metric_scale.moge.enabled", False))
    moge_model_pt: Path | None = None
    if moge_enabled:
        require_file(SCRIPT_ESTIMATE_MOGE_SCALE, "04_estimate_moge_metric_scale.py")
        moge_output_dir = tmp_dir / str(get_cfg(cfg, "metric_scale.moge.output_dir_name", "moge_metric_scale"))
        moge_scale_json = moge_output_dir / "moge_scale_estimate.json"
        moge_model_pt = resolve_project_path(get_cfg(cfg, "model_weights.moge_model_pt", env.get("MOGE_MODEL_PT")))

    def run_alignment_point_cloud_for_moge() -> Path:
        moge_aligned_scene_ply = tmp_dir / "aligned_scene_for_moge.ply"
        run([
            py,
            str(SCRIPT_APPLY_ALIGNMENT),
            "--pcd", str(scene_points_ply),
            "--poses", str(camera_poses_txt),
            "--R", str(R_align_npy),
            "--align_meta", str(alignment_transform_npz),
            "--out_pcd", str(moge_aligned_scene_ply),
            "--out_poses", str(tmp_dir / "head_trajectory_for_moge.npy"),
            "--output_voxel_size", str(get_cfg(
                cfg,
                "metric_scale.moge.aligned_scene_voxel_size",
                get_cfg(cfg, "metric_scale.final_scene_output_voxel_size", 0.01),
            )),
        ], env=env)
        return moge_aligned_scene_ply

    def run_moge_metric_scale_if_needed() -> None:
        if not moge_enabled or moge_scale_json is None:
            return
        if moge_model_pt is None:
            write_failed_scale_json(
                moge_scale_json,
                source="moge",
                reason="model_weights.moge_model_pt_not_set",
            )
            print(f"[MoGe] Skip MoGe scale: model_weights.moge_model_pt is not set. Wrote {moge_scale_json}", flush=True)
        elif not moge_model_pt.is_file():
            write_failed_scale_json(
                moge_scale_json,
                source="moge",
                reason="moge_model_pt_not_found",
                extra={"moge_model_pt": str(moge_model_pt)},
            )
            print(f"[MoGe] Skip MoGe scale: model.pt not found: {moge_model_pt}. Wrote {moge_scale_json}", flush=True)
        else:
            moge_point_cloud_in = run_alignment_point_cloud_for_moge()
            moge_cmd = [
                py,
                str(SCRIPT_ESTIMATE_MOGE_SCALE),
                "--frames-dir", str(frames_dir),
                "--timestamps-txt", str(timestamps_txt),
                "--head-trajectory-in", str(head_trajectory_tmp),
                "--point-cloud-in", str(moge_point_cloud_in),
                "--intrinsics-path", str(intrinsics_txt),
                "--moge-model-pt", str(moge_model_pt),
                "--output-dir", str(moge_output_dir),
                "--output-json", str(moge_scale_json),
                "--candidate-samples", str(get_cfg(cfg, "metric_scale.moge.candidate_samples", 48)),
                "--num-samples", str(get_cfg(cfg, "metric_scale.moge.num_samples", 24)),
                "--min-quality-brightness", str(get_cfg(cfg, "metric_scale.moge.min_quality_brightness", 25.0)),
                "--max-quality-brightness", str(get_cfg(cfg, "metric_scale.moge.max_quality_brightness", 235.0)),
                "--max-quality-dark-ratio", str(get_cfg(cfg, "metric_scale.moge.max_quality_dark_ratio", 0.45)),
                "--max-quality-bright-ratio", str(get_cfg(cfg, "metric_scale.moge.max_quality_bright_ratio", 0.45)),
                "--min-quality-laplacian-var", str(get_cfg(cfg, "metric_scale.moge.min_quality_laplacian_var", 20.0)),
                "--min-quality-texture-std", str(get_cfg(cfg, "metric_scale.moge.min_quality_texture_std", 8.0)),
                "--min-quality-edge-density", str(get_cfg(cfg, "metric_scale.moge.min_quality_edge_density", 0.01)),
                "--quality-edge-threshold", str(get_cfg(cfg, "metric_scale.moge.quality_edge_threshold", 20.0)),
                "--min-valid-frames", str(get_cfg(cfg, "metric_scale.moge.min_valid_frames", 6)),
                "--min-valid-pixels", str(get_cfg(cfg, "metric_scale.moge.min_valid_pixels", 1000)),
                "--min-depth-m", str(get_cfg(cfg, "metric_scale.moge.min_depth_m", 0.2)),
                "--max-depth-m", str(get_cfg(cfg, "metric_scale.moge.max_depth_m", 8.0)),
                "--min-scale", str(get_cfg(cfg, "metric_scale.moge.min_scale", 0.05)),
                "--max-scale", str(get_cfg(cfg, "metric_scale.moge.max_scale", 20.0)),
                "--max-frame-relative-mad", str(get_cfg(cfg, "metric_scale.moge.max_frame_relative_mad", 0.35)),
                "--max-final-relative-mad", str(get_cfg(cfg, "metric_scale.moge.max_final_relative_mad", 0.35)),
                "--max-points", str(get_cfg(cfg, "metric_scale.moge.max_points", 250000)),
                "--max-saved-match-pixels", str(get_cfg(cfg, "metric_scale.moge.max_saved_match_pixels", 200000)),
                "--projection-quality-enabled", str(get_cfg(cfg, "metric_scale.moge.projection_quality_enabled", True)),
                "--min-projection-area-ratio", str(get_cfg(cfg, "metric_scale.moge.min_projection_area_ratio", 0.01)),
                "--min-depth-range-m", str(get_cfg(cfg, "metric_scale.moge.min_depth_range_m", 0.25)),
                "--min-depth-layer-ratio", str(get_cfg(cfg, "metric_scale.moge.min_depth_layer_ratio", 0.03)),
                "--min-depth-layer-count", str(get_cfg(cfg, "metric_scale.moge.min_depth_layer_count", 2)),
                "--max-center-region-ratio", str(get_cfg(cfg, "metric_scale.moge.max_center_region_ratio", 0.90)),
                "--min-static-background-score", str(get_cfg(cfg, "metric_scale.moge.min_static_background_score", 0.35)),
                "--device", str(get_cfg(cfg, "metric_scale.moge.device", "cuda")),
                "--fp16", str(get_cfg(cfg, "metric_scale.moge.fp16", True)),
                "--resolution-level", str(get_cfg(cfg, "metric_scale.moge.resolution_level", 9)),
                "--use-intrinsics-fov", str(get_cfg(cfg, "metric_scale.moge.use_intrinsics_fov", True)),
                "--save-depth-maps", str(get_cfg(cfg, "metric_scale.moge.save_depth_maps", True)),
                "--save-projection-matches", str(get_cfg(cfg, "metric_scale.moge.save_projection_matches", True)),
                "--save-sampled-frames", str(get_cfg(cfg, "metric_scale.moge.save_sampled_frames", False)),
            ]
            moge_num_tokens = get_cfg(cfg, "metric_scale.moge.num_tokens", None)
            if moge_num_tokens is not None:
                moge_cmd.extend(["--num-tokens", str(moge_num_tokens)])
            if bool(get_cfg(cfg, "metric_scale.moge.strict", False)):
                moge_cmd.append("--strict")
            run(moge_cmd, env=env)

    # Metric scale
    # 05_apply_metric_scale.py no longer reads YAML directly. run_body_mesh.py is the single
    # config owner: it reads metric_scale.* from YAML and passes concrete CLI args here.
    if metric_scale_enabled:
        hand_scale_bones_json = tmp_dir / "hand_scale_bones.json"
        with hand_scale_bones_json.open("w", encoding="utf-8") as f:
            json.dump(
                get_cfg(cfg, "metric_scale.hand_fallback.bones", []),
                f,
                ensure_ascii=False,
                indent=2,
            )

        scale_priority_csv = cfg_list_to_csv(
            get_cfg(cfg, "metric_scale.scale_priority", None),
            ["hands", "moge", "height", "none"],
        )
        metric_scale_cmd = [
            py,
            str(SCRIPT_05_APPLY_SCALE),
            "--head-trajectory-in", str(head_trajectory_tmp),
            "--point-cloud-in", str(scene_points_ply),
            "--hamer-pkl-in", str(hamer_outputs_pkl_tmp),
            "--point-cloud-space", "raw_vggt",
            "--point-cloud-r-align", str(R_align_npy),
            "--point-cloud-align-meta", str(alignment_transform_npz),
            "--final-scene-output-voxel-size", str(get_cfg(cfg, "metric_scale.final_scene_output_voxel_size", 0.01)),
            "--head-trajectory-out", str(head_trajectory_out),
            "--point-cloud-out", str(final_scene_ply),
            "--hamer-pkl-out", str(hamer_outputs_pkl_out),
            "--metadata-out", str(metric_scale_metadata_json),
            "--floor-z", str(get_cfg(cfg, "metric_scale.floor_z", 0.0)),
            "--scale-priority", scale_priority_csv,
            "--hand-scale-enabled", "1" if bool(get_cfg(cfg, "metric_scale.hand_fallback.enabled", True)) else "0",
            "--hand-scale-bones-json", str(hand_scale_bones_json),
            "--hand-scale-min-valid-bones", str(get_cfg(cfg, "metric_scale.hand_fallback.min_valid_bones", 20)),
            "--hand-scale-max-relative-mad", str(get_cfg(cfg, "metric_scale.hand_fallback.max_relative_mad", 0.35)),
            "--hand-scale-min-scale", str(get_cfg(cfg, "metric_scale.hand_fallback.min_scale", 0.05)),
            "--hand-scale-max-scale", str(get_cfg(cfg, "metric_scale.hand_fallback.max_scale", 20.0)),
            "--hand-max-p90-head-height", str(get_cfg(cfg, "metric_scale.hand_fallback.max_p90_head_height", 1.70)),

            "--scale-sanity-enabled", "1" if bool(get_cfg(cfg, "metric_scale.scale_sanity.enabled", True)) else "0",
            "--scale-sanity-statistic", str(get_cfg(cfg, "metric_scale.scale_sanity.statistic", "percentile")),
            "--scale-sanity-percentile", str(get_cfg(cfg, "metric_scale.scale_sanity.percentile", 90.0)),
            "--scale-sanity-min-valid-head-height", str(get_cfg(cfg, "metric_scale.scale_sanity.min_valid_head_height", 0.8)),
            "--scale-sanity-max-valid-head-height", str(get_cfg(cfg, "metric_scale.scale_sanity.max_valid_head_height", 2.4)),
            "--scale-sanity-reject-moge-if-invalid-height", "1" if bool(get_cfg(cfg, "metric_scale.scale_sanity.reject_moge_if_invalid_height", True)) else "0",
            "--scale-sanity-reject-hands-if-invalid-height", "1" if bool(get_cfg(cfg, "metric_scale.scale_sanity.reject_hands_if_invalid_height", True)) else "0",

            "--height-fallback-enabled", "1" if bool(get_cfg(cfg, "metric_scale.height_fallback.enabled", True)) else "0",
            "--height-fallback-mode", str(get_cfg(cfg, "metric_scale.height_fallback.mode", "standing_head_height")),
            "--height-fallback-statistic", str(get_cfg(cfg, "metric_scale.height_fallback.statistic", "percentile")),
            "--height-fallback-percentile", str(get_cfg(cfg, "metric_scale.height_fallback.percentile", 90.0)),
            "--height-fallback-min-valid-height", str(get_cfg(cfg, "metric_scale.height_fallback.min_valid_height", 1.0)),
            "--height-fallback-max-valid-height", str(get_cfg(cfg, "metric_scale.height_fallback.max_valid_height", 2.3)),
            "--height-fallback-min-candidate-frames", str(get_cfg(cfg, "metric_scale.height_fallback.min_candidate_frames", 10)),
            "--height-fallback-min-scale", str(get_cfg(cfg, "metric_scale.height_fallback.min_scale", 0.05)),
            "--height-fallback-max-scale", str(get_cfg(cfg, "metric_scale.height_fallback.max_scale", 20.0)),
            "--height-fallback-smooth-window", str(get_cfg(cfg, "metric_scale.height_fallback.smooth_window", 1)),

            "--hands-vggt-jsonl-for-scale", str(hands_jsonl_for_pipeline),
            "--R-align-for-scale", str(R_align_npy),
            "--align-meta-for-scale", str(alignment_transform_npz),
        ]

        height_fallback_target_height = get_cfg(cfg, "metric_scale.height_fallback.target_height", None)
        if height_fallback_target_height is not None:
            metric_scale_cmd.extend(["--height-fallback-target-height", str(height_fallback_target_height)])

        if bool(get_cfg(cfg, "metric_scale.frontend.enabled", True)):
            metric_scale_cmd.extend([
                "--export-frontend-jsonl",
                "--frontend-output-dir", str(egoallo_outputs_dir),
                "--frontend-head-jsonl-name", str(get_cfg(cfg, "metric_scale.frontend.head_jsonl_name", "head_trajectory_egoallo.jsonl")),
                "--frontend-hands-jsonl-name", str(get_cfg(cfg, "metric_scale.frontend.hands_jsonl_name", "hand_keypoints_egoallo.jsonl")),
                "--timestamps-txt", str(timestamps_txt),
                "--hands-vggt-jsonl-in", str(hands_jsonl_for_pipeline),
                "--R-align", str(R_align_npy),
                "--align-meta", str(alignment_transform_npz),
            ])
            add_flag(
                metric_scale_cmd,
                "--frontend-include-mesh",
                get_cfg(cfg, "metric_scale.frontend.include_mesh", False),
            )

        priority_items = [x.strip().lower() for x in scale_priority_csv.split(",") if x.strip()]
        hands_before_moge = (
            "hands" in priority_items
            and "moge" in priority_items
            and priority_items.index("hands") < priority_items.index("moge")
        )
        run_moge_scale = bool(moge_enabled and "moge" in priority_items)
        if run_moge_scale and hands_before_moge and bool(get_cfg(cfg, "metric_scale.hand_fallback.enabled", True)):
            hand_probe_json = tmp_dir / "metric_scale_hand_probe.json"
            hand_probe_cmd = list(metric_scale_cmd)
            hand_probe_cmd[hand_probe_cmd.index("--metadata-out") + 1] = str(hand_probe_json)
            hand_probe_cmd[hand_probe_cmd.index("--scale-priority") + 1] = "hands"
            hand_probe_cmd.append("--probe-scale-only")
            run(hand_probe_cmd, env=env)
            try:
                with hand_probe_json.open("r", encoding="utf-8") as f:
                    hand_probe = json.load(f)
                hand_probe_source = hand_probe.get("scale_source")
            except Exception as exc:
                hand_probe_source = None
                print(f"[Warning] Failed to read hand scale probe metadata: {exc!r}. Fallback to MoGe.", flush=True)

            if hand_probe_source == "hands":
                run_moge_scale = False
                print("[MoGe] Skip MoGe scale: hands scale probe succeeded.", flush=True)
            else:
                print(f"[MoGe] Hands scale probe did not select hands ({hand_probe_source}); run MoGe fallback.", flush=True)

        if run_moge_scale:
            run_moge_metric_scale_if_needed()
            if moge_scale_json is not None:
                metric_scale_cmd.extend(["--moge-scale-json", str(moge_scale_json)])

        run(metric_scale_cmd, env=env)
    else:
        shutil.copy2(head_trajectory_tmp, head_trajectory_out)
        shutil.copy2(hamer_outputs_pkl_tmp, hamer_outputs_pkl_out)
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

    # Optional temporal resampling: usually from sparse (~5 FPS) to target FPS (default 30 FPS).
    # Keep this stage in the main pipeline and controlled by temporal_resampling.* in body_mesh.yaml.
    run([
        py,
        str(SCRIPT_05B_RESAMPLE),
        "--head-trajectory", str(head_trajectory_out),
        "--timestamps-txt", str(timestamps_txt),
        "--hamer-pkl", str(hamer_outputs_pkl_out),
        "--config", str(args.config.expanduser().resolve()),
        "--tmp-dir", str(tmp_dir),
        "--output-dir", str(output_dir),
    ], env=env)

    # EgoAllo：不再使用 demo/sequence_name，直接使用 output_dir 下的目录。
    # 05_run_egoallo_from_head.py 的可选参数统一由 YAML 的 egoallo 段控制，
    # 这里写入环境变量，再由 05_run_egoallo.sh 转成命令行参数。
    env["EGOALLO_GUIDANCE_MODE"] = str(get_cfg(cfg, "egoallo.guidance_mode", "hamer_wrist"))
    env["EGOALLO_GUIDANCE_INNER"] = "1" if bool(get_cfg(cfg, "egoallo.guidance_inner", True)) else "0"
    env["EGOALLO_GUIDANCE_POST"] = "1" if bool(get_cfg(cfg, "egoallo.guidance_post", True)) else "0"
    env["EGOALLO_TARGET_SEGMENT_LENGTH"] = str(get_cfg(cfg, "egoallo.target_segment_length", 256))
    env["EGOALLO_CONTEXT_FRAMES"] = str(get_cfg(cfg, "egoallo.context_frames", 0))
    env["EGOALLO_SAVE_ONLY_CENTER"] = "1" if bool(get_cfg(cfg, "egoallo.save_only_center", True)) else "0"
    env["EGOALLO_GLASSES_X_ANGLE_OFFSET"] = str(get_cfg(cfg, "egoallo.glasses_x_angle_offset", 0.0))
    env["EGOALLO_START_INDEX"] = str(get_cfg(cfg, "egoallo.start_index", 0))
    env["EGOALLO_SEGMENT_INDEX_OFFSET"] = str(get_cfg(cfg, "egoallo.segment_index_offset", 0))
    env["EGOALLO_NUM_SAMPLES"] = str(get_cfg(cfg, "egoallo.num_samples", 1))
    env["EGOALLO_FLOOR_Z"] = str(get_cfg(cfg, "egoallo.floor_z", 0.0))
    env["EGOALLO_SAVE_TRAJ"] = "1" if bool(get_cfg(cfg, "egoallo.save_traj", True)) else "0"
    env["EGOALLO_SAVE_ARGS"] = "1" if bool(get_cfg(cfg, "egoallo.save_args", True)) else "0"
    env["EGOALLO_EMPTY_CACHE_EACH_SEGMENT"] = "1" if bool(get_cfg(cfg, "egoallo.empty_cache_each_segment", False)) else "0"
    env["EGOALLO_ALLOW_TF32"] = "1" if bool(get_cfg(cfg, "egoallo.allow_tf32", False)) else "0"
    env["EGOALLO_EXPECTED_HEAD_HEIGHT_MIN"] = str(get_cfg(cfg, "egoallo.expected_head_height_min", 1.0))
    env["EGOALLO_EXPECTED_HEAD_HEIGHT_MAX"] = str(get_cfg(cfg, "egoallo.expected_head_height_max", 2.2))
    env["EGOALLO_EGO_CONFIG_PATH"] = str(args.config.expanduser().resolve())
    env["EGOALLO_JAX_CACHE_LOG"] = str(tmp_dir / "jax_cache_hits.log")
    if bool(get_cfg(cfg, "egoallo.disable_dlpack", False)):
        env["EGOALLO_DISABLE_DLPACK"] = "1"

    end_index = get_cfg(cfg, "egoallo.end_index", None)
    if end_index is not None:
        env["EGOALLO_END_INDEX"] = str(end_index)
    else:
        env.pop("EGOALLO_END_INDEX", None)

    segment_length = str(get_cfg(cfg, "egoallo.target_segment_length", 256))
    run([
        "bash",
        str(SCRIPT_RUN_EGOALLO),
        str(egoallo_inputs_dir),
        str(ego_tmp_dir),
        segment_length,
    ], env=env)

    # Merge/smooth：从 output_dir/ego_tmp 读取 EgoAllo 分段 .npz，输出到 output_dir/egoallo_outputs。
    # 06 脚本负责分段接缝 blend 和可选的全局轻量时序平滑。
    motion_quality_report_json = tmp_dir / str(get_cfg(cfg, "smooth.report_json_name", "motion_quality_report.json"))
    if bool(get_cfg(cfg, "smooth.enabled", True)):
        smooth_cmd = [
            py,
            str(SCRIPT_SMOOTH_EGOALLO),
            "--input-dir", str(ego_tmp_dir),
            "--output-dir", str(egoallo_outputs_dir),
            "--output-name", str(get_cfg(cfg, "output_layout.final_smoothed_name", "merged_smoothed_vis.npz")),
            "--report-json", str(motion_quality_report_json),
            "--boundary-blend-frames", str(get_cfg(cfg, "smooth.boundary_blend_frames", 0)),
            "--translation-smooth-window", str(get_cfg(cfg, "smooth.translation_smooth_window", 0)),
            "--quat-lowpass-alpha", str(get_cfg(cfg, "smooth.quat_lowpass_alpha", 0.0)),
        ]
        add_flag(smooth_cmd, "--print-stats", get_cfg(cfg, "smooth.print_stats", True))
        add_flag(smooth_cmd, "--strict-frame-continuity", get_cfg(cfg, "smooth.strict_frame_continuity", False))
        run(smooth_cmd, env=env)

    metric_scale_result = None
    if metric_scale_metadata_json.is_file():
        try:
            with metric_scale_metadata_json.open("r", encoding="utf-8") as f:
                metric_scale_result = json.load(f)
        except Exception as exc:
            metric_scale_result = {"status": "metadata_read_failed", "error": repr(exc)}

    summary = {
        "status": "completed",
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(output_dir),
        "tmp_dir": str(tmp_dir),
        "egoallo_inputs_dir": str(egoallo_inputs_dir),
        "egoallo_outputs_dir": str(egoallo_outputs_dir),
        "ego_tmp_dir": str(ego_tmp_dir),
        "python_cache": {
            "pycache_prefix": env.get("PYTHONPYCACHEPREFIX"),
            "cleanup_existing_scripts_pycache": python_pycache_cleanup,
        },
        "model_sources": {
            "egoallo_root_fallback": env.get("EGOALLO_ROOT"),
            "geocalib_root_fallback": env.get("GEOCALIB_ROOT"),
            "moge_root_fallback": env.get("MOGE_ROOT"),
        },
        "model_weights": {
            "egoallo_checkpoint_dir": env.get("EGOALLO_CHECKPOINT_DIR"),
            "egoallo_model_config_path": env.get("EGOALLO_MODEL_CONFIG_PATH"),
            "egoallo_model_npz": env.get("EGOALLO_MODEL_NPZ"),
            "geocalib_pinhole_tar": env.get("GEOCALIB_PINHOLE_TAR"),
            "mano_left_pkl": str(mano_left),
            "mano_right_pkl": str(mano_right),
            "smplh_npz_path": env.get("SMPLH_NPZ_PATH"),
            "vposer_dir": env.get("VPOSER_DIR"),
            "moge_model_pt": env.get("MOGE_MODEL_PT"),
        },
        "moge_metric_scale": {
            "enabled": bool(moge_enabled),
            "output_dir": str(moge_output_dir) if moge_output_dir is not None else None,
            "scale_json": str(moge_scale_json) if moge_scale_json is not None else None,
        },
        "metric_scale_result": metric_scale_result,
        "motion_quality_report": str(motion_quality_report_json),
        "egoallo_overlap": {
            "target_segment_length": int(segment_length),
            "context_frames": int(get_cfg(cfg, "egoallo.context_frames", 0)),
            "save_only_center": bool(get_cfg(cfg, "egoallo.save_only_center", True)),
        },
        "jax_guidance": {
            "compilation_cache_enabled": env.get("JAX_ENABLE_COMPILATION_CACHE") == "true",
            "compilation_cache_dir": env.get("JAX_COMPILATION_CACHE_DIR"),
            "min_compile_time_secs": env.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"),
            "min_entry_size_bytes": env.get("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"),
            "enable_xla_caches": env.get("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"),
        },
        "inputs": {
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
    #   4) EgoAllo segment inference into ego_tmp_dir
    #   5) 06_smooth_egoallo_segments.py merge-only stage, if smooth.enabled=true
    # If any previous subprocess raises an error, execution will stop before this block,
    # and all intermediate directories will be kept for debugging.
    cleanup_results: dict[str, Any] = {}

    cleanup_results["egoallo_inputs"] = remove_dir_if_requested(
        "egoallo_inputs",
        egoallo_inputs_dir,
        cleanup_flag(cfg, "egoallo_inputs", default=False),
    )
    cleanup_results["ego_tmp"] = remove_dir_if_requested(
        "ego_tmp",
        ego_tmp_dir,
        cleanup_flag(cfg, "ego_tmp", default=False),
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
