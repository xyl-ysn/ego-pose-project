from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
import argparse
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Optional, Protocol

sys.dont_write_bytecode = True

try:
    from zata_worker.contracts.types import TaskContext
except Exception:  # pragma: no cover - local/offline fallback
    class TaskContext(Protocol):
        data_id: str | int
        settings: Any
        task_metadata: Mapping[str, Any]


ModelProgressCallback = Callable[[int, str], None]
ModelUploadCallback = Callable[[Sequence[Path], str], None]

# 本文件固定放在 project/scripts/model_plugin.py
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

KEYPOINT_TOOL_KEY = "keypoint_tool"
DEFAULT_MODEL_TYPE = "human_pose_tool"
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


@dataclass(frozen=True)
class KeypointToolInputs:
    """run_body_mesh.py 只需要的上游 keypoint_tool 产物。"""

    keypoint_dir: Path
    frames_dir: Path
    cam_space_dir: Path
    hawor_hands_vggt_world: Path
    hawor_to_vggt_alignment: Path
    vggt_head_pose: Path
    scene_points_ply: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "keypoint_dir": str(self.keypoint_dir),
            "frames_dir": str(self.frames_dir),
            "cam_space_dir": str(self.cam_space_dir),
            "hawor_hands_vggt_world": str(self.hawor_hands_vggt_world),
            "hawor_to_vggt_alignment": str(self.hawor_to_vggt_alignment),
            "vggt_head_pose": str(self.vggt_head_pose),
            "scene_points_ply": str(self.scene_points_ply),
        }


class ModelPlugin(Protocol):
    def load(self) -> None:
        ...

    def unload(self) -> None:
        ...

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        context: TaskContext,
        progress_callback: ModelProgressCallback,
        upload_callback: Optional[ModelUploadCallback] = None,
        output_cache_dir: Path,
        input_cache_map: dict[str, Path],
    ) -> dict[str, Any]:
        ...


class BodyMeshPlugin:
    """Body mesh worker plugin.

    固定约定：
    1. 插件文件位置：project/scripts/model_plugin.py
    2. 算法入口位置：project/scripts/run_body_mesh.py
    3. 配置文件路径：启动 scripts/model_plugin.py 时通过 --config 传入，或通过 BODY_MESH_CONFIG 环境变量传入
    4. 上游输入目录：input_cache_map["keypoint_tool"]
    5. 最终输出目录：output_dir

    output_cache_dir 只为满足 zata_worker 的 ModelPlugin 契约保留，插件不会向其中写入任何文件。
    """

    MODEL_TYPE = DEFAULT_MODEL_TYPE

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        config_path: Path | str | None = None,
        python_executable: str | None = None,
        validate_runtime: bool = True,
    ) -> None:
        self.project_root = (project_root or PROJECT_ROOT).expanduser().resolve()
        self.scripts_dir = self.project_root / "scripts"
        self.entrypoint = self.scripts_dir / "run_body_mesh.py"
        self.config_path = _resolve_config_path(
            config_path if config_path is not None else os.getenv("BODY_MESH_CONFIG"),
            project_root=self.project_root,
        )
        self.python_executable = python_executable or os.getenv("PYTHON_EXECUTABLE") or sys.executable
        self.validate_runtime = bool(validate_runtime)
        self._loaded = False

    def load(self) -> None:
        project_str = str(self.project_root)
        scripts_str = str(self.scripts_dir)
        for p in (project_str, scripts_str):
            if p not in sys.path:
                sys.path.insert(0, p)
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        context: TaskContext,
        progress_callback: ModelProgressCallback,
        upload_callback: Optional[ModelUploadCallback] = None,
        output_cache_dir: Path,
        input_cache_map: dict[str, Path],
    ) -> dict[str, Any]:
        # 保留该参数以符合 worker 契约，但不创建、不写入 output_cache_dir。
        _ = output_cache_dir

        if not self._loaded:
            self.load()

        started_at = time.time()
        progress_callback(0, "启动 BodyMesh 插件")

        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)

        if self.validate_runtime:
            _require_file(self.entrypoint, "run_body_mesh.py")
            _require_file(self.config_path, "body_mesh.yaml config")

        keypoint_dir = _require_input_cache(input_cache_map, KEYPOINT_TOOL_KEY)
        progress_callback(5, f"读取上游 keypoint_tool 缓存: {keypoint_dir}")
        inputs = _resolve_keypoint_tool_inputs(keypoint_dir)

        command = self._build_command(inputs=inputs, output_dir=output)

        progress_callback(15, "开始执行 scripts/run_body_mesh.py")
        _run_subprocess(
            command,
            cwd=self.project_root,
            env=_subprocess_env(self.project_root, self.scripts_dir),
        )

        priority_upload_files = _resolve_priority_upload_files(
            output,
            config_path=self.config_path,
            strict=upload_callback is not None,
        )
        if upload_callback is not None:
            progress_callback(92, "优先上传点云与 JSONL 结果")
            upload_callback(priority_upload_files, self.MODEL_TYPE)

        output_files = _collect_output_files(output)
        if not output_files:
            raise RuntimeError(f"run_body_mesh.py finished but output_dir is empty: {output}")

        elapsed_sec = time.time() - started_at
        progress_callback(100, "BodyMesh 处理完成")
        return {
            "status": "completed",
            "model_type": self.MODEL_TYPE,
            "data_id": _context_value(context, "data_id"),
            "input_path": str(Path(input_path).expanduser().resolve()),
            "output_dir": str(output),
            "upstream_cache_key": KEYPOINT_TOOL_KEY,
            "upstream_keypoint_dir": str(inputs.keypoint_dir),
            "entrypoint": str(self.entrypoint),
            "config_path": str(self.config_path),
            "elapsed_sec": round(elapsed_sec, 3),
            "inputs": inputs.to_dict(),
            "priority_upload_files": [str(path.relative_to(output)) for path in priority_upload_files],
            "outputs": {
                "result_dir": str(output),
                "files": [str(path.relative_to(output)) for path in output_files],
            },
        }

    def _build_command(self, *, inputs: KeypointToolInputs, output_dir: Path) -> list[str]:
        return [
            self.python_executable,
            str(self.entrypoint),
            "--config",
            str(self.config_path),
            "--frames-dir",
            str(inputs.frames_dir),
            "--cam-space-dir",
            str(inputs.cam_space_dir),
            "--hawor-hands-vggt-world",
            str(inputs.hawor_hands_vggt_world),
            "--hawor-to-vggt-alignment",
            str(inputs.hawor_to_vggt_alignment),
            "--vggt-head-pose",
            str(inputs.vggt_head_pose),
            "--scene-points-ply",
            str(inputs.scene_points_ply),
            "--output-dir",
            str(output_dir),
        ]


def create_plugin(*, config_path: Path | str | None = None) -> ModelPlugin:
    return BodyMeshPlugin(config_path=config_path)


@dataclass(frozen=True)
class BodyMeshWorkerSettings:
    project_root: Path
    video_cache_root: Path
    service_name: str
    service_version: str
    model_type: str
    model_names: list[str]
    instance_id: str
    zata_manager_base_url: str
    zata_manager_poll_interval: int
    zata_manager_callback_timeout: int
    zata_manager_progress_interval: int
    zata_manager_final_progress_retries: int
    zata_manager_final_progress_retry_interval: int
    max_workers: int
    fps_default: int
    frame_inference_workers: int
    gpu_cleanup_after_task: bool
    sidecar_enabled: bool
    sidecar_dir: str
    pipeline_settings: dict[str, Any] = field(default_factory=dict)


def create_worker_settings(*, project_root: Path | None = None, config_path: Path | str | None = None) -> BodyMeshWorkerSettings:
    root = (project_root or PROJECT_ROOT).expanduser().resolve()
    resolved_config = _resolve_config_path(config_path if config_path is not None else os.getenv("BODY_MESH_CONFIG"), project_root=root)
    service_name = _env_str("SERVICE_NAME", "body-mesh-worker")
    model_type = _env_str("MODEL_TYPE", BodyMeshPlugin.MODEL_TYPE)
    instance_id = _env_str("INSTANCE_ID", f"{service_name}-{socket.gethostname()}")
    return BodyMeshWorkerSettings(
        project_root=root,
        video_cache_root=_env_path("VIDEO_CACHE_ROOT", root / "data" / "worker_cache", base=root),
        service_name=service_name,
        service_version=_env_str("SERVICE_VERSION", "0.1.0"),
        model_type=model_type,
        model_names=_env_list("MODEL_NAMES", [model_type]),
        instance_id=instance_id,
        zata_manager_base_url=_env_str("ZATA_MANAGER_BASE_URL", "http://10.9.103.101:31052"),
        zata_manager_poll_interval=_env_int("ZATA_MANAGER_POLL_INTERVAL", 3),
        zata_manager_callback_timeout=_env_int("ZATA_MANAGER_CALLBACK_TIMEOUT", 30),
        zata_manager_progress_interval=_env_int("ZATA_MANAGER_PROGRESS_INTERVAL", 2),
        zata_manager_final_progress_retries=_env_int("ZATA_MANAGER_FINAL_PROGRESS_RETRIES", 3),
        zata_manager_final_progress_retry_interval=_env_int("ZATA_MANAGER_FINAL_PROGRESS_RETRY_INTERVAL", 2),
        max_workers=_env_int("MAX_WORKERS", 1),
        fps_default=_env_int("DEFAULT_FPS", 1),
        frame_inference_workers=_env_int("FRAME_INFERENCE_WORKERS", 1),
        gpu_cleanup_after_task=_env_bool("GPU_CLEANUP_AFTER_TASK", True),
        sidecar_enabled=_env_bool("SIDECAR_ENABLED", False),
        sidecar_dir=_env_str("SIDECAR_DIR", ""),
        pipeline_settings={"body_mesh_config": str(resolved_config)},
    )


def run_model_worker(
    *,
    settings: BodyMeshWorkerSettings | None = None,
    plugin: ModelPlugin | None = None,
    worker_root: Path | None = None,
    config_path: Path | str | None = None,
) -> None:
    runtime_settings = settings or create_worker_settings(config_path=config_path)
    _ensure_worker_on_path(runtime_settings.project_root, worker_root=worker_root)
    try:
        from zata_worker.runtime import init_logger, run_worker
    except ModuleNotFoundError as exc:
        if exc.name == "zata_worker":
            raise ModuleNotFoundError(
                "Cannot import zata_worker. Set ZATA_WORKER_ROOT to the worker package "
                "directory or its parent before starting scripts/model_plugin.py."
            ) from exc
        raise

    init_logger(
        level=_env_str("LOG_LEVEL", "INFO"),
        log_file=_env_optional_path("WORKER_LOG_FILE", base=runtime_settings.project_root),
        detailed_format=_env_bool("LOG_DETAILED_FORMAT", False),
        enqueue=_env_bool("LOG_ENQUEUE", False),
    )

    model = plugin or create_plugin(config_path=config_path)
    loaded = False
    try:
        model.load()
        loaded = True
        run_worker(plugin=model, settings=runtime_settings)
    finally:
        if loaded:
            model.unload()


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start BodyMesh worker plugin.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to body_mesh.yaml. Relative paths are resolved first from the current "
            "working directory, then from the project root. If omitted, BODY_MESH_CONFIG "
            "must be set."
        ),
    )
    parser.add_argument(
        "--worker-root",
        type=Path,
        default=None,
        help="Optional path to zata_worker package or its parent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    run_model_worker(config_path=args.config, worker_root=args.worker_root)


def _resolve_config_path(value: Path | str | None, *, project_root: Path) -> Path:
    if value in (None, ""):
        raise ValueError(
            "BodyMesh config path is required. Start the worker with "
            "`python scripts/model_plugin.py --config /path/to/body_mesh.yaml` "
            "or set BODY_MESH_CONFIG=/path/to/body_mesh.yaml."
        )

    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    # CLI 常规语义：相对路径优先按当前工作目录解析；
    # 如果用户从其他目录启动，也兼容按项目根目录解析。
    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = (project_root / raw).resolve()
    if project_candidate.exists():
        return project_candidate

    # 文件不存在时仍返回 cwd_candidate，后续 _require_file 会给出清晰错误。
    return cwd_candidate


def _require_input_cache(input_cache_map: dict[str, Path], key: str) -> Path:
    if key not in input_cache_map:
        raise KeyError(f"input_cache_map missing required key '{key}'. Existing keys: {sorted(input_cache_map)}")
    path = Path(input_cache_map[key]).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"input_cache_map['{key}'] is not a directory: {path}")
    return path


def _resolve_keypoint_tool_inputs(keypoint_dir: Path) -> KeypointToolInputs:
    vggt_dir = keypoint_dir / "VGGT"

    inputs = KeypointToolInputs(
        keypoint_dir=keypoint_dir,
        frames_dir=keypoint_dir / "frames",
        cam_space_dir=keypoint_dir / "cam_space",
        hawor_hands_vggt_world=vggt_dir / "hawor_hands_vggt_world.jsonl",
        hawor_to_vggt_alignment=vggt_dir / "hawor_to_vggt_alignment.json",
        vggt_head_pose=vggt_dir / "vggt_head_pose.jsonl",
        scene_points_ply=keypoint_dir / "viz" / "vggt_scene_points.ply",
    )

    _require_dir(inputs.frames_dir, "frames_dir")
    _require_dir(vggt_dir, "vggt_dir")
    _require_file(inputs.vggt_head_pose, "vggt_head_pose")
    _require_file(inputs.scene_points_ply, "scene_points_ply")
    return inputs


def _require_dir(path: Path, name: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory '{name}' not found: {path}")


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file '{name}' not found: {path}")


def _run_subprocess(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        tail = (completed.stdout or "")[-4000:]
        raise RuntimeError(
            f"BodyMesh entrypoint failed with exit code {completed.returncode}. "
            f"Command: {' '.join(command)}\nLast output:\n{tail}"
        )


def _subprocess_env(project_root: Path, scripts_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    paths = [str(project_root), str(scripts_dir)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _load_body_mesh_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to resolve priority upload output paths from config.") from exc

    with Path(config_path).expanduser().resolve().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"BodyMesh config must be a YAML mapping: {config_path}")
    return data


def _get_cfg(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _safe_output_filename(value: Any, *, default: str, cfg_key: str) -> str:
    name = str(value or default).strip() or default
    path = Path(name)
    if path.name != name or path.is_absolute():
        raise ValueError(f"{cfg_key} must be a filename, not a path: {name!r}")
    return name


def _resolve_priority_upload_files(output_dir: Path, *, config_path: Path, strict: bool) -> list[Path]:
    """Return the three files that should be prioritized by upload_callback.

    Expected files are produced after 05_apply_metric_scale.py:
      1. point cloud: <output_dir>/<egoallo_outputs_dir>/<final_scene_name>
      2. head JSONL:  <output_dir>/<egoallo_outputs_dir>/<head_jsonl_name>
      3. hands JSONL: <output_dir>/<egoallo_outputs_dir>/<hands_jsonl_name>

    When strict=True, missing files raise immediately so the worker does not report
    a successful task without uploading the expected priority files.
    """
    cfg = _load_body_mesh_config(config_path)
    outputs_dir_name = _safe_output_filename(
        _get_cfg(cfg, "output_layout.egoallo_outputs_dir_name", "egoallo_outputs"),
        default="egoallo_outputs",
        cfg_key="output_layout.egoallo_outputs_dir_name",
    )
    final_scene_name = _safe_output_filename(
        _get_cfg(cfg, "output_layout.final_scene_name", "final_scene.ply"),
        default="final_scene.ply",
        cfg_key="output_layout.final_scene_name",
    )
    head_jsonl_name = _safe_output_filename(
        _get_cfg(cfg, "metric_scale.frontend.head_jsonl_name", "head_trajectory_egoallo.jsonl"),
        default="head_trajectory_egoallo.jsonl",
        cfg_key="metric_scale.frontend.head_jsonl_name",
    )
    hands_jsonl_name = _safe_output_filename(
        _get_cfg(cfg, "metric_scale.frontend.hands_jsonl_name", "hand_keypoints_egoallo.jsonl"),
        default="hand_keypoints_egoallo.jsonl",
        cfg_key="metric_scale.frontend.hands_jsonl_name",
    )

    result_dir = Path(output_dir).expanduser().resolve() / outputs_dir_name
    paths = [
        result_dir / final_scene_name,
        result_dir / head_jsonl_name,
        result_dir / hands_jsonl_name,
    ]
    existing = [p for p in paths if p.is_file()]
    if strict:
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Priority upload files are missing after BodyMesh pipeline: "
                + ", ".join(str(p) for p in missing)
            )
    return existing


def _collect_output_files(output_dir: Path) -> list[Path]:
    return sorted(path for path in output_dir.rglob("*") if path.is_file())


def _context_value(context: object, key: str, default: Any = None) -> Any:
    if isinstance(context, Mapping):
        return context.get(key, default)
    return getattr(context, key, default)


def _ensure_worker_on_path(project_root: Path, *, worker_root: Path | None = None) -> None:
    if "zata_worker" in sys.modules:
        return
    try:
        if importlib.util.find_spec("zata_worker") is not None:
            return
    except (ImportError, ValueError):
        pass

    candidates: list[Path] = []
    if worker_root is not None:
        candidates.append(worker_root)
    env_worker_root = os.getenv("ZATA_WORKER_ROOT")
    if env_worker_root:
        candidates.append(Path(env_worker_root))
    candidates.extend(parent / "zata_worker" for parent in (project_root, *project_root.parents))

    for candidate in candidates:
        import_parent = _worker_import_parent(candidate)
        if import_parent is None:
            continue
        import_parent_str = str(import_parent)
        if import_parent_str not in sys.path:
            sys.path.insert(0, import_parent_str)
        return


def _worker_import_parent(candidate: Path) -> Path | None:
    resolved = candidate.expanduser().resolve()
    if (resolved / "__init__.py").exists() and resolved.name == "zata_worker":
        return resolved.parent
    package_dir = resolved / "zata_worker"
    if (package_dir / "__init__.py").exists():
        return resolved
    return None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSE_VALUES


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return list(default)
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or list(default)


def _env_path(name: str, default: Path, *, base: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    return path if path.is_absolute() else (base / path).resolve()


def _env_optional_path(name: str, *, base: Path) -> Path | None:
    value = os.getenv(name)
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


if __name__ == "__main__":
    main()
