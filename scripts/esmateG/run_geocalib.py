#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run GeoCalib on sampled frames and save per-frame camera-frame gravity vectors.

The output should be passed to 01_estimate_gravity_da3_geocalib.py via:

    --geocalib_gravity geocalib_gravity.npz
    --geocalib_vector_type gravity

Important:
    The saved vectors are in each image's camera coordinate system. Do not average
    them here before transforming through the matching camera poses.
"""

import argparse
import json
import re
import sys
from pathlib import Path


# Resolve only the project location from this script. GeoCalib源码和权重不再在这里写死，
# 由 run_body_mesh.py 从 config/body_mesh.yaml 读出后通过环境变量或命令行传入。
import os

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

import numpy as np
import torch

visualize_calibration = None

try:
    import cv2
except Exception:
    cv2 = None


def _resolve_path(value: str | None, *, name: str, must_exist: bool = True) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{name} is required. Pass it by CLI or environment variable.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _resolve_geocalib_paths(args):
    # geocalib_root 是可选的源码路径：
    # - 如果设置且可用，优先插入 GEOCALIB_ROOT/--geocalib-root 进行源码导入；
    # - 如果源码导入失败，再回退到当前 conda/site-packages。
    geocalib_root_value = args.geocalib_root or os.getenv("GEOCALIB_ROOT")
    geocalib_root = None
    if geocalib_root_value is not None and str(geocalib_root_value).strip() != "":
        geocalib_root = _resolve_path(
            geocalib_root_value,
            name="GeoCalib source directory",
        )

    pinhole_weight = _resolve_path(
        args.geocalib_pinhole_weight or os.getenv("GEOCALIB_PINHOLE_TAR"),
        name="GeoCalib pinhole weight",
    )
    return geocalib_root, pinhole_weight


def _import_geocalib():
    from geocalib import GeoCalib

    global visualize_calibration
    try:
        from geocalib.visualization import visualize_calibration as _visualize_calibration
        visualize_calibration = _visualize_calibration
    except Exception:
        visualize_calibration = None
    return GeoCalib


def _clear_geocalib_modules() -> None:
    for name in list(sys.modules):
        if name == "geocalib" or name.startswith("geocalib."):
            del sys.modules[name]


def _load_geocalib_class(geocalib_root: Path | None):
    """Prefer YAML/GEOCALIB_ROOT source; fall back to conda-installed geocalib."""
    source_path = str(geocalib_root) if geocalib_root is not None else None
    source_exc: Exception | None = None

    if geocalib_root is not None:
        geocalib_root_str = str(geocalib_root)
        if geocalib_root_str not in sys.path:
            sys.path.insert(0, geocalib_root_str)
        try:
            GeoCalib = _import_geocalib()
            print(f"GeoCalib import source: source {geocalib_root}")
            return GeoCalib, f"source {geocalib_root}"
        except Exception as first_exc:
            source_exc = first_exc
            if geocalib_root_str in sys.path:
                del sys.path[sys.path.index(geocalib_root_str)]
            _clear_geocalib_modules()

    try:
        GeoCalib = _import_geocalib()
        print("GeoCalib import source: conda/site-packages")
        return GeoCalib, "conda/site-packages"
    except Exception as first_exc:
        if geocalib_root is None:
            raise ModuleNotFoundError(
                "Cannot import geocalib from source (not provided) and conda/site-packages."
            ) from first_exc
        if source_exc is not None:
            raise ModuleNotFoundError(
                f"Cannot import geocalib from {source_path} and conda/site-packages."
            ) from source_exc
        raise ModuleNotFoundError("Cannot import geocalib from conda/site-packages.") from first_exc

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_frame_id(path, fallback):
    """
    Extract a frame index from a filename.

    Examples:
        000123.png      -> 123
        frame_000123.jpg -> 123

    If no integer exists in the stem, return the sampled order as fallback.
    """
    matches = re.findall(r"\d+", Path(path).stem)
    if not matches:
        return int(fallback)
    return int(matches[-1])


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("GeoCalib returned a near-zero gravity vector.")
    return v / n


def tensor_to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_gravity_vector(result):
    """
    Return a camera-frame down/gravity vector from GeoCalib output.

    Preferred key is "gravity". If a package version exposes only "up", this
    function converts it to gravity/down by negating it.
    """
    if "gravity" in result:
        return normalize(tensor_to_numpy(result["gravity"]))
    if "g" in result:
        return normalize(tensor_to_numpy(result["g"]))
    if "up" in result:
        return normalize(-tensor_to_numpy(result["up"]))
    if "upright" in result:
        return normalize(-tensor_to_numpy(result["upright"]))

    keys = ", ".join(str(k) for k in result.keys())
    raise KeyError(
        "GeoCalib result does not contain gravity/up vector. "
        f"Available keys: {keys}"
    )


def get_uncertainty(result):
    cov = result.get("covariance", None)
    if cov is None:
        return None
    cov_np = tensor_to_numpy(cov)
    return float(np.trace(np.asarray(cov_np, dtype=np.float64).squeeze()))


def collect_images(img_dir):
    img_dir = Path(img_dir)
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(img_dir.glob(f"*{ext}"))
        paths.extend(img_dir.glob(f"*{ext.upper()}"))
    return sorted(set(paths))


def sample_uniform(paths, num_samples):
    if num_samples <= 0 or len(paths) <= num_samples:
        return list(enumerate(paths))
    indices = np.linspace(0, len(paths) - 1, num_samples)
    indices = np.round(indices).astype(np.int64)
    return [(int(i), paths[int(i)]) for i in indices]


def extract_camera_matrix(result, image_shape):
    """
    Best-effort extraction of a 3x3 camera matrix from GeoCalib output.

    If the installed GeoCalib version uses a different camera object layout,
    fall back to a simple pinhole approximation. The fallback is sufficient for
    a diagnostic overlay, while the saved gravity vectors remain unchanged.
    """
    h, w = image_shape[:2]
    camera = result.get("camera", None)
    candidates = []

    if camera is not None:
        if isinstance(camera, dict):
            for key in ["K", "camera_matrix", "calibration_matrix", "intrinsics"]:
                if key in camera:
                    candidates.append(camera[key])
        else:
            for attr in ["K", "camera_matrix", "calibration_matrix", "intrinsics"]:
                if hasattr(camera, attr):
                    value = getattr(camera, attr)
                    candidates.append(value() if callable(value) else value)
            for method in ["get_K", "get_camera_matrix", "get_intrinsics"]:
                if hasattr(camera, method):
                    try:
                        candidates.append(getattr(camera, method)())
                    except Exception:
                        pass

    for candidate in candidates:
        try:
            K = tensor_to_numpy(candidate)
            K = np.asarray(K, dtype=np.float64).squeeze()
            if K.shape == (3, 3) and np.isfinite(K).all():
                return K
        except Exception:
            pass

    focal = float(max(w, h))
    return np.array(
        [
            [focal, 0.0, 0.5 * (w - 1)],
            [0.0, focal, 0.5 * (h - 1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def line_segment_in_image(line, width, height):
    a, b, c = [float(x) for x in line]
    points = []

    if abs(b) > 1e-12:
        for x in [0.0, float(width - 1)]:
            y = -(a * x + c) / b
            if 0.0 <= y <= height - 1:
                points.append((int(round(x)), int(round(y))))

    if abs(a) > 1e-12:
        for y in [0.0, float(height - 1)]:
            x = -(b * y + c) / a
            if 0.0 <= x <= width - 1:
                points.append((int(round(x)), int(round(y))))

    unique = []
    for point in points:
        if point not in unique:
            unique.append(point)

    if len(unique) < 2:
        return None
    return unique[0], unique[1]


def draw_fallback_visualization(result, img_path, g_cam, viz_dir):
    if cv2 is None:
        return False

    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[Warning] Could not read image for visualization: {img_path}")
        return False

    h, w = image.shape[:2]
    K = extract_camera_matrix(result, image.shape)

    # Horizon of the horizontal plane whose normal is gravity:
    # x^T K^-T g = 0.
    try:
        horizon = np.linalg.inv(K).T @ normalize(g_cam)
        segment = line_segment_in_image(horizon, w, h)
        if segment is not None:
            cv2.line(image, segment[0], segment[1], (0, 220, 0), 3, cv2.LINE_AA)
            cv2.putText(
                image,
                "horizon",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )
    except Exception as exc:
        print(f"[Warning] Failed to draw horizon for {Path(img_path).name}: {exc}")

    center = np.array([0.5 * (w - 1), 0.5 * (h - 1)], dtype=np.float64)
    direction = np.array([g_cam[0], g_cam[1]], dtype=np.float64)
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([0.0, 1.0], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    length = 0.25 * min(w, h)
    end = center + length * direction
    start_i = tuple(np.round(center).astype(int))
    end_i = tuple(np.round(end).astype(int))

    cv2.arrowedLine(image, start_i, end_i, (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.20)
    cv2.putText(
        image,
        f"gravity [{g_cam[0]:+.2f}, {g_cam[1]:+.2f}, {g_cam[2]:+.2f}]",
        (12, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    out_path = Path(viz_dir) / f"geocalib_viz_{Path(img_path).stem}.jpg"
    cv2.imwrite(str(out_path), image)
    return True


def save_optional_visualization(img, result, img_path, viz_dir, g_cam):
    if viz_dir is None:
        return

    if cv2 is None:
        return

    if visualize_calibration is not None:
        try:
            viz = visualize_calibration(
                img,
                result.get("camera", None),
                result.get("gravity", result.get("g", None)),
                show_horizon=True,
                show_gravity=True,
            )
            viz_np = tensor_to_numpy(viz)
            if viz_np.ndim == 4:
                viz_np = viz_np[0]
            if viz_np.shape[0] in (1, 3) and viz_np.ndim == 3:
                viz_np = np.moveaxis(viz_np, 0, -1)
            viz_np = np.asarray(viz_np, dtype=np.float32)
            if viz_np.max() <= 1.0:
                viz_np = viz_np * 255.0
            viz_np = np.clip(viz_np, 0, 255).astype(np.uint8)
            if viz_np.ndim == 3 and viz_np.shape[2] == 3:
                viz_np = cv2.cvtColor(viz_np, cv2.COLOR_RGB2BGR)

            out_path = Path(viz_dir) / f"geocalib_viz_{Path(img_path).stem}.jpg"
            cv2.imwrite(str(out_path), viz_np)
            return
        except Exception as exc:
            print(
                f"[Warning] Official GeoCalib visualization failed for "
                f"{Path(img_path).name}; using fallback overlay. Reason: {exc}"
            )

    draw_fallback_visualization(result, img_path, g_cam, viz_dir)



def save_output(output_path, gravity_vectors, frame_ids, image_paths, *, metadata=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gravity_vectors = np.asarray(gravity_vectors, dtype=np.float64).reshape(-1, 3)
    frame_ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    image_names = np.asarray([Path(p).name for p in image_paths], dtype=str)
    image_paths = np.asarray([str(Path(p)) for p in image_paths], dtype=str)

    if len(gravity_vectors) == 0:
        gravity_avg_camera = np.full(3, np.nan, dtype=np.float64)
    else:
        gravity_avg_camera = normalize(np.mean(gravity_vectors, axis=0))

    metadata = dict(metadata or {})

    if output_path.suffix.lower() == ".npz":
        np.savez(
            output_path,
            gravity=gravity_vectors,
            gravity_cam=gravity_vectors,
            frame_ids=frame_ids,
            image_names=image_names,
            image_paths=image_paths,
            gravity_avg_camera=gravity_avg_camera,
            metadata_json=np.array(json.dumps(metadata, ensure_ascii=False), dtype=object),
            skipped=np.array(bool(metadata.get("skipped", False)), dtype=bool),
            skip_reason=np.array(str(metadata.get("skip_reason", "")), dtype=object),
            num_candidates=np.array(int(metadata.get("num_candidates", 0)), dtype=np.int64),
            num_selected=np.array(int(len(gravity_vectors)), dtype=np.int64),
        )
    elif output_path.suffix.lower() in [".txt", ".csv"]:
        table = np.column_stack([frame_ids, gravity_vectors]) if len(gravity_vectors) else np.zeros((0, 4))
        delimiter = "," if output_path.suffix.lower() == ".csv" else " "
        np.savetxt(
            output_path,
            table,
            fmt=["%d", "%.10f", "%.10f", "%.10f"],
            delimiter=delimiter,
            header="frame_id gravity_x gravity_y gravity_z",
            comments="# ",
        )
    else:
        np.save(output_path, gravity_vectors)
        print(
            "[Warning] Saved .npy without frame_ids. Prefer .npz so the next "
            "script can match GeoCalib vectors to the correct camera poses."
        )

    return gravity_avg_camera


def save_empty_output(output_path, *, reason, img_count, sampled_count, candidate_uncertainties=None):
    metadata = {
        "skipped": True,
        "skip_reason": str(reason),
        "num_images": int(img_count),
        "num_sampled": int(sampled_count),
        "num_candidates": 0,
        "num_selected": 0,
    }
    if candidate_uncertainties is not None:
        metadata["candidate_uncertainties"] = [
            None if u is None or not np.isfinite(float(u)) else float(u)
            for u in candidate_uncertainties
        ]
    save_output(
        output_path,
        np.zeros((0, 3), dtype=np.float64),
        np.zeros((0,), dtype=np.int64),
        [],
        metadata=metadata,
    )
    print(f"[Warning] GeoCalib produced no usable vectors; wrote empty output: {output_path}")
    print(f"[Warning] skip_reason={reason}")


def _uncertainty_sort_key(candidate):
    u = candidate.get("uncertainty")
    if u is None:
        return float("inf")
    try:
        u = float(u)
    except Exception:
        return float("inf")
    if not np.isfinite(u):
        return float("inf")
    return u


def _passes_uncertainty(candidate, threshold):
    if threshold is None:
        return True
    u = candidate.get("uncertainty")
    if u is None:
        return False
    try:
        return float(u) <= float(threshold)
    except Exception:
        return False


def run_geocalib_one(model, img_path, path_index, *, device, viz_dir=None, save_viz=False):
    img_path = Path(img_path)
    img = model.load_image(img_path).to(device)
    result = model.calibrate(
        img,
        camera_model="pinhole",
        priors=None,
    )

    uncertainty = get_uncertainty(result)
    g_cam = extract_gravity_vector(result)
    frame_id = parse_frame_id(img_path, fallback=path_index)

    if save_viz and viz_dir is not None:
        save_optional_visualization(img, result, img_path, viz_dir, g_cam)

    return {
        "path_index": int(path_index),
        "frame_id": int(frame_id),
        "image_path": img_path,
        "gravity": np.asarray(g_cam, dtype=np.float64).reshape(3),
        "uncertainty": None if uncertainty is None else float(uncertainty),
    }


def select_anchor_indices(
    *,
    candidates,
    min_confidence,
    anchor_policy,
    top_k_anchors,
):
    anchor_policy = str(anchor_policy or "top_k_best_uncertainty")
    top_k_anchors = max(1, int(top_k_anchors))
    valid = [c for c in candidates if _passes_uncertainty(c, min_confidence)]
    valid = sorted(valid, key=_uncertainty_sort_key)

    if not valid:
        return [], [], []

    if anchor_policy == "all_valid":
        anchors = valid
        selected_indices = {int(c["path_index"]) for c in valid}
    elif anchor_policy == "best_uncertainty":
        anchors = valid[:1]
        selected_indices = {int(c["path_index"]) for c in anchors}
    elif anchor_policy == "top_k_best_uncertainty":
        anchors = valid[:top_k_anchors]
        selected_indices = {int(c["path_index"]) for c in anchors}
    else:
        raise ValueError(
            f"Unsupported anchor_policy={anchor_policy!r}. "
            "Use one of: best_uncertainty, top_k_best_uncertainty, all_valid."
        )

    return sorted(selected_indices), anchors, valid


def main():
    parser = argparse.ArgumentParser(
        description="Run GeoCalib on sampled frames to estimate gravity direction"
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        required=True,
        help="Directory containing extracted frames, e.g. data/frames/demo/images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/outputs/tmp/geocalib_gravity.npz",
        help="Output .npz/.npy path. .npz is recommended because it stores frame_ids.",
    )
    parser.add_argument(
        "--geocalib-root",
        type=str,
        default=None,
        help=(
            "Optional GeoCalib source directory. If provided, this is tried first; "
            "falls back to current conda/site-packages on failure."
        ),
    )
    parser.add_argument(
        "--geocalib-pinhole-weight",
        type=str,
        default=None,
        help="GeoCalib pinhole.tar path. Default: GEOCALIB_PINHOLE_TAR from environment.",
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help="Optional output directory for GeoCalib visualization images",
    )
    parser.add_argument(
        "--save-all-sampled-viz",
        action="store_true",
        help=(
            "Save gravity visualization images for all uniformly sampled candidate frames "
            "before uncertainty filtering. With --num-samples=30 this saves 30 images; "
            "if total images are fewer than 30, it saves all images."
        ),
    )
    parser.add_argument(
        "--save-selected-viz",
        action="store_true",
        help=(
            "Also save visualization images for selected anchor frames. "
            "Disabled by default so --save-all-sampled-viz keeps exactly the uniform-sample visualizations."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Uniformly sampled candidate frames. If images are fewer than this, all images are sampled. Use <=0 to process all frames.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help=(
            "Legacy uncertainty threshold. Candidates with trace(covariance) greater "
            "than this value are rejected. Default: no threshold."
        ),
    )
    parser.add_argument(
        "--anchor-policy",
        type=str,
        default="top_k_best_uncertainty",
        choices=["best_uncertainty", "top_k_best_uncertainty", "all_valid"],
        help=(
            "How to choose frames after running GeoCalib on uniformly sampled candidates. "
            "top_k_best_uncertainty selects the lowest-uncertainty anchors only."
        ),
    )
    parser.add_argument(
        "--top-k-anchors",
        type=int,
        default=3,
        help="Number of lowest-uncertainty valid anchors to use when anchor-policy=top_k_best_uncertainty.",
    )
    parser.add_argument(
        "--empty-policy",
        type=str,
        default="skip",
        choices=["skip", "error"],
        help="If no candidate passes threshold: write an empty output and continue, or raise an error.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on",
    )

    args = parser.parse_args()

    geocalib_repo, geocalib_weight_path = _resolve_geocalib_paths(args)
    GeoCalib, geocalib_source = _load_geocalib_class(geocalib_repo)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    img_dir = Path(args.img_dir)
    output_path = Path(args.output)
    viz_dir = Path(args.viz_dir) if args.viz_dir else None

    if not img_dir.exists():
        raise FileNotFoundError(img_dir)

    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
        if visualize_calibration is None:
            print(
                "[Warning] geocalib.visualization is not available in this "
                "GeoCalib install. Gravity output will still be generated; "
                "fallback OpenCV visualization will be used."
            )
        if cv2 is None:
            print(
                "[Warning] OpenCV is not available. Gravity output will still "
                "be generated; visualization images will be skipped."
            )

    img_paths = collect_images(img_dir)
    if len(img_paths) == 0:
        raise ValueError(f"No images found in {img_dir}")

    sampled_paths = sample_uniform(img_paths, args.num_samples)
    print(f"Using device: {device}")
    print(f"Sampled {len(sampled_paths)} candidate frames from {len(img_paths)} images")
    print(
        "GeoCalib selection: "
        f"anchor_policy={args.anchor_policy}, top_k_anchors={args.top_k_anchors}, "
        f"min_confidence={args.min_confidence}, "
        f"empty_policy={args.empty_policy}, "
        f"save_all_sampled_viz={args.save_all_sampled_viz}, "
        f"save_selected_viz={args.save_selected_viz}"
    )

    print(f"Project root: {PROJECT_ROOT}")
    print(f"GeoCalib source: {geocalib_source}")
    print(f"GeoCalib weight: {geocalib_weight_path}")

    model = GeoCalib(weights=str(geocalib_weight_path)).to(device)
    if hasattr(model, "eval"):
        model.eval()

    candidates_by_index = {}
    candidate_errors = []

    def get_or_run_candidate(path_index, *, save_viz=False):
        path_index = int(path_index)
        if path_index in candidates_by_index:
            return candidates_by_index[path_index]
        try:
            candidate = run_geocalib_one(
                model,
                img_paths[path_index],
                path_index,
                device=device,
                viz_dir=viz_dir,
                save_viz=bool(save_viz),
            )
            candidates_by_index[path_index] = candidate
            return candidate
        except Exception as exc:
            candidate_errors.append({
                "path_index": int(path_index),
                "image_name": Path(img_paths[path_index]).name,
                "error": repr(exc),
            })
            print(f"  [Warning] GeoCalib failed on {Path(img_paths[path_index]).name}: {exc}")
            return None

    with torch.inference_mode():
        sampled_candidates = []
        for sample_i, (path_index, img_path) in enumerate(sampled_paths):
            candidate = get_or_run_candidate(
                path_index,
                save_viz=bool(args.save_all_sampled_viz),
            )
            if candidate is None:
                continue
            sampled_candidates.append(candidate)
            u = candidate.get("uncertainty")
            status = f"  Candidate {len(sampled_candidates)}/{len(sampled_paths)}: {Path(img_path).name}"
            if u is not None:
                status += f" uncertainty={u:.6g}"
            if args.min_confidence is not None:
                status += " PASS" if _passes_uncertainty(candidate, args.min_confidence) else " rejected"
            print(status)

        selected_indices, anchors, valid_candidates = select_anchor_indices(
            candidates=sampled_candidates,
            min_confidence=args.min_confidence,
            anchor_policy=args.anchor_policy,
            top_k_anchors=args.top_k_anchors,
        )

        if len(selected_indices) == 0:
            reason = "no_valid_geocalib_candidate_after_uncertainty_filter"
            if args.empty_policy == "error":
                raise RuntimeError(reason)
            save_empty_output(
                output_path,
                reason=reason,
                img_count=len(img_paths),
                sampled_count=len(sampled_paths),
                candidate_uncertainties=[c.get("uncertainty") for c in sampled_candidates],
            )
            return

        selected_candidates = []
        rejected_selected_candidates = []
        for path_index in selected_indices:
            candidate = get_or_run_candidate(
                path_index,
                save_viz=bool(args.save_selected_viz),
            )
            if candidate is None:
                continue
            if not _passes_uncertainty(candidate, args.min_confidence):
                rejected_selected_candidates.append(candidate)
                u = candidate.get("uncertainty")
                msg = f"  Selected frame rejected by confidence filter: {Path(candidate['image_path']).name}"
                if u is not None:
                    msg += f" uncertainty={u:.6g}"
                print(msg)
                continue
            selected_candidates.append(candidate)

    if len(selected_candidates) == 0:
        reason = "selected_geocalib_frames_all_failed_or_rejected"
        if args.empty_policy == "error":
            raise RuntimeError(reason)
        save_empty_output(
            output_path,
            reason=reason,
            img_count=len(img_paths),
            sampled_count=len(sampled_paths),
            candidate_uncertainties=[c.get("uncertainty") for c in sampled_candidates],
        )
        return

    gravity_vectors = [c["gravity"] for c in selected_candidates]
    frame_ids = [c["frame_id"] for c in selected_candidates]
    used_paths = [c["image_path"] for c in selected_candidates]

    anchor_info = [
        {
            "path_index": int(c["path_index"]),
            "frame_id": int(c["frame_id"]),
            "image_name": Path(c["image_path"]).name,
            "uncertainty": c.get("uncertainty"),
        }
        for c in anchors
    ]
    metadata = {
        "skipped": False,
        "num_images": int(len(img_paths)),
        "num_sampled": int(len(sampled_paths)),
        "num_candidates": int(len(sampled_candidates)),
        "num_valid_candidates": int(len(valid_candidates)),
        "num_selected": int(len(selected_candidates)),
        "num_rejected_selected": int(len(rejected_selected_candidates)),
        "anchor_policy": str(args.anchor_policy),
        "top_k_anchors": int(args.top_k_anchors),
        "min_confidence": None if args.min_confidence is None else float(args.min_confidence),
        "anchors": anchor_info,
        "candidate_errors": candidate_errors,
        "viz_dir": str(viz_dir) if viz_dir is not None else None,
        "save_all_sampled_viz": bool(args.save_all_sampled_viz),
        "save_selected_viz": bool(args.save_selected_viz),
        "selected_path_indices": [int(c["path_index"]) for c in selected_candidates],
        "selected_frame_ids": [int(c["frame_id"]) for c in selected_candidates],
        "selected_uncertainties": [c.get("uncertainty") for c in selected_candidates],
        "rejected_selected_path_indices": [int(c["path_index"]) for c in rejected_selected_candidates],
        "rejected_selected_frame_ids": [int(c["frame_id"]) for c in rejected_selected_candidates],
        "rejected_selected_uncertainties": [c.get("uncertainty") for c in rejected_selected_candidates],
    }

    gravity_avg_camera = save_output(
        output_path,
        gravity_vectors,
        frame_ids,
        used_paths,
        metadata=metadata,
    )

    print("\n" + "=" * 50)
    print("GeoCalib completed")
    print(f"Valid sampled candidates: {len(valid_candidates)}/{len(sampled_candidates)}")
    print(f"Selected frames saved: {len(selected_candidates)}")
    print(f"Selected frames rejected by confidence filter: {len(rejected_selected_candidates)}")
    print(f"Selected anchors: {anchor_info}")
    print(f"Saved per-frame gravity vectors: {output_path}")
    print(f"Output shape: ({len(gravity_vectors)}, 3)")
    print(f"Frame id range: {min(frame_ids)} -> {max(frame_ids)}")
    print(f"Average gravity in camera coords, for diagnostics only: {gravity_avg_camera}")
    if viz_dir is not None and cv2 is not None:
        print(f"Visualizations saved to: {viz_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
