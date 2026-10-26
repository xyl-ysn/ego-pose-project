"""Thread budgets for CPU preprocessing and small CPU tensor operations.

GPU inference keeps its existing thread settings. Thread changes for CPU tensor
operations are scoped and restored, including when an operation fails.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Iterator, Mapping


BLAS_THREAD_ENV = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def thread_count(value: object, name: str) -> int:
    """Zero retains library defaults; reject silently truncated fractions."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        count = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if count < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return count


def preprocessing_env(env: Mapping[str, str]) -> dict[str, str]:
    """Set defaults before numerical imports in a preprocessing subprocess.

    Explicit user settings for individual libraries take precedence. No parent
    environment is modified, and a zero budget leaves library defaults intact.
    """
    result = dict(env)
    count = thread_count(result.get("EGO_UNI_PREPROCESS_THREADS", "1"),
                         "EGO_UNI_PREPROCESS_THREADS")
    if count:
        for name in BLAS_THREAD_ENV:
            result.setdefault(name, str(count))
    return result


def cpu_tensor_threads() -> int:
    return thread_count(os.environ.get("EGO_UNI_TORCH_CPU_THREADS", "1"),
                        "EGO_UNI_TORCH_CPU_THREADS")


@contextmanager
def torch_cpu_threads() -> Iterator[None]:
    """Avoid large OpenMP teams for small CPU FK and rotation conversions.

    Use only in these single-threaded runner scripts: PyTorch's intra-op setting
    is process-wide. Inter-op threads and CUDA precision settings are untouched.
    """
    import torch

    requested = cpu_tensor_threads()
    previous = torch.get_num_threads()
    changed = requested != 0 and requested != previous
    if changed:
        torch.set_num_threads(requested)
    try:
        yield
    finally:
        if changed:
            torch.set_num_threads(previous)
