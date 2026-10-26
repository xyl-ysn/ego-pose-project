"""Run with the pipeline Python: python scripts/test_egoallo_refine_device.py."""

import runpy
from pathlib import Path
from unittest.mock import patch

import torch


resolve_device = runpy.run_path(
    str(Path(__file__).with_name("07_run_egoallo_refine_with_prior.py"))
)["resolve_torch_device"]

with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
    torch.cuda, "current_device", return_value=0
) as current_device:
    device = resolve_device("cuda")
    current_device.return_value = 6
    assert str(device) == "cuda:0"  # Remains pinned after JAX changes the device.
    assert str(resolve_device("cuda")) == "cuda:6"
    assert str(resolve_device("cuda:2")) == "cuda:2"

with patch.object(torch.cuda, "is_available", return_value=False):
    assert str(resolve_device("cpu")) == "cpu"
    try:
        resolve_device("cuda")
    except RuntimeError as error:
        assert "CUDA requested but unavailable" in str(error)
    else:
        raise AssertionError("Unavailable CUDA should fail explicitly")

print("CUDA device resolution checks passed")
