#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import argparse
import json
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment-json", type=Path, required=True)
    ap.add_argument("--output-npy", type=Path, required=True)
    ap.add_argument(
        "--json-key",
        default="depth_scale_report.sim3.R",
        help="Dot path to the 3x3 rotation. Default: depth_scale_report.sim3.R",
    )
    ap.add_argument(
        "--missing-is-identity",
        action="store_true",
        help="If --alignment-json is absent, write an identity rotation instead of failing.",
    )
    ap.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose the loaded rotation before saving.",
    )
    args = ap.parse_args()

    if args.missing_is_identity and not args.alignment_json.is_file():
        R = np.eye(3, dtype=np.float32)
        print(f"[HaWoR] Alignment JSON not found: {args.alignment_json}; using identity rotation.")
    else:
        data = json.loads(args.alignment_json.read_text(encoding="utf-8"))
        value = data
        for key in args.json_key.split("."):
            value = value[key]
        R = np.asarray(value, dtype=np.float32)
        if args.transpose:
            R = R.T

    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation, got {R.shape}")

    det = float(np.linalg.det(R.astype(np.float64)))
    orth_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    if abs(det - 1.0) > 1e-3 or orth_err > 1e-3:
        raise ValueError(f"Not a valid rotation: det={det}, orth_err={orth_err}")

    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, R)
    print(f"Saved: {args.output_npy}")
    print(f"det={det:.6f}, orth_err={orth_err:.6g}")

if __name__ == "__main__":
    main()
