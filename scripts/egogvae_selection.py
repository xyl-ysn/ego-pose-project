#!/usr/bin/env python3
"""Continuity-aware EgoGVAE window selection and overlap blending.

This module is pure and operates on in-memory raw window predictions. It is
used by the production EgoGVAE runner and can also be reused by diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


ROT_TRANSITION_JOINTS = [2, 0, 1, 3, 4, 6, 7]
FOOT_JOINTS = [6, 7, 9, 10]
HIGH_RISK_FRAMES = {195, 243, 304, 338, 466, 553, 595}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def window_position_cost(local: int, center: float, lo: float, hi: float) -> float:
    if lo <= local <= hi:
        return float(((local - center) / (hi - center)) ** 2)
    d = float(min(abs(local - lo), abs(local - hi)))
    return 1.0 + (d / 32.0) ** 2


def contact_confidence_cost(contacts: np.ndarray) -> float:
    return float(np.mean([1.0 - abs(2.0 * contacts[idx] - 1.0) for idx in FOOT_JOINTS]))


def slerp_quat_np(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.clip(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-8, None)
    q1 = q1 / np.clip(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-8, None)
    dot = np.clip(np.sum(q0 * q1, axis=-1), -1.0, 1.0)
    sign = np.where(dot < 0.0, -1.0, 1.0)
    q1 = q1 * sign[..., None]
    dot = np.abs(dot)
    theta = np.arccos(dot)
    small = theta < 1e-6
    denom = np.where(small, 1.0, np.sin(theta))
    s0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / denom)
    s1 = np.where(small, float(t), np.sin(float(t) * theta) / denom)
    out = s0[..., None] * q0 + s1[..., None] * q1
    return out / np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8, None)


def compute_world_joints(body_model: Any, body_quats: torch.Tensor, SE3: Any) -> torch.Tensor:
    batch, timesteps, _, _ = body_quats.shape
    device = body_quats.device
    dtype = body_quats.dtype
    betas = torch.zeros((batch, timesteps, 16), device=device, dtype=dtype)
    shaped = body_model.with_shape(betas)
    hands = torch.zeros((batch, timesteps, 30, 4), device=device, dtype=dtype)
    hands[..., 0] = 1.0
    local_quats = torch.cat([body_quats, hands], dim=2)
    posed = shaped.with_pose(T_world_root=SE3.identity(device, dtype).wxyz_xyz, local_quats=local_quats)
    return posed.Ts_world_joint


def quat_angle_deg_max_np(a_quats: np.ndarray, b_quats: np.ndarray) -> float:
    qa = a_quats[ROT_TRANSITION_JOINTS].astype(np.float64, copy=False)
    qb = b_quats[ROT_TRANSITION_JOINTS].astype(np.float64, copy=False)
    denom = np.clip(np.linalg.norm(qa, axis=-1) * np.linalg.norm(qb, axis=-1), 1e-8, None)
    dots = np.abs(np.sum(qa * qb, axis=-1) / denom)
    angles = np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))
    return float(np.max(angles))


def same_frame_discrepancy(
    a_quats: np.ndarray,
    b_quats: np.ndarray,
    a_positions: np.ndarray,
    b_positions: np.ndarray,
    a_contacts: np.ndarray,
    b_contacts: np.ndarray,
    *,
    SO3: Any | None = None,
) -> dict[str, float]:
    rot_max = quat_angle_deg_max_np(a_quats, b_quats)
    foot_max = float(np.linalg.norm(a_positions[FOOT_JOINTS, 4:7] - b_positions[FOOT_JOINTS, 4:7], axis=-1).max())
    contact_max = float(np.max(np.abs(a_contacts[FOOT_JOINTS] - b_contacts[FOOT_JOINTS])))
    return {
        "rotation_deg_max": rot_max,
        "foot_position_m_max": foot_max,
        "contact_diff_max": contact_max,
    }


def transition_cost(
    prev_quats: np.ndarray,
    curr_quats: np.ndarray,
    prev_positions: np.ndarray,
    curr_positions: np.ndarray,
    prev_contacts: np.ndarray,
    curr_contacts: np.ndarray,
    *,
    SO3: Any | None = None,
    rotation_weight: float,
    foot_position_weight: float,
    contact_jump_weight: float,
) -> dict[str, float]:
    rot_max = quat_angle_deg_max_np(prev_quats, curr_quats)
    foot_max = float(np.linalg.norm(prev_positions[FOOT_JOINTS, 4:7] - curr_positions[FOOT_JOINTS, 4:7], axis=-1).max())
    contact_max = float(np.max(np.abs(prev_contacts[FOOT_JOINTS] - curr_contacts[FOOT_JOINTS])))
    return {
        "rotation": rot_max * rotation_weight,
        "foot_position": foot_max * foot_position_weight,
        "contact": contact_max * contact_jump_weight,
        "rotation_deg_max": rot_max,
        "foot_position_m_max": foot_max,
        "contact_diff_max": contact_max,
    }


def build_blend_regions(
    switches: list[dict[str, Any]],
    windows: dict[int, dict[str, Any]],
    region_max: int,
) -> list[dict[str, int]]:
    regions = []
    for switch in switches:
        frame = int(switch["frame"])
        window_a = int(switch["from_window_start"])
        window_b = int(switch["to_window_start"])
        overlap_start = max(windows[window_a]["start"], windows[window_b]["start"])
        region_len = max(1, min(region_max, frame - overlap_start + 1))
        regions.append(
            {
                "window_a": window_a,
                "window_b": window_b,
                "start": frame - region_len + 1,
                "end": frame,
            }
        )

    for previous, current in zip(regions, regions[1:]):
        overlap_start = max(previous["start"], current["start"])
        overlap_end = min(previous["end"], current["end"])
        if previous["window_b"] == current["window_a"] and overlap_start <= overlap_end:
            boundary = (overlap_start + overlap_end) // 2
            previous["end"] = boundary
            current["start"] = boundary + 1
    return regions


def select_and_blend(
    windows: dict[int, dict[str, Any]],
    body_model: Any,
    SO3: Any,
    SE3: Any,
    params: dict[str, Any],
) -> dict[str, Any]:
    starts = sorted(windows)
    pose_arrays: dict[int, np.ndarray] = {}
    contact_probs: dict[int, np.ndarray] = {}
    fk: dict[int, np.ndarray] = {}
    for ws in starts:
        pose_arrays[ws] = windows[ws]["pose_quat"][0].astype(np.float32, copy=False)
        contact_probs[ws] = sigmoid(windows[ws]["head_contacts"][0]).astype(np.float32)
        pose_tensor = torch.from_numpy(windows[ws]["pose_quat"]).to(dtype=torch.float32)
        fk[ws] = compute_world_joints(body_model, pose_tensor, SE3).detach().cpu().numpy().astype(np.float32, copy=False)[0]

    total = max(w["end"] for w in windows.values())

    candidates: list[list[dict[str, Any]]] = []
    for frame in range(total):
        frame_cands = []
        for ws in starts:
            if windows[ws]["start"] <= frame < windows[ws]["end"]:
                frame_cands.append({"ws": ws, "local": frame - ws})
        candidates.append(frame_cands)

    uncovered = [frame for frame, frame_cands in enumerate(candidates) if not frame_cands]
    if uncovered:
        raise RuntimeError(f"no window candidate for frames {uncovered[:20]}")

    center = float(params["window_pos_center"])
    trusted_lo = float(params["window_pos_trusted_lo"])
    trusted_hi = float(params["window_pos_trusted_hi"])
    switch_penalty = float(params["switch_penalty"])
    same_frame_rotation_weight = float(params["same_frame_rotation_weight"])
    same_frame_foot_weight = float(params["same_frame_foot_weight"])
    same_frame_contact_weight = float(params["same_frame_contact_weight"])
    rotation_threshold = float(params["switch_rotation_threshold_deg"])
    foot_threshold = float(params["switch_foot_threshold_m"])

    def run_dp(rot_thresh: float | None, foot_thresh: float | None) -> tuple[list[list[float]], list[list[int]]]:
        costs: list[list[float]] = []
        predecessors: list[list[int]] = []
        for frame in range(total):
            frame_costs = []
            frame_predecessors = []
            for cand in candidates[frame]:
                ws = cand["ws"]
                local = cand["local"]
                quats = pose_arrays[ws][local]
                positions = fk[ws][local]
                contacts = contact_probs[ws][local]
                unary = (
                    float(params["window_pos_weight"]) * window_position_cost(local, center, trusted_lo, trusted_hi)
                    + float(params["contact_confidence_weight"]) * contact_confidence_cost(contacts)
                )
                if frame == 0:
                    frame_costs.append(unary)
                    frame_predecessors.append(-1)
                    continue

                best_cost = float("inf")
                best_prev = -1
                for prev_idx, prev_cand in enumerate(candidates[frame - 1]):
                    pws = prev_cand["ws"]
                    plocal = prev_cand["local"]
                    pquats = pose_arrays[pws][plocal]
                    ppositions = fk[pws][plocal]
                    pcontacts = contact_probs[pws][plocal]
                    trans = transition_cost(
                        pquats,
                        quats,
                        ppositions,
                        positions,
                        pcontacts,
                        contacts,
                        SO3=SO3,
                        rotation_weight=float(params["rotation_weight"]),
                        foot_position_weight=float(params["foot_position_weight"]),
                        contact_jump_weight=float(params["contact_jump_weight"]),
                    )
                    temporal_cost = trans["rotation"] + trans["foot_position"] + trans["contact"]
                    if pws == ws:
                        extra_cost = 0.0
                    else:
                        a_local = frame - pws
                        if a_local < 0 or a_local >= windows[pws]["end"] - windows[pws]["start"]:
                            extra_cost = float("inf")
                        else:
                            aq = pose_arrays[pws][a_local]
                            ap = fk[pws][a_local]
                            ac = contact_probs[pws][a_local]
                            sf = same_frame_discrepancy(aq, quats, ap, positions, ac, contacts, SO3=SO3)
                            gated = (
                                (rot_thresh is not None and sf["rotation_deg_max"] > rot_thresh)
                                or (foot_thresh is not None and sf["foot_position_m_max"] > foot_thresh)
                            )
                            if gated:
                                extra_cost = float("inf")
                            else:
                                extra_cost = (
                                    switch_penalty
                                    + same_frame_rotation_weight * sf["rotation_deg_max"]
                                    + same_frame_foot_weight * sf["foot_position_m_max"]
                                    + same_frame_contact_weight * sf["contact_diff_max"]
                                )
                    cand_cost = float("inf") if extra_cost == float("inf") else costs[frame - 1][prev_idx] + temporal_cost + extra_cost
                    if cand_cost < best_cost:
                        best_cost = cand_cost
                        best_prev = prev_idx
                frame_costs.append(best_cost + unary)
                frame_predecessors.append(best_prev)
            costs.append(frame_costs)
            predecessors.append(frame_predecessors)
        return costs, predecessors

    relaxations = [
        (rotation_threshold, foot_threshold, 1.0),
        (rotation_threshold * 2.0, foot_threshold * 2.0, 2.0),
        (rotation_threshold * 4.0, foot_threshold * 4.0, 4.0),
        (None, None, None),
    ]
    costs = None
    predecessors = None
    gate_relaxation = None
    for rot_thresh, foot_thresh, relax in relaxations:
        costs, predecessors = run_dp(rot_thresh, foot_thresh)
        if costs[-1] and min(costs[-1]) < float("inf"):
            gate_relaxation = relax
            break
    if costs is None or not costs[-1] or min(costs[-1]) == float("inf"):
        raise RuntimeError("DP found no finite path even with hard gates disabled.")

    path: list[dict[str, Any]] = []
    last_idx = int(min(range(len(costs[-1])), key=lambda i: costs[-1][i]))
    for frame in range(total - 1, -1, -1):
        cand = candidates[frame][last_idx]
        path.append({"global_frame": frame, "window_start": cand["ws"], "local_idx": cand["local"]})
        last_idx = predecessors[frame][last_idx]
    path.reverse()

    switches = []
    for i in range(1, total):
        prev = path[i - 1]
        curr = path[i]
        if prev["window_start"] == curr["window_start"]:
            continue
        A = prev["window_start"]
        B = curr["window_start"]
        pquats = pose_arrays[A][prev["local_idx"]]
        cquats = pose_arrays[B][curr["local_idx"]]
        ppos = fk[A][prev["local_idx"]]
        cpos = fk[B][curr["local_idx"]]
        pcontacts = contact_probs[A][prev["local_idx"]]
        ccontacts = contact_probs[B][curr["local_idx"]]
        jump = transition_cost(
            pquats,
            cquats,
            ppos,
            cpos,
            pcontacts,
            ccontacts,
            SO3=SO3,
            rotation_weight=1.0,
            foot_position_weight=1.0,
            contact_jump_weight=1.0,
        )
        a_local = i - A
        aq = pose_arrays[A][a_local]
        ap = fk[A][a_local]
        ac = contact_probs[A][a_local]
        sf = same_frame_discrepancy(aq, cquats, ap, cpos, ac, ccontacts, SO3=SO3)
        switches.append(
            {
                "frame": i,
                "from_window_start": A,
                "from_local_idx": prev["local_idx"],
                "to_window_start": B,
                "to_local_idx": curr["local_idx"],
                "temporal_lower_body_rotation_jump_deg": jump["rotation_deg_max"],
                "temporal_foot_position_jump_m": jump["foot_position_m_max"],
                "temporal_contact_jump": jump["contact_diff_max"],
                "same_frame_lower_body_rotation_diff_deg": sf["rotation_deg_max"],
                "same_frame_foot_position_diff_m": sf["foot_position_m_max"],
                "same_frame_contact_diff": sf["contact_diff_max"],
                "high_risk": i in HIGH_RISK_FRAMES,
            }
        )

    final_quats = np.stack([windows[p["window_start"]]["pose_quat"][0, p["local_idx"]] for p in path], axis=0).astype(np.float32)
    final_contacts = np.stack([contact_probs[p["window_start"]][p["local_idx"]] for p in path], axis=0).astype(np.float32)
    final_betas = np.stack([windows[p["window_start"]]["head_betas"][0, p["local_idx"]] for p in path], axis=0).astype(np.float32)

    region_max = int(params["transition_region_max"])
    blend_regions = build_blend_regions(switches, windows, region_max)
    for region in blend_regions:
        A = region["window_a"]
        B = region["window_b"]
        t0 = region["start"]
        t1 = region["end"]
        for t in range(t0, t1 + 1):
            alpha = 1.0 if t0 == t1 else (t - t0) / (t1 - t0)
            a_local = t - A
            b_local = t - B
            qA = windows[A]["pose_quat"][0, a_local].astype(np.float32)
            qB = windows[B]["pose_quat"][0, b_local].astype(np.float32)
            cA = contact_probs[A][a_local]
            cB = contact_probs[B][b_local]
            betaA = windows[A]["head_betas"][0, a_local].astype(np.float32)
            betaB = windows[B]["head_betas"][0, b_local].astype(np.float32)
            final_quats[t] = slerp_quat_np(qA, qB, alpha)
            final_contacts[t] = (1.0 - alpha) * cA + alpha * cB
            final_betas[t] = (1.0 - alpha) * betaA + alpha * betaB

    final_quats_t = torch.from_numpy(final_quats).unsqueeze(0).to(dtype=torch.float32)
    final_fk = compute_world_joints(body_model, final_quats_t, SE3).detach().cpu().numpy().astype(np.float32, copy=False)[0]
    for switch in switches:
        f = int(switch["frame"])
        qprev = final_quats[f - 1]
        qcurr = final_quats[f]
        jump = transition_cost(
            qprev,
            qcurr,
            final_fk[f - 1],
            final_fk[f],
            final_contacts[f - 1],
            final_contacts[f],
            SO3=SO3,
            rotation_weight=1.0,
            foot_position_weight=1.0,
            contact_jump_weight=1.0,
        )
        switch["blended_lower_body_rotation_jump_deg"] = jump["rotation_deg_max"]
        switch["blended_foot_position_jump_m"] = jump["foot_position_m_max"]
        switch["blended_contact_jump"] = jump["contact_diff_max"]

    compatibility = []
    for i in range(len(starts) - 1):
        A = starts[i]
        B = starts[i + 1]
        lo = max(A, B)
        hi = min(windows[A]["end"], windows[B]["end"])
        if hi <= lo:
            continue
        rot_vals = []
        foot_vals = []
        contact_vals = []
        for t in range(lo, hi):
            a_local = t - A
            b_local = t - B
            aq = pose_arrays[A][a_local]
            bq = pose_arrays[B][b_local]
            ap = fk[A][a_local]
            bp = fk[B][b_local]
            ac = contact_probs[A][a_local]
            bc = contact_probs[B][b_local]
            sf = same_frame_discrepancy(aq, bq, ap, bp, ac, bc, SO3=SO3)
            rot_vals.append(sf["rotation_deg_max"])
            foot_vals.append(sf["foot_position_m_max"])
            contact_vals.append(sf["contact_diff_max"])
        compatibility.append(
            {
                "window_a": A,
                "window_b": B,
                "overlap_start": lo,
                "overlap_end": hi,
                "mean_lower_body_rotation_diff_deg": float(np.mean(rot_vals)),
                "max_lower_body_rotation_diff_deg": float(np.max(rot_vals)),
                "mean_foot_position_diff_m": float(np.mean(foot_vals)),
                "max_foot_position_diff_m": float(np.max(foot_vals)),
                "mean_contact_diff": float(np.mean(contact_vals)),
                "max_contact_diff": float(np.max(contact_vals)),
            }
        )

    window_start_arr = np.asarray([p["window_start"] for p in path], dtype=np.int64)
    local_idx_arr = np.asarray([p["local_idx"] for p in path], dtype=np.int64)
    return {
        "body_quats": final_quats,
        "contacts": final_contacts,
        "betas": final_betas,
        "window_start": window_start_arr,
        "local_idx": local_idx_arr,
        "path": path,
        "switches": switches,
        "compatibility": compatibility,
        "gate_relaxation": gate_relaxation,
    }
