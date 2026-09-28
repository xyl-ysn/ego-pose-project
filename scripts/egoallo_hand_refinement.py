"""Small, temporally coherent arm corrections with an observation-based veto."""

import time

import numpy as np
import torch

from egoallo.transforms import SE3, SO3

OPENPOSE_TO_MANO = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
ARM_JOINTS = [[15, 17], [16, 18]]


def temporal_correction(delta, confidence, timestamps, max_degrees):
    """Cap before averaging so the cap cannot create high-frequency corners."""
    delta = delta * (np.deg2rad(max_degrees) / delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)).clamp(max=1)
    fps = 1. / float(np.median(np.diff(timestamps))) if len(delta) > 1 else 30.
    radius = max(1, int(round(.4 * fps)))
    x = torch.arange(-radius, radius + 1, device=delta.device, dtype=delta.dtype)
    kernel = torch.exp(-.5 * (x / (.2 * fps)).square())

    def filtered(values):
        channels = values.shape[-1]
        return torch.nn.functional.conv1d(values.T[None], kernel.expand(channels, 1, -1),
            padding=radius, groups=channels)[0].T

    value = filtered((delta * confidence[..., None, None]).reshape(len(delta), -1)).reshape(delta.shape)
    return value / filtered(confidence).clamp(min=1)[..., None, None]


def error_summary(error):
    return dict(mean_cm=float(error.mean() * 100), p95_cm=float(torch.quantile(error.flatten(), .95) * 100))


def no_worse(before, after):
    tolerance = 1e-7  # Floating-point FK tolerance, in meters.
    return bool((after.mean() <= before.mean() + tolerance)
                & (torch.quantile(after.flatten(), .95) <= torch.quantile(before.flatten(), .95) + tolerance))


@torch.no_grad()
def refine_hands(data, model, detections, device, max_degrees=2.):
    """One damped linearized IK step. Never move the root, torso, legs or fingers.

    The veto uses observed agreement, not ground truth or collision detection.
    A rejected side is restored for the whole clip to avoid per-frame switches.
    Unreachable targets do not drive corrections; source detections stay intact.
    """
    if not np.isfinite(max_degrees) or not 0 <= max_degrees <= 5:
        raise ValueError('hand-refinement-max-deg must be finite and between 0 and 5')
    if max_degrees == 0:
        return dict(data), dict(enabled=False, any_accepted=False)
    started = time.perf_counter()
    timestamps = np.asarray(data['timestamps'], dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Hand refinement requires finite, increasing timestamps')
    tensor = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
    q = tensor(np.concatenate([data[k][0] for k in ('body_quats', 'left_hand_quats', 'right_hand_quats')], 1))
    if q.shape != (len(timestamps), 51, 4):
        raise ValueError('Hand refinement expects one matching SMPL-H motion sample')
    confidence = torch.zeros((len(q), 2), device=device)
    targets = torch.zeros((len(q), 2, 3), device=device)
    observations = []
    cpf = tensor(data['Ts_world_cpf'])
    for side, name in enumerate(('left', 'right')):
        concat = getattr(detections, 'detections_' + name + '_concat')
        if concat is None:
            observations.append(None)
            continue
        idx = concat.indices.to(device)
        if bool(((idx < 0) | (idx >= len(q))).any()) or len(torch.unique(idx)) != len(idx):
            raise ValueError('Hand observation indices must be unique and within the motion')
        weight = concat.wrist_pose_weight.to(device).clamp(0, 1)
        keypoints = concat.keypoints_3d[:, OPENPOSE_TO_MANO].to(device)
        if not bool(torch.isfinite(keypoints).all() & torch.isfinite(weight).all()):
            raise ValueError('Non-finite hand observation')
        world_cam = SE3(cpf[idx]) @ SE3(detections.T_cpf_cam.to(device))
        keypoints = SE3(world_cam.wxyz_xyz[:, None]) @ keypoints
        confidence[idx, side], targets[idx, side] = weight, keypoints[:, 0]
        full_hand = (weight > 0) & (concat.hand_reproj_weight.to(device) > 0)
        observations.append((idx, keypoints, weight > 0, full_hand))
    if not bool((confidence > 0).any()):
        return dict(data), dict(enabled=True, any_accepted=False, reason='no_observations')

    shaped = model.with_shape(tensor(data['betas'][0]))
    root = tensor(data['Ts_world_root'][0])
    before = shaped.with_pose(root, q)
    world = before.Ts_world_joint
    p = world[..., 4:]
    upper_length = (p[:, [17, 18]] - p[:, [15, 16]]).norm(dim=-1)
    lower_length = (p[:, [19, 20]] - p[:, [17, 18]]).norm(dim=-1)
    target_distance = (targets - p[:, [15, 16]]).norm(dim=-1)
    reachable = (target_distance < upper_length + lower_length - .005)
    reachable &= target_distance > (upper_length - lower_length).abs() + .005
    rejected_counts = ((confidence > 0) & ~reachable).sum(0)
    # Keep unreachable observations in the error veto, but never fit them.
    confidence *= reachable
    axes = SO3(world[:, ARM_JOINTS, :4]).as_matrix()
    lever = p[:, [19, 20], None] - p[:, ARM_JOINTS]
    jacobian = torch.cross(axes.transpose(-1, -2), lever[..., None, :].expand_as(axes), dim=-1)
    jacobian = jacobian.reshape(len(q), 2, 6, 3).transpose(-1, -2)
    error = targets - p[:, [19, 20]]
    error *= (.03 / error.norm(dim=-1, keepdim=True).clamp(min=1e-8)).clamp(max=1)
    delta = (jacobian.transpose(-1, -2) @ torch.linalg.solve(
        jacobian @ jacobian.transpose(-1, -2) + .01 * torch.eye(3, device=device), error[..., None])).squeeze(-1)
    delta = temporal_correction(delta.reshape(len(q), 2, 2, 3), confidence, timestamps, max_degrees)
    corrected = q.clone()
    corrected[:, ARM_JOINTS] = (SO3(q[:, ARM_JOINTS]) @ SO3.exp(delta)).wxyz
    active = delta.abs().sum((-1, -2)) > 0
    after = shaped.with_pose(root, corrected)
    wrist = (SO3(after.Ts_world_joint[:, [17, 18], :4]).inverse() @ SO3(world[:, [19, 20], :4])).wxyz
    corrected[:, [19, 20]] = torch.where(active[..., None], wrist, q[:, [19, 20]])
    for side in range(2):
        ids = ARM_JOINTS[side]
        corrected[:, ids] = torch.where(active[:, side, None, None], corrected[:, ids], q[:, ids])
    after = shaped.with_pose(root, corrected)

    sides = {}
    for side, name in enumerate(('left', 'right')):
        obs = observations[side]
        accepted = False
        details = dict(accepted=False, reason='no_observations')
        if obs is not None and bool(obs[2].any()):
            idx, target, valid, full_hand = obs
            ids = [19 + side] + list(range(21 + side * 15, 36 + side * 15))
            old = (world[idx][:, ids, 4:] - target).norm(dim=-1)
            new = (after.Ts_world_joint[idx][:, ids, 4:] - target).norm(dim=-1)
            has_reachable = bool((confidence[idx, side] > 0).any())
            accepted = has_reachable and no_worse(old[valid, 0], new[valid, 0])
            details = dict(observations=int(valid.sum()), wrist_before=error_summary(old[valid, 0]),
                           wrist_proposed=error_summary(new[valid, 0]), full_hand_checked=bool(full_hand.any()))
            if bool(full_hand.any()):
                accepted = accepted and no_worse(old[full_hand], new[full_hand])
                details.update(hand_before=error_summary(old[full_hand]), hand_proposed=error_summary(new[full_hand]))
            details.update(accepted=accepted, reason=('accepted' if accepted else
                'observed_error_regression' if has_reachable else 'no_reachable_observations'))
        if not accepted:
            ids = ARM_JOINTS[side] + [19 + side]
            corrected[:, ids] = q[:, ids]
        sides[name] = details
        sides[name]['unreachable_observations'] = int(rejected_counts[side])
    result = dict(data)
    result['body_quats'] = corrected[:, :21].cpu().numpy()[None]
    result['body_rotmats'] = SO3(corrected[:, :21]).as_matrix().cpu().numpy()[None]
    if any(s['accepted'] for s in sides.values()):
        result['refinement_mode'] = np.asarray('bounded_hand_refinement')
    return result, dict(enabled=True, any_accepted=any(s['accepted'] for s in sides.values()),
        method='single_damped_step_with_whole_side_veto', max_joint_correction_deg=max_degrees,
        max_target_step_m=.03, temporal_sigma_seconds=.2, sides=sides,
        elapsed_sec=time.perf_counter() - started)
