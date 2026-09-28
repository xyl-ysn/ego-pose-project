"""Bounded smoothing, fixed-chest grounding and leg stabilization for EgoAllo.

No hand targets enter these solvers. EgoGVAE supplies contact confidence only.
"""

import dataclasses
from functools import lru_cache
import time

import numpy as np
import torch

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.transforms import SE3, SO3

LEGS = [0, 1, 3, 4, 6, 7, 9, 10]


@lru_cache(maxsize=16)
def _joint_ids(indices, device):
    """Avoid host-to-device index copies inside captured GPU work."""
    return torch.tensor(indices, dtype=torch.long, device=device)


def viewer_motion(data):
    """Normalize rotation storage without changing the shape paired with the root."""
    result = dict(data)
    for key in ('body_quats', 'left_hand_quats', 'right_hand_quats'):
        result[key] = SO3.from_matrix(SO3(torch.as_tensor(data[key])).as_matrix()).wxyz.numpy()
    return result


def pose_with_legs(raw_pose, leg_quats):
    """Reuse fixed upper-body FK; only the two leg chains depend on the solve."""
    shaped = raw_pose.shaped_model
    parents = shaped.body_model.parent_indices
    leg_parents = tuple(-1 if parents[j] == -1 else LEGS.index(parents[j]) for j in LEGS)
    indices = _joint_ids(tuple(LEGS), leg_quats.device)
    transforms = raw_pose.Ts_world_joint.clone()
    transforms[:, indices] = fncsmpl.forward_kinematics(
        raw_pose.T_world_root, leg_quats, shaped.t_parent_joint[:, indices], leg_parents)
    quats = raw_pose.local_quats.clone()
    quats[:, indices] = leg_quats
    return dataclasses.replace(raw_pose, local_quats=quats, Ts_world_joint=transforms)


def pose_with_fixed_chest(raw_pose, quats):
    """Keep the original full FK calculation and anchor the chest in world space."""
    eye = quats.new_zeros(7)
    eye[0].fill_(1.)
    local = raw_pose.shaped_model.with_pose(eye, quats)
    chest = local.Ts_world_joint[:, 8]
    anchor = raw_pose.Ts_world_joint[:, 8]
    # Match the original full-FK calculation, including stored quaternion norms.
    inverse = SO3(chest[:, :4]).inverse().wxyz / chest[:, :4].square().sum(-1, keepdim=True)
    rotation = SO3(anchor[:, :4]) @ SO3(inverse)
    root = torch.cat([rotation.wxyz, anchor[:, 4:] - rotation @ chest[:, 4:]], -1)
    return local.with_new_T_world_root(root)


def safe_mean(value):
    return value.mean() if value.numel() else value.sum()


def _loss_and_backward(parameter, objective):
    """Replay fixed-shape GPU work; optimizer updates remain outside the graph."""
    parameter.grad = torch.zeros_like(parameter)

    def evaluate():
        parameter.grad.zero_()
        terms = objective()
        loss = sum(terms.values())
        loss.backward()
        return loss, terms

    if not parameter.is_cuda:
        return evaluate
    with torch.cuda.device(parameter.device):
        stream = torch.cuda.Stream(device=parameter.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                evaluate()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = evaluate()

    def replay():
        graph.replay()
        return outputs

    return replay


def foot_vertex_ids(model):
    membership = model.weights.argmax(-1)
    return [torch.where(torch.isin(membership, membership.new_tensor(ids)))[0]
            for ids in ([7, 10], [8, 11])]


def foot_bottom(pose, side_ids):
    ids = torch.cat(side_ids)
    shaped = pose.shaped_model
    model = shaped.body_model
    rotation = SO3(pose.local_quats).as_matrix()
    verts = shaped.verts_zero[..., ids, :] + fncsmpl.einsum(model.posedirs[ids],
        (rotation - torch.eye(3, device=rotation.device, dtype=rotation.dtype)).flatten(-3),
        '... verts j joints_times_9, ... joints_times_9 -> ... verts j')
    root, joints = pose.T_world_root, pose.Ts_world_joint
    # Build the used 3x4 matrices on-device; SE3.as_matrix copies a CPU constant.
    root_matrix = torch.cat([SO3(root[..., :4]).as_matrix(), root[..., 4:, None]], -1)
    joint_matrix = torch.cat([SO3(joints[..., :4]).as_matrix(), joints[..., 4:, None]], -1)
    cat = fncsmpl.broadcasting_cat
    world = fncsmpl.einsum(cat([root_matrix[..., None, :, :], joint_matrix], dim=-3), model.weights[ids],
        cat([verts[..., :, None, :] - cat([shaped.joints_zero.new_zeros(3), shaped.joints_zero[..., None, :, :]], dim=-2),
             verts.new_ones((*verts.shape[:-1], 1 + shaped.joints_zero.shape[-2], 1))], dim=-1),
        '... joints_p1 i j, verts joints_p1, ... verts joints_p1 j -> ... verts i')
    split = len(side_ids[0])
    return torch.stack([world[..., :split, 2].amin(-1), world[..., split:, 2].amin(-1)], -1)


def contact_probabilities(data, device):
    contact = torch.as_tensor(data['contacts'][0][:, [6, 7, 9, 10]], device=device).clamp(0, 1)
    return torch.stack([contact[:, [0, 2]].mean(-1), contact[:, [1, 3]].mean(-1)], -1)


@torch.no_grad()
def reference_contacts(reference, model, device, timestamps, floor_z):
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    q = np.concatenate([reference[k][0] for k in ('body_quats', 'left_hand_quats', 'right_hand_quats')], axis=1)
    pose = model.with_shape(tensor(reference['betas'][0])).with_pose(tensor(reference['Ts_world_root'][0]), tensor(q))
    bottom = foot_bottom(pose, foot_vertex_ids(model)) - floor_z
    speed = torch.diff(pose.Ts_world_joint[:, [9, 10], 4:6], dim=0).norm(dim=-1) / tensor(np.diff(timestamps))[:, None]
    speed = torch.cat([speed[:1], speed]) if len(speed) else bottom.new_zeros(bottom.shape)
    contact = contact_probabilities(reference, device)
    return contact * (contact > 0.7) * (bottom.abs() < 0.15) * (speed < 0.20)


def rotation_mean(q, radius, sigma):
    """Gaussian-weighted Markley means, matching scipy Rotation.mean."""
    t, joints, _ = q.shape
    outer = q[..., :, None] * q[..., None, :]
    channels = outer.permute(1, 2, 3, 0).reshape(1, joints * 16, t)
    x = torch.arange(-radius, radius + 1, device=q.device, dtype=q.dtype)
    weight = torch.exp(-0.5 * (x / sigma).square())
    weight /= weight.sum()
    filtered = torch.nn.functional.conv1d(torch.nn.functional.pad(channels, (radius, radius), mode='replicate'),
        weight.expand(joints * 16, 1, -1), groups=joints * 16)
    matrices = filtered.reshape(joints, 4, 4, t).permute(3, 0, 1, 2)
    return torch.linalg.eigh(matrices).eigenvectors[..., -1]


def limited_rotation(raw, target, alpha=1.0, max_degrees=5.0):
    delta = (SO3(raw).inverse() @ SO3(target)).log() * alpha
    delta *= (np.deg2rad(max_degrees) / delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)).clamp(max=1)
    return (SO3(raw) @ SO3.exp(delta)).wxyz


@torch.no_grad()
def smooth_body(data, model, device, max_degrees=5.0):
    if max_degrees <= 0:
        return dict(data)
    result = dict(data)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    q = tensor(data['body_quats'][0])
    q = limited_rotation(q, rotation_mean(q, 2, 1.0), max_degrees=max_degrees)
    result['body_quats'] = q.cpu().numpy()[None]
    result['body_rotmats'] = SO3(q).as_matrix().cpu().numpy()[None]
    hands = np.concatenate([data['left_hand_quats'][0], data['right_hand_quats'][0]], axis=1)
    pose = model.with_shape(tensor(data['betas'][0])).with_pose(
        SE3.identity(device, q.dtype).wxyz_xyz, torch.cat([q, tensor(hands)], 1))
    result['Ts_world_root'] = fncsmpl_extensions.get_T_world_root_from_cpf_pose(pose, tensor(data['Ts_world_cpf'])).cpu().numpy()[None]
    return result


def _filter_grounding_correction(delta, sigma):
    if sigma <= 0 or len(delta) == 1:
        return delta
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=delta.device, dtype=delta.dtype)
    kernel = torch.exp(-.5 * (x / sigma).square())
    kernel /= kernel.sum()
    channels = delta.reshape(len(delta), -1).T[None]
    return torch.nn.functional.conv1d(torch.nn.functional.pad(channels, (radius, radius), mode='replicate'),
        kernel.expand(channels.shape[1], 1, -1), groups=channels.shape[1])[0].T.reshape_as(delta)


def validate_grounding_options(spine_limit, leg_limit, iterations, correction_sigma_seconds,
                               temporal_weight_scale=1., floor_z=0.):
    values = [spine_limit, leg_limit, iterations, correction_sigma_seconds, temporal_weight_scale, floor_z]
    if (not np.isfinite(values).all() or not 0 <= spine_limit <= 15 or not 0 <= leg_limit <= 20
            or iterations < 0 or int(iterations) != iterations or correction_sigma_seconds < 0
            or temporal_weight_scale < 0):
        raise ValueError('Invalid grounding correction budget or floor height')


def ground_below_chest(base, model, device, contact, spine_limit=8., leg_limit=12., iterations=200,
                       correction_sigma_seconds=.08, temporal_weight_scale=1., floor_z=0., foot_xy_limit=.05, foot_angle_limit=8.):
    """Move pelvis/lower spine and legs while preserving world chest, head and arms."""
    validate_grounding_options(spine_limit, leg_limit, iterations, correction_sigma_seconds,
                               temporal_weight_scale, floor_z)
    if not np.isfinite([foot_xy_limit, foot_angle_limit]).all() or foot_xy_limit <= 0 or foot_angle_limit <= 0:
        raise ValueError('Expected positive finite support correction limits')
    if iterations == 0:
        return dict(base), {'enabled': False}
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    times = np.asarray(base['timestamps'], dtype=np.float64)
    if (times.ndim != 1 or not len(times) or len(times) != base['body_quats'].shape[1]
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('Grounding requires finite, increasing timestamps matching the motion')
    contact = tensor(contact)
    if (contact.shape != (len(times), 2) or not torch.isfinite(contact).all()
            or torch.any((contact < 0) | (contact > 1))):
        raise ValueError('Grounding requires finite contact probabilities [T,2] in [0,1]')
    q0 = tensor(np.concatenate([base[k][0] for k in ('body_quats', 'left_hand_quats', 'right_hand_quats')], 1))
    with torch.no_grad():
        before = model.with_shape(tensor(base['betas'][0])).with_pose(tensor(base['Ts_world_root'][0]), q0)
    ids = _joint_ids(tuple(LEGS + [2, 5, 8]), q0.device)
    support_ids = _joint_ids((6, 7, 9, 10), q0.device)
    original_root = before.T_world_root
    original_feet = before.Ts_world_joint[:, [6, 7, 9, 10], 4:]
    original_foot_rotation = SO3(before.Ts_world_joint[:, [6, 7, 9, 10], :4]).normalize()
    foot_ids = foot_vertex_ids(model)
    rate = tensor((1 / 30) / np.diff(times))[:, None, None]
    sigma = correction_sigma_seconds / float(np.median(np.diff(times))) if len(times) > 1 else 0.
    limits = tensor([leg_limit] * 8 + [spine_limit] * 3)[None, :, None] * np.pi / 180
    delta = torch.nn.Parameter(torch.zeros(len(q0), len(ids), 3, device=device))
    optimizer = torch.optim.Adam([delta], lr=.012)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=.0003)
    best, saved, started = float('inf'), delta.detach().clone(), time.perf_counter()
    pair = torch.minimum(contact[:-1], contact[1:])
    history = []

    def current():
        correction = _filter_grounding_correction(delta, sigma)
        q = q0.clone()
        q[:, ids] = (SO3(q0[:, ids]) @ SO3.exp(correction)).wxyz
        return pose_with_fixed_chest(before, q), correction

    def weighted(value, weight):
        return (value * weight).sum() / weight.sum().clamp(min=1)

    def objective():
        p, correction = current()
        feet = p.Ts_world_joint[:, support_ids, 4:]
        bottom = foot_bottom(p, foot_ids) - floor_z
        root_delta = p.T_world_root[:, 4:] - original_root[:, 4:]
        root_angle = (SO3(original_root[:, :4]).normalize().inverse() @ SO3(p.T_world_root[:, :4]).normalize()).log()
        angular_step = (SO3(p.local_quats[:-1, ids]).normalize().inverse() @ SO3(p.local_quats[1:, ids]).normalize()).log()
        foot_angle = (original_foot_rotation.inverse() @ SO3(p.Ts_world_joint[:, support_ids, :4]).normalize()).log().norm(dim=-1)
        return dict(
            foot_xy_guard=1000000 * torch.relu((feet[..., :2] - original_feet[..., :2]).norm(dim=-1) - foot_xy_limit * .8).square().mean(),
            foot_angle_guard=1000000 * torch.relu(foot_angle - np.deg2rad(foot_angle_limit) * .8).square().mean(),
            penetration=60000 * torch.relu(.001 - bottom).square().mean(),
            contact_height=10000 * weighted(bottom.square(), contact),
            own_foot_xy=3000 * (feet[..., :2] - original_feet[..., :2]).square().mean(),
            foot_skating=20000 * weighted((torch.diff(feet[:, 2:], dim=0) * rate).square().sum(-1), pair),
            foot_acceleration=15000 * temporal_weight_scale * safe_mean(torch.diff(torch.diff(feet, dim=0) * rate, dim=0).square()),
            angular_acceleration=800 * temporal_weight_scale * safe_mean(torch.diff(angular_step * rate, dim=0).square()),
            root_acceleration=50000 * temporal_weight_scale * safe_mean(torch.diff(torch.diff(p.T_world_root[:, 4:], dim=0) * rate[:, 0], dim=0).square()),
            pose_prior=200 * correction[:, :8].square().mean() + 400 * correction[:, 8:].square().mean(),
            root_prior=500 * root_delta.square().mean() + 10 * root_angle.square().mean(),
            correction_velocity=100 * safe_mean((torch.diff(correction, dim=0) * rate).square()),
            root_limit=100000 * (torch.relu(root_delta[:, :2].norm(dim=-1) - .02).square().mean()
                + torch.relu(root_delta[:, 2].abs() - .15).square().mean()
                + torch.relu(root_angle.norm(dim=-1) - np.deg2rad(3)).square().mean()),
        )
    evaluate = _loss_and_backward(delta, objective)
    for step in range(iterations + 1):
        loss, terms = evaluate()
        if step % 25 == 0 or step == iterations:
            value = float(loss.detach())
            if not np.isfinite(value):
                raise FloatingPointError('Non-finite grounding loss')
            history.append(dict(step=step, loss=value, **{k: float(v.detach()) for k, v in terms.items()}))
            if value < best:
                best, saved = value, delta.detach().clone()
        if step == iterations:
            break
        optimizer.step()
        schedule.step()
        with torch.no_grad():
            delta *= (limits / delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)).clamp(max=1)
    with torch.no_grad():
        delta.copy_(saved)
        trajectory_scale = 1.
        for _ in range(40):
            p, correction = current()
            xy_change = (p.Ts_world_joint[:, [6, 7, 9, 10], 4:6] - original_feet[..., :2]).norm(dim=-1)
            foot_angle = (original_foot_rotation.inverse() @ SO3(p.Ts_world_joint[:, [6, 7, 9, 10], :4]).normalize()).log().norm(dim=-1)
            if float(xy_change.max()) <= foot_xy_limit and float(foot_angle.max()) <= np.deg2rad(foot_angle_limit):
                break
            # One clip-wide attenuation cannot introduce a framewise switching boundary.
            delta.mul_(.8)
            trajectory_scale *= .8
        else:
            raise RuntimeError('Could not satisfy support correction limits')
        keep = [8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] + list(range(21, 51))
        displacement = (p.Ts_world_joint[:, keep, 4:] - before.Ts_world_joint[:, keep, 4:]).norm(dim=-1)
        if not torch.isfinite(p.Ts_world_joint).all() or float(displacement.max()) >= 1e-5:
            raise FloatingPointError('Grounding changed the fixed upper-body world positions')
        result = dict(base, body_quats=p.local_quats[:, :21].cpu().numpy()[None],
                      body_rotmats=SO3(p.local_quats[:, :21]).as_matrix().cpu().numpy()[None],
                      Ts_world_root=p.T_world_root.cpu().numpy()[None],
                      refinement_mode=np.asarray('support_guarded_grounding'))

        def stats(value):
            return dict(mean=float(value.mean()), p95=float(torch.quantile(value, .95)), max=float(value.max()))

        return result, dict(enabled=True, trajectory_scale=trajectory_scale,
            execution_backend='cuda_graph' if delta.is_cuda else 'torch_cpu',
            max_foot_xy_displacement_cm=float(xy_change.max()) * 100,
            max_foot_world_rotation_deg=float(torch.rad2deg(foot_angle).max()), elapsed_sec=time.perf_counter() - started, history=history,
            iterations=iterations, floor_z=floor_z, spine_limit_deg=spine_limit, leg_limit_deg=leg_limit,
            correction_sigma_seconds=correction_sigma_seconds,
            max_upper_world_displacement_cm=float(displacement.max()) * 100,
            max_spine_correction_deg=float(torch.rad2deg(correction[:, 8:].norm(dim=-1)).max()),
            max_leg_correction_deg=float(torch.rad2deg(correction[:, :8].norm(dim=-1)).max()),
            root_displacement_cm=stats((p.T_world_root[:, 4:] - original_root[:, 4:]).norm(dim=-1) * 100),
            root_angle_deg=stats(torch.rad2deg((SO3(original_root[:, :4]).normalize().inverse()
                                              @ SO3(p.T_world_root[:, :4]).normalize()).log().norm(dim=-1))))


def solve_legs(data, model, device, contact, timestamps, floor_z=0.0, iterations=250,
               max_correction_degrees=3.0, support_reference=None, foot_xy_limit=.05, foot_angle_limit=8.):
    if not np.isfinite(max_correction_degrees) or not 0 <= max_correction_degrees <= 10:
        raise ValueError('Leg correction must be between 0 and 10 degrees')
    if not np.isfinite([foot_xy_limit, foot_angle_limit]).all() or foot_xy_limit <= 0 or foot_angle_limit <= 0:
        raise ValueError('Expected positive finite support correction limits')
    support_reference = data if support_reference is None else support_reference
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    qraw = tensor(np.concatenate([data['body_quats'][0], data['left_hand_quats'][0], data['right_hand_quats'][0]], axis=1))
    root = tensor(data['Ts_world_root'][0])
    shaped = model.with_shape(tensor(data['betas'][0]))
    side_ids = foot_vertex_ids(model)
    with torch.no_grad():
        raw_pose = shaped.with_pose(root, qraw)
        raw_pos = raw_pose.Ts_world_joint[..., 4:]
        pair = torch.minimum(contact[1:], contact[:-1])
    with torch.no_grad():
        support_q = tensor(np.concatenate([support_reference[k][0] for k in ('body_quats', 'left_hand_quats', 'right_hand_quats')], 1))
        support_pose = model.with_shape(tensor(support_reference['betas'][0])).with_pose(tensor(support_reference['Ts_world_root'][0]), support_q)
        support_xy = support_pose.Ts_world_joint[:, [6, 7, 9, 10], 4:6]
        support_rotation = SO3(support_pose.Ts_world_joint[:, [6, 7, 9, 10], :4]).normalize()
    if iterations <= 0 or max_correction_degrees == 0:
        return dict(data), {"enabled": False}
    dt = tensor(np.diff(timestamps))
    rate = (1.0 / 30.0) / dt[:, None, None]
    delta = torch.nn.Parameter(torch.zeros((len(root), len(LEGS), 3), device=device))
    optimizer = torch.optim.Adam([delta], lr=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=0.0003)
    best_score, best_delta, history = float('inf'), delta.detach().clone(), []
    started = time.perf_counter()
    leg_ids = _joint_ids(tuple(LEGS), qraw.device)
    support_ids = _joint_ids((6, 7, 9, 10), qraw.device)

    def weighted(value, weights):
        return (value * weights).sum() / weights.sum().clamp(min=1)

    def current():
        pose = pose_with_legs(raw_pose, (SO3(qraw[:, leg_ids]) @ SO3.exp(delta)).wxyz)
        return pose.local_quats, pose

    def objective():
        q, pose = current()
        bottom = foot_bottom(pose, side_ids) - floor_z
        feet = pose.Ts_world_joint[:, 9:11, 4:]
        angular_step = (SO3(q[:-1, leg_ids]).inverse() @ SO3(q[1:, leg_ids])).log()
        support_change = (pose.Ts_world_joint[:, support_ids, 4:6] - support_xy).norm(dim=-1)
        support_angle = (support_rotation.inverse() @ SO3(pose.Ts_world_joint[:, support_ids, :4]).normalize()).log().norm(dim=-1)
        return {
            'foot_xy_guard': 1000000 * torch.relu(support_change - foot_xy_limit * .8).square().mean(),
            'foot_angle_guard': 1000000 * torch.relu(support_angle - np.deg2rad(foot_angle_limit) * .8).square().mean(),
            'pose_prior': delta.square().sum(-1).mean(),
            'penetration': 3000 * torch.relu(-bottom).square().mean(),
            'contact_height': 900 * weighted(bottom.square(), contact),
            'foot_skating': 30000 * weighted((torch.diff(feet, dim=0) * rate).square().sum(-1), pair),
            'own_foot_xy_prior': 1000 * (feet[..., :2] - raw_pos[:, 9:11, :2]).square().sum(-1).mean(),
            'angular_acceleration': 1200 * safe_mean(torch.diff(angular_step * rate, dim=0).square()),
            'foot_acceleration': 15000 * safe_mean(torch.diff(torch.diff(feet, dim=0) * rate, dim=0).square()),
            'delta_velocity': 60 * safe_mean((torch.diff(delta, dim=0) * rate).square()),
        }
    evaluate = _loss_and_backward(delta, objective)
    for step in range(iterations + 1):
        loss, terms = evaluate()
        if step % 25 == 0 or step == iterations:
            score = float(loss.detach())
            if not np.isfinite(score):
                raise FloatingPointError('Non-finite lower-body objective')
            if score < best_score:
                best_score, best_delta = score, delta.detach().clone()
            history.append({'step': step, 'loss': score, **{k: float(v.detach()) for k, v in terms.items()}})
        if step == iterations:
            break
        optimizer.step()
        schedule.step()
        with torch.no_grad():
            # A fixed pelvis and inconsistent floor must not force a new stance.
            delta *= (np.deg2rad(max_correction_degrees) / delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)).clamp(max=1)
    with torch.no_grad():
        delta.copy_(best_delta)
        trajectory_scale = 1.
        for _ in range(40):
            q, pose = current()
            support_change = (pose.Ts_world_joint[:, [6, 7, 9, 10], 4:6] - support_xy).norm(dim=-1)
            support_angle = (support_rotation.inverse() @ SO3(pose.Ts_world_joint[:, [6, 7, 9, 10], :4]).normalize()).log().norm(dim=-1)
            if float(support_change.max()) <= foot_xy_limit and float(support_angle.max()) <= np.deg2rad(foot_angle_limit):
                break
            delta.mul_(.8)
            trajectory_scale *= .8
        else:
            raise RuntimeError('Stabilization exceeds support limits')
        keep = [j for j in range(51) if j not in LEGS]
        torch.testing.assert_close(q[:, keep], qraw[:, keep], rtol=0, atol=0)
        torch.testing.assert_close(pose.Ts_world_joint[:, [14, 19, 20]], raw_pose.Ts_world_joint[:, [14, 19, 20]], rtol=0, atol=1e-6)
        result = dict(data)
        result['body_quats'] = q[:, :21].cpu().numpy()[None]
        result['body_rotmats'] = SO3(q[:, :21]).as_matrix().cpu().numpy()[None]
        result['refinement_mode'] = np.asarray('legs_only_contact_no_hand_ik')
        for key in ('Ts_world_root', 'Ts_world_cpf', 'betas', 'left_hand_quats', 'right_hand_quats'):
            np.testing.assert_array_equal(result[key], data[key])
        foot_xy = pose.Ts_world_joint[:, [9, 10], 4:6]
        raw_foot_xy = raw_pos[:, [9, 10], :2]
        width_change = (foot_xy[:, 0] - foot_xy[:, 1]).norm(dim=-1) - (raw_foot_xy[:, 0] - raw_foot_xy[:, 1]).norm(dim=-1)
        return result, {'trajectory_scale': trajectory_scale,
            'execution_backend': 'cuda_graph' if delta.is_cuda else 'torch_cpu',
            'max_support_xy_cm': float(support_change.max()) * 100,
            'max_support_rotation_deg': float(torch.rad2deg(support_angle).max()),
            'elapsed_sec': time.perf_counter() - started, 'history': history,
            'max_allowed_leg_correction_deg': float(max_correction_degrees),
            'max_leg_correction_deg': float(torch.rad2deg(delta.norm(dim=-1)).max()),
            'max_foot_xy_displacement_m': float((foot_xy - raw_foot_xy).norm(dim=-1).max()),
            'max_stance_width_increase_m': float(width_change.clamp(min=0).max()),
            'unchanged_root_torso_arms_fingers': True, 'wrist_world_displacement_m': float((pose.Ts_world_joint[:, [19, 20], 4:] - raw_pos[:, [19, 20]]).norm(dim=-1).max())}
