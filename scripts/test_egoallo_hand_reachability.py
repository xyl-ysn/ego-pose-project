"""Run with the pipeline Python: python scripts/test_egoallo_hand_reachability.py."""

import dataclasses
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "exp" / "egoallo" / "src"))

from egoallo.transforms import SE3, SO3


@dataclasses.dataclass
class Hand:
    indices: torch.Tensor
    keypoints_3d: torch.Tensor
    wrist_pose_weight: torch.Tensor
    hand_reproj_weight: torch.Tensor
    hand_pose_weight: torch.Tensor
    constraint_weight: torch.Tensor


@dataclasses.dataclass
class Detections:
    T_cpf_cam: torch.Tensor
    detections_left_concat: Hand | None
    detections_right_concat: Hand | None


downweight = runpy.run_path(str(SCRIPT_DIR / "07_run_egoallo_refine_with_prior.py"))[
    "downweight_unreachable_hands"
]


def rejects(detections, joints, cpf, **kwargs):
    try:
        downweight(detections, joints, cpf, **kwargs)
    except ValueError:
        return
    raise AssertionError("Invalid reachability input should fail explicitly")


def make_hand(shoulders, distances):
    wrists = shoulders.clone()
    wrists[:, 0] += torch.tensor(distances)
    return Hand(
        indices=torch.arange(len(wrists)),
        keypoints_3d=wrists[:, None].expand(-1, 21, -1).clone(),
        wrist_pose_weight=torch.full((len(wrists),), .8),
        hand_reproj_weight=torch.full((len(wrists),), .6),
        hand_pose_weight=torch.full((len(wrists),), .7),
        constraint_weight=torch.full((len(wrists),), .9),
    )


timesteps = 5
joints = torch.zeros((timesteps, 51, 3))
joints[:, [15, 16], 2] = 1.
joints[:, 15, 1], joints[:, 16, 1] = .2, -.2
for shoulder, elbow, wrist in ((15, 17, 19), (16, 18, 20)):
    joints[:, elbow] = joints[:, shoulder] + torch.tensor([.3, 0., 0.])
    joints[:, wrist] = joints[:, shoulder] + torch.tensor([.6, 0., 0.])
left = make_hand(joints[:, 15], .6 * np.array([.9, 1.01, 1.175, 1.4, 1.6]))
right = make_hand(joints[:, 16], [.3] * timesteps)
right.wrist_pose_weight[0] = 0.
right.hand_reproj_weight[0] = 0.
identity = torch.tensor([1., 0., 0., 0., 0., 0., 0.])
detections = Detections(identity.clone(), left, right)
cpf = identity.expand(timesteps, -1).numpy().copy()
original = {
    side: {field.name: getattr(hand, field.name).clone() for field in dataclasses.fields(hand)}
    for side, hand in (("left", left), ("right", right))
}

result, report = downweight(detections, joints, cpf)
x = torch.tensor([0., .01 / .35, .175 / .35, 1., 1.])
smoothstep = (1 - x).square() * (1 + 2 * x)
expected_scale = .05 + .95 * smoothstep
new_left, new_right = result.detections_left_concat, result.detections_right_concat
torch.testing.assert_close(new_left.wrist_pose_weight, left.wrist_pose_weight * expected_scale)
torch.testing.assert_close(new_left.hand_reproj_weight, left.hand_reproj_weight * expected_scale)
assert bool((torch.diff(new_left.wrist_pose_weight) <= 0).all())
assert bool((new_left.wrist_pose_weight[-2:] > 0).all())
torch.testing.assert_close(new_left.wrist_pose_weight[-2:], .05 * left.wrist_pose_weight[-2:])
torch.testing.assert_close(new_left.hand_reproj_weight[-2:], .05 * left.hand_reproj_weight[-2:])
assert 0 < float(new_left.wrist_pose_weight[1]) < float(left.wrist_pose_weight[1])
torch.testing.assert_close(new_right.wrist_pose_weight, right.wrist_pose_weight)
torch.testing.assert_close(new_right.hand_reproj_weight, right.hand_reproj_weight)
assert result is not detections and new_left is not left
assert report["min_weight"] == .05
assert report["left"]["detections"] == 5
assert report["left"]["downweighted"] == 4 and report["left"]["disabled"] == 0
assert report["left"]["at_min_weight"] == 2
assert report["right"]["downweighted"] == report["right"]["disabled"] == 0
for side, source, output in (("left", left, new_left), ("right", right, new_right)):
    for field, value in original[side].items():
        torch.testing.assert_close(getattr(source, field), value)
        if field not in ("wrist_pose_weight", "hand_reproj_weight"):
            torch.testing.assert_close(getattr(output, field), value)

for minimum in (.2, 1.):
    custom, custom_report = downweight(detections, joints, cpf, min_weight=minimum)
    scale = minimum + (1 - minimum) * smoothstep
    torch.testing.assert_close(custom.detections_left_concat.wrist_pose_weight, left.wrist_pose_weight * scale)
    torch.testing.assert_close(custom.detections_left_concat.hand_reproj_weight, left.hand_reproj_weight * scale)
    torch.testing.assert_close(custom.detections_right_concat.wrist_pose_weight, right.wrist_pose_weight)
    torch.testing.assert_close(custom.detections_right_concat.hand_reproj_weight, right.hand_reproj_weight)
    assert custom_report["min_weight"] == minimum
    assert custom_report["left"]["disabled"] == 0
    if minimum == 1.:
        assert custom_report["left"]["downweighted"] == 0

# A positive reachability floor must not revive either upstream residual.
zero_wrist, zero_reproj = left.wrist_pose_weight.clone(), left.hand_reproj_weight.clone()
zero_wrist[-1], zero_reproj[-2] = 0., 0.
zeroed = dataclasses.replace(left, wrist_pose_weight=zero_wrist, hand_reproj_weight=zero_reproj)
zero_result, _ = downweight(dataclasses.replace(detections, detections_left_concat=zeroed), joints, cpf)
assert zero_result.detections_left_concat.wrist_pose_weight[-1] == 0
assert zero_result.detections_left_concat.hand_reproj_weight[-2] == 0

# Change the world's origin and orientation together, keeping camera detections fixed.
rotation = SO3.from_z_radians(torch.tensor(.8))
world_change = SE3.from_rotation_and_translation(rotation, torch.tensor([2., -1., .4]))
transformed_joints = world_change @ joints
transformed_cpf = (world_change @ SE3(torch.from_numpy(cpf))).wxyz_xyz.numpy()
transformed, _ = downweight(detections, transformed_joints, transformed_cpf)
for side in ("left", "right"):
    expected = getattr(result, "detections_" + side + "_concat")
    actual = getattr(transformed, "detections_" + side + "_concat")
    torch.testing.assert_close(actual.wrist_pose_weight, expected.wrist_pose_weight)
    torch.testing.assert_close(actual.hand_reproj_weight, expected.hand_reproj_weight)

empty, empty_report = downweight(Detections(identity.clone(), None, None), joints, cpf)
assert empty.detections_left_concat is None and empty.detections_right_concat is None
assert empty_report["left"]["detections"] == empty_report["right"]["detections"] == 0
for invalid_ratio in (0., -.1, float("nan"), float("inf")):
    rejects(detections, joints, cpf, fade_ratio=invalid_ratio)
for invalid_minimum in (0., -.1, 1.1, float("nan"), float("inf")):
    rejects(detections, joints, cpf, min_weight=invalid_minimum)
for bad_indices in (torch.tensor([-1, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 5]), torch.tensor([0, 1, 1, 3, 4])):
    invalid = dataclasses.replace(detections, detections_left_concat=dataclasses.replace(left, indices=bad_indices))
    rejects(invalid, joints, cpf)
bad_joints = joints.clone()
bad_joints[0, 15, 0] = float("nan")
rejects(detections, bad_joints, cpf)
bad_joints = joints.clone()
bad_joints[0, [17, 19]] = bad_joints[0, 15].clone()
rejects(detections, bad_joints, cpf)
bad_keypoints = left.keypoints_3d.clone()
bad_keypoints[0, 0, 0] = float("nan")
rejects(dataclasses.replace(detections, detections_left_concat=dataclasses.replace(left, keypoints_3d=bad_keypoints)), joints, cpf)
bad_cpf = cpf.copy()
bad_cpf[0, 4] = np.nan
rejects(detections, joints, bad_cpf)

print("Hand reachability checks passed")
