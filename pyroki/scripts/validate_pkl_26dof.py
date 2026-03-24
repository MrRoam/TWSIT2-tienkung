"""
PKL File Validator for TWIST2 Tienkung motions.

This validator checks the current Tienkung export format:
- DOF count: 30
- Quaternion format: xyzw
- Key body links: 26
"""

import pickle
import sys

import numpy as np


EXPECTED_JOINTS_30 = [
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "waist_yaw_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "head_roll_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]

EXPECTED_BODIES_26 = [
    "pelvis",
    "waist_yaw_link",
    "hip_roll_l_link",
    "hip_pitch_l_link",
    "hip_yaw_l_link",
    "knee_pitch_l_link",
    "ankle_pitch_l_link",
    "ankle_roll_l_link",
    "hip_roll_r_link",
    "hip_pitch_r_link",
    "hip_yaw_r_link",
    "knee_pitch_r_link",
    "ankle_pitch_r_link",
    "ankle_roll_r_link",
    "shoulder_pitch_l_link",
    "shoulder_roll_l_link",
    "shoulder_yaw_l_link",
    "elbow_pitch_l_link",
    "elbow_yaw_l_link",
    "wrist_roll_l_link",
    "shoulder_pitch_r_link",
    "shoulder_roll_r_link",
    "shoulder_yaw_r_link",
    "elbow_pitch_r_link",
    "elbow_yaw_r_link",
    "wrist_roll_r_link",
]


def validate_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print("=" * 70)
    print("PKL File Validation (Tienkung 30 DOF)")
    print("=" * 70)
    print()

    all_passed = True

    print("[1] Basic Structure")
    required_keys = ["fps", "root_pos", "root_rot", "dof_pos", "local_body_pos", "link_body_list"]
    for key in required_keys:
        if key not in data:
            print(f"  X Missing key: {key}")
            all_passed = False
    if all(key in data for key in required_keys):
        print("  OK All required keys present")
    print()

    print("[2] DOF Count")
    dof_count = data["dof_pos"].shape[1]
    print(f"  Current DOF: {dof_count}")
    print(f"  Expected DOF: {len(EXPECTED_JOINTS_30)}")
    if dof_count == len(EXPECTED_JOINTS_30):
        print("  OK DOF count correct")
    else:
        print(f"  X DOF count incorrect (diff: {dof_count - len(EXPECTED_JOINTS_30)})")
        all_passed = False
    print()

    print("[3] Body Links Count")
    body_count = data["local_body_pos"].shape[1]
    print(f"  Current body count: {body_count}")
    print(f"  Expected body count: {len(EXPECTED_BODIES_26)}")
    if body_count == len(EXPECTED_BODIES_26):
        print("  OK Body count correct")
    else:
        print(f"  X Body count incorrect (diff: {body_count - len(EXPECTED_BODIES_26)})")
        all_passed = False
    print()

    print("[4] Data Types")
    for key in ["root_pos", "root_rot", "dof_pos", "local_body_pos"]:
        dtype = data[key].dtype
        if dtype == np.float64:
            print(f"  {key}: {dtype} - OK")
        else:
            print(f"  {key}: {dtype} - X expected float64")
            all_passed = False
    print()

    print("[5] Quaternion Normalization")
    quat_norms = np.linalg.norm(data["root_rot"], axis=1)
    if np.allclose(quat_norms, 1.0, atol=0.01):
        print("  OK Quaternions normalized")
    else:
        print("  X Quaternions not normalized")
        print(f"    Norm range: {quat_norms.min():.4f} - {quat_norms.max():.4f}")
        all_passed = False
    print()

    print("[6] Joint Names")
    actual_joint_names = data.get("dof_names")
    if actual_joint_names is None:
        print("  X Missing dof_names")
        all_passed = False
    elif list(actual_joint_names) == EXPECTED_JOINTS_30:
        print("  OK Joint names and order correct")
    else:
        print("  X Joint names/order mismatch")
        print(f"    Expected: {EXPECTED_JOINTS_30}")
        print(f"    Actual:   {list(actual_joint_names)}")
        all_passed = False
    print()

    print("[7] Body Link Names")
    actual_bodies = list(data["link_body_list"])
    if actual_bodies == EXPECTED_BODIES_26:
        print("  OK Body link names and order correct")
    else:
        print("  X Body link names/order mismatch")
        print(f"    Expected: {EXPECTED_BODIES_26}")
        print(f"    Actual:   {actual_bodies}")
        all_passed = False
    print()

    print("[8] Frame Rate")
    fps = data["fps"]
    if 20 <= fps <= 60:
        print(f"  OK FPS: {fps}")
    else:
        print(f"  X FPS out of range: {fps}")
        all_passed = False
    print()

    print("[9] Data Continuity")
    dof_diff = np.diff(data["dof_pos"], axis=0)
    large_changes = np.sum(np.abs(dof_diff) > 1.0)
    total_values = dof_diff.shape[0] * dof_diff.shape[1]
    ratio = large_changes / total_values
    if ratio < 0.01:
        print(f"  OK Joint angles smooth ({ratio * 100:.2f}% changes > 1.0 rad)")
    else:
        print(f"  X Joint angles have abrupt changes ({ratio * 100:.2f}% changes > 1.0 rad)")
        all_passed = False
    print()

    print("=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)
    if all_passed:
        print("  OK ALL CHECKS PASSED")
        print()
        print("This PKL file is aligned with the current Tienkung URDF.")
    else:
        print("  X SOME CHECKS FAILED")
        print()
        print("Please review the mismatches above.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_pkl_26dof.py <pkl_file>")
        print()
        print("Example:")
        print("  python validate_pkl_26dof.py outputs_test/test_2.pkl")
        sys.exit(1)

    validate_pkl(sys.argv[1])
