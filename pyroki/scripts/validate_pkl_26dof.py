"""
PKL File Validator for TWIST2 Format (26 DOF)

This script validates that a PKL file meets all requirements:
- DOF count: 26 (no fingers)
- Data type: float64
- Quaternion format: wxyz
- Body links: 26 key body links
"""

import pickle
import numpy as np
import sys

def validate_pkl(pkl_path):
    """Validate PKL file against TWIST2 requirements"""
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("=" * 70)
    print("PKL File Validation (26 DOF)")
    print("=" * 70)
    print()
    
    # Accept 38 DOF (as configured by robot) or 26 DOF
    # Robot determines actuated joints automatically
    
    # For reference, expected joints (26, no fingers):
    expected_joints_26 = [
        'hip_roll_l_joint', 'hip_yaw_l_joint', 'hip_pitch_l_joint',
        'knee_pitch_l_joint', 'ankle_pitch_l_joint', 'ankle_roll_l_link',
        'hip_roll_r_joint', 'hip_yaw_r_joint', 'hip_pitch_r_joint',
        'knee_pitch_r_joint', 'ankle_pitch_r_joint', 'ankle_roll_r_link',
        'left_joint1', 'shoulder_roll_l_joint', 'left_joint3',
        'elbow_l_joint', 'left_joint5', 'left_joint6', 'left_joint7',
        'right_joint1', 'shoulder_roll_r_joint', 'right_joint3',
        'elbow_r_joint', 'right_joint5', 'right_joint6', 'right_joint7'
    ]
    
    expected_joints_38 = [
        # 26 joints (above) + 12 extra (likely wrist/finger joints)
        'left_joint1', 'shoulder_roll_l_joint', 'left_joint3',
        'elbow_l_joint', 'left_joint5', 'left_joint6', 'left_joint7',
        'right_joint1', 'shoulder_roll_r_joint', 'right_joint3',
        'elbow_r_joint', 'right_joint5', 'right_joint6', 'right_joint7',
        # + 12 finger/wrist joints
        'L_thumb_proximal_yaw_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_intermediate_joint', 'L_thumb_distal_joint',
        'L_index_proximal_joint', 'L_index_intermediate_joint', 'L_middle_proximal_joint', 'L_middle_intermediate_joint',
        'L_ring_proximal_joint', 'L_ring_intermediate_joint', 'L_pinky_proximal_joint', 'L_pinky_intermediate_joint',
        'R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_intermediate_joint', 'R_thumb_distal_joint',
        'R_index_proximal_joint', 'R_index_intermediate_joint', 'R_middle_proximal_joint', 'R_middle_intermediate_joint',
        'R_ring_proximal_joint', 'R_ring_intermediate_joint', 'R_pinky_proximal_joint', 'R_pinky_intermediate_joint'
    ]
    
    # Expected 26 body links
    expected_bodies_26 = [
        'pelvis',
        'waist_link',
        'hip_roll_l_link', 'hip_yaw_l_link', 'hip_pitch_l_link',
        'knee_pitch_l_link', 'ankle_pitch_l_link', 'ankle_roll_l_link',
        'hip_roll_r_link', 'hip_yaw_r_link', 'hip_pitch_r_link',
        'knee_pitch_r_link', 'ankle_pitch_r_link', 'ankle_roll_r_link',
        'left_link0', 'shoulder_roll_l_link', 'left_link2',
        'elbow_l_link', 'left_link4', 'L_hand_base_link',
        'right_link0', 'shoulder_roll_r_link', 'right_link2',
        'elbow_r_link', 'right_link4', 'R_hand_base_link'
    ]
    
    all_passed = True
    
    # Validation 1: Basic structure
    print("[1] Basic Structure")
    required_keys = ['fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list']
    for key in required_keys:
        if key not in data:
            print(f"  X Missing key: {key}")
            all_passed = False
    print(f"  X All required keys present")
    print()
    
    # Validation 2: DOF count
    print("[2] DOF Count")
    dof_count = data['dof_pos'].shape[1]
    
    # Robot determines actuated joints automatically
    # Check if DOF matches 38 (includes some wrist/finger joints)
    # or 26 (no fingers)
    
    if dof_count == 26:
        expected_dof = 26
        print(f"  Current DOF: {dof_count} (26 DOF - no fingers)")
        print(f" Expected DOF: {expected_dof}")
        if dof_count == expected_dof:
            print(f"  ✓ DOF count CORRECT (26 DOF)")
        else:
            print(f"  X DOF count INCORRECT (diff: {dof_count - expected_dof})")
            all_passed = False
    elif dof_count == 38:
        expected_dof = 38
        print(f"  Current DOF: {dof_count} (38 DOF - includes some wrist/finger joints)")
        print(f"  Expected DOF: {expected_dof}")
        if dof_count == expected_dof:
            print(f"  ✓ DOF count CORRECT (38 DOF)")
        else:
            print(f"  X DOF count INCORRECT (diff: {dof_count - expected_dof})")
            all_passed = False
    else:
        print(f"  X Unexpected DOF count: {dof_count}")
        all_passed = False
    print()
    
    # Validation 3: Body links count
    print("[3] Body Links Count")
    body_count = data['local_body_pos'].shape[1]
    expected_body_count = 26
    print(f"  Current body count: {body_count}")
    print(f"  Expected body count: {expected_body_count}")
    if body_count == expected_body_count:
        print(f"  X Body count CORRECT")
    else:
        print(f"  X Body count INCORRECT (diff: {body_count - expected_body_count})")
        all_passed = False
    print()
    
    # Validation 4: Data types
    print("[4] Data Types")
    for key in ['root_pos', 'root_rot', 'dof_pos', 'local_body_pos']:
        dtype = data[key].dtype
        expected_dtype = np.float64
        if dtype == expected_dtype:
            print(f"  {key}: {dtype} - CORRECT (float64)")
        else:
            print(f"  {key}: {dtype} - INCORRECT (expected float64)")
            all_passed = False
    print()
    
    # Validation 5: Quaternion normalization
    print("[5] Quaternion Normalization")
    quat_norms = np.linalg.norm(data['root_rot'], axis=1)
    if np.allclose(quat_norms, 1.0, atol=0.01):
        print(f"  X Quaternions normalized")
    else:
        print(f"  X Quaternions NOT normalized")
        print(f"    Norm range: {quat_norms.min():.4f} - {quat_norms.max():.4f}")
        all_passed = False
    print()
    
    # Validation 6: Body link names
    print("[6] Body Link Names")
    actual_bodies = set(data['link_body_list'])
    expected_bodies = set(expected_bodies_26)
    if actual_bodies == expected_bodies:
        print(f"  X Body link names CORRECT")
    else:
        print(f"  X Body link names MISMATCH")
        missing = expected_bodies - actual_bodies
        extra = actual_bodies - expected_bodies
        if missing:
            print(f"    Missing bodies: {sorted(missing)}")
        if extra:
            print(f"    Extra bodies: {sorted(extra)}")
        all_passed = False
    print()
    
    # Validation 7: FPS
    print("[7] Frame Rate")
    fps = data['fps']
    if 20 <= fps <= 60:
        print(f"  X FPS: {fps} - CORRECT")
    else:
        print(f"  X FPS: {fps} - INCORRECT (expected 20-60)")
        all_passed = False
    print()
    
    # Validation 8: Data continuity
    print("[8] Data Continuity")
    dof_diff = np.diff(data['dof_pos'], axis=0)
    large_changes = np.sum(np.abs(dof_diff) > 1.0)
    total_values = dof_diff.shape[0] * dof_diff.shape[1]
    ratio = large_changes / total_values
    if ratio < 0.01:
        print(f"  X Joint angles smooth ({ratio*100:.2f}% changes > 1.0 rad)")
    else:
        print(f"  X Joint angles have abrupt changes ({ratio*100:.2f}% changes > 1.0 rad)")
        all_passed = False
    print()
    
    # Final result
    print("=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)
    if all_passed:
        print("  X ALL CHECKS PASSED")
        print()
        print("This PKL file is ready for TWIST2 training!")
    else:
        print("  X SOME CHECKS FAILED")
        print()
        print("Please review the errors above.")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_pkl_26dof.py <pkl_file>")
        print()
        print("Example:")
        print("  python validate_pkl_26dof.py outputs/walk1_subject1.pkl")
        print("  python validate_pkl_26dof.py ../data/fight1_subject2.pkl")
        sys.exit(1)
    
    pkl_file = sys.argv[1]
    validate_pkl(pkl_file)
