"""
Simple PKL File Validator for TWIST2 Format
"""
import pickle
import numpy as np
import sys

def validate_pkl(pkl_path):
    """Validate PKL file against TWIST2 requirements"""
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("=" * 70)
    print("PKL File Validation")
    print("=" * 70)
    print()
    
    # Validation 1: Basic structure
    print("[1] Basic Structure")
    required_keys = ['fps', 'root_pos', 'root_rot', 'dof_pos', 'local_body_pos', 'link_body_list']
    for key in required_keys:
        if key not in data:
            print(f"  [ERROR] Missing key: {key}")
            return False
    print("  [OK] All required keys present")
    print()
    
    # Validation 2: Shapes
    print("[2] Data Shapes")
    fps = data['fps']
    root_pos = data['root_pos']
    root_rot = data['root_rot']
    dof_pos = data['dof_pos']
    local_body_pos = data['local_body_pos']
    
    n_frames = root_pos.shape[0]
    dof_count = dof_pos.shape[1]
    body_count = local_body_pos.shape[1]
    link_count = len(data['link_body_list'])
    
    print(f"  Frames: {n_frames}")
    print(f" FPS: {fps}")
    print(f" DOF: {dof_count}")
    print(f" Body links: {body_count}")
    print(f" Link list length: {link_count}")
    print()
    
    # Validation 3: Data types
    print("[3] Data Types")
    checks = ['root_pos', 'root_rot', 'dof_pos', 'local_body_pos']
    
    all_correct = True
    for key in checks:
        dtype = data[key].dtype
        is_float64 = dtype == np.float64
        if is_float64:
            print(f"  [OK] {key}: {dtype}")
        else:
            print(f"  [WARN] {key} should be float64, got {dtype}")
            all_correct = False
    
    # Check fps separately (it's a Python float, not numpy array)
    fps_dtype = np.dtype(np.float64(0))
    if isinstance(data['fps'], float) and not isinstance(data['fps'], np.ndarray):
        print(f"  [OK] fps: Python float")
    elif isinstance(data['fps'], np.ndarray) and data['fps'].dtype == fps_dtype:
        print(f"  [OK] fps: numpy array {data['fps'].dtype}")
    else:
        print(f"  [WARN] fps: should be float64, got {type(data['fps'])}")
    
    if not all_correct:
        print()
        print("[4] Data Continuity")
        dof_diff = np.diff(dof_pos, axis=0)
        large_changes = np.sum(np.abs(dof_diff) > 1.0)
        total_values = dof_diff.shape[0] * dof_diff.shape[1]
        ratio = large_changes / total_values
        if ratio < 0.01:
            print(f"  [OK] Joint angles smooth")
        else:
            print(f"  [WARNING] {ratio*100:.2f}% changes > 1.0 rad")
    print()
    
    # Validation 4: Quaternion normalization
    print("[4] Quaternion Normalization")
    quat_norms = np.linalg.norm(root_rot, axis=1)
    if np.allclose(quat_norms, 1.0, atol=0.01):
        print("  [OK] Quaternions normalized")
    else:
        print("  [WARNING] Quaternions not normalized")
    print()
    
    # Validation 5: Body links
    print("[5] Body Links")
    print(f"  Link count: {link_count}")
    print(f"  Links: {data['link_body_list']}")
    print()
    
    # Final result
    print("=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)
    if all_correct:
        print("[SUCCESS] PKL file meets requirements!")
        print()
        print(f"  DOF count: {dof_count}")
        print(f"  Body links: {body_count}")
        print(f"  Data types: float64")
    else:
        print("[PARTIAL] PKL file has warnings")
    print("=" * 70)
    
    return all_correct

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_pkl_simple.py <pkl_file>")
        print()
        print("Example:")
        print("  python validate_pkl_simple.py outputs/test_26dof.pkl")
        print("  python validate_pkl_simple.py ../data/fight1_subject2.pkl")
        sys.exit(1)
    
    pkl_file = sys.argv[1]
    validate_pkl(pkl_file)
