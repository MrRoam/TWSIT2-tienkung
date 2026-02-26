"""
Simple BVH Retargeting Runner
Just specify BVH file name, everything else is automatic!

Usage:
    python run_simple.py walk1_subject1.bvh
    python run_simple.py dance1_subject1.bvh
    python run_simple.py fight1_subject2.bvh
    
    OR with custom path:
    python run_simple.py <path/to/bvh_file.bvh
"""
import os
import sys
import subprocess

def get_output_dir(bvh_name):
    """Determine output directory based on BVH action type"""
    bvh_lower = bvh_name.lower()
    
    if 'walk' in bvh_lower:
        return 'walk'
    elif 'dance' in bvh_lower:
        return 'dance'
    elif 'fight' in bvh_lower:
        return 'fight'
    elif 'aim' in bvh_lower:
        return 'aim'
    elif 'fall' in bvh_lower:
        return 'fall'
    elif 'push' in bvh_lower:
        return 'push'
    elif 'run' in bvh_lower:
        return 'run'
    elif 'sprint' in bvh_lower:
        return 'sprint'
    elif 'jump' in bvh_lower:
        return 'jump'
    elif 'ground' in bvh_lower:
        return 'ground'
    elif 'obstacle' in bvh_lower:
        return 'obstacle'
    else:
        return 'other'

def main():
    # Get BVH filename from command line argument
    if len(sys.argv) < 2:
        print("Usage: python run_simple.py <bvh_filename>")
        print("\nAvailable BVH files:")
        
        # List available BVH files from both locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # List from data/raw_data/lafan1/
        bvh_dir = os.path.join(script_dir, "data", "raw_data", "lafan1")
        if os.path.exists(bvh_dir):
            print("\n  From data/raw_data/lafan1/:")
            for f in sorted(os.listdir(bvh_dir)):
                if f.endswith('.bvh'):
                    print(f"      {f}")
        
        # List from data/bvh_outputs/
        bvh_out_dir = os.path.join(script_dir, "data", "bvh_outputs")
        if os.path.exists(bvh_out_dir):
            print("\n  From data/bvh_outputs/:")
            for f in sorted(os.listdir(bvh_out_dir)):
                if f.endswith('.bvh'):
                    print(f"      {f}")
        
        print("\nExample:")
        print("  python run_simple.py walk1_subject1.bvh")
        print("  python run_simple.py dance1_subject1.bvh")
        print("  python run_simple.py data/bvh_outputs/your_file.bvh")
        sys.exit(1)
    
    bvh_name = sys.argv[1]
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if path is absolute or relative
    if os.path.isabs(bvh_name):
        # Absolute path provided
        bvh_full_path = bvh_name
    else:
        # Relative path - try both locations
        # First check data/raw_data/lafan1/
        bvh_path_standard = os.path.join(script_dir, "data", "raw_data", "lafan1", bvh_name)
        # Then check data/bvh_outputs/
        bvh_path_custom = os.path.join(script_dir, bvh_name)
        
        # Use whichever exists
        if os.path.exists(bvh_path_standard):
            bvh_full_path = bvh_path_standard
            output_dir = os.path.join(script_dir, "outputs")
        elif os.path.exists(bvh_path_custom):
            bvh_full_path = bvh_path_custom
            output_dir = os.path.dirname(bvh_full_path)
        else:
            print(f"ERROR: BVH file not found:")
            print(f"  Tried: {bvh_path_standard}")
            print(f"  Tried: {bvh_path_custom}")
            sys.exit(1)
    
    # Verify BVH file exists
    if not os.path.exists(bvh_full_path):
        print(f"ERROR: BVH file not found: {bvh_full_path}")
        sys.exit(1)
    
    # Determine output directory based on file location
    if "bvh_outputs" in bvh_full_path:
        output_dir_name = "bvh_outputs"
    else:
        output_dir_name = get_output_dir(bvh_name)
    
    # Generate output filename based on action type
    if output_dir_name == "bvh_outputs":
        # Already in bvh_outputs, use same name
        output_pkl = os.path.join(script_dir, "outputs", "other", "output.pkl")
    else:
        output_dir = os.path.join(script_dir, "outputs", output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        output_pkl = os.path.join(output_dir, "output.pkl")
    
    # Verify input BVH file exists
    if not os.path.exists(bvh_full_path):
        print(f"ERROR: BVH file not found: {bvh_full_path}")
        sys.exit(1)
    
    # Print summary
    print("="*60)
    print("Simple BVH Retargeting")
    print("="*60)
    print(f"BVH:      {os.path.basename(bvh_full_path)}")
    print(f"URDF:      humanoid_simple.urdf")
    print(f"Output:    {output_pkl}")
    print("="*60)
    print()
    
    # Step 1: Run IK optimization
    print("[1/2] Running IK optimization (this takes ~85 seconds)...")
    cmd1 = [
        sys.executable,
        "run_official.py",
        "--bvh", bvh_full_path,
        "--output", output_pkl
    ]
    
    result1 = subprocess.run(cmd1)
    
    if result1.returncode != 0:
        print(f"\nERROR: IK optimization failed with code {result1.returncode}")
        sys.exit(1)
    
    print("\n[1/2] IK optimization completed!")
    print()
    
    # Step 2: Start visualization
    print("[2/2] Starting visualization...")
    print("Close the PyBullet window to exit.")
    print()
    
    cmd2 = [
        sys.executable,
        "visualize_comparison.py",
        "--bvh", bvh_full_path,
        "--pkl", output_pkl
    ]
    
    subprocess.call(cmd2)

if __name__ == "__main__":
    main()
