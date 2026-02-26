#!/usr/bin/env python3
"""
Safe Batch BVH to PKL Converter (Single-threaded)

Features:
- Processes ALL BVH files in data/raw_data/lafan1/ (78 files total)
- Uses humanoid_simple.urdf (26 DOF, no fingers)
- Generates PKL files in outputs/ directory
- Clear naming: Keeps BVH filename (e.g., walk1_subject1.bvh -> walk1_subject1.pkl)
- Progress display

Usage for testing (first 10 files):
    python batch_convert.py

For full conversion (all 78 files, ~109 minutes):
    python batch_convert.py
"""

import os
import sys
import subprocess
import time

def get_bvh_files_sorted(bvh_dir):
    """Get BVH files sorted by modification time (newest first)"""
    bvh_files = [f for f in os.listdir(bvh_dir) if f.endswith('.bvh')]
    
    # Get file modification times
    files_with_time = []
    for bvh in bvh_files:
        full_path = os.path.join(bvh_dir, bvh)
        mtime = os.path.getmtime(full_path)
        files_with_time.append((mtime, bvh))
    
    # Sort by modification time (newest first)
    files_with_time.sort(reverse=True, key=lambda x: x[0])
    
    return [f[1] for f in files_with_time]

def convert_single_bvh(script_dir, bvh_dir, bvh_name):
    """Convert a single BVH file to PKL"""
    bvh_full_path = os.path.join(bvh_dir, bvh_name)
    
    # Output: outputs/walk1_subject1.pkl (keeps BVH name)
    output_pkl = os.path.join(script_dir, "outputs", bvh_name.replace('.bvh', '.pkl'))
    
    # Verify BVH exists
    if not os.path.exists(bvh_full_path):
        print(f"  ERROR: BVH not found: {bvh_name}")
        return False
    
    # Print progress
    print(f"  Converting {bvh_name} -> PKL...")
    
    # Run IK optimization
    start_time = time.time()
    cmd = [
        sys.executable,
        "run_official.py",
        "--urdf", "assets/urdf/humanoid_simple.urdf",
        "--bvh", bvh_full_path,
        "--output", output_pkl
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"  ERROR (took {elapsed:.1f}s): {result.returncode}")
            if result.stderr:
                print(f"  Error message: {result.stderr[:200]}")
            return False
        else:
            # Verify output PKL exists
            if os.path.exists(output_pkl):
                size_mb = os.path.getsize(output_pkl) / (1024 * 1024)
                print(f"  OK (took {elapsed:.1f}s, size: {size_mb:.2f}MB")
            else:
                print(f"  ERROR (took {elapsed:.1f}s): Output PKL not created")
                return False
            return True
            
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Timeout after 300s")
        return False
    except KeyboardInterrupt:
        print(f"  CANCELLED")
        # Clean up incomplete output file
        if os.path.exists(output_pkl):
            try:
                os.remove(output_pkl)
                print(f"  Cleaned up incomplete output")
            except:
                pass
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bvh_dir = os.path.join(script_dir, "data", "raw_data", "lafan1")
    output_dir = os.path.join(script_dir, "outputs")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get BVH files sorted by newest
    try:
        bvh_files = get_bvh_files_sorted(bvh_dir)
    except FileNotFoundError:
        print(f"Error: Directory not found: {bvh_dir}")
        return

    # Process all BVH files
    print("="*70)
    print("Safe Batch BVH to PKL Converter (All 78 Files)")
    print("="*70)
    print(f"Script directory: {script_dir}")
    print(f"BVH directory: {bvh_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total BVH files: {len(bvh_files)}")
    print(f"Estimated time: {len(bvh_files) * 85 / 60:.1f} minutes ({len(bvh_files) * 85 // 60:.1f} hours)")
    print("="*70)
    print()
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    try:
        for i, bvh_name in enumerate(bvh_files, 1):
            print(f"[{i:3d}/{len(bvh_files)}] {bvh_name} -> PKL...")
            success = convert_single_bvh(script_dir, bvh_dir, bvh_name)
            if success:
                success_count += 1
            else:
                fail_count += 1
                
    except KeyboardInterrupt:
        print("\n" + "="*70)
        print("INTERRUPTED by user")
        print("="*70)
        print(f"Processed: {i}/{len(bvh_files)} files")
        print(f"Success: {success_count}, Failed: {fail_count}")
        print(f"\nCompleted PKL files will be in: {output_dir}")
        sys.exit(130)  # Standard exit code for Ctrl+C
    
    # Summary
    print("="*70)
    print("COMPLETED!")
    print("="*70)
    print(f"Total processed: {len(bvh_files)} files")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"\nOutput directory: {output_dir}")
    print()
    print("PKL files naming: walk1_subject1.pkl, dance1_subject1.pkl, etc.")
    print("All PKL files are in outputs/ root directory (no subfolders)")
    print("="*70)

if __name__ == "__main__":
    main()