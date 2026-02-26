import os
import sys
import subprocess
import re
import shutil

def install_dependencies():
    """Install required packages if missing."""
    print("\n[Run Cloud] Checking and installing dependencies...")
    
    # List of standard packages
    # Note: jax[cpu] is safer for broad compatibility, but jax is standard.
    # We install 'jax' and let it figure out the backend (or use CPU default).
    packages = ["jax", "jaxlie", "yourdfpy", "trimesh", "numpy", "scipy"]
    
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to install standard packages. Error: {e}")

    # Strategy 3: Source-only usage (No pip install of pyroki itself)
    # This avoids dependency hell and python version checks
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pyroki_lib")
    
    # 1. Ensure code exists
    if not os.path.exists(lib_dir):
        print(f"Cloning PyRoki source to {lib_dir}...")
        try:
            subprocess.check_call(["git", "clone", "https://github.com/chungmin99/pyroki.git", lib_dir])
        except subprocess.CalledProcessError as e:
            print(f"Error cloning PyRoki: {e}")
            sys.exit(1)
            
    # 1.5 FORCE REGULAR PACKAGE: Create __init__.py if missing
    # This converts namespace package to regular package to fix import issues
    pyroki_pkg_dir = os.path.join(lib_dir, "pyroki")
    init_file = os.path.join(pyroki_pkg_dir, "__init__.py")
    if os.path.isdir(pyroki_pkg_dir) and not os.path.exists(init_file):
        print(f"Converting PyRoki to regular package: Creating {init_file}")
        with open(init_file, 'w') as f:
            f.write("# Manually created __init__.py to fix cloud import issues\n")
            f.write("from .robot import Robot\n") # Try to expose Robot directly

    # 2. Install ONLY essential dependencies manually to avoid conflict
    # We skip 'jax-dataclasses' strict version checks by pip
    essential_deps = ["jax", "jaxlie", "yourdfpy", "trimesh", "numpy", "scipy", "overrides", "tqdm", "jax-dataclasses"]
    print("Installing essential dependencies for PyRoki...")
    subprocess.call([sys.executable, "-m", "pip", "install"] + essential_deps)

    # 3. Add to PYTHONPATH for this script
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
        
    # Verify import works now
    try:
        import pyroki
        # Handle case where __file__ is None (namespace pkg)
        pkg_path = getattr(pyroki, '__file__', 'namespace package')
        print(f"PyRoki source loaded successfully from: {pkg_path}")
    except ImportError as e:
        print(f"Warning: Could not import pyroki from source: {e}")

    # 4. Return the lib dir so main() can add it to env
    return lib_dir

def run_command(cmd, env=None):
    """Helper: Run command and print output"""
    print(f"\n[Run Cloud] Executing: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, env=env)
        print("[Run Cloud] Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[Run Cloud] Error executing command: {e}")
        # Don't exit immediately on retargeting errors, try to continue or skip
        if "run_official.py" in cmd[1]:
            print(f"Warning: Retargeting failed for this file.")
        else:
            sys.exit(1)

def main():
    # === 1. Define Paths (Relative to Project Root) ===
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration file listing all motions
    CONFIG_YAML = os.path.join(PROJECT_ROOT, "legged_gym", "motion_data_configs", "lafan1_tienkung.yaml")
    
    # Directories
    BVH_DIR = os.path.join(PROJECT_ROOT, "pyroki", "data", "raw_data", "lafan1")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pyroki", "outputs_new")
    
    # URDF File (Updated path per user request)
    # CRITICAL: Use the fixed 38-DOF URDF
    URDF_FILE = os.path.join(PROJECT_ROOT, "assets", "Tienkung", "urdf", "humanoid_simple.urdf")
    
    # Fallback check
    if not os.path.exists(URDF_FILE):
        print(f"Warning: Fixed URDF not found at {URDF_FILE}")
        URDF_FILE = os.path.join(PROJECT_ROOT, "pyroki", "assets", "urdf", "humanoid_simple.urdf")
        print(f"Falling back to: {URDF_FILE} (Make sure this is correct!)")
    
    # Scripts
    RETARGET_SCRIPT = os.path.join(PROJECT_ROOT, "pyroki", "run_official.py")
    TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "legged_gym", "legged_gym", "scripts", "train.py")
    
    # === 0. Install Dependencies ===
    pyroki_lib_path = install_dependencies()
    
    # === 2. Verify Critical Files ===
    if not os.path.exists(CONFIG_YAML):
        print(f"Error: Config file not found at {CONFIG_YAML}")
        sys.exit(1)
        
    if not os.path.exists(URDF_FILE):
        print(f"Error: URDF file not found at {URDF_FILE}")
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 3. Setup Environment ===
    env = os.environ.copy()
    
    # Get user site-packages
    import site
    user_site = site.getusersitepackages()
    
    # Get system site-packages (approximate)
    system_sites = site.getsitepackages()
    
    # Construct PYTHONPATH
    # Priority: PyRoki Source > User Site > System Site > Legged Gym > RSL RL
    # EXCLUDE PROJECT_ROOT to avoid 'pyroki' folder shadowing 'pyroki' package
    
    paths_to_add = [
        pyroki_lib_path,
        user_site,
        *system_sites,
        os.path.join(PROJECT_ROOT, "legged_gym"),
        os.path.join(PROJECT_ROOT, "rsl_rl"),
    ]
    
    # Inherit existing PYTHONPATH but filter out PROJECT_ROOT
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        for p in current_pythonpath.split(os.pathsep):
            # Normalize paths for comparison
            p_norm = os.path.abspath(p)
            root_norm = os.path.abspath(PROJECT_ROOT)
            if p_norm != root_norm and p_norm != root_norm + os.sep:
                paths_to_add.append(p)
    
    env["PYTHONPATH"] = os.pathsep.join(paths_to_add)
    print(f"[Debug] PYTHONPATH set to: {env['PYTHONPATH']}")

    # === 4. Parse Config to identify needed files ===
    print(f"Reading config from: {CONFIG_YAML}")
    needed_pkls = []
    with open(CONFIG_YAML, 'r') as f:
        content = f.read()
        # Regex to find 'file: filename.pkl'
        matches = re.findall(r'file:\s*([\w\d_]+\.pkl)', content)
        needed_pkls = sorted(list(set(matches))) # Deduplicate
    
    print(f"Found {len(needed_pkls)} unique motion files to regenerate.")

    # === 5. Batch Retargeting ===
    print("==================================================")
    print("STEP 1: Batch Regenerating PKLs (Retargeting)")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("==================================================")
    
    success_count = 0
    
    for pkl_file in needed_pkls:
        # Assume BVH has same basename
        bvh_file = pkl_file.replace('.pkl', '.bvh')
        bvh_path = os.path.join(BVH_DIR, bvh_file)
        output_path = os.path.join(OUTPUT_DIR, pkl_file)
        
        if not os.path.exists(bvh_path):
            print(f"[Skip] BVH not found: {bvh_path}")
            continue
            
        print(f"Processing: {bvh_file} -> {pkl_file}")
        
        retarget_cmd = [
            sys.executable, 
            RETARGET_SCRIPT,
            "--bvh", bvh_path,
            "--urdf", URDF_FILE,
            "--output", output_path
        ]
        
        # We run this individually. If one fails, we continue.
        try:
            # Run with cwd set to pyroki directory to avoid CWD pollution if running from root
            # But ensure we are not in root
            cwd_dir = os.path.dirname(RETARGET_SCRIPT)
            subprocess.check_call(retarget_cmd, env=env, cwd=cwd_dir)
            success_count += 1
        except subprocess.CalledProcessError:
            print(f"[Error] Failed to retarget {bvh_file}")

    print(f"\nRetargeting Complete. Success: {success_count}/{len(needed_pkls)}")

    # === 6. Update Config File ===
    print("==================================================")
    print("STEP 2: Updating Configuration")
    print("==================================================")
    
    # Backup original config
    shutil.copy2(CONFIG_YAML, CONFIG_YAML + ".bak")
    
    # Read and replace root_path
    with open(CONFIG_YAML, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    path_updated = False
    for line in lines:
        if "root_path:" in line:
            # Update path to point to outputs_new
            # Original was likely: root_path: ../../pyroki/outputs
            # New should be: root_path: ../../pyroki/outputs_new
            new_lines.append("root_path: ../../pyroki/outputs_new\n")
            path_updated = True
            print(f"Updated root_path to: ../../pyroki/outputs_new")
        else:
            new_lines.append(line)
            
    with open(CONFIG_YAML, 'w') as f:
        f.writelines(new_lines)
        
    if not path_updated:
        print("Warning: Could not find 'root_path' in config to update!")

    # === 7. Start Training ===
    print("==================================================")
    print("STEP 3: Starting Training")
    print("Task: tienkung_mimic")
    print("==================================================")
    
    train_cmd = [
        sys.executable,
        TRAIN_SCRIPT,
        "--task", "tienkung_mimic",
        "--headless"
    ]
    run_command(train_cmd, env=env)

if __name__ == "__main__":
    main()
