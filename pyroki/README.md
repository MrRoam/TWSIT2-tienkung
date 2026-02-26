# PyRoki BVH Retargeting Project

## Project Structure

```
pyroki/
├── assets/              # Robot models and meshes
│   ├── urdf/          # URDF files
│   │   ├── humanoid.urdf              # Full robot (original)
│   │   └── humanoid_simple.urdf        # Simple robot (fixed wrist issue)
│   └── meshes/         # STL mesh files
├── data/                # Motion data
│   └── raw_data/
│       └── lafan1/     # BVH motion files
├── utils/               # Utility modules
│   ├── bvh_loader.py  # BVH file loader
│   ├── bvh_fk.py      # BVH forward kinematics
│   ├── coordinate_transform.py  # Coordinate system conversion
│   └── exporter.py    # TWIST2 format exporter
├── outputs/             # Generated PKL files (organized by action type)
│   ├── walk/         # Walking motions
│   │   ├── output.pkl
│   │   └── output_simple.pkl
│   ├── dance/        # Dancing motions
│   │   └── output_dance1.pkl
│   ├── fight/        # Fighting motions
│   │   └── fight1_subject2.pkl
│   ├── aim/          # Aiming motions (auto-created)
│   ├── fall/         # Falling motions (auto-created)
│   ├── push/         # Pushing motions (auto-created)
│   ├── run/          # Running motions (auto-created)
│   ├── sprint/       # Sprinting motions (auto-created)
│   ├── jump/         # Jumping motions (auto-created)
│   ├── ground/       # Ground motions (auto-created)
│   └── obstacle/     # Obstacle motions (auto-created)
├── scripts/              # Utility scripts
│   ├── inspect_bvh.py
│   ├── simple_verify.py
│   ├── verify_output.py
│   └── verify_reshape_correctness.py
├── run_simple.py        # Main entry point (simplified)
├── run_official.py      # IK optimization
├── visualize_comparison.py  # Visualization
└── README.md            # This file
```

## Usage

### Quick Start

```bash
# Walk motion
python run_simple.py walk1_subject1.bvh

# Dance motion
python run_simple.py dance1_subject1.bvh

# Fight motion
python run_simple.py fight1_subject2.bvh
```

### Available BVH Actions

**Walk:**
- walk1_subject1.bvh, walk1_subject2.bvh, walk1_subject5.bvh
- walk2_subject1.bvh, walk2_subject3.bvh, walk2_subject4.bvh
- walk3_subject1.bvh, walk3_subject2.bvh, walk3_subject3.bvh
- walk3_subject4.bvh, walk3_subject5.bvh
- walk4_subject1.bvh

**Dance:**
- dance1_subject1.bvh, dance1_subject2.bvh, dance1_subject3.bvh
- dance2_subject1.bvh, dance2_subject2.bvh, dance2_subject3.bvh
- dance2_subject4.bvh, dance2_subject5.bvh

**Fight:**
- fight1_subject2.bvh, fight1_subject3.bvh, fight1_subject5.bvh
- fight2_subject1.bvh, fight2_subject2.bvh, fight2_subject3.bvh
- fight2_subject4.bvh

**Other:**
- aiming, falling, jumping, pushing, running, sprinting, obstacles, etc.

## Key Files

### run_simple.py
- **Purpose**: Main entry point for retargeting
- **Usage**: `python run_simple.py <bvh_filename>`
- **Features**:
  - Automatically detects action type from BVH filename
  - Organizes output into `outputs/<action_type>/output.pkl`
  - Runs IK optimization (~85 seconds)
  - Launches visualization

### run_official.py
- **Purpose**: IK optimization
- **Input**: BVH file
- **Output**: PKL file with retargeted motion

### visualize_comparison.py
- **Purpose**: Visualize BVH skeleton vs. retargeted robot
- **Display**: Side-by-side comparison in PyBullet

## Configuration

### URDF
- **Recommended**: `humanoid_simple.urdf` (uses simplified meshes)
- **Alternative**: `humanoid.urdf` (original full meshes)
- **Location**: `assets/urdf/`

### Key Fix Applied
- **Issue**: `humanoid_simple.urdf` had incorrect pitch value for `right_joint7`
- **Fix**: Changed pitch from `+1.5708` to `-1.5708` (180-degree offset)
- **Result**: Right hand no longer appears folded on load

## Important Notes

### Output Organization
- PKL files are automatically organized by action type in `outputs/` directory
- Each action type has its own subdirectory (walk, dance, fight, etc.)
- Output file is always named `output.pkl` in the respective action folder

### BVH Motion Dataset
- Dataset: LaFan1
- Total frames: ~7840 per BVH file
- FPS: 30.0
- Duration: ~261 seconds (full walk sequence)

### Robot Model
- Robot: Tiangong (天宫) humanoid
- DOF: 38 (actuated, non-mimic joints)
- Total joints: 55 (including fixed and mimic)
- Mimic joints: 12 (finger intermediate/distal joints)

## Troubleshooting

### Right Arm Not Moving
- **Symptom**: Right arm appears stiff, elbow stays straight
- **Cause**: Mimic joint mapping error in `visualize_comparison.py:344`
- **Fix**: Joint order filtering to exclude mimic joints
  ```python
  if j.type != 'fixed' and (not hasattr(j, 'mimic') or j.mimic is None):
      config_joint_order.append(j.name)
  ```

### Right Hand Folded on Load
- **Symptom**: Right hand appears folded up when loading
- **Cause**: Incorrect pitch value in `humanoid_simple.urdf` for `right_joint7`
- **Fix**: Changed `rpy` pitch from `+1.5708` to `-1.5708`

## Development

### Dependencies
- Python 3.9+
- JAX 0.9+
- PyRoki
- yourdfpy
- PyBullet (for visualization)
- NumPy
- SciPy

### Environment
- OS: Windows
- Python: 3.13
- Working directory: `D:\Desktop\pyroki`

## History

- Feb 8, 2026: Project organization completed
  - Created `outputs/` directory structure
  - Moved PKL files to action-based subdirectories
  - Moved utility scripts to `scripts/` directory
  - Updated `run_simple.py` for automatic output organization
  - Fixed `humanoid_simple.urdf` right_joint7 pitch issue
  - Fixed `visualize_comparison.py` mimic joint mapping
