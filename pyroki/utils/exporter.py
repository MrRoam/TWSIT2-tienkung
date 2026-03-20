"""
TWIST2 PKL Format Exporter (Simplified for Repro)
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Optional

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion(s).
    Args:
        q: Quaternion array [..., 4] (x, y, z, w) or (w, x, y, z) depending on convention,
           but normalization is component-wise anyway.
    Returns:
        Normalized quaternion(s)
    """
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / (norm + 1e-8)

class TWIST2Exporter:
    """Exporter for TWIST2 PKL format."""
    
    def __init__(self, dof: int, key_bodies: List[str]):
        """
        Initialize exporter.
        
        Args:
            dof: Number of degrees of freedom (actuated joints)
            key_bodies: List of body names to track
        """
        self.dof = dof
        self.key_bodies = key_bodies
    
    def export(self, 
               root_pos: np.ndarray,
               root_rot: np.ndarray, 
               dof_pos: np.ndarray,
               local_body_pos: Optional[np.ndarray] = None,
               dof_names: Optional[List[str]] = None,
               fps: float = 30.0,
               output_path: str = "output.pkl") -> Dict:
        """
        Export motion data to TWIST2 PKL format.
        
        Args:
            root_pos: Root position array [N, 3] in meters (Z-up)
            root_rot: Root rotation quaternion array [N, 4] in xyzw order
            dof_pos: Joint angles array [N, DOF] in radians
            local_body_pos: Local body positions [N, M, 3] (optional)
            dof_names: Optional joint names aligned with dof_pos columns
            fps: Frame rate
            output_path: Output file path
        """
        n_frames = root_pos.shape[0]
        
        # Validate inputs
        assert root_pos.shape == (n_frames, 3), f"Invalid root_pos shape: {root_pos.shape}"
        assert root_rot.shape == (n_frames, 4), f"Invalid root_rot shape: {root_rot.shape}"
        assert dof_pos.shape == (n_frames, self.dof), \
            f"Invalid dof_pos shape: {dof_pos.shape}, expected [, {self.dof}]"
        
        # Normalize quaternions
        root_rot = normalize_quaternion(root_rot)
        
        # Use provided local_body_pos or create placeholder
        if local_body_pos is None:
            # Create placeholder
            num_bodies = len(self.key_bodies)
            local_body_pos = np.zeros((n_frames, num_bodies, 3), dtype=np.float64)
            print(f"Warning: local_body_pos not provided, using zeros")
        
        # Prepare PKL data (float64 for higher precision)
        pkl_data = {
            "fps": float(fps),
            "root_pos": root_pos.astype(np.float64),
            "root_rot": root_rot.astype(np.float64),
            "dof_pos": dof_pos.astype(np.float64),
            "local_body_pos": local_body_pos.astype(np.float64),
            "link_body_list": self.key_bodies
        }

        if dof_names is not None:
            pkl_data["dof_names"] = list(dof_names)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(pkl_data, f)
            
        print(f"Exported to {output_path}")
        return pkl_data
