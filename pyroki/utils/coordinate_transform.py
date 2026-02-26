# -*- coding: utf-8 -*-
"""
Coordinate Transform Module
Unified coordinate system transformation between BVH and Robot frames.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional


class CoordinateTransform:
    """
    Coordinate system transformer for BVH to Robot conversion.
    
    BVH Frame (LaFan1):
        X = Forward
        Y = Up
        Z = Right
    
    Robot Frame (PyBullet/Tiangong):
        X = Forward
        Y = Left
        Z = Up
    
    Transformation:
        Robot_X = BVH_X
        Robot_Y = -BVH_Z  (Right -> Left, flip sign)
        Robot_Z = BVH_Y   (Up -> Up)
    """
    
    def __init__(self):
        """Initialize coordinate transformer"""
        # Build transformation matrix
        # Matrix to transform vectors from BVH frame to Robot frame
        self.R_align = np.array([
            [1,  0,  0],   # BVH_X -> Robot_X
            [0,  0, -1],   # BVH_Z -> Robot_-Y
            [0,  1,  0]    # BVH_Y -> Robot_Z
        ], dtype=np.float32)
        
        # Inverse transformation (Robot -> BVH)
        self.R_align_inv = self.R_align.T  # Since it's orthogonal
        
    def transform_position(self, pos_bvh: np.ndarray) -> np.ndarray:
        """
        Transform position vector(s) from BVH to Robot frame.
        
        Args:
            pos_bvh: Position in BVH frame, shape [..., 3]
        
        Returns:
            pos_robot: Position in Robot frame, shape [..., 3]
        """
        # Handle both single vector and batch
        original_shape = pos_bvh.shape
        pos_flat = pos_bvh.reshape(-1, 3)
        
        # Apply transformation: Robot = R_align @ BVH
        pos_robot_flat = (self.R_align @ pos_flat.T).T
        
        # Restore original shape
        pos_robot = pos_robot_flat.reshape(original_shape)
        
        return pos_robot
    
    def transform_rotation(self, rot_bvh: R, preserve_yaw_only: bool = False) -> R:
        """
        Transform rotation from BVH to Robot frame.
        
        Args:
            rot_bvh: Rotation object in BVH frame
            preserve_yaw_only: If True, only preserve yaw (keep robot upright)
        
        Returns:
            rot_robot: Rotation object in Robot frame
        """
        # Get rotation matrix from BVH
        mat_bvh = rot_bvh.as_matrix()
        
        # Apply similarity transform
        # Handle both single matrix and batch
        if mat_bvh.ndim == 2:
            # Single rotation
            mat_robot = self.R_align @ mat_bvh @ self.R_align.T
            rot_robot = R.from_matrix(mat_robot)
        else:
            # Batch
            mat_robot = np.einsum('ij,njk,kl->nil', self.R_align, mat_bvh, self.R_align.T)
            rot_robot = R.from_matrix(mat_robot)
            
        return rot_robot
