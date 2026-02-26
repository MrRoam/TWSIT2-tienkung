"""
BVH Forward Kinematics Calculator
"""

import numpy as np
from typing import Dict, List, Optional
import sys
import os

from .bvh_loader import BVHLoader, BVHNode


def compute_bvh_fk(bvh_loader: BVHLoader, frame_idx: int) -> Dict[str, np.ndarray]:
    """
    Compute global positions of all joints in a BVH frame.
    
    Args:
        bvh_loader: Loaded BVH data
        frame_idx: Frame index to compute
        
    Returns:
        Dictionary mapping joint names to global positions [x, y, z] in meters
        Coordinate system: BVH standard (Y-up, Z-forward, X-right)
    """
    if frame_idx >= bvh_loader.frames:
        raise ValueError(f"Frame {frame_idx} out of range (total {bvh_loader.frames})")
    
    frame_data = bvh_loader.get_frame_data(frame_idx)
    
    # Result dictionary
    joint_positions = {}
    
    # Recursive FK computation starting from root
    _compute_joint_transform(
        node=bvh_loader.root,
        parent_transform=np.eye(4),
        frame_data=frame_data,
        result=joint_positions
    )
    
    return joint_positions


def _compute_joint_transform(node: BVHNode,
                             parent_transform: np.ndarray,
                             frame_data: np.ndarray,
                             result: Dict[str, np.ndarray]):
    """
    Recursively compute joint transforms using forward kinematics.
    """
    # BVH unit scale (cm to m)
    SCALE = 0.01
    
    # Start with parent transform
    local_transform = np.eye(4)
    
    # Apply offset (translation) - convert from cm to meters
    local_transform[:3, 3] = node.offset * SCALE
    
    # Apply rotations from channels (if any)
    if node.channels:
        rotation_matrix = _get_rotation_from_channels(node, frame_data)
        local_transform[:3, :3] = rotation_matrix
        
        # Handle position channels (only for root)
        if 'Xposition' in node.channels:
            x_idx = node.channels.index('Xposition')
            y_idx = node.channels.index('Yposition')
            z_idx = node.channels.index('Zposition')
            
            # Get position values (convert from cm to meters)
            position = np.array([
                frame_data[node.channel_indices[x_idx]] * SCALE,
                frame_data[node.channel_indices[y_idx]] * SCALE,
                frame_data[node.channel_indices[z_idx]] * SCALE
            ])
            
            local_transform[:3, 3] = position
    
    # Compute global transform
    global_transform = parent_transform @ local_transform
    
    # Extract global position
    global_position = global_transform[:3, 3]
    result[node.name] = global_position
    
    # Recursively process children
    for child in node.children:
        _compute_joint_transform(child, global_transform, frame_data, result)


def _get_rotation_from_channels(node: BVHNode, frame_data: np.ndarray) -> np.ndarray:
    """
    Extract rotation matrix from BVH channels.
    """
    # Extract euler angles from channels
    euler_angles = np.zeros(3)
    channel_order = []
    
    for i, channel in enumerate(node.channels):
        if 'rotation' in channel.lower():
            value = frame_data[node.channel_indices[i]]
            
            if 'Xrotation' in channel:
                channel_order.append('X')
                euler_angles[len(channel_order) - 1] = np.deg2rad(value)
            elif 'Yrotation' in channel:
                channel_order.append('Y')
                euler_angles[len(channel_order) - 1] = np.deg2rad(value)
            elif 'Zrotation' in channel:
                channel_order.append('Z')
                euler_angles[len(channel_order) - 1] = np.deg2rad(value)
    
    # Most BVH files use ZXY or ZYX order
    # Build rotation matrix by composing rotations
    if not channel_order:
        return np.eye(3)
    
    # Apply rotations in the order specified by channels
    rotation = np.eye(3)
    
    for i, axis in enumerate(channel_order):
        angle = euler_angles[i]
        
        if axis == 'X':
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
        elif axis == 'Y':
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
        elif axis == 'Z':
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
        else:
            R = np.eye(3)
        
        rotation = rotation @ R
    
    return rotation


def compute_bvh_fk_batch(bvh_loader: BVHLoader,
                         frame_indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
    """
    Compute FK for multiple frames.
    """
    if frame_indices is None:
        frame_indices = list(range(bvh_loader.frames))
    
    # Get joint names from first frame
    first_frame_fk = compute_bvh_fk(bvh_loader, frame_indices[0])
    joint_names = list(first_frame_fk.keys())
    
    # Initialize result arrays
    result = {name: np.zeros((len(frame_indices), 3)) for name in joint_names}
    
    # Compute FK for each frame
    for i, frame_idx in enumerate(frame_indices):
        frame_fk = compute_bvh_fk(bvh_loader, frame_idx)
        for name in joint_names:
            result[name][i] = frame_fk[name]
    
    return result
