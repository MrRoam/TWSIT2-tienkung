import numpy as np
import re
import os

class BVHNode:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.zeros(3)
        self.channels = []
        self.channel_indices = []  # Indices in the motion data frame

    def __repr__(self):
        return f"BVHNode({self.name}, children={len(self.children)})"

class BVHLoader:
    def __init__(self, filename):
        self.filename = filename
        self.nodes = {}
        self.root = None
        self.frames = 0
        self.frame_time = 0.0
        self.motion_data = None
        self.channel_names = []
        
        if filename:
            self.load(filename)

    def load(self, filename):
        self.filename = filename
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        self._parse_hierarchy(lines)
        self._parse_motion(lines)

    def _parse_hierarchy(self, lines):
        iterator = iter(lines)
        stack = []
        current_node = None
        
        # Identify start
        line = next(iterator)
        while line != "HIERARCHY":
            line = next(iterator)
        
        channel_counter = 0

        while True:
            try:
                line = next(iterator)
            except StopIteration:
                break
                
            if line == "MOTION":
                break
                
            parts = line.split()
            if not parts:
                continue
                
            token = parts[0]
            
            if token == "ROOT" or token == "JOINT":
                name = parts[1]
                new_node = BVHNode(name, parent=current_node)
                self.nodes[name] = new_node
                
                if current_node:
                    current_node.children.append(new_node)
                else:
                    self.root = new_node
                
                current_node = new_node
                stack.append(current_node)
                
            elif token == "End":
                # End Site
                new_node = BVHNode(f"{current_node.name}_End", parent=current_node)
                current_node.children.append(new_node)
                current_node = new_node
                stack.append(current_node)
                
            elif token == "OFFSET":
                current_node.offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                
            elif token == "CHANNELS":
                num_channels = int(parts[1])
                channels = parts[2:]
                current_node.channels = channels
                current_node.channel_indices = list(range(channel_counter, channel_counter + num_channels))
                channel_counter += num_channels
                
                for ch in channels:
                    self.channel_names.append(f"{current_node.name}_{ch}")
                
            elif token == "{":
                pass
                
            elif token == "}":
                stack.pop()
                if stack:
                    current_node = stack[-1]
                else:
                    current_node = None

    def _parse_motion(self, lines):
        # Find MOTION line index
        start_idx = 0
        for i, line in enumerate(lines):
            if line == "MOTION":
                start_idx = i
                break
        
        # Parse frames info
        frames_line = lines[start_idx + 1]
        frame_time_line = lines[start_idx + 2]
        
        self.frames = int(frames_line.split(":")[1].strip())
        self.frame_time = float(frame_time_line.split(":")[1].strip())
        
        # Parse data
        data_lines = lines[start_idx + 3:]
        data = []
        for line in data_lines:
            if not line.strip():
                continue
            vals = [float(x) for x in line.split()]
            data.append(vals)
            
        self.motion_data = np.array(data)

    def get_frame_data(self, frame_idx):
        if frame_idx >= self.frames:
            raise ValueError("Frame index out of bounds")
        return self.motion_data[frame_idx]

def load_bvh(filename):
    loader = BVHLoader(filename)
    return loader
