
import pickle
import numpy as np
import os
import sys

# 解决 numpy 版本兼容性问题
import numpy as np
from types import ModuleType
class FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)
sys.modules['numpy._core'] = FakeModule('numpy._core', np.core if hasattr(np, 'core') else np)
sys.modules['numpy._core.multiarray'] = FakeModule('numpy._core.multiarray', getattr(np.core, 'multiarray', None))


def inspect_pkl(file_path, label):
    print(f"\n{'='*20} {label} {'='*20}")
    print(f"File: {file_path}")
    
    if not os.path.exists(file_path):
        print("File not found!")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            print("Type: dict")
            for k, v in data.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                    # 打印前几个值作为示例，帮助判断单位或坐标系
                    if k in ['root_pos', 'root_rot', 'dof_pos'] and v.size > 0:
                        print(f"    sample[0]: {v[0]}")
                elif isinstance(v, (int, float, str, bool)):
                    print(f"  {k}: {v}")
                elif isinstance(v, list):
                     print(f"  {k}: len={len(v)}")
                else:
                    print(f"  {k}: {type(v)}")
        else:
            print(f"Type: {type(data)}")
            
    except Exception as e:
        print(f"Error reading pkl: {e}")

# 1. 分析宇树 G1 的示例数据
g1_file = r"d:\Desktop\TWIST2\assets\example_motions\0807_yanjie_walk_001.pkl"
inspect_pkl(g1_file, "Unitree G1 Data (Standard)")

# 2. 分析天工的数据 (取其中一个文件)
tienkung_dir = r"d:\Desktop\TWIST2\pyroki\outputs_new"
if os.path.exists(tienkung_dir):
    files = [f for f in os.listdir(tienkung_dir) if f.endswith('.pkl')]
    if files:
        tienkung_file = os.path.join(tienkung_dir, files[0])
        inspect_pkl(tienkung_file, "Tienkung Data (To Verify)")
    else:
        print(f"\nNo .pkl files found in {tienkung_dir}")
else:
    print(f"\nDirectory not found: {tienkung_dir}")
