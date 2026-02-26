# FBX到BVH转换指南

## 📋 概述

本指南说明如何将ASCII格式的FBX文件转换为BVH格式，以便在PyRoki项目中使用。

**重要提示**: 你的FBX文件是ASCII格式，Blender 5.0不再支持直接导入。需要使用两步转换流程。

---

## 🛠️ 环境要求

### 必需工具

1. **Blender 2.79b** (用于ASCII → Binary FBX转换)
   - 下载地址: https://download.blender.org/release/Blender2.79/blender-2.79b-windows64.zip
   - 解压到: `D:\Desktop\pyroki\tools\blender_2.79\`
   - 这是便携版，无需安装

2. **Blender 5.0** (已安装在 `D:\DevTools\Blender\`)
   - 用于Binary FBX → BVH转换

### 目录结构

转换工具已创建以下目录和文件：

```
pyroki/
├── tools/
│   └── blender_2.79/              # 需要下载Blender 2.79b到这里
│       └── blender.exe
├── data/
│   ├── fbx/                      # FBX输入目录
│   │   └── SIK_Actor_01_20260209_135451.fbx
│   └── bvh_outputs/              # BVH输出目录
│       └── [生成的BVH文件]
└── scripts/
    ├── convert_ascii_to_binary.py   # ASCII→Binary转换脚本
    ├── convert_binary_to_bvh.py     # Binary→BVH转换脚本
    └── fbx_conversion_complete.bat  # 完整转换批处理
```

---

## 🚀 快速开始

### 第1步：下载并安装Blender 2.79

1. 下载Blender 2.79b:
   ```
   https://download.blender.org/release/Blender2.79/blender-2.79b-windows64.zip
   ```

2. 解压到指定位置:
   ```
   解压到: D:\Desktop\pyroki\tools\blender_2.79\
   ```

3. 验证安装:
   ```bash
   D:\Desktop\pyroki\tools\blender_2.79\blender.exe --version
   ```

   应该输出: `Blender 2.79 (sub 0)`

### 第2步：执行转换

#### 单个文件转换

```bash
cd D:\Desktop\pyroki
scripts\fbx_conversion_complete.bat data\fbx\SIK_Actor_01_20260209_135451.fbx
```

#### 指定输出文件名

```bash
scripts\fbx_conversion_complete.bat data\fbx\SIK_Actor_01_20260209_135451.fbx data\bvh_outputs\custom_name.bvh
```

---

## 📊 转换流程详解

### 第1步：ASCII FBX → Binary FBX

使用Blender 2.79转换ASCII格式为二进制格式。

**输入**: ASCII格式FBX (你的原始文件)
**输出**: Binary格式FBX (临时文件)

**处理内容**:
- 导入ASCII FBX
- 保留所有对象和动画数据
- 导出为Binary FBX 7.4版本

### 第2步：Binary FBX → BVH

使用Blender 5.0将Binary FBX转换为标准BVH格式。

**输入**: Binary FBX (第1步的输出)
**输出**: BVH格式文件

**处理内容**:
1. **骨骼分析**
   - 自动检测骨骼数量和名称
   - 识别动画数据（帧数、FPS）
   - 查找根骨骼

2. **骨骼过滤**
   - 自动排除以下骨骼:
     - 手指骨骼 (finger, thumb, index, middle, ring, pinky)
     - 脚趾骨骼 (toe)

3. **骨骼名称映射** ⭐
   - 自动将FBX骨骼名称映射到项目标准
   - 支持常见命名约定:
     - 完整名称: `LeftArm`, `RightShoulder`
     - 缩写: `L_Arm`, `R_Shoulder`
     - 点分隔: `Left.Arm`, `Right.Shoulder`

   **示例映射**:
   | FBX名称 | 项目标准 |
   |---------|----------|
   | Hips | Hips |
   | Pelvis | Hips |
   | L_Arm | LeftArm |
   | RightForeArm | RightForeArm |
   | R_Foot | RightFoot |

4. **BVH导出**
   - 保留肢体动画
   - 导出格式: BVH
   - FPS: 30
   - 旋转模式: NATIVE

---

## ✅ 验证转换结果

### 1. 检查BVH文件

```bash
python scripts\inspect_bvh.py data\bvh_outputs\SIK_Actor_01_20260209_135451.bvh
```

### 2. 测试加载

```bash
python -c "
from utils.bvh_loader import BVHLoader
loader = BVHLoader('data/bvh_outputs/SIK_Actor_01_20260209_135451.bvh')
print('✓ BVH加载成功')
print(f'  帧数: {loader.frames}')
print(f'  FPS: {1/loader.frame_time:.2f}')
print(f'  骨骼数: {len(loader.nodes)}')
"
```

### 3. 运行完整流程

```bash
# 使用run_simple.py处理
python run_simple.py SIK_Actor_01_20260209_135451.bvh

# 或者直接在data/raw_data/lafan1/目录下创建软链接/副本
# 然后使用现有流程
```

---

## 🔧 高级配置

### 修改骨骼映射

如果自动映射不正确，可以编辑 `scripts/convert_binary_to_bvh.py`:

```python
# 在文件开头的BONE_MAPPING字典中添加你的映射
BONE_MAPPING = {
    'YourOriginalBoneName': 'StandardBoneName',
    # ... 其他映射
}
```

### 修改排除规则

在 `scripts/convert_binary_to_bvh.py` 中修改 `EXCLUDE_KEYWORDS`:

```python
# 添加或删除排除关键词
EXCLUDE_KEYWORDS = [
    'finger', 'thumb', 'index', 'middle', 'ring', 'pinky',
    'toe', 'digit',
    # 添加其他要排除的关键词
]
```

### 修改Blender路径

如果Blender安装位置不同，编辑 `scripts/fbx_conversion_complete.bat`:

```batch
set BLENDER79=你的Blender2.79路径\blender.exe
set BLENDER50=你的Blender5.0路径\blender.exe
```

---

## ❓ 常见问题

### Q1: 转换失败，提示"未找到Blender 2.79"

**A**: 需要先下载并解压Blender 2.79b到正确位置。

### Q2: 第1步成功，但第2步失败

**A**: 检查以下几点:
- 中间文件(临时Binary FBX)是否正确生成
- Blender 5.0是否能正确导入Binary FBX
- FBX文件是否包含完整的骨架结构

### Q3: 转换后的BVH骨骼名称与项目标准不符

**A**:
1. 查看转换输出中的"骨骼名称映射"部分
2. 手动添加缺失的映射到 `convert_binary_to_bvh.py`
3. 重新运行转换

### Q4: 转换后的动画看起来不正确

**A**: 可能的原因:
- 缩放比例问题: 检查 `global_scale` 参数
- 坐标系问题: 检查 `rotate_mode` 参数
- 骨骼层次不同: FBX和项目BVH的骨骼结构不同

### Q5: 想跳过第1步，直接使用Binary FBX

**A**: 可以直接调用第2步脚本:
```bash
"D:\DevTools\Blender\blender.exe" --background --python scripts\convert_binary_to_bvh.py -- your_binary.fbx output.bvh
```

---

## 📝 示例输出

### 转换成功的完整输出:

```
============================================================
FBX到BVH完整转换工具
============================================================

输入FBX: data\fbx\SIK_Actor_01_20260209_135451.fbx
临时文件: D:\Desktop\pyroki\SIK_Actor_01_20260209_135451_binary.fbx
输出BVH: data\bvh_outputs\SIK_Actor_01_20260209_135451.bvh

============================================================
第1步: ASCII FBX → Binary FBX
============================================================

使用Blender 2.79转换...

正在导入ASCII FBX: data\fbx\SIK_Actor_01_20260209_135451.fbx
✓ 成功导入ASCII FBX
  导入了 3 个对象
正在导出为Binary FBX: D:\Desktop\pyroki\SIK_Actor_01_20260209_135451_binary.fbx
✓ 成功导出Binary FBX
  输出文件大小: 15.23 MB

============================================================
ASCII FBX → Binary FBX 转换工具
============================================================
输入: data\fbx\SIK_Actor_01_20260209_135451.fbx
输出: D:\Desktop\pyroki\SIK_Actor_01_20260209_135451_binary.fbx

============================================================
✓ 转换完成！
============================================================

✓ 第1步完成
  中间文件已生成: D:\Desktop\pyroki\SIK_Actor_01_20260209_135451_binary.fbx

============================================================
第2步: Binary FBX → BVH
============================================================

使用Blender 5.0转换...

正在导入Binary FBX: D:\Desktop\pyroki\SIK_Actor_01_20260209_135451_binary.fbx
✓ 成功导入Binary FBX
  骨架名称: Armature
  总骨骼数: 78
  动作名称: Mixamo.com
  总帧数: 125
  起始帧: 1
  结束帧: 125

============================================================
骨骼分析和映射
============================================================
包含骨骼数: 45
排除骨骼数: 33

排除的骨骼:
  - LeftHandThumb1
  - LeftHandIndex1
  - LeftHandMiddle1
  - RightHandThumb1
  - RightHandIndex1
  ... 还有 28 个

骨骼名称映射:
  L_Arm → LeftArm
  R_ForeArm → RightForeArm
  L_Leg → LeftLeg
  R_Foot → RightFoot
  ... 还有 11 个映射

============================================================
正在导出BVH
============================================================
输出路径: data\bvh_outputs\SIK_Actor_01_20260209_135451.bvh
✓ BVH导出成功
  输出文件大小: 245.78 KB

✓ 第2步完成
  BVH文件已生成: data\bvh_outputs\SIK_Actor_01_20260209_135451.bvh

============================================================
清理临时文件...
============================================================
✓ 临时文件已删除

============================================================
✓ 转换完成！
============================================================

输入: data\fbx\SIK_Actor_01_20260209_135451.fbx
输出: data\bvh_outputs\SIK_Actor_01_20260209_135451.bvh

下一步:
  1. 检查BVH文件: python scripts\inspect_bvh.py "data\bvh_outputs\SIK_Actor_01_20260209_135451.bvh"
  2. 测试加载: python -c "from utils.bvh_loader import BVHLoader; loader = BVHLoader('data\bvh_outputs\SIK_Actor_01_20260209_135451.bvh'); print('✓ 加载成功')"
  3. 运行重定向: python run_simple.py [output_filename.bvh]

============================================================
```

---

## 🎯 下一步

转换成功后，可以:

1. **使用现有流程处理**
   ```bash
   python run_simple.py data\bvh_outputs\SIK_Actor_01_20260209_135451.bvh
   ```

2. **批量转换**
   将多个FBX文件放入 `data/fbx/` 目录
   循环调用转换工具

3. **自定义映射**
   根据需要修改骨骼映射表

---

## 📚 参考资料

- Blender 2.79下载: https://download.blender.org/release/Blender2.79/
- Blender 5.0文档: https://docs.blender.org/
- BVH格式规范: https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html
