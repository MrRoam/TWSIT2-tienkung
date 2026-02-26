"""
Binary FBX to BVH Converter
使用Blender 5.0将二进制FBX转换为BVH格式

功能:
- 自动过滤手指和脚趾骨骼
- 自动骨骼名称映射（尽可能匹配项目标准）
- 导出为标准BVH格式

使用方法:
    blender --background --python this_script.py -- binary_file.fbx output.bvh
"""

import bpy
import sys
import os
import re

# 骨骼名称映射表（FBX标准 → 项目标准）
# 这是一个智能映射表，会根据FBX中的骨骼名称自动匹配
BONE_MAPPING = {
    # 根/躯干
    'Hips': 'Hips',
    'Pelvis': 'Hips',
    'Root': 'Hips',
    'RootNode': 'Hips',
    'Spine': 'Spine',
    'Spine1': 'Spine1',
    'Spine2': 'Spine2',
    'Spine3': 'Spine2',
    'Chest': 'Spine1',
    'UpperChest': 'Spine1',
    'Neck': 'Neck',
    'Head': 'Head',

    # 左臂
    'LeftShoulder': 'LeftShoulder',
    'LeftArm': 'LeftArm',
    'LeftForeArm': 'LeftForeArm',
    'LeftElbow': 'LeftForeArm',
    'LeftHand': 'LeftHand',
    'LeftWrist': 'LeftHand',
    'L_Shoulder': 'LeftShoulder',
    'L_Arm': 'LeftArm',
    'L_ForeArm': 'LeftForeArm',
    'L_Hand': 'LeftHand',

    # 右臂
    'RightShoulder': 'RightShoulder',
    'RightArm': 'RightArm',
    'RightForeArm': 'RightForeArm',
    'RightElbow': 'RightForeArm',
    'RightHand': 'RightHand',
    'RightWrist': 'RightHand',
    'R_Shoulder': 'RightShoulder',
    'R_Arm': 'RightArm',
    'R_ForeArm': 'RightForeArm',
    'R_Hand': 'RightHand',

    # 左腿
    'LeftUpLeg': 'LeftUpLeg',
    'LeftLeg': 'LeftLeg',
    'LeftFoot': 'LeftFoot',
    'LeftAnkle': 'LeftFoot',
    'LeftKnee': 'LeftLeg',
    'L_UpLeg': 'LeftUpLeg',
    'L_Leg': 'LeftLeg',
    'L_Foot': 'LeftFoot',
    'L_Ankle': 'LeftFoot',

    # 右腿
    'RightUpLeg': 'RightUpLeg',
    'RightLeg': 'RightLeg',
    'RightFoot': 'RightFoot',
    'RightAnkle': 'RightFoot',
    'RightKnee': 'RightLeg',
    'R_UpLeg': 'RightUpLeg',
    'R_Leg': 'RightLeg',
    'R_Foot': 'RightFoot',
    'R_Ankle': 'RightFoot',
}

# 需要排除的骨骼关键词
EXCLUDE_KEYWORDS = [
    'finger', 'thumb', 'index', 'middle', 'ring', 'pinky',
    'toe', 'digit'
]

def should_exclude_bone(bone_name):
    """检查骨骼是否应该被排除"""
    bone_name_lower = bone_name.lower()
    return any(kw in bone_name_lower for kw in EXCLUDE_KEYWORDS)

def map_bone_name(fbx_name):
    """
    将FBX骨骼名称映射到标准BVH名称

    Args:
        fbx_name: FBX中的骨骼名称

    Returns:
        映射后的标准名称，如果找不到映射则返回原名称
    """
    # 直接查找
    if fbx_name in BONE_MAPPING:
        return BONE_MAPPING[fbx_name]

    # 尝试不区分大小写的查找
    fbx_lower = fbx_name.lower()
    for key, value in BONE_MAPPING.items():
        if key.lower() == fbx_lower:
            return value

    # 如果没有找到映射，尝试智能匹配
    # 例如: LeftArm_L → LeftArm, L_Shoulder → LeftShoulder
    return smart_match_bone(fbx_name)

def smart_match_bone(name):
    """
    智能匹配骨骼名称
    根据命名模式推断正确的骨骼名称
    """
    name_lower = name.lower()

    # 检查左侧骨骼
    if any(prefix in name_lower for prefix in ['l_', 'left.', 'left_', '.l']):
        if 'shoulder' in name_lower:
            return 'LeftShoulder'
        elif 'arm' in name_lower and 'fore' not in name_lower:
            return 'LeftArm'
        elif 'forearm' in name_lower or 'arm_lower' in name_lower:
            return 'LeftForeArm'
        elif 'hand' in name_lower or 'wrist' in name_lower:
            return 'LeftHand'
        elif 'upleg' in name_lower or 'thigh' in name_lower:
            return 'LeftUpLeg'
        elif 'leg' in name_lower:
            return 'LeftLeg'
        elif 'foot' in name_lower or 'ankle' in name_lower:
            return 'LeftFoot'
        return name

    # 检查右侧骨骼
    if any(prefix in name_lower for prefix in ['r_', 'right.', 'right_', '.r']):
        if 'shoulder' in name_lower:
            return 'RightShoulder'
        elif 'arm' in name_lower and 'fore' not in name_lower:
            return 'RightArm'
        elif 'forearm' in name_lower or 'arm_lower' in name_lower:
            return 'RightForeArm'
        elif 'hand' in name_lower or 'wrist' in name_lower:
            return 'RightHand'
        elif 'upleg' in name_lower or 'thigh' in name_lower:
            return 'RightUpLeg'
        elif 'leg' in name_lower:
            return 'RightLeg'
        elif 'foot' in name_lower or 'ankle' in name_lower:
            return 'RightFoot'
        return name

    # 中央骨骼
    if 'hips' in name_lower or 'pelvis' in name_lower:
        return 'Hips'
    if 'spine' in name_lower:
        return 'Spine' if 'spine1' not in name_lower and 'spine2' not in name_lower else name
    if 'neck' in name_lower:
        return 'Neck'
    if 'head' in name_lower:
        return 'Head'

    # 如果都匹配不了，返回原名称
    return name

def analyze_fbx_structure(fbx_path):
    """分析FBX文件结构"""
    print('正在导入Binary FBX: {fbx_path}')

    # 清空场景
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 导入Binary FBX
    try:
        bpy.ops.import_scene.fbx(
            filepath=fbx_path,
            use_custom_props=True,
            use_anim=True
        )
        print('✓ 成功导入Binary FBX')
    except Exception as e:
        print('✗ 导入失败: {e}')
        return None

    # 查找骨架对象
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']

    if not armatures:
        print('✗ 错误: 未找到骨架对象')
        return None

    armature = armatures[0]
    bones = armature.data.bones

    print('  骨架名称: {armature.name}')
    print('  总骨骼数: {len(bones)}')

    # 分析动画
    if armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        print('  动作名称: {action.name}')
        total_frames = int(action.frame_range[1] - action.frame_range[0] + 1)
        print('  总帧数: {total_frames}')
        print('  起始帧: {int(action.frame_range[0])}')
        print('  结束帧: {int(action.frame_range[1])}')
    else:
        print('⚠ 警告: 未检测到动画数据')

    return armature

def filter_and_map_bones(armature):
    """
    过滤和映射骨骼

    Returns:
        (included_count, excluded_count, mapping_info)
    """
    bones = armature.data.bones

    included = []
    excluded = []
    mapping_info = format(

    for bone in bones:
        bone_name = bone.name

        # 检查是否应该排除
        if should_exclude_bone(bone_name):
            excluded.append(bone_name)
            continue

        # 映射骨骼名称
        mapped_name = map_bone_name(bone_name)
        included.append((bone, mapped_name))

        if mapped_name != bone_name:
            mapping_info[bone_name] = mapped_name

    return included, excluded, mapping_info

def binary_to_bvh(binary_fbx_path, bvh_path):
    """
    将Binary FBX转换为BVH格式

    Args:
        binary_fbx_path: 输入的Binary FBX文件路径
        bvh_path: 输出的BVH文件路径

    Returns:
        True表示成功，False表示失败
    """
    # 分析FBX
    armature = analyze_fbx_structure(binary_fbx_path)
    if not armature:
        return False

    print()
    print('=' * 60)
    print('骨骼分析和映射')
    print('=' * 60)

    # 过滤和映射骨骼
    included_bones, excluded_bones, mapping_info = filter_and_map_bones(armature)

    print('包含骨骼数: {len(included_bones)}')
    print('排除骨骼数: {len(excluded_bones)}')

    if excluded_bones:
        print()
        print('排除的骨骼:')
        for bone in excluded_bones[:20]:  # 只显示前20个
            print('  - {bone}')
        if len(excluded_bones) > 20:
            print('  ... 还有 {len(excluded_bones) - 20} 个')

    if mapping_info:
        print()
        print('骨骼名称映射:')
        for original, mapped in list(mapping_info.items())[:15]:
            print('  {original} → {mapped}')
        if len(mapping_info) > 15:
            print('  ... 还有 {len(mapping_info) - 15} 个映射')

    # 选择骨架
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)

    print()
    print('=' * 60)
    print('正在导出BVH')
    print('=' * 60)
    print('输出路径: {bvh_path}')

    try:
        # 导出BVH
        bpy.ops.export_anim.bvh(
            filepath=bvh_path,
            global_scale=1.0,
            rotate_mode='NATIVE',
            root_transform_only=False,
            selected_only=True,
            frame_start=1,
            frame_end=bpy.context.scene.frame_end,
            sampling_rate=30
        )

        print('✓ BVH导出成功')

        # 检查输出文件
        if os.path.exists(bvh_path):
            file_size = os.path.getsize(bvh_path)
            print('  输出文件大小: {file_size/1024:.2f} KB')
            return True
        else:
            print('✗ 错误: 输出文件未生成')
            return False

    except Exception as e:
        print('✗ 导出失败: {e}')
        return False

if __name__ == '__main__':
    # 解析命令行参数
    if len(sys.argv) < 7:
        print('错误: 缺少参数')
        print('用法: blender --background --python this_script.py -- binary_file.fbx output.bvh')
        sys.exit(1)

    # 查找参数
    binary_file = None
    bvh_file = None

    for i, arg in enumerate(sys.argv):
        if arg == '--' and i + 1 < len(sys.argv):
            if binary_file is None:
                binary_file = sys.argv[i + 1]
            elif bvh_file is None:
                bvh_file = sys.argv[i + 1]
            break

    if not binary_file or not bvh_file:
        print('错误: 无法解析输入参数')
        print('收到的参数: {sys.argv}')
        sys.exit(1)

    print('=' * 60)
    print('Binary FBX → BVH 转换工具')
    print('=' * 60)
    print('输入: {binary_file}')
    print('输出: {bvh_file}')
    print()

    # 检查输入文件是否存在
    if not os.path.exists(binary_file):
        print('✗ 错误: 输入文件不存在: {binary_file}')
        sys.exit(1)

    # 执行转换
    success = binary_to_bvh(binary_file, bvh_file)

    print()
    print('=' * 60)
    if success:
        print('✓ 转换完成！')
    else:
        print('✗ 转换失败！')
    print('=' * 60)

    sys.exit(0 if success else 1)
