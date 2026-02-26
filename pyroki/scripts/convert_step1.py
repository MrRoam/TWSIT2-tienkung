"""
FBX to BVH Conversion Script - All-in-One
这个脚本整合了完整的转换流程，使用绝对路径
"""

import bpy
import sys
import os

# 配置路径
BLENDER79 = r'D:\Desktop\pyroki\tools\blender_2.79\blender-2.79b-windows64\blender-2.79b-windows64\blender.exe'
BLENDER50 = r'D:\DevTools\Blender\blender.exe'
PROJECT_DIR = r'D:\Desktop\pyroki'

def step1_ascii_to_binary(ascii_fbx, binary_fbx):
    """第1步: ASCII FBX -> Binary FBX"""
    print('=' * 70)
    print('第1步: ASCII FBX -> Binary FBX')
    print('=' * 70)
    print()

    # 清空场景
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 导入ASCII FBX
    print('正在导入ASCII FBX: {}'.format(ascii_fbx))
    try:
        bpy.ops.import_scene.fbx(
            filepath=ascii_fbx,
            use_custom_props=True,
            use_image_search=True
        )
        print('✓ 成功导入ASCII FBX')
    except Exception as e:
        print('✗ 导入失败: {}'.format(e))
        return False

    # 导出为Binary FBX
    print('正在导出为Binary FBX: {}'.format(binary_fbx))
    try:
        bpy.ops.export_scene.fbx(
            filepath=binary_fbx,
            version='BIN7400',
            use_selection=False,
            global_scale=1.0,
            use_anim=True,
            use_anim_optimize=False,
            path_mode='COPY'
        )
        print('✓ 成功导出Binary FBX')

        # 检查输出文件
        file_size = os.path.getsize(binary_fbx)
        print('  输出文件大小: {:.2f} MB'.format(file_size/1024.0/1024.0))
        print()
        return True
    except Exception as e:
        print('✗ 导出失败: {}'.format(e))
        return False

if __name__ == '__main__':
    print('=' * 70)
    print('FBX到BVH完整转换工具')
    print('=' * 70)
    print()

    # 设置输入输出路径
    ascii_fbx = os.path.join(PROJECT_DIR, 'data', 'fbx', 'SIK_Actor_01_20260209_135451.fbx')
    binary_fbx = os.path.join(PROJECT_DIR, 'SIK_Actor_01_20260209_135451_binary.fbx')
    bvh_output = os.path.join(PROJECT_DIR, 'data', 'bvh_outputs', 'SIK_Actor_01_20260209_135451.bvh')

    print('输入FBX: {}'.format(ascii_fbx))
    print('临时Binary: {}'.format(binary_fbx))
    print('输出BVH: {}'.format(bvh_output))
    print()

    # 检查输入文件
    if not os.path.exists(ascii_fbx):
        print('✗ 错误: 输入FBX文件不存在')
        sys.exit(1)

    # 检查Blender 2.79
    if not os.path.exists(BLENDER79):
        print('✗ 错误: Blender 2.79未找到')
        print('  路径: {}'.format(BLENDER79))
        sys.exit(1)

    # 执行第1步: ASCII -> Binary
    step1_success = step1_ascii_to_binary(ascii_fbx, binary_fbx)

    if not step1_success:
        print()
        print('=' * 70)
        print('✗ 第1步失败')
        print('=' * 70)
        sys.exit(1)

    print()
    print('=' * 70)
    print('✓ 第1步完成！请继续执行第2步')
    print('=' * 70)
    print()
    print('第2步将使用Blender 5.0执行')
    print('请运行以下命令:')
    print()
    print('"{}" --background --python scripts/convert_binary_to_bvh.py -- "{}" "{}"'.format(
        BLENDER50, binary_fbx, bvh_output))
    print()
