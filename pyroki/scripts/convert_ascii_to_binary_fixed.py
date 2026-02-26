"""
ASCII FBX to Binary FBX Converter
使用Blender 2.79转换ASCII格式的FBX为二进制格式

使用方法:
    blender --background --python this_script.py -- ascii_file.fbx binary_file.fbx
"""

import bpy
import sys
import os

def ascii_to_binary(ascii_fbx_path, binary_fbx_path):
    """
    将ASCII格式FBX转换为二进制格式FBX

    Args:
        ascii_fbx_path: ASCII格式的FBX文件路径
        binary_fbx_path: 输出的二进制FBX文件路径
    """

    print('正在导入ASCII FBX: {}'.format(ascii_fbx_path))

    # 清空场景
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 导入ASCII FBX (Blender 2.79支持)
    try:
        bpy.ops.import_scene.fbx(
            filepath=ascii_fbx_path,
            use_custom_props=True,
            use_image_search=True
        )
        print('✓ 成功导入ASCII FBX')
    except Exception as e:
        print('✗ 导入失败: {}'.format(e))
        return False

    # 查找导入的对象
    imported_objects = bpy.context.scene.objects
    print('  导入了 {} 个对象'.format(len(imported_objects)))

    # 导出为Binary FBX
    print('正在导出为Binary FBX: {}'.format(binary_fbx_path))

    try:
        bpy.ops.export_scene.fbx(
            filepath=binary_fbx_path,
            version='BIN7400',  # 二进制7.4版本
            use_selection=False,
            global_scale=1.0,
            use_anim=True,
            use_anim_optimize=False,
            path_mode='COPY'
        )
        print('✓ 成功导出Binary FBX')

        # 检查输出文件大小
        file_size = os.path.getsize(binary_fbx_path)
        print('  输出文件大小: {:.2f} MB'.format(file_size/1024.0/1024.0))

        return True
    except Exception as e:
        print('✗ 导出失败: {}'.format(e))
        return False

if __name__ == '__main__':
    # 解析命令行参数
    # Blender的argv格式: blender [options] -- script.py [args]
    # 我们需要找到 '--' 后面的参数
    if len(sys.argv) < 7:
        print('错误: 缺少参数')
        print('用法: blender --background --python this_script.py -- input_ascii.fbx output_binary.fbx')
        sys.exit(1)

    # 查找参数
    ascii_file = None
    binary_file = None

    for i, arg in enumerate(sys.argv):
        if arg == '--' and i + 1 < len(sys.argv):
            if ascii_file is None:
                ascii_file = sys.argv[i + 1]
            elif binary_file is None:
                binary_file = sys.argv[i + 1]
            break

    if not ascii_file or not binary_file:
        print('错误: 无法解析输入参数')
        print('收到的参数: {}'.format(sys.argv))
        sys.exit(1)

    print('=' * 60)
    print('ASCII FBX → Binary FBX 转换工具')
    print('=' * 60)
    print('输入: {}'.format(ascii_file))
    print('输出: {}'.format(binary_file))
    print()

    # 检查输入文件是否存在
    if not os.path.exists(ascii_file):
        print('✗ 错误: 输入文件不存在: {}'.format(ascii_file))
        sys.exit(1)

    # 执行转换
    success = ascii_to_binary(ascii_file, binary_file)

    if success:
        print()
        print('=' * 60)
        print('✓ 转换完成！')
        print('=' * 60)
        sys.exit(0)
    else:
        print()
        print('=' * 60)
        print('✗ 转换失败！')
        print('=' * 60)
        sys.exit(1)
