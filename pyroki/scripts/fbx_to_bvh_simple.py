import bpy
import sys
import os

fbx_path = r'D:\Desktop\pyroki\data\fbx\SIK_Actor_01_20260209_153957.fbx'
bvh_path = r'D:\Desktop\pyroki\data\bvh_outputs\SIK_Actor_01_20260209_153957.bvh'

print('正在导入FBX: {}'.format(fbx_path))

# 清空场景
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 导入FBX
try:
    bpy.ops.import_scene.fbx(
        filepath=fbx_path,
        use_custom_props=True,
        use_anim=True
    )
    print('导入成功')
except Exception as e:
    print('导入失败: {}'.format(str(e)))
    sys.exit(1)

# 查找骨架
armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']

if not armatures:
    print('未找到骨架')
    sys.exit(1)

armature = armatures[0]
print('骨架: {}'.format(armature.name))
print('骨骼数: {}'.format(len(armature.data.bones)))

# 查找动画
if armature.animation_data and armature.animation_data.action:
    action = armature.animation_data.action
    print('动作: {}'.format(action.name))
    print('总帧数: {}'.format(int(action.frame_range[1] - action.frame_range[0] + 1)))
else:
    print('未找到动画数据')

# 导出BVH
print('正在导出BVH...')
try:
    # 获取场景帧范围
    scene = bpy.context.scene
    frame_start = 1
    frame_end = 3000  # 设置一个足够大的帧数
    
    print('帧范围: {} - {}'.format(frame_start, frame_end))
    
    bpy.ops.export_anim.bvh(
        filepath=bvh_path,
        global_scale=1.0,
        rotate_mode='NATIVE',
        frame_start=frame_start,
        frame_end=frame_end
    )
    print('BVH导出成功: {} 帧'.format(frame_end - frame_start + 1))
except Exception as e:
    print('BVH导出失败: {}'.format(str(e)))
    sys.exit(1)

print('完成!')
print('输出: {}'.format(bvh_path))
