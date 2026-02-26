# ASCII FBX转换解决方案

## ❌ 问题说明

你的FBX文件 `SIK_Actor_01_20260209_135451.fbx` 是**纯ASCII格式**，而Blender 2.79和5.0都不再支持导入ASCII格式FBX。

**验证**:
```
文件头: ; FBX 7.5.0 project file
错误: ASCII FBX files are not supported
```

---

## ✅ 推荐解决方案（按优先级）

### 方案1：使用在线转换器（最简单）⭐⭐⭐

**优点**:
- 无需安装任何软件
- 操作简单，上传下载即可
- 转换质量高

**步骤**:

1. 访问在线转换网站:
   ```
   https://www.greentoken.de/online/
   或
   https://3dconverter.com/
   ```

2. 上传文件:
   - 选择: `SIK_Actor_01_20260209_135451.fbx`
   - 输出格式: Binary FBX

3. 下载转换后的文件

4. 重命名并移动到项目目录:
   ```bash
   # 下载后
   mv SIK_Actor_01_20260209_135451_binary.fbx data/
   cd D:\Desktop\pyroki
   "D:\DevTools\Blender\blender.exe" --background --python scripts/convert_binary_to_bvh.py -- data/SIK_Actor_01_20260209_135451_binary.fbx data/bvh_outputs/SIK_Actor_01_20260209_135451.bvh
   ```

---

### 方案2：使用Python FBX库（推荐开发者）⭐⭐

**优点**:
- 完全自动化
- 可以批量处理
- 可集成到Python工作流

**安装**:
```bash
pip install fbx
# 或
pip install pyfbx
```

**转换脚本**:
```python
import fbx
import os

# 读取ASCII FBX
input_file = 'data/fbx/SIK_Actor_01_20260209_135451.fbx'
output_file = 'data/SIK_Actor_01_20260209_135451_binary.fbx'

# 转换
reader = fbx.FBXReader(input_file)
data = reader.read()

# 保存为Binary FBX
writer = fbx.FBXWriter(output_file)
writer.set_binary_mode(True)
writer.write(data)

print('转换完成！')
```

---

### 方案3：重新导出为Binary（如果可能）⭐

**如果你有原始软件**:
1. 在原始软件中打开FBX文件
2. 重新导出，选择以下选项:
   - 格式: FBX 7.5
   - 类型: Binary（而非ASCII/Text）
   - 保留动画: Yes
3. 使用新的Binary FBX文件

**推荐的导出设置**:
- FBX版本: 7.4或7.5
- 坐标系: Y-up（与Blender一致）
- 单位: 厘米或米
- 动画: 包含
- 材质/纹理: 可选

---

### 方案4：使用Autodesk FBX SDK（高级用户）

**下载**:
```
https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk
```

**步骤**:
1. 下载并安装FBX SDK
2. 使用命令行工具转换:
   ```bash
   FBXConverter.exe -input SIK_Actor_01_20260209_135451.fbx -output SIK_Actor_01_20260209_135451_binary.fbx -binary
   ```

---

## 🚀 快速开始（推荐方案1）

### 第1步：在线转换

访问: https://www.greentoken.de/online/

1. 点击"选择文件"
2. 选择 `SIK_Actor_01_20260209_135451.fbx`
3. 输出格式: FBX (Binary)
4. 点击"转换"
5. 下载转换后的文件

### 第2步：使用Binary FBX

将下载的文件重命名为 `SIK_Actor_01_20260209_135451_binary.fbx`

移动到项目根目录: `D:\Desktop\pyroki\`

### 第3步：转换为BVH

```bash
cd D:\Desktop\pyroki

"D:\DevTools\Blender\blender.exe" --background --python scripts/convert_binary_to_bvh.py -- SIK_Actor_01_20260209_135451_binary.fbx data/bvh_outputs/SIK_Actor_01_20260209_135451.bvh
```

---

## 📊 转换后验证

### 1. 检查BVH文件
```bash
python scripts/inspect_bvh.py data/bvh_outputs/SIK_Actor_01_20260209_135451.bvh
```

### 2. 测试加载
```bash
python -c "
from utils.bvh_loader import BVHLoader
loader = BVHLoader('data/bvh_outputs/SIK_Actor_01_20260209_135451.bvh')
print('BVH加载成功')
print('  帧数: {}'.format(loader.frames))
print('  FPS: {:.2f}'.format(1.0/loader.frame_time))
print('  骨骼数: {}'.format(len(loader.nodes)))
"
```

### 3. 运行重定向
```bash
python run_simple.py SIK_Actor_01_20260209_135451.bvh
```

---

## 💡 为什么Blender不支持ASCII FBX？

**历史原因**:
1. ASCII FBX是旧格式，占用空间大
2. 解析速度慢，效率低
3. 安全性差，容易解析错误
4. Binary FBX更紧凑、更快、更可靠

**移除时间**:
- Blender 2.80开始逐步弃用
- Blender 5.0完全移除支持

---

## ❓ 常见问题

### Q1: 为什么我的FBX是ASCII格式？

**A**: 可能的原因:
- 导出软件默认选择ASCII格式
- 或在导出时选择了"文本"或"ASCII"选项

### Q2: 哪个方案最可靠？

**A**: 按可靠性排序:
1. 方案3（重新导出）- 最可靠，使用原始软件
2. 方案2（Python库）- 可靠，需要编程
3. 方案1（在线转换）- 可靠，依赖网站
4. 方案4（FBX SDK）- 最可靠，最复杂

### Q3: 转换后会丢失数据吗？

**A**: 不会。ASCII到Binary的转换只是格式变化，所有内容（骨骼、动画、材质）都会保留。

### Q4: 如果在线转换器无法处理怎么办？

**A**: 尝试其他在线转换器:
- https://3dconverter.com/
- https://www.meshconvert.com/
- https://conv3d.com/

---

## 🎯 总结

| 方案 | 难度 | 可靠性 | 推荐度 |
|------|-------|---------|---------|
| 在线转换器 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Python FBX库 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 重新导出Binary | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| FBX SDK | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |

**推荐**: 先尝试方案1（在线转换），如不行则用方案2（Python库）。

---

## 📝 下一步

获得Binary FBX后，执行:

```bash
cd D:\Desktop\pyroki

# 转换为BVH
"D:\DevTools\Blender\blender.exe" --background --python scripts/convert_binary_to_bvh.py -- SIK_Actor_01_20260209_135451_binary.fbx data/bvh_outputs/SIK_Actor_01_20260209_135451.bvh

# 验证
python scripts/inspect_bvh.py data/bvh_outputs/SIK_Actor_01_20260209_135451.bvh

# 测试
python run_simple.py SIK_Actor_01_20260209_135451.bvh
```

---

需要帮助？查看完整的转换工具文档:
`FBX_CONVERSION_GUIDE.md`
