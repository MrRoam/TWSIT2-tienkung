@echo off
REM ========================================
REM FBX到BVH完整转换工具
REM
REM 功能: 将ASCII FBX转换为BVH格式
REM 流程: ASCII FBX → Binary FBX → BVH
REM
REM 使用方法:
REM     fbx_conversion_complete.bat input.fbx [output.bvh]
REM ========================================

setlocal enabledelayedexpansion

REM 设置路径
set BLENDER79=D:\Desktop\pyroki\tools\blender_2.79\blender-2.79b-windows64\blender.exe
set BLENDER50=D:\DevTools\Blender\blender.exe
set SCRIPTS=%~dp0

REM 检查输入参数
set INPUT_FBX=%~1
if "%INPUT_FBX%"=="" (
    echo ========================================
    echo 错误: 未指定输入FBX文件
    echo ========================================
    echo.
    echo 用法: fbx_conversion_complete.bat input.fbx [output.bvh]
    echo.
    echo 参数说明:
    echo   input.fbx  - 输入的FBX文件路径
    echo   output.bvh - 输出的BVH文件路径(可选)
    echo.
    echo 示例:
    echo   fbx_conversion_complete.bat data\fbx\SIK_Actor_01_20260209_135451.fbx
    echo   fbx_conversion_complete.bat data\fbx\myfile.fbx data\bvh_outputs\myfile.bvh
    echo.
    pause
    exit /b 1
)

REM 检查输入文件是否存在
if not exist "%INPUT_FBX%" (
    echo ========================================
    echo 错误: 输入文件不存在
    echo ========================================
    echo 文件: %INPUT_FBX%
    echo.
    echo 请检查文件路径是否正确
    pause
    exit /b 1
)

REM 设置输出文件名
if "%~2"=="" (
    REM 如果未指定输出，使用与FBX相同的文件名
    for %%I in ("%INPUT_FBX%") do (
        set OUTPUT=%%~dpI%%~nI.bvh
    )
) else (
    set OUTPUT=%~2
)

REM 生成临时Binary FBX文件名
for %%I in ("%INPUT_FBX%") do (
    set TEMP_BINARY=%~dpI%%~nI_binary.fbx
)

echo ========================================
echo FBX到BVH完整转换工具
echo ========================================
echo.
echo 输入FBX: %INPUT_FBX%
echo 临时文件: %TEMP_BINARY%
echo 输出BVH: %OUTPUT%
echo.

REM 检查Blender 2.79是否安装
if not exist "%BLENDER79%" (
    echo ========================================
    echo 错误: 未找到Blender 2.79
    echo ========================================
    echo 路径: %BLENDER79%
    echo.
    echo 请先下载并安装Blender 2.79:
    echo   1. 下载: https://download.blender.org/release/Blender2.79/blender-2.79b-windows64.zip
    echo   2. 解压到: D:\Desktop\pyroki\tools\blender_2.79\
    echo.
    echo 或修改此批处理文件中的BLENDER79路径
    pause
    exit /b 1
)

echo.
echo ========================================
echo 第1步: ASCII FBX → Binary FBX
echo ========================================
echo.
echo 使用Blender 2.79转换...
echo.

"%BLENDER79%" --background --python "%SCRIPTS%convert_ascii_to_binary.py" -- "%INPUT_FBX%" "%TEMP_BINARY%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo 错误: 第1步失败！
    echo ========================================
    echo 请检查:
    echo   1. FBX文件是否是有效的ASCII格式
    echo   2. Blender 2.79是否正确安装
    echo   3. 输入文件路径是否正确
    echo.
    pause
    exit /b 1
)

echo.
echo ✓ 第1步完成
echo  中间文件已生成: %TEMP_BINARY%

REM 检查临时文件是否生成
if not exist "%TEMP_BINARY%" (
    echo.
    echo ========================================
    echo 错误: 临时文件未生成
    echo ========================================
    echo 文件: %TEMP_BINARY%
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo 第2步: Binary FBX → BVH
echo ========================================
echo.
echo 使用Blender 5.0转换...
echo.
echo 正在分析骨骼结构...
echo 正在应用骨骼名称映射...
echo 正在过滤不需要的骨骼...
echo.

"%BLENDER50%" --background --python "%SCRIPTS%convert_binary_to_bvh.py" -- "%TEMP_BINARY%" "%OUTPUT%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo 错误: 第2步失败！
    echo ========================================
    echo 请检查:
    echo   1. 中间文件是否正确生成
    echo   2. Blender 5.0是否能正确导入Binary FBX
    echo   3. 骨骼结构是否包含必要的肢体
    echo.
    pause
    exit /b 1
)

echo.
echo ✓ 第2步完成
echo BVH文件已生成: %OUTPUT%

REM 检查输出文件是否生成
if not exist "%OUTPUT%" (
    echo.
    echo ========================================
    echo 错误: 输出文件未生成
    echo ========================================
    echo 文件: %OUTPUT%
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo 清理临时文件...
echo ========================================
del "%TEMP_BINARY%"

if %ERRORLEVEL% EQU 0 (
    echo ✓ 临时文件已删除
) else (
    echo ⚠ 警告: 无法删除临时文件
    echo 文件: %TEMP_BINARY%
    echo 请手动删除
)

echo.
echo ========================================
echo ✓ 转换完成！
echo ========================================
echo.
echo 输入: %INPUT_FBX%
echo 输出: %OUTPUT%
echo.
echo 下一步:
echo   1. 检查BVH文件: python scripts\inspect_bvh.py "%OUTPUT%"
echo   2. 测试加载: python -c "from utils.bvh_loader import BVHLoader; loader = BVHLoader('%OUTPUT%'); print('✓ 加载成功')"
echo   3. 运行重定向: python run_simple.py [output_filename.bvh]
echo.
echo ========================================
echo.
pause
