@echo off
chcp 65001 >nul
echo ========================================
echo   会议智能分析系统 - 本地部署
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Python 未安装！请先安装 Python 3.10 或更高版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [信息] 检测到 Python
python --version
echo.

REM 检查是否在虚拟环境中
python -c "import sys; print('Virtual env:', hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))" | findstr "True" >nul 2>&1
if %errorlevel% neq 0 (
    echo [信息] 未检测到虚拟环境，正在创建...
    if not exist "venv" (
        python -m venv venv
        if %errorlevel% neq 0 (
            echo [错误] 虚拟环境创建失败
            pause
            exit /b 1
        )
    )
    echo [信息] 激活虚拟环境...
    call venv\Scripts\activate.bat
)

echo.
echo [1/3] 安装依赖包...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if %errorlevel% neq 0 (
    echo [警告] 依赖安装可能有问题，请检查错误信息
)

echo.
echo [2/3] 检查 FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] FFmpeg 未安装，部分音频功能可能受限
    echo 下载地址: https://ffmpeg.org/download.html
) else (
    echo [信息] FFmpeg 已检测到
)

echo.
echo [3/3] 启动 API 服务...
echo.
echo ========================================
echo   系统启动中...
echo ========================================
echo.
echo   Web 页面地址:
echo   http://localhost:5000
echo.
echo   按 Ctrl+C 停止服务
echo.

python app.py
