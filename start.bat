@echo off
chcp 65001 >nul
echo ========================================
echo   AI语音识别与翻译系统 - Docker部署
echo ========================================
echo.

REM 检查Docker是否安装
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] Docker 未安装或未启动，请先安装并启动 Docker Desktop
    pause
    exit /b 1
)

echo [信息] 检测到 Docker，开始部署...
echo.

echo [1/3] 构建 Docker 镜像...
docker build -t voice-recognition:latest .
if %errorlevel% neq 0 (
    echo [错误] Docker 镜像构建失败
    pause
    exit /b 1
)
echo [完成] Docker 镜像构建完成
echo.

echo [2/3] 检查并清理旧容器...
docker ps -a --format "{{.Names}}" | findstr /b "voice-api" >nul 2>&1
if %errorlevel% equ 0 (
    echo [信息] 检测到已存在的容器，正在移除...
    docker rm -f voice-api
)
echo [完成] 容器清理完成
echo.

echo [3/3] 启动 API 服务容器...
docker run -d --name voice-api -p 5000:5000 --memory=6g voice-recognition:latest
if %errorlevel% neq 0 (
    echo [错误] 容器启动失败
    pause
    exit /b 1
)
echo [完成] API 容器已启动
echo.

echo ========================================
echo   系统启动成功！
echo ========================================
echo.
echo   Web 页面地址:
echo   http://localhost:5000
echo.
echo   API 地址:
echo   http://localhost:5000
echo.
echo   常用命令:
echo     查看日志:  docker logs voice-api
echo     停止容器:  docker stop voice-api
echo     重启容器:  docker restart voice-api
echo     删除容器:  docker rm -f voice-api
echo.
echo   注意: 首次启动需要下载模型，请耐心等待...
echo.
echo   正在等待服务就绪，请稍候...
echo.

REM 等待服务就绪
set /a count=0
:waitloop
set /a count+=1
if %count% gtr 30 (
    echo [警告] 等待超时，请检查日志: docker logs voice-api
    pause
    exit /b 0
)

timeout /t 3 /nobreak >nul
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [完成] API 服务已就绪！
    echo.
    echo 正在打开浏览器...
    start http://localhost:5000
    pause
    exit /b 0
)

echo 等待中... (%count%/30)
goto waitloop
