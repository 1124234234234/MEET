@echo off
chcp 65001 >nul
echo ========================================
echo   项目备份脚本
echo ========================================
echo.

set BACKUP_DIR=backup
set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set BACKUP_NAME=voice-reco_backup_%TIMESTAMP%.zip

if not exist "%BACKUP_DIR%" (
    mkdir "%BACKUP_DIR%"
)

echo [信息] 正在创建备份: %BACKUP_NAME%
echo.

powershell -Command "Compress-Archive -Path * -DestinationPath '%BACKUP_DIR%\%BACKUP_NAME%' -Force -CompressionLevel Optimal"

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   备份成功！
    echo   文件: %BACKUP_DIR%\%BACKUP_NAME%
    echo ========================================
) else (
    echo [错误] 备份失败
)

echo.
pause
