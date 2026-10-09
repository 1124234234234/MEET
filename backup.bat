@echo off
chcp 65001 >nul
title Meeting Compliance Analysis System - Backup
cd /d "%~dp0"

REM ASCII-only on purpose: cmd.exe cannot reliably parse UTF-8 .bat files,
REM Chinese characters shift the parser and break the script.
REM Chinese messages are kept minimal and printed via ASCII-safe text.

set "BACKUP_DIR=backup"
set "TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "BACKUP_NAME=voice-reco_backup_%TIMESTAMP%.zip"

echo ============================================================
echo   Project backup
echo ============================================================
echo.

if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

echo [INFO] Creating: %BACKUP_DIR%\%BACKUP_NAME%
echo        (skipping models/ ~5GB, backup/, uploads/, __pycache__, .git)
echo.

powershell -NoProfile -Command "Compress-Archive -Path (Get-ChildItem -Path * -Exclude 'models','backup','uploads','__pycache__','.git' -ErrorAction SilentlyContinue) -DestinationPath '%BACKUP_DIR%\%BACKUP_NAME%' -Force -CompressionLevel Optimal"

if errorlevel 1 (
    echo [ERROR] Backup failed.
) else (
    echo ============================================================
    echo   Backup done: %BACKUP_DIR%\%BACKUP_NAME%
    echo ============================================================
)

echo.
pause
