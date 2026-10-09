@echo off
chcp 65001 >nul
title Meeting Compliance Analysis System
cd /d "%~dp0"

REM Kept for compatibility with the older documented startup name.
REM Equivalent to the main launcher batch ; all Chinese output is printed
REM by launcher.py because cmd.exe cannot reliably parse UTF-8 batch files.

set "PY="

python -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"

if not defined PY (
    py -3 -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
    if not errorlevel 1 set "PY=py -3"
)

if not defined PY (
    python3 -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
    if not errorlevel 1 set "PY=python3"
)

if not defined PY (
    echo [ERROR] No usable Python 3.10+ interpreter found.
    echo   Install from https://www.python.org/downloads/
    echo   and check "Add Python to PATH", then run this file again.
    echo.
    pause
    exit /b 1
)

%PY% -X utf8 launcher.py %*
set "EXITCODE=%errorlevel%"
echo.
if not "%EXITCODE%"=="0" pause
exit /b %EXITCODE%
