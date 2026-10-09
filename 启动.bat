@echo off
chcp 65001 >nul
title Meeting Compliance Analysis System
cd /d "%~dp0"

REM ============================================================
REM  This .bat file is intentionally ASCII-only.
REM  cmd.exe cannot reliably parse UTF-8 batch files: multi-byte
REM  Chinese characters shift the parser and produce bogus
REM  "not recognized as an internal or external command" errors.
REM  All Chinese messages are printed by launcher.py instead.
REM ============================================================

echo ============================================================
echo   Meeting Compliance Analysis System
echo   (voice transcription / speaker diarization / compliance)
echo ============================================================
echo.

REM Probe candidate interpreters and keep the first that really runs
REM Python 3.10+ . py.exe may exist but find no runtime, so test it.
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
    echo.
    echo   Please install Python 3.10 or newer and make sure
    echo   "Add Python to PATH" is checked during installation:
    echo   https://www.python.org/downloads/
    echo.
    echo   After installing, close this window and run this file again.
    echo.
    pause
    exit /b 1
)

echo [INFO] Using interpreter: %PY%
%PY% --version
echo.

REM Hand over to the launcher: dependency check, model check,
REM free-port selection, start service, wait, open browser.
%PY% -X utf8 launcher.py %*
set "EXITCODE=%errorlevel%"

echo.
if not "%EXITCODE%"=="0" (
    echo [ERROR] Startup failed with exit code %EXITCODE%.
    echo         Run "launcher.py --check" for an environment report.
    echo.
    pause
)
exit /b %EXITCODE%
