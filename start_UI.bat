@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo .venv not found next to this project.
    pause
    exit /b 1
)

rem ---------------------------------------------------------------------------
rem Model location - the launcher decides it. Set UNIFORGE_MODEL_DIR before
rem running this script to share a model tree with another project copy.
rem ---------------------------------------------------------------------------
if not defined UNIFORGE_MODEL_DIR if exist "%~dp0models\xibushu\UniForge3D\pipeline.json" set "UNIFORGE_MODEL_DIR=%~dp0models\xibushu\UniForge3D"
if defined UNIFORGE_MODEL_DIR echo [start-ps6] Models: "%UNIFORGE_MODEL_DIR%"
if not defined UNIFORGE_MODEL_DIR echo [start-ps6] Models: not found - run get_models.bat first.

rem ---------------------------------------------------------------------------
rem VRAM safeguards - identical to start.bat (the generation worker is shared).
rem ---------------------------------------------------------------------------
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
set "UNIFORGE_VRAM_DEBUG=1"
set "PYTHONUNBUFFERED=1"

echo [start-ps6] GPU free VRAM:
nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader 2>nul

".venv\Scripts\python.exe" PS6_UI\main.py %*
set "status=%ERRORLEVEL%"
if not "%status%"=="0" (
    echo [start-ps6] UI exited with %status%
    pause
)
exit /b %status%
