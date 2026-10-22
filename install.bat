@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Recreate .venv with the pinned runtime. Local wheels come from .whl/;
rem flash_attn is downloaded from Hugging Face.
rem Usage:
rem   install.bat            reuse/create .venv
rem   install.bat --fresh    delete .venv first, then recreate
rem   install.bat --check    only verify Python 3.12 x64 and the .whl files
rem
rem Optional env overrides:
rem   UNIFORGE_VENV=.venv-test
rem   UNIFORGE_NO_PAUSE=1

if not defined UNIFORGE_VENV set "UNIFORGE_VENV=.venv"
set "VENV=%UNIFORGE_VENV%"
set "WHEEL_DIR=.whl"
set "PY="
set "CHECK_ONLY="
set "MISSING_WHEEL="

if /I "%~1"=="--check" set "CHECK_ONLY=1"
if /I "%~1"=="--fresh" (
    if exist "!VENV!" (
        echo [install] Removing existing !VENV! ...
        rmdir /s /q "!VENV!"
        if exist "!VENV!" (
            echo [install] Could not remove !VENV!. Close anything using it and retry.
            goto :fail
        )
    )
)

if /I "!VENV!"=="." (
    echo [install] Refusing to use the project root as the venv.
    goto :fail
)

for %%W in (
    cumesh-0.0.1-cp312-cp312-win_amd64.whl
    flex_gemm-1.0.0-cp312-cp312-win_amd64.whl
    o_voxel-0.0.1-cp312-cp312-win_amd64.whl
    nvdiffrast-0.4.0-cp312-cp312-win_amd64.whl
) do (
    if not exist "!WHEEL_DIR!\%%W" (
        echo [install] Missing wheel: !WHEEL_DIR!\%%W
        set "MISSING_WHEEL=1"
    )
)
if defined MISSING_WHEEL (
    echo [install] Copy the .whl directory from the working setup, or recreate it with:
    echo             .venv\Scripts\python.exe tools\export_wheels.py
    goto :fail
)

rem ---- locate Python 3.12 x64 ----
py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) and sys.maxsize>2**32 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_LAUNCHER=py -3.12"
) else (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) and sys.maxsize>2**32 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_LAUNCHER=python"
    ) else (
        echo [install] Python 3.12 x64 was not found.
        echo [install] Install Python 3.12 x64 and ensure 'py -3.12' or 'python' works.
        goto :fail
    )
)

if defined CHECK_ONLY (
    echo [install] Check OK: Python 3.12 x64 found and all local wheels are present.
    exit /b 0
)

if not exist "!VENV!\Scripts\python.exe" (
    echo [install] Creating !VENV! using !PY_LAUNCHER! ...
    !PY_LAUNCHER! -m venv "!VENV!"
    if errorlevel 1 goto :fail
) else (
    echo [install] Reusing existing !VENV! ^(use install.bat --fresh to rebuild^).
)

set "PY=!VENV!\Scripts\python.exe"

echo [install] Python:
"!PY!" -c "import sys, platform; print(sys.version.split()[0], platform.architecture()[0])"
if errorlevel 1 goto :fail

echo [install] Installing pip 24.3.1 ...
"!PY!" -m pip install --disable-pip-version-check --upgrade "pip==24.3.1"
if errorlevel 1 goto :fail

echo [install] Installing pinned requirements ...
echo [install] torch/torchvision source: https://download.pytorch.org/whl/cu130
"!PY!" -m pip install --disable-pip-version-check --prefer-binary --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple -r requirements.txt
if errorlevel 1 goto :fail

echo [install] Installing local compiled wheels from !WHEEL_DIR! ...
"!PY!" -m pip install --disable-pip-version-check --no-index --no-deps --find-links "!WHEEL_DIR!" "cumesh==0.0.1" "flex_gemm==1.0.0" "o_voxel==0.0.1" "nvdiffrast==0.4.0"
if errorlevel 1 goto :fail

set "FLASH_ATTN_URL=https://huggingface.co/siraxe/wheels/resolve/main/flash_attn-2.8.3%%2Bcu130torch2.13-cp312-cp312-win_amd64.whl"
echo [install] Installing flash_attn from Hugging Face ...
"!PY!" -m pip install --disable-pip-version-check --no-deps "!FLASH_ATTN_URL!"
if errorlevel 1 goto :fail

echo [install] Smoke test ...
"!PY!" -c "import torch, torchvision, flash_attn, cumesh, flex_gemm, o_voxel, nvdiffrast; import uniforge3d; print('torch', torch.__version__); print('cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('uniforge3d OK')"
if errorlevel 1 goto :fail

echo.
echo [install] Done.
echo [install] CLI:  !PY! infer.py --help
echo [install] UI:   start_ps6.bat
if not defined UNIFORGE_NO_PAUSE pause
exit /b 0

:fail
echo.
echo [install] FAILED. See the messages above.
if not defined UNIFORGE_NO_PAUSE pause
exit /b 1
