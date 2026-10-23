@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Download every model file needed by UniForge3D and Hi3D v1.0 into:
rem   models\<owner>\<repo>\...
rem
rem Public sources only:
rem   Hugging Face: xibushu/UniForge3D, microsoft/TRELLIS.2-4B,
rem                 microsoft/TRELLIS-image-large,
rem                 hi3d-official/Hi3Dv1.0 (Run Hi3D 1536),
rem                 facebook/dinov2-large and
rem                 openai/clip-vit-large-patch14 (Hi3D encoders)
rem   ModelScope:   public mirrors of gated facebook/dinov3... and
rem                 briaai/RMBG-2.0
rem
rem Hi3D v1.0 adds about 10.7 GB. Existing files are skipped on rerun.
rem
rem No models\hf_cache is created: files are fetched directly.
rem
rem Usage:
rem   get_models.bat             download/resume, skip existing files
rem   get_models.bat --check     report missing files only
rem   get_models.bat --verify    also re-hash existing files
rem
rem Optional env:
rem   UNIFORGE_NO_PAUSE=1

set "PY_CMD="
if exist ".venv\Scripts\python.exe" set "PY_CMD=.venv\Scripts\python.exe"

if not defined PY_CMD (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -3.12"
)

if not defined PY_CMD (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo [get-models] Python 3.10+ was not found.
    echo [get-models] Install Python 3.12 x64 or run install.bat first.
    if not defined UNIFORGE_NO_PAUSE pause
    exit /b 1
)

set "PYTHONUNBUFFERED=1"
echo [get-models] Destination: models\ (grouped by owner and repo, no hf_cache)
%PY_CMD% "tools\download_models.py" %*
set "status=%ERRORLEVEL%"

if not "%status%"=="0" (
    echo.
    echo [get-models] FAILED with exit code %status%.
    echo [get-models] The model set is incomplete or a download failed.
    echo [get-models] Run get_models.bat again to download or resume.
    if not defined UNIFORGE_NO_PAUSE pause
    exit /b %status%
)

echo.
echo [get-models] Done.
if not defined UNIFORGE_NO_PAUSE pause
exit /b 0
