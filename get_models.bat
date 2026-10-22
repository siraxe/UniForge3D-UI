@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Download every model file needed by UniForge3D, Hi3D v1.0 and LATO.2 into:
rem   models\<owner>\<repo>\...
rem
rem Public sources only:
rem   Hugging Face: xibushu/UniForge3D, microsoft/TRELLIS.2-4B,
rem                 microsoft/TRELLIS-image-large,
rem                 hi3d-official/Hi3Dv1.0 (Run Hi3D 1536),
rem                 facebook/dinov2-large and
rem                 openai/clip-vit-large-patch14 (Hi3D encoders),
rem                 0x4c48/LATO.2 (Run LATO.2)
rem   ModelScope:   public mirrors of gated facebook/dinov3... and
rem                 briaai/RMBG-2.0
rem   Meta:         dl.fbaipublicfiles.com DINOv2-L/14-reg (LATO.2 encoder)
rem
rem Hi3D v1.0 adds about 10.7 GB; LATO.2 adds about 4.8 GB.
rem Existing files are skipped on rerun.
rem
rem CelloCut (Run CelloCut watertight button) needs NO model download; it
rem ships as compiled wheels in .whl\ installed by install.bat.
rem
rem No models\hf_cache is created: files are fetched directly.
rem
rem Usage:
rem   get_models.bat             ask which model sets to download, then
rem                              download/resume, skipping existing files
rem   get_models.bat --check     report missing files only (no prompt)
rem   get_models.bat --verify    also re-hash existing files (no prompt)
rem   get_models.bat --groups main|hi3d|lato2|all
rem                              select a model set without the prompt
rem                              (combine with commas, e.g. main,lato2)
rem
rem Optional env:
rem   UNIFORGE_NO_PAUSE=1
rem   UNIFORGE_GROUPS=main|hi3d|lato2|all   skip the selection prompt

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

rem If command-line arguments were given, use them as-is (advanced mode).
set "ARGS=%*"

if not defined ARGS if defined UNIFORGE_GROUPS set "ARGS=--groups %UNIFORGE_GROUPS%"
if defined ARGS goto have_args

set "SEL_MAIN= "
set "SEL_HI3D= "
set "SEL_LATO= "

call :show_menu

:ask_main
choice /c YN /n /m "   Download Main models? [Y/N] "
if errorlevel 2 (set "SEL_MAIN=0") else (set "SEL_MAIN=1")
call :show_menu
choice /c YN /n /m "   Download Hi3Dv1.0?   [Y/N] "
if errorlevel 2 (set "SEL_HI3D=0") else (set "SEL_HI3D=1")
call :show_menu
choice /c YN /n /m "   Download LATO.2?     [Y/N] "
if errorlevel 2 (set "SEL_LATO=0") else (set "SEL_LATO=1")
call :show_menu
echo.
if "%SEL_MAIN%"=="0" if "%SEL_HI3D%"=="0" if "%SEL_LATO%"=="0" (
    echo   Nothing selected. Please pick at least one model set.
    echo.
    goto ask_main
)
if "%SEL_MAIN%"=="1" if "%SEL_HI3D%"=="1" if "%SEL_LATO%"=="1" set "ARGS=--groups all"
if "%SEL_MAIN%"=="1" if "%SEL_HI3D%"=="1" if "%SEL_LATO%"=="0" set "ARGS=--groups main,hi3d"
if "%SEL_MAIN%"=="1" if "%SEL_HI3D%"=="0" if "%SEL_LATO%"=="1" set "ARGS=--groups main,lato2"
if "%SEL_MAIN%"=="0" if "%SEL_HI3D%"=="1" if "%SEL_LATO%"=="1" set "ARGS=--groups hi3d,lato2"
if "%SEL_MAIN%"=="1" if "%SEL_HI3D%"=="0" if "%SEL_LATO%"=="0" set "ARGS=--groups main"
if "%SEL_MAIN%"=="0" if "%SEL_HI3D%"=="1" if "%SEL_LATO%"=="0" set "ARGS=--groups hi3d"
if "%SEL_MAIN%"=="0" if "%SEL_HI3D%"=="0" if "%SEL_LATO%"=="1" set "ARGS=--groups lato2"
goto have_args

:show_menu
rem Clears the screen and redraws the list; call after each answer.
set "CB_MAIN=[ ]"
set "CB_HI3D=[ ]"
set "CB_LATO=[ ]"
if "%SEL_MAIN%"=="1" set "CB_MAIN=[x]"
if "%SEL_HI3D%"=="1" set "CB_HI3D=[x]"
if "%SEL_LATO%"=="1" set "CB_LATO=[x]"
cls
echo ============================================================
echo   Which models do you want to download?
echo ============================================================
echo.
echo   %CB_MAIN% Main models
echo         UniForge3D + TRELLIS checkpoints + DINOv3/RMBG
echo         preprocessors. Required for image-to-3D.
echo.
echo   %CB_HI3D% Hi3Dv1.0
echo         Hi3D refinement model + DINOv2/CLIP encoders.
echo         Required for "Run Hi3D 1536". Adds about 10.7 GB.
echo.
echo   %CB_LATO% LATO.2
echo         V-Flow/T-Flow mesh generation checkpoints + DINOv2-L/14-reg
echo         encoder. Required for "Run LATO.2". Adds about 4.8 GB.
echo.
exit /b 0

:have_args
echo [get-models] Destination: models\ (grouped by owner and repo, no hf_cache)
%PY_CMD% "tools\download_models.py" %ARGS%
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
