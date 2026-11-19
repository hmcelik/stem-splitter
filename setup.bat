@echo off
setlocal
cd /d "%~dp0"
title Stem Splitter - setup

echo === Stem Splitter setup ===
echo.

rem ---- find Python (3.12 / 3.11 preferred, 3.13 works) ----
set PY=
for %%V in (3.12 3.11 3.13) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set PY=py -%%V
    )
)
if not defined PY (
    python -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1 && set PY=python
)
if not defined PY if not exist .venv\Scripts\python.exe (
    echo Python was not found - installing Python 3.12 with winget...
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
if not defined PY if not exist .venv\Scripts\python.exe (
    echo.
    echo Python could not be installed automatically.
    echo Install Python 3.12 from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^)
    echo and run this script again.
    pause
    exit /b 1
)

if not exist .venv\Scripts\python.exe (
    echo Using: %PY%
    %PY% -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip wheel || goto :fail

rem ---- 1. app requirements (may pull a CPU-only PyTorch from PyPI as a dependency) ----
python -m pip install -r requirements.txt || goto :fail
python -m pip uninstall -y torchaudio >nul 2>&1

rem ---- 2. LAST step: replace PyTorch with the CUDA build so nothing can overwrite it ----
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :nogpu

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 goto :check

echo.
echo NVIDIA GPU detected - installing the CUDA build of PyTorch ^(about 3 GB^)...
python -m pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo cu128 build failed, trying cu126...
    python -m pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126 || goto :fail
)
goto :check

:nogpu
echo No NVIDIA GPU detected - using CPU PyTorch.
echo Separation will work but is slow ^(several minutes per song^).
echo For an AMD/Intel GPU run:  .venv\Scripts\pip install torch-directml

:check
echo.
python -c "import onnxruntime, audio_separator.separator; print('Packages OK')" || goto :fail
python -c "from stemsplitter.engine import Engine; print('Device:', Engine.device_info())"
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if errorlevel 1 (
        echo.
        echo WARNING: an NVIDIA GPU is present but PyTorch cannot use it.
        echo Update the NVIDIA driver, then run setup.bat again.
    )
)
echo.
echo Setup complete. Start the app with run.bat
echo ^(models download automatically the first time each one is used, 0.3-0.8 GB each^)
pause
exit /b 0

:fail
echo.
echo Setup failed - see the messages above.
pause
exit /b 1
