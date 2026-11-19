@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
    echo First start: installing Stem Splitter.
    echo This downloads about 3-4 GB ^(Python packages^) and takes a while - only needed once.
    echo.
    call setup.bat || exit /b 1
)
start "" .venv\Scripts\pythonw.exe app.py
