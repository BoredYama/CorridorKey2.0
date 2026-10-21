@echo off
rem CorridorKey 2.0 GUI launcher. Extra arguments go to gui\app.py.
cd /d "%~dp0\.."

if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" gui\app.py %*
pause
