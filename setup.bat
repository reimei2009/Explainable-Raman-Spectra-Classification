@echo off
echo ============================================================
echo  KLTN - Raman Spectra Analysis - Environment Setup
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)
echo [OK] Python found.

REM Step 0: Generate all project source files from bootstrap.py
echo [INFO] Generating project files from bootstrap.py ...
python _extract_temp.py
if errorlevel 1 (
    echo [ERROR] Failed to generate project files. Check _extract_temp.py.
    pause
    exit /b 1
)
echo [OK] Project files generated.
echo.

REM Create virtual environment
if exist venv\ (
    echo [INFO] Virtual environment already exists. Skipping creation.
) else (
    echo [INFO] Creating virtual environment at venv\ ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded.

REM Install requirements
echo [INFO] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install some dependencies. Check the output above.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.

REM Create output directories
echo [INFO] Creating output directories...
if not exist outputs\figures\     mkdir outputs\figures
if not exist outputs\processed\   mkdir outputs\processed
if not exist outputs\reports\     mkdir outputs\reports
if not exist logs\                mkdir logs
echo [OK] Output directories created.

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  To activate the environment in a new terminal, run:
echo      venv\Scripts\activate.bat
echo.
echo  To start Jupyter Notebook:
echo      jupyter notebook
echo ============================================================
pause
