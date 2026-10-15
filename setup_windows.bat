@echo off
cd /d "%~dp0"
echo ==========================================
echo Healthcare Big Data Project - Setup
echo ==========================================
python --version
if errorlevel 1 (
  echo ERROR: Python was not found. Install Python 3.11 and tick Add Python to PATH.
  pause
  exit /b 1
)
java -version
if errorlevel 1 (
  echo ERROR: Java was not found. Install Java JDK 17 and reopen this window.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup finished. Next, put diabetic_data.csv in the data folder and run run_pipeline.bat.
pause
