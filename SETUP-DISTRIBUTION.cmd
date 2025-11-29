@echo off
REM ========================================
REM   GUARDIAN SWARM - Quick Start
REM ========================================

echo.
echo ========================================
echo   GUARDIAN SWARM SETUP
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Install Python 3.11+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv .venv

echo [2/5] Activating environment...
call .venv\Scripts\activate.bat

echo [3/5] Installing dependencies...
pip install -q torch requests fastapi uvicorn pydantic

echo [4/5] Checking Ollama...
curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama not running
    echo Start Ollama manually: ollama serve
    echo Then: ollama pull gemma3:4b
) else (
    echo [OK] Ollama detected
)

echo [5/5] Setup complete!
echo.
echo ========================================
echo   READY TO RUN
echo ========================================
echo.
echo Start multi-agent debate:
echo   python neural\multi_voice_debate.py
echo.
echo Start Guardian server:
echo   python -m uvicorn guardian_server:app --port 11436
echo.
pause
