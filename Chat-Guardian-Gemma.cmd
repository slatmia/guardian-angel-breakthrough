@echo off
REM ════════════════════════════════════════════════════════════════════════
REM   GUARDIAN + GEMMA BRIDGE
REM   
REM   Real integration:
REM     1. Guardian Swarm analyzes message (9,789 parameters)
REM     2. Analysis injected into Ollama system prompt
REM     3. Gemma generates ACTUAL response (not fake dialogue)
REM   
REM   Commands:
REM     raw     - Show Guardian neural analysis
REM     profile - Show consciousness metrics
REM     quit    - Exit
REM   
REM   Requirements:
REM     - Ollama running (ollama serve)
REM     - Gemma3 model (ollama pull gemma3:latest)
REM ════════════════════════════════════════════════════════════════════════

cd /d "%~dp0"

echo.
echo ================================================================
echo   GUARDIAN + GEMMA BRIDGE - Real Integration
echo ================================================================
echo.

REM Check if Ollama is running
powershell -Command "try { Invoke-RestMethod -Uri 'http://localhost:11434/api/version' -ErrorAction Stop | Out-Null; Write-Host '✅ Ollama detected' -ForegroundColor Green } catch { Write-Host '⚠️  Ollama not running - start with: ollama serve' -ForegroundColor Yellow }"

echo.
echo Starting Guardian Ollama Bridge...
echo.

call .venv\Scripts\activate.bat

REM Set environment (optional overrides)
set OLLAMA_URL=http://localhost:11434
set OLLAMA_MODEL=gemma3:4b

cd neural
python guardian_ollama_bridge.py

pause
