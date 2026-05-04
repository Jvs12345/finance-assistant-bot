@echo off
echo Stopping Ollama...
taskkill /F /IM "ollama.exe" >nul 2>&1
taskkill /F /IM "ollama app.exe" >nul 2>&1

echo Starting Ollama on 0.0.0.0:11434...
set OLLAMA_HOST=0.0.0.0:11434
start "" "ollama" serve

echo Waiting for Ollama to be ready...
timeout /t 5 /nobreak >nul

echo Testing connection...
curl -I http://localhost:11434
if %errorlevel% equ 0 (
    echo [OK] Ollama is running and accessible!
) else (
    echo [ERROR] Could not connect to Ollama.
)
pause
