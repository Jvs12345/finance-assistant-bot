@echo off
setlocal EnableDelayedExpansion

echo ===============================================================================
echo   Financial Document Assistant - Startup Script
echo ===============================================================================
echo.

REM 1. Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running!
    echo.
    echo Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)

REM 2. Ensure Ollama is running and reachable on port 11434
echo [INFO] Checking Ollama...
curl.exe -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Ollama is not running. Starting Ollama server...
    where ollama >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Ollama is not installed or not in PATH.
        echo         Install from https://ollama.com and try again.
        echo.
        pause
        exit /b 1
    )

    start "Ollama Server" cmd /c "set OLLAMA_HOST=0.0.0.0:11434 && ollama serve"

    echo [INFO] Waiting for Ollama to become ready...
    set OLLAMA_READY=0
    for /L %%i in (1,1,20) do (
        curl.exe -s http://localhost:11434/api/tags >nul 2>&1
        if not errorlevel 1 (
            set OLLAMA_READY=1
            goto :ollama_ready
        )
        timeout /t 1 /nobreak >nul
    )

    :ollama_ready
    if "!OLLAMA_READY!"=="0" (
        echo [ERROR] Ollama did not start in time on http://localhost:11434
        echo         Try running: ollama serve
        echo.
        pause
        exit /b 1
    )
)
echo [OK] Ollama is running.

REM 3. Start services without rebuilding/downloading dependencies
echo [INFO] Starting services (no rebuild)...
cd /d "%~dp0"
docker compose -p financial-bot up -d --no-build elasticsearch postgres app
if errorlevel 1 (
    echo [ERROR] Could not start services without build.
    echo.
    echo Run DOWNLOAD_DEPENDENSIES.bat first, then run this script again.
    echo.
    pause
    exit /b 1
)

REM 4. Wait for the application to be ready
echo.
echo [INFO] Waiting for application to initialize...
echo        This may take a few seconds...
timeout /t 10 /nobreak >nul

REM 5. Open the assistant in the browser
echo.
echo [SUCCESS] System is up and running!
echo.
echo Opening: http://localhost:8100/index.html
start "" "http://localhost:8100/index.html"

echo.
echo ===============================================================================
echo   Services are running in the background.
echo   You can close this window.
echo ===============================================================================
echo.
pause
