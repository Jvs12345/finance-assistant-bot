@echo off
setlocal EnableDelayedExpansion

echo ===============================================================================
echo   Financial Document Assistant - Startup Script
echo ===============================================================================
echo.

set "ROOT=%~dp0"
cd /d "%ROOT%"
set "TARGET_MODEL=gemma3:4b"
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /B /I "OLLAMA_MODEL=" ".env"`) do (
        if not "%%B"=="" set "TARGET_MODEL=%%B"
    )
)
echo [INFO] Target Ollama model: %TARGET_MODEL%
echo.

set "OLLAMA_EXE="
for /f "delims=" %%I in ('where ollama 2^>nul') do if not defined OLLAMA_EXE set "OLLAMA_EXE=%%I"
if not defined OLLAMA_EXE if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined OLLAMA_EXE if exist "%ProgramFiles%\Ollama\ollama.exe" set "OLLAMA_EXE=%ProgramFiles%\Ollama\ollama.exe"
if not defined OLLAMA_EXE (
    echo [ERROR] Ollama is not installed or not in PATH.
    echo         Install from https://ollama.com and try again.
    echo.
    pause
    exit /b 1
)
echo [OK] Using Ollama executable: %OLLAMA_EXE%
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
curl.exe -sS http://127.0.0.1:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Ollama is not running. Starting Ollama server...
    start "Ollama Server" cmd /c "set OLLAMA_HOST=0.0.0.0:11434 && ""%OLLAMA_EXE%"" serve"

    echo [INFO] Waiting for Ollama to become ready...
    set OLLAMA_READY=0
    for /L %%i in (1,1,30) do (
        curl.exe -sS http://127.0.0.1:11434/api/tags >nul 2>&1
        if not errorlevel 1 (
            set OLLAMA_READY=1
            goto :ollama_ready
        )
        timeout /t 1 /nobreak >nul
    )

    :ollama_ready
    if "!OLLAMA_READY!"=="0" (
        echo [ERROR] Ollama did not start in time on http://127.0.0.1:11434
        echo         Try running: ollama serve
        echo.
        pause
        exit /b 1
    )
)
echo [OK] Ollama is running.

REM 2b. Verify configured model exists and is callable
echo [INFO] Verifying Ollama model is installed: %TARGET_MODEL%
"%OLLAMA_EXE%" list | findstr /I /C:"%TARGET_MODEL%" >nul
if errorlevel 1 (
    echo [ERROR] Ollama model not found: %TARGET_MODEL%
    echo         Run: ollama pull %TARGET_MODEL%
    echo.
    pause
    exit /b 1
)
echo [OK] Model is installed.

echo [INFO] Running Ollama model health probe...
curl.exe -sS -X POST "http://127.0.0.1:11434/api/generate" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"%TARGET_MODEL%\",\"prompt\":\"health check\",\"stream\":false,\"options\":{\"num_predict\":8,\"temperature\":0}}" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is reachable, but model %TARGET_MODEL% did not respond correctly.
    echo         Try running FIX_OLLAMA.bat
    echo.
    pause
    exit /b 1
)
echo [OK] Ollama model probe passed.

REM 3. Start services without rebuilding/downloading dependencies
echo [INFO] Starting services (no rebuild)...
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
