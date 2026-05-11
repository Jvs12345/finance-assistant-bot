@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "TARGET_MODEL=gemma3:4b"
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /B /I "OLLAMA_MODEL=" ".env"`) do (
        if not "%%B"=="" set "TARGET_MODEL=%%B"
    )
)

echo ===============================================================================
echo   Ollama Repair / Health Check
echo ===============================================================================
echo Target model: %TARGET_MODEL%
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

echo [INFO] Stopping existing Ollama processes...
taskkill /F /IM "ollama.exe" >nul 2>&1
taskkill /F /IM "ollama app.exe" >nul 2>&1

echo [INFO] Starting Ollama server on 0.0.0.0:11434...
start "Ollama Server" cmd /c "set OLLAMA_HOST=0.0.0.0:11434 && ""%OLLAMA_EXE%"" serve"

echo [INFO] Waiting for Ollama API...
set "OLLAMA_READY=0"
for /L %%i in (1,1,30) do (
    curl.exe -sS http://127.0.0.1:11434/api/tags >nul 2>&1
    if not errorlevel 1 (
        set "OLLAMA_READY=1"
        goto :ready
    )
    timeout /t 1 /nobreak >nul
)

:ready
if "!OLLAMA_READY!"=="0" (
    echo [ERROR] Ollama API is not reachable on http://127.0.0.1:11434
    echo         Try running manually: ollama serve
    echo.
    pause
    exit /b 1
)
echo [OK] Ollama API reachable.

echo [INFO] Verifying model exists: %TARGET_MODEL%
"%OLLAMA_EXE%" list | findstr /I /C:"%TARGET_MODEL%" >nul
if errorlevel 1 (
    echo [ERROR] Model not found: %TARGET_MODEL%
    echo         Run: ollama pull %TARGET_MODEL%
    echo.
    pause
    exit /b 1
)
echo [OK] Model is installed.

echo [INFO] Running model health probe...
curl.exe -sS -X POST "http://127.0.0.1:11434/api/generate" ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"%TARGET_MODEL%\",\"prompt\":\"health check\",\"stream\":false,\"options\":{\"num_predict\":8,\"temperature\":0}}" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is running but model probe failed for %TARGET_MODEL%.
    echo         Try: ollama run %TARGET_MODEL%
    echo.
    pause
    exit /b 1
)

echo [SUCCESS] Ollama is healthy and model %TARGET_MODEL% is callable.
echo.
pause
exit /b 0
