@echo off
setlocal

echo ===============================================================================
echo   Financial Document Assistant - Download Dependencies
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

cd /d "%~dp0"

REM 2. Pull base service images
echo [INFO] Pulling Elasticsearch and PostgreSQL images...
docker compose -p financial-bot pull elasticsearch postgres
if errorlevel 1 (
    echo [ERROR] Failed to pull service images.
    echo.
    pause
    exit /b 1
)

REM 3. Build app image (installs Python dependencies once and caches layers)
echo [INFO] Building app image and installing Python dependencies...
docker compose -p financial-bot build app
if errorlevel 1 (
    echo [ERROR] App image build failed.
    echo.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Dependencies downloaded and app image built.
echo You can now run: RUN_FINANCIAL_ASSISTANT.bat
echo.
pause

