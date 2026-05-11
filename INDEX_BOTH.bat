@echo off
setlocal EnableDelayedExpansion

rem =============================================================================
rem Index Financial/Tax Documents for Grounded Q&A
rem =============================================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PYTHONPATH=%CD%"

echo.
echo ========================================
echo   Financial Document Indexing
echo   Uses scripts\optimized_indexer.py
echo ========================================
echo.

rem Check virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found
    echo.
    echo Please create it first:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

rem Activate virtual environment
call .venv\Scripts\activate.bat

rem Ensure Elasticsearch is running (required for indexing)
echo [INFO] Checking Elasticsearch on http://localhost:39200 ...
curl.exe -s http://localhost:39200/_cluster/health >nul 2>&1
if errorlevel 1 (
    echo [WARN] Elasticsearch is not reachable. Starting Docker service...

    docker info >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Docker is not running and Elasticsearch is not reachable.
        echo         Start Docker Desktop or manually start Elasticsearch on port 39200.
        echo.
        pause
        exit /b 1
    )

    docker compose -p financial-bot up -d elasticsearch >nul 2>&1

    set ES_READY=0
    for /L %%i in (1,1,60) do (
        curl.exe -s http://localhost:39200/_cluster/health >nul 2>&1
        if not errorlevel 1 (
            set ES_READY=1
            goto :es_ready
        )
        timeout /t 1 /nobreak >nul
    )

    :es_ready
    if "!ES_READY!"=="0" (
        echo [ERROR] Elasticsearch did not become ready on port 39200 in time.
        echo.
        pause
        exit /b 1
    )
)
echo [OK] Elasticsearch is available.
echo.

rem Ensure folders exist
if not exist "Source_files" mkdir Source_files
if not exist "Existing_files" mkdir Existing_files

rem Count PDFs in both groups
set SOURCE_COUNT=0
for %%f in (Source_files\*.pdf) do set /a SOURCE_COUNT+=1

set EXISTING_COUNT=0
for %%f in (Existing_files\*.pdf) do set /a EXISTING_COUNT+=1

set /a PDF_COUNT=%SOURCE_COUNT%+%EXISTING_COUNT%

if %PDF_COUNT%==0 (
    echo [ERROR] No PDF files found in Source_files\ or Existing_files\
    echo.
    pause
    exit /b 1
)

echo Found %PDF_COUNT% PDF file(s) total:
echo   Existing_files: %EXISTING_COUNT%
for %%f in (Existing_files\*.pdf) do echo     [existing] %%~nxf
echo   Source_files: %SOURCE_COUNT%
for %%f in (Source_files\*.pdf) do echo     [uploaded] %%~nxf
echo.

rem Run optimized indexer (clears/recreates Elasticsearch index)
set "BULK_CHUNK_SIZE=500"
set "TEXT_CHUNK_SIZE=10000"
set "ELASTICSEARCH_URL=http://localhost:39200"
echo [INFO] Running scripts\optimized_indexer.py with:
echo        bulk chunk size: %BULK_CHUNK_SIZE%
echo        text chunk size: %TEXT_CHUNK_SIZE%
echo        Elasticsearch URL: %ELASTICSEARCH_URL%
echo        PYTHONPATH: %PYTHONPATH%
echo.
python scripts\optimized_indexer.py --pdf-dir Source_files --existing-dir Existing_files --chunk-size %BULK_CHUNK_SIZE% --text-chunk-size %TEXT_CHUNK_SIZE% --no-prune-pdf-files --yes

if errorlevel 1 (
    echo.
    echo ========================================
    echo [ERROR] INDEXING FAILED
    echo ========================================
    echo Check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] INDEXING COMPLETE
echo ========================================
echo.
echo The Elasticsearch index used by the LLM Q&A flow has been rebuilt.
echo Next: run RUN_FINANCIAL_ASSISTANT.bat and try a query.
echo.
pause
