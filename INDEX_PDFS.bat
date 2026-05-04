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

rem Ensure Source_files exists
if not exist "Source_files" (
    echo [ERROR] Source_files directory not found!
    echo.
    pause
    exit /b 1
)

rem Count PDFs
set PDF_COUNT=0
for %%f in (Source_files\*.pdf) do set /a PDF_COUNT+=1

if %PDF_COUNT%==0 (
    echo [ERROR] No PDF files found in Source_files\
    echo.
    pause
    exit /b 1
)

echo Found %PDF_COUNT% PDF file(s):
for %%f in (Source_files\*.pdf) do echo   - %%~nxf
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
python scripts\optimized_indexer.py --chunk-size %BULK_CHUNK_SIZE% --text-chunk-size %TEXT_CHUNK_SIZE% --yes

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
