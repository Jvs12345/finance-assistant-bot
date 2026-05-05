@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PYTHONPATH=%CD%"

echo.
echo ========================================
echo   Index Source Uploaded Files
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found
    echo Create it first: python -m venv .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

if not exist "Source_files" mkdir Source_files

set SOURCE_COUNT=0
for %%f in (Source_files\*.pdf) do set /a SOURCE_COUNT+=1

if %SOURCE_COUNT%==0 (
    echo [ERROR] No PDF files found in Source_files\
    pause
    exit /b 1
)

echo Found %SOURCE_COUNT% source PDF file(s):
for %%f in (Source_files\*.pdf) do echo   - %%~nxf
echo.

set "BULK_CHUNK_SIZE=500"
set "TEXT_CHUNK_SIZE=10000"
set "ELASTICSEARCH_URL=http://localhost:39200"

echo [INFO] Appending Source_files into existing index...
echo [INFO] Existing reference corpus is kept intact.
python scripts\optimized_indexer.py --pdf-dir Source_files --existing-dir __none__ --append --chunk-size %BULK_CHUNK_SIZE% --text-chunk-size %TEXT_CHUNK_SIZE% --yes

if errorlevel 1 (
    echo [ERROR] Source-files indexing failed
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Source uploaded documents indexed.
echo.
pause
