@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"
set "PYTHONPATH=%CD%"

echo.
echo ========================================
echo   Index Existing Reference Files
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found
    echo Create it first: python -m venv .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

if not exist "Existing_files" mkdir Existing_files

set EXISTING_COUNT=0
for %%f in (Existing_files\*.pdf) do set /a EXISTING_COUNT+=1

if %EXISTING_COUNT%==0 (
    echo [ERROR] No PDF files found in Existing_files\
    pause
    exit /b 1
)

echo Found %EXISTING_COUNT% existing PDF file(s):
for %%f in (Existing_files\*.pdf) do echo   - %%~nxf
echo.

set "BULK_CHUNK_SIZE=500"
set "TEXT_CHUNK_SIZE=10000"
set "ELASTICSEARCH_URL=http://localhost:39200"

echo [INFO] Rebuilding index from Existing_files only...
python scripts\optimized_indexer.py --pdf-dir __none__ --existing-dir Existing_files --chunk-size %BULK_CHUNK_SIZE% --text-chunk-size %TEXT_CHUNK_SIZE% --yes

if errorlevel 1 (
    echo [ERROR] Existing-files indexing failed
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Existing reference documents indexed.
echo.
pause
