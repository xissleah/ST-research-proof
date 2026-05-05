@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo   AI Search Start
echo ========================================
echo.
echo Current directory:
echo %cd%
echo.

REM ------------------------------------------------------------
REM 1. Check required project files
REM ------------------------------------------------------------

if not exist "run.py" (
    echo ERROR: run.py was not found.
    echo Put start.bat in the same folder as run.py.
    echo.
    pause
    exit /b 1
)

if not exist "index.html" (
    echo ERROR: index.html was not found.
    echo Put start.bat in the same folder as index.html.
    echo.
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM 2. Read lightweight config from cfg.txt
REM ------------------------------------------------------------

set "APP_PROFILE=local"
set "SEARCH_MODE=AUTO"
set "SEARCH_BACKEND=searxng"
set "MODEL_BACKEND=transformers"
set "LLM_API_BASE=http://127.0.0.1:8001/v1/chat/completions"

if exist "cfg.txt" (
    for /f "usebackq tokens=1,* delims==" %%A in ("cfg.txt") do (
        if /i "%%A"=="APP_PROFILE" set "APP_PROFILE=%%B"
        if /i "%%A"=="SEARCH_MODE" set "SEARCH_MODE=%%B"
        if /i "%%A"=="SEARCH_BACKEND" set "SEARCH_BACKEND=%%B"
        if /i "%%A"=="MODEL_BACKEND" set "MODEL_BACKEND=%%B"
        if /i "%%A"=="LLM_API_BASE" set "LLM_API_BASE=%%B"
    )
)

echo APP_PROFILE=%APP_PROFILE%
echo SEARCH_MODE=%SEARCH_MODE%
echo SEARCH_BACKEND=%SEARCH_BACKEND%
echo MODEL_BACKEND=%MODEL_BACKEND%
echo.

REM ------------------------------------------------------------
REM 3. Choose Python
REM ------------------------------------------------------------

set "PYTHON_EXE="

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo ERROR: Python was not found.
    echo Please run install_dependence.bat first, or install Python manually.
    echo.
    pause
    exit /b 1
)

echo Python:
%PYTHON_EXE% --version
echo.

REM ------------------------------------------------------------
REM 4. Start SearXNG only when web search is enabled
REM ------------------------------------------------------------

set "NEED_SEARXNG=1"
if /i "%SEARCH_MODE%"=="LOCAL" set "NEED_SEARXNG=0"
if /i "%SEARCH_MODE%"=="OFFLINE" set "NEED_SEARXNG=0"
if /i "%SEARCH_BACKEND%"=="none" set "NEED_SEARXNG=0"
if /i "%SEARCH_BACKEND%"=="local" set "NEED_SEARXNG=0"

if "%NEED_SEARXNG%"=="1" (
    if not exist "settings.yml" (
        echo ERROR: settings.yml was not found, but web search is enabled.
        echo Set SEARCH_MODE=LOCAL in cfg.txt if you only want local KB mode.
        echo.
        pause
        exit /b 1
    )

    where docker >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Docker was not found, but web search is enabled.
        echo Install Docker Desktop or set SEARCH_MODE=LOCAL in cfg.txt.
        echo.
        pause
        exit /b 1
    )

    docker info >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Docker is not running, but web search is enabled.
        echo Start Docker Desktop or set SEARCH_MODE=LOCAL in cfg.txt.
        echo.
        pause
        exit /b 1
    )

    echo Recreating SearXNG container with local settings.yml ...
    docker rm -f searxng >nul 2>nul

    docker run -d ^
      --name searxng ^
      -p 18080:8080 ^
      -v "%cd%\settings.yml:/etc/searxng/settings.yml:ro" ^
      searxng/searxng:latest

    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create SearXNG container.
        echo.
        pause
        exit /b 1
    )

    echo.
    echo SearXNG frontend:
    echo http://localhost:18080
    echo.

    echo Waiting for SearXNG to become ready ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; for($i=0; $i -lt 20; $i++){ try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:18080/search?q=test&format=json' -TimeoutSec 2; if($r.StatusCode -ge 200){ $ok=$true; break } } catch { Start-Sleep -Seconds 1 } }; if($ok){ exit 0 } else { exit 1 }" >nul 2>nul

    if errorlevel 1 (
        echo WARNING: SearXNG did not respond to the JSON test in time.
        echo The AI backend will still start, but web search may fail until SearXNG is ready.
        echo.
    ) else (
        echo SearXNG is ready.
        echo.
    )
) else (
    echo SEARCH_MODE is local/offline or SEARCH_BACKEND is disabled.
    echo Skipping Docker and SearXNG startup.
    echo.
)

REM ------------------------------------------------------------
REM 5. Optional external model backend check
REM ------------------------------------------------------------

if /i "%MODEL_BACKEND%"=="openai_compatible" (
    echo Checking external LLM endpoint:
    echo %LLM_API_BASE%
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $u='%LLM_API_BASE%'.Replace('/chat/completions','/models'); $r=Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>nul
    if errorlevel 1 (
        echo WARNING: External LLM endpoint did not respond to /models.
        echo If you use llama-server, start it before asking questions.
        echo.
    ) else (
        echo External LLM endpoint is reachable.
        echo.
    )
)

REM ------------------------------------------------------------
REM 6. Start AI Search backend
REM ------------------------------------------------------------

echo Starting AI Search backend ...
echo.
%PYTHON_EXE% run.py

echo.
echo AI Search stopped.
pause
exit /b 0
