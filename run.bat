@echo off
setlocal EnableExtensions
chcp 65001 >nul
title LUNA AI - Local Windows 10 Assistant

echo ========================================================
echo          LUNA AI - Local Intelligent Assistant
echo        Target: Windows 10 - Ollama - qwen3:0.6b
echo ========================================================
echo.

cd /d "%~dp0"

set "OLLAMA_MAX_LOADED_MODELS=1"
set "OLLAMA_NUM_PARALLEL=1"
set "OLLAMA_KEEP_ALIVE=5m"
set "OLLAMA_VULKAN=1"
set "OLLAMA_IGPU_ENABLE=1"
set "OLLAMA_FLASH_ATTENTION=1"
set "OLLAMA_KV_CACHE_TYPE=q8_0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if exist ".venv\Scripts\python.exe" goto :venv_ready

where python >nul 2>nul
if not errorlevel 1 (
    echo [1/4] Creating local virtual environment .venv ...
    python -m venv .venv
    goto :check_venv_created
)

where py >nul 2>nul
if not errorlevel 1 (
    echo [1/4] Creating local virtual environment .venv via py launcher ...
    py -3 -m venv .venv
    goto :check_venv_created
)

echo [ERROR] Python 3.10+ was not found on this Windows PC.
echo Please install Python from https://www.python.org/downloads/windows/
echo and check "Add Python to PATH" during setup.
pause
exit /b 1

:check_venv_created
if exist ".venv\Scripts\python.exe" goto :venv_ready
echo [WARNING] Could not create .venv, using global Python...
set "PY_EXE=python"
goto :check_deps

:venv_ready
set "PY_EXE=.venv\Scripts\python.exe"

:check_deps
"%PY_EXE%" -c "import psutil, fastapi, uvicorn, httpx, pydantic, jsonschema, sklearn, bs4, docx, openpyxl" >nul 2>nul
if not errorlevel 1 goto :deps_ready

echo [2/4] Installing required Python libraries from requirements.txt ...
echo       This only runs once on first launch. Please wait...
"%PY_EXE%" -m pip install --upgrade pip
"%PY_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python packages. Check your internet connection.
    pause
    exit /b 1
)

echo [2/4] Installing Playwright Chromium browser for web tools...
"%PY_EXE%" -m playwright install chromium

:deps_ready
where ollama >nul 2>nul
if errorlevel 1 goto :ollama_missing

ollama list >nul 2>nul
if not errorlevel 1 goto :check_model
echo [3/4] Starting Ollama background service...
start "OllamaBg" /BELOWNORMAL /B ollama serve >nul 2>nul
timeout /t 3 /nobreak >nul

:check_model
ollama list | findstr /I "qwen3:0.6b" >nul 2>nul
if not errorlevel 1 goto :launch_luna
echo [3/4] Downloading local model qwen3:0.6b - about 522MB...
ollama pull qwen3:0.6b
goto :launch_luna

:ollama_missing
echo [WARNING] Ollama was not found in PATH.
echo           Download from: https://ollama.com/download/windows

:launch_luna
echo [3/4] Checking Python and Ollama environment...
"%PY_EXE%" main.py --check-env

echo.
echo [4/4] Launching LUNA AI Studio on http://127.0.0.1:8000 ...
"%PY_EXE%" main.py --host 127.0.0.1 --port 8000 --open-browser
pause
