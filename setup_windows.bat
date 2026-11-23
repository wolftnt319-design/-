@echo off
setlocal EnableExtensions
chcp 65001 >nul
title LUNA AI - Windows 10 First-Time Setup

echo ========================================================
echo         LUNA AI - Windows 10 Environment Setup
echo ========================================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :install_pkgs

where python >nul 2>nul
if not errorlevel 1 (
    echo [1/4] Creating isolated virtual environment .venv ...
    python -m venv .venv
    goto :install_pkgs
)

where py >nul 2>nul
if not errorlevel 1 (
    echo [1/4] Creating isolated virtual environment .venv ...
    py -3 -m venv .venv
    goto :install_pkgs
)

echo [ERROR] Python 3.10+ is not installed or not in PATH.
echo Download Python from https://www.python.org/downloads/windows/
pause
exit /b 1

:install_pkgs
echo [2/4] Installing lightweight Python dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check your internet connection.
    pause
    exit /b 1
)

echo [3/4] Installing Playwright Chromium browser...
".venv\Scripts\python.exe" -m playwright install chromium

where ollama >nul 2>nul
if errorlevel 1 goto :no_ollama
echo [4/4] Pulling compact local model qwen3:0.6b - about 522MB...
ollama pull qwen3:0.6b
goto :done

:no_ollama
echo [WARNING] Ollama CLI not found in PATH!
echo Please install Ollama for Windows from https://ollama.com/download/windows
echo and then run: ollama pull qwen3:0.6b

:done
echo.
echo Setup complete! Double-click run.bat to start LUNA AI.
pause
