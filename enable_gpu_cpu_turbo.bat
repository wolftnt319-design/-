@echo off
setlocal EnableExtensions
chcp 65001 >nul
title LUNA AI - Hybrid GPU + CPU Anti-Freeze Optimizer

echo ========================================================
echo     LUNA AI - Hybrid GPU + CPU Anti-Freeze Optimizer
echo ========================================================
echo.

REM 1. Persist Ollama Vulkan + Integrated GPU + Quantized KV Cache for Windows 10
echo [1/4] Enabling Vulkan GPU + Integrated GPU + Low-RAM KV Cache...
setx OLLAMA_VULKAN "1" >nul
setx OLLAMA_IGPU_ENABLE "1" >nul
setx OLLAMA_FLASH_ATTENTION "1" >nul
setx OLLAMA_KV_CACHE_TYPE "q8_0" >nul
setx OLLAMA_MAX_LOADED_MODELS "1" >nul
setx OLLAMA_NUM_PARALLEL "1" >nul

set "OLLAMA_VULKAN=1"
set "OLLAMA_IGPU_ENABLE=1"
set "OLLAMA_FLASH_ATTENTION=1"
set "OLLAMA_KV_CACHE_TYPE=q8_0"
set "OLLAMA_MAX_LOADED_MODELS=1"
set "OLLAMA_NUM_PARALLEL=1"

REM 2. Restart Ollama at BELOW_NORMAL priority so Windows mouse/UI never freezes!
echo [2/4] Restarting Ollama background engine with Anti-Freeze priority...
taskkill /F /IM "ollama app.exe" >nul 2>nul
taskkill /F /IM "ollama.exe" >nul 2>nul
timeout /t 2 /nobreak >nul
start "OllamaBg" /BELOWNORMAL /B ollama serve >nul 2>nul
timeout /t 3 /nobreak >nul

REM 3. Build hardware-balanced `qwen3-hybrid` profile (50%% GPU layers + 2 CPU threads + compact context)
echo [3/4] Creating balanced CPU+GPU model profile (qwen3-hybrid)...
(
echo FROM qwen3:0.6b
echo PARAMETER num_thread 2
echo PARAMETER num_gpu 14
echo PARAMETER num_ctx 1024
echo PARAMETER num_batch 128
) > "%TEMP%\LunaHybrid.Modelfile"
ollama create qwen3-hybrid -f "%TEMP%\LunaHybrid.Modelfile"

echo.
echo [4/4] Checking active Ollama processor mode:
ollama list
echo.
echo Done! Now run run.bat ^(or select qwen3-hybrid in LUNA Model tab^).
pause
