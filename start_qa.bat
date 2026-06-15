@echo off
chcp 65001 >nul
title Car Standard QA (RAG Demo)
cd /d "%~dp0rag"

set "PYTHONUTF8=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "GRADIO_SERVER_NAME=127.0.0.1"

rem DeepSeek key: use existing env var, else read rag\deepseek_key.txt
if "%DEEPSEEK_API_KEY%"=="" if exist "deepseek_key.txt" set /p DEEPSEEK_API_KEY=<deepseek_key.txt
if "%DEEPSEEK_API_KEY%"=="" echo [warn] No DeepSeek key found - retrieval only. Paste key in web UI or edit rag\deepseek_key.txt

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [error] Python venv not found: %PY%
  pause & exit /b 1
)

echo ================================================
echo  Starting QA service... model loads in ~10-20s.
echo  Browser opens automatically at http://127.0.0.1:7860
echo  If the page fails, wait a few seconds and refresh.
echo  Close this window to stop the service.
echo ================================================

rem Open browser once the port is live (background poll, up to 40s)
start "" /b powershell -NoProfile -Command "for($i=0;$i -lt 40;$i++){try{if((Invoke-WebRequest 'http://127.0.0.1:7860' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200){Start-Process 'http://127.0.0.1:7860';break}}catch{Start-Sleep 1}}"

"%PY%" app.py
pause
