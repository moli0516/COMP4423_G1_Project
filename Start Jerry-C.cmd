@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create this project's .venv and install requirements.txt first. See docs/jerry_c_handoff.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -B scripts/check_c_setup.py --profile ui --port 8503
if errorlevel 1 (
  echo Setup check failed. Read the diagnostic above. No server was started.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app/streamlit_app.py --server.address=127.0.0.1 --server.port=8503 --server.maxUploadSize=32 --browser.gatherUsageStats=false
