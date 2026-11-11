@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create this project's .venv and install requirements.txt first. See docs/jerry_c_handoff.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app/streamlit_app.py --server.address=127.0.0.1 --server.port=8503 --server.maxUploadSize=32 --browser.gatherUsageStats=false
