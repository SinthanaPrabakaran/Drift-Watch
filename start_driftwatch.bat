@echo off
title DriftWatch Turnkey Launcher
echo ================================================================================
echo 🚀 STARTING DRIFTWATCH PLATFORM (Microservice + Streamlit Dashboard)
echo ================================================================================

call .\.venv\Scripts\activate.bat

echo [*] Starting ONNX Model Serving Microservice on http://127.0.0.1:8000...
start /min python 02_model_service.py

timeout /t 2 /nobreak >nul

echo [*] Starting Streamlit Observability Console on http://localhost:8501...
python -m streamlit run 06_dashboard.py

pause
