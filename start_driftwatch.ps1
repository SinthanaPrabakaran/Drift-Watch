# DriftWatch One-Click Turnkey Launcher (start_driftwatch.ps1)
# Starts the ONNX Model Serving Microservice and Streamlit Dashboard simultaneously

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🚀 STARTING DRIFTWATCH: DISTRIBUTED ML OBSERVABILITY ENGINE" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# 1. Activate Virtual Environment
if (Test-Path ".\.venv\Scripts\activate.ps1") {
    Write-Host "[*] Activating virtual environment (.venv)..." -ForegroundColor Green
    & ".\.venv\Scripts\activate.ps1"
} else {
    Write-Host "[!] Virtual environment not found at .\.venv. Proceeding with system python..." -ForegroundColor Yellow
}

# 2. Start Model Serving Service in background
Write-Host "[*] Launching ONNX Model Serving Microservice on http://127.0.0.1:8000..." -ForegroundColor Green
$modelService = Start-Process -FilePath "python" -ArgumentList "02_model_service.py" -PassThru -WindowStyle Minimized

Start-Sleep -Seconds 2

# 3. Launch Streamlit Observability Dashboard
Write-Host "[*] Launching Streamlit Observability Console on http://localhost:8501..." -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "✅ Both services active! Press CTRL+C in this terminal to stop." -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan

try {
    & python -m streamlit run 06_dashboard.py
}
finally {
    if ($modelService -and -not $modelService.HasExited) {
        Write-Host "[*] Stopping Model Serving Microservice (PID: $($modelService.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $modelService.Id -Force -ErrorAction SilentlyContinue
    }
}
