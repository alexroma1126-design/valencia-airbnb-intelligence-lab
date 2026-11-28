$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "VALENCIA INTELLIGENCE STUDIO 3.2" -ForegroundColor Cyan
Write-Host "Mat. Hector Roman" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "http://127.0.0.1:8503" -ForegroundColor Green

& .\.venv\Scripts\python.exe -m streamlit run app/executive_app.py --server.address 127.0.0.1 --server.port 8503 --server.headless false
