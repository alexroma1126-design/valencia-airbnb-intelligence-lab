param([switch]$Completo)

Set-Location $PSScriptRoot

$python = ".\.venv\Scripts\python.exe"

if ($Completo) {
    $app = "app/app.py"
    $port = 8501
} else {
    $app = "app/predictive_app.py"
    $port = 8502
}

Write-Host "VALENCIA AIRBNB INTELLIGENCE LAB 3.0" -ForegroundColor Cyan
Write-Host "http://127.0.0.1:$port" -ForegroundColor Green

& $python -m streamlit run $app --server.address 127.0.0.1 --server.port $port --server.headless false
