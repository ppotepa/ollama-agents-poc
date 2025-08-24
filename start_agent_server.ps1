# Generic Ollama Agent Server PowerShell Launcher
Write-Host "🚀 Starting Generic Ollama Agent Server..." -ForegroundColor Green

# Try to find and activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Activating virtual environment (.venv)..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Activating virtual environment (venv)..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
} elseif (Test-Path "ollama-agents\Scripts\Activate.ps1") {
    Write-Host "🔧 Activating virtual environment (ollama-agents)..." -ForegroundColor Yellow
    & .\ollama-agents\Scripts\Activate.ps1
} else {
    Write-Host "⚠️ No virtual environment found, using system Python" -ForegroundColor Yellow
}

Write-Host "🖥️ Launching Agent Server..." -ForegroundColor Cyan
python src\core\server.py

Read-Host "Press Enter to exit"
