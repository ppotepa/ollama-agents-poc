# Generic Ollama Agent Launcher for PowerShell
# Run with: powershell -ExecutionPolicy Bypass -File run_deepcoder.ps1

Write-Host "🤖 Starting Generic Ollama Agent..." -ForegroundColor Cyan

# Try to find and activate virtual environment
$venvPaths = @(".venv", "venv", "ollama-agents")
$activatedVenv = $false

foreach ($venvPath in $venvPaths) {
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Host "🔧 Activating virtual environment ($venvPath)..." -ForegroundColor Yellow
        & $activateScript
        $activatedVenv = $true
        break
    }
}

if (-not $activatedVenv) {
    Write-Host "⚠️  No Windows virtual environment found, using system Python" -ForegroundColor Yellow
}

Write-Host "🚀 Launching Generic Ollama Agent..." -ForegroundColor Green
python main.py

Write-Host "`n⏸️  Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
