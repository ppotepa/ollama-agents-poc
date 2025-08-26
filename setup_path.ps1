# Setup PATH for Pipx Tools - Run this once to configure your environment
# This script adds the pipx tools directory to your Windows PATH permanently

param(
    [switch]$Check,     # Check if PATH is already configured
    [switch]$Add,       # Add pipx path to user PATH
    [switch]$Help       # Show help
)

$pipxPath = "C:\Users\$env:USERNAME\.local\bin"
$ErrorColor = "Red"
$WarningColor = "Yellow"
$InfoColor = "Cyan"
$SuccessColor = "Green"

if ($Help) {
    Write-Host "Pipx PATH Configuration Tool" -ForegroundColor $SuccessColor
    Write-Host ""
    Write-Host "This script configures your Windows PATH to include pipx installed tools."
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\setup_path.ps1 [-Check] [-Add] [-Help]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Check     Check if pipx path is configured"
    Write-Host "  -Add       Add pipx path to user PATH (permanent)"
    Write-Host "  -Help      Show this help message"
    Write-Host ""
    Write-Host "Default: Check and add if not present"
    exit 0
}

function Test-PipxInPath {
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    $systemPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    $currentPath = $env:PATH
    
    $inUserPath = $userPath -like "*$pipxPath*"
    $inSystemPath = $systemPath -like "*$pipxPath*"
    $inCurrentPath = $currentPath -like "*$pipxPath*"
    
    return @{
        UserPath = $inUserPath
        SystemPath = $inSystemPath
        CurrentSession = $inCurrentPath
        PipxPath = $pipxPath
    }
}

function Add-PipxToPath {
    try {
        $currentUserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
        
        if ($currentUserPath -like "*$pipxPath*") {
            Write-Host "Pipx path already in user PATH: $pipxPath" -ForegroundColor $InfoColor
            return $true
        }
        
        $newUserPath = if ($currentUserPath) { "$currentUserPath;$pipxPath" } else { $pipxPath }
        [Environment]::SetEnvironmentVariable("PATH", $newUserPath, "User")
        
        Write-Host "Successfully added to user PATH: $pipxPath" -ForegroundColor $SuccessColor
        Write-Host "You may need to restart PowerShell/CMD for changes to take effect" -ForegroundColor $WarningColor
        return $true
    }
    catch {
        Write-Host "Error adding to PATH: $($_.Exception.Message)" -ForegroundColor $ErrorColor
        return $false
    }
}

function Test-PipxTools {
    $tools = @("ruff", "pylint", "vulture", "deptry", "radon", "pycodestyle", "pyflakes", "pydeps", "coverage")
    $working = @()
    $missing = @()
    
    foreach ($tool in $tools) {
        try {
            $null = & $tool --version 2>$null
            $working += $tool
        }
        catch {
            $missing += $tool
        }
    }
    
    return @{
        Working = $working
        Missing = $missing
        Total = $tools.Count
        WorkingCount = $working.Count
    }
}

Write-Host "Pipx PATH Configuration Tool" -ForegroundColor $SuccessColor
Write-Host ""

# Check current status
$pathStatus = Test-PipxInPath
$toolStatus = Test-PipxTools

Write-Host "Current PATH Status:" -ForegroundColor $InfoColor
Write-Host "  Pipx Directory: $($pathStatus.PipxPath)"
Write-Host "  In User PATH: $($pathStatus.UserPath)"
Write-Host "  In System PATH: $($pathStatus.SystemPath)"
Write-Host "  In Current Session: $($pathStatus.CurrentSession)"
Write-Host ""

Write-Host "Tool Accessibility:" -ForegroundColor $InfoColor
Write-Host "  Working Tools: $($toolStatus.WorkingCount)/$($toolStatus.Total)"
if ($toolStatus.Working.Count -gt 0) {
    Write-Host "  Available: $($toolStatus.Working -join ', ')" -ForegroundColor $SuccessColor
}
if ($toolStatus.Missing.Count -gt 0) {
    Write-Host "  Missing: $($toolStatus.Missing -join ', ')" -ForegroundColor $WarningColor
}
Write-Host ""

# If just checking, exit here
if ($Check) {
    if ($pathStatus.UserPath -or $pathStatus.SystemPath) {
        Write-Host "✓ Pipx path is configured in your PATH" -ForegroundColor $SuccessColor
    } else {
        Write-Host "✗ Pipx path is NOT in your PATH" -ForegroundColor $WarningColor
        Write-Host "Run with -Add parameter to fix this" -ForegroundColor $InfoColor
    }
    exit 0
}

# Add to PATH if requested or if not present
if ($Add -or (-not $pathStatus.UserPath -and -not $pathStatus.SystemPath)) {
    Write-Host "Adding pipx path to user PATH..." -ForegroundColor $InfoColor
    
    if (Add-PipxToPath) {
        # Update current session PATH
        $env:PATH += ";$pipxPath"
        Write-Host "Updated current session PATH" -ForegroundColor $InfoColor
        
        # Test tools again
        Write-Host ""
        Write-Host "Testing tool accessibility after PATH update..." -ForegroundColor $InfoColor
        $newToolStatus = Test-PipxTools
        
        Write-Host "Results:" -ForegroundColor $InfoColor
        Write-Host "  Working Tools: $($newToolStatus.WorkingCount)/$($newToolStatus.Total)"
        
        if ($newToolStatus.WorkingCount -eq $newToolStatus.Total) {
            Write-Host "✓ All tools are now accessible!" -ForegroundColor $SuccessColor
        } else {
            Write-Host "⚠ Some tools still not accessible" -ForegroundColor $WarningColor
            if ($newToolStatus.Missing.Count -gt 0) {
                Write-Host "  Still missing: $($newToolStatus.Missing -join ', ')" -ForegroundColor $WarningColor
                Write-Host "  Try: python -m pipx install <tool_name>" -ForegroundColor $InfoColor
            }
        }
    }
} else {
    Write-Host "✓ Pipx path is already configured" -ForegroundColor $SuccessColor
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor $InfoColor
Write-Host "1. Restart PowerShell/CMD for permanent PATH changes"
Write-Host "2. Run: .\analyze_code.ps1 -Quick to test the analysis tools"
Write-Host "3. If tools still don't work, reinstall with: python -m pipx install <tool_name>"
