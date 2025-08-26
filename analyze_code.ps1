# Ollama Agents - Code Quality Analysis Script for Windows
# Run this script to perform comprehensive code analysis

param(
    [switch]$Quick,      # Quick analysis (ruff + basic checks only)
    [switch]$Full,       # Full analysis (all tools)
    [switch]$Coverage,   # Include coverage analysis
    [switch]$Help        # Show help
)

if ($Help) {
    Write-Host "Ollama Agents Code Analysis Tool" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\analyze_code.ps1 [-Quick] [-Full] [-Coverage] [-Help]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Quick     Run quick analysis (ruff + pycodestyle)"
    Write-Host "  -Full      Run comprehensive analysis (all tools)"
    Write-Host "  -Coverage  Include test coverage analysis"
    Write-Host "  -Help      Show this help message"
    Write-Host ""
    Write-Host "Default: Quick analysis if no options specified"
    exit 0
}

# Set default to quick if no options specified
if (-not $Quick -and -not $Full -and -not $Coverage) {
    $Quick = $true
}

# Colors for output
$ErrorColor = "Red"
$WarningColor = "Yellow"
$InfoColor = "Cyan"
$SuccessColor = "Green"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor $InfoColor
    Write-Host $Title -ForegroundColor $InfoColor
    Write-Host "=" * 60 -ForegroundColor $InfoColor
}

function Test-Command {
    param([string]$Command)
    try {
        & $Command --version | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Ensure-PipxPath {
    # Ensure pipx tools are in PATH for current session
    $pipxPath = "C:\Users\$env:USERNAME\.local\bin"
    if ($env:PATH -notlike "*$pipxPath*") {
        $env:PATH += ";$pipxPath"
        Write-Host "Added pipx path to current session" -ForegroundColor $InfoColor
    }
}

function Run-Tool {
    param(
        [string]$ToolName,
        [string]$Command,
        [string]$Description
    )
    
    Write-Host ""
    Write-Host "Running $ToolName - $Description" -ForegroundColor $InfoColor
    Write-Host "-" * 40
    
    if (-not (Test-Command $ToolName)) {
        Write-Host "Warning: $ToolName not found. Install with: python -m pipx install $ToolName" -ForegroundColor $WarningColor
        Write-Host "Or ensure C:\Users\$env:USERNAME\.local\bin is in your PATH" -ForegroundColor $WarningColor
        return
    }
    
    try {
        $startTime = Get-Date
        Invoke-Expression $Command
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        Write-Host "Completed in $($duration.ToString('F2')) seconds" -ForegroundColor $SuccessColor
    }
    catch {
        Write-Host "Error running $ToolName : $($_.Exception.Message)" -ForegroundColor $ErrorColor
    }
}

# Check if we're in the right directory
if (-not (Test-Path "main.py") -or -not (Test-Path "src")) {
    Write-Host "Error: This script must be run from the Ollama project root directory" -ForegroundColor $ErrorColor
    Write-Host "Expected files: main.py, src/ directory" -ForegroundColor $ErrorColor
    exit 1
}

# Ensure pipx tools are accessible
Ensure-PipxPath

Write-Host "Ollama Agents - Code Quality Analysis" -ForegroundColor $SuccessColor
Write-Host "Starting analysis at $(Get-Date)" -ForegroundColor $InfoColor

# Quick Analysis
if ($Quick -or $Full) {
    Write-Section "QUICK ANALYSIS"
    
    # Ruff - Modern linting and formatting
    Run-Tool "ruff" "ruff check . --config .ruff.toml" "Modern Python linter and formatter"
    
    # Pycodestyle - PEP 8 style checking
    Run-Tool "pycodestyle" "pycodestyle --config=.pylintrc src/ main.py" "PEP 8 style checking"
    
    # Pyflakes - Simple error checking
    Run-Tool "pyflakes" "pyflakes src/ main.py" "Logical error detection"
}

# Full Analysis
if ($Full) {
    Write-Section "COMPREHENSIVE ANALYSIS"
    
    # Pylint - Comprehensive analysis
    Run-Tool "pylint" "pylint --rcfile=.pylintrc src/ main.py" "Comprehensive code analysis"
    
    # Vulture - Dead code detection
    Run-Tool "vulture" "vulture --config pyproject.toml src/ main.py" "Dead code detection"
    
    # Deptry - Dependency analysis
    Run-Tool "deptry" "deptry --config .deptry.toml ." "Dependency analysis"
    
    # Radon - Complexity analysis
    Write-Section "CODE COMPLEXITY ANALYSIS"
    Run-Tool "radon" "radon cc src/ -a" "Cyclomatic complexity"
    Run-Tool "radon" "radon mi src/" "Maintainability index"
    Run-Tool "radon" "radon hal src/" "Halstead complexity"
    
    # Pydeps - Dependency visualization (text output)
    Run-Tool "pydeps" "pydeps src --no-output --show-deps" "Dependency visualization"
}

# Coverage Analysis
if ($Coverage) {
    Write-Section "TEST COVERAGE ANALYSIS"
    
    if (Test-Command "coverage") {
        Write-Host "Running test coverage analysis..." -ForegroundColor $InfoColor
        
        # Run tests with coverage
        try {
            coverage run -m pytest test/ 2>$null
            coverage report --config-file .coveragerc
            coverage html --config-file .coveragerc
            
            if (Test-Path "htmlcov/index.html") {
                Write-Host "Coverage report generated: htmlcov/index.html" -ForegroundColor $SuccessColor
            }
        }
        catch {
            Write-Host "Coverage analysis failed. Make sure pytest is installed." -ForegroundColor $WarningColor
        }
    } else {
        Write-Host "Coverage tool not found. Install with: python -m pipx install coverage" -ForegroundColor $WarningColor
    }
}

Write-Section "ANALYSIS SUMMARY"

# File statistics
$pythonFiles = Get-ChildItem -Path src/, main.py -Include *.py -Recurse | Measure-Object
$totalLines = 0
Get-ChildItem -Path src/, main.py -Include *.py -Recurse | ForEach-Object {
    $content = Get-Content $_.FullName
    $totalLines += $content.Count
}

Write-Host "Project Statistics:" -ForegroundColor $InfoColor
Write-Host "  Python files: $($pythonFiles.Count)"
Write-Host "  Total lines: $totalLines"
Write-Host "  Analysis completed at: $(Get-Date)"

# Recommendations
Write-Host ""
Write-Host "Recommendations:" -ForegroundColor $InfoColor
Write-Host "  1. Fix any critical issues found by ruff and pylint"
Write-Host "  2. Remove dead code identified by vulture"
Write-Host "  3. Optimize complex functions (cyclomatic complexity > 10)"
Write-Host "  4. Maintain test coverage above 70%"
Write-Host "  5. Run this analysis regularly during development"

Write-Host ""
Write-Host "Analysis complete!" -ForegroundColor $SuccessColor
